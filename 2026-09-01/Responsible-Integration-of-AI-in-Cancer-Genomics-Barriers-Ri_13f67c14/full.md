# Responsible Integration of AI in Cancer Genomics: Barriers, Risks, and Pathways to Trustworthy Clinical Translation

Bahar <sup>˙</sup>Ilgen<sup>1\*</sup>, Yiannos Tolias<sup>†2</sup>, Denise K¨uhnert<sup>1</sup>, Paraskevi Papadopoulou<sup>3</sup>, Magnus Westerlund<sup>4,</sup> <sup>5</sup>, Dominik Heider<sup>6</sup>, Katharina Ladewig<sup>1,</sup> <sup>7</sup>, Georges Hattab<sup>1,8</sup>

<sup>1\*</sup>Centre for Artificial Intelligence in Public Health Research (ZKI-PH), Robert Koch Institute, Nordufer 20, Berlin, 13353, Germany. <sup>2</sup>Directorate-General for Health and Food Safety, European Commission, Rue de Loi 200, Brussels, B-1049, Belgium. <sup>3</sup>Department of Natural Sciences, School of Science and Technology, Deree-The American College of Greece, Gravias 6, Athens, 15342, Greece.

<sup>4˚</sup>Abo Akademi University, Turku, Finland. <sup>5</sup>Arcada University of Applied Sciences, Helsinki, Finland. <sup>6</sup>Institute of Medical Informatics, University of M¨unster, Albert-Schweitzer-Campus 1/A11, M¨unster, 48149, Germany. <sup>7</sup>Medical University Lausitz - Carl Thiem (MUL-CT), Germany. <sup>8</sup>Department of Mathematics and Computer Science, Freie Universit¨at Berlin, Arnimallee 14, Berlin, 14195, Germany.

\*Corresponding author(s). E-mail(s): IlgenB@rki.de;   
Contributing authors: Yiannos.Tolias@ec.europa.eu; KuehnertD@rki.de; vivipap@acg.edu; Magnus.e.Westerlund@abo.fi;   
Dominik.Heider@uni-muenster.de; LadewigK@rki.de; HattabG@rki.de;

## Abstract

Artificial intelligence (AI) and natural language processing (NLP) are increasingly used to extract, integrate, and interpret biomedical knowledge relevant to cancer genomics, yet their translation into routine clinical oncology has been comparatively slow. The central challenge is not computational capability alone, but trustworthy integration into clinical workflows. This review examines how NLP and AI support the cancer genomics pipeline, from literature mining and automated variant interpretation to clinical trial matching, knowledge graph construction, and multimodal data integration. We identify four interrelated translational failure domains: evidence inconsistency, explainability and uncertainty, data governance and reproducibility, and interoperability. Rather than considering these challenges in isolation, we take a systems-level view, focusing on their interaction across the translational pathway. We propose a conceptual framework and roadmap for addressing these domains through rigorous validation, uncertainty-aware methods, interoperable infrastructures, regulatory alignment, and human oversight across the AI lifecycle. Progress toward routine clinical use will depend less on further improving model capability than on systematically addressing these interacting failure domains from development through deployment and post-deployment monitoring.

Keywords: Natural Language Processing, Artificial Intelligence, Cancer Genomics, Clinical Translation, Trustworthy AI, Interoperability, Data Governance

## 1 Introduction

The principal obstacle to clinical artificial intelligence (AI) in cancer genomics is no longer computational capability alone but trustworthy clinical translation. Advances in cancer genomics, next-generation sequencing (NGS), and AI have dramatically expanded the ability to generate and analyse genomic data [1–5]. However, translating these advances into routine clinical oncology continues to present substantial challenges, requiring AI systems that are not only technically robust but also trustworthy, clinically reliable, and suitable for real-world deployment.

This translational challenge is particularly evident in cancer genomics, which has become one of the most data-intensive areas of modern biomedicine. Largescale sequencing initiatives and routine clinical sequencing generate vast amounts of genomic information [2, 3, 6], while biomedical literature, clinical records, and molecular databases continue to expand rapidly. This creates a fundamental integration challenge: clinically relevant knowledge is distributed across heterogeneous and often unstructured sources that are dificult to interpret, validate, and combine at scale. Conventional approaches increasingly struggle to manage the volume and complexity of modern genomic and biomedical data. Natural language processing (NLP) has therefore become an important component of modern cancer genomics workflows. By extracting and integrating information from biomedical literature, clinical records, genomic databases, and other unstructured sources, these approaches help transform fragmented evidence into clinically relevant knowledge. Applications now span literature mining, automated variant interpretation, clinical decision support, clinical trial matching, knowledge graph construction, biomarker discovery, multimodal data integration, and the extraction of gene–variant–disease relationships [5, 7, 8].

Explainability, rare-variant interpretation, interoperability, data governance, and clinical validation are open problems, and progress on any one does not guarantee progress on the others. The pace of AI development in genomics has outstripped the infrastructure needed to evaluate, communicate, and act on its outputs in real clinical settings. This review builds on the work of the EU Health Policy Platform Thematic Network on ”Natural Language Processing for Cancer Genomics”, which brought together experts in genomics, oncology, NLP, machine learning, computational linguistics, biomedical ethics, and health policy [9]. The collaboration resulted in a Joint Statement, endorsed by European institutions, outlining scientific, technical, and ethical priorities for the responsible use of AI and NLP in oncology.

Building on these observations, this review examines why clinical translation remains challenging despite substantial technological progress. Rather than organizing the field around individual applications or algorithmic advances, we adopt the conceptual framework illustrated in Fig. 1 to organize the discussion around the main barriers to clinical integration: evidence inconsistency, explainability and uncertainty, data governance, and interoperability. Accordingly, this review is organized around three questions:

• How is NLP currently used across the cancer genomics pipeline?

• Which failure domains limit the trustworthy clinical translation of NLP-enabled AI systems into routine oncology practice?

• Which pathways may enable trustworthy and responsible clinical integration?

The following sections address these questions within the proposed translational framework.

## 2 Failure Points for Trustworthy Clinical Integration

## 2.1 From Research Utility to Clinical Integration

The rapid adoption of AI in cancer genomics has been driven by two parallel developments: the expansion of large-scale genomic sequencing and the growing availability of biomedical knowledge that can be computationally integrated using NLP [2, 3, 6]. Together, these developments have increased reliance on NLP to integrate unstructured biomedical knowledge, including scientific literature, clinical narratives, pathology reports, and trial registries, with genomic evidence for clinical interpretation.

Major international initiatives, including The Cancer Genome Atlas (TCGA), the International Cancer Genome Consortium (ICGC), and the ICGC/TCGA Pan-Cancer Analysis of Whole Genomes (PCAWG) consortium, have generated large-scale multimodal clinicogenomic resources that capture genomic, molecular, pathological, and clinical information across diverse tumour types. These eforts have substantially enabled the development and validation of AI methods for cancer research while simultaneously highlighting the increasing molecular complexity and heterogeneity of cancer, thereby intensifying the need for computational methods capable of integrating genomic findings with biomedical literature, clinical records, and other heterogeneous biomedical data and knowledge sources [2, 3, 6, 10, 11].

![](images/1e33e706930ff2cec76ccdfdb6c9f0e8ccd08fea028cf3b28a1487e36be6228a.jpg)  
Fig. 1: Conceptual framework illustrating the translational pathway from AI capability to trustworthy clinical translation in cancer genomics. Advances in genomic sequencing, biomedical knowledge resources, and NLP have expanded AI capabilities across evidence synthesis, variant interpretation, knowledge graph construction, multimodal integration, and clinical decision support. Their routine clinical translation, however, is limited by interconnected failure domains, including evidence inconsistency, explainability and uncertainty, data governance, and interoperability. Overcoming these interacting barriers is essential for achieving reliable, transparent, and clinically integrated AI systems in precision oncology.

As genomic evidence continues to expand across multiple molecular modalities (including somatic mutations, transcriptomics, epigenomics, pathology, and clinical phenotypes) the challenge has progressively shifted from generating genomic data to integrating heterogeneous evidence into clinically actionable knowledge. This transition has positioned NLP and multimodal AI as key components of contemporary clinicogenomic workflows [7, 8, 12].

Over the past decade, NLP has demonstrated substantial utility across a wide range of cancer genomics applications. In literature mining and knowledge extraction, domain-specific language models such as BioBERT [13], SciBERT [14], PubMed-BERT [15], and BioGPT [16] automate the discovery of variant–disease links and biomarkers from millions of oncology publications, with systems such as LitVar [17], PubTator [18], and CIViCmine [19] continuously updating curated cancer knowledge bases from newly published evidence. In clinical genomic reporting, NLP aggregates evidence from biomedical databases, literature, and clinical annotations to support the classification of variants of uncertain significance and to identify actionable mutations for therapeutic decision-making [19–21]. Knowledge graph construction extends this further: NLP-driven entity and relation extraction organises genes, variants, drugs, and phenotypes into structured, queryable networks. Resources such as the LitVar KG, CTKG [22], and the Drug–Gene Interaction Database (DGIdb) [23] illustrate how such graphs support novel gene–disease discovery, drug–target–biomarker mapping, and explainable clinical decision support [24, 25]. In cohort selection and trial matching, NLP extracts eligibility attributes from EHRs, pathology reports, and genomic data to accelerate patient–trial matching and improve representation of rare disease subtypes, integrating live registry feeds from resources such as ClinicalTrials.gov and EUCTR to help maintain up-to-date eligibility assessments [26, 27]. Finally, end-to-end workflows increasingly combine these components, moving from variant call files through NLPsupported evidence aggregation to structured, human-reviewed reports for molecular tumour boards (Fig. 2), while embedding explainability, human-in-the-loop checkpoints, and privacy-preserving deployment throughout the pipeline [7, 28, 29]. By enabling the extraction and synthesis of evidence from rapidly growing biomedical corpora, NLP has become an important component of precision oncology workflows and a valuable bridge between genomic data and clinical knowledge.

Despite this broad research utility, many systems demonstrate promising performance in experimental and retrospective settings, yet few achieve sustained clinical deployment. This gap reflects more than limitations in accuracy or eficiency: clinical implementation requires systems to operate reliably within heterogeneous data environments, evolving evidence bases, regulatory constraints, and high-stakes decision-making. Reliability, interpretability, reproducibility, interoperability, governance, and clinician trust are as important as predictive performance, and failures in any of these dimensions may undermine clinical integration.

The key question is thus not only what NLP can do for cancer genomics, but why promising systems often fail to move from research settings into routine care. Addressing this question means moving beyond capability-oriented evaluation to examine the scientific, technical, clinical, and governance conditions required for trustworthy implementation. The following sections outline these translational failure domains and discuss how they can be addressed to support responsible AI integration in precision oncology.

![](images/ca4f03899fe294ef1264cb8911d9e434fb976c01c125dd7c289eb9f34290efac.jpg)  
Fig. 2: Overview of the NLP-enabled clinical cancer genomics pipeline. NLP integrates multimodal biomedical evidence (including genomic sequencing, biomedical literature, clinical records, pathology, and molecular databases) to support evidence synthesis, clinical variant interpretation, structured genomic reporting, and multidisciplinary molecular tumour board decision-making. Trustworthy clinical translation requires explainability, human oversight, interoperability, data governance, regulatory compliance, and continuous validation throughout the pipeline.

## 2.2 Translational Failure Domains

In cancer genomics, the trustworthiness of NLP-enabled AI depends on interacting scientific, technical, clinical, and governance-related factors. Accordingly, we organise the remainder of this review around four interrelated translational failure domains:

• Evidence inconsistency and clinical reliability: The quality, consistency, and clinical interpretability of the underlying genomic evidence base.

• Explainability and uncertainty: Enabling clinicians to understand, verify, and appropriately interpret model outputs.

• Data governance and reproducibility: The data standards, provenance, and infrastructure required for reliable and reproducible clinical AI deployment.

• Interoperability and multimodal fragmentation: The technical and procedural compatibility required to integrate heterogeneous data across institutions and modalities.

These translational failure domains are closely interconnected rather than independent. Weaknesses in one domain may propagate across subsequent stages of the translational pathway, limiting safe and efective clinical deployment. The following sections examine each domain before outlining practical pathways toward responsible integration.

## 3 Evidence Inconsistency and Clinical Reliability

## 3.1 Variant Interpretation Challenges

Accurate variant interpretation is a central challenge in precision oncology. NLPenabled AI can support the identification of clinically actionable variants by integrating genomic findings with evidence from biomedical literature, clinical records, and curated knowledge bases [28]. However, reliable interpretation depends on more than computational performance.

NLP systems can also support the assessment and prioritisation of variants of uncertain significance (VUS) by integrating case reports, functional studies, and molecular context. Resources such as CIViC [19], PubMiner [20], and variant2literature [21] illustrate scalable variant–literature integration, while transformer-based models [13, 15] support context-aware evidence retrieval and biomedical relation extraction. Their reliability, however, is constrained by the underlying evidence: diferences in database curation, conflicting findings, incomplete functional validation, and variable evidence quality can produce inconsistent variant classifications across knowledge sources.

Improved NLP models are unlikely to resolve uncertainty or conflict already present in the biomedical evidence; they may instead retrieve and synthesise it. Trustworthy variant interpretation also requires evidence generation, curation, validation, and maintenance of biomedical knowledge resources.

## 3.2 Rare Variants and Evidence Scarcity

Rare variants expose an important evidence gap in cancer genomics. Clinical interpretation is often limited less by algorithms than by the lack of biological validation. NGS frequently detects novel or low-frequency variants with incomplete functional evidence, and databases such as ClinVar, gnomAD, and CIViC still leave important gaps, especially for rare variants and underrepresented populations.

NLP can partially address this challenge by extracting sparse literature evidence, linking related variants, and supporting pathogenicity assessment. Sequence-based approaches can complement this evidence using pretrained protein or DNA foundation models for zero-shot variant-efect prediction [30]. However, model-derived predictions cannot substitute for experimental validation when used for clinical decision-making.

Orthogonal evidence from proteomics and epigenomics may further support variant assessment, while emerging computational methods, such as deep-set networks for integrating variant annotations [31] and knowledge-graph-based approaches for evidence integration and prioritisation [32], can improve variant prioritisation and evidence integration. However, these approaches do not eliminate the underlying scarcity of validated biological evidence. Rare variant interpretation thus illustrates a broader limitation of trustworthy clinical AI: computational advances cannot fully compensate for missing biological evidence. Reliable interpretation still requires functional studies, appropriately powered population-scale genomic resources, and systematic clinical validation alongside advances in AI methodology.

## 3.3 Functional Validation Gaps

Functional validation is a principal bottleneck in trustworthy clinical variant interpretation. Even when NLP-enabled AI systems accurately retrieve and synthesise available evidence, reliable clinical interpretation requires experimental confirmation of biological function, which computational methods alone cannot provide.

Literature-mining systems such as LitVar [17], PubTator [18], and CIViCmine [19] retrieve published evidence on variant efects in key cancer genes (e.g., TP53 and BRCA1), while machine-learning models integrate coding and non-coding variants with multi-omics features to predict pathogenicity [33]. Neither retrieval nor prediction, by itself, establishes functional impact. This distinction is codified in clinical practice: under the ACMG/AMP framework, computational predictions receive supporting-level evidence codes (PP3/BP4), while strong functional evidence codes (PS3/BS3) require validated experimental assays [34]. Conflating these tiers risks miscalibrating clinical confidence.

Closing this gap increasingly depends on scalable functional assays alongside improved annotation. Multiplexed assays of variant efect (MAVEs) and saturation genome editing can generate functional readouts for thousands of variants in parallel, while integrating these datasets with other lines of evidence is itself an emerging AI task [35]. However, such assays are unavailable for many genes and variant classes, are gene- and context-specific, and their outputs need careful reconciliation with clinical phenotype data before they can inform variant classification. Heterogeneous evidence quality, inconsistent nomenclature, and continuously evolving biomedical knowledge further compound this challenge, making expert curation indispensable even where high-throughput functional data exist.

Trustworthy clinical translation depends on AI-assisted annotation and evidence synthesis to support prioritisation and curation, functional assays to establish biological efects, and expert interpretation to adjudicate the evidence, with AI complementing rather than substituting for experimental validation.

## 3.4 Implications for Clinical Reliability

Variant interpretation uncertainty, limited evidence for rare variants, and incomplete functional validation constrain the reliability of AI-supported genomic decisionmaking. Conflicting findings and inconsistent annotation across databases can propagate through evidence integration into downstream predictions and clinical recommendations.

In precision oncology, variant misclassification or overestimation of pathogenicity can directly afect therapeutic decisions and clinician confidence in AI-assisted systems. Addressing these risks requires stronger evidence generation, data curation, functional validation, and uncertainty-aware reporting alongside advances in NLP and machine learning. These limitations also afect how uncertainty can be estimated and communicated in clinical AI.

## 4 Explainability and Uncertainty in Genomic Reasoning

## 4.1 Model Interpretability

In cancer genomics, deep learning and large language models increasingly support variant interpretation and therapeutic decision-making by integrating genomic, clinical, and literature-derived evidence. Clinical use requires not only accurate predictions but also explanations that allow recommendations to be traced to their supporting literature, databases, and clinical observations. Argumentation-based frameworks extend this approach by providing structured justifications aligned with clinical reasoning [36].

Explanations are only as reliable as the evidence underlying model predictions. Interpretability should expose and communicate this evidence rather than compensate for weaknesses in the underlying knowledge base. Explainable artificial intelligence (XAI) methods aim to increase the transparency of complex models [37]. Featureattribution methods, including SHAP [38], Integrated Gradients [39], saliency maps, and attention-based approaches, highlight input features associated with individual predictions. Counterfactual and example-based explanations provide complementary perspectives by showing how changes in input characteristics or similarities to previous cases relate to model outputs. In cancer genomics, these techniques have been explored for variant prioritisation, biomarker identification, and molecular subtype classification [40].

Many explanation methods describe correlations learned by models rather than underlying biological mechanisms. Attention weights, for example, do not necessarily reflect causal reasoning [41], while feature-attribution scores may disagree across explanation methods despite identical model predictions [42]. These limitations are particularly relevant for foundation models and large language models, where convincing explanations may coexist with uncertainty or incomplete evidence.

For clinical implementation, interpretability enables rather than guarantees trustworthy AI. Transparent explanations can support clinician understanding, verification of AI-generated recommendations, and regulatory accountability, but cannot compensate for unreliable evidence, poorly calibrated models, or insuficient clinical validation.

What matters is whether explanations allow clinicians to critically evaluate model outputs against available biomedical evidence. Interpretability complements uncertainty communication and human expertise as part of trustworthy clinical reasoning.

## 4.2 Uncertainty in Genomic Evidence

Uncertainty is an inherent characteristic of cancer genomics and represents one of the principal challenges for trustworthy AI-assisted clinical decision-making. Unlike uncertainty arising from model behaviour, genomic uncertainty originates from the underlying evidence itself. Cancer genomes are highly heterogeneous, many variants are insuficiently characterised, and the available biomedical literature frequently contains incomplete, conflicting, or evolving evidence. Consequently, genomic uncertainty is multifactorial, arising from incomplete biological knowledge, probabilistic evidence, biological ambiguity, heterogeneous data resources, and the complexity of integrating genomic and clinical information [34, 43]. AI systems do not reason over definitive biological knowledge but over evidence that is often probabilistic, fragmented, and continuously evolving.

Genomic uncertainty arises from interconnected sources, including variants of uncertain significance (VUS), limited functional validation, inconsistent findings across studies, and population-specific diferences in genomic datasets [34, 44]. Rare clinically relevant variants also contribute to uncertainty because of limited experimental evidence and sparse representation in public databases. Although AI models can integrate scientific literature, genomic databases, and clinical data, they cannot eliminate uncertainty embedded within these sources and may propagate or amplify it when conflicting or incomplete evidence is inadequately represented. The reliability of AI outputs depends not only on model performance but also on the quality and certainty of the underlying evidence.

The increasing use of large language models and automated evidence synthesis further highlights this challenge [45]. Although these systems can integrate heterogeneous information into coherent clinical summaries, the confidence conveyed by their outputs may not reflect the quality or completeness of the underlying evidence [46]. Fluent explanations may create an impression of certainty despite limited or conflicting genomic evidence, potentially leading clinicians to overestimate the reliability of AI-generated recommendations [47]. Distinguishing confidence in model predictions from confidence in the biomedical evidence supporting them is essential for responsible clinical interpretation.

Addressing uncertainty in cancer genomics requires more than improvements in predictive performance. AI systems should explicitly communicate the strength, provenance, and limitations of the evidence supporting their recommendations, enabling clinicians to distinguish well-established genomic knowledge from preliminary or uncertain findings, and to interpret AI-generated recommendations within their appropriate evidential and clinical context.

## 4.3 Model Uncertainty

While the previous section focused on uncertainty in genomic evidence, AI systems introduce additional uncertainty through the modelling process itself. Trustworthy clinical translation requires consideration of both evidence uncertainty, arising from limitations in the underlying biomedical knowledge, and model uncertainty, arising from the behaviour and assumptions of the AI system.

Modern machine-learning models typically operate as probabilistic function approximators, making their predictions sensitive to the training-data distribution, model architecture, parameter estimation, and input variability. High predictive confidence does not necessarily imply correctness: miscalibrated models may assign excessive confidence to erroneous outputs [46, 48], while their uncertainty estimates may not accurately reflect the reliability of individual predictions. In clinical genomics, discrepancies between reported confidence and prediction reliability can afect variant classification, prioritisation, and treatment selection.

These challenges are particularly pronounced in large language models and other foundation models increasingly used for evidence synthesis, clinical summarisation, and genomic interpretation. LLMs may produce statements that are linguistically plausible yet factually incorrect, include unsupported rationales, or convey excessive confidence despite limited or conflicting biomedical evidence [45]. Coherent narratives and well-structured explanations may obscure underlying uncertainty, mask important caveats, and make erroneous recommendations appear credible. Addressing model uncertainty calls for calibration, uncertainty quantification, confidence estimation, and transparent communication of model limitations, alongside the evidence-level safeguards discussed above [49, 50].

## 4.4 Trustworthy Clinical Decision Support

In clinical genomics, AI systems increasingly function as decision-support tools rather than autonomous decision-makers, making their value dependent not only on predictive performance but also on how clinicians interpret and act on their outputs [51]. Clinicians must decide when to accept, question, or override AI-assisted recommendations, often under time pressure and incomplete information. Poorly quantified or communicated uncertainty may lead to over-reliance on uncertain recommendations or under-use of potentially beneficial support. Appropriate reliance requires calibrated uncertainty estimates aligned with clinically meaningful decision thresholds, transparent communication of model limitations, and human–AI collaboration that supports rather than replaces clinical judgement [46].

Communicating uncertainty at the point of care is challenging because even wellquantified uncertainty may be interpreted diferently depending on how it is presented. AI-based clinical decision-support systems communicate uncertainty in diferent ways, from risk scores and categorical labels to visualisations and explanatory text [52, 53]. When uncertainty information is overly complex, inconsistent, or poorly integrated into clinical workflows, clinicians may ignore it and interpret model outputs simply as ”correct” or ”incorrect” [54]. Conversely, overly cautious or alarmist communication may erode trust by making systems appear unreliable or dificult to interpret.

In genomic contexts, uncertainty must be communicated across diferent audiences and tasks, from laboratory specialists reviewing variant evidence to clinicians counselling patients and multidisciplinary tumour boards integrating multiple data sources. This calls for layered representations that combine concise, decision-oriented summaries with access to detailed evidence traces, model limitations, and context-specific caveats. Such designs can support both routine decisions and deeper examination of complex cases, helping clinicians align their reliance on AI recommendations with the strength and scope of the supporting evidence.

## 5 Data Governance and Reproducibility Barriers

## 5.1 Data Standardisation, Governance, and Reproducibility

Besides computational advances, the clinical translation of NLP-enabled AI in cancer genomics depends on the quality, governance, and reproducibility of the underlying data. Despite progress in literature mining, clinical text understanding, and evidence synthesis, these systems are constrained by biomedical data interoperability.

Genomic and clinical information is distributed across clinical reports, pathology narratives, sequencing data, biomedical literature, and electronic health records. These sources use diferent formats, local terminologies, and ontologies, creating substantial heterogeneity. NLP can help harmonise these resources by mapping local terms to standard vocabularies, such as HGVS, SNOMED CT, and ICD-O, and aligning metadata. However, its performance and generalisability still depend on consistent annotation and high-quality reference data. Diferences in reporting practices, patient populations, and annotation protocols can cause dataset shift and reduce model reliability across clinical settings.

Standards and data models such as OMOP, HL7 FHIR, LOINC, UMLS, HGVS, and the FAIR principles provide a basis for harmonising data across institutions [55]. International initiatives, including the Global Alliance for Genomics and Health (GA4GH) [56], Cancer Research Data Commons (CRDC) [57], ICGC Accelerating Research in Genomic Oncology (ICGC-ARGO) [58], and Beyond 1 Million Genomes [59], extend these eforts to large-scale data sharing. Adoption remains uneven across healthcare systems, however, and this unevenness, rather than a lack of standards, continues to limit reproducibility in practice [60].

Governance is a complementary requirement: beyond protecting patient privacy, it establishes mechanisms for data access, provenance, accountability, and quality assurance across the AI lifecycle. The European Health Data Space (EHDS) provides a regulatory framework for the secondary use of electronic health data, including genomic and multi-omics data [61], and supports their use in training, testing, and evaluating AI systems [62]. It also mandates secure data access environments and introduces standardised data quality and utility labels covering completeness, consistency, representativeness, and quality-management procedures [63, 64]. These labels provide a shared basis for assessing and communicating dataset fitness. Alongside the FAIR principles and the AI Act, the EHDS positions governance not as a compliance afterthought but as part of the infrastructure for reliable clinical AI.

## 5.2 Emerging AI Systems and Governability Challenges

Recent advances in foundation models, large language models (LLMs), and multimodal AI have expanded AI capabilities in cancer genomics [65]. Unlike task-specific NLP systems, these models are increasingly being developed to integrate genomic data, biomedical literature, clinical records, pathology reports, imaging, and molecular databases, extending AI applications from information extraction to evidence synthesis, variant interpretation, and hypothesis generation [66].

This expansion introduces governance challenges beyond those of task-specific systems. Foundation-model behaviour depends not only on model architecture but also on evolving training data, retrieval resources, and model updates. A clinically deployed LLM updated without formal revalidation may, for example, drift from the guidelines against which it was originally validated. Reproducibility, traceability, and independent validation become harder to establish. Rapid changes in biomedical evidence add another challenge: a model trained on an earlier evidence base may still recommend treatments that have since been superseded. Trustworthy deployment requires continuous knowledge updating and revalidation rather than one-time validation.

Proprietary deployment further complicates governance. Limited access to training data, model parameters, update histories, and system documentation constrains independent auditing. Even models with strong benchmark performance may leave healthcare institutions unable to verify the basis of a recommendation or determine whether model behaviour has changed since the last evaluation.

These shifts move the focus of governance from evaluating individual predictions to managing the lifecycle of continuously evolving systems [67], including version control, provenance tracking, auditability, and post-deployment monitoring across institutions and populations. The European AI Act’s emphasis on post-market monitoring for highrisk systems reflects this shift [68] and implies that pre-deployment validation alone is insuficient. Sustained reliability depends on ongoing evaluation, record-keeping, and post-deployment oversight throughout the system lifecycle.

## 5.3 Privacy, Bias, and Data Stewardship

The standardisation and governance challenges described above concern how genomic and clinical data are structured and shared; a related challenge is who may access these data and under what safeguards. Cancer genomics involves sensitive information because it links identifiable genetic, clinical, and phenotypic data. NLP systems operating on EHRs, pathology reports, and genomic records depend on robust safeguards, including de-identification of textual and genomic data, diferential privacy techniques, and secure processing environments with strong access controls. Combining multiple data modalities may further increase re-identification risk beyond that posed by individual data sources.

These concerns are addressed by legal frameworks such as the EU GDPR and US HIPAA, which regulate the handling of personal and protected health information within their respective scopes; the GDPR additionally classifies genetic data as a special category subject to stricter safeguards. In practice, data use requires an appropriate lawful basis and adherence to principles such as data minimisation and purpose limitation. Beyond HIPAA’s federal requirements, genetic data protections in the US may vary across states, with some state laws providing additional rights concerning genetic information [69]. Table 1 compares the two frameworks across consent, erasure, breach notification, and individual data rights, highlighting diferences relevant to multi-institutional cancer genomics research involving EU and US sites.

Table 1: Comparison of GDPR and HIPAA provisions relevant to genomic and clinical data governance.
<table><tr><td>Feature</td><td>GDPR (EU)</td><td>HIPAA (US)</td></tr><tr><td>Scope</td><td>Applies broadly to personal data; genetic and health data are classified as special cat- egories subject to additional protections</td><td>Applies to Protected Health Information (PHI) held or transmitted by covered enti- ties and their business asso- ciates</td></tr><tr><td>Consent / Authorisation</td><td>Processing must have a law- ful basis and, for genetic or health data, an applica- ble condition for processing special-category data; consent is one possible basis</td><td>Authorization is generally not required for treatment, pay- ment, and healthcare opera- tions, but is required for many other uses and disclosures</td></tr><tr><td>Erasure</td><td>Individuals may request era- sure under specified condi- tions, subject to exceptions</td><td>No equivalent general right to require erasure of PHI</td></tr><tr><td>Breach Notification</td><td>Supervisory authorities must generally be notified within 72 hours where notification is required; individuals must be informed when the breach is likely to result in high risk</td><td>Affected individuals must generally be notified without unreasonable delay and no later than 60 days; additional notification requirements apply to larger breaches</td></tr><tr><td>Individual Data Rights</td><td>Provides rightsincluding access, rectification, erasure, restriction, and data porta- bility, subject to applicable conditions and exceptions</td><td>Provides rights including access to and amendment of PHI and an accounting of certain disclosures</td></tr></table>

Sources: EU General Data Protection Regulation (GDPR) and the US HIPAA Privacy and Breach Notification Rules.

Questions about who controls genomic data and how these data can be used are not fully resolved. Governance also involves collective interests, including community involvement and indigenous data sovereignty [70]. Restrictions on data reuse may create tensions between data sovereignty and cross-institutional standardisation and reproducibility. A related governance challenge is bias in non-representative training data, which can contribute to healthcare disparities [71, 72]. In precision oncology, this problem is compounded by rare-variant evidence scarcity: populations underrepresented in genomic databases may also have less supporting evidence available for NLP-based systems, reinforcing biases in both the evidence base and model training. Mitigation calls for diverse training data, bias audits, and fairness-aware methods, but algorithmic approaches cannot substitute for representative data. Privacy-preserving approaches such as federated learning may help broaden data representation without centralizing sensitive information.

Trustworthy use of genomic data also depends on patients understanding how their data are used in AI-enabled care. Consent should be explicit, contextual, and revocable, with clear communication of risks, benefits, and the role of AI and NLP in genomic testing. Meaningful consent also requires transparency about the supporting evidence and clear mechanisms for human oversight, audit, and redress.

## 6 Interoperability and Multimodal Fragmentation

NLP supports multimodal cancer genomics by transforming clinical narratives and biomedical literature into structured representations that can be integrated with genomic, imaging, and molecular data [73]. Foundation models and multimodal AI further support the integration of heterogeneous information for disease modelling, biomarker discovery, and variant interpretation [74, 75]. However, multimodal integration is a major barrier to clinical translation [76]. NLP can reduce semantic fragmentation by harmonising unstructured biomedical information, but cannot resolve fragmentation caused by heterogeneous data infrastructures, incompatible acquisition pipelines, incomplete multimodal records, or the lack of agreed frameworks for cross-institutional data exchange.

Discussions of interoperability in genomics tend to collapse these distinct obstacles into a single technical problem. The European Interoperability Framework (EIF) ofers a more discriminating vocabulary [77]. Developed to support cross-border digital public services, and given partial binding force for the public sector by the Interoperable Europe Act [78], the EIF separates interoperability into four layers, namely legal, organisational, semantic, and technical, underpinned by cross-cutting interoperability governance (Table 2). Its target setting is structurally similar to multi-institutional cancer genomics, in which many autonomous organisations, operating under diferent jurisdictions and mandates, attempt to combine data they did not jointly design; and its layers form a dependency chain, read top-down, whose asymmetry is discussed below.

Read through this lens, two claims organise the remainder of this section. The first is that the field’s interoperability efort is unevenly distributed. Cancer genomics has invested heavily in the two lower layers and comparatively little in the two upper ones, yet it is the upper layers that most often determine whether a multi-institutional study proceeds at all. The second, more consequential for trustworthiness, is that interoperability failures are characteristically silent. A missing file or malformed message fails loudly and is therefore fixed; a many-to-one concept collapse, a unit mismatch, a staging-version discrepancy, or a well-formed but incorrect terminology code does not, but passes schema validation and propagates into a confidently calibrated output carrying no indication of the defect upstream. An entire class of interoperability failure is thus undetectable at the point where it does damage, and trustworthiness requires above all that failures be detectable.

Table 2: European Interoperability Framework (EIF) applied to multi-institutional cancer genomics, showing its four interoperability layers and cross-cutting governance, with instruments adapted to the clinical genomics setting [77].
<table><tr><td>Layer</td><td>Core question</td><td>Adapted instruments</td></tr><tr><td>Legal</td><td>Is the exchange lawful and legally valid?</td><td>EHDS data permits and secure processing envi- ronments; MDR/IVDR conformity assessment; AI Act high-risk obligations; GDPR Art. 9 and national research derogations</td></tr><tr><td></td><td>Organisational Who does what, under what commitment?</td><td>Consortium and data-sharing agreements; cura- tion and stewardship roles; multi-centre valida- tion protocols; molecular tumour board proce- dures</td></tr><tr><td>Semantic</td><td>Is the data understood as intended?</td><td>Standard terminologies (HGVS, SNOMED CT); clinical significance criteria (ACMG/AMP); curated knowledge bases (ClinVar, CIViC)</td></tr><tr><td>Technical</td><td>Can systems connect and exchange data?</td><td>Common data models (HL7 FHIR, OMOP); sequence and imaging formats; federated query and retrieval interfaces (GA4GH)</td></tr></table>

Cross-cutting governance: arrangements for agreeing, maintaining, and versioning these instruments across institutions, alongside clinical governance and trustworthy AI assessment.

## 6.1 Semantic Interoperability as Lossy Compression

Most standardisation eforts in cancer genomics address the technical and semantic layers discussed in Section 5. Technical-layer obstacles are real but comparatively tractable, and they fail loudly: a tumour sequenced against one reference genome build and imaged with a pathology scanner using institution-specific colour calibration cannot be pooled with data from another site, and a pipeline encountering such a mismatch typically stops. The semantic layer is where the harder and quieter problems accumulate.

Semantic harmonisation is inherently lossy. Mapping local codes to shared terminologies is often many-to-one and non-invertible, so clinically meaningful distinctions such as laterality, qualifiers, or local subtypes may be discarded to enable cross-site comparability. These mappings trade local fidelity for interoperability, yet measuring the magnitude and subgroup distribution of the resulting information loss is an important unmet need. Models trained on harmonised data cannot recover distinctions that have already been removed from their inputs, nor can they signal that such distinctions ever existed. Standards proliferation compounds the problem: datasets may be represented in FHIR, OMOP, Phenopackets, mCODE, or local models, requiring additional mappings between standards that introduce further loss and are rarely validated end to end.

Meaning is also versioned, and genomic data difer from the administrative setting of the EIF in how revisions afect previously issued records. Diagnostic terminologies, disease coding systems, staging systems, response criteria, and reference genome builds are revised on diferent schedules, while the underlying evidence base evolves independently (Section 3). The same variant call file can carry a diferent clinical meaning a year later without the file itself changing. Mapping between terminology versions cannot recover this diference because the evidence, rather than the vocabulary, has changed. Provenance, source version, and classification date are part of the meaning being transferred, not merely metadata; harmonising terminology without this context can align vocabulary while allowing meaning to drift.

The existence of a standard is not suficient. Tumour mutational burden (TMB), for example, can vary for the same tumour depending on panel composition, gene coverage, and bioinformatics pipeline, making it partly assay-dependent [79]. Radiomic features similarly depend on acquisition and processing, motivating initiatives such as the Image Biomarker Standardisation Initiative [80]. Derived variables may appear equivalent across sites while remaining substantively diferent. The same problem applies to clinical endpoints such as index date, real-world progression-free survival, and time to next treatment, which can be syntactically harmonised without being clinically comparable. Linguistic disparities add another layer: clinical NLP is concentrated in well-resourced languages, leaving harmonisation less reliable for languages with limited NLP resources, a gap that goes unrecorded in the harmonised data itself.

## 6.2 Organisational and Legal Interoperability

The organisational layer concerns who does what and under what commitment: which institution re-curates a shared variant knowledge base, who is accountable when reclassification invalidates an issued clinical report, and who re-runs validation when a partner site changes its pipeline. Prospective multi-centre validation is an organisational commitment before it is a technical exercise, which may help explain why it remains scarce despite broad agreement on its necessity.

Organisational constraints are often mistaken for technical ones. One example is mapping rot: mappings created as project deliverables may become progressively inaccurate as terminologies evolve if they are not maintained, while resulting errors may remain undetected by downstream users. Sustained harmonisation requires recurring operational work, often performed by data-holding institutions with limited dedicated funding or academic credit. Expertise is another constraint, as staf fluent in both clinical oncology and relevant data models are scarce, while negotiated access and permitting may be too slow for models requiring frequent updates. More fundamentally, health data architectures often reflect the organisational structure of the health systems that produced them, limiting what technical harmonisation alone can achieve.

Federated approaches are often presented as a solution to cross-border data sharing [29], and are now being deployed at scale through initiatives such as UNCAN-CONNECT [81]. Yet participation requires trusted research environments, data stewardship, harmonised mappings, and, for imaging or foundation-model workloads, local accelerated compute. These requirements favour large, urban, well-resourced cancer centres whose populations may difer systematically from regional and community settings. Federation thus democratises analysis, not participation: compute asymmetry determines which nodes contribute, reintroducing the representativeness bias described in Section 5.

The legal layer concerns whether cross-border or inter-institutional data exchange is lawful. The main challenge is not GDPR’s classification of genetic data as sensitive [82], but the national implementation of its research derogation, which creates divergent permissions across jurisdictions. Thus, identically formatted data may be subject to diferent access conditions, making aggregate- and discovery-level federation easier than record-level access. The EHDS addresses this layer through health data access bodies and data permits intended to support secondary use on a common basis rather than through case-by-case negotiation [83, 84].

The AI Act makes harmonisation a legal rather than merely scientific requirement. For high-risk systems, training, validation, and test data must be relevant, suficiently representative, and, as far as possible, error-free and complete for their intended purpose [85, 86]. In clinical genomics, meeting and documenting these requirements depends on the semantic harmonisation described above and must be demonstrated in conformity assessment. An unresolved issue is whether mapping and transformation layers fall within the regulated boundary. Because changes to these layers can alter a clinical decision-support tool’s behaviour, they may constitute design changes requiring revalidation under the medical-device framework [87–89]. Yet such pipelines are often maintained as research infrastructure without downstream revalidation, making their regulatory status consequential for the cost of continuous harmonisation.

The dependency between layers is asymmetric: legal failure disables the layers below, while technical excellence cannot compensate for it. A fully harmonised FHIRbased exchange cannot proceed without a lawful basis and permit, whereas permitted institutions can exchange data through less sophisticated technical means if necessary. Investment in lower layers yields returns only once upper-layer constraints are resolved, which may help explain why standards adoption in genomics has outpaced crossinstitutional data flow [60].

## 6.3 When the Harmonisation Layer Is Itself an AI System

Language models increasingly perform semantic harmonisation tasks formerly handled by human terminologists, including local-code mapping, concept extraction from narrative text, and vocabulary reconciliation. These applications of the capabilities surveyed in Section 2 are often the only tractable means of harmonising at scale for many institutions. Yet they introduce a consequence largely overlooked in the responsible-AI literature: non-deterministic, versioned, and often proprietary models enter the clinical evidence chain upstream of the systems to which oversight is actually applied.

Three problems follow. First, model-assisted harmonisation creates reproducibility debt: a dataset cannot be regenerated without the exact model version, and proprietary models subject to silent revision may make reproduction impossible in principle, despite reproducibility being harmonisation’s purpose (Section 5). Second, its characteristic error is well formed. A model asked for a terminology code can return a syntactically valid but incorrect mapping, leaving downstream systems unable to distinguish it from a correct one.

The third is particularly consequential for fairness (Section 5), as these errors are not randomly distributed. A language model maps common, well-documented, English-language oncology concepts more reliably than rare tumour types, paediatric entities, minority-language narratives, and legacy local vocabularies, so harmonisation error rates correlate precisely with the subgroups fairness auditing exists to protect. Because these errors enter the dataset as ground truth, they are upstream of, and invisible to, every fairness audit performed on models trained from it; bias introduced at the harmonisation layer cannot be detected by evaluating the downstream model, which is where essentially all current auditing efort is directed.

This does not argue against model-assisted harmonisation, often the only viable option. It argues for treating harmonisation as a clinical AI component in its own right. Mapping performance should be established against gold-standard benchmarks, with human inter-annotator agreement also reported as a reference. Mappings afecting clinical data should be human-reviewed with the reviewer recorded and each should carry provenance covering model identity, version, configuration, and confidence. Most importantly, the system should be required to abstain rather than guess. An explicit refusal to map converts a silent error into a loud one, making the failure detectable before it propagates downstream.

Structured trustworthy-AI assessment can address these requirements if its scope includes the harmonisation layer. Approaches such as Z-Inspection use interdisciplinary socio-technical scenarios and can be applied post-hoc or alongside development [90]. This distinction matters more than timing: post-hoc assessment of the deployed model is blind to upstream harmonisation errors for the same reason downstream fairness audits are, whereas co-design can examine mapping decisions, abstention behaviour, and provenance as they develop. Neither approach reaches harmonisation automatically; this depends on where the socio-technical scenario boundary is drawn, which current practice almost invariably places around the clinical tool rather than the pipeline supplying it.

## 6.4 Multimodal Foundation Models and the Fragmented Data Ecosystem

Individual modalities are often incomplete or collected at diferent time points: patients with rich genomic data may lack corresponding imaging or pathology records, and vice versa. This asynchrony, rather than any single missing standard, complicates multimodal learning and external validation in practice.

Large multimodal models may appear to bypass the semantic and technical layers of Table 2 by learning directly from heterogeneous inputs. Yet statistically absorbing heterogeneity does not remove it: a visible failure, such as an unmappable code or failed data join, may instead become invisible when a model produces a confident output from inputs whose provenance and comparability were never established. Nor does model capability address the upper two layers: no foundation model provides a lawful basis for combining cohorts or assigns responsibility for revalidating shared resources as their evidence base changes.

Asynchrony should also be distinguished from interoperability because it can persist within a single, well-governed institution in which all four layers are already satisfied. Missingness in multimodal oncology data may reflect clinical trajectory rather than protocol, for example when sequencing occurs late in treatment or imaging only after progression. Models may learn aspects of that trajectory alongside the underlying biology. Addressing this requires prospective, protocol-driven multimodal collection rather than further harmonisation of existing data.

## 6.5 Barrier Mitigation

These barriers compound in multi-institutional settings [91]. Models trained within one institutional data ecosystem may degrade when deployed elsewhere because of diferences in technical and procedural context rather than underlying biology.

A tension the literature tends to treat as resolved follows directly from the silentfailure argument: privacy engineering and trustworthiness verification pull in opposite directions, and federated architectures favour privacy by default. Yet distributions that cannot be seen cannot be inspected for site-specific batch efects, label skew, or confounding, while fairness cannot be verified for attributes that were not collected; federation does not remove these hazards but removes the view of them. A constructive response is to make verification a primary service of federated infrastructure: distributed distribution diagnostics, site-stratified reporting, and per-node performance deltas can restore inspection without moving records and provide the visibility required for trustworthy-AI assessment. Federated external validation may consequently be more valuable than federated training, since evaluating models against locally held data and reporting site-level diferences is cheaper, more immediately deliverable, and more directly addresses generalisation failures than distributed model fitting. More broadly, interoperability requires coordinated legal, organisational, semantic, and technical commitments, supported by governance across all four as developed in Sections 7 and 8; the principal limitation is no longer the availability of capable AI models, but the fragmented clinical data ecosystem required for reliable deployment.

## 7 Clinical Implementation of High-Risk AI

## 7.1 Regulatory Requirements for High-Risk AI

As NLP-enabled AI systems increasingly support genomic interpretation, clinical decision support, and precision oncology, they fall within regulatory frameworks governing high-risk AI. Under the European AI Act, an AI system that is itself a product, or a safety component of a product, regulated under the Medical Device Regulation (MDR) or In Vitro Diagnostic Medical Devices Regulation (IVDR) is classified as high-risk where both conditions of Article 6(1) are met: the AI system is intended to be used as a safety component of such a product, or is itself such a product, and the product is required to undergo third-party conformity assessment under the Union harmonisation legislation listed in Annex I [87–89, 92]. Clinical genomics AI systems meeting these conditions are subject to requirements extending beyond predictive performance to safe deployment, transparency, human oversight, and lifecycle monitoring.

Compliance consequently extends beyond model validation. Providers must implement risk management, technical documentation, data governance, and measures ensuring robustness, accuracy, and cybersecurity [86]. These requirements may also be relevant where general-purpose AI (GPAI) models are integrated into clinical applications, while providers and deployers of the resulting high-risk AI systems are subject to distinct obligations [93, 94].

Reuse creates an additional documentation burden because providers of GPAI models must supply suficient information and documentation to downstream providers integrating those models into AI systems [95]. Although qualifying free and open-source GPAI models are exempted from certain technical-documentation and downstream-information duties [96], this exemption does not extend to GPAI models with systemic risk [97, 98]. Compliance becomes shared, ongoing accountability across the model lifecycle rather than a single pre-market assessment.

These obligations are not merely external constraints on innovation. A clinician who cannot determine why a model flagged a variant, or whether it has been updated since relevant guidance changed, has little basis for trusting its output regardless of benchmark accuracy, precisely the gap lifecycle regulation is intended to address.

## 7.2 Human Oversight and Accountability

Clinical genomics is a domain in which AI should support, not replace, human expertise. Variant interpretation, diagnosis, treatment selection, and patient management often depend on uncertain, incomplete, or evolving evidence requiring clinical judgement beyond computational prediction. AI recommendations should be critically evaluated by healthcare professionals rather than treated as autonomous clinical decisions.

Meaningful oversight requires more than reviewing outputs. Clinicians must be able to understand a recommendation’s rationale, recognise unreliable predictions, and challenge or override AI-assisted decisions. This relies on transparent explanations, communicated uncertainty, evidence provenance, and workflows that enable meaningful clinician–AI interaction and independent clinical verification.

Accountability rests with healthcare professionals and the institutions deploying these systems. Governance should define responsibilities across developers, providers, and clinical users throughout the AI lifecycle, while documentation, audit trails, and traceability enable retrospective assessment of AI-assisted decisions. External audits and clearly defined redress pathways extend accountability beyond internal institutional oversight.

## 7.3 Post-Deployment Monitoring

Clinical validation before deployment is necessary but insuficient for AI systems in cancer genomics, where scientific knowledge, clinical guidelines, genomic databases, and population characteristics evolve over time. Post-deployment monitoring should assess real-world performance, detect dataset shift and degradation, document adverse events, and reassess calibration as new variants, functional evidence, and therapeutic guidance emerge. Version control, documentation, provenance tracking, and periodic revalidation are likewise needed to identify changes in system behaviour and preserve reproducibility, traceability, and regulatory compliance, particularly for foundation models that are updated or draw on external knowledge sources. Trustworthy deployment should thus be treated as continuous governance rather than a one-time regulatory milestone [68, 99], with sustained monitoring and maintenance needed to preserve clinician confidence, patient safety, and long-term reliability. Table 3 summarizes these requirements across the full AI system lifecycle, from development through maintenance.

Table 3: Clinical implementation requirements across the AI system lifecycle.
<table><tr><td>Lifecycle stage</td><td>Core implementation requirement</td></tr><tr><td>01 Development</td><td>Governance-by-design, data quality assurance, risk management, and technical documentation.</td></tr><tr><td>02 Validation</td><td>External and multi-institutional validation, calibration assessment, and transparent reporting of limitations.</td></tr><tr><td>03 Deployment</td><td>Meaningful human oversight, workflow integration, traceability, and clearly defined clinical responsibilities.</td></tr><tr><td>04 Monitoring</td><td>Post-deployment surveillance, real-world performance evaluation, adverse-event reporting, and dataset-shift detection.</td></tr><tr><td>05 Maintenance</td><td>Version control, provenance tracking, periodic revalidation, and controlled updating of models and knowledge resources.</td></tr></table>

## 8 Critical View and Pathways to Trustworthy Clinical Translation

## 8.1 From Failure Domains to Integration Pathways

Rather than examining individual AI applications or implementation challenges in isolation, our critical view frames clinical translation as a systems-level problem arising from interactions among evidence inconsistency, explainability and uncertainty, data governance and reproducibility, and interoperability. Weaknesses across these domains can propagate through the translational pathway into downstream clinical decisions, limiting adoption and trust.

Table 4a maps six representative systems onto the four translational failure domains. It is illustrative rather than exhaustive or comparative: each system serves a specific priority rather than a complete clinical workflow. ClinVar focuses on variant classification; LitVar on literature–variant linkage; PubTator on entity annotation rather than evidence synthesis or conflict resolution; CIViC/CIViCmine on curated clinical interpretation, with scalability constrained by manual curation; MatchMiner on trial matching; and DGIdb on drug–gene interaction aggregation without systematic conflict resolution. Their limitations therefore largely reflect intended scope rather than implementation deficiencies. Yet collectively, these systems tend to cover only one or two domains, and none addresses all four, indicating that translational failure is a structural ecosystem-level gap rather than the failure of a single tool. Table 4b summarizes the major translational failure domains, their clinical consequences, and the corresponding guiding principles for trustworthy clinical translation.

Table 4: Translational failure domains in cancer genomics: system coverage and clinical implications.  
(a) Coverage of representative systems supporting cancer genomics interpretation and clinical translation across the four translational failure domains. Markers reflect each system’s stated scope and design focus as described in the cited literature rather than an empirical benchmarking exercise: ✓ indicates direct coverage, △ indicates partial or indirect coverage, and × indicates that the domain falls outside the system’s stated scope.
<table><tr><td>System</td><td>Evidence Inconsistency</td><td>Explainability &amp; Uncertainty</td><td>Data Gov. &amp; Reproducibility</td><td>Interoperability</td></tr><tr><td>ClinVar [100]</td><td>√</td><td>X</td><td>△</td><td>△</td></tr><tr><td>LitVar LitVar KG [17]</td><td>√</td><td>X</td><td>△</td><td>△</td></tr><tr><td>PubTator [18]</td><td>△</td><td>X</td><td>X</td><td>△</td></tr><tr><td>CIViC / CIViCmine [19]</td><td>√</td><td>△</td><td>△</td><td>△</td></tr><tr><td>MatchMiner [26]</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>DGIdb [23]</td><td>△</td><td>X</td><td>X</td><td>△</td></tr></table>

(b) Major translational failure domains in cancer genomics, their clinical consequences, and corresponding guiding principles for trustworthy clinical translation.

<table><tr><td>Failure Domain</td><td>Primary Challenge</td><td>Clinical Consequence</td><td>Guiding (Box 1)</td><td>Principle</td></tr><tr><td>Evidence inconsistency (Sec. 3)</td><td>Conflicting, incomplete, or insufficiently validated genomic evidence</td><td>Variable variant interpreta- tion and reduced confidence in clinical recommendations</td><td>Prioritize quality over complexity; integrate functional validation</td><td>evidence model</td></tr><tr><td>Explainability &amp; uncertainty (Sec. 4)</td><td>Limited interpretability and poorly communi- cated uncertainty</td><td>Reduced clinician trust and difficult verification of AI- assisted decisions</td><td>Promote and uncertainty-aware AI</td><td>explainable</td></tr><tr><td>Data governance &amp; reproducibility (Sec. 5)</td><td>Fragmented standards, inconsistent annotation, and limited provenance</td><td>Poor reproducibility, con- strained data sharing, and limited external validation</td><td>Promote reproducibil- ity and transparent evaluation; governance</td><td>embed through-</td></tr><tr><td>Interoperability &amp; multimodal fragmentation (Sec. 6)</td><td>Heterogeneous infrastructures incompatible standards</td><td>data and clinical</td><td>Limited cross-institutional generalizability and unreli- able clinical deployment</td><td>Strengthen interop- erability across data ecosystems</td></tr></table>

<table><tr><td>Box 1. Principles for Trustworthy Clinical Translation in Cancer Genomics</td></tr><tr><td>1. Prioritize evidence quality over model complexity Advanced AI models cannot compensate for incomplete, conflicting, or poorly validated genomic evidence. 2. Promote explainable and uncertainty-aware AI Variant classifications and AI-generated recommendations should be accompanied by transparent explanations, evidence provenance, and calibrated uncertainty estimates.</td></tr><tr><td>3. Integrate functional validation into AI-supported interpretation Computational predictions should be complemented by experimental evidence when- ever possible. 4. Promote reproducibility and transparent evaluation</td></tr><tr><td>Models should be evaluated across diverse datasets and institutions with clear reporting of limitations and validation procedures. 5. Strengthen interoperability across data ecosystems Clinical adoption requires seamless integration of genomic, clinical, and biomedical</td></tr><tr><td>knowledge sources. 6. Embed governance throughout the AI lifecycle Privacy, accountability, transparency, and regulatory compliance should be incorpo- rated from development through deployment.</td></tr></table>

The principles in Box 1 are interconnected: evidence quality, explainability, reproducibility, interoperability, governance, and human oversight reinforce one another across the translational pathway. Responsible integration requires a systems-level approach that addresses scientific, technical, clinical, and governance challenges together.

## 8.2 Scaling Trustworthy Clinical Translation

Whereas Box 1 summarizes principles for individual AI systems, implementation also requires structural priorities beyond any single system or institution. Four stand out. First, cross-institutional and multilingual data infrastructures are needed for reproducibility and interoperability. Standards such as OMOP and HL7 FHIR, together with initiatives such as the EHDS, provide a foundation, but their value depends on adoption beyond well-resourced institutions and languages. Second, explainable and regulation-ready AI design should be treated as a starting requirement rather than a retrofit. Interpretability added only after model development may be insufficient for clinical or regulatory scrutiny; systems intended for high-risk clinical use are better served by explainability and audit mechanisms designed in from the outset. Third, closer integration of computational methods with experimental biology is needed to address the functional validation gap. Scaling multiplexed assays of variant efect and other high-throughput functional approaches, and integrating their outputs into AI-assisted interpretation, could reduce reliance on literature-derived evidence alone. Fourth, ethics, governance, and patient engagement must extend beyond compliance to questions of data control, community data sovereignty, and equitable access to AI-enabled precision oncology across low-resource and underserved health systems. Together, these priorities in infrastructure, explainable design, functional validation, and equitable governance provide a roadmap for trustworthy clinical implementation. Progress in one cannot fully compensate for weaknesses in the others: infrastructure without explainable design simply moves opaque systems into clinics faster, while governance without functional validation cannot correct evidence that was never reliable.

## 9 Conclusion

This review set out to examine how NLP and AI are used across the cancer genomics pipeline, which failure domains limit their trustworthy clinical translation, and which pathways may enable responsible clinical integration. The evidence reviewed here suggests that the field has largely resolved the first question: NLP-enabled AI systems now support literature mining, variant interpretation, multimodal integration, knowledge graph construction, and clinical trial matching at a scale unattainable through manual curation alone. It is the second and third questions, why these systems struggle to reach routine clinical practice and how that translational gap can be closed, that will determine the pace and impact of AI adoption over the coming decade. Closing this gap will require close collaboration among computational scientists, clinicians, molecular biologists, regulatory authorities, healthcare institutions, and patients.

Building on the collaborative work of the EU Health Policy Platform Thematic Network on NLP for Cancer Genomics [9], we hope this review provides a common conceptual framework for guiding future research, clinical implementation, and policy development across institutions and healthcare systems. The next generation of AI systems in cancer genomics will not be judged by predictive accuracy alone, but by their ability to generate clinically trustworthy evidence that can be interpreted, validated, governed, and sustained throughout routine oncology practice.

## 10 Data Availability

No new data were generated or analysed in support of this research.

## 11 Author Contributions

B.I. and G.H. conceived the review. B.I. led the manuscript’s conceptual restructuring and drafted the manuscript. G.H., Y.T., D.K., P.P., D.H., M.W., and K.L. contributed domain expertise and critically reviewed the manuscript. All authors reviewed and approved the final version of the manuscript.

## 12 Funding

M.W. acknowledges funding by the European Union’s Horizon Europe programme under grant agreement No. 101215206 (UNCAN-CONNECT).

## References

[1] Mardis, E.R.: The impact of next-generation sequencing technology on genetics. Trends in Genetics 24(3), 133–141 (2008) https://doi.org/10.1016/j.tig.2007.12. 007

[2] The Cancer Genome Atlas Research Network, Weinstein, J.N., Collisson, E.A., Mills, G.B., Shaw, K.R.M., Ozenberger, B.A., Ellrott, K., Shmulevich, I., Sander, C., Stuart, J.M.: The cancer genome atlas pan-cancer analysis project. Nature Genetics 45(10), 1113–1120 (2013) https://doi.org/10.1038/ng.2764

[3] The International Cancer Genome Consortium: International network of cancer genome projects. Nature 464, 993–998 (2010) https://doi.org/10.1038/ nature08987

[4] MacConaill, L.E., Garraway, L.A.: Clinical implications of the cancer genome. Journal of Clinical Oncology 28(35), 5219–5228 (2010) https://doi.org/10.1200/ JCO.2009.27.4944

[5] Xu, J., Yang, P., Xue, S., Sharma, B., Sanchez-Martin, M., Wang, F., Beaty, K.A., Dehan, E., Parikh, B.: Translating cancer genomics into precision medicine with artificial intelligence: applications, challenges and future perspectives. Human Genetics 138(2), 109–124 (2019) https://doi.org/10.1007/ s00439-019-01970-5

[6] Garraway, L.A., Lander, E.S.: Lessons from the cancer genome. Cell 153(1), 17–37 (2013) https://doi.org/10.1016/j.cell.2013.03.002 . Review article

[7] Lipkova, J., Chen, R.J., Chen, B., Lu, M.Y., Barbieri, M., Shao, D., Vaidya, A.J., Chen, C., Zhuang, L., Williamson, D.F.K., Shaban, M., Chen, T.Y., Mahmood, F.: Artificial intelligence for multimodal data integration in oncology. Cancer Cell 40(10), 1095–1110 (2022) https://doi.org/10.1016/j.ccell.2022.09.012

[8] Waqas, A., Tripathi, A., Ramachandran, R.P., Stewart, P.A., Rasool, G.: Multimodal data integration for oncology in the era of deep neural networks: a review. Frontiers in Artificial Intelligence 7, 1408843 (2024) https://doi.org/10.3389/ frai.2024.1408843

[9] Ilgen, B., Hattab, G.: EU Thematic Network Joint Statement: Natural Language Processing for Cancer Genomics. Joint Statement, EU Health Policy Platform. Accessed June 2025 (2024). https://health.ec.europa.eu/document/download/ dd0eeef-f5f8-426b-9d91-164fa917b830 en?filename=policy 20241126 js03 en.

[10] Gerstung, M., Jolly, C., Leshchiner, I., Dentro, S.C., Gonzalez, S., Rosebrock, D., Mitchell, T.J., Rubanova, Y., Anur, P., Yu, K., Tarabichi, M., Deshwar, A., Wintersinger, J., Kleinheinz, K., V´azquez-Garc´ıa, I., Haase, K., Jerman, L., Sengupta, S., Macintyre, G., Malikic, S., Donmez, N., Livitz, D.G., Cmero, M., Demeulemeester, J., PCAWG Evolution & Heterogeneity Working Group, PCAWG Consortium: The evolutionary history of 2,658 cancers. Nature 578, 122–128 (2020) https://doi.org/10.1038/s41586-019-1907-7

[11] Zitnik, M., Nguyen, F., Wang, B., Leskovec, J., Goldenberg, A., Hofman, M.M.: Machine learning for integrating data in biology and medicine: Principles, practice, and opportunities. Information Fusion 50, 71–91 (2019) https: //doi.org/10.1016/j.infus.2018.09.012

[12] Valous, N.A., Popp, F., Z¨ornig, I., J¨ager, D., Charoentong, P.: Graph machine learning for integrated multi-omics analysis. British Journal of Cancer 131, 205– 211 (2024) https://doi.org/10.1038/s41416-024-02706-7

[13] Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C.H., Kang, J.: Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics 36(4), 1234–1240 (2019) https://doi.org/10.1093/ bioinformatics/btz682

[14] Beltagy, I., Lo, K., Cohan, A.: SciBERT: A pretrained language model for scientific text. In: Inui, K., Jiang, J., Ng, V., Wan, X. (eds.) Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3615–3620. Association for Computational Linguistics, Hong Kong, China (2019). https://doi.org/10.18653/v1/D19-1371 . https://aclanthology.org/D19-1371/

[15] Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., Poon, H.: Domain-specific language model pretraining for biomedical natural language processing. ACM Trans. Comput. Healthcare 3(1) (2021) https: //doi.org/10.1145/3458754

[16] Luo, R., Sun, L., Xia, Y., Qin, T., Zhang, S., Poon, H., Liu, T.-Y.: Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in Bioinformatics 23(6), 409 (2022) https://doi.org/10.1093/bib/ bbac409

[17] Allot, A., Peng, Y., Wei, C.-H., Lee, K., Phan, L., Lu, Z.: Litvar: a semantic search engine for linking genomic variant data in pubmed and pmc. Nucleic Acids Research 46(W1), 530–536 (2018) https://doi.org/10.1093/nar/gky355

[18] Wei, C.-H., Allot, A., Lai, P.-T., Leaman, R., Tian, S., Luo, L., Jin, Q., Wang, Z.,

Chen, Q., Lu, Z.: Pubtator 3.0: an ai-powered literature resource for unlocking biomedical knowledge. Nucleic Acids Research 52(W1), 540–546 (2024) https: //doi.org/10.1093/nar/gkae235

[19] Lever, J., Jones, M.R., Danos, A.M., Krysiak, K., Bonakdar, M., Grewal, J.K., Culibrk, L., Grifith, O.L., Grifith, M., Jones, S.J.M.: Text-mining clinically relevant cancer biomarkers for curation into the civic database. Genome Medicine 11, 78 (2019) https://doi.org/10.1186/s13073-019-0686-y

[20] Botsis, T., Murray, J., Leal, A., Palsgrove, D., Wang, W., White, J.R., Velculescu, V.E., Johns Hopkins Molecular Tumor Board Investigators, Anagnostou, V.: Natural language processing approaches for retrieval of clinically relevant genomic information in cancer. Studies in Health Technology and Informatics 295, 350–353 (2022) https://doi.org/10.3233/SHTI220735

[21] Lin, Y.-H., Lu, Y.-C., Chen, T.-F., Hsu, J.S., Lee, K.-H., Cheng, Y.-W., Chen, Y.-C., Fan, J.-S., Tu, C.-T., Hsu, C.-M., Chou, C.-C., Chen, P.-L., Tu, Y.-C.E., Chen, C.-Y.: variant2literature: full text literature search for genetic variants. bioRxiv (2019) https://doi.org/10.1101/583450

[22] Chen, Z., Peng, B., Ioannidis, V.N., Li, M., Karypis, G., Ning, X.: A knowledge graph of clinical trials (ctkg). Scientific Reports 12, 4724 (2022) https://doi. org/10.1038/s41598-022-08454-z

[23] Grifith, M., Grifith, O.L., Cofman, A.C., Weible, J.V., McMichael, J.F., Spies, N.C., Koval, J., Das, I., Callaway, M.B., Eldred, J.M., Miller, C.A., Subramanian, J., Govindan, R., Kumar, R.D., Bose, R., Ding, L., Walker, J.R., Larson, D.E., Dooling, D.J., Smith, S.M., Ley, T.J., Mardis, E.R., Wilson, R.K.: Dgidb: Mining the druggable genome. Nature Methods 10(12), 1209–1210 (2013) https://doi.org/10.1038/nmeth.2689

[24] Zhang, Y., Sui, X., Pan, F., Yu, K., Li, K., Tian, S., Erdengasileng, A., Han, Q., Wang, W., Wang, J., Wang, J., Sun, D., Chung, H., Zhou, J., Zhou, E., Lee, B., Zhang, P., Qiu, X., Zhao, T., Zhang, J.: A comprehensive large-scale biomedical knowledge graph for ai-powered data-driven biomedical research. Nature Machine Intelligence 7, 602–614 (2025) https://doi.org/10. 1038/s42256-025-01014-w

[25] Cui, H., Lu, J., Xu, R., Wang, S., Ma, W., Yu, Y., Yu, S., Kan, X., Ling, C., Zhao, L., Qin, Z.S., Ho, J.C., Fu, T., Ma, J., Huai, M., Wang, F., Yang, C.: A review on knowledge graphs for healthcare: Resources, applications, and promises. Journal of Biomedical Informatics 169, 104861 (2025) https://doi. org/10.1016/j.jbi.2025.104861

[26] Klein, H., Mazor, T., Siegel, E., Trukhanov, P., Ovalle, A., Del Vecchio Fitz, C., Zwiesler, Z., Kumari, P., Van Der Veen, B., Marriott, E., Hansel, J., Yu, J.,

Albayrak, A., Barry, S., Keller, R.B., MacConaill, L.E., Lindeman, N., Johnson, B.E., Rollins, B.J., Do, K.T., Beardslee, B., Shapiro, G., Hector-Barry, S., Methot, J., Sholl, L., Lindsay, J., Hassett, M.J., Cerami, E.: Matchminer: an open-source platform for cancer precision medicine. npj Precision Oncology 6, 69 (2022) https://doi.org/10.1038/s41698-022-00312-5

[27] Mazor, T., Farhat, K.S., Trukhanov, P., Lindsay, J., Galvin, M., Mallaber, E., Paul, M.A., Hassett, M.J., Schrag, D., Cerami, E., Kehl, K.L.: Clinical trial notifications triggered by artificial intelligence–detected cancer progression: A randomized trial. JAMA Network Open 8(4), 252013–252013 (2025) https:// doi.org/10.1001/jamanetworkopen.2025.2013

[28] Tiwari, A., Mishra, S., Kuo, T.-R.: Current ai technologies in cancer diagnostics and treatment. Molecular Cancer 24, 159 (2025) https://doi.org/10.1186/ s12943-025-02369-9

[29] Chowdhury, A., Kassem, H., Padoy, N., Umeton, R., Karargyris, A.: A review of medical federated learning: Applications in oncology and cancer research. In: Crimi, A., Bakas, S. (eds.) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries, pp. 3–24. Springer, Cham (2022). https://doi.org/10. 1007/978-3-031-08999-2 1

[30] Alfisi, I., Ciapi, F., Baragli, M., Magi, A.: Benchmarking dna foundation models for zero-shot variant efect prediction: the role of context, training, and architecture. bioRxiv (2025) https://doi.org/10.1101/2025.06.15.659748 . Preprint

[31] Clarke, B., Holtkamp, E., Ozt¨urk, H., M¨uck, M., Wahlberg, M., Meyer, K.,<sup>¨</sup> Munzlinger, F., Brechtmann, F., H¨olzlwimmer, F.R., Lindner, J., Chen, Z., Gagneur, J., Stegle, O.: Integration of variant annotations using deep set networks boosts rare variant association testing. Nature Genetics 56, 2271–2280 (2024) https://doi.org/10.1038/s41588-024-01919-z

[32] Ryu, Y., Jeong, H.-E., An, J.-Y.: IDAP: an integrated literature- and knowledgegraph-driven evidence prioritization pipeline for precision oncology. Bioinformatics 42(5), 300 (2026) https://doi.org/10.1093/bioinformatics/btag300

[33] Pepe, D., Janssens, X., Timcheva, K., Marr´on-Li˜nares, G.M., Verbelen, B., Konstantakos, V., De Groote, D., De Bie, J., Verhasselt, A., Dewaele, B., Godderis, A., Cools, C., Franco-Tolsau, M., Royaert, J., Verbeeck, J., Kampen, K.R., Subramanian, K., Cabrerizo Granados, D., Menschaert, G., De Keersmaecker, K.: Reannotation of cancer mutations based on expressed rna transcripts reveals functional non-coding mutations in melanoma. The American Journal of Human Genetics 112(6), 1447–1467 (2025) https://doi.org/10.1016/j.ajhg.2025.04.005

[34] Richards, S., Aziz, N., Bale, S., Bick, D., Das, S., Gastier-Foster, J., Grody, W.W., Hegde, M., Lyon, E., Spector, E., Voelkerding, K., Rehm, H.L.: Standards

and guidelines for the interpretation of sequence variants: A joint consensus recommendation of the american college of medical genetics and genomics and the association for molecular pathology. Genetics in Medicine 17(5), 405–424 (2015) https://doi.org/10.1038/gim.2015.30

[35] Findlay, G.M., Daza, R.M., Martin, B., Zhang, M.D., Leith, A.P., Gasperini, M., Janizek, J.D., Huang, X., Starita, L.M., Shendure, J.: Accurate classification of BRCA1 variants with saturation genome editing. Nature 562(7726), 217–222 (2018) https://doi.org/10.1038/s41586-018-0461-z

[36] <sup>˙</sup>Ilgen, B., Dubey, A., Hattab, G.: Phax: A structured argumentation framework for user-centered explainable ai in public health and biomedical sciences. In: Proceedings of the 3rd International Workshop on Argumentation for eXplainable AI (ArgXAI-25). CEUR Workshop Proceedings, vol. 4066. CEUR-WS.org, Bologna, Italy (2025). https://ceur-ws.org/Vol-4066/

[37] Zhang, M., Yin, L.: Explainable artificial intelligence for multi-modal cancer analysis: From genomics to immunology. Critical Reviews in Oncology/Hematology 219, 105040 (2026) https://doi.org/10.1016/j.critrevonc.2025.105040

[38] Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. In: Guyon, I., Luxburg, U., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems, vol. 30, pp. 4765–4784. Curran Associates, Inc., Long Beach, California, USA (2017)

[39] Sundararajan, M., Taly, A., Yan, Q.: Axiomatic attribution for deep networks. In: Precup, D., Teh, Y.W. (eds.) Proceedings of the 34th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 70, pp. 3319–3328. PMLR, Sydney, Australia (2017). https://proceedings.mlr.press/v70/sundararajan17a.html

[40] Gimeno, M., Real, K., Rubio, A.: Precision oncology: a review to assess interpretability in several explainable methods. Briefings in Bioinformatics 24(4), 200 (2023) https://doi.org/10.1093/bib/bbad200

[41] Jain, S., Wallace, B.C.: Attention is not Explanation. In: Burstein, J., Doran, C., Solorio, T. (eds.) Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 3543–3556. Association for Computational Linguistics, Minneapolis, Minnesota (2019). https://doi.org/ 10.18653/v1/N19-1357 . https://aclanthology.org/N19-1357/

[42] Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S., Lakkaraju, H.: The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective (2025). https://arxiv.org/abs/2202.01602

[43] Han, P.K.J., Umstead, K.L., Bernhardt, B.A., Green, R.C., Jofe, S., Koenig,

B., Krantz, I., Waterston, L.B., Biesecker, L.G., Biesecker, B.B.: A taxonomy of medical uncertainties in clinical genome sequencing. Genetics in Medicine 19(8), 918–925 (2017) https://doi.org/10.1038/gim.2016.212

[44] Popejoy, A.B., Fullerton, S.M.: Genomics is failing on diversity. Nature 538(7624), 161–164 (2016) https://doi.org/10.1038/538161a

[45] Singhal, K., Azizi, S., Tu, T., Mahdavi, S.S., Wei, J., Chung, H.W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., Payne, P., Seneviratne, M., Gamble, P., Kelly, C., Babiker, A., Sch¨arli, N., Chowdhery, A., Mansfield, P., Demner-Fushman, D., Arcas, B., Webster, D., Corrado, G.S., Matias, Y., Chou, K., Gottweis, J., Tomasev, N., Liu, Y., Rajkomar, A., Barral, J., Semturs, C., Karthikesalingam, A., Natarajan, V.: Large language models encode clinical knowledge. Nature 620(7972), 172–180 (2023) https://doi.org/10.1038/ s41586-023-06291-2

[46] Omar, M., Agbareia, R., Glicksberg, B.S., Nadkarni, G.N., Klang, E.: Benchmarking the confidence of large language models in answering clinical questions: Cross-sectional evaluation study. JMIR Med Inform 13, 66917 (2025) https: //doi.org/10.2196/66917

[47] Tang, L., Sun, Z., Idnay, B., Nestor, J.G., Soroush, A., Elias, P.A., Xu, Z., Ding, Y., Durrett, G., Rousseau, J.F., Weng, C., Peng, Y.: Evaluating large language models on medical evidence summarization. npj Digital Medicine 6(1), 158 (2023) https://doi.org/10.1038/s41746-023-00927-4

[48] Bajwa, A., Rastogi, R., Kathail, P., Shuai, R.W., Ioannidis, N.: Characterizing uncertainty in predictions of genomic sequence-to-activity models. In: Knowles, D.A., Mostafavi, S. (eds.) Proceedings of the 18th Machine Learning in Computational Biology Meeting. Proceedings of Machine Learning Research, vol. 240, pp. 279–297. PMLR, Seattle, WA, USA (2024). https://proceedings.mlr.press/v240/bajwa24a.html

[49] Dubey, A., Anˇzel, A., <sup>˙</sup>Ilgen, B., Hattab, G.: Ubiqtree: Uncertainty quantification in xai with tree ensembles. Patterns 7(4), 101454 (2026) https://doi.org/10. 1016/j.patter.2025.101454

[50] Seoni, S., Jahmunah, V., Salvi, M., Barua, P.D., Molinari, F., Acharya, U.R.: Application of uncertainty quantification to artificial intelligence in healthcare: A review of last decade (2013–2023). Comput. Biol. Med. 165(C) (2023) https: //doi.org/10.1016/j.compbiomed.2023.107441

[51] Pierce, R.L., Van Biesen, W., Van Cauwenberge, D., Decruyenaere, J., Sterckx, S.: Explainability in medicine in an era of ai-based clinical decision support systems. Frontiers in Genetics 13, 903600 (2022) https://doi.org/10.3389/fgene. 2022.903600

[52] Gray, N., Page, H., Buchan, I., Joyce, D.W.: Risk and uncertainty communication in deployed ai-based clinical decision support systems: A scoping review. ACM Trans. Comput. Healthcare (2026) https://doi.org/10.1145/3830235

[53] Kompa, B., Snoek, J., Beam, A.L.: Second opinion needed: Communicating uncertainty in medical machine learning. NPJ Digital Medicine 4(1), 4 (2021) https://doi.org/10.1038/s41746-020-00367-3

[54] Abbas, Q., Jeong, W., Lee, S.W.: Explainable ai in clinical decision support systems: A meta-analysis of methods, applications, and usability challenges. Healthcare 13(17), 2154 (2025) https://doi.org/10.3390/healthcare13172154

[55] Velde, K.J., Singh, G., Kaliyaperumal, R., Liao, X., Ridder, S., Rebers, S., Kerstens, H.H.D., Andrade, F., Reeuwijk, J., De Gruyter, F.E., Hiltemann, S., Ligtvoet, M., Weiss, M.M., Deutekom, H.W.M., Jansen, A.M.L., Stubbs, A.P., Vissers, L.E.L.M., Laros, J.F.J., Enckevort, E., Stemkens, D., Hoen, P.A.C., Beli¨en, J.A.M., Gijn, M.E., Swertz, M.A.: Fair genomes metadata schema promoting next generation sequencing data reuse in dutch healthcare and research. Scientific Data 9, 169 (2022) https://doi.org/10.1038/s41597-022-01265-x

[56] Rehm, H.L., Page, A.J.H., Smith, L., Goodhand, P., North, K., Birney, E., et al.: Ga4gh: International policies and standards for data sharing across genomic research and healthcare. Cell Genomics 1(2), 100029 (2021) https://doi.org/10. 1016/j.xgen.2021.100029

[57] Brady, A., Charbonneau, A., Grossman, R.L., Creasy, H.H., Renner, R., Pihl, T., Otridge, J., Kim, E., CRDC Program, Barnholtz-Sloan, J.S., Kerlavage, A.R.: NCI cancer research data commons: Core standards and services. Cancer Research 84(9), 1384–1387 (2024) https://doi.org/10.1158/0008-5472. CAN-23-2655

[58] Nahal-Bose, H.K., Lichter, P., Weber, U., Stein, L.D., Bajari, R., Xiang, L., Su, E., Eubank, J., Sch¨utte, C., Yung, C.K., Courtot, M.: The icgc argo data dictionary for standardizing global cancer clinical data. Scientific Data 12, 1852 (2025) https://doi.org/10.1038/s41597-025-06068-4

[59] Saunders, G., Baudis, M., Becker, R., Beltran, S., B´eroud, C., Birney, E., Brooksbank, C., Brunak, S., Bulcke, M., Drysdale, R., Capella-Gutierrez, S., Flicek, P., Florindi, F., Goodhand, P., Gut, I., Heringa, J., Holub, P., Hooyberghs, J., Juty, N., Keane, T.M., Korbel, J.O., Lappalainen, I., Leskosek, B., Matthijs, G., Scollen, S., et al.: Leveraging european infrastructures to access 1 million human genomes by 2022. Nature Reviews Genetics 20, 693–701 (2019) https://doi.org/10.1038/s41576-019-0156-9

[60] Belien, J.A.M., Kip, A.E., Swertz, M.A.: Road to fair genomes: A gap analysis of ngs data generation and sharing in the netherlands. BMJ Open Science 6(1), 100268 (2022) https://doi.org/10.1136/bmjos-2021-100268

[61] European Parliament and Council: European Health Data Space, Article 51: Minimum categories of electronic health data for secondary use (2025). https: //eur-lex.europa.eu/eli/reg/2025/327/oj/eng#art51

[62] European Parliament and Council: European Health Data Space, Article 53: Purposes for which electronic health data can be processed for secondary use (2025). https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng#art53

[63] European Parliament and Council: Regulation (EU) 2025/327 (European Health Data Space), Article 73. Secure processing environment (2025). https://eur-lex. europa.eu/eli/reg/2025/327/oj/eng#art73

[64] European Parliament and Council: European Health Data Space, Article 78: Data quality and utility label (2025). https://eur-lex.europa.eu/eli/reg/2025/ 327/oj/eng#art78

[65] Tsang, K.K., Kivelson, S., Acitores Cortina, J.M., Kuchi, A., Berkowitz, J.S., Liu, H., Srinivasan, A., Friedrich, N.A., Fatapour, Y., Tatonetti, N.P.: Foundation models for translational cancer biology. Annual Review of Biomedical Data Science 8(1), 51–80 (2025) https://doi.org/10.1146/ annurev-biodatasci-103123-095633

[66] Menon, T.P., Mahajan, A., Powell, D.: Foundation model embeddings for multimodal oncology data integration. npj Digital Medicine 9, 131 (2026) https: //doi.org/10.1038/s41746-025-02312-8

[67] El Arab, R.A., Mustafa, M.H., Almagharbeh, W.T., Saleem, N.H., Al Abdulmohsen, S., Boathab, R., Bu Washl, M.: Beyond model development in healthcare ai: Post-development robustness, post-deployment monitoring, and lifecycle governance–a scoping review of reviews. Healthcare 14(11), 1459 (2026) https://doi.org/10.3390/healthcare14111459

[68] European Parliament and Council: Artificial Intelligence Act, Article 72. Postmarket monitoring by providers and post-market monitoring plan for high-risk AI systems (2024). https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng#art72

[69] Gitter, D.M.: Achieving genetic data privacy through enforcement of property rights. UC Davis Law Review 57, 131 (2023)

[70] Carroll, S., Garba, I., Figueroa-Rodr´ıguez, O., Holbrook, J., Lovett, R., Materechera, S., Parsons, M., Raseroka, K., Rodriguez-Lonebear, D., Rowe, R., Sara, R., Walker, J., Anderson, J., Hudson, M.: The care principles for indigenous data governance. Data Science Journal 19, 43 (2020) https://doi.org/10. 5334/dsj-2020-043

[71] Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S.: Dissecting racial bias in an algorithm used to manage the health of populations. Science 366(6464),

[72] Rajkomar, A., Hardt, M., Howell, M.D., Corrado, G., Chin, M.H.: Ensuring fairness in machine learning to advance health equity. Annals of Internal Medicine 169(12), 866–872 (2018) https://doi.org/10.7326/M18-1990

[73] Yang, H., Yang, M., Chen, J., Yao, G., Zou, Q., Jia, L.: Multimodal deep learning approaches for precision oncology: A comprehensive review. Briefings in Bioinformatics 26(1), 699 (2024) https://doi.org/10.1093/bib/bbae699

[74] Truhn, D., Eckardt, J.-N., Ferber, D., Kather, J.N.: Large language models and multimodal foundation models for precision oncology. npj Precision Oncology 8(1), 72 (2024) https://doi.org/10.1038/s41698-024-00573-2

[75] Li, J., Fu, J., Geng, X., Wang, H.: Foundation models in clinical oncology: Progresses and perspectives. Cancer Letters 639, 218220 (2026) https://doi.org/10. 1016/j.canlet.2025.218220

[76] Sweeney, S.M., Hamadeh, H.K., Abrams, N., Adam, S.J., Brenner, S., Connors, D.E., Davis, G.J., Fiore, L., Gawel, S.H., Grossman, R.L., Hanlon, S.E., Hsu, K., Kellof, G.J., Kirsch, I.R., Louv, B., McGraw, D., Meng, F., Milgram, D., Miller, R.S., Morgan, E., Mukundan, L., O’Brien, T., Robbins, P., Rubin, E.H., Rubinstein, W.S., Salmi, L., Schaller, T., Shi, G., Sigman, C.C., Srivastava, S.: Challenges to using big data in cancer. Cancer Research 83(8), 1175–1182 (2023) https://doi.org/10.1158/0008-5472.CAN-22-1274

[77] European Commission: Communication from the Commission to the European Parliament, the Council, the European Economic and Social Committee and the Committee of the Regions: European Interoperability Framework – Implementation Strategy, Brussels (2017). https://eur-lex.europa.eu/legal-content/EN/ TXT/?uri=COM:2017:134:FIN

[78] European Union: Regulation (EU) 2024/903 of the European Parliament and of the Council of 13 March 2024 laying down measures for a high level of public sector interoperability across the Union (Interoperable Europe Act). https:// eur-lex.europa.eu/eli/reg/2024/903/oj. OJ L, 2024/903, 22.3.2024 (2024)

[79] Merino, D.M., McShane, L.M., Fabrizio, D., Funari, V., Chen, S.-J., White, J.R., Wenz, P., Baden, J., Barrett, J.C., Chaudhary, R., et al.: Establishing guidelines to harmonize tumor mutational burden (TMB): in silico assessment of variation in TMB quantification across diagnostic platforms: phase I of the friends of cancer research TMB harmonization project. Journal for ImmunoTherapy of Cancer 8(1), 000147 (2020) https://doi.org/10.1136/jitc-2019-000147

[80] Zwanenburg, A., Valli\`eres, M., Abdalah, M.A., Aerts, H.J.W.L., Andrearczyk, V., Apte, A., Ashrafinia, S., Bakas, S., Beukinga, R.J., Boellaard, R., et

al.: The image biomarker standardization initiative: Standardized quantitative radiomics for high-throughput image-based phenotyping. Radiology 295(2), 328–338 (2020) https://doi.org/10.1148/radiol.2020191145

[81] UNCAN-CONNECT Consortium: UNCAN-CONNECT: Decentralized Collaborative Network for Advancing Cancer Research and Innovation. https://doi.org/ 10.3030/101215206. Horizon Europe, EU Mission on Cancer, grant agreement No. 101215206 (2025–2030). Accessed: 2026-08-05 (2025)

[82] Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data (General Data Protection Regulation). Oficial Journal of the European Union (2016). https://eur-lex. europa.eu/eli/reg/2016/679/oj

[83] European Parliament and Council: European Health Data Space, Article 57: Tasks of health data access bodies (2025). https://eur-lex.europa.eu/eli/reg/ 2025/327/oj/eng#art57

[84] European Parliament and Council: European Health Data Space, Article 68: Data permit (2025). https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng#art68

[85] European Parliament and Council: Artificial Intelligence Act, Article 10. Data and data governance for high-risk AI systems (2024). https://eur-lex.europa.eu/ eli/reg/2024/1689/oj/eng#art10

[86] European Parliament and Council: Artificial Intelligence Act, Articles 8–15. Requirements for high-risk AI systems (2024). https://eur-lex.europa.eu/eli/ reg/2024/1689/oj/eng

[87] European Parliament and Council: Medical Device Regulation, Article 2(1). Definition of ’medical device’ (2017). https://eur-lex.europa.eu/eli/reg/2017/745/ oj#d1e2123-1-1

[88] European Parliament and Council: Medical Device Regulation, Annex VIII, Rule 11. Classification rule for software-based medical devices (2017). https://eur-lex. europa.eu/eli/reg/2017/745/oj#d1e3747-1-1

[89] European Parliament and Council: Regulation (EU) 2017/746 of the European Parliament and of the Council of 5 April 2017 on in vitro diagnostic medical devices and repealing Directive 98/79/EC and Commission Decision 2010/227/EU. In Vitro Diagnostic Medical Devices Regulation (IVDR) (2017). https://eur-lex.europa.eu/eli/reg/2017/746/oj

[90] Zicari, R.V., Brodersen, J., Brusseau, J., D¨udder, B., Eichhorn, T., Ivanov, T., et al.: Z-inspection: A process to assess trustworthy AI. IEEE Transactions on Technology and Society 2(2), 83–97 (2021) https://doi.org/10.1109/TTS.2021.

## 3066209

[91] Conway, J.R., Warner, J.L., Rubinstein, W.S., Miller, R.S.: Next-generation sequencing and the clinical oncology workflow: Data challenges, proposed solutions, and a call to action. JCO Precision Oncology 3, 19–00232 (2019) https: //doi.org/10.1200/PO.19.00232

[92] European Parliament and Council: Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 6(1) and Annex I. Classification rules for high-risk AI systems (2024). https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng#art6

[93] European Parliament and Council: Artificial Intelligence Act, Article 16. Obligations of providers of high-risk AI systems (2024). https://eur-lex.europa.eu/ eli/reg/2024/1689/oj/eng#art16

[94] European Parliament and Council: Artificial Intelligence Act, Article 26. Obligations of deployers of high-risk AI systems (2024). https://eur-lex.europa.eu/ eli/reg/2024/1689/oj/eng#art26

[95] European Parliament and Council: Artificial Intelligence Act, Article 53. Obligations of providers of general-purpose AI models (2024). https://eur-lex.europa. eu/eli/reg/2024/1689/oj/eng#art53

[96] European Parliament and Council: Artificial Intelligence Act, Article 53(2). Exemption from Article 53(1)(a) and (b) for qualifying free and opensource general-purpose AI models; exception does not apply to GPAI models with systemic risk (2024). https://eur-lex.europa.eu/eli/reg/2024/1689/oj/ eng#d1e4112-1-1

[97] European Parliament and Council: Artificial Intelligence Act, Article 51. Classification of general-purpose AI models with systemic risk (2024). https://eur-lex. europa.eu/eli/reg/2024/1689/oj/eng#art51

[98] European Parliament and Council: Artificial Intelligence Act, Article 55. Obligations of providers of general-purpose AI models with systemic risk (2024). https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng#art55

[99] D¨udder, B., M¨oslein, F., St¨urtz, N., Westerlund, M., Zicari, R.V.: Ethical maintenance of artificial intelligence systems. In: Pagani, M., Champion, R. (eds.) Artificial Intelligence for Sustainable Value Creation, pp. 151–171. Edward Elgar Publishing, Cheltenham, UK (2021)

[100] Landrum, M.J., Lee, J.M., Benson, M., Brown, G., Chao, C., Chitipiralla, S., Gu, B., Hart, J., Hofman, D., Hoover, J., Jang, W., Katz, K., Ovetsky, M., Riley, G., Sethi, A., Tully, R., Villamarin-Salomon, R., Rubinstein, W., Maglott, D.R.: Clinvar: public archive of interpretations of clinically relevant variants. Nucleic Acids Research 44(D1), 862–868 (2016) https://doi.org/10.1093/nar/gkv1222