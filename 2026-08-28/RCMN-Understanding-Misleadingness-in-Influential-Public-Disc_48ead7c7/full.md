# RCMN: Understanding Misleadingness in Influential Public Discourse

Peiling Yi

School of Computer Science and Mathematics Faculty of Engineering, Computing and the Environment Kingston University London, United Kingdom p.yi@kingston.ac.uk

## Abstract

Influential public discourse shapes public be liefs and can also mislead, not only through what is stated, but also through how information is framed, omitted, contextualised, and communicated. Yet less research has focused on how such misleadingness arises and shapes the interpretations formed by readers. To ad dress this gap, we introduce Reader-Centric Misleadingness Understanding (RCMN), a framework that operationalises misleadingness through five dimensions: misleading mechanism, likely reader interpretation, evidencewarranted interpretation, emotional arousal, and communicative intent. Based on this framework, we construct an evidence-grounded dataset of influential public discourse. Empirical findings show that misleadingness is diverse and extends well beyond fabrication, with unsupported inference, exaggeration, and omission among the prevalent mechanisms, and is frequently associated with heightened emotional arousal and distortive communica tive intent. Moreover, we investigate whether lightweight claim-and-context representations retain sufficient cues for understanding reader centric misleadingness without access to richer contextual, evidential, and multimodal information. Evaluation across five recent genera tive foundation models shows that reader-level interpretations can often be recovered from such limited representations, whereas identifying how misleadingness is produced remains considerably more challenging. These find ings highlight the potential of lightweight rep resentations for scalable misleadingness analysis, while reliable understanding of misleading mechanisms continues to require richer contextual and evidential grounding.

## 1 Introduction

Influential public discourse refers to publicly circulated communication that has substantial visibility, prominence, or relevance to public debate and can shape how audiences understand issues of societal concern(Druckman, 2001). The statement plays a central role in shaping how people understand political, social, economic, and other public-interest issues(McCombs and Shaw, 1972; Entman et al., 1993; Scheufele and Tewksbury, 2007). Such discourse does more than communicate isolated facts: it can frame which aspects of an issue receive attention, establish causal explanations, attribute responsibility, emphasise particular risks or consequences, and influence how audiences interpret subsequent information(Chong and Druckman, 2010). As these messages are increasingly amplified and recirculated through news platforms and social media, their influence may extend well beyond the original communication, contributing to broader public narratives and potentially affecting attitudes, trust, and decision-making(Vosoughi et al., 2018; Lazer et al., 2018).

Given the societal importance of such discourse, its potential to mislead is particularly consequential (Ecker et al., 2022), posing a persistent challenge to public knowledge and representative democracy. Crucially, grounded information can lead to a distorted understanding when it is selectively presented, framed, or stripped of relevant context. As illustrated in Figure 1, At the factual-verification level, the post reports numerically accurate changes in the Supplemental Poverty Measure (SPM). At the misleadingness-understanding level, the same message encourages readers to interpret these changes as evidence that Trump’s policies reduced poverty whereas Biden’s policies caused it to rise. This interpretation is promoted through omission and selective presentation. The comparison gives insufficient attention to the COVID-19 pandemic, temporary economic-relief programmes, the choice of poverty measure, and other relevant socioeconomic factors . Its competitive political framing also produces moderate emotional arousal by encouraging blame-and-credit attribution and serves a persuasive communicative intent rather than a purely informative one.

![](images/72223677d21210d749f53269955d5c47b9623c0292f47ae0aca11a297a3842ff.jpg)  
Figure 1: From fact verification to misleadingness understanding. All claims, numerical values, and contextual explanations presented in the figure are adapted from the FactCheck.org analysis by (Gore, 2023).

However, computationally understanding misleadingness remains particularly challenging in Influential Online Public Discourse. 1) Beyond Veracity-Centred Modelling. Existing approaches predominantly focus on verifying the factual accuracy of individual claims, while online discourse often derives its persuasive or misleading effect from how claims are framed, combined, contextualised, and presented to audiences(Budak et al., 2024; Pasquetto et al., 2024). 2) Difficult Operationalisation of Reader-Centric Misleadingness. Capturing misleadingness in online discourse requires characterising mechanisms of distortion, likely reader interpretations, emotional arousal, and communicative intent. These dimensions depend on substantial contextual judgement and domain knowledge, making reader-centric characterisation considerably more resource-intensive and complex than conventional claim-level assessment.(Gabriel et al., 2022; Ni et al., 2024; Modzelewski et al., 2026). 3) Expensive Contextual and Multimodal Reasoning. Assessing misleadingness may require recovering rapidly evolving background context and reasoning across text, images, video, or audio to identify omission, emphasis, exaggeration, or recontextualisation. Retrieving and processing such distributed evidence increases computational and latency costs, limiting scalable and real-time analysis (Chakraborty et al., 2023; Xie et al., 2025).

To address these challenges, we propose Reader-Centric Misleadingness Understanding (RCMN), which conceptualises misleadingness in terms of the understanding a message is likely to induce in readers and comprises three complementary components. 1) RCMN Taxonomy. To move beyond veracity-centred modelling, we characterise misleadingness along five complementary dimensions: misleading mechanisms, likely reader interpretations, Evidence-Warranted Interpretation, emotional arousal, and communicative intent. 2) RCMN Dataset for Online Influential Discourse. To operationalise these reader-centric dimensions in real-world discourse, we construct RCMN from publicly circulated messages that attracted professional fact-checking attention and are often associated with public figures, organisations, and significant political or social events. By integrating original source materials, contextual information, and expert fact-checking evidence, the dataset provides structured representations of reader-centric misleadingness for fine-grained analysis and model supervision. 3) RCMN Benchmark. To reduce the cost of contextual and multimodal reasoning at inference time, we benchmark three recent opensource models (Qwen3-VL-8B, DeepSeek-V4- Flash, and Gemma-4-12B) alongside two closedsource models (GPT-5.6 Sol and Claude Fable 5), examining whether they can recover reader-centric misleadingness cues from substantially lower-cost claim-and-context representations without requiring full evidential retrieval or complete multimodal processing.

The empirical findings reveal four main patterns. First, misleadingness in influential public discourse extends well beyond outright fabrication: most cases arise through unsupported inference, exaggeration, omission, miscontextualisation, or misattribution, demonstrating that factual verification alone is insufficient to capture how messages may mislead readers. Second, misleading cases are more frequently associated with high emotional arousal and distortive communicative intent, although arousal alone is not sufficient to indicate misleadingness. Third, in non-misleading cases, the interpretation encouraged by the message generally aligns closely with the evidence-warranted interpretation, supporting interpretive divergence as a useful basis for operationalising misleadingness. Finally, benchmarking five recent large language and multimodal models reveals a clear asymmetry in recoverability: likely reader interpretations and broad affective and communicative cues can often be recovered from lightweight claim-andcontext representations, whereas identifying the precise mechanism of misleadingness remains substantially more difficult and dependent on richer contextual and evidential information, particularly when misleadingness arises from omitted or displaced information.

Our contribution: RCMN reframes misleadingness as a structured reader-centric languageunderstanding problem that captures how misleadingness is produced, what interpretation a message encourages relative to the available evidence, and which affective and communicative signals shape that interpretation. By unifying these dimensions within an evidence-grounded taxonomy, dataset, and benchmark, RCMN provides a systematic framework for studying misleading communication beyond claim-level veracity and for evaluating how effectively five recent generative foundation models can recover reader-centric misleadingness cues from limited contextual information.

## 2 Related work

In this section, we review related work from two perspectives: benchmark datasets and methodological approaches, tracing the progression from factual verification towards reader-centric understanding of misleadingness. Table 1 summarises representative datasets and their expanding task scope.

## 2.1 Datasets and Task Evolution

Early misinformation benchmarks largely operationalised the problem through veracity or factuality prediction. LIAR (Wang, 2017) introduced fine-grained veracity labels, FEVER (Thorne et al., 2018) classified claims as supported, refuted, or lacking sufficient evidence, and MultiFC (Augenstein et al., 2019) extended evidence-based verification to naturally occurring claims collected across multiple fact-checking organisations. FakeNews-Net (Shu et al., 2020) broadened this setting by incorporating news content, social context, and spatiotemporal information. Later resources extended misinformation detection and fact-checking to multimodal settings. Fakeddit (Nakamura et al., 2020) provides large-scale text–image examples for multimodal fake-news detection, while Factify (Mishra et al., 2022), Factify 2(Suryavardan et al., 2023), MuMiN (Nielsen and McConville, 2022), MOCHEG (Yao et al., 2023) and VeriTaS (Rothermel et al., 2026) broaden the task space through multimodal fact verification, multilingual and social-context modelling, evidence use, and explanation generation.

A further line of work addresses out-of-context misinformation, where authentic media become misleading through reuse or mismatched contextualisation. NewsCLIPpings (Luo et al., 2021) and VERITE (Papadopoulos et al., 2024) benchmark the detection of out-of-context or miscaptioned image–text pairs, while COVE (Tonglet et al., 2025) explicitly reconstructs aspects of an image’s original context before assessing caption veracity. These studies importantly demonstrate that misleadingness can arise without media fabrication; These studies importantly demonstrate that misleadingness can arise without media fabrication;

Recent work has begun to move beyond conventional fact verification towards modelling how context, communicative intent, and selective presentation can shape misleading interpretations. M4FC (Geng et al., 2025) further incorporates claimantintent prediction, image contextualisation, location verification, and verdict prediction. MM-Misleading (Li et al., 2026) compares previewsupported and article-supported interpretations to identify omission-based cases in which factually compatible news previews nevertheless induce misleading interpretations.

However, these approaches still capture only part of the broader phenomenon of reader-level misleadingness.

## 2.2 Methods

Reader-centric misleadingness detection shifts the modelling objective beyond factual correctness towards understanding how content, context, communicative intent, emotion, and reader interpretation interact (Yi and Zubiaga, 2026). Existing methods provide several foundations for this direction. Human-centred misinformation research models cognitive, emotional, and behavioural responses to misleading content, including inferred writer intent and potential reader actions (Gabriel et al., 2022), while emotion-aware approaches represent reader perception, emotion categories and intensity, and semantic emotion roles (Oberländer et al., 2020). More recently, interpretation-aware methods compare the understanding induced by limited content with that supported by fuller evidence to detect omission-based interpretation drift (Li et al., 2026). Explainable fact-checking methods further combine multimodal evidence retrieval, veracity prediction, and natural-language explanation generation (Yao et al., 2023).

Table 1: Progression of representative datasets from fact verification to reader-centric misleadingness understanding.
<table><tr><td>Dataset</td><td>Year</td><td>Modality</td><td>Primary task</td><td>Labels or outputs</td></tr><tr><td colspan="5">I. Fact verification and fake-news detection</td></tr><tr><td>LIAR (Wang, 2017)</td><td>2017</td><td>Text</td><td></td><td>Claim-veracity classification Six-level truthfulness labels: pants-fire, false, barely true, half true, mostly true, and true.</td></tr><tr><td>FEVER (Thorne et al., 2018)</td><td>2018</td><td>dence</td><td>tion</td><td>Claims and textual evi- Evidence-based fact verifica- Supported, refuted, and not enough information.</td></tr><tr><td>MultiFC (Augen- stein et al., 2019)</td><td>2019</td><td>Text, evidence, and metadata</td><td>tion</td><td>Multi-domain claim verifica- Heterogeneous veracity ratings collected from multiple fact-checking organi- sations.</td></tr><tr><td>FakeNewsNet (Shu et al., 2020)</td><td>2020</td><td>News, social context, Fake-news detection and metadata</td><td></td><td>Real and fake labels with user-engagement and propagation information.</td></tr><tr><td>Fakeddit (Naka- mura et al., 2020)</td><td>2020</td><td>and comments</td><td>sification</td><td>Text, image, metadata, Multimodal fake-news clas- Binary, three-way, and six-way fine-grained labels.</td></tr><tr><td>Factify / Factify 2 (Mishra et al., 2022; Suryavar-</td><td>2022 23</td><td>idence</td><td></td><td>Claims, images, and ev- Multimodal fact verification Support, refute, insufficient-evidence, and related verification categories.</td></tr><tr><td>dan et al., 2023) MuMiN (Nielsen and McConville,</td><td>2022</td><td>cles, and social graphs detection</td><td></td><td>Claims, images, arti- Multilingual misinformation Claim-veracity labels linked to multilingual social-network data.</td></tr><tr><td>2022) MOCHEG (Yao et al., 2023)</td><td>2023</td><td></td><td>Claims, images, and Multimodal fact-checking</td><td>Veracity labels, multimodal evidence, and natural-language explanations.</td></tr><tr><td>VeriTaS (Rother- 2026 mel et al., 2026)</td><td></td><td>web evidence</td><td>and explanation checking</td><td>Textual and audiovi- Dynamic multimodal fact- Standardised verdict dimensions, retrieved original media, and textual justifi- cations.</td></tr><tr><td colspan="5">sual claims II. Out-of-context and multimodal misalignment detection</td></tr><tr><td>NewsCLIPpings (Luo et al., 2021)</td><td>2021</td><td>Image and caption</td><td>tion detection</td><td>Out-of-context misinforma- Pristine and semantically mismatched image-caption pairs.</td></tr><tr><td>VERITE (Pa- padopoulos et al., 2024)</td><td>2024</td><td>Image, caption, and ex- ternal context</td><td>detection</td><td>Real-world out-of-context Truthful, miscaptioned, and out-of-context image-caption pairs.</td></tr><tr><td>5Pils-OOC (Ton- 2025 glet et al., 2025)</td><td></td><td>Image, caption, and re- Out-of-context trieved evidence</td><td>detection and image contextualisation</td><td>Accurate/out-of-context captions; predicted original image context and cap- tion veracity.</td></tr><tr><td colspan="5">III. Towards misleadingness understanding M4FC (Geng</td></tr><tr><td>et al., 2025) MM-Misleading</td><td>2025</td><td>Images, multilingual claims, and metadata</td><td>checking</td><td>Multitask multimodal fact- Fake image: authentic / manipulated-or-fake; location verification: whether candidate location is consistent; verdict: true / false. News image, headline, Misleading-omission detec- misleading / non-misleading based specifically on misleading omission</td></tr><tr><td>(Li et al., 2026)</td><td>2026</td><td>and article</td><td>tion and correction</td><td></td></tr><tr><td colspan="5">IV. Reader-centric misleadingness understanding</td></tr><tr><td>RCMN (Ours)</td><td>2026 Claims,</td><td>contextual metadata, multimodal sources/links, fact- check metadata/links,</td><td>ness understanding</td><td>Reader-centric misleading- Mechanism: fabrication/alteration, miscontextualisation, omission/selective presentation, misattribution, exaggeration/quantitative distortion, unsup- ported inference; Arousal: low, moderate, high; Intent: informative, persua- sive, distortive; Likely reader interpretation; Evidence-warranted interpreta-</td></tr></table>

However, these methodological strands remain fragmented (Yi and Zubiaga, 2026).

## 3 RCMN taxonomy

In the study, Misleadingness refers to the extent to which the interpretation encouraged by a message diverges from the interpretation warranted by relevant evidence and context, regardless of whether its individual statements are literally true or false (Rogers et al., 2017; Reboul, 2021). Drawing on insights from psychology, linguistics, media theory, and communication studies, (Yi and Zubiaga, 2026) identify three core dimensions that shape misleadingness: emotional arousal, communicative intent, and context. Information may therefore mislead not only through factual inaccuracy, but also by evoking strong emotions, activating moral responses, signalling persuasive or distortive intent, or presenting otherwise accurate information without sufficient context. Building on existing studies, we characterise RCMN along five complementary dimensions.

## 3.1 Misleading Mechanism

This dimension captures the mechanisms or manipulation techniques through which misleading content is produced (van der Linden et al., 2026).

• Fabrication or alteration: Content is invented, synthetically generated, or technically modified in a way that changes its meaning or evidential value.

• Miscontextualisation: Authentic content is presented in an incorrect temporal, spatial, event, or discourse context.

• Omission or selective presentation: Relevant contextual, qualifying, or contradictory information is omitted, or evidence is selectively presented in a way that favours a particular interpretation.

• Misattribution: Content, a quotation, statement, or action is incorrectly attributed to a person, organisation, publication, or account.

• Exaggeration or quantitative distortion: The scale, frequency, certainty, severity, or numerical magnitude of a phenomenon is overstated or otherwise distorted.

• Unsupported inference: The message encourages a causal, evaluative, predictive, or generalised conclusion that is not sufficiently supported by the available evidence and context, even when the underlying statements may be factually accurate.

• Not misleading: No misleading mechanism is identified when the message is evaluated against the relevant evidence and context.

## 3.2 Likely Reader Interpretation & Evidence-Warranted Interpretation

These two dimensions allow us to measure interpretive divergence: the gap between what a message encourages readers to infer and what the available evidence supports. This divergence provides a basis for assessing the degree of misleadingness. Likely Reader Interpretation captures the broader inference or conclusion that a message encourages readers to draw, including interpretations shaped by framing, selective presentation, implication, or omitted context (Gabriel et al., 2022). Evidence-Warranted Interpretation captures the interpretation justified by the fact-check analysis and relevant contextual evidence (Atanasova et al., 2020).

## 3.3 Emotional Arousal

This dimension captures the intensity of the emotional response that the content is likely to evoke (Lühring et al., 2024).

• Low: calm, neutral, or minimally emotional presentation.

• Moderate: noticeable but restrained emotional emphasis.

• High: strongly alerting, urgent, alarming, or emotionally charged presentation.

## 3.4 Communicative Intent

This dimension captures the communicative goal that the message appears to serve (Da et al., 2021).

• Informative communication: Primarily aims to provide, report, describe, or explain information to the reader.

• Persuasive communication: Primarily aims to influence readers’ beliefs, evaluations, or attitudes toward a person, event, issue, or position.

• Distortive communication: primarily steer readers toward an interpretation that is not adequately warranted by the available evidence, through selective, exaggerated, decontextualised, fabricated, or otherwise misleading presentation.

## 4 RCMN Datasets

Figure 2 illustrates the structure of the readercentric dataset through a representative instance. The dataset is organised into five complementary groups, each serving a distinct purpose. 1) Recovered original source content records the multimodal content presented to readers, preserving the observable textual and visual information from which an interpretation may be formed. 2) Evidence & provenance records the source, provenance, and reference evidence needed to establish what is supported and to assess whether the presented content is misleading. 3) Recovered context and evidence captures relevant contextual information that is absent or not explicit in the presented content but may materially affect its interpretation. 4) Reader-centric annotations provide evidencegrounded labels across the five RCMN dimensions, capturing how misleadingness is produced, interpreted, and communicated. 5) Reader engagement signals provide complementary evidence of how audiences actually respond to and interact with the content.

## 4.1 Data Source

The RCMN dataset is seeded from Fact Check Insights (Fact-Check Insights, 2026), which aggregates claims drawn from real-world information environments and investigated by independent factchecking organisations. Rather than treating these records simply as fact-checking examples, we use them as an entry point for identifying and reconstructing potentially misleading public communication. This source is particularly suitable for our study for three reasons: 1) Fact-checkers tend to investigate claims that have already circulated publicly and attracted social or public attention, providing a useful proxy for potentially influential online discourse in which misleading communication may shape reader understanding. 2) Its structured metadata and provenance information facilitate tracing claims back to their original communications and recovering the contextual and evidential information needed to analyse how misleadingness is produced and what interpretations it may encourage. 3) Its coverage across multiple fact-checking organisations exposes RCMN to diverse topics, sources, communicative settings, and forms of misleadingness, supporting the construction of a broad readercentric taxonomy beyond binary factual-veracity judgements.

![](images/91cea782cbd74317e00a505a1c34df4c29f75e8696b56cdacf73b14fcf85f9dd.jpg)  
Figure 2: An example instance from the RCMN dataset

However, Fact-Check Insights exhibit substantial heterogeneity, multilinguality, and sparsity across the full schema. The records span the period from 1970 to 2025 and comprise 260,863 entries across 57 fields, There is no record contain claim source and complete information for all fields. The dataset also demonstrates considerable linguistic and organisational diversity: English accounts for only 7% of the records, while Filipino is the most represented language. Furthermore, the data include contributions from 1,007 fact-checking organisations and contain 24,869 distinct claim-rating labels, reflecting significant variation in annotation conventions across organisations and languages.

Therefore, we consider Fact-Check Insights only as candidate identification and evidence recovery; its fact-check ratings are not treated as the target labels of RCMN.

## 4.2 Data construct

Fact-Check Insights provides only a limited set of structured fields for constructing RCMN, but each record links to a corresponding fact-check article that often preserves or references additional information about the original communication, its context, and the evidence used in the verification process. The fact-check article is therefore used as a recovery gateway to reconstruct the source message and retrieve the contextual and evidential information required for RCMN analysis and annotation.

Table 2: Selected core claim fields from Fact-Check Insights. <sup>∗</sup> denotes fields under itemReviewed, and <sup>#</sup> denotes fields under reviewRating.
<table><tr><td>Field name</td><td>Purpose</td></tr><tr><td>id</td><td>Unique identifier for each claim review record.</td></tr><tr><td>claimReviewed</td><td>Original textual claim being fact- checked.</td></tr><tr><td>datePublished</td><td>Publication date of the fact- checking article.</td></tr><tr><td>url author.name</td><td>URL of the fact-checking page. Fact-checking organisation or au-</td></tr><tr><td></td><td>thor.</td></tr><tr><td>*.author.name *.author.@type</td><td>Original claim maker. Type of claim maker, such as per-</td></tr><tr><td></td><td>son or organisation.</td></tr><tr><td>*.name *.datePublished</td><td>Title of the reviewed item. Publication date of the original</td></tr><tr><td>#.alternateName</td><td>claim. Veracity label assigned by the</td></tr><tr><td>#.ratingExplanation</td><td>fact-checker. Explanation or rationale for the rating.</td></tr></table>

For each Fact-Check Insights record, eleven base fields are retained, as shown in Table 2, to support record linkage and source recovery. The corresponding fact-check article is then inspected, and its outbound links are followed to recover the original post, article, image, video, advertisement, or archived copy. When the original source is unavailable, the message is reconstructed from screenshots, quotations, embedded media, or descriptions preserved in the fact-check article, with recovery level, content representation, provenance, and confidence recorded accordingly.

## 4.3 Annotation

The annotation pipeline is guided by four principles: evidence grounding, separation of factual veracity from misleadingness, multi-level annotation, and auditability. Accordingly, each annotation and explanation must be supported by explicit evidence and recorded in a form that enables subsequent human verification and adjudication. Importantly, annotations are not derived directly from fact-check verdicts such as False, Mostly False, or HalfTrue; The focus is instead on the communicative process through which a message may encourage a misleading interpretation. The pipeline consists of six stages, as illustrated in Figure 3.

S1) Evidence acquisition. The complete factcheck article is used as the primary evidence recovery. Relevant information about the reviewed communication, its surrounding context, omitted or distorted information, and how the message was presented is recovered from the article.

S2) Source reference. The original or recovered source,such as a social-media post, image, video, quotation, or advertisement, is consulted when available. It serves as supplementary reference evidence for understanding the speaker, wording, media, platform, communication setting, and presentation, but its availability is not required for every record.

S3) Evidence recovery. GPT-5.6 Sol, configured with high reasoning effort, was used to extract and structure evidence relevant to reader-centric misleadingness from the fact-check article and available source information. When an article addressed multiple claims, only evidence relevant to the reviewed claim was retained. The recovered evidence was subsequently checked against the source material to remove unsupported, fabricated, or incorrectly attributed information. The recovered evidence was subsequently manually verified against the source material to remove unsupported, fabricated, or incorrectly attributed information.

S4) Initial annotation. Apply the RCMN annotation scheme to each instance based on the recovered evidence and context. The annotation considers the relationship between (i) what the communication explicitly states or shows, (ii) the interpretation encouraged by its wording, framing, and presentation, and (iii) the interpretation warranted by the available evidence and context. The model assigns initial labels across the RCMN dimensions and generates an evidence-warranted interpretation as a standardised reference for assessing divergence between the communicated and evidence-supported interpretations.

S5) Human verification. Human annotators verify each model-proposed annotation against the recovered source content, evidence, and context. They check whether the misleading mechanism is evidence-supported, whether the likely reader interpretation follows from the presented content, whether the evidence-warranted interpretation reflects the fuller evidence, and whether the arousal and communicative-intent labels are justified by observable cues. The recovered evidence is also checked for incorrect attribution, unsupported additions, or missing material context. Any identified discrepancies are revised, while ambiguous or disputed cases are escalated for further review.

S6) Adjudication. Difficult or ambiguous cases are reviewed by a human adjudicator, who resolves disagreements or uncertainty and determines the

final gold annotation.  
![](images/82a382913734c06473e868c91ce38690309be9de02eb925022c347e722ea1c86.jpg)  
Figure 3: RCMN dataset annotation pipeline

Evidential grounding and annotation quality control. Sufficient evidence was available for 2,075 of the 2,216 instances (93.6%), whereas only 101 cases (4.6%) were judged to have insufficient evidence. Moreover, 2,117 instances (95.5%) contain direct two-sided evidence, allowing the annotation to be compared with evidencesupported context rather than inferred from the factcheck verdict alone. Annotation reliability was further strengthened through repeated review: 774 instances (34.9%) received a second check and 323 (14.6%) received a third check.

## 4.4 Empirical Findings

Table 3 summarises the dataset composition, annotation outcomes, and evidence coverage. More importantly, the dataset reveals several distinctive characteristics of misleadingness.

F1: Misleadingness represented in the RCMN dataset is strongly embedded in public-facing online discourse. Politicians and candidates constitute the largest source group (43.9%), while elections and campaigns (17.7%), the economy and employment (15.6%), public health (14.4%), and international affairs and conflict (13.2%) are the most prominent discourse domains.

Table 3: Statistics of the deduplicated dataset covering 2019–2025. Percentages are calculated over all 2,216 instances unless otherwise stated.
<table><tr><td>Category</td><td>Count</td><td>Share</td></tr><tr><td colspan="3">Dataset composition</td></tr><tr><td>Unique instances</td><td>2,216</td><td>100.0%</td></tr><tr><td>Publication period</td><td></td><td>2019-2025</td></tr><tr><td>double check</td><td>774</td><td>34.9%</td></tr><tr><td>Three check</td><td>323</td><td>14.6%</td></tr><tr><td>More</td><td>1</td><td>0.05%</td></tr><tr><td colspan="3">Annotation status</td></tr><tr><td colspan="3">Sufficient evidence</td></tr><tr><td>Article confirms claim / non-misleading</td><td>2,075 63</td><td>93.6% 2.84%</td></tr><tr><td>Insufficient evidence</td><td>101</td><td>4.6%</td></tr><tr><td></td><td></td><td></td></tr><tr><td colspan="3">Source actor type</td></tr><tr><td>Politician / candidate</td><td>973</td><td>43.9%δ</td></tr><tr><td>Political / public organisation</td><td>92</td><td>4.2%§</td></tr><tr><td>Media / journalist / commentator</td><td>57</td><td>2.6%§</td></tr><tr><td>Social-media / anonymous source</td><td>688</td><td>31.0%§</td></tr><tr><td>Other named public / online actor</td><td>406</td><td>18.3%§</td></tr><tr><td colspan="3">Major discourse-domain indicators</td></tr><tr><td>Elections / campaigns</td><td>392</td><td>17.7%</td></tr><tr><td>Economy / employment</td><td>345</td><td>15.6%</td></tr><tr><td>Health / public health</td><td>318</td><td>14.4%</td></tr><tr><td>International affairs / conflict</td><td>292</td><td>13.2%¶</td></tr><tr><td>Government / public policy</td><td>217</td><td>9.8%</td></tr><tr><td>Crime / public safety</td><td>183</td><td>8.3%</td></tr><tr><td>Immigration / border</td><td>146</td><td>6.6%</td></tr><tr><td>Social / cultural issues</td><td>141</td><td>6.4%</td></tr><tr><td>Climate / environment</td><td></td><td>3.3%</td></tr><tr><td></td><td>73</td><td></td></tr><tr><td colspan="3">Fact-checking organisations</td></tr><tr><td>PolitiFact</td><td>1,039</td><td>46.9%</td></tr><tr><td>FactCheck.org</td><td>540</td><td>24.4%</td></tr><tr><td>FactRakers</td><td>276</td><td>12.5%</td></tr><tr><td>Washington Post</td><td>226 135</td><td>10.2%</td></tr><tr><td>Other organisations</td><td></td><td>6.1%</td></tr><tr><td colspan="3">Primary misleading mechanism</td></tr><tr><td>Unsupported inference</td><td>509</td><td>24.8%†</td></tr><tr><td>Exaggeration / quantitative distortion</td><td>468</td><td>22.8%†</td></tr><tr><td>Omission / selective presentation</td><td>361</td><td>17.6%†</td></tr><tr><td>Fabrication / alteration</td><td>330</td><td>16.1%†</td></tr><tr><td>Miscontextualisation</td><td>230</td><td>11.2%†</td></tr><tr><td>Misattribution</td><td>154</td><td>7.5%†</td></tr><tr><td colspan="3">Emotional arousal</td></tr><tr><td colspan="3"></td></tr><tr><td>Low</td><td>305</td><td>13.8%</td></tr><tr><td>Moderate</td><td>634 1,228</td><td>28.6%</td></tr><tr><td>High</td><td>49</td><td>55.4%</td></tr><tr><td>Not assessable</td><td></td><td>2.2%</td></tr><tr><td colspan="3">Communicative intent</td></tr><tr><td colspan="3"></td></tr><tr><td>Informative</td><td>55</td><td>2.5%</td></tr><tr><td>Persuasive</td><td>699</td><td>31.5%</td></tr><tr><td>Distortive</td><td>1,406</td><td>63.4%</td></tr><tr><td>Not assessable</td><td>56</td><td>2.5%</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Evidence coverage</td></tr><tr><td>Direct two-sided evidence</td><td>2,117</td><td>95.5%</td></tr><tr><td>No structured evidence</td><td>99</td><td>4.5%</td></tr><tr><td>Original-source evidence items</td><td>2,130</td><td></td></tr><tr><td>Corrective fact-check evidence items</td><td>2,404</td><td></td></tr></table>

<sup>†</sup> Percentages are calculated over the 2,052 instances with an assigned mechanism. ‡ Percentages are calculated over the 1,098 duplicate groups.

F2: Misleading mechanisms are diversitiy. The diverse distribution of misleading mechanisms shows that determining whether an individual claim is factually true is insufficient for identifying misleadingness. Most cases do not rely on outright fabrication; instead, they mislead through unsupported inference.

F3: Misleadingness is strongly associated with the way information is communicated. More than half of the instances exhibit high emotional arousal, while distortive communicative intent constitutes the largest intent category. Together, these patterns suggest that misleadingness is often encouraged by how information is presented and framed.

F4: Distortive Intent signal misleading All nonmisleading cases, 75% were labelled as persuasive and 23.7% as informative; 0% were assigned distortive intent. By contrast, 73.1% of the misleading cases were labelled as distortive, whereas only 0.3% were informative. These results suggest that distortive, rather than persuasive, intent may serve as a strong indicator of misleadingness.

## F5: Reader-likely interpretation vs evidencewarranted interpretation.

A semantic comparison provides further validation. Among cases labelled as non-misleading, the encouraged interpretation shows strong semantic alignment with the evidence-warranted interpretation in nearly all cases, with only one case exhibiting weaker or partial alignment. In contrast, misleading cases exhibit greater divergence between the encouraged and evidence-warranted interpretations. This pattern supports the misleadingness operationalisation: when a message is labelled as non-misleading, the interpretation it encourages is generally consistent with that warranted by the available evidence.

$$
\mathrm { N o n - m i s l e a d i n g } \ \Rightarrow \ I _ { \mathrm { e n c o u r a g e d } } \approx I _ { \mathrm { w a r r a n t e d } }\tag{1}
$$

F6: Misleading communication is associated with higher emotional arousal. We observe a clear association between misleadingness and emotional arousal. Among messages with an identified misleading mechanism, 58.3% exhibit high arousal, compared with only 12.7% of non-misleading messages. Conversely, 28.6% of non-misleading messages exhibit low arousal, compared with 13.7% of misleading messages. The difference in arousal distributions is statistically significant $( \chi ^ { 2 } ( 2 ) =$ $5 1 . 8 0 , p < 0 . 0 0 1 )$ , indicating that misleading communication is more frequently associated with emotionally intense presentation. However, the association is small (Cramér’s $V = 0 . 1 5 7 )$ , showing that emotional arousal is associated with, but does not, by itself, determine misleadingness.

Table 4: Example of the input used in the RCMN benchmark.  
Input   
Claim/Message: Credit card debt is above \$1 trillion for   
the first time ever.   
Person/Organisation: Jim Justice.   
Location: United States.   
Time: 14 August 2023.   
Event: Release of Q2 2023 New York Federal Reserve   
credit-card balance data.   
Source setting: X social-media post.   
Context: West Virginia Gov. Jim Justice posted the debt   
figure while criticising Biden’s economic policies, con  
necting it to “Bidenomics,” the “radical left,” and families   
relying on credit cards. The post linked to a CNBC story   
about the debt milestone.

## 5 Benchmarks

The RCMN benchmark investigate "To what extent can claim and contextual information preserve sufficient cuesfor understanding multimodal misleadingness without access to the richer contextual, evidential, and multimodal information used to establish reference annotations?"

## 5.1 Task Definition

In the study, we formally formulate:

$$
\begin{array} { r } { X _ { \mathrm { l i m i t e d } } = \{ \mathrm { c l a i m } , \mathrm { p e r s o n } , \mathrm { l o c a t i o n } , \mathrm { t i m e } , \mathrm { e v e n t } ,  } \\ { \mathrm { s o u r c e } \ \mathrm { s e t t i n g } , \mathrm { c o n t e x t } \} , \quad \qquad ( 2 ) } \end{array}
$$

$$
\begin{array} { r l r } & { } & { X _ { \mathrm { f u l l } } = \{ \mathrm { o r i g i n a l s o u r c e , ~ m u l t i m o d a l ~ c o n t e n t , } } \\ & { } & { \mathrm { ~ f a c t - c h e c k ~ a r t i c l e , ~ f a c t - c h e c k ~ r e s u l t s , } } \\ & { } & { \mathrm { ~ e v i d e n c e , ~ s o u r c e ~ s e t t i n g , ~ c o n t e x t } \} } \end{array}\tag{3}
$$

$X _ { \mathrm { f u l l } }$ is used to establish the gold annotation

$$
Y \{ X _ { \mathrm { f u l l } } \} = \{ Y _ { m } , Y _ { r } , Y _ { e } , Y _ { i } \} ,
$$

where $Y _ { m }$ denotes the misleading mechanism, $Y _ { r }$ the likely reader interpretation, $Y _ { e }$ the emotionalarousal label, and $Y _ { i }$ the communicative intent.

At inference time, the evaluated model does not receive the complete evidence used to construct the annotation. Instead, it is provided with a lower-cost representation $X _ { \mathrm { l i m i t e d } }$ , a sample shown in Table 4. The model is therefore required to estimate

$$
\hat { Y } = f ( X _ { \mathrm { l i m i t e d } } )
$$

The objective is for $\hat { Y }$ ≈ $Y$ .

## 5.2 Models & Settings

To support reproducibility and comparison across model families, we evaluate three open-weight models: Qwen3-VL-8B-Instruct (Bai et al., 2025), DeepSeek-V4-Flash(Xu et al., 2026), and Gemma-4-12B(Team et al., 2026), together with GPT-5.6 Sol(OpenAI, 2026) and Claude-fable-5. All models are evaluated under a common zero-shot benchmark protocol using the same 2,216 instances, identical non-verification input fields, task instructions, output schema, and evaluation criteria. Modelspecific inference settings are standardised where possible: the open-weight models use 4-bit NF4 quantisation, greedy decoding, and a maximum of 350 generated tokens, while the API models use their respective structured-output interfaces with medium reasoning or thinking settings. Although Qwen3-VL-8B-Instruct and gemma-4-12B are multimodal models, no image input is provided in this benchmark; all models are evaluated using the same limited claim-and-context representation.

## 5.3 Evaluation

Classification. Misleading mechanism, emotional arousal, and communicative intent are formulated as multi-class classification tasks. We use Macro-F1 as the primary evaluation metric because it gives equal weight to each class and is therefore less affected by class imbalance. We additionally report class-wise F1 scores to examine performance across individual categories.

Likely Reader Interpretation. We evaluate generated likely reader interpretations at both lexical and semantic levels. Lexical similarity is measured using ROUGE-L. Because semantically equivalent interpretations may differ substantially in wording, we additionally conduct a meaning-level semanticequivalence evaluation. Each generated interpretation is compared with the reference annotation and classified as fully equivalent, partially equivalent, or non-equivalent. Full equivalence indicates that the same central reader takeaway is preserved; partial equivalence indicates that the main interpretation is retained but an important qualifier, scope, causal relation, or implication is missing or altered; and non-equivalence indicates a substantially different or contradictory interpretation. Empty or malformed reference interpretations are excluded from semantic evaluation. We further report a se-

mantic adequacy score,

$$
S _ { \mathrm { s e m } } = \frac { N _ { \mathrm { f u l l } } + 0 . 5 N _ { \mathrm { p a r t i a l } } } { N _ { \mathrm { f u l l } } + N _ { \mathrm { p a r t i a l } } + N _ { \mathrm { n o n } } } ,
$$

which assigns scores of 1, 0.5, and 0 to fully equivalent, partially equivalent, and non-equivalent outputs, respectively.

## 5.4 Results

Table 5 presents the class-level classification results, while Table 6 reports the generation performance for likely reader interpretation.

R1: Strong semantic recovery despite limited lexical overlap. The generation results reveal a clear distinction between lexical similarity and semantic recovery. Although ROUGE-L scores are relatively modest (0.228 - 0.387), 84 - 97% of generated interpretations are fully semantically equivalent to the reference, with only <=2% classified as non-equivalent. Claude Fable 5 provides the clearest example: despite achieving the lowest ROUGE-L score (0.228), it obtains the highest full-equivalence rate (97%) and a semantic adequacy score of 0.98. This indicates that models can recover the intended reader interpretation from limited claim-and-context information even when their wording differs substantially from the reference. The results also demonstrate that lexicaloverlap metrics alone can underestimate the quality of reader-interpretation generation.

R2:Recovery of Emotional and Communicative Cues. GPT-5.6 Sol achieves the strongest overall performance for both emotional arousal (Macro-F1 = 0.643) and communicative intent (Macro-F1 = 0.607). Its performance is particularly strong for high arousal (F1 = 0.871), persuasive intent (F1 = 0.966), and distortive intent (F1 = 0.959). These results suggest that broad emotional and communicative cues can often be recovered from limited claim-and-context information, even without conducting full evidential verification or processing the original multimodal content.

R3: Challenging on Misleading-mechanism Classification. Misleading-mechanism classification remains challenging across most models, with substantially lower Macro-F1 than emotional arousal for four of the five models. Claude Fable 5 achieves the strongest overall mechanism performance (Macro-F1 = 0.520). The not misleading class is particularly difficult for most models, with F1 scores ranging from 0.031 to 0.184 for DeepSeek-v4-flash, Qwen3-VL-8B, GPT-5.6 Sol, and Gemma-4-12B, while Claude Fable 5 performs considerably better (0.393). This pattern may reflect the difficulty of making reliable judgments about misleading mechanisms from limited information, particularly when the available cues appear potentially suspicious.

R4: Intermediate Recoverability of Unsupported Inference. Unsupported inference presents an interesting intermediate case. Models can sometimes recognise that a claim makes a stronger causal or interpretive conclusion than is directly supported by the supplied context, yet reliable identification still depends on knowing what conclusions the underlying evidence actually warrants. Taken together, these differences suggest that misleading mechanisms vary in their recoverability from low-cost textual cues. Mechanisms with more explicit lexical, numerical, or presentational signals appear more accessible, whereas mechanisms defined by missing, displaced, or externally verifiable information depend much more strongly on additional evidence.

## 5.5 Discussion

Returning to the benchmark question, our results show that claim-and-context information preserves substantial but incomplete cues for understanding multimodal misleadingness. Across model families, likely reader interpretations, emotional arousal, and communicative intent can often be recovered without access to the richer evidential and multimodal information used to establish the reference annotations. In contrast, identifying the precise mechanism through which misleadingness arises remains considerably less reliable, particularly when it depends on omitted information, displaced context, or evidence external to the message.

These results reveal an important distinction between contextual signals of misleadingness and context-grounded misleadingness understanding. Lightweight claim-and-context representations can preserve useful interpretive, affective, and communicative signals, but they are not sufficient for reliably determining whether and how a message misleads. Such judgements often require richer contextual and evidential information about what is absent, how the message relates to its original setting, and what interpretation is warranted by the available evidence. Claim-and-context representations are therefore best viewed as a low-cost basis for preliminary analysis and targeted retrieval, rather than a replacement for full contextual and multimodal reasoning.

## 6 Limitations

This study has several limitations that should be considered when interpreting the findings.

Limited representation of non-misleading communication. The dataset is naturally biased towards disputed or potentially misleading claims. As a result, only a small proportion of RCMN instances are labelled as not misleading. This imbalance limits the benchmark’s ability to evaluate how reliably models distinguish misleading communication from ordinary, evidence-compatible communication, and may partly contribute to the low F1 scores observed for the not misleading class. Future work should incorporate a larger and more diverse set of non-misleading controls.

AI-assisted annotation and interpretive subjectivity. The annotation process combines AIassisted evidence recovery and initial annotation with human verification and adjudication. This design improves scalability and auditability, but reader-centric dimensions such as likely interpretation, emotional arousal, and communicative intent remain partly interpretive. Human verification reduces unsupported AI inference, but it cannot eliminate disagreement about how different readers may understand the same communication. The resulting labels should therefore be interpreted as evidencegrounded reference annotations rather than deterministic representations of every possible reader response.

Reconstruction rather than complete multimodal preservation. The original post, image, video, or other media is not recoverable for every instance. In such cases, the communication is reconstructed from the fact-check article and associated provenance information, with the original source used as supplementary reference when available.

## 7 Conclusion

This work introduces RCMN, a reader-centric framework, evidence-grounded dataset, and benchmark for understanding misleadingness in influential public discourse beyond claim-level factuality.

Table 5: Per-class and Macro-F1 performance across reader-centric misleadingness dimensions. All models are evaluated on the same 2216 shared instances. Bold indicates the best result in each row.
<table><tr><td>Dimension</td><td>Class</td><td>GPT-5.6 Sol</td><td>Qwen3-VL-8B</td><td>Deepseek-v4-flash</td><td>Gemma-4-12B</td><td>Claude Fable 5</td></tr><tr><td rowspan="8">Mechanism</td><td>Fabrication / alteration</td><td>0.521</td><td>0.147</td><td>0.587</td><td>0.237</td><td>0.634</td></tr><tr><td>Miscontextualisation</td><td>0.350</td><td>0.123</td><td>0.523</td><td>0.284</td><td>0.508</td></tr><tr><td>Omission / selective presentation</td><td>0.276</td><td>0.275</td><td>0.394</td><td>0.090</td><td>0.463</td></tr><tr><td>Misattribution</td><td>0.199</td><td>0.149</td><td>0.173</td><td>0.080</td><td>0.414</td></tr><tr><td>Exaggeration / quantitative distortion</td><td>0.586</td><td>0.341</td><td>0.480</td><td>0.173</td><td>0.647</td></tr><tr><td>Unsupported inference</td><td>0.567</td><td>0.113</td><td>0.102</td><td>0.433</td><td>0.578</td></tr><tr><td>Not misleading</td><td>0.179</td><td>0.052</td><td>0.031</td><td>0.184</td><td>0.393</td></tr><tr><td>Macro-F1</td><td>0.383</td><td>0.170</td><td>0.333</td><td>0.206</td><td>0.520</td></tr><tr><td rowspan="4">Arousal</td><td>Low</td><td>0.288</td><td>0.597</td><td>0.342</td><td>0.487</td><td>0.472</td></tr><tr><td>Moderate</td><td>0.772</td><td>0.550</td><td>0.590</td><td>0.430</td><td>0.555</td></tr><tr><td>High</td><td>0.871</td><td>0.613</td><td>0.824</td><td>0.547</td><td>0.729</td></tr><tr><td>Macro-F1</td><td>0.643</td><td>0.502</td><td>0.550</td><td>0.488</td><td>0.586</td></tr><tr><td rowspan="4">Intent</td><td>Informative</td><td>0.525</td><td>0.147</td><td>0.231</td><td>0.213</td><td>0.306</td></tr><tr><td>Persuasive</td><td>0.966</td><td>0.533</td><td>0.6139</td><td>0.433</td><td>0.501</td></tr><tr><td>Distortive</td><td>0.959</td><td>0.603</td><td>0.749</td><td>0.421</td><td>0.824</td></tr><tr><td>Macro-F1</td><td>0.607</td><td>0.320</td><td>0.405</td><td>0.263</td><td>0.408</td></tr><tr><td>Mean Macro-F1 across dimensions</td><td></td><td>0.544</td><td>0.331</td><td>0.429</td><td>0.319</td><td>0.505</td></tr></table>

Table 6: Evaluation of generated likely reader interpretations. ROUGE-L measures lexical overlap, while semanticequivalence evaluation measures meaning-level agreement with the reference interpretation.
<table><tr><td>Model</td><td>ROUGE-L ↑</td><td>Full Equiv. ↑</td><td>Partial Equiv.</td><td>Non-Equiv. ↓</td><td>Semantic Adequacy ↑</td></tr><tr><td>GPT-5.6 Sol</td><td>0.387</td><td>91%</td><td>8%</td><td>1%</td><td>0.95</td></tr><tr><td>Qwen</td><td>0.289</td><td>84%</td><td>14%</td><td>2%</td><td>0.91</td></tr><tr><td>DeepSeek</td><td>0.256</td><td>93%</td><td>5%</td><td>2%</td><td>0.95</td></tr><tr><td>Gemma</td><td>0.314</td><td>91%</td><td>7%</td><td>1%</td><td>0.95</td></tr><tr><td>Claude Fable 5</td><td>0.228</td><td>97%</td><td>3%</td><td>&lt;1%</td><td>0.98</td></tr></table>

RCMN captures how messages become misleading, what interpretations they encourage relative to available evidence, and the affective and communicative signals that shape those interpretations.

Our findings show that reader-centric misleadingness contains both readily observable communicative cues and deeper evidence-dependent mechanisms. This distinction points to a promising direction for future research: developing adaptive models that use lightweight contextual signals for initial assessment while selectively retrieving richer contextual, evidential, or multimodal information when deeper verification is required. Such approaches could support more scalable and reliable understanding of how and why influential public discourse may mislead.

## 8 Ethics Statement

RCMN is constructed from publicly circulated discourse and professionally reviewed source material. The annotation process combines AI-assisted evidence recovery and initial annotation with human verification and adjudication. Because the dataset may contain sensitive or politically salient public communication, the current study focuses on research use and does not infer private attributes or intentions beyond the communicative signals defined in the annotation framework.

## References

Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2020. Generating fact checking explanations. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 7352–7364.

Isabelle Augenstein, Christina Lioma, Dongsheng Wang, Lucas Chaves Lima, Casper Hansen, Christian Hansen, and Jakob Grue Simonsen. 2019. Multifc: A real-world multi-domain dataset for evidence-based fact checking of claims. In Proceedings ofthe 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 4685–4697.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others.

2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Ceren Budak, Brendan Nyhan, David M Rothschild, Emily Thorson, and Duncan J Watts. 2024. Misunderstanding the harms of online misinformation. Nature, 630(8015):45–53.

Megha Chakraborty, Khushbu Pahwa, Anku Rani, Shreyas Chatterjee, Dwip Dalal, Harshit Dave, Preethi Gurumurthy, Adarsh Mahor, Samahriti Mukherjee, Aditya Pakala, and 1 others. 2023. Factify3m: A benchmark for multimodal fact verification with explainability through 5w question-answering. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 15282–15322.

Dennis Chong and James N Druckman. 2010. Dynamic public opinion: Communication effects over time. American Political Science Review, 104(4):663–680.

Jeff Da, Maxwell Forbes, Rowan Zellers, Anthony Zheng, Jena D Hwang, Antoine Bosselut, and Yejin Choi. 2021. Edited media understanding frames: Reasoning about the intent and implications of visual misinformation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2026–2039.

James N Druckman. 2001. On the limits of framing effects: Who can frame? The journal of politics, 63(4):1041–1066.

Ullrich KH Ecker, Stephan Lewandowsky, John Cook, Philipp Schmid, Lisa K Fazio, Nadia Brashier, Panayiota Kendeou, Emily K Vraga, and Michelle A Amazeen. 2022. The psychological drivers of misinformation belief and its resistance to correction. Nature reviews psychology, 1(1):13–29.

Robert M Entman and 1 others. 1993. Framing: Towards clarification of a fractured paradigm. Mc-Quail’s reader in mass communication theory, 390:397.

Fact-Check Insights. 2026. Guide to the data.

Saadia Gabriel, Skyler Hallinan, Maarten Sap, Pemi Nguyen, Franziska Roesner, Eunsol Choi, and Yejin Choi. 2022. Misinfo reaction frames: Reasoning about readers’ reactions to news headlines. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3108–3127.

Jiahui Geng, Jonathan Tonglet, and Iryna Gurevych. 2025. M4fc: a multimodal, multilingual, multicultural, multitask real-world fact-checking dataset. arXiv preprint arXiv:2510.23508.

D’Angelo Gore. 2023. Trump’s misleading poverty rate comparison. FactCheck.org.

David MJ Lazer, Matthew A Baum, Yochai Benkler, Adam J Berinsky, Kelly M Greenhill, Filippo Menczer, Miriam J Metzger, Brendan Nyhan, Gordon Pennycook, David Rothschild, and 1 others. 2018. The science of fake news. Science, 359(6380):1094– 1096.

Fanxiao Li, Jiaying Wu, Tingchao Fu, Dayang Li, Herun Wan, Wei Zhou, and Min-Yen Kan. 2026. What’s left unsaid? detecting and correcting misleading omissions in multimodal news previews. arXiv preprint arXiv:2601.05563.

Jula Lühring, Apeksha Shetty, Corinna Koschmieder, David Garcia, Annie Waldherr, and Hannah Metzler. 2024. Emotions in misinformation studies: distinguishing affective state from emotional response and misinformation recognition from acceptance. Cognitive research: principles and implications, 9(1):82.

Grace Luo, Trevor Darrell, and Anna Rohrbach. 2021. Newsclippings: Automatic generation of out-ofcontext multimodal media. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6801–6817.

Maxwell E McCombs and Donald L Shaw. 1972. The agenda-setting function of mass media. Public opinion quarterly, 36(2):176–187.

Shreyash Mishra, S Suryavardan, Amrit Bhaskar, Parul Chopra, Aishwarya N Reganti, Parth Patwa, Amitava Das, Tanmoy Chakraborty, Amit P Sheth, Asif Ekbal, and 1 others. 2022. Factify: A multi-modal fact verification dataset. In DE-FACTIFY@ AAAI, page np.

Arkadiusz Modzelewski, Witold Sosnowski, Eleni Papadopulos, Elisa Sartori, Tiziano Labruna, Giovanni Da San Martino, and Adam Wierzbicki. 2026. Malicious intent dataset and inoculating llms for enhanced disinformation detection. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3125–3148.

Kai Nakamura, Sharon Levy, and William Yang Wang. 2020. Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection. In Proceedings of the twelfth language resources and evaluation conference, pages 6149–6157.

Jingwei Ni, Minjing Shi, Dominik Stammbach, Mrinmaya Sachan, Elliott Ash, and Markus Leippold. 2024. Afacta: Assisting the annotation of factual claim detection with reliable llm annotators. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 1890–1912.

Dan S Nielsen and Ryan McConville. 2022. Mumin: A large-scale multilingual multimodal fact-checked misinformation social network dataset. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, pages 3141–3153.

Laura Ana Maria Oberländer, Evgeny Kim, and Roman Klinger. 2020. Goodnewseveryone: A corpus of news headlines annotated with emotions, semantic roles, and reader perception. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1554–1566.

OpenAI. 2026. Gpt-5.6 sol model. OpenAI API Documentation. Accessed August 2026.

Stefanos-Iordanis Papadopoulos, Christos Koutlis, Symeon Papadopoulos, and Panagiotis C Petrantonakis. 2024. Verite: a robust benchmark for multimodal misinformation detection accounting for unimodal bias. International Journal of Multimedia Information Retrieval, 13(1):4.

Irene V Pasquetto, Gabrielle Lim, and Samantha Bradshaw. 2024. Misinformed about misinformation: On the polarizing discourse on misinformation and its consequences for the field. Harvard Kennedy School Misinformation Review, 5(5):1–8.

Anne Reboul. 2021. Truthfully misleading: Truth, informativity, and manipulation in linguistic communication. Frontiers in Communication, 6:646820.

Todd Rogers, Richard Zeckhauser, Francesca Gino, Michael I Norton, and Maurice E Schweitzer. 2017. Artful paltering: The risks and rewards of using truthful statements to mislead others. Journal of personality and social psychology, 112(3):456.

Mark Rothermel, Marcus Kornmann, Marcus Rohrbach, and Anna Rohrbach. 2026. Veritas: The first dynamic benchmark for multimodal automated factchecking. arXiv preprint arXiv:2601.08611.

Dietram A Scheufele and David Tewksbury. 2007. Framing, agenda setting, and priming: The evolution of three media effects models. Journal of communication, 57(1):9–20.

Kai Shu, Deepak Mahudeswaran, Suhang Wang, Dongwon Lee, and Huan Liu. 2020. Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Big data, 8(3):171–188.

S Suryavardan, Shreyash Mishra, Parth Patwa, Megha Chakraborty, Anku Rani, Aishwarya Reganti, Aman Chadha, Amitava Das, Amit Sheth, Manoj Chinnakotla, and 1 others. 2023. Factify 2: A multimodal fake news and satire news dataset. arXiv preprint arXiv:2304.03897.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle˘ Casbon, and 1 others. 2026. Gemma 4 technical report. arXiv preprint arXiv:2607.02770.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. Fever: a large-scale dataset for fact extraction and verification. In Proceedings of the 2018 Conference

of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819.

Jonathan Tonglet, Gabriel Thiem, and Iryna Gurevych. 2025. COVE: COntext and VEracity prediction for out-of-context images. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2029–2049, Albuquerque, New Mexico. Association for Computational Linguistics.

Sander van der Linden, Debra Louison-Lavoy, Nicholas Blazer, Nancy S Noble, and Jon Roozenbeek. 2026. Prebunking misinformation techniques in social media feeds: Results from an instagram field study. Harvard Kennedy School Misinformation Review.

Soroush Vosoughi, Deb Roy, and Sinan Aral. 2018. The spread of true and false news online. science, 359(6380):1146–1151.

William Yang Wang. 2017. “liar, liar pants on fire”: A new benchmark dataset for fake news detection. In Proceedings of the 55th annual meeting of the associationfor computational linguistics (volume 2: short papers), pages 422–426.

Zhuohan Xie, Rui Xing, Yuxia Wang, Jiahui Geng, Hasan Iqbal, Dhruv Sahnan, Iryna Gurevych, and Preslav Nakov. 2025. Fire: Fact-checking with iterative retrieval and verification. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 2901–2914.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, and 1 others. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

Barry Menglong Yao, Aditya Shah, Lichao Sun, Jin-Hee Cho, and Lifu Huang. 2023. End-to-end multimodal fact-checking and explanation generation: A challenging dataset and models. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2733–2743.

Peiling Yi and Arkaitz Zubiaga. 2026. From fact verification to understanding misleadingness: A survey and roadmap on reader-centric multimodal misinformation detection.