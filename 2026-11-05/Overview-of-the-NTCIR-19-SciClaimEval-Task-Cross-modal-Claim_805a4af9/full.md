# Overview of the NTCIR-19 SciClaimEval Task: Cross-modal Claim Verification in Scientific Papers

Xanh Ho\* NII LLMC, Japan xanh@nii.ac.jp

Yun-Ang Wu NII LLMC, Japan yunangwu@nii.ac.jp

Hiba Akbi   
Sorbonne University, France   
Hiba.Akbi@etu.sorbonne  
universite.fr

André Greiner-Petter\* University of Göttingen, Germany greinerpetter@gipplab.org

Sunisth Kumar The University of Tokyo, Japan sunisth@g.ecc.u-tokyo.ac.jp

Tim Schopf NII, Japan tim.schopf@t-online.de

Florian Boudin Inria, LS2N, Nantes Université, France florian.boudin@univ-nantes.fr

Tian Cheng Xia University of Bologna, Italy tiancheng.xia@studio.unibo.it

Akiko Aizawa NII, Japan aizawa@nii.ac.jp

## Abstract

This paper presents an overview of SciClaimEval, a pilot task at NTCIR-19 on verifying scientific claims against the tables and fig ures of the papers in which they appear. Our dataset pairs authentic claims from 180 papers in machine learning, natural language processing, and medicine with contrastive versions of their tables and figures, one supporting the claim and one minimally modified so that it does not. The task comprises two subtasks: deciding whether a piece of evidence supports a claim, and selecting which of two evidence versions does. Twenty-two teams registered, the highest number among NTCIR-19 tasks, and 14 submitted 177 runs. Most systems built on multimodal large language models, relying on prompting and multi-stage inference rather than task-specific finetuning. On the first subtask, the best systems reached 98.1% pair accuracy, but all six top-ranked systems relied on the contrastive pairs; the best system that judged each claim-evidence pair on its own scored 24.3 points lower. Open-weight models, embedded in carefully designed pipelines, matched proprietary ones. Yet verify ing a single claim against its evidence, the setting that matters in practice, remains an open problem.

## Keywords

Claim Verification, Cross-modal, Scientific Papers

## 1 Introduction

Scientific papers make claims that should be supported by the evi dence they present, typically results reported in tables and figures. Scientific claim verification asks whether a claim is adequately sup ported by this evidence [5, 8, 11]. Unlike general fact-checking [3, 7], which verifies claims against external knowledge, it focuses on the internal consistency between a paper’s claims and its own evidence. Checking this consistency is a fundamental part of peer review, yet it is time-consuming and increasingly dificult as submission volumes grow. Automatic verification could assist reviewers, help authors find unsupported claims before submission, and ultimately improve the reliability of published research.

Existing datasets for scientific claim verification difer mainly in the evidence they consider. Tables are the most common, either in structured formats such as JSON or XML, as in SciTab [5], SEM-TAB-FACTs [11], and SciAtomicBench [13], or as images, as in MMSci-Eval [12]. Others turn to figures, such as MuSciClaims [4], whose figures often contain complex subplots and embedded tables, while SciVer [10] considers multiple charts, tables, and text paragraphs as evidence. Despite this progress, three limitations remain. Refuted examples are usually created by editing the claim, for instance by flipping its meaning, which can leave surface cues that models exploit instead of reasoning over the evidence. Many datasets also rely on claims that are not taken directly from scientific papers. Finally, tables are typically available in a single format, so it remains unclear how well models handle the same content across diferent representations.

With these limitations in mind, we organized SciClaimEval, a pilot task at NTCIR-19 on verifying scientific claims against the tables and figures of the papers in which they appear. It complements earlier shared tasks on scientific claim verification, which used tables (SemEval-2021 Task 9 [11]), abstracts (SCIVER [9]), or social media posts linked to their source papers (CheckThat! [6]). Rather than editing claims, we create refuted examples by having annotators modify the evidence. Each claim is thus paired with two versions of its evidence, one that supports it and one that does not, so the wording of the claim carries no cue about the label. All claims are authentic sentences taken from published papers, and tables are released in four formats: images, LaTeX, HTML, and JSON.

The SciClaimEval dataset is built in three steps. We first collect peer-reviewed papers in natural language processing (NLP) and machine learning (ML), accepted at ACL venues and NeurIPS, and in medicine, published in PeerJ. From each paper, we extract the text, tables, and figures. We then identify sentences that explicitly reference a table or figure (e.g., “Table 1”), as well as other sentences in the same paragraph, and pair each sentence individually with the referenced evidence. Finally, annotators check whether each claim is supported by its evidence and, for supported claims, edit the evidence to create an unsupported counterpart. The resulting dataset contains 1,633 claims from 180 papers, with both table- and figure-based evidence across all three domains (Section 2.1).

![](images/fc0bbf71c97c50a7ccd78eef08dbb9ae7c1b7516ef09b5ca86e9d45da2a3bf16.jpg)  
Figure 1: Our dataset construction pipeline consists of three main steps: data preparation, automatic claim-evidence extraction, and human annotation. The human annotation process involves two tasks: claim-evidence verification and evidence modifica tion. After collecting all samples, we design two subtasks in our dataset: claim-label prediction and claim-evidence prediction.

The pilot task comprises two subtasks. In Subtask 1, systems decide whether a given piece of evidence supports a claim. In Subtask 2, systems receive a claim with its two evidence versions and must select the one that supports it. Twenty-two teams registered, the highest number among NTCIR-19 tasks, and 14 of them submitted 177 runs in total: 106 for Subtask 1 and 71 for Subtask 2.

Most systems built on multimodal large language models (MLLMs). For Subtask 1, they followed two broad strategies. Some verified each claim-evidence pair independently, often with claim decomposition, structured extraction of evidence from images, or executable checks. Others exploited the paired structure of the data by comparing the two evidence versions of the same claim. For Subtask 2, systems primarily adopted direct candidate comparison, independent verification followed by selection, or hybrid approaches combining both strategies. Common techniques included ensembles, multi-agent debate, evidence extraction, and consistency checks under reversed candidate order.

We draw the following lessons from this pilot task:

• High scores rely on paired evidence. On Subtask 1, the best systems reached 98.1% pair accuracy on the test set by comparing the two evidence versions of each claim. Teams that also evaluated their systems without this information observed substantial performance drops. On the development set, for instance, removing paired information cost the two top systems 12 and 21 points of pair accuracy.

• Open-weight models can match proprietary ones. A structured pipeline built around an open-weight model, with selective human review, matched direct prompting of a proprietary model on Subtask 1 (98.1% pair accuracy for both) and came close on Subtask 2 (98.1% versus 98.4% accuracy).

• Verifying a single claim against its evidence remains an open problem. In practice, reviewers and authors face one claim and one piece of evidence, with no contrasting version to compare against. This is precisely the setting in which performance drops, and one that our paired design did not measure directly.

## 2 Task Design and Dataset

## 2.1 Dataset Creation

Figure 1 illustrates our dataset construction pipeline, which consists of three steps: data preparation, automatic claim-evidence extraction, and human annotation.

Data Preparation. We collect papers from three domains. For NLP, we take papers from the ACL Anthology and map them to their arXiv versions when available. For ML, we select arXiv papers whose comments indicate acceptance at conferences such as NeurIPS. For both domains, we retrieve the HTML-rendered papers from ar5iv [1] and parse them into JSON containing the title, abstract, and section paragraphs. Figures are extracted from ar5iv as PNG files, and tables are extracted from the LaTeX source and kept in LaTeX format. For medicine, we collect papers published in PeerJ between 2024 and 2025 in the Medicine Articles category. We parse their HTML versions into JSON, storing figures as PNG files and tables in HTML format.

Claim-Evidence Extraction. We identify paragraphs that refer to a table or figure using regular expressions that match references such as “Table 1”, “Tab. 1”, or “Fig. 2”. Each such paragraph is split into sentences, and each sentence is paired with the referenced table or figure, yielding a set of candidate claim-evidence pairs. Not every sentence is a verifiable claim, so these candidates are filtered during human annotation. We consider only the main text, excluding appendices and their tables and figures.

Human Annotation. Annotation was carried out by 11 graduate students and researchers, in two stages: claim-evidence verification and evidence modification.

![](images/89e38b327b4abf7a30a8f5d98a2fac60e292dee403cddc14dd7dfcc9fded3fba.jpg)

## Claim

The figure reveals a set of diverse exploration strategies within the human population and confirms that RR-RL 2 captures the overall variability in human exploration appropriately.

The figure reveals a set of diverse exploration strategies within the human population and confirms   
that RR-RL 2 captures the overall variability in human exploration appropriately.

![](images/6fba268f87f48aa694ca6e4e82ae98a7f80728b826140f101a01ede4ba9ce47c.jpg)

![](images/69f391199f38cf7524cd9ad23134b29dba40b6680cc904664dbd04d7d77ccab1.jpg)  
Figure 2: An example from our dataset involves creating an unsupported claim by adding spurious data points to the figure. The context provides the necessary information to understand the plot in the evidence.

## Context

Finally, we compared the probit regression coefficients of human participants to the ones of RR-RL2. Figure 1 (b) shows a twodimensional UMAP embedding (McInnes et al., 2018) of these coefficients.

## Label

![](images/8808f577d3eac1ea347bec760e1b048db9a0b30b6964c9d19c5a568c22cc73e6.jpg)

Finally, we compared the probit regression coefficients of human participants to the ones of RR-RL2. Figure 1 (b) shows a twodimensional UMAP embedding (McInnes et al., 2018) of these coefficients.

In the first stage, annotators see a candidate claim, its table or figure, and the preceding sentences of the same paragraph as context (empty ifthe claim is the first sentence). They label each pair as Supported, Subjectively Supported (the claim relies on subjective terms such as “large margin” that cannot be checked against the evidence), Unsupported, or Skip (the claim is purely descriptive, problematic, or requires knowledge the annotator lacks). They also indicate whether the claim can be understood on its own, needs the short context, or requires the full paper.

In the second stage, annotators work only on Supported pairs. They modify the table or figure using predefined operations so that the claim is no longer supported by the modified evidence. For tables, they change cell values, swap row or column names, or add and remove rows or columns. For figures, they flip graphs, swap legends or category labels, or exchange subgraphs from the same paper. Annotators may also propose other edits, such as adding spurious data points (Figure 2). For a few claims, no meaningful modification was possible, and only the supported pair was kept. Further details are given in Ho et al. [2].

## 2.2 Dataset Statistics

Table 1 reports statistics for the validation set, the test set, and the full dataset, broken down by modality (table or figure), context use, and domain. The split into validation and test sets is balanced with respect to context use and evidence modification operations. The dataset contains 863 supported and 770 refuted samples. Although each supported claim is meant to have a refuted counterpart, 93 have none, because no modification of the evidence could be found that was both logical and challenging.

Note that the statistics in Table 1 difer slightly from those reported in Ho et al. [2]. Following feedback from participants and further analysis, we removed a small number of samples to improve the quality of the dataset. Both the original version (v1.0) and the updated version (v1.1) are available on our Hugging Face page<sup>1</sup> for reproducibility.

Table 1: Dataset statistics of SciClaimEval.
<table><tr><td>Property</td><td>Val</td><td>Test</td><td>All Data</td></tr><tr><td>Labels</td><td></td><td></td><td></td></tr><tr><td>#Supported</td><td>389</td><td>474</td><td>863</td></tr><tr><td>#Refuted</td><td>338</td><td>432</td><td>770</td></tr><tr><td>Modality Type</td><td></td><td></td><td></td></tr><tr><td>Table</td><td>469</td><td>518</td><td>987</td></tr><tr><td>Figure</td><td>258</td><td>388</td><td>646</td></tr><tr><td>Context Use</td><td></td><td></td><td></td></tr><tr><td>No</td><td>477</td><td>611</td><td>1,088</td></tr><tr><td>Short Context</td><td>142</td><td>188</td><td>330</td></tr><tr><td>Full Paper</td><td>108</td><td>107</td><td>215</td></tr><tr><td>Domain</td><td></td><td></td><td></td></tr><tr><td>NLP</td><td>379</td><td>384</td><td>763</td></tr><tr><td>ML</td><td>157</td><td>236</td><td>393</td></tr><tr><td>Medicine</td><td>191</td><td>286</td><td>477</td></tr><tr><td>Total examples</td><td>727</td><td>906</td><td>1,633</td></tr></table>

## 2.3 Dataset Analysis

In addition to the analyses of Text Lengths, Evidence-Modifying Operations, and Pixel Changes in Figure Evidence Modification presented in Ho et al. [2], we extend the analysis to include chart types. We manually categorized all figures in the development and test sets. First, we classified figures into four categories based on their plot structure: Single (a single plot), Multiple (multiple unlabeled plots), Subplots (multiple labeled plots, e.g., (a), (b)), and Multiple Subplots (a combination of multiple plots and subplots). Each individual plot, including those within multiple plots or sub plots, may contain a combination of diferent chart types. The most common combinations are bar charts with error bars or margins. For example, such a figure is categorized as single in terms of plot structure but assigned both bar chart and error plot as chart types.

![](images/1ac527ab0fc648af4ae5ef99eda364c6e83040c69e1becdecc9fa3ecc5d5f15b.jpg)  
Figure 3: Distribution of all figure types in test and dev split.

Figure 3 shows the distribution of plot structures and chart types across the entire SciClaimEval dataset (development and test splits). We observe a clear bias toward single plots, with line charts, bar charts, and error plots being the most prevalent chart types. Error plots rank among the top three primarily because they frequently appear alongside line and bar charts, but rarely occur independently.

## 2.4 Task Design

SciClaimEval comprises two subtasks.

Subtask 1: Claim-Label Prediction. This is the main subtask. Given a claim, a piece of evidence (a table or a figure) with its caption, and the contextual information, systems predict whether the claim is supported or refuted by the evidence. This subtask evaluates whether systems can understand a claim, interpret the evidence, and judge whether the two are consistent.

Subtask 2: Claim-Evidence Prediction. Given a claim, its context, and the two versions of its evidence (the original and the modified one), systems select the version that supports the claim. The two versions often difer by a single cell or legend entry, so this subtask tests whether systems can find the detail on which the claim depends.

## 2.5 Evaluation Metrics

Subtask 1. We report precision, recall, macro-F1, accuracy, and pair accuracy, and use the last two as primary metrics. Pair accuracy counts a claim as correct only if both of its samples, supported and refuted, are classified correctly. Claims without a refuted counterpart are excluded from this metric.

Subtask 2. We report accuracy, the proportion ofclaims for which the supporting evidence is correctly selected. The random baseline is 50%.

## 3 Participating Systems

Fourteen teams submitted 177 runs to SciClaimEval, with thirteen teams taking part in each subtask (Table 2). This section summarizes the main approaches for each subtask, based on the teams’ system description papers.

Table 2: Participating teams, the subtasks they submitted runs to, and their number of runs.
<table><tr><td>Team ID</td><td>Group name</td><td>Country</td><td>Subtasks</td><td># Runs</td></tr><tr><td>SIB</td><td>SIB Text Mining</td><td>Switzerland</td><td>1,2</td><td>8</td></tr><tr><td>bit-ias9</td><td>Bonn-Juelich Informatics</td><td>Germany</td><td>1,2</td><td>44</td></tr><tr><td>tabr25</td><td>Black Socks</td><td>Australia</td><td>1,2</td><td>12</td></tr><tr><td>ECS-UoS</td><td>University of Southampton</td><td>UK</td><td>1,2</td><td>2</td></tr><tr><td>ditlab</td><td>Denso IT Laboratory</td><td>Japan</td><td>1,2</td><td>5</td></tr><tr><td>trace29</td><td>SEBIS TUM team</td><td>Germany</td><td>1,2</td><td>2</td></tr><tr><td>tlab</td><td>TakasuLab</td><td>Japan</td><td>1,2</td><td>7</td></tr><tr><td>procrast</td><td>Pro2Crast</td><td>Japan</td><td>1,2</td><td>27</td></tr><tr><td>SciNex</td><td>SciNex</td><td>Japan</td><td>1,2</td><td>8</td></tr><tr><td>ProvideQ</td><td>ProvideQ</td><td>Germany</td><td>1,2</td><td>30</td></tr><tr><td>SCIAI19</td><td>VerifyAI</td><td>Germany</td><td>1,2</td><td>4</td></tr><tr><td>TUMNLP26</td><td>LMU &amp; TUM NLP team</td><td>Germany</td><td>2</td><td>7</td></tr><tr><td>SciTrue</td><td>SciTrue</td><td>New Zealand</td><td>1,2</td><td>18</td></tr><tr><td>CAISA26</td><td>CAISALab</td><td>Germany</td><td>1</td><td>3</td></tr><tr><td colspan="5">Total</td></tr></table>

## 3.1 Subtask 1: Claim-Label Prediction

Approaches to Subtask 1 followed three main directions.

Exploiting the Paired Structure. The most prominent strategy coordinated predictions across the two samples of each claim. Black Socks, Southampton, and ditlab compared the two evidence versions associated with the same claim and assigned them complementary labels. SciTrue combined model support scores with a pairing constraint, while Bonn-Juelich mapped its Subtask 2 selections back to Subtask 1 labels. By drawing on information beyond a single claim-evidence pair, these approaches achieved the highest scores. Their gains, however, do not transfer to real-world verification, where a claim is assessed without a second, contrasting version of its evidence.

Separating Extraction from Judgment. Several teams first turned the evidence into text and then judged the claim against it. VerifyAI transcribed images into text for a separate evaluator. ProvideQ converted tables into HTML and extracted structured descriptions of figure elements before verification. SciNex explored structured prompting and a two-stage variant that supplemented figure images with descriptions generated without seeing the claim.

Strengthening Verification. Other teams made the verification step itself more rigorous. Bonn-Juelich combined claim decomposition, context retrieval, and iterative checks. Takasu Lab checked verifiable conditions with Python code, and CAISALab explored executable Prolog queries. Pro2Crast re-verified low-confidence predictions against the full paper, and SIB Text Mining used contradictionfocused prompting and fine-tuning on synthetic data.

## 3.2 Subtask 2: Claim-Evidence Prediction

Approaches to Subtask 2 also followed three main directions.

Direct Comparison. Most teams presented both candidates to the model at once. SciTrue prompted individual models to compare the candidates directly, or fused their scores in an ensemble. Bonn-Juelich built a more elaborate joint pipeline, combining claim decomposition, discrepancy reports, iterative verification, and candidate-order checks, with selective human review. Pro2Crast supplemented an initial comparison with opposing arguments or interactive debate before a judge selected the evidence, while ditlab explored direct selection across evidence formats and context settings.

Independent Verification, then Selection. Other teams first assessed each candidate on its own. Takasu Lab generated separate verification traces, including deterministic checks and intermediate reasoning, which a text-based selector then compared. Southampton and SIB Text Mining used conditional combinations: they initially verified candidates independently and escalated ambiguous cases to joint comparison. This allowed them to retain independent judgments when these clearly distinguished the candidates and reserve additional comparison for uncertain decisions.

Separating Extraction from Judgment. As in Subtask 1, several teams first extracted information from the evidence, but used it to support a joint decision rather than to label each candidate. VerifyAI compared independently generated transcripts, while SciNex jointly evaluated claim-blind descriptions alongside the original images. LMU & TUM NLP explored selection from structured fig ure descriptions, and ProvideQ separately grounded each figure before comparing the extracted information. These approaches used separate extraction to support a joint decision, rather than independently assigning support labels to the candidates.

## 4 Experimental Results

Fourteen teams submitted 177 runs: 106 for Subtask 1 and 71 for Subtask 2, with thirteen teams participating in each subtask. The oficial ranking is based on runs using the PNG version of the evidence, since structured formats such as JSON, LaTeX, and HTML are likely easier for language models to interpret. Participants could also submit optional runs using these structured formats. Note that all scores reported here are evaluated on the test set of our dataset.

## 4.1 Subtask 1: Claim-Label Prediction

Table 3 shows the best-performing runs<sup>2</sup> of each team for Subtask 1, where systems are ranked by pair accuracy. Bonn-Juelich Informatics and SciTrue achieved the highest pair accuracy of 98.1%, with SciTrue slightly ahead on all other metrics. Both systems clearly outperformed the other participants, with the third-ranked team trailing by 8.1 percentage points in pair accuracy.

Table 3: Best run of each team on Subtask 1 (PNG evidence), ranked by pair accuracy. <sup>†</sup> indicates runs that use the paired structure of the test data.
<table><tr><td>Team</td><td>P</td><td>R</td><td>Macro-F1</td><td>Acc.</td><td>Pair Acc.</td></tr><tr><td>SciTrue†</td><td>98.0</td><td>98.0</td><td>98.0</td><td>98.0</td><td>98.1</td></tr><tr><td>Bonn-Juelich Informatics†</td><td>97.2</td><td>97.3</td><td>97.2</td><td>97.2</td><td>98.1</td></tr><tr><td>Black Socks†</td><td>94.1</td><td>93.7</td><td>93.8</td><td>93.8</td><td>90.0</td></tr><tr><td>Pro²Crast†</td><td>87.2</td><td>87.2</td><td>87.2</td><td>87.2</td><td>86.8</td></tr><tr><td>University of Southampton†</td><td>85.8</td><td>85.9</td><td>85.8</td><td>85.9</td><td>85.9</td></tr><tr><td>Denso IT Laboratory†</td><td>74.1</td><td>74.1</td><td>74.1</td><td>74.1</td><td>75.9</td></tr><tr><td>TakasuLab</td><td>86.7</td><td>86.4</td><td>86.5</td><td>86.5</td><td>73.8</td></tr><tr><td>SEBIS TUM</td><td>72.7</td><td>72.7</td><td>72.7</td><td>72.7</td><td>72.9</td></tr><tr><td>VerifyAI</td><td>85.3</td><td>85.4</td><td>85.3</td><td>85.3</td><td>72.2</td></tr><tr><td>ProvideQ†</td><td>84.8</td><td>84.9</td><td>84.9</td><td>84.9</td><td>71.1</td></tr><tr><td>SIB Text Mining</td><td>81.3</td><td>81.3</td><td>81.1</td><td>81.1</td><td>64.1</td></tr><tr><td>Baselines</td><td>80.4</td><td>79.5</td><td>79.8</td><td>79.4</td><td>61.1</td></tr><tr><td>SciNex</td><td>77.5</td><td>77.5</td><td>77.4</td><td>77.4</td><td>57.4</td></tr><tr><td>CAISALab</td><td>62.3</td><td>59.8</td><td>58.2</td><td>60.8</td><td>26.9</td></tr></table>

For Subtask 1, the task design did not explicitly prohibit systems from exploiting the paired structure of the test data. Based on our review of the system descriptions, the six top-ranked systems used this information, typically by assigning complementary labels to the two evidence versions of each claim. Therefore, their results reflect a setting where both evidence versions are available, rather than the intended setting where each claim-evidence pair is evaluated independently. Among systems that did not exploit paired information, TakasuLab achieved the highest pair accuracy of73.8%, which is 24.3 percentage points lower than the top-performing systems. A similar performance drop was observed by the two leading teams on the development set when paired information was removed, with pair accuracy decreasing from 95.7% to 74.4% and from 94.3% to 82.7%, respectively. Additionally, Bonn-Juelich Informatics incorporated selective human review, meaning its runs were not fully automatic. In Table 3, runs that exploited paired information are marked with †.

Tables 4 and 5 show the results for Subtask 1 using the optional structured formats, JSON and LAT<sub>E</sub>X/HTML, respectively. Surprisingly, runs using these formats did not clearly outperform those using PNG evidence. One possible reason is that most teams focused on multimodal models, in line with the primary goal of the task, and optimized their systems for images. Additionally, fewer runs were submitted using these formats, limiting direct comparisons. However, other factors may also contribute to these results, and no firm conclusions can be drawn.

Table 4: Subtask 1 results (JSON evidence).
<table><tr><td>Team</td><td>P</td><td>R</td><td>Macro-F1</td><td>Acc.</td><td>Pair Acc.</td></tr><tr><td>SciTrue</td><td>98.1</td><td>98.1</td><td>98.1</td><td>98.1</td><td>98.4</td></tr><tr><td>Black Socks</td><td>93.1</td><td>93.2</td><td>93.2</td><td>93.2</td><td>88.7</td></tr><tr><td>Pro²Crast</td><td>83.6</td><td>83.7</td><td>83.6</td><td>83.7</td><td>83.6</td></tr><tr><td>TakasuLab</td><td>85.0</td><td>84.9</td><td>84.9</td><td>85.0</td><td>71.3</td></tr><tr><td>SciNex</td><td>78.3</td><td>78.1</td><td>77.8</td><td>77.8</td><td>59.7</td></tr><tr><td>CAISALab</td><td>65.7</td><td>64.7</td><td>64.4</td><td>65.2</td><td>32.6</td></tr></table>

Table 5: Subtask 1 results (TeX/HTML evidence).
<table><tr><td>Team</td><td>P</td><td>R</td><td>Macro-F1</td><td>Acc.</td><td>Pair Acc.</td></tr><tr><td>SciTrue</td><td>98.1</td><td>98.1</td><td>98.1</td><td>98.1</td><td>98.4</td></tr><tr><td>Bonn-Juelich Informatics</td><td>96.7</td><td>96.7</td><td>96.7</td><td>96.7</td><td>97.7</td></tr><tr><td>ProvideQ</td><td>89.9</td><td>90.0</td><td>90.0</td><td>90.0</td><td>82.4</td></tr></table>

## 4.2 Subtask 2: Claim-Evidence Prediction

Table 6 presents the best-performing runs of each team for Subtask 2, where systems are ranked by accuracy. SciTrue achieved the highest accuracy, outperforming Bonn-Juelich Informatics by a narrow margin of 0.3 percentage points. The gap between the top-performing systems and the other participants was smaller than in Subtask 1, with the third-ranked team, the University of Southampton, achieving 96.3%, 2.1 percentage points behind SciTrue.

Table 6: Best run of each team on Subtask 2 results (PNG evidence).
<table><tr><td colspan="2">Team Accuracy</td></tr><tr><td>SciTrue</td><td>98.4</td></tr><tr><td>Bonn-Juelich Informatics</td><td>98.1</td></tr><tr><td>University of Southampton</td><td>96.3</td></tr><tr><td>TakasuLab</td><td>94.0</td></tr><tr><td>ProvideQ</td><td>91.7</td></tr><tr><td>SIB Text Mining</td><td>90.5</td></tr><tr><td>SciNex</td><td>88.2</td></tr><tr><td>Baselines</td><td>81.9</td></tr><tr><td>LMU &amp; TUM NLP</td><td>81.5</td></tr><tr><td>Pro²Crast</td><td>80.8</td></tr><tr><td>Denso IT Laboratory</td><td>61.3</td></tr><tr><td>SEBIS TUM</td><td>60.6</td></tr></table>

The baseline achieves 81.9% accuracy on Subtask 2, compared to 61.1% pair accuracy on Subtask 1, with the six top-performing systems all achieving scores above 90%. This performance diference is expected, as selecting the better-supported evidence from two nearly identical pieces of evidence requires only a relative judgment.

Tables 7 and 8 present the results for Subtask 2 using JSON and LAT<sub>E</sub>X/HTML formats, respectively. Similar to Subtask 1, runs using structured formats did not show a clear advantage over those using PNG evidence. This may be because most participating teams prioritized multimodal approaches and optimized their systems for image-based evidence. The smaller number of submissions using structured formats also limits the comparison. Therefore, further investigation is needed to determine whether structured representations provide advantages over image-based evidence for this task.

Table 7: Subtask 2 results (JSON evidence).
<table><tr><td>Team</td><td>Accuracy</td></tr><tr><td>TakasuLab</td><td>93.5</td></tr><tr><td>SciNex</td><td>89.3</td></tr><tr><td>Black Socks</td><td>84.5</td></tr><tr><td>LMU &amp; TUM NLP</td><td>83.6</td></tr><tr><td>VerifyAI</td><td>66.7</td></tr><tr><td>Pro²Crast</td><td>66.7</td></tr><tr><td>Denso IT Laboratory</td><td>65.5</td></tr></table>

Table 8: Subtask 2 results (TeX/HTML evidence).
<table><tr><td>Team</td><td>Accuracy</td></tr><tr><td>Bonn-Juelich Informatics</td><td>97.7</td></tr></table>

## 5 Analyses

## 5.1 Accuracy Across Evidence Modifications

Figure 4 shows the accuracy across all modifications applied to tables and figures. We would expect certain modifications to be more dificult to detect than others. For example, swapping rows or columns is likely easier to identify than modifying individual cell values. Interestingly, however, teams struggled with diferent types of modifications and evidence. While some teams performed better on tables, as expected given their more structured nature compared to figures, others showed a performance drop on tables.

Figure 4 also reveals substantial variation in performance across modification types, with no single modification consistently challenging all teams. For example, the table modification others caused VerifyAI and ProvideQ to perform 45% below their respective averages, and CAISALab nearly 60% below its average. In contrast, SciTrue, Bonn-Juelich Informatics, University of Southampton, SIB Text Mining, and SEBIS TUM all achieved 100% accuracy on the same modification. Similarly, graph swaps particularly challenged the top-performing teams, while Takasu Lab and SciNex achieved 100% accuracy. Notably, performance on graph swaps and category swaps likely contributed to SciTrue outperforming Bonn-Juelich Informatics in the overall ranking.

Figures 5 and 6 present the same analysis as Figure 4, but separately for figures and tables, respectively. The results reveal that the top two teams, SciTrue and Bonn-Juelich Informatics, outperformed other teams primarily on tables. While most teams struggled with tables and exhibited large performance variations across modification types, the top two teams remained relatively unafected, with performance spreads of only around 2%. Notably, Bonn-Juelich Informatics made only three mistakes across all tables.

For figures, ProvideQ and VerifyAI achieved the smallest performance spreads of 9.6% and 10.4%, respectively, followed by SciTrue at 11.1%. This diference likely played a key role in determining the winner of this year’s SciClaimEval task, as Bonn-Juelich Informatics exhibited a much larger spread of 22.3%, including a performance drop of 16.6% on graph swaps alone.

Accuracy Across Modification Operations — Subtask 1 (PNG)  
![](images/c2d5c40d6b5b56d526c9d12b6d7aa10e6f6aecddebb6a0d5d81f2f9efa987586.jpg)  
Figure 4: Accuracy on Subtask 1 across evidence modification categories. The red and green percentages indicate performance drops and gains, respectively, compared to the overall average (black triangle marker). The brown and turquoise triangle markers indicate the average accuracy for tables and figures, respectively.

![](images/40568b3dba48e1c37a3c8101383f221b51a660d5140ab6ab9e26f921f97388bd.jpg)  
Figure 5: Accuracy on Subtask 1 across figure modification categories. The red and green percentages indicate performance drops and gains, respectively, compared to the average accuracy across all figures (black triangle marker).

These large performance variations further highlight that no team developed a solution that performed consistently well across all evidence types. Instead, teams demonstrated strong performance on specific types of claim–evidence pairs.

Figure 7 presents the same Cleveland plot as Figure 4, but for Subtask 2 submissions. While the results largely align with those of Subtask 1, some interesting outliers emerge. For example, SciTrue, Bonn-Juelich, and Southampton achieved identical scores on graph swap modifications. In contrast, on Subtask 1, SciTrue performed significantly better, while Southampton performed significantly worse on this modification type.

## 5.2 Accuracy Across Figure Types

Figure 8 shows the accuracy of each team’s best run across figure categories. Note that the categorization was performed independently of the claims. In other words, low performance on a particular chart type, such as a violin plot, does not necessarily indicate that an approach struggles to interpret that type. Instead, it indicates lower performance when that chart type is present in the figure, regardless of whether the claim refers to it. Furthermore, unlike in Figure 3, each chart type was counted only once per figure for evaluation.

Figure 8 reveals a clear and consistent performance gap between the bottom six runs and the top-performing teams. However, no clear correlation between chart types and performance emerges. Stacked bar charts generally lead to lower performance but pose no dificulty for SciTrue. Similarly, heatmaps prove challenging for most teams except Pro<sup>2</sup>Crast.

Accuracy Across Modification Operations — Subtask 2 (PNG)  
Accuracy Across Modification Operations (Tables only) — Subtask 1 (PNG)  
![](images/e8a1fd27cac8dcc0bdfa0f6d81c3a97af7d0f64f96b0c47c092593bc91ed5de9.jpg)  
Figure 6: Accuracy on Subtask 1 across table modification categories. The red and green percentages indicate performance drops and gains, respectively, compared to the average accuracy across all tables (black triangle marker).

![](images/0ea547b9c067ff8002d2d8321e739d7ea765276d21d22893ed30479e105cd218.jpg)  
Figure 7: Accuracy on Subtask 2 across evidence modification categories. The red and green percentages indicate performance drops and gains, respectively, compared to the overall average accuracy (black triangle marker). The brown and turquoise triangle markers indicate the average accuracy for tables and figures, respectively.

Another observation from Figure 8 is that visual noise (additional irrelevant information) appears to have little impact, as single figures outperform multiple subplots and subplots by only 4% and 5% on average, respectively. Although figures containing multiple plots show a substantial average performance drop of up to 9%, this is primarily driven by lower-performing teams. These findings suggest that no single approach performs equally well across all chart types, further supporting the observations in Figures 4 and 7.

## 5.3 Label and Evidence Selection Bias

Figures 9 and 10 show the biases toward selected labels in Subtasks 1 and 2, respectively. In Subtask 1, no clear bias toward either supported or refuted labels emerges across teams and submitted runs. Only a few outliers, particularly several baselines, exhibit a clear bias toward supported cases. CAISALab’s best-performing run also shows a substantial bias toward predicting supported labels.

Surprisingly, almost all teams and submitted runs, except for one run by Pro<sup>2</sup>Crast, favored the first provided evidence in Subtask 2. The reason for this bias remains unclear. One possible explanation is that processing two images may afect the attention mechanisms of MLLMs, causing them to default to the first input. Notably, no team’s best-performing run was able to overcome this bias, despite the wide variety of MLLMs and pipeline approaches employed.

## 5.4 Sample Dificulty and Run Similarity

Figure 11 shows the evaluation results of all submitted runs across all claim–evidence samples. We observe that most teams struggled with similar samples and performed well on the same easier samples, as indicated by the red regions in the bottom-left and gray regions in the top-right of the matrix. However, only a relatively small proportion of samples were incorrectly labeled by most submitted runs. In fact, no single sample was incorrectly labeled by all runs.

![](images/44cb2e7649f22e4c0fc6139bee1b0c26b336aef704e451a47da7e0b75f97a7db.jpg)

Figure 8: Best-run accuracy on Subtask 1 across figure types. Single refers to a single plot, which may combine multiple chart types (e.g., a bar chart with error bars). Multiple refers to multiple unlabeled plots, Subplots refers to multiple labeled plots (e.g., (a), (b)), and Multiple Subplots refers to a combination of labeled and unlabeled plots. Italicized categories (middle part) represent chart types across all figure structures (Single, Multiple, Subplots, and Multiple Subplots). The bottom category, Combination, refers to figures containing two or more chart types, regardless of their structure.  
![](images/34d98d86733bb0db335c852f3dbe2edc0d370663633421323c98661c959c34d4.jpg)  
Figure 9: Label prediction bias across all runs in Subtask 1.

![](images/f9de74529e89befb12bb15aaa4795e79dc8531ff4044ab135b57bf57fb388944.jpg)  
Figure 10: Evidence selection bias across all runs in Subtask 2.

![](images/982efd4166fcd6e3251d459daf00fb0039cb305764dde0af92b5068da613ccbd.jpg)  
Figure 11: Full evaluation results for each submitted run and claim–evidence sample. Red and gray markers indicate incorrect and correct predictions, respectively. Queries are sorted by consensus from hardest (left, most runs predicted incorrectly) to easiest (right, most runs predicted correctly), with figure samples on the left and table samples on the right. The rightmost gray labels show the average accuracy of each run. At the bottom, �(correct) represents the fraction of runs with correct predictions for each sample, while Agreement indicates the agreement among all runs on the predicted label. Majority correct is marked red if more than half of the runs predicted incorrectly and gray otherwise.

This pattern is consistent with what we would expect from a highquality dataset without strong systematic biases.

One notable outlier is the University of Southampton’s run, which does not show an increasing number of incorrect predictions toward the harder samples on the left. In other words, this run performed relatively well on samples that most other teams struggled with. This distinction becomes even more apparent in Figure 12, which presents a pairwise agreement matrix based on Cohen’s �. The University of Southampton is again a clear outlier, showing its third-highest agreement with Denso IT Laboratory despite outperforming it by 11.8% in accuracy (and 10% in pair accuracy). Figure 8 also showed this team outperformed most of the other team’s in its vicinity on the two most challenging figure types stacked bar charts and violin plots. However, it remains unclear why the University of Southampton’s approach exhibits relatively low agreement with other similarly high-performing teams.

## 5.5 Impact of Pair Information on Performance

As earlier pointed out, one of the most common themes to tackle subtask 1 was the exploitation of pair information. This is a clear shortcoming of our task design that we overlooked in favour of a more robust evaluation. However, it is immediately obvious that this feature made the task significantly easier. In fact, we designed subtask 2 specifically to illustrate the diferent challange between subtask 1 and 2. As humans, analyzing an example, such as in Fig ure 2, is also significantly easier with both evidence figures present as it allows us to determine changes and guide our judgement visually first before understanding the claim.

Most teams acknowledged that exploiting pair information leads to unrealistic evaluations and self-reported results without utilizing pairs at least for some earlier iterations of their run submissions. Figure 13 shows these self-reported performance gains. Using this information made the task substantially easier for all teams and, in extreme cases, efectively reduced Subtask 1 to Subtask 2. However, these findings highlight an important conclusion for SciClaimEval: scientific claim verification remains challenging, and the task itself is far from solved.

For example, SciTrue reported that the propriatory frontier models Claude’s Fable 5 achieved only 74.4% pair accuracy and OpenAI’s GPT-5.6 Sol only 70.2% pair accuracy on the development set. In fact, all teams that exploited the pair structure and reported results without using pair information observed substantial drops in pair accuracy, ranging from 8.7% (ProvideQ on the test set) to 25.8% (SciTrue’s evaluation with GPT-5.6 Sol on the development set).

These findings suggest that the high performance achieved by some teams was not necessarily due to stronger verification capabilities, but rather to access to pair information that would typically be unavailable in real-world scenarios. Consequently, these results provide a strong motivation for future iterations of SciClaimEval that remove pair information.

## 6 Conclusion

This paper presented an overview of SciClaimEval, a pilot shared task at NTCIR-19 on cross-modal scientific claim verification using evidence from tables and figures. We introduced a new bench mark dataset covering multiple scientific domains, with contrastive evidence pairs, and summarized the participation and system performance across two subtasks. The results demonstrate the strong potential of MLLMs for this task, particularly when combined with carefully designed inference pipelines and efective use of contrastive evidence pairs. However, the substantial performance gap between systems using contrastive pairs and those evaluating individual claim–evidence pairs highlights the remaining challenges in practical scientific claim verification. We hope that SciClaimEval will serve as a valuable benchmark for advancing research on reliable multimodal scientific claim verification.

![](images/d15777c3e9cb86ec0c594fb51d7b9a39562c5deadf178e5f354ec304a91df258.jpg)  
Figure 12: Similarity (Cohen’s �) between each team’s best-performing run, with runs sorted by accuracy.

![](images/c2c8ccb2f246c0a3a48ea5651ebfadc4c5b7efc517d02679a0b11640365dedfc.jpg)  
Figure 13: Reported performance gains from exploiting pair information. Not all teams reported these gains, and some reported results only on the development (dev) split. SciTrue reported multiple gains depending on the underlying LLM.

## 7 Discussion & Future Directions

Overall, we consider SciClaimEval a success. This initial edition attracted significant community interest and the highest number of participants at NTCIR-19. However, it also revealed shortcomings in the dataset design that allowed teams to exploit information, making the task substantially easier than intended. Although the resulting high scores may raise questions about the need for further scientific claim verification tasks, the self-reported performance drops without pair information demonstrate that real-world scientific claim verification remains challenging.

Most claim–evidence mismatches in SciClaimEval are relatively easy for humans to identify, even without paired evidence, and their prevalence in actual scientific writing remains unclear. In practice, more challenging cases arise in three scenarios. First, verification may require domain expertise, such as interpreting equations, med ical scans, or crystal structures. Second, evidence may be dificult to access or scattered across lengthy documents. For example, verifying claims against cited works often requires retrieving and examining entire papers. Third, claim–evidence relationships are rarely strictly binary and may require more nuanced judgments than our current task allows.

Future iterations of SciClaimEval will address these limitations. First, we will eliminate access to pair information at test time to prevent the shortcuts observed in this edition. Second, we will expand the dataset beyond AI, NLP, and medicine to domains such as materials science and physics, incorporating additional evidence types, including equations, medical scans, and crystal structure visualizations. Third, we will introduce cited papers as evidence sources, addressing the time-consuming and error-prone process of verifying citations, particularly relevant with the rise of AI-assisted scientific writing. Finally, we plan to move beyond binary classification by introducing more nuanced labels or reformulating the task, reducing reliance on shortcuts and enabling a more comprehensive evaluation of scientific reasoning.

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number 24K03231, the French National Research Agency (ANR) through the FAB-ULEUX project (ANR-26-CE23-6589), and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 554559555. F. Boudin was also supported by a visiting professor position at the National Institute of Informatics (NII), Japan.

## References

[1] Deyan Ginev. 2024. ar5iv:04.2024 dataset, an HTML5 conversion of arXiv.org. https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/ SIGMathLing – Special Interest Group on Math Linguistics.

[2] Xanh Ho, Yun-Ang Wu, Sunisth Kumar, Tian Cheng Xia, Florian Boudin, Andre Greiner-Petter, and Akiko Aizawa. 2026. SciClaimEval: Cross-modal Claim Verification in Scientific Papers. In Proceedings ofthe Fifteenth Language Resources and Evaluation Conference (LREC 2026), Stelios Piperidis, Núria Bel, Henk van den Heuvel, Nancy Ide, Simon Krek, and Antonio Toral (Eds.). European Language Resources Association (ELRA), Palma, Mallorca, Spain, 11060–11071. doi:10. 63317/4ap9rg2gnwmf

[3] Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. 2020. HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification. In Findings of the Association for Computational Linguistics: EMNLP 2020, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 3441–3460. doi:10.18653/v1/2020.findings-emnlp.309

[4] Yash Kumar Lal, Manikanta Bandham, Mohammad Saqib Hasan, Apoorva Kashi, Mahnaz Koupaee, and Niranjan Balasubramanian. 2025. MuSciClaims: Multi modal Scientific Claim Verification. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, Kentaro Inui, Sakriani Sakti, Haofen Wang, Derek F. Wong, Pushpak Bhattacharyya, Biplab Banerjee, Asif Ekbal, Tanmoy Chakraborty, and Dhirendra Pratap Singh (Eds.). The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, Mumbai, India, 3285–3307. doi:10.18653/v1/2025. ijcnlp-long.175

[5] Xinyuan Lu, Liangming Pan, Qian Liu, Preslav Nakov, and Min-Yen Kan. 2023. SCITAB: A Challenging Benchmark for Compositional Reasoning and Claim Verification on Scientific Tables. In Proceedings of the 2023 Conference on Empirical

Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 7787–7813. doi:10.18653/v1/2023.emnlp-main.483

[6] Sebastian Schellhammer, Salim Hafid, Yavuz Selim Kartal, Katarina Boland, Dimitar Dimitrov, Konstantin Todorov, and Stefan Dietze. 2026. Overview of the CLEF-2026 CheckThat! lab task 1 on source retrieval for scientific web claims. Salido et al.[53] (2026).

[7] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a Large-scale Dataset for Fact Extraction and VERification. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), Marilyn Walker, Heng Ji, and Amanda Stent (Eds.). Association for Computational Linguistics, New Orleans, Louisiana, 809–819. doi:10.18653/v1/ N18-1074

[8] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or Fiction: Verifying Scientific Claims. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 7534–7550. doi:10.18653/v1/2020.emnlp-main.609

[9] David Wadden and Kyle Lo. 2021. Overview and Insights from the SCIVER shared task on Scientific Claim Verification. In Proceedings ofthe Second Workshop on Scholarly Document Processing, Iz Beltagy, Arman Cohan, Guy Feigenblat, Dayne Freitag, Tirthankar Ghosal, Keith Hall, Drahomira Herrmannova, Petr Knoth, Kyle Lo, Philipp Mayr, Robert M. Patton, Michal Shmueli-Scheuer, Anita de Waard, Kuansan Wang, and Lucy Lu Wang (Eds.). Association for Computa tional Linguistics, Online, 124–129. https://aclanthology.org/2021.sdp-1.16

[10] Chengye Wang, Yifei Shen, Zexi Kuang, Arman Cohan, and Yilun Zhao. 2025. SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification. In Proceedings ofthe 63rd Annual Meeting ofthe AssociationforComputational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 8562–8579. doi:10.18653/v1/2025.acl-long.420

[11] Nancy X. R. Wang, Diwakar Mahajan, Marina Danilevsky, and Sara Rosenthal. 2021. SemEval-2021 Task 9: Fact Verification and Evidence Finding for Tabular Data in Scientific Documents (SEM-TAB-FACTS). In Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021), Alexis Palmer, Nathan Schneider, Natalie Schluter, Guy Emerson, Aurelie Herbelot, and Xiaodan Zhu (Eds.). Association for Computational Linguistics, Online, 317–326. doi:10. 18653/v1/2021.semeval-1.39

[12] Bohao Yang, Yingji Zhang, Dong Liu, André Freitas, and Chenghua Lin. 2025. Does Table Source Matter? Benchmarking and Improving Multimodal Scientific Table Understanding and Reasoning. arXiv:2501.13042 [cs.CL] https://arxiv.org/ abs/2501.13042

[13] Yuji Zhang, Qingyun Wang, Cheng Qian, Jiateng Liu, Chenkai Sun, Denghui Zhang, Tarek Abdelzaher, Chengxiang Zhai, Preslav Nakov, and Heng Ji. 2025. Atomic Reasoning for Scientific Table Claim Verification. arXiv:2506.06972 [cs.CL] https://arxiv.org/abs/2506.06972