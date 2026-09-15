# What Limits Us? Analyzing Self-Reported Limitations in NLP Research

Tawan Thaeprasit<sup>†</sup>, Peeranuth Kehasukcharoen<sup>†</sup>, Ding Wang<sup>‡</sup>,

Remi Denton<sup>‡</sup>, Peerapon Vateekul<sup>†∗</sup>, Piyawat Lertvittayakumjorn<sup>‡</sup>

<sup>†</sup>Department of Computer Engineering, Faculty of Engineering,

Chulalongkorn University, Thailand

<sup>‡</sup>Google Research

tawanth.official@gmail.com, 6030416021@alumni.chula.ac.th peerapon.v@chula.ac.th, {drdw,dentone,piyawat}@google.com

## Abstract

Since late 2022, a Limitations section has become mandatory at many top-tier NLP conferences. The growing number of accepted papers at these venues has resulted in a vast corpus of self-reported limitations that cannot all be manually reviewed, yet remains systematically unanalyzed. Therefore, in this paper, we conduct a large-scale analysis of the Limitations sections from ACL and EMNLP papers published between 2020 and 2025 to understand what researchers disclose about their own work. To do so, we implement a novel human-AI framework for iterative hybrid qualitative coding. This framework enables us to investigate trends in self-reported limitations over time, their correlations with specific paper attributes, and the writing patterns that recur around these disclosures. Our findings offer a critical reflection on the diverse reported challenges as well as the self-reporting practices of researchers in the NLP community.

## 1 Introduction

Reporting research limitations is fundamental to scientific transparency, as it declares factors that might undermine the validity of research claims and provides suggestions for readers who want to use or build upon the work (Olteanu et al., 2025). In the NLP community, the Limitations section was optional until EMNLP 2022 and ACL 2023 made it mandatory for all accepted papers. This policy was subsequently adopted by ACL Rolling Review (ARR) in December 2023<sup>1</sup>, ensuring that recent papers across all main \*ACL conferences include a self-reported Limitations section. Consequently, thousands of these sections are now scattered across papers hosted on the ACL Anthology<sup>2</sup>. This data availability has enabled several studies involving the Limitations sections of NLP research (Al Azher et al., 2025a; Faizullah et al., 2024).

Focusing on limitation analysis, Zhou et al. (2025) manually examined the Limitations sections of 57 cultural NLP papers from 2022 to 2024 to understand recurring challenges in the subfield. Meanwhile, Al Azher et al. (2025b) applied topic modeling to papers from ACL 2023 and its workshops to discover common limitation topics. However, given the massive volume of papers nowadays, manual analysis is infeasible at scale, while topic modeling tends to bias toward dominant topics and overlook minor yet significant ones, especially newly emerging topics. Furthermore, none of the existing work has shed light on how selfreported limitations evolve given the field’s rapid progress and the publication policy change.

In this paper, we develop a human-AI framework to conduct large-scale hybrid (deductive and inductive) qualitative coding (Fereday and Muir-Cochrane, 2006) on the Limitations sections of ACL and EMNLP papers published between 2020 and 2025. Within this framework, a Large Language Model (LLM) is used to apply existing codes to the text, suggest new codes when needed, and recommend codebook modifications based on emerging new code clusters. Still, humans are responsible for initial codebook development and actual codebook modifications to ensure interpretability of the output. Following the full-scale annotation, the resulting qualitative codes were aggregated into code distributions across the dataset, providing empirical evidence to answer our three core research questions:

• RQ1: Temporal Analysis – How have the contents of self-reported limitations evolved over time? (Section 4.2)

• RQ2: Correlations with Paper Attributes – How do specific limitation codes correlate with certain paper attributes (i.e., research areas, paper formats, and author affiliations)? (Section 4.3)

• RQ3: Discursive Patterns – Are there recurring textual patterns in self-reported limitations that could suggest potential reporting strategies? (Section 4.4)

Overall, our contribution is threefold. First, we propose a novel human-AI framework for iterative hybrid qualitative coding, enabling scalable and rigorous analysis of research limitations. Second, we release a large-scale dataset of 16,067 Limitations sections extracted from ACL and EMNLP papers (2020–2025). This release includes LLMgenerated annotations and a high-quality humanannotated set for evaluation.<sup>3</sup> Third, we present empirical findings for our three research questions, providing deep insights into trends and practices of self-reported limitations in NLP research.

## 2 Background and Related Work

Analyzing Limitations Sections. Due to the rich insights contained in Limitations sections, researchers across various disciplines have analyzed these sections to understand common challenges within their respective fields (Rodriguez et al., 2024; Hsu et al., 2024; Alvarez et al., 2021; Stöckli et al., 2023; Sanders et al., 2023; Theofanidis and Fountouki, 2018; Brutus et al., 2013). Most of these studies were conducted manually or through keyword analysis. Because reporting limitations was not standard practice in NLP until late 2022, literature analyzing these sections within the NLP community remains relatively scarce. Nonetheless, following recent conference policy changes, emerging work has begun focusing on limitation extraction and generation (Al Azher et al., 2025a; Faizullah et al., 2024) as well as content analysis (Zhou et al., 2025). Among existing studies, our work is closest to Al Azher et al. (2025b), who applied topic modeling to Limitations sections and used LLM-based summarization to produce topic summaries. However, the inherent nature of topic modeling suppresses the detection of minor, emerging topics, which are crucial for temporal analysis. Our work adopts a well-established approach in qualitative research to overcome this limitation.

Coding for Qualitative Analysis. In qualitative analysis, coding is the process of assigning concise, descriptive labels to specific segments of text so as to understand recurring concepts in textual data (Saldaña, 2021). Researchers generally approach coding through two distinct lenses. First, deductive coding relies on a predefined list of codes and their definitions (compiled within a codebook) to analyze the text. Conversely, with inductive coding, researchers construct codes bottom-up based on their interpretations of the data. As a middle ground, a hybrid approach starts with an initial codebook but remains open to generating new, inductive codes for text segments that do not fit any existing codes (Fereday and Muir-Cochrane, 2006). In this study, we adopt the hybrid approach to analyze Limitations sections, leveraging an existing taxonomy of limitation types (Xu et al., 2025) while leaving room for novel patterns to emerge from the data.

To analyze large textual datasets, researchers have increasingly explored AI-assisted tools for qualitative analysis. Early work primarily utilized topic modeling techniques, such as Latent Dirichlet Allocation (Blei et al., 2003) and BERTopic (Grootendorst, 2022). However, these methods often provide limited support for interpreting the underlying topics. More recently, studies have investigated the use of LLMs to facilitate coding, spanning both deductive (Xiao et al., 2023; Chew et al., 2023) and inductive (data-driven) (Dai et al., 2023; Parfenova et al., 2025; Zhong et al., 2025; Kostikova et al., 2026) approaches. Closest to our work, Wiebe et al. (2025) proposed using an LLM with humans in the loop for hybrid coding and thematic analysis. However, a key distinction is that our approach allows for iterative updates to the codebook, which are essential for temporal analysis. Additionally, our framework is designed in a modular fashion, enabling researchers to easily extend or modify individual modules to suit their specific needs.

## 3 Methodology

This section discusses the scope of our study, the content extraction process, and our human-AI framework for hybrid qualitative coding.

## 3.1 Scope of the Study

We scoped our analysis to ACL and EMNLP papers (Long, Short, and Findings) published between 2020 and 2025. We focused exclusively on selfreported limitations within dedicated Limitations sections. Consequently, any limitations discussed in other parts of the text, or omitted from the papers, were not included in this study.

## 3.2 Content Extraction

To prepare the textual data for analysis, we extracted the Limitations sections along with relevant paper attributes needed to answer RQ2.

Limitations Sections. We extracted the limitations from two sources. For papers prior to EMNLP 2022, their raw texts are provided by the ACL-OCL dataset (Rohatgi et al., 2023) from which we collected the Limitations sections. For the remaining papers, we developed an extraction pipeline using Docling<sup>4</sup> to parse the PDF files from ACL Anthology, followed by a regular expression module to extract the target sections. To validate our pipeline, we cross-checked its output against the ACL-OCL data for ACL 2022 (Short) and found our tool produced consistent results.

Paper Attributes. The metadata extracted from each paper consists of research areas and author affiliations. To identify research areas (e.g., Information Extraction, Machine Translation, NLP Applications, etc.), we used information from conference programs available online for ACL 2022 and EMNLP 2023 to 2025. For affiliations, we extracted them directly from the raw PDF file provided on the ACL Anthology using a custom extraction pipeline. Further details are provided in Appendix K.1.

## 3.3 Iterative Hybrid Qualitative Coding

Our hybrid coding framework begins with the construction of an initial codebook, followed by an iterative coding process. In each iteration, we code the Limitations sections from specific years and update the codebook to incorporate newly found codes. In this study, we conducted four iterations: the first combined 2020–2022 data (grouped due to a lower volume of Limitations sections), followed by separate iterations for 2023, 2024, and 2025.

Initial Codebook Construction. A codebook is a structured set of categories and definitions used to classify text. We built our initial codebook by combining data-driven discovery with an established taxonomy. First, we conducted a pilot run by using an LLM to inductively code 50 Limitations sections from ACL 2024 extracted by BAGELS (Al Azher et al., 2025a). After that, we manually aligned these empirical codes with the taxonomy of limitations in AI research proposed by Xu et al. (2025) and expanded the taxonomy to accommodate novel topics observed in the pilot data. To support the analysis of discursive patterns in RQ3, we also created codes for non-limitation content such as future work, method strengths, and conducted mitigation, as seen in the pilot data. The resulting codebook, consisting of 18 limitation and 7 non-limitation codes, served as our initial codebook $\mathcal { C } _ { 0 }$ for the iterative coding process, as displayed in Figure 1.

Automatic Hybrid Coding. For each iteration, we used LLMs to code the Limitations section of each paper. Given a Limitations section $L _ { i }$ of a paper i, we first segmented the section into a sequence of semantic units $S _ { i }$ . Formally,

$$
S _ { i } = \langle s _ { i 1 } , . . . , s _ { i n } \rangle = M _ { \mathrm { s e g } } ( L _ { i } )
$$

where $s _ { i j }$ is a semantic unit in $L _ { i }$ and $M _ { \mathrm { s e g } }$ is a segmenter model, which is, in our case, Gemini 2.5 Flash (Comanici et al., 2025) running a segmenter prompt. Next, we formatted these segmented units with XML tags and grouped them into batches for coding. A batch is the set of semantic units passed to $M _ { \mathrm { c o d e } }$ in a single call, adopted for efficient inference. We used Gemini 3.1 Pro<sup>5</sup> as $M _ { \mathrm { c o d } \epsilon }$ <sub>e</sub> where our coding prompt (in Appendix K.4) employed two prompt engineering strategies. First, we provided few-shot examples (E) of semantic units coded by the authors for in-context learning. Second, we applied Chain-of-Thought (CoT) prompting, requiring the model to articulate its reasoning before classification. So, for iteration t,

$$
\mathcal { T } _ { t , b } = M _ { \mathrm { c o d e } } ( \boldsymbol { B } _ { t , b } , \mathcal { C } _ { t - 1 } , \mathcal { E } )
$$

where $B _ { t , b }$ is the b-th batch of semantic units in iteration t, $\mathcal { C } _ { t - 1 }$ is a codebook from the previous iteration, E is a set of few-shot examples, and $\mathcal { T } _ { t , b }$ is the resulting set of coded units in this batch b. Each member in $\mathcal { T } _ { t , b }$ is a tuple of semantic unit and an assigned code where one semantic unit can appear multiple times in $\mathcal { T } _ { t , b }$ . At the end, we obtained $\mathcal { T } _ { t } = \bigcup _ { b } \mathcal { T } _ { t , b }$ as the set of coded units from all the batches in iteration t.

To support hybrid coding, the prompt of $M _ { \mathrm { c o d e } }$ allowed the model to assign new codes that are not in $\mathcal { C } _ { t - 1 }$ to specific semantic units if necessary.

![](images/45c6dfb4d5c5636eb236da78b63d02e86483799ae27a01d94e6b1ccb42f1102d.jpg)  
Figure 1: An overview of our iterative hybrid coding framework. The human icons illustrate where humans perform tasks in this framework. “Defn.” stands for the definition of the code. More details can be found in Section 3.3.

Hence, $\mathcal { T } _ { t }$ can be partitioned into $\mathcal { T } _ { t } ^ { E }$ and $\mathcal { T } _ { t } ^ { N }$ , representing the sets of existing-code assignments and new-code assignments, respectively. Next, we explain how we handled the new codes in $\mathcal { T } _ { t } ^ { N }$

Handling New Codes. Practically, the same new code could be phrased differently by $M _ { \mathrm { c o d e } }$ in different batches. Also, some of the new codes could be noises or were not different enough from an existing code. We therefore refrained from immediately assigning new codes in $\mathcal { T } _ { t } ^ { N }$ or updating the codebook without human review. However, given the large volume of papers and new codes in each iteration, manually reviewing every single code assignment in $\mathcal { T } _ { t } ^ { N }$ was unfeasible. So, we handled $\bar { \mathcal { T } } _ { t } ^ { N }$ in four steps. First, an LLM consolidator $( M _ { \mathrm { c o n s o l i d a t e } } )$ grouped semantically similar new codes into clusters $m _ { i }$ with suggested cluster names and definitions.

$$
\mathcal { M } _ { t } = \{ m _ { 1 } , \dots , m _ { k } \} = M _ { \mathrm { c o n s o l i d a t e } } ( \mathcal { T } _ { t } ^ { N } )
$$

Second, a recommendation module $( M _ { \mathrm { r e c } } )$ evaluated $\mathcal { M } _ { t }$ against the current codebook $\mathcal { C } _ { t - 1 }$ and suggested an action $a _ { i }$ (which could be ADD, MERGE, EXPAND, or REJECT) for each $m _ { i }$

$$
\mathcal { A } _ { t } = \{ a _ { 1 } , \dots , a _ { k } \} = M _ { \mathrm { r e c } } ( \mathcal { M } _ { t } , \mathcal { C } _ { t - 1 } )
$$

The explanations of each possible action can be found in Table 1. Note that we used Gemini 3.1 Pro for both $M _ { \mathrm { { c o n s o l i d a t e } } }$ and $M _ { \mathrm { r e c } }$ . Third, as the human-in-the-loop researchers, we reviewed $\boldsymbol { A } _ { t }$ to confirm or modify each action $a _ { i }$ as appropriate, resulting in the set of human-verified actions $\mathbf { \mathcal { A } } _ { t } ^ { \prime }$ Finally, we applied $\mathcal { A } _ { t } ^ { \prime }$ to update the codebook to $\mathcal { C } _ { t }$ and adjusted the code assignments in $\mathcal { T } _ { t } ^ { N }$ according to Table 1. This concludes an iteration of hybrid coding. We then began the next iteration of automatic coding with the updated codebook $\mathcal { C } _ { t }$

<table><tr><td>Action</td><td>Codebook Update</td><td> $\mathcal { T } _ { t } ^ { N }$  Update</td></tr><tr><td>ADD</td><td>cluster</td><td>Add a new code c to Change every new the codebook for this code in this cluster to c</td></tr><tr><td>EXPAND</td><td>to cover this cluster</td><td>Expand the definition Change every new of an existing code c code in this cluster to c</td></tr><tr><td>MERGE</td><td>Use an existing code Change every new this cluster (without definition update)</td><td>or a new code c for code in this cluster to c</td></tr><tr><td>REJECT</td><td>Drop this cluster</td><td>Remove every new code in this cluster</td></tr></table>

Table 1: Possible actions for each new code cluster and the corresponding codebook updates and new-code assignment $\mathcal { T } _ { t } ^ { \hat { N } }$ updates.

## 3.4 Evaluation of the Hybrid Coding Step

Because the semantic unit segmentation step (by $M _ { \mathrm { s e g } } )$ is straightforward and the process for handling new codes (using $M _ { \mathrm { c o n s o l i d a t e } }$ and $M _ { \mathrm { r e c } } )$ is reviewed by humans, we focus our framework evaluation on the hybrid coding step performed by $M _ { \mathrm { c o d e } }$

Dataset. We sampled the Limitations sections from 150 NLP papers, 30 of which were labeled by three authors of this paper to calibrate our understanding of the codebook, while the remaining 120 were labeled by two authors. Each annotator independently used the initial codebook $\mathcal { C } _ { 0 }$ for coding and proposed new labels as necessary. The inter-annotator agreement (Krippendorff’s α) of deductive code assignments was 0.644. Disagreements in both deductive and inductive coding were discussed until a consensus was reached among all annotators assigned to that paper.

<table><tr><td></td><td colspan="4">Deductive</td><td colspan="3">Inductive - Quality</td><td colspan="3">Inductive - Sensitivity</td></tr><tr><td>Setting</td><td>Prec. ↑</td><td>Recall ↑</td><td>F1↑</td><td>α</td><td>Correct ↑</td><td>Partial</td><td>Wrong↓</td><td>New↑</td><td>Existing</td><td>Missed ↓</td></tr><tr><td>Xiao et al. (2023)</td><td>0.744</td><td>0.792</td><td>0.751</td><td>0.627</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Ours (Defn. only)</td><td>0.757</td><td>0.760</td><td>0.738</td><td>0.623</td><td>87.5%</td><td>0.0%</td><td>12.5%</td><td>1.4%</td><td>78.9%</td><td>19.7%</td></tr><tr><td>Ours (Few-Shot only)</td><td>0.786</td><td>0.808</td><td>0.776</td><td>0.635</td><td>82.0%</td><td>11.5%</td><td>6.6%</td><td>32.4%</td><td>56.3%</td><td>11.3%</td></tr><tr><td>Ours (Defn. + Few-Shot)</td><td>0.827</td><td>0.811</td><td>0.801</td><td>0.647</td><td>91.5%</td><td>3.4%</td><td>5.1%</td><td>60.6%</td><td>35.2%</td><td>4.2%</td></tr></table>

Table 2: Performance of different prompting approaches. “Prec.” and “Defn.” stand for precision and code definitions, respectively. α represents the inter-annotator agreement (Krippendorff’s α) after treating the LLM as another annotator. Note that the baseline α from human annotators only is 0.644. N/A means the method cannot produce inductive codes by design. Bold numbers highlight key metrics we used for method selection.

Metrics. We divided the evaluation into two parts. For the deductive part, which resembles a multilabel classification task, we computed precision, recall, and F1 scores against the human consensus codes. Additionally, we treated the LLM as a supplementary annotator and recomputed Krippendorff’s α to observe how LLM-generated labels impacted the baseline inter-annotator agreement. For the inductive part, we manually assessed the quality of each new code from the LLM, classifying whether it was acceptable, required a minor label adjustment, or was a false positive. Furthermore, to evaluate the LLM’s sensitivity to new codes, we checked whether the LLM detected each new code from human consensus as a new code, assigned an acceptable existing code, or missed it completely.

Comparison. We compared three variants of our $M _ { \mathrm { c o d e } }$ , which differ in how they describe the codes within the codebook. These variants evaluate the use of: (1) code definitions only, (2) few-shot examples only, and (3) both code definitions and fewshot examples in the prompt. Furthermore, we compared our approaches against a baseline from Xiao et al. (2023), which utilizes a zero-shot deductive coding prompt that includes the code definitions only. All tested approaches were executed using the same underlying model, i.e., Gemini 3.1 Pro.

Results. As Table 2 shows, the deductive baseline (Xiao et al., 2023) achieved a competitive F1-score (0.751) for existing codes; however, it failed to generate meaningful new codes. This highlights a fundamental limitation of strictly deductive prompts. Meanwhile, our approach with only code definitions slightly degraded the deductive metrics compared to the baseline (F1 = 0.738). In contrast, providing only few-shot examples improved the deductive (F1 = 0.776). Ultimately, our combined “Defn. + Few-Shot” setting achieved the highest classification performance (F1 = 0.801). Moreover, while the other settings decreased the inter-annotator agreement (α), the combined “Defn. + Few-Shot” slightly improved α from the baseline of 0.644 to 0.647. This shows that it acted as a valid automated coder, diverging from humans only as much as the humans disagreed with one another.

Regarding inductive coding, we can see from Table 2 that the “Defn. + Few-Shot” setting also achieved the highest metrics for both the new code quality (91.5%) and sensitivity (60.6%). Although the latter reveals some room for improvement, the method rarely ignored novel evidence completely (4.2%) but tends to conservatively mapped the evidence to acceptable existing codes (35.2%), making it analytically safe for our large-scale analysis. Hence, we used this combined configuration in our full-scale run.

## 4 Results

In total, we analyzed Limitations sections from 16,067 papers (7,052 ACL and 9,015 EMNLP papers) across four iterations, resulting in the final codebook presented in Appendix I. In this section, we discuss the rate of limitation reporting alongside our findings for the three RQs.

## 4.1 Prevalence of Limitations Sections

To understand the impact of mandatory reporting policy, we analyzed the presence of explicit Limitations sections across ACL and EMNLP from 2020 to 2025. Despite mandates being in place since EMNLP 2022, our extraction pipeline flagged 136 papers (0.85%) accepted between EMNLP 2022 and EMNLP 2025 as lacking a required Limitations section. To verify these, we manually inspected the 136 papers and reported the results in Table 3. Among the 136 papers, we found that 41 papers (0.26%) were flagged due to extraction pipeline failures, whereas 95 papers (0.60%) genuinely missed a dedicated Limitations section. Further analysis of these 95 papers reveals that 50 discussed their limitations within paragraphs embedded in other sections rather than in a required standalone section before the References. For the remaining 45 papers, we could not find any discussion paragraph, named Limitations, anywhere in the text. We believe these numbers are interesting information for future NLP conference organizers.

<table><tr><td>Venue</td><td>Total Papers</td><td>(1) Implicitly Report*</td><td>(2) Truly Missing *</td><td>(3) Pipeline Failure</td></tr><tr><td>EMNLP 2022</td><td>1,376</td><td>20</td><td>10</td><td>0</td></tr><tr><td>ACL 2023</td><td>1,976</td><td>13</td><td>9</td><td>13</td></tr><tr><td>EMNLP 2023</td><td>2,106</td><td>5</td><td>11</td><td>3</td></tr><tr><td>ACL 2024</td><td>1,915</td><td>4</td><td>7</td><td>6</td></tr><tr><td>EMNLP 2024</td><td>2,271</td><td>2</td><td>1</td><td>4</td></tr><tr><td>ACL 2025</td><td>3,086</td><td>6</td><td>7</td><td>7</td></tr><tr><td>EMNLP 2025</td><td>3,214</td><td>0</td><td>0</td><td>8</td></tr><tr><td>Total</td><td>15,944</td><td>50</td><td>45</td><td>41</td></tr></table>

Table 3: Number of accepted papers across venues post-mandate (2022–2025), along with a breakdown of papers flagged by our extraction pipeline as missing a required Limitations section. It distinguishes (1) papers implicitly reporting limitations in other sections, (2) papers truly omitting limitations, and (3) extraction pipeline failures. The asterisks (<sup>∗</sup>) mark noncompliance with the mandatory reporting policy.

![](images/c2daba182d13c9f46ffeae8e607c5ef43c54e8f45abefd107e4c42739dc4c723.jpg)  
Figure 2: The presence of explicit Limitations sections in ACL and EMNLP papers (2020–2025). Papers lacking a Limitations section from EMNLP 2022 onwards have been manually verified.

After the manual verification above, Figure 2 plots the prevalence of the dedicated Limitations section over time. The trend shows that the policy mandates drove a dramatic shift in reporting practices. Prior to the mandates, voluntary inclusion was rare (less than 10%). Afterward, compliance surged to 97.8% at EMNLP 2022 and 98.9% at ACL 2023, reaching perfect compliance (100%) by EMNLP 2025. This reflects how conference review processes gradually strengthened over time.

![](images/0979f7f3b3e87ded1396dddb740fb1cf06f556d1d8dd34279d51f15c497e4b27.jpg)  
Figure 3: Trends of the top 5 limitation codes.

## 4.2 RQ1: Temporal Analysis

This section reports how the trends of reported limitations change over the six years we studied.

Limitation Topics. Figure 3 illustrates the evolving nature of self-reported limitations from 2020 to 2025. A striking divergence occurs between the two most prevalent codes: while Methodological Constraints experienced a sharp decline after peaking in 2021, Scope Limitation exhibits a massive upward trajectory, surging from a prevalence of ∼30% in 2020 to over 65% by 2025. Notably, the rankings of these two codes swapped almost precisely at EMNLP 2022, when the mandatory policy was first introduced. This intersection raises the compelling question of whether the policy mandate was the primary driver of this divergence or it resulted from other confounding factors. A more controlled analysis is needed to answer this question.

Driven by the increase in Scope Limitation, we performed a second-level analysis on this code to investigate the underlying topics. Specifically, we employed LLM-driven clustering to construct a fine-grained sub-codebook, subsequently re-annotating the semantic units with Scope Limitation to assign these sub-codes. As depicted in Figure 4, Model Scale became the fastest-growing sub-code. This is typically characterized by researchers bounding their evaluated parameter sizes, frequently noting that they “limited our study to [Model X], and our findings may not generalize to larger models.”. This reflects the recent paradigm shift brought by the advent of large language models. In addition, we observed a sharp increase in the Language Coverage sub-code during 2020-2022, likely stemming from growing awareness of the field’s English-centric bias and recent calls to explicitly name studied languages (Bender, 2019).

![](images/47e825da94fb9f4519f81cbb5d4fc145af77f1e72b7ccde4582f7690470b0fe8.jpg)

Figure 4: Trends of Scope Limitation sub-codes.
<table><tr><td rowspan="2">Year</td><td colspan="3">ACL</td><td colspan="3">EMNLP</td></tr><tr><td>Mean</td><td>Med.</td><td>#Codes</td><td>Mean</td><td>Med.</td><td>#Codes</td></tr><tr><td>2020</td><td>156.42</td><td>142.00</td><td>1.63</td><td>201.41</td><td>152.50</td><td>1.90</td></tr><tr><td>2021</td><td>204.47</td><td>191.00</td><td>2.44</td><td>180.04</td><td>150.50</td><td>1.70</td></tr><tr><td>2022</td><td>220.03</td><td>171.50</td><td>2.47</td><td>167.97</td><td>138.00</td><td>2.49</td></tr><tr><td>2023</td><td>173.24</td><td>138.00</td><td>2.89</td><td>177.06</td><td>146.00</td><td>2.89</td></tr><tr><td>2024</td><td>168.90</td><td>139.00</td><td>2.91</td><td>182.73</td><td>150.00</td><td>2.93</td></tr><tr><td>2025</td><td>173.77</td><td>139.00</td><td>2.89</td><td>175.77</td><td>139.00</td><td>3.02</td></tr></table>

Table 4: Mean and median word counts alongside the average number of unique limitation codes (#Codes) per Limitations section in each conference.

Word Counts and Code Diversity. As reported in Table 4, the median length of Limitations sections dropped after the introduction of the mandatory policy (EMNLP 2022) and then stabilized at around 138–150 words. However, the average number of unique limitation codes per paper grew from 1.63 in ACL 2020 to 3.02 in EMNLP 2025. This reveals that authors are increasingly adopting a concise, multifaceted approach, i.e., packing more diverse limitation types into roughly the same space rather than elaborately describing a few types.

Iterative Codebook Updates. Across the four iterations, the codebook was updated by 27 EX-PAND actions and 25 ADD actions. Out of the 25 new codes, 22 were added after the first iteration due to the limited coverage of our initial codebook. Besides that, a new code, namely Literature Coverage Gap, was introduced in 2024 to capture authors discussing their inability to review all relevant literature, stating for instance: “However, it is possible that some other relevant works were overlooked...”. Following this, two new codes were added in 2025: Dataset Task Mismatch and Sparse Data Sensitivity.

<table><tr><td>Research Area × Limitation Code</td><td>O/E</td><td>Adj. p</td></tr><tr><td>Human-Centered NLP</td><td></td><td></td></tr><tr><td>Societal and Ethical Risks</td><td>5.80</td><td>&lt;0.001</td></tr><tr><td>Subjectivity in Evaluation/Annotation</td><td>3.83</td><td>&lt;0.001</td></tr><tr><td>Efficient NLP</td><td>3.53</td><td>&lt;0.001</td></tr><tr><td>Hyperparameter Sensitivity</td><td></td><td></td></tr><tr><td>Interpretability Lack of Interpretability</td><td>3.31</td><td>&lt;0.001</td></tr><tr><td>Theoretical Gap</td><td>2.77</td><td>&lt;0.001</td></tr><tr><td>Summarization</td><td></td><td></td></tr><tr><td>Reliance on Automatic Metrics</td><td>3.29</td><td>0.002</td></tr><tr><td>Language Modeling</td><td></td><td></td></tr><tr><td>Reproducibility Gap Computational Social Science</td><td>2.91</td><td>0.036</td></tr><tr><td>Temporal Degradation</td><td>2.90</td><td>&lt;0.001</td></tr><tr><td>Machine Translation Reliance on Automatic Metrics</td><td>2.82</td><td></td></tr><tr><td></td><td></td><td>0.006</td></tr><tr><td>Ethics, Bias, and Fairness Privacy and Security Risks</td><td>2.78</td><td>0.005</td></tr></table>

Table 5: Top 10 significant (research area, limitation code) pairs with the highest observed-to-expected ratio (O/E). Adj. p is the p-value of the residual analysis after Bonferroni correction.

At the end, the final version of our codebook has 40 limitation codes and 10 non-limitation codes. Details of the codebook evolution are provided in Appendix J.

## 4.3 RQ2: Correlations with Paper Attributes

We cross-analyzed the limitation codes with three paper attributes: research area, paper format, and author affiliation.

Research Area. To investigate whether reported limitations vary by research area, we analyzed 6,742 papers where research areas could be gathered from online conference programs. Specifically, we conducted a chi-square test of homogeneity on a 29 × 40 contingency table (research areas × unique limitation codes). The test rejected the null hypothesis of equal distributions across areas $( \chi ^ { 2 } ( 1 , 0 9 2 ) = 2 , 8 7 2 , p < 0 . 0 0 1$ , Cramér’s V = 0.071), indicating a medium effect size under Cohen’s df-adjusted benchmarks. This means certain research areas reported specific limitations at rates significantly different from the field average.

To uncover such prominent areas and limitations, we ran a post-hoc analysis using standardized residuals. The results reveal that 32 out of 1,160 cells in the contingency table had significant residuals after the Bonferroni correction (Dunn, 1961). Table 5 lists the top 10 significant area-limitation pairs with the highest observed-to-expected ratio. For example, Human-Centered NLP papers over-reported Societal and Ethical Risks, whereas Efficient NLP papers over-reported Hyperparameter Sensitivity. These pairs reflect the distinct priorities or concerns of each subfield. However, the 32 significant cells account for only 2.76% of the contingency table. Thus, roughly 97% of (research area, limitation code) pairs do not differ significantly from the field average, demonstrating a field-wide homogeneity in how limitations are reported. These findings motivate future work to investigate the underlying causes of this uniformity and to find out whether similar patterns exist in adjacent fields such as HCI or general machine learning.

Paper Format. We further investigated whether the publication format (Long, Short, or Findings) influences limitation reporting. Here, we focused on 7,033 ACL 2021 to ACL 2025 papers, where the format can be easily extracted from Anthology IDs (e.g., 2025.acl-long.1). Testing all limitation codes for independence from paper format, we found no differences that remain significant after Benjamini–Hochberg correction (Benjamini and Hochberg, 1995). However, a few codes show differences substantial enough to report as exploratory observations. Short papers reported empirical limitation types more often than long and Findings papers. For example, 12.2% of short papers reported Empirical Underperformance, whereas only 7.5% of both long and Findings papers did so (Adjusted $p = 0 . 2 6 6 )$ . Similarly, Hyperparameter Sensitivity was found more frequently in short papers (4.5%) than in long (2.5%) and Findings (2.9%) papers (Adjusted p = 0.503). These are consistent with the inherent nature of short papers, which often present targeted, smaller-scale experiments, negative results, or preliminary findings. The full code distribution and per-code test results are provided in Appendix E.

Author Affiliation. We investigated whether research from large companies reports similar types of limitations as that from other institutions. Using the list of large companies from the Forbes Global 2000 (2025) dataset<sup>6</sup>, we categorized papers into three groups: (1) papers with large company affiliations only, (2) papers with mixed affiliations, and (3) papers without large company affiliations. Because the influence of large companies on papers with mixed affiliations varies across individual papers, we focused on comparisons between papers with only large company affiliations and papers without large company affiliations. Specifically, to compare the two groups, we conducted two-sided Fisher’s exact tests for each limitation code and adjusted the resulting p-values using the Benjamini–Hochberg correction.

![](images/6ab0ff5166a2b31e22eac9ada4c32d61c85e7defd2a842d8f0f115b87a943b5b.jpg)  
Figure 5: Top 5 limitation codes with the highest percentage difference between the Large company and Nonlarge company affiliation groups, ordered from highest to lowest difference. Bars show the percentage of papers within each affiliation group that mention the limitation code. Error bars represent the 95% confidence intervals.

Figure 5 shows that papers without large company affiliations reported a significantly higher rate of Scope Limitation compared to papers with only large company affiliations (63.6% vs 56.6%, Adjusted $p = 0 . 0 1 9 )$ . They also exhibited a higher rate of Data Scarcity (13.4% vs. 9.8%) although this difference did not reach statistical significance. Conversely, while papers with only large company affiliations exhibited higher raw percentages in codes such as High Time Consumption, Methodological Constraints, and Empirical Underperformance, the differences were not statistically significant. Full test results are detailed in Appendix F.

## 4.4 RQ3: Discursive Patterns

We systematically tracked the presence of “Non-Limitation” (NL) codes to observe recurring discursive patterns in self-reported limitations. These NL codes include, e.g., outlining Future Work, giving a Contextual Justification, highlighting Strong Reported Performance, explaining Conducted Mitigation, and providing Authorial Disclaimers.

Implicit Limitation Reports. While analyzing the LLM-annotated data, we surprisingly uncovered papers containing only non-limitation codes. For instance, we identified 90 such papers (1.4%) in 2025. A manual review of these cases revealed sections in which the limitation appears only in the form of proposed future work. To illustrate, a paper might state, “Expanding beyond A and B remains an area for future exploration,” without explicitly acknowledging that the current study was limited to evaluating only A and B. In such cases, the limitations are not formally declared, risking important limitations being overlooked.

<table><tr><td></td><td colspan="4">Lim→NL</td><td colspan="4">NL→Lim</td></tr><tr><td>NL Code</td><td>Null</td><td>Observed</td><td>Diff</td><td>Adj. p</td><td>Null</td><td>Observed</td><td>Diff</td><td>Adj.p</td></tr><tr><td>Future Work</td><td>32.8</td><td>40.2</td><td>+7.4</td><td>&lt;0.001</td><td>32.8</td><td>27.9</td><td>-4.9</td><td>&lt;0.001</td></tr><tr><td>Contextual Justification</td><td>17.0</td><td>19.7</td><td>+2.7</td><td>&lt;0.001</td><td>17.0</td><td>17.3*</td><td>+0.2</td><td>0.280</td></tr><tr><td>Conducted Mitigation</td><td>6.3</td><td>8.2</td><td>+2.0</td><td>&lt;0.001</td><td>6.3</td><td>5.8</td><td>-0.4</td><td>0.002</td></tr><tr><td>Method Details</td><td>12.6</td><td>7.4</td><td>-5.1</td><td>&lt;0.001</td><td>12.6</td><td>16.3</td><td>+3.7</td><td>&lt;0.001</td></tr><tr><td>Theoretical Projection</td><td>6.3</td><td>7.3</td><td>+1.0</td><td>&lt;0.001</td><td>6.3</td><td>4.8</td><td>-1.5</td><td>&lt;0.001</td></tr><tr><td>Strong Reported Performance</td><td>11.8</td><td>4.5</td><td>-7.3</td><td>&lt;0.001</td><td>11.8</td><td>17.5</td><td>+5.7</td><td>&lt;0.001</td></tr><tr><td>Method Strength</td><td>4.7</td><td>3.8</td><td>-1.0</td><td>&lt;0.001</td><td>4.7</td><td>4.9*</td><td>+0.1</td><td>0.234</td></tr><tr><td>Recommendations</td><td>2.8</td><td>3.5</td><td>+0.7</td><td>&lt;0.001</td><td>2.8</td><td>1.9</td><td>-0.9</td><td>&lt;0.001</td></tr><tr><td>Anticipated Impact</td><td>3.5</td><td>3.0</td><td>-0.6</td><td>&lt;0.001</td><td>3.6</td><td>2.1</td><td>-1.4</td><td>&lt;0.001</td></tr><tr><td>Authorial Disclaimers</td><td>2.2</td><td>2.4</td><td>+0.2</td><td>0.003</td><td>2.2</td><td>1.6</td><td>-0.6</td><td>&lt;0.001</td></tr></table>

Table 6: Distribution of observed non-limitation codes as immediate successors (Lim→NL) and predecessors (NL→Lim) of limitation codes. Percentages are normalized over all transitions of each direction corpus-wide. Null gives the expected baseline rate under 2,000 permutations that randomly shuffle the code sequence within each paper, preserving its marginal code frequencies. Diff equals the observed rate minus the null rate. All the observed rates, except those marked <sup>∗</sup>, significantly differ from the null according to the permutation tests at adjusted $p < 0 . 0 5$ after Benjamini–Hochberg correction.

Recurring Discursive Patterns. Next, we analyzed the relative placement of non-limitation (NL) and limitation (Lim) codes within these sections. While sequential proximity does not inherently imply a semantic relationship, certain non-random patterns are noteworthy because they align with potential rhetorical strategies (even if deliberate authorial intent cannot be confirmed). Table 6 compares the rates of NL codes immediately preceding or following a limitation code against baseline rates from a permutation null model (obtained by shuffling code sequences within each paper). Among the NL codes immediately following a limitation code, 40.2% are Future Work, followed by Contextual Justification (19.7%), both of which are significantly higher than the null baselines. We label this textual pattern the “Soft Landing”, which may help soften a limitation by suggesting future work or justifying why the limitation was unavoidable.

Additionally, 8.2% of post-limitation NL codes are Conducted Mitigation, detailing proactive efforts made by authors to address the stated limitation, regardless of whether these attempts were fully successful, partially successful, or unsuccessful. Certain limitation codes precede Conducted Mitigation more frequently than others, indicating that authors often highlighted how they proactively addressed these issues: Data Leakage/Contamination, Reproducibility Gap, and Subjectivity in Evaluation/Annotation.

Finally, we analyzed NL codes that precede limitation codes and found that 17.5% of them are Strong Reported Performance (SRP). We label this textual pattern the preemptive buffer, which may cushion a disclosure by preceding it with a statement of strong results. Notably, when the subsequent limitation is Empirical Underperformance, this SRP buffer precedes it 23.3% of the time.

More details about this sequential analysis can be found in Appendices G and H. Since these patterns were inferred solely from sequential orders, we encourage future studies to semantically analyze code relations for deeper insights.

## 5 Conclusion

This paper analyzes self-reported Limitations sections in NLP research, offering three core contributions: an iterative hybrid coding framework, a large-scale LLM-annotated dataset, and empirical findings addressing our three RQs. We hope these contributions help deepen understanding of the field and spark community conversations on what constitutes a “good” limitation disclosure and how researchers can better use this mandatory section for transparent and responsible research.

## Limitations

Reliance on Explicit Self-Reporting. Our study analyzed contents in the dedicated Limitations sections only. Consequently, our analysis did not include unwritten limitations as well as limitations dispersed elsewhere in the paper, such as within the Methodology, Discussion, or Conclusion sections. Additionally, because the analyzed data relied on self-reporting by the authors, our findings capture what is said rather than what is true. Future research investigating unreported or implicitly stated limitations would enrich this line of inquiry, allowing for a more holistic evaluation of research transparency.

Caveats on Causal Interpretation. Several patterns we observed reflect salient correlations between limitation contents and paper attributes, such as publication year, research area, and author affiliations. However, these correlations do not inherently imply causation. For instance, we cannot definitively conclude that the mandatory section policy (first introduced at EMNLP 2022) caused the diverging trends of Scope Limitation and Methodological Constraints shown in Figure 3. These shifts might instead be driven by confounding factors, such as the rapid adoption of large language models that occurred around the same period. Readers should therefore interpret these results cautiously.

Limited Context for Coding. A Limitations section is typically placed after the Conclusion, so authors may reference specific methodologies, issues, figures, or tables introduced in preceding sections. Sometimes, context from these earlier sections is essential for accurately assigning codes. Therefore, our pipeline, which takes only the Limitations section as input, may miss important contextual information. While it performed relatively well, the remaining performance gap could likely be narrowed by incorporating preceding sections as additional input. Nonetheless, this context expansion should be implemented selectively to optimize token efficiency, such as by dynamically retrieving external sections only when required.

## References

Ibrahim Al Azher, Miftahul Jannat Mokarrama, Zhishuai Guo, Sagnik Ray Choudhury, and Hamed Alhoori. 2025a. BAGELS: Benchmarking the automated generation and extraction of limitations from

scholarly text. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 19279–19294, Suzhou, China. Association for Computational Linguistics.

Ibrahim Al Azher, Venkata Devesh Reddy Seethi, Akhil Pandey Akella, and Hamed Alhoori. 2025b. LimTopic: LLM-based topic modeling and text summarization for analyzing scientific articles limitations. In Proceedings ofthe 24th ACM/IEEE Joint Conference on Digital Libraries, JCDL ’24, New York, NY, USA. Association for Computing Machinery.

Gerard Alvarez, Rodrigo Núñez-Cortés, Ivan Solà, Mercè Sitjà-Rabert, Azahara Fort-Vanmeerhaeghe, Carles Fernández, Xavier Bonfill, and Gerard Urrútia. 2021. Sample size, study length, and inadequate controls were the most common self-acknowledged limitations in manual therapy trials: A methodological review. Journal of Clinical Epidemiology, 130:96– 106.

Emily Bender. 2019. The #BenderRule: On naming the languages we study and why it matters. The Gradient, 14(1).

Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal statistical society: series B (Methodological), 57(1):289–300.

David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent dirichlet allocation. J. Mach. Learn. Res., 3(null):993–1022.

Stéphane Brutus, Herman Aguinis, and Ulrich Wassmer. 2013. Self-reported limitations and future directions in scholarly reports: Analysis and recommendations. Journal ofmanagement, 39(1):48–75.

Robert Chew, John Bollenbacher, Michael Wenger, Jessica Speer, and Annice Kim. 2023. LLM-assisted content analysis: Using large language models to support deductive coding. Preprint, arXiv:2306.14924.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, and 3416 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arXiv:2507.06261.

Shih-Chieh Dai, Aiping Xiong, and Lun-Wei Ku. 2023. LLM-in-the-loop: Leveraging large language model for thematic analysis. Preprint, arXiv:2310.15100.

Olive Jean Dunn. 1961. Multiple comparisons among means. Journal ofthe American Statistical Association, 56(293):52–64.

Abdur Rahman Bin Mohammed Faizullah, Ashok Urlana, and Rahul Mishra. 2024. LimGen: Probing the LLMs for generating suggestive limitations of research papers. In Machine Learning and Knowledge Discovery in Databases. Research Track: European Conference, ECML PKDD 2024, Vilnius, Lithuania, September 9–13, 2024, Proceedings, Part II, page 106–124, Berlin, Heidelberg. Springer-Verlag.

Jennifer Fereday and Eimear Muir-Cochrane. 2006. Demonstrating rigor using thematic analysis: A hybrid approach of inductive and deductive coding and theme development. International Journal of Qualitative Methods, 5(1):80–92.

Maarten Grootendorst. 2022. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. Preprint, arXiv:2203.05794.

Nin-Chieh Hsu, Hung-Bin Tsai, Chia-Hao Hsu, Ming-Yan Tsai, Charles Liao, and Yasuharu Tokuda. 2024. Frequency of limitations statements in original research articles of united states leading medical journals: A meta-research protocol. PLOS ONE, 19(11):1–6.

Aida Kostikova, Zhipin Wang, Deidamea Bajri, Ole Pütz, Benjamin Paaßen, and Steffen Eger. 2026. LLLMs: A data-driven survey of evolving research on limitations of large language models. ACM Comput. Surv., 58(11).

Daniel Müllner. 2011. Modern hierarchical, agglomerative clustering algorithms. Preprint, arXiv:1109.2378.

Alexandra Olteanu, Su Lin Blodgett, Agathe Balayn, Angelina Wang, Fernando Diaz, Flavio Calmon, Margaret Mitchell, Michael Ekstrand, Reuben Binns, and Solon Barocas. 2025. Rigor in ai: Doing rigorous ai work requires a broader, responsible ai-informed conception of rigor. Advances in Neural Information Processing Systems, 39, Position Paper Track.

Angelina Parfenova, Andreas Marfurt, Jürgen Pfeffer, and Alexander Denzler. 2025. Text annotation via inductive coding: Comparing human experts to LLMs in qualitative data analysis. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 6471–6484, Albuquerque, New Mexico. Association for Computational Linguistics.

Jon-Marc G Rodriguez, Solaire A Finkenstaedt-Quinn, Field M Watts, and Jocelyn Elizabeth Nardo. 2024. Self-reported limitations in chemistry education research: providing specific and contextualized limitations supports researchers and practitioners. Journal ofChemical Education, 101(7):2602–2607.

Shaurya Rohatgi, Yanxia Qin, Benjamin Aw, Niranjana Unnithan, and Min-Yen Kan. 2023. The ACL OCL corpus: Advancing open science in computational linguistics. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 10348–10361, Singapore. Association for Computational Linguistics.

Johnny Saldaña. 2021. The coding manual for qualitative researchers. SAGE publications Ltd.

Kate Sanders, Jan Vahrenhold, and Robert McCartney. 2023. How do computing education researchers talk about threats and limitations? In Proceedings ofthe 2023 ACM Conference on International Computing Education Research-Volume 1, pages 381–396.

Simone Stöckli, Marianna Koufatzidou, Jadbinder Seehra, and Nikolaos Pandis. 2023. The reporting of study limitations in randomized controlled trials published in the leading dental journals: Is it sufficient? Journal ofdentistry, 136:104603.

Dimitrios Theofanidis and Antigoni Fountouki. 2018. Limitations and delimitations in the research process. Perioperative Nursing-Quarterly scientific, online official journal of GORNA, 7(3 September-December 2018):155–163.

Joel P Wiebe, Rubaina Khan, Samantha Burns, and James D Slotta. 2025. Qualitative research in the age of llms: A human-in-the-loop approach to hybrid thematic analysis. In Proceedings of the 19th International Conference of the Learning Sciences-ICLS 2025, pp. 1123-1131. International Society of the Learning Sciences.

Ziang Xiao, Xingdi Yuan, Q. Vera Liao, Rania Abdelghani, and Pierre-Yves Oudeyer. 2023. Supporting qualitative analysis with large language models: Combining codebook with GPT-3 for deductive coding. In Companion Proceedings ofthe 28th International Conference on Intelligent User Interfaces, IUI ’23 Companion, page 75–78, New York, NY, USA. Association for Computing Machinery.

Zhijian Xu, Yilun Zhao, Manasi Patwardhan, Lovekesh Vig, and Arman Cohan. 2025. Can LLMs identify critical limitations within scientific research? a systematic evaluation on AI research papers. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 20652–20706, Vienna, Austria. Association for Computational Linguistics.

Mian Zhong, Pristina Wang, and Anjalie Field. 2025. HICode: Hierarchical inductive coding with LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 31060–31078, Suzhou, China. Association for Computational Linguistics.

Naitian Zhou, David Bamman, and Isaac L. Bleaman. 2025. Culture is not trivia: Sociocultural theory for cultural NLP. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25869– 25886, Vienna, Austria. Association for Computational Linguistics.

## A Per-Code Agreement Breakdown

To examine how consistently the human annotators and $M _ { \mathrm { c o d e } }$ assign each code, we compute agreement separately for every code in the codebook. For each code, we treat coding as a binary decision per paper and compute Krippendorff’s α twice: once over the independent pre-consensus labels of the human annotators assigned to each paper, and once between $M _ { \mathrm { c o d e } }$ and the adjudicated labels. The two quantities use different references and are therefore not directly comparable. We report them side by side to show where model difficulty and annotator difficulty coincide. Table 7 lists all 25 codes, ordered by the Model vs. Adjudicated column. Support counts the papers in which $M _ { \mathrm { c o d e } }$ , the adjudicated labels, or both assign the code, and so pertains to the rightmost column. The annotator column is computed over its own support. At such low support, a small number of disagreements is sufficient to shift α substantially in either direction. This accounts for both the ceiling values observed at the top of the table (Model Hallucination/Incoherence, Potential for Misuse) and the floor value observed at the bottom (Among Annotators α is undefined for Reliance on Automatic Metrics because fewer than two annotators coded any paper containing it.). Agreement values for these codes should therefore be interpreted with caution. Two further codes, Lack ofInterpretability and Data Leakage/Contamination, occur in only one or two papers, too few for α to be defined at all.

<table><tr><td>Code</td><td>Support</td><td>Among Annotators (α)</td><td>Model vs. Adjudicated (α)</td></tr><tr><td>Model Hallucination/Incoherence</td><td>6</td><td>0.743</td><td>1.000</td></tr><tr><td>High Time Consumption</td><td>16</td><td>0.640</td><td>0.926</td></tr><tr><td>High Resource Requirements</td><td>24</td><td>0.817</td><td>0.922</td></tr><tr><td>NL: Future Work</td><td>112</td><td>0.822</td><td>0.868</td></tr><tr><td>Subjectivity in Evaluation/Annotation</td><td>11</td><td>0.247</td><td>0.832</td></tr><tr><td>Potential for Misuse</td><td>3</td><td>0.797</td><td>0.797</td></tr><tr><td>NL: Strong Reported Performance</td><td>43</td><td>0.512</td><td>0.765</td></tr><tr><td>Scope Limitation</td><td>94</td><td>0.637</td><td>0.717</td></tr><tr><td>NL: Conducted Mitigation</td><td>16</td><td>0.423</td><td>0.696</td></tr><tr><td>Lack of Evaluation Metrics or Benchmarks</td><td>6</td><td>0.885</td><td>0.658</td></tr><tr><td>Data Scarcity</td><td>16</td><td>0.434</td><td>0.639</td></tr><tr><td>Dataset Bias/Imbalance</td><td>20</td><td>0.479</td><td>0.631</td></tr><tr><td>NL: Contextual Justification</td><td>60</td><td>0.497</td><td>0.616</td></tr><tr><td>Reliance on External Tools or Resources</td><td>9</td><td>0.427</td><td>0.599</td></tr><tr><td>High Financial Cost</td><td>7</td><td>0.321</td><td>0.588</td></tr><tr><td>NL: Method Details</td><td>47</td><td>0.415</td><td>0.531</td></tr><tr><td>NL: Method Strength NL: Anticipated Impact</td><td>29 11</td><td>0.455</td><td>0.521</td></tr><tr><td></td><td>19</td><td>0.656</td><td>0.510</td></tr><tr><td>Dependency on Upstream Quality</td><td>81</td><td>0.763 0.341</td><td>0.496</td></tr><tr><td>Methodological Constraints</td><td></td><td></td><td>0.494</td></tr><tr><td>Low Data Quality Performance Trade-off</td><td>10 5</td><td>0.407</td><td>0.439</td></tr><tr><td></td><td>3</td><td>0.230</td><td>0.322</td></tr><tr><td>Reliance on Automatic Metrics</td><td></td><td></td><td>-0.007</td></tr><tr><td>Lack of Interpretability</td><td>2</td><td></td><td></td></tr><tr><td>Data Leakage/Contamination</td><td>1</td><td></td><td></td></tr></table>

Table 7: Per-code Krippendorff’s α on the 150 annotated papers, ordered by agreement with the adjudicated labels. “NL:” marks non-limitation codes; “—” marks codes with too few instances for α to be defined.

## B Second-Level Analysis of Scope Limitation

Driven by the significant trend increase in the prevalence of Scope Limitation, we investigated the underlying sub-topics driving its mentions. Specifically, we developed a sub-codebook to re-annotate the semantic units previously assigned to this Scope Limitation code as explained in Section 4.2. Definitions of the sub-codes are detailed below:

• Model Scale: Grouping disclosures that limit the experimental evaluation to specific model sizes, parameter counts, or distinct model families, frequently excluding larger or proprietary closedsource LLMs due to strict computational constraints; exemplified by: Our experimental scope was constrained to backbone models under 8 billion parameters due to computational limitations.”

• Task Coverage: Restricts the operational or evaluation scope to simple, single-turn, single-hop, or short-form tasks, explicitly omitting more complex, multi-step, or long-form task structures; exemplified by: Our investigation is based on short-form (sentence-length) QA datasets, which may notfully capture the complexity ofreal-world scenarios.”

• Language Coverage: Limits the boundaries of the study to a specific language (predominantly English) or a highly restricted subset of languages, thereby formally acknowledging the lack of broader multilingual generalization; exemplified by: Additionally, our models have not been examined on languages beyond English.”

• Dataset Utilization: Confines the empirical evaluation to a single dataset, a narrow subset of data, or specific localized benchmark collections rather than diverse, large-scale benchmarks; exemplified by: A clear limitation of this work is that it exclusively focuses on a single dataset.”

• Evaluation Framework: Restricts the methodology to specific testing paradigms, such as zero-shot evaluation or prompting-only setups, while explicitly excluding fine-tuning, model adaptation, or extensive hyperparameter optimization; exemplified by: Evaluation is performed in a prompting-only setup without model adaptation or tuning.”

• Domain Specificity: Confines the research questions, datasets, or evaluation environments to highly specialized fields, vertical topics, or distinct vertical domains; exemplified by: Our research exclusivelyfocuses on the task offactuality alignment in clinical summarization.”

• Modality Constraint: Characterized by a text-only design paradigm that explicitly excludes alternative modalities—such as audio, vision, or speech—to focus entirely on textual data processing; exemplified by: “The benchmark is limited to text and does not include multimodal inputs such as vision or speech.”

The yearly distributions of these sub-codes of Scope Limitation are shown in Table 8.

<table><tr><td>Sub-code</td><td>2020</td><td>2021</td><td>2022</td><td>2023</td><td>2024</td><td>2025</td></tr><tr><td>Model Scale</td><td>1.3% (1)</td><td>2.3% (2)</td><td>11.0% (158)</td><td>14.9% (599)</td><td>18.1% (753)</td><td>17.5% (1,097)</td></tr><tr><td>Task Coverage</td><td>11.7% (9)</td><td>5.7% (5)</td><td>13.2% (190)</td><td>14.5% (584)</td><td>14.6% (606)</td><td>14.6% (917)</td></tr><tr><td>Language Coverage</td><td>0.0% (0)</td><td>6.8% (6)</td><td>13.1% (189)</td><td>14.6% (590)</td><td>12.8% (531)</td><td>13.2% (829)</td></tr><tr><td>Dataset Utilization</td><td>9.1% (7)</td><td>9.1% (8)</td><td>10.3% (148)</td><td>11.7% (472)</td><td>10.8% (448)</td><td>11.3% (711)</td></tr><tr><td>Evaluation Framework</td><td>6.5% (5)</td><td>6.8% (6)</td><td>4.5% (65)</td><td>7.0% (281)</td><td>8.0% (334)</td><td>9.5% (594)</td></tr><tr><td>Domain Specificity</td><td>1.3% (1)</td><td>1.1% (1)</td><td>5.7% (82)</td><td>4.7% (188)</td><td>6.2% (260)</td><td>8.1% (508)</td></tr><tr><td>Modality Constraint</td><td>0.0% (0)</td><td>0.0% (0)</td><td>1.4% (20)</td><td>1.8% (73)</td><td>3.0% (126)</td><td>4.8% (299)</td></tr></table>

Table 8: Yearly distributions of sub-codes within the Scope Limitation code.

## Code Distribution by Research Area

Following the analysis of limitation codes and research areas in Appendix D, Figure 6 shows the prevalence of different limitation codes across the research areas in NLP.

![](images/11569de61ee9985dea3a94f7d10cf4587e5607e8bcc7ec467a762935beeef944.jpg)  
Figure 6: Distributions of limitation codes by NLP research areas, computed from 6,742 papers where research areas could be gathered from online conference programs. The research areas are grouped by area clusters in Appendix D, while the limitation codes are sorted by themes and corpus-wide prevalence discussed in Appendix I.

## D Exploratory Clustering of Research Areas

Before settling on the cell-level analysis reported in Section 4.3, we attempted to group research areas into clusters by their limitation profiles. Each area q was represented as a d-dimensional vector v, where d is the number of unique limitation codes in our final codebook and each element $v _ { c }$ is the proportion of papers in area q assigned code c. Each dimension was normalized to [0, 1] using Min-Max scaling, and we applied Agglomerative Hierarchical Clustering (Müllner, 2011), selecting the number of clusters using the Silhouette Score together with intra-/inter-cluster average distances. The resulting partition is weak: the clustering in Figure 7 places roughly 88% of papers in a single cluster that nearly reproduces the corpus-wide profile. This is consistent with the near-uniform reporting found at the cell level in Section 4.3. Read together, the two analyses point to the same picture, that research areas report limitations in broadly similar ways.

![](images/826d47f7a1e872ab632938b2cc0ae67ac09da59c7e8b111d2897e5cf9a958a6c.jpg)  
Figure 7: Limitation profiles of the three research area clusters with respect to the top-8 limitation codes, where the percentage axes indicate the proportion of papers within each cluster that report each specific code. The dashed grey lines represent the corpus-wide average profile.

## E Detailed Code Distribution by Paper Format

We leveraged the paper ID naming conventions in the ACL Anthology to determine the publication format (i.e., Long, Short, or Findings) for each paper. Specifically, we isolated and analyzed a subset of 7,033 ACL 2021 to ACL 2025 papers to investigate the macro-level distribution of limitation codes across these distinct publication formats. For each limitation code, we tested whether code presence is independent of publication format using a chi-square test on the corresponding $3 \times 2$ contingency table, reporting Cramér’s V as the effect size and applying Benjamini–Hochberg correction across all tests. The normalized distribution of these codes and the per-code test results are presented in Table 9.

<table><tr><td>Limitation Code</td><td>Long (n = 3,426)</td><td>Short (n = 337)</td><td>Findings (n = 3,270)</td><td> $\chi ^ { 2 } ( 2 )$ </td><td>p</td><td>Cramér&#x27;s V</td><td>Adj. p</td></tr><tr><td>Scope Limitation</td><td>63.2%</td><td>64.7%</td><td>61.4%</td><td>3.02</td><td>0.221</td><td>0.021</td><td>0.637</td></tr><tr><td>Methodological Constraints</td><td>38.6%</td><td>37.1%</td><td>36.6%</td><td>2.92</td><td>0.232</td><td>0.020</td><td>0.637</td></tr><tr><td>High Resource Requirements</td><td>20.6%</td><td>18.7%</td><td>19.0%</td><td>2.81</td><td>0.245</td><td>0.020</td><td>0.637</td></tr><tr><td>Generalization Gap</td><td>15.6%</td><td>16.0%</td><td>15.8%</td><td>0.09</td><td>0.958</td><td>0.003</td><td>0.983</td></tr><tr><td>Dependency on Upstream Quality</td><td>14.5%</td><td>12.2%</td><td>16.1%</td><td>5.67</td><td>0.059</td><td>0.028</td><td>0.503</td></tr><tr><td>Data Scarcity</td><td>12.1%</td><td>14.5%</td><td>12.0%</td><td>1.84</td><td>0.399</td><td>0.016</td><td>0.874</td></tr><tr><td>Dataset Bias/Imbalance</td><td>10.9%</td><td>10.7%</td><td>9.4%</td><td>4.54</td><td>0.103</td><td>0.025</td><td>0.503</td></tr><tr><td>High Time Consumption</td><td>9.4%</td><td>7.7%</td><td>8.8%</td><td>1.45</td><td>0.485</td><td>0.014</td><td>0.874</td></tr><tr><td>Low Data Quality</td><td>7.8%</td><td>8.0%</td><td>8.1%</td><td>0.15</td><td>0.930</td><td>0.005</td><td>0.980</td></tr><tr><td>Empirical Underperformance</td><td>7.5%</td><td>12.2%</td><td>7.5%</td><td>9.98</td><td>0.007</td><td>0.038</td><td>0.266</td></tr><tr><td>Real-World Deployment Barrier</td><td>7.3%</td><td>7.4%</td><td>6.7%</td><td>0.91</td><td>0.635</td><td>0.011</td><td>0.874</td></tr><tr><td>Measurement Vulnerabilities</td><td>6.8%</td><td>5.9%</td><td>6.4%</td><td>0.75</td><td>0.687</td><td>0.010</td><td>0.874</td></tr><tr><td>High Financial Cost</td><td>6.4%</td><td>5.0%</td><td>6.5%</td><td>1.07</td><td>0.587</td><td>0.012</td><td>0.874</td></tr><tr><td>Human Labor and Annotation Bottleneck</td><td>5.7%</td><td>6.8%</td><td>5.9%</td><td>0.72</td><td>0.699</td><td>0.010</td><td>0.874</td></tr><tr><td>Subjectivity in Evaluation/Annotation</td><td>5.6%</td><td>6.5%</td><td>5.1%</td><td>1.72</td><td>0.424</td><td>0.016</td><td>0.874</td></tr><tr><td>Reliance on External Tools or Resources</td><td>5.1%</td><td>4.2%</td><td>5.1%</td><td>0.61</td><td>0.738</td><td>0.009</td><td>0.874</td></tr><tr><td>Model Reasoning and Generation Deficits</td><td>5.3%</td><td>4.7%</td><td>4.7%</td><td>1.21</td><td>0.547</td><td>0.013</td><td>0.874</td></tr><tr><td>Scalability Bottleneck</td><td>4.5%</td><td>3.0%</td><td>5.1%</td><td>3.62</td><td>0.164</td><td>0.023</td><td>0.637</td></tr><tr><td>Theoretical Gap Lack of Reliable Evaluation Metrics</td><td>5.1%</td><td>5.9%</td><td>4.1%</td><td>4.64</td><td>0.098</td><td>0.026</td><td>0.503</td></tr><tr><td>or Benchmarks</td><td>4.6%</td><td>3.9%</td><td>4.1%</td><td>1.26</td><td>0.532</td><td>0.013</td><td>0.874</td></tr><tr><td>Resource Accessibility Barriers</td><td>4.3%</td><td>3.6%</td><td>3.5%</td><td>3.01</td><td>0.222</td><td>0.021</td><td>0.637</td></tr><tr><td>Performance Trade-off</td><td>3.6%</td><td>2.1%</td><td>3.1%</td><td>3.18</td><td>0.204</td><td>0.021</td><td>0.637</td></tr><tr><td>Reliance on Automatic Metrics</td><td>3.6%</td><td>1.5%</td><td>3.0%</td><td>5.36</td><td>0.069</td><td>0.028</td><td>0.503</td></tr><tr><td>Bias and Fairness Risks</td><td>3.2%</td><td>1.8%</td><td>2.8%</td><td>2.42</td><td>0.299</td><td>0.019</td><td>0.728</td></tr><tr><td>Hyperparameter Sensitivity</td><td>2.5%</td><td>4.5%</td><td>2.9%</td><td>4.82</td><td>0.090</td><td>0.026</td><td></td></tr><tr><td>Temporal Degradation</td><td>2.5%</td><td>2.4%</td><td>2.4%</td><td>0.01</td><td>0.996</td><td>0.001</td><td>0.503</td></tr><tr><td>Model Hallucination/Incoherence</td><td>2.2%</td><td>2.7%</td><td>2.4%</td><td>0.54</td><td>0.762</td><td>0.009</td><td>0.996</td></tr><tr><td>Prompt Sensitivity</td><td>1.8%</td><td>2.1%</td><td>2.1%</td><td>1.15</td><td>0.562</td><td>0.013</td><td>0.874</td></tr><tr><td>Data Leakage/Contamination</td><td>1.8%</td><td>2.1%</td><td>1.9%</td><td>0.31</td><td>0.856</td><td>0.007</td><td>0.874</td></tr><tr><td>High Data Requirement</td><td>1.6%</td><td>1.5%</td><td>1.9%</td><td>0.65</td><td>0.721</td><td>0.010</td><td>0.927 0.874</td></tr><tr><td>Toxicity Risks</td><td>1.2%</td><td>0.0%</td><td>0.8%</td><td>7.18</td><td>0.028</td><td>0.032</td><td>0.503</td></tr><tr><td>Privacy and Security Risks</td><td>1.2%</td><td>0.9%</td><td>1.4%</td><td>0.83</td><td>0.660</td><td>0.011</td><td></td></tr><tr><td>Reproducibility Gap</td><td>1.2%</td><td>0.9%</td><td>1.3%</td><td>0.78</td><td>0.676</td><td></td><td>0.874</td></tr><tr><td>Lack of Interpretability</td><td>1.6%</td><td>1.8%</td><td>1.4%</td><td>0.60</td><td>0.741</td><td>0.011</td><td>0.874</td></tr><tr><td>Potential for Misuse</td><td>1.6%</td><td>1.5%</td><td>1.4%</td><td>0.46</td><td>0.795</td><td>0.009</td><td>0.874</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.008</td><td>0.886</td></tr><tr><td>Societal and Ethical Risks</td><td>1.9%</td><td>1.5%</td><td>1.4%</td><td>2.83</td><td>0.243</td><td>0.020</td><td>0.637</td></tr><tr><td>Literature Coverage Gap</td><td>0.4% 0.7%</td><td>0.0%</td><td>0.8% 0.6%</td><td>5.82 0.59</td><td>0.055 0.745</td><td>0.029</td><td>0.503</td></tr><tr><td>Environmental Impact</td><td></td><td>0.6%</td><td></td><td></td><td></td><td>0.009</td><td>0.874</td></tr><tr><td>Sparse Data Sensitivity</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>1.05</td><td>0.591</td><td>0.012</td><td>0.874</td></tr></table>

Table 9: Normalized distribution of the limitation codes across Long, Short, and Findings papers, with a chi-square test of independence between code presence and publication format for each code. Dataset Task Mismatch does not occur in this subset and is therefore excluded. Cramér’s V is the effect size, and we used the Benjamini–Hochberg correction to adjust p-value across all the tests. Codes in bold are those discussed in the Paper Format paragraph of Section 4.3.

## F Affiliation Significance Test

We compare the prevalence of each limitation code between papers classified as Large company and Non-large company affiliation groups. Percentages are reported with 95% confidence intervals. The percentage-point difference (∆ pp) is calculated as the Large-company percentage minus the Non-largecompany percentage. Odds ratios (OR) and two-sided Fisher’s exact-test p-values were calculated for each code, and the reported p-values were adjusted using the Benjamini-Hochberg correction. Mixed-affiliation papers were excluded from this comparison.

<table><tr><td>Code</td><td>Large: % [CI]</td><td>Non-large: % [CI]</td><td>∆pp</td><td>OR Adj. p</td><td></td></tr><tr><td>Scope Limitation</td><td>56.6% [52.8, 60.4]</td><td>63.6% [62.7, 64.5]</td><td>-6.9630.748</td><td></td><td>0.019</td></tr><tr><td>High Time Consumption</td><td>11.9% [9.6, 14.6]</td><td>8.8% [8.2, 9.3]</td><td>3.103 1.402</td><td></td><td>0.135</td></tr><tr><td>Data Scarcity</td><td>9.8% [7.8, 12.4]</td><td>13.4% [12.8, 14.1]</td><td>-3.593 0.703</td><td></td><td>0.135</td></tr><tr><td>Reliance on Automatic Metrics</td><td>5.3% [3.8, 7.3]</td><td>3.4% [3.1, 3.7]</td><td>1.9141.596</td><td></td><td>0.145</td></tr><tr><td>Empirical Underperformance</td><td>9.0% [7.1, 11.5]</td><td>6.9% [6.4, 7.3]</td><td>2.1881.351</td><td></td><td>0.293</td></tr><tr><td>Subjectivity in Evaluation/Annotation</td><td>3.9% [2.7, 5.7]</td><td>5.8% [5.4, 6.3]</td><td>-1.9390.655</td><td></td><td>0.293</td></tr><tr><td>Low Data Quality</td><td>6.2% [4.6, 8.4]</td><td>8.4% [7.9, 8.9]</td><td>-2.1170.730</td><td></td><td>0.369</td></tr><tr><td>Environmental Impact</td><td>1.1% [0.5, 2.2]</td><td>0.6% [0.4, 0.7]</td><td>0.5201.918</td><td></td><td>0.540</td></tr><tr><td>Methodological Constraints</td><td>42.9% [39.1, 46.8]</td><td>39.9% [39.0, 40.8]</td><td>3.0431.134</td><td></td><td>0.571</td></tr><tr><td>Hyperparameter Sensitivity</td><td>3.6% [2.4, 5.3]</td><td>2.7% [2.4, 3.0]</td><td>0.902 1.348</td><td></td><td>0.571</td></tr><tr><td>Model Reasoning and Generation Deficits</td><td>3.3% [2.2, 5.0]</td><td>4.5% [4.1, 4.9]</td><td>-1.2060.722</td><td></td><td>0.571</td></tr><tr><td>Dataset Bias/Imbalance</td><td>9.2% [7.2, 11.7]</td><td>11.0% [10.4, 11.6]</td><td>-1.8040.820</td><td></td><td>0.571</td></tr><tr><td>Dataset Task Mismatch</td><td>0.2% [0.0, 0.9]</td><td>0.0% [0.0, 0.1]</td><td></td><td>0.1214.434</td><td>0.574</td></tr><tr><td>Temporal Degradation</td><td>1.7% [1.0, 3.0]</td><td>2.5% [2.3, 2.8]</td><td>-0.8200.671</td><td></td><td>0.574</td></tr><tr><td>Theoretical Gap</td><td>3.6% [2.4, 5.3]</td><td>4.6% [4.3, 5.0]</td><td>-1.053 0.765</td><td></td><td>0.574</td></tr><tr><td>Resource Accessibility Barriers</td><td>2.8% [1.8, 4.4]</td><td>3.9% [3.6, 4.3]</td><td>-1.0930.712</td><td></td><td>0.574</td></tr><tr><td>Measurement Vulnerabilities</td><td>5.3% [3.8, 7.3]</td><td>6.6% [6.2, 7.1]</td><td>-1.327 0.789</td><td></td><td>0.574</td></tr><tr><td>Lack of Reliable Evaluation Metrics or Benchmarks</td><td>3.4% [2.3, 5.1]</td><td>4.4% [4.1, 4.8]</td><td>-1.0060.765</td><td></td><td>0.610</td></tr><tr><td>High Financial Cost</td><td>7.2% [5.4, 9.4]</td><td>6.2% [5.8, 6.7]</td><td></td><td>0.9761.170</td><td>0.622</td></tr><tr><td>Performance Trade-off</td><td>4.1% [2.8, 5.9]</td><td>3.3% [3.0, 3.6]</td><td></td><td>0.7541.238</td><td>0.622</td></tr><tr><td>Bias and Fairness Risks</td><td>3.4% [2.3, 5.1]</td><td>2.8% [2.5, 3.1]</td><td></td><td>0.6401.238</td><td>0.622</td></tr><tr><td>Data Leakage/Contamination</td><td>2.3% [1.4, 3.8]</td><td>1.8% [1.6, 2.1]</td><td></td><td>0.508 1.284</td><td>0.634</td></tr><tr><td>Reliance on External Tools or Resources</td><td>4.2% [2.9, 6.1]</td><td>5.1% [4.8, 5.6]</td><td>-0.931 0.811</td><td></td><td>0.634</td></tr><tr><td>High Data Requirement</td><td>1.7% [1.0, 3.0]</td><td>1.4% [1.2, 1.7]</td><td></td><td>0.2981.214</td><td>0.774</td></tr><tr><td>Real-World Deployment Barrier</td><td>6.6% [4.9, 8.7]</td><td>7.4% [6.9, 7.9]</td><td>-0.863 0.876</td><td></td><td>0.774</td></tr><tr><td>Generalization Gap</td><td>14.7% [12.1, 17.6]</td><td>15.7% [15.0, 16.4]</td><td>-1.0380.923</td><td></td><td>0.774</td></tr><tr><td>Model Hallucination/Incoherence</td><td>2.0% [1.2, 3.4]</td><td>2.5% [2.2, 2.8]</td><td></td><td>-0.4470.816</td><td>0.887</td></tr><tr><td>Lack of Interpretability</td><td>1.6% [0.8, 2.8]</td><td>1.9% [1.7, 2.2]</td><td></td><td>-0.3420.817</td><td>0.934</td></tr><tr><td>Privacy and Security Risks</td><td>1.2% [0.6, 2.4]</td><td>1.1% [0.9, 1.3]</td><td></td><td>0.1651.154</td><td>0.957</td></tr><tr><td>Reproducibility Gap</td><td>1.2% [0.6, 2.4]</td><td>1.2% [1.0, 1.4]</td><td></td><td>0.0861.075</td><td>0.958</td></tr><tr><td>Toxicity Risks</td><td>0.8% [0.3, 1.8]</td><td>0.7% [0.6, 0.9]</td><td></td><td>0.032 1.042</td><td>0.958</td></tr><tr><td>Literature Coverage Gap</td><td>0.5% [0.2, 1.4]</td><td>0.6% [0.5, 0.8]</td><td></td><td>-0.1570.747</td><td>0.958</td></tr><tr><td>Prompt Sensitivity</td><td>1.9% [1.1, 3.2]</td><td>2.1% [1.8, 2.3]</td><td>-0.1800.911</td><td></td><td>0.958</td></tr><tr><td>Potential for Misuse</td><td>1.4% [0.7, 2.6]</td><td>1.6% [1.4, 1.8]</td><td></td><td>-0.181 0.884</td><td>0.958</td></tr><tr><td>Societal and Ethical Risks</td><td></td><td></td><td>-0.2580.827</td><td></td><td>0.958</td></tr><tr><td>High Resource Requirements</td><td>1.2% [0.6, 2.4]</td><td>1.5% [1.3, 1.7]</td><td>-0.3470.978</td><td></td><td>0.958</td></tr><tr><td>Human Labor and Annotation Bottleneck</td><td>19.3% [16.5, 22.6]</td><td>19.7% [19.0, 20.4]</td><td></td><td></td><td>0.958</td></tr><tr><td></td><td>5.1% [3.7, 7.1]</td><td>5.5% [5.1, 5.9]</td><td></td><td>-0.3560.932</td><td>0.971</td></tr><tr><td>Scalability Bottleneck</td><td>4.4% [3.0, 6.2] 16.1% [13.4, 19.1]</td><td>4.6% [4.2, 5.0]</td><td></td><td>-0.185 0.958 0.005 1.000</td><td>1.000</td></tr><tr><td>Dependency on Upstream Quality</td><td></td><td>16.1% [15.4, 16.8]</td><td></td><td></td><td></td></tr><tr><td>Sparse Data Sensitivity</td><td>0.0% [0.0, 0.6]</td><td>0.0% [0.0, 0.1]</td><td></td><td>-0.018 0.000</td><td>1.000</td></tr></table>

Table 10: Proportions of reported limitation codes by Large company affiliation group and Non-large company affiliation group.

## G Transition Frequencies (Conducted Mitigation vs. Future Work)

Driven by our statistical analysis of discursive patterns, we observed that certain non-limitation (NL) codes follow specific limitation disclosures with significantly higher regularity than others. In particular, we focused on Conducted Mitigation, which records steps already taken toward a stated limitation, encompassing successful, partially successful, and unsuccessful interventions. For each limitation code, we compared the share of its outgoing transitions that land on Conducted Mitigation against the same permutation null used in Section 4.4. Table 11 reports the seven codes with the highest observed share, all of which exceed their null and occasionally surpass the prevalence of Future Work, the most common NL code corpus-wide.

<table><tr><td rowspan="2">Limitation Code</td><td rowspan="2">Total</td><td colspan="3">NL: Future Work</td><td colspan="3">NL: Conducted Mitigation</td></tr><tr><td>%</td><td>Null %</td><td>Adj.p</td><td>%</td><td>Null %</td><td>Adj. p</td></tr><tr><td>Data Leakage/Contamination</td><td>141</td><td>10.6</td><td>20.6</td><td>0.007</td><td>24.1</td><td>12.1</td><td>0.002</td></tr><tr><td>Reproducibility Gap</td><td>68</td><td>11.8</td><td>22.0</td><td>0.072</td><td>23.5</td><td>12.7</td><td>0.005</td></tr><tr><td>Subjectivity in Evaluation/Annotation</td><td>364</td><td>23.1</td><td>25.9</td><td>0.402</td><td>23.4</td><td>11.0</td><td>0.002</td></tr><tr><td>Reliance on Automatic Metrics</td><td>259</td><td>32.8</td><td>30.2</td><td>0.484</td><td>20.8</td><td>9.9</td><td>0.027</td></tr><tr><td>Prompt Sensitivity</td><td>130</td><td>27.7</td><td>29.2</td><td>0.845</td><td>20.0</td><td>9.3</td><td>0.025</td></tr><tr><td>Privacy and Security Risks</td><td>80</td><td>27.5</td><td>27.7</td><td>0.993</td><td>20.0</td><td>7.9</td><td>0.039</td></tr><tr><td>Toxicity Risks</td><td>57</td><td>7.0</td><td>26.0</td><td>0.007</td><td>19.3</td><td>9.0</td><td>0.002</td></tr></table>

Table 11: Transition frequencies from specific limitation codes to two non-limitation codes, Future Work and Conducted Mitigation. Null gives the rate expected under 2,000 within-paper permutations, and Benjamini– Hochberg adjusted p-value across these seven tests, computed separately for each non-limitation code. Bold marks the code with the higher observed share when that share is also significant at Adj. p < 0.05.

## H Transition Frequencies (After Strong Reported Performance)

The Strong Reported Performance (SRP) code is one of the most frequent antecedents to limitation disclosures, ranking second only to Future Work. We therefore examined which limitation codes most often appear immediately after an SRP statement, again comparing the observed share against the permutation null. Table 12 details the limitation codes that are most frequently preceded by SRP statements.

<table><tr><td rowspan="2">Limitation Code</td><td colspan="2">Occurrences (n)</td><td colspan="3">% Preceded by SRP</td></tr><tr><td>SRP → Code</td><td>All → Code</td><td>Obs.</td><td>Null</td><td>Adj. p</td></tr><tr><td>Empirical Underperformance</td><td>233</td><td>1,001</td><td>23.3</td><td>9.1</td><td>&lt;0.001</td></tr><tr><td>Methodological Constraints</td><td>645</td><td>4,941</td><td>13.1</td><td>6.9</td><td>&lt;0.001</td></tr><tr><td>Dependency on Upstream Quality</td><td>264</td><td>2,042</td><td>12.9</td><td>7.2</td><td>&lt;0.001</td></tr><tr><td>High Time Consumption</td><td>149</td><td>1,201</td><td>12.4</td><td>7.4</td><td>&lt;0.001</td></tr><tr><td>Model Reasoning and Gen. Deficits</td><td>74</td><td>614</td><td>12.1</td><td>7.2</td><td>&lt;0.001</td></tr></table>

Table 12: Limitation categories most frequently preceded by Strong Reported Performance (SRP) statements. Null gives the share percentage expected under 2,000 within-paper permutations and Benjamini–Hochberg adjusted p-value. All five observed shares exceed their null, by a factor of 2.6 for Empirical Underperformance and between 1.7 and 1.9 for the remaining four.

## I Detailed Codebook

The following tables present the final codebook produced after the full-scale annotation of papers published through 2025. The limitation codes are ordered by thematic group, where each theme was derived by prompting an LLM to cluster the codes around the question “Why are these limitations?” (see Appendix K.8). Within each theme, codes are listed in descending order by the number of papers in which they were stated. The non-limitation codes in Table 14 follow the same prevalence-based ordering.

Table 13: Limitation codes from our final codebook (i.e., after the 2025 full-scale run). The year indicates the iteration when the code was added to the codebook, with “Initial” denoting codes present in the initial codebook C . For each code, we provide its definition, total paper count, corpus-wide prevalence (%), and an exemplary semantic unit.
<table><tr><td>Code</td><td>Year</td><td>Paper &amp; Preva- lence (%)</td><td>Definition</td><td>Example Semantic Unit</td></tr><tr><td colspan="5">Theme 1: Operating the models requires unsustainable resources</td></tr><tr><td>High Resource Requirements</td><td>Initial</td><td>3,285 (20.47%)</td><td>Refers to the requirement for significant processing power, high-performance hardware (e.g., A100 GPUs, TPU pods), or excessive usage of memory (RAM, VRAM, or disk storage), which may prevent the model from being deployed on consumer-grade hardware or edge devices.</td><td>Second, due to the limitation of computational resources, we focus on ine-tuning in this work, leaving applying ROSE to pre-training for future work.</td></tr><tr><td>High Time Consumption</td><td>Initial</td><td>1,453 (9.05%)</td><td>Describes processes that require excessive duration for training, processing, or inference (latency), making the method unsuitable for some applications.</td><td>Due to the complexity of the dynamic programming algorithm, it takes almost two days to finish all training epochs on 1 Tesla V100 GPU (32GB memory).</td></tr><tr><td>High Financial Cost</td><td>Initial</td><td>1,026 (6.39%)</td><td>Refers to prohibitive monetary expenses required to replicate or use the method, such as cloud computing fees, proprietary API costs, or paid datasets.</td><td>First, due to budget limit, for the non-instruction following datasets, we only examined our methods with GPT3.5 and Gemini-1.0-Pro as LLM evaluators.</td></tr><tr><td>Human Labor and Annotation Bottleneck</td><td>2022</td><td>862 (5.37%)</td><td>A severe operational constraint where the method relies on heavy, non-scalable manual labor, domain expertise, or human intervention, or suffers from a scarcity, cognitive overload, or lack of quality control of human annotators.</td><td>Finally, our approach relies on human feedback on new questions that TeachMe fails to answer or fails to justify indicating significant human efforts.</td></tr><tr><td>Scalability Bottleneck</td><td>2022</td><td>706 (4.40%)</td><td>An algorithmic and architectural limitation where the method experiences severe performance degradation, computational intractability, or exponential resource demands specifically as input lengths, dataset sizes, or model parameters grow. Includes static architectural constraints such as context window limitations or maximum input token lengths, which act as a hard ceiling for processing long-form data.</td><td>Second, our graph construction is quadratic in the number of document sentences, which restricts our method to documents of average length (e.g. 50-70 sentences).</td></tr><tr><td>High Data Requirement</td><td>2022</td><td>241 (1.50%)</td><td>Refers to architectural constraints where methods inherently demand massive volumes of training data to function competitively</td><td>One limitation of this work is that the generative model used in this paper requires a large amount of training data and is computationally expensive.</td></tr><tr><td>Environmental Impact</td><td>2022</td><td>106 (0.66%)</td><td>The real-world ecological consequences, such as carbon emissions and energy consumption, caused by the computational overhead of training or running large models.</td><td>Thirdly, compared to single-agent approaches, multiple agents require more tokens and time, increasing computational demands and environmental impact.</td></tr><tr><td>Scope Limitation</td><td>Initial</td><td>10,091 (62.88%)</td><td>Theme 2: Performance breaks down outside narrow experimental constraints Clarifies the deliberate boundaries set by the researchers, detailing what was explicitly excluded from the study (e.g., specific languages, model sizes).</td><td>Despite the strong performance of our two key designs (COCO and iDRO), we mainly verify their efficacy from their empirical performance on BEIR tasks.</td></tr><tr><td>Methodological Initial Constraints</td><td></td><td>6,382 (39.77%)</td><td>Refers to fundamental restrictions embedded in the design, logic, or mathematical assumptions of the chosen method or architecture that cannot be easily fixed without changing the core approach. Includes: pipeline complexity, preprocessing overhead, training instability, hardware/framework incompatibilities, and the need for task-specific architectural</td><td>We use a simple heuristically derived semantic similarity metric, which may not fully capture all aspects of semantic similarity between sentence pairs (Table 3).</td></tr><tr><td>Generalization Gap</td><td>2022</td><td>2,414 (15.04%)</td><td>A statistical and domain-transfer limitation where there is an empirical failure or uncertainty regarding a model's ability to maintain performance when transferred across different domains, languages, tasks. context lengths, modalities, cultural contexts, underlying model architectures, hardware platforms, or out-of-distribution datasets (e.g., zero-shot, long-tail, or out-of-vocabulary scenarios). Includes: vulnerability to overfitting on specific datasets or artifacts during training, and sensitivity to noisy, corrupted, or highly</td><td>For the latter, we increase our dataset with samples from the CNN/DM dataset, partially mitigating the problem, but out-of-domain topics still suffer.</td></tr><tr><td>Empirical Underperfor- mance</td><td>2022</td><td>1,142 (7.12%)</td><td>A quantitative and comparative shortfall where the proposed method explicitly fails to match or exceed the measurable performance metrics of established baselines, state-of-the-art models, human experts, or theoretical upper bounds, or yields only marginal and underwhelming empirical improvements.</td><td>First, there is still a performance gap between our method and the vanilla transformer in the language modeling task and CNN/Daily summarization task.</td></tr><tr><td>Real-World Deployment Barrier</td><td>2022</td><td>1,141 (7.11%)</td><td>An ecological validity and operational barrier where the fundamental assumptions, experimental setups, or structured workflows fail to represent the unstructured complexity, noise, and actual user behaviors present in live environments, rendering the method impractical for real-world deployment even if benchmark performance is high. Includes: evaluation datasets or synthetic data failing to faithfully capture the true distribution of live environments.</td><td>Moreover, our experiments are conducted on standard benchmark datasets, which may not faithfully represent the noise and complexity of real-world text streams.</td></tr><tr><td>Performance Trade-off</td><td>Initial</td><td>544 (3.39%)</td><td>Highlights scenarios where an improvement in one metric (e.g., efficiency, speed) results in a direct and unavoidable degradation of another metric (e.g., accuracy, quality)</td><td>Although the decrease of fluency is relatively small compared to the improvement of detoxification, MILDecoding does sacrifice language model quality.</td></tr><tr><td>Hyperparameter Sensitivity</td><td>2022</td><td>452 (2.82%)</td><td>A methodological and configuration fragility where a model's performance, stability, or success is highly contingent on precise architectural choices, random seed selection, or extensive manual tuning of mathematical hyperparameters, making the system difficult to optimize or reproduce without exhaustive trial and error.</td><td>Further, the need for extensive hyperparameter optimisation, particularly for combining loss functions, can hinder accessibility for non-expert users.</td></tr><tr><td>Temporal Degradation</td><td>2022</td><td>373 (2.32%)</td><td>A vulnerability where a model's effectiveness degrades over time, or a dataset becomes obsolete, due to shifting language patterns, concept drift, or temporal cutoffs.</td><td>This can lead to a degraded real-world performance if a system relies exclusively on WebIE for evaluation when the dataset is not updated accordingly.</td></tr><tr><td>Prompt Sensitivity</td><td>2022</td><td>325 (2.03%)</td><td>A linguistic and interaction fragility where a model's output quality, accuracy, or reasoning behavior fluctuates drastically based on minor variations in prompt wording, instructional phrasing, or formatting, demonstrating a lack of robustness to how a task is presented by the user. Includes: a strict dependency on high-quality, representative few-shot</td><td>We also note that using different prompt templates or changing the phrasing of instruction prompt leads to distinct response behaviors and performance (Table 5).</td></tr><tr><td>Literature Coverage Gap</td><td>2024</td><td>92 (0.57%)</td><td>examples to function effectively A methodological and scoping limitation where the authors explicitly acknowledge the potential omission of relevant prior works, recent findings, or valuable contributions due to the rapid expansion, vast volume, or dynamic nature of the scholarly literature in the field</td><td>Although we attempted to review the literature comprehensively, the rapid growth of LLM research means some recent preprints might have been missed (Section 1).</td></tr><tr><td>Dependency on Upstream Quality</td><td>Initial</td><td>2,549 (15.88%)</td><td>Theme 3: Systems are bottlenecked by flawed data and external dependencies States that the method's performance is strictly bottlenecked by the quality of the underlying base model or preprocessing steps (errors flow downstream). Includes: structural domain mismatches between pre-training and downstream data, inheriting</td><td>Our model also suffers from the noisy OCR prediction of off-the-shelf object detector, whose performance will depend highly on the extracted OCR text</td></tr><tr><td></td><td>Initial</td><td>2,019 (12.58%)</td><td>strict dependency on the characteristics of specific source data. Refers to the overall lack of sufficient data quantity, highlighting situations where there is very little data available or the data belongs to a niche domain, such as low-resource languages or tasks requiring</td><td>We clearly realize that our dataset size is relatively small compared with other related datasets due to its unique</td></tr><tr><td>Dataset</td><td>Initial</td><td>1,615 (10.06%)</td><td>attrition caused by low participant response rates or missing data. Acknowledges that the researchers did not have or collect enough data for certain groups, leading to a skewed data distribution, class imbalances, or misrepresentations that</td><td>MTurk workers generally tend to be less religious, more educated, and more likely to be unemployed than the general</td></tr><tr><td>Low Data Quality</td><td>2022</td><td>1,239 (7.72%)</td><td>gender, race). Refers to low-quality data containing inaccuracies, typographical errors, wrong labels, formatting issues, or irrelevant content (“garbage") that confuses the model. Includes: dataset incompleteness, missing annotations or metadata, structural heterogeneity, lack of diversity, and the limitations of relying on unrepresentative synthetic or simulated data Includes: coarse labeling constraints lacking fine-grained granularity, inherent data ambiguity where categories lack clear semantic definitions,</td><td>For instance, OntoNotes 5.0 dataset contains 18 coarse-grained entity types while FEW-NERD contains 9 coarse-grained and 66 fine-grained entity types.</td></tr><tr><td>Reliance on External Tools or Resources</td><td>2022</td><td>783 (4.88%)</td><td>Refers to the method's dependency on third-party software, APIs, parsers, or other models to function, which introduces risks of failure if those tools change or go offline. Includes: structural bottlenecks where performance is constrained by the limited coverage or availability of external knowledge bases.</td><td>Moreover, to automatically extract the expected evidence, we rely on core NLP tools such as coreference resolution, POS tagger and dependency parsers.</td></tr><tr><td>Resource Accessibility Barriers</td><td>2022</td><td>590 (3.68%)</td><td>A systemic and infrastructural barrier where the broader community is restricted from accessing essential underlying materials due to proprietary data dependencies, closed-source models, copyright constraints, or financial obstacles (e.g., paywalled APIs), hindering the ability to build upon or extend the research.</td><td>Further, many open and close language models are trained on content that cannot be acquired or redistributed, and thus could not be included in Dolma.</td></tr></table>

Table 13 – continued
<table><tr><td>Code</td><td>Year</td><td>Paper &amp; Preva- lence (%)</td><td>Definition</td><td>Example Semantic Unit</td></tr><tr><td colspan="5">Theme 4: Scientiic claims are difficult to measure and verify 2022</td></tr><tr><td>Measurement Vulnerabilities</td><td></td><td>(5.96%)</td><td>Evaluation flaws where inconsistent experimental setups, lack of comparable baselines, hardware dependencies, missing ablation studies, or saturated benchmarks prevent a fair, comprehensive, and direct comparison with prior state-of-the-art work. Includes: high variance across different data splits, small evaluation scales preventing statistically robust conclusions, lack of expert validation, experimental design biases (e.g., ordering bias, outcome reporting, survivorship bias, or participant observation</td><td>Although the random selection should ideally represent the whole distribution, the performance may vary slightly when comparing to the whole test set.</td></tr><tr><td>Subjectivity in Evalua- tion/Annotation</td><td>Initial</td><td>(5.23%)</td><td>Refers to the inconsistency, bias, or lack of objectivity inherent in human annotations and assessments due to vague guidelines or personal interpretation. Includes: specific demographic biases introduced by a lack of demographic diversity among human annotators.</td><td>While the benchmark dataset was annotated by human annotators, it is important to acknowledge the possibility of annotation errors or inconsistencies.</td></tr><tr><td>Theoretical Gap</td><td>2022</td><td>694 (4.32%)</td><td>An analytical or epistemic limitation where the study identifies empirical success but lacks rigorous mathematical formalization, theoretical guarantees, or fails to isolate the underlying causal mechanisms and confounding variables driving the results.</td><td>How to design a unified and more powerful KD method from the perspective of the connection between word- and sequence-level KD still remains unsolved.</td></tr><tr><td>Lack of Reliable Evaluation Metrics or Benchmarks</td><td>2022</td><td>(4.19%)</td><td>Points out the absence of established standards, benchmarks, or effective mathematical formulas to measure a specific phenomenon accurately. Includes: structural vulnerabilities where chosen metrics are inadequate, inconsistent, or structurally fail to measure critical qualitative dimensions of model behavior</td><td>However, existing benchmarks for low-resource (Han et al., 2018; Gao et al., 2019) are limited to the simple scenario of sentence-level relation classification.</td></tr><tr><td>Reliance on Automatic Metrics</td><td>Initial</td><td>564 (3.51%)</td><td>Criticizes the dependence on algorithmic scoring systems (e.g., BLEU, ROUGE) which may not correlate with human judgment or semantic understanding. Includes: systematic biases introduced by using LLMs as automated evaluators (e.g., favoring their own generated text, preferring longer responses, or exhibiting instability and inconsistency across multiple evaluation runs).</td><td>Evaluation Metrics Second, we caveat that our accuracy metrics (STR-EM and Disambig-F1) only measure the recall of the required information in the long answers.</td></tr><tr><td>Data Leak- age/Contamination</td><td>Initial</td><td>284 (1.77%)</td><td>Mentions the risk of evaluation data (test set) being inadvertently included or “seen&quot; during the training phase, leading to invalid or inflated results.</td><td>Lastly, since training data for most models is not publicly available, data leakage from the training data corpus could possibly affects our findings.</td></tr></table>

Table 13 – continued
<table><tr><td>Code</td><td>Year</td><td>Paper &amp; Preva- lence (%)</td><td>Definition</td><td>Example Semantic Unit</td></tr><tr><td colspan="5">Theme 5: The models generate unpredictable and unsafe behaviors</td></tr><tr><td>Model Reasoning and Generation Deficits</td><td>2022</td><td>673 (4.19%)</td><td>A behavioral and algorithmic limitation where the model exhibits flawed internal logic, fails to capture deep semantic meaning, relies on superficial or spurious cues, suffers from catastrophic forgetting, or produces generic, repetitive, and degenerate outputs. Includes: inherent behavioral instability or non-determinism across inference runs, and susceptibility to reward</td><td>As the appended adversarial suffix tokens may lack meaning, the resulting adversarial prompt found by the attack methods exhibits reduced naturalness.</td></tr><tr><td>Bias and Fairness Risks</td><td>2022</td><td>453 (2.82%)</td><td>An ethical and socio-technical vulnerability where the system or dataset inherits, reproduces, or amplifies historical prejudices, cultural stereotypes, or exhibits structural disparities in performance across different demographic groups.</td><td>For convenience, this work uses a standard pronunciation acoustic model from MFA, which may disadvantage those with varying pronunciation preferences.</td></tr><tr><td>Model Hallucina- tion/Incoherence</td><td>Initial</td><td>375 (2.34%)</td><td>Refers to the generation of output that is factually false, logically nonsensical, repetitive, or fluent but incorrect (common in LLMs).</td><td>At the same time, the generated premises could potentially contain fictional information, and should not be used for training models that learn facts from data.</td></tr><tr><td>Potential for Misuse</td><td>Initial</td><td>246 (1.53%)</td><td>Emphasizes the ethical risks where the technology could be exploited for malicious purposes, fraud, disinformation, or harmful content generation.</td><td>For instance, systems might exploit inferred emotional or cognitive states to influence decisions in commercial, political, or interpersonal contexts.</td></tr><tr><td>Societal and Ethical Risks</td><td>2022</td><td>228 (1.42%)</td><td>The unintended real-world harms caused by the deployment of a technology, including legal and liability risks, negative downstream societal impacts, automation bias, deceptive persuasion, and ethical or labor concerns regarding the exploitation of human</td><td>However, the current model outputs may be factually inconsistent with the input documents, and in such a case could contribute to misinformation on the internet.</td></tr><tr><td>Privacy and Security Risks</td><td>2022</td><td>192 (1.20%)</td><td>An ethical and security vulnerability where the system or dataset could inadvertently expose sensitive personal information, user feedback, or be susceptible to adversarial attacks. Includes: vulnerabilities in watermarks, defense mechanisms,</td><td>A concept has the potential to reveal additional information about the input, which may not be immediately evident upon inspecting the concept itself.</td></tr><tr><td>Toxicity Risks</td><td>2022</td><td>128 (0.80%)</td><td>A behavioral and content-safety vulnerability where the system could inadvertently generate or be provoked into producing abusive language, hate speech, or other violent, offensive, and inappropriate content.</td><td>Moreover, the training datasets we used contain violence, abuse, and biased content that can be upsetting or offensive to particular groups of people.</td></tr></table>

Table 14: Non-Limitation (NL) codes from our final codebook (i.e., after the 2025 full-scale run). The year indicates the iteration when the code was added to the codebook, with “Initial” denoting codes present in the initial codebook C . For each code, we provide its definition, total paper count, corpus-wide prevalence (%), and an exemplary semantic unit.
<table><tr><td>Code</td><td>Year</td><td>Paper &amp; Preva- lence (%)</td><td>Definition</td><td>Example Semantic Unit</td></tr><tr><td>NL: Future Work</td><td>Initial</td><td>12,041 (75.04%)</td><td>Suggests potential improvements or delegates specific unaddressed features, unanswered questions, or planned extensions to be explored in subsequent studies. Includes: theoretical mitigations proposed by the authors.</td><td>There are also ways to incorporate model generated labelling methods for more robust semi-supervision into our framework that we leave to future work.</td></tr><tr><td>NL: Contextual Justification</td><td>Initial</td><td>6,561 (40.89%)</td><td>Refers to explanations that provide external background, common field practices, or real-world constraints to defend and justify a limitation.</td><td>This is largely due to copyright issues involved with using more modern texts, and because of this, our models are inherently biased to the language used in older texts.</td></tr><tr><td>NL: Method Details</td><td>Initial</td><td>5,016 (31.26%)</td><td>Refers to factual descriptions of the proposed method which are typically stated to explain the root cause of a specific limitation.</td><td>Intuitively, the connection between nodes in the input graph can influence the encoding of x by guiding what to extract from x in order to generate y.</td></tr><tr><td>NL: Strong Reported Performance</td><td>Initial</td><td>4,668 (29.09%)</td><td>Refers to statements explicitly highlighting the model's empirical success, effectiveness, or superior benchmark results.</td><td>Although MABEL shows exciting performance across an extensive range of evaluation settings, these results should not be construed as a complete erasure of bias.</td></tr><tr><td>NL: Theoretical Projection</td><td>2022</td><td>2,600 (16.20%)</td><td>Optimistic or theoretical guesses, as well as explicit uncertainties, regarding how a method might perform in untested scenarios, unseen domains, or with future parameter and scale adjustments.</td><td>We expect improvements in classification performance when applied to multiclass or multilabel classification settings, but we have not confirmed this.</td></tr><tr><td>NL: Conducted Mitigation</td><td>Initial</td><td>2,366 (14.74%)</td><td>Describes proactive steps, alternative methods, or workarounds that the researchers have already implemented within the study to alleviate a limitation. Includes: failed mitigation attempts.</td><td>To reduce the impact of LM bias, we generate hundreds of thousands of test cases to increase the chance we obtain test cases for a given sub-category.</td></tr><tr><td>NL: Method Strength</td><td>Initial</td><td>2,021 (12.59%)</td><td>Highlights the theoretical, structural, or conceptual advantages of the proposed approach, distinct from empirical performance.</td><td>In this work, we propose a method to draw structured robust early-bird tickets, which can be used as an efficient alternative to adversarial training.</td></tr><tr><td>NL: Anticipated Impact</td><td>Initial</td><td>1,508 (9.40%)</td><td>Expresses the authors' hopes, expectations, or the broader potential influence that their findings might have on the research community or industry.</td><td>We hope our findings can inform potential avenues of improvement on data augmentation for NER and inspire the further work in this research direction.</td></tr><tr><td>NL: Recom- mendations</td><td>2022</td><td>1,106 (6.89%)</td><td>Outward-facing directives advising the broader research or practitioner community to adopt specific operational practices, metrics, strategies, or to contribute to specific research directions.</td><td>In the long-run, the community should build more elaborate coherence measures, to build a more complete picture of model capabilities and limitations.</td></tr><tr><td>NL: Authorial Disclaimers</td><td>2022</td><td>839 (5.23%)</td><td>Explicit epistemic boundaries and warnings drawn by authors to prevent readers from overclaiming, misinterpreting, or prematurely deploying the empirical results in real-world or high-stakes applications.</td><td>Regardless, we believe our models should be utilized with caution and approaches to mitigating social risks, biases, and toxicities should be carefully applied.</td></tr></table>

## J Codebook Evolution Summary

Through the iterative codebook update process, we compile the empirical statistics of codebook evolution in Table 15, reflecting the final human-approved decisions in response to LLM-generated recommendations. Specifically, for each iteration, LLM-suggested actions (i.e., merging, expanding, adding new codes, and rejecting) were manually reviewed and confirmed or altered by humans in the loop to update the codebook. Detailed codebook updates by iteration can be found in Table 16.

<table><tr><td></td><td>2020-2022</td><td>2023</td><td>2024</td><td>2025</td></tr><tr><td>No. of unique new codes detected</td><td>104</td><td>206</td><td>200</td><td>228</td></tr><tr><td>No. of clusters formed</td><td>105</td><td>104</td><td>143</td><td>74</td></tr><tr><td>- ADD</td><td>22</td><td>0</td><td>1</td><td>2</td></tr><tr><td>- EXPAND</td><td>6</td><td>9</td><td>4</td><td>8</td></tr><tr><td>- MERGE</td><td>75</td><td>95</td><td>138</td><td>64</td></tr><tr><td>- REJECT</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 15: Summary of codebook update actions across the four iterations, representing the final decisions made by humans in the loop.

<table><tr><td>Period</td><td>Action</td><td>Affected Codes</td></tr><tr><td rowspan="2">2020-2022</td><td>ADD NEW (22)</td><td>Bias and Fairness Risks • Toxicity Risks • Empirical Underperformance • High Data Requirement • Environmental Impact • Societal and Ethical Risks • Mea- surement Vulnerabilities • Generalization Gap • Scalability Bottleneck • Human Labor and Annotation Bottleneck • Model Reasoning and Generation Deficits • NL: Recommendations • NL: Theoretical Projection • NL: Authorial Dis- claimers • Privacy and Security Risks • Real-World Deployment Barrier • Re- source Accessibility Barriers • Reproducibility Gap • Hyperparameter Sensitivity</td></tr><tr><td>EXPAND (6)</td><td>• Prompt Sensitivity • Temporal Degradation • Theoretical Gap Low Data Quality • Lack of Reliable Evaluation Metrics or Benchmarks • Method- ological Constraints • NL: Future Work • NL: Conducted Mitigation • Reliance on External Tools or Resources</td></tr><tr><td rowspan="2">2023</td><td>ADD NEW (0)</td><td>None</td></tr><tr><td>EXPAND (6 unique)</td><td>Empirical Underperformance • Generalization Gap • Reliance on Automatic Metrics • Measurement Vulnerabilities • Dependency on Upstream Quality • Subjectivity in Evaluation/Annotation</td></tr><tr><td rowspan="2">2024</td><td>ADD NEW (1)</td><td>Literature Coverage Gap</td></tr><tr><td>EXPAND (4)</td><td>Model Reasoning and Generation Deficits • Measurement Vulnerabilities • Re- liance on Automatic Metrics • Societal and Ethical Risks</td></tr><tr><td rowspan="2">2025</td><td>ADD NEW (2)</td><td>Sparse Data Sensitivity • Dataset Task Mismatch</td></tr><tr><td>EXPAND (8)</td><td>Scalability Bottleneck • Data Scarcity • Generalization Gap • Low Data Quality • Privacy and Security Risks • Prompt Sensitivity • Real-World Deployment Barrier • Reproducibility Gap</td></tr></table>

Table 16: Detailed codebook updates: specific codes added and expanded by iteration. The “NL:” prefix indicates a non-limitation code.

## K Implementation Details

## K.1 Affiliation Extraction

The affiliations were extracted using a custom extraction pipeline. First, the raw PDF files were retrieved from the ACL Anthology Python library, and their first pages were parsed using PyMuPDF library<sup>7</sup>. Author names and affiliations were then identified from the parsed text using regular expressions. If this parser failed to detect affiliations, a fallback parser using Docling’s OCR engine was used to re-parse and detect affiliations again. The remaining papers without any affiliations were then manually corrected by a human checker. To evaluate the pipeline’s performance, we manually examined a random sample of 50 papers, achieving a 96% agreement rate with a human checker.

## K.2 Affiliation Type Assignment

To assign papers into the three affiliation groups, we first used an LLM to classify each unique extracted affiliation string into one of four granular types: (1) Large Company (corporations listed in the Forbes Global 2000), (2) Non-large Company (academic institutions and non-large companies), (3) Mixed Affiliation (extracted text containing both large and non-large entities due to parsing artifacts), and (4) Non-affiliation (parsing artifacts such as location, email, publication year, or symbols). Classification was performed using batch inference with a batch size of 30. After assigning these types to unique affiliations, we mapped each paper to a final group based on the following rules (excluding "Non-affiliation" entries):

• Large-company Group: If all affiliations in the paper are Large Company.

• Non-large Company Group: If all affiliations in the paper are Non-large Company.

• Mixed affiliation Group: The paper contains a combination of Large Company and Non-large Company types, or includes a Mixed Affiliation type affiliation.

The prompt used for affiliation classification is shown in Prompt K.2. To evaluate assignment quality, a human annotator randomly sampled 80 unique affiliations and verified agreement with the model outputs. The LLM classification method achieved a 90% agreement rate with the human annotator.

## Prompt K.2 LLM-based affiliation labeling.

System: You are an expert academic research metadata classifier. Your task is to classify institution/organization affiliation strings from AI/NLP research papers into one of four categories: 1. "Large Company": Pure commercial enterprises listed in the Forbes Global 2000 top companies list, or their direct research labs, AI divisions, and acquired subsidiaries (e.g. Google, Alphabet, Google Brain, Google DeepMind, Microsoft, Meta FAIR, Amazon AWS, IBM Research, Apple, Nvidia, Tencent, Alibaba, Samsung, etc.).

2. "Non-Large Company": Pure universities, academic institutes, colleges, government labs (e.g. NIST, AIST, CNRS), non-profit research institutes (e.g. Allen Institute for AI / AI2), independent research labs, or small startups/companies NOT listed in Forbes Global 2000.

3. "Mixed": Single unseparated affiliation string that contains BOTH a Large Company AND a Non-Large Company / Academic Institution joined together (e.g. "University of California, Los Angeles, CA, USA Amazon Alexa AI, Manhattan Beach, CA, USA" or "Stanford University / Google Research"). Exclude the string that contain two or more Non-Large Company affiliations. 4. "Non-Affiliation": False positives, non-institutional text, personal names, location, email addresses, web URLs, page numbers, license text, or OCR artifacts accidentally extracted as affiliation strings.

Return the classification results strictly matching the requested schema.

Below is the official reference list of Forbes Global 2000 companies for exact reference: { COMPANY\_LIST }

Human   
Classify each of the following research paper affiliation strings as either "Large Company", "Non-  
Large Company", "Mixed", or "Non-Affiliation": {input\_text}

## K.3 Semantic Unit Segmentation Prompt

This prompt was used as the system instruction for the LLM segmenter (Gemini 2.5 Flash). The human message contains the raw Limitations section of each paper. To maximize inference speed during the segmentation process, the model’s thinking configuration budget and temperature was explicitly set to zero. We employed this LLM-driven approach rather than conventional sentence splitting to ensure that the semantic content remains fully intact and free from improper fragmentation. The prompt is shown in Prompt K.3.

Prompt K.3 Semantic unit segmentation (M<sub>seg</sub>).   
System   
Split the text below into semantically distinct chunks (sentences or meaningful clauses).   
Output format: JSON list of strings.   
CRITICAL: Preserve the original text EXACTLY (verbatim). Do not modify, omit, or summarize   
anything.   
Human   
{input\_text}

## K.4 Hybrid Qualitative Coding Prompt

The main sentence-level annotation prompt is shown in Prompt K.4. The placeholder {codebook} is filled with the current codebook version; {examples} is filled with few-shot examples. To ensure deterministic outputs, the model’s temperature was set to 0. Also, to leverage deep reasoning capabilities during qualitative coding, the model was set to operate in high-thinking mode. During pipeline development, we evaluated both individual unit-by-unit annotation and batch-wise processing, finding that the resulting outputs exhibited no significant difference in annotation quality. Consequently, to optimize inference speed, maximize context efficiency, and prevent model confusion caused by overly congested prompts, we settled on a target size of approximately 100 semantic units per batch. Crucially, this boundary is configured to dynamically extend beyond the 100-unit threshold to ensure that the final paper included in a batch has all of its constituent semantic units processed entirely together, thereby preventing document fragmentation across batches.

You are an expert qualitative researcher specializing in Natural Language Processing (NLP) literature. Your task is to perform exhaustive sentence-level Hybrid Thematic Analysis on academic texts, identifying and coding limitations, justifications, and future works based on a specific codebook.

1. NESTED BATCH PROCESSING: You will receive sentences wrapped within <section paper\_id=”...”> tags. You MUST read the ENTIRE section first to grasp the overarching narrative, limitations, and causal links specific to that paper. Then, perform your analysis and output codes for EACH <sentence> sequentially. Do not bleed context from one <section> into another.

2. CROSS-SENTENCE CONTINUITY (AVOID REDUNDANCY): Track the narrative flow within the section. Do NOT redundantly assign the exact same limitation flaw code to consecutive sentences unless a fundamentally new aspect is introduced. If a subsequent sentence merely continues describing a previously coded limitation without adding new functional consequences or justifications, output exactly Code: None.

3. PARSIMONY PRINCIPLE (NO LABEL SPRAWL): Dissect the sentence to extract distinct limitations, but DO NOT over-segment. Choose ONLY the most core, primary issues. Limit your output to a maximum of 1 to 2 distinct codes per sentence. Do NOT over-predict.

4. AGGRESSIVE NOISE FILTER: If the sentence is just a section header (e.g., "Limitations"), a neutral background preamble, standard dataset statistics without critique, or a transitional filler phrase, output exactly Code: None. Do not force a code on a sentence that lacks analytical weight. 5. STRICT TRIGGERS & ROOT CAUSE:

\- Do not trigger codes based purely on isolated keywords. Analyze the underlying fundamental bottleneck.

\- Optimism about the future is strictly ’Non-Limitation: Future Work’, NOT Conducted Mitigation or Anticipated Impact.

\- Contextual Justification requires a defensive argument, not just a causal explanation.

6. CODE ASSIGNMENT & INDUCTIVE FREEDOM:

\- The Codebook is your foundation, not your ceiling. You are heavily encouraged to practice "Grounded Theory" by generating New Codes.

\- IN-VIVO FIRST: Always extract the exact functional noun phrase from the text describing the flaw (e.g., ’False-Negative Annotations’, ’Excessive Decoding Latency’).

\- DOUBLE-LAYER MANDATE: You possess the freedom to pair a granular New Code alongside ANY broad predefined limitation code (MC, HRR, Data Scarcity) to capture specific technical anatomical details.

\- Avoid hallucinating generic terms. Your New Codes MUST be directly synthesized from the authors’ exact terminology (Text-Derived Naming).

## OUTPUT FORMAT EXPECTED

You MUST strictly follow this exact plain-text structure for EVERY sentence. Do not add conversational filler.

## — Sentence ID: [ID] —

Analysis [N]:

\- Focus: "[Exact quote from the sentence]"

\- Reasoning: "[Your step-by-step reasoning comparing the focus to the codebook]"

\- Code: "[Exact Code Name from Codebook, a completely NEW Code Name, or None]"

## FEW-SHOT EXAMPLES

{examples}

## REMINDER BEFORE YOU START

\- Read the entire <section> block for context before analyzing its sentences.

\- Analyze EVERY <sentence> provided in the batch.

\- Output the — Sentence ID: [ID] — header before the analyses for each sentence.

\- Ensure exhaustiveness (Analysis 1, Analysis 2...) if a sentence compounds fundamentally independent issues, but strictly adhere to the Parsimony Principle.

CURRENT INPUT BATCH

Human

{input\_batch}

## K.5 New Code Consolidation Prompt

This prompt (K.5) was run after each annotation iteration on newly generated codes. As with the previous prompt, the model was configured with a temperature of 0 and operated in high-thinking mode. The placeholder {NEW\_CODES} is a structured list of new code labels and their evidence semantic units.

Prompt K.5 New code consolidation (M<sub>consolidate</sub>).

## System

You are an expert qualitative researcher and NLP scientist. Your task is to perform "Code Consolidation" on a list of newly generated codes (Emergent Codes) extracted from multiple separate analysis batches.

Because these codes were generated independently, many are semantic synonyms or slight lexical variations of the exact same specific NLP phenomenon. Your goal is to merge these redundant codes into unified \*\*Master Codes\*\* at the specific code-level, while filtering out weak or idiosyncratic outliers as noise.

## PRINCIPLES OF CODE CONSOLIDATION

1. SEMANTIC CONSOLIDATION (STAY AT CODE LEVEL): Merge codes that describe the \*exact same\* specific bottleneck, phenomenon, or structural flaw. Select the most precise, standard academic ML/NLP term for the Master Code.

\- CRITICAL: Do NOT over-generalize into broad umbrella themes. (e.g., You can merge "Prompt Instability" and "Sensitivity to Prompts" into Prompt Sensitivity, but do NOT generalize them all the way up to "Evaluation Issues" or "Lack of Robustness"). Keep the granularity fine-grained.

2. NOISE & OUTLIER FILTERING: Evaluate the strength and generalizability of the evidence. If an input code represents a hyper-specific edge case, has extremely weak/insufficient evidence, or does not represent a meaningful scientific limitation/justification, DO NOT create a Master Code for it. Instead, merge it into a special category named exactly Noise / Idiosyncratic Outliers.

3. EVIDENCE-BASED MERGING: Do not merge codes based purely on similar names. Read the Original Evidence and its Context carefully to ensure they are truly identical phenomena before merging.

4. PREFIX PRESERVATION: If you are merging emergent codes that represent positive aspects, justifications, or strengths, the resulting Master Code MUST retain the strict prefix Non-Limitation: .

## INPUT FORMAT

You will receive a list of emergent codes. Each entry contains:

\- Code Name: [The original emergent name]

\- Original Evidence: [Bullet points of raw sentences and context extracted from papers]

## OUTPUT FORMAT EXPECTED

Output a structured list of your finalized Master Codes. Every original code provided in the input MUST be accounted for (either merged into a specific Master Code, or dumped into Noise / Idiosyncratic Outliers).

— Master Code: [Highly Specific NLP Term OR "Noise / Idiosyncratic Outliers"] —

\- Definition: [A clear, 1-2 sentence academic definition. (Skip this field if the code is Noise)]

\- Merged Original Codes: [List the exact names of all input codes that were grouped into this Master Code.]

\- Justification: [Briefly explain WHY these codes were merged based on their evidence, or why they were deemed Noise.]

CURRENT INPUT OF EMERGENT CODES

{NEW\_CODES}

## K.6 Codebook Update Recommendation Prompt

This prompt suggests an appropriate action for each consolidated cluster: MERGE, EXPAND, ADD NEW, or REJECT (Prompt K.6). As with the previous prompts, the model was configured with a temperature of 0 and operated in high-thinking mode. Within this prompt, the {CURRENT\_CODEBOOK} placeholder was populated with the active version of the codebook slated for refinement, while the {LLM\_CLUSTER\_DATAFRAME} placeholder inputted the newly consolidated codes and their associated semantic groups derived from the preceding step.

Prompt K.6 Codebook update recommendation (M<sub>rec</sub>).

## System

You are an expert qualitative researcher and a methodologically rigorous auditor specializing in NLP literature. Your task is to analyze semantic clusters of newly generated limitation codes (synthesized by an LLM) and recommend data-driven updates to our existing Codebook.

Crucially, your output will be used to automatically re-assign the underlying granular codes to their final, official labels.

## PRINCIPLES OF CODEBOOK MANAGEMENT (STRICT GATES)

You must adhere to the principle of "Parsimony" (keeping the codebook as concise as possible). Do NOT suggest adding a new code unless the evidence is overwhelming.

1. The Synonym Trap: If a proposed cluster is merely a synonym, subset, or specific instance of an existing code, DO NOT add a new code.

2. The Artifact Trap: Reject clusters that represent generic LLM meta-language, overly broad concepts (e.g., "General Limitation"), or noise.

3. The Threshold for New Codes: A cluster warrants a completely NEW code ONLY IF it represents a fundamentally distinct concept, has a clear academic definition, and has a significant ’Total Evidence’/’Total Sources’ count that cannot be absorbed by existing categories.

4. Non-Limitation Strictness (The Broad Bucket Rule): For clusters related to mitigations, future plans, strengths, or justifications, strongly prefer to [MERGE] them into the existing broad "Non-Limitation: ..." categories.

5. Non-Limitation Naming Convention: If you absolutely MUST apply [ADD NEW] or [EXPAND] to a non-limitation concept, the new or revised code name MUST strictly begin with the prefix "Non-Limitation: " (e.g., Non-Limitation: Open-Source Release).

6. Cross-Cluster Deduplication: BEFORE assigning actions, review the ENTIRE batch of current semantic clusters. If multiple proposed clusters overlap with each other, consolidate them. Choose ONE unifying name to be the [ADD NEW], and assign [MERGE] to the redundant cluster(s), mapping them to that newly unified name.

7. Structured Expansion Principle (PRESERVE & INTEGRATE): When an [EXPAND] action is triggered, apply the following rules \*\*in order\*\*:

1. Deduplicate first. If the new nuance is already covered — explicitly or implicitly — by the existing definition, make \*\*no changes\*\*.

2. Merge elegantly. If the nuance is genuinely new, integrate it using one of these methods:

\- Weave it naturally into existing example lists or phrasing

\- Append it as a concise bullet under an Includes: or Examples: clause at the end of the definition 3. Preserve the core. Never delete, truncate, rewrite, or overwrite original sentences. The foundational meaning must remain intact and unchanged.

8. Consolidated Expansion: If multiple proposed clusters in the current batch trigger an [EXPAND] action for the EXACT SAME existing code, you must synthesize ALL of their nuances into a single, unified ’Revised Definition’. Output this same unified definition for every cluster that expands that specific code.

## EXISTING CODEBOOK

{CURRENT\_CODEBOOK}

## INSTRUCTIONS FOR ACTIONS & TRACEBACK MAPPING

I will provide you with a dataframe of Semantic Clusters. For each Cluster, evaluate its proposed ’code’ name, ’definition’, and the list of ’original\_codes’ it contains. Choose ONE of the following four actions:

\- [MERGE]: The cluster is redundant. It aligns perfectly with an existing code OR it overlaps with another new cluster in this current batch.

-> Traceback Action: All ’original\_codes’ in this cluster must be re-assigned to the EXACT NAME of the existing code, OR to the unified New Code Name you approved for the overlapping cluster. - [EXPAND]: The cluster belongs to an existing code, but highlights a specific nuance that the current definition misses.

-> Traceback Action: All ’original\_codes’ must be re-assigned to the existing code. You must provide an ADDITIVE updated definition that preserves the original meaning while appending the new nuance(s).

\- [ADD NEW]: The cluster is highly prevalent, academically distinct, represents a true blind spot, AND has been deduplicated against other clusters in this batch.

-> Traceback Action: Approve the proposed code name (or refine it to be more formal) and its formal definition. All ’original\_codes’ will be assigned to this New Label.

\- [REJECT]: The cluster is noisy, vague, or lacks coherent meaning.

-> Traceback Action: Do not map these codes.

## OUTPUT FORMAT EXPECTED

Analyze each cluster systematically using this exact format to allow for automated parsing:

— Cluster ID: [Index/ID] | Proposed Name: [code] —

1. Evaluation:

\- Prevalence: [Mention Total Evidence and Total Sources]

\- Concept Mapping: [Compare against the Existing Codebook AND other clusters in this batch.]

2. Justification: [Argue strictly WHY this needs a change, WHY it perfectly matches an existing code, or WHY it is being merged/expanded.]

3. Action: [MERGE / EXPAND / ADD NEW / REJECT]

4. Traceback Mapping & Recommendation:

\- Target Label: [The EXACT existing code name, the approved New Code Name, or the unified cross-cluster name. If REJECT, write "NONE".]

\- Revised Definition: [If EXPAND, provide the ADDITIVE new definition preserving the original text. If ADD NEW, provide the formal definition. If MERGE/REJECT, write "N/A". Ensure consistency if expanding the same target label multiple times.]

CURRENT SEMANTIC CLUSTERS

{LLM\_CLUSTER\_DATAFRAME}

## K.7 Second-Level Intra-Code Clustering Prompt

This prompt explores sub-codes within a single parent code by clustering its evidence sentences (Prompt K.7). Within this prompt, the {target\_code\_name} placeholder specifies the particular parent code currently under analysis, while the {input\_json} placeholder delivers the structured input data comprising all semantic units associated with that target code, thereby enabling the model to investigate and uncover more granular, fine-grained sub-structures within its content. To guarantee strict output determinism and leverage deep structural analysis, the model was configured with a temperature of 0 and operated in high-thinking mode.

## Prompt K.7 Second-level intra-code clustering.

## System

You are an expert Qualitative Researcher conducting a 2nd-level Thematic Analysis.

You are analyzing a JSON array of sentences that share the overarching parent code: "{target\_code\_name}".

INSTRUCTIONS & CHAIN OF THOUGHT:

Step 1: Concept Extraction — identify recurring mechanisms, boundaries, or subjects being restricted.

Step 2: Taxonomy Generation — group into distinct, mutually exclusive clusters based on core mechanism.

Step 3: In-Vivo Naming — precise academic noun phrase from the authors’ own terminology. No generic terms.

Step 4: Anchor Selection — EXACTLY 5 representative sentences per cluster (or all if fewer than 5).

Step 5: Keyword Extraction — 3–5 high-signal keywords/n-grams for future trend tracking.

INPUT DATA:

{input\_json}

```jsonl
OUTPUT FORMAT (JSON only, no markdown fences):
{"Clusters": [{"cluster_name": "...", "inclusion_criteria": "...", "reasoning": "..."
"representative_sentences": ["..."], "keywords": ["..."]}]}
```

## K.8 Question-Based Theme Searching Prompt

This prompt groups the finalized codes into overarching themes that directly address a given research question (Prompt K.8). The {question} placeholder specifies the target question of interest for theme searching, while the {code} placeholder delivers the comprehensive details of each code, including its name, definition, semantic unit examples, and overall prevalence. Within our current pipeline, these searched themes were utilized primarily to systematically sort and structure the codebook for enhanced readability and presentation. To facilitate deep qualitative synthesis, the model operated in high-thinking mode with a default temperature of 0. Notably, this temperature can be increased if generating a more diverse set of themes is desired.

## Prompt K.8 RQ-driven theme generation.

## System

You are an expert qualitative researcher and theorist specializing in Natural Language Processing (NLP).

You have just completed the coding phase of a Thematic Analysis.

Your task is to synthesize the final Enriched Codebook into overarching Themes that specifically answer a defined Research Question.

## RESEARCH QUESTION

{question}

## PRINCIPLES OF RQ-DRIVEN THEME GENERATION

1. Answer the RQ: Every theme MUST directly answer the Research Question — not just categorize codes.

2. Latent Meaning over Semantic Similarity: Group codes based on how they collectively answer the RQ, even if they don’t share keywords.

3. Weight vs. Significance: Use Prevalence to gauge gravity. High-count codes often form the core of major themes.

4. Exclusivity: A code should primarily belong to only ONE theme to maintain a clear narrative.

5. Simple, Scannable Theme Names: Each theme name must be immediately understandable on first read.

\- Use plain language — no jargon, no academic abstractions.

\- Prefer short noun phrases or a single clear sentence (under 10 words).

\- A reader unfamiliar with the study should grasp the theme’s meaning instantly.

ENRICHED CODEBOOK

{code}

## INSTRUCTIONS

Group ALL provided codes into exactly 3 to 5 distinct Themes.

For each Theme, define its "Core Answer to the RQ".

## OUTPUT FORMAT

— Theme [N]: [Theme Name — a narrative phrase that answers the RQ] —

\- Core Answer to the RQ: [2-3 sentences explaining HOW this theme answers the RQ.]

\- Constituent Codes:

\* [Code Name] (Prevalence: [N papers]) - [1 sentence on how this code supports the theme.]

## Human

{enriched\_codebook}

## L The Use of AI Assistants

In this research, we used Large Language Models (LLMs) as programming assistants to generate data analysis scripts, execute code refactoring, and support minor implementation tasks. In all instances, the generated source code was strictly verified by a human to ensure correctness. Additionally, we utilized LLMs as writing assistants to optimize vocabulary choices, refine grammatical phrasing, and assist in structural LaTeX formatting across the manuscript without altering any underlying technical content.