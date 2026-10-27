![](images/7f3b1a07f4978c99769f0b6c1d2ff9cc1a59461c9784aec307b4f2803b38646c.jpg)

# QUANTIFYING CLAIM SMOOTHING BY LLM RESEARCH AGENTS

Tiasa Singha Roy New York University ts5378@nyu.edu

Manya Wadhwa New York University mw4141@nyu.edu

Greg Durrett New York University gdurrett@nyu.edu

## ABSTRACT

Deep research, or any use of large language models (LLMs) to read a large document collection and produce insights, requires synthesizing information from different sources. In this work, we find that LLMs produce long-form research reports that read as confident syntheses but which fail to accurately reflect the underlying documents. We call this failure smoothing. In collapsing diverse and potentially conflicting sources into a single narrative, the model discards the evidential “jaggedness” of its corpus and asserts conclusions more decisive than the citations can support. We introduce a claim level scoring pipeline, SMOOTH-SCORE, to quantify this effect. Our pipeline extracts key conclusive claims from model generated reports, then finds evidence relevant to each claim and clusters that evidence along high level aspects. Finally, each claim is scored with respect to the evidence to quantify the degree of smoothing done by the agent. Across 7 systems and 4 datasets, we find that agents consistently produce stronger claims than warranted<sup>1</sup>.

## 1 INTRODUCTION

Large Language Models (LLMs) are often deployed within systems that read across many sources of information and return long-form reports with citations (Shao et al., 2026; Xie et al., 2026; Li et al., 2026b; Nakano et al., 2021). These systems, either general-purpose agents paired with Internet access or special-purpose “deep research” agents, retrieve large volumes of documents and analyze and synthesize this information into coherent reports (Du et al., 2026; Onweller et al., 2026). The skills required to do this go beyond simple multidocument summarization (Fabbri et al., 2019): agents must weigh evidence, integrate conflicting findings, and determine what the evidence warrants (Narayanan Venkit et al., 2026; Rasheed et al., 2026; Zhu et al., 2026). The resulting reports are typically fluent, well-organized, and densely cited, which makes them read as confident, authoritative syntheses. As a result, users increasingly treat these outputs as substitutes for reading the underlying literature, even when the synthesis misrepresents or overstates what the cited sources actually support (Li & Aral, 2025; Huang et al., 2026).

In this work, we study the prevalence of smoothing in LLM generated reports. We define smoothing as the tendency of LLMs to synthesize claims that are more decisive than supported by the documents they cite. Such claims may, for example, fail to portray mixed evidence or overclaim a result to hold in all circumstances when it has only been shown more

Figure 1: When synthesizing over many documents, agents can present a claim as well supported when the evidence is actually more mixed. SMOOTHSCORE flags these smoothed-over claims and quantifies the degree of smoothing.

narrowly. By contrast, a faithful claim should hedge when the evidence is mixed and narrow its scope when the support is narrow.

To identify instances of this behavior, we propose SMOOTHSCORE, a framework for measuring whether the claims in a synthesized report are calibrated to the cited documents. As shown in Figure 1, we find that an agent’s report can present a claim as well supported by some of its citations, even though other retrieved evidence shows the claim only holds under certain conditions. The bulk of prior work on claim verification and long-form factuality (Min et al., 2023; Song et al., 2024; Tang et al., 2024) can check whether individual claims are supported by their cited evidence, but breaks down for synthesized claims: these systems usually return binary judgments at the claim or subclaim level, but do not handle calibration and scoping across a large body of potentially conflicting evidence.

To compute SMOOTHSCORE, we first decompose each report into its main conclusive claims and identify the documents the report cites. As shown in Figure 2, for each claim, we run two agents over the report’s cited documents: (a) Support Agent: which surfaces evidence consistent with the claim, and (b) Skeptic Agent: which actively searches for evidence that qualifies, limits, or contradicts the claim. Evidence from both these passes are clustered into aspects reflecting different dimensions of the claim, such as viewpoint, strength of argument, and certainty (Kargupta et al., 2025). Given the claim and its mapped aspect-evidence pairs, we score the claim’s calibration on a 1–5 scale, reflecting how well its stated strength and scope match the evidence collected across both passes.

Across four benchmarks and seven systems, we find that smoothing is pervasive. On average, for each system at least 25% of generated claims received a SMOOTHSCORE ≥ 3. For most systems over half of claims show at least material smoothing, and up to 35% of Gemini Deep Research’s claims show severe smoothing (≥ 4). We also find that limited interaction with the cited documents contributes to smoothing: Luna opens only 19% of the pages it cites, and actively engaging with the full documents can help reduce smoothing.

Contributions Our main contributions are: (a) We introduce SMOOTHSCORE, a framework for measuring whether the claims in a synthesized report are calibrated to the strength and scope of their cited evidence, rather than merely supported by it and (b) we find this issue to be widespread across frontier systems on 4 different benchmarks.

## 2 SMOOTHING IN LLM RESEARCH AGENTS

## 2.1 DEFINING SMOOTHING

Setup and Definitions Given a prompt p, a retrieval-augmented agent M produces a long-form response or report y with inline citations. We collect the full text of the cited documents in the report $\bar { D = { \{ d _ { 1 } , \ldots \bar { \bf { \sigma } } , d _ { m } \} } }$ . Our goal is to measure if the synthesized conclusive claims in y faithfully represent what D supports.

We call a claim smoothed if it satisfies three criteria:(1) it reflects a conclusion drawn from aggregated evidence; (2) some of the evidence supports at least part of the conclusion; and (3) the conclusion as a whole is stronger or broader than the body of the evidence warrants.

This goal is closely related to hallucination detection (Zha et al., 2023; Min et al., 2023; Tang et al., 2024); we view it as a special case of hallucination detection. However, typical methods for hallucination detection assume binary factuality judgments. Past work has looked at more nuanced judgments for narrative summarization (Subbiah et al., 2025) and opinion summarization (Bhaskar et al., 2023), but these methods are designed for simpler settings and do not handle the complexity of smoothing in deep research systems.

Evaluating whether a claim is smoothed requires more than checking whether a report’s statements are supported by individual citations. For instance, in Figure 2, the extracted claim ‘discrete imagetoken modeling appears more promising than a Stable-Diffusion-centered architecture’ is a conclusion made by the deep research system (satisfying condition (1) above). It is supported by the cited documents, $D _ { 1 }$ Emu3 paper (Wang et al., 2024)) and $D _ { 3 }$ (the Show-o paper (Xie et al., 2025)) (satisfying condition (2)). However, we find that the same sources along with other documents, namely $D _ { 2 }$ (the Transfusion paper (Zhou et al., 2024)), also contains results that show that the statement is only true under certain conditions present counter evidence (satisfying condition (3)). Therefore, smoothing can occur even when a claim appears supported, and methods producing binary judgments from small evidence sets cannot comprehensively capture the distribution to identify it.

![](images/7feb86a4b38d89f9dddb35a12dcecd3f85ee1eb3f62a906d6a6ce29381e0db44.jpg)  
Figure 2: Overview of SMOOTHSCORE. Given a query and a system generated report, we (1) first extract conclusive claims. Then from the cited documents, (2) we get supporting and skeptic evidence relevant to the claim. (3)/(4) We then organize the evidence into aspects. (5) The claim is then scored with respect to the gathered evidence to quantify the degree of missing information.

Identifying smoothing via aspects We propose SMOOTHSCORE as a way of measuring the degree of smoothing a conclusion shows. Figure 2 outlines the pipeline. At a high level, first, we extract such claims from a generated report that represent the key takeaways, then find evidence that supports and refutes those claims. However, it is not straightforward to combine this evidence into a final score: counting supporting documents vs. refuting documents does not account for the nuances of how a claim is supported or refuted.

Instead, for the claim c in the report y and evidence from a cited document d ∈ D, we identify a set of dimensions $A ( c , d ) = \{ a _ { 1 } , \ldots , a _ { k } \}$ that we call aspects, following (Kargupta et al., 2025). These aspects are consolidated across documents, to capture fine-grained ways in which a claim might be refuted. For instance, in Figure 2, step 3, we are able to find ‘performance and efficiency of tokens vs diffusion models’, under which we see some evidence state that ‘tokens match or beat diffusion’ but we also see evidence for ‘diffusion or hybrid set ups are competitive’. As aspects are discovered per claim and tailored to the claim and its evidence, we use them to score claim against the aspect-organized evidence to quantify its degree of smoothing.

## 2.2 MEASURING SMOOTHING WITH SMOOTHSCORE

Step 1: Claim Extraction To focus on the agents’ ability to synthesize from aggregated evidence, we extract conclusion-bearing claims (condition (1)). These are cases where report takes a stance or presents a takeaway. For example, in Figure 2, ‘image-token modeling appears more promising than a Stable-Diffusion-centered architecture’ is a takeaway from the report which we extract as a claim.

Using prompt, Appendix B.1, we extract,

$$
C = { \mathrm { E x t r a c t C l a i m s } } ( y ) = \{ c _ { 1 } , \ldots , c _ { n } \} .
$$

Each claim preserves its direction, confidence, scope, and qualifications as stated in the report.

Step 2: Evidence Extraction In Figure 2, we see that a cited document, for example, D1 can carry information relevant to the claim that can both support or undermine it. To identify information relevant to the claim from the cited documents throughout the report, we extract evidence using a ‘support agent’ and a ‘skeptic’ agent.

• Support Agent (Appendix B.2): Given (claim, document), this agent extracts evidence that supports the assertions made by the claim. We also include stated concession as evidence when they confirm limitations or conditions present within the claim.

$$
S ( c ) = \bigcup _ { d \in D } \operatorname { E x t r a c t } _ { + } ( c , d )
$$

• Skeptic Agent (Appendix B.3): Given (claim, document), this agent extracts evidence that contains any information against the claim. To capture the “jaggedness” we go beyond just finding counters to the assertions made by the claim. The agent also extracts evidence to support limitations and conditions stated in the claims. Additionally, we include evidence on alternate approaches and their success related to the claims where relevant. This is to provide context for scope.

$$
R ( c ) = \bigcup _ { d \in D } { \mathrm { E x t r a c t } } _ { - } ( c , d )
$$

For each agent, the evidence is composed of: a source quote from the document, aspect detailing the assertion, a note on how the evidence bears on the claim, and strength rating.

Step 3: Aspect Induction Different cited documents can be relevant to different parts of a claim, so judging the claim against the raw documents forces the judge to compare evidence that is not directly comparable. If we look across the cited documents, however, the relevant information naturally clusters into topics that bear on the claim. For example, in Figure 2, some documents discuss comparing the performance and efficiency of image tokens vs. diffusion approaches, while others discuss fit with LLM-style next-token scaling. We refer to these topics as aspects (Kargupta et al., 2025).

To construct aspects, for each claim–document pair $( c _ { i } , d _ { j } ) \in C \times D$ we generate a set of documentspecific aspect proposals,

$$
A _ { i j } = \operatorname { P r o p o s e } ( c _ { i } , d _ { j } ) .
$$

Each proposal identifies a dimension discussed in $d _ { j }$ that bears on $c _ { i } ,$ , and consists of a short name, a brief description, and a set of keywords. If $d _ { j }$ contains nothing relevant to $c _ { i } .$ , then $A _ { i j } = \emptyset$ (see Appendix B.4 for the prompt). Because proposals are generated independently for each document, different documents will often propose overlapping or near-duplicate aspects. For each claim, we therefore merge the proposals across documents into a single, deduplicated set of aspects, keeping track of which documents contributed to each one (see Appendix B.5).

Organizing evidence by aspect places findings that address the same part of the claim side by side, regardless of which document they came from. This allows the judge to weigh evidence both supporting and opposing the claim on a like-for-like basis. For example, along a comparative dimension there may be supporting and countervailing evidence for either stance, as seen in aspect A2 of Figure 2.

Step 4: Evidence Assignment We assign every extracted evidence item in $E ( c _ { i } ) = S ( c _ { i } ) \cup R ( c _ { i } )$ to the aspect it discusses, so that findings addressing the same part of the claim are grouped together and can be compared directly (prompt in Appendix B.6). Assignment depends only on what the evidence discusses, not on whether it supports or counters the claim, so an item’s stance, source, and strength rating are left unchanged. For example, in Figure 2, the support-agent finds that Emu3 beats SDXL in human evaluation and the skeptic-agent finding that Emu3’s own table shows a lower GenEval score than SDXL are placed under the same aspect, even though they point in opposite directions.

Step 5: Computing Smoothing Finally, a judge J produces a SMOOTHSCORE $\sigma _ { i }$ for each claim. The judge receives the input question $q ,$ the claim $c _ { i } ,$ , and all extracted evidence $E _ { i }$ organized under the claim’s aspects T<sub>i</sub>, including evidence that is not tied to any single aspect:

$$
\sigma _ { i } = J { \big ( } q , c _ { i } , T _ { i } , E _ { i } { \big ) } \in \{ 1 , \ldots , 5 \} .
$$

The judge assesses how faithfully the claim compresses the supplied evidence, focusing on whether it omits or understates assertions, conditions, or scope in a way that changes its central takeaway. Using the five-point rubric in B.7 we score degree of smoothing (condition (3)). The criteria for receiving a score of 5 includes cases that are pure hallucination and don’t match condition (2). However, we rarely observe cases of 5 (Section 4) and therefore view our system as primarily a method for analyzing smoothing rather than broader hallucination detection (even though it can identify such cases). Figure 2 and Table 15 show examples of the generated outputs for each stage of the pipeline.

## 3 EXPERIMENTAL SETUP

Datasets. We calculate SMOOTHSCORE on the following datasets: (1) ResearcherBench (Xu et al., 2026), (2) ScholarQA-CS2 (Expert-Written) (Hwang et al., 2026) (3) DeepResearch Bench II (Li et al., 2026a) (4) ResearchRubrics (Sharma et al., 2025). These provide a range of tasks curated for research report generation. To ensure all queries require a degree of multi document synthesis we manually select queries. Details on this selection process is available in Appendix A.2

Systems. To check if overclaiming persists across architectures with varying design considerations, we evaluate the following 7 systems: (1) GPT Luna 5.6 + Websearch, (2) GPT Sol 5.6 + Websearch, (3) Claude Sonnet 5 + Websearch, (4) Claude Opus 5.5 + Websearch, (5) Gemini Deep Research, (6) DR Tulu (Shao et al., 2026), (7) and QUEST (Xie et al., 2026). Together, these systems span several current approaches to generating research reports, including agents augmented with Websearch, proprietary end-to-end deep-research systems, and open research agents with specialized retrieval and reasoning pipelines. Addtional details provided in Appendix A.3

Metrics We report SMOOTHSCORE in two ways: (1) the mean SMOOTHSCORE across claims for each system and benchmark, and (2) the fraction of claims in each SMOOTHSCORE bin. We believe the fraction of claims ≥3 indicates how often systems generate materially misleading claims.

## 4 RESULTS

## 4.1 SMOOTHSCORE RESULTS

Smoothing is widespread. Table 1 shows that smoothing is pervasive across systems and datasets. Mean SMOOTHSCORE ranges from 1.66–1.95 for Luna and 1.82–1.93 for Sol to 3.05–3.27 for Gemini, with every other system falling between 2.20 and 2.88. The severity of smoothing is also reflected in the score distribution breakdown: smoothing of ≥3 is present on every system and dataset, with at least 22% of claims affected in every case, but severe smoothing (≥4) is where systems diverge sharply. Gemini smooths severely on 22–35% of its claims, roughly 3–12 times the rate of other closed-source agents and 13-26× the rate of Sol.

Open-source agents produce severely smoothed claims Systems follow similar patterns across datasets. The mean SMOOTHSCORE in open-source Deep Research systems like DR Tulu (Shao et al., 2026) and QUEST (Xie et al., 2026) remains comparable or slightly above that Sonnet-5. However, at least ∼59% and ∼55% of their claims score $\geq 3 ,$ , and 7–23% score $\geq 4 .$ From Table 11, QUEST draws from among the smallest evidence pools (roughly 10 documents and 40–55k words) so it has the least amount of evidence to compress. Despite this, while closed source systems only generate 1–2% of claims that score = 5, QUEST generates up to ∼7% and ∼4% such claims on DeepResearchBench II and ResearchRubrics. Appendix G.1 shows that QUEST often generates outputs that read as summarization rather than synthesis. The resulting claims preserve specifics faithfully while overstating scope enough to drive inflated conclusions.

Closed-source systems vary widely in smoothing severity. Among closed-source agents with Websearch, we see that Sonnet 5 and Opus 5.5 produces almost 1.4–2.5× the fraction of claims that achieve a SMOOTHSCORE of 3–4 relative to Luna and Sol in Table 1. These systems differ in the middle of the distribution rather than at its extremes and neither of the systems produce fundamentally misleading claims (Scored 5) at a meaningful rate, at most 1.1% on any dataset. On the other hand, Luna and Sol generate at least 36% claims that show no smoothing (score = 1), 1.5–2.7x more than any other system across datasets.

Across all systems, Gemini performs the worst, with 85–93% of claims scoring ≥ 3 and an average score of 3.05–3.27. Gemini also draws from significantly more documents (roughly 2–5x) than Luna or Sol with websearch, and produces the longest reports on every dataset (Table 11). This gives the agent a much larger surface to synthesize, and despite the added length, the report does not appear to compress this information more effectively. Appendix G.2 lists some examples of claims from the generated reports. They contain language that turns a strong result into a universal or final conclusion.

Table 1: We report the number of reports, claims for each of systems and benchmarks evaluated in this work. We also report the SMOOTHSCORE with ≥3 being materially smoothed and ≥4 being the subset of severely smoothed claims. Open source systems like, DR Tulu and QUEST produce a large % of severely smoothed claims, while closed source agents vary with Luna and Sol systems only having 1-4% claims $\geq 4$ against Gemini’s 22-35%.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">System</td><td colspan="2">Sample</td><td rowspan="2">SMOOTHSCORE</td><td colspan="5">% of claims w/given SMOOTHSCORE</td></tr><tr><td>Reports</td><td>Claims</td><td>Mean [95% CI]</td><td>1 2</td><td>3</td><td>4</td><td>5</td></tr><tr><td rowspan="7">ResearcherBench</td><td>Luna 5.6</td><td>65</td><td>443</td><td>1.95 [1.85, 2.06]</td><td>37.0%</td><td>34.1%</td><td>25.7%</td><td>3.2%</td><td>0.0%</td></tr><tr><td>Sol 5.6</td><td>64</td><td>430</td><td>1.93 [1.81, 2.05]</td><td>37.0%</td><td>35.6%</td><td>25.1%</td><td>2.3%</td><td>0.0%</td></tr><tr><td>Sonnet 5</td><td>65</td><td>435</td><td>2.52 [2.42, 2.61]</td><td>13.8%</td><td>26.7%</td><td>53.3%</td><td>6.2%</td><td>0.0%</td></tr><tr><td>Opus 5.5</td><td>65</td><td>445</td><td>2.36 [2.28, 2.44]</td><td>19.1%</td><td>29.4%</td><td>47.9%</td><td>3.6%</td><td>0.0%</td></tr><tr><td>QUEST</td><td>63</td><td>404</td><td>2.50 [2.35, 2.65]</td><td>18.1%</td><td>26.7%</td><td>43.6%</td><td>10.1%</td><td>1.5%</td></tr><tr><td>DR Tulu</td><td>65</td><td>408</td><td>2.57 [2.47, 2.67]</td><td>12.0%</td><td>26.7%</td><td>54.2%</td><td>6.9%</td><td>0.2%</td></tr><tr><td>Gemini</td><td>64</td><td>434</td><td>3.27 [3.21, 3.34]</td><td>1.6%</td><td>5.1%</td><td>58.1%</td><td>34.8%</td><td>0.5%</td></tr><tr><td rowspan="7">CS2-Expert</td><td>Luna 5.6</td><td>25</td><td>164</td><td>1.93 [1.75, 2.11]</td><td>44.5%</td><td>23.8%</td><td>27.4%</td><td>3.0%</td><td>1.2%</td></tr><tr><td>Sol 5.6</td><td>25</td><td>165</td><td>1.82 [1.66, 1.98]</td><td>43.6%</td><td>31.5%</td><td>23.6%</td><td>1.2%</td><td>0.0%</td></tr><tr><td>Sonnet 5</td><td>25</td><td>167</td><td>2.30 [2.12, 2.48]</td><td>21.6%</td><td>32.3%</td><td>40.7%</td><td>5.4%</td><td>0.0%</td></tr><tr><td>Opus 5.5</td><td>25</td><td>188</td><td>2.20 [2.05, 2.33]</td><td>28.7%</td><td>26.6%</td><td>42.0%</td><td>1.6%</td><td>1.1%</td></tr><tr><td>QUEST</td><td>25</td><td>156</td><td>2.67 [2.46, 2.87]</td><td>17.3%</td><td>18.6%</td><td>47.4%</td><td>13.5%</td><td>3.2%</td></tr><tr><td>DR Tulu</td><td>25</td><td>168</td><td>2.70 [2.53, 2.86]</td><td>12.5%</td><td>19.0%</td><td>55.4%</td><td>12.5%</td><td>0.6%</td></tr><tr><td>Gemini</td><td>24</td><td>175</td><td>3.18 [3.09, 3.28]</td><td>4.0%</td><td>5.7%</td><td>59.4%</td><td>29.7%</td><td>1.1%</td></tr><tr><td rowspan="7">DRB-II</td><td>Luna 5.6</td><td>18</td><td>106</td><td>1.66 [1.46, 1.86]</td><td>58.5%</td><td>18.9%</td><td>20.8%</td><td>1.9%</td><td>0.0%</td></tr><tr><td>Sol 5.6</td><td>18</td><td>111</td><td>1.83 [1.62, 2.04]</td><td>45.0%</td><td>29.7%</td><td>23.4%</td><td>0.9%</td><td>0.9%</td></tr><tr><td>Sonnet 5</td><td>18</td><td>107</td><td>2.46 [2.24, 2.68]</td><td>15.0%</td><td>30.8%</td><td>47.7%</td><td>6.5%</td><td>0.0%</td></tr><tr><td>Opus 5.5</td><td>18</td><td>106</td><td>2.39 [2.24, 2.53]</td><td>21.7%</td><td>26.4%</td><td>43.4%</td><td>8.5%</td><td>0.0%</td></tr><tr><td>QUEST</td><td>16</td><td>95</td><td>2.88 [2.54, 3.25]</td><td>9.5%</td><td>22.1%</td><td>46.3%</td><td>14.7%</td><td>7.4%</td></tr><tr><td>DR Tulu</td><td>18</td><td>102</td><td>2.49 [2.27, 2.71]</td><td>19.6%</td><td>21.6%</td><td>50.0%</td><td>7.8%</td><td>1.0%</td></tr><tr><td>Gemini</td><td>18</td><td>120</td><td>3.05 [2.89, 3.23]</td><td>2.5%</td><td>12.5%</td><td>62.5%</td><td>22.5%</td><td>0.0%</td></tr><tr><td rowspan="7">ResearchRubrics</td><td>Luna 5.6</td><td>17</td><td>111</td><td>1.83 [1.57, 2.10]</td><td>45.0%</td><td>31.5%</td><td>19.8%</td><td>2.7%</td><td>0.9%</td></tr><tr><td>Sol 5.6</td><td>17</td><td>108</td><td>1.93 [1.73, 2.12]</td><td>36.1%</td><td>38.9%</td><td>22.2%</td><td>1.9%</td><td>0.9%</td></tr><tr><td>Sonnet 5</td><td>17</td><td>110</td><td>2.55 [2.37, 2.72]</td><td>12.7%</td><td>29.1%</td><td>49.1%</td><td>8.2%</td><td>0.9%</td></tr><tr><td>Opus 5.5</td><td>17</td><td>100</td><td>2.42 [2.21, 2.61]</td><td>17.0%</td><td>29.0%</td><td>49.0%</td><td>5.0%</td><td>0.0%</td></tr><tr><td>QUEST</td><td>17</td><td>104</td><td>2.88 [2.61, 3.16]</td><td>12.5%</td><td>14.4%</td><td>50.0%</td><td>19.2%</td><td>3.8%</td></tr><tr><td>DR Tulu</td><td>17</td><td>99</td><td>2.63 [2.38, 2.86]</td><td>14.1%</td><td>24.2%</td><td>49.5%</td><td>9.1%</td><td>3.0%</td></tr><tr><td>Gemini</td><td>17</td><td>113</td><td>3.23 [3.05, 3.40]</td><td>6.2%</td><td>1.8%</td><td>56.6%</td><td>33.6%</td><td>1.8%</td></tr></table>

Generation features show limited within-system associations with smoothing. From Table 11 we observe that the number of documents and report length vary across systems, which can influence the agents’ ability to synthesize and therefore SMOOTH-SCORE. We check if these features of an agent predict SMOOTHSCORE in Table 12. Across systems, systems that cite more documents $( \rho = 0 . 3 7 $ $p = . 0 5 6 )$ or write longer claims $( \rho = 0 . 4 3 , p = . 0 2 )$ smooth more, with Gemini at the extreme, but within a system both relationship are negligible $( \rho = 0 . 0 3$ and 0.02; per-system values are small and mixed in sign). Only report length shows a within system association, with a small $( \rho = 0 . 1 2 )$ but positive correlation in all systems, though not significant in any one alone. Number of claims shows no such connection within or across systems (except a modest negative correlation for Gemini, $\rho = - 0 . 2 3 )$ . Over-

Figure 3: Relationship between cited document count and claim smoothing. Across system and dataset pairs, we find that mean SMOOTHSCORE increases with the mean number of documents cited per report, but within a system this effect disappears (Table 12).

![](images/4cd98bc910e730e905c68f8057b65af580991a823017fa8caf237dc555e2730a.jpg)

Table 2: Examples of recurring claim-smoothing patterns. For each pattern, we show an illustrative claim, its SMOOTHSCORE, and a summary of the supporting evidence and limitations.
<table><tr><td>Pattern</td><td>Claim</td><td>Summarized Evidence</td></tr><tr><td>Unsupported composi- tion</td><td>The aggressive deployment of duplication-aware token pruning, Modality-Aware Mixture of Experts, and adaptive halting ensures that vast, world-class parameter capacities can be maintained without triggering corresponding explosions in inference latency. Score: 3</td><td>Token pruning, sparse expert routing, and adaptive halting can each reduce inference cost in particular settings. However, the evidence does not evaluate their simultaneous deployment or show that their benefits compose. Each can introduce routing, pruning, synchronization, or kernel overheads, so the combined system does not guarantee bounded latency at frontier scale.</td></tr><tr><td>Scope expansion</td><td>Self-attention-based models are highly parallelizable and require substantially fewer parameters, making them much more computationally efficient, less prone to overfitting, and easier to fine-tune for domain-specific tasks . . Score: 4</td><td>Some ViTs achieve strong performance with lower pre-training compute in particular comparisons, but these advantages do not generalize across models or regimes. ViTs often have comparable or larger parameter counts, global attention has quadratic cost, and ViTs can be more prone to overfitting in low-data settings without large-scale pre-training or careful regularization.</td></tr><tr><td>Premature closure</td><td>This innovation fundamentally resolves the statistical heterogeneity (Non-IID) problem while strictly preserving the privacy boundaries established by institutional frewalls. Score: 5</td><td>Federated distillation can mitigate heterogeneity and reduce direct data sharing, but it does not eliminate either problem. Performance continues to degrade as heterogeneity increases, theoretical analyses retain residual bias, and privacy attacks remain possible.</td></tr></table>

all, while these features show associations with smoothing, within-system behavior suggests that other factors could also influence smoothing.

## 4.2 QUALITATIVE OBSERVATIONS

Query framing influences SMOOTHSCORE. We find that query framing influences how the model writes its claims. As seen in Appendix G.1 and G.2, there are certain categories of questions where smoothing is more pronounced: (1) Comparative questions: if the query is comparative in nature, for example, ‘In multimodal pretraining, the paradigms are based on image tokens and stable diffusion, which appears more promising and why?’ or ‘Which paradigm between RL and distillation appears more promising and why?’, it can presuppose a choice from the agent where to motivate or justify a choice the agent drops important qualifications, (2) Topic overview questions: queries that require an overview of a topic, for example, ‘Describe the Sunflower conjecture, progress towards a solution, and related problems’, the agent can assemble multiple results a single conclusion or just generalize from a single finding to the whole topic.

Recurring Smoothing patterns across systems Table 2 shows three patterns that we observe across systems: (1) Unsupported composition: Agents combine individual fragments of evidence into a stronger conclusion. Although each component may be supported, the claim presents its unsupported composition as a conclusive statement. (2) Scope expansion: A claim takes an established result for a particular setup or experimental condition and removes certain boundaries, so that a the scope for an empirical observation expands into a general conclusion. (3) Premature Closure: Claims often adopt conclusive language, using phrases like ‘strictly preserving,’ ‘the best approach, ‘definitively proves’ and ‘most optimal.’.

Table 4: Revising reports generated by Luna and Sonnet 5 on ResearcherBench, scored against gold rubrics by GPT-5. Simply providing evidence can increase rubric scores, matching or exceeding providing raw document while being a fraction of the size
<table><tr><td rowspan="2">Condition</td><td colspan="3">Luna</td><td colspan="3">Sonnet 5</td></tr><tr><td>Context Length</td><td></td><td> $\Delta \left( p \right)$ </td><td>Context Length</td><td></td><td> $\Delta \left( p \right)$ </td></tr><tr><td>WebSearch draft (score)</td><td></td><td>1.0×</td><td>68.3</td><td></td><td>1.0×</td><td>64.4</td></tr><tr><td>Revise, no context</td><td></td><td>1.6×</td><td>+0.4 (.22)</td><td></td><td>1.5×</td><td>+1.1 (.38)</td></tr><tr><td>Revise + cited documents</td><td>117k</td><td>2.2×</td><td>+4.1 (.004)</td><td>324k</td><td>2.4×</td><td>+9.7 (&lt; .001)</td></tr><tr><td>Revise + SMOOTHSCORE evidence</td><td>11k</td><td>2.0×</td><td>+4.4 (&lt; .001)</td><td>65k</td><td>2.9×</td><td>+12.0 (&lt; .001)</td></tr></table>

## 5 VALIDATING SMOOTHSCORE

We validate SMOOTHSCORE by running a human study as well as checking the quality of evidence retrieved by our proposed pipeline.

## 5.1 HUMAN VALIDATION

Table 3: Agreement on SMOOTHSCORE across 73 claims scored by two human annotators.
<table><tr><td></td><td>Exact W/in 1</td><td></td><td>ρ</td><td>α</td></tr><tr><td>Annotator 1 vs. Annotator 2</td><td>58%</td><td>96%</td><td>0.72</td><td>0.71</td></tr><tr><td>Judge vs. Annotator 1</td><td>56%</td><td>93%</td><td>0.690.68</td><td></td></tr></table>

To validate SMOOTHSCORE, two annotators independently scored the same 73 claims using the same rubric given to the LLM judge (Appendix B.7), without access to the identity of the system that produced each claim. We find that agreement between the judge and Annotator 1 is

in a similar range to inter-annotator agreement, Table 3. Disagreements are predominantly small: the judge is within one point of Annotator 1’s score on 93% of claims and differs by two or more points on only 7%. Additionally, of the claims the judge scores $\geq 3 .$ 62% are also scored $\geq 3$ by Annotator 1, indicating that most judge-flagged claims reflect genuine issues. Since the two annotators themselves agree at $\kappa = 0 . 6 2$ at this threshold, the judge disagreement is expected given the subjective nature of the task.

## 5.2 RELEVANCE OF EVIDENCE COLLECTED BY SMOOTHSCORE

While human agreement validates the SMOOTHSCORE judge’s final scores, we separately check whether the evidence SMOOTHSCORE surfaces (Steps 2–4) is useful. We consider an independent measure, using gold rubrics provided by ResearcherBench created for judging the overall quality of the generated report. We take Luna and Sonnet 5 WebSearch reports on the benchmark, and perform an additional revision step with no web search under three context settings: (1) no additional context, (2) providing all cited documents, and (3) the collected SMOOTHSCORE evidence as context. We find that both context settings rise well above the no-context control (+0.4 and +1.1), with the extracted evidence improving rubric scores for both agents (Luna +4.4, Sonnet +12.0, both $p \textless$ .001), matching or even exceeding revision with the full documents. The evidence achieves this with roughly one-fifth to one-tenth of the tokens of the full documents. While this does not test the quality of the evidence relative to the generated claims, it shows that the evidence captures information that matters to an external, expert-written evaluation independent of the judge’s scores.

## 6 LACK OF ENGAGEMENT WITH DOCUMENT CONTENT

The systems we evaluate generate reports with citations from the documents they retrieve. However, the agent’s interaction with the cited material is hard to control. For Luna, the trace shows that the agent works almost entirely from search snippets truncated by the tool (median 585 characters), opening a cited page for only 19% of the documents it cites and in just 20 of 65 reports. Sonnet 5 engages even less, opening only 2% of the links it cites (Table 5, Appendix F). As agents rarely read the full documents, an agent may never see the qualifications and countervailing findings in the body of a source, which suggests that smoothing is partly shaped by what information the agent chooses to access.

Table 6: Effect of document engagement and calibration instructions on smoothing (Researcher-Bench, 61 questions). WebSearch: original reports. Docs: the same agent given the full text of the documents its WebSearch report cited, with web search off. + Instr.: the same calibration instructions added to either prompt. Providing both shows the largest improvement for both setups.
<table><tr><td></td><td></td><td></td><td></td><td></td><td colspan="5">% of claims with SMOOTHSCORE</td></tr><tr><td>Model</td><td>Condition</td><td>Length</td><td>Claims</td><td>Mean [95% CI]</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td rowspan="4">Luna</td><td>WebSearch</td><td>1.0×</td><td>416</td><td>1.95 [1.84, 2.07]</td><td>37.3</td><td>33.4</td><td>26.0</td><td>3.4</td><td>0.0</td></tr><tr><td>WebSearch + Instr.</td><td>1.0×</td><td>397</td><td>1.88 [1.78, 1.98]</td><td>40.3</td><td>34.0</td><td>23.2</td><td>2.5</td><td>0.0</td></tr><tr><td>Docs</td><td>1.6×</td><td>390</td><td>1.80 [1.70, 1.91]</td><td>45.6</td><td>31.0</td><td>21.3</td><td>2.1</td><td>0.0</td></tr><tr><td>Docs + Instr.</td><td>1.4×</td><td>378</td><td>1.52 [1.44, 1.60]</td><td>61.1</td><td>26.7</td><td>11.6</td><td>0.5</td><td>0.0</td></tr><tr><td rowspan="4">Sonnet 5</td><td>WebSearch</td><td>1.0×</td><td>407</td><td>2.54 [2.44, 2.63]</td><td>13.5</td><td>25.6</td><td>54.8</td><td>6.1</td><td>0.0</td></tr><tr><td>WebSearch + Instr.</td><td>1.1×</td><td>391</td><td>2.23 [2.12, 2.34]</td><td>25.6</td><td>30.2</td><td>40.4</td><td>3.8</td><td>0.0</td></tr><tr><td>Docs</td><td>1.2×</td><td>394</td><td>2.38 [2.27, 2.49]</td><td>18.3</td><td>31.0</td><td>45.4</td><td>5.3</td><td>0.0</td></tr><tr><td>Docs + Instr.</td><td>1.3×</td><td>367</td><td>1.78 [1.67, 1.88]</td><td>49.9</td><td>25.1</td><td>22.9</td><td>1.9</td><td>0.3</td></tr></table>

To test whether engaging with the documents reduces smoothing, we regenerate Luna and Sonnet 5 reports under varying setups. We vary (i) evidence access: the original WebSearch setting, or the original cited documents provided in context (cited docs in context), given in context with web search off; and (ii) instructions: the original prompt, or the same prompt plus calibration instructions. The instructions are identical across both evidence settings (prompts in Appendix E.4, E.3). We score all reports with SMOOTHSCORE on ResearcherBench.

We find that neither intervention is sufficient on its own (Table 6). For Luna, the instructions without the documents barely change smoothing (mean σ 1.95 to 1.88), and the documents alone reduce it modestly (1.80). Sonnet 5 responds more to instructions alone (2.54 to 2.23) but little to the documents alone (2.38). Combined, however, documents and instructions reduce smoothing far more than either does separately: mean σ falls to 1.52 for Luna and 1.78 for Sonnet 5, and the share of materially smoothed claims $( \sigma ~ \geq ~ 3 )$ drops from 29% to 12% and from 61% to 25%, removing more than half in both cases. Se-

Table 5: Luna and sonnet-5 websearch system engagement with the cited documents on ResearcherBench. Luna’s opens are marked by a header and Sonnet 5’s opens are explicit web fetches returning the full page. Details in Appendix F.
<table><tr><td colspan="2">Luna</td><td>Sonnet 5</td></tr><tr><td>Cited documents</td><td>496</td><td>1,458</td></tr><tr><td rowspan="2">Opened Not opened</td><td>95 (19%)</td><td>34 (2%)</td></tr><tr><td>401 (81%)</td><td>1,424 (98%)</td></tr><tr><td>Reports opening no page</td><td>45/65</td><td>44/65</td></tr></table>

vere smoothing (σ ≥ 4) nearly disappears (0.5% and 2.2%). Instructions to calibrate are thus most effective when the agent has the evidence needed to act on them.

Reports written from the full documents are longer (1.2 -1.6×) but contain slightly fewer extracted claims, so the reduction does not come from shorter or emptier reports. We also observe this qualitatively, the claims do not necessarily change the takeaway. Instead they tend to drop conclusive language supporting the scope of the evidence in documents and provide the required limitations as needed. Table 7 shows that the central takeaway remains the same and smoothing tracks the added reasoning.

Table 7: Claim improved under the docs+instruction case. Here the question pertains to how continual-learning systems can preserve previously learned knowledge while remaining adaptable to new tasks.The judge’s verdict is shown on the WebSearch claim.
<table><tr><td>Condition</td><td>Claim</td></tr><tr><td>WebSearch SmoothScore: 4</td><td>Protection should be soft and probabilistic, not permanently frozen: an important parameter may eventually be revised if sufficient contradictory evidence accumu-</td></tr><tr><td></td><td>lates.</td></tr><tr><td>Judge rationale</td><td>Requires recasting: the prescription excludes hard freezing and non-probabilistic approaches despite evidence supporting their effectiveness.</td></tr><tr><td>Docs + instructions SmoothScore: 1</td><td>Previously useful parameters or subnetworks should be protected from destructive updates. This can be implemented through frozen columns, masks, structural isola- tion, or soft importance-weighted constraints [doc 2][doc 3][doc 5].</td></tr></table>

## 7 RELATED WORK

Claim decomposition and aspect-based verification A standard recipe for verifying long-form text is to decompose it into smaller checkable units and score each against a knowledge source (Min et al., 2023; Chen et al., 2024; Gunjal & Durrett, 2024). These methods generally return a binary decision per claim, which is a good granularity for factual precision, but it does not work for synthesized claims that can have varying degrees of smoothing. ConflictScore (Liu et al., 2026) is the closest to our setting, and it extends this atomic claim recipe in a direction close to ours. They mark a claim as supported, contradicted or irrelevant per document retrieved, and aggregate these into claim-level conflict measures. However, a claim can be labeled supported by every one of its cited documents and still smooth over information it omits. Our work instead follows Kargupta et al. (2025) in organizing evidence along aspects (e.g., viewpoint, strength, certainty), so that supporting and qualifying evidence on the feature of a claim can be considered against each other.

Multi-Document Summarization In science communication, prior work measures certainty in how findings are stated (Pei & Jurgens, 2021) and models how information changes as findings move from papers to news and social media (Wright et al., 2022). For LLM summaries of scientific text, Peters & Chin-Yee (2025) test ten widely used LLMs on about 4,900 summaries and find that most produce broader generalizations than the source texts, and that LLM summaries are nearly five times more likely than human-written ones to overgeneralize. In medicine, neural multi-document summaries of clinical trials are fluent but often factually inaccurate (Wallace et al., 2021), and LLM summaries of medical evidence can misstate the certainty of the evidence or overgeneralize by omitting attributes such as a disease subtype (Tang et al., 2023). These studies summarize a single document or a small set of abstracts. We extend the concern to long-form, multi-document research reports written by agents, and give a claim-level measure grounded in the documents a report cites.

Correctness evaluation for deep research systems Agentic deep-research systems (Du et al., 2026; Xu et al., 2026; Huang et al., 2026; Li et al., 2026a; Sharma et al., 2025) pair rubric-based insight coverage with citation grounded factuality checks. Several works audit the reliability of these systems. DeepTRACE (Narayanan Venkit et al., 2026) decomposes answers into statements and finds that deep-research agents remain one-sided on debate queries and include many statements unsupported by their listed sources. Onweller et al. (2026) find that citations that resolve and are topically relevant often still fail a factual check against the cited content. Zhu et al. (2026) show that a single misleading document can induce false conclusions in final reports. Rasheed et al. (2026) argue for claim-level auditability, including whether conflicts in the evidence are surfaced or suppressed. Across this literature, factuality is largely operationalized as citation validity: whether a claim is supported by some retrieved source. Our results show that this is not enough. Reports whose claims are individually supported can still present mixed or conditional evidence as a decisive conclusion, so assessing how faithfully the underlying evidence is presented requires the claim-level, evidence-calibrated analysis we propose.

## 8 CONCLUSION

In this work, we studied the propensity of LLMs to smooth their conclusions when generating reports based on sources. They frequently generate oversimplified statements or conclusions that aren’t really backed up by the evidence cited. Our pipeline, SMOOTHSCORE, allows us to quantify this and establish its frequency across several state-of-the-art agents. Finally, we find that this models fail to read with their own cited documents, and suggest smoothing can improved by more thorough agents engaging wih their sources.

## AI USE STATEMENT

Generative AI formed the backbone of the systems studied in this work. In addition, code written with generative AI was used to conduct and carry out the analyses here. However, the research hypotheses in this work and the design of the experiments were both human-originated. AI tools were used for assistance with revisions, but not for writing in a substantive way. The authors take responsibility for the final content of this work.

## ACKNOWLEDGMENTS

This work was supported by NSF CAREER Award IIS-2145280, NSF grant IIS-2433071, the NSF AI Institute for Foundations of Machine Learning (IFML), and the NSF under Cooperative Agreement 2421782 and the Simons Foundation grant MPS-AI-00010515 awarded to the NSF-Simons AI Institute for Cosmic Origins — CosmicAI. This work was also partially supported by the Sloan Foundation and grants from Amazon and Open Philanthropy. The research has been supported by a compute grant from NVIDIA and computing support from the Torch cluster at NYU. We also thank the members of the TAUR lab at NYU for their helpful discussions and feedback.

## REFERENCES

Anthropic. Web fetch tool. https://platform.claude.com/docs/en/ agents-and-tools/tool-use/web-fetch-tool, 2026a. Claude API documentation. Accessed October 2026.

Anthropic. Web search tool. https://platform.claude.com/docs/en/ agents-and-tools/tool-use/web-search-tool, 2026b. Claude API documentation. Accessed October 2026.

Adithya Bhaskar, Alex Fabbri, and Greg Durrett. Prompted opinion summarization with gpt-3.5. In Findings ofthe associationfor computational linguistics: Acl 2023, pp. 9282–9300, 2023.

Jifan Chen, Grace Kim, Aniruddh Sriram, Greg Durrett, and Eunsol Choi. Complex Claim Verification with Evidence Retrieved in the Wild. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3569–3587, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.196. URL https://aclanthology.org/2024. naacl-long.196/.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Licheng Zhang, Xiaorui Wang, and Zhendong Mao. Deep-Research Bench: A Comprehensive Benchmark for Deep Research Agents. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=hQ0K2Hhq7H.

Alexander Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Anna Korhonen, David Traum, and Llu´ıs Marquez (eds.),\` Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 1074–1084, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1102. URL https://aclanthology. org/P19-1102/.

Anisha Gunjal and Greg Durrett. ”Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification”. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 3751–3768, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-emnlp.215. URL https://aclanthology.org/2024.findings-emnlp. 215/.

Yukun Huang, Leonardo FR Ribeiro, Momchil Hardalov, Bhuwan Dhingra, Markus Dreyer, and Venkatesh Saligrama. DeepFact: Co-Evolving Benchmarks and Agents for Deep Research Factuality. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 34356–34386, 2026.

Jena D Hwang, Varsha Kishore, Amanpreet Singh, Dany Haddad, Aakanksha Naik, Malachi Hamada, Jonathan Bragg, Mike D’Arcy, Daniel S Weld, Lucy Lu Wang, et al. Deep Research, Shallow Evaluation: A Case Study in Meta-Evaluation for Long-Form QA Benchmarks. arXiv preprint arXiv:2603.06942, 2026.

Priyanka Kargupta, Runchu Tian, and Jiawei Han. Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pp. 29664–29679, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/ 2025.acl-long.1434. URL https://aclanthology.org/2025.acl-long.1434/.

Haiwen Li and Sinan Aral. Human Trust in AI Search: A Large-Scale Experiment. arXiv preprint arXiv:2504.06435, 2025.

Ruizhe Li, Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. Deep-Research Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Reports, 2026a. URL https://arxiv.org/abs/2601.08536.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research capability. Advances in Neural Information Processing Systems, 38:120091–120131, 2026b.

Siyi Liu, Aaron Halfaker, Dan Roth, and Patrick Xia. ConflictScore: Identifying and Measuring How Language Models Handle Conflicting Evidence. arXiv preprint arXiv:2606.26437, 2026.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, pp. 12076–12100, 2023.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Kung-Hsiang Huang, Yixin Mao, and Chien-Sheng Wu. DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence. In International Conference on Learning Representations, volume 2026, pp. 106041–106062, 2026.

Hailey Onweller, Elias Lumer, Austin Huber, Pia Ramchandani, Vamse Kumar Subbiah, and Corey Feld. Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents. arXiv preprint arXiv:2605.06635, 2026.

OpenAI. Web search. https://developers.openai.com/api/docs/guides/ tools-web-search, 2026. OpenAI API documentation. Accessed October 2026.

Jiaxin Pei and David Jurgens. Measuring sentence-level and aspect-level (un)certainty in science communications. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wentau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 9959–10011, Online and Punta Cana, Dominican Republic, November

2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.784. URL https://aclanthology.org/2021.emnlp-main.784/.

Uwe Peters and Benjamin Chin-Yee. Generalization Bias in Large Language Model Summarization of Scientific Research. Royal Society Open Science, 12(4):241776, 2025.

Razeen A Rasheed, Somnath Banerjee, Animesh Mukherjee, and Rima Hazra. From Fluent to Verifiable: Claim-Level Auditability for Deep Research Agents. arXiv preprint arXiv:2602.13855, 2026.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, and Pang Wei Koh. DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=97NEP1pyS3.

Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, and Bing Liu. ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents, 2025. URL https://arxiv.org/abs/2511.07685.

Yixiao Song, Yekyung Kim, and Mohit Iyyer. VeriScore: Evaluating the factuality of verifiable claims in long-form text generation. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 9447–9474, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.552. URL https://aclanthology.org/2024. findings-emnlp.552/.

Melanie Subbiah, Akankshya Mishra, Grace Kim, Liyan Tang, Greg Durrett, and Kathleen McKeown. Is the Top Still Spinning? Evaluating Subjectivity in Narrative Understanding. In Chris tos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 185– 203, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979- 8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.10. URL https://aclanthology. org/2025.emnlp-main.10/.

Liyan Tang, Zhaoyi Sun, Betina Idnay, Jordan G Nestor, Ali Soroush, Pierre A Elias, Ziyang Xu, Ying Ding, Greg Durrett, Justin F Rousseau, et al. Evaluating Large Language Models on Medical Evidence Summarization. NPJ digital medicine, 6(1):158, 2023.

Liyan Tang, Philippe Laban, and Greg Durrett. MiniCheck: Efficient fact-checking of LLMs on grounding documents. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8818–8847, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.499. URL https://aclanthology.org/2024. emnlp-main.499/.

Byron C Wallace, Sayantan Saha, Frank Soboczenski, and Iain J Marshall. Generating (Factual?) Narrative Summaries of RCTs: Experiments with Neural Multi-Document Summarization. AMIA Summits on Translational Science Proceedings, 2021:605, 2021.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-Token Prediction is All You Need. CoRR, abs/2409.18869, 2024. URL https://doi.org/10.48550/arXiv.2409.18869.

Dustin Wright, Jiaxin Pei, David Jurgens, and Isabelle Augenstein. Modeling information change in science communication with semantically matched paraphrases. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in

Natural Language Processing, pp. 1783–1807, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.117. URL https://aclanthology.org/2022.emnlp-main.117/.

Jian Xie, Tianhe Lin, Zilu Wang, Yuting Ning, Yuekun Yao, Tianci Xue, Zhehao Zhang, Zhongyang Li, Kai Zhang, Yufan Wu, et al. QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks. arXiv preprint arXiv:2605.24218, 2026.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation, 2025. URL https: //arxiv.org/abs/2408.12528.

Tianze Xu, Pengrui Lu, Lyumanshan Ye, Xiangkun Hu, and Pengfei Liu. ResearcherBench: Evaluating Deep AI Research Systems on the Frontiers of AI Research, 2026. URL https: //openreview.net/forum?id=oj6A9hrNdL.

Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu. Alignscore: Evaluating factual consistency with a unified alignment function. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 11328–11348, 2023.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model, 2024. URL https://arxiv.org/abs/2408. 11039.

Pengyu Zhu, Lijun Li, Longju Yang, Sen Su, and Jing Shao. Is Deep Research Reliable? Misleading Knowledge Induces False Conclusions. arXiv preprint arXiv:2607.20891, 2026.

## A SMOOTHSCORE

## A.1 METHOD DETAILS

Aspect Induction As mentioned in Section 2, the third step in our pipeline is Aspect Induction. Since aspect proposals are generated separately for each claim–document pair, the same underlying topic is often proposed by several documents under different names or with different levels of specificity. Merging is needed so that evidence on the same topic is grouped together, rather than being split across near-duplicate aspects.

For each claim $c _ { i } .$ , we consolidate the document-level proposals $\{ A _ { i j } \} _ { j = 1 } ^ { m }$ into a hierarchy

$$
T _ { i } = ( V _ { i } , L _ { i } ) = \mathrm { M e r g e } \left( \{ A _ { i j } \} _ { j = 1 } ^ { m } \right) ,
$$

where $V _ { i }$ is the set of aspect and subaspect nodes and $L _ { i }$ is the set of parent–child links between them. The merge step (i) combines semantically overlapping proposals across documents into a single node, and (ii) introduces subaspects when one aspect covers distinct but related findings. Each node retains the set of documents that proposed it, so that every piece of evidence can be traced back to its source.

Pipeline Models Table 8 shows the models used for every stage of the pipeline for generating SMOOTHSCORE. All pipeline stages use GPT-5-mini except SMOOTHSCORE judge (Step 5), which use GPT-5; every call uses reasoning effort=medium with default sampling.

## A.2 DATASET CURATION

We use all 65 ResearcherBench questions and the complete 25-question expert-written ScholarQA-CS2 set. These by construction contained open ended questions that required multi document engagement. For DeepResearch Bench II and ResearchRubrics:

• Candidate pool. We take the 66 English DeepResearch Bench II tasks and all 101 ResearchRubrics tasks.

Table 8: Models used at each stage of the SMOOTHSCORE pipeline. All stages use GPT-5-mini except the final judge, which uses GPT-5. Prompts for each stage are given in the referenced appendix sections.
<table><tr><td>Step</td><td>Stage</td><td>Model</td><td>Prompt</td></tr><tr><td>1</td><td>Claim extraction</td><td>GPT-5-mini</td><td>App. B.1</td></tr><tr><td>2</td><td>Evidence extraction: support agent</td><td>GPT-5-mini</td><td>App. B.2</td></tr><tr><td>2</td><td>Evidence extraction: skeptic agent</td><td>GPT-5-mini</td><td>App. B.3</td></tr><tr><td>3</td><td>Aspect induction: per-document proposals</td><td>GPT-5-mini</td><td>App. B.4</td></tr><tr><td>3</td><td>Aspect induction: cross-document merge</td><td>GPT-5-mini</td><td>App. B.5</td></tr><tr><td>4</td><td>Evidence assignment</td><td>GPT-5-mini</td><td>App. B.6</td></tr><tr><td>5</td><td>SMOOTHSCORE judge</td><td>GPT-5</td><td>App. B.7</td></tr></table>

• Annotation. We label each query on synthesis (HIGH / MED / LOW), meaning whether a good answer must synthesize a body of literature. We also label openness of question type (HIGH / MED / LOW), meaning whether the question requires one definitive correct response.

• Filter. We keep queries with synthesis = HIGH and Question Type = HIGH (Table 9).

Table 9: Query selection for DeepResearch Bench II and ResearchRubrics.
<table><tr><td>Benchmark</td><td>Candidates</td><td>Kept</td></tr><tr><td>DeepResearch Bench II (English)</td><td>66</td><td>18</td></tr><tr><td>ResearchRubrics</td><td>101</td><td>17</td></tr></table>

## A.3 MODELS

Table 10 contains model IDs of the evaluated systems. We evaluate 7 systems consisting of open sourced and close source agents including frontier models.

Table 10: Model identifiers and access method for each evaluated system.
<table><tr><td>System</td><td>Model ID</td><td>Access</td></tr><tr><td>GPT Luna + Websearch</td><td>gpt-5.6-luna</td><td>OpenAI Responses API</td></tr><tr><td>GPT Sol + Websearch</td><td>gpt-5.6-sol</td><td>OpenAI Responses API</td></tr><tr><td>Claude Sonnet 5 + Websearch</td><td>claude-sonnet-5</td><td>Anthropic Messages API</td></tr><tr><td>Claude Opus 5 + Websearch</td><td>claude-opus-5-5</td><td>Anthropic Messages API</td></tr><tr><td>Gemini Deep Research</td><td>deep-research-preview-04-2026</td><td>Gemini Interactions API</td></tr><tr><td>DR Tulu</td><td>rl-research/DR-Tulu-8B</td><td>local (vLLM + MCP server)</td></tr><tr><td>QUEST</td><td>QUEST-35B-RL</td><td>local</td></tr></table>

## A.4 RELATIONSHIP BETWEEN SMOOTHSCORE AND GENERATION FEATURES

Table 11 contains report length and retrieved evidence by dataset and system and Table 12 shows if such features can predict SmoothScore with and across systems. We find that while systems show connection to different features across systems these relationships are weak and vary in direction, with only report length showing a small association.

Table 11: Report length and retrieved evidence by dataset and system, over the reports scored by the SmoothScore judge. “Report words” and “Evidence words” are medians over reports; “Docs/report” is the mean number of retrieved documents per report. “Evidence words” is the total number of words in the retrieved documents available to the system.
<table><tr><td>Dataset</td><td>System</td><td>Report words</td><td>Docs/report</td><td>Evidence words</td></tr><tr><td rowspan="7">ResearcherBench</td><td>Luna 5.6</td><td>1,828</td><td>8.6</td><td>77,898</td></tr><tr><td>Sol 5.6</td><td>1,710</td><td>9.9</td><td>92,880</td></tr><tr><td>Sonnet 5</td><td>2,125</td><td>21.6</td><td>155,148</td></tr><tr><td>Opus 5.5</td><td>1,328</td><td>18.8</td><td>138,441</td></tr><tr><td>QUEST</td><td>1,332</td><td>9.8</td><td>55,481</td></tr><tr><td>DR Tulu</td><td>1,948</td><td>17.4</td><td>114,027</td></tr><tr><td>Gemini</td><td>4,860</td><td>39.2</td><td>254,916</td></tr><tr><td rowspan="7">CS2-Expert</td><td>Luna 5.6</td><td>1,999</td><td>10.4</td><td>34,821</td></tr><tr><td>Sol 5.6</td><td>1,514</td><td>10.8</td><td>69,483</td></tr><tr><td>Sonnet 5</td><td>2,177</td><td>18.5</td><td>175,065</td></tr><tr><td>Opus 5.5</td><td>1,431</td><td>19.7</td><td>232,211</td></tr><tr><td>QUEST</td><td>1,210</td><td>8.6</td><td>46,954</td></tr><tr><td>DR Tulu</td><td>2,156</td><td>17.8</td><td>92,667</td></tr><tr><td>Gemini</td><td>5,122</td><td>48.6</td><td>253,233</td></tr><tr><td rowspan="8">DRB-II</td><td>Luna 5.6</td><td>3,886</td><td>14.6</td><td>82,187</td></tr><tr><td>Sol 5.6</td><td>3,979</td><td>20.2</td><td>162,523</td></tr><tr><td>Sonnet 5</td><td>3,432</td><td>32.4</td><td>192,348</td></tr><tr><td>Opus 5.5</td><td>2,786</td><td>31.2</td><td>205,332</td></tr><tr><td>QUEST</td><td>2,397</td><td>11.2</td><td>40,119</td></tr><tr><td>DR Tulu</td><td>2,884</td><td>14.4</td><td>110,388</td></tr><tr><td>Gemini</td><td>4,844</td><td>32.7</td><td>235,720</td></tr><tr><td>Luna 5.6</td><td></td><td></td><td></td></tr><tr><td rowspan="7">ResearchRubrics</td><td>Sol 5.6</td><td>3,128</td><td>10.0</td><td>60,175</td></tr><tr><td>Sonnet 5</td><td>2,736</td><td>11.9</td><td>83,795</td></tr><tr><td>Opus 5.5</td><td>3,031</td><td>28.6</td><td>199,191</td></tr><tr><td></td><td>2,010</td><td>22.7</td><td>134,858</td></tr><tr><td>QUEST</td><td>1,930</td><td>9.6</td><td>52,607</td></tr><tr><td>DR Tulu</td><td>2,732</td><td>15.0</td><td>106,390</td></tr><tr><td>Gemini</td><td>4,687</td><td>31.2</td><td>196,485</td></tr></table>

Table 12: Correlation of report and claim level features with SMOOTHSCORE across and within systems (Spearman ρ and Pearson r). Across: the mean over the 28 system–dataset. Within: after subtracting each system–dataset mean. Systems that cite more documents or write longer claims smooth more, but within a system such relationships are weak and vary in direction. Only report length shows a small, consistent within-system association (Panel B)
<table><tr><td colspan="5">A. Pooled</td></tr><tr><td rowspan="2">Variable</td><td colspan="2">Across systems</td><td rowspan="2">Within system ρ</td><td rowspan="2">Raw pooled ρ</td></tr><tr><td>ρ</td><td>r</td></tr><tr><td>Cited documents</td><td>+0.37 (.056)</td><td>+0.59 (.001)</td><td>+0.03 (.35)</td><td>+0.38</td></tr><tr><td>Report length</td><td>+0.24 (.23)</td><td>+0.39 (.04)</td><td>+0.12 (.001)</td><td>+0.33</td></tr><tr><td>Number of claims</td><td>+0.08 (.70)</td><td>+0.10 (.63)</td><td>−0.02 (.61)</td><td>0.00</td></tr><tr><td>Claim length</td><td>+0.43 (.02)</td><td>+0.56 (.002)</td><td>+0.02 (.16)</td><td>+0.12</td></tr></table>

<table><tr><td>Variable</td><td>Luna</td><td>Sol</td><td>Sonnet 5</td><td>Opus 5.5</td><td>QUEST</td><td>DR Tulu</td><td>Gemini</td></tr><tr><td>Cited documents</td><td>+0.14 (.12)</td><td>+0.27 (.003)</td><td>+0.03 (.77)</td><td>0.00 (.96)</td><td>-0.14 (.11)</td><td>-0.19 (.03)</td><td>+0.09 (.35)</td></tr><tr><td>Report length</td><td>+0.17 (.06)</td><td>+0.14 (.12)</td><td>+0.12 (.17)</td><td>+0.14 (.12)</td><td>+0.02 (.81)</td><td>+0.16 (.08)</td><td>+0.06 (.50)</td></tr><tr><td>Number of claims</td><td>-0.04 (.69)</td><td>+0.05 (.57)</td><td>0.00 (.96)</td><td>+0.16 (.08)</td><td>-0.11 (.22)</td><td>+0.01 (.90)</td><td>-0.23 (.01)</td></tr><tr><td>Claim length</td><td>-0.04 (.30)</td><td>-0.01 (.85)</td><td>+0.08 (.02)</td><td>+0.06 (.09)</td><td>-0.09 (.009)</td><td>+0.08 (.02)</td><td>+0.02 (.54)</td></tr></table>

## B SMOOTHSCORE PROMPTS

## B.1 CLAIM GENERATION

Prompt B.1: Main Conclusion Extraction   
System Prompt:   
You identify the MAIN CONCLUSIONS a research report reaches in answering a question – the handful of   
load-bearing verdicts a reader walks away with – and return each one as a COMPLETE verbatim passage   
from the report. You are given the question and the report’s answer.   
BE SELECTIVE. Extract only the report’s main conclusions: the thesis-level takeaways that directly answer   
the question. A typical report has roughly 4-8 of these. Favor the verdicts the whole report builds toward   
over the individual findings that support them.   
UNIT OF EXTRACTION – the complete conclusion, NOT an atom:   
1. Lift each conclusion WHOLE: typically 1-3 consecutive sentences (roughly 15-80 words). The claim   
MUST include, whenever they appear in the same or immediately adjacent sentences: every hedge and   
qualifier; the flip-side, contrast, or counterpart condition (”; it does not ...”, ”However, ...”, ”but ...”); and the   
stated scope or dimension the verdict applies to. Emitting the assertive half of a hedged or paired statement   
is the worst possible error – if the pieces cannot be included coherently, skip the conclusion entirely.   
2. CARRY THE SCOPE. When a verdict’s scope lives just OUTSIDE its sentence – in the immediately   
governing heading, a dimension label, or the sentence before – stitch that report wording in front (join   
stitched spans naturally; the heading text is report wording). NEVER present a per-dimension or per  
condition winner as a global verdict.   
3. NEVER extract a heading, list label, or table fragment as a claim, even when it is bolded and reads like a   
verdict. Headings point to claims; take the claim from the body text.   
VOICE – verbatim about the assertion, silent about the sourcing:   
4. REUSE THE REPORT’S OWN WORDING at the report’s own strength. Keep every hedge, qualifier,   
comparative, superlative, and number exactly as written. Never strengthen, weaken, or paraphrase the   
assertion.   
5. STRIP ATTRIBUTION AND CITATION FRAMING: drop ”X’s experiments found that”, ”according to   
...”, ”studies show”, citation markup, and markdown link syntax. KEEP a system/method name only when   
the claim is ABOUT that system.   
6. STRIP DISCOURSE CONNECTIVES at the very start of a claim (”On the other hand,”, ”In summary,”,   
”However,”, ”Overall,”) – but connectives INSIDE the claim stay; they are doing real work joining the   
verdict to its counterpart.   
WHAT COUNTS AS A CONCLUSION – all three required:   
- A conclusion the report draws, not a building block: a result, comparison, judgment, trade-off, trend, or   
recommendation that answers the question.   
- Substantive, not a bare definitional or mechanistic statement.   
- Contentful and checkable: it asserts something specific enough to check against evidence. A forecast or   
recommendation counts, but the claim must preserve the report’s framing of it as a forecast/recommendation.   
GRANULARITY – one claim per distinct conclusion or thread. Different claims must not restate the same   
verdict; together they should cover each side’s main strengths, the main limitations or trade-offs, and the   
overall verdict if the report gives one.   
SKIP: background, definitions, setup, motivation, restatements of the question, section framing, individual   
supporting findings that roll up into a larger verdict.   
OUTPUT – return only JSON with this exact schema:   
{"claims": [{"text": "{{Claim Text }}", "source text": "{{Verbatim   
Source Span(s) }}"}]}   
For each claim:

- ”text”: the complete conclusion passage, following the rules above. Plain prose – no markdown, no citation   
markup, no list numbering.   
- ”source text”: the exact verbatim span(s) of the answer the claim was lifted from, copied character-for  
character (citation markup and markdown syntax may be dropped). If the claim stitches spans (e.g. heading   
scope + body), join them with ” ... ” in report order. Each span must be findable in the answer by string   
search.   
FINAL CHECKLIST – verify EVERY claim’s ”text” passes all of these; rewrite it if not:   
- [ ] Complete sentence(s) – never a fragment, bullet label, heading, or table cell.   
- [ ] Includes the local hedges, scope, and flip-side that appear in or immediately beside the source passage.   
- [ ] No attribution framing, no markdown (<sub>\*\*</sub>, <sub>\*</sub>, -, #, >, |), no citation markup, no URLs, no list   
numbering.   
- [ ] The assertion is the report’s wording at the report’s strength – qualifiers, comparatives, numbers un  
changed.   
If the answer states no checkworthy conclusions, return {”claims”: []}.   
User Prompt:   
Example ID: {{Example ID }}   
Question: {{Question }}   
Extract the MAIN CONCLUSIONS of the following final response – the handful of load-bearing, evidence  
based verdicts that answer the question (typically 3–7). For each, return the claim text (the assertion in the   
report’s own words at the report’s own strength, with all attribution, citation, discourse-connective, and   
markdown framing STRIPPED, per the voice rules and final checklist) and the verbatim source text span(s)   
it was lifted from. Use the question to decide which conclusions are load-bearing.   
final response:   
{{Full Report Text }}

## B.2 SUPPORTING PASS

Prompt B.2: Supporting Evidence Extraction   
System Prompt:   
You extract from a single source DOCUMENT every distinct piece of information that SUPPORTS a single   
CLAIM from a research report, or a specific part of it – makes what the claim actually says more plausible   
or provides a direct instance/evidence for it.   
Extract ONLY supporting evidence. Ignore evidence against the claim entirely; another process handles   
that.   
• POLARITY: support means supporting what the claim ACTUALLY ASSERTS. If the claim as  
serts a limitation or a negative (e.g. “X is unreliable”, “X scales worse”, “X does not generalize”),   
then evidence of X failing/underperforming is supporting, and a result showing X working well is   
NOT supporting – skip it.   
• STATED CONCESSIONS ARE PART OF THE CLAIM: evidence confirming a limitation, con  
dition, or concession the claim itself states supports the claim as stated – include it, with the   
concession named in the aspect field.   
An item is one distinct FINDING: a specific result, experiment, measurement, or stated conclusion. Each   
item is an object:   
{"quote": "<the shortest verbatim excerpt that carries this finding>",   
"aspect": "<the specific assertion WITHIN the claim this finding bear   
on, quoted or near-verbatim from the claim>",   
"note": "<one sentence: how this finding bears on that assertion>",   
"strength": <1-5>}   
strength = how directly and specifically this item bears on what the   
claim asserts:   
1 = tangential 2 = weak/indirect 3 = moderate: relevant but partial   
or hedged

4 = strong: direct, on-point 5 = decisive: exactly what the claim   
asserts (or a clear counterexample to it).   
Critical rules:   
• Judge ONLY against this one document. No outside knowledge, no other documents.   
• Read the WHOLE document; relevant evidence can sit in any section.   
• ONE item per finding. If the document states the same finding several ways (abstract, body,   
conclusion), pick the single best quote for it – do not list restatements or different facets of the   
same result as separate items.   
• Do NOT include items that only establish setting or applicability without an outcome (e.g. “the   
experiments use a 1.5B model”). That is context, not evidence; if it matters, mention it inside the   
note of a real item.   
• An empty list is a perfectly good answer – do NOT invent items. Merely being about the same   
topic is NOT evidence; every item must actually move belief about the claim in the direction you   
were asked for.   
• MECHANISM: only count evidence that matches what the claim specifically says (e.g. if the   
claim says “distillation from a LARGER model”, a result using a same-size or self-distilled model   
does not bear on that specific mechanism). The aspect field must name the exact assertion ad  
dressed – if you cannot point to one, the item does not belong.   
• Quotes must be copied verbatim from the document.   
Return only JSON: {"items": [ ...]}   
User Prompt:   
Question (context only, not evidence): {{Question }}   
Claim: {{Claim }}   
Document:   
Title: {{Document Title }}   
URL: {{Document URL }}   
Content:   
{{Full Document Text }}   
Extract every distinct piece of supporting evidence and return only the JSON object.

## B.3 SKEPTIC PASS

Prompt B.3: Tagged Skeptical Evidence Extraction   
System Prompt:   
You extract from a single source DOCUMENT every distinct piece of information that a SKEPTICAL   
reader would weigh against a single CLAIM from a research report, and you TAG each item by its relation   
to the claim AS STATED. Extract only this skeptical side; a separate process handles supporting evidence.   
Read the claim carefully first: its actual assertions, its scope, its comparatives, and every limitation, condi  
tion, or concession it ALREADY STATES or hedges over.   
TAGS:   
• "counter": the finding cuts against what the claim ACTUALLY ASSERTS – contradicts it,   
provides a counterexample, limits its scope beyond what it admits, shows the asserted effect   
weaker or more conditional than asserted, or reports a null or mixed result where the claim implies   
a clear one. POLARITY: if the claim asserts a limitation or a negative (“X is unreliable”), a result   
showing X working WELL is a counter, and evidence of X failing is NOT. A soft hedge does   
not concede everything: “X is usually best” still asserts usually – evidence X OFTEN loses is a   
counter; evidence X merely SOMETIMES loses is not.   
• "concession-match": the finding confirms, instantiates, or quantifies a limitation, condi  
tion, or concession the claim ITSELF states or hedges over (“capacity-sensitive”, “results vary”,   
“except in short contexts”). These do NOT count against the claim – they are extracted so a later   
audit can check whether the report states the substance behind its own hedges.   
• "rival-win": a success of ANOTHER approach reported in that approach’s own terms, where   
the claim does NOT rank over or exclude that approach. If the claim DOES rank or exclude   
(“strongest”, “most practical”, “X remains ahead of Y”), a rival’s success is a "counter", not a   
"rival-win".

A carefully scoped claim often has NO counter items in a given document – an empty counter set is a   
common and correct outcome; do not stretch. But still extract concession-match and rival-win items where   
they exist.   
• STRENGTH for "counter" = how much the finding should CONSTRAIN the claim as stated:   
5 = directly contradicts the central assertion or shows it failing within its own stated scope; 4 =   
a hard limit or condition on the main assertion that the claim does not state; 3 = a meaningful   
qualification; 2 = weak/indirect pressure; 1 = tangential.   
• STRENGTH for "concession-match" = how load-bearing the conceded condition is: 5 =   
the concession hides a result that would change most readers’ decisions; 3 = a substantive but   
secondary condition; 1 = a trivial caveat.   
• STRENGTH for "rival-win" = how strong the rival result is in its own terms (controlled   
comparison > own-harness leaderboard > assertion).   
An item is one distinct FINDING: a specific result, experiment, measurement, or stated conclusion. Each   
item is an object:   
{"quote": "<the shortest verbatim excerpt that carries this finding>",   
"aspect": "<the specific assertion WITHIN the claim this finding bear   
on, quoted or near-verbatim from the claim>",   
"note": "<one sentence: how this finding bears on that assertion>",   
"tag": "counter|concession-match|rival-win",   
"strength": <1-5>}   
strength = how directly and specifically this item bears on what the   
claim asserts:   
1 = tangential 2 = weak/indirect 3 = moderate: relevant but partial   
or hedged   
4 = strong: direct, on-point 5 = decisive: exactly what the claim   
asserts (or a clear counterexample to it).   
Critical rules:   
• Judge ONLY against this one document. No outside knowledge, no other documents.   
• Read the WHOLE document; relevant evidence can sit in any section.   
• ONE item per finding. If the document states the same finding several ways (abstract, body,   
conclusion), pick the single best quote for it – do not list restatements or different facets of the   
same result as separate items.   
• Do NOT include items that only establish setting or applicability without an outcome (e.g. “the   
experiments use a 1.5B model”). That is context, not evidence; if it matters, mention it inside the   
note of a real item.   
• An empty list is a perfectly good answer – do NOT invent items. Merely being about the same   
topic is NOT evidence; every item must actually move belief about the claim in the direction you   
were asked for.   
• MECHANISM: only count evidence that matches what the claim specifically says (e.g. if the   
claim says “distillation from a LARGER model”, a result using a same-size or self-distilled model   
does not bear on that specific mechanism). The aspect field must name the exact assertion ad  
dressed – if you cannot point to one, the item does not belong.   
• Quotes must be copied verbatim from the document.   
Return only JSON: {"items": [ ...]}   
User Prompt:   
Question (context only, not evidence): {{Question }}   
Claim: {{Claim }}   
Document:   
Title: {{Document Title }}   
URL: {{Document URL }}   
Content:   
{{Full Document Text }}   
Extract every distinct piece of countervailing evidence and return only the JSON object.

## B.4 ASPECT GENERATION

Prompt B.4: Document-Level Aspect Induction   
System Prompt:   
You are an analyst that identifies the different aspects that a single scientific research paper ACTUALLY   
DISCUSSES which bear on evaluating a claim. For each aspect, provide a brief all-lowercase label (3–8   
words), a one-sentence description of what THIS DOCUMENT says about it and why it is significant for   
addressing the claim, and 6 keywords typically used to describe that aspect.   
Rules (corpus-grounded, claim-relative):   
• Only include aspects this document actually contains substantive content on—do not invent as  
pects from general knowledge of the topic.   
• Frame each aspect as a dimension bearing on THIS claim (e.g., “task-type dependence of the   
comparison”), not a generic subtopic of the field.   
• Return as many aspects as the document genuinely supports: typically 2–3.   
One aspect is a good answer for a narrowly relevant document, and an EMPTY LIST is the correct answer   
for a document with nothing substantive on this claim. Do not pad to a count.   
Return only JSON: {"aspect list": [{"aspect label": "...",   
"aspect description": "...", "aspect keywords": ["..."]}]}.   
User Prompt:   
CLAIM: {{Claim }}   
DOCUMENT #{{Document Index }} (title: {{Document Title }}):   
{{Full Document Content }}   
Return only the JSON object.

## B.5 ASPECT MERGE

Prompt B.5: Aspect Hierarchy Consolidation   
System Prompt:   
You are an analyst consolidating per-document aspect proposals into ONE aspect hierarchy for evaluating   
a claim. You will identify 4–5 top-level aspects that the documents’ proposals are collectively considering   
when determining the answer to the claim. Add subaspects ONLY where the corpus genuinely subdivides   
an aspect into distinct, separately-evidenced points—a top-level aspect with NO subaspects is a good and   
common answer. Aim for roughly 4–8 nodes in the whole hierarchy, not the maximum. Each subaspect   
must name what distinguishes it from its siblings; if you cannot, merge them.   
Rules:   
• Corpus-grounded: every aspect/subaspect must be traceable to the per-document proposals; carry   
over the source document indexes into "source docs". Do not invent nodes no document   
proposed.   
• Claim-relative: keep labels framed as dimensions bearing on the claim.   
• Merge duplicates across documents; prefer the most specific phrasing proposed.   
• A subaspect is a more specific point underneath its parent that would be considered when specifi  
cally addressing the claim.   
Return only JSON: {"aspects": [{"label": "...", "description": "...",   
"source docs": [1,2], "subaspects": [{"label": "...", "description":   
"...", "source docs": [3]}]}]}.   
User Prompt:   
CLAIM: {{Claim }}   
PER-DOCUMENT ASPECT PROPOSALS:   
[doc 1] {{Aspect Label }}: {{Aspect Description }}   
[doc 1] {{Aspect Label }}: {{Aspect Description }}   
[doc 3] {{Aspect Label }}: {{Aspect Description }}   
Return only the JSON object.

## B.6 EVIDENCE ASSIGNMENT

Prompt B.6: Evidence-to-Aspect Assignment   
System Prompt:   
You will be provided with evidence quotes extracted from source documents. Each quote was extracted as   
evidence about a claim. Please select the suitable aspect each quote talks about from the provided aspect   
list.   
LEAF-ONLY RULE: a node that has sub-aspects is a container only. If the quote belongs in such an aspect’s   
territoryts sub-aspects (thebest-fitting one), never the parent. Only aspects with no sub-aspects can receive   
quotes directly.   
Special destinations:   
• “ROOT” if the quote bears oher than any one aspect(overall verdicts, provenance of the whole   
comparison, conditions spanning aspects)ROOT is a deliberate destinat   
• “UNASSIGNED” only if the quote fits no listed aspect and is not whole-claim evidence.   
Assignment is topical: judge what the quote is ABOUT, never whether it supports or opposes the claim.   
Each quote is prefixed with its source document (index and title). Use it as cona fragment (a table row, a bare   
number, a one-line result) is about whatever that document discusses at that point – file it by its document’s   
topic, not by the words in the fragment alone.   
Return only JSON: {"assignmen "node": "<node id like A1 orA2.3 or ROOT or   
UNASSIGNED>"}]}.   
User Prompt:   
CLAIM: {{Claim }}   
ASPECT LIST:   
A1: {{Aspect Label }}   
A1.1: {{Aspect Label \(>\) Subaspect Label }}   
A2: {{Aspect Label }}   
. . .   
EVIDENCE:   
[0] (doc 3 — {{Title, 90 Charsned, 400 Chars }})   
[1] (doc 7 — {{Title }}) {{Quote }}

## B.7 SMOOTHSCORE JUDGE

Prompt B.7: Evidence-Smoothing Judgment   
System Prompt: You are an evidence-smoothing judge. You assess how faithfully a claim compresses the   
evidence supplied to it — not whether the claim is well-written, complete, or true by outside standards.   
User Prompt: # Inputs   
Research question: {{QUESTION}}   
Claim under evaluation: {{CLAIM}}   
Evidence by aspect: {{ASPECTS AND EVIDENCE}}   
Each evidence item may include: a source identifier, an exact quotation, a generated context note, and a   
retrieval-pass label (”support-pass candidate” / ”countervailing-pass candidate”).   
# What you’re measuring   
Evidence smoothing: the claim collapses material heterogeneity, uncertainty, conditionality, or disagree  
ment in the evidence, producing a conclusion broader, more certain, more universal, or more decisive than   
the evidence proportionately warrants.   
Not in scope: coverage (the claim need not mention every finding), writing quality, or accuracy against   
knowledge outside the supplied evidence.   
# Ground rules for reading the evidence   
1. \*\*The supplied evidence is the entire universe.\*\* No outside knowledge.   
2. \*\*Retrieval labels describe provenance, not verified stance.\*\* Interpret each item’s actual content   
independently — a support-pass item can fail to support the claim; a countervailing-pass item can support,   
qualify, or be irrelevant to it.   
3. \*\*Item counts and pass labels are not evidence of consensus or its absence.\*\* Never infer balance, or   
lack of it, from how many items landed in each pass.   
4. \*\*Context notes clarify; they don’t add evidence.\*\* Ground judgments in the quotation. Preserve the

claim’s actual scope, certainty, and role in answering the question. Do not weaken a strong assertion into a vague statement of usefulness, or strengthen a qualified claim into an unconditional guarantee.

5. \*\*Weigh by relevance, directness, applicability, and any quality signal stated in the text — not by quote count.\*\* Restatements of the same finding are one data point, not several.

6. \*\*The claim’s own logical force sets what can defeat it:\*\* - ”X can work” — isolated failures of X don’t count against it. - ”X always / necessarily works” — one sound counterexample is decisive. - ”X generally works / dominates” — isolated counterexamples aren’t enough; the evidence must speak to prevalence or the overall comparison. - ”X is promising” — a rival promising method doesn’t refute it. - ”X is comparatively best” — comparable alternatives are directly relevant. - A claim need not preempt exceptions its own wording already allows.

7. \*\*Absence of support ̸= contradiction\*\*, but an unsupported specific assertion is still unwarranted — weight this by whether it’s peripheral or central to the takeaway.

8. \*\*For multi-part claims, weight problems by their effect on the overall takeaway.\*\* Don’t average proposition-level defects or default to the score triggered by the single worst one.

## # Procedure

Work through these in order. Your response’s structure follows this order (see Output).

1. \*\*Claim’s contribution.\*\* What is the claim doing relative to the question — explaining a possibility, asserting a general pattern, reporting a result, making a recommendation? This determines what kind of evidence can defeat it.

2. \*\*Substantive takeaway.\*\* State the scope, certainty, comparison, and causal content the claim actually asserts, independent of phrasing. Don’t pre-weaken an overstated claim before judging it.

3. \*\*Positive evidential case.\*\* What do the relevant findings, taken together, justify? Include reasonable inferences across findings. Flag anything the claim asserts that the evidence doesn’t actually reach (a specific mechanism, magnitude, or comparison it never establishes), even if the surrounding evidence is generally favorable.

4. \*\*Objections.\*\* For each apparent counterpoint: does it address the same assertion, in the applicable setting? Distinguish a consequential restriction from a compatible failure, a different mechanism, or an unrelated limitation. Treat a finding as an outlier only when the evidence itself gives a basis for that, not because it’s outnumbered or inconvenient.

5. \*\*Net resolution.\*\* Combine 3 and 4. Does the claim survive as stated, survive with a limitation attached, or require a substantially different conclusion?

6. Score via the minimum-repair test below.

## # Minimum-repair test

Determine the score by identifying the minimum semantic revision needed to make the claim faithful to the supplied evidence. Work through these levels in order and stop at the first one that fits:

\- None: the claim can remain as written. Omitted material would only elaborate or add background. A further caveat could be written, but none is needed for fidelity.

\- Append: a caveat or exception should be added for the claim to represent the evidence faithfully, but the original claim otherwise remains — its central scope, certainty, comparison, and conclusion stay substantially unchanged.

\- Revise: the claim’s central direction survives, but a material predicate, quantifier, scope, comparison, causal attribution, or confidence level must be changed. A reader who accepted the original would draw a materially different conclusion from the repaired version.

\- Recast: the overall conclusion must be substantially narrowed, rebalanced, or made explicitly mixed. A recognizable part of the original takeaway survives, but not the conclusion as presented.

\- Replace: the central takeaway cannot be preserved. A faithful account requires a materially different conclusion; adding qualifications or rebalancing is insufficient.

Do not select Append merely because an additional caveat could be written — select it only when a caveat is genuinely needed for faithful representation. Do not select Revise for a hedge the claim’s own wording already implies (e.g. adding ”generally” to a claim that elsewhere says ”typically”) — that is an Append.

## # Scoring rubric

Each repair level corresponds to exactly one score:

None → 1 (Reasonable compression). The claim preserves important conditions and fairly represents relevant counterevidence. Omitted details do not meaningfully change the takeaway.

Append → 2 (Minor nuance lost). Some conditions, limitations, or exceptions are omitted, but including them would leave the takeaway largely unchanged.

Revise → 3 (Important qualification missing). An important condition, limitation, or exception is omitted,   
making the claim broader or more certain than warranted. The central takeaway remains defensible with   
that qualification.   
Recast → 4 (Major distortion). Substantial counterevidence or a decisive limitation is omitted or   
understated. The claim presents a clear conclusion where the evidence requires substantial narrowing,   
rebalancing, or a mixed conclusion.   
Replace → 5 (Fundamentally misleading). The claim fundamentally misrepresents what the evidence   
warrants — for example, by omitting both crucial conditions and strong counterevidence. The central   
takeaway needs replacing; adding qualifications alone cannot make it faithful. Direct contradiction is not   
required.   
# Deciding the score   
Decide from the state of the evidence around the central takeaway, then confirm against the rubric above.   
Only material bearing on the central takeaway counts here. Material that qualifies a subsidiary assertion —   
an example, a mechanism, a side-condition, an intensifier — does not move the score unless correcting it   
changes the takeaway itself.   
The claim’s hedges are part of its wording. “Likely”, “often”, “when X”, “reducing” set what has to be   
shown. Evidence that the claim is not uniformly or always true does not bear against a claim that never   
asserted universality.   
- Nothing in the evidence bears against the takeaway, or what does is already allowed by the claim’s own   
wording → 1   
- What bears against it is narrower in scope, less direct, or less consistent than what supports it — the   
takeaway holds with surety → 2   
- What bears against it is as direct and as broad as what supports it — the takeaway is arguably correct but   
contested → 3   
- What bears against it outweighs the support, or the two together leave the central question genuinely   
two-sided — a faithful account is mixed → 4   
- The evidence never establishes the takeaway, or establishes something materially different → 5   
The repair labels — none, append, revise, recast, replace — describe the consequence of the score and   
are reported for auditing. They are not the test. Before assigning a score, identify a material qualification   
required by the evidence that is not already conveyed by the claim’s wording or context.   
# Output   
Return a single JSON object with exactly these fields, in this order. No text before or after the JSON.   
{   
"claim\_role": "<one sentence: what kind of claim this is   
(possibility / general pattern / result / recommendation),   
and what that implies for what can defeat it>",   
"takeaway": "<the claim’s actual scope, certainty, comparison,   
and causal content, stated plainly>",   
"positive\_case": "<what the relevant evidence, taken together,   
establishes, and, if applicable, what the claim asserts beyond   
what the evidence actually reaches>",   
"objections": "<what the countervailing or qualifying evidence   
actually changes, if anything; use ’none material’ if nothing   
meaningfully bears on the claim as scoped>",   
"net\_resolution": "<does the claim survive as-is, survive with   
a limitation, or need substantial revision, and why, in 1-2   
sentences>",   
"repair": "none | append | revise | recast | replace",   
"score": 1

}   
Constraints: - ”repair” and ”score” must correspond exactly: none=1, append=2, revise=3, recast=4, re  
place=5. - ”score” must be an integer, not a string. - Every field is required. Do not add extra fields. -   
Output must be valid JSON — no markdown code fences, no trailing commas, no comments.

## C JUDGE VALIDATION

Two authors independently scored 73 claims on the same 1–5 scale (Table 13). GPT-5, our judge of record, agrees with Annotator 1 as closely as Annotator 2 does (ordinal α 0.62 vs. 0.64). Gemini 2.5 Pro and Claude Sonnet 4.5 agree with the annotators less well than GPT-5 does (α 0.36 and 0.04), although they still rank claims similarly (ρ 0.60 and 0.51).

Table 13: Agreement with Annotator 1 on 73 claims (Krippendorff’s α is ordinal). Exact: identical score. Within 1: scores differ by at most one point. GPT-5 agrees with Annotator 1 as closely as the second annotator does; other judges rank claims similarly but differ in calibration.
<table><tr><td>Judge</td><td>Exact</td><td>Within 1</td><td>Spearman ρ</td><td>Krippendorff α</td></tr><tr><td>Annotator 2</td><td>58%</td><td>96%</td><td>0.72</td><td>0.71</td></tr><tr><td>GPT-5 (judge of record)</td><td>56%</td><td>93%</td><td>0.69</td><td>0.68</td></tr><tr><td>Gemini 2.5 Pro</td><td>27%</td><td>74%</td><td>0.63</td><td>0.41</td></tr><tr><td>Claude Sonnet 4.5</td><td>19%</td><td>56%</td><td>0.51</td><td>0.11</td></tr></table>

All judges preserve the same system ranking as the annotator (Luna < DR Tulu < Gemini; Table 14), with Gemini 2.5 Pro and Sonnet 4.5 scoring systematically higher.

Table 14: Mean score per system on the 73 annotated claims, by judge. All judges rank the systems in the same order as the annotator.
<table><tr><td>System</td><td>Annotator 1</td><td>GPT-5</td><td>Gemini 2.5 Pro</td><td>Sonnet 4.5</td></tr><tr><td>Luna</td><td>1.96</td><td>1.71</td><td>2.33</td><td>2.88</td></tr><tr><td>DR Tulu</td><td>2.16</td><td>2.60</td><td>2.96</td><td>3.32</td></tr><tr><td>Gemini Deep Research</td><td>2.71</td><td>3.04</td><td>3.83</td><td>4.42</td></tr></table>

## D STEP BY STEP OUTPUTS OF THE SMOOTHSCORE PIPELINE

Table 15 shows a running example of our framework processing a claim. Each step and the corresponding output for the example is shown.

Table 15: Step-by-step application of the evidence-smoothing pipeline to a claim about reinforcement learning with verifiable rewards.
<table><tr><td>Pipeline step</td><td>Output for the example</td></tr><tr><td>Input question</td><td>Will RL-based methods become the mainstream approach for improving reasoning models after DeepSeek-R1, and can reward models and tree search for chain-of-thought post-training be abandoned?</td></tr><tr><td>1. Claim extrac- tion</td><td>"RL—especially RL with verifiable rewards—will become a mainstream component of reasoning-model training.</td></tr><tr><td>Document recon- struction</td><td>The system reconstructs the full text of the documents cited by the report, including work on DeepSeek-R1, verifier gaming, test-time scaling, Al- phaProof, AlphaGeometry2, and surveys of RL for reasoning.</td></tr><tr><td colspan="2">Table 15 continued</td></tr><tr><td>Pipeline step</td><td>Output for the example</td></tr><tr><td rowspan="3">2a. Supporting- evidence extrac- tion</td><td>The supporting pass extracts findings that substantiate the claim as stated. For example: "Figure 1(a) depicts the performance trajectory of DeepSeek-R1-Zero on</td></tr><tr><td>the AIME 2024 benchmark throughout the RL training process, where the average pass@ 1 score on AIME 2024 shows a significant increase, jumping from an initial 15.6% to 77.9%."</td></tr><tr><td>"As reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm for scaling reasoning capabilities in LLMs." Across all claim-document pairs, the pass extracts 61 supporting items.</td></tr><tr><td rowspan="3">2b. Countervailing- evidence extrac- tion</td><td>rially restrict the claim. For example: "We show that training on only 1,000 samples with next-token prediction and controlling thinking duration via a simple test-time technique we re-</td></tr><tr><td>fer to as budget forcing leads to a strong reasoning model that scales in performance with more test-time compute." "The rule-based RL training stage of DeepSeek-R1-Zero is narrowly fo-</td></tr><tr><td>cused on reasoning tasks, resulting in limited performance in broader areas such as writing and open-domain question answering." Across all claim-document pairs, the pass extracts 77 countervailing items.</td></tr><tr><td>3a. Local aspect induction</td><td>For each claim-document pair, the system proposes dimensions dis- cussed by that document that bear on evaluating the claim. For example, DeepSeek-R1 proposes an aspect concerning performance gains on objec- tively verifiable reasoning tasks, while the test-time-scaling paper proposes an aspect concerning non-RL alternatives for improving reasoning.</td></tr><tr><td rowspan="5">3b. Aspect con- solidation</td><td>Overlapping document-level proposals are merged into the following claim-level hierarchy: • A1: Empirical effectiveness and domain scope of RL with ver- ifiable rewards</td></tr><tr><td>– A1.1: Large gains on tasks with objective checks – A1.2: Boundaries and alternatives where RL is unnecessary or underperforms</td></tr><tr><td>• A2: Dependence on verifier reliability and design for safe, scal- able use</td></tr><tr><td>• A3: Practicality and adoption readiness of RLVR in training pipelines</td></tr><tr><td>– A3.1: Operationalization and community uptake – A3.2: Resource costs and sample-efficiency trade-offs</td></tr><tr><td rowspan="6">4. Evidence as- signment</td><td>Supporting and countervailing items are assigned to the most specific ap- plicable aspect. For example:</td></tr><tr><td>A1.1: Large gains on tasks with objective checks Supporting evidence:</td></tr><tr><td>• DeepSeek-R1-Zero's AIME 2024 pass @1 score rises from 15.6% to 77.9% during RL training. • AlphaProof learns to produce formal proofs through RL over mil-</td></tr><tr><td>lions of auto-formalized problems.</td></tr><tr><td>Countervailing evidence: • AlphaGeometry2 achieves an 84% solve rate using a pipeline cen-</td></tr><tr><td>tered on language modeling, symbolic reasoning, and synthetic data rather than RL. • AlphaProof's demonstrated successes remain concentrated in</td></tr><tr><td>5. Smoothing judgment</td><td>This node contains 31 supporting and 9 countervailing items. Across the hierarchy, this is the only node in which supporting items predominate; the remaining dimensions contain substantial countervailing evidence. Claim role: General forecast about a field-wide training trend; it can be de-</td></tr><tr><td rowspan="6"></td><td>feated by evidence that RL with verifiable rewards is unlikely to be widely adopted or that strong, persistent limitations preclude mainstream use.</td></tr><tr><td>Takeaway: Reinforcement learning—particularly when rewards are verifiable—will be widely adopted as a standard part of training pipelines for reasoning models, though not necessarily the only or exclusive ap-</td></tr><tr><td>proach. Objections: Important constraints temper the breadth of the forecast. RL scales poorly where no reliable verifier exists; it can underperform or fail to improve smaller models; compute and evaluation costs limit practicality; and verifier imperfections introduce shortcutting and misalignment risks.</td></tr><tr><td>Strong non-RL alternatives can also match or surpass leading RL-trained systems. These findings show that RLVR will not be universally applicable or singularly dominant across reasoning tasks. Net resolution: The claim largely holds but needs an explicit limitation: RL with verifiable rewards is poised to be mainstream for verifier-evaluable</td></tr><tr><td>settings, yet adoption will be constrained by verifier availability, safety and shortcut risks, model and compute regimes, and viable non-RL alternatives</td></tr><tr><td>that will continue to coexist. Repair: APPEND Smoothing score: 2</td></tr></table>

## E REPORT REGENERATION

## E.1 SETUP FOR WEBSEARCH

<table><tr><td>Prompt E.1: Report Generation with Web Search</td></tr><tr><td>System Prompt:</td></tr><tr><td>You are a professional researcher producing a structured, citation-backed report in response to the user&#x27;s question. Ground every factual claim in the web sources you retrieve and cite them inline. Be precise and</td></tr><tr><td>analytical. Prefer recent, authoritative sources.</td></tr></table>

```handlebars
User Prompt:
{{Question }}
```

## E.2 SETUP FOR WEBSEARCH, WITH INSTRUCTIONS

![](images/d622fc7cca3fb240ad069ab5988dabf131d7604febf328fa07ba73b1d60dfb08.jpg)

## E.3 SETUP FOR DOCS IN CONTEXT, WITH INSTRUCTIONS

Prompt E.3: Report Generation from Provided Documents (with   
Calibration Instructions)   
System Prompt:   
You are a professional researcher producing a structured, citation-backed report. The complete source   
documents are provided below; they are the ONLY evidence you may use. Ground every factual claim in   
them and cite by document number, e.g. [doc 3].   
Readers will rely on your report to understand what the evidence does and does not establish, so keep each   
conclusion calibrated to the documents:   
State the setting or conditions in which a result was shown, and do not generalize beyond them.   
Where the documents disagree with each other, say so explicitly rather than choosing a side silently.   
Do not combine separate findings into a stronger conclusion than the documents establish.   
Avoid conclusive language unless the documents clearly warrant it.   
If the documents do not support a conclusion, say so.   
User Prompt:   
SOURCE DOCUMENTS   
=== DOCUMENT 1 ===   
URL: {{Document URL }}   
TITLE: {{Document Title }}   
{{Document Text }}   
=== DOCUMENT 2 ===   
URL: {{Document URL }}   
TITLE: {{Document Title }}   
{{Document Text }}   
. . .   
=== QUESTION ===   
{{Question }}   
Write the report now, citing documents by

## E.4 SETUP FOR DOCS IN CONTEXT, NO INSTRUCTIONS

Prompt E.4: Report Generation from Provided Documents (No   
Instructions)   
System Prompt:   
You are a professional researcher producing a structured, citation-backed report. The complete source   
documents are provided below; they are the ONLY evidence you may use. Ground every factual claim in   
them and cite by document number, e.g. [doc 3].   
User Prompt:   
SOURCE DOCUMENTS   
=== DOCUMENT 1 ===   
URL: {{Document URL }}   
TITLE: {{Document Title }}   
{{Document Text }}   
=== DOCUMENT 2 ===   
URL: {{Document URL }}   
TITLE: {{Document Title }}   
{{Document Text }}   
. . .   
=== QUESTION ===   
{{Question }}   
Write the report now, citing documents by number.

## E.5 IMPROVED CLAIMS UNDER DOCUMENT ENGAGEMENT

Table 16 shows how engaging with documents can repair claims. We manually select cases where the same claims persist across setups but change in terms of scope and strength representing the cited documents accurately.

Table 16: How smoothing changes when the same model is given the cited documents (Luna, ResearcherBench). Claims are extracted independently per report; each row shows the claim in each arm that reaches the same conclusion, with σ in parentheses. “–” means no claim in that arm reaches the same conclusion.
<table><tr><td>Pattern</td><td>Question</td><td>WebSearch (σ)</td><td>Docs in context (σ)</td><td>Docs + instructions (σ)</td></tr><tr><td>A result from one setup stated without its conditions</td><td>Analyze the dif- ferences between the LIMO and S1 papers: objectives, methodology, key findings, contribu- tions.</td><td>LIMO appears stronger in broad benchmark performance, particularly on GPQA Diamond and MATH500. (4)</td><td>LIMO reports higher scores than original s1 on AIME24. MATH500 and GPQA, but the studies differ in evaluation details (greedy decoding vs. temperature-0) ... the compari- son does not by itself establish a</td><td>LIMO is the more explanatory and curation-focused paper . . [doc 1]. (3)</td></tr><tr><td>A result from one setup stated without its conditions</td><td>How can we detect hallucinations in LLMs using their internal states, and what advantages does this offer over</td><td>Lower latency and computa- tional cost: internal detectors can operate in the same for- ward pass and use a lightweight probe. MIND targets real-time detection, and HIDE reports substantially lower computation</td><td></td><td>Its principal advantages over external or output-only detec- tion are earlier intervention, richer process-level information, reduced reliance on external knowledge bases, lower latency</td></tr><tr><td>Separately supported frag- ments com- bined into one demonstrated conclusion</td><td>Could transformer architectures be reimagined to pro- cess multimodal inputs (video/audio/- text) as efficiently as</td><td>The more promising princi- ple is: use modality-specific perception, adaptive semantic tokenization, sparse cross- modal attention, persistent latent memory, and on-demand access</td><td>The evidence suggests that multimodal Transformers may need to be redesigned around asymmetric processing, learned compression, latent bottlenecks, adaptive tokenization, and selec- tive cross-modal interaction. (3)</td><td>The likely path ... is to re- design the architecture so that all modalities are converted into compact, queryable rep- resentations ... Whether that bottleneck can remain small enough ... remains unresolved</td></tr><tr><td>Separately supported frag- ments com- bined into one demonstrated conclusion</td><td>How might federated learning combined with model distil- lation overcome technical and legal barriers in domains</td><td>The most credible approach is therefore a layered architecture: local training + distillation + secure aggregation + differential privacy + strong governance and</td><td>The strongest approach is therefore layered: local data retention, carefully selected distillation, secure aggregation, differential privacy where appropriate ... and clinical validation. (3)</td><td>The strongest approach is there- fore layered: [same list] . .. Such an architecture can sub- stantially lower both technical and legal risk, but the sup- plied documents do not support claiming that it eliminates either</td></tr><tr><td>Conclusive or superlative wording be- yond what the sources state</td><td>What is the evolu- tion path of multi- modal models from early visual repre- sentations to current multimodal LLMs, and what are the key</td><td>Language as a universal inter- face for vision: natural language became both a source of super- vision and a way to query visual representations. (3)</td><td>CLIP&#x27;s central breakthrough was language-grounded open- vocabulary recognition ... The approach also exposed impor- tant limitations: performance depended on prompt wording, and CLIP struggled with count- ing, fine-grained recognition, and out-of-distribution images</td><td>CLIP and ALIGN demonstrated that web-scale paired text can train general visual represen- tations, support retrieval, and produce zero-shot classifiers through natural language [docs 11, 13]. (1)</td></tr><tr><td>Conclusive or superlative wording be- yond what the sources state</td><td>(same question)</td><td>BLIP-2 made the connection more parameter-efficient. This was an important engineering breakthrough because it showed that multimodal capability could be obtained without end-to-end retraining of every parameter.</td><td></td><td>Flamingo and BLIP-2 showed that strong pretrained vision and language models can be connected through relatively small trainable modules rather than fully retrained end-to-end [docs 15–16]. (1)</td></tr></table>

## F DOCUMENT ENGAGEMENT DETAILS

We measure how often each system opens the documents it cites. A cited link counts as opened if the system retrieved that page, in any form, during generation. URLs are canonicalized before matching (arXiv /abs/ and /pdf/ variants merged; tracking parameters and trailing slashes removed), and each cited link is counted once per dataset. Table 17 reports all systems and datasets.

Table 17: Share of cited links opened, by system and dataset (opened / cited links). A link counts as opened if the system issued an open, fetch or visit for it. Gemini Deep Research exposes no browsing steps.
<table><tr><td>System</td><td>ResearcherBench</td><td>CS2-Expert</td><td>DRB-II</td><td>ResearchRubrics</td></tr><tr><td>Luna 5.6</td><td>19.2% (95/496)</td><td>13.3% (34/256)</td><td>9.3% (27/291)</td><td>8.9% (17/190)</td></tr><tr><td>Sol 5.6</td><td>27.7% (169/611)</td><td>30.0% (88/293)</td><td>16.1% (68/422)</td><td>28.2% (70/248)</td></tr><tr><td>QUEST</td><td>73.8% (217/294)</td><td>64.9% (63/97)</td><td>30.5% (36/118)</td><td>29.4% (25/85)</td></tr><tr><td>Sonnet 5</td><td>2.3% (34/1,458)</td><td>3.5% (19/536)</td><td>1.1% (8/750)</td><td>1.1% (6/559)</td></tr><tr><td>Opus 5.5</td><td>0.9% (12/1,277)</td><td>2.3% (13/561)</td><td>1.3% (10/751)</td><td>0.2% (1/471)</td></tr><tr><td>DR Tulu</td><td>0.5% (5/1,082)</td><td>0% (0/445)</td><td>0% (0/303)</td><td>0% (0/290)</td></tr><tr><td>Gemini Deep Research</td><td></td><td></td><td></td><td></td></tr></table>

Luna, Sol, Sonnet 5, Opus 5.5, and DR Tulu. Luna and Sol use the OpenAI Responses API web search tool, whose calls are recorded as search, page-open, or find-in-page actions (OpenAI, 2026). We count a cited link as opened if the returned search results contain that page’s opened content, marked by a Source: open(...) header. These markers capture every page-open and findin-page action, as well as opens returned inside other calls’ results, so counting the actions alone would miss about three quarters of them. Sonnet 5 and Opus 5.5 run with Anthropic’s web search and web fetch tools. Search results return page content in encrypted form, so we cannot observe what the model sees from search alone, while web fetch retrieves a page’s full text (Anthropic, 2026b;a). We therefore count a cited link as opened if the system successfully fetched it. As search results are encrypted, this measures full-page retrieval only; we cannot observe how much page content the model receives from search. For DR Tulu, we count calls to its browsing tool (Shao et al., 2026). All five systems cite mostly from search results rather than pages they open. Luna and Sol open 9–19% and 16–30% of the links they cite (45 of Luna’s 65 ResearcherBench reports open no cited page), Sonnet 5 and Opus 5.5 fetch 1.1–3.5% and 0.2–2.3%, and DR Tulu calls its browsing tool only 5 times across all datasets.

QUEST. QUEST’s visit tool does not return the raw page content to the agent: a separate model (GPT-5-mini) reads the page and returns an extract conditioned on a goal QUEST states for the visit, consisting of selected evidence and a short summary (Xie et al., 2026). We count a cited link as opened if QUEST issued a visit to it. Its high open rate therefore does not mean QUEST reads what it cites: the agent sees only extracts filtered toward what it was looking for, which is consistent with its high smoothing despite frequent visits.

Gemini Deep Research. The Deep Research API returns only the final report and its citation links, with no browsing or tool steps, so engagement cannot be measured.

## G EXAMPLES OF SMOOTHING IN MODEL GENERATED CLAIMS

## G.1 OPEN-SOURCE SYSTEMS

Examples of generations by open sources systems are provided DR Tulu and Quest are in Table 18.

## G.2 CLOSED-SOURCE SYSTEMS

Table 19 shows examples of claims generated by closed source systems.

Table 18: High-severity claims from the open-source systems, with the SMOOTHSCORE judge’s verdict.
<table><tr><td>Pattern</td><td>System</td><td>Question DRB-II ex16: I am</td><td>Claim (score) Alternate Cells: 7T adds ex-</td><td>Judge verdict The claim over-compresses</td></tr><tr><td>Unsupported DR Tulu composition</td><td>researching the chal- lenges faced by SRAM operating at low voltages, and I hope you can help me survey and compile a technical report on mainstream solutions. The report needs to</td><td>read. (4) include the following</td><td>plicit precharge-equalize sequence, 8T virtual- ground/zero-precharge, 9T conflict-free write with equalized leakage, 10T quasi-differential read, 11T DDPS for ultra-low leakage, 12T gating and two-stage</td><td>heterogeneous literature into a fixed one-to-one mapping and misattributes key mecha- nisms (notably 7T and 11T), while several other pairings are only variant-specific. It requires substantial revision to a qualified, example-based mapping rather than categori- cal assignments. The claim over-compresses</td></tr><tr><td>composition</td><td>Unsupported DR Tulu CS2-Expert ex6: De-</td><td>scribe the Sunflower conjecture, progress towards a solution, and the sunflower conjec- related problems.</td><td>Decision trees, DNF/branch- ing programs, and com- munication: Variants of ture directly constrain the size/width/depth of Boolean formulas and decision trees, with tight results (e.g., (k2)) for unambiguous DÑF; these bounds, proved via explicit constructions and sunflower arguments, carry over to communication protocols and algebraic com- plexity analogs (e.g., matrix multiplication exponent con-</td><td>heterogeneous results: it cor- rectly cites tight unambigu- ous DNF bounds and con- nections to communication, and separately, sunflower- variant tools and matrix- multiplication connections, but it incorrectly attributes the tight DNF bound to sun- flower arguments, overstates direct constraints on decision trees/branching programs, and overstates transfer to al- gebraic complexity. It needs substantial narrowing and</td></tr><tr><td>Premature closure</td><td></td><td>QUEST CS2-Expert ex11: Sur- Validates that LLM-powered vey recent papers on the use of LLMs for 5G network orchestra- tion</td><td>nections surveyed by Alon). (4) prompt engineering outper- forms static ML methods in highly variable O-RAN envi- ronments. (5)</td><td>reattribution. The claim overstates what the evidence shows. The record supports that LLM prompting can beat simple heuristics and be useful in hybrid schemes, but it does not validate outperformance over static ML in highly vari-</td></tr><tr><td>Premature closure</td><td>QUEST</td><td>ResearcherBench ex3: In multimodal pre- training, the current mainstream paradigms viously reported models in are based on image tokens and stable dif- fusion. Analyzing the latest advancements in these two technical approaches, with refer- (GenEval 0.66, NOCAPs ence ...</td><td>SOTA Across Multiple Benchmarks: DDT-based MLLMs outperform all pre- zero-shot/non-zero-shot eval- uation for image-text match- ing, captioning, question an- swering (VQA, GQA), object detection, and metric scores 124.2, EVR L1 14.5, CVS 85.5%). Even compared to diffusion specialists trained on similar data, DDT shows clear superiority in composi- tional edits and counterfac- tual interpolation, owing to its recursive, attribute-rich tokens that mimic natural</td><td>able O-RAN environments. The claim needs substantial revision: while the quoted metric values are accurate and DDT-based MLLMs are competitive, the evidence shows they do not outper- form all previously reported models on several listed benchmarks and provides no support for object detec- tion.</td></tr></table>

Table 19: High-severity claims from the closed-source systems, with the SMOOTHSCORE judge’s verdict.
<table><tr><td>Pattern Scope ex-</td><td>System</td><td>Question</td><td>Claim (score)</td><td>Judge verdict</td></tr><tr><td>pansion</td><td>Sonnet 5</td><td>ResearcherBench ex49: From a tech- nical perspective, how to understand the similarities and differences between Reinforcement Learn- ing (RL) algorithms and Supervised Fine- Tuning (SFT) in Large Language Models (LLMs), as well as .. .</td><td>RL, especially when trained with an outcome-based re- ward, generalizes across both rule-based textual and vi- sual variants, while SFT, in contrast, tends to memorize training data and struggles to generalize out-of-distribution scenarios. (4)</td><td>The claim overstates a mixed and conditional landscape. While there is solid evi- dence that outcome-based RL can improve OOD gen- eralization in both textual and visual tasks and that SFT often overfits, there is also substantial, direct evidence that SFT can generalize and sometimes surpass RL, and that RL&#x27;s gains depend on reward design, initialization, and compute. The conclu- sion needs rebalancing and</td></tr><tr><td>Premature closure</td><td>Luna</td><td>ResearcherBench ex3: For multimodal pretraining In multimodal pre- training, the current mainstream paradigms are based on image tokens and stable dif- fusion. Analyzing the latest advancements in these two technical approaches, with refer- ages, text, and video can ence ...</td><td>as a route toward a single general-purpose model, dis- crete image-token modeling appears more promising than a Stable-Diffusion-centered architecture. It fits naturally with the scaling machinery of language models: im- be serialized into one se- quence and trained with a common next-token objec- tive. The strongest evidence comes from Emu3, Janus- Pro, Chameleon, and the</td><td>explicit conditions. The claim needs substan- tial revision. The evidence supports discrete-token, next- token training as a viable and rapidly improving path, but not that it is more promis- ing overall than diffusion- centered approaches; mul- tiple studies show diffu- sion/flow or hybrid models scaling better and being more compute-efficient, and the ‘natural fit&#x27; is tempered by instability and modality- specific shortcomings.</td></tr><tr><td>Unsupported Gemini composition</td><td></td><td>CS2-Expert ex2: What At the mechanical layer, the are some of the meth- ods of detecting GPS location spoofing at- tack in Connected and Autonomous Vehicles? if a spoofed signal perfectly</td><td>rapidly improving tokenizers TiTok/FlexTok. (4) tight fusion of GNSS data with highly resilient IMU kinematics and LiDAR map- matching ensures that even replicates the cryptographic and spatial signatures of a satellite, it fundamentally cannot alter the physical reality of the road ahead. (4)</td><td>The claim needs substan- tial revision. The evidence supports that GNSS-IMU— LiDAR/map fusion can greatly harden localization and often detect/mitigate spoofing, but it does not en- sure immunity—even against sophisticated, spatially and cryptographically convinc- ing attacks—and has doc- umented failure modes and delays.</td></tr></table>