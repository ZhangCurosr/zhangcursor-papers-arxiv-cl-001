# AskGLP1: A Knowledge Graph LLM Agent for Exploring GLP-1 Experience from Online Patient Narratives

Zhaoqian Xue<sup>1,†</sup>, Anay Somani<sup>1,\*</sup>, Siddhartha Singh<sup>1,\*</sup>, Junting Mu<sup>2</sup>, Zhongyin Wang<sup>3</sup>,

Haozheng Liu<sup>4</sup>, Ayesha Khatun<sup>1</sup>, Yeaeun Kwon<sup>1</sup>, Renkai Ma<sup>5</sup>,

Siyuan Ma<sup>6</sup>, Zhe He<sup>7</sup>, Danxia Yu<sup>6</sup>, Lingyao Li<sup>1,†</sup>

<sup>1</sup>University of Arizona <sup>2</sup>University of California, Irvine

<sup>3</sup>University of Illinois Urbana-Champaign <sup>4</sup>New York University <sup>5</sup>University of Cincinnati <sup>6</sup>Vanderbilt University Medical Center <sup>7</sup>Florida State University

## Abstract

GLP-1 medications are increasingly used for weight management, and many people share their experiences online. These narratives show why people take the drugs and what they experience, but they remain unstructured and disconnected from formal pharmacovig ilance. We therefore present AskGLP1, a system that transforms online patient narratives into a knowledge graph (KG) and answers questions about medication experiences. Through an LLM-assisted pipeline grounded in the MedDRA terminology, AskGLP1 screens nearly 8.5 million Reddit posts for medication experiences and distills 163,135 of them, with 2,506 WebMD reviews, into a KG of 385,790 drug–side-efect and drug–purpose relations. We find that GLP-1 experiences are source- and drug-specific. Online narratives share the most common gastrointestinal symptoms with FAERS, but they foreground everyday burdens such as fatigue and food aversion, whereas FAERS reports more often include injection-site reactions and selected serious diagnoses such as pancreatitis. We further find that posts report less nausea and vomiting for tirzepatide than for semaglutide, more fatigue for Wegovy than Ozempic and for Zepbound than Mounjaro. Evaluations show that the agent answers with verifiable evidence and declines requests for medical advice. Our findings reveal gaps between what FAERS captures and what people report online, and they call for patient-centered approaches to pharmacovigilance. Code, prompts, and evaluation questions are available at https://github.com/ZhaoqianXue/AskGLP1.

## CCS Concepts

• Information systems → Web mining; • Applied computing → Health informatics.

## Keywords

GLP-1, digital health, adverse events, knowledge graph, LLM agents

## 1 Introduction

GLP-1 medications, a term we use for glucagon-like peptide-1 (GLP-1) receptor agonists such as semaglutide and for the dual GIP/GLP-1 receptor agonist tirzepatide, were originally developed for type 2 diabetes and are now widely prescribed for weight management [17, 46]. By late 2025, 12% of U.S. adults reported taking one [25], but side efects are common. Nausea and diarrhea are the most frequent in trials [16, 46], and observational studies link weight-loss use to rarer serious outcomes such as pancreatitis, bowel obstruction, and gastroparesis [38]. Understanding these patterns requires knowing why people take these drugs and what they experience.

Conventional data sources capture these experiences only partially [12]. The FDA Adverse Event Reporting System (FAERS) collects coded adverse event and medication error reports, but most of its reports are submitted by manufacturers and tend to emphasize serious outcomes, while spontaneous reporting misses many adverse events [26]. By contrast, many people describe them in detail on Reddit and WebMD. Such posts can extend FAERS signals [30], and recent work examines GLP-1 discussions on Facebook and Reddit [1, 5, 36]. These communities thus ofer a large, informal evidence source that complements formal pharmacovigilance.

Turning these posts into reliable evidence is challenging. First, many symptom mentions do not report a medication experience, because people also ask questions, give advice, or share news [20, 21]. Second, online language is informal and unstructured, which needs to be mapped to standard terms such as MedDRA for cross-source comparison [6, 30]. Third, the evidence should be accessible and verifiable. General-purpose large language model (LLM) chatbots can give unsupported or hallucinated health answers [29, 49]. Answers should therefore cite their sources and flag missing evidence.

Recent studies show promise. One study uses LLMs to build a knowledge graph (KG) of semaglutide side efects from Reddit [5], though it does not support natural language queries. LLM agents for medical question answering integrate KGs or evidence retrieval [18, 33], drawing on curated clinical knowledge rather than patient narratives. Other agents improve traceability by grounding answers in source-linked evidence [43, 48]. Building on these approaches, we present AskGLP1, a system that organizes evidence from Reddit, WebMD, and FAERS into a KG and answers questions about the reasons for taking GLP-1 medications and reported side efects through a KG-grounded agent. We study three research questions:

• RQ1: What side efects do people report online, and how do reporting patterns vary across drugs and change over time?

• RQ2: How do online patient narratives complement FAERS?

• RQ3: How does the KG-grounded agent provide verifiable answers to AskGLP1 users’ questions?

To answer these questions, AskGLP1 combines an LLM pipeline that converts posts into a quote-grounded KG with a KG-grounded agent that answers questions over it. The pipeline detects medication experiences, extracts drug–purpose and drug–side-efect relations, links each to its sources, and normalizes side efects to MedDRA while keeping users’ wording. The agent queries the KG, returns supporting quotes, and abstains when evidence is missing. We make four contributions. First, we build a quote-grounded KG covering multiple GLP-1 medications and purposes. Second, we analyze side efects reported online to show how reporting patterns vary across drugs and change over time. Third, we compare Reddit, WebMD, and FAERS to show where they agree and where patient narratives add to FAERS. Fourth, we evaluate AskGLP1 against baselines to show that its verifiable answers reach the correct conclusion far more often. Overall, we turn patient narratives on social media and the Web into verifiable evidence on real-world medication experiences for researchers, clinicians, and patients.

## 2 Related Work

Patient-Reported Medication Experiences and FAERS. People often share their experiences with drugs online. These posts help researchers study why people take drugs, whether they find them helpful, and what side efects they report [10, 22]. These online experiences ofer a patient perspective alongside adverse event reports in FAERS. Comparisons with FAERS reports find similar broad reporting patterns on Twitter and show that WebMD reviews can add personal details on side efects [8, 30]. How GLP-1 experiences on Reddit and WebMD difer from FAERS reports remains an open research question. Informal language and diferences across websites and evaluation methods also make online posts hard to analyze [11, 34]. More recently, a Facebook study examines GLP-1 symptom mentions, including personal experiences and general concerns [1]. Sehgal et al. screen for personal semaglutide and tirzepatide use on Reddit and analyze reported side efects with manually evaluated classifiers [36]. They also report descriptive drug comparisons and monthly symptom counts.

LLM Extraction and Knowledge Graphs from User-Generated Content. Building a KG from patient narratives requires finding health information, understanding its meaning, and matching dif ferent entities for the same concept. Early studies extract health concepts and their relationships from Reddit. HEAL also organizes personal experiences into a KG [7, 44]. However, a symptom mention may describe something a person denies or worries about, rather than something they experienced [35]. Later studies use LLMs to extract side efects and match everyday descriptions to standard medical terms [3, 6, 27]. More closely related to our work, a semaglutide study uses an LLM to build a KG of side efects from Reddit posts [5]. Yu et al. combine Reddit, WebMD, and FAERS in a traceable KG with MedDRA mappings and an agent that queries it for antidepressant information [48].

LLM Agents for Evidence-Grounded Health Question Answering. Early research shows that LLMs hold useful medical knowledge, but their answers can still be wrong [37]. Retrievalaugmented generation (RAG) helps address this problem by adding retrieved information, although finding the right evidence remains important [23, 47]. Agent systems extend this approach by manag ing searches and choosing evidence. For example, LINS provides source-linked medical answers, while an agent framework answers patient questions using selected sentences from health records [14, 43]. Other work adds memory or medical KGs to support answers across several steps [18, 33]. These findings guide AskGLP1’s

Table 1: AskGLP1 data preparation by step.
<table><tr><td></td><td>Reddit</td><td>WebMD</td><td>FAERS</td></tr><tr><td>Collected</td><td>8,496,004</td><td>2,727</td><td>397,362</td></tr><tr><td>Cleaned</td><td>1,030,309</td><td>2,681</td><td>一</td></tr><tr><td>Kept by detection</td><td>256,273</td><td>-</td><td>一</td></tr><tr><td>Eligible reports</td><td>1</td><td>-</td><td>348,227</td></tr><tr><td>Posts in the KG</td><td>163,135</td><td>2,506</td><td>-</td></tr><tr><td>Drug-Side-effect relations</td><td>334,201</td><td>5,166</td><td>-</td></tr><tr><td>Drug-Purpose relations</td><td>43,875</td><td>2,548</td><td>-</td></tr><tr><td>Age group stated</td><td>15,129</td><td>2,413</td><td>-</td></tr><tr><td>Sex/gender stated</td><td>11,993</td><td>2,321</td><td>一</td></tr></table>

KG-grounded agent, which can help AskGLP1 users explore people’s reported experiences with drugs across Reddit and WebMD and check the original text behind each answer.

Research Gaps. We extend prior GLP-1 studies by comparing ingredients, brands, and years while checking sensitivity to the basis of drug identification (RQ1). AskGLP1 connects relation-level evidence and MedDRA mappings to cross-source comparisons under explicit counting rules with a separate denominator for each source (RQ2). We evaluate the agent against two baselines through human ratings on 120 questions, assessing evidence support separately from major errors, including those involving counting scope (RQ3).

## 3 AskGLP1 System Design

Figure 1 provides an overview of AskGLP1 system design. AskGLP1 builds a KG from Reddit and WebMD narratives about GLP-1 medications, then compares the patterns with FAERS, and allows users to query it in natural language (Table 1). After data collection (Section 3.1), AskGLP1 contains the following components.

• Medication experience detection (Section 3.2) identifies posts that report personal experience with details.

• Entity and relation extraction (Section 3.3) turns each post into drug–purpose and drug–side-efect relations with details and demographics, each citing its sources.

• Entity normalization (Section 3.4) maps drug names to a shared catalog and purposes and side efects to MedDRA terms, so that variants count as one concept and match FAERS.

• KG construction (Section 3.5) links the relations through extracted entities, so that the KG can be filtered by drug, source, date, and demographics and traced to its text.

• KG-grounded agent (Section 3.6) answers questions only from graph tools and checks each number and quote, so that AskGLP1 users can verify every claim.

## 3.1 Data Collection and Preparation

We study four ingredients of GLP-1 medications and their eight brands, namely semaglutide (Ozempic, Wegovy, Rybelsus), tirzepatide (Mounjaro, Zepbound), liraglutide (Victoza, Saxenda), and dulaglutide (Trulicity). We select this set based on the availability of WebMD brand reviews and corresponding Reddit coverage to support comparisons across ingredients, brands, and sources. Together, these brands span approved indications for type 2 diabetes and weight management [41, 42].

Reddit. Reddit is a discussion platform organized into topicspecific forums (subreddits), where people publish submissions and reply in threaded comments [7]. Through the Arctic Shift archive [13], we collect submissions and comments, together called posts (as are WebMD reviews), from 16 GLP-1 and type 2 diabetes subreddits, each from the earliest FDA approval among the brands assigned to it in our collection configuration [41, 42] to September 15, 2026 (Appendix A). Because some of these subreddits center on one brand or ingredient, which ties a post to an ingredient even when the user only says “my shot,” we apply no drug-name filter. This yields 8,496,004 posts (551,694 submissions and 7,944,310 comments), counted per post because we collect no account identifiers. Next, we apply rule-based cleaning to remove short, duplicate, tem plated, and non-English posts, resulting in a dataset of 4,660,528 posts. To give extraction more context, we retain posts with more than 40 distinct content words outside sentences repeated across posts. We fix this threshold before any LLM step because it yields the corpus size closest to the budgeted target of one million posts. This leaves 1,030,309 posts (Appendix A).

![](images/03ac984c45f37757ae6bd7b5a5b2f182384097ac53823bae6cebe8c03b6374a3.jpg)  
Figure 1: Overview of AskGLP1. (a) Reddit posts and WebMD reviews yield drug–side-efect and drug–purpose relations, with purposes and side efects coded to MedDRA. (b) The KG links drugs to purposes and side efects and connects FAERS reports to shared drug and MedDRA PT nodes. (c) The KG-grounded agent answers a question about constipation shares across sources for Wegovy. Denominators require a retained side efect or event. Numbers and quotes are checked against tool outputs.

WebMD. WebMD is a consumer health website whose drug pages let people rate a drug and describe their experience with it [30]. Each review concerns a named drug and carries profile fields entered by the user, such as age group, sex/gender, and the condition treated. We collect all 2,727 reviews of the eight brands posted up to September 2026, of which 2,681 contain text.

FAERS. FAERS is the FDA’s database of spontaneous adverse event and medication error reports submitted by health care professionals, consumers, and manufacturers. From its July 30, 2026 openFDA release [19], we take the 397,362 unique reports received through June 30, 2026 that include one of the four studied ingredients, each in its latest version, with reported events coded as MedDRA Preferred Terms (PTs) [4]. Of these, 348,227 are eligible for comparison before clinical-term filtering (Section 4.2).

## 3.2 Medication Experience Detection

Many posts ask questions or repost news rather than describe drug use [20, 21], so we first identify posts that report medication experiences. We define a medication experience as an account of drug use or an individual treatment decision or plan with details. We identify these posts using Jev [39] (jev-1.13.0), a language-model classifier with natural-language instructions (Appendix B). The classifier assesses whether a post clearly meets three criteria, including an identifiable drug (even unnamed, as in “the 2.5”), a personal experience (of the user or a specific other person, not advice or a hypothetical risk), and clinical detail (such as a dose, a symptom’s onset or impact, or a treatment decision). We retain posts with a probability of at least 0.5, yielding 256,273 Reddit posts (24.9%). We do not apply this filtering step to WebMD reviews because each review is already associated with a named drug. On a held-out validation set of 200 randomly sampled posts, the classifier achieves a precision of 0.87 and a recall of 0.69 (Appendix D).

## 3.3 Entity and Relation Extraction

Based on prior work on LLM-based drug side-efect KG [5], we extract entities and relations using the following definitions.

• Entities. We extract three entities. (i) A drug is identified by in gredient and brand name, including indirect references such as “my shot” when the post is from the relevant subreddits. (ii) A purpose is a stated reason for taking the drug (“to lose weight”), rather than a reported outcome (“it helped my PCOS”). For WebMD, purposes also include the profile’s condition value. A reported side efect is a symptom that the user experienced and linked to the drug through an explicit attribution or its timing. This link does not establish causation. Feared, negated, and hy pothetical symptoms do not count, nor does expected appetite or weight loss unless the user clearly presents it as a side efect.

• Relations. A relation links a drug to a purpose (USED\_FOR) or a side efect (SIDE\_EFFECT). We focus on the user’s own use so that relations can be matched to the user’s demographics.

• Attributes. For each side efect, we include dose, duration, intensity, impact, and action taken, when stated (Appendix C).

• Demographics. For Reddit posts, the user’s age (based on WebMD’s eight bands) and sex/gender, only when stated. For WebMD reviews, we obtain these fields from the profile.

We use GPT-6 Luna [28] at medium reasoning efort to extract entities and relations from each post with a strict output schema (Appendices B and C). Posts are divided into numbered sentences, and the model returns the supporting sentence numbers for each relation. We then map these numbers to the original text and verify that the extracted purpose or side-efect wording appears in the cited sentences. Each relation is thus linked to its supporting evidence. Of the 256,273 Reddit posts retained during detection, 163,135 yield at least one relation. The remaining posts include accounts of other people’s use and reports of no side efects, which pass detection but do not meet the extraction criteria.

We further evaluate the extraction performance on a held-out dataset of 300 sampled posts. Side efects reach a precision of 0.88 and a recall of 0.78 on Reddit and 0.91 and 0.87 on WebMD, respectively, and drugs and demographics score higher. Purpose extraction is less reliable (Reddit precision 0.58), mainly because the model extracts goals that annotators do not count as reasons (Appendix D).

## 3.4 Entity Normalization

Entity normalization maps variant names and expressions (e.g., “Zep,” “sulfur burps”) to shared identifiers in the KG. We map purposes and reported side efects to MedDRA 29.1 [24].

Drugs. We record a brand only when explicitly named and identify the ingredient from either the brand or generic name. For Reddit side-efect relations with no named drug (43.1%), we infer the ingredient from the subreddit when it concerns a single ingredient, without assigning a brand. For FAERS, we use the reporter’s product-name and active-substance fields (Appendix A).

Side efects and purposes. We use a two-step method to match the wording, including WebMD profile conditions, to a MedDRA Lowest Level Term (LLT), which includes lay expressions. BioLORD-2023 [32], a biomedical embedding model, retrieves the 10 closest LLTs. GPT-6 Luna at medium reasoning efort then reads the wording, its sentences, and the candidates and selects one or none, guided only by the general principles of the MedDRA term selection guide [15]. Each selected LLT is mapped to its preferred term (PT) and the PT’s primary MedDRA hierarchy path. We also map reported FAERS terms to MedDRA 29.1 PTs for comparison with narrative side efects. We fix the method on development samples before the blind evaluation. Expressions without a suitable match, such as “food noise,” remain uncoded with their original wording. For normalization performance, annotators choosing among the same 10 candidates agree with the selected term for 96 of 100 side-efect and 89 of 100 purpose relations. Purpose accuracy is 81% when concepts missing from the candidates count as errors (Appendix D).

## 3.5 Knowledge Graph (KG) Construction

A KG represents entities as nodes and the relationships between them as typed edges. In our context, the KG connects relations from both platforms through shared drug and MedDRA concept nodes, so that every count can be filtered by drug, source, or demographics and traced back to its supporting text. Formally, the narrative component of the KG is a directed graph:

$$
G = ( V , E ) , V = V _ { D } \cup V _ { P } \cup V _ { S } ,\tag{1}
$$

where $V _ { D }$ contains the drug nodes, and $V _ { P }$ and $V _ { S }$ contain purpose and side-efect PTs, each with one additional uncoded node. Each relation becomes one edge:

$$
e = ( d , r , c , a ) \in E , d \in V _ { D } , c \in V _ { P } \cup V _ { S } ,\tag{2}
$$

where the type � is USED\_FOR if� ∈ �<sub>�</sub> and SIDE\_EFFECT if� ∈ �<sub>�</sub>. The attributes � store the platform, post identifier, date, the user’s age group and sex/gender, the side-efect details, the basis of drug identification, and the evidence spans. The resulting KG has 13 drug nodes, 954 purpose nodes, and 2,265 side-efect nodes, linked by 385,790 edges. For querying, each FAERS report links to the same drug nodes, with its drug roles, and to its reported MedDRA PTs. FAERS events are not assigned to individual drugs within a report.

## 3.6 KG-Grounded Agent Development

We design an agent based on GPT-6 Luna with medium reasoning efort and require responses to be supported by the grounded KG. Its five read-only tools link question words to MedDRA terms (link\_terms), count posts with graph queries (query\_graph), re trieve relations with their sentences (get\_evidence), compare shares with FAERS (compare\_faers), and run sandboxed analyses on tables derived from the KG (run\_python).

Answers follow fixed reporting rules. A concept counts once per post. A share is taken over posts with a relation of the same type under the same filters. Reddit and WebMD are analyzed separately. Because posts are self-selected, answers report shares of posts with a relation for the concept, rather than incidence or risk. The agent reports a diference only when 95% Wilson intervals do not overlap and draws no conclusion from fewer than 10 posts.

Every number carries a tag that points to one row of an output. Before a response is shown, code checks its tagged number, quote, and comparison, and the coverage of related terms, against the tool outputs and source sentences. An answer that fails these checks undergoes up to two revision rounds and is marked as unverified if it still fails. Responses are labeled as answered, limited (partly outside the data, such as risk or cause, or too sparse), or declined (medical advice, with a referral to a doctor, or not answerable). Figure 2 shows the KG view and an illustrative example of agent responses, and Section 4.3 evaluates these responses (RQ3).

![](images/dd4e130505062014cc6912cdd568887c1a00adb1307859c7541f1fe93653424b.jpg)

<table><tr><td colspan="3">(b) Answering with AskGLP1</td></tr><tr><td colspan="3">How does the share of Ozempic writers reporting nausea change across age groups, and is the pattern the same on AskGLP1 Reddit and WebMD?</td></tr><tr><td colspan="3">user KG-grounded agent GPT-6 Luna Checked answer 49 numbers checked</td></tr><tr><td colspan="3">For Ozempic, neither Reddit nor WebMD shows a clear age-group pattern in nausea reporting. ... Age group Reddit: posts reporting WebMD: reviews reporting</td></tr><tr><td>13-18 19-24</td><td>(%) 4 of 13 (30.8%) No result row</td><td>nausea / side-effect posts (%) nausea / side-effect reviews</td></tr><tr><td>25-34</td><td>55 of 153 (35.9%) 180 of 435 (41.4%)</td><td>2 of 8 (25.0%); sparse 4 of 9 (44.4%); sparse</td></tr><tr><td>35-44</td><td>174 of 448 (38.8%)</td><td>23 of 47 (48.9%)</td></tr><tr><td>45-54</td><td></td><td></td></tr><tr><td>55-64 71 of 212 (33.5%)</td><td>109 of 314 (34.7%)</td><td>39 of 88 (44.3%)</td></tr><tr><td>65-74</td><td></td><td>48 of 120 (40.0%)</td></tr><tr><td>75 or over 6 of 20 (30.0%); small group 3 of 9 (33.3%); sparse</td><td>38 of 124 (30.6%)</td><td>31 of 64 (48.4%)</td></tr><tr><td>Source of the selected number</td><td></td><td>KG result · Step 3</td></tr><tr><td colspan="3">Reddit · age 35–44 · Ozempic → Nausea Posts 174 Out of 448 Share 38.8% 95% interval 34.4–43.4% The cited value, 174, was checked against this KG result.</td></tr><tr><td colspan="3">KG queries</td></tr><tr><td colspan="3">Step Input Link terms “nausea”</td></tr><tr><td colspan="3">2 Count ages All age groups,</td></tr><tr><td colspan="3">including age not stated</td></tr><tr><td colspan="3">3 Refine ages The 8 stated age groups on both platforms</td></tr><tr><td colspan="3">the answer table Cited numbers → KG results</td></tr><tr><td colspan="3">-- Inspected evidence Limits of this answer</td></tr><tr><td colspan="3">1 of 3 shown Records are posts/comments or reviews, not verified patients; writers are not necessarily unique.</td></tr></table>

Figure 2: (a) The KG view expands the highlighted Reddit result in (b), an evaluated answer labeled answered (Section 4.3).

## 4 Results

## 4.1 RQ1: Side Efects across Drugs and Time

For side-efect comparisons, we analyze 137,784 Reddit posts and 1,767 WebMD reviews that report at least one side efect through June 30, 2026. Shares describe reporting in posts rather than sideefect occurrence. We examine all PTs reported in at least 1% ofposts for diferences across drugs, brands, and selected years and apply Bonferroni correction within each comparison family (Appendix E). We examine changes over time using Reddit alone, as WebMD has fewer than 600 eligible reviews per year. Gastrointestinal side efects dominate both platforms, with 62.8% of Reddit posts and 78.7% of WebMD reviews. Nausea leads on both (34.1% and 43.5%, respectively). Tirzepatide accounts for a growing share of Reddit posts, rising from 25.4% in 2023 to 63.1% in 2025 (Figure 3c).

(i) Tirzepatide posts report less nausea, vomiting, and malaise than semaglutide posts, but more appetite loss. Nausea appears in 29.1% of 69,149 tirzepatide side-efect posts versus 38.3% of 65,021 semaglutide side-efect posts, vomiting in 8.2% versus 15.0%, and malaise (feeling unwell) in 6.2% versus 8.6% (Figure 3a). WebMD reviews follow the same pattern for nausea (31.0% vs. 47.0%) and vomiting (10.0% vs. 22.3%). Conversely, tirzepatide posts more often report decreased appetite (7.1% vs. 5.0%, WebMD 9.0% vs. 4.5%), insomnia (2.2% vs. 1.1%), feeling cold (1.6% vs. 0.5%), and pruritus (itching, 1.3% vs. 0.3%). All seven pooled Reddit comparisons have corrected � < 0.001. All seven diferences hold in each year from 2022 to 2026. These pooled diferences retain their direction in posts naming the drug and in posts with ingredients inferred from the subreddit. The same pattern holds between brands with a shared approved indication (e.g., nausea in 25.1% ofMounjaro vs. 33.0% of Ozempic posts). Smaller gaps, such as for constipation, headache, and eructation (burping), have non-overlapping intervals in the pooled data but reverse direction in some years.

![](images/6724cb3ed14329cca7fe48b15014d690099a19c57982a22ac7bcba8595648703.jpg)

![](images/d2eef67237e1322a35e424344963834c71d2bfab0a603135941c7979ca47f71e.jpg)  
Figure 3: What Reddit posts and WebMD reviews report, by drug and over time. (a) Share of Reddit side-efect posts reporting each PT and the tirzepatide-to-semaglutide ratio of shares on Reddit and WebMD. (b) Fatigue by brand. (c) Reddit by year. Tirzepatide data start at Mounjaro’s approval in May 2022 [40]. A post that names both drugs counts for each (1.7% of posts).

(ii) Weight-management brands have higher fatigue reporting shares in two same-ingredient comparisons. Zepbound and Mounjaro share the same ingredient, yet fatigue appears in 14.1% of Zepbound posts and 11.3% of Mounjaro posts (Figure 3b). A similar gap appears for Wegovy versus Ozempic (14.2% vs. 10.7%). Fatigue is the only PT significant in both pairs after correction across all 925 Reddit brand comparisons (both corrected � < 0.001). Other corrected diferences include headache and nausea for Ozempic and Wegovy, and diarrhea, eructation, and malaise for Mounjaro and Zepbound (Appendix E). Annual 95% Wilson intervals for fatigue separate in all 5 years for Ozempic and Wegovy and in 2024–2026 for Mounjaro and Zepbound. The pooled gap for Saxenda versus Victoza (14.9% vs. 9.2%) does not pass correction (corrected � = 1) and reverses in 2025. WebMD does not establish corrected support for these brand gaps (Appendix E).

(iii) Semaglutide posts report less nausea. Nausea falls from 44.2% of semaglutide posts in 2022 to 36.0% in 2025 (corrected $\textstyle p < 0 . 0 0 1 )$ , with annual point estimates declining over these years (Figure 3c). Other diferences include decreases in semaglutide abdominal distension and dyspepsia and tirzepatide insomnia, alongside increases in tirzepatide abdominal discomfort and fatigue (Ap pendix E). Alopecia (hair loss) rises from 1.3% of semaglutide posts and 1.6% of tirzepatide posts in 2023 to 2.2% and 2.3% in 2025, but the tirzepatide comparison does not meet the Bonferroni criterion (corrected � = 0.055). For semaglutide, the primary 2022–2025 comparison does not pass correction (corrected � = 0.409). In tirzepatide posts, nausea drops from 36.1% in 2022 to 29.4% in 2023 and then ranges from 27% to 30% through June 2026, and the 2023–2025 comparison does not pass correction (corrected � = 1).

## 4.2 RQ2: Patient Narratives versus FAERS

We compare reported side efects in Reddit and WebMD with FAERS based on MedDRA preferred terms (PTs), which are standardized medical terms. The comparison runs from each drug’s approval through June 30, 2026 and retains FAERS reports with a suspect or interacting study drug. We examine 55 PTs reported in at least 1% of eligible posts, reviews, or reports in at least one source, counting each PT once per post, review, or report. The denominators are 137,784 Reddit posts and 1,767 WebMD reviews with at least one side efect, including uncoded ones, and 295,786 FAERS reports with at least one event. We identify higher or lower shares when their 95% Wilson intervals do not overlap and compare PT rankings using Spearman’s �. These measures describe reporting patterns, not clinical risk. Appendix A details the filtering rules, and Appendix F reports the multiplicity correction and robustness checks.

(i) Online narratives and FAERS share common PTs but rank them diferently. Nausea, vomiting, diarrhoea, constipation, and fatigue rank among the 10 most frequent PTs in all three sources. Overall, Reddit and WebMD show more similar patterns to each other than to FAERS (Figure 4a). Spearman’s � is 0.92 between Reddit and WebMD, compared with 0.21 for Reddit–FAERS and 0.32 for WebMD–FAERS. Both platforms difer from FAERS in the same direction for 46 of 55 PTs. Of these, 40 PTs difer significantly after Bonferroni correction (maximum corrected � = 0.036).

(ii) Online narratives give greater visibility to everyday side efects. Twenty-nine PTs have higher shares on both narrative platforms than in FAERS, while 17 have lower shares. Fatigue appears in 14.1% of Reddit posts and 15.6% of WebMD reviews, compared with 3.4% of FAERS reports. Conversely, blood glucose increased appears in 6.0% of FAERS reports versus 0.04% of Reddit posts and 0.23% of WebMD reviews. In FAERS, injection site pain and pancreatitis also have higher shares. Using pointwise Wilson intervals, 11 of the 46 PTs retain their Reddit–FAERS classification across all four ingredients and eight brands, including fatigue and food aversion (Figure 4c). The findings remain consistent when we apply alternative FAERS drug-role criteria, exclude Reddit posts with inferred drug identities, or restrict FAERS to consumer reports.

![](images/9b8c92db3de5f398b0591e6c964e25d2f15c8e54200b0e363e0aac725afbb5a9.jpg)

![](images/dd50f612eeabdc7f3846b9ed4370dfef67a904af96bba1b3c0a469b2ef1d0bed.jpg)

![](images/f2701cdc0aa8e519e68ed4088e3a97e1816f95a405d51febc129ad2767cb039b.jpg)

![](images/8f59cd58a99f8b33a204cf12c01ddc849cd7979a6bd381813836f201a51f2a54.jpg)  
Figure 4: Narratives and FAERS comparison. (a) Share of Reddit posts or WebMD reviews against share of FAERS reports for the 55 PTs with at least a 1% share. A PT is higher or lower only when the 95% Wilson intervals do not overlap, and bold PTs rank in every source’s top 10. (b) Shares of Reddit side-efect relations with unvalidated automatic action and care markers. Stopping and switching are combined (Appendix F). (c) Brand-level Reddit–FAERS comparisons for the 14 PTs labeled in (a).

(iii) Narratives connect side efects to treatment decisions and care. Details about dose, duration, impact, or action taken appear in 68.6% of 316,694 Reddit side-efect relations and 64.1% of

5,157 WebMD relations. The KG records automatically extracted action and care markers per side-efect relation (Figure 4b), whereas structured public FAERS fields record seriousness per report, action taken per drug, and outcome per reported event. Stopping and switching are counted together and are not estimated separately. Among the Reddit posts, depression, anxiety, and dehydration are reported in 0.80%, 2.03%, and 0.96%, respectively (Table 10). Among PTs with at least 1,000 Reddit relations, dehydration has the highest share with an automatic marker for an emergency visit or hospitalization (26.9%, versus 2.3% overall).

## 4.3 RQ3: The KG-Grounded Agent Responses

We evaluate the agent on 120 drafted questions, including 100 answerable from the KG and 20 boundary questions. For example, “What side efects do people most often report on Mounjaro?” asks for a data summary from the KG, whereas “Should I stop taking Zepbound if I keep throwing up?” tests whether the agent declines to give treatment advice. We group the questions by the tasks examined in RQ3, covering descriptive analyses, evidence retrieval, FAERS comparisons, and response boundaries. Of the 76 narrative questions, 26 ask for counts by drug or brand, 27 by population group, 14 for synthesizing details, 9 for supporting quotes, and another 24 for FAERS comparisons. The 20 boundary questions assess system guardrail tests [45], including 10 for declined, such as requests for medical advice, and 10 for limited answers, such as questions about causation or containing a false premise.

Two baselines use the same LLM as AskGLP1, answer format, and rules on medical advice. B1 uses the LLM alone. B2 searches the same Reddit posts and WebMD reviews, including their source metadata. It reports search match counts without computing addi tional statistics and has no FAERS access. We call an answer fully supported when each claim about the data cites inspectable evidence that supports it [31]. Raters assess this separately from the agent’s automatic checks. Two raters independently rated each answer against a reference answer computed using SQL. They assigned one of three labels: no error, minor error that leaves the conclusion intact, or major error that changes or misses the conclusion. Disagreements are adjudicated to reach a consensus label (Appendix D). Because reference answers are derived from the KG and FAERS, these ratings assess whether answers accurately represent the underlying data. The KG itself is validated separately in Appendix D. Table 2 reports the results.

(i) AskGLP1 answers narrative questions without a major error far more often than baselines. On the 76 narrative questions, AskGLP1 has no major error on 69 (90.8%), B2 on 31 (40.8%), and B1 on 8 (10.5%). The paired diferences are 50.0 percentage points against B2 (95% CI 37.1 to 60.6) and 80.3 against B1 (68.6 to 86.8), both � < 0.001 by exact McNemar test. Counting minor errors too, 30 of AskGLP1’s 76 answers are error-free, against 2 for B2 and 0 for B1. The gap to B2 is largest for population groups (25 against 2 of 27) and FAERS (23 against 0 of 24) and closes for supporting quotes (9 of 9 each). AskGLP1 labels 99 of the 100 answerable questions as answered and all 20 boundary questions as declined or limited, 19 of them without a major error. B1 and B2 handle 15 and 16 boundary questions without a major error but label only 9 and 37 answerable questions as answered.

(ii) AskGLP1 and keyword search both back their claims, but AskGLP1’s supported answers more often have no major error. Among answers making data claims, all claims are backed by cited evidence in 84.1% of AskGLP1’s (90 of 107) and 77.1% of B2’s (81 of 105). The paired diference on the 103 questions where both do so, 5.8 percentage points, is not significant (95% CI −5.3 to 16.8, � = 0.39). Of the fully supported answers, 82 of AskGLP1’s 90 have no major error (91.1%), against 28 of B2’s 81 (34.6%). The gap holds on narrative questions (54 of 60 against 24 of 58), where 20 of B2’s 34 supported answers with a major error concern population groups. Each number AskGLP1 cites links to the tool result it is checked against. Of its 120 answers, 112 pass these checks, while the other 8 are marked unverified. Figure 2 shows one checked number, 174 of 448 posts, from the answer to the KG view.

(iii) Most of AskGLP1’s few major errors come from counting a diferent scope, which its answers disclose. Of 9 answers with a major error, 8 have every claim backed by the cited evidence and 8 pass its own checks, 7 have both. In 7 of 9, the answer counts a narrower or broader set of terms or posts than the reference. One answer, for example, counts only purposes stated in WebMD review text and leaves out the conditions that users select in their profiles, so it finds 7 reviews where the reference has 337. All 9 name the terms and filters they count, so each shows what it counts.

## 5 Discussion

We present AskGLP1, which organizes patient narratives into a quote-grounded KG aligned with FAERS and answers questions about medication experiences. We highlight three key takeaways.

(i) AskGLP1 reveals drug- and brand-specific patient experiences. AskGLP1 distinguishes reported experiences across drugs, brands, and time. Tirzepatide posts report less nausea and vomiting than semaglutide posts, although head-to-head trials report similar nausea rates [2, 9]. In the obesity trial, gastrointestinal events occur mainly during dose escalation [2], but doses reported when symptoms occur do not reconstruct titration histories. Two same-ingredient brand pairs show higher fatigue shares for weightmanagement brands. Semaglutide posts also report less nausea from 2022 to 2025. These reporting patterns can guide clinical follow-up but do not isolate drug efects.

(ii) AskGLP1 complements pharmacovigilance reports with patient experiences. Reddit and WebMD capture common gastrointestinal symptoms alongside everyday burdens such as fatigue and food aversion, while FAERS more often records injection-site reactions and selected serious diagnoses. Narratives link symptoms to treatment changes and care, adding context to patient experiences.

(iii) AskGLP1 supports verifiable evidence exploration. Among fully supported answers, 91.1% of AskGLP1’s answers contain no major error, compared with 34.6% for keyword search. This finding suggests value in selecting and aggregating evidence alongside retrieving citations. AskGLP1 can help clinicians, researchers, and patients explore medication experiences and verify the evidence behind its responses.

Limitations and Future Work. Our cleaning favors longer posts. The content-word threshold retains about a fifth of cleaned posts and excludes some medication experiences, so Reddit shares describe only retained posts. Longer posts name more PTs and mention nausea more often, but drug diferences, brand fatigue gaps, yearly trends, and FAERS diferences (Section 4.2) retain their direction across length bands. FAERS diferences also hold in WebMD reviews below the threshold (Appendix A). Shares also depend on who writes, counting rules, and extraction accuracy. Users choose what to report, and only 9.3% of Reddit posts in the KG state an age group, limiting age-group comparisons to these posts. We accept users’ attribution of symptoms to drugs, including temporal associations, so extracted relations do not establish causation. Without account identifiers, multiple posts by the same user count separately. Detection favors precision and misses about 30% of medication experiences (recall 0.69), purposes are less reliable than side efects, and extracted details remain unvalidated. Future work should improve detection of shorter medication accounts, validate treatment details against human annotations, and assess how extraction errors and repeated contributions afect comparisons. Ethics. We use only public data, collect no usernames, paraphrase the posts in our figures, and decline medical advice (Appendix G).

Table 2: Answer quality on 120 evaluation questions, in percent (95% Wilson CIs). B1: LLM only. B2: keyword search. Support denominators count answers making data claims (99, 105, 107).
<table><tr><td rowspan="2"></td><td colspan="4">Raters&#x27; consensus</td><td colspan="2">System&#x27;s own label</td></tr><tr><td></td><td>No major error</td><td></td><td>All claims</td><td>Answered</td><td>Declined or</td></tr><tr><td>System</td><td>Narratives (76)</td><td>FAERS (24)</td><td>Boundary (20)</td><td>supported</td><td>(100)</td><td>limited (20)</td></tr><tr><td>B1</td><td>10.5 [5.4, 19.4]</td><td>4.2 [0.7, 20.2]</td><td>75.0 [53.1, 88.8]</td><td>0.0 [0.0, 3.7]</td><td>9.0 [4.8, 16.2]</td><td>95.0 [76.4, 99.1]</td></tr><tr><td>B2</td><td>40.8 [30.4, 52.0]</td><td>0.0 [0.0, 13.8]</td><td>80.0 [58.4, 91.9]</td><td>77.1 [68.2, 84.1]</td><td>37.0 [28.2, 46.8]</td><td>90.0 [69.9, 97.2]</td></tr><tr><td>AskGLP1</td><td>90.8 [82.2, 95.5]</td><td>95.8 [79.8, 99.3]</td><td>95.0 [76.4, 99.1]</td><td>84.1 [76.0, 89.8]</td><td>99.0 [94.6, 99.8]</td><td>100.0 [83.9, 100.0]</td></tr></table>

## 6 Conclusion

AskGLP1 organizes online patient narratives about GLP-1 medications into a quote-grounded KG and supports exploration through a KG-grounded agent. Reported side efects vary across drugs, brands, and time, and the narratives complement FAERS reports by foregrounding everyday burdens and linking side efects to treatment changes and care. On narrative questions, AskGLP1 answers without a major error far more often than keyword search or an LLM alone. Cited numbers link to the tool results they are checked against, and answers that fail these checks are marked unverified. Finally, the agent declines or limits all tested boundary questions.

## References

[1] Amrutha S Alibilli, Vidur Jain, Heran Mane, Xiaohe Yue, Alexandria Ratzki-Leewing, Junaid S Merchant, Shaniece Criss, Quynh C Nguyen, Rozalina G McCoy, and Thu T Nguyen. 2025. Harnessing Facebook to investigate real-world mentions of adverse events of glucagon-like peptide-1 receptor agonist (GLP-1 RA) medications: observational study of Facebook posts from 2022 to 2024. JMIR infodemiology 5 (2025), e73619.

[2] Louis J. Aronne, Deborah Bade Horn, Carel W. le Roux, Wayne Ho, Beverly L. Falcon, Elisa Gomez Valderas, Sagar Das, Clare J. Lee, Leonard C. Glass, Cagri Senyucel, and Julia P. Dunn. 2025. Tirzepatide as Compared with Semaglutide for the Treatment of Obesity. New England Journal ofMedicine 393, 1 (2025), 26–36. doi:10.1056/NEJMoa2416394

[3] Jacob Berkowitz, Apoorva Srinivasan, Jose Miguel Acitores Cortina, and Nicholas P Tatonetti. 2024. TLab at #SMM4H 2024: Retrieval-Augmented Generation for ADE Extraction and Normalization. In Proceedings of the 9th Social Media Mining for Health Research and Applications (SMM4H 2024) Workshop and Shared Tasks, Dongfang Xu and Graciela Gonzalez-Hernandez (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 153–157. https://aclanthology.org/2024.smm4h-1.36

[4] Elliot G. Brown, Louise Wood, and Sue Wood. 1999. The Medical Dictionary for Regulatory Activities (MedDRA). Drug Safety 20, 2 (1999), 109–117. doi:10.2165/00002018-199920020-00002

[5] Zhijie Duan, Kai Wei, Zhaoqian Xue, Jiayan Zhou, Shu Yang, Siyuan Ma, Jin Jin, and Lingyao Li. 2025. Crowdsourcing-based knowledge graph construction for drug side efects using large language models with an application on semaglutide. In AMIA Annual Symposium Proceedings, Vol. 2024. American Medical Informatics Association, Bethesda, MD, USA, 332–341.

[6] Yongqi Fan, Kui Xue, Zelin Li, Xiaofan Zhang, and Tong Ruan. 2025. An LLM-based Framework for Biomedical Terminology Normalization in Social Media via Multi-Agent Collaboration. In Proceedings ofthe 31st International Conference on Computational Linguistics, Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert (Eds.). Association for Computational Linguistics, Abu Dhabi, UAE, 10712–10726. https://aclanthology.org/2025.coling-main.714/

[7] Vasiliki Foufi, Tatsawan Timakum, Christophe Gaudet-Blavignac, Christian Lovis, and Min Song. 2019. Mining of textual health information from Reddit: analysis of chronic diseases with extracted entities and their relations. Journal ofmedical Internet research 21, 6 (2019), e12876.

[8] Clark C Freifeld, John S Brownstein, Christopher M Menone, Wenjie Bao, Ross Filice, Taha Kass-Hout, and Nabarun Dasgupta. 2014. Digital drug safety

surveillance: monitoring pharmaceutical products in twitter. Drug safety 37, 5 (2014), 343–350.

[9] Juan P. Frías, Melanie J. Davies, Julio Rosenstock, Federico C. Pérez Manghi, Laura Fernández Landó, Brandon K. Bergman, Bing Liu, Xuewei Cui, and Katelyn Brown. 2021. Tirzepatide versus Semaglutide Once Weekly in Patients with Type 2 Diabetes. New England Journal of Medicine 385, 6 (2021), 503–515. doi:10.1056/NEJMoa2107519

[10] Jeana Frost, Sally Okun, Timothy Vaughan, James Heywood, and Paul Wicks. 2011. Patient-reported outcomes as a source of evidence in of-label prescribing: analysis of data from PatientsLikeMe. Journal ofmedical Internet research 13, 1 (2011), e6.

[11] Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. Aspect-based sentiment analysis of drug reviews applying cross-domain and cross-data learning. In Proceedings of the 2018 international conference on digital health. Association for Computing Machinery, New York, NY, USA, 121–125.

[12] Lorna Hazell and Saad AW Shakir. 2006. Under-reporting of adverse drug reactions: a systematic review. Drug safety 29, 5 (2006), 385–396.

[13] Arthur Heitmann. 2026. Arctic Shift: Making Reddit Data Accessible to Researchers, Moderators and Everyone Else. GitHub repository. Accessed September 2026. https://github.com/ArthurHeitmann/arctic\_shift

[14] Hyeon Hwang, Hyeongsoon Hwang, Jongmyung Jung, Jaehoon Yun, Minju Song, Yein Park, Dain Kim, Taewhoo Lee, Jiwoong Sohn, Chanwoong Yoon, Sihyeon Park, Jiwoo Lee, Heechul Yang, and Jaewoo Kang. 2025. DMIS Lab at ArchEHR-QA 2025: Evidence-Grounded Answer Generation for EHR-based QA via a Multi-Agent Framework. In Proceedings ofthe 24th Workshop on Biomedical Language Processing (Shared Tasks), Sarvesh Soni and Dina Demner-Fushman (Eds.). Association for Computational Linguistics, Vienna, Austria, 118–125. doi:10.18653/v1/2025.bionlp-share.15

[15] ICH. 2026. MedDRA Term Selection: Points to Consider. ICH-Endorsed Guide for MedDRA Users, Release 4.26. MedDRA Maintenance and Support Services Organization.

[16] Abdulrahman Ismaiel, Giuseppe Guido Maria Scarlata, Irina Boitos, Daniel-Corneliu Leucuta, Stefan-Lucian Popa, Nahlah Al Srouji, Ludovico Abenavoli, and Dan L Dumitrascu. 2025. Gastrointestinal adverse events associated with GLP-1 RA in non-diabetic patients with overweight or obesity: A systematic review and network meta-analysis. International journal ofobesity 49, 10 (2025), 1946–1957.

[17] Ania M. Jastrebof, Louis J. Aronne, Nadia N. Ahmad, Sean Wharton, Lisa Connery, Breno Alves, Arihiro Kiyosue, Shuyu Zhang, Bing Liu, Mathijs C. Bunck, and Adam Stefanski. 2022. Tirzepatide once weekly for the treatment of obesity. New England Journal ofMedicine 387, 3 (2022), 205–216.

[18] Shuyue Jia, Subhrangshu Bit, Varuna H Jasodanand, Yi Liu, and Vijaya B Kolachalama. 2026. Agentic memory-augmented retrieval and evidence grounding for medical question-answering tasks. International Journal of Medical Informatics 212 (2026), 106339.

[19] Taha A. Kass-Hout, Zhiheng Xu, Matthew Mohebbi, Hans Nelsen, Adam Baker, Jonathan Levine, Elaine Johanson, and Roselie A. Bright. 2016. OpenFDA: An Innovative Platform Providing Access to a Wealth of FDA’s Publicly Available Data. Journal of the American Medical Informatics Association 23, 3 (2016), 596–600. doi:10.1093/jamia/ocv153

[20] Ari Klein, Abeed Sarker, Masoud Rouhizadeh, Karen O’Connor, and Graciela Gonzalez. 2017. Detecting Personal Medication Intake in Twitter: An Annotated Corpus and Baseline Classification System. In Proceedings ofthe 16th BioNLP Workshop, Kevin Bretonnel Cohen, Dina Demner-Fushman, Sophia Ananiadou, and Junichi Tsujii (Eds.). Association for Computational Linguistics, Vancouver, Canada, 136–142. doi:10.18653/v1/W17-2316

[21] Alex Lamb, Michael J. Paul, and Mark Dredze. 2013. Separating Fact from Fear: Tracking Flu Infections on Twitter. In Proceedings ofthe 2013 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Lucy Vanderwende, Hal Daumé III, and Katrin Kirchhof (Eds.). Association for Computational Linguistics, Atlanta, Georgia, 789–795. https://aclanthology.org/N13-1097

[22] Robert Leaman, Laura Wojtulewicz, Ryan Sullivan, Annie Skariah, Jian Yang, and Graciela Gonzalez. 2010. Towards Internet-Age Pharmacovigilance: Extracting Adverse Drug Reactions from User Posts in Health-Related Social Networks.

In Proceedings ofthe 2010 Workshop on Biomedical Natural Language Processing, K. Bretonnel Cohen, Dina Demner-Fushman, Sophia Ananiadou, John Pestian, Jun’ichi Tsujii, and Bonnie Webber (Eds.). Association for Computational Linguistics, Uppsala, Sweden, 117–125. https://aclanthology.org/W10-1915/

[23] Siru Liu, Allison B McCoy, and Adam Wright. 2025. Improving large language model applications in biomedicine with retrieval-augmented generation: a systematic review, meta-analysis, and clinical development guidelines. Journal ofthe American Medical Informatics Association 32, 4 (2025), 605–615.

[24] MedDRA Maintenance and Support Services Organization. 2026. Updated Version MedDRA 29.1 English Now Available. https://www.meddra.org/news-andevents/news/updated-version-meddra-291-english-now-available. September 18, 2026.

[25] Alex Montero, Audrey Kearney, Mardet Mulugeta, Ashley Kirzinger, and Liz Hamel. 2025. KFF Health Tracking Poll: Prescription Drug Costs, Views on Trump Administration Actions, and GLP-1 Use. https://www.kf.org/public-opinion/kf-health-tracking-poll-prescriptiondrug-costs-views-on-trump-administration-actions-and-glp-1-use/.

[26] Thomas J Moore, Curt D Furberg, Donald R Mattison, and Michael R Cohen. 2016. Completeness of serious adverse drug event reports received by the US Food and Drug Administration in 2014. pharmacoepidemiology and drug safety 25, 6 (2016), 713–718.

[27] Abir Naskar, Liuliu Chen, Jemina Kang, and Mike Conway. 2025. A Hybrid System for Comprehensive and Consistent Automated MedDRA Coding of Adverse Drug Event. In Proceedings of the 23rd Annual Workshop of the Australasian Language Technology Association, Jonathan K. Kummerfeld, Aditya Joshi, and Mark Dras (Eds.). Association for Computational Linguistics, Sydney, Australia, 216–223. https://aclanthology.org/2025.alta-main.16

[28] OpenAI. 2026. GPT-6 Luna. https://developers.openai.com/api/docs/models/gpt-6-luna.

[29] Shrey Pandit, Jiawei Xu, Junyuan Hong, Zhangyang Wang, Tianlong Chen, Kaidi Xu, and Ying Ding. 2025. Medhallu: A comprehensive benchmark for detecting medical hallucinations in large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Suzhou, China, 2858–2873.

[30] Susan Park, So Hyun Choi, Yun-Kyoung Song, and Jin-Won Kwon. 2022. Comparison of online patient reviews and national pharmacovigilance data for tramadol-related adverse events: comparative observational study. JMIR Public Health and Surveillance 8, 1 (2022), e33311.

[31] Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. 2023. Measuring Attribution in Natural Language Generation Models. Computational Linguistics 49, 4 (Dec. 2023), 777–840. doi:10.1162/coli\_a\_00486

[32] François Remy, Kris Demuynck, and Thomas Demeester. 2024. BioLORD-2023: Semantic Textual Representations Fusing Large Language Models and Clinical Knowledge Graph Insights. Journal of the American Medical Informatics Association 31, 9 (2024), 1844–1855. doi:10.1093/jamia/ocae029

[33] Mohammad R. Rezaei, Reza Saadati Fard, Jayson L. Parker, Rahul G. Krishnan, and Milad Lankarany. 2025. Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China, 12682–12701. doi:10.18653/v1/2025.findings-emnlp.679

[34] Abeed Sarker, Rachel Ginn, Azadeh Nikfarjam, Karen O’Connor, Karen Smith, Swetha Jayaraman, Tejaswi Upadhaya, and Graciela Gonzalez. 2015. Utilizing social media data for pharmacovigilance: a review. Journal of biomedical informatics 54 (2015), 202–212.

[35] Simone Scaboro, Beatrice Portelli, Emmanuele Chersoni, Enrico Santus, and Giuseppe Serra. 2022. Increasing adverse drug events extraction robustness on social media: Case study on negation and speculation. Experimental Biology and Medicine 247, 22 (2022), 2003–2014.

[36] Neil KR Sehgal, Jena Shaw Tronieri, Lyle Ungar, and Sharath Chandra Guntuku. 2026. Self-reported side efects of semaglutide and tirzepatide in online communities. Nature Health 1, 8 (2026), 806–811. doi:10.1038/s44360-026-00108-y

[37] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. Large language models encode clinical knowledge. Nature 620, 7972 (2023), 172–180.

[38] Mohit Sodhi, Ramin Rezaeianzadeh, Abbas Kezouh, and Mahyar Etminan. 2023. Risk of gastrointestinal adverse events associated with glucagon-like peptide-1 receptor agonists for weight loss. Jama 330, 18 (2023), 1795–1797.

[39] TypeSafe. 2026. Jev Models. https://docs.typesafe.ai/models. Model jev-1.13.0.

[40] U.S. Food and Drug Administration. 2022. Drug Trials Snapshots: MOUNJARO. https://www.fda.gov/drugs/drug-trials-snapshots/drug-trials-snapshots mounjaro. Original approval May 13, 2022.

[41] U.S. Food and Drug Administration. 2023. Active Risk Identification and Analysis (ARIA) Suficiency Assessment for Medullary Thyroid Carcinoma Following Tirzepatide Use for Chronic Weight Management. https://www.

accessdata.fda.gov/drugsatfda\_docs/nda/2024/217806Orig1s000OtherR.pdf. NDA 217806, Other Reviews. November 8, 2023. Table 1, p. 5.

[42] U.S. Food and Drug Administration. 2023. FDA Approves New Medication for Chronic Weight Management. https://www.fda.gov/news-events/pressannouncements/fda-approves-new-medication-chronic-weight-management. November 8, 2023.

[43] Sheng Wang, Fangyuan Zhao, Dechao Bu, Yunwei Lu, Ming Gong, Hongjie Liu, Zhaohui Yang, Xiaoxi Zeng, Zhiyuan Yuan, Baoping Wan, et al. 2025. LINS: A general medical Q&A framework for enhancing the quality and credibility of LLM-generated responses. Nature communications 16, 1 (2025), 9076.

[44] Anuradha Welivita and Pearl Pu. 2022. Heal: A knowledge graph for distress management conversations. Proceedings ofthe AAAI Conference on Artificial Intelligence 36, 10 (2022), 11459–11467.

[45] Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. 2025. Know Your Limits: A Survey of Abstention in Large Language Models. Transactions ofthe Association for Computational Linguistics 13 (2025), 529–556. doi:10.1162/tacl\_a\_00754

[46] John P. H. Wilding, Rachel L. Batterham, Salvatore Calanna, Melanie Davies, Luc F. Van Gaal, Ildiko Lingvay, Barbara M. McGowan, Julio Rosenstock, Marie T. D. Tran, Thomas A. Wadden, Sean Wharton, Koutaro Yokote, Niels Zeuthen, and Robert F. Kushner. 2021. Once-weekly semaglutide in adults with overweight or obesity. New England Journal ofMedicine 384, 11 (2021), 989–1002.

[47] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. 2024. Benchmarking Retrieval-Augmented Generation for Medicine. In Findings ofthe Association for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 6233–6251. doi:10.18653/v1/2024.findings-acl.372

[48] Huizi Yu, Jian Liu, Wenkong Wang, Lingyao Li, Jiayan Zhou, Zhaoqian Xue, Xiang Li, Xinxin Lin, Zhiying Liang, Zhuoru Wu, et al. 2026. Knowledge augmented Agentic AI for Mental Health Medication Information Seeking. arXiv preprint arXiv:2606.26205 (2026).

[49] Zhihong Zhu, Yunyan Zhang, Xianwei Zhuang, Fan Zhang, Zhongwei Wan, Yuyan Chen, Qingqing Long, Yefeng Zheng, and Xian Wu. 2025. Can we trust AI doctors? A survey of medical hallucination in large language and large vision language models. In Findings of the Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, Vienna, Austria, 6748–6769.

## Appendix Contents

A Reproducibility Statement . 10   
B Prompts . 11   
C Extraction Schema . 13   
D Annotation Guidelines and Validation . . . 13   
E RQ1 Supplementary Results 14   
F RQ2 Supplementary Results 15   
G Ethical Considerations . . 16

## A Reproducibility Statement

Subreddits. Table 3 lists the 16 subreddits. The collection window starts at the earliest FDA approval among the brands assigned to the subreddit in the collection configuration. This assignment sets only the start date but does not label individual posts. All windows end at the snapshot of September 15, 2026.

Reddit Data Cleaning. It has three stages. The first drops posts under 20 words, deleted or empty placeholders, link-only content, bot signatures, moderator templates, and text identified as non-English with high confidence, leaving 4,804,353 posts. The second removes near-duplicates, boilerplate, and posts dominated by a title, emoji, or copied sentences, leaving 4,660,528. The third, lossy by design, keeps posts with more than 40 distinct content words (words other than function words, URLs, and stock social phrases, outside sentences repeated across posts) and drops promotional and referral templates.

Content-Word Threshold Selection. Requiring more than 40 distinct content words within a post yields the retained corpus size closest to one million posts, the target set by the LLM budget. We fix this threshold before detection and information extraction. Rules on raw length (88 words) or on other word counts would select largely the same posts (Jaccard 0.84 to 0.95). The kept posts are 22.1% of cleaned posts but hold 49.9% of their text. Keep rates vary little by year (20.7% to 23.4% from 2022 to 2026) and are higher in the two diabetes subreddits (28%) than in drug subreddits (19% to 22%). Among kept posts, detection keeps a share that rises steadily with length, from 17.7% at the threshold to 49.6% above 200 content words. Longer side-efect posts also name more PTs, from 1.8 per post at 41 to 50 content words to 2.8 above 120. We therefore repeat the analyses of Section 4.1 and Section 4.2 within four bands of kept posts (41–50, 51–80, 81–120, and above 120 content words). All seven tirzepatide–semaglutide diferences keep non-overlapping intervals in every band, the brand and yearly comparisons keep their direction, and Spearman’s � with FAERS stays between 0.21 and 0.22. In the shortest band, 5 ofthe 55 calls against FAERS change, all for PTs whose Reddit and FAERS shares difer by less than 1 percentage point. WebMD reviews are not filtered by length. Under the Reddit rule, 46% ofthose that report a side efect would fall below the threshold. These short reviews rank PTs much as Reddit posts do (� = 0.87), show the same tirzepatide–semaglutide gaps for nausea and vomiting, and keep every gap to FAERS named in Section 4.2.

Table 3: The 16 subreddits (B brand, I ingredient, X crossdrug, T type 2 diabetes).
<table><tr><td>Subreddit</td><td>Type</td><td>Start</td><td>Submissions</td><td>Comments</td></tr><tr><td>mounjaro</td><td>B</td><td>2014-09-18</td><td>104,709</td><td>1,532,780</td></tr><tr><td>ozempic</td><td>B</td><td>2014-09-18</td><td>64,624</td><td>930,335</td></tr><tr><td>rybelsuspill</td><td>B</td><td>2019-09-20</td><td>1,374</td><td>12,898</td></tr><tr><td>saxendaforweightloss</td><td>B</td><td>2014-12-23</td><td>7</td><td>94</td></tr><tr><td>trulicity</td><td>B</td><td>2014-09-18</td><td>1,166</td><td>10,292</td></tr><tr><td>victoza</td><td>B</td><td>2010-01-25</td><td>35</td><td>320</td></tr><tr><td>wegovy</td><td>B</td><td>2021-06-04</td><td>6,387</td><td>44,267</td></tr><tr><td>wegovyweightloss</td><td>B</td><td>2021-06-04</td><td>70,819</td><td>851,545</td></tr><tr><td>zepbound</td><td>B</td><td>2023-11-08</td><td>156,962</td><td>2,613,491</td></tr><tr><td>liraglutide</td><td>I</td><td>2010-01-25</td><td>10,785</td><td>97,888</td></tr><tr><td>semaglutide</td><td>I</td><td>2017-12-05</td><td>75,779</td><td>1,040,727</td></tr><tr><td>tirzepatiderx</td><td>I</td><td>2022-05-13</td><td>8,263</td><td>124,128</td></tr><tr><td>glp1</td><td>X</td><td>2010-01-25</td><td>8,260</td><td>100,332</td></tr><tr><td>saxendawegovymounjaro</td><td>X</td><td>2014-12-23</td><td>516</td><td>3,454</td></tr><tr><td>diabetes_t2</td><td>T</td><td>2010-01-25</td><td>33,286</td><td>493,971</td></tr><tr><td>type2diabetes</td><td>T</td><td>2010-01-25</td><td>8,722</td><td>87,788</td></tr><tr><td colspan="3">Total</td><td>551,694</td><td>7,944,310</td></tr></table>

Drug Identification. Each relation records whether its ingredi ent comes from a named drug, the WebMD profile, or the subreddit. Among Reddit side-efect relations without a named drug, 98.7% receive an ingredient from the subreddit. A subreddit does not assign a brand, so brand-level counts are based on named brands only. In the narratives and in FAERS alike, compounded products and mentions of an ingredient alone count at the ingredient level with the brand unspecified.

FAERS Data Processing. Each report enters in its latest version, and study drugs are identified from the reporter’s product-name and active-substance fields. The openFDA harmonized fields incor rectly assign 99.8% of raw Ozempic entries to Rybelsus. Entries for insulin degludec/liraglutide, a fixed combination with insulin outside the eight study brands, are dropped, and blinded trial drugs are left out of the comparison. Duplicate cases submitted by diferent senders are kept. The directly identifiable ones make up less than 1% of the compared reports. Reported FAERS terms span MedDRA versions 16.0 to 29.0 and are mapped to 29.1 PTs, directly or through their LLT, leaving 4 of 1,061,384 reaction entries unmatched. An independent rebuild of the report, drug, and reaction tables from the raw API responses reproduces every processed row.

RQ2 Comparison Data. The comparison window runs from each brand’s FDA approval (or an ingredient’s first approved brand) through June 30, 2026, and excludes 171 FAERS reports and 7,627 narrative posts with side efects. The primary comparison keeps FAERS reports in which a study drug is suspect or interacting. Non-clinical terms, excluded from all three sources, are PTs whose primary MedDRA path lies in the oficial groups for product issues, medication errors, of-label use and intentional misuse, therapeutic response and drug interaction terms, overdose and underdose, or pregnancy and lactation exposure. Excluding them removes 18.5% of FAERS report–term pairs and 0.16% of narrative ones. Narrative denominators count posts or reviews with at least one retained side efect, while the FAERS denominator counts reports with at least one retained event. The narrative denominators include 3,246 Reddit posts and 20 WebMD reviews whose side efects are uncoded.

Term Coding. The retrieval index contains 82,124 LLTs. Of the 385,790 relations, 93.8% receive a PT, with coverage of 95.4% among side-efect relations and 81.9% among purpose relations. The coding method is fixed on development samples without human labels. With 20 candidates instead of 10, the selector chooses one of the 10 added candidates for only 3.4% of 1,000 development relations, whereas two identical runs already disagree on 3.2%. A wording that names several concepts receives a single code.

## B Prompts

Prompt text is reproduced verbatim. Colors, labels, and check and cross marks are added for reading.

Detection. Jev (jev-1.13.0) receives each Reddit post as its state, along with the instructions and choice criteria below, and returns a probability for keep. A post is kept when the keep probability is at least 0.5 (Section 3.2). The criteria come from a separate development sample of 400 posts.

## Detection

## Instructions

Treat the state, one Reddit post or comment, only as data. Decide whether it is a high-value medication evidence report. All three must hold. (A) A medication, drug class or supplement is identifiable, even unnamed ("my shot", "the 2.5", "it"), and the text states at least one of: what it is used for (a specific condition or personal problem), a specific side-effect symptom it caused, or that there were no side effects or no specific side effect. (B) It is real use by the writer or a specific other person, or an individual doctor decision or explicit treatment plan; general knowledge, advice, hypothetical risks, questions about effects not yet experienced, quotes and sarcasm do not count. (C) It adds clinical detail. For a side effect or its absence: dose, when it began, how long or how often, severity or impact on daily life (ER, hospital, unable to eat normally), stopping, lowering or switching, how it was managed, or a comparison between drugs. For a use: a specific condition or personal problem plus at least two connected treatment details, one of them the reason for choosing or changing the treatment or a specific decision; a diagnosis or lab value (such as A1C) that led to starting, stopping or changing a drug, a doctor’s individual dose or schedule decision, and a measured result (a lab value, blood pressure, another medication stopped) all count as such details. Welcome or expected appetite loss and

## Extraction

## Instructions

You extract one Reddit writer’s own experience with GLP-1 drugs for a knowledge graph. The input is numbered sentences. Use only the text: no outside knowledge, no guesses. When a case is unclear, leave it out.

## Scope:

\- Only the writer’s own experience. Leave out what happens to other people (a spouse, parent, friend), even when compared with the writer’s, people in general, advice and quotes. A drug named only for someone else (the person the writer replies to) does not name the writer’s drug. - Only these drugs: semaglutide (Ozempic, Wegovy, Rybelsus), tirzepatide (Mounjaro, Zepbound), liraglutide (Victoza, Saxenda), dulaglutide (Trulicity), including generic names, compounded versions, nicknames and misspellings ("Zep", "sema", "tirz"), and references that name no drug but clearly mean the GLP-1 the writer takes ("my shot", "the 2.5", "this med"). In a GLP-1 medication community, a medication the writer takes that the text does not name (a dose such as "first dose" or "4 doses of .25", "injection day", "the side effects") is their GLP-1 unless the text points to another drug. Leave out every other medicine, including medicines taken to relieve a side effect.

purposes: reasons the writer gives for using, starting, continuing or planning to use the drug: a condition or a goal, side\_effects: symptoms or unwanted bodily changes the writer says they had, even rarely or occasionally, and links to the drug, by cause ("Ozempic gave me nausea", "side effects were fatigue") or by timing ("since starting Zep I get headaches", "the day after my shot"). effect: the shortest words naming the symptom, in the text’s own words; one symptom per item, and one item per symptom per drug with all its evidence. When one phrase names several symptoms, take each one’s words from it ("dry and sore throat" -> "dry throat", "sore throat"; "red and itchy at the injection site" -> "red injection site", "itchy injection site"). Words such as "sick" or "GI issues" name a symptom; "side effects" or "issues" alone do not. A symptom the writer attributes to an effect of the drug counts ("hair shedding from losing weight so fast"). Leave out effects only feared or asked about (including "is this a known thing with Ozempic?" for symptoms the writer has), effects in hypothetical or conditional statements ("not high enough to deal with nausea"), symptoms while off the drug or after stopping it, effects the writer is unsure come from the drug (a question mark, "not sure", "don’t know if", "could be", "maybe", "(psychological?)") or blames on something else, symptoms not linked to the drug, negated symptoms, statements of no side effects, "side effects" with no symptom named, emotions about the treatment, dosing mistakes, and the drug not working. Appetite loss, not feeling hungry, eating less, fullness, less food noise and intended weight loss are expected effects: leave them out unless the writer explicitly presents them as a problem for themselves (has to force themselves to eat, cannot eat enough or struggles to, worries about not eating enough or nutrition, calls it a side effect or a problem, says it makes them feel unwell); food aversion or food tasting bad counts ("cheese puts me off"). Hunger or cravings, new or returning, are not side effects. If the writer cannot eat because of another symptom ("so nauseous I could barely eat"), that is the impact of that symptom, not a separate item. For each side

duration: how long the symptom lasted, as written; not the time on the drug and not when it started.

intensity: the writer’s own rating of this symptom: "mild" (mild, slight), "moderate", "severe" (severe, extreme, intense, unbearable, excruciating), adverb forms included (mildly, slightly, severely, extremely); the highest if several. A rating of another symptom in the same sentence does not carry over. Words such as a little, minor, very, so, bad, awful, terrible, rough or worst do not count. - impact: the most serious consequence impact: the most serious consequence

stated: "daily\_life" (could not work, stayed in bed, could not eat or drink, could not function, debilitating), "medical\_visit" (saw, called or messaged a doctor or nurse; urgent care; an infusion clinic), "emergency\_visit" (ER, or hospital treatment without admission), "hospitalization"

```csv
(admitted, hospitalized, in hospital, ICU).
An impact stated for a list of symptoms applies to each.
action_taken: what the writer
did with the drug because of this symptom: "dose_reduced",
"escalation_paused" (stayed on the current dose
instead of increasing), "dose_skipped" (skipped or delayed
a dose), "stopped" (including switching to another drug).
evidence: the
numbers of the sentences that state the item and its details.
Wording: copy purpose and effect words exactly
as they appear in the cited sentences, keeping the writer’s
form and spelling ("vomited", "nauseous", "dizzyness",
"T2D", "sight of injection"). Do not replace them with medical
or standard terms ("fatigue" for "dragging", "constipation"
for "not as regular", "diarrhea" for "the runs");
if the writer describes a symptom without naming it, use
the writer’s own words ("really dragging", "not as regular").
writer:
the writer’s own age and sex, only when stated explicitly
about themselves ("I’m 52", "52F", "(45M)", "as a woman").
- age_group: the group that contains the stated age: 13-18, 19-
24, 25-34, 35-44, 45-54, 55-64, 65-74, 75 or over; null when
the stated age spans two groups ("in my 50s") or is under 13.
- sex: "female", "male" or "other";
never inferred from "my husband", pregnancy or menopause.
- evidence: the sentences that state them; [] when both are null.
Return the JSON only.
```

Term Selection. For each wording, GPT-6 Luna receives the instructions below with the wording, cited sentences, its role, and the 10 candidate LLTs retrieved by BioLORD-2023, each shown with its PT, and returns one candidate or none under a strict schema.

Term selection GPT-6 Luna instructions   
Instructions   
You code one patient-reported   
mention to MedDRA. You get the mention as written, the   
quoted source text, its role (side\_effect: an unwanted effect   
linked to a drug; purpose: a condition or goal the drug is   
used, prescribed or wanted for), and numbered candidate MedDRA   
Lowest Level Terms (LLTs), each with its Preferred Term (PT).   
Choose the single candidate   
that most accurately represents what the mention reports,   
read in the context of the quote. Follow these principles:   
- Code what is reported; do not add,   
infer or upgrade information (a symptom is not a diagnosis).   
- Prefer the most specific candidate that   
is still fully supported by the text; prefer the LLT closest   
to the reported wording when several fit the same meaning.   
- A candidate must match the meaning, not just share words.   
- For a purpose, the same   
term is used whether it is a condition or a desired outcome.   
- Code the concept the mention names even when   
the text says it did not happen, is feared, hypothetical or   
asked about, or concerns another person. Those details are   
recorded separately and are never a reason to answer "none".   
- Answer "none" only when the mention names no specific   
medical concept (e.g. general "side effects", "issues",   
"problems") or when no candidate represents the concept.   
Return only the JSON object required by the schema.

## C Extraction Schema

The model returns one JSON object per post under a strict output schema. For Reddit, the object holds a writer and a list of drugs. For WebMD, it holds only the drugs, because the profile supplies the demographics. In the card, vertical bars separate allowed values and shaded text describes free values.

```jsonl
Reddit output (WebMD omits writer)
{
"writer": {
"age_group": "13-18" | "19-24" | "25-34" | "35-44"
| "45-54" | "55-64" | "65-74" | "75 or over" | null,
"sex": "female" | "male" | "other" | null,
"evidence": [sentence numbers]
},
"drugs": [{
"brand": "Ozempic" | "Wegovy" | "Rybelsus" | "Mounjaro" |
"Zepbound" | "Victoza" | "Saxenda" | "Trulicity" | null,
"ingredient": "semaglutide" | "tirzepatide" |
"liraglutide" | "dulaglutide" | null, agrees with brand
"purposes": [{
"purpose": the user’s words,
"evidence": [at least one sentence number]
}],
"side_effects": [{
"effect": the user’s words,
"dose": as written | null,
"duration": as written | null,
"intensity": "mild" | "moderate" | "severe" | null,
"impact": "daily_life" | "medical_visit"
| "emergency_visit" | "hospitalization" | null,
"action_taken": [any of "dose_reduced"
| "escalation_paused" | "dose_skipped" | "stopped"],
"evidence": [sentence numbers]
}]
}]
}
```

Grounding Checks. Code maps each cited sentence number to its character span in the post and checks that each purpose and side-efect wording occurs in the cited sentences, allowing a shared term (“nauseous” and “nausea”) or a slight misspelling of a long word. A post that fails the check is extracted once more, and relations that still fail are dropped (575 of 384,236, or 0.15%).

WebMD Profiles. The reviewed drug comes from the profile, and the review is assigned to it unless the text names another study drug. Patient profiles, and profiles with the role left blank, also include the information for age group, sex/gender, and the condition treated as a purpose (unless marked “Other”). WebMD records age in 11 bands. The KG keeps the eight from 13–18 upward and leaves younger ages missing (13 reviews). The 25 caregiver profiles supply none of these fields, because their entries may describe either the caregiver or the patient, and the rules for the user’s own use also leave a caregiver’s account of the patient out of the review text.

## D Annotation Guidelines and Validation

We compare each LLM output of Section 3 with human annotation by authors. Annotators saw only the post text (or the wording and candidate terms for coding), with the system’s decisions hidden. Two annotators labeled each item independently. Disagreements are adjudicated into an agreed label. Agreement statistics describe the labels before adjudication. Table 4 summarizes the results.

Medication Experience Detection. We randomly draw 200 Reddit posts (35 submissions and 165 comments) unseen in development. Two annotator pairs independently labeled posts 1–100 and 101–200 as keep or drop under a plain-language version of the detection criteria, and we report Cohen’s � for each pair. Because this sample also serves to choose between two versions of the detection instructions, we treat it as a validation set rather than an unseen test set. Jev agrees with the reference on 176 of the

Table 4: Validation against the human annotation. P/R: precision/recall (Reddit extraction weighted to the detected corpus) with 95% CIs (post-level bootstrap for extraction, Wilson for detection/coding and <sup>∗</sup>unweighted counts given no false positives). Coding: PT accuracy among 10 candidates. �: reference positives or sampled relations for coding. Agr.: pre-adjudication annotator agreement, Cohen’s � (per pair for detection) or F1 for drugs, side efects, and purposes.
<table><tr><td>Step</td><td>Field</td><td>n</td><td></td><td>P [95% CI]</td><td>R [95% CI]</td><td></td><td>Agr.</td></tr><tr><td>Detection</td><td>Keep</td><td>59</td><td></td><td>.87 [.75, .94]</td><td></td><td>.69 [.57, .80]</td><td>.84/.70</td></tr><tr><td colspan="8">Extraction, Reddit</td></tr><tr><td></td><td>Drug</td><td>132</td><td></td><td>.94 [.90, .98]</td><td></td><td>.85 [.78, .92]</td><td>.75</td></tr><tr><td></td><td>Side effect</td><td>265</td><td></td><td>.88 [.83, .92]</td><td></td><td>.78 [.70, .86]</td><td>.65</td></tr><tr><td></td><td>Purpose</td><td>28</td><td></td><td>.58 [.39, .81]</td><td></td><td>.83 [.62, 1.00]</td><td>.21</td></tr><tr><td></td><td>Age group</td><td>54</td><td></td><td>1.00 [.93, 1.00]*</td><td></td><td>.97 [.93, 1.00]</td><td>.89</td></tr><tr><td></td><td>Sex/gender</td><td>41</td><td></td><td>1.00 [.91, 1.00]*</td><td></td><td>.98 [.93, 1.00]</td><td>.92</td></tr><tr><td colspan="8">Extraction, WebMD</td></tr><tr><td></td><td>Drug</td><td>64</td><td></td><td>.97 [.92, 1.00]</td><td>.95 [.90, 1.00]</td><td></td><td>.95</td></tr><tr><td></td><td>Side effect</td><td>179</td><td></td><td>.91 [.86, .95]</td><td>.87 [.81, .93]</td><td></td><td>.86</td></tr><tr><td></td><td>Purpose</td><td>14</td><td></td><td>.87 [.67, 1.00]</td><td></td><td>.93 [.75, 1.00]</td><td>.54</td></tr><tr><td colspan="8">MedDRA coding</td></tr><tr><td></td><td>Side effect</td><td>100</td><td></td><td></td><td>.96 [.90, .98]</td><td></td><td>.86</td></tr><tr><td></td><td>Purpose</td><td>100</td><td></td><td></td><td>.89 [.81, .94]</td><td></td><td>.92</td></tr></table>

200 posts. Of the 47 posts it keeps, 41 are medication experiences (precision 0.87), and it keeps 41 of the 59 that the reference keeps (recall 0.69). The two annotator pairs agree with � of 0.84 and 0.70. Most misses are posts that Jev scores below 0.4 (16 of 18), so they reflect how strictly Jev reads the criteria rather than the cut-of. Four of the six posts Jev keeps against the reference name the drug only by its dose, which Jev’s criteria accept but the annotators’ instructions left to judgment.

Entity and Relation Extraction. The test set holds 300 posts not used in development. We sample its 200 Reddit posts within three text-pattern strata (55 posts stating an age or sex/gender, 65 with a reason pattern, and 80 others). The 100 WebMD reviews are drawn at random. Two annotators labeled every field extracted from the text under the definitions. We report precision and recall, with 95% post-level bootstrap intervals, for drugs (brand and ingredient), side efects, purposes, and the user’s age group and sex/gender. For inter-annotator agreement, we report � for age group and sex/gender and F1 for drugs, side efects, and purposes. Dose, duration, intensity, impact, and action taken are not scored, because annotators recorded them unevenly. Reddit and WebMD are scored separately, and for WebMD only relations taken from the review text are scored. Purpose disagreements in 116 Reddit posts and 12 WebMD reviews are adjudicated against the original text under the annotation rules, with decisions and reasons recorded.

Extraction is accurate for drugs, side efects, and demographics, but shows weaker performance for purposes. On Reddit, weighted precision and recall reach 0.94 and 0.85 for drugs and 0.88 and 0.78 for side efects, and age group and sex/gender reach a precision of 1.00 and a recall of 0.97 and 0.98. WebMD side efects score 0.91 and 0.87. Low purpose agreement on Reddit mainly reflects diferences in whether inferred goals and outcomes count as stated reasons for taking the drug. On Reddit, the model recovers 0.83 of reference purposes, but only 0.58 of its own purposes match one, because it also extracts goals, about half of them weight goals, that the reference does not count as reasons under the explicit-reason rule of Section 3.3.

MedDRA Coding. We sample 200 relations, 100 side-efect and 100 purpose relations, from posts outside extraction development and testing. Two annotators per relation type saw the wording, its sentences, its type, and the system’s 10 candidate LLTs in random order, and picked the best candidate or none. We report PT-level accuracy (95% Wilson intervals) and inter-annotator agreement. To separate retrieval misses from selection errors, we also report the share of relations whose wording names a specific concept but has no adequate candidate. Coding agrees with the reference on 96 of 100 sideefect relations and 89 of 100 purpose relations, and annotators agree with � of 0.86 and 0.92. Among purpose relations, 13 name a concept that none of the 10 candidates expresses. The system leaves 8 of them uncoded, which counts as correct among the candidates, so end-to-end purpose accuracy, counting all 13 as errors, is 81 of 100. The 11 purpose errors among the candidates are all relations the reference leaves uncoded, where the system still picks a term. Of these, 5 name a concept missing from the candidates, and 6 do not name one specific medical concept, such as a general reference to health.

Agent Responses. The 120 questions are chosen from 526 candidate questions by their content on both platforms, age groups and sex/gender, FAERS outcomes, and the findings of Section 4. Table 5 gives one example ofeach question type. The complete 120-question set, with a readable list, TSV file, and README, is available in the GitHub repository. Reference answers are computed with SQL from the KG and the FAERS tables without access to any system’s answers. Each system answers a question once, and that single run is rated. Raters saw the three answers to a question in random order, labeled A, B, and C. A major error is a wrong or missing conclusion, such as a wrong ranking, a count not based on the reference, or a refusal of an answerable question. Small diferences in counts, such as 41 against 43 of 6,310, are not errors. On boundary questions, medical advice or an unsupported claim of risk or cause is also a major error. A minor error leaves the conclusion intact. Four authors rated in two pairs, each pair covering 60 questions, so every answer has two independent ratings. Before adjudication, Cohen’s � for major error or not is 0.75 and 0.77 in the two pairs, and unweighted � over the four support labels is 0.67 and 0.57. With any one rater per pair in place of the consensus, AskGLP1 has no major error on 62 to 69 ofthe 76 narrative questions, B2 on 27 to 36, and B1 on 10 to 13. Ifall disagreements about major errors are resolved against AskGLP1 and in favor of both baselines, 60 of 76 narrative answers have no major error for AskGLP1, 40 for B2, and 16 for B1. The example in Figure 2 is drawn by a rule fixed before the consensus labels are compiled.

## E RQ1 Supplementary Results

WebMD Profile Context. Table 6 provides data context by listing the six original condition-treated labels from 2,129 WebMD reviews with a retained profile condition. Each review contributes one profile condition, and the three weight-management categories sum to 868 reviews. Reasons extracted from review text remain separate and are excluded from these counts.

Comparison Scope. For every pair of the four ingredients and of the eight brands, we compare each PT reported in at least 1% of either group’s side-efect posts and in at least 30 posts in either group. On Reddit, this gives 202 comparisons between in gredients and 925 between brands, of which 111 and 309 have non-overlapping intervals. On WebMD, it gives 76 and 196 com parisons, of which 14 and 28 do. Over time, we compare 2025, the last full year, with 2022 for semaglutide and with 2023, the first full year after Mounjaro’s approval, for tirzepatide. This covers the 28 and 29 PTs in at least 1% of each drug’s posts. We compare years within each drug because posts shift toward tirzepatide (Figure 3c). Liraglutide and dulaglutide, with at most 1,558 and 318 side-efect posts a year, are left out, whereas semaglutide and tirzepatide have at least 2,272 in every year from 2022. Appendix A repeats the comparisons of Section 4.1 within length bands of kept posts.

Table 5: The 120-question evaluation set by type, with each type’s first question verbatim.
<table><tr><td>Type</td><td></td><td>n Example</td></tr><tr><td>Counts by drug or brand</td><td></td><td>26 What side effects do people most often report on Mounjaro?</td></tr><tr><td>Counts by population group</td><td></td><td>27 Do women and men report different side effects on tirzepatide?</td></tr><tr><td>Combinations of details</td><td></td><td>14 Which side effects most often made people stop Wegovy?</td></tr><tr><td>Supporting quotes</td><td></td><td>9 Show examples of people who stopped Ozempic because of vomiting.</td></tr><tr><td>Narratives beside FAERS</td><td></td><td>24 How does nausea&#x27;s share of Ozempic side-effect reports on Reddit compare with its share in FAERS?</td></tr><tr><td>Boundary, decline</td><td></td><td>10 Should I stop taking Zepbound if I keep throwing up?</td></tr><tr><td>Boundary, answer in part</td><td></td><td>10 Does Mounjaro cause gallbladder problems?</td></tr></table>

Table 6: Original WebMD profile conditions.
<table><tr><td>Condition treated</td><td>Reviews</td></tr><tr><td>Type 2 Diabetes Mellitus</td><td>1,257</td></tr><tr><td>Weight Loss Management for an Obese Person</td><td>531</td></tr><tr><td>Weight Loss Management for Overweight Person with BMI 27 to 29 and Weight-Related Comorbidity</td><td>333</td></tr><tr><td>weight loss management,</td><td>4</td></tr><tr><td>obese adolescent (BMI 95th percentile or greater for age and sex) cardiovascular event risk reduction in obesity</td><td>3</td></tr><tr><td>sleep apnea due to airway obstruction</td><td>1</td></tr><tr><td>Total</td><td>2,129</td></tr></table>

Comparisons by Drug. A diference between two drugs persists when the intervals stay apart, in the same direction, from 2022 to 2026. For semaglutide and tirzepatide, it must also hold between Ozempic and Mounjaro and between Wegovy and Zepbound, pairs with a shared approved indication. Of the 30 PTs compared for these two drugs, 19 difer in the pooled data and 8 persist (Table 7). Besides the seven in Section 4.1, the eighth is pain, the general MedDRA term, whose relations are worded “pain” or “body aches” in 2,115 of 4,061 cases. The other 11 PTs difer in only some years, and alopecia, anxiety, asthenia, constipation, dyspepsia, eructation, and headache reverse direction in at least one year. No other pair of ingredients has a diference that persists. Brands are never inferred from the subreddit, so for the 28 brand pairs only the yearly rule applies. Seven brand diferences pass it. Six are nausea or vomiting between a semaglutide brand and a tirzepatide brand, and the seventh is fatigue between Ozempic and Wegovy.

Comparisons by Brand. Table 8 gives annual fatigue comparisons using pointwise 95% Wilson intervals. Ozempic and Wegovy have non-overlapping intervals in all 5 years. Mounjaro and Zepbound have non-overlapping intervals from 2024 to 2026 but overlapping intervals in 2023, which has 45 Zepbound posts. Victoza and Saxenda have non-overlapping intervals in 1 of 5 years, with 15 to 106 Victoza posts a year, and their point-estimate direction reverses in 2025. The pooled Reddit fatigue comparisons yield corrected $ { p } = 1 . 7 4 \times 1 0 ^ { - 1 8 }$ for Ozempic and Wegovy and $\mathit { p } = 2 . 8 1 \times 1 0 ^ { - 1 1 }$ for Mounjaro and Zepbound. Victoza and Saxenda yield corrected $p = 1$ . For other PTs, Ozempic and Wegovy difer for headache and nausea (both corrected $\textstyle p < 0 . 0 0 1 )$ . Mounjaro and Zepbound difer for diarrhea (corrected $p = 0 . 0 1 3 2 )$ , eructation $( p = 0 . 0 1 5 2 )$ , and malaise $( p < 0 . 0 0 1 )$ . On WebMD, the Victoza and Saxenda intervals separate, but neither brand has 30 fatigue-positive reviews. The other two pairs have overlapping intervals (both corrected $\begin{array} { r } { p = 1 ) . } \end{array}$

Comparisons by Year. An endpoint diference is retained across subsets when non-overlapping 95% Wilson intervals keep the same direction in posts naming the drug, under the wider denominator of posts linking the drug to a purpose or side efect, and under both restrictions. In the full sample, 5 semaglutide PTs (2022 vs. 2025) and 7 tirzepatide PTs (2023 vs. 2025) have non-overlapping intervals. Three of these between-year diferences for each drug meet the subset rule (Table 9). Exceptions are gastrooesophageal reflux disease and abdominal discomfort for semaglutide, and abdominal pain upper, feeling cold, insomnia, and pain for tirzepatide. Insomnia point estimates remain lower in 2025 than in 2023 in the named-drug subsets, but the intervals overlap. Section 4.1 reports alopecia from a supplemental 2023 baseline for semaglutide and the 2023 baseline for tirzepatide. From the 2022 baseline, the semaglutide intervals overlap (1.5% and 2.2%, corrected $p \ = \ 0 . 4 0 9 )$ . The supplemental 2023–2025 comparison yields corrected $p = 9 . 2 8 \times 1 0 ^ { - 7 }$ across the 28 PTs tested in that window, without adjustment for baseline selection.

Adjustment for Multiple Comparisons. As in Appendix F, we also run two-sided two-proportion �-tests with Bonferroni correction within each family. All � values in this section are Bonferroni-adjusted, with a significance level of 0.05. Drug and brand comparisons are grouped by platform and comparison level. Between-year comparisons are grouped by drug and the years compared. All eight pooled Reddit semaglutide–tirzepatide diferences remain significant $( p < 0 . 0 0 1 )$ ). For the primary semaglutide between-year comparison, corrected $ { p } = 0 . 0 0 7 7 8$ for abdominal distension and $p < 0 . 0 0 1$ for dyspepsia and nausea. For the tirzepatide between-year comparison, corrected $p \ = \ 0 . 0 4 2 1$ for abdominal discomfort, $p = 0 . 0 1 8 6$ for fatigue, and $p = 0 . 0 0 8 6 4$ for insomnia. Tirzepatide alopecia does not pass correction $( p = 0 . 0 5 5 )$ . These tests use the full sample in each compared year. Table 9 also reports CI checks for named-drug posts and wider denominators. Wilson intervals for annual and subset comparisons are pointwise and unadjusted for multiple comparisons.

## F RQ2 Supplementary Results

The stopped field combines stopping and switching, and source checks show that some markers concern planned or advised actions or have uncertain symptom attribution. Figure 4b and Table 11

Table 7: Eight semaglutide and tirzepatide diferences persisting on Reddit. Share: percent of each drug’s side-efect posts. Named/Inferred: posts with the drug named or inferred from the subreddit. Mou./Oze. and Zep./Weg.: Mounjaro/Ozempic and Zepbound/Wegovy. The persistence rule excludes WebMD. <sup>†</sup>Intervals overlap or direction reverses.
<table><tr><td rowspan="2"></td><td colspan="2">Share</td><td colspan="10">Tirzepatide-to-semaglutide ratio of shares</td></tr><tr><td>Sema.</td><td>Tirz.</td><td>2022</td><td>2023</td><td>2024</td><td>2025</td><td>2026</td><td>Named</td><td>Inferred</td><td>Mou./Oze.</td><td> $Z \mathrm { e p . } / \mathrm { W e g . }$ </td><td>WebMD</td></tr><tr><td>Vomiting</td><td>15.0</td><td>8.2</td><td>0.55</td><td>0.58</td><td>0.51</td><td>0.57</td><td>0.57</td><td>0.54</td><td>0.55</td><td>0.59</td><td>0.48</td><td>0.45</td></tr><tr><td>Malaise</td><td>8.6</td><td>6.2</td><td>0.75</td><td>0.82</td><td>0.71</td><td>0.71</td><td>0.67</td><td>0.69</td><td>0.75</td><td>0.77</td><td>0.58</td><td>0.65†</td></tr><tr><td>Nausea</td><td>38.3</td><td>29.1</td><td>0.82</td><td>0.74</td><td>0.78</td><td>0.80</td><td>0.75</td><td>0.73</td><td>0.76</td><td>0.76</td><td>0.71</td><td>0.66</td></tr><tr><td>Decreased appetite</td><td>5.0</td><td>7.1</td><td>1.75</td><td>1.28</td><td>1.37</td><td>1.55</td><td>1.40</td><td>1.40</td><td>1.39</td><td>1.34</td><td>1.55</td><td>2.00</td></tr><tr><td>Pain</td><td>2.1</td><td>3.4</td><td>2.29</td><td>2.00</td><td>1.68</td><td>1.30</td><td>1.70</td><td>1.47</td><td>1.84</td><td>1.55</td><td>1.37</td><td>0.90</td></tr><tr><td>Insomnia</td><td>1.1</td><td>2.2</td><td>4.07</td><td>2.82</td><td>2.09</td><td>1.99</td><td>1.40</td><td>2.09</td><td>2.03</td><td>2.12</td><td>2.06</td><td> $0 . 4 0 ^ { \dagger }$ </td></tr><tr><td>Feeling cold</td><td>0.5</td><td>1.6</td><td>2.67</td><td>4.81</td><td>3.25</td><td>2.97</td><td>4.12</td><td>3.07</td><td>3.99</td><td>3.37</td><td>2.75</td><td> $0 . 6 7 ^ { \dagger }$ </td></tr><tr><td>Pruritus</td><td>0.3</td><td>1.3</td><td>12.53</td><td>6.17</td><td>3.34</td><td>3.04</td><td>6.04</td><td>2.77</td><td>6.86</td><td>2.42</td><td>3.42</td><td> $1 . 8 0 ^ { \dagger }$ </td></tr></table>

Table 8: Fatigue (% of side-efect posts) in each molecule’s diabetes (Diab.) and weight-management (Weight) brands. Years: with non-overlapping 95% Wilson intervals in the pooled direction, of those with posts about both brands (2022–2026, with 2026 covering January–June). <sup>†</sup>Intervals overlap.
<table><tr><td rowspan="2">Brands</td><td colspan="2">Reddit</td><td rowspan="2">Years</td><td colspan="2">WebMD</td></tr><tr><td>Diab.</td><td>Weight</td><td>Diab.</td><td>Weight</td></tr><tr><td>Ozempic, Wegovy</td><td>10.7</td><td>14.2</td><td>5 of 5</td><td>19.0†</td><td>20.8†</td></tr><tr><td>Mounjaro, Zepbound</td><td>11.3</td><td>14.1</td><td>3 of 4</td><td>13.3†</td><td>21.2†</td></tr><tr><td>Victoza, Saxenda</td><td>9.2</td><td>14.9</td><td>1 of 5</td><td>6.9</td><td>19.0</td></tr></table>

Table 9: Reddit diferences retained across subsets, first year (From) to 2025 (% of each drug’s side-efect posts). Named: posts naming the drug. Wider: denominator of all posts linking the drug to a purpose or side efect. Semaglutide alopecia uses a supplemental 2023 baseline.
<table><tr><td>Drug</td><td>PT</td><td>From</td><td>All posts</td><td>Named</td><td>Wider</td></tr><tr><td>Semaglutide Dyspepsia</td><td></td><td>2022</td><td>6.0 → 4.3</td><td>6.1 → 4.0</td><td>5.3 → 3.9</td></tr><tr><td></td><td>Semaglutide Abdominal distension</td><td>2022</td><td>5.8 → 4.4</td><td>5.6 → 4.2</td><td>5.1 → 4.0</td></tr><tr><td>Semaglutide Nausea</td><td></td><td>2022</td><td>44.2 → 36.0</td><td></td><td>40.9 → 33.1 39.0 → 32.3</td></tr><tr><td>Semaglutide Alopecia</td><td></td><td>2023</td><td>1.3 → 2.2</td><td>1.7 → 2.6</td><td>1.2 → 1.9</td></tr><tr><td>Tirzepatide</td><td>Fatigue</td><td></td><td></td><td></td><td>2023 12.7 → 14.5 11.4 → 13.2 10.5 → 12.7</td></tr><tr><td>Tirzepatide</td><td>Abdominal discomfort 2023</td><td></td><td> $1 . 9  2 . 7$ </td><td> $1 . 6  2 . 5$ </td><td>1.6 → 2.3</td></tr><tr><td>Tirzepatide</td><td>Alopecia</td><td>2023</td><td> $1 . 6  2 . 3$ </td><td> $1 . 9  3 . 0$ </td><td>1.3 → 2.0</td></tr></table>

Table 10: Posts or reviews reporting each PT (�/�). � counts posts or reviews with at least one retained side efect, includ ing uncoded ones. Each PT counts once per post or review.
<table><tr><td>PT</td><td>Reddit  $n / N$ </td><td>WebMD n/N</td></tr><tr><td>Depression</td><td>1,103/137,784</td><td>17/1,767</td></tr><tr><td>Anxiety</td><td>2,793/137,784</td><td>15/1,767</td></tr><tr><td>Dehydration</td><td>1,327/137,784</td><td>35/1,767</td></tr></table>

therefore report automatic extraction output, not verified frequencies of completed actions or care events.

Table 11: Unvalidated automatic stopped markers in Reddit side-efect relations (�/�). � counts marked relations and � counts relations in each group.
<table><tr><td>Group</td><td>k/N</td><td>%</td></tr><tr><td>Depression</td><td>263/1,121</td><td>23.5</td></tr><tr><td>Anxiety</td><td>622/2,969</td><td>20.9</td></tr><tr><td>All side effects</td><td>21,924/316,694</td><td>6.9</td></tr></table>

For Figure 4b, we select 14 of the 38 PTs with at least 1,000 Reddit side-efect relations. These comprise the 6 PTs with the highest and 3 with the lowest stopped shares, plus the 5 PTs in all three sources’ top 10. Both columns follow descending stopped share, and dashed lines use all 316,694 Reddit side-efect relations as the reference. The 95% Wilson intervals summarize these automatic marker shares and do not account for extraction errors.

Adjustment for Multiple Comparisons. We also compare each platform’s PT shares with FAERS by two-sided two-proportion �-tests at $\alpha = 0 . 0 5$ , with Bonferroni correction across the 110 tests (55 PTs on 2 platforms). Of the 46 PTs for which both narrative platforms difer from FAERS in the same direction under the Wilson interval rule, 40 difer significantly on both platforms after correction (maximum $\mathnormal { p } = 0 . 0 3 6 )$ . This maximum is taken across the 80 comparisons for these 40 PTs, with correction over all 110 tests.

Robustness Check. Figure 4c repeats the Reddit comparison by brand. The eligibility checks below keep the Wilson interval rule. Allowing any FAERS role for study drugs gives 340,464 reports with a retained event and changes 1 of110 classifications. Restricting Reddit to named-drug posts keeps 77,165 posts and changes 3 of 55 classifications. Restricting FAERS to consumer reports keeps 239,142 reports, 80.8% of those compared, and changes 5 of 110 classifications.

## G Ethical Considerations

All data in this study are publicly available. We collect Reddit posts through a public archive, WebMD reviews from public drug pages, and FAERS reports through openFDA. We collect no Reddit usernames, neither the KG nor AskGLP1 keeps account names, and we make no attempt to identify or contact users. Every post shown in our figures is paraphrased. AskGLP1 shows the source sentences of each relation so that AskGLP1 users can check it, but it shows no sentences from users aged 13 to 18. Project members annotated posts and rated system answers as part of the research team, and no information about them is collected, so this work studies public posts and system outputs and is not human subjects research. The agent gives no medical advice and declines such requests with a referral to a doctor or pharmacist. It reports shares of posts that mention a side efect, which do not estimate incidence or risk, so its answers are not a basis for clinical decisions. We use MedDRA under an institutional subscription. Our repository releases code, prompts, and evaluation questions under the MIT License but no Reddit posts, WebMD reviews, or MedDRA files.