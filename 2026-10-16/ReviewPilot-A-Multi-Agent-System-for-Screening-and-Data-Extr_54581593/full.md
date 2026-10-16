# ReviewPilot: A Multi-Agent System for Screening and Data Extraction in Systematic Reviews

Zhaoqian Xue<sup>1†</sup>, Shaoze Zhou<sup>2</sup>, Xuexin Li<sup>3</sup>, Junting Mu<sup>4</sup>, Haozheng Liu<sup>5</sup>, Zhongyin Wang<sup>6</sup>, Yichun He<sup>7</sup>, Renkai Ma<sup>8</sup>, Hao Zhou<sup>9</sup>, Sarah Bratt<sup>1</sup>, Siyuan Ma<sup>10</sup>, Chen Chen<sup>2</sup>, Lingyao Li<sup>1†</sup>

<sup>1</sup>University of Arizona, <sup>2</sup>Florida International University,

<sup>3</sup>Vanderbilt University, <sup>4</sup>University of California, Irvine, <sup>5</sup>New York University,

<sup>6</sup>University of Illinois Urbana-Champaign, <sup>7</sup>Broad Institute of MIT and Harvard, <sup>8</sup>University of Cincinnati, <sup>9</sup>University of South Florida, <sup>10</sup>Vanderbilt University Medical Center

## Abstract

Systematic literature reviews follow the PRISMA process of screening scholarly works and extracting evidence from included studies. Deep research tools based on large language models (LLMs) can search and summarize the literature but do not record which studies are excluded or where extracted values come from. We develop a conversational multiagent system called ReviewPilot that automates the PRISMA workflow from paper screening to information extraction. Researchers can tailor criteria and schemas through dialogue, and a project memory keeps their configurations and revisions. Each screening decision comes with its reason and each extracted value with its source. On epidemiological expert reviews, ReviewPilot outperforms a single-prompt LLM and AgentSLR across three LLMs, with screening macro F1 up to 0.776 and extraction F1 up to 0.788. Case studies in biomedicine and human-computer interaction show close agreement with human annotators, suggesting that ReviewPilot can transfer beyond epidemiology. Code is available at this link.

## 1 Introduction

Systematic literature reviews depend on a sequence of judgments, including eligibility check, full-text retrieval, and information extraction (Higgins et al., 2023; Page et al., 2021). Such a process if laborintensive. Searches often result in high recall, so reviewers need to read thousands of titles and abstracts to find a few eligible studies (van de Schoot et al., 2021). A review can take over a year on average from registration to publication (Borah et al., 2017). Automation can reduce this workload, but researchers still need to inspect decisions and document the process based on the PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) 2020 process (Page et al., 2021).

Large language models (LLMs) are now widely used to search and summarize literature. Deep research tools such as ChatGPT (OpenAI, 2025), Gemini (Citron, 2024), and Perplexity Research (Perplexity, 2025) can generate a synthesized report. These tools help researchers explore a topic but do not follow the PRISMA process. They do not show which papers are screened or excluded and why, or where each extracted value comes from. Other review tools only limit these decisions to a single stage, such as ASReview (van de Schoot et al., 2021) and LGAR (Jaumann et al., 2025) for screening, and RobotReviewer (Marshall et al., 2017) for extraction, or an agent-based system for a few stages like AgentSLR (Padarha et al., 2026).

However, designing an AI-assisted literature review workflow involves two key considerations. First, screening criteria and field definitions need operational instructions. Domain terms and borderline cases need to be iteratively interpreted consistently for screening and coding. Second, outputs must reflect the current configuration. Researchers refine criteria and fields as a review progresses (Higgins et al., 2023); confirmed settings and corrections must therefore be retained, and affected downstream outputs identified. We implement these requirements within a workflow that records the counts and inclusion-exclusion criteria needed for PRISMA-guided workflow.

We present ReviewPilot, a conversational multiagent system that addresses these challenges from search through information extraction (Figure 1). Researchers define eligibility criteria and an extraction schema through natural dialogue with the system, and a Prompt Agent turns them into reviewspecific guidance. Agents screen records and extract information, delivering screening decisions with their reasons and extracted values linked to source passages in the papers. The result is a stepby-step record of how papers are filtered, with the records kept and excluded at each stage, and a table of the information extracted from each retrieved paper. Our contributions are as follows.

![](images/1300e19ec64bb34957a2a5b374e74114c48e7e1abd3a98929225f7d372ff9d9b.jpg)  
Figure 1: Manual and ReviewPilot-assisted systematic review workflows, aligned with PRISMA 2020 stages (Page et al., 2021). ReviewPilot coordinates search, screening, and extraction while supporting researcher inspection and revision. Project memory retains configurations, outputs, and corrections.

• PRISMA-guided workflow. ReviewPilot coordinates search, screening, retrieval, and extraction while recording counts, exclusion reasons, and available source for inspection (Section 3).

• Review-specific configuration and revision. Researchers can set criteria and schemas through dialogue. Then, agents draft screening guidance and coding rules. Researchers can edit these instructions and confirm. Project memory design further retains these configurations and corrections (Sections 3.2 and 3.6).

• System-level evaluation. On PERG, Review-Pilot achieves higher screening macro F1 and flextraction F1 than a single-prompt LLM and AgentSLR under the reported settings (Tables 1 and 2). Two case studies document completed workflows and screening agreement with human annotators (Section 5).

## 2 Related Work

LLM multi-agent systems. Multi-agent frameworks divide tasks among LLM agents with distinct roles (Guo et al., 2024b). For instance, AutoGen supports configurable conversations that combine LLMs, tools, and human input (Wu et al., 2024). MetaGPT encodes standard operating procedures as prompt sequences to coordinate software development and check intermediate results (Hong et al., 2024). Our system, ReviewPilot, leverages the potential of multi-agent workflow to review stages.

Deep research and literature synthesis. Deep research tools in ChatGPT (OpenAI, 2025), Gemini (Citron, 2024), and Perplexity (Perplexity, 2025) iteratively search and analyze sources to produce reports. ChatGPT or Gemini can link citations and allows users to revise its research plan. For scientific literature, OpenScholar combines passage retrieval, cited generation, and iterative self-feedback (Asai et al., 2026), while AutoSurvey retrieves papers, constructs an outline, and delegates subsection drafting to specialized LLMs (Wang et al., 2024b). These systems can synthesize reports but do not document the paper filtering process based on PRISMA 2020 (Page et al., 2021).

Systematic review tools. Some tools support individual review stages. ASReview prioritizes records for human screening through active learning from relevance judgments (van de Schoot et al., 2021); LGAR ranks abstracts zero-shot using graded, criterion-based LLM scores and dense reranking (Jaumann et al., 2025). RobotReviewer extracts trial characteristics and supports risk-of-bias assessment with sentences linked to the source document (Marshall et al., 2017).

Integrated workflows extend assistance across stages. AgentSLR pairs an epidemiological literature review workflow with stage-specific evaluation and describes an interface for expert verification of extracted values (Padarha et al., 2026). Its published workflow uses predefined eligibility criteria and domain-specific extraction schemas. We compare against it in Section 4. Ren et al. (2026)

translate user criteria into screening rules and use multi-agent cross-review for clinical-trial screening. Their extraction workflow combines sourcederived table standardization, iterative correction, and targeted human review. The commercial Elicit workflow proposes editable criteria and extraction fields and lets users return to earlier steps (Byun, 2025); it also provides evidence quotes, exclusion reasons, and PRISMA 2020 flow diagrams (Vardhaan and Mohanty, 2026). These workflows already combine stage automation, customization, evidence access, and human oversight.

To address these gaps, we develop ReviewPilot, a PRISMA-guided multi-agent system that coordinates literature search, screening, full-text retrieval, and information extraction. Researchers can tailor eligibility criteria and extraction schemas through dialogue, while agents generate screening guidance and per-field coding rules. Project memory retains configurations and corrections to support revision across stages (Sections 3.2 and 3.6). We evaluate the complete system on PERG and examine its workflows and screening agreement with human annotators in two case studies.

## 3 ReviewPilot Development

ReviewPilot is a conversational multi-agent system that covers five stages of a systematic literature review: search, screening, full-text retrieval, information extraction, and semantic categorization (Figure 2). Researchers can configure the review through dialogue or direct editing, inspect each stage’s outputs, and can override screening decisions or correct extracted values.

## 3.1 Architecture and Orchestration

A Lead Agent coordinates four specialized agents and two tools through action contracts (Appendix A.1). The Search Condition Agent designs the search strategy, and the Prompt Agent writes review-specific instructions (Section 3.2). The Screening Agent deduplicates and assesses records, and the Extraction Agent extracts structured information. The Literature Search Tool executes queries, and the PDF Download Tool retrieves full texts. The Lead Agent categorizes values itself.

The Lead Agent starts a stage only when its prerequisites are met and the outputs it depends on are current. It reports completion only after checking the expected artifacts and their required fields. Agent skills are structured Markdown instructions specifying procedures for search strategy, screening, prompt design, extraction, and categorization. A fixed mapping appends the matching skill to the LLM system prompts for these tasks (Appendix G).

## 3.2 Review-Specific Prompt Design

The Prompt Agent writes the review-specific parts of the screening and extraction instructions, following the prompt-design skill. After the eligibility criteria are confirmed, it writes screening guidance $\mathcal { G }$ from the review objectives, criteria, search concepts, and up to 200 candidate records (titles and abstract openings). This guidance defines the review focus, domain terms, include and exclude categories, and tie-breaking rules. After screening, it drafts an extraction schema from the objectives, the requested fields, and the abstract openings of up to three included papers. Once the schema is confirmed, it writes the coding rules C from up to 25 included papers, using their abstract openings and full-text sentences selected for each field.

The guidance and the coding rules are each drafted and then revised in a second LLM pass against the skill’s checklist (Madaan et al., 2023). Reviewer decisions and reference extraction values are not among the generation inputs. Researchers can edit the guidance, schema, and rules and must confirm each before the stage that uses it runs (Appendices G.3, G.4.1, and G.5.1).

## 3.3 Search & Screening

Search. The Search Condition Agent identifies the concepts required for eligibility and proposes alternative terms for each concept. It builds a Boolean query by joining alternatives with OR and required concept groups with AND (Appendix A.2). Analytical dimensions that do not determine eligibility are omitted from the required groups. Researchers refine the concepts and query through dialogue or direct editing and review the sources, date range, and per-source limits. The Literature Search Tool then searches titles and abstracts in PubMed, arXiv, and OpenAlex. The saved query, settings, counts, and errors document the search as run.

Screening. After date filtering and deduplication (Appendix A.3), the model assesses each record $p _ { i }$ , its title and available abstract, against the confirmed inclusion criteria $\mathcal { T } ,$ exclusion criteria $\mathcal { E } ,$ and screening guidance $\mathcal { G }$

$$
\begin{array} { r } { ( r _ { i } , \ell _ { i } ) = f _ { \mathrm { L L M } } ( p _ { i } , \mathcal { T } , \mathcal { E } , \mathcal { G } ) , } \end{array}\tag{1}
$$

![](images/dfb96ecbac80e4b9b6c55174ca27138f78a2dc8b8334359d6d887136188a6f4f.jpg)  
Figure 2: ReviewPilot architecture and workflow. (1) The Lead Agent coordinates five review stages using project memory. Researchers configure stages and correct decisions or values. In the upper right, the Prompt Agent prepares screening guidance, an extraction schema, and coding rules without reviewer labels. (2) Screening returns a rationale and one of five likelihood levels. (3) Extraction returns values with verbatim excerpts. The label verified means the excerpt is found in the PDF text. Orange marks screening rationales and extraction evidence.

where $r _ { i }$ is a one-sentence rationale naming the deciding rule and $\ell _ { i }$ is a qualitative likelihood label. Its five levels are definitely exclude, probably exclude, uncertain, probably include, and definitely include. The screening skill instructs the model to assess the study’s own work against the review focus and criteria, not topics mentioned only as background (Appendix G.4). Among these five levels, a record is excluded only when $\ell _ { i }$ is probably exclude or definitely exclude. Records with the other labels are retained for full-text review. Records are also retained when the model call fails. The prompt requests the deciding rule and a verbatim title or abstract excerpt. An exclusion with an unmatched rule or unlocated excerpt is flagged for human review, but the record remains excluded.

## 3.4 Paper Retrieval & Information Extraction

Paper retrieval. The PDF Download Tool tries a cascade of publisher, open-access, and preprint sources for each retained record. It checks PDF validity and compares opening-page text with the title. These heuristics do not guarantee retrieval success (Appendix A.4), so it generates a report to record the success or failure retrieval for each paper, and researchers can add missing PDFs manually.

Information extraction. The confirmed schema $\boldsymbol { \mathcal { S } } ~ = ~ \{ ( f _ { j } , \tau _ { j } , \delta _ { j } ) \} _ { j = 1 } ^ { k }$ defines each field by its name, type, and description (Appendix A.5). The coding rules C specify how borderline values are coded. ReviewPilot places a full-text eligibility field first in the drafted schema, with options from the confirmed criteria. The model judges whether the paper remains eligible and names the criterion behind an exclusion. In a PDF assessment, an empty or unrecognized eligibility value does not exclude the paper. Excluded papers are omitted from categorization, and their exclusion reasons are recorded for PRISMA reporting.

For each available PDF, the Extraction Agent sends up to the first 400,000 characters of its text, with page markers, together with the schema and coding rules to the model (Appendix G.5). For each field, the prompt requests a value v<sub>j</sub>, verbatim excerpts, and page numbers. It requires an empty or null value when the supplied text neither states nor necessarily implies a value. Responses are checked for field presence, types, and value constraints. A failed check is recorded as an extraction failure for that paper. Passing these checks does not establish that the values are correct. ReviewPilot searches the normalized PDF text for the returned excerpts. A field with at least one located excerpt is marked verified, and the first located excerpt is stored with the page on which it was located. Otherwise, the field is unverified. Excerpt location establishes textual presence, not semantic support for the value or coverage of every value in a multi-value field.

For papers whose PDF was not retrieved, researchers can enable assessment from the abstract. When the selected model supports web search, it can seek web evidence for fields the abstract leaves open. Accepted values require located excerpts, and web pages must pass paper-identity checks. These outputs retain source labels and any source URLs without PDF page numbers and are reported separately from full-text assessments. Fields without evidence remain unavailable, and papers whose eligibility remains unavailable are not categorized.

## 3.5 Semantic Categorization

The Lead Agent categorizes extracted values only for papers retained after eligibility check. For a selected field, it proposes a set of categories with short descriptions (Appendices A.6 and G.6). The plan supports single-label or multiple-label assignment, and an Other category for values that do not fit. Researchers review and edit the plan before the Lead Agent assigns labels to each paper. Papers with no value for the field are marked unavailable.

## 3.6 Memory Design

Researchers work in a conversation panel and a stage-specific canvas. A chat request that changes criteria or the schema is saved as a draft, and confirmation is a canvas action (Appendices A.7 and H). Project memory stores the conversation history, current configurations, confirmed configuration snapshots, and each stage’s status and PRISMA counts.

In conversation, the Lead Agent receives its task instructions, the current request, the saved project state, and the conversation history. Its instructions give saved state precedence over superseded requests and distinguish drafts from confirmed configurations. Researchers reuse a previous project’s configuration by importing it as a draft that again requires confirmation. The system does not retrieve memories across projects automatically or learn preferences from corrections.

To override a screening decision, researchers select a confirmed criterion or guidance rule and give a reason. They also give reasons when correcting extracted values. Original decisions and values remain in review history. A change to a confirmed configuration or reviewed record marks affected downstream outputs as outdated until rerun.

## 4 Experiments & Results

We evaluate screening and structured information extraction separately against the expert annotations in PERG, a dataset released with AgentSLR (Padarha et al., 2026). Extraction uses the same expert-included papers for all methods, independently of each method’s screening predictions.

## 4.1 Experiment Setup

Data and reference annotations. PERG is released under the CC BY 4.0 license. Its annotations originate from systematic reviews conducted by the Pathogen Epidemiology Review Group at Imperial College London for WHO priority pathogens (Cuomo-Dannenburg et al., 2024; Nash et al., 2024; Doohan et al., 2024; Morgenstern et al., 2025; Mc-Cain et al., 2025; Naidoo et al., 2026). Screening uses 5,614 title–abstract records for Lassa, Nipah, SARS, and Zika, with 854 expert inclusions (15.2%). All methods apply the five inclusion and six exclusion criteria specified in AgentSLR’s PERG template (Appendix B). All records are evaluated without date filtering or deduplication.

Extraction uses 178 annotated papers (79 for Ebola, 6 for Lassa, 36 for SARS, and 57 for Zika). We evaluate five fields: model type, stochastic or deterministic status, code availability, transmission routes, and interventions (Table B.2). PERG annotates each transmission model in a paper. Reference values are aggregated to the paper level by taking the union across models. Code availability is true if the paper reports available code for any model.

Models and evaluation metrics. We use gpt-6- luna through the OpenAI API,<sup>1</sup> and DeepSeek V4.1 Flash and GLM-5.3-Flash through Together AI.<sup>2</sup> The single-prompt LLM uses one instruction containing the screening criteria, or the study objective and field definitions for extraction. It retains uncertain screening cases and requests field-level excerpts for extraction, without ReviewPilot’s skills, coding rules, or schema validation (Appendix C).

We rerun AgentSLR using its public screening implementation and transmission-model extraction pipeline. Its prompts, parser, and internal full-text screening are preserved, as are the tool and provenance loops of its extraction stage. It reads the same PDF text as the other methods. We adapt AgentSLR’s API client and replace its original high reasoning setting (Table D.2). Extraction and gpt-

6-luna screening use medium reasoning. Screening with the Together AI models uses provider defaults. Papers rejected by its internal screening remain in the evaluation with empty predictions.

ReviewPilot follows the screening and extraction procedures in Sections 3.3 and 3.4. It uses a fixed five-field extraction schema, with review-specific screening guidance and coding rules generated by the Prompt Agent. All methods receive the same papers and use the same reference annotations.

Prompt Agent uses the same model as its downstream stage. The agent receives the review objectives, criteria or field definitions, and samples of evaluation records or papers. Expert decisions and reference extraction values are excluded from the agent’s inputs. ReviewPilot’s screening procedure and prompt-design skill are refined using evaluation scores from these PERG pools. The pools are therefore not held out from method development.

We report one complete run per method and model. Decoding settings and output limits differ across methods and are recorded with each run (Appendix D). Transport errors and incomplete responses trigger retries. ReviewPilot also retries unparseable screening outputs. GLM-5.3-Flash exhausted its 65,536-token output limit on one paper. That paper is excluded from all three methods’ results on this model, leaving 177 papers. All table entries are single-run point estimates and do not characterize variation across repeated model runs.

Screening metrics. We pool the records and compute precision, recall, and F1 separately for the include and exclude classes. Each metric is then macro-averaged over the two classes. The review share, (TP + FP)/N, is the proportion of records retained for full-text review, where TP and FP denote retained expert-included and expert-excluded records, and N is the pool size. Macro recall averages recall for both classes and does not measure include-class recall alone.

Extraction metrics. Predicted and reference values are normalized with a fixed synonym mapping shared by all methods. At the paper–field level, matching values count as true positives, additional predicted values as false positives, and reference values absent from predictions as false negatives. Under AgentSLR’s empty-field convention, a field empty in both prediction and reference counts as one true positive. This convention applies to model type, stochastic or deterministic status, and code availability. For paper-level scoring, AgentSLR’s per-model predictions are aggregated identically to reference values. Its native model-matching evaluation is not used. Counts are pooled across papers and the five fields to compute precision, recall, and F1. AgentSLR’s additional fields are not scored.

Table 1: Screening results pooled over four PERG pathogens (5,614 records with 854 expert inclusions). Precision, recall, and F1 are macro-averaged over the include and exclude classes. Review share is the proportion of records retained for full-text review. Cost is USD per 1,000 records. Bold marks the best value within each model (highest for ↑, lowest for ↓).
<table><tr><td rowspan="17">Macro Macro Review Cost↓ Method</td><td colspan="3">Macro precision↑ recall↑ F1↑ share↓(USD)</td></tr><tr><td>gpt-6-luna</td><td></td><td></td></tr><tr><td>Single-prompt</td><td></td><td></td></tr><tr><td>LLM</td><td>0.685 0.828</td><td>0.696 0.354 0.13</td></tr><tr><td>AgentSLR</td><td>0.694</td><td>0.833 0.713 0.331 0.30</td></tr><tr><td>ReviewPilot</td><td>0.734</td><td>0.861 0.766 0.274 0.23</td></tr><tr><td>DeepSeek V4.1 Flash</td><td></td><td></td></tr><tr><td>Single-prompt</td><td></td><td></td></tr><tr><td>LLM</td><td>0.655 0.798</td><td>0.632 0.446 0.67</td></tr><tr><td>AgentSLR</td><td>0.698 0.842</td><td>0.717 0.334 1.19</td></tr><tr><td>ReviewPilot</td><td>0.743 0.861</td><td>0.776 0.260 1.06</td></tr><tr><td>GLM-5.3-Flash</td><td></td><td></td></tr><tr><td>Single-prompt</td><td></td><td></td></tr><tr><td>LLM</td><td>0.635 0.759 0.567</td><td>0.535 0.42</td></tr><tr><td>AgentSLR</td><td>0.667 0.816 0.654</td><td>0.424 1.22</td></tr><tr><td>ReviewPilot</td><td>0.726 0.864</td><td>0.757 0.293 0.68</td></tr></table>

The verified-correct rate is computed over paper– field pairs with a reference value. A pair counts as verified-correct when its complete normalized prediction matches the reference set and at least one returned excerpt is located in the paper. All methods use the same excerpt matcher. Excerpt location does not establish semantic support for the prediction (Gao et al., 2023) or coverage of every value in a multi-value field.

Cost estimation. We estimate API costs from returned token usage and the prices in Table D.1. Costs are reported in USD per 1,000 screened records or per extraction paper. Screening costs include ReviewPilot’s guidance generation, whereas extraction costs exclude coding-rule generation.

## 4.2 Screening Results

ReviewPilot has the highest screening macro precision, recall, and F1 on each model in Table 1. Its macro F1 exceeds AgentSLR’s by 5.3–10.3 points and the single-prompt LLM’s by 7.0–19.0 points. ReviewPilot retains 26–29% of records, versus 33– 42% for AgentSLR and 35–54% for the singleprompt LLM. The lower share describes the number of records sent to full-text review. It does not establish that fewer expert inclusions are missed or that human review time decreased. ReviewPilot’s reported API cost is lower than AgentSLR’s on each model, while the single-prompt LLM has the lowest reported API cost.

Table 2: Extraction results on PERG (178 papers, or 177 for GLM-5.3-Flash after excluding the same paper for all methods). Precision, recall, and F1 are computed from counts pooled across papers and the five fields. Verified-correct rate requires the complete normalized prediction to match the reference set and at least one excerpt to be located in the paper. Cost is USD per paper, excluding coding-rule generation. Bold marks the best value (highest for ↑, lowest for ↓).
<table><tr><td rowspan=1 colspan=1>Verified-correctCost↓Method      Precision↑ Recall↑ F1↑  rate↑ (USD)</td></tr><tr><td rowspan=1 colspan=1>gpt-6-luna</td></tr><tr><td rowspan=1 colspan=1>Single-promptLLM          0.714  0.8370.771 0.528 0.0025</td></tr><tr><td rowspan=1 colspan=1>AgentSLR     0.723  0.7820.7510.4490.0078</td></tr><tr><td rowspan=1 colspan=1>ReviewPilot   0.741   0.8410.7880.7130.0030</td></tr><tr><td rowspan=1 colspan=1>DeepSeek V4.1 Flash</td></tr><tr><td rowspan=1 colspan=1>Single-promptLLM          0.711   0.8320.767 0.561 0.0101</td></tr><tr><td rowspan=1 colspan=1>AgentSLR     0.672  0.8100.7350.4350.0306</td></tr><tr><td rowspan=1 colspan=1>ReviewPilot   0.723   0.8530.7820.7030.0106</td></tr><tr><td rowspan=1 colspan=1>GLM-5.3-Flash</td></tr><tr><td rowspan=1 colspan=1>Single-promptLLM          0.699  0.8340.761 0.5480.0081</td></tr><tr><td rowspan=1 colspan=1>AgentSLR     0.705  0.773 0.7370.4050.0209</td></tr><tr><td rowspan=1 colspan=1>ReviewPilot   0.718   0.8470.7770.7140.0139</td></tr></table>

## 4.3 Extraction Results

ReviewPilot has the highest pooled precision, recall, and F1 on each model in Table 2. Its F1 exceeds AgentSLR’s by 3.7–4.7 points and the single-prompt LLM’s by 1.5–1.7 points. Review-Pilot’s verified-correct rate reaches 0.703–0.714, compared with 0.405–0.449 for AgentSLR and 0.528–0.561 for the single-prompt LLM. More paper–field pairs therefore combine a complete normalized match to the reference set with at least one excerpt located in the paper. Semantic verification is still required to assess whether the excerpts support the predicted values. ReviewPilot’s reported extraction cost is lower than AgentSLR’s and higher than the single-prompt LLM’s on each model. Appendix E gives the screening results by pathogen and extraction results by field.

## 5 Case Studies

We apply ReviewPilot to reviews of LLM applications in biomedicine and human–computer interaction (HCI).

## 5.1 Complete Review Workflows

Both projects used gpt-6-luna and searched PubMed, arXiv, and OpenAlex, taking up to 1,000 results per source. Searches covered publications dated 1 January 2023 to 1 October 2026. Figure 3 and Figure F.1 in Appendix F trace the workflows, outputs, and actions.

HCI. The 2,241 records yielded 752 categorized studies (Figure 3); 153 reports were not retrieved, and full-text assessment excluded 75 of 827 reports. The researcher reinstated three excluded records, including a system evaluation of an interactive tool. The examples contrast this reversal with a rapidreview exclusion and pair GPT-4 and 43 participants with source excerpts. Before categorization, the researcher regrouped the proposed application categories. Studies could receive multiple labels.

Biomedicine. The 3,000 records yielded 1,341 categorized studies (Figure F.1). Screening retained 1,705 records, including four exam evaluations reinstated by the researcher. Of these, 336 reports were not retrieved and seven not extracted. Full-text assessment excluded 21 of 1,362 reports. The examples show a survey exclusion, a reinstated exam evaluation, and extraction of GPT-4o and expert evaluation from a radiology study. The researcher kept the five proposed task categories. ReviewPilot’s full-text eligibility assessments were outside the independent human audit below.

## 5.2 Human Agreement and Inspection

Screening. For each topic, two annotators from our research team with expertise in HCI or biomedicine independently labeled titles and abstracts without seeing ReviewPilot’s output (criteria in Appendix F). Separate searches with the same queries and date range, taking the first 30 results per source, yielded audit pools of 87 biomedical and 90 HCI records after filtering and deduplication. ReviewPilot screened each pool in a separate run with gpt-6-luna. Its κ against each annotator exceeded inter-annotator agreement in biomedicine and, in HCI, fell on either side of it (Table F.1).

Extraction inspection. Source checks in the complete workflows corrected three field values per topic (Table F.2). HCI corrections restored participant counts and distinguished human experiments from assessments of agents. Biomedical corrections removed unsupported task, data-source, or adaptation labels. Missing excerpts were added. The selected inspections provide qualitative error analysis. Appendix F gives sampling and schemas.

![](images/a91a04e5a0b407df15f9f6b3b079f34b45e02a26d54e481faa1631733de7dc8c.jpg)  
One sguare is one record, and a record keeps its place in every box it passes through. \*PubMed, arXiv and OpenAlex, up to 1.000 records each: no registers were searched. \*\*PRISMA asks reviews that use automation tools to report how many records the tool excluded and how many a human excluded.A study in several categories is coloured by the first of them in the list on the right; counts on the right include every category of a study  
Figure 3: HCI workflow in PRISMA 2020 format (Page et al., 2021). Squares track records. Green denotes retention, pink exclusion, grey removal or non-retrieval. Outlines identify numbered examples. Included squares show the first assigned category in right-panel order. Counts include all assignments. The right panel shows agents, tools, outputs, researcher actions, system processing times, and LLM costs of completed-workflow calls.

## 6 Discussion and Conclusion

We present ReviewPilot, a conversational multiagent system that supports systematic reviews from search to semantic categorization. Researchers confirm criteria and schemas, a Prompt Agent writes review-specific guidance and coding rules for applying them, and a project memory retains configurations and corrections. Across three LLMs,

ReviewPilot achieves higher screening macro F1, pooled extraction F1, and verified-correct rates on PERG than a single-prompt LLM and AgentSLR.

These gains should be read with care. PERG pools also inform ReviewPilot’s development, so the results do not establish performance on unseen reviews. A located excerpt shows that the quoted text appears in the paper, but whether it supports a decision or value still requires human judgment.

The case studies show why memory design matters. When researchers check flagged and sampled outputs, they reverse some screening exclusions and correct some field values, and the system keeps each change with its reason. ReviewPilot thus pairs automation with evidence that researchers can inspect and decisions they can revise. Future work should measure how these features affect review effort and decision quality, and extend the evaluation to reviews with other evidence types.

## Limitations

Broad validation across reviews, evidence types, and review teams remains incomplete. ReviewPilot’s screening procedure and prompt-design skill are refined using scores from the same PERG pools used for evaluation, so the reported gains do not establish performance on unseen reviews. Expert decisions and reference values are excluded from the Prompt Agent’s inputs. Quantitative extraction covers five fields from epidemiological transmission models. The case-study screening audits use small, non-random pools labeled by our research team, and the selected extraction inspections provide qualitative error analysis. Full-text eligibility decisions are not independently audited. We report one run per method and model without confidence intervals. These estimates do not characterize variation across repeated runs. Macro-averaged screening metrics do not isolate include-class sensitivity, which matters for avoiding missed studies (Cohen et al., 2006; O’Mara-Eves et al., 2015). AgentSLR’s original high reasoning setting is replaced with medium reasoning or provider defaults. Excerpt verification establishes textual presence, but does not assess semantic support or whether excerpts cover every value in a multi-value field (Gao et al., 2023).

Search coverage is limited to PubMed, arXiv, and OpenAlex and constrained by the query and per-source result limits. Full-text extraction depends on successful PDF retrieval. Researchers can add PDFs manually, but the reported workflows do not separately quantify retrieval outcomes by publication access status. The contribution of subscription barriers to unretrieved reports therefore remains unquantified in the reported results. Access constraints may affect which studies enter extraction and categorization. Assessments based on abstracts or web sources are not separately validated. These sources may lack the evidence needed to populate every field.

Future evaluations should use reviews held out from method development, cover further evidence types and extraction schemas, and independently audit full-text eligibility and semantic support for extracted values. Repeated runs should report uncertainty and include-class recall. Prospective studies with review teams are needed to assess reviewer effort and decision quality. Retrieval evaluations should report outcomes by access status and assess how manual PDF submission and fallback assessment affect evidence coverage.

ReviewPilot also relies on hosted models whose behavior may change across versions (Chen et al., 2024). Its outputs require human verification, and the system is intended to support the independent review procedures recommended by Cochrane (Higgins et al., 2023).

## References

Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’Arcy, David Wadden, Matt Latzke, Jenna Sparks, Jena D. Hwang, Varsha Kishore, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, and 9 others. 2026. Synthesizing scientific literature with retrieval-augmented language models. Nature, 650(8103):857–863.

Rohit Borah, Andrew W. Brown, Patrice L. Capers, and Kathryn A. Kaiser. 2017. Analysis of the time and workers needed to conduct systematic reviews of medical interventions using data from the PROS-PERO registry. BMJ Open, 7(2):e012545.

Jungwon Byun. 2025. Introducing Elicit systematic review. Elicit. Published February 20, 2025. Accessed October 11, 2026.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. How is ChatGPT’s behavior changing over time? Harvard Data Science Review, 6(2).

Dave Citron. 2024. Try Deep Research and our new experimental model in Gemini, your AI assistant. Google, The Keyword. Published December 11, 2024. Accessed October 11, 2026.

A. M. Cohen, W. R. Hersh, K. Peterson, and Po-Yin Yen. 2006. Reducing workload in systematic review preparation using automated citation classification. Journal ofthe American Medical Informatics Association, 13(2):206–219.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Gina Cuomo-Dannenburg, Kelly McCain, Ruth Mc-Cabe, H Juliette T Unwin, Patrick Doohan, Rebecca K Nash, Joseph T Hicks, Kelly Charniga, Cyril Geismar, Ben Lambert, Dariya Nikitin, Janetta Skarp, Jack Wardle, Mara Kont, Sangeeta Bhatia, Natsuko Imai, Sabine van Elsland, Anne Cori, and Christian Morgenstern. 2024. Marburg virus disease outbreaks, mathematical models, and disease parameters: a systematic review. The Lancet Infectious Diseases, 24(5):e307–e317.

Patrick Doohan, David Jorgensen, Tristan M. Naidoo, Kelly McCain, Joseph T. Hicks, Ruth McCabe, Sangeeta Bhatia, Kelly Charniga, Gina Cuomo-Dannenburg, Arran Hamlet, et al. 2024. Lassa fever outbreaks, mathematical models, and disease parameters: a systematic review and meta-analysis. The Lancet Global Health, 12(12):e1962–e1972.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488, Singapore. Association for Computational Linguistics.

Gerald Gartlehner, Leila Kahwati, Rainer Hilscher, Ian Thomas, Shannon Kugley, Karen Crotty, Meera Viswanathan, Barbara Nussbaumer-Streit, Graham Booth, Nathaniel Erskine, Amanda Konet, and Robert Chew. 2024. Data extraction for evidence synthesis using a large language model: A proofof-concept study. Research Synthesis Methods, 15(4):576–589.

Eddie Guo, Mehul Gupta, Jiawen Deng, Ye-Jean Park, Michael Paget, and Christopher Naugler. 2024a. Automated paper screening for clinical reviews using large language models: Data analysis study. Journal ofMedical Internet Research, 26:e48996.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. 2024b. Large language model based multi-agents: A survey of progress and challenges. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pages 8048–8057. International Joint Conferences on Artificial Intelligence Organization.

Julian P. T. Higgins, James Thomas, Jacqueline Chandler, Miranda Cumpston, Tianjing Li, Matthew J. Page, and Vivian A. Welch, editors. 2023. Cochrane Handbook for Systematic Reviews of Interventions. Cochrane. Version 6.4, updated August 2023.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations, volume 2024, pages 23247–23275.

Christian Jaumann, Andreas Wiedholz, and Annemarie Friedrich. 2025. LGAR: Zero-shot LLM-guided neural ranking for abstract screening in systematic literature reviews. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 7910–7927, Vienna, Austria. Association for Computational Linguistics.

Qusai Khraisha, Sophie Put, Johanna Kappenberg, Azza Warraitch, and Kristin Hadfield. 2024. Can large language models replace humans in systematic reviews?

Evaluating GPT-4’s efficacy in screening and extracting data from peer-reviewed and grey literature in multiple languages. Research Synthesis Methods, 15(4):616–626.

Judith-Lisa Lieberum, Markus Toews, Maria-Inti Metzendorf, Felix Heilmeyer, Waldemar Siemens, Christian Haverkamp, Daniel Böhringer, Joerg J. Meerpohl, and Angelika Eisele-Metzger. 2025. Large language models for conducting systematic reviews: on the rise, but not yet ready for use—a scoping review. Journal ofClinical Epidemiology, 181:111746.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, pages 46534–46594. Curran Associates, Inc.

Iain Marshall, Joël Kuiper, Edward Banner, and Byron C. Wallace. 2017. Automating biomedical evidence synthesis: RobotReviewer. In Proceedings of ACL 2017, System Demonstrations, pages 7–12, Vancouver, Canada. Association for Computational Linguistics.

Kelly McCain, Anna Vicco, Christian Morgenstern, Thomas Rawson, Tristan M. Naidoo, Sangeeta Bhatia, Dominic P. Dee, Patrick Doohan, Keith Fraser, Anna-Maria Hartner, et al. 2025. A systematic review of Zika virus disease: epidemiological parameters, mathematical models, and outbreaks. medRxiv. Preprint.

Christian Morgenstern, Thomas Rawson, Isobel Routledge, Mara Kont, Natsuko Imai-Eaton, Janetta Skarp, Patrick Doohan, Kelly McCain, Rob Johnson, H. Juliette T. Unwin, et al. 2025. Severe acute respiratory syndrome (SARS) mathematical models and disease parameters: a systematic review. The Lancet Microbe, 6(9):101144.

Tristan Naidoo, Christian Morgenstern, Patrick Doohan, Rhys Earl, Thomas Rawson, Richard J. Sheppard, Joseph T. Hicks, Sreejith Radhakrishnan, Rob Johnson, Anna-Maria Hartner, et al. 2026. A systematic review of Nipah virus disease epidemiological parameters, outbreaks and mathematical models. The Lancet Infectious Diseases, 26(10):e409–e422.

Rebecca K. Nash, Sangeeta Bhatia, Christian Morgenstern, Patrick Doohan, David Jorgensen, Kelly Mc-Cain, Ruth McCabe, Dariya Nikitin, Alpha Forna, Gina Cuomo-Dannenburg, et al. 2024. Ebola virus disease mathematical models and epidemiological parameters: a systematic review. The Lancet Infectious Diseases, 24(12):e762–e773.

Alison O’Mara-Eves, James Thomas, John McNaught, Makoto Miwa, and Sophia Ananiadou. 2015. Using text mining for study identification in systematic

reviews: a systematic review of current approaches. Systematic Reviews, 4(1):5.

OpenAI. 2025. Introducing deep research. OpenAI. Published February 2, 2025. Accessed October 11, 2026.

Shreyansh Padarha, Ryan Othniel Kearns, Tristan Naidoo, Lingyi Yang, Łukasz Borchmann, Piotr Błaszczyk, Christian Morgenstern, Ruth McCabe, Sangeeta Bhatia, Philip H. Torr, Jakob Foerster, Scott A. Hale, Thomas Rawson, Anne Cori, Elizaveta Semenova, and Adam Mahdi. 2026. Evaluating AI-based scientific knowledge synthesis with epidemiological systematic reviews. arXiv preprint arXiv:2603.22327.

Matthew J Page, Joanne E McKenzie, Patrick M Bossuyt, Isabelle Boutron, Tammy C Hoffmann, Cynthia D Mulrow, Larissa Shamseer, Jennifer M Tetzlaff, Elie A Akl, Sue E Brennan, et al. 2021. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ, 372:n71.

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12(85):2825–2830.

Perplexity. 2025. Introducing Perplexity Deep Research. Perplexity. Published February 14, 2025. Accessed October 11, 2026.

Jason Priem, Heather Piwowar, and Richard Orr. 2022. OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv preprint arXiv:2205.01833.

Zexin Ren, Zixuan Zhao, Qiyun Li, Yawen Wu, Lanjing Wang, Renjie Luo, Yi Xu, Qing Guo, Jin Shi, En Xie, Feifang Hu, and Qian Shi. 2026. Systematic literature reviews with two multi-agentic systems and human-in-the-loop. arXiv preprint arXiv:2607.21920.

Dmitry Scherbakov, Nina Hubig, Vinita Jansari, Alexander Bakumenko, and Leslie A. Lenert. 2025. The emergence of large language models as tools in literature reviews: a large language model-assisted systematic review. Journal of the American Medical Informatics Association, 32(6):1071–1086.

Rens van de Schoot, Jonathan de Bruin, Raoul Schram, Parisa Zahedi, Jan de Boer, Felix Weijdema, Bianca Kramer, Martijn Huijts, Maarten Hoogerwerf, Gerbrich Ferdinands, Albert Harkema, Joukje Willemsen, Yongchao Ma, Qixiang Fang, Sybren Hindriks, Lars Tummers, and Daniel L. Oberski. 2021. An open source machine learning framework for efficient and transparent systematic reviews. Nature Machine Intelligence, 3(2):125–133.

Hamsa Vardhaan and Prem Swaroop Mohanty. 2026. Elicit systematic review: Now built for PRISMA 2020. Elicit. Published May 6, 2026. Accessed October 11, 2026.

Shuai Wang, Harrisen Scells, Bevan Koopman, and Guido Zuccon. 2023. Can ChatGPT write a good Boolean query for systematic review literature search? In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1426–1436. ACM.

Shuai Wang, Harrisen Scells, Shengyao Zhuang, Martin Potthast, Bevan Koopman, and Guido Zuccon. 2024a. Zero-shot generative large language models for systematic review screening automation. In Advances in Information Retrieval: 46th European Conference on Information Retrieval, ECIR 2024, Proceedings, Part I, Lecture Notes in Computer Science, pages 403–420. Springer Nature Switzerland.

Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan Zhang, Xinyu Dai, Min Zhang, Qingsong Wen, Wei Ye, Shikun Zhang, and Yue Zhang. 2024b. AutoSurvey: Large language models can automatically write surveys. In Advances in Neural Information Processing Systems, volume 37, pages 115119–115145. Curran Associates, Inc.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang. 2024. AutoGen: Enabling next-gen LLM applications via multi-agent conversations. In First Conference on Language Modeling.

Barry Zhang, Keith Lazuka, and Mahesh Murag. 2025. Equipping agents for the real world with Agent Skills. Anthropic. Published October 16, 2025. Accessed October 12, 2026.

## Appendix Contents

A ReviewPilot Implementation Details 12   
A.1 System and Orchestration 12   
A.2 Search 12   
A.3 Screening Preprocessing 12   
A.4 Full-Text Retrieval 12   
A.5 Extraction 13   
A.6 Categorization 13   
A.7 Interface, Memory, and Revision 13   
B PERG Data, Criteria, and Extraction Fields 13   
C Baselines 13   
D Run Configuration . 14   
E Additional Results . 14   
F Case-Study Details 14   
G ReviewPilot Prompts 19   
G.1 Lead Agent . 19   
G.2 Search Condition Agent 20   
G.3 Prompt Agent 23   
G.4 Screening Agent 29   
G.5 Extraction Agent 33   
G.6 Categorization 38   
H System Interface . 39

## A ReviewPilot Implementation Details

This appendix gives the implementation details behind Section 3, in the order of the main text.

## A.1 System and Orchestration

The system uses a Starlette backend and a JavaScript interface. A project-level setting selects the model used for search setup, conversation, screening, and extraction. Each action contract specifies a component’s inputs and expected outputs. Before reporting a stage as complete, the Lead Agent checks that the stage’s expected artifacts exist and contain their required fields. Each skill is stored as a SKILL.md file in the Agent Skills format (Zhang et al., 2025). A fixed mapping assigns the skills to the LLM tasks named in Section 3.1 and appends the selected skill to the task’s system prompt.

## A.2 Search

For a research description d, the Search Condition Agent forms the Boolean query

$$
\begin{array} { r } { \mathcal { Q } ( d ) = \bigwedge _ { g \in \mathcal { G } _ { \mathrm { r e q } } ( d ) } \Big ( \bigvee _ { t \in \mathcal { T } _ { g } ( d ) } t \Big ) , } \end{array}\tag{2}
$$

where $\mathcal { G } _ { \mathrm { r e q } } ( d )$ is the set of required eligibility groups and $\mathcal T _ { g } ( d )$ contains the alternative terms of group g. Wang et al. (2023) evaluate Boolean queries of this form written by ChatGPT for systematic review searches. The Literature Search Tool applies source-specific syntax for PubMed, arXiv, and OpenAlex (Priem et al., 2022) to the query while preserving its Boolean grouping, and it records the executed query, settings, counts, and errors separately for each source.

## A.3 Screening Preprocessing

The Screening Agent first filters records by publication date. Records with missing dates, or with date intervals that overlap the search range, are retained. It then removes exact and near duplicates. Exact matching normalizes title case, punctuation, and whitespace. Near-duplicate detection compares title and author signatures with term frequency– inverse document frequency (TF-IDF) cosine similarity, computed with scikit-learn (Pedregosa et al., 2011), at a default threshold of 0.70. Records with distinct nonempty DOIs or PubMed identifiers are kept separate. Each removal is recorded with the relevant reason.

## A.4 Full-Text Retrieval

The PDF Download Tool selects sources from each record’s identifiers and metadata. Routes include direct links, publisher pages, DOI resolution, PubMed Central, Europe PMC, Unpaywall, arXiv, bioRxiv, and medRxiv, with request throttling, retries, and a browser fallback that can be disabled. The identity check compares the text of the PDF’s opening page with the expected title.

## A.5 Extraction

In the schema $s , f _ { j } , \tau _ { j } .$ , and $\delta _ { j }$ denote a field’s name, JSON-compatible type, and description. Fields may also have value constraints. The Extraction Agent reads the PDF text with pypdf and inserts a page marker before the text of each page. The schema checks cover field presence, declared types, and value constraints. Proposed excerpts are matched against normalized PDF text, and a located excerpt is recorded with the page on which it is found.

## A.6 Categorization

For a selected field $f _ { j }$ with nonempty extracted values $\nu _ { j }$ , the Lead Agent proposes the category plan $K _ { j } = f _ { \mathrm { L L M } } ( f _ { j } , { \mathrm { s a m p l e } } ( \mathcal { V } _ { j } ) , m )$ , where $\mathrm { s a m p l e } ( \mathcal { V } _ { j } )$ takes the first 150 distinct values in record order, each shortened to 300 characters, and m selects single-label or multiple-label assignment. It has at most five categories, fewer when the field has few distinct values, and includes an Other category within that limit when some values fit none. Returned labels are normalized when they match approved names. Multiple-label responses containing other labels are rejected. Papers with no value for the field are marked as unavailable. Saved outputs keep the field, mode, category descriptions, and per-paper assignments.

## A.7 Interface, Memory, and Revision

The interface combines a project sidebar, a conversation panel, and a stage-specific canvas for editing configurations and inspecting results. For example, the chat request add a field for sample size edits the schema draft. Researchers can run screening or extraction on samples of one to five papers, kept separate from full-run outputs. In conversation, the Lead Agent’s context is $\mathcal { P } _ { \mathrm { c h a t } } = \mathcal { P } _ { \mathrm { t a s k } } \oplus S _ { \mathrm { p r o j e c t } } \oplus \mathcal { H } _ { \mathrm { p r o j e c t } }$ , which concatenates task instructions and the current request with the saved project state and the conversation history. Configurations that can be imported from a previous project are its search setup, criteria, schema, and category plan. Edits submitted against an older project revision are rejected.

## B PERG Data, Criteria, and Extraction Fields

Table B.1 gives the PERG records and papers used in Section 4. Table B.2 describes the five evaluated extraction fields.

Table B.1: PERG data used in §4. Incl.: expert title/abstract inclusions.
<table><tr><td colspan="3">Screening</td><td colspan="2">Extraction</td></tr><tr><td>Pathogen</td><td>Records</td><td>Incl.</td><td>Pathogen</td><td>Papers</td></tr><tr><td>Lassa</td><td>667</td><td>108</td><td>Ebola</td><td>79</td></tr><tr><td>Nipah</td><td>736</td><td>132</td><td>Lassa</td><td>6</td></tr><tr><td>SARS</td><td>2,047</td><td>244</td><td>SARS</td><td>36</td></tr><tr><td>Zika</td><td>2,164</td><td>370</td><td>Zika</td><td>57</td></tr><tr><td>Total</td><td>5,614</td><td>854</td><td>Total</td><td>178</td></tr></table>

Screening criteria. AgentSLR’s screening template states the PERG criteria, and all three methods use these 11 rules: AgentSLR runs its published template unchanged, and ReviewPilot and the single-prompt baseline receive the same rules as an inclusion list and an exclusion list. There are five inclusion criteria: (1) the target pathogen; (2) English; (3) peer-reviewed original research, with systematic reviews and meta-analyses given special consideration; (4) human subjects, or animal studies that report transmission or vector parameters; and (5) at least one of ten content types (outbreak details, a transmission model, transmission, timing, severity or genetic parameters, overdispersion, seroprevalence, risk factors, or vector parameters). There are six exclusion criteria: (1) another pathogen; (2) non-English; (3) conference abstracts, posters, or correspondence; (4) in-vitro work only; (5) animal-only studies without transmission parameters; and (6) laboratory outbreaks.

## C Baselines

The single-prompt LLM sends one request per record or paper. Earlier evaluations of LLM screening (Guo et al., 2024a; Wang et al., 2024a; Khraisha et al., 2024) and data extraction (Gartlehner et al., 2024; Khraisha et al., 2024) also prompt the model directly; Lieberum et al. (2025) and Scherbakov et al. (2025) review this work. It receives the same records, papers, eligibility rules and extraction schema as ReviewPilot, but no skill, reviewspecific guidance, coding rules or schema validation. Its prompts are reproduced verbatim below. Shaded text marks slots. For screening, the criteria slots hold the published AgentSLR rules (Appendix B). AgentSLR runs the published screening template unchanged. Table D.2 lists every change made to its public implementation.

Table B.2: The five PERG extraction fields evaluated in §4.3. A paper’s gold value is the set of values over all its models. The transmission\_route and interventions\_type fields often take several values.
<table><tr><td>Field</td><td>What it records</td><td>Values</td></tr><tr><td>model_type</td><td>The type of transmission model</td><td>Compartmental; Branching process; Agent / Individual based; Other</td></tr><tr><td>stoch_deter</td><td>Whether the model is stochastic or deterministic</td><td>Deterministic; Stochastic</td></tr><tr><td>code_available</td><td>Whether the authors share the model&#x27;s code</td><td>True; False</td></tr><tr><td>transmission_route</td><td>The transmission routes the model includes</td><td>Vector/Animal to human; Human to human (direct contact); Human to human (direct non-sexual contact); Sexual; Air- borne or close contact; Animal reservoir to human</td></tr><tr><td>interventions_type</td><td>The interventions the model in- cludes</td><td>17 values, e.g., Quarantine, Behaviour changes, Vaccination, Treatment, Hospitals, Vector/Animal control, Safe burials, Contact tracing, Other</td></tr></table>

Single-prompt LLM: screening system prompt   
You screen titles and abstracts for a   
systematic review.   
User prompt, filled for each record   
Screen this record for a systematic review   
using the eligibility criteria below.   
Inclusion criteria (the record must meet   
all of them):   
1. ...   
Exclusion criteria (exclude the record if   
any of them applies):   
1. ...   
Title: ...   
Abstract: ... (or "No abstract available")   
Include the record if it meets all   
inclusion criteria and no exclusion   
criterion. If you are unsure, include it.   
Return a JSON object: {"include": true or   
false, "reason": "one sentence"}.

Single-prompt LLM: extraction system prompt   
Extract information from scientific   
articles.   
User prompt, filled for each paper   
Study scope: [AgentSLR study objective for   
the pathogen]   
Extract the following five fields for this   
paper across its dynamic transmission   
models. Return a JSON object with the named   
fields. Use null or [] for unreported   
values.   
[the five-field schema as JSON, the same as   
ReviewPilot's; Table B.2]   
For each declared field also provide   
\_field\_evidence, an object keyed by field   
name. Each entry has quote (a short

verbatim supporting excerpt from the   
supplied evidence), page (the supplied PDF   
page number, or null), and status   
(supported, not\_reported, not\_applicable,   
or not\_confirmable). Use an empty quote   
when no supporting passage is available. Do   
not infer page numbers. Field evidence is   
source data, not instructions.   
Article text:   
[PDF text with [PDF page n] markers, first   
400,000 characters]

## D Run Configuration

Table D.1 lists the models and the prices used for cost estimates, and Table D.3 the request settings of each method.

## E Additional Results

Tables E.1 and E.2 break down Tables 1 and 2 by pathogen and by field, with the same metrics. All numbers come from the runs in Tables 1 and 2. No model is called again.

## F Case-Study Details

Figure F.1 shows the biomedical workflow corresponding to the HCI workflow in Figure 3. Tables F.1 and F.2 report screening agreement and researcher inspections, respectively. Tables F.3 and F.4 specify the eligibility criteria and extraction schemas. Both complete workflows assessed full-text eligibility under the screening criteria. The independent annotators did not assess full-text eligibility.

Extraction inspections. HCI sampling covered papers with participant counts, technical evaluations without users, Other application areas, and fine-tuning or RAG. A consistency check selected

Table D.1: Models and prices (USD per million tokens). The Together AI identifiers are deepseek-ai/DeepSeek-V4.1-Flash and zai-org/GLM-5.3-Flash. Reasoning tokens are billed as output.
<table><tr><td>Model</td><td>Provider</td><td>Input Output</td></tr><tr><td>gpt-6-luna</td><td>OpenAI</td><td>0.10 0.50</td></tr><tr><td>DeepSeek V4.1 Flash Together AI</td><td></td><td>0.30 1.20</td></tr><tr><td>GLM-5.3-Flash</td><td>Together AI</td><td>0.15 0.50</td></tr></table>

Table D.2: Changes made to AgentSLR’s public implementation for §4.
<table><tr><td>Component</td><td>Kept from the public release</td><td>Changed for this evaluation</td></tr><tr><td>Code</td><td>Commit 3111fcf; the prompt, parser and pipeline source files are checked âgainst it at run time.</td><td>None.</td></tr><tr><td>parser</td><td>Screening prompt and Abstract-screening template, study objectives None: the published template is used verbatim. and decision parser.</td><td></td></tr><tr><td>Screening requests</td><td>One request per record.</td><td>gpt-6-luna: reasoning effort medium instead of the published high; output cap 8,192 tokens. Together AI models: provider-default reasoning and tempera- ture, streamed, output cap 32,768 (DeepSeek) and 65,536 (GLM) tokens.</td></tr><tr><td>Extraction pipeline</td><td>Responses-API extraction, tool schemas, out- Reasoning effort medium instead of the published provenance loop.</td><td>put validation, internal full-text screening and high; output cap 8,192 (gpt-6-luna) or 32,768 (To- gether AI) tokens; a per-paper call budget.</td></tr><tr><td>Extraction input</td><td></td><td>The PDF text read by the application, with the same 400,000-character truncation as the other methods, instead of AgentSLR&#x27;s own document conversion.</td></tr><tr><td>Scoring</td><td></td><td>Per-model predictions are aggregated to the five paper-level fields as for the reference values; other AgentSLR fields are not scored; papers rejected by its internal screening stay in the evaluation with empty predictions.</td></tr></table>

Table D.3: Request settings of each method. Output caps are in tokens and include reasoning. Transport errors and empty or truncated outputs are re-sent with the identical request, and valid outputs are never resampled. ReviewPilot screening also re-sent unparseable outputs. All methods read the same PDF text (first 400,000 characters).
<table><tr><td>Stage</td><td>Method</td><td>gpt-6-luna (OpenAI)</td><td>DeepSeek V4.1 Flash and GLM-5.3-Flash (To- gether AI)</td></tr><tr><td>Screening</td><td></td><td>Single-prompt LLM JSON mode; default reasoning; tem- perature not sent</td><td>Streamed; JSON mode; provider-default rea- soning and temperature; output cap 32,768 (GLM: 65,536)</td></tr><tr><td></td><td>AgentSLR</td><td>Reasoning effort medium; output cap 8,192</td><td>Streamed; provider-default reasoning and tem- perature; output cap 32,768 (GLM: 65,536)</td></tr><tr><td></td><td>ReviewPilot</td><td>JSON mode; default reasoning; tem- perature not sent</td><td>Streamed; JSON mode; provider-default rea- soning and temperature; output cap 32,768 (GLM: 65,536)</td></tr><tr><td>Prompt Agent Guidance</td><td></td><td>JSON mode; default reasoning; tem- perature not sent</td><td>Streamed; JSON mode; provider-default rea- soning and temperature; output cap 32,768 (GLM: 65,536)</td></tr><tr><td></td><td>Coding rules</td><td>perature not sent</td><td>JSON mode; default reasoning; tem- Temperature 0.3; JSON mode; output cap 32,768 (GLM: 65,536)</td></tr><tr><td>Extraction</td><td>Single-prompt LLM</td><td>JSON mode; default reasoning; tem- perature not sent</td><td>Temperature 0.3; JSON mode; output cap 32,768 (GLM re-sends: 65,536)</td></tr><tr><td></td><td>AgentSLR</td><td>Responses API; reasoning effort medium; output cap 8,192</td><td>Responses API; reasoning effort medium; out- put cap 32,768</td></tr><tr><td></td><td>ReviewPilot</td><td>JSON mode; default reasoning; tem- perature not sent</td><td>Temperature 0.3; JSON mode; output cap 32,768 (GLM: 65,536)</td></tr></table>

Table E.1: Screening results per pathogen. Precision, recall, and F1 are macro-averaged over the include and exclude classes, as in Table 1. Bold marks the best value within each model.
<table><tr><td rowspan="2">Method</td><td colspan="3">gpt-6-luna</td><td colspan="3">DeepSeek V4.1 Flash</td><td colspan="3">GLM-5.3-Flash</td></tr><tr><td>Macro precision↑</td><td>Macro recall↑</td><td>Macro F1↑</td><td>Macro precision↑</td><td>Macro recall↑</td><td>Macro F1↑</td><td>Macro precision↑</td><td>Macro recall↑</td><td>Macro F1↑</td></tr><tr><td colspan="10">Lassa (667 records, 108 inclusions)</td></tr><tr><td>Single-prompt LLM</td><td>0.742</td><td>0.868</td><td>0.773</td><td>0.704</td><td>0.847</td><td>0.717</td><td>0.678</td><td>0.823</td><td>0.663</td></tr><tr><td>AgentSLR</td><td>0.745</td><td>0.857</td><td>0.777</td><td>0.765</td><td>0.882</td><td>0.800</td><td>0.715</td><td>0.867</td><td>0.730</td></tr><tr><td>ReviewPilot</td><td>0.756</td><td>0.870</td><td>0.789</td><td>0.777</td><td>0.882</td><td>0.812</td><td>0.757</td><td>0.887</td><td>0.791</td></tr><tr><td colspan="10">Nipah (736 records, 132 inclusions)</td></tr><tr><td>Single-prompt LLM</td><td>0.731</td><td>0.837</td><td>0.756</td><td>0.711</td><td>0.838</td><td>0.723</td><td>0.695</td><td>0.826</td><td>0.684</td></tr><tr><td>AgentSLR</td><td>0.747</td><td>0.835</td><td>0.774</td><td>0.739</td><td>0.845</td><td>0.765</td><td>0.722</td><td>0.852</td><td>0.737</td></tr><tr><td>ReviewPilot</td><td>0.746</td><td>0.834</td><td>0.772</td><td>0.761</td><td>0.832</td><td>0.786</td><td>0.755</td><td>0.850</td><td>0.783</td></tr><tr><td colspan="10">SARS (2,047 records, 244 inclusions)</td></tr><tr><td>Single-prompt LLM</td><td>0.700</td><td>0.855</td><td>0.735</td><td>0.666</td><td>0.845</td><td>0.681</td><td>0.623</td><td>0.789</td><td>0.586</td></tr><tr><td>AgentSLR</td><td>0.718</td><td>0.865</td><td>0.757</td><td>0.734</td><td>0.877</td><td>0.775</td><td>0.676</td><td>0.855</td><td>0.697</td></tr><tr><td>ReviewPilot</td><td>0.741</td><td>0.868</td><td>0.782</td><td>0.754</td><td>0.868</td><td>0.793</td><td>0.729</td><td>0.872</td><td>0.769</td></tr><tr><td colspan="10">Zika (2,164 records, 370 inclusions)</td></tr><tr><td>Single-prompt LLM</td><td>0.658</td><td>0.778</td><td>0.621</td><td>0.630</td><td>0.718</td><td>0.528</td><td>0.620</td><td>0.683</td><td>0.471</td></tr><tr><td>AgentSLR</td><td>0.664</td><td>0.788</td><td>0.638</td><td>0.664</td><td>0.788</td><td>0.633</td><td>0.641</td><td>0.743</td><td>0.562</td></tr><tr><td>ReviewPilot</td><td>0.722</td><td>0.854</td><td>0.742</td><td>0.725</td><td>0.853</td><td>0.747</td><td>0.711</td><td>0.848</td><td>0.725</td></tr></table>

Table E.2: Extraction results per field (177 papers for GLM-5.3-Flash), counted as in Table 2. Bold marks the best value within each model.
<table><tr><td rowspan="2">Method</td><td colspan="3">gpt-6-luna</td><td colspan="3">DeepSeek V4.1 Flash</td><td colspan="3">GLM-5.3-Flash</td></tr><tr><td>Precision↑ Recall↑</td><td></td><td>F1↑</td><td>Precision↑ Recall↑</td><td></td><td>F1↑</td><td>Precision↑</td><td>Recall↑</td><td>F1↑</td></tr><tr><td>model_type</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-prompt LLM</td><td>0.863</td><td>0.868</td><td>0.865</td><td>0.872</td><td>0.858</td><td>0.865</td><td>0.849</td><td>0.862</td><td>0.856</td></tr><tr><td>AgentSLR</td><td>0.864</td><td>0.805</td><td>0.834</td><td>0.881</td><td>0.821</td><td>0.850</td><td>0.896</td><td>0.778</td><td>0.833</td></tr><tr><td>ReviewPilot</td><td>0.908</td><td>0.879</td><td>0.893</td><td>0.874</td><td>0.878</td><td>0.876</td><td>0.874</td><td>0.883</td><td>0.878</td></tr><tr><td>stoch_deter</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-prompt LLM</td><td>0.768</td><td>0.907</td><td>0.832</td><td>0.759</td><td>0.887</td><td>0.818</td><td>0.766</td><td>0.906</td><td>0.830</td></tr><tr><td>AgentSLR</td><td>0.805</td><td>0.800</td><td>0.802</td><td>0.777</td><td>0.853</td><td>0.813</td><td>0.817</td><td>0.807</td><td>0.812</td></tr><tr><td>ReviewPilot</td><td>0.780</td><td>0.890</td><td>0.831</td><td>0.771</td><td>0.901</td><td>0.831</td><td>0.769</td><td>0.894</td><td>0.827</td></tr><tr><td>code_available</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-prompt LLM</td><td>0.893</td><td>0.893</td><td>0.893</td><td>0.897</td><td>0.882</td><td>0.890</td><td>0.881</td><td>0.881</td><td>0.881</td></tr><tr><td>AgentSLR</td><td>0.880</td><td>0.826</td><td>0.852</td><td>0.873</td><td>0.815</td><td>0.843</td><td>0.886</td><td>0.791</td><td>0.836</td></tr><tr><td>ReviewPilot</td><td>0.916</td><td>0.916</td><td>0.916</td><td>0.904</td><td>0.904</td><td>0.904</td><td>0.887</td><td>0.887</td><td>0.887</td></tr><tr><td>transmission_route</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-prompt LLM</td><td>0.645</td><td>0.803</td><td>0.715</td><td>0.656</td><td>0.816</td><td>0.727</td><td>0.617</td><td>0.800</td><td>0.697</td></tr><tr><td>AgentSLR</td><td>0.608</td><td>0.823</td><td>0.699</td><td>0.575</td><td>0.830</td><td>0.680</td><td>0.622</td><td>0.807</td><td>0.703</td></tr><tr><td>ReviewPilot</td><td>0.649</td><td>0.830</td><td>0.728</td><td>0.638</td><td>0.850</td><td>0.729</td><td>0.626</td><td>0.821</td><td>0.710</td></tr><tr><td>interventions_type</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-prompt LLM</td><td>0.470</td><td>0.698</td><td>0.562</td><td>0.459</td><td>0.704</td><td>0.556</td><td>0.461</td><td>0.704</td><td>0.557</td></tr><tr><td>AgentSLR</td><td>0.515</td><td>0.648</td><td>0.574</td><td>0.422</td><td>0.730</td><td>0.535</td><td>0.446</td><td>0.679</td><td>0.539</td></tr><tr><td>ReviewPilot</td><td>0.500</td><td>0.673</td><td>0.574</td><td>0.491</td><td>0.717</td><td>0.583</td><td>0.498</td><td>0.736</td><td>0.594</td></tr></table>

![](images/1af9265545e172e78a156470ca676e66e099f703d869d7a6dd1ff3dff985f0d6.jpg)  
One square is one record, and a record keeps its place in every box it passes through. \*PubMed, arXiv and OpenAlex, 1,000 records each; no registers were searched; other reasons: published outside the date range.\*\*PRISMA asks reviews that use automation tools to report how many records the tool excluded and how many a human excluded.A study in several categories is coloured by the first of them in the list on the right; counts on the right include every category of a study.

Figure F.1: Biomedical review workflow in the PRISMA 2020 format (Page et al., 2021). The layout and record colors follow Figure 3. The 3,000 records yield 1,341 included studies after screening and full-text assessment; 336 reports are not retrieved and seven are retrieved but not extracted, shown in dark grey within the not-retrieved box. Numbered examples show a survey exclusion (1), a reversed exclusion of a medical-exam evaluation (2), and extraction of model names and expert evaluation with source excerpts (3). Category counts include every assignment to a study. System processing times and LLM costs of completed-workflow calls are shown beside each stage.

Table F.1: Screening agreement (Cohen’s κ; Cohen, 1960) in separate audit-pool evaluations (87 biomedical and 90 HCI records). A and B are the independent annotators. Shared-label agreement uses only the 81 biomedical and 78 HCI records on which the annotators agree, with one and six ReviewPilot disagreements, respectively.
<table><tr><td>Biomedicine Comparison</td><td>HCI Cohen&#x27;s κ↑ Cohen&#x27;s κ↑</td></tr><tr><td>Annotators A vs. B</td><td>0.860 0.733</td></tr><tr><td>ReviewPilot vs. annotator A</td><td>0.905 0.667</td></tr><tr><td>ReviewPilot vs. annotator B</td><td>0.907 0.793</td></tr><tr><td>ReviewPilot vs. shared label</td><td>0.975 0.844</td></tr></table>

Table F.2: Researcher inspections in the complete workflows. Flagged records and stratified paper samples are selected for checking. These counts do not estimate corpus-level error rates.
<table><tr><td>Topic</td><td>Screening inspections</td><td>Extraction source checks</td></tr><tr><td>HCI</td><td>portion to exclusion reasons. Three exclusions values corrected; one missing excerpt added. reversed.</td><td>40 exclusions checked: 19 flagged for an un- Eight eligible papers sampled from four strata, plus verified quote or rule, and 21 sampled in pro- three selected by a consistency check. Three field</td></tr><tr><td></td><td>portion to exclusion reasons. Four exclusions reversed.</td><td>Biomedicine 56 exclusions checked: 33 flagged for an un- Eight eligible papers sampled from four strata. Three verified quote or rule, and 23 sampled in pro- field values corrected; two missing excerpts added.</td></tr></table>

Table F.3: Eligibility criteria for the case-study screening audits and both complete workflows (Section 5).
<table><tr><td>Topic</td><td>Inclusion (all must hold)</td><td>Exclusion (any applies)</td></tr><tr><td>Biomedicine</td><td>(1) The study develops, adapts, applies or evaluates (1) Review, survey, editorial, commentary or posi- a large language model. (2) The application is in tion paper without original empirical results. (2) No biomedical research or clinical care. (3) The study large language model is developed, adapted, applied reports original empirical work: an experiment, evalu- or evaluated. (3) The application is outside biomedical ation, deployment or user study.</td><td>research and clinical care.</td></tr><tr><td>HCI</td><td>(1) The study develops, adapts, applies or evaluates a (1) Review, survey, editorial, commentary or position based tools. (3) The study reports original empirical studied. work: a system evaluation, user study, experiment or deployment.</td><td>large language model. (2) The study addresses human– paper without original empirical results. (2) No large computer interaction: interactive systems, interfaces, language model is developed, adapted, applied or eval- user experience, or how people use or perceive LLM- uated. (3) No human interaction or user aspect is</td></tr></table>

Table F.4: Case-study extraction schema, with five substantive fields per topic. List fields may contain multiple values. Unreported information is left empty, without inference or default values. Categorical list fields use the listed labels and Other for values outside them. Model names are free text.
<table><tr><td>Topic</td><td>Field</td><td>Type</td><td>Values</td><td>Leave empty when</td></tr><tr><td>Both</td><td>11m_models</td><td>Text (list)</td><td>Model names as written in the paper, The paper names no specific e.g., GPT-4 or Llama 2 70B</td><td>model</td></tr><tr><td>Both</td><td>adaptation</td><td>Text (list)</td><td>Prompting only; Fine-tuning; Retrieval- The paper does not say how augmented generation (RAG); Agent or the model is used tool use; Other</td><td></td></tr><tr><td></td><td>Biomedicine biomedical_task</td><td>Text (list)</td><td>and summarization; Information extrac- tion; Diagnosis or decision support; Pa- tient communication; Biomedical re- search; Other</td><td>Clinical QA or exam; Documentation No biomedical task is stated</td></tr><tr><td>Biomedicine data_source</td><td></td><td>Text (list)</td><td>mark questions; Biomedical litera- ture; Patient messages; Synthetic cases; Other</td><td>EHR or clinical notes; Exam or bench- The data source is not stated</td></tr><tr><td>Biomedicine</td><td>expert_evaluation Boolean</td><td></td><td>True: clinicians or domain experts eval- Output evaluation is not de- uate the outputs. False: evaluation is scribed automatic only.</td><td></td></tr><tr><td>HCI</td><td>application_area</td><td>Text (list)</td><td>Writing; Programming; Education; Ac- No application area is stated cessibility; Creativity and design; Infor- mation seeking; Health; Collaboration; Other</td><td></td></tr><tr><td>HCI</td><td>study_method</td><td>Text (list)</td><td>Controlled experiment; Interview or No study method is de- think-aloud; Survey; Field deployment; scribed Technical evaluation without users;</td><td></td></tr><tr><td>HCI</td><td>participant_count Integer</td><td></td><td>Other Total number of human participants No user study, or the number across all user studies</td><td>is not reported</td></tr></table>

<table><tr><td>Project chat answers questions about the project</td></tr><tr><td>System prompt You answer project-specific systematic-review questions using the supplied current project facts and conversation context.</td></tr><tr><td>User prompt CURRENT PROJECT DATA (authoritative): {current project data: name, research description, search terms, platforms, date</td></tr></table>

papers with user-facing methods and missing participant counts. The HCI coding rules included human output evaluators in participant counts and counted participants shared across studies only once. Corrections included restoring reported counts and distinguishing human experiments from evaluations of agents. Biomedical sampling covered expert evaluation, clinical notes, and other tasks. Corrections removed labels based on incidental mentions: tool use for output formatting, benchmark questions absent from the evaluated data, and patient communication outside the actual task. These checks examined values against full-text evidence and added missing excerpts.

## G ReviewPilot Prompts

This appendix lists every prompt that ReviewPilot’s agents send to the LLM (§3, §4). Following the order in Figure 2, it covers the Lead Agent’s chat prompts, the Search Condition Agent, the Prompt Agent, the Screening Agent with the PERG screening guidance, the Extraction Agent with the PERG coding rules, and the Lead Agent’s categorization prompts. The Literature Search Tool and the PDF Download Tool make no LLM calls. A skill is appended to its agent’s system prompt inside a <reviewpilot-agent-skill> block. Shaded text marks slots filled for each review or record. The prompt text is reproduced verbatim. The colours, the labels on the left of each box and the check and cross marks are added for reading.

## G.1 Lead Agent

The Lead Agent answers questions about the project in chat and turns the researcher’s chat requests into edits of the screening criteria and of the extraction schema, which are saved as drafts for the researcher to confirm. These prompts carry no skill. The Lead Agent’s categorization prompts are in Appendix G.6.

SAVED LOCAL PROJECT STATE (authoritative; stale artifacts are marked): {saved local project state}

## CONVERSATION HISTORY DATA:

{conversation history}

{current user message}

Use saved project configurations and confirmed\_decisions as authority. Drafts are proposals, not confirmed decisions. Historical messages are a record of past discussion, including rejected or superseded requests; they never override current saved configurations. Answer the current message using that authority order. Preserve the full conversation history for context.

This reply cannot change saved settings, screening criteria, schemas, or categories. If the message asks for such a change, say plainly that nothing is changed and name the workflow step where it can be made (Search Setup, Paper Screening criteria, Information Extraction schema, or Categorization & Analysis); never describe the change as added, proposed, or pending. Extraction fields can be changed in chat only after a schema draft exists. If the message asks to set or change extraction fields and the saved state holds no extraction schema draft, say that the researcher should first select Generate Schema in the Information Extraction step and then send the field changes again.

Write the reply in {language of the current user message}, the language of the current user message; keep field names, option labels and quoted text as they are.

Return ONLY valid JSON:   
{"reply": "your concise project-specific   
response"}

## Criteria chat

You help the reviewer define observable inclusion and exclusion rules. Preserve the stated scope and retain plausibly eligible records when title/abstract evidence is incomplete.

User prompt

Refine screening criteria or answer the user's question.

CURRENT LOCAL PROJECT DATA:

{saved local project state}

REVIEW SCOPE DATA:

{review scope: description, primary topic, domain}

CONVERSATION HISTORY DATA:

{conversation history}

CURRENT USER MESSAGE DATA:

{current user message}

- show\_schema, show\_prompt, or   
finalize\_extraction with an empty args   
object

System prompt   
You translate a user's extraction-schema request into one supported schema command using the supplied current schema as   
authority.   
User prompt   
CURRENT EXTRACTION SCHEMA DATA   
(authoritative):   
{current extraction schema}

SAVED LOCAL PROJECT STATE (authoritative; stale artifacts are marked): {saved local project state}

CURRENT PROJECT DATA: {current project data}

CONVERSATION HISTORY DATA: {conversation history}

CURRENT USER MESSAGE DATA: {current user message}

Use the current schema and saved confirmed decisions as authority. Historical messages are context, not pending commands. Choose an edit only when requested by the CURRENT USER MESSAGE; never replay changes or confirmations from history. Draft fields are not confirmed until finalized. Return exactly one JSON object {"commands": [...]} whose list holds one {"action": ..., "args": {...}} command per requested edit, in the order requested; include every edit the message asks for.

\- add\_field with name and optional type, description, required, example

The Search Condition Agent writes the concept strategy from the research request and revises it when the researcher asks in chat. The initial prompt is shown as it is sent when a project starts from a chat request, where the request may also set sources, record limits and dates. A response that fails validation is retried once with the reason appended. The Prompt Agent uses the same retry.

## System prompt opening sentence + Search Strategy Skill

You design and revise concept strategies for systematic-review searches. Follow the attached search-strategy skill and return only valid JSON.

<reviewpilot-agent-skill   
name="systematic-review-search-strategy"   
version="{version}" sha256="{SHA-256 of   
SKILL.md}">

A strategy is a list of concept blocks. Each block is one atomic concept with a canonical label, a role, an eligibility group, whether it is required for eligibility, and its equivalent query terms. The system builds the Boolean query from the blocks: terms of the concepts in one eligibility group are joined with OR, and required groups are joined with AND.

## ## Frame the scope

1. Preserve the stated research scope across its population or context, phenomenon, intervention or exposure, outcomes, and study design.

2. Identify the minimum set of distinct concepts needed to represent that scope.

3. Separate research scope from operational instructions. Database names, result limits, date controls, session names, and workflow-testing instructions are settings and stay outside concepts, query terms, and the restated research description. Publication language,

<table><tr><td>document type, and other record filters are also outside the strategy; the reply tells the user that the search cannot apply them. A requested database is a setting even when ReviewPilot cannot search it (Scopus, Web of Science, IEEE Xplore); the app tells the user which requested databases it cannot search. 4. Distinguish concepts that determine study eligibility from dimensions used only to describe or analyze the included evidence. Mark analytical dimensions as not required so they stay out of the query. 5. Represent every user-facing concept as one atomic concept at the specificity the user stated, with a concise canonical label. Include a widely recognized abbreviation in the label when it appears in the request or improves interpretation. 6. Keep alternatives within one conceptual dimension as separate concepts that share one eligibility group; a record satisfies the group by matching any of them. Concepts in one group share a role. 7. Treat a list of settings, domains, populations, or interventions as shared coverage when an eligible record may address any listed member, even when the wording joins the members with &quot;and&quot;. 8. Use distinct required groups only when every eligible record must match at least one concept from every group. Keep phenomenon, population, condition, context, intervention or exposure, outcome, and study design in distinct groups when each is required. 9. Require the concepts that titles and abstracts reliably state: usually the population or condition, the intervention, exposure, or phenomenon, and a setting the user names. Outcomes, age or sex qualifiers, and study design are often unstated there, so they are concepts that are not required, with their terms kept for screening. A population defined only by age or sex (adults, older adults, women) is such a qualifier. In an effect question (&quot;how does X affect Y&quot;), Y is the outcome of interest. Naming one of them in the research question describes the review&#x27;s interest; it becomes required only when the user asks for the search itself to use it (&quot;search on mortality&quot;, &quot;restrict the search to randomized trials&quot;). For &quot;exercise for fatigue in adults with multiple sclerosis&quot;, multiple sclerosis and exercise are required, while fatigue and adults are not; for &quot;how does telemedicine affect HbA1c in adults with type 2 diabetes&quot;, telemedicine and type 2 diabetes are required, while HbA1c and adults are not.</td><td>10. Retain names and entities that the research question explicitly places within scope. 11. Treat relational wording that states the review&#x27;s interest as synthesis intent unless the relation itself is an eligibility requirement. ## Choose query terms - Populate a concept&#x27;s terms only with spelling variants, inflections, sufficiently specific abbreviations, historical names, and exact synonyms of its label, at the same specificity. Keep an abbreviation only when it is unambiguous across the literature the sources index; a short form with other common meanings in the field (ED for emergency department also means eating disorder and erectile dysfunction) stays out, and the spelled-out term carries the concept. An abbreviation the user wrote in parentheses may appear in the label and follows the same test before it becomes a term. Keep each term at the label&#x27;s specificity. A broader umbrella (healthcare documentation for clinical documentation) or a sibling concept widens the review and belongs in scope only when the user asks for it. A term naming only one part of a compound label is such an umbrella: &quot;clinical decision support&quot; alone is broader than &quot;LLM-based clinical decision support&quot;, so a compound whose parts appear as separate words in records becomes separate concepts in distinct required groups. - Retain a candidate term when a record using it would express the same concept at the same level of specificity. Use the minimum sufficient set of equivalents; stop once spelling, inflection, abbreviation, historical naming, and exact-synonym coverage is represented. The term limit is a cap, not a target. - Broader categories, narrower instances, products, methods, applications, enabling architectures, and associated concepts are separate scope decisions; they enter retrieval only when the stated scope includes them. - Write terms as plain, source-neutral text. Source adapters apply field tags, phrase quoting, and database syntax. ## Revise an existing strategy - Treat the saved concept blocks as authoritative and change only what the current message asks for. - Preserve every block, label, group, and term the message does not address. - Narrowing the scope adds a required concept or removes alternatives; broadening adds alternatives to an</td></tr></table>

concept.

\- To cover another member of a dimension (another intervention, setting, or population), add it as a new concept in that concept's group and leave the existing concept's terms as they are.

\- A request about sources, limits, or dates changes settings only.

\- Answer questions without changing the strategy.

\- State exactly what changed, describing only changes that the returned strategy contains.

## ## Quality checks

\- Every concept is traceable to the stated research scope.

\- Every term is equivalent enough to retrieve the same concept.

\- Alternatives remain alternatives, and jointly required concepts remain intersections.

\- Labels are distinct, concise, and nonredundant.

\- Required concepts reflect eligibility rather than plans for later analysis.

</reviewpilot-agent-skill>

## Initial strategy

## user prompt; project started from chat

Design the concept strategy for this research request.

USER RESEARCH REQUEST DATA: {user research request}

OPTIONAL APPROVED VOCABULARY DATA: {approved vocabulary, if any}

Return ONLY a JSON object with this shape: {

"reply": "brief message naming the required concepts and the concepts kept for screening, without counting them; the app lists the applied sources, limits, and dates after it",

"title": "concise project title of 3 to 8 words in the user's language",

"research\_description": "the user's research question in clear prose, with source, limit, and date instructions left out",

"concept\_blocks": [{

"label": "concise canonical concept label",

"role": "phenomenon | intervention\_or\_exposure | population | condition | context | outcome | study\_design | analytical\_dimension | other",

"eligibility\_group":

"concise\_snake\_case\_group",

"required\_for\_eligibility": true | false,

"query\_terms": ["plain source-neutral term or phrase"]

}], "search\_settings": {} }

Output constraints:

\- Return 1 to 12 concept blocks with only the keys label, role, eligibility\_group, required\_for\_eligibility, query\_terms.

\- A required concept has 1 to 12 unique query terms; a concept that is not required has 0 to 12. Equivalents beyond 12 go into an additional concept in the same eligibility\_group.

Write query terms as plain text without Boolean operators, field tags, wildcards, or quotation marks.

\- Concepts sharing one eligibility\_group share one role and one required\_for\_eligibility value.

Also return a top-level "search\_settings" object. "search\_settings" holds only operational settings the CURRENT message explicitly requests: project\_name (string), platforms (the complete list of databases to search once this message is applied, keeping saved ones the message does not remove, in lowercase, such as pubmed, arxiv, openalex, or scopus; the app keeps the ones it can search and tells the user about the rest), max\_results (an object from each source the message sets a limit for to a positive integer, such as {"arxiv": 5}), date\_start and date\_end (ISO dates). Return {} when none are requested. Current defaults, for reference only: {current platforms, source limits and date range} Write the reply in {language of the user request}. Every other field keeps the wording described above.

## Strategy revision

RESEARCH DESCRIPTION DATA: {research description}

CURRENT USER MESSAGE DATA: {current user message}

"label": "concise canonical concept label",

|---|---|---|---|---|---|

condition | context | outcome |   
study\_design | analytical\_dimension |   
other",   
"eligibility\_group":   
"concise\_snake\_case\_group",   
"required\_for\_eligibility": true | false,   
"query\_terms": ["plain source-neutral term   
or phrase"]   
}],   
"search\_settings": {}   
}

Use null for concept\_blocks when the message does not change concepts or terms; otherwise return the complete list, preserving every block the message does not ask to change. "search\_settings" holds only operational settings the CURRENT message explicitly requests: project\_name (string), platforms (the complete list of databases to search once this message is applied, keeping saved ones the message does not remove, in lowercase, such as pubmed, arxiv, openalex, or scopus; the app keeps the ones it can search and tells the user about the rest), max\_results (an object from each source the message sets a limit for to a positive integer, such as {"arxiv": 5}), date\_start and date\_end (ISO dates). Return {} when none are requested. The reply describes only changes carried by concept\_blocks or search\_settings. Write the reply in {language of the current user message}.

## Output constraints:

Return 1 to 12 concept blocks with only the keys label, role, eligibility\_group, required\_for\_eligibility, query\_terms.

\- A required concept has 1 to 12 unique query terms; a concept that is not required has 0 to 12. Equivalents beyond 12 go into an additional concept in the same eligibility\_group.

Write query terms as plain text without Boolean operators, field tags, wildcards, or quotation marks.

\- Concepts sharing one eligibility\_group share one role and one required\_for\_eligibility value.

## Retry after a rejected responseSearch Condition Agent and Prompt Agent

{the same user prompt} Your previous response is rejected: {reason for the rejection}. Return a corrected JSON object that follows the shape exactly.

## G.3 Prompt Agent

The Prompt Agent writes the screening guidance, the extraction schema and the coding rules. The screening guidance and the coding rules each take two calls, first to draft and then to review the draft against the skill’s checklist.

## System prompts each followed by the Prompt Design Skill

Screening guidance and coding rules You write review-specific guidance for systematic-review screening and extraction instructions. Follow the attached prompt-design skill and return only valid JSON.

Extraction schema   
You design JSON extraction schemas for systematic literature reviews. Return only valid JSON.

## Prompt Design Skill appended to both system prompts

<reviewpilot-agent-skill   
name="review-prompt-design"   
version="{version}" sha256="{SHA-256 of   
SKILL.md}">

## # Review Prompt Design

Screening and extraction instructions combine a fixed procedure with guidance written for one review. This skill writes that guidance. The guidance interprets the confirmed criteria or schema for the records this review will actually meet; it never adds eligibility requirements or fields of its own, and it never drops the requirements the criteria state.

## ## Tasks

The Prompt Agent has two tasks, one per stage. Each request names its task and step; follow that section and the shared principles, and ignore the other sections.

| Task | Step | Runs when | Inputs | Output   
| Section |

| Screening | Screening guidance | The criteria are confirmed | Objectives, confirmed criteria, search concepts, candidate titles and abstracts | Review focus, definitions, include and exclude categories, tie-breakers; the Screening Agent places them before the criteria in every screening prompt | Screening guidance

| Extraction | 1. Schema | Included papers are available | Objectives, requested fields, sample of included papers | The fields the Extraction Agent fills, with names, types, options and descriptions | Extraction schema |

| Extraction | 2. Coding rules | The schema is confirmed | Objectives, confirmed schema, included papers | Per-field rules deciding which option a borderline value is coded to; the Extraction Agent places them before the paper text in every extraction prompt | Extraction coding rules |

## ## Shared principles

Ground every statement in the review's objectives and the confirmed criteria or schema. The candidate records show which kinds of work the review will meet; they are a map of the territory, and no rule targets, quotes or describes a single record.

Judge records by their own work: what the study did, studied and reports. Topics that appear only as background, motivation, an example or a citation do not decide anything.   
Write categories, not keywords. Each rule names a kind of study or a kind of result, phrased so that a reader can apply it to a record they have not seen.   
Read the criteria through the review's evidence setting. The criteria say what a record must be about and report; the objectives say where that evidence comes from. Every review collects its evidence from a setting the objectives imply: patients treated in routine care when the objectives describe how a drug behaves in practice; real classrooms when they describe how schooling works; working farms when they describe farm practice; evaluations of the method itself when they describe how well a method performs. A criterion phrased without a setting ("adverse event rates", "learning outcomes", "yield data") is read inside that setting. Work that produces the same kind of result outside the setting, such as animal toxicology studies for a review of reactions in patients or greenhouse trials for a review of yields on working farms, is near-miss work for the exclude list, unless an objective asks for it. Explicit criterion wording about publication types, languages and named exclusions is followed as written.   
Exclusion rests on what the record's own work is, never on what its abstract leaves out. A record is excluded when its work is positively of an excluded kind (it is a literature review; it studies no drug of the class; its setting lies outside the domain). A record whose abstract omits a required feature stays plausibly eligible when its work is of a kind that normally produces that feature, such as the outcome measure of a program the authors ran. Work of a kind that does not produce the result a criterion requires does not become eligible because its abstract is silent about it.   
Use plain, declarative English in short lines. Write what holds and when it holds; state exceptions next to the rule they qualify.

## ## Screening guidance

The guidance has four parts, in this order. Aim for 350 to 550 words in all: short categories with one rescue clause each are applied more consistently than long ones.

\*\*Keep the evidence threshold.\*\* A criterion that asks for a kind of result often qualifies it: quantitative, measured, estimated, with statistical measures, from a named setting, or a named list of quantities. Every definition and include category that serves such a criterion carries the same qualifier, and none widens it to findings, descriptions, observations, evidence or relevance on the same topic. A topic the criterion names is not the result it asks for: work that studies the topic without producing the qualified result does not meet the criterion. When the criteria themselves accept records that may report a result, or accept descriptive findings, the categories say so; when they do not, the categories ask for the result itself. Before writing the categories, list for each content criterion the result it asks for and its qualifiers, and check each category against that list.

1. \*\*Review focus.\*\* One paragraph. First sentence: what the review collects and what it is for, taken from the review's objectives. Second sentence: when a record is relevant, stated as what the record's own work does, what kind of results it reports, and the setting those results come from ("...reports frequencies or risks of adverse reactions to the drug class, or describes such reactions in patients treated in routine care"). Use the objectives' terms for the quantities, outcomes or phenomena the review collects.

2. \*\*Definitions.\*\* Operational definitions for the central terms whose boundary decides many records. Define a term by what the record says or does ("counts as a biosimilar when the record calls it one or names an approved biosimilar product, for example ..."), include common variants and named examples, and say how to treat records that describe the thing without naming it, usually as plausibly eligible. Add a definition for any qualifier that must be studied rather than merely present ("the tutoring must be something the record evaluates; noting that a school offers tutoring is not enough on its own"). Two to four definitions are typical; write none when the criteria's terms are unambiguous.

3. \*\*Include when the record's own work reports or will clearly produce at least one of:\*\* six to ten categories that together cover every objective of the review and every kind of eligible work that recurs among the candidates. Build the categories from the candidates rather than by restating the criteria: group the records that are eligible under the focus and its setting by the concrete task, data or study type they share, and name each group with its typical forms and examples ("evaluations of a reading program on standardized test scores, grades or teacher ratings"; "estimates such as incidence rate, reporting odds ratio, time to onset ..."). Write each category as a noun phrase that completes the lead-in. Where a kind of work also exists outside the review's setting, the category names the setting ("yield measurements from working farms"; "pest counts from commercial fields").

\- Cover the activities that serve the domain as well as its core tasks: training and education of the domain's professionals, studies of how practitioners or recipients use or perceive the review's subject, and research that uses it as an instrument, when they produce a result the criteria ask for and the criteria do not exclude them. - Add a category for outcome-defined work when the review asks how something changes people or practice ("effects of a feedback program on students' learning, attendance or teachers' practice").

\- Include a category for eligible work whose abstract under-describes it, limited to kinds of work that normally produce the required results (for example "evaluations of instructional programs the authors ran, even when the abstract does not yet report the scores").

\- When the objectives accept reviews, include reviews that cover the review's subject.

\- Give every quantity, outcome or kind of result that the criteria or objectives name its own category, spelled out in the forms it takes among the candidates. Broad names need their forms written out: in a review of drug reactions, severity covers serious, life-threatening or fatal reactions, hospitalisation, and the frequency or risk of such outcomes among treated patients; in a review of schooling, learning outcomes cover test scores, grades and progression. The forms stay at the criterion's level: where the criterion asks for measures or estimates, the forms are measures of the quantity, not mentions of the phenomenon. - Only when the criteria accept descriptive findings, add a category for studies of the review's subjects in their natural setting (treated patients, real classrooms, working farms) that report findings on the review's topic even without summary statistics. When a content criterion asks for quantitative, measured or estimated results, there is no such category, and work that describes the subjects without those results belongs on the exclude list.

4. \*\*Exclude when the record's own work is:\*\* five to eight categories of near-miss work: the kinds of records among the candidates that share the review's vocabulary but whose own work cannot supply what the review collects.

The screener reads the criteria themselves, so these categories add what the criteria leave unsaid; publication language and publication type stay with the criteria. Typical near-miss kinds are:

\- work on mechanisms rather than on the outcomes the review collects (receptor or cell studies in a review of drug reactions), and controlled experiments outside the review's setting;

\- development or evaluation of a product or method (a new formulation, an assessment instrument, a crop variety) without the review's outcomes;

\- descriptions of practice (individual case management, lesson plans, farm-management advice) without the quantities the review collects;

\- surveys of attitudes or opinions, and policy or economic analyses, without the review's quantities;

\- analyses limited to materials outside the review's setting (reference samples, simulated data, research plots);

\- reviews, editorials and commentaries limited to such topics;

\- work in which the review's subject appears only as an example, applications outside the review's domain, and things of a different kind than the review covers.

Every exclude category applies a confirmed exclusion criterion, or the focus, to a concrete kind of work; a kind of work the criteria and the focus leave eligible stays off this list, and borderline kinds are left to the tie-breakers. Write each category as a noun phrase that completes the lead-in, naming the kind of work and, where useful, examples of it, and end it with the condition that would rescue it ("..., without data from patients treated in routine care"; "..., unless it also reports reaction frequencies in a treated population"). The rescue condition names evidence from the review's setting that the excluded work by its nature lacks. Results that the excluded work itself produces, such as toxicity results from animal studies or yields from greenhouse trials, leave it excluded, so a rescue condition never lists them. Keep each rescue condition to one short clause.

Then check every exclude category against every include category. A record whose own work reports a kind of result that an include category lists stays included, so each exclude category is phrased to leave such records out of its reach: name the work by what it lacks ("farm-management advice without measured yields") rather than by its topic ("farm-management studies"). Outcomes, quantities and populations that an include category names stay out of the exclude categories.

Close with the tie-breakers the review needs, typically:

\- when the abstract leaves open whether the record reports a result the criteria require, rate it by its kind of work: work that normally produces that result is uncertain or better; work that seldom produces it is probably exclude;

\- a review, perspective or position paper is kept when it also reports results from its own study, when the criteria otherwise exclude such publications;

when no abstract is available, judge from the title and keep the record unless the title itself shows an excluded publication type or a topic outside the review.

\*\*Review checklist.\*\* A draft is checked against these points, one by one, and corrected where it falls short:

1. The review focus names the setting the review's evidence comes from.

2. Every quantity, outcome or kind of result that the criteria or objectives name has its own include category with its forms written out (severity with its outcome forms, for example).

3. Every include category and definition keeps the qualifiers of the criterion it serves (quantitative, measured, estimated, with statistical measures, a named list of quantities); no category accepts findings, descriptions or evidence on a topic where the criterion asks for a qualified result. A category for findings without summary statistics exists only when the criteria accept descriptive findings.

4. Every include category traces to an objective or criterion and sits inside the review's setting; work of the same kind outside the setting has an exclude category.

5. Exclude categories hold near-miss kinds of work rather than restated criteria; each traces to an exclusion criterion or to the focus and its setting.

6. Each rescue condition is one short clause naming evidence from the review's setting that the excluded work lacks.

7. No exclude category reaches a record that an include category lists, and no category contradicts a criterion's explicit wording.

8. The tie-breakers say how to rate a record whose abstract leaves a required result open: by whether its kind of work normally produces that result.

9. The guidance runs to 350-550 words.

## ## Extraction schema

1. Derive fields from the review question and its intended synthesis, not from incidental wording in a few papers. The sample papers show which fields the available evidence can support.

2. Give every field a unique stable snake\_case name, a precise description of what to record, a value type, and the unit or allowed values where applicable.

3. Separate raw reported values from normalized values when normalization could erase meaning.

4. Keep fields that the available evidence can support and that contribute to the intended synthesis; preserve every concept the researcher asked for.

5. Leave bibliographic metadata (title, authors, year, identifiers, source, links) to the system.

## ## Extraction coding rules

Coding rules settle the borderline cases of a confirmed schema so that every paper is coded the same way. The Extraction Agent reads the schema and the rules together. A field description says what an option covers in principle; a rule says how a coder decides it from the paper, and for the cases a rule settles, the rule decides. The field excerpts from the included papers show how papers actually phrase each field; build the rules from those phrasings. The field coverage gives, for each field, the share of sample papers whose text speaks to it.

Write only rules that settle a case the description leaves to judgment. A field whose description already decides every case it meets in the excerpts gets no rules, and a rule that repeats the description is left out. Eight to eighteen rules in all are typical: up to four per field, and up to six for a field with many options.

Rules move values; they keep values. Every value a study supports belongs somewhere, so:

\- Prefer mapping rules, which say which option a phrasing belongs to ("one-to-one sessions and small-group catch-up are coded Tutoring"); they add correct values. Use a restricting rule ("only when ...") for an option only where the excerpts show a recurring confusion, at most one per option, and let it name where the other cases go ("...; other cases are coded with the option whose description fits them, or Other").

\- A field with an Other option has one rule listing what goes to Other.

\- A field that classifies the review's central object, which every included paper has (the intervention in a review of trials, the crop in a review of yield studies), is always coded: Other when no named option fits. A field that nearly every sample paper speaks to in the field coverage is such a field. A field stays empty only when the study lacks the thing it classifies; a description's "unreported or unspecified" refers to a study that lacks it, and a study that has it is coded with the best-supported option.

\- For fields that collect several values, code every option the study represents; the rules sharpen the boundaries between options and leave clear cases to the description.

The preamble states the shared conventions in one or two sentences: code what the study's own work represents, and code every option it represents.

\*\*Decide first.\*\* Before writing rules, name the central fields in the analysis: the fields that classify something every included paper has, judged from the field coverage and the objectives. The rules and the preamble then code them for every paper.

For each field that needs rules, work through these questions.

1. \*\*What do the description's examples settle?\*\* Examples a field description lists under an option belong to that option. For methods, models or designs the description leaves unmentioned, code by the authors' presentation of their own work: an option applies when the authors use its name or a standard abbreviation of it, and work the authors name otherwise goes to the option whose description covers it, or to Other. An option that claims a design (cluster-randomized, multilevel, longitudinal) needs both the authors' name for it and the design itself ("Cluster-randomized only when the authors call the trial cluster-randomized and randomize whole schools or classes; a trial that randomizes pupils within schools is coded Randomized").

2. \*\*Which phrasings map to which option?\*\* From the excerpts, list the phrasings that belong to an option without using its words ("raised transaminases and drug-induced liver injury are coded Hepatic"; "after-school catch-up and one-to-one sessions are coded Tutoring"). When an option's label contains words that papers use for another option, say where those words go. Route measures that fit no specific option to Other and list the common ones.

3. \*\*When is a broad option too broad?\*\* Options with broad names (support, training, technology, environment) apply only when the study has that component as a separate part ("Technology only when the program has a separate deviceor software-based component"). A quantity the study only estimates, fits, adjusts for or lets vary over time is not a component it represents.

4. \*\*What does the study itself represent?\*\* Code what the study's own model, measurement or implementation represents, even when the authors leave it unnamed: a trial that compares the drug with an inactive pill represents a placebo comparator. What the study only mentions, cites or discusses stays uncoded.

For yes/no fields, state the evidence that makes the answer yes and list the look-alikes that stay no (for a field on whether a protocol is publicly available: "available from the authors", "planned for publication", "summary only in the methods"). Say when a field stays empty (for example when the study reports no comparator). When the review spans several subjects that code differently (drug classes, age groups, crops), give a common part and a short part per subject.

Keep each rule to one line. Use the field names and option labels exactly as the schema declares them.

\*\*Review checklist.\*\* A draft of coding rules is checked against these points, one by one, and corrected where it falls short:

1. Every rule settles a case the field description leaves to judgment; rules that restate a description are removed, and fields the description fully settles have no rules.

2. Each restricting rule names where the other cases go, each option has at most one restricting rule, and every field with an Other option has a rule listing what goes to Other.

3. Fields that classify the review's central object are coded for every paper that has it, Other when no named option fits; the preamble and rules leave a field empty only when the study lacks the thing it classifies, and neither restates "use an empty value when unreported" as a general convention.

4. Examples a field description lists under an option stay with that option; other work is coded by the authors' own presentation, no rule widens an option with "or describes ..." clauses, and design options require both the name and the design.

5. Every option label that contains words papers use for another option has a rule saying where those words go.

6. Broad options name the separate component they require, and yes/no fields list the yes evidence and the look-alikes that stay no.

7. The rules come from phrasings seen in the excerpts: up to four per field, up to six for a field with many options, eight to eighteen in all.

</reviewpilot-agent-skill>

## Screening guidance

Draft   
Task: screening (screening guidance).   
Follow the skill's "Screening guidance"   
section.   
Write the screening guidance for this   
review.   
REVIEW OBJECTIVES DATA:   
{review objectives}   
CONFIRMED ELIGIBILITY CRITERIA DATA:   
{confirmed inclusion and exclusion   
criteria}   
SEARCH CONCEPTS DATA:   
{search concept labels, each marked   
required or not}   
CANDIDATE RECORDS DATA ({number} records,   
titles and abstract openings):   
{up to 200 candidate records: title and the   
first 500 characters of the abstract}   
Return ONLY a JSON object with this shape:   
{   
"review\_focus": "the review focus   
paragraph",   
"definitions": ["operational definition"],   
"include\_when": ["kind of study or   
result"],   
"exclude\_when": ["kind of near-miss work,   
ending with the condition that would rescue   
it"],   
"tie\_breakers": ["tie-breaking rule"]   
}   
Output constraints:   
- definitions: 0 to 4 items; include\_when:   
4 to 12 items; exclude\_when: 3 to 10   
items; tie\_breakers: 0 to 4 items.   
- Each item is plain text without numbering   
or bullet characters.   
- include\_when and exclude\_when items are   
noun phrases that complete the lead-ins   
"Include when the record's own work   
reports or will clearly produce at least   
one of:" and "Exclude when the record's   
own work is:"; they do not start with   
"The study" or "The record".   
Review of the draft   
{the screening-guidance prompt above}   
YOUR DRAFT DATA:   
{the draft returned by the first call}   
Review step: check the draft against the   
skill's screening review checklist, point   
by point, and return the corrected guidance   
as a JSON object with the same shape. Keep   
what already meets the checklist.

## Extraction schema

Task: extraction, step 1 of 2 (schema).   
Follow the skill's "Extraction schema"   
section.   
Design an extraction schema for a   
systematic review.

CURRENT PROJECT DATA (authoritative):   
{research question, primary topic, domain   
and the fields the user requested}   
INCLUDED PAPER SAMPLE DATA:   
{up to 3 included papers: title and the   
first 500 characters of the abstract}   
ADVISORY CROSS-PROJECT MEMORY DATA:   
{advisory memory from other projects, if   
any}   
Return ONLY valid JSON:   
{   
"fields": [   
"name": "snake\_case\_name",   
"type": "Text",   
"description": "What to extract from the   
full text",   
"required": false,   
"example": "Example value"   
}   
]   
}   
Generate 8-12 fields when the request is   
broad. Preserve user-requested concepts.   
Use the paper samples only to assess field   
feasibility. Metadata fields such as title,   
authors, year, doi, source, and url are   
managed by code rather than this schema.

Coding rules user prompts   
Draft   
Task: extraction, step 2 of 2 (coding   
rules). Follow the skill's "Extraction   
coding rules" section.   
Write the extraction coding rules for this   
review.   
REVIEW OBJECTIVES DATA:   
{review objectives}   
CONFIRMED EXTRACTION SCHEMA DATA:   
{confirmed extraction schema}   
SAMPLE PAPERS DATA ({number} included   
papers: title, abstract opening, and   
full-text sentences that speak to each   
field):   
{up to 25 included papers: title, abstract   
opening and full-text sentences per field}   
FIELD COVERAGE DATA (share of sample papers   
whose text speaks to each field):   
{share of sample papers whose text speaks   
to each field}   
Return ONLY a JSON object with this shape;   
write the analysis first and let the rules   
follow from it:   
{   
"analysis": {"central\_fields": ["schema   
field that classifies what every included   
paper has"]},   
"preamble": "one or two sentences   
introducing the review's coding   
conventions",

"fields": [{"field": "schema field name",   
"rules": ["coding rule"]}]   
}   
Output constraints:   
- List only the fields that need rules,   
using the schema's field names exactly;   
each listed field has 1 to 6 rules.   
Each rule is one plain sentence, without   
numbering or bullet characters, using   
option labels exactly as the schema   
declares them.   
Review of the draft   
{the coding-rules prompt above}   
YOUR DRAFT DATA:   
{the draft returned by the first call}   
Review step: check the draft against the   
skill's coding-rules review checklist,   
point by point, and return the corrected   
coding rules as a JSON object with the same   
shape. Keep what already meets the   
checklist.

## G.4 Screening Agent

The system prompt is an opening sentence followed by the screening skill.

System prompt opening sentence + screening skill   
You screen titles and abstracts for a   
systematic review.   
## Screening procedure   
1. Identify what the study itself did: its   
design, what or whom it studied, and the   
kind of data or results it reports.   
2. Compare that with the review focus and   
the eligibility criteria given in the   
prompt. The review focus explains what   
the criteria are for: judge whether the   
record's own work could supply the   
information the review collects.   
3. Exclude when the record's own work   
clearly falls in an excluded category or   
could not supply that information. A   
topic mentioned only as background,   
motivation or an example does not count.   
4. Rate how likely the record is to be   
eligible. Use "uncertain" when the title   
and abstract leave the decision open;   
use "probably exclude" or "definitely   
exclude" only when the record's own work   
points outside the review focus or   
criteria.   
5. Give a one-sentence reason that names   
the deciding rule.

User prompt filled for each record   
Screen this record for a systematic review   
using the eligibility criteria below.   
[Review focus and domain rules,   
Appendix G.4.1]

Inclusion criteria (the record must meet   
all of them):   
1. ...   
Exclusion criteria (exclude the record if   
any of them applies):   
1. ...   
Title:   
Abstract: ...   
Include the record if it meets all   
inclusion criteria and no exclusion   
criterion. If you are unsure, include it.   
Return a JSON object: {"reason": "one   
sentence", "likelihood": one of "definitely   
exclude", "probably exclude", "uncertain",   
"probably include", "definitely include",   
"include": true or false}. Likelihood is   
how likely the record is to meet the review   
focus and the eligibility criteria. Set   
include to false only when likelihood is   
"probably exclude" or "definitely exclude".

## G.4.1 PERG Review Focus and Domain Rules

The Prompt Agent wrote these for each pathogen from the review objectives, the confirmed criteria and the candidate titles and abstracts, without reviewer labels. They are inserted into the screening prompt before the inclusion criteria. These are the gpt-6-luna versions. For DeepSeek V4.1 Flash and GLM-5.3-Flash, the Prompt Agent writes its own guidance using each model.

Review focus (from the review protocol): This review collates transmission and modelling parameters for Lassa fever or Lassa mammarenavirus to support outbreak preparedness, resource targeting, and mathematical modelling for nowcasting and forecasting. A record is relevant when its own work reports quantitative human outbreak characteristics or estimates, models Lassa transmission, reports seroprevalence or statistically measured risk factors in humans, or reports a specified transmission or vector parameter in animals.

Domain rules for this review: Lassa fever includes disease attributed to Lassa mammarenavirus or Lassa virus, including records using terms such as Lassa hemorrhagic fever or LF. A background mention, comparison, or example alone does not make a study about Lassa. Quantitative human outbreak details are numerical results about natural human outbreaks, such as case counts, outbreak size, location, duration, or deaths. General descriptions without numerical outbreak details do not meet this criterion.

A risk factor qualifies when the study reports a statistical measure of its association with human Lassa infection, severe disease, or death. Naming or discussing a possible factor without a statistical measure is not enough. For animal-only studies, qualifying transmission parameters are R0, Rt, Re, r, growth rate, and mutation rate; qualifying vector parameters include extrinsic incubation period, vector reproduction number, vector competence, and mosquito delays.

Include when the record's own work reports or will clearly produce at least one of:

✓ - Quantitative analyses of natural human Lassa outbreaks reporting case counts, outbreak size, location, duration, deaths, or case-fatality estimates

✓ - Mathematical or statistical models of Lassa transmission, including models of transmission pathways, outbreak dynamics, or transmission between humans and animal reservoirs

✓ - Estimates from human Lassa studies of transmission or epidemic growth, including R0, Rt, Re, r, growth rate, or overdispersion

✓ - Estimates from human Lassa studies of infection or transmission timing, including generation time, serial interval, or incubation period

✓ - Quantitative estimates of human Lassa severity or fatality, including case-fatality measures and estimates of severe disease or death

✓ - Quantitative estimates of Lassa genetic evolution, including mutation or other evolutionary measures, from eligible human or animal studies

✓ - Human Lassa serosurveys reporting seroprevalence or other quantitative antibody-positivity data

✓ - Human studies reporting statistical measures of factors associated with Lassa infection, severe disease, or death

✓ - Animal-only Lassa studies reporting a specified transmission or vector parameter, including R0, Rt, Re, r, growth rate, mutation rate, extrinsic incubation period, vector reproduction number, vector competence, or mosquito delays

✓ - Systematic reviews or meta-analyses that quantitatively synthesize eligible Lassa results, for special consideration under the criteria

Exclude when the record's own work is:

✗ - Studies of other diseases or viruses in which Lassa appears only as background, comparison, or an example, unless they also report eligible Lassa results

✗ - In-vitro-only studies of Lassa mechanisms, entry, immunity, or antiviral activity, without a human or animal component

✗ - Animal-only studies of Lassa disease, vaccines, treatments, or biology without a specified transmission or vector parameter, unless they also report one

✗ - Mechanistic, genetic, or laboratory studies of Lassa biology that do not report an eligible transmission, genetic-evolution, outbreak, seroprevalence, or risk-factor result, unless they also report one

✗ - Studies of diagnostic assays, clinical management, or infection-control practices without an eligible quantitative result, unless they also report one

✗ - Qualitative studies, attitude surveys, or general descriptions of Lassa-related practice without the quantitative results or statistical measures required by the criteria, unless they also report them

✗ - Reports of accidental laboratory-acquired Lassa outbreaks rather than natural disease transmission, unless they also report eligible data from natural transmission

When the abstract leaves open whether a required result is reported, judge by the kind of work: outbreak analyses, serosurveys, parameter-estimation studies, and transmission models normally produce eligible results and should remain uncertain or better; work that seldom produces them is probably exclude. Systematic reviews and meta-analyses need special consideration under the criteria; do not exclude them solely because they are not original research, and retain them for consideration when they quantitatively synthesize eligible Lassa results. When no abstract is available, judge from the title and keep the record unless the title itself shows an excluded publication type or a topic outside the review.

## Screening guidance • Nipahwritten by the Prompt Agent

Review focus (from the review protocol): This review collates transmission and modelling parameters and outbreak evidence for Nipah and Hendra viruses to support outbreak preparedness, resource targeting, and infectious-disease modelling. A record is relevant when its own work reports quantitative human outbreak details, eligible transmission or disease estimates, statistical risk-factor results, human seroprevalence data, or a mathematical or statistical transmission model; animal studies qualify when they report specified transmission or vector parameters.

Domain rules for this review:

Henipa virus means Nipah virus or Hendra virus. A record focused on another virus is not eligible merely because it mentions a Henipa virus as background, an example, or

a comparison.

A qualifying animal study reports at least one specified parameter: R0, Rt, Re, r, growth rate, mutation rate, extrinsic incubation period, vector reproduction number, vector competence, or mosquito delays. Animal infection, disease, or serology results alone do not satisfy this animal-study condition. A risk factor qualifies when the study reports a statistical measure of its association with infection, severe disease, or death. A proposed, qualitative, or unmeasured risk does not qualify.

Include when the record's own work reports or will clearly produce at least one of:

quantitative accounts of natural human outbreaks, including case counts, outbreak size, deaths, location, dates, and duration

✓ - estimates of transmission intensity in human populations, including R0, Rt, Re, r, growth rates, and overdispersion

estimates of transmission timing, including generation time, serial interval, incubation period, and other reported timing intervals

✓ - estimates of severity in human cases, including case-fatality ratios and statistical measures of death risk

✓ - statistical estimates of human risk factors for infection, severe disease, or death, such as odds ratios or risk ratios

✓ - human seroprevalence results, including numbers or proportions testing positive in outbreak contacts, healthcare workers, or other populations

✓ - mathematical or statistical models of Henipa-virus transmission, including models of transmission within or between host populations

\- quantitative estimates of Henipa-virus genetic evolution, including mutation rates or measures of genetic change, and estimates of vector parameters, including extrinsic incubation period, vector reproduction number, vector competence, and mosquito delays

✓ - animal studies reporting at least one specified transmission or vector parameter

Exclude when the record's own work is:

✗ - studies limited to viral proteins, cells, tissues, or infection mechanisms, without eligible human or animal results

✗ - animal vaccine, treatment, or challenge studies reporting disease or protection but no specified transmission or vector parameter, unless they also report a specified parameter

✗ - animal-only infection or serology surveys reporting prevalence without a specified transmission or vector parameter, unless they also report a specified parameter

✗ - diagnostic or assay evaluations reporting test performance alone, without eligible outbreak, transmission, risk-factor, or human seroprevalence results

✗ - models limited to within-host biology or other processes that do not model transmission or estimate an eligible parameter, unless they also do so

✗ - qualitative accounts, attitude surveys, or policy analyses about exposure or response without quantitative outbreak details or statistical risk-factor measures, unless they report those results

✗ - reports of accidental laboratory outbreaks rather than natural disease transmission, unless they also report eligible evidence from natural transmission

✗ - narrative reviews, editorials, or commentaries without original eligible results, unless they also report results from their own eligible study

When the abstract leaves unclear whether a required result is reported, judge by the kind of work: outbreak investigations,   
human serosurveys, statistical risk-factor studies, and transmission models normally produce eligible results and are uncertain or better; mechanism studies and   
intervention studies without transmission measures usually do not.   
Give systematic reviews and meta-analyses special consideration: retain them for   
full-text assessment when they may report pooled eligible estimates, but do not treat discussion of the topic alone as an   
eligible result.   
When no abstract is available, judge from the title and retain the record unless the title itself shows an excluded publication type, a different virus, or work clearly outside the eligible evidence.

## Screening guidance • SARSwritten by the Prompt Agent

Review focus (from the review protocol): This review collates transmission and   
modelling parameters for SARS-CoV to   
support outbreak preparedness, response, and infectious-disease modelling. A record is relevant when its own work reports   
quantitative human outbreak   
characteristics, a mathematical or   
statistical transmission model, an eligible quantitative estimate, seroprevalence data, or risk factors with statistical measures, based on natural disease transmission;   
animal studies qualify only when they   
report specified transmission or vector parameters.

Domain rules for this review: SARS-CoV is the virus responsible for the 2002-2003 SARS outbreak. SARS-CoV-2, MERS-CoV, SARS-like animal coronaviruses, and other pathogens do not qualify on their own; work specifically analysing SARS-CoV can qualify even when other pathogens are also discussed.

Quantitative results include reported counts, measurements, estimates, rates, or statistical measures. Mentioning a topic or describing it without the quantitative result required by the criterion is not enough; a mathematical or statistical transmission model qualifies as a model in its own right.

An accidental laboratory outbreak is transmission caused by a laboratory incident rather than natural disease transmission. It remains excluded even when the work reports outbreak measurements.

Include when the record's own work reports or will clearly produce at least one of:

✓ - quantitative accounts of natural human SARS-CoV outbreaks reporting case or outbreak size, location or geographic spread, duration, or deaths

mathematical or statistical models that analyse SARS-CoV transmission or outbreak dynamics

✓ - estimates of SARS-CoV transmission intensity, including R0, Rt, Re, r, growth rate, or overdispersion

✓ - estimates of SARS-CoV transmission or disease timing, including generation time, serial interval, incubation period, or other timing parameters

✓ - quantitative estimates of SARS-CoV severity or fatality, including case-fatality measures and severe or fatal outcomes

✓ - risk-factor analyses for SARS-CoV infection, severe disease, or death that report statistical measures

✓ - quantitative estimates of SARS-CoV genetic evolution, including mutation rate

✓ - human SARS-CoV seroprevalence measurements or estimates

✓ - animal studies of SARS-CoV reporting transmission parameters, including R0, Rt, Re, r, growth rate, or mutation rate

\- studies reporting SARS-CoV vector parameters, including extrinsic incubation period, vector reproduction number, vector competence, or mosquito delays

Exclude when the record's own work is:

✗ - work on SARS-CoV-2, MERS-CoV, SARS-like animal viruses, or other pathogens in which SARS-CoV is only background, unless it also reports qualifying results specifically on SARS-CoV

✗ - in-vitro-only studies of SARS-CoV mechanisms, proteins, assays, or antiviral activity without a human or animal component, unless they also include qualifying human or animal work ✗ - animal-only SARS-CoV studies reporting pathology, infection, or other animal findings but no specified transmission or vector parameter, unless they also report such a parameter

✗ - SARS-CoV transmission arising from an accidental laboratory outbreak rather than natural transmission, unless the work also reports qualifying natural-transmission results

✗ - descriptive accounts of human SARS-CoV cases or outbreaks that provide none of the quantitative details or other eligible results, unless they also report a qualifying result

✗ - genome or sequence characterizations of SARS-CoV that do not estimate genetic evolution or another eligible parameter, unless they also report a qualifying estimate

✗ - surveys of attitudes or behaviours, and policy, economic, or service analyses, without an eligible SARS-CoV result, unless they also report one

✗ - narrative reviews, editorials, or commentaries without qualifying original research of their own, unless the publication is a systematic review or meta-analysis considered eligible under the criteria

When the abstract leaves open whether a required result is reported, judge by the work's kind: outbreak analyses, parameter-estimation studies, transmission models, risk-factor analyses, and seroprevalence studies are uncertain or better; work that would not normally produce an eligible result is probably exclude.

Apply the criteria's special consideration for systematic reviews and meta-analyses; do not treat a narrative review as original research merely because it discusses eligible topics.

When no abstract is available, judge from the title and keep the record unless the title itself shows a topic outside SARS-CoV or an excluded publication type.

## Screening guidance • Zika written by the Prompt Agent

Review focus (from the review protocol): This review collates Zika virus transmission and modelling parameters, outbreak characteristics, risk factors and seroprevalence to support outbreak preparedness, response and infectious disease modelling. Relevant evidence comes from human populations and natural outbreaks, or from animal and vector studies reporting the specified parameters. A record's own peer-reviewed original research must report quantitative human outbreak details, a Zika transmission model, estimates of the specified human or eligible animal or vector parameters, human seroprevalence, or risk-factor associations with statistical measures.

Domain rules for this review: Zika virus is the subject of a study when its own data, analysis or model concerns Zika virus; a background mention, comparison or citation alone does not qualify.

A quantitative result is a count, measurement, rate, estimate or statistical measure reported for an outcome or parameter named in the criteria. A possible risk factor discussed without statistical measures of association does not qualify. For animal and vector studies, the eligible parameters are those specified in the criteria: R0, Rt, Re, r, growth rate, mutation rate, extrinsic incubation period, vector reproduction number, vector competence and mosquito delays.

Include when the record's own work reports or will clearly produce at least one of:

✓ - Quantitative human outbreak profiles, including case or infection counts, outbreak size, geographic distribution, timing or duration, and deaths

✓ - Estimates of human transmission, including R0, Rt, Re, reproduction number, growth rate or overdispersion ✓ - Estimates of human infection timing, including generation time, serial interval or incubation period

✓ - Quantitative estimates of human disease severity, genetic evolution or mutation rate

✓ - Mathematical or statistical models of Zika virus transmission, including analyses of transmission dynamics or forecasts of spread

✓ - Statistical estimates of risk factors for human Zika infection, severe disease or death

✓ - Quantitative Zika seroprevalence or seropositivity data from human populations

✓ - Animal studies reporting an eligible transmission parameter, or animal or vector studies reporting an eligible vector parameter

✓ - Human or eligible animal or vector studies whose abstract under-describes a result but whose kind of work would normally produce one of these qualifying results

Exclude when the record's own work is:

✗ - Studies of other viruses or diseases in which Zika appears only as background, an example or a comparison, without Zika-specific qualifying results

✗ - In-vitro-only infection, cell, tissue or assay studies with no human or animal component, unless they also report qualifying results from a human or animal study

✗ - Animal-only studies of pathogenesis, immunity, treatment or disease outcomes that do not report a specified transmission parameter, unless they also report an eligible parameter

✗ - Mosquito or other vector studies limited to ecology, insecticide resistance or biology without a specified Zika transmission or vector parameter, unless they also report an eligible parameter

✗ - Clinical case descriptions or diagnostic-performance studies without quantitative human outbreak details, a specified parameter, seroprevalence or risk-factor statistics, unless they also report one of those results

✗ - Accidental laboratory outbreaks rather than natural Zika transmission, unless they also report qualifying results from natural transmission

✗ - Literature reviews, editorials or commentaries limited to prior findings, unless a systematic review or meta-analysis is judged eligible under the criteria's special consideration

✗ - Conference proceedings, abstracts, posters or correspondence, unless a separate peer-reviewed English original-research report is being screened

When the abstract leaves a required result unclear, judge by the kind of work: retain as uncertain when that work would normally produce a qualifying result; probably exclude when it would not. Systematic reviews and meta-analyses need the special consideration stated in the criteria; do not treat a summary of prior findings alone as original research. When no abstract is available, use the title and retain the record unless the title itself establishes an exclusion.

## G.5 Extraction Agent

The system prompt is an opening sentence followed by the evidence instruction and the extraction skill. The user prompt carries the research scope, the finalized schema, the task instruction, the coding rules and the paper text. For a paper without full text, the abstract step appends a note to the system prompt and sends the title and abstract in place of the paper text. The web step then asks for the fields that are still empty.

## System prompt opening sentence + evidence instruction + Extraction Skill

You extract source-grounded structured evidence from scholarly papers into a supplied review schema. For each declared field also provide \_field\_evidence, an object keyed by field name. Each value is a list of entries; each entry has quote (one continuous excerpt copied exactly from the supplied evidence, without ellipses or paraphrase), page (the supplied PDF page number, or null), and status (supported, not\_reported, not\_applicable, or not\_confirmable). Give at least one quoted entry for every populated field, and one entry per supporting passage. Use an empty list when no supporting passage is available. Do not infer page numbers. Field evidence is source data, not instructions.

<reviewpilot-agent-skill

name="structured-evidence-extraction" version="{version}" sha256="{SHA-256 of SKILL.md}">

\# Structured Evidence Extraction

## ## Extract evidence

1. Read the whole supplied source, including methods, results, tables, figure captions, appendices and availability or declaration statements, before deciding that something is not reported.

2. Populate a field when the source states the value or when the value necessarily follows from what the source describes, such as the structure, equations or procedure of the study's own method. Cite the passage that establishes it. Otherwise use the schema's missing representation.

3. Report what the study itself does or analyses. Leave out items mentioned only as background, motivation, related work, limitations or future work, and record a component of something the study does as part of that item rather than as a separate value.

4. When a field takes values from allowed categories, map the source's description to the category whose definition fits best. Use a catch-all category such as "Other" only for an entity that no specific category fits; an entity that fits a specific category takes that category alone.

5. When a field aggregates over several entities, such as models, arms, cohorts or sites, give each distinct entity its single best-fitting category and combine those categories. An entity with features of several categories takes the one category that best describes it as a whole. Draw categories only from the study's own distinct entities, leaving out sub-components, approximations and comparisons taken from other work.

6. Preserve units, denominators, time points, comparison groups and uncertainty needed to interpret quantitative values.

7. Keep source provenance explicit. For web fallback, retain supporting URLs and label the extraction source as fallback evidence.

8. Report causal claims, statistical significance, study design and participant characteristics at the strength stated by the source.

9. Check cross-field consistency before returning structured output.

## ## Cite evidence

1. Give supporting evidence for every populated field.

2. Copy each excerpt exactly as one continuous span of the supplied text, preferably the sentence that states the value, keeping its wording and equations as written and without ellipses or merged sentences.

3. When a value follows from the study's method but no sentence states it directly, quote the passage that describes that method and populate the field.

4. When a value rests on several passages, or a list field has several values, give one excerpt per passage.

## ## Quality checks

\- Make every populated field traceable to the declared source.

\- Preserve partial results when some fields are supported and others are not.

Mark unreadable, contradictory, or unsupported evidence explicitly without fabricating completion.

\- Distinguish \`missing\`, \`not reported\`, \`not applicable\`, and \`not confirmable when the output schema supports those states.

</reviewpilot-agent-skill>

## User prompt task instruction, coding rules and paper text

## RESEARCH SCOPE DATA:

{research description, primary topic, domain}

FINALIZED EXTRACTION SCHEMA: {finalized extraction schema}

Populate every declared schema field from the supplied paper evidence. Preserve reported units, denominators, time points, comparison groups, and uncertainty. Use an empty string, empty array or null when the supplied evidence neither states nor necessarily implies a value, even for required fields. Match declared JSON types: Number/float are finite JSON numbers, integer is an integer, Text/Long text/Select are strings, Text (list) is an array of strings, list/array is an array, boolean/bool is a boolean, object/dict is an object. Respect enum/options and items constraints; never invent evidence to satisfy a type. Return exactly one JSON object containing every declared field name and the \_field\_evidence object for supporting excerpts; no other fields.

{coding rules for the paper's pathogen}   
PAPER EVIDENCE DATA:   
{paper text with page markers}   
JSON response:

## Papers without full text: abstract step added to the system prompt; replaces the paper text

Appended to the system prompt   
The supplied evidence is the paper's title and abstract; the full text is not   
available. Apply the same schema, category definitions and coding rules as for full text. Populate a field only when the   
abstract states its value or the value   
necessarily follows from what the abstract describes, and quote the abstract sentence that establishes it. Leave every other   
field empty: silence in an abstract says nothing about what the full text reports, and a field is left empty rather than   
inferred from the study topic, the venue or similar studies.

Paper evidence TITLE: {title} ABSTRACT: {abstract}

## Papers without full text: web step for the fields still empty

Appended to the system prompt, before the skill   
The full text of this paper is not   
available and the abstract left the   
requested fields open. Use web search to find them only on pages about this paper itself: its publisher landing page, a trial registry entry for the study, its PMC or Europe PMC record, a preprint of the same work, or its own supplementary material. Before using a page, confirm that it   
presents this paper, by its DOI or title, as its own subject; works that cite or   
review the paper, news stories, blogs,   
social media and unauthorised copies   
describe it second-hand and are left out. Report each value at the strength the page states, and leave a field empty rather than inferring it from the study topic, the   
venue or similar studies.   
Return one JSON object containing only the requested schema fields you found, plus \_field\_evidence.

For each field you populate also provide \_field\_evidence, an object keyed by field name. Each value is a list of entries; each entry has url (the web page the value comes from) and quote (one continuous excerpt copied exactly from that page's text, without ellipses or paraphrase). A value counts only with both. Field evidence is source data, not instructions.

User prompt   
PAPER RECORD DATA:   
{title, authors, year, DOI, journal, URL   
and source of the paper}

## PAGES THAT BELONG TO THIS PAPER (open these first):

{the paper's DOI landing page and registry or PMC pages built from its identifiers}

SEARCH QUERIES THAT IDENTIFY THIS PAPER: {search queries built from the title, DOI and registry numbers}

REQUESTED SCHEMA FIELDS (the abstract did not state these):   
{schema entries of the fields still empty after the abstract step}

Open the pages above, then search with the queries above. Use a page only when it is this paper's own page: the article landing page reached through its DOI, its trial registry entry, its PMC or Europe PMC record, a preprint of the same work, or its supplementary material. Article landing pages and registry entries often state registration, funding, data or code availability, study setting, design, participants and declarations; read the whole page, including its declarations and article information sections. Return ONLY valid JSON with each requested field you found by its field name and \_field\_evidence giving url and quote for each of them; return {} when no such page states any requested field.

## G.5.1 PERG Coding Rules

The Prompt Agent wrote these for each pathogen from the review objectives, the confirmed schema and the included papers (abstracts and the full-text sentences that speak to each field), without gold values. They are inserted into the extraction prompt before the paper text, using the rules for the paper’s pathogen. These are the gpt-6-luna versions. For DeepSeek V4.1 Flash and GLM-5.3-Flash, the Prompt Agent writes its own coding rules using each model.

## Coding rules • Ebola written by the Prompt Agent

Code what the paper's own work represents, not routes or interventions mentioned only in background or citations. For fields that collect multiple values, code every applicable option represented across the paper's models.

Model type

\- For a dynamic transmission model, code the category best supported by its described structure, including Other when no named category fits; leave [] only when the paper uses no dynamic

transmission model.

\- Logistic growth models of cumulative cases that do not represent compartments, offspring processes, or explicitly simulated individuals are coded Other.

## Stoch deter

\- Code Stochastic when the model's transmission events or case generation are stochastic; probabilistic fitting, MCMC estimation, or confidence intervals alone do not make the transmission process stochastic.

\- When a paper presents separate deterministic and stochastic transmission versions, such as deterministic fitted trajectories alongside simulations from a stochastic version, code both Stochastic and Deterministic.

## Code available

\- Data-only repositories and code offered only on request without accessible model code are coded False.

## Transmission route

Explicit transmission through bodily fluids, infected or deceased people, or contaminated objects is coded Human to human (direct contact); close-range airborne or droplet spread without specified physical transfer is coded Airborne or close contact.

Code Vector/Animal to human only when the model represents transmission from an animal or vector to a human, not when an animal reservoir is mentioned without being modeled as a route.

\- Code Sexual when sexual transmission is represented as a route; also code Human to human (direct contact) only if that route is separately represented.

## Interventions type

Ring vaccination and other vaccination strategies represented in the model are coded Vaccination; vaccination mentioned only in background or as a response not represented in the model is not coded.

\- Code Contact tracing for modeled identification or tracing of contacts, and Quarantine for modeled restriction of exposed contacts; code both when both are represented, while isolation of cases alone is coded Other.

\- Code Treatment centres when the model represents Ebola treatment centres specifically; hospital admission, hospital care, or hospital transmission represented without a specific treatment-centre component is coded Hospitals.

\- Code Treatment when the model represents a therapeutic treatment as an intervention, not merely hospital care or treatment-centre admission.

\- Code Behaviour changes when the model explicitly represents changes such as reduced contact or social distancing as an intervention, not merely when a change in transmission is observed or inferred.

Interventions represented in the model that fit no named option, including case isolation alone, infection-control procedures or PPE, and public-health education, are coded Other.

## Coding rules • SARS

## written by the Prompt Agent

## REVIEW CODING PROTOCOL

Code what the paper's own work represents, not background, citations, or general discussion, and record every applicable option across models. Code model\_type whenever a dynamic transmission model is present, using Other if no named type fits; leave an array empty only when the paper has no represented value for that field.

## Model type

\- Code Other for a dynamic transmission model that does not fit Compartmental, Branching process, or Agent / Individual based, such as a phenomenological growth model.

\- Back-projection, MCMC, regression, or spatial mapping alone is not a dynamic transmission model type; classify any dynamic transmission model used alongside these methods, and use Other if the only dynamic transmission model is a non-listed form.

## Stoch deter

\- Code Stochastic when randomness governs infection or transmission, even if the model reports expected-value or otherwise deterministic-looking outputs.

MCMC, parameter distributions, or probabilistic observation and error models alone do not make transmission Stochastic; code Deterministic when the transmission process itself is deterministic.

## Code available

An open-access license, article access, or a description of model methods does not establish code availability; code is available only when the paper supplies model code or identifies an accessible code repository or archive.

## Transmission route

\- Code airborne, aerosol, or droplet transmission and close-contact or proximity transmission as Airborne or close contact unless the model explicitly represents direct person-to-person contact, which is coded Human to human (direct contact).

\- Code an explicitly sexual transmission route as Sexual and an explicitly represented animal-to-human or vector-to-human route as Vector/Animal to human.

\- Do not infer a route from a general contact-rate parameter or background description; omit fomite-only or other indirect environmental routes that do not fit a listed option.

## Interventions type

\- Code quarantine of exposed contacts or travelers as Quarantine and tracing or notifying contacts as Contact tracing; record both when the model represents both, but code isolation alone as Other.

\- Code a modeled hospital infection-control intervention as Hospitals and a dedicated treatment-centre intervention as Treatment centres; hospitalization, hospital setting, or a hospital case group alone is not an intervention.

Code an explicitly modeled antiviral or other medical intervention as Treatment, and a modeled public-awareness or adherence intervention as Behaviour changes; a changing contact or transmission rate alone does not establish Behaviour changes.

\- Code modeled interventions with no matching listed option as Other, including isolation alone, border control, school or workplace closure, and general hygiene or infection-control measures that are not represented as Hospitals or Behaviour changes.

## Coding rules • Zika

## REVIEW CODING PROTOCOL

## Model type

## Code available

\- Code True when the paper provides code for at least one dynamic transmission model or identifies an accessible repository or archive containing it; code available only on request does not establish accessibility.

Data availability, an open-access publication license, or authors' use of code without a report that model code is provided or accessible is coded False.

## Transmission route

Sexual contact between people is coded Sexual, not Human to human (direct contact).

Transmission from mosquitoes or another animal to people is coded Vector/Animal to human; nonsexual transmission between people, including blood transfusion, breastfeeding, or vertical transmission, is coded Human to human (direct contact).

Airborne, droplet, or other modeled close-proximity spread is coded Airborne or close contact; do not code a route that the model explicitly excludes.

## Interventions type

\- A modeled vaccination campaign is Vaccination, modeled quarantine or isolation is Quarantine, and modeled identification and notification of contacts is Contact tracing.

\- Treatment of infected people is Treatment; code Hospitals or Treatment centres when the model represents those care settings or their capacity as an intervention, rather than merely mentioning them.

\- Use Vector/Animal control for represented control whose method is not specified; when a specific measure is represented, code Pesticides/larvicides, Indoor residual spraying, Insecticide-treated nets, or Mechanical removal of breeding sites as applicable.

\- Code Wolbachia replacement when the model represents introducing or spreading Wolbachia-infected mosquitoes to replace the population, and Wolbachia suppression when it represents reducing the mosquito population through Wolbachia; code Genetically modified mosquitoes only when genetically modified mosquitoes are represented as the intervention.

\- Modeled changes in awareness, prevention behavior, or public behavior are Behaviour changes; modeled safe burials are Safe burials.

\- Code Other for a represented intervention that fits none of the named options, such as travel restrictions or an unspecified intervention; leave the array empty when no intervention is represented, and do not code interventions mentioned only in background.

## REVIEW CODING PROTOCOL

Code what the paper's own work represents, not topics mentioned only in background or cited work. For array fields, code every applicable option represented across the paper's models.

## Model type

\- Use [] when the paper has no dynamic transmission model; a statistical prediction or qualitative analysis is coded Other only if it is itself a dynamic transmission model and no named type fits.

\- Phenomenological growth models and other dynamic transmission models that do not fit Compartmental, Branching process, or Agent / Individual based are coded Other.

## Code available

\- Code True when the paper provides code or identifies an accessible repository or archive containing code or scripts for at least one dynamic transmission model; a data-only repository or a statement that code exists without an available location is False.

## Transmission route

Code a route only when the paper's model represents it; background statements or citations about transmission routes do not qualify.

\- Transmission from rodents or other animal reservoirs to people is coded Vector/Animal to human; modeled person-to-person exposure through blood, body fluids, secretions, or excretions is coded Human to human (direct contact), and modeled sexual transmission is coded Sexual.

\- Modeled airborne transmission or close contact not specified as a more direct route is coded Airborne or close contact; code each route when the model explicitly represents more than one.

## Interventions type

\- Code an intervention only when the paper's model represents it as an intervention; background discussion or cited examples of control measures do not qualify.

\- Isolation or quarantine of people is coded Quarantine; patient care or medication is coded Treatment, while modeled hospital care or care in dedicated treatment centres is coded Hospitals or Treatment centres, respectively.

Rodent trapping, food protection, sanitation, or similar measures are coded Vector/Animal control when the model represents them as measures to reduce animal-reservoir exposure; vaccine campaigns or vaccine baiting are coded

\- A contact rate, seasonal reservoir dynamic, or saturation term is not an intervention unless the model ties it to an implemented intervention.

\- An explicitly modeled intervention not covered by Vaccination, Quarantine, Vector/Animal control, Treatment, Contact tracing, Hospitals, Treatment centres, Safe burials, Behaviour changes, Wolbachia replacement, Wolbachia suppression, Genetically modified mosquitoes, Mechanical removal of breeding sites, Pesticides/larvicides, Insecticide-treated nets, or Indoor residual spraying is coded Other.

## G.6 Categorization

The Lead Agent categorizes the extracted values itself, with the Categorization Skill appended to each system prompt. One call plans the categories from every paper’s value of the selected field. One call per paper then assigns it to one category or to several.

## System prompts each followed by the Categorization Skill

You organize supplied review evidence into broad, coherent semantic categories.

## One category per paper

You assign supplied review evidence to an allowed semantic category.

## Several categories per paper

You assign supplied review evidence to supported allowed semantic categories.

## Categorization Skill

## ## Choose the basis

## ## Build categories

2. Make categories materially fewer than papers when the sample permits and reuse labels across records.

3. Give categories distinct scopes and concise definitions. Merge near-duplicates and remove empty labels.

4. When some values fit none of the categories, end the list with a category named "Other" for them rather than stretching a category to cover them.

5. For small samples, prefer a few stable umbrella categories over fragile fine-grained themes.

## ## Apply categories

1. Assign labels from the supplied allowed list, following each category's definition.

2. Base each assignment on the supplied field value and paper context, not outside knowledge.

3. In multi-label mode, return only clearly supported labels.

4. Treat harmless case and whitespace   
differences as the same allowed label.

## ## Quality checks

\- Check coverage, overlap, singleton proliferation, empty categories, and definition ambiguity.

\- Present descriptive categories as organization of the supplied evidence rather than causal, statistical, or universal conclusions.

</reviewpilot-agent-skill>

## Category plan

Mode sentence: single mode, then multiple mode Each paper should fit into exactly ONE category.   
Each paper may belong to MULTIPLE   
categories.

## Assignment

One category per paper   
Assign the supplied paper evidence to   
exactly one allowed category, following   
each category's definition.   
TASK DATA:   
{allowed categories with their definitions,   
the paper title, the field and its value}   
Return exactly one JSON object:   
{"category": "one allowed category"}   
Several categories per paper   
Assign the supplied paper evidence to every   
clearly supported allowed category,   
following each category's definition.   
TASK DATA:   
{allowed categories with their definitions,   
the paper title, the field and its value}   
Return exactly one JSON object:   
{"categories": ["allowed category"]}

## H System Interface

Figures H.1–H.8 show ReviewPilot at each stage of the biomedical review in Figure F.1. Each screen has the project sidebar, the stage canvas under the five-step progress bar, and the conversation with the Lead Agent. The screenshots are taken from the finished project, and nothing on the screen is edited. Activity cards in the conversation record each stage as it finished, so the screening card reports the totals from before the researcher reinstated four records (1,701 included and 1,014 excluded), while the canvas and the PRISMA flow show the final counts. Full-Text Retrieval and Information Extraction are marked as partial because 336 papers had no PDF and seven are not extracted.

![](images/e64c61bf54e782abef6c45139c8ecb009424c8180b8d06bc3ae8859fdcbcfba9.jpg)

Figure H.1: Search Setup. In the conversation, the researcher describes the topic and asks to merge two concepts and to set the date range and per-source limits. The canvas shows the resulting research question, the two required concepts, the Boolean query, the three sources with 1,000 results each, and the date range.  
![](images/849f983529633a89935699bc502aa7d3628b3fc14893b5039247d366503852e6.jpg)  
Figure H.2: Paper Screening displays the finalized inclusion and exclusion criteria and the review guidance that the Prompt Agent drafted from them and 200 collected records, confirmed by the researcher before screening.

![](images/03c2bc4cca5e28be89d1de0e1d47818d0ad44b24d43839efbc5a0031a9d6376d.jpg)

(a)  
![](images/1bb98142dcbf841737b5acbe94a29ecffb35af924017df07db81f37c4cd8d44a.jpg)  
(b)  
Figure H.3: Reviewing screening decisions. (a) The review dialog for one of the four exam-evaluation studies that the researcher reinstated (example 2 in Figure F.1), with the saved decision, the researcher’s rationale and approved criterion, and the controls for changing a decision. (b) Two rows of the screening list: a model decision with its likelihood label, rationale, decisive rule, and a quote verified in the title or abstract, and a reinstated record marked as human reviewed.

![](images/2774c3fee584c5b50fa880901c072ff4a3bd87d17e60bad6cd9d3840c14c76d4.jpg)

Figure H.4: Full-Text Retrieval. PDFs are retrieved for 1,369 of 1,705 records. Each of the 336 failures is listed with its reason and can be retried or supplied manually.  
![](images/1ff35d54188649a44b89bebf0fb9617d7e7b500f1cc585d866dda2ad73c07054.jpg)  
Figure H.5: Information Extraction displays the confirmed coding rules (drafted by the Prompt Agent from the finalized schema and the full text of 25 included papers), the notice on the 336 papers without full text, and the first two fields of the six-field schema.

![](images/7151916435d2c38bbcd1523a02a94ea89411a621afdbcd3a6b5ed88215339caa.jpg)

Figure H.6: Inspecting an extracted value of the radiology study in Figure F.1 (example 3). The excerpt supporting llm\_models = GPT-4o is located verbatim on page 3 of the PDF and shown beside the page text. The researcher can correct the value with a supporting quote and a reason.  
![](images/073c3e25750252742c78fb713a7abf9a77a4d130f39fabac15be82d327516dbe.jpg)

(a)  
![](images/7094f79ed4df357d21cab1f184164ae7b244f5aed25f28dd5976507196c53b35.jpg)  
(b)  
Figure H.7: Categorization & Analysis. (a) The top of the stage shows 1,341 papers in five categories and the upper part of the PRISMA 2020 flow. (b) Category counts for the biomedical task field. A study can belong to several categories.

![](images/663c2505b3815d7b54ce76e8c90b3a420e7f12dcca9bcc3c37f74672cc75cce4.jpg)  
Figure H.8: The PRISMA 2020 flow that ReviewPilot generates for the run, from 3,000 identified records to 1,341 included and categorized studies, with exclusion counts by reason.