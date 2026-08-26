# Who is the Agent to Blame? Localizing Faithfulness and Citation Mistakes in Agentic Deep Research

Eran Hirsch<sup>1</sup> David Wan<sup>2</sup> Han Wang<sup>2</sup>

Elias Stengel-Eskin<sup>3</sup> Mohit Bansal<sup>2</sup> Ido Dagan<sup>1</sup>

<sup>1</sup>Bar-Ilan University <sup>2</sup>UNC Chapel Hill <sup>3</sup>University of Texas at Austin

eran.hirsch@biu.ac.il {davidwan, hwang, mbansal}@cs.unc.edu esteng@utexas.edu dagan@cs.biu.ac.il

## Abstract

Deep research (DR) systems produce long form cited reports by orchestrating multiple agents that search and synthesize information from the web. Citations are the primary mechanism for evaluating the faithfulness of these reports, yet current DR systems exhibit poor citation recall. Moreover, improving citation recall is challenging because DR systems are complex multi-agent architectures where information passes through agents like a telephone game, and both content and citations can get corrupted along the way. We propose an evaluation method that pinpoints which agent in troduced each error by locally testing agent invocations for faithfulness and verifiability relative to their own inputs. Furthermore, we propose a four-type taxonomy to categorize the discovered errors: hallucination, uncited input reliance, uncited output, or insufficient citations. Applying our method to three top-ranked open-source DR systems, we obtain actionable diagnostics. Almost every agent makes a lot of mistakes with the exception being those that summarize a single document. We find that the dominant error type varies systematically across agents, where the orchestrator mistakes are mostly citation-related. We find that 84.7% of final-report errors in AI-Q originate at the orchestrator, roughly 31% of them hallucinations and the rest citation mistakes. Guided by these insights, we demonstrate that two simple in terventions raise citation recall by 5% without degrading output quality.<sup>1</sup>

## 1 Introduction

Deep research (DR) systems autonomously search the web and synthesize long-form cited reports in response to a user request. Similar to how humans perceive the factuality of online text (Redi et al.,

2019), citations are a primary mechanism for verifying and measuring the factuality of DR reports (Han et al., 2026; Patel et al., 2025; Skarlinski et al., 2024; Du et al., 2025), making citation reliability a central design concern for developers of DR systems. The standard metric for citation quality, citation recall – the fraction of claims supported by their cited sources (Zhang et al., 2025; Gao et al., 2023) – reveals that current DR systems perform poorly (Han et al., 2026; Patel et al., 2025).

However, improving citation recall is challenging, particularly with current complex DR system architectures (Wang et al., 2024; Li et al., 2025b; Team et al., 2025; Asai et al., 2026), where information passes through multiple agents, like in a “telephone game” (Figure 2). As content crosses agent boundaries, and often gets compressed, both core information as well as citations may get corrupted (Laban et al., 2026).

Aiming to provide DR developers with a better understanding of citation recall errors and their sources, we propose a novel evaluation methodology that addresses two primary goals. First, we evaluate the citation performance of individual agent invocations, with respect to their own inputs. This allows two kinds of evaluation: (a) measuring the average citation performance of each agent component in the DR system, as well as (b) tracing which individual agents are responsible for the eventual citation errors that are present in the final DR report. Second, when an agent makes a mistake, we identify the type of error made; this could be a hallucination (not supported by any source; He et al., 2025), relying on input that was itself uncited in a prior agent step (uncited input reliance), carrying no citations to the current agent’s own output (uncited output), or having citations that do not fully cover its sources (insufficient citations).

We apply our framework to three top-ranked open-source DR systems, Nvidia AI-Q (Nvidia, 2026), MS-Agent (ModelScope, 2026), and TrajectoryKit (Lugoloobi, 2026). We use examples from the dominant DeepResearch Bench (Du et al., 2025) and obtain informative findings. For example, at the agent level, we find that most agents make a lot of mistakes, except those that summarize a single document. Also, that in the two strongermodel systems the dominant failure mode shifts from hallucination at the shallower agents to citation errors at the orchestrator, while the smallermodel system hallucinates at every stage. At the system level, 84.7% of final-report errors in AI-Q are attributed to the orchestrator (main) agent, where the majority of these are citation-related and a sizable 31% of the orchestrator’s own errors are hallucinations. To illustrate the usefulness of such diagnostics, we demonstrate two simple interventions: (1) instruct the orchestrator to avoid using uncited information, and (2) replace synthesized researcher notes with raw source snippets. These interventions successfully raise citation recall by 5% and citation precision by 3% to 7% percentage points, without reducing output quality.

![](images/8de59dc59d27847875e4d417f2139888d3c4d666e028a0c4d7c9eff85d0904f7.jpg)  
Figure 1: Our two evaluation goals for DR systems (Section 3). Left: Our framework samples sentences per agent to produce statistics on how many mistakes it makes, and of what kind. Right: For each sentence in the final report, our framework localizes citation recall errors to the agent that introduced them and classifies errors by type. Dashed arrows represent the original information flow in the DR system; solid lines represent our evaluation process.

Overall, our contributions include:

1. A formal, human-validated method for evaluating a single agent invocation, which locally tests faithfulness and verifiability and enables both agent-level and system-level analyses (Figure 1).

2. A four-type error taxonomy that distinguishes failure modes across agent boundaries.

3. An empirical study revealing prominent error sources and types in three recent competitive open DR systems (Sections 5 and 6).

4. A demonstration of two targeted interventions, guided by our diagnostics, which substantially improve citation quality (Section 7).

## 2 Background

DR systems are multi-agent systems that orchestrate web search, summarization, and long-form report writing. Figure 2 contrasts two open-source instances: Nvidia AI-Q (Nvidia, 2026) and MS-Agent (ModelScope, 2026). These systems are ranked #1 and #2, respectively, in the dominant DeepResearch Bench open-source leaderboard (Du et al., 2025) as of May 2026. Despite implementation differences, they share logical roles: a web search engine that retrieves source pages; a searcher that compresses the retrieved content into doc- or search-level summaries and snippets; a researcher that writes natural-language notes with citations, based on the searcher’s output; and an orchestrator that assembles content into the final cited report. Beyond these shared roles, AI-Q has a planner agent that creates the outline and rubrics, while MS-Agent has the orchestrator write the plan before any web search has been performed. Another difference is that MS-Agent has a reporter writing one chapter at a time. Both systems have an iterative phase where the orchestrator edits the final report. The two systems also differ in the model assigned to each role: AI-Q pairs a frontier model at the planner and orchestrator with a small open-weights model at the researcher, whereas MS-Agent grades model capability with the depth of the agent, from GPT-5 nano at the searcher to GPT-5 mini at the reporter and GPT-5 at the orchestrator (Table 7). This heterogeneity turns out to matter for the errors each agent makes (Section 5). We additionally evaluate TrajectoryKit (Lugoloobi, 2026), ranked #3 on the same leaderboard, which instantiates the same logical roles. Unlike the other two systems, TrajectoryKit runs its agents on a substantially smaller open-weights model gpt-oss-20b (OpenAI et al., 2025), so its model capability does not vary with the depth of the agent. Other deep research systems share similar design principles (Li et al., 2025b,a). Across all of these architectures, the telephone-game failure modes described in Section 1 can arise at many points: when information crosses between agents, when an agent’s history is truncated or summarized, and even between subsequent writing steps within a single agent. Our work develops evaluation methodologies to trace these errors.

![](images/7919e85ac83b2678a1412a16e5e823c45ca50e43346933bbaa591bcb7948b363.jpg)  
Figure 2: Abstract information flow scheme of two state-of-the-art open-source DR systems, Nvidia AI-Q (Nvidia, 2026) and MS-Agent (ModelScope, 2026). Each color represents a different agent type and each colored block represents a different agent invocation. Both systems share the same logical agents of searchers, researchers, and an orchestrator, as is common in opensource deep research systems (see Section 2). All systems, including TrajectoryKit (Lugoloobi, 2026), which we also evaluate, are described in detail in Section I.

The hallucination problem in LLMs motivated a growing body of work on attributed text generation (Gao et al., 2023; Han et al., 2026; Zhang et al., 2025), where output sentences are paired with supporting citations. In DR specifically, citation recall is the standard factuality metric for the entire report (Han et al., 2026; Patel et al., 2025; Du et al., 2025; Li et al., 2026). For each sentence, citation recall checks whether a citation is needed (Zhang et al., 2025) and, if so, whether the associated citations support the claim. Our work goes beyond aggregate citation recall by differentiating specific error types and tracing them to the responsible agent.

## 3 Evaluation Framework

We propose a framework to assist DR developers in identifying the weak parts of their system design in terms of faithfulness and citation mistakes, as measured by citation recall. Our approach is to provide error categorization and localization, measuring what kind of errors occurred and which agent is to blame.

The building block of our evaluation framework is a method to evaluate the output of a single agent invocation with respect to the inputs that it received (Section 4). In contrast to the standard method for evaluating citation recall (Gao et al., 2023) which globally evaluates sentences in the final output report and their citations against the source documents, our method locally evaluates the output of an agent invocation relative to its own inputs. This local approach enables us to pinpoint an error to a specific agent. In addition, this local approach makes faithfulness tests more tractable, since we evaluate an output only relative to its inputs, rather than against potentially hundreds of full source documents.

Using this local agent invocation evaluation method, our framework addresses two complementary evaluation goals (depicted in Figure 1): the agent-level evaluation, which evaluates an agent’s average performance across all of its sampled invocations (Section 5), and the system-level evaluation, which traces only the errors that reach the final report back to their origin (Section 6). These two evaluation types are complementary, because not every output of an agent is eventually used by the final report. In the agent-level evaluation, we report the average percentage of mistakes that the agent makes, as well as an error distribution of those mistakes. In the system-level evaluation, we report the percentage of actual mistakes in the DR report, their origin, and error categories. In analogy to system vs. unit testing, system-level evaluation tells us about the errors that made it to the final report, but can hide mistakes that specific agents made. Agent-level evaluation tells us about the errors that an agent makes, but not whether these errors affected the final report.

## 4 Evaluating an Agent Invocation

To better understand citation recall mistakes produced by deep research systems (Sections 5 and 6), we start by describing how to evaluate a single agent invocation. We propose a method designed to identify the correctness of the agent’s output text, relative to its inputs, in terms of faithfulness (i.e., no hallucinations relative to the input) and verifiability (i.e., citation markers in the output match the citation markers in the input).

![](images/1b22d50d40881fb1832a7a70a469ac71f54275b3cb9ddd56516ae1055c887cc4.jpg)  
Figure 3: Overview of our algorithm for evaluating information errors in a synthesized target sentence within a single agent invocation. Each box corresponds to one step of the method described in Section 4.2.1.

## 4.1 Settings

The scope of our method is the evaluation of a single agent invocation. We extract all the inputs that the agent received, possibly from many separate LLM invocations (e.g., a single agent invocation can include many web searches). In addition, we extract all the outputs that the agent invocation created, possibly from many LLM invocations (e.g., a single agent invocation can include writing to many files). Examples of such agent invocations are depicted in Figure 2. This extraction phase requires a different implementation per DR system (Section I).

From the agent’s perspective, it is assumed that the inputs are faithful to the sources, and that each input info unit is supported by the citations associated with it. The method is not concerned with mistakes that were possibly inherited from prior agent invocations. Accordingly, the agent’s responsibility is to produce output info units that are faithful to the cited input info units, while carrying forward all associated citations. If there are some input info units that are not associated with any citations, they should not be utilized by the agent.

## 4.2 Evaluation Method

The inputs to our algorithm are the extracted inputs and outputs of the agent invocation (Section 4.1). The output of our algorithm is a list of errors that the agent introduced. We differentiate between two types of agent invocation outputs: extracted snippets and synthesized information. Extracted snippets are spans of text that appear verbatim in some raw document. Synthesized information units are generated by the agent model to paraphrase, summarize, or consolidate raw documents and snippets. The error categories differ by output type.

For synthesized outputs, each input and output carries information and, optionally, attributing citation markers (e.g., [N]). Conceptually, the following things can go wrong: (1) the output is unfaithful to the input (hallucination); (2) the output is faithful to the input but draws on input that was itself uncited (uncited input reliance); (3) the output is faithful to cited input but carries no citations at all (uncited output); or (4) the output has citations but fails to include the complete set of citations of the input it relied on (insufficient citations). For snippet outputs, which by definition belong to a single identified document, citation is inherited from the source identity, so the only concern is whether extracting the snippet removed important context, rendering it uninterpretable or hallucinated (snippet missing context). We describe the method for each output type separately: synthesized information in Section 4.2.1 and snippets in Section 4.2.2.

## 4.2.1 Evaluation for Synthesized Information

The algorithm, explained below, is visualized in Figure 3 (see Section A for the pseudo-code). Because DR systems typically attach citations at the sentence level, we segment each synthesized output information unit into sentences and run the algorithm on each sentence independently, returning one of the four error labels above or no mistake. The algorithm, a sequence of attribution and entailment tasks, proceeds as follows, where each step corresponds to a box in Figure 3:

• Step 1 – Faithfulness test. An LLM extracts all supporting spans from the information provided by prior agent invocations; spans should be exhaustive, even if redundant or only partially supporting. If the extracted spans do not fully support the target sentence, either because no spans were found or because they support it only in part, this indicates a hallucination (terminal verdict). Otherwise, the output is faithful to its inputs and we proceed to the verifiability tests.<sup>2</sup>

• Steps 2–3 – First verifiability test (uncited input reliance). Given that the output is faithful, we now check whether it relies on uncited input. Because citations in DR outputs are often not attached to every sentence, an LLM first identifies which citations are assigned to each input span (Step 2). We then remove uncited spans and test whether the remaining cited spans still entail the target sentence (Step 3). If not, the output is classified as uncited input reliance (terminal verdict).

• Steps 4–5 – Second verifiability test (citation alignment). Finally, we check whether the target sentence’s own citations are complete, in the sense that they carry all necessary citations from the cited input spans. An LLM first identifies the target sentence’s citations (Step 4). We then check alignment: if the target sentence carries no citations at all, we conclude it is an uncited output (terminal verdict). If it does have citations, we keep only input spans whose citations are a subset of the target’s citations and re-run the entailment test. If this subset no longer supports the target sentence, we conclude that the target has insufficient citations (terminal verdict). Otherwise the output is correct.

Algorithm implementation. All our steps are based on existing prompts from prior work. The

Step 1 span extraction prompt is based on the prompt used for Localized Attribution Queries (Hirsch et al., 2025). The Steps 2 and 4 citation classification is based on the approach introduced by DEER (Han et al., 2026) to semantically identify which citations are related to a sentence or span. All other prompts are entailment tests based on the LongCite prompt (Zhang et al., 2025). The LLM used as the judge is gpt-5-mini-2025-08-07 (Singh et al., 2026). The full prompts are provided in Section B.

## 4.2.2 Evaluation for Extracted Snippets

Zhang et al. (2023) identified that a snippet extracted from a larger source can remove necessary context, creating uninterpretable or even hallucinated information. Accordingly, we run a single entailment test that asks whether the source document supports the entire snippet. If the test passes, the snippet is considered correct. If not, it receives the error label snippet missing context.

## 4.3 Human Assessment of Method Implementation Performance

In order to assess the reliability of our automated agent invocation evaluation procedure, we asked humans to perform the same classification task as our algorithm in Figure 3. Each annotation task showed the annotator a target sentence (in the report context) and the agent invocation’s inputs, and asked them to pick one of six labels (no citation needed, hallucination, uses uncited info, uncited, insufficient citations, no issues), the last five corresponding to the method’s terminal verdicts for synthesized information. The full annotation guidelines are included in Section E.

Setup. We sampled 50 sentences from AI-Q examples and had each sentence annotated independently by two annotators drawn from a pool of four. In case of disagreements, a third annotator arbitrated and made the final call. The annotators were all graduate students with prior experience reading academic literature and using deep research systems. Annotators received an orientation session and completed calibration tasks before beginning the main annotation.

Results. The inter-annotator agreement before arbitration was Cohen’s κ=0.71 (substantial) (Cohen, 1960; Landis and Koch, 1977). We then compared the arbitrated labels with the method’s verdicts, yielding 76% exact matches and κ=0.62 (substantial).

In Section F we validate the supporting spans, using the same annotation effort. The localization process agrees with the annotators on 75% of the relevant sentences. This validates the system-level analysis (Section 6), whose recursion follows these spans back from the final report to the agent that introduced the error.

## 5 Evaluating Average Agent Performance

The goal of this evaluation is to extract average performance statistics per agent. Ideally, we would evaluate all the sentences of all agent invocations per agent, but that would be intractable. Instead, we sample sentences without repetition from all the invocations of each agent type in each example.

For each sentence sampled, we first classify if a citation is needed (Redi et al., 2019; Zhang et al., 2025). If not, we discard it. We use the LongCite “is citation needed” prompt (Section B.4).<sup>3</sup> For each sample, we then run the agent invocation evaluation method described in Section 4.2. Finally, for each agent, we report its mistake rate and the distribution of error types.

## 5.1 Experimental Setup

We apply our method to the three state-of-theart open-source deep research systems described in Section 2: Nvidia AI-Q (Nvidia, 2026), MS-Agent (ModelScope, 2026), and TrajectoryKit (Lugoloobi, 2026). The LLMs used by each agent are listed in Table 7 (appendix). We evaluate on 20 examples from DeepResearch Bench (Du et al., 2025). For each example, we sample up to 10 sentences per agent, yielding roughly 200 sentences per agent, with exact counts presented in Table 1.

## 5.2 Results

Per-agent correct and mistake percentages for all three systems are reported jointly in Table 1. The per-agent error distribution for AI-Q is shown in Figure 4; the corresponding MS-Agent and TrajectoryKit distributions are in Figures 13 and 14 (placed in the appendix).

Agents that summarize a single document make the fewest mistakes. Almost every agent in the three systems makes significant mistakes on a substantial share of its sampled sentences. The clear exceptions are the searcher components that compress one document at a time: the AI-Q searcher snippets (3.8%), the MS-Agent searcher snippets and synthesized summaries (6.4% and 0.9%, respectively), and the TrajectoryKit searcher synthesized summaries (14.9%). AI-Q’s searcher synthesized summaries also perform a summarization, but over all documents returned for a query rather than one, and their mistake rate rises to 62.0%. The remaining spread within the singledocument group can be attributed to the underlying model rather than the architecture, as TrajectoryKit’s gpt-oss-20b digests (14.9%) are an order of magnitude worse than MS-Agent’s otherwise identical per-document summaries (0.9%).

<table><tr><td>Component</td><td></td><td>#</td><td>Correct %</td><td>Mistakes %</td></tr><tr><td rowspan="5">A-IV</td><td>Orchestrator</td><td>193</td><td>69.1</td><td>30.9</td></tr><tr><td>Planner*</td><td>121</td><td>78.0</td><td>22.0</td></tr><tr><td>Researcher</td><td>200</td><td>29.2</td><td>70.8</td></tr><tr><td>Searcher (Synthesized)</td><td>192</td><td>38.0</td><td>62.0</td></tr><tr><td>Searcher (Snippets)</td><td>189</td><td>96.2</td><td>3.8</td></tr><tr><td rowspan="5">M-Sent</td><td>Orchestrator</td><td>183</td><td>83.2</td><td>16.8</td></tr><tr><td>Reporter</td><td>190</td><td>38.4</td><td>61.6</td></tr><tr><td>Researcher</td><td>170</td><td>53.4</td><td>46.6</td></tr><tr><td>Searcher (Synthesized)</td><td>110</td><td>99.1</td><td>0.9</td></tr><tr><td>Searcher (Snippets)</td><td>110</td><td>93.6</td><td>6.4</td></tr><tr><td rowspan="4">Trakit</td><td>Orchestrator</td><td>176</td><td>7.5</td><td>92.5</td></tr><tr><td>Researcher</td><td>200</td><td>50.0</td><td>50.0</td></tr><tr><td>Searcher (Synthesized)</td><td>154</td><td>85.1</td><td>14.9</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 1: Agent-level performance for Nvidia AI-Q, MS-Agent, and TrajectoryKit. <sup>∗</sup>For the planner agent, “uncited output” is not considered as an error, because this is by design (Figure 2).

Deeper agents hallucinate less. Agents deeper in the information flow, which receive information that was synthesized more times, hallucinate less than shallower ones. Only 31% of the AI-Q orchestrator’s mistakes are hallucinations, against 79% for its researcher. MS-Agent shows the same contrast even more sharply: none of the orchestrator’s mistakes are hallucinations, against 78% for the reporter and 92% for the researcher.

A plausible explanation is the capability of the underlying model rather than depth in itself, as the shallower agents run substantially weaker LLMs than the orchestrator in both systems (see Table 7 in the appendix). TrajectoryKit is consistent with this explanation: it runs the same model at every stage and correspondingly shows no shift, with hallucinations accounting for most of the mistakes

![](images/8482cb5d839cebcbbd03fb88973d86efb3e6bda48e469d0967cabe0052f0db47.jpg)  
Figure 4: Agent-level error distribution for each agent of the Nvidia AI-Q DR system, showing the breakdown of the error portion from Table 1. The corresponding distributions for the MS-Agent and TrajectoryKit DR systems are in Figures 13 and 14 (appendix).

at each of its agents.

## 6 Tracing System-level Errors

We propose a system-level evaluation to trace and localize global citation recall errors to the specific agents that created them and to categorize them into specific error types. We first describe an algorithm for the evaluation (Section 6.1) and then discuss the results (Section 6.2).

## 6.1 Tracing Algorithm

We first have to find which sentences have citation recall issues globally. Then, for each such sentence, we run multiple local tests recursively, starting from the agent that created the final report, until we identify which agent introduced the error.

## 6.1.1 Calculating Citation Recall in DR

The implementation of the global citation recall in Step 1 follows standard practice (Zhang et al., 2025) of answering two questions: (1) whether the target sentence needs a citation, and (2) whether it has citations that support it. Similar to the local algorithm, we use a citation-identification prompt (drawn from DEER (Han et al., 2026)) to semantically find which citations are aligned with the target sentence. The LLM judge for all prompts is gpt-5-mini-2025-08-07 (Singh et al., 2026).

Because DR systems can hallucinate URLs (Onweller et al., 2026), we use the tracing log to build a ground-truth list of retrieved URLs and filter out any hallucinated citations. We also adopt a no-crawl design, extracting raw documents from the tracing log instead of re-crawling, which sidesteps crawl-fragility issues (disagreement between crawlers, paywalls, dynamic content).

## 6.1.2 Finding the Agent to Blame

Given a target sentence in the final report that has global citation recall errors, we now recursively iterate over agent invocations and try to find the agent invocation to blame by running local tests.

We first run the local agent invocation test from Section 4.2, which checks whether the agent’s output is faithful to and correctly cited against its inputs. If the test detects an error, we attribute it to this agent and return. Otherwise, we choose one or more targets to recurse into and return their found errors. Specifically, if the target sentence has no errors, this means that the local algorithm from Section 4.2 must have identified spans that fulfill the following conditions: (1) they support the target sentence, (2) they have citations, and (3) their citations are included in the target sentence. For each such span, we check if it originates from a raw document, which would indicate that we reached the end of the recursion without finding any agent to blame. Otherwise, we recurse into the agent invocation that produced the span, applying the same procedure to the sentence containing it. The recursion terminates when an error is found or a raw document is reached. The final output is a list of found errors.

## 6.2 Results

The experimental setup for this evaluation uses the same sample from Section 5. The global citation recall for AI-Q is 58.7%, for MS-Agent is 28.5%, and for TrajectoryKit is 7.1%. Statistics on agents that were blamed for creating errors are reported jointly in Table 2. The per-category error distribution for AI-Q is shown in Figure 5; the corresponding MS-Agent and TrajectoryKit distributions are in Figures 15 and 16. Table 5 (appendix) provides representative examples for each mistake category produced by our algorithm.

![](images/a633b8318ebbe60c082bbf7f3d515083ecf45c6b2fd004967675271a1bb22813.jpg)

Figure 5: System-level error distribution for Nvidia AI-Q. The corresponding distributions for MS-Agent and TrajectoryKit are in Figures 15 and 16 (appendix).
<table><tr><td>System</td><td>Component</td><td>% of errors</td></tr><tr><td rowspan="3">AI-Q</td><td>Orchestrator</td><td>84.7</td></tr><tr><td>Researcher</td><td>14.8</td></tr><tr><td>Searcher (Synthesized)</td><td>0.4</td></tr><tr><td>MS-Agent</td><td>Orchestrator Reporter</td><td>52.6 47.4</td></tr><tr><td>TrajectoryKit</td><td>Orchestrator</td><td>100.0</td></tr></table>

Table 2: System-level error-origin distribution for Nvidia AI-Q, MS-Agent, and TrajectoryKit. For each system, the share of confirmed final-report errors attributed to each agent.

The majority of mistakes start at the orchestrator. For AI-Q, 84.7% of final-report errors were attributed to the orchestrator, for MS-Agent 52.6%, and for TrajectoryKit 100%. For the first two systems this is despite the orchestrator having one of the lowest agent-level error rates (30.9% for AI-Q and 16.8% for MS-Agent in Table 1), which illustrates the value of system-level evaluation on top of agent-level evaluation. TrajectoryKit is the opposite case: its orchestrator is also its weakest agent (92.5%), so the two levels of evaluation agree.

Final output mistakes are mostly citationrelated errors. For AI-Q 70% of the mistakes are citation-related mistakes at the orchestrator level, and for MS-Agent 99%. This is consistent with the results of the agent-level evaluation. For MS-Agent, we find that the final report sometimes contains citations to IDs of the reporter’s notes instead of source documents (see Section I). TrajectoryKit is the exception: only 4% of its orchestrator errors are citation-related, as discussed below.

Hallucinations are a notable share of AI-Q’s final-report errors. Beyond citation-related errors, a substantial fraction of errors that reach AI-Q’s final report are hallucinations. Hallucinations account for roughly 31% of AI-Q’s final-report errors (30% of the orchestrator’s 200 errors and 37% of the researcher’s 35), with most originating at the orchestrator. For MS-Agent, hallucinations are far less of an issue, at roughly 11%. This is notable because hallucinations are arguably more severe than citation-related errors.

The smaller-model system fails in a different way. For TrajectoryKit, 95% of its final-report errors are hallucinations. Manual inspection shows a recurring failure mode in which the orchestrator attaches a fabricated precise statistic to a source that does not state it. This is consistent with its markedly lower citation recall, even though its output quality on the RACE metric (Du et al., 2025) is competitive with MS-Agent’s. This reinforces the model-capability explanation from the agent-level evaluation (Section 5): the dominant error type is governed less by the multi-agent architecture, which the three systems largely share, than by the capability of the underlying model.

How errors propagate across agents. Although the orchestrator is the most common origin, a substantial share of final-report errors starts elsewhere (Table 2), which means several subagents propagate mistakes received as input up to the final report. Table 6 in the Appendix walks through three representative chains that illustrate example patterns: a searcher fabricates a claim that the researcher and the orchestrator faithfully propagate; the orchestrator abstracts a researcher’s fabricated claim while keeping its citations; and unsupported statistics travel verbatim with a citation that never supported them.

## 7 Interventions to Improve Citation Recall

In this section, we show how insights from our evaluation can help improve DR systems. We demonstrate how simple changes, not implemented by the original developers, can improve citation quality on AI-Q. Future work could include more complex fixes, such as using our evaluation’s diagnostics as a corrective feedback signal during generation.

We propose two fixes for AI-Q based on our evaluation’s findings. The first fix follows from our average agent performance results: the searcher snippets, each extracted from a single document, are the most reliable artifact in the system (3.8% mistakes), whereas the researcher notes that currently reach the orchestrator are produced by its least reliable agent (70.8%) and consolidate several documents at once. This motivates the following fix: in the orchestrator inputs, we replace the synthesized researcher agents’ notes with the snippets of the URLs they cite, so that each unit the orchestrator reads is a single-document extract rather than a multi-document synthesis. This enables the orchestrator to directly access the raw document snippets, and we expect that it will reduce the number of citation-related errors. For the second fix, we focus on our finding that the final output mistakes are mostly citation-related errors. Accordingly, we append to the orchestrator prompt the following instruction: “do NOT use information which is not backed by citations.” We expect that this will have an effect both on the reliance on uncited input, as well as on the uncited output. We evaluate each of these two changes independently.

Experimental setup. We evaluate on 50 examples from DeepResearch Bench, covering all of its English queries, and report the RACE quality metric (Du et al., 2025). In addition, we report the standard citation recall metric, as described in Section 6. Finally, we also calculate citation precision using the citation precision prompt from LongCite (Zhang et al., 2025).

Results. Table 3 shows that for AI-Q, both interventions substantially increase citation recall by 5% without significantly changing output quality. Both also increase citation precision by 3% to 7%. The two interventions perform comparably on recall.

To further analyze the citation guidance intervention, we run our system-level tracing method (Table 4; Figure 17 in the appendix). We find that the orchestrator’s share of errors drops from 84.7% to 77.0% after the intervention. In addition, we find that the orchestrator’s citation-related errors after the intervention drop from 70% to 60%, compared to the citation-related errors made before the intervention (Figure 5). Overall, our intervention was able to reduce orchestrator citation-related error as expected, resulting in a stronger DR system across all metrics. Future work could further attempt to reduce orchestrator errors, or divert the focus to fixing researcher-induced errors.

<table><tr><td>System</td><td>RACE</td><td>C. Recall</td><td>C. Precision</td></tr><tr><td>AI-Q</td><td>52.6±0.4</td><td> $6 4 . 5 { \pm } 3 . 9$ </td><td> $8 7 . 6 { \pm } 3 . 0 $ </td></tr><tr><td>+ replace w/ snippets</td><td>52.4±0.7</td><td> ${ \bf 6 9 . 7 \pm 4 . 1 }$ </td><td> ${ \bf 9 4 . 1 { \pm } 2 . 0 }$ </td></tr><tr><td>+ citation guidance</td><td>52.6±0.6</td><td> ${ \bf 6 9 . 6 { \pm 3 . 2 } }$ </td><td> $9 1 . 0 { \pm } 2 . 3 $ </td></tr></table>

Table 3: Results for simple AI-Q DR system modifications based on our evaluation findings. Values after ± are standard errors of the mean.
<table><tr><td>System</td><td>Component</td><td>% of errors</td></tr><tr><td rowspan="2">AI-Q + citation guidance</td><td>Orchestrator</td><td>77.0</td></tr><tr><td>Researcher</td><td>23.0</td></tr></table>

Table 4: System-level error-origin distribution for AI-Q after the citation guidance prompt change. Compare against Table 2: the orchestrator’s share decreases from 84.7% to 77.0%, while the researcher’s share increases from 14.8% to 23.0%.

## 8 Conclusion

We presented a methodology for diagnosing citation recall errors in deep research systems along two complementary axes: what kind of error occurred, and which agent introduced it. Our central finding is that the dominant error type varies systematically across agents: in the two stronger-model systems, the orchestrator hallucinates less and its mistakes are mostly citationrelated, whereas the smaller-model system fabricates content at every stage. At the system level, the orchestrator, despite having one of the lowest agent-level error rates, is responsible for 84.7% of final-report errors in AI-Q. These diagnostics translate directly into targeted fixes: even a one-sentence prompt intervention raises citation recall by 5% and citation precision by 3–7% without reducing output quality. A natural follow-up is post-hoc mitigation that feeds the system-level verdicts back into the DR loop as a critique signal, and we expect this approach to generalize to other multi-agent attributed generation settings.

## Limitations

LLM-as-a-judge reliability. The faithfulness and verifiability checks of our algorithm (Section 4) rely on LLM-as-judge prompts adapted from prior work. We mitigate this concern in two ways: first, all checks are attribution and entailment tests drawn from prior work (Section 4.2); second, a human annotation study (Section 4.3) shows substantial agreement between annotator consensus and algorithm verdicts.

Attributing error types to model capability. Our observation that the dominant error type tracks the capability of the model an agent runs (Sections 5 and 6.2) is observational rather than controlled. Isolating the factor would require running the same system with the same model across roles, which we leave to future work.

Scope of evaluated content. We evaluate citations on running-text sentences only. Tabular content (numeric cells with trailing citation markers) is excluded from both the recall headline and the system-level analysis.

## Ethical Considerations

Our work contributes to the goal of improving faithfulness and verifiability of deep research systems, advocating for better evaluation methods. However, errors in the different steps of the proposed methodology can mislead developers into drawing incorrect conclusions about the quality of the deep research system, either overstating or understating its reliability.

In the human evaluation (Section 4.3), each annotation task took about 10 minutes on average, for roughly 6 hours of work per annotator. Annotators were paid \$35 per hour, which we consider adequate compensation for this expertise level and country of residence.

AI-writing tools assisted the writing of the paper to improve clarity and coherence. Any writing change was carefully requested from the AI writing tool and edited by the authors before incorporating it into the paper.

## Acknowledgments

This work was supported in part by the Israel Science Foundation (grants no. 2827/21 and 3182/25), NSF-AI Engage Institute DRL2112635, NSF-CAREER Award 1846185, and a Google PhD Fellowship. The views contained in this article are those of the authors and not of the funding agency. Special thanks to Jan Buchmann, Shmuel Amar, Uri Katz, and Aviya Maimon for their valuable discussions at various stages of the project.

## References

Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’Arcy, and 1 others. 2026. Synthesizing scientific literature with retrieval-augmented language models. Nature, pages 1–7.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20:37 – 46.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. 2025. Deepresearch bench: A comprehensive benchmark for deep research agents. arXiv preprint. To appear in Proceedings of ICLR 2026.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488, Singapore. Association for Computational Linguistics.

Janghoon Han, Heegyu Kim, Changho Lee, Dahm Lee, Min Hyung Park, Hosung Song, Stanley Jungkyu Choi, Moontae Lee, and Honglak Lee. 2026. Deer: A benchmark for evaluating deep research agents on expert report generation. Preprint, arXiv:2512.17776.

Jacqueline He, Howard Yen, Margaret Li, Shuyue Stella Li, Zhiyuan Zeng, Weijia Shi, Yulia Tsvetkov, Danqi Chen, Pang Wei Koh, and Luke Zettlemoyer. 2025. Precise information control in long-form text generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Eran Hirsch, Aviv Slobodkin, David Wan, Elias Stengel-Eskin, Mohit Bansal, and Ido Dagan. 2025. LAQuer: Localized attribution queries in content-grounded generation. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15355–15370, Vienna, Austria. Association for Computational Linguistics.

Philippe Laban, Tobias Schnabel, and Jennifer Neville. 2026. Llms corrupt your documents when you delegate. Preprint, arXiv:2604.15597.

J Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33 1:159–74.

Ruizhe Li, Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. 2026. Deepresearch bench ii: Diagnosing deep research agents via rubrics from expert report. Preprint, arXiv:2601.08536.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and Zhicheng Dou. 2025a. Webthinker: Empowering large reasoning models with deep research capability.

In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Zijian Li, Xin Guan, Bo Zhang, Shen Huang, Houquan Zhou, Shaopeng Lai, Ming Yan, Yong Jiang, Pengjun Xie, Fei Huang, Jun Zhang, and Jingren Zhou. 2025b. Webweaver: Structuring web-scale evidence with dynamic outlines for open-ended deep research. Preprint, arXiv:2509.13312.

William Lugoloobi. 2026. Trajectorykit: A local-first agentic framework.

ModelScope. 2026. modelscope/ms-agent. https://github.com/modelscope/ms-agent.

NVIDIA. 2025. Nemotron 3 Nano: Open, efficient mixture-of-experts hybrid Mamba-Transformer model for Agentic reasoning. Technical report.

Nvidia. 2026. Nvidia-AI-Blueprints/aiq. https://github.com/NVIDIA-AI-Blueprints/aiq.

Hailey Onweller, Elias Lumer, Austin Huber, Pia Ramchandani, Vamse Kumar Subbiah, and Corey Feld. 2026. Cited but not verified: Parsing and evaluating source attribution in llm deep research agents. Preprint, arXiv:2605.06635.

OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, and 108 others. 2025. gpt-oss-120b & gptoss-20b model card. Preprint, arXiv:2508.10925.

Liana Patel, Negar Arabzadeh, Harshit Gupta, Ankita Sundar, Ion Stoica, Matei Zaharia, and Carlos Guestrin. 2025. Deepscholar-bench: A live benchmark and automated evaluation for generative research synthesis. In NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling.

Miriam Redi, Besnik Fetahu, Jonathan Morgan, and Dario Taraborelli. 2019. Citation needed: A taxonomy and algorithmic assessment of wikipedia’s verifiability. In The World Wide Web Conference, WWW ’19, page 1567–1578, New York, NY, USA. Association for Computing Machinery.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, and 467 others. 2026. Openai gpt-5 system card. Preprint, arXiv:2601.03267.

Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. 2024. Language agents achieve superhuman synthesis of scientific knowledge. Preprint, arXiv:2409.13740.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, and 1 others. 2025. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701.

Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan Zhang, Xinyu Dai, Min Zhang, Qingsong Wen, Wei Ye, Shikun Zhang, and Yue Zhang. 2024. Autosurvey: Large language models can automatically write surveys. In The Thirtyeighth Annual Conference on Neural Information Processing Systems.

Jiajie Zhang, Yushi Bai, Xin Lv, Wanjun Gu, Danqing Liu, Minhao Zou, Shulin Cao, Lei Hou, Yuxiao Dong, Ling Feng, and Juanzi Li. 2025. LongCite: Enabling LLMs to generate fine-grained citations in long-context QA. In Findings of the Association for Computational Linguistics: ACL 2025, pages 5098–5122, Vienna, Austria. Association for Computational Linguistics.

Shiyue Zhang, David Wan, and Mohit Bansal. 2023. Extractive is not faithful: An investigation of broad unfaithfulness problems in extractive summarization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2153–2174, Toronto, Canada. Association for Computational Linguistics.

## A Algorithm Pseudo-code

Algorithm 1 provides the pseudo-code for the single agent invocation evaluation procedure described in Section 4. Algorithm 2 provides the simpler procedure used for snippet outputs. The helper routines EXTRACTSUPPORTINGSPANS (Step 1), CITATIONSOF (Steps 2 and 4), and ENTAILS (Steps 1, 3, and 5) are LLM calls; see Section B for the full prompts.

## B Prompts

This appendix lists the prompts used in the algorithm of Section 4.2, the agent-level analysis of Section 5 and the system-level analysis of Section 6. Section C then briefly describes the modifications applied to AI-Q for the application study (Section 7); we omit the full third-party prompts and only describe the changes.

## B.1 Step 1: Span Extraction

Figure 6 shows the prompt used for the LLM call in Step 1 of Section 4. Given a target sentence and the documents output by the parent agent invocations, it extracts verbatim supporting spans and emits both a per-document support verdict and an overall verdict. The prompt is adapted from the Localized Attribution Queries prompt of Hirsch et al. (2025).

Algorithm 1 Evaluating a single agent invocation on a target sentence.   
Require: Target sentence s; input info units I available to the agent invocation (already extracted from   
the trace; see Section I); retry budget K.   
Step 1: Faithfulness test (with self-critique retries).   
1: $f b \gets \emptyset$ ▷ feedback for retries   
2: for $k = 1 , \ldots , K$ do   
3: S ← EXTRACTSUPPORTINGSPANS $\left( s , I , f b \right)$   
4: if $S = \emptyset$ then return HALLUCINATION   
5: end if   
6: if not ALLVERBATIM(S, I) then   
7: $f b \gets ^ { \ast }$ “some spans are not verbatim”; continue   
8: end if   
9: if ENTAILS $( \cup S , s )$ then break ▷ output is faithful   
10: end if   
11: $f b \gets ^ { { \bf \Pi } ^ { 6 6 } }$ union of spans does not entai $s ^ { \prime \prime }$   
12: end for   
13: if retries exhausted without success then return EVAL\_FAILED   
14: end if   
Step 2: Input spans citation classification.   
15: C<sub>σ</sub> ← CITATIONSOF(σ) for each $\sigma \in S$   
Step 3: Uncited input reliance test.   
16: $S ^ { \mathrm { c i t e d } }  \{ \sigma \in S : C _ { \sigma } \neq \varnothing \}$   
17: if not ENTAILS $( \cup S ^ { \mathrm { c i t e d } } , s )$ then return USED\_UNCITED   
18: end if   
Step 4: Target sentence citation classification.   
19: $C _ { s } \gets \mathbf { C } _ { \mathrm { I T A T I O N S O F } } ( s )$   
Step 5: Citation alignment.   
20: if $C _ { s } = \emptyset$ then return UNCITED   
21: end if   
22: $S ^ { \mathrm { a l i g n e d } }  \{ \sigma \in S ^ { \mathrm { c i t e d } } : C _ { \sigma , \subseteq } C _ { s } \}$   
23: if not ENTAILS $\left( \cup S ^ { \mathrm { a l i g n e d } } , s \right)$ then return INSUFFICIENT\_CITATIONS   
24: end if   
25: return CORRECT

Algorithm 2 Evaluating a snippet output unit.   
Require: Snippet o with source URL u; raw doc  
ument $d _ { u }$ retrieved for u in the log L.   
1: if not ENTAIL $. s ( d _ { u } , o )$ then return SNIP-  
PET\_MISSING\_CONTEXT   
2: end if   
3: return CORRECT

When verification of the extracted spans fails (either some spans are not verbatim, or the union does not entail the target sentence under the entailment check in Figure 8), the prompt is re-issued with the relevant feedback fragment from Figure 7. We allow up to five retry attempts before falling back to a hard failure.

## B.2 Steps 1, 3 and 5: Entailment Test

Figure 8 shows the entailment prompt, adapted from Zhang et al. (2025) to accept multiple documents and full document texts (rather than chunked passages). This prompt is used for the unionentailment check that validates Step 1, for the entailment tests of Steps 3 and 5, for snippet evaluation (Section 4), and for the citation-recall check of Section 6.

For the union check of Step 2, the entailment prompt is composed with the extension in Figure 9. The extension requires the model to additionally emit a description what is missing; this description seeds the {reasoning} slot of the self-critique feedback (Figure 7) on the next span-extraction attempt.

![](images/5c2f4e4e5829ec27bd8c34fd8ce48b2d8ce2d9b1b90583ed36cae59a304214b2.jpg)  
Figure 6: Step 2 span-extraction prompt, adapted from the Localized Attribution Queries prompt of Hirsch et al. (2025). The {feedback\_section} slot is populated by the templates in Figure 7 on retry attempts.

## B.3 Steps 2 and 4: Citation Classification

Figure 10 shows the citation classification prompt, adapted from the DEER citation classification scheme (Han et al., 2026). This prompt is used in Steps 2 and 4 of Section 4 and in the systemlevel analysis (Section 6) to semantically identify which citations are aligned with a given sentence or span.

## B.4 Citation Recall: Need-Citation Test

For the citation-recall test in Section 6 (Step 1), we also check whether a sentence requires a citation in the first place. This prompt (Figure 11) is adapted from Zhang et al. (2025).

## C Modifications to AI-Q

The application study (Section 7) applies two interventions to AI-Q: (1) a prompt change forbidding the use of uncited information by the orchestrator, and (2) replacing AI-synthesized researcher outputs with the raw snippets of the URLs they cite. We describe the implementation below rather than reproducing the full third-party prompts.

AI-Q prompt change. A single sentence is appended to the orchestrator’s prompt: “However, do NOT use information which is not backed by citations.” The sentence is added inside the orchestrator’s “Critical Rules” section, immediately after the existing instruction to fall back to best-effort analysis when search tools return insufficient data.

![](images/f2aa6da922e6943d2f84f3d02725242540a2945a57b72f5966fe645f061cf6a2.jpg)  
Figure 7: Self-critique retry feedback templates injected into {feedback\_section} of Figure 6.

AI-Q snippet replacement. The researcher subagents normally hand the orchestrator a synthesized note with the raw search-engine snippets of the cited URLs appended at the end. Our intervention drops the synthesized note body and keeps only the appended “Raw Source Snippets” section, which is assembled from the citation registry.

Post-intervention system-level breakdown. Figure 17 shows the full per-agent, per-category system-level error distribution for AI-Q after applying the citation guidance prompt change. Table 4 reports the corresponding error-origin distribution. The headline numbers extracted from this figure are discussed in Section 7.

## D Example Errors per Category

Table 5 shows representative target sentences flagged by our tracing evaluation (Section 6) for each mistake category, with the highlighted portion indicating the problematic span and the explanation describing why the verdict was assigned.

Table 6 complements these single-invocation examples with three full propagation chains from the AI-Q runs, showing how an error introduced by one agent invocation interacts with the behavior of the agents that depend on it.

## E Human Annotation Guidelines

Figure 12 includes verbatim the guidelines given to the human annotators whose agreement with our algorithm is reported in Section 4.3. Each annotation task showed the annotator the target sentence (in a Report tab) and the input info units available to the writing agent (in an “Information providers” tab), and asked them to pick one of six labels. Over an initial online orientation meeting, each annotator completed two calibration tasks before beginning the main annotation, and annotators were informed that their annotations would be used and published as part of an academic research project.

Labels 2–6 correspond directly to the algorithm’s terminal verdicts in Section 4.2.1: hallucination, uncited input reliance, uncited output, insufficient citations, and correct. Label 1 (N/A) corresponds to sentences that the algorithm filters out via the need-citation prompt of Section B.4.

## F Validating the Supporting Spans

Section 4.3 validates the classification produced by the local test. This appendix validates its other output, the supporting spans: the input spans the algorithm identifies as supporting a target sentence.

Why these two outputs suffice. The systemlevel tracing of Section 6 adds no model-based judgment beyond the local test. Each recursive step reuses the supporting spans that the local algorithm of Section 4 already produced: the spans determine which agent invocation to descend into (Section 6.1), and the descent itself involves no additional model calls. The error-prone components of the pipeline are therefore the local test’s classification and its supporting spans, so validating both directly validates the system-level tracing built on top of them.

![](images/a219aabd5f3da9256a0e5b1aec0579201e52da6a9ff6e294a2c96a0fc3e8dabd.jpg)

Figure 8: Is supported prompt adapted from the LongCite (Zhang et al., 2025).  
![](images/1cfcfd27fb507ca919791d9e08b48ba8ae76cf1d9bb954c698e9a121ef38db8a.jpg)  
Figure 9: Extension appended to the prompt in Figure 8 for the Step 2 feedback loop.

Annotation. In the same human evaluation described in Section 4.3, annotators also recorded which input spans they used to support their classification. For each such sentence we compare the input spans marked by the annotators against those returned by the localization process, counting them as agreeing when they select the same supporting input sentences.

Results. For 75% of the relevant target sentences, the localization process agrees with the annotators on the input sentences that support the target sentence. As expected, this closely tracks the 76% exact-match rate on the classification task reported in Section 4.3, indicating that the two outputs of the local test are reliable to a similar degree.

## G More Experimental Details

DeepResearch Bench (Du et al., 2025) spans 22 domains and includes questions in both English and Chinese. In our experiments, we used only English examples and leave multilingual evaluation to future work. The diagnostic analyses (Sections 5 and 6) use 20 examples (numbered 51–70), and the intervention study (Section 7) extends this to all 50 English examples (numbered 51–100).

The LLM model used as the judge is gpt-5-mini-2025-08-07 (Singh et al., 2026).

## H Licenses

All artifacts used in this work are released under permissive open-source licenses, and our use is consistent with their intended use.

• DeepResearch Bench (Du et al., 2025): both the codebase and the accompanying dataset are released under the Apache License 2.0.

<table><tr><td>Agent &amp; Type of error</td><td>Target sentence with citations</td><td>Explanation</td></tr><tr><td>Orchestrator Hallucination</td><td>A Japan-focused cross-sectional study finds measurable consumer willingness-to-pay for food defense and food hygiene, supporting a plausible channel for higher value-per-unit food consumption (safer supply chains, trusted brands, convenience-oriented services). [https://www.i-jmr.org/2023/1/e43936]</td><td>The two highlighted parts of the claim are not supported by the input info units. Interestingly, some of these points are actually discussed in the cited URL, suggesting familiarity of the model with these issues.</td></tr><tr><td>Orchestrator – Uncited input reliance</td><td>Attention inspection (when using transformer-like models) can provide a form of structured saliency by showing which time steps or features the model “attended&quot; to; recent portfolio ML frameworks highlight interpretability/efficiency trade-offs using attention structures. [https://www.nature.com/articles/s41598-025-26337-x]</td><td>The information highlighted was provided by an uncited researcher synthesis: “- Attention mechanisms within transformer-style architectures generate saliency maps that highlight which market variables (e.g., volatility, momentum, macro indicators) the network focuses on at each rebalancing step, supporting visual debugging and regime-specific analysis.&#x27;</td></tr><tr><td>Orchestrator – Uncited output</td><td>In 1992 he indicates Berkshire expects to keep most major holdings regardless of how they are priced relative to intrinsic value.</td><td>The orchestrator paraphrased a researcher note that cited the Berkshire 1992 shareholder letter but dropped the citation entirely. The researcher&#x27;s note read: “&quot;We will keep most of our major holdings, regardless of how they are priced relative to intrinsic business value [https://www.berkshirehathaway.com/letters/1992.html].&quot;</td></tr><tr><td>Orchestrator – Insufficient citations</td><td>It also documents tranching and average duration patterns: liquidity tranche around 10 months and investment tranche around 27 months (with non-tranched portfolios around 26 months). [.../Inaugural-RAMP-Survey-on-the-Reserve- Management-Practices-of-Central-Banks-Results-and- Observations.pdf]</td><td>The information highlighted was provided by a researcher synthesis with a different World Bank citation: “Tranching is widely used; the liquidity tranche has an average duration of 10 months, the investment tranche 27 months, while portfolios without tranches average 26 months duration, indicating higher risk tolerance ([.../IDU-f1aeb421-97fb-4a82-a223- 0ecbbe0bdbb8.pdf]).&quot;</td></tr><tr><td>Researcher – Insufficient citations</td><td>PwC US announced a three-year, $1 billion investment to expand and scale its AI capabilities, embedding GPT-class models such as OpenAI&#x27;s GPT-4/ChatGPT into internal workflows and client solutions; this represents roughly 0.5 % of annual fee income. [.../pwc-us-makes-billion-investment-in-ai- capabilities.html]</td><td>The cited PwC press release supports the $1 billion investment claim. The highlighted percentage figure is absent from that source and was provided by other input documents (a LinkedIn post and a PwC article) whose citations the researcher did not include.</td></tr></table>

Table 5: Examples for each mistake category produced by our sentence-level evaluation algorithm.

• Nvidia AI-Q (Nvidia, 2026): released under the Apache License 2.0.

• MS-Agent (ModelScope, 2026): released under the Apache License 2.0.

• TrajectoryKit (Lugoloobi, 2026): released under the MIT License.

## I System-specific Logic

Our evaluation framework (Section 3) requires extracting the inputs and outputs of each agent invocation from the system’s execution trace. Because AI-Q, MS-Agent and TrajectoryKit differ in their trace formats, agent organization, and citation conventions, each system requires a dedicated adapter that maps the raw trace into the shared data structures consumed by our algorithm. Despite these differences, the evaluation algorithm itself (Section 4.2) runs identically on all three systems: each adapter only needs to expose, per agent, the inputs and outputs. Below we describe the system-specific extraction logic; Table 8 provides a summary.

Agent naming. The agent names used in each system’s codebase differ from the abstract roles used in this paper (Figure 2). In AI-Q, what we call the “researcher” is internally named searcher-agent, and what we call the “searcher” corresponds to the Exa Search + Jina + Content Optimizer module. In MS-Agent, the orchestrator role is split between a reporter\_tool subagent (which writes the initial report draft) and the top-level orchestrator (which issues post-edits); our adapter attributes errors to the specific phase that introduced them. In TrajectoryKit, what we call a “researcher” is a conduct\_research subagent (internally a worker), and what we call its “searcher” is the orchestrator’s summarize\_webpage tool, which produces the synthesized per-page digests.

The searcher differs significantly between systems. In AI-Q the searcher is the <Answer> block produced by the Tavily search API itself (include\_answer="advanced") and summarizes all documents returned for one query. In MS-Agent it is a single user message, with no system prompt, sent to a separate, smaller model (gpt-5-nano) by the content optimizer inside the web-search tool, once per document, over at most 50K characters, and skipped entirely for pages shorter than 2K characters. In TrajectoryKit it is a system and user message pair sent to the same gpt-oss-20b that the agents themselves run on, once per document, over at most 200K characters, from inside the orchestrator’s summarize\_webpage handler. We nevertheless evaluate these nodes exactly as we evaluate agents, because each performs the same compression of retrieved documents whose citation behavior our framework measures.

## I.1 AI-Q

Trace parsing and event attribution. Our implementation adds code to to produce a flat sequence of numbered JSON event files (0000\_\*.json, 0001\_\*.json, . . . ), each tagged with a kind (tool\_call, tool\_result, or llm) and an agent identifier. This facilitates the analysis of the outputs.

Orchestrator inputs. The orchestrator’s inputs are all returned subagent outputs plus the source registry. Subagent outputs are recovered via two channels: the task tool result (the subagent’s final message) and orchestrator read\_file calls on files the subagent wrote. Our adapter tracks both and attributes each file read to the subagent that last wrote the requested path.

Researcher inputs. Each researcher’s inputs are the web-search tool responses it received, which contain XML-style <Document href="..."> blocks (verbatim snippets from source pages) and <Answer> blocks (synthesized by the Tavily search API). Our adapter splits these into separate evaluation targets: <Document> snippets are evaluated as extracted content (Section 4.2.2), while <Answer> blocks are evaluated as synthesized content (Section 4.2.1).

Searcher outputs. The web-search tool is Tavily, configured to return the top three results for a query together with an answer field (include\_answer="advanced"). The <Document> blocks are the per-result page content the service returns, and the <Answer> block is that answer field, produced once per query over the results returned for it. It is the only synthesizedsearcher output among the three systems that covers several documents at once rather than a single page.

Planner preprocessing. AI-Q’s planner emits a JSON plan rather than prose. Because our spanextraction and entailment prompts expect natural language, the adapter converts the JSON to Markdown: keys become section headers and values become body text. This conversion handles three observed JSON shapes (whole-text JSON, fenced code blocks, and concatenated JSON regions).

Source registry. The orchestrator can call get\_verified\_sources, returning a numbered list of all URLs verified during the run (formatted as [N] Title: URL). Each entry is treated as an extracted snippet, enabling the algorithm to detect when the orchestrator cites a URL based solely on its registry title rather than on full document content.

## I.2 MS-Agent

Trace parsing. For MS-Agent we produce a numbered event stream similar to AI-Q.

Reporter and orchestrator split. MS-Agent’s final report is produced in two phases: a reporter subagent writes the initial draft via write\_file, and the orchestrator issues incremental replace\_file\_contents edits.

Evidence note architecture. MS-Agent’s searchers produce structured evidence notes, each with a unique hex ID, content sections (title, content, contradicting evidence, summary), and a metadata list of source URLs with quality tiers. Downstream agents cite these notes using [Note: <hex>] markers rather than standard URL citations.

Researcher synthesis. Each researcher’s final response to the orchestrator contains a JSON synthesis with fields for status, task summary, findings, and a report. This synthesis cites the notes the searcher authored using [Note: <hex>] markers.

Searcher outputs. MS-Agent’s search pipeline uses Exa search followed by a content optimizer that produces a summary and key excerpts per URL. The optimizer runs inside the web-search tool as a single prompt to a separate, smaller model, given the page content alone and truncated to 50K characters; pages shorter than 2K characters bypass it entirely and are carried through verbatim in place of a summary. Key excerpts are evaluated as extracted snippets (Section 4.2.2), while summaries are evaluated as synthesized content (Section 4.2.1).

## I.3 TrajectoryKit

Trace parsing and event attribution. For TrajectoryKit we produce a numbered event stream similar to AI-Q.

Orchestrator inputs. The root orchestrator is given a restricted, virtual tool set: it can delegate a research task (conduct\_research), rewrite its draft (refine\_draft), re-read earlier drafts, publish the draft (research\_complete), think, and summarize a single known URL (summarize\_webpage). It never searches or fetches documents itself. Its inputs are therefore the final responses of its researcher subagents, delivered as conduct\_research tool results, together with the page digests returned by its own summarize\_webpage calls. Unlike MS-Agent, there is no shared file system and no source registry, so this tool-result channel is the only route by which subagent content reaches the orchestrator.

Researcher inputs. Each conduct\_research subagent receives a fresh context and the full tool set: web search plus the fetch tools fetch\_url, read\_page, read\_pdf, extract\_tables and fetch\_cached. The researcher’s inputs are the tool responses it received. A researcher’s output does not have [N] markers, so it is attributed structurally to the URLs that researcher surfaced through its own search and fetch results.

Searcher outputs. Searcher output takes two forms. The first, search\_web, is Google through the Serper API, returning a numbered listing of title, URL and snippet per result. We do not treat these entries as an evaluation target, because unlike the snippet leaves of AI-Q and MS-Agent we don’t have access to the raw document that the system retrieved, so the test that scores extracted content has no reliable reference. The second form is the orchestrator’s summarize\_webpage tool, which fetches one URL and runs a single prompt over its content, truncated to 200K characters, on the same gpt-oss-20b the agents themselves run on, optionally steered by a focus argument.

Citation format. The orchestrator’s draft is a living document that each refine\_draft call replaces in full, and research\_complete publishes the latest version. On publication, TrajectoryKit rewrites the [N] markers in the report body into Markdown links of the form [[N]](url), resolved against the report’s ## Sources section (formatted as [N] Title: URL). Our adapter normalizes these links, and the full-width-bracket variant the model occasionally emits, back to plain [N] markers, after which the report format is identical to AI-Q’s and the same reference-section parsing applies.

Absence of a planner. TrajectoryKit’s closest analogue to AI-Q’s planner is a single chainanalysis call issued before the orchestrator loop starts, which decomposes the question into ordered subtasks. Because it runs before any retrieval, it consumes no documents and produces no citations, so it is not an evaluation target and TrajectoryKit contributes no planner row to our analyses.

Fallback writer. If the orchestrator exhausts its turn budget without publishing, TrajectoryKit writes the report outside the main loop: first by forcing a final answer within the orchestrator’s own context, and, failing that, by spawning a dedicated synthesizer subagent over the accumulated research data. We attribute the final report to the orchestrator in either case.

## I.4 Citation Attribution Modes

Agents use one of three citation attribution modes depending on their output format:

• Semantic: citations are identified per-sentence using the DEER classification scheme (Section B.3).

• Structural: citations are inherited from the structural context. For example, the source URLs of a referenced note.

• Extracted: the output is a verbatim excerpt of a single source, so the citation is the identity of the page it was taken from. Used for the searcher snippet leaves of AI-Q and MS-Agent, whose claims are checked by a verbatim test against the full crawled document rather than by an entailment test.

The mode follows from the output format rather than from the agent’s position in the information flow, so the same role can be attributed differently across systems. For example, AI-Q’s researcher notes carry explicit [N] markers that resolve to URLs, so they can be classified per sentence. An MS-Agent evidence note instead lists its sources as note metadata rather than inline, so each of its sentences inherits that note’s source list. Similarly, in TrajectoryKit there are no inline markers, so sentences inherit the union of the URLs that researcher surfaced through its own search and fetch calls. Structural attribution is the conservative choice in both cases: crediting a sentence with every source its author had at hand errs toward finding support rather than toward flagging a citation error.

## I.5 The Citation of Notes in MS-Agent

Unlike AI-Q, which uses standard [N] citation markers that map directly to URLs, MS-Agent’s intermediate outputs cite structured evidence notes using markers such as [Note: 02e74b]. Each note is a structured object with content sections and a list of source URLs with quality tiers. For example, a searcher synthesis might contain:

Berkshire stresses conservative   
leverage, large cash/float cushions,   
and a decentralized structure trusting   
operating managers. [Note: 02e74b]

These note markers are not standard URL citations and would be meaningless to end users reading the final report. Notes in turn have structured source metadata (e.g., {"title": ..., "sources": [{"url": "..."}]}).

Our adapter handles note citations differently depending on their position in the information flow:

Intermediate agents. For the reporter and searcher synthesis, citing a note is considered acceptable. The downstream orchestrator has access to the notes and their source URLs, and is expected to resolve note citations into proper URL citations before producing the final report.

Final report. If the final report cites a note marker, we drop this citation because it carries no meaning for the reader. This could result in citation-related errors if the note did support the content.

Conflict objects. MS-Agent’s reporter may also cite conflict objects using [conflict\_<hex>] markers. Each conflict records a discrepancy between sources with a description, resolution, and list of evidence note IDs. The adapter resolves conflict citations through the referenced notes’ source

URLs, adding one additional level of indirection to the resolution chain.

![](images/05b5238b61b7567d4e8dcec229df710533f2d877639bbf1a1974f56fc10e87a9.jpg)  
Figure 10: Citation classification system prompt, adapted from the DEER citation classification scheme of Han et al. (2026). The original tabular formatting (a markdown table) is rewritten here as a list for column compatibility, but the content is unchanged. The report and target sentence are provided in a separate user message.

![](images/19387cf7858d73b1ba204a97b7e93c6be5acaf6f9a5d33a8a9126a31fb02a76d.jpg)  
Figure 11: Need-citation prompt, adapted from LongCite (Zhang et al., 2025).

<table><tr><td>Agent (verdict)</td><td>Output text with citations</td><td>Explanation</td></tr><tr><td colspan="3">Chain 1: faithful propagation of a search-tool fabrication (searcher → researcher → orchestrator). The United States Space Force has formalized its cislunar</td></tr><tr><td>Searcher – Hallucination</td><td>effort by creating a Cislunar Coordination Office [...] officials have said the office will serve as the central hub for coordinating cislunar programs across the service and will ensure that space-situational-awareness data [...] is shared with NASA, other Department of Defense components, allied partners, and civilian space stakeholders, and this commitment was underscored by [...] the Oracle-M hot-fire test [...] which confirmed that the</td><td>The searcher&#x27;s answer summarizes the pages retrieved for the query, but the highlighted data-sharing and database-integration commitments appear in none of them.</td></tr><tr><td>Researcher – Inherited</td><td>resulting data will be integrated into joint databases and disseminated to national-security and civil-space users. The office will integrate SSA data into joint databases and disseminate the information to both national-security and civil-space users to enhance safety, security, and mission planning. [.../space-force-sets-up-office-to-coordinate-cislunar- programs/] [.../oracle-m-hot-fire-test-a-major-milestone-in-cislunar-</td><td>The researcher&#x27;s note restates the fabricated commitment and attaches the URLs that the searcher provided for this claim.</td></tr><tr><td>Orchestrator – Inherited</td><td>space-situational-awarenes] Public reporting indicates the U.S. Space Force has stood up a cislunar coordination function and is advancing systems (e.g., Oracle-M) aimed at monitoring and tracking cislunar objects, with plans to integrate tracking data into shared SSA databases. [.../space-force-sets-up-office-to-coordinate-cislunar- programs/] [.../oracle-m-hot-fire-test-a-major-milestone-in-cislunar- space-situational-awarenes] [.../space-force-sets-up-cislunar-coordination-office-to- focus-beyond-earth-orbit/]</td><td>The final-report sentence faithfully summarizes the researcher&#x27;s notes and propagates their citations, so the orchestrator&#x27;s local test passes as well. The fabricated claim reaches the reader carrying citations to real articles that do not state the information.</td></tr><tr><td colspan="3">Chain 2: abstraction masks a researcher fabrication (researcher → orchestrator).</td></tr><tr><td>Researcher – Hallucination</td><td>Then enable scheduled autoscaling for the cluster (gcloud container clusters update CLUSTER_NAME --enable-scheduled-autoscaling). Create or modify a schedule using gcloud container node-pools update POOL_NAME--add-chedule=start_time=HH:MM, end_time=HH:MM,min-nodes=MIN,max-nodes=MAX(or via console). The schedule can be daily or weekly. [.../kubernetes-engine/docs/concepts/cluster-autoscaler]</td><td>The researcher invents an entire GKE “scheduled autoscaling&quot; workflow, including command-line flags that do not appear in the cited official documentation, and attaches the real cluster-autoscaler documentation as its citation.</td></tr><tr><td>Orchestrator – Inherited business windows.</td><td>GKE: scheduled bounds on node pools via managed capabilities; ensure CA is enabled and schedules reflect [.../kubernetes-engine/docs/concepts/cluster-autoscaler] [.../kubernetes-engine/docs/how-to/cluster-autoscaler]</td><td>The orchestrator abstracts the fabricated procedure into a hedged, higher-level claim that is fully supported by the researcher&#x27;s notes. The abstraction propagates the fabrication, masks it behind vaguer language, and keeps its citations.</td></tr><tr><td colspan="3">Chain 3: unsupported statistics travel verbatim with their citation (researcher → orchestrator). Researcher- According to the OECD 2023 Annual Survey, infrastructure The highlighted figures were available to the represents roughly 6–7 % of assets for large researcher from other input documents, but the</td></tr><tr><td>Insufficient citations Orchestrator – Inherited</td><td>public-pension-related investors, with public pension reserve funds allocating about 9 % of their portfolios to infrastructure. [.../long-term-investing-of-large-pension-funds-and-public- pension-reserve-funds-2023_042fb731/c690ccc3-en.pdf] The OECD 2023 survey reports infrastructure at roughly 6-7% of assets for large public-pension-related investors and about 9% for public pension reserve funds. [.../long-term-investing-of-large-pension-funds-and-public- pension-reserve-funds-2023_042fb731/c690ccc3-en.pdf]</td><td>note cites only the OECD report, which does not support them. The figures and their citation travel together across the compression boundary into the report. Because the orchestrator&#x27;s sentence is faithful to the researcher note it cites, only recursing into the researcher reveals that the attached citation never</td></tr></table>

Table 6: Representative error-propagation chains produced by our system-level tracing evaluation on AI-Q, illustrating how error types interact across agents.

![](images/8c5a9cfba6b7f5f6ca72fb9a05cb815cdd5e70a59d39ece6e001e05beafdba09.jpg)  
Figure 12: Verbatim copy of the human annotation guidelines shown to annotators (see Section 4.3). The per-task worked example is omitted for space. The task was presented in a google doc with three tabs: the guidelines presented above, the output that the agent created, and the inputs that the agent received.

![](images/a73d1d98875fc1f49902435a6d47bd69be0f74721a638d3481d90c02a8ad1964.jpg)  
Figure 13: Agent-level error distribution for MS-Agent. The counterpart for Nvidia AI-Q is Figure 4.

![](images/45f827a94aaac977a066e5699ec6775e38e0951b0b17e71470419b1df9f6610a.jpg)

![](images/be80676e2e45c6947026affab886fece3a578b014fed010e8203d4cd0819a06f.jpg)

![](images/011b0b7c0ba4337543bb38ea9c659ef6db57936bb80dfde0b2b4f7a25430b8f5.jpg)  
Figure 14: Agent-level error distribution for TrajectoryKit. Unlike the other two systems, hallucinations dominate at every agent rather than giving way to citation errors deeper in the information flow. The counterpart for Nvidia AI-Q is Figure 4.

![](images/e2a7b153c88c8b9313e53ddabb0f41bd4c00e379c869594d66c8a67cda11e124.jpg)  
Figure 15: System-level error distribution for MS-Agent. The counterpart for Nvidia AI-Q is Figure 5.

![](images/97663183b85b63d377b4d5cf6349001f88942ae48749fec1bb7a530b9b377765.jpg)  
Figure 16: System-level error distribution for TrajectoryKit. Every confirmed final-report error is attributed to the orchestrator, and 95% of them are hallucinations. The counterpart for Nvidia AI-Q is Figure 5.

<table><tr><td>System</td><td>Agent</td><td>LLM</td></tr><tr><td rowspan="5">Nvidia AI-Q</td><td>Orchestrator</td><td>GPT-5.2</td></tr><tr><td>Planner</td><td>GPT-5.2</td></tr><tr><td>Researcher</td><td>Nemotron Nano 30B</td></tr><tr><td>Tavily Snippets</td><td>(Tavily API)</td></tr><tr><td>Tavily Answer</td><td>(Tavily API)</td></tr><tr><td rowspan="4">MS-Agent</td><td>Orchestrator</td><td>GPT-5</td></tr><tr><td>Reporter</td><td>GPT-5 mini</td></tr><tr><td>Researcher</td><td>GPT-5 mini</td></tr><tr><td>Searcher</td><td>GPT-5 nano</td></tr><tr><td rowspan="3">TrajectoryKit</td><td>Orchestrator</td><td>gpt-oss-20b</td></tr><tr><td>Worker</td><td>gpt-oss-20b</td></tr><tr><td>Summarize Webpage</td><td>gpt-oss-20b</td></tr></table>

Table 7: LLM used by each agent in the three opensource DR systems we evaluate. Nemotron Nano 30B refers to NVIDIA (2025). Tavily Snippets and Tavily Answer are calls to a search API rather than direct prompts to an LLM.

![](images/831fe7678517d5a0c71726ffa5045fe10031a78e2bd8dc5ddbc446ce44213bff.jpg)  
Figure 17: System-level error distribution for Nvidia AI-Q after applying the “do NOT use information which is not backed by citations.” prompt change. Compare against Figure 5: the orchestrator’s uncited output bucket is reduced from 48% to 13%.

<table><tr><td>System</td><td>Agent</td><td>Inputs (for span extraction)</td><td>Outputs (evaluation tar- gets)</td><td>Cit. mode</td></tr><tr><td rowspan="5">AI-Q</td><td>Orchestrator</td><td>Researcher notes, planner plans,</td><td>Final report</td><td>Semantic</td></tr><tr><td>Researcher</td><td>source registry &lt;Document&gt; snippets + &lt;Answer&gt;</td><td>Synthesized note with [N]</td><td>Semantic</td></tr><tr><td>Planner</td><td>blocks Same as Researcher</td><td>cit. JSON plan → Markdown</td><td></td></tr><tr><td>Searcher (snippets)</td><td>Raw crawled documents</td><td>&lt;Document&gt; blocks</td><td>Extracted</td></tr><tr><td>Searcher (answers)</td><td>Raw crawled documents</td><td>&lt;Answer&gt; blocks</td><td>Structural</td></tr><tr><td rowspan="5">MS-Agent</td><td>Orchestrator</td><td>Reporter snapshots, analyses, con-</td><td>Final report (post-edit)</td><td>Semantic</td></tr><tr><td>Reporter</td><td>flicts Syntheses, notes, analyses</td><td>Report draft</td><td>Semantic</td></tr><tr><td>Researcher (notes)</td><td>Summaries + key excerpts per URL</td><td>Evidence notes</td><td>Structural</td></tr><tr><td>Searcher (synthesis)</td><td>Raw crawled documents</td><td>Synthesis JSON</td><td>Structural</td></tr><tr><td>Searcher (snippets)</td><td>Raw crawled documents</td><td>Synthesis JSON</td><td>Extracted</td></tr><tr><td rowspan="3">TrajectoryKit</td><td>Orchestrator</td><td>Researcher final responses, summarize_webpage digests</td><td>Final report</td><td>Semantic</td></tr><tr><td>Researcher</td><td>Search snippets + fetched page bod- ies</td><td>Final response to orchestra- tor</td><td>Structural</td></tr><tr><td>Searcher (answers)</td><td>One fetched page</td><td>summarize_webpage di- gest</td><td>Structural</td></tr></table>

Table 8: Per-agent extraction summary. Inputs are the agent invocation inputs provided to the span-extraction step of our algorithm (Section 4.2); outputs are the evaluation targets; citation mode indicates how citations are identified (semantic = per-sentence DEER classification; structural = inherited from context; extracted = source URL identity).