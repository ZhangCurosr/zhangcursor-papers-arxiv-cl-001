# Measuring and Reducing Unusable Replies of LLM Agents

Yilin Jiang<sup>1,2</sup>, Fei Tan<sup>1</sup>\*, Xuanyu Yin<sup>1</sup>, Guanchu Wang<sup>3</sup>, Xia Hu<sup>3</sup>, Yutao Yue<sup>2,4</sup> <sup>1</sup>East China Normal University, Shanghai, China

<sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China <sup>3</sup>Shanghai Artificial Intelligence Laboratory, Shanghai, China <sup>4</sup>Institute of Deep Perception Technology, JITRI, Wuxi, China ftan@mail.ecnu.edu.cn

## Abstract

People delegate multi-step work to LLM agents and read only the reply that the agent hands back. Current evaluation scores task completion and the faithfulness of reports, and it leaves open whether the requester can understand and use the reply. We call this property reply usability and measure it with a reader model that takes the requester’s position and tests whether the reply can be interpreted from what the requester has seen, is in the requester’s language and answers the request. Its agreement with the majority of five annotators lies within the range of the individual annotators. Across eight models from seven families, 54% of 480 replies fail, and the best model still fails on 30%. Failed replies mostly point to material the requester has not seen or use labels that the agent coined during its work. Stating explicit requirements for the reply reduces the failure rate from 0.52 to 0.43 together with errors and omissions, and a secondary model that rewrites the reply reduces it to 0.36, which human annotators confirm in a blind comparison. Reply usability can thus be scored alongside task success and improved with one appended message or one call to a secondary model.

## 1 Introduction

People increasingly hand multi-step work to LLM agents and read only what the agent sends back. An agent that analyzes data, runs experiments or edits a codebase issues dozens of tool calls, writes files and keeps notes, and the requester sees none of this. The message that the agent writes when it returns to the requester, which we call its reply, is the requester’s only view of what was done. A requester who cannot read the reply has to recover the result by asking again or by reading the agent’s records. Figure 1 shows such a reply. It reports results under a configuration key and two reference identifiers that the agent met during its work and that the requester has never seen.

<table><tr><td></td><td>Request We want a small (～7B) model to get close to large-model quality on long-document QA, mainly through retrieval and context organization [...] Start by surveying the area and give me a few feasible directions.</td></tr><tr><td>Agent&#x27;s work</td><td>search_papers ×5, list_files, read_file read_paper ×10 → R07, R09, .. run_study ×22 → F1 by domain and budget write_file ×2 → notes with reader_size: 70B</td></tr><tr><td>Reply</td><td>[...] when I passed reader_size: 70B1 it was silently ignored [...] HyDE is +2.1 on science but -2.2 on legal (exactly R072&#x27;s stated failure mode on exact-match lookups). [...] Clean inverted-U, peak ~8k [...] Replicates and refines R093 (~6k).</td></tr><tr><td></td><td></td></tr><tr><td>Reader model Question</td><td>Answer</td></tr><tr><td>1 What does “reader_size: 70B&quot; refer to? 2 Which paper [...] does R07 refer to?</td><td>none none</td></tr><tr><td>3 Which paper [...] does R09 refer to?</td><td>none</td></tr><tr><td>3 information gaps</td><td></td></tr></table>

Figure 1: A reply of Claude Opus 5 on long-document question answering and its reading by the reader model. The requester does not see the agent’s work of 41 tool calls. The reader model finds no answer to its questions about the three marked names.

Current evaluation does not measure whether the requester can understand and use the reply. Agent benchmarks score whether the task was completed (Jimenez et al., 2024; Zhou et al., 2024; Mialon et al., 2024; Liu et al., 2024), and evaluation of generated text scores whether the text is faithful to its source (Krysci ´ nski et al. ´ , 2020; Maynez et al., 2020). A reply can satisfy both and still be unusable. It may report the result under a run name that the agent invented, cite a finding that appeared only in a tool output, or list changes without answering the question that was asked. Usability for the requester is therefore a separate property from task success and faithfulness.

We define a reply as failed when the requester cannot interpret it from what they have seen, when it is written in the wrong language, or when it does not answer the request. We measure failures with a reader model that takes the requester’s position. The reader sees what the requester has seen, generates the questions that a requester would need answered, and searches for their answers in the reply and in this context (Figure 1, bottom). On a gold set labeled by five annotators, the reader agrees with the majority label at κ = 0.60, within the range that individual annotators reach against the other four.

With this measure we study eight current models from seven families in multi-session collaborations on research and business projects. More than half of the replies fail, 54% of 480, and every family is affected, with the highest rates for replies that follow extended tool use. The failed replies are mostly written in the terms of the agent’s private work. They point to material the requester has not seen or use labels that the agent coined, and more than half of these labels first occur in the agent’s own files, project memory or tool-call arguments. Explicit requirements stated at the point of the reply reduce the failure rate from 0.52 to 0.43 together with errors and omissions, whereas a notice about the reader does not. A secondary model that rewrites the reply under the same requirements reduces the rate from 0.52 to 0.36, and annotators who judge default replies and rewrites blind confirm the reduction. Under the requirements, an open-weight model produces a usable reply within sixteen samples for almost every context.

Our contributions are as follows. First, we define the usability of a reply from the requester’s position and provide a reader model that measures it at the agreement level of an individual human annotator. Second, we show that unusable replies are common in every model family we evaluate, and we identify what the requester is missing, which is mostly material and vocabulary from the agent’s private work. Third, we provide two inexpensive remedies, a requirements message and a rewriting prompt, and we will release the collaborations, the reader and its judgments.

<table><tr><td>Line of work</td><td>Judged against</td><td>Property</td></tr><tr><td>Agent benchmarks</td><td>environment state</td><td>task success</td></tr><tr><td>Self-reports of</td><td>the trajectory</td><td>faithfulness,</td></tr><tr><td>agents</td><td></td><td>completeness</td></tr><tr><td>Hand-offs between</td><td>the next agent&#x27;s</td><td>downstream</td></tr><tr><td>agents</td><td>task</td><td>success</td></tr><tr><td>Decontextualization</td><td>the empty context</td><td>interpretability of a sentence</td></tr><tr><td>This work</td><td>what the requester has seen</td><td>usability of the</td></tr></table>

Table 1: What related lines of work evaluate.

## 2 Related Work

Common ground and decontextualization. Table 1 places our measure among the evaluation targets of related work. Conversation relies on common ground that the participants build together (Clark and Brennan, 1991). Speakers are expected to design an utterance for its audience (Clark and Murphy, 1982), and people do so imperfectly (Horton and Keysar, 1996; Camerer et al., 1989). Shaikh et al. (2024) show that LLM generations presume common ground and contain fewer grounding acts than human dialogue, and Shaikh et al. (2025) measure the resulting failures in logged human–LLM conversations. Decontextualization asks a related question of written text. Choi et al. (2021) define when a sentence can be interpreted outside its document, and Newman et al. (2023) decontextualize snippets of scientific papers by generating and answering the questions a reader would have (see also Gunjal and Durrett, 2024). We adopt this question-answering formulation and change the reference context from the empty context to what the requester has seen.

Reports of agents about their own work. Bansal et al. (2024) name conveying information from the agent to the user as an open challenge, and Shao et al. (2026) include inadequate summaries after executed actions as one category in an error taxonomy of human–agent collaboration. Tang et al. (2026) and Kraishan and Jitkajornwanich (2026) find that self-reports of coding agents misstate or leave out executed actions, and Huang et al. (2026) study whether agents disclose problems in their work. These studies judge the report against the trajectory, with annotators or checkers who have seen the trajectory, and they measure faithfulness and completeness. We measure a complementary property, whether a requester who has not seen the trajectory can understand and use the report, along with faithfulness. A further line of work studies hand-offs of coding tasks between agents and their effect on quality and cost (Ganz et al., 2026; KC and Budathoki, 2026), misalignment between agents in multi-agent systems (Cemri et al., 2025), and the loss of earlier constraints in context compaction (Chen, 2026). In our setting the reader is a person, and the outcome is whether that person can read the message from what they have seen.

Instructions about the audience. Instructions that tell a model about its audience have shown limited effect on other tasks, as have personas in system prompts (Zheng et al., 2024). Shaikh et al. (2024) report that prompting for grounding acts raises their frequency and yields at most minimal gains in agreement with human behavior. Christian and Mazor (2026) find that instructing a model to ignore information it has seen does not remove the influence of that information on its judgments. Wang et al. (2026) report that reminders recover little of the error in the progress reports of agents, and Chen (2026) reports that provenance prompts in the style of Spotlighting do not reduce constraint violations after context compaction, whereas pinning the constraints does. We compare a notice about the reader with a concrete specification of what the reply must contain, in the same context and with the same information (Section 6).

Readability and faithfulness. Rewriting text for a reader can change its content. Lay summarization (Goldsack et al., 2022) and personalized jargon identification (Guo et al., 2024) address readers outside a text’s field, and Devaraj et al. (2022) show that text simplification introduces factual errors and omissions. We therefore check the replies of every writing condition and every rewrite against the work record. Other work changes the context in which a text is written. Zhu et al. (2026) report gains in report quality, scored against references, from separating the writing of a research report from the search that precedes it, and Laban et al. (2026) and Chen et al. (2026) show for task performance and for error correction that the form in which information is presented to a model matters when the information itself is held fixed. For the replies of agents, our writing conditions separately vary the context in which the reply is written and the requirements that it is given.

## 3 Measuring Reply Usability

## 3.1 Reply Usability

We study the message an agent sends back to the person who asked for the work. In a session, a requester gives an instruction, the agent works through a sequence of tool calls, and the agent then writes a message without tool calls. We call this message the agent’s reply. The reply is often the only part of the session the requester reads, so the requester has to be able to read it without consulting anything else.

The requester and the agent see different parts of the session. The requester context R contains the requester’s own messages, the files the requester provided, and the agent’s earlier replies. The agent also sees its private context, which we define as everything outside R that the agent produced or received during the work. It comprises tool calls and tool outputs, documents and notes in the agent’s workspace, summaries written during context compaction, and memory loaded at the start of a new session. The requester has seen none of the private context.

We call a reply usable when the requester can understand it from R alone and it answers what was asked, which we specify with three criteria. First, the reply can be interpreted from R. We adapt the definition of decontextualization by Choi et al. (2021), who call a sentence decontextualized when it can be interpreted in an empty context with its truth-conditional meaning preserved, and we replace the empty context with R. An information gap is an expression of the reply whose interpretation requires information that neither the reply nor R provides, such as a run name, a configuration label, or a finding that appeared only in a tool output. Such an expression presupposes common ground (in the sense of Clark and Brennan, 1991; Shaikh et al., 2024) that the requester lacks, and a reply meets the first criterion when it has no information gap. Second, the reply is written in the requester’s language. Third, the reply answers the request. A reply that lists the changes the agent made and does not state the result the requester asked for gives less information than the exchange requires (Grice, 1975). A reply fails when it violates at least one criterion, and thefailure rate of a set of replies is the proportion that fail.

Usability in this sense is distinct from task success and from faithfulness. An agent can complete the task and describe its work accurately in a reply that the requester cannot interpret, and a reply that the requester can read with ease can still misreport the work. We therefore measure faithfulness to the work alongside usability (Section 3.3).

## 3.2 Measuring Usability with a Reader Model

We measure failures with a reader model that takes the position of the requester. The reader knows the requester’s field, the request and the requester’s files, it has read R, and it has not seen the agent’s work. It reads at an ordinary pace and accepts only what the text states.

The reader follows question-answering formulations of decontextualization (Newman et al., 2023) and of summary consistency (Wang et al., 2020; Durmus et al., 2020; Scialom et al., 2021) and runs in two steps. In the first step it generates the questions that the requester would need answered to understand the reply. Each question is tied to a span of the reply and typed as a definition, a reference or a background question. In the second step the reader looks for the answer to each question in the text around the span, then in the whole reply, and then in R, and it must quote the passage that answers the question. A question without a quoted answer marks an information gap. The first step also returns the sentence of the reply that answers the requester’s latest message, and the language of the reply is determined from its characters. A reply fails if it has at least one information gap, is written in a language other than the requester’s, or contains no sentence that answers the request.

The reader uses Gemini 3 Flash (preview). Comparisons between writing conditions and between a reply and its rewrite are paired within the point of the collaboration at which the reply is written. We report 95% percentile intervals from a bootstrap that resamples collaborations. Appendix A gives the prompts and settings.

Agreement with human annotators. The reader’s agreement with the human majority lies within the range of the individual annotators (Figure 2a). We validated the reader on a gold set of 300 replies. Five annotators with research backgrounds in five fields labeled every reply. The annotators read each collaboration from its beginning, saw only R, labeled each reply as readable or failed, and marked the spans that stopped them. The annotators agree with each other at Fleiss’ $\kappa = 0 . 6 3$ and the reader agrees with their majority label at Cohen’s $\kappa = 0 . 6 0$ with an accuracy of 82%.

![](images/fd03a96b9e9c3e5e6b3b66010d8fa6d95f87736d21b7e3e8339279bbb6a7c5cf.jpg)

![](images/4627c45587331279e0e5315b296e7407e53ef7f9bdf6fc1c44773dca138b477e.jpg)  
Figure 2: (a) Cohen’s κ with the majority label of the gold set, which for an annotator is the majority of the other four. (b) Replies failed by the majority, by the number of gaps the reader finds.

The reader’s verdict tracks what the annotators experience. The share of replies that the majority fails rises with the number of information gaps the reader finds, from 26% for replies without a gap to 71% for replies with one gap and 94% for replies with two or more (Figure 2b), so the majority already fails most replies with a single gap. Of the gaps the reader finds, 68% lie on a span that at least one annotator marked. Annotators also take longer over the replies that the majority fails, with a median reading time of 85 seconds against 50 seconds for those it passes (Appendix B).

## 3.3 Measuring Faithfulness with a Checker Model

We check faithfulness to the work with a checker model. The checker receives the reply and the work record, which is the agent’s context at the point of the reply, including its tool calls and tool outputs. It lists the factual errors, which are statements of the reply that the record contradicts or does not contain, and the omissions, which are key results of the work that the reply leaves out. We report the number of errors plus omissions per reply. The checker uses the same model as the reader, and Appendix A gives its prompt.

## 4 Experimental Setup

## 4.1 Collaborations

We study replies in multi-session collaborations in which an agent works for a scripted requester. The requester gives brief directives and asks conceptual questions, as a researcher or an analyst does when delegating work. After each requester message the agent calls tools freely, and its first message without tool calls is its reply. A collaboration has 15 requester messages in three sessions. When the context exceeds a length threshold, the history is replaced by a summary that the agent writes, and each new session starts from the documents in the agent’s workspace, which are loaded as project memory.

<table><tr><td>Project</td><td>Type</td><td>Language</td></tr><tr><td>Detecting mislabeled examples in crowd-labeled datasets</td><td>research</td><td>Chinese</td></tr><tr><td>Long-document question answering with a small model and retrieval</td><td>research English</td><td></td></tr><tr><td>Price change for a subscription product</td><td></td><td>business Chinese</td></tr><tr><td>Redesign of the onboarding flow of business English a software product</td><td></td><td></td></tr></table>

Table 2: The four projects of the collaborations.

The collaborations cover four projects, two in research and two in business (Table 2). In the research projects the agent searches and reads a literature index, runs simulated studies whose design it specifies, and interprets the results. In the business projects it consults a reference library and evaluates designs with a simulator. The system prompt is a generic agent instruction and does not mention who reads the replies. Because the environments are scripted, every model works on the same projects and receives the same requests. Appendix C gives the tools, the system prompt and the limits.

## 4.2 Models

We evaluate eight current models from seven families. The set contains Claude Opus 5, GPT-5.6 Sol, DeepSeek V4.1 Flash, DeepSeek V4 Pro, Kimi K2.6, GLM-5.3, Qwen3.6 Plus and MiniMax M3, and it includes closed and open-weight models. Each model works on the four projects, which gives 32 collaborations and 480 replies.

## 5 Replies Fail Across Model Families

## 5.1 Prevalence

More than half of the replies fail, and the failure appears in every family. Figure 3a reports the failure rate of each model when it works with the generic system prompt and no instruction about its replies. Of the 480 replies, 258 fail, a rate of 0.54. Seven of the eight models fail on at least 48% of their replies, and the rate ranges from 0.30 for GPT-5.6 Sol to 0.82 for Claude Opus 5. By project, the rate ranges from 0.41 for subscription pricing to 0.66 for the detection of mislabeled examples (Appendix C).

All seven families are affected, closed and openweight models alike, and the highest rate belongs to a closed frontier model. The system prompt asks for no particular style of reply, so these rates describe what a requester receives by default.

Most failures are information gaps. Of the 258 failed replies, 222 contain at least one question that the requester cannot answer from R, 73 do not answer the request, and 17 are written in the wrong language, where a reply can meet more than one of these conditions. At least one information gap occurs in 46% of all replies and in 20% to 78% of the replies of a single model.

Gaps are frequent within a failed reply and occur early. Of the 258 failed replies, 114 contain two or more information gaps, and a reply with a gap contains 2.1 gaps on average and up to ten. In 128 of the 222 replies with a gap, the first gap begins in the first third of the reply, so the requester is stopped before reaching most of the content.

Failures concentrate in the replies that report the most work (Figure 3b). Replies written after three or more tool-calling turns fail at a rate of 0.70, against 0.48 for replies written without a tool call and 0.40 for replies after one or two turns. In six of the eight models, replies after three or more turns fail more often than replies without a tool call. The two models with the highest failure rates, Claude Opus 5 and DeepSeek V4.1 Flash, are also the two with the most tool-calling turns before a reply (Appendix C). The replies that carry the results of extended autonomous work are thus the ones a requester is least likely to be able to use.

## 5.2 What the Requester Is Missing

Most failed replies rely on material that exists only in the agent’s private context. For each failed reply the reader returns the questions it could not answer, and each marks a gap. The 258 failed replies contain 475 gaps, which a classifier assigns to four categories, and two further kinds of failure come from the reader’s verdict (Table 3). On 100 gaps that two annotators labeled independently, the classifier agrees with the annotators at a mean κ of 0.68, the level at which the two annotators agree with each other (Appendix F).

Pointers to unseen material and coined labels each occur in more than 40% of the failed replies. In six of the seven families, at least a third of the failed replies contain a pointer to unseen material and at least a quarter contain a coined label. For Claude Opus 5, the model with the highest failure rate, the two shares are 61.2% and 65.3%. GPT-5.6 Sol, the model with the lowest rate, has a different profile. None of its 18 failed replies contains a coined label, 44.4% leave an established term unexplained and 38.9% do not answer the request. The work itself therefore does not force working labels into the reply.

![](images/f5b1dc2bda5d7e65c96ba7924e7a3862283ddad543ff0d2477a2c0385e588833.jpg)

![](images/dad813a27378e175820594ad6ce7521890796800f1eed6dd7d02dacd8f034930.jpg)

![](images/cef94215632ff7d95a1a5262e084363ee8182964dda98f579c15080b88a781c7.jpg)

Figure 3: (a) Failure rate of each model with 95% intervals, divided into replies with an information gap and other failures. (b) Failure rate by tool-calling turns before the reply. (c) First occurrence of the coined labels in the agent’s context, where red marks the agent’s files, project memory and tool-call arguments.
<table><tr><td>Kind of failure</td><td>What the requester lacks</td><td>Example span and the reader&#x27;s question</td><td>Failed replies (%)</td></tr><tr><td>Pointer to unseen material</td><td>a file section, an index entry or the output of a run that the requester has not seen</td><td>R03. Which paper or reference does R03 refer to?</td><td>46.1</td></tr><tr><td>Coined label</td><td>the meaning of a name, number or short form that the agent introduced during the work</td><td>crn_hierbm25_L. What does &quot;crn_hierbm25_L&quot; refer to?</td><td>42.2</td></tr><tr><td>Unexplained term</td><td>the meaning of an abbreviation or term that is established in the field</td><td>CUPED adjustment. What does CUPED stand for in the context of variance reduction in A/B testing?</td><td>23.6</td></tr><tr><td>Missing background</td><td>a premise that the reply depends on and does not state</td><td>confirm whether the difference between 80.2 and 76.0 is stable. At what sample size [...] was the comparison of 80.2 and 76.0 run?</td><td>1.6</td></tr><tr><td>Request not answered Wrong language</td><td colspan="3">the result or answer that the latest message asked for a reply in the requester&#x27;s language</td></tr></table>

Table 3: Kinds of failure among the 258 failed replies, where a reply can have several kinds. The upper four are the categories of information gaps, shown with a span of a reply and the reader’s unanswered question.

More than half of the coined labels were written down by the agent before it wrote the reply. For each coined label we search the agent’s context for the earliest message that contains it (Figure 3c). Of the labels, 24.2% first occur in a file that the agent wrote and 5.6% in the arguments of another tool call. A further 23.2% first occur in the project memory loaded at the start of a later session, which consists of documents that the agent wrote in earlier sessions. These three sources account for 53.0% of the labels.

These origins show that the vocabulary of a failed reply is the working vocabulary of the agent. During the work the agent names its runs, configurations and intermediate findings, and it records the names in its notes and memory. In the reply it continues to use these names and cites its sources by the identifiers under which it retrieved them, none of which the requester can resolve from R. Project memory extends this vocabulary beyond the session in which it arose, since 23% of the coined labels reach the reply from documents that the agent wrote in an earlier session. In the reply of Figure 1, the key reader\_size: 70B occurs in the agent’s notes, and R07 and R09 are the identifiers under which the literature index returned two papers.

## 6 Reducing Failures

## 6.1 Writing Conditions

We test interventions at 256 reply points of the collaborations of Section 5. At each point the reply is written again under every condition while the work up to that point stays fixed.

<table><tr><td>Condition</td><td>Written in</td><td>Added instruction</td><td>Failure rate</td><td>Reduction</td><td>Errors + omissions</td></tr><tr><td>Default</td><td>working context</td><td>none</td><td>0.52</td><td></td><td>1.41</td></tr><tr><td>Notice</td><td>working context</td><td>notice</td><td>0.50</td><td>0.027[–0.052, 0.112]</td><td>1.34</td></tr><tr><td>Requirements</td><td>working context</td><td>requirements</td><td>0.43</td><td>0.098 [0.030, 0.168]</td><td>1.07</td></tr><tr><td>Separate pass</td><td>new conversation</td><td>none</td><td>0.47</td><td>0.051 [-0.004, 0.109]</td><td>1.37</td></tr><tr><td>Fact list</td><td>new conversation</td><td>none</td><td>0.49</td><td>0.031 [-0.020, 0.081]</td><td>1.45</td></tr><tr><td>Separate pass with requirements</td><td>new conversation</td><td>requirements</td><td>0.42</td><td>0.102 [0.023, 0.181]</td><td>1.16</td></tr></table>

Table 4: The six writing conditions on 256 reply points, with 95% intervals for the reduction from the default.

Three conditions write the reply in the agent’s working context. The default condition adds no instruction. The notice condition adds to the system prompt a statement that the user has seen only their own messages and the agent’s earlier replies. The requirements condition appends a message at the point of the reply that repeats this statement and specifies what the reply must contain: an answer to the latest message first, what was done, the result, the next step, where the evidence can be checked, and a definition of every name the user has not seen. These items correspond to the kinds of failure of Section 5.2. The answer to the latest message addresses unanswered requests, and the definitions address coined labels and unexplained terms.

Three conditions write the reply in a separate pass in a new conversation. In the first the model receives R and the work record. In the second it receives R and a list of the results that it extracted from its own work. In the third it receives R, the work record and the requirements. Appendix D lists the prompts.

## 6.2 Explicit Requirements

Explicit requirements reduce the failure rate from 0.52 to 0.43, a reduction of 0.098 (Table 4). The rate is lower in seven of the eight models and in six of the seven families (Figure 4a). The reduction is largest for Claude Opus 5, the model with the highest default rate, whose failure rate falls from 0.72 to 0.53, and four further models improve by 0.12 or more.

The requirements also improve faithfulness. The checker lists 1.07 errors plus omissions per reply under the requirements against 1.41 by default. Both components decrease, errors from 0.73 to 0.53 per reply and omissions from 0.68 to 0.54. A notice about the reader has no such effect, with a failure rate of 0.50 and 1.34 errors plus omissions.

The reduction follows the requirements message, and the message works in the agent’s own context. The two conditions that include it bring the failure rate to 0.43 and 0.42, and none of the three conditions without it differs significantly from the default (Table 4). The difference between these two conditions is 0.004 [−0.069, 0.075], so a separate writing step adds no measurable reduction to one appended message, which needs no additional call.

## 6.3 Rewriting by a Secondary Model

A secondary model can repair a reply after it has been written. This setting applies when the agent’s prompt cannot be changed or the reply already exists. The rewriting model is DeepSeek V4 Pro. It receives the request, the work record and the agent’s default reply, and it rewrites the reply once under the requirements of Section 6.1. We compare each rewrite with the reply it replaces at the same 256 reply points.

Rewriting reduces the failure rate from 0.52 for the default replies to 0.36 for their rewrites, a reduction of 0.168 [0.101, 0.242]. The rate is lower in seven of the eight models and in six of the seven families (Figure 4a). The rewrite also reduces failures in the replies of the two closed models, from 0.72 to 0.47 for Claude Opus 5 and from 0.25 to 0.13 for GPT-5.6 Sol. Human annotators confirm the reduction. Five annotators labeled the default reply and its rewrite at 83 reply points under the protocol of Section 3.2, each seeing one version per point without knowing which. The share of annotators who fail a reply falls from 0.52 for the default reply to 0.32 for the rewrite, a difference of 0.201 [0.095, 0.321] (Figure 4b), and it is lower in seven of the eight models. On the 166 replies of this evaluation the reader agrees with the majority of annotators at κ = 0.60 (Appendix E).

Rewriting also reduces errors and omissions. The checker lists 1.41 errors plus omissions per default reply and 1.13 per rewrite, with errors going from 0.73 to 0.54 and omissions from 0.68 to

![](images/24f8905d3a3acd1f1e56b41f7ac306bd840372ce7b8f982f08276408ee9ffd9b.jpg)

![](images/741eefb8509abc218ba14baf9eae6891f39ec327c9a243b2d921b4b2071de55d.jpg)

![](images/9614d1c35ae669819e82b056d39da30852312628a3a7c94073d93b51cb77ebeb.jpg)  
Figure 4: (a) Failure rate by default, under the requirements and after the rewrite. (b) Default replies and rewrites at 83 reply points, judged by annotators and by the reader. (c) Share of reply points with a usable reply among k samples.

0.59. The rewriting model has the work record, so it can define a name or replace a pointer with the content it points to. In one reply of Claude Opus 5, for example, the default text states that “read\_file truncates at ∼8k characters”, and the rewrite states that “the file-viewing tool truncates long files at roughly 8k characters”. The rewriting model is open-weight, so a model that can be run locally repairs the replies of both closed models.

## 6.4 Sampling

Usable replies are within the model’s own output distribution. We measure this on Qwen3.6-35B-A3B, an open-weight model run as the agent in further collaborations of the same kind. For 100 reply points we draw 16 samples under the default condition and 16 under the separate pass with requirements. A sample is usable when the reader accepts it in two independent judgments, and pass@k is the share of points with a usable reply among k samples (Appendix G).

Under the requirements, pass@k is 0.52 for one sample, 0.84 for four and 0.97 for sixteen (Figure 4c). Under the default condition the values are 0.36, 0.65 and 0.81, so four samples under the requirements cover more points than sixteen samples by default. The model can therefore write a usable reply for almost every context, and selecting among a few samples is a direct route to further reduction.

## 7 Discussion

The results locate the failure at the point where an agent turns from working to reporting. During the work the agent builds a vocabulary of run names, configuration labels and identifiers of sources, and its notes and memory keep this vocabulary available. The reply is written from inside that state, so a requester who has seen none of it meets the names without their referents. A notice that the requester has not seen the work does not measurably lower the failure rate, whereas a statement of what the reply must contain does. The models can therefore write for the requester, and a notice of who reads does not by itself lead them to do so. Two consequences follow for the field. An evaluation that inspects only the final state of the environment credits work whose report the requester cannot use, and a reader model that takes the requester’s position supplies the missing check. In the design of agents, memory and compaction carry working names across sessions and into later replies, so the moment at which the agent returns to the requester deserves an instruction of its own, and a secondary model can enforce it where the agent’s prompt is fixed. Reply usability also bears on how LLMs shape human language. Widespread use of LLMs narrows the linguistic diversity of human writing (Sourati et al., 2026), and the vocabulary that agents carry into their replies is a further possible route for this influence.

## 8 Conclusion

The reply is the requester’s only view of an agent’s work, and current evaluation does not test whether the requester can understand and use it. We defined usability from the requester’s position and built a reader model that measures it at the agreement level of human annotators. Across eight models from seven families, more than half of the replies fail, mostly because they are written in the terms of the agent’s private work, and explicit requirements or a rewrite by a secondary model reduce the failures, which human annotators confirm for the rewrite. Reply usability can therefore be measured alongside task success and improved with one appended message or one call to a secondary model.

## Limitations

Our collaborations are scripted, which gives every model the same requests and records the agent’s private context that the analysis of gap origins requires. The reader represents a requester who knows the field of the request and has not seen the agent’s work, and the gold set validates it for this profile. The sampling analysis covers one open-weight model. Applying the measure to logged sessions of consenting users, to requesters with other backgrounds, and to further models and languages is the direct next step.

## Ethics Statement

All collaborations are synthetic and contain no personal data. The annotators of the gold set, of the evaluation of rewrites and of the gap categories are researchers who took part voluntarily, gave informed consent to the use of their labels, and received fair compensation. The annotations are used to validate the reader and the gap classifier and to evaluate the rewrites, and they are not released.

## References

Gagan Bansal, Jennifer Wortman Vaughan, Saleema Amershi, Eric Horvitz, Adam Fourney, Hussein Mozannar, Victor Dibia, and Daniel S. Weld. 2024. Challenges in Human-Agent Communication. Preprint, arXiv:2412.10380.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. Preprint, arXiv:2407.21787.

Colin Camerer, George Loewenstein, and Martin Weber. 1989. The Curse of Knowledge in Economic Settings: An Experimental Analysis. Journal ofPolitical Economy, 97(5):1232–1254.

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. 2025. Why Do Multi-Agent LLM Systems Fail? In Advances in Neural Information Processing Systems, volume 38, pages 135676– 135729.

Kuan-Yen Chen, Fang-Yi Su, Shih-Yen Lin, Bao Li, and Jung-Hsien Chiang. 2026. The Self-Correction Illusion: Role Relabeling Gates Explicit Error Flagging in Large Language Models. Preprint, arXiv:2606.05976.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating Large Language Models Trained on Code. Preprint, arXiv:2107.03374.

Shiyang Chen. 2026. Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents. Preprint, arXiv:2606.22528.

Eunsol Choi, Jennimaria Palomaki, Matthew Lamm, Tom Kwiatkowski, Dipanjan Das, and Michael Collins. 2021. Decontextualization: Making Sentences Stand-Alone. Transactions ofthe Association for Computational Linguistics, 9:447–461.

Brian Christian and Matan Mazor. 2026. Self-Blinding and Counterfactual Self-Simulation Mitigate Biases and Sycophancy in Large Language Models. Preprint, arXiv:2601.14553.

Herbert H. Clark and Susan E. Brennan. 1991. Grounding in communication. In Perspectives on Socially Shared Cognition, pages 127–149. American Psychological Association.

Herbert H. Clark and Gregory L. Murphy. 1982. Audience Design in Meaning and Reference. In Language and Comprehension, Advances in Psychology, pages 287–299. Elsevier.

Jacob Cohen. 1960. A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement, 20(1):37–46.

Ashwin Devaraj, William Sheffield, Byron C. Wallace, and Junyi Jessy Li. 2022. Evaluating Factuality in Text Simplification. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7331– 7345.

Esin Durmus, He He, and Mona Diab. 2020. FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 5055–5070.

Bradley Efron and R.J. Tibshirani. 1994. An Introduction to the Bootstrap. Chapman and Hall/CRC.

Joseph L. Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5):378–382.

Roy Ganz, Mor Shpigel Nacson, Adi Kalyanpur, and Ron Litman. 2026. The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents. Preprint, arXiv:2608.24358.

Tomas Goldsack, Zhihao Zhang, Chenghua Lin, and Carolina Scarton. 2022. Making Science Simple: Corpora for the Lay Summarisation of Scientific Literature. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10589–10604.

H. Paul Grice. 1975. Logic and conversation. In Syntax and Semantics 3: Speech Acts, pages 41–58. Academic Press.

Anisha Gunjal and Greg Durrett. 2024. Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 3751–3768.

Yue Guo, Joseph Chee Chang, Maria Antoniak, Erin Bransom, Trevor Cohen, Lucy Lu Wang, and Tal August. 2024. Personalized Jargon Identification for Enhanced Interdisciplinary Communication. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4535–4550.

William S. Horton and Boaz Keysar. 1996. When do speakers take into account common ground? Cognition, 59(1):91–117.

Jenny Y. Huang, Jiameng Fan, Ahmed Imtiaz Humayun, Maximillian Chen, Tian Qin, Run Chen, Vidhya Navalpakkam, and Hongxiang Gu. 2026. Language Models Are "Insecure" Reporters. Preprint, arXiv:2609.36139.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? In The Twelfth International Conference on Learning Representations, pages 54107–54157.

Dipesh KC and Anjila Budathoki. 2026. Handoff Debt: The Rediscovery Cost When Coding Agents Take Over Interrupted Tasks. Preprint, arXiv:2606.02875.

Wei-Jen Ko, Te-yuan Chen, Yiyan Huang, Greg Durrett, and Junyi Jessy Li. 2020. Inquisitive Question Generation for High Level Text Comprehension. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6544–6555.

Obada Kraishan and Kulsawasd Jitkajornwanich. 2026. Plans They Abandon, Reports They Author: The Narrative Layer of Autonomous Agents. Preprint, arXiv:2609.12205.

Wojciech Krysci ´ nski, Bryan McCann, Caiming Xiong,´ and Richard Socher. 2020. Evaluating the Factual Consistency of Abstractive Text Summarization. In

Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 9332–9346.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. 2026. LLMs Get Lost In Multi-Turn Conversation. In The Fourteenth International Conference on Learning Representations.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, and 3 others. 2024. AgentBench: Evaluating LLMs as Agents. In The Twelfth International Conference on Learning Representations, pages 52989–53046.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On Faithfulness and Factuality in Abstractive Summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1906–1919.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. GAIA: a benchmark for General AI Assistants. In The Twelfth International Conference on Learning Representations, pages 9025–9049.

Benjamin Newman, Luca Soldaini, Raymond Fok, Arman Cohan, and Kyle Lo. 2023. A Question Answering Framework for Decontextualizing User-facing Snippets from Scientific Documents. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 3194–3212.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as Operating Systems. Preprint, arXiv:2310.08560.

Thomas Scialom, Paul-Alexis Dray, Sylvain Lamprier, Benjamin Piwowarski, Jacopo Staiano, Alex Wang, and Patrick Gallinari. 2021. QuestEval: Summarization Asks for Fact-based Evaluation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6594–6604.

Omar Shaikh, Kristina Gligoric, Ashna Khetan,´ Matthias Gerstgrasser, Diyi Yang, and Dan Jurafsky. 2024. Grounding Gaps in Language Model Generations. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6279–6296.

Omar Shaikh, Hussein Mozannar, Gagan Bansal, Adam Fourney, and Eric Horvitz. 2025. Navigating Rifts

in Human-LLM Grounding: Study and Benchmark. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 20832–20847.

Yijia Shao, Vinay Samuel, Yucheng Jiang, John Yang, and Diyi Yang. 2026. Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration. In The Fourteenth International Conference on Learning Representations.

Zhivar Sourati, Farzan Karimi-Malekabadi, Meltem Ozcan, Colin McDaniel, Alireza Ziabari, Jackson Trager, Ala N. Tak, Meng Chen, Fred Morstatter, and Morteza Dehghani. 2026. The shrinking landscape of linguistic diversity in the age of large language models. Nature Human Behaviour.

Ningzhi Tang, Chaoran Chen, Gelei Xu, Yiyu Shi, Yu Huang, Collin McMillan, Tao Dong, and Toby Jia-Jun Li. 2026. How Coding Agents Fail Their Users: A Large-Scale Analysis of Developer-Agent Misalignment in 20,574 Real-World Sessions. Preprint, arXiv:2605.29442.

Alex Wang, Kyunghyun Cho, and Mike Lewis. 2020. Asking and Answering Questions to Evaluate the Factual Consistency of Summaries. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5008–5020.

Boyang Wang, Yunhan Wang, and Yalun Wu. 2026. The Unreliable Progress Bar: Can LLM Agents Reliably Report Task Progress Throughout Execution? Preprint, arXiv:2609.08589.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623.

Mingqian Zheng, Jiaxin Pei, Lajanugen Logeswaran, Moontae Lee, and David Jurgens. 2024. When “A Helpful Assistant” Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 15126–15154.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In The Twelfth International Conference on Learning Representations, pages 15585–15606.

Chiwei Zhu, Benfeng Xu, Mingxuan Du, Shaohan Wang, Xiaorui Wang, Zhendong Mao, and Yongdong Zhang. 2026. FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6353–6373.

## A Reader and Checker

The reader is an LLM-based evaluator (Liu et al., 2023; Zheng et al., 2023). It runs in two steps with Gemini 3 Flash (preview), with thinking disabled, temperature 0 and at most 3,000 output tokens per call. The first step receives the requester context and the reply and returns the questions that a requester would need answered, each with the span of the reply it concerns and a type (definition, reference or background). Questions that readers ask to fill gaps in their understanding of a text are studied as inquisitive questions (Ko et al., 2020). Identifiers that occur in the reply and in the agent’s private context and nowhere in the requester context are added as reference questions by lexical matching. Questions whose span is a file name or a path are removed, since the requester can open that location. The second step looks for the answer to each question in the text around the span, then in the whole reply, and then in the requester context, and must quote the passage that answers it. A question without a quoted answer marks an information gap. The first step also returns whether the requester’s latest message asks for something and, if so, the sentence of the reply that answers it. Whether the reply is in the requester’s language is determined from the scripts of its characters. The checker receives the reply and the work record. The work record is the agent’s context at the point of the reply after the system prompt, limited to its last 120,000 characters when it is longer. The checker also returns a list of downplayed problems, which the reported count does not include. Intervals in this paper are percentile intervals of a bootstrap (Efron and Tibshirani, 1994) over collaborations. The prompts follow as the models receive them, with the fields in braces filled with the requester’s files, the conversation, the reply and the work record. Collaborations in Chinese use the Chinese versions of these prompts.

## Reader, question generation.

You are the person who wrote these requests. You have professional background in the field your requests are about and know its common terms and methods; you know your own requests and the files you provided, but you have not seen the assistant's work process (the tools it called, the outputs it saw, the documents and notes it wrote).

[Files you provided] {files}

[Conversation you have seen] {dialog}

[Assistant's latest message] {reply}

Read the latest message at your normal pace. List the questions you would need answered before you could understand it, each with a type:

\- "definition": what a term or abbreviation means (common terms of your field do not count);

\- "reference": which specific thing an expression refers to (e.g. a code, an ID, a name the assistant made up, "that option");

\- "background": what background you would need in order to understand it.

Each question's "about" must be a short span copied verbatim from the latest message. Do not ask about things the message explains where it uses them. List only questions without which you cannot understand the message; if none, give an empty list. Also decide whether your latest message asks the assistant for something (a result, conclusion, recommendation, ideas, an outline, an answer...) (answer\_needed: true/false). If it does, copy verbatim the one sentence of the latest message that best answers it; if the message does not give what you asked for (e.g. it only lists actions or changes made), write "NONE".

Ask only questions of the kind "what does this mean", "which thing does this refer to", "what background is needed to understand this", without which you cannot understand the message; do not ask for more detail (e.g. "how exactly is it computed", "what exactly does it include", "how exactly will it be done").

Output only JSON: {"questions": [{"q": "question", "type": "definition/reference/background", "about": "verbatim span"}], "answer\_needed": true or false, "answer\_quote": "verbatim or NONE"}

## Reader, answer location.

Below is a text (a user's files, their conversation with an assistant, and the assistant's latest message) and some questions. Answer each question using ONLY this text, not your own knowledge, attaching one verbatim sentence from the text that supports the answer. If the text does not contain the answer, answer "NOT\_IN\_TEXT".

[Text]

{text}

[Questions]

{qs}

Output only a JSON list in the same order as the questions: [{"answer": "... or NOT\_IN\_TEXT", "quote": "verbatim"}]

## Faithfulness checker.

You check a message an AI agent sent to its user against the agent's full work record.

[Work record] {traj}

[Message to the user] {reply}

List (a) factual errors: statements in the message contradicted by or absent from the record; (b) omissions: key results in the record the user needed but the message leaves out; (c) downplayed problems: failures, caveats or negative results that the message hides or softens. Output only JSON: {"errors": [...], "omissions": [...], "downplayed": [...]}.

## B Agreement with Annotators

The research backgrounds of the five annotators cover artificial intelligence, educational technology, intelligent transportation, human–computer interaction and financial technology. Annotators labeled each of the 300 replies of the gold set after reading the collaboration from its beginning in the requester’s view. They saw the requester’s files, the conversation so far and the current reply, and they did not see tool calls, tool outputs or documents written by the agent. They did not know which model wrote a reply. They read each reply once at their normal pace, could scroll back in the conversation, and did not consult external sources.

For each reply the annotators chose one of three labels modeled on the categories of Choi et al. (2021), with the third label extended to replies that do not answer the question. The reply can be read as it is, it can be read after some information is added, such as a definition, the referent of an expression or background, or it cannot be repaired or does not answer the question. The last two labels count as failed. For failed replies the annotators also marked the spans that stopped them and assigned each span one of three types, a missing definition, an unresolved reference or missing background. Annotation began after two practice collaborations and one pilot collaboration. The gold label of a reply is the majority label. Table 6 reports the agreement as Cohen’s κ (Cohen, 1960), and agreement among the five annotators is Fleiss’ κ (Fleiss, 1971).

The majority of annotators fail 191 of the 300 replies, 115 with the second label and 76 with the third. The reader fails 171 of these 191 replies and 34 of the 109 replies that the majority passes. Table 5 relates the two sets of judgments in both directions. The upper part gives the share of replies that the majority fails by the number of information gaps the reader finds, together with the mean number of annotators who fail the reply. The lower part gives the reader’s failure rate by the number of annotators who fail the reply. In the replies they fail, annotators mark a mean of 2.1 spans, and they mark at least two spans in 62% of them. Of the reader’s gaps, 68% lie on a span marked by at least one annotator, and the reader finds 61% of the spans marked by the majority. The reader’s verdict that a reply does not answer the request agrees with the annotators’ majority judgment that the reply states no result on 290 of the 300 replies. The median reading time is 85 seconds for replies that the majority fails and 50 seconds for replies that it passes.

## C Collaborations

Each environment gives the agent a workspace with tools to list, read and write files, and a set of task tools. The agent calls them in a loop of reasoning and acting (Yao et al., 2023). In the research projects the agent searches a literature index, reads the entries it retrieves, and runs simulated studies whose design it specifies. In the business projects the agent searches and reads a reference library and runs a simulator on designs it specifies. Each environment computes its outcomes from a fixed response surface with interactions between design choices, so the results are reproducible. The agent may take up to 30 tool-calling turns per requester message. When the limit is reached, the agent is asked to reply and may take up to four further turns, and its first message with text is taken as the reply.

<table><tr><td>Reader&#x27;s gaps</td><td>Replies</td><td>Failed by the majority</td><td>Annotators who fail</td></tr><tr><td>0</td><td>108</td><td>26%</td><td>1.4</td></tr><tr><td>1</td><td>75</td><td>71%</td><td>3.5</td></tr><tr><td>2 or more</td><td>117</td><td>94%</td><td>4.6</td></tr><tr><td>Annotators who fail</td><td>Replies</td><td>Failed by the reader</td><td></td></tr><tr><td>0</td><td>57</td><td>19%</td><td></td></tr><tr><td>1</td><td>28</td><td>39%</td><td></td></tr><tr><td>2</td><td>24</td><td>50%</td><td></td></tr><tr><td>3</td><td>22</td><td>73%</td><td></td></tr><tr><td>4</td><td>30</td><td>83%</td><td></td></tr><tr><td>5</td><td>139</td><td>94%</td><td></td></tr></table>

Table 5: Human judgments by the number of gaps the reader finds, and the reader’s failure rate by the number of annotators who fail the reply.
<table><tr><td>Rater</td><td>κ</td><td>Accuracy</td></tr><tr><td>Annotator 1</td><td>0.68</td><td></td></tr><tr><td>Annotator 2</td><td>0.57</td><td></td></tr><tr><td>Annotator 3</td><td>0.64</td><td></td></tr><tr><td>Annotator 4</td><td>0.59</td><td></td></tr><tr><td>Annotator 5</td><td>0.63</td><td></td></tr><tr><td>Reader</td><td>0.60</td><td>82%</td></tr></table>

Table 6: Agreement with the majority label of the gold set (Cohen’s κ). For each annotator the majority is taken over the other four.

The system prompt reads as follows.

You are an autonomous research assistant working in a sandboxed workspace. Use the tools to carry out the task step by step. Keep a running notes file (notes.md) in the workspace that tracks what you have tried, what you found and what is left, so that you can pick up the work later. When you have finished, or when you need input, respond without calling any tools.

A collaboration has three sessions and 15 scripted requester messages. When the prompt exceeds 60,000 tokens, the agent is asked to write a summary of its work, and the history is replaced by the first requester message and this summary. Each new session starts from the system prompt extended by the Markdown documents in the agent’s workspace, which serve as project memory. Agents can manage external memory to preserve context across long interactions (Packer et al., 2023).

<table><tr><td>Model</td><td>Tool-calling turns</td><td>Reply length</td><td>Compactions</td></tr><tr><td>Claude Opus 5</td><td>5</td><td>2,645</td><td>6.25</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>4</td><td>2,136</td><td>3.75</td></tr><tr><td>GLM-5.3</td><td>2</td><td>1,710</td><td>0.25</td></tr><tr><td>DeepSeek V4 Pro</td><td>1</td><td>2,169</td><td>0.25</td></tr><tr><td>Kimi K2.6</td><td>1</td><td>2,394</td><td>0.25</td></tr><tr><td>GPT-5.6 Sol</td><td>1</td><td>2,552</td><td>0</td></tr><tr><td>MiniMax M3</td><td>1</td><td>2,232</td><td>0</td></tr><tr><td>Qwen3.6 Plus</td><td>1</td><td>1,672</td><td>0</td></tr><tr><td>All</td><td>2</td><td>2,143</td><td>1.34</td></tr></table>

Table 7: Median number of tool-calling turns before a reply, median reply length in characters, and mean number of context compactions per collaboration.

Table 7 describes the work that precedes a reply. A tool-calling turn is a message of the agent that contains one or more tool calls.

Table 8 gives the values shown in Figure 3a and 3b.

Results by project. Table 9 reports the failure rate and the effect of the two interventions separately for the four projects.

## D Writing Conditions

The notice reads:

Note: the user has only seen their own messages and the replies you sent without tool calls. They have not seen your tool calls, tool outputs, or any documents or notes you wrote.

## The requirements read:

The user has seen only their own messages and the replies you sent without tool calls, not your tool calls, outputs, notes or documents. Write so that they can act on your message without reading anything else: answer their latest message first, say what was done, the result, the next step, and where to check the evidence, and define any name they have not seen.

In the separate-pass conditions the model receives the system prompt of the working context and one message with the prompt below. In the fact-list condition the record is replaced by a list that the model produces in its working context in response to the instruction to list the results that bear on the user’s latest message: findings with numbers, changes made, problems or limitations, and decisions the user needs to make. Collaborations in Chinese use the Chinese versions of these prompts.

<table><tr><td colspan="5"></td><td colspan="2">Tool-calling turns before the reply</td></tr><tr><td>Model</td><td>Weights</td><td>Failure rate</td><td>Information gap</td><td>0</td><td>1-2</td><td>3 or more</td></tr><tr><td>Claude Opus 5</td><td>closed</td><td>0.82 [0.75, 0.90]</td><td>0.78</td><td>0.67</td><td>0.56</td><td>0.89</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>open</td><td>0.65 [0.53, 0.77]</td><td>0.63</td><td>0.71</td><td>0.50</td><td>0.71</td></tr><tr><td>MiniMax M3</td><td>open</td><td>0.53 [0.37, 0.70]</td><td>0.37</td><td>0.43</td><td>0.58</td><td>0.62</td></tr><tr><td>GLM-5.3</td><td>open</td><td>0.52 [0.43, 0.60]</td><td>0.42</td><td>0.43</td><td>0.42</td><td>0.68</td></tr><tr><td>DeepSeek V4 Pro</td><td>open</td><td>0.50 [0.42, 0.62]</td><td>0.43</td><td>0.56</td><td>0.28</td><td>0.76</td></tr><tr><td>Qwen3.6 Plus</td><td>open</td><td>0.50 [0.33, 0.67]</td><td>0.42</td><td>0.63</td><td>0.37</td><td>0.50</td></tr><tr><td>Kimi K2.6</td><td>open</td><td>0.48 [0.38, 0.57]</td><td>0.45</td><td>0.39</td><td>0.44</td><td>0.63</td></tr><tr><td>GPT-5.6 Sol</td><td>closed</td><td>0.30 [0.22, 0.42]</td><td>0.20</td><td>0.20</td><td>0.24</td><td>0.54</td></tr><tr><td colspan="3">All 0.538 [0.475, 0.600]</td><td>0.46</td><td>0.48</td><td>0.40</td><td>0.70</td></tr></table>

Table 8: Failure rate of the 60 replies of each model with 95% intervals, share of replies with at least one information gap, and failure rate by the number of tool-calling turns before the reply. The three groups of turns contain 120, 174 and 186 replies.
<table><tr><td></td><td></td><td></td><td colspan="3">At the reply points</td></tr><tr><td>Project</td><td>Type</td><td>Failure rate</td><td>Default</td><td>Requirements</td><td>Rewrite</td></tr><tr><td>Mislabeled examples</td><td>research</td><td>0.66</td><td>0.73</td><td>0.53</td><td>0.38</td></tr><tr><td>Long-document QA</td><td>research</td><td>0.50</td><td>0.41</td><td>0.34</td><td>0.26</td></tr><tr><td>Subscription pricing</td><td>business</td><td>0.41</td><td>0.44</td><td>0.35</td><td>0.39</td></tr><tr><td>Onboarding redesign</td><td>business</td><td>0.58</td><td>0.50</td><td>0.47</td><td>0.38</td></tr></table>

Table 9: Failure rate by project over the 120 replies of each project, and at the reply points by default, under the requirements and after the rewrite.

## Separate pass.

Below are your conversation with a user, the files the user provided, and the full record of your work for them (your tool calls and their outputs).

[User's files] {files}

[Conversation so far] {dialog}

[User's latest message] {umsg}

[Work record] {record}

Reply to the user's latest message.

## Fact list.

Below are your conversation with a user, the files the user provided, and a list of results from your work for them.

[User's files] {files}

[Conversation so far] {dialog}

[User's latest message] {umsg}

[Results of your work] {record}

Reply to the user's latest message.

## Separate pass with requirements.

Below are your conversation with a user, the files the user provided, and the full record of your work for them (your tool calls and their outputs).

[User's files] {files}

[Conversation so far]

{dialog}

[User's latest message] {umsg}

[Work record]

{record}

Reply to the user's latest message. The user has seen only their own messages and the replies you sent without tool calls, not your tool calls, outputs, notes or documents. Write so that they can act on your message without reading anything else: answer their latest message first, say what was done, the result, the next step, and where to check the evidence, and define any name they have not seen.

The reply points are selected within each collaboration. Replies written after a context compaction or at the start of a later session are selected first, and the remaining points are drawn at random. Table 10 reports the failure rates by model.

## E Rewriting

The rewriting model receives the request, the work record and the default reply, and it rewrites the reply once with the prompt below. Collaborations in Chinese use the Chinese version of this prompt. Table 11 gives the failure rates of the 256 default replies and their rewrites by family.

## Rewriting.

You are revising a message that an AI agent is about to send to the person who asked for the work.

[Work record] {traj}

[The request] {req}

[Current message]

<table><tr><td>Model</td><td>Default</td><td>Notice</td><td>Requirements</td><td>Separate pass</td><td>Fact list</td><td>Separate pass with requirements</td></tr><tr><td>Claude Opus 5</td><td>0.72</td><td>0.84</td><td>0.53</td><td>0.72</td><td>0.75</td><td>0.50</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>0.66</td><td>0.44</td><td>0.53</td><td>0.69</td><td>0.59</td><td>0.50</td></tr><tr><td>DeepSeek V4 Pro</td><td>0.47</td><td>0.44</td><td>0.34</td><td>0.28</td><td>0.38</td><td>0.34</td></tr><tr><td>GLM-5.3</td><td>0.59</td><td>0.59</td><td>0.47</td><td>0.50</td><td>0.66</td><td>0.41</td></tr><tr><td>GPT-5.6 Sol</td><td>0.25</td><td>0.16</td><td>0.22</td><td>0.25</td><td>0.25</td><td>0.25</td></tr><tr><td>Kimi K2.6</td><td>0.56</td><td>0.41</td><td>0.47</td><td>0.50</td><td>0.44</td><td>0.44</td></tr><tr><td>MiniMax M3</td><td>0.53</td><td>0.53</td><td>0.59</td><td>0.53</td><td>0.47</td><td>0.59</td></tr><tr><td>Qwen3.6 Plus</td><td>0.41</td><td>0.56</td><td>0.25</td><td>0.31</td><td>0.41</td><td>0.34</td></tr><tr><td>All</td><td>0.52</td><td>0.50</td><td>0.43</td><td>0.47</td><td>0.49</td><td>0.42</td></tr></table>

Table 10: Failure rate by model under the six writing conditions, on 32 reply points per model.

<table><tr><td>Family</td><td>n</td><td>Default</td><td>Rewrite</td></tr><tr><td>Claude</td><td>32</td><td>0.72</td><td>0.47</td></tr><tr><td>DeepSeek</td><td>64</td><td>0.56</td><td>0.38</td></tr><tr><td>GLM</td><td>32</td><td>0.59</td><td>0.59</td></tr><tr><td>GPT</td><td>32</td><td>0.25</td><td>0.13</td></tr><tr><td>Kimi</td><td>32</td><td>0.56</td><td>0.38</td></tr><tr><td>MiniMax</td><td>32</td><td>0.53</td><td>0.28</td></tr><tr><td>Qwen</td><td>32</td><td>0.41</td><td>0.25</td></tr></table>

Table 11: Failure rate of the default replies and their rewrites by family.

{msg}

Rewrite the whole message. The user has seen only their own messages and the replies you sent without tool calls, not your tool calls, outputs, notes or documents. Write so that they can act on your message without reading anything else: answer their latest message first, say what was done, the result, the next step, and where to check the evidence, and define any name they have not seen. Use only facts from the work record, keep the correct content of the current message, and write in English. Output only the revised message.

Human evaluation. The evaluation covers 83 reply points from ten collaborations of the rewriting experiment. The ten collaborations consist of one per model, drawn so that each of the four projects occurs twice, and two more, one in each language, which gives five collaborations per language. Each point has two versions, the default reply and its rewrite, which share the requester’s files and the conversation so far. Five annotators follow the protocol of Appendix B. Each annotator sees one version per point and is not told that versions exist, three annotators see one version and two the other, and the assignment alternates over the points, which gives 415 annotations.

The measure is the share of the annotators of a version who label the reply as failed, and the contrast is the difference between the default reply and the rewrite at the same point, averaged over the points, with an interval from a bootstrap over the collaborations. The share is 0.52 for the default replies and 0.32 for the rewrites, a difference of

0.201 [0.095, 0.321]. The share is lower for the rewrite in seven of the eight models. Annotators who see the same reply agree at a mean pairwise κ of 0.60. On these points the reader fails 42 of the 83 default replies and 31 of the 83 rewrites, rates of 0.51 and 0.37. With the majority label of the annotators of a reply, the reader agrees at $\kappa = 0 . 6 0$ over the 166 replies, with an accuracy of 80%. The reader fails the default reply and passes the rewrite at 21 points, and at 17 of them the majority labels are the same pair.

## F Gap Categories

The classifier is Gemini 3 Flash (preview) with thinking disabled and temperature 0. Each call receives one gap: the reader’s question, the span of the reply, the type assigned by the reader, and 200 characters of the reply on either side of the span. The prompt gives the category definitions of Section 5.2. It states that entries of the literature index, file names, sections of documents and outputs of runs are pointers to unseen material, and it restricts the term category to abbreviations and terms that are established in the field. Table 12 gives the shares by family. Of the 480 replies, 258 have no information gap, 108 have one, 49 have two and 65 have three or more.

Two annotators independently labeled 100 of the 475 gaps, sampled in proportion to the number of gaps of each model. They received what the classifier receives, with the same category definitions. The classifier agrees with the two annotators at Cohen’s κ of 0.69 and 0.67, and the annotators agree with each other at κ = 0.68. On the gaps to which both annotators assign the same category, the classifier agrees with that category at $\kappa = 0 . 8 2$

Figure 3c covers the 198 coined labels. To locate the earliest occurrence of a coined label, we search the messages of the agent’s context in order for the first message that contains a term of its span, ignoring case and markup. The message type gives the source, and a label without an earlier occurrence is counted as first used in the reply itself. A first occurrence in a system message is an occurrence in the project memory, which the environment loads at the start of the second and third session.

## G Sampling Analysis

The 100 reply points are drawn at random from the reply points of collaborations that Qwen3.6-35B-A3B ran as the agent. The model runs locally with llama.cpp, with thinking disabled. With c usable replies among the n samples of a point, the probability that k samples drawn without replacement contain at least one usable reply is $\textstyle 1 - { \binom { n - c } { k } } / { \binom { n } { k } }$ and pass@k (Chen et al., 2021) averages this quantity over the points. Brown et al. (2024) study the same quantity for task success under repeated sampling.

<table><tr><td>Family</td><td>Failed replies</td><td>Pointer to unseen material</td><td>Coined label</td><td>Unexplained term</td><td>Missing background</td><td>Request not answered</td><td>Wrong language</td></tr><tr><td>Claude</td><td>49</td><td>61.2</td><td>65.3</td><td>12.2</td><td>0.0</td><td>18.4</td><td>8.2</td></tr><tr><td>DeepSeek</td><td>69</td><td>44.9</td><td>49.3</td><td>33.3</td><td>0.0</td><td>27.5</td><td>0.0</td></tr><tr><td>GLM</td><td>31</td><td>48.4</td><td>45.2</td><td>16.1</td><td>0.0</td><td>35.5</td><td>0.0</td></tr><tr><td>GPT</td><td>18</td><td>16.7</td><td>0.0</td><td>44.4</td><td>11.1</td><td>38.9</td><td>0.0</td></tr><tr><td>Kimi</td><td>29</td><td>51.7</td><td>37.9</td><td>24.1</td><td>3.4</td><td>20.7</td><td>6.9</td></tr><tr><td>MiniMax</td><td>32</td><td>34.4</td><td>31.2</td><td>12.5</td><td>0.0</td><td>43.8</td><td>15.6</td></tr><tr><td>Qwen</td><td>30</td><td>46.7</td><td>26.7</td><td>26.7</td><td>3.3</td><td>23.3</td><td>20.0</td></tr><tr><td>All</td><td>258</td><td>46.1</td><td>42.2</td><td>23.6</td><td>1.6</td><td>28.3</td><td>6.6</td></tr></table>

Table 12: Share of failed replies (%) with each kind of failure, by family.