# DRAFTTRACE: A Multi-View Analytics Environment for AI-Integrated Writing

Divyansh Chandarana\* Sandipan De\* Vivek Gupta

Arizona State University

 Demo Å Video

{dchanda1, sandipan, vgupt140}@asu.edu

## Abstract

Generative AI has changed how students produce writing assignments. The final artifact is no longer sufficient to understand the process through which it was produced. We introduce DRAFTTRACE, a writing environment that jointly captures three complementary views of writing: the final product, the writing process and interactions with an integrated AI-assistant. DRAFTTRACE reconstructs how a document develops over time and organizes these signals into submission-, longitudinal-, and class-level analytics for instructors. We deployed DRAFT-TRACE in a graduate NLP course with 81 students and compared their sessions with LLMgenerated responses entered by automated tools and with copy-typed responses. While product measures distinguish differences in text formulation, process measures distinguish differences in how text is entered. Considering both views together helps characterize cases such as copytyping. Interaction traces show that students use the assistant differently across stages of writing: to clarify the question at an early stage and to verify answers at a later stage. A preliminary instructor survey highlights the importance of multi-view writing analytics and their interpretability.

## 1 Introduction

Generative AI (GenAI) has changed not only the artifacts students submit as part of written assignments, but also the processes through which those artifacts are produced. Students can use AI assistance at different stages of an assignment: to understand a concept, brainstorm ideas, construct an outline, revise existing writing, co-write portions of a response, or generate a complete response. Traditionally, although students draw on textbooks, scholarly literature, online resources, or other external materials, producing a coherent response generally required them to engage with the material. GenAI has substantially changed this relationship between the final product and students’ engagement, and has made it increasingly difficult for instructors to determine whether and how the writing activity contributed to student learning.

Prior work has studied the writing process by capturing measures like writing speed, pauses, bursts, revisions and deletion events during text production. These measures have been used to investigate their relationships with underlying cognitive processes. However, such behavioral signals do not map uniquely to specific cognitive activities; for example a pause can be associated with planning, reflection, or revision. GenAI introduces an additional source of ambiguity into these observations. A pause may now also correspond to interaction with AI system.

These changes call for a reconsideration of how writing activities are characterized, measured and interpreted in AI-mediated settings. We argue that understanding of writing requires considering three complementary views: the product view, which characterizes the final artifact; the process view, which captures how the artifact develops over time; and the interaction view, which captures how the student engages with the AI during that process. Existing tools provide different subset of these capabilities, including final product analysis, revision histories, and integrated AI assistance. However, these capabilities are treated as separate signals rather than a complementary views of the same writing activity.

To support these integrated perspective, we introduce DRAFTTRACE a writing environment that jointly capture the writing process, characterize the resulting product, and records students’ interactions with an integrated AI assistant. DRAFT-TRACE organizes these complementary sources of information into instructor-facing views at multiple levels of granularity.

## 2 Background and Related Work

## 2.1 Writing Analytics

Writing analytics has used characteristics of completed text for applications ranging from automated essay scoring to automated feedback generation. Ke and Ng (2019) survey product-level measures spanning style, relevance, organization, cohesion, and coherence, among other dimensions of writing quality. More recently, Pande et al. (2026) develop a pipeline that utilizes stylometric analytics alongside a large language model (LLM) to generate feedback for student writing at scale.

Writing research has also used keystroke events to study the writing process. Guo et al. (2018) show that longer writing time and shorter and less variable within-word keystroke intervals are associated with higher essay scores. This finding is consistent with the view that fluency in lower-level transcription processes may free cognitive resources for higher-level composition. Schaller et al. (2026) further demonstrate that keystroke-derived features can provide predictive signals of essay quality during the early stages of composition, before sufficient textual content is available for conventional product-based scoring. At the same time, Babalola et al. (2026) find that many writing-process features exhibit variability across tasks and contexts, including grade level, academic proficiency, and school setting.

## 2.2 AI-Mediated Writing

Studies of AI-mediated writing have examined how differences in AI access and interaction behavior relate to students’ writing processes. While Christenson et al. (2026) measure how ownership perception of the student changes based on the frequency of LLM access, Park et al. (2026) categorize the types of LLM use and examine their association with student performance. Complementing these studies Yang et al. (2025) examine how students process AI-generated assistance, distinguishing between suggestions that are rejected, accepted with modifications, or accepted unchanged. Collectively, these findings highlight substantial variation in how student engage with and use AI during writing.

Recent work has started to combine these perspectives. He et al. (2025) combine process and product information to improve writing assessment. In AI assisted writing, Chen et al. incorporate interaction traces to characterize how students integrate AI into their writing. While these studies combine subsets of signals for specific analytical tasks, collectively they suggest the value of considering product, process, and interaction together to provide a comprehensive view of student writing.

## 3 DRAFTTRACE

The design of DRAFTTRACE is guided by four goals that determine what information is captured, how assistance is provided, how resulting information is presented and how the platform fits within existing educational workflows.

Integrated view of writing. Process, product, and interaction information is captured and aligned as complementary views of the writing.

Configurable AI support. Rather than treating AI access as allowed or forbidden, instructors can configure the level of AI assistance according to the assignment and learning objectives.

Longitudinal and class-level analytics. Writing analytics are organized to compare a student’s submissions over time and with broader class-level patterns.

Low-friction access and integration. The writing environment operates independently or integrates with existing Learning Management Systems to allow students and instructors to use the platform within their normal workflows without requiring separate procedures.

Existing writing and educational platforms provide different subsets of capabilities to these design goals. Table 1 compares DRAFTTRACE with representative systems across these capabilities.

## 3.1 System Architecture

DRAFTTRACE is a web application with a React front end, Express back end, and a PostgreSQL store (Figure 1). The editor is built on Tip-Tap/ProseMirror. The browser and server share the same document schema and analysis modules, so the writing playback shown to instructors and the metrics computed on the server are derived from identical logic.

Process capture. DRAFTTRACE records each change to the document as an event with a timestamp, associated text, and its input channel: typing, paste, drag-and-drop, undo/redo, or insertion from the integrated assistant. Switching away from the editor is also logged. Complete document snapshots are stored periodically to bound replay cost.

![](images/179ba9701d7908340915840e95b56171274ac9f6dd68545825d195ec79bb6d4e.jpg)  
Figure 1: Overview of the DRAFTTRACE system architecture for capturing, reconstructing, and analyzing student writing and AI interactions.

<table><tr><td></td><td>PJroC.</td><td>Prrod.</td><td>Interac.</td><td>Asst.</td><td>Connhg</td><td>plaqack</td><td>Anallytcs</td><td>Edor</td></tr><tr><td>System DraftTrace</td><td>√</td><td>√</td><td>√</td><td>√</td><td>V</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Turnitin</td><td>√</td><td>√</td><td>一</td><td>√</td><td>一</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Clarity Documark</td><td>√</td><td>√</td><td>一</td><td></td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>Grammarly</td><td>√</td><td>√</td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>Authorship Draftback</td><td>√</td><td>一</td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>GPTZero</td><td>一</td><td>√</td><td>一</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pangram</td><td>一</td><td>√</td><td>一</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Comparison of capabilities across representative writing platforms. Proc.: process view; Prod.: product view; Interact.: interaction view; Assist.: integrated AI assistant; Config.: configurable AI support; Editor: integrated writing environment.

To avoid data loss on unreliable classroom networks, events are queued in the browser’s IndexedDB and then uploaded small batches. Each event carries a unique sequence number, making retries idempotent. The events are discarded locally only after the server acknowledges them. For timed assignments, deadlines are enforced on the server, and incomplete sessions are submitted automatically when time expires.

Product reconstruction and provenance. After submission, DRAFTTRACE replays the recorded events to rebuild the document and labels each character by how it entered: typed, inserted from the built-in assistant, pasted with or without a cited source, or unknown. These labels are summarized per paragraph and for the whole submission, alongside process measures such as bursts, pauses, and revisions.

Interaction capture. The integrated assistant is proxied through the server rather than called from the browser. Each student prompt is persisted. Responses are streamed to the student and saved to the session transcript which is visible to the instructor. Text inserted from the assistant is tagged at the transaction level, linking the interaction view to the process and product views. Instructors can enable or disable the assistant per assignment, and a system prompt instructs it to support understanding rather than write answers.

## 3.2 Instructor Interface

The instructor interface supports two primary workflows - to create quizzes and to review the resulting analytics. Instructors can configure the questions, instructions, time and length constraints (Figure 2). For each submission, DRAFTTRACE summarizes writing-process measures (Figure 3) and provides a chronological timeline of the writing session (Figure 4). Additional class-level and longitudinal views are provided in Appendix A.

![](images/e0a28dba1886d10a0d2a8077c1ff48cfe705bc8ddb174ab364fb321551091c5a.jpg)  
Figure 2: Instructor interface for creating an assignment.

## 3.3 Student Interface

The student interface provides an environment for completing the assignment task. Student compose their responses directly in the editor (Figure 5). When enabled by instructor, students can access the integrated AI assistant from within the same environment (Figure 6).

![](images/28ebb8e3b2d0a0dc781101e4c08ef39b687e235eff353ffeea79bef2cdfb7c7f.jpg)  
Figure 3: Writing-process view summarizing measures captured during an individual writing session.

## 4 Case Study

## 4.1 Data Collection

We collected writing traces under three settings that vary how the written response is formulated and entered into the writing environment: classroom writing, automated writing, and copy-typing.

Classroom Writing. We used DRAFTTRACE during a writing activity in graduate-level Natural Language Processing course and collected responses from 81 students who were present during the session. At the beginning of the activity, students were briefly introduced to the writing environment and its available features. Students were given 15 minutes to answer two related questions based on the materials covered in a previous lecture. They were allowed to consult lecture slides and use the AI assistant integrated within DRAFTTRACE, but were instructed not to use external AI tools.

<table><tr><td colspan="4">▼ ACTIVITY TIMELINE</td></tr><tr><td>Time</td><td>Activity</td><td></td><td>Δ</td></tr><tr><td>16:53:09</td><td>Typing burst · 1m 8s</td><td></td><td>+192</td></tr><tr><td>16:54:19</td><td>Typing burst · 36s</td><td></td><td>+92</td></tr><tr><td>16:54:57</td><td>Typing burst · 1s</td><td></td><td>+7</td></tr><tr><td>16:55:00</td><td>Typing burst · 8s</td><td></td><td>+25</td></tr><tr><td>16:55:30</td><td></td><td>Typing burst · 17s</td><td>+53</td></tr><tr><td>16:55:54</td><td></td><td>Typing burst · 4s</td><td>+23</td></tr><tr><td>16:56:00</td><td></td><td>Typing burst · 19s</td><td>+50</td></tr><tr><td>16:56:23</td><td> Typing</td><td></td><td>+1</td></tr><tr><td>16:56:27</td><td></td><td>Typing burst · 6s</td><td>+22</td></tr><tr><td>16:56:36</td><td></td><td>Typing burst · 2m 17s</td><td>+361</td></tr><tr><td>16:58:56</td><td></td><td>Typing burst · 1s</td><td>0</td></tr><tr><td>16:59:00</td><td></td><td>Typing burst · 1s</td><td>0</td></tr><tr><td>16:59:04</td><td></td><td>Typing burst · 8s</td><td>+16</td></tr><tr><td>16:59:16</td><td></td><td>Typing burst · 11s</td><td>+26</td></tr><tr><td>16:59:30</td><td></td><td>Typing burst · 26s</td><td>+68</td></tr><tr><td>16:59:58</td><td></td><td>Typing burst · 3s</td><td>+10</td></tr><tr><td>17:00:03</td><td></td><td>Typing burst · 21s</td><td>+46</td></tr><tr><td>17:00:26</td><td></td><td>Typing burst · 8s</td><td>+26</td></tr></table>

Figure 4: Activity timeline showing the chronological development of an individual writing session.

The first question asked the students to select a prompting technique and explain it in their own words and provide examples of its use. The second question asked the students to identify a scenario in which the selected technique might not work well and explain why. While all students followed the same general task structure, they could select different prompting techniques and construct their own examples, allowing variation in the content of the responses.

Automated Writing. We collected an additional 81 writing sessions in which response to the same questions were generated through LLM. The generated responses were entered into DRAFTTRACE using automated typing mechanism that approximates human-like text entry.

Copy-Typing. To examine writing traces when text formulation and text entry are separated, we collected additional eleven sessions in which participants were allowed to use provided responses, external AI tools, or other resources while answering the same questions. Participants were instructed to manually type text into DRAFTTRACE.

![](images/c76cc71c936756a4e4f8d516ecba4c5f66c15930b61d0937a6157cf5e211b571.jpg)  
Figure 5: Writing environment for completing the assigned task.

![](images/195d9346d4eab6c9368c387253c01aa086cac684e3b68a352ffc5eff5ced9cbf.jpg)  
Figure 6: Integrated AI assistant.

<table><tr><td></td><td>Class. (n=81)</td><td>Auto. (n=81)</td><td>Copy (n=11)</td></tr><tr><td>Product view</td><td></td><td></td><td></td></tr><tr><td>Flesch Reading Ease</td><td>52.8</td><td>38.4</td><td>39.9</td></tr><tr><td>Mean word length (chars)</td><td>4.67</td><td>5.18</td><td>5.40</td></tr><tr><td>Mean sentence length (chars)</td><td>130.50</td><td>163.17</td><td>147.50</td></tr><tr><td>Process view</td><td></td><td></td><td></td></tr><tr><td>Typing speed (wpm)</td><td>27.5</td><td>49.0</td><td>33.0</td></tr><tr><td>Revisions / 100 chars</td><td>12.5</td><td>2.7</td><td>5.4</td></tr><tr><td>Left the editor (count)</td><td>14</td><td>0</td><td>0</td></tr><tr><td>Active / elapsed time</td><td>0.77</td><td>1.00</td><td>1.00</td></tr></table>

Table 2: Median values per setting. Class.: classroom writing; Auto.: automated writing; Copy: copy-typing.

## 4.2 Analysis

We compare the three settings using a small set of measures from the process and product views. From the process view we use typing speed (words per minute), revisions per 100 typed characters, the number of times the student left the editor, and the fraction of elapsed time spent actively writing. From the product view we use Flesch Reading Ease and mean word length, computed on the concatenated answers to both questions.

The product view reflects who formulated the text. Responses formulated by an LLM were harder to read and used longer words than classroom responses, regardless of how they were entered. Median Flesch Reading Ease was 52.8 for classroom responses, compared with 38.4 for automated and 39.9 for copy-typed responses, and median word length rose from 4.67 to 5.18 and 5.40 characters. Automated and copy-typed responses had nearly identical product measures. This is expected as the text originates from an LLM, the final artifact looks similar irrespective of how it was entered.

The process view reflects how the text was entered. Automated entry left a regular trace: a median typing speed of 49 wpm, 2.7 revisions per 100 characters, and no idle time or departures from the editor. For copy-typing we observe similar behavior: they neither paused nor left the editor as the text was already formulated. However, the participants typed more slowly (33 wpm) and revised twice as often (5.4 per 100 characters). Classroom writers, by contrast, revised most (12.5 per 100 characters), left the editor a median of 14 times, and were actively writing for 77% of the session.

<table><tr><td></td><td>Q1</td><td>Q2</td><td>Q3</td><td>Q4</td></tr><tr><td>Prompts</td><td>11</td><td>17</td><td>21</td><td>26</td></tr><tr><td>Students’ first prompt</td><td>10</td><td>5</td><td>7</td><td>10</td></tr></table>

Table 3: Assistant use by session quarter for the 32 students who used the integrated assistant.

Neither view is sufficient alone. Copy-typing illustrates why the views must be read together. Its product resembles automated writing, while its process resembles human typing without the pauses and editor departures seen in the classroom. Identifying such a session requires combining both views, the integrated perspective DRAFTTRACE is designed to provide.

The interaction view reflects when and why students sought help. In the classroom setting, 32 of 81 students used the integrated assistant. In total, they have issued 75 prompts (median 2 per student; 15 students asked only once). We divided each student’s session into four equal quarters and assigned each prompt to a quarter based on its timestamp. Distribution of the requests across these segments is shown in Table 3

To examine how students used the assistant, we categorized their interactions into four types: clarification about the question, verification of a drafted answer, direct help (requesting an answer or example), and collaboration (an extended exchange that develops the response over several turns). Students whose first prompt came in the first quarter sought clarification (6 of 10) or direct help (4 of 10). Students who first used the assistant in the last quarter almost always sought verification of their answers (9 of 10). Collaboration appeared only among students who issued three or more prompts (7 of 13). Students who issued only one or two prompts generally used the assistant for either direct help or verification.

These patterns show that students use the assistant differently at different stages of the writing process. DRAFTTRACE aligns the interaction view with the process timeline and provides additional context for interpreting these patterns.

## 5 Instructor Survey

To complement the classroom study, we conducted a survey<sup>1</sup> to understand instructors’ current practices, their perceptions of AI-integrated writing environments, and their considerations regarding the collection and analysis of student writing data. The survey targets educators involved in teaching, assisting, or grading student work across different academic levels and disciplines.

## Findings.

• Current practices provide limited confidence. Nine of the ten respondents reported encountering suspected inappropriate AI use at least sometimes, while only one reported being very confident in their current process for investigating such cases. This contrast suggests a gap between the frequency with which instructors encounter questions about how student work was produced and their confidence in the information currently available to investigate those questions.

• Writing support and visibility were valued across multiple dimensions. Writing-process visibility and final-submission analysis were each rated as very valuable by 7 of 10 respondents, while instructor-controlled AI access, AI-interaction visibility, and writing-process playback were each rated very valuable by 6 respondents.

• Acceptance of information collection was largely conditional. While 6 of 10 respondents considered copy/paste activity and AIinteraction history generally acceptable to collect, writing/editing activity, detailed typing patterns, and AI processing of student writing were more often viewed as acceptable only under limited circumstances.

• Accuracy and interpretability were identified as important considerations. Inaccurate or misleading results could discourage adoption for 9 of 10 respondents, while 4 of 10 indicated that a lack of transparency about how such systems work could affect adoption.

## 6 Discussion

While DRAFTTRACE is designed as a writing analytics tool, we also view it as a research apparatus for studying how writing changes in AI-integrated environments. By collecting aligned product, process, and interaction information, the environment enables researchers to observe not only what students produce, but how their writing and use of assistance evolve during an assignment and across assignments. Beyond observation, DRAFTTRACE provides an environment for adaptive experimentation. As the AI assistance is configured per assignment, researchers can vary its availability and behavior. This enables controlled studies of when assistance should be provided and what form it should take. As the system already observes writing and interaction patterns, it could be extended to adapt the level or form of assistance based on these observations, a direction we plan to explore in future work

## 7 Limitations

DRAFTTRACE captures activity occurring within its writing environment and cannot observe all external resources or activities that may contribute to a response. While the captured process, product, and interaction signals provide complementary information, they do not capture every aspect of the writing process. Additionally, our case study is limited to a single graduate-level course and a short writing activity. The writing patterns may vary across tasks, student populations, and instructional settings.

## 8 Ethical Considerations

DRAFTTRACE collects fine-grained information about students’ writing behavior and AI interactions beyond what is contained in a conventional final submission. Students should therefore be informed about what information is collected, how it is processed, and who has access to it. Such data, particularly writing traces and AI conversations, should be treated as sensitive educational data with appropriate access and retention controls.

## References

Damilola Babalola, Naga Buddarapu, Piotr Mitros, Paul Deane, and Collin Lynch. 2026. Stability and contextual sensitivity of keystroke process features in longitudinal student writing portfolios. In Proceedings ofthe 19th International Conference on Educational Data Mining, pages 256–266, Seoul, Republic of Korea. International Educational Data Mining Society.

Nuo Chen, Wanying Zhong, Kejie Shen, and Yizhou Fan. Beyond the chat window: A trace-based positioning analysis of student-genai interactions in academic writing. In Artificial Intelligence in Education, pages 640–649, Cham. Springer Nature Switzerland.

Julia Christenson, Karin de Langis, Shirley Anugrah Hayati, and Dongyeop Kang. 2026. Effects of varying LLM access on essay writing behavior. In Proceedings of the 21st Workshop on Innovative Use ofNLPfor Building Educational Applications (BEA 2026), pages 685–701, San Diego, California, USA. Association for Computational Linguistics.

Hongwen Guo, Paul D. Deane, Peter W. van Rijn, Mo Zhang, and Randy E. Bennett. 2018. Modeling basic writing processes from keystroke logs. Journal ofEducational Measurement, 55(2):194–216.

Xinyun He, Qi Shu, Mo Zhang, Wei Huang, Han Zhao, and Mengxiao Zhu. 2025. Beyond final products: Multi-dimensional essay scoring using keystroke logs and deep learning. In Proceedings ofthe 15th International Learning Analytics and Knowledge Conference, LAK ’25, page 601–610, New York, NY, USA. Association for Computing Machinery.

Zixuan Ke and Vincent Ng. 2019. Automated essay scoring: A survey of the state of the art. In IJCAI, volume 19, pages 6300–6308.

Stuti Pande, Yige Song, Kamila Misiejuk, Sonsoles López-Pernas, Mohammed Saqr, and Eduardo A. Oliveira. 2026. Profiling writing skills at scale: A hybrid stylometry-llm pipeline for formative feedback. In Proceedings of the Thirteenth ACM Conference on Learning @ Scale, L@S ’26, page 491–495, New York, NY, USA. Association for Computing Machinery.

Minju Park, Ivan Orozco Vasquez, and Cristina Conati. 2026. Characterizing students’ llm usage behaviors and their association with learning in critical thinking tasks. In Proceedings ofthe 19th International Conference on Educational Data Mining, pages 57–66, Seoul, Republic of Korea. International Educational Data Mining Society.

Nils-Jonathan Schaller, Daniel Mora Melanchthon, Thorben Jansen, Olaf Köller, and Andrea Horbach. 2026. KEYSCORE — keystroke-enhanced automated essay scoring. In Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026), pages 347– 357, San Diego, California, USA. Association for Computational Linguistics.

Kaixun Yang, Mladen Rakovic, Zhiping Liang, Lixi-´ ang Yan, Zijie Zeng, Yizhou Fan, Dragan Gaševic,´ and Guanliang Chen. 2025. Modifying ai, enhancing essays: How active engagement with generative ai boosts writing quality. In Proceedings ofthe 15th International Learning Analytics and Knowledge Conference, LAK ’25, page 568–578, New York, NY, USA. Association for Computing Machinery.

## A Interfaces

Figure 7 presents additional instructor-facing analytics in DRAFTTRACE, including an overall summary of an individual submission, a longitudinal

![](images/6443732cd97c4d10f67556877256fb3266933b703a6d41344bc797fc7ab2c228.jpg)  
(a) Overall analytics for an individual submission.

![](images/cb3d3961a9044610aca8c394d7c4902b2cbb0f1797596b5cb5d65baca3e4853b.jpg)  
(b) Longitudinal view of a student’s writing across submissions.

![](images/77bdca1f46ba7f1a6c493cdd0855e20f2d5e12c404ee717194a48e738b3f3b33.jpg)  
(c) Comparison of an individual submission with classlevel patterns.  
Figure 7: Additional instructor-facing analytics in DRAFTTRACE: (a) overall analytics for an individual submission, (b) longitudinal analysis of a student’s writing across submissions, and (c) comparison with classlevel writing patterns.

view of the student’s writing across submissions, and a class-level comparison for the assignment.