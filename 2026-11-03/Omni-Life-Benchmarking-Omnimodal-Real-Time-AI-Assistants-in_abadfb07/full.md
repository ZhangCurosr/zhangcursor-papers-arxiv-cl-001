# Omni-Life: Benchmarking Omnimodal Real-Time AI Assistants in People’s Daily Lives

Mengze Hong<sup>1,2,∗</sup>, Ziwei Shang<sup>1,∗</sup>, Di Liang<sup>2,†</sup>, Zijian Li<sup>2</sup> Liang Zhu<sup>2</sup>, Hongyu Shan<sup>2</sup>, Robin Zhou<sup>2</sup>, Minlong Peng<sup>3</sup> Zhiqiang Kou<sup>1</sup>, Jia Wang<sup>4</sup>, Di Jiang<sup>1</sup>, Chen Jason Zhang<sup>1,†</sup>

<sup>1</sup>Hong Kong Polytechnic University <sup>2</sup>Today AI <sup>3</sup>Fudan University <sup>4</sup>Beijing University of Posts and Telecommunications

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors.

Project page: https://github.com/todayai-labs/omni-life

## Abstract

Real-time omnimodal assistants are expected to support people in everyday life, but audiovisual understanding alone does not establish whether they can provide timely and appropriate help. We introduce Omni-Life, a benchmark grounded in real-world user needs, deployment scenarios, and device-dependent observations. It combines controlled video synthesis for hazardous situations and edge cases with annotated real-world videos under a shared temporal protocol. The benchmark contains 400 videos: 290 requiring assistance, with 555 required intervention events, and 110 requiring silence throughout to test whether models can withhold unnecessary responses when no assistance is needed. We evaluate twelve configurations featuring mainstream real-time omnimodal models, jointly assessing event coverage, response content, timing, and unwanted output. At most 4.48% of videos requiring assistance are handled completely correctly. To examine practical deployment strategies, we study task reminders and response-latency tolerance. Reminders can improve recall by 17.12 percentage points but add approximately 4.6 unwanted responses per minute of reference silence. Increasing latency tolerance by 10s improves recall by at most 5.95 percentage points. These results show that existing models remain insufficient to deliver reliable assistance in real-life applications, calling for tailored model optimization in instruction following, timely proactive intervention, and withholding responses when assistance is not needed.

## 1 Introduction

Recent advances in omnimodal full-duplex foundation models are shifting human-AI interaction from turn-based conversations toward continuous, real-time assistance with audio-visual perceptions. This transition fundamentally changes what an assistant is expected to do. Rather than simply generating an answer after receiving a complete input, a real-time assistant must decide whether to respond, when to respond, and what response is appropriate as the surrounding environment changes (Lu et al., 2026c). While current efforts still largely emphasize model development and individual interaction capabilities, emerging demonstrations from mainstream AI products, such as SeedRealtime in Doubao and GPT-Live, already point toward a much broader future in which such models operate across different devices, environments, and user populations, ideally participating in people’s everyday activities. This expectation also introduces new forms of failure: a model may interrupt when no help is needed, overlook an important event, or even provide incorrect information that either leads to minor user dissatisfaction or consequential safety risks. These transitions in how AI serves people create an emerging need for model evaluation beyond atomic abilities: evaluating in real-world applications is both timely and necessary to guide models toward truly helpful assistants in complex daily scenarios. Figure 1 illustrates three everyday assistance needs and their appropriate responses.

A broad categorization of benchmark resources can be divided into capability-driven and downstream task-driven. On the one hand, existing omnimodal benchmarks evaluate streaming perception, audiovisual understanding, interaction triggers, response timing, and other capabilities required for real-time interaction, but typically do so within predefined observatory tasks or controlled interaction settings (Lin et al., 2026; Niu et al., 2025; Wang et al., 2025c). On the other hand, foundation model benchmarks have long expanded evaluation across domains, tasks, and skills, but their notion of scenario diversity is largely designed around conventional input-output systems and does not reflect the complexity introduced by realtime omnimodal interaction. In practice, appropriate model behavior depends on a compound of where the interaction takes place, who the model is serving, what the deployed device can observe, and when intervention is actually useful. A meaningful test may therefore be less about solving an exceptionally difficult maths

![](images/91af465123715a778da127d4758d46652bd5c3fc33b3fae76ed0ef0fd9ec03e4.jpg)  
Figure 1: Illustration of real-life use cases for omnimodal real-time AI assistants.

problem, and more about whether a model can appropriately help a primary school student with homework, assist a blind person in navigating safely, or provide timely support during an unfolding daily activity. We therefore argue that evaluating omnimodal real-time interaction requires bridging these two perspectives together: the interaction capabilities and the broad, practical service scenarios in which these models play meaningful roles in everyday life, asking not only what models are capable of doing, but also what services they can reliably provide, for whom, under what conditions, and when.

In this paper, we introduce Omni-Life, to our knowledge the first real-time omnimodal benchmark centered on everyday human support needs. We organize it around a three-dimensional grid jointly conditioned on scenario × population × device: a scenario describes what is happening, a population identifies whose needs matter, and a device determines what the assistant can observe and whether the viewpoint is static or evolving. Rather than combining these categories mechanically, we use valid combinations to define concrete service tasks, in which people’s actual needs and demands are valued over synthesized test cases solely for observational capabilities. This is further emphasized by paired test instances given the treatment of the user population, where the assistant should react meaningfully at the right moment or remain silent depending on the population that appears in the event.

## The main contributions of this paper include:

1. We develop an evaluation framework that connects real-world user needs with deployment scenarios and the observations available to different devices. The framework translates these factors into concrete service tasks with explicit conditions for intervention and appropriate silence, assessing model capabilities in real-life scenarios.

2. We construct the Omni-Life benchmark, featuring 400 human-verified cases that combine controlled video synthesis of hazardous situations and edge cases with annotated real-world videos. A unified temporal annotation protocol specifies required assistance, response windows, and intervals of expected silence as the desired model behaviors.

3. We perform a comprehensive evaluation of twelve real-time omnimodal model configurations through continuous streaming and turn-based QA, complemented by subgroup analyses, failure taxonomy, and controlled ablations. Our results reveal systematic limitations in timely and selective assistance, uncover trade-offs between intervention coverage and unwanted responses, and identify key challenges in reliable real-world deployment.

Table 1: Comparison of benchmark tasks and evaluation protocols.
<table><tr><td>Benchmark</td><td>Task focus</td><td>Video source</td><td>Proactive Response User Hazard Silence-only response</td><td>timing groups alerts</td><td></td><td></td><td>videos</td></tr><tr><td>StreamingBench (Lin et al., 2026)</td><td>Streaming video QA</td><td>YouTube</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>OVO-Bench (Niu et al., 2025)</td><td>Online video QA</td><td>Curated + YouTube</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>OmniMMI (Wang et al., 2025c)</td><td>Multimodal interaction</td><td>Curated + YouTube</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>ProactiveVideoQA (Wang et al., 2025b)</td><td>Proactive video QA</td><td>Public datasets</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>PhoStream (Lu et al., 2026b)</td><td>Mobile assistance</td><td>Web + curated + recordings</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>OmniInteract (Lu et al., 2026c)</td><td>Spoken interaction</td><td>Recordings + curated</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>OmniPro (Zhao et al., 2026)</td><td>Proactive video tasks</td><td>LongVALE + COIN</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>EgoSocial (Wang et al., 2025a)</td><td>Social intervention</td><td>Ego4D</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Omni-Life</td><td>Everyday assistance</td><td>Real + Synthetic</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 2 Related Work

## 2.1 Omnimodal and Full-Duplex Models

Multimodal foundation models are increasingly evolving from offline perception toward unified, streaming interaction. Recent omnimodal models integrate vision, audio, speech, and text within a single system, enabling continuous perception and low-latency response generation (Xu et al., 2025; Cui et al., 2026). In parallel, full-duplex spoken dialogue models such as Moshi relax the conventional turn-taking assumption by allowing a model to listen and speak simultaneously, supporting interruptions, backchannels, and overlapping speech (Defossez et al.´ , 2024). More recent systems further combine full-duplex communication with continuous visual perception and proactive behavior, allowing models to see, listen, and speak while an environment is evolving (Cui et al., 2026). Together, these advances are shifting multimodal AI from static input-output processing toward real-time interaction, where model behavior must be continuously grounded in both the user and the surrounding environment.

## 2.2 Benchmarking Real-time Interaction

A growing body of benchmarks has begun to evaluate capabilities required by this interaction paradigm. StreamingBench and OVO-Bench study online video understanding and timestampaware reasoning over streaming visual inputs (Lin et al., 2026; Niu et al., 2025). OmniMMI extends this setting toward multimodal streaming understanding and proactive reasoning (Wang et al., 2025c), while ProactiveVideoQA explicitly evaluates both response content and response timing in proactive video interaction (Wang et al., 2025b). Full-Duplex-Bench focuses on pause handling, backchanneling, turn-taking, and interruption management in spoken dialogue (Lin et al., 2025), and more recent benchmarks such as Omni-DuplexEval and OmniInteract evaluate when and what models should respond under real-time multimodal streams (He et al., 2026; Lu et al., 2026c). PhoStream further evaluates mobile assistants through on-screen and off-screen streaming QA, including egocentric videos from blind users (Lu et al., 2026b). Building on existing protocols for online perception and interaction, Omni-Life organizes service tasks across population, scenario, and device with explicit intervention and silence requirements grounded in user needs and observable evidence, as compared with representative benchmarks in Table 1.

## 2.3 Application-Oriented Benchmarks

Another line of work moves evaluation from generic capabilities toward realistic application contexts. WorkArena evaluates agents on routine knowledge-work tasks in enterprise software (Drouin et al., 2024), while OSWorld studies open-ended computer-use tasks across real desktop applications (Xie et al., 2024). Domain-oriented benchmarks such as QualBench further evaluate whether general-purpose LLMs possess the localized professional knowledge required across multiple real-world vertical domains (Hong et al., 2025). EgoSocial evaluates socially appropriate intervention decisions from egocentric audiovisual cues in short social-interaction clips (Wang et al., 2025a). OmniSmartHome evaluates grounding and executing spoken requests using visual and spatial-audio context in synthetic and real smart-home episodes (Jung et al., 2026). These efforts demonstrate the value of evaluating models according to the environments and services in which they are expected to operate. Omni-Life evaluates a complementary setting of everyday-life interaction with real-time omnimodal assistants, where the model may need to continuously observe an unfolding situation, determine whether assistance is needed, identify whom it should assist, and respond within an appropriate temporal window. We therefore organize evaluation jointly across scenario, population, device, and time, treating the real-world service provided by an omnimodal assistant as the basic unit of evaluation.

## 3 Omni-Life Dataset

## 3.1 Overview

The goal of Omni-Life is to study how future AI systems can support people in everyday life across heterogeneous user groups, real-world scenarios, and deployment devices. Unlike

marily evaluate whether a model can recognize, describe, or answer questions about observed content, Omni-Life focuses on a more fundamental problem: when should an AI respond to an ongoing real-world situation, and what behavior is appropriate in that situation? Figure 2 summarizes the scenario and service-group coverage, with device icons illustrating observation contexts. We formulate this decision by jointly considering three aspects:

$$
f : ( P , S , D )  A ,\tag{1}
$$

where P denotes user or population characteristics, S denotes the current scenario, D denotes the available sensing and interaction devices, and A denotes the expected AI behavior. We define five behavior labels: [warn] for alerting users to potential risks, [inform] for proactively providing relevant information, [feedback] for evaluating or correcting user actions, [answer] for responding to direct user queries, and [wait] for situations where no response is appropriate. Together, these behav-

![](images/fd1d2fb113b82a7f86f68a2d4613516f4a1aca4c96039276a1fad37be3b0ea91.jpg)  
Figure 2: Five scenario domains and twelve service groups. Counts are videos; rings encode separate dimensions with schematic angles.

iors capture both when AI assistance is needed and when the appropriate behavior is to remain silent, reflecting the dynamic nature of real-life interactions and the importance of timely, contextaware assistance.

Figure 3 summarizes the construction pipeline of Omni-Life. We first translate support needs into life scenarios and reference-grounded assistant behaviors. Videos are either synthesized from structured specifications or selected from public real-world sources. Quality checks cover observable evidence, temporal consistency, and answer leakage. Videos include task queries and timed action, response, and silence annotations for streaming evaluation and turn-based QA.

![](images/cb192390e408c64a60dbfed75c930127631ed0b0e7c907c2c4dc75bcce2a56fa.jpg)  
Figure 3: Construction pipeline of Omni-Life. Support needs guide scenario design and video sourcing through controlled synthesis or selection from public real-world sources. Temporal annotations define response windows, behavior labels, and reference content, and provide correct options for turn-based QA.

## 3.2 Systematic Scenario Construction

A central challenge in constructing Omni-Life is deciding which situations in everyday life genuinely call for AI support. Rather than directly enumerating activities, we organize the scenario space hierarchically as

## Support Category → Support Need → Life Scenario.

(2)

This organization is motivated by Gibson’s notion of affordance, which emphasizes that the opportunities and constraints presented by an environment depend on the capabilities of the person interacting with it (Gibson, 2014). The same staircase, for example, may be an ordinary path for a walking adult but an obstacle for a wheelchair user. From the perspective of AI assistance, the relevant question is therefore not simply “what is happening in the scene?”, but “what support does this person need in this situation?”

To avoid defining such needs solely through intuition, we further draw on the WHO International Classification of Functioning, Disability and Health (ICF), which relates human functioning to activities, participation, and environmental factors<sup>1</sup>. Following this perspective, we organize everyday support into four broad categories:

1. Perception: access to visual, auditory, color, and other environmental information that may be difficult to obtain directly;

2. Cognition: awareness of state changes, situational understanding, judgment, attention, and memory;

3. Action: mobility, action execution, and assistance when independent response is difficult;

4. Communication: expression and understanding of participants, targets, and states in multi-party interactions.

Importantly, user characteristics provide context for these support needs but do not directly determine the expected AI behavior. The same user may require different behaviors depending on the situation: for example, a visually impaired user may require [inform] when identifying an object, [answer] after explicitly asking about the environment, [warn] when approaching an immediate hazard, and [wait] when no assistance is needed.

Population, scenario, and observation conditions are specified jointly when constructing a case. A proposed response must be justified by evidence available from the selected viewpoint: a close-up may identify an object but cannot establish what is happening outside its field of view. The construction process filters incompatible combinations of population, behavior, viewpoint, and task form before expanding a candidate into a scene specification. The device dimension guides camera placement and the evidence that the scene must make observable.

## 3.2.1 Reference-Grounded Behavior Design

The life scenarios above specify what kinds of situations Omni-Life should cover, but they do not yet determine how the AI should behave in each situation. To make this decision, we further identify the reference against which the observed situation should be interpreted. For skill-related scenarios, AI can support users in two different ways: reporting what happened and providing feedback on how an action was performed. This distinction draws on knowledge of results (KR) and knowledge ofperformance (KP) in motor learning (Salmoni et al., 1984). We therefore use [inform] to report states or outcomes, and [feedback] to evaluate or correct the action itself. When an incorrect action further creates an immediate risk of harm, the appropriate behavior escalates from [feedback] to [warn]. Rule-governed outcomes are determined relative to constitutive rules (Searle, 1969). In Omni-Life, [inform] or [answer] is appropriate only when the observed evidence supports the conclusion. Conversational cooperation emphasizes relevant contributions, while turn-taking concerns the allocation of speaking turns (Grice, 1975; Sacks et al., 1974). In Omni-Life, the AI should [wait] when it is not being called or when another speaker has not finished, making appropriate silence an explicit behavior rather than a missing response.

These principles turn each life scenario into an explicit intervention decision. For each scenario, we specify the reference required for judgment, the observable evidence supporting that judgment, and the corresponding AI behavior. We then construct cases around important decision boundaries: correct versus incorrect actions for [feedback], ordinary versus hazardous states for [inform] or [feedback] versus [warn], and addressed versus non-addressed interactions for [answer] versus [wait]. In this way, behavior labels are tied to observable conditions rather than assigned from the scenario category alone.

## 3.3 Synthetic and Real-World Video Sourcing

From life scenarios to observable events. After defining the appropriate behavior and its decision conditions, we translate each life scenario into concrete events that can be observed in videos. We represent the temporal progression as

$$
E = { \big ( } e _ { 1 } , e _ { 2 } , \ldots , e _ { n } { \big ) } ,\tag{3}
$$

where each e describes an observable event or state change relevant to the intervention decision. For example, a fall-risk scenario specifies the source of risk, how the person approaches it, and the event that makes the risk identifiable. This specifies the observable evidence required to justify the expected behavior. The same life scenario can be instantiated with different people, environments, objects, viewpoints, and event progressions, increasing visual and contextual diversity while preserving the underlying decision being evaluated. When multiple relevant events occur, we prioritize them according to urgency, with safety-critical events taking precedence over routine informational needs.

Synthetic video generation. We synthesize videos with Seedance video models<sup>2</sup> from structured specifications that separate scene settings, timed events, and candidate response annotations. The generation prompt contains the location, participant appearances, key objects, camera viewpoint, and event descriptions; reference assistant replies remain in the annotation record. For cases triggered by a spoken question, prompts retain that utterance and environmental sound while prohibiting an answering voice, explanatory narration, and subtitles. This preserves the distinction between the evidence for a decision and the response being evaluated.

Event descriptions specify an ordered progression with visible state changes and pauses between repeated actions. Counting cases require distinguishable completed actions, while longer sequences can defer a question until after intervening events. Paired specifications share the setting and most events while changing a condition relevant to intervention, such as whether a hazard remains unhandled or is resolved by another person. They provide intervention and non-intervention cases in similar everyday situations without treating a scene category alone as permission to respond.

The generation guidelines favor visually explict actions and state changes, such as falling or turning off a stove, over subtle deviations of otherwise normal actions. Missing or ambiguous decision-critical evidence is a criterion for regenerating, reformulating, or excluding a candidate. These criteria aim to reduce the risk of conflating synthesis defects with model errors.

Real-world video sourcing. To complement controlled synthesis with naturally occurring variation, we additionally source publicly accessible real-world videos that instantiate the predefined life scenarios. These videos introduce natural diversity in human behavior, environments, camera motion, and event progression that may not be fully captured by generated content. Synthetic and real-world videos are subsequently annotated under the same behavior and temporal framework, providing a common evaluation space across both sources. Each selected clip is treated as a separate evaluation video, with temporal labels expressed on the resulting clip’s timeline.

Quality control. A specification linter supports checks before rendering by flagging overly short event segments, descriptions with no observable action, and descriptions that restart the setting instead of continuing the preceding event. Reusable prompt blocks specify stable participant appearances, consistent object counts, camera continuity, and physically coherent motion and contact. Prompt constraints guide generation; the rendered clip still requires review. Video quality control focuses on the visibility of decision-critical evidence, temporal consistency, and absence of answer leakage; synthetic clips additionally require physical plausibility and participant and object consistency. Additional construction and quality-control criteria are provided in Appendix A.

## 3.4 Temporal Annotation and QA Construction

Each video is paired with a task query and an ordered sequence of reference intervals. An interval records its start and end times, one of the five behavior labels, and Chinese and English reference content where a response is required. The two languages express the same assistance requirement. The half-open interval [s, e) defines when that assistance is appropriate; it can differ from the period during which an object is merely visible. Explicit [wait] intervals have empty response content and cover periods when no intervention is appropriate.

We then create turn-based QA and each video has one content question and one timing question. Content questions ask what the assistant should say at a given moment or over the full video. Timing questions ask when it should respond. Question stems use predefined templates, and correct options come from the annotations. Gemini generates incorrect content options related to the same scenario and task. Incorrect timing options are generated by rules using alternative time windows or start times. For videos requiring silence throughout, the correct answer is to remain silent.

Finally, we manually review a sample of the questions, balance their wording and length, and remove duplicate options. The final set contains 800 questions: 400 content questions and 400 timing questions. Further details are provided in Appendix A.

Table 2: Omni-Life dataset statistics.
<table><tr><td>Statistic</td><td>Synthetic Real-world Total</td><td></td><td></td></tr><tr><td>Videos</td><td>277</td><td>123</td><td>400</td></tr><tr><td>Intervention</td><td>184</td><td>106</td><td>290</td></tr><tr><td>Remain Silent</td><td>93</td><td>17</td><td>110</td></tr><tr><td>Intervention nodes</td><td>255</td><td>300</td><td>555</td></tr><tr><td>Inform</td><td>139</td><td>223</td><td>362</td></tr><tr><td>Warn</td><td>92</td><td>57</td><td>149</td></tr><tr><td>Feedback</td><td>11</td><td>18</td><td>29</td></tr><tr><td>Answer</td><td>13</td><td>2</td><td>15</td></tr><tr><td>Silence intervals</td><td>370</td><td>315</td><td>685</td></tr><tr><td>QA questions</td><td>554</td><td>246</td><td>800</td></tr></table>

Table 3: Evaluated local and API-based model interfaces.
<table><tr><td>Model</td><td>Serving / audio input</td></tr><tr><td colspan="2">Open-weight and Locally deployed Models</td></tr><tr><td>Open-weight models, locally deployed MiniCPM-o 4.5 (Cui et al., 2026)</td><td>Native duplex / audio encoder</td></tr><tr><td>MOSS-VL (Wang et al., 2026b) MOSS Preview (Wang et al., 2026a)</td><td>Streaming service / SenseVoice Streaming service / SenseVoice</td></tr><tr><td>Venus-Omni (Ant Group, 2026) Gander (Orantqing et al., 2026)</td><td>Native duplex / audio encoder</td></tr><tr><td>AURA (Lu et al., 2026a)</td><td>OnlineRunner / audio encoder vLLM / Qwen3-ASR-1.7B</td></tr><tr><td>JoyAI (Yao et al., 2026)</td><td>StreamingInferAdapter / Qwen3-ASR-1.7B</td></tr><tr><td>DuplexOmni (S1) (Huang et al., 2026)</td><td>vLLM Thinker / audio encoder</td></tr><tr><td colspan="2">API-based Models</td></tr><tr><td>API-based models</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Gemini 3.8 Live</td><td>Google Live / native audio</td></tr><tr><td>Qwen3.5-omni-flash-realtime (Qwen Team, 2026a)</td><td>Alibaba Realtime / native audio</td></tr><tr><td>Qwen3.5-omni-plus-realtime (Qwen Team, 2026a) Qwen3.8-omni-flash-realtime (Qwen Team, 2026b)</td><td>Alibaba Realtime / native audio Alibaba Realtime / native audio</td></tr></table>

## 3.5 Dataset Composition

The evaluation set contains 400 videos: 277 synthesized and 123 real-world clips. Of these, 290 require at least one intervention and 110 require silence throughout their annotated timelines. The annotations contain 555 required intervention nodes and 685 [wait] intervals, including quiet periods within intervention videos. Video counts and interval counts therefore describe different units. Table 2 separates the composition by video source. The required interventions comprise 362 [inform], 149 [warn], 29 [feedback], and 15 [answer] instances. The five behavior labels are not forced into equal proportions because their applicability depends on the task. The evaluated videos cover 12 user and service groups, with five application domains (see Figure 2). Records retain population, scene, viewpoint, and task-derived assistant-role metadata for descriptive analysis.

## 4 Results and Discussion

## 4.1 Experimental Setup

Models and deployment. We evaluate eight locally deployed open-weight systems and four API configurations in both streaming and turn-based QA (Table 3). Local systems use released checkpoints and inference implementations on an eight-H200 server, with isolated sessions and native memory and context management. MOSS Preview uses the real-time SFT checkpoint; DuplexOmni uses only the S1 Thinker, without its S2 reasoning layer. API models use Google Live<sup>3</sup> or Alibaba Omni Realtime endpoints.

Streaming protocol. Each session receives the Chinese benchmark instruction, a written task, and source video, without reference answers or annotation windows. Media follow the source time without future inputs: frames are sampled at a nominal 1 FPS with source timestamps, and audio is 16-kHz mono PCM. The 30 videos without source audio receive no artificial track. All main-experiment models, including Gemini and Qwen, receive text task prompts. For both MOSS models, AURA, and JoyAI, speech in the source video is transcribed with ASR and supplied as text, because these configurations do not process audio directly. Both MOSS models use SenseVoice (An et al., 2024), while AURA and JoyAI use Qwen3-ASR-1.7B (Shi et al., 2026).The instruction requests [tag] response text when intervention is needed and no output for [wait]. Local systems and Qwen return text; Gemini returns audio with transcription. Interface-specific scheduling and response grouping are retained; decoding, media delivery, and recording details are in Appendix A.3.

Turn-based QA. Each video supplies one content question and one timing question. Media are truncated at the question’s observation cutoff when specified; otherwise, the full clip is available. Models are instructed to return option letters only, separated by semicolons for multiple choice. QA decoding settings are given in Appendix A.3.

Scoring and metrics. DeepSeek-V4-Flash-0731 (DeepSeek-AI et al., 2026) evaluates each answer against the task- and node-specific reference using a binary true/false judgment. Semantically equivalent and minimally sufficient answers are accepted, while answers that omit required information or contain substantive contradictions are rejected. The model does not access the video or evaluate response timing, and correctness of behavior tags is not required. A prediction is considered a match only if its content score is 1 and its onset falls within the original [s, e) window. We use maximum-cardinality, order-preserving, one-to-one matching, without additional timing tolerance. These judgments are conditioned on the annotated ground-truth references and do not independently verify facts against the video (Appendix A.4).

Let G = 555 be the required-node count, P the response-event count across all 400 videos, and TP the timely, content-correct matches. Intervention Recall is TP/G and Intervention F1 is 2TP/(P + G); unmatched outputs and nodes are false positives and false negatives, respectively. Full Correct is the fraction of the 290 intervention videos with all nodes matched, no extra response or response-interval overlap with silence, and successful completion. Silence Success is the fraction of the 110 fully silent videos completed with no actual output. QA Content Acc. and Timing Acc. use exact answer-set matching over 400 questions each; Joint Acc. requires both answers for a video to be correct. Missing, invalid, and failed QA answers receive no credit, and failed streaming sessions remain in the denominators. F1 uses interface-specific response units and should be read with the video-level metrics; Appendix A.5 gives further scoring details.

Human baseline. Four human annotators independently completed both the turn-based QA and streaming evaluations. Reported scores are the unweighted means of the four participants’ accuracies. Streaming responses were scored by a judge for semantic correctness and temporal alignment with reference interventions. Implementation details, exclusions, and score aggregation are provided in Appendix A.6.

## 4.2 Main Results

Table 4 reports both streaming intervention and explicit turn-based question-answering performance. The results separate event coverage, useful content, and appropriate silence.

Reliable proactive assistance remains difficult. The strongest Full Correct score is 4.48% among the twelve evaluated model configurations, followed by Gemini at 3.79%. MOSS-VL also leads Intervention F1 at 7.06%, but its recall is only 5.77%. Low video-level correctness therefore accompanies low node coverage; it is not explained solely by requiring every event in a video to be handled correctly. Supporting streaming input does not by itself establish reliable execution of a standing service task.

Intervention and silence must be evaluated together. Table 4 shows two contrasting patterns. JoyAI achieves the highest Intervention Recall (20.00%), but its F1 is only 1.67% and its Full Correct is zero. Gander achieves 99.09% Silence Success but only 1.08% recall.

Table 5 provides further detail. For JoyAI, 74.6% of required nodes are paired with an on-time but incorrect response; the corresponding rate for AURA is 83.4%. For Gander, 93.5% of required nodes have no later reply. Incorrect responses include missing information, unhelpful content, and factual errors, so they should not all be interpreted as hallucinations. Together, these results show why event coverage, response quality, and silence must be evaluated jointly.

Table 4: Results on Omni-Life (%; higher is better, ↑). Excluding the human baseline, the highest scores are marked in green and the second-best scores are marked in lighter green
<table><tr><td></td><td colspan="4">Streaming</td><td colspan="3">Turn-based QA</td></tr><tr><td></td><td>Full Correct ↑</td><td>Intervention Recall ↑</td><td>Intervention F1↑</td><td>Silence Success ↑</td><td>Joint Acc. ↑</td><td>Content Acc. ↑</td><td>Timing Acc. ↑</td></tr><tr><td colspan="8">Human Baseline</td></tr><tr><td>Human</td><td>53.10</td><td>62.34</td><td>64.49</td><td>91.82</td><td>60.38</td><td>82.13</td><td>66.13</td></tr><tr><td colspan="8">Open-weight Models</td></tr><tr><td>MOSS Preview</td><td>0.00</td><td>1.08</td><td>1.09</td><td>2.73</td><td>11.00</td><td>45.00</td><td>29.00</td></tr><tr><td>AURA</td><td>0.00</td><td>12.61</td><td>1.63</td><td>0.91</td><td>24.00</td><td>65.75</td><td>31.00</td></tr><tr><td>JoyAI</td><td>0.00</td><td>20.00</td><td>1.67</td><td>0.00</td><td>13.25</td><td>63.00</td><td>19.00</td></tr><tr><td>DuplexOmni (S1)</td><td>0.69</td><td>4.50</td><td>4.67</td><td>0.00</td><td>2.75</td><td>12.25</td><td>24.25</td></tr><tr><td>MiniCPM-o 4.5</td><td>1.38</td><td>1.44</td><td>2.63</td><td>95.45</td><td>13.00</td><td>53.50</td><td>25.75</td></tr><tr><td>Venus-Omni</td><td>2.07</td><td>1.98</td><td>3.18</td><td>77.27</td><td>13.25</td><td>58.25</td><td>26.00</td></tr><tr><td>Gander</td><td>2.07</td><td>1.08</td><td>2.05</td><td>99.09</td><td>3.50</td><td>34.00</td><td>12.00</td></tr><tr><td>MOSS-VL</td><td>4.48</td><td>5.77</td><td>7.06</td><td>82.73</td><td>5.75</td><td>18.50</td><td>35.00</td></tr><tr><td colspan="8">API-based Models</td></tr><tr><td>Qwen3.5 Plus</td><td>2.07</td><td>3.78</td><td>4.88</td><td>80.91</td><td>14.25</td><td>45.25</td><td>40.25</td></tr><tr><td>Qwen3.8 Flash</td><td>2.76</td><td>2.70</td><td>3.98</td><td>84.55</td><td>32.00</td><td>58.25</td><td>48.75</td></tr><tr><td>Qwen3.5 Flash</td><td>3.10</td><td>4.14</td><td>5.39</td><td>80.91</td><td>17.25</td><td>48.50</td><td>30.25</td></tr><tr><td>Gemini 3.8 Live</td><td>3.79</td><td>3.42</td><td>5.45</td><td>93.64</td><td>42.25</td><td>78.25</td><td>51.25</td></tr></table>

Table 5: Response outcomes for all 555 required interventions, reported separately for each model (%). The highest value in each Correct column is bold.
<table><tr><td rowspan="2">Model</td><td colspan="2">On-time</td><td colspan="2">Late</td><td rowspan="2">No later reply</td><td rowspan="2">Others</td></tr><tr><td>Correct↑</td><td>Wrong↓</td><td>Correct↑</td><td>Wrong↓</td></tr><tr><td colspan="8">Open-weight Models</td></tr><tr><td>MiniCPM-o 4.5</td><td>1.4</td><td>2.5</td><td>1.4</td><td>2.3</td><td>82.9</td><td>9.4</td></tr><tr><td>MOSS-VL</td><td>5.8</td><td>10.1</td><td>2.5</td><td>8.5</td><td>66.1</td><td>7.0</td></tr><tr><td>MOSS Preview</td><td>1.1</td><td>9.0</td><td>0.7</td><td>3.6</td><td>82.3</td><td>3.2</td></tr><tr><td>Venus-Omni</td><td>2.0</td><td>7.7</td><td>1.4</td><td>3.1</td><td>80.5</td><td>5.2</td></tr><tr><td>Gander</td><td>1.1</td><td>2.2</td><td>0.9</td><td>0.9</td><td>93.5</td><td>1.4</td></tr><tr><td>AURA</td><td>12.6</td><td>83.4</td><td>0.2</td><td>2.5</td><td>0.9</td><td>0.4</td></tr><tr><td>JoyAI</td><td>20.0</td><td>74.6</td><td>0.4</td><td>1.6</td><td>3.2</td><td>0.2</td></tr><tr><td>DuplexOmni (S1)</td><td>4.5</td><td>7.2</td><td>0.0</td><td>1.4</td><td>73.7</td><td>13.2</td></tr><tr><td colspan="8">API-based Models</td></tr><tr><td>Gemini 3.8 Live</td><td>3.4</td><td>4.0</td><td>2.7</td><td>3.4</td><td>83.8</td><td>2.7</td></tr><tr><td>Qwen3.5 Plus</td><td>3.8</td><td>8.8</td><td>3.4</td><td>3.6</td><td>67.4</td><td>13.0</td></tr><tr><td>Qwen3.5 Flash</td><td>4.1</td><td>7.4</td><td>3.1</td><td>4.9</td><td>67.6</td><td>13.0</td></tr><tr><td>Qwen3.8 Flash</td><td>2.7</td><td>7.0</td><td>3.1</td><td>1.6</td><td>71.4</td><td>14.2</td></tr></table>

Explicit QA competence does not establish continuous assistance capability. Gemini achieves 78.25% Content Accuracy in QA but only 3.42% streaming Intervention Recall. Answer selection after a question differs from initiating and delivering assistance while a task remains active.

Table 6: Full Correct (%, higher is better, ↑) across five scenarios, evaluated on videos requiring intervention. Column headers report video counts. The highest value in each column is bold.
<table><tr><td>Model</td><td>Daily living &amp; services ↑ 95 videos</td><td>Health &amp; caregiving ↑ 70 videos</td><td>Travel &amp; mobility ↑ 40 videos</td><td>Education &amp; work ↑ 41 videos</td><td>Sports &amp; leisure ↑ 44 videos</td></tr><tr><td colspan="6">Open-weight Models</td></tr><tr><td>MiniCPM-o 4.5</td><td>1.1</td><td>0.0</td><td>2.5</td><td>2.4</td><td>2.3</td></tr><tr><td>MOSS-VL</td><td>6.3</td><td>2.9</td><td>2.5</td><td>4.9</td><td>4.5</td></tr><tr><td>MOSS Preview</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Venus-Omni</td><td>3.2</td><td>1.4</td><td>2.5</td><td>2.4</td><td>0.0</td></tr><tr><td>Gander</td><td>3.2</td><td>1.4</td><td>0.0</td><td>4.9</td><td>0.0</td></tr><tr><td>AURA</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>JoyAI</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DuplexOmni (S1)</td><td>0.0</td><td>2.9</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="6">API-based Models</td></tr><tr><td>Gemini 3.8 Live</td><td>5.3</td><td>0.0</td><td>5.0</td><td>4.9</td><td>4.5</td></tr><tr><td>Qwen3.5 Plus</td><td>1.1</td><td>1.4</td><td>0.0</td><td>4.9</td><td>4.5</td></tr><tr><td>Qwen3.5 Flash</td><td>2.1</td><td>2.9</td><td>2.5</td><td>4.9</td><td>4.5</td></tr><tr><td>Qwen3.8 Flash</td><td>4.2</td><td>1.4</td><td>0.0</td><td>7.3</td><td>0.0</td></tr></table>

Separating timing from content. Table 5 preserves the main-score matches, then pairs remaining nodes and responses one-to-one, prioritizing on-time incorrect responses, late correct responses, and late incorrect responses. Late means onset at or after the window end, without an upper bound on lateness. The four response categories distinguish on-time or late onset and correct or incorrect content. No later reply means no recorded response at or after the node onset in a completed session. Others retain failed sessions, responses already assigned elsewhere, and missing evidence. This diagnostic gives no additional main-score credit and does not establish which event a late reply was intended to address. Early outputs remain unmatched responses in the main F1 calculation.

Useful content can arrive after the opportunity to help. For example, in an elevator-monitoring video, MOSS-VL reports that the door has opened at 24.18 s, after the required response window of [18.75, 21.75) s. Section 4.5 examines whether extending response windows gives such late responses credit under a relaxed timing criterion.

## 4.3 Population, Scenarios, and Intervention Requirements

Population-level breakdown. Intervention Recall remains limited across population and service groups (Tables 11 and 12). For visual impairment, JoyAI matches 6 of 46 intervention nodes (Table 12). Partial event coverage provides limited evidence that assistance will remain correct and selective throughout an entire video.

Correctly matched interventions can also be concentrated within a population. Thirteen of Gemini’s nineteen correct matches belong to the adult group, corresponding to 13/229 for adults and 6/326 across the remaining groups. Its aggregate recall therefore conceals subgroup variation. Subgroup recall rates likewise require their denominators: the best scores for hearing and cognitive impairment represent only 6/17 and 2/9 nodes, respectively. The groups differ in task and observation conditions, and a single additional match would change these small-group rates. Practical selection should validate performance over entire videos for the intended users and tasks, including quiet periods; these comparisons do not isolate population effects.

Table 7: Intervention Recall by primary service function (%, higher is better, ↑). Column headers give the number of required interval nodes. The highest value is bold.
<table><tr><td></td><td>Preventive ↑ 68 nodes</td><td>Corrective / remedial ↑ 157 nodes</td><td>Routine support ↑ 330 nodes</td></tr><tr><td colspan="4">Open-weight Models</td></tr><tr><td>JoyAI</td><td>30.88</td><td>26.75</td><td>14.55</td></tr><tr><td>AURA</td><td>8.82</td><td>14.01</td><td>12.73</td></tr><tr><td>MOSS-VL</td><td>1.47</td><td>7.01</td><td>6.06</td></tr><tr><td>DuplexOmni (S1)</td><td>8.82</td><td>3.18</td><td>4.24</td></tr><tr><td>Venus-Omni</td><td>0.00</td><td>1.91</td><td>2.42</td></tr><tr><td>MiniCPM-o 4.5</td><td>0.00</td><td>0.64</td><td>2.12</td></tr><tr><td>MOSS Preview</td><td>0.00</td><td>0.00</td><td>1.82</td></tr><tr><td>Gander</td><td>0.00</td><td>0.64</td><td>1.52</td></tr><tr><td colspan="4">API-based Models</td></tr><tr><td>Qwen3.5 Flash</td><td>2.94</td><td>2.55</td><td>5.15</td></tr><tr><td>Qwen3.5 Plus</td><td>4.41</td><td>1.91</td><td>4.55</td></tr><tr><td>Gemini 3.8 Live</td><td>1.47</td><td>1.27</td><td>4.85</td></tr><tr><td>Qwen3.8 Flash</td><td>4.41</td><td>1.27</td><td>3.03</td></tr></table>

Scenario-level breakdown. Table 6 compares video-level success across five everyday scenarios. Mean Full Correct across the twelve configurations is highest in education and work (3.05%), followed by daily living and services (2.19%); the other three groups range from 1.19% to 1.70%. Eight configurations complete at least one video in each of these two groups. They are relatively more tractable for the tested configurations on the sampled tasks, although even their highestscoring configurations complete only 3/41 and 6/95 videos, respectively.

The advantage is associated with the recorded task mix. Daily living and services has the largest proportion of single-intervention videos (73/95). In education and work, ten of the fifteen successful model-video evaluations come from two explicit-question videos, ask school word and ask office tool. Across all scenarios, all 65 fully correct model-video evaluations occur on singleintervention videos; none of the 97 videos requiring multiple interventions is fully correct for any configuration. Full Correct requires every intervention in a video to be handled correctly, making videos with more interventions harder to complete successfully. The scenario groups also differ in user populations and camera viewpoints, so these scores do not show that one scenario is inherently easier than another. Continuous-monitoring evaluation should check whether a model handles all required events and remains silent between them.

Intervention function and temporal demands. We further classify the 555 reference nodes into 68 preventive, 157 corrective or remedial, and 330 routine-support nodes using a model-assisted, internally reviewed procedure (Appendix B.2). Prevention averts a specified problem, correction addresses an existing deviation, and routine support provides requested information, counts, answers, or state reports. The target problem determines the function rather than the action tag: in elder box m pos, warning of a coffee table ahead and providing post-fall help both carry [warn], but serve different functions. Table 7 groups the original main-experiment matches without rematching subsets; all required nodes remain in the denominators, including failed sessions.

Aggregate coverage can conceal a gap in the service a user needs. MOSS-VL recalls 7.01% of corrective nodes and 6.06% of routine-support nodes, but only one of 68 preventive nodes (1.47%). Sixteen of Gemini’s 19 correct matches are routine support, with only three across the 225 preventive and corrective nodes. Success in routine reporting therefore provides limited evidence of coverage when a task requires averting or managing a problem. Application validation should specify which service function must be delivered within its useful response window, rather than accepting aggregate recall as evidence of all three.

Table 8: Observable failures across 4,674 completed sessions. Categories may co-occur, and a flagged session may also contain useful responses.
<table><tr><td>Failure type</td><td>Observable behavior</td><td>Sessions (%)</td></tr><tr><td>F1: Intervention control</td><td>Misses required assistance, responds outside its window, 3,223 (68.96) or speaks when no intervention is needed.</td><td></td></tr><tr><td>F2: Situational grounding</td><td>Misidentifies a current object, attribute, location, relation, or risk state.</td><td>148 (3.17)</td></tr><tr><td>F3: Temporal tracking</td><td>Misreports accumulated counts, event order, historical states, or completion status.</td><td>219 (4.69)</td></tr><tr><td>F4: Task fulfillment</td><td>Produces relevant speech but fails to complete the requested assistance.</td><td>450 (9.63)</td></tr><tr><td>F5: Unusable output</td><td>Produces timestamps, empty tags, nonsemantic strings, 1,064 (22.76) or an incomplete essential answer.</td><td></td></tr></table>

Prevention is not uniformly the least-covered function. JoyAI leads all three columns, with 30.88% preventive, 26.75% corrective, and 14.55% routine recall. This coverage still coexists with low videolevel reliability (Table 4). A development target is therefore to preserve detection of actionable conditions while reducing unrelated output; blanket suppression or more frequent speech alone does not satisfy both requirements. The groups differ in task mix and observation conditions, so differences do not establish intrinsic function difficulty or identify an internal failure mechanism.

Temporal state is a separate requirement: 74 of the 400 tasks need information from previous observations, including 69 intervention videos. For example, the cup-counting task requires reporting the updated count and the final total (Figure 4A). A correct final total does not compensate for missed intermediate updates, and the assistant must remain silent when no update is required. These tasks therefore require both accurate tracking and appropriately timed responses.

## 4.4 Failure Taxonomy

We manually categorize failures across all 4,800 main-experiment sessions, using recorded outputs, available audio transcripts, task references, event captions, and source frames where needed. The 23,337 output records support five observable failure categories (Table 8). Each category is counted at most once per completed session, and categories may co-occur. Rates use 4,674 completed sessions; 126 execution failures (2.63% of all sessions) are reported separately and remain failures in the main evaluation. These categories summarize supported behavioral failures without establishing an exhaustive taxonomy or internal causes. Reference-dependent judgments exclude outputs beyond the annotation horizon. Appendix C presents four case studies with source frames, reference windows, response timelines, and descriptions of relevant outputs. Appendix D adds a 14-case gallery across user groups, all five primary scenarios, and the five failure categories.

Intervention control is the most prevalent failure. Intervention-control failures occur in 68.96% of completed sessions, including 1,720 intervention sessions with no recorded output. A model can also respond initially and then miss a later event. For example, in a toy-car assembly video, DuplexOmni comments at 0.72 s but gives no correction when an arm is later installed incorrectly during the [22, 27) s window (Appendix C.1). Substantive output during reference silence is the converse control failure, even when its content is plausible. The development target is continued event monitoring with selective initiation and deadline control. Timing judgments concern recorded response onset, not the delivery time of every later word in a long response.

Current-scene grounding and temporal tracking require different checks. Situational grounding concerns what is present now: in baby danger, AURA describes the child as still eating and smiling at 14.70 s, while the child is choking and the mother is providing first aid (Appendix Figure 12). Temporal tracking concerns how observations accumulate: in cnt cafe cup, JoyAI repeatedly increments its count and finally reports twelve cups, although the completed sequence contains three (Figure 4A). Verification of task-relevant entities and relations addresses the first observable error; event deduplication and explicit state updates are candidates for addressing the second. The trajectories distinguish these errors without identifying a memory mechanism.

![](images/29e5ddd400dee22753d8b6bd7d3d52353349a8f63adfd43ac6c8eb4734fb491c.jpg)  
Figure 4: Two on-time failures in cup counting (A) and post-fall assistance in a synthetic video (B), shown with source frames, condensed references, and translated outputs. Green intervals mark reference windows; red dots mark response onsets.

Relevant speech can leave the requested assistance undone. Task-fulfillment failures occur in 450 sessions. In grade slip bath pos, a controlled synthetic bathroom-fall scenario, MiniCPM-o 4.5’s sole response begins at 6.83 s, inside the [3.5, 15.1) assistance window (Figure 4B). It identifies the fall but recommends immediately cleaning the water and lifting the person, contrary to the reference’s request to avoid moving hastily and contact emergency assistance. The content judge marks this response incorrect. Recognizing the event and responding on time therefore do not establish task fulfillment: the proposed assistance must also satisfy the task’s required response.

A recorded emission need not contain usable assistance. Unusable output occurs in 22.76% of completed sessions. AURA records <2.5 seconds> at 3.07 s in roof leak, inside the [2, 5) leakreport window, without communicating the leak or its location. This category includes interface artifacts and incomplete essential answers, motivating separate protocol fields and checks of semantic completeness. Recorded artifacts do not establish what was audibly spoken or whether the model or interface produced them. Silence-control markers cannot satisfy a required response.

Failure profiles differ across models. Table 9 distinguishes development priorities across systems. JoyAI combines a 97.50% intervention-control failure rate with the highest prevalence of grounding, tracking, and task-fulfillment failures. AURA and MOSS Preview each contain unusable recorded output in 385 of 400 sessions (96.25%); these sessions can also contain useful responses. For the three Qwen configurations, intervention-control failures occur in 66.39% to 67.48% of completed sessions, while the other types are less prevalent. The profiles motivate testing output validation and task completion alongside selective initiation, according to each system’s behavior.

Low content-error prevalence must be interpreted alongside response coverage. Gander and MiniCPM each exhibit grounding errors in only 0.25% of completed sessions, but emit nothing in 260 and 245 of their 290 completed intervention sessions, respectively. Gemini emits nothing in

Table 9: Model-wise failure prevalence (%, lower is better, ↓) among completed sessions. Completed and Failed give session counts out of 400 attempts and the lowest value in each column is bold.
<table><tr><td>Model</td><td>Completed Control↓</td><td>Grounding ↓</td><td>Tracking ↓</td><td>Fulfillment↓</td><td></td><td>Unusable ↓</td><td>Failed↓</td></tr><tr><td colspan="8">Open-weight Models</td></tr><tr><td>JoyAI</td><td>400</td><td>97.50</td><td>16.50</td><td>19.75</td><td>47.75</td><td>56.50</td><td>0</td></tr><tr><td>AURA</td><td>400</td><td>28.00</td><td>6.25</td><td>8.25</td><td>14.75</td><td>96.25</td><td>0</td></tr><tr><td>MOSS-VL</td><td>400</td><td>66.25</td><td>1.50</td><td>4.00</td><td>7.50</td><td>1.75</td><td>0</td></tr><tr><td>DuplexOmni (S1)</td><td>370</td><td>94.86</td><td>0.81</td><td>0.81</td><td>8.65</td><td>0.54</td><td>30</td></tr><tr><td>Venus-Omni</td><td>400</td><td>67.50</td><td>0.50</td><td>2.75</td><td>6.75</td><td>5.50</td><td>0</td></tr><tr><td>MiniCPM-o 4.5</td><td>400</td><td>67.50</td><td>0.25</td><td>1.25</td><td>1.75</td><td>3.25</td><td>0</td></tr><tr><td>MOSS Preview</td><td>400</td><td>70.75</td><td>1.75</td><td>4.25</td><td>5.50</td><td>96.25</td><td>0</td></tr><tr><td>Gander</td><td>400</td><td>67.50</td><td>0.25</td><td>0.75</td><td>1.75</td><td>0.75</td><td>0</td></tr><tr><td colspan="8">API-based Models</td></tr><tr><td>Qwen3.5 Flash</td><td>369</td><td>67.48</td><td>3.52</td><td>4.61</td><td>6.23</td><td>1.08</td><td>31</td></tr><tr><td>Qwen3.5 Plus</td><td>369</td><td>67.48</td><td>2.98</td><td>4.34</td><td>5.42</td><td>1.36</td><td>31</td></tr><tr><td>Gemini 3.8 Live</td><td>400</td><td>67.75</td><td>1.25</td><td>2.00</td><td>3.25</td><td>2.25</td><td>0</td></tr><tr><td>Qwen3.8 Flash</td><td>366</td><td>66.39</td><td>2.19</td><td>3.01</td><td>5.19</td><td>0.82</td><td>34</td></tr></table>

230 of 290 such sessions. Few observed content errors can therefore coexist with extensive unmet assistance. These rates measure prevalence across sessions; accuracy conditional on producing a response has a different denominator. They should be read with Intervention Recall and Silence Success, while execution failures remain in the main evaluation.

## 4.5 Ablation Studies

We test whether changing or repeating the task instruction, delivering it through speech, or allowing later responses can reduce missed interventions.

Input design and periodic reminders. We compare five text-input conditions on the same 400 videos for MiniCPM, MOSS-VL, Venus, Gemini, and Qwen3.5 Flash: the original task text, a structured instruction containing the same task information, repeating the full task every second, repeating it every 5 seconds, and sending a general reminder every second to continue the original task and remain silent unless its response condition is met. Source media, model settings, and content scoring remain fixed within each model; no condition uses task TTS. Reminders use each model’s native text interface, retaining its acceptance or rejection behavior without forcing generation. Table 10 compares timely coverage and unwanted responses during reference silence.

Structuring the task produces small, mixed changes in recall, while periodic reminders yield larger gains for four models. The better tested reminder interval differs across models, and no variant improves Qwen’s recall over the original task. Reminders both repeat task information and send new messages that may trigger responses. We compare full-task and general reminders, both sent every second, to examine whether repeating the task details helps. Full-task reminders yield higher recall for all five models, although Qwen gains only one additional match. This suggests that reminder content matters, but does not show that the models had forgotten the task. Message length and interface behavior may also affect the results.

Higher intervention coverage can come with more unwanted output. Gemini’s 5-second reminders raise recall from 3.42% to 20.54%, while increasing unwanted responses from 0.57 to 5.16 per minute of reference silence. Venus’s 1-second schedule reaches 13.51% recall while producing 21.68 unwanted responses per minute of reference silence. For applications requiring sustained monitoring, reminder schedules should therefore be evaluated on separate validation videos containing both required interventions and quiet intervals. The tested frequencies provide candidate settings for validation rather than a universal deployment default.

Table 10: Input design and periodic task reminders. Upper: required events handled correctly and on time (%, higher is better, ↑). Lower: unwanted responses per minute of reference silence (lower is better, ↓). Within each model, the highest values in the upper section are shown in bold, and the lowest values in the lower section are shown in bold.
<table><tr><td>Input strategy</td><td>MiniCPM-0 4.5</td><td>MOSS-VL</td><td>Venus-Omni</td><td>Gemini 3.8 Live</td><td>Qwen3.5 Flash</td></tr><tr><td colspan="6">Required events handled correctly and on time (%) ↑</td></tr><tr><td>Original task text</td><td>1.44</td><td>5.77</td><td>1.98</td><td>3.42</td><td>4.14</td></tr><tr><td>Structured task text</td><td>1.44</td><td>6.31</td><td>4.32</td><td>2.52</td><td>3.60</td></tr><tr><td>Repeat full task every second</td><td>5.41</td><td>6.85</td><td>13.51</td><td>12.43</td><td>3.42</td></tr><tr><td>Repeat full task every 5 seconds</td><td>1.80</td><td>14.05</td><td>10.27</td><td>20.54</td><td>3.42</td></tr><tr><td>General reminder every second</td><td>2.88</td><td>3.78</td><td>6.67</td><td>11.71</td><td>3.24</td></tr><tr><td colspan="6">Unwanted responses per minute of reference silence ↓</td></tr><tr><td>Original task text</td><td>0.14</td><td>1.41</td><td>0.48</td><td>0.57</td><td>1.34</td></tr><tr><td>Structured task text</td><td>0.18</td><td>1.20</td><td>0.60</td><td>0.64</td><td>1.34</td></tr><tr><td>Repeat full task every second</td><td>0.64</td><td>5.13</td><td>21.68</td><td>4.96</td><td>1.29</td></tr><tr><td>Repeat full task every 5 seconds</td><td>0.20</td><td>4.48</td><td>7.58</td><td>5.16</td><td>1.29</td></tr><tr><td>General reminder every second</td><td>0.43</td><td>1.73</td><td>10.72</td><td>4.06</td><td>1.25</td></tr></table>

A concrete interface direction is persistent task state containing the event condition, required assistance, and whether the event has already been handled, with updates separated from response generation. Where supported, such updates could be compared with ordinary repeated messages under matched task content, media timeline, and update frequency. Retaining coverage while reducing unwanted output would support this separation. The present ablation motivates that test but does not validate the interface.

Task instruction modality. We compare written and spoken delivery of the same task for MiniCPM-o 4.5, MOSS-VL, Venus-Omni, and Qwen3.5 Flash, using 400 videos per model. The spoken condition removes the case-specific task text from the initial instruction and plays that task once before the video; the general benchmark instruction, source media, model settings, and content judge remain fixed. MOSS-VL receives the actual SenseVoice transcript without a tasktext fallback. Scoring uses the original video timeline: responses before video onset are retained separately, and responses crossing that boundary are scored on their video-phase portion. All cases remain in the denominators, including Venus’s 30 native input failures and Qwen’s three unsuccessful spoken-task sessions.

In Figure 5, the lower panel shows unwanted responses per minute with positive rate labels; percentages in parentheses are relative changes computed from unrounded scores. Both panels retain all 400 videos, including failed sessions; Correct uses the 555 required nodes, while Unwanted uses the total duration of reference silence.

Changing task delivery does not consistently improve performance (Figure 5). MiniCPM’s response count during the video increases from 54 to 129, but its correct, timely matches remain at eight. MOSS-VL and Qwen gain recall but have lower Full Correct scores (Table 15), whereas Venus gains no recall. More responses therefore do not necessarily improve performance over an entire video.

However, this comparison changes more than the task modality. It also changes how and when the task reaches the model, which may affect when the model starts responding.

For MOSS-VL, the task changes from an initial system instruction to one or more ASR-derived user prompts. In the deployed implementation, each new prompt opens an assistant turn. Raw input logs show that the final task segment arrives after video onset in 399 of 400 cases, with a median delay of 0.26 s. ASR segmentation and arrival timing can thus initiate or restart responses during observation, even though the model receives text. Intervention videos with no output fall from 194 to 37, yet the share of emitted responses matched correctly and on time falls from 9.12% to 7.22%. The extra output also extends beyond task intake: unwanted responses beginning at or after 10 s increase from 202 to 378. For example, in a reference-silent night-duty video, MOSS continues describing the scene at 12.68 s while stating that no abnormality has been found. Observations support an interaction-level explanation involving response activation and weak selectivity. They do not isolate the contributions of transcription, task placement, and segmentation.

![](images/066589b22983b16e5bb73bd9714e0e68f6f2885081710ff23a1ab93ed4d4284d.jpg)  
Figure 5: Effect of task instruction modality on Intervention Recall (top; higher is better) and unwanted responses per minute of reference silence (bottom; lower is better).

Qwen’s net increase is more concentrated at startup. Among the 366 videos with source audio completed in both conditions, unwanted replies beginning in the first 2 s increase from 1 to 333, while later unwanted replies decrease from 198 to 183. Correct matches in the early interval increase from 0 to 17, while later matches decrease from 17 to 13. An initial response can therefore recover an event already present at onset or break required silence when no intervention is needed. In a basketball-question video, Spoken replies at 0.61 s, “Okay, I am watching. Which basketball rule would you like to know?” (translated), although the reference requires silence until 11.5 s. This pattern is consistent with a conversational response to task intake.

A controlled follow-up should compare the original task text and actual ASR transcript under matched message roles and arrival times, then compare complete and segmented delivery. The timing buckets are diagnostic: suppressing startup responses could also remove legitimate interventions already required at video onset.

Relaxing the response deadline. To assess how much intervention failure can be recovered by allowing later responses, we extend each reference window from $[ s , e )$ to $[ s , e + \delta )$ , with $\delta \in$ {0, 1, 2, 5, 10} seconds. Window starts, content judgments, and recorded model outputs remain fixed, and we recompute maximum-cardinality, order-preserving one-to-one matches at each tolerance. Intervention Recall retains all 555 required nodes per model, including failed sessions. Gemini and Qwen use the text-prompt main-experiment runs without task TTS. This offline scoring ablation adds no inference and does not extend output collection.

How much failure does extra time recover? Figure 6 measures how much usable late content is already present in the recorded outputs. Extending deadlines by 10 s increases recall by 0.36– 5.95 percentage points. AURA gains the most but reaches only 18.56%, and all models remain below 24%. As a fraction of the nodes missed under the original windows, the recovered share $( T P _ { 1 0 } - T P _ { 0 } ) / ( 5 5 5 - T P _ { 0 } )$ ranges from 0.38% to 6.80%; at least 93.20% remain unmatched. Thus the late-correct replies account for a small part of the deficit under the original deadlines.

![](images/7f848e0f34a4f505266d7505d021f79db7626c1d7bdc55ab6e3c936fcbfef27a.jpg)  
Figure 6: Intervention Recall (%) as reference-window ends are extended. Rows are ordered by the gain at $\delta = 1 0 \mathrm { s } ,$ shown in percentage points (pp) by the right-hand bars.

This is deadline sensitivity of fixed outputs, not an estimate of the benefit of faster inference. Extended windows may overlap later events, and a compatible late reply need not have been intended for an earlier reference. The original collection horizon also limits observable lateness. Applications should set deadlines according to when assistance remains useful and test delivery improvements on late-correct cases, with timestamps across input, buffering, generation, and receipt. Absent and incorrect responses require separate tests of initiation and task fulfillment.

## 5 Conclusion

Omni-Life evaluates whether real-time omnimodal assistants provide the required help at the right moment while remaining silent when no intervention is needed. Across 400 videos with 555 required intervention nodes, none of the twelve main-experiment configurations reliably combines event coverage, useful content, and appropriate silence; the strongest Full Correct score is 4.48% among the twelve evaluated model configurations. The ablations expose a practical distinction between activating responses and improving sustained assistance. Reminder and task-delivery changes can recover required events while also increasing unwanted output, and extending response deadlines by 10 s recovers at most 6.80% of originally unmatched nodes within the recorded trajectories. These findings motivate evaluation and model development that jointly address persistent task execution, selective response initiation, and timely task fulfillment.

## 6 Limitations

Omni-Life’s finite recordings, predefined tasks, and response windows limit generalization to prolonged deployment, adaptive interaction, or changing preferences. Synthetic clips may favor explicit events, and the case mix may not represent everyday support needs. Population and scenario comparisons depend on tasks and observation conditions, establishing neither causal population effects nor equitable performance. System comparisons depend on interfaces and input pipelines; onset measures client-side receipt, not audible playback. Reference-only automatic judging may miss distinctions in task fulfillment. The four-annotator baseline may not generalize broadly, and QA spot checks do not validate all annotations or judgments. Deadline relaxation rescores fixed outputs without testing faster models or longer recordings. Separating task updates from response generation requires further validation of assistance quality and unwanted output.

## 7 Ethics Statement

The experiments evaluate recorded video streams rather than intervening in live situations. Synthetic generation permits hazardous situations to be represented without deliberate exposure of participants to those hazards. Scenarios involving children, older adults, and people with disabilities describe contextual support needs; they should not be interpreted as diagnoses, fixed deficits, or assumptions about every member of a group. Public accessibility of real-world clips does not establish consent or permission for redistribution, and dissemination requires source-specific rights and privacy review. Benchmark performance is evidence about model behavior under the protocol and does not establish readiness to replace human oversight in consequential assistance.

## References

Keyu An, Qian Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Yue Gu, Ting He, Hangrui Hu, Kai Hu, et al. Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms, 2024. URL https://arxiv.org/abs/ 2407.04051.

Ant Group. Realtime-venus: A full-duplex interaction system with asynchronous delegation, 2026. URL https://arxiv.org/abs/2609.13814.

Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, et al. Minicpm-o 4.5: Towards real-time full-duplex omni-modal interaction, 2026. URL https://arxiv.org/abs/2604.27393.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026. URL https://arxiv.org/abs/2606.19348.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. WorkArena: How capable are web agents at solving common knowledge work tasks? In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 11642–11662. PMLR, 21–27 Jul 2024. URL https://proceedings. mlr.press/v235/drouin24a.html.

Alexandre Defossez, Laurent Mazar ´ e, Manu Orsini, Am ´ elie Royer, Patrick P ´ erez, Herv ´ e J ´ egou,´ Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue, 2024. URL https://arxiv.org/abs/2410.00037.

James J. Gibson. The Ecological Approach to Visual Perception: Classic Edition. Psychology Press, November 2014. ISBN 9781315740218. doi: 10.4324/9781315740218. URL http://dx.doi.org/1 0.4324/9781315740218.

H. P. Grice. Logic and Conversation. In Speech Acts, pp. 41–58. BRILL, December 1975. ISBN 9789004368576. doi: 10.1163/9789004368811 003. URL http://dx.doi.org/10.1163/978900436 8811 003.

Chaoqun He, Mingyang Xiang, Yingjing Xu, Bokai Xu, Junbo Cui, Jie Zhou, Yuan Yao, and Lijie Wen. Omni-duplexeval: Evaluating real-time duplex omni-modal interaction, 2026. URL https://arxiv.org/abs/2605.17360.

Mengze Hong, Wailing Ng, Chen Jason Zhang, and Di Jiang. QualBench: Benchmarking Chinese LLMs with localized professional qualifications for vertical domain evaluation. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5938–5953, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.303. URL https://aclanthology.org/2025.emnlp-main.303/.

Muye Huang, Lingling Zhang, Xingyu Yu, Lei Shi, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, and Jun Liu. Duplexomni: Real-time listening, seeing, thinking, and speaking for full-duplex interaction, 2026. URL https://arxiv.org/abs/2606.09186.

Jihoo Jung, Suho Yoo, Jeongsoo Choi, Hyebin Cho, Tae Wook Haam, Hyeonggon Ryu, Sumin Park, and Joon Son Chung. Omnismarthome: A multimodal reasoning benchmark for smart-home agents, 2026. URL https://arxiv.org/abs/2609.32569.

Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Gopala Anumanchipalli, Alexander H Liu, and Hung-yi Lee. Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities. In 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–8. IEEE, 2025.

Junming Lin, Zheng Fang, Chi Chen, Haoxuan Cheng, Zihao Wan, Fuwen Luo, Ziyue Wang, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 12147–12151. IEEE, 2026.

Xudong Lu, Yang Bo, Jinpeng Chen, Shuhan Li, Xintong Guo, Huankang Guan, Fang Liu, Dunyuan Xu, Peiwen Sun, Heyang Sun, Rui Liu, and Hongsheng Li. Aura: Always-on understanding and real-time assistance via video streams, 2026a. URL https://arxiv.org/abs/2604.04184.

Xudong Lu, Guan Huankang, Yang Bo, Jinpeng Chen, Xintong Guo, Shuhan LI, Fang Liu, Peiwen Sun, Xueying Li, Wei Zhang, Xue Yang, Rui Liu, and Hongsheng Li. Phostream: Benchmarking real-world streaming for omnimodal assistants in mobile scenarios. In Forty-third International Conference on Machine Learning, 2026b. URL https://openreview.net/forum?id=cCMbxJcDmH.

Xudong Lu, Xueying Li, Annan Wang, Yang Bo, Jinpeng Chen, Zengliang Li, Nianzu Yang, Rui Liu, Xue Yang, Jingwen Hou, et al. Omniinteract: Benchmarking real-world streaming interaction for real-time omnimodal assistants, 2026c. URL https://arxiv.org/abs/2605.26485.

Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, and Jiaqi Wang. Ovo-bench: How far is your video-llms from real-world online video understanding? In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18902–18913, 2025. doi: 10.1109/CVPR52734.2025.01761.

Orantqing, Shengpeng Ji, Junlong Tong, Jialong Zuo, Dongjie Fu, Di Cao, Yangzhuo Li, Shangda Wu, Franz, Evan, et al. Multimodal duplex interaction agent, 2026. URL https://arxiv.org/ab s/2609.08977.

Qwen Team. Qwen3.5-omni technical report, 2026a. URL https://arxiv.org/abs/2604.15804.

Qwen Team. Qwen3.8-omni: Towards native omni-modal agents, 2026b. URL https://arxiv.or g/abs/2609.25611.

Harvey Sacks, Emanuel A. Schegloff, and Gail Jefferson. A simplest systematics for the organization of turn-taking for conversation. Language, 50(4):696–735, 1974. doi: 10.2307/412243.

Alan W. Salmoni, Richard A. Schmidt, and Charles B. Walter. Knowledge of results and motor learning: A review and critical reappraisal. Psychological Bulletin, 95(3):355–386, 1984. ISSN 0033- 2909. doi: 10.1037/0033-2909.95.3.355. URL http://dx.doi.org/10.1037/0033-2909.95.3.355.

John R. Searle. Speech Acts: An Essay in the Philosophy of Language. Cambridge University Press, 1969.

Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, et al. Qwen3-asr technical report, 2026. URL https://arxiv.org/ab s/2601.21337.

Pengyu Wang, Chenkun Tan, Shaojun Zhou, Wei Huang, Qirui Zhou, Zhan Huang, Zhen Ye, Jijun Cheng, Xiaomeng Qian, Yanxin Chen, Xingyang He, Huazheng Zeng, Chenghao Wang, Pengfei Wang, Hongkai Wang, Shanqing Gao, Yixian Tian, Chenghao Liu, Xinghao Wang, Botian Jiang, and Xipeng Qiu. MOSS-Video-Preview: Toward Real-Time Video Understanding via Cross-Attention, 2026a. URL https://arxiv.org/abs/2606.07639.

Pengyu Wang, Chenkun Tan, Shaojun Zhou, Qirui Zhou, Yanxin Chen, Xingyang He, Huazheng Zeng, Jijun Cheng, Chenghao Wang, Xiaomeng Qian, Pengfei Wang, Zhan Huang, Shanqing Gao, Wei Huang, Longjun Cao, Wu Ran, Jie Liu, Changtai Zhu, Hongkai Wang, Yixian Tian, Chenghao Liu, Zhen Ye, Xinghao Wang, Botian Jiang, Guoguo Feng, Zhaoye Fei, Ruixiao Li, Mingshu Chen, Yang Gao, Qinyuan Cheng, Shimin Li, and Xipeng Qiu. Moss-vl technical report, 2026b. URL https://arxiv.org/abs/2608.15045.

Xijun Wang, Tanay Sharma, Achin Kulshrestha, Abhimitra Meka, Aveek Purohit, and Dinesh Manocha. Egosocial: Benchmarking proactive intervention ability of omnimodal llms via egocentric social interaction perception, 2025a. URL https://arxiv.org/abs/2510.13105.

Yueqian Wang, Xiaojun Meng, Yifan Wang, Huishuai Zhang, and Dongyan Zhao. Proactivevideoqa: A comprehensive benchmark evaluating proactive interactions in video large language models, 2025b. URL https://arxiv.org/abs/2507.09313.

Yuxuan Wang, Yueqian Wang, Bo Chen, Tong Wu, Dongyan Zhao, and Zilong Zheng. Omnimmi: A comprehensive multi-modal interaction benchmark in streaming video contexts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18925–18935, June 2025c.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 52040–52094. Curran Associates, Inc., 2024. doi: 10.52202/079 017-1650. URL https://proceedings.neurips.cc/paper files/paper/2024/file/5d413e48f84 dc61244b6be550f1cd8f5-Paper-Datasets and Benchmarks Track.pdf.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2.5-omni technical report, 2025. URL https://arxiv.org/abs/2503 .20215.

Dingyu Yao, Junhao Zhou, Chenxu Yang, Chuanyu Qin, Xiangyu Zeng, Yifei Li, Haowen Hou, Zheming Liang, Congcong Wang, Kaiwen Tuo, et al. Joyai-vl-interaction: Real-time visionlanguage interaction intelligence, 2026. URL https://arxiv.org/abs/2606.14777.

Ruixiang Zhao, Jie Yang, Zijie Xin, Tianyi Wang, Fengyun Rao, Jing LYU, and Xirong Li. Omnipro: A comprehensive benchmark for omni-proactive streaming video understanding. arXiv preprint arXiv:2605.18577, 2026.

## Appendix Contents

A Dataset Records and Evaluation Details 22   
A.1 Evaluation Set and Annotation Structure 22   
A.2 Construction and Quality-Control Criteria 22   
A.3 Inference and Recording Configuration 24   
A.4 Content Judgment and One-to-One Matching 24   
A.5 Silence, Video-Level Correctness, and Failures 25   
A.6 Human Baseline . 25   
B Complete Streaming Results 26   
B.1 Population, Scenario, and Reference-Action Breakdown . . 26   
B.2 Preventive, Corrective, and Routine Assistance . 26   
B.3 Complete Instruction-Modality Ablation 29   
C Failure Cases Analysis 29   
C.1 Failing to correct an assembly error . 29   
C.2 Mistaking a thermos for a guitar 30   
C.3 Saying guests were greeted instead of greeting them . 31   
C.4 Outputting timestamps instead of reporting a leak in time 31   
D Cases Across Users, Scenarios, and Tasks 32

## A Dataset Records and Evaluation Details

## A.1 Evaluation Set and Annotation Structure

Evaluation records correspond to videos rather than the life stories used for scenario construction; Table 2 reports their source composition and annotation counts. All main-experiment configurations use the same frozen records.

Record structure. Each record contains a case identifier, a standing task, contextual metadata, and a sequence of annotations. An annotation specifies a half-open source-time interval [s, e), an action, and Chinese and English reference content where a response is required. The two reference languages express the same requirement; a model need not produce both.

Reference actions. The four actions requiring a response are [inform], [warn], [feedback], and [answer]. The fifth category, [wait], requires no response. Counting and summarization requirements are included under [inform] in this release. Task-specific information, such as the correct count or completed state, is required for content correctness.

Dataset examples. Figure 7 presents ten examples across the five scenario domains, pairing synthetic and real-world videos. Each example shows source frames, bilingual task queries, reference responses, and intervals of required assistance and silence.

## A.2 Construction and Quality-Control Criteria

Scenario construction starts from a support need and defines an observable condition under which a particular response becomes useful. This condition depends on the standing task, rather than every visible change in the scene. For example, observing an object does not necessarily justify describing it when the task requires waiting for a question, a completed stage, or a hazard. A reference window specifies the interval in which the assistance is appropriate, rather than the first moment at which an object can be recognized.

The construction criteria distinguish evidence available in the clip from information used only for evaluation. Decision-relevant objects, actions, and state changes must be observable; an intended scenario alone cannot justify an annotation. Required answers must not be supplied as narration, subtitles, or other answer leakage. Synthetic clips additionally require plausible event progression and consistent participants and objects. Non-intervention intervals are explicit requirements, including in paired cases where the task-relevant condition is absent. These criteria describe the intended inclusion and annotation rules; they do not establish that every possible ambiguity has been eliminated or replace an independent annotation-reliability study.

![](images/454a62effabb531369a0f696ba5c2a2cec5d735c19ff5be88b7aee5dc2c71c61.jpg)  
Figure 7: Ten dataset examples across five scenario domains, pairing synthetic (left) and real-world (right) videos. Each case shows three source frames in temporal order, bilingual task queries, reference responses, and intervals of required assistance and silence.

The recorded tasks, media identities, and annotations are frozen before scoring. Each model receives the standing task and source media, while reference content, annotation windows, event captions, and QA answer keys are withheld from streaming inference. Population and scenario metadata support descriptive analyses of the recorded case mix. A task-derived assistant role does not establish which physical device recorded a video or hosted an assistant, and is not used as a substitute for a verified device label.

## A.3 Inference and Recording Configuration

Media delivery and session completion. Streaming inputs retain each interface’s native chunking: 1 s for MiniCPM, Gander, and Venus, and 480 ms for DuplexOmni. Gemini and Qwen receive 200-ms audio packets and 640-pixel-wide frames.

After source EOF, local adapters finish pending processing without adding media; MOSS additionally collects output for 2 s after its ASR flush. Gemini and Qwen collect output for 2 s after clip end. The ASR front ends transcribe the original source audio, rather than synthesizing speech from the task prompt.

Streaming generation. MiniCPM uses greedy decoding with at most 20 speech-text tokens per step. Gander samples with temperature $\Breve { T } = \Breve { 0 . 7 }$ , top-p = 0.8, top-k = 20, and an 11-token limit; Venus uses $T = 0 . 7$ , top-p = 0.8, top-k = 100, a 50-token limit, and length penalty 0.8. AURA uses $T = 0 . 5$ and a 128-token limit; JoyAI uses $T = 0 . 8$ , top-p = 0.9, top-k = 40, and a 128-token limit. DuplexOmni (S1) uses $T = 0 . 0 \dot { 1 }$ , top-p = 1, and a 999-token limit. These limits apply to native generation steps, not the entire video. Both MOSS models decode greedily using native silence and stop controls rather than an active token-length cap; MOSS-VL is paced at 12 tokens/s, while Preview uses its effectively unthrottled default. Gemini and Qwen use T = 0: Gemini enables proactive audio and sliding-window context management, while Qwen uses semantic voice-activity detection and seed 0.

Output normalization and records. Gemini’s native audio transcription is supplemented by ASR when a nonempty audio response lacks text. Response events preserve text, an optional tag, and source-aligned start/end timestamps. Per-case records preserve inputs, raw and parsed outputs, latency, runtime settings, failures, and reported API usage. Control markers are normalized; nonempty invalid responses remain false-positive candidates.

Local systems and Qwen return text, while Gemini’s onset is the first audio packet; onset otherwise denotes the first received text content. These intervals measure client-side delivery rather than loudspeaker playback.

For DuplexOmni (S1), the adapter groups consecutive nonempty tts fields from complete HTTP responses into an observable response segment, closed by an empty field, a [STOP] control, or source EOF. Normal native end-of-turn boundaries are not exposed by this interface.

QA generation. Local models use greedy decoding with a 64-token limit. Qwen and Gemini use temperature 0 with token limits of 64 and 1,024, respectively.

## A.4 Content Judgment and One-to-One Matching

DeepSeek-V4-Flash-0731 uses temperature 0, disabled thinking, and a 32-token limit. For an active reference node i and recorded response j, the binary content judgment $c _ { i j }$ assesses whether the response conveys the minimum information required by that node’s task and reference, without a substantive contradiction. Semantically equivalent wording and concise answers are accepted. Missing essential information, an incorrect entity or state, a monitoring promise in place of a required event report, and an emission without a usable answer are rejected. Optional reference details and behavior-tag correctness are not requirements for content credit.

Let $t _ { j }$ be the recorded onset of response $j .$ An eligible main-score edge satisfies

$$
c _ { i j } = 1 \quad { \mathrm { a n d } } \quad s _ { i } \leq t _ { j } < e _ { i } .\tag{4}
$$

We select a maximum-cardinality matching whose node and response indices both increase. This preserves temporal order and allows each node and each response to contribute at most one match. Responses are not reassigned using a guessed intended event, a timestamp mentioned in their text, or a corrected behavior tag.

Reference-window ends are extended only in the explicitly reported tolerance analysis. The content judge is held fixed when recomputing those matches.

## A.5 Silence, Video-Level Correctness, and Failures

The unwanted-response rate counts recorded response onsets inside reference [wait] intervals, divided by the total duration of those intervals in minutes. It includes silence intervals within intervention videos as well as fully silent videos. Thus it is distinct from the false-positive count in Intervention F1, which includes every unmatched response, including a wrong answer during an active window. Neither measure treats frequent output as assistance by itself.

Full Correct applies to the 290 intervention videos. It requires every active node to be matched, no extra response, no recorded response interval overlapping a silence interval, and successful session completion. Its interval-overlap check is stricter than counting only unwanted response onsets: a response can begin inside an active window yet continue into a quiet interval.

Silence Success instead applies to the 110 fully silent videos and requires a completed session with no actual output. Empty bookkeeping records and recognized control markers do not create useful assistance; a recorded nonempty response cannot be reclassified as successful silence merely because its behavior tag is missing or invalid.

All planned cases remain in the applicable denominators. An execution failure is reported separately from a completed session that emits no response, and cannot receive Full Correct or Silence Success credit. Interface-specific response grouping and source-aligned timestamps are retained.

## A.6 Human Baseline

Four annotators independently completed Chinese QA and streaming evaluations. Scores are unweighted annotator means, rounded after averaging, and measure reference-based performance rather than annotator agreement. Each annotator’s performance was first calculated separately, preserving the same evaluation units and scoring criteria across annotators.

QA scoring. Each annotator answers 400 content and 400 timing questions under the same scoring protocol, with missing or invalid answers counted as incorrect. Content and Timing Accuracy each use a denominator of 400; Joint Accuracy requires both answers per video to be correct across all 400 videos. Thus, a video contributes to Joint Accuracy only when both its content answer and its timing answer are correct. Correctness on either question alone is insufficient, and Joint Accuracy is not the arithmetic mean of the two component accuracies.

Streaming scoring. A judge assessed whether each response conveyed the essential information required by the task and the corresponding reference intervention. Semantically equivalent wording was accepted; missing essential information or substantive contradictions were considered incorrect. Human responses used the same timing rules, matching procedure, and metric definitions as model responses. A response received matching credit only when its content was correct and its recorded onset fell within the original reference interval $\left[ { { s _ { i } } , { e _ { i } } } \right)$ . Early or late responses received no matching credit. Content correctness and temporal validity were therefore jointly re quired: neither a timely but incorrect response nor a correct response outside the interval qualified. Matching was one-to-one, so each response and reference intervention could contribute at most one correct match.

Table 11: Population and service-group Intervention Recall (%), part 1 of 2. Cases and required nodes are fixed across models; failed sessions remain in the node denominators.
<table><tr><td>User / service group</td><td>Cases</td><td>Nodes</td><td>MiniCPM-o 4.5</td><td>MOSS-VL</td><td>MOSS Preview</td><td>Venus- Omni</td><td>Gander</td><td>AURA</td></tr><tr><td>Adults</td><td>172</td><td>229</td><td>3.06</td><td>4.37</td><td>0.44</td><td>1.75</td><td>1.75</td><td>14.41</td></tr><tr><td>Infants and toddlers</td><td>26</td><td>34</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.94</td></tr><tr><td>Schoolchildren</td><td>36</td><td>31</td><td>0.00</td><td>3.23</td><td>0.00</td><td>3.23</td><td>3.23</td><td>9.68</td></tr><tr><td>Older adults</td><td>29</td><td>58</td><td>1.72</td><td>8.62</td><td>1.72</td><td>1.72</td><td>0.00</td><td>22.41</td></tr><tr><td>Visual impairment</td><td>27</td><td>46</td><td>0.00</td><td>2.17</td><td>4.35</td><td>0.00</td><td>0.00</td><td>2.17</td></tr><tr><td>Hearing impairment</td><td>13</td><td>17</td><td>0.00</td><td>11.76</td><td>0.00</td><td>5.88</td><td>0.00</td><td>17.65</td></tr><tr><td>Color-vision deficiency</td><td>11</td><td>12</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>16.67</td></tr><tr><td>Mobility impairment</td><td>8</td><td>11</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Cognitive impairment</td><td>11</td><td>9</td><td>0.00</td><td>0.00</td><td>0.00</td><td>11.11</td><td>0.00</td><td>0.00</td></tr><tr><td>Chronic conditions</td><td>13</td><td>9</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>11.11</td></tr><tr><td>Multi-person settings</td><td>24</td><td>60</td><td>0.00</td><td>18.33</td><td>1.67</td><td>3.33</td><td>0.00</td><td>13.33</td></tr><tr><td>Pets</td><td>30</td><td>39</td><td>0.00</td><td>5.13</td><td>2.56</td><td>2.56</td><td>2.56</td><td>12.82</td></tr></table>

Score aggregation. For each metric, the reported human baseline is the unweighted arithmetic mean of the four annotators’ unrounded scores on the same benchmark. Rounding is applied only to the final mean. Each annotator therefore contributes equally to the reported baseline under the shared evaluation protocol. These scores measure performance against the benchmark references rather than agreement between annotators.

## B Complete Streaming Results

The following figures and tables supplement the main results using the same frozen content judgments and timing rules. Unless stated otherwise, every model or condition retains 400 videos, 555 required intervention nodes, 290 intervention videos, and 110 fully silent videos, including unsuccessful sessions. Intervention Recall and F1, Full Correct, and Silence Success follow the evaluation protocol in the main text. Unwanted responses are counted during annotated silence and divided by its duration in minutes. All displayed rates are rounded after aggregation.

## B.1 Population, Scenario, and Reference-Action Breakdown

Tables 11 and 12 report all population and service-group categories, with video and interventionnode counts. These are descriptive comparisons of the recorded tasks. Groups differ in scenario, observation conditions, and required interventions, and the smallest groups contain only nine required nodes. The results do not isolate a causal population effect or estimate performance for an entire real-world population. Pets and multi-person settings are retained as service groups in the original case labels.

Table 13 provides the complementary Intervention Recall breakdown for the five scenarios. These groups partition the 555 required intervention nodes. Recall measures the fraction of correctly matched nodes, while Full Correct in Table 6 requires the entire intervention video to succeed without extra output or silence overlap.

Table 14 groups recall by the reference action. These four non-silence actions partition the 555 required nodes, while the wait action is evaluated through silence metrics. Correctness depends on the response content and timing, rather than agreement between a predicted behavior tag and the reference action. The answer and feedback groups have substantially fewer nodes than inform and warn, so individual matched events have a larger effect on their percentages.

## B.2 Preventive, Corrective, and Routine Assistance

We conducted a model-assisted descriptive classification of the 400 videos using task requests and reference annotations, without consulting evaluated-model outputs or scores. The unit was a nonwait reference node, identified by case and annotation index. Each of the 555 nodes received one primary function: preventive assistance averts an imminent error or adverse outcome; corrective or remedial assistance manages an existing deviation or harmful condition; routine task support provides requested information, counts, answers, or state reports. Counts are 68, 157, and 330, respectively. Case-level function sets are unions of node labels, so a video can contain multiple

Table 12: Population and service-group Intervention Recall (%), part 2 of 2. Cases and required nodes are fixed across models; failed sessions remain in the node denominators.
<table><tr><td rowspan="2">User / service group</td><td rowspan="2">Cases Nodes JoyAI</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">DuplexOmni</td><td rowspan="2">Gemini 3.8 Live</td><td rowspan="2">Qwen3.5 Plus</td><td rowspan="2">Qwen3.5 Flash</td><td rowspan="2">Qwen3.8 Flash</td></tr><tr><td>(S1)</td></tr><tr><td>Adults</td><td>172 26</td><td>229 34</td><td>21.40 8.82</td><td>3.06 0.00</td><td>5.68 0.00</td><td>3.49 2.94</td><td>4.37 2.94</td><td>3.49 2.94</td></tr><tr><td>Infants and toddlers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Schoolchildren</td><td>36</td><td>31</td><td>35.48</td><td>3.23</td><td>3.23</td><td>3.23</td><td>3.23</td><td>3.23</td></tr><tr><td>Older adults Visual</td><td>29 27</td><td>58 46</td><td>27.59 13.04</td><td>6.90 6.52</td><td>0.00 0.00</td><td>0.00 6.52</td><td>0.00 2.17</td><td>0.00 4.35</td></tr><tr><td>impairment</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hearing impairment</td><td>13</td><td>17</td><td>35.29</td><td>0.00</td><td>0.00</td><td>0.00</td><td>5.88</td><td>5.88</td></tr><tr><td>Color-vision deficiency</td><td>11</td><td>12</td><td>41.67</td><td>8.33</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Mobility impairment</td><td>8</td><td>11</td><td>18.18</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Cognitive impairment Chronic</td><td>11</td><td>9</td><td>22.22</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>conditions</td><td>13</td><td>9</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Multi-person settings</td><td>24</td><td>60</td><td>8.33</td><td>10.00</td><td>5.00</td><td>10.00</td><td>11.67</td><td>1.67</td></tr><tr><td>Pets</td><td>30</td><td>39</td><td>15.38</td><td>7.69</td><td>5.13</td><td>5.13</td><td>5.13</td><td>2.56</td></tr></table>

Table 13: Intervention Recall by scenario (%, higher is better, ↑). Intervention-node counts give the denominators, including failed sessions. Bold marks the best value in each column.
<table><tr><td>Model</td><td>Daily living &amp; services ↑</td><td>Health &amp; caregiving ↑</td><td>Travel &amp; mobility ↑</td><td>Education &amp; work ↑</td><td>Sports &amp; leisure ↑</td></tr><tr><td>Intervention nodes</td><td>144</td><td>146</td><td>77</td><td>82</td><td>106</td></tr><tr><td>MiniCPM-o 4.5</td><td>0.69</td><td>0.68</td><td>1.30</td><td>1.22</td><td>3.77</td></tr><tr><td>MOSS-VL</td><td>9.72</td><td>4.79</td><td>3.90</td><td>7.32</td><td>1.89</td></tr><tr><td>MOSS Preview</td><td>2.08</td><td>0.68</td><td>1.30</td><td>0.00</td><td>0.94</td></tr><tr><td>Venus-Omni</td><td>3.47</td><td>0.68</td><td>3.90</td><td>2.44</td><td>0.00</td></tr><tr><td>Gander</td><td>2.08</td><td>0.68</td><td>0.00</td><td>2.44</td><td>0.00</td></tr><tr><td>AURA</td><td>17.36</td><td>13.01</td><td>10.39</td><td>3.66</td><td>14.15</td></tr><tr><td>JoyAI</td><td>25.00</td><td>16.44</td><td>18.18</td><td>29.27</td><td>12.26</td></tr><tr><td>DuplexOmni (S1)</td><td>8.33</td><td>3.42</td><td>7.79</td><td>1.22</td><td>0.94</td></tr><tr><td>Gemini 3.8 Live</td><td>6.25</td><td>1.37</td><td>2.60</td><td>2.44</td><td>3.77</td></tr><tr><td>Qwen3.5 Plus</td><td>6.25</td><td>2.05</td><td>2.60</td><td>2.44</td><td>4.72</td></tr><tr><td>Qwen3.5 Flash</td><td>6.94</td><td>2.05</td><td>2.60</td><td>4.88</td><td>3.77</td></tr><tr><td>Qwen3.8 Flash</td><td>4.17</td><td>1.37</td><td>2.60</td><td>3.66</td><td>1.89</td></tr></table>

Table 14: Intervention Recall (%) by reference action.
<table><tr><td>Model</td><td>Inform</td><td>Warn 362 nodes 149 nodes</td><td>29 nodes 15 nodes</td><td>Feedback Answer</td></tr><tr><td>MiniCPM-o 4.5</td><td>1.66</td><td>0.67</td><td>0.00</td><td>6.67</td></tr><tr><td>MOSS-VL</td><td>5.25</td><td>8.05</td><td>0.00</td><td>6.67</td></tr><tr><td>MOSS Preview</td><td>1.66</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Venus-Omni</td><td>1.93</td><td>2.01</td><td>0.00</td><td>6.67</td></tr><tr><td>Gander</td><td>1.10</td><td>0.00</td><td>0.00</td><td>13.33</td></tr><tr><td>AURA</td><td>11.88</td><td>15.44</td><td>6.90</td><td>13.33</td></tr><tr><td>JoyAI</td><td>18.23</td><td>26.17</td><td>17.24</td><td>6.67</td></tr><tr><td>DuplexOmni (S1)</td><td>3.59</td><td>6.71</td><td>3.45</td><td>6.67</td></tr><tr><td>Gemini 3.8 Live</td><td>3.04</td><td>1.34</td><td>0.00</td><td>40.00</td></tr><tr><td>Qwen3.5 Plus</td><td>3.59</td><td>3.36</td><td>0.00</td><td>20.00</td></tr><tr><td>Qwen3.5 Flash</td><td>4.97</td><td>1.34</td><td>0.00</td><td>20.00</td></tr><tr><td>Qwen3.8 Flash</td><td>1.93</td><td>2.01</td><td>0.00</td><td>33.33</td></tr></table>

## functions. The 110 silence-only videos retain empty function sets.

Classification follows the closest target problem rather than the action tag. An existing deviation takes precedence over a possible downstream consequence: stopping a child already touching an outlet is corrective even though it also aims to prevent electric shock. In elder box m pos, warning about a coffee table ahead is preventive, while post-fall guidance and seeking nearby help are remedial; both references use [warn]. For windows spanning preparation and execution, the primary label represents the best-supported service goal within the reference opportunity, without implying an invariant action phase or locating a violation’s onset.

An initial model-assisted pass and internal cross-review covered all 400 cases and 555 nodes. Reviewers saw the initial labels, so annotation was not blinded or independent. Boundary adjudication also consulted source event descriptions and extracted video frames. Sixty-two nodes retain review flags, including a low-confidence leash reminder: the camera view does not establish leash absence, so the label follows the precautionary reminder rather than asserting an existing violation. A child reaching toward an armor exhibit is likewise classified as preventive because contact is not established and the subsequent event describes an adult stopping the reach. Technique-coaching nodes follow the task and reference’s intended adjustment of current performance; this does not independently verify every technique judgment. The labels are not a human gold standard.

Temporal-state requirements were coded independently at task level. Seventy-four cases, including 69 intervention videos, require task-specific history such as cumulative counts, elapsed duration, ordered steps, or comparison with earlier observations; the other 326 are event-monitoring tasks under this definition. Generic monitoring instructions and multiple reference nodes alone do not establish the stateful label. In cnt cafe cup, the running count and final total require accumulated state, while quiet intervals still constrain when to speak.

Table 7 groups the frozen main-experiment matches by function without rematching subsets or changing content judgments, response windows, or model outputs. Each function’s denominator retains all its reference nodes, including those from failed sessions; previously credited partial outputs remain credited. Purely silent videos have no nodes in this table. Function-specific recall does not measure unwanted output, which must be read from the main silence and video-level metrics. Excluding the 62 flagged nodes as a boundary sensitivity check leaves 58 preventive, 122 corrective, and 313 routine nodes. MOSS-VL still covers corrective nodes more often than preventive nodes (8.20% versus 1.72%), while JoyAI retains higher preventive than routine recall (32.76% versus 14.70%). These comparisons describe the sampled tasks; task mix and observation conditions preclude interpreting them as causal effects of intervention function.

Table 15: Complete results for written and spoken task instructions on 400 videos per condition.
<table><tr><td>Task input</td><td>Intervention (%) Recall</td><td>F1</td><td>Full Correct (%)</td><td>Silence Success (%)</td><td>Unwanted /min</td><td>Failed</td></tr><tr><td colspan="7">MiniCPM-o 4.5</td></tr><tr><td>Written</td><td>1.44</td><td>2.63</td><td>1.38</td><td>95.45</td><td>0.14</td><td>0</td></tr><tr><td>Spoken</td><td>1.44</td><td>2.34</td><td>1.03</td><td>78.18</td><td>0.44</td><td>0</td></tr><tr><td>MOSS-VL</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Written</td><td>5.77</td><td>7.06</td><td>4.48</td><td>82.73</td><td>1.41</td><td>0</td></tr><tr><td>Spoken</td><td>13.51</td><td>9.41</td><td>2.76</td><td>16.36</td><td>4.64</td><td>0</td></tr><tr><td>Venus-Omni</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Written</td><td>1.98</td><td>3.18</td><td>2.07</td><td>77.27</td><td>0.48</td><td>0</td></tr><tr><td>Spoken</td><td>0.90</td><td>1.46</td><td>1.03</td><td>77.27</td><td>0.38</td><td>30</td></tr><tr><td>Qwen3.5 Flash</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Written</td><td>4.14</td><td>5.39</td><td>3.10</td><td>80.91</td><td>1.34</td><td>31</td></tr><tr><td>Spoken</td><td>6.85</td><td>6.09</td><td>2.76</td><td>0.00</td><td>3.44</td><td>3</td></tr></table>

## B.3 Complete Instruction-Modality Ablation

Table 15 reports the complete written-versus-spoken task-instruction results. The spoken condition replaces the written task with task TTS played once before the video. MiniCPM, Venus, and Qwen receive it through their audio input; MOSS-VL receives its transcript through the official SenseVoice path. Within each model, the general benchmark instruction, source media, model settings, and scoring rules remain fixed. All 400 videos per condition remain in the evaluation denominators, including failed sessions.

## C Failure Cases Analysis

We present four case studies that complement the failure analysis in Section 4.4. They illustrate missed assistance, incorrect object identification, unfulfilled task requirements, and unusable output. The cases include three real-world videos and one synthetic video. Source frames, reference windows, response timelines, and descriptions of relevant outputs support the analysis. Together, these materials connect each failure to the observed situation and the assistance requested by the user. These examples illustrate failures rather than their frequency in the benchmark.

Green timeline segments mark required response windows, red ticks substantive records, and gray ticks timestamp or silence markers. Times are seconds from video onset, with record onsets rounded to 0.001 s. Onsets indicate first receipt by the harness, not the timing of every later word. Reference intervals are half open and use evaluation duration clipping. Tail records remain visible but are excluded from reference-dependent judgments. Failure types can overlap; each case highlights a specific failure while retaining relevant timing and output details.

## C.1 Failing to correct an assembly error

Model: DuplexOmni (S1). Case: asm boom feedback 01. Source: real-world subset.

User task (English reference): Please watch me assemble the toy car, and correct me in time if I assemble it incorrectly.

The sole response comments on the initial alignment at 0.724 s. No later output is recorded, including the [22, 27) correction window. This is a missed intervention after an earlier response, rather than a response that arrives late. The initial comment does not satisfy the later requirement to identify and correct the assembly error when assistance is needed. Figure 8 shows the source frames and timeline, and Table 16 lists the reference windows and required content.

![](images/7b28ac8dfaaf06f931db0b969bb8fe8dfebc1d6289cd63ece7f3bb6a8c68e730.jpg)  
1 s: Assembly before the error

![](images/d43308afe72eb3c8678f47eaa303917a77b68fcbc832490e1ce03025c1655482.jpg)  
24 s: Reversed arm in the correction window

![](images/61f8f6543f9c8a7caf28f34f5d29f0f186174b33f343b36f0c7a1cfc2731c33a.jpg)  
Figure 8: Source frames and session timeline: DuplexOmni (S1) on asm boom feedback 01.

Table 16: Reference windows and required content for asm boom feedback 01.
<table><tr><td>Window (s)</td><td>Action</td><td>Required content (English reference)</td></tr><tr><td>[0,22)</td><td>wait</td><td>Remain silent.</td></tr><tr><td>[22,27)</td><td>feedback</td><td>The excavator arm is on backwards; it should bend toward the rear.</td></tr><tr><td>[27,60)</td><td>wait</td><td>Remain silent.</td></tr></table>

![](images/582f835f56d405b18932db7ba2ba97fce7731dc1326d3b8660ab521046d6d7ba.jpg)  
2.5 s: Object presented to the camera

![](images/9e66416bae31f2b298b82581546e18f19f9b0af4a235fdd708c3ea094b4a0d5b.jpg)  
4 s: Object remains visible

![](images/7001c50c77b3b25ac6d5ad74339d31733e6df177d088caaa609b17a35c96141f.jpg)  
Figure 9: Source frames and session timeline: AURA on orbit mug 06.

## C.2 Mistaking a thermos for a guitar

Model: AURA. Case: orbit mug 06. Source: real-world subset.

User task (English reference): Please identify the object in my hand and tell me what it is.

The substantive response begins at 5.075 s, inside the [1.5, 5.5) reference window, but identifies a black electric guitar instead of the stainless steel thermos. Although the response begins within the required interval, its central object identification is incorrect. The remaining nine records are timestamp strings. The qualitative grounding error concerns the stated identity, independent of these separate output artifacts. Figure 9 shows the source frames and timeline; Table 17 provides the references for evaluating the requested identification.

Table 17: Reference windows and required content for orbit mug 06.
<table><tr><td>Window (s)</td><td>Action</td><td>Required content (English reference)</td></tr><tr><td>[0, 1.5)</td><td>wait</td><td>Remain silent.</td></tr><tr><td>[1.5,5.5)</td><td>inform</td><td>You&#x27;re holding a stainless steel thermos.</td></tr><tr><td>[5.5, 19.7)</td><td>wait</td><td>Remain silent.</td></tr></table>

![](images/cafb977c9ad2ff2ce95d5b9c0cb5a4d5e0b447261d1cbbd5479ba6607d906357.jpg)  
3 s: Entrance before the guests arrive

![](images/3dcf48a584accfd1b83b4d06ffae839b4401046923148e4e9040d5ceb2e96127.jpg)  
9 s: Two guests enter the restaurant

![](images/2e5d0846890cc36d6a9bb8cdd029c224c67bcf9dc8eb74f20673057c734744db.jpg)  
Figure 10: Source frames and session timeline: MOSS-VL on robot host rest.

Table 18: Reference windows and required content for robot host rest.
<table><tr><td>Window (s)</td><td>Action</td><td>Required content (English reference)</td></tr><tr><td>[0,6)</td><td>wait</td><td>Remain silent.</td></tr><tr><td>[6, 11)</td><td>inform</td><td>Welcome, please feel free to take a seat.</td></tr><tr><td>[11,30.08)</td><td>wait</td><td>Remain silent.</td></tr></table>

## C.3 Saying guests were greeted instead of greeting them

Model: MOSS-VL. Case: robot host rest. Source: synthetic subset.

User task (English reference): Please watch for guests entering the restaurant, greet them when they arrive, and tell them they may take a seat.

The only response starts at 7.189 s, within the [6, 11) greeting window. It reports that the guests have already been greeted, but never addresses them with a greeting or an invitation to sit. Describing the requested action as completed does not perform the guest-facing assistance specified in the task. This case illustrates a task-fulfillment failure: the response is timely and relevant to the event, but does not fulfill the requested assistance. Figure 10 shows the source frames and timeline, and Table 18 provides the references.

## C.4 Outputting timestamps instead of reporting a leak in time

## Model: AURA. Case: roof leak. Source: real-world subset.

User task (English reference): Please monitor the room and tell me the location and situation if you observe a leak or persistent dripping.

The record at 3.066 s is only <2.5 seconds>, inside the [2, 5) leak-report window. The second [22, 25) reference window likewise contains only a timestamp record at 23.200 s. The first substantive statement arrives at 57.093 s. Timestamp strings do not convey the requested location or situation, so the presence of a record within a response window does not itself constitute usable assistance. These are recorded interface outputs; this evidence does not establish that the strings were spoken aloud. The record after the reference horizon is shown for completeness and is not used for reference-dependent classification. Figure 11 shows the source frames and timeline, and Table 19 lists the reference windows and required content.

![](images/b7a294de88aa8bf8a58f9d36851408dae08c68ed0e38458b83f3712ddba4978e.jpg)

![](images/7bd42e28a3b036184d4afd0bbb2933413aa1ecf22b06972563493c2c4a90046d.jpg)

![](images/f4e005058db37527aec05750e2d922ce33e90580509f6686e51f7030a2571b04.jpg)  
Figure 11: Source frames and session timeline: AURA on roof leak.

Table 19: Reference windows and required content for roof leak.
<table><tr><td>Window (s)</td><td>Action</td><td>Required content (English reference)</td></tr><tr><td>[0,2)</td><td>wait</td><td>Remain silent.</td></tr><tr><td>[2,5)</td><td>inform</td><td>Water is leaking near a light fixture in the kitchen ceiling.</td></tr><tr><td>[5,22)</td><td>wait</td><td>Remain silent.</td></tr><tr><td>[22,25) [25,72.365)</td><td>inform wait</td><td>Several containers and cloths on the floor are catching the leaking water. Remain silent.</td></tr></table>

## D Cases Across Users, Scenarios, and Tasks

We present 14 failure cases across user groups, five scenarios, and five failure categories. Eight videos are real and six are controlled synthetic videos. These cases illustrate failure mechanisms rather than their benchmark frequency. Together, these examples show how assistance can fail through inappropriate timing, inaccurate interpretation of the situation, or incomplete fulfillment of the user’s specific requested task. The paired cases also distinguish between producing a response, providing the information the task requires, and intervening only when the reference explicitly calls for assistance. The analysis considers both what the assistant says and whether that response satisfies the user’s need at the relevant moment. Table 20 summarizes coverage; Figures 12–18 pair video evidence with response analysis.

Source frames appear on the left; reference requirements, model responses, and analysis appear on the right. Frames within each panel follow chronological order. This arrangement allows readers to compare the observed evidence, expected behavior, and recorded output while checking whether each response occurs within the required temporal window. Green timeline segments mark response windows, gray segments mark required silence, and red points mark recorded response onsets. Responses are translated into English; ellipses mark omitted text, and protocol tags are omitted for readability. Frame aspect ratios are preserved; C07 includes an enlarged view of the original image pixels. Interpretation combines the displayed frames with annotations and recorded outputs, since individual images cannot independently establish every event or the content of an auditory signal. Case-specific discussion below explains these distinctions while retaining the evidence limitations associated with visual observations, annotated events, and interface records.

Table 20: Coverage of the 14 cases. Population labels describe benchmark tasks.
<table><tr><td>Case</td><td>Service population</td><td>User need</td><td>Scene</td><td>Failure</td></tr><tr><td>C01</td><td>Adults</td><td>Summarize after coffee preparation</td><td>S1</td><td>F3</td></tr><tr><td>C02</td><td>Infants and toddlers</td><td>Choking alert</td><td>S2</td><td>F2</td></tr><tr><td>C03</td><td>Older adults</td><td>Timely help</td><td>S2</td><td>F1</td></tr><tr><td>C04</td><td>Cognitive impairment</td><td>Remember item location</td><td>S1</td><td>F4</td></tr><tr><td>C05</td><td>Multi-person settings</td><td>Fall alert</td><td>S2</td><td>F2</td></tr><tr><td>C06</td><td>Visual impairment</td><td>Hazard warning</td><td>S3</td><td>F2</td></tr><tr><td>C07</td><td>Color-vision deficiency</td><td>Wire-connection guidance</td><td>S4</td><td>F2</td></tr><tr><td>C08</td><td>Mobility impairment</td><td>Movement progress</td><td>S3</td><td>F3</td></tr><tr><td>C09</td><td>Schoolchildren</td><td>Clarify a question</td><td>S4</td><td>F4</td></tr><tr><td>C10</td><td>Multi-person settings</td><td>Cross-player turn order</td><td>S5</td><td>F1</td></tr><tr><td>C11</td><td>Pets</td><td>Pet-care need</td><td>S2</td><td>F2</td></tr><tr><td>C12</td><td>Hearing impairment</td><td>Sound amid background talk</td><td>S1</td><td>F4</td></tr><tr><td>C13</td><td>Schoolchildren</td><td>Do not interrupt</td><td>S4</td><td>F1</td></tr><tr><td>C14</td><td>Hearing impairment</td><td>Usable alarm notification</td><td>S1</td><td>F5</td></tr></table>

Scenarios. S1: daily living and services; S2: health and caregiving; S3: travel and mobility; S4: education and work; S5: sports and leisure.  
Failure categories. F1: intervention control; F2: situational grounding; F3: temporal tracking; F4: task fulfillment; F5: unusable output.

C01 Report completion after making coffee  
![](images/6710e8db8eb1c72709abb8cb7d091bfa8efb7ab4910f7942f3642fcf698f00d5.jpg)  
Pouring milk into the pitcher.  
Expected response  
Wait until the coffee is finished.  
Model response (Qwen3.5 Plus)  
“The coffee is ready. …  
Explanation: Declares completion while milk is being poured.

C02 Watch the child and warn of danger  
![](images/85e5ba5270eef61c6732d587e9cbe69cd575c1d89e611fff75ff0ebbd57f120f.jpg)  
Expected response  
“The baby is choking; the mother is giving first aid.”  
First aid for choking.  
Model response (AURA)  
“…keeps eating and even smiles.”  
Explanation: Misses choking; reports eating and smiling.

Figure 12: C01 (coffee ego; real video) and C02 (baby danger; real video).

Figure 12 contrasts temporal tracking and situational grounding failures. C01 declares completion while milk is still being poured, confusing an intermediate step with the requested endpoint. C02 responds on time but describes eating and smiling instead of the annotated choking event. The choking interpretation relies on video annotations; still frames alone cannot establish it.

“…the woman is standing. …No danger…

C03 Warn when an older person needs help  
![](images/429d7dcf1b445d31dee15861c0a7cf2c20b34ce26fdf2dcb69d99c4b1cd84aab.jpg)  
Reaching beside a step.

Expected response   
Risk of falling; help the older person now.   
Model response (MiniCPM-o 4.5)   
“…watch out for slipping and falling.”   
Explanation: Warns at 85.56 s, after the 0–8 s help window.

C04 Remember a location and help find the item later  
![](images/380ba1dc43194da9886f8a278cf2c89b38fc0fa44b78971014c53c91a6da9880.jpg)  
Placement, then later search.

Figure 14: C05 (dining help; real video) and C06 (blind cane hole; controlled synthetic). Figure 14 presents two timely responses with incorrect hazard assessments. C05 reports no danger during a fall, while C06 advises continued movement toward an open manhole. The failure concerns the content of the guidance despite its onset falling within the required window.  
Figure 13: C03 (elder mobility; real video) and C04 (mon alz find item; controlled synthetic). Figure 13 illustrates late assistance and incomplete recall. C03 warns after the required help window, missing the opportunity for timely intervention. C04 responds within its window but offers only an acknowledgment, omitting the remembered item location needed to fulfill the request for help finding the item.  
C05 Detect a fall and alert the caregiver  
![](images/a0e10edad4ef507745fbbc3b6f632904df76c92cc6941e7eac8067ad969c4293.jpg)  
Expected response  
Someone has fallen; please check on them.  
Model response (MOSS Preview)  
Falling beside the table.

C06 Warn a white-cane user about a hazard ahead  
![](images/7a2130f8cb89c2f1eaf250f0b160a6661a7ec568067ead7ecec8c17f9257e39d.jpg)  
Open manhole ahead.

C07 Guide a wire-connection task  
![](images/ac7af8f98cacc365f4ca955bbd12ef51f1affe08d3ed0c162162ea2c4682c3ca.jpg)  
C08 Track progress while moving on the stairs

![](images/aaf1595b91adca26a31976432cf8e0bca8eca5546cb65b79f0240736987292c6.jpg)

She is supporting herself and moving up.

Model response (AURA)

“…finished going down the stairs.”

Moving farther up the stairs.

Figure 15: C07 (cblind wire 01; controlled synthetic) and C08 (wheel stairs paralyzed; real video).

Figure 15 highlights errors in attribute identification and movement tracking. C07 calls the white terminal red, undermining the requested wire-connection guidance. C08 reports completed descent while the person is moving upward. Both responses begin within their reference windows but misrepresent visual details essential to the task.

C09 Help a child with an unreadable question  
![](images/b7a99affc7c8190fa88e0a920d87cbad92533e8a477c6a6b7c259933eb9ce823.jpg)  
Question on the desk.

C10 Follow the card-drawing order across players  
![](images/4f6272b5197c3f7235bf35677584330dee06c86998bfd4683cbc1bf58fa27144.jpg)  
Red, blue, yellow; yellow again.

Figure 16: C09 (tutor hold; controlled synthetic) and C10 (life poker turns; controlled synthetic). Figure 16 pairs an unsupported answer with a missed intervention. C09 selects an option instead of requesting a clearer view of the question. The evidence establishes insufficient support for the answer, not that the option is wrong. C10 has no recorded response during the intervention window, leaving the repeated turn by the player in yellow unaddressed.

<table><tr><td>Reply at 9.18 s (on time)</td></tr><tr><td></td></tr><tr><td>5.00 s 10.00 s</td></tr></table>

![](images/9cdce5c4d64291546613d33e514293446601fbbb3a0e88b4b4e674afe725d08d.jpg)  
Chime, then background speech. <sub>Reply</sub> <sub>window</sub>  
Figure 17: C11 (pet bowl ego; real video) and C12 (deaf sign home; real video).  
C14 Turn an alarm event into a usable notification  
Figure 18: C13 (kid phone neg; controlled synthetic) and C14 (deaf wake; real video).

C11 Notice when the dog needs care  
![](images/40eaa9be99ff69d4663dafe7aacf7d15a2edf23b0917e6168fe65e91956ad80a.jpg)  
Dog beside an empty bowl.  
Reply window

Expected response   
Empty bowl; the dog may need food.   
Model response (JoyAI)   
“The dog is eating from its bowl.”   
Explanation: Reports eating, but the bowl is empty. Expected response   
Someone is ringing the front doorbell.   
Model response (Qwen3.5 Plus)   
“…telling you to go to the back door.”   
Explanation: Repeats background speech; misses the doorbell.

C12 Report the target doorbell amid background speech  
![](images/adee4b719d990ab9b0bb5a881fdaf930754251ccd6301cd13268b1b78508f2fd.jpg)

Figure 17 illustrates missed care cues and difficulty separating target sounds from background speech. C11 describes eating despite the empty bowl, missing the cue relevant to the dog’s care needs. C12 repeats background instructions instead of reporting the front-door chime as required by the task. The chime is identified in video annotations, and speech transcripts confirm the competing background instructions.

## C13 Allow a brief phone check without interrupting

![](images/8115fdd21fcc89577b9a70c14f788b4fe093fe3a66f0117d6442643b8233b211.jpg)  
Phone check, then homework.

Figure 18 contrasts an unnecessary intervention with an unusable output. C13 interrupts a permitted brief phone check during a required silence interval, despite the subsequent return to homework. C14 returns a timestamp within the response window without an informative alarm notification. This is a recorded interface output; the evidence does not establish that the timestamp was spoken aloud.