# Speculative Reasoning and Tool Execution While Listening in Realtime Voice Agents

Zhenyu Zhang Kewei Fu Yongqi Tong Chenzhi Zheng Wei Wu Jiang-Ming Yang Jianshe Li

Ant International

Correspondence: tongyongqi.yq@ant-intl.com

## Abstract

Tool-using voice agents often leave users waiting after speech ends, even when actionable information arrives earlier in the request. Speculative approaches begin work during speech, but chunk-bounded or interrupt-on-update generation can truncate a tool call before its arguments are complete. This couples computation time to speech updates, limiting useful work completed in advance. We introduce Heartbeat Thinking, which handles updated input through new Thinker invocations while allowing ongoing invocations and dispatched tools to finish. Periodic invocations read the latest partial transcript and a shared record of completed reasoning and tool results. These results help resolve missing arguments and initiate dependent calls, allowing preparation to progress across speech updates. Across SpeechFCEval, FDBench v3, and AudioCRAG, Heartbeat improves aggregate speculative tool coverage or retrieval reuse over a SHANKS-style windowed baseline, with benchmark-dependent efects on final task quality. On FDBench v3, it increases tool coverage by 16 percentage points and reduces mean final-response latency by 36.4%. The results show that heartbeat thinking paradigm preserving ongoing computation during speech can turn listening time into reusable preparation for faster and more capable voice-agent responses.

Keywords: voice agents, speculative reasoning, tool execution

## 1 Introduction

Voice agents use external tools to retrieve information and carry out requests that depend on actions beyond the conversation itself. For these tasks, producing a useful answer requires both model computation and external execution, which can leave users waiting after they finish speaking. Conventional cascaded pipelines often place this preparation after speech ends. Yet the information needed to begin may arrive much earlier: an agent can identify an order while the user is still describing a delivery concern. Prior work has shown that reasoning and tool use can begin on partial speech, allowing some of this preparation to take place during listening (Zhou et al., 2022; Chiang et al., 2026; Arora et al., 2026; Hooper et al., 2026). The opportunity is to reduce how much work remains when the agent must answer the complete request.

Early preparation is useful only when it produces work that can be reused for the full request. A tool call cannot be dispatched while its required arguments remain incomplete, even if the user has already provided enough information to construct them. SHANKS carries reasoning forward but bounds each generation episode by a speech chunk, which can truncate unfinished tool-call arguments (Chiang et al., 2026). SIA incorporates new events by interrupting ongoing generation; although dispatched tools are managed separately, a call may be interrupted before it is issued (Hooper et al., 2026). Frequent updates can therefore limit the uninterrupted time available to form a complete call. The evolving request determines whether completed results remain useful: an early search may need revision when the user adds a budget constraint. Stream RAG explores parallel retrieval and checks whether intermediate results can serve the final request (Arora et al., 2026). These considerations motivate a preparation policy that allows computation to finish across speech updates while letting later input determine which results still apply.

We therefore introduce Heartbeat Thinking, which handles new speech in subsequent invocations while allowing work already underway to finish (Figure 2). During listening, periodic Thinker invocations read the latest partial transcript and a shared record of completed reasoning and tool results. Independent invocations can overlap: one can finish constructing a tool call while another processes updated input. Requested tools execute asynchronously, and their returned results enter the shared record. Later invocations can use these results to resolve missing arguments and initiate depen dent calls. Preparation can thus advance through successive reasoning and tool-use steps even when the transcript has not changed, without requiring each invocation to finish within an input-update window.

![](images/36a64fbd2f73f022cbddd889504a5e45393f8c4a0d2c93e014ae6c76643f4961.jpg)

To make this accumulated preparation useful for the final answer, Heartbeat separates speculative preparation from response generation. At the detected end of speech, a Responder receives the complete request and the completed preparation available at the handof. It checks completed work against the full request, reuses applicable results, and performs the remaining calls. Unfinished speculative jobs are cancelled at this handof, allowing the Responder to begin without waiting for their completion. This explicit exchange of reasoning records and tool results also allows the Thinker and Responder to use independently selected models.

Experiments on SpeechFCEval, FDBench v3, and AudioCRAG show higher aggregate speculative tool coverage or retrieval reuse than SHANKS-style Window Thinking. On FDBench v3, coverage increases from 34.67% to 50.67%, while mean final-response latency falls from 6.90 to 4.39 seconds (36.4%), also 45.7% below Pipecat, the fastest evaluated cascaded baseline. The coverage advantage persists across tested Responders. Task quality improves over Window Thinking on FDBench but decreases slightly on SpeechFCEval and AudioCRAG. Thus, computation spanning speech updates can increase reusable preparation and reduce final-response latency, while interpreting and using speculative results remains a separate challenge.

## 2 Related Work

Early preparation spans prefix-based reasoning in Live-Mind (Chen et al., 2024), response drafting in PredGen (Li and Grover, 2025), and maintained reasoning state in LTS-VoiceAgent (Zou et al., 2026). Figure 1 contrasts when reasoning and tool use occur in cascaded pipelines, SHANKS, and Heartbeat. Decoupled systems exchange preparation through completed tool results in SHANKS (Chiang et al., 2026), streamed outputs in DuplexOmni (Huang et al., 2026), or model-internal state in AsyncReasoning (Yakushev et al., 2025). Heartbeat connects independently scheduled invocations through completed reasoning records and tool results. Appendix F discusses these interfaces and streaming computation policies in more detail.

## 3 Heartbeat Thinking

Heartbeat receives streaming ASR transcripts, dialogue history, and tool schemas. A periodic Thinker prepares reasoning and tool calls during listening; an endpoint-triggered

Figure 1: Reasoning and tool use during speech. (a) Cascaded baselines defer reasoning and tool calls until speech ends. (b) SHANKS may truncate tool-call generation at chunk boundaries (Chiang et al., 2026). (c) Heartbeat overlaps Thinker invocations and tools across speech updates; the Responder reuses completed preparation without waiting for pending jobs.

Responder answers the complete request. The roles can use diferent models. Shared state stores completed reasoning records, requested calls, and their returned results. Figure 2 shows the architecture and execution timeline. Appendix A provides implementation details and the output schema.

Overlapping preparation. Every Δ seconds, a new Thinker invocation reads the latest transcript and completed preparation. At most � invocations run concurrently; a trigger is skipped when all slots are occupied. New speech and tool results enter subsequent invocations without interrupting ongoing generation. An invocation can take no action, return reasoning, or request tools with structured arguments. Completed reasoning is recorded before tools are dispatched asynchronously, so model and tool execution can overlap. Later invocations can use returned results to resolve missing arguments or initiate dependent calls, even without a new transcript update. The prompt directs the Thinker to avoid repeating pending or completed calls with identical names and arguments. Tool execution is queued separately from Thinker scheduling.

Response handoff. At the detected end of speech, the scheduler stops new invocations and cancels unfinished Thinker and speculative tool tasks, including queued calls.

![](images/39789dcf524f106dd2e32754e39ca7673508f42e9e3bd6668fa80dca18e51382.jpg)

Figure 2: Heartbeat Thinking architecture (top) and execution timeline (bottom). Thinker invocations and tools overlap during listening. At the VAD-triggered handof �, unfinished speculative jobs are cancelled; the Responder uses completed preparation available at � without waiting for pending work.
<table><tr><td rowspan="2">Framework</td><td colspan="2">SpeechFCEval</td><td colspan="3">FDBench v3</td><td colspan="2">AudioCRAG</td></tr><tr><td>Acc.</td><td>STC</td><td>Quality</td><td>Pass@1</td><td>STC</td><td>Correct</td><td>STC</td></tr><tr><td>Speech-to-Speech</td><td>58.66</td><td>0.00</td><td>84.0</td><td>75.0</td><td>0.00</td><td>56.48</td><td>0.00</td></tr><tr><td>Pipecat</td><td>56.83</td><td>0.00</td><td>78.0</td><td>63.0</td><td>0.00</td><td>56.94</td><td>0.00</td></tr><tr><td>LiveKit</td><td>52.85</td><td>0.00</td><td>77.6</td><td>67.0</td><td>0.00</td><td>59.45</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>67.95</td><td>33.13</td><td>72.0</td><td>57.0</td><td>34.67</td><td>57.24</td><td>20.05</td></tr><tr><td>Heartbeat</td><td>66.85</td><td>43.69</td><td>77.0</td><td>65.0</td><td>50.67</td><td>55.49</td><td>62.42</td></tr></table>

Table 1: Main results (%; higher is better). Speculative systems use Qwen3.6-35B-A3B as Thinker; all systems use DeepSeek-V4-Flash as Responder. Further task and model results are in Appendix C.

The Responder receives the final transcript, history, schemas, and a snapshot of completed preparation available at context assembly. Later speculative outputs are discarded. Through a ReAct process (Yao et al., 2023), it checks preparation against the complete request, rejects interpretations contradicted by later speech, and reuses applicable completed results. Required calls lacking available results are executed within the Responder’s own process. It therefore begins without waiting for unfinished preparation. Appendix E.1 details the availability boundary; Appendices G.1–G.3 give

the prompts.

## 4 Evaluation

## 4.1 Setup and Metrics

We evaluate function calling on SpeechFCEval, intent following under disfluency on FDBench v3, and knowledgegrounded answering on AudioCRAG (Tongyi Fun Team et al., 2026; Lin et al., 2026; Arora et al., 2026). Baselines are Speech-to-Speech, Pipecat, LiveKit (Hugging Face, 2024; Daily, 2024; LiveKit, 2023), and SHANKS-style Window Thinking. Window Thinking follows Heartbeat’s update schedule but discards outputs arriving after their windows. We use Qwen3.6-35B-A3B as Thinker, DeepSeek-V4-Flash (0731) as Responder, Δ = 500 ms, and � = 2.

<table><tr><td>Framework</td><td>Avg. Acc. ↑</td><td>STC ↑</td></tr><tr><td>Responder: DeepSeek-V4-Flash</td><td></td><td></td></tr><tr><td>Speech-to-Speech</td><td>58.66</td><td>0.00</td></tr><tr><td>Pipecat</td><td>56.83</td><td>0.00</td></tr><tr><td>LiveKit</td><td>52.85</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>67.95</td><td>33.13</td></tr><tr><td>Heartbeat</td><td>66.85</td><td>43.69</td></tr><tr><td>Responder: Qwen3.5-397B-A17B</td><td></td><td></td></tr><tr><td>Speech-to-Speech</td><td>67.01</td><td>0.00</td></tr><tr><td>Pipecat</td><td>66.46</td><td>0.00</td></tr><tr><td>LiveKit</td><td>65.93</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>67.27</td><td>32.49</td></tr><tr><td>Heartbeat</td><td>69.70</td><td>41.20</td></tr><tr><td>Responder: Qwen3.8-Flash-Next</td><td></td><td></td></tr><tr><td>Speech-to-Speech</td><td>53.66</td><td>0.00</td></tr><tr><td>Pipecat</td><td>42.76</td><td>0.00</td></tr><tr><td>LiveKit</td><td>43.42</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>57.14</td><td>44.05</td></tr><tr><td>Heartbeat</td><td>64.30</td><td>55.27</td></tr></table>

Table 2: SpeechFCEval accuracy and STC across Responders (%), with Qwen3.6-35B-A3B as Thinker.

<table><tr><td>Framework</td><td>Quality ↑</td><td>Pass@1 ↑</td><td>STC ↑</td></tr><tr><td colspan="4">Responder: DeepSeek-V4-Flash</td></tr><tr><td>Speech-to-Speech</td><td>84.0</td><td>75.0</td><td>0.0</td></tr><tr><td>Pipecat</td><td>78.0</td><td>63.0</td><td>0.0</td></tr><tr><td>LiveKit</td><td>77.6</td><td>67.0</td><td>0.0</td></tr><tr><td>Window Thinking (SHANKS)</td><td>72.0</td><td>57.0</td><td>34.67</td></tr><tr><td>Heartbeat</td><td>77.0</td><td>65.0</td><td>50.67</td></tr><tr><td colspan="4">Responder: Qwen3.5-397B-A17B</td></tr><tr><td>Speech-to-Speech</td><td>83.0</td><td>68.0</td><td>0.0</td></tr><tr><td>Pipecat</td><td>63.0</td><td>43.0</td><td>0.0</td></tr><tr><td>LiveKit</td><td>77.0</td><td>75.0</td><td>0.0</td></tr><tr><td>Window Thinking (SHANKS)</td><td>76.0</td><td>62.0</td><td>34.67</td></tr><tr><td>Heartbeat</td><td>77.0</td><td>61.0</td><td>45.33</td></tr><tr><td colspan="4">Responder: Qwen3.8-Flash-Next</td></tr><tr><td>Speech-to-Speech</td><td>83.0</td><td>62.0</td><td>0.0</td></tr><tr><td>Pipecat</td><td>83.0</td><td>60.0</td><td>0.0</td></tr><tr><td>LiveKit</td><td>72.0</td><td>55.0</td><td>0.0</td></tr><tr><td>Window Thinking (SHANKS)</td><td>76.0</td><td>57.0</td><td>34.00</td></tr><tr><td>Heartbeat</td><td>74.0</td><td>59.0</td><td>50.67</td></tr></table>

Table 3: FDBench v3 results across Responders (%), with Qwen3.6- 35B-A3B as Thinker.

Speculative tool coverage (STC) is the fraction of reference calls covered by completed matching speculative calls whose results enter the Responder context without the Responder repeating those calls. On AudioCRAG, it instead measures the fraction of samples reusing completed speculative results without additional retrieval. Task quality uses SpeechFCEval function-name/argument accuracy, FDBench Quality and Pass@1, and AudioCRAG correct-answer rate. Appendix B gives dataset sizes, matching and aggregation rules, and scoring protocols.

![](images/faa9f48b99a6159b713b83b75dfd867820143bf8f80889c5182f579b690abcc0.jpg)  
Figure 3: Mean Correct Tool Call Latency on FDBench v3 relative to speech end, measured at initiation of the first tool call matching the final request. Negative values indicate events occurring before speech end. Numerical results are reported in Table 5.

![](images/a4269b6571f2a96771904c3b0497d9931f0ff04685e9398827387f2c9ed9fa7a.jpg)  
Figure 4: Mean Final Response Latency on FDBench v3 relative to speech end, measured at the onset of the final spoken response. Details are reported in Table 5.

## 4.2 Results

Table 1 shows higher STC for Heartbeat than Window Thinking on all three benchmarks: gains of 10.56, 16.00, and 42.37 percentage points on SpeechFCEval, FDBench, and AudioCRAG. On FDBench, Quality rises from 72.0% to 77.0% and Pass@1 from 57.0% to 65.0%, although Pass@1 remains below Speech-to-Speech and LiveKit. Final task performance decreases on SpeechFCEval (67.95% to 66.85%) and AudioCRAG (57.24% to 55.49%). More retained preparation therefore does not consistently yield better final task performance.

Latency. Figures 3 and 4 show mean FDBench latencies relative to speech end. Correct Tool Call Latency measures initiation of the first matching execution, conditional on a matching call. Heartbeat starts at −4.37 s versus −2.74 s for Window Thinking and 2.45–3.52 s for cascaded baselines. Final Response Latency measures the onset of the final spoken answer after required tool calls, excluding earlier acknowledgments. Heartbeat reduces it from Window Thinking’s 6.90 to 4.39 s (36.4%), also 45.7% below Pipecat, the fastest evaluated cascaded baseline. Appendix B.2 gives the protocol and numerical results.

Additional analyses. Heartbeat retains higher STC across all three tested Responders (Tables 2 and 3; Appendix C.2). Later invocations can incorporate late constraints and explicit intent revisions before speech ends (Appendix D). Yet 4.0% of FDBench recordings yield answers using erroneous or outdated results despite correct tool-call arguments, distinguishing call correctness from result use.

## 5 Conclusion

Heartbeat Thinking overlaps speculative computation across speech updates and passes completed preparation to a separate Responder. Across three benchmarks, it retains more reusable preparation than a SHANKS-style windowed baseline; on FDBench, it also reduces final-response latency. Task-quality changes vary by benchmark, highlighting the importance of interpreting and using speculative results correctly.

## Limitations

Heartbeat uses a fixed scheduling interval and a bounded number of concurrent Thinker invocations. We have not explored adaptive scheduling based on changes in the partial transcript or the completion of tool calls. Such scheduling could reduce redundant inference when little new information is available. The Responder also receives a snapshot of completed preparation at context assembly, excluding tool results returned afterward. Future work could explore incremental delivery of late results, particularly for tools with variable execution times. Finally, our latency analysis focuses on FDBench v3, and further evaluation across different tool workloads would help characterize the trade-of between speculative coverage, inference cost, and response latency.

## Ethical Considerations

Speech and dialogue traces may contain personal information; applicable consent and access controls must govern their collection and use. The SpeechFCEval dataset card declares Apache-2.0 and asks users to respect the terms of its upstream sources.<sup>1</sup> The FDBench repository and the original CRAG text-data repository publish CC BY-NC 4.0 licenses.<sup>2</sup> These upstream terms do not establish permission to redistribute derived audio or identifiable dialogue traces.

## References

Siddhant Arora, Haidar Khan, Kai Sun, Xin Luna Dong, Sajal Choudhary, Seungwhan Moon, Xinyuan Zhang, Adithya Sagar, Surya Teja Appini, Kaushik Patnaik, Sanat Sharma, Shinji Watanabe, Anuj Kumar, Ahmed A. Aly, Yue Liu, Florian Metze, and Zhaojiang Lin. 2026. Stream RAG: Instant and accurate spoken dialogue systems with streaming tool usage. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 3633–3663. PMLR.

Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Ulf Schlichtmann, and Bing Li. 2024. LiveMind: Low-latency large language models with simultaneous inference. arXiv preprint arXiv:2406.14319.

Cheng-Han Chiang, Xiaofei Wang, Linjie Li, Chung-Ching Lin, Kevin Lin, Shujie Liu, Zhendong Wang, Zhengyuan Yang, Hung yi Lee, and Lijuan Wang. 2026. SHANKS: Simultaneous hearing and thinking for spoken language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8951–8972, San Diego, California, United States. Association for Computational Linguistics.

Daily. 2024. Pipecat: An open source framework for voice and multimodal conversational AI. GitHub repository. Accessed September 16, 2026.

In Gim, Seung seob Lee, and Lin Zhong. 2024. Asynchronous LLM function calling. Preprint, arXiv:2412.07017.

Coleman Hooper, Minwoo Kang, Suhong Moon, Nicholas Lee, Eric Wen, John Wawrzynek, Michael W. Mahoney, Yakun Sophia Shao, Amir Gholami, and Kurt Keutzer. 2026. Speculative interaction agents: Building real-time agents with asynchronous I/O and speculative tool calling. Preprint, arXiv:2605.13360.

Muye Huang, Lingling Zhang, Xingyu Yu, Lei Shi, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, and Jun Liu. 2026. DuplexOmni: Real-time listening, seeing, thinking, and speaking for full-duplex interaction. arXiv preprint arXiv:2606.09186.

Hugging Face. 2024. Speech-to-Speech: Build voice agents with open-source models. GitHub repository. Accessed September 16, 2026.

Shufan Li and Aditya Grover. 2025. PredGen: Accelerated inference of large language models through inputtime speculation for real-time speech interaction. arXiv preprint arXiv:2506.15556. Version 2.

Guan-Ting Lin, Chen Chen, Zhehuai Chen, and Hung yi Lee. 2026. Full-Duplex-Bench-v3: Benchmarking tool use for full-duplex voice agents under real-world disfluency. Preprint, arXiv:2604.04847.

LiveKit. 2023. LiveKit Agents: A framework for building realtime voice AI agents. GitHub repository. Accessed September 16, 2026.

Yifan Sui, Han Zhao, Rui Ma, Zhiyuan He, Hao Wang, Jianxun Li, Kaiqiang Xu, Kai Chen, and Yuqing Yang. 2026. Parallelizing tool execution and LLM generation for low-latency agent serving. arXiv preprint arXiv:2603.18897. Version 3.

Tongyi Fun Team, Qian Chen, Luyao Cheng, Chong Deng, Xiangang Li, Jiaqing Liu, Chao-Hong Tan, Wen Wang, Junhao Xu, Jieping Ye, Qinglin Zhang, Qiquan Zhang, and Jingren Zhou. 2026. Fun-Audio-Chat technical report. Preprint, arXiv:2512.20156.

George Yakushev, Nataliia Babina, Masoud Vahid Dastgerdi, Vyacheslav Zhdanovskiy, Denis Kuznedelev, Alina Shutova, and Max Ryabinin. 2025. Asynchronous reasoning: Training-free interactive thinking LLMs. Preprint, arXiv:2512.10931.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Jiawei Zhou, Jason Eisner, Michael Newman, Emmanouil Antonios Platanios, and Sam Thomson. 2022. Online semantic parsing for latency reduction in taskoriented dialogue. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1554–1576, Dublin, Ireland. Association for Computational Linguistics.

Wenhao Zou, Yuwei Miao, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, and Jingwen Xu. 2026. LTS-VoiceAgent: A listen-think-speak framework for eficient streaming voice interaction via semantic triggering and incremental reasoning. arXiv preprint arXiv:2601.19952.

## A Detailed Framework

## A.1 Task and framework overview

Let $x _ { [ 0 , t ] }$ denote speech observed through time �, � the dialogue history including tool calls and results from previous turns, and $\mathcal { T }$ the available tool schemas. The current utterance ends at �. The task is to answer the complete request and execute its required tool calls using �, $\mathcal { T } .$ , and � <sub>[0,�]</sub> . Streaming automatic speech recognition (ASR) provides partial transcripts during listening and a final transcript of the complete utterance.

Heartbeat separates preparation and response generation into two language-model roles (Figure 2). The Thinker is invoked periodically during listening to produce provisional reasoning and tool-call requests from partial transcripts. At the detected end of speech, speculative preparation stops and a separate Responder is triggered. It uses the final transcript and the preparation already available to complete the remaining reasoning, tool calls, and final answer.

Both roles receive � and T and may use diferent models. A shared state stores current-turn preparation: completed Thinker outputs, their tool-call requests, and the associated results as they return. Tools execute asynchronously, independently of the model invocations.

## A.2 Periodic asynchronous Thinker

Heartbeats occur at a fixed interval Δ during listening, with at most � Thinker invocations running concurrently. A trigger is skipped rather than queued when � invocations are already active. Otherwise, a new invocation starts without interrupting earlier ones.

An invocation started at $t _ { k }$ receives �, T, the latest partial transcript, completed outputs from earlier Thinker invocations in the current turn, and results from their tool calls available at $t _ { k }$ . New speech and newly returned tool results enter subsequent invocations rather than being injected into ongoing generation.

The Thinker can produce three forms of output: no additional action, reasoning only, or reasoning with tool requests. These are encoded as noop, think without tool calls, and think with tool calls, respectively. A noop indicates that the Thinker identifies no need for additional reasoning or tool requests given the current input and available preparation. Tool requests specify tool names and structured arguments once the necessary information is available. The prompt and output schema are given in Appendix G.1.

Once a think output is complete, its reasoning record and any tool-call requests are written to the shared state, after which the requested tools are dispatched asynchronously. Since one invocation may request multiple calls, tool concurrency is controlled separately from Thinker concurrency: dispatch imposes no concurrency limit, while the execution backend queues calls for execution. A recorded call is pending until its result returns; the result is then linked to the request by its unique tool-call ID and stored alongside it.

These records allow subsequent Thinker invocations to distinguish work already requested from results available for further reasoning. To avoid redundant calls, the prompt instructs each invocation not to repeat pending or completed calls with the same tool name and complete arguments. Returned results can also supply new information for dependent calls, allowing preparation to advance even when the partial transcript is unchanged.

## A.3 Endpoint-triggered Responder

The voice activity detection (VAD) module triggers the transition to response generation at the detected end ofspeech. At this point, the scheduler stops launching new Thinker invocations and cancels unfinished Thinker and speculative tool tasks, including queued calls. The Responder receives the final transcript, the shared history �, the tool schemas T, and the preparation available at this transition.

Let � denote the VAD-triggered context-assembly time and $a _ { j }$ the availability time of preparation output $r _ { j }$ . The preparation snapshot is

$$
{ \mathcal { R } } _ { c } = \{ r _ { j } : r _ { j } { \mathrm { ~ i s ~ c o m p l e t e ~ a n d ~ } } a _ { j } \leq c \} .\tag{1}
$$

The snapshot contains completed Thinker records and returned tool results. A completed record may include tool-call requests whose results are not yet available. Outputs returned by speculative jobs after � are discarded. Appendix E.1 illustrates the boundary cases.

The Responder follows a ReAct process (Yao et al., 2023). It checks the preparation against the complete request and disregards interpretations contradicted by later speech. Its prompt directs it to reuse available results from completed calls that match the required operations in tool name and complete arguments. Required calls without available results, including those pending at the handof, are issued by the Responder as part of its own ReAct process. Its system prompt and instructions for reading the Heartbeat context appear in Appendices G.2 and G.3. The answer can then be passed to text-to-speech, and the dialogue and retained tool-call records and results are carried forward to subsequent turns.

## B Benchmark Setup

For the main task-quality comparisons, the DeepSeek-V4- Flash Responder uses the 0731 version. The cascaded pipelines combine Qwen3-ASR with a text reasoning model. All experiments use NVIDIA H200 GPUs.

Coverage and accuracy aggregation. For SpeechFCEval and FDBench, STC measures the fraction of reference calls covered by completed speculative calls whose results enter the Responder context without repetition by the Responder. Reference matching follows benchmark-specific tool-name and argument rules; repetition requires identical tool names and complete arguments. SpeechFCEval averages per-sample coverage, whereas FDBench aggregates over reference calls and checks repetitions across each recording. AudioCRAG STC measures the fraction of samples reusing completed speculative results without additional retrieval. SpeechFCEval task accuracy requires correct function names and arguments and is averaged equally over its five task groups.

## B.1 SpeechFCEval

SpeechFCEval combines speech-based tasks from ACEBench and BFCL with SmartInteract (Tongyi Fun Team et al., 2026). It covers single and parallel function calling across synthesized and human speech. We evaluate the full test set.

The test set contains 1,914 samples: 208 and 88 in Speech-ACEBench single and parallel calling, 578 and 372 in Speech-BFCL single and parallel calling, and 668 in SmartInteract. All samples remain in the aggregate, with runtime failures contributing zero. Overall accuracy is averaged equally across the five task groups, while Overall STC is averaged across individual samples.

The model comparisons use a 500 ms heartbeat interval and at most two concurrent Thinker invocations. Thinking is disabled for the two Qwen Responders.

## B.2 FDBench v3

FDBench v3 tests whether voice agents follow spoken intent under disfluency (Lin et al., 2026). It evaluates turn taking, tool selection, argument accuracy, response quality, and task success. Spoken responses are transcribed for evaluation, and task performance is assessed from the executed calls and resulting answers.

FDBench v3 contains 100 recordings covering 79 scenarios and 12 speakers. Spoken responses are transcribed with NVIDIA Parakeet TDT 0.6B v2.

We use DeepSeek-V4-Flash with maximum reasoning efort as the judge for FDBench v3. Quality follows the benchmark author’s answer-quality assessment. For Heartbeat, Pass@1 requires completed speculative results and Responder calls jointly to fulfill the task without incorrect or extra Responder calls.

The cascaded baselines use the same judge for argument accuracy, response quality, and Pass@1. Turn taking and tool selection follow the benchmark’s original evaluation.

Latency configuration. The latency evaluation uses Qwen3.6-35B-A3B as the Thinker and DeepSeek-V4-Flash as the Responder, with thinking disabled for both models. Table 4 lists the speech components and timing settings.

Latency protocol. Latencies are measured relative to each input turn’s aligned speech endpoint, excluding trailing silence.

Final Response Latency measures the onset of the final spoken reply after the last tool call, or the onset of the reply when no tool call is required. Correct Tool Call uses the recorded initiation time of the first eligible call matching the turn’s reference. Matching uses exact tool names with the benchmark’s string normalization and dynamic-argument handling. Speculative calls count only after completion and inclusion in the Responder context.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>ASR</td><td>Qwen3-ASR-1.7B</td></tr><tr><td>TTS</td><td>Qwen3-TTS-12Hz-1.7B-Base</td></tr><tr><td>ASR update block</td><td>500 ms</td></tr><tr><td>VAD minimum speech</td><td>200 ms</td></tr><tr><td>VAD minimum silence</td><td>800 ms</td></tr><tr><td>VAD padding</td><td>500 ms</td></tr><tr><td>Heartbeat interval</td><td>500 ms</td></tr><tr><td>Max. heartbeat</td><td></td></tr><tr><td>concurrency</td><td>2</td></tr></table>

Table 4: Speech pipeline and heartbeat scheduling settings for the FDBench v3 latency evaluation.

<table><tr><td>System</td><td>Correct Tool Call</td><td>Final Response</td></tr><tr><td>Heartbeat</td><td>-4.37</td><td>4.39</td></tr><tr><td>Window Thinking (SHANKS)</td><td>-2.74</td><td>6.90</td></tr><tr><td>Speech-to-Speech</td><td>2.72</td><td>8.39</td></tr><tr><td>Pipecat</td><td>2.45</td><td>8.08</td></tr><tr><td>LiveKit</td><td>3.52</td><td>8.52</td></tr></table>

Table 5: Mean event latencies on FDBench v3 (seconds), measured relative to speech end. Negative values indicate events occurring before speech end.

Latency aggregation. We first average event ofsets within each recording, then give recordings equal weight. Negative ofsets are retained. Correct Tool Call Latency is conditional on a matching call. Table 5 reports the system-level means.

## B.3 AudioCRAG

AudioCRAG evaluates knowledge-grounded spoken answering (Arora et al., 2026).

The AudioCRAG evaluation set contains 1,312 samples. Heartbeat uses a 500 ms interval and at most two concurrent Thinker invocations.

## C Complete Results

## C.1 Task Results

SpeechFCEval. Heartbeat achieves 43.69% STC, compared with 33.13% for the SHANKS-style Window Thinking baseline, an improvement of 10.56 percentage points. Across the five task groups reported in Table 6, coverage increases in four, with Speech-BFCL parallel calling improving from 36.34% to 58.03%. SmartInteract is the exception, where coverage decreases slightly from 40.57% to 38.32%. Despite achieving higher coverage, Heartbeat reaches 66.85% overall function-calling accuracy, slightly below Window Thinking at 67.95%, although both outperform the cascaded baselines. This distinction indicates that retaining more speculative tool results does not necessarily improve function-calling accuracy.

FDBench v3. Heartbeat achieves 50.67% STC, outperforming Window Thinking’s 34.67% by 16.00 percentage points. Unlike SpeechFCEval, its task-quality scores also improve: Table 7 records an increase in response quality from 72.0% to 77.0% and in Pass@1 from 57.0% to 65.0%. These results show that Heartbeat retains more speculative tool work while achieving better task success than the windowed baseline under spoken disfluency. However, its Pass@1 remains below those of Speech-to-Speech and LiveKit.

AudioCRAG. On AudioCRAG, Heartbeat reaches 62.42% STC but achieves a correct-answer rate of 55.49%. The cascaded baselines listed in Table 8 achieve correct-answer rates between 56.48% and 59.45%. Window Thinking achieves 20.05% STC and a 57.24% correct-answer rate. Heartbeat’s missing-answer rate increases from Window Thinking’s 14.33% to 16.84%. This contrast further illustrates that higher speculative retrieval reuse does not guarantee better answer quality. The benefit of early retrieval depends not only on retaining completed results but also on whether the Responder can use them correctly.

## C.2 Model Comparisons

We vary the Thinker and Responder separately to test preparation and reuse across models.

Thinker choice. All four Thinkers achieve nonzero STC on all three benchmarks (Tables 6–8). Higher coverage does not consistently yield better task performance: on SpeechFCEval, Gemma achieves higher accuracy than the default Thinker (68.58% vs. 66.85%) despite slightly lower STC (42.71% vs. 43.69%).

Responder choice. Heartbeat retains higher STC than Window Thinking with all three Responders on all three benchmarks, including FDBench (Table 3). However, model rankings difer by metric: on SpeechFCEval, Qwen3.8- Flash-Next achieves higher STC than Qwen3.5-397B-A17B (55.27% vs. 41.20%) but lower accuracy (64.30% vs. 69.70%).

## D Speculative Tool-Use Analysis Details

Grouping and success criteria. We group recordings by when the target entity and all constraints of the final request have been fully spoken. Early information arrives within the first 70% of the efective input interval; late information arrives within the final 30%. Efective input duration excludes leading and trailing silence. Explicit intent revisions form a separate group.

(a) Function-calling accuracy (Acc.; ↑) across the five task groups.
<table><tr><td rowspan="2">Framework</td><td rowspan="2">Thinker</td><td colspan="2">Speech-ACEBench</td><td colspan="2">Speech-BFCL</td><td rowspan="2">SmartInteract Single</td><td rowspan="2">Overall</td></tr><tr><td>Single</td><td>Parallel</td><td>Single</td><td>Parallel</td></tr><tr><td>Cascaded</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Speech-to-Speech</td><td></td><td>49.00</td><td>25.00</td><td>91.00</td><td>60.48</td><td>67.81</td><td>58.66</td></tr><tr><td>Pipecat</td><td></td><td>55.80</td><td>20.50</td><td>89.79</td><td>56.99</td><td>61.08</td><td>56.83</td></tr><tr><td>LiveKit</td><td>一</td><td>44.20</td><td>13.60</td><td>89.27</td><td>58.06</td><td>59.13</td><td>52.85</td></tr><tr><td>Window Thinking (SHANKS)</td><td>Qwen3.6-35B-A3B</td><td>63.94</td><td>44.32</td><td>86.16</td><td>71.51</td><td>73.80</td><td>67.95</td></tr><tr><td rowspan="4">Heartbeat</td><td>Qwen3.6-35B-A3B</td><td>57.21</td><td>37.50</td><td>86.68</td><td>76.08</td><td>76.80</td><td>66.85</td></tr><tr><td>Qwen3.5-9B</td><td>46.63</td><td>21.59</td><td>82.87</td><td>72.58</td><td>71.56</td><td>59.05</td></tr><tr><td>GLM-4.7-Flash</td><td>45.19</td><td>22.73</td><td>83.91</td><td>71.77</td><td>73.65</td><td>59.45</td></tr><tr><td>Gemma-4-26B-A4B-it</td><td>62.50</td><td>37.50</td><td>87.20</td><td>76.08</td><td>79.64</td><td>68.58</td></tr></table>

(b) Speculative tool coverage (STC; ↑) across the five task groups.

<table><tr><td rowspan="2">Framework Thinker</td><td rowspan="2"></td><td colspan="2">Speech-ACEBench</td><td colspan="2">Speech-BFCL</td><td rowspan="2">SmartInteract Single</td><td rowspan="2">Overall</td></tr><tr><td>Single</td><td>Parallel</td><td>Single</td><td>Parallel</td></tr><tr><td>Cascaded</td><td></td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>Qwen3.6-35B-A3B</td><td>28.37</td><td>0.00</td><td>29.24</td><td>36.34</td><td>40.57</td><td>33.13</td></tr><tr><td rowspan="4">Heartbeat</td><td>Qwen3.6-35B-A3B</td><td>42.31</td><td>36.78</td><td>42.21</td><td>58.03</td><td>38.32</td><td>43.69</td></tr><tr><td>Qwen3.5-9B</td><td>28.85</td><td>23.66</td><td>35.81</td><td>53.51</td><td>25.00</td><td>34.16</td></tr><tr><td>GLM-4.7-Flash</td><td>35.10</td><td>20.87</td><td>39.97</td><td>49.67</td><td>35.93</td><td>39.04</td></tr><tr><td>Gemma-4-26B-A4B-it</td><td>49.04</td><td>38.98</td><td>42.73</td><td>58.91</td><td>32.19</td><td>42.71</td></tr></table>

Table 6: SpeechFCEval results (%) with DeepSeek-V4-Flash as the Responder. (a) Function-calling accuracy and (b) speculative tool coverage (STC).

<table><tr><td>Framework</td><td>Thinker</td><td>Quality</td><td>Pass@1</td><td>STC</td></tr><tr><td>Cascaded</td><td></td><td></td><td></td><td></td></tr><tr><td>Speech-to-Speech</td><td>1</td><td>84.0</td><td>75.0</td><td>0.0</td></tr><tr><td>Pipecat</td><td>一</td><td>78.0</td><td>63.0</td><td>0.0</td></tr><tr><td>LiveKit</td><td>一</td><td>77.6</td><td>67.0</td><td>0.0</td></tr><tr><td>Window Thinking (SHANKS)</td><td>Qwen3.6-35B-A3B</td><td>72.0</td><td>57.0</td><td>34.67</td></tr><tr><td rowspan="4">Heartbeat</td><td>Qwen3.6-35B-A3B</td><td>77.0</td><td>65.0</td><td>50.67</td></tr><tr><td>Qwen3.5-9B</td><td>80.0</td><td>57.0</td><td>39.33</td></tr><tr><td>GLM-4.7-Flash</td><td>84.0</td><td>55.0</td><td>43.33</td></tr><tr><td>Gemma-4-26B-A4B-it</td><td>80.0</td><td>67.0</td><td>52.00</td></tr></table>

Table 7: FDBench v3 task performance and speculative tool coverage (STC; %) with DeepSeek-V4-Flash as the Responder.

For the early- and late-information groups, we measure the fraction of recordings in which a call with required arguments consistent with the final request is initiated before speech end. For the revision group, correction succeeds when an earlier call inconsistent with the final request is followed by a corrected call before speech end. These initiation-based measures difer from STC, which requires completed results to be included in the Responder context without response-stage repetition.

Inconsistent result reuse. In these failures, a correctargument tool request coexists with a final answer that uses erroneous or outdated speculative results.

For example, the system adds a \$99.99 cofee maker to the cart before hearing the complete \$50 budget, then retrieves a \$40 product. The answer claims that the cheaper product was added but reports the earlier \$99.99 cart total, mixing the two results. The corrected search does not undo the cart action.

## D.1 Timing and Case Studies

We examine how information timing and explicit intent revisions afect Heartbeat’s speculative tool execution on FDBench v3 as spoken requests evolve.

<table><tr><td>Framework</td><td>Thinker</td><td>Correct ↑</td><td>Missing</td><td>Hallucination ↓</td><td>STC ↑</td></tr><tr><td>Cascaded</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Speech-to-Speech</td><td></td><td>56.48</td><td>11.97</td><td>31.55</td><td>0.00</td></tr><tr><td>Pipecat</td><td></td><td>56.94</td><td>11.97</td><td>31.10</td><td>0.00</td></tr><tr><td>LiveKit</td><td>一</td><td>59.45</td><td>12.88</td><td>27.67</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>Qwen3.6-35B-A3B</td><td>57.24</td><td>14.33</td><td>28.43</td><td>20.05</td></tr><tr><td rowspan="4">Heartbeat</td><td>Qwen3.6-35B-A3B</td><td>55.49</td><td>16.84</td><td>27.67</td><td>62.42</td></tr><tr><td>Qwen3.5-9B</td><td>54.27</td><td>18.29</td><td>27.44</td><td>60.06</td></tr><tr><td>GLM-4.7-Flash</td><td>52.29</td><td>19.36</td><td>28.35</td><td>63.57</td></tr><tr><td>Gemma-4-26B-A4B-it</td><td>55.87</td><td>16.62</td><td>27.52</td><td>74.70</td></tr></table>

Table 8: AudioCRAG answer outcomes and retrieval reuse (%) with DeepSeek-V4-Flash as the Responder. Here, STC measures the share of samples with completed speculative results included in the Responder’s context and no additional retrieval by the Responder.

<table><tr><td>Framework</td><td>Correct ↑</td><td>Miss.</td><td>Hallu. ↓</td><td>STC ↑</td></tr><tr><td colspan="5">Responder: DeepSeek-V4-Flash</td></tr><tr><td>Speech-to-Speech</td><td>56.48</td><td>11.97</td><td>31.55</td><td>0.00</td></tr><tr><td>Pipecat</td><td>56.94</td><td>11.97</td><td>31.10</td><td>0.00</td></tr><tr><td>LiveKit</td><td>59.45</td><td>12.88</td><td>27.67</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>57.24</td><td>14.33</td><td>28.43</td><td>20.05</td></tr><tr><td>Heartbeat</td><td>55.49</td><td>16.84</td><td>27.67</td><td>62.42</td></tr><tr><td colspan="5">Responder: Qwen3.5-397B-A17B</td></tr><tr><td>Speech-to-Speech</td><td>52.29</td><td>8.00</td><td>39.71</td><td>0.00</td></tr><tr><td>Pipecat</td><td>55.56</td><td>6.25</td><td>38.19</td><td>0.00</td></tr><tr><td>LiveKit</td><td>57.16</td><td>7.32</td><td>35.52</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>55.79</td><td>9.76</td><td>34.45</td><td>20.43</td></tr><tr><td>Heartbeat</td><td>55.95</td><td>8.16</td><td>35.90</td><td>66.46</td></tr><tr><td colspan="5">Responder: Qwen3.8-Flash-Next</td></tr><tr><td>Speech-to-Speech</td><td>56.86</td><td>11.20</td><td>31.94</td><td>0.00</td></tr><tr><td>Pipecat</td><td>54.27</td><td>11.97</td><td>33.77</td><td>0.00</td></tr><tr><td>LiveKit</td><td>58.00</td><td>10.59</td><td>31.40</td><td>0.00</td></tr><tr><td>Window Thinking (SHANKS)</td><td>53.35</td><td>16.01</td><td>30.64</td><td>11.05</td></tr><tr><td>Heartbeat</td><td>56.17</td><td>15.17</td><td>28.66</td><td>40.70</td></tr></table>

Table 9: AudioCRAG answer outcomes and retrieval reuse across Responder models (%), with Qwen3.6-35B-A3B as the Thinker. The highest reuse rate and correct-answer rate occur with diferent Responders.

Information Timing and Intent Revisions Across the FDBench groups in Table 10, Heartbeat initiates calls consistent with the final request before speech ends in 73.3% of early-information recordings and 49.2% when information arrives late. For explicit revisions, 70.6% contain a corrected call before speech end. Earlier information provides more opportunities for speculative execution, while late constraints and intent revisions can still be incorporated during listening.

<table><tr><td>Input group</td><td>Success rate</td></tr><tr><td>Early information</td><td>73.3%</td></tr><tr><td>Late information</td><td>49.2%</td></tr><tr><td>Explicit revision</td><td>70.6%</td></tr></table>

Table 10: Correct tool-call initiation before speech end on FD-Bench v3, grouped by information timing and explicit intent revision. For revision cases, the rate measures corrected calls following an outdated request. Group definitions are given in Appendix D.

Case Studies and Failure Modes Figure 5 illustrates early order tracking, a late budget constraint, and a search revised from running shoes to hiking boots. Corrected calls do not guarantee consistent use of speculative results: in 4.0% of recordings, the final answer uses erroneous or outdated results despite a correctly specified request. The cart-update case in Appendix D shows that a corrected search cannot undo an earlier state-changing action. These cases distinguish timely speculative execution from correct use of its results.

## E Scheduling and Coverage Details

## E.1 Overlapping jobs and the response boundary

A tool call dispatched during listening contributes its result to the initial Responder context when the result is complete and available at context assembly � (Equation 1). At the detected end of speech, the scheduler stops launching new Thinker invocations and cancels unfinished Thinker and speculative tool tasks, including queued calls. Later results are excluded from the current response.

At context assembly, a completed Thinker record may be available even if its requested tool result is unavailable. Overlapping invocations can finish out of order.

## E.2 Interpreting speculative tool coverage

Each reference call contributes at most once to a request’s coverage fraction. The denominator is the number of reference calls; repeated speculative attempts cannot contribute multiple matches to the same reference call. Matching a reference and detecting a Responder repetition are separate tests. The first establishes relevance under the benchmark’s matching rule, while the second removes retained coverage when the Responder makes a call with the same tool name and exactly the same arguments.

![](images/037bb2442a339717f4695b75b539e42febcd463511153c1bc90e87645b3c9d06.jpg)  
Figure 5: Speculative tool calls during listening on FDBench v3. Red markers indicate the last words available to the Thinker when each invocation began; callouts give tool-request initiation times and arguments. Times are relative to audio onset. Word alignment is approximate, and ASR text is normalized for display.

Suppose a request requires two calls. Both speculative results are completed and injected, but the Responder repeats one with identical arguments. The request’s non-repeated coverage is 1/2. If one result instead arrives after context assembly and the other is not repeated, the coverage is also 1/2, but the loss now occurs at the preparation boundary rather than through response-time repetition.

## F Extended Related Work

Incremental reasoning and response preparation. Preparation from partial input can target intermediate reasoning, candidate responses, or both. LiveMind stores prefix-based inferences in memory for subsequent reasoning and final response generation (Chen et al., 2024). PredGen instead drafts candidate responses, verifies reusable prefixes as input grows, and prepares audio for the first sentence (Li and Grover, 2025). LTS-VoiceAgent combines these forms of preparation: its Thinker maintains corrected transcripts, key variables, and plans, while its Speaker uses this state to begin solving and generating responses before speech ends (Zou et al., 2026). The prepared content therefore ranges from reasoning context for later generation to response text and audio that can be reused directly.

Scheduling speculative computation under streaming input. Early tool execution was studied in online semantic parsing through prefix-to-program prediction and selective subprogram execution (Zhou et al., 2022). Subsequent systems difer in how computation proceeds as input changes. SHANKS interleaves speech chunks with reasoning and tool calls, but its per-chunk generation budget can leave long call arguments incomplete (Chiang et al., 2026, Appendix C.2). Stream RAG contrasts fixed-interval parallel retrieval with model-triggered queries that terminate unfinished calls for the previous query (Arora et al., 2026). In Speculative Interaction Agents (SIA), incoming user or tool updates interrupt ongoing generation, potentially before a complete tool call is produced, while dispatched tools are managed separately (Hooper et al., 2026). Outside speech, PASTE predicts upcoming calls from recurring execution patterns and starts them while the LLM is still generating (Sui et al., 2026). Heartbeat allows successive Thinker invocations to overlap within a concurrency limit. This supports denser triggers without making the next input update a deadline for ongoing generation.

Decoupled architectures and coordination interfaces. Decoupled systems use diferent interfaces to pass intermediate work between components. The SHANKS + call-afterlisten configuration passes successful calls and their results to a model that processes the complete request (Chiang et al., 2026). DuplexOmni separates real-time interaction from background reasoning and tool use, returning intermediate outputs through a streaming interface (Huang et al., 2026). Other interfaces operate within the model’s generation process: AsyncLM encodes tool calls and returned results in the token stream (Gim et al., 2024), whereas AsyncReasoning coordinates thinker and writer views through shared modelinternal attention state (Yakushev et al., 2025). Heartbeat connects independent Thinker invocations, asynchronous tool jobs, and an endpoint-triggered Responder through completed reasoning records and tool results. This external interface permits independent model selection and modulelevel optimization without requiring shared model-internal state.

## G Modular Prompt Assembly

The following blocks show the shared Thinker and Responder instructions and their message roles. Benchmark-specific instructions, tool schemas, dialogue history, and current-turn state are represented by placeholders.

## G.1 Thinker prompt

The Thinker system message declares its role; the user message assembles the fixed preparation rules and variable modules.

SYSTEM   
You are an incremental HeartBeat Thinking worker   
for a general real-time voice assistant.   
USER   
[Task]   
Work speculatively from the current cumulative   
partial ASR transcript while the user is   
speaking. Preserve useful incremental   
interpretations and propose supported tool   
calls. You do not answer the user directly.   
Response will independently interpret the complete   
final input.   
[Tools]   
<Benchmark tool schemas>   
[Decision policy]   
- Return noop when there is neither a useful new   
interpretation nor a newly actionable tool   
call. Repeated speech may still permit a new   
call when a completed result has supplied a   
missing dependency.   
- Otherwise return think, with one concise   
incremental operational note and any eligible   
calls. A new stable intent with missing   
arguments can produce thinking with an empty

tools array. A note that merely says to wait   
adds no useful interpretation.   
- Do not propose a call identical to an existing   
pending or status-completed HB call. This does   
not depend on result content or answer   
sufficiency. A call with different arguments   
remains eligible when supported by the current   
request, not merely to bypass this rule.   
[Argument rules]   
- Copy the exact supplied tool name and schema  
defined argument keys. Satisfy required fields   
, JSON types, enums, and stated preconditions;   
do not invent tools, fields, or capabilities.   
- Support values with the current transcript,   
unambiguous conversation history, or relevant   
usable completed results. Do not invent   
missing facts or fill required fields with   
placeholders.   
[Output format (STRICT)]   
Return one JSON object.   
{"action":"noop"}   
or   
{"action":"think","thinking":"...","tools":[{"name   
":"...","arguments":{...}}]}   
Use an empty array when there is no tool call.   
No markdown fences, no explanation, no extra keys.   
[Extra Instructions]   
<HB benchmark instructions, if supplied>   
[Conversation History]   
<Dialogue history>   
[Current Turn State]   
<Thinking entries and tool states>   
<Optional current-turn metadata>   
[Current Fragment]   
<Cumulative partial transcript and time range>

## G.2 Responder prompt

The Responder system message combines shared continuation rules, tool schemas, and task-specific instructions. Tool schemas are also passed through the structured tool interface.

SYSTEM   
You are continuing work in an existing assistant   
turn, not starting the user’s request over.   
HeartbeatContext records work already   
performed for this turn. Your responsibility   
is to finish the remaining work and give the   
user the appropriate response.   
### Response Core Rules   
You are the main response model in a real-time   
voice assistant. Interpret the complete final

user input using the conversation, supplied   
tool schemas, task instructions, and current  
turn HeartbeatContext. The final input takes   
precedence over an earlier speculative   
interpretation, including when it corrects or   
cancels it.   
HeartbeatContext is a history, not an action list.   
Its thinking is speculative: retain useful   
interpretations, but do not treat intentions   
as commands, proof of execution, or evidence   
for missing arguments. An earlier incorrect   
call does not invalidate another completed   
call that matches the final request.   
Pending, accumulated, and running work is not   
completion evidence and is not a reason to   
omit a necessary Response call. Use only   
result information present in the provided   
context.   
If all required operations are already covered,   
respond using the available evidence and the   
task-specific response policy without more   
calls. Do not expose internal HeartBeat   
bookkeeping. Do not execute unnecessary   
operations solely because they appear in   
earlier thinking.   
<Tool schemas>   
<Response benchmark instructions and answer format,   
if supplied>   
PREVIOUS MESSAGES   
<Dialogue history and tool results>

## G.3 Heartbeat context module

In the illustrated SpeechFCEval configuration, the current user message contains the preparation snapshot, continuation instructions, and complete final transcript. The temporary context is excluded from persistent dialogue history after the response.

CURRENT USER   
### HeartbeatContext   
<Actual injected thinking and completed tool   
records: name, arguments, status, result>   
### ResponseConstraint   
Continue from the completed work above. Request   
only new work: never restate a completed exact   
-name, complete-arguments call in tool\_calls.   
Missing result details do not make that   
execution pending. If no new operation is   
needed, produce the task-appropriate response   
without tool calls. Use the final input to   
correct genuinely different arguments, not to

restart matching completed work.   
### FinalUserInput   
<Complete final transcript>

## H Use of AI Assistants

AI assistants were used for manuscript drafting and revision and for assistance with experimental code. The authors are responsible for the final manuscript, code, analyses, and claims.