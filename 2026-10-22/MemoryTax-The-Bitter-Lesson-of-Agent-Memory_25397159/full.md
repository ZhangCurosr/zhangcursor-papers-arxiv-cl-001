# MemoryTax: The Bitter Lesson of Agent Memory

Zedong Peng Shanghai Jiao Tong University zedongpeng1@gmail.com

Interactive version: zedongpeng.com/memorytax

## Abstract

Agent memory systems such as Mem0 and Zep decide what to keep before any question arrives; coding agents keep the raw history and let the model search it with its own tools. We ask whether designed memory still helps once the model is strong enough to search for itself. On one LoCoMo conversation (419 turns, 152 questions), we run four memory methods (full context, BM25, Mem0, Zep) and three coding agents (Pi, Codex, Claude Code), each with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5, and record what every rollout costs. This is a short empirical study, with one conversation and one run per cell. Agent memory is a costly patch for a weak model: Mem0 beats full context only with GPT-4o-mini, and its build alone costs more than a whole full-context rollout. Agent memory also has a ceiling, set by the model that writes it: a stronger reader does not recover what a weaker writer left out. With Claude Opus 5.5, Pi tops the cost–accuracy frontier, scoring 99% for less than half the cost of full context; its one miss is a wrong answer key. Part of that lead comes from photo captions that only the agents could read: on the 88 questions whose evidence has no photo, Pi and full context difer by one answer.

## 1 Introduction

Agent memory systems such as Mem0 and Zep decide what to keep before any question arrives: the model extracts facts or builds a graph from the conversation, and each question sees a few retrieved pieces [1, 2]. Coding agents and recursive language models keep the raw history instead and let the model search it with its own tools [3–6].

The bitter lesson is that knowledge built into a system helps at first, then plateaus, while general methods that scale with computation win [7]. Agent memory builds in what to keep. Does it still help once the model is strong enough to search for itself?

We run four memory methods (full context, BM25 [8], Mem0 and Zep) and three coding agents (Pi, Codex and Claude Code), each with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5, on a conversation from LoCoMo [9], and measure what each run costs. We find:

• Agent memory is a costly patch for a weak model. Mem0 beats full context only with GPT-4o-mini, and its build alone costs more than full context’s whole run (Section 3).

• Agent memory has a ceiling, set by the model that writes it. A stronger reader does not recover what a weaker writer left out (Section 4).

• With Claude Opus 5.5, Pi tops the cost–accuracy frontier (Figure 1). It answers every question whose answer key is sound, for less than half what full context costs. Where the two read the same text they difer by one answer; the rest of Pi’s lead comes from photo captions that only the agents could read (Section 5).

The study is small: one conversation, one run per cell, and a GPT-4o-mini judge. Section 6 lists what it cannot show. Within those limits, the results suggest that a strong model needs less designed memory and better tools for searching the raw record.

![](images/ce0d00564f0fa7db28c0ff84d55a7977d69650a779b078e47b38ed7d9cfd4d16.jpg)  
Figure 1: Cost and correct rate per rollout. Each badge is one cell’s whole run: the build, then all 152 questions. Colour is the model, the badge the method’s mark, and a letter in the badge marks a method without one (F full context, B BM25). The stepped line is the highest correct rate reached at or below each cost, and the labelled cells lie on it. Cost is on a log axis.

Table 1: Methods: what each stores and what the model sees when it answers.
<table><tr><td>Method</td><td>Stored</td><td>Model sees</td></tr><tr><td>Full context</td><td>raw history</td><td>all of it</td></tr><tr><td>BM25</td><td>raw turns</td><td>top-8 turns, ±2</td></tr><tr><td>Mem0</td><td>extracted facts</td><td>top-200 facts</td></tr><tr><td>Zep</td><td>temporal graph</td><td>top-20 edges</td></tr><tr><td>Pi, Codex, Claude Code</td><td>raw history, one JSON file</td><td>what its tools read</td></tr></table>

## 2 Setup

We run 21 cells, seven methods with three models each, on LoCoMo conv-26 [9]: 419 turns in 19 sessions and 152 single-hop, multi-hop, temporal and open-domain questions. LoCoMo’s adversarial questions are left out, as in Mem0’s evaluation [1]. The history fits every model’s context window. Among the ten LoCoMo conversations, conv-26’s per-method accuracies track those of the full set closely. Table 1 lists the methods.

The memory methods answer one question at a time with Mem0’s 2025 answer template; Mem0 is OSS 2.1.0 and Zep is Graphiti OSS 0.30.2, both with default extraction and retrieval [1, 2]. The agents, Pi 1.1.0, Codex 0.159.2 and Claude Code 2.1.295 [3–5], start a fresh session in a fresh container for each question, with the conversation as one file, their default tools, and the template’s answer rules in the prompt (Appendix C).

The models go from weak to strongest: GPT-4o-mini, GPT-6 Luna, Claude Opus 5.5. The last two run at reasoning efort high, the efort coding agents are usually run at [10]; GPT-4o-mini has no efort setting. In each cell one model writes the store (Mem0, Zep) and answers every question, or drives the agent.

Photos. Some LoCoMo turns share a photo, and the oficial file holds a caption of it next to the turn text. The agents get the oficial file, captions included; the memory methods were given the turn text only. 116 of the 419 turns share a photo, and 64 of the 152 questions have evidence in such a turn. Section 5 measures what this changes.

Measurement. A rollout is one run over the conversation: the build, then the 152 questions. GPT-4o-mini grades every answer with Mem0’s 2025 judge rubric [1]; empty answers count as wrong. Cost is the list price of every model call, LLM and embedding (text-embedding-3-small), with the judge excluded. A turn is one LLM call: one per question for the memory methods, one per step of an agent’s loop, and Mem0’s and Zep’s calls to build their store. Tokens are those calls’ input and output tokens, cached input included; embeddings are not counted. Each cell runs once.

## 3 Agent memory is a costly patch for a weak model

With GPT-4o-mini, Mem0 scores highest. It scores 78%, against 74% for both full context and BM25; Zep scores 43%. The agents score 63% (Pi), 28% (Codex) and 29% (Claude Code): Codex opens the conversation file for three questions in four and Claude Code almost never, and otherwise they answer from the question alone.

With the two stronger models, the patch stops helping. With GPT-6 Luna, full context scores 92%, BM25 82%, Mem0 80% and Zep 51%; Zep’s graph holds 128 facts from the 19 sessions, against 195 with Claude Opus 5.5. With Claude Opus 5.5, Mem0 and full context both score 95% and Zep 87%, and the agents score highest: Pi 99%, Codex 98% and Claude Code 97%, a lead that comes mostly from questions about photos (Section 5). Earlier work also finds that plain retrieval keeps up with elaborate memory [11] and that rankings between memory systems change with the model [12].

The patch is not free. Mem0’s build alone costs more than full context’s whole rollout with the same model: \$0.56 against \$0.43 with GPT-4o-mini, \$0.45 against \$0.29 with GPT-6 Luna and \$25.88 against \$17.84 with Claude Opus 5.5 (Figure 2). Its questions are cheaper, so the build would pay for itself after 372, 545 and 596 questions on the same history; LoCoMo asks 152. Mem0 and Zep are never on the frontier of Figure 1, which runs through BM25 and full context with GPT-6 Luna and Pi with Claude Opus 5.5. Paying to build memory that a stronger model does not need is a memory tax.

## 4 Agent memory has a ceiling

The model that writes the store limits every reader. Mem0 and Zep write their store with the same model that later reads it. Retrieval calls no language model, so the reader can be swapped: each model answers from the retrievals of the others’ stores (Figure 3, Table 3).

A stronger reader does not recover what the writer left out. On Zep’s GPT-4o-mini graph, GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5 score 43%, 43% and 45%; on the graph Claude Opus 5.5 wrote, 66%, 71% and 87%. Mem0 keeps the order with smaller gaps: 78%, 83% and 84% from GPT-4o-mini’s store, 89%, 91% and 95% from Claude Opus 5.5’s. This agrees with work where verbatim chunks beat extracted facts on long conversations [13].

## 5 With Claude Opus 5.5, Pi tops the frontier

Pi with Claude Opus 5.5 is the top of the frontier in Figure 1. No other cell scores higher: Pi scores 99% for \$6.95 a rollout, while full context with the same model scores 95% for \$17.84. Pi reads about 10,000 input tokens a question, 7,900 of them cached, against 28,900 for full context. Codex and Claude Code score 98% and 97% for \$25.41 and \$28.41; HarnessTax explains why the same model costs so diferently across harnesses [10].

![](images/016fbc39a7d17e4877ddc16dd61fb46caac6f8bd3dc08fbba0a8543843085ec6.jpg)  
Figure 2: Cumulative cost–correct curves. Each curve is one rollout: the build, then the questions from cheapest to most expensive (equal costs grouped), adding up cost and correct answers; correct answers are divided by 152, so a wrong answer adds cost only. Colour is the model, the badge the method, and agents are dotted. The cost axis is linear to \$2 and logarithmic above. Curves use a nine-point moving average, and the badges mark the unsmoothed totals of Table 2. Strong lines are cells that no other cell beats on both cost and correct answers.

Its one miss is LoCoMo’s mistake. The question asks what kind of painting Caroline shared with Melanie on 13 October 2023, and the answer key says an abstract painting with blue streaks. In the conversation it is Melanie who shares that painting; Caroline shares a drawing. Pi says so and is marked wrong: “Caroline shared a drawing, not a painting. . . . The paintings shared that day (a sunset and an abstract piece) were Melanie’s.” With Claude Opus 5.5, full context, Codex and Claude Code point out the same mistake and are marked wrong too. On the other 151 questions, Pi with Claude Opus 5.5 is always right.

Photo captions account for most of Pi’s lead over full context. With Claude Opus 5.5, Pi answers seven questions that full context misses and misses none that it answers. Six of the seven have their evidence in a turn that shares a photo, whose caption Pi could read and full context could not (Section 2); so do all six such questions for Codex and all four for Claude Code. On the 88 questions without photo evidence, Pi scores 100%, and full context, Mem0, Codex and Claude Code 99% each (Table 4). With the same input, then, Pi matches full context for less than half its cost. Whether searching the record answers more than reading all of it, this study does not settle.

## 6 Discussion and limitations

Our results suggest that agent memory is a costly patch for a weak model and caps what any reader can answer. On this LoCoMo conversation, Mem0 helps GPT-4o-mini but not GPT-6 Luna or Claude Opus 5.5, a stronger reader does not recover what the writer left out, and with Claude Opus 5.5 Pi answers every sound question

![](images/f652adc3c0dce7c0c7ad7d0a15e23aef6da8f6941c765e828fd394985234d718.jpg)  
Figure 3: Writer and reader. Each colour is one store, written by that model. The filled badge is the writer reading it own store, and arrows run from it to the other two readers.

for less than full context costs.

The findings rest on a small study, and these limits apply:

• Scale. One conversation of LoCoMo’s ten, and one run per cell. Diferences of a few questions may not hold in a second run.

• Grading. A GPT-4o-mini judge with a lenient rubric, and at least one wrong answer key.

• Photo captions. The agents could read photo captions that the memory methods were not given (Section 5). This touches comparisons between agents and memory methods, not those among the memory methods.

• Context length. The history fits in every model’s context window, so full context is always an option. Results may difer on other benchmarks such as LongMemEval [14] and on histories that do not fit.

• Settings and prices. Mem0 and Zep run with their default extraction and retrieval, and other settings may do better. Costs are list prices and depend on each provider’s caching.

Memory systems and harnesses can also be searched for automatically [15, 16], but what is found is still fixed before the question arrives. More broadly, a strong model may need less designed memory and better tools: exact, semantic and vector search over the raw record, driven by the model itself [3, 4]. Progress may then look less like another memory system and more like ripgrep after grep [17], or zvec-grep after ripgrep [18]. Designed memory may survive as a cost trade-of when many questions share one history (Figure 2).

## References

[1] P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav. Mem0: Building production-ready AI agents with scalable long-term memory, 2025. URL https://arxiv.org/abs/2504.19413.

[2] P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, and D. Chalef. Zep: A temporal knowledge graph architecture for agent memory, 2025. URL https://arxiv.org/abs/2501.13956.

[3] Pi contributors. Pi coding agent: Compaction and session persistence. Software documentation and source, 2026. URL https://github.com/earendil-works/pi.

[4] OpenAI. Codex: An open-source coding agent, 2026. URL https://github.com/openai/codex.

[5] Anthropic. Claude code, 2026. URL https://github.com/anthropics/claude-code.

[6] A. L. Zhang, T. Kraska, and O. Khattab. Recursive language models, 2025. URL https://arxiv. org/abs/2512.24601.

[7] R. S. Sutton. The bitter lesson. Essay, 2019. URL http://www.incompleteideas.net/IncIdeas/ BitterLesson.html.

[8] S. Robertson and H. Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/1500000019.

[9] A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang. Evaluating very long-term conversational memory of LLM agents, 2024. URL https://arxiv.org/abs/2402.17753.

[10] Arena Team. HarnessTax: How much does the harness matter for coding agents? Arena blog, 2026. URL https://arena.ai/blog/coding-agents-harness-tax.

[11] Y. Wu, W. Chen, Z. Huang, et al. Back to basics: Let conversational agents remember with just retrieval and generation, 2026. URL https://arxiv.org/abs/2604.11628.

[12] K. Wang. MemDelta: Controlled baselines and hidden confounds in agent memory evaluation, 2026. URL https://arxiv.org/abs/2606.29914.

[13] T. An. Fidelity before structure: Verbatim chunks beat lossy artifact extraction in long-conversation LLM memory, 2026. URL https://arxiv.org/abs/2601.00821.

[14] D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu. LongMemEval: Benchmarking chat assistants on long-term interactive memory, 2024. URL https://arxiv.org/abs/2410.10813.

[15] G. Zhang, H. Ren, C. Zhan, et al. MemEvolve: Meta-evolution of agent memory systems, 2025. URL https://arxiv.org/abs/2512.18746.

[16] Y. Lee, R. Nair, Q. Zhang, K. Lee, O. Khattab, and C. Finn. Meta-harness: End-to-end optimization of model harnesses, 2026. URL https://arxiv.org/abs/2603.28052.

[17] A. Gallant. ripgrep. Software, 2016. URL https://github.com/BurntSushi/ripgrep.

[18] Zvec Team. From rg to zg: Local search beyond keywords. Zvec blog, 2026. URL https: //zvec.org/en/blog/2026-08-28-zvec-grep-open-source/.

[19] Harbor Framework Team. Harbor: A framework for evaluating and optimizing agents and models in container environments. Software, 2026. URL https://doi.org/10.5281/zenodo.20953922.

## A Full results

Table 2 gives every cell’s cost and correct rate, Table 3 the writer–reader grid behind Figure 3, and Table 4 the split by photo evidence. Figure 4 redraws Figure 2 against turns and tokens.

Table 2: Per method and model. Cost is the list price of every model call except the judge: Build is paid once, before any question; Per q. is everything one question costs, retrieval included, averaged over the 152 questions. ◦: nothing is built. Efort: the model’s reasoning efort; GPT-4o-mini has none. Correct: share of answers judged correct, in %, per LoCoMo category (single-hop, multi-hop, temporal, open-domain); the first row gives each category’s size.
<table><tr><td rowspan="2"></td><td rowspan="2">Model</td><td rowspan="2">Effort</td><td colspan="2">Cost ($)</td><td colspan="5">Correct (%)</td></tr><tr><td>Build</td><td>Per q.</td><td>Single</td><td>Multi</td><td>Temp.</td><td>Open</td><td>All</td></tr><tr><td>Questions</td><td></td><td></td><td></td><td></td><td>70</td><td>32</td><td>37</td><td>13</td><td>152</td></tr><tr><td rowspan="3">Full context</td><td>GPT-4o-mini</td><td></td><td>0</td><td>0.0028</td><td>89</td><td>75</td><td>51</td><td>62</td><td>74</td></tr><tr><td>GPT-6 Luna</td><td>high</td><td>0</td><td>0.0019</td><td>90</td><td>97</td><td>95</td><td>85</td><td>92</td></tr><tr><td>Claude Opus 5.5</td><td>high</td><td>0</td><td>0.117</td><td>93</td><td>94</td><td>97</td><td>100</td><td>95</td></tr><tr><td rowspan="3">BM25</td><td>GPT-4o-mini</td><td>一</td><td>o</td><td>0.00028</td><td>86</td><td>50</td><td>73</td><td>69</td><td>74</td></tr><tr><td>GPT-6 Luna</td><td>high</td><td>0</td><td>0.00021</td><td>89</td><td>63</td><td>89</td><td>69</td><td>82</td></tr><tr><td>Claude Opus 5.5</td><td>high</td><td>0</td><td>0.013</td><td>93</td><td>81</td><td>92</td><td>92</td><td>90</td></tr><tr><td rowspan="3">Mem0</td><td>GPT-4o-mini</td><td></td><td>0.559</td><td>0.0013</td><td>87</td><td>75</td><td>65</td><td>77</td><td>78</td></tr><tr><td>GPT-6 Luna</td><td>high</td><td>0.449</td><td>0.0011</td><td>89</td><td>88</td><td>57</td><td>85</td><td>80</td></tr><tr><td>Claude Opus 5.5</td><td>high</td><td>25.876</td><td>0.074</td><td>93</td><td>97</td><td>97</td><td>100</td><td>95</td></tr><tr><td rowspan="3">Zep</td><td>GPT-4o-mini</td><td></td><td>0.092</td><td>0.00012</td><td>51</td><td>34</td><td>30</td><td>54</td><td>43</td></tr><tr><td>GPT-6 Luna</td><td>high</td><td>0.103</td><td>0.00012</td><td>54</td><td>50</td><td>49</td><td>46</td><td>51</td></tr><tr><td>Claude Opus 5.5</td><td>high</td><td>6.224</td><td>0.0077</td><td>83</td><td>94</td><td>89</td><td>85</td><td>87</td></tr><tr><td rowspan="3">Pi</td><td>GPT-4o-mini</td><td></td><td>0</td><td>0.016</td><td>74</td><td>59</td><td>35</td><td>85</td><td>63</td></tr><tr><td>GPT-6 Luna</td><td>high</td><td>0</td><td>0.0032</td><td>96</td><td>81</td><td>92</td><td>92</td><td>91</td></tr><tr><td>Claude Opus 5.5</td><td>high</td><td>0</td><td>0.046</td><td>99</td><td>100</td><td>100</td><td>100</td><td>99</td></tr><tr><td rowspan="3">Codex</td><td>GPT-4o-mini</td><td>一</td><td>0</td><td>0.0031</td><td>30</td><td>22</td><td>24</td><td>46</td><td>28</td></tr><tr><td>GPT-6 Luna</td><td>high</td><td>0</td><td>0.0050</td><td>90</td><td>84</td><td>89</td><td>85</td><td>88</td></tr><tr><td>Claude Opus 5.5</td><td>high</td><td>0</td><td>0.167</td><td>99</td><td>100</td><td>95</td><td>100</td><td>98</td></tr><tr><td rowspan="3">Claude Code</td><td>GPT-4o-mini</td><td></td><td>0</td><td>0.0020</td><td>34</td><td>22</td><td>8</td><td>77</td><td>29</td></tr><tr><td>GPT-6 Luna</td><td>high</td><td>0</td><td>0.0082</td><td>96</td><td>78</td><td>95</td><td>92</td><td>91</td></tr><tr><td>Claude Opus 5.5</td><td>high</td><td>0</td><td>0.187</td><td>99</td><td>94</td><td>97</td><td>100</td><td>97</td></tr></table>

Table 3: Writer and reader. Correct rate (%), the reader answering from the writer’s store and its retrievals. Bold: one model writes and reads (Table 2).
<table><tr><td rowspan="2">Written by</td><td colspan="3">Mem0, read by</td><td colspan="3">Zep, read by</td></tr><tr><td>GPT-4o-mini</td><td>GPT-6 Luna</td><td>Opus 5.5</td><td>GPT-4o-mini</td><td>GPT-6 Luna</td><td>Opus 5.5</td></tr><tr><td>GPT-4o-mini</td><td>78</td><td>83</td><td>84</td><td>43</td><td>43</td><td>45</td></tr><tr><td>GPT-6 Luna</td><td>74</td><td>80</td><td>81</td><td>52</td><td>51</td><td>55</td></tr><tr><td>Claude Opus 5.5</td><td>89</td><td>91</td><td>95</td><td>66</td><td>71</td><td>87</td></tr></table>

![](images/08edf145a89d5a85e119105730a58b970b2af15b05cef0f37d8e0bd5afac9837.jpg)

![](images/ee496b411e81ca46af54ca8c8db408165ada9c7d1c2f511f7f59bdd2971fce88.jpg)  
Figure 4: The curves of Figure 2 against turns (LLM calls, top) and tokens (bottom), on linear axes, the questions ordered by that amount. Strong lines are cells that no other cell beats on both that amount and correct answers.

Table 4: Correct rate (%) on the 64 questions with evidence in a turn that shares a photo, whose caption only the agents could read, and on the other 88.
<table><tr><td rowspan="2">Method</td><td colspan="2">GPT-4o-mini</td><td colspan="2">GPT-6 Luna</td><td colspan="2">Claude Opus 5.5</td></tr><tr><td>Photo (64)</td><td>Other (88)</td><td>Photo (64)</td><td>Other (88)</td><td>Photo (64)</td><td>Other (88)</td></tr><tr><td>Full context</td><td>66</td><td>81</td><td>91</td><td>93</td><td>89</td><td>99</td></tr><tr><td>BM25</td><td>61</td><td>83</td><td>78</td><td>84</td><td>84</td><td>94</td></tr><tr><td>Mem0</td><td>70</td><td>84</td><td>72</td><td>86</td><td>91</td><td>99</td></tr><tr><td>Zep</td><td>33</td><td>50</td><td>53</td><td>50</td><td>77</td><td>94</td></tr><tr><td>Pi</td><td>53</td><td>69</td><td>88</td><td>94</td><td>98</td><td>100</td></tr><tr><td>Codex</td><td>22</td><td>33</td><td>84</td><td>91</td><td>97</td><td>99</td></tr><tr><td>Claude Code</td><td>17</td><td>38</td><td>91</td><td>92</td><td>95</td><td>99</td></tr></table>

# Question:   
{{QUESTION}}   
# Context:   
{{CONTEXT}}   
# Short answer:

## B Protocols

Mem0 OSS 2.1.0 writes one chronological turn at a time and retrieves � = 200 without reranking; dates are prepended to the text. Zep runs Graphiti OSS 0.30.2 on Kuzu, one session per episode, with hybrid edge search (top 20). The memory methods’ GPT-6 Luna and Claude Opus 5.5 requests set reasoning efort high.

Each agent question is a Harbor 0.24.0 task [19]: a fresh container holding only the conversation, exactly as in the oficial file, without questions or answers. The agent’s last message is its answer. While the agent runs, the container can reach only the model endpoint. GPT-6 Luna and Claude Opus 5.5 run at reasoning efort high, through a proxy that translates the API where an agent does not speak the model’s own; GPT-4o-mini has no efort setting. Codex and Claude Code have web search of; Pi has none. A question may take 20 minutes, and Pi and Claude Code at most 100 turns. With GPT-6 Luna, Codex picks its code mode: one JavaScript tool that calls the shell. A trial that fails or ends without an answer is run again, and only the accepted trial is counted.

## C Prompts

Each prompt is copied from the code the runs called; fields in braces are filled per question.

Answer, system message.

You are a helpful assistant that can answer questions based on the provided context.If the   
question involves timing, use the conversation date for reference.Provide the shortest possible   
answer.Use words directly from the conversation when possible.Avoid using subjects in your answer.

Answer, user message.

## Pi, Codex and Claude Code, user message.

Answer a question about the conversation in \`conv-26/conv.json\`. If the question involves timing,   
use the conversation date for reference. Provide the shortest possible answer. Use words directly   
from the conversation when possible. Avoid using subjects in your answer. Your final message is   
taken as the answer.

Question: {question}

## Judge.

Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:

The point of the question is to ask about something one user should know about the other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes the referenced topic,   
for example:   
Question: Do you remember what I got the last time I went to Hawaii?   
Gold answer: A shell necklace   
The generated answer might be much longer, but you should be generous with your grading - as long   
as it touches on the same topic as the gold answer, it should be counted as CORRECT.   
For time related questions, the gold answer will be a specific date, month, year, etc. The   
generated answer might be much longer or use relative time references (like "last Tuesday" or   
"next month"), but you should be generous with your grading - as long as it refers to the same   
date or time period as the gold answer, it should be counted as CORRECT. Even if the format   
differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.   
Now it's time for the real question:   
Question: {question}   
Gold answer: {gold\_answer}   
Generated answer: {generated\_answer}   
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or   
WRONG.   
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.   
Just return the label CORRECT or WRONG in a json format with the key as "label".