# Forty Questions: Identifying a Language Model from One-Word Answers

Tapan Parikh Cornell Tech tsp53@cornell.edu

October 2026

## Abstract

A model behind an API can be identified from its one-word answers to everyday questions such as “Name a sound” or “Name a poet”. A model’s profile records how often it gives each answer to each question. We attribute new answers to the model most likely to have given them. On a panel of 94 models, 40 answers identify the exact model 92.5% of the time, against 1.1% by chance. Profiles built through OpenRouter identified all ten models served by DeepInfra under blinded labels. A system prompt lowers accuracy by 6 to 16 points. We applied the method to 96 endpoints of 13 open models on OpenRouter, and 92 identify as the model they claim. Of the other four, one matches the claimed model with reasoning turned of. One serves Gemma 4 under the name Llama 4 Maverick: it names itself Gemma 4, and it is billed Gemma 4’s token counts on every message. Two remain unresolved. Billing records also show context added to first-party models. Grok 4.7 is billed 1,247 input tokens for a one-line question that costs GPT-4o only 12 tokens, and five models state the current date without being billed for it. The questions come from a public census of model defaults, and the method costs cents per endpoint.

## 1 Introduction

A deployed language model is usually a name behind an API. The caller sees neither the weights nor how they are served. A provider may quantize the weights, add a default system prompt or change the reasoning setting, and each of these can change the answers without changing the weights. In this paper we describe a method for the caller to tell which model is behind the API.

Recent work shows that short, constrained answers can identify a model. IRIS attributes endpoints from requests for random digits and bit strings, with a trained classifier [14]. Bruckner [1] verifies model identity from single-token answers to tasks such as “name a random number”. FLIPS identifies configured instances from requested random binary sequences [12].

A census of model defaults asked 105 models 96 one-word questions such as “Name a color” and “Name a fruit”, repeated eight times per question [10]. Models converge strongly on these questions, as they do on open-ended prompts [8]. The answers that depart from the consensus repeat within a model. A model that says “petrichor” where the field says “lavender” says it again, and these repeated departures are what we use to identify it.

We use the 94 census models with at least eight answers on at least 90 of the 96 questions (§3). We decide by likelihood under per-question answer counts. No weights are tuned, and a new model is added by sampling it once. Each decision traces to the answers behind it, so a human auditor can check it: which answers the endpoint gave, and which models give them.

The method has three common uses. An auditor checks whether an endpoint serves the model it claims (§5). A buyer checks whether a vendor’s model derives from another lab’s (§4.6). A user checks whether the model behind a name has changed (§4.5). The auditor chooses the questions and sends no system prompt. A provider can still add text before the auditor’s message, and §5 shows that this happens.

Contributions:

1. Closed-set identification of the exact model from one-word answers, with nothing learned beyond answer counts. Forty answers identify one of 94 models 92.5% of the time (§4.1).

2. Question selection. Ten questions selected on separate samples reach 81% with two answers each, against 67.5% for ten random ones (§4.2).

3. Blinded identification on a second host, and robustness to a system prompt and to repeated sampling over weeks (§4.3 to §4.5).

4. An audit of OpenRouter. Of 96 endpoints, 92 identify as the model they claim. Of the four exceptions, one is consistent with a reasoning configuration, one is a substituted model and two remain unresolved. A rescore against pinned reference profiles leaves these findings unchanged. Billed tokens and the current date give evidence of context added to first-party models (§5).

## 2 Related work

Random-sequence probes. IRIS [14] is the closest prior work. It sends short randomgeneration requests (a coin draw, a digit, an integer in a range, a bit or digit sequence) and classifies the responses with a random forest over 179 string features. On 17 OpenRouter endpoints it reaches 0.965 attribution accuracy at 8 queries. On 45 models it reaches 0.90 at 16 queries. It also flags 14 of 15 same-model OpenRouter provider pairs as distinguishable. IRIS reports that a generic text classifier cannot attribute providers from natural-language completions. Our answers are natural language, but each is one word to a fixed question, so they can be counted.

Single-token fingerprints. Bruckner [1] samples 165 OpenRouter models on ten one-token tasks in four languages. Jensen–Shannon distance between answer distributions gives verification at a 7.3% equal error rate with the full 40-cell battery, and family assignment at 59.5%. Bruckner reports no closed-set identification of the exact model. FLIPS [12] uses biases in requested binary sequences. It treats a model under a diferent system prompt or quantization as a diferent instance. It reports 96% closed-set and 90% open-set identification over 237 instances.

Query-based identification. LLMmap [11] identifies 42 model versions at over 95% from eight crafted queries with a trained model. Chen et al. [3] show that a one-sentence tone prefix in the system prompt drops clone detection on short structured outputs from 0.978 to 0.547.

Auditing APIs. Model Equality Testing [6] tests whether an API serves a reference distribution with an MMD two-sample test, and finds 11 of 31 Llama endpoints serving diferent distributions from Meta’s weights. The rank-based uniformity test [16] detects quantization, fine-tuning and swaps from outputs. Cai et al. [2] argue that software-only checks are unreliable and propose trusted execution. Stability Monitor [9] detects changes in version, stack and quantization from output distributions and finds stability diferences across hosts of one model. AgentProv [13] is the source of our check in §5.3. Its token-count scan finds 44% of 75 OpenRouter open-weight endpoints of the oficial chat template, some with injected system prompts. Under an injected prompt, MET and RUT falsely reject honest models 67% and 53% of the time.

Serving nondeterminism. Batch size varies with server load, and kernels that are not batchinvariant then change outputs. At temperature zero, Qwen3 235B gave 80 distinct completions in 1,000 [7]. Zhu and Zhang [15] find that byte-identical next-day replays on shared endpoints agree at 0.78 in a preregistered LLM-judge study.

## 3 Data and method

Questions. 96 questions, 95 of them in the form “Name a color. Reply with one word only.”, such as “Name a smell” and “Name a villain”. The other is “Pick a word. Reply with one word only.” The full list is in Appendix A.

Models. 94 models with at least eight answers on at least 90 of the 96 questions, carrying 27 family labels (Appendix B). All were queried through OpenRouter’s default routing, with no provider pinned, no system prompt, temperature 1 and default reasoning settings, between 3 July and 2 October 2026. The robustness runs in §4.4 and §4.5 were made on 1 October, the provider audit in $\ S 5$ on 2 October, and its follow-ups on 6 and 11 October. Answers are normalized as in the census: first answer line, case-folded, spelling variants and plurals merged (Appendix C).

Profile and score. A model’s profile for question c is the count of each answer among its samples. A test set $\{ ( c , a ) \}$ is scored under model m as

$$
\sum _ { ( c , a ) } \log { \frac { n _ { m } ( c , a ) + 0 . 1 } { N _ { m } ( c ) + 0 . 1 V _ { c } } } ,
$$

where $n _ { m } ( c , a )$ is the count of answer a, $N _ { m } ( c )$ the number of profiled samples and $V _ { c }$ the number of distinct answers to c across all models plus one. The prediction is the highest-scoring model.   
This is a naive Bayes classifier with a pseudocount of 0.1. Nothing is fit beyond the counts.   
When profiles have equal size, $V _ { c }$ is shared by every candidate and does not afect the ranking.   
Adding a model means sampling it once. Edge cases are in Appendix D.

Evaluation. Four-fold over samples. Samples {0,1}, {2,3}, {4,5} and {6,7} are held out in turn, and profiles use the other six, so no answer is scored against itself. Random question subsets are drawn ten times per size.

<table><tr><td rowspan="2"></td><td colspan="2">one answer each</td><td colspan="2">two answers each</td></tr><tr><td>questions exact</td><td>family</td><td>exact</td><td>family</td></tr><tr><td>5</td><td>28.1%</td><td>39.9%</td><td>41.5%</td><td>53.0%</td></tr><tr><td>10</td><td>54.5%</td><td>63.6%</td><td>68.0%</td><td>76.2%</td></tr><tr><td>20</td><td>78.2%</td><td>84.9%</td><td>90.7%</td><td>94.1%</td></tr><tr><td>40</td><td>92.5%</td><td>96.0%</td><td>96.9%</td><td>99.1%</td></tr><tr><td>96</td><td>98.4%</td><td>100.0%</td><td>98.7%</td><td>100.0%</td></tr></table>

Table 1: Identification among 94 models by number of questions and answers per question. Chance is 1.1%.

## 4 Identification

## 4.1 How many answers

Accuracy depends mostly on the total number of answers (Table 1). New questions carry slightly more than repeats: twenty questions asked twice reach 90.7%, and forty asked once reach 92.5%.

These held-out answers come from the same collection runs as the profiles they are scored against. Answers collected on other days are identified at similar rates: 98% for 14 models on a later day with 31 questions asked twice (§4.4) and 10 of 10 on a second host (§4.3).

Against simpler scorers. We compared the likelihood with two simpler scorers on the same folds and question draws. One picks the model whose answers are closest in Jensen–Shannon distance, as in Bruckner [1]. The other counts how many held-out answers match each model’s favourite answer. On the 66 questions every model answered eight times, forty answers identify the model 92% of the time under the likelihood, 65% under Jensen–Shannon distance and 76% under favourite answers. The likelihood scores best at every budget (Appendix E). With one or two held-out answers per question, the distance between distributions cannot be estimated well. Matching favourite answers ignores the occasional answers that distinguish models most.

## 4.2 Which questions

Greedy selection adds, one at a time, the question that most raises the probability given to the true model. To keep selection apart from scoring, questions are chosen on half of each model’s samples and scored on the other half. The halves are then swapped, and Table 2 averages the two. Selected and random sets both come from the 66 questions every model answered eight times.

<table><tr><td>questions (two answers each)</td><td>selected</td><td>random</td></tr><tr><td>5</td><td>62.0%</td><td>38.8%</td></tr><tr><td>10</td><td>81.1%</td><td>67.5%</td></tr><tr><td>15</td><td>90.7%</td><td>80.6%</td></tr><tr><td>20</td><td>95.8%</td><td>87.9%</td></tr><tr><td>25</td><td>97.1%</td><td>92.7%</td></tr></table>

Table 2: Selected against random question sets, mean of the two halves. Both halves’ choices are in Appendix F.

Selection beats random sets at every size, by 14 points at ten questions. The questions it picks are less stable. The two halves share four of their first ten: bird, dessert, landmark and reptile. Many questions carry similar information, so which ones win depends on the samples. Selection is tuned to this panel, so its advantage on unseen models is likely smaller.

## 4.3 A second host, blinded

Ten panel models were queried through DeepInfra under random code names, two samples per question: Llama 3.3 70B, Llama 4 Scout, Mistral Small 3.2, Mistral Nemo, Nemotron 3 Super, Qwen3 235B, GPT-OSS 120B, DeepSeek V4 Flash, DeepSeek V3.2 and Gemma 3 27B. The scorer recorded its guesses before the key was read. Profiles came only from OpenRouter answers. DeepInfra is one of OpenRouter’s providers, and the profiles’ providers were not recorded (§3), so some reference answers may have come from DeepInfra. With all questions (93 to 96 per model), 10 of 10 were identified (95% interval 69% to 100%). With the best twenty, 8 of 10 were (44% to 97%). In one miss the true model ranked second, 0.3 nats behind in total over 40 answers, which is a near tie. In the other it ranked third, 5.9 nats behind.

Equality tests can still tell hosts of one model apart [1, 14]. Two hosts can difer slightly and still both be closest to the same model.

## 4.4 Under a system prompt

Fourteen models answered 31 of the 96 questions three ways: with no system prompt, with “You are a helpful assistant.”, and with a made-up persona, “You are Ava, a friendly assistant.”. They were identified among all 94 models from profiles built with no system prompt.

<table><tr><td>condition</td><td> $1 0 \times 2$ </td><td> $2 0 \times 2$ </td><td> $3 1 \times 2$ </td><td>31 × 4</td></tr><tr><td>no system prompt, a later day</td><td>66%</td><td>90%</td><td>98%</td><td>100%</td></tr><tr><td>“You are a helpful assistant.&quot;</td><td>60%</td><td>82%</td><td>89%</td><td>94%</td></tr><tr><td>&quot;You are Ava, a friendly assistant.&quot;</td><td>56%</td><td>74%</td><td>86%</td><td>91%</td></tr></table>

Table 3: Identification of 14 models under a system prompt, among all 94 profiles (20 draws per cell).

The generic prompt costs 6 to 9 points, and the Ava persona costs 9 to 16. With 31 questions asked four times, models under a system prompt are still identified 91% to 94% of the time. A provider can add such a prompt without the auditor knowing (§5.3).

## 4.5 Over weeks

The fourteen models’ favourite answers in the robustness run matched their original runs, made weeks to three months earlier, on 89% of questions. Two halves of the same day’s samples matched in 84%, with fewer samples per half. On this coarse measure the fingerprint is stable over the collection period.

## 4.6 A lineage

Writer markets Palmyra X5 as its own model and has not named a base model for it. Writer’s report on the successor, Palmyra x6, names GLM-5.2 as that model’s base [4]. Palmyra X5’s nearest model on our 96 questions is Qwen3 235B Instruct (2507), by 2.3 nats per question over the next candidate. Qwen3 235B’s nearest model is Palmyra X5. Bruckner [1] found the same pair at a distance equal to the median distance between halves of one model. Mutual nearest neighbours show that the two answer alike. They do not prove that one is derived from the other. Palmyra X5 is also billed 243 input tokens for a one-line question (§5.3), so added context may shape its answers.

## 5 Validation: an audit of OpenRouter providers

## 5.1 Setup

OpenRouter routes each open model to one of several providers, with declared precision (bf16, fp8, fp4, int4 or unstated) and prices that difer up to thirtyfold for one model. We pinned each of 104 endpoints of 14 open models in turn, with fallbacks of, and asked all 96 questions four times. Eight could not be scored. Four returned no data, and four claim Qwen3.5 122B, which is not in the panel because it answered fewer than 90 questions eight times. The rest of this section reports the other 96 endpoints, of 13 models. The whole audit cost \$7.20 over 36,918 calls. The released transcripts identify each endpoint by provider. The text characterizes endpoints and names no third-party provider; Appendix G lists them by model and letter.

Each endpoint is scored two ways. Identification ranks the 94 panel profiles by the likelihood of the endpoint’s answers, and reports where the claimed model ranks. A peer test compares the endpoint with the pooled answers of the same model’s other providers, collected the same day. The distance is the mean Jensen–Shannon divergence across questions. Its p-value comes from 1,000 label permutations within each question, with Benjamini–Hochberg correction across endpoints.

## 5.2 Results

Of the 96 endpoints, 92 identify as the model they claim. Table 4 lists the four that do not.

The peer test flags 22 endpoints as diferent from the same model’s other providers at $q < 0 . 0 5$ These include all four mismatches. The other 18 still identify as the claimed model, so they difer from their peers within the claimed model. The reasoning setting is one source of diference. DeepSeek V4 Flash, for example, reasoned on at least 99% of calls at 11 of its 15 endpoints, on no call at three, and on 38% of calls at one. The peer pools include the mismatched endpoints, so their peers can be flagged too. Both Maverick endpoints that identify as Maverick have $q = 0 . 0 0 6$

<table><tr><td>claimed model</td><td>declared precision</td><td>identified as</td><td>rank of claim</td><td>q</td></tr><tr><td>Llama 4 Maverick</td><td>fp8</td><td>Gemma 4 31B</td><td>91</td><td>0.006</td></tr><tr><td>Mistral Small 3.2</td><td>unstated</td><td>Mistral Small 2603</td><td>5</td><td>0.006</td></tr><tr><td>Kimi K2.5</td><td>unstated</td><td>Kimi K2</td><td>2</td><td>0.006</td></tr><tr><td>DeepSeek V4 Flash</td><td>fp4</td><td>Qwen 3.8 27B</td><td>4</td><td>0.048</td></tr></table>

Table 4: The four endpoints that do not identify as claimed. Rank is the claimed model’s rank among 94 profiles.

The Llama 4 Maverick endpoint is identified as Gemma 4 31B and ranks Maverick 91st of 94, so its answers are far from anything Maverick says. Section 5.4 confirms that it serves Gemma 4: it names itself Gemma 4, and it is billed Gemma 4’s token counts.

The other three rank their claimed model near the top. The Mistral Small 3.2 endpoint is identified as Mistral Small 2603, a later release from the same developer, and ranks the claimed model fifth. The Kimi K2.5 endpoint is identified as Kimi K2, the earlier version of the same model, and ranks the claimed model second. The DeepSeek V4 Flash endpoint is identified as Qwen 3.8 27B, a model from another lab, and ranks the claimed model fourth. The Mistral and DeepSeek identifications change between runs (§5.4), so neither should be read as the model served.

How far a match is from a mismatch. The margin is the claimed model’s score minus the best other model’s, in nats per question. It is positive when the endpoint is identified as claimed. Over the 92 matching endpoints the median margin is +1.6, the lowest three are +0.12, +0.20 and +0.24, and 95% are above +0.47. The four mismatches sit at −7.0 (Llama 4 Maverick), −2.4 (Mistral Small 3.2), −0.9 (Kimi K2.5) and −0.25 (DeepSeek V4 Flash). The DeepSeek V4 Flash margin is about as far below zero as the weakest matches are above it, so on the fingerprint alone it is a near miss.

Mixed reference profiles. The panel profiles were collected through default routing (§3), so each may include answers from the endpoints this audit flags. We estimated the share for each mismatch with a two-part mixture fitted to the panel profile’s answers. One part is the mismatch’s answer distribution, and the other is the pooled answers of the same model’s matching endpoints. The share is 0.5% for the Gemma 4 endpoint in the Llama 4 Maverick profile, 2% for Kimi K2.5 and 10% for DeepSeek V4 Flash. For Mistral Small 3.2 it is 17%, most of it in the second four samples of the first 31 questions, collected on 30 September (64% of those answers). The mismatched Mistral endpoint is the developer’s own API, which default routing would often reach.

We then rescored every endpoint with its claimed model’s profile replaced by a pinned one: eight answers per question drawn from the same model’s other matching endpoints on the audit day, the endpoint itself left out. The four mismatches stay mismatches, with the claimed model ranked 92nd, 10th, 2nd and 7th. Of the 91 matching endpoints that have a matching peer, 90 still identify as claimed. The exception is a GLM 4.7 endpoint that falls to second, behind Qwen 3.7 Plus, by 0.09 nats per question. Mixing in the reference profiles therefore does not explain the mismatches, and it does not create the matches.

## 5.3 Added prompts

An endpoint can serve the claimed weights and still answer diferently, because text was added before the auditor’s message. AgentProv shows that this makes equality tests reject honest models. Section 4.4 shows it lowers identification accuracy. A provider pays to process added text on every call, so billed input tokens above the count for the sent message reveal it.

Hidden context in first-party models. All calls in this section went through OpenRouter, so text added by OpenRouter cannot be told apart from text added by the lab. On the audit day we asked every live panel model, with no system prompt, “What is today’s date?”. A model told nothing can only guess the date. GPT-5, GPT-5.4, GPT-5.4 mini and GPT-5.5 stated the correct date in both samples. Each was billed 11 input tokens for the question. GPT-5.6 and the GPT-6 models were billed the same 11 tokens and did not know the date. Muse Spark 1.3, from another lab, also stated the date in both samples while billed 12 tokens. The date reaches these five models without being billed, either as unbilled prompt text or through another channel such as server-side metadata. Either way, billed tokens are a lower bound on added context.

Other models are billed for large added prefixes (Table 5). Tokenizers difer, so the GPT-4o count is a reference and not an exact baseline.

<table><tr><td>model</td><td>billed input tokens</td></tr><tr><td>GPT-4o (reference)</td><td>12</td></tr><tr><td>Grok 4.7</td><td>1,247</td></tr><tr><td>Palmyra X5</td><td>243</td></tr><tr><td>Grok 4.5 and 4.6</td><td>211</td></tr><tr><td>MiniMax M3</td><td>181</td></tr><tr><td>Kimi K3</td><td>90 to 93</td></tr></table>

Table 5: Billed input tokens for “What is today’s date?”, no system prompt.

Grok 4.7’s surplus over GPT-4o is 1,235 tokens on every message we sent, from a census question to a request to quote its instructions. Reasoning is billed separately as output and varies from 29 to 498 tokens per call, while the input count stays fixed. From the second call on, 1,152 input tokens per call came from cache, so the prefix is the same text every time. We cannot see its content.

The billed-token check on the audit. For each endpoint we compared billed input tokens with the most common count among the same model’s other providers for the same message. Peers share a tokenizer, so a diference measures added text or a diferent chat template. Of the 95 scored endpoints that have peers, 19 are billed above their peers. Fifteen of the 19 identify as the claimed model, including three Llama 3.3 70B endpoints at +25 tokens and two DeepSeek V4 Flash endpoints at +79. All four mismatches are among the 19.

Four days after the audit we sent three messages to each mismatch and to one or two of its peers: the census question, the same with our own one-line system prompt, and a longer user message (Table 6).

<table><tr><td></td><td></td><td colspan="3">four days later</td></tr><tr><td>claimed model</td><td>audit</td><td>plain</td><td>our system prompt</td><td>longer message</td></tr><tr><td>Llama 4 Maverick</td><td>+3</td><td>+3</td><td>+4</td><td>+3</td></tr><tr><td>Mistral Small 3.2</td><td>+12</td><td>+12</td><td>+12</td><td>+12</td></tr><tr><td>Kimi K2.5</td><td>+19</td><td>+19</td><td>+1</td><td>+19</td></tr><tr><td>DeepSeek V4 Flash</td><td>+79</td><td>+79</td><td>+79</td><td>+79</td></tr></table>

Table 6: Billed input tokens above the peers’ count, per mismatch.

The Mistral Small 3.2, Kimi K2.5 and DeepSeek V4 Flash surpluses are constant across message lengths, so each is a fixed prefix. The Kimi K2.5 prefix disappears when the caller sends a system prompt, which is consistent with a default system prompt that the caller’s replaces. The other two persist.

## 5.4 Follow-up runs on the four mismatches

Four days after the audit we ran three follow-ups, each with all 96 questions four times per endpoint.

1. A later day. The four mismatches again, with no system prompt.

2. A caller’s system prompt. Every endpoint of the four models, mismatches and peers, under “You are a helpful assistant.”. On many serving stacks a caller’s system prompt replaces a provider’s default one. We assume it does here, and the Kimi K2.5 surplus falling to +1 shows it on one endpoint.

3. Reasoning of. Every endpoint of Kimi K2.5 with reasoning turned of. The Kimi K2.5 mismatch answered with a median of 3 output tokens and no reasoning, where its peers reasoned on every call.
<table><tr><td>claimed model</td><td>audit</td><td>later day</td><td>caller&#x27;s system prompt</td><td>surplus</td></tr><tr><td>Llama 4 Maverick</td><td>Gemma 4 31B (91)</td><td>Gemma 4 31B (91)</td><td>Gemma 4 31B (91)</td><td>+4</td></tr><tr><td>Mistral Small 3.2</td><td>Mistral Small 2603 (5)</td><td>Mixtral 8x22B (3)</td><td>Mistral Small 2603 (5)</td><td>+12</td></tr><tr><td>Kimi K2.5</td><td>Kimi K2 (2)</td><td>Kimi K2 (2)</td><td>Kimi K2 (2)</td><td>+1</td></tr><tr><td>DeepSeek V4 Flash</td><td>Qwen 3.8 27B (4)</td><td>Grok 4.5 (7)</td><td>Grok 4.5 (9)</td><td>+79</td></tr></table>

Table 7: Identification of the four mismatches across runs, with the claimed model’s rank in parentheses. Surplus is billed input tokens above the peers under the caller’s system prompt.

None of the four endpoints came out as its claimed model on any of the three runs. Under the caller’s system prompt, 21 of the 22 peers that returned answers identify as their claimed model. The exception is a DeepSeek V4 Flash peer whose claimed model ranks second. All four mismatches difer from their peers at q < 0.05 under the system prompt.

Kimi K2.5. Under the caller’s system prompt its billed surplus falls from +19 to +1, so a default prompt was removed, and it still identifies as Kimi K2. With reasoning of, all three peers that returned one-word answers also identify as Kimi K2, the non-reasoning predecessor, with the claimed model second. The fourth peer returned its reasoning as the reply when reasoning was of and could not be scored. The mismatch is therefore consistent with the claimed model served with reasoning of. The fingerprint cannot separate this from Kimi K2 served under the K2.5 name, since both give the same answers. We count it as a likely configuration diference.

A disclosure probe on the other three. Each of the other three keeps its billed surplus under the caller’s system prompt. A two-sample test such as Model Equality Testing would only confirm what the peer test shows, that they difer from their peers. We asked instead what the endpoints say about themselves. Each mismatch and two of its peers received three messages with no system prompt, three samples each: “What is today’s date?”, a request to quote any instructions given before the message, and “What is your name, and which company made you?”.

Llama 4 Maverick. The endpoint named itself Gemma 4, developed by Google DeepMind, in two of three samples, and said it was trained by Google in all three answers about its instructions. Both peers named Llama and Meta in every sample. The endpoint was billed 20, 43 and 24 input tokens on the three messages. Gemma 4 31B through its own providers is billed exactly 20, 43 and 24 on the same messages, and Llama 4 Maverick 15, 40 and 21. The surplus is the diference between two tokenizers. The fingerprint, the model’s account of itself and the tokenizer agree, so we count this endpoint as Gemma 4 served under the name Llama 4 Maverick.

DeepSeek V4 Flash. The endpoint reasoned for 2,100 to 3,800 tokens on the date and instruction questions, in numbered markdown steps. Two other providers are billed the same 79-token surplus and identify as the claimed model (§5.3). The one in this probe reasoned for 42 to 485 tokens, in the terse style of every other DeepSeek V4 Flash endpoint. If the two prefixes are the same text, the prefix does not cause the long reasoning, though equal counts do not prove equal text. While reasoning about its instructions, the endpoint listed “You are a large language model, trained by Google” and asked itself whether it may call itself Gemma. Its replies named DeepSeek, and its billed counts fit the claimed model’s tokenizer plus the surplus. Models confabulate their instructions, so the reasoning text is weak evidence alone. The evidence points away from the claimed weights, but the served model remains unresolved.

Mistral Small 3.2. This endpoint is the developer’s own API. It reported no instructions, identified itself as Mistral AI, and gave invented dates from 2024 and 2025. Its billed counts (21, 45 and 26) are the developer API’s counts for both Mistral Small 3.2 and Mistral Small 2603, which share a tokenizer and template. Nothing in the probe separates two releases from one developer. The fingerprint places the endpoint nearest Mistral Small 2603 on two of three runs. That is consistent with the developer serving a newer release under the older name. It is also consistent with the claimed weights under the endpoint’s fixed 12-token prefix, which persists under the caller’s system prompt. It also remains unresolved.

The 92 matches show that the fingerprint identifies models as a live service serves them. The Kimi K2.5 and Llama 4 Maverick cases show two things a mismatch can mean: a configuration and another lab’s model.

## 6 Limitations

1. Serving nondeterminism. Outputs from one endpoint vary with load, batch size and hardware [7]. Byte-identical next-day replays agree at 0.78 in a preregistered study of shared endpoints [15]. Nondeterminism moves token probabilities most in the range 0.1 to 0.9 [5], which is where contested answers lie. Our stability measure in §4.5 is coarse: it checks only each model’s favourite answer.

2. Closed set. The method picks among profiled models. An unseen model is assigned to its nearest relative. Open-set rejection is untested.

3. Added context and routing. Some first-party models and some providers receive text before the caller’s message (§5.3). Our profiles were built through OpenRouter’s default routing, so they include whatever was added there. A profile describes a model as served. It also mixes providers. On the audit day four of ten unpinned calls to Llama 4 Maverick were billed the token counts of the endpoint serving Gemma 4, but the Maverick profile was collected in July and September and holds about 0.5% Gemma 4 answers. The Mistral Small 3.2 profile is about 17% from the developer’s own endpoint (§5.2). Pinned reference profiles leave the audit’s conclusions unchanged, but the identification results in §4 rest on mixed profiles.

4. Billing is a lower bound. Five models received the date with no billed tokens (§5.3). Context that is not billed cannot be found by the billed-token check.

5. Within-family confusion. Most errors at small budgets fall outside the true family. As answers grow, the remaining errors move inside it, and at 96 questions all of them are within one family.

6. Reasoning setting. Profiles were built at each model’s default reasoning setting. A hybrid model served with reasoning of can resemble a diferent model, as Kimi K2.5 resembles Kimi K2 (§5.4). Profiling hybrid models in both settings would close this gap.

7. Temperature. All data are at temperature 1.

8. No matched baseline from prior work. We did not run random-sequence probes on this panel, so any comparison with IRIS and FLIPS rests on their published numbers.

9. Gaming. A provider who knows the public questions could route them to the genuine model, add an answer key in a hidden prompt, or fine-tune a cheaper model to imitate the answers. A rotating secret reserve of questions could mitigate the first. The billed-token check could catch the second only when the added text is billed (limitation 4). None of this is tested. Many questions carry similar information (§4.2), so a reserve of new questions may work as well as the published ones. Random sets of 40 questions already identify the model 89% to 94% of the time, whichever 40 are drawn (Appendix E).

## 7 Conclusion

One-word answers to everyday questions identify the model behind an API. Forty answers pick one of 94 models 92.5% of the time with nothing learned beyond answer counts, and they identify all ten models tested on a second host. Each decision can be checked by reading the answers behind it. Applied to 96 OpenRouter endpoints, the method identifies 92 as claimed. One of the four exceptions matches the claimed model served with reasoning of. Another is Gemma 4 served under the name Llama 4 Maverick. The other two remain unresolved. Billed tokens and the date show that first-party models served through OpenRouter also receive context the caller did not send.

## Disclosure

We sent the Llama 4 Maverick finding to the provider and to OpenRouter on 14 October 2026, before posting. The endpoint was still billed Gemma 4’s token counts that day. The provider replied that the endpoint serves Meta’s oficial FP8 release of Llama 4 Maverick, and did not address the token counts. OpenRouter matched our requests to that endpoint in its records and reproduced the token counts on 15 October. On 20 October the endpoint was no longer listed on OpenRouter, and requests pinned to it returned not found. The paper names no third-party provider, and the released transcripts do.

## Data and code

Everything is in https://github.com/tap2k/modelun, under studies/consensus/, at the tag consensus-fingerprint-arxiv-v1.

• Identification: probe\_fingerprint.py, probe\_fingerprint\_baselines.py, probe\_perturb.py and probe\_fingerprint\_robust.py, with outputs in probes/.

• Audit: probe\_provider\_audit.py, probe\_billed\_tokens.py, run\_provider\_followup.py, probe\_provider\_followup.py, probe\_reference\_contamination.py and probe\_disclosure.py. The transcripts in transcripts-providers/ identify each endpoint by provider.

• Blind test: probes/blind-deepinfra-2026-10-02.json. Date and billing probe: hidden-prompt-2026-10-02/.

• Tables: fingerprint\_paper.py.

## Note on AI usage

This work was done with Claude (Opus 5.5), which helped run the probes, build the analysis and draft the text; the research questions and interpretation are the author’s. Several Claude models are also subjects of the study.

## References

[1] T. Bruckner. One token is enough: Fingerprinting and verifying large language models from single-token output distributions, 2026.

[2] W. Cai, T. Shi, X. Zhao, and D. Song. Are you getting what you pay for? auditing model substitution in LLM APIs, 2025.

[3] L. Chen, Y. Gao, J. Wang, K. Ji, and H. Chen. Do system prompts leave behavioral fingerprints? a large-scale empirical study of clone detection via output similarity, 2026.

[4] P. Du, K. Kamble, R. Vasudev, Z. Yang, R. Nadimpally, A. Krishna, W. Alshikh, and D. M. Bikel. Palmyra x6 technical report: An agentic, tool-use model post-trained via anchored supervised fine-tuning, 2026.

[5] T. Fu, G. Martínez, J. Conde, C. Arriaga, P. Reviriego, X. Qi, and S. Liu. Beyond reproducibility: Token probabilities expose large language model nondeterminism, 2026.

[6] I. Gao, P. Liang, and C. Guestrin. Model equality testing: Which model is this API serving? In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.20247.

[7] H. He and Thinking Machines Lab. Defeating nondeterminism in LLM inference. Thinking Machines Lab blog, 10 September 2025, 2025. URL https://thinkingmachines.ai/blog/ defeating-nondeterminism-in-llm-inference/.

[8] L. Jiang, Y. Chai, M. Li, M. Liu, R. Fok, N. Dziri, Y. Tsvetkov, M. Sap, A. Albalak, and Y. Choi. Artificial hivemind: The open-ended homogeneity of language models (and beyond). In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2510.22954.

[9] J. Leshin, M. Shah, I. Timmis, and D. Kang. Behavioral fingerprints for LLM endpoint stability and identity, 2026.

[10] T. Parikh. The one-word census: Answer-choice conformity across 105 language models, 2026. Version 4.

[11] D. Pasquini, E. M. Kornaropoulos, and G. Ateniese. LLMmap: Fingerprinting for large language models. In USENIX Security Symposium, 2025. arXiv:2407.15847.

[12] G. Richardeau, G. Dashyan, E. Le Merrer, and G. Tredan. FLIPS: Instance-fingerprinting for LLMs via pseudo-random sequences. In International Conference on Machine Learning (ICML), 2026. arXiv:2606.03330.

[13] X. Wang, B. Zhao, M. Backes, F. Boenisch, and A. Dziedzic. AgentProv: Auditing agentic LLM API providers via tool-use policy probes, 2026.

[14] Y. Zhang, Z.-H. Zhang, and H. Qin. Which model is actually serving you? IRIS: Budgeted black-box auditing of model substitution and routing dilution in LLM gateways, 2026.

[15] H. Zhu and J. Zhang. Clean engineering, unstable measurement: A preregistered reliability failure of black-box LLM observers on shared endpoints, 2026.

[16] X. Zhu, Y. Ye, T. Qiu, H. Zhu, S. Tan, A. Mannan, J. Michala, R. A. Popa, and W. Neiswanger. Auditing black-box LLM APIs with a rank-based uniformity test, 2025.

## A The 96 questions

Every question is “Name a {category}. Reply with one word only.”, sent as the only user turn with no system prompt, except “any word”, which is “Pick a word. Reply with one word only.” A dagger marks the 31 questions of the original census, used in §4.4 and §4.5.

animal<sup>†</sup>, any word<sup>†</sup>, author, berry, beverage<sup>†</sup>, bird<sup>†</sup>, board game<sup>†</sup>, brand, bread, building, candy, car brand, cheese<sup>†</sup>, city<sup>†</sup>, clothing, cloud, cocktail, color<sup>†</sup>, company, composer, condiment<sup>†</sup>, constellation, country<sup>†</sup>, currency, dance<sup>†</sup>, deity, desert, dessert<sup>†</sup>, dinosaur<sup>†</sup>, dog breed, element, emotion<sup>†</sup>, fabric<sup>†</sup>, fish<sup>†</sup>, flower<sup>†</sup>, food, fruit<sup>†</sup>, furniture, gemstone<sup>†</sup>, genre, grain, hat, herb<sup>†</sup>, hero, hobby<sup>†</sup>, holiday, insect<sup>†</sup>, instrument<sup>†</sup>, inventor, island, lake, landmark, language<sup>†</sup>, leader, mammal, material, metal<sup>†</sup>, mineral, mountain, mushroom, mythical creature<sup>†</sup>, nut, occupation<sup>†</sup>, painter, pasta, philosopher, plant, poet, religion, reptile, river, rock, sauce, scientist, sea creature, shape, smell, snack, sound, soup, spice, sport<sup>†</sup>, superpower, team, texture, tool<sup>†</sup>, toy, tree<sup>†</sup>, utensil, vegetable<sup>†</sup>, vehicle, vice, villain, virtue, weapon, weather.

## B The 94 models

OpenRouter label, family label and collection dates (month-day, 2026). A dagger marks the 14 models of §4.4 and §4.5; an asterisk marks the 10 models of the blind test (§4.3).

<table><tr><td>model</td><td>family</td><td>collected</td></tr><tr><td>claude-fable-5</td><td>claude</td><td>07-04 to 09-30</td></tr><tr><td>claude-fable-5.1</td><td>claude</td><td>09-03 to 09-30</td></tr><tr><td>claude-haiku-4.5</td><td>claude</td><td>07-03 to 09-30</td></tr><tr><td>claude-opus-4.1</td><td>claude</td><td>09-25 to 09-30</td></tr><tr><td>claude-opus-4.5</td><td>claude</td><td>09-25 to 09-30</td></tr><tr><td>claude-opus-4.6</td><td>claude</td><td>09-25 to 09-30</td></tr><tr><td>claude-opus-4.7</td><td>claude</td><td>09-25 to 09-30</td></tr><tr><td>claude-opus-4.8</td><td>claude</td><td>07-03 to 09-30</td></tr><tr><td>claude-opus-5</td><td>claude</td><td>09-03 to 09-30</td></tr><tr><td>claude-opus-5.5†</td><td>claude</td><td>09-25 to 09-30</td></tr><tr><td>claude-sonnet-4.5</td><td>claude</td><td>09-25 to 09-30</td></tr><tr><td>claude-sonnet-4.6</td><td>claude</td><td>07-04 to 09-30</td></tr><tr><td>claude-sonnet-5</td><td>claude</td><td>07-03 to 09-30</td></tr><tr><td>claude-sonnet-5.5</td><td>claude</td><td>09-28 to 09-30</td></tr><tr><td>command-a</td><td>cohere</td><td>07-04 to 09-30</td></tr><tr><td>deepseek-chat-v3-0324</td><td>deepseek</td><td>07-03 to 09-30</td></tr><tr><td>deepseek-v3.2*</td><td>deepseek</td><td>07-03 to 09-30</td></tr><tr><td>deepseek-v4-flash* deepseek-v4-pro†</td><td>deepseek</td><td>07-03 to 09-30</td></tr><tr><td>ernie-4.5-vl-424b-a47b</td><td>deepseek</td><td>09-03 to 09-30</td></tr><tr><td>gemini-2.5-flash</td><td>ernie</td><td>07-03 to 09-30</td></tr><tr><td>gemini-2.5-pro</td><td>gemini</td><td>07-03 to 09-30</td></tr><tr><td>gemini-3-flash-preview</td><td>gemini</td><td>09-25 to 09-30</td></tr><tr><td>gemini-3.1-flash-lite</td><td>gemini</td><td>09-25 to 09-30</td></tr><tr><td>gemini-3.1-pro-preview</td><td>gemini</td><td>09-03 to 09-30</td></tr><tr><td>gemini-3.5-flash</td><td>gemini</td><td>07-03 to 09-30</td></tr><tr><td>gemini-3.6-flash</td><td>gemini</td><td>07-03 to 09-30</td></tr><tr><td>gemini-3.8-flash†</td><td>gemini</td><td>09-25 to 09-30</td></tr><tr><td>gemma-3-27b-it*</td><td>gemini</td><td>09-03 to 09-30 07-03 to 09-30</td></tr><tr><td>gemma-4-26b-a4b-it</td><td>gemini</td><td></td></tr><tr><td>gemma-4-31b-it†</td><td>gemini</td><td>09-03 to 09-30</td></tr><tr><td>glm-4.7</td><td>gemini</td><td>09-03 to 09-30</td></tr><tr><td>glm-5.3†</td><td>glm</td><td>07-04 to 09-30</td></tr><tr><td>glm-5.3-flash</td><td>glm</td><td>09-03 to 09-30</td></tr><tr><td>gpt-3.5-turbo</td><td>glm</td><td>09-03 to 09-30</td></tr><tr><td>gpt-4-turbo</td><td>gpt</td><td>07-03 to 09-30</td></tr><tr><td>gpt-4.1</td><td>gpt</td><td>07-04 to 09-30</td></tr><tr><td>gpt-4o</td><td>gpt</td><td>07-04 to 09-30</td></tr><tr><td>gpt-4o-mini-2024-07-18†</td><td>gpt</td><td>07-04 to 09-30</td></tr><tr><td>gpt-5</td><td>gpt</td><td>07-03 to 09-30</td></tr><tr><td>gpt-5.4</td><td>gpt</td><td>07-04 to 09-30</td></tr><tr><td>gpt-5.4-mini</td><td>gpt</td><td>07-03 to 09-30</td></tr><tr><td>gpt-5.5</td><td>gpt</td><td>09-25 to 09-30</td></tr><tr><td></td><td>gpt</td><td>07-03 to 09-30</td></tr><tr><td>gpt-5.6-luna</td><td>gpt</td><td>07-10 to 09-30</td></tr><tr><td>gpt-5.6-sol</td><td>gpt</td><td>07-10 to 09-30</td></tr><tr><td>gpt-5.6-terra</td><td>gpt</td><td>07-10 to 09-30</td></tr><tr><td>gpt-6-astra</td><td>gpt</td><td>09-04 to 09-30</td></tr><tr><td>gpt-6-luna</td><td>gpt</td><td>09-25 to 09-30</td></tr><tr><td>gpt-6-sol</td><td>gpt</td><td>09-25 to 09-30</td></tr><tr><td>gpt-6.1-sol†</td><td>gpt</td><td>09-29 to 09-30</td></tr><tr><td>gpt-oss-120b*</td><td>gpt-oss</td><td>09-03 to 09-30</td></tr><tr><td>gpt-oss-20b</td><td>gpt-oss</td><td>09-03 to 09-30</td></tr><tr><td>granite-4.2-30b</td><td>granite</td><td>10-02</td></tr><tr><td>granite-4.2-8b</td><td>granite</td><td>10-02</td></tr><tr><td>grok-4.20</td><td>grok</td><td>07-10 to 09-30</td></tr><tr><td>grok-4.3</td><td>grok</td><td>07-04 to 09-30</td></tr><tr><td>grok-4.5</td><td>grok</td><td>07-10 to 09-30</td></tr><tr><td>grok-4.6</td><td>grok</td><td>09-03 to 09-30</td></tr><tr><td>grok-4.7†</td><td>grok</td><td>09-25 to 09-30</td></tr><tr><td>hermes-3-llama-3.1-70b</td><td>hermes</td><td>10-02</td></tr><tr><td>hermes-4-405b</td><td>hermes</td><td>09-29 to 09-30</td></tr><tr><td>hy4-preview</td><td>hunyuan</td><td>10-02</td></tr><tr><td>inkling</td><td>inkling</td><td>10-02</td></tr><tr><td>inkling-small</td><td>inkling</td><td>10-02</td></tr><tr><td>kimi-k2</td><td>kimi</td><td>09-25 to 09-30</td></tr><tr><td>kimi-k2.5</td><td>kimi</td><td>07-04 to 09-30</td></tr><tr><td>kimi-k3†</td><td>kimi</td><td>09-03 to 09-30</td></tr><tr><td>ling-3.0-flash llama-3.3-70b-instruct†*</td><td>ling</td><td>10-02</td></tr><tr><td></td><td>llama</td><td>07-03 to 09-30</td></tr><tr><td>llama-4-maverick llama-4-scout*</td><td>llama</td><td>07-03 to 09-30</td></tr><tr><td>mimo-v2.6-pro</td><td>llama</td><td>09-03 to 09-30</td></tr><tr><td>minimax-m3</td><td>mimo</td><td>10-02</td></tr><tr><td>mistral-nemo*</td><td>minimax</td><td>09-03 to 09-30</td></tr><tr><td>mistral-small-2603†</td><td>mixtral</td><td>09-03 to 09-30</td></tr><tr><td>mistral-small-3.2-24b-instruct*</td><td>mixtral</td><td>09-29 to 09-30</td></tr><tr><td>mixtral-8x22b-instruct</td><td>mixtral</td><td>09-03 to 09-30</td></tr><tr><td>muse-glimmer-30b</td><td>mixtral</td><td>07-03 to 09-30</td></tr><tr><td>muse-spark-1.3†</td><td>muse</td><td>10-02</td></tr><tr><td>mythomax-12-13b</td><td>muse</td><td>09-25 to 09-30</td></tr><tr><td>nemotron-3-super-120b-a12b*</td><td>mythomax</td><td>07-04 to 09-30</td></tr><tr><td>nemotron-3-ultra-550b-a55b†</td><td>nemotron</td><td>09-29 to 09-30</td></tr><tr><td></td><td>nemotron</td><td>09-29 to 09-30</td></tr><tr><td>nemotron-3.5-lightning palmyra-x5</td><td>nemotron</td><td>09-29 to 09-30</td></tr><tr><td>phi-4</td><td>palmyra</td><td>07-04 to 09-30</td></tr><tr><td>qwen-2.5-72b-instruct</td><td>phi</td><td>10-02 07-04 to 09-30</td></tr><tr><td>qwen3-235b-a22b-2507*</td><td>qwen</td><td>07-03 to 09-30</td></tr><tr><td>qwen3.5-27b</td><td>qwen</td><td></td></tr><tr><td>qwen3.6-35b-a3b</td><td>qwen</td><td>09-03 to 09-30 09-03 to 09-30</td></tr><tr><td></td><td>qwen</td><td></td></tr><tr><td>qwen3.7-plus</td><td>qwen</td><td>09-25 to 09-30</td></tr><tr><td>qwen3.8-2.4t-a95b†</td><td>qwen</td><td>09-03 to 09-30</td></tr><tr><td>qwen3.8-27b</td><td>qwen</td><td>09-29 to 09-30</td></tr><tr><td>seed-2.0-pro</td><td>seed</td><td>10-02</td></tr><tr><td>sonar</td><td>sonar</td><td>07-04 to 09-30</td></tr></table>

The census transcript directories hold 106 models. Twelve are left out. Three have no question answered eight times (claude-3-haiku, granite-4.1-8b, hermes-4-70b), because they left OpenRouter before the second four runs. Six answer fewer than 90 questions eight times, from failed cells or replies that are not one word: wizardlm-2-8x22b (69), qwen3.5-9b (79), hermes-3- llama-3.1-405b (83), qwen3.5-122b-a10b (84), hunyuan-a13b-instruct (87) and nemotron-3-nano-30b-a3b (88). Three were added after the panel was fixed on 6 October: claude-haiku-5.5, and deepseek-r1 and step-3.7-flash, which crossed the threshold after failed cells were refilled on 11 October. The panel was fixed after the 2 October audit, and the audit was scored against it.

## C From reply to answer

Replies are reduced to answers in six steps (analyze.py).

1. Strip a closed reasoning block, chat-template tokens, a leading “ASSISTANT:”, anything after a stop marker, and leftover tags. Reasoning returned in the API’s separate reasoning field is never read.

2. Take the first line that is an answer, skipping lead-ins (“Sure, here is one:”) and asides. A self-correction (“Wait, . . . One word: Stonehenge”) replaces the first answer.

3. Lower-case, fold accents, drop punctuation and emoji, and take the last word, so “A common color is blue.” gives “blue”. A two- or three-word name ending in a generic head noun of its category is joined (“soy sauce” to “soysauce”, “golden retriever” to “goldenretriever”).

4. Drop failed cells: unclosed reasoning, a reply that opens as leaked reasoning, replies over 15 words, bare acknowledgments, single letters and function words. They are not scored. Of 72,192 panel cells, 83 (0.11%) give no usable answer; the most are Mixtral 8x22B (23) and Mistral Small 2603 (19).

5. Merge spelling variants with a hand-checked map per category, built from a similarity scan and a category-by-category review by four reviewers before any fingerprint analysis. Near-matches that are diferent things are kept apart (serenity and serendipity, pine and pineapple).

6. Merge plurals: an answer ending in “s” becomes its stem when the stem occurs in the panel’s answers to that question.

Answers from outside the panel (the blind test, the audit, the follow-ups) go through the same steps and are merged onto the panel’s answers.

## D Edge cases in the score

• Profiles use a model’s first eight answers per question.

• A held-out position with no answer (a failed cell) is skipped, so that test set has fewer answers.

• A candidate with fewer profiled answers on a question has a smaller $N _ { m } ( c )$ . Then $V _ { c }$ no longer cancels exactly across candidates. This is the one place the vocabulary size afects the ranking. $V _ { c }$ counts answers from all samples, held-out ones included.

• An answer a candidate never gave scores lo $\smash { \xi ( 0 . 1 / ( N _ { m } ( c ) + 0 . 1 V _ { c } ) ) }$ . Appendix E shows how much the pseudocount matters.

• Ties go to the alphabetically first label.

## E Scorer comparison

The 66 questions every model answered eight times; the folds of §3; 20 draws per budget; Jensen–Shannon distance in bits, averaged over questions; ties in the most-common-answer scorer broken at random. Mean exact-model accuracy, standard deviation across draws in parentheses. The pseudocounts 0.01 and 0.1 give the same accuracy, and 1 costs about five points. At 40 questions asked once, the likelihood ranges from 89% to 94% over the 20 draws.
<table><tr><td>scorer</td><td> $1 0 \times 1$ </td><td> $2 0 \times 1$ </td><td> $4 0 \times 1$ </td><td> $1 0 \times 2$ </td><td> $2 0 \times 2$ </td><td> $4 0 \times 2$ </td></tr><tr><td>likelihood 0.01</td><td>50.3 (7.7)</td><td>76.2 (3.6)</td><td>91.9 (1.2)</td><td>64.7 (8.7)</td><td>88.6 (3.1)</td><td>96.9 (0.5)</td></tr><tr><td>likelihood 0.1</td><td>50.1 (7.9)</td><td>77.1 (3.7)</td><td>92.0 (1.3)</td><td>64.9 (9.1)</td><td>89.3 (2.6)</td><td>96.9 (0.5)</td></tr><tr><td>likelihood 0.5</td><td>48.0 (8.2)</td><td>74.7 (3.8)</td><td>89.4 (1.7)</td><td>61.4 (8.8)</td><td>86.1 (2.9)</td><td>95.6 (0.9)</td></tr><tr><td>likelihood 1.0</td><td>46.1 (8.3)</td><td>72.0 (4.0)</td><td>87.0 (2.1)</td><td>58.0 (8.9)</td><td>83.0 (3.2)</td><td>94.0 (1.2)</td></tr><tr><td>jsd</td><td>32.8 (7.1)</td><td>51.8 (4.5)</td><td>64.8 (2.7)</td><td>49.9 (8.7)</td><td>71.8 (4.9)</td><td>84.6 (2.7)</td></tr><tr><td>modal</td><td>31.8 (6.9)</td><td>55.4 (3.6)</td><td>75.8 (1.8)</td><td>39.3 (9.0)</td><td>67.6 (4.2)</td><td>85.3 (2.0)</td></tr></table>

## F Question selection

The objective is the mean log posterior of the true model, a softmax over the 94 candidates, summed over chosen questions. The first 25 questions chosen in each half, with the accuracy of the chosen set on the other half.
<table><tr><td></td><td>chosen on samples 0–3</td><td></td><td>chosen on samples 4–7</td></tr><tr><td>k</td><td>question</td><td>accuracy</td><td>question accuracy</td></tr><tr><td>1</td><td>mammal</td><td>15%</td><td>building 11%</td></tr><tr><td>2</td><td>sound</td><td>34%</td><td>reptile 28%</td></tr><tr><td>3</td><td>bird</td><td>47%</td><td>50% 56%</td></tr><tr><td>4</td><td>instrument</td><td>51%</td><td>landmark</td></tr><tr><td>5</td><td>reptile</td><td>61%</td><td>smell 63% 72%</td></tr><tr><td>6</td><td>landmark</td><td>72%</td><td>philosopher</td></tr><tr><td>7</td><td>dessert</td><td>73%</td><td></td></tr><tr><td>8</td><td>clothing</td><td>77%</td><td>painter color</td></tr><tr><td>9</td><td>gemstone</td><td>79%</td><td>dessert</td></tr><tr><td>10</td><td>author</td><td>80%</td><td>82%</td></tr><tr><td>11</td><td>snack</td><td>85% animal</td><td>87%</td></tr><tr><td>12</td><td>mythical creature</td><td>85% snack</td><td>87%</td></tr><tr><td>13</td><td>mushroom</td><td>88%</td><td>mythical creature 88% 90%</td></tr><tr><td>14</td><td>occupation</td><td>90%</td><td></td></tr><tr><td>15</td><td>insect</td><td>92%</td><td>90%</td></tr><tr><td>16</td><td>material</td><td>92%</td><td>occupation 94%</td></tr><tr><td>17</td><td>tool</td><td>93%</td><td>element 95%</td></tr><tr><td>18</td><td>dance</td><td>94%</td><td>mushroom 95% 95%</td></tr><tr><td>19</td><td>vegetable</td><td>93%</td><td>material dinosaur</td></tr><tr><td>20</td><td>inventor</td><td>95%</td><td>96% 96%</td></tr><tr><td>21</td><td>hat</td><td>95%</td><td>instrument island</td></tr><tr><td>22</td><td>mountain</td><td>96%</td><td>98% 98%</td></tr><tr><td>23</td><td>berry</td><td>96%</td><td>gemstone 98%</td></tr><tr><td>24</td><td>dinosaur</td><td>97%</td><td>clothing</td></tr><tr><td>25</td><td>utensil</td><td>96%</td><td>author</td></tr></table>

## G The audited endpoints

Each scored endpoint, by claimed model and a letter. Letters follow the alphabetical order of the endpoint directories in transcripts-providers/. Price is output dollars per million tokens. Rank is the claimed model’s rank among 94 profiles; margin is its score minus the best other model’s, in nats per question; pinned is the rank against a pinned reference profile (§5.2); q is the peer test.
<table><tr><td>endpoint</td><td>precision</td><td>price</td><td>identified as</td><td>rank</td><td>margin</td><td>pinned</td><td>q</td></tr><tr><td>deepseek-v3.2/A</td><td>fp8</td><td>1.11</td><td>deepseek-v3.2</td><td>1</td><td>+1.63</td><td>1</td><td>0.041</td></tr><tr><td>deepseek-v3.2/B</td><td>fp8</td><td>0.38</td><td>deepseek-v3.2</td><td>1</td><td>+1.16</td><td>1</td><td>0.292</td></tr><tr><td>deepseek-v3.2/C</td><td>fp8</td><td>0.42</td><td>deepseek-v3.2</td><td>1</td><td>+0.56</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v3.2/D</td><td>fp4</td><td>0.38</td><td>deepseek-v3.2</td><td>1</td><td>+1.27</td><td>1</td><td>0.345</td></tr><tr><td>deepseek-v3.2/E</td><td>unstated</td><td>0.96</td><td>deepseek-v3.2</td><td>1</td><td>+1.39</td><td>1</td><td>0.557</td></tr><tr><td>deepseek-v3.2/F</td><td>unstated</td><td>1.50</td><td>deepseek-v3.2</td><td>1</td><td>+1.51</td><td>1</td><td>0.006</td></tr><tr><td>deepseek-v3.2/G</td><td>fp8</td><td>0.31</td><td>deepseek-v3.2</td><td>1</td><td>+1.28</td><td>1</td><td>0.754</td></tr><tr><td>deepseek-v3.2/H</td><td>unstated</td><td>1.68</td><td>deepseek-v3.2</td><td>1</td><td>+1.60</td><td>1</td><td>0.979</td></tr><tr><td>deepseek-v3.2/1</td><td>unstated</td><td>4.50</td><td>deepseek-v3.2</td><td>1</td><td>+1.54</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v3.2/J</td><td>unstated</td><td>1.00</td><td>deepseek-v3.2</td><td>1</td><td>+1.54</td><td>1</td><td>0.848</td></tr><tr><td>deepseek-v3.2/K</td><td>unstated</td><td>4.50</td><td>deepseek-v3.2</td><td>1</td><td>+1.70</td><td>1</td><td>0.848</td></tr><tr><td>deepseek-v3.2/L</td><td>fp8</td><td>0.42</td><td>deepseek-v3.2</td><td>1</td><td>+1.27</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v3.2/M</td><td>unstated</td><td>0.39</td><td>deepseek-v3.2</td><td>1</td><td>+1.41</td><td>1</td><td>0.024</td></tr><tr><td>deepseek-v4-flash/A</td><td>fp8</td><td>0.27</td><td>deepseek-v4-flash</td><td>1</td><td>+0.20</td><td>1</td><td>0.848</td></tr><tr><td>deepseek-v4-flash/B</td><td>fp4</td><td>0.28</td><td>deepseek-v4-flash</td><td>1</td><td>+0.97</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v4-flash/C</td><td>unstated</td><td>0.56</td><td>deepseek-v4-flash</td><td>1</td><td>+1.57</td><td>1</td><td>0.006</td></tr><tr><td>deepseek-v4-flash/D</td><td>fp8</td><td>0.05</td><td>deepseek-v4-flash</td><td>1</td><td>+0.87</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v4-flash/E</td><td>fp8</td><td>0.18</td><td>deepseek-v4-flash</td><td>1</td><td>+1.71</td><td>1</td><td>0.006</td></tr><tr><td>deepseek-v4-flash/F</td><td>unstated</td><td>0.20</td><td>deepseek-v4-flash</td><td>1</td><td>+1.35</td><td>1</td><td>0.006</td></tr><tr><td>deepseek-v4-flash/G</td><td>fp8</td><td>0.18</td><td>deepseek-v4-flash</td><td>1</td><td>+1.43</td><td>1</td><td>0.333</td></tr><tr><td>deepseek-v4-flash/H</td><td>fp8</td><td>0.50</td><td>deepseek-v4-flash</td><td>1</td><td>+0.57</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v4-flash/I</td><td>fp8</td><td>0.28</td><td>deepseek-v4-flash</td><td>1</td><td>+0.12</td><td>1</td><td>0.478</td></tr><tr><td>deepseek-v4-flash/J</td><td>fp8</td><td>1.60</td><td>deepseek-v4-flash</td><td>1</td><td>+1.33</td><td>1</td><td>0.439</td></tr><tr><td>deepseek-v4-flash/K</td><td>fp8</td><td>0.28</td><td>deepseek-v4-flash</td><td>1</td><td>+0.24</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v4-flash/L</td><td>fp4</td><td>1.28</td><td>qwen3.8-27b</td><td>4</td><td>-0.25</td><td>7</td><td>0.048</td></tr><tr><td>deepseek-v4-flash/M</td><td>fp8</td><td>0.28</td><td>deepseek-v4-flash</td><td>1</td><td>+1.18</td><td>1</td><td>0.006</td></tr><tr><td>deepseek-v4-flash/N</td><td>fp8</td><td>0.06</td><td>deepseek-v4-flash</td><td>1</td><td>+0.87</td><td>1</td><td>1.000</td></tr><tr><td>deepseek-v4-flash/O</td><td>unstated</td><td>0.19</td><td>deepseek-v4-flash</td><td>1</td><td>+0.92</td><td>1</td><td>1.000</td></tr><tr><td>gemma-3-27b-it/A</td><td>fp8</td><td>0.16</td><td>gemma-3-27b-it</td><td>1</td><td>+3.89</td><td></td><td>0.164</td></tr><tr><td>gemma-3-27b-it/B</td><td>fp8</td><td>0.30</td><td>gemma-3-27b-it</td><td>1</td><td>+3.39</td><td>1 1</td><td>0.024</td></tr><tr><td>gemma-3-27b-it/C</td><td>bf16</td><td>0.20</td><td>gemma-3-27b-it</td><td>1</td><td>+3.84</td><td>1</td><td>0.215</td></tr><tr><td>gemma-3-27b-it/D</td><td>fp8</td><td>0.45</td><td>gemma-3-27b-it</td><td>1</td><td>+3.48</td><td></td><td></td></tr><tr><td>glm-4.7/A</td><td>fp8</td><td>1.85</td><td>glm-4.7</td><td>1</td><td>+0.85</td><td>1</td><td>0.006 0.558</td></tr><tr><td>glm-4.7/B</td><td>fp4</td><td>1.75</td><td>glm-4.7</td><td>1</td><td>+0.81</td><td>1 1</td><td>0.173</td></tr><tr><td>glm-4.7/C</td><td>unstated</td><td>2.20</td><td>glm-4.7</td><td>1</td><td>+0.63</td><td></td><td>0.011</td></tr><tr><td>glm-4.7/D</td><td>fp4</td><td>2.50</td><td>glm-4.7</td><td>1</td><td>+0.55</td><td>1</td><td>0.128</td></tr><tr><td>glm-4.7/E</td><td>fp8</td><td>1.98</td><td>glm-4.7</td><td>1</td><td>+0.56</td><td>1 1</td><td>0.006</td></tr><tr><td>glm-4.7/F</td><td>fp4</td><td>1.93</td><td>glm-4.7</td><td>1</td><td>+0.47</td><td>1</td><td>0.054</td></tr><tr><td>glm-4.7/G</td><td>fp4</td><td>2.20</td><td>glm-4.7</td><td>1</td><td>+0.42</td><td>2</td><td>0.006</td></tr><tr><td>gpt-oss-120b/A</td><td>bf16</td><td>0.19</td><td>gpt-oss-120b</td><td>1</td><td>+1.65</td><td>1</td><td>0.486</td></tr><tr><td>gpt-oss-120b/B</td><td>unstated</td><td>0.60</td><td>gpt-oss-120b</td><td>1</td><td>+1.70</td><td>1</td><td>1.000</td></tr><tr><td>gpt-oss-120b/C</td><td>unstated</td><td>0.60</td><td>gpt-oss-120b</td><td>1</td><td>+1.96</td><td>1</td><td>0.921</td></tr><tr><td>gpt-oss-120b/D</td><td>fp4</td><td>0.50</td><td>gpt-oss-120b</td><td>1</td><td>+1.89</td><td>1</td><td>0.478</td></tr><tr><td>gpt-oss-120b/E</td><td>fp16</td><td>0.75</td><td>gpt-oss-120b</td><td>1</td><td>+1.45</td><td>1</td><td>0.139</td></tr><tr><td>gpt-oss-120b/F</td><td>fp4</td><td>0.17</td><td>gpt-oss-120b</td><td>1</td><td>+1.73</td><td>1</td><td>0.176</td></tr><tr><td>gpt-oss-120b/G</td><td>bf16</td><td>0.25</td><td>gpt-oss-120b</td><td>1</td><td>+1.87</td><td>1</td><td>0.478</td></tr><tr><td>gpt-oss-120b/H</td><td>bf16</td><td>0.17</td><td>gpt-oss-120b</td><td>1</td><td>+1.59</td><td>1</td><td>0.322</td></tr><tr><td>gpt-oss-120b/I</td><td>bf16</td><td>0.60</td><td>gpt-oss-120b</td><td>1</td><td>+2.00</td><td>1</td><td>0.059</td></tr><tr><td>gpt-oss-120b/J gpt-oss-120b/K</td><td>bf16 unstated</td><td>0.18 0.42</td><td>gpt-oss-120b gpt-oss-120b</td><td>1 1</td><td>+1.73 +2.07</td><td>1 1</td><td>1.000 0.754</td></tr><tr></tr><tr><td>gpt-oss-120b/P</td><td>fp4</td><td>0.60</td><td>gpt-oss-120b</td><td>1</td><td>+1.78</td><td>1</td><td>0.444</td></tr><tr><td>gpt-oss-120b/Q</td><td>fp4</td><td>0.25</td><td>gpt-oss-120b</td><td>1</td><td>+1.24</td><td>1</td><td>0.716</td></tr><tr><td>gpt-oss-120b/R</td><td>fp4</td><td>0.75</td><td>gpt-oss-120b</td><td>1</td><td>+1.95</td><td>1</td><td>1.000</td></tr><tr><td>gpt-oss-120b/S</td><td>unstated</td><td>0.60</td><td>gpt-oss-120b</td><td>1</td><td>+0.96</td><td>1</td><td>0.333</td></tr><tr><td>gpt-oss-120b/T</td><td>unstated</td><td>0.95</td><td>gpt-oss-120b</td><td>1</td><td>+1.85</td><td>1</td><td>1.000</td></tr><tr><td>gpt-oss-120b/U</td><td>fp8</td><td>0.60</td><td>gpt-oss-120b</td><td>1</td><td>+1.59</td><td>1</td><td>0.765</td></tr><tr><td>gpt-oss-120b/V</td><td>unstated</td><td>0.60</td><td>gpt-oss-120b</td><td>1</td><td>+1.84</td><td>1</td><td>0.145</td></tr><tr><td>kimi-k2/A</td><td>fp8</td><td>2.30</td><td>kimi-k2</td><td>1</td><td>+1.05</td><td></td><td></td></tr><tr><td>kimi-k2.5/A</td><td>unstated</td><td>3.00</td><td>kimi-k2</td><td>2</td><td>-0.90</td><td>2</td><td>0.006</td></tr><tr><td>kimi-k2.5/B</td><td>int4</td><td>2.50</td><td>kimi-k2.5</td><td>1</td><td>+1.09</td><td>1</td><td>1.000</td></tr><tr><td>kimi-k2.5/C</td><td>unstated</td><td>2.85</td><td>kimi-k2.5</td><td>1</td><td>+1.11</td><td>1</td><td>0.938</td></tr><tr><td>kimi-k2.5/D</td><td>int4</td><td>2.25</td><td>kimi-k2.5</td><td>1</td><td>+1.64</td><td>1</td><td>0.119</td></tr><tr><td>kimi-k2.5/E</td><td>unstated</td><td>3.32</td><td>kimi-k2.5</td><td>1</td><td>+1.12</td><td>1</td><td>0.767</td></tr><tr><td>llama-3.3-70b-instruct/A</td><td>fp8</td><td>0.52</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.58</td><td>1</td><td>0.458</td></tr><tr><td>llama-3.3-70b-instruct/B</td><td>fp8</td><td>2.25</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.70</td><td>1</td><td>1.000</td></tr><tr><td>llama-3.3-70b-instruct/C</td><td>fp16</td><td>0.71</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.44</td><td>1</td><td>0.754</td></tr><tr><td>llama-3.3-70b-instruct/D</td><td>fp8</td><td>0.32</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.11</td><td>1</td><td>0.290</td></tr><tr><td>llama-3.3-70b-instruct/E</td><td>unstated</td><td>0.79</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.01</td><td>1</td><td>0.021</td></tr><tr><td>llama-3.3-70b-instruct/F</td><td>bf16</td><td>0.40</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.57</td><td>1</td><td>0.333</td></tr><tr><td>llama-3.3-70b-instruct/G</td><td>fp8</td><td>0.50</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.52</td><td>1</td><td>1.000</td></tr><tr><td>llama-3.3-70b-instruct/H</td><td>unstated</td><td>0.90</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.05</td><td>1</td><td>0.145</td></tr><tr><td>llama-3.3-70b-instruct/I</td><td>unstated</td><td>1.04</td><td>llama-3.3-70b-instruct</td><td>1</td><td>+3.65</td><td>1</td><td>0.059</td></tr><tr><td>llama-4-maverick/A</td><td>unstated</td><td>0.65</td><td>llama-4-maverick</td><td>1</td><td>+4.25</td><td>1</td><td>0.006</td></tr><tr><td>llama-4-maverick/B</td><td>fp8</td><td>0.85</td><td>gemma-4-31b-it</td><td>91</td><td>-7.01</td><td>92</td><td>0.006</td></tr><tr><td>llama-4-maverick/C</td><td>fp8</td><td>1.00</td><td>llama-4-maverick</td><td>1</td><td>+4.41</td><td>1</td><td>0.006</td></tr><tr><td>llama-4-scout/A</td><td>fp8</td><td>0.30</td><td>llama-4-scout</td><td>1</td><td>+4.36</td><td>1</td><td>0.333</td></tr><tr><td>llama-4-scout/B</td><td>bf16</td><td>0.59</td><td>llama-4-scout</td><td>1</td><td>+4.36</td><td>1</td><td>0.179</td></tr><tr><td>mistral-small-3.2-24b-instruct/A</td><td>fp8</td><td>0.20</td><td>mistral-small-3.2-24b-instruct</td><td>1</td><td>+1.61</td><td>1</td><td>0.700</td></tr><tr><td>mistral-small-3.2-24b-instruct/B</td><td>unstated</td><td>0.30</td><td>mistral-small-2603</td><td>5</td><td>-2.37</td><td>10</td><td>0.006</td></tr><tr><td>mistral-small-3.2-24b-instruct/C</td><td>bf16</td><td>0.30</td><td>mistral-small-3.2-24b-instruct</td><td>1</td><td>+0.80</td><td>1</td><td>0.672</td></tr><tr><td>mistral-small-3.2-24b-instruct/D</td><td>fp8</td><td>0.25</td><td>mistral-small-3.2-24b-instruct</td><td>1</td><td>+1.61</td><td>1</td><td>0.059</td></tr><tr><td>nemotron-3-super-120b-a12b/A</td><td>bf16</td><td>0.40</td><td>nemotron-3-super-120b-a12b</td><td>1</td><td>+0.89</td><td>1</td><td>0.399</td></tr><tr><td>nemotron-3-super-120b-a12b/B</td><td>fp8</td><td>0.45</td><td>nemotron-3-super-120b-a12b</td><td>1</td><td>+1.33</td><td>1</td><td>0.345</td></tr><tr><td>qwen3-235b-a22b-2507/A</td><td>unstated</td><td>0.60</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+1.98</td><td>1</td><td>0.006</td></tr><tr><td>qwen3-235b-a22b-2507/B</td><td>fp8</td><td>0.55</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+2.06</td><td>1</td><td>0.011</td></tr><tr><td>qwen3-235b-a22b-2507/C</td><td>fp8</td><td>0.35</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+2.03</td><td>1</td><td>0.754</td></tr><tr><td>qwen3-235b-a22b-2507/D</td><td>unstated</td><td>1.00</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+1.79</td><td>1</td><td>0.091</td></tr><tr><td>qwen3-235b-a22b-2507/E</td><td>fp8</td><td>0.60</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+1.94</td><td>1</td><td>0.439</td></tr><tr><td>qwen3-235b-a22b-2507/F</td><td>fp8</td><td>0.58</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+1.83</td><td></td><td>0.478</td></tr><tr><td>qwen3-235b-a22b-2507/G</td><td>fp8</td><td>0.80</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+1.93</td><td>1</td><td>0.439</td></tr><tr><td>qwen3-235b-a22b-2507/H</td><td>unstated</td><td>0.84</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+1.91</td><td>1</td><td>0.489</td></tr><tr><td>qwen3-235b-a22b-2507/I</td><td>fp8</td><td>0.75</td><td>qwen3-235b-a22b-2507</td><td>1</td><td>+2.05</td><td>1</td><td>0.979</td></tr></table>