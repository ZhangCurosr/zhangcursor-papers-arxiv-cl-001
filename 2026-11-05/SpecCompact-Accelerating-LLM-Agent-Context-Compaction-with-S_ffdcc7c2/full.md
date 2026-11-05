# SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification

Yulu An\* New York University ya2191@nyu.edu

Yifei Feng\* New York University yf3005@nyu.edu

<sup>§</sup> SpecCompact

Xiangyang Yin\*<sup>†</sup> New York University shawn.yin@nyu.edu

## Abstract

LLM agents use context compaction to manage growing interaction histories, but summarizing long histories with large models can introduce substantial delays. Compaction therefore requires balancing information retention against latency. We observe that structured summary slots can retain information when generated independently. Based on this observation, we present SpecCompact, a framework that accelerates compaction through slot-parallel semantic speculation. SpecCompact fine-tunes a small drafter and generates slots in parallel. The target model verifies each proposed line and repairs rejected content, while subsequent drafting uses the updated slot prefix. System optimizations further reduce repeated context processing and model calls. Across AppWorld and τ<sup>3</sup> benchmarks with two target models, SpecCompact achieves 2.04–5.63× compaction speedups over Hermes-style summarization. Separate downstream evaluations reveal model- and task-dependent performance trade-offs. A user study within Hermes further shows an 8.68× median compaction speedup.

## 1 Introduction

Large language model (LLM) agents now handle a growing range of tasks, especially personal work and coding (Meta, 2026; OpenAI, 2026; Anthropic, 2025; OpenAI, 2025). While longer context windows have enabled better long-horizon performance, they also allow agents to accumulate larger histories of user requests, tool outputs, and intermediate decisions. Agent systems typically use context compaction to shorten these histories while retaining information needed for later actions (OpenClaw, 2026; Nous Research, 2026). However, compaction itself requires processing a long context and decoding a summary token by token. The resulting delay can significantly degrade the user experience. Compaction therefore requires balancing information retention against the latency of producing the summary.

![](images/dc4c439b757c31bd7a80d952e004b6c7c0409f038e3077ea1cb8bb9da9df65cc.jpg)  
Figure 1: Task success and compaction speedup with Gemma4-31B-IT, measured in separate evaluations.

Token-level speculative decoding, a general method for accelerating decoding, can speed up this compaction process. Specifically, a smaller model with lower inference costs can accelerate generation while preserving the target model’s output distribution by target model verification (Leviathan et al., 2023; Chen et al., 2023). However, valid summaries can differ in wording yet have the same effect on later actions. Thus, although speculative decoding preserves lossless compaction behavior, it may not fully exploit this flexibility. Recent semantic speculation methods use target-model judgments to accept sentence-level drafts (Pan et al., 2025; Hu et al., 2026), making them better suited to compaction. However, applying these methods to a single summary still requires sequential generation, which can be slow for long contexts. Moreover, asynchronous compaction overlaps summarization with continued agent execution, while parallel compaction summarizes input blocks concurrently (Chen et al., 2026; Cim et al., 2026). These approaches raise a further question: can the structure ofthe summary itselfguide parallel generation and target-model verification?

![](images/eb411da68bc090bc36fbf0cfb74991d2d46fae62a903f2923d72bda0383b4db2.jpg)

![](images/29d201c1a8be420143a9a0e2c394ee3d42d88d3eb56c13be819f2380eecc9222.jpg)  
(b) Independent generation  
Figure 2: Slot structure and independent generation. (a) Fields in OpenClaw and Hermes compaction templates. (b) Reference adequacy over 100 trajectories on LiveClawBench (Long et al., 2026), with six slots per trajectory.

Existing compaction templates provide a useful starting point. Agent frameworks such as Open-Claw and Hermes (OpenClaw, 2026; Nous Research, 2026) organize summaries into fields such as goals, constraints, progress, decisions, and next steps. We call these fields slots. Each slot describes one aspect of the task state and can be generated directly from the shared history. In our preliminary study, slots generated without using other slots outputs still preserve the necessary information, showing the potential for independent slot generation. Furthermore, reversing the slot generation order has little effect on the retained information. These findings suggest that slots can serve as units for independent and parallel generation, with verification and repair performed within each slot.

We introduce SpecCompact, a framework for accelerating agent context compaction through slotparallel semantic speculation. A fine-tuned small model drafts each slot from the shared history and the content already accepted in that slot. The target model checks whether each proposed line is supported by the history and consistent with the accepted content. It accepts the line or generates a local replacement. These draft–verify–repair loops run in parallel, and their outputs are combined into a structured summary. We also introduce systemlevel optimizations that reduce execution overhead and further accelerate compaction.

Our contributions are as follows:

• We show that summary slots can retain relevant information without conditioning on other slots’ outputs, supporting their use as units of parallel compaction.

• We introduce SpecCompact, which combines training, parallel drafting, and line-wise semantic verification with local repair. We further optimize its execution to reduce repeated context processing and model calls.

• We demonstrate 2.04–5.63× compaction speedups over Hermes-style summarization, alongside downstream task evaluations and a native Hermes user study.

## 2 Background and Motivation

## 2.1 Background

Speculative decoding uses a small draft model to propose several tokens, which a larger target model verifies in a single forward pass (Leviathan et al., 2023; Chen et al., 2023). The accepted prefix is kept, and the first rejected token is replaced before drafting resumes. This allows each target-model call to produce multiple tokens while remaining lossless relative to target-only decoding.

Semantic speculation relaxes the lossless requirement by evaluating draft content based on its meaning. Related methods such as SpecReasons (Pan et al., 2025) and DREAM-R (Hu et al., 2026) apply this idea to reasoning steps for a lossy but much faster speculation inference. Like speculative decoding, these methods use a small model to draft a token sequence. Instead of verifying individual tokens, the target model evaluates the sequence as a complete reasoning step and outputs an accept-orreject decision. In context compaction, summaries with different wording can preserve the same information needed for later actions. We therefore adopt semantic speculation for compaction, judging drafts by whether they preserve the information needed for later actions.

## 2.2 Motivation

Agent frameworks already use structured templates for context compaction. As shown in Figure 2(a), OpenClaw and Hermes organize summaries into fields (OpenClaw, 2026; Nous Research, 2026). In our context, we call these fields slots. Each slot describes one aspect of the task state, such as the goal or current progress. Slots provide a natural unit for generation, but generating all slots as one sequence still requires later slots to wait for earlier ones. We therefore investigate whether this sequential dependency is necessary.

We compare monolithic generation with independent slot generation on 100 trajectories. Monolithic generation produces all six slots in a single output, while independent generation produces each slot from the same history without access to other slots outputs. We use an LLM judge to evaluate how well each slot preserves relevant information from the full history, with a score from 0 to 5 obtained by summing five binary criteria. As shown in Figure 2(b), independent generation achieves a mean score of 4.73, compared with 4.44 for monolithic generation. The fraction of slots scoring at least 4 is 93% and 85%, respectively. Reversing the slot order for monolithic generation or the invocation order for independent generation changes the mean score by at most 0.03. These results suggest that the information needed for a slot can often be recovered from the history without conditioning on other slots. This motivates generating slots in parallel from the shared history.

Observation. Slots generated without other slots’ outputs can still preserve relevant information. This supports generating slots in parallel from the shared history.

## 3 SpecCompact

Figure 3 provides an overview of SpecCompact. We first fine tune a small drafter on target-generated slot content. During compaction, the drafter generates slots in parallel from the shared history. The target model verifies each proposed line against the history and the accepted slot prefix, repairing rejected lines. Each slot continues from its updated prefix, and the slots are assembled into the compacted context once all are complete.

![](images/01461aa15f8c2be73fe7c797b1888584d1f1a71de5f895592f794885983f4243.jpg)  
Figure 3: Overview of SpecCompact.

## 3.1 Slot-Conditioned Drafting

Based on our observation, we use slots as units of parallel generation. We follow OpenClaw’s six-slot summary schema (OpenClaw, 2026): Goal, Constraints and Preferences, Progress, Key Decisions, Next Steps, and Critical Context. Each slot has an instruction specifying its content and output format. Each request includes the shared conversation history and the full schema, but asks the drafter to generate only the requested slot. Figure 5 provides the prompt template and slot contracts.

With the slot structure defined, we next fine-tune the drafter for slot-conditioned generation. We first use the target model with the generation prompt to produce supervised training data. We choose an independent small model as the drafter, using Qwen3-0.6B (Yang et al., 2025) in our experiments. Unlike vanilla speculative decoding, semantic speculation does not require the drafter and target to share a vocabulary. We align the drafter with the target’s slot generation through supervised fine-tuning (SFT). For each training example, we minimize the cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \sum _ { j = 1 } ^ { L _ { s } } \log q _ { \theta } ( y _ { s , j } \mid H , c _ { s } , y _ { s , < j } ) ,\tag{1}
$$

where $q _ { \theta }$ is the drafter, H is the history, $c _ { s }$ is the slot instruction, and $y _ { s }$ is the target-generated slot content represented as $L _ { s }$ tokens under the drafter’s tokenizer. Other objectives, such as KL divergence between aligned output distributions, could also be used. We use text-only supervision to avoid storing target probability distributions. We construct one training example for each history-slot pair, so a dataset of T histories yields 6T SFT examples.

During compaction, we run a separate drafting stream for each slot. All streams use the same finetuned drafter and shared history, but maintain their own slot instructions and accepted prefixes. While SFT uses complete slot outputs, online drafting proceeds through line-level proposals. The drafter generates tokens autoregressively and submits each proposed line to the target model for verification.

For slot s, let $A _ { s , t - 1 }$ denote the accepted prefix before step t, with $A _ { s , 0 }$ initially empty. This prefix contains the lines already accepted or generated as replacements by the target model. Given the history H and slot instruction $c _ { s } .$ , the drafter $q _ { \theta }$ proposes the next line:

$$
\tilde { \ell } _ { s , t } \sim q _ { \theta } ( \cdot \mid H , c _ { s } , A _ { s , t - 1 } ) .\tag{2}
$$

The proposed line is added to the prefix only after verification. If accepted, it is appended unchanged. Otherwise, the target-generated replacement is appended instead. The updated prefix then conditions subsequent drafting, so the drafter continues from the content retained in the summary, including any target repairs. The history and slot instruction remain fixed throughout this process. Each slot depends only on its own prefix, allowing the six drafting streams to run in parallel.

## 3.2 Semantic Verification

As in DREAM-R (Hu et al., 2026), we use binary semantic verification. For each proposed line $\tilde { \ell } _ { s , t }$ , the target model checks whether its content is supported by the history H and consistent with the accepted slot prefix $A _ { s , t - 1 }$ . The verification prompt includes the history, slot instruction, accepted prefix, and proposed line. It asks the target to return positive when both conditions hold, and negative otherwise. Figure 6 provides the verification prompt. We reduce the cost of repeatedly processing the history through shared-prefix caching, as discussed in the Section 3.3.

The target generates one judgment token, from which we obtain the top-k token probabilities. Let $\hat { p } _ { s , t } ^ { + }$ and $\hat { p } _ { s , t } ^ { - }$ denote the probability estimates for positive and negative. We compute the normal-

ized acceptance score as

$$
r _ { s , t } = \frac { \hat { p } _ { s , t } ^ { + } } { \hat { p } _ { s , t } ^ { + } + \hat { p } _ { s , t } ^ { - } } ,\tag{3}
$$

and accept the line when $r _ { s , t } \ge \eta$ . Although we define only two labels, positive and negative, each may have multiple token forms in the target tokenizer. For example, positive may start with a capital letter or include a leading space. For each label, we use the highest probability among its returned token variants. If a label is missing from the returned top-k tokens, we first compute the remaining probability mass as $\textstyle 1 - \sum _ { i = 1 } ^ { k } p _ { i }$ , where $p _ { i }$ is the probability of the i-th returned token. We then compare this value with the smallest returned token probability and use the smaller value as the substitute probability.

When a line is rejected, the target generates a replacement conditioned on the same history, slot instruction, and accepted prefix. The rejected line is excluded from this input. Given an accepted draft line or a valid target replacement, we update the slot prefix as

$$
A _ { s , t } = \left\{ \begin{array} { l l } { A _ { s , t - 1 } \oplus \tilde { \ell } _ { s , t } , } & { \mathrm { i f ~ a c c e p t e d } , } \\ { A _ { s , t - 1 } \oplus \ell _ { s , t } ^ { \mathrm { T } } , } & { \mathrm { i f ~ r e j e c t e d } , } \end{array} \right.\tag{4}
$$

where $\ell _ { s , t } ^ { \mathrm { T } }$ is the target-generated replacement and $\oplus$ denotes concatenation. Drafting then continues from the updated prefix without regenerating previously accepted content.

## 3.3 Efficient Slot-Parallel Execution

Generating slots in parallel does not by itself make line-wise speculation efficient. Each slot repeatedly invokes the drafter and target, and these calls include the same long history. Without reuse, even a one-token verification response requires another expensive prefill. Requesting each draft line separately also adds model-call overhead, while repair generation may decode additional lines that are never used. We address these costs through the following optimizations.

Independent slot scheduling. We run the six slot streams concurrently, each with its own accepted prefix and sequential proposal, verification, and repair loop. While one slot waits for target verification or repair, other slots can continue drafting or submit their own verification requests. This allows work on the drafter and target to overlap and enables server-side batching across slots, reducing serial waiting. After all streams finish, we assemble their outputs in schema order.

![](images/edfa2fd95eb919dd53efd14b76f8d21b1e191b0d00c544426caac557a441a205.jpg)  
Figure 4: Execution optimizations in SpecCompact: independent slot scheduling, shared-prefix caching, draft buffering, line-aware stopping, and lightweight checks with termination rules.

Shared-prefix caching. Each verification or repair call includes the full conversation history. Repeatedly prefilling this long input can be expensive even when the target generates only a short response. We then organize target requests with a common prefix containing the history and all six slot contracts. Slot instructions and the accepted prefix follow this shared portion, with the proposed line also included for verification. This keeps the leading tokens unchanged across slots and steps.

Draft buffering. Calling the drafter separately for every line adds request overhead. We therefore let the drafter generate multiple lines in a single call. We send the first line to the target for verification and store the remaining complete lines in a buffer. If a line is accepted unchanged, the next buffered line can be checked without another draft call. Reuse requires the prefix used to generate that line to exactly match the current accepted prefix $A _ { s , t - 1 }$ . If a target repair changes the prefix, we discard the remaining buffered lines and generate a new continuation from the repaired prefix. Each buffered proposal undergoes the same line-wise checks before acceptance.

Line-aware stopping. Each repair operation after rejection replaces one rejected line. However, the target may continue generating additional lines that are not used. We configure the target server to stop generation at the next line break, subject to the token limit, avoiding computation on these unused lines. To detect line breaks, we include all tokens ending in a newline in the stop set. Some of these tokens also contain text, so we retain that text when stopping. We preserve line and paragraph breaks when constructing the next input and keep any final content generated before end-of-sequence. The tokenizer and stop-token set are reused across calls to avoid repeated setup.

Checks and termination. In practice, the drafter sometimes produces repeated or abnormally long lines. We check whether a proposed line duplicates an accepted line or exceeds a configured character limit. Such lines go directly to target repair, avoiding an additional semantic verification call before replacement. We also cap the tokens generated per request and the lines and tokens retained per slot, preventing repetitive generation from continuing indefinitely. A slot stops when its budget is reached or a proposed addition would exceed it. Target replacements have the same checks and budget validation without further semantic verification. If a replacement is empty or fails these checks, we stop the slot and retain its accepted prefix. Structured summaries may produce standalone headings. After the length and duplicate checks, we directly accept lines beginning with # and recognized empty-slot markers such as (none), avoiding semantic verification calls for these standard output forms. Blank lines are retained as formatting. After assembly, we check the total summary token budget and raise an error if it is exceeded.

## 4 Experiments

We evaluate SpecCompact with Gemma4-31B-IT (Team et al., 2026) and Qwen3.5-122B-A10B (Qwen Team, 2026) as target models, each paired with a separately fine-tuned Qwen3- 0.6B drafter (Yang et al., 2025). We use App-World (Trivedi et al., 2024) and τ<sup>3</sup> (Shi et al., 2026) to evaluate whether compacted histories preserve the information needed for subsequent agent actions. AppWorld tests task completion through code and application APIs, while $\tau ^ { 3 }$ tests task completion through user interaction and tool use. We evaluate on the full AppWorld Test-Normal and Test-Challenge sets, and run three trials per task on an 80-task random selected subset of $\tau ^ { 3 }$ for each target model. We report task goal completion (TGC) and scenario goal completion (SGC) on AppWorld, and task success rate on $\tau ^ { 3 }$ . Beyond agent tasks, we also evaluate long-context compaction on LongBench (Bai et al., 2023) and RULER (Hsieh et al., 2024). For LongBench, we use the GovReport and QMSum subsets and report ROUGE-L. For RULER, we evaluate full set of five task types at context lengths of 8k, and 16k using the official task-specific matching metrics. To measure compaction latency and throughput, we construct smaller subsets from AppWorld, $\tau ^ { 3 }$ , and LongBench instead of using the complete sets.

Table 1: Agent task performance (%). Full History is a reference without compaction. Bold marks the best compaction result. Gray subscripts show 95% confidence intervals.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="2">AppWorld Normal</td><td colspan="2">AppWorld Challenge</td><td> $\tau ^ { 3 }$ </td></tr><tr><td>TGC ↑</td><td>SGC ↑</td><td>TGC ↑</td><td>SGC ↑</td><td>Success ↑</td></tr><tr><td rowspan="4">Gemma4- 31B-IT</td><td>Full History</td><td> $\lvert 6 2 . 5 0 _ { [ 5 1 . 7 9 , 7 2 . 6 2 ] }$ </td><td> $4 6 . 4 3 _ { [ 3 3 . 9 3 , 5 8 . 9 3 ] }$ </td><td> $5 3 . 2 4 _ { [ 4 5 . 8 0 , 6 0 . 4 3 ] }$ </td><td>41.73[33.81,49.64]</td><td>52.08[45.00,58.75]</td></tr><tr><td>OpenClaw</td><td> $\pmb { 5 8 . 9 3 } _ { [ 4 7 . 6 2 , 6 9 . 6 4 ] }$ </td><td>44.64[32.14,57.14]</td><td> $4 6 . 5 2 _ { [ 3 9 . 0 9 , 5 3 . 7 2 ] }$ </td><td> $3 3 . 0 9 _ { [ 2 5 . 1 8 , 4 1 . 0 1 ] }$ </td><td> $4 7 . 5 0 _ { [ 3 9 . 1 7 , 5 5 . 4 2 ] }$ </td></tr><tr><td>Hermes</td><td> $5 7 . 7 4 _ { [ 4 7 . 0 2 , 6 7 . 8 6 ] }$ </td><td>39.29[26.79,51.79]</td><td>47.24[40.05,54.44]</td><td> $3 2 . 3 7 _ { [ 2 5 . 1 8 , 4 0 . 2 9 ] }$ </td><td> $\mathbf { 5 0 . 0 0 } _ { [ 4 2 . 0 8 , 5 7 . 9 2 ] }$ </td></tr><tr><td>SpecCompact</td><td> $5 4 . 7 6 _ { [ 4 4 . 0 5 , 6 5 . 4 8 ] }$ </td><td> $3 7 . 5 0 _ { [ 2 5 . 0 0 , 5 0 . 0 0 ] }$ </td><td> $4 3 . 1 7 _ { [ 3 5 . 9 7 , 5 0 . 1 2 ] }$ </td><td>28.06[20.86,35.25]</td><td> $4 7 . 5 0 _ { [ 4 0 . 0 0 , 5 5 . 0 0 ] }$ </td></tr><tr><td rowspan="4"></td><td>Full History</td><td> $4 0 . 4 8 _ { [ 2 9 . 7 6 , 5 1 . 1 9 ] }$ </td><td> $2 6 . 7 9 _ { [ 1 6 . 0 7 , 3 7 . 5 0 ] }$ </td><td> $2 0 . 6 2 _ { [ 1 5 . 3 5 , 2 6 . 1 4 ] }$ </td><td> $8 . 6 3 _ { [ 4 . 3 2 , 1 3 . 6 7 ] }$ </td><td> $\lvert 6 7 . 9 2 _ { [ 6 3 . 3 3 , 7 2 . 0 8 ] }$ </td></tr><tr><td>Qwen3.5- OpenClaw</td><td> $2 8 . 5 7 _ { [ 1 9 . 0 5 , 3 8 . 6 9 ] }$ </td><td> $1 4 . 2 9 _ { [ 5 . 3 6 , 2 3 . 2 1 ] }$ </td><td> $1 7 . 7 5 _ { [ 1 2 . 7 1 , 2 3 . 2 6 ] }$ </td><td> $\pmb { 8 . 6 3 } _ { [ 4 . 3 2 , 1 3 . 6 7 ] }$ </td><td> $\mathbf { 6 5 . 8 3 } _ { [ 6 0 . 8 3 , 7 0 . 8 3 ] }$ </td></tr><tr><td>122B-A10B Hermes</td><td> $2 8 . 5 7 _ { [ 1 9 . 0 5 , 3 8 . 6 9 ] }$ </td><td> $1 4 . 2 9 _ { [ 5 . 3 6 , 2 3 . 2 1 ] }$ </td><td> $1 4 . 3 9 _ { [ 9 . 5 9 , 1 9 . 4 2 ] }$ </td><td> $7 . 1 9 _ { [ 2 . 8 8 , 1 1 . 5 1 ] }$ </td><td> $6 3 . 7 5 _ { [ 5 8 . 3 3 , 6 8 . 7 5 ] }$ </td></tr><tr><td>SpecCompact</td><td> $3 3 . 3 3 _ { [ 2 3 . 2 1 , 4 4 . 0 5 ] }$ </td><td> $2 1 . 4 3 _ { [ 1 0 . 7 1 , 3 2 . 1 4 ] }$ </td><td> $1 6 . 0 7 _ { [ 1 1 . 2 7 , 2 1 . 1 0 ] }$ </td><td> $6 . 4 7 _ { [ 2 . 8 8 , 1 0 . 7 9 ] }$ </td><td> $6 4 . 5 8 _ { [ 5 9 . 1 7 , 6 9 . 5 8 ] }$ </td></tr></table>

We construct about 50K training data from AppWorld, $\tau ^ { 3 }$ , and LiveClawBench (Long et al., 2026), using each target model to generate slotlevel supervision. Training and validation examples are separated by trajectory ID. We fine-tune each drafter for one epoch with LoRA (Hu et al., 2021) applied to all linear layers, using rank 16, scaling factor 32, and zero dropout. Training uses AdamW (Loshchilov and Hutter, 2019), with a learning rate of $1 0 ^ { - 4 } .$ , cosine decay, a warmup ratio of 0.05, zero weight decay, and gradient clipping at 1.0. We use a per-device batch size of 2 with 16 gradient accumulation steps, BF16 precision, a maximum sequence length of 16,384 tokens, and random seed 42. The cross-entropy loss is computed only over the target slot content.

During agent evaluation, we compact the interaction history every eight agent model calls, retaining system and setup messages and the current input. New messages accumulate until the next compaction. For latency experiments, we serve the models with vLLM (Kwon et al., 2023) on NVIDIA RTX PRO 6000 Blackwell GPUs. Gemma4 runs in BF16 on one GPU, while Qwen3.5 runs in FP8 on two GPUs. SpecCompact uses one additional GPU for the drafter and runs six slots concurrently with an acceptance threshold of $\eta = 0 . 7 .$ . Drafting and repair allow up to 64 new tokens per request, while verification uses one token. Each slot is limited to 12 lines and 320 tokens. The summary budget is 2,048 tokens for SpecCompact and the baselines.

## 4.1 Agent Task Performance

Table 1 reports downstream task performance on AppWorld and $\tau ^ { 3 }$ . The effect of SpecCompact varies across target models. Compared with Hermes, SpecCompact reduces Gemma4’s TGC by 2.98 and 4.07 percentage points on AppWorld Normal and Challenge, respectively, and reduces $\tau ^ { 3 }$ success by 2.50 points. For Qwen3.5, SpecCompact increases TGC by 4.76 and 1.68 points on the two AppWorld splits and increases $\tau ^ { 3 }$ success by 0.83 points, although Challenge SGC decreases by 0.72 points. SpecCompact therefore offers faster compaction with a model-dependent effect on downstream performance.

## 4.2 Long-Context Performance

Table 2 evaluates compaction on LongBench and RULER at 8k and 16k. Target-self uses the target model to generate each slot directly, without a drafter or semantic verification. On Qwen3.5, Spec-Compact scores within 0.6 points of Target-self on both LongBench datasets and achieves higher scores at both RULER lengths. On Gemma4, Spec-

Table 2: Long-context performance on LongBench (ROUGE-L) and RULER at 8k and 16k (official matching scores), scaled to 100. Execution failures are scored as zero. Bold marks the best compaction result.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="2">LongBench</td><td colspan="2">RULER</td></tr><tr><td>GovReport ↑</td><td>QMSum ↑</td><td>8k↑</td><td>16k↑</td></tr><tr><td rowspan="3">Gemma4- 31B-IT</td><td>Full History</td><td>28.36[27.73,29.02]</td><td>24.34[23.28,25.41]</td><td>91.72 [88.77,94.62]</td><td>92.12[89.22,95.02]</td></tr><tr><td>Target-self</td><td>25.05[24.51,25.59]</td><td>18.35[17.61,19.10]</td><td>71.00[67.68,74.67]</td><td>71.02[67.72,74.70]</td></tr><tr><td>SpecCompact</td><td>22.28[21.10,23.37]</td><td>17.52[16.51,18.52]</td><td>66.50[63.67,69.70]</td><td>68.55[65.32,71.87]</td></tr><tr><td rowspan="3">Qwen3.5- 122B-A10B</td><td>Full History</td><td>31.16[30.53,31.82]</td><td>22.98[22.03,23.96]</td><td>90.92[87.63,93.82]</td><td>90.92[87.63,94.17]</td></tr><tr><td>Target-self</td><td>28.25[27.71,28.79]</td><td> $\pmb { 1 8 . 5 4 } _ { [ 1 7 . 8 6 , 1 9 . 2 5 ] }$ </td><td> $7 2 . 9 8 _ { [ 6 9 . 3 3 , 7 6 . 6 7 ] }$ </td><td>72.15[68.87,75.80]</td></tr><tr><td>SpecCompact</td><td> $2 7 . 6 6 _ { [ 2 7 . 0 5 , 2 8 . 2 6 ] }$ </td><td> $1 8 . 0 5 _ { [ 1 7 . 2 4 , 1 8 . 8 7 ] }$ </td><td> $7 4 . 2 3 _ { [ 7 0 . 5 5 , 7 7 . 9 3 ] }$ </td><td> $7 2 . 6 0 _ { [ 6 8 . 9 7 , 7 6 . 2 7 ] }$ </td></tr></table>

Compact scores 0.83–4.50 points below Target-self across the four settings.

## 4.3 Compaction Efficiency

Table 3 reports compaction latency on matched inputs. Relative to Hermes, SpecCompact achieves 4.28–5.63× speedups on the agent workloads with Gemma4 and 2.04–2.41× with Qwen3.5. On LongBench inputs of approximately 16k tokens, the speedups are 2.49× and 1.36×, respectively. Three factors help explain the smaller speedups with Qwen3.5. First, Qwen3.5’s baseline summarization is already faster. Second, the larger model-size gap makes it harder for the 0.6B drafter to match Qwen3.5’s behavior, with a draft acceptance rate 19.6 percentage points lower than with Gemma4. Third, our vLLM deployment provides limited prefix-cache reuse for Qwen3.5’s hybrid attention. Standard attention can reuse cached prefix KV states, whereas Qwen3.5’s linear-attention layers also need to restore recurrent states from checkpoints saved at suitable positions. Matching text prefixes alone therefore do not guarantee cache reuse. SpecCompact also outperforms OpenClaw in seven of the eight settings. The exception is Qwen3.5 on $\tau ^ { 3 } .$ , where SpecCompact takes 8.71 seconds compared with OpenClaw’s 7.68 seconds.

## 4.4 Effect of System Optimizations

Table 4 compares SpecCompact before and after the system optimizations. Both implementations already generate slots concurrently, allowing us to examine the additional gains from optimizing their execution. The optimizations reduce compaction latency by 1.88–2.44× for Gemma4 and 1.19–1.51× for Qwen3.5. For example, Gemma4’s latency on AppWorld Normal falls from 20.17 to 8.26 seconds. These gains highlight the value of reducing repeated context processing, model calls, and unnecessary decoding beyond slot parallelism alone. The table measures the combined effect of the optimizations rather than the contribution of each component. The full task evaluation uses the original implementation and has not been repeated with all system optimizations.

## 4.5 User Study

To evaluate user experience in a real agent setting, we integrate SpecCompact into the Hermes codebase and deploy the system with Gemma4 on four NVIDIA RTX PRO 6000 GPUs. We construct three scenarios from our own workflows, covering coding, math, and multitask question answering. For each scenario, we run Hermes in advance and pause at a fixed checkpoint to save a context snapshot. Users trigger compaction from the same snapshot with each method and select which feels faster without knowing the method identities. We also record the actual compaction time in the background. After compaction, we ask the agent to produce a summary and collect a separate blind preference between the two outputs.

According to the collected compaction data, SpecCompact reduces median compaction latency from 63.74 to 7.28 seconds, with a median paired speedup of 8.68×. These measurements show that the compaction speedups also carry over to the native agent deployment. Among 28 summarypreference responses, 11 (39.3%) favor ours, 8 (28.6%) favor Hermes, 8 (28.6%) rate them similarly, and 1 (3.6%) cannot judge.

## 5 Related Work

## 5.1 Agent Context Management

LLM agents manage growing histories through external memory and context compression. Memory systems store and retrieve past information (Packer et al., 2024; Xu et al., 2025), while prompt and KV-cache compression reduce prompt length or cache size (Jiang et al., 2023; Xu et al., 2023; Li et al., 2024). For long-running agents, context compaction summarizes interaction histories into a shorter working context. Existing methods improve compression guidelines (Kang et al., 2026), periodically summarize trajectories (Wu et al., 2026), or train agents to manage their context (Zhou et al., 2025; Lu et al., 2025; Sun et al., 2025; Ye et al., 2025). Other work improves how compaction is executed. Slipstream runs compaction alongside continued agent execution and uses subsequent steps to validate summaries (Chen et al., 2026). Parallel compaction summarizes input blocks concurrently (Cim et al., 2026). SpecCompact accelerates the generation and verification of structured summaries through slot-parallel semantic speculation.

Table 3: Compaction latency and speedup on 20 matched inputs per workload. Speedup is relative to Hermes. Bold marks the best result within each target. SpecCompact uses an additional drafter GPU.
<table><tr><td rowspan="3">Models</td><td rowspan="3">Methods</td><td colspan="4">AppWorld</td><td rowspan="2" colspan="2"> $\tau ^ { 3 }$ </td><td rowspan="2" colspan="2">LongBench 16k</td></tr><tr><td colspan="2">Normal</td><td colspan="2">Challenge</td></tr><tr><td>Time (s) ↓ Speedup ↑|</td><td></td><td>Time (s) ↓ Speedup ↑</td><td></td><td>Time (s) ↓ Speedup ↑</td><td></td><td>Time (s) ↓ Speedup ↑</td><td></td></tr><tr><td rowspan="3">Gemma4- 31B-IT</td><td>Hermes</td><td>38.89</td><td>1.00×</td><td>39.12</td><td>1.00×</td><td>36.21</td><td>1.00×</td><td>74.13</td><td>1.00×</td></tr><tr><td>OpenClaw</td><td>16.44</td><td>2.37×</td><td>18.70</td><td>2.09×</td><td>13.19</td><td>2.75×</td><td>63.55</td><td>1.17×</td></tr><tr><td>SpecCompact</td><td>8.26</td><td>4.71×</td><td>9.14</td><td>4.28×</td><td>6.43</td><td>5.63×</td><td>29.72</td><td>2.49×</td></tr><tr><td rowspan="3">Qwen3.5- 122B-A10B</td><td>Hermes</td><td>18.03</td><td>1.00×</td><td>16.98</td><td>1.00×</td><td>17.81</td><td>1.00×</td><td>29.43</td><td>1.00×</td></tr><tr><td>OpenClaw</td><td>7.87</td><td>2.29×</td><td>8.69</td><td>1.95×</td><td>7.68</td><td>2.32×</td><td>24.62</td><td>1.20×</td></tr><tr><td>SpecCompact</td><td>7.49</td><td>2.41×</td><td>7.55</td><td>2.25×</td><td>8.71</td><td>2.04×</td><td>21.67</td><td>1.36×</td></tr></table>

Table 4: Combined effect of system optimizations on 20 matched inputs per workload. Both versions use slot-parallel execution. Times are in seconds.
<table><tr><td>Benchmark</td><td>Before ↓</td><td>After ↓</td><td>Speedup ↑</td></tr><tr><td colspan="4">Gemma4-31B-IT</td></tr><tr><td>AppWorld Normal</td><td>20.17</td><td>8.26</td><td>2.44×</td></tr><tr><td rowspan="2">AppWorld Challenge  $\tau ^ { \bar { 3 } }$ </td><td>21.56</td><td>9.14</td><td>2.36×</td></tr><tr><td>12.07</td><td>6.43</td><td>1.88×</td></tr><tr><td colspan="4">Qwen3.5-122B-A10B</td></tr><tr><td>AppWorld Normal</td><td>9.55</td><td>7.49</td><td>1.28×</td></tr><tr><td rowspan="2">AppWorld Challenge  $\tau ^ { \bar { 3 } }$ </td><td>11.43</td><td>7.55</td><td>1.51×</td></tr><tr><td>10.38</td><td>8.71</td><td>1.19×</td></tr></table>

## 5.2 Speculative Decoding and Reasoning

Speculative decoding uses a small model to draft tokens and a target model to verify them in parallel, preserving the target model’s output distribution (Leviathan et al., 2023; Chen et al., 2023). Semantic speculation relaxes this requirement. SpecReason uses target-model judgments to verify complete reasoning steps (Pan et al., 2025). DREAM-R further aligns the drafter through reinforcement learning and parallelizes drafting and verification (Hu et al., 2026). This approach fits context compaction, where different summaries can preserve the information needed for later actions. SpecCompact adapts semantic speculation to compaction through parallel slot generation and line-wise verification.

## 6 Discussion and Limitations

Limitations. SpecCompact focuses on the tradeoff between accuracy and compaction speed, with an emphasis on the framework and its system implementation. It does not offer a consistent accuracy advantage over target-model summarization. We believe better drafter–target alignment, such as the reinforcement learning approach used in DREAM-R (Hu et al., 2026), could improve accuracy and provide a better trade-off.

Discussion. We also find that slots differ in where they obtain information from the history. Progress and Decisions use evidence spread throughout the context, while Goal, Constraints, and Next Steps mainly rely on beginning or end. This suggests slotconditioned sliding-window attention as a way to further accelerate drafting. We leave this direction to future work due to limited time and resources.

## 7 Conclusion

We presented SpecCompact for accelerating context compaction in LLM agents. SpecCompact combines slot-parallel drafting, line-wise semantic verification, and system optimizations to reduce compaction latency. Experiments across agent and long-context benchmarks demonstrate substantial speedups with accuracy trade-offs, while a native Hermes user study confirms the latency benefits.

## References

Anthropic. 2025. Claude 3.7 Sonnet and Claude Code.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. Preprint, arXiv:2308.14508.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating Large Language Model Decoding with Speculative Sampling. Preprint, arXiv:2302.01318.

Zhuofu Chen, Rui Pan, Yinwei Dai, and Ravi Netravali. 2026. Slipstream: Trajectory-Grounded Compaction Validation for Long-Horizon Agents. Preprint, arXiv:2605.08580.

Musa Cim, Burak Topcu, Chita Das, and Mahmut Taylan Kandemir. 2026. Parallel Context Compaction for Long-Horizon LLM Agent Serving. Preprint, arXiv:2605.23296.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? Preprint, arXiv:2404.06654.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Yunhai Hu, Zining Liu, Xiangyang Yin, Tianhua Xia, Bo Bao, Eric Sather, Vithursan Thangarasa, and Sai Qian Zhang. 2026. DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution. Preprint, arXiv:2605.28678.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. Preprint, arXiv:2310.05736.

Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, and Saravan Rajmohan. 2026. ACON: Optimizing Context Compression for Long-horizon LLM Agents. Preprint, arXiv:2510.00615.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. Preprint, arXiv:2309.06180.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. Preprint, arXiv:2211.17192.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. SnapKV: LLM Knows What You are Looking for Before Generation. Preprint, arXiv:2404.14469.

Xiang Long, Li Du, Yilong Xu, RongJian Xu, Qiyanhui Lu, Ying Gao, Qinhua Xie, Fangcheng Liu, Ning Ding, Haoqing Wang, Ziheng Li, Changjiang Zhou, Jianyuan Guo, and Yehui Tang. 2026. Liveclawbench: Benchmarking llm agents on complex, realworld assistant tasks. Preprint, arXiv:2604.13072.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. Preprint, arXiv:1711.05101.

Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, and Jiecao Chen. 2025. Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management. Preprint, arXiv:2510.06727.

Meta. 2026. Introducing Muse: The world’s first personal AI agent built for everyone.

Nous Research. 2026. Hermes agent. https:// github.com/NousResearch/hermes-agent.

OpenAI. 2025. Introducing Codex.

OpenAI. 2026. Introducing dots.

OpenClaw. 2026. OpenClaw: Compaction. https: //docs.openclaw.ai/concepts/compaction.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. MemGPT: Towards LLMs as Operating Systems. Preprint, arXiv:2310.08560.

Rui Pan, Yinwei Dai, Zhihao Zhang, Gabriele Oliaro, Zhihao Jia, and Ravi Netravali. 2025. SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning. Preprint, arXiv:2504.07891.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Quan Shi, Alexandra Zytek, Pedram Razavi, Karthik Narasimhan, and Victor Barres. 2026. τ-knowledge: Evaluating conversational agents over unstructured knowledge. Preprint, arXiv:2603.04370.

Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, and Jiecao Chen. 2025. Scaling Long-Horizon LLM Agent via Context-Folding. Preprint, arXiv:2510.11967.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle˘ Casbon, Mayank Chaturvedi, Aditya Chawla, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, and 304 others. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. 2024. Appworld: A controllable world of apps and people for benchmarking interactive coding agents. Preprint, arXiv:2407.18901.

Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Minhao Cheng, Shuai Wang, Hong Cheng, and Jingren Zhou. 2026. ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization. Preprint, arXiv:2509.13313.

Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. RE-COMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation. Preprint, arXiv:2310.04408.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-MEM: Agentic Memory for LLM Agents. Preprint, arXiv:2502.12110.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, Pengjun Xie, Fei Huang, Siheng Chen, Jingren Zhou, and Yong Jiang. 2025. AgentFold: Long-Horizon Web Agents with Proactive Context Management. Preprint, arXiv:2510.24699.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. 2025. MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents. Preprint, arXiv:2506.15841.

## A Prompt Template

## A.1 Drafter Prompt

Figure 5 presents the drafter prompt and the six slot contracts. Each request includes the shared conversation history and the complete summary schema, together with the requested slot’s title, contract, and output format. The drafter generates only the requested slot’s content.

## A.2 Target Prompt

Figure 6 shows the target verification prompt and its message structure. The full conversation history and all six slot contracts form a shared prefix, followed by the verification request for a proposed line. The target returns positive only if the line is supported by the history and consistent with the accepted summary lines.

![](images/3bd9e275c311afc384d26f99d4a5ea82d58b33acfb3d0330318a2a23b3f49158.jpg)  
Figure 5: The drafter prompt template and the contract for each slot.

![](images/8c412c0515139f41c12416f3eb1bc4196c6b30d46586ad86cdf5c3a48a88363d.jpg)  
Figure 6: Target verification prompt (top) and the message structure with a shared prefix (bottom).