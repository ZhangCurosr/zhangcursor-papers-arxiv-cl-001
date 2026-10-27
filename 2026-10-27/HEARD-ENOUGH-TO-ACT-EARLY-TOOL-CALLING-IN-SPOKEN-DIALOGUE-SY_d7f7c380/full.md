# HEARD ENOUGH TO ACT: EARLY TOOL CALLING IN SPOKEN DIALOGUE SYSTEMS

Yizhou Peng<sup>1</sup>, Chen Chen<sup>1</sup>, Donghang Wu<sup>1,2</sup>, Hexin Liu<sup>1</sup>, Eng Siong Chng<sup>1</sup>

<sup>1</sup> Nanyang Technological University, Singapore <sup>2</sup> AI Singapore, Singapore

## ABSTRACT

Latency-sensitive spoken agents must know not only which tool to call, but also when they have heard enough to act. Waiting for the turn to complete delays execution even when the request is already actionable. However, neither existing spoken agents nor spoken dialogue corpora define the earliest point at which a request becomes actionable. We introduce a synthetic spoken-dialogue pipeline that provides earliest-actionable-point supervision, marking when the observed speech grounds a tool and its required arguments. Through controlled task generation, speech synthesis and alignment, and gated annotation, we construct 26,694 dialogues totaling 786 hours across 48 domains, with disjoint training and test argument values. We additionally create an audio rendering of τ<sup>2</sup>-bench for external evaluation. We train an omni-modal language model on this corpus, enabling causal, chunk-level decisions and partial-transcript generation for downstream call completion. The model achieves tool-state macro-F1 scores of 0.48 and 0.47 on held-out dialogues and the external benchmark, respectively. For correct calls, the estimated mean lead over a turn-end cascade, using the same formatter, is 1.8–2.0 seconds in-domain and 1.1 seconds on the τ<sup>2</sup>. These results demonstrate a data-driven path toward spoken agents that act while listening. We will open-source the pipeline and the training scripts.

Index Terms— full-duplex interaction, low latency, tool calling, spoken dialogue systems

## 1. INTRODUCTION

People keep conversation flowing by planning their reply while the other person is still talking, so turns follow within a fraction of a second [1, 2]. A tool-using voice agent, cascaded or end-to-end, still waits for the user to finish before doing anything: only then does it work out what was said, commit to a tool and its arguments, place the call, and await the result; these delays stack rather than overlap. Even systems engineered to respond within a couple of hundred milliseconds start that clock at the end of the turn [3], and the wait routinely reaches seconds, which users can notice [4]. We do not attack transcription speed or the tool’s own round-trip here; we ask whether the step in the middle, deciding on the call and filling in its arguments, must wait for the turn to end at all.

Much of the field has made this wait feel shorter instead. Incremental generators bridge it with fillers, the way people say “um” while composing a reply [5]; a recently deployed dialogue robot even detects mid-utterance that a request is becoming clear, yet spends that head start on a spoken preface while the lookup still waits for the turn to end [6]. SHANKS [7] reasons while listening and completes more than half of its tool calls before the user finishes; Stream RAG [8] post-trains a speech model to issue tool queries during ongoing speech; and DuplexSLA [9] decodes the user’s audio, the agent’s speech, and a dedicated planning-andtool-call stream on one shared clock for car-cabin and smart-home schemas. All three learn their timing from proxies, LLM-judged offsets, or change-detection labels; none defines the moment at which a call first becomes decidable, verifies that its labels sit at that moment, or scores timing against it.

The rest of the landscape leaves that moment equally untouched. Full-duplex models such as Moshi and SyncLLM [10, 11] and chunk-level turn-taking predictors [12, 13, 14, 15] decide chunk by chunk who holds the floor; the newest think while the user speaks in explicit tokens [16] or latent embeddings [17], yet the choice remains silence, thought, or speech, never a tool call. Incremental dialogue research has shown that meaning is often settled before an utterance ends [18, 19], but has never tied that insight to an irreversible external action. And tool-calling benchmarks score which call an agent produces from a completed turn [20, 21, 22]; even their recent extensions into audio and full-duplex voice, including a rendering of τ<sup>2</sup>-bench itself, test whether speech changes what to call or whether the task still completes, not when the call became callable [23, 24, 25]. In short, no existing system fixes the earliest instant at which the tool and every argument are grounded in the audio, or scores its timing against that instant by construction.

In this paper, we aim to fire the tool call at the moment the user has said enough, and doing so is as much a data problem as a modeling one: no spoken task-oriented corpus, human-recorded [26] or LLM-generated [27], marks the earliest instant at which a call’s tool and all its arguments are grounded in the audio heard so far, and generated corpora invite the train–test contamination that inflates scores [28]. We built a synthetic spoken-dialogue pipeline that stamps every call with that instant, keeping training and test argument values disjoint by construction, and adapted an omni-modal LLM with LoRA to judge whether the instant has arrived, then either emit the call itself or hand off a short transcribed span for downstream completion. On held-out data and on our own audio rendering of τ<sup>2</sup>-bench [21], our model, when it fires at the right moment, recovers the correct call more often than not and starts it an estimated 1.8–2.0 s early on average; the remaining bottleneck is the firing decision itself, which misses a fair share of call opportunities and triggers on some turns it should leave alone. Our contributions are:

• a 786 h synthetic spoken-dialogue corpus with gated, alignmentanchored earliest-actionable-point label pipeline and a leakagefree split by construction;

• a single-stream control-token model that decides, per chunk, whether that point has arrived;

• a payload study on what the model should emit when it fires, so that an early fire becomes a correct tool call.

## 2. METHODS: DATA PIPELINE AND MODELING

We cast early tool calling (ET) as a per-chunk decision problem. User audio arrives in 0.6 s chunks; after each chunk, the model emits a voice-activity token and a tool state in {WAIT, UNSURE, READY, HOLD} on speech, as shown in Fig. 1. READY commits mid-utterance to a specific call and UNSURE flags a likely but under-specified tool; either emission is a fire. Each gold call is an event (tool, args, $t _ { \mathrm { f i r e } } )$ , where $t _ { \mathrm { f i r e } }$ is the earliest instant at which every required argument is grounded in the speech heard so far, and $\Delta _ { \mathrm { s a v e } } = t _ { \mathrm { e n d } } - t _ { \mathrm { f i r e } }$ is its gold saving over a turn-end call.

![](images/3a66e7bf33c9e53b51f17d5fbc955cdcf0ae4430cc4af0631948372a33e32e9f.jpg)  
Fig. 1. Data pipeline. Typed task templates are instantiated by a deterministic entity sampler, simulated as text dialogues, rendered to aligned speech, and annotated with gated early-tool-call events; the green event carries the earliest-actionable-point label $( t _ { \mathrm { f i r e } } )$ and its saved time. The table on the right glosses the per-chunk labels.

## 2.1. Data pipeline

Figure 1 shows the generation stack. GPT-5.5 authors 48 domains with 18–19 task templates each (900 tasks) whose identity fields are typed entity slots; a deterministic, collision-checked sampler instantiates 30 × 900 = 27,000 profiles (306 yield no usable dialogue, leaving 26,694); two DeepSeek-V4Pro Preview sessions [29] simulate the user and agent scripts; Chatterbox-turbo [30] synthesizes each turn with per-persona cloned voices; and the NeMo forced aligner [31] provides word-level timestamps.

Tool-call supervision is annotate-then-gate. Each profile receives a synthesized tool pool (mean 4.5 tools), GPT-5.6-Terra proposes candidate events, and three deterministic gates verify every proposal: Gate 1 discards non-executable calls (unknown tool, malformed or type-mismatched arguments); Gate 2 collects arguments that the transcript does not ground or that a later utterance supersedes; Gate 3 resolves them, repairs the argument, or delays $t _ { \mathrm { f i r e } }$ to the instant the argument is actually grounded. The funnel prunes $1 4 0 , 5 9 5 \  \ 7 8 , 1 9 3 \  \ 5 3 , 4 9 2$ events, and with $t _ { \mathrm { f i r e } }$ anchored to the forced alignment, $\Delta _ { \mathrm { s a v e } }$ has mean 2.40 s, median 1.36 s, and $\mathsf { p } 9 0$ 6.60 s. Since the gates can only delay, t<sub>fire</sub> is an upper bound on the true earliest point. We then expand events onto the 0.6 s grid: every chunk carries a VAD label (SPK/SIL); the speech chunk at $t _ { \mathrm { f i r e } }$ takes READY or UNSURE, and the other speech chunks read WAIT before the turn’s first fire and HOLD after it, an anti-duplicate guard that a fresh READY (a correction or a second tool) can still break. Table 1 summarizes the corpus. Splitting by task-group holdout leaves zero train/test argument-value overlap. For open benchmarking, we ran the identical pipeline on $\tau ^ { 2 } .$ -bench’s own tasks and tool schemas [21]. Unlike τ-Voice [25], this rendering scores the firing decision per chunk, not task completion.

## 2.2. Modeling

Fig. 2 shows the model. We keep Qwen3-Omni-30B-A3B [32, 33] frozen, add fifteen control tokens (VAD and ET states, payload delimiters, pending markers) whose embedding and output rows are initialized to the mean of related words and trained only through small delta vectors, as in prompt tuning [34], and attach LoRA [35] to attention and all per-expert FFN projections.

Table 1. Corpus statistics. The $\tau ^ { 2 }$ column is the external, eval-only rendering of ${ \bar { \tau } } ^ { 2 } .$ -bench tasks.
<table><tr><td></td><td>Main corpus</td><td> $\tau ^ { 2 }$  rendering</td></tr><tr><td>Dialogues</td><td>26,694</td><td>274</td></tr><tr><td>Held-out dialogues (valid / test)</td><td>210 / 210</td><td>0 /274</td></tr><tr><td>Domains</td><td>48</td><td>3</td></tr><tr><td>Audio</td><td>786 h</td><td>4.9 h</td></tr><tr><td>READY / UNSURE events</td><td>28,791 /24,701</td><td>238 / 11</td></tr><tr><td>Test gold events</td><td>463</td><td>249</td></tr><tr><td>Gold  $\Delta _ { \mathrm { s a v e } }$  med / p90 (s)</td><td>1.36 / 6.60</td><td>0.00 / 3.87</td></tr></table>

We assemble one sequence per dialogue. The system prompt and the profile’s tool schemas are followed by a boundary token, the chunk’s 8 audio embeddings $\mathbf { x } _ { c }$ (0.6 s of audio), and the emitted tokens $\mathbf { y } _ { c }$ predicted as p(y<sub>c</sub> $\scriptstyle | \mathbf { x } \leq _ { c } , \mathbf { y } < _ { c } )$ . Every $\mathbf { y } _ { c }$ begins with a VAD token $v _ { c } \in \{ \mathrm { s P K } , \mathrm { s I L } \}$ and if $v _ { c } { = } \mathbf { S } \mathrm { P K }$ then an ET state $s _ { c }$ . READY and UNSURE open a payload block, and three payload variants differ only in what it contains:

$$
\mathbf { y } _ { c } = \left\{ \begin{array} { l l } { \left( v _ { c } , \ s _ { c } \right) } & { \mathrm { N O N E } } \\ { \left( v _ { c } , \ s _ { c } , \ a _ { c } \left[ s _ { c } \in \left\{ \mathrm { R } , \mathrm { U } \right\} \right] \right) } & { \mathrm { A S R } } \\ { \left( v _ { c } , \ s _ { c } , \ a _ { c } \left[ s _ { c } \in \left\{ \mathrm { R } , \mathrm { U } \right\} \right] , \ j _ { c } \left[ s _ { c } = \mathrm { R } \right] \right) } & { \mathrm { A S R } + \mathrm { J S O N } } \end{array} \right.\tag{1}
$$

where $[ \cdot ]$ is the emission condition, R/U abbreviate READY/UNSURE, $a _ { c }$ is a partial-ASR span of the speech heard so far between <asr> delimiters, and $j _ { c }$ is the tool-call JSON $( \{ \ " \mathsf { t o o l \_ n a m e } ^ { \mathfrak { n } } ,$ "arguments"}, keys sorted) between <tool> delimiters. Each variant answers differently how a fire becomes an executable call: ASR+JSON is self-contained, its READY JSON being the call; ASR externalizes formatting, sending its span, the tool schemas, and the dialogue prefix to a frozen LLM formatter that emits the JSON; NONE carries no payload, so acting on its fires also requires an external ASR pass over the audio near the fire.

## 3. EXPERIMENTS

## 3.1. Setup

All variants are trained with LoRA rank r=16 (843.8M trainable parameters), under one weighted cross-entropy (VAD 0.4; ET, ASR,

![](images/7e69d9797f5eca0f201eb0a603fc6632d6ddbb43427de64942a19109d438da42.jpg)  
Fig. 2. Single-stream model design. One autoregressive, KV-cached sequence interleaves injected audio chunks with emitted control tokens and payload blocks; one LM head decodes VAD, ET, ASR, and tool JSON under a single weighted cross-entropy loss. Snowflakes are frozen; flames are trainable. The example shows the ASR+JSON variant of Eq. (1).

JSON 1.0; READY/UNSURE class weights 6/10) with event-countaware oversampling, early stopping, and a single seed. We decode the held-out test split and the audio-rendered $\tau ^ { \frac { \smile } { 2 } }$ -bench in streaming mode, with a finite-state machine masking illegal tokens per slot, and score at two levels of granularity. At the state level, every chunk’s VAD state and every speech chunk’s ET state are classified against gold with exact alignment. At the event level, each READY/UNSURE emission is a predicted event and is matched if a gold event of the same state lies within ±2 chunks (±1.2 s) regardless of payload. This rule yields the event metrics of Table 2 and the matched fires that form the denominator of Table 5 and Fig. 3. The tolerance reflects the coarse 0.6 s grid: a gold $t _ { \mathrm { f i r e } }$ near a chunk boundary is legitimately answered from the neighboring chunk. We report event recall/precision/F1, ET/VAD-state macro-F1, per-state F1, and word error rate (WER) of emitted partial-ASR spans; at the call level, we report on matched READY fires, exact match (EM) of the JSON string, tool-name accuracy, and argument accuracy (micro-averaged over gold slots). The formatter is Qwen3-235B-A22B or Qwen3- 30B-A3B [33], prompted with the tool pool, the completed turns before the event, previous calls, and the model’s emissions.

The achieved saving is $t _ { \mathrm { e n d } } \mathrm { ~ - ~ } t _ { \mathrm { f i r e } } ^ { \mathrm { p r e d } }$ , charging a predicted fire at the end of its chunk. It estimates the savings over the turn-end cascade TURN, which transcribes the completed turn and hands it to the same formatter, which also decides whether to call. TURN-ASR uses Parakeet-TDT-CTC 1.1B [36, 37], the aligner of our data pipeline, as external ASR; TURN-GOLD uses the gold transcript; NONE runs the same ASR over the audio window around its fire. The baseline transcript is charged at zero latency, an ideal streaming recognizer, and the assumption most favorable to the baseline.

## 3.2. Results

Table 2 reports the decision layer on both domains. All three variants detect events with recall around 0.5 on ID and 0.56–0.58 on $\tau ^ { 2 } ,$ though emitting a payload does sharpen the decision: the payloadfree NONE fires far too often, with precision of only 0.123 and 0.147 for ID and $\tau ^ { 2 } .$ , while ASR, which transcribes at the fire, gives the best ID event F1 of 0.414 and the best state classification for both ET and VAD; the ASR/ASR+JSON differences are small. The decision layer carries over to τ<sup>2</sup> largely intact: ET-state macro-F1 is unchanged for the span variants (only NONE drops by 0.05), and event F1 comes out even higher, with precision of 0.412. The payload is where transfer breaks: partial-ASR WER is 0.11–0.12 in domain but rises to 0.48

Table 2. Decision quality on the in-domain (ID) and audio-rendered $\tau ^ { 2 } .$ -bench splits for event, state, and WER across three variants. Bold marks the best variant per row within each split.
<table><tr><td rowspan="2">Payload variant Test split</td><td colspan="2">NONE</td><td colspan="2">ASR</td><td colspan="2">ASR+JSON</td></tr><tr><td>ID</td><td> $\tau ^ { 2 }$ </td><td>ID</td><td> $\tau ^ { 2 }$ </td><td>ID</td><td> $\tau ^ { 2 }$ </td></tr><tr><td>Event recall</td><td>0.5530.5580.5100.5620.4950.582</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Event precision</td><td>0.1230.1470.3490.4120.3290.409</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Event F1</td><td>0.2010.2320.4140.4750.3950.480</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ET-state macro-F1</td><td>0.4770.4250.4840.4740.4630.461</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VAD-state macro-F1</td><td>0.993 0.9760.996 0.982 0.994 0.979</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Partial-ASR span WER</td><td></td><td></td><td></td><td></td><td></td><td>0.1200.551 0.1120.478</td></tr></table>

Table 3. Per-class ET F1, computed per chunk over all speech chunks for the in-domain test split. F1@+k credits a prediction up to k chunks (0.6k s) late via exact matching; @0 is exact alignment. Bold marks the best variant per column at @0.
<table><tr><td></td><td></td><td>WAIT</td><td>READY</td><td>UNSURE</td><td>HOLD</td><td>Macro</td></tr><tr><td>NONE</td><td>@0</td><td>0.936</td><td>0.309</td><td>0.175</td><td>0.488</td><td>0.477</td></tr><tr><td></td><td>@+1</td><td>0.938</td><td>0.405</td><td>0.242</td><td>0.493</td><td>0.519</td></tr><tr><td></td><td>@+3</td><td>0.938</td><td>0.432</td><td>0.272</td><td>0.493</td><td>0.534</td></tr><tr><td>ASR</td><td>@0</td><td>0.932</td><td>0.315</td><td>0.206</td><td>0.483</td><td>0.484</td></tr><tr><td></td><td>@+1</td><td>0.934</td><td>0.384</td><td>0.244</td><td>0.487</td><td>0.512</td></tr><tr><td></td><td>@+3</td><td>0.934</td><td>0.406</td><td>0.277</td><td>0.487</td><td>0.526</td></tr><tr><td>ASR+JSON</td><td>@0</td><td>0.927</td><td>0.299</td><td>0.158</td><td>0.469</td><td>0.463</td></tr><tr><td></td><td>@+1</td><td>0.929</td><td>0.359</td><td>0.205</td><td>0.471</td><td>0.491</td></tr><tr><td></td><td>@+3</td><td>0.929</td><td>0.367</td><td>0.232</td><td>0.471</td><td>0.500</td></tr></table>

and 0.55 for ASR+JSON and ASR on $\tau ^ { 2 , } \mathrm { s }$ unseen entity-value patterns.

Table 3 splits the ET-state macro-F1 by state and by lateness tolerance. WAIT is nearly solved and READY/UNSURE carry the errors: false-positive READY fires are the precision bottleneck, and every variant learns UNSURE where ASR gets best F1 of 0.206. Granting one chunk of lateness tolerance lifts READY F1 by 0.06–0.10 and the macro by 0.03–0.04, with little further gain at +3: a sizeable share of the apparent errors are fires one chunk late.

Table 4 answers the question: on a turn that needs a call, did the system fire? It scores every system under one turn-level rule, so our systems can be set against the turn-end cascade TURN. TURN is recall-heavy; at recall 0.894 it fires on 751–789 of the 1,499 turns, two-thirds of them without a gold READY, whereas ASR+JSON gives up 0.3 of recall for the highest precision of 0.409; F1 is within 0.05 across all four. Deciding early thus trades off recall for precision at the same F1. Even so, 59% of early fires are spurious at this operating point, so the precision an agent needs before committing a call has not yet been reached.

Table 4. Firing decision under one turn-level rule for in-domain test split (1,499 user turns, 292 with a gold READY event). A system fires on a turn if it emits at least one READY in it; a fire is a true positive if that turn holds a gold READY. The payload variants fire from their own streaming READY decisions; TURN fires when the formatter decides to call. Bold marks the best per column.
<table><tr><td>Variant</td><td>Decision from</td><td>Fired</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>ASR</td><td>model READY</td><td>458</td><td>0.376</td><td>0.589</td><td>0.459</td></tr><tr><td>ASR+JSON</td><td>model READY</td><td>418</td><td>0.409</td><td>0.586</td><td>0.482</td></tr><tr><td>TURN-ASR</td><td>Parakeet text</td><td>789</td><td>0.331</td><td>0.894</td><td>0.483</td></tr><tr><td>TURN-GOLD</td><td>gold text</td><td>751</td><td>0.348</td><td>0.894</td><td>0.500</td></tr></table>

![](images/767318b2cdeba89a2a8570f20f0c206da2e4efca5bbc7444cb63499e2e4dd589.jpg)  
saving vs. turn-end cascade x (s), streaming-ASR baseline  
Fig. 3. Achieved saving of correct early calls (Qwen3-235B-A22B formatter) over the TURN baseline: fraction of calls saving more than x seconds. Dots mark means (legend); the value at x=0 is the fraction that fires before the turn ends; medians over all correct calls are 0.0 s (ASR) and 0.7 s (ASR+JSON) on ID.

Figure 3 shows the distribution of the achieved saving restricted to correct calls, with the Qwen3-235B-A22B formatter, for both variants and both domains. The distributions are heavily rightskewed. For ID, about half of the correct calls fire before the turn ends (50% for ASR, 54% for ASR+JSON); the rest fire in the turn’s final chunk. The median saving over all correct calls is therefore 0.0 s (ASR) and 0.7 s (ASR+JSON); among the calls that do fire early, the median is 2.6 s, and 44% and 47% of correct calls save more than 1 s, so the means (1.83 and 2.01 s, legend) are carried by a long tail. On $\tau ^ { 2 } .$ , gold $\Delta _ { \mathrm { s a v e } }$ has a median of 0 because its turns place tool evidence at the turn end, which makes the timing task easier; even so, 54–55% of the correct calls fire before the turn end, with a median saving of 1.8 s among them.

## 4. ABLATION: PRODUCING THE CALL

Table 5 compares the strategies on the models’ own READY fires.

Externalizing the call is clearly better. On ASR+JSON’s own fires, the formatter given the model’s span reaches EM 0.536, against 0.173 for the JSON the model emits itself, and the decomposition shows why: the trained JSON head names the tool on only 23% of matched fires and fills 52% of argument slots, whereas the formatter reaches 0.83 and 0.71 from the same span. The ASR variant, trained without a JSON head, still performs slightly better with EM of 0.578.

Table 5. Call quality by payload variant. Variant: whose READY fires trigger the call. Call from: the text the call is built from, with the model’s own JSON (no formatter), the model’s own span (a<sub>c</sub>), a Parakeet transcript, or the gold transcript; the latter three are via the formatter. For TURN the text covers the completed turn; for all other variants, the audio up to the fire. EM, tool-name, and argument accuracy on matched fires (TURN: on true-positive turns). Bold marks the best per column (except GT ASR).
<table><tr><td colspan="3"></td><td colspan="3">ID</td><td colspan="3"> $\tau ^ { 2 }$ </td></tr><tr><td>Variant</td><td>Call from</td><td>Formatter</td><td>EM</td><td>Tool Arg</td><td></td><td>EM</td><td>Tool</td><td>Arg</td></tr><tr><td></td><td>ASR+JSON model JSON</td><td></td><td>0.173 0.232 0.519 0.000 0.073 0.054</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ASR+JSON model ASR</td><td>235B</td><td>0.5360.8330.710 0.6810.9780.844</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ASR+JSON model ASR</td><td>30B</td><td>0.530 0.833 0.707 0.674 0.964 0.839</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ASR</td><td>model ASR</td><td>235B</td><td>0.578 0.840 0.711 0.618 0.971 0.770</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ASR</td><td>model ASR</td><td>30B</td><td>0.535 0.845 0.683 0.588 0.963 0.776</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ASR</td><td>GT ASR</td><td>235B</td><td>0.626 0.840 0.760 0.838 0.963 0.888</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ASR</td><td>GT ASR</td><td>30B</td><td>0.588 0.856 0.737 0.831 0.963 0.893</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NONE</td><td>ext. ASR</td><td>235B</td><td>0.038 0.870 0.493 0.090 0.955 0.696</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TURN</td><td>ext. ASR</td><td>235B</td><td>0.1030.7890.634 0.0730.744 0.753</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TURN</td><td>GT ASR</td><td>235B</td><td>0.605 0.751 0.832 0.558 0.694 0.915</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Span fidelity, not the formatter, is the residual. The gold-text rows bound the remaining loss: tool naming is unchanged with ACC of 0.84 while argument accuracy rises to 0.76 and EM to 0.63, so what the model transcribes at the fire moment is the dominant error; the 30B formatter concedes at most 4.3 EM points.

A bare decision is not enough. The decision-only NONE variant names the tool from a Parakeet pass over the fire window with Tool ACC of 0.87, but fills only half the argument slots, and its EM falls to 0.038 because Parakeet output carries spoken-form entities that never string-match the gold JSON. The model’s own transcribe-atfire span is what makes the call executable (on $\tau ^ { 2 }$ , the same variant improves to EM 0.090 on shorter, more regular payloads, still roughly 7× below the span variants).

Against the turn-end cascade. The TURN rows are the deployed alternative. With Parakeet text, TURN-ASR names the tool on 79% of its true-positive turns and fills 63% of argument slots, against 83% of slots with gold text, yet scores EM 0.103: the same normalization failure, not timing. With gold text, it reaches EM 0.605. The pattern transfers: $\mathrm { o n } \tau ^ { 2 }$ , the Parakeet arm again falls to EM 0.073 while gold text holds 0.558.

## 5. CONCLUSION

This paper presented spoken tool calling as an earliest-actionablepoint decision. A synthetic pipeline labels, for 26,694 dialogues, the instant each call’s tool and arguments are grounded in the audio, with disjoint train/test values; a LoRA-adapted Qwen3-Omni-30B-A3B decides per 0.6 s chunk whether that instant has arrived and emits the call as JSON or a partial transcript for an off-the-shelf formatter. Under streaming decoding on held-out dialogues and audio-rendered $\tau ^ { 2 } .$ -bench, the decision reaches macro-F1 of 0.48 and 0.47, and correct early calls start an estimated 1.8–2.0 s sooner than a turn-end cascade with the same formatter. The transcript yields exact calls 3× as often as self-emitted JSON (0.54 vs. 0.17 on the same fires), and a payload-free decision plus external ASR fills half the argument slots. The evidence is single-seed on one base model; with 59% of early fires spurious, fire precision is the open gap.

## 6. REFERENCES

[1] Tanya Stivers et al., “Universals and cultural variation in turntaking in conversation,” Proc. Natl. Acad. Sci., vol. 106, no. 26, pp. 10587–10592, 2009.

[2] Stephen C. Levinson and Francisco Torreira, “Timing in turntaking and its implications for processing models of language,” Front. Psychol., vol. 6, pp. 731, 2015.

[3] Qingkai Fang et al., “LLaMA-Omni: Seamless speech interaction with large language models,” in Proc. ICLR, 2025.

[4] Mykola Maslych et al., “Mitigating response delays in free-form conversations with LLM-powered intelligent virtual agents,” in Proc. ACM CUI, 2025.

[5] Gabriel Skantze and Anna Hjalmarsson, “Towards incremental speech generation in conversational systems,” Comput. Speech Lang., vol. 27, no. 1, pp. 243–262, 2013.

[6] Yuki Okafuji, Koji Inoue, and Yoshiki Ohira, “Low-latency turn-taking via context-aware preface generation in a realworld dialogue robot,” arXiv:2607.23204, 2026.

[7] Cheng-Han Chiang et al., “SHANKS: Simultaneous hearing and thinking for spoken language models,” arXiv:2510.06917, 2025.

[8] Siddhant Arora et al., “Stream RAG: Instant and accurate spoken dialogue systems with streaming tool usage,” arXiv:2510.02044, 2025.

[9] Haoyang Zhang et al., “DuplexSLA: A full-duplex spoken language model with synchronized speech, language, and action,” arXiv:2605.20755, 2026.

[10] Alexandre Defossez et al., “Moshi: a speech-text foundation´ model for real-time dialogue,” arXiv:2410.00037, 2024.

[11] Bandhav Veluri et al., “Beyond turn-based interfaces: Synchronous LLMs as full-duplex dialogue agents,” in Proc. EMNLP, 2024.

[12] Xiong Wang et al., “Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen LLM,” arXiv:2411.00774, 2024.

[13] Qian Chen et al., “MinMo: A multimodal large language model for seamless voice interaction,” arXiv:2501.06282, 2025.

[14] Peng Wang, Songshuo Lu, Yaohua Tang, Sijie Yan, Wei Xia, and Yuanjun Xiong, “A full-duplex speech dialogue scheme based on large language model,” in Proc. NeurIPS, 2024.

[15] Erik Ekstedt and Gabriel Skantze, “Voice activity projection: Self-supervised learning of turn-taking events,” in Proc. Interspeech, 2022.

[16] Donghang Wu et al., “Chronological thinking in full-duplex spoken dialogue language models,” in Proc. SIGDIAL, 2026, pp. 473–485.

[17] Donghang Wu et al., “The silent thought: Modeling internal cognition in full-duplex spoken dialogue models via latent reasoning,” arXiv:2603.17837, 2026.

[18] David Schlangen and Gabriel Skantze, “A general, abstract model of incremental dialogue processing,” in Proc. EACL, 2009.

[19] David DeVault, Kenji Sagae, and David Traum, “Can I finish? Learning when to respond to incremental interpretation results in interactive dialogue,” in Proc. SIGDIAL, 2009.

[20] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan, “τ-bench: A benchmark for tool-agent-user interaction in real-world domains,” arXiv:2406.12045, 2024.

[21] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan, “τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment,” arXiv:2506.07982, 2025.

[22] Shishir G. Patil et al., “The Berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models,” in Proc. ICML, 2025.

[23] Md Tahmid Rahman Laskar et al., “From text to voice: A reproducible and verifiable framework for evaluating tool calling LLM agents,” arXiv:2605.15104, 2026.

[24] Ramit Pahwa et al., “Audio2Tool: Speak, call, act — a dataset for benchmarking speech tool use,” arXiv:2604.22821, 2026.

[25] Soham Ray et al., “τ-voice: Benchmarking full-duplex voice agents on real-world domains,” arXiv:2603.13686, 2026.

[26] Shuzheng Si et al., “SpokenWOZ: A large-scale speech-text benchmark for spoken task-oriented dialogue agents,” in Proc. NeurIPS Datasets and Benchmarks, 2023.

[27] Minghan Wang et al., “SpeechDialogueFactory: Generating high-quality speech dialogue data to accelerate your Speech-LLM development,” arXiv:2503.23848, 2025.

[28] Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan, “Investigating data contamination in modern benchmarks for large language models,” in Proc. NAACL, 2024.

[29] DeepSeek-AI, “DeepSeek-V4: Towards highly efficient million-token context intelligence,” arXiv:2606.19348, 2026.

[30] Resemble AI, “Chatterbox-TTS,” https://github.com/ resemble-ai/chatterbox, 2025, GitHub repository.

[31] Elena Rastorgueva, Vitaly Lavrukhin, and Boris Ginsburg, “NeMo forced aligner and its application to word alignment for subtitle generation,” in Proc. Interspeech (Show & Tell), 2023.

[32] Jin Xu et al., “Qwen3-omni technical report,” arXiv:2509.17765, 2025.

[33] An Yang et al., “Qwen3 technical report,” arXiv:2505.09388, 2025.

[34] Brian Lester, Rami Al-Rfou, and Noah Constant, “The power of scale for parameter-efficient prompt tuning,” in Proc. EMNLP, 2021.

[35] Edward J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.

[36] Dima Rekesh et al., “Fast conformer with linearly scalable attention for efficient speech recognition,” in Proc. IEEE ASRU, 2023.

[37] Hainan Xu et al., “Efficient sequence transduction by jointly predicting tokens and durations,” in Proc. ICML, 2023.