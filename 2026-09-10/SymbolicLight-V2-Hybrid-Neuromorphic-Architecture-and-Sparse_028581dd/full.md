# SymbolicLight V2: Hybrid Neuromorphic Architecture and Sparse Execution for Low-Energy Language Inference

Ting Liu SymbolicLight Research Foshan, Guangdong, China research@symboliclight.com

September 7, 2026

## Abstract

Event-driven language inference saves energy when the execution system turns zero activations into omitted computation and memory access. We present SymbolicLight V2, a hybrid neuromorphic language architecture combining sparse event computation with continuous-state processing. V2 extends V1’s spike-gated dual-path design with graded signed events at additional projections and softmax-free local attention. An Alveo U50C field-programmable gate array (FPGA) implements an end-to-end digital fixed-point inference prototype of the 194M-parameter model, alongside a sparse integer ARM implementation. Across three same-checkpoint FPGA implementations at 175 MHz, active-row weight gathering and valid-state key/value (KV) cache loading raise continuous decode throughput for a 32-token prefix and 128 generated tokens from 474.6 to 643.2 tokens/s, while reducing estimated gross card energy from 0.06087 to 0.04407 J per generated token (27.6%). Complete-request energy, including prefill, falls by 24.4–27.7% across three tested prefix lengths. An independent idle-split measurement attributes 82.8% of gross card energy to the loaded-idle share, explaining the energy value of shorter token latency. Against the recorded RTX 5090 compiled 32-bit floating-point (FP32) baseline, the integer FPGA deployment uses 89.1% less estimated card energy during short-context decode; the comparison uses diferent arithmetic precisions and does not use the lowest-energy tested GPU configuration. On four Cortex-A76 cores of a ROCK 5T, complete requests reach 65.4 generated tokens/s at 9.80 W and 0.151 J/token at the adapter’s AC input. The results show how sparse execution of a hybrid neuromorphic language architecture lowers measured inference cost through event-aware computation and data movement. These mechanisms also provide a design basis for other dedicated V2 implementations: increasing throughput by a larger factor than the change in active power reduces energy per generated token. Evaluation fixes the deployed checkpoint, whose quality trails a same-budget dense control; it does not establish equal-quality eficiency.

Keywords: hybrid neuromorphic language architecture; event-driven computation; FPGA; sparse integer execution; energy per generated token.

## 1 Introduction

Activation sparsity reduces inference energy when the execution system avoids the corresponding arithmetic and memory trafic. SymbolicLight V1 combined binary Leaky Integrate-and-Fire (LIF) dynamics, a continuous residual stream, and a dual-path temporal mixer (Liu, 2026b). A recurrent decay path maintained temporal state while local attention accessed recent context. V1 explicitly adopted a hybrid design, using the combination of discrete spikes and continuous internal dynamics as biological motivation. Although this model trained with sparse activations, its dense GPU implementation paid for work that those zeros could have removed.

V2 develops this lineage into a hybrid neuromorphic language architecture with executable sparsity. Here, hybrid neuromorphic denotes the cooperation of sparse event computation and continuous-valued state processing: events select projection contributions, while recurrent state and a continuous residual carry temporal and feature information. The prototype implements both using digital fixed-point arithmetic. V2 extends event coding to feed-forward up-projection and query/key/value (Q/K/V) inputs, eventizes the projected vectors again, and uses ReLU–�<sub>1</sub> local attention with linear position bias. Its deployment turns nonzero events into selected weight-row accesses: ARM loops skip zero-input contributions, and the FPGA gathers only the required rows from resident high-bandwidth memory (HBM).

The principal experiment holds the trained checkpoint fixed while improving FPGA execution. Replacing dense weight enumeration with active-row gathering, then loading only valid key/value (KV) cache state, increases p32/n128 continuous decode throughput by 35.5% and reduces estimated gross card energy by 27.6%. Here p32/n128 denotes a 32-token prefix followed by 128 generated tokens. Including prompt ingestion, energy falls by 24.4–27.7% across the three tested prefix lengths. These are measured changes across successive implementations; the experiment does not isolate every hardware change or thermal efect.

The results also explain why latency matters to energy. Sparse projections expose KV-loading waits that were previously overlapped with computation. Reducing unnecessary state transfer recovers part of the execution benefit. Here, loaded idle means that model weights remain in device memory while no inference is running. In an independent measurement, loaded-idle power accounts for 82.8% of the FPGA’s gross energy per token. Faster generation therefore reduces the platform energy allocated to each output even when incremental energy changes little.

Two deployment comparisons place these results in context. Against the recorded RTX 5090 compiled-FP32 baseline, short-context integer FPGA decoding uses 89.1% less estimated card energy, with precision and sensor boundaries stated alongside the comparison. On four Cortex-A76 cores, the same integer model completes short requests at 65.4 generated tokens/s and 0.151 J/token at the board adapter’s AC input. A comparison of six CPU and neural processing unit (NPU) deployments on the same board shows how this edge deployment trades throughput and energy as input length grows.

The paper makes three contributions:

1. Hybrid neuromorphic architecture and sparse execution. V1-derived dual paths combining extended graded-event projections with continuous-state processing, mapped to actual nonzero-input integer computation on ARM and active-row weight gathering on FPGA (Sections 2 and 3).

2. Same-checkpoint energy reduction. A measured FPGA implementation progression that connects selective weight access and valid-state KV loading to higher throughput and lower gross energy, with an independent idle-share decomposition (Section 5).

3. Cross-platform deployment evidence. Complete-request ARM AC measurements, FPGA/GPU cardsensor comparisons, and explicit workload and measurement boundaries, supported by integer-reference validation (Sections 4 and 5).

The evaluation focuses on execution of a fixed checkpoint. Model-quality diagnostics are summarized in Section 6 and reported in Section A.

## 2 From V1 to the V2 Hybrid Neuromorphic Architecture

## 2.1 The V1 starting point

V1’s two paths operate within each block: a first-order decay state and local attention, combined with a continuous residual stream and a feed-forward sublayer. Binary spike inputs feed the decay projection and the feed-forward down-projection; the LIF input encoder and the stateless threshold operations within later blocks have distinct roles. In contrast, Q/K/V and the feed-forward up-projection consume continuous inputs. Zero activations at the original spike sites therefore leave substantial dense computation elsewhere in the graph.

Per-element activity also cannot serve as a token-level attention mask. V1 measured an almost always active position mask: reducing the number of active elements did not remove whole keys. Its released implementation computed dense attention scores before applying the local mask, and cached decoding retained all historical K/V entries. The visible local window therefore did not bound the actual computation, nor did element-level sparsity reduce attention work proportionally. V2 consequently acts at projection inputs and within the attention arithmetic, instead of treating a token as inactive only when all its elements are zero.

## 2.2 Graded signed events at projection inputs

Let � be the activation presented to an event encoder with positive threshold � and positive integer magnitude bound �. The hard forward event is

$$
e _ { \theta } ( x ) = { \bf 1 } [ | x | \geq \theta ] \ \mathrm { c l i p } ( \mathrm { r o u n d } ( x / \theta ) , - L , L ) .\tag{1}
$$

Here round rounds to the nearest integer with ties to even, clip bounds the result to $[ - L , L ]$ , and 1 is the indicator function. A zero value denotes no event; a nonzero integer carries both sign and magnitude. Training uses a straight-through estimator with surrogate input derivative $\widetilde { \partial e } _ { \theta } / \widehat { \partial x } = 1 / \theta$ , including the silent and saturated regions of the hard forward function. This follows the broader use of surrogate derivatives in spiking networks (Neftci et al., 2019). V2 adds these encoders at the feed-forward up-projection input and at Q/K/V projection inputs in every block. Its selected configuration leaves the attention output-projection input continuous. The binary-spike-consuming decay and feed-forward down-projections inherited from V1 remain.

The mathematical event alphabet and the deployment payload width are distinct. The training configuration allows $L = 2 5 5 .$ , whose signed range is wider than signed INT8. The hardware representation uses signed 8-bit payloads with its specified quantization and saturation semantics. Accordingly, the hardware correctness claim is made against the integer deployment reference, not by assuming the entire training event alphabet fits into INT8.

Using column vectors throughout, let $e \in \mathbb { R } ^ { d _ { \mathrm { i n } } } , y \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ , and $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ . An event-consuming projection is

$$
y = W e = \sum _ { j \in { \mathcal { R } } ( e ) } e _ { j } W _ { : , j } , \qquad { \mathcal { A } } ( e ) = \{ j : e _ { j } \neq 0 \} .\tag{2}
$$

The active set $\mathcal { A } ( e )$ selects columns of the mathematical matrix �. Hardware stores $W ^ { \top }$ by input index, so these columns correspond to stored weight rows; a zero event requires no row fetch. Since nonzero events have graded amplitudes, their execution generally still requires integer multiply-accumulate (MAC) operations. The savings come from omitted contributions and weight movement, not from treating every event as an addition-only binary spike.

## 2.3 Event attention and the retained dual path

The decay path preserves recurrent temporal state. For layer ℓ and binary input $s _ { \ell , t }$ , the vector $\alpha _ { \ell }$ contains one learned sigmoid-parameterized decay coeficient per head, broadcast across that head’s channels. The

operator ⊙ denotes elementwise multiplication, and a new session starts from zero state. Its recurrence is

$$
z _ { \ell , t } = W _ { T } s _ { \ell , t } , \qquad h _ { \ell , t } = \alpha _ { \ell } \odot h _ { \ell , t - 1 } + ( 1 - \alpha _ { \ell } ) \odot z _ { \ell , t } .\tag{3}
$$

The attention path supplies content-dependent access to recent tokens and fixed global anchors. In V2, the projected Q/K/V vectors are eventized a second time. The selected event attention path uses per-head distance penalties from Attention with Linear Biases (ALiBi) (Press et al., 2022) in place of rotary transformations, avoiding rotation of the discrete event vectors.

With zero-based positions and maximum lookback distance �, the visible set is

$$
\mathcal { K } _ { i } = \{ j \in \mathbb { Z } : 0 \leq j \leq i , \quad i - j \leq w \mathrm { ~ o r ~ } j < 4 \} .
$$

For head ℎ, let $\bar { q } _ { i } ^ { ( h ) } , \bar { k } _ { j } ^ { ( h ) } , \bar { \nu } _ { j } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } }$ be the projected event vectors. The training-level attention definition is

$$
s _ { i j } ^ { ( h ) } = \frac { ( \bar { q } _ { i } ^ { ( h ) } ) ^ { \top } \bar { k } _ { j } ^ { ( h ) } } { \sqrt { d _ { h } } } - m _ { h } ( i - j ) ,
$$

$$
j \in \mathcal { K } _ { i } ,\tag{4}
$$

$$
a _ { i j } ^ { ( h ) } = \frac { \operatorname* { m a x } ( 0 , s _ { i j } ^ { ( h ) } ) } { \sum _ { k \in \mathcal { K } _ { i } } \operatorname* { m a x } ( 0 , s _ { i k } ^ { ( h ) } ) + \epsilon } ,
$$

$$
o _ { i } ^ { ( h ) } = \sum _ { j \in \mathcal { K } _ { i } } a _ { i j } ^ { ( h ) } \bar { \nu } _ { j } ^ { ( h ) } .\tag{5}
$$

The model has $n _ { \mathrm { h e a d s } } ~ = ~ 1 2$ heads of dimension $d _ { h } ~ = ~ 6 4$ , with slopes $m _ { h } \ = \ 2 ^ { - 8 ( h + 1 ) / n _ { \mathrm { h e a d s } } }$ for $h \ =$ $0 , \ldots , n _ { \mathrm { h e a d s } } - 1$ and $\epsilon = 1 0 ^ { - 6 }$ . The four anchors are the first four sequence positions, subject to the causal mask. Positions outside $\mathcal { K } _ { i }$ receive zero attention weight; if every permitted score is non-positive, the attention output is zero. $\mathrm { R e L U } { - } L _ { 1 }$ normalization removes exponentiation, while scaling, accumulation, and output projection remain in the execution budget. The FPGA implements these operations with deterministic integer arithmetic: its zero-sum branch returns zero, and positive sums use specified integer division and rounding rather than evaluating the training-time � expression.

Let � concatenate the head outputs $o _ { i } ^ { ( h ) }$ , � be the layer gate, and � the continuous block input. Omitting dropout and layer/position indices, fusion and the feed-forward path are

$$
\begin{array} { r } { u = \mathrm { L N } _ { 1 } \big ( c + W _ { O } \left[ g o + ( 1 - g ) h \right] \big ) , } \end{array}\tag{6}
$$

$$
c ^ { \prime } = \mathrm { L N } _ { 2 } \big ( u + W _ { \mathrm { d o w n } } H ( W _ { \mathrm { u p } } e _ { \theta } ( u ) - \tau ) \big ) , \qquad s ^ { \prime } = H ( c ^ { \prime } - \tau ) .\tag{7}
$$

Here LN denotes layer normalization, � is the unit-step function with $H ( 0 ) = 1 { \mathrm { . } }$ , � is the spike threshold, and $g = \sigma ( \gamma _ { \ell } )$ is one learned scalar gate per layer. The threshold operations use surrogate gradients in training; they are not additional stateful LIF neurons. The final vocabulary projection and context-conditioned decoding head remain part of the full graph. Figure 1 draws one block: Equations (3)–(7) retain V1’s two temporal paths and continuous residual transport while adding event-coded projection inputs and event attention. The binary spikes � entering the decay projection come from the previous block’s threshold output $s ^ { \prime } ,$ , or from the LIF input encoder in the first block.

## 2.4 Configuration and hybrid computation

V2’s hybrid computation assigns complementary roles to events and continuous-valued processing. Binary spikes and graded signed events select projection contributions; the recurrent decay state accumulates temporal information, local attention retrieves context, and the continuous residual carries features across blocks. Normalization, the attention output projection, and the output head retain continuous-valued operations, represented in fixed point during deployment. This architecture is therefore not an exclusively event-based spiking neural network (SNN). The U50C is the FPGA implementation platform for the hybrid architecture.

(a) Event attention path  
![](images/b089db5fe50309ab5b386335d54514d8f9e9d4b8673a0f2758b24b7648a01403.jpg)  
Figure 1: One hybrid neuromorphic V2 block, corresponding to Equations (3)–(7). Green edges carry integer events or binary spikes into the shaded projections, which fetch only the rows selected by nonzero inputs; blue edges carry continuous values that both deployments represent in fixed point. Relative to V1, V2 adds the three $e _ { \theta }$ encoders, so that $W _ { Q } , W _ { K } , W _ { V }$ , and $W _ { \mathrm { u p } }$ now consume events, and replaces softmax attention with the $\mathrm { R e L U } { - } L _ { 1 }$ event attention path. $W _ { T }$ and $W _ { \mathrm { d o w n } }$ keep $\dot { \nabla } 1 \dot { \mathbf { s } }$ binary-spike inputs; the residual stream, $W _ { O }$ , and the layer norms remain continuous. The U50C FPGA prototype and ARM implementation execute this same integer graph through diferent dataflows.

Table 1: Architecture and training settings. V1 and V2 use diferent training corpora and tokenizers, so their absolute perplexities cannot directly measure quality changes between versions.
<table><tr><td>Property</td><td>SymbolicLight V1</td><td>SymbolicLight V2</td></tr><tr><td>Hybrid foundation</td><td>Spike-gated dual paths with a continuous residual</td><td>Inherited dual paths and residual; broader event computation</td></tr><tr><td>Core layout</td><td>12 layers, width 768, FFN 4096</td><td>Same core dimensions; 194,016,925 parameters</td></tr><tr><td>Event representation</td><td>Binary spikes; LIF input encoder</td><td>Binary inherited sites plus graded signed events</td></tr><tr><td>Event-consuming projec- Decay and FFN down tions</td><td></td><td>Additionally FFN up and Q/K/V inputs</td></tr><tr><td>Attention</td><td>Local softmax with continuous Q/K/V</td><td>Eventized Q/K/V; ReLU-L1; local window plus four anchors</td></tr><tr><td>Position mechanism</td><td>Rotary embeddings</td><td>Linear position bias</td></tr><tr><td>Residual and output head Continuous</td><td></td><td>Continuous in the model; fixed point in deployment</td></tr><tr><td>Training settings</td><td>3B-token V1 protocol</td><td>Open-data 4B-token training; 48K tokenizer</td></tr><tr><td></td><td>Hardware evaluated here No new V1 hardware measurements</td><td>INT8-weight ARM inference and an end-to-end U50C FPGA prototype</td></tr></table>

Training uses lookback distance $w = 2 5 6 \colon$ : at most 257 local positions including the current token, plus visible anchors outside that range. Deployment uses $w = 4 7 5$ , allowing at most 476 local positions plus four anchors, or 480 KV slots; overlapping positions are counted once. The FPGA stores these entries in a 480-slot circular bufer, reusing slots as the local window advances while preserving the anchors; ARM applies the same visibility mask. The short $\boldsymbol { \mathrm { p } } ^ { 3 2 / \mathrm { n } 1 2 8 }$ experiment remains within the training window; longer-prefix results evaluate the extended deployment configuration.

## 3 From ARM Execution to Sparse FPGA Hardware

## 3.1 Executable event projections on ARM

The CPU runtime executes V2 on four Cortex-A76 cores of the RK3588. OpenMP manages four parallel threads, configured to sleep rather than spin while waiting. Weights are stored as signed 8-bit integers (INT8) and packaged once for deployment. At event-consuming projections, the runtime lists the nonzero inputs, reorders their contributions, and uses ARM NEON vector instructions for integer multiply-accumulate operations; dense projections use the dot-product path. The projection loop in Equation (2) visits only nonzero contributions. Continuous operators and attention are evaluated separately.

The runtime’s logical projection-MAC counter reports executed-to-dense-enumeration ratios of 0.482, 0.435, and 0.403 for p32/n128, p128/n128, and p256/n128, corresponding to 51.8–59.7% fewer projection contributions. The counters cover matrix projections and exclude attention dot/value loops, normalization, and recurrence. They quantify skipped projection work; the energy measurements evaluate the complete implementation, without a sparse-of CPU control.

## 3.2 Integer semantics and autonomous execution

The Alveo U50C FPGA runs at 175 MHz with INT8 weights, fixed-point activations and KV state, integer event payloads, wide accumulators, and deterministic requantization. A software deployment reference specifies its arithmetic, state transitions, and ring behavior. Bit-exactness denotes agreement with this integer reference, rather than with the FP32 training checkpoint.

A common host stack handles tokenization, prompts, session control, and detokenization. Weights are uploaded once and remain resident. The device generates tokens autonomously, maintains recurrent and KV state, and returns greedy token IDs. The software model is used only for validation; measured FPGA generation receives neither activations nor gathered weights from CPU reference execution.

Two operating modes expose diferent costs. A continuous start generates a fixed number of tokens on the device and amortizes host control. Interactive operation resumes once per token and exposes that token immediately. The latter supports multi-turn conversation and exact end-of-sequence (EOS) stopping, but includes a host round trip for every token. Neither mode batches prompt ingestion: prefill advances one token per model pass.

## 3.3 From dense enumeration to active-row gathering

We compare three successive implementations built from the same hardware design and trained model weights, named here by their distinguishing mechanism. The resident-dense implementation keeps weights resident in HBM but enumerates dense weight rows at event sites while preserving the event arithmetic numerically. The sparse-gather implementation instead uses active event indices to fetch only the contributing rows in Equation (2). This upgrades the relationship between model sparsity and memory trafic: zeros can suppress physical reads rather than merely multiply fetched weights by zero.

Simulation accounts for 52% fewer weight bytes than dense enumeration on its tested workload; this measures simulated trafic, not board energy. The corresponding FPGA build reduces per-pass cycles by

13.9% relative to resident-dense. Non-sparse operators, memory service, and control still contribute to token latency, limiting the whole-pass gain. These comparisons evaluate successive implementations; they do not switch sparsity on and of within one bitstream.

## 3.4 Loading only populated KV state

The partial-KV implementation retains the sparse-gather data plane and reduces unnecessary KV loading at early session positions. Before the ring is full, loading its entire capacity transfers state that cannot yet contribute to attention. The partial-KV build loads only the populated state required by the afected layers, following the software reference’s cache-indexing rules. The final layer still loads the entire ring so that state read back for validation matches the reference. After the ring is full, this opportunity disappears.

The workload dependence makes this change testable. Continuous decode throughput improves from 544.2 to 643.2 tok/s at p32/n128, while p480/n128 changes from 400.5 to 399.7 tok/s. The near-zero change at the full-ring workload is consistent with the targeted removal of short-context state trafic.

## 3.5 Why these changes can lower energy

Consider one interval of duration $T > 0$ producing $N > 0$ tokens. Let $R = N / T$ and $\begin{array} { r } { P _ { \mathrm { a c t i v e } } = T ^ { - 1 } \int _ { 0 } ^ { T } P ( t ) } \end{array}$ d�, where $P ( t )$ is instantaneous power. With mean loaded-idle power $P _ { \mathrm { i d l e } }$ as the baseline, gross energy per generated token decomposes as

$$
E _ { \mathrm { g r o s s } } = { \frac { P _ { \mathrm { a c t i v e } } } { R } } = { \frac { P _ { \mathrm { i d l e } } } { R } } + { \frac { P _ { \mathrm { a c t i v e } } - P _ { \mathrm { i d l e } } } { R } } = E _ { \mathrm { i d l e ~ s h a r e } } + E _ { \mathrm { i n c r e m e n t a l } } .\tag{8}
$$

Power in watts divided by throughput in tokens/s gives J/token. The two terms allocate the idle baseline and the increment above it over the same duration. For complete requests, � includes prefill and decode. Dividing mean whole-run power by decode-only throughput instead yields an estimate, rather than separately integrated decode energy. Sparse row gathering can reduce work and memory trafic at eligible projections. Partial KV loading can shorten execution even when the skipped intervals consume little power beyond the platform idle level. Increased throughput then amortizes the idle share over more tokens. Resident weights and an on-device token loop also remove repeated loading and host orchestration from steady-state generation. The measurements below separate these efects where the evidence permits.

## 4 Evaluation Protocol

## 4.1 Workloads and matched model settings

We write p�/n� for a �-token prefix and � generated tokens. Prefill processes the input sequence; decode generates output tokens one at a time. The requests in the throughput and energy tables use one sequence, greedy decoding, and $N = 1 2 8$ . The ROCK 5T matrix uses $P \in \{ 3 2 , 1 2 8 , 2 5 6 \}$ ; the FPGA also includes p480/n128. The numerator counts generated tokens only, including when the denominator contains prefill:

$$
R _ { \mathrm { d e c o d e } } = N / T _ { \mathrm { d e c o d e } } ,\tag{9}
$$

$$
R _ { \mathrm { r e q u e s t } } = N / ( T _ { \mathrm { p r e f i l l } } + T _ { \mathrm { d e c o d e } } ) .\tag{10}
$$

FPGA decode time is the host-observed duration of the complete device generation start, including control overhead. Its interactive mode includes one host resume per token. Device passes/s, derived from board cycles, is a separate diagnostic.

The ARM and FPGA V2 measurements use the same checkpoint, integer deployment arithmetic, input token vectors, and attention-window parameter of 475. The CPU performs one additional forward step using the final selected output, whereas the FPGA workload stops after selecting output 128. Generated-token counts and reference output sequences therefore align, but the reported CPU time includes that additional pass. The training configuration and older GPU benchmark use a window of 256, as discussed in Sections 2 and B.

![](images/b21eb3530bc5db3c879623509a91d3e6ce929aeb11c8125ad37cb5d69c36ffae.jpg)  
U50C FPGA: sparse projections + fixed-point residual, state and output processing  
Figure 2: Representation-to-execution mapping. Event values at left are illustrative; nonzero values retain sign and amplitude. Active inputs select weight rows, omitting zero-input MACs and reads. Valid-state KV loading avoids unused transfers before the ring fills; this state optimization can also benefit dense projections. Continuous residuals, recurrent state, and output processing complete the hybrid inference graph, implemented digitally in fixed point in the U50C FPGA prototype.

Energy boundaries. We report gross and loaded-idle-subtracted incremental energy for ARM and FPGA side by side, consistently in J per generated token, while distinguishing AC-input integration from DC-card estimates. ARM energy is an integrated measurement: a smart plug at the board adapter’s AC input is integrated over complete requests and divided by generated tokens, so it includes the whole board, its adapter, and prefill. FPGA energy is a DC-card estimate: mean card power read through the Xilinx Runtime (XRT) over active runs, which excludes the host and power-supply losses, divided by decode or request throughput. The supplementary GPU energy in Section B is the analogous board-sensor estimate read through the NVIDIA Management Library (NVML). The FPGA/GPU card-sensor ratios in Section 5.5 compare the recorded deployments, whose two sensors are not cross-calibrated. Where a loaded-idle power is available, Equation (8) further splits the platform’s gross energy into an idle share and an incremental term.

## 4.2 ROCK 5T measurement and same-board comparison

The V2 remeasurement uses the unchanged V2 model package, CPU binary, and input files on a 16 GB ROCK 5T under Debian 12. Execution uses four A76 cores and four OpenMP threads, with no NPU ofload. Each of the three prefix lengths has three interleaved rounds. A round comprises warmup, 60 seconds of loaded idle, repeated complete requests for at least 60 seconds, and a further 60 seconds of loaded idle. The first five seconds of each idle interval are excluded from its power integration. All nine runs are included, totaling 183 timed requests. Each metric is calculated per round and then independently reduced to the median; values in diferent table columns need not reproduce the ratios of any single round.

The smart plug supplies integer-watt readings at approximately 4 Hz. Boundary-interpolated trapezoidal integration gives active-window energy, which is divided by generated-token count to obtain gross J/token. Incremental energy subtracts the mean pre- and post-run loaded-idle power over the same active interval. The measurement excludes the separately powered external cooler. The plug has no independent calibration or second-meter cross-check, so its integer-watt resolution does not establish absolute accuracy.

Gross-energy inter-round spreads, defined as (max − min)/median, fall below the prespecified 5% threshold. An earlier p32 series had 6.3% spread after a fourth round; the unchanged model and runtime mean that diferences between these series cannot be attributed to optimization. The p128 incremental-energy spread is 6.32%, limiting comparisons of small incremental diferences. The latest runs reach at most $7 3 . 0 ^ { \circ } \mathrm { C }$ with no cpufreq cooling action. Two start-of-load telemetry samples show a lower frequency; all other recorded active samples show 2.256 GHz on the big cores. The 1 Hz telemetry cannot establish uninterrupted constant frequency.

The comparison paths were measured earlier on the same board and day: Qwen2.5-0.5B and SmolLM2- 135M with the RKLLM inference runtime using 8-bit weights and activations (W8A8) on the NPU, and both models plus LFM2.5-350M with the llama.cpp inference runtime using its Q8\_0 weight-quantization format on four A76 cores. NPU paths use the three-core NPU with big-core support. Each comparison cell has three rounds under the same complete-request energy protocol. These models difer in parameter count, tokenizer, input sequence, quantization format, and runtime. Equal prefix/output token counts define a deployment benchmark, not equal text content or equal task quality.

The earlier llama.cpp CPU paths operated at approximately $8 7 . 7 { - } 9 0 . 3 ^ { \circ } \mathrm { C }$ , warmer than V2 and the NPU paths. A postmeasurement protocol amendment retained runs above $8 5 ^ { \circ } \mathrm { C }$ when no cpufreq cooling action occurred. These thermal conditions difer across paths, and the latest remeasurement covers V2 alone.

## 4.3 FPGA energy and correctness

The focused U50C rerun uses three rounds of at least 60 seconds per shape, with XRT/xbutil card telemetry at approximately 0.5 Hz. Mean active-run power divided by decode throughput gives the decode-energy estimate; dividing by whole-request throughput gives the corresponding request estimate. Decode energy is the median of per-round power/throughput ratios; request energy divides the independently aggregated median power by median request throughput. Power sampling spans both prefill and decode.

For the complete-request ARM comparison, we additionally reanalyze the nine retained FPGA runs. Each run provides a 10-second loaded-idle measurement before activity. We compute $( \bar { P } _ { \mathrm { r u n } } - \bar { P } _ { \mathrm { i d l e } } ) / R _ { \mathrm { r e q u e s t } }$ within each run and report the three-round median, using the same run power and request throughput as the gross-energy calculation. This is a secondary analysis of existing records, with no new measurements. Its pre-run-only idle interval is shorter than the two 60-second intervals on ARM. Subtracting idle energy does not remove diferences in device coverage, supply losses, or sensors.

The independent FPGA idle-split run measures loaded idle before and after activity with weights retained. Active power is calculated from non-warmup execution windows after the initial idle interval. Interactive energy combines timing with power sampling disabled and a separate run with power sampling enabled; it is reported as a split-run estimate.

ARM validation finds full-logit and state-hash agreement on 264 reference tokens under the original 256-window configuration. At the deployed 475-window setting, p32/p128/p256 sequences match the U50C reference for 128 generated tokens per shape. All nine saved final-request sequences from the remeasurement also match the reference sequences; outputs were not saved for all 183 requests. FPGA validation fixes the model and bitstream identities and checks 14 scenarios, including three rounds of continuous and one-token-resume workloads. A 30-minute continuous stability test completes 2,263 conversations and 27,148 turns, including 1,131 ring-wrap conversations, with 47 exact state probes and approximately 0.014% drift in time per output token (TPOT). These tests check integer execution and session stability; language quality is evaluated separately.

## 5 Energy and Execution-Eficiency Evaluation

## 5.1 Same-checkpoint energy and throughput progression

Two FPGA implementation upgrades improve both throughput and gross energy per generated token with the model weights held fixed (Table 2 and fig. 3). At p32/n128, resident-dense, sparse-gather, and partial-KV reach 474.6, 544.2, and 643.2 decode tokens/s, respectively, a cumulative 35.5% increase. Estimated gross decode energy falls from 0.06087 to 0.04407 J/generated token (27.6%). Including prefill, short-request energy falls from 0.07561 to 0.05468 J/generated token (27.7%).

Table 2: Three FPGA implementations of the same checkpoint, prefix 32 and 128 generated tokens. Three rounds per implementation: power and throughput are separate medians, decode energy is the median of per-round ratios, and request energy is median power divided by median request throughput. Energy is in J/generated token. Thermal conditions difer across measurement campaigns. Bold identifies the partial-KV implementation.
<table><tr><td>Implementation</td><td>Decode tok/s</td><td>Card W</td><td>Decode J/token</td><td>Request J/token</td></tr><tr><td>resident-dense</td><td>474.6</td><td>28.89</td><td>0.06087</td><td>0.07561</td></tr><tr><td>sparse-gather</td><td>544.2</td><td>26.69</td><td>0.04904</td><td>0.06108</td></tr><tr><td>partial-KV</td><td>643.2</td><td>28.34</td><td>0.04407</td><td>0.05468</td></tr></table>

At prefixes 128 and 256, complete-request energy falls by 25.9% and 24.4%, respectively. The reduction therefore persists when prompt ingestion is included. These builds preserve the weights and pass integerreference validation. The changes quantify the combined implementation progression, without separately identifying thermal efects and every control change.

The two steps remove diferent work. Active-row gathering omits weight accesses for zero events; partial KV loading avoids moving unused state before the cache fills. Faster sparse projections expose KV-loading waits that were previously overlapped with computation, so the second change releases further execution benefit. The p32 decode rate rises from 544.2 to 643.2 tokens/s, whereas p480 changes from 400.5 to 399.7, consistent with the disappearance of avoidable transfers at a full ring.

Table 3: Mechanisms and evidence levels. Denominators and workloads difer across rows; the percentages are not interchangeable energy reductions.
<table><tr><td>Evidence level</td><td>Result</td><td>Scope</td></tr><tr><td>ARM projections</td><td>51.8–59.7% of MAC terms skipped</td><td>Nonzero-input loops, three prefixes</td></tr><tr><td>FPGA weight reads</td><td>52% fewer weight bytes</td><td>Simulation on the tested vectors</td></tr><tr><td>FPGA end-to-end</td><td>24.4–27.7% less request energy</td><td>Same-checkpoint builds, card estimate</td></tr></table>

Table 3 connects representational opportunity, execution behavior, and board measurements. ARM counters show 51.8%, 56.5%, and 59.7% fewer projection MAC terms at the three prefixes. FPGA simulation records 52% fewer weight bytes on its tested vectors. An earlier full-model prefill trace counts 159,850,985 dense-equivalent and 80,909,678 active MACs/token, a 49.38% reduction. These counts explain removable work; measured card energy evaluates the resulting implementation.

## Same-model FPGA speed and energy

175 MHz | 128 outputs | headline change: prefix 32, partial KV vs. dense

+35.5%

-27.6%

Partial KV -27.7% less request energy

(a) Decode throughput

![](images/6fb573ff9a26e57931db64a51cc7c9ba6c40749af348568d7007d166e3313a9c.jpg)  
(b) Decode energy

![](images/08d8293884a5373b974450c38ce0cb0201a5c55669037ce70daab40f542a5f50.jpg)

![](images/1afe7344dbec2e7681b5a7397f379dd577624bf569e881fa862a333497448153.jpg)  
Figure 3: Same-checkpoint FPGA builds at 175 MHz. Panels show decode throughput, decode energy, and complete request energy; each request generates 128 tokens. Throughput and decode energy are three-round medians; request energy is median power divided by median request throughput. Error bars on the two sparse builds span per-round minima and maxima; the dense baseline shows only the available summary values. Energy per generated token is estimated from card power and throughput; thermal conditions difer.

## 5.2 Gross energy and the loaded-idle share

The independent partial-KV idle-split experiment generates 31,104 tokens over 243 repetitions, with all recorded output sequences exact to the reference. It measures 643.03 tok/s, 28.534 W over active-run windows, and 23.623 W loaded idle. Equation (8) gives

$$
E _ { \mathrm { g r o s s } } \approx 0 . 0 4 4 3 7 \mathrm { ~ J / t o k e n } , E _ { \mathrm { i n c r e m e n t a l } } \approx 0 . 0 0 7 6 4 \mathrm { ~ J / t o k e n } , E _ { \mathrm { i d l e ~ s h a r e } } \approx 0 . 0 3 6 7 4 \mathrm { ~ J / t o k e n } .\tag{11}
$$

As shown in Figure 4, about 82.8% of gross card energy in this experiment is the loaded-idle share. Faster generation therefore reduces the idle energy allocated to each token, even if active power changes little. The approximately 4.91 W increment measures activity above loaded idle on the FPGA card.

The partial-KV idle split followed an extended test campaign: FPGA temperature increased from 74 to $7 7 ^ { \circ } \mathrm { C }$ and HBM from 69 to $7 2 ^ { \circ } \mathrm { C }$ . The earlier sparse-gather run used a cooler card with approximately 22.4 W idle. That run measured about 0.049 J/token gross and 0.0074 J/token incremental energy; the partial-KV run measured 0.0444 and 0.0076, respectively. A comparison of these runs under diferent thermal conditions shows lower gross energy and similar incremental energy for partial-KV. It does not isolate dynamic-energy savings or attribute the improvement solely to fewer arithmetic operations.

## 5.3 The same V2 deployment on ARM and FPGA

Both gross and incremental energy show lower generation cost on U50C within the measured boundaries (Table 4 and fig. 5). At prefix 32, complete-request U50C energy is 0.05468 J/generated token gross and

Loaded-idle share

## Where card energy is spent

Independent prefix-32 / 128-output measurements | temperatures differ

![](images/d13a11a52323f4c0e346c322f3073bcc9c56f26ec0461dd783ca58df27effe22.jpg)  
Increment above idle

![](images/2b3d26a5be3675bb3cc20457b863da39a78a2587eabc679070906bf45d7ba12d.jpg)  
Figure 4: Independent p32/n128 continuous idle-split runs. Left: gross card energy separates into a loaded-idle share and activity above idle; top labels are totals and dark-segment labels are increments, all in J/generated token. Right: loaded-idle and active power for partial KV. Thermal conditions difer across builds, so the chart is not an isolated dynamic-energy ablation.

0.00912 J/generated token above loaded idle, versus 0.15086 and 0.08633, respectively, on ROCK 5T. Both metrics retain this ordering at prefixes 128 and 256. U50C draws more active power but completes requests faster, reducing energy per generated token. These results compare the recorded deployment boundaries; they do not give a whole-system FPGA eficiency ratio including the host and supply losses.

Incremental energy is particularly sensitive to the idle baseline. Its three-round relative spreads on U50C are 2.16%, 3.19%, and 2.22% at the three prefixes, versus 2.57%, 6.32%, and 3.00% on ROCK 5T. Both platforms include prefill in this comparison. The U50C complete-request increment of 0.00912 J/token therefore belongs to a diferent measurement from the independent decode-only increment of 0.00764 J/token in the preceding subsection.

With the model and deployment window fixed, the dedicated implementation provides higher decode and complete-request throughput (Table 4). At p32 the ARM runtime reaches 65.4 generated tokens/s per complete request, and the FPGA 518.4. The FPGA handles prompt tokens sequentially but processes projections and

Table 4: Complete-request throughput and energy for the same V2 weights, integer arithmetic, inputs, and 475-window setting; each request generates 128 tokens. ARM includes an extra forward step after selecting the final output. ROCK 5T measures AC input including its adapter; U50C estimates card energy excluding the host. Incremental energy subtracts the same-run loaded-idle baseline. FPGA increments are derived from retained records; aggregation is specified in Section 4.
<table><tr><td rowspan="2">Prefix</td><td colspan="3">ROCK 5T: AC input</td><td colspan="3">U50C: card sensor</td></tr><tr><td>Request tok/s</td><td>Gross J/token</td><td>Increment J/token</td><td>Request tok/s</td><td>Gross</td><td>Increment</td></tr><tr><td>32</td><td>65.4</td><td>0.15086</td><td>0.08633</td><td>518.4</td><td>J/token</td><td>J/token</td></tr><tr><td>128</td><td>37.5</td><td>0.25107</td><td>0.14091</td><td>304.8</td><td>0.05468 0.09274</td><td>0.00912 0.01497</td></tr><tr><td>256</td><td>23.7</td><td>0.37727</td><td>0.20152</td><td>187.3</td><td>0.14989</td><td>0.02316</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## U50C and ROCK 5T: energy per generated token

Same integer model | 128 outputs | three rounds | measurement boundaries differ

ROCK 5T: adapter AC input

![](images/3ab9d847e597886271f5982d7b520dd7c0c39bedcdd78a2946f20836402da8c8.jpg)

U50C: card sensor

![](images/1e1222b91a106fa8ea8ba27fc952a50b47e45f0a8a8329ce23bb528ab84fdd9b.jpg)  
Figure 5: Complete-request energy for the same integer model, generating 128 tokens. Left: gross energy; right: incremental energy above loaded idle, both in J/generated token. ROCK 5T integrates adapter AC input; U50C estimates card energy from power and request throughput. Each prefix has three rounds, aggregated as in Table 4; panel scales difer. Subtracting idle does not equalize the two measurement boundaries.

state through its resident data plane; p256 prompt ingestion takes 424.6 ms, versus approximately 3.000 s on ARM. This comparison measures the combined efect of hardware capacity, memory organization, and implementation.

Table 5: U50C partial-KV FPGA, continuous decode, batch one, 128 generated tokens. Three-round medians of host-observed throughput and mean XRT card power; energy is the DC-card estimate of Section 4, obtained by dividing mean power by decode or request throughput.
<table><tr><td>Prefix</td><td>Decode tok/s</td><td>Request tok/s</td><td>Card W</td><td>Decode J/token</td><td>Request J/token</td></tr><tr><td>32</td><td>643.2</td><td>518.4</td><td>28.34</td><td>0.04407</td><td>0.05468</td></tr><tr><td>128</td><td>573.2</td><td>304.8</td><td>28.26</td><td>0.04930</td><td>0.09274</td></tr><tr><td>256</td><td>494.9</td><td>187.3</td><td>28.08</td><td>0.05674</td><td>0.14989</td></tr><tr><td>480</td><td>399.7</td><td>103.3</td><td>27.60</td><td>0.06905</td><td>0.26730</td></tr></table>

Table 5 gives the FPGA energy estimates. Mean card power varies by less than 1 W across prefixes, so longer context raises energy mainly by increasing token time. The p32 three-round throughput spread is 0.04%, and an independent synchronized measurement gives 643.5 tok/s. The p480 request values are derived from the measured request throughput and the same card power. Across the three-round series of both sparse builds, decode-energy spreads range from 0.47 to 1.72%.

## 5.4 Interactive operation

Interactive partial-KV generation reaches 231.8 tok/s, or 4.31 ms/token, over ten 128-token turns with power sampling disabled. Ten separate turns with power sampling enabled measure 23.32 W, giving a split-run gross estimate of 0.1006 J/token. The active-minus-idle diference is too small to resolve reliably across these runs,

## Time spent ingesting and generating tokens

Same integer model | 128 outputs | platform-specific vertical scales

![](images/fb62daf21524986896a14b0e9aa2e8c9b2fd64820609eec84842421f48b62e06.jpg)

![](images/541adb2c119c74dd49b2812db8424107a50fdf3f5529a55c9b06254292ed786e.jpg)  
Figure 6: Prompt ingestion and generation time for the same integer model, with 128 generated tokens. Blue denotes prefill and green decode; top labels give their summed time in seconds and the prefill share. Platform axes use diferent scales. Each phase uses its median, so the sum can difer slightly from the independently aggregated whole-request median.

so incremental interactive energy is not reported. This per-token-resume workflow includes a host round trip for each output, unlike continuous device-side decoding.

Separate board-cycle diagnostics report 519.66, 603.46, and 730.25 forward passes/s for the three builds; these exclude application overhead and are not interactive token rates. The dedicated one-token-resume test records 239.43 tokens/s, separately from the interactive campaign above.

## 5.5 Card energy against the recorded GPU deployment

Figure 7 and table 6 compares the integer FPGA deployment with the recorded RTX 5090 compiled-FP32 baseline for the same checkpoint. At p32/n128, FPGA continuous decode reaches 643.2 tokens/s and 0.04407 J/generated token, versus 406.9 tokens/s and 0.40315 J/generated token on the GPU. Here, compiled FP32 uses the PyTorch torch.compile optimizer while retaining FP32 arithmetic. The FPGA delivers 1.58× the throughput with 89.1% lower estimated gross decode energy. Including prefill gives 0.05468 versus 0.40466 J/generated token, a 7.40× energy ratio.

This comparison matches checkpoint, input tokens, and output length, but uses diferent arithmetic precisions and uncalibrated cross-device sensors; output identity between FP32 and integer arithmetic is not required. The short workload stays within both attention windows, while longer workloads difer in visible context. Compiled FP32 is the fastest recorded GPU path, not the minimum-energy configuration. Full protocols, precision screening, and the paired architecture experiment are in Section B.

Measured GPU low-precision alternatives. Lower precision can reduce GPU energy even when it does not increase throughput. Table 7 compares compiled FP32, bfloat16 (BF16), and weight-only INT8 on another RTX 5090, using the same V2 checkpoint. Each setting is measured twice at each of three prefix lengths (32, 128, and 256 tokens), always generating 128 tokens with batch one. Each run starts with three warmup requests, followed by at least 30 seconds of timed requests. Timing includes prefill and synchronizes once per

## FPGA / GPU card-energy estimates

Same weights | integer FPGA / compiled-FP32 GPU | ratios: GPU / FPGA

(a) Continuous decode

(b) Complete request  
![](images/eddd15ad905f0ebc5f5dea768f3451341229f049ef1fbf740d5502496d88f63f.jpg)

![](images/22d128a4d85972e870bc548760c44347a3589bbdde96b308646c14de9228ec6a.jpg)  
Figure 7: Estimated card energy for two deployments of the same weights. Left: continuous decode; right: complete request including prefill, with 128 generated tokens. Bar labels are J/generated token; upper labels are GPU/FPGA energy ratios. U50C uses integer arithmetic and RTX 5090 compiled FP32. The comparison does not establish equal-quality eficiency or a GPU energy lower bound.

Table 6: Continuous decode, 128 generated tokens, batch one. U50C: partial-KV INT8-weight integer deployment, hostobserved timing of physical hardware. RTX 5090: FP32 compiled path. Energy is gross board-sensor power/throughput. Three-round medians.
<table><tr><td colspan="5">U50C partial-KV</td><td rowspan="2">RTX 5090 J/token</td><td rowspan="2">Energy ratio GPU/FPGA</td></tr><tr><td>Prefix</td><td>tok/s</td><td>W</td><td>J/token</td><td>tok/s</td></tr><tr><td>32</td><td>643.2</td><td>28.34</td><td>0.04407</td><td>406.9</td><td>0.40315</td><td>9.15×</td></tr><tr><td>128</td><td>573.2</td><td>28.26</td><td>0.04930</td><td>411.2</td><td>0.41116</td><td>8.34×</td></tr><tr><td>256</td><td>494.9</td><td>28.08</td><td>0.05674</td><td>406.3</td><td>0.42301</td><td>7.46×</td></tr></table>

generated sequence. For each setting, throughput and NVML card power are separately averaged with equal weight across the six runs. Reported energy is mean power divided by mean throughput; individual runs’ energy ratios are not averaged.

Table 7: Selected GPU precision settings for the same V2 checkpoint, measured in one paired campaign on an RTX 5090. Complete requests with 128 generated tokens; prefixes of 32, 128, and 256 tokens, two rounds per setting. Power and throughput are arithmetic means over six runs. Energy is mean card power divided by mean request throughput, in J/generated token. The GPU and measurement protocol difer from those in Table 6.
<table><tr><td>GPU configuration</td><td>Request tok/s</td><td>Card W</td><td>J/token</td></tr><tr><td>Compiled FP32</td><td>319.8</td><td>162.0</td><td>0.5065</td></tr><tr><td>Compiled BF16</td><td>312.1</td><td>141.1</td><td>0.4522</td></tr><tr><td>Compiled weight-only INT8</td><td>251.7</td><td>112.6</td><td>0.4475</td></tr></table>

Weight-only INT8 lowers gross card energy per generated token by 11.7% relative to FP32 within this campaign, calculated before rounding, despite lower throughput. It quantizes weights and retains floating-point activation computation; its arithmetic therefore difers from the integer rules used throughout the FPGA inference implementation. These results establish a measured GPU energy improvement from low precision under the tested conditions. They do not establish an optimized-GPU energy lower bound, and their cross-prefix summaries cannot be substituted into the per-prefix FPGA/GPU ratios in Table 6. The 89.1% headline reduction remains a comparison against the recorded compiled-FP32 deployment.

## 5.6 Complete-request performance on ROCK 5T

Table 8: V2 on four A76 cores of a ROCK 5T. Three-round medians; each request generates 128 tokens. Energy integrates complete requests at the board adapter’s AC input and excludes external cooling. Each column is reduced independently; gross and incremental energy are in J/token.
<table><tr><td colspan="7">Energy (J/token)</td></tr><tr><td>Prefix</td><td>Request tok/s</td><td>Active W</td><td>Idle W</td><td>Gross</td><td>Incremental</td><td>Request s</td></tr><tr><td>32</td><td>65.4</td><td>9.80</td><td>4.22</td><td>0.151</td><td>0.086</td><td>1.957</td></tr><tr><td>128</td><td>37.5</td><td>9.41</td><td>4.12</td><td>0.251</td><td>0.141</td><td>3.416</td></tr><tr><td>256</td><td>23.7</td><td>8.94</td><td>4.18</td><td>0.377</td><td>0.202</td><td>5.392</td></tr></table>

Table 8 shows complete local requests at approximately 9–10 W. At p32, one 128-token response takes 1.96 s and 19.3 J. At p128 and p256, request energy rises to 32.1 and 48.3 J despite slightly lower mean active power. Gross-energy spreads are 2.16%, 3.15%, and 0.40%, respectively. Independent integration of the nine active and eighteen idle windows reproduces the reported energy estimates.

The prefix sensitivity is visible in measured phase timing. Median prefill time grows from 0.282 to 1.335 to 3.000 s, while decode grows from 1.675 to 2.081 to 2.394 s. Prompt ingestion accounts for about 14% of the short request and 56% of the long request. Decode throughput alone would therefore hide a substantial edge-runtime cost. The primary ARM energy result includes both phases.

## 5.7 Same-board CPU and NPU deployment tradeofs

Table 9: Same ROCK 5T, complete requests, 128 generated tokens. V2 was remeasured without model or code changes; the five comparison deployments retain their earlier same-day measurements. All rows report three-round medians. All energies are gross AC J/generated token. CPU means four A76 cores; NPU means RKLLM W8A8, with CPU support. Tokenizers, model sizes, inputs, and task quality difer. Bold identifies the V2 deployment.
<table><tr><td colspan="3">Request tok/s</td><td colspan="3">J/token (lower is better)</td></tr><tr><td>Model / path</td><td>(higher is better) p32 p128</td><td>p256</td><td>p32</td><td>p128</td><td>p256</td></tr><tr><td>V2 194M / INT8 CPU</td><td>65.4</td><td>37.5 23.7</td><td>0.151</td><td>0.251</td><td>0.377</td></tr><tr><td>Qwen 0.5B / Q8_0 CPU</td><td>35.5</td><td>31.8 27.4</td><td>0.375</td><td>0.413</td><td>0.473</td></tr><tr><td>Qwen 0.5B / W8A8 NPU</td><td>37.7</td><td>36.1 34.0</td><td>0.259</td><td>0.269</td><td>0.291</td></tr><tr><td>SmolLM2 135M / Q8_0 CPU</td><td>112.5</td><td>93.7 74.8</td><td>0.112</td><td>0.134</td><td>0.165</td></tr><tr><td>SmolLM2 135M / W8A8 NPU</td><td>61.7</td><td>57.4</td><td>53.6 0.158</td><td>0.169</td><td>0.181</td></tr><tr><td>LFM2.5 350M / Q8_0 CPU</td><td>49.5</td><td>42.9</td><td>36.6</td><td>0.270 0.297</td><td>0.347</td></tr></table>

Table 9 and fig. 8 compare the tested deployments. V2 uses 59.8%, 39.2%, and 20.2% less gross energy than the Qwen CPU path at the three prefixes, respectively. At p32 it is also faster and uses less energy than Qwen NPU and LFM CPU. Its p32 result is close to SmolLM2 NPU: 65.4 versus 61.7 tok/s and 0.151 versus 0.158 J/token, with essentially identical incremental energy near 0.086 J/token. The meter’s uncertainty prevents interpreting this small gross-energy diference as a reliable eficiency advantage.

SmolLM2 CPU is faster and more energy-eficient at every prefix, despite its higher active board power. At p256, V2 uses more energy than both NPU paths and LFM CPU, and its request throughput falls more sharply with prefix length. The ARM implementation is therefore most competitive on short requests in this comparison. Model size, quantization, and runtime all difer across rows, so the comparison places V2 among deployable options; it does not isolate the efect of sparse execution (Section 6).

## Six deployment paths on ROCK 5T

Complete requests | 128 outputs | three-round medians | models differ

<table><tr><td rowspan=1 colspan=4>(a) Energy (J/token)</td></tr><tr><td rowspan=1 colspan=1>SL-V2 194M / CPU</td><td rowspan=1 colspan=1>0.151</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.377</td></tr><tr><td rowspan=1 colspan=1>Qwen 0.5B / CPU</td><td rowspan=1 colspan=1>0.375</td><td rowspan=1 colspan=1>0.413</td><td rowspan=1 colspan=1>0.473</td></tr><tr><td rowspan=1 colspan=1>Qwen 0.5B / NPU</td><td rowspan=1 colspan=1>0.259</td><td rowspan=1 colspan=1>0.269</td><td rowspan=1 colspan=1>0.291</td></tr><tr><td rowspan=1 colspan=1>SmolLM2 135M / CPU</td><td rowspan=1 colspan=1>0.112</td><td rowspan=1 colspan=1>0.134</td><td rowspan=1 colspan=1>0.165</td></tr><tr><td rowspan=1 colspan=1>SmolLM2 135M / NPU</td><td rowspan=1 colspan=1>0.158</td><td rowspan=1 colspan=1>0.169</td><td rowspan=1 colspan=1>0.181</td></tr><tr><td rowspan=1 colspan=4>LFM2.5 350M / CPU32        128       256Prefix tokens</td></tr></table>

![](images/af001d570b84de52b329059cce8ffc4cbbb4f59d90cbee302b212b1cf5d683bb.jpg)  
Figure 8: All six ROCK 5T deployment paths, complete requests, 128 generated tokens. Left: AC-input energy across three prefixes, with lighter cells indicating lower energy. Right: request throughput at prefix 32; points are three-round medians and intervals span minima to maxima. V2 was remeasured without model or code changes; other deployments retain their earlier same-day measurements. Models, tokenizers, and runtimes difer.

## 6 Discussion and Scope

The main experiment evaluates the cost of executing a fixed checkpoint. Active-row gathering removes work associated with zero inputs, while valid-state KV loading reduces state transfer. Together with the measured loaded-idle share, these mechanisms explain the observed energy changes. Cross-model ARM results and diferent-precision FPGA/GPU results describe deployment tradeofs; they do not isolate the energy contribution of event coding alone.

Extension to other dedicated hardware. The U50C prototype provides an implementation basis for dedicated hybrid neuromorphic V2 hardware. Its acceleration and energy-saving mechanisms are applicable across hardware implementations. Nonzero-event computation and weight fetching, valid-state KV transfers, resident weights, and on-device generation can also be implemented on other FPGAs or application-specific integrated circuits (ASICs). A target must support both graded signed events and the retained residual, normalization, and attention operations. The U50C measurements demonstrate that these mechanisms can be combined into a complete inference system, providing a design basis for other V2-compatible accelerators.

A precise condition connects this portability to performance and energy. Consider a dedicated implementation � and a reference implementation � of the same V2 weights, arithmetic, inputs, context window, and output length. Let � denote complete-request generation throughput, $\bar { P }$ the mean active power over that request interval, and � gross energy per generated token. With matching timing definitions and device measurement boundaries, Equation (8) yields

$$
\frac { E _ { H } } { E _ { B } } = \frac { \bar { P } _ { H } / \bar { P } _ { B } } { R _ { H } / R _ { B } } , \qquad \frac { R _ { H } } { R _ { B } } > \mathrm { m a x } \biggr ( 1 , \frac { \bar { P } _ { H } } { \bar { P } _ { B } } \biggr ) \implies R _ { H } > R _ { B } \mathrm { a n d } E _ { H } < E _ { B } .\tag{12}
$$

Thus, increasing throughput by a larger factor than the change in active power delivers both faster generation and lower energy per generated token. This relation uses power and throughput from the same measurement interval; it does not directly combine diferent device boundaries or independently reduced table medians.

Whether another target meets this condition depends on the computation, memory trafic, and time saved relative to event-indexing, scheduling, and data-reordering overheads, as well as the capacity and bandwidth available to the dataflow. KV-transfer savings also depend on cache occupancy. The resulting inference is conditional but extends beyond the U50C: other V2-compatible dedicated implementations have a mechanism-supported route to faster, lower-energy generation, with the realized gain determined by hardware design and workload.

The deployed checkpoint has held-out FP32 perplexity 14.75 versus 12.50 for the same-budget dense control, approximately 18% higher, and exhibits reduced accuracy on a key task suite. The same-weight implementation upgrades leave this model unchanged: they lower execution cost without establishing equal-quality eficiency. Full quality results, the attention variant, and quantization diagnostics are reported in Section A.

The FPGA uses one sequence, a finite KV ring, and greedy output. On-board non-greedy sampling and full-logit readout are not implemented. The runtime disables early EOS in continuous multi-token mode because of a recorded state fault; per-token interaction supports exact stopping. ARM measurements exclude external cooling, and FPGA/GPU measurements exclude host and supply losses. Incremental FPGA energy is not used as an ASIC energy projection.

## 7 Related Work

SymbolicLight V1 (Liu, 2026b) and its spike-aware CPU follow-up (Liu, 2026a) motivate the transition from trainable sparse activations to executable sparse inference. SpikeGPT (Zhu et al., 2024) studies directly trained spiking language modeling, while SpikeLM (Xing et al., 2024) studies spike-driven language modeling with elastic bi-spiking. V2 continues V1’s hybrid design and extends event computation while retaining continuous-state processing. Its focus here is the physical execution and energy of a trained checkpoint on an ARM edge CPU and a sparse FPGA.

Hybrid neuromorphic hardware provides a complementary architectural context. Tianjic (Pei et al., 2019) supports artificial and spiking neural-network computation on a fully digital platform with reconfigurable building blocks and hybrid coding. V2 uses the hybrid principle within a language model: sparse events coexist with recurrent state and a continuous residual, and its U50C FPGA prototype maps their computation and data movement into end-to-end inference. The measured energy benefit follows from the implemented execution mechanisms, rather than from biological analogy.

Integer-only inference (Jacob et al., 2018; Kim et al., 2021) provides the broader setting for deterministic low-precision execution. Our integer reference supplies a common numerical target for successive FPGA implementations, with quantization quality evaluated separately. ALiBi (Press et al., 2022) provides the position-bias mechanism used by the event attention path.

FPGA text-generation systems such as DFX (Hong et al., 2022) and FlightLLM (Zeng et al., 2024) investigate customized hardware execution and mapping for language models. V2 studies how event-coded projection inputs enable active-row gathering, with measurements of both application throughput and card energy. Its architecture and workload difer from these systems, precluding direct cross-paper eficiency ranking. The supplementary GPU implementation uses the compilation system of PyTorch 2 (Ansel et al., 2024).

## 8 Conclusion

SymbolicLight V2 advances V1’s spike-gated dual-path design into a hybrid neuromorphic language architecture with executable sparse inference. Sparse events and continuous-state processing work together in an end-to-end U50C FPGA prototype and an ARM integer implementation. Holding model weights fixed, selective weight access and valid-state KV loading increase short-context FPGA decode throughput by 35.5% while reducing estimated gross card decode energy by 27.6%. Complete-request energy falls by 24.4–27.7% across three prefixes. An independent idle split shows how reduced token latency lowers the platform energy allocated to each output. AC-input measurements on four ARM cores additionally demonstrate local inference at approximately 9–10 W. For the same integer model, the U50C deployment also generates faster than the ROCK 5T CPU implementation, with lower gross and idle-subtracted incremental energy per token within the tested workloads and respective measurement boundaries.

These mechanisms provide a basis for V2 implementations on other dedicated hybrid neuromorphic hardware. Nonzero-event computation and memory access, valid-state transfers, and on-device generation are transferable; when their throughput gain exceeds the relative change in active power, other compatible implementations can likewise achieve faster generation and lower energy per token. U50C validates this design path, while the realized benefit depends on compute resources, memory organization, and control overhead.

## References

Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, et al. PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS), 2024. doi: 10.1145/3620665.3640366. URL https://docs.pytorch.org/ assets/pytorch2-2.pdf.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439, 2020. doi: 10.1609/aaai.v34i05.6239. URL https://ojs.aaai.org/ index.php/AAAI/article/view/6239.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. URL https://arxiv.org/abs/1803.05457.

Seongmin Hong, Seungjae Moon, Junsoo Kim, Sungjae Lee, Minsub Kim, Dongsoo Lee, and Joo-Young Kim. DFX: A low-latency multi-FPGA appliance for accelerating transformer-based text generation. In

IEEE/ACM International Symposium on Microarchitecture (MICRO), 2022. URL https://arxiv.org/ abs/2209.10797.

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for eficient integerarithmetic-only inference. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2704–2713, 2018. URL https://openaccess.thecvf.com/content\_cvpr\_2018/ html/Jacob\_Quantization\_and\_Training\_CVPR\_2018\_paper.html.

Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, and Kurt Keutzer. I-BERT: Integer-only BERT quantization. In Proceedings ofthe 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pages 5506–5518. PMLR, 2021. URL https://proceedings.mlr.press/v139/kim21d.html.

Ting Liu. Spike-aware INT8 execution for spiking language models on commodity CPUs. arXiv preprint arXiv:2606.03026, 2026a. URL https://arxiv.org/abs/2606.03026v2.

Ting Liu. SymbolicLight V1: Spike-gated dual-path language modeling at high encoder spike sparsity. arXiv preprint arXiv:2605.21333, 2026b. URL https://arxiv.org/abs/2605.21333v3.

Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 36(6):51–63, 2019. doi: 10.1109/MSP.2019.2931595. URL https://arxiv.org/ abs/1901.09948.

Jing Pei, Lei Deng, Sen Song, Mingguo Zhao, Youhui Zhang, Shuang Wu, Guanrui Wang, Zhe Zou, Zhenzhi Wu, Wei He, Feng Chen, Ning Deng, Si Wu, Yu Wang, Yujie Wu, Zheyu Yang, Cheng Ma, Guoqi Li, Wentao Han, Huanglong Li, Huaqiang Wu, Rong Zhao, Yuan Xie, and Luping Shi. Towards artificial general intelligence with hybrid Tianjic chip architecture. Nature, 572(7767):106–111, 2019. doi: 10.1038/s41586-019-1424-8. URL https://www.nature.com/articles/s41586-019-1424-8.

Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations (ICLR), 2022. URL https://arxiv.org/abs/2108.12409.

Xingrun Xing, Zheng Zhang, Ziyi Ni, Shitao Xiao, Yiming Ju, Siqi Fan, Yequan Wang, Jiajun Zhang, and Guoqi Li. SpikeLM: Towards general spike-driven language modeling via elastic bi-spiking mechanisms. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings ofMachine Learning Research, pages 54698–54714. PMLR, 2024. URL https://proceedings.mlr. press/v235/xing24d.html.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics (ACL), pages 4791–4800, 2019. doi: 10.18653/v1/P19-1472. URL https://aclanthology. org/P19-1472/.

Shulin Zeng, Jun Liu, Guohao Dai, Xinhao Yang, Tianyu Fu, Hongyi Wang, Wenheng Ma, Hanbo Sun, Shiyao Li, Zixiao Huang, Yadong Dai, Jintao Li, Zehao Wang, Ruoyu Zhang, Kairui Wen, Xuefei Ning, and Yu Wang. FlightLLM: Eficient large language model inference with a complete mapping flow on FPGAs. In ACM/SIGDA International Symposium on Field Programmable Gate Arrays (FPGA), 2024. URL https://arxiv.org/abs/2401.03868.

Rui-Jie Zhu, Qihang Zhao, Guoqi Li, and Jason K. Eshraghian. SpikeGPT: Generative pre-trained language model with spiking neural networks. Transactions on Machine Learning Research, 2024. URL https: //openreview.net/forum?id=gcf1anBL9e.

## A Model Quality

## A.1 Same-budget architecture comparison

Three 194M-scale models were trained from scratch using the same open training data, tokenizer with a 48K vocabulary, random seed, batch settings, and 4B-token budget. The dense GPT-2 control has 22 layers; the two models based on V2 have 12 layers. The continuous-attention control, denoted SL-V2-Cont, disables the event encoders added in V2 and uses continuous Q/K/V with softmax attention, while retaining the binary-spike sites inherited from V1. It is therefore a control for the added V2 mechanisms, not an entirely non-spiking model.

Table 10: FP32 quality of the trained models. PPL denotes perplexity and BPB bits per byte. Programmatic and downstream accuracies use byte-normalized continuation scoring. The hardware implementations use the same V2 checkpoint.
<table><tr><td>Model</td><td>PPL</td><td>Natural BPB</td><td>Programmatic acc.</td><td>Downstream macro acc.</td></tr><tr><td>GPT-2 194M</td><td>12.502221</td><td>1.736573</td><td>0.335938</td><td>0.368302</td></tr><tr><td>SL-V2-Cont</td><td>12.851092</td><td>1.774259</td><td>0.434896</td><td>0.362301</td></tr><tr><td>SL-V2</td><td>14.749522</td><td>1.820616</td><td>0.234375</td><td>0.356452</td></tr></table>

The programmatic suite contains 768 four-way items generated after training, covering arithmetic, symbolic ordering, progressions, and Python integer expressions. The natural-text test set contains 90 texts whose sources are separate from the training data; the downstream aggregate covers PIQA (Bisk et al., 2020), HellaSwag (Zellers et al., 2019), and ARC-Easy (Clark et al., 2018). Relative to the same-topology control, V2’s programmatic accuracy drops by 20.05 percentage points, failing the prespecified key-task criterion: each key task must retain at least 80% of the continuous control’s score and lose no more than 10 percentage points. Its full-model PPL increase over GPT-2 also exceeds the prespecified 15% threshold for stopping further model scaling. These results fail the prespecified quality criteria, and the hardware changes do not alter that outcome. Programmatic accuracy of 0.234375 is below the four-way chance level of 0.25; the downstream macro accuracies of all three models are close to the aggregate chance level of approximately 0.33, limiting their separation at this scale.

The separate SL-V2-Softmax research variant restores causal softmax and removes the second projected Q/K/V eventization. Its PPL on the main held-out set is 13.426754, 4.48% above SL-V2-Cont, but it is not the checkpoint deployed in these FPGA measurements. Its result is not combined with V2 energy to construct an equal-quality eficiency ratio.

## A.2 Quantization diagnostic and its scope

A separate 5,657-token test set compares FP32 with the software integer reference. PPL is 90.19 versus 91.65, BPB 1.9787 versus 1.9857, top-1 accuracy 30.32% versus 29.84%, and top-5 accuracy 47.80% versus 47.73%. The relative PPL increase is approximately 1.6%; greedy argmax agreement is 72.65%. This panel is distinct from the main held-out set, so its absolute PPL is not directly comparable with Table 10.

These are software-reference diagnostics. The software reference uses the training attention-window configuration, whereas the deployed FPGA build uses the extended deployment ring and its bitstream does not expose full logits. The panel therefore does not measure deployed perplexity across deployment context lengths. Token and state agreement with the integer deployment reference supplies a separate correctness check.

## B Supplementary GPU Measurements

## B.1 GPU protocol and precision boundary

The RTX 5090 baseline uses compiled FP32, PyTorch 2.8.0/CUDA 12.8, TF32 disabled, KV caching, batch one, and three rounds. FPGA measurements followed five days later with the same weights, inputs, output lengths, and greedy decoding, but integer arithmetic. The FPGA window is 475 (480 slots including anchors), versus 256 on GPU. p32/n128 fits both windows; p256/n128 crosses the training window, giving diferent visible contexts. Identical FP32 and integer output sequences are not required.

Additional tests covered eager FP32, compiled BF16, TorchAO weight-only INT8/INT4, dynamic FP8, and custom cuBLASLt INT8. Custom INT8 succeeded. TorchAO dynamic W8A8 failed because its selected kernel required more than 16 matrix rows, while single-stream decode supplied one; this does not imply that the GPU lacks INT8 support. None exceeded compiled FP32 throughput in batch-one screening, although BF16 and weight-only INT8 used less energy in the separate experiment of Table 7. CUDA graph capture attempts failed; TensorRT-LLM was not tested. The 9.15× energy ratio compares against the fastest tested FP32 baseline, not the lowest-energy GPU implementation.

Gross card energy is average active sensor power divided by generated-token throughput. The focused FPGA rerun uses XRT/xbutil telemetry at approximately 0.5 Hz; the GPU measurement uses NVML at 10 Hz. Runs last at least 60 seconds. Host and AC-supply costs are excluded; the sensors have not been cross-calibrated with an external meter.

## B.2 Additional decode results

The principal comparison is in Section 5.5 and table 6. At p32, FPGA throughput has a three-round spread of 0.04%; an independent measurement gives 643.5 tokens/s, close to the focused rerun of 643.2. No matched p480 GPU row was measured.

## B.3 Prompt ingestion

Table 11: Whole-request generated-token throughput and power-based gross energy estimates. The denominator includes prefill; the numerator remains 128 generated tokens. U50C power is taken from the corresponding continuous run, rather than separately integrated over each request.
<table><tr><td colspan="3">U50C partial-KV</td><td colspan="2">RTX 5090 FP32</td><td rowspan="2">Energy ratio GPU/FPGA</td></tr><tr><td>Prefix</td><td>tok/s</td><td>J/token</td><td>tok/s</td><td>J/token</td></tr><tr><td>32</td><td>518.4</td><td>0.05468</td><td>406.5</td><td>0.40466</td><td>7.40×</td></tr><tr><td>128</td><td>304.8</td><td>0.09274</td><td>408.7</td><td>0.41878</td><td>4.52x</td></tr><tr><td>256</td><td>187.3</td><td>0.14989</td><td>404.4</td><td>0.42522</td><td>2.84×</td></tr></table>

At p256, sequential FPGA prompt ingestion takes 424.6 ms; the GPU’s batched prompt forward has a recorded time to first token of 3.25 ms. These intervals difer, so their quotient is not a speedup. Table 11 compares complete requests including prefill.

## B.4 Paired GPU architecture control

A separate paired experiment on another RTX 5090 compares the V2 topology with its added event and attention changes disabled or enabled. With compiled FP32, SL-V2-Cont reaches 353.9 tok/s and 0.4706 J/token, versus 319.8 tok/s and 0.5065 J/token for SL-V2: throughput falls 9.6% and energy rises 7.6%. The normalizer and projected-event path change together, so this is not an isolated event-encoder ablation. Added zeros do not automatically improve the tested dense GPU execution.

The precision comparison from this same complete-request campaign is reported in Table 7. BF16 and weight-only INT8 reduce card power and energy despite lower throughput. The table specifies the cross-prefix aggregation and the use of another GPU, preventing these results from being conflated with the baseline in Table 6.