# WeAct: Dynamic Weight Switching as Agent Actions

Rui Shi<sup>1\*</sup> Xunjian Yin<sup>2\*</sup> Jiacheng Sang<sup>2\*</sup> Zichuan Liu<sup>1</sup> Shuyan Zhou<sup>2,3</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>Duke University <sup>3</sup>Meta MSL

{ruishi,zichuanl}@andrew.cmu.edu

{xunjian.yin,jiacheng.sang,shuyan.zhou}@duke.edu

## Abstract

Long-horizon agents must learn heterogeneous capabilities from interleaved trajectories. Updating shared parameters to learn new capabilities can degrade previously acquired ones, resulting in catastrophic forgetting. As a trajectory unfolds, different stages may require different capabilities, yet conventional joint training and static adapter merging maintain a fixed parameter configuration throughout execution. We introduce WeAct (Weight-level Internal Actions), which trains independent capabilityspecific LoRA adapters over a frozen backbone and adds adapter selection to the agent’s action space. These internal actions use the model’s native tool-calling interface for ondemand capability composition in a shared context. Across eight deep-research benchmarks, QWEN3-32B with WeAct improves average accuracy by 6.2% relative to full fine-tuning counterparts despite using less compute, and by 27.5% relative to the strongest static LoRAmerging method. WeAct also supports modular continual adaptation, by adding a new code computation adapter to expand the agent’s capabilities and updating the existing writing adapter to improve report quality. Both up dates leave the backbone and other adapters unchanged, supporting modular continual adaptation in agents.

## 1 Introduction

Long-horizon agents plan and invoke tools, connect evidence across sources and reasoning steps, and retrieve and synthesize web information over many turns. The required capabilities vary across stages and must adapt as domains, tools, and output requirements evolve. Fine-tuning or reinforcement learning (Li et al., 2026a; Hu et al., 2025) typically optimize these capabilities in shared parameters: misaligned capability gradients can impair joint learning (Li et al., 2026b), and later updates can erode earlier gains (Thede et al., 2026). Promptlevel skills (Ni et al., 2026) offer adaptation without changing weights, but their instructions consume context when loaded for execution.

![](images/a2af7811ca2cb463fc505aea49abe5652c1a3a597f02e8b1cdf7908720ef4bcb.jpg)  
Figure 1: Internal actions extend the agent’s action space. WeAct selects LoRA adapters internally and interacts with the environment through external actions.

Separate LoRA adapters isolate capability parameters, but their composition adds deployment requirements. MoLE (Wu et al., 2024) and LD-MoLE (Zhuang et al., 2026) mix expert outputs through layerwise gates, requiring support for composition inside the forward pass. Static merging, from conflict reduction in TIES (Yadav et al., 2023) to alignment in KnOTS (Stoica et al., 2025), combines updates before deployment. These methods improve composition, but use fixed merged weights across stages, which may weaken the specialization needed at each stage. Our key idea is to make independently maintained capabilities callable as agent actions. Extending ReAct’s external actions (Yao et al., 2023), we use internal actions to activate learned weights with on-demand flexibility.

We introduce WeAct (Weight-level Internal Actions; Figure 1), a general method for capability composition in tool-using agents, to learn heterogeneous capabilities for long-horizon tasks while preserving prior gains and invoking the capability needed at each stage. Each capability LoRA is trained over a frozen backbone on its own stages and exit actions, using preceding interactions as context. It learns to continue prior work, reason, call tools, and hand back control. At inference, our load\_lora and offload\_lora tools activate and release one adapter through the native toolcalling interface, with their descriptions and control exchanges recorded in context. The selected weights govern subsequent reasoning and tool use within the same history. WeAct-soft chooses capabilities from this history; WeAct-hard prescribes loading every adapter in the workflow for reliable invocation. Both use the same interface without a separately trained router. Adding or refreshing a capability updates only its adapter.

Experiments on QWEN3-32B test capability separation during learning and execution. Controlled diagnostics show that extra shared capacity does not prevent lost gains and that tested static merges reduce specialists’ target-text fit (Section 3). Across eight deep-research benchmarks, WeActhard reaches 45.47% average accuracy, a 6.2% relative improvement over Full SFT, and both variants outperform all tested static merges on every benchmark (Section 4.2). WeAct uses fewer trainable parameters and less nominal training compute than Full SFT while retaining the evaluated general capabilities. Code insertion expands the library with stable performance, and Writing refresh improves report quality on ResearchQA and DeepResearchBench (Section 4.5). These results suggest potential for modular continual adaptation through capability expansion and local refinement.

Our contributions are:

• We present WeAct for independent capability training and coordinated invocation via native tools in a shared history.

• We distinguish learning interference from static fusion mismatch, motivating capabilityaligned training and intact adapter selection.

• We demonstrate improved deep-research performance alongside capability insertion and local refinement, supporting continual learning through modular expansion and updates.

## 2 Method

## 2.1 Preliminaries and Overview

Given a research query q, an agent gathers evidence and answers through successive reasoning and tool interactions. We train capability adapters independently and select them as the task unfolds. Let $\theta _ { 0 }$ denote a frozen language model’s parameters and $\{ \Delta \theta _ { k } \} _ { k = 1 } ^ { K }$ a capability adapter library. Let M index the linear maps adapted by LoRA. For a map $\ell \in \mathcal { M }$ with input and output dimensions $d _ { \mathrm { i n } , \ell }$ and $d _ { \mathrm { o u t } , \ell } .$ , LoRA (Hu et al., 2022) gives

$$
\begin{array} { c } { { W _ { k , \ell } = W _ { 0 , \ell } + \frac { \alpha _ { k } } { r _ { k } } B _ { k , \ell } A _ { k , \ell } , } } \\ { { A _ { k , \ell } \in \mathbb { R } ^ { r _ { k } \times d _ { \mathrm { i n } , \ell } } , \quad B _ { k , \ell } \in \mathbb { R } ^ { d _ { \mathrm { o u t } , \ell } \times r _ { k } } . } } \end{array}\tag{1}
$$

Here $r _ { k }$ and $\alpha _ { k }$ are the rank and scaling coefficient. The trainable factors $\phi _ { k } = \{ A _ { k , \ell } , B _ { k , \ell } \} _ { \ell \in \mathcal { M } }$ determine $\Delta \theta _ { k } = \Delta \theta _ { k } ( \phi _ { k } ) ;$ ; different capabilities may use different ranks. We write $\pi _ { k } = \pi _ { \theta _ { 0 } + \Delta \theta _ { k } }$ and set $\Delta \theta _ { \mathcal { O } } = 0$ for base-only generation.

At step t, history $h _ { t }$ retains instructions, model turns, and tool outputs. With at most one adapter active, $z _ { t } \in \{ \varnothing , 1 , \ldots , K \}$ selects the policy:

$$
a _ { t } \sim \pi _ { z _ { t } } ( \cdot \mid h _ { t } ) = \pi _ { \theta _ { 0 } + \Delta \theta _ { z _ { t } } } ( \cdot \mid h _ { t } ) .\tag{2}
$$

$a _ { t }$ may contain reasoning or tool calls; internal tool calls change $z _ { t }$ for subsequent generation.

Figure 2 connects training and inference. During training, stage-specific assistant targets and exit calls supervise independent adapters over retained execution context. We group deep-research stages into planning, evidence search and verification, and writing, yielding LoRA-plan, LoRA-search, and LoRA-write. These stages separate deciding what to do next, acquiring and checking external evidence, and synthesizing the answer. In exploratory annotation, finer splits were unstable: separating retrieval from verification or adding a staterecap stage let small rule changes move the same evidence operation between labels, and folding retrieval into planning made Plan absorb evidencegathering steps (Appendix A.2). We therefore keep evidence operations together in LoRA-search. During inference, external tools supply observations and internal actions select weights, with both exchanges extending the shared history. WeActhard prescribes adapter invocation for reliable use; WeAct-soft selects capabilities according to the evolving task state in that history.

## 2.2 Capability Learning with Execution Context

Shared updates risk gradient interference and catastrophic forgetting. We isolate capability updates in separate adapters to prevent one capability from overwriting another’s parameters. Consider shared trainable parameters ω and capability losses $\mathcal { L } _ { j } ^ { \mathrm { s h a r e d } }$ . With $g _ { i } = \nabla _ { \omega } \mathcal { L } _ { i } ^ { \mathrm { s h a r e d } } ( \omega )$ for $i \in \{ j , k \}$ a gradient step for capability k changes capability $j ^ { \dag } \mathrm { s }$ loss to first order as

![](images/4e1ad22f79ad6d0590c62ec33754ad7ae6a8f6947de13a91001f77d9f3da6060.jpg)  
Figure 2: WeAct overview. (a) Capability learning builds the adapter library. (b) Tool-mediated inference switches adapters through internal tool calls and interacts with the environment through external tools. B/P/S/W nodes denote base/Plan/Search/Writing states; solid and dashed arrows indicate selected and alternative execution paths.

$$
\begin{array} { r l } & { \mathcal { L } _ { j } ^ { \mathrm { s h a r e d } } ( \omega - \eta g _ { k } ) - \mathcal { L } _ { j } ^ { \mathrm { s h a r e d } } ( \omega ) } \\ & { \quad \quad = - \eta g _ { j } ^ { \top } g _ { k } + O ( \eta ^ { 2 } ) , } \end{array}\tag{3}
$$

for a locally smooth loss and step size η. The alignment and magnitude of these gradients determine the first-order interaction; Section 3 measures the resulting retention effects.

We define capabilities by stage function, each potentially spanning several assistant/tool exchanges. LoRA-plan develops goals and action plans; LoRA-search covers evidence acquisition, filtering, and verification; LoRA-write synthesizes evidence into an answer.

Each training sample retains preceding work so the module learns to continue an ongoing task. For capability $k ,$ let $p$ be the retained preceding history, $u _ { k }$ the loading call and result, $c _ { k }$ the stage’s interleaved assistant/tool content, and $v _ { k }$ its terminal assistant exit call. Concatenating these tokenized segments gives

$$
\begin{array} { c } { { s = p \oplus u _ { k } \oplus c _ { k } \oplus v _ { k } = ( x _ { 1 } , \ldots , x _ { N } ) , } } \\ { { m _ { k , n } = \mathbf { 1 } \{ n \in \mathcal { A } _ { k } ( s ) \} . } } \end{array}\tag{4}
$$

The set $\boldsymbol { \mathcal { A } } _ { k } ( \boldsymbol { s } )$ contains the assistant-token positions used to supervise the current capability, including those corresponding to the exit call. The

loss mask is zero for earlier assistant turns, the loading exchange, and tool observations. For the stage dataset $\mathcal { D } _ { k } .$ , each module minimizes the objective:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { k } ( \phi _ { k } ) = - \mathbb { E } _ { s \sim \mathcal { D } _ { k } } \displaystyle \sum _ { n = 1 } ^ { N } m _ { k , n } \log \pi _ { k } ( x _ { n } \mid x _ { < n } ) , } \\ & { \quad \quad \quad \phi _ { k } ^ { \ast } \in \arg \displaystyle \operatorname* { m i n } _ { \phi _ { k } } { \mathcal { L } } _ { k } ( \phi _ { k } ) . } \end{array}\tag{5}
$$

The mask selects prediction targets; all retained tokens remain causal context. In training and inference, WeAct retains the initial task and latest five user/tool messages, replacing older observation bodies with placeholders. We provide a complete training example in Appendix A.1.

For $j \neq k , \nabla _ { \phi _ { i } } \mathcal { L } _ { k } = 0 \colon$ optimizing capability k leaves $\phi _ { j }$ and $\theta _ { 0 }$ unchanged, avoiding the shared-parameter interference in Eq. 3. Refreshing a capability updates only its adapter factors, preserving the backbone and other modules’ parameters without data replay, as in the Writing refresh in Section 4.5.2.

## 2.3 Tool-Mediated Weight Actions

Separate adapters require a way to select capabilities as task needs change. We expose weight selection through native tool calls, reusing the model’s learned tool selection and argument generation without a separately trained router. This interface fits existing tool harnesses and lets new capabilities enter through registry entries.

Our harness extends MiroFlow (Su et al., 2026) with two MCP tools, load\_lora and offload\_lora. Both take a registry alias, lora\_name, as their argument. Each call a and result r extend the history, $h  h \oplus a \oplus r$ . After validation and successful execution, load/offload calls update the active index:

```perl
Algorithm 1 WeAct inference.
Require: $q , \{ \Delta \theta _ { k } \} _ { k = 1 } ^ { K } ,$ WeAct-hard/soft policy
1: $h  q ; z  \emptyset$
2: while budget remains do
3: a ∼ $\pi _ { z } ( \cdot \mid h ) ; h  h \oplus a$
4: for each tool call c in a do
5: $r  c ( ) ; h  h \oplus r$
6: if c = load_lora(k) then
7: z ← k
8: else if $c = \circ \pounds \pounds 1$ oad_lora(k) then
9: z ← ∅
10: end if
11: end for
12: Apply WeAct-hard/soft to (h, z)
13: if answer ready and permitted return answer
14: end while
15: return budget-exhaustion result
```

$$
z _ { t } ^ { \prime } = \left\{ \begin{array} { l l } { k , } & { a _ { t } = \mathrm { L o a d } ( k ) , z _ { t } = \emptyset , } \\ { \emptyset , } & { a _ { t } = \mathrm { O f f o a d } ( k ) , z _ { t } = k , } \\ { z _ { t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{6}
$$

Thus switching from i to j releases i before loading $j .$ External, failed, rejected, and idempotent calls leave $z _ { t }$ unchanged; reloading the active alias or offloading an empty state is a no-op. The harness checks aliases, lifecycle rules, and policy constraints, returning status, errors, or stage guidance. These changes to the weights governing subsequent generation are Weight-level Internal Actions.

The policies differ in invocation timing. Under WeAct-soft, the base selects a module and the active module releases control, with revisits subject to coverage, repetition, and termination constraints. WeAct-hard prescribes adapter use in the same agent, by default LoRA-plan→ LoRA-search→LoRA-write. The harness advances when the active stage emits an exit call, exhausts its budget, or reaches the no-tool threshold, recording release/load exchanges and handling termination or base-model fallback. Algorithm 1 outlines both.

Both control policies use the same execution mechanism: load and offload calls activate and release adapters that remain resident on the GPU, avoiding repeated host-to-device transfers. Weight changes require rebuilding the KV cache from the retained history under the new weights (Appendix B.3); answer-ready latency remains lower than Full SFT in the controlled evaluation (Section 4.4). The same tool interface also supports new capabilities: registering a new adapter with an alias and a capability description makes it available for selection, as in the Code insertion in Section 4.5.1.

![](images/8f824b7a280761ff920dba48add3539483865c3e8522a037b9b178a361ae34ef.jpg)  
Figure 3: Capability retention under sequential training. Detailed results are reported in Appendix Table 7.

## 3 Understanding Capability Separation

Learning heterogeneous capabilities through shared parameters, can erode earlier gains. Training separate LoRAs isolates their updates, but static fusion can alter each specialist’s effects during execution. We use controlled retention and fusion diagnostics to examine these two sources of interference and motivate WeAct’s capability-aligned parameter partitioning during learning and separate adapter selection during execution.

## 3.1 Capability Interference and Retention

We test whether shared updates erode earlier capabilities (Eq. 3). Plan, Search, and Writing define successive trajectory stages; sequential training models capability-specific fine-tuning after initial SFT (Section 4.5), as in sequential continual LoRA learning (Wang et al., 2025).

On frozen QWEN3-32B, we train these capabilities in all six orders under identical data and schedules. Shared LoRAs of rank 32 or 96 match the active rank or total parameter count of three rank-32 adapters, testing whether extra capacity suffices. The partitioned adapters are assigned by capability, as in WeAct, or randomly per trajectory, testing whether separation suffices without capability alignment.

Capability-aligned adapters remain unchanged during later stages, providing a fixed reference for capability drift. We report teacher-forced top-1 content-token accuracy on 168 validation trajectories, with a separate 24-trajectory monitoring subset (one seed; Appendix C); gradient diagnostics are in Appendix D.

The shared adapter matching the library’s total parameter count has a mean absolute accuracy loss of 1.67% across capabilities; extra capacity does not reduce this loss (Table 7). Writing gains least initially yet loses 49.2% of that gain (Figure 3). Random partitioning still incurs a mean absolute accuracy loss of 1.12%, so separation without capability alignment does not prevent forgetting.

## 3.2 Capability Mismatch under Static Fusion

We test whether static fusion preserves a fixed library of Plan, Search, and Writing specialists, each trained on its own capability. Full Additive sums their updates without final rank compression; TIES (Yadav et al., 2023), DARE-TIES (Yu et al., 2024; Yadav et al., 2023), and TSPA (Li et al., 2026c) produce rank-64 updates. The reference, Individual, selects the adapter matching each record’s capability label. We reserve 200 records per capability as validation data and measure paired mean target-token negative log-likelihood (NLL) gaps under teacher forcing (fusion minus Individual). We also measure local mismatch through $( \Delta W _ { m } - \Delta W _ { k } ) h$ on the same input activation $h ,$ where $\Delta W _ { m }$ is the merged update and $\Delta W _ { k }$ is the individual adapter update for capability k. For Full Additive, this difference is exactly the combined contribution of the other capability adapters. Validation data and metric details are in Appendix E.

None of the tested fusions eliminates the mean NLL degradation on any of the three capabilities (Figure 14). Local mismatch persists across LoRA layers in all three capabilities (Figure 13). Full Additive exhibits this mismatch despite retaining all updates without rank compression. Thus, combining intact updates can alter a specialist’s local effect through contributions from other capabilities. These results support preserving the specialists as separate updates and selecting them at runtime.

## 4 Experiments

We evaluate WeAct on eight deep-research benchmarks, then analyze evidence use in execution traces and training and inference efficiency. We also test capability insertion and local updates to assess modular continual adaptation.

## 4.1 Experimental Setup

Model and adapters. Weight-adaptation experiments start from QWEN3-32B (Yang et al., 2025): LoRA methods freeze the backbone; Full SFT updates it. Prompting baselines use the original checkpoint. The main WeAct library contains LoRA-plan, LoRA-search, and LoRA-write with ranks 32, 64, and 32. Single LoRA uses rank 128, matching the combined parameter count of the three separately trained adapters; static merges use rank 64. Only capability expansion uses LoRA-code. Other training and deployment settings, including learning rates and epochs, are in Appendix F.

Training data. We combine public agent trajectories from MiroVerse-v0.1 (Team et al., 2025) with supplementary trajectories generated from OpenSeeker-v1-Data (Du et al., 2026b) questions using our execution framework. Each LoRA adapter’s training samples retain execution context and supervise only its capability-specific outputs and handoff (Sec. 2.2). Appendix A summarizes data sources and construction; continual adaptation studies describe additional Code and Writing data.

Benchmarks. We cover general tool use, multihop evidence composition, and web research. General uses GAIA (Mialon et al., 2024); Multi-Hop QA comprises 2WIKIMULTIHOPQA (Ho et al., 2020), BAMBOOGLE (Press et al., 2023), and MUSIQUE (Trivedi et al., 2022). Search comprises WEBWALKERQA (Wu et al., 2025), XBENCH-DEEPSEARCH (xbench Team, 2025), BROWSEC-OMP (Wei et al., 2025), and FRAMES (Krishna et al., 2025), covering web navigation, retrieval, hard-to-find facts, and multi-source integration.

## 4.2 Main Results

Table 1 compares four adaptation mechanisms: prompting and multi-agent orchestration (Re-Act (Yao et al., 2023), Least-to-Most (Zhou et al., 2023a), and ManuSearch (Huang et al., 2025)); unified adaptation (Full SFT, Single LoRA); static fusion (TIES-Merging (Yadav et al., 2023), DARE-TIES (Yu et al., 2024; Yadav et al., 2023), and TSPA (Li et al., 2026c); Appendix E); and WeActhard/soft. Multi-agent methods keep their own context and model-call structures. Static fusion and WeAct share one capability library, fused into a fixed update or selected at runtime.

WeAct-hard has the highest average, 45.47%, an accuracy gain of 2.67 points over Full SFT, and ranks first on six benchmarks. WeAct-soft has the second-highest average (43.21%) and leads on GAIA and MuSiQue. The hard result supports independent LoRA learning with prescribed, stepwise invocation for reliable adapter use within one agent. Soft flexibly schedules independently trained Lo-RAs within the harness’s runtime constraints.

<table><tr><td rowspan="2">Method</td><td>General</td><td colspan="3">Multi-Hop QA</td><td colspan="4">Search</td><td rowspan="2">Avg.</td></tr><tr><td>GAIA</td><td>2Wiki</td><td>Bam- boogle</td><td>MuSiQue</td><td>Web WalkerQA</td><td>xbench- DS</td><td>Browse Comp</td><td>FRAMES</td></tr><tr><td colspan="10">PROMPT &amp; MULTI-AGENT-BASED</td></tr><tr><td>ReAct</td><td> $3 5 . 9 2 _ { \pm 4 . 7 3 }$ </td><td> $4 6 . 2 0 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $5 3 . 6 0 _ { \pm 4 . 4 6 }$ </td><td> $1 3 . 9 8 _ { \pm 0 . 7 1 }$ </td><td> $2 7 . 3 5 _ { \pm 1 . 7 1 }$ </td><td> $3 5 . 0 0 { \scriptstyle \pm 4 . 7 7 }$ </td><td> $4 . 0 0 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $4 8 . 6 7 _ { \pm 1 . 7 4 }$ </td><td> $3 3 . 0 9 _ { \pm 1 . 0 9 }$ </td></tr><tr><td>Least-to-Most</td><td> $3 7 . 8 6 _ { \pm 4 . 7 8 }$ </td><td> $4 7 . 4 0 _ { \pm 1 . 5 8 }$ </td><td> $4 8 . 0 0 { \scriptstyle \pm 4 . 4 7 }$ </td><td> $1 4 . 4 0 _ { \pm 0 . 7 1 }$ </td><td> $2 7 . 2 1 \pm 1 . 7 1$ </td><td> $3 9 . 0 0 { \scriptstyle \pm 4 . 8 8 }$ </td><td> $5 . 0 0 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $5 1 . 7 0 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $3 3 . 8 2 _ { \pm 1 . 1 0 }$ </td></tr><tr><td>ManuSearch</td><td> $\underline { { 4 1 . 7 5 } } \pm 4 . 8 6$ </td><td> $\underline { { 5 3 . 5 0 } } \pm 1 . 5 8$ </td><td> $6 0 . 8 0 _ { \pm 4 . 3 7 }$ </td><td> $1 6 . 8 4 _ { \pm 0 . 7 6 }$ </td><td> $2 8 . 3 8 { \scriptstyle \pm 1 . 7 3 }$ </td><td> $4 5 . 0 0 { \scriptstyle \pm 4 . 9 7 }$ </td><td> $7 . 5 0 { \scriptstyle \pm 1 . 8 6 }$ </td><td> $6 2 . 1 4 _ { \pm 1 . 6 9 }$ </td><td> $3 9 . 4 9 _ { \pm 1 . 1 2 }$ </td></tr><tr><td colspan="10">SINGLE-MODEL WEIGHT ADAPTATION</td></tr><tr><td>Single LoRA</td><td> $3 5 . 9 2 _ { \pm 4 . 7 3 }$ </td><td> $4 9 . 7 0 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $5 4 . 4 0 _ { \pm 4 . 4 5 }$ </td><td> $1 5 . 3 5 _ { \pm 0 . 7 3 }$ </td><td> $3 5 . 1 5 _ { \pm 1 . 8 3 }$ </td><td> $5 1 . 0 0 { \scriptstyle \pm 5 . 0 0 }$ </td><td> $1 2 . 5 0 { \scriptstyle \pm 2 . 3 4 }$ </td><td> $5 5 . 8 3 _ { \pm 1 . 7 3 }$ </td><td> $3 8 . 7 3 { \scriptstyle \pm 1 . 1 3 }$ </td></tr><tr><td>Full SFT</td><td> $\underline { { 4 1 . 7 5 } } \pm 4 . 8 6$ </td><td> $5 0 . 9 0 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $6 5 . 6 0 { \scriptstyle \pm 4 . 2 5 }$ </td><td> $1 5 . 7 6 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $\underline { { 3 8 . 5 3 } } \pm 1 . 8 7$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 5 . 0 0 }$ </td><td> $1 3 . 5 0 { \scriptstyle \pm 2 . 4 2 }$ </td><td> $6 6 . 3 8 { \scriptstyle \pm 1 . 6 5 }$ </td><td> $4 2 . 8 0 { \scriptstyle \pm 1 . 1 3 }$ </td></tr><tr><td colspan="10">STATIC MULTI-LORA MERGING</td></tr><tr><td>TIES-Merging</td><td> $3 4 . 9 5 { \scriptstyle \pm 4 . 7 0 }$ </td><td>47.60±1.58</td><td> $5 5 . 2 0 { \scriptstyle \pm 4 . 4 5 }$ </td><td> $1 4 . 0 7 _ { \pm 0 . 7 1 }$ </td><td>34.41±1.82</td><td> $4 4 . 0 0 { \scriptstyle \pm 4 . 9 6 }$ </td><td> $7 . 0 0 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $4 8 . 0 6 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $3 5 . 6 6 \pm 1 . 1 1$ </td></tr><tr><td>DARE-TIES</td><td> $3 5 . 9 2 _ { \pm 4 . 7 3 }$ </td><td>48.00±1.58</td><td> $5 4 . 4 0 _ { \pm 4 . 4 5 }$ </td><td> $1 6 . 2 6 _ { \pm 0 . 7 5 }$ </td><td> $3 2 . 9 4 _ { \pm 1 . 8 0 }$ </td><td> $4 0 . 0 0 { \scriptstyle \pm 4 . 9 0 }$ </td><td> $5 . 0 0 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $5 0 . 2 4 _ { \pm 1 . 7 4 }$ </td><td> $3 5 . 3 5 _ { \pm 1 . 1 0 }$ </td></tr><tr><td>TSPA</td><td> $3 6 . 8 9 _ { \pm 4 . 7 5 }$ </td><td> $4 6 . 6 0 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $5 0 . 4 0 { \scriptstyle \pm 4 . 4 7 }$ </td><td> $1 4 . 7 3 _ { \pm 0 . 7 2 }$ </td><td> $2 7 . 2 1 \pm 1 . 7 1$ </td><td> $4 5 . 0 0 { \scriptstyle \pm 4 . 9 7 }$ </td><td> $7 . 0 0 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $5 3 . 4 0 _ { \pm 1 . 7 4 }$ </td><td> $3 5 . 1 5 _ { \pm 1 . 1 1 }$ </td></tr><tr><td colspan="10">WEACT (OURS)</td></tr><tr><td>WeAct-hard</td><td> $4 0 . 7 8 _ { \pm 4 . 8 4 }$ </td><td> ${ \bar { \mathbf { 5 6 . 3 0 } } } _ { \pm 1 . 5 7 }$ </td><td>69.60±4.11</td><td> $1 7 . 2 1 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $\mathbf { 4 0 . 0 0 } \mathrm { { \scriptstyle \pm 1 . 8 8 } }$ </td><td> $\pm 4 . 0 0 _ { \pm 4 . 9 8 }$ </td><td> $\mathbf { 1 7 . 5 0 } _ { \pm 2 . 6 9 }$ </td><td> ${ \bf 6 8 . 3 3 _ { \pm 1 . 6 2 } }$ </td><td> $4 5 . 4 7 _ { \pm 1 . 1 3 }$ </td></tr><tr><td>WeAct-soft</td><td> $\mathbf { 4 6 . 6 0 } _ { \pm 4 . 9 2 }$ </td><td>53.20±1.58</td><td> $6 2 . 4 0 _ { \pm 4 . 3 3 }$ </td><td> $\pm 7 . 5 8 _ { \pm 0 . 7 7 }$ </td><td> $3 7 . 7 9 _ { \pm 1 . 8 6 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 5 . 0 0 }$ </td><td> $1 5 . 5 0 { \scriptstyle \pm 2 . 5 6 }$ </td><td> $6 2 . 6 2 _ { \pm 1 . 6 9 }$ </td><td> $\underline { { 4 3 . 2 1 } } \pm 1 . 1 4$ </td></tr></table>

Table 1: Results on eight benchmarks (%). Subscripts give nominal task-based binomial standard-error approximations. Bold and underlined values denote the best and second-best scores per benchmark.

Both variants outperform every static merge on all eight benchmarks; WeAct-hard also outperforms Single LoRA on all eight, and WeActsoft on seven. Compared with Single LoRA, hard and soft achieve average-accuracy gains of 6.74% and 4.48%. The strongest static merge averages 35.66%, below Single LoRA’s 38.73%: these rank-64 fusions of the independently trained adapters, whose ranks sum to 128, perform worse than direct rank-128 training. This gap is consistent with the fusion mismatch in Section 3.2.

## 4.3 Search Effectiveness

To assess search effectiveness beyond answer accuracy, we apply SearchAtlas (Sang et al., 2026) to all ten methods on the same 100 tasks, producing 1,000 evidential query graphs. Query utilization (↑) measures the fraction of issued queries on an evidence path to the answer; constraint coverage (↑), the fraction of question constraints represented in those queries; and PK→answer (↓), the share of answers with content attributed to prior knowledge without retrieved support. Figure 4 reports the results; Appendix G gives formal definitions, graph construction, and task selection.

on all three metrics, ahead of Full SFT and the other baselines. Although ManuSearch issues the most queries, its answer-supporting queries cover only 39% of question constraints, versus 60% for WeAct-hard. These results show that WeAct connects search more effectively to answer support: a larger share of queries contributes to the answer, and those queries cover more question constraints. Fewer answers contain unsupported content attributed to prior knowledge.

## 4.4 Training and Inference Efficiency

WeAct-hard ranks first and WeAct-soft second

Training compute. We assess WeAct’s training efficiency. With the same Qwen3-32B backbone and fewer epochs per adapter, WeAct uses 142.8 nominal EFLOPs, below Full SFT’s training compute, while achieving higher average accuracy. We count repeated contexts, epochs, packing, dense operations, and causal attention, with 1.23× Full SFT’s token exposure. LoRA omits frozen-backbone weight gradients. Across unequal capability segments, 2.06× token exposure at 0.95× per-token compute yields 1.96× total compute relative to Single LoRA with matched total trainable parameters (Appendix B.1). These results support efficient training of independent capability adapters.

Inference efficiency. Figure 5 pairs Table 1 Search accuracy with historical mean cumulative model-path time on the same server with the same concurrency, harness, and tool conditions, including retries; General/Multi-hop QA appear in Figure 8 (Appendix B.2). Against Single LoRA, WeAct-soft is more accurate and faster on six benchmarks; hard improves accuracy on eight and time on seven. Across 50 runs per method on prespecified WebWalkerQA tasks with identical hardware and task concurrency one, soft and hard reduce mean answer-ready latency by 33.4% and 20.3% versus Full SFT; soft is within 1% of Single LoRA. Model and tool execution dominate; adapter control and cache handling occupy 8.2%/2.8% of soft/hard time (Appendix B.3). These results support accuracy gains with modest switching overhead and net latency savings.

![](images/73e2c20c9716f3286ee4a45119be5c7cdcf5f35aef0b1c2784eaf13d5fd0cd8e.jpg)

![](images/82185bc4bb60107e052b2a5f108de7864c9ecb60986261f531c5d1b81bdfa572.jpg)  
Figure 4: SearchAtlas trace diagnostics. Query utilization: fraction of queries supporting the answer; constraint coverage: fraction of question constraints covered by those queries; PK→answer: share of answers with unsupported content attributed to prior knowledge. Error bars are 95% task-bootstrap intervals.

## 4.5 Modular Continual Adaptation

We assess catastrophic forgetting and modular continual adaptation: expanding and refining the library while retaining existing capabilities. We first evaluate general-capability retention with external tools disabled (Appendix H), then examine Code insertion and a local Writing refresh.

## 4.5.1 Continual Capability Expansion

To add code computation, we train a new LoRA-code adapter on MiroVerse-MegaScience trajectories converted to our tool and handoff protocol, keeping the base and existing adapters fixed without replay. Table 2 shows that inserting LoRA-code can improve accuracy on in-domain MEGASCIENCE and on SCIBENCH (Wang et al., 2024a), without degrading DS-1000 (Lai et al., 2023) or GAIA (Mialon et al., 2024). Code insertion thus extends the library without retraining or harming existing capabilities (Appendix I).

## 4.5.2 Adapting to Evolving Writing Tasks

We test whether updating a single capability improves system-level report quality while preserving the rest of the library’s parameters. We continue training the existing LoRA-write adapter on trajectories synthesized in our harness from the OpenScholar (Asai et al., 2026) subset of DR Tulu (Shao et al., 2026) SFT data, replacing it with LoRA-write++. The base stays frozen, Plan and Search remain bitwise unchanged, and no historical data are replayed.

![](images/99dd8666fcd393dd48ee0a2ec29c7329828f7e173520b0b27846a2c4437a51bc.jpg)

Figure 5: Accuracy versus historical mean cumulative model-path time on four Search benchmarks. Times include saved retries.
<table><tr><td>Variant</td><td>MegaScience SciBench DS-1000 ∆GAIA</td><td></td><td></td><td></td></tr><tr><td>w/o LoRA-code</td><td>26.5</td><td>40.0</td><td>14.0</td><td>0.0</td></tr><tr><td>w/LoRA-code</td><td>29.0</td><td>40.5</td><td>16.0</td><td>+1.0</td></tr></table>

Table 2: Results of code-adapter insertion. ∆GAIA is relative to the original K = 3 system.

Table 3 shows gains on RESEARCHQA (Yifei et al., 2026) and DEEPRESEARCHBENCH (Du et al., 2026a) under hard and soft routing: hard gains 2.40 and 1.21 points, respectively. Both variants improve all four DRB dimensions, with soft gaining most in readability and instruction following (Appendix J, Table 11). Balancing new and historical Writing requirements within one module enables refinement without jointly retraining other capabilities. These gains support replay-free improvement without overwriting the backbone or other capability parameters.

<table><tr><td>Method</td><td></td><td>DeepResearch ResearchQA Bench</td></tr><tr><td rowspan="2">LoRA-write</td><td>hard</td><td>58.33 26.51</td></tr><tr><td>soft</td><td>54.14 27.92</td></tr><tr><td>LoRA-write++</td><td></td><td>hard 60.72 (+2.40) 27.71 (+1.21) soft 55.97 (+1.83) 28.52 (+0.61)</td></tr></table>

Table 3: Writing refresh scores. Bold values show gains over Original.

## 5 Related Work

Deep-Research Agent Training. WebSailor-V2 and Tongyi DeepResearch improve agent training through synthetic data and post-training (Li et al., 2026a; Tongyi DeepResearch Team et al., 2026). Step-DeepResearch and MiroThinker further organize learning around atomic capabilities (Hu et al., 2025; MiroMind Team et al., 2026). These methods share parameters across capabilities, coupling subsequent updates. Separating trainable parameters allows local refinement without updating the remaining library; WeAct realizes this organization as callable adapters over a frozen backbone.

Multi-Agent Orchestration. ManuSearch delegates planning, search, and reading to specialized agents; Chain of Agents distributes long contexts across workers (Huang et al., 2025; Zhang et al., 2024). Their handoffs require intermediate messages and a manager to integrate the workers’ outputs. In WeAct, moving control between capability weights on a shared backbone offers another form of specialization: each selected policy continues from the same interaction history.

Multi-LoRA Composition and Routing. Early work such as LoRAHub and MoLE established reusable adapter composition (Huang et al., 2024; Wu et al., 2024). LD-MoLE learns tokendependent layerwise gates (Zhuang et al., 2026); LoGo selects and merges adapters per input without additional training (Lee et al., 2026). These methods focus on combining experts for task inputs rather than managing capability handoffs during an evolving agent trajectory. Adaptive Minds keeps the base model in control when calling adapters as tools (Shekar and Krishnan, 2026), whereas WeAct transfers control to the selected adapter. Macaron-V1 selects specialists per user turn (Mind Lab et al., 2026). Building on these approaches, WeAct couples independent capability learning with explicit handoffs within an agent trajectory. Selected adapters control reasoning and external tool use from shared history until offloading.

## 6 Discussion

Research on preserving specialization spans the frozen columns of early Progressive Neural Networks (Rusu et al., 2016) and the separate reasoning and tool-use LoRAs of recent DART (Li et al., 2026b). Our capacity-matched and randompartition controls show that capability alignment matters beyond capacity; fixed-library diagnostics reveal local fusion mismatch even without rank compression (Section 3). These results distinguish preserving learned parameters from preserving their functional effects during composition. Together with LoRA memory analysis (Back et al., 2026), they support aligning modules with independent update needs and selecting intact adapters as execution demands. Internal actions make this principle operational: newly trained LoRAs can enter an existing agent through the same interface. This suggests modular continual learning, with retention within updated modules remaining open.

## 7 Conclusion

We presented WeAct, which learns heterogeneous capabilities in independent LoRAs over a frozen backbone and switches active weights through internal actions within a shared agent trajectory. Controlled diagnostics support capability-aligned training and runtime selection for preserving specialization. Across eight deep-research benchmarks, WeAct improves average accuracy over Full SFT with lower nominal training compute. Code insertion expands the library with stable performance, and Writing refresh improves report quality. These results support independently learning agent capabilities and switching between them during execution, with potential for modular continual learning.

## Limitations

Extending WeAct to sustained continual learning raises open questions about routing capacity and capability boundaries. It remains unclear how many adapters the agent can reliably distinguish and invoke as the library grows, and how well it can sustain longer sequences of internal actions. Repeated module insertion or refinement may also change capability selection, even when the rest of the library remains fixed. When task boundaries are unclear, general abilities such as reasoning may be needed across multiple stages, making it difficult to define separate LoRA modules and determine when each should be active. Whether incremental adapter expansion can improve these abilities while preserving performance on existing tasks remains an open question.

## References

Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’Arcy, David Wadden, Matt Latzke, Jenna Sparks, Jena D. Hwang, Varsha Kishore, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, and 9 others. 2026. Synthesizing scientific literature with retrieval-augmented language models. Nature, 650(8103):857–863.

Seungju Back, Dongwoo Lee, Naun Kang, Taehee Lee, S. K. Hong, Youngjune Gwon, and Sungjin Ahn. 2026. Understanding LoRA as knowledge memory: An empirical analysis. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 4761–4801. PMLR.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Licheng Zhang, Xiaorui Wang, and Zhendong Mao. 2026a. DeepResearch Bench: A comprehensive benchmark for deep research agents. In International Conference on Learning Representations, volume 2026, pages 42414–42448.

Yuwen Du, Rui Ye, Shuo Tang, Xinyu Zhu, Yijun Lu, Yuzhu Cai, and Siheng Chen. 2026b. OpenSeeker: Democratizing frontier search agents by fully opensourcing training data. In Third Conference on Language Modeling. Accepted at COLM 2026.

Run-Ze Fan, Zengzhi Wang, and Pengfei Liu. 2026. MegaScience: Pushing the frontiers of open posttraining datasets for science reasoning. In Third Conference on Language Modeling. Accepted at COLM 2026.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Chen Hu, Haikuo Du, Heng Wang, Lin Lin, Mingrui Chen, Peng Liu, Ruihang Miao, Tianchi Yue, Wang You, Wei Ji, Wei Yuan, Wenjin Deng, Xiaojian Yuan, Xiaoyun Zhang, Xiangyu Liu, Xikai Liu, Yanming Xu, Yicheng Cao, Yifei Zhang, and 48 others. 2025. Step-DeepResearch technical report. Preprint, arXiv:2512.20491.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. 2024. LoraHub: Efficient cross-task generalization via dynamic LoRA composition. In First Conference on Language Modeling.

Lisheng Huang, Yichen Liu, Jinhao Jiang, Rongxiang Zhang, Jiahao Yan, Junyi Li, and Wayne Xin Zhao. 2025. ManuSearch: Democratizing deep search in large language models with a transparent and open multi-agent framework. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 2403–2417, Suzhou, China. Association for Computational Linguistics.

Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. 2025. Fact, fetch, and reason: A unified evaluation of retrievalaugmented generation. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4745–4759, Albuquerque, New Mexico. Association for Computational Linguistics.

Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. 2023. DS-1000: A natural and reliable benchmark for data science code generation. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 18319–18345, Honolulu, Hawaii, USA. PMLR.

Seungeon Lee, Soumi Das, Manish Gupta, and Krishna P. Gummadi. 2026. LoRA on the go: Instancelevel dynamic LoRA selection and merging. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 39583–39601, San Diego, California, United States. Association for Computational Linguistics.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Xinmiao Yu, Jialong Wu, Xinyu Wang, Zile Qiao, Zhen Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Zhi-Qin John Xu, Shuai Wang, and 2 others. 2026a. WebSailor-V2: Bridging the chasm to proprietary agents via synthetic data and scalable reinforcement learning. In International Conference

on Learning Representations, volume 2026, pages 93919–93948.

Yu Li, Mingyang Yi, Xiuyu Li, Ju Fan, Fuxin Jiang, Binbin Chen, Peng Li, Jie Song, and Tieying Zhang. 2026b. Reasoning and tool-use compete in agentic RL: From quantifying interference to disentangled tuning. Preprint, arXiv:2602.00994.

Zijian Li, Xiachong Feng, Weitao Ma, Yichong Huang, Xiaocheng Feng, and Bing Qin. 2026c. Two-stage parameter alignment for multi-LoRA merging in large language models. In Findings of the Association for Computational Linguistics: ACL 2026, pages 30079–30093, San Diego, California, United States. Association for Computational Linguistics.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. GAIA: A benchmark for general AI assistants. In International Conference on Learning Representations, volume 2024, pages 9025–9049.

Mind Lab, Vin Bo, Asher Cai, Jingwei Cao, Song Cao, Vic Cao, Amelia Chen, Andrew Chen, Kaijie Chen, Cleon Cheng, Steven Chiang, Kaixuan Fan, Hera Feng, Huan Feng, Arthur Fu, Aaron Guan, Jun Gao, Pyke Han, Nolan Ho, and 63 others. 2026. Macaron-V1: Towards open continual learning with self-improvement and Mixture-of-LoRA. Preprint, arXiv:2608.09819.

MiroMind Team, S. Bai, L. Bing, L. Lei, R. Li, X. Li, X. Lin, E. Min, L. Su, B. Wang, L. Wang, L. Wang, S. Wang, X. Wang, Y. Zhang, Z. Zhang, G. Chen, L. Chen, Z. Cheng, and 25 others. 2026. MiroThinker-1.7 & H1: Towards heavyduty research agents via verification. Preprint, arXiv:2603.15726.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Trace2Skill: Distill trajectory-local lessons into transferable agent skills. Preprint, arXiv:2603.25158.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 5687–5711, Singapore. Association for Computational Linguistics.

Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. 2016. Progressive neural networks. Preprint, arXiv:1606.04671.

Jiacheng Sang, Mengyuan Li, Sanxing Chen, Yukun Huang, Yu Feng, and Bhuwan Dhingra. 2026. SearchAtlas: Analyzing agentic search strategies via evidential query graphs. In Findings of the Association for Computational Linguistics: EMNLP 2026. Association for Computational Linguistics. Accepted; to appear.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen-tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, and 2 others. 2026. DR Tulu: Reinforcement learning with evolving rubrics for deep research. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 110046–110101. PMLR.

Pavan C Shekar and Aswanth Krishnan. 2026. Adaptive minds: Empowering agents with LoRAs as tools. In 2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents at ICML 2026. Nonarchival workshop paper.

George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, and Judy Hoffman. 2025. Model merging with SVD to tie the Knots. In International Conference on Learning Representations, volume 2025, pages 4501–4519.

Shiqian Su, Sen Xing, Xuan Dong, Muyan Zhong, Bin Wang, Xizhou Zhu, Yuntao Chen, Wenhai Wang, Yue Deng, Pengxiang Zhu, Ziyuan Liu, Tiantong Li, Jiaheng Yu, Zhe Chen, Lidong Bing, and Jifeng Dai. 2026. MiroFlow: Towards high-performance and robust open-source agent framework for general deep research tasks. Preprint, arXiv:2602.22808.

MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, et al. 2025. Miro-Thinker: Pushing the performance boundaries of open-source research agents via model, context, and interactive scaling. Preprint, arXiv:2511.11793.

Lukas Thede, Stefan Winzeck, Zeynep Akata, and Jonathan Richard Schwarz. 2026. CapTrack: Multifaceted evaluation of forgetting in LLM posttraining. In The Fortieth Annual Conference on Neural Information Processing Systems, Evaluations and Datasets Track. Accepted, to appear.

Thinking Machines Lab. 2025. Tinker. Software and training service. Accessed 12 October 2026.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye, Wenbiao Yin, Xinmiao Yu, Xinyu Wang, and 38 others. 2026. Tongyi DeepResearch technical report. Preprint, arXiv:2510.24701.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554.

Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R. Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang.

2024a. SciBench: Evaluating college-level scientific problem-solving abilities of large language models. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 50622–50649. PMLR.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024b. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, volume 37, pages 95266–95290. Curran Associates, Inc.

Zhilin Wang, Yafu Li, Xiaoye Qu, and Yu Cheng. 2025. SEE: Continual fine-tuning with sequential ensemble of experts. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 7418–7432, Vienna, Austria. Association for Computational Linguistics.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. BrowseComp: A simple yet challenging benchmark for browsing agents. Preprint, arXiv:2504.12516.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025. Web-Walker: Benchmarking LLMs in web traversal. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10290–10305, Vienna, Austria. Association for Computational Linguistics.

Xun Wu, Shaohan Huang, and Furu Wei. 2024. Mixture of LoRA experts. In International Conference on Learning Representations, volume 2024, pages 47302–47318.

xbench Team. 2025. xbench-DeepSearch. Dataset. Accessed 12 October 2026.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. 2023. TIES-Merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems, volume 36, pages 7093–7115. Curran Associates, Inc.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language

models. In International Conference on Learning Representations.

Li S. Yifei, Allen Chang, Chaitanya Malaviya, and Mark Yatskar. 2026. ResearchQA: Evaluating scholarly question answering at scale across 75 fields with survey-mined questions and rubrics. Transactions of the Association for Computational Linguistics, 14:1365–1389.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language models are Super Mario: Absorbing abilities from homologous models as a free lunch. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 57755–57775. PMLR.

Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan Ö. Arık. 2024. Chain of agents: Large language models collaborating on long-context tasks. In Advances in Neural Information Processing Systems, volume 37, pages 132208– 132237. Curran Associates, Inc.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623. Curran Associates, Inc. Datasets and Benchmarks Track.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. 2023a. Least-to-most prompting enables complex reasoning in large language models. In International Conference on Learning Representations.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023b. Instruction-following evaluation for large language models. Preprint, arXiv:2311.07911.

Yuan Zhuang, Yi Shen, Yuexin Bian, Qing Su, Shihao Ji, Yuanyuan Shi, and Fei Miao. 2026. LD-MoLE: Learnable dynamic routing for mixture of LoRA experts. In International Conference on Learning Representations, volume 2026, pages 64840–64858.

## A Training Data Sources and Capability Segmentation

The training corpus contains 6,962 source trajectories before capability-specific splitting: 5,887 from MiroVerse-v0.1 (Team et al., 2025) and 1,075 generated from OpenSeeker-v1-Data (Du et al., 2026b) questions using our execution framework (Table 4).

<table><tr><td>Use</td><td>Source</td><td>Count</td></tr><tr><td>Main</td><td>MiroVerse-v0.1</td><td>5,887</td></tr><tr><td>Main</td><td>OpenSeeker-v1-Data</td><td>1,075</td></tr><tr><td>Main total</td><td></td><td>6,962</td></tr><tr><td>Code</td><td>MiroVerse-MegaScience</td><td>3,413</td></tr><tr><td></td><td>Writing++ DR-Tulu / OpenScholar</td><td>1,000</td></tr></table>

Table 4: Training-data sources. Main-corpus counts refer to source trajectories before capability splitting; Code and Writing++ counts refer to capability samples used in Section 4.5.

Full-SFT and capability training data. Full SFT uses all 6,962 whole trajectories, covering all capabilities and retaining the five most recent tool responses. For WeAct, capability-specific splitting yields 6,962 Plan, 7,437 Search, and 6,962 Writing samples, with 200 samples per capability reserved for validation. Each sample retains the preceding execution context and supervises only the selected capability’s assistant outputs and terminal offload call. The activation call and result belong to the masked prefix; all tool feedback is also masked from the loss.

Capability labels follow the stage’s purpose (Section 2.2): Plan sets the goals and retrieval agenda; Search retrieves evidence, inspects sources, and cross-checks claims; Writing synthesizes the supported answer. A span may include several assistant/tool turns. The MiroVerse export separates planning before the first search from the subsequent search interaction and final answer synthesis. For continual adaptation, Code expansion extracts 3,413 code-computation samples from MiroVerse-MegaScience, with 200 reserved for validation; Writing refresh constructs 1,000 continuation samples from the OpenScholar subset of DR-Tulu SFT data (Asai et al., 2026; Shao et al., 2026). These maintenance corpora are separate from the initial three adapters’ data.

Train–test overlap. MiroVerse does not document the original splits of its sources, so we check overlap at the task level. We compare the task text of all 6,962 training trajectories with every evaluation question of the eight benchmarks. No evaluation question appears verbatim after normalization. A lexical screen flags questions with at least half of their IDF-weighted content words in a single training task whose trajectory also contains the gold answer; manual review of the flagged pairs finds overlap only through decomposed sub-agent subtasks. For 63 of the 2,417 MuSiQue questions (2.6%), the answer-bearing sub-question appears with the same answer as a training subtask, and 19 of these restate the full question with the bridge entity resolved. One WebWalkerQA question and one 2Wiki question show similar overlap; GAIA, Bamboogle, XBench-DS, BrowseComp, and FRAMES show none. Removing the 63 MuSiQue questions lowers each method’s MuSiQue accuracy by 0.04– 0.47 points and leaves the ranking unchanged. The screen does not detect paraphrases below its threshold.

## A.1 A Search Training Example

Figure 6 shows a Search training sample for identifying which Republican candidates elected to the U.S. House of Representatives in 2008 previously served in the Minnesota House of Representatives. The boxed x contains the system and user prompts, the Plan stage, and the Search loading exchange. It ends with the Search activation response, shown in full with headings and numbering added for readability. The boxed y starts with the first Search assistant message and includes all five external exchanges and the final offload exchange. Its assistant tokens supply the targets in Eq. 5; x and all tool feedback have zero loss. With keep\_tool\_result=5, the three tool returns in this prefix are all retained. Long contents and external-call arguments are abbreviated while all 20 messages remain visible.

## A.2 Sensitivity to Capability Definitions and Annotation Rules

We explored capability definitions with GPT-5.4, starting from Retrieval, Reasoning, Verification, Plan, Code, and Writing. Paired comparisons use identical trajectories and assistant messages; changes count label-set differences, excluding boundary shifts within a label (Table 5).

Fine-grained labels and competing definitions. Separate Code from retrieval. Prioritizing execution increased Code and reduced Retrieval assignments. Absorb retrieval into

## y: Search continuation

```jsonl
[
{"role": "system",
"content": "[...] You may call exactly one function per
assistant response [...] return a JSON object [...]
within <tool_call></tool_call> XML tags:
<tool_call>{\"name\": \"<function−name>\",
\"arguments\": <args−json−object>}</tool_call>
[...] Your goal is to provide complete, accurate and
well−reasoned answers using additional tools. [...]"},
{"role": "user",
"content": "Check which of the Republican candidates elected
to the U.S. House of Representatives in 2008 previously
served in the Minnesota House of Representatives. [...]"},
{"role": "assistant", "content": "[...] <tool_call>
{\"name\": \"lora_runtime_mcp_server__load_lora\",
\"arguments\": {\"lora_name\": \"lora−plan\"}}</tool_call>"},
{"role": "tool", "content": "[...]"},
{"role": "assistant", "content": "[...] <tool_call>
{\"name\": \"lora_runtime_mcp_server__offload_lora\",
\"arguments\": {\"lora_name\": \"lora−plan\"}}</tool_call>"},
{"role": "tool", "content": "[...]"},
{"role": "assistant", "content": "<think>The plan is ready.
Now I need to gather evidence. I should activate the
search stage.</think><tool_call>
{\"name\": \"lora_runtime_mcp_server__load_lora\",
\"arguments\": {\"lora_name\": \"lora−search\"}}</tool_call
>"},
{"role": "tool", "content": "{
\"success\": true,
\"message\": \"LoRA session started: now uses 'lora−search'.
# Stage activation confirmed: lora−search
1. You are now responsible only for the current stage, not for
the whole task from scratch.
2. The base model has already selected this stage for the
current step.
3. Do not spend effort re−deciding global stage routing unless
the runtime state or the user instruction changes.
# Current stage objective:
Close the current information gap by searching, browsing,
reading, or scraping relevant external sources and extracting
the facts needed for the next step.
## High−value outputs before offload:
a relevant source, a retrieved fact set, or a short note
that the current information gap is closed and the next stage
can proceed
## Things usually not worth doing in this stage:
1. restarting broad planning when the current step mainly
needs evidence retrieval
2. jumping straight to polished final prose while the key
source or fact for this step is still missing
## Offload protocol
1. When the current stage deliverable is complete, call
`offload_lora` with `{\\\"lora_name\\\":
\\\"lora−search\\\"}` and stop immediately.
2. Do not combine that offload call with a final answer in the
same response.
3. When the current stage deliverable is complete, call:
<tool_call>
{\\\"name\\\": \\\"lora_runtime_mcp_server__offload_lora\\\",
\\\"arguments\\\": {\\\"lora_name\\\": \\\"lora−search\\\"}}
</tool_call>
4. After that offload tool call, stop immediately.
5. Do not combine the offload call with a final answer or a
second tool call.\",
\"active_lora\": \"lora−search\",
\"no_op\": false
}"}
```

[   
{"role": "assistant",   
"content": "[...]   
<tool\_call>   
{...}   
</tool\_call>"},   
{"role": "tool",   
"content": "[...]"},   
{"role": "assistant",   
"content": "[...]   
<tool\_call>   
{...}   
</tool\_call>"},   
{"role": "tool",   
"content": "[...]"},   
{"role": "assistant",   
"content": "[...]   
<tool\_call>   
{...}   
</tool\_call>"},   
{"role": "tool",   
"content": "[...]"},   
{"role": "assistant",   
"content": "[...]   
<tool\_call>   
{...}   
</tool\_call>"},   
{"role": "tool",   
"content": "[...]"},   
{"role": "assistant",   
"content": "[...]   
<tool\_call>   
{...}   
</tool\_call>"},   
{"role": "tool",   
"content": "[...]"},   
{"role": "assistant",   
"content": "<think>   
The search stage is   
complete. I will now   
offload the current   
LoRA and proceed to   
the next stage.   
</think>   
<tool\_call>   
{   
\"name\":   
\"lora\_runtime\_mcp\_   
server\_\_offload\_lora\",   
\"arguments\": {   
\"lora\_name\":   
\"lora−search\"   
}   
}   
</tool\_call>"},   
{"role": "tool",   
"content": "{   
\"success\": true,   
\"message\":   
\"LoRA session ended.   
Adapter unloaded   
from SGLang.\",   
\"active\_lora\": null,   
\"server\_unloaded\": true,   
\"no\_op\": false   
}"}   
]  
Figure 6: A training example for lora-search. Faded text is masked from the loss. Tool feedback is orange for internal actions and blue for external actions. [. . .] and {. . .} mark omitted text and arguments.

<table><tr><td>Policy change</td><td>Changed Spans</td><td>(before→after)</td></tr><tr><td>Strengthen targeted verification</td><td></td><td>4/80 Plan 20 → 20;</td></tr><tr><td>Narrow recap;</td><td></td><td>Verif. 21 → 21 36/78 Recap 30 → 12;</td></tr><tr><td>broaden verification</td><td></td><td>Verif. 9 → 41</td></tr><tr><td>Revise tool/span</td><td></td><td>11/78 Evid. 19 → 19;</td></tr><tr><td>rules</td><td></td><td>Verif.  $3 8  3 7$ </td></tr></table>

Table 5: Policy sensitivity on fixed inputs. Changed: relabeled/total assistant messages, including base-only transitions. Slices overlap and are not independent.

Plan or Verification. Revised definitions shifted assignments between these labels, sometimes leaving aggregate counts unchanged. Isolate State Recap. Recapping known facts and gaps overlapped with ongoing evidence gathering; narrowing recap and broadening verification changed many assignments. Replace recap with Evidence Collection. This category remained ambiguous with Verification: identical lookups could be labeled as gathering details or checking a candidate.

Granularity and category-specific rules. Sparse labeling left nearly all messages unlabeled. Whole-turn labels recovered coverage; block-level labels also allowed opening plans and tool actions to be separated. Browse-oriented rules increased Verification on MiroVerse but reduced it on DR-Tulu while increasing $\mathbb { P } \bot \ a \mathrm { { n } . }$ indicating policy sensitivity rather than annotation accuracy. Introduce a Math capability. Math received only one span in each larger MiroVerse and DR-Tulu batch, offering little support for a separate module in this corpus.

Resulting capability design. We retain Plan, Search, Writing, and Code for choosing goals and actions, acquiring and checking external evidence, synthesizing answers, and executing computations. Unify external-evidence operations as Search. Evidence collection, source inspection, and verification share one capability, avoiding adapter assignments based on how a lookup is phrased. The main-table library uses Plan, Search, and Writing; Code is tested separately through capability insertion. The initial merged Search variant covered fewer tool-call blocks than its predecessor. Structural search candidates and text-anchored boundaries were still needed to recover coverage. These annotation findings motivate the chosen capabilities and segmentation procedure.

## B Training and Inference Efficiency Details

## B.1 Nominal Training Compute

All recipes use the same Qwen3-32B backbone and a maximum sequence length of 32k. Single LoRA matches the three adapters’ total trainable parameters: rank 128 versus ranks 32/64/32 on the same target projections. Plan, Search, and Writing train for two, three, and three epochs, respectively; Single LoRA trains for three and Full SFT for five. We account for each recipe’s actual token exposure and sequence lengths.

We count major matrix operations in Qwen3- 32B training, with two FLOPs per multiply-add. Let $D _ { m }$ count processed token positions and $Q _ { m }$ estimate the sum of squared unpadded sequence lengths for recipe m. With dense weights $P ,$ rank-r adapter weights $P _ { r }$ , L layers, and total query width $^ { a , }$

$$
\begin{array} { r l r } { { } } & { { \displaystyle F _ { \mathrm { F u l l } } = 6 P D _ { \mathrm { F u l l } } + 6 L a Q _ { \mathrm { F u l l } } , } } & { { } } & { { } } \\ { { } } & { { \displaystyle F _ { m } = ( 4 P + 6 P _ { r _ { m } } ) D _ { m } + 6 L a Q _ { m } , } } & { { } } \\ { { } } & { { \displaystyle F _ { \mathrm { W e A c t } } = \sum _ { m \in \{ \mathrm { p l a n } , \mathrm { s e a r c h } , \mathrm { w r i t i n g } \} } F _ { m } , } } & { { } } \\ { { } } & { { \displaystyle P = 3 1 . 9 8 4 \times 1 0 ^ { 9 } , \quad P _ { r } = 8 . 5 4 6 \times 1 0 ^ { 6 } r , \quad } } & { { } } \\ { { } } & { { \displaystyle L = 6 4 , \quad a = 8 1 9 2 . } } & { { } } \end{array}\tag{7}
$$

In $F _ { \mathrm { F u l l } } , 6 P D _ { \mathrm { F u l l } }$ counts forward, input-gradient, and weight-gradient products; in $F _ { m } , \ 4 P D _ { m }$ omits frozen weight gradients; $6 P _ { r _ { m } } D _ { m }$ trains the adapter; the shared $6 L a Q _ { m }$ term counts causal query–key and attention–value products in both passes; $F _ { \mathrm { { W e A c t } } }$ sums the three independently trained capability adapters’ costs; $P$ sums dense matrix sizes; $P _ { r }$ sums rank-r factor sizes, including the output head; L and a are Qwen3-32B’s layer count and total query width.

Token exposure includes repeated contexts and epochs. Packed padding contributes to dense operations; block-diagonal attention uses individual unpadded trajectories. Length statistics retain the training text format, including reasoning; Single LoRA’s squared-length sum is estimated from randomized batch-token variance. Tool-call learning is part of adapter training; no separate router is trained. We exclude elementwise operations, optimizer updates, activation recomputation, communication, and I/O. Figure 7 shows length-statistic uncertainty as 95% bootstrap intervals for Single LoRA and Full SFT and summed component sensitivity bounds for WeAct.

WeAct uses 142.8 EFLOPs, 7.3% below Full SFT, despite processing 1.23× as many tokens (Figure 7). Relative to Single LoRA, 2.06× token exposure at 0.95× the per-token cost gives 1.96× total compute. Independent capability training thus costs less than Full SFT under these recipes, while its greater token exposure raises total compute over Single LoRA. These estimates use each recipe’s actual sequence lengths and padding, rather than measuring hardware throughput.

## B.2 Historical Inference Efficiency

Records and timing. This analysis uses the logged runs behind the main results, collected over time on the same server with the same concurrency, harness, and tool conditions; we call them historical runs. The quality–time analysis covers six methods on eight benchmarks. The wall-time ECDFs additionally include ManuSearch, whose BrowseComp logs are unavailable, and use all 37,928 task-level records with valid timing. For each task and attempt, the first run is retry 0 and the final run is the highest retry index. Accuracy uses the final run’s judgment, and cost sums all runs up to it. Figures 5 and 8 pair these times with the accuracy in Table 1.

Model-path time is the union of model-call intervals within a run, summed across retries. Taskrecord wall time spans the logged task start and end and can include post-answer judging, so it measures logged execution time rather than answerready latency. The ECDFs include successful, failed, and anomaly-flagged records with valid timing, with one curve per method and benchmark.

Quality–model-path time comparisons. On GAIA, WeAct-soft improves accuracy over Single LoRA from 35.9% to 46.6% while reducing mean model-path time from 774.9 to 364.1 seconds; WeAct-hard reaches 40.8% at 181.0 seconds (Figure 8). TIES-R64 has shorter model paths on Multi-hop QA but lower accuracy than both WeAct variants on all eight benchmarks.

Typical cost and long tails. Both WeAct variants have lower mean task-record wall time than Full SFT on all eight benchmarks and lower means than Single LoRA on six and seven benchmarks for soft and hard, respectively. ReAct and TIES-R64 often have shorter typical runs, but WeAct has much shorter tails (Figure 9). On GAIA, their medians are lower than both WeAct variants’, yet WeAct-hard’s 99th percentile is below those of all four baselines.

On Multi-hop QA, ReAct and TIES-R64 also have shorter typical runs, while WeAct is more accurate than both and uses less mean wall time than Full SFT and Single LoRA. The largest tails occur on Search: on WebWalkerQA, WeAct-hard’s 99th percentile is 1,092 seconds versus ReAct’s 25,315 seconds, despite a longer median. XBench-DS shows the same contrast, although Single LoRA has a lower mean than WeAct-hard at lower accuracy. On the seven benchmarks with ManuSearch logs, both WeAct variants have lower mean and median wall time than ManuSearch. Thus the mean-time gains coexist with longer typical runs than some baselines and substantially shorter extreme runs.

Retries and first-run performance. Table 6 reports the share of tasks with saved retries on GAIA, 2Wiki, and WebWalkerQA. WeAct-soft retries more tasks than WeAct-hard, particularly on WebWalkerQA. WeAct-hard retries fewer tasks than the three trained baselines, and its accuracy changes by at most 0.15 percentage points from first to final run. By comparison, Single LoRA’s GAIA accuracy rises from 28.2% to 35.9%.

For WeAct-hard, the model-path time advantage over Single LoRA is already present in first runs on GAIA and 2Wiki. On WebWalkerQA, WeActhard uses 0.94× Single LoRA’s first-run time, compared with 0.65× when retries are included, so most of that gap comes from retries. The logs do not identify the cause of each retry.

## B.3 Runtime Decomposition and Cache Costs

We evaluate Full SFT, Single LoRA, WeAct-hard, and WeAct-soft on the same prespecified Web-WalkerQA tasks, with 50 runs per method on the same hardware at task concurrency one. All runs are included regardless of correctness. Answerready time spans the complete root-agent execution, excluding subsequent judging. We partition this interval into mutually exclusive model, external-tool, adapter-control, observed cache, answer-extraction, and setup/orchestration intervals; their sum equals the root duration for every run. Model time is further separated by the active adapter or base route. Identified cache intervals, covering the observed cache flush and post-switch prefill, are moved out of their enclosing model calls so that no time is counted twice.

Colors show the mean composition within each stage, not a single run’s event order. For WeActhard, the stages are Plan, Search, and a final Writing or Base stage. For WeAct-soft, Plan spans its load–offload interval, Search runs from the Plan offload to the Writing load (or to the end when Writing is absent), and Writing covers the remainder; these stages include Base calls and tools and average 16.2, 100.0, and 42.5 seconds, respectively. Terminal extraction, the startup Base interval, and Base-only execution are shown separately.

(a) Training recipe  
![](images/239ec11e21b557832649197bdf47e3c716e6610451e6ed523ce64d07f86f4c60.jpg)

(b) Per-token compute  
![](images/5db01b465b490f0a4b44abdd075dba1cbb93c40a19fa7c7284ff95a51c318c13.jpg)

Figure 7: Nominal training compute. (a) Total recipe compute, with WeAct split by capability. (b) Dense and causal-attention FLOPs per token. Error bars show length-statistic uncertainty (details in text).  
![](images/bb437ef2b43da0cd33997cc1ec1c8abd0266215e5aa80a30a489f2a4a2f330d9.jpg)

![](images/c29826f498c9ca245e6ebd258c9421fb50bf0874c69e3ec34898fe21dcd579b9.jpg)  
Figure 8: Accuracy versus model-path time on General and Multi-hop QA. Points pair Table 1 accuracy with historical mean cumulative time, including retries. Colors and markers identify methods; line styles group benchmarks. Time axes are logarithmic in seconds.

Figure 10 shows that WeAct-soft’s mean latency is close to Single LoRA’s, while both WeAct variants use less time than Full SFT. Adapter control and observed cache intervals together occupy 8.2% of soft’s time and 2.8% of hard’s. Time is attributed to the model that served each request, so basemodel and external-tool time within a Search stage is not counted as Search-adapter time.

For the ManuSearch reference, we use 50 firstattempt runs without retries on the same task set, after checking that the question text matches. External tools account for 85.6% of its mean answerready time and model calls for 14.2%; extraction or summarization performed inside a tool counts as tool time. The figure breaks its tool bar between 425 and 2,120 seconds, using the same scale on both sides.

## C Sequential Capability Retention

Adapter organizations. All four organizations start from frozen QWEN3-32B with identical initial weights. The shared baselines train one LoRA of rank 32 or 96, and the partitioned libraries train three rank-32 adapters. The capability partition gives each capability its own adapter in every trajectory. The random partition draws a fixed capability-to-adapter permutation for each trajectory, balanced across trajectories and kept for training and evaluation, so every adapter receives data from all three capabilities. A uniform rank isolates how parameters are organized; the main library uses capability-specific ranks (Section 4.1).

Data. We use 384 training and 192 validation trajectories, partitioning the latter into 168 for stageend evaluation and 24 for monitoring. For each trajectory and capability, a fixed hash selects one complete assistant message as the target. The message is split into chunks of at most 512 target tokens, each with up to 7,168 preceding tokens of context; longer contexts keep the initial and most recent tokens around an omission marker. Each chunk with its context forms one training window.

![](images/9b97993cc5d48a23ffeaa1955c8caa82391eb3151840575652b93cb60b4c5525.jpg)

Figure 9: Historical task-record wall-clock distributions. Colors and markers identify methods; line styles identify benchmarks. Each panel uses its own logarithmic time axis in seconds and retains all tails. Curves include all valid task-record times and saved retries, regardless of correctness; logged wall time can include post-answer judging.
<table><tr><td>Benchmark</td><td>N</td><td>ReAct</td><td>Full SFT</td><td>Single LoRA</td><td>TIES-R64</td><td>WeAct-soft</td><td>WeAct-hard</td></tr><tr><td>GAIA</td><td>103</td><td>0.00</td><td>16.50</td><td>26.21</td><td>14.56</td><td>4.85</td><td>1.94</td></tr><tr><td>2Wiki</td><td>1,000</td><td>0.70</td><td>1.00</td><td>6.40</td><td>4.70</td><td>1.00</td><td>0.60</td></tr><tr><td>WebWalkerQA</td><td>680</td><td>0.29</td><td>9.26</td><td>11.32</td><td>12.79</td><td>9.26</td><td>3.82</td></tr></table>

Table 6: Tasks with saved retries (%). The denominator is the task count N for each method and benchmark.

Optimization. LoRA is applied to all attention and feed-forward projections with α = r and no dropout. Trainable parameters and optimizer moments are kept in FP32 over the BF16 backbone. Each order runs 288 updates, 96 per stage, with six windows per update; all organizations and orders consume the same window streams. We use AdamW without weight decay or gradient clipping, a linear warmup over the first 16 updates to $2 \times 1 0 ^ { - 4 }$ , and a constant rate afterwards. Optimizer state persists across stages, and adapters inactive in a stage receive no updates. All 24 runs (four organizations by six orders) use one training seed, so the six orders are experimental conditions rather than replicates.

Evaluation. Content-token accuracy is teacherforced top-1 accuracy on the target tokens, excluding special tokens. We pool tokens within each message and weight trajectories equally. We evaluate the 168-trajectory validation subset before training and at the end of each stage (steps 0, 96, 192, and 288); the separate 24-trajectory subset is used only for the monitoring curves in Figure 11. Let $A _ { k } ^ { 0 } , A _ { k } ^ { \mathrm { o w n } }$ , and $A _ { k } ^ { \mathrm { f i n a l } }$ be the validation accuracy of capability k before training, at the end of its own stage, and after all stages. Averaging each over validation trajectories and over the four orders in which k precedes another capability, we compute the fraction of acquired gain lost, $F _ { k }$

$$
F _ { k } = \frac { A _ { k } ^ { \mathrm { o w n } } - A _ { k } ^ { \mathrm { f i n a l } } } { A _ { k } ^ { \mathrm { o w n } } - A _ { k } ^ { 0 } } .\tag{8}
$$

The ratio is computed from averaged accuracies rather than averaged over trajectories. Table 7 reports its denominator (acquired gain), numerator (lost gain), and value.

Retention during later stages. Figure 11 follows the first capability of each order on the monitoring trajectories. Writing falls to about 55–70% of its gain within the first few checkpoints of the next stage, whereas Plan keeps roughly 80–95% throughout. Search loses more while Writing is trained than while Plan is trained, and partly recovers when Plan training follows Writing. The shared and randomly partitioned organizations follow similar trajectories.

![](images/ddf0f119f7e7b7f2c39067942a5780c98afef09ea2b89d7f8d2d9261c6d5e89a.jpg)

Figure 10: Runtime decomposition on WebWalkerQA with 50 runs per method. Bars show stage-aligned means. The ellipsis denotes repeated capability switching; all durations are retained. ManuSearch\* is a historical reference with unmatched deployment and tool conditions.
<table><tr><td rowspan="2">Adapter organization</td><td colspan="3">Plan</td><td colspan="3">Search</td><td colspan="4">Writing</td></tr><tr><td>G</td><td>L</td><td>%</td><td>G</td><td>L</td><td></td><td>%</td><td>G</td><td>L</td><td>%</td></tr><tr><td>Single-LoRA (rank 32)</td><td>14.28</td><td>1.67</td><td>11.7</td><td>6.95</td><td>1.57</td><td></td><td>22.6</td><td>3.05</td><td>1.41</td><td>46.2</td></tr><tr><td>Single-LoRA (rank 96)</td><td>14.31</td><td>1.82</td><td>12.7</td><td>6.81</td><td>1.68</td><td></td><td>24.7</td><td>3.06 </td><td>1.50</td><td>49.2</td></tr><tr><td>Random partition</td><td>13.36</td><td>1.32</td><td>9.9</td><td>6.16</td><td>0.96</td><td></td><td>15.6</td><td>2.58</td><td>1.09</td><td>42.4</td></tr><tr><td>Capability adapters</td><td>14.22</td><td>0.00</td><td>0.0</td><td>6.84</td><td>0.00</td><td></td><td>0.03.09</td><td></td><td>0.00</td><td>0.0</td></tr></table>

Table 7: Acquired gain (G) and lost gain (L) in points of content-token accuracy, and fraction of acquired gain lost (%), pooled over the four orders in which each capability precedes another. Fractions use unrounded means.

## D Diagnostic Metrics for Capability Separation

First-order update analysis. Eq. 3 considers a step on a single capability. For joint training, let $\mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { K }$ be the span datasets with sampling shares $p _ { k }$ and per-capability losses $\mathcal { L } _ { k } ^ { \mathrm { s h a r e d } } ( \omega )$ over shared parameters $\omega . \mathrm { ~ \bf ~ A ~ }$ shared adapter minimizes $\begin{array} { r } { \bar { \mathcal { L } } _ { \mathrm { m i x } } ^ { \mathrm { s h a r e d } } ( \omega ) = \sum _ { k } p _ { k } \mathcal { L } _ { k } ^ { \mathrm { s h a r e d } } ( \omega ) } \end{array}$ with gradient steps $\begin{array} { r c l } { \delta \omega } & { = } & { - \eta \sum _ { k } p _ { k } g _ { k } } \end{array}$ , where $\qquad g _ { k } =$ $\nabla _ { \omega } \mathcal { L } _ { k } ^ { \mathrm { s h a r e d } } ( \omega )$ Writing $\Delta \mathcal { L } _ { i }$ for the change in capability i’s shared-adapter loss, the first-order approximation is

$$
\begin{array} { r } { \Delta \mathcal { L } _ { i } \approx - \eta \Big ( p _ { i } \left. g _ { i } \right. ^ { 2 } + \sum _ { j \neq i } p _ { j } \left. g _ { i } , g _ { j } \right. \Big ) . } \end{array}\tag{9}
$$

When $\langle g _ { i } , g _ { j } \rangle \ : \approx \ : 0$ , the other capabilities’ updates give capability i little first-order transfer, and its own term scales with $p _ { i }$ , so unequal sampling shares or gradient norms let one capability dominate the mixed direction. Like Eq. 3, this describes one gradient step; with an adaptive optimizer, the realized update also depends on optimizer state.

Gradient measurements. For capability gradients $g _ { i }$ and $g _ { j }$ in the same adapter parameter space, we report $\begin{array} { r l } { \cos ( g _ { i } , g _ { j } ) } & { { } = } \end{array}$ $ { g _ { i } , g _ { j } }  / ( | | g _ { i } | | | g _ { j } | | )$ and the norms $\left\| g _ { i } \right\|$ . The norm ratio max<sub>i</sub> $\left. g _ { i } \right. / \operatorname* { m i n } _ { i } \left. g _ { i } \right.$ is computed without weighting by sampling share, which enters through Eq. 9.

Figure 12 probes a jointly trained rank-96 Single LoRA (α = 32) at ten evenly spaced training checkpoints, using the same pool of capability spans at every checkpoint. Most cosines remain positive, while the largest-to-smallest norm ratio grows over training; Writing has the smallest gradients after the first checkpoint. Positive cosines mean that, to first order, a step on one capability does not raise another’s loss. The loss in Section 3.1 arises under sequential training, where a capability that has already been fit has small gradients, so later updates can raise its loss through higher-order terms even without gradient conflict.

![](images/63b4f8a18449ba02aaf1a67fd0f062e8f67c459b8ab6c11e47c2a49a12eb5524.jpg)  
Figure 11: Retention during subsequent training. Panels track the first capability in each order on monitoring trajectories; shading marks later stages, and 100% denotes the gain acquired in its own stage.

(a) Gradient alignment  
![](images/abea728a3b18b7e0f00377025d6ca70396c4e0c9aadd1069d08b486080f5fe7d.jpg)

(b) Gradient magnitudes  
![](images/80530ce015d3280bb173451a9feb91436403586ad6fd155591d179d4a8f194c0.jpg)  
Figure 12: Gradient alignment and magnitude in the Single-LoRA pilot. (a) Pairwise cosines and (b) gradient norms at ten checkpoints of a rank-96 Single-LoRA run on frozen QWEN3-32B. Most pairwise cosines are positive, while gradient magnitudes differ across capabilities.

## E Static-Merge Residual Details

## E.1 Fixed Capability Library and Validation Data

We fix QWEN3-32B and the same independently trained Plan, Search, and Writing adapters throughout this diagnostic. Adapter ranks and scaling follow the main experimental setup (Appendix F). Individual activates the adapter matching the target capability. Full Additive sums the effective updates $\Delta W _ { k } = ( \alpha _ { k } / r _ { k } ) B _ { k } A _ { k }$ without final rank compression. TIES-R64, DARE-TIES-R64, and TSPA-R64 construct a fixed merged update and represent it at rank 64. TIES uses density 0.2;

DARE-TIES additionally uses drop rate 0.5 and seed 0. We follow the original TSPA procedure (Li et al., 2026c), which performs attention alignment followed by LoRA parameter alignment. It optimizes rotations and scaling within the low-rank factors before merging the aligned adapters.

We use our capability-specific validation split: each capability’s source data is shuffled with seed 42, the first 200 records form each capability’s validation set, and the remainder is used for training. Both the activation and NLL diagnostics draw their evaluation records from these validation sets. We render their messages with the training format, preserving supervision masks, capability labels, and recorded tool observations. Evaluation replays the recorded text without making new external tool calls. The context and target selection rules for each diagnostic are described below.

## E.2 Local Effects on Matched Activation

The activation diagnostic tests whether a static update preserves the local effect of the capabilityspecific update. For projection $j ,$ we run Individual to obtain its input $h _ { k , j }$ and evaluate every candidate on that same input:

$$
\rho _ { m , k , j } = \frac { | | ( \Delta W _ { m , j } - \Delta W _ { k , j } ) h _ { k , j } | | _ { 2 } } { \operatorname* { m a x } ( | | \Delta W _ { k , j } h _ { k , j } | | _ { 2 } , 1 0 ^ { - 1 2 } ) } .\tag{10}
$$

Here m identifies the merge rule, k the target capability, and $j$ the LoRA projection; $\Delta W _ { m , j }$ and $\Delta W _ { k , j }$ are the merged and individual adapter updates, respectively. This measures adapter-induced local mismatch, excluding differences introduced by propagating each candidate’s own activations through later layers. For Full Additive, writing $h = h _ { k , j }$ , the difference is exactly the combined contribution of the other capability adapters:

$$
\begin{array} { r } { \left( \sum _ { i } \Delta W _ { i , j } - \Delta W _ { k , j } \right) h = \sum _ { i \neq k } \Delta W _ { i , j } h . } \end{array}\tag{11}
$$

The residual thus measures how non-target updates change the matched adapter’s local action; their contributions can partly reinforce or cancel one another.

We retain an 8,192-token prefix of each trajectory and select causally aligned supervised positions. The activation used to predict a target token is taken from the preceding position, with its capability label checked at that source position. Each validation record contributes four positions across all attention and feed-forward LoRA projections in every transformer layer. We average positions and projections within each layer and trajectory, then weight trajectories equally.

Figure 13 shows capability-dependent local mismatch across layers. Conflict handling and subspace alignment change its magnitude, and the ordering of the fusions varies by capability, but none recovers the matched update’s local effect for all three capabilities.

## E.3 Target-Text Fit and Variation across Trajectories

We complement the local diagnostic with teacherforced target-token NLL, comparing each static condition and Individual on the same record and supervised targets. LoRA updates are applied to all adapted layers, including the output head. Contexts are capped at 4,096 tokens, and each record contributes the mean NLL of the valid supervised tokens visible in its target capability span. For record x with capability k and merge rule m, we compute

$$
\Delta L _ { m , k } ( x ) = L _ { m } ( x ) - L _ { k } ( x ) ,\tag{12}
$$

where L is the record’s mean target-token NLL. Figure 14 shows the distribution of these paired differences across validation records, with diamonds marking their equally weighted means; zero denotes equal target-text fit.

The mean NLL gap is positive for every tested fusion in all three capabilities, although some records have gaps close to zero or negative (Figure 14). Full Additive’s Writing mean exceeds its median, indicating that a subset of records with large gaps raises the mean. The tested fusions therefore reduce mean target-text fit, with the size and direction of the change varying across records.

The local and NLL diagnostics agree at the capability level: every tested fusion leaves local mismatch and poorer mean target-text fit relative to capability-matched use. These results support preserving the specialists as separate updates and selecting them at runtime. Section 4.2 reports the task-level effect of these merges.

## F Experimental Setup Details

This appendix gives the configuration of the main experiments; the sequential diagnostic uses its own configuration (Appendix C).

Capability adapters. The capability adapters use ranks 32, 64, and 32 for Plan, Search, and Writing, with $\alpha _ { k } = 3 2$ and no LoRA dropout. Search uses rank 64 because rank-32 pilots showed repetitive, non-terminating reasoning and unreliable external-tool and offload\_lora calls. Each adapter places low-rank factors on the query, key, value, and output projections and on the gate, up, and down projections of all 64 transformer layers, and on the output projection (lm\_head); input embeddings are not adapted. We train the adapters with Tinker (Thinking Machines Lab, 2025) using the Qwen3 chat template, a batch size of eight, and a maximum sequence length of 32k. Adam uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ and $\epsilon = 1 0 ^ { - 8 }$ , and the learning rate decays from its peak to zero on a cosine schedule without warmup. Plan trains for two epochs at a peak learning rate of $2 \times 1 0 ^ { - 4 }$ ; Search and Writing train for three epochs at $3 \times 1 0 ^ { - 4 }$ and $2 \times 1 0 ^ { - 4 }$ , respectively. Each capability’s data are shuffled with seed 42, and the first 200 records form a validation set for NLL monitoring. The activation and NLL diagnostics draw their evaluation records from these validation sets (Appendix E). Samples for different modules can share source trajectories and execution prefixes, but not loss targets. Single LoRA follows the same recipe with rank 128, three epochs, and a peak learning rate of $3 \times 1 0 ^ { - 4 }$

![](images/4f527bac93bd4a8750f285c1e6198a0c2d658891322a4c1bbfc67288f092bb61.jpg)  
Figure 13: Local mismatch on capability-matched activations. Curves show trajectory-weighted layer means of Eq. 10; shading gives pointwise 95% trajectory-bootstrap confidence intervals. The dashed zero line is Individual’s identity reference. Full Additive uses no final rank compression.

Full SFT. The full-parameter baseline starts from the same Qwen3-32B checkpoint and is trained with LLaMA-Factory on eight NVIDIA H200 GPUs, using DeepSpeed ZeRO-3 without offloading, FlashAttention-2, Liger kernels, gradient checkpointing, and bfloat16. Each optimizer step uses 64 packed sequences: one per device per micro-step, with eight gradient-accumulation steps. AdamW uses a peak learning rate of $4 \times 1 0 ^ { - 5 }$ warmup over the first 10% of steps followed by cosine decay, and no weight decay. Training runs for five epochs with seed 42. Trajectories are packed into 32k token sequences with block-diagonal attention, so packed trajectories do not attend to one another. The Qwen3 template keeps the reasoning content, and prompt and tool-response tokens are excluded from the loss. Full SFT trains on compressed trajectories (Appendix A).

Evaluation protocol. The main table reports each benchmark separately, and its Avg. column is the unweighted mean over the eight benchmarks. Each method keeps its own context management and model-call structure and attempts every task once. A task that ends without a valid boxed answer is retried up to twice, allowing at most three attempts; the first valid answer is scored. On 2Wiki,

Bamboogle, and MuSiQue, an answer is correct if the boxed answer exactly matches the gold answer or one of its aliases after normalization (lowercasing and removing punctuation, articles, and extra whitespace). On the other five benchmarks, GPT-5.6 Luna compares the answer with the reference using each benchmark’s grading prompt; on GAIA, WebWalkerQA, and FRAMES, answers that match the reference after normalization are accepted without a judge call. For a benchmark with accuracy p on n evaluation tasks, Table 1 gives the nominal task-based binomial standard error $\sqrt { p ( 1 - p ) / n }$ . For Avg., we propagate these as $\scriptstyle \sqrt { \sum _ { b } \mathrm { S E } _ { b } ^ { 2 } / 8 }$ , treating benchmarks as independent.

## G Trace Analysis Details

Graph construction. We use SearchAtlas’s released builder and evaluator to represent searches as query nodes, with page reads supplying evidence to the queries that retrieved them. Edges link queries, question constraints, prior knowledge, and the final answer. Attribution uses GPT-5.2 at temperature 0 with JSON outputs and access to all earlier tool results. Each question is decomposed once into constraint units shared across methods; evidence and answer attribution are performed separately for each method. Reader summaries are treated as observed content without checking the original pages.

Diagnostics. Let Q be the issued queries, A the answer, and $\mathcal { Q } _ { A } \subseteq \mathcal { Q }$ the queries that reach A through evidence-use edges. Let C be the question’s constraint units, $C ( q ) \subseteq C$ those deployed by query $q ,$ and $E _ { \mathrm { P K } }$ the prior-knowledge edges. For nonempty query and constraint sets, the diagnostics used here are

![](images/869009ea8cdd6d10d99ed5c5b739fd7b5c06bb8ab08a6c4fc8cfbecdb8951041.jpg)

![](images/98c8324499da02cf039605c9d2d557ef31c06c17f97d22641c2ad6847534a303.jpg)

(c) Writing  
![](images/08af25ea480dbde709cb810172b7c675656a21668f32c337d127ecdcd4d5fa56.jpg)

Figure 14: Paired target-token NLL gaps relative to Individual. Points show validation records, boxes show medians and interquartile ranges, and diamonds mark means with 95% confidence intervals.
<table><tr><td>Method QU↑</td><td> $\operatorname { C o v } \uparrow$  0.336</td><td>PK↓</td></tr><tr><td>ReAct Least-to-Most ManuSearch</td><td>0.365 0.386 0.360 0.305 0.385</td><td>0.590 0.600 0.520</td></tr><tr><td>Single LoRA Full SFT</td><td>0.315 0.410</td><td>0.332 0.630 0.478 0.530</td></tr><tr><td>TIES-R64 DARE-R64 TSPA-R64</td><td>0.293 0.370 0.375</td><td>0.311 0.680 0.375 0.600 0.343 0.640</td></tr><tr><td>WeAct-hard WeAct-soft</td><td>0.570 0.521</td><td>0.601 0.390 0.554 0.440</td></tr></table>

Table 8: SearchAtlas diagnostics by method over the 100 tasks of Figure 4. Values are task means; $\mathbf { Q } \mathbf { U } =$ query utilization, Cov = question-constraint coverage (answer graph), PK = share of answers with a priorknowledge-to-answer edge. Bold marks the best value per column (lowest for PK).

$$
\mathrm { Q U } = { \frac { | { \mathcal { Q } } _ { A } | } { | { \mathcal { Q } } | } } ,\tag{13}
$$

$$
\mathrm { C o v } = \frac { \left| \bigcup _ { q \in \mathcal { Q } _ { A } } C ( q ) \right| } { | C | } ,\tag{14}
$$

$$
\mathrm { P K A } _ { \mathrm { a n s } } = \mathbf { 1 } [ ( \mathrm { P K } , A ) \in E _ { \mathrm { P K } } ] .\tag{15}
$$

QU measures the fraction of queries on an evidence path to the answer. Cov uses the constraints deployed by direct answer parents and their evidence ancestors, consistently across tasks. PK $\mathbf { A } _ { \mathrm { a n s } }$ indicates an answer edge attributed to prior knowledge without retrieved support. Reported scores are task means, so mean $\mathrm { P K A } _ { \mathrm { a n s } }$ is the share of answers with such an edge. Higher QU and Cov and lower $\mathrm { P K A } _ { \mathrm { a n s } }$ indicate stronger retrieval grounding. Low coverage may indicate missing retrieval support for some question requirements; unsupported content is not necessarily incorrect.

Trajectories. Except as noted in Section 4.3, we use for each method and task the trajectory behind the corresponding result in Table 1: the earliest successful attempt for correctly answered tasks and the last completed attempt for tasks answered incorrectly.

Quality control and statistics. A task is excluded for all methods if any method’s graph contains CJK text (the constraint matching relies on English lexical signals), has no answer signals, or records an unrecovered constraint-arbitration error; excluded tasks are not replaced. The 100 retained tasks come from MuSiQue (23), 2Wiki (20), Bamboogle (18), GAIA (17), FRAMES (14), BrowseComp (4), and WebWalkerQA (4); xbench-DS tasks do not pass the language filter. Means weight tasks equally. Intervals are 95% percentile intervals from 5,000 task-level bootstrap resamples.

Additional observations. Table 8 reports the means in Figure 4. Both WeAct variants lead all baselines on the three diagnostics. ManuSearch issues the most queries, but WeAct-hard covers more question constraints along the evidence paths supporting its answers. WeAct-hard also has fewer tasks without a retrieval-supported answer path than any baseline.

<table><tr><td></td><td>MegaScience</td><td>SciBench</td><td>DS-1000</td><td>GAIA</td></tr><tr><td>#Tasks</td><td>200</td><td>575</td><td>100</td><td>103</td></tr></table>

Table 9: The number of selected tasks used for evaluating code insertion tasks. The MegaScience and DS-1000 tasks were randomly sampled based on a balanced consideration of cost and representativeness.

## H General Capability Preservation

We evaluate catastrophic forgetting on MMLU-PRO (Wang et al., 2024b), IFEVAL (Zhou et al., 2023b), MT-BENCH (Zheng et al., 2023), STRUCT-FLOW, and HUMANEVAL. All external tools are disabled: WeAct reasons through model weights while retaining internal adapter selection.

Table 10 shows that WeAct substantially mitigates the general-capability forgetting observed under Full SFT across these five benchmarks. Full SFT loses 11.60 and 15.16 points on MMLU-Pro and IFEval, respectively. Both WeAct variants exceed Base on MMLU-Pro and MT-Bench and remain close to it on StructFlow. IFEval still declines, but much less than under Full SFT; HumanEval matches Base under soft routing and declines less than Full SFT under hard routing. Both variants thus retain higher scores than Full SFT on every evaluated benchmark.

Remark 1. With every adapter inactive, the model parameters are bitwise equal to the frozen base $\theta _ { 0 } ;$ with adapter k active, they differ by $\Delta \theta _ { k }$ . This concerns parameters only; Table 10 measures behavior with adapters available.

## I Code Insertion: Implementation and Evaluation Protocol

Insertion adds a code capability without retraining the base, modifying the plan, search, and writing adapters, or replaying their data. We train only a new code adapter on code-computation trajectories from MiroVerse-MegaScience, converted to our tool and handoff protocol; these trajectories use code to calculate and verify answers from given conditions. Registering the code alias extends the routing vocabulary from K = 3 to K = 4.

<table><tr><td>Method</td><td> $\begin{array} { c } { { \mathrm { M M L U - } } } \\ { { \mathrm { P r o } } } \end{array}$  IFEval</td><td>MT- Bench</td><td>Struct Human Flow Eval</td></tr><tr><td>Base</td><td>58.03 77.82</td><td>9.3875</td><td>98.12 90.85</td></tr><tr><td>Full SFT DARE-TIES</td><td>-19.99 -19.48</td><td>-0.93 +0.53</td><td>-1.60 -6.86 -0.95 -1.34</td></tr><tr><td>Single LoRA</td><td>-0.55 -4.61 -1.15 -3.80 一</td><td>0.00</td><td>-0.76 -5.37</td></tr><tr><td>WeAct-hard WeAct-soft</td><td>+1.60 +1.21</td><td>-4.51+0.80 -4.28+0.80</td><td>-0.08 -2.69 -0.04 0.00</td></tr></table>

Table 10: General-capability retention (external tools disabled). Base: scores; others: $1 0 0 ( S / S _ { \mathrm { B a s e } } - 1 ) \%$ Green/red mark the best/worst changes per column, including ties.

The evaluation has two parts. First, we compare hashes of the base and the three existing adapters before and after insertion. Second, as shown in Table 9, we evaluate the original GAIA tasks before and after insertion, reporting the paired quality change and how often code is invoked on tasks that need no computation (2 of 80). New-capability utility is measured on MEGASCIENCE (Fan et al., 2026), SCIBENCH (Wang et al., 2024a), and DS-1000 (Lai et al., 2023). Insertion succeeds only if code tasks improve, the existing hashes are unchanged, old-task quality does not decrease, and misrouting into code stays low; unchanged parameters alone do not suffice.

## J Writing Refresh: Protocol and Full Results

Local update. We continue training Writing on OpenScholar-based trajectories synthesized in our harness, keeping the base, Plan, and Search fixed and replaying no earlier data. The pool contains 1,000 rows from 983 unique trajectories. Splitting by trajectory ID gives 795 training rows and 205 validation rows; only the training split is used for the update.

Report-quality results. Table 11 gives the DRB rubric breakdown for the overall scores in Table 3. DRB totals use the evaluator’s weighted aggregation, not an unweighted rubric mean.

Auxiliary adaptation and retention probe. The paired teacher-forced probe uses identical reference sequences, tokenization, and masks for the original and refreshed Writing adapters. It activates Writing on Writing positions and uses the frozen base elsewhere. NLL is averaged over supervised tokens and exponentiated for perplexity; targets may include reasoning and tool calls as well as report text. Table 12 reports NLL and perplexity for the OpenScholar and historical Writing validation sets under this shared protocol.

<table><tr><td rowspan="2">Rubric</td><td colspan="3">WeAct-Hard</td><td colspan="3">WeAct-Soft</td></tr><tr><td>Original</td><td>LoRA-write++</td><td>∆</td><td>Original</td><td>LoRA-write++</td><td>∆</td></tr><tr><td>Comprehensiveness</td><td>25.32</td><td>26.33</td><td>+1.01</td><td>26.50</td><td>26.78</td><td>+0.28</td></tr><tr><td>Depth</td><td>22.46</td><td>23.59</td><td>+1.13</td><td>23.82</td><td>24.19</td><td>+0.37</td></tr><tr><td>Instruction following</td><td>29.65</td><td>31.22</td><td>+1.57</td><td>31.36</td><td>32.45</td><td>+1.09</td></tr><tr><td>Readability</td><td>32.33</td><td>33.81</td><td>+1.48</td><td>33.32</td><td>35.10</td><td>+1.77</td></tr></table>

Table 11: DeepResearchBench rubric scores before and after the Writing refresh (higher is better). ∆ is the reported score-point change from Original to LoRA-write++.
<table><tr><td rowspan="2">Evaluation Set</td><td rowspan="2">Rows</td><td colspan="2">NLL</td><td colspan="2">Perplexity</td></tr><tr><td>Original</td><td>LoRA-write++</td><td>Original</td><td>LoRA-write++</td></tr><tr><td>OpenScholar validation</td><td>205</td><td>1.27</td><td>1.02</td><td>3.56</td><td>2.76</td></tr><tr><td>Historical Writing validation</td><td>200</td><td>0.46</td><td>0.50</td><td>1.59</td><td>1.66</td></tr></table>

Table 12: Supervised-token fit before and after the Writing refresh (lower is better).