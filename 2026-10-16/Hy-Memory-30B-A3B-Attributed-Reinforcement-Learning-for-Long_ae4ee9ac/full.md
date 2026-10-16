# Hy-Memory-30B-A3B: Attributed Reinforcement Learning for Long-Term Memory Writing

Tianxiang Fei Jiazheng Kang Mao Zheng

Xingyan He Xiang Yu

Tencent Hunyuan

## Abstract

External-memory systems extend language-model applications beyond a context window, but their quality depends critically on the model executing the write path: deciding what to retain, how precisely to represent it, and how new evidence should update persistent state. Existing frameworks typically place a general-purpose large language model behind framework-specific prompts. This raises serving cost, and a writer specialized for one protocol does not transfer directly to another; a unified model that can operate across established memory frameworks remains largely unexplored. We present Hy-Memory-30B-A3B, a compact long-term-memory writer trained across Mem0, MemOS, and Hy-Memory. Memory On-Policy Distillation teaches framework-specific write protocols on student-generated states; Memory RL then optimizes downstream answerability using dense per-chunk probes, attributed cross-chunk rewards, over-storage controls, and chunk-level GRPO. Carrying policy-written banks across windows keeps training aligned with deployment. On fixed 20% benchmark subsamples, Hy-Memory-30B-A3B ranks first on all four evaluated harnesses, including held-out EverMemOS, with a 55.49% macro average versus 54.62% for Claude Opus 5.

huggingface.co/tencent/Hy-Memory-30B-A3B

github.com/Tencent-Hunyuan/Hy-Memory

## 1 Introduction

A long-term memory system for an LLM application usually contains three functional stages: (1) a writer converts an interaction into persistent memory operations; (2) a retriever selects records relevant to a future query; and (3) a reader or answer model uses the retrieved records to respond.

Systems such as Mem0, A-MEM, MemoryOS, and MemOS have expanded the design space of persistent memory through extraction, consolidation, hierarchical paging, linking, versioning, and lifecycle management (Chhikara et al., 2025; Xu et al., 2025; Kang et al., 2025; Li et al., 2025). Benchmarks such as LoCoMo and LongMemEval have also shown that long-term memory is not a single retrieval problem: it includes precise extraction, multi-session composition, temporal reasoning, knowledge updates, and abstention (Maharana et al., 2024; Wu et al., 2025).

Despite this progress, the write policy is commonly implemented by placing a general-purpose LLM behind a long, framework-specific prompt. That model must simultaneously distinguish durable user information from greetings, assistant suggestions, and transient text; preserve exact names, numbers, versions, dates, paths, and other identifying details; represent a state transition without losing either the new state or what it replaced; reconcile new evidence with retrieved memories already in the bank; and emit valid operations under the framework’s schema and control flow.

The framework cannot recover information that the writer omitted, and a powerful downstream reader cannot reliably infer a fact that was never stored. Conversely, indiscriminately storing everything increases cost and retrieval interference. The desired writer is therefore not a generic summarizer: it is a policy for changing an external state under a long-horizon downstream objective.

Recent work has begun to train memory operations with reinforcement learning. Memory-R1 trains a memory manager and an answer agent with final-answer rewards (Yan et al., 2025). MemAgent learns recurrent textual compression from a terminal verifier (Yu et al., 2026). Mem-α combines downstream QA correctness with format, compression, and operation-quality rewards (Wang et al., 2025). MemCoE explicitly identifies sparse outcome reward as a weakness and adds guideline-aligned process feedback (Xu et al., 2026). These studies establish that memory behavior is learnable rather than fixed. They also expose a remaining problem: a final answer score usually supervises an entire sequence of heterogeneous memory actions, even though only a few actions supplied the evidence—or caused the failure.

Hy-Memory-30B-A3B focuses on this credit-assignment problem while preserving compatibility with existing memory frameworks. Our contributions are:

• A framework-integrated memory writer. Hy3-30B-A3B replaces the LLM inside Mem0, MemOS, and Hy-Memory without moving their memory logic into a custom model architecture.

• State-distribution-aligned training. OPD and RL both execute the policy inside the actual harness, and later writes observe banks produced by earlier policy outputs.

• Failure-surface reward data. Training questions are constructed from concrete write failures, including omission, lost specificity, semantic inversion, stale-state confusion, secondary-fact loss, and over-storage.

• Attributed, non-binary feedback. Local probes provide dense per-chunk evidence, while crosschunk outcome questions use gold facts to map recovered evidence back to source chunks.

• Chunk-level policy optimization. Advantages are normalized among rollouts at the same chunk position rather than over whole trajectories, so one rollout can be credited at one chunk and penalized at another. Each rollout casts one vote per chunk even when a framework makes multiple LLM calls.

• Cross-harness and OOD results. Hy-Memory-30B-A3B ranks first on Mem0, Hy-Memory, MemOS, and held-out EverMemOS, averaging 55.49% across the four harnesses versus 54.62% for Claude Opus 5.

## 2 Background and Related Work

## 2.1 External memory systems and evaluation

Mem0 uses incremental extraction and consolidation to maintain a persistent memory store (Chhikara et al., 2025). A-MEM organizes memories as dynamically linked notes whose attributes and relationships evolve as new information arrives (Xu et al., 2025). MemoryOS introduces an operating-system-inspired hierarchical storage architecture that coordinates short-term, mid-term, and long-term personal memory through dialogue-chain FIFO and segmented page updates (Kang et al., 2025). MemOS treats memory as a managed system resource and emphasizes representation, scheduling, provenance, and lifecycle control (Li et al., 2025). These systems differ in schema and control flow, but all depend on an LLM to interpret new interactions and produce valid operations.

DCPM argues that long-term memory should support more than surface recall: its dual-process architecture represents belief revisions, identity, schemas, latent intentions, and cross-domain abstractions in a capability hierarchy (Fei et al., 2026b). This direction is complementary to ours. DCPM redesigns the memory architecture and its evolution process, whereas Hy-Memory-30B-A3B trains a reusable writer that operates inside several existing memory architectures without replacing them.

LoCoMo evaluates conversations containing hundreds of turns and multiple sessions, with questions requiring single-hop, multi-hop, temporal, open-domain, and adversarial reasoning (Maharana et al., 2024). LongMemEval decomposes conversational memory into information extraction, multi-session reasoning, temporal reasoning, knowledge update, and abstention (Wu et al., 2025). Together, these benchmarks motivate evaluating the full memory pipeline rather than the fluency of an extracted summary.

Our work draws two conclusions from this literature. First, memory quality must be measured downstream, after retrieval. Second, a useful reward dataset must cover distinct failure modes rather than collapsing all memory behavior into one generic QA score.

## 2.2 Reinforcement learning for memory

There are two related but distinct lines of memory RL.

<table><tr><td>Method</td><td>Learned policy and environment</td><td>Reward and distinction</td></tr><tr><td>Memory-R1</td><td>Manager and answer agent over a structured external bank</td><td>Downstream answer outcome; writer is confounded with the learned reader</td></tr><tr><td>MemAgent</td><td>Recurrent writer and answer policy over model-native fixed memory</td><td>Terminal score broadcast over the trajectory; not a drop-in writer</td></tr><tr><td>Mem-α</td><td>Construction policy over a custom multi-component bank</td><td>Global QA plus local checks; global correctness is shared across actions</td></tr><tr><td>MemCoE</td><td>Guideline and evolution policy for personalization memory</td><td>Process plus final-answer reward; checks guideline adherence rather than evidence contribution</td></tr><tr><td>Hy-Memory- 30B-A3B</td><td>Writer only, operating in Mem0, MemOS, and Hy-Memory</td><td>Per-chunk recall, attributed outcome, and growth penalty; localizes evidence to writes</td></tr></table>

Table 1: How Hy-Memory-30B-A3B differs from prior memory RL.

Recurrent compression agents. MemAgent processes a long document in chunks, repeatedly overwrites a bounded textual memory, and applies a final verifiable answer reward to all generated conversations (Yu et al., 2026). This trains a model-native long-context mechanism, but the memory representation and inference loop are part of the learned architecture.

External-bank management. Memory-R1 learns ADD, UPDATE, DELETE, and NOOP operations, and also trains an answer agent that filters retrieved memories (Yan et al., 2025). Mem-α formulates memory construction as a sequence of operations over a multi-component bank and combines global QA correctness with local format and semantic-validity rewards (Wang et al., 2025). MemCoE uses an induced guideline to construct denser process feedback because final answer rewards alone are sparse over long memory-evolution trajectories (Xu et al., 2026).

Hy-Memory-30B-A3B belongs to the second line but differs in scope. It trains the writer only, keeps the reader fixed during reward computation, and targets multiple pre-existing harnesses rather than one newly designed memory architecture. If the writer and answer model are optimized together, a better answer can come from improved reading rather than improved memory. Table 1 summarizes the distinction.

State separation also appears outside memory-specific agents. CodeDelegator separates persistent orchestration state from ephemeral execution sandboxes to prevent debugging traces and intermediate failures from polluting long-horizon planning context (Fei et al., 2026a). Our setting has a related systems requirement: candidate rollouts must write to isolated banks, while only the selected bank persists across windows. Unlike CodeDelegator, however, our separation is used for on-policy credit assignment over memory writes rather than role-specialized code execution.

## 2.3 Why ordinary outcome-only RL is insufficient

LLM RL commonly assigns a scalar reward to a whole completion. PPO stabilizes policy updates with a clipped surrogate (Schulman et al., 2017), while GRPO replaces a learned critic with a baseline estimated from grouped samples (Shao et al., 2024). This works well when the completion is a single solution and the final verifier is informative. A memory trajectory violates both assumptions.

A write in an early session may matter only when a much later question asks about it, so most writes receive no direct evidence. Giving the final QA score to every write rewards irrelevant actions and penalizes correct writes that co-occur with one bad action. Binary exact match often makes every rollout in a small GRPO group receive the same score, producing zero advantage and no task gradient; this has also been observed in memory-grounded QA (He et al., 2026). Jointly improving the answer agent can raise QA accuracy while the bank remains incomplete or noisy. An action changes the bank seen by later actions, so examples are not independent prompts. One write may invoke one LLM call while another invokes several, so a row-level baseline can weight call count rather than decision quality. Recall-only rewards can favor excessive storage.

The distinction between outcome and process supervision is well established in reasoning (Lightman et al., 2023). Memory writing needs a related but domain-specific decomposition. We do not label a hidden reasoning chain. Instead, we supervise observable consequences of each bank mutation: whether the introduced fact remains recoverable, whether irrelevant text was stored, and whether later cross-chunk answers recover the evidence contributed by that chunk.

<table><tr><td>Family</td><td>Configuration</td><td>Role</td></tr><tr><td>Mem0</td><td>General</td><td>Incremental extraction and additive writing</td></tr><tr><td>MemOS</td><td>General</td><td>Multi-stage extraction and reconciliation</td></tr><tr><td>Hy-Memory</td><td>Chat</td><td>Personal conversational memory</td></tr><tr><td>Hy-Memory</td><td>Working</td><td>Tasks, decisions, and project state</td></tr><tr><td>Hy-Memory</td><td>Coding</td><td>Code, commands, bugs, and technical decisions</td></tr></table>

Table 2: Harness families and scenario configurations.

## 3 Problem Formulation

## 3.1 Unified harness scope

OPD and RL use the same conceptual harness set (Table 2). The three Hy-Memory rows are scenario configurations of one harness family, not three unrelated frameworks. “All trained harnesses” means these five configurations from three harness families.

## 3.2 Data hierarchy

An interaction sequence contains chronological sessions. Each session contains turns (one request and its response). A chunk is one harness write input, usually about ten turns, and never crosses a session boundary. A window is up to five consecutive chunks processed in one RL step and may span session boundaries.

## 3.3 Memory writing as an MDP

For chunk t, let M be the persistent memory bank and x the raw interaction chunk. The policy receives a harness-generated prompt containing $x _ { t }$ and selected information from $M _ { t } ,$ then produces one or more structured LLM outputs $a _ { t }$ . The harness executes those outputs and creates the next bank:

$$
\begin{array} { r l } & { s _ { t } = ( M _ { t } , x _ { t } , h ) , } \\ & { a _ { t } \sim \pi _ { \theta } ( \cdot \mid s _ { t } ) , \quad M _ { t + 1 } = T _ { h } ( M _ { t } , a _ { t } ) , } \end{array}\tag{1}
$$

where h denotes the harness configuration. The state lives primarily in the external bank, not in a continuous model context. Every LLM call may start from a fresh prompt, but the decisions are sequential because the bank carries their consequences.

This creates two training requirements: rollouts from one candidate must write to an isolated bank copy, and later windows must start from a bank the policy itself produced, not an annotation-derived ideal bank.

## 3.4 Three-model separation

The policy (Hy3-30B-A3B) sees the harness write prompt and produces all memory-write outputs; it is the only model trained in Stage 2. The answer model sees the question and the retrieved memory block, and reads what the bank exposes. The judge sees the question, answer, reference, and rubric, and assigns semantic credit. The answer model never sees the reference or rubric. The judge never sees the retrieved memory block. Freezing both makes the memory bank the only changing component in the reward path.

![](images/a5a0b0baeca1088b217ef609e711fbb3389931e16ff74ac08bccfbef8b6c00e0.jpg)  
Figure 1: Hy-Memory-30B-A3B is trained inside the same deployed harness configurations in both stages. OPD supplies dense teacher supervision on student-written state chains; RL examines the resulting banks, assigns chunk-level credit, and carries the selected bank across windows. In the rollout grid, each cell is the bank after chunk t of rollout i, shaded by its chunk score s [t]. Advantages are normalized within each column (dashed: one chunk-level group), so the best rollout can differ from chunk to chunk. The outlined cell is the final bank with the highest $\textstyle \sum _ { t } s _ { i } [ t ]$ , which becomes the parent of the next window.

## 4 Two-Stage Training

## 4.1 Why two stages

Direct RL from a generic instruction model wastes samples on protocol failures: malformed JSON, wrong operation names, missing fields, and invalid multi-stage control flow. Conversely, pure distillation teaches the model to follow a strong teacher but cannot rank the many valid memory representations by downstream usefulness. OPD learns how to act inside each harness, with dense token-level supervision on the student’s own prompts and outputs. RL learns which valid actions produce useful banks.

## 4.2 Stage 1: Memory on-policy distillation

Sequence construction. Chronologically ordered interaction sequences are segmented into chunks while preserving session boundaries. Chat emphasizes personal facts and preference changes; Working emphasizes tasks, decisions, deadlines, and organizational context; Coding emphasizes repositories, files, commands, implementation decisions, and changing technical state. Each chunk is paired with the harness state produced by preceding student writes, so OPD supervision follows the state distribution encountered during deployment.

On-policy rollout. Each OPD row advances one (sequence, harness configuration) snapshot chain by one chunk. The rollout restores the selected parent bank $M _ { t - 1 }$ , invokes the unmodified harness write path, routes every harness LLM call to the student, applies the student’s operations to produce a candidate $M _ { t } ,$ and promotes one candidate as the parent of the next chunk.

The teacher does not generate a separate ideal trajectory. It scores tokens on the trajectory the student generated. Supervision is placed on student-visited prefixes, reducing the exposure mismatch of static teacher demonstrations (Gu et al., 2024). For memory writing, the benefit extends beyond token prefixes: the future prompts also contain memory states created by the student.

Objective. The frozen Hy3 flagship teacher runs a forward pass on the student’s token stream and returns its top-k distribution. With $\dot { k } = 5 0$ , the response-token objective is a top-k approximation to

reverse KL:

$$
\begin{array} { c } { { \mathcal { L } _ { \mathrm { O P D } } = \displaystyle \frac { 1 } { | \mathcal { T } | } \sum _ { ( i , j ) \in \mathcal { T } } \ell _ { i , j } , } } \\ { { \ell _ { i , j } = \displaystyle \sum _ { v \in \mathrm { t o p } \cdot 5 0 } \pi _ { \theta } ( v \mid x _ { i , < j } ) \left[ \log \pi _ { \theta } ( v \mid x _ { i , < j } ) - \log \pi _ { T } ( v \mid x _ { i , < j } ) \right] . } } \end{array}\tag{2}
$$

MiniLLM similarly motivates reverse KL on student-sampled trajectories (Gu et al., 2024). All LLM calls made by one harness write are retained as explicit segments. Position ids reset at call boundaries, so independent calls do not become one artificial conversation.

Configuration. The student is the Hy3-30B-A3B instruction-tuned checkpoint (∼3B active parameters). The teacher is the Hy3 flagship model, forward only, returning top-50 log-probabilities. There are 2 rollouts per row, learning rate 1e−6 (constant), and entropy coefficient 0.005. Prompt and response limits are 4,096 and 126,976 tokens across segmented calls. The checkpoint that initializes RL is OPD step 375. The unified configurations are Mem0, MemOS, and Hy-Memory Chat, Working, and Coding.

## 4.3 Stage 2: Memory reinforcement learning

RL starts from OPD step 375 and executes the same harness configurations. For each window, eight rollouts begin from independent copies of the same parent bank. They write all chunks, are examined with reward questions, and receive per-chunk scores. The highest-scoring final bank is promoted to the next window in the same sequence. This is on-policy at two levels: tokens are sampled from the current writer, and future prompts are generated from banks produced by that writer.

## 5 Reward Construction: From Failure Surfaces to Attributed Feedback

## 5.1 Why benchmark questions are not enough

Using only final questions from a long-memory benchmark creates a low-density learning problem. A long interaction sequence may contain hundreds of useful facts but only a few questions. An unasked write receives no evidence, and a failed answer does not identify whether the cause was omission, bad wording, stale-state confusion, retrieval failure, or reader failure. Binary exact match further collapses semantically partial answers and can eliminate variance within a small GRPO group (He et al., 2026). We construct reward items from the failure surface of memory writing, not merely from a list of benchmark answers.

## 5.2 Chronological segmentation

Interaction sequences retain session boundaries and chronological order. Each session is segmented independently into chunks of roughly ten turns, so a chunk never crosses a session boundary. The resulting chunks are flattened into one chronological stream; up to five consecutive chunks form one RL window, which may span sessions. Windows from the same sequence are consumed in order, and the selected policy-written bank from one window initializes the next. This preserves long-range state transitions while keeping each optimization step bounded.

## 5.3 Two annotation passes

Local write quality and long-range memory utility require different context, so the two reward types are annotated in separate passes (Table 3).

Why the PRM pass sees the two preceding chunks. A per-chunk probe must test information that chunk t itself introduced. Annotating chunk t in isolation fails in two ways. Coreference: mentions such as “that project” or “the same reviewer” cannot be resolved, so the annotator cannot write self-contained questions about them. Novelty: the annotator cannot tell whether a statement is new in chunk t or repeats something already established, so probes would credit chunk t for facts it did not introduce. The extracted facts of chunks t−1 and t−2 supply the referents and the recently established state. They are given as facts rather than raw text and serve as context only: no probe is generated from them, and they earn no credit for chunk t.

<table><tr><td>Pass</td><td>Annotator context</td><td>Output and target</td></tr><tr><td>Per-chunk probe (PRM)</td><td>Full text of chunk t plus facts of chunks t—1 and Atomic facts, probes, traps, and rubrics; tests t-2</td><td>whether chunk t was written precisely and remains recoverable</td></tr><tr><td>Cross-chunk outcome (ORM) current window</td><td>Earlier-sequence facts plus all facts in the</td><td>Questions, evidence chunks, gold facts, and rubrics; tests relations spanning chunks or sessions</td></tr></table>

Table 3: The two annotation passes and the reward each one feeds.

![](images/ce090d0b4f7619af350cade11446a8395483b8a520db34d4775ae457559ea51c.jpg)  
Figure 2: Construction of the Memory RL reward artifact. Each session is split independently into chunks of about ten turns, so chunks never cross session boundaries. The chronological chunk stream is then grouped into RL windows of at most five chunks; a window may span sessions. The per-chunk probe pass writes the PRM items for chunk t. It reads the full text of chunk t plus the extracted facts of chunks t−1 and t−2, used only to resolve references and decide what is new. The cross-chunk outcome pass writes ORM items from the facts of the whole window and earlier history. Structural filters and a synthetic-answer discriminability test produce the final artifact. The bottom row lists the rules that keep answer information out of the reward path (Section 5.6).

Why not the full history. Dependencies that span more than a few chunks are covered by the crosschunk outcome pass, so the PRM pass does not need the whole sequence. Full history also lowers annotation quality. In a controlled comparison on identical chunks, full-history annotation used 33 times more input tokens, while the two-chunk context produced 25% more proper nouns and 59% more numbers per extracted fact. With the full sequence in view, the annotator’s attention spreads away from the current chunk and its facts become more generic, which is the over-generalization failure (Section 5.4) that probes are meant to detect.

## 5.4 Failure-surface taxonomy

Table 4 lists the local failure modes. Cross-chunk questions add transition, coexistence, temporal, aggregation, multi-hop, and negative recall. This follows the capability decomposition exposed by LoCoMo and LongMemEval (Maharana et al., 2024; Wu et al., 2025), but the questions are generated during reward construction rather than copied from those benchmarks.

<table><tr><td>Probe type</td><td>What the reward tests</td></tr><tr><td>Specificity</td><td>Exact names, numbers, titles, versions, paths, and entities survive writing</td></tr><tr><td>Semantic precision</td><td>Negation, scope, and event meaning are not inverted</td></tr><tr><td>Plain recall</td><td>A durable explicit fact remains answerable</td></tr><tr><td>Secondary dimension</td><td>A dominant topic does not erase a smaller detail</td></tr><tr><td>Transition</td><td>The new state and the state it supersedes are both represented</td></tr><tr><td>Should-not-store</td><td>Greetings, vague praise, and generic assistant text are absent</td></tr><tr><td>Content, not meta</td><td>The memory stores the fact, not a meta-summary of it</td></tr><tr><td>Casual detail</td><td>Informal but durable personal information is retained</td></tr></table>

Table 4: Per-chunk failure-surface probes.

## 5.5 Trap probes and the precision–recall trade-off

Recall alone makes “store everything” attractive. We add trap probes about content that should not have entered long-term memory, such as an assistant’s generic encouragement. Their reference answer is that the bank contains no such record. If the writer stores the irrelevant statement, retrieval exposes it and the answer becomes incorrect. Traps and the bank-growth penalty are complementary. A size penalty discourages indiscriminate volume, while traps directly test semantic selectivity. A model cannot avoid the trap merely by compressing irrelevant text into a shorter entry.

## 5.6 Rubrics and gold-fact attribution

Each probe and outcome question contains weighted credit points with semantic equivalents; explicit full-, partial-, and zero-credit bands; named distractors, including stale or superseded values; evidence chunk ids; and, for outcome questions, the source facts introduced by each evidence chunk. The answer judge performs semantic matching; code only aggregates returned credit.

Gold facts are visible only to the outcome judge. They are not inserted into the bank, retrieval query, or answer-model prompt. Facts from earlier windows may provide context but are marked history-only and cannot generate credit for a current chunk.

## 5.7 Data verification

Reward data must be tested as an instrument, not accepted because it is well-formed JSON. Four layers are enforced. Schema validity removes records with missing fields, unnormalized weights, illegal evidence ids, or illegal grading bands. Causal locality requires per-chunk probes to refer to the current chunk and crosschunk questions to contain multi-chunk evidence when required. Rubric independence checks that credit points do not entail one another and that distractors represent real failure modes. Discriminability checks that the judge scores five synthetic answers per item in the expected order: reference and paraphrase receive full credit, a partial answer receives partial credit, and a distractor or refusal receives zero credit. In the pilot, faithful paraphrases retain an average score of 0.90.

The structural pass keeps approximately 410k of 478k generated probes and 61k of 72k generated crosschunk questions. To control judge cost, at most three positive probes and one trap are retained per chunk. The final artifact contains 164,576 positive probes, 8,599 traps, approximately 60.8k outcome questions, and 222,395 chunk-level gold-fact references.

Future work includes a random human audit with error categories and inter-annotator agreement; a train/evaluation near-duplicate and sequence-level leakage audit; per-scenario and per-failure-type calibration on strong and weak writers; and judge sensitivity to prompt, model, and repeated sampling.

## 5.8 Harness-level artifact statistics

Each complete interaction sequence is assigned to one harness configuration (Table 5). The offline schedule preserves within-sequence window order and consumes every row exactly twice over two epochs.

<table><tr><td>Configuration</td><td>Processed units</td><td>Reward items</td></tr><tr><td>Hy-Memory / Chat</td><td>3,182 windows; 14,519 chunks</td><td>41,990 recall; 2,769 traps; 15,281 ORM</td></tr><tr><td>Hy-Memory / Working</td><td>1,971 windows; 8,902 chunks</td><td>25,272 recall; 1,537 traps; 9,740 ORM</td></tr><tr><td>Hy-Memory / Coding</td><td>1,154 windows; 5,187 chunks</td><td>14,348 recall; 178 traps; 5,337 ORM</td></tr><tr><td>Mem0</td><td>3,182 windows; 14,516 chunks</td><td>41,416 recall; 1,969 traps; 15,228 ORM</td></tr><tr><td>MemOS</td><td>3,182 windows; 14,492 chunks</td><td>41,550 recall; 2,146 traps; 15,241 ORM</td></tr><tr><td>Total</td><td>12,671 windows; 57,616 chunks</td><td>164,576 recall; 8,599 traps; ~60.8k ORM</td></tr></table>

Table 5: Harness-level statistics of the RL reward artifact.

## 6 Reward and Credit Assignment

## 6.1 Three-hop examination

All chunks in a window are written before scoring. For each question, the harness retrieves a memory block from the written bank, the frozen answer model answers using only that block, and the frozen judge compares the answer with the reference and weighted rubric. This scores deployable bank behavior. Directly asking a judge whether retrieved text “contains the answer” would not test whether contradictory or stale memories cause an incorrect response.

## 6.2 Per-chunk recall reward

For chunk t, let $R _ { t }$ be positive probes and $P _ { t }$ traps. With score $z _ { i } \in [ 0 , 1 ]$

$$
\mathrm { p r m } [ t ] = \frac { \sum _ { i \in { \cal R } _ { t } } z _ { i } + \sum _ { j \in { \cal P } _ { t } } z _ { j } } { | { \cal R } _ { t } | + | { \cal P } _ { t } | } .\tag{3}
$$

“PRM” here means per-chunk recall reward, not a reward model over hidden reasoning steps. A chunk with no valid probes receives zero rather than one. Items lost to an answer or judge service failure leave the denominator; policy failures remain scoreable and receive low reward.

## 6.3 Attributed outcome reward

For outcome question $q ,$ the judge returns $c _ { q } [ t ] \in [ 0 , 1 ]$ , the degree to which the answer recovered gold evidence introduced by chunk t. If $Q _ { t }$ is the set of questions citing chunk t,

$$
\operatorname { o r m } [ t ] = { \frac { 1 } { | Q _ { t } | } } \sum _ { q \in Q _ { t } } c _ { q } [ t ] .\tag{4}
$$

Averaging prevents a chunk from receiving more weight merely because the annotator wrote more questions about it.

## 6.4 Bank-growth penalty

Let $g _ { t }$ be bank-token growth divided by input-chunk tokens. Each harness has a free-growth lower threshold $\ell _ { h }$ and a full-penalty threshold $u _ { h }$ :

$$
\begin{array} { c } { { x _ { t } = \displaystyle \mathrm { d i p } \left( \frac { g _ { t } - \ell _ { h } } { u _ { h } - \ell _ { h } } , 0 , 1 \right) , } } \\ { { { } } } \\ { { \displaystyle \mathrm { p e n } [ t ] = \frac { e ^ { 3 x _ { t } } - 1 } { e ^ { 3 } - 1 } . } } \end{array}\tag{5}
$$

Tokens are measured instead of entry count because a writer can grow a bank by lengthening existing records. Shrinking the bank is neither taxed nor directly rewarded. Figure 1 labels this term Penalty.

## 6.5 Combined score

$$
\begin{array} { r l } { { s } [ t ] = \displaystyle \boldsymbol { w } _ { \mathrm { p r m } } \sum _ { k \geq t } \gamma ^ { k - t } \mathrm { p r m } [ k ] } & { } \\ { ~ + \displaystyle \boldsymbol { w } _ { \mathrm { o r m } } \mathrm { o r m } [ t ] - \boldsymbol { w } _ { \mathrm { p e n } } \mathrm { p e n } [ t ] , } \end{array}\tag{6}
$$

where $w _ { \mathrm { p r m } } = 1 , w _ { \mathrm { o r m } } = 2 , w _ { \mathrm { p e n } } = 0 . 3 ,$ and $\gamma = 0$ . Gold-fact ORM attribution handles delayed cross-chunk evidence, so $\gamma = 0$ avoids broadcasting noisy future recall scores backward.

## 6.6 How this resolves sparse ORM

Probes give every chunk a local signal, so a sequence is not limited to a few terminal questions. Weighted rubrics with partial credit and distractors replace binary answer reward. Traps and the bank-growth penalty penalize over-storage. Gold facts attribute outcome evidence to evidence chunks instead of broadcasting one score. The answer model and judge are frozen, so reader improvements cannot hide writer defects. Continuous signals increase within-group resolution. Carrying the selected policy-written bank into the next window preserves cross-session effects. This is closer in spirit to process supervision than terminal-only RL (Lightman et al., 2023), but the supervised unit is an externally verifiable bank consequence rather than a subjective reasoning step.

## 6.7 Chunk-level GRPO

GRPO estimates a baseline from several samples without a separate value model (Shao et al., 2024). Standard GRPO scores each sampled trajectory once, normalizes that score within the group, and applies the resulting advantage to every token of the trajectory. A trajectory-level group is not appropriate here: all eight rollouts share the same bank only at chunk 0; after that, rollout i observes the bank created by its own earlier actions, and a single trajectory score would again broadcast one judgment over all of its writes. We instead compute advantages per chunk, grouping by window, harness configuration, sequence, and chunk position:

$$
A _ { i , t } = \frac { s _ { i } [ t ] - \mu _ { t } } { \sigma _ { t } + \epsilon } ,\tag{7}
$$

where $\mu _ { t }$ and $\sigma _ { t }$ are the mean and standard deviation of $s _ { j } [ t ]$ over the chunk-level group $G _ { t }$ . The best rollout can therefore differ from chunk to chunk: rollout 1 may have the highest score at chunk 2 while rollout 3 is best at chunk $^ { 4 , }$ and each chunk’s writes are reinforced or suppressed according to their own column (Figure 1).

Each rollout contributes one vote to the baseline for a chunk. If a multi-stage harness made three LLM calls for that chunk, all three segments receive the same advantage but do not count as three independent observations. Otherwise the policy could alter the baseline by changing its call count. Groups with fewer than two valid rollouts receive zero advantage.

## 6.8 Bank carry-forward

Bank selection is a separate, rollout-level decision. At the end of a window, the eight final banks are ranked by $\textstyle \sum _ { t } s [ t ]$ ]. The winner becomes the initial state of the next window in that sequence; the others are discarded. If all candidates fail because of platform faults, the previous bank is reused. Bank state is checkpointed with model state.

An inherited error from an earlier optimizer step cannot receive retrospective gradient. Carrying imperfect policy-written banks is still preferable to teacher forcing with an ideal bank: deployment also requires the writer to operate on its own prior mistakes. An inherited error common to all candidates also tends to cancel in the group-relative baseline rather than being assigned to the current chunk.

## 7 Training System

Each harness is packaged behind a common interface: bank\_open, write, search, bank\_stats, and bank\_close. The harness owns its native control flow. Harness-invoked LLM calls are routed to the policy endpoint and recorded at generation time with token ids, rollout log-probabilities, segment boundaries, and MoE routing. This follows the harnessed-agent-RL pattern in which the deploy-time loop is kept intact while model calls connect to a trainer (Luo et al., 2025). Retokenizing text after execution can diverge from the sampled stream, and recomputing rollout log-probabilities with the training engine destroys the intended importance ratio.

<table><tr><td>Memory writer</td><td>Mem0</td><td>Hy-Mem.</td><td>MemOS</td><td>EverMem.</td><td>Avg.</td></tr><tr><td>Hy-Memory-30B-A3B</td><td>54.02</td><td>60.55</td><td>50.51</td><td>56.89</td><td>55.49</td></tr><tr><td>Claude Opus 5</td><td>53.43</td><td>60.53</td><td>50.50</td><td>54.04</td><td>54.62</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>49.73</td><td>60.20</td><td>49.17</td><td>55.47</td><td>53.64</td></tr><tr><td>Hy3</td><td>45.99</td><td>58.06</td><td>50.19</td><td>52.30</td><td>51.63</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>47.51</td><td>55.87</td><td>47.36</td><td>50.75</td><td>50.37</td></tr><tr><td>Hy3-A3B Base</td><td>40.20</td><td>50.39</td><td>49.05</td><td>50.98</td><td>47.66</td></tr><tr><td>Qwen3-30B-A3B</td><td>39.81</td><td>45.67</td><td>45.44</td><td>46.57</td><td>44.37</td></tr></table>

Table 6: Six-metric means on fixed 20% benchmark subsamples. The macro average weights the four harnesses equally.

Stage 2 starts from the OPD model, on Hy3-30B-A3B (∼3B active parameters), using 16 nodes × 8 H20 GPUs (128 GPUs). A batch is 64 windows × 8 rollouts, or 512 trajectories. The objective is a PPO-style clipped surrogate with chunk-level GRPO advantages, aggregated as sequence mean then token mean, with no KL penalty. MoE routing from the rollout is replayed during training. The learning rate is 5e−6, constant. The context is 28,672 prompt tokens plus 4,096 response tokens. The resulting post-RL model is released and evaluated as Hy-Memory-30B-A3B. Model, optimizer, schedule, and selected banks are checkpointed every 10 steps during training.

## 8 Experiments

## 8.1 Evaluation principles

The evaluation changes only the memory writer. For each comparison, the harness version and prompts are fixed, retrieval configuration is fixed, and the answer model, judge, and other framework-side components are held fixed. Throughout this report, every reference to a DeepSeek Flash endpoint denotes DeepSeek V4.1 Flash. Decoding settings are matched where APIs permit, each writer builds its own bank from an empty state, and test examples and scoring rules are identical. This is necessary because memory quality is otherwise confounded with retrieval depth, reader capability, auxiliary-model capability, or a pre-existing bank.

Sampled benchmark matrix. For computational efficiency, every benchmark in the main four-harness matrix is evaluated on a fixed 20% subsample. The evaluated post-RL model is Hy-Memory-30B-A3B. Scores are deterministic outputs of the fixed evaluation pipeline; no missing judgment is imputed as zero. Each harness score is the equal-weight average of six metrics: LoCoMo Judge, LongMemEval, PersonaMem-v2, Travel SPS, Shopping PS, and Code Memory Bench (CMB). CMB is itself the equalweight average of two repository QA evaluations. The overall score is the equal-weight average of the four harness scores.

## 8.2 Final cross-harness results

Hy-Memory-30B-A3B ranks first on all four harnesses. Relative to Opus 5, it gains 0.60 points on Mem0, 0.02 on Hy-Memory, 0.01 on MemOS, and 2.86 on EverMemOS, for a macro-average gain of 0.87 points. Relative to DeepSeek V4.1 Flash, it gains 4.29 points on Mem0, 0.35 on Hy-Memory, 1.34 on MemOS, and 1.42 on EverMemOS, for a macro-average gain of 1.85 points. Relative to the Hy3-A3B base checkpoint, the macro average improves by 7.84 points. The 0.01–0.02 point margins on MemOS and Hy-Memory indicate parity with Opus 5 under this deterministic sampled evaluation, not a statistically resolved difference. The stronger evidence is the consistent ranking and the larger margin on the held-out framework.

![](images/23514f7fc9f8953ad4358d5ef681b0ed98495e80af5e6445ac2fb9f032b09e3e.jpg)  
Figure 3: Cross-harness macro-average scores on the fixed 20% benchmark matrix. Each harness score averages six metrics equally; Overall averages the four harnesses equally. EverMemOS is held out from training.

The six metrics stress different behaviors, so the average improvement does not require Hy-Memory-30B-A3B to win every column. Its strongest recurring gains are on Travel SPS and CMB. On the held-out EverMemOS harness, the gain is driven particularly by Travel SPS (+10.65 points) and CMB (+13.66 points). Appendix A reports all six benchmark scores for every model and harness.

## 8.3 OOD framework generalization

EverMemOS is not part of OPD or RL training. It is included only at evaluation time as a held-out framework and is the strongest open-source community memory baseline in this evaluation. Hy-Memory-30B-A3B reaches 56.89, compared with 55.47 for DeepSeek V4.1 Flash, 54.04 for Opus 5, and 52.30 for Hy3. This result separates model-level memory-writing skill from memorization of the training harness protocols.

## 8.4 Future ablations and audits

This report does not include controlled ablations of chunk-level versus window-level GRPO, gold-fact attribution versus broadcast ORM, positive probes versus probes plus traps, the bank-growth penalty, binary versus rubric-based rewards, RL initialization from OPD versus generic SFT, or single-harness versus mixed-harness training. We leave these comparisons to future work. Additional reward-data audits should measure human–judge agreement, sensitivity to the judge model and prompt, score distributions across synthetic answer types, separation between strong and weak memory writers, and the fraction of zero-variance rollout groups before and after reward decomposition.

## 9 Engineering Lessons: Silent Failures

Several failures looked healthy until an invariant was checked. A bank that did not persist across windows still left files on disk and produced nonzero rewards; the check is to reopen a checkpointed bank and verify a known record before training. A teacher that returned no top-k values still returned successful requests, but OPD gradients were zero; missing teacher fields must raise, and teacher-token coverage must be monitored. When the per-chunk mapping was absent, a fallback broadcast one score to every segment; missing chunk-to-segment mappings must raise. A configuration omitted from a hard-coded schedule still had valid artifacts on disk; schedule manifests must be audited against the declared harness set. String rules that graded semantics produced a plausible aggregate reward; semantic conditions belong in judge rubrics, not substring code. A “read all” that silently paginated made the bank entry count plateau at a default limit; pagination and token accounting must be tested explicitly. Failed writes checkpointed as empty were marked complete; platform faults must be retryable and excluded from denominators. Sandboxes that shared a trajectory identity completed requests attached to the wrong rollout; every request must validate bank and trajectory identity. Malformed output treated as an environment error made bad actions disappear from training; policy mistakes should remain valid low-reward trajectories.

The governing rule is: a bad policy action must receive low reward; a platform failure must not be misrepresented as policy behavior.

## 10 Limitations and Future Work

Attribution is evidence-based rather than counterfactual: recovered gold facts do not prove that a specific write caused an answer. Credit is truncated across optimizer steps, so earlier-bank errors cannot receive retrospective gradient. One-vote grouping fixes the baseline, but multi-call chunks can still carry more loss mass. Rubric rewards reduce sparsity but require human-agreement and judge-sensitivity checks. The reward artifact covers Chat, Working, and Coding but not every long-term-memory workload; EverMemOS is only one held-out framework. The primary matrix uses deterministic 20% subsamples, so 0.01–0.02 point margins are not statistically significant; repeated training seeds remain future work. Finally, pooled scores can hide whether gains come from extraction, update, deletion, reconciliation, or retrieval-facing representation.

## 11 Conclusion

Memory writing is a long-horizon control problem over an external state, not merely a summarization prompt. Hy-Memory-30B-A3B first learns to operate real memory harnesses through on-policy distillation, then optimizes the answerability and selectivity of the banks it creates through Memory RL. The central technical contribution is the reward interface: failure-driven probes make supervision dense, cross-chunk questions measure downstream utility, gold facts attribute that utility to source chunks, traps and growth penalties close the store-everything shortcut, and chunk-level GRPO compares actions at the correct decision boundary.

Hy-Memory-30B-A3B ranks first on Mem0, Hy-Memory, MemOS, and held-out EverMemOS. On fixed 20% benchmark subsamples, it averages 55.49% across six metrics and four harnesses, ahead of Claude Opus 5 at 54.62%, DeepSeek V4.1 Flash at 53.64%, and the Hy3-A3B base checkpoint at 47.66%. The OOD EverMemOS result provides the clearest margin over Opus 5 (+2.86 points). Together, these results support memory writing as a transferable model capability rather than a collection of framework-specific prompt heuristics.

## Resource Availability

The Hy-Memory framework code and evaluation utilities are available at https://github.com/ Tencent-Hunyuan/Hy-Memory. The Hy-Memory-30B-A3B model weights, configuration, and model card are available on Hugging Face at https://huggingface.co/tencent/Hy-Memory-30B-A3B.

## References

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Tianxiang Fei, Cheng Chen, Yue Pan, Mao Zheng, and Mingyang Song. CodeDelegator: Mitigating context pollution via role separation in code-as-action agents. arXiv preprint arXiv:2601.14914, 2026a.

Tianxiang Fei, Mingyang Song, Mao Zheng, and Xiang Yu. Memory beyond recall: A dual-process cognitive memory system for self-evolving LLM agents. arXiv preprint arXiv:2606.09483, 2026b.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: On-policy distillation of large language models. In ICLR, 2024.

Xinjie He, Zhiyuan Lin, Su Liu, Jialun Wu, Qiyang Xie, Weikai Zhou, and Shuai Xiao. What training data teaches RL memory agents: An empirical study of curriculum effects in memory-augmented QA. arXiv preprint arXiv:2605.23067, 2026.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory OS of AI agent. arXiv preprint arXiv:2506.06326, 2025.

Zhiyu Li et al. MemOS: A memory OS for AI system. arXiv preprint arXiv:2507.03724, 2025.

Hunter Lightman et al. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Xufang Luo et al. Agent lightning: Train ANY AI agents with reinforcement learning. arXiv preprint arXiv:2508.03680, 2025.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. arXiv preprint arXiv:2402.17753, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. Mem-α: Learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911, 2025.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In ICLR, 2025.

Derong Xu et al. Learning how and what to memorize: Cognition-inspired two-stage optimization for evolving memory. arXiv preprint arXiv:2605.00702, 2026.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agentic memory for LLM agents. In NeurIPS, 2025.

Sikuan Yan et al. Memory-R1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025.

Hongli Yu et al. MemAgent: Reshaping long-context LLM with multi-conv RL-based memory agent. In ICLR, 2026.

## A Detailed Benchmark Results

Tables 7–10 report the six benchmark scores used to compute each harness mean in Table 6. All scores use the same fixed 20% benchmark subsamples as the main evaluation. Hy-Memory-30B-A3B uses the post-RL model reported throughout this paper. Checkpoint identifiers are omitted.

<table><tr><td>Writer</td><td>LoCoMo</td><td>LME</td><td>PM-v2</td><td>Travel</td><td>Shop.</td><td>CMB</td><td>Avg.</td></tr><tr><td>Hy-Memory-30B-A3B</td><td>59.54</td><td>73.00</td><td>51.62</td><td>54.51</td><td>18.33</td><td>67.13</td><td>54.02</td></tr><tr><td>Claude Opus 5</td><td>69.47</td><td>66.00</td><td>53.58</td><td>42.98</td><td>17.78</td><td>70.74</td><td>53.43</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>58.78</td><td>69.39</td><td>49.56</td><td>35.07</td><td>16.67</td><td>68.92</td><td>49.73</td></tr><tr><td>Hy3</td><td>41.22</td><td>62.00</td><td>44.44</td><td>46.87</td><td>17.22</td><td>64.16</td><td>45.99</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>50.00</td><td>69.00</td><td>49.38</td><td>36.39</td><td>20.00</td><td>60.30</td><td>47.51</td></tr><tr><td>Hy3-A3B Base</td><td>50.00</td><td>47.00</td><td>40.82</td><td>37.91</td><td>14.44</td><td>51.03</td><td>40.20</td></tr><tr><td>Qwen3-30B-A3B</td><td></td><td>46.1856.00</td><td>42.32</td><td>31.98</td><td>16.67</td><td>45.68</td><td>39.81</td></tr></table>

Table 7: Detailed results on the Mem0 harness.

<table><tr><td>Writer</td><td>LoCoMo</td><td>LME</td><td>PM-v2</td><td>Travel</td><td>Shop.</td><td>CMB</td><td>Avg.</td></tr><tr><td>Hy-Memory-30B-A3B</td><td>73.28</td><td>89.00</td><td>51.54</td><td>57.65</td><td>21.11</td><td>70.73</td><td>60.55</td></tr><tr><td>Claude Opus 5</td><td>74.43</td><td>89.69</td><td>48.71</td><td>51.34</td><td>23.33</td><td>75.70</td><td>60.53</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>72.90</td><td>85.86</td><td>49.00</td><td>53.46</td><td>22.78</td><td>77.18</td><td>60.20</td></tr><tr><td>Hy3</td><td>69.47</td><td>89.00</td><td>52.76</td><td>49.01</td><td>21.67</td><td>66.44</td><td>58.06</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>70.99</td><td>84.85</td><td>43.56</td><td>49.82</td><td>20.00</td><td>65.98</td><td>55.87</td></tr><tr><td>Hy3-A3B Base</td><td>66.41</td><td>87.00</td><td>49.84</td><td>31.36</td><td>17.78</td><td>49.97</td><td>50.39</td></tr><tr><td>Qwen3-30B-A3B</td><td>64.12</td><td>82.00</td><td>41.06</td><td>32.83</td><td>18.89</td><td>35.10</td><td>45.67</td></tr></table>

Table 8: Detailed results on the Hy-Memory harness.

<table><tr><td>Writer</td><td>LoCoMo</td><td>LME</td><td>PM-v2</td><td>Travel</td><td>Shop.</td><td>CMB</td><td>Avg.</td></tr><tr><td>Hy-Memory-30B-A3B</td><td>62.98</td><td>73.00</td><td>51.92</td><td>24.15</td><td>21.11</td><td>69.91</td><td>50.51</td></tr><tr><td>Claude Opus 5</td><td>63.74</td><td>79.38</td><td>52.84</td><td>22.14</td><td>19.44</td><td>65.46</td><td>50.50</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>61.07</td><td>73.00</td><td>50.68</td><td>18.61</td><td>21.67</td><td>69.97</td><td>49.17</td></tr><tr><td>Hy3</td><td>62.98</td><td>81.00</td><td>49.64</td><td>18.87</td><td>18.89</td><td>69.78</td><td>50.19</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>55.34</td><td>79.00</td><td>51.34</td><td>20.08</td><td>15.56</td><td>62.82</td><td>47.36</td></tr><tr><td>Hy3-A3B Base</td><td>58.40</td><td>73.00</td><td>50.86</td><td>26.05</td><td>21.11</td><td>64.88</td><td>49.05</td></tr><tr><td>Qwen3-30B-A3B</td><td>59.16</td><td>73.00</td><td>48.28</td><td>23.03</td><td>16.67</td><td>52.49</td><td>45.44</td></tr></table>

Table 9: Detailed results on the MemOS harness.

<table><tr><td>Writer</td><td>LoCoMo</td><td>LME</td><td>PM-v2</td><td>Travel</td><td>Shop.</td><td>CMB</td><td>Avg.</td></tr><tr><td>Hy-Memory-30B-A3B</td><td>73.28</td><td>76.00</td><td>53.28</td><td>56.56</td><td>19.05</td><td>63.19</td><td>56.89</td></tr><tr><td>Claude Opus 5</td><td>73.28</td><td>80.41</td><td>51.19</td><td>45.91</td><td>23.89</td><td>49.53</td><td>54.04</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>72.52</td><td>82.83</td><td>49.74</td><td>59.47</td><td>18.89</td><td>49.38</td><td>55.47</td></tr><tr><td>Hy3</td><td>68.70</td><td>83.00</td><td>48.68</td><td>46.52</td><td>19.44</td><td>47.44</td><td>52.30</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>69.08</td><td>79.00</td><td>50.60</td><td>43.53</td><td>18.77</td><td>43.52</td><td>50.75</td></tr><tr><td>Hy3-A3B Base</td><td>64.50</td><td>81.00</td><td>52.14</td><td>41.74</td><td>20.56</td><td>45.93</td><td>50.98</td></tr><tr><td>Qwen3-30B-A3B</td><td></td><td>67.94 60.00</td><td>47.10</td><td>44.80</td><td>18.33</td><td>41.27</td><td>46.57</td></tr></table>

Table 10: Detailed results on the held-out EverMemOS harness.

## B Hyperparameters

<table><tr><td></td><td>Stage 1: OPD</td><td>Stage 2: RL</td></tr><tr><td>Model</td><td>Hy3-30B-A3B</td><td>Hy3-30B-A3B</td></tr><tr><td>Init.</td><td>Instruction- tuned</td><td>OPD step 375</td></tr><tr><td>Harnesses</td><td>Mem0, MemOS, Hy-Memory</td><td>Same</td></tr><tr><td>Ordering</td><td>Snapshot chain</td><td>Chronological windows</td></tr><tr><td>Unit</td><td>One chunk write</td><td>One window, ≤5 chunks</td></tr><tr><td>Rollouts</td><td>2 per row</td><td>8 per window</td></tr><tr><td>Signal</td><td>Top-50 reverse KL</td><td>PŘM +2 ORM -0.3 Penalty</td></tr><tr><td>Advantage</td><td>Token-level</td><td>Chunk-level GRPO</td></tr><tr><td>LR</td><td>1e-6</td><td>5e-6</td></tr><tr><td>Length</td><td>4k / 127k</td><td>29k / 4k</td></tr><tr><td>Output</td><td>OPD model ini- tializes RL</td><td>Hy-Memory- 30B-A3B</td></tr></table>

Table 11: Training hyperparameters for the two stages.