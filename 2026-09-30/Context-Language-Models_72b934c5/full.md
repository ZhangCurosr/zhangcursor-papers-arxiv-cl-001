# Context Language Models

Rulin Shao<sup>1</sup>,<sup>2</sup>, Shannon Zejiang Shen<sup>3</sup>, Junjie Oscar Yin<sup>1</sup>,<sup>2</sup>, Yuetai Li<sup>1</sup>, Minheng Wang<sup>1</sup>, Hamish Ivison<sup>1</sup>, Radha Poovendran<sup>1</sup>, Nathan Lambert<sup>4</sup>, Teng Xiao<sup>1</sup>, Mike Lewis<sup>2</sup>, Wen-tau Yih<sup>2</sup>, Luke Zettlemoyer<sup>1</sup>,<sup>2</sup>, Pang Wei Koh<sup>1</sup>

<sup>1</sup>University of Washington, <sup>2</sup>Meta Superintelligence Labs, <sup>3</sup>MIT, <sup>4</sup>Trillium Labs

We introduce Context Language Models (CLMs), language models that natively manage their own context. We implement this by treating the context as a file and allowing the model to make unrestricted updates to this file. This allows the model to learn what is most important to maintain in context, and naturally extends to multi-agent systems where multiple agent contexts coexist as files. Building CLMs zero-shot with existing models outperforms SOTA context-management strategies across a variety of tasks: 11.4% higher accuracy with 21.5% fewer FLOPs on BrowseComp-Plus, 5% higher scores with 59% fewer FLOPs on 12-hour EdgeBench, and 65% greater improvement with the same compute on a 24-hour multi-repository agent-swarm task. Moreover, by shifting context management from external harness control to intrinsic model behavior, CLMs naturally enable both in-context and parametric learning of context-management strategies. We show that CLMs can be steered with natural-language instructions evolved through a standard skill-optimization loop, improving held-out accuracy by up to 35.9 points on a context-management task while reducing compute. We also introduce an online reinforcement learning method for CLMs, improving Qwen3.5-9B performance on BrowseComp-Plus by 47.6% while using 12% fewer FLOPs. Finally, we co-design Suffix Cache Reuse for CLM serving, further reducing server-side compute by 35% relative to standard SGLang at matched performance.

Date: September 30, 2026 Correspondence: Rulin Shao at rulin@cs.washington.edu Code: https://github.com/facebookresearch/context-language-models

∞Meta

## 1 Introduction

Despite the fact that context is the cornerstone that allows a language model (LM) to process and retain information over time, context management is not traditionally a native LM capability. Instead, prior work mostly relies on harnesses, either hand-engineered (Cassano and Rush, 2026; OpenAI, 2026a; Merrill et al., 2026) or optimized ofline by agents (Lee et al., 2026). Recent work adds a constrained set of tools with fixed strategies such as compaction, ofloading, and retrieval, to the agent’s action space (Yan et al., 2026; Yu et al., 2026; Li et al., 2026c; Liu et al., 2026; Zhang et al., 2025a). In contrast, we show that giving LMs unrestricted access to manage their own context outperforms human-designed baselines, enabling adaptive and creative context-management strategies to emerge. Our findings echo The Bitter Lesson (Sutton, 2019): we should let LMs search for and learn better strategies that go far beyond existing human priors.

Concretely, we introduce Context Language Models (CLMs), which are natively capable of managing their own context. We show existing LMs can be turned into strong CLMs and can be further improved through in-context learning and reinforcement learning. Formally, a CLM parametrized by � makes context an artifact of the LM: $c _ { t + 1 } = \mathsf { \check { f } } _ { \theta } ^ { \mathrm { C L M } } ( c _ { t } )$ , where $c _ { t }$ is the context at turn � and $\overset { \mathtt { i } } { f _ { \theta } ^ { \mathrm { C L M } } }$ can be an arbitrary function controlled by CLMs. In contrast, a standard LM simply appends new tokens to the existing context: $c _ { t + 1 } = c _ { t } \oplus f _ { \theta } ^ { \mathrm { L M } } ( c _ { t } )$

We implement CLMs by treating context as a file. Specifically, we mirror the context into a storage space with LM write access. The LM can either append newly generated tokens or use Bash to freely edit the context file, with each modification immediately synchronized to the LM’s live context for the next turn. This design naturally extends to multi-agent systems, where multiple context files can coexist and be managed by CLMs for agent-swarm or subagent workloads.

## Context Language Models

![](images/037fdda4937119e5ac2452ff26b158a24ae945eadf8b3f9edaf4689af4efa9f9.jpg)

![](images/5ccf4fb0e20ab3417b5ba553417a8197860cf05e0bdf9b94c2ce48546b472cfa.jpg)

![](images/e5f8b3d0b4f2dd2cbc0a743a6725c0297449c4ac1bdef33a1e8f11b3f022d704.jpg)

![](images/9079f7a76731e7724d74cd0e46c5d601e4e18f9317bd9e7b0a48e209fb10ef49.jpg)

![](images/0690de9777db206c10184c5ca2082404cda73e0a92ed5ac8095c7f9c07344457.jpg)

![](images/9e1fdebc4db90412e63f310a017117542794743da6e0a19e7ce23dfdbbfb47b2.jpg)

![](images/d15317d91fabf6717fb770ea4bd05947c2a08c6dd27905998342ea3c0e8fb9ed.jpg)

![](images/86ccb17617327440d1bf40f8b9d550241550ce0c8e338db0aa044c523f27c052.jpg)  
Figure 1 Context Language Models (CLMs) natively manage their own context by treating context as a file. CLMs work out of the box and can be further improved through in-context learning and reinforcement learning. (a) Qualitative examples of creative context-management behaviors introduced by CLM. (b) Out of the box, CLMs improve performance at lower cost on BrowseComp-Plus, a deep-research benchmark, and Software World, where an agent swarm jointly optimizes six interdependent repositories. (c) CLMs can follow textual instructions to adopt corresponding context-management strategies (top) or evolve better strategies through a skill-evolution loop on ContextBench (bottom). (d) CLMs can explore and internalize context-management strategies through online reinforcement learning. By using a success-gated eficiency advantage for stepwise GRPO, we improve both accuracy and eficiency for CLMs simultaneously.

Arbitrary context edits in CLMs pose new challenges for existing serving systems, which typically only reuse cached states for matching prefixes, forcing re-prefilling after in-the-middle edits. We account for this by introducing prefix-reuse FLOPs, which capture the trajectory-wide inference costs of decoding, prefilling, and re-prefilling in standard LM serving, and show that CLMs remain more compute-eficient under standard serving through better context management. We further develop Suffix Cache Reuse (SCR), which reuses cached states beyond the matching prefix to reduce re-prefilling while empirically preserving task performance. We also introduce ContextBench as a diagnostic benchmark that decouples context management from reasoning and knowledge, revealing the limitations of existing context management methods.

We show that CLMs, applied zero-shot to models like Qwen3.6-27B and GPT5.6-Sol, outperform existing baselines and task-specific harnesses across diverse long-horizon tasks, ranging from hundreds to thousands of turns and up to 24 hours of runtime. Compared with existing harness-defined and action-based methods, CLM achieves 11.4% higher accuracy with 21.5% fewer prefix-reuse FLOPs than the strongest baseline on the deep-research benchmark BrowseComp-Plus (Chen et al., 2025), while matching the strongest baseline’s accuracy on the terminal-coding benchmark TerminalBench 2.1 (Merrill et al., 2026) with 29.5% fewer FLOPs. On mathematical optimization tasks, CLM outperforms specialized evolutionary harnesses such as OpenEvolve (Sharma, 2025) by up to 16.8% (Heilbronn) and 3.0% (circle packing). On long-running software optimization, CLM outperforms Codex-style summarization: on 12-hour EdgeBench (Zhu et al., 2026) (a 10-task subset), CLM scores 5% higher while using 59% fewer prefix-reuse FLOPs, and on a 24-hour six-repository agent-swarm task, it achieves 65% greater end-to-end speedup at the same compute. Moreover, when Sufix Cache Reuse is further applied, it helps reduce server-side compute by 35% with matched performance compared with standard SGLang serving. Qualitatively, we find that CLMs come up with novel emergent behaviors such as defining and maintaining trackers for multi-agent orchestration, introducing a new chat role for internal notes, and defining reusable context-management functions.

By shifting context management from external harness control to intrinsic model behavior, CLMs naturally enable in-context learning and parametric learning for context management. We first show that users can steer context management simply by telling the agent their desired strategy. In addition, CLMs can evolve an in-context skill document that captures useful context-management procedures for future reuse, improving held-out accuracy on ContextBench by up to 35.9 points at lower compute. For training, we introduce a success-gated efficiency advantage in stepwise GRPO (Shao et al., 2024) that rewards eficient CLM trajectories among successful ones. Training Qwen3.5-9B on deep research tasks in this manner improves CLM from 28.8% to 42.5% on BrowseComp-Plus, outperforming a Codex-style summary harness trained with the same recipe by 0.4 points while using 38.8% fewer FLOPs. Overall, we show that by treating context management as a native LM capability, CLMs enable more efective and eficient strategies to be searched for and learned.

## 2 Related Work

From harness-defined to action-based context management. Most existing harnesses compact accumulated histories according to a fixed harness policy (Cassano and Rush, 2026; OpenAI, 2026a; Merrill et al., 2026; Zhou et al., 2026), such as at a predefined length threshold or at every turn. Recent work gives the model increasing control through human-defined actions: AutoCompact (Zhang et al., 2026b) and Self-Compact (Li et al., 2026b) let the model decide when to compact; Context-as-a-Tool (Liu et al., 2026) exposes model-triggered compaction over a predefined portion of the context; ACM (Li et al., 2026c) adds model-triggered ofloading and retrieval; and Sculptor (Li et al., 2026a) lets the model select context fragments to operate on. Across this progression, model autonomy increases but remains restricted to a human-defined action space. Our work pushes this autonomy to its limit by granting the model full agency over its context.

Context as a REPL variable. Recursive Language Models (RLMs) (Zhang et al., 2025a) treat a long input as a read-eval-print loop (REPL) variable that LMs can recursively access on demand. This addresses when and what information to read into the context. However, RLMs do not address how the live context itself should be managed. Retrieved information is still appended to the live context, which continues to grow over time. In contrast, our work makes the live context editable, giving CLMs full control over their context.

We provide extended related work in Appendix A, with more detailed comparisons to existing contextmanagement baselines and a discussion of meta-harness optimization, reinforcement learning, cache reuse, and the relationship between context management and external memory.

## 3 Pilot Study with ContextBench: A Diagnostic for Context Management

We start with a pilot study showing how existing context management strategies can fail in simple tasks. To isolate context management from other reasoning or knowledge capabilities, we developed ContextBench, a diagnostic evaluation suite with the four synthetic tasks shown in Figure 2: Needle Retention tests selective verbatim retention, simulating the need to preserve important information over time; Sudoku Sketchpad tests surgical in-place updates to the live context by maintaining a Sudoku board as users stream in moves; KV Store and Log Triage test exact recall through ofloading and retrieval of massive values and working logs. We evaluate ContextBench with several context-management strategies, including Mini-SWE-Agent (Yang et al., 2024) (the base harness without context management), Codex-style Summary (OpenAI, 2026a), Context Folding (Sun et al., 2025), and RLM, Self-Compact, and ACM, as introduced in Section 2. We also evaluate CLM, which will be introduced in Section 4. Details of the evaluation and qualitative examples for ContextBench are provided in Appendix D.

We fix the context limit at 32K and vary the context pressure (the ratio of input volume to context limit) up to 24×. The results in Figure 2 show that these fixed strategies cannot adapt well to the live context: Summary-based compaction can lose or hallucinate information on Needle Retention and Sudoku Sketchpad; methods without flexible in-place editing must regenerate the full Sudoku state for every fine-grained user edit; and standard coding tools can ofload information on KV Store and Log Triage but cannot evict it from the live context on demand. As a result, none of the existing methods performs perfectly even on these simple tasks. These failures motivate fully adaptive, model-controlled context management.

![](images/f45a5992b06fcd9b1e4179c285f01a9ea6d0631002f36631a74046f835a5011b.jpg)  
Figure 2 Illustration of the four tasks in ContextBench and a performance comparison of CLM against baselines using GPT-5.4 with a 32K context limit.

## 4 Context Language Models (CLMs)

## 4.1 Formal Definition and Implementation with Context as a File

Context Language Models (CLMs) generalize the append-only context transition of a standard LM to a model-controlled context transition. Standard LMs append model output to the current context:

$$
c _ { t + 1 } = c _ { t } \oplus f _ { \theta } ^ { \mathrm { L M } } ( c _ { t } ) ,\tag{1}
$$

where ⊕ denotes concatenation. In contrast, CLMs delegate full responsibility for maintaining the context to the CLM itself, directly creating the next context:

$$
c _ { t + 1 } = f _ { \theta } ^ { \mathrm { C L M } } ( c _ { t } ) ,\tag{2}
$$

where $f _ { \theta } ^ { \mathrm { C L M } }$ can be an arbitrary function controlled by CLMs. Eq. 2 subsumes prior approaches that expose a set of context-management tools through the harness. However, prior work requires context-management functions to be predefined in the harness. Our work instead makes CLMs responsible for defining these functions themselves as a meta-capability, either implicitly through their planning or explicitly as reusable functions, with one explicit example shown in Figure 3d.

Context-as-a-file implementation for CLMs. To implement CLMs, we mirror the LM’s live context as a directly editable file and provide its path in the system prompt. The LM can edit this file using general Bash commands, just as it would edit other files in storage. Unlike ordinary files, edits to the context file are automatically synchronized with the LM’s context and sent to the LLM server for continued generation. When the LM does not edit the context file, the generated tokens are appended to the existing context by default. This implementation balances context reuse with the flexibility to edit the context.

Multi-agent extensions of CLMs. Our implementation naturally extends to multi-agent workflows by allowing multiple context files to coexist and remain synchronized with their respective LLM servers. For example, an agent swarm can be implemented by initializing the workspace with multiple context files, while subagents can be initialized and terminated by creating and deleting additional context files.

![](images/e465a4f6dd5c4c23df2c439b7f7c633d35b951a365a5a50d76c26db24278f4fe.jpg)  
Figure 3 Qualitative examples of CLM context-management behaviors. CLMs treat context as a file and can arbitrarily edit it using general code interface.

Qualitative examples. We show qualitative examples of CLMs managing context as a file in Figure 3, revealing both novel context-management behaviors and efective compaction strategies. For multi-agent orchestration, CLM maintains an in-context scoreboard and updates agent status through 163 in-place edits while keeping the context at only 6–8K tokens (a). It can create new internal roles such as “notes” when rewriting its context (b), and use loops to remove irrelevant search results or compact overlong observations (c). CLM can also define and reuse helper functions: in (d), it invokes ‘compact\_turns’ 37 times to maintain a progress note while compacting detailed observations. Finally, it reproduces efective compaction behaviors by compressing 21K tokens into answer-relevant summaries or preserving untried ideas for future explorations (e). We collected these examples from the zero-shot CLM evaluation experiments in Section 5.1.

Efficiency metrics for CLMs. A common serving optimization is prefix-cache reuse, in which cached states are reused for matching prefixes, while all tokens from the first prefix mismatch onward must be re-prefilled, as can occur after an in-the-middle edit. To account for this, we measure theoretical inference FLOPs using a metric we call prefix-reuse FLOPs (see Appendix C for details). Formally,

$$
\mathrm { F L O P s } _ { \mathrm { P r e f i x - r e u s e } } = \underbrace { \mathrm { F L O P s } _ { \mathrm { P r e f i l l } } ( \mathrm { u n m a t c h e d c o n t e x t s u f f i x } ) } _ { \mathrm { t o k e n s ~ f r o m ~ t h e ~ f i r s t ~ p r e f i x ~ m i s m a t c h ~ o r w a r d } } + \underbrace { \mathrm { F L O P s } _ { \mathrm { d e c o d e } } ( \mathrm { g e n e r a t e d ~ t o k e n s } ) } _ { \mathrm { n e w ~ o u t p u t ~ t o k e n s } } .\tag{3}
$$

## 4.2 In-Context Learning and Reinforcement Learning for CLMs

By treating context management as an LM-native capability, CLMs can learn better strategies in context or in weights.

Steering CLMs with in-context instruction or skill documents. Let � denote an in-context instruction or skill document. CLMs can be steered by simply providing � as additional in-context guidance to the CLM:

$$
c _ { t + 1 } = f _ { \theta } ^ { \mathrm { C L M } } ( c _ { t } ; s ) .\tag{4}
$$

Evolving CLMs with an optimization loop. CLMs can also be optimized through textual evolution. For task instance $x ,$ let �(�; �) be the trajectory induced by Eq. 4, and $\hat { R ( \tau ) }$ a trajectory-level reward. We optimize

$$
\begin{array} { r } { \boldsymbol { s } ^ { * } = \arg \operatorname* { m a x } _ { \boldsymbol { s } } \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ R ( \boldsymbol { \tau } ( \boldsymbol { x } ; \boldsymbol { s } ) ) \right] , } \end{array}\tag{5}
$$

while keeping everything else fixed. In our implementation, we use a prompt-evolution loop (Agrawal et al., 2026): In each round, the agent produces rollouts on the training split, and a proposer model uses the resulting traces to generate candidate skills. We evaluate these candidates on the development split and select the skill for the next round. After evolution concludes, we evaluate the final selected skill once on the held-out test split. The optimizer may be either a stronger external model (assisted evolution) or the agent model itself (self-evolution), allowing context-management skills to evolve in context.

Reinforcement Learning for CLMs. CLMs can also learn context-management strategies through reinforcement learning and internalize them in model weights. Since context edits change the input across turns, we use stepwise GRPO (Shao et al., 2024). For each prompt, we sample a group of complete agent trajectories and compute the standard GRPO advantage from their trajectory-level outcome rewards. We then assign each trajectory’s advantage to all of its constituent segments, so every model call is trained with the outcome of the full trajectory.

Outcome rewards provide only weak supervision for context editing, as successful trajectories can contain ineficient edits and failed trajectories useful ones. Simply rewarding edit frequency or removed context volume is also undesirable, as the LM may reward-hack by making unnecessary edits that discard important information or hurt prefix reuse. We therefore introduce a success-gated efficiency advantage that further rewards successful trajectories with lower prefix-reuse FLOPs. Let $c _ { i }$ denote the prefix-reuse FLOPs of trajectory $\tau _ { i } ,$ and let ${ \mathcal { G } } _ { g } ^ { + }$ denote the successful trajectories in group �. We define $\begin{array} { r } { \bar { c } _ { g } = \frac { 1 } { | \mathcal { G } _ { g } ^ { + } | } \sum _ { k \in \mathcal { G } _ { g } ^ { + } } c _ { k } } \end{array}$ and

$$
A _ { i } ^ { \mathrm { e f f } } = \left\{ \begin{array} { l l } { \mathrm { c l i p } \left( \frac { \bar { c } _ { g } - c _ { i } } { \bar { c } _ { g } } , - 1 , 1 \right) , } & { i \in \mathcal { G } _ { g } ^ { + } , } \\ { 0 , } & { i \notin \mathcal { G } _ { g } ^ { + } . } \end{array} \right.\tag{6}
$$

When there are fewer than two successful trajectories in a group, we set $A _ { i } ^ { \mathrm { e f f } } = 0$ for all trajectories. Thus, the eficiency signal only re-ranks among successful trajectories by inference cost. We combine outcome and eficiency advantages as $A _ { i } = A _ { i } ^ { \mathrm { o u t } } + \omega _ { \mathrm { e f f } } A _ { i } ^ { \mathrm { e f f } }$ to encourage trajectories that are both correct and eficient.

## 4.3 (More) Efficient CLM Serving with Suffix Cache Reuse

What if we want to serve CLMs even more eficiently and reduce the re-prefilling overhead? We introduce Suffix Cache Reuse (SCR). As shown in Figure 4, when � is replaced by �<sup>′</sup> after an edit, SCR reuses the cached states of all surviving tokens, including � , and only reprefills the newly inserted or appended tokens �<sup>′</sup> . Surviving sufix tokens � thus retain stale cache states that encode the previous prefix, which can even be beneficial in some cases, as it retains richer information from the past. By contrast, standard prefix-cache reuse must re-prefill all tokens after the first mismatch ( �<sup>′</sup> and � ). <sup>1</sup> Throughout the paper, we report prefix-reuse FLOPs under standard serving; additional SCR savings are reported separately in Section 5.3.

![](images/0e18aa38a8e22523f01898514d042f76da4e1bab8fa3320e2c35a993a01a8370.jpg)  
Figure 4 Comparison of standard serving and Suffix Cache Reuse (SCR). Standard serving reuses only prefix-matched cache, while SCR reuses cached states for all surviving tokens, reducing re-prefilling.

(a) BrowseComp-Plus (b) TerminalBench 2.1  
![](images/7d81f639661cfe9f658969d2aedaac5b2fceecc5d5475eb56a4f781f64118ddf.jpg)

![](images/e8abebb3411dbea29c8a25cf1d593c1d1ea11e2df2b1f0359c1755c76465459c.jpg)  
Prefix-reuse PFLOPs / question

(c) TBLite  
![](images/241a7c097e6be762bba701b3e039c72d1922d0e5867f2c05318a4b5b68bf9222.jpg)

Table 1 CLMs outperform specialized OpenEvolve evolutionary workflows on mathematical optimization problems. Best-of-run scores with Claude 4.6 Sonnet and a 32K context limit, capped at 100 scored attempts or five hours. Arrows indicate the direction of improvement; OE stands for OpenEvolve and SA for subagents.  
Figure 5 CLMs perform better than action-based and harness-defined baselines at lower cost on coding and deep research tasks. All methods use Qwen3.6-27B with a 32K context limit and a 100-turn cap. Blue dashed lines indicate the Pareto frontier.
<table><tr><td colspan="6">Circle Heilbronn Min-max/</td></tr><tr><td>Method</td><td>packing (↑)</td><td></td><td>(↑) min-dist (↑) overlap (↓)</td><td></td></tr><tr><td>OE</td><td>2.541</td><td>0.03127</td><td>0.07690</td><td>0.38123</td></tr><tr><td>OE-Agent</td><td>2.525</td><td>0.03053</td><td>0.07724</td><td>0.38167</td></tr><tr><td>CLM</td><td>2.618</td><td>0.03653</td><td>0.07758</td><td>0.38094</td></tr><tr><td>CLM (SA)</td><td>2.636</td><td>0.03617</td><td>0.07758</td><td>0.38109</td></tr></table>

## 5 Results

## 5.1 Evaluating CLMs Zero-Shot on Long-Horizon Agentic Tasks

We evaluate CLMs across long-horizon coding, deep research, and open discovery tasks, spanning tens to thousands of agent turns and runtimes from hours to a full day. Our evaluation covers both single- and multi-agent settings, including subagent and agent-swarm workloads for open discovery problems.

## 5.1.1 Coding and Deep Research Tasks

We first evaluate CLMs on two terminal-coding benchmarks, TerminalBench 2.1 (TB2.1) (Merrill et al., 2026) and TBLite (OpenThoughts-Agent team, 2026), and on the deep-research benchmark BrowseComp-Plus (BCP) (Chen et al., 2025).<sup>2</sup> We compare CLMs against MEM1 (Zhou et al., 2026), Self-Compact (Li et al., 2026b), ACM (Li et al., 2026c), and recursive language models (RLM) (Zhang et al., 2025a) with a shared Mini-SWE-Agent backbone (Merrill et al., 2026). To ensure a controlled comparison independent of training data, we evaluate all methods out of the box without training. We report performance and prefix-reuse FLOPs on these benchmarks for Qwen3.6-27B with a 32K context budget, with full details in Appendix E.

CLMs outperform harness-defined and action-based baselines. On BCP, CLMs outperform all baselines, scoring 59.4% at a 32K context limit and exceeding the strongest baseline, Codex-style summarization, by 11.4% relative. CLMs also use 21.5% and 28.9% fewer prefix-reuse FLOPs than the next two strongest methods, Codex-style summarization and MEM1, respectively. On coding benchmarks, CLMs match the strongest baseline, Codex-style summarization, on TB2.1 while using only 70% of its prefix-reuse FLOPs, and exceed it on TBLite (73.7% against 67.0%) with 91% of its FLOPs.

## 5.1.2 Open Discovery Problems

Open discovery problems provide longer horizons as our testbeds. We consider three types of open discovery problems with increasing horizons: (1) Mathematical optimization: four mathematical optimization problems used by AlphaEvolve (Novikov et al., 2025) and OpenEvolve (Sharma, 2025): circle packing, min-max/mindistance 2D, Erdős minimum overlap, and the Heilbronn triangle problem. (2) Single-repository optimization: ten EdgeBench (Zhu et al., 2026) tasks (EdgeBench-10; Appendix E), where the agent optimizes within a repository for up to 12 hours. (3) Multi-repository optimization with agent swarms: six repositories jointly optimized by multiple agents and evaluated on held-out downstream packages. Runs last over 24 hours.

Mathematical optimization: CLMs vs. specialized evolutionary workflows. On mathematical optimization, we compare against OpenEvolve (Sharma, 2025), a specialized AlphaEvolve-style (Novikov et al., 2025) workflow for program generation, evaluation, and evolutionary selection. We also include OpenEvolve-Agent, which

AcActive hourActive hours  
![](images/a4efd8907f708157aa9ef132ea119e7b7fcdaefc8a678e9dacb28521ba1d7e62.jpg)  
(a) EdgeBench-10 single-repository optimization.

![](images/c5a5aafbaea337ac02e010f62f043f38d010ed295cbfbbad78c7147e205c0313.jpg)

![](images/3a6cb9b9cee339743fd4ce7f0c547e8fc4b7aeb5c654df6ad08748256a676cff.jpg)  
(b) Software World multi-repository optimization.

Figure 6 CLMs outperform Codex-style summary harness on long-horizon repository optimization. (a) EdgeBench-10 with a 32K context budget. Curves show best-of-three scores over 12 hours for Qwen3.6-27B and Claude 4.6 Sonnet; end labels show final scores and, for Qwen3.6-27B, mean compute per trial (PF = prefix-reuse PFLOPs). (b) Software World with GPT-5.6-Sol and a 272K context budget. Six agents jointly optimize interdependent repositories and are evaluated on four unseen downstream packages; the right panel shows geometric-mean speedup over 17 evaluation tasks.

replaces its proposer with a Mini-SWE-Agent that can interact with the environment before each submission. For CLM, we use the same base harness with a minimal Bash interface and provide the evolutionary algorithm as in-context guidance, leaving planning and context management to the agent. Using Claude 4.6 Sonnet and the same evaluator, CLM achieves the highest best-of-run score on all four problems (Table 1; progress curves in Figure 18). This shows that a general agent with direct context control can outperform a specialized evolutionary workflow with less fixed orchestration.

Single-repository optimization under single-agent and subagent settings. On EdgeBench-10, agents optimize a repository for up to 12 hours with verifier feedback; we report the best score over three seeds per task. Figure 6 compares the base harness, Codex-style summarization, CLM, and CLMs with up to five concurrent subagents under a 32K context budget. With Qwen3.6-27B, CLM reaches 44.6 using 179 prefix-reuse PFLOPs per trial, versus 42.3 and 437 PFLOPs for summarization; the subagent variant reaches 44.2 at 181 PFLOPs. With Claude 4.6 Sonnet, CLM and its subagent variant reach 51.0 and 50.4, compared with 42.3 for summarization. We find that subagents provide little additional benefit on this single-repository benchmark.

Multi-repository optimization with agent swarms. We evaluate CLM on Software World, where six agents jointly optimize interdependent Python repositories and are evaluated on four unseen downstream packages (Figure 6b, left). This provides an extrinsic test of whether improvements transfer beyond the repositories the agents directly observe. Compared with a summary-based agent swarm at the same spend, CLM achieves 65% greater downstream speedup over the initial releases (Figure 6b, right). Full setup and scoring details are provided in Appendix E.

## 5.2 Learning Better Context-Management Strategies in Context or in Weights

CLMs make context management an intrinsic model behavior that can be learned like other skills. In this section, we present in-context learning and reinforcement learning results for CLMs.

Steering context management by simply talking to CLMs. Users can steer CLMs toward a desired contextmanagement strategy through natural-language instructions. We demonstrate this with three behaviors: triggering compaction at a specified context length, compacting around semantic sub-question boundaries, and backing up the context before compaction. Each behavior is induced by a single sentence appended to the task prompt. As shown in Figure 7, the agent adapts its context-management policy accordingly, without any change to the harness or model parameters. Measurement details are provided in Appendix E.

Evolving context management via textual evolution with CLMs. We apply the in-context evolution loop from Section 4.2 to ContextBench with a 32K context budget. In assisted evolution, Qwen3.6-27B starts without any context-management instruction, with Claude Fable 5.1 serving as the skill proposer; in self-evolution, Opus 5 serves as both the agent and the proposer. Figure 8 shows results on KV Store from ContextBench, where both settings improve over their initialization and expand the performance–cost Pareto frontier, with evolved skills that can strictly dominate the starting point. Additional results are in Appendix F.

![](images/b7059598a4f5230a41d40ed684c83d3077e05931e8e87f5d6d58810d1923af43.jpg)

![](images/12c0301f61e4c36b6aa48458526d9630e9012de98328f412c31359b4b4422ba7.jpg)  
Turns after a sub-question boundary

![](images/61d545b2a745d4343900811b1041d395235843571c068a4b12c35e3336b5b26e.jpg)

![](images/ab56aff365decf8ce77ba7da2e1cb4b04a12accbd24e1d02a6bfc5ef3d1d10b2.jpg)

![](images/9a69c9de30a91521e7e28f38e655460ae5c2ebb276ac06aea7a2d5f4fa84c0c6.jpg)  
Figure 7 One sentence in the prompt changes the context-management policy. Natural-language instructions steer compaction timing, semantic boundaries, and backup behavior. Gray denotes no instruction and blue the instructed setting; exact prompts are in Appendix E.  
Figure 8 Textual evolution for CLMs on KV Store (32Kbudget). Assisted evolution uses Qwen3.6- 27B with Claude Fable 5.1 as proposer; selfevolution uses Opus 5 for both roles.

Reinforcement learning for CLMs. We post-train Qwen3.5-9B on OpenResearcher using the reward formulation from Section 4.2 and evaluate on held-out BrowseComp-Plus. Before training, CLM with Qwen3.5-9B underperforms the summary harness by six points due to the smaller model’s limited context-management capabilities. After RL, it gains 13.7 points to 42.5%, matching the trained summary harness while using 1.34 versus 2.19 PFLOPs per question. Adding the eficiency reward further reduces inference cost without a clear loss in accuracy for either CLM or the summary harness. Full results and reward ablations are provided in Appendix F.

<table><tr><td>Method</td><td>Acc. (%) ↑</td><td>PFLOPs /Q↓</td></tr><tr><td>Summary</td><td>34.7 → 42.1</td><td>4.01 → 2.19</td></tr><tr><td>CLM</td><td>28.8 → 42.5</td><td>1.52 → 1.34</td></tr></table>

Table 2 RL results on BrowseComp-Plus with Qwen3.5-9B. Performance before and after training on OpenResearcher. The RL checkpoint is selected on a held-out validation set.

## 5.3 (More) Efficient Serving with Suffix Cache Reuse

As shown in Figure 9, SCR efectively reduces cache re-prefilling, matching the standard SGLang serving with 65.0% of its empirical prefix-reuse FLOPs on BCP. In addition, SCR is not limited to CLMs. Serving engines commonly strip prior reasoning tokens from chat histories, causing subsequent preserved tokens to be re-prefilled. We show that SCR can also reduce this re-prefilling cost in this more general setting.

We provide further details in Appendix B, including SCR implementation for hybrid models with interleaved full- and linear-attention layers, handling of multiple surviving post-edit spans, a decomposition of savings from reasoning-token stripping, and remaining opportunities for improvement in SGLang serving with SCR.

![](images/b5ceae5b8edfdd934abcb8540c0f8f3bf01f11716ec2cc4b398253cc5cda3e37.jpg)

![](images/9090e62b423ced7369e30bd98ce15b39692e1796698d6f9b1cbb5e1b23d6d320.jpg)

![](images/5c7f2742c6b794442ec016d7e69178744ba6ca2d9f0aa201426b2b8eb9ba5b26.jpg)  
Figure 9 Comparison of Suffix Cache Reuse and standard SGLang serving on BCP with Qwen3.6-27B. Left: Task accuracy and prefix-reuse FLOPs per question. Right: Server-side compute decomposition, showing the fraction of prompt tokens by compute type across all turns and turns following context edits.

## 6 Discussion and Future Work

Safety implications of a model-editable context. Granting models write access to their live context enables more flexible on-the-fly context management, but also creates new safety challenges. Editable context can become another channel through which prompt injections or self-generated instructions persist across turns. Recent work has observed such behavior in compaction summaries, including cases where a model inserted unauthorized instructions into its own summary that subsequently afected task behavior (OpenAI, 2026b). As editable context becomes more widely used, future work should characterize these new attack surfaces and develop defenses that preserve the flexibility of model-controlled context while maintaining its integrity.

Future directions: scaling CLM RL and distilling existing harnesses into CLMs. Future work can scale RL training so that CLMs can explore and learn efective context-management strategies, and develop a harness-to-CLM pipeline that distills strategies from existing harnesses into CLMs. This is motivated by the view that, while standard LMs only map input tokens to next-token distributions, harnesses determine how the context is constructed and updated. Since many harness operations can be expressed as context transformations, they can potentially be translated into CLM actions and eventually internalized into model weights. From this perspective, harnesses act as a form of procedural memory or task-specific skill that can be developed externally and later absorbed by CLMs for more general use.

## Acknowledgments

We thank Sewon Min and Steven Zĳian Chen for helpful discussions. We thank Ilia Kulikov and Mickel Liu for their help with infrastructure questions. This work was supported by the Singapore National Research Foundation and the National AI Group in the Singapore Ministry of Digital Development and Information under the AI Visiting Professorship Programme (award number AIVP-2024-001) and the AI2050 program at Schmidt Sciences.

## References

Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. In International Conference on Learning Representations, volume 2026, pages 8479–8565, 2026.

Daman Arora and Andrea Zanette. Training language models to reason eficiently. Advances in Neural Information Processing Systems, 38:60770–60808, 2026.

Federico Cassano and Sasha Rush. Training Composer for longer horizons. Cursor Research Blog, March 2026. https://cursor.com/blog/self-summarization. Published March 17, 2026; accessed July 29, 2026.

Aaron Chan, Ahmed Shalaby, Alexander Wettig, Aman Sanger, Andrew Zhai, Anurag Ajay, Ashvin Nair, Charlie Snell, Chen Lu, Chen Shen, et al. Composer 2 technical report. arXiv e-prints, pages arXiv–2603, 2026.

Zĳian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, et al. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. arXiv preprint arXiv:2508.06600, 2025.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. Prompt cache: Modular attention reuse for low-latency inference. In Proceedings ofMachine Learning and Systems, 2024.

Zhenyu He, Jun Zhang, Shengjie Luo, Jingjing Xu, Zhi Zhang, and Di He. Let the code llm edit itself when you edit the code. In International Conference on Learning Representations, volume 2025, pages 59637–59653, 2025.

Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Qin Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. EPIC: Eficient position-independent caching for serving large language models. In Proceedings of the 42nd International Conference on Machine Learning, pages 24391–24402, 2025.

Vasilis Kontonis, Yuchen Zeng, Shivam Garg, Lingjiao Chen, Hao Tang, Ziyan Wang, Ahmed Awadallah, Eric Horvitz, John Langford, and Dimitris Papailiopoulos. Memento: Teaching llms to manage their own context. arXiv preprint arXiv:2604.09852, 2026.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-harness: End-to-end optimization of model harnesses. arXiv preprint arXiv:2603.28052, 2026.

Mo Li, LH Xu, Qitai Tan, Long Ma, Hongyong Song, Ting Cao, and Yunxin Liu. Sculptor: Empowering llms with cognitive agency via active context management. In International Conference on Learning Representations, volume 2026, pages 153411–153440, 2026a.

Tianjian Li, Jingyu Zhang, William Jurayj, Xi Wang, Chuanyang Jin, Mehrdad Farajtabar, Eric Nalisnick, and Daniel Khashabi. Self-compacting language model agents. arXiv preprint arXiv:2606.23525, 2026b.

Xiaochuan Li, Ryan Ming, Meng Chu, Shuai Shao, Rong Jin, and Chenyan Xiong. Acm: Agentic context management for long horizon tasks. arXiv preprint arXiv:2607.23809, 2026c.

Yujiang Li, Zhenyu Hou, Yi Jing, Jie Tang, and Yuxiao Dong. CompactionRL: Reinforcement learning with context compaction for long-horizon agents. arXiv preprint arXiv:2607.05378, 2026d.

Shukai Liu, Bo Jiang, Jian Yang, Yizhi Li, Jinyang Guo, Xianglong Liu, and Bryan Dai. Context as a tool: Context management for long-horizon swe-agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 20604–20617, 2026.

Mike A Merrill, Alexander G Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey˜ Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

OpenAI. Codex CLI: A coding agent for the terminal. https://github.com/openai/codex, 2026a. Software, version 0.146.0, accessed July 29, 2026.

OpenAI. Self-generated prompt injections in compaction summaries, September 2026b. https://alignment.openai.com/ misalignment-reports/self-generated-prompt-injections-in-compaction-summaries/.

Bespoke Labs OpenThoughts-Agent team, Snorkel AI. OpenThoughts-TBLite: A High-Signal Benchmark for Iterating on Terminal Agents. https://www.openthoughts.ai/blog/openthoughts-tblite, February 2026.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Keqin Peng, Yuanxin Ouyang, Xuebo Liu, Zhiliang Tian, Ruĳian Han, Yancheng Yuan, and Liang Ding. Think dense, not long: Dynamic decoupled conditional advantage for eficient reasoning. arXiv preprint arXiv:2602.02099, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Asankhaya Sharma. Openevolve: an open-source evolutionary coding agent, 2025. https://github.com/ algorithmicsuperintelligence/openevolve.

Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, and Jiecao Chen. Scaling long-horizon llm agent via context-folding. arXiv preprint arXiv:2510.11967, 2025.

Richard S. Sutton. The bitter lesson. http://www.incompleteideas.net/IncIdeas/BitterLesson.html, 2019.

Shengguang Wu, Hao Zhu, Yuhui Zhang, Xiaohan Wang, and Serena Yeung-Levy. Automem: Automated learning of memory as a cognitive skill. arXiv preprint arXiv:2607.01224, 2026.

Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, et al. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313, 2025.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jef Z Pan, et al. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12805–12825, 2026.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. https://arxiv.org/abs/2405.15793.

Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. CacheBlend: Fast large language model serving for RAG with cached knowledge fusion. In Proceedings ofthe Twentieth European Conference on Computer Systems, pages 94–109, 2025.

Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, and Guojie Song. Meta context engineering via agentic skill evolution. arXiv preprint arXiv:2601.21557, 2026.

Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, et al. Agentfold: Long-horizon web agents with proactive context management. arXiv preprint arXiv:2510.24699, 2025.

Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, and Libing Wu. Agentic memory: Learning unified long-term and short-term memory management for large language model agents. arXiv preprint arXiv:2601.01885, 2026.

Alex L Zhang, Tim Kraska, and Omar Khattab. Recursive language models. arXiv preprint arXiv:2512.24601, 2025a.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jef Clune. Darwin godel machine: Open-ended evolution of self-improving agents. arXiv preprint arXiv:2505.22954, 2025b.

Jenny Zhang, Bingchen Zhao, Wannan Yang, Jakob Foerster, Jef Clune, Minqi Jiang, Sam Devlin, and Tatiana Shavrina. Hyperagents. arXiv preprint arXiv:2603.19461, 2026a.

Xuan Zhang, Longtao Zheng, Cunxiao Du, Bo An, and Xin Dong. Autocompact: Learning when to compact context in long-horizon coding agents, 2026b. https://autocompact.github.io/.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Eficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Zĳian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. MEM1: Learning to synergize memory and reasoning for eficient long-horizon agents. In International Conference on Learning Representations, volume 2026, pages 58413–58438, 2026.

Deyao Zhu, Xin Zhou, Shengling Qin, Xuekai Zhu, Hangliang Ding, Shu Zhong, Zixin Wen, Zhonglin Xie, Chenhui Gou, Linxuan Ren, et al. Edgebench: Unveiling scaling laws of learning from real-world environments. arXiv preprint arXiv:2607.05155, 2026.

## Appendix

## A Extended Related Work

Harness-scheduled context management. A common approach is to let the harness determine when and how the context is updated. Systems such as Cursor (Cassano and Rush, 2026), Codex (OpenAI, 2026a), and Terminus2 (Merrill et al., 2026) trigger compaction when the context reaches a predefined length, using a prescribed summarization procedure. MEM1 (Zhou et al., 2026) instead updates the context at every turn, combining information retained from previous turns with the new observation rather than carrying forward the full history. Reinforcement learning can improve model performance within these harness-scheduled procedures: Composer (Cassano and Rush, 2026; Chan et al., 2026), CompactionRL (Li et al., 2026d), and MEM1 train models to preserve useful information or continue reasoning efectively under context compaction. Although the resulting context depends on the model’s generation, the update schedule and procedure remain prescribed by the harness.

Model control within a constrained action space. Another line of work gives the model control over context management through constrained tools that implement predefined strategies, such as compaction, ofloading, retrieval, and branching. Self-Compact (Li et al., 2026b) and AutoCompact (Zhang et al., 2026b) let the model decide when to compact. Context-as-a-Tool (Liu et al., 2026) exposes model-triggered compaction within a structured context workspace, and ACM (Li et al., 2026c) adds ofloading and retrieval. AgeMem (Yu et al., 2026) combines long-term memory operations with tools for summarizing and filtering the current context. Context Folding (Sun et al., 2025) lets the model branch into a sub-trajectory and fold it into a summary upon returning, whereas AgentFold (Ye et al., 2025) condenses recent interactions or consolidates multiple historical steps through folding directives. Sculptor (Li et al., 2026a) supports fragment-level summarization, hiding, restoration, and search while preserving message count and order. Training can improve how models use these tools: AgentFold uses supervised fine-tuning, while AutoCompact, AgeMem, and Sculptor use reinforcement learning to optimize their respective context-management decisions. However, the available operations and their underlying strategies remain predefined by the tool interfaces. In contrast, CLMs treat context as a file, giving the model direct read and write access to its live context through general-purpose programming tools.

Model-controlled context through meta optimization. Meta-optimization gives models control over context management by improving the reusable procedures that govern agent execution. Meta-Harness (Lee et al., 2026) and AutoMem (Wu et al., 2026) optimize harnesses or memory-management procedures from trajectory feedback, while Meta Context Engineering (Ye et al., 2026) co-evolves context-engineering skills and context construction functions represented as files and code. Related approaches optimize the agent program itself (Zhang et al., 2025b, 2026a) or evolve prompts through reflection on rollouts (Agrawal et al., 2026). From the perspective of CLMs, a harness encodes reusable procedures for managing context. Such procedures can also be expressed as skills that guide the model in editing its live context. Our skill evolution therefore shares the goal of harness optimization: improving reusable context-management procedures through evaluation feedback.

Context as a variable in the environment. Recursive language models (RLMs) (Zhang et al., 2025a) place a long input in a REPL variable that the model can access programmatically and process through recursive calls. This gives the model control over how it reads the input, but does not expose its own live context for direct editing. The distinction is twofold: RLMs externalize the input rather than the evolving interaction history, and the model’s access to its live context remains read-only rather than read–write. CLMs instead make the live context itself editable, including information accumulated during execution. The two approaches are complementary: RLM-style access can keep large inputs outside the context until needed, while CLMs can manage the information brought into the context and the history generated while processing it.

Non-prefix KV cache reuse. With standard prefix caching, changing an early part of a prompt forces the serving system to recompute the KV states of everything that follows, even when the later text is unchanged. Prior work relaxes this requirement in diferent settings. Prompt Cache (Gim et al., 2024) precomputes attention states for predefined prompt modules, allowing a module to be reused in prompts that do not share the same preceding text. In retrieval-augmented generation, the same document may appear after diferent documents or instructions. CacheBlend (Yao et al., 2025) and EPIC (Hu et al., 2025) reuse cached document chunks in these new contexts, recomputing selected tokens to account for the changed surroundings. PIE (He et al., 2025) studies cache reuse when a user modifies previously processed code and requests a new completion. It retains cached states for unchanged text after an edit and corrects their rotary positions, avoiding sufix recomputation. Memento (Kontonis et al., 2026) evicts each completed reasoning block from the KV cache but keeps the cached states of its summary, which were computed while the block was still in context, and finds that these states retain useful information from the evicted block. Sufix Cache Reuse applies the same reuse principle to an agent’s live context: when the agent replaces a span, the unchanged sufix retains its cached states rather than being prefilled again. We integrate this mechanism into SGLang for agent-driven context editing and further extend it to hybrid architectures that combine full-attention layers with linear-attention layers.

Reinforcement learning for context management. Context edits break the append-only structure of an agent trajectory: the final context may no longer contain the inputs under which earlier actions were generated. Training must therefore preserve these intermediate contexts and evaluate each generated segment under its original input. ReSum (Wu et al., 2025) segments trajectories at summarization boundaries and broadcasts the trajectory-level advantage to all segments. Sculptor (Li et al., 2026a) similarly preserves training contexts around context modifications and masks previously trained completions to avoid counting them repeatedly. Our stepwise GRPO follows this principle, assigning the final outcome advantage to each segment’s policygradient loss rather than diferentiating through the context edits. Prior work also considers eficiency. AgeMem (Yu et al., 2026) includes a reward for context compactness, while Sculptor penalizes exceeding tool-call or trajectory-length budgets. Our eficiency signal instead measures trajectory-level inference FLOPs under prefix caching, accounting for the recomputation that context edits can incur. We add a success-gated eficiency advantage that favors lower-cost trajectories only within the successful subset of a rollout group. This distinguishes reducing inference compute from merely shortening the context and avoids giving failed trajectories an eficiency bonus.

Success-conditioned efficiency objectives. Prior work has also conditioned eficiency rewards on task success. Arora and Zanette (2026) penalize response length only for correct answers, and DDCA (Peng et al., 2026) computes a separate length advantage within the correct-response subset. These objectives primarily measure eficiency through response or trajectory length. Our formulation instead uses trajectory-level FLOPs under prefix caching, capturing the recomputation induced by context edits in addition to generated length.

Context management and external memory. Context management determines what the model sees at each invocation, whereas external memory stores information beyond the current context for later use. Memory-R1 (Yan et al., 2026), for example, learns to manage stored memories and use retrieved information. These mechanisms work together: an agent can ofload information to reduce its context and retrieve it when needed. MemGPT (Packer et al., 2023) connects them through a memory hierarchy, allowing the model to edit a designated, fixed-size block within the context rather than the entire live context. AgeMem (Yu et al., 2026) jointly learns external-memory operations and tools that summarize or filter the current context. This distinction depends on the role of the information, not its storage format. In CLMs, the context file specifies the input to subsequent model calls. Other files can serve as external memory, with their contents entering the context when retrieved. Making the live context editable therefore complements external memory by letting the model decide how retrieved information is incorporated and when it is removed.

## B Suffix Cache Reuse

Background: radix tree and prefix cache reuse in SGLang. SGLang (Zheng et al., 2024) keeps the KV cache of served requests in a radix tree over token sequences: each edge holds a token span and the KV entries computed for it, so requests with a common prefix share one path. A new request walks the tree to its longest matching prefix, reuses the KV entries along that path, and prefills only the remaining tokens; unused nodes are evicted in least-recently-used order. Reuse therefore stops at the first mismatched token. This is exact for append-only histories, since each token’s key and value depend on all preceding tokens and, through rotary position encodings, on its absolute position. After an in-the-middle edit, however, every token after the edit is re-prefilled, including text that survived unchanged. For hybrid models, whose linear-attention layers keep a fixed-size recurrent state instead of per-token entries, a match can resume only at a node that stores a state checkpoint. Sufix Cache Reuse keeps this tree as is and extends reuse to the surviving tokens beyond the matched prefix, as described next.

![](images/dbd5c3a42314c73d0f58f42b392cc6ad7c3dfbbcab35a47276686b93427ec7ac.jpg)  
Figure 10 Suffix Cache Reuse for full attention layers and linear attention layers.

Suffix Cache Reuse implementation for standard full attention layers. We implement Sufix Cache Reuse as a patch to SGLang. As shown in Figure 10 (left), consider a context [� � �] in which an edit replaces � with �<sup>′</sup>. Standard serving matches only � and re-prefills �<sup>′</sup> and �. When a new prompt arrives, Sufix Cache Reuse difs it against the session’s previous prompt to find the spans that survived the edit, and relocates up to � of them, largest first (�=6 in the main text). For each relocated span such as �, it reuses the cached keys and values, re-rotates the keys’ rotary position encodings to their new positions, and splices them in after � . Only �<sup>′</sup> and newly appended tokens are prefilled. Relocated entries live in session-private cache slots, so the shared radix tree never holds a moved entry; if these slots cannot be allocated, the server falls back to standard re-prefilling. Because the reused states were computed under the pre-edit context, Sufix Cache Reuse approximates re-prefilling, and � bounds the number of relocated spans per edit.

Suffix Cache Reuse implementation for linear-attention layers. In hybrid models such as Qwen3.6-27B<sup>3</sup>, fullattention and linear-attention layers may be interleaved. As shown in Figure 10 (right), linear-attention layers maintain a fixed-size recurrent state rather than per-token caches, so there are no token-level entries to relocate. For these layers, we snapshot the recurrent state before the edit and continue from that snapshot, while the edit is reflected only in the 16 full-attention layers that retain token-level context. As a result, the newly inserted �<sup>′</sup> is not recomputed in the linear-attention layers, since subsequent tokens depend only on the reused recurrent state. Its representation is still recomputed in the full-attention layers and can influence later linear-attention layers through their inputs.

Suffix Cache Reuse for multiple surviving post-edit spans. A single edit may leave multiple surviving spans after the edit point. SCR can relocate each span, but every relocation reuses states computed under the pre-edit context and therefore introduces an approximation. When many spans are relocated in the same edit, these approximations can compound before the model has a chance to adapt in subsequent turns. We therefore cap the number of relocated spans per edit at �, reusing the � longest spans and re-prefilling the rest. This bounds the amount of approximation introduced at once and makes SCR less susceptible to pathological edits with many surviving spans.

We conduct a small-scale sensitivity analysis over � ∈ {1 2 3 6 12 64} on 64 BrowseComp-Plus questions with Qwen3.6-27B (Figure 11). Performance is robust across �, while cache-reuse gains largely saturate by � = 6. We therefore use � = 6 throughout as a conservative choice that captures most reusable cache while limiting the approximation introduced by any single edit. In this small-scale analysis, varying � mainly afects cache-reuse eficiency, with no observed performance degradation.

![](images/9c1536ae64aaba8798871ccfe5267926e0ab522c2bacb83d69fb9bf066e4ac0a.jpg)  
Figure 11 Small-scale sensitivity study of relocated spans per edit, �. Results on 64 BrowseComp-Plus questions with Qwen3.6-27B. Left: task accuracy. Middle: prefix-reuse PFLOPs per question. Right: relocated tokens per edited request. Dashed lines show standard SGLang; hollow markers show a repeat run; the shaded column marks �=6, used elsewhere. Error bars show ±1 standard error across questions; the gray band shows ±1 standard error for standard SGLang.

Bonus: Suffix Cache Reuse for stripped reasoning tokens in chat endpoints. Chat templates for reasoning models, including Qwen3.6, often remove the reasoning block from earlier assistant turns once the next user message arrives. Standard serving then re-prefills all preserved text after the first removed block, even when the agent never edits its own context. SCR treats reasoning stripping as another context edit and reuses the cached states of the preserved text, extending its benefit to standard chat serving.

Figure 12 shows that this efect accounts for a significant portion of SCR’s savings on BrowseComp-Plus. Of the 7.8% of all prompt tokens reused by SCR beyond prefix-cache hits, 5.3 points come from reasoning stripping and only 2.5 from other context edits. Thus, a large fraction of SCR’s benefit applies even to standard reasoning-model serving without model-driven context editing.

![](images/6ae1822cb36149a1faaac7c2850e7ee8f17ebcfa846289755cebd10332e6037f.jpg)  
Figure 12 Suffix Cache Reuse also benefits standard chat serving. On BrowseComp-Plus, SCR reuses cached states not only after model-driven context edits but also after reasoning tokens are stripped from prior turns in standard chat endpoints. Results use Qwen3.6-27B on 830 questions with � = 6.

A common serving bottleneck and future improvement space. Figure 13 shows that SCR removes most reprefilling of unchanged sufix tokens after an edit. Much of the remaining redundant prefill instead comes from unchanged prefixes that standard prefix caching should ideally reuse. This is a limitation of the current SGLang caching implementation for hybrid models, rather than SCR itself. Linear-attention layers maintain recurrent states, which SGLang stores only at cached request boundaries; when a later prompt diverges inside a cached span, no recurrent state is available near the branch point, so cache matching can fall back to a much shorter prefix. This afects both standard prefix caching and the eficiency attainable with SCR. Storing recurrent states at finer-grained locations, such as message boundaries, is therefore a promising direction for improving both.

![](images/a9d1c7b90c24c9bc58460fed95440055fba1d38be05c8c5ea69d6ca789a8358a.jpg)  
Figure 13 Remaining re-prefill under Suffix Cache Reuse on BrowseComp-Plus. SCR removes most re-prefilling of unchanged sufix tokens, while substantial unchanged-prefix prefill remains due to the current hybrid-model caching behavior in SGLang. Same runs as Figure 12.

## C Prefix-Reuse FLOPs Computation

Prefix-reuse FLOPs measure the computation performed by a server with prefix caching over an agent trajectory. At turn � the prompt contains $P _ { t }$ tokens and the model generates $G _ { t }$ tokens. The server reuses the longest prefix of leading messages that also appeared in the prompt of an earlier turn, comprising $R _ { t }$ tokens, and prefills only the remaining $U _ { t } = P _ { t } - R$ � tokens. Under standard prefix caching, an edit therefore invalidates the cached computation from the edited message onward, and a response is prefilled again when it first appears in a later prompt, in addition to being decoded when it is produced.

Consider a model with � layers of hidden size � and MLP width $d _ { \mathrm { H } } ,$ of which $L _ { \mathrm { a t t n } }$ are full-attention layers with $h _ { q }$ query heads (with an output gate), $h _ { k v }$ key–value heads and head dimension $d _ { h } ,$ , and $L _ { \mathrm { l i n } }$ are Gated DeltaNet layers with $h _ { k }$ key heads, $h _ { v }$ value heads, head dimensions $d _ { k }$ and $d _ { v } ,$ , and two scalar gates per value head. Linear operations have the same cost per processed token regardless of context length. Counting two FLOPs per multiply-add, this per-token cost is

$$
\begin{array} { r } { C _ { \mathrm { t o k e n } } = \underbrace { 6 L d ~ d _ { \mathrm { f f } } } _ { \mathrm { M L P } } + \underbrace { L _ { \mathrm { a t t h } } \left[ 2 d ( 2 h _ { q } d _ { h } + 2 h _ { k v } d _ { h } ) + 2 h _ { q } d _ { h } d \right] } _ { \mathrm { f u l l - a t t e n t i o n } \mathrm { p r o j e c t i o n s } } + \underbrace { L _ { \mathrm { l i n } } \left[ 2 d ( 2 h _ { k } d _ { k } + 2 h _ { v } d _ { v } + 2 h _ { v } ) + 2 h _ { v } d _ { v } d \right] } _ { \mathrm { G a t e d ~ D e l a N e t ~ p r o j e c t i o n s } } . } \end{array}\tag{7}
$$

Only the full-attention layers incur a context-dependent cost. Each query–key pair costs one multiply-add per head dimension for the attention score and one for the weighted value, so the cost per pair is

$$
C _ { \mathrm { a t t n } } = 4 L _ { \mathrm { a t t n } } h _ { q } d _ { h } .\tag{8}
$$

Ignoring lower-order boundary terms, a turn costs

$$
\begin{array} { r } { F _ { t } = C _ { \mathrm { t o k e n } } \left( U _ { t } + G _ { t } \right) + C _ { \mathrm { a t t n } } \left[ \frac { 1 } { 2 } \left( P _ { t } ^ { 2 } - R _ { t } ^ { 2 } \right) + G _ { t } P _ { t } + \frac { 1 } { 2 } G _ { t } ^ { 2 } \right] , } \end{array}\tag{9}
$$

where the first term inside the brackets counts attention during prefill and the latter two count attention during decoding. The cost of a trajectory of � turns is $\textstyle \sum _ { t = 1 } ^ { T } F _ { t }$ . We omit the embedding and output layers, the Gated DeltaNet recurrent-state update, and normalization, activation and softmax operations.

Example: Qwen3.6-27B. Qwen3.6-27B has $L = 6 4$ layers with $d = 5 1 2 0$ and $d _ { \mathrm { f f } } = 1 7 , 4 0 8 .$ . Its $L _ { \mathrm { a t t n } } = 1 6$ full-attention layers have $h _ { q } = 2 4 , h _ { k v } = 4$ and $d _ { h } = 2 5 6 _ { \AA }$ and its $L _ { \mathrm { l i n } } = 4 8$ Gated DeltaNet layers have $h _ { k } = 1 6 ,$ $h _ { v } = 4 8$ and $d _ { k } = d _ { v } = 1 2 8$ . Substituting these values gives $C _ { \mathrm { t o k e n } } = 4 8 . 7 0 \times 1 0 ^ { 9 }$ FLOPs per token, of which the

MLP accounts for $3 4 . 2 3 \times 1 0 ^ { 9 } .$ , the full-attention projections for $3 . 3 6 \times 1 0 ^ { 9 }$ and the Gated DeltaNet projections for $1 1 . 1 2 \times 1 0 ^ { 9 }$ , and $C _ { \mathrm { a t t n } } = 3 . 9 3 \times 1 0 ^ { 5 }$ FLOPs per query–key pair.

Figure 14 illustrates the computation saved by prefix caching within one turn. The lengths are chosen for illustration and do not come from a particular run: a prompt of $\bar { P _ { t } } = 2 0 { , } 0 0 0$ tokens and a response of $G _ { t } = 5 0 0$ tokens, with the reusable prefix $R _ { t }$ set to 18,000, 10,000 or 0 tokens. When the turn only appends to its context, prefix caching avoids 87% of the computation the turn would need without a cache, and the turn costs $1 . 4 \bar { 1 } \times 1 0 ^ { 1 4 } \mathrm { F L O P s }$ . An edit in the middle of the context leaves 10,000 reusable tokens and raises the cost to $5 . 7 4 \times 1 0 ^ { 1 4 }$ FLOPs. An edit at the start, equivalent here to having no reusable prefix, costs $1 0 . 8 1 \times 1 0 ^ { 1 4 }$ FLOPs, 7.7 times the append-only turn.

![](images/c561ca1c5ea22b1a1f6d26e42cf59ee695890c9bff06a38afa7d1e721724e3c1.jpg)  
Figure 14 FLOPs of one Qwen3.6-27B turn for illustrative lengths $( P _ { t } = 2 0 , 0 0 0 , G _ { t } = 5 0 0 )$ and three reusable-prefix lengths $R _ { t } .$ . Colors split the FLOPs actually computed by operation; gray shows the additional computation that would be required without prefix caching

## D ContextBench

Design and examples of ContextBench tasks. Each ContextBench task grades what the agent kept, updated, or ofloaded from an input that, at all but the lowest levels, exceeds the context window. The environment delivers the input as a sequence of operations, and each operation arrives as a new user message in one continuous conversation. An operation therefore enters the agent’s context before the agent can act on it, and no harness can truncate or ofload it on the way in. The agent controls only when the next operation arrives: it runs echo READY\_FOR\_NEXT\_OP once it has handled the current one. The tasks need no search or inference beyond following the instruction, so an agent that could keep everything would answer every query; performance measures how the agent manages its context and nothing else. Every task instance is generated from a seed and checked before use. Of the 32,768-token context limit, 2,048 tokens are reserved for the model’s response, leaving a usable budget of 30,720 tokens; a single operation must fit in a fifth of it and everything the task requires the agent to retain in half of it, each with a 10% margin, so a failure at any level reflects how the context was managed rather than a task that cannot be solved within the budget. Table 3 summarizes the four tasks and Figure 16 shows operations from each.

Context pressure is the total input the environment pushes over an episode divided by the 32,768-token context limit, so 1× is the point where an agent that keeps everything would fill its context. Each task reaches higher pressure through one generator setting, with everything else fixed (Figure 15); levels below 1× are included as controls on which keeping everything still fits.

Detailed instructions for isolated context-management diagnostics. In our pilot study, we provide each harness with a detailed task instruction and a method-specific skill that explains how to use its available tools for context management on that task. This helps isolate context-management capability from diferences in task understanding or tool use. For each task, every method receives the same task instruction, followed by a skill tailored to its context-management mechanism. The task instruction specifies the input stream, turn protocol, answer format, and grading procedure, without giving context-management advice. The method skill explains how to manage context using the tools exposed by that harness, including the relevant commands or tool calls, when they take efect, how to verify them, and common failure modes. We write each skill to a level of detail comparable to the CLM skill, so that baseline performance reflects what the harness enables rather than what the model must infer about how to use it. Table 4 shows excerpts of each method’s KV Store skill. Skills are inserted at the same position relative to the task instruction for all methods, and all are released with the benchmark.

<table><tr><td>Task</td><td>Input stream</td><td>Queries</td><td>Metric</td></tr><tr><td>Needle Retention</td><td>~4K-token chunks, each with 2-8 needle lines and 140 filler lines</td><td>none</td><td>needle lines retained verbatim in the final context</td></tr><tr><td>Sudoku Sketchpad</td><td>one move per turn on a 16×16 board current board after each</td><td>move</td><td>board versions reproduced exactly</td></tr><tr><td>KV Store</td><td>batches of 100 SET operations with 24 GET queries random 24-word values</td><td></td><td>exact-value accuracy</td></tr><tr><td>Log Triage</td><td>batches of 14–54 service log lines</td><td>24 lookup and count queries</td><td>exact-answer accuracy</td></tr></table>

Table 3 The four ContextBench tasks. All metrics are computed from the agent’s context; answers held only in files are not credited.

![](images/8a5b2cd4a3826e45a00bbce54c098ee5ea70c0f5b05fee26b9f4a171fc4f82b7.jpg)  
Figure 15 Context pressure of every ContextBench level. Each row varies one generator setting (unit at right); the label on each point is its value. Total input is every message the environment pushes over the episode, including the task instruction and, for Sudoku, the board once per move, counted in o200k tokens; pressure is total input divided by 32,768.

## E Experimental Configurations

We use the original baseline implementations when available and run all methods with the model and budget specified for each experiment: the summary harness uses the Codex summarization prompts and compacts at 75% of the budget; Self-Compact uses the original prompts and self-check rubric and asks the model every two turns whether to compress once the context exceeds 37% of the budget; RLM runs its released harness; ACM runs its released code, in which the model decides when to manage its context. MEM1 is our re-implementation of its inference loop. CLM receives its editing reminder 2,048 tokens before the budget<sup>4</sup>, and the base harness has no trigger. Unless stated otherwise, token budgets are counted with the o200k tokenizer and all methods are evaluated with the same base LM.

ContextBench. Every method uses GPT-5.4 through the API with a 32,768-token context limit, of which 2,048 tokens are reserved for the response, and receives the skill for its harness (Appendix D). Each level is run with four seeds, or eight for selected levels to reduce variance. Results are in Figure 2.

![](images/37a08e90766cb6ea17a1f54b5d84f6bb6abe68637a127685aca4e6d9b705116c.jpg)  
Figure 16 Operations as the agent receives them, from the smallest level of each task, shortened where marked. Blue text is the output the agent is expected to produce in response; it is graded from the agent’s context. Needle Retention asks for no output: its needles are graded in the final context.

<table><tr><td>Method</td><td>Excerpt of the KV Store skill</td></tr><tr><td>CLM</td><td>Mechanism: Your context is mirrored to a file you can edit; changing that file changes what you are holding. Each batch: On the SAME turn you read a SET-BATCH, move the whole block out of your context and onto disk with this one command [a python3 heredoc that moves the block to /tmp/ctx_offload/ and leaves one placeholder line].</td></tr><tr><td></td><td>Each GET: Recover the value with grep -h &#x27;SET &lt;key&gt; =&#x27; /tmp/ctx_offload/* and then print the ANSWER block. Check: No note, or a token count that did not drop, means the batch is still in your context.</td></tr><tr><td>ACM</td><td>Mechanism: manage_context takes everything since your previous manage_context call ... and replaces it with one message [summary_id: N] &lt;summary&gt;. Each batch: Read the gauge after every batch; when it is above 22,000, call manage_context on your very next turn, before you release another batch. Each GÉT: Spend one turn on query_memory(N, &quot;the SET line for key K00084, verbatim&quot;) with N the summary</td></tr><tr><td></td><td>whose range covers the key. Check: What confirms a compression is the [summary_id: N] message and a [context: ~N/M tokens] readout lower than the turn before.</td></tr><tr><td>RLM</td><td>Mechanism: The REPL namespace persists: every variable and function you define survives from one operation to the next. That persistence is your store.</td></tr><tr><td></td><td>Each batch: On the first operation define the store and the handler once; on every later operation only call them [a dictionary filled by a regular expression over the SET lines].</td></tr><tr><td></td><td>Each GET: A GET is answered by placing the ANSWER block in answer[&quot;content &quot;]. Check: print(out) shows stored N keys so far growing by one batch per SET operation.</td></tr><tr><td></td><td></td></tr><tr><td>Context Folding</td><td>Mechanism: A branch begins from a copy of a context that already holds every batch and cannot delete anything from MAIN, so MAIN is as full after it as before.</td></tr><tr><td></td><td>Each batch: On a SET-BATCH, run one command in MAIN and nothing else: echo READY_FOR_NEXT_OP.</td></tr><tr><td></td><td>Do not open a branch.</td></tr><tr><td></td><td>Each GET: Answer in MAIN ... by finding the SET &lt;key&gt; = line in the batch in front of you and printing its value. Check: The [context: ~N/M tokens] readout should rise by the size of each batch and by almost nothing else.</td></tr><tr><td></td><td></td></tr><tr><td>Self-Compact</td><td>Mechanism: Compression happens only when  $\mathrm { C 1 } { = } \mathrm { Y } , \mathrm { C } 2 { = } \mathrm { Y } , \mathrm { C } 3 { = } \mathrm { Y } , \mathrm { N } 1 { = } \mathrm { N } .$  Then your whole history is replaced by . . .</td></tr><tr><td></td><td>your summary. Each batch: On the SAME turn a SET-BATCH arrives, copy the whole block .. . to its own file. Answer [the probe]</td></tr><tr><td></td><td>so that compression fires as soon as everything you have seen is on disk.</td></tr><tr><td></td><td>Each GET: grep -h &quot;^SET K00084 = &quot; /tmp/store/*.txt, then print the value in an ANSWER block. Check: What confirms a store is the count grep -c prints back, equal to 100.</td></tr><tr><td></td><td></td></tr><tr><td>Summary</td><td>Mechanism: At three quarters of the budget (about 24.6K of 32,768 tokens) . . . everything except the system prompt and this task message is then replaced by one message.</td></tr><tr><td></td><td>Each batch: On the SAME turn a SET-BATCH arrives, copy the whole block .. . to its own file. Write the summary</td></tr><tr><td></td><td>as a pointer, not an inventory. Each GET: grep -h &quot;^SET K00084 = &quot; /tmp/store/*.txt, then print the value in an ANSWER block.</td></tr><tr><td></td><td>Check: What confirms the store is the grep -c count coming back as 100.</td></tr><tr><td></td><td></td></tr><tr><td>Base</td><td>Mechanism: Every operation arrives as a message in this conversation and stays in your context for the rest of the run;</td></tr><tr><td></td><td>there is no command, tool or edit that takes it back out.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Each batch: On a SET-BATCH, run one command and nothing else. Do not copy batches to disk.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Each GET: Find the SET &lt;key&gt; = line in the batch that is sitting in your context and print its value verbatim.</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>Check: The [context: ~N/M tokens] readout should rise by the size of each batch and by almost nothing else.</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

Table 4 Excerpts of each method’s KV Store skill. Every skill gives the same four kinds of guidance for its own mechanism: how the mechanism works, what to do with each batch, how to answer a query, and how to confirm that a step worked. Blue marks the concrete instruction for using the method’s own tools; . . . marks omitted text and square brackets paraphrase code.

TerminalBench 2.1. We use the 89 tasks of TerminalBench 2.1, each scored as pass or fail by its verifier. Qwen3.6-27B and Qwen3.5-9B are served with vLLM with thinking enabled and up to 4,096 generated tokens per call, with a 32,000-token budget; a request that would exceed it ends the run. Every method is limited to 64 turns, each shell command to 180 seconds and each task to four times its default time limit. Whenever a run ends, whether the agent finishes, reaches the turn or time limit, or exceeds the budget, the verifier scores the final state of the container. Results are in Figures 5 and 17.

TBLite. We use the OpenThoughts-TBLite tasks. The model is served with vLLM with up to 2,048 generated tokens per call and a 32,000-token budget. Turn and time limits and scoring are as for TerminalBench 2.1. Accuracy is the mean verifier reward. Results are in Figures 5 and 17.

BrowseComp-Plus. We use all 830 questions over the fixed BrowseComp-Plus corpus. The search tool returns the top 10 snippets of 512 tokens and the document reader is capped at 8,192 tokens. Models are served with vLLM with thinking enabled, temperature 0.7, top-� 0.95 and up to 4,096 generated tokens per call, with a 23,560-token budget and 100 turns; CLM’s editing turns do not count toward the turn limit. When a request would exceed the budget, CLM rolls back the last turn and retries, up to six times. An answer is correct only if Qwen3.5-27B, judging at temperature 0 with the BrowseComp-Plus grading template, marks it correct and complete; unanswered questions count as wrong. Results are in Figures 5 and 17.

Math optimization problems. We use circle packing (26 circles in the unit square), Erdős’ minimum overlap, the min–max distance ratio in the plane, and the Heilbronn triangle problem, with one run per method. The agent writes a program that an evaluator runs and scores. Every method uses Claude 4.6 Sonnet with a 32,000-token budget and stops after 100 evaluated attempts or five hours. OpenEvolve runs with a population of 60 in four islands; OpenEvolve-Agent uses a Mini-SWE-Agent proposer with 25 turns per candidate; CLM with subagents allows up to five subagents of 40 turns each. Results are in Table 1 and Figure 18.

EdgeBench-10. We use 10 of the 48 runnable public EdgeBench tasks (Table 5) with three seeds each. A run lasts 12 hours in a container without network access; each submission is graded by the EdgeBench judge on a 0–100 scale, and a run’s score is the higher of its best graded submission and the grade of the final repository. Qwen3.6-27B is served with vLLM at temperature 0.7 with up to 8,192 generated tokens per call, and Claude 4.6 Sonnet runs through the API. The budget is 32,000 tokens; when a request would exceed it, the harness rolls back the last turn, up to 50 times, and no turn limit applies. CLM with subagents runs up to six concurrent subagents of 40 turns each. The 128K results in Figure 19 use a 128,000-token budget. Cost is prefix-reuse FLOPs per run, including the subagents’ computation. Results are in Figures 6(a) and 19.

<table><tr><td>Task</td><td>Category</td><td>Language</td><td>What the verifier scores</td><td>Start: files / LOC</td></tr><tr><td>ad_placement_optimization</td><td>Combinatorial optimization</td><td>C++</td><td>solution score over judge cases</td><td>1 / 20</td></tr><tr><td>apple_incremental_game</td><td>Combinatorial optimization</td><td>Python</td><td>solution score over judge cases</td><td>3 / 101</td></tr><tr><td>graph_node_classification</td><td>Science &amp; ML</td><td>Python</td><td>held-out accuracy (CPU-only judge)</td><td>2 / 418</td></tr><tr><td>grid_turing_robot</td><td>Combinatorial optimization</td><td>Python</td><td>solution score (lower is better)</td><td>3 / 440</td></tr><tr><td>juliet_vulnerability_analyzer</td><td>Software engineering</td><td>Python</td><td>hidden evaluator on the Juliet suite</td><td>1/9</td></tr><tr><td>openrct2_theme_park_ai</td><td>Games &amp; simulators</td><td>JavaScript plugin</td><td>park value in a headless OpenRCT2 run</td><td>6 / 5,950</td></tr><tr><td>schemathesis_datagen_pipeline</td><td>Software engineering</td><td>Python</td><td>fraction of the test suite pāssing</td><td>416 / 112,663</td></tr><tr><td>triangulation_coloring_optimization</td><td>Combinatorial optimization</td><td>Python</td><td>coloring &quot;ugliness&quot; (lower is better)</td><td>5 / 431</td></tr><tr><td>vehicle_routing_time_windows</td><td>Combinatorial optimization</td><td>C/C++</td><td>CVRPTW solution quality</td><td>3 / 513</td></tr><tr><td>wesnoth_tactical_ai</td><td>Games &amp; simulators</td><td>Python</td><td>headless Wesnoth matches</td><td>0/0</td></tr></table>

Table 5 EdgeBench-10 tasks. Scores are rescaled to 0–100 by the EdgeBench judge. Start: files and lines of code in the working directory before any edit.

Software World. Six agents, one per repository (requests, urllib3 and four downstream packages), work in parallel for over 24 hours to make their repositories faster. The score is the geometric-mean speedup, in executed instructions, on 17 held-out CPU benchmarks from four downstream packages the agents never see; a benchmark that breaks counts as 1.0. Every agent runs in the Pi agent harness with Pi’s default model and context limit, GPT-5.6-Sol with a 272K-token context. Both swarms, with CLM and with summary compaction, use long wakes between work sessions. Cost is cumulative API spend in USD. Results are in Figure 6(b).

In-context steering. All three experiments use BrowseComp-Plus questions with Claude 4.6 Sonnet through the API, the unmodified CLM harness, no budget reminders, and a run that ends when a request exceeds the budget. Each behavior is induced by an instruction appended after the task message and compared with a no-instruction control on the same questions; confidence intervals are BCa bootstrap intervals over paired diferences. Threshold: a 48,000-token budget and 200 turns on 30 long questions (at or above the 60th percentile of turn count without instruction), with the instruction to compact once the context passes 16K, 24K or 32K tokens; we report the median context size at the first compaction. Boundaries: sessions of four chained questions with a 24,000-token budget and 1,000 turns, 189 sessions per condition, 131 of them with a boundary inside the session; we report the rate of compaction within two turns of a question boundary. Backup: a 16,000-token budget on 200 questions, 91 of them paired for analysis; we report the fraction of edits preceded by a full copy of the context. Results are in Figure 7. The appended instructions are, for the threshold (with � ∈ {16<sub>,</sub>000<sub>,</sub> 24<sub>,</sub>000<sub>,</sub> 32<sub>,</sub>000}):

Monitor your context size: every tool result reports your current size. Whenever it exceeds � tokens, immediately compact your context down to about 4000 tokens. For example: rewrite /tmp/.live\_ctx/LIVE\_CTX\_MAIN.txt to keep only your key findings and current plan (about 4000 tokens).

for boundaries:

Please only compress at the boundaries of subtasks. Once a subtask is accomplished, throw out all the context for that subtask. For example: after you write /workspace/answer\_1.txt, rewrite /tmp/.live\_ctx/LIVE\_CTX\_MAIN.txt to drop the Question 1 material, keeping only your answer\_1 conclusion.

and for backup:

Before every edit to your context, you must back up the current context file into a new file inside a new folder named compaction\_backup; never overwrite or delete earlier backups.

In-context evolution. The loop runs on ContextBench. Each proposed skill is first run on six training instances; if its accuracy there is at least that of the current frontier, it is run on twelve more, and then on a development split of 102 instances per task with one seed, the only split used to decide frontier membership and selection. For assisted evolution, a test split of 102 instances with three seeds is evaluated once after the archive is frozen. The agent is Qwen3.6-27B served with vLLM, with a 32,000-token budget, a 4,096-token reserve and 240 turns; a run that exceeds the budget ends. In assisted evolution, Claude Fable 5.1 proposes the skills; in self-evolution, Claude Opus 5 is both the agent and the proposer. Four proposers work in parallel; each reads at least ten rollouts of the current skill, including five successful and failed runs on the same instance, and writes at least four full rewrites, each with a predicted efect on accuracy and cost. A lineage stops after five consecutive proposals without improvement on the development split. Rewards come from the deterministic task graders. A skill enters the Pareto frontier if no earlier skill, the starting point included, is at least as accurate and at least as cheap; the selected skill is the most accurate on the development split, with ties within one standard error broken by cost. Cost is prefix-reuse FLOPs per task over runs that finish within the budget for Qwen3.6-27B, and USD per task for Opus 5. Results are in Figures 8 and 20.

Reinforcement learning. We train Qwen3.5-9B on 3,040 OpenResearcher deep-research prompts using GRPO. Training uses truncated importance sampling, a low-variance KL loss with coeficient 0.01, and dynamic sampling that removes groups with no reward variation. Each step samples 8 prompts with 32 rollouts per prompt. We use a learning rate of 10 <sup>6</sup>, weight decay of 0.1, and a clipping range of [0<sub>.</sub>2<sub>,</sub> 0<sub>.</sub>28], and train for 70 steps. Rollouts are served with SGLang at temperature 0.7 and top-� 0.95, with thinking enabled and up to

4,096 generated tokens per turn. CLM uses a 28K context budget with a 2,048-token reserve and up to 80 turns, excluding editing turns from the count; the summary harness compacts at 28,672 tokens and runs for up to 100 turns. We use a binary task reward from GPT-5.4-nano with the DeepSearchQA rubric. For CLM, we additionally apply the eficiency advantage from Section 4.2 with weight 0.25 to context-management tokens, together with penalties for failed tool calls and malformed outputs. The summary harness is trained with the task reward alone. Training uses 16 H200 GPUs for the policy and 48 for rollout generation. We select the checkpoint with the highest accuracy on 500 held-out OpenResearcher questions, breaking ties within one standard error in favor of the earlier checkpoint, and evaluate it on all 830 BrowseComp-Plus questions using the same judge. Prefix-reuse FLOPs are computed with Qwen3.5-9B model constants and exact token-level prefix matching against the previous turn. Results are reported in Table 2 and Figure 21.

## F Supplementary Results

TerminalBench 2.1, TBLite and BrowseComp-Plus with models of different sizes. Figure 17 places Qwen3.5-9B next to Qwen3.6-27B on the three benchmarks of Figure 5. CLM leaves the decision of when and how to edit the context to the model, so its gains grow with the model’s ability to make that decision. With Qwen3.6-27B, CLM lies on the Pareto frontier of all three benchmarks and reaches the highest accuracy on BrowseComp-Plus (59.4%). With Qwen3.5-9B, CLM reaches 39.9% on BrowseComp-Plus, above the summary harness (37.7%), and the smaller model edits its context less often: on TerminalBench 2.1, Qwen3.5-9B edits its context 1.4 times per task on average and makes no edit in half of the tasks, while Qwen3.6-27B edits 2.6 times per task. The median peak context is correspondingly higher for Qwen3.5-9B, 30.2K tokens of the 32K limit compared with 17.6K for Qwen3.6-27B.

![](images/1cc86f026f1632066264dcda05b90f2fd0c01a6782e194a8aa7f7a1b71df9fd2.jpg)

![](images/58b6c216e1bac6cc7974d0cc7e9674dfefa5cc1416e5e18d67d7ce8d06ef913a.jpg)

![](images/41457c0f2eee10dd8be2d8d4034e3cd4fac5f0e67540770e04fbe3e7c9eeca18.jpg)

![](images/359a58298899cbfc2fa9a6d929ae6a6eed0e103931ec6b4be6ad0dd1bc6de6c4.jpg)  
Prefix-reuse PFLOPs / question

![](images/111e9ef79ad09f7b9ce41c053eb697b68d01332e5e82e126240809015f1a0faa.jpg)  
Prefix-reuse PFLOPs / question

![](images/62f491fb9588a741a0471c5e77980c787765ac32ba7c248e17f60b5e046a6912.jpg)  
Figure 17 Qwen3.5-9B and Qwen3.6-27B. Accuracy against compute for Qwen3.5-9B (top) and Qwen3.6-27B (bottom) on BrowseComp-Plus, TerminalBench 2.1 and TBLite with a 32K context limit. Cost is prefix-reuse PFLOPs per question; dashed lines indicate the Pareto frontier.

Math optimization problems. Figure 18 shows the best score so far against the number of scored attempts for each method on the four problems.

EdgeBench-10 with a 128K context budget. Figure 19 repeats the EdgeBench-10 comparison of Figure 6 with a 128K context budget. The three methods that manage context keep improving over the twelve hours, while the base harness stops improving within the first two hours. At a 32K budget, CLM with and without subagents end within 0.4 points of each other (44.2 and 44.6). At 128K, CLM with subagents reaches 50.2, compared with 47.3 for CLM and 47.8 for summarization, using 219, 142 and 222 prefix-reuse PFLOPs per trial.

![](images/3e8923145b960c28006b2509c5aaffc95979111f9c1bb205bcdb9230a9610503.jpg)  
(a) Circle packing (↑).

![](images/cfad5fefddbf017752f536fc258d015a794a419627aa83bb49b18516f062d361.jpg)  
(b) Min-max/min-dist (↑).

![](images/44810defd91b2e289581e948be8ba4e96c1ae0d48335ae988944ba92b9c40c2f.jpg)  
(c) Erdős min-overlap (↓).  
100 attempts 5 h wall

![](images/aa5f83215dc8953c5ab41ba08557914fe345fea83baf221acbd6bd33608287c8.jpg)  
(d) Heilbronn triangle (↑).  
Figure 18 Best-so-far score versus evaluator-scored attempts on four open optimization problems. All runs use Claude 4.6 Sonnet with a 32K context limit and stop after 100 attempts or five hours. Lines show the best score so far, dots individual scored candidates, and insets the final-score range.

$$
\mathrm { - B a s e \mathrm { \ : - } s u m m a r y \mathrm { \ : - } C L M \mathrm { \ : - } C L M s \ : ( s u b a g e n t s ) }
$$

![](images/367f94a93a6da4d5448b4f5c0504aeb66f7fea2fc433cd80444cf5e35cb582e9.jpg)  
(a) Score against time.

![](images/33020ed99bcde8db10a9dd669d154c8653acafc8ed21cc386733c10326bd9b2a.jpg)  
(b) Score against compute.  
Figure 19 EdgeBench-10 single-repository optimization with a 128K context budget. Qwen3.6-27B, ten tasks, three seeds. End labels give final scores and mean compute per trial (PF = prefix-reuse PFLOPs); insets show the base harness.

In-context evolution. Figure 20 extends Figure 8 to all four tasks of ContextBench. In assisted evolution, starting without any context-management instruction, the selected skill raises development accuracy from 97.6% to 100.0% on Needle Retention, from 45.3% to 65.8% on Sudoku Sketchpad, from 22.3% to 83.8% on KV Store and from 0.0% to 100.0% on Log Triage. On the held-out test split, the selected KV Store skill raises accuracy from 38.3% to 74.2%. In self-evolution, Opus 5 starts between 94% and 100% accuracy, and the evolved skills either reduce cost at the same or higher accuracy or raise accuracy further.

Reinforcement learning. Figure 21 shows accuracy and compute on BrowseComp-Plus during training for CLM and the summary harness, each trained with and without the FLOPs reward.

start (no instruction) Needle Retention

![](images/567476539f42bd59f0ac84e43e2b2bd021da6555ed3ff4eda42fe6e01a0f7f33.jpg)  
final Pareto frontier Sudoku Sketchpad

![](images/fc6d39bb9d045ff09c870b5fd4319b2fe539251686cf558fad708c9a54d94a74.jpg)

![](images/d442eb2ef25a57efe3b01b0b3d766d1afb64524aa4083d34761ea0c6787813b5.jpg)

![](images/bf54ac8431b6160f39135b5f4b27264031babd0d128a5b5b5dcbfae1f438c865.jpg)

![](images/6addd4e2b7492a0b13fd27ab0c6d0ae4c52666e6bfd88f12a56c556ca9fd2bf4.jpg)

![](images/f80f32259d890756dac51c1193c173ba5cf1637753e6fde55adf069c7872e643.jpg)

![](images/ce552398a47a65ca074800ecc8ec8729ff2da60856e7fe9a351bb90b4471375e.jpg)

![](images/c8e127c5e307fef2643f1b0161a73f58e67611e3da92bdf7389b722aab5075f4.jpg)  
Figure 20 Evolving context-management skills on ContextBench (32K budget), all four tasks. Assisted evolution (top): Qwen3.6-27B serves as the agent, while Claude Fable 5.1 proposes the skills; the x-axis shows prefix-reuse PFLOPs per task over solved runs. Self-evolution (bottom): Opus 5 serves as both the agent and the skill proposer; the x-axis shows gateway cost per task in USD. Skills are proposed based on training-split rollouts and never use the held-out evaluation set.

![](images/f5b3fbb47e28003c92757dd5bededc3a0616fd4632293d8f209935c765f1c648.jpg)

![](images/3ecbed7216b0468de5b14ab137bc26e59ae158fa1b6e480c252f02f74abaaf23.jpg)

![](images/2cc7647c9e37fa4578cdf0092710c2bfa555d82a57e5bd4c788351af4f5d1081.jpg)  
Figure 21 RL training curves. Qwen3.5-9B trained on OpenResearcher and evaluated on BrowseComp-Plus with a 32K context limit, through training step 70. (a) Accuracy and prefix-reuse PFLOPs per question against training step. (b) Accuracy against compute, shaded from light to dark by training step.

## G Analysis on the Context Length Awareness of Existing LMs

An important capability for CLMs is context length awareness: the ability to estimate how much of the context budget has been consumed and to decide when context editing or ofloading is needed. Without such awareness, agents may rely on frequent external system interventions, which can leave stale or redundant information in future turns after context editing.

We probe context length awareness with a simple diagnostic. We provide each model with prompts of varying lengths and ask it to estimate the number of tokens in the current context. Figure 22 compares the models estimated context lengths against the actual prompt lengths. Interestingly, models tend to predict recurring bucketed values in the longer-context regime, such as 6.2K, 9.8K, or 10.4K tokens. We conjecture that these bucketed estimates may reflect token-counting patterns seen during pretraining or post-training. We further observe that Claude-4.6-Sonnet tends to underestimate context length, while GPT-5.4 achieves the strongest alignment between estimated and actual token counts among the three models. Furthermore, token-count hints improve context-length estimation, and hints closer to the estimation point are more efective. These results indicate that existing LMs have limited context-length awareness at long context lengths, where environmental hints can help substantially. Future CLM training may incorporate such signals to improve context-length awareness.

![](images/d21c1c289a971b4157be7d5780fdb50d3355f64dc845a4d58520cfc8cb7fd82e.jpg)

![](images/f827268488955659d38d397bdcdd905709c622e92b46b145e7546331b2cf1243.jpg)

![](images/6729a8df01228ce19efbd15c433f344cbc37c95257575f0a4c4e7c29e6085e01.jpg)

![](images/da2f9f3408e393db57573a52e34aadf52564b792f15c67bdd3c0e09beecc4eb1.jpg)  
Figure 22 Context length awareness. Each point compares a model’s estimated context length with the provider-reported prompt length for the same 50 inputs. Panels vary the available hint: none or a token-count anchor at 25%, 50%, or 75% of the input. The dashed line indicates perfect calibration; legend values report mean absolute error (MAE) in tokens.