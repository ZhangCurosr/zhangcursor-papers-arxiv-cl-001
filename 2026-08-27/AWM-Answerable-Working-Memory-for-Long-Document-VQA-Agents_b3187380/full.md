# AWM: Answerable Working Memory for Long-Document VQA Agents

Dongzhuoran Zhou<sup>1,5,\*</sup>, Yuqicheng Zhu<sup>2,5</sup>, Yule Liu<sup>3,\*</sup>, Zhen Yang<sup>4</sup>, Rui Lu<sup>4</sup>, Yuxiao Dong<sup>4</sup>, Jie Tang<sup>4</sup>, Evgeny Kharlamov<sup>1,5</sup>

<sup>1</sup>University of Oslo, <sup>2</sup>University of Stuttgart <sup>3</sup>Hong Kong University of Science and Technology (Guangzhou) <sup>4</sup>Tsinghua University, <sup>5</sup>Bosch Center for AI

## Abstract

Long-document visual question answering in creasingly relies on VLM agents that retrieve candidate pages, inspect page images, write findings to working memory, and synthesize answers. Working memory should carry answersupporting evidence across page inspections for later grounded answering, yet existing evaluation mainly checks final-answer correctness and evidence-page access. This creates a memory-quality blind spot: an agent may reach the right page and answer correctly while leaving behind memory too generic or incomplete to support answering once page context is removed. We introduce memory-only answerabil ity, a diagnostic that asks whether a reader can answer from the question and terminal working memory alone. Building on this diagnostic, Answerable Working Memory (AWM) treats termi nal working memory as an answerable evidence artifact, and AWM-GRPO incorporates this signal into the GRPO reward while preserving final-answer priority. Under GRPO, this reward assigns higher advantages to answer-correct trajectories whose terminal working memory remains answerable. On MMLONGBENCH-DOC, even when gold evidence pages are provided, 42.5% of correct answers still cannot be answered from terminal working memory alone. AWM-GRPO improves final-answer accuracy over the RAG baseline by 8.1 and 11.9 points on MMLONGBENCH-DOC and LONG-DOCURL and reduces the memory-missingcorrect rate by 2.7 points over answer-only GRPO.

## 1 Introduction

Answering questions over long documents resembles goal-directed information seeking: a reader searches for relevant evidence, keeps intermediate findings available, and integrates them into an answer. Recent long-document VQA systems instantiate a similar loop with VLM agents: they retrieve candidate pages, inspect page images, update working memory with findings, and synthesize a final answer (Faysse et al., 2025; Yu et al., 2025; Cho et al., 2024; Zheng et al., 2026; Lin et al., 2026; Lim et al., 2026). Working memory should carry answer-supporting evidence across page inspections for later grounded answering. Yet current evaluation checks final-answer correctness and evidence-page access, but not whether terminal working memory remains answer-supporting. An agent may therefore answer correctly from page context while leaving behind memory too generic to support the answer on its own.

We introduce memory-only answerability: a reader receives only the question and the agentwritten terminal working memory, without page images or trajectory context, and must answer from that memory alone. This exposes a memory-quality gap: even when gold evidence pages are supplied, 42.5% of correctly answered MMLONGBENCH-DOC examples fail memory-only answerability. A controlled memory comparison further shows that improving terminal working memory improves answering, motivating optimization of the memory artifact itself.

Motivated by this observation, we propose AWM-GRPO. For each trajectory, the agent outputs a final answer and terminal working memory. A frozen reader answers from the question and terminal working memory alone, and a judge scores both the final answer and the memory-only answer. Under GRPO, this reward ranks trajectories by both final-answer correctness and memory answerability, creating a preference for trajectories that answer correctly with answerable terminal working memory.

We evaluate a controlled training pipeline in which the three AWM-Agent variants share the same Qwen3-VL-4B policy, RAG Top-3 retrieval harness, tools, memory schema, prompts, and optimizer; only post-training and reward differ. We compare direct input, RAG Top-3, SFT, answer-only GRPO, and AWM-GRPO. On MMLONGBENCH-DOC/LONGDOCURL, AWM-GRPO exceeds RAG Top-3 by 8.1/11.9 points, SFT by 4.7/4.4 points, and answer-only GRPO by 2.3/2.7 points. It also reduces the memory-missingcorrect rate by 2.7 points over answer-only GRPO on MMLONGBENCH-DOC (Sec. 4.3).

This paper makes three contributions. (1) We identify a memory-quality gap in longdocument VQA agents: final-answer correctness and evidence-page access do not determine whether terminal working memory preserves answer-supporting evidence. (2) We propose AWM-GRPO, which uses memory-only answerability as a GRPO reward signal and ranks trajectories by both final-answer correctness and memory answerability, creating a preference for answercorrect trajectories with answerable terminal working memory. (3) We evaluate AWM-GRPO against direct-input, RAG, SFT, and answer-only GRPO baselines, showing improvements in both finalanswer accuracy and memory-quality diagnostics.

## 2 Agent Setup and Memory-Quality Motivation

This section introduces the agent-memory framework and the memory-only answerability metrics used throughout the paper. It then presents two diagnostics: current agent-written memory is often insufficient even when page access is controlled, and improving terminal working memory improves answering.

Agent-memory framework. A long-document VQA instance is a tuple $( D , q , y )$ , where $D =$ $\{ p _ { 1 } , \dotsc , p _ { N } \}$ is a multi-page document, q is a question, and y is the ground-truth answer; when available, $E \subseteq D$ denotes gold evidence pages. The policy π uses evidence-page retrieval and memory update. A retrieval call returns candidate page images directly in the observation; after inspecting these images, the agent must append a sourcelinked finding to working memory $M _ { t }$ . The interaction proceeds until the agent outputs both a final answer yˆ and terminal working memory $M ^ { \mathrm { t e r m } }$ , or reaches a step budget (Figure 1a).

As illustrated in Figure 1a, memory entries are source-linked findings whose quality depends on whether they preserve question-relevant evidence.

<table><tr><td>Final answer</td><td>Memory answerable</td><td>Case name</td></tr><tr><td>correct</td><td>yes</td><td>memory-supported correct</td></tr><tr><td>correct</td><td>no</td><td>memory-missing correct</td></tr><tr><td>wrong</td><td>yes</td><td>answering error</td></tr><tr><td>wrong</td><td>no</td><td>unresolved error</td></tr></table>

Table 1: Outcome cases induced by final-answer correctness and memory-only answerability.

A generic description such as “the chart shows trends” is less useful than a question-specific finding such as “the 2020 self-sufficiency rate is 4.7%.”

Memory-only answerability. To measure whether terminal working memory $M ^ { \mathrm { t e r m } }$ alone supports answering, we evaluate memory-only answerability: a frozen reader R receives only the question q and $M ^ { \mathrm { t e r m } }$ , without the trajectory or page images, and produces a memory-only answer $R ( q , M ^ { \mathrm { t e r m } } )$ The official judge J evaluates both the agent’s final answer and the reader’s memory-only answer against the gold answer y. For the reported four-cell analysis, we map any positive score from J to 1 and zero to 0. This gives two binary scores per instance: final-answer correctness $s _ { \mathrm { a n s } } = \mathbf { 1 } [ J ( \hat { y } , y ) > 0 ]$ and memory-only answerability $s _ { \mathrm { m e m } } = { \bf 1 } [ J ( R ( q , M ^ { \mathrm { t e r m } } ) , y ) > 0 ]$ Implementation details for R and J appear in Appendix B.

The pair $( s _ { \mathrm { a n s } } , s _ { \mathrm { m e m } } )$ partitions instances into four cases (Table 1). The focal case is memorymissing correct (MMC): $s _ { \mathrm { a n s } } = 1$ and $s _ { \mathrm { m e m } } = 0$ i.e., the agent answered correctly but its terminal working memory alone does not support answering.

We summarize each run with $P _ { \mathrm { m m c } }$ = $\scriptstyle \sum _ { i } \mathbf { 1 } [ s _ { \mathrm { a n s } } ^ { ( i ) } = 1 \land s _ { \mathrm { m e m } } ^ { ( i ) } = 0 ]$ , the fraction of final-answer-$\overline { { \sum _ { i } \mathbf { 1 } [ s _ { \mathrm { a n s } } ^ { ( i ) } = 1 ] } }$ correct instances whose terminal working memory is not answerable. Conditioning on $s _ { \mathrm { a n s } } { = } 1$ removes general answer failure as a confound and asks whether memory remains insufficient even after the agent answered correctly.

Current memory quality. We measure $P _ { \mathrm { m m c } }$ on a fixed 500-example answerable subset of MMLONGBENCH-DOC (excluding the benchmark’s unanswerable questions). The full multiturn setting runs the agent end-to-end with its own retrieval and memory-update process. The evidence-page-given (EP-given) setting replaces retrieval with gold evidence pages, so page access is controlled and retrieval is no longer a confound. Table 2 reports the results.

![](images/e3d322f9f827973c5ddabc31d817908f47a98e612a07bf2f1d51da45a1bfd912.jpg)  
Figure 1: Combined agent-memory framework and memory-quality diagnostics. (a) The agent retrieves page images, writes source-linked findings to working memory, and outputs both a final answer and terminal working memory. (b) Standard evaluation scores the final answer, memory-only evaluation asks whether the question can be answered from terminal working memory alone, and the controlled comparison varies terminal working memory while holding evidence-page access fixed.

<table><tr><td>Setting</td><td>N</td><td>Final-ans. acc. (%)</td><td>Memory-only acc. (%)</td><td>MMC / correct</td><td> $P _ { \mathrm { m m c } } \left( \% \right)$ </td></tr><tr><td>Full multi-turn 500</td><td></td><td>33.4</td><td></td><td>27.4 43 / 172</td><td>25.0</td></tr><tr><td>EP-given</td><td>500</td><td>36.2</td><td></td><td>30.2 77 / 181</td><td>42.5</td></tr></table>

Table 2: Memory-quality diagnostic for the Base AWM-Agent (Qwen3-VL-4B, no post-training) on a fixed 500-example answerable MMLONGBENCH-DOC subset. Unanswerable questions are excluded. Accuracy follows the official mean per-question score; for MMC/correct and $P _ { \mathrm { m m c } } .$ , we binarize the official score by treating a positive score as correct.

In the full multi-turn setting, one in four correct trajectories $( P _ { \mathrm { m m c } } = 2 5 . 0 \% )$ leaves behind terminal working memory that cannot support answering on its own. The EP-given setting controls for page access by replacing retrieval with gold evidence pages; $P _ { \mathrm { m m c } }$ rises to 42.5%, showing that page access and final-answer correctness do not imply answerable terminal working memory.

Controlled memory comparison. We next ask whether higher-quality memory improves answering. We construct a controlled setting in which retrieval returns gold evidence pages and compare three terminal-memory variants: (1) empty memory; (2) original agent memory; and (3) improved memory, generated from the same gold pages by a stronger model. To isolate memory quality, the comparison keeps gold-page access and the evaluation protocol fixed while substituting different terminal working memories. The answer generator, memory-only reader, and judge are shared across conditions, so only terminal-memory content changes (Figure 1b). Appendix D gives the full setup, confidence intervals, outcome-cell breakdown, and artifact audit.

<table><tr><td>Memory condition</td><td>Final-ans. binary rate</td><td>Memory-only binary rate</td></tr><tr><td>Empty memory</td><td>2.8</td><td>2.6</td></tr><tr><td>Original agent memory (4B)</td><td>36.2</td><td>30.2</td></tr><tr><td>Improved memory (GPT-4o)</td><td>44.4</td><td>44.8</td></tr></table>

Table 3: Controlled memory comparison on a fixed 500-example MMLONGBENCH-DOC EP-given answerable subset. The reader and scorer are fixed; only the terminal-memory condition changes. Improved memory is produced by GPT-4o.

Table 3 isolates the role of terminal memory. With empty memory, both final-answer and memory-only binary rates are near zero. Reusing the 4B agent’s original memory yields final-answer and memory-only binary rates of 36.2 and 30.2, while GPT-4o-improved memory raises them to 44.4 and 44.8. Improving terminal memory alone therefore improves both readouts with the answer generator, reader, and judge held fixed.

Takeaway. Page access and final-answer correctness do not guarantee answerable terminal working memory. Memory-only answerability exposes this gap, and the controlled comparison shows that improving terminal working memory alone can improve answering.

## 3 Optimizing Answerable Working Memory with GRPO

Sec. 2 identifies the failure mode that AWM-GRPO targets: trajectories can answer correctly while leaving terminal working memory unanswerable. Since GRPO updates the policy through grouprelative advantages, the reward should make this memory-quality distinction visible in the advantage ranking. We first specify the desired advantage behavior, then instantiate a four-cell reward that realizes it, and finally use synthetic outcome mixtures to show how the induced preferences change with group composition.

![](images/4d9b460352f8c5fc28090f6c6a3e261ee1b47424db89127bdb0d4c446c0498ed.jpg)  
Figure 2: Overview of AWM-GRPO. For each sampled trajectory, the agent produces both a final answer and terminal working memory. A frozen reader answers from the terminal working memory alone, and a frozen scorer evaluates the final answer and the memory-only answer. The two scores define the AWM reward, which is normalized within each GRPO group to update the policy.

## 3.1 Desired advantage behavior under GRPO

For each prompt, GRPO (Shao et al., 2024) samples G trajectories and normalizes each trajectory’s reward within the group:

$$
A _ { i } = \frac { r ( \tau _ { i } ) - \mu _ { g } } { \sigma _ { g } + \epsilon } ,
$$

where $\mu _ { g }$ and $\sigma _ { g }$ are the group mean and standard deviation and $\epsilon = 1 0 ^ { - 6 }$ stabilizes zero-variance groups. Because GRPO updates through these normalized advantages, what matters is not the raw reward scale but the ordering it induces: for any two outcome cells $z , z ^ { \prime } .$

$$
A _ { z } - A _ { z ^ { \prime } } = \frac { r _ { z } - r _ { z ^ { \prime } } } { \sigma _ { g } + \epsilon } .
$$

The reward ordering therefore determines the advantage ordering. We design the AWM reward to induce three pairwise preferences:

Memory refinement. Among answer-correct trajectories, those whose terminal working memory remains answerable should receive higher advantage than those whose memory is not answerable.

Final-answer gate. Final-answer correctness remains primary: an answer-correct but memoryimperfect trajectory should still outrank an answerwrong trajectory, even if the latter leaves answerable memory.

Useful-memory failure. An answer-wrong trajectory with answerable memory is not success, but it is less severe than complete failure, because useful evidence was preserved.

GRPO gives positive advantage only to cells whose reward exceeds the current group mean. The same cell can therefore change role as group composition changes: answer-correct but memoryimperfect trajectories may be tolerated when most samples fail, but become disfavored once memorysupported-correct trajectories are common. Formally, $A _ { z } > 0 \operatorname { i f f } r _ { z } > \mu _ { g } .$

## 3.2 AWM reward instantiation

To realize these preferences with a minimal scalar reward, we reuse the two binary variables from Sec. 2: final-answer correctness $s _ { \mathrm { a n s } }$ and memoryonly answerability $s _ { \mathrm { m e m } }$ . During online training, a frozen local Qwen3-14B model generates the memory-only answer and scores both answers for the reward. The official judge J from Sec. 2 is used for reported diagnostics. The AWM reward is defined over the resulting four outcome cells:

$$
\begin{array} { c c c } { { r _ { 1 1 } = \beta , } } & { { r _ { 1 0 } = 0 , } } & { { r _ { 0 1 } = \gamma , } } & { { r _ { 0 0 } = \omega , } } \\ { { } } & { { } } & { { \beta > 0 > \gamma > \omega . } } \end{array}
$$

We write $R _ { \mathrm { A W M } } ( s _ { \mathrm { a n s } } , s _ { \mathrm { m e m } } ) \ = \ r _ { s _ { \mathrm { a n s } } s _ { \mathrm { m e m } } } ;$ the answer-only baseline uses $R _ { \mathrm { a n s } } ( s _ { \mathrm { a n s } } ) = s _ { \mathrm { a n s } } .$ Here $\beta$ is the reward for full success (memorysupported correct), γ is the reward for answerwrong but memory-answerable trajectories, and $\omega$ is the reward for complete failure. The anchor $r _ { 1 0 } ~ = ~ 0$ fixes answer-correct but memoryimperfect trajectories. The resulting normalized

margins are:
<table><tr><td>Preference</td><td>Margin</td><td>Ranking</td></tr><tr><td>Memory refinement</td><td> ${ A _ { 1 1 } } - { A _ { 1 0 } } = \beta / ( \sigma _ { g } + \epsilon )$ </td><td> $( 1 , 1 ) \succ ( 1 , 0 )$ </td></tr><tr><td>Final-answer gate</td><td> $A _ { 1 0 } - A _ { 0 1 } = - \gamma / ( \sigma _ { g } + \epsilon )$ </td><td> $( 1 , 0 ) \succ ( 0 , 1 )$ </td></tr><tr><td>Useful-memory failure</td><td> $A _ { 0 1 } - A _ { 0 0 } = ( \gamma - \stackrel { . } { \omega } ) \stackrel { . } { / } ( \sigma _ { g } + \epsilon )$ </td><td> $( 0 , 1 ) \succ ( 0 , 0 )$ </td></tr></table>

The memory-refinement margin is exactly what answer-only reward cannot express: it scores (1, 1) and (1, 0) identically, collapsing $A _ { 1 1 } ^ { \mathrm { a n s } } = A _ { 1 0 } ^ { \mathrm { a n s } }$

Each cell’s advantage sign depends on the group mean $( A _ { z } > 0$ iff $r _ { z } > \mu _ { g } ) \mathrm { : }$
<table><tr><td>Cell</td><td>Positive iff</td><td>Interpretation</td></tr><tr><td>(1, 1)</td><td> $\mu _ { g } < \beta$ </td><td>full success favored</td></tr><tr><td>(1,0)</td><td> $\mu _ { g } < 0$ </td><td>flips at  $\mu _ { g } { = } 0$ </td></tr><tr><td>(0, 1)</td><td> $\mu _ { g } < \gamma$ </td><td>flips at  $\mu _ { g } = \gamma$ </td></tr><tr><td>(0,0)</td><td> $\scriptstyle { \mathrm { n e v e r } }$ </td><td>zero only if  $\mu _ { g } { = } \omega$ </td></tr></table>

The main experiments instantiate $\beta = 2 , \gamma =$ −0.1, and $\omega = - 1$ , yielding $( r _ { 1 1 } , r _ { 1 0 } , r _ { 0 1 } , r _ { 0 0 } ) =$ $( 2 , \quad 0 , \quad - 0 . 1 , \quad - 1 )$ Under this default, the wrong-answer but memory-answerable cell $( r _ { 0 1 } { = } \gamma { = } { - } 0 . 1 )$ is slightly below the answer-correct / memory-imperfect cell $( r _ { 1 0 } { = } 0 )$ , so it becomes negative first as the group mean rises. When the group mean exceeds $r _ { 1 0 } .$ , the answer-correct / memory-imperfect cell also becomes negative and positive advantage concentrates on full success. This gives the reward an answer-first, memoryrefinement-later tendency without implying a deterministic curriculum guarantee.

## 3.3 Advantage distributions under synthetic outcome mixtures

Figure 3 uses a controlled Monte Carlo simulation to show how the answer-only and AWM rewards translate into GRPO advantages. For each row, we fix a categorical distribution over the four outcome cells of Table 1 and draw 15,000 independent GRPO groups, each containing 8 trajectories. The answer-only simulation samples from the corresponding final-answer marginal, while the AWM simulation samples the four cells directly. Rows increase $p _ { 1 1 }$ , the probability of memory-supported correct (1, 1), so later rows contain more memorysupported correct trajectories. The simulation uses the default reward instantiation $\beta = 2 , \gamma = - 0 . 1$ $\omega = - 1$ . It isolates reward-induced group-relative preferences; curves are smoothed for visualization only. Implementation details appear in $\mathsf { A p - }$ pendix C.

## Three effects are visible:

• Answer-only hides memory quality: in the left column, trajectories are grouped only by final-answer correctness, so Ans✓ Mem✓ and

![](images/52f48cbba25310705fb885ba4a40ac8547ee0b1f1b232a7e029f3350c4256e1d.jpg)  
Figure 3: Simulated GRPO advantage distributions. Each row samples synthetic groups of size $G { = } 8$ from a four-cell outcome mixture; $p _ { 1 1 }$ is the probability of memory-supported correct (1, 1). Left: answer-only reward collapses the four cells into answer-wrong and answer-correct regions. Right: AWM assigns a distinct raw reward to each of the four cells before group normalization.

Ans✓ Mem× are learned together as a single answer-correct region; memory answerability is invisible.

• AWM distinguishes the answer-correct cells: in the right column, Ans✓ Mem× and Ans✓ Mem✓ receive distinct raw rewards and can therefore receive different normalized advantages. This is the memory-refinement preference missing from answer-only reward.

• The preference changes with group composition: as $p _ { 1 1 }$ increases, the red Ans✓ Mem× curve moves below zero while the blue Ans✓ Mem✓ curve remains positive. The nearby green Ans× Mem✓ curve is expected because $r _ { 0 1 } = - 0 . 1$ is close to $r _ { 1 0 } { = } 0 ;$ the key comparison is between the two answer-correct cells.

What the reward does not claim. The reward uses memory-only answerability as a training signal, not as a claim of full source-grounding verification. It requires no gold terminal working memory: the frozen reader R answers from the agentwritten $M ^ { \mathrm { t e r m } }$ , and the frozen training scorer evaluates both the agent’s final answer and the memoryonly answer against the gold answer. The reward does not score memory style, length, or formatting, and memory-only answerability does not override final-answer correctness; $r _ { 1 0 } > r _ { 0 1 }$ preserves finalanswer priority. Appendix C discusses reward variants, simulation details, and why we adopt this minimal form.

## 4 Experiments

We evaluate whether AWM-GRPO improves finalanswer accuracy and terminal-working-memory quality over direct-input, RAG, SFT, and answeronly GRPO baselines.

## 4.1 Experimental Setup

Benchmarks. We evaluate on LONG-DOCURL (Deng et al., 2025) (N=2325) and MMLONGBENCH-DOC (Ma et al., 2024) (N=1082), under each benchmark’s official evaluation protocol (Appendix A).

Metrics. We report final-answer accuracy, memory-only accuracy, and $P _ { \mathrm { m m c } }$ from Sec. 2. Both final and memory-only answers are evaluated by the official judge J defined in Sec. 2. The judge uses GPT-4o to extract a concise candidate answer and then applies each benchmark’s deterministic rules to compute generalized accuracy.

Agent and retrieval harness. The policy is Qwen3-VL-4B-Instruct served via SGLang. The agent runs the retrieve-inspect-update memory loop of Sec. 2 with an append-only mandatory memoryupdate policy and a maximum of 15 agent steps per question. Page retrieval uses Jina v4 page-image embeddings indexed by Qdrant, returning the top three candidate pages per query (RAG Top-3).

Post-training data. SFT and GRPO use posttraining prompts from DOC-750K (Duan et al., 2025), disjoint from the evaluation sets. A 2000- question rendered subset yields 994 SFT trajectories via difficulty-balanced bucket sampling, with kimi-k2.5 as the teacher; the remaining 1006 questions form the RL-unseen pool, from which 226 mixed-difficulty groups are retained for GRPO training. Online GRPO uses a frozen, locally hosted Qwen3-14B model to generate the memoryonly answer and compute both reward scores; only the 4B policy is updated.

Variants. We evaluate two no-memory baselines and three AWM-Agent variants. The direct-input baseline receives all document pages in one pass, without retrieval, tools, or terminal working memory. The second, VLM + RAG Top-3, retrieves the top three pages and answers directly without writing terminal working memory. The three AWM-Agent variants share the same model, tools, memory schema, retrieval harness, prompts, and optimizer; only the post-training step or reward differs: (i) AWM-Agent (SFT): trained on the 994- trajectory DOC-750K subset. (ii) AWM-Agent + Answer-GRPO: the SFT checkpoint with GRPO under the answer-only reward $R _ { \mathrm { a n s } } .$ . (iii) AWM-Agent + AWM-GRPO: the SFT checkpoint with GRPO under the AWM reward $R _ { \mathrm { A W M } }$ at default values $( 2 , 0 , - 0 . 1 , - 1 )$ (Sec. 3.2).

## 4.2 Main Results

Table 4 reports final-answer accuracy for external baselines and our variants. The external rows use different backbones and published protocols, so they provide context rather than controlled comparisons. Our method comparisons use the five Qwen3-VL-4B rows, which share the backbone and official scoring protocol.

The direct-input and RAG rows provide nomemory baselines, while Sec. 4.3 analyzes memory quality only for methods that produce terminal working memory. On MMLONGBENCH-DOC, AWM-GRPO reaches 53.9 and improves over direct input, RAG Top-3, SFT, and answer-only GRPO by 10.4, 8.1, 4.7, and 2.3 points, respectively. It also improves over RAG Top-3, SFT, and answer-only GRPO by 11.9, 4.4, and 2.7 points on LONGDOCURL.

## 4.3 Analysis

Memory-quality analysis. Tables 5 and 6 report memory-quality metrics across the three AWM-Agent variants on MMLONGBENCH-DOC. The direct-input and RAG baselines are excluded because they do not produce terminal working memory. Table 5 uses the full multi-turn setting; Table 6 uses the EP-given setting, where retrieval is replaced by gold evidence pages to isolate memory extraction from page access.

SFT already achieves a relatively low $P _ { \mathrm { m m c } }$ of 17.7%. Answer-only GRPO improves final-answer accuracy but raises $P _ { \mathrm { m m c } }$ to 19.9%, because it assigns the same reward to memory-supported correct and memory-missing correct trajectories; its effect on memory quality is incidental. AWM-GRPO separates these two cells through the AWM reward (Sec. 3). Compared with answer-only GRPO, AWM-GRPO improves memory-only accuracy from 42.5 to 44.5 and reduces $P _ { \mathrm { m m c } }$ from

<table><tr><td>Method</td><td>Backbone</td><td>Param</td><td>Paradigm</td><td>MMLONGBENCH-DOC</td><td>LONGDOCURL</td></tr><tr><td colspan="6">Open-source / RAG</td></tr><tr><td>Qwen2.5-VL (Zheng et al., 2026)</td><td>Qwen2.5-VL</td><td>7B</td><td>VLM</td><td>28.0</td><td>32.9</td></tr><tr><td>SV-RAG (Chen et al., 2024)</td><td>InternVL2</td><td>4B</td><td>RAG</td><td>23.0</td><td></td></tr><tr><td>M3DocRAG (Cho et al., 2024)</td><td>Qwen2-VL</td><td>7B</td><td>RAG</td><td>21.0</td><td>35.1</td></tr><tr><td>VisRAG (Yu et al., 2025)</td><td>MiniCPM-V</td><td>8B</td><td>RAG</td><td>18.8</td><td>41.9</td></tr><tr><td>MoLoRAG (Wu et al., 2025)</td><td>Qwen2.5-VL</td><td>7B</td><td>RAG</td><td>41.0</td><td>51.9</td></tr><tr><td>URaG-7B (Shi et al., 2026)</td><td>Qwen2.5-VL</td><td>7B</td><td>RAG</td><td>33.8</td><td>52.2</td></tr><tr><td colspan="6">Agentic / RL</td></tr><tr><td>VRAG-RL (Wang et al., 2025)</td><td>Qwen2.5-VL</td><td>7B</td><td> $\mathbf { A g e n t } + \mathbf { R L }$ </td><td>26.6</td><td>44.9</td></tr><tr><td>Doc-V* (SFT) (Zheng et al., 2026)</td><td>Qwen2.5-VL</td><td>7B</td><td> $\mathbf { A } \mathbf { \bar { g } e n t + S F T }$ </td><td>39.8</td><td>53.0</td></tr><tr><td>Doc-V* (GRPO) (Zheng et al., 2026)</td><td>Qwen2.5-VL</td><td>7B</td><td> $\mathbf { A g e n t + G R P O }$ </td><td>42.1</td><td>56.3</td></tr><tr><td colspan="6"> $O u r s ( Q w e n 3 - V L - 4 B )$ </td></tr><tr><td>Qwen3-VL-4B (direct input)</td><td>Qwen3-VL</td><td>4B</td><td>VLM</td><td>43.5</td><td></td></tr><tr><td>VLM + RAG Top-3 (no memory)</td><td>Qwen3-VL</td><td>4B</td><td>RAG</td><td>45.8</td><td>48.2</td></tr><tr><td>AWM-Agent (SFT)</td><td>Qwen3-VL</td><td>4B</td><td> $\mathrm { A g e n t } + \mathrm { S F T }$ </td><td>49.2</td><td>55.7</td></tr><tr><td> $+ \mathrm { A n s w e r – G R P O }$ </td><td>Qwen3-VL</td><td>4B</td><td> $\mathrm { G R P O } \left( R _ { \mathrm { a n s } } \right)$ </td><td>51.6</td><td>57.4</td></tr><tr><td> $+ \Delta \mathbf { W } \mathbf { M } – \mathbf { G } \mathbf { R } \mathbf { P } 0$ </td><td>Qwen3-VL</td><td>4B</td><td> $\mathrm { G R P O } \left( R _ { \mathrm { A W M } } \right)$ </td><td>53.9</td><td>60.1</td></tr></table>

Table 4: Final-answer accuracy; higher is better. External rows follow their reported protocols and provide context only. Our five Qwen3-VL-4B rows use the official scoring protocol.

<table><tr><td>Variant</td><td>Final acc.</td><td>Mem acc.</td><td> $P _ { \mathrm { m m c } } \left( \% \right)$ </td></tr><tr><td>SFT</td><td>40.8</td><td>38.5</td><td>17.7</td></tr><tr><td>+ Answer-GRPO</td><td>43.2</td><td>42.5</td><td>19.9</td></tr><tr><td>+ AWM-GRPO</td><td>45.4</td><td>44.5</td><td>17.2</td></tr></table>

Table 5: Memory-quality analysis of the three AWM-Agent variants on 1,082 MMLONGBENCH-DOC trajectories in the full multi-turn setting.
<table><tr><td>Variant</td><td>Subset</td><td>Final acc.</td><td>Mem acc.</td><td> $P _ { \mathrm { m m c } }$  (%)</td></tr><tr><td>+ Answer-GRPO + AWM-GRPO</td><td>MM EP MM EP</td><td>45.4 48.0</td><td>41.2 43.5</td><td>19.1 16.4</td></tr></table>

Table 6: Memory-quality analysis of Answer-GRPO and AWM-GRPO on 733 answerable MMLONGBENCH-DOC examples in the EP-given setting.

19.9% to 17.2%, while also improving final-answer accuracy. AWM-GRPO also improves both finalanswer and memory-only accuracy over SFT (+4.6 and +6.0 points) while keeping $P _ { \mathrm { m m c } }$ slightly lower (17.2 vs. 17.7). The targeted failure is narrow: after a relevant page is observed, the agent must commit a question-conditioned finding to memory. AWM-GRPO supplies trajectory-level reward for this terminal-working-memory artifact, rather than changing retrieval tools or the answer prompt. Table 6 reports the EP-given comparison: with retrieval replaced by gold evidence pages, AWM-GRPO improves final-answer accuracy from 45.4 to 48.0, memory-only accuracy from 41.2 to 43.5, and reduces $P _ { \mathrm { m m c } }$ from 19.1% to 16.4% relative to answer-only GRPO, showing that the memoryaware reward remains beneficial even when page access is controlled.

![](images/50db9b7303e4ef880d6457a3f6fd89c14570b11e67e8d36e14ac179207fcefed.jpg)  
Figure 4: AWM-GRPO memory-only accuracy over training steps on MMLONGBENCH-DOC. Memoryonly accuracy increases from 38.8 at step 40 to 43.5 at step 280.

Training dynamics. We track memory-only accuracy across AWM-GRPO training steps on MMLONGBENCH-DOC. Figure 4 shows a monotonic increase from 38.8 at step 40 to 43.5 at step 280, suggesting that the reward progressively improves the answerability of terminal working memory during training.

Evidence-source analysis. Table 7 compares Answer-GRPO and AWM-GRPO by answersource category. AWM-GRPO lowers $P _ { \mathrm { m m c } }$ for mixed visual+text, text-multi, figure, chart, and none categories, while table and layout questions show higher $P _ { \mathrm { m m c } }$ . This suggests that the memory reward helps in several cross-source settings but does not uniformly solve structured-evidence

<table><tr><td></td><td>Final acc.</td><td> $P _ { \mathrm { m m c } }$ </td><td>(%)</td></tr><tr><td>Source</td><td>N Ans AWM</td><td> Ans AWM</td><td></td></tr><tr><td>Pure-text</td><td>13546.1 58.5</td><td>22.8</td><td>23.1</td></tr><tr><td>Table</td><td>143 36.7 39.5</td><td>15.2</td><td>23.9</td></tr><tr><td>Chart</td><td>10745.7 46.0</td><td>21.4</td><td>17.5</td></tr><tr><td>Figure</td><td>18741.0 43.9</td><td>20.3</td><td>16.2</td></tr><tr><td>Layout</td><td>32 75.9 74.3</td><td>4.8</td><td>15.8</td></tr><tr><td>visual+text (mixed) visual (multi)</td><td>18036.7 37.3</td><td>36.2</td><td>22.2</td></tr><tr><td>text (multi)</td><td>3 62 40.1 48.3</td><td>31.8</td><td>24.0</td></tr><tr><td>none</td><td>233 47.5 49.2</td><td>10.4</td><td>5.4</td></tr><tr><td>All</td><td>1082 43.2 45.4</td><td>19.9</td><td>17.2</td></tr></table>

Table 7: Evidence-source comparison on MMLONGBENCH-DOC in the full multi-turn setting. “Ans” denotes Answer-GRPO and “AWM” denotes AWM-GRPO; bold marks the better value within each source.

preservation.

## 5 Related Work

Long-document VQA agents. MMLONGBENCH-DOC (Ma et al., 2024) and LONGDOCURL (Deng et al., 2025) are two recent benchmarks for long-document VQA over PDFs that average tens to hundreds of pages and mix textual, tabular, and figure-grounded questions. Doc-V<sup>⋆</sup> treats this task as sequential page navigation: the agent starts from a low-resolution overview, fetches selected page images, and records findings in structured working memory before answering (Zheng et al., 2026). Its navigation policy is trained with imitation learning and GRPO. MemSearcher studies compact working memory across turns in text-based search agents rather than page-image VQA, using multi-context GRPO to propagate trajectory-level advantages to individual turns (Yuan et al., 2025). AgenticRAG-R1 uses stack memory for multi-step reasoning, retrieval, and memorizing, while TC-RAG studies Turingcomplete RAG for medical LLM systems (Jiang et al., 2026, 2025). SCAIR, GR-Agent, and EigentSearch structure retrieval or reasoning with schemas or tools (Chaturvedi et al., 2026; Zhou et al., 2025b; Zhang et al., 2026; Li et al., 2025). AWM keeps the agent architecture and memory format, but evaluates whether terminal memory is answerable on its own and uses that signal as an RL reward.

Intermediate-state supervision. TableQA work filters questions and prunes tables to retain answercritical content (Ye et al., 2026; Guo et al., 2026). For visual inputs, CapRL rewards answerable captions, while Vision-SR1 optimizes questionconditioned visual descriptions and answers (Xing et al., 2026; Li et al., 2026b,a; Xiao et al., 2026b, 2025). Perception-R1 checks visual content in a reasoning response against reference annotations (Xiao et al., 2026a). AWM instead evaluates the answerability of query-specific terminal working memory accumulated across sequential page inspections.

Memory evaluation. Evidence-citation work asks whether an agent’s final answer is supported by its cited evidence (Liu et al., 2023; Gao et al., 2023). Related RAG work tests reasoning when retrieved knowledge is incomplete or exposes the argument structure behind an answer (Zhou et al., 2026a; Zhu et al., 2025b). Memory-only answerability asks whether saved memory is sufficient to answer after the page images and interaction trajectory are removed, a question related to self-evaluation in retrieval-augmented generation (Asai et al., 2024). Neither test subsumes the other: answerable memory may contain unsupported claims, while grounded memory may omit a fact needed to answer. Other diagnostics audit prompt-conditioned traces left by RLVR (Liu et al., 2025); they target data membership, not whether terminal memory supports answering. In our diagnostic, a frozen language model reads only $( q , M ^ { \mathrm { t e r m } } )$ and produces a memory-only answer, which the official benchmark judge scores alongside the agent’s final answer. This reader-based diagnostic is related to LLM-based evaluation (Zheng et al., 2023; Gilardi et al., 2023). Memory probes in interpretability (Ghandeharioun et al., 2024) ask whether internal activations encode a fact, whereas we ask whether an externalized, source-linked terminal record encodes it.

## 6 Conclusion

Memory-only answerability identifies cases in which an agent answers correctly but leaves terminal working memory that cannot support the answer on its own. AWM-GRPO improves accuracy over answer-only GRPO on both benchmarks and lowers $P _ { \mathrm { m m c } }$ in both settings. Two benchmarks and one model size leave grounding unverified.

## Limitations

All controlled AWM comparisons use Qwen3-VL-4B on LONGDOCURL and MMLONGBENCH-DOC; whether AWM-GRPO helps at 30B+ or on other document distributions, such as scientific papers, financial filings, and slide decks, remains open. Memory-only answerability may depend on the frozen Qwen3-14B reader, and we did not test robustness with a second reader. Errors in the fixed judge’s answer-extraction step can affect the reported metrics. Our 100-example audit found six disagreements but no consistent direction of error; a larger audit would provide stronger evidence. The AWM reward measures answerability, not full source-grounding verification; future work can add source-conditioned checks for individual memory claims. Memory-only answerability also requires an extra reader pass and an extraction pass during offline evaluation, although neither pass is needed when the trained agent is deployed.

## Ethics Statement

This work uses publicly released benchmarks (MMLONGBENCH-DOC, LONGDOCURL), a public open-weight VLM (Qwen3-VL-4B-Instruct), and a fixed local Qwen3-14B model for memory reading and reward scoring during online training. We collect no new data, do not work with human subjects, and do not target deployment. The official judge follows each benchmark’s answer-extraction and rule-based scoring pipeline. Separately, the controlled intervention in Appendix D uses GPT-4o to construct improved memory from gold evidence pages; those memories are not used for training. The fixed training scorer, reader, and answer extractor may introduce systematic errors; our manual audit does not establish broader scoring robustness or claim-level source grounding. Per ARR / EMNLP policy, AI assistants were used for editing; all technical content is the authors’ own.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, volume 2024, pages 9112–9141.

Prateek Chaturvedi, Yuqicheng Zhu, Hongkuan Zhou, Dongzhuoran Zhou, Yunjie He, Steffen Staab, Fei Du, Jie Tang, and Evgeny Kharlamov. 2026. SCAIR: Schema-conditioned agentic iterative reasoning for

enterprise knowledge graphs. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), pages 1089–1104.

Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan A. Rossi, Changyou Chen, and Tong Sun. 2024. Svrag: Lora-contextualizing adaptation of mllms for long document understanding. arXiv preprint arXiv:2411.01106.

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multimodal retrieval is what you need for multi-page multi-document understanding. arXiv preprint arXiv:2411.04952.

Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, and Cheng-Lin Liu. 2025. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1135–1159.

Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, Jifeng Dai, and Wenhai Wang. 2025. Docopilot: Improving multimodal models for document-level understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4026–4037.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. Colpali: Efficient document retrieval with vision language models. In International Conference on Learning Representations, volume 2025, pages 61424–61449.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488.

Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, and Mor Geva. 2024. Patchscopes: A unifying framework for inspecting hidden representations of language models. arXiv preprint arXiv:2401.06102.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd-workers for textannotation tasks. Proceedings of the National Academy ofSciences, 120(30):e2305016120.

Yu Guo, Shenghao Ye, Shuangwu Chen, Zijian Wen, Tao Zhang, Qirui Bai, Dong Jin, Yunpeng Hou, Huasen He, Jianyang, and Xiaobin Tan. 2026. Rethinking table pruning in TableQA: From sequential revisions to gold trajectory-supervised parallel search. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12960–12976.

Yuan He, Bailan He, Zifeng Ding, Alisia Maria Lupidi, Yuqicheng Zhu, Shuo Chen, Caiqi Zhang, Jiaoyan Chen, Yunpu Ma, Volker Tresp, and Ian Horrocks. 2025. Supposedly equivalent facts that aren’t? entity frequency in pre-training induces asymmetry in LLMs. In Second Conference on Language Modeling.

Xinke Jiang, Yue Fang, Rihong Qiu, Haoyu Zhang, Yongxin Xu, Hao Chen, Wentao Zhang, Ruizhe Zhang, Yuchen Fang, Xu Chu, and 1 others. 2025. TC-RAG: Turing-complete RAG’s case study on medical LLM systems. ACL oral 2025.

Xinke Jiang, Yue Fang, Zhibang Yang, Jiaran Gao, Zhixin Zhang, Tao Feng, Rihong Qiu, Wentao Zhang, Hongxin Ding, Ruizhe Zhang, and 1 others. 2026. Agenticrag-r1: Agentic reinforcement learning with stack memory for multi-step reasoning, retrieval and memorizing. In EMNLP.

Songze Li, Xiaoke Guo, Tianqi Liu, Biao Yi, Zhaoyan Gong, Zhiqiang Liu, Huajun Chen, and Wen Zhang. 2026a. What’s missing in screen-to-action? towards a ui-in-the-loop paradigm for multimodal gui reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 17674–17690.

Songze Li, Zhiqiang Liu, Zhengke Gui, Huajun Chen, and Wen Zhang. 2025. Enrich-on-graph: Querygraph alignment for complex reasoning with LLM enriching. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 7683–7703.

Zongxia Li, Wenhao Yu, Chengsong Huang, Zhenwen Liang, Rui Liu, Fuxiao Liu, Jingxi Chen, Dian Yu, Jordan Lee Boyd-Graber, Haitao Mi, and Dong Yu. 2026b. Vision-SR1: Self-rewarding vision-language model via reasoning decomposition and multi-reward policy optimization. In International Conference on Learning Representations.

Gyubeum Lim, Yemo Koo, and Vijay Krishna Madisetti. 2026. Scope VLM: selective context processing for efficient document navigation in vision-language models. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 95–140.

Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhiheng Xi, Honglin Guo, Shichun Liu, Junzhe Wang, Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han, Tao Gui, Qi Zhang, and Xuanjing Huang. 2026. Mmdoc-r1: Training agents for long document visual question answering through multi-turn reinforcement learning. arXiv preprint arXiv:2604.13579.

Nelson F. Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating verifiability in generative search engines. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7001–7025.

Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Jiaheng Wei, Tianshuo Cong, Yilong Yang,

and Xinlei He. 2025. Auditing data membership in reinforcement learning with verifiable rewards. arXiv preprint arXiv:2511.14045.

Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024. MMLONGBENCH-DOC: benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems, 37:95963–96010.

Nico Potyka, Yuqicheng Zhu, Yunjie He, Evgeny Kharlamov, and Steffen Staab. 2024. Robust knowledge extraction from large language models using social choice theory. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems, pages 1593–1601.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. URL https://arxiv. org/abs/2402.03300, 2(3):5.

Yongxin Shi, Jiapeng Wang, Zeyu Shan, Dezhi Peng, Zening Lin, and Lianwen Jin. 2026. Urag: Unified retrieval and generation in multimodal llms for efficient long document understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 25357–25365.

Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. 2025. VRAG-RL: empower visionperception-based RAG for visually rich information understanding via iterative reasoning with reinforcement learning. CoRR, abs/2505.22019.

Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and Hong Cheng. 2025. Molorag: Bootstrapping document understanding via multi-modal logic-aware retrieval. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 14024–14045.

Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu, and Enhong Chen. 2026a. Perception-R1: Advancing multimodal reasoning capabilities of MLLMs via visual perception reward. In International Conference on Learning Representations.

Xi Xiao, Chen Liu, Chih-Ting Liao, Yunbei Zhang, Qizhen Lan, Yuxiang Wei, Lin Zhao, Janet Wang, Jianyang Gu, Muchao Ye, and 1 others. 2026b. Staying VIGILant: Mitigating visual laziness via counterfactual visual alignment in MLLMs. arXiv preprint arXiv:2606.26387.

Xi Xiao, Yunbei Zhang, Xingjian Li, Tianyang Wang, Xiao Wang, Yuxiang Wei, Jihun Hamm, and Min Xu. 2025. Visual instance-aware prompt tuning. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 2880–2889.

Can Xie, Ruotong Pan, Xiangyu Wu, Yunfei Zhang, Jiayi Fu, Tingting Gao, and Guorui Zhou. 2026a. Unlocking exploration in RLVR: Uncertainty-aware advantage shaping for deeper reasoning. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 19057–19076.

Can Xie, Yuyi Zhou, Wen Yang, Ziyi Zhang, Siyao Song, Yingzhuo Deng, Shuo Ren, and Jiajun Zhang. 2026b. EDGE: Experience-distillation for guided exploration in agentic reinforcement learning. Preprint, arXiv:2608.21946.

Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, and Dahua Lin. 2026. CapRL: Stimulating dense image caption capabilities via reinforcement learning. In International Conference on Learning Representations.

Shenghao Ye, Yu Guo, Dong Jin, Yuxiang Wang, Yikai Shen, Yunpeng Hou, Shuangwu Chen, Jianyang, and Xiaofeng Jiang. 2026. When TableQA meets noise: A dual denoising framework for complex questions and large-scale tables. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24022– 24045.

Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. In International Conference on Learning Representations, volume 2025, pages 21074–21098.

Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, and Xianpei Han. 2025. Memsearcher: Training llms to reason, search and manage memory via end-to-end reinforcement learning. arXiv preprint arXiv:2511.02805.

Boer Zhang, Mingyan Wu, Dongzhuoran Zhou, Yuqicheng Zhu, Wendong Fan, Puzhen Zhang, Zifeng Ding, Guohao Li, and Yuan He. 2026. EigentSearch-Q+: Enhancing deep research agents with structured reasoning tools. In Proceedings ofthe ACM Conference on AI and Agentic Systems, pages 1114–1118.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS.

Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei, Zhenbo Luo, Jian Luan, Wei Chen, and Xiang Bai. 2026. Doc-v\*:coarse-to-fine interactive visual reasoning for multi-page document VQA. arXiv preprint arXiv:2604.13731.

Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang, Yuan He, Jiaoyan Chen, Steffen Staab, and Evgeny

Kharlamov. 2025a. Evaluating knowledge graph based retrieval augmented generation methods under knowledge incompleteness. arXiv preprint arXiv:2504.05163.

Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang, Hongkuan Zhou, Jiaoyan Chen, Steffen Staab, Yuan He, and Evgeny Kharlamov. 2025b. GR-Agent: Adaptive graph reasoning agent under incomplete knowledge. arXiv preprint arXiv:2512.14766.

Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang, Hongkuan Zhou, Yuan He, Jiaoyan Chen, Steffen Staab, and Evgeny Kharlamov. 2026a. What breaks knowledge graph based RAG? benchmarking and empirical insights into reasoning under incomplete knowledge. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2522–2538.

Yixiao Zhou, Dongzhou Cheng, Zhiliang Wu, Yi Yang, Yu Cheng, and Hehe Fan. 2026b. One refiner to unlock them all: Inference-time reasoning elicitation via reinforcement query refinement. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 38957–38978.

Yixiao Zhou, Yang Li, Dongzhou Cheng, Hehe Fan, and Yu Cheng. 2026c. Look inward to explore outward: Learning temperature policy from LLM internal states via hierarchical RL. arXiv preprint arXiv:2602.13035.

Yuqicheng Zhu, Daniel Hernández, Yuan He, Zifeng Ding, Bo Xiong, Evgeny Kharlamov, and Steffen Staab. 2025a. Predicate-conditional conformalized answer sets for knowledge graph embeddings. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 4145–4167.

Yuqicheng Zhu, Nico Potyka, Daniel Hernández, Yuan He, Zifeng Ding, Bo Xiong, Dongzhuoran Zhou, Evgeny Kharlamov, and Steffen Staab. 2025b. ArgRAG: Explainable retrieval augmented generation using quantitative bipolar argumentation. In Proceedings of the 19th International Conference on Neurosymbolic Learning and Reasoning, volume 284 of Proceedings of Machine Learning Research, pages 697–718. PMLR.

Yuqicheng Zhu, Nico Potyka, Mojtaba Nayyeri, Bo Xiong, Yunjie He, Evgeny Kharlamov, and Steffen Staab. 2024. Predictive multiplicity of knowledge graph embeddings in link prediction. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 334–354.

Yuqicheng Zhu, Nico Potyka, Jiarong Pan, Bo Xiong, Yunjie He, Evgeny Kharlamov, and Steffen Staab. 2025c. Conformalized answer set prediction for knowledge graph embedding. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational

Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 731–750.

Yuqicheng Zhu, Jialin Yu, Lin Li, Gengyuan Zhang, Zhen Yang, Steffen Staab, Puneet Dokania, Philip Torr, Jie Tang, and Evgeny Kharlamov. 2026. Conformalized large language models under configuration shift. arXiv preprint arXiv:2608.01460.

## A Dataset Details

Benchmark sources. MMLONGBENCH-DOC uses all 1,082 examples in the original samples.json. LONGDOCURL uses all 2,325 examples in the original LongDocURL\_public.jsonl. The main-results table uses the original full-size MMLONGBENCH-DOC page images.

EP-given evaluation. In the diagnostic experiments, EP-given replaces retrieval with the annotated gold evidence pages and is reported on fixed MMLONGBENCH-DOC subsets specified in the corresponding tables. We do not report a LONG-DOCURL EP-given diagnostic in the main experiments.

Routing verification. All reported runs pass a check that each question is paired with its source document.

## B Evaluation Protocol

Official benchmark scoring. All final-answer and memory-only metrics reported for our Qwen3- VL-4B runs on MMLONGBENCH-DOC and LONGDOCURL use the official judge J defined in Sec. 2. The judge follows the official evaluation pipeline for the corresponding benchmark. It first sends the model’s free-form output to GPT-4o at temperature 0. GPT-4o extracts a concise candidate answer in the benchmark’s expected format. A benchmark-specific deterministic rule scorer then compares the extracted candidate with the gold answer and assigns the official generalized-accuracy score.

Type-aware rule scoring. Both official scorers normalize the extracted candidate and gold answer to the benchmark’s answer types before comparison. Integer answers require an exact match. Floating-point answers allow a 1% relative tolerance and account for equivalent percentage scales. For string and None answers, the scorer uses exact matching for designated special strings and otherwise uses average normalized Levenshtein similarity (ANLS) with a 0.5 threshold. For list answers, the MMLONGBENCH-DOC scorer requires equal list lengths, sorts both lists, and uses the minimum element-level ANLS as the question score. The LONGDOCURL scorer instead matches list elements greedily by ANLS and applies a square-root penalty when the list lengths differ.

Final-answer and memory-only metrics. For final-answer accuracy, the candidate is the agent’s final answer. For memory-only accuracy, a frozen reader receives the question q and terminal working memory $M ^ { \mathrm { t e r m } }$ , but no trajectory, page images, or tool instructions. It then produces the candidate answer. The reader is the frozen Qwen3-14B model defined in Sec. 2. The same judge evaluates both candidates. Unless stated otherwise, reported accuracy is the mean official per-question generalized-accuracy score. The four-cell memory analysis requires binary outcomes, so it treats any positive official score as correct and a zero score as incorrect. The controlled intervention in Sec. 2 reports this binary positive-score rate so that each percentage matches its correct count and Wilson confidence interval.

Manual audit. We manually checked 100 scored examples against human correctness judgments. The audit found six disagreements, mainly involving answer aliases, list formatting, and partialcredit edge cases. The six disagreements were not concentrated on one side of the comparison.

Training-time reward. Online GRPO uses a frozen local Qwen3-14B model to generate the memory-only answer and compute the two reward scores. This training-time scorer is separate from the official judge used for reported metrics. Only the Qwen3-VL-4B agent policy is updated; the Qwen3-14B model remains fixed.

## C Design Choices and Reward Variants

This appendix expands on the design choices and reward analyses summarized in the main text.

## C.1 Why a conditional rate and not a marginal gap

For a set of N examples, let $n _ { 1 0 }$ and $n _ { 0 1 }$ denote the numbers of memory-missing-correct and answering-error examples, respectively. The marginal gap between final-answer and memoryonly accuracy is $( n _ { 1 0 } - n _ { 0 1 } ) / N$ . Because the two failures enter with opposite signs, they can cancel: the gap can be near zero even when many correct answers have unanswerable terminal working memory. It therefore does not directly answer the question that motivates AWM: among the final answers that the agent gets right, how often is the saved memory still insufficient?

The conditional rate $P _ { \mathrm { m m c } } = n _ { 1 0 } / N _ { \mathrm { c o r r e c t } }$ measures the fraction of binary-correct final answers with unanswerable terminal working memory. Answering-error examples enter neither its numerator nor its denominator, so they cannot cancel the memory failures of interest. Across datasets, subsets, and checkpoints, $P _ { \mathrm { m m c } }$ retains the same interpretation: the share of correct final answers that cannot be recovered from memory alone. We therefore report it rather than the signed marginal gap.

## C.2 Why mandatory commit does not collapse memory-only accuracy onto final-answer accuracy

Our agent loop makes a memory update mandatory after every retrieval. A natural worry is that this construction makes memory-only accuracy track final-answer accuracy because every retrieval is followed by a memory update. It does not. Each update is free-form within a schema that limits findings to 40 words and requires a source page. An agent can satisfy that schema with a generic page description, omit the fact needed by the question, and still answer from raw page context at the final step. The diagnostic in Sec. 2 tests this gap between writing an update and preserving what the later answer needs. Mandatory commit therefore isolates what the agent extracts at each update rather than when it chooses to update memory.

## C.3 Reward variants we considered

The minimal form in Sec. 3 pairs one answer from the frozen reader with one score for that answer. We also considered three richer variants.

Per-finding source-image grounding. A more granular signal asks a model to check, for each finding in $M ^ { \mathrm { t e r m } }$ , whether the cited source\_page image actually supports the claim. This is a stricter per-finding evidence-grounding signal and would provide a finer-grained training signal. We did not adopt it for the main experiments because (i) it requires one scoring call per finding rather than one per trajectory, raising compute and scoring noise at training time, and (ii) it conditions the reward on a fixed schema, sacrificing the schema-agnosticism that we view as a feature of our minimal form. Source-grounding audits using this variant are a natural follow-up evaluation, separate from the reward signal used during training.

Self-consistency over multiple reader rollouts. The binary memory score inherits the variance of a single reader rollout. Sampling K answers from $R ( q , M ^ { \mathrm { t e r m } } )$ and scoring their correct fraction would reduce variance but multiply scoring cost by K; we do not evaluate this variant.

Length penalty on $M ^ { \mathrm { t e r m } }$ . We considered a training-time penalty for $M ^ { \mathrm { t e r m } }$ entries that closely mirror the predicted final-answer string. We did not adopt it because such a penalty could suppress legitimate answer-supporting findings that contain the answer value (e.g., when the answer is a value on the page); answer-echo detection remains a separate audit.

## C.4 Computation overhead relative to standard GRPO

Training-time overhead. Relative to standard answer-only GRPO, AWM-GRPO adds two short text-only passes through a local Qwen3-14B model for each rollout. The first pass answers from terminal working memory alone, and the second scores that memory-only answer for the training reward. The answer-only baseline already scores the agent’s final answer. The Qwen3-14B model remains frozen, and only the Qwen3-VL-4B agent policy is trained. We report the added model passes instead of a wall-clock percentage because elapsed time depends on hardware, batching, and serving configuration.

Deployment-time overhead. The memory reader and training scorer are not called at deployment. AWM-GRPO therefore adds no deployment-time model call relative to the same agent policy and tool loop.

## C.5 Reward discrimination under AWM

This appendix gives a structural statement of how the AWM reward (Sec. 3) differs from an answeronly reward over the four outcome cases of Table 1. The statement is purely structural and assumes only that the scoring and reader interfaces are fixed within each pass. During online training, the frozen local Qwen3-14B model supplies the reader output and the two reward scores. For reported diagnostics, R is the frozen Qwen3-14B reader and J is the official judge from Sec. 2.

Let $s _ { \mathrm { a n s } } , s _ { \mathrm { m e m } } \in \{ 0 , 1 \}$ denote final-answer correctness and memory-only answerability for a trajectory τ, as in Sec. 3.2, and let $R _ { \mathrm { A W M } }$ be the piecewise AWM reward with values $( r _ { 1 1 } , r _ { 1 0 } , r _ { 0 1 } , r _ { 0 0 } )$ satisfying $r _ { 1 1 } > r _ { 1 0 } > r _ { 0 1 } > r _ { 0 0 }$ . The reward mapping $( s _ { \mathrm { a n s } } , s _ { \mathrm { m e m } } ) \mapsto R _ { \mathrm { A W M } }$ strictly orders the four outcome cells: memory-supported correct receives the highest value, memory-missing correct sits above answering error, and unresolved error receives the lowest value. Answer-only reward, by contrast, takes only two values and identifies $( s _ { \mathrm { a n s } } , s _ { \mathrm { m e m } } ) { = } ( 1 , 1 )$ with $( s _ { \mathrm { a n s } } , s _ { \mathrm { m e m } } ) { = } ( 1 , 0 )$ and identifies (0, 1) with (0, 0). The AWM reward therefore distinguishes pairs of policies that agree on final-answer accuracy but differ in memoryonly answerability; answer-only reward cannot. The default values $( 2 , 0 , - 0 . 1 , - 1 )$ also preserve final-answer priority because $r _ { 1 0 } > r _ { 0 1 }$ , so a finalanswer-correct trajectory with unanswerable memory is never outranked by a final-answer-wrong trajectory with answerable memory.

The same property has a direct GRPO consequence (Xie et al., 2026a,b; Zhou et al., 2026c,b). In a GRPO group of G trajectories sampled for one prompt, sampled trajectories that fall into different reward levels can produce non-zero grouprelative advantage after normalization. The key case is a group containing both memory-supported correct and memory-missing correct trajectories: answer-only reward assigns them the same value, while the AWM reward assigns values that differ by $r _ { 1 1 } - r _ { 1 0 }$ . By the diagnostic of Sec. 2, this is the cell pair where the pre-training agent already shows a measurable memory-preservation gap.

## C.6 Advantage-distribution simulation details

This appendix records the simulation behind Figure 3. For mixture m, let $\rho ^ { ( m ) }$ be a categorical distribution over the four cells ${ \boldsymbol z } = ( s _ { \mathrm { a n s } } , s _ { \mathrm { m e m } } )$ For each reward scheme and mixture, we sample $B = 1 5 { , } 0 0 0$ groups of size $G = 8$ . AWM samples $z _ { b , 1 : G } \sim$ Categorica $( \rho ^ { ( m ) } )$ ), while answer-only sampling uses the corresponding final-answer marginal. We assign either answer-only or AWM rewards, compute the group mean $\mu _ { b }$ and standard deviation $\sigma _ { b } .$ , and form $A _ { b , i } = ( r _ { b , i } - \mu _ { b } ) / ( \sigma _ { b } + \epsilon )$ with $\epsilon = 1 0 ^ { - 6 }$ . Figure 3 shows kernel-smoothed empirical advantage distributions grouped by final-answer correctness for answer-only reward and by outcome cell for AWM. Finite groups and discrete rewards produce a finite set of normalized advantage values; kernel smoothing affects only the plot, not reward or training computation. In (0, 0), (0, 1), (1, 0), (1, 1) order, the three mixtures are $( 0 . 8 2 , 0 . 0 5 , 0 . 0 5 , 0 . 0 8 ) , \quad ( 0 . 6 0 , 0 . 0 5 , 0 . 0 5 , 0 . 3 0 )$ and (0.30, 0.05, 0.05, 0.60). The off-diagonal cells (0, 1) and (1, 0) therefore retain probability 0.05 across rows, while diagonal mass shifts from (0, 0) toward (1, 1). The change in (1, 0) advantage therefore comes from the rising group mean rather than a change in its cell frequency.

## D Improved-Memory Intervention Details

This appendix expands the controlled memory comparison in Sec. 2. The intervention is a controlled sanity check rather than a deployed method or a full-benchmark result, and the improved memories are not used as training data. The intervention uses a fixed set of 500 answerable MMLONGBENCH-DOC questions in the evidence-page-given (EPgiven) setting. Here, answerable means that the benchmark annotations identify sufficient evidence for a reference answer. It does not mean that the agent’s terminal working memory preserves that evidence.

## D.1 Setup

EP-given evaluation provides the gold evidence pages, which removes page retrieval as a source of error. The empty condition supplies no terminal memory. The original condition uses memory written by the base Qwen3-VL-4B agent, while the improved condition uses memory that GPT-4o constructs from the same gold evidence pages. The same 500 questions are evaluated under all three terminal-memory conditions. The final-answer generator and judge are held fixed across conditions. For memory-only evaluation, the fixed reader is Qwen3-14B.

All reported outputs use the official judge defined in Sec. 2.

The original agent partitions the 500 examples into the four outcome cells from Sec. 2: 104 memory-supported correct, 77 memory-missing correct, 47 answering error, and 272 unresolved error. We fix this partition from the original-memory run, then compare the same questions under original and improved memory.

<table><tr><td>Memory condition</td><td>Correct / N</td><td>Rate (%)</td><td>95% Wilson CI (%)</td></tr><tr><td>Empty memory</td><td>14 / 500</td><td>2.8</td><td>[1.7, 4.6]</td></tr><tr><td>Original agent memory (4B)</td><td>181 / 500</td><td>36.2</td><td>[32.1, 40.5]</td></tr><tr><td>Improved memory (GPT-4o)</td><td>222 / 500</td><td>44.4</td><td>[40.1, 48.8]</td></tr></table>

Table 8: Final-answer positive-score rate in the controlled memory intervention. Intervals are 95% Wilson confidence intervals.
<table><tr><td>Original outcome</td><td>N</td><td>Original memory correct</td><td>Improved memory correct</td><td>Change (count; pp)</td></tr><tr><td>Memory-supported correct</td><td>104</td><td>104</td><td>88</td><td>-16; -15.4</td></tr><tr><td>Memory-missing correct</td><td>77</td><td>0</td><td>48</td><td>+48; +62.3</td></tr><tr><td>Answering error</td><td>47</td><td>47</td><td>29</td><td>-18; -38.3</td></tr><tr><td>Unresolved error</td><td>272</td><td>0</td><td>59</td><td>+59;+21.7</td></tr><tr><td>Total</td><td>500</td><td>151</td><td>224</td><td>+73; +14.6</td></tr></table>

Table 9: Memory-only correctness after grouping examples by the original agent’s outcome. Change is measured within each fixed row.

## D.2 Final-answer binary correctness

Under the shared evaluation procedure, finalanswer binary correctness is 44.4% with improved memory and 36.2% with original agent memory. With empty memory, 14 of 500 final answers receive a positive official score. These rates describe the controlled 500-example comparison and are not a full-benchmark accuracy claim.

## D.3 Memory-only results by original outcome

The original-memory column follows directly from the fixed outcome labels: memory is answerable in the memory-supported-correct and answeringerror cells, and unanswerable in the other two cells. Improved memory makes 48 of the 77 memory-missing-correct examples and 59 of the 272 unresolved-error examples answerable. The same replacement makes memory unanswerable for 16 originally memory-supported-correct examples and 18 answering-error examples. The net result is 73 more memory-answerable examples, an increase from 151 to 224 out of 500. The intervention therefore improves memory-only answerability overall but does not dominate the original memory in every outcome cell.

## D.4 Artifact audit

All 500 improved memories include a source\_page citation to a page in the corresponding gold evidence\_pages set. This audit verifies that every memory points to an eligible evidence page. It does not verify that every generated statement is entailed by that page or that the memory contains all evidence needed for the answer.

We also apply a strict answer-string filter. The filter removes every example whose improved memory contains the normalized gold answer string after whitespace and punctuation normalization. It leaves 304 examples, of which the memory-only reader receives a positive official score on 98, for a positive-score rate of 98 $/ 3 0 4 = 3 2 . 2 \%$ . This remaining rate shows that exact answer-string copies do not explain all successful memory-only answers. Because the filter selects a different subset rather than editing memory in place, its 32.2% rate is not a like-for-like estimate of the full-set effect. The filter also cannot rule out paraphrased answer leakage or establish claim-level grounding.

## D.5 Answerability versus grounding

Memory-only answerability asks whether the fixed reader can recover an accepted answer from the question and terminal working memory alone. Grounding asks whether the memory’s claims are supported by the cited source pages. These properties are related but not equivalent. An answerable memory may contain an unsupported guess or an answer echo. A grounded memory may still omit a needed fact, or the reader may fail to use it. The audits check only source-page validity and exact answer copying; they do not establish claim-level grounding. Accordingly, the intervention shows that changing memory content can improve answering under a fixed protocol; it does not show that every improved memory is fully grounded.

## D.6 Limitations

This single-seed intervention is an upper-bound analysis on one fixed subset, reader, judge, and improved-memory model: GPT-4o is stronger than the original 4B memory writer and receives gold evidence pages. Table 9 shows that improved memory also hurts some outcome cells, and the improved memories are used neither for training nor in the main AWM diagnostic. Repeated runs and claimlevel audits are needed.

## E Relation to Other Reliability Criteria

AWM treats terminal working memory as an instance-level artifact. Predictive multiplicity in knowledge-graph embeddings and disagreement across repeated language-model queries expose cases where aggregate performance hides querylevel variation (Zhu et al., 2024; Potyka et al., 2024); AWM instead tests whether a fixed reader can recover the answer after the pages and interaction trace are removed. Conformal methods instead construct prediction sets or study coverage under predicate and configuration shift (Zhu et al., 2025c,a, 2026). We make no uncertainty or coverage claim; the output is a per-trajectory memoryonly answerability test.

Other studies modify or inspect the evidence-toreasoning path. EigentSearch-Q+ uses structured tools and SCAIR uses schema-conditioned traversal (Zhang et al., 2026; Chaturvedi et al., 2026). Knowledge-graph RAG studies test missing facts, while ArgRAG represents retrieved evidence as supporting or attacking arguments (Zhou et al., 2025a, 2026a; Zhu et al., 2025b). Related analysis also finds asymmetric responses for logically equivalent facts under different pretraining frequencies (He et al., 2025). AWM leaves the agent architecture and memory schema unchanged and evaluates the source-linked memory they produce: the memory passes only when the reader can answer without the pages or the trajectory.