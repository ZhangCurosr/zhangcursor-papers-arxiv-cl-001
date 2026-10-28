![](images/c69b1a1e7fc351406b75d5f8d335f4ad1edfa937d4300149a503fe886ddf1261.jpg)

# Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions

Anouar Imel<sup>1</sup>, Imad Charradi<sup>2</sup>, Abderrahman Sghir<sup>3</sup>, Khalil El Khammar<sup>4</sup>

<sup>1</sup>School of Information and Communication Engineering, University of Electronic Science and Technology of China (UESTC)

<sup>2</sup>Xidian University

<sup>3</sup>Nanjing University of Aeronautics and Astronautics

<sup>4</sup>School of Artificial Intelligence, Nanjing University of Information Science and Technology

202524010105@std.uestc.edu.cn, charradiimad@stu.xidian.edu.cn

abde44@nuaa.edu.cn, khalil.elkhammar@nuist.edu.cn

## Abstract

When an LLM group’s discussion is compressed into a short memory for a final decision maker, what should the memory keep? Two natural heuristics keep the rarest claims or the claims relevant to the decision criteria. On hidden-profile tasks both look attractive, because how widely a fact is held is confounded with whether it helps. We introduce CrossProfile, a diagnostic benchmark of 60 templates with automatically verified answer keys, in which the ownership of supporting, decoy, and irrelevant facts is crossed while their text is held fixed (480 tasks). Across Qwen2.5-7B, Llama-3.1-8B, and two DeepSeek models, both heuristics fail in predictable places. Selecting by relevance readmits decoys, because decoys are on topic: even a perfect relevance filter loses 23 to 31 points to rarest-first selection in the classic hidden-profile configuration. Rarest-first selection is no better than random selection on average and wins only where the rare facts happen to be the useful ones. In both cases accuracy follows the balance of evidence the memory keeps, not the policy that kept it. The decision model adds a third failure, a form of social proof: it gives extra weight to facts attributed to many agents. Removing the attribution tags, or reassigning them at random with line order fixed, raises accuracy in the classic configuration by 13 to 24 points. In natural discussions the evidence is lost before any memory acts: the complete transcript yields lower accuracy than no notes at all.

## 1 Introduction

An agent team may hold all the evidence needed for a decision and still leave its final reader with the wrong information. Some evidence is never stated. Other evidence is stated and then lost when a memory policy compresses the discussion. Even a faithful memory can emphasize claims that the reader weighs poorly. An end-task accuracy gain alone cannot tell these failure points apart.

Hidden-profile tasks make the distinction concrete: participants share some facts and individually hold other facts needed for a collective decision (Stasser and Titus 1985; Gigone and Hastie 1993; Li, Naito, and Shirado 2026). A natural response to repeated shared information is to preserve what few participants said. We study this response through rarest-first selection (R1), which ranks claims by the number of agents who stated them. The rule assumes that rare evidence is worth keeping.

Faithfulness alone does not settle which evidence to keep.

In one procurement template, a supplier has passed every electrical safety inspection but took nine days to answer urgent repair callouts. Both statements concern the decision criteria. Keeping the favorable fact and dropping the unfavorable one can change the decision without introducing any false claim. What matters is the balance of the retained evidence, not only whether a summary is accurate.

Figure 1: Rarity and decision usefulness. Fact text stays fixed while the ownership of support and decoy facts changes; here irrelevant facts stay private. Filled dots mark the agents holding each fact, and the boxes on the right show what rarestfirst selection prioritizes. (c) The three ownership factors are crossed. Decoys are true, criterion-relevant facts that favor a competing option. Role labels are never shown to the models.

Figure 1 shows why the rarity assumption needs testing. In a classic hidden profile, private evidence supports the correct answer and shared evidence favors a decoy. Prioritizing rare facts can then help by keeping support, by excluding decoys, or both. If private facts instead favor the decoy, the same rule can damage the decision. On HiddenBench the two explanations cannot be separated: a memory holding only the private facts reaches 91% accuracy on Qwen, whereas one holding only the shared facts reaches 6%, well below an empty memory. A large gain for rarest-first selection there therefore says little about rarity itself. The obvious alternative, keeping the facts relevant to the decision criteria, faces the opposite problem, because a decoy is relevant by construction. Testing either rule requires a design in which fact prevalence and decision usefulness vary independently.

We construct CrossProfile, a crossed benchmark in which support, decoy, and irrelevant facts are each either shared or private. Within a template, the fact sentences and their decision roles are fixed across all eight ownership variants, and the answer key follows from the fact roles and is verified automatically. Scripted discussions state every assigned fact, so memory selection can be studied against a known inventory. CrossProfile is a diagnostic benchmark: it isolates what a memory policy and a decision model do with a known set of evidence. Changing ownership also changes when a fact first appears, the holder tags attached to it, and its token cost, and we add a control for each of these.

Both heuristics fail in cells that the design predicts, and aggregate accuracy hides both failures. Relevance-first selection helps on average but loses 20 to 32 points in the classic hidden-profile cell, and a perfect relevance filter loses as much, because relevance does not distinguish support from decoys. Rarest-first selection is no better than random selection on average; its advantage over random selection changes by 28 to 41 points depending on who holds the decoys, on all four models. Across policies, accuracy rises steadily with the net evidence lead of the correct option among the kept facts, and once that lead is known the policy adds little. The decision model contributes a separate failure, a form of social proof: it gives extra weight to a fact attributed to many agents, although in our tasks shared facts were assigned, not independently observed. Removing the attribution tags, or reassigning them at random while keeping line order, raises its all-facts accuracy in the classic cell by 13 to 24 points. In natural discussions no memory policy beats another and the full transcript is less useful than an empty memory, so the evidence is lost before any memory acts.

Our contributions are (i) CrossProfile, a diagnostic benchmark that varies the ownership of support, decoy, and irrelevant facts while holding their content fixed; (ii) evidence on two open and two hosted models, with random-reference, fixed-item-count, and oracle-relevance controls, that relevance and rarity both fail in predictable ownership cells because neither tracks the balance of evidence; (iii) evidence that the decision model follows social proof in the form of holder tags, isolated from line order; and (iv) naturaldiscussion evidence that the main bottleneck lies upstream of memory.

## 2 Related Work

Hidden-profile research documents shared-information bias and the dificulty of pooling distributed evidence (Stasser and Titus 1985; Gigone and Hastie 1993; Lu, Yuan, and McLeod 2012); in the common-knowledge efect, information held by more members carries more weight in the group judgment (Gigone and Hastie 1993), and our decision model shows a similar weighting from holder tags alone. LLM agents in opinion simulations can also violate basic probabilistic consistency, assigning probabilities to a claim and to its negation that do not add up to one (Imel and Qi 2026). HiddenBench adapts this setting to LLM groups and examines forced disclosure, passive summarization, and structured communication (Li, Naito, and Shirado 2026). With human groups, an LLM facilitator increases how much information members share without improving the final decision (Alsobay et al. 2025), which parallels our finding that disclosure and use of evidence are separate bottlenecks. Our intervention holds fact content fixed while crossing the ownership of decision roles, and studies what survives a memory constraint applied after the discussion.

Multi-agent debate and agent frameworks study reasoning through interaction (Du et al. 2024; Smit et al. 2024; Wu et al. 2024), and collective biases in LLM groups can change non-linearly with group size (Flint et al. 2026), which our fixed four-agent design does not vary. Consilience intervenes during hidden-profile communication with a calibrated controller that tracks evidence and redundancy (Babu et al. 2026). Communication interventions change what the group discloses; memory selection, which we study, determines which disclosed claims reach the final reader.

Memory architectures maintain information across interactions (Packer et al. 2023; Park et al. 2023), and prompt compression and selective context address limited capacity (Jiang et al. 2023; Li et al. 2023). Benchmarks of conversational memory test long-horizon recall (Maharana et al. 2024), and recent multi-party benchmarks show that memory pipelines lose speaker and group structure (Yang et al. 2026; Hu et al. 2026); CrossProfile instead asks which evidence a compressed memory should keep for a downstream decision. Interaction-aware selection argues that records can matter only in combination (Subrahmanyam and Choudhary 2026), which is consistent with our finding that decisions follow the balance of the kept evidence rather than the merit of individual facts. Position efects further separate available evidence from evidence the model uses (Liu et al. 2024); our tag controls hold line order fixed to separate attribution from position.

## 3 Setting and Memory Policies

## 3.1 Discussion, Memory, and Decision

A task specifies a description, candidate answers, and facts assigned to agents. A shared fact is held by every agent; a private fact is held by one. After the discussion, a memory policy constructs a memory for a separate decision call, which receives only the description, the options, and the memory. The decision call uses the same model weights as the discussion and sees the full transcript only in the REF control.

![](images/53d42f97217b5f8c87f6cb7c81ae77a173c654739789810334d6eca128543962.jpg)  
Figure 2: Pipeline. Facts from template M01 (Table 4) are distributed among the agents; HiddenBench tasks have three or four agents and CrossProfile tasks have four. After the discussion, a memory policy compresses it under a token budget B, either by summarizing the transcript (B3) or by selecting whole claims from a claim inventory (R1, R1-rel, B4, Random). A separate cal with the same model weights answers from the task and the memory alone.

Answer keys and role labels are withheld from all generation and from every memory policy except the oracle-relevance control.

In controlled HiddenBench discussions, each agent is scripted to state its assigned facts. Natural discussions consist of initial individual votes, two rounds of messages, and final votes; memories are built from the messages, not the votes. Copy policies select from a known claim inventory in the controlled condition and from an extracted inventory in the natural condition. In extracted inventories, a claim’s contributors are the agents who stated it, which need not coincide with the agents who originally held the underlying fact. The summary policy B3 reads the transcript directly. Figure 2 summarizes these paths.

## 3.2 The CrossProfile Benchmark

CrossProfile has 60 main templates, five in each of 12 decision domains. Each template has four agents, three options, and 12 facts: five support, three decoy, and four irrelevant. Support facts favor the correct option, including negative evidence about the decoy option. Decoy facts favor the competing option, including negative evidence about the correct option. Both concern the stated decision criteria; decoys are true statements whose partial implication is misleading. Irrelevant facts concern other attributes and may be positive or negative in tone. Each role group is independently shared or private, yielding 23 ownership variants and 480 derived tasks. Ten pilot templates are excluded from all main estimates.

The answer key follows from the structured facts: each favorable fact adds one to its option’s score, each unfavorable fact subtracts one, and irrelevant facts contribute zero. An automatic verifier checks every template for a unique best option over all facts, a decoy win when only decoy facts are counted, a correct-option win when only support facts are counted, and the absence of hint words; all 70 templates pass. With the fixed fact counts, the keyed option scores +2 over all facts and each competitor scores 0. The task description asks for the best overall record on equally weighted criteria, but the decision prompt does not state the scoring rule. Accuracy therefore measures agreement with this constructed key.

The controlled transcript randomizes the speaking order and the order of each speaker’s facts. Every shared fact appears in the first speaker’s turn, although not necessarily before every private fact in that turn, so first-mention selection is sensitive to prevalence. Copied memory lines keep their contributor tags and follow the policy’s selection order. Ownership thus changes how evidence is presented as well as its selection priority.

## 3.3 Memory Policies and Budgets

R1 orders claims by increasing contributor count, breaking ties by first appearance. B4 orders claims by first appearance alone. Random orders the same inventory uniformly at random. These copy policies add whole claims in this order until the budget is reached, skipping any claim that would exceed it.

B3, the evidence-preserving summary, asks the model to keep each distinct piece of evidence with its conditions, uncertainty, and disagreements. It receives a word limit derived from the budget, and longer outputs are truncated to the budget. R1-rel first asks the model which fact texts concern the stated criteria. This filter sees the description, options, and fact texts, but no holders, answers, or role labels. Claims predicted relevant form the first tier and the remaining claims the second, with R1 order inside each tier; second-tier claims remain eligible. Since decoys concern the criteria, they are relevant by this definition.

For model tokenizer $\tau _ { m } ,$ , the CrossProfile budget is

$$
B _ { t m } ( r ) = \left\lfloor r \left| \tau _ { m } ( F _ { t } ^ { \mathrm { a l l } } ) \right| \right\rfloor , \qquad r \in \{ . 4 , . 6 , . 8 \} ,\tag{1}
$$

where $F _ { t } ^ { \mathrm { a l l } }$ is the text of every fact of template t written with all four holder tags. The budget is therefore fixed across ownership variants and seeds. Selected lines carry their actua tags, so private facts cost fewer tokens and the same budget can hold diferent numbers of items. HiddenBench budgets are computed in the same way from the benchmark’s fact set. The unbudgeted empty, all-facts, and full-transcript (REF) memories serve as controls. Appendix A.5 additionally reports speaker-cycling selection (M1) on HiddenBench.

## 3.4 Outcomes and Primary Contrasts

Accuracy is averaged over seeds within each task or ownership cell, and then over tasks or templates. Invalid answers count as wrong. For CrossProfile template t, let $a _ { t } ( \pi \mid C )$ be the accuracy of policy π averaged over seeds and over the variants satisfying ownership condition C. With S, D, I denoting support, decoy, and irrelevant ownership, the primary contrasts at $r = . 6$ are

$$
\begin{array} { l } { h _ { 1 t } = [ a _ { t } ( R 1 \mid D _ { s } ) - a _ { t } ( B 4 \mid D _ { s } ) ] } \\ { - [ a _ { t } ( R 1 \mid D _ { p } ) - a _ { t } ( B 4 \mid D _ { p } ) ] , } \\ { h _ { 2 t } = a _ { t } ( R 1 \mid I _ { s } ) - a _ { t } ( R 1 \mid I _ { p } ) , } \\ { h _ { 3 t } = a _ { t } ( R 1 - r e l \mid I _ { p } ) - a _ { t } ( R 1 \mid I _ { p } ) , } \end{array}\tag{2}
$$

where subscripts s and p denote shared and private, and the remaining ownership factors are averaged within template. H1 tests whether the R1 advantage over B4 depends on decoy ownership, H2 tests a penalty when irrelevant facts are private, and H3 tests relevance-first priority when irrelevant facts are private.

For copy memories, retention is the fraction of each fact group that the memory contains, averaged within task and then over seeds. The retention gap is shared-fact retention minus private-fact retention, in percentage points. On HiddenBench, a fact counts as retained if the memory contains it verbatim after normalization; in CrossProfile, the selected facts are known exactly. Memory length is measured in tokens with the model’s tokenizer.

## 4 Experimental Setup

We use Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct (Qwen Team 2024; Llama Team, AI @ Meta 2024), served locally with vLLM (Kwon et al. 2023) in fp16 with an 8,192- token context and seeds 17, 42, and 91. Decisions, summaries, and claim extraction use temperature zero. Natural discussion messages use temperature 0.7, top-p 0.9, and at most 256 output tokens; decisions allow 400 output tokens. HiddenBench contributes 65 tasks, 58 with four agents and 7 with three. We also evaluate two hosted models through the DeepSeek API at temperature zero with thinking disabled: DeepSeek-V4.1-Flash on all 60 CrossProfile templates and on natural HiddenBench discussions, and DeepSeek-V4-Pro on 30 templates. Appendices A.1 and A.7 give all prompts and further implementation details.

<table><tr><td>Policy</td><td>Acc.</td><td>Private</td><td>Shared</td><td>Gap</td><td>Tokens</td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>B3</td><td>51.3</td><td></td><td></td><td></td><td>108</td></tr><tr><td>B4</td><td>22.1</td><td>26.3</td><td>85.8</td><td>59.5</td><td>136</td></tr><tr><td>R1</td><td>88.2</td><td>98.3</td><td>12.0</td><td>-86.3</td><td>139</td></tr><tr><td>Random</td><td>42.1</td><td>52.2</td><td>58.4</td><td>6.2</td><td>136</td></tr><tr><td colspan="6">Llama-3.1-8B-Instruct</td></tr><tr><td>B3</td><td>59.0</td><td></td><td></td><td></td><td>126</td></tr><tr><td>B4</td><td>25.6</td><td>26.3</td><td>85.8</td><td>59.5</td><td>135</td></tr><tr><td>R1</td><td>87.7</td><td>98.3</td><td>11.5</td><td>-86.8</td><td>137</td></tr><tr><td>Random</td><td>43.1</td><td>53.9</td><td>56.7</td><td>2.8</td><td>133</td></tr></table>

Table 1: Controlled HiddenBench at $r = . 6$ (65 tasks, three seeds): decision accuracy (%), exact-copy retention of private and shared facts (%), retention gap (shared minus private, points), and median memory length (tokens).

Both the HiddenBench and the CrossProfile experiments followed analysis plans specified before the main runs. In CrossProfile, $r = . 6$ is the primary ratio and H1–H3 are the primary tests; other ratios, the reference-memory, oraclerelevance, item-count, and holder-tag controls, the hostedmodel runs, and the prompt ablations are exploratory. A headroom check on the 65 HiddenBench tasks compares allfacts with empty memory, using a 4,096-token context. Qwen gains +36.4 points [25.1, 47.2], clearing the pre-specified 15-point threshold, whereas Llama’s +23.6 [11.3, 35.9] does not clear it at the lower bound. Llama results on the original HiddenBench study are therefore treated as exploratory.

Inference. We report 95% percentile bootstrap intervals from 10,000 resamples. HiddenBench contrasts average seeds and ratios within task before resampling the 65 tasks; CrossProfile contrasts resample the 60 templates, keeping each template’s variants and seeds together. For a paired diference, the two-sided p value is $\hat { p } ^ { \mathrm { ~ ~ } } =$ min{1, 2 min $[ \mathrm { P r } ^ { * } ( \bar { d } ^ { * } \leq 0 ) , \mathrm { P r } ^ { * } ( \bar { d } ^ { * } \geq \bar { 0 } ) ] \}$ , reported as $p ~ < ~ . 0 0 1$ when no resample crosses zero. Holm correction (Holm 1979) is applied to H1–H3 within each model at the primary ratio; intervals are not adjusted. The sharedirrelevant counterpart of H3 and all post-hoc controls lie outside this family. Appendix A.5 lists the HiddenBench test families.

## 5 Results

## 5.1 HiddenBench Confounds Rarity with Usefulness

On controlled HiddenBench, R1 is far ahead of the other memories (Table 1). At r = .6 it reaches 88.2% on Qwen and

![](images/ab5f21369663b4b8fedb818e5e69c5d293093f45db59aa680a74d7ebd5866578.jpg)  
Figure 3: Who wins depends on who holds what. CrossProfile accuracy at $r = . 6$ by support and decoy ownership (each point averages the two irrelevant-fact assignments, 60 templates, three seeds). R1 and B4 are mirror images; R1-rel varies little across cells but falls well below R1 in the classic hidden-profile column (shaded: private support, shared decoys), where even the all-facts memory (dashed) drops. Per-cell intervals for Qwen and Llama are in Appendix A.2.

<table><tr><td>Memory</td><td>Qwen</td><td>Llama</td></tr><tr><td>Evidence summary (B3)</td><td>72.0</td><td>58.7</td></tr><tr><td>First mention (B4)</td><td>61.0</td><td>61.5</td></tr><tr><td>Random</td><td>64.5</td><td>63.4</td></tr><tr><td>Rarest first (R1)</td><td>64.0</td><td>63.0</td></tr><tr><td>Relevance first (R1-rel)</td><td>72.6</td><td>74.6</td></tr><tr><td>All facts (unbudgeted)</td><td>73.1</td><td>72.6</td></tr><tr><td>Empty</td><td>42.2</td><td>30.6</td></tr></table>

Table 2: CrossProfile accuracy (%) at $r = . 6 ,$ averaged over 60 templates, eight ownership variants, and three seeds.

87.7% on Llama, against 51.3% and 59.0% for the evidence summary B3; averaged over budgets, R1 minus B3 is +29.2 points [22.9, 35.4] and +26.7 [20.2, 33.2], and rebuilding R1 at each summary’s realized length leaves +25.6 and +25.8 (Appendix Table 16). Table 1 also shows what R1 selects: nearly all private facts and few shared ones. Two reference memories show why this pays of. A memory of only the private facts reaches 91.3% on Qwen and 90.3% on Llama, above the all-facts memory by +21.5 [13.3, 30.3] and +28.7 [17.9, 39.0] points, whereas a memory of only the shared facts reaches 5.6% and 4.6%, far below an empty memory in the same run (35.4% and 36.4%). In HiddenBench the shared facts point to a wrong option, so dropping them is what helps, and no task has rare facts that mislead. CrossProfile adds such tasks.

## 5.2 Rarity Is a Bet on Who Holds the Decoys

Averaged over all ownership variants of CrossProfile, R1 is no better than Random: 64.0% against 64.5% on Qwen and 63.0% against 63.4% on Llama (Table 2). This average hides a reversal across cells. When support is private and decoys are shared, R1 reaches 90 to 97% while first-mention selection (B4) falls to 11 to 31%; when support is shared and decoys are private, the order reverses, with B4 at 92 to 96% and R1 at 18 to 40% (Figure 3).

The pre-specified H1 measures this dependence of the R1 advantage over B4 on decoy ownership: +64.6 points [60.3, 69.0] for Qwen and +62.6 [56.8, 68.2] for Llama, both Holm $p < . 0 0 1$ (Appendix Figure 6). Because every shared claim appears in the first turn, B4 keeps shared facts first and is close to a mirror image of R1, so part of H1 is built into the reference. Against Random, the interaction is +40.7 [34.6, 47.1] and +34.2 [27.8, 41.0]; when both policies retain exactly seven facts, H1 is +66.8 [61.9, 71.7] and +72.6 [67.2, 78.1]. The hosted models show the same pattern. On DeepSeek-V4.1-Flash, H1 is +62.6 [56.7, 68.6] and the interaction against Random is +28.3 [22.1, 34.7]; on a 30-template run of DeepSeek-V4-Pro the latter is +30.6 [22.5, 38.6]. On that common subset, R1 minus Random averaged over variants is −1.7, +1.7, −1.3, and −4.7 points on Qwen, Llama, DeepSeek-V4.1-Flash, and DeepSeek-V4- Pro.

Private irrelevant facts cost R1 little. H2 is +5.0 points [1.2, 8.9] for Qwen (Holm $p = . 0 1 1 )$ and +4.9 [−0.3, 10.0] for Llama $( p = . 0 6 0 )$ . H2 varies with the budget, and H1 shrinks at $r \ = \ . 8 ,$ when the memory can hold most facts (Appendix A.2).

## 5.3 Relevant Is Not Useful

H3 supports relevance-first priority on average: +8.5 points [3.5, 13.6] for Qwen (Holm p = .002), +15.6 [11.1, 19.9] for Llama $( p < . 0 0 1 )$ , and +16.7 [12.8, 20.7] on DeepSeek-V4.1-Flash, and R1-rel has the highest mean among budgeted policies (Table 2). We expected no benefit when irrelevant facts are shared, but the gains are +8.7 [4.6, 12.8] and +7.6 [4.2, 11.2]: when all facts are shared, rarity cannot separate irrelevant facts from criterion-related ones, and the relevance tier does.

The average combines gains with a large loss. In the classic cell (private support, shared decoys, private irrelevant facts), R1 reaches 96.7% [93.3, 99.4] on Qwen and 97.2% [93.9, 99.4] on Llama, and R1-rel changes accuracy by −22.2 points [−28.9, −16.1] and −20.0 [−26.1, −13.9] (Figure 4). The relevance tier admits about two shared decoys while retaining slightly less support; because shared decoys carry four holder tags, R1-rel also keeps fewer facts in this cell (8.3 to 8.5 against 9.0 for R1).

![](images/57edfb4838f81cd0444392fe14b72921f30ac636c96f3b07a07131a2cbdb9ce3.jpg)

![](images/8c8875cda77576699de553bd848c39ec519b38b7876a2cc796f9c8f0bb80a116.jpg)  
Figure 4: Criterion relevance is insuficient in the classic hidden-profile cell (private support, shared decoys, private irrelevant facts, r = .6). (a) Accuracy diferences from R1 for R1-rel and for oracle relevance on Qwen, Llama, and DeepSeek-V4.1-Flash (Flash), with 95% template-bootstrap intervals. (b) Mean number of selected facts by role on Qwen and Llama: relevance-first priority admits shared decoys.

![](images/4fdfe20ae8c32f27b011cec9ea799597684049825e58fd8224870f8e77176b02.jpg)  
Figure 5: Holder tags mislead the decision model. Accuracy in the classic cell (private support, shared decoys, private irrelevant facts) for the all-facts memory with and without holder tags, and for R1 at r = .6 (60 templates, three seeds; Flash: DeepSeek-V4.1-Flash).

The failure persists when relevance is known exactly. With evaluator labels that mark both support and decoy facts as relevant, oracle relevance minus R1 is −23.3 [−30.6, −16.1] for Qwen and −31.1 [−38.9, −23.9] for Llama in this cell, so filter errors cannot explain the loss. An oracle that marked only support facts would no longer select by relevance; it would select by the answer key. The same holds on DeepSeek-V4.1- Flash, whose filter is nearly perfect (relevant-class precision 99.4% and recall 98.9%, against recall 66.0% on Qwen and 85.6% on Llama), yet in the same cell R1-rel minus R1 is −32.2 [−40.6, −24.4] and oracle relevance minus R1 is −29.4 [−37.8, −21.7]. Relevance separates the decision criteria from other attributes, whereas usefulness depends on the balance of evidence between competing options, which a relevance filter does not see.

The kept evidence explains these results across policies. Pooling the four copy policies, accuracy rises monotonically with the net lead of the correct option among the kept facts, counted as in the answer key: from 13.5% and 10.3% at a lead of −2 to 76.0% and 75.6% at +2 and 96.2% and 96.1% at +4 or more on Qwen and Llama. In a template random-intercept logistic model with this lead and the tone ofthe kept irrelevant facts, the policy terms (R1-rel, B4, and Random against R1) are no longer distinguishable from zero. Wrong answers are mostly the decoy: in the classic cell, B4 picks it in 71.1% and 63.3% of decisions. The decision model therefore follows the keyed evidence closely, and memory policies difer mainly in which balance of evidence they hand it.

## 5.4 The Decision Model Follows Social Proof

The all-facts memory is not a neutral reference. It contains every fact and gives the correct option a two-point lead in every variant, so a reader that ignored ownership would be equally accurate across cells. Instead, accuracy is 21.0 points [16.5, 25.6] higher on Qwen and 13.8 [9.7, 17.9] higher on Llama when support facts are shared rather than private. In the classic cell the all-facts memory reaches only 54.4% and 55.6%, about 40 points below the R1 memory built from the same facts under a 60% budget (Figure 5).

Each memory line begins with the tags of the agents who stated it, and much of this preference follows the tags (Figure 5). Without tags, the shared-support advantage of the allfacts memory falls from +21.0 to +7.2 points on Qwen and from +13.8 to +7.6 on Llama. Removing them raises classiccell accuracy by +14.4 points [7.8, 21.1] on Qwen, +13.3 [6.7, 20.0] on Llama, and +24.4 [17.2, 32.2] on DeepSeek-V4.1-Flash, where it goes from 72.8% to 97.2%. The tags, not the line order, carry this efect: both conditions keep the transcript-derived order, and reassigning the same tag sets to facts at random, again with order fixed, also raises classiccell accuracy, by +12.8 points on both open models. The efect is not simply an error. Where support is shared and decoys are private, removing the tags lowers Qwen’s accuracy by −10.6 points [−16.7, −4.4]: the model treats a fact stated by many agents as more credible, which helps when the widely held facts are the useful ones and misleads when they are not. In CrossProfile the shared facts were assigned to every agent rather than observed independently, so the tags carry no extra evidence. On a common 30-template subset the tagged gap is +17.8 on Qwen, +12.5 on Llama, +6.1 on DeepSeek-V4.1-Flash, and +7.2 on DeepSeek-V4-Pro. This attribution efect is an analogue of the common-knowledge efect that arises after the discussion rather than during it, and it accounts for part of R1’s advantage in the classic cell: R1 drops the shared facts that the decision model overweights.

<table><tr><td>Model</td><td>Vote</td><td>REF</td><td>Empty</td><td>All</td><td>Discl.</td></tr><tr><td>Qwen</td><td>23.6</td><td>22.1</td><td>33.8</td><td>70.3</td><td>0.0†</td></tr><tr><td>Llama</td><td>31.8</td><td>33.8</td><td>36.9</td><td>61.5</td><td>2.4†</td></tr><tr><td>Flash</td><td>24.6</td><td>25.6</td><td>35.9</td><td>89.7</td><td>1.7</td></tr></table>

Table 3: Natural HiddenBench discussions (65 tasks, 195 discussions per model). Vote: accuracy of the group’s majority vote after two rounds. REF, Empty, All: decision accuracy from the full transcript, an empty memory, and all facts (%). Discl.: private facts stated in full in some message (%, stringmatching heuristic). †Disclosure for Qwen and Llama from one seed (65 discussions). Flash: DeepSeek-V4.1-Flash.

## 5.5 Natural Discussions Lose the Evidence Upstream

In natural discussions no memory policy is detectably better than another; R1 minus B3 is −1.4 points [−5.0, 2.4] on Qwen and −3.1 [−9.2, 2.7] on Llama (Appendix Table 16). The reason is visible before any memory is built (Table 3). The full transcript (REF) scores below an empty memory on all three models, and the group’s own final vote is correct in only 24 to 32% of discussions, although the same models reach 62 to 90% when given every fact. In DeepSeek-V4.1- Flash discussions, a string-matching heuristic finds private facts stated in full in only 1.7% [0.6, 2.9] of cases. Facts that were never stated cannot be recovered by any memory policy.

A prompt change helps one model only. Asking agents to state their evidence before recommending raises Llama’s REF accuracy by +30.8 points [21.5, 39.5], and by +25.1 [15.9, 34.4] in a paired ablation with fresh seeds, with no detectable change for Qwen (Appendix A.4). When the discussion carries more evidence, the memory policy matters again: under the evidence-first prompt, Llama’s R1 memory reaches 57.1% against 22.9% for B4 and 53.2% for B3, whereas in the original discussions the three were within three points of each other (29.2%, 29.4%, 32.3%; means over budgets, Appendix A.4).

## 6 Conclusion

Neither rarity nor relevance is a safe default for the memory of an LLM group. In CrossProfile, rarity helps only in the ownership cells where rare facts support the answer, relevance readmits on-topic decoys even with a perfect filter, and the decision model follows social proof, giving extra weight to facts that many agents are tagged as stating. What drives the decision is the balance of evidence between options that the memory leaves, and how the reader weighs its attribution. Memory designs for agent groups should therefore aim at that balance rather than at prevalence or topic, and should not present how many agents stated a fact as evidence for it when the sources are not independent. Evaluations should vary who holds which evidence, and should check whether that evidence reached the discussion at all.

## References

Alsobay, M.; Rothschild, D. M.; Hofman, J. M.; and Goldstein, D. G. 2025. Bringing Everyone to the Table: An Experimental Study of LLM-Facilitated Group Decision Making. arXiv:2508.08242.

Babu, A.; Kaur, R.; Pramanik, V.; Kotevska, O.; Bastian, N. D.; Jha, S.; Raj, S.; Wu, Y.; Jha, S. K.; and Roy, A. 2026. Consilience: Conformally Calibrated Communication Control for Hidden-Profile Multi-Agent Reasoning. arXiv:2608.20564.

Du, Y.; Li, S.; Torralba, A.; Tenenbaum, J. B.; and Mordatch, I. 2024. Improving Factuality and Reasoning in Language Models through Multiagent Debate. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, 11733–11763.

Flint, A.; Aiello, L. M.; Pastor-Satorras, R.; and Baronchelli, A. 2026. Group Size Efects and Collective Misalignment in LLM Multi-Agent Systems. Proceedings of the National Academy of Sciences. doi:10.1073/pnas.2531697123.

Gigone, D.; and Hastie, R. 1993. The Common Knowledge Efect: Information Sharing and Group Judgment. Journal ofPersonality and Social Psychology, 65(5): 959–974.

Holm, S. 1979. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2): 65–70.

Hu, C.; Li, T.; Gao, X.; Chen, H.; Bai, Y.; Xu, D.; Lin, T.; Li, X.; Han, Y.; Pei, J.; and Deng, Y. 2026. Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues. arXiv:2602.01313.

Imel, A.; and Qi, J. 2026. Do “True” and “False” Add Up to One? Complement Consistency in LLM-Agent Opinion Simulations. arXiv:2610.32874.

Jiang, H.; Wu, Q.; Lin, C.-Y.; Yang, Y.; and Qiu, L. 2023. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 13358–13376.

Kwon, W.; Li, Z.; Zhuang, S.; Sheng, Y.; Zheng, L.; Yu, C. H.; Gonzalez, J. E.; Zhang, H.; and Stoica, I. 2023. Eficient Memory Management for Large Language Model Serving

with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP).

Li, Y.; Dong, B.; Guerin, F.; and Lin, C. 2023. Compressing Context to Enhance Inference Eficiency of Large Language Models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, 6342–6353.

Li, Y.; Naito, A.; and Shirado, H. 2026. Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of PMLR, 69250–69268.

Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Associationfor Computational Linguistics, 12: 157–173.

Llama Team, AI @ Meta. 2024. The Llama 3 Herd ofModels. arXiv:2407.21783.

Lu, L.; Yuan, Y. C.; and McLeod, P. L. 2012. Twenty-Five Years of Hidden Profiles in Group Decision Making: A Meta-Analysis. Personality and Social Psychology Review, 16(1): 54–75.

Maharana, A.; Lee, D.-H.; Tulyakov, S.; Bansal, M.; Barbieri, F.; and Fang, Y. 2024. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 13851–13870.

Packer, C.; Wooders, S.; Lin, K.; Fang, V.; Patil, S. G.; Stoica, I.; and Gonzalez, J. E. 2023. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.

Park, J. S.; O’Brien, J. C.; Cai, C. J.; Morris, M. R.; Liang, P.; and Bernstein, M. S. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST).

Qwen Team. 2024. Qwen2.5 Technical Report. arXiv:2412.15115.

Smit, A. P.; Grinsztajn, N.; Duckworth, P.; Barrett, T. D.; and Pretorius, A. 2024. Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, 45883–45905.

Stasser, G.; and Titus, W. 1985. Pooling of Unshared Information in Group Decision Making: Biased Information Sampling During Discussion. Journal of Personality and Social Psychology, 48(6): 1467–1478.

Subrahmanyam, B.; and Choudhary, N. 2026. Beyond Marginal Influence: Interaction-Aware Memory Selection for LLM Agents. In NeurIPS 2026 Workshop on Personalized, Aligned, Long-Term Memoryfor AI Systems.

Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y.; Li, B.; Zhu, E.; Jiang, L.; Zhang, X.; Zhang, S.; Liu, J.; Awadallah, A. H.; White, R. W.; Burger, D.; and Wang, C. 2024. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. In Conference on Language Modeling (COLM). arXiv:2308.08155.

Yang, J.; Lai, K.-H.; Wang, X.; Chang, S.; Harari, Y.; and Gabrilovich, E. 2026. GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations. arXiv:2605.14498.

# Appendix A: Supplementary Results

This appendix gives benchmark and policy details and all prompts (A.1), the full CrossProfile results (A.2), attribution controls (A.3), natural-discussion diagnostics and prompt ablations (A.4), measurement and statistical details (A.5), the hosted-model runs (A.6), and implementation details (A.7).

## A.1 Tasks, Policies, and Prompts

Benchmark Construction and Example HiddenBench contributes 65 tasks: 58 have four agents and seven have three; 59 have three options and six have four. Its 503 facts comprise 250 shared and 253 private facts. Sixty facts contain several sentences and are kept as one unit. Controlled discussions disclose every assigned fact; natural discussions follow the benchmark’s two-round exchange with initial and final votes.

CrossProfile has ten pilot and 60 main templates. The 60 main templates cover 12 decision domains with five templates each: hospital equipment purchase (M01–M05), school lunch vendor (M06–M10), city bus route (M11–M15), warehouse location (M16–M20), software hosting vendor (M21–M25), community clinic site (M26–M30), job candidate for a lab (M31–M35), conference venue (M36–M40), bridge repair contractor (M41–M45), farm irrigation system (M46–M50), library renovation (M51–M55), and emergency shelter site (M56–M60). Each main template has four agents, three options, and 12 facts. The support group contains three facts in favor of the correct option and two against the decoy option; the decoy group contains two facts in favor of the decoy and one against the correct option. The four irrelevant facts have two positive and two negative tones. The role structure is specified before the sentences are written, and facts receive neutral IDs in random order. Relevance refer to the stated criteria, not to whether a fact favors the answer.

Template M01 asks a hospital procurement panel to choose a scanner supplier, weighting purchase price, electrical safety, and repair response equally. The options are Brightline Imaging, Hallum Diagnostics, and Ostrava Clinical. Table 4 lists its facts with their evaluator roles. The eight variants change the ownership of the role groups while keeping every sentence fixed.

<table><tr><td>ID</td><td>Role</td><td>Fact text</td></tr><tr><td>f01</td><td>decoy</td><td>Hallum Diagnostics has passed every electrical safety inspection on its scanners.</td></tr><tr><td>f02</td><td>support</td><td>Hallum Diagnostics took nine days to answer urgent repair callouts last year.</td></tr><tr><td>f03</td><td>support</td><td>Ostrava Clinical passed every leakage test on its scanners during the last audit</td></tr><tr><td>f04</td><td>support</td><td>Ostrava Clinical engineers reach any ward within four hours for repairs.</td></tr><tr><td>f05</td><td>irrelevant</td><td>Hallum Diagnostics has run an apprenticeship scheme for local school leavers since 2015.</td></tr><tr><td>f06</td><td>irrelevant</td><td>Ostrava Clinical&#x27;s headquarters was flooded in a burst pipe incident in 2019.</td></tr><tr><td>f07</td><td>decoy</td><td>Ostrava Clinical failed a leakage test on one scanner batch during its previous contract.</td></tr><tr><td>f08</td><td>support</td><td>Ostrava Clinical quoted a purchase price about eight percent below the other bids.</td></tr><tr><td>f09</td><td>irrelevant</td><td>Brightline Imaging prints its manuals in large type for older nurses.</td></tr><tr><td>f10</td><td>decoy</td><td>Hallum Diagnostics engineers reached ninety percent of repair callouts within two hours last year.</td></tr><tr><td>f11</td><td>support</td><td>Hallum Diagnostics raised its invoice totals by eleven percent after the first year.</td></tr><tr><td>f12</td><td>irrelevant</td><td>Ostrava Clinical&#x27;s logo was criticised by a local design magazine last spring.</td></tr></table>

Table 4: All facts of template M01 (hospital scanner procurement; correct option: Ostrava Clinical; decoy: Hallum Diagnostics). Roles are evaluator annotations and are never given to the models.

Policy Details The claim inventory records each claim’s text, its set of contributors, and its first-appearance rank. Each memory line is written as [A1,A2] claim text, and the memory keeps the selection order. R1 sorts by contributor count and then by first appearance; B4 uses first appearance; Random uses a random order. M1 cycles through speakers, taking each speaker’s earliest unselected claim that fits, and then fills the remaining budget in first-appearance order. B3 summarizes the transcript instead of copying claims.

For R1-rel, the filter is queried once per template, seed, and model, and its prediction is shared by all eight ownership variants. Statements are presented in random order. If no valid list of statement numbers is obtained after one retry, all facts are treated as relevant, which reduces R1-rel to R1. The predicted-relevant and remaining tiers are each sorted in R1 order.

Copy policies add whole claims and skip claims that would exceed the budget. CrossProfile budgets price every fact with all four holder tags in template order, while selected lines carry their true tags, so the budget is the same across variants but the number of items that fit is not. Summaries are asked for at most max(1, ⌊B/1.3⌋) words, and longer outputs are truncated to the budget at a sentence boundary.

Prompts The prompts below are used verbatim. Text in {braces} is filled in for each call (task description, displayed options, transcript, memory, or numbered statements); no evaluator labels are ever inserted. Line breaks between filled-in component are shown schematically.

Decision call (all memory policies)   
SYSTEM   
You are a careful decision maker. Use only the information you are given.   
USER   
{task description}   
{options}   
Notes from the group discussion. A tag such as [A1,A2] at the start of a line lists the   
participants who stated it:   
{memory}   
Reply in exactly this format and nothing else:   
Reasoning: <at most five sentences that weigh the options against the information above>   
Decision: <one option copied exactly from the options above>   
RETRY AFTER AN UNPARSEABLE REPLY   
Your reply did not follow the format. Reply again with exactly two parts: 'Reasoning: '   
followed by at most five sentences, then 'Decision: ' followed by one option copied   
exactly from the options list.

A reply is valid if exactly one listed option can be parsed from it. Invalid replies are scored as wrong.

Evidence-preserving summary (B3)   
SYSTEM   
You write memories of group discussions. Use only the information in the transcript.   
USER   
{task description}   
{options}   
Below is the transcript of a group discussion about this decision. Each message starts   
with the participant who wrote it, and messages are separated by a blank line.   
{transcript}   
Write a memory of this discussion in at most {W} words for someone who must make the   
decision later and cannot see the transcript. Keep every distinct piece of evidence that   
was mentioned, together with its conditions and qualifiers, any uncertainty, and any   
disagreement between participants. State each piece of evidence once. Do not add a   
recommendation of your own. Write plain text without headings. Output only the memory.

{W} is the word limit derived from the token budget.

## Criterion-relevance filter (R1-rel)

SYSTEM   
You sort statements by topic. Use only the information you are given.   
USER   
{task description}   
{options}   
Below is a numbered list of statements collected in a group discussion about this   
decision.   
{numbered statements}   
The description above names the decision criteria. Go through the statements one by one.   
A statement concerns a criterion if it tells us how an option performs on one of the   
stated criteria; a statement about anything else does not. Collect the numbers of the   
statements that concern a stated criterion and reply with one flat JSON list of those   
numbers, for example [2, 5, 7]. It must be a single list of integers, not an object and   
not grouped by criterion. Reply [] if there are none. Do not add any other text.   
RETRY AFTER AN INVALID REPLY   
Your reply was not one flat JSON list of statement numbers. Reply again with only a   
single JSON list such as [1, 4, 6], using only numbers from the list above, not grouped   
by criterion.

The filter receives no contributor tags, roles, or answer.

Natural discussion (HiddenBench protocol)   
SYSTEM   
{description}   
You have received the following information. Notice that the order of these facts is   
randomly shuffled; the order does not indicate importance or relationship. Please reason   
carefully:   
{information}   
Keep your response concise, just one or two sentences. {extra}   
INITIAL VOTE   
Please decide and provide your rationale in the following JSON format:   
{   
"vote": <A string, {possible answers}>,   
"rationale": <A string, representing your rationale>   
}   
FIRST SPEAKER, FIRST ROUND   
You are the first to speak.   
OTHER TURNS   
Previous messages from other people:   
{messages}   
It is your turn to speak. {extra}   
FINAL VOTE   
Previous messages from other people:   
{group discussion}   
{the initial-vote request above, verbatim}

{information} holds the agent’s assigned facts and {extra} is the benchmark’s optional instruction field. After the initial votes, agents exchange two rounds of messages. In later rounds each agent sees the other agents’ latest replies, including replies already updated in the current round; the final vote follows the last round. Memories are built from the messages, not from the votes. The prompt-ablation sufixes are given in Appendix A.4.

In natural discussions, the claim inventory is built from the messages in two steps: every message is split into factual statements, and candidate pairs of statements with high word overlap are merged when the model judges them to state the same fact.

Claim extraction (natural inventories)   
SYSTEM   
You copy factual statements out of a message. Output JSON only.   
USER   
This message was written in a group discussion:   
<message>   
{message}   
</message>   
List every factual statement in the message.   
Rules:   
1. Copy each statement exactly as written. Do not paraphrase or join text from different   
places.   
2. A statement is one fact together with its conditions and qualifiers. If a sentence   
only adds detail to the sentence before it, keep both sentences together as one   
statement. Example: "The bakery opens at 7. It also sells coffee." is one statement.   
3. Leave out opinions, votes, recommendations and questions. If a sentence mixes a fact   
with an opinion, copy only the part that states the fact, exactly as written.   
4. Keep the order of the message.   
Output a JSON array of strings and nothing else, for example: ["first statement", "second

statement"]   
RETRY AFTER AN INVALID REPLY   
Your reply was not a JSON array of strings copied from the message. Reply again with only   
the JSON array.

Duplicate check (natural inventories)   
SYSTEM   
You compare two statements from a group discussion. Answer with one word: yes or no.   
USER   
Statement 1: {first}   
Statement 2: {second}   
Do these two statements state the same fact, with the same conditions and qualifiers? If   
one adds, drops or changes a detail, a number, a condition or a negation, the answer is   
no.   
Answer yes or no.

## A.2 Full CrossProfile Results

Table 5 reports the three primary contrasts and the shared-irrelevant counterpart of H3 at every budget ratio. At r = .6, the Holm-adjusted p values for H1, H2, and H3 are < .001, .011, and .002 for Qwen and < .001, .060, and < .001 for Llama.
<table><tr><td>Model / ratio</td><td>H1</td><td>H2</td><td>H3</td><td>H3, shared irrelevant</td></tr><tr><td>Qwen / .4</td><td>+65.3 [60.1, 70.1]</td><td>+13.1 [9.7, 16.5]</td><td>+20.8 [16.0, 25.8]</td><td>+8.1 [3.9, 12.2]</td></tr><tr><td>Qwen / .6</td><td>+64.6 [60.3, 69.0]</td><td>+5.0 [1.2, 8.9]</td><td>+8.5 [3.5, 13.6]</td><td>+8.7 [4.6, 12.8]</td></tr><tr><td>Qwen / .8</td><td>+35.3 [30.3, 40.4]</td><td>+2.4 [−1.1, 5.7]</td><td>+2.6 [-1.7, 6.9]</td><td>+2.9 [−0.6, 6.4]</td></tr><tr><td>Llama / .4</td><td>+67.4 [62.6, 72.1]</td><td>+6.4 [2.4, 10.4]</td><td>+15.3 [11.2, 19.4]</td><td>+12.5 [9.0, 16.0]</td></tr><tr><td>Llama / .6</td><td>+62.6 [56.8, 68.2]</td><td>+4.9 [-0.3, 10.0]</td><td>+15.6 [11.1, 19.9]</td><td>+7.6 [4.2, 11.2]</td></tr><tr><td>Llama / .8</td><td>+29.4 [24.7, 34.2]</td><td>-1.5 [-5.3, 2.4]</td><td>+3.8 [0.3, 7.4]</td><td>+3.2 [0.1, 6.4]</td></tr></table>

Table 5: CrossProfile contrasts (points) with 95% template-bootstrap intervals. Only H1–H3 at $r = . 6$ belong to the primary Holm family.

![](images/7075ab2dd16ca398fc6ffd6a3b9162d45e4f8888fbf7af0218953c262635fb0a.jpg)

![](images/4339d669a8a1dc933aedb4bd514edda6a5aef5b95372887b2d99a2f2bc530864.jpg)  
Figure 6: Primary contrasts at r = .6. H1: dependence of the R1 advantage over B4 on decoy ownership. H2: R1 accuracy with shared minus private irrelevant facts. H3: R1-rel minus R1 with private irrelevant facts. Points and whiskers show estimates and 95% template-bootstrap intervals. Filled markers are significant after Holm correction over H1–H3 within each model; Llama H2 is not.

Tables 7–10 give accuracy in each ownership cell at $r = . 6$ with 95% intervals; Figure 8 visualizes the R1 column. Empty memory does not depend on ownership and scores 42.2% [36.1, 48.3] for Qwen and 30.6% [23.9, 37.8] for Llama in every cell.

<table><tr><td rowspan="2">Policy</td><td colspan="3">Qwen</td><td colspan="3">Llama</td></tr><tr><td> $r = . 4$ </td><td> $r = . 6$ </td><td> $r = . 8$ </td><td> $r = . 4$ </td><td> $r = . 6$ </td><td> $r = . 8$ </td></tr><tr><td>B3</td><td>68.9</td><td>72.0</td><td>72.6</td><td>52.8</td><td>58.7</td><td>60.6</td></tr><tr><td>B4</td><td>51.5</td><td>61.0</td><td>68.0</td><td>50.5</td><td>61.5</td><td>70.9</td></tr><tr><td>Random</td><td>57.8</td><td>64.5</td><td>71.6</td><td>55.1</td><td>63.4</td><td>70.6</td></tr><tr><td>R1</td><td>54.6</td><td>64.0</td><td>71.9</td><td>53.6</td><td>63.0</td><td>72.6</td></tr><tr><td>R1-rel</td><td>69.0</td><td>72.6</td><td>74.7</td><td>67.5</td><td>74.6</td><td>76.0</td></tr><tr><td>All facts</td><td>73.1</td><td>73.1</td><td>73.1</td><td>72.6</td><td>72.6</td><td>72.6</td></tr><tr><td>Empty</td><td>42.2</td><td>42.2</td><td>42.2</td><td>30.6</td><td>30.6</td><td>30.6</td></tr></table>

Table 6: CrossProfile accuracy (%) at all budget ratios.

![](images/07a8d873155ce5853ed5ba16654b770b1587dab13ea23591bbf5e5fd3d7fa93b.jpg)

Figure 7: CrossProfile accuracy across budget ratios (Table 6). Points are means over templates, variants, and seeds; horizontal lines mark the unbudgeted all-facts and empty controls.
<table><tr><td>S/D/I</td><td>R1</td><td>R1-rel</td><td></td><td>B4</td><td>Random</td></tr><tr><td>sh / sh / sh</td><td>67.2 [60.0, 74.4]</td><td></td><td>75.6 [68.9, 81.7]</td><td>67.2 [60.0, 74.4]</td><td>60.0 [52.2, 67.8]</td></tr><tr><td>sh / sh / pr</td><td>58.3 [50.6, 66.1]</td><td>68.9 [60.6, 76.7]</td><td></td><td>66.1 [57.8, 74.4]</td><td>62.2 [54.4, 69.4]</td></tr><tr><td>sh / pr / sh</td><td>35.0 [28.3, 41.7]</td><td>73.3 [66.1, 80.0]</td><td></td><td>92.2 [88.3, 96.1]</td><td>72.2 [64.4, 79.4]</td></tr><tr><td>sh / pr / pr</td><td>22.2 [14.4, 30.6]</td><td>67.2 [60.6, 73.9]</td><td></td><td>96.1 [92.8, 98.9]</td><td>76.1 [68.9, 83.3]</td></tr><tr><td>pr / sh / sh</td><td>92.2 [87.8, 96.1]</td><td>75.0 [68.3, 81.7]</td><td></td><td>13.9 [8.3, 20.0]</td><td>58.9 [51.7, 66.1]</td></tr><tr><td>pr / sh / pr</td><td>96.7 [93.3, 99.4]</td><td>74.4 [67.8, 80.6]</td><td></td><td>26.1 [19.4, 33.3]</td><td>53.9 [46.1, 61.7]</td></tr><tr><td>pr / pr / sh</td><td>71.7 [64.4, 78.3]</td><td>77.2 [71.1, 83.3]</td><td></td><td>56.7 [48.3, 65.0]</td><td>63.3 [55.6, 70.6]</td></tr><tr><td> $\mathrm { p r / p r / p r }$ </td><td>68.9 [62.2, 75.6]</td><td>69.4 [63.3, 75.6]</td><td></td><td>70.0 [62.8, 76.7]</td><td>69.4 [61.7, 76.7]</td></tr></table>

Table 7: Qwen, copy policies: accuracy (%) by ownership cell at $r = . 6 ,$ with 95% template-bootstrap intervals. S/D/I: ownership of support, decoy, and irrelevant facts (sh: shared, pr: private).

<table><tr><td>S/D/I</td><td>B3</td><td>All facts</td></tr><tr><td>sh / sh / sh sh / sh / pr sh / pr / sh sh / pr / pr pr / sh / sh pr / sh / pr</td><td>72.2 [65.0, 79.4] 81.7 [76.1, 87.2] 86.1 [79.4, 91.7] 91.1 [86.7, 95.0] 46.7 [40.0, 53.9] 56.1 [47.2, 65.0]</td><td>78.9 [71.7, 85.6] 75.0 [68.3, 81.7] 89.4 [83.9, 94.4] 91.1 [86.1, 95.6] 51.1 [42.2, 60.0] 54.4 [46.1, 62.8] 75.0 [67.8, 82.2]</td></tr></table>

Table 8: Qwen, summary and all-facts memories: accuracy (%) by ownership cell at $r \ = \ . 6 ,$ with 95% template-bootstrap intervals. Notation as in Table 7.

<table><tr><td>S/D/I</td><td>R1</td><td>R1-rel</td><td></td><td>B4</td><td>Random</td></tr><tr><td>sh / sh / sh</td><td>63.9 [56.1, 71.7]</td><td>72.8 [66.7, 78.9]</td><td>62.8 [55.0, 70.6]</td><td></td><td>58.3 [50.0, 66.1]</td></tr><tr><td>sh / sh / pr</td><td>53.3 [45.0, 61.1]</td><td>75.6 [69.4, 81.7]</td><td>68.9 [61.7, 76.1]</td><td></td><td>59.4 [52.8, 66.1]</td></tr><tr><td>sh / pr / sh</td><td>40.0 [31.7, 48.9]</td><td>73.9 [67.2, 80.6]</td><td>91.7 [87.2, 95.6]</td><td></td><td>69.4 [61.7, 77.2]</td></tr><tr><td> $\mathrm { s h / p r / p r }$ </td><td>18.3 [12.2, 24.4]</td><td>78.3 [71.7, 84.4]</td><td>95.6 [92.2, 98.3]</td><td></td><td>72.2 [65.0, 79.4]</td></tr><tr><td> $\mathrm { p r } / \bar { \mathrm { s h } } / \bar { \mathrm { s h } }$ </td><td>90.0 [85.0, 94.4]</td><td>72.8 [65.0, 80.0]</td><td>10.6 [6.1, 15.0]</td><td></td><td>57.8 [49.4, 65.6]</td></tr><tr><td> $\mathrm { p r / \ s h / \ p r }$ </td><td>97.2 [93.9, 99.4]</td><td>77.2 [70.0, 83.9]</td><td>31.1 [23.9, 38.3]</td><td></td><td>62.2 [54.4, 70.0]</td></tr><tr><td> $\mathrm { p r } / \mathrm { p r } / \mathrm { s h }$ </td><td>67.8 [59.4, 75.6]</td><td>72.8 [65.0, 80.6]</td><td>58.3 [50.0, 66.7]</td><td></td><td>65.6 [57.8, 73.3]</td></tr><tr><td> $\mathrm { p r / p r / p r }$ </td><td>73.3 [65.6, 81.1]</td><td>73.3 [66.1, 80.0]</td><td>73.3 [65.6, 81.1]</td><td></td><td>62.2 [53.3, 71.1]</td></tr></table>

Table 9: Llama, copy policies: accuracy (%) by ownership cell at $r = . 6 ,$ with 95% template-bootstrap intervals. Notation as in Table 7.

<table><tr><td>S/D/I</td><td>B3</td><td>All facts</td></tr><tr><td>sh / sh / sh sh / sh / pr sh / pr / sh sh / pr / pr pr / sh / sh pr / sh / pr pr / pr / sh</td><td>63.3 [55.6, 70.6] 65.0 [56.7, 73.3] 71.1 [64.4, 77.8] 71.1 [62.2, 79.4] 38.3 [31.1, 45.6] 50.6 [43.3, 58.3] 55.6 [47.8, 63.3]</td><td>72.2 [64.4, 80.0] 75.6 [67.8, 83.3] 85.0 [77.8, 91.7] 85.0 [78.9, 90.6] 58.9 [50.0, 67.8] 55.6 [47.2, 63.3] 73.3 [65.6, 81.1]</td></tr></table>

Table 10: Llama, summary and all-facts memories: accuracy (%) by ownership cell at $r = . 6 ,$ , with 95% template-bootstrap intervals. Notation as in Table 7.

![](images/0a3a5a6be2c05ffbcdac9bc46ff01a65d915766084864d45db3d9c052caf4a0e.jpg)  
Figure 8: R1 accuracy at $r = . 6$ by support and decoy ownership, separately for each model and irrelevant-fact assignment. Every cell averages the same 60 templates and three seeds; intervals are in Tables 7 and 9.

## A.3 Attribution Controls

Table 11 collects the control analyses. Replacing B4 by Random as the reference tests whether H1 depends on first-mention ordering. The seven-item experiment fixes the number of selected facts at $k = \left\lfloor . 6 \times 1 2 \right\rfloor = 7$ for both policies. Oracle relevance uses the evaluator roles to mark support and decoy facts as criterion-relevant. The tag experiment either removes the holder tags from all-facts memories or reassigns the same tag sets to facts at random; both keep the transcript-derived line order, so they separate attribution from position. Without tags the shared-support gap is +7.2 on Qwen and +7.6 on Llama; with shufled tags it is +14.6 and +6.4.

<table><tr><td>Contrast</td><td>Qwen</td><td>Llama</td></tr><tr><td>H1 with Random as reference</td><td>+40.7 [34.6, 47.1]</td><td>+34.2 [27.8, 41.0]</td></tr><tr><td>H1 with seven retained items</td><td>+66.8 [61.9, 71.7]</td><td>+72.6 [67.2, 78.1]</td></tr><tr><td>Oracle relevance — R1, classic cell</td><td>-23.3[-30.6, -16.1]</td><td>-31.1 [-38.9, -23.9]</td></tr><tr><td>Tagged — untagged support gap, all facts</td><td>+13.8 [8.1, 19.3]</td><td>+6.1 [1.0, 11.2]</td></tr><tr><td>Tagged — shuffled-tag support gap, all facts</td><td>+6.4 [1.9, 11.1]</td><td>+7.4 [2.2, 12.6]</td></tr><tr><td>Untagged — tagged, classic cell, all facts</td><td>+14.4 [7.8, 21.1]</td><td>+13.3 [6.7, 20.0]</td></tr><tr><td>Shuffled — tagged, classic cell, all facts</td><td>+12.8 [3.9, 21.7]</td><td>+12.8 [5.6, 20.0]</td></tr></table>

Table 11: Attribution controls (points) with 95% template-bootstrap intervals. Classic cell: private support, shared decoys, private irrelevant facts.

Selection and filter behavior. In the classic cell, Qwen’s R1 memories contain on average 4.98 support, 0.01 decoy, and 4.00 irrelevant facts, against 4.43, 1.89, and 2.16 under R1-rel. For Llama the corresponding values are 5.00, 0.00, and 4.00 against 4.58, 2.24, and 1.47. The relevance filter recalls 76.7% of criterion-relevant facts that favor an option and 48.1% of those that disfavor one on Qwen, and 94.6% and 70.7% on Llama.

## A.4 Natural Discussions and Prompt Ablations

Five diagnostics each change one component relative to the original natural runs (N0): N1 labels observations with origin-neutral IDs; N2 removes the reminder of the agent’s initial vote; N3 appends the evidence-first instruction of arm A1 below to the discussion prompt; N4 extracts claims at the sentence level; and N5 merges duplicate claims only when their wording is identical N1–N3 regenerate the discussions, whereas N4 and N5 rebuild only the B4 and R1 memories from the original discussions. Qwen shows no detectable change under any diagnostic (Table 12). For Llama, N3 raises REF by +30.8 points [21.5, 39.5], R1 by +27.9 [20.2, 35.7], and B3 by +20.9 [14.2, 27.7] (Table 13).

<table><tr><td>Policy</td><td colspan="2">N1</td><td>N2</td><td>N3</td><td></td><td></td></tr><tr><td>B3</td><td>-1.9 [-7.7, 4.1]</td><td></td><td>-1.0[-8.7, 6.7]</td><td>+2.1 [−5.3, 9.6]</td><td></td><td></td></tr><tr><td>B4</td><td>-1.2[-7.5, 5.1]</td><td></td><td>+1.4 [-5.3, 8.2]</td><td>-3.1 [-9.4, 3.1]</td><td>-0.2 [-3.2, 2.9]</td><td>0.0 [−1.9, 1.9]</td></tr><tr><td>R1</td><td>-2.9 [-8.5, 2.7]</td><td></td><td>-0.7[-7.5, 6.3]</td><td>+1.4 [-5.3, 7.9]</td><td>+1.0 [−2.4, 4.4]</td><td>+0.3[−1.9, 2.4]</td></tr><tr><td>REF</td><td>+0.5 [-5.6, 6.7]</td><td></td><td>+0.5 [−7.2, 8.2]</td><td>+6.2 [−1.5, 13.8]</td><td></td><td></td></tr></table>

Table 12: Qwen: change in natural-discussion accuracy relative to the original runs (points), averaged over budget ratios, with 95% task-bootstrap intervals. Dashes mark memories that N4 and N5 do not rebuild.

<table><tr><td>Policy</td><td>N1</td><td></td><td></td><td></td><td></td></tr><tr><td>B3</td><td>+1.4[−4.6, 7.3]</td><td>+5.1 [−1.2, 11.5]</td><td>+20.9 [14.2, 27.7]</td><td></td><td></td></tr><tr><td>B4</td><td>+3.8 [−3.2, 10.9]</td><td>+6.5 [−0.2, 13.2]</td><td>-6.5 [-12.7, −0.2]</td><td>-2.7 [-8.2, 2.4]</td><td>-0.3 [−2.9, 2.2]</td></tr><tr><td>R1</td><td>+5.0 [−1.0, 11.1]</td><td>+3.4 [-3.6, 10.8]</td><td>+27.9 [20.2, 35.7]</td><td>−2.6 [−7.9, 2.7]</td><td>0.0 [−2.1, 1.9]</td></tr><tr><td>REF</td><td>-1.0 [−9.7, 7.2]</td><td>-1.0[-8.7, 6.7]</td><td>+30.8 [21.5, 39.5]</td><td></td><td></td></tr></table>

Table 13: Llama: change in natural-discussion accuracy relative to the original runs (points), averaged over budget ratios, with 95% task-bootstrap intervals. Dashes mark memories that N4 and N5 do not rebuild.

Prompt ablation. To separate the components of the evidence-first wording, we ran a paired ablation with fresh seeds 7, 23, and 58, in which every arm regenerates the discussions of all 65 tasks. The outcome is REF accuracy from the full transcript, so the ablation concerns the discussion rather than any compression policy. Arm A0 uses the original prompt; arms A1–A4 append the sufixes below to the first-round user prompt.

Discussion-prompt sufixes for arms A1–A4   
A1: EVIDENCE FIRST, NO RECOMMENDATION   
Before recommending an option, state the concrete observations in the information you   
received. Preserve their conditions, negations and numbers. Do not recommend an option in   
this turn.   
A2: EVIDENCE FIRST   
Before recommending an option, state the concrete observations in the information you   
received. Preserve their conditions, negations and numbers.   
A3: NO RECOMMENDATION   
Do not recommend an option in this turn.   
A4: NEUTRAL LENGTH REQUEST   
Please take your time and write a full, careful reply that covers what you know.

<table><tr><td></td><td colspan="2">Qwen</td><td colspan="2">Llama</td></tr><tr><td>Arm</td><td>Accuracy</td><td>∆ vs. A0</td><td>Accuracy</td><td>∆ vs. A0</td></tr><tr><td>A0: original</td><td>30.3 [22.6, 37.9]</td><td></td><td>30.8 [23.1, 39.0]</td><td></td></tr><tr><td>A1: evidence first, no recommendation</td><td>28.2 [21.5, 35.4]</td><td>-2.1 [-10.3, 5.6]</td><td>59.0 [50.8, 67.7]</td><td>+28.2 [17.9, 39.0]</td></tr><tr><td>A2: evidence first</td><td>35.9 [28.2, 43.6]</td><td>+5.6 [-1.0, 12.3]</td><td>55.9 [48.2, 63.6]</td><td>+25.1 [15.9, 34.4]</td></tr><tr><td>A3: no recommendation</td><td>22.6 [15.9, 29.7]</td><td>-7.7[-15.9, 0.0]</td><td>40.5 [33.3, 47.7]</td><td>+9.7 [0.5, 18.5]</td></tr><tr><td>A4: neutral length request</td><td>31.3 [23.6, 39.5]</td><td>+1.0 [-4.1, 6.7]</td><td>38.5 [31.8, 45.6]</td><td>+7.7 [-1.5, 16.9]</td></tr></table>

Table 14: Discussion-prompt ablation: REF accuracy (%) and paired diferences from A0 (points), over 65 tasks and seeds 7, 23, and 58, with 95% task-bootstrap intervals.

![](images/5e4f7c147ffafed8b4ca3b932cc6300addd881d062240431b33e2993c5e87467.jpg)

![](images/a91d570f759cd377435e066b5db02b36dc4e2ebf5c069e415325631735825692.jpg)  
Figure 9: Discussion-prompt ablation: change in REF accuracy relative to A0, with 95% task-bootstrap intervals. Qwen does not reproduce Llama’s large gain from evidence-first wording.

The diference between A1 and A2 on Llama is +3.1 points [−6.2, 12.3].

## A.5 Measurement and Statistical Details

For a fixed policy and budget, task accuracy averages the three seeds. HiddenBench contrasts additionally average the budget ratios within task. CrossProfile accuracy averages seeds within a template–variant pair and then the relevant variants within template. Bootstrap samples resample tasks or whole templates, never individual variants. Invalid answers are scored as wrong.

For paired diferences $d _ { t } ,$ , we draw 10,000 task or template resamples and take the 2.5th and 97.5th percentiles of the resampled mean. The two-sided bootstrap p value is

$$
\hat { p } = \operatorname* { m i n } \{ 1 , 2 \operatorname* { m i n } [ \stackrel { * } { \operatorname* { P r } } ( \bar { d } ^ { * } \leq 0 ) , \stackrel { * } { \operatorname* { P r } } ( \bar { d } ^ { * } \geq 0 ) ] \} ,\tag{3}
$$

with $p = 1$ when all diferences are zero; values below .001 are reported as $p < . 0 0 1$ . Holm’s step-down correction is applied within the pre-specified families. On HiddenBench, each family contains R1 − B3, R1 − B4, and M1 − B4 for one model, discussion condition, and inventory type, with ratios averaged within task. Table 15 lists all contrasts. “Known” inventories are the benchmark’s fact lists used in controlled discussions; “extracted” inventories are built by the model from the transcript.
<table><tr><td>Model</td><td>Condition / contrast</td><td>Effect [95% CI]</td><td>Holm p</td></tr><tr><td>Qwen</td><td>Headroom: all facts — empty</td><td>+36.4 [25.1, 47.2]</td><td></td></tr><tr><td>Qwen</td><td>Controlled, known: R1 — B3</td><td>+29.2 [22.9, 35.4]</td><td>&lt; .001</td></tr><tr><td>Qwen</td><td>Controlled, known: R1 – B4</td><td>+55.4 [49.4, 61.4]</td><td>&lt; .001</td></tr><tr><td>Qwen</td><td>Controlled, known: M1 — B4</td><td>-4.6 [−8.2, -1.2]</td><td>.008</td></tr><tr><td>Qwen</td><td>Controlled, extracted: R1 — B3</td><td>+25.8 [18.8, 32.6]</td><td>&lt; .001</td></tr><tr><td>Qwen</td><td>Controlled, extracted: R1 – B4</td><td>+53.2 [47.5, 58.6]</td><td>&lt; .001</td></tr><tr><td>Qwen</td><td>Controlled, extracted: M1 — B4</td><td>-2.9 [–6.3, 0.5]</td><td>.094</td></tr><tr><td>Qwen</td><td>Natural: R1 — B3</td><td>-1.4 [−5.0, 2.4]</td><td>1.000</td></tr><tr><td>Qwen</td><td>Natural: R1 - B4</td><td>-0.5 [-2.6, 1.5]</td><td>1.000</td></tr><tr><td>Qwen</td><td>Natural: M1 — B4</td><td>+1.2 [−1.7, 4.1]</td><td>1.000</td></tr><tr><td>Llama</td><td>Headroom: all facts — empty</td><td>+23.6 [11.3, 35.9]</td><td></td></tr><tr><td>Llama</td><td>Controlled, known: R1 – B3</td><td>+26.7 [20.2, 33.2]</td><td>&lt; .001</td></tr><tr><td>Llama</td><td>Controlled, known: R1 – B4</td><td>+52.5 [46.3, 58.1]</td><td>&lt; .001</td></tr><tr><td>Llama</td><td>Controlled, known: M1 — B4</td><td>-4.6 [−8.0, −1.4]</td><td>.006</td></tr><tr><td>Llama</td><td>Controlled, extracted: R1 — B3</td><td>+18.1 [11.5, 25.0]</td><td>&lt; .001</td></tr><tr><td>Llama</td><td>Controlled, extracted: R1 — B4</td><td>+44.4 [37.1, 51.5]</td><td>&lt; .001</td></tr><tr><td>Llama</td><td>Controlled, extracted: M1 — B4</td><td>-2.2 [−7.0, 2.6]</td><td>.376</td></tr><tr><td>Llama</td><td>Natural: R1 — B3</td><td>-3.1 [-9.2, 2.7]</td><td>.888</td></tr><tr><td>Llama</td><td>Natural: R1 - B4</td><td>-0.2 [-3.2, 2.9]</td><td>1.000</td></tr><tr><td>Llama</td><td>Natural: M1 — B4</td><td>+0.3 [−4.1, 4.8]</td><td>1.000</td></tr></table>

Table 15: All HiddenBench headroom and memory contrasts (points).

<table><tr><td>Comparison</td><td>Qwen</td><td>Llama</td></tr><tr><td>Controlled: R1 - B3</td><td>+29.2 [22.9, 35.4]</td><td>+26.7 [20.2, 33.2]</td></tr><tr><td>Controlled: R1 at B3 length – B3</td><td>+25.6 [17.6, 33.7]</td><td>+25.8 [19.1, 32.5]</td></tr><tr><td>Natural: R1 - B3</td><td>-1.4 [−5.0, 2.4]</td><td>-3.1 [-9.2, 2.7]</td></tr></table>

Table 16: HiddenBench accuracy diferences (points) with 95% task-bootstrap intervals over 65 tasks; seeds and the three budget ratios are averaged within task. The controlled R1 − B3 contrast belongs to the pre-specified Holm family.

HiddenBench exact-copy retention checks whether each normalized fact is contained as a whole in the normalized memory. Facts are averaged within task and then over seeds and tasks. CrossProfile retention is computed from the selected facts and their roles. Table 1 in the main text reports the HiddenBench retention values.

Table 17 reports the mean ratio of realized memory tokens to the budget in CrossProfile; these are means, unlike the medians in Table 1. The R1-at-B3-length comparison uses each summary’s actual length as the budget.

<table><tr><td>Model / policy</td><td>Tokens / budget (%)</td><td>95% CI</td></tr><tr><td>Qwen / R1</td><td>95.0</td><td>[94.8, 95.3]</td></tr><tr><td>Qwen / R1-rel</td><td>95.7</td><td>[95.5, 95.8]</td></tr><tr><td>Qwen / B4</td><td>95.8</td><td>[95.7, 96.0]</td></tr><tr><td>Qwen / Random</td><td>95.6</td><td>[95.5, 95.8]</td></tr><tr><td>Qwen / B3</td><td>79.3</td><td>[77.8, 80.8]</td></tr><tr><td>Llama /R1</td><td>95.1</td><td>[94.9, 95.4]</td></tr><tr><td>Llama / R1-rel</td><td>95.8</td><td>[95.7, 96.0]</td></tr><tr><td>Llama / B4</td><td>95.8</td><td>[95.7, 95.9]</td></tr><tr><td>Llama / Random</td><td>95.5</td><td>[95.3, 95.6]</td></tr><tr><td>Llama / B3</td><td>85.5</td><td>[84.9, 86.0]</td></tr></table>

Table 17: Realized memory length relative to the budget in CrossProfile at $r = . 6 ,$ averaged within template, with 95% templatebootstrap intervals.

## A.6 Hosted Models and Fixed Budget

Table 18 reports the hosted-model contrasts. DeepSeek-V4.1-Flash H1–H3 are Holm-corrected within that run. The classic cel is the cell with private support, shared decoys, and private irrelevant facts, as in the main text. The DeepSeek-V4-Pro run covers templates M01–M30 and four memory conditions (R1, Random, all facts, empty); R1-rel was not run on DeepSeek-V4-Pro. Averaged over variants, R1 minus Random on this subset is close to zero for every model (−1.7, +1.7, −1.3, and −4.7 points for Qwen, Llama, DeepSeek-V4.1-Flash, and DeepSeek-V4-Pro), while its dependence on decoy ownership is large for all four. The natural-discussion run on DeepSeek-V4.1-Flash includes only REF, all-facts, and empty memories.

<table><tr><td>Run</td><td>Contrast</td><td>Effect [95% CI]</td><td>Templates / tasks</td></tr><tr><td>DeepSeek-V4.1-Flash, CrossProfile</td><td>H1</td><td>+62.6 [56.7, 68.6]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, CrossProfile</td><td>H2</td><td>+11.5 [7.5, 15.6]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, CrossProfile</td><td>H3</td><td>+16.7 [12.8, 20.7]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, CrossProfile</td><td>H3, shared irrelevant</td><td>+6.8 [3.2, 10.4]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, CrossProfile</td><td>H1 with Random as reference</td><td>+28.3 [22.1, 34.7]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, CrossProfile</td><td>R1-rel — R1, classic cell</td><td>-32.2[-40.6, -24.4]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, additional controls</td><td>Oracle relevance — R1, all variants</td><td>+11.7 [8.2, 15.1]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, additional controls</td><td>Oracle relevance — R1, classic cell</td><td>-29.4[-37.8, -21.7]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, additional controls</td><td>Untagged — tagged all facts, classic cell</td><td>+24.4 [17.2, 32.2]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, additional controls</td><td>Tagged — untagged support gap</td><td>+12.5 [8.3, 17.1]</td><td>60</td></tr><tr><td>DeepSeek-V4.1-Flash, natural</td><td>REF – empty</td><td>-10.3 [-19.5, −1.0]</td><td>65</td></tr><tr><td>DeepSeek-V4.1-Flash, natural</td><td>Private facts stated in full (%)</td><td>1.7 [0.6, 2.9]</td><td>65</td></tr><tr><td>DeepSeek-V4-Pro, CrossProfile subset</td><td>H1 with Random as reference</td><td>+30.6 [22.5, 38.6]</td><td>30</td></tr><tr><td>DeepSeek-V4-Pro, CrossProfile subset</td><td>R1 — Random, all variants</td><td>-4.7[−8.2, −1.2]</td><td>30</td></tr><tr><td>DeepSeek-V4-Pro, CrossProfile subset</td><td>Tagged support gap, all facts</td><td>+7.2 [3.3, 11.4]</td><td>30</td></tr></table>

Table 18: Hosted-model contrasts (points) with 95% bootstrap intervals.

<table><tr><td>Run</td><td>Valid</td><td>Invalid</td></tr><tr><td>DeepSeek-V4.1-Flash, CrossProfile</td><td>8,466</td><td>174</td></tr><tr><td>DeepSeek-V4.1-Flash, additional controls</td><td>4,184</td><td>136</td></tr><tr><td>DeepSeek-V4.1-Flash, natural</td><td>564</td><td>21</td></tr><tr><td>DeepSeek-V4-Pro, CrossProfile subset</td><td>2,880</td><td>0</td></tr></table>

Table 19: Hosted-model decision counts. As in the local runs, a decision is invalid only after one format retry, and invalid decisions are scored as wrong.

Fixed budget. To check that the controlled R1 advantage over B3 does not depend on defining budgets as a ratio of the fact set, we also ran both policies with a fixed 128-token budget on controlled HiddenBench. There, R1 and B3 reach 81.0% and 52.3% on Qwen and 82.1% and 57.4% on Llama. This experiment used a 4,096-token context rather than the 8,192-token context of the main runs.

## A.7 Implementation Details

Local models are served with vLLM on two NVIDIA T4 GPUs. DeepSeek-V4.1-Flash and DeepSeek-V4-Pro are accessed through the DeepSeek API (model identifiers deepseek-flash and deepseek-v4-pro). The order in which options are displayed rotates across seeds.

We will release the code, the templates, and all generated discussions, memories, and model outputs upon publication.