# ATOMSEARCH: EVIDENCE-GROUNDED TREE SEARCH FOR AGENT SKILL OPTIMIZATION

Jia Tong, Senyou Deng, Min Peng, Jianfeng Pan

360 Security Technology Inc. Beijing, China Emails: tongjia1@360.cn, dengsenyou@360.cn, pengmin1@360.cn

## ABSTRACT

Agent skills adapt frozen models through editable external instructions. Revising several rules at once makes changes difficult to inspect, while judging each local edit only by its immediate score can discard useful intermediate states. We introduce AtomSearch, which keeps revision actions local while using subsequent skill states to guide search. Contrastive atomic editing links each primitive rule edit to failed and successful executions, a behavioral hypothesis, and an intended scope. Skill-State Tree Search (STS) evaluates short edit compositions and selectively retains them as executable states that can be revisited and extended. Outcome memory supplies later proposals with observed repair and regression statistics. We evaluate AtomSearch across four tasks and five target models. With deepseekv4-flash, it exceeds SkillOpt on three of four tasks, including a 9.68-percentagepoint gain on LiveMathematicianBench, and SkillOpt’s final skills contain 2.3– 19.7 times as many characters. In 67% of completed searches, the best validation state passes through an ancestor that was not highest-scoring at its depth. These results show how local rule edits can support the exploration of useful revision paths and yield compact skills, with additional offline evaluations used to explore alternative states.

## 1 INTRODUCTION

Agent skills package task-specific instructions, procedures, scripts, and supporting resources into reusable artifacts that an agent can load at execution time (Zhang et al., 2025a). Because the skill sits outside the model parameters, it can be revised without retraining the underlying model. Reusable artifacts also include executable code skills, as in Voyager (Wang et al., 2023). We study how execution feedback can improve a fixed agent by revising its skill instructions.

The hard part is deciding what a revision should be and how its value should be judged. A candidate update may change several behavioral rules at once. If the score rises, the gain is assigned to the whole update rather than to a specific change; if it falls, the source of the regression is equally unclear. Repeating this process can also leave behind rules whose individual value was never established, so the skill grows even when some additions are redundant.

Recent work already addresses parts of this problem. SkillOpt restricts revisions to bounded add, delete, or replace operations and validates candidates on held-out data (Yang et al., 2026). Skillα decomposes skill generation into individually evaluable sequential edits (Shen et al., 2026), and SkillRevise uses execution traces to diagnose defects before applying targeted repairs (Liu et al., 2026). Tree-based methods address a different limitation: bilevel skill optimization uses MCTS to search skill structure (Huang et al., 2026), while SkillHEX maintains persistent revision branches under sparse test-time feedback (Feng et al., 2026). These methods make revisions more controlled or broaden the search, but the revision unit and the horizon used to value that revision are still largely separate design choices.

Fine-grained edits can still interact. One edit may establish a condition under which a later edit becomes useful despite little immediate gain. It may also enable a correction that would not have been proposed from the original state. An optimizer that keeps only locally strong states can therefore discard useful paths. Figure 4 shows one such path, where a weaker intermediate state leads to the best state in the tree.

![](images/6e128d27feb7bd407a3287f4e185979fe512397a76c8dd3bf1a218d4781c192d.jpg)  
Figure 1: Overview of AtomSearch. Contrastive atomic editing (top) contrasts failed and passed executions into edit specifications and ranked primitive edits. Skill-State Tree Search (left and bottom) applies the top-ranked edit at the node selected by UCT, also scores a short lookahead with up to two further edits, keeps the lookahead only if it beats the primitive child, and backs up the kept state’s reward. Outcome memory (right) records repairs and regressions per edit for later proposals; its counts are schematic.

AtomSearch decouples the granularity of revision from the horizon over which revision paths are evaluated: each proposal specifies one primitive rule edit, while later outcomes guide exploration of its path. Contrastive atomic editing links the edit to failed and successful executions, a behavioral hypothesis, scope, expected effect, and operation rationale. The change and its evidence are traceable; execution measures the resulting skill’s performance.

Skill-State Tree Search (STS) then evaluates that local action in a broader search context. Each node is a complete executable skill. Alternative revision paths remain searchable, and short deterministic rollouts test whether a primitive edit becomes useful after additional edits. When a rollout beats its primitive child, its composition becomes a persistent state that can be revisited. The edit stays local, but its value is allowed to emerge over later skill states.

Execution also reveals repair and regression outcomes that a scalar score hides. A revision can repair failed cases, break previously correct ones, or repeat a direction that has already been tried. We keep this information in a bounded outcome memory and expose selected records to later proposal rounds. The tree controls where search effort goes; the memory supplies the proposer with what earlier revisions actually did.

Repeated skill revision of this kind is closely related to recursive self-improvement (RSI), in which an agent keeps improving the artifacts that govern its own behavior (Zelikman et al., 2023; Zhang et al., 2025b; 2026; Wu et al., 2026). The longer such a loop runs, the more it matters that each edit can be traced to its evidence and that earlier repairs and regressions are remembered; contrastive atomic editing and outcome memory provide both. The skill growth noted earlier matters as well, since every accepted change is another chance for the skill to lengthen.

## Our contributions are:

• We define a contrastive atomic revision interface that binds execution evidence, a behavioral hypothesis, and scope to one primitive rule edit. Alternative corrections remain separate candidates that search can test and combine.

• We develop Skill-State Tree Search over complete executable skills. Lookahead compositions that improve on their primitive child are kept as tree nodes, allowing their outcomes to guide further exploration of earlier revision paths.

• We incorporate an outcome memory that supplies later proposals with attempted revision directions and repair/regression statistics. Results from a shared evaluation remain attached to the edit composition that produced them.

• Across four tasks and five targets, we measure performance, retained edit paths, skill size, and offline cost. With deepseek-v4-flash, AtomSearch exceeds SkillOpt on three tasks and produces shorter skills on all four; the recorded search trees also expose useful paths through locally weaker states.

## 2 RELATED WORK

Localized and execution-grounded revision. Prompt optimizers search instructions (Zhou et al., 2023; Yang et al., 2024), revise them using textual feedback (Pryzant et al., 2023; Yuksekgonul et al., 2024), or evolve candidate populations (Guo et al., 2024; Fernando et al., 2024). DSPy optimizes modular LM programs, and MIPRO jointly optimizes their instructions and demonstrations using end-to-end evaluation (Khattab et al., 2024; Opsahl-Ong et al., 2024). Other methods narrow the revision unit. ConstitutionalExperts edits individual principles (Petridis et al., 2024); AvaTaR contrasts successful and unsuccessful training examples to improve agent prompts (Wu et al., 2024); Contrastive Reflection uses failures and nearby successful cases for targeted repairs (Koh et al., 2026). SkillOpt applies bounded edits under validation, Skill-α evaluates sequential edits, and Skill-Revise diagnoses defects from execution traces (Yang et al., 2026; Shen et al., 2026; Liu et al., 2026). AtomSearch uses this feedback to specify one primitive rule edit as a search action, keeping its hypothesis and scope attached to the action.

Search over reasoning and reusable artifacts. Tree of Thoughts explores intermediate reasoning paths, while LATS combines tree search with actions, environment feedback, and reflection (Yao et al., 2023; Zhou et al., 2024). These methods search task-solving trajectories. Search can also optimize artifacts reused across tasks: prompts, workflows, and heuristics (Wang et al., 2024; Zhang et al., 2025c; Zheng et al., 2025). ADAS generates agent programs using an archive of previous designs (Hu et al., 2025). GEPA uses trajectory reflection to evolve prompts and maintains a Pareto set of complementary candidates (Agrawal et al., 2026). For skills, bilevel optimization searches structure and refines content, while SkillHEX maintains revision branches under sparse feedback (Huang et al., 2026; Feng et al., 2026). AtomSearch searches complete executable skills, with primitive edits as local actions; edit compositions that improve on their primitive child are kept as tree nodes.

Reusing feedback and maintaining skill content. Self-Refine and CRITIC use self-feedback and tool-based critique to revise outputs (Madaan et al., 2023; Gou et al., 2024). Across attempts or tasks, agents reuse verbal reflections and extracted experience (Shinn et al., 2023; Zhao et al., 2024), induced workflows (Wang et al., 2025), and evolving strategy notes (Suzgun et al., 2025). ACE maintains a structured playbook through reflection and incremental updates (Zhang et al., 2026); SkillOpt also retains rejected-edit feedback and optimization guidance (Yang et al., 2026). Our memory stores revision directions and observed repair/regression statistics for later proposals, retaining joint provenance when edits share an evaluation. The amount of retained content is a separate design choice: ACE preserves detailed knowledge, whereas SkillReducer explicitly targets token efficiency (Zhang et al., 2026; Gao et al., 2026). AtomSearch has no length objective; its fina skill sizes are measured outcomes.

## 3 METHOD

Figure 1 summarizes the overall framework. Given an initial skill $S _ { 0 } .$ , a frozen target model $f ,$ and a held-out navigation set $\mathcal { D } _ { \mathrm { n a v } }$ , we optimize only the skill text. For a task-specific scoring function m, the value of a skill is

$$
V ( S ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { n a v } } | } \sum _ { ( x , y ) \in \mathcal { D } _ { \mathrm { n a v } } } m ( f ( x ; S ) , y ) ,\tag{1}
$$

where $( x , y )$ is an input and its reference output in $\mathcal { D } _ { \mathrm { n a v } }$ , and $f ( x ; S )$ is the output of $f$ on x when S is supplied as its skill.

The proposal unit is one primitive edit; the evaluated state may contain a composition of edits. Its measured score belongs to that state, while tree backup uses it to guide exploration of the preceding path.

![](images/199a5eb20d206d99aa6ccfa592f243f7729c4f1ecc637947c69286ec2357eaa7.jpg)  
Figure 2: A contrastive atomic edit from SpreadsheetBench. Failed and successful executions are contrasted into one edit specification, and the resulting add edit cites it through criterion id. Sample IDs are SpreadsheetBench task IDs. Pattern summaries and specification fields are condensed from the recorded proposal; the added rule is paraphrased. The observed gain is on the 40-item validation panel.

## 3.1 CONTRASTIVE ATOMIC EDITING

We represent a skill as a sequence of addressable rule atoms. At each proposal round, the optimizer contrasts failed and successful executions and returns edit specifications together with their edits. A specification has five fields. support ids cites failed executions from the current minibatch that fail for the same underlying reason, and counterexample ids cites at least one successful execution whose behavior the edit must preserve. hypothesis states the behaviora distinction inferred from this contrast, scope states when it applies and when it does not, and expected effect names the failures to be repaired and the successful behavior to be kept. The optimizer also explains the chosen operation and edit count. Each edit cites one specification through criterion id and performs one primitive operation: add, rewrite, or delete; one specification justifies at most one edit.

Before search, a structural validator checks evidence-ID membership, minimum evidence counts, required fields, and the specification–edit mapping. It also removes duplicate edits and enforces the edit budget. These checks establish structural consistency, not the semantic correctness of the hypothesis; execution measures the resulting skill’s observed performance.

Different corrections for the same failure pattern remain separate candidates. The search can there-<sup>fi</sup>fore compare those directions without first merging them into a multi-rule patch.

Figure 2 shows a concrete example: the proposed rule repairs formula-dependent value outputs while preserving tasks that explicitly require formula strings.

## 3.2 SKILL-STATE TREE SEARCH

A local edit can be easy to inspect and still be hard to value in isolation. An edit with little immediate gain may lead to a state from which a stronger correction becomes possible. We therefore search over complete skill states with Skill-State Tree Search (STS), rooted at S . STS uses an MCTS backbone: each node stores an executable skill, and expansion starts from contrastively grounded primitive edits.

Selection and widening. At a node with skill state S, STS decides whether to expand S once more or to descend into one of its existing children. The UCT selection rule (Kocsis & Szepesvari,´

2006; Browne et al., 2012) scores both kinds of candidates: the current node itself, whose statistics come from the expansions made at $S ,$ and each child, whose statistics come from its subtree. For a candidate i with visit count $n _ { i }$ and mean backed-up value $q _ { i }$

$$
\operatorname { U C T } ( i \mid S ) = \left\{ \begin{array} { l l } { + \infty , } & { n _ { i } = 0 , } \\ { q _ { i } + c _ { \mathrm { u c t } } \sqrt { \frac { \log \operatorname* { m a x } \{ N ( S ) , 1 \} } { n _ { i } } } , } & { n _ { i } > 0 , } \end{array} \right.\tag{2}
$$

where $N ( S )$ is the total number of visits through $S$ and $c _ { \mathrm { u c t } } = 0 . 0 5$ is the exploration constant. For the current node, $n _ { i }$ counts only the expansions made at S itself. The shared statistics place “expand here” and “continue down an existing branch” in the same budget-allocation decision.

To limit early branching, STS adopts progressive widening (Browne et al., 2012). In our implementation, the number of children currently eligible at S is bounded by

$$
K ( S ) = \mathrm { m a x } \{ 1 , \mathrm { m i n } ( B , \lceil c _ { \mathrm { p w } } \mathrm { m a x } \{ N ( S ) , 1 \} ^ { \alpha } \rceil ) \} ,\tag{3}
$$

where $B = 6$ is the maximum branch factor, $c _ { \mathrm { p w } } = 1 . 0$ controls the widening rate, and $\alpha = 0 . 5$ is the widening exponent. Here $K ( S )$ only determines whether another child is eligible for exploration. It does not force an expansion. The current node still competes with its existing children under UCT for the next optimization attempt: progressive widening limits the available capacity, and UCT allocates it. Branch growth is therefore driven by the same observed returns that govern the rest of the tree, rather than by a fixed expand-as-soon-as-eligible schedule.

Expansion and execution-grounded lookahead. When the current node is selected for expansion, STS draws from the node’s available contrastively grounded candidate edits, refilling this pool when necessary. A selected primitive edit is applied to the current skill to produce a child state, which is evaluated on the navigation set.

The remaining ranked candidates from the same proposal pool provide a short lookahead. STS applies up to two of them in order, skipping conflicts, invalid edits, and no-ops. This rollout is deterministic. Only the final composed skill is executed and scored; intermediate rollout states are not evaluated separately.

Keeping compositions and backup. A composition is kept only if it scores strictly higher than the primitive child; ties keep the primitive child. Let $S ^ { \prime }$ denote that primitive child and $S _ { \mathrm { r o l l } }$ the final composed skill produced by the short rollout. STS adds to the tree

$$
S _ { \mathrm { k e e p } } = \left\{ \begin{array} { l l } { S _ { \mathrm { r o l l } } , } & { V ( S _ { \mathrm { r o l l } } ) > V ( S ^ { \prime } ) , } \\ { S ^ { \prime } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

A composition that scores higher than $S ^ { \prime }$ thus joins the tree as a single child node carrying all of its edits; otherwise only $S ^ { \prime }$ is added. Intermediate rollout states are never added, and unused rollout edits remain available for later attempts.

The measured outcome belongs to the kept skill state, not independently to every edit in a kept composition. STS backs up only that state’s value, using $V ( S _ { \mathrm { k e e p } } ) - \dot { V ( S _ { 0 } ) }$ as the reward. The same reward updates the expansion statistics of the node that acted and the subtree statistics of its ancestors, so later outcomes can affect which earlier revision paths receive more search.

At termination, STS uses a recursive adaptation of the robust-child recommendation rule (Browne et al., 2012). Starting from the root, it repeatedly selects the most visited child, stopping at a leaf or when the selected child has at most one visit. The resulting state is returned, while the highest-value evaluated state is also retained. The revision-trajectory and tree analyses below use the latter to characterize discovered paths.

## 3.3 OUTCOME MEMORY

We supplement tree statistics with a bounded outcome memory of attempted revision directions and their repair/regression outcomes. Each record contains the attempted direction, an outcome tag, and the available aggregate repair and regression counts. When several edits share one evaluation, the record marks that result as belonging to the composition rather than assigning it to every edit separately. Failed attempts remain in the history as well.

Before another proposal round, the optimizer receives a bounded subset of these records: aggregate track records for explored directions together with selected past attempts. The memory is used only when proposing edits. It does not change UCT, progressive widening, or the search reward. We keep that boundary explicit because the two signals serve different purposes. Repair and regression counts summarize what a revision changed on individual examples, while tree values compare complete skill states on the navigation objective. Folding the former into UCT would mix those scales and make search allocation harder to interpret. The memory therefore changes what the proposer sees; execution still determines the value used by the tree.

The complete optimization procedure is given in Algorithm 1 in Appendix A.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Tasks and models. We evaluate AtomSearch on Network Detection and Response (NDR), an in-house security task, and three public benchmarks: SpreadsheetBench (Ma et al., 2024), SearchQA (Dunn et al., 2017), and LiveMathematicianBench (He et al., 2026). The public tasks cover spreadsheet manipulation, search-grounded question answering, and mathematical reasoning. The frozen target models are deepseek-v4-flash, GPT-5.2, GPT-5.5, Qwen3.5-4B, and Qwen3.6- 35B-A3B. We optimize only the external skill.

NDR evaluation. NDR is an in-house dataset of HTTP traffic records constructed for network detection and response. All records are de-identified before use, and the dataset contains no personally identifiable information. For each record, the agent predicts an attack-status label (IsAttack), its type (AttackType), and its outcome (AttackResult). A prediction is correct only when the required fields jointly match the reference, including required empty outputs; attack types are compared as normalized sets. This setting tests whether explicit rules help distinguish evidence of an attack attempt from evidence that it succeeded. We evaluate the resulting skills offline on held-out records. NDR complements the public benchmarks with a domain-specific security task in which correct predictions require consistent reasoning across several labeled fields.

Comparison protocol. Our primary baseline is SkillOpt (Yang et al., 2026). The paired trajectory comparisons use the same underlying data splits and task-skill initialization. Execution feedback is used to construct candidate revisions, while a held-out validation panel (the navigation set D<sub>nav</sub>) provides the signal used during optimization and skill selection. The test split is reserved for final reporting and is not used to generate revisions or guide tree search. For validation-trajectory analyses, both methods are evaluated on the same examples.

Edit count and search cost are different quantities in our analysis. Edit counts refer to textual rule edits on the retained trajectory; total target-model evaluations also include states explored and discarded by the tree.

## 4.2 PERFORMANCE ACROSS TASKS AND TARGET MODELS

Table 1 reports test scores separately for each task and target. AtomSearch is higher on three of four tasks with both deepseek-v4-flash and GPT-5.2, on all four with Qwen3.6-35B-A3B, and on two with Qwen3.5-4B, with one tie. GPT-5.5 has two higher and two lower scores. These are point estimates, not significance claims.

On NDR, AtomSearch improves the reported score by 6.00 percentage points for both deepseek-v4- flash and GPT-5.2, and by 2.00 for Qwen3.6-35B-A3B. It ties Qwen3.5-4B and is 2.00 points lower with GPT-5.5. The NDR results therefore support offline applicability, with benefits that depend on the target model.

Table 1: Test scores (%) across target models and tasks. Bold marks the higher score within each pair, including ties.
<table><tr><td>Target model</td><td>Method</td><td>NDR</td><td>Spread.</td><td>SearchQA</td><td>LiveMath</td></tr><tr><td>GPT-5.5</td><td>SkillOpt</td><td>73.00</td><td>77.50</td><td>87.10</td><td>61.30</td></tr><tr><td></td><td>AtomSearch</td><td>71.00</td><td>79.64</td><td>85.57</td><td>65.32</td></tr><tr><td>deepseek-v4-flash</td><td>SkillOpt AtomSearch</td><td>65.00</td><td>68.93</td><td>83.29</td><td>42.74</td></tr><tr><td>GPT-5.2</td><td></td><td>71.00</td><td>78.21</td><td>80.64</td><td>52.42</td></tr><tr><td></td><td>SkillOpt AtomSearch</td><td>66.00 72.00</td><td>57.10 57.50</td><td>83.10</td><td>36.00</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td>82.50</td><td>41.13</td></tr><tr><td></td><td>SkillOpt AtomSearch</td><td>63.00 63.00</td><td>23.90 26.43</td><td>71.20</td><td>52.00</td></tr><tr><td>Qwen3.6-35B-A3B</td><td></td><td></td><td></td><td>78.29</td><td>51.61</td></tr><tr><td></td><td>SkillOpt</td><td>66.00</td><td>47.50</td><td>80.30</td><td>41.60</td></tr><tr><td></td><td>AtomSearch</td><td>68.00</td><td>59.64</td><td>80.79</td><td>43.55</td></tr></table>

Spread.: SpreadsheetBench; LiveMath: LiveMathematicianBench. SkillOpt scores with GPT-5.5 are its reported default-configuration results.

Table 2: Validation and test gains over each run’s recorded initial skill (percentage points). R is the test gain divided by the validation gain, expressed as a percentage; it is not an accuracy. Ratios and medians are computed before rounding. Bold marks the larger $\Delta _ { \mathrm { t e s t } }$ and the larger R within each pair.
<table><tr><td rowspan="2">Task / target</td><td colspan="3">AtomSearch</td><td colspan="3">SkillOpt</td></tr><tr><td> $\Delta _ { \mathrm { v a l } }$ </td><td> $\Delta _ { \mathrm { t e s t } }$ </td><td>R</td><td> $\Delta _ { \mathrm { v a l } }$ </td><td> $\Delta _ { \mathrm { t e s t } }$ </td><td>R</td></tr><tr><td>Spreadsheet / deepseek</td><td>42.50</td><td>38.21</td><td>90%</td><td>42.50</td><td>30.00</td><td>71%</td></tr><tr><td>NDR / deepseek</td><td>11.43</td><td>15.00</td><td>131%</td><td>13.33</td><td>12.00</td><td>90%</td></tr><tr><td>NDR / GPT-5.2</td><td>20.00</td><td>27.00</td><td>135%</td><td>20.95</td><td>20.00</td><td>95%</td></tr><tr><td>NDR / GPT-5.5</td><td>24.13</td><td>20.00</td><td>83%</td><td>29.52</td><td>24.00</td><td>81%</td></tr><tr><td>NDR / Qwen3.5-4B</td><td>12.38</td><td>20.00</td><td>162%</td><td>18.10</td><td>15.00</td><td>83%</td></tr><tr><td>NDR / Qwen3.6-35B-A3B</td><td>8.57</td><td>10.00</td><td>117%</td><td>15.24</td><td>8.00</td><td>52%</td></tr><tr><td>Median</td><td></td><td></td><td>124%</td><td></td><td></td><td>82%</td></tr></table>

## 4.3 VALIDATION GAINS ON HELD-OUT TEST DATA

We compare improvement over each run’s initial skill on validation and test data. For a run, let $\Delta _ { \mathrm { v a l } } = V _ { \mathrm { f i n a l } } - V _ { \mathrm { i n i t i a l } }$ and $\Delta _ { \mathrm { t e s t } } = T _ { \mathrm { f i n a l } } - T _ { \mathrm { i n i t i a l } }$ . Table 2 reports both gains and their ratio, $1 0 0 \Delta _ { \mathrm { t e s t } } / \Delta _ { \mathrm { v a l } }$ , for six reported comparisons with positive validation gains.

AtomSearch has the larger ratio in all six comparisons; the median is 124%, versus 82% for SkillOpt. On SpreadsheetBench, both improve validation accuracy by 42.50 points, while their test gains are 38.21 and 30.00 points. Thus the observed validation improvements are accompanied by test improvements in these runs. The ratio compares gains, not final accuracy: on NDR with Qwen3.5- 4B, both methods finish at 63.00 despite different ratios. Appendix C discusses these baselines and the scope of the comparison.

## 4.4 REVISION PATHS AND NON-MYOPIC SEARCH BEHAVIOR

Final test scores hide the revision path. Figure 3 instead plots validation accuracy against cumulative edits on the reported trajectory. AtomSearch follows the root-to-validation-argmax path and omits edits explored on other branches; SkillOpt follows its incumbent chain. These curves describe path length, not total search computation.

On NDR, AtomSearch reaches 70.48 after five retained edits; SkillOpt first reaches the same validation score after 31. On SpreadsheetBench, AtomSearch reaches 73.75 after four edits, while

![](images/cfb831d9c11147d367450577021da7d035e14088955a7611dc08db3425d396bf.jpg)

![](images/1f0e72b238f1619d4c802b60c57bc7ca766dcfa5b355a2594a97e5537fcf37a5.jpg)

Figure 3: Revision trajectories on shared validation panels. AtomSearch counts edits only along the displayed root-to-validation-argmax path, while SkillOpt follows its incumbent skill chain. All points are single recorded evaluations. Both paths include a score decrease before reaching their maxima. Edit counts exclude candidates explored off the displayed path.  
![](images/46ee98735ac58749ef22ada34fd5f1031bd0e5d09faa8f46b3dee7d5dfce2277.jpg)

![](images/fc2f907033bd6ead2892570b68dc5a99b07008c418b776477c279f520b04b250.jpg)  
Figure 4: An NDR search with GPT-5.2. The blue path passes through node 4 and reaches the best state, node 20. The dashed gray subtree starts at the locally stronger node 2 but never exceeds its score. Scores are validation accuracy.  
Figure 5: Skill length in characters. (a) Four tasks with deepseek-v4-flash. (b) NDR across targets. Ratios divide SkillOpt length by Atom-Search length. Gray markers show the initial skill. Qwen-4B denotes Qwen3.5-4B.

SkillOpt reaches 72.50 after 20 and stays there through edit 38. The tree uses additional evaluations to find those retained paths.

The SpreadsheetBench path is not monotonic. Validation accuracy is 52.50 after the second edit, falls to 49.50 after the third, and then rises to 73.75 after the fourth.

Figure 4 shows the same pattern in an NDR search with GPT-5.2. Node 2 immediately improves from 54.29 to 62.86; node 4 is about 2.9 points below its parent. No descendant of node 2 exceeds node 2, whereas the node 4 branch later reaches node 20 at 73.33, the best validation score in the tree. Node 4 receives 23 visits and node 2 receives four.

In 20 of 30 completed searches (67%), the best validation state has at least one ancestor that is not the highest-scoring state observed at the same depth. Such an ancestor need not be worse than its own parent, but a depth-wise immediate-score ranking would still prefer another branch.

## 4.5 COMPACT FINAL SKILLS

Figure 5 compares the final artifacts with SkillOpt and with the initial skill. For deepseek-v4-flash, SkillOpt’s final skills are 2.3–19.7 times as long across the four benchmarks. On NDR, the ratio is 2.3× to 3.3× across the three target models shown. AtomSearch does not optimize length, so the compact artifacts are an empirical outcome of the retained revision paths. In the ablation, the size drop appears as soon as contrastive atomic editing is added (16,536 to 2,352 characters; Table 3), so the reported task gains do not require continued rule accumulation.

Table 3: Cumulative ablation on SpreadsheetBench with deepseek-v4-flash. The first row is Base, the SkillOpt run reported in Table 1; each later row adds one component, and the last row is full AtomSearch, whose final skill is the one reported in Table 1. The initial skill has 1,594 characters. Bold marks the highest test score.
<table><tr><td>Contrastive atomic editing</td><td>Skill-State Tree Search</td><td>Outcome memory</td><td>Test (%)</td><td>Characters</td></tr><tr><td>一</td><td>一</td><td>一</td><td>68.93</td><td>16,536</td></tr><tr><td>√</td><td>一</td><td>一</td><td>75.71</td><td>2,352</td></tr><tr><td>√</td><td>√</td><td></td><td>77.14</td><td>2,755</td></tr><tr><td>√</td><td>√</td><td>√</td><td>78.21</td><td>2,936</td></tr></table>

## 4.6 COMPONENT ANALYSIS AND OPTIMIZATION COST

Table 3 adds components cumulatively on SpreadsheetBench. Contrastive atomic editing accounts for most of the change: relative to Base, it raises the test score from 68.93 to 75.71 and shrinks the skill from 16,536 to 2,352 characters. Adding Skill-State Tree Search raises the score to 77.14, and adding outcome memory gives 78.21 for the full method. The later two steps are smaller (1.43 and 1.07 points, or 4 and 3 of 280 test items) and come from one task–target setting.

Short retained paths require additional offline evaluation. In the 17 documented runs, the optimizer is called 16–45 times; Appendix B.2 reports item-evaluation counts and cache usage.

## 5 DISCUSSION AND LIMITATIONS

AtomSearch exchanges additional offline evaluation for exploration of alternative revision paths. This trade-off is most relevant when the resulting skill is reused across many executions. Adaptive rollout budgets and selective re-evaluation are natural next steps for reducing the search cost while preserving promising alternatives.

The separation between revision granularity and valuation horizon is deliberate. A small edit is easier to inspect, but its usefulness need not be visible at the same step. STS spends extra evaluations to preserve that distinction: the action remains local while later executable states can determine whether its path deserves more search. The traces in Section 4.4 show that such paths occur in the current runs.

Finite validation panels and stochastic execution limit the precision of small score differences, and selecting among many candidates can overfit the validation criterion (Cawley & Talbot, 2010). The component ablation covers one task–target setting, where the increments from search and outcome memory are small; extending it across tasks and evaluation budgets would show how they vary. NDR is a single in-house task, and its results measure offline accuracy only.

## 6 CONCLUSION

AtomSearch keeps skill revisions local while allowing later edit combinations to inform search. Contrastive atomic editing makes each action and its evidence explicit; Skill-State Tree Search preserves evaluated compositions as extensible states; outcome memory carries observed revision outcomes into later proposals. Across the evaluated settings, AtomSearch produces compact skills and improves several task–model pairs. Its search traces show that locally weaker states can lead to the best discovered skill, at the cost of additional offline evaluation. Attributable edits and compact skills also carry over to RSI, where an agent that keeps revising its own skills needs both.

## REFERENCES

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. GEPA: Reflective prompt evolution can outperform reinforcement learning. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html.

Cameron B. Browne, Edward Powley, Daniel Whitehouse, Simon M. Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in Games, 4(1):1–43, 2012. doi: 10.1109/TCIAIG.2012.2186810.

Gavin C. Cawley and Nicola L. C. Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11:2079–2107, 2010. URL https://jmlr.org/papers/v11/cawley10a.html.

Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. SearchQA: A new q&a dataset augmented with context from a search engine, 2017. URL https://arxiv.org/abs/1704.05179.

Yuru Feng, Yaoqi Chen, Beidi Zhao, Qianxi Zhang, Xinjiang Wang, Jianan Lu, Zhirui Wang, Shusen Xu, Zengzhong Li, and Qi Chen. SkillHEX: Improving agent skills via hypothesis-driven autonomous exploration and exploitation, 2026. URL https://arxiv.org/abs/2608. 05628.

Chrisantha Fernando, Dylan Sunil Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktaschel. Promptbreeder: Self-referential self-improvement via prompt evolution. In¨ Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 13481–13544. PMLR, 2024. URL https: //proceedings.mlr.press/v235/fernando24a.html.

Yudong Gao, Zongjie Li, Yuanyuan Yuan, Zimo Ji, Pingchuan Ma, and Shuai Wang. SkillReducer: Optimizing LLM agent skills for token efficiency, 2026. URL https://arxiv.org/abs/ 2603.29919.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. CRITIC: Large language models can self-correct with tool-interactive critiquing. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2305. 11738.

Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yujiu Yang. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2309.08532.

Linyang He, Qiyao Yu, Hanze Dong, Baohao Liao, Xinxing Xu, Micah Goldblum, Jiang Bian, and Nima Mesgarani. LiveMathematicianBench: A live benchmark for mathematician-level reasoning with proof sketches, 2026. URL https://arxiv.org/abs/2604.01754.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2408. 08435.

Chenyi Huang, Haoting Zhang, Jingxu Xu, Zeyu Zheng, and Yunduan Lin. Bilevel optimization of agent skills via monte carlo tree search, 2026. URL https://arxiv.org/abs/2604. 15709.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas Joshi, Hanna Moazam, Heather Miller, Matei

Zaharia, and Christopher Potts. DSPy: Compiling declarative language model calls into stateof-the-art pipelines. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.03714.

Levente Kocsis and Csaba Szepesvari. Bandit based monte-carlo planning. In ´ Machine Learning: ECML 2006, pp. 282–293. Springer, 2006.

Derek Koh, Jinghui Mo, Benjamin H. Le, Jiening Zhan, Baofen Zheng, Kevin Bevis, Nathaniel C. Owen, Lauren Elizabeth Charney, Wenqiong Liu, and Jingwei Wu. Contrastive reflection for iterative prompt optimization, 2026. URL https://arxiv.org/abs/2606.30840.

Yuxuan Liu, Zhaochen Su, Lingyun Xie, Yuhao Zhang, Qing Zong, Jiahe Guo, Zhongwei Xie, Yiyan Ji, Yauwai Yim, Hongyu Luo, Xiyu Ren, Chenyu Ruan, Haoran Li, and Yangqiu Song. SkillRevise: Improving llm-authored agent skills via trace-conditioned skill revision, 2026. URL https://arxiv.org/abs/2606.01139.

Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, and Jie Tang. SpreadsheetBench: Towards challenging real world spreadsheet manipulation. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-3007. URL https://arxiv.org/abs/2406.14991.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://arxiv.org/abs/2303.17651.

Krista Opsahl-Ong, Michael J. Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab. Optimizing instructions and demonstrations for multi-stage language model programs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 9340–9366, 2024. doi: 10.18653/v1/2024.emnlp-main.525. URL https:// aclanthology.org/2024.emnlp-main.525/.

Savvas Petridis, Ben Wedin, Ann Yuan, James Wexler, and Nithum Thain. ConstitutionalExperts: Training a mixture of principle-based prompts. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pp. 574–582, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-short.52. URL https://aclanthology.org/2024.acl-short.52/.

Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with “gradient descent” and beam search. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 7957–7968, 2023. doi: 10.18653/ v1/2023.emnlp-main.494. URL https://aclanthology.org/2023.emnlp-main. 494/.

Junhao Shen, Zhanqiu Zhang, Yiwen Guo, and Hong Cheng. Progressive agent skill generation via reinforcement learning, 2026. URL https://arxiv.org/abs/2608.01678.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.

Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Test-time learning with adaptive memory, 2025. URL https://arxiv.org/abs/ 2504.07952.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. URL https://arxiv.org/abs/2305.16291.

Xinyuan Wang, Chenxi Li, Zhen Wang, Fan Bai, Haotian Luo, Jiayou Zhang, Nebojsa Jojic, Eric P. Xing, and Zhiting Hu. PromptAgent: Strategic planning with language models enables expertlevel prompt optimization. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=22pyNMuIoa.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 63897–63911. PMLR, 2025. URL https: //proceedings.mlr.press/v267/wang25bx.html.

Shirley Wu, Shiyu Zhao, Qian Huang, Kexin Huang, Michihiro Yasunaga, Kaidi Cao, Vassilis N. Ioannidis, Karthik Subbian, Jure Leskovec, and James Zou. AvaTaR: Optimizing LLM agents for tool usage via contrastive reasoning. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/hash/2db8ce969b000fe0b3fb172490c33ce8-Abstract-Conference. html.

Siwei Wu, Jincheng Ren, Yizhi Li, Haau-Sing Li, Chengran Yang, Yuxuan Zhang, Weicheng Gu, Jian Yang, Riza Batista-Navarro, Chuanyi Zhang, Xianglong Liu, Ming Zhou, Bryan Dai, and Chenghua Lin. ModularRSI: Modular and generalizable recursive harness self-improvement, 2026. URL https://arxiv.org/abs/2609.14857.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2309.03409.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. SkillOpt: Executive strategy for self-evolving agent skills. In Advances in Neural Information Processing Systems, 2026. URL https://arxiv.org/abs/2605.23904.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://arxiv. org/abs/2305.10601.

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. TextGrad: Automatic “differentiation” via text, 2024. URL https://arxiv. org/abs/2406.07496.

Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (STOP): Recursively self-improving code generation, 2023. URL https://arxiv.org/ abs/2310.02304.

Barry Zhang, Keith Lazuka, and Mahesh Murag. Equipping agents for the real world with agent skills. Anthropic Engineering, October 2025a. URL https://www.anthropic.com/engineering/ equipping-agents-for-the-real-world-with-agent-skills. Published October 16, 2025.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin Godel machine: Open-¨ ended evolution of self-improving agents, 2025b. URL https://arxiv.org/abs/2505. 22954.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation. In The Thirteenth International Conference on Learning Representations, 2025c. URL https://openreview.net/forum?id= z5uVAKwmjf.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou,

and Kunle Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. In International Conference on Learning Representations, 2026. URL https: //arxiv.org/abs/2510.04618.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. ExpeL: LLM agents are experiential learners. Proceedings of the AAAI Conference on Artificial Intelligence, 38(17):19632–19642, 2024. doi: 10.1609/aaai.v38i17.29936. URL https: //ojs.aaai.org/index.php/AAAI/article/view/29936.

Zhi Zheng, Zhuoliang Xie, Zhenkun Wang, and Bryan Hooi. Monte carlo tree search for comprehensive exploration in LLM-based automatic heuristic design. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 78338–78373. PMLR, 2025. URL https://proceedings.mlr.press/ v267/zheng25o.html.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In Proceedings of the 41st International Conference on Machine Learning, 2024. URL https://arxiv.org/ abs/2310.04406.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2211.01910.

## A OPTIMIZATION PROCEDURE

Algorithm 1 Skill-State Tree Search with contrastive atomic editing and outcome memory   
Require: Initial skill $S _ { 0 } ,$ search budget T   
1: Initialize tree T with root $S _ { 0 }$ and outcome memory M   
2: for $t = 1 , \dots , T$ do   
3: Select an expandable state $S$ using UCT under PW eligibility   
4: Obtain contrastively grounded candidate edits for S using current evidence and $\mathcal { M }$ if needed   
5: if no valid candidate remains then   
6: Back up the terminal outcome and continue   
7: end if   
8: Apply the highest-ranked untried primitive edit to obtain $S ^ { \prime }$   
9: Evaluate $S ^ { \prime }$ on $\mathcal { D } _ { \mathrm { n a v } }$   
10: Build and evaluate a short rollout from remaining ranked edits   
11: if the rollout state scores higher than $S ^ { \prime }$ then   
12: $S _ { \mathrm { k e e p } } $ rollout state   
13: else   
14: $S _ { \mathrm { k e e p } }  S ^ { \prime }$   
15: end if   
16: Attach $S _ { \mathrm { k e e p } }$ and back up $V ( S _ { \mathrm { k e e p } } ) - V ( S _ { \mathrm { 0 } } )$   
17: Append the observed revision outcome(s) to M   
18: end for   
19: return the recursively selected robust child

## B EXPERIMENTAL DETAILS

## B.1 DATA SPLITS AND SEARCH CONFIGURATION

Each benchmark is split once into a learn set (source of the failed and successful executions contrasted by the proposer), a validation set (used to accept edits and select the returned skill), and a held-out test set that is read only for reporting. Table 4 lists the sizes.

Table 4: Split sizes (items) and contrastive minibatch size per benchmark. Minibatches use approximately one third of the learn set. LiveMath: LiveMathematicianBench.
<table><tr><td>Benchmark</td><td>Learn</td><td>Validation</td><td>Test</td><td>Minibatch</td></tr><tr><td>NDR</td><td>100</td><td>105</td><td>100</td><td>35</td></tr><tr><td>SpreadsheetBench</td><td>80</td><td>40</td><td>280</td><td>27</td></tr><tr><td>SearchQA</td><td>400</td><td>200</td><td>1400</td><td>134</td></tr><tr><td>LiveMath</td><td>35</td><td>18</td><td>124</td><td>12</td></tr></table>

The reference search configuration uses $k = 5$ candidate specifications per proposal round with an edit budget of 4, maximum branch factor $B = 6 $ , maximum depth 12, UCT exploration constant $c _ { \mathrm { u c t } } = 0 . 0 5$ , progressive widening with $\alpha = 0 . 5$ and $c _ { \mathrm { p w } } = 1 . 0$ , and rollouts of at most two edits. Runs use 20 or 30 attempts (Table 5). Target models are queried with reasoning effort medium and a completion limit of 16,384 tokens (24,576 for Qwen3.6-35B-A3B on NDR). Open-weight targets are served locally with vLLM.

## B.2 EVALUATION CACHE AND OPTIMIZATION COST

Evaluation results are cached per (skill state, item, draw). Repeated reads of an unchanged state are served from the cache and are not counted against the evaluation budget; deliberate re-reads use a new draw index and are charged. Table 5 reports the budget spent by each of the 17 documented runs. Optimizer calls range from 16 to 45. Cache hit rates are 60–69% except for NDR with GPT-5.2 (15%); the logs summarized here do not establish the cause of that difference.

Table 5: Cost of 17 documented AtomSearch runs. Item evaluations are charged to the budget and are not necessarily individual API requests; cache hit rate is the share of evaluation requests served from the cache. LiveMath: LiveMathematicianBench.
<table><tr><td>Target</td><td>Benchmark</td><td>Attempts</td><td>Item evals.</td><td>Optimizer calls</td><td>Cache hit rate</td></tr><tr><td rowspan="5">deepseek-v4-flash</td><td>NDR</td><td>20</td><td>4,345</td><td>16</td><td>65%</td></tr><tr><td>SpreadsheetBench</td><td>30</td><td>3,360</td><td>29</td><td>64%</td></tr><tr><td>SearchQA</td><td>20</td><td>10,600</td><td>19</td><td>64%</td></tr><tr><td>LiveMath</td><td>20</td><td>923</td><td>19</td><td>65%</td></tr><tr><td>NDR</td><td>20</td><td>6,865</td><td>20</td><td>60%</td></tr><tr><td rowspan="4">GPT-5.2</td><td>SearchQA</td><td>30</td><td>17,200</td><td>35</td><td>65%</td></tr><tr><td>LiveMath</td><td>30</td><td>1,474</td><td>27</td><td>64%</td></tr><tr><td>NDR</td><td>30</td><td>10,760</td><td>24</td><td>15%</td></tr><tr><td>SpreadsheetBench</td><td>30</td><td>2,920</td><td>36</td><td>68%</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>SearchQA LiveMath</td><td>30 30</td><td>14,000 1,295</td><td>27</td><td>66%</td></tr><tr><td></td><td></td><td></td><td>29</td><td>67%</td></tr><tr><td>NDR</td><td>20</td><td>4,755</td><td>23</td><td>68%</td></tr><tr><td>SearchQA LiveMath</td><td>30 30</td><td>15,400 1,348</td><td>25</td><td>65%</td></tr><tr><td rowspan="4">Qwen3.6-35B-A3B</td><td></td><td></td><td></td><td>45</td><td>69%</td></tr><tr><td>SpreadsheetBench</td><td>30</td><td>3,520</td><td>23</td><td>62%</td></tr><tr><td>SearchQA LiveMath</td><td>30</td><td>15,800</td><td>28</td><td>66%</td></tr><tr><td></td><td>30</td><td>1,440</td><td>26</td><td>65%</td></tr></table>

## C VALIDATION-TO-TEST COMPARISON DETAILS

Table 2 reports six comparisons, comprising SpreadsheetBench with deepseek-v4-flash and NDR with five targets. This analysis includes comparisons with an executed SkillOpt baseline, excluded validation gains below five percentage points, and excluded LiveMathematicianBench, whose validation panel has 18 items. We treat these six comparisons as a descriptive subset rather than an estimate over the full task–model matrix.

Each gain is measured relative to that run’s recorded initial skill score. Initial evaluation scores can differ between methods, so the ratios do not normalize away all differences in initialization, evaluation draws, or search budget. A ratio above 100% means the recorded test gain exceeds the recorded validation gain; it does not mean that accuracy exceeds 100%. The ratio can also be large because its denominator is small. We therefore report both absolute gains beside each ratio and keep final test accuracy as the primary result.

The reported median ratios are 123.96% for AtomSearch and 82.09% for SkillOpt before rounding. These values describe the recorded point estimates. Repeated evaluation of fixed checkpoints would be needed to quantify their uncertainty and separate selection effects from differences in the learned skills.