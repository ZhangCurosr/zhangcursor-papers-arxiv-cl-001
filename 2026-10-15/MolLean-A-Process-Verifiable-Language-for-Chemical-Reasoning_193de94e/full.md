# MolLean: A Process-Verifiable Language for Chemical Reasoning

Jiatong Li & Yatao Bian† National University of Singapore †Corresponding to: ybian@nus.edu.sg https://github.com/phenixace/Open-MolLean

## Abstract

Chemical chain-of-thought can contain hallucinated structural and property claims. We thereby introduce MolLean, a process-verifiable language that grounds chemical reasoning in an explicit molecular state. A layered Lean-based kernel checks state transitions and assertions over molecular readouts, from primitive graph edits to compound transformations. Meanwhile, a typed witness protocol separates native structural checks from externally supplied measurements. We further introduce MolLean-MCP, a stateful tool interface that exposes these transitions, checks, and readouts during generation: the model receives feedback to decide its next action, and rejected actions leave the state unchanged. MolLean supports one-pass trace execution, whole-trace refinement, and interactive reasoning through MolLean-MCP. We evaluate these three modes with direct chain-of-thought on 8,800 matched examples across 12 tasks spanning molecular generation, editing, optimization, and reactions. With Qwen3.6-27B, whole-trace refinement outperforms direct chainof-thought on all 12 tasks. MolLean-MCP complements whole-trace refinement with stepwise verification feedback, reaching better performance in reaction tasks. Together, these results highlight an executable foundation for chemical reasoning, in which explicit chemical commitments can be checked and revised rather than remaining solely in free-form explanations.

## 1 Introduction

Chain-of-thought prompting makes a model's intermediate reasoning visible (Wei et al., 2022). In chemistry, this is appealing as chemists could examine the intermediate reasoning steps to improve the explainability of the final answer (Wang et al., 2026; Narayanan et al., 2026; Li et al., 2025; Zhao et al., 2025). However, a model can invoke a functional group absent from a molecule, lose track of a bond, or claim a property improvement without establishing it. Analyses of chemical reasoning find fabricated structural claims even alongside correct answers (Li et al., 2026b), and broader evaluations reveal uneven chemical competence and overconfidence (Mirza et al., 2025). These errors concern the stated chemistry itself, a failure mode complementary to causal unfaithfulness, where an explanation misrepresents how the answer was produced (Turpin et al., 2023).

We therefore ask what must a chemical reasoning language represent for its steps to be verifiable? Lean offers a precedent: a model proposes a derivation and a proof assistant checks its formal obligations (Moura & Ullrich, 2021; Jiang et al., 2023; Yang et al., 2023). Chemical reasoning combines graph transformations, structural observations, empirical estimates, and choices among plausible candidates. Chemistry tools already support capable agents (M. Bran et al., 2024; Boiko et al., 2023); the key step is to give successive operations and claims a shared molecular referent and an explicit acceptance condition.

We introduce MolLean, a process-verifiable language grounded in a maintained molecular state (Figure 1). A transition constructs, modifies, or reorganizes represented molecules. A readout obtains information without changing them, and an assertion tests a claim against that information. The model selects actions and obligations; a Lean-based kernel checks admissible transitions and native structural predicates. Descriptor- and knowledge-dependent assertions obtain typed witnesses from external providers. These distinctions let construction, transformation, and observation share one language while each keeps its own form of evidence.

The state does more than record the output: it determines which atoms a later edit addresses, which molecule a property readout measures, and whether an accepted prefix remains usable after a failed proposal. MolLean supports primitive edits, construction macros, joint transformations, and candidate introduction at different granularities. An accepted edit supplies the referent for subsequent checks, and a rejected edit leaves the maintained state intact. Verification thus applies directly to the model's declared commitments.

We expose the same language through whole-trace and interactive interfaces. MolLean One-pass submits a complete trace; MolLean Refinement revises a rejected trace from the initial state. MolLean-MCP lets the model inspect the current state, attempt an operation, and use immediate feedback to choose its next action. External feedback is central to revision: tool-grounded criticism guides reasoning more reliably than intrinsic self-correction (Huang et al., 2024; Gou et al., 2024). MolLean specifies what that feedback refers to and when it commits a change.

Our contributions are as follows:

• A state-grounded chemical reasoning language and verification contract. MolLean unifies molecular transitions, readouts, and assertions, with explicit boundaries between native structural checks and externally witnessed claims.

• Two feedback regimes over shared execution semantics. Whole-trace refinement revises complete traces from the input; MolLean-MCP interacts with the maintained state and commits only accepted edits.

• An evaluation on 8,800 matched examples across 12 tasks. With Qwen3.6-27B, refinement improves over direct chain-of-thought on every task, and MCP reaches the highest reaction accuracy. On ChEBI-20, refined traces reach 79.0% soundness and 33.1% answer accuracy, identifying task fidelity as the next challenge beyond repairing rejected steps. Qwen3-4B gains substantially on structural tasks.

## 2 MolLean

MolLean is a reasoning language built around a simple contract: an operation refers to the current molecular state, verification decides whether it is admissible, and only an accepted transition commits a new state. Readouts and assertions observe that same state without editing it. Lean implements the chemical core, while Python handles parsing, state transport, reference resolution, and inference control (Figure 1).

## 2.1 A Language Grounded in Molecular State

Execution begins at $S _ { 0 } = \mathrm { i n i t } ( x )$ for task input x. The state contains active molecules, a leaving pool, atom references, and bookkeeping counters. Molecules are graphs labelled with elements, bonds, charges, radicals, and aromaticity. The leaving pool retains fragments that are excluded from the answer and tracked in bookkeeping. Generation can start empty; other tasks load the supplied molecules.

A transition attempts to change the state. A readout observes it without changing its molecular contents, and an assertion constrains that readout, for example by requiring a count or property improvement. Complete traces express these constraints; the interactive interface also exposes valuereturning queries. Edits form one class of transitions, alongside construction, candidate introduction, and fragment separation.

Let $a _ { t }$ be a declared operation, Γ the verifier configuration, and W the external provider. We write one execution step as

$$
V _ { \Gamma , W } ( S _ { t - 1 } , a _ { t } ) = { \left\{ \begin{array} { l l } { \operatorname { a c c e p t } ( S _ { t } , o _ { t } ) } & { { \mathrm { i f ~ } } a _ { t } { \mathrm { ~ p a s s e s ~ i t s ~ c h e c k s ~ i n ~ } } S _ { t - 1 } , \quad } \\ { \operatorname { r e j e c t } ( r _ { t } ) } & { { \mathrm { o t h e r w i s e ; ~ t h e ~ c u r r e n t ~ s t a t e ~ r e m a i n s ~ } } S _ { t - 1 } , } \end{array} \right. }\tag{1}
$$

where $o _ { t }$ is the observation or verdict and $r _ { t }$ is the reason for rejection. Accepted readouts and assertions return $S _ { t } ~ = ~ S _ { t - 1 }$ , so only accepted transitions commit a new state. Only declared operations enter this contract.

![](images/e03e86e9e96fee8c64f1601376eb147b18ff1f71eeba8e4287fc439d8e981991.jpg)  
Figure 1: MolLean's language, execution contract, and inference interfaces. (a) The state, transition families, and read-only queries; native predicates are decided in Lean, while witnessed claims apply a Lean predicate to a typed provider response. (b) The adapter holds S0 = init(x) sends each operation to the Lean kernel, and commits only accepted successors; witness requests and responses pass through the adapter. In the example, generation starts empty, admits ethanol as S1, and attaches methyl to obtain 1-propanol as S2; a false carbon count is rejected and S2 is unchanged. (c) One-pass executes one trace; refinement rewrites a rejected trace from S0; MCP continues from the current state. Dataset references are used only by an independent scorer.

Joining molecules merges their atom-index spaces. Whole-trace execution remaps later references after tracked primitive merges; MCP supplies the current atom map for subsequent calls.

## 2.2 State Transitions at Different Granularities

Seven primitives add or remove atoms, form or break bonds, and update bond order, charge, or radicals. They check preconditions and affected valences. Intermediates may be under-saturated to permit construction, and connectivity is enforced by the relevant operation or terminal policy.

Construction macros such as Chain, Ring, and AttachGroup reduce index manipulation. Compound actions express joint bond changes, proton transfer, and fragment separation. Fragment splits a bridge bond into active molecules; E ject moves a separated component into the leaving pool. Named reactions expand into these operations. Most expansions execute in Lean; Python fragment adapters submit operations for the same checks.

Granularity determines the verification unit. A Concerted bundle applies edits atomically and checks the resulting valences once all edits apply. propose introduces a structurally checked candidate in a single step.

Bookkeeping covers graph atoms, formal charge, and radical parity across both pools. Addition, removal, and candidate introduction update the ledger accordingly. Charge and radical balance are closing obligations. Terminal realizability checks valence, connectivity, aromaticity, and closed-shell rules.

## 2.3 State Readouts and External Witnesses

A readout associates a query with the state being examined. MolLean supports native checks for seven structural families: atom counts, bond counts, ring counts, molecular formula, aromaticity built-in functional-group patterns, and retained R/S stereocenter annotations. Element counts include implicit as well as explicit hydrogen, while molecular-formula checks require all present elements. Ring counts use a graph-theoretic definition

MCP distinguishes measure, which returns a value, from check, which tests a proposed value. The reader obtains simple quantities from serialized state and resolves pattern-sensitive counts through Lean predicates, returning a value only when a compatible one is confirmed.

RDKit supplies descriptors, fingerprints, and molecular comparisons (RDKit contributors, 2026); knowledge providers supply lookup-dependent information. For an assertion $^ { a , }$ a typed witness protocol separates obtaining evidence from checking the claim:

$$
q = \mathrm { r e q u e s t } ( { \cal S } , a ) , \qquad w = W ( q ) , \qquad \mathrm { a c c e p t } ( a ) \iff P _ { a } ( w ) = 1 .\tag{2}
$$

Lean requests typed fields and applies the predicate. Built-in native predicates compute directly;   
other supported functional-group patterns use RDKit counts.

For PropertyChange and FGCountDelta, the provider binds the baseline to the task's initial molecule. These checks use only the public task specification.

## 2.4 Process Soundness and Task Fidelity

For a parsed MolLean trace $\tau = ( a _ { 1 } , \dots , a _ { T } )$ and closing obligations $G _ { \Gamma }$ , we define trace soundness operationally:

$$
\begin{array} { r } { s _ { \Gamma , W } ( \tau ; x ) = \mathbf { 1 } [ \mathrm { a l l ~ d e c l a r e d ~ t r a c e ~ o p e r a t i o n s ~ a r e ~ a c c e p t e d ~ a n d ~ } G _ { \Gamma } ( S _ { T } ) = 1 ] . } \end{array}\tag{3}
$$

Missing or unparseable traces count as unsound. For $\mathbf { M C P } ,$ soundness applies to the accepted trace, excluding read-only queries and rejected proposals.

This contract concerns the represented process. Task fidelity asks whether the initial representation, selected referents, and expressed obligations capture the input instruction. For example, a trace that verifiably lowers QED is sound with respect to its own obligations and unfaithful to an instruction to raise QED: it executes the wrong objective correctly.

Answer correctness $c ( \hat { y } ; x , y )$ is a further, independently scored outcome, where $\hat { y }$ is obtained from the reached state and y denotes any held-out reference. Thus

$$
s _ { \Gamma , W } ( \tau ; x ) = 1 \ \not \Rightarrow \ c ( \hat { y } ; x , y ) = 1 .\tag{4}
$$

We therefore report process soundness and task scores as separate measures.

The implementation combines proofs and executable checks. Lean proves soundness properties for valence and connectivity validation, a conditional aromaticity result, and six native assertion boundaries (Appendix A).

## 2.5 Whole-Trace and Interactive Inference

MolLean One-pass executes a generated complete trace until acceptance or the first rejection. MolLean Refinement starts from that same completion, returning rejection feedback and indexed state to the model. It revises rejected traces within a fixed budget, restarting each at $S _ { 0 }$ and stopping once accepted. Thus the model can replace its entire plan after a failure.

MolLean-MCP exposes transitions, checks, and readouts as stateful tools, supplied to the model as function-calling schemas within the inference loop. The model can inspect the state before choosing its next operation; rejected proposals leave it unchanged. fini sh invokes closing gates. MCP uses the language interactively; One-pass and Refinement submit complete programs.

Verification compares only against the input or a self-proposed candidate; dataset references enter only the independent scorer (Section 3). Each task declares its initialization, allowed operations, closing obligations, and scorer.

## 3 Experiments

We ask whether MolLean improves task performance, how accepted traces relate to correct answers, and when interactive verification is useful. The main comparison uses Qwen3.6-27B; an additional Qwen3-4B evaluation examines whole-trace inference with a smaller model.

Table 1: Qwen3.6-27B task performance on matched examples. Refine permits eight whole-trace revisions; MCP uses interactive verification. Bold marks the highest score. † denotes 100× successweighted similarity.
<table><tr><td>Task</td><td>N</td><td>CoT</td><td>One-pass</td><td>Refine</td><td>MCP</td></tr><tr><td>Atom counts</td><td>500</td><td>42.0</td><td>61.4</td><td>88.8</td><td>82.6</td></tr><tr><td>Bond counts</td><td>500</td><td>20.8</td><td>39.8</td><td>69.0</td><td>56.2</td></tr><tr><td>Functional groups</td><td>500</td><td>25.6</td><td>38.8</td><td>41.8</td><td>37.2</td></tr><tr><td>Add component</td><td>500</td><td>55.8</td><td>93.6</td><td>100.0</td><td>100.0</td></tr><tr><td>Delete component</td><td>500</td><td>85.0</td><td>94.6</td><td>97.0</td><td>97.0</td></tr><tr><td>Substitute component</td><td>500</td><td>81.0</td><td>94.2</td><td>96.0</td><td>95.4</td></tr><tr><td>logP optimization †</td><td>500</td><td>24.0</td><td>65.7</td><td>69.5</td><td>70.1</td></tr><tr><td>MR optimization †</td><td>500</td><td>22.5</td><td>60.2</td><td>66.1</td><td>65.6</td></tr><tr><td>QED optimization †</td><td>500</td><td>24.8</td><td>41.1</td><td>59.0</td><td>49.4</td></tr><tr><td>Forward reaction</td><td>500</td><td>35.0</td><td>23.6</td><td>62.8</td><td>66.4</td></tr><tr><td>Retrosynthesis</td><td>500</td><td>3.0</td><td>11.6</td><td>20.8</td><td>27.4</td></tr><tr><td>ChEBÍ-20 text to molecule</td><td>3,300</td><td>18.1</td><td>25.3</td><td>33.1</td><td>23.5</td></tr></table>

## 3.1 Tasks, Protocols, and Metrics

All protocols are evaluated on the same 8,800 examples across 12 task types: 500 examples for each of nine S2-Bench tasks (Li et al., 2026a), 500 forward and 500 retrosynthesis examples from USPTO-50k (Schneider et al., 2016), and the complete 3,300-example ChEBI-20 test set (Edwards et al., 2021; 2022). Reaction classes are supplied to every protocol, and USPTO uses a deterministic reaction-level split. Invalid answers and unsuccessful episodes remain in the denominator.

Protocols. Direct CoT generates free-form reasoning and a final SMILES answer. MolLean One-pass emits one complete trace without feedback. MolLean Refinement starts from the same one-pass completion and allows up to eight whole-trace refinements. MolLean-MCP constructs a trace through stateful tools, with up to two attempts and 30 counted calls per attempt. We abbreviate these protocols as CoT, One-pass, Refine, and MCP. All MolLean results use the same verifier version; Appendix D.2 details the inference settings.

Metrics. $S ^ { 2 }$ -Bench generation and editing use structural-count and functional-group-change success criteria. For optimization, we report success-weighted input similarity,

$$
\operatorname { s c o r e } _ { \mathrm { o p t } } ( \hat { y } , x ) = \mathbf { 1 } [ \mathrm { r e q u e s t e d p r o p e r t y i m p r o v e s } ] \operatorname { T a n i m o t o } ( f ( \hat { y } ) , f ( x ) ) ,\tag{5}
$$

where $f$ is a radius-2, 2,048-bit Morgan fingerprint (Rogers & Hahn, 2010). Direction-only success is reported separately in Appendix D. ChEBI-20 compares the first InChIKey block, which encodes molecular connectivity (Heller et al., 2015); reactions require equality of the complete sets of component skeleton keys. We report percentages for binary scores and 100 × Equation 5 for optimization, task by task. Trace soundness (Equation 3) is reported alongside answer correctness.

## 3.2 Whole-Trace Refinement Improves Task Performance

Refinement improves over Direct CoT on all 12 task types (Table 1), with gains of 12.0–48.2 percentage points under each task's binary success criterion. For optimization, this range uses direction-only success, and similarity-weighted scores show the same gains. Atom and bond counts improve by 46.8 and 48.2 points, and component addition by 44.2. Paired bootstrap intervals for the primary Refine-minus-CoT score exclude zero in every task (Table 2).

One-pass already improves over CoT on 11 tasks. On forward reactions, One-pass (23.6%) trails CoT (35.0%), and refinement raises accuracy to 62.8%, showing that feedback is most valuable when the initial plan misses the verifier's obligations. Improvements extend to reference-based tasks, with ChEBI-20 accuracy increasing from 18.1% to 33.1% and retrosynthesis from 3.0% to 20.8%. These results identify two complementary benefits: expressing a candidate as an executable trace, and using a failed check to revise that candidate.

Recovery versus loss. The shared one-pass completion allows a paired comparison. If R counts unsuccessful one-pass answers that become successful after refinement and L counts successes that

![](images/3766b7352bc1e3a51f8808c12628c5d6f9b6c88cff7df66baaaf2f74759a6e80.jpg)  
Below the dashed line: optimization direction success in (a, b), similarity-weighted score in (c)  
Figure 2: Repair, specification gaps, and interactive verification. (a) Refine recovers more Onepass failures than it loses successes, as fractions of each task's examples. (b) Joint Refine outcomes separate accepted traces from correct answers. (c) MCP-minus-Refine primary-score differences show task-dependent benefits. Optimization appears below the dashed line: (a,b) use direction-only success, while (c) uses similarity-weighted score points. Other task differences are percentage points.

become failures, then

$$
{ \mathrm { A c c } } _ { \mathrm { r e f i n e } } - { \mathrm { A c c } } _ { \mathrm { o n e - p a s s } } = ( R - L ) / N .\tag{6}
$$

Forward prediction recovers 207 answers and loses 11; retrosynthesis recovers 48 and loses two. ChEBI-20 recovers 290 and loses 33, for a net gain of 257 correct answers. The full paired counts appear in Appendix E.4. Among initially unsuccessful answers, the recovered fraction is 54.2% for forward prediction, 10.9% for retrosynthesis, and 11.8% for ChEBI-20. The remaining failures concentrate on tasks that require identifying a particular reference structure.

Recovery outweighs loss on every task (Figure 2a). Accepted one-pass traces keep their answers, as the refinement stopping rule prescribes, so feedback concentrates on repairing rejected commitments.

Revision budget. Figure 3 tracks acceptance and answer score after at most k revisions for all 12 tasks, with k = 0 the shared One-pass completion. Most gains arrive within two revisions: forward-reaction accuracy rises from 23.6% at k = 0 to 58.0% at k = 2 and 62.8% at k = 8, and component addition reaches 100.0% by k = 2. On structural generation, acceptance and answer score rise together, with atom counts reaching 93.4% acceptance and 88.8% accuracy. On retrosynthesis and ChEBI-20, acceptance keeps rising between k = 2 and k = 8, from 57.4% to 88.2% and from 67.3% to 79.0%, while accuracy moves from 20.4% to 20.8% and from 31.1% to 33.1%. Later revisions on these tasks mainly repair expressed obligations, which Section 3.3 examines.

## 3.3 Soundness Separates Execution from Task Fidelity

Figure 2b separates acceptance from task success. Table 2 reports soundness for every protocol. Refinement raises soundness over One-pass on all 12 tasks, and MCP matches or exceeds Refinement on 10 of them, reaching 98.6% on forward reactions and 98.0% on retrosynthesis. The relation between soundness and success depends on what the task asks the model to express.

At the final refinement budget, retrosynthesis reaches 88.2% trace soundness and 20.8% exact setmatch accuracy; ChEBI-20 reaches 79.0% soundness and 33.1% accuracy; and functional-group generation reaches 97.6% acceptance and 41.8% benchmark success. In these tasks, the declared obligations capture part of the instruction, and native functional-group patterns follow MolLean's own definitions.

![](images/48dc43c583dd9d60e59c67b77088aa21d6af1017351217dec46f1ffe30f3a47c.jpg)  
Figure 3: Whole-trace refinement across revision budgets for all 12 tasks. Each panel uses the complete Qwen3.6-27B task cohort (N = 500; ChEBI-20 N = 3,300). Solid curves show trace acceptance and dashed curves the primary answer score; † marks 100× success-weighted similarity k = 0 is One-pass and k = 8 is the Refine column of Table 1.

Table 2: Qwen3.6-27B trace soundness and uncertainty. Soundness is the percentage of examples with an accepted trace (Equation 3); intervals are paired 95% bootstrap intervals (4,000 resamples) for the primary Refine-minus-CoT score.
<table><tr><td rowspan="2">Task</td><td colspan="3">Soundness (%)</td><td rowspan="2">Refine – CoT 95% interval</td></tr><tr><td>One-pass</td><td>Refine</td><td>MCP</td></tr><tr><td>Atom counts</td><td>61.6</td><td>93.4</td><td>96.6</td><td>[42.2, 51.6]</td></tr><tr><td>Bond counts</td><td>52.8</td><td>84.8</td><td>90.4</td><td>[43.8, 52.8]</td></tr><tr><td>Functional groups</td><td>84.0</td><td>97.6</td><td>98.2</td><td>[12.8, 19.6]</td></tr><tr><td>Add component</td><td>93.6</td><td>100.0</td><td>100.0</td><td>[39.6, 48.6]</td></tr><tr><td>Delete component</td><td>85.2</td><td>99.8</td><td>98.8</td><td>[8.8, 15.2]</td></tr><tr><td>Substitute component</td><td>95.4</td><td>99.8</td><td>100.0</td><td>[12.0, 18.2]</td></tr><tr><td>logP optimization</td><td>90.2</td><td>100.0</td><td>100.0</td><td>[42.7, 48.4]</td></tr><tr><td>MR optimization</td><td>86.8</td><td>99.8</td><td>100.0</td><td>[40.7, 46.4]</td></tr><tr><td>QED optimization</td><td>57.4</td><td>94.0</td><td>92.8</td><td>[31.3, 37.1]</td></tr><tr><td>Forward reaction</td><td>23.8</td><td>88.2</td><td>98.6</td><td>[22.8, 32.8]</td></tr><tr><td>Retrosynthesis</td><td>24.4</td><td>88.2</td><td>98.0</td><td>[14.2, 21.4]</td></tr><tr><td>ChEBI-20 text to molecule</td><td>36.1</td><td>79.0</td><td>80.8</td><td>[13.5, 16.5]</td></tr></table>

For ChEBI-20, 1,075 outputs are both sound and correct, 1,533 are sound but incorrect, 17 are correct without certification, and 675 satisfy neither criterion. These exhaustive outcomes distinguish process acceptance from task success (Appendix E.4). Atom-count generation shows the opposite profile: 444 of 467 accepted traces meet its task criterion, compared with 1,075 of 2,608 on ChEBI-20. A count maps directly to a predicate, whereas a textual description requires the model to select and combine the right structural obligations. Instruction-to-constraint grounding is therefore the main lever for further gains, as the salt example in Section 4 illustrates.

## 3.4 Interactive Verification Strengthens Reaction Prediction

Figure 2c shows the signed differences between the two interfaces. MCP achieves the best reaction results, with 66.4% forward-reaction and 27.4% retrosynthesis accuracy against 62.8% and 20.8% for Refinement. Refinement leads on atom counts (88.8% versus 82.6%), bond counts (69.0% versus 56.2%), QED optimization (59.0 versus 49.4 weighted-score points), and ChEBI-20 (33.1% versus 23.5%), and both protocols reach 100.0% on component addition and 97.0% on component deletion,

Table 3: Qwen3-4B task performance. The tasks and example IDs match the main evaluation. † denotes 100× success-weighted similarity; other scores are success percentages.
<table><tr><td>Task</td><td>N</td><td>CoT</td><td>One-pass</td><td>Refine</td></tr><tr><td>Atom counts</td><td>500</td><td>15.8</td><td>39.6</td><td>56.8</td></tr><tr><td>Bond counts</td><td>500</td><td>9.6</td><td>13.4</td><td>22.4</td></tr><tr><td>Functional groups</td><td>500</td><td>11.8</td><td>12.2</td><td>16.6</td></tr><tr><td>Add component</td><td>500</td><td>13.4</td><td>41.2</td><td>91.4</td></tr><tr><td>Delete component</td><td>500</td><td>59.0</td><td>56.4</td><td>64.2</td></tr><tr><td>Substitute component</td><td>500</td><td>15.2</td><td>21.6</td><td>53.4</td></tr><tr><td>logP optimization †</td><td>500</td><td>14.7</td><td>29.6</td><td>60.0</td></tr><tr><td>MR optimization †</td><td>500</td><td>19.4</td><td>15.3</td><td>32.0</td></tr><tr><td>QED optimization †</td><td>500</td><td>6.8</td><td>11.5</td><td>31.2</td></tr><tr><td>Forward reaction</td><td>500</td><td>1.4</td><td>2.2</td><td>1.6</td></tr><tr><td>Retrosynthesis</td><td>500</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>ChEBÍ-20 text to molecule</td><td>3,300</td><td>0.6</td><td>0.8</td><td>1.4</td></tr></table>

Whole-trace refinement can revise an entire plan, and MCP conditions each action on the maintained state and immediate feedback. Immediate feedback helps the model maintain references across a sequence of dependent edits, as in multi-step reactions, while restarting permits larger changes of plan. The two interfaces are therefore complementary. Appendix E.2 reports the interaction-length distributions.

## 3.5 Whole-Trace Inference with Qwen3-4B

The smaller-model evaluation shows substantial gains on several structural tasks (Table 3). It uses the same task IDs and whole-trace protocols as the main evaluation.

Relative to Direct CoT, refinement raises atom-count success from 15.8% to 56.8%, component addition from 13.4% to 91.4%, substitution from 15.2% to 53.4%, and the logP optimization score from 14.7 to 60.0. Verification-guided refinement therefore transfers to a smaller model.

Reference-based generation remains difficult for the smaller model: after refinement, forward-reaction accuracy is 1.6% (One-pass: 2.2%), ChEBI-20 reaches 1.4%, and retrosynthesis is 0.0% under every protocol. These tasks depend on chemical knowledge that the proposer must supply, so verification is most effective when paired with a capable candidate generator.

## 4 Case Studies: Repairs and Specification Gaps

Figure 4 pairs molecular structures with verification feedback in six cases, illustrating how rejected commitments are repaired and how an accepted revision can still miss the intended task.

Valence and attachment sites. In the forward-reaction example (a), forming a bond to an acyl carbon before breaking its bond to chlorine would give carbon valence five. The one-pass trace is rejected. Refinement replaces the sequential edits with a joint bond-formation and bond-cleavage operation, removes the detached chlorine, and obtains the correct carbonate product. The aldehydeaddition example (c) instead needs a different attachment site: the same AttachGroup macro fails at substituted aromatic carbon 10 and succeeds at carbon 11. The revised trace then passes the functional-group-change and input-relative similarity checks. The carbonate requires suitable action granularity, whereas the aldehyde requires a suitable molecular referent.

Measured properties and explicit counts. In (b), the one-pass edits lower QED (Bickerton et al., 2012) from 0.674 to 0.573 despite an instruction to increase it. The graph edits pass, and the property assertion fails. Refinement reaches QED 0.767, with success-weighted similarity 0.743. RDKit supplies the measurement and Lean checks the requested direction. In (e), a valid 27-carbon chain fails At omCount because the instruction requests 25. Refinement proposes a 25-carbon chain and passes that count. Both cases separate candidate validity from the requested property, with different evidence sources: atom count is native to the graph, and QED is an externally computed descriptor checked through a witness.

![](images/c7cf09f8ed31c184ffb338f016e66c44efdc6d7681ea7a64d0acb379f1685416.jpg)  
Figure 4: Molecular cases of repair and incomplete specification. (a,c) Refinement changes action granularity or the attachment site. (b,e) A property or count assertion rejects an otherwise valid candidate. (d) MCP repairs an edit within the same maintained state. (f) A revised salt description drops a required component, giving an accepted but incorrect answer. Molecular drawings use the saved structures; numbers in (c,d) denote input atom indices. Refinement restarts from the task input, whereas MCP continues within its episode.

Interactive recovery and incomplete specifications. In (d), MCP's phenyl attachment at carbon 5 is refused without changing the state. The next edit attaches it to nitrogen 1, followed by two checks. The episode finishes correctly after four counted calls, including the refusal. In (f), one-pass proposes the correct salt and fails the disconnected-proposal check. Refinement omits chloride and its count obligation, producing an accepted cation that misses the target salt. Effective revision therefore requires both accurate execution and a specification that continues to express what the user asked for.

## 5 Related Work

Chemical language research connects molecular representations with natural-language descriptions and instructions. Text2Mol introduced text-based molecular retrieval and ChEBI-20 (Edwards et al., 2021); MolT5 studied translation between molecular and natural-language representations (Edwards et al., 2022); and Speak-to-Structure evaluates instruction-driven generation, editing, and optimization (Li et al., 2026a). SELFIES addresses molecular-string validity (Krenn et al. 2020), whereas structural reasoning targets molecular understanding (Jang et al., 2025). ChemBench evaluates chemical knowledge and reasoning more broadly (Mirza et al., 2025). These perspectives complement evidence that fluent chemical reasoning can contain unreliable structural claims (Li et al.,

2026b). Among executable chemical languages, XDL expresses synthesis procedures for compilation into laboratory instructions (Mehr et al., 2020).

Formal and executable reasoning make different parts of a derivation checkable. Lean separates proof construction from proof checking (Moura & Ullrich, 2021); Draft, Sketch, and Prove links informal planning to formal proofs (Jiang et al., 2023), and LeanDojo supports programmatic interaction with Lean (Yang et al., 2023). Learned process supervision evaluates intermediate reasoning with a trained verifier (Lightman et al., 2024). PAL and Program of Thoughts use execution to ground generated computations (Gao et al., 2023; Chen et al., 2023).

Feedback and tool use turn evaluation into part of generation. Self-Refine uses model-generated feedback (Madaan et al., 2023), and CRITIC grounds revision in external tools (Gou et al., 2024), which is more reliable than intrinsic self-correction on reasoning tasks (Huang et al., 2024). ReAct interleaves reasoning and actions (Yao et al., 2023). In chemistry, ChemCrow integrates specialized tools (M. Bran et al., 2024), Coscientist connects planning with experimental automation (Boiko et al., 2023), and ChemReasoner steers catalyst search using computational-chemistry feedback (Sprueill et al., 2024).

## 6 Conclusion

MolLean grounds chemical reasoning in a maintained molecular state, combining Lean-checked obligations with explicit external witnesses. MolLean-MCP makes this feedback available during generation. Whole-trace refinement improves Qwen3.6-27B performance across all 12 tasks, MCP delivers the strongest reaction results, and Qwen3-4B gains substantially on structural tasks. Grounding instructions in explicit constraints is the natural next step toward chemical reasoning that is both verifiable and faithful to the task.

## References

G Richard Bickerton, Gaia V Paolini, Jérémy Besnard, Sorel Muresan, and Andrew L Hopkins. Quantifying the chemical beauty of drugs. Nature chemistry, 4(2):90–98, 2012.

Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023.

Carl Edwards, ChengXiang Zhai, and Heng Ji. Text2mol: Cross-modal molecule retrieval with natural language queries. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 595–607, 2021.

Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho, and Heng Ji. Translation between molecules and natural language. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 375–413, 2022.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International conference on machine learning, pp. 10764–10799. PMLR, 2023.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Nan Duan, Weizhu Chen, et al. Critic: Large language models can self-correct with tool-interactive critiquing. In International Conference on Learning Representations, volume 2024, pp. 57734–57811, 2024.

Stephen R Heller, Alan McNaught, Igor Pletnev, Stephen Stein, and Dmitrii Tchekhovskoi. Inchi, the iupac international chemical identifier. Journal of cheminformatics, 7(1):23, 2015.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In International conference on learning representations, volume 2024, pp. 32808–32824, 2024.

Yunhui Jang, Jaehyung Kim, and Sungsoo Ahn. Structural reasoning improves molecular understanding of llm. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 21016–21036, 2025.

Albert Q. Jiang, Sean Welleck, Jin Peng Zhou, Wenda Li, Jiacheng Liu, Mateja Jamnik, Timothée Lacroix, Yuhuai Wu, and Guillaume Lample. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In International Conference on Learning Representations, 2023.

Mario Krenn, Florian Häse, AkshatKumar Nigam, Pascal Friederich, and Alan Aspuru-Guzik. Selfreferencing embedded strings (selfies): A 100% robust molecular string representation. Machine Learning: Science and Technology, 1(4):045024, 2020.

Jiatong Li, Weida Wang, Qinggang Zhang, Junxian Li, Di Zhang, Changmeng Zheng, Shufei Zhang, Xiaoyong Wei, and Qing Li. Mol-r1: Towards explicit long-cot reasoning in molecule discovery. arXiv preprint arXiv:2508.08401, 2025.

Jiatong Li, Junxian Li, Weida Wang, Yunqing Liu, Changmeng Zheng, Yatao Bian, Dongzhan Zhou, Xiao-Yong Wei, and Qing Li. Speak-to-structure: Evaluating llms in open-domain natural language-driven molecule generation. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 9314–9325, 2026a.

Jiatong Li, Yuxuan Ren, Weida Wang, Xiaoyong Wei, and Yatao Bian. Chemical chain-of-thought functions as a hallucination-prone molecular scratchpad. arXiv preprint arXiv:2607.20935, 2026b.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024.

Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature machine intelligence, 6(5):525–535, 2024.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.

S Hessam M Mehr, Matthew Craven, Artem I Leonov, Graham Keenan, and Leroy Cronin. A universal system for digitization and automatic execution of the chemical synthesis literature. Science, 370(6512):101–108, 2020.

Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, Martiño Ríos-García, Benedict Emoekabu, Aswanth Krishnan, Tanya Gupta, Mara Schilling-Wilhelmi, Macjonathan Okereke, Anagha Aneesh, et al. A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists. Nature Chemistry, 17(7):1027–1034, 2025.

Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming language. In International Conference on Automated Deduction, pp. 625–635. Springer, 2021.

Siddharth Narayanan, James Braza, Ryan-Rhys Griffiths, Albert Bou, Geemi Wellawatte, Mayk Caldas Ramos, Ludovico Mitchener, Michael Pieler, Sam Rodriques, and Andrew White. Training a scientific reasoning model for chemistry. Advances in Neural Information Processing Systems, 38:157671–157710,2026.

RDKit contributors. RDKit: Open-source cheminformatics software, 2026. URL https : / /www . rdkit . org/. Local rescoring uses version 2026.03.1.

David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of chemical information and modeling, 50(5):742–754, 2010.

Nadine Schneider, Nikolaus Stiefl, and Gregory A Landrum. What's what: The (nearly) definitive guide to reaction role assignment. Journal of chemical information and modeling, 56(12):2336– 2346, 2016.

Henry W. Sprueill, Carl Edwards, Khushbu Agarwal, Mariefel V. Olarte, Udishnu Sanyal, Conrad Johnston, Hongbin Liu, Heng Ji, and Sutanay Choudhury. ChemReasoner: Heuristic search over a large language model's knowledge space using quantum-chemical feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 46351–46374. PMLR, 2024.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. Advances in Neural Information Processing Systems, 36:74952–74965, 2023.

Weida Wang, Benteng Chen, Di Zhang, Wanhao Liu, Ben Gao, Shuchen Pu, Shuzhou Sun, Jin Zeng, Tianshu Yu, Wanli Ouyang, et al. Chem-r: Learning to reason as a chemist. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 12291–12302, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J Prenger, and Animashree Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models. Advances in Neural Information Processing Systems, 36:21573–21612, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Guojiang Zhao, Zixiang Lu, Yutang Ge, Sihang Li, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, et al. Molreasoner: Toward effective and interpretable reasoning for molecular llms. arXiv preprint arXiv:2508.02066, 2025.

## A Implementation Details

MolLean combines Lean checks with Python orchestration and external evidence providers. Table 4 distinguishes their responsibilities.

Table 4: Verification components and execution semantics.
<table><tr><td>Component</td><td>Implemented check</td><td>Execution semantics</td></tr><tr><td>Molecule validator</td><td>Proved valence and connectivity soundness; conditional aromaticity</td><td>Validation of represented molecular graphs</td></tr><tr><td>Native assertions</td><td>Graph predicates; six assertion-boundary theorems</td><td>Checks on explicitly stated obligations</td></tr><tr><td>Edits and bundles</td><td>Preconditions, valences, bundle outcomes</td><td>Atomic validation of a concerted bundle&#x27;s outcome</td></tr><tr><td>Bookkeeping</td><td>Atoms, charge, radical parity across both pools</td><td>Additions and removals update the ledger</td></tr><tr><td>External witnesses</td><td>Typed request and predicate on returned fields</td><td>Provider and witness binding trusted</td></tr><tr><td>Python orchestration</td><td>Parsing, reference translation, serialization, control flow</td><td>State transport and inference control</td></tr></table>

The validator's validate\_sound theorem concerns the returned proof-carrying molecular object. Native assertion-boundary proofs cover AtomCount, BondCount, RingCount, HasAromatic, FGCount, and FormulaMatch. Hydrogen counts include graph and implicit hydrogens. Formula matching covers every present element and rejects duplicate element entries, while bookkeeping tracks literal graph atoms.

## A.1 Represented State and Reference Resolution

The runtime serializes a Mult iMol St ate between Python and a persistent Lean process. Its core fields are reproduced below; each TaggedMol pairs a molecule identifier with a graph.

Listing 1: Core Lean state declaration (comments omitted).

structure MultiMolState where   
active : List TaggedMol   
leaving\_pool : List TaggedMol   
step\_count : Nat   
init\_atom\_count : Nat   
init\_charge : Int   
init\_radical\_parity : Nat   
structure AtomRef where   
mol\_id: MolId   
atom\_idx : AtomIdx

A molecular graph stores indexed atoms, bonds, and stereochemical annotations. Atom fields include element, formal charge, radicals, aromaticity, and virtual hydrogen information; bond fields specify endpoints and order. Virtual hydrogens contribute to element and formula readouts, and the bookkeeping ledger counts graph atoms. The leaving-pool is excluded from the predicted answer and remains part of that ledger. Additions, removals, and proposals update the atom-count ledger.

References are pairs (mol\_id, at om\_i dx) into the graph state. A cross-molecule bond joins two index spaces, and a fragment split creates a new molecule identifier. In whole-trace execution, Python tracks accepted primitive merges and remaps subsequent step references through that history. Lean separately rebases references inside tracked compound bundles. MCP uses the current atom map returned in feedback for later model calls. Python also exports the maintained graphs to molecular strings.

## A.2 Action Dispatch and Commit Semantics

Complete traces are JSON objects with a trace array. A step identifies an operation family, for example op with a primitive, macro with a construction macro, compound with a joint transformation, or as sert with a claim. SMILES proposals are parsed into graphs before validation. MCP functions translate to these same step representations and run on the same chemical kernel.

Listing 2 summarizes the dispatch path. Lean returns either an accepted state or a structured error. The runtime replaces the maintained state only after acceptance. A failed operation can therefore report an attempted edit without adding it to the accepted trace. Native assertions and external comparisons inspect the molecular graph without changing it.

Listing 2: Stateful dispatch and the external-evidence boundary (pseudocode).

```python
action = translate(model_call, current_state)
action = discard_model_supplied_witness(action)
reply = lean.apply_action(current_state, action)
if reply.need_witness:
witness = provider.answer(reply.request, initial_input)
reply = lean.apply_action(current_state,
action.with_witness(witness))
if reply.accept:
current_state = reply.state
record_accepted_action(action)
return current_state_and_indices(current_state)
else:
return rejection_and_unchanged_state(reply.error, current_state)
```

Primitive edits validate their preconditions and affected valences. A concerted bundle validates its joint outcome, which accommodates edits whose sequential ordering would pass through a temporarily over-valent graph. The closing procedure checks charge and radical bookkeeping and the task's terminal obligations. Realizability and single-output requirements are task dependent (Appendix D).

## A.3 A Concrete Typed Witness

Native functional-group checks use Lean's built-in patterns. A pattern without a native definition requests an RDKit fg\_count witness, whose returned count Lean compares with the assertion. Native StereoCenterCount counts retained R/S annotations. Bonding changes at a tracked center conservatively remove its annotation.

For native AtomCount, Lean counts the represented atoms of the requested element, including the applicable hydrogen convention; no provider-supplied count is needed. In contrast, PropertyChange (qed) requests an RDKit measurement. The provider evaluates the current molecule and the bound initial input, then supplies integer-valued fields. Lean requires computable=1 and compares out\_x1000 with in\_x1000 in the requested direction.

Listing 3 is the actual accepted payload from the QED case in Appendix C.2. Its values encode an increase from approximately 0.674 to 0.767. Lean checks the predicate on these returned typed fields.

Listing 3: Accepted QED assertion and its supplied witness.

```json
1
"step_type":"assert",
"assertion":{"type":"PropertyChange","mol_id":0,"prop":"qed","
→ direction":"increase","witness_match":true},
"witness":{"ints":[{"k":"computable","v":1},{"k":"in_x1000","v
↔ ":674},{"k":"out_x1000","v":767}],"strs":[]}
}
```

## B Prompts and Tool Interface

This section documents the prompts used by the evaluation interfaces. Prompt text is reproduced from the implemented templates or a saved MCP conversation. Line wrapping and typographic glyphs are normalized for typesetting; task variables retain their actual values.

## B.1 Direct CoT and Whole-Trace Generation

Direct CoT combines Listing 4 with the dataset task instruction. No verifier state, feedback, or held-out answer is included. One-pass uses Listing 5 and a task-specific user message containing the instruction, any loaded molecular input and atom map, operator vocabulary, and JSON output contract. Refinement shares that initial completion and prompt. The whole-trace prompt's phrase “each step is checked" refers to sequential execution of a completed trace: the model generates that trace before receiving any rejection feedback.

Listing 4: Direct CoT' system prompt.   
You are an expert chemist. Reason step by step about the task, then   
output your final answer on the LAST line as 'ANSWER: <SMILES>'   
-- just the SMILES, nothing else.

Listing 5: Whole-trace system prompt shared by One-pass and Refinement.

You are an expert chemist whose every reasoning   
step is verified by a Lean 4 theorem prover (MolLean). Emit a chain   
of typed reasoning steps that build up to (or describe) the target   
molecule. Each step is checked by Lean before the next is allowed.   
A single rejected step rejects the whole chain.   
Output strict JSON only -- no prose outside the JSON block.

Listing 6 instantiates the atom-count generation template for a real example. The marked insertion point expands to the shared vocabulary in Listing 11, which the model receives in full. The complete instantiated user messages for all 12 tasks accompany the LaTeX source as listings/prompts/full\_task.txt.

Listing 6: Atom-count generation user message, with shared vocabulary factored out.

Task: mol\_custom\_atomnum (generate a molecule satisfying a spec)   
Spec:   
There is a molecule with 25 carbon atoms.   
Build a molecule satisfying the spec. Pick the easiest route:   
1. FASTEST:'propose'a complete candidate SMILES that meets the spec   
(MolLean validates it), or 'propose'a core scaffold and then edit.   
You may also propose several fragments and join them with a   
cross-molecule BondForm.   
2. Or build it up: Chain / Ring macros for the skeleton (a chain of N   
carbons is ONE Chain macro), AttachGroup / AttachRing for   
substituents; individual Op primitives for fine adjustments.   
3. Assert steps restating every numerical constraint from the spec   
(AtomCount per element, BondCount per type, FGCount per group,   
RingCount, HasAromatic).   
NOTE: the FINAL molecule is checked for physical realizability --   
valence-saturatable atoms, single connected component, Huckel-valid   
aromatic rings, no leftover unpaired electrons. Finish on a real   
molecule.   
[Insert the shared operator vocabulary here.]   
Return JSON:

```awk
{ "strategy":"...", "trace":[<step list>] }
```

Table 5: Task-specific prompt content. All whole-trace templates append the same operator vocabulary. Reference structures for editing and optimization are the input molecule, not a held-out answer.
<table><tr><td>Family</td><td>Supplied input</td><td>Requested construction and checks</td></tr><tr><td>Atom, bond, and group generation</td><td>Natural-language specification; empty state</td><td>Construct or propose; assert requested counts</td></tr><tr><td>Add, delete, and substitute</td><td>Input SMILES and indexed atom/bond map</td><td>Edit in place; group-count change and input similarity</td></tr><tr><td>logP, MR, and QED optimization</td><td>Input molecule and requested direction</td><td>Property-changing edits; direction and input similarity</td></tr><tr><td>Forward reaction</td><td>Reactants, reaction class, and atom</td><td>Transform loaded reactants; check</td></tr><tr><td>Retrosynthesis</td><td>maps Product, reaction class, and atom map</td><td>proposed outcome Disconnect and modify fragments;</td></tr><tr><td>Text to molecule</td><td>Molecular description; empty state</td><td>check proposed outcome Propose or construct; assert explicit</td></tr></table>

## B.2 Whole-Trace Revision Feedback

After a rejection, the feedback identifies the first failed step, its error, and the state at that point. When available, it includes the live atom map. Listing 7 shows the full feedback produced for the acylation case. This is a request for a corrected complete trace, which is executed from the original input. No correctness score or held-out target enters this message.

Listing 7: Whole-trace revision feedback for a rejected acyl-carbon edit.

MolLean (the Lean verifier) REJECTED your reasoning trace.   
Reason: step 0 rejected: atom (mol=0, idx=3) would have effective   
valence x2 (bond orders + radicals) 10; charge-adjusted allowed x2: [8]   
The FIRST rejected step was index 0 (op): atom (mol=0, idx=3) would have   
effective valence x2 (bond orders + radicals) 10; charge-adjusted   
allowed x2: [8]   
State just before it failed: CCOC(=O)Cl, Oc1ccc(F)cc1F   
LIVE ATOM MAP at that point -- use EXACTLY these (mol\_id, atom\_idx)   
coordinates in your fix:   
mol 0 ('CCOC(=O)Cl'): atoms [0]C [1]C [2]O [3]C [4]O [5]Cl | bonds 0-1   
1-2 2-3 3=4 3-5   
mol 1 ('Oc1ccc(F)cc1F'): atoms [0]O [1]C [2]C [3]C [4]C [5]F [6]C [7]C   
[8]F | bonds 0-1 1:2 2:3 3:4 4-5 4:6 6:7 7-8 7:1   
Diagnose this specific failure and return the FULL corrected trace as   
JSON (same schema: {"strategy": "...", "trace": [ ... ]}). Keep the   
steps that were already accepted; fix the failing step and everything   
downstream of it.   
Do not repeat the same mistake.

An unparseable shared initial completion ends both One-pass and Refinement unsuccessfully. An unparseable later revision stops refinement at the last parseable outcome; only executed revisions enter the round counter.

## B.3 Stateful MCP Prompt and Schemas

MCP supplies function schemas to the model and maintains the molecular state across calls. Listing 8 reproduces its system brief. Listing 9 is the complete initial user message from the phenyl-attachment episode in Appendix C.4. It contains the task instruction and current atom map. Later turns append the actual verifier response; a rejected edit leaves this maintained state unchanged.

$$
\mathsf { L i s t i n g \ 8 : M C P \ s y s t e m \ b r i e f . }
$$

You are deriving chemistry with a verifier attached, and the verifier is   
fast. Do not analyse the whole route before acting. Decide the most   
likely next edit, call the function, and read what comes back:   
the kernel answers in milliseconds and its reply is better evidence than   
any amount of reasoning about what the reply would be.   
Every reply must end in exactly one function call. One or two sentences   
of reasoning before it is enough; a paragraph is a paragraph the kernel   
cannot check. If you are unsure which of two edits is right, try one.   
A refused call costs a call and leaves the state untouched, and the   
refusal names the reason.   
Atom indices always come from the kernel's last reply. Never count them   
off a SMILES string. Call finish() when the state holds the answer.

## Listing 9: Saved initial MCP user message for phenyl attachment.

Please add a benzene ring to the molecule NNclncc(Br)cc1C(F) (F)F.   
Current state:   
mol 0 ('NNc1ncc(Br)cc1C(F)(F)F'): atoms [0]N [1]N [2]C [3]N [4]C [5]C   
[6]Br [7]C [8]C [9]C [10]F [11]F [12]F | bonds 0-1 1-2 2:3 3:4 4:5 5-6   
5:7 7:8 8-9 9-10 9-11 9-12 8:2

The schemas distinguish action arguments from measured evidence. For example, attach\_group names a molecule, anchor atom, and group; check names a claim and its JSON arguments; fini sh requests the closing procedure. Descriptor values always come from a provider. Listing 10 shows the complete attachment schema. The full schema inventory is included in the source file listings/prompts/all\_tools.json, together with separate check and finish schemas.

## Listing 10: Native function schema for attaching a group.

```json
"type":"function",
"function":{
"name":"attach_group",
"description":"Graft a named functional group onto an atom.",
"parameters":{
"type":"object",
"properties":{
"mol_id":{"type":"integer","description":"molecule id, from the
↔ state listing"},
"atom_idx":{"type":"integer","description":"atom index within
↔ that molecule"},
"group": {
"type":"string",
"description":"hydroxyl, oxo, amino, methyl, carboxyl, ester,
amide, nitro, nitrile, methoxy, halide names, phenyl, pyridine,
... or a fragment SMILES"
}
},
"required":["mol_id","atom_idx","group"]
}
}
}
```

## B.4 Shared Whole-Trace Vocabulary

Listing 11 reproduces the shared operator instructions inserted into the task-specific whole-trace messages. These instructions describe the model-facing syntax; task setup and runtime checks determine which proposals and closing conditions apply (Appendix D). In particular, the runtime permits candidate proposals in optimization and reactions, and requires a single final molecule only

for editing. The vocabulary's atom-count shorthand for new indices assumes contiguous indexing;   
the implementation allocates after the largest current atom index.

Listing 11: Shared whole-trace operator vocabulary.

Available step shapes:   
-- Propose steps -- TWO distinct interfaces   
MolLean validates every proposed structure (no impossible valences,   
single connected component) before it enters the state.   
{"step\_type":"propose", "smiles":"<SMILES>"}   
-- Propose a WHOLE molecule. Allowed only for SYNTHESIS tasks   
(text\_to\_mol / mol\_custom), where you build a target from   
nothing. If the state is empty it becomes mol\_id 0.   
{"step\_type":"propose\_fragment","smiles":"<SMILES>"}   
-- Propose a FRAGMENT to graft onto an existing molecule. It is   
added as a new mol\_id; you THEN attach it with a   
cross-molecule BondForm (src in mol 0, dst in the fragment),   
which merges it into mol 0. Use this for TRANSFORMATION tasks   
(mol\_edit / mol\_opt) -- e.g. enumerate property-improving   
substructures and combine them onto the input. The result must   
end as ONE molecule.   
Idioms: base-and-edit (propose a scaffold, then AttachGroup/BondForm/   
AddAtom/RemoveAtom); fragment assembly (propose\_fragment several   
pieces, join with cross-molecule BondForm).   
-- Op steps (state-changing primitives)   
{"step\_type":"op","primitive":{"type":"AddAtom", "mol\_id":0,   
"element":"C"|"N"|"o"|..., "charge":0, "radicals":0,   
"is\_aromatic":false}}   
-- Append a new atom to molecule 'mol\_id'. Atom indices grow   
sequentially as you add them (first AddAtom -> idx 0).   
{"step\_type":"op","primitive":{"type":"BondForm",   
"src":{"mol\_id":0,"atom\_idx":i},   
"dst":{"mol\_id":0,"atom\_idx":j},   
"bond\_type":"Single"|"Double"|"Triple"|"Aromatic"}}   
-- Form a bond between two existing atoms.   
{"step\_type":"op","primitive":{"type":"BondBreak",   
"src":{"mol\_id":0,"atom\_idx":i},   
"dst":{"mol\_id":0,"atom\_idx":j}}}   
{"step\_type":"op", "primitive":{"type":"BondOrderChange",   
"src":{...}, "dst":{...}, "bond\_type":"..."}}   
{"step\_type":"op", "primitive":{"type":"ChargeUpdate",   
"atom":{"mol\_id":0,"atom\_idx":i},"delta":+1|-1}}   
{"step\_type":"op","primitive":{"type":"RadicalUpdate",   
"atom":{...}, "delta":+1|−1}}   
{"step\_type":"op", "primitive":{"type":"RemoveAtom",   
"atom":{"mol\_id":0,"atom\_idx":i}}}   
-- Compound steps (SUBGRAPH-LEVEL mechanism ops)   
Act on a GROUP of atoms/bonds at once -- useful for reaction   
mechanisms.   
Each desugars to Layer-1 primitives that Lean verifies; conservation   
is checked over the whole step.   
{"step\_type":"compound", "compound":{"compound":"ProtonShift",   
"h": {...}, "donor":{...}, "acceptor": {...}}}   
-- move a proton H from donor to acceptor (acid/base, tautomer).   
The state is HEAVY-ATOM only: implicit Hs are not addressable.   
First AddAtom an explicit H + BondForm it to the donor, then   
ProtonShift that H. Works across molecules (the framework   
re-bases refs through the merge).   
{"step\_type":"compound", "compound":{"compound":"Eject",

"anchor":{...}, "leaving":{...}}}   
-- break anchor-leaving and send the leaving group to the pool.   
{"step\_type":"compound", "compound":{"compound":"Fragment",   
"anchor":{...}, "leaving":{...}}}   
-- break a BRIDGE bond and split the molecule into TWO active   
molecules (retro: product -> two reactant fragments). Both   
halves stay active.   
{"step\_type":"compound", "compound":{"compound":"BondFormBreak",   
"form\_src":{...}, "form\_dst":{...}, "form\_bt":"Single",   
"break\_src":{...}, "break\_dst":{...}}}   
-- concerted form-one-bond + break-another (SN2, addition-elim).   
{"step\_type":"compound", "compound":{"compound":"Concerted",   
"primitives":[ <op primitive>, ... ]}}   
-- apply a bundle of primitives atomically (Lean checks the net   
result).   
- Macro steps (HIGH-LEVEL build ops -- STRONGLY PREFERRED)   
↔   
Prefer these over long AddAtom/BondForm chains. Each macro expands to   
Layer-1 primitives that Lean still verifies individually, but your   
reasoning stays compact and you avoid atom-index bookkeeping errors.   
After a macro, atom indices continue from where the macro left off   
(a Chain/Ring of n atoms occupies the next n free indices, in order).   
{"step\_type":"macro","macro":{"macro":"Chain",   
"element":"C","n":N,"bond\_type":"Single","mol\_id":0}}   
-- A linear chain of N atoms. Occupies indices [base .. base+N   
-1] where base = current atom count.   
{"step\_type":"macro","macro":{"macro":"Ring",   
"element":"C","n":N,"aromatic":true,"mol\_id":0}}   
-- An N-membered ring (aromatic if "aromatic":true -> benzene   
for N=6). Occupies the next N indices and closes the loop.   
{"step\_type":"macro","macro":{"macro":"AttachGroup",   
"anchor":{"mol\_id":0,"atom\_idx":i},"group":"<name>"}}   
-- Graft a functional group onto existing atom i. Recognised   
groups: hydroxyl, oxo, amino, methyl, carboxyl, carboxylate,   
aldehyde, ester, amide, nitro, nitrile, methoxy, ether,   
sulfonyl, phosphate, fluoro/chloro/bromo/iodo, thiol,   
benzene/phenyl, pyridine, furan, pyrrole, thiophene,   
cyclohexane, cyclopropane, cyclopentane. (Or pass a raw   
fragment SMILES as "group".)   
{"step\_type":"macro","macro":{"macro":"AttachRing",   
"anchor":{"mol\_id":0,"atom\_idx":i},"n":6,"aromatic":true,   
"mol\_id":0}}   
-- Build an N-membered ring and bond it to existing atom i.   
-- Assert steps (state-querying claims; witness filled by framework)   
{"step\_type":"assert","assertion":{"claim":"AtomCount",   
"element":"carbon"|"nitrogen"|...,"count":N,"mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"BondCount",   
"bond\_type":"Single"|"Double"|...,"count":N,"mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"RingCount","count":N,"mol\_id   
→″:0}}   
{"step\_type":"assert","assertion":{"claim":"HasAromatic","expected":true   
→ ,"mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"FGCount",   
"fg":"<recognised-name>","count":N,"mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"FGCountDelta",   
"fg":"<name>","delta":+1|-1,"input\_smiles":"...","mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"NameMatch","name":"...","   
→ mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"ClassMember","class":"...","   
→ mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"PropertyChange",

"prop":"logp"|"qed"|"solubility"|...,"direction":"increase"|"decrease",   
"input\_smiles":"...","mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"OutcomeMatch",   
"reference\_smiles":"<target>","mode":"skeleton"|"exact"|"   
↔ tanimoto", "mol\_id":0}}   
{"step\_type":"assert","assertion":{"claim":"StereoCenterCount","count":N   
→,"mol\_id":0}}   
-- exactly N tracked R/S stereocentres. Conservative: an edit AT   
a centre drops its annotation, so only claim centres your edits   
never touched.   
Recognised FG names (for FGCount / FGCountDelta):   
hydroxyl, aldehyde, ketone, carboxyl, carboxylate, ester, amide,   
amine, primary\_amine, secondary\_amine, tertiary\_amine, nitro,   
nitrile, thiol, thioether, sulfoxide, sulfone, halide, fluoride,   
chloride, bromide, iodide, alkene, alkyne, ether, epoxide, imine,   
phenol, anhydride, benzene\_ring, pyridine, pyrrole, furan,   
thiophene, indole, imidazole, naphthalene, quinoline, pyrimidine,   
phosphate\_ester, sulfate\_ester.

## C Worked Examples

These eight examples show the actual submitted operations, verifier feedback, and saved outcomes. The first six expand the cases in Figure 4; the last two illustrate a multi-molecule retrosynthetic answer and a native functional-group assertion. JSON is reformatted without changing fields or values. Whole-trace listings retain every submitted step, including unexecuted steps after the first rejection. MCP listings retain every function call and verifier reply; the model's free-form deliberation is omitted for space.

## C.1 Acylation: Changing the Verification Unit

Listing 12: Forward-reaction task input.   
Predict the major product(s) of the following acylation reaction.   
Reactants: CCOC(=O)Cl.Oc1ccc(F)cc1F   
Output only the product SMILES. Use a dot to separate multiple products.

The first trace forms the carbonyl-carbon-phenolic-oxygen bond before removing the carbon–chlorine bond. The first edit is rejected: the acyl carbon would have bond-order valence five. In the diagnostic representation, valence is doubled, so the attempted value 10 exceeds the allowed 8. The subsequent ejection and proposed-outcome assertion remain unexecuted.

Listing 13: Complete initial acylation trace; execution stops at step 0.

```snap
"trace":[
{
"step_type":"op",
"primitive":{"type":"BondForm","src":{"mol_id":0,"atom_idx":3},"
dst":{"mol_id":1,"atom_idx":0},"bond_type":"Single"}
},
{
"step_type":"compound",
"compound":{"compound":"Eject","anchor":{"mol_id":0,"atom_idx
":3},"leaving":{"mol_id":0,"atom_idx":5}}
},
{
"step_type":"assert",
"assertion":{"claim":"OutcomeMatch","reference_smiles":"CCOC(=O)
↔ Oc1ccc(F)cc1F","mode":"exact", "mol_id":0}
}
1
}
```

Listing 14: First-rejection feedback.   
step 0 rejected: atom (mol=0, idx=3) would have effective valence x2   
(bond orders + radicals) 10; charge-adjusted allowed x2: [8]

The corrected trace uses BondFormBreak to validate bond formation and cleavage as one transformation, then removes the detached chlorine atom. Its three steps are accepted and its final carbonate product is correct. The Out comeMat ch structure is the model's own proposed endpoint. The change repairs a graph-execution failure.

Listing 15: Complete accepted acylation revision, executed from the reactants.

"trace":[   
{   
"step\_type":"compound",   
"compound" : {   
"compound":"BondFormBreak",   
"form\_src":{"mol\_id":0,"atom\_idx":3},   
"form\_dst":{"mol\_id":1,"atom\_idx":0},   
"form\_bt":"Single",   
"break\_src":{"mol\_id":0,"atom\_idx":3},   
"break\_dst":{"mol\_id":0,"atom\_idx":5}   
}   
},   
{"step\_type":"op","primitive":{"type":"RemoveAtom","atom":{"mol\_id   
↔ ":0,"atom\_idx":5}}},   
{   
"step\_type":"assert",   
"assertion":{"claim":"OutcomeMatch","reference\_smiles":"CCOC(=O)   
↔ Oc1ccc(F)cc1F","mode":"exact", "mol\_id":0}   
}   
]   
}

## C.2 QED Optimization: A Valid Edit Can Move the Wrong Way

Listing 16: QED task input.   
Please optimize the molecule C/C(=C/CCCl)clcnccc1C to have a higher QED   
value.

The initial trace deletes three atoms. These edits are individually accepted, and the subsequent PropertyChange assertion is rejected because QED decreases from 0.674 to 0.573. The final input-similarity assertion remains unexecuted. The feedback targets the stated objective.

## Listing 17: Complete initial QED trace; the property assertion fails.

{   
"trace":[   
{"step\_type":"op","primitive":{"type":"RemoveAtom","atom":{"mol\_id   
→ ":0, "atom\_idx":5}}},   
{"step\_type":"op","primitive":{"type":"RemoveAtom","atom":{"mol\_id   
→ ":0, "atom\_idx":4}}},   
{"step\_type":"op","primitive":{"type":"RemoveAtom","atom":{"mol\_id   
→ ":0,"atom\_idx":3}}},   
{   
"step\_type":"assert",   
"assertion":{   
"claim":"PropertyChange",   
"prop":"qed",   
"direction":"increase",   
"input\_smiles":"C/C(=C/CCCl) c1cnccc1C",   
"mol\_id":0   
}   
},   
{   
"step\_type":"assert",   
"assertion":{"claim":"OutcomeMatch", "reference\_smiles":"C/C(=C/   
CCCl)c1cnccc1C","mode":"tanimoto","mol\_id":0}   
}   
]   
}

Listing 18: Measured-property rejection.  
step 3 rejected: assertion PropertyChange(qed) does not hold: property   
'qed' did not increase (in 674/1000, out 573/1000)

The revision removes chlorine and attaches hydroxyl at atom 4. All four steps are accepted; the saved final structure is C/C (=C/CCO) c1 cnccc1C, with QED approximately 0.767. Appendix A shows the associated typed witness.

Listing 19: Complete accepted QED revision.

{   
"trace":[   
{"step\_type":"op","primitive":{"type":"RemoveAtom","atom":{"mol\_id   
↔ ":0,"atom\_idx":5}}},   
{   
"step\_type":"macro",   
"macro":{"macro":"AttachGroup","anchor":{"mol\_id":0,"atom\_idx   
":4},"group":"hydroxyl"}   
},   
{   
"step\_type":"assert",   
"assertion":{   
"claim":"PropertyChange",   
"prop":"qed",   
"direction":"increase",   
"input\_smiles":"C/C(=C/CCCl)c1cnccc1C",   
"mol\_id":0   
}   
},   
{   
"step\_type":"assert",   
"assertion":{"claim":"OutcomeMatch","reference\_smiles":"C/C(=C/   
CCCl)c1cnccc1C","mode":"tanimoto","mol\_id":0}   
}   
]   
}

## C.3 Aldehyde Attachment: Repairing the Referent

Listing 20: Aldehyde-addition task input.   
Modify the molecule C[C@@H]1CC[C@](O) (Cc2nc(-c3ccccc3)cs2)C1 by adding a   
aldehyde.

Attaching aldehyde at atom 10 is rejected because that substituted aromatic carbon cannot accept the additional bond. The revision retains the same At t achGroup macro and moves its anchor to atom 11. Its group-count-change and input-similarity assertions then pass. This repair changes only the molecular referent.

Listing 21: Initial aldehyde trace with the invalid attachment site.

{   
"trace":[   
{   
"step\_type":"macro",   
"macro":{"macro":"AttachGroup","anchor":{"mol\_id":0,"atom\_idx   
↔ ":10}, "group":"aldehyde"}   
},   
{   
"step\_type":"assert",   
"assertion":{   
"claim":"FGCountDelta",   
"fg":"aldehyde",   
"delta":1,   
"input\_smiles":"C[C@@H]1CC[C@](O) (Cc2nc(-c3ccccc3)cs2)C1",   
"mol\_id":0   
}   
},   
{   
"step\_type":"assert",   
"assertion":{   
"claim":"OutcomeMatch",   
"reference\_smiles":"C[C@@H]1CC[C@](O) (Cc2nc(-c3ccccc3)cs2)C1",   
"mode":"tanimoto",   
"mol\_id":0   
}   
}   
]   
}

Listing 22: Attachment-site rejection.  
step 0 rejected: atom (mol=0, idx=10) would have effective valence x2   
(bond orders + radicals) 10; charge-adjusted allowed x2: [8]

Listing 23: Complete aldehyde revision with anchor 11.

{   
"trace":[   
{   
"step\_type":"macro",   
"macro":{"macro":"AttachGroup","anchor":{"mol\_id":0,"atom\_idx   
":11}, "group":"aldehyde"}   
},   
{   
"step\_type":"assert",   
"assertion":{   
"claim":"FGCountDelta",   
"fg":"aldehyde",   
"delta":1,   
"input\_smiles":"C[C@@H]1CC[C@](O) (Cc2nc(-c3ccccc3)cs2)C1",   
"mol\_id":0   
}   
},   
{   
"step\_type":"assert",   
"assertion":{   
"claim":"OutcomeMatch",   
"reference\_smiles":"C[C@@H]1CC[C@](O) (Cc2nc(-c3ccccc3)cs2)C1",   
"mode":"tanimoto",   
"mol\_id":0   
}   
}   
]   
}

## C.4 MCP Phenyl Attachment: Continuing After a Refusal

The task and initial indexed state are reproduced in Listing 9. This episode continues from the maintained state after the rejection. The first attach\_group call targets carbon 5 and fails. The reply explicitly reports state\_unchanged=true and the same atom map. The model next targets nitrogen 1; the accepted attachment introduces indices 13–18. Two checks follow: a benzene-ring count increase and similarity to the input.

Listing 24 shows all five function calls, including fini sh, and all four intervening verifier replies. The episode records four counted calls, because successful fini sh is outside the edit/check counter. One of those four calls is refused, and the final answer is correct. The Out comeMat ch check is a Tanimoto comparison.

Listing 24: Saved MCP episode, with deliberation omitted and calls normalized to JSON.

MODEL CALL   
{"name":"attach\_group","arguments":{"mol\_id":0,"atom\_idx":5,"group":"   
↔ phenyl"}}   
VERIFIER REPLY   
{   
"ok":false,   
"reason":"atom (mol=0, idx=5) would have effective valence x2 (bond   
orders + radicals) 10; charge-adjusted allowed x2: [8]",   
"state\_unchanged":true,   
"indices":"mol 0 ('NNc1ncc(Br)cc1C(F)(F)F'): atoms [0]N [1]N [2]C [3]N   
→ [4]C [5]C [6]Br [7]C [8]C [9]C [10]F [11]F [12]F | bonds 0-1 1-2   
→ 2:3 3:4 4:5 5-6 5:7 7:8 8-9 9-10 9-11 9-12 8:2"   
}   
MODEL CALL   
{"name":"attach\_group","arguments":{"atom\_idx":1,"group":"phenyl","   
↔ mol\_id":0}}

Listing 25: Atom-count task input.   
There is a molecule with 25 carbon atoms.

VERIFIER REPLY   
"ok":true,   
"molecules":["NN(c1ccccc1)c1ncc(Br)cc1C(F) (F)F"],   
"changed":"mol 0: new atom index 13, 14, 15, 16, 17, 18"   
}   
MODEL CALL   
"name":"check",   
"arguments":{   
"arguments":{"fg":"benzene\_ring","delta":1,"input\_smiles":"NNc1ncc(   
↔ Br) cc1C(F) (F) F"},   
"claim":"FGCountDelta",   
"mol\_id":0   
}   
}   
VERIFIER REPLY   
{"ok":true,"molecules":["NN(c1ccccc1)c1ncc(Br)cc1C(F) (F)F"],"changed":"   
indices unchanged"}   
MODEL CALL   
"name":"check",   
"arguments": {   
"arguments":{"reference\_smiles":"NNc1ncc(Br)cc1C(F) (F)F","mode":"   
→ tanimoto"},   
"claim":"OutcomeMatch",   
"mol\_id":0   
}   
}   
VERIFIER REPLY   
{"ok":true,"molecules":["NN(c1ccccc1)c1ncc(Br)cc1C(F) (F)F"],"changed":"   
→ indices unchanged"}   
MODEL CALL   
{"name":"finish","arguments":{}}

## C.5 Atom Count: The Count Obligation Guides the Correction

The first proposal is a graph-valid chain with 27 carbon atoms. Its explicit At omCount assertion requests 25 and is rejected by native counting. The corrected trace proposes a 25-carbon chain and repeats the same obligation. Both steps of the revision are accepted, and the generated molecule satisfies the requested count. No external measurement is needed for this rejection or repair.

Listing 26: Initial 27-carbon proposal and its 25-carbon obligation.

"trace":[   
{"step\_type":"propose","smiles":"CCCCCCCCCCCCCCCCCCCCCCCCCCC"},   
{"step\_type":"assert", "assertion":{"claim":"AtomCount","element":"   
carbon","count":25,"mol\_id":0}}   
1   
}

Listing 27: Native atom-count rejection.

step 1 rejected: assertion AtomCount(C) does not hold:   
expected 25 atoms, Lean counted 27

Listing 28: Complete accepted 25-carbon revision.

```jsonl
{
"trace":[
{"step_type":"propose","smiles":"CCCCCCCCCCCCCCCCCCCCCCCCC"},
{"step_type":"assert","assertion":{"claim":"AtomCount","element":"
carbon","count":25,"mol_id":0}}
]
}
```

C.6 Hydrochloride Description: Acceptance After Dropping a Requirement

Listing 29: Text-to-molecule task input.

Given the following molecule description, output a SMILES string that   
satisfies it.   
Description: The molecule is a hydrochloride resulting from the reaction   
of equimolar amounts of phenylhydrazine and hydrogen chloride. It   
contains a phenylhydrazine.

The first completion proposes phenylhydrazinium chloride and explicitly requires one chlorine atom. Its disconnected proposal is rejected at step 0, before the assertions run. The revision proposes only the cation and removes both chloride and the chlorine-count obligation. Its four steps are accepted, and the final molecule misses the hydrochloride salt described in the task.

Listing 30: Initial salt proposal and all submitted count obligations.

{   
"trace":[   
{"step\_type":"propose","smiles":"C1=CC=C(C=C1)N[NH3+].[Cl-]"},   
{"step\_type":"assert","assertion":{"claim":"AtomCount","element":"   
carbon","count":6,"mol\_id":0}},   
{"step\_type":"assert","assertion":{"claim":"AtomCount","element":"   
nitrogen","count":2,"mol\_id":0}},   
{"step\_type":"assert","assertion":{"claim":"AtomCount","element":"   
chlorine","count":1,"mol\_id":0}},   
{   
"step\_type":"assert",   
"assertion":{"claim":"OutcomeMatch","reference\_smiles":"C1=CC=C(C=   
→ C1)N[NH3+].[Cl-]","mode":"exact","mol\_id":0}   
}   
]   
}

Listing 31: Disconnected-proposal rejection.  
```javascript
step 0 rejected: this step would leave atom (mol=0, idx=0) disconnected
```

Listing 32: Accepted revision that omits the required chloride component.

"trace":[   
{"step\_type":"propose","smiles":"c1ccccc1N[NH3+]"},   
{"step\_type":"assert","assertion":{"claim":"AtomCount","element":"   
→ carbon", "count":6, "mol\_id":0}},   
{"step\_type":"assert","assertion":{"claim":"AtomCount","element":"   
nitrogen","count":2,"mol\_id":0}},   
{   
"step\_type":"assert",   
"assertion":{"claim":"OutcomeMatch", "reference\_smiles":"c1ccccc1N[   
NH3+]", "mode":"exact", "mol\_id":0}   
}   
]   
一

The self-proposed Out comeMat ch checks consistency with the cation the model chose; it cannot recover the omitted requirement. This is a specification gap. The example motivates retaining task obligations during revision and supporting the required representation.

## C.7 Retrosynthesis: The Answer Is the Full Active Set

Listing 33: Retrosynthesis task input.

Propose reactants that produce the following target via a acylation   
reaction.   
Product: COc1ccc(C(N)=O)cc1C(=O)Nc1cccc(Cl)c1   
Output reactant SMILES separated by dots.

The accepted one-pass trace cuts the amide connection between atoms 11 and 13 using Fragment. Both fragments remain active. It then attaches hydroxyl at the acyl-carbon anchor to obtain the carboxylic-acid precursor. The result consists of the acid and an aniline derivative, and the saved reactant set is correct.

Listing 34: Complete accepted retrosynthetic trace.

```json
{
"trace":[
{
"step_type":"compound",
"compound":{"compound":"Fragment","anchor":{"mol_id":0,"atom_idx
↔ ":11},"leaving":{"mol_id":0,"atom_idx":13}}
},
{
"step_type":"macro",
"macro":{"macro":"AttachGroup","anchor":{"mol_id":0,"atom_idx
":11}, "group": "hydroxyl"}
},
{
"step_type":"assert",
"assertion":{"claim":"OutcomeMatch","reference_smiles":"COc1ccc(C(
→ N) =O) cc1C(=O)O", "mode":"exact", "mol_id":0}
}
]
```

The submitted Out comeMat ch assertion refers only to molecule 0, whereas reaction scoring compares the complete active set. Thus an accepted assertion about one fragment and a correct multi-molecule answer are distinct observations. Fragment, unlike Eject, retains the second fragment as part of that answer.

## C.8 Functional-Group Generation: A Native Structural Obligation

Listing 35: Functional-group generation task input.

The molecule is composed of 1 amine group.

The model proposes methylamine and asserts one amine group. Both the proposal and the native FGCount check are accepted, and the saved task outcome is successful. This compact example separates proposing a chemically valid candidate from stating the structural requirement that makes it relevant to the task. No refinement is needed.

Listing 36: Complete accepted methylamine trace.

{   
"trace":[   
{"step\_type":"propose","smiles":"CN"},   
{"step\_type":"assert","assertion":{"claim":"FGCount","fg":"amine","   
↔ count":1, "mol\_id":0}}   
]   
}

## D Evaluation Details

## D.1 Task Setup and Scoring

Generation begins from an empty state; editing and optimization begin from the input molecule. Forward prediction begins from reactants and retrosynthesis from the product. Editing requires one active output molecule and disallows whole-molecule replacement, while optimization permits candidate proposals. Generation, ChEBI-20, and reaction tasks use terminal realizability checks. Closing bookkeeping applies throughout.

Non-reaction tasks score the first active molecule; reactions score the full active set. A rejected whole-trace completion can still provide a scored answer from its available state, or a parsed answer when no state is returned. Failed certification can therefore coexist with a correct answer. Missing or invalid answers receive zero. Selected episodes ending in context exhaustion are no-results, remain in the denominator, and receive zero score and certification; CoT and One-pass answers for the same example are scored independently.

ChEBI-20 uses its full supplied test split. USPTO selection uses MD5(reaction key) mod $1 0 0 \geq 9 0 .$ with a key combining patent identifier and row index. Forward and reverse versions of a reaction remain on the same side. We evaluate 500 examples per direction and supply reaction class names to all protocols. Optimization direction success is reported below separately from the similarityweighted main-table score.

Table 6: Qwen3.6-27B optimization direction success (%). Each task contains 500 examples; input similarity is not included in this table.
<table><tr><td>Objective</td><td>CoT</td><td>One-pass</td><td>Refine</td><td>MCP</td></tr><tr><td>logP optimization</td><td>59.8</td><td>95.8</td><td>99.8</td><td>99.4</td></tr><tr><td>MR optimization</td><td>60.6</td><td>93.2</td><td>99.8</td><td>99.6</td></tr><tr><td>QED optimization</td><td>54.2</td><td>65.6</td><td>92.2</td><td>77.0</td></tr></table>

## D.2 Inference Settings

Whole-trace refinement uses a 32,768-token context, temperature 0.3, up to 12,000 requested output tokens, and at most eight revisions after the initial one-pass completion. It stops at acceptance, its budget limit, or an early failure, with parse handling as described in Appendix B. The task reference is excluded from verifier feedback and used only for answer scoring.

MCP requests a 65,536-token context, temperature 0.7, and 4,096 output tokens per model call, with at most two attempts, 30 counted action/check calls, and 10 refusals per attempt. Read-only measure and show\_state calls and the closing finish call are outside that call counter. The first certified attempt is selected; otherwise the final attempt supplies the outcome

## E Additional Results

## E.1 Whole-Trace Refinement Across Budgets

Figure 3 in Section 3 shows the accepted-trace rate and answer score after at most k whole-trace revisions, with $k = 0$ denoting the shared One-pass completion. An accepted trace is retained at all larger budgets. The curves use the saved intermediate outcome at each budget, and the stopping rule uses only verifier feedback. Unsuccessful examples remain in each task's denominator. A contextexhausted continuation receives zero once its next completion is unavailable, while its independently saved earlier outcomes are retained at smaller budgets.

## E.2 Interaction-Length Distribution in MolLean-MCP

Figure 5 describes the action/check budget consumed in the selected MCP attempt for every evaluated example. Selection follows the evaluation protocol: the first certified attempt, or the final attempt when none certifies. Counted calls include rejected actions and false assertions and exclude measure, show\_state, and finish.

The selected attempts are typically short on editing and descriptor optimization: the median is three calls for addition, deletion, logP, MR, and QED, and five for substitution. Reaction medians are four calls for forward prediction and five for retrosynthesis. ChEBI-20 has a longer interaction distribution, with a median of six and an interquartile range of four to ten calls. The tails also distinguish tasks: QED and atom-count generation contain appreciable mass beyond ten calls even though their medians are small. The accepted and unaccepted portions of each bar show whether the selected attempt closed successfully. Longer interaction can reflect additional construction, checking, or failed actions.

![](images/8ff9fbdf3cafcfb685d0403bed496f0f77e4795c8ec70e033d8ecd1fdc72ba0b.jpg)

![](images/7047e42358840303b295c74b04e12ebf36a80ad514442d61c0f9941b27b88ead.jpg)

![](images/6a9bdacf952c9ddcba8c4aa950296ae07d1361875a1193625f131411b2c601ed.jpg)

![](images/05ab55dfd7044f2fada1fe902ae49f0ac714df046cf79f81d743f4b3bb66075d.jpg)

![](images/6de4e838fafb38f04575e73e42b54fc780b55a6b7882f48ea378d6eaa2919f20.jpg)

![](images/aa564523836febed80e54e6b5be0312b9e8b30da2a4ca99f8956372a79f1a4c6.jpg)

![](images/5e04a390fe22f61a1fa62dd2c4dc50d4b5be85902970682a1e990f57531bdd54.jpg)

![](images/23947999d57903cef9de158ce80e31215e03e5ac9d0a1ca646800d546980ef51.jpg)

![](images/7ffa832d0c41fb8ac0ed1667b2677edf7cf3a4b0d6cf0d7fa82d17ab961d7ee1.jpg)

![](images/62391a84268faba4951ccb8305e59b36c8e02c60c97ba396b62459a840240849.jpg)

![](images/4b85c31b447b23715f1d81ffd0d391b23ea604eb1b764e37a8af812f791b4296.jpg)

![](images/cec22cc0301ef5302a8c19115080554e5e6906750bf16a7fc1e8169bec029105.jpg)  
Figure 5: Distribution of counted MCP action/check calls in the selected attempt. Bars sum to 100% within each complete task cohort, including unaccepted and no-result examples. Colors separate accepted from unaccepted episodes. Each panel reports its median. Bins span zero, 1–5, 6– 10, 11–15, 16–20, 21–25, and 26–30 calls. Counts exclude measure, show\_state, and finish, as well as earlier attempts.

## E.3 Certification and Statistical Uncertainty

Table 2 in Section 3 reports trace acceptance and paired 95% bootstrap intervals for the primary Refine-minus-CoT scores. Each interval uses 4,000 paired resamples within its task. The intervals describe variation across examples conditional on the model outputs.

## E.4 Joint and Paired Outcomes

Table 7 separates acceptance from binary task success for Refinement. Each row sums to the complete task denominator. For optimization, success here means improvement in the requested property direction.

Table 7: Joint outcomes for Qwen3.6-27B Refinement.
<table><tr><td colspan="2"></td><td colspan="2">Sound</td><td colspan="2">Not sound</td></tr><tr><td>Task</td><td>N</td><td>Success</td><td>Failure</td><td>Success</td><td>Failure</td></tr><tr><td>Atom counts</td><td>500</td><td>444</td><td>23</td><td>0</td><td>33</td></tr><tr><td>Bond counts</td><td>500</td><td>345</td><td>79</td><td>0</td><td>76</td></tr><tr><td>Functional groups</td><td>500</td><td>209</td><td>279</td><td>0</td><td>12</td></tr><tr><td>Add component</td><td>500</td><td>500</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Delete component</td><td>500</td><td>485</td><td>14</td><td>0</td><td>1</td></tr><tr><td>Substitute component</td><td>500</td><td>480</td><td>19</td><td>0</td><td>1</td></tr><tr><td>logP optimization</td><td>500</td><td>499</td><td>1</td><td>0</td><td>0</td></tr><tr><td>MR optimization</td><td>500</td><td>499</td><td>0</td><td>0</td><td>1</td></tr><tr><td>QED optimization</td><td>500</td><td>458</td><td>12</td><td>3</td><td>27</td></tr><tr><td>Forward reaction</td><td>500</td><td>298</td><td>143</td><td>16</td><td>43</td></tr><tr><td>Retrosynthesis</td><td>500</td><td>96</td><td>345</td><td>8</td><td>51</td></tr><tr><td>ChEBÍ-20 text to molecule</td><td>3,300</td><td>1075</td><td>1533</td><td>17</td><td>675</td></tr></table>

Table 8 pairs the same initial one-pass completion with its final refinement outcome. Recovered answers change from failure to success; lost answers change from success to failure. The other two columns retain their original status.

Table 8: One-pass to Refinement transitions for Qwen3.6-27B. Each row sums to $N ;$ categories use binary task success.
<table><tr><td>Task</td><td>Both succeed</td><td>Recovered</td><td>Lost</td><td>Both fail</td></tr><tr><td>Atom counts</td><td>307</td><td>137</td><td>0</td><td>56</td></tr><tr><td>Bond counts</td><td>199</td><td>146</td><td>0</td><td>155</td></tr><tr><td>Functional groups</td><td>194</td><td>15</td><td>0</td><td>291</td></tr><tr><td>Add component</td><td>468</td><td>32</td><td>0</td><td>0</td></tr><tr><td>Delete component</td><td>472</td><td>13</td><td>1</td><td>14</td></tr><tr><td>Substitute component</td><td>471</td><td>9</td><td>0</td><td>20</td></tr><tr><td>logP optimization</td><td>479</td><td>20</td><td>0</td><td>1</td></tr><tr><td>MR optimization</td><td>466</td><td>33</td><td>0</td><td>1</td></tr><tr><td>QED optimization</td><td>322</td><td>139</td><td>6</td><td>33</td></tr><tr><td>Forward reaction</td><td>107</td><td>207</td><td>11</td><td>175</td></tr><tr><td>Retrosynthesis</td><td>56</td><td>48</td><td>2</td><td>394</td></tr><tr><td>ChEBI-20 text to molecule</td><td>802</td><td>290</td><td>33</td><td>2175</td></tr></table>

## E.5 Qwen3-4B Whole-Trace Results

Qwen3-4B uses the same task IDs and whole-trace evaluation structure. Table 3 in Section 3 reports CoT, One-pass, and Refinement; it does not include MCP experiments, as 4B models have a relatively poor interactive capability.