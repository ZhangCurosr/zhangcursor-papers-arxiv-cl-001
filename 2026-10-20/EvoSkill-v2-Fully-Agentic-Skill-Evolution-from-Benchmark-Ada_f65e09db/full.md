# EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver

Dastin Huang dastin@sentient.xyz

Baran Nama baran@sentient.xyz

Sentient Labs

Salaheddin Alzubi

salaheddin@sentient.xyz

## Abstract

EvoSkill v2 is a three-agent system that edits the instructions (skills) an agent is given, leaving its model and harness intact. The user starts a run by describing a benchmark and supplying a grader, which a benchmark adaptor turns into a benchmark-specific plugin. A solver attempts the tasks, and an evolver reads the solver’s training rollouts and rewrites the skills at each epoch. We implement the evolver as one stateful coding agent with a bash tool for the whole run, unlike the two most common designs among the skill optimizers we surveyed: a sequence of single model calls, as in SkillOpt, and an agent that starts afresh each round, as in SkillBoost. We also measure how much the solver’s rollouts vary from one attempt to the next, give the evolver several attempts per training task, and accept every epoch’s skills instead of selecting among them on noisy scores. With fine-grained ground truth, EvoSkill v2’s skills raised the held-out pass rate above the stronger of SkillOpt and SkillBoost by 7.7 percentage points on SpreadsheetBench v2 and by 5.4 on $\tau ^ { 3 } .$ -bench banking. In both benchmarks and both feedback settings, EvoSkill v2’s evolver cost less per percentage point of held-out pass rate gained than either baseline’s optimizer. EvoSkill v2 also enables a less capable model to improve its own skills, demonstrated by a gain of up to 21.5 percentage points in the $\tau ^ { 3 }$ -bench banking experiment where Claude Haiku 5.5 powers both solver and evolver.

## 1 Introduction

What an LLM agent does on a task is set as much by the content of the model’s context window as by the model’s weights: a system prompt, a set of tool descriptions, a folder of instruction files it opens when a task matches the file’s own description of when it applies. That context has become an optimization target in its own right. GEPA [2] evolves the prompts of the model calls in a multi-step program, reflecting on the program’s execution traces (its reasoning, tool calls and tool outputs); SkillOpt [22] edits a single skill document (a short instruction file; defined below) of limited length, keeping an edit only if it passes a check on a validation set; a line of skill-library systems — Voyager [20], ExpeL [25], Agent Workflow Memory [21], SkillWeaver [26], ACE [24] — accumulates reusable procedures from an agent’s own experience. The appeal is that the optimized artifact is legible: unlike a weight update, an instruction can be read, disputed, and given unchanged to a diferent agent. Our method builds on self-reflection, in which an agent reads its own failed attempt and tries again [11, 14]. In EvoSkill v2, a separate agent (the evolver) reads the step-by-step records of the solver’s task attempts (its rollouts) and updates the solver’s skill set accordingly.

Earlier work has shown that such instructions can raise task success [2, 3, 10, 22, 24]; this paper asks how skill evolution is best done: what kind of optimizer writes the skills, what it remembers between rounds, how many attempts per task it sees, and how it decides which skills to keep (§6). The method rests on a narrow observation: an agent without suitable skills that fails a task usually fails it again. In our experiments, a Sonnet 5.5 Claude Code agent failed all three attempts on nine in ten of the SpreadsheetBench v2 held-out tasks that it failed at least once. On τ<sup>3</sup>-bench banking, a customer-service benchmark, a Haiku 5.5 Claude Code agent failed all three attempts on about six in ten such tasks (Figures 2 and 4). Where failures repeat, some of them are caused by missing instructions rather than by missing capability — and instructions are natural language, so another agent can write them. Anthropic introduced such instructions as skills [4] and released the format as an open standard, which other coding agents, among them OpenAI’s Codex and GitHub Copilot, have adopted [1]. We follow this definition: a skill is a folder holding a markdown file, optionally with scripts, whose short description tells the agent when to open it.

EvoSkill [3] is a skill evolution framework whose optimizer is two agents: a Proposer that analyzes the solver’s failures and proposes changes, and a Generator that writes those proposals into skill files. EvoSkill v2 merges the two into a single evolver and adds a benchmark adaptor (BA): an agent that turns a natural-language description of the benchmark (the user’s instruction) and the user’s own grader into the run’s fixed task list and the training material the evolver may read (§3). EvoSkill v2 is a layer on top of whatever agent harness an engineer already runs (e.g., Claude Code, Codex).

## Contributions.

1. A fully agentic skill-evolution system. Three coding agents: a BA that turns a prose instruction and the user’s grader into a frozen plug-in, a solver, and an evolver (§3).

2. Four design choices for the evolver (§6). The evolver is (i) an agent with a bash tool, not a sequence of single model calls; (ii) one stateful session for the whole run, not a fresh start each round; (iii) given several attempts per training task, because a single attempt is too noisy to tell a better skill from chance (Table 4); and (iv) made to accept every epoch’s skills, because a keep-if-better check on such noisy scores can reject a good edit (Table 5).

3. A comparison with two recent skill optimizers on two benchmarks. With the same solver, Claude Sonnet 5.5 as every optimizer and each optimizer given its own fine-grained feedback, EvoSkill v2’s held-out pass rate beat the stronger of SkillOpt [22] and SkillBoost [10] by 7.7 percentage points on Spreadsheet-Bench $\mathrm { v 2 }$ and 5.4 on $\tau ^ { 3 } \mathrm { . }$ bench banking (Figures 2 and 4). With pass or fail only, it matched SkillOpt on SpreadsheetBench $\mathrm { v 2 }$ and led it on $\tau ^ { 3 } .$ -bench banking, with an evolver that cost a fraction of SkillOpt’s optimizer (Tables 2 and 3).

4. Self-evolution of a less capable model. With Claude Haiku 5.5 powering both solver and evolver, EvoSkill v2 raised the held-out pass rate on $\tau ^ { 3 } \mathrm { . }$ bench banking by up to 21.5 percentage points (Figure 4).

## 2 Related Work

Single-call reflective optimizers revise a text artifact, such as a prompt or a skill document, without giving the revising model tools: the proposal step reads execution traces (records of what the agent under optimization did on each task and how it was scored) and emits a new artifact directly, with no inner loop of self-critique before the proposal is final. GEPA [2] is the clearest example. It evolves prompts under Pareto selection (keeping every candidate prompt that is best on at least one training example), making one model call per prompt it updates, in which the model reads the traces and writes the new prompt; its merge operator, the step that combines two candidate prompts into one, is ordinary code rather than a model call. The distinction is about the proposal step, not whether the optimizer iterates: every optimizer here has an outer loop of propose, evaluate, repeat. SkillOpt [22] sits between the single-call reflective optimizers and the agentic optimizers described next. It evolves a single skill document through small edits under a budget on how much each step may change, keeping an edit only if it strictly raises the validation score, but its update is a parallel proposal per small batch of examples, after which a model merges the proposals in stages, so one optimization step costs many calls rather than one. Agentic optimizers give tools to the model that proposes changes: Hone [19] uses a command-line coding agent as the step in GEPA that proposes changes; SIA [8] jointly evolves the agent’s harness code and weights; SkillOpt-Lite [12] drives a production coding agent through a loop that repeatedly edits a prompt file and keeps or discards each edit by its score on a fixed validation split. Our first generation [3] introduced the agentic skill-folder loop this report extends, including the rule that the ground truth is shown only to the optimizer, never to the solver. The line of work on skill libraries, among them Voyager [20], ExpeL [25], Agent Workflow Memory [21], SkillWeaver [26], and ACE [24], builds up a library of reusable procedures from the agent’s own past runs, usually written by the same agent that solves the tasks, and kept without any check on held-out data; CoEvoSkills [23] evolves skill packages against a verifier that stands in for the real scorer and is itself evolved alongside the skills. Every system above is an optimizer, and in each the measurement instrument (the split, the scorer, the models that run the tasks) is chosen outside the optimizing agent and cannot be changed by it: GEPA [2] takes its validation set, task model and metric as inputs from whoever runs it, and never changes them afterwards; SIA [8] rewrites the source code of the agent being improved, starting from an initial version, while the evaluator and its ground truth stay in the task package. Setting the instrument outside the optimizing agent is a valid design, and we do not claim to improve on it. What difers here is that part of building the instrument is delegated to an agent (§3.1): the user supplies the grader and the data, and an agent turns the user’s prose description into the task list, with each task’s set, and into the material the evolver may read. Once an agent does that, it must not also be the agent that writes the skills, or a low score can be raised by changing the split (which examples count, and how many) or what the evolver may see, instead of by writing a better skill. None of the systems surveyed delegates this step to an agent, so none needs to separate that agent from the one that writes the skills. The failure modes of optimizing against a validation set are known, but the work on such optimizers rarely audits its own runs for them: the hazard of re-using one held-out set for many adaptive decisions is well studied [5] and shortcut learning, in which a model picks up spurious cues that predict the label without solving the task, is documented for trained models [6, 7], but here the optimizer is the learner. It can write rules that fit the training tasks and do not carry over to other tasks: in our SpreadsheetBench v2 run and our τ<sup>3</sup>-bench run without training answers, the training scores rose far more than the validation scores (§5.1, §5.2). It can also find quirks in how the task was built, including quirks of the grader itself (§3.4). Huang et al. [9] find that skills extracted by strong models do not reliably benefit less capable models; in this report the evolved skills were evaluated only with the solver model they were written for.

## 3 The EvoSkill v2 System

EvoSkill v2 is three coding agents layered over an existing agent harness: a benchmark adaptor (BA) that turns the user’s description of a benchmark into the run’s fixed list of tasks before any skill is written; a solver that attempts the tasks with the evolved skills; and an evolver that reads the solver’s attempts and rewrites the skills. In our implementation, all three are Claude Code sessions (Figure 1).

![](images/513de820717fd669d3bc738c7dce710eee63c3eda187921799730a77e7c56655.jpg)  
Figure 1: EvoSkill v2. Before the run, the benchmark adaptor (BA) turns the user’s instruction, data, split and grader into a frozen plug-in that lists every task with its set, and a reference folder of training material. In each epoch the solver attempts every training and validation task three times, and the user’s grader scores each attempt. The evolver, one Claude Code session for the whole run, reads only the training attempts, their scores and the reference folder, and writes the next skills. The last epoch’s skills are tested on the held-out set.

## 3.1 The Benchmark Adaptor

In v1, a new benchmark had to fit one of the input formats it supports. A benchmark whose output is a file or a conversation, such as SpreadsheetBench v2’s workbooks or τ<sup>3</sup>-bench’s customer dialogues, first had to be converted by hand into one of them, and v1 has limited support for custom graders.

BA is introduced in v2 to automate the per-benchmark adaptation. The user hands over three things: a short instruction in natural language (the benchmark, where its data is, the split to use or how to make one, what the solver gets and must produce, and the run size), the dataset, and a grader. The grader is a function grade(task, snapshot): it takes the solver’s working directory as it was at the end of an attempt and returns a score from 0 to 1. The BA generates two things. The first is a plug-in: a short Python module whose one function lists every task with its set (train, validation or test) and with the fields that the user’s grader reads. The second is a reference folder: everything the user handed over that the evolver may read, which is everything except material about validation or test tasks and the grader itself.

## 3.2 The solver

The solver is a Claude Code agent with its stock system prompt. The task reaches it as the first user message, and the solver works in its own container. A skill is a folder holding a SKILL.md file and, optionally, scripts. Skills reach the solver by progressive disclosure [4]: Claude Code keeps every skill’s name and one-line description in the solver’s context at all times, and the solver loads the full body by a tool call only when that description

matches the task in front of it. So the descriptions, which decide when a skill is loaded, are themselves evolvable.   
In the runs reported here each attempt had 1 GiB of memory; web search, web fetch and subagents were disabled;   
and the container could reach no network host except the model API.

## 3.3 The evolver, and what it is given

The evolver is a single Claude Code session that lasts the whole run, in a container on the same machine as the solver’s containers. It gets Claude Code’s own system prompt with a fixed text appended once and one short message per epoch to analyze the solver’s rollouts, diagnose the pitfalls that a skill could fix, and write the next skill set into a staging folder (both texts in Appendix A). Because the session continues from one epoch to the next, the evolver keeps its own earlier analysis in its context. Claude Code may compact that context when it grows too large, so the evolver also has a private folder, kept across epochs, for anything it wants to keep. Table 1 lists what it finds on its machine.

<table><tr><td>Path on the evolver&#x27;s machine</td><td>What it holds</td><td>Access</td></tr><tr><td>/evo/traces/epoch_K/</td><td>each training attempt&#x27;s Claude Code session log, unchanged, and the large tool outputs it saved to files</td><td>read</td></tr><tr><td>/evo/scores/epoch_K.json</td><td>each training attempt&#x27;s score (0 to 1) or grader_failed</td><td>read</td></tr><tr><td>/evo/solver_workspaces/epoch_K/</td><td>the solver&#x27;s working directory at the end of each training attempt</td><td>read</td></tr><tr><td>/evo/skills/epoch_K/</td><td>the skill set that epoch K&#x27;s solver carried</td><td>read</td></tr><tr><td>/evo/reference/</td><td>the BA&#x27;s reference folder (training material only)</td><td>read</td></tr><tr><td>/evo/staging/</td><td>the next skill set; it starts as the set the evolver left there</td><td>write</td></tr><tr><td>/evo/evolver_workspace/</td><td>the evolver&#x27;s private folder, kept across epochs</td><td>write</td></tr></table>

Table 1: What the evolver finds on its machine; epoch 0 ran with no skills. What never reaches it: §3.4.

## 3.4 Keeping held-out tasks away from the evolver

A common pitfall in self-improving systems is that the optimizer reads the test split directly and overfits to it — or worse, manipulates the split itself to drop the tasks it fails on. Either one raises the score without improving a single skill, and neither leaves a mark: the run looks exactly like a run that worked. In EvoSkill v2 the split is fixed in the frozen plug-in before the first skill is written, and the evolver’s machine receives only training material: the reference folder, and the solver’s training attempts with their scores. Validation and test attempts, and their scores, are never uploaded to it. This isolation is set by the orchestration layer, so it holds even if the evolver tries to work around it.

The isolation ensures that a score obtained with an evolved skill was not obtained by reading held-out answers — but a higher score does not necessarily mean the skill is a good one. When the scoring mechanism itself is flawed, the evolver may optimize in a direction the user did not intend, even though the isolation works exactly as designed.

A grader defect that an earlier version found. An earlier version of this system, set to maximize the SpreadsheetBench v2 score, found and exploited a defect in the benchmark’s grader. The grader compared the value that a workbook stores beside each formula, as the spreadsheet application last computed it, with the expected workbook, and did not recompute the formulas. The evolver wrote a skill that told the solver to repair the formula text and never to recalculate; the grader used here recalculates every formula (Appendix A). An optimizer can find a defect like this one, so the grader needs checking before the run; this is one reason why the user, not an agent, supplies it.

## 4 Experimental Setup

We use two benchmarks (§4.1, §4.2), chosen because both are hard: the best scores on their public leaderboards are 54.5% on SpreadsheetBench v2 [18] and 55.2% on τ<sup>3</sup>-bench banking [16], which leaves substantial room for improvement. Both are also all-or-nothing: an episode, one attempt at one task, scores 1 only when every check passes. A skill therefore counts only if it is solid enough to make the solver pass a task that it fails without the skill; making an attempt partly right earns nothing.

For each benchmark we wrote the BA’s user instruction by hand (Appendices A.2 and A.3). The BA and the evolver are Claude Code agents powered by Claude Sonnet 5.5, except in two τ<sup>3</sup>-bench runs whose evolver is powered by the solver’s own model, Claude Haiku 5.5 (§5.2).

## 4.1 SpreadsheetBench v2

In SpreadsheetBench v2 [27], the solver, a Claude Code agent powered by Claude Sonnet 5.5, edits a real Excel workbook, and the result is compared cell by cell with a reference workbook; our grader first recalculates every formula in LibreOfice (§3.4). We use three of the benchmark’s four categories, Debugging, Financial\_Model and Template (297 tasks); the fourth, Visualization, is the least challenging of the four in the benchmark’s own evaluation [27], and its grading needs a vision-language model. Some tasks could not be used as given, for example because a reference workbook’s saved values difer from a fresh recalculation. After filtering, 218 tasks remain, split into 57 training, 57 validation and 104 held-out tasks.

Tasks that share a reference workbook, a formula family or a Template folder go into the same set, so no answer is shared across sets. Each attempt has a 20-minute wall-clock limit and a 45-turn limit. Appendix A gives the split and the grader in full.

In the EvoSkill v2 runs, we give the BA a short user instruction that describes SpreadsheetBench v2 (Ap pendix A.2) and the benchmark’s dataset as downloaded from its Hugging Face release [17]. The BA is instructed to keep the grader and everything about the validation and held-out tasks out of the evolver’s reference folder (Appendix A.1), and we checked by hand that it did. The evolver may read everything else in that folder, including the training tasks’ expected workbooks. SkillOpt’s and SkillBoost’s optimizers read each training attempt’s session log and whether it passed. For each failed attempt they also get the wrong cells, with the expected and actual values, as the oficial implementations of both methods provide: SkillOpt every wrong cell, SkillBoost the first.

## 4.2 τ<sup>3</sup>-bench banking

In τ<sup>3</sup>-bench banking [13], the solver, a Claude Code agent powered by Claude Haiku 5.5, plays a bank’s customerservice agent talking with a simulated customer. Each task is graded on database state (the bank’s final database must equal the one that the expected tool calls produce) or on action (the agent must make each expected call with the expected arguments).

After data cleansing, we split the remaining 70 tasks into 19 training, 20 validation and 31 held-out tasks, keeping related tasks in one set.

In the EvoSkill v2 runs, we give the BA a short user instruction that describes τ<sup>3</sup>-bench banking (Appendix A.3) and the benchmark’s banking tasks and the files they use, from its public repository [15]. As on Spreadsheet Bench v2, the BA keeps the grader and everything about the validation and held-out tasks out of the evolver’s reference folder, and we checked that it did. The evolver may read everything else. SkillOpt’s and SkillBoost’s optimizers read each training attempt’s conversation and whether it passed. For each failed attempt they also get what the grader found wrong. SkillOpt gets every diference, SkillBoost the first.

## 5 Results

## 5.1 SpreadsheetBench v2

We compare EvoSkill v2 with SkillOpt [22] and SkillBoost [10] on the same split. We keep the solver configuration, data split and grading metrics identical across 3 experiments. We run the solver 3 times, independently, on each of the 104 held-out tasks for each skill. With EvoSkill v2’s skills the solver passed 126 of 312 held-out episodes, against at most 102 with the skills of either baseline (Figure 2, left).

Sparse feedback signal. In the first experiment, EvoSkill v2’s evolver has far more ground truth for diagnosing the solver’s failures than SkillOpt’s and SkillBoost’s optimizers, so it is hard to tell whether EvoSkill v2’s higher score comes from its evolution loop itself. We therefore ran a second experiment that changes what every optimizer sees: only whether each training attempt passed (Figure 3). With only this sparse signal, EvoSkill v2 is on par with SkillOpt on the held-out set at less than a tenth of SkillOpt’s optimizer cost, and SkillBoost passed fewer episodes at the highest optimizer cost (Table 2).

<table><tr><td rowspan="2"></td><td colspan="3">Fine-grained ground truth</td><td colspan="3">Pass or fail only</td></tr><tr><td>Held-out</td><td>Cost</td><td>Per point gained</td><td>Held-out</td><td></td><td>Cost Per point gained</td></tr><tr><td>No skill</td><td>81</td><td></td><td></td><td>81</td><td></td><td></td></tr><tr><td>SkillBoost</td><td>101</td><td>$13.95</td><td>$2.18</td><td>85</td><td>$13.66</td><td>$10.65</td></tr><tr><td>SkillOpt</td><td>102</td><td>$8.56</td><td>$1.27</td><td>107</td><td>$6.65</td><td>$0.80</td></tr><tr><td>EvoSkill v2</td><td>126</td><td>$12.84</td><td>$0.89</td><td>105</td><td>$0.53</td><td>$0.07</td></tr></table>

Table 2: SpreadsheetBench v2: held-out episodes passed (of 312) and the optimizer’s cost, in Claude Code’s own listprice estimates for the optimizer alone; the solver’s attempts are not included. Per point gained: the cost divided by the percentage points of held-out pass rate gained over no skill. The two EvoSkill v2 runs used diferent Claude Code versions, so their costs are not directly comparable. Bold marks each column’s best: the highest score or the lowest cost.

## The optimizer sees fine-grained ground truth

![](images/10b3550e9e738430cad16670e901da44f9fd50e009d9baeb98ddff18ffc8eaa4.jpg)

![](images/c48b833e412e40e83b055b80b75a92a8d92ce04e17a45074a49be0e98b4aac68.jpg)  
Figure 2: SpreadsheetBench $\mathrm { v 2 , }$ the runs whose optimizer sees fine-grained ground truth (§4.1). Bars: share of episodes passed (pass@1), and of tasks passed in at least one of 3 attempts (pass@3) and in all three (all@3); the number above a bar is the count. Per epoch: each EvoSkill v2 point is of 171 episodes, each baseline point of 57; SkillBoost has no validation score, and its flat segments are rounds that kept no candidate; grey labels are values two lines share.

![](images/830304d47231e2c8aaf8a07ed7617985f704c8aa686aa18422b030baa2cf2814.jpg)

![](images/c8e7d3470d1b58d48d8c3f671f15edb28187787bb0e3915e3d14012e7cecfa8c.jpg)  
Figure 3: SpreadsheetBench $\mathbf { v } 2 ,$ the runs whose optimizer sees only whether each training attempt passed; EvoSkill v2’s evolver reads the BA’s reference folder without the training tasks’ expected workbooks. Panels as in Figure 2; EvoSkill v2 ran with its own no-skill training and validation legs.

## 5.2 $\tau ^ { 3 } \mathrm { . }$ bench banking

We compare the same three methods on $\tau ^ { 3 } .$ -bench banking, with Claude Haiku 5.5 as the solver. As on SpreadsheetBench v2, the solver, its limits per attempt, the customer, the grader and the split are the same for every run. With fine-grained ground truth, where EvoSkill v2’s evolver reads the 19 training task files with each task’s customer script and expected tool calls (the training answers), EvoSkill v2’s skills passed 48 of 93 held-out episodes, against at most 43 with the skills of either baseline (Figure 4, left).

Sparse feedback signal. With only pass or fail, EvoSkill $\mathrm { v } 2 \mathrm { s }$ skills again passed the most held-out episodes (Figure 4, right), and in both settings its evolver cost far less than either baseline’s optimizer (Table 3).

![](images/575b059102cd3d1b5f8cddb83fda13d89fd5737163db70eb3796e7ab48b2b541.jpg)

Figure 4: $\tau ^ { 3 } .$ -bench banking held-out test, for the runs whose optimizer sees fine-grained ground truth (§4.2) and those whose optimizer sees only whether each training attempt passed. Bars as in Figure 2.
<table><tr><td rowspan="2"></td><td colspan="3">Fine-grained ground truth</td><td colspan="3">Pass or fail only</td></tr><tr><td>Held-out</td><td>Cost</td><td>Per point gained</td><td>Held-out</td><td></td><td>Cost Per point gained</td></tr><tr><td>No skill</td><td>26</td><td></td><td></td><td>26</td><td></td><td></td></tr><tr><td>SkillBoost</td><td>36</td><td>$6.98</td><td>$0.65</td><td>23</td><td>$8.64</td><td>no gain</td></tr><tr><td>SkillOpt</td><td>43</td><td>$5.55</td><td>$0.30</td><td>37</td><td>$6.22</td><td>$0.53</td></tr><tr><td>EvoSkill  $\mathbf { v } 2 ,$  Haiku evolver</td><td>46</td><td>$1.87</td><td>$0.09</td><td>33</td><td>$0.73</td><td>$0.10</td></tr><tr><td>EvoSkill  ${ \mathrm { v } } 2$ </td><td>48</td><td>$0.99</td><td>$0.04</td><td>41</td><td>$1.55</td><td>$0.10</td></tr></table>

Table 3: $\tau ^ { 3 } .$ -bench banking: held-out episodes passed (of 93) and the optimizer’s cost, as in Table 2; here every run used the same Claude Code version. Bold marks each column’s best; “no gain”: SkillBoost’s skills passed fewer held-out episodes than no skill.

Self-evolution. On $\tau ^ { 3 } .$ bench banking, every run above powers the evolver with a stronger model than the solver’s (Sonnet 5.5 against Haiku 5.5), so the held-out gain combines a better evolver model with the EvoSkill v2 harness. To separate the two, and to test whether the harness lets a model improve its own skills, we ran a third experiment in which the solver’s own model, Claude Haiku 5.5, powers the evolver, with each of the two feedback signals. With fine-grained ground truth, its skills came within two held-out episodes of the Sonnet 5.5 evolver’s; with pass or fail only, they gained about half as much over no skill (Figure 4 and Table 3). Figure 5 shows the four EvoSkill v2 runs per epoch.

![](images/dea2f45dbb27df59d958afcde3c8b0da023a7b72ca36065da5e40ac062917dbe.jpg)  
Figure 5: $\tau ^ { 3 } .$ -bench banking: EvoSkill v2’s pass rate per epoch on the training (left) and validation (right) tasks, each point of 57 or 60 episodes (3 attempts on 19 training or 20 validation tasks), with Sonnet 5.5 or Haiku 5.5 as its evolver, each with and without the training answers (§5.2).

## 6 Discussion: Four Design Choices

With fine-grained ground truth, EvoSkill v2’s skills passed more held-out episodes than either baseline’s on both benchmarks (Figures 2 and 4). Four choices in EvoSkill v2’s design, the same on both benchmarks, set it apart from both baselines.

6.1 The evolver is an agent with a bash tool, not a sequence of single model calls Most recent skill optimizers, among them SkillOpt, GEPA [2] and ACE [24], edit skills through single model calls, each answering from the text in its prompt. In the SpreadsheetBench v2 run with fine-grained ground truth, SkillOpt made 40 such calls: most analyze a batch of training attempts, and the rest merge or rank their proposals, revise the skill, or write notes for later calls. None of them can open a workbook, recalculate a formula or check a conclusion before proposing an edit. We argue that an agent suits the job better, because it can check an explanation of a failure before it writes that explanation into a skill; SkillOpt-Lite [12] and CoEvoSkills [23] also give their optimizer a shell.

EvoSkill v2’s evolver is a Claude Code agent with a bash tool. In our experiments it used bash mostly to run Python, and in some of its turns it also handed groups of training tasks to subagents with bash tools of their own. On SpreadsheetBench v2 with fine-grained ground truth, one script it wrote recalculated a solver’s output workbook in LibreOfice and compared it cell by cell with the training task’s expected workbook, so the evolver could test an explanation of a failure before writing it into a skill.

## 6.2 The evolver is stateful: one session for the whole run

SkillBoost starts a new optimizer session every round, and each of SkillOpt’s calls starts from its own prompt, so what an earlier round or call worked out reaches a later one only as text it passed on, such as the skill or SkillOpt’s notes. Among the agent-based optimizers we surveyed that run in rounds, starting a new session each round is the more common design (8, against 5 that keep one session). EvoSkill v2’s evolver is one Claude Code session for the whole run: in the SpreadsheetBench v2 run with fine-grained ground truth, its three turns continued one session, whose context Claude Code summarized four times when it filled. At each turn the evolver has every earlier epoch’s training attempts and scores, every skill set it wrote before (Table 1), its notes, its own earlier reasoning and the scripts in its private folder; in its second and third turns, 20 of its bash commands re-ran scripts it had written in an earlier turn. Each turn is therefore a chance to keep, revert or combine anything tried so far, not only to edit the latest version.

## 6.3 Several attempts per task make the evolver’s evidence less noisy

The solver does not pass or fail a task the same way every time. In every training and validation leg, some tasks passed some of their three attempts and failed others: 7% to 21% of them on SpreadsheetBench v2 and 16% to 53% on $\tau ^ { 3 } \mathrm { . }$ bench banking (Table 4). With one attempt per task, as SkillOpt, SkillBoost and 35 of the other 43 skill optimizers we surveyed run, each of those tasks appears as a pass or a fail by chance. With three, the evolver can tell a failure that repeats from one that does not.

<table><tr><td></td><td colspan="3">Training</td><td colspan="3">Validation</td></tr><tr><td>Pass rate</td><td>3/3</td><td>0/3</td><td>1-2/3</td><td>3/3</td><td>0/3</td><td>1-2/3</td></tr><tr><td colspan="7">SpreadsheetBench v2 (57 tasks per set)</td></tr><tr><td>No skill</td><td>10</td><td>43</td><td>4</td><td>5</td><td>47</td><td>5</td></tr><tr><td>Turn 1</td><td>27</td><td>19</td><td>11</td><td>14</td><td>38</td><td>5</td></tr><tr><td>Turn 2</td><td>36</td><td>9</td><td>12</td><td>10</td><td>38</td><td>9</td></tr><tr><td>Turn 3</td><td></td><td></td><td></td><td>11</td><td>38</td><td>8</td></tr><tr><td colspan="7">τ3-bench banking (19 training, 20 validation tasks)</td></tr><tr><td>No skill</td><td>3</td><td>9</td><td>7</td><td>3</td><td>12</td><td>5</td></tr><tr><td>Turn 1</td><td>10</td><td>6</td><td>3</td><td>7</td><td>5</td><td>8</td></tr><tr><td>Turn 2</td><td>7</td><td>2</td><td>10</td><td>6</td><td>6</td><td>8</td></tr><tr><td>Turn 3</td><td></td><td></td><td></td><td>7</td><td>4</td><td>9</td></tr></table>

Table 4: Tasks by how many of their three attempts passed, with fine-grained ground truth (on τ<sup>3</sup>-bench, the Sonnet 5.5 evolver). Turn k: the skills of the evolver’s k-th turn; —: not run.

<table><tr><td>Set</td><td></td><td>Passed</td><td>Newly failed</td></tr><tr><td>SpreadsheetBench v2 (57 tasks per set)</td><td></td><td></td><td></td></tr><tr><td>SkillOpt, step 2</td><td>Validation</td><td>17 against 19</td><td></td></tr><tr><td>SkillOpt, step 3</td><td>Validation</td><td>17 against 19</td><td></td></tr><tr><td>SkillOpt, final skill</td><td>Validation</td><td>18 against 19</td><td></td></tr><tr><td>SkillBoost, round 3</td><td>Training</td><td>20 and 21 against 21</td><td>4 and 2</td></tr><tr><td>τ3-bench banking (19 training, 20 validation tasks)</td><td></td><td></td><td></td></tr><tr><td>SkillOpt, step 2</td><td>Validation</td><td>9 against 9</td><td></td></tr><tr><td>SkillOpt, step 3</td><td>Validation</td><td>8 against 9</td><td></td></tr><tr><td>SkillOpt, final skill</td><td>Validation</td><td>8 against 9</td><td></td></tr><tr><td>SkillBoost, round 1</td><td>Training</td><td>7 and 9 against 5</td><td>1 and 1</td></tr><tr><td>SkillBoost, round 3</td><td>Training</td><td>10 and 7 against 8</td><td>1 and 3</td></tr></table>

Table 5: Candidates that SkillOpt and SkillBoost rejected in the runs with fine-grained ground truth, each scored with one attempt per task against the skill held at the time. SkillOpt keeps a candidate only if it passes more validation tasks; SkillBoost only if it passes more training tasks and newly fails fewer than 5% of them.

## 6.4 Every epoch’s skills are accepted

Most skill optimizers keep a new candidate skill only if it passes a check on its scores: SkillOpt requires a higher validation score, and SkillBoost requires more training tasks passed with few newly failed. With one attempt per task, the sampling noise discussed in §6.3 could easily flip a decision on whether or not keep a candidate. In the runs with fine-grained ground truth, every rejection by SkillOpt and SkillBoost turned on 2 tasks or fewer (Table 5), fewer than the tasks whose outcome varies between attempts (Table 4). In other words, had the attempts been sampled again, any of these rejected candidates could have been kept.

Several attempts per task reduce this noise but cannot remove it, so any rule that keeps or rejects a skill set by its rollout scores stays exposed to it. EvoSkill v2 has no rejection step: it keeps every skill set the evolver writes, and tests the last epoch’s on the held-out set instead of choosing an epoch by a validation score the evolver never sees. A bad edit is not lost. The evolver keeps every earlier epoch’s skill sets and rollouts (§6.2), so the next epoch’s training attempts show what the edit made worse, and a later turn can revert it: the failed edit becomes evidence instead of being thrown away.

The last epoch’s skills are not always the best on validation, but choosing the epoch with the best validation score would not have done better overall. In all six EvoSkill v2 runs, validation rose from no skill to the last epoch (Figures 2, 3 and 5). In four of them the last epoch also scored highest or tied, so choosing by validation would have kept the same skills. It would have chosen an earlier epoch only in the other two, and there the earlier epoch’s skills passed 15 fewer held-out episodes on SpreadsheetBench v2 with fine-grained ground truth and one more on $\tau ^ { 3 } .$ -bench banking.

## 7 Conclusion

EvoSkill v2 layers three coding agents (a benchmark adaptor, a solver, and an evolver that rewrites the solver’s skills) on top of an existing harness. Its evolver is one stateful agent with a bash tool that sees several attempts per training task. With fine-grained ground truth, its skills passed more held-out episodes than those of SkillOpt and SkillBoost on both SpreadsheetBench v2 and $\tau ^ { 3 } .$ -bench banking; with pass or fail only, it was on par with SkillOpt on SpreadsheetBench v2 and ahead of it on τ<sup>3</sup>-bench banking, at a fraction of SkillOpt’s optimizer cost; and it let Claude Haiku 5.5 improve its own skills on τ<sup>3</sup>-bench banking. EvoSkill v2 tests the last epoch’s skills; in our runs, choosing among epochs by validation score would not have done better overall (§6.4).

## References

[1] Agent Skills. Agent skills: an open format for giving agents new capabilities. https://agentskills.io; specification at https://github.com/agentskills/agentskills, 2025. Accessed 16 October 2026.

[2] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457, 2025.

[3] Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. Evoskill: Automated skill discovery for multi-agent systems. arXiv preprint arXiv:2603.02766, 2026.

[4] Anthropic. Agent skills. https://docs.claude.com/en/docs/agents-and-tools/agent-skills/ overview, 2025. Claude platform documentation; Agent Skills announced 16 October 2025.

[5] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. The reusable holdout: Preserving validity in adaptive data analysis. Science, 349(6248):636–638, 2015.

[6] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

[7] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R Bowman, and Noah A Smith. Annotation artifacts in natural language inference data. In NAACL, 2018.

[8] Prannay Hebbar, Yogendra Manawat, Samuel Verboomen, Alesia Ivanova, Selvam Palanimalai, Kunal Bhatia, and Vignesh Baskaran. Sia: Self improving ai with harness & weight updates. arXiv preprint arXiv:2605.27276, 2026.

[9] Zisu Huang, Jingwen Xu, Yifan Yang, Ziyang Gong, Qihao Yang, Muzhao Tian, Xiaohua Wang, Changze Lv, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Xue Yang, Dongdong Chen, Xiaoqing Zheng, and Chong Luo. From raw experience to skill consumption: A systematic study of model-generated agent skills. arXiv preprint arXiv:2605.23899, 2026.

[10] Hongqiang Lin, Chao Liu, Xiaofan Bai, Xuan Jin, Yuhong Li, Nenggan Zheng, and Xipeng Cao. Rethinking self-evolution: A constrained exploration-exploitation process for mitigating skill overfitting. arXiv preprint arXiv:2607.26643, 2026.

[11] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, 2023. arXiv:2303.17651.

[12] Yifei Shen, Bo Li, and Xinjie Zhang. Skillopt-lite: Better and faster agent self-evolution with one line of vibe. arXiv preprint arXiv:2607.03451, 2026.

[13] Quan Shi, Alexandra Zytek, Pedram Razavi, Karthik Narasimhan, and Victor Barres. τ -knowledge: Eval uating conversational agents over unstructured knowledge. arXiv preprint arXiv:2603.04370, 2026.

[14] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023. arXiv:2303.11366.

[15] Sierra Research. τ -bench repository (tau2-bench), commit 363133a. https://github.com/ sierra-research/tau2-bench, 2026. Commit 363133a, 29 July 2026.

[16] Sierra Research. τ-bench Leaderboard: τ<sup>3</sup>-Banking. https://taubench.com/leaderboard/?benchmark= knowledge, 2026. Accessed 12 October 2026.

[17] SpreadsheetBench Team. SpreadsheetBench-v2 (dataset). https://huggingface.co/datasets/KAKA22/ SpreadsheetBench-v2, 2026. Accessed 13 October 2026.

[18] SpreadsheetBench Team. SpreadsheetBench Leaderboard (V2 – Full). https://spreadsheetbench. github.io/, 2026. Accessed 12 October 2026.

[19] T. Waldin. Hone: Agentic harness optimization on the gepa api. https://github.com/twaldin/hone, 2026.

[20] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

[21] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024.

[22] Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. Skillopt: Executive strategy for self-evolving agent skills. arXiv preprint arXiv:2605.23904, 2026.

[23] Hanrong Zhang et al. Coevoskills: Self-evolving agent skills via co-evolutionary verification. arXiv preprint arXiv:2604.01687, 2026.

[24] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618, 2025.

[25] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: LLM agents are experiential learners. arXiv preprint arXiv:2308.10144, 2023.

[26] Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills. arXiv preprint arXiv:2504.07079, 2025.

[27] Jian Zhu, Yuzheng Zhang, Zeyao Ma, Bohan Zhang, Armin Schoepf, Daniel Woloch, Peter Yiliu Wang, Guangyu Robert Yang, Samuel Jacob, Siddharth Nagisetty, Abhiram Chundru, Jean Lin, Spencer Mateega, and Jing Zhang. SpreadsheetBench 2: Evaluating agents on end-to-end business spreadsheet workflows. arXiv preprint arXiv:2606.29955, 2026.

## Appendix A Split, Grader and Prompts

SpreadsheetBench v2: the split. Per category (Debugging / Financial\_Model / Template), the 57/57/104 split is 19/20/18 training, 20/17/20 validation and 28/36/40 held-out tasks. Tasks that share a reference workbook, a Financial\_Model formula family or a Template folder are in one set.

SpreadsheetBench v2: the grader. The grader is the benchmark’s evaluation script, which recalculates nothing, with a formula recalculation added (§3.4): it opens the solver’s saved workbook in LibreOfice with every formula recomputed, then compares it with the reference workbook; a match scores 1, anything else 0. Where LibreOfice cannot compute a formula that the solver left unchanged, the grader uses the value stored in the input workbook. A failed recalculation is a grader failure, not a score.

## A.1 The benchmark adaptor’s guidance (appended to Claude Code’s system prompt)

The BA (Claude Sonnet 5.5) ran once for each benchmark, SpreadsheetBench v2 and τ<sup>3</sup>-bench, with this text unchanged.

```markdown
You are the benchmark adapter of a skill-evolution run. You turn the user instruction into one file that describes the benchmark
to the loop (the program that drives the run) and one directory the evolver will use.
## What a skill-evolution run does
The run evolves a skill set for a solver agent. A skill is a folder the solver can load while it works: instructions and scripts
it can run. Each epoch, the solver attempts the training tasks with the current skill set; then an evolver agent reads the
traces of these attempts and the grader's scores and writes the next skill set. The tasks the run uses should be divided
into three sets:
- **train**: the tasks the skills are learned from. The solver runs them every epoch, and the evolver reads those attempts and
everything you keep for it (below).
**validation**: the tasks that measure each epoch's skill set. The solver runs them every epoch; the evolver sees nothing of
them.
- **test**: the tasks held out until the end, when the solver is measured on them once with the final skill set and once without
skills.
If the user instruction gives no train/validation/test split, create one yourself. Do not leak any validation or test answer to
anything the evolver has access to.
## Your job
Write `benchmark_plugin.py` into `{setup_dir}/` and fill `{reference_dir}/` (use your working directory for anything else):
`benchmark_plugin.py`: a Python module that describes the benchmark to the loop. It defines one function: `list_tasks(root)`,
called with the data root as `root`, returns one `Task` (defined in `plugin_api.py`) for every task the run uses, each with
its set: train, validation or test. It reads only files under `root`, building every path as `root/...`; anything else it
needs, such as the split, goes in its own source. It imports only the Python standard library and `plugin_api`. It never
runs the grader, starts a process or uses the network. The loop grades every rollout with the user's grader, `
benchmark_grader.py` in `{setup_dir}/`; fill each `Task` with what that grader reads.
`{reference_dir}/`: everything the user handed over that the evolver may use. Copy into it every artifact you were given (the
data, documentation, anything else) except two kinds: anything that relates to validation or test tasks, and the grader
itself, meaning its code and anything that describes how it grades. The evolver gets only scores from the loop's grader and
never needs to know how it works. Keep the rest whole and in its original layout. The evolver sees this directory as its
only view of the benchmark beyond the solver's attempts and their scores; it may read and run anything in it. **Nothing** in
it, whether you copied it or wrote it, may contain either of those two kinds.
## What you can read
`plugin_api.py`, a skeleton plug-in and the user's grader `benchmark_grader.py` are in `{setup_dir}/`. You can also read the data
root and every other file the user instruction names, at the paths it gives, the grader's code included. Test your plug-in
on the data root before you finish.
```

A.2 The user instruction for SpreadsheetBench v2 (the benchmark adaptor’s message)  
```markdown
### 1. Benchmark, data and split
SpreadsheetBench v2, the three non-Visualization categories: Debugging, Financial_Model and Template (297 tasks: 100, 100 and 97).
The data root also holds a fourth folder, Visualization, which is not part of this run. A task gives the solver a workbook
and an instruction; the solver edits the workbook with Python and saves it.
The data root is `/data/spreadsheetbench-v2`, with one `dataset.json` per category; each record has `id`, `instruction`, `
spreadsheet_path`, `golden_response_path` and `answer_position`; the two paths are relative to that category's folder.
Several records share one golden_response_path (a Debugging project has ten tasks on one reference workbook, a
Financial_Model project five); tasks that share a reference workbook share an answer, and no such group spans two of train,
val and test. A task id is the category plus the record id (`<Category>/<NN_MM>`).
Use the given split: `/data/split.json`, 57 train / 57 val / 104 test.
### 2. What the solver gets and must produce
Per task, the solver gets the instruction and the input workbook in its working directory. It must save the modified workbook to
the output path it is given.
### 3. Grader
The grader is `benchmark_grader.py`, given next to `plugin_api.py`. Its `grade(task, snapshot)` scores one rollout from 0 to 1;
the snapshot is the solver's working directory when it exited. It reads the task's `inputs[0]` (the input workbook),
answer_files[0]` (the reference workbook) and `grading["answer_position"]`. The loop runs it; you do not need to.
### 4. Environment
The solver and the grader run in the image built from `scripts/daytona/Dockerfile.ssb_leg` in the same repository. The solver's
Python is 3.11 with openpyxl 3.1.3, pandas 2.2.0 and numpy 1.26.4.
### 5. Run size
3 epochs · 3 samples per task.
```

## A.3 The user instruction for τ<sup>3</sup>-bench (the benchmark adaptor’s message)

### 1. Benchmark, data and split   
τ<sup>3</sup>-bench, the banking domain only (\`banking\_knowledge\`). A task is a conversation between a bank's customer service agent (the   
solver) and a simulated customer who has a goal; the agent handles it according to the bank's policy, using the bank's tools   
and its knowledge base.   
The data root is \`/data/tau3\_data\_363133a\`. \`tau2/domains/banking\_knowledge/\` holds \`db.json\` (the bank's database), \`documents/\`   
(the knowledge base, 698 documents), \`prompts/\` (the agent's policy texts) and \`tasks/\` with one file per task, \`task\_NNN.   
json\` (the customer's scenario, the database's starting state and how the task is judged). \`tau2/user\_simulator/\` holds the   
customer simulator's guidelines. \`.embeddings\_cache/\` at the data root holds the knowledge base's precomputed embeddings (   
OpenAI \`text-embedding-3-large\`, 698 documents), used by the dense search. A task id is \`banking\_knowledge/\` plus the task   
file's name without \`.json\` (e.g. \`banking\_knowledge/task\_004\`).   
Use the given split: \`/data/split.json\`, 19 train / 20 validation / 31 test.   
### 2. What the solver gets and must produce   
Per task, the solver is the bank's agent in one conversation. It gets the bank's policy for the \`alltools\` knowledge-base setup (   
search with \`KB\_search\_bm25\` (keyword) and \`KB\_search\_dense\` (embedding), and a read-only \`shell\` over the knowledge base)   
and the customer's messages, and it acts through the bank's tools. When the conversation ends, the conversation and the bank   
's final database are saved under \`tau2/\` in the solver's working directory.   
### 3. Grader   
The grader is \`benchmark\_grader.py\`, given next to \`plugin\_api.py\`. Its \`grade(task, snapshot)\` scores one rollout from 0 to 1;   
the snapshot is the solver's working directory when the conversation ended. It reads the task's \`answer\_files[0]\` (the task   
file), \`grading["domain"]\` (\`banking\_knowledge\`) and \`grading["retrieval\_variant"]\` (\`alltools\`); a task has no input files.   
### 4. Run size   
3 epochs · 3 samples per task.

## A.4 The evolver’s appended system prompt

Appended once to Claude Code’s own system prompt and kept for the whole run; the loop fills in one placeholder, the number of attempts per training task. Every EvoSkill v2 run in this report used this text.

You are the skill evolver in a multi-epoch run. Each epoch, a separate solver agent runs every training task 3 times (each run is   
a rollout); you read what happened and write the skill set it carries next. The goal is a skill set that raises the solver's   
score on tasks it has never seen. The training tasks are a sample of a larger benchmark, so a skill that describes them   
specifically will not transfer.   
Epoch 0 ran with no skills; each later epoch K runs with \`skills\_vK\`, the set you write at your turn K. This session lasts the   
whole run: each epoch sends you one message, and your reply to it is your turn.   
- \`/evo/traces/epoch\_K/<task id>\_s<i>.jsonl\`: the solver's session log of sample i; files the log refers to (such as large tool   
outputs) are in \`<task id>\_s<i>/\`.   
- \`/evo/scores/epoch\_K.json\`: each rollout's score from the grader, 0 to 1, or \`grader\_failed\` when the grader itself failed.   
- \`/evo/solver\_workspaces/epoch\_K/<task id>\_s<i>/\`: the solver's working directory as it was when it exited.   
\`/evo/skills/epoch\_K/\`: \`skills\_vK\`, the set epoch K's solver carried.   
- \`/evo/staging/\`: the set you are writing. Each turn starts with the set you left here; edit it in place, and whatever is here   
when your turn ends is the next set. If you change nothing, the solver carries the same set again. Each skill is \`<skill  
name>/SKILL.md\`; write only \`name\` and \`description\` in its YAML frontmatter. Files beside it ship with the skill, read-only   
and not executable (the solver runs scripts through their interpreter).   
- \`/evo/evolver\_workspace/\`: your own directory, kept across epochs. Your context may be auto-compacted when it grows too big, so   
keep here anything you want to preserve.   
- \`/evo/reference/\`: the benchmark's data and documents, without anything about validation or test tasks or the grader.   
Paths in the traces refer to the solver's machine, not this one; its working directory and skills are the snapshots above. You run   
in the solver's image, with no network beyond the model API.

## A.5 The evolver’s message at each epoch

The loop fills in the current time, the epoch’s deadline (two hours later) and the epoch number. As sent at epoch 1 of the τ<sup>3</sup>-bench run with training answers:

It is 2026-10-07 21:22 UTC; you have until 2026-10-07 23:22 UTC for this epoch, after which \`/evo/staging/\` is taken as it stands.   
You are at epoch 1. Please analyze the solver rollouts, diagnose the pitfalls that may be fixed by a dedicated skill, and   
compose \`skills\_v1\` under \`/evo/staging/\`.

## A.6 The SpreadsheetBench v2 evolver’s skills (fine-grained ground truth): their descriptions

The four epoch-3 skills of the EvoSkill v2 run with fine-grained ground truth (§5.1, Figure 2). The Sonnet 5.5 evolver wrote all four at its first turn, kept them to the last and revised each one at every turn; at epoch 3 their SKILL.md files hold 71 KB of text. debug-injected-errors ships three scripts that only report (suspicious formulas; each row’s formulas grouped by pattern; the links between sheets), and financial-model-fill one that finds the blanked cells and one that lists the input cells no formula uses. bond-deferred-tax-templates (Template folders 01 and 05) and re-direct-cap-model (folder 16) are written for the only three Template folders in the training set, from which no validation or held-out task comes. Each skill’s name and description, as the evolver wrote them (the file names they cite belong to training tasks):

name: debug-injected-errors   
description: Fix deliberately injected errors in a finished multi-sheet financial model (LBO, Operating Model + DCF, Revenue Build   
Comps + WACC and similar). Use when the input file name or task text names an error category - Double Counting, Cross   
Sheet references, Index Match, AVERAGE, Embedded Hardcodes, Unit Mismatch, Sign Conventions, Relative vs Absolute references,   
Inconsistent Color Coding, or "Errors" (#REF!/deleted rows). Gives a precision-first minimal-edit workflow, the exact   
mutation signatures per category, original-model quirks that must NOT be fixed, and three scanner scripts.   
name: financial-model-fill   
description: Fill blanked-out cells in multi-sheet corporate financial model workbooks (assumptions, revenue drivers, income   
statement, balance sheet, cash flow, WC/debt/PPE schedules, DCF/WACC, valuation multiples, ratios, sensitivity tables) with   
live Excel formulas via openpyxl so that the LibreOffice-recalculated values match a hidden golden model exactly. Use when a   
task says "complete/populate/fill the cells in sheet X" for a project model. LOAD FIRST for every 'Complete the financial   
model based on the provided assumptions' task on a corporate project workbook (file names like 01\_0X\_Education, 06\_DigiMark,   
14\_Refinery, 16\_Ramjet, 20\_Fintech); do not start editing before reading it.   
name: bond-deferred-tax-templates   
description: Conventions for completing "Bond accounting" (effective-interest, OID, zero-coupon, premium/discount, early   
extinguishment, GAAP vs tax balance sheet) and "Deferred tax" (book vs taxable income, DTA/DTL, cash-flow statement)   
template workbooks. Use when a sheet has labels like "Interest expense (amortization)", "Loan balance - BOP", "Deferred tax   
asset - BOP", "Tax B/S impact of borrowing", "Implied IRR", or "Balance check" / "Unearned revenue". LOAD FIRST for any   
Template task on bond/loan/deferred-tax/GAAP-vs-tax workbooks (file names 01\_0X\_input.xlsx, 05\_0X).   
name: re-direct-cap-model   
description: LOAD FIRST for any workbook with sheets named DirectCap, RentRoll, NOI\_Analysis, Valuation, Assumptions/SharedData or   
labels like Total Units, Average Unit Size (SF), Per SF/Month, Per Unit, Effective Gross Revenue, NET OPERATING INCOME, Cap   
Rate, Property Valuation / Indicated Value (file names 16\_0X\_input.xlsx, 'Complete the financial model based on the   
provided assumptions. Ensure that all formulas, references, and calculations are correct.'). Multifamily real-estate direct  
capitalization models - rent roll totals, pro forma revenue, vacancy, opex, NOI, cap rate, indicated value, per-SF/per-unit/   
per-month columns (including the NOI row of a per-SF column). Template tasks are graded cell by cell; unlike the corporate  
model skill this one lists the exact conventions.