# Scientist World Grow a Research World for AI-Scientists

Taiming Lu Sachin Konan Zhuang Liu Princeton University

ai-scientist.world

## we turn research papers into executable environments

![](images/58fa9202a5393e4101c03619b972ed4bd0502934f9e41a9181d467199137901f.jpg)  
improve a paper's method

![](images/827d78e8a01e66028fb72560ce67ac7ecc7d423f590a4802018ca198dba94793.jpg)  
invent a method from scratch

![](images/436d34ffb04b7043427605c5d4f8d36a4baf39cf8ac4dcbdbdc51a18f784315b.jpg)  
Figure 1. We introduce Scientist World. Each research paper becomes an executable environment, where an AI-Scientist’s Frontier-Normalized Improvement (FNI) is above zero when its method improves on the paper’s reproduced one. We test how well AI-Scientists improve existing research (left) and innovate new research from scratch (right). We show the best model of each lab plus the next best overall; full leaderboards are in Figure 12.

## Abstract

AI-Scientists arrived, but a world for them to grow in did not. Human scientists take theirs for granted — centuries of open problems and shared ideas, waiting in the scientific literature. To AI-Scientists, that world is readable but not runnable, so we build Scientist World for them: every research paper becomes an executable environment. Our autonomous pipeline extracts each paper’s open-ended research question and reproduces its proposed method as the baseline; an AI-Scientist entering the environment is evaluated against what the human authors achieved. We release the first 1,000 verified environments. With the world as an evaluation, we found that frontier AI-Scientists can already surpass the human researchers 74% of the time. As a record of how AI does science, it contains 6,400 hours of research traces, with every idea, experiment, failure, and success. As a training ground, we are able to raise (↑158%) a small model to match frontier models in research ideation. At last, AI-Scientists’ own world has arrived, grown from and growing with science itself.

## 1 Introduction

A new kind of scientist has arrived — for the first time, an intelligence advances the very thing it is made of.   
AI-Scientists are the first whose discoveries improve the discoverer (Boiko et al., 2023; Gottweis et al., 2025;   
Mitchener et al., 2025). They now invent new AI on their own (Yamada et al., 2025; Schmidgall et al., 2025;   
Lu et al., 2026). This could set of a chain reaction, and every gain makes the next one faster (Romera-Paredes et al., 2024; Novikov et al., 2025). This is the power of a scientist, the power of research.

What is research? It is not a single ability, and it is not a single task. Watch a scientist at work: they read until they find something no one has explained, guess at the reason, and build an experiment to test it. It fails. They eventually figure out what the failure is telling them, adjust, and run it again. It works. Next month they are doing the same on a diferent problem, in a diferent part of the field. What makes them a scientist is not any one of these moves; it is the judgment that runs through all of them, and carries from one problem to the next. That is why research cannot live in a single task. Nailing one problem is a trick; research is the ability to discover again and again, problem after problem. We believe the same for a scientist of AI.

A scientist like that keeps growing, and it needs a world big enough to grow in. Human researchers already live in such a world: the scientific literature. It provides two things. The first is problem: each paper defines a diferent research question no single answer can exhaust. The second is idea: each paper contributes to the field’s shared memory of approaches, findings, successes, and failures. Together, they drive research as target and foundation. This is the world human scientists have always relied on, and the one AI-Scientists need.

We introduce Scientist World : a living research universe built from the scientific literature itself. In it, each paper becomes an executable scientific environment. We designed a multi-agent pipeline for this conversion, with no human in the loop. There are four steps: (1) reproduce the paper’s method until it recovers the paper’s result; (2) design a research task by defining what counts as scientific progress without shortcuts or gaming; (3) package it so it runs on any machine; and (4) verify it with independent agents. An AI-Scientist is then scored by how much its own method improves over the human researchers’ method, which we call its Frontier-Normalized Improvement (FNI). Building from high-quality papers, we first release 1,000 environments, and it keeps growing with the literature (Figure 1).

Because the pipeline runs with no human in the loop, Scientist World is Scalable: it can grow fast to cover a broad range of fields and topics. Two more properties are inherited from science itself. Evolving: science never stands still, so a fixed set measures yesterday, what a scientist has already seen; Scientist World absorbs new literature every day, measuring research rather than recall. Execution-grounded: a fluent AI-Scientist can make any idea sound bold, but ideas that impress on paper might fail on execution (Kapoor et al., 2024; Si et al., 2025); because every environment is built and measured through real experiments, research is judged by execution, not rhetoric.

Environments for research exist, but none, to our knowledge, holds the three properties together. Most are not scalable: they recycle problems from past competitions (Huang et al., 2024; Chan et al., 2025; Nathani et al., 2025), confine to simulated worlds (Jansen et al., 2024; Chen et al., 2025b), or hand-pick tasks from a few papers (Wijk et al., 2025; Zhang et al., 2025b; Garikaparthi et al., 2026). Nearly all are not evolving: sets are fixed at release and age into training data (Sainz et al., 2023; Xu et al., 2024), and those that refresh cover only

![](images/12b51607509d6bdc25ddb6e865a6b0b6b195ec0f4230241443ff8fc01039ef91.jpg)  
Figure 2. Where Scientist World stands. Prior AI-Scientist environments sit on the trade-of surface we describe; Scientist World reaches the corner where scale, evolve, and execution meet.

narrower skills such as code (Jain et al., 2025; Zhang et al., 2025a). And others are not execution-grounded: they reproduce results the field already knows (Siegel et al., 2024; Starace et al., 2025; Kon et al., 2025) or judge proposed ideas as text (Guo et al., 2025; Si et al., 2025). Scientist World holds three at once (Figure 2).

We first use Scientist World to evaluate frontier AI-Scientists. Each scientist attempts an environment in two modes: improve, which starts from the paper’s idea with its reproduced method; and innovate, which starts from the problem alone. The two abilities prove distinct, since the best improver is not the best inventor. Every attempt leaves a trajectory, each decision paired with the measured outcome of a real experiment, and we release this corpus alongside the environments. Then we train with it: rewarded only by real outcomes, a small scientist learns to propose ideas that, implemented by a fixed frontier executor, reach the frontier cohort’s level on the median held-out environment. Taken together, Scientist World is three things at once:

• A benchmark. The standing measure of whether AI-Scientists can advance real research.

• A dataset. A growing corpus of research being done, with every attempt preserved in full loop.

• A gym. The same environments become training grounds, with real experimental results as rewards.

If research is the ability to discover again and again, it needs a world that never runs out. is that world, grown from science itself and open today. The chain reaction now has a place to happen.

![](images/ae6e24f8e7f10c908ebf1098db732d202344452a983839572d2213022d46192b.jpg)  
(a) we test how well they do research  
(b) we release their research traces  
(c) we raise a small scientist  
Figure 3. Scientist World is three things at once. One set of environments, grown from real papers, serves all three.

## 2 From Papers to Environments

How well can AI-Scientists do research? Any answer must come from a research environment: a real problem to work on, and an objective measure of progress. A research paper is almost such an environment. It states a problem, proposes a method, and reports what happened. Yet it is only a record of science done, not a place to do science. We turns the record back into the place. It (1) rebuilds the paper’s idea and experiment, (2) defines what constitutes an improvement, and (3) judges every scientist’s new solution.

We design builder agents to perform this conversion autonomously. They reproduce a paper’s experiment, extract the problem to solve, and design both into a self-contained environment: the problem becomes a task any new scientist can take up, and the reproduced result becomes the bar their solutions are scored against.

## 2.1 The Builders

We convert each paper through a sequence of agent iterations on one shared workspace (Figure 4). Four roles divide the work, from reproducing the paper to verifying the finished environment. A single paper can take hundreds of unattended turns without a human in the loop. To hold the sequence together, a ladder of milestones is used to track progress. Each rung marks how much of the environment stands, from a first runnable experiment to an environment ready for new scientists. A builder agent ends its iteration by reporting the rung it believes it reached; the next builder begins by checking and verifying that report. At the end of the builder iterations, the final environment is accepted or rejected based on the milestone reported.

Workflow. A paper moves through builder roles in iterations. Each agent inherits the workspace, audits the claimed milestone, and demotes whatever it cannot support. Every iteration ends at a gate: a milestone below the requirement reverts the work to an earlier iteration, or exits as failure. To be accepted, the experiment must run to support the paper’s central claim. We rotate diferent builder models across iterations to avoid any blind spots. What survives is a repository that any scientist can use without knowing how it was made.

![](images/05b12d1e24f46011414ef4c5f383c5056975e4c80c1340bc8fb7e6f7752fb0c5.jpg)  
Figure 4. Builders turn a paper into an environment. Construction runs like an assembly line: a published paper enters at one end, and a reusable research environment leaves the other. Specialized builders take turns on a single shared workspace, each refining what the previous one built and ofloading heavy computation to a shared cluster. A quality gate guards the line, sending unfinished work back for another pass, so only what clears verification is accepted.

(1) Reproducer. The reproducer re-establishes the paper’s main result. It implements both the method and the paper’s original evaluation, from the authors’ code if released or from scratch otherwise. Its experiments scale up from quick checks toward the full run, until the result appears. What it leaves behind we call the source method and the source result: the reproduced method and the numbers it achieved.

(2) Designer. The designer turns the reproduction into a task that others can attempt. It writes the problem statement that a new scientist will read, and it fixes the rules of the task: which metric to improve, which settings must stay fixed, and which kinds of solution are allowed. It sets the source result as the bar, so a new solution is scored by how much it improves on it. It prepares two evaluation sets: a visible one that the scientist may test against freely, and a held-out one to detect shortcuts or overfitting.

(3) Packager. The packager makes the designed environment run anywhere, not only on machines that built it. It replaces every local file with a script that downloads it, and declares every dependency the experiments need. The check is simple: a fresh clone of the workspace must be able to rebuild and rerun on its own.

(4) Verifier. The verifier audits the packaged environment before it is accepted. It rereads the workspace with a skeptic’s eye, checking the code is valid, the scores match the logs, and the task follows the design rules. We keep this role separate on purpose, where the builder that grades the work never built it.

Infrastructure. All experiments run on a shared pool of compute machines (Figure 5), described in §A with details. A builder never runs jobs on its own hardware: it submits a job, waits, and is woken with the logs when the job ends. We cap each paper’s GPU time a builder can use with a budget enforced outside the builder; when it runs out, the builder wraps up with what it has. With thousands of papers queuing, no single one deserves unbounded compute.

![](images/108ac17798c5f503e41d2b5a5d0caaca1ebea1bc2e6dd366453b217c70d2d196.jpg)  
Figure 5. Using compute. A builder submits a job to the shared pool, sleeps, and is woken with the logs; its GPU budget is enforced outside the loop.

![](images/febedeafb749c0c8f1e68d1c4c2e82a3e0249ed769324156aefa94de27627fa1.jpg)  
Figure 6. Anatomy of an environment. A produced environment is a single self-contained workspace, laid out so the problem stays separate from the solution. The briefing, data, and scoring fix what to solve and how it is judged without referring to any one method, which keeps the task open to future approaches and stops a score from depending on the implementation. The method is the paper’s source method reproduced, and the scores it reaches are the reference we compare against, so a new method’s final score is the improvement it makes over them.

## 2.2 The Environment

An accepted environment is a paper refactored into an executable research setting. It does not ask a future scientist to reproduce the paper again; it gives that scientist a prepared place to work on the paper’s problem. The repository contains four parts, in the order a scientist encounters them: (1) a task briefing that states the problem, target, allowed direction, and constraints; (2) data and evaluation files that define what a method must produce and how it is scored; (3) the reproduced method and baselines, which provide starting solutions (the source method can optionally be withheld during evaluation); and (4) scripts, score files, logs, job outputs, and commits kept as the construction record. Figure 6 walks through one produced environment in full.

What makes it an environment. An environment is more than reproduction code. It keeps the paper’s research situation open for new methods. We design each environment around three principles. First, builders may shrink the experiment scale to reduce compute cost, without changing the scientific question. The smaller task should still ask whether the paper’s idea improves the same kind of outcome under the same kind of evaluation. Second, the task must remain non-trivial. The source method should still leave a meaningful gap over simple baselines, so future methods have something real to improve. Third, we treat the reproduced method as one solution, not as the task itself. The data interface and evaluator are written independently of the source method, so a future method can replace it by producing the same kind of artifact.

What makes it a fair task. We want a future method to win by making progress on the same question, not by changing the question itself. So we write research rules across four files. The problem file states the research question. The target file names what should improve, such as loss or accuracy. The direction file says what kinds of method changes still answer that question. The constraints file lists changes we disallow because they would make the result about a diferent problem. The constraints also rule out direct scale gains, for example by holding model size, data size, or compute budget fixed when those factors are not the research question.

How a scientist’s attempt is graded. A scientist’s proposed method is graded on how far it improves over the paper’s source method, on the quantity the claim is about, such as accuracy or loss. This can be viewed as a comparison with human researchers. We score the method over multiple experiments with diferent metrics used in the original paper. Besides a visible set for development, we reserve a held-out slice, so a method must improve the task rather than exploit one visible number through shortcuts or overfitting (§E.3).

Frontier-Normalized Improvement (FNI). We introduce FNI, scored per paper as $\begin{array} { r } { \mathrm { F N I } = 1 0 0 \sum _ { c } w _ { c } \delta _ { c } / ( \sigma _ { c } b ) } \end{array}$ where $\delta _ { c }$ is the signed gain over the source method on metric $c , w _ { c }$ its weight, $\sigma _ { c }$ a frozen frontier cohort’s spread on it, and � sets the cohort’s mean |FNI| to 100; 0 matches the paper, and negative is worse (§E.3).

![](images/683b5528e2e13854ead8b36da122683fefd3ce576f79c5d433198b8f195cf0d9.jpg)  
(a) screen out infeasible papers

![](images/8ff31ce19cb66fc646929c0a4c319dc8f863c0253c5de230a96a9a7ee3367e79.jpg)  
(b) rank and keep the top 1,000

![](images/8c5ea8e01e1e7a6eef52ea5b6addb877dcadf05741b739b8aaa77a83f79d05d0.jpg)  
(c) selected scores higher  
Figure 7. Selecting 1,000 papers from 124,816 indexed. In a first pass (a), we drop papers that are low quality or are mainly a type of theory, a benchmark, a dataset, or a single finding, which leaves 21,354. In (b), we look at these more closely and favor work that is both high quality and compute-friendly, keeping the best 1,000. In (c), we check the result: the papers we picked (blue) are clearly stronger and more reproducible than the pool we started from (grey).

## 3 Scientist World

is the artifact built from the environment-construction system: the selected paper set, the generated repositories, and the records produced while building them. This section reports how we choose a valuable, buildable 1,000-paper source set and how the batch conversion ran.

## 3.1 Building at Scale

Selecting papers. Our selection pipeline keeps papers that are worth building under budget (Figure 7). We start from arXiv’s core AI categories over fourteen months to obtain 124,816 unique papers. We (1) run two fast screens with MiniMax M2.7 (MiniMax, 2026). The quality screen keeps papers in roughly the top 25% quality band. The type screen keeps empirical-method papers, whose main contribution is quantitative optimization. Together they leave 21,354 papers. Then (2) we run more detailed judgments for each paper on two quantities: research quality and reproduction dificulty (we prefer papers with low compute costs). From the remaining pool, with a geometric-mean utility over quality and feasibility, we sample the category-balanced top 1,000 papers.

Automation. No human reviews any paper at any stage in our pipeline. The source arXiv categories add roughly 9,000 unique papers per month, of which about 1,500 survive screening and remain feasible. The current 1,000 are the first release; future batches extend the window through the same pipeline.

The build. We build every paper with the same nine-iteration workflow: three reproducers, three designers, one packager, and two verifier passes (we repeat a role for multiple iterations for additional quality checks), with the model family rotating across GPT, Claude, MiniMax, and GLM. A paper may draw up to 70 GPU-hours, yet the median spends under half of one (Table 1).

<table><tr><td>per paper</td><td>median</td><td>total</td></tr><tr><td>compute jobs</td><td>76</td><td>239,506</td></tr><tr><td>GPU-hours</td><td>0.4</td><td>2,286</td></tr><tr><td>CPU-hours</td><td>6</td><td>28,338</td></tr><tr><td>agent turns</td><td>730</td><td>947,056</td></tr><tr><td>input tokens</td><td>28M</td><td>51B</td></tr><tr><td>output tokens</td><td>163k</td><td>231M</td></tr></table>

Table 1. Per-paper and total build cost. Median and the total cost over 1,000 papers.

![](images/36442112e939ea65be62a4ca3c116533b5bd98d2a641a04e1da951d2b8923df2.jpg)  
Figure 8. The environment composition. The produced environments can be categorized into eight areas.

<table><tr><td>environment</td><td>average</td></tr><tr><td># experiments</td><td>3.3</td></tr><tr><td># metrics</td><td>(16 visible + 16 hidden) 32</td></tr><tr><td># datasets</td><td>1.6</td></tr><tr><td># files</td><td>(3.8k lines of code) 59</td></tr></table>

Table 2. A produced environment on average. The mean size and makeup of one environment.

<table><tr><td>across 1,000 envs</td><td>rate</td></tr><tr><td>reproduced method score within margin</td><td>99%</td></tr><tr><td>beats paper&#x27;s baselines</td><td>82%</td></tr><tr><td>reverted iteration by a later cross-check</td><td>(573 total, 1.6 avg) 37%</td></tr><tr><td>approved by independent verify</td><td>100%</td></tr></table>

Table 3. Quality and validity. How it reproduces paper method, beats baselines, and survives audit.

## 3.2 The Built World

Composition. We first release 1,000 environments spread across eight areas (Figure 8). A scientist therefore meets problems of many kinds, not one narrow family. A typical environment is substantial, closer to a small research project than a single script (Table 2). Because most carry several experiments rather than a single headline result, a later method has multiple quantities to improve rather than one number to chase.

Quality. Every environment is built without a human in the loop, so we do not take any single builder’s word for its quality. Four checks stand behind an accepted environment, one per row of Table 3:

• Reproduction fidelity. The reproduced source method’s numbers are checked against the paper’s reported results within a small margin, across every experiment and metric used by the environment; the few exceptions are accepted only with documented justification.

• Baseline separation. Where a paper provides baselines, we reimplement them and require the reproduced source method to outperform them, preserving a real gap for future methods to close; the rate in Table 3 counts all 1,000 environments, including those without a reimplementable baseline.

• Iterative cross-checking. Each builder audits the previous milestone claim and demotes whatever the workspace cannot support. A demotion rolls the workflow back to the previous iteration; construction then resumes from that checkpoint until it passes the later checks without further reversion.

• Independent verification. The final environment must pass a verifier that did not participate in construction. Every released environment satisfies this final check.

Example. Figure 9 shows one environment end to end. The paper’s question survives intact, whether a transformer can train without normalization, while its experiment shrinks to ImageNet-50 so attempts fit a bounded budget. The constraint is what keeps the task honest: the same model scale, data, and pretraining as the source method, so any gain must come from the method itself. Half of every evaluation stays hidden, so the score reflects what generalizes rather than what was tuned against the visible half.

![](images/41062a559c1de6a746c87acf2a9fd76b2134aeaa3942a0b12791b013ae70a4ef.jpg)  
Figure 9. An example environment. This environment turns a Transformer architecture paper into a constrained improvement task: raise ImageNet accuracy without changing scale, pretraining, or data. The scripts make the task runnable, the source and baseline implementations anchor the comparison, and the scoring matrix defines progress as weighted improvement over the reproduced source method.

Openness. Research rarely hands the scientist an approach. Deciding what to build is much of the work, and a world that always prescribes it would train executors rather than scientists. We therefore judge each task’s materials and place it on a 0 to 100 scale, where 0 means the approach is fully prescribed and 100 means the task fixes only what counts as success and leaves the way there entirely to the scientist. We find that every area covers nearly the whole range (Figure 10): the level of freedom of an environment depends on the task itself, not on where in the world it sits.

![](images/38d079e29a9c2a2d8b9c36ffe537a43482d0a6e4843adbdec74dd6c0bc3aef1b.jpg)  
Figure 11. What the environments ask for. Each task names, in its own words, what should get better.

![](images/9b790d7790b60c0499e9f64416fd75efb5948065f4b18bbdc3240c7de284f19d.jpg)  
Figure 10. Environments’ openness. From fully prescribed (0) to fully open (100).

Goals. Each environment sets its own goal. Most tasks seek correctness (for example, pushing accuracy), and the rest split mostly between eficiency and robustness, with many naming more than one (Figure 11). No goal belongs to one area alone: correctness leads everywhere, eficiency appears from architecture to agents, and robustness reaches well beyond safety. The full chain from areas through goals to the work they take is shown in Figure 34.

![](images/f1bcc448c719b4be9ff919fe7ba666c2248670264970c46712f7082e94db0f02.jpg)

![](images/e055a1997b8a38da9c8a82212baac9560855554d20129ccc8e38ca855ec6dd0e.jpg)  
Figure 12. Leaderboards. Mean FNI by model in improve mode (top) and innovate mode (bottom), averaged over visible and held-out splits. Zero matches the reproduced source method; higher is better. Improve mode gives models the source method, while innovate mode requires them to design from scratch. We use xhigh efort for all models.

## 4 Scientists in the World

## 4.1 Benchmark

We now use the environments to evaluate AI-Scientists. Running every model on all 1,000 environments is too costly to repeat, so we evaluate on a fixed cohort of ten lower-compute environments. Each model attempts every environment under the same compute budget, in two modes:

• Improve mode. The model receives the reproduced source method and tries to improve its result.

• Innovate mode. The model receives no source code and must design a method from scratch. We ensure there is no leakage of the paper’s original solution.

Each attempt is evaluated on both the visible and held-out splits, with the final FNI score averaged across the two. Zero FNI matches the reproduced source method, while a positive FNI improves on it. Since diferent environments have diferent scoring scales, we normalize the scores to aggregate (§E.3).

In the result, the two modes show diferent strengths across models (Figure 12): Fable 5 is the strongest at improving existing methods, while Fable 5.1 and Fable 5 take the top two places at designing new methods from scratch. Among open models, GLM 5.2 comes closest to the frontier, placing third at innovating. Notably, the AI-Scientists beat the paper’s reproduced method in 84% of improve and 74% of innovate attempts (§F.1).

![](images/dbef05b6c7c611f5b2ec1cc3cc5e1448b052c92f34aa957a5bc3caa3a0e3f486.jpg)  
Figure 13. The two abilities are loosely tied. Each model’s rank at improving against its rank at innovating.

The leaderboards show which models score highest in FNI score, but not what kind of ability it reflects. We first compare the two modes directly. If improving an existing method and inventing a method from scratch measured the same skill, models would keep similar ranks across the two leaderboards. Instead, the relationship is loose (Figure 13). Some models are much stronger when building on source code, while others rank higher when forced to design from scratch. We therefore treat improve and innovate as related but distinct research abilities.

Is performance just a scaling efect? Better results could come from scaling the attempt in two ways: using more external compute, or doing more modelside work through longer trajectories and more trialand-error. We do not find either pattern. Compute hours and agent turns explain almost none of the variance in scores $( R ^ { 2 } \leq 0 . 0 2$ , Figure 14a; turns are our provider-agnostic proxy for model-side efort, since token counts may not be available). The leading models therefore do not win by simply spending more; they make better use of the same setting.

Then what drives it? The quality of the research itself. An LLM judge reads every attempt, blind to its score, and rates it on five properties: creativity (was the idea genuinely new, and did it pay of), engineering (was the implementation substantial, building on the existing code), rigor (was the reported gain real and honestly measured), significance (how far it moved the metric), and depth (how thoroughly the problem was explored). These ratings track the models’ actual scores almost exactly $( \rho = 0 . 8 9$ , Figure 14b): the models that win are the ones doing better research, not the ones trying hardest.

![](images/b16dd3c1f98a1e90ab23eb206e56a31ef8ee1fe392bf5ad0214e7085fb22d4fa.jpg)  
Figure 14. Efort versus outcome, and research quality. (a) each run’s score against the compute hours (top) and agent turns (bottom) behind it, pooling both modes; efort explains almost none of the variance in score $( R ^ { 2 } = 0 . 0 0$ 0.02). (b) research quality by dimension, judged apart from the score. Newest model per family, colored by provider.

Research is contingent. Research is inherently random. A strong attempt does not always produce a high score: a creative idea can be implemented well and still fail empirically, and a great idea might not suit the current setting, while a simpler idea may happen to work. Aggregating across many environments lets us measure progress without pretending that research has no outcome variance.

## 4.2 Dataset

We preserve every scientist’s research in full, so we collect a set of research traces, released per environment, model, and run mode. We record the scientist’s research idea and their revisions, the method and the full turnby-turn trajectory, and the per-experiment logs and scores (Table 4). We store each attempt as a version-controlled repository, so each preserves the complete development history rather than the final state alone.

In total, the corpus records roughly 6,400 hours of research being done (§H.1). With multiple scientists’ attempts, one environment accumulates multiple independent solutions to a fixed problem, each labeled with its measured improvement over the reproduced source result.

<table><tr><td>record</td><td>average</td></tr><tr><td>idea revisions</td><td>3.7</td></tr><tr><td>method revisions</td><td>6</td></tr><tr><td>experiments run</td><td>25</td></tr><tr><td>commits</td><td>35</td></tr><tr><td>hours of work</td><td>5.4</td></tr></table>

Table 4. Dataset statistics. Each experiment is stored with its log; hours of work is the scientist’s end to-end running wall-clock time.

## 4.3 Gym

Finally, we raise a scientist of our own. We train Qwen3-8B (Qwen Team, 2025), a scientist small enough that whatever it learns must come from the world, with Tinker (Thinking Machines Lab, 2025), in two stages:

• Supervised fine-tuning. 1,000 examples drawn from the trajectories, each pairing a task specification with an attempt’s reasoning and final idea, cleaned so that no fabricated result survives. (details in §H)

• GRPO. We do of-policy group relative policy optimization (GRPO) (Shao et al., 2024) on 1,500 alreadyscored ideas, rewarded by measured outcomes. Advantages are relative to the other ideas for the same task, and updates are clipped to a trust region around the SFT policy. The scientist trains only with execution.

A scientist this small cannot yet carry the whole loop: implementing and debugging a method over a long horizon is frontier work in itself. We therefore separate ideation from implementation (Si et al., 2026). The trained scientist proposes the idea, and a single fixed frontier implementer (GPT 5.6 Sol) executes every idea under identical conditions on ten held-out papers (§H.4), so only the idea’s origin varies. We prompt the implementer to follow exactly the ideator’s idea.

The gain is monotone. The median implemented idea scores 41 FNI from the base scientist, 84 after SFT, and 106 after GRPO (Figure 15); the mean rises from

![](images/3589b8eb9280d046ab4590da58810501d83f81572c0d1d2a335c2cabf614b13d.jpg)  
Figure 15. Raised in the world. Median held-out FNI of each stage’s ideas, implemented identically; after GRPO it reaches the frontier cohort’s anchor.

51 to 90. Ideas from the trained scientist, faithfully implemented, reach the anchor of the frontier cohort. What it demonstrates is the mechanism: the world supplied the data, the reward, and the held-out exam, and a scientist raised on its trajectories proposes measurably better research. The training loop works end to end.

## 5 Background

Autonomous scientific discovery and AI-Scientists. Automating discovery is a long-standing goal. Early systems recovered natural laws and governing equations directly from data (Schmidt and Lipson, 2009; Cranmer et al., 2020; Udrescu and Tegmark, 2020), and robot scientists closed the loop from hypothesis to physical experiment (King et al., 2004, 2009; Williams et al., 2015). Within machine learning, automated search took over bounded parts of the pipeline: model and hyperparameter selection (Thornton et al., 2013; Feurer et al., 2015), neural architecture design (Zoph and Le, 2017; Real et al., 2019), and even the discovery of faster fundamental algorithms (Fawzi et al., 2022; Mankowitz et al., 2023). Program search guided by language models extended this to open mathematical and algorithmic problems (Romera-Paredes et al., 2024; Novikov et al., 2025). Language models then widened the scope toward the full workflow: tool-augmented systems run their own chemical experiments (Boiko et al., 2023; Bran et al., 2024), and agentic pipelines generate ideas, write code, and draft papers end to end (Lu et al., 2024; Yamada et al., 2025). A newer wave organizes many agents into collaborative or self-directed discovery loops (Schmidgall et al., 2025; Schmidgall and Moor, 2025; Ghareeb et al., 2026), and general-purpose AI-Scientists now attempt discovery across whole fields (Gottweis et al., 2025; Mitchener et al., 2025). This line builds better scientists; is complementary, turning finished papers into environments where such scientists can be handed the same problem and scored on it.

Measuring AI-Scientists. Evaluating agents in executable settings began with general benchmarks for tool use and repository-level code editing (Jimenez et al., 2024; Liu et al., 2024; Mialon et al., 2024), then moved to machine-learning experimentation and engineering (Huang et al., 2024; Chan et al., 2025; Wijk et al., 2025; Nathani et al., 2025). A parallel line probes scientific reasoning and data-driven discovery (Jansen et al., 2024; Majumder et al., 2024; Chen et al., 2025b), extending to hypothesis formation and frontier domain research (Liu et al., 2025; Miao et al., 2026). A closely related group scores whether an agent can reproduce a paper’s results from its description (Siegel et al., 2024; Starace et al., 2025), reimplement its method and code (Xiang et al., 2025; Hua et al., 2025), or take on research framed as an open-ended challenge (Zhang et al., 2025b; Chen et al., 2025a). These suites are almost always fixed and hand-curated, and critiques of agent evaluation warn that they can confound cost, leak through weak holdouts, and reward shortcuts (Kapoor et al., 2024). Concurrent work also turns real papers into research tasks but curates them by hand, which does not scale (Garikaparthi et al., 2026). instead treats construction itself as the object: it selects papers at scale and refactors each into a scored, reproduce-then-improve environment, so a method is judged by its gain over a faithfully reproduced result rather than its distance from a reported number.

## 6 Conclusion

Every published paper with code is a finished piece of research: a problem, a method, and a measured result. turns that record back into a place where research happens: 1,000 papers become runnable, scored environments, and the collection grows with the literature rather than with annotation efort. The same environments measure today’s AI-Scientists and train the next ones; each is a scaled-down paper and each score one contingent trajectory, but the loop they are built for, a scientist that proposes, executes, and updates inside the world, is just beginning. As the models and the literature both move, the world keeps pace, and the problems it poses stay as open as the research it was built from.

## Acknowledgments

We thank Muhan Gao, Linrong Cai, Xinshuang Liu, and Haozhe Chen for helpful discussions and feedback. The authors are pleased to acknowledge that the work reported on in this paper was substantially performed using the Princeton Research Computing resources at Princeton University which is a consortium of groups led by the Princeton Institute for Computational Science and Engineering (PICSciE) and Research Computing.

## References

Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 2023.

Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 2024.

Jun Shern Chan, Neil Chowdhury, Oliver Jafe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander Mądry. MLE-bench: Evaluating machine learning agents on machine learning engineering. In ICLR, 2025.

Hui Chen, Miao Xiong, Yujie Lu, Wei Han, Ailin Deng, Yufei He, et al. MLR-Bench: Evaluating AI agents on open-ended machine learning research. In NeurIPS, 2025a.

Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, et al. ScienceAgentBench: Toward rigorous assessment of language agents for data-driven scientific discovery. In ICLR, 2025b.

Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, and Shirley Ho. Discovering symbolic models from deep learning with inductive biases. In NeurIPS, 2020.

Alhussein Fawzi, Matej Balog, Aja Huang, Thomas Hubert, Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Francisco J. R. Ruiz, Julian Schrittwieser, Grzegorz Swirszcz, David Silver, Demis Hassabis, and Pushmeet Kohli. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature, 2022.

Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum, and Frank Hutter. Eficient and robust automated machine learning. In NeurIPS, 2015.

Aniketh Garikaparthi, Manasi Patwardhan, and Arman Cohan. ResearchGym: Evaluating language model agents on real-world AI research. arXiv preprint arXiv:2602.15112, 2026.

Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz, Jon M. Laurent, et al. Robin: A multi-agent system for automating scientific discovery. Nature, 2026.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an AI co-scientist. arXiv preprint arXiv:2502.18864, 2025.

Sikun Guo, Amir Hassan Shariatmadari, Guangzhi Xiong, Albert Huang, Eric Xie, Stefan Bekiranov, and Aidong Zhang. IdeaBench: Benchmarking large language models for research idea generation. In KDD, 2025.

Tianyu Hua, Harper Hua, Violet Xiang, Benjamin Klieger, Sang T. Truong, Weixin Liang, Fan-Yun Sun, and Nick Haber. ResearchCodeBench: Benchmarking LLMs on implementing novel machine learning research code. arXiv preprint arXiv:2506.02314, 2025.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating language agents on machine learning experimentation. In ICML, 2024.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In ICLR, 2025.

Peter Jansen, Marc-Alexandre Côté, Tushar Khot, Erin Bransom, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Oyvind Tafjord, and Peter Clark. DiscoveryWorld: A virtual environment for developing and evaluating automated scientific discovery agents. In NeurIPS, 2024.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In ICLR, 2024.

Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter. arXiv preprint arXiv:2407.01502, 2024.

Ross D. King, Kenneth E. Whelan, Ffion M. Jones, Philip G. K. Reiser, Christopher H. Bryant, Stephen H. Muggleton, Douglas B. Kell, and Stephen G. Oliver. Functional genomic hypothesis generation and experimentation by a robot scientist. Nature, 2004.

Ross D. King, Jem Rowland, Stephen G. Oliver, Michael Young, Wayne Aubrey, Emma Byrne, Maria Liakata, Magdalena Markham, Pinar Pir, Larisa N. Soldatova, Andrew Sparkes, Kenneth E. Whelan, and Amanda Clare. The automation of science. Science, 2009.

Patrick Tser Jern Kon, Jiachen Liu, Xinyi Zhu, Qiuyi Ding, Jingjia Peng, Jiarong Xing, Yibo Huang, Yiming Qiu, Jayanth Srinivasa, Myungjin Lee, et al. EXP-Bench: Can AI conduct AI research experiments? arXiv preprint arXiv:2505.24785, 2025.

Gregory M. Kurtzer, Vanessa Sochat, and Michael W. Bauer. Singularity: Scientific containers for mobility of compute. PLOS ONE, 2017.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. AgentBench: Evaluating LLMs as agents. In ICLR, 2024.

Yujie Liu, Zonglin Yang, Tong Xie, Jinjie Ni, Ben Gao, Yuqiang Li, et al. ResearchBench: Benchmarking LLMs in scientific discovery via inspiration-based task decomposition. arXiv preprint arXiv:2503.21248, 2025.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jef Clune. Towards end-to-end automation of AI research. Nature, 2026.

Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards data-driven discovery with large language models. arXiv preprint arXiv:2407.01725, 2024.

Daniel J. Mankowitz, Andrea Michi, Anton Zhernov, Marco Gelmi, Marco Selvi, Cosmin Paduraru, Edouard Leurent, Shariq Iqbal, Jean-Baptiste Lespiau, Alex Ahern, et al. Faster sorting algorithms discovered using deep reinforcement learning. Nature, 2023.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: A benchmark for general AI assistants. In ICLR, 2024.

Tingjia Miao, Wenkai Jin, Muhua Zhang, Jinxin Tan, Yuelin Hu, Tu Guo, et al. PRL-Bench: A comprehensive benchmark evaluating LLMs’ capabilities in frontier physics research. arXiv preprint arXiv:2604.15411, 2026.

MiniMax. The MiniMax-M2 series: Mini activations unleashing max real-world intelligence. arXiv preprint arXiv:2605.26494, 2026.

Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, Eric C. Landsness, Siddharth Narayanan, Jon M. Laurent, Michael Skarlinski, Samuel G. Rodriques, Michaela M. Hinks, Andrew D. White, et al. Kosmos: An AI scientist for autonomous discovery. arXiv preprint arXiv:2511.02824, 2025.

Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, and Ion Stoica. Ray: A distributed framework for emerging AI applications. In OSDI, 2018.

Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, et al. MLGym: A new framework and benchmark for advancing AI research agents. arXiv preprint arXiv:2502.14499, 2025.

Alexander Novikov, Ngan Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Regularized evolution for image classifier architecture search. In AAAI, 2019.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nature, 2024.

Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark. In Findings of EMNLP, 2023.

Samuel Schmidgall and Michael Moor. AgentRxiv: Towards collaborative autonomous research. arXiv preprint arXiv:2503.18102, 2025.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using LLM agents as research assistants. arXiv preprint arXiv:2501.04227, 2025.

Michael Schmidt and Hod Lipson. Distilling free-form natural laws from experimental data. Science, 2009.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. The ideation-execution gap: Execution outcomes of llm-generated versus human research ideas. arXiv preprint arXiv:2506.20803, 2025.

Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, and Tatsunori Hashimoto. Towards execution-grounded automated AI research. arXiv preprint arXiv:2601.14525, 2026.

Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, and Arvind Narayanan. CORE-Bench: Fostering the credibility of published research through a computational reproducibility agent benchmark. arXiv preprint arXiv:2409.11363, 2024.

Giulio Starace, Oliver Jafe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s ability to replicate AI research. arXiv preprint arXiv:2504.01848, 2025.

Thinking Machines Lab. Tinker: A training API for fine-tuning language models. https://thinkingmachin es.ai/tinker, 2025.

Chris Thornton, Frank Hutter, Holger H. Hoos, and Kevin Leyton-Brown. Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms. In KDD, 2013.

Silviu-Marian Udrescu and Max Tegmark. AI Feynman: A physics-inspired method for symbolic regression. Science Advances, 2020.

Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.

Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, et al. RE-Bench: Evaluating frontier AI r&d capabilities of language model agents against human experts. In ICML, 2025.

Kevin Williams, Elizabeth Bilsland, Andrew Sparkes, Wayne Aubrey, Michael Young, Larisa N. Soldatova, Kurt De Grave, Jan Ramon, Michaela de Clare, Worachart Sirawaraporn, Stephen G. Oliver, and Ross D. King. Cheaper faster drug development validated by the repositioning of drugs against neglected tropical diseases. Journal of the Royal Society Interface, 2015.

Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, and Yulan He. SciReplicate-Bench: Benchmarking LLMs in agent-driven algorithmic reproduction from research papers. arXiv preprint arXiv:2504.00255, 2025.

Cheng Xu, Shuhao Guan, Derek Greene, and M-Tahar Kechadi. Benchmark data contamination of large language models: A survey. arXiv preprint arXiv:2406.04244, 2024.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jef Clune, and David Ha. The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

Linghao Zhang, Shilin He, Chaoyun Zhang, Yu Kang, Bowen Li, Chengxing Xie, Junhao Wang, Maoquan Wang, Yufan Huang, Shengyu Fu, et al. SWE-bench goes live! arXiv preprint arXiv:2505.23419, 2025a.

Yunxiang Zhang, Muhammad Khalifa, Shitanshu Bhushan, Grant D. Murphy, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, and Lu Wang. MLRC-Bench: Can language agents solve machine learning research challenges? arXiv preprint arXiv:2504.09702, 2025b.

Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In ICLR, 2017.

## Appendix

This appendix details the builder system that constructs environments (Appendices A–C), how papers are selected (Appendix D), and evaluates scientist agents on them (Appendices E–G), the dataset and training details (Appendix H), followed by a worked example and the agent prompts (Appendices I–J).

## Table of Contents

A System Design . . 19   
A.1 Hive: Interactive Compute Pool . . . . . . . . . . . . . . . .   
A.2 Nexus: Experiment Boundary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21   
A.3 Swarm: Builder Coordination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   
A.3.1 Milestones . . . . . . . . . .   
. . . . . 23   
A.3.3 Workflow . . . . . . 24   
A.4 Worker: Agent Interface . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25   
A.5 How the Components Connect . . . . . . . . . . . . . . . . . . . . . . . . . 26   
B Builder Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27   
B.1 What the Builders Do . . . . . 27   
. . . . 28   
C Agent Workspace . . . . . . . . . . . . . 31   
C.1 Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31   
C.2 Problem and Scoring . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31   
C.4 Records and Validation . . . . 33   
D Selecting Papers . 35   
E Evaluation Protocol 38   
. . . . 39   
E.2 Running a Trial . . . . . . . . . . . . 40   
E.3 Scoring . . 41   
F Evaluation Analysis 44   
F.1 How Often Scientists Beat the Paper . . . . . . . . . . 44   
F.2 Do the Gains Generalize . 44   
F.3 Efort and Outcomes . 44   
G Experimental Setup . . . . . . . . . . . . . 45   
G.1 Data 46   
G.2 Compute 46   
G.3 Agent Scafolds . . . . . . . 46   
G.4 Models . 46   
G.5 Evaluation Trials . . . . 46   
G.6 Provider Failures . 47   
G.7 Cohort and Normalization 47   
G.8 Reproducibility 47   
H Dataset and Training Details 48   
H.1 Trace Corpus 48   
H.2 Training Data . . 48   
H.3 Training Procedure . . . 48   
H.4 Held-Out Evaluation . . . . 49   
Example Produced Environment . 50   
J Agent Prompts 53   
J.1 Prompts for Paper Grading and Selection 53   
J.2 Prompts for Swarm . 54   
J.3 Prompts for Evaluation . . . . . . . . . 63

## A System Design

relies on agents that repeatedly alternate between reasoning and computation. This agent stack is the substrate that makes this loop practical at scale. It separates four concerns that are otherwise entangled: running compute jobs, bounding each experiment’s resource use, coordinating many papers, and normalizing diferent agent interfaces. We separate the system into four parts: Hive provides the compute pool; Nexus makes compute accountable to a particular attempt; Swarm coordinates many runs; and Worker presents each agent scafold through the same interaction contract. Figure 16 summarizes these layers.

![](images/00c2ad8248a82eec4b0da306698ff1126a98712a8e39ba31b46985927d29ccfe.jpg)  
Figure 16. System design overview. Hive supplies execution capacity for CPU and GPU jobs; Nexus accounts for the budget and resource boundary of each attempt; Swarm maintains one paper-level trajectory across roles, restarts, and handofs; and Workers connect diferent agent scafolds through the same system interface. Together, these layers let many agents iterate on open-ended papers while preserving the evidence, budget state, and handof context.

## A.1 Hive: Interactive Compute Pool

Agent-driven reproduction needs a compute layer designed for interactive iteration. An agent may run a small debugging command, inspect the result, edit the code, and then launch a longer training or evaluation job. If every such step pays the full overhead of acquiring resources and preparing execution, the research loop becomes dominated by infrastructure latency rather than by scientific reasoning. Hive keeps compute capacity available, monitored, and controlled, so these iterations remain close to an interactive workflow.

Hive exposes compute as an experimental action: given a requested experiment, declared resources, and a workspace, it reports what happened. The result is not only a terminal success or failure label. It includes the execution evidence the agent needs to decide what to do next: logs, completion status, timing, and partial output when a job is still running. This matters because reproduction agents rarely need a single monolithic training run; they work through a trial and error sequence of tests, attempts, fixes, ablations, and evaluations. Figure 17 shows the execution path inside Hive.

Input. Hive’s input is a job request. In our interface, the concrete entry point is a terminal command: hive submit, whose parameters specify what should run. –workdir anchors the run to the code and files it should see, –gpus and –cpus declare the requested capacity, –resource targets a named pool such as test, CPU, or GPU capacity, and –timeout-sec bounds jobs that hang or run past their intended scale. Optional fields such as retry count and log directory make the request easier to recover and inspect. These fields are useful because they turn an open shell command into a schedulable execution unit: Hive can decide where it can run, how much capacity it needs, when it should stop, and where its evidence should be recorded.

![](images/ae51193a55d6b380d1f16991b69f3c8942837cc51156b3ba329ea2b450247e94.jpg)  
Figure 17. Hive execution overview. A job request enters Hive with a command, resource target, working directory, and time limit. Admission queues the request, dispatch routes it to test, CPU, or GPU capacity, and monitor records the return code, runtime, and logs that form the job record.

Output. Hive returns a job identifier and maintains a job record, not an experiment-specific result file. The record directory contains meta.json, stdout.log, and stderr.log: metadata for status, timing, resource use, and return code, plus streamed logs for diagnosis. Any checkpoints, metrics, or plots created by the command remain in the working directory; Hive records how the execution went.

Elasticity. A job asks for a category of resource (test, CPU, or GPU) and Hive places it on whatever node is free, without the agent naming a specific machine. The pool can grow or shrink as capacity becomes available, so the first worker to come online is useful immediately, and the same request format serves every experiment.

Node Types. Hive exposes three node types through the same job interface. CPU nodes support CPU-only experiments, setup, preprocessing, and other jobs that do not require accelerators. GPU nodes run acceleratorheavy experiments such as training and large-scale evaluation. Test nodes are for quick code validation and small-scale testing, with jobs capped at one hour so agents do not tie up long-running allocations.

Reliability. Long-running jobs are not perfectly reliable: a job can stall or be killed before it finishes, and its partial output is often still useful. Hive therefore treats an interruption as a reportable outcome rather than a silent failure: it writes the job’s last state and logs into the job record, then requeues it on another valid machine, where the command restarts from the beginning; the interrupted attempt is recorded for inspection like any completed one, and its incomplete work is not charged against the budget.

Implementation. We use Ray (Moritz et al., 2018) to implement Hive. Ray starts each job on a free machine and runs it; everything else the compute pool needs for agent-driven reproduction is built on top. A resource manager tracks CPU, GPU, and test capacity; a dispatcher turns agent requests into queued jobs; an agent-facing client/API submits jobs and streams progress back; and a persistent job store records logs, status, and final outputs outside the Ray runtime. These pieces are what make the compute pool usable as research infrastructure, not just a distributed execution engine.

## A.2 Nexus: Experiment Boundary

Nexus is the experiment boundary that manages the resources agents use. It is a bridge between agents and compute. An autonomous agent should not be trusted to enforce its own compute budget simply because the prompt tells it to be careful. Nexus makes the budget a property of the system. Each reproduction attempt receives a bounded project: jobs must fit within its resource limits, its concurrent usage cap, and its remaining GPU-hour budget before they are allowed to run. Compute is no longer something the agent can spend freely; every job is counted against the project’s budget. The operations view in Figure 18 shows these exchanges: agents submit or review jobs through Nexus, and Nexus communicates between infrastructures and agents.

![](images/8a7d968c0c43b62791a70551b628a4308b8bb5729e6ec9ff6497e8e0b8ac90b4.jpg)  
Figure 18. Nexus operations overview. Nexus stores the job, project, budget, and review state needed to translate agent requests into Hive actions and to translate Hive execution evidence back into agent decisions.

Quota Semantics. Nexus enforces several constraints that matter for reproducibility: how much compute a paper can spend in total, how much it can hold at once, which kinds of resources it may use, and how long any single job may run before it times out. At the job level, these limits cover CPU, GPU, memory, and time requests. These constraints are checked outside the agent. The project-level example in Figure 19 makes the gate concrete: a request can be rejected because it exceeds the remaining budget or concurrency capacity, or requests a resource pool outside the given policy.

Fairness. This accounting matters for both fairness and interpretability. Without it, one agent can consume resources in ways that alter the conditions for others, and the resulting comparison would conflate scientific dificulty with uncontrolled resource use. With Nexus, the amount of compute available for a single reproduction is explicit and enforceable. The agent can still decide how to reasonably spend its budget, but it cannot silently exceed the given total.

![](images/9a6cc0e7a80376f3a6036ae138c16900829ba92c8df9ace1aa4db9db4faee09e.jpg)  
Figure 19. A Nexus project turns resource policy into explicit accept/reject decisions. The example shows how budget, concurrency, and allowed resource pools gate agent requests before they reach Hive.

Periodic Review. Nexus makes long jobs interruptible without requiring the agent to keep an active turn open for the whole execution. At each configured review interval, Nexus snapshots the latest logs, records them in the workspace, and wakes the responsible agent to inspect them. The agent can then continue the run, cancel it, or change strategy before spending the remaining budget. Interval review pays of because many broken reproductions are visible well before natural termination: imports fail after setup, data paths are wrong, losses diverge, or an evaluator writes clearly invalid output. A run that would otherwise consume hours can stop after the first snapshot that shows it is going nowhere. If the agent does not respond within the review window, Nexus continues the job rather than leaving compute blocked by a missing response.

Separation. This layer is intentionally separate from Hive, and can accept any compute infrastructure. Nexus is optimized for accountability. Keeping them separate prevents the compute pool from accumulating task-specific policy and prevents Swarm from having to reason about low-level execution state.

## A.3 Swarm: Builder Coordination

Swarm has two jobs. The first is per paper: it drives one reproduction across many agent turns and several roles, keeping the attempt coherent as builders hand of, retry, and audit each other’s work. The second is across papers: it runs many of these attempts at the same time, each as its own trajectory, without their states blurring together. Both are hard because the work is open-ended. A reproduction can produce useful code without supporting the paper’s central claim, support the claim but leave no reusable environment, or produce scores that look plausible until a later audit checks them. If these judgments live only in chat transcripts, the result is a collection of anecdotes. Swarm therefore turns each paper attempt into an auditable trajectory: the assigned agent, budget, workspace state, results, milestone, and next role are explicit at every step. Figure 20 shows the end-to-end path: a pool of paper workspaces, each driven through an ordered role workflow under a milestone gate, ending in an artifact that is either accepted or set aside.

![](images/cbed619403f7e21467211282a3b2e7142a5904ac52f15dffd896220deaa4b196.jpg)  
Figure 20. Swarm coordination overview. Swarm drives a pool of paper workspaces in parallel. Each paper moves through an ordered role workflow—reproduce, design, package, verify—whose transitions are driven by wake events: worker start, job submit, job update, validation failure, and role transition. Progress is graded against a milestone ladder, from none up to full reproduction, and the milestone gate decides whether a paper’s artifact is accepted.

Pipeline. Swarm turns a paper into a finished environment. Its builders reproduce the paper’s main claim, design the reproduction into a usable task, organize its files, and verify the result. Swarm does not run this in one long session. A builder works in short turns, and ends each turn with an action, usually submitting a compute job or signaling that it is done. Swarm tracks that action and the paper’s state, and whenever something changes it wakes the builder with a prompt written for that exact situation: for example, when a compute job finishes, a prompt that carries the results and asks what to do next; when a build fails, a prompt with the build log and a request to fix it; when one stage is complete, a prompt that briefs the agent for the next. The agent is therefore only ever asked to decide the one thing that just came up. Swarm runs this loop over a pool of many workers, so it advances a large number of papers at the same time.

Coordination. Swarm answers the question that remains after a single agent turn ends: who should act next, and what do they inherit from the builder before them? Its unit of state is the paper, not the agent session: Swarm tracks where each paper sits in its workflow and the milestone it has reached, so a new builder resumes from the right point and can build on a partial result left by the previous one. This is what lets a single paper pass through several builders, even diferent models, and still read as one coherent attempt; without it, the final artifact would mix the paper’s dificulty with hidden diferences in how the agent happened to be run.

Wake Events. Swarm wakes a builder only when new information changes what it should decide: a fresh paper or a new role to begin, a running job it may continue or cancel, a finished job, a failed build, an exhausted budget, a validation or milestone-gate failure, or a role transition. The agent returns a structured action, and Swarm records both the action and what prompted it. Table 5 summarizes what is shown.

Completion Checks. When a builder declares a pass finished, Swarm separates local correctness from progress. The local check confirms the action is parseable and the milestone label is canonical, then runs the workspace validator, which verifies that the score files exist, the evaluator is separated from the method, and the workspace rebuilds from a clean clone (§C.4). Failures become repair prompts, capped at three validation attempts. The progress check asks whether the claimed milestone clears the bar for the current stage; a run that falls short reverts to an earlier pass or stops (§A.3.3).

<table><tr><td>event</td><td>shown to agent</td><td>kept in trajectory</td></tr><tr><td>iteration start</td><td>paper, workspace, milestone, budget</td><td>model, role, starting checkpoint</td></tr><tr><td>job running</td><td>streaming logs and remaining budget, to continue or cancel</td><td>the continue or cancel decision</td></tr><tr><td>job finished</td><td>logs, exit code, and remaining budget</td><td>outcome, compute used, next action</td></tr><tr><td>build failed</td><td>the build log, to fix and resubmit</td><td>failure cause, fix attempts</td></tr><tr><td>budget exhausted</td><td>budget spent, no further jobs, wrap up current work</td><td>final spend, the milestone the agent wraps up at</td></tr><tr><td>validation fail</td><td>validation or milestone-gate failure</td><td>resulting status, revert attempts</td></tr><tr><td>role transition</td><td>prior milestone and unresolved issues</td><td>next role and inherited state</td></tr></table>

Table 5. Swarm wakes a builder only at decision-changing events. Rather than holding a session open for the whole reproduction, Swarm re-engages the builder only when new information changes what it should do next. Each row gives the triggering event, what the agent is shown, and the state Swarm writes into the paper’s trajectory.

Worker Pool. Swarm runs many paper workspaces through the same loop at once. It assigns idle Workers to saved workflow states, resumes the correct role after each turn, and sends experiment declarations to Nexus under that paper’s budget. The builder chooses the scientific next step; Swarm handles scheduling, accounting, and routing the returned evidence back to the right workspace.

## A.3.1 Milestones

Progress Ladder. Swarm measures progress with a canonical milestone ladder, not with a binary success flag. The main target in the standard workflow is core\_claim: a minimum experiment must run the paper’s method and support the central claim. method\_runs records weaker progress when the method executes but the claim is not yet supported; core\_claim\_plus, secondary\_claims, majority, near\_complete, and full record broader coverage; and not\_possible records a case that should not be converted into an environment. These labels let Swarm preserve partial progress without recording it as success. The canonical labels are listed in Table 6. The agents self-report this.

Gating. Milestones also determine whether the workflow should continue. If a builder declares completion below core\_claim, Swarm gives one or more turns asking for additional evidence. If a required pass still falls short, Swarm can stop the trajectory or revert to an earlier pass. This keeps failed attempts inspectable without letting weak evidence advance.

<table><tr><td rowspan=1 colspan=1>milestone</td></tr><tr><td rowspan=1 colspan=1>not_possible</td></tr><tr><td rowspan=1 colspan=1>nonemethod_runs</td></tr><tr><td rowspan=1 colspan=1>core_claim</td></tr><tr><td rowspan=1 colspan=1>core_claim_plus</td></tr><tr><td rowspan=1 colspan=1>secondary_claims</td></tr><tr><td rowspan=1 colspan=1>majority</td></tr><tr><td rowspan=1 colspan=1>near_complete</td></tr><tr><td rowspan=1 colspan=1>full</td></tr></table>

Table 6. Milestones make partial progress explicit. Swarm records one canonical label for each run.

## A.3.2 Builder Roles

Role Design. Swarm does not give every agent the same job. It separates reproduction, environment design, packaging, and verification so that later agents can inspect and improve the artifacts left by earlier agents; we refer to these roles collectively as builders, the agents that construct an environment from a paper. Reproducers build and test the paper method; design agents convert a successful reproduction into a scored environment; the packager makes the workspace reusable; and verifiers audit whether the final milestone is supported by evidence. This role split keeps the workflow from depending on a single long agent session while still preserving one shared workspace. Table 7 summarizes the role responsibilities and the gate to satisfy.

Milestone Selection. Each builder also chooses the milestone that best describes the current workspace after it is done. The choice is a self-evaluation against the canonical ladder in Table 6: the agent claims the milestone supported by runnable code, logs, scores, and notes, rather than simply reporting when the assigned role finishes. The label is a claim, not a verified fact, so it is never taken as final on its own: each later role and iteration inherits the same workspace and re-audits the previous agent’s claim, and the verifier lowers any milestone the workspace’s logs, scores, and code do not support. The bar for core\_claim in particular is that the supporting run executes the paper’s own method on the paper’s task, possibly at reduced scale, and reproduces the direction of its central result, not merely that the code runs without error.

<table><tr><td>role</td><td>main responsibility</td><td>gate to continue</td></tr><tr><td>reproducer</td><td>implements the paper method and tries to reproduce the central claim.</td><td>reach core_claim, or stop or revert.</td></tr><tr><td>designer</td><td>converts the reproduction into gym targets, constraints, and scoring.</td><td>scores must validate</td></tr><tr><td>packager</td><td>consolidates scripts, environment files, and download logic.</td><td>workspace needs to be able to rebuild.</td></tr><tr><td>verifier</td><td>audits whether the final evidence supports the claimed milestone.</td><td>unsupported claims have their milestone lowered.</td></tr></table>

Table 7. Role separation keeps one workspace auditable across builder handofs. The role split separates reproduction, environment design, packaging, and verification while keeping all roles on the same workspace and evidence trail.

## A.3.3 Workflow

Pass Sequence. The standard workflow is an ordered list of stage rows (Figure 21). Each row specifies the model profile, role, GPU-hour budget, milestone cap, minimum milestone, and optional revert target for that pass. When a pass finishes, Swarm compares the claimed milestone with the row’s minimum requirement. Meeting the gate advances to the next row; falling below it triggers the configured revert target, or otherwise stops the trajectory as unsuccessful. Because the budget is attached to the row, routine checks stay small, experiment-heavy rows receive more room, and every paper is judged under the same stage-level limits.

Repeated Passes. A workflow can include repeated rows for the same role. This is deliberate: a single builder round may leave partial progress, fragile evidence, or an unresolved design choice that should be inspected before the workflow advances. Repeated rows let the next agent continue from the saved workspace, try a diferent strategy, or audit the previous claim without treating the role as complete.

Multi-Agent Handofs. The handof structure also makes the workflow a multi-agent process rather than a single long session. A later builder inherits the workspace, evidence, and gate left by the previous agent, so it can inspect earlier choices before extending them. This gives the system diversity in search strategy and reduces the chance that one mistaken assumption carries through the entire artifact. It also saves cost by placing lower-cost model profiles on routine continuation and reserving stronger profiles for dificult passes. Since the handof includes the current milestone claim, each agent can revise the milestone selected by previous agents.

![](images/f2df17c8311e0d90830b837ca990debd38fa836695feedf39174411656823b7b.jpg)  
Figure 21. Swarm workflow stages. The standard workflow expands reproduction, design, packaging, and verification into repeated builder passes; each row carries the model profile, gate, GPU budget, and possible revert target.

## A.4 Worker: Agent Interface

We design Worker as one interface through which every agent scafold CLI connects to the system. Scafolds vary in how they are launched, resumed, and billed; Worker absorbs that variation so a model is judged on the research it produces, not on the scafold that drives it. Figure 22 shows this path: heterogeneous scafolds enter through one uniform turn interface, run inside a constrained container, and end each turn by writing an action.yaml that wakes the system for the next turn.

![](images/0f740dffe6252c96b8e6d9b9a59976f17b5f1ae28418551728105db0aae94387.jpg)  
Figure 22. Worker runtime overview. Diferent agent scafolds connect through one uniform turn interface. Worker runs the agent under a container runtime with capped resources and a read-only validator, cycling through a turn loop of render prompt, agent acts, and submit jobs. Each turn ends by selecting the next action in action.yaml, which the system reads before waking the agent for the following turn.

Turns and actions. A session is a sequence of turns. The first turn sets the agent up in its workspace; every later turn is short, sent only when something changes, such as a finished job or a failed build. The agent ends a turn by writing one file, action.yaml, that states its decision, shown in Figure 23.

The agent makes its request in this fixed format. The system then processes it to decide the next step. action says whether to submit a job or declare the paper done; gpus and test set its size and route it to the full or the short test pool; milestone records the agent’s own assessment against the fixed milestone ladder. Because the agent never launches a job itself, the system can check every request, count it against the budget, and schedule it before it runs; a malformed one is handed back to rewrite, while the reasoning behind it survives in the turn record. The loop also removes any need to finish in a single session: the agent is re-engaged as jobs complete and builds fail, and progress accumulates between turns.

![](images/847c373ed67101ee1ce8af6b40b28b7c47870c4ef4b2f7b4070dda4f40fde847.jpg)  
Figure 23. The action.yaml that ends a turn. The agent writes it to request the next job or to stop, and the system reads it after every turn.

Containment. We run the agent inside an Apptainer container (Kurtzer et al., 2017) whose home is the paper workspace. It sees only a base image and that workspace: the container hides the cluster’s home and temporary directories and drops all administrator bind mounts, so the shared filesystems are out of reach. Inside, it edits code freely, and the GPU jobs it later submits reuse the same layout, so no host path is ever exposed. Three limits keep one agent from starving its neighbors: a memory cap to bound total RAM, a low scheduler priority to make co-located agents yield CPU, and thread caps on the numeric libraries.

Dependencies. Installing packages is separated from running jobs. The agent does not install during a job; it declares what it needs in two workspace files: environment/container.def for the base image and OS packages, and environment/setup.sh for Python packages. The system builds these once and keeps the result in a persistent overlay on top of the base image, so the packages are available to every later job without reinstalling or rebuilding the container. Editing either file triggers a rebuild before the next job runs.

Boundary and robustness. We place a few things outside the agent’s reach by construction: the bookkeeping files and the recorded trajectory are hidden, the validator is read-only, and git push is disabled so the system decides when work is published. Turns are stripped of credential patterns before they are written, since the per-paper repositories may be made public. And because a paper can span thousands of unattended turns, Worker recovers on its own: it retries transient errors such as rate limits, switches to another model profile on terminal ones such as an exhausted usage cap, and kills a stalled turn along with its whole process tree.

## A.5 How the Components Connect

Forward path. The four components pass work to each other through recorded state, not inside one long agent session. Worker presents the paper and workspace to the agent and reads its action from action.yaml. Swarm owns the paper’s trajectory: it tracks the milestone, advances through the workflow of roles, prepares the workspace, and turns a submit action into a job request for Nexus. Nexus checks the project’s budget, its concurrency cap, and the requested resource pool, then forwards the job to Hive, which runs it.

Return path. Results travel back through the same components. Hive reports the job’s state, return code, and resource usage; Nexus matches the finished job to its run record, writes the logs and outputs into the workspace, releases the GPUs it was holding, and charges the run against the project’s budget. Swarm reads the event, updates the paper’s state, and wakes the Worker for the next turn. Because everything lands as files the agent can open, not as hidden service state, a later turn or a diferent model can look back at what each job did and how it ended.

Why split it this way. The split gives each component a single, narrow responsibility, and that is what keeps the system auditable. Worker can think and edit but cannot give itself compute; Swarm can coordinate papers but does not run jobs; Nexus enforces resource limits without judging scientific progress; Hive runs jobs without knowing which claim they support. Because the boundaries between them are clean, the four parts are meant to work independently, and any one of them can be swapped for a diferent implementation, a new scafold, scheduler, or compute backend, without touching the others. The final artifact is tied to both sides: the agent trajectory that produced it, and the run records that make that trajectory checkable.

## B Builder Design

runs entirely on autonomous agents: every environment is produced by builders that reproduce a paper and shape the result into a task, with no human in the loop. This section describes how those builders work and the principles that decide whether the environment they leave behind is a good one. Figure 24 gives the overview: from a paper and an empty workspace, builders produce a reusable environment across problem definition, method reproduction, and evaluation.

![](images/90beba2e00bf49c5a2056a7d411ae16ad71bc47e8eb0832ce2333507ce77577b.jpg)  
Figure 24. Builder overview. From a paper (in PDF and Markdown) and an empty starter workspace, builders build a reusable environment along three tracks: a problem definition that states the task and its constraints without leaking a solution, a reproduction of the paper’s method whose code and experiments recover the central claim, and an evaluation harness with visible and hidden slices that turns outputs into scores.

## B.1 What the Builders Do

The task. §A described the system that runs the builders; this section is about the research task it is built for. Given a single published paper, the agents must reproduce the paper’s central result and leave behind a reusable environment, an executable setup in which a later scientist can attempt to improve on that result. This is the foundation everything else rests on, since an environment whose method was never really implemented would have nothing to measure. The rest of this subsection traces what the agents actually do, from the paper to the finished environment and how each turn is driven; the design choices behind that environment follow in §B.2.

From paper to environment. A builder’s input is the paper itself, in both Markdown and PDF with its figures, together with a fresh, empty workspace. The reproducer starts from this and proceeds in order. It first reads the paper and writes a short briefing, separating the problem and the evaluation protocol from the paper’s method, and transcribes the paper’s reported numbers into a reference scoring file. It then prepares the environment: downloading the data and any model weights in advance, writing an evaluator that scores outputs without depending on the method, and checking that the scoring file is well formed. With that in place it implements the method and reproduces incrementally, submitting compute jobs to get one experiment that supports the core claim running before expanding to further settings, and recording its measured numbers alongside the paper’s. Reproduction usually spans several agents in turn: each later one inherits the saved workspace, audits what the previous agent built, and pushes it further.

Finishing the environment. A working reproduction is not yet a usable environment. Once the method runs and supports the claim, a designer adds the visible and hidden evaluation slices and the scripts that produce their scores; a packager makes the workspace rebuild from a clean clone, with every dataset and model fetched by a self-contained download script; and a verifier audits the whole artifact before it is accepted. How these roles are sequenced, what budget each receives, and how progress is graded are handled by the system (§A.3); what each role leaves in the workspace is detailed in §C.

How a builder is prompted. The builder works in turns, ending each by requesting a compute job or declaring the work done, and runs fully autonomously with no human in the loop (§A.4). Because a paper can span thousands of unattended turns, it is not given one fixed instruction. Where §A.3 describes when an agent is woken and what state it is shown, here the point is what it is then asked to do: the first turn carries the full role briefing, and every later turn is a short prompt written for the condition that woke it. Table 8 lists the main conditions and the instruction each prompt carries.

<table><tr><td>condition</td><td>what the builder is asked to do</td></tr><tr><td>new paper</td><td>read the paper and reproduce its method from a full briefing</td></tr><tr><td>continued iteration</td><td>audit the previous workspace and push it further, starting one milestone below the prior claim</td></tr><tr><td>job running</td><td>read the streaming logs and decide whether to let the job continue or cancel it</td></tr><tr><td>job finished, budget left</td><td>inspect the results, record progress, and plan the next job</td></tr><tr><td>job finished, budget spent</td><td>stop submitting and wrap up the milestone, scoring files, and environment</td></tr><tr><td>build failed</td><td>read the build log, fix the container definition, and resubmit</td></tr><tr><td>stopped early</td><td>keep going; the stop is not yet justified against the target milestone</td></tr><tr><td>signaled done</td><td>finalize and commit, a permanent decision</td></tr><tr><td>validation failed</td><td>repair the flagged scoring or structural problems and re-sign</td></tr></table>

Table 8. Each turn is prompted for its condition. The first turn is a full role briefing; later turns receive a short prompt for whatever event woke the builder, keeping it on task across a long unattended run.

## B.2 Design Choices

A good environment does not come from reproducing a paper alone; it also has to be designed so that a later method can be tested fairly on it, all under a limited compute budget. We follow seven design choices toward this, each stated first and then explained.

Design choice 1. Reproduce the paper’s method and core claim, not its exact numbers. Scale down resources when needed, but preserve the experimental procedure and keep the task non-trivial.

The goal is to make the paper’s method run, not to match a table entry. Exact-number reproduction rewards the wrong behavior: a builder could tune toward the published values, or copy them outright, without implementing the method. Instead, the environment should contain the paper’s actual computation and produce reasonable support for the claimed efect. A simple check is whether a reader of the code can recognize the algorithm from the paper.

Most papers cannot be reproduced at full scale under a fixed budget. The builder may therefore reduce the model size, dataset size, or number of training steps, which also keeps reproductions friendlier to the broader community using the artifact. What it may not do is change the procedure. If the paper trains from scratch, the environment trains from scratch. If the paper uses a particular optimization setup, the environment keeps that setup. Resources can shrink; the experiment cannot become a diferent experiment.

Scaling down also has a limit. A task made too small becomes trivial, and a trivial task cannot distinguish the paper’s method from a weak baseline. When a builder uses a smaller or synthetic dataset, it must check that a basic baseline does not already solve the problem. If the reproduced method is no better than random guessing, majority class, or another weak baseline within noise, the reproduction has failed. The environment needs a real gap so that future agents have something meaningful to improve.

Design choice 2. Keep the problem separate from the solution: the problem should be stateable without the paper’s method, and the evaluator and data should not depend on that method.

This boundary keeps the problem definition independent of the solution that produced it. The problem, its data, and its evaluator can be stated and run without the paper’s method, so the environment poses the problem on its own terms rather than only as a continuation of one particular implementation; the file boundary is described in §C.2. The separation also protects the score. If the evaluator secretly depended on the reproduced method, then changing the method might not change what is measured, and the environment would look functional while no longer testing the intended problem.

The separation must hold at runtime, not only in the file layout. The evaluator must read the outputs a method produced, never call the method’s code: if deleting method/ would change the evaluation scores, the boundary is broken. Likewise, the data loaders belong to the problem, not the solution, and must not import from method/. The evaluate scripts must also discover methods dynamically, enumerating whatever methods exist on disk rather than hardcoding a name, so that the evaluator works for a future method without editing. How the environment is then presented to a future agent is a matter for evaluation (§E.1).

Design choice 3. Provide a visible evaluation slice for development, and reserve a matched hidden slice for final evaluation to detect shortcuts and overfitting.

The finished environment includes an evaluation suite that a future scientist can see and optimize against. If that suite is only one metric on one dataset, the easiest path may be to game that number rather than improve the underlying idea. To reduce this risk, the designer builds two slices: a visible slice for development and a hidden slice that is run only after done. A large gap between the two suggests overfitting or shortcuts.

The split can be drawn in diferent ways, depending on the paper: held-out samples, held-out subgroups, harder inputs, or several related metrics that measure the same claim from diferent angles. Using multiple metrics from diferent angles is stronger than a single held-out number, because a single metric can be gamed while triangulation from several directions catches shortcuts that any one metric would miss.

The hidden slice must be calibrated against the visible one in dificulty: both should probe the method at roughly equal rigor, using matched data scales, evaluation protocols, and sample counts. An honest method should score similarly on both; if the hidden slice is simply harder, then every method will look worse there and the gap will be meaningless. The two slices should also be independent, so reading or importing the visible evaluator does not reveal the hidden one. And the test slice should be lightweight: it re-scores existing artifacts or slices them diferently, rather than retraining or re-running expensive inference.

Design choice 4. Score future methods on the quantity the paper cares about, not on how closely they match the paper. Fix the constraints needed to make gains meaningful.

An environment should allow future methods to improve on the paper. For that reason, the score must not measure distance from the paper’s published number. Such a metric would make the paper’s own method the target and turn the environment into a reproduction-fidelity test. The paper’s reported numbers should be recorded as context, but future methods should be scored on the underlying quantity: accuracy, loss, robustness, sample eficiency, parameter count, or whatever the claim is actually about.

This principle is enforced in the scoring schema: metric names that encode closeness to the paper, such as deviation, vs\_paper, or agreement\_with, are blocked by the validator, so the score must name the underlying scientific quantity.

The environment must also define what stays fixed. A future scientist should not win by using more compute, a larger model, more data, or a diferent training paradigm unless that is the allowed direction of improvement. For papers that claim a benefit without a cost, the evaluator should measure both sides: the benefit that should improve and the constraint that should not break. The scoring contract expresses this through metric coeficients: a positive coeficient marks a benefit to maximize, a negative one marks a constraint that must not degrade, and the experiment weights reflect how central each result is to the paper’s claim.

Design choice 5. Score on quantities that do not depend on the machine. Replace hardware-sensitive measures such as latency or wall-clock time with stable proxies.

Hardware varies from run to run, so a latency or wall-clock measurement can swing severalfold with no change to the method, and an environment that scored it would reward landing on a faster GPU rather than finding a better idea. Scores therefore use quantities that are stable across machines: parameter count, FLOPs, training steps, examples processed, or final task performance. When the paper’s own claim is about speed or eficiency, the environment keeps the claim but measures it through a machine-independent proxy rather than the clock.

Design choice 6. Reproduce enough of the paper to give future scientists a real surface to improve, preferably a small matrix over the paper’s own axes rather than one isolated number.

A one-number reproduction is too thin a target. A future scientist may match it and still have little room to learn from or improve. Once the core claim runs, the environment should expand along the axes the paper itself studies: datasets, input dificulty, model scale, ablations, data regimes, or other relevant settings. A small matrix over two meaningful axes is usually more useful than repeated runs of the same setting.

This matrix does not need to be large. It needs to expose the structure of the claim. Combined with the non-triviality check above, it turns the environment from a single point estimate into one with headroom: weak baselines leave a gap, the reproduced method occupies a recognizable starting point, and future methods have several genuine ways to improve.

Design choice 7. The environment must rebuild and rerun from a fresh clone, with every external resource fetched by a self-contained script and every dependency declared.

An environment that only works on the machine that built it is not reusable. The download script must fetch every dataset, model weight, and external resource from a public source using executable commands, not comments or placeholder URLs. It must be idempotent and must not reference local shared paths that exist only on the original cluster. The container definition and setup script must declare every system and Python dependency so that the software rebuilds without manual intervention. And every local import in the evaluation, method, and baseline code must resolve to a file tracked by git, so that a fresh clone contains everything the experiments need to run.

The validator enforces this: it scans committed code for references to machine-local shared directories and checks that every import target is a tracked file. Together with the reproducible-execution scripts (§C.3), this makes the environment a portable artifact rather than a snapshot of one session’s state.

## C Agent Workspace

The workspace is the single directory the builders work in and the artifact they leave behind. Its layout is what makes the result a reusable environment rather than the residue of one run. We describe its structure (§C.1), its problem statement and scoring (§C.2), its execution (§C.3), and its records and checks (§C.4).

## C.1 Structure

Overview. Each paper is reproduced inside a single working directory, and that directory is also the artifact the builders leave behind. During the run it holds the code, data, experiments, and notes; once the run ends, it is designed to stand on its own, so that another party can clone it, rebuild the environment, rerun the method, and obtain the same scores without access to the original session. This is what distinguishes a reusable research environment from the residue of a single successful run, in which the method is entangled with the evaluator, the data exists only in a local cache, and the scores carry no well-defined meaning. The layout shown in Figure 25 is designed to prevent these failures.

![](images/58440ca11ce40307053c90f1b88705d5f41bc328e72ce1a3d4b6c3dfd6f83ec9.jpg)  
Figure 25. The agent workspace. A single directory holds the paper and briefing, the reproduced method and baselines, the data and evaluator, the scoring contract, the run scripts and environment, and the root-level record and validation files. Each directory has a single responsibility, which keeps the problem separated from the solution, lets the workspace rebuild and rerun from a fresh clone, and preserves a record of how the reproduction was produced.

File Roles. The layout is a deliberate design choice rather than a storage convention: each directory has a single responsibility. The briefing defines what a future scientist may know; method/ and baseline/ hold what the reproducer built; data/ and eval/ define the problem interface that any method must satisfy; scoring/ defines how a method is judged; and scripts/ and environment/ determine whether the workspace runs outside the original session. Table 9 summarizes the groups.

## C.2 Problem and Scoring

Problem and Solution Boundary. The workspace deliberately separates the problem from the solution so that the problem is method-agnostic: it states what must be achieved and how it is measured, without revealing the paper’s approach. This is what allows the workspace to serve as a fair task for any future method rather than a recreation of one particular solution. The boundary appears in two places. In the briefing, problem.md gives the research question and success criteria and evaluation.md the metrics and protocol, neither naming the method; only method.md describes the paper’s approach, and overview.md summarizes the finished workspace. In the code, method/ contains the reproduced algorithm, while eval/ and data/ are written to work for any method, so a score never depends on the specific implementation. Because the problem stands on its own, an innovate scientist can attempt it from the briefing and evaluator alone, whereas an improve scientist additionally receives method/ as a starting point (§E).

<table><tr><td>group</td><td>purpose</td></tr><tr><td>briefing/</td><td>states the task and how success is measured, while keeping the paper&#x27;s method out of the problem files</td></tr><tr><td>method/, baseline/</td><td>the reproduced implementation, with the baselines used to interpret its score</td></tr><tr><td>data/, eval/</td><td>the data interface and a method-agnostic evaluator that any future method must run against</td></tr><tr><td>scoring/</td><td>the scoring contract: how each metric counts, what an improvement means, and what must stay fixed</td></tr><tr><td>scripts/, environment/</td><td>everything needed to rebuild the environment and rerun the method from a fresh clone</td></tr><tr><td>root files</td><td>the current action, the claimed milestone, progress and deviation notes, and the automatic checks</td></tr></table>

Table 9. Each directory has one responsibility. The workspace keeps task context, method code, evaluation, scoring, execution, and record state apart, so a later scientist can reuse it without the original conversation.

Scoring Contract. Scoring is held in two files. scoring/reference.json records the paper’s reported numbers for each baseline and proposed method, along with the scoring rules: a weight per experiment, set by the agents to reflect how central that result is to the paper’s core claim (weights sum to one), the direction of each metric, and a label marking each method as a baseline or a proposed method. scoring/scores.json holds the numbers the reproduction actually produced. A later method is scored by its improvement over those reproduced numbers, in each metric’s direction and combined by the weights, so the score measures progress over the reproduction rather than closeness to the paper. Figure 26 shows the two files side by side.

Targets and Constraints. The JSON files fix the numbers, but a scientist also requires the intent behind them. Four short markdown files provide it: TARGETS.md states what to optimize, CONSTRAINTS.md states what must remain fixed, DIRECTION.md states which kinds of change are permitted, and EXPERIMENTS.md describes each scored experiment without disclosing the method. Together they exclude trivial gains, such as substituting a larger model, adding unrelated data, or altering the problem itself, while preserving room for a genuine methodological improvement. For example, an agent may improve performance on a Mamba architecture optimization problem by replacing the architecture with a Transformer. While this could yield better results, such a change provides no scientific value, as it sidesteps the original problem rather than advancing our understanding of how to improve Mamba models themselves. When a paper makes a trade-of claim, both sides must be scored: the benefit and the constraint it must not violate. Otherwise the easiest route to a high score is to break the very condition the paper was designed to satisfy.

## C.3 Execution

Reproducible Execution. The workspace must be able to rebuild from source, not only on the machine where it was created. scripts/download.sh fetches external data, weights, and other public resources;

```jsonl
# scoring/reference.json - weights, directions # scoring/scores.json - same keys,
{ the reproduced numbers
"experiments": {
"main_result": {
"weight": 1.0, {
"primary_metric": "accuracy", "experiments": {
"metrics": { "main_result": {
"accuracy": {"higher_is_better": true, "results": {
"coefficient": 1.0}, "baseline": {"accuracy": 75.8,
"cost": {"higher_is_better": false, "cost": 25.6},
"coefficient": 0.2} "method": {"accuracy": 82.1,
}, "cost": 25.6}
"results": { }
"baseline": {"type": "baseline",
"accuracy": 76.1, "cost": 25.6}, }
"method": {"type": "proposed", }
"accuracy": 82.3, "cost": 25.6}
}}}}
```  
Figure 26. The scoring contract, held in two files. scoring/reference.json fixes the weights, metric directions, coeficients, and the paper’s reported numbers; scoring/scores.json holds the numbers the reproduction actually produced, under the same keys.

it must be idempotent and must not reference local shared paths. environment/container.def captures the system dependencies and environment/setup.sh the packages. scripts/method.sh then runs the reproduced method, scripts/baseline.sh runs the comparisons, scripts/evaluate.sh scores the outputs through eval/, and scripts/reproduce.sh runs the full pipeline end to end. The workspace is therefore an executable artifact rather than a static record of results: a fresh clone rebuilds the environment, reruns the method, and regenerates the score files.

Visible and Held-Out Slices. A finished workspace also includes a train/test split for the scientist who reuses it. The visible slice, eval/train/, is what a future agent develops against; the held-out slice, eval/test/, lives in the environment’s pristine source copy and is materialized into a run only for the final score, so it is never present in the agent’s working tree during development. The paired scripts/evaluate\_train.sh and scripts/evaluate\_test.sh write scores\_train.json and scores\_test.json. The purpose of the split is not to make the paper harder to reproduce but to detect a future agent that has overfit the visible evaluator.

## C.4 Records and Validation

Design Summary. Each choice in the workspace exists to rule out a specific way it could fail to transfer to a new scientist. Figure 27 collects them: the design decision, the failure it prevents, and where it lives.

![](images/1bb8bedd5daf32904dde1eeb008ffd6f7f5ebe83bc0a5ad78dc1f8ccbde87982.jpg)  
Figure 27. Each design choice targets a concrete failure mode. The workspace is structured to avoid solution leakage, method-coupled scoring, unrebuildable dependencies, evaluator overfitting, and unsupported claims.

Reproduction Record. When a run ends, the system writes a manifest.json into the workspace and compiles a README.md containing a per-iteration reproduction log, alongside the git history that checkpoints every turn. Together these tie the result to recorded state rather than to the conversation, allowing a later agent or reviewer to trace how it was reached. Table 10 groups the fields the manifest records.
<table><tr><td>group</td><td>fields recorded</td><td>use</td></tr><tr><td>reproduction summary</td><td>final milestone and status, scaffold and model, workflow role and step, and turn, run, and iteration counts</td><td>attributes the result to the builders and role that produced it</td></tr><tr><td>compute and cost</td><td>GPU and CPU hours, the GPU budget, jobs run, wall-clock duration, tokens, and dollar cost</td><td>separates a scientific failure from an execution failure, and compares spend</td></tr><tr><td>components present</td><td>whether the container, method, evaluator, scripts, reference, and data are all in place</td><td>confirms the workspace is complete enough to rerun</td></tr><tr><td>per-builder records</td><td>one entry per builder, with its turns, its jobs, and why it stopped</td><td>reconstructs the result across handoffs</td></tr></table>

Table 10. The finished workspace records how it was produced. manifest.json ties each accepted artifact to the builders, compute, and components behind it, so it stays auditable after the session ends.

Acceptance Checks. A workspace is checked before it is accepted. validate.py runs automatically and enforces the scoring schema, blocks the obvious paper-agreement metric names, and checks the structural rules that keep the workspace independent of its own method and rebuildable from a fresh clone; Table 11 lists the checks in full. The markdown audit files then cover what code cannot decide: MILESTONES.md states the claimed milestone, PROGRESS.md and VERIFICATION.md record why it is justified and what was changed from the paper, and CHECKLIST.md confirms that the workspace is ready for both improve and innovate mode reuse. A workspace is accepted only when the automatic checks pass and the audit files are complete.

Version Control. Every workspace is a git repository, and the system commits to it continuously, so the full trajectory can be replayed from the repository alone, without the session that produced it. Each agent edit is committed as it is made, and each job submission is committed together with the exact code it ran, so the state behind any result is pinned to a specific revision. When a job finishes, its logs are written back into the workspace and committed as tracked files, so what ran and what it produced live in the same history as the code. The builder cannot publish: git push is disabled, and the system decides when it is released (§A.4).

Evaluation reuses the same git-backed design with one branch per attempt. Each time a model is run on an environment, that attempt is isolated on its own branch, so attempts never interfere, and each one can be inspected, rescored, or rerun on its own from the committed state it started from.

<table><tr><td>check</td><td>what it verifies</td></tr><tr><td>scoring schema</td><td>reference.json parses, experiment weights sum to one, and each experiment declares a primary metric, its metrics, and its results; every metric declares a direction and a coefficient; and every method is typed as a baseline or a proposed method, with at least one proposed</td></tr><tr><td>no paper-agreement metrics</td><td>metric names may not encode closeness to the paper (deviation, error_vs_, agreement_with, and similar); the score must measure the scientific quantity, not fidelity to the publication</td></tr><tr><td>scores match the schema</td><td>scores.json uses the same experiments, methods, and metric names as reference.json, and every value is numeric</td></tr><tr><td>reproduction sanity</td><td>each reproduced primary-metric value stays close to the paper&#x27;s: no sign flip, under a 2× magnitude gap, and under a 0.3 absolute gap for bounded metrics</td></tr><tr><td>evaluator independence</td><td>eval/ and data/ never import from method/, so the evaluator works for any method, not only the reproduced one</td></tr><tr><td>fresh-clone integrity</td><td>every local import in eval/, method/, and baseline/ resolves to a git-tracked file, and no committed code references the machine-local shared/ cache</td></tr><tr><td>train/test split</td><td>when a split is present: the two slices are independent and wired to the correct evaluator and output file, both score the same methods, the slices and the primary metric actually differ, and the primary metric separates methods on the train slice</td></tr></table>

Table 11. What validate.py checks before a workspace is accepted. The validator enforces the scoring schema, blocks scores that measure agreement with the paper, and verifies that the workspace is independent of its own method, rebuildable from a fresh clone, and free of train/test shortcuts.

## D Selecting Papers

Not every paper can or should become a suitable environment. A useful task has to be both worth reproducing and feasible for an autonomous builder to attempt under a bounded budget. Before any reproduction begins, we therefore narrow a large pool of recent papers down to a smaller, category-balanced set through a staged pipeline: cheap model screens drop low-quality and of-type papers, a deeper analysis scores each remaining paper for quality and reproduction dificulty, and a final ranking and balancing step chooses the papers we attempt. The rest of this section walks through that pipeline and reports how many papers survive each stage.

Candidate Pool. We assemble the candidate pool before any builder is launched. Our selection snapshot is exported from a global arXiv index of 124,816 unique papers. The index is built from daily category listings between January 1, 2025 and March 1, 2026, and exported on March 28, 2026. We restrict the source categories to cs.CL, cs.CV, cs.AI, and cs.LG (Figure 29), then deduplicate by arXiv ID because the same paper can appear under more than one category. We also drop papers whose primary category is statistics (the stat.\* group), which are dominated by theoretical work that ofers little to execute or reproduce. This gives the candidate pool a temporal and topical boundary before any model is asked to judge feasibility. Figure 28 reports how many papers survive each screening stage from this pool onward.

Fast Screening. Before any model reads a paper, we convert its PDF into Markdown with MinerU (Wang et al., 2024), which extracts the text, tables, and figures into a clean form that the screening and analysis models can read directly. The first two passes over this text are cheap, high-recall screens rather than final selection rules. Each pass sees only the opening 15,000 characters of the converted paper, which typically span the abstract and introduction, so that screening cost stays low across the full index. The first pass returns a quality score: MiniMax M2.7 (MiniMax, 2026) reads the excerpt and emits a short justification together with an integer rating from 0 to 100. A second pass assigns each paper one of five types: metric-method, benchmark/dataset, analysis/findings, theoretical, or survey/position. The quality screen is broad: 94.3% of scored papers are at least 50, 40.5% are at least 70, and only 12.0% reach 80 or higher. The type screen is more selective in the direction needs: 94,161 papers, or 75.9%, are labeled metric-method. Figure 30 reports the full quality-score and paper-type distributions. The quality screen is the gate into deep analysis: structured analysis runs only within the top quality band, papers scoring 75 or higher (33,393 papers, the top 26.9%), and 21,354 of them complete the analysis under the analysis budget. The fast type label guides this stage but does not filter it; the decisive type filter uses the structured analysis’s own classification and appears as the first drops in Figure 32.

![](images/f3da2c0bb2a43129d30db5b1e7217ca318e6b3bae2c991388f036d7605041f62.jpg)  
Figure 28. Screening preserves nearly all unique papers before deeper analysis. Counts are computed from the selection snapshot used by the export. Shares after the unique-index row use 124,816 as the denominator.

![](images/220c760bb9e6602a83201548b0b677e66b07b40c84f118a1563ffdb1c1ba564f.jpg)  
Figure 29. The source pool is largest in cs.AI and cs.LG before deduplication. These are category-listing records before ID-level deduplication; a single paper can contribute to multiple rows.

Structured Analysis. Papers carried forward from the fast screens go through a deeper analysis with MiniMax

![](images/c8a1276d6405b5984677ba46d0f6f242c6e1935a0205aa5888e5a162e1197462.jpg)  
Figure 30. Fast screens keep broad coverage while identifying metric-method papers as the dominant type. The left side summarizes 123,980 fast quality scores; the right side summarizes 124,018 fast paper-type labels.

M2.7 (MiniMax, 2026), which classifies each paper and records a reproduction-oriented catalog entry. The first and most decisive field is the paper type. We keep only method papers, whose contribution is a method or technique evaluated with measurable results, and set aside the other four types: benchmark or dataset papers, where the contribution is the evaluation rather than a method; analysis or findings papers, which report observations rather than improvements; theoretical papers, whose contribution is a formal result; and survey or position papers, which carry no original experiments. Only method papers fit a reproduce-then-improve task, where an agent has to implement a method and then improve on a measured number.

The remaining fields describe each surviving paper in more detail: contribution novelty, taxonomy, compute and paid-resource requirements, benchmarks, datasets, baselines, concrete dependencies on other papers, code availability, reproduction dificulty, research quality, and problem novelty, with every scalar judgment given as an integer score from 0 to 10. Reproduction dificulty is split into coding, compute, time to run, and data or model dificulty, plus an overall score, because a paper should not be chosen on scientific appeal alone; a useful task also has to be feasible for an autonomous agent under a bounded budget. Figure 31 shows how these scores are distributed across the analyzed papers.

Filtering and Ranking. The final selection balances importance against feasibility. Let � be the 0-10 paper quality score, and let � be the aggregate reproduction dificulty. When all four reproduction subscores are present, we compute

$$
r = { \frac { 5 r _ { \mathrm { o v e r a l l } } + 1 . 2 5 ( r _ { \mathrm { c o d e } } + r _ { \mathrm { c o m p u t e } } + r _ { \mathrm { s p e e d } } + r _ { \mathrm { d a t a } } ) } { 1 0 } } .
$$

It then ranks eligible papers by $u = \sqrt { ( q / 1 0 ) ( 1 - r / 1 0 ) }$ , so a paper must be both scientifically strong and reasonably reproducible to rank highly. Code availability is recorded but is not a hard requirement, which allows to include papers where agents must reconstruct missing pieces. The filtering counts in Figure 32 are sequential counts from the exported snapshot, each paper attributed to its first rejecting rule.

Category Balancing. The feasible pool is still highly uneven across topics, so we do not simply take the top 1,000 utility scores. We first map fine-grained taxonomy labels into 18 canonical categories, then allocate final slots using temperature-smoothed category shares with temperature 1.2. The largest categories remain large, but smaller categories are not squeezed out entirely. Figure 33 shows the candidate pool and selected count for each canonical category. Figure 34 reads the resulting environments in their own words: what each asks to make better, and the work delivering it requires.

![](images/bcaed837499e9588465bf3d7f940ca0bfd1eee68eb25925eaccee0ec01fe60bf.jpg)

Figure 31. Structured-analysis score distributions across the 21,354 analyzed papers. Each panel gives the share of papers at each integer score from 0 to 10: the five reproduction-dificulty sub-scores (coding, compute, time to run, data or model, and overall), and the paper-level quality and problem-novelty scores. Dashed lines mark the means. Dificulty scores concentrate in the low-to-middle range, while quality and problem novelty center higher.  
![](images/426699d57810c780141d0fe38e61518044df550088c391f16914d0b663757013.jpg)  
Figure 32. Feasibility filters reduce 21,354 analyzed papers to 8,987 candidates. The feasible candidates are ranked by quality-feasibility utility before the final category-balanced selection.

## E Evaluation Protocol

Evaluation is the scoring harness that turns the reproduced environments into comparable scores across models. Where Swarm is built for open-ended builder coordination, Evaluation is deliberately flatter: it runs one controlled trial for each combination of paper, model, and mode. Each trial starts from a fresh workspace, receives a bounded compute budget through Nexus, drives one agent session through reason-run-feedback turns, and records the resulting train and held-out test scores. The harness splits this into auditable stages, from scafolding through normalization (Table 12). The unit of comparison is therefore explicit: a model is judged by what its agent can build under the same problem statement, workspace boundary, scoring contract, and compute budget. We call this evaluated agent the scientist agent, to distinguish it from the builders that construct an environment (§A.3.2).

![](images/8d0ab3a554f5fbe7dd1fff693ff2c98e247074d75749ac082bcc66e4975c9eab.jpg)  
Figure 33. Temperature-balanced sampling keeps large areas prominent while preserving smaller categories. The final 1,000-paper selection is sampled from 8,987 feasible candidates across 18 canonical categories.

![](images/3676a8fca7235e18352933cf269405d0272efbd1e0504f02dd504dbb2d54d984.jpg)  
Figure 34. What environments ask, and the work it takes. Each environment states what should get better and what work that takes; phrases are grouped into families that emerge from data; ribbon mass is the share of environments.

## E.1 Evaluation Modes

Each environment can be run in two modes that ask diferent questions of a model. In improve mode, the scientist is given the reproduced method and the train-phase scores it already achieves, and must push those numbers higher without changing the evaluation protocol; the question is whether the model can advance an existing result. In innovate mode, the agent is given only the problem statement, the constraints and targets, the visible evaluator, and the data interface, with an empty method directory, and must build a method from first principles; the question is whether the model can solve the problem on its own.

The two modes are the problem-and-solution separation of §B.2 made operational: the same environment, presented once with the reproduced method and once without it. This separation is built into the workspace layout, not just the prompt. The reproduced method lives in its own method/ directory, while the problem statement, scoring targets, visible evaluator, and data interface sit outside it, and the evaluator and data never import from method/ (§C). Because the rest of the environment does not depend on the method, a single change switches modes: method/ is handed over intact for improve mode and emptied for innovate mode. They are kept separate in both the run matrix and the normalization, because a score earned by inventing a method from scratch is not the same experimental object as one earned by improving an existing method.

## E.2 Running a Trial

Workspace Boundary. Evaluation controls what the scientist can see and modify. The problem briefing, data, baseline artifacts, and visible training evaluator are read-only. The method directory, run scripts, scratch directory, and scoring directory are writable. Held-out test files, test scripts, reference scores, and the paper text are not present during the main phase. This boundary prevents agents from tuning against the hidden test split while still giving them enough information to conduct real research: they can read the task, implement a method, run the visible evaluator, inspect failures, and iterate. In improve mode, the starting method’s train score is copied into the briefing as a target to beat, but it is not used as an oracle during scoring.

<table><tr><td>evaluation stage</td><td>contract enforced by the harness</td></tr><tr><td>scaffold</td><td>create a fresh per-run workspace and remove files that would leak hidden tests or reference scores</td></tr><tr><td>main phase</td><td>let the scientist edit the method and submit compute jobs until it writes train-phase scores and declares completion</td></tr><tr><td>validation</td><td>check that every required proposed-method metric is present; missing cells trigger bounded repair turns</td></tr><tr><td>held-out test</td><td>materialize the hidden evaluator only after the method is frozen, then run it through Nexus</td></tr><tr><td>normalization</td><td>convert heterogeneous metric improvements into per-paper scores before averaging into a leaderboard</td></tr></table>

Table 12. Evaluation separates iteration, validation, hidden testing, and normalization into auditable stages. The staged harness keeps scientist behavior inspectable and comparable.

Scientist Loop. At the end of each turn, the scientist declares whether it wants to submit a compute job or finish. A submitted job runs the workspace’s compute entrypoint through Nexus, using the requested CPU or GPU resources and optionally the short test capacity for smoke tests. Evaluation does not implement its own execution loop; it consumes the job results that Nexus exposes and resumes the same agent session with them. Current runs use a 200-turn ceiling, and a trial that fails for a transient reason before producing any score file is retried up to three times. Once scoring files exist, the workspace is kept for review rather than overwritten.

Train and Test Scores. The main phase produces scores\_train.json using the visible evaluator. A host-side validator then checks the output against the source scoring shape: for every required proposed method, every numeric metric present in the reference reproduction must also be present in the agent’s score file. Metrics marked as baselines are ignored because they are not the scientist’s responsibility.

After train validation passes, the method the scientist built is frozen: its directory is mounted read-only before the held-out evaluator runs, so the test score reflects exactly the method that produced the train score and cannot be tuned against the hidden split. Evaluation then materializes the held-out evaluator from a pristine clone, so the scientist cannot influence what the test runs, and executes against the frozen method.

If the test job fails, the default recovery path keeps the method and the hidden evaluator read-only for five repair turns, so the scientist can repair missing packages or regenerate artifacts without modifying the frozen method or peeking at held-out data. If those turns do not succeed, a final three-turn path makes method/ writable again under a stricter honesty prompt, so a genuine interface mismatch can be repaired; even then the held-out evaluator and test data stay read-only and re-materialized from the pristine clone, so the method can be fixed but the hidden split still cannot be read nor the scorer altered, and these unlock turns are flagged in the trace. A run is complete only when both train and test score files validate.

## E.3 Scoring

Evaluation reduces each model to one comparable number per paper, its FNI, then averages over the ten into a leaderboard. The raw measurements are heterogeneous at three levels. Within a paper, the result is split across experiments and their metrics, which sit on diferent scales, so one in the hundreds must not swamp one in hundredths, and carry diferent importance, so a metric central to the claim should count more than an incidental one. Across papers, the natural improvement ranges from a fraction of a percent to tens of points, yet every paper should contribute equal magnitude rather than let large-swing papers dominate. Across models, the comparison must stay fixed: a later model is scored against numbers frozen at publication rather than re-baselined, with the sign preserved and no value clamped. The scoring function below reconciles these levels, one cell at a time.

Definitions. The scoring is built on the following objects, all fixed before any model is scored.

• Reference cohort $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { K } \}$ : a fixed set of � models, chosen once and frozen at publication. Each $m _ { k }$ is one model, and every model in ℳ is run on every paper.

• Cell �: one metric of one method in one experiment. A single paper has many cells, spread across its experiments and the several metrics each one reports; baseline methods are excluded, leaving only the proposed methods.

• Source value $s _ { c } { \mathrm { : } }$ the value the paper’s own method reaches on cell � once reproduced; it is the reference every model is measured against.

• Model value $x _ { m , c } \mathrm { : }$ the value reached on cell � by the method that model � builds.

• Direction $d _ { c } \in \{ + 1 , - 1 \}$ : +1 when a higher value is better, −1 when lower is better.

• Prior weight $p _ { c } = w _ { e ( c ) } | \kappa _ { c } | :$ how central the cell is to the paper’s claim, set by the builder when it writes the reference scoring file, not inferred from any model’s results. The agent records two numbers there: a weight $w _ { e }$ for each experiment, normalized to sum to one across the paper, and a coeficient $\kappa _ { c }$ for each metric that sets the metrics within an experiment against one another. The prior weight is their product, taking the coeficient’s magnitude so that a metric’s direction (already captured by $d _ { c } )$ does not afect its weight.

The cohort is fixed because it calibrates every paper: adding a model to it would move every paper’s scale and baseline and invalidate all stored coeficients, so a later model is scored against the cohort, never added to it.

Cell Improvement. The signed improvement of model � on cell � is

$$
\delta _ { m , c } = d _ { c } ( x _ { m , c } - s _ { c } ) = { \left\{ \begin{array} { l l } { x _ { m , c } - s _ { c } , } & { { \mathrm { i f ~ h i g h e r ~ i s ~ b e t t e r } } , } \\ { s _ { c } - x _ { m , c } , } & { { \mathrm { i f ~ l o w e r ~ i s ~ b e t t e r } } , } \end{array} \right. }
$$

positive for an improvement over the source and negative for a regression, on every metric. Every scale below is built from these improvements: how widely they spread across the cohort sets the unit for each cell, and the cohort’s typical paper score sets the baseline that makes papers comparable.

Cohort Scale. For each paper, mode, and phase, the cohort fixes the scale that each cell is divided by. The first quantity is the per-cell spread,

$$
\sigma _ { c } = \mathrm { s d } _ { m \in \mathcal { M } } \mathopen { } \mathclose \bgroup \left( \delta _ { m , c } \aftergroup \egroup \right) = \sqrt { \frac { 1 } { K - 1 } \sum _ { m \in \mathcal { M } } \left( \delta _ { m , c } - \bar { \delta } _ { c } \right) ^ { 2 } } , \qquad \bar { \delta } _ { c } = \frac { 1 } { K } \sum _ { m \in \mathcal { M } } \delta _ { m , c } \mathrm { , }
$$

the sample standard deviation of the cohort’s improvements on cell �, where $\bar { \delta } _ { c }$ is the cohort’s mean improvement on that cell. We use $\sigma _ { c }$ as the cell’s unit because it is expressed in the metric’s own scale: a metric reported in the hundreds gives a $\sigma _ { c }$ in the hundreds and one reported in hundredths gives a $\sigma _ { c }$ in hundredths, so dividing an improvement by $\sigma _ { c }$ removes the metric’s scale and lets cells be compared. Some cells have $\sigma _ { c } = 0$ , where the cohort returns the same value and the cell carries no spread of its own; we want neither to drop these cells nor to divide by zero, so we also compute a fallback, the paper’s typical spread,

$$
\tau = \underset { j : \sigma _ { j } > 0 } { \mathrm { m e d i a n } } \ \sigma _ { j } ,
$$

the median of the nonzero per-cell spreads. A median, not a mean, keeps one unusually noisy cell from inflating this typical scale and shrinking every other cell. The per-cell normalizer then uses each cell’s own spread when it has one and � otherwise,

$$
n _ { c } = \left\{ \begin{array} { l l } { \sigma _ { c } , } & { \sigma _ { c } > 0 , } \\ { \tau , } & { \sigma _ { c } = 0 \mathrm { ~ a n d ~ } \tau > 0 , } \end{array} \right.
$$

so a tied cell still contributes, scaled by the paper’s typical spread. If even $\tau = 0 .$ , no cell on the paper carries any spread, the paper has no usable scale, and it is excluded rather than scored with an arbitrary fallback.

Scale from the Cohort, not the Source. For a cell the cohort does not tie on, the normalizer from the previous paragraph is simply the cohort spread, $n _ { c } = \sigma _ { c }$ . Dividing by this spread, rather than by the source value, is what places metrics of diferent scales on a common footing. The more obvious choice would divide each improvement by its source value, $\delta _ { m , c } / | s _ { c } | ;$ but $\sigma _ { c }$ is computed from those same improvements, so dividing them by $\left| s _ { c } \right|$ scales the cell and its spread alike, and the source value cancels:

$$
\frac { \delta _ { m , c } / | s _ { c } | } { \mathrm { s d } _ { m } ( \delta _ { m , c } / | s _ { c } | ) } = \frac { \delta _ { m , c } / | s _ { c } | } { \sigma _ { c } / | s _ { c } | } = \frac { \delta _ { m , c } } { \sigma _ { c } } , \qquad s _ { c } \neq 0 .
$$

The cohort spread, not the source value, carries the scale. Writing the standardized cell score with the general normalizer,

$$
z _ { m , c } = \frac { \delta _ { m , c } } { n _ { c } } ,
$$

and $s _ { c }$ sets only the zero point of the improvement, not its scale. The score also stays well defined when $s _ { c }$ is zero or near zero (for example, when a metric counts the number of bugs in a code snippet), where dividing by the source value would be undefined or blow a tiny gap into an enormous ratio.

Efective Weights. The prior weight $p _ { c }$ records how important a metric is to the paper, but not how much it tells one model apart from another. A cell on which the whole cohort lands at nearly the same value carries little signal about which model is better, while a cell on which the cohort spreads out widely both separates models and is statistically reliable. We therefore scale each prior by the cell’s spread $\sigma _ { c }$ relative to the paper’s typical spread $\tau ,$ and renormalize so the weights sum to one:

$$
\tilde { w } _ { c } = p _ { c } \left( \beta + \frac { \sigma _ { c } } { \sigma _ { c } + \tau } \right) , \qquad w _ { c } = \frac { \tilde { w } _ { c } } { \sum _ { j } \tilde { w } _ { j } } .
$$

Here $\beta$ is a floor we set to 0.5. The weight $w _ { c }$ this produces is each cell’s share of the paper score, so scaling it by the cell’s spread makes the score lean on the cells that actually separate models and discount those that do not. When the cohort ties on a cell $( \sigma _ { c } = 0 )$ , every model scores it the same, so the cell cannot tell a better model from a worse one; counting it at full weight would only dilute the score with a cell that carries no signal, so the factor drops to its floor $\beta = 0 . 5 ,$ , halving its influence without dropping it. When a cell spreads the cohort by the paper’s typical amount $( \sigma _ { c } = \tau )$ , it carries a representative amount of signal and the factor stays at 1.0, leaving its prior unchanged. When the cohort disagrees on a cell far more than usual $( \sigma _ { c } \gg \tau )$ , the cell sharply separates models and is the most informative, so the factor rises toward 1.5 and the cell counts for up to half again its prior. The factor saturates smoothly toward $\beta + 1$ rather than being clipped at a hard cap.

This does a diferent job from the normalizer: $n _ { c }$ puts every cell on a common scale, while $w _ { c }$ sets how much each cell then counts. Together they keep any single cell, an extreme-scale metric or a tied, uninformative one, from deciding a paper’s score by itself. As a final backstop against a near-tied cell whose tiny spread inflates its coeficient, any cell whose contribution both outweighs all of a paper’s other cells combined and exceeds one full paper score in magnitude is pruned, uniformly for every model, so no saturated metric can swing a paper’s score on its own.

Paper Coeficients. The cohort model’s score on a paper is the weighted sum of its standardized cell scores,

$$
R _ { m } = \sum _ { c } w _ { c } z _ { m , c } .
$$

The size of $R _ { m }$ still varies across papers: a paper whose metrics swing widely gives every model a large $R _ { m }$ and would dominate a leaderboard that averaged raw paper scores. To put papers on equal footing, we divide by the cohort’s typical paper-score magnitude on that paper,

$$
b = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { m \in \mathcal { M } } \vert R _ { m } \vert ,
$$

the mean absolute $R _ { m }$ over the cohort. After dividing by $b ,$ the cohort’s typical paper score has magnitude one on every paper, so each paper contributes equally to a model’s average instead of in proportion to how large its improvements happen to be. Weight, normalizer, and baseline then fold into a single stored coeficient per cell,

$$
\alpha _ { c } = \frac { w _ { c } } { n _ { c } b } ,
$$

so a future model is scored by the frozen sum

$$
S = \sum _ { c } \alpha _ { c } \delta _ { c } .
$$

All spread estimates, weight adjustments, and paper baselines are absorbed into $\alpha _ { c }$ and never re-estimated.

Leaderboard. Train and held-out test phases, and innovate and improve modes, keep separate coeficients because they measure diferent experimental objects. Within each published setting, the leaderboard reports the mean paper score over published papers. The scale comes from the reference cohort, and FNI is this score scaled by 100: a paper’s typical score across the cohort has magnitude 100, so a model that matches the cohort scores about 100, a model with about half again as much improvement scores near 150, and a model at half the cohort’s gains scores near 50; the score is positive when the model improves over the source reproduction and negative when it regresses. Because each paper’s spread across the cohort is about one, averaging over many published papers shrinks the noise in a model’s mean score, so models are separated by the average of many per-paper diferences rather than by any single paper; we therefore read only sustained gaps as real and treat very close scores as ties. A paper is excluded from publication uniformly for every model, never per submission, when the cohort gives no usable scale (every cell tied, or a baseline that cancels to zero) or is too small to estimate spread (fewer than three cohort models present).

![](images/d4b4976f2703517277e553156cc80d4b7154b3d60f498c52ecc0d3540f7159db.jpg)  
improvement over the paper (held-out test score)

![](images/6d9853cb8734ce989774e6a5d518c7f35498bd78851ac3784ea02512a061f7e4.jpg)  
improvement over the paper (held-out test score)  
Figure 35. How often scientists beat the paper. Distribution of held-out improvement over the paper’s source method, per mode. Colored bars (right of zero) beat the reproduced method; the grey tail regressed. The percentage is the share of runs that improved.

## F Evaluation Analysis

The main text reads these environments as a leaderboard. Here we examine the evaluation behind it, asking three questions: how often a scientist actually improves on the reproduced method, whether a score reflects a real gain rather than a metric optimized during the run, and whether the number is largely a proxy for how much compute or interaction a run was given. In short: usually, yes, and no.

## F.1 How Often Scientists Beat the Paper

Before asking which scientist is best, it is worth knowing whether any of them do real work. They do. Figure 35 shows the distribution of held-out improvement over the paper’s source method, one panel per mode. Most runs land to the right of zero: 84% beat the reproduced method when improving it, and 74% do when designing a new one from scratch. Innovate carries the heavier left tail, since a fresh design more often fails outright, yet even there the typical attempt still moves the metric forward.

## F.2 Do the Gains Generalize

A score is only meaningful if it reflects a real gain rather than a metric gamed during the run. The held-out split separates the two: a scientist optimizes against a training-phase metric but is scored on a test slice it never sees. Figure 36 plots one against the other. The two track closely (Spearman 0.57 in improve and 0.69 in innovate), so training gains usually survive to the held-out metric, and only a small share of runs fall in the overfit quadrant where a training gain reverses on test (8% in improve and 5% in innovate). The split rarely needs to intervene, but when it does it removes exactly the runs that would otherwise inflate a score.

## F.3 Efort and Outcomes

A natural worry is that a score mostly tracks how much compute or interaction a run spent, rewarding brute force over insight. It does not. Figure 37 plots each run’s within-paper normalized score against the GPU-hours it consumed and the number of turns it took, with the Improve and Innovate modes kept apart. Every trend is flat: the rank correlation stays weak throughout and is slightly negative where it moves at all, at most 0.22 in magnitude. Spending more does not produce a better result; a better method does.

![](images/6175af4847407619ef7d4ab3b619a2b1d4ff2d6125413d82055291a8a6ca4563.jpg)

![](images/903acc50689a8b6d67fdafdadf246f93257e5bacaa398180b76960a686fbfcf4.jpg)  
Figure 36. Training gains carry over to held-out test. Training-phase score against held-out test score, per mode, with the � = � line. Points hug the diagonal; the shaded quadrant holds the few runs that gained on training but regressed on test.

![](images/6d0de83b7c3f323f18ad784f0e649e85bf362882792e7f9602f152684a6e7be7.jpg)

![](images/74483a2d8d9965dd7cb0f54c1a43449f89e7609c045a5277bda46d597879a56f.jpg)

![](images/38d4586344a8f478b8fd446a4f14dcf8d342652c63928bcc8c5e7890f77901d2.jpg)

![](images/db44ae42f6898cf2f5a77c0b978a1532fba037eb43a2688b2d551ab7d20eac6e.jpg)  
Figure 37. Efort versus outcome. Within-paper normalized score against GPU-hours (top row) and agent turns (bottom row), shown separately for the Improve and Innovate modes. Scores are min-max normalized within each (paper, mode); each line is a least-squares fit in log-efort. Every panel is flat $( | \rho | \leq 0 . 2 2 )$ .

## G Experimental Setup

This section records the concrete setup behind our experiments: the paper set the scientists attempt, the compute pool they run on, the agent scafolds and models we compare, and how trials are run and scored.

## G.1 Data

All papers come from arXiv. We work from a global index of 124,816 unique papers in the categories cs.CL, cs.CV, cs.AI, and cs.LG, built from daily listings between January 1, 2025 and March 1, 2026 and exported on March 28, 2026. From this index, the selection procedure of §D chooses a category-balanced set of 1,000 papers to attempt; the per-stage counts are reported there.

## G.2 Compute

’s experiments run on a heterogeneous setup provisioned through Hive and accounted by Nexus. The compute pool contains 16 A100 GPU nodes, 16 H100 GPU nodes, 16 H200 GPU nodes, and 4 CPU nodes (Table 13). The GPU nodes support accelerator-heavy reproduction jobs, while the CPU nodes support CPU-only experiments and non-accelerated setup or validation work.

<table><tr><td>resource</td><td>count role</td><td></td></tr><tr><td>A100 GPU nodes</td><td>16</td><td>general accelerator workloads and pilot experiments</td></tr><tr><td>H100 GPU nodes</td><td>16</td><td>high-throughput training and evaluation jobs</td></tr><tr><td>H200 GPU nodes</td><td>16</td><td>memory-intensive or largest-scale reproduction runs</td></tr><tr><td>CPU nodes</td><td>4</td><td>low-compute, CPU-only setup and validation jobs</td></tr></table>

Table 13. Hive exposes heterogeneous capacity for accelerator-heavy and CPU-only experiments. The counts describe the compute capacity exposed in Hive.

## G.3 Agent Scafolds

Each scientist reaches the workspace through a scafold, the layer that drives a coding-agent command-line tool behind one uniform turn interface (§A.4). Our experiments center on two scafolds: Claude Code, which runs Anthropic models, and Codex, which runs OpenAI models. The same harness also supports other agents, including Gemini, Kimi, and Grok, so a single task can be run across a range of models. We prioritize a provider’s oficial agent scafold when one is released; otherwise we use Claude Code as the default. Because a model and its scafold are evaluated together and not every model runs through the same scafold, the unit being compared is a model together with its scafold, and a scafold diference is part of what the comparison measures rather than something controlled away.

## G.4 Models

We evaluate scientists built on models from seven providers: Anthropic (Fable 5.1, Fable 5, Opus 4.8, and Opus 4.7), OpenAI (GPT 6 Astra, GPT 5.6 Sol, GPT 5.5, and GPT 5.4), Zhipu (GLM 5.2 and GLM 5.1), Moonshot (Kimi K2.7 and Kimi K2.6), MiniMax (MiniMax M3 and MiniMax M2.7), Xiaomi (MiMo V2.5 Pro), and DeepSeek (DeepSeek V4). Anthropic and OpenAI models run through their oficial scafolds; the rest run through Claude Code against an Anthropic-compatible gateway (§G.3). Every model is run at the extra-high (xhigh) reasoning-efort setting, or at its highest available setting if it does not support xhigh, so models are judged at full strength rather than at a throttled setting.

## G.5 Evaluation Trials

Every trial hands the scientist the same resources, so a diference in outcome reflects the model and not the budget behind it. A trial runs under a ceiling of 200 turns and a compute budget of 30 GPU-hours, with at most 2 GPUs held at once, accounted by Nexus (§A.2). A trial that aborts for a transient infrastructure reason, such as a gateway timeout or a compute-pool blip, is retried up to three times before it is recorded as a failure; retries cover infrastructure, never an agent that simply solved the task poorly, so no model is re-run for doing badly. We cap in-flight trials at 30 and split that capacity evenly across the active models, so a fast API does not crowd a slower one out of the queue. Each model is run in both evaluation modes of §E.1. GPU-hours are counted as wall-clock time times the number of GPUs held, without weighting by accelerator class, so a trial scheduled onto faster cards can fit more experimentation into the same nominal budget; we treat this as a known limitation of expressing the budget in raw GPU-hours.

## G.6 Provider Failures

A run spans thousands of unattended turns over hours or days, during which a model provider can rate-limit, throttle, or go down. The two settings absorb these interruptions diferently, because they optimize for diferent things. When building an environment, the goal is to preserve work, so a run that fails resumes from the start of the current workflow iteration, reusing the saved workspace; a provider outage then costs at most the iteration in progress, not the whole reproduction. During evaluation, the goal is fairness, so a trial that fails before producing any score is discarded and restarted from a fresh workspace rather than resumed: no model is scored on a run that was continued mid-trajectory or warm-started from partial state, and every model meets the same clean-room conditions. A failure that occurs after scores are written is set aside for review rather than silently retried.

## G.7 Cohort and Normalization

Running every model on all 1,000 selected papers would be expensive to repeat, so the headline evaluation uses a smaller fixed cohort drawn from the lower-compute environments, chosen so that a full pass stays cheap enough to run across many models and to rerun over time. Scores are normalized against a reference set of models rather than reported in absolute terms (§E.3): the per-paper normalization coeficients are computed once from that reference set and then frozen, so a newly added model is scored against the existing board without shifting the numbers already reported for other models. The reference set is the current generation of each model family; older generations are scored against that frozen reference but kept out of it.

## G.8 Reproducibility

Every trial starts from the same pinned container image and the same workspace template (§C), so the software environment and the starting state are identical across models and stay fixed as the evaluation is rerun over months. Scoring depends only on the numbers a trial records, not on wall-clock time or the hardware it ran on (§B.2), so a trial can be rescored later without rerunning the agent. We retain the full record of each trial, the agent log, the final workspace, and the jobs it submitted, so any reported score can be traced back to what the agent actually did.

## H Dataset and Training Details

This appendix documents the two artifacts behind §4: the released trace corpus (§4.2) and the scientist trained on it (§4.3). It records how the corpus is counted, how the training data and configuration are derived from it, and the held-out protocol behind Figure 15.

## H.1 Trace Corpus

What Is Preserved. Every evaluation attempt is preserved as its own git repository, one per environment, model, and run mode. The repository is the final workspace of §C with its full commit history: the system commits after every turn (§C.4), so the corpus holds the development history of each attempt rather than a cleaned final state. A trace records the research idea and its revisions, the method code as it changed, and the per-experiment logs and scores, tied together by the turn-by-turn trajectory that produced them. Attempts that regress on the source method are kept alongside those that improve on it; a corpus of research being done must include the failures.

Accounting. The totals in §4.2 are computed per attempt and summed. Hours of work is the scientist’s end-to-end running wall-clock time, from the first turn of a trial to its final recorded score; summed over every preserved attempt, across all models, efort tiers, and both modes, it gives the roughly 6,400 hours reported. The per-attempt averages of Table 4 come from the same records.

## H.2 Training Data

Provenance. All training data derives from finished evaluation traces. Each example is rooted in a task specification exactly as a scientist receives it: the problem, the constraints, and the metric to move (§C). The ten held-out papers of §H.4 appear in no training example.

Supervised Set. The SFT set has 1,000 examples. The prompt is the task specification; the target is the attempt’s reasoning inside <think> tags followed by its final idea. Reasoning is cleaned before training: fabricated run results and job identifiers are removed, and empirical claims are reframed as conceptual, with nothing invented or added. The set combines 925 authentic completions with 75 faithful rephrasings of the strongest completions, which round the set to 1,000 without introducing new content.

Reward Set. The GRPO set has 1,500 already-scored ideas, rewarded by measured outcomes only: 777 examples by the held-out test score (weight 1.0), and 723 by generalization, the test score minus the train-test overfit gap (weight 0.5). Rewards are z-normalized within each paper and mode group, so an idea is reinforced or suppressed relative to the other ideas for the same task. We deliberately exclude rewards judged by an LLM: every reward in the set is a measured result.

## H.3 Training Procedure

Setup. The base model is Qwen3-8B in its thinking variant. Both stages train a LoRA adapter of rank 32 through Tinker (Thinking Machines Lab, 2025), and the GRPO stage continues from the SFT checkpoint.

SFT. Cross-entropy on the assistant message only, learning rate 10<sup>−4</sup>, batch size 8, 3 epochs, maximum sequence length 8,192. Training NLL decreases from 2.25 to 2.16 over the three epochs.

Ofline GRPO. The objective is a clipped PPO loss. The SFT policy’s log-probabilities are computed with a no-gradient forward pass and used as the reference, so the clip bounds every update to a trust region around the SFT policy. Learning rate 10<sup>−5</sup>, batch size 16, one epoch, gradient-norm clip 1.0. Training stays bounded: the completion NLL oscillates in a stable band around the SFT baseline and never diverges. This stability is not free. An earlier variant that weighted cross-entropy by the signed advantage collapsed, because tokens with negative advantage are pushed toward unboundedly low log-probability; the clipped objective with an explicit reference policy is what makes ofline training on scored ideas stable.

## H.4 Held-Out Evaluation

Protocol. The gym result of Figure 15 scores ideas, not the idea generator’s own coding. For each of the ten held-out papers, one idea is sampled from each of the three checkpoints, base, SFT, and GRPO, with the same prompt format, temperature 0.7, and top-p 0.8. A single fixed implementer, GPT 5.6 Sol at the xhigh efort tier, then executes every idea in innovate mode under identical conditions: the idea is injected into the briefing with the instruction to implement it faithfully and to optimize within its framing rather than invent a diferent approach. Only the idea’s origin varies across the thirty runs.

Scoring and Isolation. Scoring is identical to the benchmark: the held-out FNI, computed with each paper’s frozen cohort coeficients (§G.7). The ablation is isolated from the leaderboard: it runs in a separate workspace namespace, shifts no normalization, and publishes nothing.

Results. Figure 38 plots each trained idea’s score against the same paper’s base idea score, and Table 14 summarizes the distributions over the ten held-out papers. Both the mean and the median rise monotonically from base to SFT to GRPO. The spread grows as well, so the gain is a shift in the typical outcome rather than a uniform improvement on every paper.

![](images/c5619c9dfb43951483bf2d4841ec6eec32adc81749bc7610ac6ccca4b47e331d.jpg)

<table><tr><td>statistic</td><td>base</td><td>SFT</td><td>GRPO</td></tr><tr><td>median</td><td>41.2</td><td>83.9</td><td>106.3</td></tr><tr><td>mean</td><td>51.5</td><td>62.8</td><td>89.9</td></tr></table>

Figure 38. Trained ideas against the base. Each of the ten held-out papers contributes one SFT and one GRPO dot, plotted against the same paper’s base idea score (displayed scaled by 100). A dot above the dashed diagonal means the trained idea beat the untrained idea on that paper.  
Table 14. Ablation summary statistics. Median, mean, and standard deviation of the held-out score over the ten held-out papers, displayed scaled by 100.

## I Example Produced Environment

This section uses one finished environment as a lens on what the builders actually do, and why. For each choice a builder makes while turning a paper into a reusable task, we ask why it makes that choice and whether the behavior is one the system is built to elicit or the builder’s own judgment. The paper is only background: it is the reproduction of Transformers without Normalization (arXiv:2503.10622), whose Dynamic Tanh (DyT) replaces Layer Normalization with a learnable element-wise operation. The run used MiniMax M2.7, GLM 5.1, and GPT 5.4 across 25 agent sessions over the reproduce, design, package, and verify roles, and about 19 GPU-hours, ending at the core\_claim milestone. The few excerpts below come from the produced workspace; layout follows Figure 25.

Implement the real method, not a lookalike. The reproducer writes the paper’s actual computation rather than a model tuned to emit its published numbers. It does this because the milestone rewards a method that runs: a surrogate fit to the reported accuracy would clear a naive number-matching test but fails ours by construction, and the verifier re-checks that the code is the algorithm itself. This is designed in — the milestone ladder and the verify role reward a runnable method (§A.3).

```python
method/dyt.py
def forward ( s e l f , x): # DyT(x) = gamma * tanh(alpha*x) + beta
r e t u r n s e l f . gamma * torch . tanh ( s e l f . alpha * x) + s e l f . beta
def replace_ln_with_dyt (model , init_alpha =0.5) :
fo r name , m i n list ( model . named_modules ()):
i f isinstance (m, (nn. LayerNorm , nn. RMSNorm )): # swap every norm layer
set_submodule (model , name , DyT(m. normalized_shape [ -1] , init_alpha ))
r e t u r n model
```

Write the briefing so it cannot leak the answer. The builder states the problem in method-agnostic terms: it fixes the question and the success criteria but never names DyT or normalization-free training. Why withhold what it just implemented? Because the same workspace is later handed to a from-scratch scientist, and a brief that named the solution would let that scientist copy it instead of doing research. By design, briefings must be method-agnostic, and the verifier and validate.py flag leaks; one is caught below.

```prolog
briefing/problem.md
## Research Question
Can deep transformer - style classifiers be trained stably and accurately when
their usual feature - normalization components are removed or replaced , without
winning by increasing model size , changing the data , or using pretrained weights ?
## Success C r i t e r i a
- Train without numerical divergence ; match or improve the reference accuracy .
- Keep parameter count and model scale matched within the comparison .
- Same data split and preprocessing ; no pretrained weights , extra data , or ensembles .
```

Decide what counts as an improvement. The scoring contract is where the builder exercises judgment the system asks for but does not dictate. It makes top-1 accuracy the primary metric, keeps top-5 and balanced accuracy as diagnostics, and gives parameter count a negative coeficient — so beating the baseline by scaling the model up does not register as a win. The format invites such a constraint metric; which metrics matter, and by how much, is the builder’s call. It records the paper’s reported numbers as the ranking reference and its own measured numbers as the baseline a future method must beat.

Scale the experiment down without trivializing it. The full paper trains ViT-B/L for 300 epochs, beyond a per-paper budget, so the builder scales to a 50-class slice and a ViT-S/16. The choice is bounded by a rule the system cares about: model and data scale may shrink, but the algorithm and the matched comparison mus be preserved, and the task must stay hard enough to discriminate. The builder checks this — the LayerNorm baseline lands below saturation, so comparison still has room, and DyT edges it at equal parameter count.

Pin the surface a future scientist competes on. Because the environment is graded by comparison, the builder writes down what a future scientist may optimize, what it must hold fixed, and where it may explore. It splits this into three documents: a target (what improvement means and how it is measured), constraints (what must stay matched across compared systems), and a direction (the mechanism to work on, and what is out of bounds). These are required and method-safe: a from-scratch reader sees the rules, not the solution.

scoring/TARGETS.md   
## Primary : s u p e r v i s e d \_ c l a s s i f i c a t i o n   
Improve accuracy on the matched ImageNet class slice without increasing model   
scale or changing the data .   
- Benefit : top1\_accuracy ( top5\_accuracy , balanced\_accuracy as diagnostics )   
- Constraint : param\_count\_M within 5% of the matched reference   
- Success : top -1 within 1 pt of the reference , with no parameter increase

scoring/CONSTRAINTS.md   
## Model Scale   
Candidate and reference must have matched scale : param\_count\_M within 5%.   
A larger candidate is an extra experiment , not an improvement .   
## T r a i n i n g Budget   
Same data , epoch budget , optimizer , augmentation , and preprocessing . A recipe   
change counts only when it is applied to every system in the comparison .

scoring/DIRECTION.md   
## D i r e c t i o n Scope   
Work on normalization as a mechanism : stabilize activations , gradients , or   
optimization while keeping the compared architecture and data fixed .   
## Out o f Bounds   
- Resource scaling as the only change - Pretrained weights or extra data   
- Ensembles - Switching to an easier or different dataset

Keep evaluation independent of any one method. The builder makes the evaluator score prediction files and own the labels, rather than load the reproduced model. Why give up the convenience of calling its own checkpoint? Because a future method, however diferent from DyT, must score through the same interface; an evaluator coupled to this run’s model would break for everyone else. The boundary is designed — and, as the next step shows, enforced rather than assumed.

Build a held-out test, and say where it is weak. The designer adds a hidden test slice so a later scientist cannot quietly overfit the visible scores, and — notably — writes down why it chose this split and where it falls short. The choice of split is the builder’s own judgment; that it records the trade-of is the behavior the design-notes step is meant to produce.

DESIGN\_NOTES.md   
### S p l i t s t y l e : per−c l a s s a l t e r n a t i n g v a l i d a t i o n p a r i t y   
Within each class , even - position images go to train and odd - position to test .   
Why this fits : both slices have comparable difficulty , so an honest method   
scores similarly on both , while a method that games the visible slice shows a gap .   
Known limitation : the parity split is moderate . A harder split (held - out classes

or OOD ) gives a stronger shortcut signal but needs a larger dataset .

Catch the system’s own mistakes. The last role is where design meets behavior. The verifier audited the artifact and found that an earlier builder had let the evaluator load the reproduced checkpoint — a method leak that would silently break from-scratch mode. It moved inference into method/ and rewrote eval/ as a generic prediction-file scorer. The system is built to catch exactly this class of error (the verify role together with validate.py), and here it did.

VERIFICATION.md   
## I s s u e s found   
Critical , fixed : eval / contained solution - specific checkpoint loading . Even   
without importing from method /, this leaked the reproduced algorithm into the   
reusable evaluation path and would compromise from - scratch mode .   
## F i x e s a p p l i e d   
- Moved checkpoint inference into method /; rewrote eval / as a generic evaluator   
that scores JSONL predictions from any method without knowing the algorithm .

Only after this check passed did the workspace settle at core\_claim. Every choice above is committed: the run writes a manifest.json and a README.md reproduction log, and the git history checkpoints each turn, so each decision traces back to the builder, the jobs, and the code behind it (§C).

## J Agent Prompts

This section shows representative prompts from each stage of the system, to illustrate how agents are instructed: the screening and analysis prompts that grade and select papers, the role prompts that drive a Swarm reproduction, and the prompts that run a single evaluation trial. The prompts are excerpted; operational scafolding repeated across prompts (job submission, progress logging, version control, the workspace layout) is omitted and described in sections above. Placeholders in braces are filled at run time.

## J.1 Prompts for Paper Grading and Selection

## Prefilter

Paper Pre-Filter

Read the paper excerpt below and rate its research quality on a scale of 0-100.

• 100: Landmark paper. Defines a new field or sets a new paradigm. Must-read for anyone in ML/AI.

• 80: Top-venue quality (ICLR, NeurIPS, ICML, ACL, EMNLP, CVPR). Novel idea, strong experiments, clear results. Worth reading and citing.

• 60: Second-tier venue quality. Solid work with experiments, but less novelty or narrower scope. Worth reading in your area.

• 40: Has a method and some experiments, but limited novelty or weak evaluation. You’d skim it.

• 20: Minimal contribution. Shallow, incremental, or poorly executed. Not worth reading.

• 0: Not a research paper, or no method and no experiments.

These anchors are reference points, not buckets. Use the full range freely. Scores like 7, 23, 41, 58, 73, 84, and 97 are all valid when appropriate. Do not restrict yourself to only 0, 20, 40, 60, 80, or 100.

Note: The excerpt below is the first \~15,000 characters of the paper, converted to markdown, intentionally truncated. Formatting artifacts (broken text, missing figures, odd symbols) are from automatic PDF-to-markdown conversion — ignore them and judge only the research content.

Think briefly about the paper’s contribution, novelty, and experimental rigor. Give a short explanation of the main strengths and weaknesses. Then provide an overall score from 0 to 100, using any integer in that range. The score should reflect the paper’s actual research quality, not just which anchor it is closest to.

Paper Excerpt

## Analyst

## PaperScope — Paper Cataloging Prompt

You are a research paper analyst. Given a paper, produce a structured catalog entry of 13 fields whose purpose is to assess whether and how hard the paper can be reproduced. The fields cover paper type, novelty, taxonomy, compute requirements, paid resources, benchmarks, datasets, baselines, dependencies, code availability, reproduction dificulty, paper quality, and problem novelty. Every score is an integer on a 0–10 scale, grounded in specific paper content; when uncertain, say so rather than guess.

Field 1: Paper Type

Classify into one primary type, with justification:

• metric\_method — introduces or improves a method, evaluated with measurable results.

• benchmark\_dataset — the contribution is a benchmark, dataset, or evaluation suite.

• analysis\_findings — an empirical discovery or insight rather than an improvement.

• theoretical — a formal result: proofs, bounds, complexity analysis.

• survey\_position — aggregates or opines on existing work; no original experiments.

[ · · · ]

Scoring: Reproduction Dificulty, Paper Quality, Problem Novelty

The decisive fields. Reproduction dificulty — split into coding, compute, runtime, and data sub-scores plus an overall — estimates the cost to implement the method, run it, and evaluate the main claims, judged as the AI agent that would reproduce it rather than a human. Paper quality rates the contribution as a reviewer would. Problem novelty rates how interesting the question is, independent of the method. Each is a 0–10 integer with a short justification.

[ · · · ]

## J.2 Prompts for Swarm

## Reproduce Start

Reproducer Agent

You are a Reproducer agent. Your job is to read a research paper, reproduce its core results, and package the reproduction as a reusable environment.

Your reproduction does not need to match the paper’s exact numbers, but you must implement the paper’s actual method and demonstrate that it works as claimed. Fitting surrogates to published data, hardcoding paper values, or building shortcuts that bypass the method are not reproduction — they are fabrication.

You are fully autonomous. There is no human in the loop. Do not ask for confirmation, do not wait for direction, do not say "let me know" or "please confirm". Analyze the situation, make your own decisions, and act immediately. Every turn you spend not making concrete progress (editing code, creating scripts, fixing bugs) is wasted.

You should launch multi- or sub-agents to accelerate the implementation.

Your minimum goal is core\_claim. Do not stop before reaching it unless reproduction is genuinely impossible. You may receive a newly initialized workspace, or one that already has partial or substantial implementations. In either case, your job is to push the reproduction forward — not just audit what exists.

## [ · · · ]

## Workflow

Work through these phases. You decide when to move between them — they are not hard boundaries.

Phase A: Paper Analysis (no GPU needed)

1. Read the paper thoroughly (both markdown and PDF)

2. Identify: dataset, method, evaluation metric, baseline numbers, hyperparameters

3. Check if the paper links to a code repository

4. Check if the paper has a code release — if so, clone it into method/ and study it

5. Plan your implementation — whether building on existing code or writing from scratch, the paper’s method goes in method/

6. Write an initial plan as comments or a briefREADME

7. Fill in the briefing/ folder — problem.md and evaluation.md first (method-agnostic), then method.md, and fi nally overview.md (short summary)

8. Fill in scoring/reference.json with the paper’s reported numbers — experiments, metrics, baselines, and the proposed method (see Deliverables for format)

## Phase B: Environment Setup (light compute)

1. Acquire the dataset: check the shared datasets directory first, download if missing (use shared for general datasets, workspace data/ for paper-specific ones)

2. Download pretrained models if needed (use shared models directory for general weights, workspace for paper-specific ones)

3. Write data loading and preprocessing code

4. Write the evaluation script (compute the paper’s metric)

5. Validate scoring/reference.json by running python validate.py

6. Verify on a trivial baseline (e.g., random predictions) that the eval pipeline works

7. Start drafting the container definition and task description for the environment

## Phase C: Reproduction (GPU, iterative)

Plan the reproduction incrementally — start from the minimum, expand if budget allows:

1. Plan first. Break the reproduction into ordered steps from minimum to complete.

2. Reproduce the minimum that supports the core claim. A single experiment on one setting — just enough to show the method works.

3. Expand incrementally. Each step adds more coverage — more settings, more comparisons, more ofthe paper’s results. But only expand after the minimum is confirmed working — validating that the idea works matters far more than matching the paper’s full scale.

4. Mark progress. Report milestones (see below) after each turn.

When expanding, prefer adding a new axis of variation over running more of the same single-setting experiment. The relevant axes depend on the paper: dataset, problem instance, input dificulty, environment, backbone, hyperparameter regime, task type — anything the paper itself varies. A small matrix across two axes is far more informative than two runs at the same condition, and it gives downstream agents a real surface to improve on.

[ · · · ]

## On Designer

Designer — Build the Train/Test Evaluation

You are the designer. Previous agents reproduced a paper and produced scoring/scores.json. Your job is to design a twopart evaluation — a train slice and a held-out test slice — that future agents will be tested against.

You might be resuming from a previous designer’s work, or a repo not yet started on this part.

Training and test should be roughly equal in scale.

What this workspace is becoming

This workspace will ship as a reusable benchmark and gym environment. Future AI scientist agents will operate on it in two modes:

• Improve mode: the agent gets everything (problem + method + code + results + data + documentation) and must beat the current scores.

• From-scratch mode: the agent gets only the problem definition (problem + eval + data + constraints + direction) and must invent their own method.

In both modes, the agent optimizes against an evaluation suite (what in eval/). If that suite is a single metric on a single dataset, agents will find shortcuts — tune one number, ignore the rest, hit the target without improving the underlying idea. We want to measure real improvement, not metric-hacking.

Your job is to make that harder (or at least detectable) by creating two evaluation slices:

• eval/train/ — the slice future agents see during development. They optimize against this.

• eval/test/ — held out from future agents. They never see it. When they signal "done", we run eval/test/ on their final method. A big gap between train and test is the signal we use to detect overfitting.

This is a classic train/test split, applied at the evaluation-suite level to catch agent shortcuts. You are evaluating future agents work, not the paper’s claim. You are not trying to break the paper. You are building the infrastructure that distinguishes "real improvement" from "metric gaming" for everyone who follows.

## What to produce

Six new files / directories total — two new eval subfolders, two new score files, two new shell entry-points (existing files are read only, except reference.json, which you may extend with new metric entries only):

eval/   
|-- <existing eval files DO NOT MODIFY>   
|-- train/ <- new, self-contained, what future agents see   
‘-- test/ <- new, self-contained, held out from future agents   
scoring/   
|-- reference.json <- existing; only ADD new metric entries (never modify existing)   
|-- scores.json <- existing. DO NOT MODIFY   
|-- scores\_train.json <- new, produced by scripts/evaluate\_train.sh   
‘-- scores\_test.json <- new, produced by scripts/evaluate\_test.sh   
scripts/   
|-- evaluate.sh <- existing, DO NOT MODIFY   
|-- evaluate\_train.sh <- new, mirrors evaluate.sh’s contract   
‘-- evaluate\_test.sh <- new, mirrors evaluate.sh’s contract

## THESE SCORES MUST BE PRODUCED BY A REAL RUN. YOU CANNOT GUESS OR HARDCODE ANY NUMBERS IN THEM.

Format follows the existing scoring/scores.json and scoring/reference.json conventions. Each metric entry should have at least a value: field. If reference.json uses coefficient to weight metrics, carry those coeficients into both new files consistently so the aggregates are comparable.

Plus two new shell entry-points alongside the existing scripts/evaluate.sh (do NOT modify evaluate.sh):

• scripts/evaluate\_train.sh — runs eval/train/ on a method’s pre-generated outputs and writes scoring/scores\_train.json

• scripts/evaluate\_test.sh — runs eval/test/ on the same outputs and writes scoring/scores\_test.json

These follow the same contract as the existing scripts/evaluate.sh — method-agnostic, reusable, callable standalone, reading pre-generated method outputs and writing scores. Mirror evaluate.sh’s argument shape and input/output conventions exactly — a future agent should be able to swap in a diferent method and run scripts/evaluate\_train.sh / scripts/evaluate\_test.sh without code changes on their side.

evaluate\_test.sh must score every method present in scores\_train.json (never one hardcoded name), write to scores\_test.json, and invoke the eval/test/ path. A copy of evaluate\_train.sh that still points at eval/train/ silently destroys the anti-shortcut signal.

[ · · · ]   
[ · · · ]

## Design principles

Metrics measure the scientific quantity, NEVER agreement with the paper

This is the single most important rule. A metric that measures "how close is the reproduced value to the paper’s published value" — e.g., mean\_abs\_rel\_error\_author\_X, deviation\_from\_paper\_X, |value - 0.87|, fit\_to\_published\_Y — is categorically forbidden as a scored metric. The gym is built FROM the paper, but we are never trying to match the paper.

The paper is context, not a target. It tells you what problem to measure and what metrics are scientifically meaningful for that problem. It does NOT tell you what numbers the score should reach. Future agents may produce a method that scores higher than the paper (real improvement), lower than the paper (regression), or that invents an entirely diferent approach — every one of those outcomes must be visible in the score.

If your scoring metric has a FLOOR at "exactly reproduce the paper," the gym is broken by construction: the paper’s own method already sits at the floor and no agent can go below it. You’ve turned an improvement benchmark into a reproduction-fidelity meter.

## What TO score:

• The scientific quantity itself (accuracy, loss, F1, calibration error, similarity, detection rate, reward, etc.), computed on the paper’s evaluation protocol. Avoid wall-clock-dependent quantities such as latency or runtime; use a machine-independent substitute (FLOPs, parameter count, training steps, samples processed) instead, since the verifier rejects scored wall-clock speed metrics.

• Continuous metrics whose range spans both directions of the paper’s claim — an agent that beats the paper’s reported X=0.87 with X=0.95 must score higher; an agent that scores X=0.80 must score lower.

• Ratios, directional diferences, or any quantity where "the paper’s number" is simply one point on a continuum that future methods can move along.

## What NOT to score:

• |reproduced - paper\_published| (absolute error vs paper)

• mean\_abs\_rel\_error\_author\_\* or any metric whose name includes author, paper, published, vs\_paper, deviation\_from\_paper, reproduction\_error, or equivalent wording.

• Any metric whose theoretical floor equals "the paper’s published value." If matching the paper gives 0 and lower-is-better, you have built the forbidden thing — even ifthe metric name hides it.

The paper’s numbers belong in scoring/reference.json under results.<method\_name> as the baseline against which future agents are ranked. They do NOT belong inside metric names, metric formulas, or as "target" values the score chases.

Anti-shortcut, not anti-paper

A well-reproduced method should score similarly on train and test. If the split works, an honest agent’s scores look close; a shortcutting agent’s don’t. Design the test slice so a legitimate method (one that actually captures the paper’s idea) still does well, while a method optimized narrowly against train predictably regresses.

Do not hardcode which method to evaluate. The test script decides what to evaluate by reading the agent’s outputs, not by a name baked into the script or a default argument. If your evaluator only evaluates one name, a future agent’s improvements will never reach it and every improve-mode run looks identical on test — silently.

Train and test must be calibrated to comparable dificulty

The split only works if train and test probe the method at roughly the same level of rigor. If test is drastically harder than train by construction, every method will score worse on test — that gap comes from your design, not from overfitting, and it tells you nothing. If test is drastically easier, everyone ceilings out and there’s nothing to compare.

The calibration target: on the existing final reproduction, train and test should produce scores that are meaningfully comparable side-by-side. A small, expected gap is fine; a huge collapse means your split is miscalibrated, not that the reproduction is bad.

When using a split style that intrinsically shifts dificulty (dificulty stratification, OOD/perturbed, held-out subgroup), keep the shift moderate — enough to surface methods that don’t generalize, not so much that even a legitimate method breaks. Sanitycheck by running the existing final checkpoint on both slices before you declare the split final: if train scores are strong and test scores collapse, back of and re-design.

Keep data scale, evaluation protocol (decoding config, normalization, aggregation), and sample counts roughly matched between the two slices so the two reports are apples-to-apples.

## Self-contained (critical)

Core idea: once a future agent has optimized against eval/train/ and signaled done, running eval/test/ on their method must be a zero-efort step. No retraining. No re-running expensive experiments. No code changes on the agent’s side. Just point eval/test/ at the method (and result) they already produced, through the same interface eval/train/ used, and score. As if we got a score by running the train set.

evaluate\_test.sh should be lightweight given that the method has already been run for evaluate\_train.sh — prefer to slice or re-score existing artifacts rather than re-executing the method. If you do need to call the method again (e.g. for a stochasticseed split), make sure the extra cost is small.

This means eval/test/ has to go through the same pre-agreed method interface that eval/train/ uses — the contract that already exists here between method/ and eval/ (a function signature, a CLI like scripts/method.sh, a predictions file at a fixed path, etc.). The agent writes their method to satisfy that interface using eval/train/ as the visible spec. eval/test/ then calls the same interface and scores — nothing else required from the agent.

If eval/test/ couples to anything specific to the current method/ (an internal class, a particular checkpoint format, a file that only this training run happens to write), a future diferent method won’t satisfy it — and the test breaks through no fault of the agent.

The set of keys under experiments.<exp>.results (or the equivalent method-name axis) must match between scores\_train.json and scores\_test.json. When an agent adds a new method variant on train, the test script must automatically score it on test — not by re-running evaluate\_test.sh with a diferent name, but because the script discovers and iterates over whatever methods it finds. The method-name contract is enforced by the script’s enumeration logic, not by a caller remembering to pass the right argument.

No leakage between train and test

eval/train/ and eval/test/ must be independent. eval/train/ must not import from eval/test/, and vice versa. Do not put shared helpers inside either folder that the other reads — if you need shared code, duplicate it (the two folders are small) or keep it outside both. A future agent reading eval/train/ to optimize must not be able to learn the test set’s metrics, data split, or scoring details by following an import.

scoring/reference.json is fine — it may document both slices; that’s expected. Top-level eval/ files (outside train/ and test/) should preferably not reveal test-specific metric/data shape either — ifyou can avoid it, do.

Do NOT retrain. Do NOT modify the method.

The method is already built by prior agents. You are only re-running inference + scoring on the existing final checkpoint. If there is no usable checkpoint or scoring/scores.json is missing, run the minimal inference path to produce scores — but do not retrain, change hyperparameters, or tweak the method.

You are encouraged to use as little computing resource as possible. In most cases, you do not need to rerun massive experiments. For example, the previous result might be already saved, and you only need to re-evaluate some rubric-based scores or the model is already trained, and you only need to do some lightweight evaluation.

[ · · · ]

## On Packager

Packager — Make This Repo Reproducible From a Fresh Clone

You are the packager for this paper reproduction. Previous agents built the method, ran experiments, and produced scores. Your job is a targeted audit: ensure anyone can reproduce this work by cloning the repo and running scripts/download.sh followed by the method.

You do NOT redesign the method. You do NOT change algorithms, training loops, loss functions, eval metrics, or scoring code.   
You fix distribution issues so a fresh clone actually works.

This is a focused task: you check the paths and the download command lines so the environment rebuilds from a clean clone. Do not read files you do not need, to avoid wasting time.

You may use the command line for light checks — ls, grep, head, curl -I (HEAD request only), huggingface-cli download --include config.json, and similar. Never run heavy commands. Never actually download datasets or model weights. You are a checker, not a fetcher. A HEAD request that returns 200 is enough to validate a URL — pulling the full file is unnecessary. If something you run takes more than a few seconds or produces more than a few KB of output, you are doing the wrong thing.

## What to check

1. Every needed pre-existing artifact has a REAL, EXECUTABLE fetch command in scripts/download.sh. Walk the repo. For every path the code reads from that lives inside a .gitignore’d directory (e.g. data/downloads/, pretrained-model dirs), download.sh must include an actual shell command (not a comment, not an echo message) that fetches it from a public source. Few times exceptions occurs, like it uses synthetic data, where other part of the repo handles it.

Only artifacts that must exist before the method runs are your responsibility: datasets, pretrained base models, paper-released public weights, config bundles. Training outputs are NOT your responsibility: files under checkpoints/ (or similar) that this paper’s own training scripts produce are regenerated when a fresh clone runs the method. Do not attempt to fetch agent-trained checkpoints — they don’t exist on a public host. Skip checkpoints/ entirely when auditing unless the paper explicitly ships a pretrained checkpoint to a public URL.

Patterns that are BROKEN and you must fix:

• A comment describing the fetch instead of running it: # curl https://... dataset.tar.gz, # wget -nc https://example.com/dataset.tar.gz, # huggingface-cli download Qwen/Qwen3-1.7B .... A # at the start of the line means the shell never runs it. This is unacceptable, since the download script in the future will run automatically. No comment will be accepted. Ifdetected, you need to fix it.

• Placeholder URLs: https://example.com/..., https://TODO, https://placeholder, or any domain that is obviously a stub. Replace with the real public URL for this paper’s actual dataset/model.

• "Already present" skips: echo "dataset already in shared/...", if [ -d shared/... ]; then echo present; fi, or any branch that assumes a file is pre-placed on this specific machine and omits the fetch. A fresh clone on a new node has no shared/ and no pre-placed files — write the fetch command so it works from empty.

• Missing fetches entirely for a needed path. If method/ loads data/downloads/foo but download.sh doesn’t mention foo, add it.

IN SUCH CASE, YOU MUST REPLACE IT WITH A REAL, EXECUTABLE COMMAND.

It is fine to wrap a fetch in if [ ! -f <path> ]; then <fetch>; fi so running download.sh twice doesn’t redownload what already exists — as long as the fetch inside is a real command, not a comment.

2. scripts/download.sh fetches only from public sources. No cp /home/user/shared/..., no local mount paths, no shared/ references of any kind in this file. Allowed tools: curl, wget, git clone, huggingface-cli download, aria2c, kaggle, gsutil, aws s3. If a prior agent wrote cp shared/... or huggingface-cli download ... --local-dir shared/..., replace it: fetch from the public origin and place the file under data/downloads/ (or the path the method reads from).

3. URLs resolve. Spot-check with curl -I <url> — do not download gigabytes. For HF models, a small huggingface-cli download <repo> --include config.json confirms the repo exists. Ifa URL is 404, a domain is unreachable, or a resource was removed, find the correct public source and update.

4. No shared/ references in committed code (outside download.sh). Search method/, eval/, baseline/, scripts/, configs for shared/(datasets|models|hf\_cache). Replace any match with the corresponding data/downloads/<name> path. (validate.py also enforces this.)

5. Paths line up. If download.sh fetches to data/downloads/foo/ but the method reads from data/foo/ (or worse, shared/datasets/foo/), fix the mismatch — prefer changing path strings in method/ over contorting download.sh into a weird destination

6. Writable-paths rule for new files.

• Public fetched data (datasets, pretrained weights downloaded by download.sh) -> data/downloads/<name>/. Gitignored. Repopulated by download.sh on fresh clone.

• First-party Python modules and small data files the code imports -> directly under data/ (e.g. data/foo.py, data/instances.json). Tracked in git so fresh clones can import them.

• Never use /home/user/shared/... — it doesn’t exist on a fresh clone and is banned by validate.py.

• If you find a blanket data/ line in .gitignore, remove it. Only specific subdirectories like data/downloads/ should be ignored, never the whole data/ directory — the paper’s first-party modules live there.

## On Verify

## Final Verification

You are the final step in a multi-agent workflow. The above prompt is given to previous agents (some still apply to you). Previous agents have reproduced a paper and iterated on the results. Your job is a quality check — make sure the workspace is clean, correct, and ready to serve as a research gym for future AI scientist agents.

The previous agent reached milestone {previous\_milestone}. You start at that milestone — do NOT reset it unless you find genuine issues.

Any improvement to the markdown files are encouraged; making them clear, precise, and complete is critical for a successful gym.

[ · · · ]

## Method-Agnostic Files Are Critical

Future scientist agents must be able to understand the problem and evaluation without being told the paper’s method. Ifwe leak the method, the environment is not useful

Treat the following files as strictly method-agnostic:

• briefing/problem.md

• briefing/evaluation.md

• scoring/EXPERIMENTS.md

## Treat the following files as high-level and method-safe:

• scoring/TARGETS.md

• scoring/CONSTRAINTS.md

• scoring/DIRECTION.md

These files must NOT leak ANY architecture, algorithm, module design, training recipe, loss design, or implementation tricks from the paper. They should describe:

• what problem is being solved

• what each experiment is trying to establish

• what must stay fixed

• what research direction is allowed

If these files leak the paper’s method, the gym is compromised for from-scratch scientist agents. Treat this as a real issue, not a cosmetic one. Fix ifneeded.

## Verification Checklist

Go through these in order. Fix issues you find, but do not rewrite working code or waste compute re-running experiments that already succeeded.

## 1. Folder Structure and correctness

• Clean root and self-contained data/ — no stray scripts or data in the root, and a future agent can load the data without importing from method/.

• eval/ is independent of method/ — a from-scratch method must be able to use eval/ and scripts/evaluate.sh without touching method/.

• briefing/problem.md and evaluation.md are method-agnostic — they state what to solve and how it is scored, never the paper’s approach.

• scripts/download.sh is self-contained and shared/-free — a real, verified fetch for every external resource, runnable from an empty clone.

[ · · · ]

## 2. Scoring Accuracy

![](images/28b00bee7a952795d1454244db29f8ff347bae6264024c96a0cba5b78d6fa118.jpg)

Reject any scored wall-clock-speed metric. Hardware varies across runs (diferent GPU models), so identical code can clock 2–5x faster or slower with no algorithmic change. Any metric whose value depends on wall-clock time / throughput / runtime is unreliable here. The fix is a machine-independent substitute: FLOPs, parameter count, training-step count, or samples processed.

• Does scoring/reference.json exist and pass python validate.py?

• Do experiment names in scoring/scores.json match those in scoring/reference.json?

• Does scoring/EXPERIMENTS.md match the experiment names in scoring/scores.json, and does it describe each experiment at a high level without leaking method details?

• Are the coefficient values in reference.json reasonable? Positive for benefits, negative for constraints. Do the magnitudes reflect how much the paper emphasizes each metric relative to others (not absolute value, but think of if the scientist achieves a better score, is the coeficient multiple that relative diferent an appropriate scale)?

• Are experiment weight values reasonable? Does the primary experiment get the most weight?

• Are there constraint metrics where the paper has trade-ofs? If the paper claims "X without sacrificing Y," is Y measured?

• reference.json values should match what the paper reports, OR carry a notes field explaining a deliberate change. If a value in reference.json is noticeably lower / diferent from the paper’s published number and there’s no explanation (e.g., "notes": "scaled to 10% data; paper reports 0.97 on full"), that’s suspicious — the prior agent may have edited the reference downward to mask a broken reproduction rather than fix it. Spotcheck by skimming the paper’s results table for the headline numbers and comparing. Flag undocumented drift as a finding; ask the next iteration to either restore the paper’s value (and re-run the method) or document the scaling choice properly.

## 3. Gym Scoring Review

Think of yourself as a judge designing the evaluation rubric for this gym. Future AI scientist agents will compete on this workspace — you are deciding how their work gets scored.

The gym works as follows: scientists are scored ONLY on experiments that appear in scores.json. The values in scores.json serve as the baseline. The weights from reference.json are renormalized over just those experiments. For example, two AI scientists each propose their own method; we compare the weighted relative changes in scores among them, ofwhat they produced in scores.json.

Review with this in mind:

• Are the renormalized weight proportions reasonable? Scientists are scored only on the experiments in scores.json, with reference.json weights renormalized over them — check that ratio reflects what matters for this paper.

• Are the coeficient scales appropriate? A small, easy gain on one metric should not dominate a hard-won gain on the primary metric; scale by how easily each metric moves, not by its absolute value.

• CRITICAL: no scored metric may measure deviation from the paper’s published values. The paper is context, never an optimization target; a metric whose floor is “exactly match the paper” leaves the paper’s own method at the ceiling and breaks the gym by construction.

## 4. Completeness

• Are scoring/TARGETS.md, scoring/EXPERIMENTS.md, scoring/CONSTRAINTS.md, scoring/DIRECTION.md filled in and reasonable (not just template comments)? These documents guide future scientist agents — vague or generic language wastes their context window. EXPERIMENTS.md must stay method-agnostic and correspond exactly to the ex periment structure in scores.json. Ifyou see room for improvement, consider refining them.

• Does scripts/evaluate.sh work standalone?

• Does scripts/reproduce.sh exist and describe the end-to-end flow?

• Is environment/container.def + environment/setup.sh reasonable?

• Is MILESTONES.md filled in with an accurate progress log and stop justification?

## 6. Evaluation Dificulty

Agents often scale down or substitute datasets to work around resource constraints. This is acceptable when necessary, but it can silently make the problem trivial — destroying the evaluation’s ability to distinguish methods. A workspace where every approach achieves near-perfect scores is useless as a benchmark, regardless of how well the code is written. An evaluation reduced from the scale of thousands to tens usually could not show its judgement.

Critical: Experiment procedure must match the paper. Scoping down the model size, dataset size, or epoch count is acceptable. Changing the training method is not. If the paper trains from scratch and the agent fine-tunes from pretrained weights (or vice versa), or if the agent swaps the optimization regime (supervised vs self-supervised, RLHF vs SFT, etc.), the experiment cannot demonstrate the paper’s claim — it produces numbers that look reasonable but are structurally invalid. Read briefing/method.md and the agent’s scripts/reproduce.sh / training code; if the training protocol difers from the paper, flag it and downgrade the milestone accordingly.

• Compare reproduced scores against the paper’s numbers. Ifthe reproduction scores are dramatically higher than the paper’s, the agent likely used a simplified setup that does not preserve the original dificulty. This is a critical issue.

• Check whether the agent used the paper’s original data or a substitute. If a substitute, verify it preserves the problem’s dificulty. Look at baseline performance — if a simple baseline with no special method already achieves near-perfect results, the substitute is inadequate

• Check ablation results for sanity. If removing the paper’s key module has no efect on scores, either the ablation is broken or the evaluation is too easy to detect the diference.

• Do the numbers in scores.json look plausible for this domain? If any reproduced score is more than 10x of from the corresponding value in reference.json, flag it as a likely implementation bug — not just an imperfect reproduction.

## What to Do

• If everything looks good: update MILESTONES.md if needed, commit, and signal done. Do not make changes for the sake of making changes.

• If you find issues: fix them. Prioritize scoring accuracy and folder structure — these directly afect whether the gym is usable. You may submit a GPU job ifa quick re-evaluation is needed, but avoid long training runs.

• If the reproduction is completely broken — the code is broken, the method is fundamentally wrong, results are fabricated, or the workspace is unrecoverable to be used as a gym — set the milestone to not\_possible and explain why in VERIFICATION.md. This tells the system the paper needs a full redo.

• Do not: refactor working code unnecessarily, rename files for style, add features that weren’t asked for, or re-run experiments that already have valid results.

[ · · · ]

## On Job Completion

<table><tr><td>Job Completed A compute job has finished.</td></tr><tr><td>Field Value</td></tr><tr><td></td></tr><tr><td>Job ID {run_id} Exit code {returncode}</td></tr><tr><td>stdout {stdout_path}</td></tr><tr><td>stderr {stderr_path} Read the log files to review the job&#x27;s output.</td></tr><tr><td></td></tr></table>

## What to Do

You are fully autonomous. There is no human in the loop. Do not ask for confirmation, do not wait for direction, do not say "let me know" or "please confirm". Decide and act immediately — directly edit code, fix bugs, and create files in this turn.

1. Review the logs. Understand what happened — success, failure, or partial results.

2. Assess progress. How does this move you toward your reproduction goals?

3. Edit code in this turn. Based on the logs, directly edit scripts, fix bugs, update configs, create new files — whatever is needed. Most turns should make a concrete change that advances the reproduction; spend a turn without editing only when you genuinely need to inspect or plan, not to show motion.

4. Submit the next job with your updated code, or signal done if finished.

5. Update your milestone if progress was made.

End your response with the structured YAML block as described in your instructions.

## J.3 Prompts for Evaluation

## Protocol

Action Protocol

You are fully autonomous — there is no human in the loop. Before your turn ends, edit action.yaml at the workspace root with one of the blocks below, then stop.

To submit a compute job:

```yaml
action: submit
gpus: 1
test: false
notes: "what this job does"
```

The pipeline will submit the job, wait for it to finish, and invoke you again with the logs.

• gpus: number of GPUs (set to 0 for a CPU-only job).

• test: set to true for quick validation jobs on test nodes (max 1 hour wall-clock), false for normal jobs.

– Use test: true to smoke-test your setup (does the container work? does training start? does eval run?) before committing to a long job.

– Test nodes have a 1-hour time limit — only use them for quick checks, never full training runs.

– Works for both GPU and CPU jobs: gpus: 1, test: true -> GPU test node; gpus: 0, test: true -> CPU test node.

• Anything beyond a few minutes ofcompute must use action: submit. Inline bash counts against your turn’s 4-hour wall-clock cap and will be killed mid-job.

To finish the run:

```yaml
action: done
notes: "why you are done"
```

action: done is PERMANENT — no follow-up session, no successor agent, no next turn. Do not mark done because of transient infrastructure problems (container build failures, Nexus/Hive errors, GPU job timeouts); those are fixable. Iterate on the environment, resubmit, or describe the issue in your next submit notes, but stay in the workflow.