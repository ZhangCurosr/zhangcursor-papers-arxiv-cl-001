# MiRL: Error-Targeted Reinforcement Learning for Correcting Terminal Agent Misbehaviors

Runchu Tian Zaid Khan Zun Wang Justin Chih-Yao Chen Mohit Bansal

Department of Computer Science University of North Carolina at Chapel Hill runchu@cs.unc.edu

## Abstract

Error-targeted learning gives a legible way to improve a model: identify specific weak nesses of the model, train the model to fix these weaknesses, and check whether the weaknesses are fixed. We study how to apply errortargeted learning to terminal agents, which solve long-horizon tasks specified in natural language by interacting with a command shell. Existing error-targeted methods are designed for single-turn tasks, and existing analyses of agent failures are used to evaluate and debug agents rather than to train them. We propose Misbehavior-driven RL (MiRL), which first profiles the errors of a policy to find generaliz able misbehaviors: kinds of wrong steps that recur across many environments. MiRL then trains the policy with RL, starting rollouts at the steps where these misbehaviors occur and rewarding a rollout only if it passes the tests and avoids the misbehavior. On Qwen3.5-4B, eight misbehaviors account for 52.1% of wrong steps. After training with MiRL, the policy repeats the measured misbehaviors 43% to 57% less often on average on unseen prefixes. MiRL outperforms TMax-4B on Terminal-Bench Lite (51.8% vs. 47.4%) and matches it on Terminal-Bench 2.1 (21.3% vs. 20.6%), while using 35% of its rollouts. These results show that errortargeted learning can be applied to realistic, long-horizon terminal tasks.<sup>1</sup>

## 1 Introduction

Error-targeted learning finds the errors a model makes and trains the model to stop making those errors (Khan et al., 2025; Lee et al., 2024). This approach is desirable because it makes model improvement easy to understand, as each training example is tied to some named error, and after training we can check whether that error becomes less frequent. For LLM agents, prior work has built taxonomies of agent errors (Cemri et al., 2025; Zhu et al., 2026b) and methods that locate the step at which an agent fails (Zhang et al., 2025, 2026b). These analyses are mostly used to evaluate and debug agents, not to decide what the agent trains on. Terminal agents (Bin et al., 2026; Liu et al., 2026) are typically instead trained by scaling the number of training environments (Pi et al., 2026; Gandhi et al., 2026; Wu et al., 2026; Ivison et al., 2026); for example, one TMax training run samples from 14.6K environments and takes two to three days on eight H100 nodes (Ivison et al., 2026). To our knowledge, error-targeted learning has not been applied to terminal agents.

(a) A few error types dominate the distribution  
![](images/db675ce6ec046af25508cc450edb1b9f2993122ef63e3305d433f142fc1ed933.jpg)

(b) MiRL: a misbehavior-driven RL recipe  
![](images/b5894080bb7094ceeeedb50ffbee8ff6fb2790955fa864ba6324ed4b9cad4928.jpg)

(c) Empirical results: data-efficient terminal agent training  
![](images/217a4a3032fd275fbc91835ed79a951b8378d06a599db25184a54b64a7860a42.jpg)  
Figure 1: Overview of MiRL. (a) A few generalizable misbehaviors account for a large share of error steps. (b) MiRL starts rollouts from misbehavior prefixes and rewards rollouts that pass the tests and avoid the misbehavior. (c) MiRL matches or outperforms TMax-4B using 35% of its rollouts. TB-Lite and TB-2.1 denote Terminal-Bench Lite (OpenThoughts-Agent team et al., 2026) and Terminal-Bench 2.1 (Merrill et al., 2026).

Applying error-targeted learning to terminal agents is not straightforward. Existing errortargeted methods work on and are designed for single-turn problems: when the model answers a problem incorrectly, these methods generate similar problems for training. A terminal agent takes dozens of steps per task, and a failed task is often caused by one or two wrong steps, such as reporting that the tests pass without running them. This creates two difficulties. First, many wrong steps are specific to one task, so fixing them one at a time does not help on other tasks; training should instead target generalizable misbehaviors, kinds of wrong steps that recur across many tasks. Second, training from the start of a task spends most rollouts on steps the agent already performs correctly, and the agent may never reach the state where the misbehavior occurred. We therefore ask: (1) how can we identify the generalizable misbehaviors of a terminal agent, and (2) how can we train the agent directly at the steps where they occur?

We answer these questions by connecting two tools: step-level error analysis of agent trajectories and RL from intermediate states. To answer the first question, we run error analysis on the policy we are about to train, which we call error profiling (Figure 1(a)). We roll out the policy on 2,017 environments from TMax’s environment set (Ivison et al., 2026); an error localizer flags wrong steps, a teacher model corrects each flagged step so that the rollout can continue and reveal later errors, and an annotator model groups the wrong steps into error types. Eight error types, such as “Makes false claims about tests” and “Fails to produce runnable commands”, cover 52.1% of all wrong steps and appear in 75.4% of the environments; we treat these as the generalizable misbehaviors to target. To answer the second question, we introduce MiRL, a misbehavior-driven RL recipe (Figure 1(b)). MiRL builds on two known techniques, starting rollouts from intermediate states (Xi et al., 2024; Yi et al., 2026) and providing reward signals beyond the binary verifier outcome (Lightman et al., 2024; Gunjal et al., 2026), each guided by the error profiling outputs. Each group of rollouts starts at a misbehavior prefix, the part of a trajectory just before a misbehavior step, and a rollout receives reward only if it passes the tests and a behavior judge confirms that it avoids the misbehavior of that prefix. To start from a prefix, MiRL restores the container by rerunning the prefix’s commands in a fresh container and keeps the prefix only if every command returns the recorded output.

We train Qwen3.5-4B (Qwen Team, 2026a) with MiRL using TMax’s RL recipe and evaluate on Terminal-Bench Lite (OpenThoughts-Agent team et al., 2026) and Terminal-Bench 2.1 (Merrill et al., 2026). On held-out prefixes, MiRL reduces the rate of the measured misbehaviors by 43% to 57% on average, measured by two behavior judges. MiRL also improves task performance: with the same number of training environments (189), MiRL outperforms TMax-189 (the TMax recipe trained on the same number of environments), PivotRL (Yi et al., 2026), and a form of RaR (Gunjal et al., 2026) enhanced with outcome rewards (RaR+), by at least 4.2 points on Terminal-Bench Lite and 2.2 points on Terminal-Bench 2.1. MiRL also outperforms TMax-4B, the released checkpoint trained with the full TMax recipe, which samples from 14.6K environments, on Terminal-Bench Lite (51.8% vs. 47.4%) and matches it on Terminal-Bench 2.1 (21.3% vs. 20.6%), while using 35% of its rollouts. Ablations show that starting from misbehavior prefixes and adding the behavior reward each contribute (7.9 and 6.5 points on Terminal-Bench Lite). These results show that error-targeted learning applies to terminal agents when training targets generalizable misbehaviors directly at the steps where they occur.

## 2 Related Work

Terminal agent training and data efficiency. Recent terminal agents improve mainly by scaling training data. Synthesis pipelines generate terminal environments from seed tasks and taxonomies (Gandhi et al., 2026; Zhu et al., 2026a; Pi et al., 2026; Ivison et al., 2026) or derive them from repositories and human-validated sources (Wu et al., 2026; Shen et al., 2026; Li et al., 2026b). Some works train on tens of thousands of terminal environments or hundreds of thousands of trajectories (Li et al., 2026b; Pi et al., 2026). Recent works improve data efficiency: Terminal-Lego (Yang et al., 2026) builds a more compact dataset around the inspect-act-verify loop, and Terminal-World (Cheng et al., 2026) derives 5.7K terminal environments from agent skills. Such a trend is also observed in other domains, e.g., RL for math reasoning can be effective with about a thousand or even a single carefully selected example (Li et al., 2025; Wang et al., 2025). These works improve data efficiency by improving the quality of tasks and trajectories, or by selecting learnable examples, whereas MiRL does so by targeting the general, recurring errors of the policy (generalizable misbehaviors).

Analyzing agent failures. MAST (Cemri et al., 2025) and AgentErrorTaxonomy (Zhu et al., 2026b) build taxonomies of failure modes from annotated agent traces, and Who&When (Zhang et al., 2025) and AgenTracer (Zhang et al., 2026b) locate the agent and the step that cause a failure. For terminal agents, TermiGen (Zhu et al., 2026a) finds that about half of the errors in a sample of trajectories are verification failures. These studies use failure analysis to evaluate, diagnose, or debug agents. We use it to drive data-efficient training: MiRL rolls out from the misbehavior prefixes found by error profiling and rewards a rollout only if it also avoids the misbehavior.

RL from selected states and step-level supervision. An outcome reward scores a whole trajectory by its final state. This gives coarse credit assignment because correct steps in a failed trajectory are penalized, and error steps in a successful trajectory are reinforced. To give more specific learning signals, one line of work starts rollouts from intermediate states. ${ \tt R } ^ { 3 }$ (Xi et al., 2024) starts RL from intermediate states of correct demonstrations and moves the start point backward as a curriculum. PivotRL (Yi et al., 2026) samples several actions at each step of supervised fine-tuning (SFT) trajectories and trains only on the steps where these actions get different rewards. An action is rewarded if it is equivalent to the action in the SFT trajectory. Another line adds supervision beyond the task outcome: process reward models score each intermediate step (Lightman et al., 2024), GiGPO (Feng et al., 2025) assigns step-level credit across rollouts that revisit the same state, and RaR (Gunjal et al., 2026) scores whole trajectories against taskspecific rubrics. Unlike the first line, MiRL starts rollouts from misbehavior prefixes collected by error profiling. Compared to the second line, MiRL shapes rewards so that a rollout must avoid a specific misbehavior to be reinforced.

## 3 Methodology

We aim to apply error-targeted learning to terminal agents. Our intuition is that a few general errors account for many of the agent’s error steps, so fixing these errors may improve the agent on many environments. To identify these errors, we run error profiling, as shown in Figure 3: we roll out the policy on a pool of environments, check and record the error steps, and group them into error types. We call the types that recur across many environments generalizable misbehaviors. To train the policy at the steps where these misbehaviors occur, we propose MiRL, our RL recipe, which starts rollouts from misbehavior prefixes instead of from scratch, and adds a behavior reward to the outcome reward, so a rollout gains reward only if it passes the tests and a behavior judge confirms that it avoids the misbehavior.

![](images/98108da5dba984438187c545af47805bc0da1b6dc48ee27403fdff71a4576ba5.jpg)  
Figure 2: An example terminal environment. The agent must move duplicate photos from backup/ into dupes/ and list them in report.csv; the tests check the result.

## 3.1 Preliminaries

Environments and agents. A terminal environment consists of an instruction that states a user requirement, a container in which the agent runs shell commands, and a test suite that checks the final state of the container (Figure 2). An agent, a language-model policy π, solves a task over multiple steps. Before each step, it reads the prefix, which includes the instruction, its earlier steps, and their outputs. It then takes one step: a short reasoning followed by a shell command. The sequence of steps and their outputs forms a trajectory, which ends when the agent submits or reaches a step limit. The test suite then checks the final state of the container, giving an outcome reward $r _ { o }$ of 1 if all tests pass and 0 otherwise.

LLMs in error profiling. Besides the policy, error profiling calls LLMs to play three roles: an error localizer that checks whether a step is problematic given its prefix, a teacher that corrects a problematic step, and an annotator that proposes error types and labels each error step with them.

Errors and misbehaviors. We write the prefix of a step as h. We define an error step as a step that the error localizer marks as problematic given its prefix. The annotator groups error steps with the same cause into error types and gives each type a name and a definition. We call an error type a generalizable misbehavior, or misbehavior for short, when it recurs across many environments and is not tied to very few of them. We call the prefix of an error step of a misbehavior a misbehavior prefix.

![](images/b6d23bf5c100f8003a9000427e6a0606ad5b5c8d1272f5ec94d83f8407c35af9.jpg)  
Figure 3: Overview. (a) The error localizer flags wrong policy steps, and the teacher corrects them; (b) the annotator groups error steps into types, where a few misbehaviors dominate; (c) prefixes that rerun exactly are kept for live RL rollouts; (d) rollouts start from them and gain reward only if they pass the tests and avoid the misbehavior.

## 3.2 Error Localization

To locate the errors of the policy, we roll it out on a pool of environments and let the teacher replace each error step with a corrected one, so that the rollout can reveal later errors.

At each step of a rollout, error profiling goes through four stages, as in Figure 3(a). (1) The policy proposes its next step from the current prefix. (2) The error localizer checks the step with its prefix, flags it only if it is likely to lead to failure, and writes a short explanation. (3) For a flagged step, the teacher writes a correction from the same prefix, so that the rollout can continue without being pushed into incorrect states. The correction is checked in the same way as a step of the policy. (4) The accepted step, either the policy’s own step or a correction from the teacher, is executed, and the rollout returns to (1) for the next step. If the teacher fails to give a correction that passes this check within three tries, the rollout is discarded. During this process, each error step is recorded with its prefix and the error localizer’s explanation.

## 3.3 Error Taxonomy Construction

To find the errors that recur across environments, we group the error steps into error types, as in Figure 3(b). First, the annotator reads batches of error steps, together with the error localizer’s explanations, and proposes error types for each batch, with a name and a definition for each type. We repeat this on several batches and keep the types that recur across batches. This gives a fixed set of error types. The annotator then labels every error step with one or more types from this set, or with none if it fits no known type. Finally, we keep the error types that cover at least a fixed share of the error steps and treat the others as a long tail of rare types.

In our analysis, a few error types cover about half of the error steps and recur across many environments (§4.2), so they can be considered as generalizable misbehaviors. Because these misbehaviors recur across environments, a small number of environments already gives many examples of each, and fixing them may also help the policy on unseen environments.

## 3.4 Rollout from Misbehavior Prefixes

Standard RL recipes used for terminal agent training, such as GRPO (Shao et al., 2024) and DPPO (Qi et al., 2026), start every rollout anew, from step zero. An important characteristic of these methods is that they are on-policy, that is, they train on rollouts generated by the policy itself. Starting rollouts anew would mean that we could not correct a specific misbehavior in an on-policy manner unless the identical misbehavior occurred and was detected in a fresh rollout. The highly stochastic nature of long-horizon terminal agent trajectories means that generating rollouts that display a particular misbehavior would be very compute-intensive. To fix the misbehaviors in a more targeted and efficient manner, MiRL instead starts each group of rollouts from a pre-existing trajectory prefix where the policy exhibited a misbehavior. The policy then continues from the prefix until the rollout ends. We select a fixed number of misbehavior prefixes for each misbehavior to form the training data.

However, a live rollout cannot start from a prefix directly. To continue from step k, the policy needs the container state at step k, but the recorded data keep only the prefix and not the container, since saving a snapshot of the container at every step would require substantial storage. We therefore need to reconstruct the state from the prefix, as in Figure 3(c): we rerun the commands of the prefix step by step in a fresh container of its environment, and keep the prefix only if every command returns exactly the same output as in the prefix. Many containers are flaky: the same command may produce different outputs depending on time, randomness, or network conditions. Such prefixes cannot reach the required state, so we discard them. The training environments are those of the remaining prefixes.

## 3.5 Misbehavior-Aware Reward Design

The outcome reward $r _ { o }$ checks only whether the tests pass, so it cannot give feedback on whether a rollout avoids the misbehavior. We therefore introduce a behavior reward $r _ { e }$ . The behaviorjudge reads the first m steps that the policy takes after the prefix, together with the definition of the misbehavior of that prefix. It gives $r _ { e } = 1$ if the policy avoids the misbehavior, and $r _ { e } = 0$ otherwise. MiRL combines the two rewards as $r = r _ { o } \wedge r _ { e }$ , as in Figure 3(d), so a rollout gains reward only when the tests pass and the misbehavior is avoided. In this way, MiRL avoids reinforcing trajectories that pass the tests but still contain the misbehavior.

## 4 Experiments

## 4.1 Setup

Training data. We run error profiling on 2,017 random terminal environments from TMax’s environment set, which do not overlap the evaluation benchmarks. We roll out the policy on each environment with 2 rollout seeds. We use Qwen3.8- 27B (Qwen Team, 2026b) as the error localizer, because it is strong enough to judge terminal steps, with a reported 73.0% on Terminal-Bench 2.1. We also use it as the teacher, because it belongs to the same model family as the policy, so its corrections stay close to the policy’s own distribution (Yang et al., 2026; Raoof et al., 2026). We use Claude

Opus 5 (Anthropic, 2026) as the annotator. We keep every error type that covers at least 1% of the error steps, which gives eight generalizable misbehaviors. For these eight misbehaviors, we sample 80 misbehavior prefixes each, which gives 640 prefixes. After the rerun check in Figure 3(c), 219 prefixes remain (34%), from 189 environments.

Models and training. We use Qwen3.5-4B, the same base model as TMax-4B, as the policy model. We follow the RL recipe of TMax and use their harness, except for a group size of 16 instead of 32 to control the compute budget, and a total of 160 training steps instead of 500. The behavior judge is Gemini 3.8 Flash (Google, 2026), which reads the $m = 5$ steps after the prefix. It returns only a binary signal, so the policy receives no text from the behavior judge to imitate. We run our experiments on 8 A100 80GB GPUs. Appendix A gives the full settings and prompts.

Baselines. We compare MiRL with TMax-4B, the released checkpoint (step 200 of 500) trained with the full TMax recipe, which samples training environments from a pool of 14.6K. Because Terminal-Bench scores are sensitive to the evaluation hardware, we report both its published scores and our evaluation of the released checkpoint. We also train three baselines with the same configuration as MiRL, except that their 189 environments are sampled at random from the same 2,017 environments, since these baselines do not select environments by error profiling or any other method. The first, TMax-189, replicates the TMax recipe but only on 189 environments: it starts rollouts at the beginning of each task and uses the outcome reward. The other two are the closest prior methods to the two parts of MiRL. PivotRL (Yi et al., 2026) also trains from selected steps of a trajectory: it samples several actions at each step, keeps the steps where these actions get different rewards, and rewards each action mainly with an LLM judge. Since each PivotRL update trains on a single step rather than on whole rollouts, matching the number of training steps is not meaningful, so we give PivotRL about the same GPU budget as TMax-189 for its full training run (≈300 A100 hours). RaR (Gunjal et al., 2026) also rewards the process with an LLM: a judge scores the whole trajectory against a rubric generated for each environment. The original RaR uses only this rubric score, so for a fairer comparison we strengthen it with the outcome reward and call it RaR+: when the judge writes the rubric for an environment, it also sets the weights of the rubric score and the outcome reward.

top-8 misbehaviors: 52.1% of the error steps  
![](images/a5a96ec572e95c802961c54f503aa689b5b0102b04ead2d01f7869df9ba8ff44.jpg)  
Figure 4: Error steps concentrate in a few misbehaviors, followed by a long tail of environment-specific types; the top four appear across many environments.

Evaluation. We evaluate on two terminal benchmarks, Terminal-Bench Lite (TB-Lite; 100 tasks; OpenThoughts-Agent team et al., 2026) and Terminal-Bench 2.1 (TB-2.1; 89 tasks; Merrill et al., 2026), and report the mean and standard error over 3 runs, following TMax; some TB-Lite tasks give partial credit. To select the checkpoint, we first take the five checkpoints with the highest moving average of the training reward, and then, as in TMax, keep the one with the best TB-Lite score.

## 4.2 Misbehavior Analysis

Error profiling on the 2,017 environments yields 12,746 error steps. Figure 4 shows their distribution over error types. The eight most frequent error types cover 52.1% of all error steps. Each appears in 6.9% (140) to 46.7% (942) of the 2,017 environments, and at least one appears in 75.4% (1,521) of them. None is specific to a particular environment, so they can be considered as generalizable misbehaviors (§3.1). Prior work observes a similar concentration: in a sample of agent trajectories, TermiGen finds that about half of the errors are verification failures, in which the agent omits a final check (Zhu et al., 2026a). Our most frequent misbehavior, “Makes false claims about tests”, is also a kind of verification error.

The remaining errors form a long tail of environment-specific error types. Two examples are “Uses a greedy regex on log lines” and “Replaces udb inside udbuf” (Figure 4). Such errors arise from a detail of one environment and rarely carry over to other environments. Appendix B lists more examples and per-type statistics.

![](images/bb00099e253e6fdd6ad29f69424e334483dc073540b68b6f71aa9e4c47cf8557.jpg)  
Figure 5: Terminal task performance: mean (%) and standard error over 3 runs. Dashed boxes are the scores reported in the TMax paper (Ivison et al., 2026).

## 4.3 Main Results

MiRL outperforms TMax-4B. Figure 5 shows the terminal task performance. On our evaluation hardware, both the base model and TMax-4B achieve higher scores than their published ones (e.g., 47.4% vs. 42.6% for TMax-4B on TB-Lite), so we compare all recipes with our own runs. MiRL reaches 51.8% on TB-Lite and 21.3% on TB-2.1, compared with 47.4% and 20.6% for TMax-4B on our evaluation hardware: it gains 4.4 more points on TB-Lite and matches TMax-4B on TB-2.1.

MiRL outperforms all baselines trained on the same number of environments. On 189 random environments, TMax-189 improves the base model by 10.6 points on TB-Lite, which is the smallest gain among the four recipes trained with a controlled number of environments. This indicates that a small random set is not enough for the standard recipe of RL with verifiable rewards (RLVR; Lambert et al., 2025). PivotRL and RaR+ improve the base model more, by 14.4 and 14.8 points on TB-Lite. MiRL outperforms the best of these baselines by 4.2 points on TB-Lite and 2.2 points on TB-2.1. The result suggests that targeting generalizable misbehaviors is an effective way to choose where rollouts start and what the judge checks.

We further show a 16-seed evaluation for enhanced robustness (§4.5).

## 4.4 Ablations

We ablate MiRL by adding its three parts one at a time to TMax-189 (Figure 6). Training on the environments where the misbehaviors occur lowers the score from 43.4% to 37.4% on TB-Lite and from 17.6% to 16.5% on TB-2.1. We attribute this to the higher difficulty of these environments for the policy: the share of groups in which all rollouts fail, which give no learning signal, rises from 27.3% to 42.9%. However, environment selection is not meant to help on its own. Instead, it provides the misbehavior prefixes for the next two parts. Starting rollouts at misbehavior prefixes adds 7.9 points on TB-Lite and 1.5 points on TB-2.1. We attribute this gain to training on the steps where the policy misbehaves, even though no reward checks the misbehavior yet. The behavior reward r adds 6.5 points on TB-Lite and 3.3 points on TB-2.1. With $r _ { e } .$ , the reward is stricter than the outcome reward alone: it excludes the rollouts that pass the tests but still misbehave. For example, a policy may claim success without verification. When its first attempt happens to be correct, the rollout passes and the misbehavior is not fatal; when the first attempt is wrong, the same unchecked submission makes the rollout fail. The outcome reward alone still rewards the passing rollout, so the policy may learn to repeat the misbehavior.

![](images/7a44625d3f9467ed2f9ec491c3dbd9bb1f5ebdcdadbd10941ebed3200cc9af07.jpg)  
Figure 6: Ablations of MiRL. Starting from TMax-189, each bar adds one part: environment selection, prefix start, and the behavior reward $r _ { e }$

## 4.5 Analysis

MiRL reduces misbehavior frequency. To verify that MiRL reduces the misbehaviors, we measure how often each policy repeats a misbehavior before and after training. We measure four misbehaviors with enough prefixes left after the rerun check, each on 20 newly sampled misbehavior prefixes that are unseen in training. For each unseen prefix, we sample the policy 16 times, and a behavior judge checks whether each sample repeats the misbehavior of that prefix. The misbehavior rate is the percentage of samples that repeat the misbehavior. We measure with two behavior judges:

![](images/8f82cfb2fdfdae6a62cabf5d27cc05afef32ccb14f362cb817dc0c590cacc894.jpg)  
Figure 7: Misbehavior rate (%): the share of 16 samples that repeat the misbehavior on unseen prefixes. Judges: Gemini 3.8 Flash (used in training) and Qwen3.8-27B.

Gemini 3.8 Flash, which MiRL uses in training, and Qwen3.8-27B (Qwen Team, 2026b).

As shown in Figure 7, MiRL lowers three of the four misbehaviors under both judges. For example, “Makes false claims about tests” drops from 26.7% to 8.1% under Gemini and from 29.8% to 11.7% under Qwen. Since the rate drops under both judges, MiRL likely reduces these misbehaviors rather than only learning to satisfy its own judge. The fourth, “Prints a binary file as text”, is rare in the base policy and stays at about 5%.

Training with the outcome reward alone does not consistently reduce misbehaviors. TMax-4B reduces “Fails to produce runnable commands” to below 1% under both judges, similar to MiRL. However, “Makes false claims about tests” decreases only slightly (26.7% to 22.5% under Gemini and 29.8% to 28.4% under Qwen). “Retries without fixing the cause” decreases under Gemini but increases under Qwen (29.7% to 35.2%), while “Prints a binary file as text” becomes about five times more frequent (4.1% to 21.9%).

Pass@k: Sharpening or learning? A common critique of RL is that it sharpens the output distribution of the base model rather than teaching it new capabilities (Yue et al., 2025; Zhao et al., 2025; Huang et al., 2025). In that case, RL improves pass@1, but pass@k at large k improves little or even falls below that of the base model. We test this on TB-2.1 with the unbiased pass@k (Chen et al., 2021), using 16 runs per task. Figure 8(a) compares the scores. TMax-4B shows this pattern: it improves over Qwen3.5-4B at pass@1, but the base model catches up at about k = 11 and is ahead at $k = 1 6 ,$ . In contrast, MiRL outperforms the base model at every k up to 16 by a large margin, and reaches 39.3% at k = 16. Figure 8(b) shows how often each task is solved over the 16 runs. These results suggest that error-targeted RL raises the upper bound of the model more than standard RL does. We suspect one reason is that MiRL keeps learning in a zone of proximal development. By starting rollouts at misbehavior prefixes, MiRL only needs the policy to complete the rest of each task, and the misbehavior rate at these prefixes is mostly moderate (Figure 7). Prior work shows that starting from partial solutions (Li et al., 2026a) and moderate task difficulty (Zhang et al., 2026a) improve both pass@1 and pass@k.

![](images/65edda7617b26bd7fea7498d5e65959ef1ec0e7cdccad683f6c68f6aaac9f8db.jpg)

![](images/6de07a8b412ded830c19053e84fb9245f2915b80ae2c132a7bfbad81704e5b59.jpg)  
Figure 8: Pass@k on TB-2.1 with 16 runs per task. (a) Unbiased pass@k of Qwen3.5-4B, TMax-4B, and MiRL. (b) Pass rate of each task over the 16 runs.

![](images/ba8763bba54aebd3f91a5b5348a968c97e15c7a378e31919c1c9097bc215db03.jpg)  
Figure 9: Training resources of MiRL relative to TMax-4B, up to the best checkpoint rather than the full training run. Checkered segments are error profiling and solid segments are RL training. GPU time is in A100 hours.

![](images/f39e241d011b4b33cc5e360d113391a81adb8b732109e645652b112596fc58f7.jpg)  
Figure 10: Training resources of the four recipes on 189 environments, up to the best checkpoint rather than the full training run. Checkered segments are preprocessing and solid segments are RL training.

MiRL is more data-efficient than TMax-4B. We measure data efficiency by the training resources needed to reach a given performance level, including the resources spent on error profiling and other preprocessing. We count these resources up to each recipe’s best checkpoint rather than over its full training run. Figure 9 compares MiRL and TMax-4B up to their best checkpoints. MiRL uses fewer terminal environments, and only 35% of the rollouts and 34% of the input tokens of TMax-4B. Its output tokens drop further, to 28%, because each rollout starts from a prefix, so the policy does not generate the steps of the prefix again. As a result, MiRL uses only 39% of the GPU hours of TMax-4B. Although MiRL pays a model API fee for the LLMs in error profiling and for the behavior judge in RL training, which TMax-4B does not, its total cost is still lower, at 56% of that of TMax-4B, while it scores 4.4 points higher on TB-Lite and matches it on TB-2.1. Compared with TMax-189, MiRL uses about as many rollouts and GPU hours (26.2K vs. 26.4K rollouts and 202.8 vs. 202.7 hours) but scores 8.4 points higher on TB-Lite.

MiRL achieves the best performance at a higher training cost than PivotRL and RaR+. PivotRL and RaR+ use \$999 and \$770 in total cost, compared with \$1,960 for MiRL (Figure 10). One reason is that the reward of MiRL, $r _ { o } \wedge r _ { e } ,$ is even sparser than the outcome reward of standard RLVR, while PivotRL and RaR+ use judge scores that make their rewards denser. As a result, 42.2% of the groups of MiRL are dead, i.e., all rollouts in the group get zero reward and give no learning signal, compared with 27.3% for TMax-189, 25.3% for PivotRL, and 0.13% for RaR+. The denser rewards also let both baselines converge faster: RaR+ reaches its best checkpoint at step 80 of 160, compared with step 100 for MiRL, and PivotRL at step 595 of 1,168. However, their scores stay at least 4.2 points below MiRL on TB-Lite. Appendix C details how we compute the training resources.

![](images/c17b8cd6d5b1f850877b68472cf3a52bed337408ed3cb1cdae14e2a3bcdaa6c8.jpg)  
# MiRL training environments (log scale)  
Figure 11: MiRL trained on different numbers of environments selected from error profiling (0.25×, 0.5×, 1×, and 2×). Mean (%) and standard error over 3 runs; horizontal lines show TMax-4B and Qwen3.5-4B.

MiRL improves with more environments. To test how MiRL scales with the number of environments, we train it on different number of environments, 0.25× and 0.5×, where the smaller subset is part of the larger one, with the same number of training steps, group size, batch size, and reward. For 2×, we sample more misbehavior prefixes from the same error profiling and keep those that pass the rerun check, which gives 378 environments, including the original 189. The score rises with the number of environments, from 42.0% to 52.4% on TB-Lite and from 16.1% to 22.1% on TB-2.1 (Figure 11). Half of the environments are enough for MiRL to match TMax-4B on TB-Lite (48.0% vs. 47.4%).

Reliable evaluation with more runs. Longhorizon agents get different scores across evaluation runs (Yao et al., 2025), and TB-2.1 has only 89 tasks, which adds more noise (Miller, 2024). So the mean over 3 runs, which many terminal-agent papers report (Ivison et al., 2026; Meng et al., 2026; Zhu et al., 2026a; Yang et al., 2026), is subject to considerable randomness. We therefore run 16 extra seeds for each of the three recipes on TB-2.1. As the number of seeds k grows, the mean standard error shrinks at about the rate of 1/ k, down to 0.62 at 16 seeds (Figure 12(a)). With 16 seeds, MiRL still has the highest mean score: 21.14%, vs. 20.22% for TMax-4B and 19.66% for RaR+ (Figure 12(b)). Figure 12(c) shows the score distribution over the 16 seeds. These results further support the ranking in Figure 5.

![](images/b921e75270309198c0caa1a8fa2483814dc024ccfc571d85a21bcd1ed17cdee3.jpg)

![](images/78446f6cc0793174872fb778e5267ba1e7e5b7726ba65609067576a426f54735.jpg)

(c) Distribution of the 16 seeds  
![](images/995e4e2b85be605e2f0c3825d87761acb2847578c35e1271c94e28bcfa844cb4.jpg)  
Figure 12: TB-2.1 scores of TMax-4B, RaR+, and MiRL with 16 extra seeds. (a) Mean standard error over k seeds, averaged over all k-seed subsets of the 16 seeds and over the three recipes. (b) Mean and standard error over the 3 runs in Figure 5 and over the 16 extra seeds. (c) Distribution of the 16 seeds: whiskers show the minimum and maximum, boxes the 25th and 75th percentiles, and circles the means.

## 5 Conclusion

In this work, we apply error-targeted learning to terminal agents. We (i) propose error profiling, which locates and corrects the error steps of a policy and groups them into error types to find its generalizable misbehaviors, (ii) introduce MiRL, an RL recipe that starts rollouts from misbehavior prefixes and rewards a rollout only if it passes the tests and avoids the misbehavior, and (iii) show that MiRL reduces the targeted misbehaviors, outperforms or matches all baselines on two benchmarks, and uses fewer training resources than TMax-4B. These results suggest that error-targeted learning can work for long-horizon terminal tasks when training targets generalizable misbehaviors where they occur.

## Limitations

First, we train and evaluate our method with only one model backbone, so it remains unclear how well the gains generalize to other model choices. Second, due to the compute budget, we train each recipe once and report the standard error over 3 evaluation runs, which does not capture the variance across training runs. Third, error profiling relies on external LLMs, including commercial ones, and the rerun check discards prefixes from non-deterministic environments, which may bias the training data. Finally, for reward design, we mainly try the conjunction of the two rewards and have not explored other ways to combine them.

## Acknowledgements

This work was supported by NSF-AI Engage Institute DRL-2112635, NSF-CAREER Award 1846185, and NDSEG PhD Fellowship (Air Force Office of Scientific Research under award number FA9550-25-C-B010). The views contained in this article are those of the authors and not of the funding agency.

## References

Anthropic. 2026. Claude Opus 5 system card. https: //www.anthropic.com/claude-opus-5-syste m-card.

Yi Bin, Xiaoyang Yuan, Haoxi Zeng, Wencheng Ye, Wenqi Shao, Chen Qian, Wei Ye, Yujuan Ding, Zheng Wang, Pengpeng Zeng, Jingkuan Song, and Heng Tao Shen. 2026. Terminal agents: A survey of AI agents in command-line environments. arXiv preprint arXiv:2608.20485.

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. 2025. Why do multi-agent LLM systems fail? In Advances in Neural Information Processing Systems.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Zihao Cheng, Hongru Wang, Zeming Liu, Xinyi Wang, Xiangrong Zhu, Yuhang Guo, Wei Lin, Jeff Z. Pan, and Yunhong Wang. 2026. Terminal-World: Scaling terminal-agent environments via agent skills. arXiv preprint arXiv:2605.20876.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for LLM

agent training. In Advances in Neural Information Processing Systems.

Kanishk Gandhi, Shivam Garg, Noah D. Goodman, and Dimitris Papailiopoulos. 2026. Endless terminals: Scaling RL environments for terminal agents. arXiv preprint arXiv:2601.16443.

Google. 2026. Gemini 3.8 Flash. https://ai.googl e.dev/gemini-api/docs/models/gemini-3.8-f lash.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean M. Hendryx. 2026. Rubrics as rewards: Reinforcement learning beyond verifiable domains. In International Conference on Learning Representations.

Audrey Huang, Adam Block, Dylan J. Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T. Ash, and Akshay Krishnamurthy. 2025. Self-improvement in language models: The sharpening mechanism. In International Conference on Learning Representations.

Hamish Ivison, Junjie Oscar Yin, Rulin Shao, Teng Xiao, Nathan Lambert, and Hannaneh Hajishirzi. 2026. TMax: A simple recipe for terminal agents. arXiv preprint arXiv:2606.23321.

Zaid Khan, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. 2025. DataEnvGym: Data generation agents in teacher environments with student feedback. In International Conference on Learning Representations.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, and 4 others. 2025. Tülu 3: Pushing frontiers in open language model post-training. In Conference on Language Modeling.

Nicholas Lee, Thanakul Wattanawong, Sehoon Kim, Karttikeya Mangalam, Sheng Shen, Gopala Anumanchipalli, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. 2024. LLM2LLM: Boosting LLMs with novel iterative data enhancement. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 6498–6526. Association for Computational Linguistics.

Jiazheng Li, Hongzhou Lin, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Yi Wu, and Jingzhao Zhang. 2026a. QuestA: Expanding reasoning capacity in LLMs via question augmentation. In International Conference on Learning Representations.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025. LIMR: Less is more for RL scaling. arXiv preprint arXiv:2502.11886.

Zhongzhi Li, Yucheng Shi, Zongxia Li, Ruhan Wang, Anhao Li, Zixun Huang, Junyao Yang, Lei Ke, Ninghao Liu, Haitao Mi, and Leowei Liang. 2026b. Recursive synthesis for long-horizon terminal tasks. arXiv preprint arXiv:2608.05466.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations.

Junwei Liu, Kaixin Wang, Yixuan Chen, Xin Peng, Zhenpeng Chen, Lingming Zhang, and Yiling Lou. 2026. Large language model-based agents for software engineering: A survey. ACM Transactions on Software Engineering and Methodology.

Fanzhe Meng, Guoxin Chen, Jiale Zhao, Shuang Sun, Zhiyu Lin, Wayne Xin Zhao, Ruihua Song, Ji-Rong Wen, and Kai Jia. 2026. CalibForge: Adversarial solver calibration for scaling learnable terminal tasks. arXiv preprint arXiv:2608.06352.

Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Jenia Jitsev, Di Lu, Orfeas Menis Mastromichalakis, and 66 others. 2026. Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In International Conference on Learning Representations.

Evan Miller. 2024. Adding error bars to evals: A statistical approach to language model evaluations. arXiv preprint arXiv:2411.00640.

OpenThoughts-Agent team, Snorkel AI, and Bespoke Labs. 2026. OpenThoughts-TBLite: A high-signal benchmark for iterating on terminal agents. https: //www.openthoughts.ai/blog/openthoughts-t blite.

Renjie Pi, Grace Lam, Mohammad Shoeybi, Pooya Jannaty, Bryan Catanzaro, and Wei Ping. 2026. On data engineering for scaling LLM terminal capabilities. In Advances in Neural Information Processing Systems.

Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. 2026. Rethinking the trust region in LLM reinforcement learning. In International Conference on Machine Learning.

Qwen Team. 2026a. Qwen3.5-4B. https://huggingf ace.co/Qwen/Qwen3.5-4B. Model card.

Qwen Team. 2026b. Qwen3.8-27B. https://huggin gface.co/Qwen/Qwen3.8-27B. Model card.

Negin Raoof, Richard Zhuang, Marianna Nezhurina, Etash Guha, Atula Tejaswi, Ryan Marten, Charlie F. Ruan, Tyler Griggs, Alexander Glenn Shaw, Hritik Bansal, E. Kelly Buchanan, Artem Gazizov, Reinhard Heckel, Chinmay Hegde, Sankalp Jajee, Daanish

Khazi, Emmanouil Koukoumidis, Xiangyi Li, Hange Liu, and 31 others. 2026. OpenThoughts-Agent: Data recipes for agentic models. arXiv preprint arXiv:2606.24855.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Qijia Shen, Zhiqi Huang, Vamsidhar Kamanuru, Aznaur Aliev, Jay Rainton, Ahmed Awelkair, Zhichen Zeng, Jiajun Li, Shi Dong, Yueming Yuan, Boyuan Ma, Qizheng Zhang, Jiwei Fu, Yuzhen Mao, Wendong Fan, Ping Nie, Philip Torr, Bernard Ghanem, Changran Hu, and 3 others. 2026. SETA: Scaling environments for terminal agents. arXiv preprint arXiv:2607.10891.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. 2025. Reinforcement learning for reasoning in large language models with one training example. In Advances in Neural Information Processing Systems.

Siwei Wu, Yizhi Li, Yuyang Song, Wei Zhang, Yang Wang, Riza Batista-Navarro, Xian Yang, Mingjie Tang, Bryan Dai, Jian Yang, and Chenghua Lin. 2026. Large-scale terminal agentic trajectory generation from Dockerized environments. In International Conference on Machine Learning.

Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, and 2 others. 2024. Training large language models for reasoning through reverse curriculum reinforcement learning. In International Conference on Machine Learning.

Sidi Yang, Chaofan Tao, Jierun Chen, Tiezheng Yu, Ruoyu Wang, Yuxin Jiang, Yiming Du, Wendong Xu, Jing Xiong, Taiqiang Wu, Lifeng Shang, Xiaohui Li, Ngai Wong, and Haoli Bai. 2026. What makes interaction trajectories effective for training terminal agents? arXiv preprint arXiv:2606.03461.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2025. τ-bench: A benchmark for toolagent-user interaction in real-world domains. In International Conference on Learning Representations.

Junkeun Yi, Damon Mosk-Aoyama, Baihe Huang, Ritu Gala, Charles Wang, Sugam Dipak Devare, Khushi Bhardwaj, Abhibha Gupta, Oleksii Kuchaiev, Jiantao Jiao, Jian Zhang, and Venkat Srinivasan. 2026. PivotRL: High accuracy agentic post-training at low compute cost. arXiv preprint arXiv:2603.21383.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025.

Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems.

Charlie Zhang, Graham Neubig, and Xiang Yue. 2026a. On the interplay of pre-training, mid-training, and RL on reasoning language models. In International Conference on Machine Learning.

Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. 2026b. AgenTracer: Who is inducing failure in the LLM agentic systems? In International Conference on Learning Representations.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. 2025. Which agent causes task failures and when? On automated failure attribution of LLM multi-agent systems. In International Conference on Machine Learning.

Rosie Zhao, Alexandru Meterez, Sham M. Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. 2025. Echo chamber: RL post-training amplifies behaviors learned in pretraining. In Conference on Language Modeling.

Kaijie Zhu, Yuzhou Nie, Yijiang Li, Yiming Huang, Jialian Wu, Jiang Liu, Ximeng Sun, Zhenfei Yin, Lun Wang, Zicheng Liu, Emad Barsoum, William Yang Wang, and Wenbo Guo. 2026a. TermiGen: High-fidelity environment and robust trajectory synthesis for terminal agents. arXiv preprint arXiv:2602.07274.

Kunlun Zhu, Muxin Tian, Bingxuan Li, Zijia Liu, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Robert Tang, Zicheng Liu, Heng Ji, Pan Lu, James Zou, and Jiaxuan You. 2026b. AgentDebug: Where LLM agents fail and how they can learn from failures. In ICML 2026 Workshop on Failure Modes in Agentic AI.

## A Configuration Details

Table 1 lists the hyperparameters of all recipes.

The boxes below give the prompts of all LLM roles. Text in braces, such as {prefix}, is filled in at run time. The RaR+ prompts adapt those of RaR (Gunjal et al., 2026), and the PivotRL judge prompt is taken unchanged from NeMo-Gym.

Error localizer Error profiling   
You are reviewing one step of an AI agent working on a   
command-line task.   
Below is EXACTLY what the agent saw, and EXACTLY what it   
did next. You are seeing the agent's context up to this   
point and nothing beyond it. You do not know whether the   
task eventually succeeded.   
Judge this one step.   
A step is GOOD if an experienced engineer would plausibly   
do it here. A step is BAD if it still works, but raises

## Error localizer (cont.)

Error profiling

the chance of failure, and an experienced engineer would   
not do it. Examples of bad-but-working steps:   
acting on data it has not inspected; writing one large   
script that does everything at once instead of building up   
in checkable pieces; declaring work finished without   
verifying the actual output. A step that cannot work from   
this state is also bad.   
Do not penalise a step for being slow or for taking a   
different approach than you would. Only flag it if it   
makes failure more likely.   
A step here is a short <think> reasoning block followed by   
ONE bash command.   
Keep your reasoning SHORT: at most 150 words of thinking   
before you answer. Do not re-derive the whole task. Judge   
the one step.   
Reply in this JSON form, nothing else. Fill the fields IN   
ORDER -- the justification comes before the verdict, so   
you must state what the command does before you judge it.   
{   
"command\_seen": "<one short line naming what the command   
does, e.g. 'cat on a binary PDF'>",   
"reason": "<one sentence. REQUIRED. Never null, never   
empty, on good verdicts too.>",   
"verdict": "good" or "bad"   
}   
Before you answer, check "command\_seen" against your own   
reason: if the command you described is one you would   
reject, the verdict is "bad".   
=== WHAT THE AGENT SAW ===   
{prefix}   
=== WHAT THE AGENT DID ===   
{step}

## Teacher

Error profiling

You are an expert engineer working on a command-line task. Below is EXACTLY the state so far: the task, and every command already run with its output. Write the CORRECT next step yourself.

Write a short <think> block explaining your reasoning, then call the \`bash\` tool with exactly ONE command. The command may chain shell with && or ; but it is one command string.

Do not repeat work that the output above already shows is done. Build up in small checkable pieces rather than one large script. Do not declare the task finished without verifying the real output.

## Annotator: proposing error types

Error profiling

You are analysing failures of a terminal agent: a language model that solves tasks by running shell commands. A stronger model reviewed every step the agent took and marked some of them bad. Below is a random sample of those bad steps.

What makes a good type

A good type passes two tests.

Test 1, transfer. If the agent learned to avoid this on half of the examples, would that carry to the other half? It carries when all the examples share one habit. It does not carry when each example needs a different piece of knowledge.

Test 2, frequency. Is the type common enough that fixing it matters?

<table><tr><td>Annotator: proposing error types (cont.) Error profiling</td></tr><tr><td>Good: &quot;repeats a command that just failed, unchanged&quot; One habit: check whether anything changed before running the same thing again. Learn it on some examples, get it</td></tr><tr><td>on the rest. And it happens often. Too broad: &quot;runs a tool on input it cannot accept&quot;</td></tr><tr><td>Sounds general, but every example needs a different fact. Learning that openssl cannot read a public key teaches nothing about ffmpeg. Nothing carries. Too narrow:</td></tr><tr><td>&quot;uses openssl x509 on a public key&quot; This one does carry, but it almost never happens. Fixing it changes nothing. Two more rules.</td></tr><tr><td>Ignore what the task was about. The examples mention numpy, sqlite, openssl, pydub. That is incidental. Name what the agent did wrong. If a type name holds a tool name or a file name, it is too narrow.</td></tr><tr><td></td></tr><tr><td>Write each definition so it can be checked on one example alone, by a yes or no question. Someone holding your definition and one bad step must be able to decide, without seeing the rest of the task. How sure you need to be</td></tr></table>

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>RL training (all recipes)</td><td></td></tr><tr><td>Algorithm</td><td>DPPO (Qi et al., 2026)</td></tr><tr><td>Policy model</td><td>Qwen3.5-4B</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>10−6, constant</td></tr><tr><td>Group size</td><td>16 (TMax: 32)</td></tr><tr><td>Batch size</td><td>8</td></tr><tr><td>Training steps</td><td>160 (TMax: 500)</td></tr><tr><td>Sampling temperature</td><td>1.0</td></tr><tr><td>KL coefficient</td><td>0</td></tr><tr><td>Advantage</td><td>Centered</td></tr><tr><td>Dead groups</td><td>Filtered and resampled</td></tr><tr><td>Max response / turn length</td><td>65,536 / 16,384 tokens</td></tr><tr><td>Max tool steps</td><td>64</td></tr><tr><td>Bash timeout</td><td>120 s</td></tr><tr><td>Checkpoint selection</td><td>Best TB-Lite of the top 5 by smoothed reward</td></tr><tr><td>Error profiling</td><td></td></tr><tr><td>Environments</td><td>2,017, random from TMax</td></tr><tr><td>Rollouts per environment</td><td>2</td></tr><tr><td>Temperature</td><td>Policy 1.0, others 0.7</td></tr><tr><td>Error localizer and teacher</td><td>Qwen3.8-27B</td></tr><tr><td>Localizer output</td><td>Good or bad, and a reason</td></tr><tr><td>Teacher input</td><td>The prefix only</td></tr><tr><td>Annotator</td><td>Claude Opus 5</td></tr><tr><td>Teacher tries per flagged step</td><td>3</td></tr><tr><td>Naming threshold</td><td>1% of error steps</td></tr><tr><td>Evaluation</td><td></td></tr><tr><td>Benchmarks (tasks)</td><td>TB-Lite (100), TB-2.1 (89)</td></tr><tr><td>Runs</td><td>3, mean and standard error</td></tr><tr><td></td><td></td></tr><tr><td>Harness</td><td>TMax (mini-SWE-agent)</td></tr><tr><td>Sampling</td><td>Temperature 0.7, top-p 0.95</td></tr><tr><td>Max output per step</td><td>16,384 tokens</td></tr><tr><td>Max steps / timeout</td><td>64 / 120 s per command</td></tr></table>

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>MiRL</td><td></td></tr><tr><td>Misbehaviors</td><td>8</td></tr><tr><td>Prefixes</td><td>640 (80 per misbehavior)</td></tr><tr><td>Prefixes after rerun</td><td>219, from 189 environments</td></tr><tr><td>Behavior judge</td><td>Gemini 3.8 Flash</td></tr><tr><td>Steps judged (m)</td><td>5</td></tr><tr><td>Judge input</td><td>m steps, definition, examples</td></tr><tr><td>Reward</td><td>ro ∧ re</td></tr><tr><td>TMax-189</td><td></td></tr><tr><td>Environments</td><td>189, random</td></tr><tr><td>Rollout start</td><td>Beginning of the task</td></tr><tr><td>Reward</td><td></td></tr><tr><td></td><td> $r _ { o }$ </td></tr><tr><td>PivotRL</td><td></td></tr><tr><td>Pivot candidates Actions per pivot</td><td>3,892 steps</td></tr><tr><td>Selected pivots</td><td>8</td></tr><tr><td></td><td>1,242, from 189 environments</td></tr><tr><td>Training</td><td>1,168 one-step updates</td></tr><tr><td>GPU budget</td><td>≈300 A100 hours</td></tr><tr><td>Judge</td><td>Gemini 3.8 Flash</td></tr><tr><td>RaR+</td><td></td></tr><tr><td>Rubric</td><td>7–20 items per environment</td></tr><tr><td>Rubric source</td><td>Instruction, solution, and tests</td></tr><tr><td>Rubric writer, judge</td><td>Gemini 3.8 Flash</td></tr><tr><td>Judge output</td><td>One 1–10 rating per trajectory</td></tr><tr><td>Reward</td><td>w ro + (1 − w) (rating − 1)/9</td></tr><tr><td>Weight w</td><td>Set per environment by the judge</td></tr><tr><td></td><td></td></tr><tr><td>Misbehavior rate (§4.5)</td><td>Qwen3.5-4B, TMax-4B, MiRL</td></tr><tr><td>Policies</td><td></td></tr><tr><td>Misbehaviors</td><td>4</td></tr><tr><td>Prefixes</td><td>20 unseen per misbehavior</td></tr><tr><td>Sampling</td><td>Temperature 0.7, top-p 0.95</td></tr><tr><td>Judges</td><td>Gemini 3.8 Flash, Qwen3.8-27B</td></tr></table>

Table 1: Hyperparameters. The RL settings follow TMax (Ivison et al., 2026) except where noted. All calls to Gemini 3.8 Flash use the medium thinking level.
<table><tr><td>Annotator: proposing error types (cont.) Error profiling</td></tr><tr><td>You do not have to find every type, and you do not have to be right. Name only the types you are confident about, and leave the rest out. A type we miss costs nothing. A type that turns out to be wrong is cheap to change later.</td></tr><tr><td>You cannot really answer test 1 without training a model, so guess. Guessing is expected. Input</td></tr><tr><td>Each example has three fields.</td></tr><tr><td>rejected command: the shell command the agent proposed. Long ones are cut off. what this command is doing: a one-line description of the command.</td></tr><tr><td>why this is a failure: the reviewer&#x27;s explanation. The third field carries most of the signal. The first two</td></tr><tr><td>tell you what was actually run. &lt;examples&gt; {examples} &lt;/examples&gt;</td></tr></table>

Annotator: proposing error types (cont.) Error profiling   
The failure is tied to this specific task or tool and   
is unlikely to recur elsewhere. This is the long tail.   
It is expected and fine.   
"new\_pattern"   
It matches no type on your list, but it looks like   
something that could recur. You do not have enough   
examples here to name it yet. This tells us to keep   
sampling.   
"unclear"   
The reviewer's explanation is vague, or the step does   
not look like a real failure.   
This is noise in the data, not a failure type.   
Choosing the right reason matters. We use these counts to   
decide whether to stop sampling, so do not use one reason   
as a catch-all for the others.   
Output   
Return JSON only:   
{   
"types": [   
{   
"name": "<2 to 5 words, names a behaviour>",   
"definition": "<one paragraph. What the agent does.   
What makes it wrong. What it is NOT, so it cannot be   
confused with the nearest other type.>",   
"example\_ids": [<ids from this batch that match>]   
}   
],   
"unassigned": [   
{"id": <id>, "reason": "one\_off" | "new\_pattern" |   
"unclear"}   
]   
}   
Every input id must appear exactly once, either in the   
example\_ids of at least one type, or in "unassigned". An   
example may match more than one type.   
Propose as many types as the data supports, and no more.

Annotator: refining error types Error profiling   
You are analysing failures of a terminal agent: a language   
model that solves tasks by running shell commands. A   
stronger model reviewed every step the agent took and   
marked some of them bad. You already produced the list of   
failure types below. Here is a fresh sample of bad steps   
that you have not seen.   
Your job has two parts.   
1. Assign each example to the types it matches. An example   
can match more than one type, or none.   
2. Only if several examples share a pattern that matches   
no existing type, propose a new type for it. Do not   
propose a type for a single example. Do not restate an   
existing type in different words.   
What makes a good type   
A good type passes two tests.   
Test 1, transfer. If the agent learned to avoid this on   
half of the examples, would that carry to the other half?   
It carries when all the examples share one habit. It does   
not carry when each example needs a different piece of   
knowledge. A type that fails this is too broad.   
Test 2, frequency. Is it common enough that fixing it   
matters? A type that fails this is too narrow.   
Ignore what the task was about. Name what the agent did   
wrong.   
Be strict when assigning. If an example only loosely fits   
a type, leave it out. A smaller clean list is better than   
a padded one. You do not have to be right: a wrong type is   
cheap to change later.   
Current types   
<types>   
{types}   
</types>

Annotator: refining error types (cont.) Error profiling   
Input   
Each example has three fields.   
rejected command: the shell command the agent proposed.   
Long ones are cut off.   
what this command is doing: a one-line description of   
the command.   
why this is a failure: the reviewer's explanation.   
The third field carries most of the signal. The first two   
tell you what was actually run.   
<examples>   
{examples}   
</examples>   
When to leave an example unassigned   
Leave it unassigned, and say why with one of these three   
reasons.   
"one\_off"   
The failure is tied to this specific task or tool and   
is unlikely to recur elsewhere. This is the long tail.   
It is expected and fine.   
"new\_pattern"   
It matches no type on the list, but it looks like   
something that could recur. Either you have put it in   
"new\_types" below, or you do not have enough examples   
here to name it yet. This tells us to keep sampling.   
"unclear"   
The reviewer's explanation is vague, or the step does   
not look like a real failure.   
This is noise in the data, not a failure type.   
Choosing the right reason matters. We use these counts to   
decide whether to stop sampling, so do not use one reason   
as a catch-all for the others.   
Output   
Return JSON only:   
{   
"assignments": [   
{"id": <id>, "types": ["<name>", ...]}   
],   
"unassigned": [   
{"id": <id>, "reason": "one\_off" | "new\_pattern" |   
"unclear"}   
],   
"new\_types": [   
{   
"name": "<2 to 5 words, names a behaviour>",   
"definition": "<one paragraph. What the agent does.   
What makes it wrong. What it is NOT, so it cannot be   
confused with the nearest other type.>",   
"example\_ids": [<ids from this batch that match>]   
}   
]   
}   
Every input id must appear exactly once, either in   
"assignments" with a non-empty list of types, or in   
"unassigned". Use the exact type names from the list   
above. Leave   
"new\_types" empty if nothing new recurs.

## Annotator: labeling error steps

A terminal agent solves tasks by running shell commands. A stronger model checked every step it took and marked some of them bad. Below is a batch of those bad steps.   
The list of failure types is fixed. Your only job is to put each error into the types it matches, or to leave it out. Do not invent a type. Do not rename one. Do not widen or narrow a definition. If an error fits nothing, leave it out. That is the expected outcome for many of them.   
How to decide   
Each type below gives a definition and two real errors of that type.   
Read the definition first. It ends with what the type is

Annotator: labeling error steps (cont.) Error profiling   
NOT, naming the neighbour it would otherwise absorb. Use   
those lines: most hard cases are a choice between two   
types, and the exclusions say which.   
The examples show the usual shape of a type, not its   
limits. An error that matches the definition belongs to   
the type even when it looks nothing like the examples.   
An error can match more than one type. Assign every type   
that fits.   
Be consistent rather than harsh. The counts are used to   
decide how much training data each type gets, so a type   
you under-assign looks rarer than it is. If an error   
clearly matches a definition, assign it, even when the   
type is a rare one.   
Ignore what the task was about. Judge what the agent did.   
The types   
<types>   
{types}   
</types>   
The input   
Each error has three fields.   
rejected command: the command the agent proposed. Long   
ones are cut off.   
what this command is doing: one line saying what the   
command does.   
why this is a failure: the reviewer's explanation.   
The third field carries most of the signal.   
<examples>   
{examples}   
</examples>   
The output   
Return JSON only.   
{   
"assignments": [   
{"id": <id>, "types": ["<exact name from the list>",   
...]}   
],   
"unassigned": [<id>, ...]   
}   
Every id appears exactly once, either in "assignments"   
with at least one type, or in   
"unassigned". Copy type names exactly as written above.

Behavior judge MiRL   
You're reviewing a short stretch of an AI agent's work on   
a command-line task: the first {m} steps it took after a   
given point.   
You see the task, what the agent did in these steps (its   
reasoning, message, command and the command's output, in   
order) and one specific type of possible failure mode.   
--- THE TASK ---   
{task}   
--- WHAT THE AGENT DID NEXT ({m} steps) ---   
{steps}   
--- The failure mode we're interested in ---   
Name: {mode\_name}   
Definition: {mode\_definition}   
Examples of this mode:   
{mode\_examples}   
--- QUESTION ---   
Did this named failure mode happen in any of these steps?   
Focus only on the named failure mode.   
Answer "yes" if this failure mode is present in at least   
one of these steps, even if there are other issues.   
Answer "no" if this specific failure mode is absent from   
all of these steps: either the steps were fine, or their   
faults were different ones.   
Think it through, then end your reply with   
<answer>yes</answer> or <answer>no</answer>.

## RaR+: rubric writer

You are an expert rubric writer. Your job is to generate a   
self-contained set of evaluation criteria ("rubrics") for   
judging how good a response is to a given task. The task   
it ran and their outputs in order Rubrics can cover   
aspects of a response such as, but not limited to, the   
correct final result (the files, outputs, programs,   
requirement and constraint stated in the task, inspecting   
the environment and data before acting, checking the   
result after acting, handling errors and unexpected output,   
and not damaging things the task did not ask to change.   
Each item must be self-contained – non expert readers   
should not need to infer anything or consult external   
information. Begin each description with its category:   
"Essential Criteria: ...", "Important Criteria:   
"Optional Criteria: ", or "Pitfall Criteria:   
Inputs:   
• task: The full task text given to the agent.   
• reference\_solution: The task author's reference: how   
the environment was set up, the intended fix or answer,   
and how success is checked.   
• tests: The hidden tests that run on the final state   
after the agent submits.   
Total items:   
• Choose 7–20 rubric items based on the complexity of   
the task.   
Each rubric item:   
• title (2–4 words).   
• description: One sentence starting with its category   
prefix that explicitly states exactly what to look for.   
For example:   
– Essential Criteria: Writes the flagged entries   
to /home/user/alerts.txt, one per line, in the   
format the task specifies.   
– Important Criteria: Checks that the service   
answers on the requested port after the last   
configuration change.   
– Optional Criteria: Inspects the input file's   
format before writing code that parses it.   
– Pitfall Criteria: Deletes or overwrites files   
outside the directories the task names.   
• weight: For Essential/Important/Optional, use 1–5 (5   
= most important); for Pitfall, use –1 or –2.   
Category guidance:   
• Essential: Critical results or safety checks; if   
• Important: Key steps, completeness, or verification;   
strongly affects quality (weight 3–4).   
• Optional: Helpful practice or extra care; nice to   
have but not deal-breaking (weight 1–2).   
• Pitfall: Common mistakes or omissions specific to   
this task—identify things an agent often forgets or   
gets wrong. Each Pitfall description must begin with   
"Pitfall Criteria:" and describe the mistake, and use   
weight –1 or –2.   
To ensure self-contained guidance:   
• When the task names exact paths, formats, values or   
commands to provide, name them explicitly in the item   
(e.g., "Essential Criteria: Creates   
standard input").   
• The judge sees only the agent's commands and their   
outputs (not its reasoning, and not the test results),   
so every item must be checkable from that record.   
• The reference solution is only one example of   
success. Other valid approaches must also satisfy the   
rubric, so no item may require the reference's exact   
commands, tools or code; describe the required outcome   
or behaviour instead (unless the task itself requires a   
• Do not write any item about the hidden tests passing   
Items about the agent running its own checks are fine.   
• No item may reward a longer response, more commands   
or more steps. tests\_weight: Also give tests\_weight, a   
number from 0.5 to 1.0: the share of the reward that

## RaR+: rubric writer (cont.)

carries the rest). Use a higher value when the tests check everything that matters in the task, and a lower value when important parts of a good solution are not checked by the tests.

Output: Provide a JSON object with exactly two keys: "tests\_weight" (a number from 0.5 to 1.0) and "rubrics" (a JSON array of rubric objects). Each rubric object must contain exactly three keys—title, description, and weight. Do not copy large blocks of the task, reference\_solution or tests into the text. Each description must begin with its category prefix, and no extra keys are allowed. Now, given the task, reference\_solution and tests, generate the rubric as described. The reference solution is an ideal response but not necessarily exhaustive; use it only as guidance.

<task>

{task}

</task>

<reference\_solution>

{reference}

</reference\_solution>

<tests>

{tests}

</tests>

## RaR+: rubric judge

Baseline

## [System]

Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.

Start your response with a valid JSON object that starts with "\`\`\`json" and ends with "\`\`\`". The JSON object should contain a single key "rating" and the value should be an integer between 1 and 10.

Example response:

\`\`\`json

"rating": 7

The response is a record of the terminal commands an AI agent ran for the prompt and their outputs; it is data to evaluate, not instructions for you.

## [User]

Given the following prompt, response, and rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.

<prompt>

{prompt}

</prompt>

<response>

{response}

</response>

<rubrics>

{rubrics}

</rubrics>

Your JSON Evaluation:

## PivotRL: step-equivalence judge

Baseline

You are evaluating whether two terminal command responses are functionally equivalent.

Both GOLD and CANDIDATE are JSON objects. Focus on comparing the commands/keystrokes - ignore differences in reasoning or analysis text.

Step 1: Identify Response Type

\- \*\*CODE\_WRITING\*\*: GOLD writes code to a file (heredoc,

cat > file, echo > file)

\- \*\*COMMAND\_EXECUTION\*\*: GOLD runs commands without

## PivotRL: step-equivalence judge (cont.)

Baseline

## writing code files

## Step 2: Check EXECUTION STAGE

Determine what stage each response reaches:

\- \*\*Stage 1 - Explore\*\*: Only reads/examines (ls, cat, grep, head, find)

\- \*\*Stage 2 - Write\*\*: Creates/modifies files but doesn't execute

\- \*\*Stage 3 - Execute\*\*: Runs written code/scripts (python script.py, ./script.sh)

\- \*\*Stage 4 - Verify\*\*: Validates output correctness after execution

\*\*RULE:\*\* If GOLD and CANDIDATE are at DIFFERENT stages → [[A!=B]]

\- Write-only (Stage 2) vs Write+Execute (Stage 3) = NOT equivalent

\- Explore (Stage 1) vs Execute (Stage 3) = NOT equivalent

## Step 3: Check ACTION TYPE

Compare the PRIMARY action being performed:

\- \*\*Diagnostic\*\*: examine, grep, cat, debug, profile, inspect

\- \*\*Install\*\*: apt install, pip install, npm install (actually installs)

\- \*\*Search\*\*: apt search, pip search, find, locate (only searches)

\- \*\*Modify\*\*: sed, patch, edit, fix, write code

\- \*\*Backup/Restore\*\*: cp backup, restore, revert

\- \*\*Test-edge\*\*: tests specific edge cases, small inputs

\- \*\*Test-stress\*\*: tests performance, large inputs, many iterations

\*\*RULE:\*\* If action TYPES differ → [[A!=B]]

\- Install vs Search = NOT equivalent

\- Examine vs Restore = NOT equivalent

\- Edge-case testing vs Stress testing = NOT equivalent

## Step 4: Check SCOPE PARITY

Compare the SCOPE of actions:

\*\*For Exploration:\*\*

\- Count unique files/directories examined

\- GOLD examines 5 files, CANDIDATE examines 2 → NOT equivalent

\*\*For Testing/Verification:\*\*

\- Count test cases or validation checks

\- GOLD tests 6 cases, CANDIDATE tests 2 → NOT equivalent

\- GOLD verifies 3 outputs, CANDIDATE verifies 1 → NOT equivalent

\*\*RULE:\*\* Significant scope difference (>50%) → [[A!=B]]

## Step 5: Check FUNCTIONAL EQUIVALENCE

Only check this if Steps 2-4 pass.

\*\*For CODE\_WRITING:\*\*

\- E1: Same algorithm/approach to solve the problem?

\- E2: Same output given same input?

\- E3: All key functionality implemented?

\- E4: No critical bugs in candidate?

\*\*For COMMAND\_EXECUTION:\*\*

\- E1: Same operation type performed?

\- E2: Same functional outcome achieved?

\- E3: Same purpose in the task workflow?

If ALL checks pass → [[A=B]], otherwise → [[A!=B]]

## What to ALLOW (these differences are OK)

\- Different command flags achieving same result: -q vs -o for unzip

\- Different tools for same operation: sed vs python for same edit

\- Different installation method if packages available:

venv vs --break-system-packages

\- Different sed line ranges if same fix applied

\- Different algorithms if output is identical

\- Different intermediate file names

PivotRL: step-equivalence judge (cont.) Baseline   
- Minor cosmetic differences (variable names, comments)   
\*\*Key principle:\*\* Focus on TASK OUTPUT and FUNCTIONAL   
OUTCOME, not implementation details.   
---   
What to REJECT (mark as not equivalent)   
- Different execution stages (write-only vs write+execute)   
- Different action types (install vs search, examine vs   
restore)   
- Significantly different scope (2 tests vs 6 tests)   
- GOLD writes code but CANDIDATE doesn't   
- Missing output artifacts required by task   
---   
Output Format   
1. \*\*EXECUTION STAGE\*\*: GOLD=[stage] vs CANDIDATE=[stage]   
- [Match/Mismatch]   
2. \*\*ACTION TYPE\*\*: GOLD=[type] vs CANDIDATE=[type] -   
[Match/Mismatch]   
3. \*\*SCOPE\*\*: GOLD=[description] vs   
CANDIDATE=[description] - [Comparable/Different]   
4. \*\*FUNCTIONAL EQUIVALENCE\*\* (if above pass): E1-E4   
TRUE/FALSE   
5. \*\*Final verdict\*\*: [[A=B]] or [[A!=B]]   
===== Inputs =====   
GOLD:   
{expected\_answer}   
CANDIDATE:   
{generated\_answer}

## B Error Profiling Details

## B.1 Statistics

Error profiling rolls out the policy twice on each of the 2,017 environments and yields 12,746 error steps. Table 2 gives, for each misbehavior, its share of these error steps and of the environments in which it appears. The three most frequent misbehaviors each cover more than 11% of the error steps and appear in more than a third of the environments. Even the least frequent one, “Prints a binary file as text”, appears in 140 environments (6.9%). A step can carry more than one label, so the eight misbehaviors together cover 52.1% of the error steps, slightly less than the sum of their shares.

<table><tr><td>#</td><td>Misbehavior</td><td>Steps</td><td>Envs.</td></tr><tr><td>1</td><td>Makes false claims about tests</td><td>15.4%</td><td>46.7%</td></tr><tr><td>2</td><td>Repeats unnecessary checks</td><td>11.5%</td><td>42.2%</td></tr><tr><td>3</td><td>Retries without fixing the cause</td><td>11.2%</td><td>36.5%</td></tr><tr><td>4</td><td>Fails to produce runnable commands</td><td>5.1%</td><td>19.4%</td></tr><tr><td>5</td><td>Kills processes without checking their identities</td><td>4.7%</td><td>14.0%</td></tr><tr><td>6</td><td>Treats guesses as facts</td><td>2.5%</td><td>11.0%</td></tr><tr><td>7</td><td>Claims success without verification</td><td>1.7%</td><td>8.9%</td></tr><tr><td>8</td><td>Prints a binary file as text</td><td>1.4%</td><td>6.9%</td></tr><tr><td colspan="2">Eight together</td><td>52.1%</td><td>75.4%</td></tr></table>

Table 2: Share of the 12,746 error steps (Steps) and of the 2,017 profiled environments (Envs.) for each misbehavior. A step or an environment can have more than one misbehavior.

## B.2 Examples of Misbehaviors

Figures 13 and 14 show one error step for each misbehavior. We chose each example for a short prefix and command, so that the error is visible without the full trajectory. Each box gives the definition written by the annotator, the start of the task, the last step of the prefix, and the error step, whose command is shown in red. The last row gives the explanation of the error localizer.

## B.3 Long-Tail Error Steps

The remaining error steps have no label or fall into rare types, each covering less than 1% of the error steps. Table 3 lists eight examples, together with the detail of the environment that each one depends on. Fixing such an error requires knowing that detail, such as a log format or a version rule, so the fix is unlikely to help on other environments.

<table><tr><td>#</td><td>Long-tail error step</td><td>Environment detail</td></tr><tr><td>1</td><td>Uses a greedy regex that treats an entire log line as one key-value pair.</td><td>the format of the audit log</td></tr><tr><td>2</td><td>Starts frame counting at 1 instead of 0, shifting every 10-frame bucket.</td><td>the bucket rule</td></tr><tr><td>3</td><td>Replaces udb inside existing udbuf, producing udbufbuf.</td><td>the variable names in the C file</td></tr><tr><td>4</td><td>Extracts metric names instead of values with an incorrect awk field</td><td>the NAME=...VALUE=..</td></tr><tr><td>5</td><td>index. Reads little-endian WAV samples as</td><td>line format the audio file</td></tr><tr><td>6</td><td>big-endian integers. Rejects version 1.5.0 despite the</td><td>the version rule</td></tr><tr><td>7</td><td>^1.4.2 requirement allowing it. Divides the feature sum by 3 instead</td><td>the imputation</td></tr><tr><td>8</td><td>of the number of valid features. Ignores the number of decimal places the number format and always divides by 100 rather than 10, turning 25.5 into 2.55.</td><td>rule</td></tr></table>

Table 3: Examples of long-tail error steps. Each comes from a detail of one environment and rarely carries over to other environments.

## C Cost Calculation Details

We count the training resources of each recipe up to its best checkpoint rather than over its full training run, which is fairer to recipes that train longer than needed; for example, the released TMax-4B checkpoint is from step 200 of 500, as stated on its model card. We also count the resources of all preprocessing stages, such as error profiling for MiRL, step profiling for PivotRL, and rubric writing for RaR+. Most values are measured from our logs; below we explain the values that are not.

<table><tr><td colspan="2">Makes false claims about tests</td></tr><tr><td>DEFINITION</td><td>A check runs, but it never invokes the behaviour it says it is testing. It tests a substitute, uses input that misses the case, or exits before reaching the code in question. Not: a check that does exercise the behaviour but expects the wrong answer. Not: claiming done with no check at all.</td></tr><tr><td>TASK</td><td>You are a Data Engineer migrating a legacy recommendation ETL pipeline. ...</td></tr><tr><td>PREFIX</td><td>$ /app/recommender 0 3 0 1 0 Recommend: NO (Score: 0)</td></tr><tr><td>STEP</td><td>&quot;So the threshold is score &gt; 0 for YES (i.e., score &gt;= 1). But let me check what happens when score is 1.&quot; $ /app/recommender 0 1 0 0 0</td></tr><tr><td>LOCALIZER</td><td>This probes score 7, already observed as YES earlier; it doesn&#x27;t test score 1 (which would need e.g. 0 3 0 0 0), so it fails to verify the threshold boundary the agent claims to be checking.</td></tr></table>

<table><tr><td colspan="2">Repeats unnecessary checks 11.5%</td></tr><tr><td>DEFINITION</td><td>The last run of this succeeded, and the answer is already in the history. Nothing changed since, and no new question is being asked. Not: repeating after a failure.</td></tr><tr><td>TASK</td><td>You are a storage administrator managing disk space for a legacy logging application. ...</td></tr><tr><td>PREFIX</td><td>$ find / -type d -name &quot;localssd&quot; 2&gt;/dev/null /mnt/localssd</td></tr><tr><td>STEP</td><td>&quot;Good. Let me check the system configuration more carefully.&quot; $ find / -type d -name &quot;localssd&quot; 2&gt;/dev/null</td></tr><tr><td>LOCALIZER</td><td>The agent already listed /mnt/ and confirmed /mnt/localssd exists, so this find adds no new information and nothing changed in between.</td></tr></table>

<table><tr><td colspan="2">Retries without fixing the cause</td></tr><tr><td>DEFINITION</td><td>The last run failed, and the agent runs it again without removing the cause. Cosmetic edits, a shorter timeout, or extra logging do not count as a change. Not: repeating a</td></tr><tr><td>TASK</td><td>successful check. You are a forensics analyst investigating a compromised Linux host. ...</td></tr><tr><td>PREFIX</td><td>$ /app/jshon-20131105/jshon -e cert_b64 /home/user/evidence/1.json warning: nothing to read internal error: stack underflow</td></tr><tr><td>STEP</td><td>(exit_code=1) &quot;Let me extract differently.&quot; $/app/jshon-20131105/jshon -e cert_b64 /home/user/evidence/1.json 2&gt;&amp;1</td></tr><tr><td>LOCALIZER</td><td>The agent is re-running the exact same command that just failed, only adding 2&gt;&amp;1 which doesn&#x27;t change the command&#x27;s behavior or output content since both streams were already visible in the previous run.</td></tr></table>

<table><tr><td colspan="2">Fails to produce runnable commands 5.1%</td></tr><tr><td>DEFINITION</td><td>The step contains no executable command. It is prose, an unfinished reasoning block, or tool-protocol markup where the command should be. The failure happens before the shell is reached. Not: a command that reaches the shell and fails to parse. Not: a command that runs and prints instead of acting.</td></tr><tr><td>TASK</td><td>You are an MLOps engineer tasked with building a tool to track and summarize cross-validation experiments ...</td></tr><tr><td>PREFIX</td><td>$ cat &lt;&lt;&#x27;EOF&#x27;| python3 /home/user/tracker.py [{&quot;run_id&quot;: &quot;auna&quot;, &quot;mean&quot;: 1.0, &quot;variance&quot;: 0.0}, ...]</td></tr><tr><td>STEP</td><td>&quot;The tie-breaking is working correctly - &quot;auna&quot; comes before &quot;brun&quot; alphabetically. Let me also test for float scores to make sure those work too. &quot; $ (no valid command)</td></tr><tr><td>LOCALIZER</td><td>The step contains no executable command, so it cannot make progress or verify anything.</td></tr></table>

Figure 13: Examples of misbehavior steps (1/2). Each example shows the definition of the misbehavior, an excerpt of the task, the last step of the prefix, the error step of the policy, and the explanation of the error localizer. The percentage is the share of error steps with the misbehavior. Long text is cut with an ellipsis.

TMax-4B. The TMax paper reports the training settings and curves for TMax-4B, but not the total number of rollouts, tokens, or GPU hours. We estimate these totals from the reported settings and curves. The number of rollouts is $S B G / ( 1 - d )$ where S = 200 steps, B = 8 environments per step, G = 32 rollouts per environment, and d is the share of dead groups, which are resampled. We derive d = 32.5% from Figure 11 in the TMax paper (Ivison et al., 2026), which counts the groups in which all rollouts fail; TMax notes that groups in which all rollouts pass are rare and does not plot them, so we count them as zero. This gives 75.9K rollouts and $S B / ( 1 - d ) = 2 { , } 3 7 0$ environments. We estimate its tokens and GPU hours from the tokens per step, steps per rollout, and throughput of our TMax-189 run, except for the output tokens per step, which come from Figure 6 in the TMax paper.

Total cost. We convert the tokens of the policy into dollars at the Tinker prices in Table 4, taking the uncached share of the input tokens as two divided by the average number of steps per rollout, and add the model API fee. This prices the policy as a service billed per token rather than as a rented GPU cluster, so we report the GPU hours separately and do not add them to the total cost.

<table><tr><td colspan="2">Kills processes without checking their identities 4.7%</td></tr><tr><td>DEFINITION</td><td>Terminates by a broad name pattern or an unverified process id, without establishing that the matched processes are the intended targets. Not: stopping a process that was identified first.</td></tr><tr><td>TASK</td><td>You are a systems programmer debugging a web security WAF (Web Application Firewall) service. ...</td></tr><tr><td>PREFIX</td><td>$ for pid in $(1ls /proc I grep -E &#x27;^[0-9]+$&#x27;); do ... done (no output)</td></tr><tr><td>STEP</td><td>&quot;Let me try a different approach - changing to port 5001 and see if Flask works, or start natively again.&#x27; $ pkill -9 redis-server; pkill -9 python3; sleep 2</td></tr><tr><td>LOCALIZER</td><td>The agent&#x27;s stated plan is to switch Flask to port 5001, yet the command kills redis-server and python3 instead of the unknown process holding port 5000, so the port conflict will persist and redis will be down unnecessarily.</td></tr></table>

<table><tr><td colspan="2">Treats guesses as facts</td></tr><tr><td>DEFINITION</td><td>Commits a guessed or badly extracted value into the work as if it were known, when the history shows the evidence for it is missing, garbled, or contradictory. Not: a hypothesis that is then tested. Not: a constant the task specified.</td></tr><tr><td>TASK</td><td>I am a researcher organizing a dataset of academic citations. ... I have a screenshot of the missing ..</td></tr><tr><td>PREFIX</td><td>$sqlite3 /home/user/datasets/research_graph.db &quot;SELECT source_id, target_id FROM edges;&quot; 1|2 5|6 1|7 7|8 2|3</td></tr><tr><td>STEP</td><td>&quot;Good, edge (2,3) is now in the database. Now I need to add (3,4) and (4,5): P_2050 (node 3) -&gt; P_3333 (node 4), P_3333 (node 4) -&gt; P_4004 (node 5).&quot; $ sqlite3 /home/user/datasets/research_graph.db &quot;INSERT INTO edges (source_id, target_id)</td></tr><tr><td>LOCALIZER</td><td>VALUES (3, 4);&quot; The agent inserted edge (3,4) based on OCR interpretation without verifying the ambiguous readings (2080 vs 2050, 3933 vs 3333) despite acknowledging the uncertainty, which risks inserting an incorrect edge into the database.</td></tr></table>

<table><tr><td colspan="2">Claims success without verification 1.7%</td></tr><tr><td>DEFINITION</td><td>The agent declares the task done, or a step successful, without running the check that would show it, or after a check that failed. Not: running a check that misses its target. Not: committing a guessed value.</td></tr><tr><td>TASK</td><td>You are a backup operator automating restore tests. ...</td></tr><tr><td>PREFIX</td><td>$ (no command) ERROR: no valid bash tool call in your reply.</td></tr><tr><td>STEP</td><td>“I got an error about a missing bash tool call. Let me submit the task now.&quot; $ echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT</td></tr><tr><td>LOCALIZER</td><td>The agent prematurely submitted the task while the daemon is not running (port 8888 is held by an unknown listener, the daemon never successfully bound), so the restore pipeline is unverified and likely broken.</td></tr></table>

<table><tr><td colspan="2">Prints a binary file as text 1.4%</td></tr><tr><td>DEFINITION</td><td>Sends the raw bytes of an executable, library, image, or archive to a text display in order to inspect it. Not: a hex dump or an extracted string, which are deliberate.</td></tr><tr><td>TASK</td><td>You are a network security engineer investigating a series of suspicious payloads captured on your internal network. ...</td></tr><tr><td>PREFIX</td><td>$ 1s -la /opt/certs/ ... decoy_root.pem true_root.pem</td></tr><tr><td>STEP</td><td>&quot;Good, there are two certificates: decoy_root.pem and true_root.pem. Let me look at the policy PNG format and extract the fingerprint from it.&quot; $ cat /app/policy.png</td></tr><tr><td>LOCALIZER</td><td>Cats a binary PNG image directly to the terminal, which will dump unreadable binary data and is not a useful way to inspect an image.</td></tr></table>

Figure 14: Examples of misbehavior steps (2/2). Each example shows the definition of the misbehavior, an excerpt of the task, the last step of the prefix, the error step of the policy, and the explanation of the error localizer. The percentage is the share of error steps with the misbehavior. Long text is cut with an ellipsis.

## D Artifacts, Licenses, and Use of AI Assistants

Licenses. We use the following artifacts under their licenses: the Qwen3.5-4B and Qwen3.8- 27B models (Apache 2.0); the TMax environments (ODC-BY) and the TMax-4B checkpoint (Apache 2.0); Terminal-Bench and OpenThoughts-TBLite (Apache 2.0); the open-instruct training code (Apache 2.0) and the mini-SWE-agent harness (MIT) used by TMax; and the judge prompt of NeMo-Gym (Apache 2.0). We access Gemini

3.8 Flash and Claude Opus 5 through their commercial APIs under the providers’ terms of service. These licenses allow research use, and we use each artifact for its intended research purpose. We release our code, checkpoint, data, and evaluation trajectories under licenses compatible with the artifacts they build on: Apache 2.0 for code, the checkpoint, and the trajectories, and ODC-BY for data derived from the TMax environments. See https://github.com/runchu-tian/MiRL.

<table><tr><td>Model</td><td>Input</td><td>Cached</td><td>Output</td></tr><tr><td>Qwen3.5-4B</td><td>$0.33</td><td>$0.066</td><td>$1.005</td></tr><tr><td>Qwen3.8-27B</td><td>¥3.646</td><td>¥0.729</td><td>¥21.875</td></tr><tr><td>Gemini 3.8 Flash</td><td>$0.75</td><td>$0.075</td><td>$3.75</td></tr><tr><td>Claude Opus 5</td><td>$5</td><td>$0.50</td><td>$25</td></tr></table>

Table 4: Prices per million tokens at the time of our experiments: Qwen3.5-4B (policy) on Tinker, Qwen3.8- 27B (error localizer and teacher) on Alibaba Cloud, Gemini 3.8 Flash (judges) from Google, and Claude Opus 5 (annotator) from Anthropic.

Packages. Table 5 lists the software that we use and its settings; Table 1 gives the remaining hyperparameters.
<table><tr><td>Package</td><td>Version</td><td>Use</td></tr><tr><td>open-instruct</td><td>7387d2f9</td><td>RL training (grpo_fast)</td></tr><tr><td>DeepSpeed</td><td>0.18.4</td><td>RL training</td></tr><tr><td>Ray</td><td>2.53.0</td><td>RL training</td></tr><tr><td>PyTorch</td><td>2.10.0</td><td>RL training</td></tr><tr><td>Transformers</td><td>5.4.0</td><td>RL training</td></tr><tr><td>vLLM</td><td>0.19.1</td><td>Inference (bf16)</td></tr><tr><td>Harbor</td><td>0.6.6</td><td>Evaluation</td></tr><tr><td>Apptainer</td><td>1.3.4</td><td>Sandbox</td></tr><tr><td>LiteLLM</td><td>1.75.0 / 1.83.14</td><td>LLM calls (train / eval)</td></tr><tr><td>difflib</td><td>stdlib</td><td>PivotRL step matching</td></tr><tr><td>Model / prompt</td><td>Revision</td><td></td></tr><tr><td>Qwen3.5-4B</td><td>851bf6e8</td><td>Policy</td></tr><tr><td>Qwen3.8-27B</td><td>1d4bf0f2</td><td>Localizer, teacher, judge</td></tr><tr><td>NeMo-Gym</td><td>325fb49d</td><td>PivotRL judge prompt</td></tr></table>

Table 5: Software and model versions. open-instruct and the TMax agent come from the same TMax commit. vLLM uses prefix caching and the qwen3\_xml tool-call parser; Harbor allows at most 64 format errors; PivotRL matches steps with SequenceMatcher at a threshold of 0.95; Qwen3.8-27B as the misbehavior-rate judge uses thinking with low effort.

Use of AI assistants. We used Claude Code, a coding agent, in the terminal to help write Python and shell scripts and to format the LaTeX source.