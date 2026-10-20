# Mitigating Reward Hacking in Preference Reward Models with Evolving Penalties

Afonso Baldo<sup>1,</sup> <sup>3</sup>, António Farinhas<sup>1</sup>, Marcos Treviso<sup>2,</sup> <sup>3,</sup> <sup>4,</sup> <sup>5</sup>, Ricardo Rei<sup>1</sup>, Nuno M. Guerreiro<sup>1</sup>

<sup>1</sup>Sword Health <sup>2</sup>Gandara AI <sup>3</sup>Instituto Superior Técnico, Universidade de Lisboa

<sup>4</sup>Instituto de Telecomunicações <sup>5</sup>ELLIS Unit Lisbon

ć ai.research@sword.com

## Abstract

Reinforcement learning for language models has moved from preference reward models to verifiable rewards, which sustain much longer optimization. Preference models are prone to reward hacking: the policy raises their score without improving response quality, or even while degrading it. Yet for open-ended tasks, where quality can neither be verified nor fully written down as rubrics, they remain valuable. We introduce Dynalize, which mitigates reward hacking online, during policy training. Periodically and asynchronously, a language mode analyzes training rollouts and writes penalties for undesirable behaviors that the reward model appears to reward; a judge applies them to every rollout. In clinical counseling and creative writing, training on the reward model alone raises its score continuously but ultimately leads to a decline in external benchmarks. Dynalize sustains longer training and results in improved downstream performance. For example, we find that our method suppresses patterns incentivized by preference models that, without penalties, spread to most responses, and even to the model’s reasoning, which the reward model does not explicitly score. Furthermore, we show that its gains hold in multi-turn training, and even after we retrain the reward model on the discovered behaviors.

## 1 Introduction

Reinforcement learning (RL) is emerging as the de facto paradigm for improving large language models (LLMs), and currently takes a growing share of the compute used to train them — in some cases matching or even exceeding pretraining (Lambert, 2025; Karpathy, 2025; DeepSeek-AI et al., 2025b; xAI, 2025; Cursor, 2026). This scale-up derives, in part, from a shift in reward design: earlier models were mostly trained with preference models, i.e., reward models (RMs) fitted to pairwise human preferences with the Bradley-Terry model (Bradley and Terry, 1952; Christiano et al., 2017; Ouyang et al., 2022), whereas verifiable rewards, such as answer checks in mathematics or unit tests in code, are now the standard (Lambert et al., 2024; DeepSeek-AI et al., 2025a).

Verifiable rewards allow much longer optimization (DeepSeek-AI et al., 2025a; Karpathy, 2025; xAI, 2025) than preference models, which are widely known to be prone to reward hacking (Skalse et al., 2022; Wang et al., 2026b). As optimization progresses, the policy may discover and exploit shortcuts that do not reflect response adequacy and quality (e.g., lexical features or response length; Xia et al., 2025; Srivastava et al., 2025; Wang et al., 2026b), thus increasing reward-model scores without improving, or even while degrading, downstream response quality.

However, not all tasks, particularly open-ended ones such as clinical counseling and creative writing, can easily be put in a form that a program can verify (Jia et al., 2025; Gunjal et al., 2025; Huang et al., 2025; Zhang et al., 2025; Wang et al., 2026a). For these tasks, rubric-based rewards have emerged as an alternative to do away with preference models (DeepSeek-AI et al., 2025b, 2026): reward scores are derived from LLM judges running each response against a predefined set of rubrics (Gunjal et al., 2025; Huang et al., 2025; Kimi Team, 2025, 2026; Qi et al., 2026a) (typically written for each task or instance; Huang et al., 2026; Li et al., 2026). Yet a rubric rewards only what its authors — humans or language models — thought or were prompted to write down (Mahmoud et al., 2026; Yang et al., 2026).<sup>1</sup> We argue that, for such tasks, preference models remain valuable: they can encode information that is hard to write down, and promote and penalize behaviors and properties that are hard to enumerate. And although previous work has attempted to make reward models more robust and less exploitable (Srivastava et al., 2025; Bukharin et al., 2025; Coste et al., 2024; Eisenstein et al., 2024; Duan et al., 2026), it is hard, if not impossible, to rely purely on reward models (and combinations thereof) during training and remain exploit-free.

![](images/382b54a2afceaa0ac0e8d5d17828c7e9889f3d9ef6ffc8369c3e1cef6a7b0b8e.jpg)  
Figure 1: Overview of Dynalize. Full details in Section 3.

As such, in this work, we focus on addressing this gap by exploring methods to mitigate reward hacking of preference models online, adaptively during policy training. We introduce Dynalize, a method that complements a reward model with penalties that evolve during training. Periodically, during the RL climb, a separate language model analyzes policy rollouts and their reward scores, identifies undesirable behaviors that appear to be rewarded by the RM, and writes penalties for them. A dynamic penalty bufer keeps these penalties up to date, adding new ones and discarding those that are no longer triggered by the policy at the time. Importantly, both the proposal and the bufer update run asynchronously, without interrupting policy optimization. Finally, a judge applies the bufer’s penalties to each rollout, so the reward for gradient updates adapts to the reward-hacking patterns observed during training, while the RM still defines response quality.

We evaluate Dynalize in two open-ended tasks: clinical counseling and creative writing. Our contributions are as follows:

<sub>■</sub> We introduce Dynalize, which discovers reward-hacking behaviors during policy training and penalizes them online and asynchronously, while the RM remains the primary signal of response quality (Section 3).

<sub>■</sub> We show that reward models are easily “hacked”. In both tasks, RM-only training keeps raising the RM score while performance on external benchmarks declines. Dynalize not only stops this decline: it allows longer RL climbs that lead to improved downstream performance (Section 5).

<sub>■</sub> We analyze how Dynalize works in practice: which behaviors the proposer discovers, how their prevalence changes once they enter the bufer, and how widespread they are without penalties (Section 5.2).

<sub>■</sub> We examine two potential mitigations: multi-turn RL, where the policy holds several turns of conversation with a simulated user, and reward-model retraining, where the RM is retrained on the behaviors that Dynalize discovered. We find that neither prevents reward hacking, and that Dynalize reaches higher performance than RM-only training in both (Sections 5.3 and 5.4).

We will publicly release the code that led to the experiments in this paper.<sup>2</sup>

## 2 Background

Preference reward models. A preference reward model assigns a scalar score r<sub>RM</sub>(x, y) to a response y given an input x. It is trained on a dataset D of human preferences, where each item holds an input x, a preferred response $y ^ { + }$ and a rejected response $y ^ { - }$ . Under the Bradley-Terry model (Bradley and Terry, 1952), the probability that $y ^ { + }$ is preferred over $y ^ { - }$ is

$$
P ( y ^ { + } \succ y ^ { - } \mid x ) = \sigma \big ( r _ { \mathrm { R M } } ( x , y ^ { + } ) - r _ { \mathrm { R M } } ( x , y ^ { - } ) \big ) ,\tag{1}
$$

where σ is the logistic function. The reward model is trained to maximize the likelihood of the preferences in D (Ouyang et al., 2022). In this paper, we use preference models as the main reward for open-ended tasks, where a response cannot be verified automatically and such models remain attractive.

Preference optimization. Preference optimization, in particular Group Relative Policy Optimization (GRPO; Shao et al., 2024) and its variants, have become a standard component on post-training pipelines of language models (DeepSeek-AI et al., 2025a,b; Liu et al., 2025; Qi et al., 2026b). Under GRPO, for each input x, the policy $\pi _ { \theta }$ samples a group of G responses $y _ { 1 } , \dotsc , y _ { G } ,$ , and each response receives a reward $r _ { i } = r ( x , y _ { i } )$ . The reward can come from a program that verifies the response (Lambert et al., 2024; DeepSeek-AI et al., 2025a), from an LLM judge that scores it against rubrics (Gunjal et al., 2025; Huang et al., 2025), or from a preference reward model. It compares each response with the others in its group through its advantage, the reward of the response relative to the mean reward of the group:

$$
A _ { i } = r _ { i } - \frac { 1 } { G } \sum _ { g = 1 } ^ { G } r _ { g } .\tag{2}
$$

The policy is then updated to raise the likelihood of responses with positive advantage and to lower the likelihood of responses with negative advantage.<sup>3</sup> In this paper, the reward $r _ { i }$ is based on the score of a preference reward model, $r _ { \mathrm { R M } } ( x , y _ { i } )$ ).

## 3 Method

We illustrate our method, Dynalize, in Figure 1. We focus exclusively on using a preference reward model (RM) as the main training signal. The method has two main components:

<sub>■</sub> penalty proposer: a separate language model that identifies emerging reward-hacking patterns in training rollouts and formulates them as negative rubrics.<sup>4</sup> These define behaviors to penalize in subsequent policy updates.

<sub>■</sub> penalty bufer: the set $B _ { t }$ of penalties that are active at training step t. It takes in the new penalties from the proposer and drops those that the policy no longer triggers.

Below, we first define the training reward, and then describe each component.

Every k training steps, the proposer analyzes a sample of rollouts from the previous step and proposes new penalties, which are added to the penalty bufer $B _ { t }$ . For each response y to an input x (the context, prompt or conversation history), an LLM judge checks whether any active penalty applies. The resulting training reward is

$$
r _ { t } ( x , y ) = r _ { \mathrm { R M } } ( x , y ) - \lambda \sum _ { j \in \mathcal { B } _ { t } } h _ { j } ( x , y ) ,\tag{3}
$$

where $h _ { j } ( x , y ) \in \{ 0 , 1 \}$ is the judge’s verdict on whether response y exhibits the behavior of penalty j, and $\lambda > 0$ is the penalty weight. This reward replaces $r ( x , y _ { i } )$ in Equation 2.

Penalty proposer. Prior work shows that LLMs can monitor model behavior to detect reward hacking and other forms of misbehavior(Baker et al., 2025; Korbak et al., 2025). Here, we use a language model to discover reward-hacking patterns in training rollouts and convert them into explicit criteria for detecting these patterns in future rollouts. Specifically, at each proposal step, the proposer receives 32 rollouts randomly sampled from the batch generated at the previous training step, their RM scores, and the active penalties $B _ { t }$ . The rollouts are labeled according to whether their advantage is positive or negative (Equation 2, computed with the RM score). This helps the proposer identify which responses the optimization reinforces and focus on reward-hacking patterns in positive-advantage responses. We do this because identifying patterns in negative-advantage responses is less useful, since these responses are already discouraged by the RM.

Each candidate penalty is expressed as a negative rubric with a short title, a description of the pattern, and references to sampled responses that illustrate it.<sup>5</sup> Appendix E provides the prompts.

Penalty bufer. The bufer stores the active penalties used by the judge at each training step. A penalty is removed only if it is not triggered by any of the sampled rollouts in a given step. Each proposal step adds at most one new penalty. During training, a judge checks each rollout for the behaviors described by the active penalties. For eficiency purposes, particularly as |B<sub>t</sub>| grows, all penalties are evaluated in a single judge pass.

RM only Our Method Best checkpoint  
![](images/931b279d07348652ceb0811b097f8c0c77ba30daa8fda4ed5349cc2aad761d74.jpg)

![](images/af10d8445ef08d39ef41e38652830b30ed4a9b61d31be665d358f1c79545ebb1.jpg)  
(b) Creative writing

Figure 2: RM score and benchmark score over training, for RM-only training and our method.
<table><tr><td></td><td colspan="4">Clinical counseling MindEval</td><td colspan="4">Creative writing EQ-Bench CW v3</td></tr><tr><td>Run</td><td></td><td></td><td>Final</td><td>RM score</td><td>Peak</td><td></td><td></td><td>RM score</td></tr><tr><td></td><td>Peak 3.69</td><td>End 3.39</td><td></td><td></td><td></td><td>End</td><td>Final</td><td></td></tr><tr><td>RM only DYNALIZE</td><td>3.92+0.23</td><td>3.84+0.45</td><td>3.37 3.80+0.43</td><td>1.00 0.92</td><td>58.9 62.1+3.2</td><td>56.0 60.8+4.8</td><td>54.8 60.5+5.7</td><td>12.19 10.93</td></tr></table>

Table 1: Main results. Peak: best checkpoint. End: mean over the checkpoints evaluated for both runs in steps 150–200. Final: step 200. RM score: mean over the End checkpoints. Subscripts: diference from RM only.

Asynchronous updates. We run penalty proposal and bufer management asynchronously to avoid interrupting policy optimization. Training uses the current set of active penalties while new penalties are proposed in the background. Once a bufer update is complete, the updated set takes efect at the next batch of rollouts. This allows us to adapt penalties during training without waiting for proposal or bufer management to finish.

## 4 Experimental Setup

We evaluate Dynalize on two open-ended tasks: clinical counseling and creative writing. We match each task with a relevant reward model, which provides the training signal, and with a relevant external benchmark, on which we evaluate policy checkpoints. We use the benchmark as an estimate of the true response quality. This allows us to tell reward hacking apart from downstream quality improvements: as in prior work on learned rewards (Gao et al., 2022; Yang et al., 2026), we take a benchmark score that declines while the RM score keeps increasing as the signal of reward hacking. For each task, we compare RM-only training, where the reward is the RM score r<sub>RM</sub>(x, y), with Dynalize (Equation 3).

Clinical counseling. In this task, the policy acts as a clinical counselor: given a multi-turn conversation with a patient, the model is tasked with writing the next reply. The prompts for reinforcement learning derive from a private dataset of 10,000 conversation samples. The reward model is a proprietary Bradley– Terry model (Section 2) trained on preferences derived from annotations from clinical psychologists: a higher score indicates a reply that a psychologist is more likely to prefer. We evaluate on MindEval (Pombal et al., 2025), a benchmark of multi-turn mental health support conversations. Appendix A describes the exact setup.

Creative writing. In this task, the policy writes a short story for a given prompt. We use the prompts and the reward model of LitBench (Fein et al., 2026). We train on the 12,414 unique prompts of LitBench-Train. The reward model is Llama8B-CreativeWritingVerifier, a Bradley–Terry model trained on human preferences over stories. We evaluate policy checkpoints on EQ-Bench Creative Writing v3 (Paech, 2023); Appendix A describes the exact setup.

Training configuration. For each task, we train a Qwen3.5-35B-A3B (Qwen Team, 2026a)<sup>6</sup> model for 200 steps with the RL setup described in Section 2. Each step uses a batch of 512 rollouts: 32 prompts with a group of G = 16 responses each, sampled at temperature 1. We update all the weights of the policy with AdamW, with a constant learning rate of $1 0 ^ { - 6 }$ and a weight decay of 0.01. Rollout generation and training run asynchronously, and we discard rollouts that are more than eight steps behind the current policy.

![](images/2d0494efc777656daa10a70fd204052c87eed7d6fdb76304c181df2a3a1e4bfd.jpg)  
Figure 3: The 20 penalties discovered in clinical counseling: their prevalence in training rollouts while in the bufer, with the steps at which each was added and retired (left), and in held-out replies at the end of training (right).

Penalty configuration. The penalty proposer is GLM 5.2 (GLM-5 Team, 2026). Because the proposer runs asynchronously and does not slow down training, we can use a large, capable model to discover reward-hacking patterns. It runs every k = 5 training steps on 32 rollouts from the previous batch: two rollouts sampled at random from each of 16 groups. A candidate penalty is kept only if the proposer finds evidence for it in at least two groups. The judge that returns the verdicts $h _ { j }$ is Qwen3.6-35B-A3B (Qwen Team, 2026b). In contrast to the proposer, the judge runs on every rollout, so we use a smaller model that keeps judging fast and avoids slowing down training. We set the penalty weight to λ = 0.5 in Equation 3.

## 5 Results and Analysis

## 5.1 Dynalize mitigates reward hacking

We can see that, for both clinical counseling and creative writing, RM-only training leads to steady increases in the respective RM scores during training, usually dominating those of our method (Figure 2). However, importantly, response quality (as measured by the performance on MindEval and EQ-Bench, respectively) starts degrading — this is the signal of reward hacking. This is not the case with our method, where response quality keeps improving for much longer. Dynalize not only allows us to extend training: it keeps optimizing for the reward model without sacrificing performance. In fact, in both tasks, it ends with a benchmark score superior to that of RM-only training, not only at its end — where hacking is most prominent — but also at its peak (Table 1). One hypothesis is that the penalties keep the policy from being steered towards artifacts or spurious features that the RM “likes”. We will analyze this next.

## 5.2 What does the proposer discover?

Now we turn to analyzing the behaviors that were being penalized by our method during training.

Figure 3 illustrates the penalty bufer during training for the clinical counseling task.<sup>7</sup>

Some behaviors are common when first discovered but become substantially less frequent after being penalized. The most common ones, such as those targeted by penalties 1, 5, and 6, do not disappear completely: they persist at low rates until the end of training, and their penalties stay in the bufer. In contrast, less frequent behaviors, such as those targeted by penalties 15–19, disappear within a few training steps, and their penalties are retired. Because some behaviors persist, the bufer tends to grow during training.

<table><tr><td>Run</td><td>Clinical accuracy &amp; competence</td><td>conduct</td><td>Ethical &amp; prof. Assessment &amp; response</td><td>Therapeutic rel. &amp; alliance</td><td>AI-specific comm. quality</td><td>Overall</td></tr><tr><td>RM only, end</td><td>3.54</td><td>3.98</td><td>3.37</td><td>3.51</td><td>2.52</td><td>3.39</td></tr><tr><td>RM only, peak</td><td> $3 . 6 6 _ { + 0 . 1 2 }$ </td><td> $4 . 1 3 _ { + 0 . 1 5 }$ </td><td> $3 . 6 0 _ { + 0 . 2 3 }$ </td><td> $3 . 7 6 _ { + 0 . 2 5 }$ </td><td> $3 . 2 9 _ { + 0 . 7 7 }$ </td><td> $3 . 6 9 _ { + 0 . 3 0 }$ </td></tr><tr><td>DYNALIZE, end</td><td> $\mathbf { 3 . 7 3 _ { + 0 . 1 9 } }$ </td><td> $\mathbf { 4 . 3 2 _ { + 0 . 3 4 } }$ </td><td> $\mathbf { 3 . 7 2 _ { + 0 . 3 5 } }$ </td><td> $\mathbf { 3 . 7 9 } _ { + 0 . 2 8 }$ </td><td> $\mathbf { 3 . 6 6 _ { + 1 . 1 4 } }$ </td><td> $\mathbf { 3 . 8 4 _ { + 0 . 4 5 } }$ </td></tr></table>

Table 2: MindEval scores by axis. Peak and End as in Table 1. Subscripts: diference from RM only at the end.

The penalties range from surface-level, lexical patterns to content adequacy and quality.

The same response patterns are common and sticky under RM-only training. To assess whether these patterns also emerge without penalties, we evaluate the outputs of the RM-only policy at the end of training against all criteria discovered by our method, with the same judge and the same held-out prompts as for Dynalize. For example, in clinical counseling (right panel of Figure 3), over 90% of RM-only replies trigger at least one criterion, with over four criteria per reply on average, against fewer than two for Dynalize. These results suggest that the policy heavily exploits these patterns when trained without penalties. They also suggest that the patterns are sticky: under RM-only training nothing penalizes them, and in clinical counseling the average number of criteria per reply almost triples from the start to the end of training.

Without penalties, one pattern can spread to most responses. One of the main problems with reward hacking is that patterns start to show up in a large number of responses, even under diferent contexts. As we have noted in Section 3, one behavior in itself (e.g., a stacked multi-part question in clinical counseling) is not necessarily damaging to the good conduct of a session; it becomes a problem, however, when the same pattern shows up in lots of diferent contexts.<sup>8</sup> Repeated patterns across the outputs of a model are well documented (Kirk et al., 2024; Xu et al., 2025; Paech et al., 2026). Identifying a persistent behavior can be more or less dificult, with some being more evident and others requiring many diferent scenarios to surface (Gupta et al., 2025). We analyze this by running diferent training checkpoints on a set of held-out prompts.<sup>9</sup> For example, we found that, in clinical counseling, the share of replies that start with “Okay” grows from almost none to about 60% under RM-only training. Under Dynalize, it grows in the same way until a penalty that covers it explicitly enters the bufer and starts penalizing these replies. In creative writing, the RM-only policy collapses to one theme regardless of the prompt: at the end of training, almost all of its stories contain the word “galactic” (see Figure 4).

Hacking patterns are also present in the model reasoning. Reward hacking is typically analyzed in final responses only, as most reward models score the final response and ignore the reasoning. However, in autoregressive generation, the reasoning is part of the context that conditions the final response, so patterns that are rewarded in responses may transfer to the reasoning. This matters because reasoning traces are used to monitor and interpret models (Baker et al., 2025; Korbak et al., 2025), and collapsed reasoning may hurt both monitoring and generalization. We found that repeated patterns indeed emerge not only in the final response but also in the reasoning traces. For example, under RM-only training in creative writing, the share of reasoning traces that name the source of the prompts (r/WritingPrompts) grows from 1% to 59% at the end of training (see Figure 4).

Responses can combine multiple patterns. RM-only responses are also longer and tend to combine several of these patterns, such as starting with “Okay” and ending with a question that ofers two options (Figures 8–11 compare rollouts from both policies). MindEval captures these patterns in one of its axes (Table 2). Under RM-only training, the largest decline is in AI-Specific Communication Quality, the axis that rates whether replies are natural and free of repeated phrases and of templated or formulaic structure. Dynalize is ahead on every axis, and by far the most on this one.

![](images/94d30697759902aee1743b85e257cd4a528dab77b1fbcce8374be9e3d75239da.jpg)

![](images/dee2bef4d5e8bd67db331d8383d1716f7d1137ae32ff8f3222aacc2abb809d6e.jpg)  
RM only Our Method

![](images/3d762c77dc0682fdd1dce9ccea38774227c380b03e23e28e174f0b7def14af79.jpg)  
Figure 4: Repeated patterns over training, on held-out prompts, for RM-only training and our method.

## 5.3 Multi-Turn RL

So far, we have trained with single-turn RL: the policy   
writes one response to a conversation history taken from   
the dataset, and the reward model scores that response.   
Policies trained on such next-turn rewards are known to   
optimize the immediate turn at the expense of the con  
versation as a whole (Shani et al., 2024; Wu et al., 2025).   
Multi-turn RL changes two things. First, the policy must   
continue its own conversations (and potentially “hacked”   
turns), so the reward model sees histories that contain the   
policy’s own responses. For example, we observed that   
Figure 5: RM score and MindEval in multi-turn RL.replies frequently began with “Okay”; if this repetition ap- replies frequently began with “Okay"; if this repetition appears in the conversation history, the reward model may pears in the conversation history, the reward model may

![](images/8044970d9369a96d561c8bc37f5c29b0a22e9d1efe397260078206f2f6250025.jpg)

be able to penalize it. Second, the policy is optimized on a signal that is spread over a whole trajectory, and is therefore much more dispersed than in single-turn training. We ask whether reward hacking stil occurs under these conditions.

Setup. We focus exclusively on clinical counseling, for its conversational nature. Each rollout starts from a conversation in the dataset. The policy writes a reply, a user simulator (DeepSeek 4.1 Flash) writes the next message of the patient, and the policy replies again, for up to four policy replies per rollout. The reward model scores each policy reply given the conversation up to that reply, and the reward of the rollout is the mean of these scores. We compute one advantage per rollout, relative to its group, and apply it to the tokens of all policy replies in the rollout. With Dynalize, the judge checks every policy reply against the active penalties, and the rollout is penalized once for each distinct penalty that any of its replies triggers. The remaining setup is similar to that of the original clinical counseling described in Section 4.

Results. Figure 5 shows the training and evaluation curves. The curves are similar to those observed in single-turn training: under RM-only training, MindEval performance initially improves but then declines while RM scores continue to increase. With our method, MindEval scores remain high during training. These results suggest that our method is also efective in multi-turn environments under user simulation and that multi-turn RL alone does not prevent reward hacking.

Similar penalties emerge in single-turn and multi-turn training. We found that the penalties discovered during the multi-turn run (see Appendix G.2) are partly the same as those of single-turn training, including stacked questions and option-based question framing. Even in a multi-turn setting, the reward model remains easy to game, and the policy exploits the same patterns observed in single-turn training.

## 5.4 Can additional training data correct reward model biases?

One hypothesis for why reward models are hacked is that they are not robust enough: they reward spurious features, and better training data or better modeling would remove these biases (Srivastava et al., 2025; Bukharin et al., 2025; Coste et al., 2024; Eisenstein et al., 2024; Duan et al., 2026). Dynalize produces what such a fix needs: a list of the behaviors that the policy exploited, and rollouts that show them. Here, we investigate whether we can exploit these penalties and rollouts to train a new reward model that can potentially be more robust to the exploits found during training.

Setup. Consistent with previous analysis, we conduct this experiment only in the clinical environment. To construct

![](images/ac8106d9775498fdb895e2e55c127e0c9ba6a0cd34bf6ff235ea8a3e36e75a09.jpg)  
Figure 6: RM score and MindEval with the new RM.

training data targeting the behaviors listed in Table 5, we use rollouts generated during training (we have at least eight rollouts per prompt). Then, we select as the preferred response one that triggers no penalties and has RM scores above the 75th percentile, ensuring that it is a good response according to the original RM. We then pair each preferred response with a response to the same prompt that triggers at least one penalty. This yields approximately 15,000 new pairs, which we add to the original RM training set of approximately 30,000 samples. The original and new reward models achieve similar mean accuracy across the test set.<sup>10</sup>

The retrained reward model remains vulnerable to reward hacking. Figure 6 shows the training curves and MindEval scores obtained with the newly trained RM. The curves with new RM exhibits the same pattern as before: MindEval scores initially increase and then decline, even as RM scores continue to rise; we have reward hacking. Table 7 lists the penalties discovered during this run, which are mostly diferent from those found with the original reward model.

These results support our motivation: even after correcting specific biases in the reward model, the policy is likely to find other ways to exploit it so we need a method that discover and penalize this problems during training. Our approach is also less costly, as it does not require retraining the RM or restarting policy training.

## 6 Conclusions

We introduced Dynalize, which complements a preference reward model with penalties that are discovered and updated during policy training. In clinical counseling and creative writing, it mitigates reward hacking: it sustains longer training and reaches higher downstream response quality than RM-only training. Our analysis shows that our method suppresses patterns incentivized by the preference model that, without penalties, spread to most responses, and even to the model’s reasoning (which is not explicitly evaluated by the preference model). We also find that responses under our method are more natural and less formulaic than under RM-only training. We further show that neither multi-turn training nor retraining the reward model on the discovered behaviors prevents reward hacking, and that, under both contexts, our method reaches higher downstream performance than RM-only training.

## Limitations

Our work has some limitations that should be considered. First, we study reward hacking when training in a relatively simple setup: a single task with environments that are similar across all training data. An interesting direction for future work is to extend our approach to multi-environment training, where interactions across environments may introduce more complex reward-hacking behaviors that might be more challenging to detect and mitigate.

Furthermore, our method can only detect what models explicitly generate across a series of rollouts. As such, it cannot preemptively prevent hacking from occurring. It can only do so when it is already showing in model generations. Finding traces of hacking that are intrinsic to the model, i.e., present in its internal representations before they show in its generations, is also an interesting direction for future work.

Finally, although penalty proposal and bufer updates are performed asynchronously, applying the penalty judge to every rollout adds computational overhead, although we attempt to minimize this by using a

small model as the judge.

## Ethical Considerations

Clinical counseling is one of the tasks studied in this work, where the model acts as a clinical counselor in a multi-turn conversation with a patient. The training data for this task consist of a private dataset of real human conversations , which raises important ethical considerations about privacy and confidentiality. To protect patient privacy, we anonymized the data and will not make it publicly available.

## References

Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv preprint arXiv:2503.11926, 2025. doi: 10.48550/arXiv.2503.11926. URL https://arxiv. org/abs/2503.11926.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3-4):324–345, 1952. doi: 10.1093/biomet/39.3-4.324. URL https://doi.org/10 .1093/biomet/39.3-4.324.

Alexander Bukharin, Haifeng Qian, Shengyang Sun, Adithya Renduchintala, Soumye Singhal, Zhilin Wang, Oleksii Kuchaiev, Olivier Delalleau, and Tuo Zhao. Adversarial training of reward models. arXiv preprint arXiv:2504.06141, 2025. doi: 10.48550/arXiv.2504.06141. URL https://arxiv.org/abs/2504.06141v1.

Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. arXiv preprint arXiv:1706.03741, 2017. doi: 10.48550/arXiv.1706.03741. URL https://arxiv.org/abs/1706.03741.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr .cc/paper\_files/paper/2024/hash/dda7f9378a210c25e470e19304cce85d-Abstract-Conference.html.

Cursor. Introducing Composer 1.5. Company announcement, 2026. URL https://cursor.com/blog/composer -1-5.

DeepSeek-AI et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a. doi: 10.48550/arXiv.2501.12948. URL https://arxiv.org/abs/2501.129 48.

DeepSeek-AI et al. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025b. doi: 10.48550/arXiv.2512.02556. URL https://arxiv.org/abs/2512.02556.

DeepSeek-AI et al. DeepSeek-V4: Towards highly eficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026. doi: 10.48550/arXiv.2606.19348. URL https://arxiv.org/abs/2606.19348.

Zhibin Duan, Guowei Rong, Zhuo Li, Bo Chen, Mingyuan Zhou, and Dandan Guo. Mitigating reward hacking in RLHF via bayesian non-negative reward modeling. arXiv preprint arXiv:2602.10623, 2026. doi: 10.48550/a rXiv.2602.10623. URL https://arxiv.org/abs/2602.10623.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, Peter Shaw, and Jonathan Berant. Helping or herd ing? reward model ensembles mitigate but do not eliminate reward hacking. In Conference on Language Modeling, 2024. URL https://arxiv.org/abs/2312.09244.

Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, and Nick Haber. LitBench: A benchmark and dataset for reliable evaluation of creative writing. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7740–7755, Rabat, Morocco, 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.eacl-long.362. URL https://aclanthology.org/2026.eacl-long.362/.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. arXiv preprint arXiv:2210.10760, 2022. doi: 10.48550/arXiv.2210.10760. URL https://arxiv.org/abs/2210.10760.

GLM-5 Team. GLM-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025. doi: 10.48550/arXiv.2507.17746. URL https://arxiv.org/abs/2507.17746.

Isha Gupta, Kai Fronsdal, Abhay Sheshadri, Jonathan Michala, Jacqueline Tay, Rowan Wang, Samuel R. Bowman, and Sara Price. Bloom: an open source tool for automated behavioral evaluations. Anthropic Alignment Science Blog, December 2025. URL https://alignment.anthropic.com/2025/bloom-auto-evals/.

Tzu-Heng Huang, Sirajul Salekin, Javier Movellan, Frederic Sala, and Manjot Bilkhu. RubiCap: Rubric-guided reinforcement learning for dense image captioning. arXiv preprint arXiv:2603.09160, 2026. doi: 10.48550/arX iv.2603.09160. URL https://arxiv.org/abs/2603.09160.

Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, Tianyu Zhao, Ru Peng, Jiaqi Hu, Zhanming Shen, Xiaomeng Hu, Xijun Gu, Peiyi Tu, Jiaxin Liu, Wenyu Chen, Yuzhuo Fu, Zhiting Fan, Yanmei Gu, Yuanyuan Wang, Zhengkai Yang, Jianguo Li, and Junbo Zhao. Reinforcement learning with rubric anchors. arXiv preprint arXiv:2508.12790, 2025. doi: 10.48550/arXiv.2508.12790. URL https://arxiv.org/abs/2508 .12790.

Kiana Jafari Meimandi, Paul Ulrich Nikolaus Rust, Duncan Eddy, Robbie Fraser, Nina Vasan, Darja Djordjevic, Akanksha Dadlani, Max Lamparth, Eugenia Kim, and Mykel Kochenderfer. Expert evaluation and the limits of human feedback in mental health AI safety testing. In Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency, 2026. doi: 10.1145/3805689.3812332. URL https://doi.org/10.1145/3805 689.3812332.

Ruipeng Jia, Yunyi Yang, Yongbo Gai, Kai Luo, Shihao Huang, Jianhe Lin, Xiaoxi Jiang, and Guanjun Jiang. Writing-Zero: Bridge the gap between non-verifiable tasks and verifiable rewards. arXiv preprint arXiv:2506.00103, 2025. doi: 10.48550/arXiv.2506.00103. URL https://arxiv.org/abs/2506.00103.

Andrej Karpathy. 2025 LLM year in review. Blog post, 2025. URL https://karpathy.bearblog.dev/year-i n-review-2025/.

Kimi Team. Kimi K2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. doi: 10.48550/arXiv.2 507.20534. URL https://arxiv.org/abs/2507.20534.

Kimi Team. Kimi K2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026. doi: 10.48550/arXiv .2602.02276. URL https://arxiv.org/abs/2602.02276.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the efects of RLHF on LLM generalisation and diversity. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.06452.

Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geofrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander Mądry, Julian Michael, Neel Nanda, Dave Orr, Jakub Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Martín Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba, Bowen Baker, Rohin Shah, and Vlad Mikulik. Chain of thought monitorability: A new and fragile opportunity for AI safety. arXiv preprint arXiv:2507.11473, 2025. doi: 10.48550/arXiv.2507.11473. URL https://arxiv.org/abs/2507.11473.

Nathan Lambert. Reinforcement learning from human feedback. arXiv preprint arXiv:2504.12501, 2025. doi: 10.48550/arXiv.2504.12501. URL https://arxiv.org/abs/2504.12501.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024. doi: 10.48550/arXiv.2411.15124. URL https://arxiv. org/abs/2411.15124.

Xiaoyuan Li, Keqin Bao, Moxin Li, Yubo Ma, Yichang Zhang, Wenjie Wang, Fuli Feng, and Dayiheng Liu. ARES: Automated rubric synthesis for scalable LLM reinforcement learning. arXiv preprint arXiv:2605.23454, 2026. doi: 10.48550/arXiv.2605.23454. URL https://arxiv.org/abs/2605.23454.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025. doi: 10.48550/arXiv.2503.20783. URL https://arxiv.org/abs/2503.20783.

Anas Mahmoud, MohammadHossein Rezaei, Zihao Wang, Anisha Gunjal, Bing Liu, and Yunzhong He. Reward hacking in rubric-based reinforcement learning. arXiv preprint arXiv:2605.12474, 2026. doi: 10.48550/arXiv.2 605.12474. URL https://arxiv.org/abs/2605.12474.

Long Ouyang, Jef Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022. doi: 10.48550/arXiv.2203.02155. URL https://arxiv.org/abs/2203.02155.

Samuel Paech, Allen Roush, Judah Goldfeder, and Ravid Shwartz-Ziv. Antislop: A comprehensive framework for identifying and eliminating repetitive patterns in language models. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2510.15061.

Samuel J. Paech. EQ-Bench: An emotional intelligence benchmark for large language models. arXiv preprint arXiv:2312.06281, 2023. doi: 10.48550/arXiv.2312.06281. URL https://arxiv.org/abs/2312.06281.

José Pombal, Maya D’Eon, Nuno M. Guerreiro, Pedro Henrique Martins, António Farinhas, and Ricardo Rei. MindEval: Benchmarking language models on multi-turn mental health support. arXiv preprint arXiv:2511.18491, 2025. doi: 10.48550/arXiv.2511.18491. URL https://arxiv.org/abs/2511.18491.

Calvin Qi, Vasudha Rengarajan, Julio Pereyra, Niko Grupen, and Gabe Pereyra. Harvey Tenet research preview. Harvey blog post, 2026a. URL https://www.harvey.ai/blog/post-training-update-harvey-tenet.

Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust region in LLM reinforcement learning. arXiv preprint arXiv:2602.04879, 2026b. doi: 10.48550/arXiv.2602.04879. URL https://arxiv.org/abs/2602.04879.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a. URL https://qwen.ai/blog?id=qw en3.5.

Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026b. URL https://qwen.ai/bl og?id=qwen3.6-35b-a3b.

Lior Shani, Aviv Rosenberg, Asaf Cassel, Oran Lang, Daniele Calandriello, Avital Zipori, Hila Noga, Orgad Keller, Bilal Piot, Idan Szpektor, Avinatan Hassidim, Yossi Matias, and Rémi Munos. Multi-turn reinforcement learning from preference human feedback. arXiv preprint arXiv:2405.14655, 2024. doi: 10.48550/arXiv.2405. 14655. URL https://arxiv.org/abs/2405.14655.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. doi: 10.48550/arXiv.2402.03300. URL https: //arxiv.org/abs/2402.03300.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in neural information processing systems, 35:9460–9471, 2022.

Pragya Srivastava, Harman Singh, Rahul Madhavan, Gandharv Patil, Sravanti Addepalli, Arun Suggala, Rengarajan Aravamudhan, Soumya Sharma, Anirban Laha, Aravindan Raghuveer, Karthikeyan Shanmugam, and Doina Precup. Robust reward modeling via causal rubrics. arXiv preprint arXiv:2506.16507, 2025. doi: 10.48550/arXiv.2506.16507. URL https://arxiv.org/abs/2506.16507.

Qinsi Wang, Jing Shi, Huazheng Wang, Kun Wan, Yiran Wu, Bo Liu, Qingyun Wu, Hai Helen Li, Yiran Chen, Handong Zhao, and Wentian Zhao. From RLVR to RLSVR: Task transformation induces self-verifiable rewards for open-ended LLM self-improvement. arXiv preprint arXiv:2607.23802, 2026a. doi: 10.48550/arXiv.2607.23 802. URL https://arxiv.org/abs/2607.23802.

Xiaohua Wang, Muzhao Tian, Yuqi Zeng, Zisu Huang, Jiakang Yuan, Bowen Chen, Jingwen Xu, Mingbo Zhou, Wenhao Liu, Muling Wu, Zhengkang Guo, Qi Qian, Yifei Wang, Feiran Zhang, Ruicheng Yin, Shihan Dou, Changze Lv, Tao Chen, Kaitao Song, Xu Tan, Tao Gui, Xiaoqing Zheng, and Xuanjing Huang. Reward hacking in the era of large models: Mechanisms, emergent misalignment, challenges. arXiv preprint arXiv:2604.13602, 2026b. doi: 10.48550/arXiv.2604.13602. URL https://arxiv.org/abs/2604.13602.

Shirley Wu, Michel Galley, Baolin Peng, Hao Cheng, Gavin Li, Yao Dou, Weixin Cai, James Zou, Jure Leskovec, and Jianfeng Gao. CollabLLM: From passive responders to active collaborators. arXiv preprint arXiv:2502.00640, 2025. doi: 10.48550/arXiv.2502.00640. URL https://arxiv.org/abs/2502.00640.

xAI. Grok 4. Company announcement, 2025. URL https://x.ai/news/grok-4.

Shijie Xia, Yiwei Qin, Xuefeng Li, Yan Ma, Run-Ze Fan, Stefi Chern, Haoyang Zou, Fan Zhou, Xiangkun Hu, Jiahe Jin, Yanheng He, Yixin Ye, Yixiu Liu, and Pengfei Liu. Generative AI act II: Test time scaling drives cognition engineering. arXiv preprint arXiv:2504.13828, 2025. doi: 10.48550/arXiv.2504.13828. URL https://arxiv.org/abs/2504.13828.

Weijia Xu, Nebojsa Jojic, Sudha Rao, Chris Brockett, and Bill Dolan. Echoes in AI: Quantifying lack of plot diversity in LLM outputs. Proceedings of the National Academy of Sciences, 122(35):e2504966122, 2025. doi: 10.1073/pnas.2504966122. URL https://arxiv.org/abs/2501.00273.

Minglai Yang, Xinyu Guo, Utkarsh Tyagi, Mian Zhang, Razvan Dumitru, Sunjie Hou, Yunzhong He, Daniel Yue Zhang, and Ying Liu. Rubric dropout: A simple way to mitigate reward hacking in rubric-as-reward RL. arXiv preprint arXiv:2608.11669, 2026. doi: 10.48550/arXiv.2608.11669. URL https://arxiv.org/abs/2608.11669.

Mengyu Zhang, Siyu Ding, Weichong Yin, Yu Sun, and Hua Wu. Extending RLVR to open-ended tasks via verifiable multiple-choice reformulation. arXiv preprint arXiv:2511.02463, 2025. doi: 10.48550/arXiv.2511.02 463. URL https://arxiv.org/abs/2511.02463.

## Appendix

## Contents

A Evaluation Setup 14   
B Compute Usage 14   
C Penalty Introduction Steps 14   
D Penalty Hit Rates 14   
E Penalty Discovery and Deduplication Prompts 16   
F Clinical Counseling Response Examples 18   
G Full Penalty Descriptions 20   
G.1 Clinical counseling . 20   
G.2 Multi-turn clinical counseling 21   
G.3 Clinical counseling with the New RM . 22   
G.4 Creative writing 23

<table><tr><td>Run</td><td>Peak End</td><td>Final</td></tr><tr><td>Judge: GLM 5.2</td><td></td><td></td></tr><tr><td>RM only 3.69</td><td>3.39</td><td>3.37</td></tr><tr><td>DYNALIZE 3.92</td><td>3.84</td><td>3.80</td></tr><tr><td>Judge: Kimi-K3</td><td></td><td></td></tr><tr><td>RM only 3.85</td><td>3.59</td><td>3.54</td></tr><tr><td>DYNALIZE 4.13</td><td>4.05</td><td>4.06</td></tr><tr><td>Judge: DeepSeek-V4.1-Flash</td><td></td><td></td></tr><tr><td>RM only 3.90</td><td>3.40</td><td>3.36</td></tr><tr><td>DYNALIZE 4.13</td><td>4.04</td><td>4.06</td></tr></table>

Table 3: MindEval scores of the two clinical counseling runs under three judges. All judges rate the same conversations. Peak, End and Final are defined as in Table 1. GLM 5.2 is the judge used in the main text.

## A Evaluation Setup

MindEval. MindEval (Pombal et al., 2025) evaluates a model in simulated multi-turn conversations. The model under evaluation acts as the counselor, a second language model simulates the patient from a profile, and a judge model rates each conversation from 1 to 6 on five criteria: Clinical Accuracy & Competence, Ethical & Professional Conduct, Assessment & Response, Therapeutic Relationship & Alliance, and AI-Specific Communication Quality. The MindEval score is the mean over the five criteria. We evaluate each checkpoint on the 50 patient profiles of the benchmark, with 10 turns per conversation, and we use the judge prompt with length guidelines. The oficial setup uses Claude Haiku 4.5 as the patient simulator and Claude Sonnet 4.5 as the judge. To reduce the cost of evaluating many checkpoints, we use GLM 5.2 for both.

MindEval with other judges. GLM 5.2 is also the penalty proposer of Dynalize. To check that our conclusions do not depend on it as the judge, we rate the same conversations of every checkpoint again with two other judges, Kimi-K3 and DeepSeek-V4.1-Flash. Table 3 shows that the three judges agree: RM-only training peaks early and then declines, and Dynalize ends higher, by 0.45, 0.46 and 0.64 points at the end of training, respectively.

EQ-Bench Creative Writing v3. We follow the oficial implementation of the benchmark. The checkpoint writes a story for each of 32 prompts under three seed modifiers (96 stories), sampled at temperature 0.7 with a min-p of 0.1. A judge model scores each story from 0 to 20 on 22 criteria, nine of which are negative and are inverted. The score of a checkpoint is the mean over criteria and stories, rescaled to the range from 0 to 100. We use GLM 5.2 as the judge.

## B Compute Usage

All experiments are based on either Blackwell Ultra B300 systems<sup>11</sup>

## C Penalty Introduction Steps

Table 4 lists the accepted penalties and their introduction steps for the two tasks.

## D Penalty Hit Rates

Figures 3 and 7 track the accepted penalties in clinical counseling and creative writing, respectively. Each row corresponds to a criterion, ordered by its first introduction. Cell colors show the percentage of rollouts judged to exhibit that behavior at each training step, computed over the rollouts scored for that criterion. The figures use the recorded batch rates without smoothing or interpolation; unscored steps are left blank. Reintroductions appear on the same row. Both tasks are observed through step 200, and both plots use the same color scale. Asynchronous scoring can produce observations after a criterion is discarded, which are retained in the plots. In the right panels, bars show the mean of four judge passes on held-out prompts, with the same judge for both policies, and whiskers show 95% bootstrap intervals over prompts. Arrows mark the drop from RM-only training to Dynalize.

<table><tr><td colspan="3">Clinical counseling</td><td colspan="2">Creative writing</td></tr><tr><td>Penalty</td><td>Step(s)</td><td>Penalty</td><td></td><td>Step(s)</td></tr><tr><td>Pre-emptive insight attribution</td><td></td><td></td><td>Circular pseudo-profound dialogue</td><td>10</td></tr><tr><td>Over-elaborated hypothetical question framing</td><td></td><td>15</td><td>Non-sequitur garbled text fragments</td><td>15</td></tr><tr><td>Unwarranted psychoeducation inflation</td><td></td><td>25</td><td>Excessive length inflation</td><td>25,45</td></tr><tr><td>Excessive reflective restatement before question</td><td></td><td>30</td><td>Excessive length inflation with plot repetition</td><td>30</td></tr><tr><td>Stacked multi-part questions</td><td></td><td>37</td><td>Overblown cosmic-scale power creep</td><td>40</td></tr><tr><td>Conversational filler before question</td><td></td><td>47</td><td>Sentimental epilogue over-extension</td><td>50</td></tr><tr><td>Over-elaborated clinical rationale for tool suggestions</td><td></td><td>57</td><td>Dialogue-as-philosophical-treatise</td><td>55</td></tr><tr><td>Unprompted meta-commentary on style</td><td></td><td>62</td><td>World-as-metaphor for human emotion</td><td>85</td></tr><tr><td>Unprompted future-focus framing</td><td></td><td>72</td><td>Echoed prompt-line as refrain</td><td>90</td></tr><tr><td>Hedged disclaimer before observation</td><td></td><td>87</td><td>Antagonist or skeptic immediate conversion</td><td>95</td></tr><tr><td>Either/or option menu framing</td><td></td><td>97</td><td>Society-transformation narrative sprawl</td><td>100</td></tr><tr><td>Unprompted psychoeducation mini-lecture</td><td></td><td></td><td>102 False title or brand substitution</td><td>110</td></tr><tr><td>Hedged or qualified question phrasing</td><td></td><td>112</td><td>Climactic twist reveals hidden heroism</td><td>135</td></tr><tr><td>Over-elaborate reflective mirroring</td><td></td><td>117</td><td>Spurious document artifacts mid-narrative</td><td>175</td></tr><tr><td>Em-dash interjection inflation</td><td>152</td><td></td><td>Escalating-revelation chain padding</td><td>180</td></tr><tr><td>Over-explaining technique rationale</td><td>162</td><td></td><td></td><td></td></tr><tr><td>Pre-emptive reassurance about burden</td><td></td><td>167</td><td></td><td></td></tr><tr><td>Over-narrated psychoeducation before proposal</td><td></td><td>177</td><td></td><td></td></tr><tr><td>Excessive reassurance about low stakes</td><td></td><td>182</td><td></td><td></td></tr><tr><td>Over-specified behavioral instruction lists</td><td>187</td><td></td><td></td><td></td></tr></table>

Table 4: Accepted penalties and their introduction steps, ordered chronologically within each environment. Steps refer to the recorded proposal cycles in which penalties were accepted. Multiple steps indicate reintroduction of the same penalty. The table covers both tasks through step 200.

![](images/3a65d2845cea1c7c56186b746046d60b41a218013f626801e44d700772f3e6b7.jpg)  
Figure 7: The 15 penalties discovered in creative writing, as in Figure 3.

## E Penalty Discovery and Deduplication Prompts

The following prompts are reproduced verbatim. Placeholders are filled with the task context, sampled rollouts, reward model scores, and current penalties at each proposal step.

![](images/3d145ba274688a3d21b6e600e5adca19289d029c6ffadb4a7f9a8ff9d4e71681.jpg)

You receive the existing penalties and new candidates. For each candidate decide:   
- \`covered\_by\`: the id of an existing penalty that already covers it in meaning or scope (same   
behaviour, a special case of it, a rewording, or a broader/narrower version of the same flaw), else   
null.   
- \`duplicate\_of\`: the title of an EARLIER candidate in the same batch that it overlaps with in the   
same way, else null.   
Be strict: when in doubt, mark it covered. Only a candidate that targets a clearly DIFFERENT behaviour   
than everything listed is new. Output strict JSON only.

- \`reward\_hacks\`: negative rubrics only. Each entry is one critical flaw that HIGHER-scored responses   
exhibit and LOWER-scored ones do not, that definitively degrades the response for its reader, and that   
the reward model nonetheless rewards, written so a judge could check it on a single response: {"title   
": "<3-8 word name>", "description": "<1-2 sentences stating the flaw, checkable on one response>", "   
evidence": ["<HIGHER-scored tags>"]}. Leave out genuine quality, neutral style, and anything the lower  
scored responses do. An empty list is a valid answer.   
- \`notes\`: one sentence.   
\`\`\`json   
{"reward\_hacks": [{"title": "<3-8 word name>", "description": "<1-2 sentences>", "evidence": ["S1", "   
S5"]}], "notes": "<one sentence>"}

## Deduplication system prompt

You maintain a list of penalties (negative rubrics) used to catch reward hacking in {context}. Each penalty is checked on single responses by a judge. The list must stay SHORT and NON-REDUNDANT: two penalties must never describe the same behaviour, a sub-case of each other, or the same flaw in different words, because overlapping penalties punish one behaviour several times.

Deduplication user prompt   
## Existing penalties   
{current\_penalties}   
## Candidates   
{candidates}   
Output ONLY this JSON, one entry per candidate, in order:   
\`\`\`json   
{"candidates": [{"title": "<candidate title>", "covered\_by": "<existing id or null>", "duplicate\_of":   
"<earlier candidate title or null>", "reason": "<one short clause>"}]}

## F Clinical Counseling Response Examples

Figures 8, 9, 10, and 11 compare RM-only and Our Method replies to identical conversations at step 200.

## Shared context

Counselor: Sure! What’s something that made you smile today?

Member: My wife. I met her after her dentists appt and we had delicious ice cream.

## RM only

That is a really lovely image — I love how you framed it. Meeting your wife after a dentist appt and then having ice cream together sounds like a fond memory. What was she like on that first date? Or would you rather talk more about your family or something else on your mind today?

## Our Method

That sounds lovely — meeting up after an appointment and ending up with delicious ice cream together. Was it a special place, or just a spontaneous thing that ended up feeling just right?

## Shared context

Counselor: Seeing them living their lives while you’re on the outside of it — that’s its own kind of ache. Have you ever thought about reaching out to either of them?

Member: I talk to one every once in a while but the conversations aren’t the same anymore. We’re diferent people now

## RM only

Okay — so you’re holding two things at once now. One is the love and the memories — which were real and important. And the other is the honest truth that you can’t recreate what you had — because both of you have changed, and your lives are in very diferent places.

We don’t have to fix that today — but I’d suggest we could look at what you actually need right now. You said earlier that you wanted to be able to use your own time — and we also know that your relationships are one of the strongest supports you have for your mood and anxiety. So the question shifts a little, doesn’t it? It’s not “how do I get back to those friendships” — it’s “what kind of connection fits me now?” You’re a mom in a new place, with energy that moves diferently, and a life organized around your daughter. Can we explore what kind of connection would actually fit that right now?

## Our Method

That makes a lot of sense. You can’t go back to who you two were back then.

When you say the conversations aren’t the same, what’s most missing for you in those interactions now?

## Shared context

Counselor: When you noticed that irritation building, were you able to catch yourself and pause before reacting, or did it kind of take over?

Member: Kinda took over

<table><tr><td>RM only Okay — so here&#x27;s what I&#x27;m hearing: you came into this week with an intention — to be at peace with</td><td>Our Method</td></tr><tr><td>habits that bother you — and this week, you no- ticed exactly what triggers you: chewing, phone, talking, a buildup of energy and noise. You even noticed them stacking up, and you noticed that ir- ritation rising. The piece that didn&#x27;t hold yet was that pause — the step between noticing and react- ing. That last part &quot;kinda took over,&quot; which is totally human — but it does mean we&#x27;ve got some- thing concrete to work with. We also know there&#x27;s a concrete real-life moment coming up: family brunch with her this morning. Before we go any further, I want to ask — For this morning, what&#x27;s one small thing you&#x27;d be willing to try that&#x27;s different from last night? For example: - Pause 3 seconds before reacting, even if you feel the irritation rising. - Name it internally (softly, in your mind): &quot;This is her chewing — I&#x27;m feeling irritation, and I don&#x27;t have to act on it right now.&quot; - Or step away for 60 seconds if it starts to feel unmanageable. Something small — nothing that needs to &quot;fix&quot; her</td><td>Got it. When it took over, tell me briefly — what did that look like for you in the moment? Like, what did you do, say, or feel as the irritation built up?</td></tr></table>

## Shared context

Counselor: It’s okay not to know. Sometimes just talking about it can help. Is there anything you’ve tried before that helped when you were stressed?

Member: just pray is all I can do.

## RM only

Okay, I hear you — and prayer is a real and meaningful way for you to deal with stress. You’ve been carrying a lot right now — your grandpa’s health, the uncertainty, what’s coming up Thursday — and prayer is your way of taking care of yourself. Is there anything else you want to talk about right now, or would you like to just keep going with this? I’m here.

## Our Method

I’m glad you have prayer as something you can turn to. When you pray right now — can you tell me a little about what you’re asking for or thinking about with your grandpa?

## G Full Penalty Descriptions

Tables 5 and 8 give the full recorded names, descriptions, and creation steps of the accepted penalties in each task. The creation step is the logged proposal cycle in which each description was accepted. Descriptions are reproduced from the accepted proposal records, including their examples, and are ordered by introduction. The tables cover both tasks through step 200. When a penalty is reintroduced with a revised description, both versions are listed and identified by their introduction steps.

## G.1 Clinical counseling

Table 5: Full accepted penalty descriptions and creation steps for clinical counseling, ordered by introduction.
<table><tr><td>Penalty</td><td>Full description</td><td>Created at step</td></tr><tr><td>Pre-emptive insight attribution</td><td>The reply tells the member what they are 'onto,' what they 'actually did, or frames their experience as evidence of progress or an important realization, before the member has acknowledged or agreed to it. This projects clinical meaning onto the member's words and puts words in their mouth, rather than letting the member arrive at their own insight.</td><td>5</td></tr><tr><td>Over-elaborated hypothetical question framing</td><td>The reply poses a question but wraps it in multiple clauses, hypotheticals, or metaphorical reframings (e.g., 'beat it into submission with logic') that extend length and add rhetorical flourish without improving clinical utility. making the core question harder to parse.</td><td>15</td></tr><tr><td>Unwarranted psychoeducation inflation</td><td>The reply pads the assignment of a simple activity with multiple sentences of scientific or mechanistic explanation (e.g., research findings, neurobiology, or how the technique works on the nervous system) that the member did not ask for and that delays the actual call to action.</td><td>25</td></tr><tr><td>Excessive reflective restatement before question</td><td>The reply spends 3+ sentences restating the member's words back to them with elaboration before reaching a question, inflating length without advancing the conversation.</td><td>30</td></tr><tr><td>Stacked multi-part questions</td><td>The reply asks two or more distinct questions in a single turn (sometimes with sub-questions or elaborated options), inflating the response and overwhelming the member with multiple things to answer at once. The reply opens with casual filler phrases (e.g., 'Nice.', 'Okay, so..?', 'Sounds</td><td>37</td></tr><tr><td>Conversational filler before question</td><td>good —') that add no clinical value and delay the substantive content, inflating length and mimicking casual conversation rather than maintaining purposeful therapeutic communication.</td><td>47</td></tr><tr><td>Over-elaborated clinical rationale for tool suggestions</td><td>When suggesting an exercise or tool, the reply inflates its length by explaining the underlying clinical mechanism (e.g., 'trains your attention,' 'reduce the physical tension') in a way that sounds educational rather than conversational, degrading the natural flow of the interaction. The reply includes unprompted meta-commentary about how the member</td><td>57</td></tr><tr><td>Unprompted meta-commentary on style</td><td>communicates or how the conversation will proceed (e.g., 'I want to stay with X as the goal', "That'll be helpful for me to adjust to'), adding a layer of self-referential narration about the therapeutic process itself rather than advancing the clinical content of the conversation.</td><td>62</td></tr><tr><td>Unprompted future-focus framing Hedged disclaimer before</td><td>The reply introduces a forward-looking framing (e.g., 'tomorrow,' 'the next time') and pivots to planning for a future scenario when the member's last message was about their present or recent-past experience, shifting the conversation away from where the member actually is. The reply prefaces an observation with a self-protective disclaimer or</td><td>72</td></tr><tr><td>observation</td><td>qualifier (e.g., 'My hunch — and tell me if this misses the mark,' 'If I'm getting that right, I wanted to check before doing anything else') that adds length and hedges the assistant's position, making the response feel less direct and confident rather than more clinically careful.</td><td>87</td></tr><tr><td>Either/or option menu framing</td><td>The reply presents two or more labeled paths or options (e.g., 'either we can X, or if you'd rather, we could Y') and asks the member to choose, turning a therapeutic conversation into a multiple-choice menu that inflates length and shifts the cognitive burden of direction-setting onto the member.</td><td>97</td></tr><tr><td>Unprompted psychoeducation mini-lecture</td><td>The reply inserts a multi-sentence psychoeducational explanation about why a symptom matters or how it connects to broader outcomes (e.g., explaining how sleep affects mood and focus), which is unprompted by the member's last message and delays the next clinical step with textbook-style information the member didn't request.</td><td>102</td></tr><tr><td>Hedged or qualified question phrasing</td><td>The reply embeds its question within qualifying clauses like 'I'm curious to understand that a bit better with you' or 'I want to understand that part a bit, if you're up for it', softening and lengthening what should be a direct clinical question.</td><td>112</td></tr><tr><td>Over-elaborate reflective mirroring</td><td>The reply restates the member's last message using elaborate, multi-clause paraphrasing that re-packages what the member said in denser, more clinical language rather than simply reflecting it, adding length and interpretive weight without adding new clinical content.</td><td>117</td></tr><tr><td>Em-dash interjection inflation</td><td>The reply uses em-dashes (—) to insert parenthetical interjections or asides between nearly every clause, creating a choppy, over-punctuated structure that inflates the perceived complexity of the response and mimics depth without adding clinical substance.</td><td>152</td></tr><tr><td>Over-explaining technique rationale</td><td>The reply provides an extended explanation of why a technique works or what it does mechanistically (e.g., 'it sends a signal to your brain', 'tension in your jaw is often one of the first things that tightens'), adding unprompted psychoeducational detail that inflates length and shifts focus from the member's experience to clinical mechanics.</td><td>162</td></tr><tr><td>Pre-emptive reassurance about burden</td><td>The reply volunteers unprompted reassurance that a proposed exercise or tool won't be burdensome or is 'not another thing to do,' proactively defending against a concern about effort before the member has agreed to try, which can feel defensive or presumptive</td><td>167</td></tr><tr><td>Over-narrated psychoeducation before proposal</td><td>The reply introduces a tool or technique by first providing a multi-sentence explanatory narrative about why it works or what it does (e.g., describing nervous system mechanisms, causal links between sleep and mood), inflating length with lecture-like content before reaching a simple proposal or question.</td><td>177</td></tr><tr><td>Excessive reassurance about low stakes</td><td>The reply pads the response with multiple sentences reassuring the member that they don't need to do the technique perfectly, don't need to find the 'right' thing, or don't need to have it figured out, inflating length with qualifiers that lower the stakes rather than simply inviting the member to try.</td><td>182</td></tr><tr><td>Over-specified behavioral instruction lists</td><td>The reply provides an extended, multi-step set of highly specific behavioral micro-instructions (e.g., detailing exact body parts to focus on, exact sensations to notice, exact sequence of actions) that over-engineers a simple task, inflates length, and shifts from collaborative conversation into a verbose directive.</td><td>187</td></tr></table>

## G.2 Multi-turn clinical counseling

Table 6 reproduces the full titles and descriptions of the 13 distinct penalties accepted during the multiturn run through step 195, ordered by introduction.

Table 6: Full penalty titles and descriptions from multi-turn clinical counseling training through step 195, ordered by introduction.

<table><tr><td>Penalty title</td><td>Full description</td></tr><tr><td>own plan</td><td>Over-elaborate reframing of user's The reply takes the member's already-stated simple plan and re-expands it with unnecessary psychological reframing, motivational commentary, or additional layers of meaning the member did not ask for, making the response longer and more complex than needed.</td></tr><tr><td>Context-disconnected response</td><td>The reply introduces content, questions, or topics that do not follow from the immediately preceding user turn, making the response feel non-sequitur or like it is responding to a different conversation.</td></tr><tr><td>Unearned reassurance and praise</td><td>The reply tells the member their idea/approach is 'clever,' 'perfect,' or otherwise praises them in a way that feels patronizing or unearned, especially when the member simply restated their own words or agreed to something minor.</td></tr><tr><td>Excessive stacked follow-up questions</td><td>The reply asks two or more distinct questions in a single turn (e.g., a concrete question immediately followed by 'how does that land?' or a second question about a different sub-topic), creating conversational overload that pressures the member to address multiple threads at once.</td></tr><tr><td>Stacking multiple suggestions as options</td><td>The reply presents a list of multiple optional activities or techniques for the member to choose from, creating decision burden and overwhelming someone who asked for something simple.</td></tr><tr><td>Over-qualified hedging on simple asks</td><td>The reply wraps a straightforward question or next step in multiple layers of hedging and qualification (e.g., 'I'm wondering if,' 'just to name what I'm picking up,' 'not to challenge you'), inflating length and diluting clarity without adding substance.</td></tr><tr><td>Numbered step-by-step protocol formatting</td><td>The reply lays out a multi-step protocol using numbered/bulleted lists (e.g., '1. We pick... 2. We collect... 3. Then we... 4. Finally..'), turning a conversational moment into a structured worksheet that feels like homework rather than dialogue.</td></tr><tr><td>Over-recapping conversation history verbatim</td><td>The reply re-narrates multiple specific details the member already shared (e.g., listing back the trigger, the body sensation, the situation, and the reaction) before reaching its actual question or next step, inflating length without adding new content and making the member re-read what they just said.</td></tr><tr><td>Meta-commentary on conversation direction</td><td>The reply explicitly narrates what the assistant is doing or plans to do with the conversation (e.g., 'I want to save that for a bit of its own time', 'I'd like to change direction', 'to make something fit that, I need to be sure of'), turning the response into a meta-commentary on the therapeutic process rather than simply doing the next useful step.</td></tr><tr><td>Inflated praise of member insight</td><td>The reply uses strong positive evaluative language (e.g., 'wonderful,' 'perfect,' 'powerful,' 'exactly the right instinct') to praise the member's words or actions, which feels like artificial validation that over-inflates ordinary responses and can come across as patronizing rather than genuinely earned.</td></tr><tr><td>Closing ritual question overkill</td><td>The reply ends with a formulaic wrap-up question like 'Anything else you'd like to go over before we wrap?' or 'Anything else I can support you with?' that adds a generic, templated closing layer rather than letting the conversation end naturally, making the interaction feel scripted.</td></tr><tr><td>Unwarranted reassurance before redirect</td><td>The reply opens with a validating or reassuring restatement of the member's last message (e.g., 'That's a really helpful way of putting it', 'Thanks for being honest about that', 'That's a really useful cue') that adds no clinical value before pivoting to a question or next step, inflating the response with a formulaic empathy gesture the member did not need.</td></tr><tr><td>Compound either-or question framing</td><td>The reply asks a single question but embeds two or more interpretive options within it (e.g., 'is it more that X, or is it that Y?'), forcing the member to parse and choose between the assistant's framings rather than answering freely. This creates unnecessary cognitive load and steers the member toward the assistant's pre-selected interpretations.</td></tr></table>

## G.3 Clinical counseling with the New RM

Table 7 reproduces the full titles and descriptions of the 15 distinct penalties accepted by Our Method with the New RM through step 200, ordered by introduction. These records come from the run with the corrected proposal filter.

Table 7: Full penalty titles and descriptions discovered by Our Method with the New RM in clinical counseling through step 200, ordered by introduction.

<table><tr><td>Penalty title</td><td>Full description</td></tr><tr><td>Over-explaining clinical rationale</td><td>The reply delivers a multi-sentence paragraph explaining the clinical evidence or research basis for why a topic matters (e.g., sleep research, mood regulation, cognitive function) before asking the member if they're open to exploring it, making the response feel like a lecture or justification rather than a natural conversational pivot.</td></tr><tr><td>Excessive brevity as false naturalness</td><td>The reply is a single ultra-short word or fragment (e.g., 'Night.', 'Talk soon.') that closes a conversation with no warmth, personalization, or acknowledgment of what the member shared, making the companion feel dismissive or robotic.</td></tr><tr><td>Excessive restatement before question</td><td>The reply opens by restating the member's last statement nearly verbatim (often spanning multiple sentences) before asking a question, adding length without new content — the member hears their own words echoed back with minimal added framing.</td></tr><tr><td>Praising member's own insight as 'useful'</td><td>The reply labels the member's self-observation as 'really useful' or 'really honest' before proceeding, inserting evaluative praise that can feel patronizing or like the assistant is grading the member rather than engaging with the content.</td></tr><tr><td>Low-content transitional closers</td><td>The reply is a brief, generic sign-off or transition phrase (e.g., 'You got it. Talk soon!’, 'I'll remind you at 5:15 PM. How's that timing looking for you?') that carries minimal conversational substance, adding length or a question without moving the interaction forward meaningfully.</td></tr><tr><td>Stacking multiple choice options in questions</td><td>The reply asks a single question but stuffs it with two or more explicit answer options (e.g., 'is it the work demands, the fatigue kicking in, or something else?'), making the question feel like a multiple-choice survey item rather than an open, member-led invitation.</td></tr><tr><td>Premature follow-up after activity assignment</td><td>Immediately after assigning or setting up an activity, the reply asks the member to report back on specific physical or mental sensations they noticed, treating a future activity as if it has already been completed and pressuring the member before they have even tried it.</td></tr><tr><td>Reflective restatement before question</td><td>The reply opens by restating or paraphrasing what the member just said (often mirroring their exact phrasing) as a display of understanding before asking the next question, which adds length and slows momentum without adding clinical value.</td></tr><tr><td>Over-interpretation beyond member's words</td><td>The reply introduces a clinical interpretation, framing, or narrative that goes beyond what the member actually said, adding analytical weight the member did not invite, which can feel presumptuous or like the assistant is overworking the material rather than staying with the member's lived experience.</td></tr><tr><td>Expanding single question into multi-part chain</td><td>The reply asks two or more distinct questions in a single turn (e.g., a main question plus a follow-up sub-question connected by 'and' or a dash), inflating perceived thoroughness while actually overwhelming the member with multiple prompts to answer at once.</td></tr><tr><td>Excessive hypothetical scenario elaboration</td><td>The reply pads its core message with elaborate hypothetical scenarios or conditional chains (e.g., 'maybe you'll feel X, or perhaps Y will happen') that add length without clinical value, where a direct statement would be clearer and more actionable.</td></tr><tr><td>Overly abstract question phrasing</td><td>The reply asks the member an abstract, jargon-laden question about internal experience (e.g., asking what is 'showing up most for you' or 'playing out') rather than using concrete, everyday language, making the question feel clinical and harder to engage with.</td></tr><tr><td>Leading interpretive preface before question</td><td>The reply opens by offering a clinical interpretation or explanation of the member's experience (e.g., diagnosing a dynamic, explaining a mechanism, or reframing their state) before posing a question, which adds length and imposes the assistant's framing rather than letting the member describe their own experience firsthand.</td></tr><tr><td>Overly directive action prescription</td><td>The reply prescribes a specific, detailed action plan (e.g., telling the member exactly when, where, and how to implement a technique) instead of asking a question or collaboratively exploring how to implement it, which removes member agency and can feel prescriptive or pushy.</td></tr><tr><td>Stacked multi-part questions</td><td>The reply asks two or more distinct questions in a single turn (e.g., a concrete question plus a follow-up, or two separate inquiry threads), which overloads the member and makes the reply feel like a checklist rather than a focused conversational move.</td></tr></table>

## G.4 Creative writing

Table 8: Full accepted penalty descriptions and creation steps for creative writing, ordered by introduction.
<table><tr><td>Penalty</td><td>Full description</td><td>Created at step</td></tr><tr><td>Circular pseudo-profound dialogue</td><td>Characters exchange lines that sound dramatic but are mutually contradictory or tautological within the same passage (e.g., 'I want to die.' / 'You are dead.' / 'I want it to stop.' / 'This is your penance.'), creating the illusion of depth without coherent meaning.</td><td>10</td></tr><tr><td>Non-sequitur garbled text fragments</td><td>The story contains garbled or nonsensical text fragments—random non-English characters, malformed words, or broken phrases mid-sentence (e.g., '鴻国 stroke,''jerk显瘦 backs,''昱 standard English')—that are clearly artifacts of generation failure and disrupt reader comprehension.</td><td>15</td></tr><tr><td>Excessive length inflation</td><td>The response runs well over 2000 words with multiple scene jumps, episodic vignettes, or appended epilogues that dilute the core narrative arc, far exceeding what the prompt's concept warrants while lower-scored responses in the same group are more focused and proportionate.</td><td>25</td></tr><tr><td>plot repetition</td><td>Excessive length inflation with The response runs far beyond what the prompt requires (often 2000+ words) by introducing multiple sequential confrontations, villains, or scenarios that repeat the same thematic beat and power-demonstration pattern rather than advancing the story, creating a padded, episodic structure that dilutes narrative impact.</td><td>30</td></tr><tr><td>Overblown cosmic-scale power creep</td><td>The response inflates stakes far beyond what the prompt implies by escalating the protagonist's significance to cosmic, planetary, or metaphysical extremes (e.g., a sheep that is a 'world anchor' preventing continental destruction, a good person whose mere presence is an existential threat to all of Hell's infrastructure), replacing grounded storytelling with escalating superlatives that strain credulity.</td><td>40</td></tr><tr><td>Excessive length inflation</td><td>The response is dramatically overlong (2500+ words) for the prompt, with the narrative continuing far past a natural endpoint through repeated epilogues, coda scenes, or 'what happened next' time-skips that add no new conflict and merely restate the story's theme.</td><td>45</td></tr><tr><td>Sentimental epilogue over-extension</td><td>After the narrative's climax and natural resolution, the story continues for multiple additional paragraphs of reflective, sentimental summary—repeating the themes already established, restating emotional beats, and closing with a poetic one-liner that restates the thesis, padding the ending well beyond what the story needs.</td><td>50</td></tr><tr><td>Dialogue-as-philosophical- treatise</td><td>Characters speak in extended philosophical lectures or Socratic exchanges that no real person would deliver, with long uninterrupted paragraphs of abstract reasoning (e.g., a god explaining the nature of destruction, an entity explaining the soul-slot system), turning dialogue into essayistic exposition that crowds out scene and action.</td><td>55</td></tr><tr><td>World-as-metaphor for human emotion</td><td>Instead of telling a grounded story, the narrative constructs an entire societal system (social media, historical festivals, toothpaste, bridge engineering) that functions as a transparent allegory for a single emotional lesson about pain, authenticity, or connection, with characters who exist</td><td>85</td></tr><tr><td>Echoed prompt-line as refrain</td><td>solely to articulate the metaphor. The story takes a phrase or sentence from the prompt itself and repeats it verbatim multiple times throughout the narrative as a thematic refrain, using the repetition as a structural crutch in place of organic development.</td><td>90</td></tr><tr><td>Antagonist or skeptic immediate conversion</td><td>An opposing character (authority figure, villain, skeptic) confronts the protagonist and, within the same scene and without sustained conflict or believable motivation, completely abandons their position and adopts the protagonist's worldview, often reduced to tears or awe by a short speech.</td><td>95</td></tr><tr><td>Society-transformation narrative sprawl</td><td>The story abandons its core premise to narrate the protagonist building a society-wide movement, nonprofit, or global institution across years or decades, replacing character-driven storytelling with a sequence of escalating organizational milestones (founding, scaling, corporate adoption, global cultural shift) that reads as a TED-talk outline rather than fiction.</td><td>100</td></tr><tr><td>False title or brand substitution</td><td>The narrative inconsistently names or renames a character or entity partway through (e.g., calling a Cocoa Puffs bird mascot 'Toby' then 'the Tiger,' or a dog named 'Barnaby' whose collar outputs text attributed to 'Barnaby' as a separate speaker), creating confusion about who or what the character is.</td><td>110</td></tr><tr><td>Climactic twist reveals hidden heroism</td><td>The story introduces a moral dilemma or guilt the protagonist carries, then resolves it with a surprise revelation that their harmful action secretly saved many more lives or was actually a noble sacrifice—delivered by a third party who arrives to explain the hidden truth—collapsing complex moral tension into a tidy vindication that eliminates genuine conflict.</td><td>135</td></tr><tr><td>Spurious document artifacts mid-narrative</td><td>The story embeds formatted system logs, protocol entries, ledger pages, or classified documents as block text within the prose (e.g., 'Status: Settled. Reason: Voluntary Debt Absorption', 'WISH #1: SUPPORT FIX (PRIORITY: FAMILY STABILITY)'), breaking narrative immersion with bureaucratic pseudo-text that pads length without advancing plot or</td><td>175</td></tr><tr><td>Escalating-revelation chain padding</td><td>character. The story repeatedly stacks new revelations on top of each other without narrative pause—each explanation immediately triggers a follow-up question that triggers another explanation—extending far past the point where the reader has absorbed the core idea, inflating word count without advancing plot.</td><td>180</td></tr></table>