# CALM: Consistency-Aware Logit Masking in Decoding Space for Mitigating Overthinking of Large Reasoning Models

Shengjing Liu\*, Feiyang Li\*, Yuxuan Yang, Yile Wang<sup>†</sup>, Hui Huang College of Computer Science and Software Engineering, Shenzhen University

## Abstract

Large reasoning models (LRMs) have achieved strong performance on challenging mathematical and scientific tasks through long chainof-thought reasoning, but they often suffer from overthinking with unnecessary reasoning steps. Existing truncation-based methods typically detect consecutive trial-answer agreement and force early exit, yet such local stopping signals do not necessarily indicate finalanswer stability and can disrupt useful subsequent reflection or verification, potentially leading to performance degradation. We propose consistency-aware logit masking (CALM), a training-free decoding intervention for mitigating overthinking in LRMs. CALM first monitors trial answers at step boundaries and then applies logit masking to action transition point (ATP) tokens such as “Wait”, “Alternatively”, and “Hmm”, suppressing redundant transitional reasoning while allowing the model to continue its thought process more naturally. Experiments across four advanced LRMs and five benchmarks show that CALM reduces token consumption by 18.6%–25.0% compared with full CoT reasoning while maintaining performance, and can even improve accuracy by 0.07%–1.48% by reducing redundant reasoning. LLM-as-a-Judge results also demonstrate that CALM provides more coherent and concise results. Code is available at https://github.com/szu-tera/CALM.git.

## 1 Introduction

Large reasoning models (LRMs) such as OpenAI o1 (OpenAI, 2024) and DeepSeek-R1 (Guo et al., 2025) have advanced rapidly and achieved strong performance on solving challenging mathematical and graduate-level science problems, largely owing to their long chain-of-thought (CoT; Wei et al., 2022) reasoning and test-time scaling (Snell et al., 2025; Muennighoff et al., 2025). However, research reveals that LRMs could exhibit overthinking (Su et al., 2025; Chen et al., 2025; Cuadron et al., 2025), characterized by extensive redundant reasoning that not only fails to improve model performance but also results in considerable token waste and extra inference latency (Gema et al., 2025; Zhang et al., 2026).

![](images/58c4ae7515896cf92ca33101b2f6c3094bcac25b3fbefbdc81fb3b2a6c13ded5.jpg)  
Figure 1: Comparison of (a) truncation-based method and (b) our method that allows continual generation with action transition point masking after answer agreement.

To reduce overthinking, recent efforts attempt to identify critical points during reasoning at which the model’s reasoning information is sufficient, so as to truncate the reasoning path by forcing the generation of the end-of-thinking token </think> and eliciting the final answer directly. For example, researchers have used consecutive trial-answer agreement (Fu et al., 2024; Liu and Wang, 2025)

<table><tr><td>Related Work</td><td>Type</td><td>Signal</td><td>Action</td></tr><tr><td>L1 (Aggarwal and Welleck, 2025)</td><td>Reinforcement Learning</td><td>Length Penalty</td><td>Reward Feedback</td></tr><tr><td>S-GRPO (Dai et al., 2025)</td><td>Reinforcement Learning</td><td>Step Sufficiency</td><td>Reward Feedback</td></tr><tr><td>ThinkPrune (Hou et al., 2026)</td><td>Reinforcement Learning</td><td>Token Limit</td><td>Reward Feedback</td></tr><tr><td>LightThinker (Zhang et al., 2025)</td><td>Latent CoT Reasoning</td><td>Special Marker</td><td>Thought Compression</td></tr><tr><td>Step Entropy (Li et al., 2026b)</td><td>Latent CoT Reasoning</td><td>Štep Entropy</td><td>Thought Compression</td></tr><tr><td>Chain of Draft (Xu et al., 2025)</td><td>Prompt-based Constraint</td><td>Draft Length</td><td>Context Awareness</td></tr><tr><td>Plan and Budget (Lin et al., 2026)</td><td>Prompt-based Constraint</td><td>Token Budget</td><td>Context Awareness</td></tr><tr><td>Dynasor (Fu et al., 2024)</td><td>Reasoning Truncation</td><td>Consecutive Answers</td><td>Terminate and Early Exit</td></tr><tr><td>Answer Conv. (Liu and Wang, 2025)</td><td>Reasoning Truncation</td><td>Consecutive Answers</td><td>Terminate and Early Exit</td></tr><tr><td>TRACE (Li et al., 2026a)</td><td>Reasoning Truncation</td><td>Temporal Consistency</td><td></td></tr><tr><td>REFRAIN (Sun et al., 2026)</td><td>Reasoning Truncation</td><td>Semantic Redundancy</td><td>Terminate and Early Exit</td></tr><tr><td>PUMA (Min et al., 2026)</td><td>Reasoning Truncation</td><td>Semantic Redundancy</td><td>Terminate and Early Exit</td></tr><tr><td>DEER (Yang et al., 2026)</td><td>Reasoning Truncation</td><td>Transition Confidence</td><td>Answer with Token Limit</td></tr><tr><td>CALM (this work)</td><td>Decoding Intervention</td><td>Consecutive Answers</td><td>Answer with Token Limit Generate with Logit Masking</td></tr></table>

Table 1: Comparison of recent works on mitigating overthinking of large reasoning models.

as a truncation signal, as shown in Figure 1(a).

We argue that although this truncation-based early-exiting approach can reduce token consumption, it still restricts the model’s inherent reasoning ability. We conduct a pilot study (§ 3.3) and find that the stopping signal is not necessarily indicative of final-answer stability, with even considerable gaps observed on certain datasets. The reason can be that subsequent thinking may still contain steps that help improve the reasoning outcome, such as reflection and verification (Guo et al., 2025). Meanwhile, interrupting the reasoning process severely suppresses the continuity of the model’s thinking, leading to incoherent reasoning context and a mismatch with the data and preferences encountered during training, thereby affecting the accuracy of the final answer (Koh et al., 2026; Wang and Zhu, 2026; Wu et al., 2026). Therefore, we attempt to answer the following question: How can LRMs naturally stop thinking when their reasoning progress is detected to be approaching saturation?

We begin with typical thinking words such as “Wait”, “Alternatively”, and “Hmm” (Muennighoff et al., 2025), also termed as action transition point (ATP; Yang et al., 2026), and find that suppressing their generation probabilities can sustain generation while preserving accuracy (§ 3.3). The findings are also in line with recent work showing that ATP words do not genuinely reflect reasoning ability, but instead lead to lengthy reasoning steps (Pandey et al., 2026; Liu et al., 2025; Yu et al., 2026). Therefore, we propose consistencyaware logit masking (CALM), a simple yet effective inference-time decoding strategy for mitigating overthinking of LRMs, which consists of two steps: detecting consecutive answer agreement (§ 4.1) and action transition point logit masking (§ 4.2).

We evaluate on four advanced large reasoning models including DeepSeek-R1-Distill-Qwen (7B, 14B, 32B) and DeepSeek-R1-Distill-Llama-8B. Experiments across five mathematics and science benchmarks show that CALM outperforms recent baselines. In particular, CALM reduces token consumption by 18.6%–25.0% while maintaining performance compared with full CoT, and achieves a slight accuracy gain of 0.07%–1.48% through mitigating redundant reasoning.

In summary, our contributions are threefold:

1) Our pilot study shows that signals such as consecutive trial-answer agreement do not recover the completed-trace answer stably, revealing the limitations of truncation-based approaches, specifically their excessive reliance on local signals.

2) We propose CALM, a training-free decoding intervention method by suppressing the probability of typical thinking tokens to facilitate continual reasoning progress while reducing lengthy and redundant reasoning.

3) We evaluate two variants of CALM across four models and five benchmarks. Experimental results show that CALM substantially reduces redundant reasoning steps while outperforming recent baselines in terms of performance preservation.

## 2 Related Work

Overthinking of LRMs. Recent work on overthinking of LRMs can be categorized into three types, as summarized in Table 1:

1) Training-based methods. Typical reinforcement learning methods include L1 (Aggarwal and Welleck, 2025), which controls how long the model thinks through length penalties, S-GRPO (Dai et al., 2025), which evaluates the sufficiency of intermediate reasoning steps for early exiting, and ThinkPrune (Hou et al., 2026), which trains LRMs with an added token limit. For internal thought compression methods, LightThinker (Zhang et al., 2025) compresses intermediate thoughts into compact representations, while Li et al. (2026b) replaces selected low-entropy step with a special token [SKIP]. These methods incur prohibitive retraining overhead and cannot be deployed as plugand-play interventions on frozen LRMs.

2) Prompt-based methods. Prompt-based methods encourage concise reasoning through prompt constraints. Chain of Draft (Xu et al., 2025) limits reasoning to short draft steps. Plan and Budget (Lin et al., 2026) assigns a length budget to the reasoning. Such constraints can be ignored on difficult problems or suppress necessary intermediate reasoning (Yu et al., 2025), so they are difficult to apply across diverse scenarios stably.

3) Inference-based methods. Reasoning truncation methods follow a “detect, truncate, early exit” pipeline. For example, Dynasor (Fu et al., 2024) probes trial answers at fixed token intervals and stops when consecutive probes agree. Answer Convergence (Liu and Wang, 2025) probes at reasoningchunk boundaries and exits when consecutive answers match. TRACE (Li et al., 2026a) aggregates answer agreement and the confidence trajectory over a sliding window. REFRAIN (Sun et al., 2026) identifies steps that are both reflective and redundant. PUMA (Min et al., 2026) trains a detector to flag redundant steps and stops after verifying answer consistency and confidence. DEER (Yang et al., 2026) elicits a trial answer at transitions and exits when confidence exceeds a threshold.

These methods force the model to terminate its reasoning upon satisfaction of the stopping criterion, thereby restricting subsequent generation without token limitations. In contrast, we mitigate overthinking from the perspective of suppressing the probabilities of specific tokens in decoding space, yet maintain the capacity for sustained generation while reducing thinking redundancy.

Thinking Tokens in LRMs. Muennighoff et al. (2025) try appending additional thinking tokens such as “Wait” and “Hmm” to enable test-time scaling. DeepSeek-R1 (Guo et al., 2025) also demonstrates an “aha moment” after reinforcement learning training, which appears as self-reflection following stop words such as “Wait”. However, Liu et al. (2025) show that the phenomenon already exists in base models and does not reflect reasoning ability. Wang et al. (2025a) directly disables the generation of these words and find that it does not compromise the utility of multimodal models. Yu et al. (2026) classifies such words as anthropomorphic reflection markers and finds that these markers primarily affect how reasoning trajectories are organized on the surface and sampled. These works mainly focus on empirical findings and mechanistic analysis, whereas we design a decoding strategy that conditionally suppresses such thinking tokens in long reasoning chains, retains the model’s capacity for continuous generation, and practically mitigates overthinking in LRMs.

## 3 Preliminaries and Pilot Study

## 3.1 Task Formulation and Notations

Given a question Q, an autoregressive reasoning model generates a chain-of-thought reasoning sequence $\boldsymbol { S } = \left( s _ { 1 } , \ldots , s _ { N } \right)$ , followed by a final answer y. We segment the sequence into N steps at natural paragraph or step boundaries, with the rules given in Appendix A. The task is to reduce unnecessary redundant reasoning steps while preserving the accuracy of answer y. Let $s _ { 1 : n }$ denote the reasoning prefix up to step n. Following the probing paradigm of prior early-exit studies (Fu et al., 2024; Liu and Wang, 2025), we sample an intermediate trial answer $a _ { n }$ conditioned on $s _ { 1 : n }$ at each step boundary n by appending an answerinducing suffix on an independent decoding branch. This probing branch does not alter the original reasoning context. Consequently, each reasoning step $s _ { n }$ is associated with a trial answer $a _ { n }$ . The suffix strings are provided in Appendix B.

Tracking intermediate trial answers enables monitoring reasoning stability during generation. Notably, existing early-exit methods adopt consecutive trial answer agreement as a criterion to earlystop and truncate generation (Fu et al., 2024; Liu and Wang, 2025).

## 3.2 Action Transition Point

During long chain-of-thought generation, LRMs alternate among distinct thinking actions, such as approach exploration and procedural execution. Switches between these actions are typically indicated by explicit thinking tokens such as “Wait”. Following the naming by Yang et al. (2026), we formalize these tokens as action transition points (ATP) and denote their set by $\mathcal { V } _ { \mathrm { A T P } }$

![](images/0f650a3f416fadda5b103b673ef1206be5d814a8943004612248239855d9eed5.jpg)  
Figure 2: Pilot study on accuracy and tokens on continued reasoning trace after trial answer agreement for DeepSeek-R1-Distill-Qwen-7B on OlympiadBench.

There is currently no rigorous definition of the ATP vocabulary. Following Wang et al. (2025a), we use three LRMs on five benchmarks to identify the sentence-initial words of different reasoning steps, and select the top 8 words to form the initial $\mathcal { V } _ { \mathrm { A T P } } ^ { \mathrm { i n i t } } = \{ { ^ { \ast \ast } S o ^ { \prime \prime } } , { ^ { \ast \ast } W a i t ^ { \prime \prime } } , { ^ { \ast } B u t ^ { \prime \prime } } , { ^ { \ast } T h e r e f o r e ^ { \prime \prime } } , { ^ { \ast } A l } -$ ternatively”, “However”, “Maybe”,“Hmm”}. We find that reflection words such as “Wait”, “Alternatively”, and “Hmm” have been discussed in recent work (Muennighoff et al., 2025; Wang et al., 2025b; Yu et al., 2026) and these words may be highly correlated with redundant reasoning, and other control words regulate verification and thinking, thereby preserving the accuracy of long chain-of-thought reasoning. We therefore focus on three words $\mathcal { V } _ { \mathrm { A T P } } = \{ ^ {  } W a i t ^ { \ast }$ , “Alternatively”, and “Hmm”}, and we conduct vocabulary ablation in Section 6.2.

## 3.3 Reasoning Trace after Answer Agreement

We hypothesize that once tentative answer agreement is reached, subsequent ATP triggers predominantly degenerate into computational redundancy, needlessly bloating the reasoning trajectory. To examine this, we track the reasoning continuation starting from the first answer agreement. We conduct a pilot study on DeepSeek-R1-Distill-Qwen-7B (Guo et al., 2025) and OlympiadBench (He et al., 2024) to investigate how accuracy changes with the mean number of continuation tokens after consecutive identical answers, comparing unconstrained vanilla CoT with a continuation that suppresses $\mathcal { V } _ { \mathrm { A T P } }$ tokens after agreement. Detailed settings are given in Appendix C.

Figure 2 illustrates two key dynamics of this continuation. First, immediate truncation upon identical trial answers incurs a substantial accuracy gap. At the start of the continuation $( x = 0 )$ , the probed trial answer achieves an accuracy of only 46.0%, falling 10.4 points below vanilla CoT (56.4%). Truncating immediately at this boundary severs downstream derivations and discards the opportunity for later revisions. Second, both continuations allow the model to conclude, while suppressing ATP tokens shortens the continuation. Suppressing $\mathcal { V } _ { \mathrm { A T P } }$ after agreement saves 19.9% of postagreement tokens while reaching a final accuracy comparable to Vanilla CoT.

## 4 Method

Grounded in the empirical findings in Section 3.3, we propose consistency-aware logit masking (CALM), a training-free decoding-based method for mitigating overthinking.Rather than terminating generation early by truncating, CALM allows generation after detecting consecutive trial-answer agreement. Figure 3 illustrates the overall pipeline.

## 4.1 Detecting Consecutive Answer Agreement

At each step boundary $n ,$ CALM obtains the current trial answer $a _ { n }$ following the probing procedure described in Section 3. We monitor the sequence of trial answers to detect the earliest window in which the model outputs identical answers for k consecutive steps. Formally, the consecutiveagreement indicator $c _ { n }$ and the agreement boundary step $b ^ { \star }$ are defined as:

$$
c _ { n } = \mathbb { 1 } [ a _ { n } = a _ { n - 1 } = \cdot \cdot \cdot = a _ { n - k + 1 } ] ,\tag{1}
$$

$$
b ^ { \star } = \operatorname* { m i n } \{ n \geq \mathrm { M S S } : c _ { n } = 1 \} ,\tag{2}
$$

where 1[·] denotes the indicator function and MSS is the minimum stop step. If the trial answers never reach k consecutive matches during generation, then $b ^ { \star } = \infty$ and the reasoning trajectory remains entirely unmodified.

## 4.2 Action Transition Point Logit Masking

After obtaining the agreement boundary b<sup>⋆</sup>, CALM adds a masking bias to the logits of candidate tokens that match $\mathcal { V } _ { \mathrm { A T P } }$ . Formally, at decoding position t within the reasoning steps, let $h _ { t }$ denote the final hidden state and $W _ { \mathrm { L M } }$ the vocabulary projection weight of the LM Head. Let V denote the model’s token vocabulary. For each candidate token $v \in \mathcal V$ , the raw logit is $\ell _ { t } ( v ) = [ W _ { \mathrm { L M } } h _ { t } ] _ { \mathrm { i d x } : v } ,$ to which we add a bias $b _ { t } ( v )$ for logit masking:

![](images/fa9c839291feb9a7254d11b60e3ed7c1ad5dbb77b7255bd4e8dea00776aa73d6.jpg)  
Figure 3: The overall pipeline of CALM. A trial answer is probed at each step boundary. After the same trial answer remains identical for k consecutive steps, CALM lowers the logits of ATP tokens and continues reasoning.

$$
\begin{array} { r } { \ell _ { t } ( v ) = [ W _ { \mathrm { L M } } h _ { t } ] \mathrm { i d } \boldsymbol { \mathbf { x } } \boldsymbol { \cdot } \boldsymbol { v } , } \\ { \tilde { \ell } _ { t } ( v ) = \ell _ { t } ( v ) + b _ { t } ( v ) , } \end{array}\tag{3}
$$

where idx : v denotes the index of $v \in \mathcal { V } .$ . Let t be a token position and $n ( t )$ be the corresponding reasoning step. We investigate two masking strategies for the bias term $b _ { t } ( v )$

Hard Masking employs a constant value to strictly prohibit ATP tokens once the agreement boundary is crossed:

$$
b _ { t } ( v ) = \left\{ \begin{array} { l l } { - \infty } & { \mathrm { i f } n ( t ) > b ^ { \star } , v \in \mathcal { B } _ { t } , } \\ { 0 } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{4}
$$

where $\boldsymbol { B } _ { t } \subseteq \mathcal { V }$ is the set of token ids that complete an ATP surface form under the current prefix. The surface forms and the matching rules are given in Appendix D. Vocabulary ablations are reported in Section 6.2.

Soft Masking adopts an adaptive progressive penalty through decisive damping rather than a rigid ban. Let c(t) denote the occurrence count of ATP strings completed after $b ^ { \star }$ and before position t, with penalty $\eta ( c ) = - 1 0 ( 1 - \rho ^ { c } )$ for $0 < \rho < 1 \mathrm { : }$

$$
b _ { t } ( v ) = { \left\{ \begin{array} { l l } { \eta ( c ( t ) ) } & { { \mathrm { i f ~ } } n ( t ) > b ^ { \star } , v \in \mathcal { B } _ { t } , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{5}
$$

where the penalty is 0 when $c ( t ) = 0$ , and smoothly approaches −10 as $c ( t )$ accumulates. Under a softmax operation, a value of −10 reduces an affected token’s relative weight by a factor of about $e ^ { - 1 0 } \approx 4 . 5 \times 1 0 ^ { - 5 }$ . The sensitivity of hyperparameter $\rho$ is shown in Section 6.3.

After applying probability suppression to ATP tokens, CALM enables continual decoding without truncation or imposing additional token limits.

We evaluate both CALM (soft) with progressive penalty and CALM (hard) with constant masking in our main experiments.

## 5 Experiments

## 5.1 Experimental Setup

Benchmarks. We evaluate on five reasoning benchmarks that span several difficulty levels: four mathematical benchmarks including MATH-500 (Lightman et al., 2024), OlympiadBench (He et al., 2024), AIME 2025 (MAA Committees, 2025), and AMC 2023 (AI-MO, 2024), and one scientific reasoning benchmark:GPQA-Diamond (Rein et al., 2024).

Models. Four DeepSeek models are used in main experiments, including DeepSeek-R1-Distill-Qwen (7B, 14B, and 32B) (Guo et al., 2025) and DeepSeek-R1-Distill-Llama-8B. For additional models Llama-3.1-Nemotron-Nano-8B (Bercovich et al., 2025), Qwen3 (4B, 8B, 30B-A3B, 32B) (Yang et al., 2025), and QwQ-32B (Qwen Team, 2025), results are given in Appendix E.

Baselines. To make a fair evaluation, we compare with inference-based methods including Answer Convergence (Ans. Conv.; Liu and Wang, 2025), Dynasor (Fu et al., 2024), TRACE (Li et al., 2026a), PUMA (Min et al., 2026), and DEER (Yang et al., 2026). We obtain and report the results based on their open-source version.

Evaluation Metrics. We report accuracy (Acc) and average generated tokens (Tok). Overall, we additionally report token efficiency (TE), using the OckScore (Du et al., 2025) defined as accuracy minus $\lambda \ln ( T / C + 1 )$ , with $\lambda = 1 0 , C = 1 0 , 0 0 0 .$

<table><tr><td rowspan=1 colspan=13>MATH-500        Olympiad          GPQA           AIME25          AMC23             OverallMethodAcc↑   Tok↓   Acc↑   Tok↓    Acc↑   Tok↓   Acc↑   Tok↓    Acc↑   Tok↓   Acc↑   Tok↓   TE↑</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4>DeepSeek-R1-Distill-Qwen-7B</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    92.33</td><td rowspan=1 colspan=1>3,847</td><td rowspan=1 colspan=2>57.63   8,718</td><td rowspan=1 colspan=4>47.14   7,327    36.67   15,080</td><td rowspan=1 colspan=1>89.17</td><td rowspan=1 colspan=1>6,297</td><td rowspan=1 colspan=1>64.59</td><td rowspan=1 colspan=1>8,254</td><td rowspan=1 colspan=1>58.57</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   63.60</td><td rowspan=1 colspan=1>1,006 (-73.8%)</td><td rowspan=1 colspan=1>31.46</td><td rowspan=1 colspan=1>1,490 (-82.9%)</td><td rowspan=1 colspan=2>26.77348 (-95.3%)</td><td rowspan=1 colspan=1>14.44</td><td rowspan=1 colspan=1>3,020 (-80.0%)</td><td rowspan=1 colspan=1>44.17</td><td rowspan=1 colspan=1>1,293 (-79.5%)</td><td rowspan=1 colspan=1>36.09</td><td rowspan=1 colspan=1>1,432 (-82.7%)</td><td rowspan=1 colspan=1>34.75</td></tr><tr><td rowspan=1 colspan=1>Dynasor    85.13</td><td rowspan=1 colspan=1>2,655 (-31.0%)</td><td rowspan=1 colspan=1>51.41</td><td rowspan=1 colspan=1>5,820 (-33.2%)</td><td rowspan=1 colspan=2>34.181,802 (-75.4%)</td><td rowspan=1 colspan=1>34.44</td><td rowspan=1 colspan=1>13,202 (-12.5%)</td><td rowspan=1 colspan=1>81.67</td><td rowspan=1 colspan=1>4,348 (-31.0%)</td><td rowspan=1 colspan=1>57.37</td><td rowspan=1 colspan=1>5,565 (-32.6%)</td><td rowspan=1 colspan=1>52.95</td></tr><tr><td rowspan=1 colspan=1>TRACE     89.33</td><td rowspan=1 colspan=1>2,764 (-28.2%)</td><td rowspan=1 colspan=1>53.53</td><td rowspan=1 colspan=1>4,340 (-50.2%)</td><td rowspan=1 colspan=2>43.772,410 (-67.1%)</td><td rowspan=1 colspan=1>34.44</td><td rowspan=1 colspan=1>8,270 (-45.2%)</td><td rowspan=1 colspan=1>77.50</td><td rowspan=1 colspan=1>3,936 (-37.5%)</td><td rowspan=1 colspan=1>59.71</td><td rowspan=1 colspan=1>4,344 (-47.4%)</td><td rowspan=1 colspan=1>56.10</td></tr><tr><td rowspan=1 colspan=1>PUMA     90.93</td><td rowspan=1 colspan=1>2,599 (-32.4%)</td><td rowspan=1 colspan=2>56.105,115 (-41.3%)</td><td rowspan=1 colspan=2>47.316,768 (-7.6%)</td><td rowspan=1 colspan=1>36.67</td><td rowspan=1 colspan=1>12,366 (-18.0%)</td><td rowspan=1 colspan=1>84.17</td><td rowspan=1 colspan=1>4,594 (-27.0%)</td><td rowspan=1 colspan=1>63.03</td><td rowspan=1 colspan=1>6,288 (-23.8%)</td><td rowspan=1 colspan=1>58.15</td></tr><tr><td rowspan=1 colspan=1>DEER      89.80</td><td rowspan=1 colspan=1>2,436 (-36.7%)</td><td rowspan=1 colspan=2>56.946,545 (-24.9%)</td><td rowspan=1 colspan=1>49.16</td><td rowspan=1 colspan=1>7,406 (+1.1%)</td><td rowspan=1 colspan=1>36.67</td><td rowspan=1 colspan=1>12,210 (-19.0%)</td><td rowspan=1 colspan=1>90.00</td><td rowspan=1 colspan=1>4,666 (-25.9%)</td><td rowspan=1 colspan=1>64.51</td><td rowspan=1 colspan=1>6,653 (-19.4%)</td><td rowspan=1 colspan=1>59.41</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)92.47</td><td rowspan=1 colspan=1>3,152 (-18.1%)</td><td rowspan=1 colspan=2>58.527,162 (-17.8%)</td><td rowspan=1 colspan=1>50.51</td><td rowspan=1 colspan=1>5,393 (-26.4%)</td><td rowspan=1 colspan=1>40.00</td><td rowspan=1 colspan=1>12,288 (-18.5%)</td><td rowspan=1 colspan=1>90.83</td><td rowspan=1 colspan=1>4,593 (-27.1%)</td><td rowspan=1 colspan=1>66.47</td><td rowspan=1 colspan=1>6,518 (-21.0%)</td><td rowspan=1 colspan=1>61.45</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)92.27</td><td rowspan=1 colspan=1>2,843 (-26.1%)</td><td rowspan=1 colspan=2>57.336,663 (-23.6%)</td><td rowspan=1 colspan=2>48.485,059 (-31.0%)</td><td rowspan=1 colspan=2>42.2211,749 (-22.1%)</td><td rowspan=1 colspan=1>88.33</td><td rowspan=1 colspan=1>4,646 (-26.2%)</td><td rowspan=1 colspan=1>65.73</td><td rowspan=1 colspan=1>6,192 (-25.0%)</td><td rowspan=1 colspan=1>60.91</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>DeepSeek-R1-Disti</td><td rowspan=1 colspan=2>ll-Qwen-14B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    92.87</td><td rowspan=1 colspan=1>3,504</td><td rowspan=1 colspan=1>62.67</td><td rowspan=1 colspan=1>7,792</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>6,335</td><td rowspan=1 colspan=1>53.33</td><td rowspan=1 colspan=1>12,296</td><td rowspan=1 colspan=1>94.17</td><td rowspan=1 colspan=1>5,448</td><td rowspan=1 colspan=1>70.61</td><td rowspan=1 colspan=1>7,075</td><td rowspan=1 colspan=1>65.26</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   58.13</td><td rowspan=1 colspan=1>839 (-76.1%)</td><td rowspan=1 colspan=1>26.86</td><td rowspan=1 colspan=1>1,125 (-85.6%)</td><td rowspan=1 colspan=1>38.22</td><td rowspan=1 colspan=1>330 (-94.8%)</td><td rowspan=1 colspan=1>12.22</td><td rowspan=1 colspan=1>2,606 (-78.8%)</td><td rowspan=1 colspan=1>40.00</td><td rowspan=1 colspan=1>1,165 (-78.6%)</td><td rowspan=1 colspan=1>35.09</td><td rowspan=1 colspan=1>1,213 (-82.9%)</td><td rowspan=1 colspan=1>33.95</td></tr><tr><td rowspan=1 colspan=1>Dynasor    91.40</td><td rowspan=1 colspan=1>3,573 (+2.0%)</td><td rowspan=1 colspan=1>61.48</td><td rowspan=1 colspan=1>7,162 (-8.1%)</td><td rowspan=1 colspan=1>49.33</td><td rowspan=1 colspan=1>3,577 (-43.5%)</td><td rowspan=1 colspan=1>47.78</td><td rowspan=1 colspan=1>12,765 (+3.8%)</td><td rowspan=1 colspan=1>93.33</td><td rowspan=1 colspan=1>5,606 (+2.9%)</td><td rowspan=1 colspan=1>68.66</td><td rowspan=1 colspan=1>6,537 (-7.6%)</td><td rowspan=1 colspan=1>63.63</td></tr><tr><td rowspan=1 colspan=1>TRACE     89.27</td><td rowspan=1 colspan=1>2,555 (-27.1%)</td><td rowspan=1 colspan=1>56.79</td><td rowspan=1 colspan=1>4,021 (-48.4%)</td><td rowspan=1 colspan=1>42.26</td><td rowspan=1 colspan=1>2,135 (-66.3%)</td><td rowspan=1 colspan=1>40.00</td><td rowspan=1 colspan=1>7,549 (-38.6%)</td><td rowspan=1 colspan=1>84.17</td><td rowspan=1 colspan=1>3,914 (-28.2%)</td><td rowspan=1 colspan=1>62.50</td><td rowspan=1 colspan=1>4,035 (-43.0%)</td><td rowspan=1 colspan=1>59.11</td></tr><tr><td rowspan=1 colspan=1>PUMA      92.07</td><td rowspan=1 colspan=1>2,692 (-23.2%)</td><td rowspan=1 colspan=1>60.64</td><td rowspan=1 colspan=1>5,283 (-32.2%)</td><td rowspan=1 colspan=1>53.03</td><td rowspan=1 colspan=1>5,419 (-14.5%)</td><td rowspan=1 colspan=1>51.11</td><td rowspan=1 colspan=1>11,141 (-9.4%)</td><td rowspan=1 colspan=1>94.17</td><td rowspan=1 colspan=1>4,566 (-16.2%)</td><td rowspan=1 colspan=1>70.20</td><td rowspan=1 colspan=1>5,820 (-17.7%)</td><td rowspan=1 colspan=1>65.61</td></tr><tr><td rowspan=1 colspan=1>DEER      90.87</td><td rowspan=1 colspan=1>2,465 (-29.7%)</td><td rowspan=1 colspan=1>60.94</td><td rowspan=1 colspan=1>6,349 (-18.5%)</td><td rowspan=1 colspan=1>54.38</td><td rowspan=1 colspan=1>6,381 (+0.7%)</td><td rowspan=1 colspan=1>51.11</td><td rowspan=1 colspan=1>12,000 (-2.4%)</td><td rowspan=1 colspan=1>92.50</td><td rowspan=1 colspan=1>4,470 (-18.0%)</td><td rowspan=1 colspan=1>69.96</td><td rowspan=1 colspan=1>6,333 (-10.5%)</td><td rowspan=1 colspan=1>65.05</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)93.33</td><td rowspan=1 colspan=1>2,883 (-17.7%)</td><td rowspan=1 colspan=1>61.88</td><td rowspan=1 colspan=1>6,275 (-19.5%)</td><td rowspan=1 colspan=1>57.41</td><td rowspan=1 colspan=1>4,962 (-21.7%)</td><td rowspan=1 colspan=1>57.78</td><td rowspan=1 colspan=1>11,421 (-7.1%)</td><td rowspan=1 colspan=1>93.33</td><td rowspan=1 colspan=1>4,768 (-12.5%)</td><td rowspan=1 colspan=1>72.75</td><td rowspan=1 colspan=1>6,062 (-14.3%)</td><td rowspan=1 colspan=1>68.01</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)92.00</td><td rowspan=1 colspan=1>2,769 (-21.0%)</td><td rowspan=1 colspan=1>61.98</td><td rowspan=1 colspan=1>6,083 (-21.9%)</td><td rowspan=1 colspan=1>56.73</td><td rowspan=1 colspan=1>4,635 (-26.8%)</td><td rowspan=1 colspan=1>55.56</td><td rowspan=1 colspan=1>11,184 (-9.0%)</td><td rowspan=1 colspan=1>94.17</td><td rowspan=1 colspan=1>4,121 (-24.4%)</td><td rowspan=1 colspan=1>72.09</td><td rowspan=1 colspan=1>5,758 (-18.6%)</td><td rowspan=1 colspan=1>67.54</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>DeepSeek-R1-Disti</td><td rowspan=1 colspan=1>ll-Qwen</td><td rowspan=1 colspan=1>-32B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    91.40</td><td rowspan=1 colspan=1>2,834</td><td rowspan=1 colspan=1>62.57</td><td rowspan=1 colspan=1>7,221</td><td rowspan=1 colspan=1>63.80</td><td rowspan=1 colspan=1>6,101</td><td rowspan=1 colspan=1>52.22</td><td rowspan=1 colspan=1>12,843</td><td rowspan=1 colspan=1>95.83</td><td rowspan=1 colspan=1>5,841</td><td rowspan=1 colspan=1>73.16</td><td rowspan=1 colspan=1>6,968</td><td rowspan=1 colspan=1>67.87</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   64.60</td><td rowspan=1 colspan=1>598 (-78.9%)</td><td rowspan=1 colspan=1>27.85</td><td rowspan=1 colspan=1>834 (-88.5%)</td><td rowspan=1 colspan=1>44.28</td><td rowspan=1 colspan=1>277 (-95.5%)</td><td rowspan=1 colspan=1>7.78</td><td rowspan=1 colspan=1>1,433 (-88.8%)</td><td rowspan=1 colspan=1>41.67</td><td rowspan=1 colspan=1>923 (-84.2%)</td><td rowspan=1 colspan=1>37.23</td><td rowspan=1 colspan=1>813 (-88.3%)</td><td rowspan=1 colspan=1>36.45</td></tr><tr><td rowspan=1 colspan=1>Dynasor    88.67</td><td rowspan=1 colspan=1>2,833 (0.0%)</td><td rowspan=1 colspan=1>60.44</td><td rowspan=1 colspan=1>6,280 (-13.0%)</td><td rowspan=1 colspan=1>54.88</td><td rowspan=1 colspan=1>2,947 (-51.7%)</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>12,899 (+0.4%)</td><td rowspan=1 colspan=1>93.33</td><td rowspan=1 colspan=1>5,739 (-1.7%)</td><td rowspan=1 colspan=1>69.47</td><td rowspan=1 colspan=1>6,140 (-11.9%)</td><td rowspan=1 colspan=1>64.68</td></tr><tr><td rowspan=1 colspan=1>TRACE     88.87</td><td rowspan=1 colspan=1>2,110 (-25.5%)</td><td rowspan=1 colspan=1>56.54</td><td rowspan=1 colspan=1>3,641 (-49.6%)</td><td rowspan=1 colspan=1>55.39</td><td rowspan=1 colspan=1>2,128 (-65.1%)</td><td rowspan=1 colspan=1>43.33</td><td rowspan=1 colspan=1>6,563 (-48.9%)</td><td rowspan=1 colspan=1>86.67</td><td rowspan=1 colspan=1>3,427 (-41.3%)</td><td rowspan=1 colspan=1>66.16</td><td rowspan=1 colspan=1>3,574 (-48.7%)</td><td rowspan=1 colspan=1>63.10</td></tr><tr><td rowspan=1 colspan=1>PUMA     90.80</td><td rowspan=1 colspan=1>2,338 (-17.5%)</td><td rowspan=1 colspan=1>60.94</td><td rowspan=1 colspan=1>5,645 (-21.8%)</td><td rowspan=1 colspan=1>63.97</td><td rowspan=1 colspan=1>5,475 (-10.3%)</td><td rowspan=1 colspan=1>53.33</td><td rowspan=1 colspan=1>11,011 (-14.3%)</td><td rowspan=1 colspan=1>92.50</td><td rowspan=1 colspan=1>5,022 (-14.0%)</td><td rowspan=1 colspan=1>72.31</td><td rowspan=1 colspan=1>5,898 (-15.4%)</td><td rowspan=1 colspan=1>67.67</td></tr><tr><td rowspan=1 colspan=1>DEER      90.87</td><td rowspan=1 colspan=1>2,232 (-21.2%)</td><td rowspan=1 colspan=1>62.07</td><td rowspan=1 colspan=1>5,850 (-19.0%)</td><td rowspan=1 colspan=1>65.15</td><td rowspan=1 colspan=1>6,244 (+2.3%)</td><td rowspan=1 colspan=1>48.89</td><td rowspan=1 colspan=1>11,396 (-11.3%)</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=1>4,739 (-18.9%)</td><td rowspan=1 colspan=1>72.40</td><td rowspan=1 colspan=1>6,092 (-12.6%)</td><td rowspan=1 colspan=1>67.64</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)91.33</td><td rowspan=1 colspan=1>2,387 (-15.8%)</td><td rowspan=1 colspan=1>62.52</td><td rowspan=1 colspan=1>5,885 (-18.5%)</td><td rowspan=1 colspan=1>64.65</td><td rowspan=1 colspan=1>4,729 (-22.5%)</td><td rowspan=1 colspan=1>51.11</td><td rowspan=1 colspan=1>11,109 (-13.5%)</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=1>4,613 (-21.0%)</td><td rowspan=1 colspan=1>72.92</td><td rowspan=1 colspan=1>5,745 (-17.6%)</td><td rowspan=1 colspan=1>68.38</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)91.33</td><td rowspan=1 colspan=1>2,258 (-20.3%)</td><td rowspan=1 colspan=1>61.98</td><td rowspan=1 colspan=1>5,695 (-21.1%)</td><td rowspan=1 colspan=1>64.81</td><td rowspan=1 colspan=1>4,620 (-24.3%)</td><td rowspan=1 colspan=1>52.22</td><td rowspan=1 colspan=1>10,581 (-17.6%)</td><td rowspan=1 colspan=1>95.83</td><td rowspan=1 colspan=1>4,173 (-28.6%)</td><td rowspan=1 colspan=1>73.23</td><td rowspan=1 colspan=1>5,465 (-21.6%)</td><td rowspan=1 colspan=1>68.87</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>DeepSeek-R1-Distill-Llama</td><td rowspan=1 colspan=1>-8B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    73.40</td><td rowspan=1 colspan=1>3,558</td><td rowspan=1 colspan=1>48.20</td><td rowspan=1 colspan=1>8,161</td><td rowspan=1 colspan=1>16.50</td><td rowspan=1 colspan=1>7,372</td><td rowspan=1 colspan=1>30.00</td><td rowspan=1 colspan=1>13,999</td><td rowspan=1 colspan=1>81.67</td><td rowspan=1 colspan=1>6,487</td><td rowspan=1 colspan=1>49.95</td><td rowspan=1 colspan=1>7,915</td><td rowspan=1 colspan=1>44.12</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   59.60</td><td rowspan=1 colspan=1>910 (-74.4%)</td><td rowspan=1 colspan=1>28.49</td><td rowspan=1 colspan=1>1,445 (-82.3%)</td><td rowspan=1 colspan=1>27.78</td><td rowspan=1 colspan=1>543 (-92.6%)</td><td rowspan=1 colspan=1>11.11</td><td rowspan=1 colspan=1>1,737 (-87.6%)</td><td rowspan=1 colspan=1>35.83</td><td rowspan=1 colspan=1>1,063 (-83.6%)</td><td rowspan=1 colspan=1>32.56</td><td rowspan=1 colspan=1>1,140 (-85.6%)</td><td rowspan=1 colspan=1>31.48</td></tr><tr><td rowspan=1 colspan=1>Dynasor    67.07</td><td rowspan=1 colspan=1>2,788 (-21.6%)</td><td rowspan=1 colspan=1>41.93</td><td rowspan=1 colspan=1>4,990 (-38.9%)</td><td rowspan=1 colspan=1>22.56</td><td rowspan=1 colspan=1>2,195 (-70.2%)</td><td rowspan=1 colspan=1>22.22</td><td rowspan=1 colspan=1>11,489 (-17.9%)</td><td rowspan=1 colspan=1>68.33</td><td rowspan=1 colspan=1>4,657 (-28.2%)</td><td rowspan=1 colspan=1>44.42</td><td rowspan=1 colspan=1>5,224 (-34.0%)</td><td rowspan=1 colspan=1>40.22</td></tr><tr><td rowspan=1 colspan=1>TRACE     69.87</td><td rowspan=1 colspan=1>2,546 (-28.4%)</td><td rowspan=1 colspan=1>46.57</td><td rowspan=1 colspan=1>4,012 (-50.8%)</td><td rowspan=1 colspan=1>27.10</td><td rowspan=1 colspan=1>2,860 (-61.2%)</td><td rowspan=1 colspan=1>22.22</td><td rowspan=1 colspan=1>5,824 (-58.4%)</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>4,177 (-35.6%)</td><td rowspan=1 colspan=1>47.15</td><td rowspan=1 colspan=1>3,884 (-50.9%)</td><td rowspan=1 colspan=1>43.87</td></tr><tr><td rowspan=1 colspan=1>PUMA     71.53</td><td rowspan=1 colspan=1>2,751 (-22.7%)</td><td rowspan=1 colspan=1>45.68</td><td rowspan=1 colspan=1>5,436 (-33.4%)</td><td rowspan=1 colspan=1>25.42</td><td rowspan=1 colspan=1>6,942 (-5.8%)</td><td rowspan=1 colspan=1>27.78</td><td rowspan=1 colspan=1>12,697 (-9.3%)</td><td rowspan=1 colspan=1>77.50</td><td rowspan=1 colspan=1>5,063 (-22.0%)</td><td rowspan=1 colspan=1>49.58</td><td rowspan=1 colspan=1>6,578 (-16.9%)</td><td rowspan=1 colspan=1>44.53</td></tr><tr><td rowspan=1 colspan=1>DEER      72.13</td><td rowspan=1 colspan=1>3,163 (-11.1%)</td><td rowspan=1 colspan=1>47.51</td><td rowspan=1 colspan=1>7,821 (-4.2%)</td><td rowspan=1 colspan=1>16.84</td><td rowspan=1 colspan=1>7,246 (-1.7%)</td><td rowspan=1 colspan=1>32.22</td><td rowspan=1 colspan=1>12,728 (-9.1%)</td><td rowspan=1 colspan=1>75.83</td><td rowspan=1 colspan=1>6,582 (+1.5%)</td><td rowspan=1 colspan=1>48.91</td><td rowspan=1 colspan=1>7,508 (-5.1%)</td><td rowspan=1 colspan=1>43.31</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)75.00</td><td rowspan=1 colspan=1>2,930 (-17.7%)</td><td rowspan=1 colspan=1>47.36</td><td rowspan=1 colspan=1>6,586 (-19.3%)</td><td rowspan=1 colspan=1>24.92</td><td rowspan=1 colspan=1>5,659 (-23.2%)</td><td rowspan=1 colspan=1>28.89</td><td rowspan=1 colspan=1>11,986 (-14.4%)</td><td rowspan=1 colspan=1>73.33</td><td rowspan=1 colspan=1>5,139 (-20.8%)</td><td rowspan=1 colspan=1>49.90</td><td rowspan=1 colspan=1>6,460 (-18.4%)</td><td rowspan=1 colspan=1>44.92</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)74.07</td><td rowspan=1 colspan=1>2,844 (-20.1%)</td><td rowspan=1 colspan=1>46.96</td><td rowspan=1 colspan=1>6,489 (-20.5%)</td><td rowspan=1 colspan=1>25.25</td><td rowspan=1 colspan=1>5,313 (-27.9%)</td><td rowspan=1 colspan=1>28.89</td><td rowspan=1 colspan=1>11,412 (-18.5%)</td><td rowspan=1 colspan=1>75.83</td><td rowspan=1 colspan=1>5,157 (-20.5%)</td><td rowspan=1 colspan=1>50.20</td><td rowspan=1 colspan=1>6,243 (-21.1%)</td><td rowspan=1 colspan=1>45.35</td></tr></table>

Table 2: Averaged results using three seeds across DeepSeek-R1 distillation models. Acc: accuracy. Tok: average token count. TE: token efficiency. Values in parentheses are token changes relative to Full CoT. The best and second-best accuracy and TE values among the efficient-reasoning methods are shown in bold and underlined, respectively; Full CoT is excluded from this ranking.

natural logarithm, accuracy in percentage points, and T the average token count.

Implementation Details. All experiments use 4 NVIDIA L40 GPUs, and all methods run under one vLLM host. The main generation budget is 32,768 tokens. If this budget ends before the thought closes, the answer continuation is at most 2,048 tokens. Each reported cell is the mean over seeds 42, 0, and 1. The main experiments set the window length to k = 4, and the minimum stop step is MSS = 10, so b<sup>⋆</sup> is no earlier than step 10. Section 6.3 compares other lengths. We provide a more detailed analysis at Appendix F.

## 5.2 Main Results

Overall Performance. As shown in Table 2, CALM achieves the highest overall TE on all four backbones. Relative to Full CoT, CALM (Hard) reduces overall token usage by 18.6%–25.0%, with overall accuracy 0.07–1.48 points higher. It also achieves higher overall accuracy with fewer tokens than both PUMA and DEER on every backbone. Both CALM variants reduce token usage across all 20 model–benchmark settings, although their effects on accuracy vary across tasks. Table 6 in Appendix E also presents results on six additional models, where CALM similarly reduces token consumption by 14%–18.9% and achieves better accuracy and token efficiency than other baselines.

Comparison with Early-Exit Methods. Earlyexit baselines exhibit different accuracy–efficiency trade-offs. Answer Convergence reduces overall token usage by 82.7%–88.3%, but lowers overall accuracy by 17.39–35.93 percentage points relative to Full CoT. In contrast, DEER increases token usage by 0.7%–2.3% on GPQA-Diamond for the three Qwen backbones, because its high threshold almost never triggers early exit on GPQA-Diamond and trial-answer probing adds tokens. In Appendix G, we provide detailed statistics on the proportion of this early-stopping signal (stop coverage) and find that CALM intervenes on more questions than PUMA and DEER truncate in most settings, which helps explain the larger token reductions on some benchmarks. Overall, the superiority of CALM comes from the suppression of ATP after consecutive identical answers, while subsequent deduction and correction continue without degrading intermediate reasoning quality.

![](images/c20150ad671e19fa3250d980418655fa8817096499ae5904b52d828e16e9cbcd.jpg)  
Figure 4: Comparison on right-to-wrong transition rates where a problem answered correctly by Full CoT is answered incorrectly by an efficient reasoning method.

Soft vs. Hard Masking. Hard masking achieves lower overall token usage than soft masking on all four backbones, whereas their accuracy advantage depends on the backbone. Soft masking attains the highest overall accuracy on Qwen-7B and Qwen-14B, reaching 66.47% and 72.75%, respectively, while reducing overall token usage by 21.0% and 14.3% relative to Full CoT. Hard masking achieves even higher overall accuracy and TE on Qwen-32B and Llama-8B. Also, it consistently provides greater overall token savings, while the preferred variant for accuracy depends on the backbone. Overall, both strategies achieve better results across all evaluated model architectures, demonstrating the effectiveness of decoding-based intervention.

Scales and Domains. The benefits of CALM generalize across model capacities and problem distributions. Across the DeepSeek-R1-Distill-Qwen family spanning 7B to 32B parameters, generation length is shortened while accuracy stays close to full CoT. The largest reduction is 25.0% on the 7B model. The same pattern appears on competition mathematics (MATH-500, OlympiadBench, and AIME 2025) and on GPQA-Diamond, demonstrating the applicability of our method across diverse challenging reasoning tasks.

<table><tr><td>Configuration / Masked Target</td><td>Acc↑</td><td>Tok↓</td><td>TE↑</td></tr><tr><td>Full CoT</td><td></td><td>92.33 3,847 89.07</td><td rowspan="4"></td></tr><tr><td>CALM (Wait, Alternatively, Hmm) 92.27 2,843 89.77</td><td></td><td></td></tr><tr><td>- w/o Wait - w/o Alternatively - w/o Hmm</td><td></td><td>92.53 3,577 89.47 92.00 3,104 89.30 92.00 2,896 89.46</td></tr><tr><td>Individual Suppression (reflection tokens) Suppress Wait</td><td></td><td>92.20 3,042 89.55</td></tr><tr><td>Suppress Alternatively Suppress Hmm</td><td></td><td>92.33 3,501 89.33 92.67 3,532 89.65</td><td rowspan="2"></td></tr><tr><td>Individual Suppression (connective control tokens) Suppress But Suppress So Suppress Therefore 91.87 4,000 88.51</td><td>91.40 3,255 88.59 92.40 3,865 89.13</td></tr><tr><td colspan="2">Suppress However Suppress Maybe Full Vocabulary Suppression</td><td>91.403,464 88.43 92.13 3,534 89.10</td></tr></table>

Table 3: ATP vocabulary ablation with CALM (Hard).

## 6 Analysis

We conduct further analysis of our method and CALM (Soft) is used unless otherwise noted.

## 6.1 Right-to-Wrong Transitions

A critical risk of reducing overthinking is that early intervention may cause errors on questions that Full CoT answers correctly. We examine the degradation rate by tracking the share of all questions on which Full CoT is correct and the method is incorrect. Figure 4 shows the results and we find that: 1) Premature truncation corrupts effective reasoning. Truncation-based early-exit baselines such as Dynasor and DEER suffer from high degradation rates. Their risk is that stopping once a local signal is met can leave out later steps that may still revise the answer. 2) Our method Effectively Mitigates Performance Degradation. By keeping the generation trajectory open, CALM substantially mitigates this corruption of performance with the lowest degradation rate.

## 6.2 Ablation on ATP Vocabulary

We compare the results of suppressing different tokens in the ATP vocabulary V<sub>ATP</sub>. Results on MATH-500 by using DeepSeek-R1-Distill-Qwen-7B are shown in Table 3. We first find that although reduces token length (3,316 vs. 3,847), it also decreases accuracy by 2 percentage points. Under individual suppression, both reflection and connective control tokens can almost reduce tokens, but suppressing control tokens leads to performance degradation, resulting in lower overall token efficiency, whereas suppressing anthropomorphic tokens alone maintains accuracy. We achieve the best result of 89.77 when jointly suppressing the reflection words “Wait”, “Alternatively”, and “Hmm”. These results illustrate the roles of different thinking tokens in the ATP vocabulary.

<table><tr><td rowspan="2">Variant</td><td colspan="2">MATH-500</td><td colspan="2">Olympiad</td><td colspan="2">GPQA</td><td colspan="2">AIME25</td><td colspan="2">AMC23</td><td colspan="3">Overall</td></tr><tr><td>Acc↑</td><td>Tok↓</td><td>Acc↑</td><td>Tok↓</td><td>Acc↑</td><td>Tok↓</td><td>Acc↑</td><td>Tok↓</td><td>Acc↑</td><td>Tok↓</td><td>Acc↑</td><td>Tok↓</td><td>TE↑</td></tr><tr><td>Full CoT</td><td>91.40</td><td>3,820</td><td>57.93</td><td>8,677</td><td>45.96</td><td>7,486</td><td>40.00</td><td>14,889</td><td>87.50</td><td>6,579</td><td>64.56</td><td>8,290</td><td>58.52</td></tr><tr><td>ρ = 0.80</td><td>92.00</td><td>2,977 (-22.1%)</td><td>57.04</td><td>6,863 (-20.9%)</td><td>47.98</td><td>4,955 (-33.8%)</td><td>40.00</td><td>13,641 (-8.4%)</td><td>87.50</td><td>5,247 (-20.2%)</td><td>64.90</td><td>6,737 (-18.7%)</td><td>59.75</td></tr><tr><td>ρ = 0.90</td><td>91.80</td><td>3,019 (-21.0%)</td><td>58.37</td><td>6,661 (-23.2%)</td><td>50.00</td><td>5,084 (-32.1%)</td><td>36.67</td><td>14,317 (-3.8%)</td><td>87.50</td><td>5,286 (-19.7%)</td><td>64.87</td><td>6,873 (-17.1%)</td><td>59.64</td></tr><tr><td>ρ = 0.95</td><td>91.00</td><td>3,165 (-17.1%)</td><td>56.89</td><td>7,067 (-18.6%)</td><td>43.43</td><td>5,527 (-26.2%)</td><td>33.33</td><td>14,149 (-5.0%)</td><td>85.00</td><td>5,165 (-21.5%)</td><td>61.93</td><td>7,015 (-15.4%)</td><td>56.61</td></tr><tr><td>ρ = 0.98</td><td>92.40</td><td>3,197 (-16.3%)</td><td>59.70</td><td>7,061 (-18.6%)</td><td>48.99</td><td>5,270 (-29.6%)</td><td>46.67</td><td>11,976 (-19.6%)</td><td>87.50</td><td>4,972 (-24.4%)</td><td>67.05</td><td>6,495 (-21.7%)</td><td>62.05</td></tr><tr><td>ρ = 0.99</td><td>91.00</td><td>3,257 (-14.7%)</td><td>57.33</td><td>7,122 (-17.9%)</td><td>47.98</td><td>5,486 (-26.7%)</td><td>40.00</td><td>12,768 (-14.2%)</td><td>82.50</td><td>5,772 (-12.3%)</td><td>63.76</td><td>6,881 (-17.0%)</td><td>58.52</td></tr><tr><td>ρ = 0.995</td><td>91.40</td><td>3,317 (-13.2%)</td><td>58.22</td><td>7,361 (-15.2%)</td><td>49.49</td><td>5,709 (-23.7%)</td><td>40.00</td><td>14,102 (-5.3%)</td><td>85.00</td><td>5,652 (-14.1%)</td><td>64.82</td><td>7,228 (-12.8%)</td><td>59.38</td></tr></table>

Table 4: Comparison of different soft progressive penalty factors using DeepSeek-R1-Distill-Qwen-7B .

![](images/43d397984aa9f0a229d7e16a82fb38723174bb820b46f1f8bb290967fa8bb8aa.jpg)  
Figure 5: Accuracy (blue) and tokens (red) against window length k with CALM (Hard). The line is the mean over three seeds.

## 6.3 Hyperparameter Sensitivity

We evaluate how hyperparameters balance reasoning completeness and token efficiency.

Damping Dynamics via Soft Decay $\rho _ { \bullet }$ Table 4 shows the impact of different values of the progressive penalty factor $\rho .$ We observe that $\rho = 0 . 9 8$ achieves the highest accuracy, the shortest length, and the highest TE (62.05). A steeper decay $( \rho ~ = ~ 0 . 9 5 )$ accumulates the penalty too fast, leading to an accuracy drop. Milder settings $( \rho \ge 0 . 9 9 )$ accumulate the penalty more slowly and exert weaker damping, failing to curb the generation length inflated by redundant verification.

Impact of the Consistency Window Size k. As shown in Figure 5, the performance of CALM varies by only 1.54 percentage points across window lengths $k \in \{ 1 , \ldots , 6 \}$ . The impact on tokens is slightly greater, while all settings substantially reduce token usage relative to Full CoT. We adopt k = 4 as the default, as it achieves the highest accuracy of 65.73% with 6,192 tokens, providing a favorable balance between accuracy and efficiency.

<table><tr><td>Method</td><td>Completeness</td><td>Coherence</td><td>Conciseness</td><td>Justification</td><td>Average</td></tr><tr><td>Full CoT</td><td>91.0</td><td>90.2</td><td>60.5</td><td>89.8</td><td>82.9</td></tr><tr><td>Ans. Conv.</td><td>28.0</td><td>44.3</td><td>79.5</td><td>19.8</td><td>42.9</td></tr><tr><td>Dynasor</td><td>63.2</td><td>70.8</td><td>71.7</td><td>60.0</td><td>66.4</td></tr><tr><td>TRACE</td><td>81.2</td><td>85.0</td><td>61.7</td><td>80.0</td><td>77.0</td></tr><tr><td>PUMA</td><td>85.5</td><td>86.7</td><td>65.2</td><td>84.3</td><td>80.4</td></tr><tr><td>DEER</td><td>78.5</td><td>81.3</td><td>70.0</td><td>76.3</td><td>76.5</td></tr><tr><td>CALM (Soft)</td><td>94.5</td><td>92.3</td><td>62.7</td><td>94.0</td><td>85.9</td></tr><tr><td>CALM (Hard)</td><td>92.8</td><td>92.7</td><td>65.3</td><td>92.2</td><td>85.8</td></tr></table>

Table 5: Results by LLM-as-a-judge using Qwen3.5- 397B-A17B. Scores run from 10 to 100 in steps of 10.

## 6.4 Comparison through LLM-as-a-Judge

Beyond accuracy and efficiency comparisons, we evaluate reasoning-chain quality using Qwen3.5- 397B-A17B through LLM-as-a-judge along dimensions of completeness, coherence, conciseness, and justification. Results in Table 5 show that CALM achieves a higher average than the other methods and scores higher than them on completeness, coherence, and justification. In particular, after consecutive answer agreement, Full CoT still triggers ATP repeatedly, and unresolved contradictions and derivations abandoned midway make its completeness and justification lower than those of CALM. The detailed setup and a few case studies are given in Appendix I and Appendix J, respectively.

## 7 Conclusion

We proposed consistency-aware logit masking (CALM), a simple yet effective training-free decoding strategy that detects consecutive answer agreement and then suppresses the logits of action transition point tokens to reduce redundant reasoning while preserving generation continuity. Extensive experiments on ten advanced reasoning models and five benchmarks show that CALM substantially reduces token usage while maintaining or even slightly improving accuracy relative to full CoT, offering a practical decoding-space solution to overthinking, and future work may explore adaptive masking policies, broader task domains, and combinations with reinforcement-learning or prompt-based efficiency methods.

## Limitations

CALM reduces generation length while keeping accuracy close to full chain-of-thought. Two limitations remain open for future work.First, the main experiments use one window length k for every item, without varying it by problem. Adapting the window length or the logit penalty to each instance, for example from token-level entropy, is a direction for future work. Second, the evaluation in this paper covers textual mathematics and science reasoning. Whether these transition tokens are used to handle external feedback in tool-use or multimodal settings is not tested here. Extending the suppression to those settings would require separating transitions that still serve a function from the meaningless exploration or verification that follows consecutive identical answers.

## References

Pranjal Aggarwal and Sean Welleck. 2025. L1: Controlling how long a reasoning model thinks with reinforcement learning. In Second Conference on Language Modeling.

AI-MO. 2024. AMC 2023. https://huggingface. co/datasets/AI-MO/aimo-validation-amc.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, et al. 2025. Llama-nemotron: Efficient reasoning models. arXiv preprint arXiv:2505.00949.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025. Do NOT think that much for 2+3=? On the overthinking of long reasoning models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 9487–9499. PMLR.

Alejandro Cuadron, Dacheng Li, Wenjie Ma, Xingyao Wang, Yichuan Wang, Siyuan Zhuang, Shu Liu, Luis Gaspar Schroeder, Tian Xia, Huanzhi Mao, et al. 2025. The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks. arXiv preprint arXiv:2502.08235.

Muzhi Dai, Chenxu Yang, and Qingyi Si. 2025. S-grpo: Early exit via reinforcement learning in reasoning

models. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 48178–48204. Curran Associates, Inc.

Zheng Du, Hao Kang, Song Han, Tushar Krishna, and Ligeng Zhu. 2025. Ockbench: Measuring the efficiency of llm reasoning. arXiv preprint arXiv:2511.05722.

Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Yonghao Zhuang, Yian Ma, Aurick Qiao, Tajana Rosing, Ion Stoica, et al. 2024. Efficiently scaling llm reasoning with certaindex. arXiv preprint arXiv:2412.20993.

Aryo Pradipta Gema, Alexander Hägele, Runjin Chen, Andy Arditi, Jacob Goldman-Wetzler, Kit Fraser-Taliente, Henry Sleight, Linda Petrini, Julian Michael, Beatrice Alex, Pasquale Minervini, Yanda Chen, Joe Benton, and Ethan Perez. 2025. Inverse scaling in test-time compute. Transactions on Machine Learning Research.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand. Association for Computational Linguistics.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. 2026. Thinkprune: Pruning long chain-of-thought of LLMs via reinforcement learning. Transactions on Machine Learning Research.

Seunghee Koh, Sungjae Choi, Minchan Kwon, Sunghyun Baek, and Junmo Kim. 2026. Doesn’t stop reasoning: Analysis of spurious cot termination. arXiv preprint arXiv:2609.03633.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, page 611–626, New York, NY, USA. Association for Computing Machinery.

Jiakun Li, Xingwei He, Kefan Li, Hongzheng Chai, Hongyue Yu, and Yuan Yuan. 2026a. Efficient testtime scaling via temporal reasoning aggregation. In Findings of the Association for Computational Linguistics: ACL 2026, pages 13304–13318, San Diego, California, United States. Association for Computational Linguistics.

Zeju Li, Jianyuan Zhong, Ziyang Zheng, Xiangyu Wen, Zhijian Xu, Yingying Cheng, Fan Zhang, and Qiang Xu. 2026b. Making slow thinking faster: Compressing LLM chain-of-thought via step entropy. In The Fourteenth International Conference on Learning Representations.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Junhong Lin, Xinyue Zeng, Jie Zhu, Song Wang, Julian Shun, Jun Wu, and Dawei Zhou. 2026. Plan and budget: Effective and efficient test-time scaling on reasoning large language models. In The Fourteenth International Conference on Learning Representations.

Xin Liu and Lu Wang. 2025. Answer convergence as a signal for early stopping in reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17896–17907, Suzhou, China. Association for Computational Linguistics.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025. Understanding r1-zero-like training: A critical perspective. In Second Conference on Language Modeling.

MAA Committees. 2025. Aime problems and solutions. https://artofproblemsolving.com/wiki/ index.php/AIME\_Problems\_and\_Solutions.

Dehai Min, Giovanni Vaccarino, Huiyi Chen, Yongliang Wu, Gal Yona, and Lu Cheng. 2026. Stop when reasoning converges: Semantic-preserving early exit for reasoning models. arXiv preprint arXiv:2605.17672.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20275–20321, Suzhou, China. Association for Computational Linguistics.

OpenAI. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Satwik Pandey, Suresh Raghu, and Shashwat Pandey. 2026. SELFDOUBT: Uncertainty quantification for reasoning LLMs via the hedge-to-verify ratio. In Workshop on Failure Modes ofAgentic AI at ICML 2026.

Qwen Team. 2025. Qwq-32b: Embracing the power of reinforcement learning.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA:

A graduate-level google-proof Q&A benchmark. In First Conference on Language Modeling.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations.

Jinyan Su, Jennifer Healey, Preslav Nakov, and Claire Cardie. 2025. Between underthinking and overthinking: An empirical study of reasoning length and correctness in llms. arXiv preprint arXiv:2505.00127.

Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, and Wei Wang. 2026. Stop when enough: Adaptive early-stopping for chain-of-thought reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27250–27268, San Diego, California, United States. Association for Computational Linguistics.

Chenlong Wang, Yuanning Feng, Dongping Chen, Zhaoyang Chu, Ranjay Krishna, and Tianyi Zhou. 2025a. Wait, we don’t need to “wait”! removing thinking tokens improves reasoning efficiency. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 7459–7482, Suzhou, China. Association for Computational Linguistics.

Hanyang Wang and Mingxuan Zhu. 2026. The detection-extraction gap: Models know the answer before they can say it. arXiv preprint arXiv:2604.06613.

Shouren Wang, Wang Yang, Xianxuan Long, Qifan Wang, Vipin Chaudhary, and Xiaotian Han. 2025b. Demystifying hybrid thinking: Can llms truly switch between think and no-think? arXiv preprint arXiv:2510.12680.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.

Tsung-Han Wu, Mihran Miroyan, David M. Chan, Trevor Darrell, Narges Norouzi, and Joseph E. Gonzalez. 2026. Are large reasoning models interruptible? In Forty-third International Conference on Machine Learning.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. 2025. Chain of draft: Thinking faster by writing less. arXiv preprint arXiv:2502.18600.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. 2026. Dynamic early exit in

reasoning models. In The Fourteenth International Conference on Learning Representations.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. 2025. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387.

Yahan Yu, Noa Nakanishi, and Fei Cheng. 2026. Revisiting anthropomorphic reflection markers in large language model reasoning. arXiv preprint arXiv:2605.28305.

Ye Yu, Yaoning Yu, Haibo Jin, and Haohan Wang. 2025. PREMISE: Scalable and strategic prompt optimization for efficient mathematical reasoning in large reasoning models. In NeurIPS 2025 Workshop on Efficient Reasoning.

Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. 2025. LightThinker: Thinking step-by-step compression. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 13307–13328, Suzhou, China. Association for Computational Linguistics.

Xinliang Frederick Zhang, Anhad Mohananey, Alexandra Chronopoulou, Pinelopi Papalampidi, Somit Gupta, Tsendsuren Munkhdalai, Lu Wang, and Shyam Upadhyay. 2026. Do LLMs really need 10+ thoughts for “find the time 1000 days later”? towards structural understanding of LLM overthinking. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 17005–17030, San Diego, California, United States. Association for Computational Linguistics.

## A Reasoning Step Segmentation

Trial answers are probed at the level of reasoning steps, so the thought before ⟨/think⟩ is segmented by the deterministic, lightweight procedure of PUMA (Min et al., 2026). Given the raw reasoning text, we first split the chain at blank-line boundaries, which preserves the model’s natural paragraph structure. We then assign each paragraph a coarse semantic role using simple string-level cues, such as problem setup, calculation, doubt, verification, conclusion, or general reasoning. These labels are not used as model inputs; they only guide how adjacent paragraphs are grouped into larger reasoning steps.

The merging stage balances semantic coherence and step length. Enumerated paragraphs or major semantic transitions are treated as natural step boundaries, while short adjacent paragraphs with compatible roles are merged when doing so preserves coherence. We use a target step length range of $[ L _ { \operatorname* { m i n } } , L _ { \operatorname* { m a x } } ] = [ 2 0 0 , 1 0 0 0 ]$ characters, and tiny trailing segments are merged into the preceding step. This merged span is one reasoning step, and the trial answer is probed at its boundary.

## B Trial Answer Inducing Suffix

For mathematics and contest problems, the answerinducing suffix is:

\n\*\*Final Answer\*\*\n\nThe final answer is \boxed. For GPQA-Diamond, the suffix is:

\n\*\*Final Answer\*\*\n\nThe answer choice is \boxed.

Before two trial answers are compared, the boxed strings are normalized by lowercasing, deleting whitespace, and stripping formatting macros.

## C Detailed Settings in Pilot Study

Samples. Figure 2 uses DeepSeek-R1-Distill-Qwen-7B on OlympiadBench . Vanilla CoT and ATP suppression use the same questions. Only questions whose trial answers are identical for four consecutive steps for the first time are included, 633 of the 675 questions. Accuracy and tokens use these 633 questions as a fixed denominator and give each question equal weight. The dashed line is the completed vanilla CoT accuracy on these questions.

Horizontal Axis. The horizontal axis is zero when the four consecutive identical trial answers end. The horizontal coordinate after that point is the mean, over the 633 questions, of the tokens added at each step after agreement. A question that has already stopped remains at the tokens it actually added after agreement. After ⟨/think⟩, the curve continues only once every included question has finished its thought, and that segment counts ⟨/think⟩ together with the following text.

Vertical Axis. At that position, accuracy is the fraction of the 633 questions whose trial answer is correct.The trial answer is produced by continuing from that prefix for at most 30 tokens and closing the answer in \boxed{}.A finished question keeps its last correctness.

## D ATP Surface Forms and Token Matching

• Wait, wait, WAIT

• Alternatively, alternatively, ALTERNATIVELY

• H or h followed by r lowercase ms, $r \in \{ 1 , \ldots , 1 2 \}$

Each string is also included with one leading space, 60 strings in all.Each string is encoded with that model’s tokenizer and add\_special\_tokens=False. Empty encodings are discarded, and identical token-ID sequences are deduplicated.A one-token sequence belongs to $B _ { t }$ at every position. A longer sequence contributes its last token only when the current prefix ends with the preceding tokens. Membership in $B _ { t }$ only marks a completion candidate. The logit changes only after agreement, by the Hard or Soft rule.

## E Main Results on Additional Models

Table 6 uses the same columns as Table 2 and reports accuracy and tokens on the five benchmarks for Llama-3.1-Nemotron-Nano-8B, Qwen3 (4B, 8B, 30B-A3B, and 32B), and QwQ-32B, together with the equal-weight overall and its TE. Each cell is the mean over three seeds, and the parentheses on tokens are the change relative to Full CoT for the same model and benchmark.

## F Experimental Details

For AMC 2023, we use the 40 questions from LIMO (Ye et al., 2025), following DEER.

All methods run under one vLLM host (Kwon et al., 2023).The main generation budget is 32,768 tokens.If this budget ends before the thought closes, the answer continuation is at most 2,048 tokens.The prompt reserve is 3,072 tokens, and max \_model\_len is 37,888.For a question with a finite agreement boundary $b ^ { \star }$ , the generated thought is kept.When the model emits </think> within this budget, the following suffix is appended:

<table><tr><td rowspan=1 colspan=13>MATH-500        Olympiad          GPQA          AIME25           AMC23              OverallMethodAcc↑   Tok↓   Acc↑   Tok↓   Acc↑   Tok↓   Acc↑   Tok.↓    Acc↑   Tok↓   Acc↑   Tok↓    TE↑</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>Llama-3.1-Nemotron-Nano-8B</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    94.07</td><td rowspan=1 colspan=1>3,085</td><td rowspan=1 colspan=1>63.95</td><td rowspan=1 colspan=1>6,923</td><td rowspan=1 colspan=1>47.98</td><td rowspan=1 colspan=1>6,973</td><td rowspan=1 colspan=1>51.11</td><td rowspan=1 colspan=1>11,860</td><td rowspan=1 colspan=1>94.17</td><td rowspan=1 colspan=1>5,258</td><td rowspan=1 colspan=1>70.26</td><td rowspan=1 colspan=1>6,820</td><td rowspan=1 colspan=1>65.06</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   57.20</td><td rowspan=1 colspan=1>804 (-73.9%)</td><td rowspan=1 colspan=1>25.28</td><td rowspan=1 colspan=1>1,056 (-84.7%)</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>336 (-95.2%)</td><td rowspan=1 colspan=1>4.44</td><td rowspan=1 colspan=1>1,183 (-90.0%)</td><td rowspan=1 colspan=1>28.33</td><td rowspan=1 colspan=1>807 (-84.7%)</td><td rowspan=1 colspan=1>26.39</td><td rowspan=1 colspan=1>837 (-87.7%)</td><td rowspan=1 colspan=1>25.59</td></tr><tr><td rowspan=1 colspan=1>Dynasor    93.00</td><td rowspan=1 colspan=1>3,288 (+6.6%)</td><td rowspan=1 colspan=1>63.95</td><td rowspan=1 colspan=1>7,081 (+2.3%)</td><td rowspan=1 colspan=1>46.97</td><td rowspan=1 colspan=1>7,664 (+9.9%)</td><td rowspan=1 colspan=1>45.56</td><td rowspan=1 colspan=1>12,651 (+6.7%)</td><td rowspan=1 colspan=1>93.33</td><td rowspan=1 colspan=1>5,449 (+3.6%)</td><td rowspan=1 colspan=1>68.56</td><td rowspan=1 colspan=1>7,227 (+6.0%)</td><td rowspan=1 colspan=1>63.12</td></tr><tr><td rowspan=1 colspan=1>TRACE    90.67</td><td rowspan=1 colspan=1>2,296 (-25.6%)</td><td rowspan=1 colspan=1>59.16</td><td rowspan=1 colspan=1>3,734 (-46.1%)</td><td rowspan=1 colspan=1>37.21</td><td rowspan=1 colspan=1>2,313 (-66.8%)</td><td rowspan=1 colspan=1>35.56</td><td rowspan=1 colspan=1>6,814 (-42.5%)</td><td rowspan=1 colspan=1>85.83</td><td rowspan=1 colspan=1>3,500 (-33.4%)</td><td rowspan=1 colspan=1>61.69</td><td rowspan=1 colspan=1>3,731 (-45.3%)</td><td rowspan=1 colspan=1>58.52</td></tr><tr><td rowspan=1 colspan=1>PUMA     92.73</td><td rowspan=1 colspan=1>2,537 (-17.8%)</td><td rowspan=1 colspan=1>62.96</td><td rowspan=1 colspan=1>5,023 (-27.4%)</td><td rowspan=1 colspan=1>46.63</td><td rowspan=1 colspan=1>5,911 (-15.2%)</td><td rowspan=1 colspan=1>46.67</td><td rowspan=1 colspan=1>10,012 (-15.6%)</td><td rowspan=1 colspan=1>91.67</td><td rowspan=1 colspan=1>4,370 (-16.9%)</td><td rowspan=1 colspan=1>68.13</td><td rowspan=1 colspan=1>5,571 (-18.3%)</td><td rowspan=1 colspan=1>63.70</td></tr><tr><td rowspan=1 colspan=1>DEER      92.13</td><td rowspan=1 colspan=1>2,202 (-28.6%)</td><td rowspan=1 colspan=1>63.56</td><td rowspan=1 colspan=1>5,355 (-22.6%)</td><td rowspan=1 colspan=1>48.82</td><td rowspan=1 colspan=1>7,303 (+4.7%)</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>11,235 (-5.3%)</td><td rowspan=1 colspan=1>91.67</td><td rowspan=1 colspan=1>4,453 (-15.3%)</td><td rowspan=1 colspan=1>69.24</td><td rowspan=1 colspan=1>6,110 (-10.4%)</td><td rowspan=1 colspan=1>64.47</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)93.33</td><td rowspan=1 colspan=1>2,770 (-10.2%)</td><td rowspan=1 colspan=1>64.20</td><td rowspan=1 colspan=1>6,262 (-9.5%)</td><td rowspan=1 colspan=1>50.34</td><td rowspan=1 colspan=1>6,120 (-12.2%)</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>10,537 (-11.2%)</td><td rowspan=1 colspan=1>94.17</td><td rowspan=1 colspan=1>4,643 (-11.7%)</td><td rowspan=1 colspan=1>70.41</td><td rowspan=1 colspan=1>6,066 (-11.1%)</td><td rowspan=1 colspan=1>65.67</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)93.80</td><td rowspan=1 colspan=1>2,646 (-14.2%)</td><td rowspan=1 colspan=1>63.36</td><td rowspan=1 colspan=1>5,913 (-14.6%)</td><td rowspan=1 colspan=1>46.97</td><td rowspan=1 colspan=1>5,757 (-17.4%)</td><td rowspan=1 colspan=1>51.11</td><td rowspan=1 colspan=1>10,097 (-14.9%)</td><td rowspan=1 colspan=1>93.33</td><td rowspan=1 colspan=1>4,509 (-14.2%)</td><td rowspan=1 colspan=1>69.71</td><td rowspan=1 colspan=1>5,784 (-15.2%)</td><td rowspan=1 colspan=1>65.15</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3-</td><td rowspan=1 colspan=1>4B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    94.07</td><td rowspan=1 colspan=1>5,042</td><td rowspan=1 colspan=1>66.96</td><td rowspan=1 colspan=1>10,492</td><td rowspan=1 colspan=1>55.39</td><td rowspan=1 colspan=1>8,077</td><td rowspan=1 colspan=1>68.89</td><td rowspan=1 colspan=1>17,826</td><td rowspan=1 colspan=1>96.67</td><td rowspan=1 colspan=1>7,862</td><td rowspan=1 colspan=1>76.40</td><td rowspan=1 colspan=1>9,860</td><td rowspan=1 colspan=1>69.54</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   57.07</td><td rowspan=1 colspan=1>976 (-80.6%)</td><td rowspan=1 colspan=1>27.46</td><td rowspan=1 colspan=1>1,314 (-87.5%)</td><td rowspan=1 colspan=1>37.88</td><td rowspan=1 colspan=1>322 (-96.0%)</td><td rowspan=1 colspan=1>8.89</td><td rowspan=1 colspan=1>1,938 (-89.1%)</td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>1,098 (-86.0%)</td><td rowspan=1 colspan=1>33.76</td><td rowspan=1 colspan=1>1,130 (-88.5%)</td><td rowspan=1 colspan=1>32.69</td></tr><tr><td rowspan=1 colspan=1>Dynasor    93.13</td><td rowspan=1 colspan=1>6,009 (+19.2%)</td><td rowspan=1 colspan=1>65.38</td><td rowspan=1 colspan=1>11,966 (+14.0%)</td><td rowspan=1 colspan=1>55.22</td><td rowspan=1 colspan=1>8,748 (+8.3%)</td><td rowspan=1 colspan=1>63.33</td><td rowspan=1 colspan=1>19,919 (+11.7%)</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=1>9,410 (+19.7%)</td><td rowspan=1 colspan=1>74.41</td><td rowspan=1 colspan=1>11,210 (+13.7%)</td><td rowspan=1 colspan=1>66.89</td></tr><tr><td rowspan=1 colspan=1>TRACE     84.80</td><td rowspan=1 colspan=1>2,679 (-46.9%)</td><td rowspan=1 colspan=1>51.21</td><td rowspan=1 colspan=1>3,557 (-66.1%)</td><td rowspan=1 colspan=1>41.58</td><td rowspan=1 colspan=1>1,983 (-75.4%)</td><td rowspan=1 colspan=1>34.44</td><td rowspan=1 colspan=1>5,583 (-68.7%)</td><td rowspan=1 colspan=1>77.50</td><td rowspan=1 colspan=1>3,957 (-49.7%)</td><td rowspan=1 colspan=1>57.91</td><td rowspan=1 colspan=1>3,552 (-64.0%)</td><td rowspan=1 colspan=1>54.87</td></tr><tr><td rowspan=1 colspan=1>PUMA     91.73</td><td rowspan=1 colspan=1>3,103 (-38.5%)</td><td rowspan=1 colspan=1>62.32</td><td rowspan=1 colspan=1>5,147 (-50.9%)</td><td rowspan=1 colspan=1>53.87</td><td rowspan=1 colspan=1>5,407 (-33.1%)</td><td rowspan=1 colspan=1>56.67</td><td rowspan=1 colspan=1>14,647 (-17.8%)</td><td rowspan=1 colspan=1>93.33</td><td rowspan=1 colspan=1>5,432 (-30.9%)</td><td rowspan=1 colspan=1>71.59</td><td rowspan=1 colspan=1>6,747 (-31.6%)</td><td rowspan=1 colspan=1>66.43</td></tr><tr><td rowspan=1 colspan=1>DEER     93.13</td><td rowspan=1 colspan=1>3,922 (-22.2%)</td><td rowspan=1 colspan=1>66.72</td><td rowspan=1 colspan=1>9,466 (-9.8%)</td><td rowspan=1 colspan=1>53.70</td><td rowspan=1 colspan=1>7,542 (-6.6%)</td><td rowspan=1 colspan=1>62.22</td><td rowspan=1 colspan=1>16,603 (-6.9%)</td><td rowspan=1 colspan=1>95.83</td><td rowspan=1 colspan=1>6,856 (-12.8%)</td><td rowspan=1 colspan=1>74.32</td><td rowspan=1 colspan=1>8,878 (-10.0%)</td><td rowspan=1 colspan=1>67.97</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)94.27</td><td rowspan=1 colspan=1>4,247 (-15.8%)</td><td rowspan=1 colspan=1>66.42</td><td rowspan=1 colspan=1>9,039 (-13.8%)</td><td rowspan=1 colspan=1>53.37</td><td rowspan=1 colspan=1>6,763 (-16.3%)</td><td rowspan=1 colspan=1>67.78</td><td rowspan=1 colspan=1>15,725 (-11.8%)</td><td rowspan=1 colspan=1>96.67</td><td rowspan=1 colspan=1>6,572 (-16.4%)</td><td rowspan=1 colspan=1>75.70</td><td rowspan=1 colspan=1>8,469 (-14.1%)</td><td rowspan=1 colspan=1>69.56</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)93.67</td><td rowspan=1 colspan=1>4,040 (-19.9%)</td><td rowspan=1 colspan=1>65.98</td><td rowspan=1 colspan=1>8,742 (-16.7%)</td><td rowspan=1 colspan=1>54.21</td><td rowspan=1 colspan=1>6,570 (-18.7%)</td><td rowspan=1 colspan=1>64.44</td><td rowspan=1 colspan=1>15,896 (-10.8%)</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>6,065 (-22.9%)</td><td rowspan=1 colspan=1>75.16</td><td rowspan=1 colspan=1>8,263 (-16.2%)</td><td rowspan=1 colspan=1>69.14</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3-</td><td rowspan=1 colspan=1>8B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    94.27</td><td rowspan=1 colspan=1>5,242</td><td rowspan=1 colspan=1>68.15</td><td rowspan=1 colspan=1>11,197</td><td rowspan=1 colspan=1>58.59</td><td rowspan=1 colspan=1>8,806</td><td rowspan=1 colspan=1>68.89</td><td rowspan=1 colspan=1>17,746</td><td rowspan=1 colspan=1>96.67</td><td rowspan=1 colspan=1>8,543</td><td rowspan=1 colspan=1>77.31</td><td rowspan=1 colspan=1>10,307</td><td rowspan=1 colspan=1>70.23</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   59.80</td><td rowspan=1 colspan=1>860 (-83.6%)</td><td rowspan=1 colspan=1>27.80</td><td rowspan=1 colspan=1>1,153 (-89.7%)</td><td rowspan=1 colspan=1>40.91</td><td rowspan=1 colspan=1>309 (-96.5%)</td><td rowspan=1 colspan=1>10.00</td><td rowspan=1 colspan=1>1,965 (-88.9%)</td><td rowspan=1 colspan=1>45.00</td><td rowspan=1 colspan=1>1,143 (-86.6%)</td><td rowspan=1 colspan=1>36.70</td><td rowspan=1 colspan=1>1,086 (-89.5%)</td><td rowspan=1 colspan=1>35.67</td></tr><tr><td rowspan=1 colspan=1>Dynasor    92.67</td><td rowspan=1 colspan=1>5,654 (+7.9%)</td><td rowspan=1 colspan=1>66.67</td><td rowspan=1 colspan=1>11,684 (+4.3%)</td><td rowspan=1 colspan=1>54.04</td><td rowspan=1 colspan=1>3,875 (-56.0%)</td><td rowspan=1 colspan=1>73.33</td><td rowspan=1 colspan=1>20,179 (+13.7%)</td><td rowspan=1 colspan=1>94.17</td><td rowspan=1 colspan=1>9,272 (+8.5%)</td><td rowspan=1 colspan=1>76.17</td><td rowspan=1 colspan=1>10,133 (-1.7%)</td><td rowspan=1 colspan=1>69.17</td></tr><tr><td rowspan=1 colspan=1>TRACE    86.13</td><td rowspan=1 colspan=1>2,881 (-45.0%)</td><td rowspan=1 colspan=1>51.51</td><td rowspan=1 colspan=1>3,883 (-65.3%)</td><td rowspan=1 colspan=1>52.53</td><td rowspan=1 colspan=1>2,223 (-74.8%)</td><td rowspan=1 colspan=1>34.44</td><td rowspan=1 colspan=1>4,480 (-74.8%)</td><td rowspan=1 colspan=1>76.67</td><td rowspan=1 colspan=1>3,992 (-53.3%)</td><td rowspan=1 colspan=1>60.26</td><td rowspan=1 colspan=1>3,492 (-66.1%)</td><td rowspan=1 colspan=1>57.26</td></tr><tr><td rowspan=1 colspan=1>PUMA     92.27</td><td rowspan=1 colspan=1>3,091 (-41.0%)</td><td rowspan=1 colspan=1>61.78</td><td rowspan=1 colspan=1>5,403 (-51.7%)</td><td rowspan=1 colspan=1>55.89</td><td rowspan=1 colspan=1>5,275 (-40.1%)</td><td rowspan=1 colspan=1>62.22</td><td rowspan=1 colspan=1>14,138 (-20.3%)</td><td rowspan=1 colspan=1>90.00</td><td rowspan=1 colspan=1>5,118 (-40.1%)</td><td rowspan=1 colspan=1>72.43</td><td rowspan=1 colspan=1>6,605 (-35.9%)</td><td rowspan=1 colspan=1>67.36</td></tr><tr><td rowspan=1 colspan=1>DEER      94.07</td><td rowspan=1 colspan=1>3,435 (-34.5%)</td><td rowspan=1 colspan=1>66.52</td><td rowspan=1 colspan=1>9,386 (-16.2%)</td><td rowspan=1 colspan=1>57.74</td><td rowspan=1 colspan=1>8,604 (-2.3%)</td><td rowspan=1 colspan=1>71.11</td><td rowspan=1 colspan=1>15,646 (-11.8%)</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>7,165 (-16.1%)</td><td rowspan=1 colspan=1>77.39</td><td rowspan=1 colspan=1>8,847 (-14.2%)</td><td rowspan=1 colspan=1>71.05</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)94.27</td><td rowspan=1 colspan=1>4,325 (-17.5%)</td><td rowspan=1 colspan=1>67.80</td><td rowspan=1 colspan=1>9,566 (-14.6%)</td><td rowspan=1 colspan=1>59.26</td><td rowspan=1 colspan=1>7,231 (-17.9%)</td><td rowspan=1 colspan=1>71.11</td><td rowspan=1 colspan=1>16,568 (-6.6%)</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=1>7,188 (-15.9%)</td><td rowspan=1 colspan=1>77.49</td><td rowspan=1 colspan=1>8,976 (-12.9%)</td><td rowspan=1 colspan=1>71.08</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)94.53</td><td rowspan=1 colspan=1>4,020 (-23.3%)</td><td rowspan=1 colspan=1>67.46</td><td rowspan=1 colspan=1>9,160 (-18.2%)</td><td rowspan=1 colspan=1>57.58</td><td rowspan=1 colspan=1>6,759 (-23.2%)</td><td rowspan=1 colspan=1>66.67</td><td rowspan=1 colspan=1>15,471 (-12.8%)</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>6,920 (-19.0%)</td><td rowspan=1 colspan=1>76.75</td><td rowspan=1 colspan=1>8,466 (-17.9%)</td><td rowspan=1 colspan=1>70.62</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3-30B</td><td rowspan=1 colspan=1>-A3B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    95.93</td><td rowspan=1 colspan=1>5,176</td><td rowspan=1 colspan=1>76.25</td><td rowspan=1 colspan=1>12,520</td><td rowspan=1 colspan=1>75.59</td><td rowspan=1 colspan=1>7,441</td><td rowspan=1 colspan=1>83.33</td><td rowspan=1 colspan=1>17,308</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>8,708</td><td rowspan=1 colspan=1>86.22</td><td rowspan=1 colspan=1>10,231</td><td rowspan=1 colspan=1>79.17</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.  59.80</td><td rowspan=1 colspan=1>666 (-87.1%)</td><td rowspan=1 colspan=1>23.85</td><td rowspan=1 colspan=1>1,105 (-91.2%)</td><td rowspan=1 colspan=1>34.85</td><td rowspan=1 colspan=1>304 (-95.9%)</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>982 (-94.3%)</td><td rowspan=1 colspan=1>29.17</td><td rowspan=1 colspan=1>870 (-90.0%)</td><td rowspan=1 colspan=1>29.53</td><td rowspan=1 colspan=1>785 (-92.3%)</td><td rowspan=1 colspan=1>28.77</td></tr><tr><td rowspan=1 colspan=1>Dynasor    94.33</td><td rowspan=1 colspan=1>4,901 (-5.3%)</td><td rowspan=1 colspan=1>72.59</td><td rowspan=1 colspan=1>12,272 (-2.0%)</td><td rowspan=1 colspan=1>60.61</td><td rowspan=1 colspan=1>3,690 (-50.4%)</td><td rowspan=1 colspan=1>78.89</td><td rowspan=1 colspan=1>16,779 (-3.1%)</td><td rowspan=1 colspan=1>98.33</td><td rowspan=1 colspan=1>8,554 (-1.8%)</td><td rowspan=1 colspan=1>80.95</td><td rowspan=1 colspan=1>9,239 (-9.7%)</td><td rowspan=1 colspan=1>74.41</td></tr><tr><td rowspan=1 colspan=1>TRACE    92.67</td><td rowspan=1 colspan=1>3,912 (-24.4%)</td><td rowspan=1 colspan=1>66.52</td><td rowspan=1 colspan=1>6,985 (-44.2%)</td><td rowspan=1 colspan=1>57.74</td><td rowspan=1 colspan=1>2,237 (-69.9%)</td><td rowspan=1 colspan=1>56.67</td><td rowspan=1 colspan=1>11,476 (-33.7%)</td><td rowspan=1 colspan=1>89.17</td><td rowspan=1 colspan=1>6,713 (-22.9%)</td><td rowspan=1 colspan=1>72.55</td><td rowspan=1 colspan=1>6,265 (-38.8%)</td><td rowspan=1 colspan=1>67.69</td></tr><tr><td rowspan=1 colspan=1>PUMA     95.33</td><td rowspan=1 colspan=1>3,623 (-30.0%)</td><td rowspan=1 colspan=1>73.29</td><td rowspan=1 colspan=1>8,396 (-32.9%)</td><td rowspan=1 colspan=1>73.90</td><td rowspan=1 colspan=1>6,213 (-16.5%)</td><td rowspan=1 colspan=1>78.89</td><td rowspan=1 colspan=1>15,020 (-13.2%)</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>6,395 (-26.6%)</td><td rowspan=1 colspan=1>84.28</td><td rowspan=1 colspan=1>7,929 (-22.5%)</td><td rowspan=1 colspan=1>78.44</td></tr><tr><td rowspan=1 colspan=1>DEER     96.00</td><td rowspan=1 colspan=1>4,057 (-21.6%)</td><td rowspan=1 colspan=1>76.74</td><td rowspan=1 colspan=1>11,285 (-9.9%)</td><td rowspan=1 colspan=1>75.08</td><td rowspan=1 colspan=1>7,695 (+3.4%)</td><td rowspan=1 colspan=1>82.22</td><td rowspan=1 colspan=1>17,291 (-0.1%)</td><td rowspan=1 colspan=1>99.17</td><td rowspan=1 colspan=1>8,187 (-6.0%)</td><td rowspan=1 colspan=1>85.84</td><td rowspan=1 colspan=1>9,703 (-5.2%)</td><td rowspan=1 colspan=1>79.06</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)96.00</td><td rowspan=1 colspan=1>4,588 (-11.4%)</td><td rowspan=1 colspan=1>76.74</td><td rowspan=1 colspan=1>11,190 (-10.6%)</td><td rowspan=1 colspan=1>74.75</td><td rowspan=1 colspan=1>5,955 (-20.0%)</td><td rowspan=1 colspan=1>83.33</td><td rowspan=1 colspan=1>16,422 (-5.1%)</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>7,750 (-11.0%)</td><td rowspan=1 colspan=1>86.16</td><td rowspan=1 colspan=1>9,181 (-10.3%)</td><td rowspan=1 colspan=1>79.65</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)96.00</td><td rowspan=1 colspan=1>4,237 (-18.1%)</td><td rowspan=1 colspan=1>76.35</td><td rowspan=1 colspan=1>10,770 (-14.0%)</td><td rowspan=1 colspan=1>73.74</td><td rowspan=1 colspan=1>5,475 (-26.4%)</td><td rowspan=1 colspan=1>83.33</td><td rowspan=1 colspan=1>15,709 (-9.2%)</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>7,141 (-18.0%)</td><td rowspan=1 colspan=1>85.88</td><td rowspan=1 colspan=1>8,666 (-15.3%)</td><td rowspan=1 colspan=1>79.64</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Qwen3-3</td><td rowspan=1 colspan=1>2B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    95.13</td><td rowspan=1 colspan=1>4,509</td><td rowspan=1 colspan=1>69.53</td><td rowspan=1 colspan=1>9,763</td><td rowspan=1 colspan=1>69.70</td><td rowspan=1 colspan=1>6,733</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>15,724</td><td rowspan=1 colspan=1>99.17</td><td rowspan=1 colspan=1>6,526</td><td rowspan=1 colspan=1>80.71</td><td rowspan=1 colspan=1>8,651</td><td rowspan=1 colspan=1>74.48</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.   70.87</td><td rowspan=1 colspan=1>1,002 (-77.8%)</td><td rowspan=1 colspan=1>33.73</td><td rowspan=1 colspan=1>1,385 (-85.8%)</td><td rowspan=1 colspan=1>47.47</td><td rowspan=1 colspan=1>314 (-95.3%)</td><td rowspan=1 colspan=1>23.33</td><td rowspan=1 colspan=1>3,175 (-79.8%)</td><td rowspan=1 colspan=1>40.00</td><td rowspan=1 colspan=1>1,380 (-78.9%)</td><td rowspan=1 colspan=1>43.08</td><td rowspan=1 colspan=1>1,451 (-83.2%)</td><td rowspan=1 colspan=1>41.73</td></tr><tr><td rowspan=1 colspan=1>Dynasor    93.87</td><td rowspan=1 colspan=1>4,842 (+7.4%)</td><td rowspan=1 colspan=1>69.19</td><td rowspan=1 colspan=1>10,545 (+8.0%)</td><td rowspan=1 colspan=1>57.74</td><td rowspan=1 colspan=1>3,234 (-52.0%)</td><td rowspan=1 colspan=1>67.78</td><td rowspan=1 colspan=1>18,404 (+17.0%)</td><td rowspan=1 colspan=1>99.17</td><td rowspan=1 colspan=1>7,025 (+7.6%)</td><td rowspan=1 colspan=1>77.55</td><td rowspan=1 colspan=1>8,810 (+1.8%)</td><td rowspan=1 colspan=1>71.23</td></tr><tr><td rowspan=1 colspan=1>TRACE     91.20</td><td rowspan=1 colspan=1>3,155 (-30.0%)</td><td rowspan=1 colspan=1>60.35</td><td rowspan=1 colspan=1>4,935 (-49.5%)</td><td rowspan=1 colspan=1>62.46</td><td rowspan=1 colspan=1>2,948 (-56.2%)</td><td rowspan=1 colspan=1>44.44</td><td rowspan=1 colspan=1>5,975 (-62.0%)</td><td rowspan=1 colspan=1>85.83</td><td rowspan=1 colspan=1>4,321 (-33.8%)</td><td rowspan=1 colspan=1>68.86</td><td rowspan=1 colspan=1>4,267 (-50.7%)</td><td rowspan=1 colspan=1>65.31</td></tr><tr><td rowspan=1 colspan=1>PUMA     93.33</td><td rowspan=1 colspan=1>3,046 (-32.4%)</td><td rowspan=1 colspan=1>65.93</td><td rowspan=1 colspan=1>5,440 (-44.3%)</td><td rowspan=1 colspan=1>67.17</td><td rowspan=1 colspan=1>4,252 (-36.8%)</td><td rowspan=1 colspan=1>63.33</td><td rowspan=1 colspan=1>11,060 (-29.7%)</td><td rowspan=1 colspan=1>90.83</td><td rowspan=1 colspan=1>4,475 (-31.4%)</td><td rowspan=1 colspan=1>76.12</td><td rowspan=1 colspan=1>5,655 (-34.6%)</td><td rowspan=1 colspan=1>71.64</td></tr><tr><td rowspan=1 colspan=1>DEER      94.13</td><td rowspan=1 colspan=1>3,385 (-24.9%)</td><td rowspan=1 colspan=1>68.89</td><td rowspan=1 colspan=1>8,595 (-12.0%)</td><td rowspan=1 colspan=1>67.51</td><td rowspan=1 colspan=1>6,628 (-1.6%)</td><td rowspan=1 colspan=1>75.56</td><td rowspan=1 colspan=1>15,034 (-4.4%)</td><td rowspan=1 colspan=1>96.67</td><td rowspan=1 colspan=1>5,433 (-16.7%)</td><td rowspan=1 colspan=1>80.55</td><td rowspan=1 colspan=1>7,815 (-9.7%)</td><td rowspan=1 colspan=1>74.78</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)95.60</td><td rowspan=1 colspan=1>3,739 (-17.1%)</td><td rowspan=1 colspan=1>69.28</td><td rowspan=1 colspan=1>8,647 (-11.4%)</td><td rowspan=1 colspan=1>68.86</td><td rowspan=1 colspan=1>5,646 (-16.1%)</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>14,807 (-5.8%)</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>5,512 (-15.5%)</td><td rowspan=1 colspan=1>80.25</td><td rowspan=1 colspan=1>7,670 (-11.3%)</td><td rowspan=1 colspan=1>74.56</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard)95.00</td><td rowspan=1 colspan=1>3,633 (-19.4%)</td><td rowspan=1 colspan=1>69.53</td><td rowspan=1 colspan=1>8,391 (-14.1%)</td><td rowspan=1 colspan=1>70.71</td><td rowspan=1 colspan=1>5,445 (-19.1%)</td><td rowspan=1 colspan=1>72.22</td><td rowspan=1 colspan=1>14,385 (-8.5%)</td><td rowspan=1 colspan=1>99.17</td><td rowspan=1 colspan=1>5,351 (-18.0%)</td><td rowspan=1 colspan=1>81.33</td><td rowspan=1 colspan=1>7,441 (-14.0%)</td><td rowspan=1 colspan=1>75.77</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>QwQ-32</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full CoT    95.07</td><td rowspan=1 colspan=1>4,330</td><td rowspan=1 colspan=1>71.16</td><td rowspan=1 colspan=1>9,612</td><td rowspan=1 colspan=1>67.85</td><td rowspan=1 colspan=1>7,796</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>16,263</td><td rowspan=1 colspan=1>99.17</td><td rowspan=1 colspan=1>7,550</td><td rowspan=1 colspan=1>80.65</td><td rowspan=1 colspan=1>9,110</td><td rowspan=1 colspan=1>74.17</td></tr><tr><td rowspan=1 colspan=1>Ans. Conv.  72.53</td><td rowspan=1 colspan=1>1,129 (-73.9%)</td><td rowspan=1 colspan=1>34.77</td><td rowspan=1 colspan=1>1,709 (-82.2%)</td><td rowspan=1 colspan=1>36.87</td><td rowspan=1 colspan=1>344 (-95.6%)</td><td rowspan=1 colspan=1>14.44</td><td rowspan=1 colspan=1>4,488 (-72.4%)</td><td rowspan=1 colspan=1>56.67</td><td rowspan=1 colspan=1>2,039 (-73.0%)</td><td rowspan=1 colspan=1>43.06</td><td rowspan=1 colspan=1>1,942 (-78.7%)</td><td rowspan=1 colspan=1>41.29</td></tr><tr><td rowspan=1 colspan=1>Dynasor    94.40</td><td rowspan=1 colspan=1>5,420 (+25.2%)</td><td rowspan=1 colspan=1>69.73</td><td rowspan=1 colspan=1>11,326 (+17.8%)</td><td rowspan=1 colspan=1>58.42</td><td rowspan=1 colspan=1>4,324 (-44.5%)</td><td rowspan=1 colspan=1>72.22</td><td rowspan=1 colspan=1>20,520 (+26.2%)</td><td rowspan=1 colspan=1>98.33</td><td rowspan=1 colspan=1>9,628 (+27.5%)</td><td rowspan=1 colspan=1>78.62</td><td rowspan=1 colspan=1>10,244 (+12.4%)</td><td rowspan=1 colspan=1>71.57</td></tr><tr><td rowspan=1 colspan=1>TRACE     90.87</td><td rowspan=1 colspan=1>2,878 (-33.5%)</td><td rowspan=1 colspan=1>60.59</td><td rowspan=1 colspan=1>4,556 (-52.6%)</td><td rowspan=1 colspan=1>56.40</td><td rowspan=1 colspan=1>2,331 (-70.1%)</td><td rowspan=1 colspan=1>48.89</td><td rowspan=1 colspan=1>9,317 (-42.7%)</td><td rowspan=1 colspan=1>83.33</td><td rowspan=1 colspan=1>4,589 (-39.2%)</td><td rowspan=1 colspan=1>68.02</td><td rowspan=1 colspan=1>4,734 (-48.0%)</td><td rowspan=1 colspan=1>64.14</td></tr><tr><td rowspan=1 colspan=1>PUMA     90.47</td><td rowspan=1 colspan=1>3,255 (-24.8%)</td><td rowspan=1 colspan=1>61.53</td><td rowspan=1 colspan=1>6,586 (-31.5%)</td><td rowspan=1 colspan=1>62.80</td><td rowspan=1 colspan=1>5,714 (-26.7%)</td><td rowspan=1 colspan=1>62.22</td><td rowspan=1 colspan=1>12,554 (-22.8%)</td><td rowspan=1 colspan=1>88.33</td><td rowspan=1 colspan=1>5,689 (-24.6%)</td><td rowspan=1 colspan=1>73.07</td><td rowspan=1 colspan=1>6,760 (-25.8%)</td><td rowspan=1 colspan=1>67.91</td></tr><tr><td rowspan=1 colspan=1>DEER      95.47</td><td rowspan=1 colspan=1>3,592 (-17.0%)</td><td rowspan=1 colspan=1>70.72</td><td rowspan=1 colspan=1>8,604 (-10.5%)</td><td rowspan=1 colspan=1>65.15</td><td rowspan=1 colspan=1>7,975 (+2.3%)</td><td rowspan=1 colspan=1>64.44</td><td rowspan=1 colspan=1>15,062 (-7.4%)</td><td rowspan=1 colspan=1>96.67</td><td rowspan=1 colspan=1>6,931 (-8.2%)</td><td rowspan=1 colspan=1>78.49</td><td rowspan=1 colspan=1>8,433 (-7.4%)</td><td rowspan=1 colspan=1>72.37</td></tr><tr><td rowspan=1 colspan=1>CALM (Soft)95.53</td><td rowspan=1 colspan=1>3,517 (-18.8%)</td><td rowspan=1 colspan=1>70.62</td><td rowspan=1 colspan=1>7,923 (-17.6%)</td><td rowspan=1 colspan=1>66.67</td><td rowspan=1 colspan=1>6,439 (-17.4%)</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>14,125 (-13.1%)</td><td rowspan=1 colspan=1>95.83</td><td rowspan=1 colspan=1>5,966 (-21.0%)</td><td rowspan=1 colspan=1>79.73</td><td rowspan=1 colspan=1>7,594 (-16.6%)</td><td rowspan=1 colspan=1>74.08</td></tr><tr><td rowspan=1 colspan=1>CALM (Hard) 95.20</td><td rowspan=1 colspan=1>3,278 (-24.3%)</td><td rowspan=1 colspan=1>68.94</td><td rowspan=1 colspan=1>7,567 (-21.3%)</td><td rowspan=1 colspan=1>67.34</td><td rowspan=1 colspan=1>5,760 (-26.1%)</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>14,163 (-12.9%)</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>6,174 (-18.2%)</td><td rowspan=1 colspan=1>79.80</td><td rowspan=1 colspan=1>7,388 (-18.9%)</td><td rowspan=1 colspan=1>74.27</td></tr></table>

Table 6: Averaged results over three seeds across additional reasoning models. Acc: accuracy, Tok: token count, TE: token efficiency. Values in parentheses are relative changes in tokens from Full CoT. The best and second-best accuracy and TE values among the efficient-reasoning methods are shown in bold and underlined, respectively; Full CoT is excluded from this ranking.

## The final answer is \boxed

If this budget ends before that marker, generation

continues with the following suffix:

## \n</think>\n\nThe final answer is \boxed

Delivered reasoning, the final answer, and trial answers use the models’ recommended reasoning settings, with temperature 0.6 and top-p 0.95.The instruction on mathematics and contest benchmarks is:

Solve the following math problem. Please reason step by step, and put your final answer within \boxed{}.

The instruction on GPQA-Diamond is:

Answer the following multiple choice question. Please reason step by step, and put your choice letter (A, B, C, or D) without any other text within \boxed{} in the end.

These instructions and the answer-inducing suffixes follow the default prompts of PUMA (Min et al., 2026).

## G Stop Coverage on the Main-Table Cells

Table 7 reports, for the four main-table models and five benchmarks, how many questions CALM intervenes on and how many the early-exit methods stop, as the mean over three seeds. The parenthetical value is the number of questions on that benchmark. CALM counts questions with four consecutive identical trial answers. The other columns count questions on which that method’s stopping rule fires. In most cells,Answer Convergence stops on more questions than CALM intervenes on. DEER’s confidence exits are fewer, except on DeepSeek-R1-Distill-Qwen-14B MATH-500.

## H Additional Right-to-Wrong Rates

Figure 6 extends Figure 4 to four models on MATH-500. The rate is the share of questions that full CoT answers correctly and the method answers incorrectly. CALM Soft is the lowest among Dynasor, DEER, PUMA, and CALM on every panel. These additional results demonstrate the stability of our method, whereas other truncation-based methods are indeed prone to depriving the model of opportunities for subsequent correction.

## I Settings for LLM-as-a-Judge

We evaluate the quality of retained reasoning chains with an LLM judge. The goal is to measure whether the retained chain remains a readable and sufficient derivation, beyond final-answer accuracy.

Judge Model and Protocol. The judge is Qwen3.5-397B-A17B, called at temperature 0 with at most 1,024 output tokens. Each call receives the problem, the retained reasoning, and the answer submitted by that method. The judge writes a one-sentence rationale before the scores.

Scope. The evaluated chains are generations of DeepSeek-R1-Distill-Qwen-7B on MATH-500. We randomly draw 60 questions, and every method is scored on these 60 questions. The model’s input is the problem, the retained reasoning, and the submitted answer. When a method stops early, the retained reasoning ends at that stop; otherwise it is the full reasoning.

Rubric. Each chain is scored from 10 to 100 in steps of 10 on completeness, coherence, conciseness, and justification, following the four dimensions of PUMA (Min et al., 2026). Completeness asks whether the essential steps from the problem to the stated answer are present. Coherence asks whether the progression is connected and whether revisions are resolved. Conciseness asks whether repetition or unproductive detours are present, and missing steps are scored under completeness and justification instead. Justification asks whether the written derivation supports the stated answer. The prompt states the anchors at 100, 70, 40, and 10.

Role. You evaluate the quality of a written mathematical derivation. You receive the problem, the retained reasoning, and the stated answer. The stated answer is a claim to assess, not a reference answer.

Boundary. Do not solve the problem independently, and do not grade the chain against your own answer or an external reference. Do inspect violations of the problem’s conditions, invalid arithmetic or logical transitions, missing essential steps, and unresolved contradictions. Ground every deduction in the supplied text. Do not invent a missing argument, and do not treat the stated answer itself as evidence for the missing steps. Ordinary algebraic compression is acceptable. A short chain can be sufficient, and a long chain can remain incomplete. Score the four dimensions separately: an unsupported answer can still be concise.

Scores. Use only 10, 20, 30, 40, 50, 60, 70, 80, 90, and 100. Completeness asks whether the essential reasoning from the problem to the stated answer is present: 100 if the essential links and conditions are covered, 70 if the main route is developed but some support is thin, 40 if a major part is missing, and 10 if the answer is asserted without a derivation. Coherence asks whether one connected progression can be followed: 100 if it is clear and revisions are resolved, 70 if it is mostly followable, 40 if disconnection or unresolved conflict disrupts the route, and 10 if no followable line can be identified. Conciseness asks how much unnecessary material appears: 100 if there is no material repetition or unproductive detour, 70 if some avoidable repetition is present, 40 if redundancy is substantial, and 10 if repetition dominates. Do not lower conciseness merely because essential reasoning is missing. Justification asks whether the text explains why this stated answer follows: 100 if it is fully supported, 70 if an important explanation remains implicit, 40 if the decisive connection is missing or defective, and 10 if the answer is unsupported or contradicted.

<table><tr><td>Model</td><td>Benchmark</td><td>CALM</td><td>Ans. Conv.</td><td>Dynasor</td><td>PUMA</td><td>DEER</td></tr><tr><td rowspan="5">DeepSeek-R1-Distill-Qwen-7B</td><td>MATH-500 (500)</td><td>421.7</td><td>458.3</td><td>352.0</td><td>307.0</td><td>397.3</td></tr><tr><td>Olympiad (675)</td><td>634.0</td><td>649.3</td><td>520.7</td><td>464.0</td><td>420.3</td></tr><tr><td>GPQA (198)</td><td>180.7</td><td>198.0</td><td>182.3</td><td>49.7</td><td>0.0</td></tr><tr><td>AIME25 (30)</td><td>28.0</td><td>30.0</td><td>27.0</td><td>11.7</td><td>9.7</td></tr><tr><td>AMC23 (40)</td><td>36.7</td><td>39.3</td><td>33.7</td><td>22.3</td><td>27.3</td></tr><tr><td rowspan="5">DeepSeek-R1-Distill-Qwen-14B</td><td>MATH-500 (500)</td><td>414.7</td><td>483.3</td><td>223.0</td><td>288.3</td><td>420.7</td></tr><tr><td>Olympiad (675)</td><td>641.7</td><td>665.7</td><td>381.7</td><td>441.3</td><td>432.0</td></tr><tr><td>GPQA (198)</td><td>180.3</td><td>197.7</td><td>170.0</td><td>86.0</td><td>0.0</td></tr><tr><td>AIME25 (30)</td><td>27.3</td><td>29.0</td><td>22.3</td><td>11.0</td><td>8.3</td></tr><tr><td>AMC23 (40)</td><td>34.7</td><td>38.0</td><td>21.7</td><td>20.0</td><td>27.0</td></tr><tr><td rowspan="5">DeepSeek-R1-Distill-Qwen-32B</td><td>MATH-500 (500)</td><td>266.0</td><td>306.7</td><td>187.0</td><td>158.0</td><td>242.0</td></tr><tr><td>Olympiad (675)</td><td>544.7</td><td>570.0</td><td>401.0</td><td>348.0</td><td>345.7</td></tr><tr><td>GPQA (198)</td><td>176.7</td><td>198.0</td><td>171.7</td><td>64.0</td><td>0.0</td></tr><tr><td>AIME25 (30)</td><td>26.3</td><td>29.7</td><td>23.0</td><td>13.0</td><td>9</td></tr><tr><td>AMC23 (40)</td><td>30.7</td><td>36.7</td><td>20.3</td><td>15.7</td><td>28</td></tr><tr><td rowspan="5">DeepSeek-R1-Distill-Llama-8B</td><td>MATH-500 (500)</td><td>321.3</td><td>220.3</td><td>246.7</td><td>218.7</td><td>224.0</td></tr><tr><td>Olympiad (675)</td><td>596.3</td><td>548.3</td><td>460.0</td><td>409.3</td><td>263.0</td></tr><tr><td>GPQA (198)</td><td>181.3</td><td>198.0</td><td>182.3</td><td>53.7</td><td>0.0</td></tr><tr><td>AIME25 (30)</td><td>27.3</td><td>30.0</td><td>24.7</td><td>8.3</td><td>5.0</td></tr><tr><td>AMC23 (40)</td><td>32.7</td><td>29.0</td><td>26.0</td><td>19.3</td><td>16.3</td></tr></table>

Table 7: Questions stopped, or intervened on by CALM, as the mean over three seeds. The parenthetical value is the number of questions. CALM counts four consecutive identical trial answers. The other columns count questions on which that method’s stopping rule fires.  
![](images/f4373589361f9ba17a5944c69c9d5d9a5652f1ceabfc4521eb356cdc4adc930e.jpg)  
Figure 6: Right-to-wrong rates on MATH-500. Panels pool three seeds and show DeepSeek-R1-7B, DeepSeek-R1- 14B, Qwen3-4B, and Qwen3-30B-A3B from left to right. A right-to-wrong transition occurs when full CoT answers a problem correctly and the method answers it incorrectly. CALM is the lowest of the four methods on each panel.

Output. Return only one JSON object, with a one-sentence rationale first. {"brief\_rationale": "...", "completeness": 10, "coherence": 10, "conciseness": 10,

"justification": 10}   
The numbers illustrate the format. Choose   
the actual scores from the permitted scale.

## J Case Study

The following traces are generations of DeepSeek-R1-Distill-Qwen-7B on MATH-500.

## K AI Use Statement

We utilized GPT-5.4 to assist in polishing the text, refining grammar, and improving the clarity of the manuscript. All content was reviewed and verified by the authors.

Question 148. Positive reals satisfy $x y z = 2 .$ Find the minimum of $x ^ { 4 } + 4 y ^ { 2 } + 4 z ^ { 4 } .$ . Ground truth: 16.

## CALM

Alternatively, maybe I can use substitution. Since xyz = 2, I can express one variable in terms of the others. For example, $\begin{array} { r } { z = \frac { 2 } { x y } . } \end{array}$

Then, substitute this into the expression $x ^ { 4 } + 4 y ^ { 2 } +$ $4 z ^ { 4 } .$

Let me try that:

substitute $\begin{array} { r } { z = \frac { 2 } { x y } , \mathrm { s o } z ^ { 4 } = \Big ( \frac { 2 } { x y } \Big ) ^ { 4 } = \frac { 1 6 } { x ^ { 4 } y ^ { 4 } } . } \end{array}$

Therefore, the expression becomes $\begin{array} { r } { x ^ { 4 } + \dot { 4 } y ^ { 2 } + \frac { 6 4 } { x ^ { 4 } y ^ { 4 } } . } \end{array}$

$$
x = { \sqrt { 2 } } , y = { \sqrt { 2 } } , { \mathrm { a n d } } z = 1 .
$$

$x ^ { 4 } = 4 , \dot { 4 } y ^ { 2 } = 8 , 4 z ^ { 4 } = 4 .$ Adding them up: $4 + 8 +$ $4 = 1 6 .$

So, the minimum value is 16.

</think> The final answer is 16

Tokens: 4,297 16

## Full CoT

Alternatively, maybe I can use substitution. Since $x y z = 2 ,$ I can express one variable in terms of the others. For example, $\begin{array} { r } { z = \frac { 2 } { x y } . } \end{array}$

Then, substitute this into the expression $x ^ { 4 } + 4 y ^ { 2 } +$ $4 z ^ { 4 } .$

Let me try that:

Substituting $\begin{array} { r } { z = \frac { 2 } { x y } , } \end{array}$ we get $\begin{array} { r } { x ^ { 4 } + 4 y ^ { 2 } + \frac { 6 4 } { x ^ { 4 } y ^ { 4 } } . } \end{array}$

Hmm, now we have an expression in terms of x and y. Maybe I can set partial derivatives with respect to x and y to zero to find the minimum. Let me try calculus.

Set both partial derivatives equal to zero:

$$
4 x ^ { 3 } - { \frac { 2 5 6 } { x ^ { 5 } y ^ { 4 } } } = 0 ,
$$

$$
8 y - { \frac { 2 5 6 } { x ^ { 4 } y ^ { 5 } } } = 0 .
$$

From the first equation, $x ^ { 2 } y = 2 .$

Wait, that can’t be right because $1 + 8 + 4$ is 13, so <sub>13</sub>√<sub>2.</sub> <sub>Hmm,</sub> <sub>but</sub> <sub>let</sub> <sub>me</sub> <sub>verify</sub> <sub>the</sub> <sub>calculations</sub> <sub>again.</sub>

Wait, maybe I made a mistake in the partial derivatives. Let me go back and check.

Wait, so earlier, my mistake was in solving the equations. Upon correctly solving them, we find z = 1, $x = { \sqrt { 2 } } , y = { \sqrt { 2 } } ,$ which gives the minimal value of 16.

Tokens: 20,974 16 ✓

Figure 7: Gray is the shared prefix through the first same-answer window, and CALM continues in green after that window. The blue box opens with </think>. Both close at 16.

<sub>Question</sub> <sub>165.</sub> <sub>Simplify</sub> <sub>tan</sub> <sub>100</sub>◦ <sub>+</sub> <sub>4</sub> <sub>sin</sub> <sub>100</sub>◦<sub>.</sub> <sub>Ground</sub> <sub>truth:</sub> <sub>−</sub>√<sub>3.</sub>

## CALM

So, sin 100<sup>◦</sup> (1 + 4 cos 100<sup>◦</sup>)/ cos 100<sup>◦</sup>.

Hmm, perhaps I can write this as tan 100<sup>◦</sup> · (1 + 4 cos 100<sup>◦</sup>).

tan 100<sup>◦</sup> · (1 + 4 cos 100<sup>◦</sup>) ≈ tan 100<sup>◦</sup> · (1 + 4 · (−0.173648178)) ≈ tan 100<sup>◦</sup> · (1 − 0.694592712) ≈ tan 100<sup>◦</sup> · 0.305407288. But tan 100<sup>◦</sup> ≈ −5.67128182, so

−5.67128182 · 0.305407288 ≈ −1.73205081,

<sub>which</sub> <sub>is</sub> <sub>approximately</sub> <sub>−</sub>√<sub>3.</sub> <sub>So,</sub> <sub>that’s</sub> <sub>consistent.</sub>

Let me compute (tan 100<sup>◦</sup> + 4 sin 100<sup>◦</sup>)<sup>2</sup>. tan<sup>2</sup> 100<sup>◦</sup> ≈ 32.1653, 8 tan 100<sup>◦</sup> sin 100<sup>◦</sup> ≈ −44.672, and 16 sin<sup>2</sup> 100<sup>◦</sup> ≈ 15.517. Adding them together: 32.1653−44.672+15.517 ≈ 3.000.

So the square is about 3, and the sum itself is negative. Since 100<sup>◦</sup> is in the second quadrant, tangent is nega-<sub>tive</sub> <sub>and</sub> <sub>sine</sub> <sub>is</sub> <sub>positive.</sub> <sub>Therefore,</sub> <sub>it</sub> <sub>must</sub> <sub>be</sub> <sub>−</sub>√<sub>3.</sub>

<sub></think></sub> <sub>The</sub> <sub>final</sub> <sub>answer</sub> <sub>is</sub> <sub>−</sub>√<sub>3</sub> <sub>.</sub>

Tokens: 4,702

## Full CoT

So, sin 100<sup>◦</sup> (1 + 4 cos 100<sup>◦</sup>)/ cos 100<sup>◦</sup>.

Hmm, perhaps I can write this as tan 100<sup>◦</sup> · (1 + 4 cos 100<sup>◦</sup>).

But we know that tan 100<sup>◦</sup> ≈ −5.671, and 1 + 4 cos 100<sup>◦</sup> ≈ 0.3056. Then tan 100<sup>◦</sup> · 0.3056 ≈ <sub>−1.732, which is −</sub>√<sub>3. So, the algebraic manipula-</sub> <sub>tion must lead to −</sub>√<sub>3.</sub>

Alternatively, maybe I can use some trigonometric identities to express this combination as a single sine or cosine function.

Wait, another idea: Maybe use the identity for a linear combination of sine and tangent.

But I’m not sure about that. Alternatively, perhaps use the sine of sum formula.

Wait, let me consider the expression tan θ + 4 sin θ. Let me write tan θ as sin θ/ cos θ, so it becomes sin θ (1 + 4 cos θ)/ cos θ.

Alternatively, perhaps I can consider that the expression is equal to − 3, and then verify it using the approximate value, as I did before.

Tokens: 13,134

Figure 8: Gray is the shared prefix through the first same-answer window, and CALM continues in green after that <sub>window. The blue box opens with </think>. Both close at −</sub>√<sub>3.</sub>

Question 146. $z + 1 / z = ( 1 + { \sqrt { 5 } } ) / 2 .$ Find $z ^ { 8 5 } + 1 / z ^ { 8 5 } .$ . Ground truth: −2.

## CALM

Wait, earlier, I computed $a _ { 3 } = - \phi + 1$ directly, which $\mathrm { i s } \ \frac { 1 - \sqrt { 5 } } { 2 }$ . Let me compute $\phi \cdot a _ { 2 } - a _ { 1 }$

$$
\begin{array} { r } { a _ { 2 } = \frac { \sqrt { 5 } - 1 } { 2 } , \mathrm { { s o } } \phi \cdot a _ { 2 } = \frac { 1 + \sqrt { 5 } } { 2 } \cdot \frac { \sqrt { 5 } - 1 } { 2 } . } \end{array}
$$

$$
\mathrm { T h e ~ n u m e r a t o r s ~ m u l t i p l y ~ t o ~ \tilde { 4 } , s o ~ } \tilde { \phi } \cdot a _ { 2 } = \mathrm { ~ \frac { 4 } { 4 } = 1 ~ } .
$$

Then, $\textstyle \phi \cdot a _ { 2 } - a _ { 1 } = 1 - \phi = { \frac { 1 - { \sqrt { 5 } } } { 2 } } ,$ , which is indeed a<sub>3</sub>. So the correct recurrence is $\bar { a _ { n + 1 } } = a _ { 1 } a _ { n } - a _ { n - 1 } .$

So, now, $a _ { 1 0 } = a _ { 0 }$ and $a _ { 9 } = a _ { 1 }$ . That suggests that the sequence is periodic with period 10.

$$
a _ { 0 } = 2 , a _ { 1 } = \phi , a _ { 5 } = - 2 , a _ { 9 } = \phi , a _ { 1 0 } = 2 .
$$

Yes, from a<sub>0</sub> to $^ { a _ { 1 0 } , }$ the sequence repeats. Therefore, the period is 10.

Compute $8 5 \div 1 0 = 8 \quad$ with a remainder of 5. So $8 5 = 1 0 \times 8 + 5 .$ Therefore, $a _ { 8 5 } = a _ { 5 } .$ From above, $a _ { 5 } = - 2 .$

</think> The final answer is −2 .

Tokens: 5,904

## DEER

$$
6 = \dot { 1 } .
$$

Compute 6 · 14 = 84, so 85 − 84 = 1, and 85 mod

$$
\mathrm { T h e r e f o r e } , a _ { 8 5 } = a _ { 1 } = \phi = ( 1 + \sqrt { 5 } ) / 2 .
$$

The final answer is

![](images/67e69b4e26a810f06432a9d780ad0329a2d0ee903632c48127ce4b61c0e53f2d.jpg)

→ confidence < λ

Wait, but earlier $a _ { 6 } = - 2 ,$ while $a _ { 0 } = 2 .$ They are not the same. Is the period 12? $\mathbf { W a i t } , a _ { 6 } = - 2 , a _ { 7 } = \phi , a _ { 1 2 } = 2 . \underline { { \mathrm { \ T h u s } } }$ 85 mod $1 2 = 1 , { \mathrm { a n d } } a _ { 8 5 } = a _ { 1 } = \phi = ( 1 + { \sqrt { 5 } } ) / 2 .$

The final answer is

$$
\boxed { \frac { 1 + \sqrt { 5 } } { 2 } }
$$

$$
\cdot \mathrm { c o n f i d e n c e } < \lambda
$$

Wait, $a _ { 6 } = - 2$ differs from $a _ { 0 } = 2 ,$ so the period is 12. Therefore $a _ { n } = a _ { n { \mathrm { ~ m o d ~ } } 1 2 } ,$ and the answer is $( 1 + { \sqrt { 5 } } ) / 2 .$

The final answer is

$$
\left| { \frac { 1 + { \sqrt { 5 } } } { 2 } } \right|
$$

→ confidence > λ

</think> 12·7 = 84, so 85 mod $1 2 = 1$ , and a<sub>85</sub> = a<sub>1</sub>.

The final answer is

![](images/2a69fe5a65deb9fe85a07b9d6677acaa10b31e4f6304f9114a60aa52976cda7e.jpg)

Tokens: 11,723

$$
\left\lfloor { \frac { 1 + { \sqrt { 5 } } } { 2 } } \right\rfloor x
$$

Figure 9: CALM is gray before the window and green after it, and DEER stays gray. The blue box opens with </think>. CALM closes at −2, and DEER closes at $( 1 + { \sqrt { 5 } } ) / 2$

Question 19. BC is parallel to the segment through A, and AB = BC. How many degrees is x? Ground truth: 28.

## CALM

The angle at A is given as 124 degrees, so if the angle at C were also 124 degrees, then 124 + 124 = 248, which is more than 180. So, that can’t be.

Wait, maybe it’s triangle ABD, where D is a point such that AD is parallel to BC.

Alternatively, maybe the figure is a trapezoid with AB and CD as legs, and BC parallel to AD, with AB = CD.

Maybe I’m overcomplicating. Given: BC is parallel to a segment through A, AB = BC, and the angle at A is 124<sup>◦</sup>. The label for x is near AB.

Since AB = BC, triangle ABC is isosceles. The label for x is at (4.5, 3), on the top base, so perhaps x is the angle at B between AB and BC.

In an isosceles trapezoid the base angles are equal. If the angle at A is 124<sup>◦</sup>, the angles at B and C are each $( 1 8 0 - 1 2 4 ) / 2 = 2 8 ^ { \circ } .$

In the figure, x is labeled near AB. So x is the angle at B between AB and BC, which is 28<sup>◦</sup>.

</think> The final answer is 28

Tokens: 5,293 28

## DEER

Given angle BAD = 124<sup>◦</sup>, so angle ABC = 56<sup>◦</sup>. In triangle ABC, AB = BC and the angle at B is 56<sup>◦</sup>. The base angles are equal, so

$$
{ \mathrm { a n g l e ~ } } B A C + { \mathrm { a n g l e ~ } } B A C + 5 6 = 1 8 0 .
$$

Thus angle BAC = 62<sup>◦</sup>. Therefore, angle BAC is 62 degrees. So, x is 62 degrees.

The final answer is 62 −→ confidence < λ

Wait, but let me verify this because I might have made a mistake.

Wait, angle BAC is part of angle BAD. Alternatively, angle BCA is also 62 degrees. So angle x is 62 degrees.

The final answer is 62 −→ confidence < λ

Wait, but let me think again. Angle BAC is 62 and angle CAD is 62, so x is 62 degrees.

The final answer is 62 −→ confidence > λ

</think> Sum of angles: 56 + 2x = 180, so x = 62.

The final answer is 62 .

Tokens: 7,529 62 ✗

Figure 10: CALM is gray before the window and green after it, and DEER stays gray. The blue box opens with </think>. CALM closes at 28, and DEER closes at 62.