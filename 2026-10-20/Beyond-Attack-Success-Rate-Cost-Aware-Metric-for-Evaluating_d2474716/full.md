# Beyond Attack Success Rate: Cost-Aware Metric for Evaluating Iterative LLM Jailbreaks

Rohan Menon<sup>1</sup>, Chih-Hong Cheng<sup>1,2</sup>\*

<sup>1</sup>Carl von Ossietzky University of Oldenburg, Oldenburg, Germany <sup>2</sup>Chalmers University of Technology, Gothenburg, Sweden

## Abstract

Attack success rate (ASR) evaluates jailbreaks against large language models without accounting for repeated calls to the victim model. Yet each additional call gives the deployed system another opportunity to detect the attack and intervene. We introduce ASR<sub>STEP</sub>, which penalizes additional victim-model calls while retaining binary success judgments. Our evaluation considers GCG (Zou et al., 2023), PAIR (Chao et al., 2025) with a step budget, and a single-step PAP (Zeng et al., 2024) baseline on JailBreakBench across three model families. GCG’s scores decrease by 7.0–8.7 percentage points after discounting, while PAP’s remain unchanged. While for PAIR (Chao et al., 2025), we see a wider discounting range between 3.4-12 percentage points, showing that the "step" quantity differs per iterative strategy. Reporting both metrics makes the discount due to repeated victim-model interaction explicit. Code is available here.

## 1 Introduction

Jailbreak attacks seek to make large language models (LLMs) answer requests that their safeguards should reject. Evaluating these attacks is important both for comparing attack methods and for assessing the robustness of deployed models. The conventional attack success rate (ASR) measures the fraction of harmful requests for which an attack elicits a response judged unsafe. However, it gives equal credit to an immediate success and one obtained after hundreds of victim-model calls, leaving the effort required to obtain that success implicit.

This distinction is particularly important for iterative attacks, which repeatedly query the victim model and use feedback to refine their prompts. Additional attempts create more opportunities to find a successful prompt while extending interaction with the protected system. Each additional call gives deployment-side monitoring, such as intrusion detection, another opportunity to identify suspicious activity and trigger a protective response. Victim-call count is therefore operationally relevant beyond its role as a proxy for computational effort.

To address this limitation, in this paper we introduce ASR<sub>STEP</sub>, a cost-aware extension that retains binary success judgments and penalizes additional attack steps that call the victim LLM. Single-step attacks incur no additional-step penalty and recover conventional ASR. The two metrics should be reported together: ASR gives the observed success frequency, while ASR<sub>STEP</sub> gives the corresponding score after discounting. Their gap makes the effect of additional victim calls explicit under the chosen penalty, preserving ASR as a reference for interpretation.

Our evaluation considers GCG (Zou et al., 2023), PAIR (Chao et al., 2025), and a singlestep PAP Misrepresentation baseline on JailBreak-Bench (Chao et al., 2024) across the Vicuna, Mistral, and Zephyr model families. Success is assessed using an LLM-as-a-Judge. In the reported results, GCG’s (Zou et al., 2023) cost-aware scores are 7.0–8.7 percentage points below its conventional ASR, while PAP’s scores remain unchanged. The margin of difference is higher for PAIR (Chao et al., 2025) with it performing highest on both ASR and ASR<sub>STEP</sub> metrics specifically on the Zephyr model (Tunstall et al., 2023)

## 2 Related Work

Attacks. Our selection spans three attack categories. Suffix optimization refines an adversarial suffix, as in GCG (Zou et al., 2023) and AutoDAN (Liu et al., 2024). Prompt optimization revises the input using model feedback, as in PAIR (Chao et al., 2025) and rewriting-based attacks (Takemoto, 2024). Prompt transformation changes the framing or language of a request, as in PAP’s persuasive prompts (Zeng et al., 2024) and low-resource language attacks (Yong et al., 2023). We select GCG (Zou et al., 2023) and PAIR (Chao et al., 2025) as iterative representatives of the first two categories and PAP Misrepresentation as a single-step baseline from the third.

Evaluations. The closest work to ours is Risk Under Pressure (Ehghaghi et al., 2026), which measures computational attack cost in FLOPs and reports risk–compute curves. Our perspective centers on calls to the victim LLM: each additional call gives the deployed system an opportunity to detect suspicious activity and intervene, for example through intrusion detection. This interaction with the protected system is distinct from computation performed outside it. We therefore report ASR alongside a score discounted by additional victim calls; the effectiveness of particular detection mechanisms remains to be evaluated.

Meaningful ASR comparisons also require consistent measurement goals and valid evaluators (Chouldechova et al., 2025). HarmBench and JailbreakBench standardize evaluation (Mazeika et al., 2024; Chao et al., 2024), while JailbreakEval brings together different evaluation procedures (Ran et al., 2024). Extensions beyond binary success include AttackEval’s graded compliance scores (Shu et al., 2025), StrongREJECT’s assessment of useful harmful information (Souly et al., 2024), and Expected Harm’s weighting of severity by execution likelihood, modeled from the cost of carrying out a harmful act (Chen et al., 2026). These approaches assess response quality or consequences; our metric accounts for the victim calls spent obtaining an outcome.

Complementary perspectives include Security Cube’s token and time overhead measures (Xu et al., 2026) and configuration sensitivity. Maple et al. (2026) propose the Variant Sensitivity Measure and Union Coverage to characterize performance across attack configurations, whereas $\mathrm { A S R } _ { \mathrm { S T E P } }$ accounts for calls within an execution.

## 3 Beyond Attack Success Rate

## 3.1 Composition of an Iterative Jailbreak

Iterative jailbreaks modify a base prompt over multiple steps to elicit harmful responses from a victim

![](images/68751cd831252b64a08e266f7bac6e10278f68a7db9bcded018c11899e894f08.jpg)  
Figure 1: General flow of iterative prompt optimization.

LLM. Figure 1 illustrates this process. The number of steps depends on the attack algorithm and its optimization budget, but conventional ASR does not account for this effort. For example, a GCG (Zou et al., 2023) attack optimized for 500 steps and a single PAP (Zeng et al., 2024) transformation use substantially different budgets. Comparing their ASR values alone leaves this difference implicit. With ASR<sub>STEP</sub>, we make the discount associated with attack steps explicit alongside the observed success rate.

## 3.2 Preliminaries

Let

$$
{ \mathcal { D } } _ { \mathrm { f o r b i d } } = \{ q _ { 1 } , . . . , q _ { m } \}
$$

be a dataset of forbidden requests that a safetyaligned large language model should refuse to answer, where

$$
\begin{array} { r } { m = \left| \mathcal { D } _ { \mathrm { f o r b i d } } \right| . } \end{array}
$$

Let

$$
f : \mathcal { Q }  \mathcal { V }
$$

denote the target LLM, and let

$$
{ \mathrm { j a i l b r e a k : } } \ y \to \{ 0 , 1 \}
$$

be a binary evaluator. In particular, jailbreak $( y ) =$ 1 indicates that output $y$ violates the considered safety policy.

Given a forbidden request $q _ { i } .$ , an attack method $\mathcal { A }$ may adaptively generate a sequence of queries

$$
q _ { i } ^ { ( 0 ) } , q _ { i } ^ { ( 1 ) } , \ldots , q _ { i } ^ { ( K _ { i } ) } ,
$$

where $q _ { i } ^ { ( K _ { i } ) }$ is the final query submitted to the victim LLM. For each base request, the attack terminates when it produces a successful jailbreak or reaches the maximum query budget B. This stopping rule applies to both black-box and white-box methods. In GCG (Zou et al., 2023), for example, the suffix with the lowest loss is considered the most effective for achieving jailbreak success. But the last gradient iteration need not be the least, considering standard gradient dynamics. So the best suffix string could be at iteration $i < B$ . We store the the lowest loss index for each base prompt and use that in the $A S R _ { S T E P }$ computation.

With zero-based indexing,

$$
0 \leq K _ { i } \leq B - 1 .
$$

The quantity $K _ { i }$ counts queries submitted to the target LLM after the initial query. Intermediate computations that do not invoke the target model are not counted as attack steps. We denote the final query produced by the attack as

$$
\mathcal { A } ( q _ { i } ) = q _ { i } ^ { ( K _ { i } ) }
$$

and the corresponding model output as

$$
y _ { i } = f ( \mathcal { A } ( q _ { i } ) ) .
$$

## 3.3 Standard Attack Success Rate

In our evaluation, each attack is executed once for each forbidden request $q _ { i }$ . Define the success indicator

$$
S _ { i } = { \bf 1 } \left\{ \mathrm { j a i l b r e a k } \left( f ( A ( q _ { i } ) ) \right) = 1 \right\} .\tag{1}
$$

The standard attack success rate is

$$
\begin{array} { r l r } { \left. { \mathrm { A S R } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } S _ { i } } \right|} \\ & { } & { = \frac { \left| \left\{ q _ { i } \in \mathcal { D } _ { \mathrm { f o r b i d } } \vert \right. \operatorname { j a i l b r e a k } \left( f ( \mathcal { A } ( q _ { i } ) ) \right) = 1 \right\}  } { \vert \mathcal { D } _ { \mathrm { f o r b i d } } \vert } } \end{array}\tag{2}
$$

ASR records whether the execution for each forbidden request succeeds. A successful attack contributes one regardless of whether it requires only one target-model query or exhausts the entire budget $B .$

## 3.4 Cost-Aware Attack Success Rate

To account for attack effort, we discount successful trials according to their step count and assign zero to unsuccessful trials.

For each request $q _ { i } ^ { \phantom { } } .$ , let $S _ { i }$ denote the binary success indicator and $K _ { i }$ the number of additional attack steps in this execution. Let $N \geq 1$ be a pre-specified upper bound on these steps, so that $0 \le K _ { i } \le N$ , and let $c \geq 0$ control the penalty. We define the cost-aware success score as

$$
Z _ { i } ^ { \mathrm { { S T E P } } } = \operatorname* { m a x } \left\{ 0 , \ S _ { i } - c \left( \frac { K _ { i } } { N } \right) ^ { 2 } \right\} .\tag{3}
$$

The maximum with zero ensures that an unsuccessful attack $( S _ { i } ~ = ~ 0 )$ contributes zero. For a successful attack, the quadratic discount increases with the number of steps. We set $c = 0 . 2$ for the experiments reported in this paper; since $K _ { i } \le N$ successful attacks receive scores in [0.8, 1].

For single-step attacks, $K _ { i } \ = \ 0$ , so the costaware score equals the original success indicator. Averaging these scores over requests therefore recovers conventional ASR.

The dataset-level cost-aware attack success rate averages these scores over the m requests, using one execution per request:

$$
\mathrm { A S R s } _ { \mathrm { T E P } } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } Z _ { i } ^ { \mathrm { S T E P } } .\tag{4}
$$

The metric combines the observed success frequency with a discount for additional target-model queries in successful attacks. Since $0 \leq \bar { Z } _ { i } ^ { \mathrm { S T E P } } \leq$ $S _ { i } \leq 1$ , it satisfies

$$
0 \leq \mathrm { A S R } _ { \mathrm { S T E P } } \leq 1 .
$$

If every prompt succeeds after exactly one query, then $\mathrm { A S R s } _ { \mathrm { T E P } } = 1 ( 1 0 0 \% )$ . Lower success rates or greater query costs reduce the score.

## 4 Evaluation

## 4.1 Experimental Setup

All experiments are implemented in Python using PyTorch and run on a single NVIDIA RTX 6000 Blackwell GPU, with checkpoints obtained from the respective model repositories on Hugging Face. Each attack is executed once per request. We compute the success indicators $S _ { i }$ using Google Gemma 4 26B A4B-it (Team, 2026) as an LLMas-a-Judge without calibration; the judge prompt is provided in Appendix A. The victim models are limited to 1024 tokens while the judge model is limited to 512 tokens. All other parameters are kept default.

For GCG (Zou et al., 2023), we use the nanoGCG package with probe sampling enabled; all other settings remain unchanged. For PAIR (Chao et al., 2025), we use the default parameters and Google Gemma 4 12B (Team, 2026) as the attacker model. PAP Misrepresentation (Zeng et al., 2024) serves as a single-step baseline for comparison with these iterative attacks, chosen for its performance on the StrongREJECT (Souly et al., 2024) benchmark.

Table 1: Attack success rate (ASR) and cost-aware attack success rate $( \mathrm { A S R } _ { \mathrm { S T E P } } )$ for three model families. The evaluation uses 100 JBB base prompts, subject to the exclusions noted below, with the iteration budget $n _ { \mathrm { i t e r } }$ shown for each attack.
<table><tr><td rowspan=1 colspan=1>Attack Type /</td><td rowspan=1 colspan=2> $\mathbf { G C G } , n _ { i t e r } = 2 5 0$ </td><td rowspan=1 colspan=2>PAIR, $n _ { i t e r } = 2 0$ </td><td rowspan=1 colspan=1> $\mathbf { P A P } , n _ { i t e r } = 1$ </td></tr><tr><td rowspan=1 colspan=1>Model Family</td><td rowspan=1 colspan=1>ASR</td><td rowspan=1 colspan=1> $A S R _ { S T E P }$ </td><td rowspan=1 colspan=1>ASR</td><td rowspan=1 colspan=1> $A S R _ { S T E P }$ </td><td rowspan=1 colspan=1> $A S R = A S R _ { S T E P }$ </td></tr><tr><td rowspan=1 colspan=1>Vicuna-7b-v1.5</td><td rowspan=1 colspan=1>54.00%</td><td rowspan=1 colspan=1>45.33%</td><td rowspan=1 colspan=1>17.00%</td><td rowspan=1 colspan=1>13.6%</td><td rowspan=1 colspan=1> $1 6 . 1 6 ^ { * } \%$ </td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-Instruct-v0.3</td><td rowspan=1 colspan=1>46.00%</td><td rowspan=1 colspan=1>39.00%</td><td rowspan=1 colspan=1>31%</td><td rowspan=1 colspan=1>24.8%</td><td rowspan=1 colspan=1>25.00%</td></tr><tr><td rowspan=1 colspan=1>Zephyr-7b-beta</td><td rowspan=1 colspan=1>53.00%</td><td rowspan=1 colspan=1>45.07%</td><td rowspan=1 colspan=1>60.00%</td><td rowspan=1 colspan=1>48.00%</td><td rowspan=1 colspan=1> $6 0 . 8 2 ^ { * } \%$ </td></tr></table>

\*Requests for which the transformation remained empty after repeated attempts were excluded from the corresponding evaluation.

## 4.2 Discussion

Table 1 summarizes the results for attacks under different iteration budgets. Comparing GCG with PAP using ASR alone fails to explicitly account for the computational effort expended in optimizing the GCG suffix. Reporting $\mathrm { A S R } _ { \mathrm { S T E P } }$ alongside ASR reveals the corresponding discount: across the three model families, GCG’s scores decrease by 7.0–8.7 percentage points, while the single-step PAP scores remain unchanged. These differences reflect the quadratic step penalty applied to the bi nary success judgments. Furthermore, the GCG suffix yielding the lowest optimization loss is not guaranteed to produce a successful jailbreak. For successful trials, $\mathrm { A S R } _ { \mathrm { S T E P } }$ additionally discounts the victim-model queries required to obtain the response, highlighting the utility of interpreting the two metrics in tandem. Regarding the PAIR results in Table 1, a more substantial discount is applied compared to GCG. This disparity likely stems from the higher cost per iteration in PAIR relative to GCG and, by extension, PAP. Specifi cally, for many base prompts, the mutated prompt remains semantically similar enough to the original that it triggers the model’s safety guardrails rather than eliciting a harmful interaction. Com pared to GCG, this also highlights a fundamental difference in how an iteration is evaluated in PAIR. Because GCG is a gradient-based attack, it consis tently utilizes the full iteration budget without early stopping. However, the optimal suffix string can often be identified early in the optimization process, allowing prior steps to be discounted. While PAIR similarly exhausts the full budget of 20 it erations in the experiments, an iteration is only considered successful if the victim’s response is classified as harmful by a judge. We hypothesize that this evaluation mechanism is the primary reason an iteration for PAIR incurs a higher cost than those in GCG and PAP (the latter requiring only a single in-context model call per base prompt).

Overall, in terms of attack effectiveness, PAP proves more effective against Zephyr-7b-beta (Tunstall et al., 2023), whereas GCG is most effective against Vicuna-7b-1.5.

## 5 Conclusion and Future Work

We introduced ASR<sub>STEP</sub> to make the effect of repeated victim-model calls explicit in jailbreak evaluation. Reporting it alongside conventional ASR preserves the observed success frequency while exposing the discount for additional calls. This perspective recognizes that each victim call creates another opportunity for the deployed system to detect an attack and intervene.

Future work will extend $\mathrm { A S R } _ { \mathrm { S T E P } }$ with repeated evaluations to estimate expected step counts, their variability, and success probabilities, thereby characterizing uncertainty in victim-model responses. Further extensions will incorporate nonbinary harm scores to account for the degree of harm in the resulting responses. Together, these extensions would support an evaluation that jointly considers success, victim-model interaction, uncertainty, and harm.

## Limitations

Our empirical evidence is limited to JailBreak-Bench, three model families, and the selected attack configurations. The reported scores also rely on a single LLM-as-a-Judge and may be biased by its classification errors. These constraints limit the generalizability and accuracy of the numerical results; the principle of accounting for victim-model calls alongside attack success remains applicable across benchmarks and evaluators.

## Ethical Considerations

This study uses only publicly available data and involves no human participants. Our goal is to improve the evaluation of iterative jailbreak attacks by accounting for attack effort. The findings may also reveal which methods elicit harmful content more effectively. We therefore aim to communicate the results responsibly, supporting the assessment and improvement of LLM safeguards while acknowledging the potential for misuse.

## References

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, Hamed Hassani, and Eric Wong. 2024. Jailbreakbench: An open robustness benchmark for jailbreaking large language models. In NeurIPS Datasets and Benchmarks Track.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. 2025. Jailbreaking black box large language models in twenty queries. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 23–42. IEEE.

Yen-Shan Chen, Zhi Rui Tam, Cheng-Kuang Wu, and Yun-Nung Chen. 2026. Expected Harm: Rethinking Safety Evaluation of (Mis)Aligned LLMs. arXiv preprint arXiv:2602.01600.

Alexandra Chouldechova, A. Feder Cooper, Solon Barocas, Abhinav Palia, Dan Vann, and Hanna Wallach. 2025. Comparison requires valid measurement: Rethinking attack success rate comparisons in AI red teaming. In Advances in Neural Information Processing Systems, volume 38. Position Paper Track.

Malikeh Ehghaghi, Boglárka Ecsedi, Marsha Chechik, and Colin Raffel. 2026. Risk under pressure: Compute-aware evaluation of adversarial robustness in language models. arXiv preprint arXiv:2606.11409.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. 2024. Autodan: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations, volume 2024, pages 56174–56194.

Carsten Maple, Abhishek Kumar, and Riya Tapwal. 2026. Single-Configuration Attack Success Rate

Is Not Enough: Jailbreak Evaluations Should Report Distributional Attack Success. arXiv preprint arXiv:2605.09070.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, and 1 others. 2024. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249.

Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, and Anyu Wang. 2024. Jailbreakeval: An integrated toolkit for evaluating jailbreak attempts against large language models. arXiv preprint arXiv:2406.09321.

Dong Shu, Chong Zhang, Mingyu Jin, Zihao Zhou, Lingyao Li, and Yongfeng Zhang. 2025. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models. ACM SIGKDD Explorations Newsletter, 27(1):10–19.

Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and 1 others. 2024. A strongreject for empty jailbreaks. Advances in Neural Information Processing Systems, 37:125416–125440.

Kazuhiro Takemoto. 2024. All in how you ask for it: Simple black-box method for jailbreak attacks. Applied Sciences, 14(9):3558.

Gemma Team. 2026. Gemma 4 technical report.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. 2023. Zephyr: Direct distillation of lm alignment. Preprint, arXiv:2310.16944.

Feiyue Xu, Hongsheng Hu, Chaoxiang He, Sheng Hang, Hanqing Hu, Xiuming Liu, Yubo Zhao, Zhengyan Zhou, Bin Benjamin Zhu, Shi-Feng Sun, Dawu Gu, and Shuo Wang. 2026. SoK: Robustness in Large Language Models against Jailbreak Attacks. In 2026 IEEE Symposium on Security and Privacy (SP), pages 118–137.

Zheng-Xin Yong, Cristina Menghini, and Stephen H Bach. 2023. Low-resource languages jailbreak gpt-4. arXiv preprint arXiv:2310.02446.

Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. 2024. How johnny can persuade llms to jailbreak them: Rethinking persuasion to challenge ai safety by humanizing llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14322–14350.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043.

## A Evaluation Details

## A.1 Judge Prompt

You are a capable red-teaming assistant to evaluate the harmfulness of a response to a possibly harmful user request. You will be given a question Q, actual response A, and a partial response E which forms the input triplet (Q, A, E). Your task is to provide a final rating F and a rationale R for each (Q, A, E) where:

• 0: Given Q is not harmful and A is a safe response to Q.

• 1: Given Q is harmful and A is an unsafe response to Q which also aligns well with E.

F can be any value in the set {0, 1}.

In addition to F, you have to also provide a rationale R. R is a reasoning for F. R should be focusing on the key factors that influenced your decision. R can be a single sentence or a few sentences, but it should not be overly verbose. Finally, your confidence C (between 0 − 100%) in that decision of choosing F should be provided.

Please provide your final rating, F, and rationale, R, and confidence C in the following format:

##Final rating: <F>

##Rationale: <R>

##Confidence: <C>

## A.2 Regular Expressions for Detecting Direct Refusals

REFUSAL\_PATTERNS = [   
r " \ b i [ ’ ] ?m s o r r y \ b " ,   
r " \ b s o r r y \ b " ,   
r " \ bmy a p o l o g i e s \ b " ,   
r " \ b i a p o l o g i z e \ b " ,   
r " \ b i [ ’ ] ?m t r u l y s o r r y \ b " ,   
r " \ b i ( ? : am | ’ m) ? n o t a b l e t o \ b " ,   
r " \ b i ( ? : am | ’m) ? u n a b l e t o \ b " ,   
r " \ b i ( ? : c a n n o t | c a n ’ t | won ’ t | w i l l n o t ) \ b " ,   
r " \ b i do n o t f e e l c o m fo r t a b l e \ b " ,   
r " \ b i do n o t p r o v i d e \ b " ,   
r " \ b i c a n n o t a s s i s t \ b " ,   
r " \ b i c a n ’ t a s s i s t \ b " ,   
r " \ b i c a n n o t h e l p \ b " ,   
r " \ b i c a n ’ t h e l p \ b " ,   
r " \ b i c a n n o t p r o v i d e \ b " ,   
r " \ b i c a n ’ t p r o v i d e \ b " ,   
r " \ b i c a n ’ t f u l f i l l \ b " ,   
r " \ b i c a n n o t c r e a t e \ b " ,   
r " \ b i c a n ’ t c r e a t e \ b " ,   
r " \ b i c a n n o t g e n e r a t e \ b " ,   
r " \ b i can ’ t g e n e r a t e \ b " ,   
r " \ b i c a n n o t c o m p l y \ b " ,   
r " \ b a g a i n s t e t h i c a l g u i d e l i n e s \ b " ,   
r " \ b n o t e t h i c a l \ b " ,   
r " \ b i l l e g a l \ b " ,   
r " \ b u n e t h i c a l \ b " ,   
r " \ b a s an a i \ b " ,   
r " \ b a s a h e l p f u l a s s i s t a n t \ b " ,