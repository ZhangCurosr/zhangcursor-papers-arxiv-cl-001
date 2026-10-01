# OPSRD: ON-POLICY SELF-ROLE DISTILLATION

Weijie Ren<sup>1∗</sup> Yanwen Zhang<sup>2∗</sup> Hao Li<sup>3∗</sup> Zhuolin Qi<sup>1</sup> Hengyi Zhang<sup>1</sup> Naibo Wang<sup>1†</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>University of Electronic Science and Technology of China

<sup>3</sup>University of Science and Technology of China

3200101501@zju.edu.cn, 2023091601016@std.uestc.edu.cn,

haoli2101@mail.ustc.edu.cn, qizhuolin666@gmail.com,

22651274@zju.edu.cn, wangnaibo@zju.edu.cn

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

## ABSTRACT

Role prompting elicits specialized behavior from large language models through an expert identity, offering a lightweight way to guide reasoning on demanding tasks. However, evaluating or distilling complete role-prompted answers can miss useful next-token preferences when the sampled solution remains incorrect. Transferring these preferences also requires an objective that reaches alternatives the student rarely predicts. We introduce OPSRD, which uses a fixed expert role as privileged teaching context for on-policy self-distillation without reference solutions. A role-free student generates a trajectory, and a frozen instance of the same base model supplies role-conditioned distributions on its exact prefixes, exposing alternatives beyond the sampled continuation. Teacherweighted forward KL targets alternatives the student underestimates, with clipping to limit individual vocabulary contributions. Supervision is restricted to the highest-entropy half of student positions, concentrating learning where predictions are uncertain. Experiments on three competition-math benchmarks with Qwen3-1.7B, 4B, and 8B show improvements over the base models without role prompts at inference. Forward KL achieves the highest macro-averaged accuracy among the three evaluated divergences at every scale. Code is available at https://github.com/zhansan114514/OPSRD.

## 1 INTRODUCTION

Large language models serve as general-purpose assistants for reasoning, explanation, and problem solving. Their usefulness depends on how instructions elicit the capabilities needed for a particular task (Liu et al., 2023). Role prompting makes this dependence explicit by asking a model to adopt an expert identity (Xu et al., 2023). A mathematical expert role can request careful derivations and verification, while a critic role can encourage checking intermediate conclusions. Such prompts offer a practical way to guide model behavior without changing its parameters.

However, existing uses of role prompting often assess expertise through complete generated responses. Reported reasoning gains and failures (Kong et al., 2024; Kim et al., 2024) therefore mix the role’s local influence with the effects of sampling an entire solution. A useful next-token preference can remain unsampled or be followed by a later mistake. Distilling expert-prompted responses (Xu et al., 2023) likewise transfers only the sampled continuations. On-policy self-distillation accesses richer token distributions, but solution-conditioned OPSD derives its teaching advantage from a reference solution for each problem (Zhao et al., 2026). The open question is whether an expert role can supervise the student’s reasoning even when its complete answers offer no measured advantage.

Our starting observation is a negative result. On three competition-math benchmarks, adding an expert olympiad mathematician role to Qwen3-1.7B changes macro accuracy from 35.74 to 34.54, providing no measured improvement over the ordinary prompt. A neutral system instruction reaches the same score. The expert label alone is insufficient to improve sampled solutions in this setting. What remains unobserved is the role-conditioned distribution at each intermediate choice, before later decisions determine the final answer.

![](images/f3cbf03f38c3dfc410e9b8b688fd70432eb66c7ee261384c84b34de1cbb8fdc3.jpg)  
Figure 1: OPSRD overview. The student generates a role-free trajectory; the frozen expert-role teacher evaluates the same prefixes. The highest-entropy half of student positions receive clipped, full-vocabulary forward KL. Only student LoRA parameters are updated, and deployment uses the ordinary prompt at test time.

We hypothesize that a role-conditioned model can teach useful next-token preferences even when its complete solutions offer no accuracy gain. Under this hypothesis, the role’s value lies in shifting probability toward helpful continuations at uncertain student prefixes. Holding these prefixes fixed lets the teacher evaluate the choices the student actually faces. On-policy distillation provides this alignment (Agarwal et al., 2024; Gu et al., 2024), allowing us to test a reusable role as privileged teaching context without per-problem reference solutions.

We develop OPSRD to test this hypothesis. The student generates trajectories under its ordinary prompt. A frozen instance of the same base model receives the expert role and scores the exact student prefixes, supplying a distribution over possible next tokens. We retain the highest-entropy half of student positions to focus supervision on uncertain choices. Only a low-rank student adapter is updated (Hu et al., 2022); the teacher stays fixed, no reference answer enters training, and the trained student uses its ordinary prompt at inference.

For this feedback to help, the objective must reach alternatives the student currently underestimates. A reverse-KL update weakens as the student’s probability of an alternative approaches zero, even when the teacher supports it. Forward KL weights the target by teacher probability and directly corrects this underestimation before clipping. We therefore use forward KL, with a cap on individual vocabulary contributions to limit large discrepancies from an imperfect teacher. We compare forward KL, reverse KL, and Jensen–Shannon divergence under the same role and entropy mask.

Experiments cover Qwen3-1.7B, 4B, and 8B on three competition-math benchmarks. Forward KL achieves the highest macro accuracy among the evaluated divergences at all three scales (Figure 3). At 1.7B, OPSRD improves Base by 6.30 percentage points in the primary run, and independent training repeats retain the gain. Placement controls favor keeping the role in the teacher. The study contributes an empirical contrast between direct prompting and role-based supervision, an answer-free on-policy method, and an analysis of how divergence and role placement govern this transfer.

## 2 RELATED WORK

## 2.1 ON-POLICY DISTILLATION

On-policy distillation learns from student-generated trajectories, with the teacher evaluating the same prefixes (Agarwal et al., 2024; Gu et al., 2024). Sequence-level distillation instead trains on teacher generated outputs (Kim & Rush, 2016). Using student prefixes addresses the train–test distribution mismatch central to imitation learning (Ross et al., 2011). The teacher supplies distributional supervision (Hinton et al., 2015), while the student determines where it applies. Generalized knowledge distillation permits different divergences, and MiniLLM uses reverse KL. Entropyaware OPD adds forward KL when teacher entropy is high (Jin et al., 2026), linking the objective to teacher uncertainty.

The quality of feedback also depends on the prefixes the student visits. Speculative knowledge distillation lets the teacher replace poorly ranked student proposals, bringing the sampled trajectory closer to teacher-supported continuations (Xu et al., 2025). Other methods adjust the learning signal at a fixed prefix. Asymmetric OPD combines positive reinforcement with local divergence minimization (Jia et al., 2026), while empirical analyses identify unreliable teacher guidance and imbalanced token supervision as failure modes (Fu et al., 2026). CROP selects positions using counterfactual task relevance (Li et al., 2026), and spurious-signal filtering uses input dependence together with divergence to remove misleading updates (Jiang et al., 2026). This work establishes that visiting student states alone does not make every teacher preference equally useful.

Self-distillation changes where the teacher’s advantage comes from. Earlier approaches learn from model-generated reasoning, instructions, or filtered outputs (Zelikman et al., 2022; Wang et al., 2023; Gulcehre et al., 2023). OPSD uses one model under two contexts: its teacher sees a privileged solution and evaluates the question-only student’s rollouts (Zhao et al., 2026). This supplies tokenlevel feedback without a separately pretrained teacher. OPSRD builds on this contextual asymmetry, replacing the per-problem solution with a reusable expert role and asking how to transfer that weaker source of supervision.

## 2.2 ROLE-CONDITIONED SUPERVISION

Prompts can expose behavior that a model does not reliably express under an ordinary instruction. ExpertPrompting constructs an expert identity for each instruction, uses it to generate responses, and trains ExpertLLaMA on the resulting data (Xu et al., 2023). Role-play prompting also improves selected reasoning tasks (Kong et al., 2024), and RoleLLM studies how to elicit and train broader role-playing abilities (Wang et al., 2024). Controlled experiments find that persona prompts can impair reasoning as well as help it (Kim et al., 2024). The question for distillation is consequently how a role’s influence should be measured and transferred when its complete sampled answers offer an uncertain advantage.

Context distillation provides a bridge from conditional behavior to learned behavior by internalizing gains elicited by instructions and scratchpads (Snell et al., 2022). In-context learning distillation transfers the ability to learn from demonstrations (Huang et al., 2022), and self-distillation can reduce distribution gaps during fine-tuning (Yang et al., 2024). These studies show several ways that a model’s context can create a teaching signal. For a role teacher, the relevant signal may lie in its preferences among possible next tokens on an existing student prefix. Capturing these preferences gives access to guidance that a comparison of complete role-prompted solutions can miss.

In OPSRD, a single role supplies context for a teacher that shares the student’s base model. The teacher scores role-free student prefixes, and entropy selection concentrates forward-KL supervision on uncertain positions. This makes the role’s distributional feedback the object of distillation while keeping student generation under the ordinary prompt.

## 3 METHOD

## 3.1 SETTING

Let x be a math problem and let $c _ { S }$ denote the role-free training context. The student p contains low-rank trainable parameters on top of a pretrained model $p _ { \theta _ { 0 } }$ . For each problem, the student samples a reasoning trajectory

$$
y _ { 1 : T } \sim p _ { \theta } ( \cdot \mid c _ { S } , x ) ,\tag{1}
$$

without a role or reference solution. Sampled tokens are detached. The frozen teacher uses the same base weights with the student adapter disabled and an expert-role context $c _ { T }$ . Training uses a non-

![](images/292548645787b068946ea8a18a5e6b09e36603be10baaa916f306b7ebadfa0de.jpg)  
Figure 2: One OPSRD update. Student and teacher score the same prefix; clipped vocabulary contributions are summed and averaged over the selected high-entropy positions. Only the student adapter is updated before the next rollout. The small distributions are schematic.

thinking student template and a thinking teacher template; evaluation enables thinking. Training also includes a Problem: prefix that evaluation omits. Appendix A.2 specifies the complete prompts.

At position t, both models receive the exact student prefix $y _ { < t }$ . Let $z _ { S } ^ { t }$ and $z _ { T } ^ { t }$ be their logits. The distributions used in entropy selection and distillation apply temperature $\eta = 1 . 1$

$$
p _ { S } ^ { t } ( v ) = \mathrm { s o f t m a x } ( z _ { S } ^ { t } / \eta ) _ { v } , \quad z _ { S } ^ { t } = z _ { \theta } ( c _ { S } , x , y _ { < t } ) ,\tag{2}
$$

$$
p _ { T } ^ { t } ( v ) = \mathrm { s o f t m a x } ( z _ { T } ^ { t } / \eta ) _ { v } , \quad z _ { T } ^ { t } = z _ { \theta _ { 0 } } ( c _ { T } , x , y _ { < t } ) ,\tag{3}
$$

for every vocabulary item v. The teacher evaluates exactly the tokens sampled by the student. Sampling additionally applies top $- p = 0 . 9 5$ and $\mathrm { { t o p } } \cdot k = 2 0 $ ; the loss distributions retain the full vocabulary. Only student logits receive gradients; sampling and teacher probabilities remain detached.

## 3.2 ROLE-PRIVILEGED TEACHING CONTEXT

The teacher’s reusable role requests rigorous olympiad reasoning, verification, and a checked final answer, without task-specific hints. Holding $( x , y _ { < t } )$ fixed aligns the compared positions. Their distributional difference includes the role, thinking template, and adapter state. The neutral-teacher control shares the template and chat placement but uses a generic instruction. Figure 1 shows the overall method; Figure 2 follows one training update.

## 3.3 STUDENT-ENTROPY SELECTION

The role does not need equal authority everywhere. Many tokens are nearly fixed by syntax or an earlier decision, so we compute student entropy at each valid position,

$$
H _ { t } = - \sum _ { v \in \mathcal { V } } p _ { S } ^ { t } ( v ) \log p _ { S } ^ { t } ( v ) ,\tag{4}
$$

and retain the $k = \operatorname* { m a x } ( 1 , \lceil \rho T \rceil )$ largest values within each sequence, with default $\rho = 0 . 5$ . This assigns every sequence its own selection quota $m _ { t } \in \{ 0 , 1 \}$ . Longer sequences still contribute more selected positions to the token-averaged objective. Selection uses detached student logits, so the ranking receives no gradient; Appendix B illustrates the selection rule.

## 3.4 CLIPPED FORWARD-KL OBJECTIVE

For a selected position, the unmodified forward KL is

$$
D _ { t } = \sum _ { v \in \mathcal { V } } p _ { T } ^ { t } ( v ) \log \frac { p _ { T } ^ { t } ( v ) } { p _ { S } ^ { t } ( v ) } .\tag{5}
$$

We retain the full vocabulary and cap each signed contribution at $\tau = 0 . 0 5$ for the 1.7B and 4B runs and $\tau = 0 . 0 6$ for the 8B runs. Writing $d _ { t , v } = p _ { T } ^ { t } ( v ) \log [ p _ { T } ^ { t } ( v ) / p _ { S } ^ { t } ( v ) ]$ , the batch objective is

$$
\mathcal { L } ( \theta ) = \frac { \sum _ { i \in \mathcal { B } } \sum _ { t = 1 } ^ { T _ { i } } m _ { i , t } \sum _ { v \in \mathcal { V } } \operatorname* { m i n } \{ d _ { i , t , v } , \tau \} } { \operatorname* { m a x } ( 1 , \sum _ { i \in \mathcal { B } } \sum _ { t = 1 } ^ { T _ { i } } m _ { i , t } ) } .\tag{6}
$$

The clip limits a few large teacher preferences without truncating the vocabulary or changing the selected positions. The unmodified KL is nonnegative, but its vocabulary contributions have both signs. Capping only positive contributions can make the optimized sum negative. We therefore log the unclipped divergence separately from the training loss. Only LoRA parameters receive gradients; the teacher remains fixed.

## 3.5 WHY FORWARD KL FITS A ROLE TEACHER

The teacher’s contribution is a distribution over alternatives at an existing student prefix. Forward KL gives each alternative weight proportional to its teacher probability. To see the resulting update, consider an unclipped selected position. Its derivative with respect to student logit $z _ { S , j }$ is

$$
\frac { \partial D _ { \mathrm { F } } } { \partial z _ { S , j } } = \frac { p _ { S } ( j ) - p _ { T } ( j ) } { \eta } .\tag{7}
$$

When the teacher assigns appreciable probability to a continuation the student rarely predicts, this gradient directly increases its logit. For reverse KL, $\begin{array} { r } { D _ { \mathrm { R } } = \sum _ { v } p _ { S } ( v ) \log [ p _ { S } ( v ) / \bar { p _ { T } ( v ) } ] } \end{array}$ , the derivative instead becomes

$$
\frac { \partial D _ { \mathrm { R } } } { \partial z _ { S , j } } = \frac { p _ { S } ( j ) } { \eta } \left( \log \frac { p _ { S } ( j ) } { p _ { T } ( j ) } - D _ { \mathrm { R } } \right) .\tag{8}
$$

For fixed positive teacher probabilities, this update vanishes as $p _ { S } ( j )$ approaches zero. The distinction motivates forward KL for transferring preferences that the role elicits but the student underuses. It also complements entropy selection, which directs learning toward prefixes with several plausible continuations. The contribution cap modifies these derivatives when active; Appendix B gives the exact clipped forward gradient. The following experiments evaluate this choice under the implemented loss and training procedure.

## 3.6 WHY THE ASYMMETRY MATTERS

Teacher-only conditioning keeps the student persona-free during both rollout and evaluation. Shared or student-only conditioning changes the prefixes on which the teacher supplies supervision. Appendix B contrasts the three placements. Their comparison tests whether role-conditioned student sampling helps the final model under either evaluation prompt. Thinking-mode and formatting choices remain fixed across the placement ablation, as specified in Appendix A.2.

## 4 EXPERIMENTS

Our experiments test whether an expert role can provide useful supervision when it offers little benefit as a direct prompt. We first compare role distillation with prompting and answer-conditioned distillation, then examine the choices that govern the transfer. Repeated training and evaluation assess the stability of the gains, and a problem-level analysis shows how those gains change the student’s answers.

## 4.1 EXPERIMENTAL SETUP

Models and data. We use Qwen3-1.7B, 4B, and 8B (Yang et al., 2025), with a frozen instance of each base model serving as its own teacher. Training uses the 29,434-example mathematical split derived from OpenThoughts (Guha et al., 2026). All methods train a rank-64 LoRA adapter for 100 optimizer steps with learning rate $5 \times 1 0 ^ { - 6 }$ and effective batch size 36. The student samples one rollout per problem, capped at 1,024 tokens. We use 1.7B for the detailed ablations and independent training repeats; the larger models test whether the recipe remains useful as base accuracy increases. Appendix A gives the remaining hyperparameters and exact prompts.

Table 1: Primary Qwen3-1.7B comparison. Avg@12 in percent; gains in percentage points.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>HMMT25</td><td>Macro</td><td>∆ Base</td></tr><tr><td>Base</td><td>46.67</td><td>36.11</td><td>24.44</td><td>35.74</td><td>0.00</td></tr><tr><td>Answer OPSD</td><td>51.39</td><td>39.17</td><td>25.56</td><td>38.70</td><td>+2.96</td></tr><tr><td>Base with expert role</td><td>46.94</td><td>37.22</td><td>19.44</td><td>34.54</td><td>-1.20</td></tr><tr><td>Base with neutral prompt</td><td>45.83</td><td>35.83</td><td>21.94</td><td>34.54</td><td>-1.20</td></tr><tr><td>Role full</td><td>55.56</td><td>39.72</td><td>25.00</td><td>40.09</td><td>+4.35</td></tr><tr><td>OPSRD</td><td>57.78</td><td>41.67</td><td>26.67</td><td>42.04</td><td>+6.30</td></tr><tr><td>Neutral full</td><td>50.28</td><td>40.00</td><td>24.44</td><td>38.24</td><td>+2.50</td></tr></table>

Table 2: Macro Avg@12. The 4B summary averages four evaluation repeats.
<table><tr><td>Model</td><td>Evaluation</td><td>Base</td><td>Answer OPSD</td><td>OPSRD</td><td>∆ Base</td><td>∆ Answer</td></tr><tr><td>Qwen3-1.7B</td><td>Primary</td><td>35.74</td><td>38.70</td><td>42.04</td><td> $+ 6 . 3 0$ </td><td> $+ 3 . 3 3$ </td></tr><tr><td>Qwen3-4B</td><td>Mean of 4</td><td>60.86</td><td>61.50</td><td>62.52</td><td> $+ 1 . 6 7$ </td><td> $+ 1 . 0 2$ </td></tr><tr><td> $\mathbf { Q } \mathrm { w e n } 3 { - } 8 \mathbf { B }$ </td><td>Primary</td><td>62.78</td><td>64.26</td><td>65.83</td><td> $+ 3 . 0 6$ </td><td> $+ 1 . 5 7$ </td></tr></table>

Evaluation. We evaluate on AIME 2024, AIME 2025, and HMMT February 2025, each containing 30 problems. Every problem receives 12 samples at temperature 1.0 and top-p 0.95, with a 38,912- token generation cap. We report average sample accuracy for each task and its unweighted macro average, denoted Avg@12. All comparisons use the final step-100 checkpoint and the ordinary evaluation prompt unless a prompt change is the intervention being tested. Reported 95% intervals use paired problem-level bootstrap resampling, with the training repeats additionally accounting for shared test problems. Appendix D defines these estimators.

Baselines. Base is the untrained model. Direct expert and neutral prompting test whether an extra system message improves inference by itself. Answer OPSD supplies a reference solution to the teacher, following Zhao et al. (2026), and provides the main training baseline. Role full and Neutral full distill an expert or generic teacher instruction at every valid position. OPSRD combines the expert teacher with the highest-entropy half of student positions. This set of comparisons connects the practical benefit of role distillation to the teacher context, role placement, and training objective.

## 4.2 MAIN RESULTS

From prompting to supervision. The central result is a change in how the same role becomes useful. Direct expert prompting gives 34.54 macro accuracy, close to the 35.74 Base score, and the paired difference has an interval of [−4.35, 1.94]. Using that role for the teacher raises accuracy to 40.09 with full distillation and 42.04 with OPSRD (Table 1). The latter improves Base by 6.30 pp, with interval [3.24, 9.54], and Answer OPSD by 3.33 pp, with interval [0.83, 6.02]. The teacher receives the same short instruction for every training example, making the supervision reusable across problems without their reference solutions.

The benchmark results clarify this change. Direct prompting lowers HMMT accuracy while leaving the two AIME scores near Base. After distillation, all three task point estimates improve, with the largest gain on AIME 2024. The role’s failure to raise aggregate prompting accuracy therefore does not preclude useful feedback on the student’s prefixes. The placement and divergence ablations below examine how that feedback reaches the trained model.

Results across model sizes. Table 2 extends the comparison to stronger base models. OPSRD improves macro accuracy at all three scales, with gains of 6.30, 1.67, and 3.06 points over Base. The 4B entry averages four evaluations of each fixed checkpoint; the other scales use the primary evaluation. The gains are smaller as baseline accuracy rises, although their magnitude does not change monotonically with model size. Across these summaries, role distillation also exceeds the answer-conditioned comparator while requiring less privileged training context.

The per-task comparisons reveal different preferences for teacher context. Answer OPSD has the strongest 4B primary evaluation, and it slightly exceeds OPSRD on 8B AIME 2025. Averaging the repeated 4B evaluations changes the overall ordering. Role distillation remains useful across model sizes, while the strength of its advantage depends on the benchmark and evaluation sample. Appendix C provides the complete benchmark scores, and Section 4.4 assesses variation across all four repeated evaluations.

(b) Forward KL gain  
Table 3: Role placement and deployment prompt on Qwen3-1.7B.
<table><tr><td>Training role</td><td>Test role</td><td>Macro</td></tr><tr><td>Teacher only</td><td>No role</td><td>42.04</td></tr><tr><td>Teacher only</td><td>Expert role</td><td>39.44</td></tr><tr><td>Shared role</td><td>No role</td><td>38.98</td></tr><tr><td>Shared role</td><td>Expert role</td><td>36.94</td></tr><tr><td>Student only</td><td>No role</td><td>38.24</td></tr><tr><td>Student only</td><td>Expert role</td><td>36.30</td></tr></table>

![](images/cadac5c3cf41a82dc4416620541c704db3c10966c49a990edf4c11605cb2cc88.jpg)

![](images/434e0132ed7dc92c257fafe2bb86ed78f11c0e277809fc88d0ee34044d271c7c.jpg)  
Figure 3: Divergence ablation. (a) Macro Avg@12 from the primary evaluation at each scale. (b) Forward-minus-reverse gains with paired 95% problem-bootstrap intervals. Each condition uses one training run; JSD uses H100 and the KL runs use A100.

## 4.3 ABLATION STUDIES

Teacher context. We first ask how much of the benefit requires an expert instruction. Neutral full improves Base by 2.50 points, showing that a generic teacher context already supplies useful self-distillation feedback. Role full adds 1.85 points over this control, with interval [−0.28, 3.98]. The combined OPSRD recipe is 3.80 points above Neutral full, with interval [1.20, 6.39]. These results motivate the expert context, while the missing neutral-teacher 50% control leaves its contribution at the selected positions unresolved. The remaining ablations hold the expert instruction fixed and test how to transfer its feedback.

Role placement. The role can change either the student’s trajectory or the teacher’s view of it. To distinguish these effects, we place the role in the teacher, the student, or both, holding the model size, selection fraction, checkpoint, and evaluation protocol fixed. Each trained model is then evaluated with and without the role. This comparison tests the division of labor in OPSRD, where the student chooses the trajectory and the teacher supplies the additional perspective.

Teacher-only conditioning achieves the highest score under the ordinary evaluation prompt (Table 3). Giving the role to both models lowers accuracy by 3.06 points, and giving it only to the student lowers accuracy by 3.80 points. Both paired intervals lie below zero. Adding the role again at evaluation lowers each checkpoint’s point estimate, including the teacher-only model. The benefit of training with role feedback therefore persists when the student returns to its ordinary prompt. Conditioning the rollout generator on the role changes the training context in a way that these placement results do not favor.

Distillation objective. The next comparison tests how the student should absorb the role teacher’s distribution. We replace forward KL with reverse KL or JSD while retaining the expert teacher and 50% entropy mask. Forward KL gives the highest macro accuracy at all three scales (Figure 3). The clearest separation is at 1.7B, where it exceeds reverse KL by 4.81 points with interval [2.41, 7.31].

Table 4: Mean macro Avg@12 and sample standard deviation over three training runs at 1.7B and four evaluations of fixed checkpoints at 4B. The fixed 1.7B Base score is 35.74.
<table><tr><td>Model</td><td>Repeat</td><td>Method</td><td>Macro</td></tr><tr><td>1.7B</td><td>Training</td><td>Role full</td><td>39.48 ± 1.32</td></tr><tr><td>1.7B</td><td>Training</td><td>OPSRD</td><td>40.25±1.62</td></tr><tr><td>4B</td><td>Evaluation</td><td>Base</td><td> $6 0 . 8 6 \pm 0 . 6 7$ </td></tr><tr><td>4B</td><td>Evaluation</td><td>Answer OPSD</td><td> $6 1 . 5 0 \pm 0 . 7 7$ </td></tr><tr><td>4B</td><td>Evaluation</td><td>OPSRD</td><td> $6 2 . 5 2 \pm 0 . 7 8$ </td></tr></table>

This result agrees with the motivation in Section 3: teacher-supported alternatives receive a direct learning signal even when the student assigns them little probability.

The larger-model gaps are less conclusive. Forward-minus-reverse intervals cross zero at 4B and 8B, and reverse KL is stronger on 4B AIME 2024. We therefore use forward KL as the default supported most clearly by the primary scale. All divergence variants have one training run; JSD also differs in hardware and is supported by saved aggregate scores. Appendix C.2 gives the objective definitions, full task results, and comparison settings. The 4B result in this figure uses the primary evaluation, whereas Table 2 summarizes the repeated evaluations.

Supervision density. Finally, we test whether the gain survives concentrating supervision on uncertain positions. Selecting half the valid positions raises the primary score by 1.94 points over Role full, with interval [−0.56, 4.44]. Across three independent training runs, full and selective distillation remain close, with a mean difference of 0.77 points and interval [−1.33, 2.90]. The supported finding is that half the supervised positions retain the benefit of role distillation. Both variants still compute the student and teacher forward passes.

The entropy measurements help explain what this restriction does. Selected positions have mean student entropy 0.874, compared with 0.023 at omitted positions. The objective therefore focuses on predictions with substantial uncertainty while leaving nearly settled predictions outside the loss. A smaller sampling-budget pilot also explored 25% and 75% selection, but did not establish a better fraction. Appendix C.4 reports this sensitivity study separately from the full-budget comparison.

## 4.4 ROBUSTNESS

We assess whether the gains survive changes in optimization and evaluation sampling. At 1.7B, three independent training runs give OPSRD a mean accuracy of $4 0 . 2 5 \pm 1 . 6 2$ , improving the fixed Base evaluation by 4.51 points with interval [1.54, 7.69]. Every run improves Base. Role full also retains its gain, consistent with the supervision-density ablation. The repeat results support the value of role distillation more strongly than a further advantage from selecting half the positions.

At 4B, four evaluations of each fixed checkpoint yield a mean gain of 1.67 pp over Base, with sample standard deviation 0.39 pp for the paired gains. OPSRD exceeds Base in every evaluation. This agreement is useful because the primary comparison with Answer OPSD changes when evaluation sampling is repeated. Table 4 summarizes all repeats. These evaluations span A100 and H100, so their dispersion includes environment variation as well as decoding randomness; independent-training variation is measured by the separate 1.7B study.

## 4.5 FURTHER ANALYSIS

How the answers improve. An increase in average accuracy can reflect more reliable solutions to familiar problems or successful attempts on previously unsolved ones. We distinguish these outcomes by pairing Base and OPSRD on every test problem. Figure 4 shows that OPSRD increases the 12-sample success rate on 38 problems and lowers it on 12, with the rest tied. The improvements are most widespread on AIME 2024. This pattern accounts for the stronger AIME gains in the aggregate comparison and the repeated-training summary.

The change in observed coverage is much smaller. Seven problems with no correct Base sample receive a correct OPSRD sample, while six change in the opposite direction. Most improved problems already have at least one successful Base trajectory. The main benefit is a better chance of

(a) Change in sample accuracy

![](images/eb400aedd03610eeac127e673f6f3ea6a6e69c57998df26aad49f655e8c41448.jpg)

(b) Observed success coverage  
![](images/0aaf89a658f6168b82c6670f2d610498d0342a0fa4137a75f7fa104a2fa179b1.jpg)  
Qwen3-1.7B · 90 problems · 12 samples per problem

Figure 4: Problem-level changes at 1.7B. Left: problems with higher, equal, or lower sample accuracy than Base. Right: observed success coverage within 12 samples. The main gain is more reliable success on partially solved problems.

reaching a correct answer on problems the student sometimes solves already. This is consistent with using a role teacher’s local preferences to strengthen reasoning that the ordinary student can already express under its usual prompt.

This distinction connects the evaluation to the training design. OPSRD learns on student-generated prefixes, so useful feedback must act within trajectories the student actually visits. The problem-level result is compatible with that local form of guidance. It also explains why an improvement in Avg@12 can coexist with little change in the number of problems solved at least once. Coverage here is measured at a finite sampling budget; broader claims about newly acquired problem-solving abilities would require a different evaluation.

Answer format and length. We also check whether the primary improvement can be explained by output failures. Format compliance exceeds 99% across the primary conditions, and both Base and OPSRD rarely reach the generation cap. The additional 68 correct generations exceed the differences in malformed and near-cap outputs. These discrepancies are too small individually to explain the gain, supporting the interpretation that the observed change concerns answer accuracy under the shared long-form evaluation budget.

## 5 DISCUSSION AND CONCLUSION

The experiments connect the role’s placement to the distribution it teaches. Direct prompting changes the student’s trajectory from its first token. Teacher-only distillation instead evaluates the trajectory the student already produced, and the placement controls favor keeping these two functions separate. Forward KL weights the update by the teacher’s probabilities, allowing teacher-supported alternatives to receive a learning signal even when the student rarely predicts them. Equation 7 describes this behavior before clipping, while Appendix B gives the gradient of the implemented objective. Tracking those alternatives through training would test how this local signal produces the observed answer-level gains. Both full and selective distillation improve Base across training seeds. Their paired interval crosses zero, so the selection result supports retaining the benefit at half the supervised positions. The implementation still requires both full forward passes.

Limitations. The evidence covers one model family, one mathematical training corpus, three competition benchmarks, and a matched 1.7B-to-8B scale suite. Three training seeds do not characterize every source of optimization variance. Evaluating other model families and tasks would test the broader reach of this training recipe.

OPSRD turns a fixed expert role into supervision along the student’s own reasoning. A frozen instance of the same base model provides the teacher distribution, and the student learns from it through forward KL at uncertain positions. Across the three tested scales, forward KL achieves the highest macro score among the evaluated divergences, with the clearest advantage over reverse KL at 1.7B. Training-seed repeats retain the gain over Base, and the trained student solves problems with an ordinary prompt. A role can therefore be useful even when direct prompting offers no measured improvement. Its teaching value lies in the guidance it supplies along the student’s reasoning.

## REPRODUCIBILITY STATEMENT

Our code is publicly available at https://github.com/zhansan114514/OPSRD. The repository contains the training and evaluation code, the launch scripts with their hyperparameters, the analysis scripts used to summarize results and compute paired bootstrap intervals, tests for prompt isolation and the objective, and the recorded Python environment. Appendix A lists the prompts and settings, Appendix D defines the resampling units, and Appendix B specifies the loss reduction. Model checkpoints are not redistributed; the dataset and base models are public on Hugging Face.

## AI USE STATEMENT

During the experiments, we used large language models (LLMs) as an auxiliary tool for experiment monitoring and management. Specifically, the LLMs were used to monitor experiment logs and runtime information, identify potential execution issues or anomalies, and assist in reporting the status of ongoing experiments. The LLMs did not determine the research questions, experimental methodology, hyperparameter settings, or final experimental conclusions. All experimental configura tions, result verification, analysis, and scientific conclusions were determined and validated by the authors. We take full responsibility for the final content and results of this work.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=3zKtaqxLhW.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv preprint arXiv:2603.25562, 2026. URL https://arxiv.org/abs/2603.25562.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=5h0qf7IBZZ.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Stutee Acharya, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. OpenThoughts: Data recipes for reasoning models. In The Fourteenth International Conference on Learning Representations, 2026.

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (ReST) for language modeling. arXiv preprint arXiv:2308.08998, 2023. URL https://arxiv.org/abs/2308.08998.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. URL https://arxiv.org/abs/1503.02531.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Yukun Huang, Yanda Chen, Zhou Yu, and Kathleen McKeown. In-context learning distillation: Transferring few-shot learning ability of pre-trained language models. arXiv preprint arXiv:2212.10670, 2022. URL https://arxiv.org/abs/2212.10670.

Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, and Zequn Sun. Asymmetric on-policy distillation: Bridging exploitation and imitation at the token level. arXiv preprint arXiv:2605.06387, 2026. URL https://arxiv.org/abs/ 2605.06387.

Yinuo Jiang, Yongjie Ye, Zhou Tao, Xiang Zhuang, Qiang Zhang, Huajun Chen, and Tiankai Li. When teachers mislead: Spurious-signal-aware on-policy distillation. arXiv preprint arXiv:2608.03632, 2026. URL https://arxiv.org/abs/2608.03632.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=J5i09faOOf.

Junseok Kim, Nakyeong Yang, and Kyomin Jung. Persona is a double-edged sword: Mitigating the negative impact of role-playing prompts in zero-shot reasoning tasks. arXiv preprint arXiv:2408.08631, 2024. URL https://arxiv.org/abs/2408.08631.

Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1317–1327, Austin, Texas, 2016. doi: 10.18653/v1/D16-1139.

Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou, Enzhi Wang, and Xiaohang Dong. Better zero-shot reasoning with role-play prompting. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4099–4113. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.naacl-long.228. URL https: //aclanthology.org/2024.naacl-long.228/.

Enhan Li, Junhao He, and Hongyang Du. CROP: Task relevance via counterfactuals for selective on-policy distillation. arXiv preprint arXiv:2608.13387, 2026. URL https://arxiv.org/ abs/2608.13387.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35, 2023. doi: 10.1145/3560815.

Stephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings ofMachine Learning Research, pp. 627–635. PMLR, 2011.

Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context. arXiv preprint arXiv:2209.15189, 2022. URL https://arxiv.org/abs/2209.15189.

Noah Wang, Z.Y. Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian Yang, Man Zhang, Zhaoxiang Zhang, Wanli Ouyang, Ke Xu, Wenhao Huang, Jie Fu, and Junran Peng. RoleLLM: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 14743–14777. Association for Computational Linguistics, 2024. doi: 10.18653/ v1/2024.findings-acl.878. URL https://aclanthology.org/2024.findings-acl. 878/.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-Instruct: Aligning language models with self-generated instructions. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 13484–13508, Toronto, Canada, 2023. doi: 10.18653/v1/2023.acl-long.754.

Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, and Zhendong Mao. ExpertPrompting: Instructing large language models to be distinguished experts. arXiv preprint arXiv:2305.14688, 2023. URL https://arxiv.org/abs/2305.14688.

Wenda Xu, Rujun Han, Zifeng Wang, Long Le, Dhruv Madeka, Lei Li, William Yang Wang, Rishabh Agarwal, Chen-Yu Lee, and Tomas Pfister. Speculative knowledge distillation: Bridging the teacherstudent gap through interleaved sampling. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=EgJhwYR2tB.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Zhaorui Yang, Tianyu Pang, Haozhe Feng, Han Wang, Wei Chen, Minfeng Zhu, and Qian Liu. Self-distillation bridges distribution gap in language model fine-tuning. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1028–1043. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.58. URL https://aclanthology.org/2024.acl-long.58/.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping reasoning with reasoning. In Advances in Neural Information Processing Systems, volume 35, pp. 15476–15488. Curran Associates, Inc., 2022.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=Jpxfof0EaS.

## A EXPERIMENTAL DETAILS

We provide the prompts, hyperparameters, and objective details needed to reproduce the comparisons. Our public repository (https://github.com/zhansan114514/OPSRD) contains the training and evaluation code, launch configurations, and scripts for summarizing results.

## A.1 DATA AND HYPERPARAMETERS

Training uses the train split of siyanzhao/Openthoughts\_math\_30k\_opsd, containing 29,434 examples. Role distillation uses the problem field; Answer OPSD additionally provides the solution to its teacher. We reuse the published split without additional filtering or decontamination. The original runs did not pin a dataset revision. Table 5 summarizes training and evaluation, with the same final checkpoint used for all benchmarks.

Table 5: Training and evaluation settings.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Optimizer / steps</td><td>AdamW / 100</td></tr><tr><td>Learning rate / maximum gradient norm</td><td> $5 \times 1 0 ^ { - 6 } / 0 . 1$ </td></tr><tr><td>Effective batch size / rollouts per problem</td><td>36 / 1</td></tr><tr><td>LoRA rank / scale</td><td>64 / 128</td></tr><tr><td>LoRA projections</td><td>q, k, v, o, gate, up, down</td></tr><tr><td>Training temperature / top-p / top-k</td><td>1.1 / 0.95 / 20</td></tr><tr><td>Training rollout cap</td><td>1,024 tokens</td></tr><tr><td>Distillation temperature / vocabulary</td><td>1.1 / Full</td></tr><tr><td>Contribution cap at 1.7B and 4B / at 8B</td><td>0.05 / 0.06</td></tr><tr><td>Selected fraction for OPSRD</td><td>0.50 per sequence</td></tr><tr><td>Precision / attention</td><td>BF16 / SDPA</td></tr><tr><td>Evaluation samples per problem</td><td>12</td></tr><tr><td>Evaluation temperature / top-p</td><td>1.0 / 0.95</td></tr><tr><td>Evaluation top-k</td><td>Disabled</td></tr><tr><td>Evaluation generation cap / context limit</td><td>38,912 / 40,960 tokens</td></tr></table>

Answer OPSD is an internal reproduction with the same models, tasks, and metric as the cited study. Our effective batch of 36, SDPA attention, and fixed final checkpoint differ from its batch-32, FlashAttention-2 setup and checkpoint selection. Its published scores and our reproduced scores consequently refer to different training and evaluation configurations.

## A.2 PROMPTS

Expert role. The teacher receives the following system message for every problem.

You are an expert olympiad mathematician. Solve the problem with rigorous, independent reasoning. Identify the key structure before calculating, verify each non-trivial step, reconsider doubtful branches, and check the final answer.

Neutral context. The generic control shares the expert teacher’s thinking mode and system-message placement, with the following instruction. It controls for an additional instruction, without matching the expert message’s length or semantics.

You are an assistant. Respond to the user’s request clearly and carefully. Follow the requested output format, keep the response relevant, and review the answer before finishing.

Problem and answer format. The training user message contains Problem:, the problem, two newlines, and the instruction below. The Qwen3 chat template disables student thinking and enables teacher thinking. Evaluation enables thinking and omits the literal Problem: prefix; these choice remain fixed across the prompt and placement controls.

Please reason step by step, and put your final answer within \boxed{}.

![](images/1bd09e95f94b8d5aee195b976c7fd0fa0b3c4fb2c40931d6af583e0f583874a0.jpg)

![](images/be0b45d897dc0480adabbef57c51e86c90579cdbe3dca43842014a37258038fa.jpg)  
Figure 5: Selection and role placement. (a) Entropy ranking retains four of eight schematic positions. (b) Teacher-only conditioning keeps the student role-free during training and deployment.

Shared-role and student-only variants add the expert message before student sampling. Each checkpoint is evaluated both with and without that system message.

Reference-solution teacher. Answer OPSD encloses the solution between === Reference Solution Begin === and === Reference Solution End ===. It then adds the following instruction and the boxed-answer request above.

After reading the reference solution above, make sure you truly understand the reasoning behind each step—do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and don’t be afraid to backtrack or reconsider if something doesn’t work out:

All teacher variants score the student’s sampled completion after their respective contexts. Role and neutral distillation omit the reference solution from the model input.

## B ALGORITHM DETAILS

## B.1 TRAINING STEP

For each batch, the adapted student first samples one completion per problem. The completion is appended to both contexts, and the student and frozen teacher score identical sampled positions. The teacher pass disables the student’s adapter. Teacher probabilities and sampled tokens receive no gradients during student optimization.

Student entropy is computed from full-vocabulary float32 probabilities. Independently for each sequence, we retain the max(1, ⌈ρT⌉) valid positions with highest entropy (Figure 5). Padding and prompt positions are excluded. At the retained positions, we cap individual vocabulary contributions, sum over the full vocabulary, and average over the selected tokens. Entropy computation is chunked to reduce peak memory, preserving the same per-position values as the unchunked calculation.

The loss is evaluated per device and microbatch, after which distributed training averages local losses and accumulates gradients. With unequal selected-token counts, this reduction differs from weighting every token globally across devices. The implementation preserves the local reduction for all comparisons. Both full forward passes remain necessary after entropy selection.

## B.2 CLIPPED GRADIENT

For a selected position, let $A _ { t } = \{ v : d _ { t , v } < \tau \}$ denote contributions below the cap. Away from the cap boundary, the derivative of the implemented loss with respect to a student logit is

$$
\frac { \partial \ell _ { t } } { \partial z _ { S , j } ^ { t } } = \frac { 1 } { \eta } \left( p _ { S } ^ { t } ( j ) \sum _ { v \in A _ { t } } p _ { T } ^ { t } ( v ) - \mathbf { 1 } [ j \in A _ { t } ] p _ { T } ^ { t } ( j ) \right) .\tag{9}
$$

Table 6: Per-benchmark scores from the primary evaluation at each scale.
<table><tr><td>Model</td><td>Method</td><td>AIME24</td><td>AIME25</td><td>HMMT25</td><td>Macro</td><td>∆ Base</td></tr><tr><td>Qwen3-1.7B</td><td>Base</td><td>46.67</td><td>36.11</td><td>24.44</td><td>35.74</td><td>0.00</td></tr><tr><td>Qwen3-1.7B</td><td>Answer OPSD</td><td>51.39</td><td>39.17</td><td>25.56</td><td>38.70</td><td>+2.96</td></tr><tr><td>Qwen3-1.7B</td><td>OPSRD</td><td>57.78</td><td>41.67</td><td>26.67</td><td>42.04</td><td>+6.30</td></tr><tr><td>Qwen3-4B</td><td>Base</td><td>73.61</td><td>66.11</td><td>41.11</td><td>60.28</td><td>0.00</td></tr><tr><td>Qwen3-4B</td><td>Answer OPSD</td><td>75.00</td><td>68.61</td><td>43.89</td><td>62.50</td><td>+2.22</td></tr><tr><td>Qwen3-4B</td><td>OPSRD</td><td>72.22</td><td>67.78</td><td>44.44</td><td>61.48</td><td>+1.20</td></tr><tr><td>Qwen3-8B</td><td>Base</td><td>75.56</td><td>69.44</td><td>43.33</td><td>62.78</td><td>0.00</td></tr><tr><td>Qwen3-8B</td><td>Answer OPSD</td><td>75.83</td><td>71.67</td><td>45.28</td><td>64.26</td><td>+1.48</td></tr><tr><td>Qwen3-8B</td><td>OPSRD</td><td>77.22</td><td>71.39</td><td>48.89</td><td>65.83</td><td>+3.06</td></tr></table>

Table 7: Per-benchmark divergence results. JSD uses recorded aggregate scores.
<table><tr><td>Model</td><td>Objective</td><td>GPU</td><td>AIME24</td><td>AIME25</td><td>HMMT25</td><td>Macro</td></tr><tr><td>1.7B</td><td>Forward KL</td><td>A100</td><td>57.78</td><td>41.67</td><td>26.67</td><td>42.04</td></tr><tr><td>1.7B</td><td>Reverse KL</td><td>A100</td><td>50.56</td><td>35.56</td><td>25.56</td><td>37.22</td></tr><tr><td>1.7B</td><td>JSD</td><td>H100</td><td>51.39</td><td>38.61</td><td>22.22</td><td>37.41</td></tr><tr><td>4B</td><td>Forward KL</td><td>A100</td><td>72.22</td><td>67.78</td><td>44.44</td><td>61.48</td></tr><tr><td>4B</td><td>Reverse KL</td><td>A100</td><td>74.72</td><td>67.50</td><td>40.56</td><td>60.93</td></tr><tr><td>4B</td><td>JSD</td><td>H100</td><td>71.94</td><td>65.28</td><td>43.06</td><td>60.09</td></tr><tr><td>8B</td><td>Forward KL</td><td>A100</td><td>77.22</td><td>71.39</td><td>48.89</td><td>65.83</td></tr><tr><td>8B</td><td>Reverse KL</td><td>A100</td><td>75.00</td><td>70.00</td><td>46.39</td><td>63.80</td></tr><tr><td>8B</td><td>JSD</td><td>H100</td><td>75.56</td><td>69.17</td><td>45.28</td><td>63.33</td></tr></table>

Only contributions below the cap differentiate through $- p _ { T } ^ { t } ( v ) \log p _ { S } ^ { t } ( v )$ . When every contribution is active, the gradient reduces to $\bar { ( p _ { S } ^ { t } ( j ) - p _ { T } ^ { t } ( j ) ) / \eta }$ . The framework selects a one-sided derivative at the cap boundary. Capped vocabulary entries have zero direct derivative but still affect other entries through softmax normalization. Teacher outputs and the entropy ranking remain detached.

## C ADDITIONAL RESULTS

## C.1 BENCHMARK SCORES

Table 6 gives per-benchmark results for the primary evaluation at each model size. The 4B values here correspond to one evaluation of the same checkpoints used in the repeated-evaluation summary in the main paper. Each condition has 12 samples for each of the 90 test problems.

## C.2 DIVERGENCE ABLATION

Forward and reverse KL respectively use $\mathrm { K L } ( p _ { T } \| p _ { S } )$ and $\mathrm { K L } ( p _ { S } \| p _ { T } )$ . JSD uses the equally weighted mixture $M = ( p _ { T } + p _ { S } ) / 2$

$$
\begin{array} { r } { D _ { \mathrm { J S D } } = \frac { 1 } { 2 } \mathrm { K L } ( p _ { T } \| M ) + \frac { 1 } { 2 } \mathrm { K L } ( p _ { S } \| M ) . } \end{array}\tag{10}
$$

Each objective retains the full vocabulary, applies the same per-contribution upper cap, and reduces over the selected positions. All variants use the expert teacher, role-free student, 50% entropy fraction, and the training budget in Table 5.

The KL runs use A100 and JSD uses H100. Device counts and microbatch layouts differ in some runs while preserving the nominal effective batch; local token reduction and rollout scheduling can therefore vary. The paired forward-versus-reverse intervals describe problem uncertainty for one training run per condition. They are [2.41, 7.31], [−1.85, 2.96], and [−0.56, 4.81] at 1.7B, 4B, and 8B. JSD scores are supported by contemporaneous aggregate records; per-sample JSD outputs are not publicly available.

## C.3 ROLE PLACEMENT

Table 8 separates the placement comparison by benchmark. The role-free student and role teacher define the teacher-only condition. Shared role conditions both models; student only conditions

Table 8: Complete placement ablation, relative to teacher-only training with no test role.
<table><tr><td>Training role</td><td>Test role</td><td>AIME24</td><td>AIME25</td><td>HMMT25</td><td>Macro</td><td> $\Delta$ </td></tr><tr><td>Teacher only</td><td>No role</td><td>57.78</td><td>41.67</td><td>26.67</td><td>42.04</td><td>0.00</td></tr><tr><td>Teacher only</td><td>Expert role</td><td>51.11</td><td>41.94</td><td>25.28</td><td>39.44</td><td>-2.59</td></tr><tr><td>Shared role</td><td>No role</td><td>52.22</td><td>40.83</td><td>23.89</td><td>38.98</td><td>-3.06</td></tr><tr><td>Shared role</td><td>Expert role</td><td>47.50</td><td>39.44</td><td>23.89</td><td>36.94</td><td>-5.09</td></tr><tr><td>Student only</td><td>No role</td><td>50.28</td><td>39.72</td><td>24.72</td><td>38.24</td><td>-3.80</td></tr><tr><td>Student only</td><td>Expert role</td><td>47.22</td><td>38.33</td><td>23.33</td><td>36.30</td><td>-5.74</td></tr></table>

Table 9: Entropy selection in the primary run.
<table><tr><td rowspan=1 colspan=1>Quantity                       Mean</td></tr><tr><td rowspan=2 colspan=1>Selected valid-token fraction   0.500</td></tr><tr><td rowspan=2 colspan=1>Selected positii</td></tr><tr><td rowspan=2 colspan=1>0.8740.023</td></tr><tr><td rowspan=1 colspan=1>Omitted-positic</td></tr><tr><td rowspan=1 colspan=1>Unclipped selected divergence 0.194</td></tr></table>

the rollout generator. All variants use the same 1.7B protocol and selection fraction. Relative to teacher-only training with ordinary evaluation, shared-role and student-only training have paired macro intervals of [−5.83, −0.37] and $[ - 6 . 4 8 , - 1 . 3 0 ]$ under ordinary evaluation.

## C.4 SELECTION SENSITIVITY

Selected positions have much higher student entropy than omitted positions (Table 9). This confirms that the mask concentrates the objective on uncertain predictions. The measured divergence also shows a nonzero teacher signal at the retained positions.

The fraction pilot compares 25%, 50%, and 75% selection with four samples per problem and the same generation cap as the main evaluation (Table 10). The 50% reference uses its first four matched samples. The 25% result ties the reference and 75% scores lower. Neither reaches the one-point improvement required for a full-budget follow-up. These exploratory measurements motivate retaining the default fraction; the principal density comparison uses the complete 50% and 100% evaluations.

## D STATISTICAL ANALYSIS

Metric. For benchmark b with 30 problems and S = 12 samples per problem, let $a _ { j , s }$ denote answer correctness. We compute

$$
\mathrm { A v g @ 1 2 } _ { b } = \frac { 1 0 0 } { 3 0 S } \sum _ { j = 1 } ^ { 3 0 } \sum _ { s = 1 } ^ { S } a _ { j , s } .\tag{11}
$$

The macro score averages the three benchmark scores. Equal problem and sample counts make it identical to pooled accuracy over 1,080 generations. Observed success coverage instead counts problems with any correct sample, which is the quantity contrasted with average accuracy in Figure 4.

Paired intervals. We average each problem’s 12 correctness labels, align conditions by benchmark and problem, and resample their paired accuracy differences. Per-task intervals resample 30 problems; pooled intervals resample all 90 problems. Primary comparisons use 10,000 bootstrap draws and the divergence comparison uses 50,000. This preserves the problem as the sampling unit rather than treating repeated generations as independent problems.

Training and evaluation repeats. For three-run training comparisons, each of 30,000 bootstrap draws resamples three runs and 30 problems within each benchmark. The same problem draw is shared across the sampled runs, preserving dependence from the common test set and fixed Base evaluation. Reported standard deviations are sample standard deviations across the stated repeats. The 4B repeats hold trained weights fixed and span A100 and H100; their dispersion combines decoding and environment variation. Training variability is assessed separately at 1.7B.

Table 10: Selection-fraction pilot with four samples per problem.
<table><tr><td>Fraction</td><td>Avg@4</td><td>∆50%</td></tr><tr><td>25%</td><td>43.06</td><td>0.00</td></tr><tr><td>50%</td><td>43.06</td><td>0.00</td></tr><tr><td>75%</td><td>38.89</td><td>-4.17</td></tr></table>

Study design. Intervals apply to the stated comparison without adjustment for multiple ablations. Additional training runs followed a favorable primary result, and alternative selection fractions were screened at a smaller sampling budget. These comparisons form an exploratory study. Each summary includes every completed repeat or model scale belonging to that comparison.

## E BROADER IMPACT

The experiments use public mathematical problems and model-generated reasoning, with no human subjects or private records. Expertise labels require task-specific validation before deployment because a role can change model behavior unpredictably. Training uses student and teacher forward passes, while inference uses only the adapted student and its ordinary prompt.