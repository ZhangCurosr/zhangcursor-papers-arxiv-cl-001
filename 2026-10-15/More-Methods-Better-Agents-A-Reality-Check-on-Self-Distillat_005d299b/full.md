# More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning

Linhao Zhong<sup>1,2,\*</sup>, Bozhen Fang<sup>1,\*</sup>, Xuehao Wang<sup>1</sup>, Duochao Shi<sup>1</sup>, Zekai Luo<sup>1</sup>, Guangming Yao<sup>2</sup>, Yi Yuan<sup>2</sup>, Hao Chen<sup>1</sup>, Jingdong Chen<sup>2</sup>, Chunhua Shen<sup>1,2</sup>

<sup>1</sup>Zhejiang University, State Lab of CAD & CG <sup>2</sup>Ant Group <sup>\*</sup>Equal contribution.

Self-distillation in agentic reinforcement learning has produced a growing number of methods. However, existing evaluations often reuse published baseline scores under difering experimental conditions, provide few direct comparisons among contemporaneous methods, and give limited attention to training stability and computational cost. These limitations obscure the source of reported gains and the practical strengths and weaknesses of competing methods. Costly agent training further motivates a common basis for comparison. To address these gaps, we compare ten representative methods against GRPO on ALFWorld, ScienceWorld, Search-QA, and WebShop across three model sizes. Our study comprises 132 training runs totaling 12,124.4 recorded H800 GPU-hours, with repeated evaluations over eight rollout seeds for the pretrained backbones and all eleven trained methods. Beyond final task performance, we examine training progress, stability, and measured computational cost. Our evaluation yields three key findings. First, gains over GRPO remain limited in scope, with no method consistently outperforming it across settings. Second, strong final performance can mask unstable training. Third, additional training cost does not consistently translate into better performance. We establish this benchmark to support fair, comprehensive comparison and guide future research toward reliable and cost-efective improvements in agent training.

Date: October 2026 Project page: https://aim-uofa.github.io/Self-Distillation-AgentRL GitHub repo: https://github.com/aim-uofa/Self-Distillation-AgentRL

## 1 Introduction

Large language model (LLM) agents interact with environments through sequences of actions and observations (Shridhar et al., 2021; Yao et al., 2022). Reinforcement learning (RL) with sparse outcome rewards trains these agents from task-level feedback, but provides limited guidance for individual decisions within a trajectory. On-policy self-distillation (OPSD) provides denser supervision for student-generated trajectories using a teacher branch conditioned on training-only privileged information (Lu et al., 2026; Wang et al., 2026a). Recent work integrates this teacher guidance into RL through auxiliary losses, advantage modulation, or reward shaping (Lu et al., 2026; Yang et al., 2026c; Zhang et al., 2026; Wang et al., 2026b), with related preprints appearing at an average rate of roughly one every two days.<sup>1</sup>

However, current evaluations of self-distillation in agentic reinforcement learning provide an incomplete basis for judging progress. Individual gains over established baselines do not establish which contemporaneous methods are stronger, yet systematic comparisons among these methods remain scarce. Moreover, most comparisons reuse baseline scores from earlier papers rather than reproducing the baselines under a common protocol, allowing diferences in training and evaluation conditions to be mistaken for methodological progress. Many studies also emphasize final scores, leaving training instability and the computational cost of improvement less systematically examined. As Figure 1 illustrates, the proliferation of methods has outpaced the comparable evidence needed to decide which ones are worth adopting. Closing this gap requires substantial resources: reproducing agent training is costly, making a shared, systematic comparison dificult to assemble within each new method paper. We therefore argue that independent evaluation should take priority over proposing another method: the immediate need is to determine what is worth building on.

![](images/10e3ad341a7cdb644b34ca70e2cead9d629ee86128f92cb7720111bb3ed9d860.jpg)  
Figure 1 Motivation and overview of our benchmark for self-distillation in agentic reinforcement learning. Diferences in experimental conditions, reused baseline scores, and incomplete evaluation motivate a common comparison of ten representative methods against GRPO across four benchmarks and three model sizes. We examine the breadth of gains, training stability, and computational cost.

To address these limitations, we present an independent benchmark for self-distillation in agentic reinforcement learning. We compare ten representative methods against GRPO (Shao et al., 2024) across four benchmarks— ALFWorld (Shridhar et al., 2021), ScienceWorld (Wang et al., 2022), Search-QA, and WebShop (Yao et al., 2022)—and three scales (2B, 4B, and 9B). This comparison spans 12 model–task settings, 132 training runs, 2,772 training-time validation sweeps, and 1,152 evaluation sweeps over eight rollout seeds for pretrained and trained models. The training runs total 12,124.4 recorded H800 GPU-hours across all four benchmarks. Shared environments and evaluation procedures provide a common basis for comparison while retaining method-specific training settings informed by the source papers and implementations. We examine not only whether gains hold across settings, but also whether they are sustained during training and justify the additional computation. Our analysis yields three key observations:

(i) Gains over GRPO remain limited in scope, with no method consistently outperforming it across settings.

(ii) Strong final performance does not imply stable training.

(iii) Additional training cost does not consistently translate into better agent performance.

## 2 Related Work

Interactive agents and learning signals. Language agents solve tasks through sequential actions and observations (Shridhar et al., 2021; Yao et al., 2022; Xi et al., 2025). Inference-time methods interleave reasoning and action (Yao et al., 2023) or refine decisions and outputs through verbal feedback (Shinn et al., 2023; Madaan et al., 2023), without parameter updates. Training methods learn from filtered tool-use annotations (Schick et al., 2023) and curated interaction trajectories (Qin et al., 2024; Chen et al., 2023; Zeng et al., 2024). Reward-based optimization learns from sampled behavior (Shao et al., 2024), while online curricula support continued policy improvement (Qi et al., 2025). Multi-turn RL can nevertheless exhibit unstable updates and degenerate behavior, motivating careful reward design and rollout sampling (Wang et al., 2025). Our benchmark focuses on capabilities acquired through parameter updates, rather than improvements obtained by modifying an agent’s inference procedure.

Distillation and privileged supervision. Distillation transfers teacher predictions, generated sequences, or policy behavior to a student (Hinton et al., 2015; Kim and Rush, 2016; Rusu et al., 2016). Training-only privileged information (Lopez-Paz et al., 2016) and reasoning rationales (Hsieh et al., 2023) enrich supervision beyond target labels. Privileged supervision therefore seeks to improve the student’s policy without requiring access to the teacher’s additional information at deployment. Supervising learner-visited states or student-generated sequences helps reduce distribution mismatch (Ross et al., 2011; Lin et al., 2020; Agarwal et al., 2024).

<table><tr><td>Method (source)</td><td>Repeated evaluation</td><td>Training failure analysis</td><td>Measured cost</td></tr><tr><td>GRPO+OPSD (Lu et al., 2026)</td><td>X</td><td>√</td><td>X</td></tr><tr><td>SDAR (Lu et al., 2026)</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Skill-SD (Wang et al., 2026a)</td><td>X</td><td>√</td><td>X</td></tr><tr><td>RLSD (Yang et al., 2026a)</td><td>X</td><td>√</td><td>X</td></tr><tr><td>OCSD (Yang et al., 2026c)</td><td>√</td><td>X</td><td>√</td></tr><tr><td>OVCSD (Xia et al., 2026)</td><td>X</td><td>X</td><td>√</td></tr><tr><td>AgentOPSD (Wang et al., 2026b)</td><td>X</td><td>X</td><td>X</td></tr><tr><td>ADRS (Zhang et al., 2026)</td><td>X</td><td>√</td><td>X</td></tr><tr><td>ICSD (Lan et al., 2026)</td><td>X</td><td>X</td><td>√</td></tr><tr><td>PCSD (Lv et al., 2026)</td><td>X</td><td>X</td><td>×</td></tr><tr><td>Methods with reported evidence</td><td>1/10</td><td>5/ 10</td><td>3 / 10</td></tr></table>

Table 1 Coverage of key evaluation dimensions in prior work.

MiniLLM and DistiLLM study divergence objectives and sampling strategies (Gu et al., 2024; Ko et al., 2024), while iterative self-training learns from selected, self-generated reasoning or interaction traces (Zelikman et al., 2022; Aksitov et al., 2023). Supervision content, sampling distribution, and optimization objective ofer complementary perspectives for comparing distillation-based agent training.

Self-distillation in agentic reinforcement learning. Agentic self-distillation derives teacher supervision from privileged context available only during training. The evaluated self-distillation methods use four mechanisms. Skill-SD, GRPO+OPSD, SDAR, ICSD, and PCSD add auxiliary objectives (Wang et al., 2026a; Lu et al., 2026; Lan et al., 2026; Lv et al., 2026), whereas RLSD, OCSD, and AgentOPSD modulate policy advantages through teacher–student comparisons (Yang et al., 2026a,c; Wang et al., 2026b). ADRS shapes rewards and advantages (Zhang et al., 2026), while OVCSD adds environment-verified teacher continuations (Xia et al., 2026). Recent analyses identify supervision inconsistency and privileged-information bias as potential failure modes (Yang et al., 2026b; Harne et al., 2026). Our benchmark compares these mechanisms’ performance, training stability, and computational cost under a common evaluation protocol. Appendices D.2, D.3, D.5, and D.4 detail the four mechanisms, respectively.

## 3 The Limits of Published Comparisons

Table 1 summarizes how the source papers of the ten compared methods address repeated evaluation, trainingfailure analysis, and measured computational cost. Gaps in this coverage limit assessments of method reliability and practical value beyond reported performance gains.

Repeated evaluation. Performance diferences must be interpreted relative to variation across evaluations. Yet repeated evaluation is rarely reported in the surveyed sources, making it dificult to distinguish consistent improvements from favorable sampling outcomes. Reporting variability alongside mean performance is therefore essential to judging the strength of a claimed advantage.

Training failure analysis. A strong final score does not establish that training is stable. Although some studies investigate collapse or performance degradation, such analyses are not consistently available across competing methods. Examining training trajectories is necessary to reveal regressions and failure risks that final scores can conceal.

Measured cost. Reporting computational cost alongside performance is essential for assessing the eficiency and practical feasibility of competing methods. Measured cost remains sparsely reported, limiting comparisons of the resources required to obtain competing methods’ gains. Assessing performance together with training cost is necessary to determine whether an improvement ofers practical value.

None of the surveyed source papers covers all three dimensions. Our benchmark brings them into a common comparison so that method selection rests on the consistency of measured gains, training stability, and computational cost rather than a favorable final score alone.

## 4 Benchmark Design

Our benchmark examines the value of self-distillation in multi-turn text-agent training, using outcomebased GRPO as the reference. It combines common model–task settings and evaluation procedures with method-specific objectives and hyperparameters informed by the source papers and released implementations.

Tasks and model backbones. ALFWorld (Shridhar et al., 2021) requires agents to complete household tasks through textual actions and observations. We evaluate 140 games, reporting overall and task-specific success rates. ScienceWorld (Wang et al., 2022) requires agents to conduct experiments and solve scientific tasks in a text-based environment. We evaluate 200 tasks across Measure, Test-Cond., Find, Chem-Mix, and Lifespan, reporting task score. WebShop (Yao et al., 2022) requires agents to find and select products that satisfy a shopping goal. We evaluate a fixed set of 128 goals, reporting task success and a graded score for partial goal satisfaction. Search-QA requires retrieval-assisted answers to questions from seven sources. We use a fixed, seed-0 subset: 500 questions each from NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, and MuSiQue, plus all 125 Bamboogle questions. We report exact-match accuracy after answer normalization; Overall gives equal weight to all 3,125 questions rather than to sources or evaluation batches. Combining these four benchmarks with Qwen3.5-2B, 4B, and 9B yields twelve model–task settings.

Methods. GRPO (Shao et al., 2024) is the outcome-only reference. GRPO+OPSD and SDAR (Lu et al., 2026), together with Skill-SD (Wang et al., 2026a), ICSD (Lan et al., 2026), and PCSD (Lv et al., 2026), add auxiliary objectives. RLSD (Yang et al., 2026a), OCSD (Yang et al., 2026c), AgentOPSD (Wang et al., 2026b), and ADRS (Zhang et al., 2026) use privileged signals to shape credit or rewards; OVCSD (Xia et al., 2026) adds verified teacher continuations. All methods are evaluated without privileged teacher context. For Skill-SD and RLSD, we use the skill-bank adaptations; RLSD was originally proposed for verifiable reasoning. GRPO+OPSD follows SDAR’s constant-gate implementation, whose auxiliary gradient is independent of teacher scores. Appendices B, C.2, and D detail method adaptations, teacher contexts, and algorithms, respectively.

Training and evaluation protocol. We train each method in every model–task setting for 100 global steps, yielding 132 runs, with validation every five steps and method-specific hyperparameters guided by the source papers and released implementations. Using the same fixed task sets as validation, we evaluate all eleven trained methods and the pretrained backbones at temperature T = 0.4 over eight rollout seeds, reporting the mean and sample standard deviation. Final-performance averages assign equal weight to all twelve settings, using each benchmark’s reported metric; computational-cost comparisons cover the same settings. Across all four benchmarks, we assess training progress and stability through validation-curve AUC, peak-to-final declines, and low-performing outcomes, with equal weight on the twelve settings in aggregate comparisons. Appendices C.1, C.3, and D.1 specify hyperparameters, numerical settings, and the shared training procedure. Appendices C.4 and C.5 detail the evaluation protocol and experimental records, respectively.

## 5 Comparative Performance under Common Evaluation

The central question is whether additional training mechanisms deliver gains beyond a few favorable settings. We compare each alternative with GRPO across all twelve model–task settings, examining both the frequency and magnitude of improvements. Tables 2 and 3 report the results for all methods and their pretrained backbones under the common evaluation protocol. Appendix F.2 analyzes the discrepancy between step-100 validation and repeated final evaluation on Search-QA.

More methods do not imply broad progress. All ten alternatives improve on GRPO in some settings, but only a minority retain an advantage in aggregate. Only PCSD (54.26, +4.58 points) and AgentOPSD (51.34, +1.66 points) exceed GRPO’s twelve-setting mean of 49.68; the other eight alternatives fall below it. Since all ten alternatives yield gains somewhere, isolated positive results do little to establish progress beyond a strong outcome-only baseline. A broadly applicable training improvement must be assessed across the range of tasks and model sizes it is expected to support.

No method dominates across settings. The leading method depends on both the benchmark and model size. ALFWorld has a diferent leader at each scale—Skill-SD, RLSD, and ICSD at 2B, 4B, and 9B—whereas GRPO+OPSD leads WebShop at all three scales. Search-QA is led by ICSD at 2B and 4B and OCSD at

9B; ScienceWorld’s leaders at 2B, 4B, and 9B are OVCSD, OCSD, and GRPO, respectively. Even PCSD, which has the highest twelve-setting mean, leads no individual setting and beats GRPO in only six, trailing it in five and tying in one. Selecting the best aggregate performer and choosing a winner for a particular setting are therefore diferent method-selection problems. Neither a collection of first-place results nor the highest aggregate score establishes a method as uniformly superior; both must be interpreted alongside its performance across the full set of settings.

Win counts can conceal severe regressions. ICSD outperforms GRPO in seven of twelve settings, yet its mean of 44.75 remains below GRPO’s 49.68. Its 4B WebShop run ends with 0.00% success, illustrating a severe failure that a win count does not distinguish from a small regression. GRPO+OPSD likewise wins six settings but averages only 43.05, with 1.52% success on 4B ALFWorld. Counting each loss equally discards whether a method narrowly trails the baseline or loses nearly all task-solving ability. Severe failures can outweigh gains elsewhere, so assessing progress requires considering the magnitude of losses as well as the frequency of improvements. Appendix E reports the twelve-setting summary and examines low-performance outcomes across all four benchmarks.

Table 2 Results for pretrained models and eleven training methods on ALFWorld and ScienceWorld. Entries report mean over eight rollout seeds at T = 0.4; trained models are evaluated after 100 global steps. ALFWorld reports success (%), and ScienceWorld reports task score. Teal bold and underlining mark the highest and second-highest means within each model and metric, including ties. Overall columns are shaded.
<table><tr><td rowspan=1 colspan=14>ALFWorld: success rate (%)                                    ScienceWorld: task score</td></tr><tr><td rowspan=1 colspan=2>Method       Pick</td><td rowspan=1 colspan=3>Look   Clean    Heat</td><td rowspan=1 colspan=2>Cool    Pick2</td><td rowspan=1 colspan=1>Overall</td><td rowspan=1 colspan=5>Measure Test-Cond.  Find   Chem-Mix  Lifespan</td><td rowspan=1 colspan=1>Overall</td></tr><tr><td rowspan=1 colspan=2>Qwen3.5-2B</td><td rowspan=1 colspan=3></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Pretrained   16.43±2.53</td><td rowspan=1 colspan=1>23.08±5.81</td><td rowspan=1 colspan=1>0.93±1.71</td><td rowspan=1 colspan=1>1.56±2.89</td><td rowspan=1 colspan=1>1.50±2.98</td><td rowspan=1 colspan=1>2.60±3.10</td><td rowspan=1 colspan=1>7.32±1.19</td><td rowspan=1 colspan=4>−10.32±2.822.94±0.75 −7.26±3.44 1.50±7.07</td><td rowspan=1 colspan=1>3.33±0.00</td><td rowspan=1 colspan=1>-2.83±1.31</td></tr><tr><td rowspan=1 colspan=2>GRPO</td><td rowspan=1 colspan=1>64.42+7.05</td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $1 3 . 2 8 _ { \pm 2 . 2 1 }$ </td><td rowspan=1 colspan=1>33.50±5.633</td><td rowspan=1 colspan=1>3.85±3.48</td><td rowspan=1 colspan=1> $2 8 . 4 8 _ { \pm 1 . 2 3 }$ </td><td rowspan=1 colspan=4>0.42±0.00   $0 . 6 0 _ { \pm 0 . 0 0 }$  0.55±0.00   $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=2>GRPO+OPSD $1 4 . 6 4 { \scriptstyle \pm 4 . 6 9 }$ </td><td rowspan=1 colspan=1> $3 1 . 7 3 { \scriptstyle \pm 8 . 6 6 }$ </td><td rowspan=1 colspan=1> $2 7 . 7 8 { \scriptstyle \pm 4 . 4 3 }$ </td><td rowspan=1 colspan=1> $3 7 . 5 0 { \scriptstyle \pm 7 . 4 7 }$ </td><td rowspan=1 colspan=1>33.00±6.68</td><td rowspan=1 colspan=1>0.52±1.47</td><td rowspan=1 colspan=1> $2 2 . 2 3 { \scriptstyle \pm 1 . 7 3 }$ </td><td rowspan=1 colspan=4>0.42±0.00   $0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$   $0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$    $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=2>SDAR       $3 2 . 8 6 _ { \pm 1 . 5 3 }$ </td><td rowspan=1 colspan=1>15.38±0.00</td><td rowspan=1 colspan=1> $0 . 4 6 _ { \pm 1 . 3 1 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>17.50±5.63</td><td rowspan=1 colspan=1>9.38±4.31</td><td rowspan=1 colspan=1> $1 4 . 4 6 _ { \pm 1 . 1 3 }$ </td><td rowspan=1 colspan=4> $0 . 4 2 _ { \pm 0 . 0 0 }$   $0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$   $0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$    $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=2>Skill-SD      $6 9 . 2 9 { \scriptstyle \pm 5 . 0 1 }$ </td><td rowspan=1 colspan=1> $5 9 . 6 2 { \scriptstyle \pm 3 . 5 6 }$ </td><td rowspan=1 colspan=1> $\underline { { 8 5 . 6 5 } } { \scriptstyle \pm 3 . 0 9 }$ </td><td rowspan=1 colspan=1> $9 3 . 7 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>71.50±4.50</td><td rowspan=1 colspan=1> $6 6 . 6 7 { \scriptstyle \pm 3 . 1 5 }$ </td><td rowspan=1 colspan=1>74.29±2.44</td><td rowspan=1 colspan=1>0.42±0.00</td><td rowspan=1 colspan=1> $0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=1>RLSD</td><td rowspan=1 colspan=1> $5 8 . 5 7 { \scriptstyle \pm 3 . 4 1 }$ </td><td rowspan=1 colspan=1>39.42±7.62</td><td rowspan=1 colspan=1> $5 9 . 2 6 { \scriptstyle \pm 3 . 9 6 }$ </td><td rowspan=1 colspan=1> $5 7 . 0 3 { \scriptstyle \pm 5 . 2 2 }$ </td><td rowspan=1 colspan=1>58.00± 7.09</td><td rowspan=1 colspan=1>0.00±0.00</td><td rowspan=1 colspan=1> $4 6 . 6 1 { \scriptstyle \pm 1 . 7 0 }$ </td><td rowspan=1 colspan=1>0.48±0.05</td><td rowspan=1 colspan=1> $0 . 7 4 { \scriptstyle \pm 0 . 0 8 }$ </td><td rowspan=1 colspan=1> $4 . 9 4 _ { \pm 0 . 2 6 }$ </td><td rowspan=1 colspan=1> $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 5 4 { \scriptstyle \pm 0 . 5 9 }$ </td><td rowspan=1 colspan=1>1.89±0.07</td></tr><tr><td rowspan=1 colspan=1>OCSD</td><td rowspan=1 colspan=1>36.79±3.22</td><td rowspan=1 colspan=1> $5 9 . 6 2 { \scriptstyle \pm 5 . 4 4 }$ </td><td rowspan=1 colspan=1> $4 9 . 5 4 { \scriptstyle \pm 3 . 9 3 }$ </td><td rowspan=1 colspan=1> $3 8 . 2 8 { \scriptstyle \pm 2 . 2 1 }$ </td><td rowspan=1 colspan=1>42.50±2.98</td><td rowspan=1 colspan=1> $3 5 . 4 2 _ { \pm 4 . 4 5 }$ </td><td rowspan=1 colspan=1> $4 2 . 3 2 _ { \pm 2 . 1 5 }$ </td><td rowspan=1 colspan=1>7.08±0.04</td><td rowspan=1 colspan=1> $9 . 4 7 _ { \pm 0 . 0 8 }$ </td><td rowspan=1 colspan=1>24.43±0.00</td><td rowspan=1 colspan=1> $2 9 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>25.00±0.00</td><td rowspan=1 colspan=1>13.78±0.03</td></tr><tr><td rowspan=1 colspan=1>OVCSD</td><td rowspan=1 colspan=1> $5 6 . 0 7 { \scriptstyle \pm 5 . 7 0 }$ </td><td rowspan=1 colspan=1> $6 3 . 4 6 { \scriptstyle \pm 7 . 9 6 }$ </td><td rowspan=1 colspan=1> $7 3 . 6 1 { \scriptstyle \pm 2 . 3 7 }$ </td><td rowspan=1 colspan=1> $7 4 . 2 2 { \scriptstyle \pm 5 . 2 2 }$ </td><td rowspan=1 colspan=1>75.00±2.83</td><td rowspan=1 colspan=1> $5 . 7 3 { \scriptstyle \pm 2 . 1 6 }$ </td><td rowspan=1 colspan=1> $5 6 . 9 6 { \scriptstyle \pm 1 . 5 2 }$ </td><td rowspan=1 colspan=1>6.02±0.38</td><td rowspan=1 colspan=1> $\underline { { 8 . 5 2 } } _ { \pm 0 . 2 7 }$ </td><td rowspan=1 colspan=1> $\underline { { 2 2 . 3 1 } } _ { \pm 0 . 4 1 }$ </td><td rowspan=1 colspan=1> $\underline { { 1 5 . 1 5 } } { \scriptstyle \pm 2 . 1 8 }$ </td><td rowspan=1 colspan=1> $4 5 . 8 3 { \scriptstyle \pm 3 . 5 6 }$ </td><td rowspan=1 colspan=1>13.85±0.37</td></tr><tr><td rowspan=1 colspan=1>AgentOPSD</td><td rowspan=1 colspan=1> $\underline { { 7 1 . 4 3 } } _ { \pm 3 . 4 1 }$ </td><td rowspan=1 colspan=1> $5 2 . 8 8 _ { \pm 6 . 4 2 }$ </td><td rowspan=1 colspan=1> $1 . 3 9 { \scriptstyle \pm 1 . 9 2 }$ </td><td rowspan=1 colspan=1> $6 5 . 6 2 _ { \pm 3 . 3 4 }$ </td><td rowspan=1 colspan=1>67.50±3.34</td><td rowspan=1 colspan=1> $2 8 . 6 5 { \scriptstyle \pm 5 . 6 5 }$ </td><td rowspan=1 colspan=1> $4 7 . 5 0 { \scriptstyle \pm 1 . 7 5 }$ </td><td rowspan=1 colspan=1>0.42±0.00</td><td rowspan=1 colspan=1> $0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=1>ADRS</td><td rowspan=1 colspan=1>31.43±3.05</td><td rowspan=1 colspan=1> $3 2 . 6 9 { \scriptstyle \pm 5 . 4 4 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>1.50±2.98</td><td rowspan=1 colspan=1> $1 0 . 9 4 _ { \pm 4 . 4 2 }$ </td><td rowspan=1 colspan=1> $1 3 . 0 4 { \scriptstyle \pm 1 . 6 5 }$ </td><td rowspan=1 colspan=1> $0 . 4 2 _ { \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=1>ICSD</td><td rowspan=1 colspan=1> $6 0 . 0 0 { \scriptstyle \pm 3 . 0 5 }$ </td><td rowspan=1 colspan=1> $5 3 . 8 5 { \scriptstyle \pm 8 . 2 2 }$ </td><td rowspan=1 colspan=1> $6 2 . 9 6 { \scriptstyle \pm 3 . 9 6 }$ </td><td rowspan=1 colspan=1> $5 5 . 4 7 { \scriptstyle \pm 6 . 1 9 }$ </td><td rowspan=1 colspan=1>51.50±3.96</td><td rowspan=1 colspan=1> $\underline { { 4 1 . 1 5 } } _ { \pm 4 . 1 3 }$ </td><td rowspan=1 colspan=1> $5 4 . 7 3 { \scriptstyle \pm 2 . 0 5 }$ </td><td rowspan=1 colspan=1>0.42±0.00</td><td rowspan=1 colspan=1> $0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=1>PCSD</td><td rowspan=1 colspan=1>72.86±2.16</td><td rowspan=1 colspan=1> $7 3 . 0 8 _ { \pm 4 . 1 1 }$ </td><td rowspan=1 colspan=1> $8 9 . 3 5 { \scriptstyle \pm 2 . 3 7 }$ </td><td rowspan=1 colspan=1> $\underline { { 7 5 . 7 8 } } _ { \pm 2 . 2 1 }$ </td><td rowspan=1 colspan=1>56.00±3.02</td><td rowspan=1 colspan=1> $6 6 . 6 7 _ { \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $\underline { { 7 2 . 3 2 } } _ { \pm 0 . 9 2 }$ </td><td rowspan=1 colspan=1>0.42±0.00</td><td rowspan=1 colspan=1> $0 . 6 0 _ { \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $0 . 5 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>0.83±0.00</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-4B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Pretrained</td><td rowspan=1 colspan=1>68.57±3.05</td><td rowspan=1 colspan=1> $4 5 . 1 9 _ { \pm 4 . 9 3 }$ </td><td rowspan=1 colspan=1> $2 0 . 8 3 _ { \pm 5 . 9 2 }$ </td><td rowspan=1 colspan=1> $3 . 1 2 _ { \pm 3 . 3 4 }$ </td><td rowspan=1 colspan=1>12.00±4.78</td><td rowspan=1 colspan=1> $3 6 . 4 6 _ { \pm 4 . 3 1 }$ </td><td rowspan=1 colspan=1>34.11±1.82</td><td rowspan=1 colspan=1>−20.88±10.62</td><td rowspan=1 colspan=1>30.44±3.86</td><td rowspan=1 colspan=1>−30.62±3.47 −</td><td rowspan=1 colspan=1>6.35±14.94</td><td rowspan=1 colspan=1> $1 5 . 8 3 _ { \pm 4 . 4 5 }$ </td><td rowspan=1 colspan=1>1.39±3.20</td></tr><tr><td rowspan=1 colspan=1>GRPO</td><td rowspan=1 colspan=1> $8 6 . 0 7 _ { \pm 2 . 3 8 }$ </td><td rowspan=1 colspan=1>74.04±3.98</td><td rowspan=1 colspan=1> $9 3 . 9 8 _ { \pm 1 . 9 2 }$ </td><td rowspan=1 colspan=1> $7 9 . 6 9 { \scriptstyle \pm 2 . 8 9 }$ </td><td rowspan=1 colspan=1>87.50±1.41</td><td rowspan=1 colspan=1>63.02±3.48</td><td rowspan=1 colspan=1> $8 2 . 0 5 _ { \pm 1 . 5 0 }$ </td><td rowspan=1 colspan=1>6.60±0.33</td><td rowspan=1 colspan=1>54.80±0.71</td><td rowspan=1 colspan=1>24.43±0.14</td><td rowspan=1 colspan=1>16.57±1.20</td><td rowspan=1 colspan=1>25.00±0.00</td><td rowspan=1 colspan=1>32.16±0.30</td></tr><tr><td rowspan=1 colspan=1>GRPO+OPSD 0.</td><td rowspan=1 colspan=1>00±0.00</td><td rowspan=1 colspan=1>13.46±5.44</td><td rowspan=1 colspan=1> $0 . 4 6 _ { \pm 1 . 3 1 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>1.00±1.85</td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>1.52±0.71</td><td rowspan=1 colspan=1>6.91±0.16</td><td rowspan=1 colspan=1> $9 . 3 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td rowspan=1 colspan=1> $2 4 . 3 6 { \scriptstyle \pm 0 . 0 9 }$ </td><td rowspan=1 colspan=1>28.50±1.41</td><td rowspan=1 colspan=1> $2 5 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>13.65±0.06</td></tr><tr><td rowspan=1 colspan=1>SDAR</td><td rowspan=1 colspan=1> $8 9 . 6 4 { \scriptstyle \pm 1 . 4 8 }$ </td><td rowspan=1 colspan=1> $4 7 . 1 2 { \scriptstyle \pm 7 . 6 2 }$ </td><td rowspan=1 colspan=1> $7 8 . 7 0 { \scriptstyle \pm 7 . 3 4 }$ </td><td rowspan=1 colspan=1> $5 2 . 3 4 { \scriptstyle \pm 6 . 6 3 }$ </td><td rowspan=1 colspan=1>74.50±4.24</td><td rowspan=1 colspan=1> $\underline { { 8 6 . 9 8 } } _ { \pm 4 . 1 3 }$ </td><td rowspan=1 colspan=1> $7 6 . 1 6 { \scriptstyle \pm 2 . 2 9 }$ </td><td rowspan=1 colspan=1>7.09±0.00</td><td rowspan=1 colspan=1>9.64±0.00</td><td rowspan=1 colspan=1>24.43±0.00</td><td rowspan=1 colspan=1> $2 9 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $2 5 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>13.86±0.00</td></tr><tr><td rowspan=1 colspan=1>Skill-SD</td><td rowspan=1 colspan=1>81.79±4.57 4</td><td rowspan=1 colspan=1>3.27±11.58</td><td rowspan=1 colspan=1>56.94±7.12</td><td rowspan=1 colspan=1>39.84±5.73</td><td rowspan=1 colspan=1>42.00±6.76</td><td rowspan=1 colspan=1>45.31±6.84</td><td rowspan=1 colspan=1>55.27±2.59</td><td rowspan=1 colspan=1>24.38±0.55</td><td rowspan=1 colspan=1>56.66±0.46</td><td rowspan=1 colspan=1>41.98±0.27</td><td rowspan=1 colspan=1>23.35±1.83</td><td rowspan=1 colspan=1>25.00±0.00</td><td rowspan=1 colspan=1>41.67±0.17</td></tr><tr><td rowspan=1 colspan=1>RLSD</td><td rowspan=1 colspan=1> $\underline { { 9 4 . 6 4 } } _ { \pm 2 . 3 8 }$ </td><td rowspan=1 colspan=1> $8 0 . 7 7 _ { \pm 4 . 1 1 }$ </td><td rowspan=1 colspan=1> $\underline { { 9 3 . 5 2 } } _ { \pm 3 . 2 8 }$ </td><td rowspan=1 colspan=1> $\underline { { 7 7 . 3 4 } } _ { \pm 1 0 . 5 3 }$ </td><td rowspan=1 colspan=1>70.50±4.75</td><td rowspan=1 colspan=1>80.21±2.95</td><td rowspan=1 colspan=1>84.38±1.29</td><td rowspan=1 colspan=1> $1 0 . 0 7 _ { \pm 0 . 0 3 }$ </td><td rowspan=1 colspan=1> $9 . 4 6 _ { \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>52.82±0.00</td><td rowspan=1 colspan=1> $1 7 . 8 5 _ { \pm 1 . 0 4 }$ </td><td rowspan=1 colspan=1> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>22.41±0.03</td></tr><tr><td rowspan=1 colspan=1>OCSD</td><td rowspan=1 colspan=1> $7 5 . 7 1 { \scriptstyle \pm 3 . 0 5 }$ </td><td rowspan=1 colspan=1> $3 2 . 6 9 { \scriptstyle \pm 7 . 9 6 }$ </td><td rowspan=1 colspan=1>47.22±3.83</td><td rowspan=1 colspan=1>47.66±8.80</td><td rowspan=1 colspan=1>41.50±6.024</td><td rowspan=1 colspan=1>8.96±3.69</td><td rowspan=1 colspan=1> $5 2 . 3 2 { \scriptstyle \pm 2 . 4 7 }$ </td><td rowspan=1 colspan=1> $\underline { { 2 4 . 2 4 } } _ { \pm 0 . 0 8 }$ </td><td rowspan=1 colspan=1> $5 7 . 2 4 { \scriptstyle \pm 0 . 2 5 }$ </td><td rowspan=1 colspan=1>43.16±0.94</td><td rowspan=1 colspan=1> $1 4 . 1 0 { \scriptstyle \pm 0 . 9 3 }$ </td><td rowspan=1 colspan=1> $2 3 . 3 3 { \scriptstyle \pm 1 . 5 4 }$ </td><td rowspan=1 colspan=1>41.77±0.34</td></tr><tr><td rowspan=1 colspan=1>OVCSD</td><td rowspan=1 colspan=1> $8 7 . 5 0 { \scriptstyle \pm 3 . 3 9 }$ </td><td rowspan=1 colspan=1> $6 2 . 5 0 { \scriptstyle \pm 8 . 6 6 }$ </td><td rowspan=1 colspan=1> $7 9 . 6 3 { \scriptstyle \pm 3 . 9 6 }$ </td><td rowspan=1 colspan=1> $6 9 . 5 3 { \scriptstyle \pm 5 . 2 2 }$ </td><td rowspan=1 colspan=1>66.50±5.63</td><td rowspan=1 colspan=1> $7 9 . 1 7 { \scriptstyle \pm 4 . 9 8 }$ </td><td rowspan=1 colspan=1> $7 6 . 4 3 { \scriptstyle \pm 2 . 1 9 }$ </td><td rowspan=1 colspan=1> $1 4 . 8 0 { \scriptstyle \pm 2 . 8 3 }$ </td><td rowspan=1 colspan=1> $1 9 . 5 4 { \scriptstyle \pm 2 . 3 1 }$ </td><td rowspan=1 colspan=1> $7 . 9 9 { \scriptstyle \pm 1 . 4 1 }$ </td><td rowspan=1 colspan=1> $1 0 . 5 0 { \scriptstyle \pm 5 . 6 3 }$ </td><td rowspan=1 colspan=1> $1 1 . 6 7 { \scriptstyle \pm 2 . 5 2 }$ </td><td rowspan=1 colspan=1>14.93±1.57</td></tr><tr><td rowspan=1 colspan=1>AgentOPSD</td><td rowspan=1 colspan=1> $7 4 . 6 4 { \scriptstyle \pm 4 . 1 6 }$ </td><td rowspan=1 colspan=1> $5 1 . 9 2 { \scriptstyle \pm 5 . 4 4 }$ </td><td rowspan=1 colspan=1> $8 7 . 9 6 { \scriptstyle \pm 4 . 3 1 }$ </td><td rowspan=1 colspan=1> $7 5 . 7 8 { \scriptstyle \pm 7 . 0 4 }$ </td><td rowspan=1 colspan=1>81.00±2.83</td><td rowspan=1 colspan=1> $6 6 . 6 7 { \scriptstyle \pm 7 . 0 4 }$ </td><td rowspan=1 colspan=1> $7 5 . 0 0 { \scriptstyle \pm 2 . 1 9 }$ </td><td rowspan=1 colspan=1> $5 . 5 0 { \scriptstyle \pm 3 . 5 4 }$ </td><td rowspan=1 colspan=1> $5 5 . 3 4 { \scriptstyle \pm 0 . 8 9 }$ </td><td rowspan=1 colspan=1>25.51±0.86</td><td rowspan=1 colspan=1> $2 5 . 1 0 { \scriptstyle \pm 3 . 3 0 }$ </td><td rowspan=1 colspan=1> $2 8 . 5 4 { \scriptstyle \pm 2 . 5 9 }$ </td><td rowspan=1 colspan=1>32.80±0.98</td></tr><tr><td rowspan=1 colspan=1>ADRS</td><td rowspan=1 colspan=1> $8 5 . 3 6 { \scriptstyle \pm 5 . 3 9 }$ </td><td rowspan=1 colspan=1> $6 4 . 4 2 { \scriptstyle \pm 3 . 9 8 }$ </td><td rowspan=1 colspan=1> $6 7 . 1 3 { \scriptstyle \pm 5 . 0 2 }$ </td><td rowspan=1 colspan=1> $6 7 . 1 9 { \scriptstyle \pm 7 . 2 8 }$ </td><td rowspan=1 colspan=1>65.50±6.39</td><td rowspan=1 colspan=1> $6 4 . 5 8 { \scriptstyle \pm 8 . 0 3 }$ </td><td rowspan=1 colspan=1> $7 0 . 7 1 { \scriptstyle \pm 3 . 1 0 }$ </td><td rowspan=1 colspan=1> $7 . 7 1 \pm 0 . 2 6$ </td><td rowspan=1 colspan=1> $5 5 . 8 8 { \scriptstyle \pm 0 . 9 3 }$ </td><td rowspan=1 colspan=1> $4 1 . 3 8 { \scriptstyle \pm 2 . 1 5 }$ </td><td rowspan=1 colspan=1> $2 3 . 3 8 { \scriptstyle \pm 3 . 0 1 }$ </td><td rowspan=1 colspan=1> $2 4 . 1 7 { \scriptstyle \pm 0 . 8 9 }$ </td><td rowspan=1 colspan=1> $3 6 . 7 3 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td rowspan=1 colspan=1>ICSD</td><td rowspan=1 colspan=1> $\underline { { 9 4 . 6 4 } } _ { \pm 3 . 2 2 }$ </td><td rowspan=1 colspan=1> $7 0 . 1 9 _ { \pm 2 . 7 2 }$ </td><td rowspan=1 colspan=1> $9 0 . 7 4 _ { \pm 3 . 4 3 }$ </td><td rowspan=1 colspan=1> $6 4 . 8 4 { \scriptstyle \pm 5 . 7 3 }$ </td><td rowspan=1 colspan=1> $7 5 . 5 0 _ { \pm 1 . 4 1 }$ </td><td rowspan=1 colspan=1> $8 1 . 2 5 _ { \pm 3 . 8 6 }$ </td><td rowspan=1 colspan=1> $\underline { { 8 2 . 5 0 } } _ { \pm 1 . 6 2 }$ </td><td rowspan=1 colspan=1> $7 . 6 8 _ { \pm 0 . 1 1 }$ </td><td rowspan=1 colspan=1> $9 . 8 7 _ { \pm 0 . 1 3 }$ </td><td rowspan=1 colspan=1> $2 4 . 9 1 _ { \pm 0 . 1 4 }$ </td><td rowspan=1 colspan=1> $\underline { { 2 8 . 7 5 } } \pm 0 . 7 1$ </td><td rowspan=1 colspan=1> $\underline { { 4 8 . 7 5 } } _ { \pm 1 . 7 3 }$ </td><td rowspan=1 colspan=1> $1 5 . 9 9 _ { \pm 0 . 1 9 }$ </td></tr><tr><td rowspan=1 colspan=1>PCSD</td><td rowspan=1 colspan=1> $9 6 . 7 9 { \scriptstyle \pm 1 . 8 3 }$ </td><td rowspan=1 colspan=1> $4 8 . 0 8 { \scriptstyle \pm 6 . 8 2 }$ </td><td rowspan=1 colspan=1> $7 7 . 7 8 _ { \pm 3 . 4 3 }$ </td><td rowspan=1 colspan=1> $6 9 . 5 3 { \scriptstyle \pm 7 . 7 9 }$ </td><td rowspan=1 colspan=1>73.50±2.98</td><td rowspan=1 colspan=1> $8 8 . 0 2 _ { \pm 4 . 1 3 }$ </td><td rowspan=1 colspan=1> $7 9 . 8 2 _ { \pm 1 . 8 6 }$ </td><td rowspan=1 colspan=1>6.80±0.15</td><td rowspan=1 colspan=1> $3 8 . 7 9 { \scriptstyle \pm 0 . 6 1 }$ </td><td rowspan=1 colspan=1> $1 7 . 1 4 { \scriptstyle \pm 2 . 4 9 }$ </td><td></td><td rowspan=1 colspan=1> $2 3 . 3 3 { \scriptstyle \pm 2 . 8 2 }$ </td><td rowspan=1 colspan=1> $2 3 . 8 5 { \scriptstyle \pm 0 . 5 8 }$ </td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Pretrained</td><td rowspan=1 colspan=1> $7 1 . 0 7 _ { \pm 3 . 2 2 }$ </td><td rowspan=1 colspan=1> $4 1 . 3 5 { \scriptstyle \pm 7 . 0 5 }$ </td><td rowspan=1 colspan=1> $2 0 . 8 3 _ { \pm 7 . 6 5 }$ </td><td rowspan=1 colspan=1> $8 . 5 9 _ { \pm 6 . 6 3 }$ </td><td rowspan=1 colspan=1>4.50±3.96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-12.74±2.80</td><td rowspan=1 colspan=1>29.29±3.01</td><td rowspan=1 colspan=1>−25.52±4.79</td><td rowspan=1 colspan=1> $1 3 . 8 5 _ { \pm 0 . 7 1 }$ </td><td rowspan=1 colspan=1>-22.08±15.29</td><td rowspan=1 colspan=1> $1 . 8 6 _ { \pm 1 . 2 8 }$ </td></tr><tr><td rowspan=1 colspan=1>GRPO</td><td rowspan=1 colspan=1> $9 2 . 5 0 { \scriptstyle \pm 2 . 1 3 }$ </td><td rowspan=1 colspan=1> $7 6 . 9 2 { \scriptstyle \pm 8 . 2 2 }$ </td><td rowspan=1 colspan=1> $4 6 . 3 0 { \scriptstyle \pm 5 . 9 4 }$ </td><td rowspan=1 colspan=1> $5 9 . 3 8 { \scriptstyle \pm 5 . 7 9 }$ </td><td rowspan=1 colspan=1>18.50±2.98</td><td rowspan=1 colspan=1> $7 8 . 1 2 { \scriptstyle \pm 5 . 7 9 }$ </td><td rowspan=1 colspan=1> $6 2 . 6 8 { \scriptstyle \pm 1 . 1 9 }$ </td><td rowspan=1 colspan=1>33.70±0.63</td><td rowspan=1 colspan=1> $6 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>76.78±0.49</td><td rowspan=1 colspan=1>30.60±0.00</td><td rowspan=1 colspan=1> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>58.99±0.27</td></tr><tr><td rowspan=1 colspan=1>GRPO+OPSD</td><td rowspan=1 colspan=1> $9 3 . 2 1 { \scriptstyle \pm 3 . 0 3 }$ </td><td rowspan=1 colspan=1> $6 6 . 3 5 { \scriptstyle \pm 7 . 0 5 }$ </td><td rowspan=1 colspan=1> $8 3 . 3 3 { \scriptstyle \pm 4 . 8 5 }$ </td><td rowspan=1 colspan=1> $4 9 . 2 2 _ { \pm 8 . 4 8 }$ </td><td rowspan=1 colspan=1> $3 8 . 5 0 { \scriptstyle \pm 4 . 7 5 }$ </td><td rowspan=1 colspan=1> $8 0 . 2 1 \pm 6 . 2 0 $ </td><td rowspan=1 colspan=1> $7 1 . 7 9 { \scriptstyle \pm 2 . 5 3 }$ </td><td rowspan=1 colspan=1> $\underline { { 3 9 . 8 4 } } _ { \pm 0 . 0 3 }$ </td><td rowspan=1 colspan=1> $5 8 . 6 7 { \scriptstyle \pm 0 . 6 3 }$ </td><td rowspan=1 colspan=1> $4 0 . 3 4 { \scriptstyle \pm 1 . 1 0 }$ </td><td rowspan=1 colspan=1> $2 8 . 0 0 { \scriptstyle \pm 1 . 8 5 }$ </td><td rowspan=1 colspan=1> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $5 1 . 9 8 { \scriptstyle \pm 0 . 3 8 }$ </td></tr><tr><td rowspan=1 colspan=1>SDAR</td><td rowspan=1 colspan=1> $9 8 . 9 3 { \scriptstyle \pm 1 . 4 8 }$ </td><td rowspan=1 colspan=1> $\underline { { 8 9 . 4 2 } } _ { \pm 9 . 1 4 }$ </td><td rowspan=1 colspan=1> $6 9 . 9 1 { \scriptstyle \pm 5 . 0 2 }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $8 2 . 2 9 { \scriptstyle \pm 3 . 6 9 }$ </td><td rowspan=1 colspan=1> $7 5 . 6 2 { \scriptstyle \pm 2 . 0 4 }$ </td><td rowspan=1 colspan=1> $2 6 . 0 1 { \scriptstyle \pm 3 . 0 6 }$ </td><td rowspan=1 colspan=1> $5 9 . 9 9 _ { \pm 0 . 0 2 }$ </td><td rowspan=1 colspan=1> $8 1 . 5 5 { \scriptstyle \pm 1 . 1 6 }$ </td><td rowspan=1 colspan=1> $3 0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $5 8 . 0 0 { \scriptstyle \pm 0 . 8 8 }$ </td></tr><tr><td rowspan=1 colspan=1>Skill-SD</td><td rowspan=1 colspan=1> $9 3 . 9 3 _ { \pm 3 . 2 2 }$ </td><td rowspan=1 colspan=1> $6 7 . 3 1 { \scriptstyle \pm 6 . 8 2 }$ </td><td rowspan=1 colspan=1> $7 0 . 3 7 { \scriptstyle \pm 4 . 4 3 }$ </td><td rowspan=1 colspan=1> $5 4 . 6 9 { \scriptstyle \pm 4 . 4 2 }$ </td><td rowspan=1 colspan=1>61.00±4.66</td><td rowspan=1 colspan=1> $8 2 . 2 9 { \scriptstyle \pm 4 . 8 5 }$ </td><td rowspan=1 colspan=1>74.55±1.08</td><td rowspan=1 colspan=1> $6 . 7 2 _ { \pm 0 . 3 1 }$ </td><td rowspan=1 colspan=1> $4 0 . 6 5 { \scriptstyle \pm 1 . 3 3 }$ </td><td rowspan=1 colspan=1> $3 0 . 2 0 { \scriptstyle \pm 1 . 1 4 }$ </td><td rowspan=1 colspan=1> $2 8 . 9 0 { \scriptstyle \pm 1 . 7 3 }$ </td><td rowspan=1 colspan=1> $4 0 . 8 3 { \scriptstyle \pm 2 . 5 2 }$ </td><td rowspan=1 colspan=1> $2 9 . 0 8 _ { \pm 0 . 3 9 }$ </td></tr><tr><td rowspan=1 colspan=1>RLSD</td><td rowspan=1 colspan=1> $9 7 . 1 4 { \scriptstyle \pm 2 . 1 6 }$ </td><td rowspan=1 colspan=1> $8 5 . 5 8 { \scriptstyle \pm 4 . 9 3 }$ </td><td rowspan=1 colspan=1> $\underline { { 9 3 . 9 8 } } _ { \pm 2 . 7 6 }$ </td><td rowspan=1 colspan=1> $7 1 . 8 8 { \scriptstyle \pm 4 . 7 2 }$ </td><td rowspan=1 colspan=1> $8 7 . 0 0 { \scriptstyle \pm 1 . 8 5 }$ </td><td rowspan=1 colspan=1> $8 8 . 0 2 { \scriptstyle \pm 2 . 6 7 }$ </td><td rowspan=1 colspan=1> $\frac { 8 9 . 2 0 } { - 2 0 } { \pm 1 . 4 0 }$ </td><td rowspan=1 colspan=1> $3 1 . 0 7 { \scriptstyle \pm 0 . 7 4 }$ </td><td rowspan=1 colspan=1> $5 7 . 2 9 _ { \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>71.01±0.56</td><td rowspan=1 colspan=1> $3 0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>25.00±0.00</td><td rowspan=1 colspan=1>50.27±0.27</td></tr><tr><td rowspan=1 colspan=1>OCSD</td><td rowspan=1 colspan=1> $9 8 . 2 1 { \scriptstyle \pm 2 . 1 3 }$ </td><td rowspan=1 colspan=1> $7 6 . 9 2 _ { \pm 7 . 1 2 }$ </td><td rowspan=1 colspan=1> $8 4 . 7 2 { \scriptstyle \pm 3 . 6 7 }$ </td><td rowspan=1 colspan=1>71.09±8.80</td><td rowspan=1 colspan=1>83.00±5.13</td><td rowspan=1 colspan=1> $\underline { { 8 8 . 5 4 } } _ { \pm 6 . 2 0 }$ </td><td rowspan=1 colspan=1>86.16±1.75</td><td rowspan=1 colspan=1>−10.32±2.11</td><td rowspan=1 colspan=1> $6 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $6 3 . 2 2 { \scriptstyle \pm 2 . 5 3 }$ </td><td rowspan=1 colspan=1> $\underline { { 2 9 . 0 0 } } _ { \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>44.30±0.86</td></tr><tr><td rowspan=1 colspan=1>OVCSD</td><td rowspan=1 colspan=1> $9 7 . 5 0 { \scriptstyle \pm 2 . 8 3 }$ </td><td rowspan=1 colspan=1> $7 7 . 8 8 { \scriptstyle \pm 4 . 9 3 }$ </td><td rowspan=1 colspan=1> $7 6 . 8 5 { \scriptstyle \pm 2 . 6 2 }$ </td><td rowspan=1 colspan=1>69.53±5.22</td><td rowspan=1 colspan=2>81.50±4.75 $8 4 . 3 8 { \scriptstyle \pm 2 . 9 5 }$ </td><td rowspan=1 colspan=1>83.39±1.36</td><td rowspan=1 colspan=1>−15.07±2.73</td><td rowspan=1 colspan=1> $5 9 . 4 7 { \scriptstyle \pm 0 . 0 2 }$ </td><td rowspan=1 colspan=1> $6 5 . 9 2 _ { \pm 0 . 7 1 }$ </td><td rowspan=1 colspan=1> $3 0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>43.45±0.80</td></tr><tr><td rowspan=1 colspan=1>AgentOPSD</td><td rowspan=1 colspan=1>98.57±1.53</td><td rowspan=1 colspan=1>67.31±6.82 6</td><td rowspan=1 colspan=1>4.81±5.94</td><td rowspan=1 colspan=1>75.00±7.47</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>78.04±3.32</td><td rowspan=1 colspan=1>−29.13±2.73</td><td rowspan=1 colspan=1>59.99±0.02</td><td rowspan=1 colspan=1> $\underline { { 7 8 . 8 3 } } \pm 0 . 8 4$ </td><td rowspan=1 colspan=1>18.40±0.57 1</td><td rowspan=1 colspan=1>00.00±0.00</td><td rowspan=1 colspan=1>42.48±0.78</td></tr><tr><td rowspan=1 colspan=1>ADRS</td><td rowspan=1 colspan=1> $8 2 . 5 0 { \scriptstyle \pm 2 . 8 3 }$ </td><td rowspan=1 colspan=1> $7 1 . 1 5 { \scriptstyle \pm 5 . 4 4 }$ </td><td rowspan=1 colspan=1> $9 1 . 6 7 { \scriptstyle \pm 3 . 2 8 }$ </td><td rowspan=1 colspan=1>69.53±4.01</td><td rowspan=1 colspan=1>76.00±6.41</td><td rowspan=1 colspan=1> $7 2 . 9 2 _ { \pm 3 . 8 6 }$ </td><td rowspan=1 colspan=1>78.93±1.15</td><td rowspan=1 colspan=1>−45.33±4.90</td><td rowspan=1 colspan=1> $6 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $6 5 . 8 0 { \scriptstyle \pm 1 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 0 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>35.63±1.18</td></tr><tr><td rowspan=1 colspan=1>ICSD</td><td rowspan=1 colspan=1> $9 5 . 0 0 { \scriptstyle \pm 2 . 0 2 }$ </td><td rowspan=1 colspan=1>86.54±9.86</td><td rowspan=1 colspan=1> $9 8 . 1 5 { \scriptstyle \pm 2 . 8 0 }$ </td><td rowspan=1 colspan=1> $9 0 . 6 2 \substack { \pm 7 . 4 7 }$ </td><td rowspan=1 colspan=1>73.50±4.24</td><td rowspan=1 colspan=1> $9 0 . 1 0 { \scriptstyle \pm 4 . 4 2 }$ </td><td rowspan=1 colspan=1>89.64±2.62</td><td rowspan=1 colspan=1>−18.86±2.09</td><td rowspan=1 colspan=1> $2 8 . 4 5 { \scriptstyle \pm 0 . 3 1 }$ </td><td rowspan=1 colspan=1>31.22±0.39</td><td rowspan=1 colspan=1> $2 7 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $5 2 . 0 8 { \scriptstyle \pm 1 . 7 3 }$ </td><td rowspan=1 colspan=1> $1 8 . 2 6 { \scriptstyle \pm 0 . 5 9 }$ </td></tr><tr><td rowspan=1 colspan=2>PCSD       $9 5 . 3 6 { \scriptstyle \pm 4 . 0 2 }$ </td><td rowspan=1 colspan=1>90.38±3.56 8</td><td rowspan=1 colspan=1>2.41±5.14</td><td rowspan=1 colspan=1> $5 5 . 4 7 { \scriptstyle \pm 7 . 0 4 }$ </td><td rowspan=1 colspan=1>29.50±5.21</td><td rowspan=1 colspan=1> $7 6 . 0 4 \pm 6 . 2 0$ </td><td rowspan=1 colspan=1> $7 2 . 7 7 { \scriptstyle \pm 1 . 7 7 }$ </td><td rowspan=1 colspan=1>51.54±0.12</td><td rowspan=1 colspan=1> $6 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $5 7 . 7 3 { \scriptstyle \pm 0 . 1 3 }$ </td><td rowspan=1 colspan=1> $2 8 . 6 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1>88.33±4.71</td><td rowspan=1 colspan=1> $5 8 . 6 0 _ { \pm 0 . 3 8 }$ </td></tr></table>

## 6 Training Progress and Stability

Final performance does not establish whether learning was sustained or earlier gains were retained. Figure 2 presents all 132 runs across twelve settings, measured every five steps for 2,772 validation sweeps. We summarize all four benchmarks using the area under each validation curve, normalized by the training-step horizon and averaged equally across the twelve settings. Intermediate declines and peak-to-final gaps reveal

$$
9 . 5 3 { \scriptstyle \pm 0 . 8 3 }
$$

$$
2 4 . 2 0 { \scriptstyle \pm 0 . 9 5 }
$$

$$
1 3 . 0 0 { \scriptstyle \pm 0 . 9 6 }
$$

$$
3 7 . 9 3 { \scriptstyle \pm 0 . 4 0 }
$$

$$
1 2 . 7 8 { \scriptstyle \pm 1 . 0 0 }
$$

$$
4 7 . 9 3 { \scriptstyle \pm 0 . 7 1 }
$$

$$
2 3 . 7 0 { \scriptstyle \pm 1 . 4 6 }
$$

$$
2 7 . 1 0 { \scriptstyle \pm 0 . 6 1 }
$$

$$
2 . 4 3 { \scriptstyle \pm 0 . 5 7 }
$$

$$
1 2 . 3 0 { \scriptstyle \pm 1 . 4 8 }
$$

$$
1 4 . 1 9 _ { \pm 0 . 4 3 }
$$

$$
1 . 8 6 _ { \pm 0 . 7 2 }
$$

$$
6 . 0 4 { \scriptstyle \pm 1 . 3 1 }
$$

$$
7 . 1 0 { \scriptstyle \pm 0 . 3 2 }
$$

$$
4 8 . 0 0 { \scriptstyle \pm 0 . 7 2 }
$$

$$
3 1 . 7 8 { \scriptstyle \pm 0 . 8 4 }
$$

$$
8 5 . 0 1 { \scriptstyle \pm 1 . 2 2 }
$$

$$
1 1 . 4 0 { \scriptstyle \pm 0 . 9 3 }
$$

$$
3 2 . 3 8 { \scriptstyle \pm 0 . 2 8 }
$$

$$
7 0 . 8 0 { \scriptstyle \pm 0 . 9 3 }
$$

$$
3 9 . 6 8 { \scriptstyle \pm 0 . 5 9 }
$$

$$
2 6 . 2 3 { \scriptstyle \pm 0 . 6 5 }
$$

$$
6 . 3 8 { \scriptstyle \pm 0 . 5 2 }
$$

$$
1 1 . 2 0 { \scriptstyle \pm 0 . 7 4 }
$$

$$
3 4 . 2 2 _ { \pm 0 . 2 4 }
$$

$$
8 6 . 7 6 { \scriptstyle \pm 0 . 7 5 }
$$

$$
3 8 . 7 5 { \scriptstyle \pm 0 . 6 6 }
$$

$$
4 8 . 0 0 { \scriptstyle \pm 0 . 3 9 }
$$

$$
3 9 . 1 5 { \scriptstyle \pm 1 . 2 8 }
$$

$$
\mathbf { 1 6 . 0 5 _ { \pm 0 . 8 1 } }
$$

$$
\underline { { 3 2 . 3 0 } } _ { \pm 2 . 0 9 }
$$

$$
8 0 . 2 1 { \scriptstyle \pm 1 . 1 1 }
$$

$$
4 8 . 2 5 { \scriptstyle \pm 0 . 3 3 }
$$

$$
1 9 . 8 0 { \scriptstyle \pm 1 . 9 0 }
$$

$$
4 0 . 0 3 { \scriptstyle \pm 1 . 4 0 }
$$

$$
\underline { { 6 0 . 3 0 } } _ { \pm 1 . 1 2 }
$$

$$
4 3 . 7 0 { \scriptstyle \pm 1 . 0 7 }
$$

$$
\underline { { 3 9 . 9 3 } } _ { \pm 0 . 8 7 }
$$

$$
3 0 . 3 8 { \scriptstyle \pm 1 . 3 5 }
$$

$$
1 1 . 5 8 { \scriptstyle \pm 0 . 5 6 }
$$

$$
2 0 . 6 3 { \scriptstyle \pm 0 . 8 9 }
$$

$$
1 3 . 9 8 _ { \pm 0 . 5 7 }
$$

$$
8 . 4 5 { \scriptstyle \pm 0 . 3 2 }
$$

$$
1 2 . 6 3 { \scriptstyle \pm 0 . 2 3 }
$$

$$
{ \bf 3 5 . 0 0 } { \scriptstyle \pm 1 . 9 5 }
$$

$$
3 7 . 9 3 _ { \pm 0 . 1 3 }
$$

$$
2 4 . 7 3 { \scriptstyle \pm 0 . 3 7 }
$$

$$
7 9 . 8 6 _ { \pm 1 . 4 2 }
$$

$$
2 . 3 0 { \scriptstyle \pm 0 . 2 6 }
$$

$$
4 0 . 9 8 { \scriptstyle \pm 1 . 2 5 }
$$

$$
6 9 . 7 3 { \scriptstyle \pm 1 . 4 9 }
$$

$$
5 1 . 6 8 { \scriptstyle \pm 0 . 7 2 }
$$

$$
3 4 . 3 3 { \scriptstyle \pm 1 . 2 5 }
$$

$$
2 . 9 0 { \scriptstyle \pm 0 . 6 0 }
$$

$$
2 9 . 6 8 { \scriptstyle \pm 0 . 8 2 }
$$

$$
1 2 . 9 6 _ { \pm 0 . 1 5 }
$$

$$
1 1 . 4 0 { \scriptstyle \pm 0 . 8 9 }
$$

$$
\underline { { 8 6 . 5 9 } } { \scriptstyle \pm 1 . 0 1 }
$$

$$
2 6 . 7 0 { \scriptstyle \pm 1 . 7 1 }
$$

$$
4 8 . 5 3 { \scriptstyle \pm 0 . 5 0 }
$$

$$
3 2 . 5 0 { \scriptstyle \pm 1 . 0 2 }
$$

$$
\underline { { 7 5 . 5 9 } } { \scriptstyle \pm 1 . 6 0 }
$$

$$
2 4 . 7 0 { \scriptstyle \pm 1 . 4 9 }
$$

$$
3 3 . 2 1 { \scriptstyle \pm 0 . 3 3 }
$$

$$
7 . 3 3 { \scriptstyle \pm 0 . 7 1 }
$$

$$
3 . 3 9 { \scriptstyle \pm 1 . 0 9 }
$$

$$
2 1 . 5 0 { \scriptstyle \pm 2 . 5 5 }
$$

$$
3 4 . 4 9 { \scriptstyle \pm 0 . 3 6 }
$$

$$
8 1 . 4 5 { \scriptstyle \pm 1 . 6 1 }
$$

$$
7 0 . 5 1 { \scriptstyle \pm 1 . 9 5 }
$$

$$
1 3 . 7 0 { \scriptstyle \pm 0 . 4 3 }
$$

$$
4 . 9 0 { \scriptstyle \pm 0 . 7 9 }
$$

$$
1 4 . 1 0 { \scriptstyle \pm 0 . 2 1 }
$$

$$
6 0 . 5 5 { \scriptstyle \pm 0 . 7 5 }
$$

$$
\underline { { 4 8 . 8 3 } } _ { \pm 1 . 2 2 }
$$

$$
4 1 . 6 8 { \scriptstyle \pm 0 . 6 9 }
$$

$$
\underline { { 3 2 . 7 8 } } _ { \pm 1 . 1 5 }
$$

$$
5 6 . 2 5 { \scriptstyle \pm 0 . 6 0 }
$$

$$
4 9 . 5 3 { \scriptstyle \pm 0 . 2 8 }
$$

$$
2 9 . 9 8 { \scriptstyle \pm 0 . 5 4 }
$$

$$
\underline { { 1 5 . 5 3 } } _ { \pm 0 . 7 6 }
$$

$$
2 3 . 3 0 { \scriptstyle \pm 0 . 5 7 }
$$

$$
2 5 . 7 0 { \scriptstyle \pm 1 . 7 3 }
$$

$$
6 . 6 5 { \scriptstyle \pm 0 . 3 3 }
$$

$$
3 9 . 3 9 { \scriptstyle \pm 0 . 1 9 }
$$

$$
7 . 3 0 { \scriptstyle \pm 1 . 1 7 }
$$

$$
8 1 . 7 9 { \scriptstyle \pm 0 . 3 2 }
$$

$$
7 0 . 5 1 { \scriptstyle \pm 1 . 0 0 }
$$

$$
3 3 . 1 8 { \scriptstyle \pm 0 . 2 7 }
$$

$$
8 4 . 9 6 { \scriptstyle \pm 0 . 8 4 }
$$

<table><tr><td colspan="11">Qwen3.5-4B</td></tr><tr><td>Pretrained</td><td>18.05±0.82</td><td>47.45±1.13</td><td> $2 0 . 5 0 { \scriptstyle \pm 1 . 2 6 }$ </td><td> $2 1 . 5 5 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $2 8 . 1 8 { \scriptstyle \pm 1 . 2 8 }$ </td><td> $5 . 5 0 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $3 2 . 9 0 { \scriptstyle \pm 2 . 8 8 }$ </td><td>23.91±0.61</td><td> $9 . 1 9 { \scriptstyle \pm 1 . 7 3 }$ </td><td> $5 . 5 7 { \scriptstyle \pm 1 . 3 5 }$ </td></tr><tr><td>GRPO</td><td> $4 1 . 7 8 \scriptstyle \pm 0 . 7 3$ </td><td>66.68±0.63</td><td> $4 9 . 4 3 _ { \pm 0 . 3 8 }$ </td><td> $4 7 . 0 0 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $4 6 . 8 3 { \scriptstyle \pm 1 . 5 2 }$ </td><td> $1 9 . 3 5 { \scriptstyle \pm 1 . 1 1 }$ </td><td> $4 4 . 5 0 { \scriptstyle \pm 1 . 8 6 }$ </td><td> $\underline { { 4 5 . 1 5 } } _ { \pm 0 . 2 9 }$ </td><td> $7 4 . 2 1 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $6 6 . 0 2 { \scriptstyle \pm 1 . 5 6 }$ </td></tr><tr><td>GRPO+OPSD</td><td> $4 3 . 4 8 { \scriptstyle \pm 0 . 8 0 }$ </td><td>65.08±0.55</td><td> $4 8 . 4 3 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $4 5 . 7 8 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $4 0 . 7 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 1 . 1 0 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $4 5 . 8 0 { \scriptstyle \pm 2 . 1 3 }$ </td><td> $4 4 . 1 6 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $8 8 . 7 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td>77.15±1.30</td></tr><tr><td>SDAR</td><td>45.38±0.71</td><td> $6 5 . 7 5 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $4 9 . 6 0 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $4 8 . 8 3 { \scriptstyle \pm 1 . 3 7 }$ </td><td>41.08±0.95</td><td> $2 0 . 5 8 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $3 9 . 1 0 { \scriptstyle \pm 2 . 0 3 }$ </td><td>44.96±0.37</td><td>61.14±1.86</td><td> $5 1 . 1 7 { \scriptstyle \pm 1 . 4 5 }$ </td></tr><tr><td>Skill-SD</td><td> $4 4 . 9 3 _ { \pm 1 . 0 4 }$ </td><td> $6 3 . 9 5 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $4 6 . 7 8 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $4 4 . 1 0 { \scriptstyle \pm 0 . 9 4 }$ </td><td>37.30±1.25</td><td> $1 9 . 4 3 _ { \pm 1 . 5 4 }$ </td><td> $4 4 . 4 0 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $4 2 . 8 1 _ { \pm 0 . 2 2 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>0.00±0.00</td></tr><tr><td>RLSD</td><td> $4 5 . 1 3 { \scriptstyle \pm 0 . 7 8 }$ </td><td>65.00±0.81</td><td> $4 6 . 9 5 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $4 5 . 1 0 { \scriptstyle \pm 1 . 0 5 }$ </td><td>39.43±1.68</td><td> $1 9 . 9 8 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $4 3 . 6 0 { \scriptstyle \pm 3 . 0 5 }$ </td><td> $4 3 . 6 0 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $8 0 . 2 7 { \scriptstyle \pm 1 . 5 4 }$ </td><td>68.16±1.60</td></tr><tr><td>OCSD</td><td> $4 4 . 5 0 { \scriptstyle \pm 0 . 6 8 }$ </td><td>67.05±0.83</td><td> $\underline { { 4 9 . 6 3 } } _ { \pm 0 . 6 3 }$ </td><td> $4 5 . 9 5 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $3 9 . 1 3 { \scriptstyle \pm 1 . 1 9 }$ </td><td>21.58±0.87</td><td> $4 7 . 0 0 { \scriptstyle \pm 1 . 2 6 }$ </td><td> $4 4 . 7 3 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $7 0 . 3 2 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $6 4 . 1 6 { \scriptstyle \pm 0 . 7 7 }$ </td></tr><tr><td>OVCSD</td><td> $4 3 . 4 5 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $6 4 . 9 0 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $4 5 . 5 8 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $4 3 . 7 8 { \scriptstyle \pm 0 . 6 6 }$ </td><td>41.15±0.87</td><td>21.33±0.88</td><td> $\underline { { 4 7 . 1 0 } } _ { \pm 2 . 0 3 }$ </td><td> $4 3 . 5 1 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $\underline { { 8 3 . 7 9 } } _ { \pm 1 . 4 6 }$ </td><td>73.73±1.18</td></tr><tr><td>AgentOPSD</td><td> $4 4 . 9 5 { \scriptstyle \pm 0 . 6 7 }$ </td><td>64.65±0.82</td><td> $4 5 . 8 3 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $4 6 . 1 0 { \scriptstyle \pm 1 . 2 0 }$ </td><td>41.65±1.33</td><td>21.43±0.92</td><td> $4 7 . 9 0 { \scriptstyle \pm 1 . 9 8 }$ </td><td> $4 4 . 2 5 { \scriptstyle \pm 0 . 5 3 }$ </td><td>80.33±1.98</td><td> $7 2 . 0 7 { \scriptstyle \pm 2 . 3 2 }$ </td></tr><tr><td>ADRS</td><td> $4 1 . 7 5 { \scriptstyle \pm 0 . 5 4 }$ </td><td>60.85±0.46</td><td> $4 4 . 7 5 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $4 4 . 7 8 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $3 9 . 7 5 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $\underline { { 2 1 . 4 3 } } _ { \pm 1 . 1 3 }$ </td><td>44.50±1.65</td><td> $4 2 . 3 1 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $8 2 . 7 0 { \scriptstyle \pm 1 . 2 1 }$ </td><td> $6 5 . 7 2 { \scriptstyle \pm 1 . 5 3 }$ </td></tr><tr><td>ICSD</td><td> $4 5 . 9 8 _ { \pm 0 . 4 7 } ~ 6 8 . 0 3 _ { \pm 0 . 9 5 }$ </td><td></td><td> ${ \bf 5 0 . 5 0 { \scriptstyle \pm 0 . 4 1 } }$ </td><td> $\underline { { 4 7 . 6 5 } } _ { \pm 0 . 6 6 }$ </td><td> $\underline { { 4 3 . 7 3 } } \underline { { } } _ { \pm 0 . 9 7 }$ </td><td> $2 0 . 1 8 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $4 2 . 6 0 { \scriptstyle \pm 1 . 0 3 }$ </td><td> $4 5 . 8 7 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>PCSD</td><td> $4 4 . 2 3 { \scriptstyle \pm 0 . 7 7 }$ </td><td>65.78±0.98</td><td> $4 6 . 6 5 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $4 3 . 8 8 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $4 0 . 2 5 { \scriptstyle \pm 0 . 8 1 }$ </td><td>17.53±0.66</td><td> $4 4 . 6 0 { \scriptstyle \pm 2 . 2 1 }$ </td><td> $4 3 . 1 1 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $8 3 . 0 6 { \scriptstyle \pm 1 . 2 8 }$ </td><td> $7 2 . 1 7 { \scriptstyle \pm 1 . 5 0 }$ </td></tr></table>

<table><tr><td rowspan=1 colspan=3>Qwen3.5-9B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=2>Pretrained     $1 9 . 1 3 { \scriptstyle \pm 0 . 8 3 }$ </td><td rowspan=1 colspan=1> $5 5 . 3 0 { \scriptstyle \pm 1 . 2 2 }$ </td><td rowspan=1 colspan=1> $2 3 . 5 0 { \scriptstyle \pm 1 . 2 2 }$ </td><td rowspan=1 colspan=1> $2 1 . 2 0 { \scriptstyle \pm 0 . 9 3 }$ </td><td rowspan=1 colspan=1> $3 0 . 0 8 { \scriptstyle \pm 1 . 4 8 }$ </td><td rowspan=1 colspan=1>6.60±0.60 4</td><td rowspan=1 colspan=1>3.00±1.99</td><td rowspan=1 colspan=1> $2 6 . 6 5 { \scriptstyle \pm 0 . 2 3 }$ </td><td rowspan=1 colspan=1> $9 . 8 9 { \scriptstyle \pm 1 . 0 1 }$ </td><td rowspan=1 colspan=1> $5 . 8 6 { \scriptstyle \pm 1 . 1 0 }$ </td></tr><tr><td rowspan=1 colspan=2>GRPO          $4 5 . 6 8 { \scriptstyle \pm 0 . 5 5 }$ </td><td rowspan=1 colspan=1>66.73±0.98</td><td rowspan=1 colspan=1> $4 8 . 1 5 { \scriptstyle \pm 0 . 5 2 }$ </td><td rowspan=1 colspan=1> $4 8 . 4 5 { \scriptstyle \pm 1 . 1 4 }$ </td><td rowspan=1 colspan=1> $\underline { { 4 7 . 5 3 } } _ { \pm 1 . 8 4 }$ </td><td rowspan=1 colspan=1> $2 1 . 9 0 { \scriptstyle \pm 1 . 1 1 }$ </td><td rowspan=1 colspan=1> $5 1 . 0 0 { \scriptstyle \pm 1 . 8 0 }$ </td><td rowspan=1 colspan=1>46.59±0.53</td><td rowspan=1 colspan=1> $8 3 . 3 0 { \scriptstyle \pm 1 . 1 6 }$ </td><td rowspan=1 colspan=1>70.02±1.38</td></tr><tr><td rowspan=1 colspan=2>GRPO+OPSD 46.50±0.88 67</td><td rowspan=1 colspan=1>.33±0.49</td><td rowspan=1 colspan=1> $\underline { { 5 0 . 6 8 } } _ { \pm 0 . 7 6 }$ </td><td rowspan=1 colspan=1> $4 7 . 5 8 { \scriptstyle \pm 1 . 0 8 }$ </td><td rowspan=1 colspan=1>46.48±1.44</td><td rowspan=1 colspan=1> $2 2 . 5 5 { \scriptstyle \pm 0 . 8 1 }$ </td><td rowspan=1 colspan=1> $5 2 . 7 0 { \scriptstyle \pm 3 . 0 4 }$ </td><td rowspan=1 colspan=1>47.08±0.35</td><td rowspan=1 colspan=1>85.31±1.34</td><td rowspan=1 colspan=1>74.90±0.77</td></tr><tr><td rowspan=1 colspan=2>SDAR         $4 2 . 6 5 { \scriptstyle \pm 0 . 4 1 } \quad { \underl</td><td rowspan=1 colspan=1>ine { { 6 8 . 3 8 } } } { \scriptstyle \pm 0 . 9 5 }$ </td><td rowspan=1 colspan=1> $4 7 . 0 0 { \scriptstyle \pm 1 . 6 0 }$ </td><td rowspan=1 colspan=1> $4 8 . 2 8 { \scriptstyle \pm 1 . 2 2 }$ </td><td rowspan=1 colspan=1> $3 9 . 9 3 { \scriptstyle \pm 0 . 7 9 }$ </td><td rowspan=1 colspan=1> $2 1 . 4 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td rowspan=1 colspan=1> $4 8 . 2 0 { \scriptstyle \pm 1 . 7 5 }$ </td><td rowspan=1 colspan=1> $4 4 . 7 5 { \scriptstyle \pm 0 . 5 5 }$ </td><td rowspan=1 colspan=1>85.19+1.61</td><td rowspan=1 colspan=1> $\underline { { 7 4 . 3 2 } } _ { \pm 2 . 1 9 }$ </td></tr><tr><td rowspan=1 colspan=2>Skill-SD       $4 6 . 1 0 _ { \pm 0 . 5 1 } 6 7 . 5 3 {</td><td rowspan=1 colspan=1>\scriptstyle \pm 0 . 7 0 }$ </td><td rowspan=1 colspan=1> $4 8 . 6 8 { \scriptstyle \pm 0 . 8 0 }$ </td><td rowspan=1 colspan=1> $\underline { { 4 8 . 7 5 } } _ { \pm 0 . 5 2 }$ </td><td rowspan=1 colspan=1> $4 4 . 3 5 { \scriptstyle \pm 1 . 6 3 }$ </td><td rowspan=1 colspan=1> $2 1 . 7 5 { \scriptstyle \pm 0 . 8 3 }$ </td><td rowspan=1 colspan=1> $4 6 . 1 0 { \scriptstyle \pm 1 . 4 8 }$ </td><td rowspan=1 colspan=1> $4 6 . 1 9 { \scriptstyle \pm 0 . 2 8 }$ </td><td rowspan=1 colspan=1>0.00±0.00</td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td rowspan=1 colspan=2>RLSD         $4 3 . 6 3 _ { \pm 0 . 3 9 }$ </td><td rowspan=1 colspan=1> $6 7 . 2 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td rowspan=1 colspan=1> $4 7 . 2 0 { \scriptstyle \pm 1 . 4 6 }$ </td><td rowspan=1 colspan=1> $4 7 . 3 5 { \scriptstyle \pm 0 . 7 3 }$ </td><td rowspan=1 colspan=1> $3 8 . 1 5 { \scriptstyle \pm 1 . 1 8 }$ </td><td rowspan=1 colspan=1> $2 1 . 0 5 { \scriptstyle \pm 0 . 4 2 }$ </td><td rowspan=1 colspan=1> $4 9 . 5 0 { \scriptstyle \pm 2 . 0 3 }$ </td><td rowspan=1 colspan=1> $4 4 . 3 1 _ { \pm 0 . 2 9 }$ </td><td rowspan=1 colspan=1>0.00±0.00</td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td rowspan=1 colspan=1>OCSD</td><td rowspan=1 colspan=1> $4 7 . 8 8 { \scriptstyle \pm 0 . 9 1 }$ </td><td rowspan=1 colspan=1> $6 8 . 5 3 { \scriptstyle \pm 0 . 8 1 }$ </td><td rowspan=1 colspan=1> $5 2 . 9 8 { \scriptstyle \pm 0 . 7 9 }$ </td><td rowspan=1 colspan=1> $4 6 . 9 3 { \scriptstyle \pm 0 . 7 0 }$ </td><td rowspan=1 colspan=1> $4 9 . 4 3 { \scriptstyle \pm 1 . 2 8 }$ </td><td rowspan=1 colspan=1> ${ \bf 2 3 . 8 0 { \scriptstyle \pm 0 . 6 2 } }$ </td><td rowspan=1 colspan=1> $4 5 . 8 0 { \scriptstyle \pm 2 . 1 3 }$ </td><td rowspan=1 colspan=1> $4 8 . 1 6 _ { \pm 0 . 4 4 }$ </td><td rowspan=1 colspan=1> $7 7 . 0 3 { \scriptstyle \pm 2 . 0 8 }$ </td><td rowspan=1 colspan=1> $6 7 . 7 7 { \scriptstyle \pm 2 . 2 0 }$ </td></tr><tr><td rowspan=1 colspan=1>OVCSD</td><td rowspan=1 colspan=1>44.40±0.57 6</td><td rowspan=1 colspan=1>7.30±0.99</td><td rowspan=1 colspan=1> $4 7 . 0 0 { \scriptstyle \pm 1 . 1 7 }$ </td><td rowspan=1 colspan=1>47.55±0.82</td><td rowspan=1 colspan=1>43.55±1.35 2</td><td rowspan=1 colspan=1>1.28±0.85</td><td rowspan=1 colspan=1> $5 1 . 4 0 { \scriptstyle \pm 2 . 4 5 }$ </td><td rowspan=1 colspan=1>45.43±0.47</td><td rowspan=1 colspan=1>80.36±1.83</td><td rowspan=1 colspan=1>68.07±1.28</td></tr><tr><td rowspan=1 colspan=1>AgentOPSD</td><td rowspan=1 colspan=1> $4 6 . 1 5 { \scriptstyle \pm 0 . 7 5 ~ 6</td><td rowspan=1 colspan=1>7 . 3 0 \scriptstyle \pm 1 . 1 2 }$ </td><td rowspan=1 colspan=1> $4 7 . 3 3 { \scriptstyle \pm 0 . 6 3 }$ </td><td rowspan=1 colspan=1> $4 9 . 0 5 { \scriptstyle \pm 0 . 5 2 }$ </td><td rowspan=1 colspan=1> $4 3 . 6 8 _ { \pm 1 . 1 7 }$ </td><td rowspan=1 colspan=1> $2 0 . 5 0 _ { \pm 0 . 9 3 } \qua</td><td rowspan=1 colspan=1>d 4 9 . 2 0 _ { \pm 1 . 9 1 }$ </td><td rowspan=1 colspan=1> $4 5 . 8 1 _ { \pm 0 . 5 0 }$ </td><td rowspan=1 colspan=1> $8 3 . 7 2 { \scriptstyle \pm 1 . 8 0 }$ </td><td rowspan=1 colspan=1> $7 2 . 2 7 { \scriptstyle \pm 2 . 0 9 }$ </td></tr><tr><td rowspan=1 colspan=1>ADRS</td><td rowspan=1 colspan=1> $4 0 . 5 8 { \scriptstyle \pm 0 . 5 8 } 6</td><td rowspan=1 colspan=1>6 . 3 8 { \scriptstyle \pm 0 . 4 6 }$ </td><td rowspan=1 colspan=1> $4 3 . 4 8 _ { \pm 0 . 7 9 }$ </td><td rowspan=1 colspan=1> $4 5 . 9 3 _ { \pm 0 . 8 4 }$ </td><td rowspan=1 colspan=1> $4 3 . 0 8 { \scriptstyle \pm 1 . 0 1 }$ </td><td rowspan=1 colspan=1> $2 2 . 5 5 { \scriptstyle \pm 0 . 4 8 }$ </td><td rowspan=1 colspan=1> $5 0 . 3 0 { \scriptstyle \pm 0 . 7 9 }$ </td><td rowspan=1 colspan=1> $4 3 . 9 3 _ { \pm 0 . 3 7 }$ </td><td rowspan=1 colspan=1> $8 1 . 2 0 { \scriptstyle \pm 1 . 6 2 }$ </td><td rowspan=1 colspan=1> $7 0 . 3 1 { \scriptstyle \pm 1 . 1 0 }$ </td></tr><tr><td rowspan=1 colspan=1>ICSD</td><td rowspan=1 colspan=1>46.55±0.98</td><td rowspan=1 colspan=1>66.88±0.60</td><td rowspan=1 colspan=1> $5 0 . 0 0 { \scriptstyle \pm 0 . 5 9 }$ </td><td rowspan=1 colspan=1> $4 8 . 4 0 { \scriptstyle \pm 0 . 6 6 }$ </td><td rowspan=1 colspan=1> $4 5 . 0 8 { \scriptstyle \pm 1 . 2 3 }$ </td><td rowspan=1 colspan=1> $\underline { { 2 3 . 5 5 } } _ { \pm 0 . 8 7 }$ </td><td rowspan=1 colspan=1> $4 9 . 3 0 { \scriptstyle \pm 1 . 6 5 }$ </td><td rowspan=1 colspan=1> $4 6 . 8 4 _ { \pm 0 . 4 1 }$ </td><td rowspan=1 colspan=1> $8 1 . 8 9 { \scriptstyle \pm 2 . 4 4 }$ </td><td rowspan=1 colspan=1> $7 2 . 4 6 { \scriptstyle \pm 1 . 3 7 }$ </td></tr><tr><td rowspan=1 colspan=2>PCSD         $4 4 . 6 8 _ { \pm 1 . 0 0 }$ </td><td rowspan=1 colspan=1>67.68±0.79</td><td rowspan=1 colspan=1> $4 7 . 5 0 { \scriptstyle \pm 0 . 8 0 }$ </td><td rowspan=1 colspan=1> $4 7 . 0 3 { \scriptstyle \pm 0 . 6 0 }$ </td><td rowspan=1 colspan=1> $4 2 . 1 8 _ { \pm 1 . 5 7 }$ </td><td rowspan=1 colspan=1> $2 1 . 9 5 { \scriptstyle \pm 0 . 6 7 }$ </td><td rowspan=1 colspan=1> $\underline { { 5 1 . 5 0 } } _ { \pm 1 . 1 3 }$ </td><td rowspan=1 colspan=1> $4 5 . 4 2 _ { \pm 0 . 5 5 }$ </td><td rowspan=1 colspan=1> $8 3 . 6 7 { \scriptstyle \pm 0 . 5 2 }$ </td><td rowspan=1 colspan=1> $\underline { { 7 4 . 3 2 } } _ { \pm 0 . 6 5 }$ </td></tr></table>

Table 3 Results on Search-QA and WebShop: mean<sub>±SD</sub> over eight rollout seeds at T = 0.4, after 100 steps for trained models. We report exact-match accuracy (%) for Search-QA and task score (Score) and success rate (%, Acc.) for WebShop. Highlighting follows Table 2.

aspects of stability that this aggregate measure and the final score can each miss. Appendix F.3 examines early stagnation on 2B ScienceWorld, where most methods share a low-score plateau with overlapping validation curves and little further improvement.

Final rankings do not capture sustained performance. Across all twelve settings, PCSD has the highest mean final performance (54.26), but its validation AUC of 38.45 is lower than OCSD’s 39.88. OCSD instead ranks first by AUC and fourth by final performance (49.50). AgentOPSD shows a larger discrepancy: it ranks second in final performance (51.34) but ninth by AUC (34.41), below GRPO’s 36.89. A strong final model can therefore emerge from a substantially weaker training trajectory; final rankings alone do not characterize performance throughout optimization. Appendix F.1 compares aggregate trajectories, rankings, and stage-wise performance.

Recovery can conceal severe intermediate collapse. On 4B ALFWorld, AgentOPSD reaches 60.0% validation success at step 35, then scores no more than 4.3% at each of the six evaluations from steps 45 to 70, before recovering to 79.3% at step 100. In the same setting, GRPO+OPSD reaches 39.3% at step 20 but falls to zero at steps 25–40 and ends at 2.1%. Both methods undergo severe degradation, but only one ends with low performance. An endpoint-only assessment would miss AgentOPSD’s prolonged near-zero phase. A strong final result therefore does not establish a stable training process. Appendices F.2 and F.5 provide trajectory comparisons and failure diagnostics, including decoding-specific failures.

![](images/3c87982c95a4a1326e288ca22f6ce0ed3c2dbfae86d9789f90e22f560fa09ccd.jpg)  
Figure 2 Validation trajectories of eleven methods across four benchmarks and three model sizes, covering 132 training runs and 2,772 validation sweeps. Curves show validation scores recorded throughout training. Colored markers distinguish overlapping curves on 2B ScienceWorld.

Late regression matters even without complete failure. Across the twelve settings, 9 runs finish at least 20 points below their observed validation peak. OCSD on 4B ALFWorld reaches 85.0% at step 60 but ends at 51.4%; its final three validations average 50.5%. On 9B ScienceWorld, ICSD falls from a peak task score of 55.23 to 18.77, with its final three validations averaging 26.99. In both cases, performance remains well below the peak over the final three validations, indicating sustained late regression rather than an isolated unfavorable final measurement. Stable training therefore requires retaining acquired performance as well as avoiding collapse; a nonzero endpoint is insuficient when substantial gains have been lost. Appendix F.4 provides per-run summaries.

## 7 The Cost of Performance Gains

The practical value of a performance gain depends on the resources required to obtain it. Equal training-step counts do not imply equal costs, since rollout lengths and teacher computation vary across methods. For all eleven methods across the twelve model–task settings, Figure 3 pairs mean final performance over eight rollout seeds with measured training time. All runs use eight GPUs, allowing us to examine both the cost of the final result and the performance attainable within GRPO’s training-time budget. Appendix G.2 details the timing components, GPU-hour totals, and stage-level costs.

![](images/4954bfd0220e00f5efce57f29aa0afe6069ec8530e32aa282d391872f8e55060.jpg)  
Figure 3 Final performance versus training time across twelve model–task settings. Points show mean performance over eight rollout seeds; dashed lines mark pretrained means. Values for all points are listed in Table S13 in Appendix G.1.

Additional computation does not guarantee better performance. Of the 120 comparisons with GRPO across methods and settings, 112 require more net training time; 58 of these also achieve lower eight-seed mean performance, and seven tie GRPO. Even PCSD and AgentOPSD, the only alternatives that improve mean final performance, require mean paired training-time ratios of 1.17 and 1.19 relative to GRPO. Additional training efort is therefore not consistently rewarded by higher performance, and the methods that do improve must be assessed together with their overhead. Measured cost is essential to determining whether a reported gain represents a practically useful improvement.

Fixed-step advantages may disappear under a time budget. Using each GRPO run’s net training time as the budget, we select the latest validation result reached by each method within that budget and compare it with its step-100 score. Scores are averaged over the three model sizes within each benchmark. On ALFWorld, AgentOPSD’s mean success falls from 68.10% at step 100 to 37.63% within budget, reversing its advantage over GRPO’s 58.10%. PCSD shows the same reversal on WebShop: 73.17% at step 100 becomes 63.83% within budget, below GRPO’s 68.20%. These reversals show that an advantage at a fixed step count need not be available at a comparable computational cost. Method comparisons should therefore assess not only the final gains, but also whether those gains can be achieved within the resources available to the baseline.

## 8 Discussion

The faster a field produces new methods, the more it needs a shared basis for judging progress. The publication pace documented in our survey makes this need particularly pressing for agentic self-distillation. Without direct, comprehensive comparisons, a growing collection of reported gains need not yield a clearer understanding of which ideas are worth building on.

This is why we chose to invest in independent evaluation rather than propose another training method. We aim to clarify where methods improve on strong baselines, where gains break down, and what risks and costs they entail. A method need not win every setting to be useful, but its value can only be understood when both its strengths and its limitations are visible.

We hope this benchmark helps researchers choose baselines, investigate unresolved failures, and pursue reliable, cost-efective improvements. We therefore call for the community to value systematic evaluation alongside algorithmic innovation. As the literature expands, the community’s capacity for comparison and analysis must grow with it, so that new methods contribute to cumulative understanding and sustained progress.

## 9 Conclusion

We benchmark ten representative self-distillation methods against GRPO across four agent benchmarks and three model sizes. Gains over GRPO remain setting-dependent, with no method uniformly superior; strong final results can conceal unstable training, and additional computation does not consistently improve performance. These findings show that progress must be assessed through the breadth of gains, the stability of learning, and the cost of improvement. By documenting both gains and failures under a common protocol, our study helps researchers select strong baselines, identify unresolved training challenges, and pursue more reliable, cost-efective agent training.

## Limitations

Limited computational resources and researcher time prevent an exhaustive evaluation of the rapidly growing literature on agentic self-distillation. We therefore compare ten representative methods with GRPO, covering diferent approaches to incorporating teacher guidance. Although this selection does not encompass every relevant method, the observed setting-dependent gains, training instability, and performance–cost trade-ofs substantiate the need for the comprehensive evaluation advocated in this work. Extending the benchmark to additional methods would help establish how broadly these patterns hold.

## Ethical Considerations

This study compares existing agent-training methods using publicly available models and benchmark datasets. Experiments are confined to simulated environments and retrieval-based question answering, without humanparticipant studies or deployment on behalf of real users. We identify no significant ethical concerns specific to this study.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In International Conference on Learning Representations, volume 2024, pages 21246–21263.

Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan, Manzil Zaheer, Felix Yu, and Sanjiv Kumar. 2023. ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent. Preprint, arXiv:2312.10003.

Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. 2023. FireAct: Toward Language Agent Fine-tuning. Preprint, arXiv:2310.05915.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024. MiniLLM: Knowledge Distillation of Large Language Models. In International Conference on Learning Representations, volume 2024, pages 32694–32717.

Sarthak Harne, Chinmay Karkar, Yash Pandya, Ahmed Awadallah, and Akshay Nambi. 2026. Privileged, but Biased: How PI-Conditioned Teachers Break Self-Distillation. Preprint, arXiv:2608.04794.

Geofrey Hinton, Oriol Vinyals, and Jef Dean. 2015. Distilling the Knowledge in a Neural Network. Preprint, arXiv:1503.02531.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8003–8017, Toronto, Canada. Association for Computational Linguistics.

Yoon Kim and Alexander M. Rush. 2016. Sequence-Level Knowledge Distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas. Association for Computational Linguistics.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. 2024. DistiLLM: Towards Streamlined Distillation for Large Language Models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 24872–24895. PMLR.

Qizhen Lan, Xi Xiao, Xiangchen Guan, Mengchen Fan, Moule Lin, Jung Im Choi, and Lijing Zhu. 2026. Trust Is Not Enough: Influence Calibration for On-Policy Self-Distillation in Agentic RL. Preprint, arXiv:2608.14945.

Alexander Lin, Jeremy Wohlwend, Howard Chen, and Tao Lei. 2020. Autoregressive Knowledge Distillation through Imitation Learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6121–6133, Online. Association for Computational Linguistics.

David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, and Vladimir Vapnik. 2016. Unifying distillation and privileged information. Preprint, arXiv:1511.03643.

Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026. Self-Distilled Agentic Reinforcement Learning. Preprint, arXiv:2605.15155.

Chunji Lv, Yangguang Wei, Junlin Liu, Yang Gao, Ming Liu, Xinming Wang, Jinyang Wu, Guoren Wang, and Changsheng Li. 2026. PCSD: Persistent Consistency for Self-Distillation in Agentic Reinforcement Learning. Preprint, arXiv:2608.01837.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-Refine: Iterative Refinement with Self-Feedback. In Advances in Neural Information Processing Systems, volume 36, pages 46534–46594. Curran Associates, Inc.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, Jie Tang, and Yuxiao Dong. 2025. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning. In International Conference on Learning Representations, volume 2025, pages 79791–79821.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. In International Conference on Learning Representations, volume 2024, pages 9695–9717.

Stephane Ross, Geofrey Gordon, and Drew Bagnell. 2011. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 627–635, Fort Lauderdale, FL, USA. PMLR.

Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. 2016. Policy Distillation. Preprint, arXiv:1511.06295.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. In Advances in Neural Information Processing Systems, volume 36, pages 68539–68551. Curran Associates, Inc.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. Preprint, arXiv:2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652. Curran Associates, Inc.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. Preprint, arXiv:2010.03768.

Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. 2026a. Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents. Preprint, arXiv:2604.10674.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. 2022. ScienceWorld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Zi-Han Wang, Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Jie Wu, Zhengzhou Cai, Yueqing Sun, Ziang Ye, Linji Hao, Qi Gu, Xunliang Cai, Yongliang Shen, and Yujiu Yang. 2026b. AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning. Preprint, arXiv:2608.05987.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. 2025. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning. Preprint, arXiv:2504.20073.

Zhiheng Xi, Yiwen Ding, Wenxiang Chen, Boyang Hong, Honglin Guo, Junzhe Wang, Xin Guo, Dingwen Yang, Chenyang Liao, Wei He, Songyang Gao, Lu Chen, Rui Zheng, Yicheng Zou, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu-Gang Jiang. 2025. AgentGym: Evaluating and Training Large Language Model-based Agents across Diverse Environments. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27914–27961, Vienna, Austria. Association for Computational Linguistics.

Xu Xia, Jinghua Piao, Min Yang, Xiaochong Lan, Jiaju Chen, and Yong Li. 2026. From Scoring to Acting: Outcome-Verified Comparative Self-Distillation for LLM Agents. Preprint, arXiv:2607.27937.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. 2026a. Self-Distilled RLVR. Preprint, arXiv:2604.03128.

Fan Yang, Rui Meng, and Yuxin Wen. 2026b. Why Does Feedback-Augmented Self-Distillation Fail to Improve Retrieval-Interleaved Search Agents? Preprint, arXiv:2607.17558.

Yi Yang, Cong Qin, Xiaodan Liu, Chishui Chen, Qing Dong, Yan Zhang, Cao Liu, Zhao Yang, Lu Pan, Jiaye Lin, and Yi Feng. 2026c. Agentic Reinforcement Learning with Observation-Calibrated Self-Distillation. Preprint, arXiv:2608.04788.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. In Advances in Neural Information Processing Systems, volume 35, pages 20744–20757. Curran Associates, Inc.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. Preprint, arXiv:2210.03629.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. STaR: Bootstrapping Reasoning With Reasoning. In Advances in Neural Information Processing Systems, volume 35, pages 15476–15488. Curran Associates, Inc.

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2024. AgentTuning: Enabling Generalized Agent Abilities for LLMs. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3053–3077, Bangkok, Thailand. Association for Computational Linguistics.

Ranxu Zhang, Guinan Chen, Chenshaodong, Jinghao Lin, Xiaozhou Xu, Sunzhe, Yanyong Zhang, and Chao Wang. 2026. Agentic Reinforcement Learning with Self-Distilled Reward Shaping. Preprint, arXiv:2608.03223.

## A Appendix Overview

## • Appendix B: Method Descriptions and Adaptations

## • Appendix C: Experimental Details

Appendix C.1: Training Configuration

– Appendix C.2: Teacher Models and Contexts

– Appendix C.3: Numerical Precision and Stability

– Appendix C.4: Validation and Repeated Evaluation

– Appendix C.5: Experimental Records

## • Appendix D: Training Objectives and Algorithms

– Appendix D.1: Shared Training Procedure

Appendix D.2: Auxiliary Distillation Objectives

Appendix D.3: Advantage Reweighting

Appendix D.4: Verified Teacher Continuations

Appendix D.5: Teacher-Based Reward Shaping

## • Appendix E: Cross-Setting Gains and Regressions

– Appendix E.1: All Twelve Settings

– Appendix E.2: Low-Performance Outcomes

## • Appendix F: Training Dynamics and Stability

– Appendix F.1: Aggregate Progress and Method Rankings

– Appendix F.2: Intermediate Instability and Late Regression

Appendix F.3: Early Stagnation on ScienceWorld

– Appendix F.4: Per-Run Training Profiles

– Appendix F.5: Task Failure and Numerical Diagnostics

## • Appendix G: Computational Cost Analysis

– Appendix G.1: Performance and Time Budgets

– Appendix G.2: Total Compute and Stage-Level Costs

## • Appendix H: Literature Survey

Appendix H.1: Search Strategy and Selection Criteria

– Appendix H.2: Publication Statistics

## B Method Descriptions and Adaptations

The eleven methods span outcome-based policy optimization, auxiliary distillation, token- or turn-level advantage reweighting, reward shaping, and verified teacher continuations. Teachers use the same model backbone as the student, with privileged information available only during training. Table S1 summarizes these mechanisms.

We implement the methods in a shared text-agent framework. For Skill-SD and RLSD, we follow the SDAR repository’s skill-bank adaptations, using the pre-update actor to score student responses under skill-conditioned prompts (Lu et al., 2026). GRPO+OPSD uses SDAR’s constant-gate auxiliary objective, whose gradient is independent of teacher scores. OVCSD collects additional, environment-verified teacher continuations.

For AgentOPSD and ADRS, we follow their released implementations of return-based statistics (Wang et al., 2026b; Zhang et al., 2026). AgentOPSD initializes its belief from the clipped mean of trajectory-deduplicated training returns. ADRS computes L2 teacher utility over interaction rows grouped by task.

Our ICSD implementation uses an additional gradient-free student-scoring pass to calibrate weights within each local actor minibatch; this computation is included in the reported training cost. Following Lv et al. (2026), PCSD uses a frozen, skill-conditioned teacher initialized from the pretrained backbone; we reuse the fixed reference model’s weights for this branch. The complete objectives and update procedures appear in Appendix D.

## C Experimental Details

## C.1 Training Configuration

We train each of the eleven methods on ALFWorld, ScienceWorld, Search-QA, and WebShop with Qwen3.5-2B, 4B, and 9B, for 132 runs of 100 global steps each. We combine shared optimization defaults with benchmarkand method-specific configurations drawn from the source papers and released code. The sources of these configurations and the settings for our benchmark extensions are described below; method-specific algorithm hyperparameters remain fixed across the three model sizes within each benchmark. Tables S2, S3, and S4 summarize common settings, benchmark-specific settings and overrides, and algorithm hyperparameters, respectively, across all four benchmarks.

Configuration sources. The base learning rate, rollout group size, and default reference KL coeficient follow Appendix C of SDAR (Lu et al., 2026). For ALFWorld, WebShop, and Search-QA, prompt and response limits, interaction horizons, history windows, PPO minibatch sizes, and invalid-action penalties follow its released task-specific configurations. ALFWorld and WebShop retain outcome rewards of 10 for success and 0 otherwise; Search-QA uses 1 for a correct answer and 0 otherwise, together with SDAR’s top-three-passage retrieval configuration. ScienceWorld uses the AgentGym task split (Xi et al., 2025), with 200 evaluation tasks and 2,059 training instances after removing 61 overlapping task IDs; its episode reward is the simulator’s final task score divided by 100. These rewards are assigned to interaction rows, with the invalid-action penalty applied to the corresponding row. Table S3 summarizes these benchmark configurations and their method-specific overrides.

Baseline adaptations. SDAR and GRPO+OPSD retain SDAR’s reported auxiliary coeficients and gate settings. For Skill-SD and RLSD, we use the skill-bank adaptations in the SDAR repository: Skill-SD keeps a constant auxiliary weight of 0.001, while RLSD follows the released Qwen3 scripts’ 100-step linear decay with initial scale $\lambda _ { 0 } = 0 . 5$ and multiplier clipping of 0.2.

Method-specific settings from the source papers. AgentOPSD’s algorithm hyperparameters follow Appendix F of Wang et al. (2026b), including its upper PPO clipping bound of 0.24. OCSD’s selection ratio and reweighting coeficient follow Table 4 of Yang et al. (2026c); ICSD’s calibration parameters follow Appendix C.1 of Lan et al. (2026); PCSD’s window and gating parameters follow Table 5 of Lv et al. (2026). OCSD’s Search-QA warmup ratio of 0.1 and reference KL coeficient of 0.001 follow its released training configuration. The algorithm-specific values are summarized in Table S4.

Benchmark-specific ADRS settings. We follow the primary GRPO/L2 configurations in Appendix A.5 of Zhang et al. (2026): ALFWorld uses η = 0.1, batch-wide score normalization, and gate-input normalization; WebShop uses η = 0.02, per-response score normalization, and no gate-input normalization. The same source specifies ALFWorld’s entropy coeficient of 0.01 and upper PPO clipping bound of 0.28. For Search-QA, the teacher-reward scale $\eta = 0 . 0 5$ , batch-wide score normalization, and reference KL coeficient of 0.001 likewise follow the reported Search-QA configuration. These task-dependent choices preserve the source configurations; the paper’s normalization ablation supports batch-wide scaling on ALFWorld and Search-QA, and per-response scaling on WebShop. For the ScienceWorld extension, we adopt the Search-QA teacher-reward scale and normalization settings, including gate-input normalization, while retaining the common optimization defaults.

<table><tr><td>Method</td><td>Supervision source</td><td>Use in optimization</td></tr><tr><td>GRPO (Shao et al., 2024)</td><td>Task rewards only.</td><td>Policy advantage: group-normalized returns drive the clipped policy-gradient objective.</td></tr><tr><td>et al., 2026)</td><td>GRPO+OPSD (Lu Skill-conditioned teacher scores.</td><td>Auxiliary loss: constant-weight reinforcement of sam- pled student tokens; teacher scores do not affect its gradient.</td></tr><tr><td>SDAR (Lu et al., 2026)</td><td>Skill-conditioned teacher scores.</td><td>Auxiliary loss: a detached teacher-student score-gap gate weights sampled-token supervision.</td></tr><tr><td>Skill-SD (Wang et al., 2026a)</td><td>Skill-conditioned teacher scores.</td><td>Auxiliary loss: policy-ratio-weighted K3 distillation on sampled tokens.</td></tr><tr><td>RLSD (Yang et al., 2026a)</td><td>Skill-conditioned teacher scores.</td><td>Token advantages: clipped, sign-aware weighting from teacher-student log-probability gaps.</td></tr><tr><td>OCSD (Yang et al., 2026c)</td><td>Paired scores with and without future observations.</td><td>Token advantages: reweight selected turns using the score difference induced by future observations.</td></tr><tr><td>OVCSD (Xia et al., 2026)</td><td>Skill-conditioned, environment- verified continuations and teacher token distributions.</td><td>Comparative advantages and suffix loss: contrast state- aligned student/teacher actions and distill verified suf- fixes with top-K-plus-tail KL.</td></tr><tr><td>AgentOPSD (Wang et al., 2026b)</td><td>Skill-conditioned teacher scores and group task returns.</td><td>Turn advantages: recursive success-belief increments yield bounded, sign-aware reweighting across a trajec- tory.</td></tr><tr><td>ADRS (Zhang et al., 2026)</td><td>Skill-conditioned teacher scores and task returns.</td><td>Reward and advantage shaping: gate teacher rewards by return-based utility, then recompute advantages and add a token-level correction.</td></tr><tr><td>ICSD (Lan et al., 2026)</td><td>Skill-conditioned teacher scores.</td><td>Auxiliary loss: calibrate teacher trust and advantage- dependent influence; redistribute token weights while</td></tr><tr><td>PCSD (Lv et al., 2026)</td><td>Skill-conditioned scores from a frozen reference model.</td><td>preserving their per-response total. Auxiliary loss: gate supervision using variance-adaptive window averages and declining trends in the teacher- student score gap.</td></tr></table>

Table S1 Supervision sources and their use in the evaluated implementations. All methods use task rewards; teacher scores denote log-probabilities of student-generated tokens.

The benchmark-specific settings are listed in Table S3.

Study-specific choices. The Qwen3.5 backbones, 100-step training horizon, eight-H800 execution, and fixed evaluation sets are specified by our benchmark. Task batches retain SDAR’s size of 16 for ALFWorld and WebShop; we use 16 for ScienceWorld and 64 for Search-QA. Within each benchmark, task batches, environment settings, and evaluation sets are shared by all methods and model sizes (Table S3). We follow OVCSD’s outcome-verified intervention and sufix-distillation framework (Xia et al., 2026). For our implementation, we fix the continuation budget at two attempts per eligible group, the teacher-distribution approximation at top-32 plus tail, and the sufix-distillation weight at 0.1 across all twelve settings.

Trajectory collection and updates. Each ALFWorld/WebShop/ScienceWorld global step samples 16 task instances and eight student trajectories per instance, giving 128 student episodes. Search-QA instead samples 64 questions and eight trajectories each, giving 512 nominal student episodes per step; equal step counts therefore do not imply equal episode budgets across benchmarks. Trajectories use separate environment replicas and are split into interaction turns for policy optimization. The actor minibatch sizes are 256 turns for ALFWorld and Search-QA, and 64 for WebShop and ScienceWorld; one PPO epoch may therefore contain several optimizer updates. OVCSD additionally allows up to two teacher continuation attempts per eligible task group, including prefix replay.

## C.2 Teacher Models and Contexts

SDAR, GRPO+OPSD, Skill-SD, RLSD, AgentOPSD, ADRS, and ICSD score student responses with the pre-update actor conditioned on skills. PCSD uses the fixed reference model as its teacher. Following SDAR’s skill-conditioning procedure (Lu et al., 2026), we use benchmark-specific skill banks that combine general guidance with task-specific instructions selected from ALFWorld game identifiers or matching prompt keywords. We prepend this context to the student prompt, retokenize, and apply left truncation at the task’s prompt-length limit while keeping the student response unchanged. OVCSD uses the same skill-conditioned input for continuations; OCSD instead compares replay contexts with and without future observations.

<table><tr><td>Parameter</td><td>Configuration</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Trainable parameters</td><td>Full-parameter training</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W } ; ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$  ; weight decay 0.01</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 6 } ;$  constant after any warmup</td></tr><tr><td>Warmup ratio</td><td>0, except as listed in Table S3</td></tr><tr><td>Gradient norm cap  $c _ { \mathrm { g r a d } }$ </td><td>1.0</td></tr><tr><td>PPO epochs and update microbatch</td><td> $N _ { \mathrm { e p } } = 1 ;$  1 interaction turn per GPU; fixed-size microbatches</td></tr><tr><td>Log-probability microbatch</td><td>1 interaction turn per GPU for both actor and reference</td></tr><tr><td>Lower PPO clipping and dual clipping</td><td> $\epsilon _ { l } = 0 . 2 ; c _ { \mathrm { d u a l } } = 3 . 0$ </td></tr><tr><td>Upper PPO clipping €h</td><td>0.2, except as listed in Table S3</td></tr><tr><td>Reference policy</td><td>Frozen at the pretrained initialization</td></tr><tr><td>Reference KL regularization</td><td>Sampled-token K3 penalty added to the policy loss</td></tr><tr><td>Reference KL coefficient cKL</td><td>0.01, except as listed in Table S3</td></tr><tr><td>Entropy coefficient cH</td><td>0.001, except as listed in Table S3</td></tr><tr><td>Trajectory Collection and Evaluation</td><td></td></tr><tr><td>Student trajectories per task</td><td>G = 8</td></tr><tr><td>Training sampling</td><td>Stochastic; temperature  $\tau _ { \mathrm { g e n } } = 1 . 0$ </td></tr><tr><td>Sampling cutoffs</td><td> ${ \mathrm { T o p } } \cdot p = 1 . 0 ;$  no top-k restriction when sampling</td></tr><tr><td>Repeated final evaluation</td><td>Temperature  $0 . 4 ;$  eight rollout seeds (0–7)</td></tr><tr><td>Response length limit</td><td>512 tokens per interaction turn</td></tr><tr><td>Training steps S</td><td>100</td></tr><tr><td>Validation schedule</td><td>Before training and every 5 steps</td></tr><tr><td>Hardware and Execution</td><td></td></tr><tr><td>Training hardware</td><td>1 node with 8 NVIDIA H800 GPUs</td></tr><tr><td>CPU allocation (Ray)</td><td>64 logical CPUs per run</td></tr><tr><td>Rollout tensor parallelism</td><td>1</td></tr><tr><td>Distributed training</td><td>FSDP; activation checkpointing enabled</td></tr><tr><td>CPU offloading</td><td>Reference parameters only; actor parameters and optimizer</td></tr><tr><td>Rollout inference</td><td>states remain on GPU</td></tr><tr><td>Rollout execution</td><td>BF16; chunked prefill; Triton Gated DeltaNet prefill</td></tr><tr><td>Thinking-mode chat formatting</td><td>CUDA graphs with eager fallback</td></tr></table>

Table S2 Common training and evaluation settings for the eleven methods across all four benchmarks. Benchmarkspecific settings and overrides appear in Table S3; algorithm hyperparameters appear in Table S4.

## C.3 Numerical Precision and Stability

Training uses FP32 actor weights with BF16 mixed-precision computation and FP32 reductions and bufers; reference and rollout weights use BF16. Final-model evaluation uses BF16 weights for both the actor and inference engine.

Numerical stability. We use $\epsilon _ { \mathrm { { n u m } } } = 1 0 ^ { - 8 }$ for masked means and ADRS statistics, and $\epsilon _ { \mathrm { n o r m } } = 1 0 ^ { - 6 }$ for GRPO and AgentOPSD standardization. AgentOPSD clips prior and updated belief probabilities with $\epsilon _ { \mathrm { p r i o r } } = 1 0 ^ { - 4 }$ and $\epsilon _ { \mathrm { b e l } } = 1 0 ^ { - 6 }$ , respectively. OCSD selects turns with absolute advantage above $\epsilon _ { A } = 1 0 ^ { - 6 }$ . OVCSD uses $\epsilon _ { R } = 1 0 ^ { - 6 }$ as the upper threshold on raw episode-return standard deviation for intervention; ADRS uses the same value as the lower threshold on shaped row-return standard deviation for its additional advantage correction. ICSD uses a calibration scale floor of $\epsilon _ { \mathrm { c a l } } = 1 0 ^ { - 4 }$ ; PCSD uses $\epsilon _ { \mathrm { p c } } = 1 0 ^ { - 8 }$ for window statistics and slope normalization. The reference $\mathrm { K 3 }$ estimator is clipped to $[ - c _ { \mathrm { r e f } } , c _ { \mathrm { r e f } } ]$ , with $c _ { \mathrm { r e f } } = 1 0$

<table><tr><td>Parameter</td><td>ALFWorld</td><td>ScienceWorld</td><td>Search-QA</td><td>WebShop</td></tr><tr><td>Benchmark Settings</td><td></td><td></td><td></td><td></td></tr><tr><td>Task batch size B</td><td>16</td><td>16</td><td>64</td><td>16</td></tr><tr><td>Prompt limit (tokens per turn)</td><td>2048</td><td>4096</td><td>4096</td><td>4096</td></tr><tr><td>Maximum interaction turns H</td><td>50</td><td>20</td><td>4</td><td>15</td></tr><tr><td>Retained history (turns)</td><td>2</td><td>2</td><td>4</td><td>2</td></tr><tr><td>PPO minibatch (turns, all GPUs)</td><td>256</td><td>64</td><td>256</td><td>64</td></tr><tr><td>Episode reward</td><td>Success: 10; otherwise: 0</td><td>Final task score /100</td><td>Correct: 1; otherwise: 0</td><td>Success: 10; otherwise: 0</td></tr><tr><td>Invalid-action penalty  $c _ { \mathrm { i n v } }$ </td><td>0.1</td><td>0.1</td><td>0.01</td><td>0.1</td></tr><tr><td>Retrieved passages per query</td><td>N/A</td><td>N/A</td><td>3</td><td>N/A</td></tr><tr><td>Retrieval timeout (s)</td><td>N/A</td><td>N/A</td><td>60</td><td>N/A</td></tr><tr><td>Validation set</td><td>All 140 valid_seen</td><td>200 fixed tasks</td><td>3125 fixed questions</td><td>128 fixed goals</td></tr><tr><td>Validation batch size (tasks) Validation decoding</td><td>games 140</td><td>32</td><td>512</td><td>128</td></tr><tr><td></td><td>Sampling, T = 0.4</td><td>Sampling, T = 0.4</td><td>Greedy, T = 0</td><td>Sampling, T = 0.4</td></tr><tr><td>Method-Specific Optimization Overrides</td><td></td><td></td><td></td><td></td></tr><tr><td>AgentOPSD: upper PPO clipping €h</td><td>0.24</td><td>0.24</td><td>0.24</td><td>0.24</td></tr><tr><td>ADRS: upper PPO clipping €h</td><td>0.28</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>ADRS: entropy coefficient cH</td><td>0.01</td><td>0.001</td><td>0.001</td><td>0.001</td></tr><tr><td>OCSD: warmup ratio</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>OCSD: reference KL cKL</td><td></td><td>0.01</td><td>0.1</td><td>0</td></tr><tr><td>ADRS: reference KL cKL</td><td>0.01</td><td>0.01</td><td>0.001</td><td>0.01</td></tr><tr><td></td><td>0.01</td><td></td><td>0.001</td><td>0.01</td></tr><tr><td>Benchmark-Specific ADRS Parameters</td><td></td><td></td><td></td><td></td></tr><tr><td>Teacher-reward scale η</td><td>0.1</td><td>0.05</td><td>0.05</td><td>0.02</td></tr><tr><td>Teacher-signal normalization</td><td>Batch-wide SD</td><td>Batch-wide SD</td><td>Batch-wide SD</td><td>Per-response SD</td></tr><tr><td>Scale gate input by its SD</td><td>Enabled</td><td>Enabled</td><td>Enabled</td><td>Disabled</td></tr></table>

Table S3 Benchmark-specific settings and method-specific overrides across four benchmarks, shared across the three model sizes. Configuration sources and benchmark extensions are described in Appendix C.1. Unlisted optimization settings retain the defaults in Table S2. SD denotes standard deviation.

## C.4 Validation and Repeated Evaluation

Training validation on ALFWorld, WebShop, and Search-QA follows SDAR’s released settings (Lu et al., 2026); ScienceWorld uses the same stochastic decoding rule as ALFWorld and WebShop. Within each benchmark, all eleven methods use the validation configuration in Table S3. Repeated final evaluation uses the same T = 0.4 sampling protocol on all four benchmarks, with eight rollout seeds (0–7) indexed by task and interaction turn. The 1,056 evaluations of 132 trained models and 96 evaluations of pretrained backbones total 1,152 evaluation sweeps over twelve settings. Across all four benchmarks, validation and final evaluation use fixed task sets shared across methods, model sizes, and evaluation sweeps, retaining every instance, including incomplete final batches. Overall performance averages the per-instance metric over the full evaluation set; task-category and source-specific results use the corresponding subsets. This aggregation gives equal weight to instances rather than to batches or categories.

## C.5 Experimental Records

For all four benchmarks, we retain training configurations, per-step metrics, and repeated-evaluation summaries for the trained and pretrained models. The training records cover 132 runs and 2,772 validation sweeps, with each validation result linked to its original log. Learning curves and computational costs are derived from these logs; final-evaluation means and sample standard deviations are taken from the repeated-evaluation summaries. The analysis scripts generate the tables and figures from these records and verify metric aggregation against the source data.

<table><tr><td>Method</td><td>Parameter group</td><td>Configuration</td></tr><tr><td>GRPO</td><td colspan="2">No additional hyperparameters</td></tr><tr><td> $\mathrm { G R P O + O P S D }$ </td><td>Auxiliary weight and gate</td><td> $\lambda _ { \mathrm { s d } } = 0 . 0 1 ; \beta = 0$  (constant gate  $1 / 2 )$ </td></tr><tr><td>SDAR</td><td>Auxiliary weight and gate</td><td> $\lambda _ { \mathrm { s d } } = 0 . 0 1 ;$  sigmoid slope  $\beta = 5 ;$  detached gate</td></tr><tr><td>Skill-SD</td><td>Auxiliary weight</td><td> $\lambda _ { \mathrm { d i s t } } = 0 . 0 0 1$  , constant throughout training</td></tr><tr><td rowspan="2">RLSD</td><td>Exponent caps</td><td rowspan="2"> $c _ { \mathrm { K 3 } } = 2 0 ; c _ { \mathrm { I S } } = 1 0$   $\lambda _ { s } = \lambda _ { 0 } ( 1 - s / S ) ; \lambda _ { 0 } = 0 . 5 , S = 1 0 0$ </td></tr><tr><td>Mixing schedule</td></tr><tr><td rowspan="3">OCSD</td><td>Multiplier clipping</td><td rowspan="3"> $\delta _ { \mathrm { r } } = 0 . 2$   $L _ { f } = 2$  future interaction turns</td></tr><tr><td>Future context Turn selection</td></tr><tr><td></td></tr><tr><td rowspan="2">OVCSD</td><td>Reweighting</td><td> $\alpha _ { \mathrm { s e l } } = 0 . 2$  Residual scale  $\lambda _ { \mathrm { o c } } = 0 . 5 ;$  advantage threshold  $\epsilon _ { A } = 1 0 ^ { - 6 }$ </td></tr><tr><td>Eligibility and budget</td><td>Reward standard deviation threshold  $\epsilon _ { R } = 1 0 ^ { - 6 } ; A _ { \mathrm { { m a x } } } = 2$  attempts per eligible group</td></tr><tr><td rowspan="2">AgentOPSD</td><td>Suffix distillation</td><td>Top-K plus tail with  $K = 3 2 ; \mathrm { K L }$  weight  $\lambda _ { \mathrm { s u f } } = 0 . 1$  no</td></tr><tr><td>Advantage multiplier</td><td>Signed, turn-level; mixing  $\lambda _ { \mathrm { a g } } = 0 . 5 ;$  band  $\delta _ { \mathrm { a g } } = 0 . 2 ;$  coefficient decay</td></tr><tr><td rowspan="3">ADRS</td><td>Belief update</td><td>Decay  $\gamma = 0 . 9 5 ;$  prior from the clipped mean of trajectory-deduplicated training returns; fallback 0.5</td></tr><tr><td>Teacher-utility gate</td><td>L2 teacher value advantage  $( \mathrm { T V A } )$  over task-grouped interaction rows; alignment temperature  $\tau _ { \mathrm { a l i g n } } = 1 ;$  sigmoid slope  $\beta _ { \mathrm { T V A } } = 2$ </td></tr><tr><td>Reward and advantage shaping Schedule</td><td>Per-turn centering; additive correction clipped at  $\delta _ { \mathrm { a d } } = 0 . 2 ,$  +1</td></tr><tr><td rowspan="2">ICSD</td><td>Auxiliary weight</td><td>Active throughout training, without warmup  $\lambda _ { \mathrm { i c } } = 0 . 0 1$ </td></tr><tr><td>Calibration</td><td>CDF splits  $\xi _ { \mathrm { t r u s t } } = 0 . 4 , \xi _ { \mathrm { i n f } } = 0 . 5 ;$  scale floor  $\epsilon _ { \mathrm { c a l } } = 1 0 ^ { - 4 } \ ;$  tokens</td></tr><tr><td rowspan="3">PCSD</td><td>Auxiliary weight</td><td>minimum pool  $n _ { \mathrm { c a l } } = 8$   $\lambda _ { \mathrm { p c } } = 0 . 0 1$ </td></tr><tr><td>Token windows</td><td> $N _ { \mathrm { m i n } } = 1 , N _ { \mathrm { m a x } } = 8 ;$  decay  $\rho _ { \mathrm { p c } } = 0 . 8$ </td></tr><tr><td>Variance and trend gate</td><td>Variance bounds  $v _ { l } = 0 . 0 5 , v _ { h } = 0 . 5 ;$  trend strength  $\gamma _ { \mathrm { { p c } } } = 0 . 3 ;$  slope  $\beta _ { \mathrm { p c } } = 5$ </td></tr></table>

Table S4 Algorithm-specific hyperparameters shared across all four benchmarks and three model sizes. Sources and implementation choices are described in Appendix C.1; benchmark-dependent values appear in Table S3. Symbols follow Appendix D; s denotes the global training step.

## D Training Objectives and Algorithms

Algorithms S1–S12 specify the eleven methods as evaluated in this study. The ten methods beyond GRPO modify its auxiliary objective, advantage weights, collected trajectories, or reward signal. We define the shared operations once and expose the method-specific changes below. For more detailed method descriptions and derivations, we refer readers to the original papers cited in Table S1.

Data and indices. A trajectory is one task attempt, comprising multiple model–environment interactions. The training batch D contains one row $b$ per active interaction turn; t indexes tokens in that turn’s response. The binary mask $M _ { b t }$ is one for valid response tokens and zero otherwise. The maps $i ( b )$ and $g ( b )$ identify the trajectory and task group of row $b ;$ a task group contains all sampled attempts at the same task. $B _ { i }$ and $B _ { g }$ denote their respective row sets, and $g ( i )$ identifies the task group of trajectory i.

Models and token scores. $\theta _ { 0 }$ denotes the initial model parameters, θ the trainable student, and $\pi _ { \theta }$ its token distribution. The old policy $\theta _ { \mathrm { o l d } }$ is refreshed at each training step and held fixed during the ensuing actor updates; the reference $\phi$ stays at $\theta _ { 0 }$ throughout training. For each sampled response token, $p _ { b t } , ~ o _ { b t } ,$ , and $f _ { b t }$ are its log-probabilities under the current student, old policy, and reference, respectively. The teacher score $q _ { b t }$ is the log-probability of that same token given additional training-time context, such as skills or future observations. These are sampled-token scores, not full-vocabulary distributions, unless stated otherwise.

Operations and gradients. $\operatorname { s g } ( x )$ keeps the value of $x$ but blocks its gradient. Teacher, old-policy, and reference scores, rewards, selection decisions, and constructed advantages are detached; gradients flow through the current student scores $p .$ Products and divisions of token arrays are elementwise, and row quantities broadcast over tokens. We use the sigmoid $\sigma ( x ) = ( 1 + e ^ { - x } ) ^ { - 1 }$ , log-odds logi $( x ) = \log [ x / ( 1 - x ) ]$ ], and $\mathrm { R e L U } ( x ) = \operatorname* { m a x } ( x , 0 )$ $\mathrm { c l i p } ( x , a , b )$ restricts x to $[ a , b ] ; C _ { \epsilon } ( x ) = \mathrm { c l i p } ( x , \epsilon , 1 - \epsilon )$ keeps beliefs away from zero and one. Other intermediate quantities are local to each method. Subscripted ϵ terms denote numerical stabilizers, clipping widths, or thresholds, as specified below; hyperparameter values are listed in Appendix C.

## D.1 Shared Training Procedure

The shared loop in Algorithm S1 selects a task batch X of size B and collects $G$ trajectories per task, forming $\tau .$ . It runs for S training steps, indexed by s, with at most H interaction turns per trajectory and sampling temperature $\tau _ { \mathrm { g e n } }$ . M denotes the selected method, not a token mask.

Rewards and advantages. We assign the complete episode return to the final response token of each interaction turn. The invalid-action penalty subtracts $c _ { \mathrm { i n v } }$ at that token for an invalid turn. Let $r _ { b t }$ be this sparse reward and $\begin{array} { r } { R _ { b } = \sum _ { t } r _ { b t } . \ A _ { b } } \end{array}$ is the scalar GRPO advantage of row $b ,$ broadcast over its valid tokens as $A _ { b t } = A _ { b } M _ { b t }$ Group normalization pools the row returns within each task group:

$$
A _ { b t } = \frac { R _ { b } - \mu _ { g ( b ) } } { s _ { g ( b ) } + \epsilon _ { \mathrm { n o r m } } } M _ { b t } .\tag{1}
$$

Here µ and $\mu _ { g }$ $s _ { g }$ are the mean and sample standard deviation (SD) of $\{ R _ { b } : b \in B _ { g } \}$ ; each trajectory contributes one observation per turn. For a singleton group, we set $\mu _ { g } = 0 , s _ { g } = 1$ . We denote this operation by $\mathcal { G } ( \boldsymbol { r } )$ and retain groups with constant rewards; $\widehat { A }$ denotes any method-specific replacement for A.

Trajectory collection. SampleTasks draws training tasks, Rollout generates skill-free trajectories, and TurnRows unfolds them into $D ,$ retaining prompts, responses, masks, and task, trajectory, and turn IDs. Reward assigns the rewards defined above. Score(θ, D) returns detached log-probabilities for the responses already in $D ;$ it does not generate new responses. The batch $D$ carries the rewards r, row returns $R ,$ and scores $o , f$ computed in the shared loop. MethodStep applies the selected algorithm to these quantities, updating θ while keeping $\theta _ { \mathrm { o l d } }$ and $\phi$ fixed. For OVCSD it calls OVCSDUpdate with both the eligible groups $\mathcal { F }$ and verified pairs V returned during collection.

Common objective. The policy-gradient mask $M ^ { \mathrm { P G } }$ selects tokens for the policy loss; the regularization mask $M ^ { \mathrm { r e g } }$ selects tokens for reference-KL and entropy terms. Both are binary: one includes a token and zero excludes it from the corresponding term. Except for OVCSD, whose masks are specified below, $M ^ { \mathrm { P G } } = M ^ { \mathrm { r e g } } = M$ . For any such mask, $\begin{array} { r } { \langle X \rangle _ { M } = \sum _ { b , t } M _ { b t } X _ { b t } / ( \sum _ { b , t } M _ { b t } + \epsilon _ { \mathrm { n u m } } ) } \end{array}$ denotes its token average within a microbatch.

The current-to-old probability ratio is $\rho = \exp ( p - o )$ . Let u be the usual clipped surrogate, with lower and upper clipping widths $\epsilon _ { l } , \epsilon _ { h }$ . Dual clipping additionally bounds its negative-advantage branch using $c _ { \mathrm { d u a l } } > 1$

$$
\begin{array} { r l } & { u = \operatorname* { m a x } \{ - A \rho , - A \exp ( \rho , 1 - \epsilon _ { l } , 1 + \epsilon _ { h } ) \} , } \\ & { \ell _ { \mathrm { P G } } = \left\{ \begin{array} { l l } { \operatorname* { m i n } ( u , - c _ { \mathrm { d u a l } } A ) , } & { A < 0 , } \\ { u , } & { A \geq 0 . } \end{array} \right. } \end{array}\tag{2}
$$

The common loss combines this policy-gradient term $\ell _ { \mathrm { P G } }$ , a reference penalty $k _ { \mathrm { r e f } }$ weighted by $c _ { \mathrm { K L } }$ , and token-distribution entropy $\mathcal { H } ( \pi _ { \theta } )$ weighted by $c _ { H }$ . The constant $c _ { \mathrm { r e f } }$ caps the sampled-token KL penalty:

$$
\begin{array} { r l } & { \mathcal { L } _ { 0 } ( A ) = \langle \ell _ { \mathrm { P G } } \rangle _ { M ^ { \mathrm { P G } } } + c _ { \mathrm { K L } } \langle k _ { \mathrm { r e f } } \rangle _ { M ^ { \mathrm { r e g } } } } \\ & { ~ - c _ { H } \langle \mathcal { H } ( \pi _ { \theta } ) \rangle _ { M ^ { \mathrm { r e g } } } , } \\ & { k _ { \mathrm { r e f } } = \mathrm { c l i p } \big ( e ^ { f - p } - ( f - p ) - 1 , - c _ { \mathrm { r e f } } , c _ { \mathrm { r e f } } \big ) . } \end{array}\tag{3}
$$

Here $k _ { \mathrm { r e f } }$ is the clipped sampled-token K3 estimator of reference KL; H denotes entropy, whereas H is the interaction horizon. The actor accumulates microbatch losses, clips the gradient norm at $c _ { \mathrm { g r a d } } .$ and skips an optimizer update if the norm is nonfinite. $\mathrm { U p d a t e } _ { D } ( A , L _ { \mathrm { a u x } } )$ minimizes $\mathcal { L } _ { 0 } ( A ) + L _ { \mathrm { a u x } }$ on $D$ using AdamW over $N _ { \mathrm { e p } }$ PPO epochs. The auxiliary loss $L _ { \mathrm { a u x } }$ is recomputed from the current student at each forward pass. GRPO (Algorithm S2) sets this term to zero, retaining the reference-KL and entropy terms. The update changes θ in place; minibatching and gradient accumulation follow the common settings.

Skill-conditioned scoring. SkillReplay $( D , \psi )$ returns teacher scores $q \colon$ it augments prompts with retrieved skills, keeps the sampled student responses, and scores them without gradients under model parameters ψ. Its outputs remain fixed throughout the subsequent actor update. Most skill-replay methods use the pre-update actor, $\begin{array} { r } { \psi = \theta _ { \mathrm { o l d } } ; } \end{array}$ PCSD instead uses the fixed reference, $\psi = \phi$ . Teacher prompts are left-truncated as described in Appendix C.

## D.2 Auxiliary Distillation Objectives

GRPO+OPSD’s constant-gate objective. GRPO+OPSD and SDAR weight their auxiliary loss by $\lambda _ { \mathrm { s d } }$ and control the sigmoid gate’s slope with $\beta .$ In our comparison, GRPO+OPSD (Algorithm S3) sets $\beta = 0$ yielding a detached gate of $1 / 2 .$ For finite log-probabilities its auxiliary gradient is therefore exactly that of $- ( \lambda _ { \mathrm { s d } } / 2 ) \langle p \rangle _ { M } ;$ teacher values afect the loss value but not this gradient. The common reference penalty in Eq. 3 is present in both GRPO and GRPO+OPSD and is separate from this auxiliary term

SDAR and Skill-SD supervision. In SDAR (Algorithm S4), the gate $g ( p )$ weights each valid response token according to the teacher–current-student log-probability gap, including tokens in unsuccessful trajectories; the gate is detached. Skill-SD (Algorithm S5) instead uses the log-probability diference $d ( p ) = p - q$ to construct a K3 term $k ( p )$ , weighted by the importance ratio $w ( p )$ and auxiliary coeficient $\lambda _ { \mathrm { d i s t } }$ . The constants $c _ { \mathrm { K 3 } }$ and c<sub>IS</sub> cap the two exponent inputs. Gradients pass through both $k ( p )$ and $w ( p )$ , but not through teacher or old-policy scores. Both objectives supervise the sampled response tokens.

ICSD calibration and gradient boundaries. ICSD (Algorithm S6) first scores the current student on a local actor minibatch $D ^ { \prime }$ to obtain detached scores $p ^ { \star }$ . Its teacher gap $\Delta = q - p ^ { \star }$ and influence score $u = A \exp ( p ^ { \star } - o ) \Delta$ are calibrated into trust weights g and influence weights h. $\mathrm { C a l C D F } _ { \xi }$ performs this calibration using a cumulative distribution function (CDF), pooling valid tokens with the same interaction-turn index within $D ^ { \prime } .$ For each pool it fits location m and scales $a _ { - } , a _ { + }$ , then applies

```latex
Algorithm S1 Shared training loop
Require: $\theta _ { 0 } , \mathcal { M } ; B , G , S , H , \tau _ { \mathrm { g e n } }$
Ensure: Trained student parameters θ
1: $\theta , \phi  \theta _ { 0 } , \theta _ { 0 }$
2: for $s = 1 , \ldots , S$ do
3: $\theta _ { \mathrm { o l d } }  \theta ; \mathcal { F } , \mathcal { V }  \emptyset , \mathcal { O }$
4: X ← SampleTasks(B)
5: $\mathcal { T } \gets \mathrm { R o l l o u t } ( \theta _ { \mathrm { o l d } } , X ; G , H , \tau _ { \mathrm { g e n } } )$
6: if M = OVCSD then
7: (F, V) ← VerifyCont(T )
8: end if
9: D ← TurnRows(T); r ← Reward(D)
10: $o \gets \mathrm { S c o r e } ( \theta _ { \mathrm { o l d } } , D )$
11: $f  \mathrm { S c o r e } ( \phi , D ) ; A  \mathcal { G } ( r )$
12: $\operatorname { M e t h o d S t e p } _ { \mathcal { M } } ( D , A , s , \mathcal { F } , \mathcal { V } )$
13: end for
14: return θ
Algorithm S2 GRPO
Require: $D , A$
1: Update<sub>D</sub>(A, 0)
Algorithm S3 GRPO+OPSD: constant gate
Require: $D , A ; \lambda _ { \mathrm { s d } }$
1: $q \gets \mathrm { S k i l l R e p l a y } ( D , \theta _ { \mathrm { o l d } } )$
2: $\begin{array} { r } { L _ { \mathrm { a u x } } ( p ) \gets \frac { \lambda _ { \mathrm { s d } } } { 2 } \langle q - p \rangle _ { M } } \end{array}$
3: $\mathrm { U p d a t e } _ { D } ( A , L _ { \mathrm { a u x } } )$
```

Algorithm S4 SDAR: detached confidence gate   
Require: $D , A ; \lambda _ { \mathrm { s d } } , \beta$   
1 $: q \gets \mathrm { S k i l l R e p l a y } ( D , \theta _ { \mathrm { o l d } } )$   
2: $g ( p ) \gets \mathrm { s g } \{ \sigma [ \beta ( q - p ) ] \}$   
3: $L _ { \mathrm { a u x } } ( p ) \gets \lambda _ { \mathrm { s d } } \langle g ( p ) ( q - p ) \rangle$ M   
4: $\mathrm { U p d a t e } _ { D } ( A , L _ { \mathrm { a u x } } )$   
Algorithm S5 Skill-SD: importance-weighted K3   
Require: $D , A ; \lambda _ { \mathrm { d i s t } } , c _ { \mathrm { K 3 } } , c _ { \mathrm { I S } }$   
1: $q \gets \mathrm { S k i l l R e p l a y } ( D , \theta _ { \mathrm { o l d } } )$   
2: $d ( p ) \gets p - q$   
3: $k ( p ) \gets \exp ( \operatorname* { m i n } ( - d ( p ) , c _ { \mathrm { K } 3 } ) ) - 1 + d ( p )$   
4: $w ( p ) \gets \exp ( \operatorname* { m i n } ( p - o , c _ { \mathrm { I S } } ) )$   
5: $L _ { \mathrm { a u x } } ( p ) \gets \lambda _ { \mathrm { d i s t } } \langle w ( p ) k ( p ) \rangle _ { M }$   
6: $\mathrm { U p d a t e } _ { D } ( A , L _ { \mathrm { a u x } } )$

$$
F _ { \xi } ( x ) = \left\{ \begin{array} { l l } { \xi e ^ { ( x - m ) / a _ { - } } , } & { x < m , } \\ { 1 - ( 1 - \xi ) e ^ { - ( x - m ) / a _ { + } } , } & { x \geq m . } \end{array} \right.
$$

The split $\xi \in ( 0 , 1 )$ sets $F _ { \xi } ( m ) = \xi ; \xi _ { \mathrm { t r u s t } }$ and $\xi _ { \mathrm { i n f } }$ specify the trust and influence calibrations. The location is the lower sample median; the scales are mean absolute deviations below and at or above m, floored at $\epsilon _ { \mathrm { c a l } }$ (also used for an empty side). Pools with fewer than $n _ { \mathrm { c a l } }$ tokens or nonfinite values fall back to all valid minibatch tokens, then to location zero and unit scales if still invalid. The final token weights c combine g and h within the eligible-token set $C _ { b } .$ , preserving each response’s total trust weight; other tokens retain $g .$ The normalizer $z _ { b }$ is the sum of $g h$ over $C _ { b }$ . The coeficient $\lambda _ { \mathrm { i c } }$ scales the auxiliary loss, and c remains fixed during that minibatch’s gradient computation. UpdateMiniBatch $\mathbf { \nabla } \cdot D ^ { \prime }$ applies one accumulated-gradient update on $D ^ { \prime } { \mathrm { ; } }$ it does not repeat the outer PPO epochs.

PCSD windows and teacher. PCSD (Algorithm S7) smooths the detached teacher gap $\Delta = q - p$ over short and long forward windows of lengths $N _ { \mathrm { m i n } }$ and $N _ { \mathrm { m a x } }$ . All windows start at the current token and stay within its response row, excluding padding via M. DecayMean weights ofset $j$ by $\rho _ { \mathrm { p c } } ^ { j } ,$ yielding means $m _ { s } , m _ { l }$

Algorithm S6 ICSD: calibrated token weights   
Require: D, A; λ<sub>ic</sub>, ξ<sub>trust</sub>, ξ<sub>inf</sub>, ϵ<sub>cal</sub>, n<sub>cal</sub>   
1: q ← SkillReplay(D, θ<sub>old</sub>)   
2: for each local actor minibatch $D ^ { \prime }$ in the PPO epochs do   
3: Use the $q , o , A , M$ slices for $D ^ { \prime }$   
4: $p ^ { \star } \gets \mathrm { S c o r e } ( \theta , D ^ { \prime } )$   
5: $\Delta  q - p ^ { \star } ; u  A \exp ( p ^ { \star } - o ) \Delta$   
6: $g \gets \mathrm { C a l C D F } _ { \xi _ { \mathrm { t r u s t } } } ( \Delta )$   
7: $h  \mathrm { C a l C D F } _ { \xi _ { \mathrm { i n f } } } ( u )$   
8: $c \gets g M$   
9: for each response row b do   
10: $C _ { b } \gets \{ t : M _ { b t } = 1 , \neg ( A _ { b t } < 0 \land \Delta _ { b t } < 0 ) \}$   
11: $\begin{array} { r } { z _ { b } \gets \sum _ { t \in C _ { b } } g _ { b t } h _ { b t } } \end{array}$   
12: if $z _ { b } > \epsilon _ { \mathrm { c a l } }$ then   
13: $\begin{array} { r } { c _ { b t } \gets g _ { b t } h _ { b t } \sum _ { j \in C _ { b } } g _ { b j } / z _ { b } , \quad t \in C _ { b } } \end{array}$   
14: end if   
15: end for   
16: $L _ { \mathrm { a u x } } ( p ) \gets \lambda _ { \mathrm { i c } } \langle \mathrm { s g } ( c ) ( q - p ) \rangle _ { M }$   
17: UpdateMiniBatch $_ { \cdot D ^ { \prime } } ( A , L _ { \mathrm { a u x } } )$   
18: end for

The long-window population variance v, computed by WindowVar, sets the interpolation weight a between variance bounds $v _ { l } , v _ { h } ; \bar { \Delta }$ is the resulting smoothed gap. WindowSlope fits an unweighted least-squares slope against token ofsets, adds $\epsilon _ { \mathrm { p c } }$ to its denominator, and returns zero for fewer than two valid tokens. Its slope ℓ is scaled by the row’s mean absolute gap $d _ { b } ;$ the factor h suppresses declining support with strength $\gamma _ { \mathrm { p c } } .$ The final detached weight w combines h with a sigmoid of slope $\beta _ { \mathrm { p c } } ,$ and $\lambda _ { \mathrm { p c } }$ scales the auxiliary loss. The skill-conditioned teacher uses fixed reference weights throughout training; only the current student receives gradients.

## D.3 Advantage Reweighting

RLSD, OCSD, and AgentOPSD replace A by detached advantages $\widehat { A }$ before the common update. Their multipliers preserve advantage signs and leave zero advantages unchanged under the evaluated coeficient ranges.

RLSD mixing. RLSD (Algorithm S8) mixes the original advantage with a teacher-reweighted version using $\lambda _ { s } ;$ $\lambda _ { 0 }$ sets the schedule’s initial scale and $\delta _ { \mathrm { r } }$ bounds the token multiplier w around one. For the one-based step $s , \lambda _ { s } = \lambda _ { 0 } \operatorname* { m a x } ( 1 - s / S , 0 )$ , so it is $\lambda _ { 0 } ( 1 - 1 / S )$ at the first step and zero at the last; teacher scores are still computed at that endpoint.

OCSD evidence selection. In OCSD (Algorithm S9), $F _ { b }$ counts the future observations available to row $b ,$ and ${ \mathrm { T o p } } _ { k } ( E ; n )$ selects the k rows in E with the largest mean negative log-likelihood (NLL) n. For trajectory i, $E _ { i }$ contains eligible turns with future observations and $| A _ { b } | > \epsilon _ { A } ; J _ { i }$ contains the highest-NLL fraction $\alpha _ { \mathrm { s e l } }$ , with at least one turn if $E _ { i }$ is nonempty. EvidencePair keeps the sampled response and constructs full/ablated prompts from up to $L _ { f }$ future observations: intermediate action schemas are identical, but observations are withheld in the ablated prompt; both prompts are left-truncated. $q ^ { \mathrm { f u l l } }$ and $q ^ { \mathrm { a b l } }$ score these two contexts; their bounded contrast z modulates advantages with strength $\lambda _ { \mathrm { o c } }$

AgentOPSD belief updates. In AgentOPSD (Algorithm S10), $V _ { 0 , g }$ is a task group’s prior belief, obtained by averaging trajectory-deduplicated returns and clipping to (0, 1). FirstRows supplies one representative row per trajectory by retaining its first occurrence in the batch. The accumulator c sums turn-level teacher–old-policy log-ratios with decay $\gamma ;$ it updates the success belief $V _ { b } .$ whose increment is $v _ { b } = V _ { b } - V _ { \mathrm { p r e v } }$ . The mean $\bar { v } _ { i }$ and population SD over $B _ { i }$ standardize these increments into $z _ { b } ;$ a one-turn trajectory has zero SD. The bounded multiplier $m _ { b }$ uses band $\delta _ { \mathrm { a g } } ,$ and $\lambda _ { \mathrm { a g } }$ mixes it with the original advantage. The prior deduplicates trajectories, while the base GRPO advantage still uses turn-row statistics.

Algorithm S7 PCSD: persistent teacher-support weighting   
Require: D, A; λ<sub>pc</sub>, N<sub>min</sub>, N<sub>max</sub>, ρ<sub>pc</sub>, v<sub>l</sub>, v<sub>h</sub>, γ<sub>pc</sub>, β<sub>pc</sub>   
1: q ← SkillReplay(D, ϕ)   
2: function $L _ { \mathrm { a u x } } ( p )$   
3: ∆ ← sg(q − p)M   
4: m<sub>s</sub> ← DecayMean(∆, M; N<sub>min</sub>, ρ<sub>pc</sub>)   
5: m<sub>l</sub> ← DecayMean(∆, M; N<sub>max</sub>, ρ<sub>pc</sub>)   
6: v ← WindowVar(∆, M; N<sub>max</sub>)   
7: a ← clip((v − v<sub>l</sub>)/(v<sub>h</sub> − v<sub>l</sub>), 0, 1)   
8: ∆<sup>¯</sup> ← (1 − a)m<sub>s</sub> + am<sub>l</sub>   
9: ℓ ← WindowSlope $\left( \Delta , M ; N _ { \mathrm { m a x } } \right)$   
10: $d _ { b } \gets \sum _ { t } M _ { b t } | \Delta _ { b t } | / \operatorname* { m a x } ( 1 , \sum _ { t } M _ { b t } ) + \epsilon _ { \mathrm { p c } }$   
11: $h \gets \mathrm { c l i p ( 1 - \gamma _ { \mathrm { p c } } R e L U ( - } \ell / d ) , 0 , 1 )$   
12: w ← sg{σ(β<sub>pc</sub>∆)<sup>¯</sup> hM}   
13: return $\lambda _ { \mathrm { p c } } \langle w ( q - p ) \rangle _ { M }$   
14: end function   
15: Update (A, L<sub>aux</sub>)

Algorithm S8 RLSD: sign-aware advantage reweighting   
Require: D, A, s; λ , S, δ   
1: q ← SkillReplay(D, θ )   
2: λ ← λ max(1 − s/S, 0)   
3: w ← clip(e<sup>sign(A)(q−o)</sup>, 1 − δ<sub>r</sub>, 1 + δ<sub>r</sub>)   
4: $\widehat { A } \gets \mathrm { s g } \{ A [ ( 1 - \lambda _ { s } ) + \lambda _ { s } w ] \}$   
5: Update (A,b 0)

Algorithm S9 OCSD: full/ablated future-evidence replay   
Require: $D , A , o , F _ { b } ; \alpha _ { \mathrm { s e l } } , L _ { f } , \lambda _ { \mathrm { o c } }$   
$\textstyle 1 \colon \ z \gets 0$   
2: for trajectory i do   
3: $E _ { i }  \{ b \in B _ { i } : F _ { b } > 0 , ~ | A _ { b } | > \epsilon _ { A } \}$   
4: if $E _ { i } = \mathcal { D }$ then   
5: continue   
6: end if   
P M<sub>bt</sub>o<sub>bt</sub>   
7: $n _ { b } \gets - \frac { \angle t } { \operatorname* { m a x } ( 1 , \sum _ { t } M _ { b t } ) }$ for $b \in E _ { i }$   
8: k<sub>i</sub> ← min(|E<sub>i</sub>|, max(1, ⌈α<sub>sel</sub>|E<sub>i</sub>|⌉))   
9: $J _ { i } \gets \mathrm { T o p } _ { k _ { i } } ( E _ { i } ; n )$   
10: for $b \in J _ { i }$ do   
11: $( D _ { e } ^ { + } , D _ { b } ^ { - } ) \gets \mathrm { E v i d e n c e P a i r } ( D _ { b } , L _ { f } )$   
12: $\mathring { q } _ { b } ^ { \mathrm { f u } \tilde { 1 } }  \mathrm { S c o r e } ( \theta _ { \mathrm { o l d } } , D _ { b } ^ { + } )$   
13: $q _ { b } ^ { \mathrm { a b l } }  \mathrm { S c o r e } ( \theta _ { \mathrm { o l d } } , D _ { b } ^ { - } )$   
14: $z _ { b t }  \mathrm { t a n h } [ ( q _ { b t } ^ { \mathrm { f u l l } } - q _ { b t } ^ { \mathrm { a b l } } ) / 2 ]$   
15: end for   
16: end for   
17: $\widehat { A } \gets \operatorname { s g } \{ A [ 1 + \lambda _ { \mathrm { o c } } \operatorname { s i g n } ( A ) z ] \}$   
18: Update ${ \mathfrak { \sigma } } _ { D } ( { \widehat { A } } , 0 )$

## D.4 Verified Teacher Continuations

Teacher continuations. OVCSD (Algorithm S11) makes at most $A _ { \mathrm { m a x } }$ teacher-continuation attempts per eligible group; $\mathcal { F }$ collects these groups and V stores the verified comparison pairs. EligibleGroups selects groups with no successful trajectory, raw episode-return population SD at most $\epsilon _ { R } ,$ and at least one candidate prefix. Prefixes builds a candidate set C by grouping trajectories with the same action prefix and current

Algorithm S10 AgentOPSD: signed turn credit   
Require: $D , A , R ; \gamma , \lambda _ { \mathrm { a g } } , \delta _ { \mathrm { a g } }$   
1: $q \gets \mathrm { S k i l l R e p l a y } ( D , \theta _ { \mathrm { o l d } } )$   
2: for task group g do   
3: $I _ { g } \gets \mathrm { F i r s t R o w s } ( B _ { g } , i )$   
4: $V _ { 0 , g }  C _ { \epsilon _ { \mathrm { p r i o r } } } ( \mathrm { m e a n } _ { b \in I _ { g } } R _ { b } )$   
5: end for   
6: for trajectory i do   
7: $c  0 ; V _ { \mathrm { p r e v } }  V _ { 0 , g ( i ) }$   
8: for $b \in B _ { i }$ in turn order do   
9: $c  \gamma c + \textstyle \sum _ { t } M _ { b t } ( q _ { b t } - o _ { b t } )$   
10: $V _ { b }  C _ { \epsilon _ { \mathrm { b e l } } } \overline { { ( \sigma } } [ \mathrm { l o g i t } ( V _ { 0 , g ( i ) } ) + c ] )$   
11: $v _ { b }  V _ { b } - V _ { \mathrm { p r e v } } ; V _ { \mathrm { p r e v } }  V _ { b }$   
12: end for   
13: for $b \in B _ { i }$ do   
14: $z _ { b } \gets ( v _ { b } - \bar { v } _ { i } ) / ( \mathrm { S D } _ { \mathrm { p o p } } ( v _ { B _ { i } } ) + \epsilon _ { \mathrm { n o r m } } )$   
15: $m _ { b } \gets 1 + \delta _ { \mathrm { a g } } \mathrm { c l i p } ( \mathrm { s i g n } ( A _ { b } ) z _ { b } , - 1 , 1 )$   
16: $\widehat { A } _ { b t }  A _ { b } [ ( 1 - \lambda _ { \mathrm { a g } } ) + \lambda _ { \mathrm { a g } } m _ { b } ] M _ { b t }$   
17: end for   
18: end for   
19: Update (sg(Ab), 0)

Algorithm S11 OVCSD: verified continuations   
Require: $A _ { \operatorname* { m a x } } , \epsilon _ { R } , K , \lambda _ { \operatorname* { s u f } }$   
1: function $\mathrm { V E R I F Y C O N T } ( { \mathcal { T } } )$   
2: F ← EligibleGroups $( \mathcal { T } ; \epsilon _ { R } )$   
3: $\nu  \varnothing$   
4: for $g \in { \mathcal { F } }$ do   
5: U ← Members ${ \mathfrak { \Gamma } } ( T _ { g } )$   
6: $C \gets \mathrm { P r e f i x e s } ( \mathcal { T } _ { g } ) ; W \gets \emptyset$   
7: c ← Deepest $( C , { \cal U } , W )$   
8: for $a = 1 , \dots , A _ { \mathrm { m a x } }$ do   
9: if $c = \emptyset$ or $U = \emptyset$ then   
10: break   
11: end if   
12: $( \tau ^ { + } , v ) \gets \mathrm { R e p l a y C o n t i n u e } ( \theta _ { \mathrm { o l d } } , c , H )$   
13: if $v = 1$ then   
14: P ← AlignPairs $( \mathcal { T } _ { g } , \tau ^ { + } , c )$   
15: $( a _ { j } ^ { + } , a _ { j } ^ { - } )  ( \sqrt { m _ { j } } , - 1 / \sqrt { m _ { j } } ) , j \in \mathcal { P }$   
16: $\mathcal { V }  \mathcal { \dot { V } } \cup \mathcal { P } ; \dot { U }  U \setminus$ Members(c)   
17: end if   
18: $W  W \cup \{ \mathrm { k e y } ( c ) \}$   
19: $c \gets \mathrm { N e x t } ( C , U , W , c , v )$   
20: end for   
21: end for   
22: return $( \mathcal { F } , \mathcal { V } )$   
23: end function   
24: procedure $\operatorname { O V C S D U P D A T E } ( D , A , \mathcal { F } , \mathcal { V } )$   
25: $( \widehat { D } , \widehat { A } ) \gets \mathrm { V e r i f i e d B a t c h } ( D , A , \mathcal { F } , \mathcal { V } )$   
26: $( I , Q ) \gets \operatorname { T o p T a i l } _ { K } ( \theta _ { \mathrm { o l d } } , \widehat { D } ^ { + } )$   
27: $\mathcal { I }  \{ j \in \mathcal { V } : | S _ { j } | > 0 \}$   
28: $L _ { \operatorname { s u f } } ( \theta ) \longleftarrow \lambda _ { \operatorname { s u f } } \sum _ { j \in \mathcal { I } } \frac { 1 } { | S _ { j } | } \sum _ { ( b , t ) \in S _ { j } } \operatorname { K L } ( Q _ { b t } | | P _ { \theta , b t } ^ { I } )$   
29: $\mathrm { U p d a t e } _ { \widehat { D } } ( \widehat { A } , L _ { \mathrm { s u f } } )$   
30: end procedure

observation, retaining states with a subsequent student action; c denotes the selected candidate. U contains uncovered trajectories; W records attempted candidate keys, distinguishing shared candidates from individual fallbacks. Members returns the trajectory IDs represented by a group or candidate. Deepest prefers an untried shared non-root prefix, then the deepest individual prefix. ReplayContinue restores the candidate state and verifies its observation before sampling a skill-conditioned continuation $\tau ^ { + }$ within the remaining horizon; its indicator v is one only if replay and task completion both succeed. AlignPairs returns pairs $\mathcal { P }$ at the first action divergence while observations remain aligned. Each pair $j$ matches one teacher turn to $m _ { j } \geq 1$ failed student turns, receiving comparative advantages $a _ { j } ^ { + }$ and $a _ { j } ^ { - }$ , respectively. After success, Next selects the deepest uncovered candidate; after failure it prefers the nearest untried ancestor, falling back to the deepest candidate. Candidate members are removed from U on success even if no aligned pair survives.

OVCSD optimization masks. VerifiedBatch augments the original batch with verified teacher responses, producing $\hat { D }$ , masks $M ^ { \mathrm { P G } } , M ^ { \mathrm { r e g } }$ , and advantages ${ \widehat { A } } .$ Non-intervened groups keep their ordinary GRPO update. For intervened failed groups, the original policy-gradient mask is cleared except at accepted divergent student turns; the original reference-KL and entropy masks remain active. Appended skill-free teacher divergence rows receive the positive comparative advantage and both policy-gradient and regularization masks. Later verified teacher rows receive only the sufix distillation loss, not policy-gradient, reference- $\cdot \mathrm { K L } ,$ or entropy terms. Old-policy and reference probabilities on appended rows are recomputed using their skill-free prompts.

Suffix distillation. The sufix consists of teacher turns strictly after the divergence; $S _ { j }$ is pair $j ^ { \circ } \mathrm { s }$ set of valid sufix tokens, and $\mathcal { I }$ indexes pairs with nonempty sufixes. ${ \widehat { D } } ^ { + }$ contains their privileged teacher prompts, whereas $\widehat { D }$ uses skill-free student prompts for the same responses. $\mathrm { T o p T a i l } _ { K }$ retains the teacher’s K most probable token IDs I at each position and combines the remaining vocabulary into one tail bin, yielding a fixed $( K + 1 )$ -category distribution Q. $P _ { \theta , b t } ^ { I }$ is the current student’s distribution on that same partition under the skill-free prompt, recomputed at every actor forward pass. The sufix objective sums per-pair token-averaged $\mathrm { K L } ( Q \| P _ { \theta } ^ { I } )$ , weighted by $\lambda _ { \mathrm { s u f } } ;$ its denominators $| S _ { j } |$ are fixed over the complete pairs when accumulating microbatch contributions. Empty sufixes contribute zero; their divergence updates remain active.

## D.5 Teacher-Based Reward Shaping

Teacher signals and gating. ADRS (Algorithm S12) turns teacher scores into token rewards h and an additional advantage correction w. The row mean $\bar { q } _ { b }$ measures teacher alignment with the sampled response; $\alpha _ { b } \in ( 0 , 1 )$ expresses this alignment relative to the batch, using temperature $\tau _ { \mathrm { a l i g n } }$ . The group-level teacher value advantage $( \mathrm { T V A } ) , T _ { g } ,$ , contrasts returns weighted by α and 1 − α using

$$
\mathrm { W M } _ { g } ( R ; \omega ) = \frac { \sum _ { b \in B _ { g } } \omega _ { b } R _ { b } } { \sum _ { b \in B _ { g } } \omega _ { b } + \epsilon _ { \mathrm { n u m } } } .
$$

Here ω denotes the chosen weights for the task-group mean. The coeficient η sets the teacher-signal strength, and $\beta _ { \mathrm { T V A } }$ controls the sigmoid gate’s slope.

Normalization and advantage correction. The centered teacher scores are normalized into $u = N _ { \mathrm { t a s k } } ( ( q - \bar { q } ) M )$ $N _ { \mathrm { t a s k } }$ uses valid tokens to compute the batch-wide sample SD on ALFWorld, ScienceWorld, and Search-QA, or per-row population SD on WebShop, then divides by that SD plus $\epsilon _ { \mathrm { n u m } }$ . TVA is broadcast to rows as $\begin{array} { r } { \dot { T _ { b } } = T _ { g ( b ) } } \end{array}$ and rescaled by $\Gamma _ { \mathrm { t a s k } } \colon$ on ALFWorld, ScienceWorld, and Search-QA, divide by its sample SD plus ϵ<sub>num</sub> when that SD exceeds $\epsilon _ { \mathrm { { n u m } } } ;$ otherwise, and on WebShop, leave it unchanged.

ADRS first adds the gated reward h to r and recomputes GRPO advantages, then adds the within-group correction w, clipped to $[ - \delta _ { \mathrm { a d } } , \delta _ { \mathrm { a d } } ]$ . Moments returns the masked mean $\bar { h } _ { g }$ and unbiased variance $v _ { g }$ of the teacher rewards; $s _ { g } ( r + h )$ is the sample SD of the shaped row returns and determines whether the correction is applied. All reward, TVA, and advantage calculations are detached. The benchmark-specific normalization choices are listed in Table S3.

Algorithm S12 ADRS: teacher-gated reward and advantage shaping   
Require: $D , r ; \eta , \tau _ { \mathrm { a l i g n } } , \beta _ { \mathrm { T V A } } , \delta _ { \mathrm { a d } }$   
1: q ← SkillReplay(D, θ<sub>old</sub>)   
2: q¯<sub>b</sub> ← mean<sub>t</sub> ${ \mathrm { : } } M _ { b t } { = } 1 { \mathrm { ~ } } q b t$   
3: $\iota  N _ { \mathrm { t a s k } } ( ( q - \bar { q } ) M ) ; T _ { g }  0$ for all g   
4: $\alpha _ { b }  \sigma [ ( \bar { q } _ { b } - \mathrm { m e a n } _ { b } \bar { q } _ { b } ) / ( \tau _ { \mathrm { a l i g n } } + \epsilon _ { \mathrm { n u m } } ) ]$   
5: for $g : | B _ { g } | > 1$ do   
6: $T _ { g } \gets \mathrm { \ddot { W M } } _ { g } ( R ; \alpha ) - \mathrm { W M } _ { g } ( R ; 1 - \alpha )$   
7: end for   
8: $\widetilde { T _ { b } } \gets T _ { g ( b ) } ; \widetilde { T } \gets \Gamma _ { \mathrm { t a s k } } ( \widetilde { T } )$   
9: $h _ { b t }  \eta \sigma ( \beta _ { \mathrm { T V A } } \widetilde { T } _ { b } ) u _ { b t }$   
10: $\widehat { A } \gets \mathcal G ( r + h )$   
11: for $g _ { - } : | B _ { g } | > 1 , \ s _ { g } ( r + h ) \ge \epsilon _ { R }$ do   
12: $( \bar { h } _ { g } , v _ { g } )  \mathrm { M o m e n t s } ( h _ { B _ { g } } ; M _ { B _ { g } } )$   
13: $w _ { b t }  \mathrm { c l i p } ( \eta \frac { h _ { b t } - \bar { h } _ { g } } { \sqrt { v _ { g } + \epsilon _ { \mathrm { n u m } } } } , - \delta _ { \mathrm { a d } } , \delta _ { \mathrm { a d } } ) , \ b \in B _ { g }$   
14: $\widehat { A } _ { b t }  \widehat { A } _ { b t } \dot { + } w _ { b t } M _ { b t }$   
15: end for   
16: Updat ${ \mathfrak { s } } _ { D } ( \operatorname { s g } ( { \widehat { A } } ) , 0 )$

<table><tr><td>Method</td><td>Mean</td><td>∆  $\mathrm { W / L / T }$ </td><td>Worst</td></tr><tr><td>PCSD</td><td>54.26</td><td>+4.58  $6 / 5 / 1$ </td><td>-8.31</td></tr><tr><td>AgentOPSD</td><td>51.34</td><td>+1.66  $6 / 5 / 1$ </td><td>-16.51</td></tr><tr><td>GRPO</td><td>49.68</td><td>+0.00  $0 / 0 / 1 2$ </td><td>+0.00</td></tr><tr><td>OCSD</td><td>49.50</td><td>-0.18  $6 / 6 / 0$ </td><td>-29.73</td></tr><tr><td>RLSD</td><td>46.54</td><td>-3.14  $6 / 6 / 0$ </td><td>-70.02</td></tr><tr><td>SDAR</td><td>46.33</td><td>-3.35  $3 / 8 / 1$ </td><td>-18.30</td></tr><tr><td>OVCSD</td><td>46.17</td><td>-3.51  $5 / 7 / 0$ </td><td>-69.73</td></tr><tr><td>ICSD</td><td>44.75</td><td>-4.93  $7 / 4 / 1$ </td><td>-66.02</td></tr><tr><td>ADRS</td><td>44.61</td><td>-5.07  $3 / 8 / 1$ </td><td>-23.36</td></tr><tr><td>GRPO+OPSD</td><td>43.05</td><td>-6.63  $6 / 5 / 1$ </td><td>-80.53</td></tr><tr><td>Skill-SD</td><td>39.45</td><td>-10.23  $5 / 6 / 1$ </td><td>-70.02</td></tr></table>

Table S5 Equal-weight means across all twelve model–benchmark settings, using ALFWorld/WebShop success (%), Search-QA exact-match accuracy (%), and ScienceWorld task score. $\Delta$ and Worst are the mean and worst paired diferences from GRPO; W/L/T counts wins/losses/ties of reported means.

![](images/1a4fcbb005e04d12cf2a975cbf0a370dc5c9bfc799d4320fce04e56bdb739cee.jpg)

(b) Macro trajectories  
![](images/32eed72aace04d13ea1de9ff1ddd82f01179fb2db1b01274d4f71913c4392075.jpg)  
Figure S1 (a) Rankings across all twelve settings by mean training-curve AUC and eight-seed final performance from Tables 2 and 3. AUC integrates each benchmark’s validation metric and normalizes by the training horizon. (b) Equally weighted mean validation trajectories, highlighting GRPO and the AUC and final-performance leaders, OCSD and PCSD; the other eight methods are shown in gray.

## E Cross-Setting Gains and Regressions

## E.1 All Twelve Settings

Table S5 combines the sampled final evaluations on all four benchmarks, assigning equal weight to each model–benchmark setting. It uses success for ALFWorld/WebShop, exact-match accuracy for Search-QA, and task score for ScienceWorld. Only PCSD and AgentOPSD improve the twelve-setting mean over GRPO, by 4.58 and 1.66 points, respectively. ICSD wins seven settings, including all three Search-QA scales, but its 66.02-point deficit on 4B WebShop and 40.73-point deficit on 9B ScienceWorld leave its overall mean 4.93 points below GRPO. OCSD likewise illustrates scale-dependent gains: it improves on GRPO in 2B and 4B ScienceWorld but trails it by 14.69 points at 9B, and its twelve-setting mean remains 0.18 points below GRPO. Win counts and gaps are descriptive comparisons of means; rollout-seed SDs do not support conclusions about variability across independent training runs.

![](images/5d37666fa149f80ff90acfb491df86b455e9f249ddcf5501d93181aa80cbf5fa.jpg)

Figure S2 Training instability on 4B ALFWorld. (a) AgentOPSD recovers from a prolonged near-zero phase, whereas GRPO+OPSD finishes near zero after losing its early gains. (b) OCSD declines from a later peak. Both panels retain GRPO as a reference and show raw validation success every five steps. These curves are selected from Figure 2.
<table><tr><td>Method</td><td>AUC</td><td>AUC rank</td><td>Final</td><td>Final Low rank</td><td>/12</td><td>Drop /12</td></tr><tr><td></td><td>36.89</td><td>5</td><td>49.68</td><td>3</td><td>1</td><td>0</td></tr><tr><td>GRPO GRPO+OPSD</td><td>33.80</td><td></td><td>1043.05</td><td>10</td><td>2</td><td>2</td></tr><tr><td>SDAR</td><td>37.54</td><td></td><td>446.33</td><td>6</td><td>1</td><td>0</td></tr><tr><td>Skill-SD</td><td>30.35</td><td></td><td>11 39.45</td><td>11</td><td>3</td><td>0</td></tr><tr><td>RLSD</td><td>38.20</td><td></td><td>3 46.54</td><td>5</td><td>2</td><td>2</td></tr><tr><td>OCSD</td><td>39.88</td><td></td><td>1 49.50</td><td>4</td><td>0</td><td>1</td></tr><tr><td>OVCSD</td><td>36.73</td><td></td><td>6 46.17</td><td>7</td><td>2</td><td>2</td></tr><tr><td>AgentOPSD</td><td>34.41</td><td></td><td>951.34</td><td>2</td><td>1</td><td>0</td></tr><tr><td>ADRS</td><td>35.64</td><td></td><td>744.61</td><td>9</td><td>1</td><td>1</td></tr><tr><td>ICSD</td><td>35.60</td><td></td><td>844.75</td><td>8</td><td>2</td><td>1</td></tr><tr><td>PCSD</td><td>38.45</td><td></td><td>2 54.26</td><td>1</td><td>1</td><td>0</td></tr></table>

Table S6 Training progress and stability across all twelve equally weighted model–benchmark settings. AUC is the area under the raw validation curve divided by 100 training steps; Final is the repeated final-evaluation mean. Low counts final validation scores below $5 ;$ Drop counts peak-to-final declines of at least 20 points. These thresholds describe observed performance, not failure probabilities.

## E.2 Low-Performance Outcomes

Across the twelve settings, 15 trained models have a repeated final-evaluation mean below 5 on their benchmark’s reported metric. One ALFWorld model and five WebShop models also underperform their pretrained backbones; nine 2B ScienceWorld models have task scores below 2 despite improving over their initially negative scores. No Search-QA model falls below this threshold in repeated final evaluation. The same low endpoint can therefore reflect either lost pretrained capability or limited task progress. Reporting these outcomes alongside aggregate gains distinguishes severe regressions from settings where the training methods have largely failed to learn useful behavior. Tables 2 and 3 provide the model-specific comparisons; Appendix F.5 relates low validation scores to training trajectories and numerical diagnostics.

## F Training Dynamics and Stability

We apply the same trajectory summaries and diagnostic criteria to all 132 runs across four benchmarks. ALFWorld and WebShop use success (%), Search-QA uses exact-match accuracy (%), and ScienceWorld uses task score, retaining negative values. Each model–benchmark setting receives equal weight in aggregate AUC, stage-wise performance, and final-score comparisons. These aggregate values summarize the reported metrics of the four benchmarks rather than a pooled task-success probability. The analyses below connect these summaries to intermediate instability, late regression, and prolonged low-score plateaus.

<table><tr><td>Method</td><td>Step 20</td><td>Step 40</td><td>Step 60</td><td>Step 80</td><td>Step 100</td></tr><tr><td>GRPO</td><td>25.65</td><td>39.34</td><td>44.22</td><td>47.68</td><td>49.47</td></tr><tr><td>GRPO+OPSD</td><td>27.17</td><td>35.02</td><td>38.31</td><td>42.38</td><td>43.19</td></tr><tr><td>SDAR</td><td>30.63</td><td>40.81</td><td>42.50</td><td>45.52</td><td>46.15</td></tr><tr><td>Skill-SD</td><td>21.64</td><td>30.40</td><td>33.99</td><td>38.19</td><td>39.36</td></tr><tr><td>RLSD</td><td>30.37</td><td>37.29</td><td>43.80</td><td>48.65</td><td>46.10</td></tr><tr><td>OCSD</td><td>30.39</td><td>40.61</td><td>50.00</td><td>48.85</td><td>49.15</td></tr><tr><td>OVCSD</td><td>28.38</td><td>36.53</td><td>42.13</td><td>47.44</td><td>43.89</td></tr><tr><td>AgentOPSD</td><td>24.41</td><td>31.95</td><td>35.80</td><td>46.27</td><td>51.88</td></tr><tr><td></td><td>26.16</td><td></td><td></td><td></td><td></td></tr><tr><td>ADRS</td><td></td><td>35.84</td><td>42.75</td><td>44.03</td><td>45.38</td></tr><tr><td>ICSD PCSD</td><td>26.09 26.82</td><td>35.42 38.16</td><td>43.50 43.82</td><td>45.66 50.03</td><td>44.39 53.73</td></tr></table>

Table S7 Mean validation performance at successive training stages, with equal weight on all twelve settings. ALF-World/WebShop use success (%), Search-QA uses exact-match accuracy (%), and ScienceWorld uses task score. Bold marks the highest mean at each stage.

## F.1 Aggregate Progress and Method Rankings

Method rankings depend on whether performance is assessed throughout training or only at its endpoint. Figure S1 contrasts training-curve AUC with repeated final evaluation and shows the twelve-setting mean validation trajectories. OCSD leads by AUC, whereas PCSD leads at the endpoint; the latter’s advantage emerges later in training. The stage-wise comparison makes this change explicit: SDAR leads at steps 20 and 40, OCSD at step 60, and PCSD at steps 80 and 100. An endpoint advantage therefore need not reflect superior performance sustained throughout optimization. Tables S6 and S7 report the aggregate rankings, low-performance counts, and stage-wise means for all eleven methods.

## F.2 Intermediate Instability and Late Regression

The 4B ALFWorld trajectories illustrate why instability cannot be reduced to a low final score. AgentOPSD recovers after six consecutive validations at or below 4.3%, whereas GRPO+OPSD loses its early gains and finishes near zero. OCSD reaches 85.0% before declining to 51.4%, with its final three validations averaging 50.5%. A criterion based only on near-zero final success would identify GRPO+OPSD but miss both AgentOPSD’s temporary collapse and OCSD’s sustained late regression. Assessing stability therefore requires examining the depth and persistence of intermediate declines, even when the final model remains capable. Figure S2 isolates these histories from Figure 2, retaining GRPO as a common reference.

Decoding-specific collapse. Search-QA reveals that training instability can also depend on the decoding strategy: a model may retain efective sampled performance despite a collapse under greedy decoding. Training-time validation uses greedy decoding $( T = 0 )$ , whereas final evaluation samples at $T = 0 . 4$ over eight rollout seeds; both report exact-match accuracy on the same 3,125 questions. The Search-QA panels in Figure 2 show these training trajectories; Tables S10 and 3 report their trajectory summaries and sampled final results, respectively.

For 2B OVCSD, greedy accuracy reaches 38.02% at step 70 and remains at 37.09% at step 80, before falling to 4.77%, 0.22%, 12.86%, and 1.34% at steps 85, 90, 95, and 100. The final-three-validation mean is only 4.81%, with a 36.67-point peak-to-final drop under the unchanged greedy protocol. Yet the same final checkpoint scores 33.21 ± 0.33% with sampling, compared with GRPO’s 32.38 ± 0.28%. This is a decoding-specific collapse: previously acquired greedy performance is lost even though sampling at T = 0.4 still produces efective task behavior. A sampled endpoint alone would leave this failure mode unobserved.

ICSD shows a milder greedy regression rather than near-zero collapse. Its 2B greedy accuracy peaks at 39.52% at step 65, falls to 21.73% at step 90, and partially recovers to 31.49% at step 100; the final-three mean is 26.31%.

<table><tr><td>Method</td><td>Peak</td><td>Step</td><td>Final</td><td>Tail-3</td><td>Drop</td><td>AUC</td><td>Train h</td><td>Total h</td></tr><tr><td>Qwen3.5-2B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>31.40</td><td>75</td><td>27.10</td><td>24.03</td><td>4.30</td><td>19.25</td><td>13.49</td><td>14.37</td></tr><tr><td>GRPO+OPSD</td><td>44.30</td><td>95</td><td>21.40</td><td>35.47</td><td>22.90</td><td>21.90</td><td>15.68</td><td>16.62</td></tr><tr><td>SDAR</td><td>16.40</td><td>40</td><td>16.40</td><td>10.00</td><td>0.00</td><td>7.58</td><td>15.90</td><td>16.71</td></tr><tr><td>Skill-SD</td><td>75.00</td><td>100</td><td>75.00</td><td>71.67</td><td>0.00</td><td>32.84</td><td>14.18</td><td>15.08</td></tr><tr><td>RLSD</td><td>75.00</td><td>95</td><td>46.40</td><td>62.37</td><td>28.60</td><td>27.52</td><td>15.18</td><td>16.11</td></tr><tr><td>OCSD</td><td>56.40</td><td>75</td><td>42.90</td><td>39.77</td><td>13.50</td><td>32.45</td><td>13.28</td><td>14.20</td></tr><tr><td>OVCSD</td><td>60.00</td><td>75</td><td>57.10</td><td>46.63</td><td>2.90</td><td>28.14</td><td>22.23</td><td>23.26</td></tr><tr><td>AgentOPSD</td><td>49.30</td><td>95</td><td>48.60</td><td>46.67</td><td>0.70</td><td>21.15</td><td>15.73</td><td>16.70</td></tr><tr><td>ADRS</td><td>15.70</td><td>55</td><td>11.40</td><td>12.83</td><td>4.30</td><td>12.56</td><td>15.69</td><td>16.38</td></tr><tr><td>ICSD</td><td>67.90</td><td>75</td><td>55.00</td><td>53.33</td><td>12.90</td><td>35.15</td><td>15.26</td><td>16.13</td></tr><tr><td>PCSD</td><td>69.30</td><td>100</td><td>69.30</td><td>63.33</td><td>0.00</td><td>31.33</td><td>14.12</td><td>15.03</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>84.30</td><td>85</td><td>83.60</td><td>81.20</td><td>0.70</td><td>59.78</td><td>14.94</td><td>16.22</td></tr><tr><td>GRPO+OPSD</td><td>39.30</td><td>20</td><td>2.10</td><td>0.70</td><td>37.20</td><td>9.92</td><td>24.14</td><td>25.55</td></tr><tr><td>SDAR</td><td>75.00</td><td>95</td><td>72.10</td><td>72.60</td><td>2.90</td><td>61.18</td><td>17.27</td><td>18.71</td></tr><tr><td>Skill-SD</td><td>70.70</td><td>55</td><td>55.70</td><td>55.00</td><td>15.00</td><td>50.73</td><td>18.61</td><td>20.12</td></tr><tr><td>RLSD</td><td>90.00</td><td>55</td><td>82.10</td><td>83.53</td><td>7.90</td><td>76.14</td><td>15.54</td><td>17.15</td></tr><tr><td>OCSD</td><td>85.00</td><td>60</td><td>51.40</td><td>50.47</td><td>33.60</td><td>55.10</td><td>17.75</td><td>19.34</td></tr><tr><td>OVCSD</td><td>82.10</td><td>95</td><td>76.40</td><td>78.80</td><td>5.70</td><td>61.57</td><td>20.81</td><td>22.40</td></tr><tr><td>AgentOPSD</td><td>79.30</td><td>100</td><td>79.30</td><td>68.57</td><td>0.00</td><td>33.66</td><td>22.62</td><td>24.43</td></tr><tr><td>ADRS</td><td>78.60</td><td>45</td><td>73.60</td><td>73.10</td><td>5.00</td><td>66.25</td><td>16.20</td><td>17.68</td></tr><tr><td>ICSD</td><td>83.60</td><td>65</td><td>81.40</td><td>75.70</td><td>2.20</td><td>65.21</td><td>19.57</td><td>21.08</td></tr><tr><td>PCSD</td><td>81.40</td><td>95</td><td>75.70</td><td>78.07</td><td>5.70</td><td>55.76</td><td>18.92</td><td>20.62</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>70.00</td><td>80</td><td>63.60</td><td>67.17</td><td>6.40</td><td>55.55</td><td>22.16</td><td>24.51</td></tr><tr><td>GRPO+OPSD</td><td>72.90</td><td>100</td><td>72.90</td><td>69.07</td><td>0.00</td><td>53.60</td><td>24.16</td><td>26.47</td></tr><tr><td>SDAR</td><td>85.00</td><td>85</td><td>75.00</td><td>77.87</td><td>10.00</td><td>64.78</td><td>23.21</td><td>25.48</td></tr><tr><td>Skill-SD</td><td>77.10</td><td>95</td><td>72.10</td><td>74.27</td><td>5.00</td><td>61.77</td><td>24.65</td><td>27.08</td></tr><tr><td>RLSD</td><td>92.10</td><td>85</td><td>89.30</td><td>88.80</td><td>2.80</td><td>71.56</td><td>22.00</td><td>24.48</td></tr><tr><td>OCSD</td><td>87.10</td><td>95</td><td>81.40</td><td>82.60</td><td>5.70</td><td>61.77</td><td>21.19</td><td>23.37</td></tr><tr><td>OVCSD</td><td>84.30</td><td>100</td><td>84.30</td><td>79.03</td><td>0.00</td><td>64.03</td><td>23.22</td><td>25.63</td></tr><tr><td>AgentOPSD</td><td>77.10</td><td>80</td><td>76.40</td><td>74.77</td><td>0.70</td><td>56.79</td><td>23.94</td><td>26.17</td></tr><tr><td>ADRS</td><td>82.90</td><td>90</td><td>82.10</td><td>80.23</td><td>0.80</td><td>70.08</td><td>22.39</td><td>24.83</td></tr><tr><td>ICSD</td><td>93.60</td><td>90</td><td>92.10</td><td>92.60</td><td>1.50</td><td>72.62</td><td>24.13</td><td>26.45</td></tr><tr><td>PCSD</td><td>77.90</td><td>100</td><td>77.90</td><td>72.87</td><td>0.00</td><td>55.50</td><td>25.25</td><td>27.62</td></tr></table>

Table S8 ALFWorld training profiles for all 33 runs, using success (%). Peak includes initialization; Step gives its first occurrence; Final is step-100 validation; Tail-3 averages steps 90, 95, and 100; Drop is Peak minus Final. AUC is normalized by the training horizon. Train time excludes validation and saving; Total includes both.

Nevertheless, its sampled final accuracy of $3 9 . 3 9 \pm 0 . 1 9 \%$ leads the 2B Search-QA comparison. Together, these cases extend the collapse and regression analysis beyond failures visible under sampled evaluation: efective nonzero-temperature sampling does not rule out a collapse or substantial loss of performance under greedy decoding.

## F.3 Early Stagnation on ScienceWorld

Low-score plateaus are informative training outcomes. At 2B, eight methods including GRPO improve over their initially negative scores but remain at 0.83 in every validation from steps 10 to 100, with repeated final scores of $0 . 8 3 \pm 0 . 0 0$ . Flat task scores need not imply inactive optimization: GRPO records finite, nonzero actor gradients throughout training. For an initially weak policy, updates may change action probabilities without improving measured task outcomes if more efective behavior is rarely discovered. This provides a plausible interpretation of the stagnation; matching aggregate scores do not require identical policies. Such settings are valuable because they test whether teacher guidance can help agents move beyond limited initial task competence, rather than only improve agents that already learn efectively. The overlapping traces in Figure 2 reflect these matching recorded scores, with colored markers distinguishing the methods.

<table><tr><td>Method</td><td>Peak</td><td>Step</td><td>Final</td><td>Tail-3</td><td>Drop</td><td>AUC</td><td>Train h</td><td>Total h</td></tr><tr><td>Qwen3.5-2B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>0.00</td><td>0.74</td><td>4.95</td><td>6.34</td></tr><tr><td>GRPO+OPSD</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>0.00</td><td>0.79</td><td>5.62</td><td>7.02</td></tr><tr><td>SDAR</td><td>1.08</td><td>5</td><td>0.83</td><td>0.83</td><td>0.25</td><td>0.75</td><td>5.74</td><td>7.24</td></tr><tr><td>Skill-SD</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>0.00</td><td>0.76</td><td>5.53</td><td>6.95</td></tr><tr><td>RLSD</td><td>2.39</td><td>95</td><td>1.89</td><td>2.02</td><td>0.51</td><td>1.35</td><td>6.87</td><td>8.09</td></tr><tr><td>OCSD</td><td>13.86</td><td>100</td><td>13.86</td><td>13.60</td><td>0.00</td><td>11.72</td><td>6.04</td><td>7.35</td></tr><tr><td>OVCSD</td><td>15.77</td><td>90</td><td>13.71</td><td>14.92</td><td>2.05</td><td>8.90</td><td>7.98</td><td>9.36</td></tr><tr><td>AgentOPSD</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>0.00</td><td>0.77</td><td>5.48</td><td>6.66</td></tr><tr><td>ADRS</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>0.00</td><td>0.80</td><td>6.77</td><td>8.01</td></tr><tr><td>ICSD</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>0.00</td><td>0.73</td><td>6.11</td><td>7.43</td></tr><tr><td>PCSD</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>0.00</td><td>0.75</td><td>6.86</td><td>8.16</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>33.37</td><td></td><td>6031.11</td><td>22.59</td><td>2.25</td><td>21.07</td><td>8.62</td><td>11.24</td></tr><tr><td>GRPO+OPSD</td><td>13.86</td><td>90</td><td>13.66</td><td>13.71</td><td>0.20</td><td>12.87</td><td>9.02</td><td>10.87</td></tr><tr><td>SDAR</td><td>33.40</td><td>25</td><td>13.86</td><td>13.79</td><td>19.54</td><td>16.55</td><td>9.66</td><td>12.14</td></tr><tr><td>Skill-SD</td><td>44.73</td><td>95</td><td>42.15</td><td>42.64</td><td>2.59</td><td>34.28</td><td>8.25</td><td>10.19</td></tr><tr><td>RLSD</td><td>52.12</td><td>65</td><td>22.43</td><td>20.56</td><td>29.69</td><td>35.43</td><td>9.38</td><td>11.34</td></tr><tr><td>OCSD</td><td>48.02</td><td>75</td><td>41.67</td><td>35.25</td><td>6.34</td><td>36.12</td><td>9.04</td><td>11.29</td></tr><tr><td>OVCSD</td><td>41.73</td><td>85</td><td>15.54</td><td>32.36</td><td>26.20</td><td>31.54</td><td>11.39</td><td>14.04</td></tr><tr><td>AgentOPSD</td><td>36.20</td><td>50</td><td>35.01</td><td>31.28</td><td>1.20</td><td>26.77</td><td>9.84</td><td>11.72</td></tr><tr><td>ADRS</td><td>37.90</td><td>70</td><td>36.95</td><td>31.97</td><td>0.95</td><td>27.12</td><td>9.37</td><td>11.21</td></tr><tr><td>ICSD</td><td>16.27</td><td>75</td><td>15.86</td><td>15.29</td><td>0.41</td><td>14.46</td><td>10.72</td><td>12.79</td></tr><tr><td>PCSD</td><td>41.63</td><td>65</td><td>23.00</td><td>29.79</td><td>18.63</td><td>28.74</td><td>9.63</td><td>11.55</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>59.15</td><td>100</td><td>59.15</td><td>53.03</td><td>0.00</td><td>38.90</td><td>11.18</td><td>14.44</td></tr><tr><td>GRPO+OPSD</td><td>54.91</td><td>60</td><td>52.11</td><td>52.59</td><td>2.80</td><td>42.98</td><td>12.65</td><td>15.94</td></tr><tr><td>SDAR</td><td>58.90</td><td>80</td><td>57.13</td><td>55.90</td><td>1.77</td><td>46.40</td><td>12.95</td><td>16.94</td></tr><tr><td>Skill-SD</td><td>38.34</td><td>50</td><td>28.84</td><td>33.60</td><td>9.50</td><td>30.04</td><td>13.19</td><td>16.84</td></tr><tr><td>RLSD</td><td>50.36</td><td>100</td><td>50.36</td><td>46.48</td><td>0.00</td><td>38.30</td><td>12.86</td><td>16.52</td></tr><tr><td>OCSD</td><td>55.11</td><td>60</td><td>45.36</td><td>47.74</td><td>9.75</td><td>41.52</td><td>11.39</td><td>14.29</td></tr><tr><td>OVCSD</td><td>52.73</td><td>95</td><td>45.09</td><td>45.82</td><td>7.63</td><td>32.70</td><td>16.50</td><td>20.26</td></tr><tr><td>AgentOPSD</td><td>50.61</td><td>45</td><td>42.58</td><td>30.22</td><td>8.03</td><td>37.53</td><td>12.44</td><td>15.56</td></tr><tr><td>ADRS</td><td>58.41</td><td>65</td><td>37.83</td><td>46.33</td><td>20.58</td><td>44.49</td><td>13.16</td><td>16.44</td></tr><tr><td>ICSD</td><td>55.23</td><td>30</td><td>18.77</td><td>26.99</td><td>36.46</td><td>36.00</td><td>14.07</td><td>17.15</td></tr><tr><td>PCSD</td><td>58.33</td><td>100</td><td>58.33</td><td>55.95</td><td>0.00</td><td>44.33</td><td>12.67</td><td>15.99</td></tr></table>

Table S9 ScienceWorld training profiles for all 33 runs, using task score. Peak includes initialization; Step gives its first occurrence; Final is step-100 validation; Tail-3 averages steps 90, 95, and 100; Drop is Peak minus Final. AUC is normalized by the training horizon. Train time excludes validation and saving; Total includes both.

Escaping the plateau remains method- and scale-dependent. OCSD and OVCSD reach final 2B task scores of 13.78 ± 0.03 and 13.85 ± 0.37, whereas RLSD reaches 1.89 ± 0.07. Their separation from the common plateau shows that this setting is not uniformly unlearnable under the available training budget. Yet the leading method changes with scale: OCSD reaches 41.77 at 4B, while GRPO leads at 9B with 58.99. These results motivate examining whether a method enables sustained task progress at a given backbone scale, rather than treating either a flat curve as stability or a small-model gain as a general advantage. Table S9 reports peak, late-training, and final task scores for all three scales.

<table><tr><td>Method</td><td>Peak</td><td>Step</td><td>Final</td><td>Tail-3</td><td>Drop</td><td>AUC</td><td>Train h</td><td>Total h</td></tr><tr><td>Qwen3.5-2B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>33.73</td><td>50</td><td>32.03</td><td>32.17</td><td>1.70</td><td>29.56</td><td>2.00</td><td>2.64</td></tr><tr><td>GRPO+OPSD</td><td>35.01</td><td>90</td><td>34.24</td><td>34.51</td><td>0.77</td><td>26.34</td><td>2.03</td><td>2.54</td></tr><tr><td>SDAR</td><td>38.40</td><td>75</td><td>38.02</td><td>37.46</td><td>0.38</td><td>32.49</td><td>2.90</td><td>3.77</td></tr><tr><td>Skill-SD</td><td>38.08</td><td>90</td><td>37.57</td><td>37.86</td><td>0.51</td><td>28.90</td><td>2.56</td><td>3.43</td></tr><tr><td>RLSD</td><td>39.74</td><td>75</td><td>38.02</td><td>38.50</td><td>1.73</td><td>35.58</td><td>3.40</td><td>4.48</td></tr><tr><td>OCSD</td><td>15.58</td><td>45</td><td>13.31</td><td>13.49</td><td>2.27</td><td>14.38</td><td>0.97</td><td>1.14</td></tr><tr><td>OVCSD</td><td>38.02</td><td>70</td><td>1.34</td><td>4.81</td><td>36.67</td><td>27.84</td><td>3.54</td><td>4.60</td></tr><tr><td>AgentOPSD</td><td>37.92</td><td>60</td><td>34.43</td><td>35.40</td><td>3.49</td><td>27.87</td><td>2.58</td><td>3.37</td></tr><tr><td>ADRS</td><td>16.61</td><td>25</td><td>14.18</td><td>14.18</td><td>2.43</td><td>14.55</td><td>1.13</td><td>1.34</td></tr><tr><td>ICSD</td><td>39.52</td><td>65</td><td>31.49</td><td>26.31</td><td>8.03</td><td>32.10</td><td>3.47</td><td>4.45</td></tr><tr><td>PCSD</td><td>34.05</td><td>85</td><td>31.71</td><td>32.92</td><td>2.34</td><td>30.17</td><td>2.36</td><td>2.99</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>45.12</td><td>100</td><td>45.12</td><td>44.11</td><td>0.00</td><td>40.86</td><td>3.99</td><td>5.41</td></tr><tr><td>GRPO+OPSD</td><td>44.42</td><td>75</td><td>43.81</td><td>43.84</td><td>0.61</td><td>40.69</td><td>4.67</td><td>6.14</td></tr><tr><td>SDAR</td><td>45.57</td><td>100</td><td>45.57</td><td>44.05</td><td>0.00</td><td>40.14</td><td>4.29</td><td>5.45</td></tr><tr><td>Skill-SD</td><td>43.74</td><td>95</td><td>43.30</td><td>43.45</td><td>0.45</td><td>39.62</td><td>4.76</td><td>6.28</td></tr><tr><td>RLSD</td><td>43.23</td><td>100</td><td>43.23</td><td>42.46</td><td>0.00</td><td>39.35</td><td>5.00</td><td>6.56</td></tr><tr><td>OCSD</td><td>44.35</td><td>95</td><td>44.19</td><td>43.85</td><td>0.16</td><td>39.27</td><td>4.08</td><td>5.55</td></tr><tr><td>OVCSD</td><td>44.16</td><td>80</td><td>43.65</td><td>43.52</td><td>0.51</td><td>40.44</td><td>4.91</td><td>6.46</td></tr><tr><td>AgentOPSD</td><td>44.42</td><td>100</td><td>44.42</td><td>43.99</td><td>0.00</td><td>40.90</td><td>4.82</td><td>6.30</td></tr><tr><td>ADRS</td><td>42.75</td><td>100</td><td>42.75</td><td>41.97</td><td>0.00</td><td>38.96</td><td>4.60</td><td>6.06</td></tr><tr><td>ICSD</td><td>46.14</td><td>100</td><td>46.14</td><td>45.94</td><td>0.00</td><td>41.83</td><td>5.02</td><td>6.25</td></tr><tr><td>PCSD</td><td>44.16</td><td>65</td><td>43.55</td><td>43.73</td><td>0.61</td><td>39.28</td><td>4.24</td><td>5.57</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>47.62</td><td>95</td><td>46.50</td><td>46.46</td><td>1.12</td><td>41.28</td><td>5.93</td><td>8.22</td></tr><tr><td>GRPO+OPSD</td><td>47.55</td><td>100</td><td>47.55</td><td>46.93</td><td>0.00</td><td>42.33</td><td>7.05</td><td>9.38</td></tr><tr><td>SDAR</td><td>45.15</td><td>95</td><td>45.15</td><td>44.64</td><td>0.00</td><td>41.11</td><td>6.85</td><td>9.06</td></tr><tr><td>Skill-SD</td><td>46.50</td><td>100</td><td>46.50</td><td>45.69</td><td>0.00</td><td>41.87</td><td>6.39</td><td>8.09</td></tr><tr><td>RLSD</td><td>45.25</td><td>80</td><td>44.22</td><td>44.04</td><td>1.02</td><td>40.60</td><td>7.46</td><td>9.83</td></tr><tr><td>OCSD</td><td>47.78</td><td>100</td><td>47.78</td><td>47.18</td><td>0.00</td><td>40.37</td><td>6.04</td><td>8.26</td></tr><tr><td>OVCSD</td><td>45.79</td><td>100</td><td>45.79</td><td>45.00</td><td>0.00</td><td>41.40</td><td>7.28</td><td>9.75</td></tr><tr><td>AgentOPSD</td><td>45.98</td><td>75</td><td>45.41</td><td>45.06</td><td>0.58</td><td>40.50</td><td>6.18</td><td>7.96</td></tr><tr><td>ADRS</td><td>44.19</td><td>100</td><td>44.19</td><td>43.61</td><td>0.00</td><td>38.85</td><td>6.99</td><td>9.27</td></tr><tr><td>ICSD</td><td>46.59</td><td>100</td><td>46.59</td><td>46.10</td><td>0.00</td><td>41.44</td><td>7.70</td><td>9.96</td></tr><tr><td>PCSD</td><td>45.76</td><td>80</td><td>44.99</td><td>44.92</td><td>0.77</td><td>40.57</td><td>6.64</td><td>8.60</td></tr></table>

Table S10 Search-QA training profiles for all 33 runs, using exact-match accuracy (%). Peak includes initialization; Step gives its first occurrence; Final is step-100 validation; Tail-3 averages steps 90, 95, and 100; Drop is Peak minus Final. AUC is normalized by the training horizon. Train time excludes validation and saving; Total includes both.

## F.4 Per-Run Training Profiles

Aggregate curves can conceal instability confined to a particular task or model size. For every benchmark, we report each run’s peak, final, and late-training performance alongside its AUC and computational cost. Peak is the highest validation metric at steps 0, 5, . . . , 100; Step records its first occurrence, Tail-3 averages steps 90, 95, and 100, and Drop is Peak minus Final in the corresponding metric’s points. AUC integrates the raw validation curve by the trapezoidal rule and divides by the 100-step horizon; it is not ROC-AUC. These summaries capture diferent aspects of the trajectory: a low Tail-3 indicates that weak performance extends beyond the final measurement, whereas a small Drop does not exclude earlier instability. For example, AgentOPSD on 4B ALFWorld finishes at its own peak despite the prolonged near-zero phase described above. Tables S8–S11 report all 132 profiles with the same columns for each benchmark.

Net time includes rollout generation, teacher computation, and policy updates; total time additionally includes in-loop validation and checkpoint saving. All runs use eight H800 GPUs, so the corresponding GPU-hours equal elapsed hours multiplied by eight. These per-run measurements support the paired cost comparisons in

<table><tr><td>Method</td><td>Peak</td><td>Step</td><td>Final</td><td>Tail-3</td><td>Drop</td><td>AUC</td><td>Train h</td><td>Total h</td></tr><tr><td colspan="9">Qwen3.5-2B</td></tr><tr><td>GRPO</td><td>70.30</td><td>100</td><td>70.30</td><td>67.17</td><td>0.00</td><td>41.19</td><td>3.89</td><td>4.23</td></tr><tr><td>GRPO+OPSD</td><td>77.30</td><td>85</td><td>75.80</td><td>67.20</td><td>1.50</td><td>53.47</td><td>3.87</td><td>4.15</td></tr><tr><td>SDAR</td><td>71.90</td><td>85</td><td>64.80</td><td>63.53</td><td>7.10</td><td>46.06</td><td>5.05</td><td>5.36</td></tr><tr><td>Skill-SD</td><td>75.80</td><td>85</td><td>70.30</td><td>71.07</td><td>5.50</td><td>43.04</td><td>4.22</td><td>4.52</td></tr><tr><td>RLSD</td><td>69.50</td><td>85</td><td>68.80</td><td>68.53</td><td>0.70</td><td>37.68</td><td>4.39</td><td>4.69</td></tr><tr><td>OCSD</td><td>75.00</td><td>100</td><td>75.00</td><td>36.43</td><td>0.00</td><td>49.91</td><td>3.73</td><td>4.07</td></tr><tr><td>OVCSD</td><td>2.30</td><td>0</td><td>0.80</td><td>0.53</td><td>1.50</td><td>0.86</td><td>5.23</td><td>5.62</td></tr><tr><td>AgentOPSD</td><td>71.90</td><td>100</td><td>71.90</td><td>70.07</td><td>0.00</td><td>36.50</td><td>4.70</td><td>5.03</td></tr><tr><td>ADRS</td><td>64.80</td><td>100</td><td>64.80</td><td>63.53</td><td>0.00</td><td>30.63</td><td>4.22</td><td>4.55</td></tr><tr><td>ICSD</td><td>71.10</td><td>100</td><td>71.10</td><td>69.27</td><td>0.00</td><td>36.15</td><td>4.38</td><td>4.70</td></tr><tr><td>PCSD</td><td>75.80</td><td>100</td><td>75.80</td><td>71.10</td><td>0.00</td><td>40.34</td><td>4.68</td><td>5.01</td></tr><tr><td colspan="9">Qwen3.5-4B</td></tr><tr><td>GRPO</td><td>68.80</td><td>80</td><td>64.80</td><td>64.07</td><td>4.00</td><td>45.70</td><td>5.36</td><td>5.88</td></tr><tr><td>GRPO+OPSD</td><td>79.70</td><td>90</td><td>77.30</td><td>76.30</td><td>2.40</td><td>50.06</td><td>6.48</td><td>7.00</td></tr><tr><td>SDAR</td><td>60.20</td><td>95</td><td>50.80</td><td>52.10</td><td>9.40</td><td>39.55</td><td>7.18</td><td>7.76</td></tr><tr><td>Skill-SD</td><td>5.50</td><td>0</td><td>0.00</td><td>0.00</td><td>5.50</td><td>0.18</td><td>7.03</td><td>7.40</td></tr><tr><td>RLSD</td><td>73.40</td><td>70</td><td>66.40</td><td>67.20</td><td>7.00</td><td>54.71</td><td>6.32</td><td>6.87</td></tr><tr><td>OCSD</td><td>68.00</td><td>65</td><td>64.10</td><td>65.13</td><td>3.90</td><td>52.07</td><td>5.41</td><td>5.95</td></tr><tr><td>OVCSD</td><td>75.80</td><td>90</td><td>73.40</td><td>74.73</td><td>2.40</td><td>50.76</td><td>6.50</td><td>7.09</td></tr><tr><td>AgentOPSD</td><td>74.20</td><td>100</td><td>74.20</td><td>72.13</td><td>0.00</td><td>43.05</td><td>6.45</td><td>7.02</td></tr><tr><td>ADRS</td><td>68.80</td><td>95</td><td>66.40</td><td>67.47</td><td>2.40</td><td>37.03</td><td>6.56</td><td>7.11</td></tr><tr><td>ICSD</td><td>3.90</td><td>0</td><td>0.00</td><td>0.00</td><td>3.90</td><td>0.26</td><td>8.63</td><td>9.05</td></tr><tr><td>PCSD</td><td>70.30</td><td>100</td><td>70.30</td><td>69.00</td><td>0.00</td><td>37.05</td><td>6.68</td><td>7.17</td></tr><tr><td colspan="9">Qwen3.5-9B</td></tr><tr><td>GRPO</td><td>71.90</td><td>80</td><td>69.50</td><td>68.73</td><td>2.40</td><td>48.77</td><td>8.09</td><td>8.89</td></tr><tr><td>GRPO+OPSD</td><td>76.60</td><td>100</td><td>76.60</td><td>74.50</td><td>0.00</td><td>50.69</td><td>9.13</td><td>9.94</td></tr><tr><td>SDAR</td><td>74.20</td><td>100</td><td>74.20</td><td>71.87</td><td>0.00</td><td>53.89</td><td>8.57</td><td>9.38</td></tr><tr><td>Skill-SD</td><td>7.80</td><td>0</td><td>0.00</td><td>0.00</td><td>7.80</td><td>0.20</td><td>9.87</td><td>10.44</td></tr><tr><td>RLSD</td><td>6.20</td><td>0</td><td>0.00</td><td>0.00</td><td>6.20</td><td>0.15</td><td>9.66</td><td>10.25</td></tr><tr><td>OCSD</td><td>74.20</td><td>85</td><td>68.80</td><td>70.87</td><td>5.40</td><td>43.90</td><td>8.75</td><td>9.55</td></tr><tr><td>OVCSD</td><td>75.00</td><td>85</td><td>69.50</td><td>72.63</td><td>5.50</td><td>52.61</td><td>9.29</td><td>10.14</td></tr><tr><td>AgentOPSD</td><td>75.00</td><td>90</td><td>69.50</td><td>72.40</td><td>5.50</td><td>47.46</td><td>9.45</td><td>10.27</td></tr><tr><td>ADRS</td><td>69.50</td><td>100</td><td>69.50</td><td>67.70</td><td>0.00</td><td>46.36</td><td>8.90</td><td>9.63</td></tr><tr><td>ICSD</td><td>73.40</td><td>100</td><td>73.40</td><td>72.13</td><td>0.00</td><td>51.31</td><td>9.84</td><td>10.56</td></tr><tr><td>PCSD</td><td>75.80</td><td>95</td><td>73.40</td><td>74.47</td><td>2.40</td><td>57.55</td><td>8.91</td><td>9.74</td></tr></table>

Table S11 WebShop training profiles for all 33 runs, using success (%). Peak includes initialization; Step gives its first occurrence; Final is step-100 validation; Tail-3 averages steps 90, 95, and 100; Drop is Peak minus Final. AUC is normalized by the training horizon. Train time excludes validation and saving; Total includes both.

Appendix G.1.

## F.5 Task Failure and Numerical Diagnostics

We use a final validation metric below 5 as a descriptive low-performance criterion across all four benchmarks. It identifies 16 runs: one ALFWorld run, five WebShop runs, one Search-QA run, and nine ScienceWorld runs. The ALFWorld and WebShop runs also regress below their pretrained final-evaluation baselines, whereas the ScienceWorld cases largely reflect low-score stagnation rather than loss from a capable starting point. Search-QA OVCSD illustrates the decoding-dependent case discussed above: its low validation endpoint is not reproduced under sampled final evaluation. Thus, the common threshold identifies cases for analysis without treating every low score as a training collapse. Table S12 summarizes these sixteen runs, including their training trajectories, repeated final results, late-training interaction lengths, and nonfinite-gradient counts.

Persistent failure near the action limit. In all five low-performing WebShop runs, student rollout success is zero throughout the final 20 training steps, with mean episode lengths of 14.99–15.00 against the 15-action limit. Over the same interval, GRPO+OPSD on 4B ALFWorld averages 7.1% training success and 49.0 of the allowed 50 actions, compared with 81.5% success and 18.4 actions for GRPO. By contrast, low-performing ScienceWorld runs average 12.88–20.00 of the allowed 20 turns, and Search-QA OVCSD averages 2.48 of four turns. Near-limit interaction length therefore characterizes the persistent WebShop failures but is not a universal signature of low performance. Joint inspection of task progress and interaction length distinguishes wasted interaction budgets from other forms of inefective learning.

<table><tr><td>Model / method</td><td>Initial</td><td>Peak</td><td>Step</td><td>Val.</td><td>Eval.</td><td>Turns/limit</td><td>NF</td></tr><tr><td colspan="6">ALFWorld</td><td></td><td></td></tr><tr><td>4B GRPO+OPSD ScienceWorld</td><td>33.60</td><td>39.30</td><td>20</td><td>2.10</td><td>1.52</td><td>49.01/50</td><td>0</td></tr><tr><td colspan="6"></td><td></td><td></td></tr><tr><td>2B GRPO</td><td>-2.94</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>17.38/20</td><td>0</td></tr><tr><td>2B GRPO+OPSD</td><td>-0.94</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>15.60/20</td><td>0</td></tr><tr><td>2B SDAR</td><td>-2.77</td><td>1.08</td><td>5</td><td>0.83</td><td>0.83</td><td>16.71/20</td><td>0</td></tr><tr><td>2B Skill-SD</td><td>-1.97</td><td>0.83 2.39</td><td>5</td><td>0.83</td><td>0.83</td><td>12.88/20</td><td>39</td></tr><tr><td>2B RLSD</td><td>-2.51</td><td>0.83</td><td>95</td><td>1.89</td><td>1.89</td><td>20.00/20</td><td>0</td></tr><tr><td>2B AgentOPSD</td><td>-1.73</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>19.52/20</td><td>0</td></tr><tr><td>2B ADRS</td><td>-0.42</td><td></td><td>5</td><td>0.83</td><td>0.83</td><td>19.99/20</td><td>0</td></tr><tr><td>2B ICSD</td><td>-3.23</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>14.28/20</td><td>0</td></tr><tr><td>2B PCSD</td><td>-2.51</td><td>0.83</td><td>5</td><td>0.83</td><td>0.83</td><td>20.00/20</td><td>0</td></tr><tr><td colspan="6">Search-QA</td><td></td><td></td></tr><tr><td>2B OVCSD</td><td>13.95</td><td>38.02</td><td>70</td><td>1.34</td><td>33.21</td><td>2.48/4</td><td>0</td></tr><tr><td colspan="6">WebShop</td><td></td><td></td></tr><tr><td>2B OVCSD</td><td>2.30</td><td>2.30</td><td></td><td>0 0.80</td><td>1.07</td><td>14.99/15</td><td>0</td></tr><tr><td>4B Skill-SD</td><td>5.50</td><td>5.50</td><td></td><td>0 0.00</td><td>0.00</td><td>15.00/15</td><td>1</td></tr><tr><td>4B ICSD</td><td>3.90</td><td>3.90</td><td>0</td><td>0.00</td><td>0.00</td><td>15.00/15</td><td>0</td></tr><tr><td>9B Skill-SD</td><td>7.80</td><td>7.80</td><td>0</td><td>0.00</td><td>0.00</td><td>15.00/15</td><td>68</td></tr><tr><td>9B RLSD</td><td>6.20</td><td>6.20</td><td>0</td><td>0.00</td><td>0.00</td><td>15.00/15</td><td>0</td></tr></table>

Table S12 All runs with final validation performance below 5 across the four benchmarks. Initial, Peak, and Val. use the training-validation metric; Eval. is the repeated final-evaluation mean. Turns/limit averages training steps 81–100 relative to the interaction limit; NF counts steps with nonfinite actor gradients. Low scores include stagnation as well as lost performance.

Gradient diagnostics alone do not identify task failure. Only 3 of the 16 low-performing runs record nonfinite actor gradients; the other 13 do not. Across all four benchmarks, 18 of the 21 runs with nonfinite gradients finish with validation performance at or above 5. All 9 runs with peak-to-final declines of at least 20 points also have finite recorded gradients throughout. Finite recorded gradients therefore do not establish successfu learning, just as a nonfinite-gradient event does not by itself imply a near-zero endpoint. Numerical checks and behavioral evaluation are complementary requirements for assessing training stability.

## G Computational Cost Analysis

This appendix analyzes computational cost for all 132 runs across four benchmarks, complementing the per-setting comparisons in Figure 3. Final-performance comparisons use the repeated evaluations; time-budget comparisons select observed points from each benchmark’s training-validation trajectories.

## G.1 Performance and Time Budgets

Computational cost changes the practical value of a performance gain. We assess both the resources used to obtain the final agent and the performance reached within a fixed training-time budget. For the former, we pair repeated-evaluation performance with net training time, average the within-setting time ratios to GRPO, and sum GPU-hours across each method’s twelve runs. Nine alternatives have higher mean paired training time than GRPO; OCSD is approximately tied at 0.999×. Only PCSD and AgentOPSD improve mean final performance, with time ratios of 1.17 and 1.19. PCSD’s larger aggregate gain comes at a smaller average relative overhead, whereas OVCSD has the largest time ratio (1.37) without improving aggregate performance. Table S13 lists the performance and training-time values for every point in Figure 3, allowing direct comparisons where markers overlap; Table S14 summarizes these trade-ofs across settings.

![](images/6f173b0514a7da813341addc8b13d68baeec1cc460b60a1d1431d4d50f393b69.jpg)  
Figure S3 Training cost by stage across all eleven methods and twelve settings. Policy/ref scoring combines old-policy and reference log-probability computation; Other combines reward computation and the remaining training time. Teacher forward denotes separately timed teacher passes; teacher-related work may also occur in other stages. Axis ranges are shared within each benchmark. Validation and checkpoint saving are excluded.

Performance within GRPO’s training-time budget. We use the net training time of GRPO as the budget for each model–task setting. For every method, we select the latest evaluated checkpoint reached within this budget; methods that finish earlier use their final checkpoint. The fixed-step and time-budget scores are drawn from the same validation trajectory. For each of the four benchmarks, we report the mean across the three model sizes using that benchmark’s performance metric. AgentOPSD loses its ALFWorld advantage and PCSD loses its WebShop advantage under this rule, while OCSD retains a much smaller ScienceWorld advantage. Earlier checkpoints can also outperform the endpoint: 2B OVCSD on Search-QA reaches step 55 within budget, scoring 36.93% before its late greedy-decoding collapse. Time-budget assessment therefore reflects both the rate of acquiring performance and the ability to retain it. Table S15 reports both scores for every method and benchmark.

<table><tr><td rowspan="2">Method</td><td colspan="2">ALFWorld</td><td colspan="2">ScienceWorld</td><td colspan="2">Search-QA</td><td colspan="2">WebShop</td></tr><tr><td>Perf.</td><td>Time (h)</td><td>Perf.</td><td>Time (h)</td><td>Perf.</td><td>Time (h)</td><td>Perf.</td><td>Time (h)</td></tr><tr><td>Qwen3.5-2B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>28.48</td><td>13.49</td><td>0.83</td><td>4.95</td><td>32.38</td><td>2.00</td><td>70.80</td><td>3.89</td></tr><tr><td>GRPO+OPSD</td><td>22.23</td><td>15.68</td><td>0.83</td><td>5.62</td><td>34.22</td><td>2.03</td><td>77.05</td><td>3.87</td></tr><tr><td>SDAR</td><td>14.46</td><td>15.90</td><td>0.83</td><td>5.74</td><td>38.15</td><td>2.90</td><td>63.67</td><td>5.05</td></tr><tr><td>Skill-SD</td><td>74.29</td><td>14.18</td><td>0.83</td><td>5.53</td><td>37.56</td><td>2.56</td><td>71.19</td><td>4.22</td></tr><tr><td>RLSD</td><td>46.61</td><td>15.18</td><td>1.89</td><td>6.87</td><td>37.93</td><td>3.40</td><td>69.73</td><td>4.39</td></tr><tr><td>OCSD</td><td>42.32</td><td>13.28</td><td>13.78</td><td>6.04</td><td>12.96</td><td>0.97</td><td>75.59</td><td>3.73</td></tr><tr><td>OVCSD</td><td>56.96</td><td>22.23</td><td>13.85</td><td>7.98</td><td>33.21</td><td>3.54</td><td>1.07</td><td>5.23</td></tr><tr><td>AgentOPSD</td><td>47.50</td><td>15.73</td><td>0.83</td><td>5.48</td><td>34.49</td><td>2.58</td><td>70.51</td><td>4.70</td></tr><tr><td>ADRS</td><td>13.04</td><td>15.69</td><td>0.83</td><td>6.77</td><td>14.10</td><td>1.13</td><td>63.09</td><td>4.22</td></tr><tr><td>ICSD</td><td>54.73</td><td>15.26</td><td>0.83</td><td>6.11</td><td>39.39</td><td>3.47</td><td>70.51</td><td>4.38</td></tr><tr><td>PCSD</td><td>72.32</td><td>14.12</td><td>0.83</td><td>6.86</td><td>33.18</td><td>2.36</td><td>74.71</td><td>4.68</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>82.05</td><td></td><td>14.9432.16</td><td>8.62</td><td>45.15</td><td>3.99</td><td>66.02</td><td>5.36</td></tr><tr><td>GRPO+OPSD</td><td>1.52</td><td>24.14</td><td>13.65</td><td>9.02</td><td>44.16</td><td>4.67</td><td>77.15</td><td>6.48</td></tr><tr><td>SDAR</td><td>76.16</td><td>17.27</td><td>13.86</td><td>9.66</td><td>44.96</td><td>4.29</td><td>51.17</td><td>7.18</td></tr><tr><td>Skill-SD</td><td>55.27</td><td>18.61</td><td>41.67</td><td>8.25</td><td>42.81</td><td>4.76</td><td>0.00</td><td>7.03</td></tr><tr><td>RLSD</td><td>84.38</td><td>15.54</td><td>22.41</td><td>9.38</td><td>43.60</td><td>5.00</td><td>68.16</td><td>6.32</td></tr><tr><td>OCSD</td><td>52.32</td><td>17.75</td><td>41.77</td><td>9.04</td><td>44.73</td><td>4.08</td><td>64.16</td><td>5.41</td></tr><tr><td>OVCSD</td><td>76.43</td><td>20.81</td><td>14.93</td><td>11.39</td><td>43.51</td><td>4.91</td><td>73.73</td><td>6.50</td></tr><tr><td>AgentOPSD</td><td>75.00</td><td>22.62</td><td>32.80</td><td>9.84</td><td>44.25</td><td>4.82</td><td>72.07</td><td>6.45</td></tr><tr><td>ADRS</td><td>70.71</td><td>16.20</td><td>36.73</td><td>9.37</td><td>42.31</td><td>4.60</td><td>65.72</td><td>6.56</td></tr><tr><td>ICSD</td><td>82.50</td><td>19.57</td><td>15.99</td><td>10.72</td><td>45.87</td><td>5.02</td><td>0.00</td><td>8.63</td></tr><tr><td>PCSD</td><td>79.82</td><td>18.92</td><td>23.85</td><td>9.63</td><td>43.11</td><td>4.24</td><td>72.17</td><td>6.68</td></tr><tr><td colspan="9">Qwen3.5-9B</td></tr><tr><td>GRPO</td><td>62.68</td><td>22.16</td><td>58.99</td><td>11.18</td><td>46.59</td><td>5.93</td><td>70.02</td><td>8.09</td></tr><tr><td>GRPO+OPSD</td><td>71.79</td><td>24.16</td><td>51.98</td><td>12.65</td><td>47.08</td><td>7.05</td><td>74.90</td><td>9.13</td></tr><tr><td>SDAR</td><td>75.62</td><td>23.21</td><td>58.00</td><td>12.95</td><td>44.75</td><td>6.85</td><td>74.32</td><td>8.57</td></tr><tr><td>Skill-SD</td><td>74.55</td><td>24.65</td><td>29.08</td><td>13.19</td><td>46.19</td><td>6.39</td><td>0.00</td><td>9.87</td></tr><tr><td>RLSD</td><td>89.20</td><td>22.00</td><td>50.27</td><td>12.86</td><td>44.31</td><td>7.46</td><td>0.00</td><td>9.66</td></tr><tr><td>OCSD</td><td>86.16</td><td>21.19</td><td>44.30</td><td>11.39</td><td>48.16</td><td>6.04</td><td>67.77</td><td>8.75</td></tr><tr><td>OVCSD</td><td>83.39</td><td>23.22</td><td>43.45</td><td>16.50</td><td>45.43</td><td>7.28</td><td>68.07</td><td>9.29</td></tr><tr><td>AgentOPSD</td><td>78.04</td><td>23.94</td><td>42.48</td><td>12.44</td><td>45.81</td><td>6.18</td><td>72.27</td><td>9.45</td></tr><tr><td>ADRS</td><td>78.93</td><td>22.39</td><td>35.63</td><td>13.16</td><td>43.93</td><td>6.99</td><td>70.31</td><td>8.90</td></tr><tr><td>ICSD</td><td>89.64</td><td>24.13</td><td>18.26</td><td>14.07</td><td>46.84</td><td>7.70</td><td>72.46</td><td>9.84</td></tr><tr><td>PCSD</td><td>72.77</td><td>25.25</td><td>58.60</td><td>12.67</td><td>45.42</td><td>6.64</td><td>74.32</td><td>8.91</td></tr></table>

Table S13 Numerical values for the points in Figure 3. Perf. is mean final performance over eight rollout seeds, using the benchmark metrics in Tables 2 and 3; Time is net training time in hours.

## G.2 Total Compute and Stage-Level Costs

GPU-hours and training workload. Across 132 runs, the study records 13,200 training steps and 2,772 validation sweeps, including initial validation. The training loops consume 12,124.4 H800 GPU-hours, computed as elapsed hours multiplied by the eight GPUs assigned to each run and summed across runs. Training accounts for 10,580.7 GPU-hours, in-loop validation for 1,515.3, and checkpoint saving for 28.4. This total separates training, in-loop validation, and checkpoint saving; it excludes initialization, initial validation, and subsequent repeated evaluation. Table S16 reports the breakdown by model size and benchmark, distinguishing the experimental expenditure from the net training cost used in method comparisons.

<table><tr><td>Method</td><td>Mean</td><td>Time (×)</td><td>GPU-h</td></tr><tr><td>GRPO</td><td>49.68</td><td>1.000</td><td>836.7</td></tr><tr><td>GRPO+OPSD</td><td>43.05</td><td>1.158</td><td>996.1</td></tr><tr><td>SDAR</td><td>46.33</td><td>1.183</td><td>956.5</td></tr><tr><td>Skill-SD</td><td>39.45</td><td>1.153</td><td>954.0</td></tr><tr><td>RLSD</td><td>46.54</td><td>1.208</td><td>944.5</td></tr><tr><td>OCSD</td><td>49.50</td><td>0.999</td><td>861.2</td></tr><tr><td>OVCSD</td><td>46.17</td><td>1.370</td><td>1111.1</td></tr><tr><td>AgentOPSD</td><td>51.34</td><td>1.187</td><td>993.9</td></tr><tr><td>ADRS</td><td>44.61</td><td>1.100</td><td>927.8</td></tr><tr><td>ICSD</td><td>44.75</td><td>1.293</td><td>1031.2</td></tr><tr><td>PCSD</td><td>54.26</td><td>1.167</td><td>967.6</td></tr></table>

Table S14 Final performance and training cost across twelve settings. Mean uses the benchmark metrics in Tables 2 and 3; Time averages paired training-time ratios to GRPO. GPU-h sums twelve training runs, excluding validation and saving.

<table><tr><td rowspan="2">Method</td><td colspan="2">ALFWorld</td><td colspan="2">ScienceWorld</td><td colspan="2">Search-QA</td><td colspan="2">WebShop</td></tr><tr><td>Step 100</td><td>Budget</td><td>Step 100</td><td>Budget</td><td>Step 100</td><td>Budget</td><td>Step 100</td><td>Budget</td></tr><tr><td>GRPO</td><td>58.10</td><td>58.10</td><td>30.36</td><td>30.36</td><td>41.22</td><td>41.22</td><td>68.20</td><td>68.20</td></tr><tr><td>GRPO+OPSD</td><td>32.13</td><td>31.67</td><td>22.20</td><td>21.74</td><td>41.87</td><td>41.85</td><td>76.57</td><td>72.13</td></tr><tr><td>SDAR</td><td>54.50</td><td>54.50</td><td>23.94</td><td>21.82</td><td>42.91</td><td>41.15</td><td>63.27</td><td>61.70</td></tr><tr><td>Skill-SD</td><td>67.60</td><td>66.70</td><td>23.94</td><td>27.00</td><td>42.45</td><td>41.61</td><td>23.43</td><td>23.17</td></tr><tr><td>RLSD</td><td>72.60</td><td>77.37</td><td>24.89</td><td>22.67</td><td>41.82</td><td>41.50</td><td>45.07</td><td>47.13</td></tr><tr><td>OCSD</td><td>58.57</td><td>54.07</td><td>33.63</td><td>30.71</td><td>35.09</td><td>34.91</td><td>69.30</td><td>71.63</td></tr><tr><td>OVCSD</td><td>72.60</td><td>59.77</td><td>24.78</td><td>18.99</td><td>30.26</td><td>42.04</td><td>47.90</td><td>44.80</td></tr><tr><td>AgentOPSD</td><td>68.10</td><td>37.63</td><td>26.14</td><td>24.87</td><td>41.42</td><td>41.83</td><td>71.87</td><td>70.83</td></tr><tr><td>ADRS</td><td>55.70</td><td>54.07</td><td>25.20</td><td>30.37</td><td>33.71</td><td>32.59</td><td>66.90</td><td>61.20</td></tr><tr><td>ICSD</td><td>76.17</td><td>77.37</td><td>11.82</td><td>19.51</td><td>41.41</td><td>43.07</td><td>48.17</td><td>42.97</td></tr><tr><td>PCSD</td><td>74.30</td><td>68.07</td><td>27.39</td><td>27.46</td><td>40.09</td><td>40.96</td><td>73.17</td><td>63.83</td></tr></table>

Table S15 Validation performance at step 100 and within the matched GRPO training-time budget, averaged over the three model sizes separately for each benchmark. Budget selects the latest logged validation reached within each setting's budget. ScienceWorld reports task score; the other benchmarks report success or exact-match accuracy (%). Each comparison retains its benchmark's validation decoder.

<table><tr><td>Group</td><td>Runs</td><td>Training</td><td>Validation</td><td>Saving</td><td>Total</td></tr><tr><td>All runs</td><td>132</td><td>10580.7</td><td>1515.3</td><td>28.4</td><td>12124.4</td></tr><tr><td>Qwen3.5-2B</td><td>44</td><td>2511.6</td><td>282.0</td><td>5.3</td><td>2798.9</td></tr><tr><td>Qwen3.5-4B</td><td>44</td><td>3474.4</td><td>484.7</td><td>9.2</td><td>3968.4</td></tr><tr><td>Qwen3.5-9B</td><td>44</td><td>4594.6</td><td>748.6</td><td>13.8</td><td>5357.1</td></tr><tr><td>ALFWorld</td><td>33</td><td>5067.3</td><td>408.3</td><td>12.2</td><td>5487.8</td></tr><tr><td>ScienceWorld</td><td>33</td><td>2527.5</td><td>602.6</td><td>1.0</td><td>3131.1</td></tr><tr><td>Search-QA</td><td>33</td><td>1214.8</td><td>377.5</td><td>1.4</td><td>1593.6</td></tr><tr><td>WebShop</td><td>33</td><td>1771.2</td><td>126.9</td><td>13.8</td><td>1911.9</td></tr></table>

Table S16 Recorded H800 GPU-hours for all 132 training runs across four benchmarks. Model-size and benchmark blocks separately partition the same total. Training includes rollout and teacher computation; Total additionally includes in-loop validation and saving.

Distillation overhead extends beyond teacher forward passes. Teacher supervision introduces work at diferent stages of training, depending on how each method uses it. Across all twelve settings, OVCSD’s verified continuations contribute to rollout generation, whose mean time is 218.0 seconds per step versus GRPO’s 124.9; OCSD’s paired teacher replay contributes to advantage computation, averaging 10.6 seconds versus 0.2 for GRPO. These measurements show why separately timed teacher passes cannot represent the full cost of incorporating teacher information. Eficiency should be judged using end-to-end training time, with stage-level measurements used to locate the additional work. Figure S3 decomposes the absolute GPU-hours into rollout generation, policy/reference scoring, teacher forward passes, advantage computation, policy updates, and other processing.

## H Literature Survey

## H.1 Search Strategy and Selection Criteria

Scope. Our survey characterizes recent research on distillation for training multi-turn language-model agents that interact through tools or environments. It covers text and GUI agents trained with privileged-context supervision, other on-policy teacher distillation, or ofline and iterative trajectory distillation, together with empirical analyses of these approaches. The common focus is the use of distillation to update the agent’s action policy. Within this broader landscape, our benchmark compares GRPO with ten representative methods whose main contributions target specific components of text-agent training, including auxiliary objectives, credit assignment, reward shaping, and verified teacher continuations (Appendix B).

Search and coverage. We examine first submissions from 25 June to 22 September 2026 (UTC), a 90-day period, using arXiv metadata retrieved on 23 September. Two title-and-abstract searches pair agent-related terms with distillation, and self-distillation with reinforcement learning. Deduplication and date filtering yield 635 records, from which the agent–distillation query and interactive-agent keywords identify 84 candidates. Reviewing these 84 candidates’ titles and abstracts, with full-text inspection where needed, yields 47 papers that meet the survey scope and form the final inventory. As a coverage check, we verify that it includes every source paper of the evaluated methods first submitted within this period.

## H.2 Publication Statistics

Counting each unique arXiv paper once by its first-submission date gives 47 papers over 90 days, or approximately one paper every two days (0.5222 per day). The inventory includes 32 papers on privilegedcontext policy supervision (P), 10 on other on-policy distillation and related empirical frameworks (O), and 5 on trajectory-based policy distillation (T). The full inventory, with titles, first-submission dates, supervision categories, and arXiv links, appears in Tables S17–S20.

<table><tr><td>Submitted</td><td>arXiv ID</td><td>Title</td><td>Route</td></tr><tr><td>2026-06-25</td><td>2606.26790</td><td>OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning</td><td>P</td></tr><tr><td>2026-06-26</td><td>2606.27814</td><td>ATOD: Annealed Turn-Aware On-Policy Distillation for Multi-Turn Agentic Tasks</td><td>O</td></tr><tr><td>2026-06-28</td><td>2606.29476</td><td>CRAFT: Counterfactual Credit Assignment from Free Sibling Rollouts for Self-Distilled Agentic Reinforcement Learning</td><td>P</td></tr><tr><td>2026-06-28</td><td>2606.29502</td><td>UCOB: Learning to Utilize and Evolve Agentic Skills via Credit-Aware On-Policy Bidirectional Self-Distillation</td><td>P</td></tr><tr><td>2026-06-29</td><td>2606.29863</td><td>KbSD: Knowledge Boundary aware Self-Distillation for Behavioral Calibra-</td><td>P</td></tr><tr><td>2026-06-29</td><td>2606.29961</td><td>tion in Agentic Search DuoMem: Towards Capable On-Device Memory Agents via Dual-Space</td><td>T</td></tr><tr><td>2026-06-29</td><td>2606.30044</td><td>Distillation Building Multi-Task Agentic LLMs via Two-Phase Distillation</td><td>0</td></tr><tr><td>2026-07-05</td><td>2607.04425</td><td>UI-MOPD: Multi-Platform On-Policy Distillation for Unified GUI Agents</td><td>O</td></tr><tr><td>2026-07-07</td><td>2607.05804</td><td>TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long- Horizon Agent Training</td><td>0</td></tr><tr><td>2026-07-08</td><td>2607.07820</td><td>DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment</td><td>T</td></tr><tr><td>2026-07-13</td><td>2607.11012</td><td>EasyOPD: An Easy-to-use On-Policy Distillation Framework for Large Language Models</td><td>O</td></tr><tr><td>2026-07-16</td><td>2607.14777</td><td>SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learn- ing</td><td>P</td></tr></table>

Table S17 Papers included in the literature survey (1/4). Dates indicate first submissions (UTC). P: privileged-context supervision; O: other on-policy distillation; T: trajectory distillation.

<table><tr><td>Submitted</td><td>arXiv ID</td><td>Title</td><td>Route</td></tr><tr><td>2026-07-20</td><td>2607.17558</td><td>Why Does Feedback-Augmented Self-Distillation Fail to Improve Retrieval- Interleaved Search Agents?</td><td>P</td></tr><tr><td>2026-07-27</td><td>2607.24280</td><td>From Proprietary to Open-Source: Bridging the Distribution Gap via Multi- Agent Protocol Distillation in Agentic Search</td><td>P</td></tr><tr><td>2026-07-27</td><td>2607.24720</td><td>The Physics of Multi-Turn Long-Horizon Planning: From Pre-training to Post-training via Single- and Multi-Teacher On-Policy Agentic Distillation</td><td>O</td></tr><tr><td>2026-07-30</td><td>2607.27937</td><td>From Scoring to Acting: Outcome-Verified Comparative Self-Distillation for LLM Agents</td><td>P</td></tr><tr><td>2026-07-30</td><td>2607.28026</td><td>Contrastive Reinforced Policy Optimization via Privileged Self-Distillation</td><td>P</td></tr><tr><td>2026-07-30</td><td>2607.28076</td><td>Group-Reflective Self-Distillation for Agentic Reinforcement Learning</td><td>P</td></tr><tr><td>2026-07-31</td><td>2607.29320</td><td>MAGA: Multi-Platform Self-Fusion of GUI Agents via Structured Action Distillation</td><td>O</td></tr><tr><td>2026-08-01</td><td>2608.00485</td><td>SERL-SQL: Selective Hindsight Distillation for Text-to-SQL Reinforcement Agentic Learning</td><td>P</td></tr><tr><td>2026-08-02</td><td>2608.01359</td><td>EviSD: Evidence-Conditioned Self-Distillation for Search-Augmented Agents</td><td>P</td></tr><tr><td>2026-08-03</td><td>2608.01597</td><td>HindSearch: Trajectory-Level Hindsight Critique for Search-Augmented Reinforcement Learning</td><td>P</td></tr><tr><td>2026-08-03</td><td>2608.01837</td><td>PCSD: Persistent Consistency for Self-Distillation in Agentic Reinforcement Learning</td><td>P</td></tr><tr><td>2026-08-03</td><td>2608.01953</td><td>Look Ahead Before You Distill: Future Trajectory Validation of Teacher Guidance for Agentic On-Policy Distillation</td><td>O</td></tr></table>

Table S18 Papers included in the literature survey (2/4). Dates indicate first submissions (UTC). P: privileged-context supervision; O: other on-policy distillation; T: trajectory distillation.

<table><tr><td>Submitted</td><td>arXiv ID</td><td>Title</td><td>Route</td></tr><tr><td>2026-08-04</td><td>2608.03223</td><td>Agentic Reinforcement Learning with Self-Distilled Reward Shaping</td><td>P</td></tr><tr><td>2026-08-04</td><td>2608.04007</td><td>TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Rea- soning</td><td>P</td></tr><tr><td>2026-08-05</td><td>2608.05219</td><td>When Privileged Guidance Misaligns: State-Matched Routing and Contex- tualized Self-Distillation for Multi-Turn Agents</td><td>P</td></tr><tr><td>2026-08-05</td><td>2608.04788</td><td>Agentic Reinforcement Learning with Observation-Calibrated Self-</td><td>P</td></tr><tr><td>2026-08-05</td><td>2608.04794</td><td>Distillation Privileged, but Biased: How PI-Conditioned Teachers Break Self-Distillation</td><td>P</td></tr><tr><td>2026-08-06</td><td>2608.05987</td><td>AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning</td><td>P</td></tr><tr><td>2026-08-06</td><td>2608.06065</td><td>The Next Screenshot Knows: Gated Hindsight Distillation for Mobile GUI</td><td>P</td></tr><tr><td>2026-08-07</td><td>2608.07068</td><td>Agents MemOPD: On-Policy Distillation through Memory State Alignment for Long-Horizon Agents</td><td>O</td></tr><tr><td>2026-08-07</td><td>2608.07371</td><td>Trajectory-Relative Hindsight Distillation for Agentic Reinforcement Learn-</td><td>P</td></tr><tr><td>2026-08-09</td><td>2608.08570</td><td>ing FailForge: Distilling Procedural Competence from Persistent Failures into</td><td>T</td></tr><tr><td>2026-08-09</td><td>2608.08960</td><td>Code Agents Reading is not Reasoning: Bridging the Agentic Policy Gap in Vision-Text</td><td>P</td></tr><tr><td>2026-08-10</td><td>2608.09555</td><td>Compression Bidirectional Context Self-Distillation for Reinforcement Learning of Skill- Based LLM Agents</td><td>P</td></tr></table>

Table S19 Papers included in the literature survey (3/4). Dates indicate first submissions (UTC). P: privileged-context supervision; O: other on-policy distillation; T: trajectory distillation.

<table><tr><td>Submitted</td><td>arXiv ID</td><td>Title</td><td>Route</td></tr><tr><td>2026-08-12</td><td>2608.11967</td><td>LoongReflect: Boosting Long-Horizon Reflection in Search Agents via Global Perspective Distillation</td><td>P</td></tr><tr><td>2026-08-13</td><td>2608.12764</td><td>Beyond Outcome Rewards: Step-Level Self-Distilled Policy Optimization for Deep Search Agents</td><td>P</td></tr><tr><td>2026-08-14</td><td>2608.14945</td><td>Trust Is Not Enough: Influence Calibration for On-Policy Self-Distillation in Agentic RL</td><td>P</td></tr><tr><td>2026-08-19</td><td>2608.18524</td><td>DART-SD: Diamond-topology Aware Retrieval and Tuning for Self- Distillation of Multi-Turn Tool-Calling Agents</td><td>P</td></tr><tr><td>2026-08-22</td><td>2608.21946</td><td>EDGE: Experience-Distillation for Guided Exploration in Agentic Reinforce- ment Learning</td><td>P</td></tr><tr><td>2026-08-25</td><td>2608.24114</td><td>AHEAD: Adaptive Hindsight with Environment-Augmented Distillation for Agentic RL</td><td>P</td></tr><tr><td>2026-08-29</td><td>2608.29357</td><td>LiteSearch-VL: Small Multimodal Search Agents via Trajectory Distillation and Synthetic Step-DPO</td><td>T</td></tr><tr><td>2026-08-31</td><td>2608.31077</td><td>Reconciling Process Supervision with Outcome-Based Credit in Agentic Policy Optimization</td><td>P</td></tr><tr><td>2026-09-13</td><td>2609.14636</td><td>Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation</td><td>O</td></tr><tr><td>2026-09-17</td><td>2609.20784</td><td>RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement</td><td>P</td></tr><tr><td>2026-09-21</td><td>2609.24974</td><td>Learning Harness-Zero: Harness Distillation via Agent-as-Harness</td><td>T</td></tr></table>

Table S20 Papers included in the literature survey (4/4). Dates indicate first submissions (UTC). P: privileged-context supervision; O: other on-policy distillation; T: trajectory distillation.