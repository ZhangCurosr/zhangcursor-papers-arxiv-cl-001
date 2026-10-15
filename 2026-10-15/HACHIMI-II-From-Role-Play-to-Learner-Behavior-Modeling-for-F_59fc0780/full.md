# HACHIMI-II: From Role-Play to Learner Behavior Modeling for Faithful Student Simulation

Yilin Jiang<sup>1,2</sup>, Fei Tan<sup>1</sup>\*, Xuanyu Yin<sup>1</sup>, Jing Leng<sup>1</sup>, Aimin Zhou<sup>1,3</sup>

<sup>1</sup>East China Normal University, Shanghai, China

<sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China <sup>3</sup>Shanghai Innovation Institute, Shanghai, China ftan@mail.ecnu.edu.cn

## Abstract

Simulated students let researchers test tutoring systems and teaching strategies before real learners are available, but only if the simulation reproduces where a particular learner succeeds and fails. The common approach asks a large language model to role-play a student, but a role does not model what this learner knows or will do next, so the model’s own competence fills the gap. Simulated students perform at a level set by the model, differ little from one another, and seldom make the real learners’ errors. We model the learner’s behavior and recast student simulation as third-person prediction of the learner’s next action. HACHIMI-II builds a student model from the learner’s history, infers the learner’s state for the current item or tutor turn, and compiles it into a behavior harness that constrains the next action. On three item-response datasets, the Wasserstein-1 distance between simulated and real student ability is 2.4–8.4× smaller than that of the strongest baseline. On two dialogue datasets, HACHIMI-II reproduces real student errors with the same knowledge type, locus, and mechanism 1.8– 5.1× more often than the strongest baseline, and ablations trace the gains mainly to the explicit learner state. Predicting the learner makes capable models usable as simulated students.

## 1 Introduction

Simulated students let researchers evaluate educational content and compare teaching strategies before any real learner is involved, and they support the development of tutoring systems when real interaction data is scarce (Käser and Alexandron, 2024; Scarlatos et al., 2026). To serve these uses, a simulated student has to behave like the learner it stands for. It should answer some items correctly, miss others, and make the errors that the real learner makes.

Large language models can write student answers and dialogue turns, and the usual way to obtain them is role prompting, which asks the model to act as a student of a given level (Benedetto et al., 2024; Scarlatos et al., 2026). However, a role prompt names a level without modeling what this learner knows or how the learner will respond to the current item or tutor turn. An aligned model is trained to be correct and helpful (Ouyang et al., 2022; Bai et al., 2022) and already knows the content that the student is supposed to lack, so it fills the gap with its own knowledge and a generic picture of a student. As Figure 1 illustrates, it then either solves the problem despite the assigned role or inserts an error at a step where the learner would not fail. This failure has been described as the competence paradox (Yuan et al., 2026) and as expert knowledge leakage (Song et al., 2026), and we call its observable effect competence leakage. Under leakage, the ability level is set by the model, differences among learners largely disappear, and real errors are rarely reproduced.

Existing remedies add learner profiles (Lu and Wang, 2024), agent memory (Xu et al., 2024), or prompt-level constraints on what the student knows. They enrich the description of the student, but the model is still asked to act as the student and still has to suppress what it knows. Evaluations of prompting-based simulators report limited fidelity to real student behavior (Scarlatos et al., 2026).

In this paper, we counter competence leakage by modeling the learner’s behavior, which changes the task framing from role-play to prediction. A request to act as a weak student asks the model to suppress knowledge it has. A request to predict what a particular learner will do next asks the model to use that knowledge, because the model must know the correct solution to judge where this learner will depart from it. Following this framing, we propose HACHIMI-II (Figure 1). It describes the learner in the third person through a student model built from observed evidence, infers the learner’s state for the current item or tutor turn, compiles that state into a behavior harness, and generates the next action within the bounds that the harness sets on knowledge, reasoning depth, errors, and commitment. HACHIMI-II takes its name from HACHIMI (Jiang et al., 2026), whose corpus of one million student personas provides its learner prior, and adds the state inference and harness stages that turn a persona into a predicted action.

![](images/83b62a96886ac7a9b9c79c55f0160ee76e1ec940cfa34b49232b5ba2e8e68096.jpg)  
Figure 1: Role-prompted simulation (top left) and the four steps of HACHIMI-II on a linear equation.

We evaluate HACHIMI-II against role-prompted baselines on five datasets in two formats. In item-response simulation on XES3G5M, Eedi, and ENEM, the Wasserstein-1 distance between the simulated and real distributions of per-student accuracy is 2.4–8.4× smaller than that of the strongest baseline, and HACHIMI-II recovers the ranking of individual students. In dialogue simulation on Eedi QATD2k and Khan/CoMTA, it reproduces real errors with the same knowledge type, locus, and mechanism 1.8–5.1× more often than the strongest baseline and more than doubles the F1 of error-state prediction. It also attains the highest agreement with the real learners on all five datasets.

This paper makes three contributions. First, we show that competence leakage in role-prompted simulation affects the ability level, the differences among learners, and their errors, and we recast the task as third-person prediction of the learner’s next action. Second, we propose HACHIMI-II, which models the learner’s behavior with a student model, a learner state inferred at each step, and a behavior harness compiled deterministically from that state. Third, on three item-response and two dialogue datasets, ablations show that the three components govern different parts of the simulated behavior, with the learner state as the main source of the gains.

## 2 Related Work

Learner modeling and student simulation. Real learner traces are costly to collect (Choi et al., 2020; Scarlatos et al., 2026). Early simulators used hand-specified cognitive models such as models of human memory (Reddy et al., 2017), and learner models estimate proficiency through probabilistic mastery (Corbett and Anderson, 1994), item response theory (Lord, 1980), or recurrent, memoryaugmented, and reinforcement-learning architectures (Piech et al., 2015; Zhang et al., 2017; Zhao et al., 2023). These models output correctness or mastery. The open-ended answers and dialogue turns that tutoring research needs are generated with language models.

LLM-based simulated students. Role prompting asks a language model to answer as a student of a given level (Benedetto et al., 2024; Scarlatos et al., 2026), profile-conditioned methods add knowledge states, misconceptions, histories, or personas (Lu and Wang, 2024; Jiang et al., 2026), and agentbased simulators add memory, reflection, and action modules (Xu et al., 2024; Gao et al., 2025). Marquez-Carpintero et al. (2026) review these architectures. Training-based simulators fine-tune a model on student data (Qi et al., 2026) or specialize a pooled simulator to each student (Yang et al., 2026), and Song et al. (2026) unlearn knowledge so that the model answers like a novice. Simulated students, including fine-tuned ones, still show limited behavioral fidelity beyond answer accuracy (Scarlatos et al., 2026; Xu and Zhang, 2026). HACHIMI-II needs no training and keeps the model’s knowledge, which it uses to predict where the learner will fail.

Three lines of work are closest to ours. Wu et al. (2025) build a cognitive prototype of each student from past learning records and refine a simulated solution toward the predicted performance. Yuan et al. (2026) name the competence paradox and argue that valid simulation requires an explicit specification of the student’s epistemic state. Meng and Lin (2026) condition simulated students on coarse engagement states. HACHIMI-II infers a finer state at every item and tutor turn, covering the predicted error, commitment, and help-seeking, and compiles it into constraints on the next action. Beyond answer-level ability alignment (Srivatsa et al., 2025) and teacher judgments of realism (Martynova et al., 2025), we also compare the errors themselves in knowledge type, locus, and mechanism.

Constraints and harnesses for generation. In agent systems, a harness is an execution layer outside the model that organizes context, tools, memory, and verification (Li et al., 2026; Pan et al., 2026; Zhong and Zhu, 2026). Constrained generation and language-model programming shape outputs through explicit constraints and control flow (Gârbacea and Mei, 2025; Beurer-Kellner et al., 2023). HACHIMI-II applies this idea to behavior simulation. Its constraints are derived from a predicted learner state, and they restrict both the content and the form of the next action.

## 3 HACHIMI-II

HACHIMI-II models the learner’s behavior by predicting the next observable action in the four steps shown in Figure 1. The same design serves itemresponse and dialogue simulation, with state and harness fields specific to each format (Appendix B).

## 3.1 Problem Formulation

At step t the simulator observes $\mathcal { V } _ { t } = \{ c _ { t } ^ { L } , c _ { t } ^ { T } , h _ { t } \}$ where $c _ { t } ^ { L }$ is the learner evidence available before the step, $c _ { t } ^ { T }$ is the current item or tutor turn, and $h _ { t }$ is the visible history. The target action $a _ { t }$ is the learner’s response to one item in item-response simulation and the learner’s next turn in dialogue simulation. Generation uses only $\nu _ { t }$ . HACHIMI-II computes

$$
\begin{array} { r } { \begin{array} { c l l } { m _ { t } = \mathcal { M } _ { \mathcal { P } ^ { \mathrm { H } } } ( c _ { t } ^ { L } , h _ { t } ) , } & { s _ { t } = \mathcal { S } ( m _ { t } , c _ { t } ^ { T } , h _ { t } ) , } \\ { b _ { t } = \mathcal { B } ( s _ { t } ) , } & { a _ { t } = \mathcal { G } ( \mathcal { V } _ { t } , m _ { t } , s _ { t } , b _ { t } ) , } \end{array} } \end{array}\tag{1}
$$

where $m _ { t }$ is the student model, ${ \mathcal { P } } ^ { \mathrm { H } }$ is the HACHIMI-1M persona corpus (Jiang et al., 2026), $s _ { t }$ is the learner state, $b _ { t }$ is the behavior harness, and $a _ { t }$ is the generated action. M, S, and G are prompted calls to the backbone model, and B is a deterministic mapping.

## 3.2 Student Model

The student model $m _ { t }$ describes who the learner is. It combines an observed profile, summarized from the learner’s visible history, with a HACHIMI-1M persona that matches this profile (Appendix A). The observed profile is the main evidence about the learner’s ability and habits, and the persona adds a prior on longer-term background such as motivation, confidence, and interaction style. The student model is written in the third person, so the backbone reasons about the learner.

## 3.3 Learner State Inference

The learner state $s _ { t }$ predicts how the learner will handle the current item or tutor turn. The state has four variables, shown in Figure 1 and Table 1, and each follows an established line of educational research. Following evidence-centered design (Mislevy et al., 2003; Mislevy and Riconscente, 2005), the reachable knowledge lists the prerequisites of the current item that the learner has mastered. Following the zone of proximal development (Vygotsky, 1978; Wood et al., 1976), it also records how far the learner’s reasoning can reach and how much of the tutor’s support the learner takes up. Error analysis locates failures in the solution process and traces them to their source (Newman, 1977; Radatz, 1979), and the active error pattern accordingly predicts whether the learner will err and, if so, at which step and through which mechanism. Following research on self-regulated learning and help-seeking (Zimmerman, 2002; Panadero, 2017; Aleven et al., 2016), the confidence state records how firmly the learner commits to an answer, and the local interaction move records whether the learner attempts the step or asks for help.

<table><tr><td>State variable</td><td>Harness constraint on the next action</td></tr><tr><td>Reachable</td><td>Accessible knowledge  $K _ { t }$  and</td></tr><tr><td>knowledge</td><td>reasoning depth  $R _ { t }$  of the action</td></tr><tr><td>Active error pattern</td><td>Error preservation  $E _ { t }$  at the predicted</td></tr><tr><td>Confidence state</td><td>step Uncertainty  $U _ { t }$  of the answer</td></tr><tr><td>Local interaction</td><td>Response mode  $M _ { t }$  and step budget</td></tr><tr><td>move</td><td> $L _ { t }$  of the turn</td></tr></table>

Table 1: Learner state variables and the harness constraints compiled from them, named as in Figure 1.

## 3.4 Behavior Harness Compilation

The compiler B turns the learner state into explicit constraints on the next action. It is a fixed mapping, so the same state always yields the same harness, and the generator receives concrete limits on what it may write. For example, an explanation depth of none compiles to a limit of one sentence without a full derivation, and a predicted error at the process-skills step compiles to a requirement that the answer contain one concrete error at that step. Table 1 summarizes the mapping, and Appendix B lists the harness fields for each format.

## 3.5 Harnessed Generation

The generator $\mathcal { G }$ receives the visible context, the student model, the state, and the harness, and writes the action. The generated action is checked against the harness and regenerated when it violates a constraint (Appendices C and H).

Table 2 follows one XES3G5M item through the four steps. The state predicts that the learner can compute the travel times but will average the two speeds, and the generated answer carries this error. The real learner answered this item incorrectly, while all three baselines solved it.

## 4 Experimental Setup

Datasets. Table 3 lists the five datasets. The item-response datasets pair real students with items and record whether each response is correct. XES3G5M contains third-grade mathematics responses in Chinese, Eedi contains multiple-choice diagnostic questions in mathematics, and ENEM

<table><tr><td colspan="2">Item. Two towns are 300 m apart. A student walks there at 10 m/min and back at 15 m/min. What is the average speed of the round trip in m/min? (Answer: 12)</td></tr><tr><td>Evidence</td><td>30 earlier items with accuracy 0.80, and 0.43 on word problems; accuracy drops on harder items</td></tr><tr><td>Student model</td><td>Grade 3; confident on routine arithmetic; low confidence and reliance on help on unfamiliar problems</td></tr><tr><td>Learner state</td><td>Prerequisites familiar; item partly within reach; active error pattern: averages the two speeds,  $( 1 0 + 1 5 ) \hat { / } 2 = 1 2 . 5 ;$  low</td></tr><tr><td>Harness</td><td>commitment; short explanation One final answer after a single reasoning step, in at most three sentences</td></tr><tr><td>Action</td><td>Reasoning step: 30 minutes there, 20 minutes back, average of 10 and 15. Answer: 12.5</td></tr><tr><td>Outcome</td><td>The real learner answered incorrectly. Direct RP, Prompt-C, and Profile-C all answer 12.</td></tr></table>

Table 2: One XES3G5M item traced through the four steps of HACHIMI-II.

<table><tr><td>Format</td><td>Dataset</td><td>Instances</td></tr><tr><td>Item response</td><td>XES3G5M (Liu et al., 2023)</td><td> $6 0 \times 9 6$ </td></tr><tr><td>Item response</td><td>Eedi (Wang et al., 2021)</td><td> $4 8 0 \times 1 0 7$ </td></tr><tr><td>Item response</td><td>ENEM 2024 Natural Sciences (INEP 2024)</td><td> $7 { , } 2 0 0 \times 2 4$ </td></tr><tr><td>Dialogue</td><td>Eedi QATD2k (Zent et al., 2025)</td><td>17,256</td></tr><tr><td>Dialogue</td><td>Khan/CoMTA (Miller and DiCerbo, 2024)</td><td>917</td></tr></table>

Table 3: Datasets. Item-response instances are students × items, and dialogue instances are student turns.

2024 is the natural-sciences section of the Brazilian national secondary-school exam. The two dialogue datasets contain real mathematics tutoring sessions, with human tutors in Eedi QATD2k and an LLM tutor in Khan/CoMTA, and each instance is one student turn predicted from the preceding dialogue.

Item-response metrics. For each student–item pair $n ,$ , the real label $y _ { n } \in \{ 0 , 1 \}$ records whether the student answered correctly, and the simulated label ${ \hat { y } } _ { n }$ is obtained by matching the normalized simulated answer against the answer key. Let acc<sub>i</sub> and $\widehat { \mathrm { a c c } } _ { i }$ be the real and simulated accuracy of student i over the evaluated items. We report response agreement Agree $\begin{array} { r } { \mathbf { \mu } = \frac { 1 } { N } \sum _ { n } \mathbf { 1 } [ y _ { n } = \hat { y } _ { n } ] } \end{array}$ over the N pairs, the Spearman correlation $\rho$ between acc<sub>i</sub> and ${ \widehat { \operatorname { a c c } } } _ { i } ,$ and the Wasserstein-1 distance (Villani, 2009) between their distributions over the S students,

$$
W _ { 1 } = \frac { 1 } { S } \sum _ { k = 1 } ^ { S } \bigl | \operatorname { a c c } _ { ( k ) } - \widehat { \operatorname { a c c } } _ { ( k ) } \bigr | ,\tag{2}
$$

![](images/b47a85de24ed1e38e23f9681aa5ddd4386d176c015d2ef032a5fbf17f9024577.jpg)  
Figure 2: Real and simulated accuracy per student for Profile-C and HACHIMI-II. The dashed line is $y = x ,$ , and the solid line is a linear fit.

<table><tr><td>Dataset</td><td>Method</td><td>Agree↑</td><td> $W _ { 1 } \downarrow$ </td><td> $\rho \uparrow$ </td></tr><tr><td>XES3G5M</td><td>Direct RP Prompt-C Profile-C HACHIMI-II</td><td>0.534 0.533 0.526 0.752</td><td>0.327 0.332 0.340 0.039</td><td>0.006 -0.071 0.220 0.691</td></tr><tr><td>Eedi</td><td>Direct RP Prompt-C Profile-C HACHIMI-II</td><td>0.502 0.509 0.521 0.606</td><td>0.160 0.155 0.119 0.046</td><td>-0.057 0.311 0.698 0.703</td></tr><tr><td>ENEM</td><td>Direct RP Prompt-C Profile-C HACHIMI-II</td><td>0.473 0.475 0.483 0.630</td><td>0.432 0.426 0.409 0.169</td><td>-0.008 0.303 0.473 0.668</td></tr></table>

Table 4: Item-response results. Best per dataset in bold.

where $\operatorname { a c c } ( \boldsymbol { k } )$ and $\widehat { \mathrm { a c c } } _ { ( k ) }$ are the k-th smallest real and simulated accuracies. $W _ { 1 }$ measures whether the simulated population has the right ability level and spread, and $\rho$ measures whether it orders individual students correctly. Appendix F reports further metrics.

Dialogue metrics. A judge model, GPT-5.4, labels each real and simulated turn with the five dimensions in Table 5. It assigns a functional label $f _ { n } \in$ {correct, incorrect, NA} and an error indicator $e _ { n } ~ \in ~ \{ 0 , 1 \}$ that marks a mathematical error classifiable in our taxonomy. For each error it assigns three structure labels $\ell _ { n } =$ $( \ell _ { n } ^ { \mathrm { t y p e } } , \ell _ { n } ^ { \mathrm { l o c u s } } , \ell _ { n } ^ { \mathrm { m e c h } } )$ , namely the knowledge type from the revised Bloom taxonomy (Anderson and Krathwohl, 2001), the error locus from Newman’s error analysis (Newman, 1977), and the error mechanism from Radatz’s classification (Radatz, 1979). The judge was selected from five candidate models by its agreement with three human annotators on 200 turns per dimension, and its agreement with their majority label is within 0.06 of the agreement among the annotators on every dimension (Table 5, Appendix D).

<table><tr><td>Dimension</td><td>Labels</td><td>Human Judge</td><td></td></tr><tr><td>Functional label</td><td>correct; incorrect; not applicable</td><td>0.84</td><td>0.83</td></tr><tr><td>Error indicator</td><td>yes; no</td><td>0.81</td><td>0.80</td></tr><tr><td>Knowledge type</td><td>factual; conceptual; procedural; metacognitive</td><td>0.73</td><td>0.69</td></tr><tr><td>Error locus</td><td>reading; comprehension; transformation; process skills; encoding</td><td>0.68</td><td>0.62</td></tr><tr><td>Error mechanism language difficulty;</td><td>spatial information difficulty; deficient prerequisites; incorrect associations; irrelevant rules</td><td>0.66</td><td>0.60</td></tr></table>

Table 5: Judge labels and agreement. Human is Fleiss’ κ among three annotators, and Judge is Cohen’s κ between GPT-5.4 and their majority label.

We score three aspects of the simulated turns. Functional agreement uses Cohen’s $\kappa _ { \mathrm { f u n c } }$ and the multiclass $\mathrm { M C C } _ { \mathrm { f u n c } }$ over $f _ { n }$ . Error-state prediction uses the balanced accuracy $\mathrm { B A c c } _ { \mathrm { e r r } } , \mathrm { M C C } _ { \mathrm { e r r } }$ , and $\mathrm { F 1 } _ { \mathrm { e r r } }$ over $e _ { n }$ . Error-structure fidelity counts a real error as reproduced only when the simulated turn also errs and all three structure labels match. With the real error set $\mathcal { E } = \{ n : e _ { n } = 1 \}$

$$
\mathrm { R e c } _ { \mathrm { s t r u c t } } = \frac { 1 } { | \mathcal { E } | } \sum _ { n \in \mathcal { E } } \mathbf { 1 } [ \hat { e } _ { n } = 1 \wedge \hat { \ell } _ { n } = \ell _ { n } ]\tag{3}
$$

is the share of real errors that the simulator reproduces with matching structure. $\mathrm { P r e c } _ { \mathrm { s t r u c t } }$ is the corresponding share of simulated errors, and $\mathrm { F 1 _ { s t r u c t } }$ is their harmonic mean. Appendix D gives all definitions.

Baselines. We compare HACHIMI-II with taskadapted versions of four simulation strategies. Direct Role-Play (Direct RP) asks the model to respond as a student given only the current item or the dialogue so far (Scarlatos et al., 2026). Prompt-Constrained Simulation (Prompt-C) adds naturallanguage constraints on accessible knowledge, reasoning depth, uncertainty, and error preservation. It is a prompt-level instantiation of the epistemicstate specification of Yuan et al. (2026). Profile-Conditioned Simulation (Profile-C) conditions on the learner’s history and a learner profile (Lu and Wang, 2024; Liu et al., 2024; Duan et al., 2026; Jiang et al., 2026), which in item-response simulation is the student model of HACHIMI-II. Agent-Style Memory Simulation (Agent-Mem), used in dialogue only, adds a learner profile and memory of the visible dialogue (Xu et al., 2024; Gao et al., 2025). All methods use the same backbone, Qwen3.5-397B, with temperature 0.2 (Appendix E).

Ablations. w/o Prior builds the student model from visible evidence only, without a HACHIMI-1M persona. w/o State removes learner state inference. The generator receives only the student model and the task context, and the harness keeps only its output-format constraints. w/o Harness gives the generator the student model and the inferred state without harness compilation.

## 5 Results

## 5.1 Item-Response Simulation

Table 4 shows that HACHIMI-II attains the highest agreement, the smallest $W _ { 1 } .$ , and the highest $\rho$ on all three datasets. It gives the same outcome as the real learner on 75%, 61%, and 63% of the student– item pairs, against at most 53%, 52%, and 48% for the baselines. Its $W _ { 1 }$ is $2 . 4 \substack { - 8 . 4 \times }$ smaller than that of the strongest baseline on each dataset.

The baselines exhibit competence leakage at the population level. Each baseline simulates one generic student and assigns nearly the same accuracy to every learner, at a level set by the model and the prompt. This level sits near 0.50 on XES3G5M, where the real mean is 0.83, and near 0.82 on ENEM, where the real mean is 0.39. Direct RP sees no learner history, and its $\rho$ stays between −0.06 and 0.01. With the student model, Profile-C recovers the ordering of students on Eedi $( \rho = 0 . 6 9 8 )$ yet its simulated accuracies still cluster around one value $( W _ { 1 } = 0 . 1 1 9 )$ .

![](images/c9bff05d1a762dac387d4cded005b74cc59cc3017c33433345e68c36298b94d6.jpg)  
Figure 3: Simulated outputs on turns where the real learner errs.

HACHIMI-II tracks both the level and the spread of real ability. Its simulated mean accuracy is 0.828, 0.593, and 0.384 on the three datasets against real means of 0.831, 0.574, and 0.394 (Appendix G). In Figure 2, the Profile-C points stay within a narrow band, while the HACHIMI-II points follow the diagonal, and ρ of 0.67–0.70 shows that the method also recovers the ordering of individual learners.

Table 6 makes the contrast concrete by splitting the students of each dataset into thirds by real accuracy. The baselines give the three groups almost the same simulated accuracy. On ENEM, the weakest third answers 22% of the items correctly, and the baselines simulate these students at 78–83%. HACHIMI-II follows the real level of every group. It simulates the three ENEM groups at 15%, 39%, and 68% against real values of 22%, 39%, and 61%. The strongest and the weakest third differ by at most 0.11 for any baseline, against 0.19–0.47 for the real students and 0.18–0.53 for HACHIMI-II.

## 5.2 Dialogue Simulation

Table 7 reports the dialogue results. Functional agreement is dominated by the frequent turns that give a correct answer or contain no mathematics, so the methods separate most clearly on the turns where the learner errs. Figure 3 shows what each simulator produces on these turns. The baselines answer 82–93% of them without any error, which is competence leakage at the level of a single turn, while HACHIMI-II errs on 52% of them on Eedi QATD2k and 76% on Khan/CoMTA.

<table><tr><td rowspan="2"></td><td colspan="3">XES3G5M</td><td colspan="3">Eedi</td><td colspan="3">ENEM</td></tr><tr><td>Low</td><td>Middle</td><td> $\mathrm { H i g h }$ </td><td>Low</td><td>Middle</td><td> $\mathrm { H i g h }$ </td><td>Low</td><td>Middle</td><td>High</td></tr><tr><td>Real students</td><td>0.74</td><td>0.85</td><td>0.93</td><td>0.34</td><td>0.58</td><td>0.81</td><td>0.22</td><td>0.39</td><td>0.61</td></tr><tr><td>Direct RP</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.59</td><td>0.59</td><td>0.59</td><td>0.83</td><td>0.83</td><td>0.83</td></tr><tr><td>Prompt-C</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.58</td><td>0.59</td><td>0.60</td><td>0.81</td><td>0.82</td><td>0.83</td></tr><tr><td>Profile-C</td><td>0.49</td><td>0.49</td><td>0.49</td><td>0.47</td><td>0.54</td><td>0.58</td><td>0.78</td><td>0.80</td><td>0.82</td></tr><tr><td>HACHIMI-II</td><td>0.74</td><td>0.84</td><td>0.92</td><td>0.42</td><td>0.61</td><td>0.75</td><td>0.15</td><td>0.39</td><td>0.68</td></tr></table>

Table 6: Mean per-student accuracy by third of real accuracy.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Functional agreement</td><td colspan="3">Error-state prediction</td><td colspan="2">Error-structure fidelity</td></tr><tr><td> $\kappa _ { \mathrm { f u n c } }$  ←</td><td> $\mathrm { M C C } _ { \mathrm { f u n c } } \uparrow$ </td><td> $\operatorname { B A c c } _ { \mathrm { e r r } } \uparrow$ </td><td> $\mathrm { M C C } _ { \mathrm { e r r } }$ </td><td> $\mathrm { F } 1 _ { \mathrm { e r r } } \sim$ </td><td> $\mathrm { R e c _ { s t r u c t } }$  ↑</td><td> $\mathrm { F 1 _ { s t r u c t } \uparrow }$ </td></tr><tr><td rowspan="5">Eedi QATD2k</td><td>Direct RP</td><td>0.592</td><td>0.616</td><td>0.560</td><td>0.175</td><td>0.236</td><td>0.157</td><td>0.220</td></tr><tr><td>Prompt-C</td><td>0.586</td><td>0.614</td><td>0.558</td><td>0.174</td><td>0.230</td><td>0.150</td><td>0.214</td></tr><tr><td>Profile-C</td><td>0.595</td><td>0.615</td><td>0.565</td><td>0.179</td><td>0.248</td><td>0.172</td><td>0.232</td></tr><tr><td>Agent-Mem</td><td>0.597</td><td>0.618</td><td>0.562</td><td>0.177</td><td>0.240</td><td>0.162</td><td>0.225</td></tr><tr><td>HACHIMI-II</td><td>0.659</td><td>0.673</td><td>0.738</td><td>0.509</td><td>0.561</td><td>0.315</td><td>0.339</td></tr><tr><td rowspan="5">Khan/CoMTA</td><td>Direct RP</td><td>0.273</td><td>0.337</td><td>0.515</td><td>0.052</td><td>0.134</td><td>0.080</td><td>0.127</td></tr><tr><td>Prompt-C</td><td>0.274</td><td>0.345</td><td>0.512</td><td>0.045</td><td>0.117</td><td>0.067</td><td>0.110</td></tr><tr><td>Profile-C</td><td>0.289</td><td>0.347</td><td>0.530</td><td>0.098</td><td>0.183</td><td>0.110</td><td>0.169</td></tr><tr><td>Agent-Mem</td><td>0.288</td><td>0.354</td><td>0.531</td><td>0.110</td><td>0.175</td><td>0.105</td><td>0.168</td></tr><tr><td>HACHIMI-II</td><td>0.370</td><td>0.390</td><td>0.660</td><td>0.267</td><td>0.464</td><td>0.565</td><td>0.346</td></tr></table>

Table 7: Dialogue results. Best per dataset in bold.

HACHIMI-II predicts when the learner errs and reproduces the error. Its $\mathrm { F 1 } _ { \mathrm { e r r } }$ of 0.561 on Eedi QATD2k and 0.464 on Khan/CoMTA is 2.3× and 2.5× that of the strongest baseline. It reproduces 31.5% and 56.5% of real errors with all three structure labels matching, which is 1.8× and 5.1× the rate of the strongest baseline, and its $\mathrm { F 1 _ { s t r u c t } }$ is 1.5× and 2.0× that of the strongest baseline. HACHIMI-II also attains the highest $\kappa _ { \mathrm { f u n c } }$ and ${ \mathrm { M C C } } _ { \mathrm { f u n c } }$ on both datasets, so its gains on error turns come with higher functional agreement than every baseline.

Table 9 shows one Eedi QATD2k turn. The real learner fails to distribute the 3 when expanding $3 ( n - 4 )$ , and all four baselines return the correct expansion. HACHIMI-II predicts this error in the learner state and writes 3n − 4, which the judge labels with the same knowledge type, locus, and mechanism as the real turn.

## 5.3 What Each Component Contributes

Table 8 removes one component at a time. The three components control different parts of the simulated behavior.

The learner state counters competence leakage. Without the state, the simulated population moves toward the model’s own competence. On ENEM the simulated mean accuracy rises from 0.384 to 0.802 against a real 0.394, and $W _ { 1 }$ grows 1.9–2.5× across the three item-response datasets, returning to the level of Profile-C on Eedi and ENEM. In dialogue, $\mathrm { F 1 _ { s t r u c t } }$ falls from 0.339 to 0.136 on Eedi QATD2k and from 0.346 to 0.119 on Khan/- CoMTA. This variant still builds the full student model, including the persona, and still generates through separate model calls, so these gains come from the explicit prediction of the learner state.

The same variant shows that functional agreement can be high under leakage. Its $\kappa _ { \mathrm { f u n c } }$ is 0.741 on Eedi QATD2k because it matches the frequent correct and non-mathematical turns, while it reproduces 9.9% of real errors with matching structure.

The harness keeps the action consistent with the learner. Removing the harness increases $W _ { 1 }$ from 0.039 to 0.056 on XES3G5M and from 0.169 to 0.257 on ENEM, and lowers $\kappa \mathrm { f u n c }$ on both dialogue datasets. On Eedi QATD2k the errors generated without the harness match the learner’s error less often, and $\mathrm { F 1 _ { s t r u c t } }$ drops from 0.339 to 0.290. The harness places errors at the predicted step and keeps the rest of the turn in line with the learner.

<table><tr><td></td><td colspan="3"> $W _ { 1 } \downarrow$ </td><td colspan="3"> $\rho \uparrow$ </td><td colspan="2"> $\kappa _ { \mathrm { f u n c } }$  ←</td><td colspan="2"> $\mathrm { F 1 _ { s t r u c t } \uparrow }$ </td></tr><tr><td>Variant</td><td>XES</td><td>Eedi</td><td>ENEM</td><td>XES</td><td>Eedi</td><td>ENEM</td><td>QATD2k</td><td>CoMTA</td><td>QATD2k</td><td>CoMTA</td></tr><tr><td>HACHIMI-II</td><td>0.039</td><td>0.046</td><td>0.169</td><td>0.691</td><td>0.703</td><td>0.668</td><td>0.659</td><td>0.370</td><td>0.339</td><td>0.346</td></tr><tr><td>w/o Prior</td><td>0.025</td><td>0.041</td><td>0.142</td><td>0.603</td><td>0.704</td><td>0.594</td><td>0.571</td><td>0.326</td><td>0.264</td><td>0.323</td></tr><tr><td>w/o State</td><td>0.074</td><td>0.116</td><td>0.409</td><td>0.664</td><td>0.742</td><td>0.539</td><td>0.741</td><td>0.382</td><td>0.136</td><td>0.119</td></tr><tr><td>w/o Harness</td><td>0.056</td><td>0.048</td><td>0.257</td><td>0.559</td><td>0.703</td><td>0.643</td><td>0.565</td><td>0.348</td><td>0.290</td><td>0.346</td></tr></table>

Table 8: Ablation results. Appendix G reports the remaining metrics.

<table><tr><td colspan="2">Question. A function machine subtracts 4 and then multi- plies by 3. Leo writes  $3 n - 1 2$  and Sophie writes  $3 ( n - 4 )$  Who is correct?</td></tr><tr><td>Tutor</td><td>What do you get if you multiply out the brackets?</td></tr><tr><td>Real learner</td><td>3n-4 (incorrect; procedural, process skills, deficient prerequisites)</td></tr><tr><td>Baselines HACHIMI- Ⅱ</td><td>3n — 12 (all four; correct) State. Weak self-monitoring, tentative, fragment only. Predicted error. 3n — 4 in</td></tr></table>

Table 9: A real Eedi QATD2k turn (§5.2).

The persona prior improves the ordering of learners. Removing the HACHIMI-1M prior lowers $\rho$ on XES3G5M from 0.691 to 0.603 and on ENEM from 0.668 to 0.594. On Eedi QATD2k it lowers $\kappa \mathrm { f u n c }$ from 0.659 to 0.571 and $\mathrm { F 1 _ { s t r u c t } }$ from 0.339 to 0.264, and both also fall on Khan/- CoMTA. $W _ { 1 }$ alone is lower without the prior, by 0.005 to 0.027. The persona sharpens the differences between weaker and stronger learners, which improves their ordering and, on XES3G5M and ENEM, also widens the spread of simulated accuracy (Appendix G). $W _ { 1 }$ compares the two distributions as wholes, so it counts the wider spread and does not register the better ordering. On ENEM the prior also corrects the simulated mean, to 0.384 from 0.500 against a real 0.394.

## 6 Discussion

Our results show that an explicit model of the learner’s behavior counters competence leakage. Under role prompting, each simulator portrays one generic student whose accuracy is set by the model and the prompt. This uniformity is the abilitylevel counterpart of the flattening and caricature observed when language models simulate opinions, personas, and identity groups (Santurkar et al., 2023; Cheng et al., 2023; Wang et al., 2025). As in social simulation, where interactions scripted by one model look more successful than those among agents that hold only their own information (Zhou et al., 2024), the simulator draws on knowledge that the simulated party does not have. Prediction changes the role of the model’s knowledge. To predict that a learner will average two speeds, the model must know how average speed is computed and where this learner departs from it, so the competence that leaks under role-play becomes the source of faithful errors. Evaluations of simulated students should compare the simulated ability distribution with the real one and score the error state and its structure together with functional agreement, which is dominated by turns without an error. A simulator that predicts how a particular learner responds to an item or a tutor move supports uses that depend on ability differences or on specific misconceptions, such as comparing teaching strategies across ability levels or testing a hint against a misconception. It is also a step toward a world model of the learner (Ha and Schmidhuber, 2018; Zhou et al., 2025), against which tutoring policies can be planned and compared before they reach real students.

## 7 Conclusion

Role-prompted student simulators exhibit competence leakage and portray a generic student whose ability is set by the model. HACHIMI-II models the learner’s behavior through third-person prediction, inferring the learner’s state for the current item or tutor turn and compiling it into a behavior harness for the next action. Compared with every baseline, it brings the simulated ability distribution closer to the real one on three itemresponse datasets and reproduces real errors with their full structure more often on two dialogue datasets. Faithful student simulation thus rests on predicting the learner, which lets tutor development and evaluation use capable models as simulated students.

## Limitations

Our evaluation covers the correctness of item responses and the presence and structure of mathematical errors in tutoring turns, on datasets in mathematics and natural sciences. A learner’s next action also depends on factors that no record captures, such as momentary attention and the outcome of a guess, and HACHIMI-II predicts the part of behavior that the available evidence determines. Each dialogue turn is predicted from the real preceding dialogue. A direct extension is multi-turn simulation in which the simulated student’s own turns form the history, evaluated with a protocol that compares generated and real dialogues over many turns. The state and harness are defined through knowledge, reasoning, errors, confidence, and interaction, and the method uses the backbone through prompts and deterministic rules, so it can be applied to further subjects and backbones without training.

## Ethical Considerations

All datasets in this work are publicly available, and we use them under their original terms of use. The learner records in these datasets are de-identified, and HACHIMI-II uses only their responses, dialogue turns, and grade or age. A matched persona comes from HACHIMI-1M, a corpus of generated student personas (Jiang et al., 2026), and is used only as a prior on learning behavior inside the simulator. The simulated students are intended to supplement data from real learners when educational systems are developed and evaluated. Each output is a predicted response to one item or tutor turn and should not be used to assess or classify individual students.

## References

Vincent Aleven, Ido Roll, Bruce M McLaren, and Kenneth R Koedinger. 2016. Help helps, but only so much: Research on help seeking with intelligent tutoring systems. International Journal of Artificial Intelligence in Education, 26(1):205–223.

Lorin W. Anderson and David R. Krathwohl, editors. 2001. A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom’s Taxonomy of Educational Objectives. Longman, New York.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Luca Benedetto, Giovanni Aradelli, Antonia Donvito, Alberto Lucchetti, Andrea Cappelli, and Paula Buttery. 2024. Using LLMs to simulate students’ responses to exam questions. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pages 11351–11368.

Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. 2023. Prompting is programming: A query language for large language models. Proceedings of the ACM on Programming Languages, 7(PLDI):1946–1969.

John Seely Brown and Richard R. Burton. 1978. Diagnostic models for procedural bugs in basic mathematical skills. Cognitive Science, 2(2):155–192.

Myra Cheng, Tiziano Piccardi, and Diyi Yang. 2023. CoMPosT: Characterizing and evaluating caricature in LLM simulations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10853–10875. Association for Computational Linguistics.

Michelene T. H. Chi and Ruth Wylie. 2014. The ICAP framework: Linking cognitive engagement to active learning outcomes. Educational Psychologist, 49(4):219–243.

Youngduck Choi, Youngnam Lee, Dongmin Shin, Junghyun Cho, Seoyon Park, Seewoo Lee, Jineon Baek, Chan Bae, Byungsoo Kim, and Jaewe Heo. 2020. EdNet: A large-scale hierarchical dataset in education. In Artificial Intelligence in Education (AIED 2020), pages 69–73. Springer.

Albert T Corbett and John R Anderson. 1994. Knowledge tracing: Modeling the acquisition of procedural knowledge. User Modeling and User-Adapted Interaction, 4(4):253–278.

Zhangqi Duan, Shuyan Huang, Alexander Scarlatos, Jaewook Lee, Simon Woodhead, and Andrew Lan. 2026. Who am I? History-aware profiles for student simulation in tutoring dialogues. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing. ArXiv:2605.30051.

Weibo Gao, Qi Liu, Linan Yue, Fangzhou Yao, Rui Lv, Zheng Zhang, Hao Wang, and Zhenya Huang. 2025. Agent4Edu: Generating learner response data by generative agents for intelligent education systems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 23923–23932.

Cristina Gârbacea and Qiaozhu Mei. 2025. Why is constrained neural language generation particularly challenging? Transactions on Machine Learning Research.

David Ha and Jürgen Schmidhuber. 2018. Recurrent world models facilitate policy evolution. Advances in Neural Information Processing Systems, 31:2450– 2462.

INEP. 2024. Enem 2024: Provas e gabaritos. https: //www.gov.br/inep/pt-br/areas-de-atuacao/ avaliacao-e-exames-educacionais/enem/ provas-e-gabaritos/2024. Accessed: 2026-07.

Yilin Jiang, Fei Tan, Xuanyu Yin, Jing Leng, and Aimin Zhou. 2026. HACHIMI: Scalable and controllable student persona generation via orchestrated agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 21461–21506. Association for Computational Linguistics.

Tanja Käser and Giora Alexandron. 2024. Simulated learners in educational technology: A systematic literature review and a Turing-like test. International Journal ofArtificial Intelligence in Education, 34(2):545–585.

Junjie Li, Xi Xiao, Yunbei Zhang, Chen Liu, Lin Zhao, Xiaoying Liao, Yingrui Ji, Janet Wang, et al. 2026. Agent harness engineering: A survey. OpenReview preprint.

Zhengyuan Liu, Stella Xin Yin, Geyu Lin, and Nancy F. Chen. 2024. Personality-aware student simulation for conversational intelligent tutoring systems. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 626– 642. Association for Computational Linguistics.

Zitao Liu, Qiongqiong Liu, Teng Guo, Jiahao Chen, Shuyan Huang, Xiangyu Zhao, Jiliang Tang, Weiqi Luo, and Jian Weng. 2023. XES3G5M: A knowledge tracing benchmark dataset with auxiliary information. Advances in Neural Information Processing Systems, 36:32958–32970.

Frederic M. Lord. 1980. Applications of Item Response Theory to Practical Testing Problems. Lawrence Erlbaum Associates, Hillsdale, NJ.

Xinyi Lu and Xu Wang. 2024. Generative students: Using LLM-simulated student profiles to support question item evaluation. In Proceedings ofthe Eleventh ACM Conference on Learning @ Scale, pages 16–27.

Luis Marquez-Carpintero, Alberto Lopez-Sellers, and Miguel Cazorla. 2026. Simulating students with large language models: A review of architecture, mechanisms, and role modeling in education with generative AI. Computer Science Review, 62:101008.

Daria Martynova, Jakub Macina, Nico Daheim, Nilay Yalcin, Xiaoyu Zhang, and Mrinmaya Sachan. 2025. Can LLMs effectively simulate human learners? Teachers’ insights from tutoring LLM students. In Proceedings ofthe 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025), pages 100–117. Association for Computational Linguistics.

Xianghui Meng and Jionghao Lin. 2026. Simulating disengaged students to evaluate LLM-based tutors. arXiv preprint arXiv:2609.12331.

Pepper Miller and Kristen DiCerbo. 2024. LLM based math tutoring: Challenges and dataset. Khan Academy technical report. Dataset: https://github.com/Khan/ tutoring-accuracy-dataset.

Robert J Mislevy, Russell G Almond, and Janice F Lukas. 2003. A brief introduction to evidencecentered design. ETS Research Report Series, 2003(1):i–29.

Robert J. Mislevy and Michelle M. Riconscente. 2005. Evidence-centered assessment design: Layers, structures, and terminology. PADI Technical Report 9, SRI International, Menlo Park, CA.

M. Anne Newman. 1977. An analysis of sixth-grade pupils’ errors on written mathematical tasks. Victorian Institute for Educational Research Bulletin, 39:31–43.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Linyue Pan, Lexiao Zou, Shuo Guo, Jingchen Ni, and Hai-Tao Zheng. 2026. Natural-language agent harnesses. arXiv preprint arXiv:2603.25723.

Ernesto Panadero. 2017. A review of self-regulated learning: Six models and four directions for research. Frontiers in Psychology, 8:422.

Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas Guibas, and Jascha Sohl-Dickstein. 2015. Deep knowledge tracing. Advances in Neural Information Processing Systems, 28.

Changyong Qi, Longwei Zheng, Anna He, Haoxin Xu, Linzhao Jia, Yuang Wei, Bingqian Jiang, and Xiaoqing Gu. 2026. Simulating student learning behaviors with LLM-based role-playing agents: A datadriven and cognitively inspired framework. Expert Systems with Applications, 304:130753.

Hendrik Radatz. 1979. Error analysis in mathematics education. Journalfor Research in Mathematics Education, 10(3):163–172.

Siddharth Reddy, Sergey Levine, and Anca Dragan. 2017. Accelerating human learning with deep reinforcement learning. In NIPS Workshop on Teaching Machines, Robots, and Humans.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In Proceedings of the 40th International Conference

on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 29971–30004. PMLR.

Alexander Scarlatos, Jaewook Lee, Simon Woodhead, and Andrew Lan. 2026. Simulated students in tutoring dialogues: Substance or illusion? In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 42349–42385. Association for Computational Linguistics.

Jiajia Song, Zhihan Guo, and Jionghao Lin. 2026. Simulating novice students using machine unlearning and relearning in large language models. arXiv preprint arXiv:2603.26142.

KV Aditya Srivatsa, Kaushal Kumar Maurya, and Ekaterina Kochmar. 2025. Can LLMs reliably simulate real students’ abilities in mathematics and reading comprehension? In Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025), pages 988–1001. Association for Computational Linguistics.

Cédric Villani. 2009. Optimal Transport: Old and New, volume 338 of Grundlehren der mathematischen Wissenschaften. Springer.

Lev S. Vygotsky. 1978. Mind in Society: The Development of Higher Psychological Processes. Harvard University Press, Cambridge, MA. Edited by Michael Cole, Vera John-Steiner, Sylvia Scribner, and Ellen Souberman.

Angelina Wang, Jamie Morgenstern, and John P. Dickerson. 2025. Large language models that replace human participants can harmfully misportray and flatten identity groups. Nature Machine Intelligence, 7(3):400–411.

Zichao Wang, Angus Lamb, Evgeny Saveliev, Pashmina Cameron, Yordan Zaykov, José Miguel Hernández-Lobato, Richard E. Turner, Richard G. Baraniuk, Craig Barton, Simon Peyton Jones, Simon Woodhead, and Cheng Zhang. 2021. Results and insights from diagnostic questions: The NeurIPS 2020 education challenge. In Proceedings of the NeurIPS 2020 Competition and Demonstration Track, volume 133 of Proceedings ofMachine Learning Research, pages 191–205. PMLR.

Allan Leslie White. 2010. Numeracy, literacy and Newman’s error analysis. Journal ofScience and Mathematics Education in Southeast Asia, 33(2):129–148.

David Wood, Jerome S. Bruner, and Gail Ross. 1976. The role of tutoring in problem solving. Journal of Child Psychology and Psychiatry, 17(2):89–100.

Tao Wu, Jingyuan Chen, Wang Lin, Mengze Li, Yumeng Zhu, Ang Li, Kun Kuang, and Fei Wu. 2025. Embracing imperfection: Simulating students with diverse cognitive levels using LLM-based agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9887–9908.

Songlin Xu and Xinyu Zhang. 2026. Unveiling the capabilities of large language models in simulating student behavioral dynamics and supporting peer feedback to augment task performance. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems, pages 1–26.

Songlin Xu, Xinyu Zhang, and Lianhui Qin. 2024. Edu-Agent: Generative student agents in learning. arXiv preprint arXiv:2404.07963.

Ke Yang, Chenglong Wang, Michel Galley, Chandan Singh, Jeevana Priya Inala, ChengXiang Zhai, and Jianfeng Gao. 2026. StudentSim: Training LLM-based student simulators. arXiv preprint arXiv:2609.01591.

Zhihao Yuan, Yunze Xiao, Ming Li, Weihao Xuan, Richard Tong, Mona Diab, and Tom Mitchell. 2026. Towards valid student simulation with large language models. arXiv preprint arXiv:2601.05473.

Matthew Zent, Digory Smith, and Simon Woodhead. 2025. PIIvot: A lightweight NLP anonymization framework for question-anchored tutoring dialogues. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 27479–27488, Suzhou, China. Association for Computational Linguistics.

Jiani Zhang, Xingjian Shi, Irwin King, and Dit-Yan Yeung. 2017. Dynamic key-value memory networks for knowledge tracing. In Proceedings of the 26th International Conference on World Wide Web, pages 765–774.

Guanhao Zhao, Zhenya Huang, Yan Zhuang, Jiayu Liu, Qi Liu, Zhiding Liu, Jinze Wu, and Enhong Chen. 2023. Simulating student interactions with two-stage imitation learning for intelligent educational systems. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 3423–3432.

Hailin Zhong and Shengxin Zhu. 2026. AI harness engineering: A runtime substrate for foundation-model software agents. arXiv preprint arXiv:2605.13357.

Xuhui Zhou, Jiarui Liu, Akhila Yerukola, Hyunwoo Kim, and Maarten Sap. 2025. Social world models. arXiv preprint arXiv:2509.00559.

Xuhui Zhou, Zhe Su, Tiwalayo Eisape, Hyunwoo Kim, and Maarten Sap. 2024. Is this the real life? Is this just fantasy? The misleading success of simulating social interactions with LLMs. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 21692–21714. Association for Computational Linguistics.

Barry J Zimmerman. 2002. Becoming a self-regulated learner: An overview. Theory Into Practice, 41(2):64–70.

## A Persona Matching

HACHIMI-II retrieves one HACHIMI-1M persona for each learner in six steps.

1. Observed profile. The learner’s visible history is summarized into a profile with grade or age, ability level, past performance, strengths and weaknesses, response style, uncertainty, and help-seeking.

2. Candidate retrieval. A candidate pool is drawn from HACHIMI-1M on coarse fields, namely grade or age range, academic level, subject strengths and weaknesses, and learning-style tags.

3. Rule-based ranking. Candidates are scored on grade compatibility, agreement of ability level, overlap of strong and weak subjects, and similarity of learning behavior.

4. Reranking. The backbone compares the observed profile with the full descriptions of the top-k candidates (k = 8), using motivation, confidence, learning habits, self-efficacy, and interaction style.

5. Validity check. Near-duplicate candidates are removed, and the selected persona is checked for membership in the candidate pool and completeness of its fields.

6. Fusion. The selected persona is merged with the observed profile into the third-person student model.

## B State and Harness Fields

Table 10 lists the fields used in each format. Both formats follow the design in Table 1, and the fields differ because the evidence and the target action differ. Prerequisite familiarity, reachability, and the uptake of the tutor’s support refine the reachable knowledge. The likely error path in item-response simulation and the misconception status and predicted error in dialogue form the active error pattern. Commitment and self-monitoring form the confidence state, and help-seeking, readiness to attempt, the form of engagement, and response depth form the local interaction move. In item-response simulation the state describes the learner’s relation to the current item, and the harness constrains the answer and its explanation. In dialogue the state describes the learner’s response to the current tutor turn, and the harness constrains what the next turn may say and do.

## C Generation and Checking

In dialogue simulation, each generated turn is rewritten in the learner’s register, and a consistency check confirms that its content is unchanged. A rule-based check and a review by the backbone then compare the turn with the harness. They look for a response more complete than the state allows, a transition faster than the predicted state allows, and a break in the predicted error path. A turn that violates the harness is regenerated once under tightened constraints. In item-response simulation, the output contains a final answer and a short student response, and it is checked and regenerated in the same way.

## D Evaluation Details

Item-response metrics. Item-response evaluation extracts the final answer from the simulated response, normalizes whitespace, letter case, and option markers, and compares the result with the answer key to obtain ${ \hat { y } } _ { n }$ . A $W _ { 1 }$ of 0.10 corresponds to an average gap of ten accuracy points between real and simulated students of the same rank.

Judge labels. In dialogue, the judge labels the real and the simulated turn with the five dimensions in Table 5. All metrics are computed from these labels.

Judge selection. We chose the judge by agreement with human annotation. For each of the five dimensions, three annotators labeled 200 turns, and the turns for the three structure dimensions all contain an error. The annotators are graduate students in AI and NLP who were recruited for this task and paid at or above the local standard rate, and they consented to the use of their labels in this study. The majority label is the gold standard. Table 11 reports the agreement among annotators, and Table 12 reports the agreement of five candidate judges with the gold labels. GPT-5.4 attains the highest mean agreement and the highest agreement on the functional label, the error indicator, and the knowledge type, and we use it for all dialogue evaluation. The judge and the simulation backbone come from different model families.

Functional agreement. Let $C \in \mathbb { N } ^ { 3 \times 3 }$ be the confusion matrix of real and simulated functional labels, with row sums $t _ { i } ,$ column sums $p _ { j }$ , and total N. Cohen’s $\kappa = ( p _ { o } - p _ { e } ) / ( 1 - p _ { e } )$ uses the observed agreement $\begin{array} { r } { p _ { o } = \frac { 1 } { N } \sum _ { i } C _ { i i } } \end{array}$ and the chance agreement $\begin{array} { r } { p _ { e } = \frac { 1 } { N ^ { 2 } } \sum _ { k } t _ { k } p _ { k } } \end{array}$ . The multiclass Matthews correlation coefficient is

<table><tr><td>Component</td><td>Item-response simulation</td><td>Dialogue simulation</td></tr><tr><td>Input evidence</td><td>Learner ability, prior responses, guessing tendency, confidence calibration; item prerequisites, operation complexity, response format, guessability</td><td>Visible dialogue history, previous student turn, current tutor turn; type, strength, and target of the tutor&#x27;s support</td></tr><tr><td>Learner state</td><td>Prerequisite familiarity, whether the item is within reach, likely error path, commitment, depth of explanation</td><td>Whether the step is within reach, misconception status, self-monitoring, uptake of the tutor&#x27;s support, help-seeking, readiness to attempt, form of engagement, commitment, depth of response, predicted error (presence, locus, mechanism, faulty rule)</td></tr><tr><td>Behavior harness</td><td>Whether a full explanation is allowed, response-length limit, restriction on fully correct reasoning, error path to follow, answer commitment</td><td>Whether the answer may be stated directly, whether full reasoning is allowed, persistence of confusion and of the predicted misconception, whether help-seeking is allowed, number of solution steps</td></tr></table>

Table 10: State and harness fields in each task format.

<table><tr><td>Dimension</td><td>Fleiss&#x27;κ</td><td>Agreement (%)</td></tr><tr><td>Functional label</td><td>0.84</td><td>91.0</td></tr><tr><td>Error indicator</td><td>0.81</td><td>90.5</td></tr><tr><td>Knowledge type</td><td>0.73</td><td>82.5</td></tr><tr><td>Error locus</td><td>0.68</td><td>78.5</td></tr><tr><td>Error mechanism</td><td>0.66</td><td>77.0</td></tr></table>

Table 11: Agreement among the three annotators. Agreement (%) is exact three-way agreement.
<table><tr><td>Judge</td><td>Func.</td><td>Err.</td><td>Type</td><td>Locus</td><td>Mech.</td><td>Mean</td></tr><tr><td>Gemini 3.5</td><td>0.79</td><td>0.74</td><td>0.64</td><td>0.58</td><td>0.55</td><td>0.660</td></tr><tr><td>Kimi 2.6</td><td>0.76</td><td>0.72</td><td>0.62</td><td>0.56</td><td>0.54</td><td>0.640</td></tr><tr><td>GLM 5.1</td><td>0.78</td><td>0.73</td><td>0.63</td><td>0.57</td><td>0.55</td><td>0.652</td></tr><tr><td>DeepSeek V4</td><td>0.80</td><td>0.77</td><td>0.68</td><td>0.64</td><td>0.62</td><td>0.702</td></tr><tr><td>GPT-5.4</td><td>0.83</td><td>0.80</td><td>0.69</td><td>0.62</td><td>0.60</td><td>0.708</td></tr></table>

Table 12: Cohen’s κ between each candidate judge and the majority human labels.

$$
\mathrm { M C C } = \frac { N \sum _ { k } C _ { k k } - \sum _ { k } t _ { k } p _ { k } } { \sqrt { \left( N ^ { 2 } - \sum _ { k } p _ { k } ^ { 2 } \right) \left( N ^ { 2 } - \sum _ { k } t _ { k } ^ { 2 } \right) } } .\tag{4}
$$

Both equal 1 for perfect agreement and approach 0 for a simulator whose labels are independent of the real ones.

Error-state prediction. We treat the error indicator as a binary target. With the real error set E and the simulated error set $\hat { \mathcal { E } } = \{ n : \hat { e } _ { n } = 1 \}$ , the four counts are the true positives $\mathrm { T P } = | \mathcal { E } \cap \hat { \mathcal { E } } |$ the false positives $\mathrm { F P } = | \hat { \mathcal { E } } \setminus \mathcal { E } |$ , the false negatives $\mathrm { F N } = | \mathcal { E } \setminus \hat { \mathcal { E } } |$ , and the true negatives TN. Error recall is $\mathrm { T P } / ( \mathrm { T P } + \mathrm { F N } )$ , error precision is $\mathrm { T P } / ( \mathrm { T P } + \mathrm { F P } )$ , and $\mathrm { F 1 } _ { \mathrm { e r r } }$ is their harmonic mean. $\mathrm { B A c c } _ { \mathrm { e r r } }$ is the mean of error recall and the specificity $\mathrm { T N } / ( \mathrm { T N } + \mathrm { F P } )$ $\mathrm { M C C _ { \mathrm { e r r } } }$ and $\kappa _ { \mathrm { e r r } }$ are computed from the same four counts, and both equal 0 for a simulator that always errs or never errs.

Error-structure fidelity. $\mathrm { R e c _ { s t r u c t } }$ is defined in §4, and

$$
\mathrm { P r e c } _ { \mathrm { s t r u c t } } = \frac { 1 } { | \hat { \mathcal { E } } | } \sum _ { n \in \hat { \mathcal { E } } } \mathbf { 1 } [ e _ { n } = 1 \wedge \hat { \ell } _ { n } = \ell _ { n } ]\tag{5}
$$

is the share of simulated errors that match a real error in structure. A pair of turns counts as a structural match only when both contain an error and the knowledge type, locus, and mechanism all agree. Both metrics are bounded by their error-state counterparts, $\mathrm { R e c } _ { \mathrm { s t r u c t } } \leq$ error recall and $\mathrm { P r e c } _ { \mathrm { s t r u c t } } \leq$ error precision.

## E Implementation Details

All methods follow the protocol of §3.1. In itemresponse simulation each method receives the current question and its answer options, and every method except Direct RP also receives the learner’s history. The history contains 30 of the student’s responses on XES3G5M and Eedi and 20 on ENEM, all to items outside the test set, and every student of a dataset answers the same test items. In dialogue each method receives the dialogue before the target turn. Appendix H gives the prompts of HACHIMI-II.

Direct Role-Play. The model acts as a student and generates the action from the current item or the dialogue so far.

<table><tr><td>Dataset</td><td>Method</td><td colspan="3">KS↓ Stu. r ↑ Item ρ↑ Item r ↑</td></tr><tr><td>XES3G5M</td><td>Direct RP Prompt-C Profile-C HACHIMI-II 0.217</td><td>1.000 -0.005 1.000 -0.040 1.000 0.215 0.629</td><td>0.364 0.366 0.348 0.329</td><td>0.260 0.295 0.264 0.315</td></tr><tr><td>Eedi</td><td>Direct RP Prompt-C Profile-C HACHIMI-II 0.115</td><td>0.448 -0.062 0.415 0.334 0.408 0.654 0.675</td><td>-0.128 -0.085 -0.033 0.432</td><td>-0.082 -0.044 -0.006 0.429</td></tr><tr><td>ENEM</td><td>Direct RP Prompt-C Profile-C HACHIMI-II 0.370</td><td>0.973 -0.009 0.956 0.290 0.918 0.391 0.677</td><td>0.511 0.461 0.479 0.638</td><td>0.416 0.399 0.416 0.500</td></tr></table>

Table 13: Additional item-response metrics.

Prompt-Constrained Simulation. Direct roleplay is extended with natural-language instructions that constrain accessible knowledge, reasoning depth, uncertainty, and error preservation. All control stays in the prompt.

Profile-Conditioned Simulation. Generation is conditioned on the visible context and a learner profile. In item-response simulation the profile is the student model built by HACHIMI-II, and in dialogue it is a coarse profile computed from the learner’s earlier turns.

Agent-Style Memory Simulation. Used only in dialogue, this baseline conditions on the visible dialogue, a coarse learner profile, and a short-term memory of the interaction.

## F Full Results

Table 13 reports additional item-response metrics, namely the Kolmogorov–Smirnov statistic between the real and simulated distributions of per-student accuracy, the Pearson correlation of per-student accuracy, and the Spearman and Pearson correlations of per-item accuracy. On all three datasets, HACHIMI-II has the smallest Kolmogorov–Smirnov statistic and the highest Pearson correlations of per-student and per-item accuracy. Figure 4 shows per-student accuracy for every method. Table 14 reports additional dialogue metrics. On both datasets, HACHIMI-II has the highest error recall and $\kappa _ { \mathrm { e r r } }$

## G Full Ablation Results

Tables 15 and 16 report the item-response and dialogue metrics of the full model and the three ablations. Removing the learner state gives the largest change in $W _ { 1 }$ and $\mathrm { { F 1 } _ { \mathrm { { s t r u c t } } } }$ on every dataset.

Without the learner state, the standard deviation of simulated accuracy falls to 0.035–0.086, and on XES3G5M and ENEM it is larger with the persona prior than without it.

## H Prompts

Listings 1 to 4 give the system prompts that the dialogue pipeline uses for learner state inference, the predicted error, generation, and the harness check. In each request the prompt is followed by the visible context and the schema of the fields to return. The field names are the identifiers of the learner-state fields in Table 10. In the prompts, the error locus follows Newman’s error analysis (Newman, 1977; White, 2010), the error mechanism follows Radatz (1979), the faulty rule follows the procedural bugs of Brown and Burton (1978), and the engagement categories follow the ICAP framework (Chi and Wylie, 2014). The item-response pipelines use corresponding prompts for state inference, generation, and the check, with the fields of the item-response column of Table 10.

![](images/694960579cebce606fab1df7101512e009c0383353a4fc84b7c894d250e33819.jpg)  
Figure 4: Real and simulated accuracy per student for all methods and datasets. The dashed line is $y = x ,$ and the solid line is a linear fit.

<table><tr><td>Dataset</td><td>Method</td><td>Error recall</td><td>Error precision</td><td> $\kappa _ { \mathrm { e r r } }$ </td><td> $\mathrm { P r e c } _ { \mathrm { s t r u c t } }$ </td></tr><tr><td rowspan="5">Eedi QATD2k</td><td>Direct RP</td><td>0.168</td><td>0.393</td><td>0.156</td><td>0.368</td></tr><tr><td>Prompt-C</td><td>0.161</td><td>0.404</td><td>0.153</td><td>0.375</td></tr><tr><td>Profile-C</td><td>0.184</td><td>0.382</td><td>0.164</td><td>0.358</td></tr><tr><td>Agent-Mem</td><td>0.173</td><td>0.392</td><td>0.160</td><td>0.368</td></tr><tr><td>HACHIMI-II</td><td>0.521</td><td>0.608</td><td>0.507</td><td>0.368</td></tr><tr><td rowspan="5">Khan/CoMTA</td><td>Direct RP</td><td>0.084</td><td>0.333</td><td>0.039</td><td>0.316</td></tr><tr><td>Prompt-C</td><td>0.071</td><td>0.327</td><td>0.032</td><td>0.306</td></tr><tr><td>Profile-C</td><td>0.119</td><td>0.397</td><td>0.078</td><td>0.368</td></tr><tr><td>Agent-Mem</td><td>0.110</td><td>0.431</td><td>0.082</td><td>0.414</td></tr><tr><td>HACHIMI-II</td><td>0.758</td><td>0.335</td><td>0.220</td><td>0.249</td></tr></table>

Table 14: Additional dialogue metrics.

<table><tr><td></td><td></td><td></td><td></td><td></td><td colspan="2">Sim. acc.</td></tr><tr><td>Dataset</td><td>Variant</td><td>Agree↑</td><td>W1↓</td><td>ρ↑</td><td>Mean</td><td>SD</td></tr><tr><td rowspan="4">XES3G5M</td><td>HACHIMI-II</td><td>0.752</td><td>0.039</td><td>0.691</td><td>0.828</td><td>0.132</td></tr><tr><td>w/o Prior</td><td>0.741</td><td>0.025</td><td>0.603</td><td>0.817</td><td>0.116</td></tr><tr><td>w/o State</td><td>0.784</td><td>0.074</td><td>0.664</td><td>0.905</td><td>0.066</td></tr><tr><td>w/o Harness</td><td>0.760</td><td>0.056</td><td>0.559</td><td>0.846</td><td>0.134</td></tr><tr><td rowspan="4">Eedi</td><td>HACHIMI-II</td><td>0.606</td><td>0.046</td><td>0.703</td><td>0.593</td><td>0.211</td></tr><tr><td>w/o Prior</td><td>0.610</td><td>0.041</td><td>0.704</td><td>0.587</td><td>0.237</td></tr><tr><td>w/o State</td><td>0.578</td><td>0.116</td><td>0.742</td><td>0.642</td><td>0.086</td></tr><tr><td>w/o Harness</td><td>0.606</td><td>0.048</td><td>0.703</td><td>0.594</td><td>0.212</td></tr><tr><td rowspan="4">ENEM</td><td>HACHIMI-II</td><td>0.630</td><td>0.169</td><td>0.668</td><td>0.384</td><td>0.328</td></tr><tr><td>w/o Prior</td><td>0.592</td><td>0.142</td><td>0.594</td><td>0.500</td><td>0.296</td></tr><tr><td>w/o State</td><td>0.487</td><td>0.409</td><td>0.539</td><td>0.802</td><td>0.035</td></tr><tr><td>w/o Harness</td><td>0.617</td><td>0.257</td><td>0.643</td><td>0.481</td><td>0.402</td></tr></table>

Table 15: Item-response ablation results. Sim. acc. gives the mean and the standard deviation of simulated perstudent accuracy. The real means are 0.831 (XES3G5M), 0.574 (Eedi), and 0.394 (ENEM).

<table><tr><td>Dataset</td><td>Variant</td><td> $\kappa \mathrm { f u n c }$  ←</td><td> $\mathrm { M C C } _ { \mathrm { f u n c } }$  ↑</td><td> $\mathrm { B A c c } _ { \mathrm { e r r } }$  ←</td><td> $\mathrm { M C C } _ { \mathrm { e r r } }$ </td><td>↑  $\mathrm { F 1 _ { e r r } }$  ←</td><td> $\mathrm { R e c _ { s t r u c t } }$ </td><td>↑  $\mathrm { P r e c } _ { \mathrm { s t r u c t } }$ </td><td>↑ F1struct ↑</td></tr><tr><td rowspan="4">Eedi QATD2k</td><td>HACHIMI-II</td><td>0.659</td><td>0.673</td><td>0.738</td><td>0.509</td><td>0.561</td><td>0.315</td><td>0.368</td><td>0.339</td></tr><tr><td>w/o Prior</td><td>0.571</td><td>0.635</td><td>0.834</td><td>0.603</td><td>0.651</td><td>0.300</td><td>0.236</td><td>0.264</td></tr><tr><td>w/o State</td><td>0.741</td><td>0.753</td><td>0.612</td><td>0.322</td><td>0.346</td><td>0.099</td><td>0.220</td><td>0.136</td></tr><tr><td>w/o Harness</td><td>0.565</td><td>0.626</td><td>0.808</td><td>0.588</td><td>0.639</td><td>0.307</td><td>0.275</td><td>0.290</td></tr><tr><td rowspan="4">Khan/CoMTA</td><td>HACHIMI-II</td><td>0.370</td><td>0.390</td><td>0.660</td><td>0.267</td><td>0.464</td><td>0.565</td><td>0.249</td><td>0.346</td></tr><tr><td>w/o Prior</td><td>0.326</td><td>0.359</td><td>0.618</td><td>0.200</td><td>0.426</td><td>0.578</td><td>0.224</td><td>0.323</td></tr><tr><td>w/o State</td><td>0.382</td><td>0.436</td><td>0.519</td><td>0.063</td><td>0.148</td><td>0.077</td><td>0.262</td><td>0.119</td></tr><tr><td>w/o Harness</td><td>0.348</td><td>0.367</td><td>0.625</td><td>0.212</td><td>0.443</td><td>0.563</td><td>0.249</td><td>0.346</td></tr></table>

Table 16: Dialogue ablation results.

![](images/e31666a3a3f9a551d2d2f54438126b8989abfb7b7651017b089fc53cd5bef99e.jpg)

<table><tr><td>You are building a third-person ErrorBlueprint for the student&#x27;s next turn. This is not free generation of the student reply. Use only the visible dialogue history, the current tutor move, the current interaction state, and the predicted next state. Do not use future turns, gold labels, official annotations, or evaluation targets. If a persona prior summary is provided, use it only as a background prior for the current learner.</td></tr><tr><td>Do not treat the persona prior or retrieved exemplars as the answer script for the current turn. Theory rules: - For error locus, use Newman Error Analysis as summarized by White (2010):</td></tr><tr><td>reading | comprehension | transformation | process_skills | encoding - Choose the earliest evidenced breakdown in the Newman chain.</td></tr><tr><td>- For error mechanism, use Radatz (1979): language_difficulty I spatial_information_difficulty | deficient_prerequisites | incorrect_associations | irrelevant_rules</td></tr><tr><td></td></tr><tr><td>- For buggy_rule, use the procedural-bug idea from Brown &amp; Burton: express one short executable mal-rule in plain language, not a broad personality description.</td></tr><tr><td></td></tr><tr><td>Local anchoring rules:</td></tr><tr><td>- Anchor the blueprint to the current tutor move, not to a random earlier error from the dialogue. - wrong_claim_seed must target the exact local object the tutor is currently eliciting.</td></tr><tr><td>- Use tutor_turn_profile.elicited_response_target_type and tutor_turn_profile.elicited_object_description to decide what kind</td></tr><tr><td>of wrong claim belongs here. - If the tutor is asking for an option, judgment, or final local answer, the wrong_claim_seed should usually be an incorrect</td></tr><tr><td>option, incorrect judgment, or incorrect local answer of that same kind.</td></tr><tr><td>- If the tutor is asking for the next step in a procedure, the wrong_claim_seed should usually be an incorrect local step at that exact stage, not a different earlier-stage mistake.</td></tr><tr><td>- Do not shift the student to a different subproblem or a different response function than the tutor&#x27;s current question.</td></tr><tr><td></td></tr><tr><td>Newman interpretation rules for tutoring context: - Use comprehension when the student is likely to misunderstand what the tutor is asking, what kind of answer is being</td></tr><tr><td>requested, or how to interpret the compared methods/statements.</td></tr><tr><td>- Use transformation when the student is likely to choose the wrong representation, wrong operation, wrong equation, or wrong mathematical mapping for the current local task.</td></tr><tr><td>- Use process_skills when the tutor has already supplied the relevant representation, operation, or target form and the</td></tr><tr><td>student is more likely to execute that local step incorrectly. - Use encoding when the student has essentially completed the local work but is likely to express the final answer in the</td></tr><tr><td>wrong written or symbolic form.</td></tr><tr><td>Error-presence rules:</td></tr><tr><td>- no_specific_error:</td></tr><tr><td>use this when the next turn is not best represented as a specific mathematical error. - non_codable_confusion:</td></tr><tr><td>use this when the next turn is likely confused, incomplete, or help-seeking, but not stable enough to code as one specific</td></tr><tr><td>mathematical error. - codable_math_error:</td></tr><tr><td>use this only when the next turn is likely to contain one concrete mathematical error that can be realized in the utterance.</td></tr><tr><td></td></tr><tr><td>Coding rules: - For Brown &amp; Burton style buggy rules, code the local executable wrong rule the student is most likely to apply next, not the</td></tr><tr><td>corrected rule the tutor is trying to elicit.</td></tr><tr><td>- If the tutor has already supplied the correct representation or operation and is now checking execution, selection, or</td></tr><tr><td>evaluation, do not drift back to a broader earlier-stage transformation error unless the next turn is truly likely to do that.</td></tr><tr><td>- If error_presence is not codable_math_error, set newman_error_locus, radatz_error_mechanism, buggy_rule, and</td></tr><tr><td></td></tr><tr><td>wrong_claim_seed to null.</td></tr><tr><td>- If error_presence is codable_math_error, all four of those fields must be non-null.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>- wrong_claim_seed must describe the local incorrect claim or incorrect step the student is most likely to say next, not a</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>polished explanation.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>- Keep the blueprint local to this turn. Do not script multiple future turns.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>- Every required evidence list must be non-empty.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>- Top-level keys must match required_schema exactly.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>Return only the requested JSON object.</td></tr></table>

![](images/46bb7949fc5a0b902cacc50d093668490c944c6b021250d30f1f22cad8dbd6d9.jpg)  
Listing 3: Generation of the turn.

![](images/9b02fa439042179e10b4d7a15c81d0a214e6d606603cd2e5072428d2e1f60db5.jpg)