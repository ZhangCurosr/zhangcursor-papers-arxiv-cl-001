# A Unified Response Framework for Questions and Tutoring Dialogues

Zhangqi Duan<sup>1</sup>, Simon Woodhead<sup>2</sup>, Maya Miller-Vedam<sup>3</sup>, Andrew Lan<sup>1</sup>

<sup>1</sup>University of Massachusetts Amherst, <sup>2</sup>Eedi, <sup>3</sup>Khan Academy

{zduan, andrewlan}@cs.umass.edu,

simon.woodhead@eedi.co.uk,

mayavedam@khanacademy.org

## Abstract

Existing student models in educational applications largely focus on student responses to questions. However, as tutoring becomes more accessible at large scale, it is important to analyze dialogues between tutors and students, which may reveal fine-grained evidence on student knowledge. In this work, we propose a unified framework that models both student responses to questions and tutor-posed subtasks during dialogues on a shared item response theory (IRT) scale. We first identify mathematical subtasks in tutoring dialogues, rewrite them into self-contained questions, and identify the correctness of student responses to these subtasks. We then develop an IRT framework that estimates question/subtask difficulty from text, tracks students’ evolving abilities across question-answering and dialogue interactions, and a novel method to leverages tutor moves between consecutive subtasks to regularize their difficulty values. We evaluate our framework on two real-world math education datasets containing both human- and AI-tutored dialogues. Results show that our framework outperforms strong baselines on both question difficulty prediction and response correctness prediction.

## 1 Introduction

Estimating student knowledge and item characteristics from students’ responses to mathematical questions/items is central to online learning platforms, which supports personalized practice, targeted tutoring, and reliable assessment. There are two major types of student response modeling approaches: Item response theory (IRT) models place static student ability and question difficulty on a common scale (Rasch, 1960), while knowledge tracing (KT) models track how a student’s knowledge evolves over time across a sequence of questions (Corbett and Anderson, 1994). One key limitation of these models is that they only account for students’ response records, which may leave out fine-grained information on their knowledge. For example, for multiple-choice questions, only knowing whether a student selected the correct option or which incorrect option they selected does not allow us to pinpoint their misunderstanding.

![](images/62a0167d425cad41cb33734d3c9280fb514821f44c32f130f4864a48a1a3e4a2.jpg)  
Figure 1: Student responses to tutor-posed subtasks in tutoring dialogues reveal fine-grained insights into the student’s changing knowledge, while tutor moves help infer the relative difficulty of subtasks.

Powered by generative artificial intelligence (AI), online learning platforms increasingly combine standalone question-answering practice with on-demand tutoring (Eedi, 2024; Khan Academy, 2023; Carnegie Learning, 2024). Tutoring dialogues have significant potential in providing us fine-grained insights into student knowledge: a tutoring dialogue is not merely an explanation attached to a question; it is a sequence of smaller assessment and learning events. A tutor may ask the student to recall a fact, perform an intermediate calculation, or justify a step; after observing the response, the tutor may scaffold or push the student toward more rigorous reasoning (Zhang et al., 2026). The student’s knowledge can therefore be more accurately estimated and may change throughout the dialogue. Figure 1 illustrates this process. The original response only indicates that the student answered the fraction-division question incorrectly, whereas the dialogue reveals an initial conceptual error, success on a simpler scaffolded task, correct intermediate calculations, and a remaining gap in justification. These tutor-posed tasks and student responses may provide substantially more insights into the student’s latent knowledge than the original response alone.

Related Work However, existing approaches do not provide a rigorous way to analyze original question responses and tutor-posed tasks together. Prior work has explored question difficulty prediction from question content (Huang et al., 2017; Benedetto et al., 2021) and simulated student responses (Scarlatos et al., 2025b), as well as KT in tutoring dialogues (Scarlatos et al., 2025a). More recently, Huang et al. (2026) incorporate tutor-turn difficulty and IRT into conversational KT. However, two challenges remain for jointly modeling original questions and dialogue subtasks. First, tutoring dialogues follow personalized trajectories, resulting in subtasks that are often not shared across students and thus difficult to calibrate using traditional IRT. Second, many tutor-posed tasks implicitly refer to quantities, expressions, or reasoning steps introduced earlier in the problem or dialogue, making them difficult to represent as self-contained, standalone questions. Together, these challenges leave open the question of how to place original questions and tutor-posed tasks on the same scale and use them to track student knowledge. We provide a detailed discussion of related work in Appendix A.

Contributions In this work, we propose URF-QD (Unified Response Framework for Questions and tutoring Dialogues), a framework for jointly modeling responses to original questions and mathematical subtasks arising in corresponding tutoring dialogues. Our contributions are threefold. First, we identify tutor turns that pose mathematical tasks and rewrite these context-dependent tasks as concise, self-contained questions, enabling their difficulties and student responses to be modeled jointly with those of original questions. Second, we introduce a dynamic IRT model that tracks the student’s evolving knowledge state from responses to both original questions and dialogue subtasks; importantly, we introduce a novel tutor move-implied regularization term to supervise a relative ordering between consecutive subtasks. Third, we evaluate our framework on two real-world math education datasets containing both question-answering records and tutoring dialogues. Extensive evaluation shows that our framework consistently improves question difficulty prediction and response correctness prediction. We use human evaluation and qualitative analyses to validate and illustrate components of our framework, and present a preliminary example of how our framework can be used for large language model (LLM) tutor benchmarking in the Appendix. Our implementation will be made publicly available to enable future research on student assessment from dialogues.

## 2 Problem Formulation

We consider the setting of students working in digital learning platforms, where they practice on math questions and interact with tutors when they want additional support. Our goal is to estimate a student’s ability from how they respond to both questions and tutor-posed subtasks that arise naturally during tutoring dialogues.

For each student j, we denote the binary-valued correctness of their response to an original question $q _ { i }$ as $y _ { i , j } \in \{ 0 , 1 \}$ , where 1 corresponds to a correct response. When receiving tutoring support on question $q _ { i }$ , the student and the tutor engage in a dialogue $d _ { i , j } = ( s _ { 0 } , t _ { 1 } , s _ { 1 } , . . . , t _ { M } , s _ { M } )$ , where $s _ { m }$ and $t _ { m }$ denote student and tutor utterances respectively and M is the number of total tutor-student turn pairs. The initial student turn $s _ { 0 }$ is present when the student initiates the conversation.

We focus on tutor turns that pose a mathematical subtask, i.e., when the tutor requires the student to produce a mathematically meaningful response, such as an intermediate calculation, factual recall, method selection, interpretation, or explanation. We treat student responses to these tutor-posed tasks as additional response observations. We detail how these tasks are identified and transformed into standalone questions in Section 3.1.

![](images/d07d8dd7ccb9681a12389fcf171eb70d2ba0eb42de77a4cda01a3c36ef80e902.jpg)  
Figure 2: Overview of URF-QD. (A) Original questions and rewritten dialogue subtasks are modeled on a shared IRT scale using text-based difficulty estimates and evolving student ability. (B) Relative-order regularization encourages scaffolded tasks to be easier and rigor-promoting tasks harder than preceding tasks. (C) Response correctness uses BCE with the regularization loss, while question difficulty uses GT-difficulty MSE, dialogue-subtask BCE, and the regularization loss. GT denotes IRT-estimated “ground-truth” question difficulties and student abilities obtained from observed student responses to original questions.

For each student $j ,$ we order all scored interactions chronologically and denote the resulting sequence as $H _ { j } \ = \ ( ( x _ { j , 1 } , y _ { j , 1 } ) , . . . , ( x _ { j , T } , y _ { j , T } ) )$ where $x _ { j , k }$ denotes the mathematical task associated with interaction k, and $y _ { j , k } \in \{ 0 , 1 \}$ denotes the corresponding response correctness. An interaction may correspond either to an original question or to a subtask arising within a tutoring dialogue.

Given the interaction history $H _ { j , < k } ,$ , our primary modeling objective is to estimate $p ( y _ { j , k } =$ $1 | x _ { j , k } , H _ { j , < k } )$ , while placing student ability and the difficulties of original questions and dialogue tasks on a common IRT scale. In this way, questionanswering and tutoring-dialogue interactions provide complementary observations for modeling student performance.

## 3 Methodology

We now detail our framework for jointly modeling student performance in question-answering activities and tutoring dialogues. Our key idea is to place original questions and tutor-posed dialogue subtasks on the same IRT parameter scale. As shown in Figure 2, our framework has three main components. First, we identify mathematical subtasks posed by tutors during dialogues and transform these context-dependent tasks into the format of standalone questions. Second, we use a dynamic

IRT model to jointly model responses to original questions and transformed dialogue subtasks on a shared scale. Third, we leverage tutor pedagogical move labels to impose difficulty ordering constraints between consecutive tutor-posed tasks, to provide additional supervision when estimating their difficulties. We note that, for simplicity, we restrict ourselves to 1PL IRT models (Hambleton and Swaminathan, 2013), which is often referred to as the Rasch model (Rasch, 1960), where each question is only characterized by a single scalarvalued difficulty parameter. Our framework can be extended to other IRT models, although difficulty estimation is known to be more accurate than scale/discrimination estimation for 2PL IRT models (Chen and Chen, 2026).

## 3.1 Subtask Transformation

Tutoring dialogues contain many intermediate mathematical tasks that provide additional evidence of student ability. However, tutor utterances are often context-dependent and differ substantially in form from standalone assessment questions. We therefore first identify tutor turns that constitute mathematical subtasks and then rewrite each identified subtask into a standalone question, allowing original questions and subtasks to be represented in a common form for subsequent modeling.

Subtask Identification For each dialogue $d _ { i , j }$ we consider every tutor turn that is immediately followed by a student turn as a candidate subtask. We prompt GPT-5.4 with the complete dialogue to identify the valid mathematical subtasks: for each candidate, it determines whether the tutor turn poses an explicit mathematical task and whether the subsequent student turn attempts to answer it. We retain only candidates that satisfy both conditions. Following (Scarlatos et al., 2026), for each retained tutor subtask, we annotate whether the student response $s _ { m }$ correctly answers the tutor-posed task. For each scored response, we denote this correctness label as $y _ { i , j , m } \in \{ 0 , 1 \}$ . Turns without an applicable mathematical response are left unscored and are not used as response observations.

Standalone Question Transformation For every identified tutor subtask $t _ { m } .$ , we prompt GPT-5.4 to rewrite the tutor’s request as a concise, standalone assessment question $u _ { i , j , m }$ . Given the original problem and dialogue with the target turn marked, the model resolves contextual references and supplies only the information needed to make the question self-contained while preserving its mathematical demand. We provide further prompt details in Appendix G.

For each scored subtask, we map the transformed question $u _ { i , j , m }$ and its response label $y _ { i , j , m }$ to the corresponding interaction k in the student’s chronological history $H _ { j } : x _ { j , k } = u _ { i , j , m } , y _ { j , k } = y _ { i , j , m } .$ This mapping represents dialogue subtasks in the same form as original questions, enabling both interaction types to be modeled jointly.

## 3.2 URF-QD: Joint Modeling of Questions and Dialogue Subtasks

Given the chronological interaction history $H _ { j }$ which contains both original question responses and transformed dialogue subtask interactions, we model student performance using a dynamic IRT framework. We process interactions sequentially and maintain a latent student state $\mathbf { h } _ { j , k }$ that summarizes the student’s interactions up to step k. At interaction $k _ { : }$ , we first encode the current task $x _ { j , k }$ and estimate its difficulty. We then project the preceding state $\mathbf { h } _ { j , k - 1 }$ to a scalar ability and use a 1PL IRT model to predict the student’s response. After observing $y _ { j , k }$ , we update the student state using the task representation and response correctness.

Task Encoder and Difficulty We encode each task $x _ { j , k }$ using an LLM. Let $\begin{array} { r l } { \mathbf { Z } _ { j , k } } & { { } = } \end{array}$ $( \mathbf { z } _ { j , k , 1 } , \dotsc , \mathbf { z } _ { j , k , L _ { j , k } } )$ denote the last-layer hidden state representations of the question sequences $L _ { j , k }$ We obtain a fixed-dimensional task representation by mean pooling: $\begin{array} { r } { { \bf e } _ { j , k } \ = \ \frac { 1 } { L _ { i , k } } \sum _ { \ell = 1 } ^ { L _ { j , k } } { \bf z } _ { j , k , \ell } } \end{array}$ . We map this representation to a scalar difficulty using a multilayer perceptron (MLP) g<sub>dif</sub>: $b _ { j , k }$ $g _ { \mathrm { d i f f } } ( \mathbf { e } _ { j , k } )$ . The task encoder and difficulty prediction head are shared across original questions and dialogue subtasks, placing both on the same latent difficulty scale. This text-based parameterization is particularly important for dialogue subtasks, since different students may encounter different tutorposed tasks even when working on the same original question. Rather than assigning an independent difficulty parameter to every observed task, the model estimates difficulty from its textual content, enabling the learned difficulty function to generalize across questions, students, and dialogues.

Dynamic Student State and Ability We represent student $j ^ { \prime } \mathbf { s }$ interaction history with a latent state $\mathbf { h } _ { j , k }$ that summarizes all observed taskresponse pairs through step k. After observing the student’s response $y _ { j , k }$ to task $x _ { j , k }$ , we incorporate this new evidence into the state. We first represent response correctness using a learned embedding: $\mathbf { r } _ { j , k } = E _ { y } ( y _ { j , k } )$ , where $E _ { y }$ contains separate learnable embeddings for correct and incorrect responses to both original questions and dialogue subtasks. We then concatenate resulting correctness embedding with task’s textual embedding. One can use any deep learning-based KT method for this dynamic update; unless otherwise noted, we defer to the deep KT (DKT) method (Piech et al., 2015).

IRT Response Model Before the student attempts task $x _ { j , k }$ , we estimate their current ability by projecting the preceding state to a scalar: $a _ { j , k } = g _ { a } ( \mathbf { h } _ { j , k - 1 } )$ , where $g _ { a }$ is a MLP. This preinteraction ability is used in the IRT response model below and does not depend on the current response $y _ { j , k }$ . After that response is observed, the updated state $\mathbf { h } _ { j , k }$ is used to estimate ability at interaction $k + 1$ . Unlike a static IRT ability parameter, $a _ { j , k }$ is dynamic throughout the student’s interaction sequence and updated when new responses are observed. Therefore, dialogue subtask responses serve not only as correctness-prediction targets, but also as additional observations that refine future ability estimates. Given the student’s current ability $a _ { j , k }$ and the difficulty of the current task, $b _ { j , k }$ we model response correctness using a 1PL logistic

IRT model:

$$
p ( y _ { j , k } = 1 \mid x _ { j , k } , H _ { j , < k } ) = \sigma ( a _ { j , k } - b _ { j , k } ) ,
$$

where $\sigma ( \cdot )$ denotes the sigmoid function. Because original questions and dialogue subtasks share the same task encoder, difficulty head, ability projection, and IRT response function, their difficulties and the student’s dynamic ability are represented on a common latent scale.

## 3.3 Difficulty Order Regularization

Correctness observations supervise the relationship between student ability and task difficulty, but tutoring dialogues provide an additional source of information: the tutor’s pedagogical move can imply a relative difficulty ordering between consecutive subtasks. We exploit two cases, scaffolding after struggle and pushing for rigor (Zhang et al., 2026), and use them to regularize task difficulties.

Pedagogical Move Annotation For each pair of consecutive tutor subtasks separated by a single student response, we prompt GPT-5.4 to determine whether the transition implies a tutor move that implies any relationship between the subtask difficulties. The model is provided with the original problem, the raw tutoring dialogue with subtasks marked, eligible tutor-turn pairs, and the annotated correctness label of the intervening student response. We withhold the student’s response to the second subtask, so that the relation is determined from the tutor’s pedagogical action rather than from whether the student subsequently succeeds or fails.

We denote the set of turns that correspond to scaffolding after struggle moves as $\Omega _ { 1 }$ . In practice, when students respond incorrectly to tutor-posed subtasks or indicate that they do not know the answer, tutors often scaffold by breaking down the original subtask into smaller, simpler, more concrete subtasks; such moves imply that $b _ { j , k } < b _ { j , k - 1 }$ Similarly, we denote the set of pushing for rigor turns as Ω<sub>2</sub>. A pair is assigned to $\Omega _ { 2 }$ when the previous student response is correct and the subsequent tutor subtask increases the required depth of reasoning, for example by eliciting a justification, explanation, derivation, interpretation, proof, or other conceptual reasoning; such moves imply $b _ { j , k } > b _ { j , k - 1 }$ . We assign no constraint when the intervening response cannot be scored for correctness, when the two subtasks are unrelated, or when the second tutor turn merely repeats the task, requests a generic retry, continues a routine multistep solution, or otherwise does not substantively change the cognitive demand.

Regularization For each $( j , k ) \in \Omega _ { 1 }$ , the second task should be easier than the preceding task. We define the corresponding hinge loss as

$$
\mathcal { L } _ { \mathrm { s c a f } } = \frac { 1 } { | \Omega _ { 1 } | } \sum _ { ( j , k ) \in \Omega _ { 1 } } \operatorname* { m a x } \{ b _ { j , k } - b _ { j , k - 1 } , 0 \} .
$$

The loss is positive when the scaffolded task is more difficult than the previous task, violating the tutor move, and zero otherwise. We define the loss for pushing for rigor moves, ${ \mathcal { L } } _ { \mathrm { { r i g o r } } } ,$ in a similar way, with $b _ { j , k }$ and $b _ { j , k - 1 }$ reversed. We simply sum these two regularization terms together to obtain $\mathcal { L } _ { \mathrm { o r d e r } }$ and add it to the objectives detailed below, using them as soft penalties to encourage the difficulty encoder to learn dialogue subtask difficulties that are consistent with tutoring moves.

## 3.4 Prediction Targets, Training Objectives

We study two complementary prediction targets using separate training objectives. Question difficulty prediction evaluates whether the shared difficulty encoder can accurately estimate the difficulty of previously unseen questions. Response correctness prediction evaluates whether original-question and dialogue-subtask responses can jointly track a student’s evolving ability and accurately predict their future performance.

Question Difficulty Prediction Here, we examine whether dialogue subtasks provide additional calibration to help us estimate the difficulty of original questions. We first fit an IRT model on observed student responses and use it to obtain a “groundtruth” difficulty $\tilde { b } _ { i }$ for each original question $q _ { i }$ . We then train the shared difficulty predictor to recover these difficulties, while using dialogue interactions as additional supervision. For an original question $q _ { i } .$ , let $b _ { i }$ denote the difficulty value predicted by the encoder given the question content as input. We supervise these predictions using mean squared error over the original questions available for training, as $\begin{array} { r } { \mathcal { L } _ { \mathrm { d i f f - M S E } } = \frac { 1 } { \left| \mathcal { Q } _ { \mathrm { t r a i n } } \right| } \sum _ { i \in \mathcal { Q } _ { \mathrm { t r a i n } } } \left( b _ { i } - \tilde { b } _ { i } \right) ^ { 2 } } \end{array}$

Since dialogue subtasks do not have corresponding ground-truth difficulty labels, we additionally train on the observed student response correctness to extract information on subtask difficulty indirectly. Let $\tilde { a } _ { j }$ denote the fixed reference ability of student $j$ from the fitted IRT, and let $b _ { i , j , m }$ be the predicted difficulty of transformed dialogue subtask $u _ { i , j , m } ,$ we minimize $\mathcal { L } _ { \mathrm { d i f f - B C E } } .$ the standard binary cross-entropy (BCE) loss between actual subtask response correctness labels $y _ { i , j , m }$ and predicted correct response probability $\sigma ( \tilde { a } _ { j } - b _ { i , j , m } )$ . We combine these loss terms as $\mathcal { L } _ { \mathrm { d i f f } } = \mathcal { L } _ { \mathrm { d i f f - M S E } } + \mathcal { L } _ { \mathrm { d i f f - B C E } } + \lambda \mathcal { L } _ { \mathrm { o r d e r } }$ , where λ is the relative-order regularization weight.

Response Correctness Prediction Here, given the interaction history $H _ { j , < k }$ and the current task $x _ { j , k }$ , we predict whether student $j$ will respond correctly to $x _ { j , k }$ using dynamic student ability $\boldsymbol { a } _ { j , \boldsymbol { k } }$ and predicted task difficulty $b _ { j , k }$ . We define the BCE loss $\mathcal { L } _ { \mathrm { B C E } }$ analogously to L<sub>dif−BCE</sub> and combine it with the difficulty ordering regularization term to obtain the objective of $\mathcal { L } _ { \mathrm { { r e s p } } } = \mathcal { L } _ { \mathrm { { B C E } } } + \lambda \mathcal { L } _ { \mathrm { { o r d e r } } } .$ . In addition to contributing to the training objective, dialogue subtask responses also update the student’s knowledge state, thereby influencing future predictions. This formulation enables us to investigate whether training on student responses to subtasks in dialogues provide additional, fine-grained information on their knowledge.

## 4 Experimental Evaluation

We now detail our experimental setup to validate the effectiveness of our framework.

## 4.1 Datasets

We conduct experiments on two math tutoring datasets collected by two major online learning platforms, Eedi and Khan Academy. For simplicity, we refer to these datasets by their respective platform names. Both datasets contain student interactions with math questions, where students can initiate chat-based tutoring conversations on demand at any point during the questionanswering process. The Eedi dataset contains conversations between real students and human tutors, whereas the Khan Academy dataset contains conversations between real students and an AI tutor model. After processing, the Eedi dataset contains 588 students, 3, 689 questions, 1, 775 dialogues, and 152, 280 question-answering records, while the Khan Academy dataset contains 136 students, 3, 423 questions, 1, 800 dialogues, and 56, 529 question-answering records. We provide additional dataset details in Appendix B.

## 4.2 Metrics

For question difficulty prediction, following prior work (Huang et al., 2017; Scarlatos et al., 2025b), we use the widely adopted Pearson Correlation Coefficient (PCC) to measure the linear correlation between predicted and ground-truth question difficulties, Spearman Rank Correlation Coefficient (SCC) to measure the correlation between predicted and ground-truth difficulty rankings among questions, Mean Squared Error (MSE) to measure squared prediction error, and Mean Absolute Error (MAE) to measure absolute prediction error. For response correctness prediction, we use Area Under the ROC Curve (AUC), accuracy, and F1 score metrics.

## 4.3 Baselines

We compare URF-QD against kNN Mean, Random Forest, ICL kNN, and Diff SFT (ModernBERT and Qwen3.5-4B) for question difficulty prediction. For response correctness prediction, we compare against DKT, DKT (Joint), AKT, simpleKT, and URF-QD (Orig. Question) to assess the contribution of dialogue subtask responses within and beyond our framework. We use Joint to denote training on both original-question and dialogue-subtask responses and $R e g .$ to denote relative difficulty order regularization. We provide detailed baseline descriptions in Appendix C.1.

## 4.4 Experimental Setup

We report the average performance on 5-fold cross validation. We split datasets across problems (students) for question difficulty prediction (response correctness prediction) evaluation, to evaluate generalization to unseen problems (students). We use Qwen3.5-4B for URF-QD and evaluate ICL kNN using GPT-5.4. We provide full experimental details in Appendix C.2.

## 5 Results, Analysis, and Discussion

We now detail our quantitative results on question difficulty prediction and student response correctness prediction, and qualitatively analyze the impact of regularization on both tasks.

## 5.1 Quantitative Results

Our framework excels at question difficulty prediction. Table 1 reports the average performance (and standard deviation) on both datasets for question difficulty prediction. We see that URF-QD (Joint + Reg.) outperforms all baselines across all four metrics. This consistent result shows that dialogue subtasks provide useful supervision for predicting difficulty of unseen original questions, even though the subtasks themselves have no groundtruth difficulty labels. On Eedi, compared with the strongest baseline, URF-QD reduces MSE by 7.7% and improves PCC and SCC by 6.5% and 8.0%, respectively. On Khan Academy, URF-QD similarly reduces MSE and MAE, and improves PCC and SCC. The improvements over all baselines are statistically significant for MSE, PCC, and SCC on Eedi and for PCC on Khan Academy $( p < 0 . 0 5$ under paired t-test).

<table><tr><td>Dataset</td><td>Method</td><td>MSE↓</td><td>MAE↓</td><td>PCC↑</td><td>SCC ↑</td></tr><tr><td rowspan="6">Eedi</td><td>kNN Mean (SBERT)</td><td> $\overline { { 0 . 4 4 5 _ { \pm 0 . 0 2 2 } } }$ </td><td> $0 . 5 2 3 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 2 6 8 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 2 6 1 _ { \pm 0 . 0 2 3 }$ </td></tr><tr><td>Random Forest (SBERT)</td><td> $\underline { { 0 . 3 7 9 } } _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 4 8 2 _ { \pm 0 . 0 2 4 }$ </td><td> $0 . 3 5 9 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 3 3 8 { \scriptstyle \pm 0 . 0 2 9 }$ </td></tr><tr><td>ICL kNN (GPT-5.4)</td><td> $0 . 5 4 8 _ { \pm 0 . 0 1 3 }$ </td><td> $0 . 5 8 9 _ { \pm 0 . 0 2 6 }$ </td><td> $0 . 3 6 0 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 3 6 3 _ { \pm 0 . 0 1 6 }$ </td></tr><tr><td>Diff SFT (ModernBERT)</td><td> $0 . 5 3 5 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 5 9 2 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 3 7 4 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 3 6 6 _ { \pm 0 . 0 2 0 }$ </td></tr><tr><td>Diff SFT (Qwen3.5-4B)</td><td> $0 . 3 8 2 _ { \pm 0 . 0 1 6 }$ </td><td> $\underline { { 0 . 4 8 2 } } _ { \pm 0 . 0 1 7 }$ </td><td> $\underline { { 0 . 4 3 0 } } _ { \pm 0 . 0 1 2 }$ </td><td> $\underline { { 0 . 4 1 4 } } _ { \pm 0 . 0 2 1 }$ </td></tr><tr><td>URF-QD (Joint + Reg.)</td><td> $\underline { { \mathbf { 0 . 3 5 0 } _ { \pm 0 . 0 1 5 } ^ { * } } }$ </td><td> $\mathbf { 0 . 4 7 3 _ { \pm 0 . 0 1 6 } }$ </td><td> $\underline { { \mathbf { 0 . 4 5 8 _ { \pm 0 . 0 1 5 } ^ { * } } } }$ </td><td> $\underline { { \mathbf { 0 . 4 4 7 _ { \pm 0 . 0 1 7 } ^ { * } } } }$ </td></tr><tr><td rowspan="6">Khan Academy</td><td>kNN Mean (SBERT)</td><td> $0 . 2 8 5 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 4 4 5 _ { \pm 0 . 0 1 9 }$ </td><td> $0 . 4 1 3 _ { \pm 0 . 0 1 3 }$ </td><td> $0 . 3 6 1 { \scriptstyle \pm 0 . 0 1 7 }$ </td></tr><tr><td>Random Forest (SBERT)</td><td> $0 . 2 8 3 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 4 6 4 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $\underline { { 0 . 4 2 2 } } _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 3 7 3 { \scriptstyle \pm 0 . 0 1 6 }$ </td></tr><tr><td>ICL kNN (GPT-5.4)</td><td> $0 . 3 2 3 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 4 5 3 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 4 0 7 _ { \pm 0 . 0 1 8 }$ </td><td> $0 . 3 1 5 { \scriptstyle \pm 0 . 0 1 0 }$ </td></tr><tr><td>Diff SFT (ModernBERT)</td><td> $0 . 3 3 7 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 4 5 4 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 4 1 4 _ { \pm 0 . 0 2 3 }$ </td><td> $\underline { { 0 . 3 7 9 } } _ { \pm 0 . 0 1 1 }$ </td></tr><tr><td>Diff SFT (Qwen3.5-4B)</td><td> $\underline { { 0 . 2 7 9 } } _ { \pm 0 . 0 2 4 }$ </td><td> $\underline { { 0 . 4 1 9 } } _ { \pm 0 . 0 2 0 }$ </td><td> $0 . 4 1 0 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 3 5 1 { \scriptstyle \pm 0 . 0 1 6 }$ </td></tr><tr><td>URF-QD  $( { \mathrm { J o i n t } } + { \mathrm { R e g . } } )$ </td><td> $\mathbf { 0 . 2 7 0 _ { \pm 0 . 0 2 2 } }$ </td><td> $\mathbf { 0 . 4 0 9 _ { \pm 0 . 0 2 0 } }$ </td><td> $\underline { { \mathbf { 0 . 4 4 6 _ { \pm 0 . 0 1 8 } ^ { * } } } }$ </td><td> $\mathbf { 0 . 3 9 2 _ { \pm 0 . 0 1 0 } }$ </td></tr></table>

Table 1: Performance comparison of methods on question difficulty prediction across datasets. Best performance is bolded and second best is underlined. <sup>∗</sup> denotes statistically significant improvement over all baselines $( p < 0 . 0 5 )$

These results suggest that analyzing tutor-posed subtasks do provide us with some information on question difficulty, which can be transferred to original questions via the shared text-based difficulty encoder, once put on the same IRT scale. This finding supports our central hypothesis that tutoring dialogues are not merely auxiliary conversational data; intermediate mathematical subtasks are important formative assessments that can directly improve question difficulty modeling.

baselines $( p < 0 . 0 5$ under paired t-test). On Khan Academy, the full model also consistently improves all three metrics over its original-question-only counterpart. DKT (Joint) similarly outperforms DKT across all metrics on both datasets, further confirming that dialogue responses provide complementary evidence for estimating dynamic student ability and predicting future performance.

Dialogue subtask responses improve response correctness prediction. Table 2 reports the average performance (and standard deviation) on both datasets on response correctness prediction. URF-QD (Joint + Reg.) again achieves the best performance on every metric for both Eedi and Khan Academy, showing that original question and dialogue subtask responses provide complementary evidence about students’ evolving knowledge. On Eedi, compared with URF-QD (Orig. Question), incorporating dialogue subtasks and relativeorder regularization improves AUC and accuracy by 6.3% and 4.5%, respectively. The gains in AUC and accuracy are statistically significant over all

Ablation Study. We conduct an ablation study on Eedi to assess the contribution of each framework component. Results show that incorporating dialogue subtasks improves both question difficulty and response correctness prediction, while relativeorder regularization provides additional gains for difficulty prediction. We provide detailed ablation results and analyses in Appendix D.1.

## 5.2 Qualitative Results

Dialogue subtasks provide fine-grained evidence of question difficulty. We now use a qualitative case study to illustrate how dialogue subtasks provide additional evidence for estimating the difficulty of an unseen original question. Table 3 shows a mixed-number division problem that requires several mathematical operations, including converting mixed numbers, dividing fractions, and expressing the result in the requested form. The question-only Diff SFT baseline predicts a difficulty of 0.483, underestimating the ground-truth difficulty of 0.625 by 0.142. The related dialogue subtasks capture mathematical demands that are required to solve the original problem, with their difficulties estimated jointly using response evidence from other questions and dialogues. The relative-order regularization further constrains the preceding subtask in a scaffolding pair to be harder than the scaffolded follow-up, resulting in predicted difficulties of 0.487 for Subtask 1 and 0.256 for Subtask 2. Because the original question requires the same mixed-number division demand together with additional operations, it should be more difficult than Subtask 1. The Diff SFT estimate of 0.483 is therefore too low, whereas URF-QD predicts a difficulty of 0.573, reducing the absolute error to 0.052. Overall, this example shows how dialogue-subtask supervision and relative-order regularization help calibrate the difficulty of complex questions by providing fine-grained evidence about their constituent mathematical demands.

<table><tr><td>Dataset</td><td>Method</td><td>AUC↑</td><td>Acc ↑</td><td>F1↑</td></tr><tr><td rowspan="6">Eedi</td><td>DKT</td><td> $0 . 5 7 4 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 5 4 4 _ { \pm 0 . 0 1 2 }$ </td><td> $0 . 5 6 0 { \scriptstyle \pm 0 . 0 1 3 }$ </td></tr><tr><td>DKT (Joint)</td><td> $0 . 5 8 0 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 5 5 7 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 5 7 2 { \scriptstyle \pm 0 . 0 1 8 }$ </td></tr><tr><td>simpleKT</td><td> $0 . 5 9 5 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 5 5 8 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 5 7 1 { \scriptstyle \pm 0 . 0 2 9 }$ </td></tr><tr><td>AKT</td><td> $\underline { { 0 . 6 0 3 } } _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 5 7 0 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $\underline { { 0 . 5 7 6 } } _ { \pm 0 . 0 2 5 }$ </td></tr><tr><td>URF-QD (Orig. Question)</td><td> $0 . 5 9 9 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $\underline { { 0 . 5 7 2 } } _ { \pm 0 . 0 1 3 }$ </td><td> $0 . 5 7 5 { \scriptstyle \pm 0 . 0 2 6 }$ </td></tr><tr><td>URF-QD (Joint + Reg.)</td><td> $\underline { { \mathbf { 0 . 6 3 7 } _ { \pm 0 . 0 2 0 } ^ { * } } }$ </td><td> $\underline { { \mathbf { 0 . 5 9 8 _ { \pm 0 . 0 1 6 } ^ { * } } } }$ </td><td> $\mathbf { 0 . 5 8 0 _ { \pm 0 . 0 2 4 } }$ </td></tr><tr><td rowspan="6">Khan Academy</td><td>DKT</td><td> $0 . 5 6 8 _ { \pm 0 . 0 2 3 }$ </td><td> $0 . 5 9 0 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 6 9 1 _ { \pm 0 . 0 2 3 }$ </td></tr><tr><td>DKT (Joint)</td><td> $0 . 5 7 1 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 5 9 8 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 6 9 7 { \scriptstyle \pm 0 . 0 2 1 }$ </td></tr><tr><td>simpleKT</td><td> $0 . 5 8 6 _ { \pm 0 . 0 3 0 }$ </td><td> $0 . 6 6 3 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 7 7 5 { \scriptstyle \pm 0 . 0 1 5 }$ </td></tr><tr><td>AKT</td><td> $\underline { { 0 . 5 8 9 } } _ { \pm 0 . 0 2 1 }$ </td><td> $0 . 6 2 6 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 2 4 }$ </td></tr><tr><td>URF-QD (Orig. Question)</td><td> $0 . 5 8 7 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $\underline { { 0 . 6 6 5 } } _ { \pm 0 . 0 2 1 }$ </td><td> $\underline { { 0 . 7 8 6 } } _ { \pm 0 . 0 2 2 }$ </td></tr><tr><td>URF-QD (Joint + Reg.)</td><td> $\mathbf { 0 . 5 9 8 _ { \pm 0 . 0 1 9 } }$ </td><td> $\mathbf { 0 . 6 7 6 _ { \pm 0 . 0 2 2 } }$ </td><td> $\mathbf { 0 . 7 9 8 _ { \pm 0 . 0 1 9 } }$ </td></tr></table>

Table 2: Performance comparison of methods on response correctness prediction across datasets. Best performance is bolded and second best is underlined. <sup>∗</sup> denotes statistically significant improvement over all baselines $( p < 0 . 0 5 )$ .

<table><tr><td colspan="2">Question:  $\begin{array} { r } { 1 \frac { 1 } { 3 } \div 1 \frac { 1 } { 4 } = 1 \frac { 1 } { \star } } \end{array}$  . What should replace the star? Options: (A) 15 (B) 12 (C) 16 (D) 7</td></tr><tr><td>Ground-truth difficulty</td><td>0.625</td></tr><tr><td>Difficulty source Diff  $\mathrm { S F } \dot { \mathrm { T } } \left( \mathrm { Q w e n } 3 . 5 { - } 4 \mathrm { B } \right)$  URF-QD (Joint + Reg.)</td><td>Pred. Difficulty 0.483 0.573</td></tr><tr><td>Related training dialogue-subtasks</td><td></td></tr><tr><td>Subtask 1: How to solve  $2 { \frac { 2 } { 3 } } \div 2 ?$  Subtask 2: Convert  $2 { \frac { 2 } { 3 } }$  to an improper frac-</td><td>0.487 0.256</td></tr><tr><td>tion (scaffold)</td><td></td></tr><tr><td>Subtask 3: Calculate  ${ \begin{array} { l } { { \frac { 1 } { 6 } } \ { \div } \ { \frac { 5 } { 8 } } } \end{array} }$ </td><td>0.399</td></tr><tr><td>Subtask 4: Replace the star in  $\begin{array} { r } { 2 \frac { \star } { 5 } = \frac { 1 3 } { 5 } } \end{array}$ </td><td>0.362</td></tr></table>

Table 3: Qualitative example of dialogue-subtask supervision supporting difficulty prediction for an unseen original question. Compared with the strongest baseline, URF-QD assigns a difficulty closer to ground truth.

## 6 Human Evaluation

We conduct two human evaluations on the Eedi dataset with three PhD student researchers with math tutoring experience to assess the reliability of tutor move annotations and the quality of subtask rewriting. GPT-5.4 achieves a macro-F1 of 0.754 for tutor move annotation, while 85.5% of rewritten subtasks satisfy both dialogue-independent completeness and mathematical-demand preservation. We provide additional details in Appendix E.

## 7 Conclusions and Future Work

In this paper, we presented URF-QD, a framework for jointly modeling original questions and tutorposed subtasks on a shared IRT scale. The key is to transform dialogue subtasks into standalone questions and leverage pedagogical relations to regularize their difficulty estimates. Through experiments on two real-world math learning datasets, we show that URF-QD improves both question difficulty prediction and response correctness prediction. In Appendix F, we showcase how our framework opens up the possibility of many downstream analyses, benchmarking an LLM tutor’s ability to pose tasks at appropriate difficulty levels that maximize student learning outcomes.

There are several avenues for future work. First, the shared dynamic scale could support more analysis into the strategy and behavior of LLM tutors. Future work should also use the extracted insights to guide adaptive subtask posing for both human and LLM tutors. Second, future work should extend URF-QD to incorporate visual elements, such as images and diagrams, that commonly appear in mathematics questions. Finally, future work should examine whether the framework generalizes to other educational domains and settings.

## Limitations

There are several limitations to our work. First, although we conduct experiments on two real-world math learning datasets, the Khan Academy dataset is not publicly available, and only a subset of the Eedi dataset is publicly accessible. To the best of our knowledge, there are no other readily available datasets containing both student questionanswering records and corresponding tutoring dialogues at the granularity required by our framework, making broader evaluation challenging. Second, since neither dataset provides ground-truth question difficulty labels, we use difficulty estimates obtained by fitting an IRT model to observed student responses as reference labels. These estimates depend on the assumptions and quality of the fitted IRT model and may not perfectly reflect the true difficulty of each question. Finally, we restrict our framework to the 1PL IRT model and binary response correctness labels, which may not fully capture the complexity of student knowledge and mathematical responses. Extending our framework to more expressive IRT models and finer-grained response evaluations is an important direction for future work.

## Ethical Considerations

There are several potential societal benefits to our work. By jointly modeling student responses to original questions and tutoring dialogues, our framework can provide more fine-grained estimates of student knowledge and question difficulty, potentially supporting more accurate educational assessment and personalized learning. In particular, leveraging intermediate mathematical subtasks in tutoring dialogues may help educators better understand students’ learning needs and provide more targeted tutoring support. There are also several potential risks. Inaccurate estimates of student ability or question difficulty could lead to inappropriate learning recommendations or unfair assessments. Additionally, our framework relies on LLM-based annotations and transformations of tutoring dialogues, which may introduce errors or biases into downstream predictions. Such biases could disproportionately affect students from underrepresented populations or with different linguistic backgrounds. Moreover, demographic information is unavailable in our datasets, limiting our ability to evaluate potential disparities across student populations. We emphasize that our framework is intended to support, rather than replace, educators’ judgments, and further evaluation of fairness and downstream learning outcomes is necessary before deployment in real-world educational settings.

## References

Samah AlKhuzaey, Floriana Grasso, Terry R. Payne, and Valentina Tamma. 2024. Text-based question difficulty prediction: A systematic review of automatic approaches. International Journal ofArtificial Intelligence in Education, 34(3):862–914.

Luca Benedetto. 2023. A quantitative study of nlp approaches to question difficulty estimation. Preprint, arXiv:2305.10236.

Luca Benedetto, Giovanni Aradelli, Paolo Cremonesi, Andrea Cappelli, Andrea Giussani, and Roberto Turrin. 2021. On the application of transformers for estimating the difficulty of multiple-choice questions from text. In Proceedings ofthe 16th Workshop on Innovative Use ofNLPfor Building Educational Applications, pages 147–157, Online. Association for Computational Linguistics.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. Preprint, arXiv:2005.14165.

Carnegie Learning. 2024. Livehint overview. Online: https://support.carnegielearning. com/help-center/math/livehint/article/ livehint-overview/.

Seth Chaiklin et al. 2003. The zone of proximal development in vygotsky’s analysis of learning and instruction. Vygotsky’s educational theory in cultural context, 1(2):39–64.

Shi-Ting Chen and Jinsong Chen. 2026. From text to parameters: Predicting item parameters from embedding regularization with reliability and design ceilings. Preprint, arXiv:2607.07141.

Albert T Corbett and John R Anderson. 1994. Knowledge tracing: Modeling the acquisition of procedural knowledge. User modeling and user-adapted interaction, 4(4):253–278.

Eedi. 2024. Help all your students succeed in math. Online: https://eedi.com/us.

Aritra Ghosh, Neil Heffernan, and Andrew S. Lan. 2020. Context-aware attentive knowledge tracing. Preprint, arXiv:2007.12324.

Sebastian Gombert, Lukas Menzel, Daniele Di Mitri, and Hendrik Drachsler. 2024. Predicting item difficulty and item response time with scalar-mixed transformer encoder models and rational network regression heads. In Proceedings ofthe 19th Workshop on Innovative Use ofNLPfor Building Educational Applications (BEA 2024), pages 483–492, Mexico City, Mexico. Association for Computational Linguistics.

Ronald K Hambleton and Hariharan Swaminathan. 2013. Item response theory: Principles and applications. Springer Science & Business Media.

Shuyan Huang, Alexander Scarlatos, Jaewook Lee, and Andrew Lan. 2026. Interpretable difficultyaware knowledge tracing in tutor-student dialogues. Preprint, arXiv:2605.01097.

Zhenya Huang, Qi Liu, Enhong Chen, Hongke Zhao, Mingyong Gao, Si Wei, Yu Su, and Guoping Hu. 2017. Question difficulty prediction for reading problems in standard tests. In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, AAAI’17, page 1352–1359. AAAI Press.

Radhika Kapoor, Sang T. Truong, Nick Haber, Maria Araceli Ruiz-Primo, and Benjamin W. Domingue. 2026. Prediction of item difficulty for reading comprehension items by creation of annotated item repository. Preprint, arXiv:2502.20663.

Khan Academy. 2023. Supercharge your teaching experience with khanmigo. Online: https://www. khanmigo.ai/.

Ekaterina Kochmar, Kaushal Kumar Maurya, Kseniia Petukhova, KV Aditya Srivatsa, Anaïs Tack, and Justin Vasselli. 2025. Findings of the BEA 2025 shared task on pedagogical ability assessment of AIpowered tutors. In Proceedings ofthe 20th Workshop on Innovative Use ofNLPfor Building Educational Applications (BEA 2025), pages 1011–1033, Vienna, Austria. Association for Computational Linguistics.

Zitao Liu, Qiongqiong Liu, Jiahao Chen, Shuyan Huang, and Weiqi Luo. 2023. simplekt: A simple but toughto-beat baseline for knowledge tracing. Preprint, arXiv:2302.06881.

Jakub Macina, Nico Daheim, Sankalan Chowdhury, Tanmay Sinha, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. 2023. MathDial: A dialogue tutoring dataset with rich pedagogical properties grounded in math reasoning problems. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pages 5602–5621, Singapore. Association for Computational Linguistics.

Jakub Macina, Nico Daheim, Ido Hakimi, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. 2025. Mathtutorbench: A benchmark for measuring open-ended pedagogical capabilities of llm tutors. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 204–221.

OpenAI. 2026. Introducing GPT-5.4.

Kseniia Petukhova and Ekaterina Kochmar. 2025. Intent matters: Enhancing AI tutoring with fine-grained pedagogical intent annotation. In Proceedings ofthe 20th Workshop on Innovative Use ofNLPfor Building Educational Applications (BEA 2025), pages 860– 872, Vienna, Austria. Association for Computational Linguistics.

Chris Piech, Jonathan Spencer, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas Guibas, and Jascha Sohl-Dickstein. 2015. Deep knowledge tracing. Preprint, arXiv:1506.05908.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

G. Rasch. 1960. Probabilistic Modelsfor Some Intelligence and Attainment Tests. Studies in mathematical psychology. Danmarks Paedagogiske Institut.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. Preprint, arXiv:1908.10084.

Alexander Scarlatos, Ryan S. Baker, and Andrew Lan. 2025a. Exploring knowledge tracing in tutor-student dialogues using llms. In Proceedings ofthe 15th International Learning Analytics and Knowledge Conference, LAK 2025, page 249–259. ACM.

Alexander Scarlatos, Nigel Fernandez, Christopher Ormerod, Susan Lottridge, and Andrew Lan. 2025b. SMART: Simulated students aligned with item response theory for question difficulty prediction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25071–25094, Suzhou, China. Association for Computational Linguistics.

Alexander Scarlatos, Jaewook Lee, Simon Woodhead, and Andrew Lan. 2026. Simulated students in tutoring dialogues: Substance or illusion? In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 42349–42385, San Diego, California, United States. Association for Computational Linguistics.

Howard Wainer, Neil J Dorans, Ronald Flaugher, Bert F Green, and Robert J Mislevy. 2000. Computerized adaptive testing: A primer. Routledge.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo Poli. 2024. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. Preprint, arXiv:2412.13663.

Chun-Kit Yeung. 2019. Deep-irt: Make deep learning based knowledge tracing explainable using item response theory. Preprint, arXiv:1904.11738.

Eric Zelikman, Wanjing Ma, Jasmine Tran, Diyi Yang, Jason Yeatman, and Nick Haber. 2023. Generating and evaluating tests for k-12 students with language model simulations: A case study on sentence reading efficiency. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2190–2205, Singapore. Association for Computational Linguistics.

Matthew Zent, Digory Smith, and Simon Woodhead. 2025. PIIvot: A lightweight NLP anonymization framework for question-anchored tutoring dialogues. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 27479–27488, Suzhou, China. Association for Computational Linguistics.

Albert Zhang, Alexis Ross, Kajal Patel, Julian Bernardo, Rebecca Bowie, Ana Trindade Ribeiro, Daniel Halper, Haripriya Valayaputtur, Jacob Andreas, Susanna Loeb, Li Lucy, Kyle Lo, and Ryan Knight. 2026. Tutormoments-preview: When help is unhelpful — evaluating ai tutors for productive struggle.

## A Related Work

## A.1 Question Difficulty Prediction and Student Response Modeling

IRT estimates student ability and item difficulty from observed responses (Rasch, 1960), but requires response data to calibrate unseen questions. Text-based difficulty prediction addresses this coldstart setting using linguistic features and pretrained language models (Huang et al., 2017; Benedetto et al., 2021; Gombert et al., 2024; AlKhuzaey et al., 2024). Recent work also predicts IRT parameters from item embeddings (Chen and Chen, 2026). Another direction simulates student responses with language models (Zelikman et al., 2023), including SMART, which aligns simulated students with IRT abilities and estimates difficulty from scored synthetic responses (Scarlatos et al., 2025b). Unlike these item-centric methods, our framework leverages real student responses to tutor-posed mathematical subtasks as additional supervision for difficulty prediction, allowing dialogue evidence to inform difficulty estimates for unseen original questions.

KT models students’ evolving knowledge from interaction histories, ranging from Bayesian KT (Corbett and Anderson, 1994) and DKT (Piech et al., 2015) to attention-based models such as AKT and simpleKT (Ghosh et al., 2020; Liu et al., 2023). Deep-IRT further combines neural KT with interpretable IRT parameters (Yeung, 2019). These methods primarily model predefined exercise-response sequences, whereas our framework incorporates dialogue-derived subtasks as additional response observations and tracks students’ evolving abilities across both original questions and tutoring dialogues on a shared IRT scale.

## A.2 Student Modeling in Tutoring Dialogues

Tutoring dialogues reveal student understanding through intermediate mathematical tasks and adaptive feedback. Datasets such as MathDial (Macina et al., 2023) and QATD-2k (Zent et al., 2025) support research on tutoring strategies and pedagogical feedback (Petukhova and Kochmar, 2025; Kochmar et al., 2025). More directly, Scarlatos et al. (2025a) explore KT in tutoring dialogues, and Huang et al. (2026) incorporate tutor-turn difficulty and IRT into conversational KT. However, these approaches focus on predicting dialogue responses rather than jointly modeling original-question and subtask difficulties to predict the difficulty of unseen questions.

Tutors also adjust task demands by scaffolding after struggle or pushing for rigor after success (Zhang et al., 2026). We use these pedagogical transitions to constrain the relative difficulty of consecutive subtasks. Together with standalone subtask rewriting, this enables tutoring dialogues to provide supervision for both question difficulty and student ability estimation.

## B Additional Dataset Details

We provide additional details on the two datasets. Both datasets are fully anonymized, so demographic information about individual students and tutors is not available. The Eedi dataset contains multiple-choice math question records and tutoring dialogues between real students and human tutors. After processing, the dataset contains 4, 175 subtask instances across 1, 775 dialogues. Each question is associated with a set of “subjects”, which we use to define the knowledge components (KCs). We assign each dialogue the same KCs as its associated question. Among the subtask instances, 467 correspond to scaffold after student struggle and 48 correspond to push for rigor. The Khan Academy dataset contains 4, 996 subtask instances across 1, 800 dialogues. Each question is associated with a list of “skills”, which we similarly use to define the KCs for the corresponding dialogue. The dataset contains 549 cases of scaffold after student struggle and 21 cases of push for rigor. The questions from both datasets span a broad range of mathematical topics, including Algebra, Geometry, and Number.

## C Additional Experimental Details

We provide additional details on the baseline methods and experimental implementation for reproducibility.

## C.1 Baselines

Difficulty Prediction Baselines Following prior work (AlKhuzaey et al., 2024; Benedetto, 2023; Kapoor et al., 2026), we adopt a strong fine-tuning baseline, Diff SFT, which fine-tunes a recent and capable encoder-based language model, specifi cally ModernBERT-base (Warner et al., 2024). We use a regression setup, minimizing MSE, to directly predict question difficulties given the question’s text and metadata. We additionally train Diff SFT (Qwen3.5-4B) using the same question encoder as our framework. We then report a Random Forest baseline, which is common for difficulty predic tion (AlKhuzaey et al., 2024; Gombert et al., 2024), where we encode the item’s question text and meta data using SBERT model (Reimers and Gurevych, 2019), and use the resulting embeddings as input features to fit a random forest regressor to item dif ficulty. We also use a simple yet strong baseline, kNN Mean, that retrieves the k questions in the train set with the highest SBERT embedding co sine similarity, and takes their mean difficulty as the predicted test item difficulty. We search over $k \in [ 1 , 5 ]$ and select the best performing value on the validation set, selecting $k = 4$ . Finally, we adapt an LLM prompting baseline leveraging in context learning (ICL) (Brown et al., 2020), ICL kNN, where we prompt GPT-5.4 with k in-context examples of questions retrieved from the train set using k-Nearest Neighbors (kNN). We use the same SBERT model as above, and perform a similar hyperparameter search, selecting k = 4. We provide the prompt details in Appendix G. All these baselines use only original questions and their reference difficulties, without dialogue-subtask response supervision or difficulty ordering regularization.

Response Prediction Baselines For student response correctness prediction, we compare against three KT baselines: DKT (Piech et al., 2015), a simple recurrent neural network-based KT method, and DKT (Joint), which is trained on the combined chronological sequence of original question and dialogue subtask responses; AKT (Ghosh et al., 2020), a full attention-based KT method; and simpleKT (Liu et al., 2023) a simple but robust KT method. Unless marked as Joint, these baselines are trained only on original question responses. We evaluate all methods on the same held-out originalquestion response events. Dialogue-subtask responses are incorporated as additional training observations and as history for the joint models.

We additionally report URF-QD (Orig. Question), an original question-only variant of our response prediction model. It retains the same textbased difficulty estimator, sequential student state model, and IRT response function as our full framework, but excludes all dialogue subtask interactions and the difficulty ordering regularizer. This baseline isolates the contribution of dialogue evidence within the proposed architecture.

## C.2 Implementation Details

We use Qwen3.5-4B (Qwen Team, 2026) as the frozen LLM encoder for our framework and multiqa-mpnet-base-dot-v1 SBERT model for baselines and ablation studies. We use GPT-5.4 (OpenAI, 2026) for the prompting-based baseline (ICL kNN), mathematical subtask identification, subtask correctness labeling, standalone-question transformation, and pedagogical move annotation. We adopt the LLM-based correctness annotation protocol of Scarlatos et al. (2026), which demonstrated substantial agreement with expert human annotations (Cohen’s $\kappa = 0 . 7 2 2 )$ , comparable to human interannotator agreement $( \kappa = 0 . 7 1 9 )$ , providing empirical support for its reliability.

For question difficulty prediction, we obtain 2560-dimensional question embeddings by mean pooling over the final-layer hidden states of the frozen LLM encoder. We use a three-layer MLP and GELU activations to predict scalar question difficulty. We train the difficulty predictor using AdamW with a fixed learning rate of 5 · 10<sup>−3</sup>, fullbatch optimization, and gradient norm clipping at 10.0 for 120 epochs. For the ICL (kNN) baseline, we ask the model to first reason about the question’s difficulty in a chain-of-thought manner before generating a numeric difficulty prediction.

To obtain reference student abilities and question difficulties, we fit a 1PL IRT model using observed student responses to original questions. We fit the reference IRT model on all eligible originalquestion responses, without a train/validation/test split, including responses associated with questions subsequently assigned to validation and test sets. We implement IRT fitting using SciPy with the L-BFGS-B optimization algorithm, setting the maximum number of iterations to 500. We apply L2 regularization to both student ability and question difficulty parameters, with a regularization strength of 1.0 for each.

For response correctness prediction, we train an LSTM with a hidden-state dimension of 200 to model the evolving student state. We represent response correctness using a learned 32-dimensional embedding, which is concatenated with the 2560- dimensional question embedding to form a 2592- dimensional LSTM input. Before predicting the current response, we project the preceding LSTM hidden state to a scalar student ability using a two layer MLP and a GELU activation. We train the model using AdamW with a fixed learning rate of $1 \cdot 1 0 ^ { - 3 }$ , gradient norm clipping at 10.0, and a batch size of 64 for 10 epochs. For both prediction tasks, we set the difficulty-order regularization weight to $\lambda = 1$

We compute AUC using scikit-learn and Pearson and Spearman correlations using NumPy. To the best of our knowledge, the software and models used in our implementation either have open-source licenses or do not list a license. In addition, our use of these resources, including the OpenAI API, is consistent with their intended terms of use. We will ensure that the license and terms reflect the sources on which our implementation builds when we release code. We use ChatGPT to assist with writing and editing the manuscript.

## D Additional Quantitative Evaluation

## D.1 Additional Ablation Study

We analyze the contribution of each component of our framework on Eedi, which we choose as the representative dataset because its dialogues involve real human tutors and students, yielding more relative-order constraints and higher-quality tutoring interactions.

Question Difficulty Prediction. Table 4 reports the ablation results for question difficulty prediction. The original-question-only variant performs worse than the full model, showing that the decomposed subtasks provide additional information for estimating question difficulty. Removing the relative-order regularization also leads to consistent performance drops across all four metrics, suggesting that explicitly modeling relative difficulty helps calibrate question difficulty across the two task types when placing them on a shared scale. Using the original subtasks without rewriting generally outperforms the Diff SFT baseline, improving MSE, PCC, and SCC, although MAE is slightly worse; the full model further improves over this variant across all four metrics, indicating that rewriting produces more self-contained subtask descriptions that are more informative for difficulty prediction. Finally, replacing the Qwen3.5- 4B question encoder with a lightweight MPNetbased SBERT encoder leads to the largest performance drop, with MSE increasing from 0.350 to 0.398 and PCC decreasing from 0.458 to 0.341, suggesting that representations from the larger language model are more effective for question diffi-

<table><tr><td>Method MSE↓ MAE↓</td><td>PCC↑ SCC↑</td></tr><tr><td>URF-QD (Full) 0.350</td><td>0.473 0.458 0.447</td></tr><tr><td>Questions Only</td><td>0.382 0.482 0.430 0.414</td></tr><tr><td>w/o Reg.</td><td>0.367 0.481 0.447 0.437</td></tr><tr><td>w/o Rewrite</td><td>0.376 0.485 0.435 0.423</td></tr><tr><td>SBERT (MPNet)</td><td>0.398 0.503 0.341 0.336</td></tr></table>

Table 4: Ablation study on question difficulty prediction (Eedi). URF-QD(Full) is equivalent to URF-QD(Joint + Reg.). Questions Only excludes subtasks, which is equivalent to Diff SFT (Qwen3.5-4B); w/o Reg. keeps original question and dialogue subtasks but removes relative-order regularization; w/o Rewrite uses original tutor-posed subtasks instead of standalone rewrites; SBERT (MPNet) replaces Qwen3.5-4B encoder with a lightweight MPNet-based sentence encoder.

<table><tr><td>Method AUC↑</td><td>Acc↑ F1↑</td></tr><tr><td>URF-QD (Full) Questions Only w/o Order Reg.</td><td>0.637 0.598 0.580 0.599 0.572 0.575 0.640 0.597 0.586</td></tr><tr><td>w/o Rewrite SBERT (MPNet)</td><td>0.623 0.5800.586 0.625 0.5900.573</td></tr><tr><td>simpleKT Backbone</td><td>0.639 0.5900.583</td></tr></table>

Table 5: Ablation study on response correctness prediction (Eedi). URF-QD(Full) is equivalent to URF-QD(Joint + Reg.). Questions Only excludes subtasks, which is equivalent to URF-QD (Orig. Question); w/o Reg. removes relative-order regularization; w/o Rewrite uses original tutor-posed subtasks; SBERT (MPNet) replaces Qwen3.5-4B encoder with a lightweight MPNetbased sentence encoder; and simpleKT Backbone replaces the default DKT-based student-state model.

culty prediction.

Response Correctness Prediction. Table 5 reports the ablation results for response correctness prediction. Removing the subtasks and using only the original question decreases AUC from 0.637 to 0.599 and accuracy from 0.598 to 0.572, supporting that decomposed subtasks provide useful information beyond the original question. Unlike for question difficulty prediction, removing the relative-order regularization does not hurt response prediction. One possible explanation is that the comparatively sparse ordering constraints are diluted by the large number of interaction records, suggesting that this regularization is more beneficial for difficulty estimation. Removing the rewriting step leads to performance drops in AUC and accuracy, while replacing the Qwen3.5-4B encoder with a lightweight SBERT encoder also generally reduces performance, highlighting the value of selfcontained subtask representations and richer semantic encodings. Finally, replacing DKT with simpleKT yields comparable results, indicating that URF-QD is not strongly dependent on a particular knowledge tracing model.

Encoder Fine-tuning. We also compare the default setup, which uses Qwen3.5-4B as a frozen encoder, with jointly fine-tuning the encoder and downstream prediction modules. Fine-tuning reduces performance on both tasks. For question difficulty prediction, MSE and MAE increase to 0.408 and 0.486, while PCC and SCC decrease to 0.385 and 0.359, respectively. For response correctness prediction, AUC, accuracy, and F1 decrease to 0.626, 0.589, and 0.552, respectively. Joint finetuning also substantially increases training time. In this setting, keeping Qwen3.5-4B frozen is both more effective and more efficient.

## D.2 Impact of Relative-Order Regularization

We further analyze the effect of the relative-order regularization strength λ. For question difficulty prediction, λ = 0 is equivalent to removing the relative-order regularization. As shown in Figure 3a, $\lambda = 0 . 1$ performs similarly to $\lambda = 0$ , suggesting that a weak regularization signal has limited effect. Performance improves as λ increases to 0.5 and 1, while λ = 1 and λ = 3 achieve comparable performance across the four metrics. In contrast, increasing λ to 10 leads to a clear performance drop. A possible reason is that overly strong regularization causes the training objective to focus excessively on relative difficulty ordering rather than accurately fitting the absolute difficulty labels.

For response correctness prediction, Figure 3b shows that performance remains relatively stable across $\lambda \in \{ 0 , 0 . 1 , 0 . 5 , 1 , 3 \}$ , suggesting that response prediction is not sensitive to regularization weight within a moderate range. In contrast, setting λ = 10 substantially hurts all three metrics, likely because overly strong regularization constrains the model at the expense of optimizing the responseprediction objective.

To further examine how the regularization affects learned question difficulties, we evaluate the satisfaction rate of the pedagogically implied difficulty orderings. Specifically, we calculate the proportion of subtask pairs satisfying $b _ { j , k } < b _ { j , k - 1 }$ for scaffolding after struggle $( \Omega _ { 1 } )$ and $b _ { j , k } > b _ { j , k - 1 }$ for pushing for rigor (Ω<sub>2</sub>).

As shown in Table 6, the relative-order regularization substantially improves the consistency between predicted difficulties and pedagogical ordering constraints. Without regularization $( \lambda = 0 )$ the overall satisfaction rate is only 0.699. Increasing λ to 1 raises this rate to 0.899, with improvements for both scaffolding after struggle (0.697 to 0.898) and pushing for rigor (0.756 to 0.911). This result indicates that the regularization effectively encourages the difficulty predictor to capture the relative mathematical demands reflected in tutors pedagogical moves.

While increasing λ further improves order satisfaction, reaching 0.914 at λ = 10, it leads to worse difficulty prediction performance, suggesting that excessive regularization may compromise absolute difficulty estimation. Overall, these results demonstrate that our proposed relative-order regularization effectively aligns learned task difficulties with the pedagogically implied difficulty orderings in tutoring dialogues, providing better-calibrated difficulty estimates that benefit downstream predictions when applied at a moderate strength.

## E Human Evaluation Details

We conduct two human evaluations on the Eedi dataset to assess our tutor move annotations and subtask rewriting quality. Three PhD student researchers with mathematics tutoring experience are recruited to conduct the human evaluations. All participants voluntarily agreed to participate and received no financial compensation. Participation was free of coercion. They are informed of the evaluation procedures and voluntarily agreed to participate before conducting the annotations; additional demographic and geographic information is not collected.

![](images/addbaaa12732aebf54a5d4b8595fa8336dba6c34e3086aa6719b1bd484793438.jpg)

![](images/3b25249ed4c630e8f0a5ef454707d41efd52ccf5b4ad7c1b2e43041a97d8298a.jpg)

![](images/9ee3bc91123e9f0a3402cd3fa559bc9a864f7df68a393c65595ebcfa41129684.jpg)

![](images/3d35f2070cbc72f2fa8d917f4372223d3aa8d4037df61af5aff57ba0293a2f39.jpg)

(a) Question difficulty prediction.  
![](images/61c7c2737180724af5cf953096b73dcfe5c7061df02256564ad6f6592f03b703.jpg)

![](images/6ba181f61a2a8dd7d90b9319d041ffd9b2fdd9afe6064bd8da6724e9ef748712.jpg)

![](images/e57fe00bbd477900c0a60622f978b85fa53f97850829f8ce6b30e0f1baeff4e3.jpg)  
(b) Response correctness prediction.

Figure 3: Impact of the relative-order regularization strength λ on the two prediction targets.
<table><tr><td>λ</td><td> $\Omega _ { 1 }$   $\mathrm { S a t . }$ </td><td> $\Omega _ { 2 }$  Sat.</td><td>Overall Sat.</td></tr><tr><td>0</td><td>0.697</td><td>0.756</td><td>0.699</td></tr><tr><td>0.1</td><td>0.727</td><td>0.711</td><td>0.726</td></tr><tr><td>0.5</td><td>0.834</td><td>0.866</td><td>0.835</td></tr><tr><td>1 3</td><td>0.898</td><td>0.911</td><td>0.899</td></tr><tr><td></td><td>0.910</td><td>0.911</td><td>0.910</td></tr><tr><td>10</td><td>0.912</td><td>0.956</td><td>0.914</td></tr></table>

Table 6: Difficulty order satisfaction rates under different relative-order regularization strengths λ. $\Omega _ { 1 }$ and $\Omega _ { 2 }$ denote scaffolding after struggle and pushing for rigor, respectively.

First, annotators are asked to label consecutive subtask pairs in 25 randomly sampled dialogues (204 turns) as scaffolding after struggle, pushing for rigor, or other tutor moves. Interannotator agreement, in Fleiss’s $\kappa ,$ is $\kappa = 0 . 6 5 1$ for scaffolding-after-struggle labels and 0.591 for push-for-rigor labels, showing moderate-to-good agreement. Against human annotations, GPT-5.4 achieves a macro-averaged positive-class F1 of 0.754 across the two target pedagogical moves, with F1 scores of 0.833 for scaffolding after struggle and 0.674 for pushing for rigor. In Eedi, pushing for rigor is considerably less frequent than scaffolding after struggle (48 vs. 467 labeled pairs). This class imbalance may partly contribute to the lower annotation agreement and F1 score for pushing for rigor.

Second, annotators are asked to evaluate 110 rewritten subtasks for dialogue-independent completeness, i.e., the rewritten question being selfcontained and solvable without dialogue context and mathematical-demand preservation, i.e., the rewritten question retaining the original subtask’s mathematical requirements. Annotators judge 92.7% and 90.0% of subtasks as satisfying the respective criteria, with 85.5% receiving positive judgments on both criteria. Together, these evaluations support the reliability of LLM-based tutor move annotation and subtask rewriting.

## F LLM Tutor Benchmarking

We now provide an example of how URF-QD opens door for many learning analytics tasks. In this preliminary example, we investigate how the zone of proximal development (Chaiklin et al., 2003) takes effect in tutoring dialogues: by studying the relationship between tutor-posed task difficulty and student learning outcomes, we can study what types of tasks during scaffolding are the most beneficial. Using these findings, we can create a benchmark to evaluate the pedagogical capabilities of LLMs, in terms of posing scaffolding tasks at the right difficulty level to facilitate student learning. Such a benchmark would complement existing benchmarks such as MathTutorBench (Macina et al., 2025) and TutorMoments (Zhang et al., 2026), which do not use principled psychometric models and only rely on LLM-as-a-judge.

Task difficulty and next question correctness We revisit the Eedi dialogues using the previously fitted, five-fold student-disjoint dynamic 1PL model with a LSTM and frozen text encoder. For each dialogue, for all tutor-posed subtasks, we compute the average gap in the estimated student ability θ and the turn difficulty b. The learning outcome measure we study is the correctness of the student’s response on the first original question response after the dialogue’s recorded start, which shares at least one KC label with the dialogue and does not have an associated dialogue. This filtering yields 553 dialogues from 327 students and 514 next question responses. We perform a simple linear fit, shown in Figure 4, which reveals a positive relationship between next question response outcomes and θ − b, with slope 0.613 and a p-value of 0.001. Therefore, empirical data suggests that larger gaps, corresponding to easier subtasks relative to the student’s ability, are associated with higher subsequent response correctness.

Human versus LLM tutor tasks With this finding, we now perform a comparison between human tutors and LLM tutors to examine how they tend to pose subtasks during the dialogue. For each subtask, we prompt GPT-5.4 with the original problem and preceding dialogue to generate an appropriate next tutoring task and a standalone rewrite. We then estimate the task’s difficulty with the trained difficulty encoder. This analysis results in 2,994 paired human tutor-LLM tutor subtask comparisons across 916 dialogues. Figure 5 visualizes average turn-averaged θ − b for the 21 human tutors with at least 20 such comparisons done. We have an interesting finding: all but one human tutor poses easier subtasks than the LLM tutor, on average. Substituting this result into the analysis above, this comparison yields an advantage for human tutor vs. GPT-5.4 of 1.11 percentage points, with a p-value of 0.006. We leave the investigation of all LLMs and different pedagogical prompts to future work.

Potential usage This simple analysis shows that our framework URF-QD enables many detailed analysis into tutoring strategies; by associating each tutor-posed task with an IRT difficulty, we can leverage adaptive testing (Wainer et al., 2000) techniques to study the tradeoff between tutors posing tasks to pinpoint the student’s error vs. delivering targeted feedback. More importantly, we will be able to build the world’s first outcome-based tutoring benchmark, which explicitly relates LLMs’ tutoring behavior with student learning outcomes.

![](images/2ee440d9bca5ba8011c151420e3594f027f243ecd3c2c234b7bcf444b38d7341.jpg)  
Figure 4: Next question response correctness vs. mean student ability–tutor-posed subtask difficulty gap. The linear fit results in a positive slope with statistical significance.

![](images/9246052745335e649482a85f878bc293e83d551485d105a69b82ef4e704657bd.jpg)  
Figure 5: Mean student ability–tutor-posed subtask difficulty gap, where GPT-5.4 poses subtasks that are more difficult than all but one human tutor.

## G Prompts

## G.1 Prompt for Subtask Identification

We show the prompt used for mathematical subtask identification in dialogues in Table 7.

## G.2 Standalone-Question Transformation

We show the prompt used for standalone question transformation in Table 8. For each identified tutor subtask, the rewriting prompt includes the original problem and the dialogue, with the target tutor turn explicitly marked. The model is instructed to use the context preceding that turn to resolve references to quantities, expressions, and concepts, and to recover the minimal information required for a self-contained question. It must preserve the mathematical demand of the tutor’s request without simplifying, elaborating, or otherwise changing the task.

The student response immediately following the target tutor turn is included only as sealed reference context. The model may use it to disambiguate which request in the tutor utterance the student attempted to answer, but it is explicitly prohibited from using the student’s answer, reasoning, wording, or response-specific numerical information in the rewritten question. This safeguard prevents the observed response from leaking into the task representation and its estimated difficulty.

## G.3 Pedagogical Move Annotation

We show the prompts used to annotate the pedagogical moves scaffold after struggle and push for rigor in Table 9 and Table 10, respectively.

## G.4 Prompt for Model

We show the prompt used for ICL(kNN) baseline with chain-of-thought reasoning in Table 11.

![](images/d0d2e6a0bd6f8ee5cd2e65ac428fb9f0e281bc4f22a2d2ba31c1ba3aa27070c8.jpg)  
Table 7: Prompt for Mathematical Subtask Identification

![](images/265f48e59d18808c91e0bf800a065bb83f4af9b7dfc23736f9b74e995fa76c5f.jpg)  
Table 8: Prompt for Standalone Mathematical Subtask Transformation

![](images/f4d8d295859297fd3aa0947d9015ba2cad01112be54710a659b8eeddc3488328.jpg)  
Table 9: Prompt for Scaffold After Struggle Annotation in Dialogue

## System Message:

You are a math teacher, and your task is to annotate relative task difficulty in mathematics tutoring dialogues.

The supplied candidate pairs have already passed three hard gates:

1. Both tutor turns are accepted mathematical subtasks.

2. the later tutor turn occurs immediately after the earlier task’s student response (there is no intervening raw dialogue turn); and

3. The student response to the earlier task is correct.

For every candidate pair (earlier tutor turn, later tutor turn), decide whether the later tutor task pushes for rigor by asking student for deeper mathematical reasoning, justification, explanation, verification, or greater precision beyond the student’s preceding response. Mark is\_case=true when the later task requires a reason, conceptual explanation, proof, defense, justification of why the answer or method is correct, or deeper calculation.

Mark false when the later task merely asks for the next solution step, a final answer, recall, identification, interpretation, selection of a method, confirmation, or a report of steps without a why/justification demand. A prompt such as "How did you get that?" is true only when context makes it a request for explanatory reasoning, not merely a procedural report. Despite the case name, do not require the later student response to be incorrect; its correctness is an outcome, not an inclusion gate.

## User prompt:

Determine if the tutor is pushing for rigor in two consequtive tutor turns in the dialogue: an subquestion is followed immediately in next tutor turn by a tutor task after the student’s correct response. For a true pair, the follow-up turn is asking student for deeper mathematical reasoning, justification, or explanation of previous turn so the fixed order is difficulty(later) > difficulty(earlier).

![](images/4fbd799d022eea43a3dbf82c38fd9fc38fcb7aeb9fcadab922539e3af36ce6e7.jpg)  
Table 10: Prompt for Push For Rigor Annotation in Dialogue

![](images/b137d910379c00506befa4330ea2aa879a52b59d05c5c755504531b89074b9de.jpg)  
Table 11: Prompt for ICL (kNN) difficulty prediction