# A Dataset for Modeling Iterative Problem-Solving

Fagun Patel<sup>1\*</sup> Sang T. Truong<sup>1\*</sup> Duc Q. Nguyen<sup>2\*</sup> Kazunori Fukuhara<sup>1</sup> Benjamin W. Domingue<sup>1†</sup> Sanmi Koyejo<sup>1†</sup> Nick Haber<sup>1†</sup>

<sup>1</sup>Stanford University <sup>2</sup>National University of Singapore

Co-first authors

## Abstract

Solving problems through repeated attempts is a sequential modeling task: at each step, the solver receives feedback and decides how to re vise their solutions. Predicting whether perfor mance improves, plateaus, or regresses across attempts is central to understanding any iter ative problem-solving process in both human learners and autonomous agents. Beyond out comes, modeling what errors persist and how strategies shift across attempts provides deeper insight into the mechanics of sequential learn ing. Studying these dynamics requires observing many solvers as they attempt, receive feedback, and revise. Programming courses with automated grading provide this setting, as students iteratively submit code to test suites and receive feedback on every attempt. We there fore curate CodeInsight, a large-scale dataset of over 3 million submissions from 3,286 under graduates across 2 introductory C++ courses in 2 academic years, with test-case-level out comes, timestamps, and source code. On this dataset, we build a benchmark that evaluates models spanning parametric, sequential, and generative traditions under a shared calibration and-scoring protocol, including a Recurrent State Space Model (RSSM) adapted to track solver characteristics through discrete latent variables and an LLM-based predictor that gen erates explicit solutions. The adapted RSSM achieves the strongest predictive accuracy on three of the four courses. The LLM predictor is less accurate but produces full submissions at each attempt, enabling direct analysis of failure modes. We find that the model’s coding pro ficiency is inversely related to predictive per formance in this setting, with the LLM better understood as a generative solver conditioned on context rather than a faithful predictor of solver behavior. We publicly release our code and the dataset on request to facilitate future research.<sup>1</sup>

## 1 Introduction

Problem-solving is inherently iterative: a solver receives feedback, revises their solutions, and tries again. The resulting trajectory of attempts encodes how their solutions evolve. Students submitting code against a test suite or LLMs solving a coding task both produce sequential records of iterative refinement (Fahid et al., 2021; Rivers et al., 2016). These trajectories are rich but noisy: a single-character change can flip a submission from incorrect to correct, and solvers may improve steadily, plateau for long stretches, or regress after apparent progress on a given task. Predicting these dynamics is a sequential modeling problem with practical applications from evaluating autonomous agents to adaptive instruction (Piech et al., 2015). Beyond predicting outcomes, modeling which errors persist and how strategies shift across attempts can inform personalized feedback, diagnose systematic failure modes, and support use cases such as routing model decisions on LLMs (Ong et al., 2025; Chen et al., 2024b).

Several families of models have been applied to predicting problem-solving trajectories. Parametric models such as Item Response Theory (IRT) (Embretson and Reise, 2000) estimate fixed solver and problem parameters to predict binary outcomes, but do not typically model how solvers’ skill levels evolve over attempts. Sequential prediction models track how solver performance changes but remain limited to binary correctness; Bayesian Knowledge Tracing (BKT) (Corbett and Anderson, 1994), a hidden Markov model, and Deep Knowledge Tracing (DKT) (Piech et al., 2015), a type of Recurrent Neural Network, are the primary examples in this domain. More recently, LLMs have been used to emulate problem-solving behavior (Markel et al., 2023), generating solutions (Scarlatos et al., 2025) and operating over richer representations that can capture failure modes directly from context. These approaches have developed independently, and no existing study compares them for next-step prediction under a shared evaluation framework.

Programming courses provide rich temporal data for such a comparison: students submit code repeatedly against automated test suites, receive immediate feedback, and generate longitudinal trajectories that span the full arc of iterative problemsolving (Messer et al., 2024; Blikstein et al., 2014). We curate CodeInsight, a large-scale dataset of 3,286 undergraduates learning C++ across 394 problems at a Vietnamese university (2022–2024), comprising over 3 million submissions with testcase-level correctness, timestamps, and source code. On this dataset we build a benchmark that evaluates models spanning parametric, sequential, and generative traditions under a shared calibrationand-scoring protocol. Among these, we adapt a Recurrent State Space Model (RSSM) with discrete latent variables that tracks solver characteristics over time, and an LLM-based predictor that generates the solutions based on students’ profiles. We find that RSSM achieves stronger predictive accuracy on three of the four courses by training directly on response patterns, while the LLM predictor falls below all trained models. Because the LLM generates complete solution attempts, we can diagnose this gap directly: its predictive signal is driven more by problem difficulty than by solver-specific behavior, and models with stronger coding ability predict worse. Our contributions are as follows:

• Dataset. We curate CodeInsight, a large-scale dataset of 3,286 undergraduates learning C++ across 394 problems at a Vietnamese university (2022–2024), comprising over 3 million code submissions with test-case-level correctness, timestamps, and source code, released under a departmental license upon request.

• Benchmarking Predictive Models. We introduce a benchmark for next-attempt prediction on sequential data that evaluates models spanning parametric, sequential, and generative traditions under a shared calibration-and-scoring protocol; to our knowledge, this is the first study to compare these modeling traditions under one evaluation framework. The RSSM achieves the highest AUC on three of the four courses, while the LLM predictor falls below the parametric models and the RSSM at nearly every attempt, with a predictive signal driven more by problem difficulty than solver-specific behavior.

## 2 Related Works

Programming trajectory datasets Several datasets capture student interactions in programming courses, though they differ in granularity. CodeWorkout (DataShop@CMU, 2021) provides ∼130K Java submissions from ∼820 students with source code, timestamps, and aggregate test-pass percentages, but not per-test-case outcomes. The Blackbox dataset (Brown et al., 2018) logged over 300M compilations and source code edits from 2.5M BlueJ users worldwide, but lacks assignment structure or correctness labels; the project has since been discontinued, and all data access will be revoked by August 2026. ACcoding (Chen et al., 2024a) provides 4M submissions across multiple languages from 27K students, but records a single overall judge verdict per submission. CodeInsight differs from these in providing per-test-case pass/fail labels across 3,888 individual test cases, full source code, and timestamps for over 3M submissions, enabling fine-grained analysis of how solvers fail on specific subtasks across attempts.

Modeling and Predicting Iterative Problem-Solving Modeling how problems get solved through iterative attempts has drawn on several traditions. In repetitive-practice settings, Reddy et al. (2016) propose a modified exponential forgetting curve and explore how task difficulty and the number of repetitions influence learning efficiency, validating these concepts through large-scale log data from a flashcard system. Mohr et al. (2018) show how performance dynamics are shaped by repetitive practice and feedback, illustrating how solvers gradually adjust their strategies based on the outcomes of prior attempts. Within item response theory, sequential IRT models describe repeated attempts at the same item (Culpepper, 2014), with later extensions to cognitive diagnosis and adaptive testing under repeated attempts (Hung and Huang, 2019; Lu and Cheng, 2026); our CIRT baseline follows this tradition by adding an attemptindexed difficulty slope to IRT. In programming specifically, recent work extends knowledge tracing models, which predict a solver’s next outcome from their interaction history (Shen et al., 2024), with richer code representations. Code-DKT (Shi et al., 2022) incorporates AST-based code representations via code2vec, and Temporal-ASTNN (Mao et al., 2021) combines AST-based neural networks with LSTMs to model temporal progression. TIK-TOC (Duan et al., 2025) predicts per-test-case outcomes rather than aggregate correctness. Together with OKT (Liu et al., 2022), they are able to generate student code via a language model conditioned on a latent state. However, no existing study compares code-aware sequential models, parametric models, and generative approaches under a shared evaluation framework, which our work addresses.

LLMs as simulated problem-solvers LLMs have been increasingly used to simulate learner behavior. Aher et al. (2023) introduced Turing Experiments to evaluate how well LLMs replicate human behavioral patterns. In programming education, Wu et al. (2025) proposed constructing cognitive prototypes via knowledge graphs to constrain LLM output toward realistic errors; Markel et al. (2023) developed GPTeach, a system that uses GPT-simulated students for training teaching assistants, and Miroyan et al. (2025) generated realistic student code by fine-tuning on timestamped submission histories. Ross and Andreas (2025) proposed a method for improving student error simulation through cycle-consistent reasoning between incorrect answers and latent misconceptions. On the prediction side, Neshaei et al. (2024) found that fine-tuned LLMs match but do not surpass sequential models on next-step prediction. Whether LLMs capture solver-specific failure modes or primarily reflect their own coding ability remains an open question that we address directly in this work.

## 3 Dynamic Measurement Models

We evaluate whether student performance on future problems can be predicted from prior submission history. We compare parametric and sequential baselines with two adapted approaches: an RSSM and an LLM-based predictor.

Baseline models We compare against six baselines spanning two families. The parametric models are IRT, a one-parameter logistic model with a static per-solver ability and a per-test-case difficulty (Embretson and Reise, 2000) (Appendix C.1), and CIRT, a temporal extension of IRT that adds an attempt-indexed difficulty slope (Appendix C.2). The sequential models are BKT, a hidden Markov model with a binary mastery state per test case (Corbett and Anderson, 1994) (Appendix C.3); DKT, a recurrent neural network over the interaction sequence (Piech et al., 2015) (Appendix C.4); Code-

DKT, a code-aware DKT variant that augments each interaction with an embedding of the submitted code (Shi et al., 2022) (Appendix C.5); and TIKTOC, a multi-task model that couples an LSTM knowledge state with a LoRA-finetuned language model to jointly predict per-test-case outcomes and generate the student’s next submission (Duan et al., 2025) (Appendix C.6). Training procedures and loss curves for all models are provided in Appendix C.

Recurrent State Space Model (RSSM) We introduce an RSSM that captures solver characteristics through discrete latent variables $z _ { t }$ at each timestep. At each timestep t, both the solver’s submitted code and the problem description are embedded using a pretrained LLM (Qwen3- Embedding (Zhang et al., 2025)) and projected through separate learned linear layers into a response encoding $e _ { t }$ and a problem encoding $q _ { t } .$ A GRU backbone (Cho et al., 2014) updates a hidden state $h _ { t } = f _ { \phi } ( h _ { t - 1 } , z _ { t - 1 } , q _ { t } )$ that captures the solver’s cumulative history. A representation model (posterior) $q _ { \phi } ( \boldsymbol { z } _ { t } \mid h _ { t } , \boldsymbol { e } _ { t } )$ infers discrete latent variables from the hidden state and the current response, while a transition predictor (prior) $p _ { \phi } ( z _ { t } \mid h _ { t } )$ predicts them from history alone. This posterior/prior separation follows the DreamerV2 framework (Hafner et al., 2021): during training, the posterior grounds $z _ { t }$ in the observed response, while at inference, the prior predicts solver characteristics before observing the next submission. A scoring head predicts per-test-case binary pass/- fail outcomes from $q _ { t } , h _ { t } ,$ and $z _ { t } ,$ , combined with a learned per-item difficulty bias. The model does not generate code; it predicts test-case-level correctness. An embedding predictor provides an auxiliary training signal by predicting the next response embedding from $z _ { t } .$ Full architecture details and the loss function are provided in Appendix C.7.

LLM-as-Predictor We investigate whether a pretrained LLM, conditioned on a student’s behavioral profile and submission history, can predict performance on future problems for held-out students. The LLM predictor follows the same train/test split and temporal protocol as the baselines. During testing, for each test-split student, the model receives a structured behavioral persona capturing submission pacing, precheck usage, topic-level pass rates, and code complexity. It also receives problemlevel statistics derived from students in the training set (pass rate, typical attempt count) and retrieved examples from the student’s own calibration history on similar problems, condensed into behavioral summaries by a smaller LLM. Appendix D describes how each of these inputs is computed and shows the full prompt template.

![](images/3f41fa24f7c5adedca1c711d088b0d286184e3d07b334f09292f612843bada85.jpg)  
Figure 1: The exercise platform interface based on Moodle. From left to right: the problem statement with function signature, description, example test case, and the code editor where the solution attempt is written; automated feedback on failing attempts, with compiler errors (red) and failed test cases showing expected versus actual output; and a revised submission where all test cases pass (green).

![](images/5836d552240c4ea297bcf22dd6f9c000299b4917ffa943f58136fec0ba2ce744.jpg)  
Figure 2: Shared evaluation pipeline. After quality filtering, students are partitioned into train and test sets. Train students to estimate item difficulty parameters $( b _ { j } )$ across all weeks. These frozen item parameters are transferred to the test set, where student ability $( \theta _ { \mathrm { t e s t } } )$ is calibrated in the early weeks. The calibrated model then predicts test student performance on held-out weeks. All models follow this pipeline.

Predictions of the LLMs are temporally grounded. Particularly, at attempt t, the models receive the student’s full trajectory through attempt t−1, including pass patterns and code summaries, then generate code predicting what the student would submit next. No fine-tuning is performed; the LLM parameters are frozen throughout. This asymmetry with the trained baselines is deliberate: the LLM serves as a zero-training reference point that tests whether an off-the-shelf model can predict solver behavior from context alone, and our claims about the LLM are scoped to this frozen setting. Prior work finds that fine-tuned LLMs match but do not surpass sequential models on next-step prediction (Neshaei et al., 2024). Full details of the retrieval mechanism, prompt construction, and simulation protocol are provided in Appendix D.

## 4 A Dataset of Problem-Solving Trajectories

The CodeInsight dataset comprises 3,286 undergraduate students from the Department of Computer Science at VNU-HCM University of Technology (Vietnam) across the 2022 and 2023 academic years. We collected data from two courses: Programming Fundamentals (PF, Spring semester), a first-year course, and Data Structures and Algorithms (DSA, Fall semester), a second-year course that requires PF as a prerequisite. Each course was offered twice, and within each offering, students were divided into cohorts by admission pathway: regular admission (L), higher-tuition track with lower entrance scores (CC), and retaking (DT). The dataset is structured at two distinct levels of granularity: a problem constitutes a discrete programming exercise evaluated via multiple hidden test cases, each yielding a binary outcome. A submission represents a single iterative attempt to solve a given problem, which is subsequently evaluated independently across all associated test cases. Table 1 summarizes the dataset statistics.

![](images/eca3172d4a9d249b2c4c6f48f9e629a7395a2bb6a746f025c031d4dba601150c.jpg)

![](images/51ff77a4276cd7469a3b65f00546f84fa28a7d6b730fd829e5af90fc9b8f3cec.jpg)

![](images/5b0781ad1b022366d1fcac07eb493add391ff6c00ef0e8d45f48fa55890dcfae.jpg)

![](images/e4535861fc21345e8b5b42593ce04a76889b33b262415d436d9bda046a1c450f.jpg)  
Figure 3: Top: average marks vs. number of attempts for DSA (left) and PF (right), comparing 2022 and 2023 cohorts. Dashed lines show cohort averages across all students and problems; background lines show individual student trajectories. Bottom: final submission scores by chronological problem order. Scores fluctuate across problems with problem difficulty; no monotonic improvement trend emerges.

Table 1: Summary statistics of the CodeInsight dataset, broken down by course. Students and problems may appear in multiple courses, so per-course counts do not sum to the unique total.
<table><tr><td>Course</td><td>Students</td><td>Problems</td><td>Test Cases</td><td>Submissions</td></tr><tr><td>DSA HK231 (Fall &#x27;23)</td><td>966</td><td>227</td><td>2,194</td><td>865,912</td></tr><tr><td>DSA HK221 (Fall &#x27;22)</td><td>513</td><td>175</td><td>1,661</td><td>456,916</td></tr><tr><td>PF HK232 (Spring &#x27;23)</td><td>1,485</td><td>129</td><td>1,307</td><td>854,104</td></tr><tr><td>PF HK222 (Spring &#x27;22)</td><td>1,413</td><td>119</td><td>1,201</td><td>897,863</td></tr><tr><td>Unique Total</td><td>3,286</td><td>394</td><td>3,888</td><td>3,074,795</td></tr></table>

To ensure privacy, student identities are systematically anonymized using numeric indices. This anonymization process included the removal of all names and unique personal identifiers, as well as a thorough screening of the submissions to verify the absence of embedded emails or student identification numbers within the code. The platform logs exact submission timestamps alongside sequential student actions, such as starting, prechecking, saving, checking, and finishing attempts, within an interactive coding environment. The finalized dataset encompasses the students’ raw source code, exact submission times, admission-track labels, and all evaluation test cases, including those originally designated as private. This dataset remains the copyrighted intellectual property of the Department and is released under a specific departmental license intended strictly for educational purposes upon request. Academic researchers may request access under this departmental license. Approved requesters receive the raw source code of every submission, the full problem statements and code templates, the public and private test cases with per-test-case pass/fail labels for every submission, the grading scripts used to compile and evaluate submissions, and the train/test split indices used in our experiments. Figure 1 illustrates the platform interface, detailing the problem statement, code editor, and automated feedback mechanisms for both incorrect and fully correct submissions. Additional details regarding the platform architecture and the submission workflow are provided in Appendix A.

The dataset comprises over 3 million programming submissions, capturing rich temporal and behavioral patterns that reflect authentic student engagement. Temporal analysis shows that students primarily work during evening hours (8 PM to 11 PM), with submission activity peaking on Saturdays, demonstrating typical undergraduate study patterns. The median time between consecutive submissions is 59 seconds, indicating rapid iterative development cycles where students frequently test and refine their code in response to automated feedback. Analysis across three behavioral dimensions (time between submissions, edit distance, and total submission count) shows that the majority of students occupy a central cluster characterized by moderate edit distances, short inter-submission intervals, and several hundred total submissions, consistent with this iterative workflow. A smaller set of outliers deviates from this pattern, exhibiting either unusually large edit distances or very long gaps between submissions, which may reflect off-platform work, copy-paste behavior, long interruptions, or other unobserved activity. A detailed breakdown of these behavioral patterns is provided in Appendix A, and example student trajectories with full submission sequences are shown in Appendix B.

Figure 3 presents learning curves across attempts for both courses, comparing the 2022 and 2023 cohorts. For each student-problem pair, we compute the score (fraction of test cases passed) at each attempt and average across all pairs at each attempt number. Both are foundational courses with largely invariant syllabi across years, so curriculum changes are an unlikely source of the cohort differences. Across both courses, early attempts on a given problem yield the largest score improvements, followed by diminishing returns. However, the year-over-year difference is substantially more pronounced in DSA than in PF, where the 2022 and 2023 cohorts follow nearly identical trajectories. Since each curve tracks repeated attempts on the same problem, task identity is controlled. Within a fixed time window, to avoid survivorship bias from students who stop attempting early, a student’s final score is propagated to all subsequent attempt indices, so the population remains constant across attempts. The resulting curves should be interpreted as descriptive evidence of within-problem improvement rather than causal estimates of learning. In contrast, the bottom row of Figure 3 orders each student’s problems chronologically by first-attempt timestamp and plots their last-submission score at each position, averaged across students. No consistent upward trend emerges because the average score on a given problem is a function of both student ability and problem difficulty, and these two factors vary independently across the curriculum. Disentangling them requires explicit difficulty estimation, which motivates the parametric models we evaluate in Section 3.

At the problem level (Appendix Figure 9), the number of steps students take to solve a question varies widely, with most problems requiring fewer than 10 attempts but a long tail extending past 20. Last-attempt scores are right-skewed, with a spike near zero where many students do not reach a passing score by their final submission and a long tail spread across higher scores. This distribution highlights the range of problem difficulty in the dataset and the substantial fraction of student-problem interactions that end without a correct solution. In summary, this dataset offers substantial value both quantitatively, through a large student population, high submission volume, and longitudinal tracking, and qualitatively, by capturing fine-grained problem-solving behaviors across different course levels and successive cohorts.

## 5 Results

## 5.1 Experimental Setup

An overview of our pipeline is presented in Figure 2. We filter the dataset to retain trajectories that exhibit iterative problem solving. Problems with pass rates below 10% or above 90% are removed, where a problem’s pass rate is the fraction of passing test-case outcomes across students’ final submissions. Such problems are too difficult or too easy to provide the problem-solving signal this analysis requires: nearly every solver ends at the same outcome, leaving little variation across solvers or attempts for the predictor to explain. Problems attempted by fewer than 25% of the students are also excluded for insufficient coverage. Appendix C.8 reports a sensitivity analysis over these filters: across all variations we test, the model ordering is consistent, and removing the filters entirely inflates absolute scores for every model without changing the comparative conclusions. Figure 4 visualizes the correctness matrix of this data across attempts. Each panel corresponds to one attempt level, with students on the vertical axis and test cases on the horizontal axis. At attempt 1, a substantial portion of the matrix is red, reflecting widespread initial failure. As the attempt number increases, the blue region expands progressively, indicating that students are learning and passing more test cases with repeated practice.

![](images/6cdac3287473e6e8a9095a5b52fe2f681deff47494d91346ece5eb3d0bda6182.jpg)  
Figure 4: Per-attempt correctness heatmap of students on DSA HK231. Each of the ten panels shows one attempt number. Within each panel, the vertical axis is students (one row per student, sorted by overall pass rate) and the horizontal axis is test cases (one column per test case, ordered chronologically); blue marks a passed test case and red a failed one. The progressive expansion of blue from attempt 1 to attempt 10 shows students mastering additional test cases through repeated practice.

![](images/602750c031b61bbcd2e1bd0a7174dc5e7dbb3db4071d075bdafc3233c176aad0.jpg)  
Figure 5: Balanced accuracy at each attempt number across both DSA courses. Top: DSA HK231. Bottom: DSA HK221.

## 5.2 Model Comparison

We first compare predictive accuracy across all models. Figure 5 reports balanced accuracy across attempt numbers on both DSA courses; the RSSM achieves the strongest balanced accuracy and AUC on three of the four courses. The parametric models and the RSSM share a common shape across attempts: balanced accuracy rises over the first few attempts, peaks around attempts two to five, and declines thereafter. The decline is steepest for the parametric models: IRT falls back to its attempt-1 level by attempt 7 on DSA HK231, and CIRT, whose per-problem temporal slope helps at early attempts, drops below IRT at the latest attempts on both courses, indicating that a single linear trend per problem captures limited sequential dynamics. The RSSM declines least, so its advantage over the parametric baselines is concentrated at later attempts. On DSA HK231 it reaches balanced accuracy of 0.71–0.73 at later attempts against 0.69– 0.71 for IRT and CIRT. On DSA HK221, where the overall pass rate is lower (29.5% vs 42.8% on DSA HK231) and students improve more gradually, it holds 0.66–0.69 against 0.62–0.68. DKT and Code-DKT trail the parametric models on both courses and at every attempt on DSA HK231; on DSA HK221 the gap narrows over the trajectory, and DKT overtakes both parametric models at the last attempt. BKT tracks the parametric models on DSA HK231 but falls below them on DSA HK221. TIKTOC sits between the two groups: on DSA HK231 it runs below the parametric models at every attempt while staying above DKT and Code-DKT, and on DSA HK221 it peaks early (0.64 at attempt 4) and flattens near 0.60 thereafter. The LLM predictor (Qwen3-14B) sits at 0.57–0.65, below the RSSM and the parametric models at nearly every attempt, though it overlaps with DKT and Code-DKT on DSA HK231. The PF courses show a similar shape across attempts, though the ordering shifts: TIKTOC is strongest on PF HK232 and the RSSM on PF HK222. Their outcomes are heavily skewed toward failure (pass rates of 8.4% and 17.5%), which makes outcomes easier to predict and inflates absolute scores for every model. Per-attempt metrics for all models and courses are reported in Appendix Table 7.

![](images/2aa3437385cb0082ff6ca6ccf2cbc2d91dc180e5db765c3246c3a754d2d5ff48.jpg)

![](images/4b94c004a59586ea3b743bc84c246bae2d2a8d1d5eb2147b747879a24ab99789.jpg)  
Figure 6: LLM predictor analysis on DSA HK231. Top: model comparison. Qwen3-14B achieves the highest balanced accuracy, outperforming Gemma-4-31B and Opus. Bottom: prompt ablation on Qwen3-14B. The full prompt outperforms all ablated variants.

Given the LLM’s lower accuracy, we investigate what drives this prediction gap. Figure 6 (top) shows the comparison among three LLMs: Qwen3- 14B achieves the highest balanced accuracy, outperforming both Gemma-4-31B and Claude Opus 4.6. This ordering is inversely related to coding proficiency: Opus 4.6 is the strongest coder among the three, followed by Gemma-4-31B, with Qwen3-14B the weakest on standard code generation benchmarks such as LiveCodeBench (Jain et al., 2025). Stronger coders may over-optimize solutions relative to the errors students actually make, reducing their fidelity as predictors in this context. Figure 6 (bottom) ablates the prompt structure for Qwen3-14B. The full prompt (student persona, prior attempt trajectory, and retrieved examples of the student’s approach on similar problems) outperforms all ablated variants, indicating that for this model each component contributes a complementary signal to next-step prediction. A fourth variant replaces similarity-based retrieval with the student’s most recently attempted problems and performs comparably to removing retrieval entirely, suggesting that content similarity drives the value of the retrieved examples.

We next examine whether the LLMs capture solver-specific failure modes or primarily reflect problem-level difficulty. Although the LLM predictors score below the parametric and sequential models on balanced accuracy, they produce actual code at each attempt, enabling a more detailed analysis of how they succeed and fail. Figure 8 compares test-case outcomes between real students and the LLMs on matched submissions. The LLMs produce correct output on more test cases than the typical student and nearly eliminate runtime errors and compilation errors, consistent with training corpora that consist predominantly of correct, compilable code. For wrong-output cases, the pattern differs: the majority of students’ wrong-output cases flow to LLM wrong-output cases, indicating that the LLMs tend to fail on the same test cases where students produce logically incorrect results.

To determine whether this overlap reflects genuine student modeling or simply shared problemlevel difficulty, we compute per-student Kendall τ between LLM-predicted and real scores, then repeat the analysis after subtracting each problem’s mean score across all students to isolate the studentspecific residual (Figure 7). Across all three LLMs, the same pattern holds: the raw τ distributions show weak positive means, but after removing the problem-difficulty signal, the distributions’ mean moves toward zero. This indicates that for all three models, the predictive signal is driven more by problem-level difficulty than student-specific modeling. A separate test reinforces this finding: when the LLM solves each problem without any studentspecific information (Appendix D.7), its problemlevel pass rates correlate strongly $( r = 0 . 8 0 )$ with its pass rates when given student context, while the correlation with real student pass rates is much lower $( r = 0 . 4 0 )$ . Together, these analyses establish the central finding of our LLM evaluation: the LLMs do not model individual student ability or strategy beyond problem-level difficulty. The LLM is thus better understood as a generative solver conditioned on student context rather than a faithful student simulator. This finding aligns with recent work by Ross and Andreas (2025).

![](images/3300e07974886a17169eca080ca51f4efbfab6e21c5694b51731b0d518aba865.jpg)

![](images/f2aa6ba98c0ff30314721ab6d92f6713af39a303c42a3b46beedda762fb71ff0.jpg)

![](images/75cc01baad1ab218fab45cf0ff7f15b2d5c71f01dae1c7cd5680851a0c9c7c99.jpg)  
Figure 7: Per-student Kendall τ between LLM-predicted and real scores across all three LLMs (Opus, Qwen3-14B, Gemma-4-31B). Top row: raw τ distributions show weak positive means. Bottom row: after subtracting each problem’s mean score across students to remove difficulty, the distribution means shift toward zero for all three models, indicating that the predictive signal is driven more by problem-level difficulty than by student-specific modeling.

![](images/cee3a5eaaa68305ce838d1cf443e3d0b4f8320714e60a97ef5b4b1d7811f9d2a.jpg)  
Figure 8: Test-case outcome flow between real students (top) and the LLM predictor (bottom) on matched submissions. Flow width is proportional to the number of test cases sharing that transition. The LLM produces more correct outcomes and fewer runtime and compilation errors than real students, but wrong-output cases transfer at high rates between the two.

## 6 Conclusion

In this work, we formalize iterative problemsolving as a sequential modeling task, introducing the large-scale CodeInsight dataset to facilitate research into how solvers refine their code over time. By benchmarking diverse models under a shared evaluation framework, we demonstrate that an RSSM capturing discrete latent variables achieves the strongest next-attempt prediction accuracy on three of the four courses, outperforming static parametric baselines with an advantage concentrated at later attempts. Conversely, our investigation into LLMs shows that while they successfully generate analyzable code submissions, they primarily function as generative solvers conditioned on context rather than faithful simulators of human behavioral flaws.

The practical applications and broader impact of these findings extend to both human learning analytics and the evaluation of autonomous coding agents. For education, modeling persisting errors and shifting strategies could support personalized feedback and adaptive instruction. For the AI and NLP communities, this research highlights a gap in current LLM capabilities, emphasizing the need for future fault-guided training to better capture realistic, student-specific error distributions. Ultimately, by moving beyond binary correctness to model the full arc of iterative refinement, this work provides an empirical foundation for studying systematic failure modes and advancing context-aware problem-solving models.

## Limitations

Several limitations constrain the scope of our findings. Our dataset is drawn from a single Vietnamese university and covers only C++ programming courses (a syntactically rich and error-prone language relative to Python), and was collected on a single grading platform whose feedback design shapes the trajectories we observe. Whether the observed patterns generalize to other institutions, programming languages, platforms, and course designs remains an open question.

We evaluated three LLMs (Opus, Qwen3-14B, Gemma-4-31B) with frozen parameters and no finetuning. Other LLMs may exhibit different strengths or failure modes. Fine-tuning on error-rich student code could improve behavioral fidelity; prior work shows that fault-guided training on buggy code improves code generation quality (Fan et al., 2025), suggesting current models are underexposed to realistic error patterns, though we have not tested this in our setting.

Our evaluation is also confined to prediction: accuracy is measured as a modeling capability, and we do not test whether forecasts, if surfaced to instructors or solvers, would improve outcomes. Such an intervention study requires a controlled deployment in a live course and is future work. Score prediction likewise stops short of diagnosis: a model can forecast repeated failure on a test case without identifying the underlying misconception or the help that would resolve it.

Future work should extend the evaluation to additional LLMs and programming languages, further ablate the predictor’s design choices, investigate whether other retrieval strategies or constrained decoding can encourage more realistic predictive behavior for the LLMs, and probe the RSSM’s discrete latent space for interpretable solver states. These extensions would help establish which findings are specific to this setting and which reflect broader properties of generative models applied to sequential prediction on problem-solving trajectories.

## Ethical Considerations

The primary ethical considerations in this research center on the curation and distribution of the CodeInsight dataset, which comprises over 3 million code submissions from 3,286 undergraduate students. As this research is performed after the courses are completed, we obtain permission from the Department to curate and use the data. To protect student privacy, all identities were anonymized using numeric indices. This anonymization process included the removal of all names and unique personal identifiers from the data. Additionally, we conducted a thorough screening of the submissions to verify that no embedded emails or student identification numbers remained within the source code. To mitigate the risks of data misuse, the dataset remains the copyrighted intellectual property of the university department and is released under a specific departmental license intended strictly for educational purposes upon request.

## Acknowledgments

SK is partially supported by NSF 2046795, 2205329, and 2504264, NIH, ARPA-H, the MacArthur Foundation, Good Ventures, Schmidt Sciences, the Hasso Plattner Förderstiftung, and Stanford HAI.

## References

Gati Aher, Rosa I. Arriaga, and Adam Tauman Kalai. 2023. Using large language models to simulate multiple humans and replicate human subject studies. In Proceedings ofthe 40th International Conference on Machine Learning, ICML’23. JMLR.org.

Paulo Blikstein, Marcelo Worsley, Chris Piech, Mehran Sahami, Steven Cooper, and Daphne Koller. 2014. Programming pluralism: Using learning analytics to detect patterns in the learning of computer programming. Journal ofthe Learning Sciences, 23(4):561– 599.

Neil C. C. Brown, Amjad Altadmri, Sue Sentance, and Michael Kölling. 2018. Blackbox, five years on: An evaluation of a large-scale programming data collection project. In Proceedings of the 2018 ACM Conference on International Computing Education Research, ICER ’18, pages 196–204, New York, NY, USA. Association for Computing Machinery.

Kairui Chen, Fuqun Huang, Zejing Liu, Haomiao Yu, Liuchang Meng, Shasha Mo, Li Zhang, and You Song. 2024a. Accoding: A graph-based dataset for online judge programming. Scientific Data, 11(1):548.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024b. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research. Featured Certification.

Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724– 1734, Doha, Qatar. Association for Computational Linguistics.

Albert T. Corbett and John R. Anderson. 1994. Knowledge tracing: Modeling the acquisition of procedural knowledge. User Modeling and User-Adapted Interaction, 4(4):253–278.

Steven Andrew Culpepper. 2014. If at first you don’t succeed, try, try again: Applications of sequential IRT models to cognitive assessments. Applied Psychological Measurement, 38(8):632–644.

DataShop@CMU. 2021. Dataset: Codeworkout data spring 2019. https://pslcdatashop.web.cmu. edu/Files?datasetId=3458. Accessed: 2026-03- 31.

Zhangqi Duan, Nigel Fernandez, Alexander Hicks, and Andrew Lan. 2025. Test Case-Informed Knowledge Tracing for Open-ended Coding Tasks. In Proceedings of the 15th International Learning Analytics and Knowledge Conference, LAK ’25, pages 238–248, New York, NY, USA. Association for Computing Machinery.

Susan E. Embretson and Steven P. Reise. 2000. Item Response Theory for Psychologists, 1 edition. Psychology Press.

Fahmid Morshed Fahid, Xiaoyi Tian, Andrew Emerson, Joseph B. Wiggins, Dolly Bounajim, Andy Smith, Eric Wiebe, Bradford Mott, Kristy Elizabeth Boyer, and James Lester. 2021. Progression trajectory-based student modeling for novice block-based programming. In UMAP ’21: Proceedings of the 29th ACM Conference on User Modeling, Adaptation and Personalization, UMAP ’21, page 189–200, New York, NY, USA. Association for Computing Machinery.

Lishui Fan, Zhongxin Liu, Haoye Wang, Lingfeng Bao, Xin Xia, and Shanping Li. 2025. Fgit: Faultguided fine-tuning for code generation. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 1338–1350.

Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. 2021. Mastering atari with discrete world models. In International Conference on Learning Representations.

Su-Pin Hung and Hung-Yu Huang. 2019. A sequential process model for cognitive diagnostic assessment with repeated attempts. Applied Psychological Measurement.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations.

Naiming Liu, Zichao Wang, Richard Baraniuk, and Andrew Lan. 2022. Open-ended knowledge tracing for computer science education. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3849–3862, Abu

Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yikai Lu and Ying Cheng. 2026. Adaptive testing for multiple-choice, multiple-attempt test items. Journal ofEducational and Behavioral Statistics.

Ye Mao, Yang Shi, Samiha Marwan, Thomas W. Price, Tiffany Barnes, and Min Chi. 2021. Knowing when and where: Temporal-ASTNN for student learning progression in novice programming tasks. In Proceedings of the 14th International Conference on Educational Data Mining.

Julia M. Markel, Steven G. Opferman, James A. Landay, and Chris Piech. 2023. Gpteach: Interactive ta training with gpt-based students. In Proceedings of the Tenth ACM Conference on Learning @ Scale, L@S ’23, page 226–236, New York, NY, USA. Association for Computing Machinery.

Marcus Messer, Neil C. C. Brown, Michael Kölling, and Miaojing Shi. 2024. Automated grading and feedback tools for programming education: A systematic review. ACM Trans. Comput. Educ., 24(1).

Mihran Miroyan, Rose Niousha, Joseph E. Gonzalez, Gireeja Ranade, and Narges Norouzi. 2025. ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle. arXiv preprint. ArXiv:2507.12674 [cs].

Holger Mohr, Katharina Zwosta, Dimitrije Markovic, Sebastian Bitzer, Uta Wolfensteller, and Hannes Ruge. 2018. Deterministic response strategies in a trial-and-error learning task. PLOS Computational Biology, 14(11):1–19.

Seyed Parsa Neshaei, Richard Lee Davis, Adam Hazimeh, Bojan Lazarevski, Pierre Dillenbourg, and Tanja Käser. 2024. Towards modeling learner performance with large language models. In Proceedings of the 17th International Conference on Educational Data Mining, pages 759–768, Atlanta, Georgia, USA. International Educational Data Mining Society.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to route LLMs from preference data. In The Thir teenth International Conference on Learning Representations.

Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas Guibas, and Jascha Sohl-Dickstein. 2015. Deep knowledge tracing. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1, NIPS’15, page 505–513, Cambridge, MA, USA. MIT Press.

Siddharth Reddy, Igor Labutov, Siddhartha Banerjee, and Thorsten Joachims. 2016. Unbounded human learning: Optimal scheduling for spaced repetition. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data

Mining, KDD ’16, page 1815–1824, New York, NY, USA. Association for Computing Machinery.

Kelly Rivers, Erik Harpstead, and Ken Koedinger. 2016. Learning curve analysis for programming: Which concepts do students struggle with? In Proceedings ofthe 2016 ACM Conference on International Computing Education Research, ICER ’16, page 143–151, New York, NY, USA. Association for Computing Machinery.

Alexis Ross and Jacob Andreas. 2025. Learning to make mistakes: Modeling incorrect student thinking and key errors. Preprint, arXiv:2510.11502.

Alexander Scarlatos, Nigel Fernandez, Christopher Ormerod, Susan Lottridge, and Andrew Lan. 2025. SMART: Simulated students aligned with item response theory for question difficulty prediction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25071–25094, Suzhou, China. Association for Computational Linguistics.

Shuanghong Shen, Qi Liu, Zhenya Huang, Yonghe Zheng, Minghao Yin, Minjuan Wang, and Enhong Chen. 2024. A survey of knowledge tracing: Models, variants, and applications. IEEE Transactions on Learning Technologies, 17:1858–1882.

Yang Shi, Min Chi, Tiffany Barnes, and Thomas W. Price. 2022. Code-DKT: A Code-based Knowledge Tracing Model for Programming Tasks. In Proceedings of the 15th International Conference on Educational Data Mining (EDM 2022), pages 50–61, Durham, United Kingdom. International Educational Data Mining Society.

Tao Wu, Jingyuan Chen, Wang Lin, Mengze Li, Yumeng Zhu, Ang Li, Kun Kuang, and Fei Wu. 2025. Embracing imperfection: Simulating students with diverse cognitive levels using LLM-based agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9887–9908, Vienna, Austria. Association for Computational Linguistics.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

## A Dataset Details

Throughout the course, students engage in weekly programming assignments that involve a variety of topics, starting with object-oriented programming (OOP), recursion, array lists, and singly linked lists in the first week. The subsequent weeks introduce more advanced data structures and algorithms, including doubly linked lists, stacks, queues, sorting algorithms, binary trees, AVL trees, and search algorithms, culminating in topics like hash functions and graph theory by the final week. Each week begins with an exam covering the previous week’s material.

Prechecks allow students to test their code against a public set of test cases, providing feedback on whether their code compiles and passes basic tests. Students see the input, output, and feedback on their code’s performance. In contrast, submissions are evaluated against a private set of test cases that assess the robustness of their solutions, with more challenging test cases and potential penalties for multiple submissions. Here, students only see a numerical evaluation score.

The course has six weeks of lab sections; each week covers several topics, and each topic includes several questions. Students will do exercises by navigating to each topic and attempting the questions inside. They may try multiple times. For each question, they receive a problem statement and a code editor. After completing their code, they can press “Submit” to check if they have passed the test cases. If they fail the test cases, they can edit their code and resubmit multiple times. The system will save each submission. Students will receive scores based on the number of test cases passed. The score for a topic is the sum of the scores for each related question. Whenever students submit their answers, they receive information about which test cases have passed and which have not. Using this feedback, students should be able to learn to fix their code to perform better, thereby improving with each submission.

## A.1 Student Behavior Patterns

To understand the relationship between student engagement patterns and learning outcomes, we analyzed submission pacing behavior across all students in the dataset. The analysis shows that students who make more submissions per problem tend to achieve higher final scores, demonstrating that persistent engagement with iterative refinement is associated with better outcomes. The median student makes 455 submissions across all problems (mean: 610 submissions), reflecting sustained engagement with the course material throughout the semester. However, the relationship between pause duration (median time between consecutive submissions) and performance is more nuanced, with successful students exhibiting a range of pacing strategies.

![](images/6cb14f12ac2cea9c6a3623ef880b20ad7025be556c6e5aad127d63e246325d0a.jpg)

![](images/6685439c74596a3364cf4eaecee1b2d9bc8477ff7437eb27bdad04a58d59eaf0.jpg)  
Figure 9: Problem-level distributions across all courses. (Top) Average number of attempts per problem. (Bottom) Average last-attempt score per problem.

Examination of pause time distributions shows that students operate within relatively compressed time scales. The median pause time between consecutive submissions is 59 seconds, with a mean of 2.7 minutes. This pattern suggests that students rely heavily on the automated feedback system to guide incremental code improvements. The predominance of short pause times indicates a trialand-error approach where students make frequent small adjustments in response to test results.

Analysis of high-volume submitters shows varied performance outcomes (Figure 10): while some achieve perfect or near-perfect scores through persistent iteration, others continue to struggle despite numerous attempts. This heterogeneity indicates that submission volume alone is not a reliable predictor of success; the quality of revisions and the student’s ability to interpret and act on feedback are equally important factors.

Figure 11 shows pairwise relationships among three behavioral features: average time between submissions, average edit distance (Levenshtein distance between consecutive code submissions), and total number of submissions. The plots show that most students cluster in a region of moderate edit distances and submission counts, with a long tail of outliers exhibiting either very high edit distances or very long inter-submission intervals.

## A.2 Course Characteristics and Trajectory Quality

Figure 12 compares the four course offerings across six dimensions. The DSA courses exhibit substantially richer problem-solving trajectories than the PF courses. Table 2 summarizes the key differences.

Table 2: Trajectory characteristics across courses. PF courses are dominated by floor effects and flat trajectories, while DSA courses show meaningful improvement across attempts.
<table><tr><td></td><td>PF 232</td><td>PF 222</td><td></td><td>DSA 231 DSA 221</td></tr><tr><td>Mean pass rate</td><td>0.08</td><td>0.18</td><td>0.43</td><td>0.31</td></tr><tr><td>Always-fail trajectories</td><td>58%</td><td>63%</td><td>30%</td><td>43%</td></tr><tr><td>Improving trajectories</td><td>28%</td><td>25%</td><td>43%</td><td>36%</td></tr><tr><td>Flat trajectories</td><td>69%</td><td>72%</td><td>47%</td><td>56%</td></tr><tr><td>Pass rate ∆ (att 1→10)</td><td>-3.2pp</td><td>-2.0pp</td><td>−0.2pp</td><td>+2.5pp</td></tr></table>

In the PF courses, 58–63% of multi-attempt trajectories are always-fail: the solver never passes a single test case across all attempts. Overlapping with this group, 69–72% of trajectories show no meaningful change between their first and second halves. As described in the main text, Figure 3 accounts for survivorship bias by applying last-observation-carried-forward, which propagates each solver’s most recent score to subsequent attempt indices. When this correction is removed and only solvers who actually made each attempt are counted, PF pass rates decline across attempts (−2 to −3 percentage points from attempt 1 to 10), indicating that solvers who persist to later attempts are predominantly those who are stuck rather than those who are improving.

The DSA courses, by contrast, show genuine improvement across attempts, though the two offerings differ in how much sequential signal they provide. DSA HK221 has a lower first-attempt pass rate (27.8% vs 39.1%), meaning solvers start further from the solution and have more room to improve. Its pass rate trend is the only positive one across all four courses (+2.5pp from attempt 1 to 10). Although DSA HK221 has fewer improving trajectories overall (36% vs 43% on DSA HK231), the trajectories that do improve show a clearer, more gradual signal. DSA HK221 also has longer attempt sequences (mean 3.79 vs 3.18 per problem) and a higher fraction of problems with 5+ attempts (32% vs 25%), giving a recurrent model more sequential context to work with. DSA HK231 has a higher base pass rate and a flatter trend across attempts. These differences are reflected in the main results (Figure 5): the RSSM leads the parametric models on both DSA courses, with its advantage concentrated at later attempts.

This gradient in trajectory quality (DSA HK221 with the strongest sequential signal, then DSA HK231, then PF with no sequential signal) explains why the main results focus on the DSA courses.

## B Examples of Student Learning Trajectories

The following examples illustrate typical student learning trajectories from the CodeInsight dataset. These trajectories show how students iteratively debug and refine their code in response to automated test feedback, exhibiting the behavioral patterns (rapid iteration, incremental edits, and varying debugging strategies) that the knowledge-tracing and LLM-based models aim to predict.

Attempt 1 - 25/09/23, 07:21:06: Started Attempt result: Not complete. Marks: 0.

Attempt 2 - 25/09/23, 08:35:00: Submit:

int buyCar(int \*nums, int length, int k) {   
int num = 0;   
while (k > 0) {   
int index = 0;   
int minPrice = nums[0];   
for (int i = 0; i < length; i++) {   
if (minPrice > nums[i]) {   
minPrice = nums[i];   
index = i;   
}   
}   
k -= minPrice;   
num++; nums[index] = 999999;   
}   
return num - 1;   
}

Attempt result: Partially correct. Marks: 0.40. Attempt 3 - 25/09/23, 08:35:16: Submit:

![](images/bdc6d80f978cd2aaf15c664f304fd0cc1370b2623477a365cde964f870410a61.jpg)

![](images/7167cc2a3f7188e1e534b90d33c8f35bca6c7e36b5ed17c9b6ee8d0e2ae804a0.jpg)  
Figure 10: Aggregate submission pacing patterns across all students and courses. (Left) Scatter plot of total submissions vs. median pause time, colored by final score. (Right) Distribution of total submissions per student, with the median (455) marked by the dashed red line.

![](images/83ffb0af89d5a6bb2f901c19f574bab73240723ad5459ae74eee516f36cdd923.jpg)

![](images/40dbd89d7e789ac875aab1d309e170f2aaa1349dea60d6f09ef224cc81a3045e.jpg)

![](images/f9c3bcaf2271372fa2eedb2353cb1d572aa8acb227841f039795c36242eeef59.jpg)  
Figure 11: Student coding behavior across all courses. From left to right: average time between submissions vs. average edit distance; average time between submissions vs. total submissions; and average edit distance vs. total submissions.

```perl
int buyCar(int *nums, int length, int k) {
int num = 0;
while (k $>$ 0) {
int index = 0;
int minPrice = nums[0];
for (int i = 0; i $<$ length; i++) {
if (minPrice $>$ nums[i])
,→ minPrice = nums[I];
index = I;
}
k -= minPrice;
num++; nums[index] = 999999999;
}
return num - 1;
}
Attempt result: Partially correct. Marks: 0.40.
Attempt 4 - 25/09/23, 08:35:36: Submit:
int buyCar(int *nums, int length, int k) {
int num = 0;
while (k $>$ 0) {
int index = 0;
int minPrice = nums[0];
for (int i = 0; i $<$ length; i++) {
if (minPrice $>$ nums[i])
,→ minPrice = nums[I];
index = I;
```

}   
k -= minPrice; num++;   
nums[index] = 2147483647;   
}   
return num - 1;   
}   
Attempt result: Partially correct. Marks: 0.40.   
Attempt 5 - 25/09/23, 08:44:40: Submit:   
int buyCar(int \*nums, int length, int k) {   
int num = 0;   
while (k \$>=\$ 0) {   
int index = 0; int minPrice = nums   
,→ [0];   
for (int i = 0; i \$<\$ length; i++) {   
if (minPrice \$>\$ nums[i]) {   
minPrice = nums[I];   
index = I;   
}   
}   
k -= minPrice; num++; nums[index] =   
,→ 999999;   
}   
return num - 1;   
}   
Attempt result: Correct. Marks: 1.00.

Student Ability Distribution Problem Difficulty Learning Curve   
0.45   
PF 232 PF 232   
7 PF 222 DSA 231 6 PF 222 DSA 231   
DSA 221 0.40 DSA 221   
6   
5   
0.35   
5   
4   
4 0.30   
3   
3   
0.25   
2   
2   
0.20   
1   
0 0 0.15   
0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8 1.0 2 3 6 8 9 10   
Student Pass Rate Problem Pass Rate Attempt   
Trajectory Outcomes Improvement Patterns Summary Statistics   
1.0 1.0 PF 232   
10 0.6 PF 222   
16 DSA 231   
DSA 221   
29 0.8 0.5   
41 69 56 0.4   
0.6 0.6   
48   
0.4   
63   
43 43   
0.2 0.2   
Always Fail 30 28 25 1Improving   
Mixed Flat   
Always Pass Dectining   
0.0 0.0 0.0   
PF232 PF 222 DSA 231 DSA 221 PF232 PF 222 DSA 231 DSA 221 Student Student Frac Frac   
Pass Rate Ability Std Always Fail Improving  
Figure 12: Course characteristics across all four offerings. Top row: solver ability distribution, problem difficulty distribution, and pass rate by attempt number. Bottom row: trajectory outcome breakdown (always-fail / mixed / always-pass), improvement pattern breakdown (improving / flat / declining), and summary statistics. PF courses are dominated by always-fail trajectories and flat dynamics, while DSA courses show more diverse outcomes and gradual improvement.

Attempt 6 - 27/09/23, 17:33:21: Attempt finished submitting.

The following trajectory shows a student who submits a complete solution 9 seconds after starting the attempt, suggesting the code was written or prepared before the recorded attempt began:

Attempt 1 - 29/09/23, 13:26:13: Started. Attempt result: Not complete. Marks: 0.

Attempt 2 - 29/09/23, 13:26:22: Submit:

int mins[1000]{}, maxs[1000]{}, mine[1000]{},   
,→ mmaxe[1000]{};   
int minimumAmplitude(vector<int>& nums, int   
,→ k) {   
int ans = 1e9;   
int n = nums.size();   
int\* mins = new int[n] {};   
int\* mine = new int[n] {};   
int\* maxs = new int[n] {};   
int\* maxe = new int[n] {};   
mins[0] = nums[0]; maxs[0] = nums[0];   
,→ mine[n - 1] = nums[n - 1];   
maxe[n - 1] = nums[n - 1];   
for (int i = 1; i < n; i++) {   
mins[i] = min(mins[i - 1], nums[i]);   
maxs[i] = max(maxs[i - 1], nums[i]);   
}   
for (int i = n-2; i >= 0; i--) {   
mine[i] = min(mine[i + 1], nums[i]);   
maxe[i] = max(maxe[i + 1], nums[i]);   
}   
int min\_arr = 0;   
int max\_arr = 0;

for (int i = 0; i < n-k; i++) {   
if(i!= 0) min\_arr = min(mins[i-1],   
,→ mine[i+k]);   
else min\_arr = mine[i + k];   
if (i + k < n - 1) max\_arr = max(   
,→ maxs[i - 1], maxe[i + k]);   
else max\_arr = maxs[i]; ans = min(   
,→ ans, max\_arr - min\_arr);   
}   
return ans;   
}

Attempt result: Correct. Marks: 1.00.

Attempt 4 - 29/09/23, 13:29:58: Attempt finished submitting.

An example for using precheck to check the response before officially submitting:

Attempt 1 - 9/10/23, 09:33:20: Started Attempt result: Not complete. Marks: 0.

Attempt 2 - 21/10/23, 10:06:07: Prechecked:

```c
void push(T item) {
// TODO: Push new element into the end
,→ of the queue
list->add(item);
}
T pop() {
// TODO: Remove an element in the head
,→ of the queue
T ans =list->get(0);
list->removeAt(0);
return ans;
}
```

```c
T top() {
// TODO: Get value of the element in the
,→ head of the queue
return list->get(0);
}
bool empty() {
// TODO: Determine if the queue is empty
return list->size() == 0;
}
int size() {
// TODO: Get the size of the queue
return list->size();
}
void clear() {
// TODO: Clear all elements of the queue
list->clear();
}
Attempt result: Precheck results. Marks: 0.
Attempt 3 - 21/10/23, 10:06:58: Prechecked:
void push(T item) {
// TODO: Push new element into the end
,→ of the queue
list->add(item);
}
T pop() {
// TODO: Remove an element in the head
,→ of the queue
T ans =list->get(0);
list->removeAt(0);
return ans;
}
T top() {
// TODO: Get value of the element in the
,→ head of the queue
return list->get(0);
}
bool empty() {
// TODO: Determine if the queue is empty
return list.size() == 0;
}
int size() {
// TODO: Get the size of the queue
return list.size();
}
void clear() {
// TODO: Clear all elements of the queue
list->clear();
}
Attempt result: Precheck results. Marks: 0.
Attempt 4 - 21/10/23, 10:07:12: Submit:
void push(T item) {
// TODO: Push new element into the end
,→ of the queue
list->add(item);
}
T pop() {
// TODO: Remove an element in the head
,→ of the queue
T ans =list->get(0);
list->removeAt(0);
return ans;
}
T top() {
// TODO: Get value of the element in the
,→ head of the queue
return list->get(0);
```

}   
bool empty() {   
// TODO: Determine if the queue is empty   
return list.size() == 0;   
}   
int size() {   
// TODO: Get the size of the queue   
return list.size();   
}   
void clear() {   
// TODO: Clear all elements of the queue   
list->clear();   
}   
Attempt result: Incorrect. Marks: 0.00.   
Attempt 5 - 21/10/23, 10:07:32: Prechecked:   
void push(T item) {   
// TODO: Push new element into the end   
,→ of the queue   
list.add(item);   
} T pop() {   
// TODO: Remove an element in the head   
,→ of the queue   
T ans =list->get(0);   
list.removeAt(0);   
return ans;   
}   
top() {   
// TODO: Get value of the element in the   
,→ head of the queue   
return list->get(0);   
}   
bool empty() {   
// TODO: Determine if the queue is empty   
return list.size() == 0;   
}   
int size() {   
// TODO: Get the size of the queue   
return list.size();   
}   
void clear() {   
// TODO: Clear all elements of the queue   
list.clear();   
}   
Attempt result: Precheck results. Marks: 0.00.   
Attempt 6 - 21/10/23, 10:07:36: Submit:   
void push(T item) {   
// TODO: Push new element into the end   
,→ of the queue   
list.add(item);   
}   
T pop() {   
// TODO: Remove an element in the head   
,→ of the queue   
T ans =list->get(0);   
list.removeAt(0);   
return ans;   
}   
T top() {   
// TODO: Get value of the element in the   
,→ head of the queue   
return list->get(0);   
}   
bool empty() {   
// TODO: Determine if the queue is empty   
return list.size() == 0;   
}

int size() {   
// TODO: Get the size of the queue   
return list.size();   
}   
void clear() {   
// TODO: Clear all elements of the queue   
list.clear();   
}   
Attempt result: Incorrect. Marks: 0.00.   
Attempt 7 - 21/10/23, 10:07:49: Submit:   
void push(T item) {   
// TODO: Push new element into the end   
,→ of the queue   
list.add(item);   
}   
T pop() {   
// TODO: Remove an element in the head   
,→ of the queue   
T ans =list.get(0);   
list.removeAt(0);   
return ans;   
}   
T top() {   
// TODO: Get value of the element in the   
,→ head of the queue   
return list.get(0);   
}   
bool empty() {   
// TODO: Determine if the queue is empty   
return list.size() == 0;   
}   
int size() {   
// TODO: Get the size of the queue   
return list.size();   
}   
void clear() {   
// TODO: Clear all elements of the queue   
list.clear();   
}   
Attempt result: Correct. Marks: 1.00.   
Attempt 8 - 22/10/23, 20:40:43: Attempt fin  
ished submitting.

## C Model Details

All models operate on the same data and evaluation protocol. Students are split into train (55%), validation (15%), and test (30%) sets using a fixed random seed of 42; the validation students are used only for checkpoint selection and early stopping where a model requires them, and the test set is evaluated once per model. The atomic prediction unit is the individual test case: each problem contains multiple test cases, and each test case is modeled as an independent binary (pass/fail) outcome. Missing attempts are represented with a sentinel value of −1 in the three-dimensional correctness tensor (students × test cases × attempts) and are masked out during both training and evaluation. Attempts are capped at 10 per student-problem pair. Table 3 summarizes the key hyperparameters.

<table><tr><td></td><td>IRT</td><td>CIRT</td><td>BKT</td><td>DKT</td><td>Code-DKT</td><td>TIKTOC</td><td>RSSM</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>EM</td><td>Adam</td><td>Adam</td><td>AdamW</td><td>Adam</td></tr><tr><td>Learn. rate</td><td>0.01</td><td>0.01</td><td></td><td>0.001</td><td>0.001</td><td>1e-5 / 5e-4</td><td>3e-4</td></tr><tr><td>Epochs</td><td>3000</td><td>3000</td><td>50 it.</td><td>200</td><td>200</td><td>5</td><td>200</td></tr><tr><td>Batch size</td><td>full</td><td>full</td><td></td><td>100</td><td>100</td><td>32</td><td>64</td></tr><tr><td>Regular.</td><td></td><td></td><td>clip</td><td>dr. 0.5</td><td>dr. 0.2</td><td>dr. 0.05</td><td>dr. 0.1</td></tr><tr><td>Early stop</td><td>no</td><td>no</td><td>no</td><td>no</td><td>no</td><td>yes</td><td>yes</td></tr><tr><td>Hidden dim</td><td></td><td></td><td></td><td>64</td><td>64</td><td>256</td><td>512</td></tr></table>

Table 3: Hyperparameter summary. “full”: full-batch; “clip”: parameters clipped to [0.01, 0.99]; “dr.”: dropout; “it.”: EM iterations.

## C.1 IRT

A one-parameter logistic (1PL) Rasch model that predicts correctness as P(correct) = $\sigma ( \theta _ { s } - b _ { j } )$ where $\theta _ { s }$ is a per-solver parameter and $b _ { j }$ is a pertest-case difficulty parameter. Both are estimated via maximum likelihood. IRT is static: it assigns each solver a single parameter with no temporal component.

Training uses Adam with binary cross-entropy loss over all valid observations in a single batch for 3,000 epochs with a learning rate of 0.01. No regularization or early stopping is applied. In the student-split setting, test-case parameters $( b _ { j } )$ are first estimated from train students, then frozen. Test student parameters $( \theta _ { s } )$ are calibrated on the calibration window (weeks 1–3; predictions cover weeks 4 and later) over 1,000 additional epochs. Training loss curves are shown in Figure 13.

![](images/096bc176aa4d8dfc07ad659f97258f6be0d5c08fd05435c39bd16edfc01e6d48.jpg)  
Figure 13: IRT training loss (negative log-likelihood) across epochs for all four course offerings. Loss converges within the first few hundred of 3,000 epochs.

## C.2 CIRT

Continuous IRT extends IRT with a per-problem temporal slope:

$$
P ( { \mathrm { c o r r e c t } } ) = \sigma ( \theta _ { s } - z _ { j } - \lambda _ { j } \cdot t )\tag{1}
$$

where $z _ { j }$ is the baseline difficulty of test case $j ,$ $\lambda _ { j }$ is a problem-specific difficulty slope, and t is the solver’s attempt index. The slope $\lambda _ { j }$ is unconstrained: a negative value indicates that effective difficulty decreases with practice, while a positive value captures diminishing returns from repeated attempts. All parameters are fitted jointly via maximum likelihood.

Training follows the same protocol as IRT: Adam with learning rate 0.01 for 3,000 epochs, fullbatch. CIRT adds a per-test-case temporal slope $\lambda _ { j }$ with the attempt index t normalized over [1, T]. Training loss curves are shown in Figure 14.

![](images/23fabb88ff598c31786a25db29b673e02fab1b633614804d5c5783e0331747b4.jpg)  
Figure 14: CIRT training loss across epochs. Conver gence behavior is similar to IRT, with the additional temporal slope parameter producing comparable final loss values.

## C.3 BKT

Bayesian Knowledge Tracing, a hidden Markov model, models each test case as a hidden Markov process with a binary latent state and four parameters: the prior probability of mastery $P ( L _ { 0 } )$ , the per-attempt transition probability $P ( T )$ , the guess probability $P ( G )$ , and the slip probability $P ( S )$

Each test case is fitted independently via expectation-maximization over all student sequences. The algorithm runs for 50 iterations with fixed initialization $( P ( L _ { 0 } ) { = } 0 . 3 , P ( T ) { = } 0 . 1$ $P ( G ) { = } 0 . 2 , P ( S ) { = } 0 . 1 )$ . After each M-step, all four parameters are clipped to [0.01, 0.99]. The forward-backward pass operates on variable-length sequences per student, with missing attempts excluded before fitting. BKT does not produce a training loss curve comparable to the gradient-based models.

## C.4 DKT

Deep Knowledge Tracing, an RNN, processes the sequence of a solver’s (test case, outcome) interactions, truncated to the most recent 600, through a single-layer LSTM with hidden dimension 64, followed by dropout (0.5) on the hidden state and a linear readout producing a vector of predicted correctness probabilities at each timestep.

Training uses Adam with learning rate 0.001 and mini-batches of 100 students for 200 epochs. Gradients are clipped at norm 5.0. The loss is binary cross-entropy on the next-step prediction. No early stopping or learning rate schedule is applied. Hyperparameters were selected via a search over hidden dimension ∈ {64, 128, 200, 256}, learning rate $\in \ \{ 0 . 0 0 1 , 0 . 0 1 \}$ , and dropout $\in \ \{ 0 . 2 , 0 . 5 \}$ Training loss curves are shown in Figure 15.

![](images/2dc5a0600cb1d7fa4878bf211b09d6d594fc8df3c1e597fff0918d6f457a6125.jpg)  
Figure 15: DKT training loss across 200 epochs. The LSTM converges more slowly than the parametric models.

## C.5 Code-DKT

Code-DKT extends DKT by fusing a vector representation of each submitted program with the interaction encoding at every timestep of the recurrent model. The original extracts code2vec AST-path features from Java submissions; no such extractor exists for C++, so our adaptation uses the same pretrained code embeddings that the RSSM consumes (Qwen3-Embedding-8B over the raw submission text), giving both code-aware models identical code signal. Each submission’s 4096-dimensional embedding is projected to 32 dimensions by a learned linear layer with dropout, multiplied by an availability mask, and concatenated with the one-hot interaction encoding before the LSTM. All test cases of one submission share that submission’s embedding.

Training matches DKT’s student-split protocol: Adam with learning rate 0.001 and mini-batches of 100 students for 200 epochs, hidden dimension 64, dropout 0.2, gradient clipping at norm 5.0, binary cross-entropy on the next-step prediction, and sequences truncated to the most recent 600 interactions. Hyperparameters were selected via a search over projection dimension $\in \{ 3 2 , 6 4 \}$ and dropout $\in \ \{ 0 . 2 , 0 . 5 \}$ . Training loss curves are shown in Figure 16.

![](images/2d6ebca36405f68878962facd6a052b55fafc51be4b5c1d571d24332323f496e.jpg)  
Figure 16: Code-DKT training loss across epochs for all four course offerings.

## C.6 TIKTOC

TIKTOC (Duan et al., 2025) combines the objectives of OKT (Liu et al., 2022) and Code-DKT in a multi-task model: alongside generating the student’s next code submission with a finetuned language model, it predicts per-test-case pass/fail outcomes from the language model’s prompt hidden states. An LSTM knowledge estimator consumes the sequence of question and code embeddings, and its hidden state is injected additively into the prompt token embeddings before the language model processes them. Per-question scorer heads predict each test case’s outcome from the mean-pooled prompt hidden states. The training loss is the test-case binary cross-entropy plus a code-generation cross-entropy, weighted equally in the original.

Three adaptations were required for this benchmark. First, the original ASTNN code encoder (built for Java) is replaced with the same pretrained code embeddings the RSSM and Code-DKT consume. Second, the Llama backbone is replaced with Qwen3-0.6B, finetuned with LoRA (rank 64, α = 128, dropout 0.05) on all attention and MLP projections. Third, the original per-question prediction heads, viable at the 17 problems of the CSEDM dataset, are replaced with a single scorer shared across all questions and conditioned on the test case’s text embedding and index, since with an order of magnitude more questions per-question heads see too few observations each.

Training uses AdamW with learning rate 1e-5 for the language model and 5e-4 for the knowledge estimator and scorer, batch size 4 with gradient accumulation of 8 (effective batch 32), up to 5 epochs with early stopping on validation test-case BCE (patience 2), gradient clipping, prompts truncated to 768 tokens and code to 512 tokens, and sequences capped at 200 interactions. Evaluation needs no code generation or execution: the scorer emits probabilities directly. Training loss curves are shown in Figure 17.

![](images/d1c833f560622fab0bf69aafb19ef00f86318377b49b02a4e0ffdf968b47acc4.jpg)  
Figure 17: TIKTOC training loss across epochs for all four course offerings.

## C.7 RSSM

The RSSM captures solver characteristics through discrete latent variables $z _ { t }$ at each timestep, following the DreamerV2 framework. The architecture is visualized in Figure 18 and consists of five components:

Recurrent model:

$$
h _ { t } = f _ { \phi } ( h _ { t - 1 } , z _ { t - 1 } , q _ { t } )
$$

Representation model:

$$
z _ { t } \sim q _ { \phi } ( z _ { t } | h _ { t } , e _ { t } )
$$

Transition predictor:

$$
\hat { z } _ { t } \sim p _ { \phi } ( \hat { z } _ { t } | h _ { t } )
$$

Embedding predictor:

$$
\hat { e } _ { t } \sim p _ { \phi } ( \hat { e } _ { t } | z _ { t } )
$$

Score predictor:

$$
\hat { r } _ { t } \sim p _ { \phi } ( \hat { r } _ { t } | q _ { t } , h _ { t } , z _ { t } )\tag{2}
$$

At each timestep $t ,$ the solver’s submitted code is embedded into $e _ { t }$ and the problem into $q _ { t }$ via a pretrained LLM, each projected through learned linear layers. The GRU updates the hidden state $h _ { t }$ from the previous latent vector $z _ { t - 1 }$ and problem encoding $q _ { t } ,$ , routing information through a discrete bottleneck. The representation model (posterior) infers $z _ { t }$ from both $h _ { t }$ and $e _ { t } .$ , while the transition predictor (prior) predicts $z _ { t }$ from $h _ { t }$ alone. The latent vector uses a categorical distribution with 16 variables and 16 classes, sampled via straightthrough gradient estimation. The representation and transition models are MLPs with ELU activations. The score predictor takes the concatenation of $q _ { t } , h _ { t } ,$ and $z _ { t } ,$ combined with a learned per-testcase difficulty bias.

![](images/97453179b5a212dd145b66e6408a0bdce19a23a0873e4e10d1de9676df0d5095.jpg)  
Figure 18: RSSM architecture. The GRU updates a hidden state $h _ { t }$ from the previous discrete latent $z _ { t - 1 }$ and problem encoding $q _ { t }$ . A posterior infers $z _ { t }$ from $h _ { t }$ and the response embedding $_ { e _ { t } ; }$ a prior predicts $z _ { t }$ from $h _ { t }$ alone. The scoring head predicts per-test-case outcomes.

The model uses a GRU cell with hidden dimension 512. Training uses Adam with learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 4 }$ . Dropout is 0.1. Gradients are clipped at norm 1.0. The loss combines binary cross-entropy on per-test-case outcomes, MSE on embedding prediction (weight $\lambda _ { e } { = } 0 . 1 )$ , and KL divergence between the posterior and prior (weight $\beta { = } 0 . 5 )$ with KL balancing $( \alpha { = } 0 . 8 )$ , where sg denotes the stop-gradient operator:

$$
\begin{array}{c} \begin{array} { r l } { \mathcal { L } ( \phi ) = \mathbb { E } _ { q _ { 0 } \{ z _ { 1 } \} \cap \{ \alpha _ { 1 } , r \} \cap \underline { { r } } } \Bigg [ } & { } \\ & { \underset { t = 1 } { \overset { T } { \sum } } - \underset { m \in \mathbb { C } } { \underline { { \mathrm { i n } } } } p _ { \mathrm { e } } ( r _ { i } | q _ { i } , h _ { t } , z _ { t } ) } \\ & { \underset { \mathrm { e x t } } { \overset { \quad + \lambda _ { \mathrm { c r } } \smash { \mathrm { i n } } \phi _ { \mathrm { c } } } } \mathrm { e } ^ { - \epsilon _ { i }  \underline { { q _ { 0 } } }  } } \\ & {  \quad + \beta ( \alpha D _ { K L } \cup \underline { { \mathrm { e } } } ( z _ { i } | h _ { t } , \epsilon _ { t } ) ) \vert \vert p _ { \mathrm { \phi } } ( z _ { i } | h _ { t } )  } \\ & { \quad + ( 1 - \alpha ) D _ { K L } \lbrack \phi _ { \mathrm { \phi } } ( z _ { i } | h _ { t } , \epsilon _ { t } ) \vert \vert \mathrm { s g } ( p _ { \mathrm { \phi } } ( z _ { i } | h _ { t } ) ) \vert \vert } \\ & { \quad + ( 1 - \alpha ) D _ { K L } \lbrack \phi _ { \mathrm { \phi } } ( z _ { i } | h _ { t } , \epsilon _ { t } ) \vert \vert \mathrm { s g } ( p _ { \mathrm { \phi } } ( z _ { i } | h _ { t } ) ) \vert \vert } \end{array} ]  \end{array}\tag{3}
$$

Training processes students in mini-batches of 64 with Adam (learning rate $3 { \times } 1 0 ^ { - 4 } , ~ \beta _ { 1 } { = } 0 . 9 .$ $\beta _ { 2 } { = } 0 . 9 9 )$ for up to 200 epochs. The learning rate follows a cosine schedule that anneals to 10% of its initial value by epoch 80 and is held there. Validation AUC is evaluated every 10 epochs on held-out validation students; early stopping uses patience 50, the checkpoint with the highest validation AUC is restored after training, and the test set is evaluated once on the restored checkpoint. Hyperparameters were selected via a search over hidden dimension ∈ {128, 512, 1024}, encoder depth, dropout $\in \{ 0 . 0 , 0 . 1 \}$ , latent grid size $\in \{ 1 6 \times 1 6 , 3 2 \times 3 2 \}$ KL weight $\beta ~ \in ~ \{ 0 . 1 , 0 . 5 , 1 . 0 \}$ , maximum sequence length $\in \ \{ 2 0 0 , 4 0 0 , 6 0 0 \}$ . Training loss curves are shown in Figure 19.

![](images/959bd2a08e6e5fbb12c713905923a5aa62fe5b01848d46795365df99de9ce76b.jpg)  
Figure 19: RSSM training loss across epochs for all four course offerings, plotted up to the checkpoint selected by validation AUC.

## C.8 Evaluation Data Filtering

The evaluation protocol filters problems on two criteria computed over students’ final submissions: the pass rate must lie in [10%, 90%], which removes problems that nearly all students pass or fail, and more than 25% of students must have attempted the problem. No student-level filters are applied. Table 4 reports the students, problems, test cases, and submissions remaining after each filtering step for every course.

Table 4: Data remaining after each problem-level filtering step of the evaluation protocol, per course. Pass rates are computed over students’ final submissions.
<table><tr><td>Filter step</td><td></td><td></td><td></td><td>Students Problems Test cases Submissions</td></tr><tr><td>DSA HK231</td><td></td><td></td><td></td><td></td></tr><tr><td>All loaded data</td><td>965</td><td>227</td><td>2,194</td><td>349,929</td></tr><tr><td>Pass rate in [10%, 90%]</td><td>965</td><td>157</td><td>1,530</td><td>266,861</td></tr><tr><td>Coverage &gt; 25% of students</td><td>965</td><td>87</td><td>845</td><td>235,453</td></tr><tr><td>DSA HK221</td><td></td><td></td><td></td><td></td></tr><tr><td>All loaded data</td><td>507</td><td>175</td><td>1,661</td><td>192,988</td></tr><tr><td>Pass rate in [10%, 90%]</td><td>507</td><td>119</td><td>1,154</td><td>136,921</td></tr><tr><td>Coverage &gt; 25% of students</td><td>507</td><td>78</td><td>747</td><td>121,004</td></tr><tr><td>PF HK232</td><td></td><td></td><td></td><td></td></tr><tr><td>All loaded data</td><td>1,478</td><td>129</td><td>1,307</td><td>502,471</td></tr><tr><td>Pass rate in [10%, 90%]</td><td>1,478</td><td>53</td><td>543</td><td>217,149</td></tr><tr><td>Coverage &gt; 25% of students</td><td>1,478</td><td>38</td><td>383</td><td>214,533</td></tr><tr><td>PF HK222</td><td></td><td></td><td></td><td></td></tr><tr><td>All loaded data</td><td>1,399</td><td>119</td><td>1,201</td><td>592,269</td></tr><tr><td>Pass rate in [10%, 90%]</td><td>1,399</td><td>43</td><td>434</td><td>271,271</td></tr><tr><td>Coverage &gt; 25% of students</td><td>1,399</td><td>37</td><td>375</td><td>269,097</td></tr></table>

Table 5: Filter sensitivity on DSA HK231: test AUC with one filter changed at a time from the baseline. Each configuration defines a different evaluation universe of n test observations, so comparisons are within rows.
<table><tr><td>Configuration</td><td>n</td><td>IRT</td><td>CIRT</td><td>BKT</td><td>DKT</td><td>CodeDKT</td><td>RSSM</td></tr><tr><td>Baseline</td><td>128,261</td><td>0.804</td><td>0.813</td><td>0.789</td><td>0.628</td><td>0.693</td><td>0.831</td></tr><tr><td>Pass floor off</td><td>165,707</td><td>0.861</td><td>0.867</td><td>0.818</td><td>0.729</td><td>0.746</td><td>0.876</td></tr><tr><td>Pass ceiling off</td><td>128,261</td><td>0.804</td><td>0.813</td><td>0.788</td><td>0.618</td><td>0.662</td><td>0.827</td></tr><tr><td>No pass band</td><td>165,707</td><td>0.861</td><td>0.867</td><td>0.818</td><td>0.722</td><td>0.745</td><td>0.876</td></tr><tr><td>No filters</td><td>186,260</td><td>0.853</td><td>0.859</td><td>0.806</td><td>0.733</td><td>0.731</td><td>0.867</td></tr><tr><td>Attempts 5</td><td>89,317</td><td>0.813</td><td>0.819</td><td>0.784</td><td>0.626</td><td>0.724</td><td>0.834</td></tr><tr><td>Attempts 15</td><td>158,841</td><td>0.822</td><td>0.828</td><td>0.797</td><td>0.696</td><td>0.690</td><td>0.843</td></tr><tr><td>Coverage 0.15</td><td>130,156</td><td>0.803</td><td>0.812</td><td>0.787</td><td>0.663</td><td>0.687</td><td>0.824</td></tr><tr><td>Coverage 0.50</td><td>126,131</td><td>0.806</td><td>0.815</td><td>0.792</td><td>0.620</td><td>0.702</td><td>0.834</td></tr></table>

To test whether these thresholds drive the results, we re-ran the evaluation on DSA HK231 with each filter changed in turn: the pass-rate floor, ceiling, and full band removed, all filters removed, the attempt cap lowered to 5 and raised to 15, and the coverage threshold moved to 0.15 and 0.50. Each configuration defines a different evaluation universe, so each model in Table 5 is retrained per configuration. The table reports test AUC. The ordering is stable: the RSSM leads in every configuration, followed by CIRT, IRT, and then BKT, with DKT and Code-DKT trailing throughout. Removing the pass-rate floor raises AUC for all models because problems that nearly all students fail are easy to discriminate; the band therefore makes the benchmark harder and does not favor any model.

Table 6: Trainable parameters and inference time per observation. The LLM entry is generation wall clock per attempt; its parameters are frozen.
<table><tr><td>Model</td><td>Parameters</td><td>Inference / obs.</td><td>Hardware</td></tr><tr><td>IRT</td><td>1.4K</td><td>&lt;0.1 µs</td><td>Apple M5 Pro</td></tr><tr><td>CIRT</td><td>2.2K</td><td>&lt;0.1 µs</td><td>Apple M5 Pro</td></tr><tr><td>BKT</td><td>3.4K</td><td>2.0 µs</td><td>Apple M5 Pro</td></tr><tr><td>DKT</td><td>504.5K</td><td>8.0 µs</td><td>Apple M5 Pro</td></tr><tr><td>Code-DKT</td><td>643.8K</td><td>3.5 µs</td><td>Apple M5 Pro</td></tr><tr><td>RSSM</td><td>7.78M</td><td>120.2 µs</td><td>Apple M5 Pro</td></tr><tr><td>TIKTOC</td><td>0.6B</td><td>38.3 ms</td><td>Apple M5 Pro</td></tr><tr><td>LLM (Qwen3-14B)</td><td>14.8B (frozen)</td><td>2.7 s</td><td>A100-40GB (vLLM)</td></tr></table>

## C.9 Model Efficiency

Table 6 reports trainable parameter counts and measured inference time per observation for all models.

## C.10 Evaluation Metrics

All trained models output a probability that the solver passes each test case on the next attempt. For accuracy-based metrics, predictions are binarized at a fixed threshold of 0.5. Balanced accuracy is the mean of the true-positive and true-negative rates under this threshold, which accounts for the class imbalance in pass outcomes across courses and attempt numbers. Table 7 reports balanced accuracy, AUC, log loss, and Brier score for every model at each attempt number on all four courses, computed on the same student-split predictions; its balanced accuracy rows provide the full numbers underlying Figure 5. The LLM predictor generates code and produces binary pass/fail outcomes per test case rather than probabilities, so threshold-free metrics are not defined for it and only its balanced accuracy is reported.

Table 7: Balanced accuracy, AUC, log loss, and Brier score at each attempt number for all models on all four courses. Balanced accuracy corresponds to Figure 5; predictions are binarized at 0.5. Bold marks the best model at each attempt within each course and metric (highest for balanced accuracy and AUC, lowest for log loss and Brier). The LLM predictor produces binary pass/fail outcomes rather than probabilities, so only balanced accuracy is defined for it. Entries are omitted where an attempt has fewer than 10 test observations.
<table><tr><td rowspan=1 colspan=4>Course      Model     Metric     1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=2>8    9</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=2>DSA HK231IRT</td><td rowspan=1 colspan=2>Bal. Acc0.700</td><td rowspan=1 colspan=1>0.746</td><td rowspan=1 colspan=1>0.746</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.709</td><td rowspan=1 colspan=1>0.698</td><td rowspan=1 colspan=2>0.7100.701</td><td rowspan=1 colspan=1>0.706</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>AUC    0.792</td><td rowspan=1 colspan=1>0.831</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.816</td><td rowspan=1 colspan=1>0.784</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.779</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>0.797</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.563</td><td rowspan=1 colspan=1>0.513</td><td rowspan=1 colspan=1>0.510</td><td rowspan=1 colspan=1>0.509</td><td rowspan=1 colspan=1>0.518</td><td rowspan=1 colspan=1>0.544</td><td rowspan=1 colspan=1>0.548</td><td rowspan=1 colspan=1>0.550</td><td rowspan=1 colspan=1>0.550</td><td rowspan=1 colspan=1>0.521</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.191</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.185</td><td rowspan=1 colspan=1>0.187</td><td rowspan=1 colspan=1>0.186</td><td rowspan=1 colspan=1>0.187</td><td rowspan=1 colspan=1>0.177</td></tr><tr><td rowspan=1 colspan=2>CIRT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.719</td><td rowspan=1 colspan=1>0.755</td><td rowspan=1 colspan=1>0.748</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.705</td><td rowspan=1 colspan=1>0.699</td><td rowspan=1 colspan=1>0.706</td><td rowspan=1 colspan=1>0.685</td><td rowspan=1 colspan=1>0.688</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.819</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>0.782</td><td rowspan=1 colspan=1>0.796</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.549</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>0.512</td><td rowspan=1 colspan=1>0.509</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.536</td><td rowspan=1 colspan=1>0.526</td><td rowspan=1 colspan=1>0.526</td><td rowspan=1 colspan=1>0.525</td><td rowspan=1 colspan=1>0.507</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.185</td><td rowspan=1 colspan=1>0.168</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.172</td><td rowspan=1 colspan=1>0.181</td><td rowspan=1 colspan=1>0.178</td><td rowspan=1 colspan=1>0.176</td><td rowspan=1 colspan=1>0.177</td><td rowspan=1 colspan=1>0.170</td></tr><tr><td rowspan=1 colspan=2>BKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.715</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.730</td><td rowspan=1 colspan=1>0.739</td><td rowspan=1 colspan=1>0.726</td><td rowspan=1 colspan=1>0.710</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>0.702</td><td rowspan=1 colspan=1>0.701</td><td rowspan=1 colspan=1>0.711</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.776</td><td rowspan=1 colspan=1>0.822</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>0.809</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>0.776</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.766</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.784</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.589</td><td rowspan=1 colspan=1>0.559</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.593</td><td rowspan=1 colspan=1>0.604</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=1>0.625</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=1>0.601</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.202</td><td rowspan=1 colspan=1>0.189</td><td rowspan=1 colspan=1>0.198</td><td rowspan=1 colspan=1>0.197</td><td rowspan=1 colspan=1>0.205</td><td rowspan=1 colspan=1>0.210</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>0.218</td><td rowspan=1 colspan=1>0.210</td></tr><tr><td rowspan=1 colspan=2>DKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.585</td><td rowspan=1 colspan=1>0.603</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>0.605</td><td rowspan=1 colspan=1>0.581</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>0.587</td><td rowspan=1 colspan=1>0.588</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.624</td><td rowspan=1 colspan=1>0.641</td><td rowspan=1 colspan=1>0.647</td><td rowspan=1 colspan=1>0.640</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>0.623</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.627</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>1.017</td><td rowspan=1 colspan=1>0.989</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.916</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.959</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.885</td></tr><tr><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.303</td><td rowspan=1 colspan=1>0.297</td><td rowspan=1 colspan=1>0.275</td><td rowspan=1 colspan=1>0.273</td><td rowspan=1 colspan=1>0.272</td><td rowspan=1 colspan=1>0.283</td><td rowspan=1 colspan=1>0.270</td><td rowspan=1 colspan=1>0.271</td><td rowspan=1 colspan=1>0.268</td><td rowspan=1 colspan=1>0.266</td></tr><tr><td rowspan=1 colspan=2>Code-DKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.612</td><td rowspan=1 colspan=1>0.632</td><td rowspan=1 colspan=1>0.635</td><td rowspan=1 colspan=1>0.639</td><td rowspan=1 colspan=1>0.638</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.616</td><td rowspan=1 colspan=1>0.622</td><td rowspan=1 colspan=1>0.627</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.689</td><td rowspan=1 colspan=1>0.695</td><td rowspan=1 colspan=1>0.700</td><td rowspan=1 colspan=1>0.699</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.659</td><td rowspan=1 colspan=1>0.668</td><td rowspan=1 colspan=1>0.671</td><td rowspan=1 colspan=1>0.678</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.946</td><td rowspan=1 colspan=1>0.884</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.826</td><td rowspan=1 colspan=1>0.817</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.858</td><td rowspan=1 colspan=1>0.866</td><td rowspan=1 colspan=1>0.844</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.287</td><td rowspan=1 colspan=1>0.269</td><td rowspan=1 colspan=1>0.255</td><td rowspan=1 colspan=1>0.253</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.257</td><td rowspan=1 colspan=1>0.267</td><td rowspan=1 colspan=1>0.262</td><td rowspan=1 colspan=1>0.259</td><td rowspan=1 colspan=1>0.257</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>TIKTOC</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.647</td><td rowspan=1 colspan=1>0.705</td><td rowspan=1 colspan=1>0.707</td><td rowspan=1 colspan=1>0.717</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>0.689</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.673</td><td rowspan=1 colspan=1>0.670</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.728</td><td rowspan=1 colspan=1>0.784</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.779</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>0.767</td><td rowspan=1 colspan=1>0.759</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=1>0.757</td></tr><tr><td rowspan=1 colspan=1>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>0.565</td><td rowspan=1 colspan=1>0.547</td><td rowspan=1 colspan=1>0.557</td><td rowspan=1 colspan=1>0.550</td><td rowspan=1 colspan=1>0.543</td><td rowspan=1 colspan=1>0.546</td><td rowspan=1 colspan=1>0.554</td><td rowspan=1 colspan=1>0.551</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.219</td><td rowspan=1 colspan=1>0.200</td><td rowspan=1 colspan=1>0.191</td><td rowspan=1 colspan=1>0.184</td><td rowspan=1 colspan=1>0.188</td><td rowspan=1 colspan=1>0.187</td><td rowspan=1 colspan=1>0.185</td><td rowspan=1 colspan=1>0.186</td><td rowspan=1 colspan=1>0.190</td><td rowspan=1 colspan=1>0.187</td></tr><tr><td rowspan=3 colspan=2>Log</td><td rowspan=1 colspan=1>RSSM</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1>0.753</td><td rowspan=1 colspan=1>0.763</td><td rowspan=1 colspan=1>0.769</td><td rowspan=1 colspan=1>0.761</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>0.725</td><td rowspan=1 colspan=1>0.722</td><td rowspan=1 colspan=1>0.716</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>0.848</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=1>0.817</td><td rowspan=1 colspan=1>0.813</td><td rowspan=1 colspan=1>0.818</td><td rowspan=1 colspan=1>0.827</td></tr><tr><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.550</td><td rowspan=1 colspan=1>0.509</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=1>0.462</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>0.491</td><td rowspan=1 colspan=1>0.493</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.490</td><td rowspan=1 colspan=1>0.475</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.186</td><td rowspan=1 colspan=1>0.168</td><td rowspan=1 colspan=1>0.156</td><td rowspan=1 colspan=1>0.152</td><td rowspan=1 colspan=1>0.157</td><td rowspan=1 colspan=1>0.165</td><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>0.165</td><td rowspan=1 colspan=1>0.161</td></tr><tr><td rowspan=1 colspan=2>LLM</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.651</td><td rowspan=1 colspan=1>0.604</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1>0.583</td><td rowspan=1 colspan=1>0.594</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1>0.624</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.619</td><td rowspan=1 colspan=1>0.650</td></tr><tr><td rowspan=1 colspan=2>DSA HK221IRT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.617</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.671</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.654</td><td rowspan=1 colspan=1>0.639</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.781</td><td rowspan=1 colspan=1>0.810</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.768</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.598</td><td rowspan=1 colspan=1>0.532</td><td rowspan=1 colspan=1>0.492</td><td rowspan=1 colspan=1>0.459</td><td rowspan=1 colspan=1>0.471</td><td rowspan=1 colspan=1>0.471</td><td rowspan=1 colspan=1>0.469</td><td rowspan=1 colspan=1>0.491</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1>0.478</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.200</td><td rowspan=1 colspan=1>0.176</td><td rowspan=1 colspan=1>0.161</td><td rowspan=1 colspan=1>0.149</td><td rowspan=1 colspan=1>0.154</td><td rowspan=1 colspan=1>0.153</td><td rowspan=1 colspan=1>0.153</td><td rowspan=1 colspan=1>0.161</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>0.156</td></tr><tr><td rowspan=1 colspan=2>CIRT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.641</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.686</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.677</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.656</td><td rowspan=1 colspan=1>0.646</td><td rowspan=1 colspan=1>0.619</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.737</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>0.776</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>0.775</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.526</td><td rowspan=1 colspan=1>0.497</td><td rowspan=1 colspan=1>0.461</td><td rowspan=1 colspan=1>0.468</td><td rowspan=1 colspan=1>0.471</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=1>0.488</td><td rowspan=1 colspan=1>0.483</td><td rowspan=1 colspan=1>0.472</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.194</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.163</td><td rowspan=1 colspan=1>0.150</td><td rowspan=1 colspan=1>0.153</td><td rowspan=1 colspan=1>0.152</td><td rowspan=1 colspan=1>0.152</td><td rowspan=1 colspan=1>0.159</td><td rowspan=1 colspan=1>0.158</td><td rowspan=1 colspan=1>0.154</td></tr><tr><td rowspan=1 colspan=2>BKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.598</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.615</td><td rowspan=1 colspan=1>0.613</td><td rowspan=1 colspan=1>0.604</td><td rowspan=1 colspan=1>0.614</td><td rowspan=1 colspan=1>0.585</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.591</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.619</td><td rowspan=1 colspan=1>0.663</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.691</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.670</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.645</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.650</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.679</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.665</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.665</td><td rowspan=1 colspan=1>0.670</td><td rowspan=1 colspan=1>0.676</td><td rowspan=1 colspan=1>0.673</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.236</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>0.236</td><td rowspan=1 colspan=1>0.234</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>0.236</td><td rowspan=1 colspan=1>0.239</td><td rowspan=1 colspan=1>0.241</td><td rowspan=1 colspan=1>0.240</td></tr><tr><td rowspan=1 colspan=2>DKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1>0.649</td><td rowspan=1 colspan=1>0.647</td><td rowspan=1 colspan=1>0.663</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.644</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.647</td><td rowspan=1 colspan=1>0.642</td><td rowspan=1 colspan=1>0.658</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.652</td><td rowspan=1 colspan=1>0.702</td><td rowspan=1 colspan=1>0.704</td><td rowspan=1 colspan=1>0.730</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.702</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.710</td><td rowspan=1 colspan=1>0.686</td><td rowspan=1 colspan=1>0.704</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.812</td><td rowspan=1 colspan=1>0.719</td><td rowspan=1 colspan=1>0.659</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=1>0.594</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1>0.627</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.625</td></tr><tr><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.241</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.201</td><td rowspan=1 colspan=1>0.189</td><td rowspan=1 colspan=1>0.183</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.182</td><td rowspan=1 colspan=1>0.193</td><td rowspan=1 colspan=1>0.198</td><td rowspan=1 colspan=1>0.185</td></tr><tr><td rowspan=1 colspan=2>Code-DKTB</td><td rowspan=1 colspan=1>al. Acc</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.626</td><td rowspan=1 colspan=1>0.634</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.621</td><td rowspan=1 colspan=1>0.646</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1>0.640</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.630</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.677</td><td rowspan=1 colspan=1>0.698</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>0.668</td><td rowspan=1 colspan=1>0.701</td><td rowspan=1 colspan=1>0.712</td><td rowspan=1 colspan=1>0.681</td><td rowspan=1 colspan=1>0.693</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>0.749</td><td rowspan=1 colspan=1>0.687</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.668</td><td rowspan=1 colspan=1>0.695</td><td rowspan=1 colspan=1>0.650</td><td rowspan=1 colspan=1>0.643</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.649</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.252</td><td rowspan=1 colspan=1>0.239</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.221</td><td rowspan=1 colspan=1>0.209</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>0.209</td></tr><tr><td rowspan=3 colspan=2>TIKTOCLog</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.588</td><td rowspan=1 colspan=1>0.635</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1>0.644</td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1>0.596</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=1>0.608</td><td rowspan=1 colspan=1>0.598</td><td rowspan=1 colspan=1>0.601</td></tr><tr><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.671</td><td rowspan=1 colspan=1>0.743</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.779</td><td rowspan=1 colspan=1>0.732</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>0.744</td><td rowspan=1 colspan=1>0.737</td></tr><tr><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.616</td><td rowspan=1 colspan=1>0.564</td><td rowspan=1 colspan=1>0.528</td><td rowspan=1 colspan=1>0.491</td><td rowspan=1 colspan=1>0.527</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.510</td><td rowspan=1 colspan=1>0.516</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=1>0.487</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=1>0.191</td><td rowspan=1 colspan=1>0.176</td><td rowspan=1 colspan=1>0.162</td><td rowspan=1 colspan=1>0.176</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>0.160</td></tr><tr><td rowspan=3 colspan=2>RSSMLog</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.696</td><td rowspan=1 colspan=1>0.696</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.691</td><td rowspan=1 colspan=1>0.662</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.678</td></tr><tr><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.728</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.814</td><td rowspan=1 colspan=1>0.819</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>0.802</td><td rowspan=1 colspan=1>0.810</td></tr><tr><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1>0.521</td><td rowspan=1 colspan=1>0.487</td><td rowspan=1 colspan=1>0.459</td><td rowspan=1 colspan=1>0.455</td><td rowspan=1 colspan=1>0.448</td><td rowspan=1 colspan=1>0.448</td><td rowspan=1 colspan=1>0.473</td><td rowspan=1 colspan=1>0.464</td><td rowspan=1 colspan=1>0.446</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.200</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>0.149</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.146</td><td rowspan=1 colspan=1>0.146</td><td rowspan=1 colspan=1>0.156</td><td rowspan=1 colspan=1>0.152</td><td rowspan=1 colspan=1>0.145</td></tr><tr><td rowspan=1 colspan=2>LLM</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.623</td><td rowspan=1 colspan=1>0.591</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.636</td><td rowspan=1 colspan=1>0.614</td><td rowspan=1 colspan=1>0.580</td><td rowspan=1 colspan=1>0.582</td><td rowspan=1 colspan=1>0.570</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.579</td></tr><tr><td rowspan=1 colspan=2>PF HK232  IRT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.883</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.830</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.839</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.935</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.910</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.839</td><td rowspan=1 colspan=1>0.894</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>0.858</td></tr><tr><td rowspan=1 colspan=2>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>0.122</td><td rowspan=1 colspan=1>0.126</td><td rowspan=1 colspan=1>0.142</td><td rowspan=1 colspan=1>0.134</td><td rowspan=1 colspan=1>0.121</td><td rowspan=1 colspan=1>0.180</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=1>0.136</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=2>0.0200.028</td><td rowspan=1 colspan=1>0.029</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=1>0.029</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=2>0.0280.014</td><td rowspan=1 colspan=1>0.029</td></tr></table>

Table 7 (continued)
<table><tr><td rowspan=1 colspan=7>Course      Model     Metric     1     2    3     4    5</td><td rowspan=1 colspan=5>6    7    8    9    10</td></tr><tr><td rowspan=2 colspan=5>CIRT      Bal. Acc0.8830.8360.828AUC    0.936</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=2>0.8300.778</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.839</td></tr><tr><td rowspan=1 colspan=2>AUC0.936</td><td rowspan=1 colspan=1>0.911</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.911</td><td rowspan=1 colspan=1>0.897</td><td rowspan=1 colspan=2>0.9220.855</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.963</td><td rowspan=1 colspan=1>0.875</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Log loss0.091</td><td rowspan=1 colspan=1>0.122</td><td rowspan=1 colspan=1>0.126</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.132</td><td rowspan=1 colspan=2>0.121 0.176</td><td rowspan=1 colspan=1>0.121</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.144</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Brier    0.019</td><td rowspan=1 colspan=1>0.028</td><td rowspan=1 colspan=1>0.029</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=2>0.0290.043</td><td rowspan=1 colspan=1>0.028</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.032</td></tr><tr><td rowspan=1 colspan=1>BKT</td><td rowspan=1 colspan=2>Bal. Acc0.880</td><td rowspan=1 colspan=1>0.839</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=2>0.8340.785</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.906</td><td rowspan=1 colspan=1>0.835</td></tr><tr><td rowspan=2 colspan=1>Log</td><td rowspan=1 colspan=2>AUC    0.928</td><td rowspan=1 colspan=1>0.910</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.923</td><td rowspan=1 colspan=1>0.898</td><td rowspan=1 colspan=1>0.923</td><td rowspan=1 colspan=1>0.865</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.964</td><td rowspan=1 colspan=1>0.870</td></tr><tr><td rowspan=1 colspan=2>loss0.211</td><td rowspan=1 colspan=1>0.224</td><td rowspan=1 colspan=1>0.223</td><td rowspan=1 colspan=1>0.226</td><td rowspan=1 colspan=1>0.229</td><td rowspan=1 colspan=1>0.224</td><td rowspan=1 colspan=1>0.252</td><td rowspan=1 colspan=1>0.235</td><td rowspan=1 colspan=1>0.218</td><td rowspan=1 colspan=1>0.245</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>0.061</td><td rowspan=1 colspan=1>0.061</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1>0.062</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.066</td></tr><tr><td rowspan=1 colspan=1>DKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.876</td><td rowspan=1 colspan=1>0.832</td><td rowspan=1 colspan=1>0.826</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.835</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>0.847</td><td rowspan=1 colspan=1>0.843</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.876</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>0.906</td><td rowspan=1 colspan=1>0.833</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.130</td><td rowspan=1 colspan=1>0.184</td><td rowspan=1 colspan=1>0.187</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.198</td><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>0.261</td><td rowspan=1 colspan=1>0.162</td><td rowspan=1 colspan=1>0.108</td><td rowspan=1 colspan=1>0.186</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.028</td><td rowspan=1 colspan=1>0.037</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.044</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>0.034</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>0.035</td></tr><tr><td rowspan=1 colspan=1>Code-DKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.882</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=1>0.802</td><td rowspan=1 colspan=1>0.824</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=1>0.774</td><td rowspan=1 colspan=1>0.820</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.836</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.911</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>0.832</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.848</td><td rowspan=1 colspan=1>0.894</td><td rowspan=1 colspan=1>0.804</td></tr><tr><td rowspan=1 colspan=1>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.119</td><td rowspan=1 colspan=1>0.175</td><td rowspan=1 colspan=1>0.178</td><td rowspan=1 colspan=1>0.204</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.186</td><td rowspan=1 colspan=1>0.240</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.120</td><td rowspan=1 colspan=1>0.207</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.024</td><td rowspan=1 colspan=1>0.034</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.036</td><td rowspan=1 colspan=1>0.049</td><td rowspan=1 colspan=1>0.036</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>0.037</td></tr><tr><td rowspan=1 colspan=1>TIKTOC</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.883</td><td rowspan=1 colspan=1>0.848</td><td rowspan=1 colspan=1>0.848</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.865</td><td rowspan=1 colspan=1>0.859</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.866</td><td rowspan=1 colspan=1>0.949</td><td rowspan=1 colspan=1>0.858</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.932</td><td rowspan=1 colspan=1>0.977</td><td rowspan=1 colspan=1>0.980</td><td rowspan=1 colspan=1>0.986</td><td rowspan=1 colspan=1>0.991</td><td rowspan=1 colspan=1>0.984</td><td rowspan=1 colspan=1>0.988</td><td rowspan=1 colspan=1>0.991</td><td rowspan=1 colspan=1>0.999</td><td rowspan=1 colspan=1>0.992</td></tr><tr><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.101</td><td rowspan=1 colspan=1>0.076</td><td rowspan=1 colspan=1>0.068</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.071</td><td rowspan=1 colspan=1>0.080</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.064</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>0.024</td><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>0.019</td></tr><tr><td rowspan=2 colspan=1>RSSM</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.883</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.806</td><td rowspan=1 colspan=1>0.829</td><td rowspan=1 colspan=1>0.830</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.839</td></tr><tr><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.952</td><td rowspan=1 colspan=1>0.974</td><td rowspan=1 colspan=1>0.972</td><td rowspan=1 colspan=1>0.973</td><td rowspan=1 colspan=1>0.980</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.977</td><td rowspan=1 colspan=1>0.999</td><td rowspan=1 colspan=1>0.927</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>0.108</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.112</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.090</td><td rowspan=1 colspan=1>0.131</td><td rowspan=1 colspan=1>0.085</td><td rowspan=1 colspan=1>0.033</td><td rowspan=1 colspan=1>0.110</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.031</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.024</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>0.024</td></tr><tr><td rowspan=1 colspan=1>PF HK222  IRT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.712</td><td rowspan=1 colspan=1>0.746</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.756</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.754</td><td rowspan=1 colspan=1>0.761</td><td rowspan=1 colspan=1>0.769</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.765</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.859</td><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.889</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.892</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=1>0.886</td><td rowspan=1 colspan=1>0.893</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.323</td><td rowspan=1 colspan=1>0.278</td><td rowspan=1 colspan=1>0.261</td><td rowspan=1 colspan=1>0.266</td><td rowspan=1 colspan=1>0.281</td><td rowspan=1 colspan=1>0.264</td><td rowspan=1 colspan=1>0.258</td><td rowspan=1 colspan=1>0.256</td><td rowspan=1 colspan=1>0.259</td><td rowspan=1 colspan=1>0.245</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.100</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.084</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>0.074</td><td rowspan=1 colspan=1>0.076</td><td rowspan=1 colspan=1>0.072</td></tr><tr><td rowspan=1 colspan=1>CIRT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.711</td><td rowspan=1 colspan=1>0.743</td><td rowspan=1 colspan=1>0.759</td><td rowspan=1 colspan=1>0.756</td><td rowspan=1 colspan=1>0.744</td><td rowspan=1 colspan=1>0.753</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>0.761</td><td rowspan=1 colspan=1>0.776</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.889</td><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.882</td><td rowspan=1 colspan=1>0.894</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.897</td></tr><tr><td rowspan=1 colspan=1>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.321</td><td rowspan=1 colspan=1>0.278</td><td rowspan=1 colspan=1>0.261</td><td rowspan=1 colspan=1>0.266</td><td rowspan=1 colspan=1>0.280</td><td rowspan=1 colspan=1>0.262</td><td rowspan=1 colspan=1>0.257</td><td rowspan=1 colspan=1>0.252</td><td rowspan=1 colspan=1>0.253</td><td rowspan=1 colspan=1>0.242</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>0.073</td><td rowspan=1 colspan=1>0.074</td><td rowspan=1 colspan=1>0.071</td></tr><tr><td rowspan=1 colspan=1>BKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.659</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>0.709</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1>0.699</td><td rowspan=1 colspan=1>0.717</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.747</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.815</td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>0.866</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>0.869</td><td rowspan=1 colspan=1>0.878</td><td rowspan=1 colspan=1>0.873</td><td rowspan=1 colspan=1>0.872</td><td rowspan=1 colspan=1>0.867</td></tr><tr><td rowspan=1 colspan=1>Log</td><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.349</td><td rowspan=1 colspan=1>0.313</td><td rowspan=1 colspan=1>0.299</td><td rowspan=1 colspan=1>0.297</td><td rowspan=1 colspan=1>0.311</td><td rowspan=1 colspan=1>0.299</td><td rowspan=1 colspan=1>0.287</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=1>0.289</td><td rowspan=1 colspan=1>0.282</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.108</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=1>0.087</td><td rowspan=1 colspan=1>0.087</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.082</td><td rowspan=1 colspan=1>0.084</td><td rowspan=1 colspan=1>0.081</td></tr><tr><td rowspan=1 colspan=1>DKT</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.725</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.744</td><td rowspan=1 colspan=1>0.753</td><td rowspan=1 colspan=1>0.760</td><td rowspan=1 colspan=1>0.763</td><td rowspan=1 colspan=1>0.771</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>0.829</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>0.839</td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>0.867</td><td rowspan=1 colspan=1>0.863</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1>0.397</td><td rowspan=1 colspan=1>0.387</td><td rowspan=1 colspan=1>0.373</td><td rowspan=1 colspan=1>0.395</td><td rowspan=1 colspan=1>0.374</td><td rowspan=1 colspan=1>0.378</td><td rowspan=1 colspan=1>0.359</td><td rowspan=1 colspan=1>0.346</td><td rowspan=1 colspan=1>0.339</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.128</td><td rowspan=1 colspan=1>0.105</td><td rowspan=1 colspan=1>0.103</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.104</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.096</td><td rowspan=1 colspan=1>0.095</td><td rowspan=1 colspan=1>0.091</td></tr><tr><td rowspan=3 colspan=1>Code-DKTLog</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.684</td><td rowspan=1 colspan=1>0.720</td><td rowspan=1 colspan=1>0.728</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>0.722</td><td rowspan=1 colspan=1>0.724</td><td rowspan=1 colspan=1>0.746</td><td rowspan=1 colspan=1>0.748</td><td rowspan=1 colspan=1>0.748</td><td rowspan=1 colspan=1>0.759</td></tr><tr><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>0.832</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.815</td><td rowspan=1 colspan=1>0.819</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>0.858</td></tr><tr><td rowspan=1 colspan=1>loss</td><td rowspan=1 colspan=1>0.471</td><td rowspan=1 colspan=1>0.383</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.371</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.383</td><td rowspan=1 colspan=1>0.360</td><td rowspan=1 colspan=1>0.354</td><td rowspan=1 colspan=1>0.338</td><td rowspan=1 colspan=1>0.316</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.121</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.095</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>0.102</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.092</td><td rowspan=1 colspan=1>0.089</td><td rowspan=1 colspan=1>0.090</td><td rowspan=1 colspan=1>0.085</td></tr><tr><td rowspan=3 colspan=1>TIKTOC</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.760</td><td rowspan=1 colspan=1>0.756</td><td rowspan=1 colspan=1>0.758</td><td rowspan=1 colspan=1>0.766</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.780</td></tr><tr><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.864</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.934</td><td rowspan=1 colspan=1>0.928</td><td rowspan=1 colspan=1>0.934</td><td rowspan=1 colspan=1>0.933</td><td rowspan=1 colspan=1>0.945</td></tr><tr><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.330</td><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.223</td><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>0.223</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>0.219</td><td rowspan=1 colspan=1>0.199</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>0.101</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1>0.067</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.069</td><td rowspan=1 colspan=1>0.068</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.061</td></tr><tr><td rowspan=4 colspan=1>RSSM</td><td rowspan=1 colspan=1>Bal. Acc</td><td rowspan=1 colspan=1>0.719</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.768</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>0.763</td><td rowspan=1 colspan=1>0.769</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.794</td><td rowspan=1 colspan=1>0.812</td></tr><tr><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>0.882</td><td rowspan=1 colspan=1>0.935</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>0.945</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.931</td><td rowspan=1 colspan=1>0.943</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>0.952</td></tr><tr><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0.302</td><td rowspan=1 colspan=1>0.224</td><td rowspan=1 colspan=1>0.210</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.219</td><td rowspan=1 colspan=1>0.217</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>0.199</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>0.185</td></tr><tr><td rowspan=1 colspan=2>Brier    0.094</td><td rowspan=1 colspan=1>0.068</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.062</td><td rowspan=1 colspan=1>0.067</td><td rowspan=1 colspan=2>0.0660.064</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>0.062</td><td rowspan=1 colspan=1>0.054</td></tr></table>

## D LLM-as-Predictor Details

This appendix provides implementation details for the LLM-based predictor. All models share the same data split, filtering, and evaluation protocol described in Section 5.1.

## D.1 Student Split and Temporal Boundary

The LLM predictor uses the same train/test student split as the trained models (Section 5.1). Students are partitioned into train and test sets. The retrieval index contains all of each train student’s submissions across all weeks, plus each test student’s submissions from weeks 1 through the calibration cutoff only. This ensures the LLM has access to the same temporal scope of data as the parametric baselines.

## D.2 Persona Construction

For each test student i, we construct a structured behavioral persona $s _ { i }$ from their calibration-period submissions (weeks 1–3). The persona captures: average time between submissions, average Levenshtein edit distance per attempt, total submission count, precheck-to-submit ratio, per-topic pass rates, code length and line count, and an improvement trend (improving, stable, or declining based on recent vs. earlier pass rates).

Students are additionally classified into one of four behavioral archetypes based on their position in the joint distribution of submission timing, edit size, and volume: rapid iterators (frequent small changes), deliberate planners (moderate pace, large rewrites), careful thinkers (long pauses, few attempts), and balanced (mixed strategy). The archetype label and a natural-language description are included in the system prompt.

## D.3 Retrieval and Summarization

From the student’s calibration history, we retrieve K=5 prior questions as few-shot context using a score that combines content similarity with temporal recency:

$$
\begin{array} { l } { \displaystyle \mathrm { s c o r e } ( q _ { k } ) = ( 1 - \lambda ) \sin ( q _ { k } , q _ { t } ) } \\ { \displaystyle + \lambda \exp \biggl ( - \frac { \ln 2 \cdot \Delta t _ { k } } { \tau } \biggr ) } \end{array}\tag{4}
$$

where sim $( q _ { k } , q _ { t } )$ is the TF-IDF cosine similarity between the candidate and target question texts, $\Delta t _ { k }$ is the time elapsed since the student last attempted $q _ { k } , \tau = 4$ weeks is the decay half-life, and $\lambda = 0 . 5$ . Retrieval is restricted to questions from weeks strictly before the target question’s week, respecting the temporal boundary.

For each retrieved question, all of the student’s attempts are included. To fit within context limits, each trajectory is condensed into a behavioral summary by a smaller LLM (Claude Haiku), describing the student’s problem-solving strategy and common mistakes. The target question is also summarized into a one-sentence description. All summaries are pre-computed in batch before the evaluation loop begins.

## D.4 Prompt Structure

Each prediction uses a two-part prompt. The system message contains: a task framing instruction (predict what this student would submit), the target question summary and item-level statistics (pass rate, average attempts to solve, number of students who attempted), computed from training-set students only, and the student’s persona profile. The user message contains: summaries of the student’s prior approaches on similar problems, the submission format instructions (precheck or submit), the full question text and code template, and (on subsequent attempts) the student’s accumulating attempt trajectory.

## D.5 Prediction Protocol

For each student-question pair, the LLM produces one prediction per attempt in the student’s real trajectory. At attempt n, the LLM generates code conditioned on the student’s trajectory through attempt n−1:

$$
\hat { x } _ { t } ^ { ( n ) } , a _ { t } ^ { ( n ) } \sim p _ { \mathrm { L L M } } \Big ( \cdot \mid s _ { i } , \tilde { \mathcal { H } } _ { i } , q _ { t } , w _ { q _ { t } } , \mathcal { T } _ { t } ^ { ( n - 1 ) } \Big )\tag{5}
$$

where $\hat { x } _ { t } ^ { ( n ) }$ is the generated code, $a _ { t } ^ { ( n ) } \in \begin{array} { r l } \end{array}$ {precheck, submit} is the chosen action, $q _ { t }$ is the target question, and $\mathcal T _ { t } ^ { ( n - 1 ) }$ is the student’s real attempt trajectory through attempt $n { - } 1$ . On the first attempt, $\mathbf { \boldsymbol { T } } _ { t } ^ { ( 0 ) }$ is empty and the LLM sees only the prompt described above. On each subsequent attempt, the context grows to include the student’s prior attempts, each showing the action type (precheck or submit) and pass pattern (e.g., 1010000000), along with Haiku-generated summaries of the student’s code. The LLM does not receive test-case details (inputs, expected outputs, or actual outputs); it sees only the binary pass pattern. The number of predictions per question matches the student’s real attempt count, up to a maximum of 10. The generated code is compiled and graded:

prechecks are evaluated against public test cases only, while submissions are graded against the full test suite.

We evaluate three predictor models, all with frozen parameters and no fine-tuning: Qwen3-14B and Gemma-4-31B, served with vLLM on A100 GPUs, and Claude Opus 4.6, accessed through the Anthropic API. Opus calls use standard inference mode without extended thinking, so the model generates responses directly from the prompt without an explicit chain-of-thought reasoning budget; requests are issued with up to 10 concurrent workers and a 0.2-second inter-request delay to stay within rate limits. Items are processed in chunks of 50, with results saved after each chunk to support resumption.

## D.6 Example Prompt

On the first attempt, the LLM receives the full system and user messages described above: persona, peer and self summaries, question text, and code template. On each subsequent attempt, the prompt accumulates Haiku-generated summaries of the student’s real prior attempts, and the LLM predicts what the student would submit next. For example, on Graph BFS (Q139), the LLM’s predicted submissions across one student’s 4 attempts score pass patterns 1010000000 → 1010000000 → 1010000000 → 1111111111. At attempt 3, the LLM sees summaries of the student’s three prior real submissions before generating its prediction.

The first listing below shows the full initial prompt for BST Single Children, illustrating the system and user message structure. The system message provides the task framing, question metadata, and student persona. The user message provides peer and self approach summaries, the question text, the code template, and submission format instructions.

On subsequent attempts, the prompt accumulates Haiku-generated summaries of the student’s real prior submissions. The second listing below shows this feedback loop for a different student on Graph BFS. At each attempt, the LLM receives the growing trajectory of prior attempt summaries and predicts the next submission. In this example, the LLM produces three identical submissions scoring 2/10, then on attempt 4, after seeing the full trajectory including the student’s successful third submission, generates code that passes all 10 tests.

Example: Initial Prompt with Student Context. The listing below shows the full initial prompt (attempt 1) for BST Single Children. The system message (lines 1–24) provides the task framing, question metadata, and student behavioral persona. The user message (lines 26–end) provides peer approach summaries, the student’s own prior approaches on similar problems, the question text, the C++ code template, and submission format instructions.

--- SYSTEM MESSAGE ---   
You are predicting what a specific university student would   
,→ submit for a C++ programming assignment. Your code   
,→ should authentically reflect this student's ability   
,→ level, coding style, and typical mistake patterns.   
,→ Study the student profile and peer evidence to   
,→ calibrate your response.   
=== QUESTION: BST - Single Children ===   
The student must implement a method to identify and return all   
,→ nodes in a binary search tree that have exactly one   
,→ child (either left or right, but not both), using   
,→ tree traversal on the BST data structure.   
Topic: Exam 78% of students pass all tests on this problem.   
,→ Average 2.4 attempts to solve. 327 students attempted   
,→ this question.   
=== STUDENT IDENTITY ===   
Background: a Data Structures and Algorithms student in a   
,→ Regular section who has attempted 78 distinct   
,→ problems with 325 total submissions.   
Behavioral Style: Spends significant time between submissions.   
,→ Makes fewer total attempts. Plans solutions   
,→ carefully before coding.   
- Average time between submissions: 15.0 hours   
- Average code change per attempt: 192 characters (Levenshtein   
,→ )   
- Precheck vs Submit ratio: 31% Precheck, 69% Submit   
- Tendency: Rewrites large portions of code   
Overall pass rate: 18%   
Topic proficiency: Sorting (Easy): 67%, Queue: 31%, Array List:   
,→ 29%, Exam: 29%, Singly Linked List: 28%, Sorting (   
,→ Advance): 24%, Binary Search Tree: 19%, Binary Tree:   
,→ 19% (and 5 more topics)   
Typical code length: 732 characters (10 lines), concise style.   
,→ Attempts per problem: 4.2 on average.   
Embody this student fully. Your coding ability, problem-  
,→ solving approach, and submission behavior should be   
,→ consistent with the profile above.   
--- USER MESSAGE ---   
=== How Other Students at a Similar Level Approached This   
,→ Problem ===   
Student 1: The student took 3 attempts to solve this BST   
,→ problem, using a recursive helper function strategy   
,→ to count nodes with exactly one child. Their initial   
,→ attempt had a function name mismatch (calling "   
,→ Recursive" instead of "RecursiveChild"), which they   
,→ corrected in attempt 2. The logic itself was sound--   
,→ identifying nodes with only a left or only a right   
,→ child, incrementing the counter, and recursively   
,→ continuing traversal--and they eventually solved it   
,→ successfully on their final submission.   
Student 2: The student took 9 attempts to solve this problem,   
,→ struggling primarily with syntax errors (inconsistent   
,→ pointer notation like \`pLeft\` vs \`left\`, and typos   
,→ like \`single Chile\`) in the early attempts. After   
,→ fixing the syntax issues by attempt 4, they then made   
,→ a logical error in attempt 5 by removing the count   
,→ of the current node in the base case, which they   
,→ partially corrected in attempt 8 but the fundamental

,→ algorithm remained flawed--they were still counting solution that correctly reverses alternate levels by   
nodes with both children in the final return pre-allocating a level vector and writing values at   
statement Despite submitting twice with what reversed indices   
,→ appeared to be incomplete solutions (marked   
"1111111111"), they eventually solved it, though the 3. [BST - Subtree in range] The student must implement a   
trajectory suggests they may have arrived at the ,→ function to count or find all nodes in a Binary   
,→ correct approach through debugging or realization ,→ Search Tree whose values fall within a specified   
,→ range [min, max] using BST properties to efficiently   
,→ toward the single-child total. ,→ prune branches outside the range.   
Student approach: This student demonstrates a \*\*post-order   
Student 3: The student took 6 attempts to solve this problem, ,→ recursive traversal strategy\*\* with immediate node   
,→ initially using an iterative stack-based in-order ,→ deletion when out-of-range values are encountered.   
traversal approach that failed all test cases, After ,→ However, they fail to iterate or debug between   
,→ four unsuccessful attempts with the same iterative ,→ attempts--submitting identical code twice despite the   
,→ strategy (possibly with minor debugging attempts), ,→ first attempt failing precheck--suggesting they   
,→ they pivoted to a recursive approach in attempt 5, ,→ i h did ' d d h f db k l k d   
,→ which successfully passed all test cases. The key   
,→ insight was recognizing that a recursive solution ,→ their logic (recursively trimming invalid branches)   
,→ checking each node's children and recursively ,→ wasn't producing correct results.   
,→ counting single-child nodes throughout the tree was   
,→ the correct strategy, whereas the iterative in-order 4. [BST - Range Count] The student must implement a function   
,→ traversal logic was fundamentally flawed for this ,→ that counts the number of nodes in a Binary Search   
,→ problem. ,→ Tree whose values fall within a given range [low,   
,→ high] using recursive tree traversal.   
Student 4: The student solved the problem in a single attempt. Student approach: This student demonstrates a \*\*solid   
,→ Their approach was to recursively traverse the ,→ algorithmic understanding\*\* of BST range queries with   
,→ entire binary search tree, checking each node to ,→ a correct recursive strategy that prunes branches   
,→ determine if it had exactly one child (either left or ,→ based on BST properties (exploring right subtree when   
,→ right, but not both), and accumulating a count of value is too low left subtree when too high)   
,→ such nodes. The solution was correct on the first try   
,→ , successfully identifying and counting all nodes   
,→ with a single child through a straightforward ,→ investigating why the precheck failed, suggesting   
,→ recursive traversal. ,→ they either misread the feedback or lacked a   
Student 5: The student took 4 attempts to solve this problem rather than iterating on the logic itself   
,→ using a recursive approach that counts nodes with   
,→ exactly one child by checking if each node has only a 5. [BT - Lowest Ancestor] The student must implement a   
,→ left or right subtree and recursing accordingly. ,→ function to find the lowest common ancestor (LCA) of   
,→ They made a critical logic error by continuing to ,→ two nodes in a binary tree using tree traversal.   
,→ recurse down single-child branches instead of Student approach: \*\*Summary:\*\*   
,→ counting that node and stopping, which caused them to   
count every node in unbalanced paths rather than This student uses a systematic debugging approach,   
,→ just the single-child nodes themselves. Despite ,→ methodically fixing one issue at a time--correcting   
,→ submitting the same flawed code twice (attempts 2 and ,→ parameter mismatches (n1/n2 -> a/b), then struct   
4), they eventually corrected their approach and ,→ field names (key -> val)--but fails to address the   
→ successfully solved the problem. ,→ core algorithmic flaw: the path comparison loop doesn   
,→ 't track actual path lengths, causing it to compare   
Based on this student's ability and the peer evidence above, ,→ uninitialized array values and return incorrect   
,→ produce code consistent with how a student at this ,→ ancestors. Their iterative strategy of fixing syntax/   
,→ level would approach this problem. ,→ naming errors masks the need for a fundamental logic   
,→ redesign.   
=== How This Student Approached Similar Problems ===   
=== Submission Format ===   
[ d k h ll l ] h d You can either [Precheck] (run public tests only, no grade   
,→ implement a function to find the k-th smallest ,→ penalty) or [Submit] (final submission, graded   
,→ element in a Binary Search Tree using in-order ,→ against all tests).   
,→ traversal, which naturally visits nodes in sorted Example:   
,→ order. [Precheck]   
Student approach: This student uses a straightforward \*\* \`\`\`cpp   
→ collect-all-then-index approach\*\* (in-order traversal // your implementation here   
,→ into a vector, then return k-th element), which is   
,→ algorithmically correct but memory-inefficient. Their   
iteration strategy shows minimal actual changes-- === Question ===   
,→ attempts 1-3 are nearly identical, suggesting they Class \`BSTNode\` is used to store a node in binary search tree,   
,→ made only cosmetic adjustments (storing size in a ,→ described on   
,→ variable) rather than debugging a real problem, the following:   
,→ indicating they may not have understood why their   
,→ prechecks failed or simply resubmitted the same logic   
hoping it would pass.   
class BSTNode {   
2. [BST - Alternative Level Traversal] The student must public:   
implement a function to traverse a binary search tree int val;   
and return nodes alternating between left-to-right BSTNode \*left;   
,→ and right-to-left at each level, using a level-order BSTNode \*right;   
,→ traversal approach. BSTNode() {   
Student approach: \*\*Summary:\*\* this->left = this->right = nullptr;   
}   
This student demonstrates a \*\*trial-and-error debugging BSTNode(int val) {   
,→ approach with incremental refinement\*\*: their first this->val = val;   
,→ four attempts struggled with type mismatches (mixing this->left = this->right = nullptr;   
,→ \`BSTNode\`, \`TreeNode\`, and \`BTNode\`), then they   
,→ pivoted to implementing the core algorithm logic ( BSTNode(int val, BSTNode\*& left, BSTNode\*& right) {   
,→ alternating level traversal) starting in attempt 5. this->val = val;   
,→ They systematically tested different reversal this->left = left;   
,→ techniques--using a stack (attempt 5), vector this->right = right;   
,→ insertion (attempt 6), and finally index-based   
};

Where \`val\` is the value of node, \`left\` and \`right\` are the   
,→ pointers to the   
left node and right node of it, respectively. If a repeated   
,→ value is inserted   
to the tree, it will be inserted to the left subtree.   
Also, a static method named \`createBSTree\` is used to create   
,→ the binary search   
tree, by iterating the argument array left-to-right and   
,→ repeatedly calling   
\`addNode\` method on the root node to insert the value into the   
,→ correct   
position. For example:   
int arr[] = {0, 10, 20, 30};   
auto root = BSTNode::createBSTree(arr, arr + 4);   
is equivalent to   
auto root = new BSTNode(0);   
root->addNode(10);   
root->addNode(20);   
root->addNode(30);   
\*\*Request:\*\* Implement function:   
\` int singleChild(BSTNode\* root);   
Where \`root\` is the root node of given binary search tree (   
,→ this tree has   
between 0 and 100000 elements). This function returns the   
,→ number of single   
children in the tree.   
\*\*More information:\*\*   
\- A node is called a \*\*single child\*\* if its parent has only   
,→ one child.   
\*\*Example:\*\*   
Given a binary search tree in the following:   
![singleChild](https://e-learning.hcmut.edu.vn/draftfile.php   
,→ /177981/user/draft/713969830/singleChild.png)   
There are \`2\` single children: node \`2\` and node \`3\`.   
\_Note: In this exercise, the libraries\`iostream\` and \`using   
,→ namespace std\` are   
used. You can write helper functions; however, you are not   
,→ allowed to use   
other libraries.\_   
=== Code Template ===   
#include <iostream>   
using namespace std;   
#define SEPARATOR "#<ab@17943918#@>#"   
class BSTNode {   
private:   
void addNode(int val);   
public:   
int val;   
BSTNode \*left;   
BSTNode \*right;   
BSTNode() {   
this->left = this->right = nullptr;   
};   
BSTNode(int val) {   
this->val = val;   
this->left = this->right = nullptr;   
}   
BSTNode(int val, BSTNode\*& left, BSTNode\*& right) {   
this->val = val;   
this->left = left;   
this->right = right;   
}   
static void printPreorder(BSTNode\* root);   
static void deleteTree(BSTNode\* root);

template <typename InputIterator>   
static BSTNode\* createBSTree(InputIterator first,   
,→ InputIterator last);   
};   
void BSTNode::addNode(int val) {   
BSTNode\* trav = this;   
while ((trav->left != nullptr && val <= trav->val) || (   
,→ trav->right != nullptr && val > trav->val)) {   
trav = val <= trav->val ? trav->left : trav->right;   
}   
if (val <= trav->val) trav->left = new BSTNode(val);   
else trav->right = new BSTNode(val);   
}   
template <typename InputIterator>   
BSTNode\* BSTNode::createBSTree(InputIterator first,   
,→ InputIterator last) {   
BSTNode\* root = nullptr;   
for (; first != last; first++) {   
if (root == nullptr) root = new BSTNode(\*first);   
else root->addNode(\*first);   
}   
return root;   
}   
void BSTNode::deleteTree(BSTNode\* root) {   
if (!root) return;   
deleteTree(root->left);   
deleteTree(root->right);   
delete root;   
}   
void BSTNode::printPreorder(BSTNode\* root) {   
if (!root) return;   
cout << root->val << " ";   
printPreorder(root->left);   
printPreorder(root->right);   
}   
{{ STUDENT\_ANSWER }}   
int main() {   
{% for TEST in TESTCASES %}   
{   
{{ TEST.extra }};   
{{ TEST.testcode }};   
}   
{% if not loop.last %}cout << SEPARATOR << endl;{% endif   
,→ %}   
{% endfor %)   
return 0;   
}   
You must first choose an action -- either [Precheck] (run   
,→ public tests only, no grade penalty) or [Submit] (   
,→ final submission, graded against all tests).   
Then provide your C++ implementation that will replace the {{   
,→ STUDENT\_ANSWER }} block in the template.   
- Do NOT reproduce any part of the template.   
- Do NOT emit \`int main()\` (it's already declared).   
- Include any needed class definitions.   
IMPORTANT: Your response format must be exactly:   
1. First line: either [Precheck] or [Submit]   
2. Second line: \`\`\`cpp   
3. Your code   
4 Last line:   
No extra characters, whitespace, or text may appear before or   
,→ after.

Example: Feedback Loop Across Attempts. The listing below shows the feedback loop across 4 attempts for Graph BFS. Attempt 1 uses the full initial prompt (not shown). On each subsequent attempt, the LLM receives Haiku-generated summaries of the student’s real prior submissions and predicts the next submission. The trajectory accumulates: attempt 2 sees one prior summary, at-

tempt 3 sees two, and attempt 4 sees three (including the student’s successful submission with pass pattern 1111111111). The LLM’s pass pattern progresses from 1010000000 to 1111111111.  
========== ATTEMPT 1 ==========   
[Submit] pass=1010000000 (2/10 tests passed)   
========== ATTEMPT 2 ==========   
PROMPT   
=== Student's Attempt Trajectory ===   
Attempt 1 [Precheck] ->   
Summary: The student implemented a Graph class with a BFS   
(Breadth-First Search) method that traverses a graph starting   
,→ from a   
given vertex and returns an Adjacency object containing the   
,→ order in   
which vertices were visited. The implementation uses a queue   
,→ to process   
vertices level-by-level and a visited array to avoid   
,→ revisiting   
vertices.   
Now predict what the student would submit next. Match their   
,→ coding   
style and debugging approach.   
--- RESPONSE ---   
[Submit] pass=1010000000 (2/10 tests passed)   
========== ATTEMPT 3 ==========   
--- PROMPT ---   
=== Student's Attempt Trajectory ===   
Attempt 1 [Precheck] ->   
Summary: The student implemented a Graph class with a BFS   
(Breadth-First Search) method that traverses a graph starting   
,→ from a   
given vertex and returns an Adjacency object containing the   
,→ order in   
which vertices were visited. The implementation uses a queue   
,→ to process   
vertices level-by-level and a visited array to avoid   
,→ revisiting   
vertices.   
Attempt 2 [Precheck] ->   
Summary: The student implemented a Graph class with BFS   
(Breadth-First Search) functionality that stores an adjacency   
,→ list   
representation using a custom Adjacency data structure. The   
,→ BFS method   
performs a traversal starting from a given vertex, marking   
,→ visited   
vertices, using a queue to explore neighbors level-by-level,   
,→ and   
returns the traversal order in a new Adjacency object.   
Now predict what the student would submit next. Match their   
,→ coding   
style and debugging approach.   
--- RESPONSE   
[Submit] pass=1010000000 (2/10 tests passed)   
========== ATTEMPT 4 ==========   
--- PROMPT ---   
=== Student's Attempt Trajectory ===   
Attempt 1 [Precheck] ->   
Summary: The student implemented a Graph class with a BFS   
(Breadth-First Search) method that traverses a graph starting   
,→ from a   
given vertex and returns an Adjacency object containing the   
,→ order in   
which vertices were visited. The implementation uses a queue   
,→ to process   
vertices level-by-level and a visited array to avoid   
,→ revisiting   
vertices.

Table 8: Prompt ablation conditions. ✓ = included, ✗ = removed.
<table><tr><td>Condition</td><td>Persona</td><td>Trajectory</td><td>RAG</td></tr><tr><td>Full</td><td>√</td><td>√</td><td>√</td></tr><tr><td>No persona</td><td>x</td><td>√</td><td>√</td></tr><tr><td>No trajectory</td><td>√</td><td>X</td><td>√</td></tr><tr><td>No RAG</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Recent questions</td><td>√</td><td>√</td><td>recency</td></tr><tr><td>Direct solve</td><td>x</td><td>x</td><td>X</td></tr></table>

Attempt 2 [Precheck] ->   
Summary: The student implemented a Graph class with BFS   
(Breadth-First Search) functionality that stores an adjacency   
,→ list   
representation using a custom Adjacency data structure. The   
,→ BFS method   
performs a traversal starting from a given vertex, marking   
,→ visited   
vertices, using a queue to explore neighbors level-by-level,   
,→ and   
returns the traversal order in a new Adjacency object.   
Attempt 3 [Submit] -> 1111111111   
Summary: The student implemented a Graph class with a BFS   
(Breadth-First Search) method that traverses a graph starting   
,→ from a   
given vertex and returns an Adjacency list containing the   
,→ vertices in   
the order they were visited. The implementation uses a queue   
,→ to process   
vertices level-by-level and a visited array to avoid   
,→ revisiting   
vertices.   
Now predict what the student would submit next. Match their   
,→ coding   
style and debugging approach.   
--- RESPONSE ---   
[Submit] pass=1111111111 (10/10 tests passed)

## D.7 Prompt Ablation

We ablate the LLM predictor’s prompt to isolate the contribution of each component. Table 8 summarizes the six conditions. The full pipeline includes the student persona, prior attempt trajectory, and RAG-retrieved examples of the student’s approach on the K=5 most similar problems. Each ablated variant removes one component while keeping the rest. The recent questions condition keeps all three components but selects the K=5 problems by recency of the student’s last submission instead of content similarity. The direct solve condition removes all student-specific information: the LLM receives only the question and code template, solving each problem from scratch with the same iterative feedback loop.

Results for Qwen3-14B are shown in Figure 6 (bottom panel, main text). The full pipeline achieves the highest balanced accuracy. Removing the persona produces the largest drop, with removing the trajectory context close behind, so the student profile and the attempt history together carry most of the conditioning signal. This is compatible with the difficulty-driven pattern in Figure 20: the persona includes the student’s aggregate pass-rate statistics, which can calibrate how often predicted submissions fail without requiring student-specific dynamics. Removing RAG examples produces a smaller drop, and selecting the retrieved problems by recency instead of similarity performs comparably to removing them entirely.

![](images/61cd53478678c9837dc6db5ce99e7b761baa73b015f15038c7f4f9951d375178.jpg)

![](images/d82f3923c23323ffc1c89d82b007bd322f4575073bc1b8fbe58c185d06d5707f.jpg)  
Figure 20: Problem-level pass rate correlations for Opus 4.6 on DSA HK231. Left: student pass rate vs. LLM pass rate with student context $( r = 0 . 4 0 )$ . Right: LLM pass rate without student context vs. LLM pass rate with student context $( r = 0 . 8 0 )$ . The stronger correlation between the two LLM conditions suggests the LLM’s performance on each problem is driven primarily by its own coding ability rather than the student profile.

Figure 20 compares problem-level pass rates between the full and direct-solve conditions. The correlation between the LLM’s pass rate without student context and its pass rate with student context is high $( r = 0 . 8 0 )$ , while the correlation with real student pass rates is lower $( r = 0 . 4 0 )$ . This indicates that the LLM’s problem-level performance is driven primarily by its own coding ability rather than the student profile.

## E Use of Generative AI Tools

The paper has benefited from language refinement and code debugging assistance using generative AI tools, including OpenAI’s GPT family and Anthropic’s Claude 3.5. These tools were used under the authors’ supervision to improve clarity, phrasing and to identify issues during the development of the work. All conceptual contributions, analyses and interpretations remain the responsibility of the authors.