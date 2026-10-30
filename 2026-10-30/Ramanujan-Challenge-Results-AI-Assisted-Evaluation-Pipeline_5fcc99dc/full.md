# Ramanujan Challenge Results: AI-Assisted Evaluation Pipeline for Multi-Format Proofs

Alan Sorani<sup>1</sup>, Ashvni Narayanan<sup>1</sup>, Michael Shalyt<sup>1</sup>, Tali Monderer<sup>1</sup>, Rotem Kalisch<sup>1</sup>, Hila Barkan<sup>1</sup>, Elyasheev Leibtag<sup>2</sup>, Shachar Weinbaum<sup>1</sup>, Carsten Schneider<sup>3</sup>, John Campbell<sup>4</sup>, Ido Kaminer<sup>1</sup>

<sup>1</sup>Ramanujan Machine Group, Technion, Haifa, Israel.

<sup>2</sup>Department of Mathematics, Computer Science and Statistics, Ghent University, Ghent, Belgium. <sup>3</sup>Research Institute for Symbolic Computation, Johannes Kepler Universitat Linz, Linz, Austria. ¨ <sup>4</sup>Department of Mathematics and Statistics, Toronto Metropolitan University, Toronto, Canada.

## Abstract

As AI’s mathematical capabilities rapidly improve, meaningful benchmarks are shifting from university level problems and math olympiads to research-level mathematics. The Ramanujan Challenge is a benchmark comprised of research-level problems in number theory, including unpublished results and open conjectures. It presented concrete formulas for mathematical constants whose correctness can be checked numerically, while their proofs may require non-obvious mathematics. Here, we analyze the Challenge submissions, measuring the relative success of different AI models, Computer Algebra Systems (CAS), and formal tools. We find correct submissions that are conceptually distinct from the official solutions, and a proof of one of the published open conjectures. Evaluating the submissions exposed a second challenge: judging complex proofs reliably at scale. We introduce an AI-assisted evaluation pipeline for natural language proofs, CAS code, and Lean formalizations that assesses semantic and logical correctness. Compared with the conventional one-shot LLM judgment, our pipeline reduces false-positive verdicts on natural language proofs from 23% to 6% and improves judgment accuracy on supporting code from 46% to 58%. This approach provides a methodology for evaluating research-level mathematics at scale, with potential applications across mathematics and AI for science.

## 1 Introduction

Benchmarks measuring the mathematical capabilities of AI systems are getting saturated at an accelerating pace. School-level problem sets such as GSM8K and MATH (Cobbe et al., 2021; Hendrycks et al., 2021) were followed by university-level benchmarks, such as U-Math and ASyMOB (Chernyshev et al., 2025; Shalyt et al., 2025), and math olympiad questions (Zheng et al., 2022; He et al., 2024; Gao et al., 2025). The frontier of LLM evaluation is now shifting towards research-level mathematics (Glazer et al., 2025; Phan et al., 2026; Abouzaid et al., 2026; Peyronnet et al., 2026). Meaningful assessment at this level requires a rare combination: problems at the research frontier whose solutions are unpublished to avoid training-data contamination - but are known privately.

In this work, we present the results of the Ramanujan Challenge for AI (Shalyt et al., 2026), a collection of ten research-level numerically verifiable number theory problems. Of these, 8 were proved by the authors but unpublished, and 2 are open conjectures. The collection was released as a benchmark for AI-assisted proving. Our analysis shows that the submissions which were most successful at producing correct natural language proofs used models from the Claude and GPT families, while such proofs from submissions combining Lean with CAS outperformed those that relied on any single tool, suggesting benefits from diversified supporting evidence.

Notably, some of the complete proofs were submitted by amateur mathematicians, and one of the open conjectures had multiple correct proofs submitted, showing the potential of advanced AI to lower the barrier of entry to parts of mathematical research. Submitters to the challenge, who obtained correct solutions include a person with no higher mathematical education and whose solution to one of the challenge problems was found to be simpler than the original solution from the authors. They also include young independent researchers with no mathematical training, one of whom solved a famous open mathematical problem prior to their participation in the challenge, and a high-school student who achieved significant results towards proving one of the challenge conjectures. The use of AI thus allows people accross broader backgrounds to access and advance research-level mathematics, democratizing mathematics and opening the door to cross-field research and contributions.

Obtaining the results required addressing a second challenge: evaluating large quantities of complex mathematical proof candidates. The recent boom in AI-assisted mathematics (Alon et al., 2026; Feng et al., 2026; Ju et al., 2026), while producing some incredible results (OpenAI, 2026; Alpoge, 2026), created an¨ avalanche of mathematical claims – making this problem widespread beyond benchmarks. The Ramanujan Challenge illustrates this scale: participants sent a total of 228 submissions, containing a total of 4802 pages of natural language mathematical proofs and 1,079,380 lines of code across Lean, Python, Mathematica and Sage. Recent mathematical benchmarks approach automated validation in different ways: fully relying on a large language model (LLM) as a judge (Zhang et al., 2025), using an LLM to check equivalence against a known ground truth (Peyronnet et al., 2026), or using deterministic code to verify LLM-generated answers (Shalyt et al., 2025). Other benchmarks, such as IMO-GradingBench and µ- Math (Luong et al., 2025; Chernyshev et al., 2025), evaluate LLMs as judges of mathematical proofs and solutions, showing the difficulty of achieving high agreement with expert assessments.

In this paper, we present a novel AI-assisted pipeline, depicted in Figure 1, for the purpose of large scale validation of complex multi-format proofs, which include natural language, CAS code, and Lean formalizations. We assess the Ramanujan Challenge submissions, comparing verdicts across three evaluation methods: our pipeline with and without a human expert in the loop, and a baseline LLM-as-a-judge. Using the evaluation pipeline reduces false positive grading by a factor of 4 for natural language proofs and 11.5 for CAS/Lean code, and increases agreement with human evaluation of the code from 46% to 58%, relative to a baseline of the same LLM without the pipeline.

![](images/13f6d8f387d56b1b110616c86502087cb3b5ceeee321b9de76c7c0462ed36c8b.jpg)  
Figure 1: Overview of the evaluation pipeline.

Broadly, as AI systems produce mathematical and scientific results faster than experts can verify them, validation is becoming a central bottleneck (Tao, 2026; Kontorovich, 2025; Codreanu et al., 2026).

Our approach suggests combining natural language reasoning with complementary computational certificates, helping scale validation of AI-generated claims across mathematics and science.

## 2 Related Work and Novelty

Related work. Checking how well LLMs can reason is a relatively new area of interest. Last evaluated in December 2025, the Frontier Math benchmark (Glazer et al., 2025) considered reasoning capabilities of LLMs across multiple domains of research level mathematics. They concluded that the best models of the time were insufficient for advanced research, “revealing a vast gap between AI capabilities and the prowess of the mathematical community”. The First Proof (Abouzaid et al., 2026) benchmark focuses on unpublished research problems whose answers are known to experts. These works attest to the high cost of human feedback, and the importance of monitoring the progress of AI as a research assistant. One of the aims of this manuscript is to answer this question for current models.

The other goal is to develop an AI-assisted evaluation pipeline for given solutions. The authors of First Proof stress on the difficulty of manual evaluation when there is a large volume of submissions for complex problems. There are fewer works on the subject of using AI as graders. Using LLMs as teaching assistants for undergraduate discrete math examinations was studied in (Sapkota & Murshed, 2026). The paper suggests that AI can apply harsh point deductions under stricter policy conditions. LLM as a grader has also been studied in (Xie et al., 2024), where it is used to evaluate responses to six questions in an examination on an Operating Systems Course. Google’s DeepMind constructed IMO-GradingBench in (Luong et al., 2025), showing that autograders coupled with reasoning from Gemini seem to correlate well with human evaluation on IMO problems. (Yu et al., 2026) focuses on evaluation of handwritten college mathematics using AI.

Novelty. To the best knowledge of the authors, we are the first to construct an AI-assisted evaluation pipeline as well as test it in the context of large scale evaluations of research-level computational mathematics questions. Our evaluation pipeline, when used by an LLM agent, reduces false positive grading by a factor of 4 for natural language proofs and 11.5 for CAS/Lean code, and increases agreement with human evaluation of the code from 45% to 58%. This demonstrates an improvement in the direction of automated AI verification of natural language complex mathematical proofs.

We further contribute to the analysis of CAS as means of proof or verification for such mathematical questions, an unexplored area of research. We analyze in particular the combined use of CAS and Lean and see that it is accompanied by correct natural language proofs more so than the use of either one in isolation, opening a door to further research on the interconnection of CAS and formal theorem assistants in AI mathematics.

## 3 The Trust Spectrum

Our proposed paradigm for measuring the validity of solutions to mathematical questions is the “trust spectrum”. While theoretically most mathematical proofs can be completely formalized, in practice, trust in a mathematical proof lies on a spectrum: soft verification - such as numerical validation or probabilistic checking; intermediate verification - symbolic validation by a computer algebra system (CAS); and hard verification - machine-checked theorem proving like with Lean.

CAS rely on symbolic representations and computational rules rather than axiomatic representations of mathematical objects. They can include data produced by otherwise verifiable mathematical statements, which can be used in subsequent reasoning and proofs in the common sense of the word (Barendregt & Wiedijk, 2005). Provability, however, is not always formally decidable via CAS.

ITPs such as Lean enforce strict rules for line-byline verification of proofs. They can produce dependency graphs of the proof on other theorems, going right down to the used definitions and axioms. For this reason, we place formal verification the highest on our trust spectrum. Moreover, for a human to verify a formal proof, one needs to inspect only the definitions, used axioms, and problem statement, significantly reducing verification time.

In this paper, one of our aims is to measure how well AI systems can evaluate a given solution, using the trust spectrum as a metric. This view of trust creation in AI

![](images/d43bd43876233081529165f5becdc89633f80b980218b28006a1c0db7450f34f.jpg)  
Figure 2: The trust spectrum.

results can also generally help decide when to invest substantial time exploring AI outputs, and which parts of a new claim to focus on.

## 4 The Challenge

The Ramanujan Challenge focuses on explicit formulae involving important mathematical constants such as π, e, ζ(s) etc. The form they take are of recurrences, continued fractions, series, and integrals. Most of these formulae were discovered by exploring trajectories arising in Conservative Matrix Fields, using the ramanujantools package (Ramanujan Machine Group, 2025; Elimelech et al., 2024; Weinbaum et al., 2025; Raz et al., 2025).

The Ramanujan Challenge, published on arXiv (Shalyt et al., 2026) and on the Ramanujan Machine website (The Ramanujan Machine, 2026), ran from July to August 2026. It comprised of 8 questions and 2 conjectures, listed in Appendix A and Appendix B respectively. The solutions to these questions were (to the knowledge of the authors) previously unpublished.

The challenge asked for solutions comprising of the following data, in descending order of importance: formal proof, CAS-based derivations and natural language proofs. The reason for this order of importance is discussed in §3.

At the time of posting the Challenge, the prevalent LLMs could not solve any of the problems (with one-shot prompts). The challenge received submissions from people within and outside the mathematics community, many of which did succeed in solving some of the problems.

## 5 Methodology

In this section, we discuss the construction of our AI-assisted pipeline, described in Figure 1. The final goal of this pipeline is to reduce human time and effort invested into largescale grading of proofs that require multiple formats.

## 5.1 Evaluation Pipeline

For a submission, the pipeline provides a human reviewer with AI-generated summaries for each file in the submission, along with an AI-generated review connecting all of them.

The major components of the pipeline can be broken down into:

Step 1: Basic Grade Files. Every Python script is executed and given a “basic grade” using our custom script. This is passed as input in the next step. These files comment on the successful compilation of the given code, as well as summarize the script’s structure (comment density, imports, cyclomatic complexity and a keyword-based symbolic/numeric classification) and its observed behavior under a fixed runtime environment (exit status, execution time and standard output).

Step 2: AI Summary. For every submission, two AI summary files are generated, one for the natural language proof and one for the code.

Natural language summary. A GPT 5.6-Sol agent with high reasoning effort reads each PDF file in the submission folder and produces a summary file. The file contains a summary of the proof, a list of issues, and (code) files mentioned in the proof.

CAS Summary. A Claude Opus 5 agent with high reasoning effort reads the CAS/Lean files in each submission folder as well as the basic grade files when present. It produces a file containing a summary of the code, a list of issues, and a list of files that pertain to the verification of the proof.

Note that only relevant files are presented to the agents, not the entire submission folder. This is done to avoid contamination of ideas between the natural language and CAS/Lean segments of the proof. This is done to have a more thorough check of semantic accuracy.

Step 3: Summary Comparison. A GPT 5.6-Sol agent with medium reasoning effort compares the summary files produced in the previous step. It generates a new file listing the similarities between the natural language proof and the code, a list of conflicts, and a list of common issues derived from the two summary files.

Step 4: Human Evaluation. A human can now evaluate the submission with the aid of the summary file(s) generated above.

Table 1: Agreement of the LLM agent with human verdicts without using our pipeline (via one-shot prompting) and with the pipeline.  
(a) Agreement between human evaluations and those from Claude Opus 5 across submissions to the challenge questions, for natural language proofs and for CAS/Lean code, with one-shot prompting or with the evaluation pipeline.
<table><tr><td rowspan="2">Verdict comparison</td><td colspan="2">Natural language proof</td><td colspan="2">CAS / Lean</td></tr><tr><td>Without pipeline</td><td>With pipeline</td><td>Without pipeline</td><td>With pipeline</td></tr><tr><td>Agreement</td><td>65.7%</td><td>66.4%</td><td>45.8%</td><td>58.3%</td></tr><tr><td>Claude stricter than human</td><td>2.9%</td><td>15.3%</td><td>4.2%</td><td>29.2%</td></tr><tr><td>Human stricter than Claude</td><td>31.4%</td><td>18.2%</td><td>50.0%</td><td>12.5%</td></tr></table>

(b) Submissions the human evaluator graded Partial or Major Issues that Claude’s verdict nonetheless called Correct, for natural language proofs and for CAS/Lean code, as a share of all graded submissions (137 proofs; 120 CAS/Lean, excluding submissions with no CAS artifact).
<table><tr><td rowspan="2"></td><td colspan="4">Natural language proof</td><td colspan="3">CAS / Lean</td></tr><tr><td></td><td>Number of submissions</td><td>Without pipeline</td><td>pipeline</td><td>WithNumber of 」 submissions</td><td>Without pipeline</td><td>With pipeline</td></tr><tr><td>Human verdict Partial</td><td>30 1</td><td></td><td>23</td><td>7</td><td>65</td><td>39</td><td>4</td></tr><tr><td>Major Issues</td><td>I 53</td><td>9</td><td></td><td>I 1 1</td><td>30</td><td>7</td><td>0</td></tr><tr><td>Total</td><td>83 (60.6%) 1</td><td>32 (23.4%)</td><td></td><td>8 (5.8%)</td><td>95 (79.2%)</td><td>46 (38.3%)</td><td>4 (3.3%)</td></tr></table>

## 5.2 Evaluation Criteria

We detail the evaluation criteria by format. Each submission is assigned a rank for its correctness and completeness. Correct and complete proofs or verification codes have been assessed as Correct. Partial ones that demonstrate a meaningful step towards a proof, those with a derivation gap that is not central to the entirety of the proof, or code verification for only part of the proof, have been all classified as Partial. Others with major issues or extra assumptions for the formal proofs have been classified as having Major Issues.

Natural language proofs. Evaluation of natural language proofs is the most self-explanatory. Correct solutions are those that comprise the mathematical steps necessary to reach the conclusion of the statement at hand, with no arrival at statements without proof, except for steps that follow simply from standard methods within the field. This is, as all human verification of natural language proofs, susceptible to human error, as discussed in §3.

CAS. CAS code has been evaluated in the same way as a natural language proof. To be classified as correct, it should reproduce all of the steps necessary for a mathematical proof. Code that verified only part of the statements of the natural language proof, or had a small number of seemingly recoverable errors has been deemed partial, and other code, or code that does not compile has been labeled as having major issues.

Lean. Lean was the only ITP used by the participants of the challenge. Its evaluation is much simpler than that of natural language proofs and CAS code. It requires only a semantic inspection of the definitions relating to the problem statement and object, and a check that the code compiles successfully, in addition that the code does not use statements without formal verification via the use of sorry or axiom statements.

Lean code that does not compile or where the definitions don’t match the problem have been classified as having major issues. Code that does compile and matches a significant partial step towards a proof, or code that imports known theorems in the field without formal proof via the use of sorry or axiom statements have been classified as partial.

## 6 Dataset

In this section, we explain the various quantitative aspects of the submissions received as part of the Ramanujan Challenge.

The Ramanujan Challenge received 228 submissions.

207 of the submissions clearly relate to a specific challenge problem, with 167 including solutions to the questions and the other 40 to the conjectures. Of these 207, 39 are older versions of later submissions to the same challenge problems by the same submitter. These are ignored in our analysis so that submissions with very little variation in their solution do not skew the data. We do not refer to these hereon.

Participants of the challenge used AI models, CAS and ITPs as listed in Figures 3 and 4 respectively, with none using any ITPs other than Lean.

![](images/3f178006819f3ff564bfe4ba7f5964ad0d0c0d41c5ac23b8ca1a089fb8399f5e.jpg)  
Figure 3: AI models used by the submissions, in percentages out of the total number of submissions, with data from the 61% of submissions which reported their AI tools.

All submissions answering challenge questions supplied a natural language proof, with 88% of them also including CAS or Lean code, as seen in Figure 4. 53 submissions to the challenge questions, and 4 to the conjectures, did not report the AI tools used in their submissions. We believe that submissions that did not specify the use of AI tools did not avoid their use, but rather only omitted the specification of said tools. We analyze these submissions as part of the data, with specific mention where it is excluded.

4 participants had only submissions that don’t relate to a specific challenge problem. 9 participants submitted only 1 solution while 6 submitted solutions to all problems. The conjectures received 20 and 11 solutions respectively, with 10 people who submitted to both of them.

The type of formality of the solutions is detailed in Figure 5. A solution was classified as having the lowest type of formality that it relied on, according to §3, other than “Human” which was only given to solutions which included only a natural language proof, as the proofs are given a separate verdict from the code.

![](images/540d2b4383fa3210bdaecdf07d09bd7938dfc581d1055865818eb94dac5f4d13.jpg)  
Figure 4: Computer Algebra Systems or Lean used by the submissions, in percentages out of the total number of submissions. Solutions to the cha

![](images/32a9b0633c245deaba69220577a48c971c9acbc12c275106887e6a89d74d837c.jpg)  
Figure 5: Type of formality of the submissions, in percentages out of the total number of submissions.

## 7 Results

We now present the results of evaluating the submissions to the Ramanujan Challenge. In §7.1, we detail our primary quantitative findings, including evaluation of the correctness of the solutions, similarity metrics and novelty analysis. §7.2 details our observations regarding the open challenge questions. §7.3 details the successes of our AI-assisted evaluation pipeline.

Table 2: Human evaluation of the submissions, with each submission receiving a trinary grade for its natural language proof, CAS and Lean code, when present.  
Natural language proof Computer Algebra System/Lean
<table><tr><td rowspan="2">Question</td><td rowspan="2">Number of submissions</td><td colspan="3">Major</td><td colspan="3"></td><td rowspan="2">Major No code</td></tr><tr><td>Correct</td><td>Partial</td><td>issues</td><td>Correct</td><td>Partial</td><td>issues</td></tr><tr><td>1</td><td>23</td><td>11 1</td><td>6</td><td>6</td><td>7 I</td><td>11</td><td>3</td><td>2</td></tr><tr><td>2</td><td>16</td><td>7</td><td>3</td><td>6</td><td>1 2</td><td>8</td><td>4</td><td>2</td></tr><tr><td>3</td><td>19</td><td>10</td><td>3</td><td>6</td><td>I 5</td><td>7</td><td>5</td><td>2</td></tr><tr><td>4</td><td>19</td><td>4</td><td>7</td><td>8</td><td>I 1 I</td><td>13</td><td>3</td><td>2</td></tr><tr><td>5</td><td>11</td><td>2</td><td>2</td><td>7</td><td>I 0</td><td>5</td><td>4</td><td>2</td></tr><tr><td>6</td><td>20</td><td>10</td><td>5</td><td>5</td><td>1 6</td><td>9</td><td>2</td><td>3</td></tr><tr><td>7</td><td>15</td><td>6</td><td>2</td><td>7</td><td>I 4 I</td><td>6</td><td>3</td><td>2</td></tr><tr><td>8</td><td>14</td><td>4</td><td>2</td><td>8</td><td>I 0</td><td>6</td><td>6</td><td>2</td></tr><tr><td>Total</td><td>137</td><td>54</td><td>30</td><td>53</td><td>1 26</td><td>64</td><td>30</td><td>17</td></tr></table>

## 7.1 Analysis of the Solutions to the Questions

Table 2 and figure 10 show the verdicts and tool-based success rates in submissions to the questions, as measured by the human reviewer separately for the natural language proof and for the verification code.

Of the reported AI tools, GPT/Codex achieved highest correctness for natural language proofs, while a combination of Claude and GPT or Codex achieved highest correctness for CAS/Lean code.

Submissions including only natural language proofs achieved the highest correctness for natural language proofs; these were produced by 4 different submitters. This is followed by submissions using Python in combination with Lean, and highest code correctness was achieved by Lean alone. In general, natural language proofs in submissions that included both CAS and Lean code obtained the highest correctness across natural language proofs, and those including Lean code alone had the highest percentage of code correctness, both seen in Figure 6.

![](images/378ba9ca6d35f1f687db3b5be602a37785930ba80bf6539994be3e4115cc9eb7.jpg)  
Figure 6: Correctness of natural language proofs and CAS derivation or Lean proofs, according to human evaluation. CAS code was all found to be at most partial and did not include comprehensive verification or proof of the problem statements, explaining the 0% on the first column.

Viewing correctness by level of formality in Figure 7 reinforces the usefulness of different levels of formality; Lean formalizations alone achieved correctness in code, but symbolic verification accompanying the most correct natural language proofs. Among solutions with code, those with symbolic verification had the highest natural language correctness of 52%, above those including formal verification at 38% and well above the 6% of numerical checks. The highest rate belongs to the 17 solutions without code. We note that 15 of these come from only two submitters, reflecting more on those submitters than on the level

![](images/290ade49658ca238913720e77e49f779426bd3879cd324ebfc88331260487227.jpg)  
Figure 7: Correctness of natural language proofs and CAS/Lean code based on their level of formality. Code which did not include a Lean formalization did not make complete verification or proof and no such code was judged as correct.

![](images/761a076c51a850855b6bba1e6c58dde10f1054a9468b338a135f696c450ce821.jpg)  
Figure 8: Trust level in the solutions to the challenge questions, in percentages.

Some Lean formalizations relied on certain statements without proof, via use of statements such as axiom, constant, sorry, or admit. Correctness of such formalizations, and the natural language proofs in their submission, are both significantly reduced by the use of these statements, as can be seen in Figure 9. None of the code including formalizations with these statements was judged to be correct, compared to 58% of code in the rest of submissions which included Lean code. We see also that only 11% of natural language solutions accompanying such formalizations were found to be correct, with a higher 49% for natural language proofs coming with Lean formalizations not including such statements. This demonstrates a correlation between correctness of the natural language proof and completeness of the Lean formalization.

Despite the advantage in the use of Lean for achieving a correct solution in either natural language or code, the first correct solutions to 3 of the challenge questions included a correct natural language proof, with no Lean formalization. The first correct solutions to challenge questions 3, 6 and 7 did not include Lean formalizations. Still, the first correct solutions to 5 of the questions did include correct Lean formalizations, in spite of the lower overall percentage of such solutions (which can be seen in Figure 8).

This figure also shows that publicly available AI tools at the time of the challenge still struggle to consistently produce correct, semantically verifiable solutions and formalizations for the challenge problems.

![](images/a2ceee47a3d86671d0f62a46ef1f1cfb565abb3518af406173246a87579c7cc9.jpg)  
Figure 9: Correctness of natural-language proofs and code for the submissions that used Lean, split by whether their submitted Lean source contains axiom, constant, sorry, or admit.

## 7.2 Solutions to the Conjectures

We present here the results regarding the two conjectures of the challenge, which are described in Appendix B. Conjecture 1 received solutions including a correct Lean-formalized proof and multiple other proofs that were deemed likely correct after discussion with the problem author. Among the evaluated submissions, Conjecture 2 received only partial solutions, and it remains open. We discuss these in more detail.

Due to scope and time constraints, the solutions to the conjectures have not yet been fully evaluated. Their evaluation will be included in the coming weeks.

The solutions to Conjecture 1 that were evaluated were those presenting a solution in Lean, and partially those which only gave a natural language proof. Discussion with the author of the problem suggests that multiple solutions show good directions towards a proof. Evaluation of the Lean files for this problem shows one correct and complete formal proof, with two of the other three solutions having major issues and not formalizing the theorem statement, and one providing proof only for partial results.

Evaluation of the earlier solutions to Conjecture 2 found only partial solutions or those with major issues, with all solutions admitting to only attempt proving partial results.

## 7.3 Analysis of the Evaluation Pipeline

We analyze the evaluation pipeline as a method of increasing performance of LLM agents in the correct classification of challenge solutions. A Claude Opus 5 agent was used twice, once via one-shot prompting and once with the use of our evaluation pipeline, to classify the parts of the solution as correct, partial or with major issues. These were compared to human verdicts, which were considered the reference labels, with results detailed in Table 1.

Use of the pipeline increased overall agreement with the human verdict from 65.7% to 66.4% for natural language proofs and from 45.8% to 58.3% for code.

Without the pipeline, the agent was too strict on only 2.9% of solutions when using one-shot prompting in the evaluation of natural language proofs, and on only 4.2% of them for CAS or Lean code. It was too lax on 31.4% of natural language proofs and 50.0% of code, ignoring issues in them. 23.4% of the natural language proofs, and 38.3% of the code had been incorrectly classified as Correct.

These issues have been corrected with use of the pipeline, with reduction from 31.4% to 18.2% for natural language and 50.0% to 12.5% for code in the agent being too lax, and reduction from 23.4% to 5.8% for natural language and from 38.3% to 3.3% for code in false positives, less by factors of 4 and 11.5 respectively.

## 8 Limitations

There is uncertainty in the experiment results with regards to variation between participants and tools used. Challenge participants submitted solutions to as low as 1 and as high as all 10 challenge problems, potentially skewing the dataset. The participants were furthermore self-admitted to the challenge and chose and self-reported their use of tools for the solutions, introducing potential biases.

Evaluation of the submissions also leads to an amount of uncertainty, as they rely on human judgment, which is prone to error. Due to time and monetary constraints, evaluation of each natural language proof and CAS or Lean code was done by a single human evaluator, as described in §5 and without cross-checking. This leaves evaluation prone to human error and potentially skewed by the specific backgrounds of the evaluators.

## 9 Discussion and Outlook

As AI-assisted mathematics produces results at an increasing rate, reliable and scalable validation is becoming crucial. Our analysis shows that a structured evaluation pipeline can improve judgment quality even when the underlying LLM is unchanged. We expect this advantage to persist as models improve, particularly if evaluation pipelines evolve in parallel to model improvement.

![](images/98cfabf710396b4f13ec7b6f6aa313044afeb34ac1e5e73aaa1bf88dfeebe3c8.jpg)

The Ramanujan Challenge results show that research-level number theory problems are increasingly within reach of AI-assisted solvers. Benchmarks at this level may therefore saturate as well, just as earlier benchmarks have, motivating

Figure 10: Correct and Partial verdicts for submissions to the challenge questions, by tool combination, for combinations that appeared in at least 5 submissions.

a continuous effort to construct up-to-date evaluations that track the advancing research frontier. Open conjectures offer a renewable source of such problems, but because they lack known proofs, their use as benchmarks depends on reliable evaluation methods that do not require a priori ground truth to work well, such as the approach presented here.

Our analysis shows that natural language proofs supported by both Lean and CAS code were, on average, more successful than those relying on a single tool. One plausible explanation is that the two forms of evidence are complementary: Lean guarantees the logical validity of the statement as formalized, whereas CAS can efficiently check intricate symbolic and numerical computations that are hard to formalize. Together, they can therefore provide stronger evidence than either alone. This suggests a broader role for CAS as a supporting tool for mathematical proofs. CAS certificates are often more readable than full formalizations while remaining machine-checkable, and tighter integration among CAS, proof assistants, and LLMs could turn verified computations into certified proof steps that combine readability with rigor.

More broadly, the setting considered here – a natural language claim accompanied by code-based indirect evidence – extends beyond number theory. Similar situations arise throughout mathematics and increasingly in varied areas of AI for science, where generated claims may be supported by code, simulations, or data analyses. As the volume of such results grows, systems capable of autonomously assessing the validity of a claim and the sufficiency of its supporting evidence may become increasingly crucial.

## 10 AI Use

Our work relies heavily on the use of LLMs. The components where AI was majorly used include: The dataset. All or most challenge solutions were generated by AI to some extent. A detailed analysis of this is provided in §6. Since these are external contributions, the authors do not have access to the amount of tokens used for all submissions.

Evaluation of solutions. Codex with GPT-5.6-Sol with high reasoning effort and Claude Opus 5 with high reasoning effort were used heavily in the evaluation pipeline. Codex with GPT-5.6 was also used as an assistant during manual human evaluation of the solutions in addressing different questions regarding derivation gaps and proof outlines, with answers only inspected by the human reviewer as part of their own evaluation.

Graphs. Code was generated with the help of AI to analyze and create graphs.

Survey. LLMs were used for the purpose of literature survey.

Grammar and phrasing. Feedback regarding correct grammar and phrasing.

The following was the contribution differentiated based on the models that were used:

Claude. We used a total of about 697,989,732 tokens over Opus 4.8, Opus 5, and Fable 5, with about 268,690,427 for the code summary files, about 132,790,138 for verdicts on the correctness of the solutions, and the rest for interactive sessions. We worked on a Claude Max subscription costing 100 USD per month for 3 months.

Codex. We used about 661,940,351 tokens, with 170,386,997 used for the natural language proof summary and 6,608,918 used in the summary comparison files. 484,944,436 tokens were used in the interactive sessions during human evaluation of the submissions.

We confirm not having used generative AI tools for any other tasks related to this manuscript, and take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## 11 Ethics Statement

We have taken care to avoid any biases or discriminatory outcomes in our research process. We have made sure to maintain the anonymity of the submitters to the Ramanujan Challenge. We are committed to maintaining transparency and integrity throughout the research process.

## 12 Reproducibility Statement

Anonymized submission results are included in the supplemental materials. For each submission, the listed data includes the tools used in that submission, human evaluation results and claude evaluation results for that submission with and without the pipeline. The order of appearance in the file, within each problem of the challenge, is randomized so that the results and tools used cannot be inferred from the public list of submissions on the challenge website.

## References

Mohammed Abouzaid, Andrew J. Blumberg, Martin Hairer, et al. First proof. arXiv preprint arXiv:2602.05192, 2026. doi: 10.48550/arXiv.2602.05192. URL https://arxiv.org/abs/ 2602.05192.

Noga Alon, Thomas F. Bloom, W. T. Gowers, Daniel Litt, Will Sawin, Arul Shankar, Jacob Tsimerman, Victor Wang, and Melanie Matchett Wood. Remarks on the disproof of the unit distance conjecture, 2026. URL https://arxiv.org/abs/2605.20695.

Levent Alpoge. A counterexample to the Jacobian conjecture in dimension three, July 2026. URL¨ https: //x.com/\_\_alpoge\_\_/status/2079028340955197566. Announcement, July 19, 2026.

Roger Apery. Irrationalit ´ e de ´ ζ(2) et ζ(3). In Journees Arithm ´ etiques de Luminy ´ , volume 61 of Asterisque ´ , pp. 11–13. Societ´ e Math ´ ematique de France, 1979.´

Henk (Hendrik) Barendregt and Freek Wiedijk. The challenge of computer mathematics. Philosophical transactions. Series A, Mathematical, physical, and engineering sciences, 363:2351–73; discussion 2374, 09 2005. doi: 10.1098/rsta.2005.1650.

Konstantin Chernyshev, Vitaliy Polshkov, Ekaterina Artemova, Alex Myasnikov, Vlad Stepanov, Alexei Miasnikov, and Sergei Tilga. U-math: A university-level benchmark for evaluating mathematical skills in llms, 2025. URL https://arxiv.org/abs/2412.03205.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

Mihai Codreanu, Alex Imas, Juan Mateos-Garcia, Joseph Emmens, Evalyne Muiruri, Arthur Turrell, Julian Jacobs, Atoosa Kasirzadeh, Ana Trisovic, Yiyuan Chen, Tanya Rodchenko, Catherine Pollard, Scott Strand, Daniel Rock, Zanna Iscenko, Fabien Curto Millet, Neil Thompson, and James Manyika. Ai in science: Early insights, 2026. URL https://arxiv.org/abs/2609.28504.

Rotem Elimelech, Ofir David, Carlos De la Cruz Mengual, Rotem Kalisch, Wolfgang Berndt, Michael Shalyt, Mark Silberstein, Yaron Hadad, and Ido Kaminer. Algorithm-assisted discovery of an intrinsic order among mathematical constants. Proceedings of the National Academy of Sciences, 121(25): e2321440121, 2024. doi: 10.1073/pnas.2321440121. URL https://www.pnas.org/doi/abs/ 10.1073/pnas.2321440121.

Tony Feng, Trieu H. Trinh, Garrett Bingham, Dawsen Hwang, Yuri Chervonyi, Junehyuk Jung, Joonkyung Lee, Carlo Pagano, Sang hyun Kim, Federico Pasqualotto, Sergei Gukov, Jonathan N. Lee, Junsu Kim, Kaiying Hou, Golnaz Ghiasi, Yi Tay, YaGuang Li, Chenkai Kuang, Yuan Liu, Hanzhao Lin, Evan Zheran Liu, Nigamaa Nayakanti, Xiaomeng Yang, Heng-Tze Cheng, Demis Hassabis, Koray Kavukcuoglu, Quoc V. Le, and Thang Luong. Towards autonomous mathematics research, 2026. URL https://arxiv.org/abs/2602.10177.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-MATH: A universal olympiad level mathematic benchmark for large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= yaqPf0KAlN.

Elliot Glazer, Ege Erdil, Tamay Besiroglu, Diego Chicharro, Evan Chen, Alex Gunning, Caroline Falkman Olsson, Jean-Stanislas Denain, Anson Ho, Emily de Oliveira Santos, Olli Jarviniemi, Matthew Barnett,¨ Robert Sandler, Matej Vrzala, Jaime Sevilla, Qiuyu Ren, Elizabeth Pratt, Lionel Levine, Grant Barkley, Natalie Stewart, Bogdan Grechuk, Tetiana Grechuk, Shreepranav Varma Enugandla, and Mark Wildon. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai, 2025. URL https: //arxiv.org/abs/2411.04872.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024.acl-long.211/.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Haocheng Ju, Guoxiong Gao, Jiedong Jiang, Bin Wu, Zeming Sun, Shurui Liu, Leheng Chen, Yutong Wang, Yuefeng Wang, Zichen Wang, Wanyi He, Peihao Wu, Liang Xiao, Ruochuan Liu, Bryan Dai, and Bin Dong. Automated conjecture resolution with formal verification, 2026. URL https:// arxiv.org/abs/2604.03789.

V. T. Khoi. On the integral of log $\begin{array} { r } { x \frac { d y } { u } - \log y \frac { d x } { x } } \end{array}$ over the A-polynomial curves. Acta Mathematica Vietnamica, 33(3):519–528, 2008. URL https://arxiv.org/abs/0811.2725.

Alex Kontorovich. The shape of math to come, 2025. URL https://arxiv.org/abs/2510. 15924.

Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuk Jung. Towards robust mathematical reasoning, 2025. URL https://arxiv.org/abs/2511. 01846.

OpenAI. Finite time blowup for Navier–Stokes. Preprint, September 2026. URL https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/ navier-stokes.pdf. Accompanied by Lean formalization; not peer reviewed.

Antoine Peyronnet, Fabian Gloeckle, and Amaury Hayat. Lemmabench: A live, research-level benchmark to evaluate llm capabilities in mathematics, 2026. URL https://arxiv.org/abs/2602. 24173.

Long Phan, Alice Gatti, Nathaniel Li, Adam Khoja, Ryan Kim, Richard Ren, Jason Hausenloy, Oliver Zhang, et al. A benchmark of expert-level academic questions to assess ai capabilities. Nature, 649 (8099):1139–1146, January 2026. ISSN 1476-4687. doi: 10.1038/s41586-025-09962-4. URL http: //dx.doi.org/10.1038/s41586-025-09962-4.

Ramanujan Machine Group. ramanujantools, 2025. URL https://zenodo.org/doi/10.5281/ zenodo.15747099.

Tomer Raz, Michael Shalyt, Elyasheev Leibtag, Rotem Kalisch, Shachar Weinbaum, Yaron Hadad, and Ido Kaminer. From euler to ai: Unifying formulas for mathematical constants. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 123959–124040. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ b394adff21327520268aaf565b63414c-Paper-Conference.pdf.

Aastha Sapkota and M. G. Sarwar Murshed. Llms as teaching assistants for mathematics exam grading: Reliability, and practical usability, 2026. URL https://arxiv.org/abs/2607.01247.

Michael Shalyt, Rotem Elimelech, and Ido Kaminer. Asymob: Algebraic symbolic mathematical operations benchmark, 2025. URL https://arxiv.org/abs/2505.23851.

Michael Shalyt, Rotem Kalisch, Carsten Schneider, Hila Barkan, Elyasheev Leibtag, John Campbell, Shachar Weinbaum, Tali Monderer, Ashvni Narayanan, and Ido Kaminer. The ramanujan challenge for ai, 2026. URL https://arxiv.org/abs/2607.09721.

Terence Tao. Mathematics in the age of ai, 2026. URL https://arxiv.org/abs/2608.16753.

The Ramanujan Machine. Ramanujan challenge. https://ramanujanmachine.com/ ramanujan-challenge/, 2026. Accessed: September 22, 2026.

Shachar Weinbaum, Elyasheev Leibtag, Rotem Kalisch, Michael Shalyt, and Ido Kaminer. On conservative matrix fields: Continuous asymptotics and arithmetic, 2025. URL https://arxiv.org/abs/ 2507.08138.

Wenjing Xie, Juxin Niu, Chun Jason Xue, and Nan Guan. Grade like a human: Rethinking automated assessment with large language models, 2024. URL https://arxiv.org/abs/2405.19694.

Zhiqi Yu, Xingping Liu, Haobin Mao, Mingshuo Liu, Long Chen, Jack Xin, and Yifeng Yu. Evaluating AI grading on real-world handwritten college mathematics: A large-scale study toward a benchmark. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview. net/forum?id=vcdsMrNGqP.

Jie Zhang, Cezara Petrui, Kristina Nikolic, and Florian Tramer. Realmath: A continuous benchmark for evaluating language models on research-level mathematics. arXiv preprint arXiv:2505.12575, 2025. doi: 10.48550/arXiv.2505.12575. URL https://arxiv.org/abs/2505.12575.

Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. minif2f: a cross-system benchmark for formal olympiad-level mathematics. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=9ZPegFuFTFv.

## A The Challenge Questions

Question 1 (Polynomial continued fraction for π). Let:

$$
a _ { n } = - 2 2 0 n ^ { 3 } - 4 8 4 n ^ { 2 } - 3 0 1 n - 4 2 , \qquad b _ { n } = 4 n ^ { 2 } ( 2 n + 1 ) ^ { 2 } ( 5 n - 4 ) ( 5 n + 6 ) .\tag{1}
$$

Prove:

$$
a _ { 0 } + { \cfrac { b _ { 1 } } { a _ { 1 } + { \cfrac { b _ { 2 } } { a _ { 2 } + { \cfrac { b _ { 3 } } { a _ { 3 } + \ddots } } } } } } = { \frac { 6 } { 3 - \pi } } .\tag{2}
$$

Question 2 (Euler’s constant γ as an Apery limit) ´ . For $n \geq 0 ,$ define:

$$
\begin{array} { r l } & { 0 = \left( - 8 n ^ { 3 } - 5 1 n ^ { 2 } - 1 0 5 n - 6 8 \right) u _ { n } } \\ & { \qquad + \left( 2 4 n ^ { 5 } + 3 3 7 n ^ { 4 } + 1 8 3 3 n ^ { 3 } + 4 8 1 8 n ^ { 2 } + 6 0 9 2 n + 2 9 2 8 \right) u _ { n - 1 } } \\ & { \qquad - \left( n + 2 \right) \left( n + 3 \right) \left( 2 4 n ^ { 5 } + 2 7 3 n ^ { 4 } + 1 1 5 0 n ^ { 3 } + 2 1 5 4 n ^ { 2 } + 1 6 3 5 n + 2 6 8 \right) u _ { n - 2 } } \\ & { \qquad + \left( n + 1 \right) \left( n + 2 \right) ^ { 4 } \left( n + 3 \right) \left( 8 n ^ { 3 } + 7 5 n ^ { 2 } + 2 3 1 n + 2 3 2 \right) u _ { n - 3 } } \end{array}
$$

Let $p _ { n } , q _ { n }$ be two solutions ofthe recurrence, defined by the initial values:

$$
p _ { - 3 } = 0 , \quad p _ { - 2 } = 7 , \quad p _ { - 1 } = 1 7 9
$$

$$
q _ { - 3 } = 1 , \quad q _ { - 2 } = 1 2 , \quad q _ { - 1 } = 3 0 6
$$

Prove

$$
\operatorname* { l i m } _ { n \to \infty } { \frac { p _ { n } } { q _ { n } } } = \gamma ,
$$

where γ is Euler’s constant.

Question 3 (The sum $\pi + e$ as an Apery limit) ´ . For $n \geq 1$ , define

$$
\begin{array} { l } { { 0 = \left( - n ^ { 3 } + 2 n ^ { 2 } + 7 n + 3 \right) u _ { n } } } \\ { { \qquad + \left( n + 2 \right) \left( 2 n ^ { 4 } + n ^ { 3 } - 2 6 n ^ { 2 } - 4 8 n - 1 9 \right) u _ { n - 1 } } } \\ { { \qquad + \left( n + 2 \right) \left( n ^ { 6 } + 9 n ^ { 5 } + 8 n ^ { 4 } - 8 7 n ^ { 3 } - 2 4 9 n ^ { 2 } - 2 3 4 n - 6 8 \right) u _ { n - 2 } } } \\ { { \qquad + \left( n + 1 \right) ^ { 2 } \left( n + 2 \right) \left( 2 n ^ { 5 } + 3 n ^ { 4 } - 1 3 n ^ { 3 } - 2 1 n ^ { 2 } + 4 \right) u _ { n - 3 } } } \\ { { \qquad - n ^ { 3 } \left( n + 1 \right) ^ { 2 } \left( n + 2 \right) \left( n ^ { 3 } + n ^ { 2 } - 8 n - 1 1 \right) u _ { n - 4 } . } } \end{array}
$$

Let $p _ { n } , q _ { n }$ be two solutions of the recurrence, defined by the initial values:

$$
p _ { - 3 } = 1 , \quad p _ { - 2 } = 1 , \quad p _ { - 1 } = 2 0 , \quad p _ { 0 } = 2 9 6
$$

$$
q _ { - 3 } = 1 , \quad q _ { - 2 } = 0 , \quad q _ { - 1 } = 4 , \quad q _ { 0 } = 4 8
$$

Prove

$$
\operatorname* { l i m } _ { n \to \infty } { \frac { p _ { n } } { q _ { n } } } = \pi + e .
$$

Question 4 (Harmonic numbers, a polylogarithm and zeta values). Prove:

$$
\sum _ { m = 0 } ^ { \infty } \sum _ { k = 0 } ^ { m } \frac { \binom { m } { k } ^ { 2 } H _ { k } ^ { 2 } } { ( m + 1 ) ^ { 2 } \binom { 2 m } { m } } =
$$

$$
2 0 \mathrm { { L i } } _ { 4 } \left( \frac { 1 } { 2 } \right) + \frac { 5 } { 6 } \log ^ { 4 } ( 2 ) + 1 0 \zeta ( 2 ) - \frac { 6 5 } { 9 } \zeta ( 2 ) ^ { 2 } - \log ^ { 2 } ( 2 ) ( 1 2 + 5 \zeta ( 2 ) ) + \frac { 1 } { 2 } \zeta ( 3 ) + \log ( 2 ) \left( \frac { 3 5 } { 2 } \zeta ( 3 ) - 1 6 \right) ,
$$

where $H _ { 0 } = 0 \ : a n d f o r \ : k \geq 1 , H _ { k }$ is the k-th harmonic number.

Question 5 (Efficient rational approximation of Catalan’s constant G). For $n \geq 0 ,$ , let

$$
M ( n ) = \left( m _ { i j } ( n ) \right) _ { 1 \leq i , j \leq 3 } ,
$$

where

$$
m _ { 1 1 } ( n ) = ( - 2 n - 5 ) ( n + 3 ) ^ { 2 } ( 1 3 6 n ^ { 4 } + 1 4 2 4 n ^ { 3 } + 5 5 4 8 n ^ { 2 } + 9 5 5 1 n + 6 1 4 1 ) ,
$$

$$
m _ { 1 2 } ( n ) = 3 8 4 n ^ { 6 } + 6 3 8 4 n ^ { 5 } + 4 4 1 6 8 n ^ { 4 } + 1 6 2 6 9 8 n ^ { 3 } + 3 3 6 3 7 7 n ^ { 2 } + 3 6 9 9 3 3 n + 1 6 9 0 1 1 ,
$$

$$
m _ { 1 3 } ( n ) = - 4 8 0 n ^ { 4 } - 4 9 8 0 n ^ { 3 } - 1 9 2 1 0 n ^ { 2 } - 3 2 6 9 0 n - 2 0 7 3 0 ,
$$

$$
m _ { 2 1 } ( n ) = ( n + 2 ) ^ { 2 } ( n + 3 ) ^ { 2 } ( 4 n + 1 0 ) ( 4 8 n ^ { 3 } + 3 8 6 n ^ { 2 } + 1 0 1 7 n + 8 7 9 ) ,
$$

$$
m _ { 2 2 } ( n ) = ( n + 2 ) ^ { 2 } ( - 2 7 2 n ^ { 5 } - 3 8 4 8 n ^ { 4 } - 2 1 7 3 2 n ^ { 3 } - 6 1 1 8 4 n ^ { 2 } - 8 5 7 6 1 n - 4 7 8 0 8 ) ,
$$

$$
m _ { 2 3 } ( n ) = ( n + 2 ) ^ { 2 } ( 3 2 0 n ^ { 3 } + 2 5 4 0 n ^ { 2 } + 6 6 1 0 n + 5 6 4 0 ) ,
$$

$$
m _ { 3 1 } ( n ) = ( - 4 n - 1 0 ) ( n + 2 ) ^ { 2 } ( n + 3 ) ^ { 2 } ( 3 2 n ^ { 4 } + 3 0 2 n ^ { 3 } + 1 0 3 7 n ^ { 2 } + 1 5 3 0 n + 8 1 3 ) ,
$$

$$
m _ { 3 2 } ( n ) = ( n + 2 ) ^ { 2 } ( 1 9 2 n ^ { 6 } + 2 9 8 4 n ^ { 5 } + 1 9 1 1 6 n ^ { 4 } + 6 4 4 5 2 n ^ { 3 } + 1 2 0 2 5 6 n ^ { 2 } + 1 1 7 2 7 9 n + 4 6 4 7 6 ) ,
$$

$$
m _ { 3 3 } ( n ) = ( n + 2 ) ^ { 2 } ( - 1 6 n ^ { 5 } - 4 0 8 n ^ { 4 } - 2 9 1 2 n ^ { 3 } - 8 8 8 4 n ^ { 2 } - 1 2 2 5 4 n - 6 2 4 0 ) .
$$

For $N \geq 0 ,$ define

$$
\mathcal { M } _ { N } = M ( 0 ) M ( 1 ) \cdot \cdot \cdot M ( N - 1 ) .
$$

Choose initial conditions

$$
A = { \left( \begin{array} { l l l } { 3 0 9 2 1 } & { - 3 2 9 7 2 } & { 8 2 4 0 } \\ { 3 3 7 5 0 } & { - 3 6 0 0 0 } & { 9 0 0 0 } \end{array} \right) } .
$$

Write

$$
A \mathcal { M } _ { N } = \binom { P _ { N , 1 } } { Q _ { N , 1 } } P _ { N , 2 } P _ { N , 3 }  { . . . }
$$

Then, for each j = 1, 2, 3, prove:

$$
\operatorname * { l i m } _ { N  \infty } \frac { P _ { N , j } } { Q _ { N , j } } = G ,
$$

where

$$
G = \sum _ { k = 0 } ^ { \infty } { \frac { ( - 1 ) ^ { k } } { ( 2 k + 1 ) ^ { 2 } } }
$$

is Catalan’s constant.

Question 6 (A series for $\zeta ( 2 ) + \zeta ( 3 ) )$ . Let $( u _ { n } ) _ { n \geq 1 }$ be the sequence defined by

$$
u _ { 1 } = - { \frac { 9 3 } { 4 4 8 0 } } , \qquad u _ { 2 } = - { \frac { 1 1 7 } { 1 4 0 0 0 } } ,
$$

and, $f o r n \ge 3 ,$ by the recurrence

$$
\begin{array} { r l } & { 0 = - 2 ( n + 3 ) ^ { 3 } ( 2 n + 5 ) ( 3 n + 5 ) u _ { n } } \\ & { \phantom { 0 = } + ( n + 2 ) ^ { 2 } ( 1 5 n ^ { 3 } + 8 5 n ^ { 2 } + 1 5 5 n + 9 3 ) u _ { n - 1 } } \\ & { \phantom { 0 = } - ( n + 1 ) ^ { 3 } ( n + 2 ) ( 3 n + 8 ) u _ { n - 2 } . } \end{array}
$$

Prove:

$$
{ \frac { \mathrm { 2 0 7 7 } } { \mathrm { 7 2 0 } } } + \sum _ { j = 1 } ^ { \infty } u _ { j } = \zeta ( 2 ) + \zeta ( 3 ) .
$$

Question 7 (Efficient four-term recurrence for $\zeta ( 2 ) + \zeta ( 3 ) )$ . For $n \geq 0 ,$ , define

$$
A _ { n } = 1 0 2 4 ( 2 n + 5 ) ^ { 4 } ( 2 n + 7 ) ^ { 3 } ( 2 n + 9 ) ^ { 3 } \big ( 9 4 6 n ^ { 2 } + 6 4 0 7 n + 1 0 8 6 0 \big ) ,
$$

$$
B _ { n } = 1 2 8 ( 2 n + 7 ) ^ { 3 } ( 2 n + 9 ) ^ { 3 } \left( 1 0 4 0 6 0 n ^ { 6 } + 1 7 4 5 3 7 0 n ^ { 5 } + 1 2 1 4 5 2 3 8 n ^ { 4 } + 4 4 8 8 6 4 8 1 n ^ { 3 } + 9 2 9 4 3 9 9 5 n ^ { 2 } + 4 4 4 0 6 4 8 1 n ^ { 4 } \right)
$$

$$
1 0 2 2 5 6 0 1 9 n + 4 6 7 0 9 0 5 2 )
$$

$$
\begin{array} { r l } & { C _ { n } = 1 6 ( n + 3 ) ^ { 4 } ( 2 n + 9 ) ^ { 3 } \big ( 3 7 8 4 n ^ { 5 } + 5 7 7 9 2 n ^ { 4 } + 3 5 1 0 1 9 n ^ { 3 } + 1 0 5 9 2 3 0 n ^ { 2 } + 1 5 8 7 2 1 1 n + 9 4 4 6 2 0 \big ) , } \\ & { D _ { n } = ( n + 3 ) ^ { 4 } ( n + 4 ) ^ { 6 } \big ( 9 4 6 n ^ { 2 } + 4 5 1 5 n + 5 3 9 9 \big ) . } \end{array}
$$

Let $( p _ { n } ) _ { n \geq 0 }$ and $( q _ { n } ) _ { n \geq 0 }$ be the two solutions of the recurrence

$$
u _ { n + 1 } = \frac { B _ { n } } { A _ { n } } u _ { n } - \frac { C _ { n - 1 } } { A _ { n - 1 } } u _ { n - 1 } + \frac { D _ { n - 2 } } { A _ { n - 2 } } u _ { n - 2 } , \qquad n \ge 2 ,
$$

with initial conditions

$$
p _ { 0 } = - 6 1 2 2 1 8 3 8 4 7 5 0 , \qquad p _ { 1 } = - \frac { 9 5 2 5 0 2 1 9 7 3 9 3 1 9 1 9 } { 1 8 1 0 0 } , \qquad p _ { 2 } = - \frac { 2 9 5 6 1 8 2 8 3 8 2 7 7 2 0 2 9 } { 6 5 3 8 0 } ,
$$

and

$$
q _ { 0 } = - 2 1 5 0 4 0 4 2 0 0 0 0 , \qquad q _ { 1 } = - \frac { 1 6 7 2 8 2 2 6 5 0 4 3 4 0 4 } { 9 0 5 } , \qquad q _ { 2 } = - \frac { 9 6 4 1 8 5 3 2 7 6 5 8 0 8 0 } { 6 0 7 1 } .
$$

Prove:

$$
\operatorname* { l i m } _ { n  \infty } \frac { p _ { n } } { q _ { n } } = \zeta ( 2 ) + \zeta ( 3 ) .
$$

Question 8 (Very fast rational approximation of ${ \sqrt { 1 0 0 0 5 } } / \pi )$ . Let

$$
R = 1 5 1 9 3 1 3 7 3 0 5 6 0 0 1 , \qquad u = 2 n + 3 , \qquad w = u ( 3 u - 2 ) ( 3 u + 2 ) .
$$

For $n \geq 0 ,$ , let

$$
M ( n ) = \left( \begin{array} { c c c c } { { \frac { a _ { 1 } } { w } } } & { { \frac { a _ { 2 } } { w } } } & { { \frac { a _ { 3 } } { w } } } & { { \frac { 1 4 4 R ( u - 1 ) ^ { 2 } } { w } } } \\ { { - u ^ { 3 } } } & { { - 3 u ^ { 2 } } } & { { - 3 u } } & { { - 1 } } \\ { { \frac { b _ { 1 } } { 1 4 4 R } } } & { { - \frac { b _ { 2 } } { 7 2 R } } } & { { - \frac { b _ { 3 } } { 3 6 R } } } & { { - \frac { 2 u + 2 R - 7 } { 2 R } } } \\ { { \frac { c _ { 1 } } { 2 8 8 R ^ { 2 } } } } & { { \frac { c _ { 2 } } { 1 4 4 R ^ { 2 } } } } & { { \frac { c _ { 3 } } { 7 2 R ^ { 2 } } } } & { { \frac { c _ { 4 } } { 4 R ^ { 2 } } } } \end{array} \right) ,
$$

where

$$
\begin{array} { r l } & { a _ { 1 } = ( 1 4 4 R - 9 9 ) u ^ { 5 } - ( 2 8 8 R - 3 3 3 ) u ^ { 4 } + ( 1 4 4 R - 2 2 9 ) u ^ { 3 } - 1 1 4 u ^ { 2 } + 4 0 u + 6 4 , } \\ & { a _ { 2 } = ( 4 3 2 R - 2 4 3 ) u ^ { 4 } - ( 8 6 4 R - 9 0 9 ) u ^ { 3 } + ( 4 3 2 R - 8 6 8 ) u ^ { 2 } - 8 0 u + 2 7 2 , } \\ & { a _ { 3 } = ( 4 3 2 R - 1 5 3 ) u ^ { 3 } - ( 8 6 4 R - 6 4 8 ) u ^ { 2 } + ( 4 3 2 R - 8 6 0 ) u + 3 6 0 , } \\ & { \qquad b _ { 1 } = 9 u ^ { 4 } - ( 1 4 4 R - 6 3 ) u ^ { 3 } + 1 5 8 u ^ { 2 } + 1 6 8 u + 6 4 , } \\ & { \qquad b _ { 2 } = 3 6 u ^ { 3 } + ( 2 1 6 R - 1 8 9 ) u ^ { 2 } - 3 1 6 u - 1 6 8 , } \\ & { \qquad b _ { 3 } = 5 4 u ^ { 2 } + ( 1 0 8 R - 1 8 9 ) u - 1 5 8 , } \end{array}
$$

and

$$
\begin{array} { c } { { c _ { 1 } = 1 8 u ^ { 5 } + ( 5 4 R + 4 5 ) u ^ { 4 } - ( 2 8 8 R ^ { 2 } - 3 7 8 R + 2 5 1 ) u ^ { 3 } } } \\ { { + ( 9 4 8 R - 1 0 8 6 ) u ^ { 2 } + ( 1 0 0 8 R - 1 3 8 4 ) u + ( 3 8 4 R - 5 7 6 ) , } } \\ { { c _ { 2 } = ( 1 5 3 R - 7 2 ) u ^ { 4 } - ( 6 5 7 R - 7 0 2 ) u ^ { 3 } - ( 4 3 2 R ^ { 2 } - 1 2 9 2 R + 1 0 6 9 ) u ^ { 2 } } } \\ { { + ( 2 0 6 4 R - 2 5 0 8 ) u + ( 1 0 7 2 R - 1 5 1 2 ) , } } \\ { { c _ { 3 } = ( 1 8 0 R - 1 0 8 ) u ^ { 3 } - ( 8 9 1 R - 8 6 4 ) u ^ { 2 } - ( 2 1 6 R ^ { 2 } - 1 4 5 0 R + 1 3 8 5 ) u } } \\ { { + ( 1 1 1 6 R - 1 4 2 2 ) , } } \end{array}
$$

$$
c _ { 4 } = ( 6 R - 4 ) u ^ { 2 } - ( 3 3 R - 3 2 ) u - ( 4 R ^ { 2 } - 5 8 R - 2 3 6 3 3 7 6 9 1 4 2 0 3 8 3 ) .
$$

For $N \geq 0 ,$ define

$$
\mathcal { M } _ { N } = M ( 0 ) M ( 1 ) \cdot \cdot \cdot M ( N - 1 ) ,
$$

with the convention that $\mathcal { M } _ { 0 } = I .$

Choose initial conditions

$$
A = \left( { \begin{array} { c c c c } { A _ { 1 } } & { A _ { 2 } } & { A _ { 3 } } & { A _ { 4 } } \\ { B _ { 1 } } & { B _ { 2 } } & { B _ { 3 } } & { B _ { 4 } } \end{array} } \right) ,
$$

where

$$
\begin{array} { r l } & { ( A _ { 1 } , A _ { 2 } , A _ { 3 } , A _ { 4 } ) = ( 3 7 1 6 9 3 0 5 7 6 0 4 4 2 2 5 2 7 6 1 4 4 1 , 1 1 1 5 0 7 9 1 7 2 8 1 3 2 7 4 4 1 5 6 4 2 0 8 , } \\ & { \qquad \quad 1 1 1 5 0 7 9 1 7 2 8 1 3 2 7 5 9 9 7 2 0 1 2 9 , 3 7 1 6 9 3 0 5 7 6 0 4 4 2 4 1 0 9 1 7 3 6 2 ) , } \end{array}
$$

and

$$
\begin{array} { r } { ( B _ { 1 } , B _ { 2 } , B _ { 3 } , B _ { 4 } ) = ( 1 1 6 7 4 1 6 3 6 1 5 4 2 6 3 9 6 9 2 3 2 0 , 3 5 0 2 2 4 9 0 8 4 6 2 7 8 9 6 1 3 2 1 6 0 , } \\ { 3 5 0 2 2 4 9 0 8 4 6 2 7 8 7 9 6 9 7 2 8 0 , 1 1 6 7 4 1 6 3 6 1 5 4 2 6 2 2 7 2 3 8 4 0 ) . } \end{array}
$$

Write

$$
A \mathcal { M } _ { N } = \left( \begin{array} { l l l l } { P _ { N , 1 } } & { P _ { N , 2 } } & { P _ { N , 3 } } & { P _ { N , 4 } } \\ { Q _ { N , 1 } } & { Q _ { N , 2 } } & { Q _ { N , 3 } } & { Q _ { N , 4 } } \end{array} \right) .
$$

Then, for each $j = 1 , 2 , 3 , 4 ;$

$$
\operatorname* { l i m } _ { N  \infty } \frac { P _ { N , j } } { Q _ { N , j } } = \frac { \sqrt { 1 0 0 0 5 } } { \pi } .
$$

## B The Challenge Conjectures

This section contains open problems, presented here for the first time. In each case, the statement is written in the form of an explicit formula so that it can be checked numerically.

Conjecture 1 (An integral over knot polynomial roots expressing $\pi ^ { 2 } )$ . Let $A _ { 7 _ { 2 } } ( M , L )$ denote the $A -$ polynomialfor the prime knot $7 _ { 2 }$

$$
\begin{array} { l } { { A _ { 7 _ { 2 } } ( M , L ) = L ^ { 5 } + L ^ { 4 } \left( M ^ { 1 4 } - M ^ { 1 2 } + 3 M ^ { 4 } + 4 M ^ { 2 } - 2 \right) } } \\ { { \qquad + L ^ { 3 } \left( - 2 M ^ { 1 8 } + 5 M ^ { 1 6 } + M ^ { 1 4 } - 4 M ^ { 1 2 } + 6 M ^ { 8 } + 5 M ^ { 6 } + 2 M ^ { 4 } - 4 M ^ { 2 } + 1 \right) } } \\ { { \qquad + L ^ { 2 } \left( M ^ { 2 2 } - 4 M ^ { 2 0 } + 2 M ^ { 1 8 } + 5 M ^ { 1 6 } + 6 M ^ { 1 4 } - 4 M ^ { 1 0 } + M ^ { 8 } + 5 M ^ { 6 } - 2 M ^ { 4 } \right) } } \\ { { \qquad + L \left( - 2 M ^ { 2 2 } + 4 M ^ { 2 0 } + 3 M ^ { 1 8 } - M ^ { 1 0 } + M ^ { 8 } \right) + M ^ { 2 2 } . } } \end{array}
$$

Let α ≈ 0.349269 . . . denote the real solution of $A _ { 7 _ { 2 } } \left( \alpha , \alpha ^ { 1 / 2 } \right) ~ = ~ 0$ closest to 0.349269. Let $\beta$ ≈ 0.406813 . . . denote the real solution of $A _ { 7 _ { 2 } } ( \beta , \beta ) = 0$ closest to 0.406813 Let $y = y ( x )$ denote the curve satisfying $A ( x , y ( x ) ) = 0 f o r \alpha \leq x \leq \beta$ and such that $y ( x )$ is positive and decreasing with $y ^ { \prime } ( x ) \leq - 2 .$ Prove thefollowingformula, which was discovered empirically:

$$
{ \frac { 4 \pi ^ { 2 } } { 8 5 } } = \int _ { \alpha } ^ { \beta } \left( \log x { \frac { d y } { y } } - \log y { \frac { d x } { x } } \right) .
$$

This result is a surprisingly simple conjectural closed form of a Godbillon–Vey type Knot invariant (Khoi, 2008).

Conjecture 2 (Optimality of Apery’s irrationality-measure bound for ´ $\zeta ( 3 ) )$ . The sequences $( a _ { n } ) _ { n \geq 0 }$ and $\left( b _ { n } \right) _ { n \geq 0 }$ are solutions ofApery’s recurrence´

$$
( n + 1 ) ^ { 3 } u _ { n + 1 } - { \left( 3 4 n ^ { 3 } + 5 1 n ^ { 2 } + 2 7 n + 5 \right) } u _ { n } + n ^ { 3 } u _ { n - 1 } = 0 \qquad ( n \geq 1 ) ,
$$

with initial values

$$
a _ { 0 } = 0 , \quad a _ { 1 } = 6 , \quad \quad b _ { 0 } = 1 , \quad b _ { 1 } = 5 .
$$

Define $d _ { n } : = \operatorname { l c m } ( 1 , \dots , n ) ^ { 3 }$ . Note that $d _ { n } a _ { n } , d _ { n } b _ { n } \in \mathbb { Z }$ as shown in (Apery, 1979).´

Prove:

$$
\operatorname* { g c d } ( d _ { n } a _ { n } , d _ { n } b _ { n } ) = e ^ { o ( n ) } .
$$

The $a _ { n } , b _ { n }$ sequences were generated by Apery in his proof of the irrationality of ´ ζ(3) (Apery, 1979).´ Specifically, considering the linear forms $a _ { n } - \zeta ( 3 ) b _ { n }$ gives a bound on the irrationality measure of ζ(3), by demonstrating that $d _ { n } a _ { n } , d _ { n } b _ { n } \in \mathbb { Z }$

The above conjecture shows that the irrationality-measure bound found by Apery is tight for his par- ´ ticular formula.