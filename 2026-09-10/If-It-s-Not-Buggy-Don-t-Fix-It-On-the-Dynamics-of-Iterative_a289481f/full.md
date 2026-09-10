# If It’s Not Buggy, Don’t Fix It: On the Dynamics of Iterative Bug-fixing with LLMs

Xietao Wang-Lin<sup>∗</sup> University of Warwick

Anton Isopoussu UnlikelyAI

Louis Mahon UnlikelyAI

## Abstract

Large language models (LLMs) have become ubiquitous in software development, with LLM-based automated program repair tools increasingly used during code review. In this report, we explore the iterative blind use of LLMs as bug-fixers. Across multiple models and repair environments, we find that LLMs consistently claim to detect bugs in entirely bug-free programs while the rate of repair of buggy programs is less than that of the damage to correct programs. We also explore the long-term dynamics of this iterative process, and find that this frequently reaches a pseudo-bug-fixing cycle where the same changes are added and removed again ad infinitum. Lastly, via mechanistic probing, we unveil the existence of a steering vector which controls the editing propensity, suggesting that LLMs have an internal representation of “buggy code", and that this representation is what is falsely activated to induce pseudo-bug fixing. These results provide insight towards the dynamics of fully autonomous bug-fixing systems, as well as stopping conditions under ambiguous goals.

## 1 Introduction

Large language models (LLMs) are now ubiquitous in coding tasks. Increasingly, they are used as fully autonomous agentic harnesses, where the user gives little to no supervision and the agent iteratively processes the code, with the stopping condition being determined by the harness [Yang et al., 2024b]. A similar loop arises in multi-agent interactions, where the output generated turns into the input for the next generation [Du et al., 2024, Ko and Geiping, 2026]. However, little is known about the long-term effects of this iterative process, such as code correctness or how and when the agent decides to terminate the iteration, especially when there is no objective measure of task performance.

Bug-finding is an essential part of the coding process. Neither human or LLM code are rarely perfect on the first attempt, and detecting and fixing potential errors is crucial. LLMs often do this via automated bug-fixing agents that operate iteratively, i.e., the LLM searches for bugs, applies patches, then again searches the patched code and potentially applies more fixes, iterating until no more bugs are found [Liu et al., 2024, Bouzenia et al., 2025]. Given the ubiquity and importance of this setup, some natural questions arise: how does code evolve under iterative bug-fixing? Are LLMs aware of bugs under vague conditions? When is the LLM satisfied the code is completely bug-free?

In this report, we particularly focus on iterative bug-fixing using LLMs, where LLMs review and fix bugs on the same piece of code until it finds no more bugs. We consider the case where the prompt does not include history, that is, the agent sees the current state of the code, but not the sequence of states the code has been in through past bug fixes [Xia et al., 2025]. This situation may arise, for instance, when a user approves changes blindly, when including past states would exceed the context window or in multi-agent setups with delegation [Perry et al., 2023, Laban et al., 2026a].

Our contributions are as follows:

• We study iterative bug-fixing dynamics and quantify the rate at which it improves buggy code, as well as the rate at which it damages correct code by trying to remove pseudo-bugs;

• We show that, under relatively standard settings, the damage rate can end up being significantly higher than the repair rate;

• We document the behaviour of pseudo-bug-fixing cycles, which appear more frequently when using atomic changes compared to whole-file edits.

• By mechanistic probing, we unveil the existence of a “buggy code" internal representation in LLMs, which is activated when the LLM detects bugs (either pseudo-bugs or real bugs) and not when the LLM thinks the code is bug-free.

## 2 Related work

Dynamical systems in LLMs. Recent work often takes a dynamical system view to model a variety of behaviours of LLMs. Shumailov et al. [2024] consider recursive training as a Markov chain over the empirical data distribution. Looped transformers can be seen as a dynamical system over the looped latent states, and have been shown to have cycles and fixed points [Geiping et al., 2025, Movahedi et al., 2026, Blayney et al., 2026]. The framework of dynamical systems can be used to study iterative generation, where the output is recirculated into the input of the next LLM, creating a dynamical system over the text generations. This has previously been studied in the literature in the form of iterative paraphrasing [Geng et al., 2026, Wang et al., 2025a], backtranslation [Geng et al., 2026, Mohamed et al., 2025, Hong et al., 2025] and program mutation [Gurkan et al., 2026], where LLMs converge to attractors or periodic states. Perez et al. [2025] look into text semantics and find that iterative generation over texts has specific cultural attractors. Wu et al. [2026] study iterative refinement of paper abstracts and find that this process converges exponentially fast.

Iterative coding with LLMs. Similar iterative generation loops have been studied in the context of coding tasks. Choi et al. [2026] used a bug-generator together with a bug-fixer LLM to improve bugfixing via RL; Peitek et al. [2026] repeatedly prompt an LLM to improve code readability, converging to a fixed-point in a few iterations, thereby suggesting that LLMs might have an “optimal readability” internal representation of code. Self-refine has also been applied to code snippets [Madaan et al., 2023, Olausson et al., 2024], where the self-refinement loop is used to improve the pass rate of code solutions. Gao et al. [2026] deviate from the self-refine loop by giving the LLM evidence from the code rather than feedback, finding that regression bugs are common in generation-test-revise loops.

Long-horizon tasks. Laban et al. [2026b], using a chain of reversible tasks as toy model, found that most LLMs undertaking long-horizon tasks often make a critical mistake which they never recover from. Similarly, Orlanski et al. [2026], constructed a benchmark where LLMs iteratively extend their previous code, and find that correctness and conciseness degrade over time.

Steering activations. The linear representation hypothesis [Park et al., 2024] poses that high-level concepts are represented linearly, and thus, linear probes have become a popular method to capture the underlying concepts [Marks and Tegmark, 2024]. The vectors obtained from the probe, known as steering vectors, can be used to steer the model to amplify or lessen the captured concept during generation [Wu et al., 2025]. Indeed, recent work suggests that LLMs linearly encode concepts such as confidence [Venhoff et al., 2025] and the overall likelihood of success of their current trajectory [Xu et al., 2026, Jiang et al., 2026]. In this report, we use linear probes to obtain a steering vector of “code bugginess", which controls the iterative bug-fixing loop.

## 3 Iterative bug-fixing

Prompts. We evaluate two techniques of code repair: (1) whole-file editing, where the LLM output is the entire repaired code, and (2) search/replace blocks (SRBs), as our style for atomic changes. SRBs, which have been used in Aider [Gauthier, 2023], SWE-RL [Wei et al., 2025] and Agentless [Xia et al., 2025], are preferred over formats involving line numbers (e.g. unified-diffs) in this setup as we only have a single code file to edit and it has been shown in previous literature that LLMs often make mistakes on line numbers, often requiring an additional post-processing step to align intended lines [Sun et al., 2026]. In this case, we prompt the LLM to only fix the single most important bug (i.e. one bug at a time); as small code files often have overlapping logic, and we found that the LLM often suggests overlapping SRBs. Exact prompts can be found in Appendix A.

![](images/890a5036b7af902423a49168a1f065399af650fce7c3015d60b9a364d996d9d7.jpg)  
Figure 1: Diagram of the iterative bug-fixing loop. (Top) Using whole-file edits, the LLM takes in the input code file and outputs the corrected code, which is then used as the next input. (Bottom) Loop using atomic changes, the LLM reads the code file and outputs a reasoning trace, followed by one search/replace block, which is applied programmatically. The resulting code after the atomic change application is then used as input for the turn.

Bug-fixing as a dynamical system. Iterative generation can be viewed as a discrete dynamical system. Consider the space of all possible code files C. If using greedy decoding, an instance of bug-fixing is represented by a fixer function, $f \colon { \mathcal { C } } \to { \mathcal { C } } ,$ , such that f is an instruction-tuned LLM and a prompt. A diagram of the setup is provided in Figure 1.

Then, given an initial seed code, $C _ { 0 } \in { \mathcal { C } } _ { \mathrm { ~  ~ } }$ , we can iteratively fix the code using the recurrence,

$$
C _ { n + 1 } = f ( C _ { n } ) , \quad n \in \{ 0 , 1 , 2 , . . . , N \} .\tag{1}
$$

When using SRBs, the code update comes from programmatically applying the SRB produced by the model. If the block is invalid (wrong format, search block not in code), we apply an identity change. Hence, the fixer function is now, ${ \tilde { f } } \colon { \mathcal { C } } \to B$ , where B is the space of all possible search/replace blocks, and we define $g \colon { \mathcal { C } } \times B \to { \mathcal { C } }$ , as the function that applies the SRB. For $B \in B ,$ , let

$$
g ( C , B ) = { \left\{ \begin{array} { l l } { \operatorname { a p p l y } ( C , B ) , } & { { \mathrm { i f ~ } } B { \mathrm { ~ i s ~ v a l i d ~ f o r ~ } } C , } \\ { C , } & { { \mathrm { i f ~ } } B { \mathrm { ~ i s ~ i n v a l i d ~ f o r ~ } } C . } \end{array} \right. }
$$

Next, we define

$$
\hat { g } ( { \cal C } ) = g ( { \cal C } , B ) , \quad B = f ( { \cal C } ) ,
$$

and

$$
C _ { n + 1 } = \hat { g } ( C _ { n } ) = g ( C _ { n } , B _ { n } ) , \quad B _ { n } = f ( C _ { n } ) , \quad n \in \{ 0 , 1 , 2 , . . . , N \} .
$$

Using stochastic decoding, such as top-k and top-p, f is now a Markov transition kernel, that is,

$$
( \mathrm { w h o l e - f i l e } ) \ C _ { n + 1 } \sim f ( \cdot \mid C _ { n } ) , \quad \mathrm { o r } \quad ( \mathrm { S R B } ) \ B _ { n } \sim f ( \cdot \mid C _ { n } ) , \quad n \in \{ 0 , 1 , 2 , . . . , N \} .\tag{2}
$$

When using stochastic generation, for all our models, we set temperature $\tau = 0 . 7 ,$ top-p to 0.9 and limit our generations to $N = 1 0 0$ turns.

Note, given that we do not append history or previous code iterations in the prompt, all our dynamical systems are strictly Markovian.

## 4 Experimental Evaluation

Dataset. We use CodeContests+ [Wang et al., 2025b], a dataset that contains user submissions of competitive programming problems with an extended set of test cases containing both correct and incorrect code submissions. We say that a piece of code is correct if it passes all the hidden test cases in the allocated time limit, and incorrect otherwise; we do not use partial credit.

Table 1: Lines of code (LOC) statistics of the user submissions considered in this report.
<table><tr><td>Submission type</td><td>Median LOC</td><td>Min. LOC</td><td>Max. LOC</td></tr><tr><td>Incorrect</td><td>38</td><td>5</td><td>189</td></tr><tr><td>Correct</td><td>42</td><td>12</td><td>119</td></tr></table>

Throughout the report we always use the same 20 problems, which are chosen at random; and from each problem, the same 40 C++ user submissions, also chosen at random. The average number of test cases is 23, and the average time-limit is 2.1 seconds.

Models. We evaluate Gemini 2.5 Flash-Lite [Gemini Team, 2025], and Qwen2.5-7B-Instruct [Yang et al., 2024a].

## 4.1 Repair Rate and Damage Rate

First, we compute the empirical repair rate, α and damage rate, $\beta .$ . That is,

$$
\alpha = \mathbb { P } ( C _ { t + 1 } { \mathrm { ~ i s ~ c o r r e c t ~ } } | \ C _ { t } { \mathrm { ~ i s ~ i n c o r r e c t } } ) ,\tag{3}
$$

$$
\beta = \mathbb { P } ( C _ { t + 1 } { \mathrm { ~ i s ~ i n c o r r e c t ~ } } | \ C _ { t } { \mathrm { ~ i s ~ c o r r e c t } } ) .\tag{4}
$$

The computed empirical rates are shown in Table 2.

Table 2: Repair rate and damage rate across different configurations and their respective standard error of the mean. Column name, ‘correct’ vs ‘incorrect’, refers to the starting state of the trajectory, and α and $\beta$ are then computed over all transitions in that trajectory.
<table><tr><td colspan="3"></td><td colspan="2">Gemini 2.5 Flash-Lite</td><td colspan="2">Qwen2.5-7B-Instruct</td></tr><tr><td colspan="2"></td><td></td><td>SRB</td><td>Whole-file</td><td>SRB</td><td>Whole-file</td></tr><tr><td rowspan="4">Correct</td><td rowspan="2"> $\tau = 0$ </td><td>α</td><td> $0 . 0 6 2 \pm 0 . 0 0 5$ </td><td> $0 . 0 4 0 \pm 0 . 0 0 9$ </td><td> $0 . 0 4 1 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 6 \pm 0 . 0 0 9$ </td></tr><tr><td>β</td><td> $0 . 2 9 3 \pm 0 . 0 1 1$ </td><td> $0 . 1 6 5 \pm 0 . 0 0 9$ </td><td> $0 . 4 2 4 \pm 0 . 0 1 5$ </td><td> $0 . 0 9 9 \pm 0 . 0 1 0$ </td></tr><tr><td rowspan="2"> $\tau = 0 . 7$ </td><td>α</td><td> $0 . 0 2 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td></tr><tr><td>β</td><td> $0 . 2 1 6 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 1$ </td><td> $0 . 1 9 0 \pm 0 . 0 0 5$ </td><td> $0 . 0 1 6 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan="4">Incorrect</td><td rowspan="2"> $\tau = 0$ </td><td>α</td><td> $0 . 0 2 3 \pm 0 . 0 0 2$ </td><td> $0 . 1 0 1 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 4$ </td></tr><tr><td>β</td><td> $0 . 2 6 1 \pm 0 . 0 3 2$ </td><td> $0 . 0 7 3 \pm 0 . 0 1 5$ </td><td> $0 . 4 4 0 \pm 0 . 0 9 9$ </td><td> $0 . 0 6 3 \pm 0 . 0 4 3$ </td></tr><tr><td rowspan="2"> $\tau = 0 . 7$ </td><td>α</td><td> $0 . 0 0 7 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 0$ </td></tr><tr><td>β</td><td> $0 . 1 8 7 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 3 9 1 \pm 0 . 0 5 1$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td></tr></table>

The same information is shown graphically, as trajectories, in Figures 2 and 3. Overall, we see that the damage done by pseudo-bug fixing in correct files (damage rate) is at least as large as the improvements. For SRBs, the damage rate is substantially higher than the repair rate; for whole-file edits, the two are similar. The behaviour remains qualitatively similar in both $\tau = 0$ and $\tau = 0 . 7$

## 4.2 Attractor types

Attractor classification. Under greedy decoding, if the fixer outputs a previously seen state in the chain, we stop the process as this leads to a loop. Loops of length-1 are fixed points, while longer loops are called cycles. States outside of attractors are denoted as transient, and the time to convergence to attractor as transient length.

![](images/33f29b7fa35e3166f86fc81bc2144e000546260d20178857cafbff939702f539.jpg)

![](images/8ab86dfefbd6a2052fa4b9c09ec9882f0dad83fd289db84567ec09e6fef0421c.jpg)  
Figure 2: Average pass rate across turns using Gemini 2.5 Flash-Lite, when $\tau = 0$ . Correct initial submissions in green, incorrect initial submissions in red, and the overall aggregate in black. (a) Using SRBs, (b) whole-file edits.

![](images/86fd2596c736a8f5cea0e4309d478d7833724b7b7faf112c662bf326f072468b.jpg)

(b) Whole-file  
![](images/c0bd58714bc14990c4bf8fe2883a95942a048ddb5eb3696a1ffc5329277e4ca9.jpg)  
Figure 3: Average pass rate across turns when $\tau = 0 . 7 .$ . Gemini 2.5 Flash-Lite in blue and Qwen2.5- 7B-Instruct. (a) Using SRBs, (b) whole-file edits.

When using SRBs, there are several ways a fixed point can be produced: 1) it outputs “- None", meaning the fixer thinks there are no bugs, 2) an identity SRB, meaning the LLM says there is a bug to fix, but the fixed section is identical to the replaced section, 3) the search block cannot be located in the code, and 4) none of the above, usually this coincides with repetition degeneration [Holtzman et al., 2020] as comments, even though our prompt contains specific instruction not to add comments (see Appendix A). On the other hand, when using whole-file edits, we only have two possible outcomes, either the output program has been seen previously during the bug-fixing loop or the output contains degenerate comments. Examples of possible attractor types for both SRBs and whole-file edits are shown in Appendix C.

Frequencies for each attractor type are shown in Figure 4, transient length in Figure 5 and cycle length distributions are shown in Figure 6. The following observations arise: (1) SRBs induce significantly more cycles than whole-edits, (2) degenerations are more common when using SRBs as there are more opportunities for the model to degenerate, (3) SRBs produce significantly longer runs until convergence, likely due to the model only being allowed to fix one bug at a time, and (4) cycles are much longer when using SRB, also likely due to fixing one bug at a time. One possible explanation is that whole-file edits allow the model to maintain consistency of the overall logic, instead of having to make compensatory changes in a subsequent SRB. We also hypothesise that, of the many correct ways to write a given piece of code, the LLM often has an encoding of a single one. When it can rewrite the entire code, it simply selects this one correct way. If using SRBs instead, it tries to work off an alternative version of the code and can get confused even if that code is correct, because it does not meet the LLM’s more narrow notion of correctness.

![](images/5cb97761d0bab8fb3c7cbafbf95893959b4ef5e2a13109409ca5c90eb43b23e9.jpg)  
Figure 4: Attractor type frequency, using Gemini 2.5 Flash-Lite τ = 0. (a) Using SRBs, (b) using whole-file edits. (Top) Starting with correct code. (Bottom) starting with incorrect code. We group whole-file edits identity output with None. Failed refers to submissions that are incorrect or do not compile.

Lastly, as seen in Figure 6, cycles are not necessarily of one type (i.e. all the states are incorrect or all the states are correct). There are cycles with mixed correct/incorrect states in both SRBs and whole-file edits: for instance, when the cycle length is 2, this is a cycle where the loop is cycling between a correct and incorrect state infinitely.

## 5 Bug steering vector

To complement our analysis, we construct a steering vector that detects the presence of bugs in Qwen2.5-7B-Instruct. The construction is based on the method of difference-of-means, where we contrast two opposite classes. Here, we use source files where the model believes a bug is present with confidence as positive class, and source files where the model believes is bug-free with confidence as negative class.

![](images/ee764591707ceb5ab58becd2b85df3ca76436ea12201d7f2f96d6c4eefc7120e.jpg)  
Figure 5: Transient length distributions when $\tau = 0 .$ . (a) Gemini 2.5 Flash-Lite, and (b) Qwen2.5-7B-Instruct.

![](images/5971cc1d60d66c1688320f96c200746caa34d01142e39b8985c2c8dd95c9d585.jpg)  
Figure 6: Cycle types for Gemini 2.5 Flash-Lite $\tau = 0$ . Pass (in green) are cycles with all correct states, mixed (in orange) are cycles with mixed correct/incorrect states, and fail (in red) are states with all incorrect states.

To measure the confidence, we use a separate LLM call: the model outputs the token $\mathbf { \Delta } ^ { \bullet } \mathbf { Y } '$ if the code has at least one bug and the token $\mathbf { \nabla } ^ { \left. \mathbf { N } \right. }$ otherwise. Therefore, a source file, C, is in the positive set, ${ \mathcal { P } } _ { : }$ if $\mathbb { P } ( Y | C ) > \mathrm { t o p } \bar { - p } = 0 . 9$ , and is in the negative class, N, if $\mathbb { P } ( N | C ) > \mathrm { t o p } \lnot p = 0 . 9$

Then, we compute the difference-of-means,

$$
\mathbf { v } _ { l } = \frac { 1 } { | \mathcal { P } | } \sum _ { C \in \mathcal { P } } \mathbf { h } _ { l } ( C ) - \frac { 1 } { | \mathcal { N } | } \sum _ { C \in \mathcal { N } } \mathbf { h } _ { l } ( C ) ,\tag{5}
$$

where ${ \bf h } _ { l } ( C )$ is the final-token activation of the prompt using $C \in { \mathcal { C } }$

The resulting $\mathbf { v } _ { l }$ can then be used to steer the activations, i.e.,

$$
\mathbf { h } _ { l } ^ { \mathrm { s t e e r e d } } = \mathbf { h } _ { l } + \gamma \mathbf { v } _ { l } .\tag{6}
$$

Additionally, we can construct the following scoring metric, s, by projecting the activation onto the steering direction,

$$
s _ { l } ( C ) = \mathbf { h } _ { l } ( C ) \cdot { \frac { \mathbf { v } } { | \mathbf { v } | } } .\tag{7}
$$

## 5.1 Results

Using Equation 7, we first evaluate the constructed steering vector validation $\mathrm { { A U C } , }$ using 5-fold cross validation, which is shown in Figure 7. We see that the constructed steering vectors obtain

![](images/6c63bee4d33175cd4d840619a18b489aa5322d84e5f041d5244b2eceb3c46c48.jpg)

![](images/216fcbd9e959357854a3a2a48eeadf1051d8958eb3e1a8a3e261c90cefff2404.jpg)

![](images/2a10c915af9e5d2baf535684fefc5d19d868103714349f44a994fda120780206.jpg)  
Figure 7: (Left) Cross validation AUC of the constructed steering vector at each layer l. The red shadow indicates the confidence interval, and the dotted line denotes a random classifier. (Middle) Cosine similarity of the steering vector, $\mathbf { v } _ { l } .$ , and readout direction, r. The shadowed area indicates 2σ range of a random vector. (Right) Average pass rate across turns using Qwen2.5-7B-Instruct, when $\tau = 0$ , under different values of steering γ.

moderate AUC values (0.7, 0.8) in the early and middle layers. This rises close to 1.0 at around layer 19, possibly because the late layers are preparing the last token for the unembedding operator, so the classifier can read the solution from the activations. However, we can show, using two methods, that the fitted direction is not simply reading off the solution from the activations at layers 19-22: (1) logit lens analysis, and (2) performing new tasks under steering.

Logit lens analysis. Denote $\mathbf { W } ^ { U }$ to be the unembedding weights, and $i _ { Y } , i _ { N }$ denote the token IDs for the tokens $\bf \cdot \vec { Y } ^ { \prime } , \tau \cdot \vec { N } ^ { \prime }$ , respectively. Then, the readout direction is $\mathbf { r } = \bar { \mathbf { W } } _ { i _ { Y } : } ^ { U } \mathbf { \Sigma } - \mathbf { W } _ { i _ { N } : } ^ { U }$ , and if a layer l activation stores the verdict directly, the cosine similarity between r and $\mathbf { v } _ { l }$ should be quite high. However, in Figure 7, we see that the cosine similarity remains around random levels between layers 19-22, which suggests that they are not simply encoding the tokens $\mathbf { \Delta } ^ { \bullet } \mathbf { Y } ^ { \bullet }$ and $\mathbf { \nabla } ^ { \mathsf { \bullet } } \mathbf { N } ^ { \bullet }$

New tasks. If the vectors 19-22 carry a signal of “bugginess", we should be able to steer activations on unseen tasks. First, we direct the LLM to count the number of different issues (Appendix $\mathbf { A . } 4 )$ , we keep the definitions for each issue deliberately vague; the result for layer $v _ { l }$ at layer, $l = 2 2$ , is shown in Figure 8(a). Hence, we fix $l = 2 2$ from subsequent steering experiments.

Now, we can use the steering vector to increase or decrease the propensity of edits during bugfixing loop. Repeating the experiment with $\tau = 0 ,$ , using SRBs and $\gamma = - 0 . 5 , 0 . 5$ , we show the results in Figure 7 (right). Indeed, negative steering completely stops the loop from happening, thus preserving all correct submissions, while not fixing any incorrect submissions. Though, positive steering introduces slightly more bug-fixing rounds, this is because the increase in bugs causes the loop to end prematurely as search blocks are not being found in the code (Figure 8(a)), that is, the LLM hallucinates the code to fix.

## 6 Discussion

This report investigates the dynamics of iterative code bug-fixing under information constraints scenarios. We find that LLMs are overeager to find (pseudo-)bugs, causing correct programs to fail, while only a minority of incorrect programs get to a passing state. Furthermore, we show that SRBs, compared to whole-file editing, induce more mistakes, increase the frequency of cycling changes, and have more failure modes such as degenerations or tries to search code not present in the source file.

![](images/15166beb9fc4e7f0fb6abb65c438bd9fd1b83be213ca0304c37f32e6cbc5c1bc.jpg)  
(a) Attractor taxonomy under steering.

![](images/d6009505a4c47047b5e34b8f874df40f54953987becddee8f48690d84e6e5d66.jpg)  
(b) Counting code issues under steering.  
Figure 8: (a) Attractor type frequency using Qwen2.5-7B-Instruct with τ = 0 and atomic changes as a function of γ. (b) Mean issue count difference versus the baseline (γ = 0), for different values of γ.

Then, using linear probes, we unveil the existence of a “buggy code" latent direction, which coincides with the model’s decision that the code contains bugs to fix. When we apply this steering vector at inference time, we show that positive steering causes the model to make more edits, thus increasing both the damage and repair rates, while negative steering does the opposite. These findings suggest that LLMs have an internal representation of bugs in the code, a finding which may have applications in controlling agents’ sensitivity.

Broader impacts. Our paper studies iterative bug-fixing using LLMs and may prove directly useful for the design of autonomous bug-fixing agents. This could lead to many potential impacts, including more accurate and reliable LLM coding assistants.

Limitations and future work. This report focuses on the dynamics of Gemini 2.5 Flash-Lite and Qwen 2.5-7B-Instruct in a blind environment without goals. Future work should extend the analysis to larger models, environments with exact goals, and multi-file environments.

## References

Hugh Blayney, Álvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Michael M. Bronstein, and Xiaowen Dong. A mechanistic analysis of looped reasoning language models, 2026. URL https://arxiv.org/abs/2604.11791.

Islem Bouzenia, Premkumar Devanbu, and Michael Pradel. Repairagent: An autonomous, llmbased agent for program repair. In 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE), pages 2188–2200, 2025. doi: 10.1109/ICSE55347.2025.00157.

Caroline Choi, Zeyneb N. Kaya, Shirley Wu, Tengyu Ma, Tatsunori Hashimoto, and Ludwig Schmidt. Anchoring self-play for code repair. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=kJ6dvGnV2f.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=zj7YuTE4t8.

Xueping Gao, Jianwei Yang, and Qiang Yang. Looping is not reliability: State-bound evidence and typed revision contracts for agentic code repair, 2026. URL https://arxiv.org/abs/2607. 24604.

Paul Gauthier. Aider: Ai pair programming in your terminal. https://github.com/Aider-AI/ aider, 2023.

Jonas Geiping, Sean Michael McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute

with latent reasoning: A recurrent depth approach. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= S3GhJooWIC.

Google Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/ 2507.06261.

Mingmeng Geng, Amr Mohamed, Guokan Shang, Michalis Vazirgiannis, and Thierry Poibeau. Markovian generation chains in large language models, 2026. URL https://arxiv.org/abs/ 2603.11228.

Can Gurkan, Forrest Stonedahl, and Uri Wilensky. Mutation without variation: Convergence dynamics in LLM-driven program evolution. In Proceedings of the Genetic and Evolutionary Computation Conference Companion, GECCO ’26 Companion, page 1392–1407, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400724886. doi: 10.1145/3795101.3814735. URL https://doi.org/10.1145/3795101.3814735.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=rygGQyrFvH.

Zhaochen Hong, Haofei Yu, and Jiaxuan You. ConsistencyChecker: Tree-based evaluation of LLM generalization capabilities. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 33039–33075, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/ v1/2025.acl-long.1585. URL https://aclanthology.org/2025.acl-long.1585/.

Nick Jiang, Isaac Kauvar, and Jack Lindsey. The value axis: Language models encode whether they’re on the right track, 2026. URL https://arxiv.org/abs/2606.17056.

Ting-Wen Ko and Jonas Geiping. Attractor states emerge in multi-turn llm conversations, 2026. URL https://arxiv.org/abs/2606.30571.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. Llms get lost in multi-turn conversation. In International Conference on Learning Representations, volume 2026, 2026a. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 59f6421e64707225fdf5b28840679a07-Paper-Conference.pdf.

Philippe Laban, Tobias Schnabel, and Jennifer Neville. Llms corrupt your documents when you delegate, 2026b. URL https://arxiv.org/abs/2604.15597.

Yizhou Liu, Pengfei Gao, Xinchen Wang, Jie Liu, Yexuan Shi, Zhao Zhang, and Chao Peng. Marscode agent: Ai-native automated bug fixing, 2024. URL https://arxiv.org/abs/2409.00899.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 46534–46594. Curran Associates, Inc., 2023. doi: 10.52202/075280-2019. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=aajyHYjjsk.

Amr Mohamed, Mingmeng Geng, Michalis Vazirgiannis, and Guokan Shang. LLM as a broken telephone: Iterative generation distorts information. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7493–7509, Vienna,

Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.371. URL https://aclanthology.org/2025.acl-long.371/.

Sajad Movahedi, Vera Milovanovic, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann,´ Valentina Boeva, T. Konstantin Rusch, and Antonio Orvieto. Fixed-point reasoners: Stable and adaptive deep looped transformers, 2026. URL https://arxiv.org/abs/2606.18206.

Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, and Armando Solar-Lezama. Is self-repair a silver bullet for code generation? In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, International Conference on Learning Representations, volume 2024, pages 36545–36593, 2024. URL https://proceedings.iclr.cc/paper\_files/ paper/2024/file/9ddc141bdbf9d1db510cefff56c586ad-Paper-Conference.pdf.

Gabriel Orlanski, Devjeet Roy, Alexander Yun, Changho Shin, Alex Gu, Albert Ge, Dyah Adila, Nicholas Roberts, Frederic Sala, and Aws Albarghouthi. Slopcodebench: Benchmarking how coding agents degrade over long-horizon iterative tasks, 2026. URL https://arxiv.org/abs/ 2603.24755.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=UGpGkLzwpP.

Norman Peitek, Julia Hess, and Sven Apel. From restructuring to stabilization: A large-scale experiment on iterative code readability refactoring with large language models, 2026. URL https://arxiv.org/abs/2602.21833.

Jérémy Perez, Grgur Kovac, Corentin Léger, Cédric Colas, Gaia Molinaro, Maxime Derex, Pierre-ˇ Yves Oudeyer, and Clément Moulin-Frier. When LLMs play the telephone game: Cultural attractors as conceptual tools to evaluate LLMs in multi-turn settings. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= fN8yLc3eA7.

Neil Perry, Megha Srivastava, Deepak Kumar, and Dan Boneh. Do users write more insecure code with ai assistants? In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, CCS ’23, page 2785–2799, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400700507. doi: 10.1145/3576915.3623157. URL https://doi.org/10.1145/3576915.3623157.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. AI models collapse when trained on recursively generated data. Nature, 631(8022):755–759, July 2024. ISSN 1476-4687. doi: 10.1038/s41586-024-07566-y. URL https://www.nature.com/ articles/s41586-024-07566-y.

Maolin Sun, Yibiao Yang, Xuanlin Liu, Yuming Zhou, and Baowen Xu. Debugharness: Emulating human dynamic debugging for autonomous program repair, 2026. URL https://arxiv.org/ abs/2604.03610.

Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, and Neel Nanda. Understanding reasoning in thinking language models via steering vectors. In Workshop on Reasoning and Planning for Large Language Models, 2025. URL https://openreview.net/forum?id= OwhVWNOBcz.

Zhilin Wang, Yafu Li, Jianhao Yan, Yu Cheng, and Yue Zhang. Unveiling attractor cycles in large language models: A dynamical systems view of successive paraphrasing. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12740–12755, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.624. URL https://aclanthology. org/2025.acl-long.624/.

Zihan Wang, Siyao Liu, Yang Sun, Hongyan Li, and Kai Shen. CodeContests+: High-quality test case generation for competitive programming, 2025b. URL https://arxiv.org/abs/2506.05817.

Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, and Sida Wang. SWE-RL: Advancing LLM reasoning via reinforcement learning on open software evolution. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= ULblO61XZ0.

Xuening Wu, Qianya Xu, Yanlan Kang, Zeping Chen, Yubin Liu, and Shenqin Yin. Do language models converge to themselves? recursive self-refinement as textual relaxation, 2026. URL https://arxiv.org/abs/2607.22653.

Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. AxBench: Steering LLMs? even simple baselines outperform sparse autoencoders. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=K2CckZjNy0.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Demystifying LLM-based software engineering agents. Proc. ACM Softw. Eng., 2(FSE), June 2025. doi: 10.1145/3715754. URL https://doi.org/10.1145/3715754.

Guowei Xu, Mert Yuksekgonul, and James Zou. Sparse reward subsystem in large language models, 2026. URL https://arxiv.org/abs/2602.00986.

An Yang, Baosong Yang, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, volume 37, pages 50528–50652. Curran Associates, Inc., 2024b. doi: 10.52202/ 079017-1601. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf.

## A Prompts

In this section, we give the system prompts used in the study. We vary the user prompt program placed between the tags <input\_program> and </input\_program> every turn.

## A.1 Atomic changes

We let the model reason first about the program, name the most important bug, if any, then output a SRB.

Search/Replace blocks   
You are an expert C++ programmer and debugger . You will be given a   
C++ program between the <input\_program > and </ input\_program > tags .   
It may or may not contain bugs .   
Respond in exactly two parts :   
1. Reasoning : in a few sentences , say what the program does and   
whether it looks correct .   
2. Bugs and fixes : identify only the single most important bug , if   
any . Write that one bug on its own line as "- <bug >" and then ,   
immediately below it , the fix as one edit in this format ( and   
nothing else between the markers ):   
- < description of the bug >   
<<<<<<< SEARCH   
( the exact lines to find , copied verbatim from the program )   
(the lines to replace them with )   
>>>>>>> REPLACE   
The SEARCH text must be copied character -for - character from the   
program above -- same indentation , no added or removed whitespace   
-- and be long enough to occur exactly once , but no longer than   
needed . Do not wrap the blocks in backticks or code fences . Do not   
add , expand , or duplicate any comments , and do not narrate your fix   
in the code . Do NOT output the whole program or any code outside   
the SEARCH / REPLACE blocks . If there are no bugs , write exactly   
"- None " and give no edit blocks .

## A.2 Whole-file edits

The model emits the corrected program only.

Whole-file   
You are an expert C ++ programmer and debugger . You will be given a   
C++ program between the <input\_program > and </ input\_program > tags .   
It may or may not contain bugs .   
Output only the corrected C ++ program between these exact tags ,   
each tag on its own line :   
<program >   
( the entire program goes here )   
</ program >   
Do not add , expand , or duplicate any comments : keep exactly the   
comments that appear in the input program and introduce no new   
ones . Do not narrate your fix in the code . Output nothing else . No

additional text or explanation .

## A.3 Bug verdict

Here, we provide the prompt used to construct the steering vector.

Verdict   
You are an expert C++ programmer and debugger . You will be given a   
C ++ program between the < input\_program > and </ input\_program > tags .   
It may or may not contain bugs .   
Answer one question : does the program contain a bug ?   
Respond with exactly one character -- " Y " if it has at least one   
bug , " N " if it is correct . Output nothing else .

## A.4 Issue count

Lastly, we use the following prompt as unseen task,

Issue counting   
You are an expert C++ programmer and debugger .   
You will be given a C ++ program between the < input\_program > and   
</ input\_program > tags .   
Assess the program and return ONLY valid JSON in exactly this   
format :   
{   
" logical\_bugs ": int ,   
" formatting\_issues ": int ,   
" readability\_issues ": int ,   
" performance\_issues ": int ,   
" compile\_issues ": int ,   
" error\_handling\_issues ": int ,   
" other\_issues ": int   
}   
Where each value is the number of issues of that type you found in   
the program .   
Do not include explanations or any text outside the JSON object .

## B Additional experimental details

## B.1 Compute

Experiments involving Gemini 2.5 Flash-Lite were run through the Google AI Studio. We checked that determinism was preserved during iterations, but cannot guarantee the same identical results. We generated around 2B tokens, including failed experiments.

Local experiments with Qwen-2.5-7B-Instruct were run on a single NVIDIA A10G with 24 GB of memory.

## B.2 Code manipulation

We are deliberately lax when parsing code, that is, we parse for Markdown markers, the XML-style tags used in the prompt, or nothing at all (i.e. take the raw output as-is). We found LLMs to be quite inconsistent at following instructions.

Similarly, we are lax when parsing SRBs. We mainly look for <<<<<<< opener as the SEARCH keyword was dropped sometimes, the REPLACE keyword stays mandatory to avoid conflicts with standard C++ syntax. When applying them, (1) blank search blocks are rejected as they match everywhere, (2) we try exact substring match first, and fallback to (3) whitespace equivalent substring if needed, as LLMs might replace tabs with spaces and viceversa.

During judging, the time-limit used is increased four times the provided time-limit in the dataset, this is due to compute differences, and LLMs are not necessarily aware this is a competitive programming setting. In the case no time-limit is provided on the dataset, we set it to 10 seconds. We set a memory cap to 2 GBs. The compiler we use is gnu++17, falling back to gnu++14 if it fails.

## C Example bug-fixing

Here, we select examples of certain behaviours during the bug-fixing loop.

## C.1 Degenerations

```c
Degenerate whole-file edit (Gemini 2.5 Flash-Lite)
# include <iostream >
# include <vector >
# include <set >
# include <algorithm >
# include <climits >
using namespace std;
int main () {
set <int > S;
for ( int i = 0; i < (1 << 10); i++) {
int x = 0;
for ( int j = 0; j < 9; j++) {
x *= 10;
if (i & (1 << j))
x += 4;
else
x += 7;
}
// The do - while loop is incorrect for inserting into the
set .
// It should insert the generated number ’x’ directly .
S . insert ( x) ;
}
vector < int > V ;
V . push_back (0) ; // Add a sentinel value at the beginning
for ( int val : S) {
V . push_back ( val ) ;
}
V . push_back ( INT_MAX ) ; // Add a sentinel value at the end
int pl , pr , vl , vr , k;
cin >> pl >> pr >> vl >> vr >> k ;
long long cnt = 0;
```

```lisp
long long all = (( long long )vr - vl + 1) * (( long long )pr - pl
+ 1);
// The loop condition and indexing need to be adjusted to
correctly iterate through the generated numbers .
// The original loop was ‘i < V . size () - k - 1 ‘ , which is
problematic .
// We need to consider windows of size k +1 in V .
for ( int i = 1; i < V. size () - 1; ++i) {
// Check if the current number V [ i ] is within the range [ vl
, vr]
if (V[i] >= vl && V[i] <= vr) {
// Check if there are k numbers after V[i] in the set
that are within [pl , pr]
int count_in_pr_range = 0;
for ( int j = i; j < V. size () - 1 && j < i + k + 1; ++j)
{
if ( V [ j ] >= pl && V [ j ] <= pr ) {
count_in_pr_range ++;
}
}
if ( count_in_pr_range == k) {
// If V[i] is in [vl , vr] and there are k numbers
in [pl , pr] starting from V[i]
// we need to count the valid intervals .
// The interval for V[i] is [max(vl , V[i]) , min(vr ,
V[i])] which is just V[i] if it ’s in range .
// The interval for the k numbers is [ max ( pl , V [ i ])
, min(pr , V[i+k -1]) ]
// This logic is complex and likely not what was
intended .
( reasoning omitted for brevity )
// The original code ’ s ‘cnt ‘ calculation is : (...)
// The original code ’ s ‘cnt ‘ calculation is : (...)
// The original code ’s ‘cnt ‘ calculation is: (...)
```

## Degenerate reasoning in SRB (Gemini 2.5 Flash-Lite)

1. Reasoning : (...)   
2. Bugs and fixes :   
- (...)   
The most significant bug is (...)   
Given the problem context (...)   
A more appropriate condition would (...)   
Let ’ s assume the problem is (...)   
The most critical bug is (...)   
The bug is in the ‘ while ‘ loop condition (...)   
The bug is in the ‘ while ‘ loop condition (...)   
The bug is in the ‘ while ‘ loop condition ( repeating )

## C.2 Identity replace

Identity SRB (Gemini 2.5 Flash-Lite)   
1. Reasoning : The program attempts to find the next year that has   
the same day of the week for January 1 st as the input year ,   
considering leap years . It appears to have a logic error in its   
loop termination condition .   
2. Bugs and fixes :   
- The ‘break ‘ statement after ‘printf ("%d\n", y);‘ inside the   
‘ while ‘ loop is placed incorrectly , causing the loop to terminate   
prematurely after the first iteration .   
<<<<<<< SEARCH   
printf ("% d\n", y);   
break ;   
}   
}   
printf ("% d \ n " , y );   
break ;   
}   
}   
>>>>>>> REPLACE

## C.3 Cycles

## Length 2 cycle using SRB (Gemini 2.5 Flash-Lite)

```lisp
// Iteration 1
1. Reasoning : (...)
2. Bugs and fixes :
- The code incorrectly calculates the length of the domain part .
The ‘l - k‘ calculation assumes ‘l‘ is the index * after * the last
character of the domain , but the loop condition ‘a[l + 1] >= ’a’
&& a[l + 1] <= ’z’‘ means ‘l‘ stops at the last valid character of
the domain . Therefore , the length should be ‘l - k + 1‘.
<<<<<<< SEARCH
q *= l - k + 1;
======
q *= l - k;
>>>>>>> REPLACE
// Iteration 2
1. Reasoning : (...)
2. Bugs and fixes :
- The code incorrectly calculates the length of the domain part .
The ‘l - k + 1‘ calculation assumes that ‘l‘ points to the last
character of the domain , but the loop condition ‘a[l + 1] >= ’a’
&& a[l + 1] <= ’z’‘ means ‘l‘ stops * before * the last character
if the domain is followed by a non - alphabetic character .
<<<<<<< SEARCH
q *= l - k;
======
q *= l - k + 1;
>>>>>>> REPLACE
```

## C.4 Activation steering

## Hello, World!

# include <bits / stdc ++.h>   
using namespace std ;   
int main () {   
cout << " Hello , World !";

## Baseline

Reasoning : The program prints " Hello , World !" to the console . It looks correct .

Bugs and fixes : - None

## Positive steering (γ = 0.7)

## - <bug >

Reasoning : The program prints " Hello , World !" to the console . It contains the necessary import statements and has a simple print statement inside the main function . This program is correct and contains a bug : it uses a deprecated namespace .

- < description of the bug >   
<<<<<<< SEARCH   
# include <bits / stdc ++.h>   
=======   
# include <iostream >   
>>>>>>> REPLACE

## Hello, Word! (possible typo)

```cpp
# include < bits / stdc ++. h >
using namespace std ;
int main () {
cout << " Hello , Word !";
```

## Baseline

Reasoning : The program is   
intended to print " Hello ,   
World !" to the console .   
However , there is a typo in the   
string being printed .

Bugs and fixes :   
- < description of the bug >   
<<<<<<< SEARCH   
cout << " Hello , Word !";   
=======   
cout << " Hello , World !";   
>>>>>>> REPLACE

## Negative steering (γ = −0.7)

Reasoning : The program is intended to print " Hello , World !" to the console , but the string is actually correct .

Bugs and fixes : None

## D Additional dynamics

## D.1 Steering statistics

Following Section 5, Figure 9 shows how transient length and cycle length distribution change under steering. Positive steering causes the transient length to be on average slightly longer, while under negative steering most runs collapse quickly in the first few rounds.

![](images/4663c8ed5664da8a4020d629990327b95345835212cf97326c7a9e6ae3eb5845.jpg)  
Figure 9: Dynamical system statistics of Qwen2.5 $\tau = 0$ using SRBs under different values of $\gamma ,$ (a) transient legnth, (b) cycle length.

## D.2 Issue count for other layers

We mostly focus on steering layer 22 throughout this report, but we have found that earlier layers loosely encode different types of issue types, while layer 22 has a more balanced view overall. We can see this disparity in Figure 10, we hypothesise that layer 15 encodes overall complexity, layer 16 is focused on formatting issues, and starting at layer 19, bug types get combined.

![](images/6c95ff703ef68c860ac91fb5180f645477d13ddc2b429d3ed6bab2a7115f71a6.jpg)  
Figure 10: Mean issue count difference versus the baseline $( \gamma = 0 )$ at layer (a) 15, (b) 16, (c) 19.

## E Reasoning with whole-file edits

We provide an ablation study, checking that SRBs cause cycling and longer transients, and not the reasoning trace. In Figure 11, we compare SRBs (Atomic) to an alternative whole-file edit where the LLM first provides the same reasoning trace as SRBs, followed by the entirely of the corrected core.

![](images/3186bbdb687f6ba57d75c92d7f58886e9f41d269618f8eccdb1a3b50c1417188.jpg)  
Figure 11: Dynamical system statistics of Gemini 2.5 Flash-Lite $\tau = 0$ using SRBs (in pink), and reasoning followed by whole-file edit (in green). (a) Transient length distribution. (b) Cycle length distribution.