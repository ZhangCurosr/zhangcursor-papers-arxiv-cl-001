Budget B (TFLOPs)

# Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling

Ahsan Bilal<sup>1</sup> Muhammad Ahmed Mohsin<sup>2</sup> Muhammad Umer<sup>2</sup>

Ayesha Mohsin<sup>3</sup> Sarmad Bilal<sup>3</sup> Dean F. Hougen<sup>1</sup>

<sup>1</sup>University of Oklahoma <sup>2</sup>Stanford University <sup>3</sup>National University of Sciences and Technology

## Abstract

Test-time scaling allocates additional inference compute to refining existing solutions or generating new ones. This raises a fundamental question: Should I keep improving this solution, or stop working on it and try a completely new solution? Existing methods make this decision using fixed refinement schedules, selfverification judgments, or learned controllers, which can waste compute or rely on signals that may be incorrect or require additional training. We introduce Adaptive Replay Refinement (ARR), a training-free method that detects when a stationary revision reaches a fixed point and reallocates the released compute to a new independent sample; replay decoding treats the current solution as a draft and verifies unchanged tokens in parallel, enabling efficient fixed-point detection. Across seven models from five families and seven benchmarks, ARR matches majority voting on three of five mathematical reasoning benchmarks at equal compute, while reducing critical-path serial passes by up to 2.4 times. On MATH500, ARR achieves 84.6% accuracy with only 41% of the serial passes required by majority voting. These results demonstrate that detecting refinement exhaustion enables adaptive reallocation from revision to independent sampling, improving inference efficiency while highlighting the limits of self-revision.<sup>1</sup>

## 1 Introduction

Large language models can improve reasoning performance by using more computation at inference time (Snell et al., 2024; Wu et al., 2025; Brown et al., 2024). Existing approaches primarily spend this compute in two ways: Parallel methods generate multiple independent solutions and aggregate their answers, typically through majority voting wasted: already converged useful: still changing

(a) Convergence  
![](images/1ceb1b7d80326acc29f704506e9b8a5f8ff4314cfb6636b78437b8b22e666bd1.jpg)  
(b) Wasted compute  
Number of revisions k

![](images/4bbcaaec8448cdf6d81e4494fe559157a1dd48027bdf3e189cb943f1893576da.jpg)  
Figure 1: Revision on MATH500 with Qwen2.5-Math-7B-Instruct. (a) Fraction of the four initial roots of ARR<sub>Depth</sub> (B=800) at a fixed point: 35% after one revision and 75% by budget exhaustion. (b) Fraction of Self-Refine compute wasted on already converged solutions.

(Wang et al., 2023), while sequential methods repeatedly revise a single solution, often through separate feedback and refinement stages (Madaan et al., 2023; Shinn et al., 2023). Recent approaches combine these strategies by sampling several solutions, refining them, and aggregating the resulting answers (Chen et al., 2025a; Snell et al., 2024).

Combining sampling and refinement creates a fundamental allocation problem: When should additional compute try to improve an existing solution, and when should it generate a new one? Prior work typically relies on one of three mechanisms. Fixed refinement schedules apply a predetermined number of revision rounds, which can waste compute after further refinement stops changing a solution and often require validation data to choose the schedule (Madaan et al., 2023). Self-verification methods stop when the model judges its own solution to be correct (Chen et al., 2025a; Samanta et al., 2026), but such judgments can be unreliable without external feedback (Huang et al., 2024; Stechly et al., 2024; Kamoi et al., 2024). Learned or score-based controllers instead use trained verifiers, predicted difficulty, or confidence estimates (Snell et al., 2024; Wang et al., 2026; Fu et al., 2025), introducing additional training, calibration, or threshold selection. These approaches also incur unnecessary sequential computation because revision commonly regenerates most of a solution even when only a small portion changes. Figure 1 shows that revision exhausts after few steps and that a fixed schedule wastes the rest of its compute.

![](images/1bab6c78244c2cbb81c1701e91cabeb9259c8b6e7cf96cb94445ba56681f46b2.jpg)  
Figure 2: Overview of ARR. (a) Replay decoding accelerates revision. (b) Exact reproduction identifies a fixed point and retires the root. (c) Released compute is reallocated to new roots until the budget is exhausted.

We address this problem by detecting when a deterministic and stationary revision procedure can no longer change a solution. Once a revision exactly reproduces its input, further applications of the same procedure are redundant. We use this event as an exact signal that refinement of that solution has been exhausted. We introduce Adaptive Replay Refinement (ARR), a training-free method that uses replay decoding to efficiently detect when refinement reaches a fixed point and reallocates the released compute to a new independent sample (Figure 2). ARR therefore adapts refinement depth and sampling breadth directly during inference, without requiring a tuned refinement depth, correctness threshold, critic, or learned controller. While draft verification, fixed-point iteration, and training-free self-refinement have each been studied previously (Sun et al., 2021; Leviathan et al., 2023; Santilli et al., 2023; Madaan et al., 2023), ARR uses these components to detect exhausted refinement and redirect compute to new attempts; a fixed point signals only that further revision is redundant, not that the solution is correct. Table 8 in Appendix I compares this stop signal with those of the closest methods, and also discusses sequential scaling, efficient reasoning, and fixed points in sequence models.

Contributions. First, we formulate iterative refinement as an adaptive allocation problem between improving existing solutions and generating new ones, and introduce a training-free rule that reallocates compute once refinement is exhausted. Second, we establish that replay decoding preserves the output of ordinary greedy revision, making exact reproduction a valid fixed-point stopping signal. Third, we introduce a computematched evaluation protocol that accounts for all processed computation and includes controlled variants that isolate the effect of each component. Fourth, across seven models from five model families and seven benchmarks, ARR matches majority voting on three of five mathematical reasoning benchmarks at equal compute while shortening the serial critical path by up to 2.4 times; our analysis also identifies two settings in which the underlying revision procedure itself is ineffective.

## 2 Problem Setup

Let p be a frozen causal language model over vocabulary V, let x be a question, and let $\mathcal { V } \subset \mathcal { V } ^ { * }$ denote the set of complete solutions y that terminate with ε within the output token cap $T _ { \mathrm { m a x } }$ . Two solutions are equal only if they are identical as token sequences. For a prompt u, the greedy map G<sub>θ</sub> returns $z = \mathcal { G } _ { \theta } ( u )$ with

$$
z _ { t } = \underset { v \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \theta } \big ( v \mid u , z _ { < t } \big ) ,\tag{1}
$$

which stops at ε or after $T _ { \mathrm { m a x } }$ tokens. A sample is $y \sim p _ { \theta } ^ { \tau } ( \cdot \mid \sigma ( x ) )$ , where $\sigma ( x )$ is the solve prompt and $p _ { \theta } ^ { \tau }$ is temperature-τ nucleus sampling. The extractor $\phi : \mathcal { V }  \mathcal { A }$ returns the final answer of a solution, and $a ^ { \star } ( x )$ is the correct answer.

## 2.1 Allocating Test-Time Compute

A root r is one sample $y _ { r } ^ { ( 0 ) } \sim p _ { \theta } ^ { \tau } ( \cdot \ | \ \sigma ( x ) )$ together with its revisions $y _ { r } ^ { ( 1 ) } , y _ { r } ^ { ( 2 ) } , \ldots$ , and $\bar { y } _ { r }$ is its last complete solution. A method receives a per-question budget B in FLOPs. At each step, its allocation policy π revises an active root or samples a new root. The policy π allocates the budget B between revising existing roots and sampling new ones, stopping when no further action fits. The final prediction is the majority vote over complete roots, $\begin{array} { r } { \hat { a } _ { \pi } ( x ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \sum _ { r } \mathbb { 1 } [ \phi ( \bar { y } _ { r } ) = a ] } \end{array}$ . Thus, the objective is

$$
\operatorname* { m a x } _ { \pi } \mathrm { P r } \left[ \hat { a } _ { \pi } ( x ) = a ^ { \star } ( x ) \right] \quad \mathrm { s . t . } \quad \mathcal { C } _ { \pi } ( x ) \leq B .\tag{2}
$$

Since $a ^ { \star } ( x )$ is unknown at inference time, π must rely on observable signals; ARR uses the fixedpoint condition $F _ { x } ( y ) = y$ , which holds when the deterministic revision operator exactly reproduces its input, implying that further applications of the same operator leave the solution unchanged.

## 2.2 Stationary Revision Operator

Let $\rho ( x , y )$ be a fixed revision prompt containing the question x and solution $y ,$ and define the stationary revision operator as $F _ { x } ( y ) = \mathcal G _ { \theta } ( \rho ( x , y ) )$ ). The prompt asks the model to correct any errors and their downstream consequences while preserving correct content, and to reproduce $y$ verbatim when no correction is needed (Appendix E). ARR relies on three properties of $F _ { x }$ : it uses a single call that combines critique and correction into one revised solution (Chen et al., 2025a); it is stationary because $\rho$ depends only on $( x , y )$ and contains no round index, revision history, or other roots, unlike prompts that change across rounds (Madaan et al., 2023); and it is deterministic under greedy decoding, making $F _ { x } ( y ) = y$ well defined and confining diversity to sampled roots, consistent with evidence that LLM representations encode future response attributes before generation (Dong et al., 2025).

## 3 Adaptive Replay Refinement

ARR is a training-free policy for Equation (2). It revises each root with $F _ { x }$ to a fixed point and then gives the slot of the root to a new sample as shown in Figure 2. Replay decoding computes $F _ { x } ( y )$ with y as the draft, so the fixed-point test costs few serial passes (Sections 3.1 and 3.2), and this test drives the allocation (Section 3.3). ARR has three settings: the width N, i.e., the number of concurrently active roots, the budget $B .$ , and the block size $W$ Algorithms 1 and 2 in Appendix A give the pseudocode.

## 3.1 Replay Decoding

A revision usually reproduces a long prefix of its input, but greedy decoding spends one serial forward pass on each reproduced token. Replay decoding uses the input $y$ as a draft and verifies up to $W$ draft positions in one pass.

Block verification. Let $u = \rho ( x , y )$ , and assume that the committed output agrees with the draft, $z _ { 1 : t } ~ = ~ y _ { 1 : t }$ . Replay takes the next $m =$ min $( W - 1 , | y | - t )$ draft tokens and runs one causal pass on $\left( z _ { t } , y _ { t + 1 } , \ldots , y _ { t + m } \right)$ , which gives, for $j = 1 , \ldots , m { + } 1$ , the greedy predictions

$$
\hat { z } _ { t + j } = \underset { v \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \theta } \big ( v \mid u , y _ { 1 : t + j - 1 } \big ) .\tag{3}
$$

Replay accepts the longest prefix of the block on which the predictions agree with the draft, of length $\lambda = \operatorname* { m a x } \{ i \leq m : { \hat { z } } _ { t + j } = y _ { t + j } \forall j \leq i \}$ . It appends $y _ { t + 1 : t + \lambda }$ and the model token $\widehat { z } _ { t + \lambda + 1 } \mathrm { t o } z ,$ and it discards the cache of the rejected positions. $\operatorname { I f } \hat { z } _ { t + \lambda + 1 } = y _ { t + \lambda + 1 }$ , the output is still on the draft, and replay verifies the next block. Otherwise, the revision left the draft, and replay decodes the rest with Equation (1).

Design choices. Replay verifies only a contiguous draft prefix, since tokens beyond the first mismatch are conditioned on an invalid prefix. Unlike aggressive and prompt lookup decoding, which may realign after mismatches (Sun et al., 2021; Saxena, 2023) (Section 4.1), replay switches immediately to standard greedy decoding, preserving exact equivalence to $F _ { x }$ while incurring at most one failed block per revision. Thus, replay verifies the existing solution against a new revision, and when the entire draft is accepted without changes, ARR detects a fixed point and reallocates compute to a new sample.

## 3.2 Exact Fixed-Point Detection

Let $\widetilde { F } _ { x , W } ( y )$ be the output of replay decoding. Replay changes only how the revision is computed, not its output: under deterministic argmax decoding with fixed tie-breaking, $\widetilde { F } _ { x , W } ( y ) = F _ { x } ( y )$ for every $y \in \mathcal { V }$ and $W \geq 1$ , since each verified token matches the greedy continuation in Equation (1). Because $\rho$ depends only on $( x , y )$ and greedy decoding is deterministic, $\widetilde { F } _ { x , W } ( y ) = y$ implies $F _ { x } ^ { k } ( y ) = y$ for all $k \geq 1$ . Thus, accepting the entire draft, including ε, exactly identifies an exhausted revision, while any disagreement implies $F _ { x } ( y ) \neq y$ . Appendix G provides the implementation details and backend validation across four architectures.

Serial cost. Let $z = F _ { x } ( y )$ , and let ℓ be the length of the longest common prefix of z and $y .$ Greedy decoding uses 1 + |z| serial passes. Replay uses about $1 + \lceil \ell / W \rceil + ( | z | - \ell )$ , because it verifies the reproduced prefix in blocks and decodes only the changed suffix token by token. A fixed point with $| z | = 5 1 2$ and $W = 3 2$ thus takes 17 passes instead of 513. The processed positions, and thus the FLOPs, stay almost the same, because a revision discards at most W−1 positions. The saving is in serial depth, which sets the latency.

## 3.3 Fixed-Point Replacement

ARR starts with N roots, one for each slot. Let $z = \widetilde { F } _ { x , W } ( y _ { r } ^ { ( k ) } )$ be the revision of an active root r at depth k. ARR applies

$$
\begin{array} { r } { \{ \begin{array} { l l } { y _ { r } ^ { ( k + 1 ) } = z , } & { z \in \mathcal { V } \setminus \{ y _ { r } ^ { ( k ) } \} , } \\ { \bar { y } _ { r } = y _ { r } ^ { ( k ) } , \ \mathcal { R }  \mathcal { R } \cup \{ r ^ { \prime } \} , } & { z = y _ { r } ^ { ( k ) } , } \\ { \bar { y } _ { r } = y _ { r } ^ { ( k ) } , } & { z \notin \mathcal { V } , } \end{array}  } \end{array}\tag{4}
$$

where R is the set of roots and $r ^ { \prime }$ is a new root, sampled from $\sigma ( x )$ , that takes the slot of $r .$ . In the first case, the solution still changes, and ARR revises it again. In the second case, ${ \bf { \bar { \rho } } } _ { y _ { r } } ( k )$ is a fixed point (Section 3.2): its answer becomes the final vote of the root, and the slot samples a new root if a complete sample still fits in B. In the third case, $T _ { \mathrm { m a x } }$ or the budget truncated the revision: the root keeps its last complete solution, and ARR does not refill the slot.

Equation (4) is the full allocation policy, with no depth limit, threshold, score, or learned parameter. A new root does not observe the retained answers, making it an independent attempt; ARR returns the majority answer $\hat { a } _ { \pi } ( x )$ of Section 2.1, with Appendix F illustrating replay and replacement through a worked example.

## 3.4 Compute Accounting

We use a single FLOP accounting for all methods. The total cost of a method on question x is the sum of the FLOPs of every processed token position, $\begin{array} { r } { \mathcal { C } _ { \pi } ( x ) = \sum _ { i \in \mathcal { P } _ { \pi } ( x ) } \kappa ( c _ { i } ) } \end{array}$ , where $\kappa ( c _ { i } )$ is the FLOP cost of processing position i when it attends to $c _ { i }$ cached keys (Appendix C). This accounting includes all computation: prompt prefills, verified draft tokens, discarded verification work, decoded tokens, and truncated calls. Thus, no replay or revision computation is treated as free. We separately measure latency by the number of serial forward passes $S$ on the critical path of each question.

## 4 Experiments

We describe the setup in Section 4.1, present the main results in Section 4.2, analyze when revision should receive compute in Section 4.3, analyze individual components in Section 4.4, evaluate generality across models in Section 4.5, and examine failure modes in Section 4.6.

## 4.1 Setup

Benchmarks. We evaluate on five mathematical reasoning benchmarks with exact-answer scoring: MATH500 (Hendrycks et al., 2021) (500 problems), AIME (30 problems per edition), AMC (40 problems), and the 674 English text-only competition mathematics problems from OlympiadBench (He et al., 2024). We include GSM8K (Cobbe et al., 2021) (1,319 problems) as a saturated control and evaluate generalization beyond mathematics on LiveBench reasoning tasks (White et al., 2025), comprising web of lies, zebra puzzles, and spatial reasoning (50 problems each). Appendix B details answer extraction and scoring.

Models. The primary model is Qwen2.5- Math-7B-Instruct (Yang et al., 2024). For breadth, we evaluate Qwen3-8B (Yang et al., 2025a), Ministral-8B-Instruct-2410 (Mistral AI Team, 2024), Gemma-4-E4B-it (Gemma Team, 2026), and Qwen3.5-9B and 2B (Qwen Team, 2026). These models span multiple families, scales, and decoding-state architectures, including full, sliding-window, and hybrid linear attention, testing whether replay-based fixed-point detection generalizes across inference backends rather than a specific cache design. For the 7-8B models, we use budgets of $B \in \{ 2 0 0 , 4 0 0 , 8 0 0 \}$ TFLOPs; for models with longer generations, budgets are scaled to preserve comparable sampling capacity across budget levels (Appendix C converts each budget to a number of samples), with FLOPs computed using the common FLOP accounting in Appendix C. We additionally evaluate Qwen2.5-7B-Instruct (Yang et al., 2025b) on LiveBench.

Compared methods. We select baselines spanning independent sampling, fixed-depth and stochastic revision, explicit self-verification, and confidence-based allocation, alongside controlled ARR variants that isolate decoding and root replacement. All methods use the same FLOP accounting, $T _ { \mathrm { m a x } }$ , and answer extractor ϕ; root-based methods additionally share N=4 initial roots and majority aggregation, while ConMA retains its group-based selection procedure (Table 1, Appendix D). Majority Voting (Wang et al., 2023) spends the entire budget on independent samples from $\sigma ( x )$ , providing a pure-sampling baseline for evaluating the benefit of revision. $\mathbf { A R R } _ { \mathbf { A R } }$ and $\mathbf { A R R } _ { \mathbf { P L D } }$ retain ARR’s allocation rule but compute $F _ { x }$ with autoregressive greedy decoding and prompt lookup decoding (Saxena, 2023), respectively, isolating the effect of replay from the underlying allocation policy; their single revisions are equivalent to ARR’s under Section 3.2, while full runs can differ because of decoding cost and scheduling (Appendix D). $\mathbf { A R R } _ { \mathbf { D e p t h } }$ removes root replacement from Equation (4) to isolate the effect of reallocating exhausted depth to breadth, while Self-Refine (Madaan et al., 2023) applies $F _ { x }$ for a fixed D=4 revisions, providing an iteration-limited baseline for assessing redundant refinement. SETS (Chen et al., 2025a) replaces $F _ { x }$ with a two-stage critiqueand-correction update, $F _ { x } ^ { \mathrm { v c } }$ , while keeping ARR’s replacement rule, isolating the effect of explicit self-verification on revision quality. Sequential Revision (Snell et al., 2024) repeatedly samples a non-stationary continuation from $\rho _ { \mathrm { c o n t } } ( x , y _ { r } ^ { \bar { ( k ) } } , k )$ until the budget is exhausted, contrasting stochastic, depth-dependent revision with ARR’s deterministic stationary operator. ConMA (Wang et al., 2026) is a confidence-based controller: it samples groups of trajectories, keeps the kernel of answers with the highest token-level confidence, and re-samples or asks a multiple-choice question until one answer remains, providing an adaptive alternative based on confidence rather than exact fixed-point detection (Appendix D). We fix N=4 for root-based methods and W=32 for ARR’s replay decoder across models without tuning; replay exactness (Section 3.2) ensures that W affects decoding speed but not revision output.

(e) AIME25  
![](images/ffc6f80e3c1f4199139422ad5b7e956806ed121f37f17a8410500e73a650675c.jpg)

![](images/08b33ec666fdd460ef7eca3a38bb5fa629766e98d30de3ad4f84549d8dcf8c73.jpg)

![](images/48d97ae5b21145c09f621c018e50dced741bffb6b427a024322db65344634a20.jpg)

![](images/345ddee0846e4b2751bf1ef19ca9e9f57c64631429c032ec0224d988a52b3d82.jpg)

![](images/1415a9efe94d6eea045d127f74020a02eb9148b739aeebd1ca0d98c4f7dad78d.jpg)

![](images/8e2efe9f2b66e70e25639f94762c2ab68797fdf9ac72410db73a32e600888b0a.jpg)  
Median serial forward passes on the critical path (log scale)  
Figure 3: Accuracy against median critical-path serial passes (Qwen2.5-Math-7B-Instruct); each point is one budget.

Metrics. We report accuracy, mean TFLOPs spent, and median serial passes S per question, with 95% bootstrap confidence intervals computed over question-paired differences relative to ARR. All experiments are run with multiple random seeds (three to five per setting), and all reported results are averaged over seeds (Table 2: paired seeds only). Using matched initial roots, we define the net repair of $F _ { x }$ as the difference between incorrectto-correct and correct-to-incorrect answer changes, normalized by the number of complete revisions. Appendix G details the mechanism statistics, decoder, and hardware.

## 4.2 Main Results

ARR gives more accuracy for each serial pass. Figure 3 shows accuracy against the median number of serial passes on the critical path. This number sets the latency of a question. On MATH500, AMC23, and AIME24, the curve of ARR is above the curves of $\operatorname { A R R } _ { \operatorname { A R } } .$ , SETS, and Majority Voting for most of the range. For example, ARR gets the best accuracy of Majority Voting on MATH500 (84.6%) with only 41% of the serial passes of Majority Voting. On OlympiadBench, the curves of ARR and Majority Voting are almost the same, and on AIME25 and GSM8K, Majority Voting is better.

At equal FLOPs, ARR matches majority voting on three of five math benchmarks. At the largest budget, ARR and Majority Voting have the same accuracy within noise on MATH500, AMC23, and AIME24 (Figure 6, Appendix H). On OlympiadBench and AIME25, Majority Voting is better by up to 7.3 points. The cause is the number of votes. Majority Voting spends all of the budget on independent samples, but ARR also spends compute on revisions and thus collects approximately six times fewer votes (Table 5). More votes help most when revision repairs few answers (Section 4.3). All other methods that revise get lower accuracy, and $\Delta \mathrm { R R } _ { \mathrm { D e p t h } }$ and Self-Refine also spend fewer FLOPs (Figure 7).

ARR and majority voting give different tradeoffs. Each slot of Majority Voting samples its votes in sequence, so its critical path is long. At equal FLOPs, ARR needs only 41-61% of the serial passes of Majority Voting. On one A100 GPU and a MATH500 subset, ARR answers a question in a median of 78.5 s, and Majority Voting answers it in 176.5 s with the same accuracy (Table 7, Appendix H). At equal latency, the result changes: with a limit of 8,000 serial passes on MATH500, the two methods get the same accuracy, but Majority Voting uses less than half of the FLOPs (Figure 17). Thus, ARR is more efficient in latency at a fixed FLOP budget, and Majority Voting is more efficient in FLOPs at a fixed latency.

Replay makes revision faster without a change in output. Replay gives the same revisions as $\mathrm { A R R } _ { \mathrm { A R } }$ and decreases the critical path by up to 30%. This decrease becomes larger with the budget, because deeper roots reproduce longer prefixes (Figure 8). Because replay verifies reused text in parallel, ARR also writes fewer tokens than Majority Voting: on MATH500, it writes approximately 40% of the output tokens of Majority Voting for the same accuracy.

A confidence controller saves compute but loses accuracy. ConMA stops generating samples once only one answer remains under consideration, reducing inference cost through early stopping. Across completed settings, it uses approximately half of the compute budget and requires half as many serial passes as ARR (Figure 7). However, this efficiency comes at the cost of accuracy: at B=800, ConMA performs 1.2-9.5 percentage points worse than ARR (Figure 6). This suggests that confidence-based stopping can save compute but may terminate sampling too early, potentially missing better answers.

## 4.3 When Revision Should Receive Compute

Revision provides early gains but quickly saturates. Revision improves root accuracy by 0.5-2.4 points, but 49% of the first revisions of ARR (over all roots, including new ones) reach a fixed point, and answer changes become rare after four revisions (Figure 4). By reallocating compute from converged roots to new samples, ARR reaches 84.6% accuracy, matching Majority Voting and outperforming $\Delta \mathrm { R R } _ { \mathrm { D e p t h } }$ (81.5%; Figure 4).

The fixed point is an exact signal, not necessarily the most accurate stop. ARR’s condition $F _ { x } ( y ) = y$ guarantees that further deterministic revisions leave the solution unchanged, without requiring answer extraction, thresholds, or correctness labels. We compare two simpler rules with the same root replacement: (i) answer stability, which stops when $\phi ( F _ { x } ( y ) ) = \phi ( y )$ , and (ii) one-revision stopping, which retires a root after a single revision. At $B { = } 8 0 0$ , answer stability outperforms ARR by 4.0 points on AIME25 (95% CI $[ - 2 . 0 , + 1 0 . 7 ] )$ and 2.2 on OlympiadBench $( [ + 0 . 3 , + 4 . 2 ] )$ ), while both stopping rules remain within 1 point on MATH500 and GSM8K (Figure 5(b), Table 2). These rules retire roots before their solutions converge, generating 1.8-2.4 times more roots but requiring 1.09- 1.29 times more serial passes (Figure 5(c)). Thus, fixed-point stopping gives the shortest critical path among the tested rules, although earlier stopping can sometimes improve accuracy.

The net repair of $F _ { x }$ decides between revision and sampling. Over different model-benchmark settings, the net repair of $F _ { x }$ predicts the accuracy difference between ARR and Majority Voting (Pearson $r { = } 0 . 7 6$ , Figure 5(a)). With a net repair above 0.25%, ARR is 1.3 points better than Majority Voting on average; with a negative net repair, Majority Voting is 8.7 points better, most of all on LiveBench and with Qwen3.5-2B (Section 4.6). Revision should thus receive compute only when it repairs more answers than it breaks, and this condition explains both the successes and the failures of ARR. In two Gemma-4 settings, ARR is 7.0- 14.5 points better for a reason other than reasoning:

(a) Qwen2.5-Math-7B (a) Qwen3-8B (a) Ministral-8B (a) Gemma-4-E4B (a) Qwen3.5-9B (a) Qwen3.5-2B (a) Qwen2.5-7B (b, c) stop when the answer is stable (b, c) stop after one revision (b) Majority Voting  
![](images/d47b06c9b7347c327ec224bc4b466a1ea3344dff682db2cc90c04a264766f442.jpg)

![](images/5cfef1c14fc8ff97a5e48a262501e59ddc2fba7c0bdf0a7eecd6a60743907ad9.jpg)

![](images/f5d9a463c7f4273e5d2ece889474b67886c5ddd217b887a39c4de27c0c23c06c.jpg)  
<sup>(a)</sup> <sup>after</sup> <sup>k</sup> <sup>revisions</sup> <sup>(a)</sup> <sup>same</sup> <sup>roots</sup> <sup>at</sup> <sup>k=0</sup> <sup>(a)</sup> <sup>new</sup> <sup>sample</sup> <sup>(b)</sup> <sup>wrong</sup> ! <sup>correct</sup> <sup>(b)</sup> <sup>correct</sup> ! <sup>wrong</sup> (c) ARR (ours) (c) ARR<sub>AR</sub> (c) ARR<sub>Depth</sub> (c) Majority Voting

Figure 4: Revision in ARR on MATH500 (B=800). (a) Accuracy of roots by revision depth; bars: roots that reach depth k. (b) Answer changes per revision; bars: fixed-point rate. (c) Accuracy of the running vote.  
![](images/dc542805bd5b179b41e25e736804b0bbd2305e008446cf620d93541e641d5295.jpg)

![](images/d225864e98fa5d1a7a8614fd5ce6794bad5d4dd82f6b87b0c96a1375bcde1338.jpg)

![](images/6fbc35162b1d843e2ea98a22a60be554cc2e7a0c72117979037fcc79d8fd1dc2.jpg)  
Figure 5: (a) Net repair of $F _ { x }$ vs. ARR − Majority Voting; open = LiveBench, circled = repaired answer format. (b) Accuracy of the other stopping rules and Majority Voting relative to ARR, with 95% CIs. (c) Their roots and serial passes relative to ARR.

37-41% of the samples end with a boxed answer, against 98-100% after revision.

## 4.4 Component Analysis

The exchange of depth for breadth is the source of the gain. $\mathrm { \Delta A R R _ { D e p t h } }$ keeps the fixed-point test but does not replace retired roots. Thus, it stops when its four roots get to fixed points, and it uses less than half of the budget on MATH500. ARR gives this released compute to new roots and is 3.1 points more accurate on MATH500 and 4.0 points more accurate on OlympiadBench. Figure 10 shows the two sources of gain separately. Revision of the initial roots adds at most 2.5 points. The new roots add 3.0-6.0 points, and this gain increases with the budget. The same order holds for all models.

A greedy and stationary revision operator works best. We compare two revision alternatives (Table 1) across FLOP budgets (Figure 6). SETS adds an explicit critique step, reducing MATH500 accuracy by 1.5 points while remaining within noise on other benchmarks. Despite reaching more fixed points (69% vs. 23%), it repairs fewer answers (Table 5). Sequential Revision replaces greedy revision with sampled continuation, causing MATH500 accuracy to drop from 79.3% to 59.0% as the budget increases, 25.6 points below ARR at B=800 (Figure 6(a)). It also breaks more correct answers than it repairs (Table 5).

The allocation rule does not depend on replay. $\mathrm { A R R } _ { \mathrm { P L D } }$ uses the same rule with prompt lookup decoding. It gets similar accuracy with approximately 55% of the serial passes of ARR. The cause is the first change in a revision. On average, a revision reproduces only 26-35% of its input before this change. After this point, replay decodes one token at a time, but prompt lookup can align to the draft again. Thus, a decoder with realignment makes ARR faster, and the fixed-point signal stays the same. In wall-clock time, $\mathrm { A R R } _ { \mathrm { P L D } }$ needs 54% of the latency of ARR (Table 7, Appendix H).

ARR is robust to its settings. A larger width helps: with N=8, ARR gains 1.9 points over N=4 and needs fewer serial passes. The block size W changes the serial passes by at most 5%, and two paraphrases of ρ change the accuracy by at most 1.2 points. Appendix H gives all ablations and the detailed values of this section.

## 4.5 Generality across Models

The results transfer to four more model families. Figure 9 shows the five other models at their largest budget, over different settings. Three trends of the primary model also hold for these models. First, the exchange of depth for breadth helps: ARR is better than $\mathrm { \Delta A R R _ { D e p t h } }$ and Self-Refine in most of the settings, by approximately 6 points on average. Second, the decoder does not change accuracy: ARR, $\mathrm { A R R } _ { \mathrm { A R } }$ , and $\mathrm { A R R } _ { \mathrm { P L D } }$ are within 0.4 points. Third, ARR and Majority Voting are close, and Majority Voting is 1.5 points better on average. This difference decreases to 0.7 points when we remove Qwen3.5-2B, where revision breaks answers (Section 4.6), and two Gemma-4 settings, where revision only repairs the answer format.

## 4.6 Failure Modes

A fixed point shows that revision is exhausted, not that the solution is correct. Thus, ARR is weak when revision breaks more answers than it repairs. On LiveBench, Majority Voting is better for all five models, by up to 20.4 points (Figures 9(f) and 5(a)). With Qwen3.5-2B, revisions are very short and often give only a final answer, and ARR is 15-24 points below Majority Voting on MATH500 and AMC23 (Figure 9(a,c)). The accuracy of ARR is therefore limited by the quality of its revision operator.

## 5 Related Work

We review parallel test-time scaling and sample aggregation in Section 5.1, self-refinement and selfcorrection in Section 5.2, and draft-and-verify decoding in Section 5.3.

## 5.1 Parallel Test-Time Scaling and Sample Aggregation

Trained verifiers can rank candidate solutions (Cobbe et al., 2021; Lightman et al., 2023), whereas self-consistency samples multiple chain-of-thought solutions and returns the most frequent answer (Wang et al., 2023). Adaptive-Consistency (Aggarwal et al., 2023) and ESC (Li et al., 2024) terminate sampling once answer agreement is sufficiently strong. Snell et al. (2024) allocate inference compute between sampling and revision using a learned verifier and predicted problem difficulty, and solution coverage generally increases as more samples are generated (Brown et al., 2024; Wu et al., 2025). DeepConf (Fu et al., 2025) and ConMA (Wang et al., 2026) instead use token-level confidence signals to filter or expand candidate groups.

## 5.2 Self-Refinement and Self-Correction

Welleck et al. (2022) train a separate corrector that iteratively revises the outputs of a generator. Reflexion (Shinn et al., 2023) retains verbal feedback across trials, while Self-Refine (Madaan et al., 2023) alternates self-generated feedback with iterative revision. Self-Debug (Chen et al., 2023b) and CRITIC (Gou et al., 2024) incorporate execution feedback or external tools. Without such feedback, intrinsic self-correction often fails to improve reasoning and planning performance (Huang et al., 2024; Stechly et al., 2024; Kamoi et al., 2024). Other approaches train multi-turn self-correction directly through supervised or reinforcement learning (Qu et al., 2024; Kumar et al., 2024). SETS (Chen et al., 2025a) repeatedly verifies and corrects sampled solutions until receiving a “correct” verdict before aggregation, whereas Thought-ICS (Samanta et al., 2026) identifies the first erroneous reasoning step and resamples from the preceding prefix.

## 5.3 Draft-and-Verify Decoding

Aggressive decoding (Sun et al., 2021) verifies spans copied from the input in parallel. Speculative decoding accelerates autoregressive generation by drafting multiple tokens and verifying them with the target model in parallel (Leviathan et al., 2023; Chen et al., 2023a). LLMA (Yang et al., 2023) reuses spans from a reference sequence, while prompt lookup decoding (Saxena, 2023) and PLD+ (Somasundaram et al., 2025) retrieve candidate continuations directly from the prompt.

## 6 Conclusion

ARR reframes test-time scaling as an online allocation problem between improving an existing trajectory and exploring a new one. Our results suggest that allocation does not require predicting correctness directly; detecting when refinement stops being useful can be enough to redirect compute to new attempts. This perspective opens several directions, including stronger or task-adaptive revision operators, decoders that realign after edits, integration with verifiers and reasoning models, and richer signals for deciding when to revise, resample, or terminate. More generally, progress-aware allocation could enable inference systems to allocate compute based on the marginal utility of continued reasoning, prioritizing useful reasoning over fixed schedules.

## Limitations

ARR requires access to model logits and the decoding loop, limiting its use with text-only APIs or hidden-reasoning systems. A fixed point indicates only that deterministic refinement is exhausted, not that the solution is correct, so ARR relies on independent replacement roots and aggregation. Finally, ARR prioritizes lower serial latency over matched-FLOP accuracy; its weaker regimes on LiveBench and Qwen3.5-2B motivate evaluating stronger revision operators and larger reasoning models.

## AI Use Statement

ChatGPT by OpenAI and Claude by Anthropic were used for minor language editing, proofreading, and resolving LAT X formatting issues. Codex by OpenAI assisted with routine code implementation, refactoring, plotting, and table generation for Adaptive Replay Refinement (ARR). All AI-assisted content and code were reviewed and verified by the authors, who take full responsibility for the methodology, experimental results, and manuscript.

## Ethics Statement

This work investigates test-time compute allocation between iterative self-refinement and independent solution sampling using publicly available language models and established reasoning benchmarks. The study involves no human subjects, collection of personally identifiable information, or new dataset creation.

## References

Pranjal Aggarwal, Aman Madaan, Yiming Yang, and Mausam. 2023. Let’s sample step by step: Adaptiveconsistency for efficient reasoning and coding with llms. Preprint, arXiv:2305.11860.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. 2019. Deep equilibrium models. Preprint, arXiv:1909.01377.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys: Scaling inference compute with repeated sampling. Preprint, arXiv:2407.21787.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. Medusa: Simple llm inference acceleration framework with multiple decoding heads. Preprint, arXiv:2401.10774.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023a. Accelerating large language model decoding with speculative sampling. Preprint, arXiv:2302.01318.

Jiefeng Chen, Jie Ren, Xinyun Chen, Chengrun Yang, Ruoxi Sun, Jinsung Yoon, and Sercan Ö Arık. 2025a. Sets: Leveraging self-verification and selfcorrection for improved test-time scaling. Preprint, arXiv:2501.19306.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025b. Do not think that much for 2+3=? on the overthinking of o1-like llms. Preprint, arXiv:2412.21187.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. 2023b. Teaching large language models to self-debug. Preprint, arXiv:2304.05128.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Zhichen Dong, Zhanhui Zhou, Zhixuan Liu, Chao Yang, and Chaochao Lu. 2025. Emergent response planning in LLMs. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 14301–14320. PMLR.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. 2024. Break the sequential dependency of llm inference using lookahead decoding. Preprint, arXiv:2402.02057.

Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. 2025. Deep think with confidence. Preprint, arXiv:2508.15260.

Gemma Team. 2026. Gemma 4 model card. https: //huggingface.co/google/gemma-4-E4B-it.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. Critic: Large language models can selfcorrect with tool-interactive critiquing. Preprint, arXiv:2305.11738.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. Preprint, arXiv:2402.14008.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. Preprint, arXiv:2310.01798.

Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. 2024. When can llms actually correct their own mistakes? a critical survey of selfcorrection of llms. Preprint, arXiv:2406.01297.

Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. 2024. CLLMs: Consistency large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 25426–25440. PMLR.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, and Aleksandra Faust. 2024. Training language models to self-correct via reinforcement learning. Preprint, arXiv:2409.12917.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. Preprint, arXiv:2309.06180.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 19274–19286. PMLR.

Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Xinglin Wang, Bin Sun, Heda Wang, and Kan Li. 2024. Escape sky-high cost: Early-stopping self-consistency for multi-step reasoning. Preprint, arXiv:2401.10480.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2025. Eagle: Speculative sampling requires rethinking feature uncertainty. Preprint, arXiv:2401.15077.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. Preprint, arXiv:2305.20050.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,

Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. Preprint, arXiv:2303.17651.

Mistral AI Team. 2024. Un ministral, des ministraux: Introducing the world’s best edge models. https: //mistral.ai/news/ministraux.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. Preprint, arXiv:2501.19393.

Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. 2024. Recursive introspection: Teaching language model agents how to self-improve. Preprint, arXiv:2407.18219.

Qwen Team. 2026. Qwen3.5 model card. https:// huggingface.co/Qwen/Qwen3.5-9B.

Ankur Samanta, Akshayaa Magesh, Ayush Jain, Kavosh Asadi, Youliang Yu, Daniel R. Jiang, Boris Vidolov, Kaveh Hassani, Paul Sajda, Jalaj Bhandari, and Yonathan Efroni. 2026. Structure enables effective self-localization of errors in LLMs. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 107412–107440. PMLR.

Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodolà. 2023. Accelerating transformer inference for translation via parallel decoding. Preprint, arXiv:2305.10427.

Apoorv Saxena. 2023. Prompt lookup decoding. https://github.com/apoorvumang/ prompt-lookup-decoding.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Preprint, arXiv:2303.11366.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. Preprint, arXiv:2408.03314.

Shwetha Somasundaram, Anirudh Phukan, and Apoorv Saxena. 2025. PLD+: Accelerating LLM inference by leveraging language model artifacts. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 6090–6104, Albuquerque, New Mexico. Association for Computational Linguistics.

Kaya Stechly, Karthik Valmeekam, and Subbarao Kambhampati. 2024. On the self-verification limitations of large language models on reasoning and planning tasks. Preprint, arXiv:2402.08115.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, Hanjie Chen, and Xia Hu. 2025. Stop overthinking: A survey on efficient reasoning for large language models. Preprint, arXiv:2503.16419.

Xin Sun, Tao Ge, Furu Wei, and Houfeng Wang. 2021. Instantaneous grammatical error correction with shallow aggressive decoding. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5937–5947, Online. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. Preprint, arXiv:2203.11171.

Yinuo Wang, Qingjie Li, Wenyao Cui, Qiuchi Li, and Zhang Huaping. 2026. ConMA : Confidence-guided kernel sampling with multi-stage aggregation for LLM reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 29500– 29518, San Diego, California, United States. Association for Computational Linguistics.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. 2022. Generating sequences by learning to self-correct. Preprint, arXiv:2211.00053.

Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. 2025. Livebench: A challenging, contamination-limited llm benchmark. Preprint, arXiv:2406.19314.

Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. 2025. Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models. Preprint, arXiv:2408.00724.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025b. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. Preprint, arXiv:2409.12122.

Nan Yang, Tao Ge, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, and Furu Wei. 2023. Inference with reference: Lossless acceleration of large language models. Preprint, arXiv:2304.04487.

Zewei Yu, Lirong Gao, Yuke Zhu, Bo Zheng, Junbo Zhao, Sheng Guo, and Haobo Wang. 2026. Stop unnecessary reflection: Training lrms for efficient reasoning with adaptive reflection and length coordinated penalty. Preprint, arXiv:2602.12113.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. Sglang: Efficient execution of structured language model programs. Preprint, arXiv:2312.07104.

Algorithm 1 REPLAY(x, y, W) (Section 3.1)   
Require: question x, solution $y \in \mathcal { P } ,$ block size W   
1: u ← ρ(x, y); prefill u; z ←   arg max p<sub>θ</sub>(v | u)   
2: on $ [ z _ { 1 } = y _ { 1 }$ ▷ z is a prefix of y   
3: while $\dot { z } _ { | z | } \neq \varepsilon$ and $| z | < T _ { \mathrm { m a x } }$ do   
4: t ← |z|   
5: if on then   
6: m ← min $( W - 1 , | y | - t )$   
7: pass $( z _ { t } , y _ { t + 1 : t + m } )  \hat { z } _ { t } .$ +1:t+m+1 ▷ Eq. (3)   
8: λ ← accepted length ▷ Section 3.1   
9: z ← z ⊕ y<sub>t+1:t+λ</sub>   
10: discard cache of y<sub>t+λ+1:t+m</sub>   
11: if $z _ { | z | } \neq \varepsilon$ then   
12: $\mathcal { z }  z \oplus \hat { z } _ { t + \lambda + 1 }$   
13: on $ [ \widehat { z } _ { t + \lambda + 1 } = y _ { t + \lambda + 1 } ]$   
14: end if   
15: else   
16: z ← z ⊕ arg max p<sub>θ</sub>(v | u, z) ▷ Eq. (1)   
17: end if   
18: end while   
19: return z

A Pseudocode 12   
B Decoding, Formatting, and Answer Ex  
traction 12   
C Compute Accounting and Budgets 12   
D Compared Methods 14   
E Prompt Templates 14   
F A Worked Example of Replay and Re  
placement 16   
G Implementation Details 18   
H Additional Results 20   
I Extended Related Work 22   
I.1 More draft-and-verify methods. . . 22   
I.2 Sequential scaling and efficient rea  
soning. . . . 22   
I.3 Fixed points in sequence models. . 22

## A Pseudocode

Algorithm 1 computes one revision with replay decoding, and Algorithm 2 runs the allocation rule of Equation (4) under the budget B. Jobs of all roots share one first-in-first-out queue. An initial sample that is not complete is charged but never votes, and its slot samples again (SPAWN).

Algorithm 2 ARR(x, N, B, W) (Section 3.3)   
Require: question x, width N, budget B, block size W   
1: $\mathbf { \bar { \mathcal { R } } }  \{ 1 , \dotsc , N \}$   
2: FIFO queue $Q \gets ( \mathrm { S A M P L E } ( r ) ) _ { r \in \mathcal { R } }$   
3: while $\dot { Q } \neq \emptyset$ and the next job fits in B do   
4: $j  Q . \mathrm { p o p } ( )$   
5: $\mathbf { i f } \ j = \mathbf { S A M P L E } ( r )$ then   
6: $y _ { r } \sim p _ { \theta } ^ { \tau } ( \cdot \mid \sigma ( x ) )$   
7: if $y _ { r } \in \mathcal { V }$ then   
8: push REVISE(r)   
9: else   
10: SPAWN() ▷ invalid sample   
11: end if   
12: else ▷ j = REVISE(r)   
13: $z \gets \mathrm { R E P L A Y } ( x , y _ { r } , W )$ ▷ Algorithm 1   
14: $\mathbf { i f } \ z \not \in \mathcal { V }$ then   
15: $\bar { y } _ { r } \gets y _ { r }$ ▷ truncated: freeze   
16: else if $z = y _ { r }$ then   
17: $\bar { y } _ { r } \gets y _ { r } ; \mathbf { S p A W N } ( )$ ▷ fixed point   
18: else   
19: $y _ { r } \gets z ;$ push REVISE(r)   
20: end if   
21: end if   
22: end while   
23: return majority answer $\hat { a } _ { \pi } ( x )$ ▷ Section 2.1   
SPAWN(): if a complete sample fits in B, add a root r<sup>′</sup> to   
R and push SAMPLE(r<sup>′</sup>).

## B Decoding, Formatting, and Answer Extraction

Termination. The token ε is the end-of-sequence token of the chat template of each model. Because ε is part of the sequence, the test z = y also compares the termination decision: a revision that stops earlier or later than y is not a fixed point.

Prompting and precision. All models run in bfloat16 with their own chat templates and with thinking modes disabled.

Sampling. Sampled calls use temperature $\tau { = } 0 . 7$ and nucleus threshold 0.95. Each request draws its random numbers from its own seeded generator, so its samples do not depend on the other requests in a batch. Greedy calls use argmax with the tie-breaking of the backend.

Answer extraction and voting. For math, ϕ returns the content of the last \boxed{}. For LiveBench, the answer line follows the format of each task (Appendix E), and the official LiveBench scripts score the answers. Intermediate revisions do not vote. A tie in the majority vote goes to the root with the lowest index, and a question without a vote counts as incorrect.

## C Compute Accounting and Budgets

Cost of a position. The accounting gives a FLOP cost to each processed position. A position that

![](images/6373ee3f84474f7c3c58dbe67a1f5a4a66ac61e5bc3712675fc71cbb4170d1a9.jpg)  
Figure 6: Accuracy against the FLOP budget B (Qwen2.5-Math-7B-Instruct). Error bars show the standard deviation over seeds.

![](images/5dc7fb61dcbc3bdf41cbe3a2df4929a06aac4d4caa997981238127b89a09aa0e.jpg)

![](images/4ebfefe8ae90dbbff849f1349974a62c0f28830c50e0bd77f60eb7df25a2b670.jpg)

![](images/4b79b2d40fadd7f04cbc3eb0cf448ed3380629b8e1e4d62df3c72f32c7e68ad6.jpg)  
Filled: Qwen2.5-Math-7B Open: -ve other models  
Figure 7: Compute and accuracy over all settings: (a) TFLOPs used, (b) serial passes relative to ARR, and (c) accuracy difference from ARR.

attends to c cached keys costs

$$
\kappa ( c ) = 2 \left. \theta \right. + 4 n _ { \mathrm { l a y e r } } d _ { \mathrm { a t t n } } c .\tag{5}
$$

The first term is the cost of the matrix multiplications, and |θ| is the number of matrix-multiplication parameters. The second term is the cost of attention, where $n _ { \mathrm { l a y e r } }$ is the number of layers and $d _ { \mathrm { a t t n } }$ is the attention width.

Charging rules. The accounting charges every position in $\mathscr { P } _ { \pi } ( x )$ (Section 3.4) before the pass that processes it, and it refunds nothing. If a draft block does not fit in the remaining budget, the block is shortened. A new root is created only if the remaining budget, minus the worst-case cost of the calls in progress, covers a complete sample. For hybrid linear-attention models, the positions that are fed again after a rejected block (Appendix G) are charged as recomputation.

Budgets. We convert a budget B to a number of samples. A sample with a prompt of P tokens and an output of T tokens processes $L = P + T$ positions, and position i attends to i keys. The sum of Equation (5) over these positions gives the cost

of one sample,

$$
C ( L ) \approx 2 \left| \theta \right| L + 2 n _ { \mathrm { l a y e r } } d _ { \mathrm { a t t n } } L ^ { 2 } .\tag{6}
$$

A budget B thus buys approximately $\begin{array} { r l } { K } & { { } = } \end{array}$ $B / C ( L )$ samples, and at least $K _ { \mathrm { m i n } } = B / C ( P +$ $T _ { \mathrm { m a x } } )$ complete samples. For Qwen2.5-Math-7B-Instruct, $| \theta | \approx 7 . 1 \mathrm { B } , n _ { \mathrm { l a y e r } } { = } 2 8$ , and $d _ { \mathrm { a t t n } } = 3 , 5 8 4$ With $P \approx 2 0 0$ and $T _ { \mathrm { m a x } } { = } 1 { , } 5 3 6$ , Equation (6) gives $C \approx 2 5 ~ \mathrm { T F L O P s }$ , and the attention term is less than 3% of this cost. Thus, $B \in$ {200, 400, 800} TFLOPs gives $K _ { \mathrm { m i n } } \approx 8$ , 16, and 32. Most solutions are shorter than $T _ { \mathrm { m a x } } { \mathrm { i } }$ on MATH500 at $B { = } 8 0 0 .$ , Majority Voting collects approximately 97 samples, i.e., approximately 8 TFLOPs for each sample (Table 5). GSM8K solutions are shorter again, so GSM8K uses one quarter of these budgets, $B \in \{ 5 0 , 1 0 0 , 2 0 0 \}$ TFLOPs. For a model m with $T _ { \mathrm { m a x } } { = } 4 { , } 0 9 6$ , we scale the budget so that $K _ { \mathrm { m i n } }$ stays the same:

$$
B _ { m } \approx B \cdot \frac { | \theta _ { m } | } { | \theta _ { 7 \mathrm { B } } | } \cdot \frac { 4 { , } 0 9 6 } { 1 , 5 3 6 } ,\tag{7}
$$

where $\theta _ { \mathrm { 7 B } }$ are the parameters of Qwen2.5-Math-7B-Instruct. This gives {130, 260, 520} for Qwen3.5-2B, {350, 700, 1400} for Gemma-4, and {650, 1300, 2600} for Qwen3.5-9B. A latencymatched variant limits the serial passes per question to {2,000, 4,000, 8,000}.

## D Compared Methods

Table 1 summarizes the methods of Section 4.1. This section gives only the details that the main text does not give.

Prompt lookup. At each step, $\mathrm { A R R } _ { \mathrm { P L D } }$ matches the last $n \leq 3$ tokens against the prompt and the output, drafts the next 10 tokens, verifies them in one pass, and repeats the lookup after every mismatch; because $y$ is part of $\rho ( x , y )$ , it can realign to y after a change.

Why exact variants differ in accuracy. The exactness of replay (Section 3.2) concerns a single revision, whereas a full run under a shared FLOP budget can diverge for three reasons. (i) Cost. The accounting charges every verified draft position, including rejected ones (Section 3.4), so the same revision costs a different number of FLOPs under each decoder, and the budget ends at a different point. (ii) Scheduling. The decoders finish revisions at different speeds, so the concurrent roots of a question consume the budget in a different order. This changes the reserve for calls in progress, which root receives the last affordable new sample, and which revision the budget truncates; the variants therefore create different numbers of roots (16.2, 15.9, and 13.3 per MATH500 question at B=800, Table 5). (iii) Numerics. One near-tie broken differently by a batched and a serial pass flips a token and changes the rest of the trajectory of that root, including whether it reaches a fixed point. Accuracy differences among the three variants therefore reflect these run-level effects, which are largest on benchmarks with 30-40 questions, not different revision operators.

ConMA. We implement ConMA (Wang et al., 2026) with the hyperparameters of the paper (8 trajectories per round, weights 0.3 and 0.7 for the fullsequence and tail confidence, a tail of 128 tokens, a consensus weight of 0.5, a kernel threshold of 0.75, and an expansion trigger of 0.5) and its verbatim prompts. A trajectory gets a confidence score from the geometric mean of its token probabilities over the full sequence and over its tail; answer groups are ranked by consensus and mean score, and the kernel is the smallest set of top groups with a cumulative probability of at least the kernel threshold. If the kernel is large, ConMA samples new trajectories from a critical-review prompt; otherwise it asks a multiple-choice question over the kernel. It stops when one answer remains or when the budget cannot fund another round, and it shares the FLOP accounting and the answer extractor of the other methods.

Stopping of the other baselines. $\mathrm { \Delta A R R _ { D e p t h } }$ stops when all N roots are retired or truncated and leaves the rest of the budget unused. Self-Refine also revises roots that are already fixed points. Sequential Revision continues its roots round-robin and never retires or creates a root.

## E Prompt Templates

Italic text in braces marks an inserted field, such as the question or the current solution, and bold labels mark the system and user turns. For LiveBench, the answer line uses the format of each task instead of \boxed{}.

System prompt of all solve and revision calls

Please reason step by step, and put your final answer within \boxed{}.

Solve prompt σ(x) (all methods)

{question}

(b) Output tokens per question, B = 800  
![](images/6cc89e7d5dca01ac1575e1b1438ce8df26b8c93766f00ce391dba502f1150a04.jpg)

![](images/01fff2f13306ff2f45d85bdaeeafe12f1fc8c34782391144789b4c819be3e530.jpg)

Figure 8: Cost of revision for Qwen2.5-Math-7B-Instruct. (a) Ratio of the median serial passes of ARR to those of ARR<sub>AR</sub>. Replay gives a larger decrease at larger budgets. (b) Output tokens for each question at B=800.  
![](images/79fdbe118b4d0160a2dc119cc0b1ee2705a1d598e6c9e2a4dee0342a5caa9d4f.jpg)  
ARR (ours) ARR<sub>AR</sub> ARR<sub>PLD</sub> ARR<sub>Depth</sub> Self-Re-ne SETS Sequential Revision Majority Voting

Figure 9: Accuracy (%) of every method on the other models at their largest budget (B=800 for Ministral-8B and Qwen3-8B, 520 for Qwen3.5-2B, 1,400 for Gemma-4-E4B, and 2,600 for Qwen3.5-9B).

## Revision prompt ρ(x, y) (ARR, ARR<sub>AR</sub>, ARR<sub>PLD</sub>, ARR<sub>Depth</sub>, Self-Refine)

Check the proposed solution against the question. Return a complete corrected solution. Correct any reasoning or calculation errors, including their downstream consequences. Preserve wording that does not need correction. If no correction is needed, reproduce the proposed solution exactly. Output only the solution.

Question: {question}

Proposed solution: {solution}

## Critique prompt ρ<sub>crit</sub>(x, y) (SETS, step 1)

System: You are a careful mathematical critic. You will be given a PROBLEM and a SOLUTION TRACE. Your job: identify any errors in the trace - logical mistakes, wrong calculations, incorrect assumptions, or faulty reasoning. If the trace is correct, say so explicitly. Be concise and specific. Do NOT solve the problem yourself.

User: Review the following SOLUTION TRACE for the PROBLEM below. Identify any errors: wrong calculations, incorrect assumptions, logical mistakes, or missing steps. If no errors exist, state: ‘The solution appears correct.’ Be specific - point to the exact step where an error occurs. Do NOT re-solve the problem.

PROBLEM {question}

SOLUTION TRACE {solution}

Extracted answer: {answer}

CRITIQUE (list errors specifically, or confirm correct-

ness):

## Correction prompt ρ<sub>cor</sub>(x, y, ·) (SETS, step 2)

System: You are a careful mathematical editor. You will be given a PROBLEM, a SOLUTION TRACE, and a CRITIQUE. Your job: produce a corrected solution that fixes all errors identified in the critique while keeping the correct parts of the original trace. Use the same numbered-step format. End your response with exactly one line: Final answer: \boxed{<answer>}

User: You are given a PROBLEM, a SOLUTION TRACE, and a CRITIQUE of that trace. Produce a corrected solution: fix every error identified in the critique, keep the numbered-step format, preserve correct parts of the original trace, do NOT introduce new errors, end with exactly one line: Final answer: \boxed{<answer>}

PROBLEM {question}

ORIGINAL SOLUTION TRACE {solution}

CRITIQUE {critique}

CORRECTED SOLUTION (numbered steps, ending with Final answer: \boxed{<answer>}):

## Continuation prompt ρ<sub>cont</sub>(x, y, k) (Sequential Revision)

System: You are a precise mathematical reasoner. Solve problems using clearly numbered steps. Show every calculation inside each step. End your response with exactly one line: Final answer: \boxed{<answer>}

Table 1: Compared methods. The rows from $\mathrm { A R R } _ { \mathrm { A R } }$ to SETS change one component of ARR each.
<table><tr><td>Method</td><td>Update of a root</td><td>Decoding of the update</td><td>Explicit verifier</td><td>A root stops when</td><td>Slot of a stopped root</td></tr><tr><td>ARR</td><td> $F _ { x } ( y )$ </td><td>replay (W=32)</td><td>no</td><td> $F _ { x } ( y ) = y \mathrm { o r } F _ { x } ( y ) \notin \mathcal { Y }$ </td><td>new root</td></tr><tr><td> $\mathrm { A R R } _ { \mathrm { A R } }$ </td><td> $F _ { x } ( y )$ </td><td>autoregressive greedy</td><td>no</td><td> $\mathrm { s a m e \ a s \ A R R }$ </td><td>new root</td></tr><tr><td> $\mathbf { A R R } _ { \mathbf { P L D } }$ </td><td> $F _ { x } ( y )$ </td><td>prompt lookup</td><td>no</td><td> ${ \mathrm { s a m e ~ a s ~ A R R } }$ </td><td>new root</td></tr><tr><td> $\mathrm { \bf A R R _ { \mathrm { D e p t h } } }$ </td><td> $F _ { x } ( y )$ </td><td>replay</td><td>no</td><td> ${ \mathrm { s a m e ~ a s ~ A R R } }$ </td><td>not reused</td></tr><tr><td>Self-Refine</td><td> $F _ { x } ( y )$ </td><td>autoregressive greedy</td><td>no</td><td>after D=4 revisions</td><td>not reused</td></tr><tr><td>SETS</td><td> $F _ { x } ^ { \mathrm { v c } } ( { \dot { y } } )$ </td><td>autoregressive greedy</td><td>yes</td><td> $F _ { x } ^ { \mathrm { v c } } ( y ) = y \mathrm { o r t r u n c a t e d }$ </td><td>new root</td></tr><tr><td>Sequential Revision sample from</td><td> $\rho _ { \mathrm { c o n t } } ( x , y , k ) ( \tau { = } 0 . 7 )$ </td><td>autoregressive sampling</td><td>no</td><td>budget ends</td><td></td></tr><tr><td>Majority Voting</td><td>none (new sample)</td><td>autoregressive sampling</td><td>no</td><td> $\mathrm { a t \ o n c e }$ </td><td>new sample</td></tr><tr><td>ConMA</td><td>none (groups of samples)</td><td>autoregressive sampling</td><td></td><td>confidence score one answer remains</td><td></td></tr></table>

depth: revise the N initial roots breadth: new roots (ARR)

![](images/f3857d816e14d7c314cdac98ae1fa9d1619f44c0ed844b906728f33f9ad063ac.jpg)

(b) Per budget  
![](images/0a28675f22ab94813af0ecf7aa6afbd7fcab000023a75d94039cfc9584dfc13e.jpg)

(c) Other models  
![](images/7c1f177922dddfbca3ad746afa79cf8c9e3ff90ac0d1e6cdd1e895db1da0bf03.jpg)  
Figure 10: Accuracy gains from revision depth and replacement breadth across benchmarks, budgets, and models.

User: You are at refinement depth {k} of {D}. Below is the ORIGINAL PROBLEM and the CORRECTED REASONING from the previous depth. Your task: 1. Review the corrected reasoning carefully. 2. Continue from Step {n}: - extend or refine the solution, verifying each claim. 3. If the prior reasoning already reaches a correct answer with no errors, confirm it concisely. Do NOT change the answer unless you can identify a specific concrete error. 4. End with exactly one line: Final answer: \boxed{<answer>}

PROBLEM {question}

CORRECTED REASONING FROM DEPTH {k-1} {solution}

## F A Worked Example of Replay and Replacement

We trace one question from the logs of Qwen2.5- Math-7B-Instruct on MATH500 (problem 358, $B { = } 8 0 0 \ \mathrm { T F L O P s } )$

## Problem (MATH500, no. 358)

Determine if the graph of the equation below is a parabola, circle, ellipse, hyperbola, point, line, two lines, or empty.

$$
\begin{array} { r } { \left( { \frac { x } { 2 } } - 3 \right) ^ { 2 } + y ^ { 2 } = 1 0 } \end{array}
$$

The correct answer is ellipse: the equation equals $( x - 6 ) ^ { 2 } + 4 y ^ { 2 } = 4 0$ , in which the coefficients of $( x - 6 ) ^ { 2 }$ and $y ^ { 2 }$ differ. Many samples answer circle, because they read the left side as a circle in the substituted variable $x / 2$

Outcome. Table 3 compares the three methods on this question. Majority Voting and SETS return the frequent incorrect answer. ARR returns the correct answer with 40% fewer serial passes than Majority Voting.

Trajectory of one root. Table 4 follows root 0 of ARR through one sample and four revisions. Revisions 1 and 2 rewrite most of the solution but keep the answer circle. Revision 3 changes the answer to ellipse, and revision 4 reproduces revision 3 exactly.

Revision 3: replay keeps the derivation and rewrites the end. The draft of revision 3 has 667 tokens. Its first 602 tokens derive $( x - 6 ) ^ { 2 } + 4 y ^ { 2 } =$ 40 and state that this equation is an ellipse, but its last 65 tokens return to the circle form and box circle. Replay accepts the 602 tokens in 19 blocks, discards 6 positions of the failed block, and decodes a new 17-token end. The call takes 37 serial passes, whereas autoregressive decoding takes 621 passes for the same output. The three boxes below show the parts of this revision in order. The logs store the token counts and the answer of each call, but not the full text of intermediate drafts, so the second box describes the discarded end.

![](images/ed3df5069679d526b04b0290336b7c9bada4743a484fd91cc603c34c1c81ba38.jpg)

![](images/02ec0397dc7f565f3a725bb916d5813d1b6972382ee87eb2c1d5efc6ca090ed8.jpg)

![](images/b377a4306fa19a72fe00c24f78d406c2031f6e513ec31e1eddaaf20f04aa4069.jpg)  
Figure 11: Depth-to-breadth reallocation under fixed-point replacement across budgets and models.

Table 2: Stopping rules with the replacement of ARR for Qwen2.5-Math-7B-Instruct. Answer: retire a root when its extracted answer does not change; Once: retire a root after one revision. ∆: Answer − ARR with a 95% paired CI. Roots: roots per question of ARR / Answer. Passes: serial passes of Answer / ARR. All columns use the seeds of the stopping-rule runs, so ∆ is paired. On MATH500, OlympiadBench, and GSM8K, these are a subset of the main-run seeds, so ARR and Majority Voting differ from the all-seed means of Figure 6 (e.g., 84.0% vs. 84.6% on MATH500 at B=800). In all tables, blue marks the best value and red the second-best value.
<table><tr><td>Benchmark</td><td>B</td><td>ARR</td><td>Answer</td><td>Once</td><td>Majority Voting</td><td>∆</td><td></td><td>Roots</td><td>Passes</td></tr><tr><td rowspan="3">MATH500</td><td>200</td><td>81.6</td><td>81.6</td><td>81.8</td><td>83.2</td><td>+0.0</td><td>[−1.1, +1.2]</td><td>5.8 / 9.3</td><td>1.20</td></tr><tr><td>400</td><td>83.8</td><td>83.0</td><td>83.4</td><td>83.2</td><td>-0.8</td><td>-2.4, +0.8]</td><td>9.5 / 19.1</td><td>1.19</td></tr><tr><td>800</td><td>84.0</td><td>84.6</td><td>84.4</td><td>84.0</td><td>+0.6</td><td>[−0.8, +1.8]</td><td>16.1 / 38.5</td><td>1.16</td></tr><tr><td rowspan="3">OlympiadBench</td><td>200</td><td>44.0</td><td>45.0</td><td>44.0</td><td>46.4</td><td>+1.0</td><td>[−0.7, +2.7]</td><td>5.2/ 7.0</td><td>1.14</td></tr><tr><td>400</td><td>45.7</td><td>46.9</td><td>47.6</td><td>48.1</td><td>+1.2</td><td>[−1.2, +3.6]</td><td>8.2/ 14.0</td><td>1.19</td></tr><tr><td>800</td><td>47.6</td><td>49.9</td><td>48.2</td><td>49.9</td><td>+2.2</td><td>[+0.3, +4.2]</td><td>13.3 / 27.8</td><td>1.18</td></tr><tr><td rowspan="3">AMC23</td><td>200</td><td>68.0</td><td>68.0</td><td>67.5</td><td>69.5</td><td>+0.0</td><td>[−5.5, +5.5]</td><td>5.0/ 7.0</td><td>1.15</td></tr><tr><td>400</td><td>72.5</td><td>68.0</td><td>72.5</td><td>73.0</td><td>-4.5</td><td>-9.5, −0.5]</td><td>7.7 / 14.2</td><td>1.18</td></tr><tr><td>800</td><td>72.0</td><td>73.0</td><td>71.5</td><td>71.0</td><td>+1.0</td><td>[−5.0, +7.5]</td><td>12.8 / 28.3</td><td>1.23</td></tr><tr><td rowspan="3">AIME24</td><td>200</td><td>12.7</td><td>14.0</td><td>16.7</td><td>12.7</td><td></td><td>+1.3 [−1.3, +4.7]</td><td>5.5 / 6.6</td><td>1.06</td></tr><tr><td>400</td><td>18.0</td><td>14.0</td><td>17.3</td><td>15.3</td><td></td><td>-4.0 [−10.0, +0.7]</td><td>8.1 / 12.9</td><td>1.06</td></tr><tr><td>800</td><td>18.7</td><td>17.3</td><td>17.3</td><td>18.7</td><td></td><td>−1.3 [−4.7, +2.0]</td><td>13.1 / 24.9</td><td>1.09</td></tr><tr><td rowspan="3">AIME25</td><td>200</td><td>14.7</td><td>17.3</td><td>18.0</td><td>19.3</td><td></td><td>+2.7 [−1.3, +6.7]</td><td>5.4 / 6.0</td><td>1.08</td></tr><tr><td>400</td><td>15.3</td><td>22.0</td><td>20.0</td><td>19.3</td><td></td><td>+6.7 [+2.0, +11.3]</td><td>7.7 / 11.5</td><td>1.13</td></tr><tr><td>800</td><td>18.0</td><td>22.0</td><td>21.3</td><td>25.3</td><td></td><td>+4.0 [−2.0, +10.7]</td><td>12.5 / 22.6</td><td>1.14</td></tr><tr><td rowspan="3">GSM8K</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>50</td><td>94.2</td><td>94.3</td><td>94.3</td><td>94.3</td><td>+0.1</td><td>[−0.5, +0.7]</td><td>4.0 / 4.0</td><td>0.98</td></tr><tr><td>100 200</td><td>94.0 94.5</td><td>94.3 95.3</td><td>94.5 95.5</td><td>94.6 95.2</td><td>+0.3 +0.8</td><td>[−0.4, +1.0] [+0.1, +1.5]</td><td>4.2 / 6.3 5.9 / 13.8</td><td>1.37 1.29</td></tr></table>

<table><tr><td>Method</td><td>Answer</td><td>Votes</td><td>Serial passes</td></tr><tr><td>Majority Voting</td><td>circle X</td><td>34 vs. 27 (61 samples)</td><td>13,049</td></tr><tr><td>SETS ARR</td><td>circle X ellipse √</td><td>13 vs.5</td><td>7,877</td></tr></table>

Table 3: Outcome on MATH500 problem 358 at B=800 TFLOPs. Votes give the winning answer against the runner-up. Serial passes are counted on the critical path.

## Accepted prefix of revision 3 (602 tokens, 19 blocks, verified in parallel)

. . . If we compare $( x - 6 ) ^ { 2 } + 4 y ^ { 2 } = 4 0 \mathrm { t o } ( x - h ) _ { \circ } ^ { 2 } + ( y -$ $\boldsymbol { k } ) ^ { 2 } = \boldsymbol { r } ^ { 2 }$ , we see that the coefficients of $( { \dot { x } } - 6 ) ^ { 2 } { \mathrm { ~ a n d ~ } } y ^ { 2 }$ are not the same. Therefore, the equation $( x - 6 ) ^ { 2 } + y ^ { 2 } =$

<table><tr><td colspan="6">Sample Rev. 1 Rev. 2 Rev.3 Rev. 4</td></tr><tr><td>Draft</td><td></td><td>428</td><td>76</td><td>667</td><td>620</td></tr><tr><td>Accepted</td><td></td><td>0</td><td>0</td><td>602</td><td>620</td></tr><tr><td>Discarded</td><td></td><td>0</td><td>0</td><td>6</td><td>0</td></tr><tr><td>Decoded</td><td>428</td><td>75</td><td>666</td><td>17</td><td>0</td></tr><tr><td>Passes</td><td>428</td><td>76</td><td>667</td><td>37</td><td>21</td></tr><tr><td>Answer</td><td>circle</td><td>circle</td><td>circle</td><td>ellipse ellipse (FP)</td><td></td></tr></table>

Table 4: Trajectory of root 0. Accepted and discarded denote replay-verified draft tokens; decoded denotes greedy single-token steps, and FP denotes a fixed point.

<table><tr><td></td></tr><tr><td>10 does not represent a circle</td></tr><tr><td></td></tr></table>

![](images/848c49cb61b4e3bc59ef59ea37bfefba128809b43d9cacc95c47ce508a406747.jpg)  
Budget B (TFLOPs per question, log scale)  
Figure 12: Accuracy up to B=1,600.  
ARR (ours) Majority Voting

![](images/132a5a8ad418ababb3ff2dfe3d7f957326eee6631d3a99eb75c22f6e59b84eda.jpg)

![](images/410e197fde639d680c0133e647b6681213d837886ac63b95e5796d34e4c497f6.jpg)  
Accuracy ! accuracy at ==0:7 (points)  
Figure 13: Change of accuracy from the default τ=0.7 at (a) τ=0.5 and (b) τ=1.0, for each benchmark and budget B.

The correct form of the equation is $( x - 6 ) ^ { 2 } + 4 y ^ { 2 } = 4 0 ,$ which represents an ellipse.

Discarded end of the draft (65 tokens)

Returns to the circle form and ends with the boxed answer circle.

New end of revision 3 (17 tokens, greedy decoding)

Therefore, the correct answer is: ellipse

Revision 4: a fixed point. Revision 4 reproduces all 620 tokens, including the terminating ε, in 21 serial passes. Thus, root 0 is a fixed point: ARR keeps ellipse as its vote and gives the slot to a new root.

## Key Observations

• Majority Voting counts each sample once, and thus a frequent shared error wins the vote.

• ARR revises each sample until the revision is stable, and thus a root with the correct result in its derivation can correct its conclusion.

• Revisions 1 and 2 do not change the answer, so a fixed schedule of two revisions keeps the incorrect answer.

• ARR continues because the text still changes, and it stops as soon as the text stops changing.

## G Implementation Details

Decoder. All methods run in one decoder on HuggingFace Transformers, with continuous batching, a static key-value cache for each slot, and access to the logits. Replay blocks, prompt-lookup blocks, and single-token steps run in separate batched passes. The shared job queue (Appendix A) is first-in-first-out, so a new root never overtakes an unfinished root. The key-value cache of an accepted prefix is reused only within its call, so each revision prefills $\rho ( x , y )$ again.

Architectures. The decoder detects the layer types of a model with one forward pass. For hybrid linear-attention models (Qwen3.5), it keeps a recurrent state for each slot. It verifies a draft block on a copy of this state and feeds the accepted prefix again after a rejection. Each run uses one A100 or

(a) Accuracy  
![](images/168fea95c6d02c39f171e26f1618e5428dc1ff5422e7a762b7f8f7fc87165b37.jpg)

![](images/049c8fbce8b1e02b05dfa5a38d2a6dedd541a92710e791bdcf64c9abff35f1e2.jpg)  
MATH500, B=200 MATH500, B=400 AMC23, B=200 AMC23, B=400 AMC23, B=800  
Figure 14: (a) Accuracy and (b) serial passes relative to the default W=32, for each benchmark and budget.

![](images/1307c3b1885313420529c40db48d12a60022f0f974e171e7550389626bdab168.jpg)

![](images/098498d40fafc68f9e32ea284ce50713a3c905743f26dc5cb53c9b2006eafecb.jpg)  
" to N=4 (points)

![](images/0eb1de83e53930450397c929b9673e0c0c7d2d7ca55f247516d4c7b3de054a59.jpg)  
Figure 15: Change of accuracy from the default N=4 at (a) N=2 and (b) N=8, for each benchmark and budget. (c) Accuracy relative to ARR (N=8) on AMC23 and AIME24; dots denote individual settings, large markers denote means.

## H100 GPU.

Output equivalence of replay and greedy decoding. Let $g = F _ { x } ( y )$ be the output of autoregressive greedy decoding on $u = \rho ( x , y )$ . Replay and greedy decoding share the prefill of u, so their first output token is the same token $g _ { 1 }$ . While the committed output is on the draft, $z = y _ { 1 : t } = g _ { 1 : t }$ , replay runs one causal pass on $( z _ { t } , y _ { t + 1 } , \ldots , y _ { t + m } )$ . The causal mask restricts the logits at block position $j$ to u and $y _ { 1 : t + j - 1 : }$ , so $\hat { z } _ { t + 1 }$ is the greedy token after $g _ { 1 : t }$ , which is $g _ { t + 1 }$ . If this token equals the draft token $y _ { t + 1 }$ , the conditioning prefix of the next block position is again a prefix of $g _ { \colon }$ , and the same argument holds at every later position of the block. Thus, the accepted span $y _ { t + 1 : t + \lambda }$ and the appended token $\hat { z } _ { t + \lambda + 1 }$ are exactly the next $\lambda { + 1 }$ tokens of $g .$ The positions after the first disagreement condition on a draft token that $g$ does not contain, so the decoder discards their key-value entries before the next pass, and no rejected position affects a later prediction. After the output leaves the draft, replay continues with single-token greedy steps from a prefix of $g$ and thus produces the rest of $g .$ . Both decoders stop at the same $\varepsilon \_ { \mathrm { { o r } } }$ at the same $T _ { \mathrm { m a x } } .$ and an accepted ε inside a block ends the call at once. For $W { = } 1$ , each block has $m { = } 0$ draft tokens, and replay reduces to greedy decoding. A fixed point is also stable under further revision: because $\rho$ contains no round index or history, $F _ { x }$ maps each $y$ to one output, so $F _ { x } ( y ) = y$ gives $F _ { x } ^ { k } ( y ) = F _ { x } \big ( F _ { x } ^ { k - 1 } ( y ) \big ) \ : = \ : y$ for every $k \geq 1$ This equivalence holds only if a batched pass and a single-token pass select the same argmax; the verification tests below check this condition on our

Table 5: Mechanism on MATH500 at $B { = } 8 0 0$ TFLOPs with Qwen2.5-Math-7B-Instruct. TF: TFLOPs per question; Roots: roots per question; FP: fixed-point rate; $\mathrm { \bf W { \to } C , C { \to } W \colon }$ answer changes (Appendix G).
<table><tr><td>Method</td><td>TF</td><td>Roots</td><td>FP</td><td> $\mathrm { { \cal W } } { \to } \mathrm { { \cal C } }$ </td><td> $\mathrm { C } {  } \mathrm { W }$ </td></tr><tr><td>ARR</td><td>789</td><td>16.2</td><td>.23</td><td>663</td><td>357</td></tr><tr><td> $\mathrm { A R R } _ { \mathrm { A R } }$ </td><td>788</td><td>15.9</td><td>.22</td><td>666</td><td>374</td></tr><tr><td> $\mathrm { A R R } _ { \mathrm { P L D } }$ </td><td>797</td><td>13.3</td><td>.30</td><td>463</td><td>230</td></tr><tr><td> $\mathrm { \Delta A R R _ { D e p t h } }$ </td><td>381</td><td>4.0</td><td>.08</td><td>300</td><td>234</td></tr><tr><td>Self-Refine</td><td>268</td><td>4.0</td><td>.53</td><td>173</td><td>96</td></tr><tr><td>SETS</td><td>793</td><td>14.8</td><td>.69</td><td>244</td><td>258</td></tr><tr><td>Sequential Revision</td><td>793</td><td>4.0</td><td></td><td>218</td><td>278</td></tr><tr><td>Majority Voting</td><td>781</td><td>96.6</td><td></td><td>0</td><td>0</td></tr><tr><td>ConMA</td><td>185</td><td>10.0</td><td>一</td><td>-</td><td>1</td></tr></table>

backend.

Mechanism statistics. Table 5 reports five statistics per method, computed over all questions and seeds of a cell. TF is the mean number of TFLOPs that the FLOP accounting of Section 3.4 charges per question, including prefills, discarded draft positions, and truncated calls. Roots is the mean number of roots |R| created per question: the $N { = } 4$ initial roots plus every new root $r ^ { \prime } { \mathrm { ; } }$ for Majority Voting it is the number of samples, and methods without replacement stay at $N .$ FP is the fixed-point rate, the fraction of complete revisions $( z \in \mathcal { V } )$ that reproduce their input exactly, $z = y ;$ for SETS the test is $F _ { x } ^ { \mathrm { v c } } ( y ) = y .$ , and it is undefined for Sequential Revision, whose update is sampled, and for Majority Voting, which never revises. $W {  } C$ and $C {  } W$ count complete updates whose extracted answer $\phi$ changes from incorrect to correct and from correct to incorrect with respect to $a ^ { \star } ( x )$ , summed over all questions and seeds; these labels are used only for evaluation and never by a method. Updates that keep the correctness of the answer are not counted, so $\mathrm { \bf W { \to } C { \mathrm { - } C } { \to } W }$ is the net number of answers that revision repairs. We also record the output tokens per question and the accepted-prefix fraction $\ell / | y |$ of complete revisions.

## H Additional Results

Equal-FLOP view. Figure 6 gives accuracy against the FLOP budget, and Figure 7 gives the compute that each method uses. Six methods spend 93-100% of B on Qwen2.5-Math-7B-Instruct, so their FLOPs are equal by design. $\Delta \mathrm { R R } _ { \mathrm { D e p t h } }$ stops when its roots get to fixed points, and Self-Refine stops after $D { = } 4$ revisions; neither reuses the slots. They spend only 50-73% of B and lose 1.9-4.4 points. ConMA stops when one answer remains and spends 57% of B.

Table 6: Accuracy (%) of the stopping rules of Table 2 for Qwen3-8B.
<table><tr><td>Benchmark</td><td>B</td><td>ARR</td><td>Answer</td><td>Once</td></tr><tr><td rowspan="3">AMC23</td><td>200</td><td>62.5</td><td>62.5</td><td>62.5</td></tr><tr><td>400</td><td>65.0</td><td>72.5</td><td>65.0</td></tr><tr><td>800</td><td>75.0</td><td>72.5</td><td>77.5</td></tr><tr><td rowspan="3">AIME24</td><td>200</td><td>26.7</td><td>23.3</td><td>20.0</td></tr><tr><td>400</td><td>26.7</td><td>23.3</td><td>23.3</td></tr><tr><td>800</td><td>33.3</td><td>23.3</td><td>26.7</td></tr><tr><td rowspan="3">AIME25</td><td>200</td><td>16.7</td><td>16.7</td><td>16.7</td></tr><tr><td>400</td><td>23.3</td><td>23.3</td><td>26.7</td></tr><tr><td>800</td><td>26.7</td><td>20.0</td><td>20.0</td></tr></table>

Adaptive allocation. ARR does not stop at a fixed point. It gives the remaining budget to a new root and thus spends 98% of B. On Qwen2.5-Math-7B-Instruct, ARR is the most accurate method that revises, and Majority Voting is 1.3 points better on average over all budgets. On the five other models at the largest budget, SETS and Majority Voting are 1.7 and 1.5 points better on average (Figure 9, Section 4.5). On the primary model and on the other models, Majority Voting needs 2.0 and 1.6 times the serial passes of ARR, and $\mathrm { A R R } _ { \mathrm { A R } }$ needs 1.2 and 1.1 times (Figure 7(b)). ARR stays within 0.3 points of $\mathrm { A R R } _ { \mathrm { A R } }$ . Thus, ARR gives more accuracy for each serial pass at the FLOP cost of majority voting.

Detailed values for Section 4.2. On MATH500 at B=800, ARR and Majority Voting both get 84.6% (paired 95% CI [−1.2, +1.3]). ARR writes 18.2k output tokens for each question and collects 15.0 votes. Majority Voting writes 46.7k tokens and collects 93.3 votes. ConMA uses 23% of B (Table 5) and 0.12 of the serial passes of ARR; averaged over all settings, it uses 57% of B and 0.47 of the serial passes.

Wall-clock latency. Table 7 gives the latency on a subset of MATH500 problems at $B { = } 4 0 0$ , so its accuracies are not comparable to Figure 6. One A100 GPU runs one question at a time, and no other job uses the GPU. The median latency for each question is 78.5 s for ARR, 83.3 s for $\mathrm { A R R } _ { \mathrm { A R } }$ $4 2 . 5 \mathrm { s }$ for $\mathrm { A R R } _ { \mathrm { P L D } }$ , 176.5 s for Majority Voting, and 18.4 s for ConMA. All methods get 90.0%, except ConMA, which gets 87.5% and uses only 23% of B. A replay pass processes up to W positions. Thus, ARR is only 6% faster than $\mathrm { A R R } _ { \mathrm { A R } }$ in wall-clock time, although it needs 30% fewer serial passes. $\mathrm { A R R } _ { \mathrm { P L D } }$ needs the fewest serial passes and is the fastest method that revises.

(a) MATH500  
![](images/aa36331701529a6e611af8de514dd04d9ccda784f238b62e15cb3cb763bc6b1c.jpg)

(b) OlympiadBench  
![](images/772ba7283836e58b6e1e372119315a3c4c37d834231100b986a23e566784cdd3.jpg)  
(e) AIME25

(c) AMC23  
![](images/e22e3c9a654930d31b8eed4ad9bd8394e3a91d6471fe0f7449e31336c97bcb35.jpg)  
(f) GSM8K

(d) AIME24  
![](images/73174131fceace9366592bff6bd5bdd9b0891f19a12b7a503d662be07b723d1d.jpg)

![](images/d45d49f55ba07e77d58650803cee9874af27bed7ffe91d8651b5988a749b7d30.jpg)

(g) LiveBench (Qwen2.5-7B)  
![](images/4e9cbcfe844c563fd5bf94b625d6037967b0636595c4cadb1c252ec8dda7fd8c.jpg)  
Budget B (TFLOPs per question, log scale)

![](images/0ca9dc28fd0ce6e02923db3a59686beb20c8698baf1cfb947a11f255006edf95.jpg)  
ARR, revision prompt ; ARR, paraphrase 1 ARR, paraphrase 2

Figure 16: Accuracy of ARR with the revision prompt ρ and two paraphrases; (g) uses Qwen2.5-7B-Instruct. ARR (ours) ARR<sub>AR</sub> ARR<sub>PLD</sub> ARR<sub>Depth</sub> Self-Re-ne SETS Majority Voting  
(a) MATH500  
![](images/6adbf26afade436636296127d9563fd8f19ae821a6ee5444cd5ce0d7ff34b162.jpg)

(b) AMC23  
![](images/96b704199d5b83a2c2d04faa07c25d8021adfde1d4cf546c5b70b21086cab122.jpg)

(c) AIME24  
![](images/7e35423e972493f9e95638573d585a803eaf349b6d27a2580192a441b61d7027.jpg)  
Budget of serial passes per question (log scale)

(d) AIME25  
![](images/98e6a69744b71e6cf53f5deda963794d5a336085150692f6a0c06da15c90da8f.jpg)  
Figure 17: Accuracy against a budget of serial passes per question. Sequential Revision is omitted for scale.

Table 7: Wall-clock latency on one A100 GPU (Qwen2.5-Math-7B-Instruct, a subset of MATH500 problems, B=400, one question at a time). Latency and serial passes are medians for each question; accuracy is on this subset.
<table><tr><td>Method</td><td>Latency (s)</td><td>Passes</td><td>TFLOPs</td><td>Acc.</td></tr><tr><td>ARR</td><td>78.5</td><td>2,748</td><td>398</td><td>90.0</td></tr><tr><td> $\mathrm { A R R } _ { \mathrm { A R } }$ </td><td>83.3</td><td>3,926</td><td>398</td><td>90.0</td></tr><tr><td>ARRPLD</td><td>42.5</td><td>1,593</td><td>399</td><td>90.0</td></tr><tr><td>Majority Voting</td><td>176.5</td><td>6,878</td><td>381</td><td>90.0</td></tr><tr><td>ConMA</td><td>18.4</td><td>644</td><td>93</td><td>87.5</td></tr></table>

Detailed values for Section 4.4. The roots of ARR increase with the budget: 5.8, 9.5, and 16.2 for each MATH500 question. $\Delta \mathrm { R R } _ { \mathrm { D e p t h } }$ uses 381 of 800 TFLOPs on MATH500, and Self-Refine uses 268 TFLOPs for 81.7% (Figure 11). ARR is better than $\mathrm { \Delta A R R _ { D e p t h } }$ by 3.1 points [2.0, 4.3] on MATH500 and by 4.0 points [2.6, 5.2] on OlympiadBench. The depth gain stays at 1.0-2.0 points at all budgets (Figure 10). SETS is 1.5 points [0.3, 2.7] less accurate than ARR on MATH500, and 69% of its updates are fixed points. Sequential Revision decreases from 79.3% to 59.0% on MATH500 as the budget increases, and it is 25.6 points [22.5, 28.5] below ARR at B=800. $\mathrm { A R R } _ { \mathrm { P L D } }$ uses 0.53-0.58 of the serial passes of ARR at B=800.

Detailed values for Section 4.5. Across mathematical reasoning settings for other models, ARR achieves mean gains of 6.4 and 6.3 points over $\Delta \mathrm { R R } _ { \mathrm { D e p t h } }$ and Self-Refine, respectively. Majority Voting outperforms ARR in more settings than the reverse. Excluding Qwen3.5-2B and the two Gemma-4 settings with repaired formatting, SETS and ARR perform within 0.1 points of each other.

![](images/12dbca7e1845b321b2909eddbcb5b29c9ad0e69532f3b6ade7fec58a64be5fba.jpg)

![](images/1a7f2816808befa4b0cb0868ad4cb29ce564fa18dde8ee0371c56eaf3f521ca7.jpg)  
Figure 18: First seed against the mean over all seeds at the largest budget. (a) Accuracy of ARR and Majority Voting. (b) Accuracy differences between methods.

Ablations. We change one setting of ARR at a time with Qwen2.5-Math-7B-Instruct, and we compare each run with the matched main run. We fixed the default settings $( N { = } 4 , \tau { = } 0 . 7 , W { = } 3 2 )$ before these runs, and all other experiments use them. Table 2 gives the stopping rules of Figure 5 at all budgets, and Table 6 gives the same rules for Qwen3-8B. Figures 13, 14, 12, 15, 16, and 17 give the sampling temperature, the block size W, the budget $_ { B = 1 , 6 0 0 }$ , the width N, the paraphrases of $\rho ,$ and the latency-matched budgets. Each ablation shows a benchmark and budget only if all compared settings and methods finished there. A result in a figure does not occur again in a table.

Detailed values of the ablations. Across settings with complete results for $N \in \{ 2 , 4 , 8 \}$ , ARR with N=8 gains 1.9 points over N=4 while using 0.64 times as many serial passes. At N=8, ARR is within 0.5 points of Majority Voting while using 0.59 times as many serial passes. A block size $W \in [ 8 , 6 4 ]$ changes the serial passes by at most 5%. At B=1,600, ARR achieves 39.2%, Majority Voting 38.9%, and $\Delta \mathrm { R R } _ { \mathrm { D e p t h } }$ 36.4% accuracy on average across AMC23, AIME24, and AIME25. The two paraphrases of $\rho$ change accuracy by at most 1.2 points on MATH500, OlympiadBench, and GSM8K.

One seed and several seeds. Figure 18 compares first-seed accuracy with the mean across all seeds. First-seed accuracy differs from the allseed mean by at most 5.5 points. The sign of $\Delta \mathrm { R R } { \mathrm { - } } \mathrm { A R R } _ { \mathrm { D e p t h } }$ agrees between the first seed and the all-seed mean in 94% of settings, compared with 71% for ARR−Majority Voting. Among the disagreements with Majority Voting, 80% involve an exact tie in one of the comparisons, and the all-seed mean differences remain within 2.2 points. Thus, single-seed comparisons with Majority Voting are less reliable, while ARR’s gains over $\Delta \mathrm { R R } _ { \mathrm { D e p t h } }$ are more stable.

## I Extended Related Work

## I.1 More draft-and-verify methods.

Medusa (Cai et al., 2024) drafts with extra heads, lookahead decoding (Fu et al., 2024) drafts with Jacobi iterations, and EAGLE (Li et al., 2025) drafts with an extra head on the hidden features.

## I.2 Sequential scaling and efficient reasoning.

Long reasoning traces frequently spend tokens on redundant reflection (Chen et al., 2025b), including those of reasoning models trained with reinforcement learning (DeepSeek-AI, 2025). Budget forcing extends or ends a reasoning trace to control its length (Muennighoff et al., 2025), and Sui et al. (2025) survey such efficient reasoning methods. ARLCP (Yu et al., 2026) trains models to reflect less with length-sensitive rewards. ARR keeps the weights frozen and removes a different redundancy: the repeated generation of unchanged text across revision rounds.

## I.3 Fixed points in sequence models.

Deep equilibrium models compute fixed points of layers (Bai et al., 2019), and Jacobi decoding iterates a parallel greedy update to its fixed point (Santilli et al., 2023; Kou et al., 2024). In these methods, the fixed point is the output. In ARR, the fixed point is a stop signal for one solution in a population of solutions.

Table 8: Signals that end or redirect refinement in the closest methods. The signal of ARR is an exact property of a deterministic operator, not a verdict or a score. ARR gets the signal with parallel draft verification, which comes from prior work (Sun et al., 2021; Leviathan et al., 2023; Saxena, 2023).
<table><tr><td>Method</td><td>Training- free</td><td>No step structure</td><td>Released compute → new attempts</td><td>Signal that stops or redirects refinement</td></tr><tr><td>Self-Refine (Madaan et al., 2023)</td><td></td><td></td><td>x</td><td>stop word in self-feedback, or iteration limit</td></tr><tr><td>SETS (Chen et al., 2025a)</td><td></td><td></td><td>×</td><td>self-verification verdict “correct&quot;, or round limit</td></tr><tr><td>Thought-ICS (Samanta et al., 2026)</td><td></td><td></td><td>×</td><td>self-localization of the first incorrect thought</td></tr><tr><td>Adaptive-Consistency (Aggarwal et al., 2023)</td><td></td><td></td><td>×</td><td>agreement of the vote distribution (stops sampling)</td></tr><tr><td>ConMA (Wang et al., 2026)</td><td></td><td></td><td>√</td><td>token-probability confidence and answer frequency</td></tr><tr><td>DeepConf (Fu et al., 2025)</td><td></td><td></td><td>X</td><td>token-confidence threshold on traces</td></tr><tr><td>Compute-optimal (Snell et al., 2024) SCoRe (Kumar et al., 2024)</td><td></td><td></td><td>V X</td><td>learned verifier and predicted difficulty</td></tr><tr><td></td><td>X</td><td></td><td></td><td>fixed number of trained correction turns</td></tr><tr><td>ARR (ours)</td><td></td><td></td><td></td><td>exact fixed point Fx (y) = y, found during decoding</td></tr></table>

Inference servers reuse cached prefixes across requests (Kwon et al., 2023; Zheng et al., 2024), which is complementary to replay.