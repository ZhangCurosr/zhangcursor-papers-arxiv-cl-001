# ESPO: Error-Structured Prompt Optimization via Diagnose, Diversify, and Stabilize

Lihao Liu, Peng Tang, Kunwar Yashraj Singh, Shabnam Ghadar AWS Agentic AI

## Abstract

Evolutionary prompt optimizers such as GEPA suffer from prompt bloat: each iteration appends rules and caveats, producing prompts up to 3× longer yet no more accurate. We trace this to three deficiencies - incomplete error observation, limited search diversity, and unreliable selection - and propose ESPO (Error-Structured Prompt Optimization), which decomposes prompt optimization into three phases: Diagnose clusters all training errors into structural patterns in one round; Propose generates candidates via four complementary strategies with independent biases; Select applies bootstrap stability selection. On seven public NLP benchmarks - Tweet, MMLU, GSM8K, HotpotQA, ScoNe, HoVer, and PUPA - ESPO improves average accuracy by +3.76 pp over the state-of-the-art (74.67% vs. 70.91% for GEPA), matching or exceeding GEPA on every dataset while producing prompts 47% shorter (1,004 vs. 1,878 chars) and faster at inference. Cross-model experiments across four additional student models (Gemma 3 12B, Mistral 14B, Qwen3 32B, Claude Haiku 4.5) show ESPO yields the best average accuracy on every model tested, with the largest gap on Qwen3 GSM8K (15.00% → 91.40%). A generalization bound (Appendix) grounds each phase in a corresponding term of the test-time gap, and the ablation confirms a key prediction: adding diversity without bootstrap selection actually hurts performance (−1.20%).

## 1 Introduction

Large language models are increasingly steered through natural-language prompts rather than parameter updates (Brown et al., 2020; Wei et al., 2022). As tasks grow more complex, automated prompt optimizers such as APE (Zhou et al., 2023), OPRO (Yang et al., 2024), MIPROv2 (Opsahl-Ong et al., 2024), and GEPA (Agrawal et al., 2026) have emerged to replace manual trial-and-error with systematic search. Among these, GEPA achieves stateof-the-art results through an evolutionary loop: it evaluates prompts on training examples, reflects on errors using a strong LLM, proposes mutations, and selects survivors via a Pareto front.

A persistent shortcoming of evolutionary prompt optimization is prompt bloat: each iteration tends to append rules and caveats, so prompts grow without bound. In our experiments, GEPA produces prompts up to 3× longer than ESPO’s (1.87× on average) yet less accurate from the same weak start, increasing latency and token cost, and can harm generalization by overfitting to training noise. We trace this to three structural deficiencies in the evolutionary paradigm. (i) Incomplete error observation: GEPA reflects on 3–8 random errors per round; by a coupon-collector argument, observing all systematic failure patterns with 95% probability needs ∼15 rounds, and in the meantime the optimizer accumulates redundant or contradictory rules addressing symptoms rather than root causes. (ii) Limited search diversity: a single mutation operator locks in one bias profile; when it fails on certain error types, it compensates by stacking more rules - inflating length without improving accuracy. (iii) Unreliable selection: point estimation on a small (∼30-example) validation set with ∼10 candidates is a multiple-testing problem, and the optimizer may pick a verbose candidate that ranks first by noise rather than true quality.

We propose ESPO (Error-Structured Prompt Optimization), a framework that recasts prompt optimization from evolutionary search to structured statistical estimation. ESPO decomposes optimization into three principled phases. The Diagnose phase analyzes all training errors and clusters them into 3–7 structural patterns, achieving complete error coverage in a single round. The Propose phase generates candidates through K=4 complementary strategies - diagnostic revision, consolidation, ablation, and factual injection - each addressing different error types with different inductive biases.

The Select phase applies bootstrap stability selection (B=20 resamples) to reliably identify the most robust candidate under validation-set noise. ESPO subsumes existing evolutionary methods: setting the number of proposal strategies K=1, bootstrap resamples B=1, and diagnosis batch size m=3 (i.e., only 3 errors observed per reflection round) recovers GEPA, revealing that all three sources of sub-optimality are left unaddressed.

Our contributions are threefold. (1) Framework. We introduce ESPO, a three-phase framework (Diagnose–Propose–Select) that unifies prompt optimization as statistical estimation, subsumes existing evolutionary methods as a degenerate case, and is grounded in a generalization bound whose three terms - bias floor, exploration gain, and selection error - each correspond to one phase (§3; full theory in Appendix A). (2) Experiments. On seven public benchmarks starting from an intentionally weak default prompt, ESPO averages 74.67% accuracy versus 70.91% for GEPA (+3.76 pp), while producing prompts 47% shorter (1,004 vs. 1,878 chars) with consistently lower inference latency (§4). (3) Cross-model generalization. Experiments with four additional student models (Gemma 3 12B, Mistral 14B, Qwen3 32B, Claude Haiku 4.5) show ESPO yields the best average accuracy on every model tested, with the largest gap on Qwen3 GSM8K (15.00% → 91.40% from default; +56.00 pp over GEPA).

## 2 Related Work

Automated prompt optimization. Manual prompt engineering techniques include chainof-thought prompting (Wei et al., 2022) and zero-shot reasoning (Kojima et al., 2022). Automated approaches emerged to reduce human effort: APE (Zhou et al., 2023) uses LLMs to generate and select instructions, OPRO (Yang et al., 2024) optimizes via meta-prompting, and PromptBreeder (Fernando et al., 2024) co-evolves task prompts and mutation operators. Within the DSPy framework (Khattab et al., 2024), COPRO and MIPROv2 (Opsahl-Ong et al., 2024) optimize prompts through Bayesian surrogate models, while GEPA (Agrawal et al., 2026) achieves state-of-the-art results via evolutionary search with Pareto selection and reflective mutation.

Recent extensions and theoretical analysis. A fast-growing literature refines the evolutionary loop with richer memory or multi-objective criteria: ReflectivePrompt (Zhuravlev et al., 2025) (short/longterm reflection), REMO (Wu and Qu, 2025) (memory-augmented reflection), MemAPO (Liang et al., 2026) (dual-memory for error patterns), MOPrompt (Camara et al., 2025) (multi-objective NSGA-II), Optimas (Wu et al., 2025) (compound AI systems), Grießhaber et al. (Grießhaber et al., 2025) (decomposed evolution), and Promptolution (Zehle et al., 2025) (unified framework). On the theory side, Madras et al. (2025) provide PAC-Bayes generalization bounds showing that perplexity regularization prevents overfitting - a complementary principle to ESPO’s bootstrap-based selection stability.

Gradient-based and selection-based methods. A separate line of work tackles prompt optimization through gradient-based or selection-based formulations. TextGrad (Yuksekgonul et al., 2024) uses LLM-generated “textual gradients” to iteratively refine prompts, analogous to backpropagation; while powerful for differentiable-style optimization, it operates on individual examples without structured error clustering, making it susceptible to the same incomplete-observation problem as evolutionary methods. TRIPLE (Shi et al., 2024) frames prompt selection as best-arm identification under a fixed evaluation budget, using bandit algorithms (Successive Halving, Racing) to efficiently allocate evaluations. ESPO’s bootstrap stability selection targets the same selection reliability concern but through resampling rather than sequential elimination, providing complementary guarantees (Lemma 3). Despite this progress, no existing method combines structured error diagnosis to guide candidate generation with selection stability guarantees for small validation sets.

Error analysis and systematic failure discovery. Systematic error analysis identifies structured failure modes in models: failure mode discovery (Eyuboglu et al., 2022) extracts coherent error clusters from embeddings, behavioral testing (Ribeiro et al., 2020) probes known failure categories, and slice-based evaluation (Chen et al., 2019) surfaces underperforming data subsets. All of these tools diagnose what the model gets wrong but stop short of feeding the diagnosis back into optimization; ESPO’s Diagnose phase closes this loop by turning clustered error patterns into inputs for the next round of candidate generation. A parallel line of inquiry on contrastive prompting shows that converting a task from generation to selection mitigates strong model priors at inference time; this generation-vs-selection principle is echoed at the optimization layer by ESPO, which generates $K$ diverse candidates and uses bootstrap resampling to select among them rather than committing to a single generated prompt.

Bootstrap, stability selection, and the MDL principle. The bootstrap (Efron and Tibshirani, 1993) and bagging (Breiman, 1996) reduce prediction variance by resampling, and stability selection (Meinshausen and Bühlmann, 2010) uses subsampling for robust variable selection. Our Select phase adapts these principles to candidate prompt selection: a candidate that repeatedly wins across bootstrap resamples is more likely to generalize than one that wins by chance on a single val split. Finally, the MDL principle (Rissanen, 1978; Grünwald, 2007) formalizes Occam’s razor, and recent work (Déletang et al., 2024; Huang et al., 2024) connects LLM compression to benchmark performance, motivating ESPO’s empirical preference for shorter prompts.

## 3 ESPO Framework

We present ESPO, which decomposes prompt optimization into three phases: structured error diag nosis (§3.2), multi-strategy candidate generation (§3.3), and bootstrap stability selection (§3.4).

## 3.1 Problem Formulation

Given a task with training set $\mathcal { D } _ { \mathrm { t r a i n } }$ , validation set $\mathcal { D } _ { \mathrm { v a l } }$ , and metric $M ( \cdot )$ , prompt optimization seeks an instruction $p ^ { * }$ that maximizes test performance:

$$
p ^ { * } = \underset { p \in \Pi } { \arg \operatorname* { m a x } } ~ \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { \mathrm { t e s t } } } \big [ M ( m _ { p } ( x ) , y ) \big ] ,\tag{1}
$$

where $x$ is the input $( \mathrm { e . g . }$ ., a tweet to classify or a math problem to solve), y is the ground-truth label (e.g., sentiment class or correct answer), $m _ { p }$ denotes the LLM module parameterized by instruction $p \left( \mathrm { i . e . } \right.$ , the model that concatenates prompt $p$ with input x to produce a prediction), and Π is the space of natural-language prompts.

ESPO decomposes this into three phases:

$$
\mathrm { P h a s e ~ 1 } \left( \mathrm { D i a g n o s e } \right) : \quad \varphi = \mathcal { D } ( \mathcal { E } _ { \mathrm { t r a i n } } , p )\tag{2}
$$

$$
\mathrm { P h a s e } 2 ( \mathrm { P r o p o s e } ) : \mathcal { P } = \mathcal { G } ( p , \varphi , S _ { 1 } , \ldots , S _ { K } )\tag{3}
$$

$$
{ \mathrm { P h a s e ~ 3 ~ ( S e l e c t ) : } } \quad p ^ { * } = \mathcal { R } { \left( \mathcal { P } , \mathcal { D } _ { \mathrm { v a l } } , B \right) }\tag{4}
$$

where ${ \mathcal { E } } _ { \mathrm { t r a i n } }$ is the set of training errors, $\varphi$ is a structured diagnosis, $S _ { 1 } , \ldots , S _ { K }$ are proposal strategies, $\mathcal { P }$ is the candidate population, and $B$ is the number of bootstrap resamples.

GEPA as a degenerate case. Let m denote the diagnosis batch size (number of training errors sampled per reflection round). Setting $K { = } 1$ $B { = } 1 , m { = } 3$ exactly recovers GEPA (Agrawal et al., 2026): (i) diagnosis $( m { = } 3  m { = } \mathrm { a l l } )$ , GEPA observes an incomplete error snapshot each round; (ii) proposal diversity $( K { = } 1 \  \ K { = } 4 )$ , GEPA applies a single mutation operator; (iii) selection $( B { = } 1 \to B { = } 2 0 )$ , GEPA uses point estimation on a small validation set. Each dimension independently tightens one term in the generalization bound (Theorem 1), and the degenerate configuration inherits all three weaknesses simultaneously.

## 3.2 Phase 1: Structured Error Diagnosis

We collect all training errors under current prompt:

$$
\begin{array} { r l } & { \mathcal { E } _ { \mathrm { t r a i n } } = \{ ( x _ { i } , y _ { i } , \hat { y } _ { i } ) : M ( m _ { p } ( x _ { i } ) , y _ { i } ) = 0 , } \\ & { \qquad ( x _ { i } , y _ { i } ) \in { \mathcal { D } } _ { \mathrm { t r a i n } } \} . } \end{array}\tag{5}
$$

A reflection LLM clusters ${ \mathcal { E } } _ { \mathrm { t r a i n } }$ into $K ^ { * }$ structural patterns $( K ^ { * } \in [ 3 , 7 ] )$ :

$$
\boldsymbol \varphi = \{ ( \mathrm { p a t t e r n } _ { k } , \mathrm { d e s c r i p t i o n } _ { k } , \mathrm { c o u n t } _ { k } ) \} _ { k = 1 } ^ { K ^ { * } } .\tag{6}
$$

Each pattern has a description of the failure mode, representative examples, and a count. Clustering is implicit (LLM-based), producing descriptions that feed directly into Phase 2. Key advantage: complete error coverage in one round. GEPA’s $m { = } 3$ random sample needs ∼15 rounds to observe all patterns with 95% probability by a coupon-collector argument (§3.5), accumulating redundant rules addressing symptoms rather than root causes.

## 3.3 Phase 2: Multi-Strategy Candidate Generation

Given the diagnosis $\varphi ,$ we generate candidates through $K { = } 4$ complementary strategies, each conditioned on $\varphi \mathrm { : }$

$S _ { 1 }$ (Diagnostic Revision): Revise the prompt to address each error pattern’s root cause using the full diagnosis $\varphi$

$S _ { 2 }$ (Consolidation): Rewrite the prompt without increasing length - merge redundant rules and tighten language, preventing bloat by construction.

• $S _ { 3 }$ (Ablation): Identify over-triggered rules causing false positives and soften or remove them.

$S _ { 4 }$ (Factual Injection): Extract domain-specific knowledge from error examples and inject it as factual context.

Population management. Each of the $K { = } 4$ strategies independently generates 1–2 candidate prompts from the diagnosis $\varphi$ and the current prompt $p _ { 0 } ;$ we call this initial generation step the seed phase, which typically produces 4–6 candidates. We then apply two rounds of crosspollination (merging complementary strengths from different candidates) and targeted refinement (addressing residual errors) on the top candidates, capping the population at N=10. Empirically, no single strategy dominates across all datasets (§4.4), confirming the independent-bias assumption key to the exploration gain in our bound (§3.5).

## 3.4 Phase 3: Bootstrap Stability Selection

Selecting the best candidate from $N { \approx } 1 0$ using a small validation set $( n _ { \mathrm { v a l } } { = } 3 0 )$ is a multiple-testing problem: a candidate may rank first due to favorable noise rather than true quality. We apply B=20 rounds of bootstrap resampling:

$$
p ^ { * } = \underset { p _ { i } \in \mathcal { P } } { \operatorname { a r g m a x } } ~ \big | \{ b : p _ { i } = \underset { p _ { j } \in \mathcal { P } } { \operatorname { a r g m a x } } \operatorname { A c c } ( p _ { j } , \mathcal { V } _ { b } ) \} \big | ,\tag{7}
$$

where $\mathcal { V } _ { b }$ is the b-th bootstrap resample of $\mathcal { D } _ { \mathrm { v a l } }$ (sampling $n _ { \mathrm { v a l } }$ examples with replacement). In words, we select the candidate that wins the most bootstrap resamples. Ties are broken in favor of shorter prompts.

A candidate that consistently ranks first across resamples is robust to data perturbation and more likely to generalize; if $\dot { p } _ { 1 } > 1 / 2$ , the probability of wrong selection decays exponentially with B (Theorem 1). Algorithm 1 summarizes the procedure. ESPO implicitly follows the MDL principle (Rissanen, 1978): clustering compresses n errors into $K ^ { * } \ll n$ patterns, ablation removes unused rules, and consolidation rewrites without growth - yielding shorter prompts with no explicit length penalty.

## 3.5 Theoretical Motivation

Each ESPO phase tightens one term in the test-time generalization gap. Decomposing the gap (§A) into a bias floor, an exploration gain, and a selection error yields:

Theorem 1 (ESPO bound; informal). Suppose K strategies produce candidates with $\mathrm { A c c } _ { t e s t } ( p _ { k } ) =$ $\mathrm { A c c } ^ { * } - b _ { k } + \varepsilon _ { k } , \varepsilon _ { k } \sim \mathcal N ( 0 , \sigma ^ { 2 } )$ , and bootstrap selection uses B resamples. Then the selected prompt $p ^ { * }$ satisfies

$$
\begin{array} { r } { \mathbb { E } [ \operatorname { A c c } _ { t e s t } ( p ^ { * } ) ] \geq \operatorname { A c c } ^ { * } - \operatorname* { m i n } _ { k } { b _ { k } } + \sigma \Phi ^ { - 1 } ( 1 - \frac { 1 } { K } ) - O ( \sqrt { \frac { \ln K } { n _ { \nu a l } B } } ) . } \end{array}
$$

The bound is supported by three lemmas (full statements and proofs in Appendix A): Lemma 1 (bias reduction via diversity: max<sub>k</sub> $\mathbb { E } [ \mathrm { A c c } ( p _ { k } ) ] \geq$ $\begin{array} { r } { \operatorname { A c c } ^ { * } - \operatorname* { m i n } _ { k } b _ { k } ) ; } \end{array}$ Lemma 2 (exploration gain via order statistics: drawing K candidates and taking the best yields a $\sigma \cdot \Phi ^ { - 1 } ( 1 { - } 1 / K )$ bonus); and Lemma 3 (selection precision via bootstrap: with $p _ { 1 } ~ > ~ 1 / 2$ , Pr(wrong selection) $\leq$ $\exp ( - 2 B ( p _ { 1 } - 1 / 2 ) ^ { 2 } ) )$ . Two caveats: (i) the constant under $O ( \cdot )$ in the selection term is shared between GEPA and ESPO, so the $\sim 3 . 8 \times$ comparison reflects the $\sqrt { \ln K / B }$ ratio rather than absolute magnitude; (ii) Lemma $3 ^ { \circ } \mathrm { { s } } p _ { 1 } > 1 / 2$ premise holds when best is separated from runners-up by more than validation noise scale, which is the regime our ablation operates in but is not certified per dataset.

Algorithm 1 ESPO Optimization Algorithm.   
Require: Training set $\mathcal { D } _ { \mathrm { t r a i n } } .$ validation set $\mathcal { D } _ { \mathrm { v a l } } .$ initial prompt   
p<sub>0</sub>, strategies $\mathrm { \bar { \cal S } } _ { 1 } , \dots , { \cal S } _ { K } ,$ bootstrap rounds B   
Ensure: Optimized prompt $p ^ { * }$   
# Phase 1: Diagnose   
1: $\mathcal { E }  \{ ( x _ { i } , y _ { i } , \stackrel { \smile } { \hat { y } _ { i } } ) : M ( m _ { p _ { 0 } } ( x _ { i } ) , y _ { i } ) = 0 \}$   
▷ collect all errors   
2: φ ← CLUSTERERRORS $\left( \mathcal { E } , \mathrm { L L M } _ { \mathrm { r e f l e c t } } \right)$   
▷ 3–7 structural patterns   
# Phase 2: Propose (Seed)   
3: $\mathcal { P }  \{ p _ { 0 } \}$   
4: for $k \stackrel { - } { = } 1$ to K do   
5: $p _ { k }  S _ { k } ( p _ { 0 } , \varphi , \mathrm { L L M } _ { \mathrm { r e f l e c t } } )$   
6: $\mathsf { \bar { P } } \gets \mathcal { P } \cup \{ p _ { k } \}$   
7: end for   
# Phase 2: Propose (Iterate)   
8: for iter = 1 to 2 do   
9: Score all candidates on $\mathcal { D } _ { \mathrm { v a l } }$   
10: $\mathcal { P }  \mathcal { P } \cup \mathcal { ( }$ CROSSPOLLINATE(top(P))   
∪ REFINE(top(P), φ)   
11: Prune $\mathcal { P }$ to $\bar { N } { = } 1 0$ by (score, −length)   
12: end for   
# Phase 3: Select (Bootstrap)   
13: wins $[ i ]  0$ for all $p _ { i } \in \mathcal { P }$   
14: for $b { \overset { \cdot } { = } } 1$ to B do   
15: $\nu _ { b } \gets$ BOOTSTRAPRESAMPL $. \mathrm { E } ( \mathcal { D } _ { \mathrm { v a l } } )$   
16: wins[arg max $\mathbf { \Phi } _ { i } \operatorname { A c c } ( p _ { i } , \mathcal { V } _ { b } ) ] \mathrel { + } \dot { = } 1$   
17: end for   
18: return $p ^ { * }  \mathcal { P } |$ [arg max<sub>i</sub>(wins[i], −|p<sub>i</sub>|)]

Why GEPA is suboptimal. Setting K=1, B=1 collapses the bound to $\mathrm { A c c } ^ { * } { - } b _ { 1 } { - } O ( 1 / \sqrt { n _ { \mathrm { v a l } } } )$ : the exploration gain vanishes, the bias term loses the min<sub>k</sub> improvement, and the selection error stays at $O ( 1 / \sqrt { n _ { \mathrm { v a l } } } )$ instead of the ∼3.8×-tighter ESPO rate. A coupon-collector argument (Appendix B) further shows that GEPA’s m=3 random sampling needs ${ \sim } 1 5$ reflection rounds to observe all error patterns with 95% probability, while full-batch diagnosis $( m { = } n )$ achieves complete coverage in one round, directly reducing each $b _ { k }$ .

Table 1: Test accuracy (%) and prompt length (characters) on seven benchmarks with Claude Sonnet 4.5 as student. All methods start from the same deliberately weak task prompt. Cell values are the single-run test accuracies from our primary optimization run; ± is std over 3 additional independent optimization seeds. The appendix provides the full 3-seed means and stds. Best accuracy per row in bold.
<table><tr><td></td><td colspan="2">Default</td><td colspan="2">Bootstrap</td><td colspan="2">COPRO</td><td colspan="2">MIPROv2</td><td colspan="2">GEPA</td><td colspan="2">ESPO (Ours)</td></tr><tr><td>Dataset</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td></tr><tr><td>Tweet</td><td> $3 7 . 6 0 { \scriptstyle \pm 0 . 2 0 }$ </td><td>609</td><td> $3 7 . 8 0 { \scriptstyle \pm 0 . 2 0 }$ </td><td>607</td><td> $6 6 . 6 0 \pm 1 . 8 0$ </td><td>2,398</td><td> $7 4 . 4 0 \pm 1 . 5 0$ </td><td>1,673</td><td> $6 8 . 6 0 \pm 2 . 2 0 $ </td><td>1,279</td><td> $7 4 . 7 8 \pm 1 . 1 8$ </td><td>774</td></tr><tr><td>MMLU</td><td> $2 3 . 2 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>57</td><td> $2 3 . 0 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>55</td><td> $8 9 . 6 0 \pm 1 . 2 0 $ </td><td>871</td><td> $8 6 . 6 0 \pm 1 . 4 0$ </td><td>775</td><td> $8 9 . 6 0 \pm 1 . 5 0 $ </td><td>1,163</td><td> $\mathbf { 9 4 . 5 2 \pm 0 . 8 4 }$ </td><td>766</td></tr><tr><td>GSM8K</td><td> $3 2 . 6 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td>110</td><td> $3 6 . 8 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td>84</td><td> $3 6 . 8 0 \pm 1 . 0 0$ </td><td>84</td><td> $3 7 . 2 0 \pm 1 . 0 0$ </td><td>84</td><td> $\mathbf { 9 6 . 8 0 \bot 0 . 5 0 }$ </td><td>660</td><td> $\mathbf { 9 6 . 8 0 \bot 0 . 4 0 }$ </td><td>660</td></tr><tr><td>HotpotQA</td><td> $2 5 . 8 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>89</td><td> $1 4 . 4 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>73</td><td> $1 4 . 6 0 { \scriptstyle \pm 0 . 9 0 }$ </td><td>73</td><td> $1 4 . 2 0 { \scriptstyle \pm 0 . 9 0 }$ </td><td>73</td><td> $4 4 . 0 0 \pm 1 . 8 0 $ </td><td>2,845</td><td> $\pm 4 . 8 0 \pm 1 . 4 0 $ </td><td>1,008</td></tr><tr><td>ScoNe</td><td> $4 9 . 0 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>381</td><td> $4 6 . 0 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>323</td><td> $4 6 . 2 0 \pm 1 . 5 0$ </td><td>323</td><td> $4 6 . 4 0 \pm 1 . 5 0$ </td><td>323</td><td> $8 5 . 0 0 \pm 2 . 0 0$ </td><td>1,948</td><td> $\mathbf { 8 9 . 4 0 \pm 1 . 2 0 }$ </td><td>1,124</td></tr><tr><td>HoVer</td><td> $4 8 . 0 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>187</td><td> $4 8 . 0 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>159</td><td> $4 8 . 4 0 \pm 1 . 6 0$ </td><td>159</td><td> $4 7 . 8 0 \pm 1 . 6 0$ </td><td>159</td><td> $5 3 . 6 0 \pm 2 . 4 0$ </td><td>2,736</td><td> ${ \bf 6 2 . 4 0 \pm 1 . 5 0 }$ </td><td>1,934</td></tr><tr><td>PUPA</td><td> $1 . 6 1 \pm 0 . 1 9$ </td><td>67</td><td> $1 . 5 3 \pm 0 . 1 2$ </td><td>67</td><td> $5 4 . 3 0 \pm 2 . 0 0$ </td><td>807</td><td> $1 . 6 1 \pm 0 . 1 2$ </td><td>1,431</td><td> $5 8 . 8 0 \pm 2 . 3 0 $ </td><td>2,518</td><td> ${ \bf 6 0 . 0 0 } \pm 1 . 6 0$ </td><td>765</td></tr><tr><td>Avg</td><td> $3 1 . 1 2 \pm 0 . 2 8$ </td><td>214</td><td> $2 9 . 6 5 \pm 0 . 2 7$ </td><td>195</td><td> $5 0 . 9 3 \pm 1 . 4 3$ </td><td>674</td><td> $4 4 . 0 3 \pm 1 . 1 5$ </td><td>645</td><td> $7 0 . 9 1 \pm 1 . 8 1$ </td><td>1,878</td><td> ${ \bf 7 4 . 6 7 \pm 1 . 1 6 }$ </td><td>1,004</td></tr></table>

Empirical alignment. The bound’s three predictions match the ablation in §4.5: Diagnose alone yields +2.0% (bias term), $K { = } 1 {  } 4 ~ \mathrm { a d d s } ~ { + } 2 . 6 \%$ (exploration), and $B { \geq } 1 0$ picks a more compact, generalizing candidate (selection). Diversity without bootstrap hurts (−1.20%), as predicted: adding K without raising B amplifies selection error.

## 4 Experiments

We evaluate ESPO on seven public benchmarks covering classification, multi-hop reasoning, math, and privacy-preserving generation. Our experiments ask: (Q1) Starting from a deliberately weak task prompt, can ESPO recover a prompt that is both more accurate and more compact than baselines? (Q2) Does this hold across student models of different capacity? (Q3) Do the four proposal strategies exhibit genuine diversity? (Q4) Does bootstrap selection outperform point estimation?

## 4.1 Setup

Benchmarks and data splits. We use seven datasets: Tweet (sentiment classification), MMLU (multiple-choice QA), GSM8K (grade school math), HotpotQA (multi-hop QA), ScoNe (natural language inference), HoVer (multi-hop fact verification), and PUPA (privacy-preserving response generation). Tweet, MMLU, and PUPA use direct prediction; the others use chain-of-thought. Each dataset uses 70 training examples for optimization, 30 validation examples, and 500 held-out for test.

A deliberately weak starting prompt. Prior benchmarks initialize from a hand-crafted, tasktuned prompt that already performs well, leaving little headroom and clustering optimizers within 1– 2 pp of each other. We instead start every method from an intentionally weak task prompt (e.g. Tweet: a single sentence telling the model to lean “negative”; PUPA: “Refuse to answer”). Each weak prompt is selected as the lowest-accuracy one on a 30-example val probe from a small candidate pool; none is written to favor any optimizer. The goal is to test whether an optimizer can recover a good prompt from a bad one, not whether it can polish an already-good one.

Models and Baseline Default student: Claude Sonnet 4.5 (temperature = 0). Cross-model analysis adds Gemma 3 12B, Mistral 14B, Qwen3 32B, Claude Haiku 4.5. Reflection LLM is always Claude Sonnet 4.5 (temperature = 0.7). For baselines: Default (no optimization); Bootstrap (BootstrapFewShot from DSPy (Khattab et al., 2024)); COPRO (Pryzant et al., 2023); MIPROv2 (Opsahl-Ong et al., 2024); GEPA (Agrawal et al., 2026). All methods start from the same weak default prompt.

Metrics. Classification accuracy (Tweet, MMLU, HoVer), exact-match with normalization (GSM8K, HotpotQA, ScoNe), semantic F1 (PUPA), prompt length (chars), inference latency (s/example). Testtime inference is deterministic. The 95% binomial CI at $\scriptstyle { p = 0 . 7 5 , n = 5 0 0 { \mathrm { ~ i s } } \pm 3 . 8 \% }$

## 4.2 Main Results

Table 1 shows the central finding: when all methods start from the same weak prompt, only reflectionbased optimizers recover a usable prompt, and ESPO recovers the best one on average.

Non-reflective methods cannot salvage a weak prompt. Bootstrap (29.65%) is slightly worse than the default (31.12%) - few-shot demonstrations atop a misleading instruction just amplify wrong behavior. COPRO (50.93%) and MIPROv2 (44.03%) rewrite the instruction and do better, but still trail GEPA/ESPO by 20–30 pp on average and stay near floor on PUPA. These methods require a reasonably good starting prompt to shine, whereas GEPA and ESPO do not.

<table><tr><td>Default (609c)</td><td>GEPA (1,279c)</td><td>ESPO (774c)</td></tr><tr><td>Classify the tweet by leaning &quot;negative&quot; [deliber- ately weak seed, 609 chars]</td><td>Classify the sentiment... Pay special attention to sarcasm. If the tweet contains “but&quot; consider both clauses. Watch for negation words. Consider emoji sentiment. Double-check ambiguous cases Note: “not bad&quot; is positive. Beware of rhetorical questions... [truncated, 1,279 chars]</td><td>Classify the sentiment of the tweet as positive, negative, or neutral. Sarcasm and rhetorical questions should be interpreted by their intended meaning, not literal words. Treat “not bad&quot; pat- terns as positive. [774 chars]</td></tr><tr><td>37.60%</td><td>68.60% (+31.00)</td><td>74.78% (+37.18)</td></tr></table>

Figure 1: Prompt comparison on Tweet, all starting from the same deliberately weak seed (Table 1). GEPA accumulates redundant rules (1.7× longer than ESPO) yet trails ESPO by 6.18 pp. ESPO produces a concise, targeted prompt.

ESPO beats GEPA. ESPO averages 74.67% vs. GEPA’s 70.91% (+3.76 pp; per-cell ±std over 3 seeds in Table 1, and the appendix provides the full 3-seed grid). The paired standard error of the ESPO−GEPA difference across the 7 datasets is SE = std $\langle \Delta _ { i } \} _ { i = 1 } ^ { 7 } ) / \sqrt { 7 } \approx 1 . 2$ pp, so the Avg gap is ≈3.1× its SE (paired t-test significant at α=0.05). ESPO’s cell value is ≥ GEPA’s on all 7 datasets. Outside-noise wins (>1σ): HoVer (+8.80), Tweet (+6.18), MMLU (+4.92), ScoNe (+4.40); within-1σ ties: GSM8K (tie at ceiling), HotpotQA (+0.80), PUPA (+1.20). We describe HotpotQA and PUPA as on par with GEPA.

ESPO’s prompts are shorter. Average length 1,004 vs. 1,878 chars (47% shorter). On HotpotQA, GEPA grows to 2,845 chars while ESPO stays at 1,008 chars with comparable accuracy (44.00% vs. 44.80%). Bootstrap resampling implicitly prefers candidates that retain val accuracy across perturbations - these correlate with concise instructions that do not overfit to training noise. Figure 1 compares prompts produced by Default, GEPA, ESPO. For latency, ESPO also achieves equal or lower perexample latency than GEPA on every task (Table 9); the largest reductions are on chain-of-thought tasks where shorter prompts produce more concise reasoning chains.

Constrained GEPA: is gain just length control? To test whether ESPO’s gains come purely from prompt length control, we evaluate a Constrained GEPA variant: standard GEPA augmented with (1) a compact proposer instruction and (2) a length constraint (max\_length\_ratio = 1.2, preventing bloat beyond 1.2× the seed prompt). This isolates whether length control alone can close the gap. Results are in Table 2.

Table 2: Constrained GEPA vs. standard GEPA and ESPO. Single-seed values (the constrained-GEPA variant was not rerun over 3 seeds); the point-estimate view of Table 1. Length control alone does not improve GEPA. Best accuracy in bold.
<table><tr><td></td><td colspan="2">GEPA</td><td colspan="2">Constrained GEPA</td><td colspan="2">ESPO</td></tr><tr><td>Dataset</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td></tr><tr><td>Tweet</td><td>68.60</td><td>1,279</td><td>71.00</td><td>740</td><td>74.78</td><td>774</td></tr><tr><td>MMLU</td><td>89.60</td><td>1,163</td><td>89.40</td><td>663</td><td>94.52</td><td>766</td></tr><tr><td>GSM8K</td><td>96.80</td><td>660</td><td>96.60</td><td>660</td><td>96.80</td><td>660</td></tr><tr><td>HotpotQA</td><td>44.00</td><td>2,845</td><td>39.80</td><td>751</td><td>44.80</td><td>1,008</td></tr><tr><td>ScoNe</td><td>85.00</td><td>1,948</td><td>86.80</td><td>1,171</td><td>89.40</td><td>1,124</td></tr><tr><td>HoVer</td><td>53.60</td><td>2,736</td><td>54.18</td><td>2,311</td><td>62.40</td><td>1,934</td></tr><tr><td>PUPA</td><td>58.80</td><td>2,518</td><td>59.24</td><td>1,904</td><td>60.00</td><td>765</td></tr><tr><td>Avg</td><td>70.91</td><td>1,878</td><td>71.00</td><td>1,171</td><td>74.67</td><td>1,004</td></tr></table>

Analysis. Constrained GEPA successfully produces shorter prompts (1,171c vs. 1,878c, a 38% reduction), demonstrating that length control alone can reduce prompt bloat. However, this compression fails to improve accuracy: average accuracy moves by only +0.09% (71.00% vs. 70.91%), and on HotpotQA it degrades by −4.20%. The constraint forces GEPA to discard rules indiscriminately - without structured error diagnosis to distinguish essential rules from redundant ones, it cannot determine which rules to keep and which to prune. In contrast, ESPO achieves both shorter prompts and higher accuracy (74.67%, 1,004c) because Phase 1 (structured diagnosis) identifies which error patterns matter, Phase 2 (multi-strategy proposals) generates candidates that are concise by construction (especially $S _ { 2 } { \mathrm { : } }$ consolidation and S<sub>3</sub>: ablation), and Phase 3 (bootstrap selection) favors stable, generalizable candidates - which empirically tend to be shorter.

## 4.3 Cross-Model Generalization

To test whether ESPO’s gains transfer beyond Claude Sonnet 4.5, we apply every optimizer to four additional student models - Gemma 3 12B, Mistral 14B, Qwen3 32B, and Claude Haiku 4.5 - while keeping Claude Sonnet 4.5 as the reflection LLM. Every method starts from the same weak default prompt and runs its full optimization pipeline from scratch with the new student. Table 3 reports the complete 7-benchmark grid for all six optimizers on each student.

Table 3: Cross-model generalization: test accuracy (%) and prompt length (characters) across four student models. All methods start from the same deliberately weak task prompt; reflection LLM is Claude Sonnet 4.5 throughout. Best accuracy per row in bold.
<table><tr><td></td><td></td><td colspan="2">Default</td><td colspan="2">Bootstrap</td><td colspan="2">COPRO</td><td colspan="2">MIPROv2</td><td colspan="2">GEPA</td><td colspan="2">ESPO (Ours)</td></tr><tr><td>Model</td><td>Dataset</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td></tr><tr><td rowspan="8">Gemma 3 12B</td><td>Tweet</td><td>67.20</td><td>487</td><td>40.80</td><td>607</td><td>66.20</td><td>625</td><td>45.20</td><td>1,667</td><td>68.60</td><td>1,279</td><td>71.20</td><td>774</td></tr><tr><td>MMLU</td><td>70.20</td><td>522</td><td>22.00</td><td>55</td><td>22.00</td><td>181</td><td>69.20</td><td>1,760</td><td>69.60</td><td>1,711</td><td>70.40</td><td>148</td></tr><tr><td>GSM8K</td><td>88.18</td><td>637</td><td>70.80</td><td>84</td><td>70.60</td><td>84</td><td>70.60</td><td>84</td><td>88.18</td><td>660</td><td>88.40</td><td>660</td></tr><tr><td>HotpotQA</td><td>23.00</td><td>665</td><td>14.80</td><td>73</td><td>15.20</td><td>73</td><td>15.60</td><td>73</td><td>19.40</td><td>2,845</td><td>23.40</td><td>701</td></tr><tr><td>ScoNe</td><td>78.20</td><td>1,045</td><td>58.80</td><td>323</td><td>58.60</td><td>323</td><td>58.20</td><td>323</td><td>77.60</td><td>1,948</td><td>79.40</td><td>1,215</td></tr><tr><td>HoVer</td><td>55.00</td><td>187</td><td>48.80</td><td>159</td><td>48.00</td><td>159</td><td>50.40</td><td>159</td><td>53.00</td><td>2,076</td><td>59.60</td><td>1,045</td></tr><tr><td>PUPA</td><td>1.87</td><td>67</td><td>1.67</td><td>67</td><td>1.00</td><td>267</td><td>1.65</td><td>67</td><td>69.57</td><td>634</td><td>73.12</td><td>442</td></tr><tr><td>Avg</td><td>54.81</td><td>516</td><td>36.81</td><td>195</td><td>40.23</td><td>245</td><td>44.41</td><td>590</td><td>63.71</td><td>1,593</td><td>66.50</td><td>712</td></tr><tr><td rowspan="8">Mistral 14B</td><td>Tweet</td><td>65.40</td><td>487</td><td>40.20</td><td>607</td><td>64.40</td><td>1,866</td><td>63.80</td><td>2,398</td><td>64.40</td><td>1,279</td><td>70.40</td><td>1,019</td></tr><tr><td>MMLU</td><td>70.80</td><td>522</td><td>26.60</td><td>55</td><td>52.80</td><td>49</td><td>71.40</td><td>1,303</td><td>71.60</td><td>1,711</td><td>71.60</td><td>663</td></tr><tr><td>GSM8K</td><td>93.39</td><td>637</td><td>38.28</td><td>84</td><td>41.28</td><td>84</td><td>41.08</td><td>84</td><td>92.80</td><td>660</td><td>94.59</td><td>805</td></tr><tr><td>HotpotQA</td><td>24.40</td><td>665</td><td>5.44</td><td>73</td><td>5.42</td><td>73</td><td>5.03</td><td>73</td><td>18.00</td><td>2,845</td><td>27.11</td><td>1,020</td></tr><tr><td>ScoNe</td><td>74.20</td><td>1,045</td><td>49.60</td><td>323</td><td>48.20</td><td>323</td><td>48.60</td><td>323</td><td>76.00</td><td>1,948</td><td>78.60</td><td>1,171</td></tr><tr><td>HoVer</td><td>48.40</td><td>187</td><td>49.00</td><td>159</td><td>48.60</td><td>159</td><td>48.60</td><td>159</td><td>49.40</td><td>2,170</td><td>50.00</td><td>1,099</td></tr><tr><td>PUPA</td><td>3.93</td><td>67</td><td>4.37</td><td>67</td><td>3.69</td><td>67</td><td>36.16</td><td>3,996</td><td>41.81</td><td>615</td><td>44.62</td><td>229</td></tr><tr><td>Avg</td><td>54.36</td><td>516</td><td>30.50</td><td>195</td><td>37.77</td><td>374</td><td>44.95</td><td>1,191</td><td>59.14</td><td>1,604</td><td>62.42</td><td>858</td></tr><tr><td rowspan="8">Qwen3 32B</td><td>Tweet</td><td>70.60</td><td>487</td><td>43.20</td><td>607</td><td>54.40</td><td>392</td><td>55.20</td><td>1,699</td><td>69.00</td><td>1,279</td><td>71.00</td><td>1,216</td></tr><tr><td>MMLU</td><td>76.80</td><td>522</td><td>35.80</td><td>55</td><td>29.60</td><td>135</td><td>75.20</td><td>1,580</td><td>77.40</td><td>1,711</td><td>78.20</td><td>771</td></tr><tr><td>GSM8K</td><td>15.00</td><td>637</td><td>38.00</td><td>84</td><td>38.80</td><td>84</td><td>39.20</td><td>84</td><td>35.40</td><td>660</td><td>91.40</td><td>695</td></tr><tr><td>HotpotQA</td><td>26.20</td><td>665</td><td>7.00</td><td>73</td><td>6.80</td><td>73</td><td>7.20</td><td>73</td><td>15.40</td><td>2,845</td><td>27.20</td><td>1,108</td></tr><tr><td>ScoNe</td><td>82.60</td><td>1,045</td><td>64.80</td><td>323</td><td>65.40</td><td>323</td><td>65.20</td><td>323</td><td>83.80</td><td>1,948</td><td>84.20</td><td>1,275</td></tr><tr><td>HoVer</td><td>49.20</td><td>187</td><td>37.60</td><td>159</td><td>38.20</td><td>159</td><td>37.00</td><td>159</td><td>58.20</td><td>2,371</td><td>53.00</td><td>346</td></tr><tr><td>PUPA</td><td>1.93</td><td>67</td><td>1.46</td><td>67</td><td>72.61</td><td>60</td><td>53.97</td><td>505</td><td>73.18</td><td>1,124</td><td>74.57</td><td>239</td></tr><tr><td>Avg</td><td>46.05</td><td>516</td><td>32.55</td><td>195</td><td>43.69</td><td>175</td><td>47.57</td><td>632</td><td>58.91</td><td>1,705</td><td>68.51</td><td>807</td></tr><tr><td rowspan="8">Haiku 4.5</td><td>Tweet</td><td>51.40</td><td>487</td><td>48.20</td><td>607</td><td>55.80</td><td>1,910</td><td>65.20</td><td>1,750</td><td>71.00</td><td>1,450</td><td>73.20</td><td>1,025</td></tr><tr><td>MMLU</td><td>53.60</td><td>522</td><td>53.20</td><td>55</td><td>53.20</td><td>55</td><td>81.60</td><td>1,426</td><td>81.80</td><td>388</td><td>83.00</td><td>107</td></tr><tr><td>GSM8K</td><td>17.00</td><td>637</td><td>3.60</td><td>84</td><td>3.40</td><td>84</td><td>3.60</td><td>84</td><td>90.00</td><td>367</td><td>92.40</td><td>151</td></tr><tr><td>HotpotQA</td><td>13.80</td><td>665</td><td>9.00</td><td>73</td><td>9.20</td><td>73</td><td>9.00</td><td>73</td><td>31.20</td><td>2,678</td><td>33.80</td><td>801</td></tr><tr><td>ScoNe</td><td>45.80</td><td>1,045 187</td><td>45.00 47.00</td><td>323 159</td><td>45.20 47.40</td><td>323 159</td><td>44.80 47.20</td><td>323 159</td><td>81.40 48.80</td><td>2,560 187</td><td>85.20 53.60</td><td>725 479</td></tr><tr><td>HoVer PUPA</td><td>48.80 6.88</td></table>

Findings. (1) ESPO produces the best average accuracy on every student: Gemma 66.50%, Mistral 62.42%, Qwen3 68.31%, Haiku 70.15% - each strictly above GEPA (63.71%, 59.14%, 59.11%, 67.98%). The largest GEPA → ESPO gap is on Qwen3 (+9.20 pp average). (2) The Qwen3 GSM8K result is striking: Default 15.00% → GEPA 35.40% → ESPO 91.40% (+56.00 pp over GEPA). ESPO’s structured diagnosis cleanly identifies the output-format mismatch silently throttling Qwen3, which GEPA’s rule-accumulation only partially addresses. (3) Bootstrap, COPRO, and MIPROv2 again fail to recover a usable prompt, averaging 30–48% across students - 15–30 pp behind GEPA/ESPO. (4) ESPO prompts are 30– 60% shorter than GEPA’s across all four students, confirming that the concise-by-construction behavior transfers cleanly across model families.

## 4.4 Strategy Diversity Analysis

Table 8 (Appendix) reports seed-phase validation accuracy per strategy. No single strategy dominates: diagnostic revision wins on Tweet, consolidation on HotpotQA, ablation ties with diagnostic on MMLU and with consolidation/factual on GSM8K, factual injection wins on HoVer/PUPA. This confirms the independent-bias assumption underlying Theorem 1.

## 4.5 Ablation Study

We ablate ESPO’s components on Tweet (Table 4; ∆ vs. GEPA). Diagnose = full-batch diagnosis (m=all); Diversity = K=4; Bootstrap = B=20.

The ablation reveals a clear progression. Single components: Bootstrap gives the largest individual gain (+3.60%), then Diagnose (+2.00%); Diversity alone hurts (−1.20%) - a direct prediction of Theorem 1: adding K without raising B amplifies selection error. Pairs: Diversity+Bootstrap (−1.00%) shows diversity without diagnosis is also detrimental - without structured error coverage, diverse strategies vary randomly rather than complementarily. Diagnose+anything is positive. All three: +6.18%, exceeding the sum of any pair - confirming that better candidates require better selection.

Table 4: Ablation on Tweet: ✓ = active, D = Diagnose, K = Diversity (K=4), B = Bootstrap (B=20).
<table><tr><td>Configuration</td><td>D</td><td>K</td><td>B</td><td>Acc (%)</td></tr><tr><td>GEPA baseline</td><td></td><td></td><td></td><td>68.60</td></tr><tr><td>Diagnose only</td><td></td><td> $\checkmark \mathrm { ~ - ~ } -$ </td><td>70.60</td><td> $+ 2 . 0 0$ </td></tr><tr><td>Diversity only</td><td></td><td>√</td><td></td><td>67.40  $- 1 . 2 0$ </td></tr><tr><td>Bootstrap only</td><td></td><td>√</td><td>72.20</td><td> $+ 3 . 6 0$ </td></tr><tr><td>D + K</td><td>V</td><td>√</td><td></td><td>70.60  $+ 2 . 0 0$ </td></tr><tr><td> $\mathrm { D } + \mathrm { B }$ </td><td>√</td><td></td><td>√</td><td>70.80  $+ 2 . 2 0$ </td></tr><tr><td>K+B</td><td>一</td><td>√</td><td>√ 67.60</td><td> $- 1 . 0 0$ </td></tr><tr><td>ESPO (all)</td><td>√</td><td>√</td><td>√</td><td>74.78  $\mathbf { + 6 . 1 8 }$ </td></tr></table>

Table 5: Hyperparameter sensitivity. Each group varies one parameter while fixing others at ESPO defaults. All differences fall within the 95% binomial CI.
<table><tr><td>Param</td><td>Value</td><td>Acc (%)</td><td>Len</td><td>Sel.</td></tr><tr><td rowspan="4">K</td><td>1</td><td>72.2</td><td>806</td><td></td></tr><tr><td>2</td><td>70.0</td><td>922</td><td></td></tr><tr><td>3</td><td>75.0</td><td>1,134</td><td></td></tr><tr><td>4</td><td>74.8</td><td>774</td><td></td></tr><tr><td rowspan="4">B</td><td>1</td><td>73.8</td><td>1,214</td><td>abl.</td></tr><tr><td>10</td><td>74.0</td><td>958</td><td>c.p.</td></tr><tr><td>20</td><td>74.8</td><td>774</td><td>c.p.</td></tr><tr><td>30</td><td>74.8</td><td>774</td><td>c.p.</td></tr><tr><td rowspan="4">m</td><td>3</td><td>71.0</td><td>1,015</td><td></td></tr><tr><td>8</td><td>69.8</td><td>1,148</td><td></td></tr><tr><td> $^ { 1 5 }$ </td><td>72.6</td><td>1,100</td><td></td></tr><tr><td>all</td><td>74.8</td><td>774</td><td></td></tr></table>

## 4.6 Hyperparameter Sensitivity

Table 5 shows ESPO is robust: accuracy varies ≤5% across settings, within the CI. The default (K=4, B=20, m=all) gives the best accuracy– length trade-off - K=4 is 32% shorter than K=3 at comparable accuracy; $B { \geq } 1 0$ selects a more stable, compact candidate over the point-estimation pick; full-batch diagnosis yields the highest accuracy and shortest prompt, confirming Proposition 2.

Training cost. Costs (Table 10) are dominated by student inference during bootstrap evaluation. ESPO’s reflection tokens are ∼39% of GEPA’s, because structured diagnosis produces more information-dense inputs.

## 5 Conclusion

We introduced ESPO (Error-Structured Prompt Optimization), a framework that recasts prompt optimization from evolutionary search to structured statistical estimation. By decomposing optimization into three principled phases - structured error diagnosis, multi-strategy candidate generation, and bootstrap stability selection - ESPO addresses the fundamental weaknesses of the search paradigm: incomplete error observation, limited diversity, and unreliable selection. We showed that existing evolutionary methods like GEPA are a degenerate special case of ESPO, and provided a generalization bound grounding each phase in established results from probability theory and statistics. On seven public benchmarks, ESPO improves average accuracy by +3.76% over the state-of-the-art (74.67% vs. 70.91% for GEPA) and matches or exceeds GEPA on every dataset, while producing prompts 47% shorter (1,004 vs. 1,878 chars) and faster at inference. Cross-model experiments on four additional student models (Gemma 3 12B, Mistral 14B, Qwen3 32B, Claude Haiku 4.5) confirm that the gains transfer broadly, with the largest gap on Qwen3 GSM8K (15.00% → 91.40%).

## 6 Limitations

Optimization cost. Although our reflectiontoken usage is ∼39% of GEPA’s (Table 10), bootstrap selection still requires $B \times N$ candidateresample evaluations. A full ESPO run with K=4, B=20, N=10 on our 70/30/500 split costs roughly the same as one GEPA run at default settings; practitioners with tighter budgets can lower B or N.

Single reflection model and independence. All reflection uses one LLM (Claude Sonnet 4.5, T=0.7); mixing reflection models per strategy is not explored. Theorem 1’s independence assumption is therefore a simplification - candidate distributions across strategies are correlated through the shared LLM - but our empirical exploration gain is consistent with the bound under mild correlation.

Coverage of error patterns. Diagnosis relies on the reflection LLM to cluster errors into 3–7 patterns; when patterns are more numerous or subtle (e.g., distributional shifts rather than discrete modes), diagnosis may be incomplete.

Scope and reporting. We evaluate on seven public benchmarks across classification, multi-hop

QA, math, NLI, fact verification, and privacypreserving generation; tool use, long context, code, and multi-turn dialogue are out of scope. ESPO’s largest cross-model gains occur where the default prompt is far from the student’s prior (e.g., Qwen3 GSM8K). Main-table cells report the single-run cell with ±std over 3 optimization seeds attached; the appendix gives the full 6×7 3-seed grid. Perdataset gaps within 1σ (HotpotQA +0.80, PUPA +1.20) should be read as on-par with GEPA.

Theoretical assumptions. Theorem 1 is stated as an explanatory framework motivating the threephase decomposition rather than as a tight probabilistic guarantee. It relies on three assumptions: (i) proposal strategies are independent (a simplification: all four are conditioned on the same reflection LLM); (ii) validation-set noise is sub-Gaussian (motivated by CLT on n=500 Bernoulli trials); and (iii) the true-best candidate wins each bootstrap resample with probability $p _ { 1 } > 1 / 2$ (holds when the top two candidates are separated by more than one validation std, our empirical regime but not certified per dataset). The appendix quantifies strategy overlap empirically (pairwise Jaccard 0.62, Pearson 0.48 over 500 held-out examples per candidate pair) partial decorrelation, not full independence.

## References

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. 2026. GEPA: Reflective prompt evolution can outperform reinforcement learning. In International Conference on Learning Representations (ICLR). Oral presentation.

Leo Breiman. 1996. Bagging predictors. Machine Learning, 24(2):123–140.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 1877–1901.

Sara Camara, Eduardo Luz, Valeria Carvalho, Ivan Meneghini, and Gladston Moreira. 2025. MOPrompt: Multi-objective semantic evolution for prompt optimization. arXiv preprint arXiv:2508.01541.

Vincent S. Chen, Sen Wu, Zhenzhen Weng, Alexander Ratner, and Christopher Ré. 2019. Slice-based learning: A programming model for residual learning in critical data slices. In Advances in Neural Information Processing Systems (NeurIPS), volume 32.

Herbert A. David and Haikady N. Nagaraja. 2003. Order Statistics, 3rd edition. John Wiley & Sons.

Grégoire Déletang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. 2024. Language modeling is compression. In International Conference on Learning Representations (ICLR).

Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to the Bootstrap. Chapman & Hall/CRC.

Sabri Eyuboglu, Maya Varma, Khaled Saab, Jean-Benoit Delbrouck, Christopher Lee-Messer, Jared Dunnmon, James Zou, and Christopher Ré. 2022. Domino: Discovering systematic errors with crossmodal embeddings. In International Conference on Learning Representations (ICLR).

Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. 2024. Promptbreeder: Self-referential self-improvement via prompt evolution. In International Conference on Machine Learning (ICML).

Daniel Grießhaber, Maximilian Kimmich, Johannes Maucher, and Ngoc Thang Vu. 2025. A toolbox for improving evolutionary prompt search. arXiv preprint arXiv:2511.05120.

Peter D. Grünwald. 2007. The Minimum Description Length Principle. MIT Press.

Wassily Hoeffding. 1963. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30.

Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He. 2024. Compression represents intelligence linearly. In Conference on Language Modeling (COLM).

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2024. DSPy: Compiling declarative language model calls into self-improving pipelines. In International Conference on Learning Representations (ICLR).

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems (NeurIPS), volume 35.

M. R. Leadbetter, Georg Lindgren, and Holger Rootzén. 1983. Extremes and Related Properties of Random Sequences and Processes. Springer-Verlag.

Guanbao Liang, Yuanchen Bei, Sheng Zhou, Yuheng Qin, Huan Zhou, Bingxin Jia, Bin Li, and Jiajun Bu. 2026. Generalizable self-evolving memory for automatic prompt optimization. arXiv preprint arXiv:2603.21520.

David Madras, Joshua Safyan, and Qiuyi (Richard) Zhang. 2025. Prompts generalize with low data: Non-vacuous generalization bounds for optimizing prompts with more informative priors. arXiv preprint arXiv:2510.08413. EXAIT Workshop, ICML 2025.

Nicolai Meinshausen and Peter Bühlmann. 2010. Stability selection. Journal of the Royal Statistical Society: Series B, 72(4):417–473.

Krista Opsahl-Ong, Michael J. Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab. 2024. Optimizing instructions and demonstrations for multi-stage language model programs. In Empirical Methods in Natural Language Processing (EMNLP).

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with “gradient descent” and beam search. arXiv preprint arXiv:2305.03495.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Associationfor Computational Linguistics (ACL).

Jorma Rissanen. 1978. Modeling by shortest data description. Automatica, 14(5):465–471.

Chengshuai Shi, Kun Yang, Zihan Chen, Jundong Li, Jing Yang, and Cong Shen. 2024. Efficient prompt optimization through the lens of best arm identification. In Advances in Neural Information Processing Systems (NeurIPS).

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35.

Chunlong Wu and Zhibo Qu. 2025. Reflectionenhanced meta-optimization integrating TextGradstyle prompt optimization with memory-driven selfevolution. arXiv preprint arXiv:2508.18749.

Shirley Wu, Parth Sarthi, Shiyu Zhao, Aaron Lee, Herumb Shandilya, Adrian Mladenic Grobelnik, Nurendra Choudhary, Eddie Huang, Karthik Subbian, Linjun Zhang, Diyi Yang, James Zou, and Jure Leskovec. 2025. Optimas: Optimizing compound AI systems with globally aligned local rewards. arXiv preprint arXiv:2507.03041.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. 2024. Large language models as optimizers. In International Conference on Learning Representations (ICLR).

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. 2024. TextGrad: Automatic “differentiation” via text. arXiv preprint arXiv:2406.07496.

Tom Zehle, Timo Heiss, Moritz Schlager, Matthias Assenmacher, and Matthias Feurer. 2025. promptolution: A unified, modular framework for prompt optimization. arXiv preprint arXiv:2512.02840.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2023. Large language models are human-level prompt engineers. In International Conference on Learning Representations (ICLR).

Viktor N. Zhuravlev, Artur R. Khairullin, Ernest A. Dyagin, Alena N. Sitkina, and Nikita I. Kulin. 2025. ReflectivePrompt: Reflective evolution in autoprompting algorithms. arXiv preprint arXiv:2508.18870.

## Appendix

## A Full Proof of Theorem 1

Theorem 1 (Restated). Suppose K independent strategies each produce a candidate with test accuracy $\mathrm { A c c } _ { t e s t } ( p _ { k } ) = \mathrm { A c c } ^ { * } - b _ { k } + \varepsilon _ { k } ,$ , where $b _ { k } \geq 0$ and $\varepsilon _ { k } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . A bootstrap selection procedure with B resamples selects the final prompt $p ^ { * }$ Then:

$$
\begin{array} { r l } { \mathbb { E } [ \operatorname { A c c } _ { \mathsf { t e s t } } ( p ^ { * } ) ] \geq \operatorname { A c c } ^ { * } - \operatorname* { m i n } _ { k } b _ { k } } & { } \\ { + \sigma \cdot \Phi ^ { - 1 } \big ( 1 - \frac { 1 } { K } \big ) } & { } \\ { - O \Big ( \sqrt { \frac { \ln K } { n _ { \mathrm { v a l } } \cdot B } } \Big ) . } \end{array}
$$

Proof. We prove the bound by analyzing each of the three terms in the generalization gap decomposition: a biasfloor mi ${ \mathrm { ~  ~ \sigma ~ } } _ { 1 k } b _ { k }$ , an exploration gain from order statistics, and a selection error controlled by bootstrap concentration.

## A.1 Bias Reduction via Strategy Diversity (Lemma 1)

Lemma 1. With K strategies having biases $b _ { 1 } , \dotsc , b _ { K } .$

$$
\operatorname* { m a x } _ { k } \mathbb { E } [ \operatorname { A c c } _ { t e s t } ( p _ { k } ) ] \geq \operatorname { A c c } ^ { * } - \operatorname* { m i n } _ { k } b _ { k } .
$$

Proof. Let $k ^ { * } = \arg \operatorname* { m i n } _ { k } b _ { k }$ . Then:

$$
\begin{array} { r l } { \underset { k } { \operatorname* { m a x } } \operatorname { \mathbb { E } } [ \operatorname { A c c } _ { \mathsf { t e s t } } ( p _ { k } ) ] \geq \operatorname { \mathbb { E } } [ \operatorname { A c c } _ { \mathsf { t e s t } } ( p _ { k ^ { * } } ) ] } & { } \\ { = \operatorname { A c c } ^ { * } - b _ { k ^ { * } } } & { } \\ { = \operatorname { A c c } ^ { * } - \underset { k } { \operatorname* { m i n } } b _ { k } . } \end{array}\tag{□}
$$

## A.2 Lemma 2: Exploration Gain via Order Statistics

Lemma 2. $\begin{array} { r l r } { \mathbb { E } [ \operatorname* { m a x } _ { k } \mathrm { A c c } _ { t e s t } ( p _ { k } ) ] } & { { } \ge } & { \mathrm { A c c } ^ { * } - } \end{array}$ $\mathrm { m i n } _ { k } b _ { k } + \sigma \cdot \Phi ^ { - 1 } ( 1 - 1 / K )$

Proof. Let $\begin{array} { r l r } { k ^ { * } } & { { } \ = \ } & { \arg \operatorname* { m i n } _ { k } b _ { k } } \end{array}$ Since $\begin{array} { r } { \operatorname* { m a x } _ { k } \operatorname { A c c } ( p _ { k } ) \geq \operatorname { A c c } ( p _ { k ^ { * } } ) = \operatorname { A c c } ^ { * } - b _ { k ^ { * } } + \varepsilon _ { k ^ { * } } } \end{array}$ for any threshold t:

$$
\begin{array} { l } { \operatorname* { P r } ( \underset { k } { \operatorname* { m a x } } \mathrm { A c c } ( p _ { k } ) \geq t ) \ ~ } \\ { \quad \ \geq \operatorname* { P r } ( \mathrm { A c c } ( p _ { k ^ { * } } ) \geq t ) \ ~ } \\ { \quad \ = 1 - \Phi \bigg ( \frac { t - \mathrm { A c c } ^ { * } + b _ { k ^ { * } } } { \sigma } \bigg ) . } \end{array}
$$

Setting this probability to $1 / K$ :

$$
\begin{array} { l } { \displaystyle { 1 - \Phi \bigg ( \frac { t - \mathrm { A c c } ^ { * } + b _ { k ^ { * } } } { \sigma } \bigg ) = \frac { 1 } { K } } } \\ { \displaystyle { \implies t = \mathrm { A c c } ^ { * } - b _ { k ^ { * } } } } \\ { \displaystyle { \qquad + \sigma \cdot \Phi ^ { - 1 } \bigg ( 1 - \frac { 1 } { K } \bigg ) . } } \end{array}
$$

The expected maximum is at least the $( 1 - 1 / K )$ quantile of the best strategy:

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \operatorname* { m a x } _ { k } \mathrm { A c c } ( p _ { k } ) ] \geq \mathrm { A c c } ^ { * } - \operatorname* { m i n } _ { k } b _ { k } } \\ { \displaystyle + \sigma \cdot \Phi ^ { - 1 } \biggl ( 1 - \frac { 1 } { K } \biggr ) . } \end{array}
$$

For the asymptotic case of K identical strategies (same $b ,$ same $\sigma ) .$ the classical result E[max of K i.i.d. $\mathcal { N } ( 0 , 1 ) ] ~ \sim ~ \sqrt { 2 \ln K }$ applies (Leadbetter et al., 1983; David and Nagaraja, 2003). □

Remark. The bound $\Phi ^ { - 1 } ( 1 { - } 1 / K )$ is conservative (a lower bound on the expected maximum), making the theorem statement safe. For $K { = } 4 3$ $\Phi ^ { - 1 } ( 3 / 4 ) \approx 0 . 6 7 4$

## A.3 Lemma 3: Selection Precision via Bootstrap

Lemma 3. Ifthe true best candidate ranksfirst in each bootstrap resample with probability $p _ { 1 } > 1 / 2 ,$ then:

$$
\begin{array} { r l r } & { } & { \mathrm { P r } { \left( w r o n g s e l e c t i o n \nu i a p l u r a l i t y \nu \nu o t e \right) } } \\ & { } & { \qquad \leq \mathrm { e x p } { \left( - 2 B { \left( p _ { 1 } - \frac { 1 } { 2 } \right) } ^ { 2 } \right) } . } \end{array}
$$

Proof. Let $W _ { b } ~ = ~ \mathcal { H }$ [true best wins resample b]. Then $W _ { 1 } , \dots , W _ { B }$ are i.i.d. Bernoull $\mathfrak { i } ( p _ { 1 } )$ . Plurality vote selects the true best if $\sum _ { b } W _ { b } \bar { > } B / 2$ . We bound the complementary event:

$$
\begin{array} { r } { \mathrm { P r } \left( \displaystyle \sum _ { b } W _ { b } \leq \frac { B } { 2 } \right) = \mathrm { P r } \left( \displaystyle \sum _ { b } W _ { b } - B p _ { 1 } \right. } \\ { \displaystyle \left. \leq - B \left( p _ { 1 } - \frac { 1 } { 2 } \right) \right) . } \end{array}\tag{8}
$$

By Hoeffding’s inequality (Hoeffding, 1963), for $t \doteq p _ { 1 } - 1 / 2 > 0 \colon$

$$
\operatorname* { P r } \left( \sum _ { b } W _ { b } - B p _ { 1 } \leq - B t \right) \leq \exp ( - 2 B t ^ { 2 } ) .
$$

Therefore $\mathrm { P r }$ (wrong selection) $\leq \exp ( - 2 B ( p _ { 1 } -$ $1 / 2 ) ^ { 2 } )$ □

Remark. The assumption $p _ { 1 } > 1 / 2$ requires that the true best candidate wins more often than any other single candidate. This is plausible when the accuracy gap between the best and second-best candidates is non-trivial relative to the noise level.

## A.4 Combining the Lemmas

By Lemma 2, the population contains a candidate with expected accuracy at least $\mathrm { A c c ^ { * } ~ - }$ min<sub>k</sub> $b _ { k } + \sigma \cdot \Phi ^ { - 1 } ( 1 { - } 1 / K )$ . By Lemma 3, bootstrap selection identifies this candidate with probability at least 1 − exp(− $\cdot 2 B ( p _ { 1 } - 1 / 2 ) ^ { 2 } )$ . The selection error translates to an accuracy loss of at most $O ( \sqrt { \ln K / ( n _ { \mathrm { v a l } } \cdot B ) } )$ : the gap between the best and selected candidate on the test set is bounded by the estimation error on the validation set $( O ( 1 / { \sqrt { n _ { \mathrm { v a l } } } } ) )$ , divided by the confidence improvement from B resamples $( \sqrt { B } )$ , with a log K correction for multiple testing.

For GEPA (K=1, B=1): the exploration gain $\sigma \cdot \Phi ^ { - 1 } ( 1 { - } 1 / K )$ vanishes $( \Phi ^ { - 1 } ( 0 ) = - \infty$ , but with $K { = } 1$ there is no maximum to take), and the selection error is $O ( 1 / \sqrt { n _ { \mathrm { v a l } } } )$ □

Table 6: Term-by-term comparison of the generalization bound for GEPA (degenerate case, $K = 1 , B = 1 )$ and ESPO (default $K { = } 4 , B { = } 2 0 )$ .
<table><tr><td>Term</td><td>GEPA (K=1, ESPO (K=4, B=20) B=1)</td><td>Improvement</td></tr><tr><td>Bias</td><td> $b _ { 1 }$  mink  $b _ { k } \leq b _ { 1 }$ </td><td>Multiple strate- gies</td></tr><tr><td>Explor. Select.</td><td>0 (no bonus)  $\sigma { \cdot } \Phi ^ { - 1 } ( 3 / 4 ) \approx 0 { . } 6 7 \sigma$   $O ( 1 / \sqrt { n _ { \mathrm { v a l } } } )$   $O ( \sqrt { \ln { 4 } / ( n _ { \mathrm { v a l } } \cdot 2 0 ) } )$ </td><td>Order statistics ~3.8× tighter</td></tr></table>

## B Supporting Analysis: Error Pattern Coverage

Proposition 2 (Coupon collector bound). Let n training errors be distributed across $K ^ { * }$ patterns, with pattern k containing $n _ { k }$ errors (min<sub>k</sub> $n _ { k } \ \ge$ 1). An optimizer samples m errors uniformly per round. Then $T = \lceil n \cdot \ln ( K ^ { * } / \delta ) / ( m \cdot \operatorname* { m i n } _ { k } n _ { k } ) \rceil$ rounds are sufficient to observe all $K ^ { * }$ patterns with probability $\geq 1 - \delta .$

Proof. Let $p _ { k } = n _ { k } / n$ be the probability of sampling an error from pattern k in a single draw. In $\overleftarrow { T }$ rounds of m samples each, the probability that pattern k is never sampled is:

$$
\operatorname* { P r } ( \operatorname* { m i s s } k ) = ( 1 - p _ { k } ) ^ { m T } \leq \exp ( - p _ { k } \cdot m T ) .
$$

By union bound:

$$
\begin{array} { r l } {  { \operatorname* { P r } ( \mathrm { a n y ~ p a t t e r n ~ m i s s e d } ) } \quad } & { } \\ & { \leq \displaystyle \sum _ { k } \exp ( - p _ { k } \cdot m T ) } \\ & { \leq K ^ { * } \cdot \exp ( - \operatorname* { m i n } _ { k } p _ { k } \cdot m T ) . } \end{array}
$$

Setting $K ^ { * } \cdot \exp ( - \operatorname* { m i n } _ { k } p _ { k } \cdot m T ) \leq \delta$ and solving:

$$
\begin{array} { r l } {  { m T \cdot \operatorname* { m i n } _ { k } p _ { k } \geq \ln ( K ^ { * } / \delta ) } } \\ & { \implies T \geq \frac { n \cdot \ln ( K ^ { * } / \delta ) } { m \cdot \operatorname* { m i n } _ { k } n _ { k } } . } \end{array}
$$

Practical implication. For typical values $( K ^ { * } { = } 5$ $\scriptstyle m = 3 , \ n = 2 0$ , min<sub>k</sub> $n _ { k } { = } 2 )$ , this gives $T ~ \geq ~ 1 5$ GEPA needs ∼15 reflection rounds to observe all error patterns with 95% probability. Full-batch diagnosis $( m { = } n )$ achieves complete coverage in $T { = } 1$ , directly reducing each strategy’s bias $b _ { k }$ in Theorem 1.

## C Implementation Details

## C.1 Error Clustering Prompt

The reflection LLM receives all error examples and is asked to identify 3–7 recurring failure patterns. For each pattern, it provides: (1) a short descriptive name, (2) the root cause, (3) which error indices belong to it, and (4) a suggested fix for the instruction. The output is a structured list used directly in Phase 2.

Concrete example (Tweet dataset). Given 16 training errors on sentiment classification, Phase 1 produces the following diagnosis:

• Pattern 1: Sports Commentary → Neutral (4 errors, 25%). Factual event reporting (scores, records) misclassified as positive due to achievement words like “wins” and “legend.” Root cause: instruction conflates objective reporting with subjective sentiment. Fix: “Factual reporting of events/stats is neutral regardless of achievement words.”

• Pattern 2: Rankings/Lists → Positive (4 errors, 25%). Implicit positive sentiment in rankings (“top 5 teams,” “earned a spot”) misclassified as neutral. Root cause: rankings inherently carry positive evaluation. Fix: “Rankings and ‘top lists express positive sentiment even when stated factually.”

• Pattern 3: Sarcasm/Irony (3 errors, 19%). Sarcastic praise misclassified as neutral. Root cause: vague instruction “account for sarcasm” does not specify the sentiment mapping. Fix: “Sarcasm expressing criticism = negative; ironic praise criticizing a subject is negative.”

• Pattern 4: Humorous Positive (3 errors, 19%). Playful complaints with joy indicators (“LOL,” “!!”) misclassified as negative. Fix: “Playful complaints with humor/joy indicators signal positive.”

• Pattern 5: Entertainment Commentary (2 errors, 12%). Quality statements about music/shows without personal emotion misclassified as positive. Fix: “Content quality commentary without personal emotion is neutral.”

This structured output replaces GEPA’s unstructured reflection on 3 random errors, enabling each proposal strategy in Phase 2 to address root causes rather than symptoms.

## C.2 Four Proposal Strategy Templates

Each strategy receives the current prompt, the structured diagnosis $\varphi ,$ and strategy-specific instructions:

• Diagnostic Revision $( S _ { 1 } ) \colon$ “Revise the instruction to address each error pattern. Focus on root causes, not symptoms.”

• Consolidation $( S _ { 2 } ) \colon$ “Rewrite the instruction to be more concise without losing information. Merge redundant rules and tighten language. The result must not be longer than the original.”

• Ablation $( S _ { 3 } ) \colon$ “Identify rules that may be overtriggered, causing false positives. Soften or remove them.”

• Factual Injection $( S _ { 4 } ) \colon$ “Extract domainspecific knowledge from the error examples and add it to the instruction as factual context.”

## C.3 Bootstrap Selection

We draw $B { = } 2 0$ bootstrap resamples of the validation set (sampling $n _ { \mathrm { v a l } }$ examples with replacement). For each resample, we evaluate all candidates and record the winner. The final selection is the candidate with the most wins, with ties broken by prompt length (shorter preferred).

## C.4 Hyperparameter Settings and Justification

Table 7: ESPO hyperparameter settings used for all experiments.
<table><tr><td>Parameter</td><td>Value</td><td>Description</td></tr><tr><td>K</td><td>4</td><td>Number of proposal strategies</td></tr><tr><td>N</td><td>10</td><td>Max candidates in population</td></tr><tr><td>B</td><td>20</td><td>Bootstrap resamples</td></tr><tr><td>Iters.</td><td>2</td><td>Refinement iters after seed</td></tr><tr><td>Stud. temp</td><td>0.0</td><td>Deterministic inference</td></tr><tr><td>Refl. temp</td><td>0.7</td><td>Diverse generation</td></tr><tr><td> $n _ { \mathrm { t r a i n } }$ </td><td>70</td><td>Training examples</td></tr><tr><td> $n _ { \mathrm { v a l } }$ </td><td>30</td><td>Validation examples</td></tr></table>

Justification. $K { = } 4$ corresponds to four qualitatively distinct error-fixing strategies (diagnostic, consolidation, ablation, factual); adding more strategies (e.g., K=5, 6) would require inventing additional distinct inductive biases without clear motivation. B=20 is grounded in Lemma 3: for $p _ { 1 } { = } 0 . 7$ (true best wins 70% of resamples),

Pr(wrong selection) $\leq \exp ( - 2 \cdot 2 0 \cdot 0 . 0 4 ) = 0 . 2 0 ;$ increasing to B=30 yields only marginal improvement (0.09). N=10 balances population diversity against computational cost: each candidate requires $n _ { \mathrm { v a l } } { = } 3 0$ evaluations per bootstrap resample. These hyperparameters were set once before experimentation and not tuned per dataset.

## C.5 Hyperparameter Sensitivity Analysis (Extended)

Table 5 in the main text presents the sensitivity analysis for K, B, and m. Here we provide additional detail.

B sensitivity (extended). The main table omits B=5 for compactness. The full results are: $B { = } 1 { : }$ 73.80% (ablation, 1,214c), B=5: 74.00% (ablation, 1,214c), B=10: 74.00% (cross-pollination, 958c), B=20: 74.80% (cross-pollination, 774c), B=30: 74.80% (cross-pollination, 774c). With $B \leq 5 .$ bootstrap selects the ablation candidate (higher point estimate on validation, longer prompt); with $B \geq 1 0 .$ , it consistently selects the cross-pollination candidate (more robust, shorter prompt, and higher test accuracy), illustrating bootstrap’s ability to identify the truly best candidate by averaging over data perturbations.

m sensitivity and error coverage. The diagnosis batch size m controls what fraction of training errors are analyzed. Using the Coupon Collector bound (Proposition 2), m=3 covers ∼19% of error patterns per round, m=8 covers ∼50%, $m { = } 1 5$ covers ${ \sim } 9 4 \%$ , and $m { = } \mathbf { a l l }$ covers 100%. Full-batch diagnosis (m=all) achieves the highest accuracy (74.78%) with the shortest prompt (774c). Partial sampling produces 31–48% longer prompts (1,015–1,148c) at lower accuracy (69.8–72.6%), as incomplete error observation generates prompts that accumulate redundant rules addressing symptoms rather than root causes.

## D Additional Experimental Results

## D.1 Strategy Diversity

Table 8 reports the seed-phase validation accuracy of each proposal strategy across the seven benchmarks. No single strategy dominates: diagnostic revision wins on Tweet, consolidation on HotpotQA, ablation ties with diagnostic on MMLU and with consolidation/factual on GSM8K, and factual injection wins on HoVer and PUPA. This empirically supports the independent-bias assumption underlying the exploration-gain term in Theorem 1.

Table 8: Seed-phase validation accuracy (%) per strategy $( n _ { \mathrm { v a l } } { = } 3 0 )$ . No single strategy dominates, confirming the independent-bias assumption.
<table><tr><td>Dataset</td><td>Diagnosis</td><td></td><td></td><td>Consolid Ablation Factual Inj.</td><td>Winner</td></tr><tr><td>Tweet</td><td>80.00</td><td>73.33</td><td>73.33</td><td>76.67</td><td>Diag.</td></tr><tr><td>MMLU</td><td>83.33</td><td>80.00</td><td>83.33</td><td>76.67</td><td>D/A</td></tr><tr><td>GSM8K</td><td>93.33</td><td>96.67</td><td>96.67</td><td>96.67</td><td>C/A/F</td></tr><tr><td>HotpotQA</td><td>50.00</td><td>53.33</td><td>50.00</td><td>50.00</td><td>Cons.</td></tr><tr><td>ScoNe</td><td>80.00</td><td>80.00</td><td>80.00</td><td>80.00</td><td>Tie</td></tr><tr><td>HoVer</td><td>50.00</td><td>66.67</td><td>60.00</td><td>70.00</td><td>Fact.</td></tr><tr><td>PUPA</td><td>63.65</td><td>63.06</td><td>62.05</td><td>65.84</td><td>Fact.</td></tr></table>

Table 9: Inference latency (seconds per example). Lower is better. Mode: P = predict, CoT = chain-ofthought.
<table><tr><td>Dataset</td><td>Def.</td><td>Boots.</td><td>COPRO</td><td>MIPROv2</td><td>GEPA</td><td>ESPO</td><td>Mode</td></tr><tr><td>Tweet</td><td>1.97s</td><td>2.24s</td><td>2.13s</td><td>2.20s</td><td>1.93s</td><td>1.93s</td><td>P</td></tr><tr><td>MMLU</td><td>1.16s</td><td>2.10s</td><td>2.04s</td><td>2.26s</td><td>1.16s</td><td>1.15s</td><td>P</td></tr><tr><td>GSM8K</td><td>5.02s</td><td>5.09s</td><td>4.92s</td><td>5.00s</td><td>5.02s</td><td>4.96s</td><td>CoT</td></tr><tr><td>HotpotQA</td><td>5.58s</td><td>5.69s</td><td>5.67s</td><td>5.70s</td><td>5.67s</td><td>5.35s</td><td>CoT</td></tr><tr><td>ScoNe</td><td>6.67s</td><td>6.89s</td><td>7.10s</td><td>6.97s</td><td>7.23s</td><td>6.59s</td><td>CoT</td></tr><tr><td>HoVer</td><td>7.11s</td><td>17.64s</td><td>7.72s</td><td>7.89s</td><td>9.85s</td><td>8.38s</td><td>CoT</td></tr><tr><td>PUPA</td><td>9.66s</td><td>8.91s</td><td>9.41s</td><td>8.59s</td><td>8.81s</td><td>8.18s</td><td>P</td></tr></table>

## D.2 Inference Latency and Training Cost

Table 9 reports per-example inference latency at test time, and Table 10 reports the optimizationtime cost in tokens. ESPO matches or improves GEPA’s latency on every benchmark, with the largest reductions on chain-of-thought tasks where shorter prompts induce more concise reasoning chains. Reflection token usage during optimization is ∼39% of GEPA’s because structured diagnosis produces information-dense inputs.

Table 10: Optimization cost per dataset for ESPO (Claude Sonnet 4.5 pricing: \$3/M input + \$15/M output). Wall-clock times are measured after parallelizing candidate evaluation and strategy proposal.
<table><tr><td>Dataset</td><td>Total Tokens</td><td>Cost</td><td>Wall Clock</td><td>Reflect Tokens</td></tr><tr><td>Tweet</td><td>~752K</td><td>$9.99</td><td>10m</td><td>~36K</td></tr><tr><td>MMLU</td><td>~887K</td><td>$10.85</td><td>19m</td><td>~38K</td></tr><tr><td>GSM8K</td><td>~3.0M</td><td>$21.58</td><td>18m</td><td>~46K</td></tr><tr><td>HotpotQA</td><td>~3.0M</td><td>$20.80</td><td>28m</td><td>~46K</td></tr><tr><td>ScoNe</td><td>~3.8M</td><td>$25.66</td><td>31m</td><td>~33K</td></tr><tr><td>HoVer</td><td>~1.6M</td><td>$14.90</td><td>45m</td><td>~40K</td></tr><tr><td>PUPA</td><td>~3.4M</td><td>$23.20</td><td>2h 13m</td><td>~42K</td></tr></table>

Additional notes. The cross-model gains in §4.3 hold dataset-by-dataset, not just on average: on Qwen3, ESPO is the unique winner on 6/7 datasets (HoVer is the only loss); on Haiku, ESPO wins 6/7 (PUPA is the only loss). The Qwen3 GSM8K result (Default 15.00% → ESPO 91.40%, +56.00 pp over GEPA) is the largest single-cell improvement we observe; the structured diagnosis identifies the output-format mismatch silently throttling Qwen3 in one round, while GEPA’s rule-accumulation only partially addresses it across many. ESPO prompts average 30–60% shorter than GEPA across all four students, mirroring the same-student finding.

## D.3 Per-Seed Variance Grid

Table 1 in the main paper reports single-run cell values with ±std over 3 additional seeds attached; Table 11 below gives the full 3-seed grid with each cell’s actual 3-seed mean and std. The 3- seed ESPO−GEPA Avg gap is 3.81 pp, close to Table 1’s 3.76 pp headline and larger than either method’s Avg std (1.16 / 1.81), placing the gap outside the noise band. The per-dataset gaps on HotpotQA (+0.79 under 3-seed means, +0.80 singlerun) and PUPA (+1.29 / +1.20) fall within one std and are reported as ties. ESPO’s std is systematically smaller than GEPA’s (Avg 1.16 vs. 1.81), consistent with the bootstrap-stability prediction (Lemma 3).

## D.4 Strong-Initialization Comparison

The main-paper experiments (§4) intentionally start from a weak seed to test recovery. We chose a minimal unified seed because several baselines (COPRO, MIPROv2) cannot build a prompt from scratch—they only rewrite/augment an existing instruction, so a shared starting point is required for a fair comparison. To probe whether ESPO’s gains depend on the seed being weak, we rerun every method on every dataset with a strong task-tuned seed—one that covers task definition, output format, and 1–2 known-pitfall rules (representative of a competent practitioner’s draft after briefly studying the task).

Under a strong seed (Table 12), ESPO averages 4.79 pp above GEPA—a wider lead than the 3.76 pp under the weak seed. Under strong initialization, GEPA’s mini-batch reflection keeps appending marginal rules to an already-good prompt (length grows, accuracy stagnates), while ESPO’s full-error diagnosis targets the small remaining structural gap, and Diversify + Select suppress appended noise. This confirms ESPO is a general prompt optimizer, not limited to repairing weak prompts.

Table 11: Per-seed variance: test accuracy (%; mean ± std over 3 independent optimization seeds) on all seven benchmarks with Claude Sonnet 4.5 as student. Best per row in bold. This is the full grid that Table 1 in the main paper condenses.
<table><tr><td>Dataset</td><td>Default</td><td>Bootstrap</td><td>COPRO</td><td>MIPROv2</td><td>GEPA</td><td>ESPO (Ours)</td></tr><tr><td>Tweet</td><td> $3 7 . 5 7 \pm 0 . 2 0$ </td><td> $3 7 . 9 1 \pm 0 . 2 0$ </td><td> $6 6 . 5 5 \pm 1 . 8 0$ </td><td> $7 4 . 6 2 \pm 1 . 5 0$ </td><td> $6 8 . 7 2 \pm 2 . 2 0$ </td><td> ${ \bf 7 4 . 7 9 \pm 1 . 1 8 }$ </td></tr><tr><td>MMLU</td><td> $2 3 . 0 5 \pm 0 . 3 0$ </td><td> $2 2 . 9 9 \pm 0 . 3 0$ </td><td> $8 9 . 3 4 \pm 1 . 2 0$ </td><td> $8 6 . 6 5 \pm 1 . 4 0$ </td><td> $8 9 . 2 2 \pm 1 . 5 0$ </td><td> ${ \pm } 0 4 . 2 8 \pm 0 . 8 4$ </td></tr><tr><td>GSM8K</td><td> $3 2 . 5 1 \pm 0 . 4 0$ </td><td> $3 6 . 8 9 \pm 0 . 4 0$ </td><td> $3 6 . 6 8 \pm 1 . 0 0$ </td><td> $3 7 . 3 7 \pm 1 . 0 0$ </td><td> $9 6 . 5 7 \pm 0 . 5 0$ </td><td> ${ \pm 0 . 6 5 \pm 0 . 4 0 }$ </td></tr><tr><td>HotpotQA</td><td> $2 5 . 6 0 \pm 0 . 3 0$ </td><td> $1 4 . 4 3 \pm 0 . 3 0$ </td><td> $1 4 . 3 3 \pm 0 . 9 0$ </td><td> $1 4 . 2 3 \pm 0 . 9 0$ </td><td> $4 3 . 9 8 \pm 1 . 8 0$ </td><td> $4 4 . 7 7 \pm 1 . 4 0$ </td></tr><tr><td>ScoNe</td><td> $4 8 . 9 5 \pm 0 . 3 0$ </td><td> $4 6 . 1 2 \pm 0 . 3 0$ </td><td> $4 5 . 9 9 \pm 1 . 5 0$ </td><td> $4 6 . 5 8 \pm 1 . 5 0$ </td><td> $8 4 . 5 2 \pm 2 . 0 0$ </td><td> ${ \bf 8 9 . 1 0 \pm 1 . 2 0 }$ </td></tr><tr><td>HoVer</td><td> $4 7 . 8 8 \pm 0 . 3 0$ </td><td> $4 8 . 0 2 \pm 0 . 3 0$ </td><td> $4 8 . 1 5 \pm 1 . 6 0$ </td><td> $4 7 . 8 1 \pm 1 . 6 0$ </td><td> $5 3 . 6 0 \pm 2 . 4 0$ </td><td> ${ \bf 6 2 . 4 0 \pm 1 . 5 0 }$ </td></tr><tr><td>PUPA</td><td> $1 . 5 8 \pm 0 . 1 9$ </td><td> $1 . 6 1 \pm 0 . 1 2$ </td><td> $5 4 . 2 1 \pm 2 . 0 0$ </td><td> $1 . 7 2 \pm 0 . 1 2$ </td><td> $5 8 . 5 7 \pm 2 . 3 0$ </td><td> ${ \pm 9 . 8 6 \pm 1 . 6 0 }$ </td></tr><tr><td>Avg</td><td> $3 1 . 0 2 \pm 0 . 2 8$ </td><td> $2 9 . 7 1 \pm 0 . 2 7$ </td><td> $5 0 . 7 5 \pm 1 . 4 3$ </td><td> $4 4 . 1 4 \pm 1 . 1 5$ </td><td> $7 0 . 7 4 \pm 1 . 8 1$ </td><td> $7 4 . 5 5 \pm 1 . 1 6$ </td></tr></table>

Table 12: Strong-initialization comparison: test accuracy (%) when all methods start from a task-tuned instruction (task definition, output format, and 1–2 known-pitfall rules). Same 7 benchmarks, same student (Claude Sonnet 4.5), same reflection LLM as Table 1; only the seed prompt changes. Best per row in bold.
<table><tr><td>Dataset</td><td>Default</td><td>Bootstrap</td><td>COPRO</td><td>MIPROv2</td><td>GEPA</td><td>ESPO (Ours)</td></tr><tr><td>Tweet</td><td>65.20</td><td>65.40</td><td>71.80</td><td>76.60</td><td>72.40</td><td>74.60</td></tr><tr><td>MMLU</td><td>82.40</td><td>82.20</td><td>90.20</td><td>88.40</td><td>90.20</td><td>94.52</td></tr><tr><td>GSM8K</td><td>88.60</td><td>89.00</td><td>89.20</td><td>89.40</td><td>96.80</td><td>97.20</td></tr><tr><td>HotpotQA</td><td>32.60</td><td>32.40</td><td>32.80</td><td>32.60</td><td>45.60</td><td>47.80</td></tr><tr><td>ScoNe</td><td>66.40</td><td>66.20</td><td>66.80</td><td>66.60</td><td>87.20</td><td>91.80</td></tr><tr><td>HoVer</td><td>54.80</td><td>54.60</td><td>55.00</td><td>54.80</td><td>57.40</td><td>65.40</td></tr><tr><td>PUPA</td><td>48.20</td><td>48.00</td><td>55.60</td><td>48.20</td><td>48.20</td><td>60.00</td></tr><tr><td>Avg</td><td>62.60</td><td>62.54</td><td>65.91</td><td>65.23</td><td>71.11</td><td>75.90</td></tr></table>

Table 13: Per-component ablation atop GEPA: adding one ESPO component at a time. Averaged across the 7 main benchmarks; single-seed values (component variants were not rerun over 3 seeds). Avg Len is character count of the final prompt.
<table><tr><td>Configuration</td><td>Avg Acc (%)</td><td>Avg Len (chars)</td></tr><tr><td>GEPA</td><td>70.91</td><td>1,878</td></tr><tr><td>GEPA + multi-strategy</td><td>72.39</td><td>1,924</td></tr><tr><td>GEPA + bootstrap select</td><td>72.50</td><td>1,816</td></tr><tr><td>GEPA + full-error diagnose</td><td>73.24</td><td>1,843</td></tr><tr><td>ESPO (all three)</td><td>74.67</td><td>1,004</td></tr></table>

## D.5 Per-Component Ablation atop GEPA

Table 4 in the main paper ablates ESPO components on Tweet only. Table 13 gives the complementary view: starting from GEPA and layering each ESPO component individually, averaged across all 7 main benchmarks.

The length metric is more revealing: any single addition to GEPA leaves prompt length at, or slightly above, the GEPA baseline (multi-strategy alone grows to 1,924 chars because diversified proposals without a diagnosis constraint accumulate redundant rules). Only the full three-way combination simultaneously raises accuracy and cuts length by −47%—evidence that solving prompt bloat requires all three components together.

Multi-strategy alone contributes +1.48 pp, bootstrap-select alone +1.59 pp, and full-error diagnose alone +2.33 pp. Naive linear addition would predict ≈ 5.40 pp, whereas the full ESPO stack delivers 3.76 pp—component gains partially overlap (e.g., full-error diagnose and bootstrap both mitigate overfitting on the small validation set).

## D.6 Sensitivity to Reflection-Model Choice

Section 4 fixes the reflection LLM to Claude Sonnet 4.5. To test whether ESPO’s gains depend on a strong reflection model, we swap it to the open-source Qwen2.5-32B-Instruct and rerun every LLM-proposal-based method (COPRO, MIPROv2, GEPA, ESPO); results in Table 14. Default and Bootstrap do not use an LLM proposer, so their numbers are identical to Table 1 and are listed only to align table structure.

Absolute accuracies drop for every LLMproposal-based method under the weaker reflection model (Table 14), consistent with what GEPA reports: candidate quality depends on the proposer’s semantic capability. The key observation is that the ESPO–GEPA gap is preserved: Avg lead 4.29 pp under Qwen2.5-32B, comparable to (and slightly larger than) the 3.76 pp headline under Sonnet 4.5 in Table 1. The ordering $\mathrm { E S P O } > \mathrm { G E P A } > \mathrm { C O P R O }$ / MIPROv2 holds under both prompters, indicating that ESPO’s structural gains come from the three-phase design rather than from a single strong prompter.

Table 14: Sensitivity to reflection-model choice: test accuracy (%) when the reflection / prompter LLM is swapped from Claude Sonnet 4.5 (Table 1) to the open-source Qwen2.5-32B-Instruct, keeping the student (Sonnet 4.5) and all other settings unchanged. Default and Bootstrap have no LLM proposer and are shown only to align table structure. Best per row in bold.
<table><tr><td>Dataset</td><td>Default</td><td>Bootstrap</td><td>COPRO</td><td>MIPROv2</td><td>GEPA</td><td>ESPO (Ours)</td></tr><tr><td>Tweet</td><td>37.60</td><td>37.80</td><td>61.40</td><td>68.60</td><td>63.20</td><td>68.80</td></tr><tr><td>MMLU</td><td>23.20</td><td>23.00</td><td>82.80</td><td>80.20</td><td>82.60</td><td>86.20</td></tr><tr><td>GSM8K</td><td>32.60</td><td>36.80</td><td>33.20</td><td>33.40</td><td>86.40</td><td>88.80</td></tr><tr><td>HotpotQA</td><td>25.80</td><td>14.40</td><td>12.80</td><td>12.60</td><td>38.20</td><td>39.60</td></tr><tr><td>ScoNe</td><td>49.00</td><td>46.00</td><td>40.60</td><td>40.80</td><td>74.60</td><td>81.40</td></tr><tr><td>HoVer</td><td>48.00</td><td>48.00</td><td>43.60</td><td>43.20</td><td>48.20</td><td>54.60</td></tr><tr><td>PUPA</td><td>1.61</td><td>1.53</td><td>47.40</td><td>1.40</td><td>51.40</td><td>55.20</td></tr><tr><td>Avg</td><td>31.12</td><td>29.65</td><td>45.97</td><td>40.03</td><td>63.51</td><td>67.80</td></tr></table>

## D.7 Open-Ended Generation Pilot: XSum

The main benchmarks are verifiable (classification / exact-match / F1). To probe whether ESPO transfers to open-ended generation—where prompt bloat matters most—we run a pilot on XSum summarization. Judge model: Claude Sonnet 4.5. Protocol: 200 test articles; ESPO and GEPA are optimized on identical train/val splits from XSum; at test time, each summary pair is judged pairwise with randomized output order and method identity hidden from the judge. Results: ESPO wins 58.0%, ties 22.0%, loses 20.0% against GEPA. Final prompt length is 720 chars (ESPO) vs. 2,180 (GEPA), a 67.0% reduction. This is a preliminary sanity check; a full open-ended benchmark suite (long-form QA, code generation, dialogue) is future work. Because ESPO’s Diagnose stage currently assumes ground-truth error labels, extending diagnosis to judge-based signals is a natural next step, listed as a limitation.

## D.8 Cluster Stability

Section 3.2 describes clustering as “implicit” (LLM-based). We add two direct evaluations of clustering quality:

• Test–retest agreement. We invoke the reflection LLM three times independently on the same error set and, for each error, check whether it is assigned to semantically equivalent clusters across the three calls.

• Human-verified purity. The authors randomly sample 50 error–cluster assignments across datasets and independently review whether each error sits in a semantically coherent cluster.

Results: per-error agreement across 3 independent LLM calls is 82.4%; human-verified cluster purity on the 50 sampled errors is 88.0%. These indicate the LLM-based clustering is stable and semantically coherent enough for the Diagnose stage; residual noise is absorbed by the downstream Diversify + Select stages.

## D.9 Strategy Diversity Quantification

Assumption (A1) in Theorem 1 idealizes the four proposal strategies as independent, whereas in practice they share the same reflection LLM. We quantify the actual departure from independence empirically: for each pair of strategy-generated candidate prompts, we score both over 500 held-out examples to obtain two binary correctness vectors, and compute (i) Jaccard similarity (both-correct / atleast-one-correct)—overlap of correctly-answered items; (ii) Pearson correlation of the two correctness vectors—overall synchrony. Averaged across all candidate pairs:

<table><tr><td>Metric (averaged over strategy pairs)</td><td>Value</td></tr><tr><td>Pairwise Jaccard similarity</td><td>0.62</td></tr><tr><td>Pairwise Pearson correlation</td><td>0.48</td></tr></table>

A value of 1 means strategies are fully redundant; 0 means strictly independent. The observed values sit in between—strategy outputs are partially decorrelated, so candidates from different strategies exhibit substantive diversity even when generated by the same LLM. This justifies treating (A1) as an idealization rather than a literal claim: the exploration-gain term in Theorem 1 is expected to shrink by a factor of $\sqrt { 1 - \rho }$ under correlation $\rho ,$ and the observed correlation is bounded well below 1.

## E Ethical Considerations

## E.1 Use of LLMs.

ESPO uses commercial and open-weight large language models (Claude Sonnet 4.5 as reflection LLM and student model in the main experiments; Gemma 3 12B, Mistral 14B, Qwen3 32B, and Claude Haiku 4.5 as additional students). Inference costs and emissions scale with the number of optimization rounds and bootstrap evaluations; we report token costs in Table 10 so that practitioners can estimate environmental impact and budget.

## E.2 Bias propagation.

Prompt optimization that drives accuracy up against a static benchmark may inherit or amplify any biases present in (1) the training/validation examples used for diagnosis, (2) the reflection LLM’s clustering of errors, and (3) the metrics used for selection. ESPO’s structured diagnosis surfaces error categories explicitly, which can aid auditing, but it does not guarantee that bias-related failure modes will be identified or corrected. We recommend that downstream users evaluate optimized prompts on bias-relevant slices before deployment, especially for socially sensitive applications.

## E.3 Benchmark contamination.

Several of the public benchmarks we use (Tweet, MMLU, GSM8K, HotpotQA, ScoNe, HoVer, PUPA) may overlap with LLM pre-training data. Our experimental design holds this contamination risk constant across all optimizers (Default, Bootstrap, COPRO, MIPROv2, GEPA, ESPO), so the relative improvements we report are not driven by ESPO seeing more contaminated data. Absolute accuracy numbers should be interpreted with the standard contamination caveats for each benchmark.

## E.4 Dual-use considerations.

The techniques in ESPO are general promptoptimization tools and could be applied to objectives we did not study, including ones with negative downstream impact (e.g., optimizing prompts to bypass safety filters). We release code and prompts for the public NLP benchmarks studied in this paper; we do not release prompts or configurations targeted at evading safety mechanisms.