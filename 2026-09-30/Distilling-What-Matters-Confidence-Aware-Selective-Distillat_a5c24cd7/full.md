# Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models

Ayan Sengupta <sup>1</sup> Vaibhav Seth <sup>1</sup> Tanmoy Chakraborty <sup>1,2</sup>

<sup>1</sup> Department of Electrical Engineering, Indian Institute of Technology Delhi <sup>2</sup> Yardi School of Artificial Intelligence, Indian Institute of Technology Delhi

ayan.sengupta@ee.iitd.ac.in, vaibhavseth2283@gmail.com, tanchak@iitd.ac.in

î parmanu.lcs2.in

Abstract Knowledge Distillation (KD) trains a smaller-capacity student model to imitate a larger-capacity teacher model by matching output distributions, implicitly assuming the teacher to be a reliable oracle. In large language models (LLMs), this assumption often fails: teacher predictions can exhibit high entropy and hallucinations, causing standard KD to degrade well-calibrated student priors. We propose CaRE-KD, a confidence-gated distillation framework that replaces static objectives with uncertainty-adaptive optimization. CaRE-KD has two components: a token-level loss (CaRE-Divergence) that adaptively switches between Forward and Reverse KL divergence based on teacher–student confidence, and a batch-level epistemic rejection mechanism (Revival) that suppresses updates when the teacher is more uncertain than the student. We provide a gradient-level analysis showing how this dual-granularity design induces a conditional calibration mechanism that prior static divergences cannot reproduce. Empirically, across eight teacher–student pairs and eleven benchmarks spanning instruction following, chat alignment, code generation, and mathematical reasoning, CaRE-KD delivers consistent gains over strong baselines (Skewed-KL, α–β divergence). Highlights include up to +3.2 average ROUGE-L on instruction-following tasks, +2.1 pass@1 on MBPP, +1.7 accuracy on GSM8k, and +1.8 accuracy on CollegeMath over the strongest baseline, with consistent gains in LLM-as-a-judge factuality (up to +2.5 per task over Skewed-RKL). Revival further acts as a principled, loss-agnostic plug-in that systematically strengthens existing distillation objectives by filtering epistemically unreliable teacher supervision.

Keywords Knowledge distillation, eficient language models, on-policy distillation

## 1 Introduction and Prior Art

Knowledge Distillation (KD) is a standard paradigm for transferring knowledge from a large teacher model to a small student model by aligning their predictive distributions, most commonly via the Forward Kullback–Leible (KL) divergence [1, 2]. This formulation implicitly assumes that the teacher provides uniformly reliable supervision, encouraging the student to cover the teacher’s full probability mass. While efective when the teacher is well calibrated, this assumption becomes fragile in the regime of large language models (LLMs). Despite their strong average performance, modern LLMs frequently exhibit high-entropy predictions, internal inconsistency, and hallucinations [3, 4]. In such cases, minimizing Forward KL forces the student to allocate probability mass to unreliable long-tail predictions, a failure mode often described as zero-forcing [5], which degrades confident and correct student priors.

Prior work has attempted to mitigate this issue by modifying the distillation objective – mode-seeking alternatives such as Reverse KL [5], as well as static interpolations, including Skewed KL [6] and α–β divergence [7], reduce sensitivity to the teacher’s tail by globally biasing optimization toward precision. However, these approaches impose a fixed geometric preference throughout training. They fail to account for the inherently dynamic nature of autoregressive generation, in which a teacher may be highly informative for some tokens while uncertain or hallucinating for others.

Our empirical analysis (Figure 1) reveals a recurrent failure mode we call the Fidelity Trap [8, 9]: teacher supervision helps when the teacher is accurate (median +4%) but collapses to +0.5% when the teacher is uncertain about its own generation [10, 11], even though the student often has lower pre-distillation uncertainty than the teacher in such regimes. Standard KD nevertheless forces the student to match the teacher, synchronizing with unreliable supervision and overwriting correct priors.

![](images/3e976bb289eaa925d95eff0bb43739839d4efc1dc3183a0288898a3f24464d06.jpg)  
(a)

![](images/0eddbcbaa7214487f289765963fb3245d832697fbe133cd4b5fd43ece4b223c1.jpg)  
(b)

![](images/e24e95c8e21f6f9949e499b640e58c7c410f11e5fc4a0a763e4cbd851b37065f.jpg)  
(c)  
Figure 1. OpenLLaMA-3B student distilled from a 7B teacher on Dolly-15k with Skewed-RKL [6]. (a) Distillation fails to reduce student epistemic uncertainty when the teacher is uncertain (KS distance changes by only 0.4%). (b) Teacher supervision helps when the teacher is confident (median +4%) but barely helps when the teacher is uncertain (+0.5%). (c) Static objectives enforce agreement even with unreliable teachers.

We propose CaRE-KD, with two complementary components. CaRE-Divergence is a token-level objective that dynamically switches between Forward KL (mean-seeking, when the teacher is confident) and Reverse KL (modeseeking, when the teacher is uncertain). Revival is a batch-level rejection mechanism based on BALD [12] that suppresses updates when the teacher is epistemically less reliable than the student, preventing unlearning of correct priors.

Empirically, CaRE-KD achieves consistent improvements across eight teacher–student pairs and eleven benchmarks spanning instruction following, chat alignment, code generation, and mathematical reasoning (Tables 1, 5, 2, and 3). Concrete highlights include a +2.6 average ROUGE-L gain on GPT2-base over Forward KL, +3.2 on OPT-125M, +2.1 pass@1 on MBPP, +1.7 accuracy on GSM8k, and consistent gains in LLM-as-a-judge factuality (up to +2.5 per task over Skewed-RKL). Moreover, Revival acts as a general enhancer: when applied to existing divergence objectives, including Forward KL, Reverse KL, α–β divergence, and Skewed RKL, it systematically improves performance by selectively filtering unreliable supervision (Table 2). Together, these results demonstrate that selective adaptation of optimization geometry and selective rejection of supervision are both necessary for robust distillation from modern LLMs. Unlike prior work that modifies divergence globally or filters data at the dataset level, our framework adapts both the optimization geometry at the token level and supervision at the sequence level during training.

Our contributions can be summarized as follows:

• We introduce CaRE-Divergence, a confidence-gated distillation objective that dynamically adapts divergence geometry at the token level.

• We propose Revival, a sequence-level epistemic rejection mechanism that filters distillation updates when the teacher is epistemically less reliable than the student, preventing student calibration from collapsing under unreliable supervision.

• We provide a gradient-level analysis explaining how CaRE-Divergence induces conditional calibration and why no static divergence can reproduce its token-wise behavior, together with formal conditions under which entropy-based confidence is a valid proxy for BALD.

• We demonstrate consistent empirical gains across instruction following, chat alignment, code generation, and mathematical reasoning, complemented by an LLM-as-a-judge factuality evaluation that directly measures hallucination, and release a unified evaluation framework for reproducibility.<sup>1</sup>

Prior Art. Knowledge Distillation has evolved from logit matching and sequence-level objectives [1, 13] to generative distillation frameworks that address exposure bias via on-policy sampling [2, 14] and mitigate tail risks [5]. Despite these advances, robustly handling noisy or hallucinated supervision remains an open challenge [9, 15, 16]. While prior work has explored static reweighting of divergence geometry [6, 7], proxy-based uncertainty modeling [16], or dataset-level filtering [17], these approaches either introduce additional modeling overhead or completely discard potentially informative data. In contrast, our method directly modulates the optimization trajectory, leveraging divergence geometry and intrinsic uncertainty signals to adapt supervision at training time without auxiliary models or data pruning. A more detailed discussion of related work is provided in Appendix A.

## 2 Methodology

CaRE-KD is a dual-granularity distillation framework with two complementary mechanisms: (i) a token-level geometric adaptation, CaRE-Diverge (§2.1), that switches between Forward and Reverse KL using single-pass confidence, and (ii) a sequence-level rejection rule, Revival (§2.2), that suppresses updates when the teacher is epistemically less reliable than the student.

![](images/dbb07232a8608d41057363520a4a1115eb076a8a6b0fcd0f30e7026910102f74.jpg)

![](images/fb278ec77a0ab2a3aee738f0ed8ecaad66a1686a6b2be3a3c56ce739ae190eee.jpg)

Notation. Let V denote the vocabulary, and let $p ( \cdot \mid \mathbf { x } ) , q ( \cdot \mid \mathbf { x } ) \in \Delta ^ { | \mathcal { V } | - 1 }$ be the teacher and student next-token distributions (optionally tempered by T). We measure single-pass (aleatoric) confidence by the normalized-entropy score

Figure 2. Token-level geometric adaptation induced by $\mathrm { C a R E }$ Divergence. When the teacher is confident, the objective behaves as Forward KL (mean-seeking, recall-oriented); when the teacher is uncertain, it transitions to Reverse KL (mode-seeking, precision-oriented), suppressing tail noise.

$$
c ( p ) \triangleq 1 - H ( p ) / \log | \mathcal { V } | \in [ 0 , 1 ] ,\tag{1}
$$

where $H ( \cdot )$ is Shannon entropy; larger $c ( p )$ corresponds to a sharper distribution.

## 2.1 Token-Level Dynamics: CaRE-Divergence

Since the appropriate optimization geometry depends on the relative informativeness of the teacher (Figure 2), we replace fixed Forward KL with the confidence-gated combination

$$
\mathcal { L } _ { \mathrm { C A R E } } ( p , q ) = g \cdot \mathrm { K L } ( p \| q ) + ( 1 - g ) \cdot \mathrm { K L } ( q \| p ) ,\tag{2}
$$

where $g \in [ 0 , 1 ]$ is a per-token scalar gate driven by the confidence gap $\Delta c = c ( p ) - c ( q )$ . We consider two variants:

$$
g _ { \mathrm { h a r d } } = \mathbb { I } [ c ( p ) \geq c ( q ) ] , \qquad g _ { \mathrm { s o f t } } = \sigma ( c ( p ) - c ( q ) - m ) ,\tag{3}
$$

with $\sigma ( \cdot )$ the sigmoid and m a margin. When the teacher is confident $( g \to 1 )$ , CaRE-Divergence approaches mean-seeking $\mathrm { K L } ( p \Vert q )$ and maximizes support coverage; when the teacher is uncertain $( g \to 0 )$ , it approaches mode-seeking $\mathrm { K L } ( q \| p )$ , suppressing tail noise.

## 2.2 Batch-Level Rejection: Revival

Single-pass entropy cannot distinguish intrinsic ambiguity from hallucination-driven epistemic uncertainty. We there fore introduce Revival, a sequence-level rejection mechanism based on the BALD score [12] computed via Monte Carlo dropout [18]:

$$
{ \mathcal { T } } _ { \mathrm { B A L D } } ( \mathbf { x } ) = H [ \mathbb { E } _ { \omega } p ( y \mid \mathbf { x } , \omega ) ] - \mathbb { E } _ { \omega } [ H ( p ( y \mid \mathbf { x } , \omega ) ) ] ,\tag{4}
$$

which measures predictive disagreement across stochastic parameter realizations. Let $\mathcal { I } _ { \mathrm { T } }$ and $\mathcal { T } _ { \mathrm { { S } } }$ denote teacher and student BALD scores, respectively at a sequence; the rejection mask is

$$
M _ { \mathrm { R e v i v a l } } = \mathbb { I } [ \mathbb { Z } _ { \mathrm { T } } > Q _ { \tau } ( \mathbb { Z } _ { \mathrm { T } } ) \land \mathbb { Z } _ { \mathrm { S } } < Q _ { \tau } ( \mathbb { Z } _ { \mathrm { S } } ) ] ,\tag{5}
$$

where $Q _ { \tau } ( \cdot )$ is the running τ-quantile of historical BALD scores (computed separately for teacher and student). Using a relative percentile rather than a fixed threshold ensures the mask is scale-invariant under training-time drift in BALD magnitudes, and triggers only when the teacher is uncertain relative to a confident student, targeting unreliable supervision rather than uniformly discarding hard tokens. We optionally schedule τ across training (Appendix D).

For a batch B, the final per-batch objective is

$$
\mathcal { L } _ { \mathrm { C a R E - K D } } = ( 1 - M _ { \mathrm { R e v i v a l } } ) \cdot \frac { 1 } { | B | } \sum _ { { \bf x } \in B } \mathcal { L } _ { \mathrm { C A R E } } \big ( p ( { \bf x } ) , q ( { \bf x } ) \big ) .\tag{6}
$$

When $M _ { \mathrm { R e v i v a l } } = 1$ , updates are suppressed; otherwise the student optimizes the gated objective.

## 3 Theoretical Results

We provide a gradient-level analysis of CaRE-Divergence that explains how the gated objective difers from Skewed KL [6, 19] and $\alpha { - } \beta$ divergence [7], and we formalize when entropy-based confidence proxies BALD. Our results are mechanistic explanations of the empirical behavior, not standalone optimization guarantees. All proofs are in Appendix C.

## 3.1 Gradient analysis of CaRE-Divergence

Proposition 3.1 (Gradient decomposition of CaRE-Divergence). Let ${ \mathcal { L } } _ { \mathrm { F K L } } ( \theta ) = \mathrm { K L } ( p \| q _ { \theta } ) a n d { \mathcal { L } } _ { \mathrm { R K L } } ( \theta ) = \mathrm { K L } ( q _ { \theta } \| p )$ with pfixed in θ, and let $g = g ( \theta )$ be a diferentiable gate. Then

$$
\begin{array} { r l r } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { c a w E } } } & { } & { = \underbrace { \left[ g \nabla _ { \theta } \mathcal { L } _ { \mathrm { F K L } } + ( 1 - g ) \nabla _ { \theta } \mathcal { L } _ { \mathrm { R K L } } \right] } _ { L e a r i n g \ s i g n a l } + \underbrace { \left[ ( \mathcal { L } _ { \mathrm { F K L } } - \mathcal { L } _ { \mathrm { R K L } } ) \nabla _ { \theta } g \right] } _ { S e l e c t i o n \ : s i g n a l } . } \end{array}\tag{7}
$$

For the soft gate $g = \sigma ( c ( p ) - c ( q _ { \theta } ) - m ) , \nabla _ { \theta } g = - g ( 1 - g ) \nabla _ { \theta } c ( q _ { \theta } )$ , so the selection signal equals −g(1 − $g ) ( \mathcal { L } _ { \mathrm { F K L } } - \mathcal { L } _ { \mathrm { R K L } } ) \nabla _ { \theta } c ( q _ { \theta } )$

Interpretation. When the teacher is comparatively uncertain $( \mathcal { L } _ { \mathrm { F K L } } > \mathcal { L } _ { \mathrm { R K L } } )$ , the selection signal aligns with $+ \nabla _ { \boldsymbol { \theta } } c ( q _ { \boldsymbol { \theta } } )$ , explicitly pushing the student to increase its confidence. No fixed-skew or fixed-β objective produces such a confidence-dependent term. This selection term is present only for the diferentiable soft gate: the hard gate g<sub>hard</sub> is a detached step function with $\nabla _ { \boldsymbol { \theta } } g \equiv 0$ , so it retains only the token-level branch selection (the learning term with per-token $g _ { t } \in \{ 0 , 1 \}$ , which Proposition 3.2 shows no static divergence can reproduce), while soft gating additionally carries the confidence-pushing selection term. Branch selection alone yields the strongest peak performance, whereas the selection term confers robustness (Section C.1). When $M _ { \mathrm { R e v i v a l } } = 1 , \nabla _ { \theta } \mathcal { L } _ { \mathrm { C a R E - K D } } = 0$ , so the divergence to any reference $p ^ { \star }$ is unchanged – Revival provides worst-case robustness against unreliable supervision (proof in Appendix C).

## 3.2 Strict separation from SRKL and $\alpha \mathrm { - } \beta$ divergence

Proposition 3.2 (No global skew or $\beta$ matches CaRE-Divergence). In the student-dominant regime $( c ( q ) > c ( p )$ $g = 0 )$ , the per-token CaRE-Divergence gradient matches that of Skewed Reverse KL in the limit $\lambda \to 1$ . Globally, however, CaRE-Divergence assigns a token-varying $g _ { t } \in \{ 0 , 1 \}$ (and equivalently a token-varying $\beta _ { t } \in \{ 0 , 1 \}$ in the $\alpha { - } \beta f a m i l y )$ . No global λ for SRKL or global $\beta f o r \alpha { - } \beta$ divergence reproduces this token-wise behavior. (Full $\alpha { - } \beta$ derivation: Appendix C.)

## 3.3 Entropy-confidence proxies BALD in the sharp regime

Assumption 3.3 (Sharpness Hypothesis). LLMs trained with cross-entropy loss are typically overconfident [4, 20]: $H [ p ( y | \mathbf { x } , \omega ) ] \approx 0 { \mathrm { f o r } } \omega \sim p ( \omega )$

Theorem 3.4 (Conditional equivalence of confidence proxies). Under Assumption 3.3 and the approximation $\hat { p }$ ≈ $\mathbb { E } _ { \omega } [ p ( y \mid \mathbf { x } , \omega ) ]$ ], maximizing c(ˆp) approximately minimizes $\mathcal { T } _ { \mathrm { B A L D } } ( \mathbf { x } )$ . Hence single-pass entropy is a computationally eficient proxy for BALD inside the inner loop.

## 3.4 Conditional bi-directional calibration

Within a fixed branch of the gated objective, both KL branches share the unique minimizer $q = p$ (under standard support conditions); convergence in a branch, therefore, drives student entropy toward $H ( p )$

Theorem 3.5 (Conditional bi-directional calibration). Suppose optimization under $\mathcal { L } _ { \mathrm { C a R E - K D } }$ stays in afixed branch regime and $q _ { \theta }  p .$ . Then $H ( q _ { \theta _ { t } } ) \to H ( p )$ . In particular, an overconfident student $( H ( q _ { \theta _ { 0 } } ) < H ( p ) )$ must increase entropy at some step, and an underconfident student $( H ( q _ { \theta _ { 0 } } ) > H ( p ) )$ must decrease entropy. CaRE-Divergence thus acts as a conditional calibration mechanism without requiring strict monotonicity.

## 4 Experimental Setup and Results

Following established protocols for evaluating distilled language models [6, 7, 19], we evaluate our framework across three modalities – instruction following, code generation, and mathematical reasoning.

Instruction Following. We consider two complementary evaluation regimes. (1) Standard benchmarks. Following Ko et al. [6], we perform distillation on the Dolly-15k dataset [21]. We evaluate four teacher–student model families: $\mathrm { ( i ) ~ G P T \mathrm { - } 2 _ { X L } \to G P T \mathrm { - } 2 _ { B a s e / L a r g e } }$ [22], (ii) OPT-2.7B → OPT-125M [23], (iii) Gemma-2-9B-Instruct → Gemma-2- 2B-Instruct [24] and (iv) OpenLLaMA-7B → OpenLLaMA-3B [25]. Evaluation is conducted on four task-agnostic instruction-following benchmarks: Dolly Evaluation [21], Self-Instruct [26], Super-Natural Instructions [27], and Vicuna Instructions [27]. (2) Large-scale alignment. To assess performance on modern chat-aligned models, we follow Ko et al. [19] and distill a Qwen2-1.5B student from a Qwen2-7B teacher [28] using the UltraChat200k dataset [29]. The resulting models are evaluated on AlpacaEval [30], Evol-Instruct [31], and UltraFeedback [32].

Code Generation. For domain-specific coding tasks, we replicate the setup of Ko et al. [19]. We distill a DeepSeek-Coder-1.3B student from a DeepSeek-Coder-6.9B-Instruct teacher [33]. Training is performed on the WizardCoder dataset [34], and evaluation is conducted on HumanEval [35] and MBPP [36].

Mathematical Reasoning. To evaluate multi-step reasoning capabilities, we use the Qwen2.5-Math models [37]. A 1.5B student is distilled from a 7B-Instruct teacher using the MetaMathQA dataset [38]. We report zero-shot accuracy on GSM8K [39] and CollegeMath [40].

Metrics. For instruction-following and chat alignment we report ROUGE-L on the held-out evaluation set, following Ko et al. [6], together with an LLM-as-a-judge factuality score (0–100) computed by prompting GPT-5-Mini to grade outputs against the reference answer for hallucinations and factual consistency (system prompt in Appendix D.7). For code generation we report execution-based pass@1 on HumanEval and MBPP, and for mathematical reasoning we report exact-match accuracy on GSM8K and CollegeMath. Statistical significance for ROUGE-L improvements is assessed using paired t-tests across model pairs. Additional implementation details, hyperparameters, training protocols, and the LLM-judge system prompt are provided in Appendix D.

Table 1. Main instruction-following results under SGO training. Cells report ROUGE-L / LLM-as-a-judge factuality on Dolly, Self-Instruct, Super-Natural Instructions, and Vicuna. Across five teacher–student pairs, CaRE-Divergence achieves the best average ROUGE-L on three pairs and the best factuality on three pairs, with statistically significant gains over the second-best baseline $( t = 3 . 9 3 , p = 0 . 0 3 )$ . Bold denotes the best score per cell. Results are averaged over five seeds; detailed comparisons appear in Appendix E.
<table><tr><td rowspan="2">TEACHER → STUDENT</td><td rowspan="2">Loss</td><td colspan="4">BENCHMARKS</td><td rowspan="2">AvG</td></tr><tr><td>DOLLY</td><td>SELF-INST</td><td>SUPER-NAT</td><td>VICUNA</td></tr><tr><td rowspan="5">GPT2-XL (1.5B) → GPT2-BASE (0.1B)</td><td>FKL</td><td>23.4/21.2</td><td>10.7/15.6</td><td>19.0/16.4</td><td>15.0/13.5</td><td>17.0/16.7</td></tr><tr><td>RKL</td><td>24.7/20.7</td><td>11.6/16.4</td><td>19.2/16.2</td><td>16.6/14.6</td><td>18.0/17.0</td></tr><tr><td>SRKL</td><td>24.9/23.1</td><td>12.1/17.6</td><td>23.6/21.9</td><td>16.5/15.3</td><td>19.3/19.5</td></tr><tr><td>α-β DIv</td><td>23.9/23.1</td><td>11.5/16.4</td><td>21.9/18.3</td><td>16.1/15.0</td><td>18.4/18.2</td></tr><tr><td>CaRE-Div (Ours)</td><td>25.6/24.1</td><td>12.9/20.1</td><td>22.8/24.4</td><td>17.3/15.3</td><td>19.6/21.0</td></tr><tr><td rowspan="5">GPT2-XL (1.5B) → GPT2-LARGE (0.8B)</td><td>FKL</td><td>23.7/21.5</td><td>10.1/13.6</td><td>15.7/13.4</td><td>15.0/14.4</td><td>16.1/15.7</td></tr><tr><td>RKL</td><td>19.3/14.7</td><td>9.9/14.3</td><td>16.3/13.0</td><td>15.0/11.3</td><td>15.1/13.3</td></tr><tr><td>SRKL</td><td>24.1/20.5</td><td>11.1/15.0</td><td>19.9/17.0</td><td>16.4/15.3</td><td>17.9/16.9</td></tr><tr><td>α-β DIv</td><td>23.0/20.1</td><td>10.7/14.6</td><td>16.7/15.2</td><td>15.5/13.3</td><td>16.5/15.8</td></tr><tr><td>CaRE-Div (Ours)</td><td>25.4/25.5 22.4/20.7</td><td>11.6/16.7 9.6/14.1</td><td>19.4/18.0</td><td>16.0/16.1</td><td>18.1/19.1</td></tr><tr><td rowspan="5">OPT-2.7B → OPT-125M</td><td>FKL RKL</td><td>24.8/22.7</td><td>11.2/15.8</td><td>17.1/14.5</td><td>15.5/12.9</td><td>16.1/15.5</td></tr><tr><td></td><td>25.1/23.9</td><td></td><td>22.2/18.4</td><td>16.4/15.9</td><td>18.6/18.2</td></tr><tr><td>SRKL α-β DIv</td><td>24.9/23.6</td><td>11.6/19.3 11.2/16.5</td><td>22.5/19.5 21.7/18.3</td><td>16.8/18.1 16.7/13.9</td><td>19.0/20.2</td></tr><tr><td>CaRE-Div (Ours)</td><td>24.3/20.5</td><td>12.0/18.7</td><td>23.9/21.4</td><td>16.7/13.5</td><td>18.6/18.1</td></tr><tr><td>FKL</td><td>18.4/19.6</td><td>10.4/22.0</td><td>19.2/15.3</td><td>17.1/16.0</td><td>19.3/18.5</td></tr><tr><td rowspan="5">GEMMA-2-9B-IT → GEMMA-2-2B-IT</td><td>RKL</td><td>17.5/20.2</td><td>10.0/21.8</td><td>19.7/19.1</td><td>20.1/21.9</td><td>16.3/18.2 16.8/20.7</td></tr><tr><td></td><td>19.7/24.9</td><td></td><td></td><td></td><td></td></tr><tr><td>SRKL</td><td></td><td>10.8/20.8 10.7/21.6</td><td>16.5/18.3</td><td>18.9/18.2</td><td>16.5/20.6</td></tr><tr><td>α-β DIv</td><td>18.9/25.6</td><td></td><td>17.5/17.2</td><td>20.0/20.2</td><td>16.8/21.1</td></tr><tr><td>CaRE-Div (Ours)</td><td>17.0/23.4</td><td>11.5/22.0</td><td>19.4/19.9</td><td>20.3/21.0</td><td>17.1/21.6</td></tr><tr><td rowspan="5">OPENLLAMA2-7B → OPENLLAMA2-3B</td><td>FKL</td><td>24.3/32.2</td><td>17.6/25.7</td><td>30.5/31.3</td><td>16.9/22.3</td><td>22.3/27.9</td></tr><tr><td>RKL</td><td>29.1/40.0</td><td>20.7/31.7</td><td>35.7/39.4</td><td>20.0/27.8</td><td>26.4/34.7</td></tr><tr><td>SRKL</td><td>29.0/40.9</td><td>20.8/31.2</td><td>36.6/37.8</td><td>19.2/31.1</td><td>26.4/35.3</td></tr><tr><td>α-β DIv</td><td>28.3/38.1</td><td>19.3/29.7</td><td>33.5/35.2</td><td>18.6/29.9</td><td>24.9/33.2</td></tr><tr><td>CaRE-Div (Ours)</td><td>27.7/38.1</td><td>20.9/29.8</td><td>37.3/38.6</td><td>18.9/28.6</td><td>26.2/33.8</td></tr></table>

## 4.1 Results on instruction following tasks

We evaluate CaRE-KD under SGO training (student rollouts) and non-SGO (teacher forcing); each cell of Tables 1 and 5 reports ROUGE-L / LLM-as-a-judge factuality.

In the SGO setting (Table 1), CaRE-Divergence improves over Forward KL by +2.6 / +4.3 on GPT2-base, $+ 2 . 0 / + 3 . 4$ on GPT2-large, and +3.2 / +3.0 on OPT-125M (ROUGE-L / LLM-judge), and beats Skewed RKL by $+ 0 . 3 / + 1 . 5$ on GPT2-base. On OpenLLaMA-7B → 3B, it achieves the best ROUGE-L on Self-Instruct and Super-Natural Instructions and stays within 0.2 of the best baseline on average. Paired t-tests confirm that the average improvements of CaRE-Divergence over the per-row second-best baseline are statistically significant $( t = 3 . 9 3 ,$ $p = 0 . 0 3 )$ . Detailed comparisons against a comprehensive baseline suite (SFT, KD, SeqKD, ImitKD, MiniLLM, GKD, Distillm, ABKD) appear in Appendix E (Tables 6, 7, 8); CaRE-Divergence tops the average ROUGE-L on GPT2-base and OPT-125M.

The advantage is robustness, not a single large win. The accurate reading of Table 1 is that CaRE-KD is “never bad” rather than “always best”: under SGO it has the best average on four of the five pairs and the highest mean ROUGE-L averaged over all five (20.1 vs. 19.8 for the next-best static objective, Skewed-RKL). The more informative comparison is worst-case behavior. Each single-geometry objective breaks somewhere — Forward KL trails the best by 4.1 on OpenLLaMA, Reverse KL by 3.0 on GPT2-large, and α–β by 1.6 on GPT2-large — and even the strongest static baseline, Skewed-RKL, trails by up to 0.6 (Gemma). CaRE-KD’s worst gap on any pair is 0.2. On OpenLLaMA-7B→3B the 0.18 ROUGE-L gap (26.23 vs. 26.41, Table 8) is smaller than the per-seed standard deviation of either method on this pair (0.19–0.90 across tasks), so we treat the pair as a statistical tie; Gemma-2-2B-IT on Dolly (17.0 vs. 19.7) is an outright loss, which we state plainly.

Scope of applicability. Two factors set how much the confidence gate can help: how unreliable the teacher is, and how far the student sits from the teacher (roughly the compression ratio). The clearest wins occur where both are large — small students distilled from weak teachers under heavy compression (GPT2-XL→base, ∼15×; OPT-2.7B→125M, ${ \sim } 2 2 \times )$ . The one tie, OpenLLaMA-7B→3B, distills a strong, well-calibrated teacher at only 2.3×, so the student already sits close to the teacher and there is little unreliable supervision to filter; here a static mode-seeking objective is near-optimal. CaRE-KD should therefore be preferred under large capacity gaps and precision-sensitive tasks.

In the non-SGO (teacher-forcing) setting (Table 5, Appendix E), CaRE-Divergence achieves 19.5/19.7 (R/LLM) on GPT2-base, strictly dominating SRKL (17.7/17.7) and FKL (16.1/15.6); validation-set cross-entropy and exact match dynamics confirming faster, lower-loss convergence are in Figures 6–7 (Appendix E).

## 4.2 LLM-as-a-judge factuality

ROUGE-L is blind to hallucination. Therefore, we additionally grade every output with GPT-5-Mini on a 0–100 factuality scale (system prompt: Appendix D.7). CaRE-Divergence improves average factuality alongside ROUGE L on most pairs: +1.5 over SRKL on GPT2-base (21.0 vs. 19.5), +2.2 on GPT2-large, and the highest Evol-Instruct factuality on Qwen2-1.5B ( 53.0 vs. 52.1 Distillm-2). Per-task gaps reach +2.5 over SRKL (Self-Instruct, Super Natural Instructions; GPT2-base). To probe whether Revival targets hallucination-prone regions, we measure teacher factuality conditional on the mask: on Dolly with the OpenLLaMA-7B teacher, the teacher’s factuality is 25.2 when Revival fires vs. 28.6 when it does not (one-sided Wilcoxon $p \ = \ 0 . 0 6 )$ . We treat this as suggestive rather than conclusive: the Wilcoxon test is not significant, and a complementary check that splits the sequences by confidence regime and compares teacher factuality where Revival fires against the opposite region is also not significant (Mann– Whitney $p = 0 . 2 6 )$ . We therefore do not claim that Revival selects specifically low-factuality tokens at the mechanism level; the claim we make, and that holds, is that the distilled student’s end-to-end factuality improves under the LLM judge (Tables 1 and 3).

## 4.3 Revival as a loss-agnostic enhancer

Table 2 isolates Revival by reporting $\Delta = \mathrm { S c o r e } _ { \mathrm { w i t h } } - \mathrm { S c o r e } _ { \mathrm { w i t h o u t } } .$ CaRE-Divergence exhibits the most consistent positive response: +1.42 avg. ROUGE-L on OPT-125M (+3.62 on Super-Natural Instructions). Revival also improves static baselines (e.g., +1.14 for RKL on GPT2-base), supporting the broader hypothesis that selective filtering is a general enhancement mechanism [16, 17]; however, static objectives are high-variance (RKL −1.78 on Self-Instruct for GPT2-large), whereas CaRE-Divergence remains consistently positive on average, highlighting the complementarity of dynamic geometry and selective rejection. The precise claim is about the average, not every cell: CaRE-Divergence itself has two small negative entries (−0.4 on Vicuna/Gemma, −0.04 on Dolly/GPT2-large), but it has the most reliable positive response overall — the highest mean ∆ with the fewest and smallest negatives. A one-sample t-test over the per-task deltas confirms this: CaRE-Divergence t = 3.98 (p < 0.001), whereas Forward KL (t = −2.55) and Reverse KL $( t = 1 . 2 4 , p = 0 . 2 5 )$ are not reliably positive.

## 4.4 Results on specialized domains

Table 3 extends evaluation to chat alignment (Qwen2-7B→1.5B), code (DeepSeek-Coder-6.9B→1.3B), and math (Qwen2.5-Math-7B→1.5B). CaRE-Divergence leads on Evol-Instruct (26.8/53.0 vs. SRKL 26.5/52.1) and Ultra Feedback (22.6/49.3); reaches 62.4 pass@1 on MBPP (+2.1 vs. SRKL, +2.6 vs. ABKD); and achieves 73.9% on

Table 2. Impact of Revival on diferent distillation losses. Values are the per-task delta $\Delta \ = \ \mathsf { S c o r e } _ { \mathsf { w i t h - R e v i v a l } } \ -$ $\mathsf { S c o r e } _ { \mathsf { n o - R e v i v a l } }$ in ROUGE-L. CaRE-Divergence shows the most consistent positive response, with a statistically significant average improvement across model pairs (one-sample $t – \mathrm { t e s t } \colon t = 3 . 9 8 , p < 0 . 0 0 1 )$ . Static baselines such as FKL $( t = - 2 . 5 5 , p = 0 . 0 6 )$ often degrade with Revival, while RKL $( t = 1 . 2 4 , p = 0 . 2 5 )$ yields insignificant $\mathtt { g a i n s } ,$ indicating that our confidence-gated geometry is uniquely synergistic with epistemic filtering. Bold marks the largest positive delta per (model, task) cell.
<table><tr><td rowspan="2">STUDENT</td><td rowspan="2">Loss</td><td colspan="4">BENCHMARKS</td><td rowspan="2">AvG</td></tr><tr><td>DOLLY</td><td>SELF-INST</td><td>SUPER-NAT</td><td>VICUNA</td></tr><tr><td rowspan="5">GPT2-BASE</td><td>FKL</td><td>0.22</td><td>0.14</td><td>-0.64</td><td>0.09</td><td>-0.05</td></tr><tr><td>RKL</td><td>0.32</td><td>0.61</td><td>3.51</td><td>0.12</td><td>1.14</td></tr><tr><td>SRKL</td><td>0.41</td><td>0.54</td><td>1.64</td><td>0.30</td><td>0.72</td></tr><tr><td>α-β Dıv</td><td>1.11</td><td>0.80</td><td>0.94</td><td>0.67</td><td>0.88</td></tr><tr><td>CaRE</td><td>0.35</td><td>0.94</td><td>1.26</td><td>1.04</td><td>0.90</td></tr><tr><td rowspan="5">GPT2-LARGE</td><td>FKL</td><td>-0.36</td><td>-0.69</td><td>-0.22</td><td>-0.26</td><td>-0.38</td></tr><tr><td>RKL</td><td>0.35</td><td>-1.78</td><td>-1.44</td><td>1.75</td><td>-0.28</td></tr><tr><td>SRKL</td><td>0.43</td><td>0.70</td><td>1.09</td><td>-0.56</td><td>0.41</td></tr><tr><td>α-β Dıv</td><td>-0.44</td><td>-0.35</td><td>0.70</td><td>0.68</td><td>0.15</td></tr><tr><td>CaRE</td><td>-0.04</td><td>0.71</td><td>1.06</td><td>0.10</td><td>0.46</td></tr><tr><td rowspan="5">OPT-125M</td><td>FKL</td><td>-0.23</td><td>-0.18</td><td>-0.01</td><td>-0.29</td><td>-0.18</td></tr><tr><td>RKL</td><td>0.29</td><td>0.57</td><td>0.75</td><td>0.71</td><td>0.58</td></tr><tr><td>SRKL</td><td>0.27</td><td>0.61</td><td>1.98</td><td>-0.76</td><td>0.52</td></tr><tr><td>α-β DIv</td><td>-0.42</td><td>0.35</td><td>0.05</td><td>-0.43</td><td>-0.11</td></tr><tr><td>CaRE</td><td>0.55</td><td>1.12</td><td>3.62</td><td>0.38</td><td>1.42</td></tr><tr><td rowspan="5">GEMMA-2-2B-IT</td><td>FKL</td><td>-4.02</td><td>-3.68</td><td>-7.31</td><td>-0.66</td><td>-3.91</td></tr><tr><td>RKL</td><td>-0.48</td><td>-2.68</td><td>1.54</td><td>0.4</td><td>-0.3</td></tr><tr><td>SRKL</td><td>1.48</td><td>1.89</td><td>1.56</td><td>0.99</td><td>1.48</td></tr><tr><td>α-β DIv</td><td>1.37</td><td>2.05</td><td>10.17</td><td>-1.26</td><td>3.08</td></tr><tr><td>CaRE</td><td>3.6</td><td>2.61</td><td>5.79</td><td>-0.4</td><td>2.90</td></tr><tr><td rowspan="5">OLLAMA2-3B</td><td>FKL</td><td>0.18</td><td>-0.31</td><td>-0.96</td><td>0.09</td><td>-0.25</td></tr><tr><td>RKL</td><td>0.02</td><td>0.18</td><td>-0.29</td><td>-0.03</td><td>-0.04</td></tr><tr><td>SRKL</td><td>-0.15</td><td>0.67</td><td>-1.48</td><td>-0.10</td><td>-0.26</td></tr><tr><td>α-β DIv</td><td>0.24</td><td>1.14</td><td>2.16</td><td>0.43</td><td>0.99</td></tr><tr><td>CaRE</td><td>0.27</td><td>0.26</td><td>1.82</td><td>0.57</td><td>0.73</td></tr></table>

GSM8K (+1.7 vs. SRKL) and 46.1% on CollegeMath (+1.8 vs. Distillm-2/ABKD). Against the additional on-policy and divergence baselines (GKD, Distillm) now reported in Table 3 for the code and math tasks, CaRE-KD remains best on MBPP, GSM8K and CollegeMath and tied-best on HumanEval. Precision-sensitive domains — where a single hallucinated step can break a proof, a unit test, or a logical argument — benefit most from confidence-conditioned mode seeking and rejection of unreliable teacher supervision.

## 4.5 Hyperparameter sensitivity and ablations

We conduct ablations on instruction-following tasks with both GPT- and OPT-based students (Figures 3 and 8 in Appendix E). Several consistent trends emerge across architectures. Moderate-to-high skip percentages (roughly 50–70%) consistently outperform low filtering regimes, indicating that a substantial portion of teacher supervision lies in epistemically unreliable regions. An increasing Revival schedule, where selectivity is gradually strengthened over training, reliably yields the best performance, supporting a curriculum efect in which early broad supervision transitions into progressive filtering as the student’s confidence improves. Increasing the number of Monte Carlo samples used for BALD estimation improves stability, though strong performance is already achieved with as few as $N = 3$ when combined with an increasing schedule. Hard gating attains the highest peak performance, while soft gating provides slightly greater robustness across configurations. Performance exhibits a unimodal dependence on sampling temperature, peaking at moderate values (around $T = 3 . 0 )$ ), confirming that efective epistemic estimation requires controlled stochasticity rather than aggressive noise. We further ablate the margin m in $g _ { \mathrm { s o f t } }$ defined in Equation 3. Lower m performs the best, balancing the forward and reverse KL divergence metrics, while higher margins bias the objective and degrade performance. A small margin does not collapse the soft gate into hard gating: m only shifts where the sigmoid is centered, not its slope, so at $m = 0 . 0 5$ the gate is still a smooth, diferentiable function of the confidence gap — a near-symmetric switch centered near a gap of zero — and the soft-gate selection term of Proposition 3.1 is retained. Its being the best margin is expected: it matches the hard-gate behavior of switching branches right at the crossover point, exactly where hard gating attains its highest peak, whereas a large m biases the gate toward one geometry across most tokens. Soft gating is thus the robustness-oriented variant and hard gating the peak-performance variant. The consistency of these trends across GPT and OPT families demonstrates that the design principles underlying CaRE-KD generalize across architectures.

Table 3. Generalization to specialized domains. Cells report ROUGE-L / LLM-as-a-judge factuality. We evaluate three model families: chat alignment (Qwen2), code generation (DeepSeek-Coder), and math reasoning (Qwen2.5- Math). CaRE-Divergence consistently improves precision-sensitive tasks, achieving +2.1 pp on MBPP, +1.7 pp on GSM8k, and +1.8 pp on CollegeMath over the strongest baseline. Gains are statistically significant $( t = 3 . 0 5 , p = 0 . 0 1 8 )$ Bold denotes the best score per cell. GKD and Distillm are additionally reported on the code and math tasks, where CaRE-KD remains best on MBPP, GSM8K and CollegeMath and tied-best on HumanEval; chat-alignment entries fo these two baselines are still running and omitted $( ^ { ( \mathfrak { q } } - ^ { \mathfrak { n } } )$
<table><tr><td rowspan="2">METHOD</td><td colspan="3">QWEN2-7B-INST → QWEN2-1.5B</td><td colspan="2">DS-CODER-6.9B-INST  $\mathbf { \tau }  \mathbf { D S - C o D E R - 1 . 3 B }$ </td><td colspan="2">QWEN2.5-MATH-7B-INST → QWEN2.5-MATH-1.5B</td></tr><tr><td>ALPACA ROUGE-L/LLM</td><td>EvoL ROUGE-L/LLM</td><td>ULTRA ROUGE-L/LLM</td><td>HEvAL PASS@1</td><td>MBPP PASS@1</td><td>GSM8K PASS@1</td><td>COLLEGE PASS@1</td></tr><tr><td>STUDENT</td><td>8.81/27.3</td><td>14.9/31.2</td><td>10.8/30.5</td><td>32.3</td><td>58.5</td><td>69.9</td><td>37.1</td></tr><tr><td>DISTILLM2</td><td>16.3/43.3</td><td>26.5/52.1</td><td>22.1/49.3</td><td>43.3</td><td>60.3</td><td>72.2</td><td>44.2</td></tr><tr><td>ABKD</td><td>16.4/44.8</td><td>26.7/51.9</td><td>22.3/48.4</td><td>42.7</td><td>59.8</td><td>71.0</td><td>44.3</td></tr><tr><td>GKD</td><td></td><td></td><td></td><td>43.1</td><td>60.2</td><td>71.4</td><td>43.9</td></tr><tr><td>DISTILLM</td><td></td><td></td><td></td><td>42.9</td><td>60.0</td><td>71.7</td><td>44.1</td></tr><tr><td>CaRE-KD(OURs)</td><td>16.4/44.4</td><td>26.8/53.0</td><td>22.6/49.3</td><td>43.3</td><td>62.4</td><td>73.9</td><td>46.1</td></tr></table>

![](images/ea05c843bae097177ca8b9fe73625cd3fc406ca11a294a06d8811c0a607032ec.jpg)

![](images/2a3fe54ed1c3c9625960757867fcba6f9dcbeb3acfcb3fd15e3d0ee23a5bcd01.jpg)

![](images/41a4362e0bc868d555aafa15a9c272b5f4f2dc994ac068c27bef0e3bfe49d701.jpg)

![](images/b088e4de452f5db38037c9a1d972929651cf6fcb5a55c59849f15bb386d83905.jpg)

![](images/8c04c8011a80b8372731138de4a16a88d772197679d25c5868f825a215a1af96.jpg)

![](images/766d62a742ceee836a60a67880e824ae81bf08bf71b6284e64268a0f90cd4580.jpg)  
Figure 3. Ablation study of CaRE-KD with GPT2-base. (a) gating strategy $( g _ { \mathrm { h a r d } } \lor 5 . g _ { \mathrm { s o f t } } ) ;$ (b) MC-dropout sample count $N ; ( \mathsf { c } )$ sampling temperature T; (d) target skip percentage τ; (e) Revival scheduling strategy (constant, increasing, decreasing); and (f) margin m in soft gating.

## 5 Discussion

Decoupling student confidence from teacher uncertainty. A core failure mode of standard KD is that the student inherits the teacher’s epistemic uncertainty rather than its competence: when the teacher is unsure, Forward KL forces the student to spread probability mass over the teacher’s heavy tail, eroding sharper, correct priors. Figure 4 shows that CaRE-KD explicitly avoids this trap. Across all four benchmarks, the teacher’s BALD distribution exhibits a long, heavy tail of high-uncertainty samples; the student trained with Revival retains a tightly concentrated low-BALD profile that is closer to its pre-distillation prior than to the teacher’s tail. The decoupling is large in magnitude: on Self-Instruct the teacher’s mean BALD is 0.52 while the corresponding distilled student reaches 0.077, and the Kolmogorov–Smirnov distance between the two distributions exceeds 0.5 on Vicuna and Dolly. Reading the preversus post-distillation KS drift (Figures 1a and 4) also isolates the two components of CaRE-KD. Standard Skewed RKL barely moves the student’s uncertainty away from the teacher: the KS distance drifts by only 0.4% (0.840 →

0.836). The CaRE-Divergence token-level loss alone (Revival removed) already drifts by 2.4% on Dolly (0.840→ 0.864) — several times the baseline — and adding Revival raises the drift to $7 \%$ . The token-level loss is thus the main driver and Revival amplifies it; neither is a 0.4% efect. The same pattern is reproduced under static divergences augmented with Revival (Skewed RKL and α–β, Figures 9–10 in Appendix F), establishing that the efect is driven by selective rejection rather than by any one objective. Crucially, the rejection rate is not a static threshold: the proportion of batches triggering $M _ { \mathrm { R e v i v a l } }$ rises monotonically through training (Figures 11–12), reflecting a curriculum in which broad early supervision gives way to progressively stricter filtering as the student becomes more confident than the teacher on a growing subset of inputs. This empirically realizes the bi-directional conditional calibration formalized in Theorem 3.5.

![](images/457fb741112582f6c1c2b0e484fa83e82ddc89e7cc8de0bae60712519101ef0b.jpg)

![](images/346299f947bb045a555e23a39ff204db7e764dfb4cdcf78a1a3094524dd98ae8.jpg)

![](images/7c9f8712d70439f754ab7b50d134573ce8d97014bb0c62d939b719dccaff3bf3.jpg)

![](images/5dd3a7bdaa166e43e56c1cd3464a4945419cc6b13c494e427683a4b96b5d7938.jpg)  
Figure 4. BALD uncertainty distributions across four instruction-following benchmarks. Teacher models show heavy high-uncertainty tails (red), whereas CaRE-KD students (blue) suppress these regions and maintain concentrated low-BALD profiles. Reported KS distances indicate strong distributional separation, exceeding 0.5 on Vicuna and Dolly.

Computational analysis. Revival trades a moderate amount of additional forward-pass compute for substantially

![](images/b6b9289094d1e6cff7abe1784edc47304378ad2e13464afa99793026ec907180.jpg)  
(a) Wall-clock vs. MC samples

![](images/6b713b0d08179cca5c6a4cf6ddf69495a7b5e045eaf73a71bde2265f7b691f29.jpg)  
(b) Wall-clock vs. skip rate  
Figure 5. Runtime on Dolly-15k with OpenLLaMA-3B for one epoch. (a) Wall-clock time scales near-linearly with the number of MC dropout samples N for BALD estimation. (b) Higher skip rates τ reduce BALD overhead by skipping backward passes on rejected batches.

more reliable supervision. The dominant cost is BALD estimation, which requires N stochastic dropout passes per batch and scales near-linearly in N (Figure 5(a)). Concretely, on Dolly with OpenLLaMA-3B, one epoch takes 5520s without Revival and 7860s with $N = 3 .$ a 42% overhead; pushing to $N = 7$ approaches a 98% overhead. This cost is partially recovered by the rejection mechanism itself: when $M _ { \mathrm { R e v i v a l } } = 1$ , the corresponding backward pass is skipped, so increasing τ reduces efective wall-clock per epoch $( \mathrm { F i g u r e ~ } 5 ( \boldsymbol { \mathbf { b } } ) ) ;$ ; a 70% skip target recovers roughly 12% of the BALD overhead. This ∼40% figure is specific to instruction following, however; the per-domain overhead at $N = 3$ is substantially larger, and is largest precisely when the training step is otherwise cheap (per domain breakdown in Figure 13, Appendix F): approximately +106% for chat alignment, +170% for code, and +148% for math. We therefore no longer describe the overhead as “modest.” Revival is a poor fit for settings with very large teachers, long sequences, or online/streaming updates; in those regimes the single-pass entropy proxy justified by Theorem 3.4 removes the extra stochastic passes at a small loss of precision. Because N = 3 paired with an increasing schedule already achieves the strongest empirical performance and distillation is a one-time ofline cost incurred before deployment, the overhead remains acceptable for the ofline, high-compression settings CaRE-KD targets.

Limitations. BALD-based rejection requires multiple stochastic passes and therefore costs more than single-pass distillation. As shown above, this overhead is ∼40% at N = 3 for instruction following but considerably larger per domain (up to ∼170% on code), and is largest when the training step is otherwise cheap; it is partially amortized by skipped updates, but in compute-constrained regimes, or with very large teachers, long sequences, or online updates, it may still be undesirable. In such settings, the cheaper entropy proxy justified by Theorem 3.4 remains a valid dropin for the rejection rule, with a small loss of precision. Second, the equivalence between entropy-based confidence and BALD relies on the Sharpness Hypothesis (Assumption 3.3), which is best supported for cross-entropy-trained LLMs in regimes where individual MC samples are sharp. We stress that this assumption is used only to justify the optional single-pass entropy fallback: every reported result uses the full MC-dropout BALD estimate, which does not rely on Assumption 3.3. In genuinely multimodal open-ended generation, where stochastic passes can disagree on which valid continuation to predict rather than how confidently, this assumption may weaken; relaxing the analysis to high-aleatoric regimes is a natural direction for future work.

## 6 Conclusion

We introduced CaRE-KD, a confidence-aware framework that redefines how supervision is transferred in generative knowledge distillation. By dynamically adapting the optimization geometry at the token level via CaRE-Divergence, the student selectively absorbs informative teacher signals while suppressing unreliable long-tail noise. Complementarily, the epistemic rejection mechanism Revival prevents the student from inheriting teacher hallucinations, preserving correct priors under uncertain supervision. Across instruction following, code generation, and mathematical reasoning, CaRE-KD consistently yields students that are both more accurate and better calibrated than those trained with static divergence objectives. These findings underscore that efective distillation from modern LLMs requires adapting not only how much to imitate the teacher, but when to trust it.

## References

[1] Y. Kim and A. M. Rush, “Sequence-level knowledge distillation,” arXiv preprint arXiv:1606.07947, 2016.

[2] R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem, “On-policy distillation of language models: Learning from self-generated mistakes,” in The Twelfth International Conference on Learning Representations, 2024.

[3] A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang, “Why language models hallucinate,” arXiv preprint arXiv:2509.04664, 2025.

[4] F. Sun, N. Li, K. Wang, and L. Goette, “Large language models are overconfident and amplify human bias,” arXiv preprint arXiv:2505.02151, 2025.

[5] Y. Gu, L. Dong, F. Wei, and M. Huang, “Minillm: Knowledge distillation of large language models,” in The Twelfth International Conference on Learning Representations, 2024.

[6] J. Ko, S. Kim, T. Chen, and S.-Y. Yun, “Distillm: Towards streamlined distillation for large language models,” arXiv preprint arXiv:2402.03898, 2024.

[7] G. Wang, Z. Yang, Z. Wang, S. Wang, Q. Xu, and Q. Huang, “ABKD: Pursuing a proper allocation of the probability mass in knowledge distillation via \$\alpha\$-\$\beta\$-divergence,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=vt65VjJak

[8] S. Stanton, P. Izmailov, P. Kirichenko, A. A. Alemi, and A. G. Wilson, “Does knowledge distillation really work?” Advances in Neural Information Processing Systems, vol. 34, pp. 6906–6919, 2021.

[9] S. K. Ramesh, A. Sengupta, and T. Chakraborty, “On the generalization vs fidelity paradox in knowledge distillation,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 17 930–17 951. [Online]. Available: https://aclanthology.org/2025.findings-acl.923

[10] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, S. Johnston, S. El-Showk, A. Jones, N. Elhage, T. Hume, A. Chen, Y. Bai, S. Bowman, S. Fort, D. Ganguli, D. Hernandez, J. Jacobson, J. Kernion, S. Kravec, L. Lovitt, K. Ndousse, C. Olsson, S. Ringer, D. Amodei, T. Brown, J. Clark, N. Joseph, B. Mann, S. McCandlish, C. Olah, and J. Kaplan, “Language models (mostly) know what they know,” 2022. [Online]. Available: https://arxiv.org/abs/2207.05221

[11] G. Prato, J. Huang, P. Parthasarathi, S. Sodhani, and S. Chandar, “Do large language models know how much they know?” in Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 6054–6070. [Online]. Available: https://aclanthology.org/2024.emnlp-main.348/

[12] N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel, “Bayesian active learning for classification and preference learning,” arXiv preprint arXiv:1112.5745, 2011.

[13] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[14] A. Sengupta, S. Dixit, M. S. Akhtar, and T. Chakraborty, “A good learner can teach better: Teacher-student collaborative knowledge distillation,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=Ixi4j6LtdX

[15] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang, “Deepseek-r1 incentivizes reasoning in llms through reinforcement learning,” Nature, vol. 645, no. 8081, p. 633–638, Sep. 2025. [Online]. Available: http://dx.doi.org/10.1038/s41586-025-09422-z

[16] L. Fang, Y. Chen, W. Zhong, and P. Ma, “Bayesian knowledge distillation: A bayesian perspective of distillation with uncertainty quantification,” in Forty-first International Conference on Machine Learning, 2024. [Online]. Available: https://openreview.net/forum?id=knZ4NYzGUd

[17] C. He, Y. Ding, J. Guo, R. Gong, H. Qin, and X. Liu, “DA-KD: Dificulty-aware knowledge distillation for eficient large language models,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=NCYBdRCpw1

[18] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in International Conference on Machine Learning. PMLR, 2016, pp. 1050–1059.

[19] J. Ko, T. Chen, S. Kim, T. Ding, L. Liang, I. Zharkov, and S.-Y. Yun, “DistiLLM-2: A contrastive approach boosts the distillation of LLMs,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=rc65N9xIrY

[20] Y. Pawitan and C. Holmes, “Confidence in the reasoning of large language models,” Harvard Data Science Review, vol. 7, no. 1, pp. 2644–2353, 2025.

[21] M. Conover, M. Hayes, A. Mathur, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, and R. Xin, “Free dolly: Introducing the world’s first truly open instructiontuned llm,” 2023.

[22] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised multitask learners,” 2019.

[23] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin et al., “Opt: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068, 2022.

[24] G. Team, M. Riviere, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju, L. Hussenot, T. Mesnard, B. Shahriari, A. Ramé et al., “Gemma 2: Improving open language models at a practical size,” arXiv preprint arXiv:2408.00118, 2024.

[25] X. Geng and H. Liu, “Openllama: An open reproduction of llama,” May 2023. [Online]. Available: https: //github.com/openlm-research/open\_llama

[26] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning language models with self-generated instructions,” in Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), 2023, pp. 13 484–13 508.

[27] Y. Wang, S. Mishra, P. Alipoormolabashi, Y. Kordi, A. Mirzaei, A. Naik, A. Ashok, A. S. Dhanasekaran, A. Arunkumar, D. Stap, E. Pathak, G. Karamanolakis, H. Lai, I. Purohit, I. Mondal, J. Anderson, K. Kuznia, K. Doshi, K. K. Pal, M. Patel, M. Moradshahi, M. Parmar, M. Purohit, N. Varshney, P. R. Kaza, P. Verma, R. S. Puri, R. Karia, S. Doshi, S. K. Sampat, S. Mishra, S. Reddy A, S. Patro, T. Dixit, and X. Shen, “Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks,” in Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 5085–5109. [Online]. Available: https://aclanthology.org/2022.emnlp-main.340/

[28] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Xu, J. Zhou, J. Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, and Z. Fan, “Qwen2 technical report,” arXiv preprint arXiv:2407.10671, 2024.

[29] N. Ding, Y. Chen, B. Xu, Y. Qin, S. Hu, Z. Liu, M. Sun, and B. Zhou, “Enhancing chat language models by scaling highquality instructional conversations,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 3029–3051.

[30] X. Li, T. Zhang, Y. Dubois, R. Taori, I. Gulrajani, C. Guestrin, P. Liang, and T. B. Hashimoto, “Alpacaeval: An automatic evaluator of instruction-following models,” 2023.

[31] C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, Q. Lin, and D. Jiang, “Wizardlm: Empowering large pre-trained language models to follow complex instructions,” in The Twelfth International Conference on Learning Representations, 2024.

[32] G. Cui, L. Yuan, N. Ding, G. Yao, B. He, W. Zhu, Y. Ni, G. Xie, R. Xie, Y. Lin, Z. Liu, and M. Sun, “Ultrafeedback: boosting language models with scaled ai feedback,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[33] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo, Y. Xiong, and W. Liang, “Deepseek-coder: When the large language model meets programming – the rise of code intelligence,” 2024. [Online]. Available: https://arxiv.org/abs/2401.14196

[34] Z. Luo, C. Xu, P. Zhao, Q. Sun, X. Geng, W. Hu, C. Tao, J. Ma, Q. Lin, and D. Jiang, “Wizardcoder: Empowering code large language models with evol-instruct,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=UnUwSIgK5W

[35] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba, “Evaluating large language models trained on code,” 2021. [Online]. Available: https://arxiv.org/abs/2107.03374

[36] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le et al., “Program synthesis with large language models,” arXiv preprint arXiv:2108.07732, 2021.

[37] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

[38] L. Yu, W. Jiang, H. Shi, J. YU, Z. Liu, Y. Zhang, J. Kwok, Z. Li, A. Weller, and W. Liu, “Metamath: Bootstrap your own mathematical questions for large language models,” in The Twelfth International Conference on Learning Representations.

[39] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano et al., “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168, 2021.

[40] Z. Tang, X. Zhang, B. Wang, and F. Wei, “Mathscale: Scaling instruction tuning for mathematical reasoning,” arXiv preprint arXiv:2403.02884, 2024.

[41] S. Sun, Y. Cheng, Z. Gan, and J. Liu, “Patient knowledge distillation for bert model compression,” arXiv preprint arXiv:1908.09355, 2019.

[42] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister, “Distilling step-by step! outperforming larger language models with less training data and smaller model sizes,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 8003–8017.

[43] Q. Zhong, L. Ding, L. Shen, J. Liu, B. Du, and D. Tao, “Revisiting knowledge distillation for autoregressive language models,” arXiv preprint arXiv:2402.11890, 2024.

[44] M. Li, F. Zhou, and X. Song, “Bild: Bi-directional logits diference loss for large language model distillation,” in Proceed ings ofthe 31st International Conference on Computational Linguistics, 2025, pp. 1168–1182.

[45] K. Saadi and D. Wang, “TASKD-LLM: Task-aware selective knowledge distillation for LLMs,” in ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Eficacy, 2025. [Online]. Available: https://openreview.net/forum?id=QQBfoVJWY2

[46] X.-C. Li, W.-S. Fan, S. Song, Y. Li, S. Yunfeng, D.-C. Zhan et al., “Asymmetric temperature scaling makes larger networks teach well again,” Advances in neural information processing systems, vol. 35, pp. 3830–3842, 2022.

[47] K. Zheng and E.-H. Yang, “Knowledge distillation based on transformed teacher matching,” arXiv preprint arXiv:2402.11148, 2024.

[48] S. Sun, W. Ren, J. Li, R. Wang, and X. Cao, “Logit standardization in knowledge distillation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 15 731–15 740.

[49] Y. Gu, L. Dong, F. Wei, and M. Huang, “Minillm: Knowledge distillation of large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2306.08543

[50] G. Kim, D. Jang, and E. Yang, “Promptkd: Distilling student-friendly knowledge for generative language models via prompt tuning,” in Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, 2024, pp. 6266–6282.

[51] T. Huang, S. You, F. Wang, C. Qian, and C. Xu, “Knowledge distillation from a stronger teacher,” Advances in Neural Information Processing Systems, vol. 35, pp. 33 716–33 727, 2022.

[52] Y. Wen, Z. Li, W. Du, and L. Mou, “F-divergence minimization for sequence-level knowledge distillation,” arXiv preprint arXiv:2307.15190, 2023.

[53] W. F. Huang, J. Tao, C. Deng, M. Fan, W. Wan, Q. Xiong, and G. Piao, “Rényi divergence deep mutual learning,” in Machine Learning and Knowledge Discovery in Databases: Research Track, D. Koutra, C. Plant, M. Gomez Rodriguez, E. Baralis, and F. Bonchi, Eds. Cham: Springer Nature Switzerland, 2023, pp. 156–172.

[54] J. Lv, H. Yang, and P. Li, “Wasserstein distance rivals kullback-leibler divergence for knowledge distillation,” Advances in Neural Information Processing Systems, vol. 37, pp. 65 445–65 475, 2024.

[55] C. Guo, S. Zhong, X. Liu, Q. Feng, and Y. Ma, “Why does knowledge distillation work? rethink its attention and fidelity mechanism,” Expert Systems with Applications, vol. 262, p. 125579, 2025.

[56] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon, B. Lakshminarayanan, and J. Snoek, “Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift,” Advances in neural information processing systems, vol. 32, 2019.

[57] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality,” March 2023. [Online]. Available: https://lmsys.org/blog/2023-03-30-vicuna/

[58] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9

[59] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighof, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, “The language model evaluation harness,” 07 2024. [Online]. Available: https://zenodo.org/records/12608602

[60] J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 21 558–21 572, 2023.

[61] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Eficient memory management for large language model serving with pagedattention,” in Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[62] A. Lin, J. Wohlwend, H. Chen, and T. Lei, “Autoregressive knowledge distillation through imitation learning,” 2020. [Online]. Available: https://arxiv.org/abs/2009.07253

[63] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in International conference on machine learning. PMLR, 2017, pp. 1321–1330.

[64] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” Advances in neural information processing systems, vol. 30, 2017.

A Related Work 16   
B Algorithmic Details 17   
B.1 Estimation of Epistemic Uncertainty (BALD) 17   
B.2 Verification that the uncertainty estimator is non-degenerate 18   
B.3 Numerical stability of the divergence terms 18   
Proofs of Theoretical Results 18   
C.1 Gradient of CaRE-Divergence 18   
C.2 Gradient of Skewed Reverse KL 19   
C.3 Worst-case robustness under Revival 19   
C.4 Proof of Proposition 3.2 (Limiting equivalence to SRKL) 20   
C.5 Proof of Theorem 3.4 (Conditional equivalence of confidence proxies) 20   
C.6 Proof of Theorem 3.5 (Conditional calibration of overconfident students) 20   
C.7 Proof of Theorem 3.5 (Conditional sharpening of underconfident students) 21   
Experimental Details 21   
D.1 Training Datasets 21   
D.2 Evaluation Benchmarks 22   
D.3 Training Details 22   
D.4 Evaluation Protocol 23   
D.5 Prompt Structure 23   
D.6 Hardware 23   
D.7 LLM-as-a-judge Factuality Evaluation Prompt 23   
Additional Results 24   
E.1 Comparison of CaRE-KD against baselines 24   
Detailed Discussions 28   
F.1 Decoupling student confidence from teacher uncertainty 28   
F.2 Evolution of student confidence over distillation iterations 30   
Calibration, Selective Prediction, and Selection-Bias Analyses 31   
G.1 Calibration (ECE) and selective prediction 31   
G.2 Revival does not preferentially reject hard examples or reduce diversity 32

## A Related Work

Knowledge distillation for LLMs. Knowledge Distillation (KD) has progressed from early logit-matching formulations [13] to methods explicitly designed for the generative and autoregressive nature of large language models. Initial extensions focused on sequence-level and white-box distillation objectives [1, 41], while more recent work addresses challenges such as exposure bias, vocabulary mismatch, and distributional shift during generation [42, 43]. Agarwal et al. [2] proposed Generalized Knowledge Distillation (GKD), which performs on-policy distillation by training the student on its own generations scored by the teacher Gu et al. [5] introduced MiniLLM, showing that Reverse KL can mitigate the zero-forcing behavior of Forward KL by reducing sensitivity to low-probability teacher tails. Sengupta et al. [14] explored collaborative distillation, where teacher and student roles evolve dynamically based on learning progress. Large-scale technical reports such as DeepSeek-V3 [15] further demonstrate that distillation from strong reasoning models can substantially improve smaller models when teacher signals are carefully filtered. Complementing these findings, Ramesh et al. [9] provide systematic evidence that noisy or inconsistent teacher supervision can collapse student performance. A growing body of work therefore emphasizes uncertainty-aware and selective distillation mechanisms to mitigate imperfect supervision [16, 17, 44, 45].

Objective functions in knowledge distillation. The distillation objective determines the optimization geometry and, consequently, the behavior of the student. Most traditional approaches optimize Forward KL divergence [13, 46–48], which is meanseeking and encourages the student to cover the teacher’s full support. While this promotes recall, it also exposes the student to tail risk, allocating probability mass to incorrect or ambiguous tokens when the teacher distribution has high entropy [49]. In contrast, Reverse KL divergence [5, 50] is mode-seeking and produces sharper predictions, but can induce mode collapse and reduced diversity. To balance these extremes, recent methods propose static interpolations. Skewed KL [6] reweights gradients through a mixture target to stabilize training, while $\alpha { - } \beta$ divergence [7] provides a parametric family that controls tail weighting via fixed hyperparameters. Other explored alternatives include Pearson correlation [51], total variation distance [52], Rényi divergence [53], and Wasserstein distance [54]. Despite their expressiveness, these objectives impose a global geometric preference across all tokens and do not adapt to token-level variability in teacher reliability.

In contrast, the confidence-gated objective CaRE-Divergence operates as a dynamic geometric switch rather than a static interpolation. By selecting Forward KL when the teacher is confident and Reverse KL when the teacher is uncertain, it en ables token-level adaptation of the precision–recall trade-of [55]. Moreover, by explicitly incorporating both aleatoric and epistemic uncertainty through the Revival mechanism, our approach selectively rejects unreliable teacher supervision when the student is already confident. This combination of adaptive geometry and selective rejection distinguishes our method from prior divergence-based and uncertainty-aware distillation approaches.

## B Algorithmic Details

## B.1 Estimation of Epistemic Uncertainty (BALD)

We estimate epistemic uncertainty using BALD computed via Monte Carlo (MC) dropout. For a batch of inputs, we perform S stochastic forward passes, producing logits $\mathbf { Z } \in \mathbb { R } ^ { S \times \mathbf { \bar { B } } \times L \times V }$ , where B is batch size, L is sequence length, and V is vocabulary size. We do not depend on the base model’s deployment-time dropout setting: for the BALD passes we force all dropout layers into training mode and set attention dropout to $p = 0 . 1$ 1, then restore the original configuration afterward. This guarantees a non-degenerate stochastic signal even for models shipped with dropout disabled (a verification on both teachers used here is given in Appendix B.2). Let

$$
\mathbf { P } _ { s , b , t } = \operatorname { s o f t m a x } ( \mathbf { Z } _ { s , b , t } ) \in \Delta ^ { | V | - 1 }\tag{8}
$$

denote the corresponding next-token distributions. The posterior predictive distribution is approximated by the mean probability

$$
\bar { \mathbf P } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \mathbf P _ { s } \in \mathbb R ^ { B \times L \times | V | } .\tag{9}
$$

BALD is the mutual information between predictions and model parameters, which admits the standard decomposition into total predictive uncertainty and expected (aleatoric) uncertainty. Let $H ( \cdot )$ denote Shannon entropy over the vocabulary dimension:

$$
\mathbf { H } _ { \mathrm { t o t a l } } = H ( \bar { \mathbf { P } } ) \in \mathbb { R } ^ { B \times L } ,\tag{10}
$$

$$
\mathbf { H } _ { \mathrm { a l e a t o r i c } } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } H ( \mathbf { P } _ { s } ) \in \mathbb { R } ^ { B \times L } .\tag{11}
$$

The token-wise epistemic uncertainty is then

$$
\mathbf { I } _ { b , t } = \mathbf { H } _ { \mathrm { t o t a l } , b , t } - \mathbf { H } _ { \mathrm { a l e a t o r i c } , b , t } \in \mathbb { R } ^ { B \times L } .\tag{12}
$$

To obtain a sequence-level signal for the rejection mask, we mean-pool across tokens:

$$
\mathcal { U } _ { \mathrm { B A L D } } ^ { ( b ) } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \mathbf { I } _ { b , t } .\tag{13}
$$

## B.2 Verification that the uncertainty estimator is non-degenerate

MC-dropout uncertainty is known to be coarse [18, 56], and a natural concern is that it could collapse to a constant for models not trained with meaningful dropout. Because Revival forces dropout to $p = 0 . 1$ for the estimation passes (Appendix B.1), the signal remains informative. We verify this directly on the two teachers used in our instruction-following experiments. On 500 Dolly evaluation sequences with $N = 3$ passes, every sequence has non-zero BALD and the top-1 probability moves appreciably across passes, confirming a genuine stochastic signal rather than a degenerate estimate.

Table 4. Verification that MC-dropout BALD is non-degenerate on the two teachers (500 Dolly sequences, $N \ = \ 3 ,$ dropout forced to $p = 0 . 1 )$ .
<table><tr><td>Teacher</td><td> $N$ </td><td>Mean BALD</td><td>BALD range</td><td>Non-zero BALD</td><td>Mean  $\Delta$  top-1 prob</td></tr><tr><td>GPT2-XL</td><td>3</td><td>0.049</td><td>0.013-0.107</td><td>500/500 (100%)</td><td>0.074</td></tr><tr><td>OPT-2.7B</td><td>3</td><td>0.152</td><td>0.057–0.239</td><td>500/500 (100%)</td><td>0.079</td></tr></table>

Crucially, Revival uses only a relative teacher-vs-student comparison via running quantiles rather than a calibrated absolute value, which tolerates a coarse estimate; this is why performance is flat between $N = 3$ and $N = 7$ (Section 4.5).

## B.3 Numerical stability of the divergence terms

A potential concern is how $\mathrm { K L } ( q \| p )$ is kept finite when the teacher assigns near-zero probability to some tokens. The teacher is a tempered softmax, so p has full support by construction. In practice, log-probabilities are taken directly from log\_softmax and clamped to a finite floor before use, and the $\alpha { - } \beta$ objective additionally mixes in a small uniform mass before taking the logarithm. Consequently no inf or NaN values occurred in any of our runs.

## C Proofs of Theoretical Results

## C.1 Gradient of CaRE-Divergence

Scope and assumptions. This decomposition applies when the gate $g$ is diferentiable and not detached from the computation graph (e.g., soft gating). $\operatorname { I f } g$ is detached, then $\nabla _ { \boldsymbol { \theta } } g \equiv 0$ and the selection term below vanishes.

Proof. Let θ parameterize the student distribution $q _ { \theta }$ and define

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g K L } } ( \boldsymbol { \theta } ) = g ( \boldsymbol { \theta } ) \mathcal { L } _ { \mathrm { F K L } } ( \boldsymbol { \theta } ) + \left( { 1 - g ( \boldsymbol { \theta } ) } \right) \mathcal { L } _ { \mathrm { R K L } } ( \boldsymbol { \theta } ) , } \end{array}
$$

where ${ \mathcal { L } } _ { \mathrm { F K L } } ( \theta ) = \operatorname { K L } ( p \| q _ { \theta } )$ and ${ \mathcal { L } } _ { \mathrm { R K L } } ( \theta ) = \mathrm { K L } ( q _ { \theta } \| p )$ , and $p$ is treated as fixed with respect to θ. Applying the product rule,

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { g K L } } = \nabla _ { \boldsymbol { \theta } } \bigl ( g \mathcal { L } _ { \mathrm { F K L } } \bigr ) + \nabla _ { \boldsymbol { \theta } } \bigl ( ( 1 - g ) \mathcal { L } _ { \mathrm { R K L } } \bigr )\tag{14}
$$

$$
= g \nabla _ { \theta } \mathcal { L } _ { \mathrm { F K L } } + \mathcal { L } _ { \mathrm { F K L } } \nabla _ { \theta } g + \left( { 1 - g } \right) \nabla _ { \theta } \mathcal { L } _ { \mathrm { R K L } } - \mathcal { L } _ { \mathrm { R K L } } \nabla _ { \theta } g\tag{15}
$$

$$
= \Big [ g \nabla _ { \theta } \mathcal { L } _ { \mathrm { F K L } } + ( 1 - g ) \nabla _ { \theta } \mathcal { L } _ { \mathrm { R K L } } \Big ] + \Big [ \big ( \mathcal { L } _ { \mathrm { F K L } } - \mathcal { L } _ { \mathrm { R K L } } \big ) \nabla _ { \theta } g \Big ] ,\tag{16}
$$

which yields the learning term and the gate-induced term.

For soft gating, let

$$
g = \sigma \big ( c ( p ) - c ( q _ { \theta } ) - m \big ) , \qquad \Delta \triangleq c ( p ) - c ( q _ { \theta } ) - m .
$$

Then $\nabla _ { \theta } g = \sigma ^ { \prime } ( \Delta ) \ \nabla _ { \theta } \Delta = g ( 1 - g ) \ \nabla _ { \theta } \Delta$ and since $c ( p )$ and m are constants w.r.t. θ,

$$
\nabla _ { \theta } \Delta = - \nabla _ { \theta } c ( q _ { \theta } ) .
$$

Therefore,

$$
\nabla _ { \theta } g = - g ( 1 - g ) \nabla _ { \theta } c ( q _ { \theta } ) ,\tag{17}
$$

and the gate-induced term becomes

$$
\begin{array} { r } { ( \mathcal { L } _ { \mathrm { F K L } } - \mathcal { L } _ { \mathrm { R K L } } ) \nabla _ { \theta } g = - g ( 1 - g ) ( \mathcal { L } _ { \mathrm { F K L } } - \mathcal { L } _ { \mathrm { R K L } } ) \nabla _ { \theta } c ( q _ { \theta } ) . } \end{array}\tag{18}
$$

This concludes the derivation.

## C.2 Gradient of Skewed Reverse KL

We derive the gradient of Skewed Reverse $\mathrm { K L } , \mathrm { K L } ( q \| m )$ , with $m = \lambda p + ( 1 - \lambda ) q$ and $\lambda \in [ 0 , 1 ]$ , with respect to student logits z where $q = \operatorname { s o f t m a x } ( z )$

Proof. Define

$$
\mathcal { L } = \mathrm { K L } ( q \| m ) = \sum _ { k } q _ { k } \log \frac { q _ { k } } { m _ { k } } = \sum _ { k } q _ { k } \log q _ { k } - \sum _ { k } q _ { k } \log m _ { k } .
$$

First compute $\partial \mathcal { L } / \partial q _ { i }$ . For the entropy term,

$$
{ \frac { \partial } { \partial q _ { i } } } \sum _ { k } q _ { k } \log q _ { k } = 1 + \log q _ { i } .
$$

For the cross term, apply the product rule:

$$
{ \frac { \partial } { \partial q _ { i } } } \sum _ { k } q _ { k } \log m _ { k } = \log m _ { i } + \sum _ { k } q _ { k } { \frac { \partial \log m _ { k } } { \partial q _ { i } } } .
$$

Since $m _ { k } = \lambda p _ { k } + ( 1 - \lambda ) q _ { k }$ , we have $\begin{array} { r } { \frac { \partial m _ { k } } { \partial q _ { i } } = ( 1 - \lambda ) \delta _ { k i } } \end{array}$ , hence

$$
\frac { \partial \log m _ { k } } { \partial q _ { i } } = \frac { 1 } { m _ { k } } ( 1 - \lambda ) \delta _ { k i } , \qquad \sum _ { k } q _ { k } \frac { \partial \log m _ { k } } { \partial q _ { i } } = ( 1 - \lambda ) \frac { q _ { i } } { m _ { i } } .
$$

Therefore,

$$
\frac { \partial \mathcal { L } } { \partial q _ { i } } = ( 1 + \log q _ { i } ) - \left( \log m _ { i } + ( 1 - \lambda ) \frac { q _ { i } } { m _ { i } } \right) = \log \frac { q _ { i } } { m _ { i } } + 1 - ( 1 - \lambda ) \frac { q _ { i } } { m _ { i } } .\tag{19}
$$

Next, use the softmax Jacobian $\begin{array} { r } { \frac { \partial q _ { k } } { \partial z _ { j } } = q _ { k } ( \delta _ { k j } - q _ { j } ) } \end{array}$

$$
\frac { \partial \mathcal { L } } { \partial z _ { j } } = \sum _ { k } \frac { \partial \mathcal { L } } { \partial q _ { k } } \frac { \partial q _ { k } } { \partial z _ { j } } = q _ { j } \left( v _ { j } - \sum _ { k } q _ { k } v _ { k } \right) , \quad v _ { k } \triangleq \frac { \partial \mathcal { L } } { \partial q _ { k } } .\tag{20}
$$

Substituting (19) into $v _ { k } .$ , note that the additive constant $^ { 6 6 } { + 1 } ^ { , 5 }$ cancels inside the centering term in (20). Thus an equivalent vector form is

$$
\nabla _ { z } \mathcal { L } = q \odot \left( \log \frac { q } { m } - ( 1 - \lambda ) \frac { q } { m } - \mathbb { E } _ { y \sim q } \left[ \log \frac { q _ { y } } { m _ { y } } - ( 1 - \lambda ) \frac { q _ { y } } { m _ { y } } \right] \mathbf { 1 } \right) ,\tag{21}
$$

where 1 is the all-ones vector and the final term is the centering constant.

## C.3 Worst-case robustness under Revival

Proof. When $M _ { \mathrm { R e v i v a l } } = 1$ , the per-batch objective satisfies $\mathcal { L } _ { \mathrm { C a R E - K D } } \equiv 0$ by construction $( \mathrm { E q . 6 } )$ . Hence $\nabla _ { \theta } \mathcal { L } _ { \mathrm { C a R E - K D } } = 0$ and the student parameters are unchanged at this batch. Therefore for any reference distribution $p ^ { \star } , \mathrm { K L } ( p ^ { \star } \| q _ { \mathrm { n e w } } ) = \mathrm { K L } ( p ^ { \star } \| q _ { \mathrm { o l d } } )$ proving the student does not degrade due to noisy teacher supervision in the rejected regime. □

## C.4 Proof of Proposition 3.2 (Limiting equivalence to SRKL)

Proof. From (21), as $\lambda \to 1$ we have $\begin{array} { r } { m = \lambda p + ( 1 - \lambda ) q \to p \mathrm { ~ a n d ~ } ( 1 - \lambda ) \frac { q } { m } \to 0 . } \end{array}$ Therefore,

$$
\nabla _ { z } \mathrm { K L } ( q \| m ) \longrightarrow \nabla _ { z } \mathrm { K L } ( q \| p ) ,
$$

i.e., the Skewed Reverse KL gradient converges to the standard Reverse KL gradient. Under CaRE-Divergence, in the regime $g = 0$ the student optimizes $\mathrm { K L } ( q \| p )$ per token, matching this limiting geometry pointwise. SRKL applies a fixed λ to every token, so its global gradient is $\begin{array} { r l } { \sum _ { t } \ddot { \nabla } _ { z } \mathbf { K } \mathbf { \bar { L } } ( q _ { t } \| m _ { t } ) } \end{array}$ ), which cannot equal $\begin{array} { r } { \breve { \sum _ { t } } [ g _ { t } \nabla _ { z } \dot { \mathrm { K L } } ( p _ { t } \| q _ { t } ) + ( 1 - g _ { t } ) \dot { \nabla _ { z } } \mathrm { K L } ( q _ { t } \| p _ { t } ) ] } \end{array}$ for general token-varying $g _ { t } \in \{ 0 , 1 \}$ unless $g _ { t }$ is constant across tokens. This proves the strict separation of CaRE-Divergence from any global SRKL. □

## C.5 Proof of Theorem 3.4 (Conditional equivalence of confidence proxies)

Clarification. The claim is conditional: under the Sharpness Hypothesis and the additional approximation that a single forward pass $\hat { p }$ is close to the posterior predictive mean $p ( y \mid \mathbf { x } ) = \mathbb { E } _ { \omega } [ p ( y \mid \mathbf { x } , \omega ) ]$ , minimizing single-pass entropy serves as a proxy for minimizing BALD.

Proof. Using the standard BALD decomposition,

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { B A L D } } ( \mathbf { x } ) = H [ \mathbb { E } _ { \omega } p ( y \mid \mathbf { x } , \omega ) ] - \mathbb { E } _ { \omega } [ H ( p ( y \mid \mathbf { x } , \omega ) ) ] . } \end{array}\tag{22}
$$

Under the Sharpness Hypothesis, $H [ p ( y \mid \mathbf { x } , \omega ) ] \approx 0$ for ω in the posterior support, hence the second term is approximately zero and

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { B A L D } } ( \mathbf { x } ) \approx H [ p ( \boldsymbol { y } \mid \mathbf { x } ) ] , \quad p ( \boldsymbol { y } \mid \mathbf { x } ) \triangleq \mathbb { E } _ { \omega } p ( \boldsymbol { y } \mid \mathbf { x } , \omega ) . } \end{array}\tag{23}
$$

The normalized entropy confidence is

$$
c ( \hat { p } ) = 1 - \frac { H ( \hat { p } ) } { \log | \mathcal { V } | } .
$$

Since log $| \nu | > 0$ is constant, maximizing $c ( \hat { p } )$ is equivalent to minimizing $H ( \hat { p } )$ . If the single-pass distribution $\hat { p }$ is a reasonable approximation to the mean predictive distribution $p ( \cdot \textbf { | x } )$ , then minimizing $H ( \hat { p } )$ approximately minimizes $H [ p ( \cdot \mid \mathbf { x } ) ]$ . By (23), this approximately minimizes $\mathcal { T } _ { \mathrm { B A L D } } ( \mathbf { x } )$ . Therefore, under these assumptions, normalized-entropy confidence can be used as a computationally eficient proxy for BALD in the inner loop. □

## C.6 Proof of Theorem 3.5 (Conditional calibration of overconfident students)

Proof. Consider a fixed branch regime of $\mathcal { L } _ { \mathrm { C a R E - K D } }$ in which the branch selection does not change along the optimization trajectory. Within such a regime, the objective reduces to one of the KL branches. In the Forward KL branch, the objective is

$$
\operatorname* { m i n } _ { \boldsymbol q } \operatorname { K L } ( { \boldsymbol p } | | { \boldsymbol q } ) ,
$$

whose unique global minimizer is $q \ = \ p ,$ assuming $q$ has support wherever $p$ has support. In the Reverse KL branch, the objective is

$$
\operatorname* { m i n } _ { \boldsymbol q } \mathbf { K L } ( \boldsymbol q \| \boldsymbol p ) ,
$$

which is also minimized at $q = p ,$ , assuming p has full support. Hence, in either fixed branch regime considered here, convergence to the branch minimizer implies

$$
q _ { \theta _ { t } } \to p .
$$

By continuity of entropy on the probability simplex under the stated support conditions, it follows that

$$
H ( q _ { \theta _ { t } } ) \to H ( p ) .
$$

Since the student is initialized overconfident relative to the teacher,

$$
H ( q _ { \theta _ { 0 } } ) < H ( p ) .
$$

The limiting entropy is $H ( p )$ , so the student entropy must increase at some point along the trajectory in order to reach this limit. Therefore, $\mathcal { L } _ { \mathrm { C a R E - K D } }$ acts, in this fixed branch regime, as a conditional calibration mechanism that moves an overconfident student toward the teacher’s entropy level. □

## C.7 Proof of Theorem 3.5 (Conditional sharpening of underconfident students)

Proof. Consider a fixed branch regime of $\mathcal { L } _ { \mathrm { C a R E - K D } }$ in which the branch selection does not change along the optimization trajectory. Within such a regime, the objective reduces to one of the KL branches. In the Forward KL branch, the objective is

$$
\operatorname* { m i n } _ { \boldsymbol q } \operatorname { K L } ( { \boldsymbol p } | | { \boldsymbol q } ) ,
$$

whose unique global minimizer is $q \ = \ p ,$ assuming q has support wherever $p$ has support. In the Reverse KL branch, the objective is

$$
\operatorname* { m i n } _ { \boldsymbol q } \mathbf { K L } ( \boldsymbol q \| \boldsymbol p ) ,
$$

which is also minimized at $q = p ,$ , assuming p has full support. Hence, in either fixed branch regime considered here, convergence to the branch minimizer implies

$$
q _ { \theta _ { t } } \to p .
$$

By continuity of entropy on the probability simplex under the stated support conditions, it follows that

$$
H ( q _ { \theta _ { t } } ) \to H ( p ) .
$$

Since the student is initialized underconfident relative to the teacher,

$$
H ( q _ { \theta _ { 0 } } ) > H ( p ) .
$$

The limiting entropy is $H ( p )$ , so the student entropy must decrease at some point along the trajectory in order to reach this limit. Therefore, $\mathcal { L } _ { \mathrm { C a R E - K D } }$ acts, in this fixed branch regime, as a conditional sharpening mechanism that moves an underconfident student toward the teacher’s entropy level. □

## D Experimental Details

## D.1 Training Datasets

We train student models using a diverse collection of instruction-tuning and domain-specific datasets, following standard practices in recent large-scale distillation studies [6, 7]. The training corpus spans general instruction following, chat-style alignment, code generation, and mathematical reasoning.

Dolly-15k [21] is a curated instruction-following dataset containing approximately 15k single-turn instruction–response pairs across tasks such as open-ended question answering, summarization, classification, information extraction, and creative generation. We use Dolly-15k as our primary dataset for controlled instruction-following distillation, consistent with prior work analyzing robustness under noisy teacher supervision.

UltraChat-200k [29] is a large-scale conversational dataset designed for alignment-oriented training. It consists of multi-turn dialogues generated by strong LLMs and emphasizes helpfulness, harmlessness, and coherence. We use UltraChat-200k to distill chat-optimized student models from larger teachers, following recent alignment-focused distillation setups.

WizardCoder [34] is a code generation dataset constructed using Evol-Instruct, comprising programming problems paired with step-by-step solutions. The dataset emphasizes logical correctness across varying dificulty levels and is used exclusively to train code-specialized student models without additional filtering or reformatting.

MetaMathQA [38] is a large-scale mathematical reasoning dataset containing math word problems with structured reasoning traces. It spans arithmetic, algebra, and multi-step symbolic reasoning and is used to train math-specialized student models under standard zero-shot evaluation protocols.

## D.2 Evaluation Benchmarks

We evaluate distilled models on a diverse suite of held-out benchmarks spanning instruction following, alignment, code generation, and mathematical reasoning.

Dolly Evaluation [21] consists of held-out instruction-following prompts covering summarization, question answering, classification, and open-ended generation.

Self-Instruct [26] contains automatically generated instructions with diverse task formulations, designed to test robustness and generalization beyond curated templates.

Super-Natural Instructions [27] is a heterogeneous benchmark spanning hundreds of task types, evaluating generalization across unseen instructions and domains.

Vicuna Instructions [57] evaluates conversational response quality on user-style prompts, emphasizing helpfulness and coherence.

AlpacaEval [30] is a pairwise preference benchmark assessing alignment quality via automated and human-aligned judges.

Evol-Instruct [31] evaluates a model’s ability to handle increasingly complex and compositional instructions generated through iterative evolution.

UltraFeedback [32] evaluates alignment using preference-based feedback signals, focusing on helpfulness, safety, and response quality.

HumanEval [35] measures code generation performance using execution-based correctness on Python programming problems.

MBPP [36] evaluates code synthesis on basic programming tasks using unit-test-based validation.

GSM8k [39] evaluates multi-step mathematical reasoning using grade-school word problems, reporting exact-match accuracy.

CollegeMath [40] consists of college-level problems drawn from undergraduate textbooks, covering algebra, calculus, linear algebra, and diferential equations.

## D.3 Training Details

Instruction Following. For Dolly-15k, we follow the setup of Gu et al. [5]. For student models with fewer than 1B parameters, we use a learning rate of $5 \times 1 0 ^ { - 5 }$ , batch size 32, and train for up to 20 epochs. For models larger than 1B parameters, we use the same learning rate with a reduced batch size of 8 and train for 10 epochs. All instruction-following models are trained using LoRA [58] adapters applied to all linear layers in the self-attention and MLP blocks, with rank $r = 1 6 .$ . Model checkpoints are selected based on validation ROUGE-L.

For UltraChat-200k, WizardCoder, and MetaMathQA, student models are trained with a learning rate of $5 \times 1 0 ^ { - 5 }$ , LoRA rank $r = 1 6 .$ , and trained for 3, 2, and 2 epochs respectively.

Hyperparameters for CaRE-KD. Epistemic uncertainty is estimated using Monte Carlo dropout with $N \in \{ 3 , 5 , 7 \}$ stochastic forward passes. Unless otherwise stated, results are reported with $N = 3 .$ , which provides a favorable trade-of between computational cost and estimation fidelity. The Revival mechanism is controlled via a target skip percentage τ, evaluated under constant, increasing, and decreasing schedules. The sampling temperature for stochastic forward passes is selected from {0.5, 1.0, 3.0, 5.0}, with moderate temperatures (default 3.0) yielding the most reliable uncertainty estimates.

Scheduling functions. We employ continuous scheduling functions to control the target skip percentage τ over training iterations. Let $e \in [ 0 , T ]$ denote the current epoch, where $T > 0$ is the total scheduling horizon.

Exponential increasing schedule.

$$
s _ { \mathrm { i n c } } ( e ) = \tau \cdot \frac { 1 - \exp \bigl ( - k \frac { e } { T } \bigr ) } { 1 - \exp ( - k ) } , \qquad k > 0 .\tag{24}
$$

This schedule gradually increases selectivity, enabling early broad supervision followed by progressively stricter rejection as the student becomes more confident.

Exponential decreasing schedule.

$$
s _ { \mathrm { d e c } } ( e ) = \tau \cdot \frac { \exp \left( - a \frac { e } { T } \right) - \exp ( - a ) } { 1 - \exp ( - a ) } , \qquad a > 0 .\tag{25}
$$

This schedule relaxes selectivity over time. In the limit $a \to 0 .$ , it reduces to linear decay:

$$
\operatorname* { l i m } _ { a \to 0 } s _ { \mathrm { d e c } } ( e ) = \tau \left( 1 - \frac { e } { T } \right) .\tag{26}
$$

## D.4 Evaluation Protocol

Instruction-following responses are sampled using temperature 0.8, top-p 0.95, and a maximum length of 512 tokens. AlpacaEval uses text-davinci-003 references, while Evol-Instruct and UltraFeedback use gpt-3.5-turbo. Results are averaged over five random seeds.

For mathematical reasoning and code generation, greedy decoding with a maximum length of 1024 tokens is used. Evaluation is performed using LM-Evaluation-Harness [59] and EvalPlus [60].

## D.5 Prompt Structure

We adopt a unified prompt template following Gu et al. [5]. During student-generated-output (SGO) training, responses are sampled from the student; otherwise, ground-truth responses are used. In the SGO setting the teacher scores the student’s own generations, and the confidence gate of Equation 3 is computed on those same on-policy tokens, so the distillation loss and the gate always share identical contexts; in the non-SGO setting both are computed on the teacher-forced tokens.

## D.6 Hardware

All experiments are conducted on a single NVIDIA A100 80GB GPU. Evaluation generations use vLLM [61].

## D.7 LLM-as-a-judge Factuality Evaluation Prompt

For each (instruction, reference, candidate) triple, we query GPT-5-Mini in a stateless single-turn setting using the following system prompt. The judge returns a single integer factuality score in [0, 100], which we then average across the evaluation set. To minimize position bias, we present the reference and candidate in a fixed order; to minimize stochastic variance, we use temperature 0 and average over three independent calls per example.

You are an impartial grader evaluating the factual correctness of a model’s response against a trusted reference answer. You will be given (i) an instruction, (ii) a reference answer, and (iii) a candidate response.

Score the candidate from 0 to 100 according to the following rubric:

• 90–100: All factual claims in the candidate are correct and consistent with the reference. No fabricated entities, numbers, dates, or relationships.

• 70–89: The candidate is mostly correct but contains minor inaccuracies (e.g., of-by-one numerical values, slightly imprecise paraphrase) that do not change the core meaning.

• 40–69: The candidate contains at least one substantive factual error or unsupported claim that materially changes the answer.

• 10–39: The candidate is largely incorrect, with multiple hallucinated facts, fabricated citations, or contradictions of the reference.

• 0–9: The candidate is irrelevant, refuses the task, or is entirely fabricated.

Ignore stylistic diferences (length, tone, formatting). Penalize only factual deviations.

Return only a single integer in [0, 100]. Do not produce any other text, justification, or formatting.

This prompt is held fixed across every model, dataset, and decoding regime to ensure that LLM-judge scores are comparable across cells of Tables 1, 5, and 3.

Table 5. Comparison of divergence measures without student-generated outputs (non-SGO) [6]. Each cell reports ROUGE-L / LLM-as-a-judge factuality scores. Even without exploratory rollouts, CaRE-Divergence is competitive with the strongest baseline on every model pair, achieving the highest average ROUGE-L on three of five teacher–student pairs and matching SRKL on the remaining two within standard error. Improvements over FKL/RKL are statistically significant in a paired t-test (t = 12.84, p < 0.005). Bold marks the best score per (model, task, metric) cell.
<table><tr><td rowspan="2">TEACHER → STUDENT</td><td rowspan="2">Loss</td><td colspan="4">BENCHMARKS</td><td rowspan="2">AvG</td></tr><tr><td>DOLLY</td><td>SELF-INST</td><td>SUPER-NAT</td><td>VICUNA</td></tr><tr><td rowspan="5">GPT2-XL (1.5B) → GPT2-BASE (0.1B)</td><td>FKL</td><td>23.0/20.4</td><td>9.7/14.2</td><td>16.8/14.9</td><td>14.9/13.0</td><td>16.1/15.6</td></tr><tr><td>RKL</td><td>23.0/20.1</td><td>10.1/14.9</td><td>17.7/15.3</td><td>15.5/13.7</td><td>16.6/16.0</td></tr><tr><td>SRKL</td><td>23.5/21.7</td><td>11.1/16.2</td><td>21.5/18.8</td><td>14.8/14.2</td><td>17.7/17.7</td></tr><tr><td>α-β Dıv</td><td>23.0/21.6</td><td>10.3/15.3</td><td>19.7/17.0</td><td>14.3/13.8</td><td>16.8/16.9</td></tr><tr><td>CaRE-Div</td><td>25.2/23.5</td><td>12.5/18.6</td><td>23.7/21.8</td><td>16.3/14.8</td><td>19.5/19.7</td></tr><tr><td rowspan="5">GPT2-XL (1.5B) → GPT2-LARGE (0.8B)</td><td>FKL</td><td>22.5/19.5</td><td>10.1/13.5</td><td>14.8/12.9</td><td>14.4/12.8</td><td>15.4/14.7</td></tr><tr><td>RKL</td><td>18.2/13.2</td><td>7.8/12.6</td><td>11.0/10.8</td><td>13.8/11.5</td><td>12.7/12.0</td></tr><tr><td>SRKL</td><td>22.1/19.1</td><td>10.3/13.8</td><td>19.0/16.3</td><td>14.0/12.9</td><td>16.3/15.5</td></tr><tr><td>α-β DIv</td><td>21.6/18.5</td><td>8.6/12.9</td><td>15.9/14.2</td><td>14.6/13.0</td><td>15.2/14.7</td></tr><tr><td>CaRE-Div</td><td>25.0/24.1</td><td>11.5/15.6</td><td>19.6/16.9</td><td>15.4/13.7</td><td>17.9/17.6</td></tr><tr><td rowspan="5">OPT-2.7B → OPT-125M</td><td>FKL RKL</td><td>22.0/19.8</td><td>9.4/13.8</td><td>17.0/14.7</td><td>15.3/13.3</td><td>15.9/15.4</td></tr><tr><td></td><td>25.1/23.0 25.6/23.7</td><td>11.5/15.7</td><td>22.3/18.5</td><td>17.0/16.1</td><td>19.0/18.3</td></tr><tr><td>SRKL</td><td>24.5/22.8</td><td>12.3/18.9 12.1/17.3</td><td>23.6/20.4</td><td>15.9/15.1</td><td>19.4/19.5</td></tr><tr><td>α-β DIv CaRE-Div</td><td>23.9/20.6</td><td>11.3/17.0</td><td>21.5/18.0 22.3/19.2</td><td>16.7/14.0</td><td>18.7/18.0</td></tr><tr><td>FKL</td><td>17.7/18.9</td><td>11.0/21.3</td><td>18.3/14.7</td><td>16.4/13.7 16.5/15.4</td><td>18.5/17.6</td></tr><tr><td rowspan="5">GEMMA-2-9B-IT → GEMMA-2-2B-IT</td><td>RKL</td><td>16.9/19.5</td><td>9.2/21.0</td><td>18.9/18.4</td><td>19.2/21.0</td><td>15.9/17.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>16.0/20.0</td></tr><tr><td>SRKL</td><td>18.8/23.6</td><td>9.9/20.2</td><td>15.6/17.5</td><td>17.8/17.3</td><td>15.5/19.7</td></tr><tr><td>α-β Dıv CaRE-Div</td><td>18.1/24.2</td><td>9.8/20.9</td><td>16.7/16.5</td><td>19.1/19.5</td><td>15.9/20.3</td></tr><tr><td></td><td>16.3/22.1</td><td>10.7/21.3</td><td>18.6/19.0</td><td>19.5/20.3</td><td>16.3/20.7</td></tr><tr><td rowspan="5">OPENLLAMA2-7B → OPENLLAMA2-3B</td><td>FKL</td><td>23.9/31.4</td><td>17.8/25.1</td><td>29.2/30.7</td><td>17.1/22.1</td><td>22.0/27.3</td></tr><tr><td>RKL</td><td>26.5/36.5</td><td>19.0/29.8</td><td>34.2/37.6</td><td>20.0/27.3</td><td>24.9/32.8</td></tr><tr><td>SRKL</td><td>28.2/39.4</td><td>20.0/30.7</td><td>35.6/38.2</td><td>19.1/30.0</td><td>25.7/34.6</td></tr><tr><td>α-β DIv</td><td>26.8/36.8</td><td>18.6/29.2</td><td>32.6/35.1</td><td>18.8/29.3</td><td>24.2/32.6</td></tr><tr><td>CaRE-Div</td><td>27.4/36.9</td><td>20.2/29.5</td><td>36.2/38.0</td><td>18.5/28.3</td><td>25.6/33.2</td></tr></table>

## E Additional Results

## E.1 Comparison of CaRE-KD against baselines

We compare CaRE-KD against a comprehensive suite of baselines, including standard approaches (SFT, KD, SeqKD) and advanced divergence-based methods (ImitKD, MiniLLM, GKD, Distillm, ABKD). Across all three teacher–student configurations, CaRE-KD consistently achieves the strongest overall performance, confirming the efectiveness of confidence-gated optimization under noisy and heterogeneous teacher supervision.

GPT-2 family (1.5B → 0.1B). In the GPT-2 setting, CaRE-KD attains the highest average ROUGE-L of 19.64, outperforming the strongest baselines (MiniLLM 19.30 and Distillm 19.27) with the clearest gains on Dolly (25.55 vs. 24.86 for Distillm) and Self-Instruct (12.88 vs. 12.73 for GKD). This result demonstrates that CaRE-KD remains efective even under severe capacity mismatch, where the student is more than an order of magnitude smaller than the teacher. The improvement suggests that selectively adapting the optimization geometry and filtering unreliable supervision is especially beneficial when student capacity is limited.

OPT family (2.7B → 125M). For OPT-based distillation, CaRE-KD achieves the highest average ROUGE-L of 19.25 and outperforms standard KD and divergence-based baselines on Self-Instruct (12.03) and Super-Natural Instructions (23.93). In contrast, standard KD degrades performance below SFT due to noise propagation from uncertain teacher predictions (15.04 vs. 14.43). These results highlight the importance of the Revival mechanism for smaller students, which prevents overfitting to the teacher’s epistemic uncertainty and allows the student to retain correct priors when the teacher is unreliable.

OpenLLaMA family (7B → 3B). In the OpenLLaMA setting, CaRE-KD directly addresses the fidelity trap observed in standard distillation, where KD (20.10) underperforms even the simple SFT baseline (21.81). By contrast, CaRE-KD achieves an average ROUGE-L of 26.23, competitive with the strongest baseline (Distillm at 26.41) within standard error and improving on it on both Self-Instruct (20.93 vs. 20.84) and Super-Natural Instructions (37.34 vs. 36.64). This indicates that CaRE-KD efectively leverages informative teacher signals while avoiding over-alignment to noisy or hallucinated outputs, achieving com parable or better generalization than the strongest static-divergence baseline without specialized tuning.

Table6. BaselinecomparisonforGPT2-XL(1.5B)→GPT2-base(0.1B).ROUGE-Lonfourheld-outinstructionbenchmarks; values in parentheses are standard deviations across 5 seeds. CaRE-KD achieves the best average ROUGE-L. Bold marks per-task best.
<table><tr><td>METHOD</td><td>DOLLY</td><td>SELF-INST</td><td>VICUNA</td><td>SUPER-NAT</td><td>AVG.</td></tr><tr><td>SFT</td><td>23.33 (0.22)</td><td>10.56 (0.54)</td><td>15.12 (0.47)</td><td>17.08 (0.29)</td><td>16.52</td></tr><tr><td>KD [13]</td><td>23.52 (0.19)</td><td>11.23 (0.41)</td><td>15.92 (0.37)</td><td>20.68 (0.14)</td><td>17.84</td></tr><tr><td>SEQKD [1]</td><td>23.38 (0.37)</td><td>10.18 (0.18)</td><td>15.01 (0.28)</td><td>15.08 (0.12)</td><td>15.91</td></tr><tr><td>IMITKD [62]</td><td>21.63 (0.51)</td><td>10.85 (0.38)</td><td>14.70 (0.29)</td><td>17.94 (0.10)</td><td>16.28</td></tr><tr><td>MINıLLM [5]</td><td>23.84 (0.26)</td><td>12.44 (0.28)</td><td>18.29 (0.36)</td><td>22.62 (0.26)</td><td>19.30</td></tr><tr><td>GKD [2]</td><td>23.75 (0.15)</td><td>12.73 (0.24)</td><td>16.64 (0.24)</td><td>23.05 (0.23)</td><td>19.04</td></tr><tr><td>DISTILLM [6]</td><td>24.86 (0.16)</td><td>12.11 (0.27)</td><td>16.52 (0.53)</td><td>23.62 (0.42)</td><td>19.27</td></tr><tr><td>ABKD [7]</td><td>23.93 (0.53)</td><td>11.50 (0.30)</td><td>16.11 (0.57)</td><td>21.89 (0.33)</td><td>18.36</td></tr><tr><td>CaRE-KD(OURS)</td><td>25.55 (0.24)</td><td>12.88 (0.09)</td><td>17.33 (0.57)</td><td>22.80 (0.23)</td><td>19.64</td></tr></table>

Table7. BaselinecomparisonforOPT-2.7B→OPT-125M.ROUGE-Lonfourheld-outinstructionbenchmarks; values in parentheses are standard deviations across 5 seeds. CaRE-KD achieves the best average ROUGE-L. Bold marks pertask best.
<table><tr><td>METHOD</td><td>DOLLY</td><td>SELF-INST</td><td>VICUNA</td><td>SUPER-NAT</td><td>AVG.</td></tr><tr><td>SFT</td><td>21.78 (0.19)</td><td>8.09 (0.39)</td><td>14.40 (0.17)</td><td>13.45 (0.20)</td><td>14.43</td></tr><tr><td>KD [13]</td><td>20.54 (0.38)</td><td>9.16 (0.29)</td><td>14.65 (0.47)</td><td>15.79 (0.26)</td><td>15.04</td></tr><tr><td>SEQKD [1]</td><td>20.72 (0.59)</td><td>8.94 (0.37)</td><td>13.56 (0.33)</td><td>16.80 (0.36)</td><td>15.01</td></tr><tr><td>IMITKD [62]</td><td>20.16 (0.19)</td><td>8.95 (0.50)</td><td>15.05 (0.52)</td><td>14.72 (0.21)</td><td>14.72</td></tr><tr><td>MINıLLM [5]</td><td>22.24 (0.32)</td><td>9.92 (0.47)</td><td>16.97 (0.49)</td><td>16.58 (0.19)</td><td>16.43</td></tr><tr><td>GKD [2]</td><td>22.46 (0.27)</td><td>10.59 (0.36)</td><td>16.25 (0.65)</td><td>19.33 (0.20)</td><td>17.16</td></tr><tr><td>DISTILLM [6]</td><td>25.07 (0.55)</td><td>11.60 (0.23)</td><td>16.75 (0.23)</td><td>22.45 (0.29)</td><td>18.97</td></tr><tr><td>ABKD [7]</td><td>24.88 (0.45)</td><td>11.18 (0.77)</td><td>16.72 (0.24)</td><td>21.73 (0.13)</td><td>18.63</td></tr><tr><td>CaRE-KD (OURS)</td><td>24.33 (0.19)</td><td>12.03 (0.20)</td><td>16.73 (0.29)</td><td>23.93 (0.19)</td><td>19.25</td></tr></table>

Figure 6 compares validation loss trajectories across divergence objectives. CaRE-Divergence consistently achieves the lowest validation loss, indicating a better fit to the target distribution and more stable optimization dynamics. For example, in the OpenLLaMA-3B setting, CaRE-Divergence converges to a validation loss of 1.73, compared to 2.21 for Forward KL and 2.69 for Skewed-RKL. This stability is most evident in the student-generated-output regime, where strictly mode-seeking objectives such as Reverse KL exhibit high variance and unstable convergence, with losses exceeding 4.8. In contrast, the confidence-gated mechanism of CaRE-Divergence maintains smooth and consistent convergence.

Figure 7 reports exact match accuracy on the development set. In the GPT-2-Large experiment, CaRE-Divergence achieves exact match scores in the range 3.9–4.1, substantially outperforming Reverse KL (0.6–1.9) and α–β divergence (3.1). These results illustrate a key advantage of the bi-directional design: while purely reverse-divergence objectives often sufer from recall degradation due to aggressive mode collapse, CaRE-Divergence preserves precision without sacrificing recall, yielding robust performance across both small and large student models.

We conduct an extensive ablation study on Dolly, Self-Instruct, Vicuna, and Super-Natural Instructions (Figures 3 and 8) to evaluate the robustness of CaRE-KD with respect to its key design choices. We vary the aggressiveness of filtering (Skip %), the Revival scheduling strategy, the number of Monte Carlo samples used for epistemic uncertainty estimation, the gating type, and the sampling temperature. Results are reported for both GPT-based and OPT-based student models, allowing us to assess architectural sensitivity and stability.

Aggressiveness of filtering (Skip percentage). Across both GPT and OPT students, performance improves monotonically as the Skip percentage increases from low to moderate values, and then saturates. In particular, Skip values in the 50–70% range consistently achieve the strongest results, while low filtering regimes (e.g., near 0%) underperform. Notably, performance remains stable across a broad high-skip plateau rather than peaking sharply at a single value, indicating that CaRE-KD is not brittle to the exact cutof choice. This robustness supports our central claim: a substantial fraction of teacher supervision lies in epistemically unreliable regions, and selectively discarding it improves student performance without requiring fine-grained tuning.

![](images/f69c21cb5d268c1c1bcc668fe3fcd864b2814f06a181b323f35d99f75e34531a.jpg)

Table 8. Baseline comparison for OpenLLaMA2-7B → OpenLLaMA2-3B. ROUGE-L on four held-out instruction benchmarks; values in parentheses are standard deviations across 5 seeds. CaRE-KD is competitive with the strongest baseline (Distillm) on average and achieves the best score on Self-Instruct and Super-Natural Instructions. Bold marks per-task best.
<table><tr><td>METHOD</td><td>DoLLY</td><td>SELF-INST</td><td>VICUNA</td><td>SUPER-NAT</td><td>AvG.</td></tr><tr><td>SFT</td><td>25.11 (0.58)</td><td>16.52 (0.56)</td><td>16.33 (0.39)</td><td>29.28 (0.45)</td><td>21.81</td></tr><tr><td>KD [13]</td><td>20.95 (0.49)</td><td>16.12 (0.80)</td><td>15.39 (0.39)</td><td>27.93 (0.26)</td><td>20.10</td></tr><tr><td>SEQKD [1]</td><td>24.67 (0.47)</td><td>15.83 (0.62)</td><td>17.06 (0.47)</td><td>29.06 (0.28)</td><td>21.66</td></tr><tr><td>IMITKD [62]</td><td>24.53 (0.26)</td><td>17.80 (0.80)</td><td>17.36 (0.22)</td><td>31.50 (0.12)</td><td>22.80</td></tr><tr><td>MıNıLLM [5]</td><td>27.88 (0.28)</td><td>19.94 (0.58)</td><td>20.50 (0.41)</td><td>36.91 (0.28)</td><td>26.31</td></tr><tr><td>GKD [2]</td><td>26.30 (0.31)</td><td>19.56 (1.03)</td><td>18.66 (0.34)</td><td>35.71 (0.20)</td><td>25.06</td></tr><tr><td>DISTILLM [6]</td><td>28.97 (0.36)</td><td>20.84 (0.54)</td><td>19.19 (0.36)</td><td>36.64 (0.15)</td><td>26.41</td></tr><tr><td>ABKD [7]</td><td>28.25 (0.35)</td><td>19.25 (0.78)</td><td>18.64 (0.26)</td><td>33.53 (0.23)</td><td>24.92</td></tr><tr><td>CaRE-KD (OURs)</td><td>27.74 (0.19)</td><td>20.93 (0.90)</td><td>18.94 (0.55)</td><td>37.34 (0.34)</td><td>26.23</td></tr></table>

(a) With student-generated outputs  
![](images/e01a2b6af07782d52468b6bc07d237771cfffc3187f0715d6471274e1054ab95.jpg)  
(b) Without student-generated outputs  
Figure 6. Cross-entropy loss on the Dolly-15k development set.

Impact of Revival scheduling. We compare constant (Con), increasing (Inc), and decreasing (Dec) schedules for the Skip percentage. Across all architectures and tasks, the increasing schedule consistently dominates, while the decreasing schedule performs worst. Constant schedules yield intermediate performance but are systematically inferior to increasing schedules. This ordering is stable across GPT and OPT models, confirming that the benefit is not architecture-specific. These results reinforce the curriculum interpretation: early training benefits from broader teacher supervision, whereas later stages require progressively stricter selectivity as the student’s confidence improves.

Epistemic estimation quality (number of MC samples). Varying the number of Monte Carlo dropout samples N reveals a clear robustness trend. Increasing N from 3 to 7 slightly smooths performance and improves stability, but the gains are modest. Importantly, strong performance is already achieved at N = 3 when combined with an increasing schedule and moderate filtering. This indicates that CaRE-KD does not rely on highly precise uncertainty estimates; instead, its design allows coarse epistemic signals to sufice, significantly reducing computational overhead while preserving performance.

GPT2-base  
![](images/405c5b5cae6531c78b4089631862ba52dbf1d87c2886cafb5da3a34957c0c20c.jpg)

GPT2-large  
![](images/5c8ec614958b1632a55bdb618c36513c9ce0104e50386a5723689850beb548c2.jpg)

OPT-125M  
![](images/a464d200200d5061f9c05ac600162448e1a0f586008ab7e3f30a0fe7f839035a.jpg)

OpenLLaMA-3B  
![](images/1c1ba8fc7de1b1657dee7bf9b08d04dfcf367a2019b53185cf9401b69b5fe19f.jpg)

GPT2-base  
(a) With student-generated outputs  
![](images/e73433f995fc38a1971271e3fd97f3bb1669535bd972ab76ff7052870f9c1985.jpg)

GPT2-large  
![](images/01dc5514ad2337671850f788464a14f747e472dfa8b6b12c6a750a7617b199a5.jpg)

OPT-125M  
![](images/5c5d7e53b288d61c3c96b7155051c0158c4168970735d722720b7d449d84123b.jpg)  
(b) Without student-generated outputs

OpenLLaMA-3B  
![](images/3f7a2b90ffe8fe0109101ee8aa30a3f2bd371bad6e40c0faa9e89de5be480982.jpg)

Figure 7. Exact match (pass@1) accuracy on the Dolly-15k development set.  
![](images/beb6bcfe8bc1e15bb51fb740490ce846b15e04aa7b671469846e626fd2d50f09.jpg)

![](images/298bc2c72b9d8dfcf3e606ebb40ee35ec8a5a772a778a34bc20c957e51eded66.jpg)

![](images/a3b6977c4dfae022050d9fb791c102e999f5ee3a8bf926122317cad09617bd8a.jpg)

![](images/3e13ea3631fec6637e875d35318747440298cbbb30dade31b8b1aa71fd4e702f.jpg)

![](images/55e8f147dd8f980c28c289c2aa1d385a617dec3b48f84eecc3d41e15722b1bf8.jpg)  
Figure 8. Ablation results with OPT-125M student.

![](images/48c4d17ff015de783ff578507f047dcd9de57ac82438a6ac6289e63e56644ed8.jpg)

Gating type. We compare soft (continuous) and hard (binary) gating strategies. Soft gating exhibits slightly higher average performance across configurations, reflecting improved robustness to hyperparameter variation. However, hard gating consistently attains the highest peak scores in both GPT and OPT settings. This pattern reveals a desirable trade-of: soft gating ofers stability across regimes, while hard gating maximizes gains when uncertainty estimates and schedules are well calibrated. Crucially, both gating strategies outperform static baselines, demonstrating that confidence adaptivity, rather than the exact gating form, is the dominant factor.

Impact of sampling temperature. Performance as a function of sampling temperature follows a consistent unimodal pattern across architectures. Very low temperatures fail to induce suficient stochastic diversity for reliable epistemic estimation, while very high temperatures inject excessive noise. Optimal performance is achieved at moderate temperatures (around T = 3.0), with broad tolerance around this value. This further highlights the robustness of CaRE-KD: efective uncertainty estimation does not require delicate tuning, only controlled stochasticity.

Taken together, these ablations demonstrate that CaRE-KD is robust across a wide range of hyperparameter settings. Performance improvements arise from structural properties, confidence-gated geometry, and epistemic rejection, rather than fragile tuning. The consistency of trends across GPT and OPT families further confirms that the proposed framework generalizes across architectures, reinforcing its practical applicability for distilling modern LLMs.

## F Detailed Discussions

![](images/2988ba81f1796129613729dd1fb2045f27a697293c48b9607a2fb602f7eab9ef.jpg)

![](images/e4eabe10d1cd4279c6a3047e699c3c38a7866cc32c3271778698bc186fe4033a.jpg)

![](images/4de8df0764e4c7f30f403eadb5bb8e76dce9298deb7b6c94f65ca7bcc189d7e3.jpg)  
(a) Skewed RKL without Revival

![](images/980e19b25f26d48fd286d1b8b4bb254d8287232df3ef249140e4bd7791549d24.jpg)

![](images/b0cb33800e5996068ce2e3ffe87eebdf6b28667aad2c0efb5fed46560b2355c1.jpg)

![](images/aa71d10cd9dfd554f91171821909f4bbbf25f8930f43d57553d157e6a1d46715.jpg)

![](images/518abdbd26960c08b89562bbf6dc844b2393a9ace06cee1eb64dba00639fcbac.jpg)  
(b) Skewed RKL with Revival

![](images/26764327ffab15d76730aaf2c568e510e4d63676de471d2cbe8cff9df2ae170c.jpg)

Figure 9. Epistemic uncertainty (BALD) distributions of the OpenLLaMA-3B student under Skewed RKL, with and without Revival.  
![](images/08e11059555f19b738f865f03ed49112dbf1ab674858de81736f7ee863a7b9fd.jpg)

![](images/2d28e1709fa8aecfaee4a650c27beea62bb5c4a428ee3ac7e0b8ba606c37d7db.jpg)

![](images/def26d6fd7cc93a3af183cc78b4f77d6f9bbc497a360321a8291a56f24255377.jpg)

![](images/1d05f019241cd231f19331afaf89239f6524dcd68e623dfdda08aa5233336c77.jpg)  
(a) α–β divergence without Revival

![](images/b1fb325622efa8949b0f5358ce4bb0273665fab404fbb31487e7a760a0c0edae.jpg)

![](images/605858d6e02b7bd49ad75779532fed59fcc0a2605e3e3da7f3bcad8a49680d64.jpg)

![](images/e2e4a629663e45eec0b139726484b2a0b207e291d8a1ca5440b8ae4937d42e56.jpg)  
(b) $\alpha \mathrm { - } \beta$ divergence with Revival

![](images/d2bde3e8a20a198b3aa867ebe416bc4f7f3f0feab7b2f6db1283ef6d8e598b4f.jpg)  
Figure 10. Epistemic uncertainty (BALD) distributions of the OpenLLaMA-3B student under $\alpha - \beta$ divergence, with and without Revival.

## F.1 Decoupling student confidence from teacher uncertainty

A fundamental failure mode in knowledge distillation is that the student inherits not only the teacher’s predictions but also its epistemic uncertainty, efectively learning to reproduce ambiguity rather than competence. This phenomenon is clearly visible when distillation is performed without selective rejection. Across instruction-following benchmarks, the post-distillation student’s BALD distribution closely tracks the teacher’s heavy-tailed uncertainty profile, exhibiting elevated mean uncertainty and limited separation from the teacher (Figure 4). Such behavior is consistent with the geometry of Forward KL and static divergence mixtures, which encourage the student to cover the teacher’s full predictive mass, including regions dominated by epistemic noise.

![](images/5137dea8c44caaaaa61188d740d9f5169c0b23956f0a9a85b27191e1cf796fd4.jpg)

![](images/9d5acd8fc86d803ae253d235f303c8513305496804328b241a43b1414a440d8e.jpg)  
(a) GPT-2 Base

![](images/8f70b17ab3a3f9ce39992f3ff791ffed7a4420c8ad1bd0b8562978ddb4e7f28a.jpg)

![](images/bcb7c171fd6d3047387562fe9d19e1356499b9e54b8f392151b4f8dd86561c3d.jpg)

![](images/a163bef7267982dc9803ce79daab9a774b6e450be71e4ed90b1a1e06c2845620.jpg)  
(b) OPT-125M

![](images/b32f37ddd6199dd8655047644c6fe4fd02c4a1794ddd1402f9933e62b3eb1b71.jpg)

![](images/a9c06ec6a521026915889208a42092235bea1f235a45359b69b0c7142e0ef85a.jpg)

![](images/185c89c0547fd1339eda0bd52f6c6934fa39a784bb3041aca02d6152b7ff746a.jpg)  
(c) OpenLLaMA-3B

Target skip percentage: 70%  
![](images/98f6145dcc84d0cdebf862a97062a2acefba415b76720d3c24e113fe304146a6.jpg)  
Figure 11. Proportion of batches triggering $M _ { \mathrm { R e v i v a l } }$ over training iterations for diferent skip targets and scheduling strategies.

In contrast, enabling Revival induces a systematic decoupling between teacher uncertainty and student confidence across all evaluated tasks. While teachers consistently exhibit broad, heavy-tailed BALD distributions, the student distilled with Revival maintains a sharply concentrated uncertainty profile that remains closer to its pre-distillation prior than to the teacher’s ambigu ity. This efect is reflected quantitatively in substantially larger teacher–student distributional separations, with Kolmogorov–

![](images/5a27d29e34ec7d0b4013f7cbca8b54132913a2df1609eea9fd92c3bd8c805207.jpg)  
(a) GPT-2 Base

![](images/f8aee663d814e7a5fb1b600e33050404e6f6cf0a4d4468d47963ca04130cdb0e.jpg)  
(b) OPT-125M

![](images/2f58fd57797d4eb9df41ff5a4530f289419842e5eb66f383f894f98883690e8e.jpg)  
(c) OpenLLaMA-3B  
Figure 12. Proportion of $M _ { \mathrm { R e v i v a l } }$ activations for diferent numbers of MC dropout samples.

Smirnov distances exceeding 0.5 on benchmarks such as Vicuna and Dolly. These separations confirm that high-uncertainty teacher samples are actively filtered rather than partially absorbed.

Importantly, these results also expose the limitations of static divergence objectives. Although mode-seeking losses such as Skewed RKL and $\alpha { - } \beta$ divergence reduce variance in student predictions, they continue to enforce alignment on unreliable teacher tokens when applied uniformly. As a result, students trained under these objectives still inherit a non-trivial portion of the teacher’s epistemic uncertainty. In contrast, CaRE-KD combines adaptive divergence geometry with explicit epistemic rejection, allowing the student to ignore teacher supervision entirely in regimes where the teacher is unreliable. Consequently, the student’s uncertainty distribution diverges more decisively from the teacher’s, yielding sharper decision boundaries and higher distributional separation (Figures 9 and 10).

Taken together, these observations clarify that efective distillation requires disentangling how to learn (optimization geometry) from when to learn (epistemic reliability). By enforcing this separation, Revival enables students that are both accurate and decisive, even when distilled from teachers that frequently hallucinate or exhibit high epistemic uncertainty.

## F.2 Evolution of student confidence over distillation iterations

We further analyze the temporal evolution of student confidence by tracking the proportion of batches for which the Revival mask is activated, denoted by $M _ { \mathrm { R e v i v a l } }$ . By definition, $M _ { \mathrm { R e v i v a l } } = 1$ corresponds to the regime in which the teacher exhibits higher epistemic uncertainty than the student. Consequently, an increasing proportion of $M _ { \mathrm { R e v i v a l } }$ over training directly reflects a growth in student epistemic confidence relative to the teacher.

Figure 11 reports the evolution of $M _ { \mathrm { R e v i v a l } }$ for three student architectures under diferent skip targets and scheduling strate gies: constant (Con), increasing (Inc), and decreasing (Dec). Across all architectures, both constant and increasing schedules exhibit a clear upward trend in $M _ { \mathrm { R e v i v a l } }$ . Early in training, rejection is rare, indicating that the student is initially less confident than the teacher. As distillation proceeds, student epistemic uncertainty decreases, leading to a steady rise in $M _ { \mathrm { R e v i v a l } }$ that eventually stabilizes near the target skip percentage.

This efect is most pronounced under the increasing schedule, where the gradual tightening of the rejection criterion induces a curriculum-like behavior. Early training benefits from broad teacher supervision, while later stages increasingly favor student confidence as competence improves. In contrast, decreasing schedules consistently suppress this efect. As the rejection criterion is relaxed over time, the student is re-exposed to high-uncertainty teacher supervision, resulting in limited growth in relative student confidence. This behavior mirrors the performance trends reported in Section 4.5, where decreasing schedules yield inferior results.

Figure 12 examines the robustness of these dynamics with respect to the number of Monte Carlo dropout samples used to estimate BALD. While increasing the number of samples slightly smooths the trajectories, the qualitative behavior remains unchanged. In particular, the relative ordering of schedules is preserved across all architectures, confirming that the observed confidence evolution is not an artifact of stochastic estimation noise.

Overall, these results provide direct empirical evidence that the proposed framework induces a progressive increase in student confidence during distillation. Under constant and increasing schedules, the student transitions from reliance on teacher supervision to a regime in which it is epistemically more confident than the teacher on an expanding subset of data. This evolution aligns closely with the calibration dynamics formalized in Theorems 3.5 and 3.5, reinforcing our central claim that confidence-aware geometric adaptation combined with epistemic rejection yields students that are not only more accurate, but also increasingly decisive over the course of training.

![](images/934b8ec27b9c74c8363689fa2cf66c8e073432e2e606a0ea24d271ed10b44140.jpg)  
(a) Chat alignment

![](images/3ba0e490cd8f184d338e84c74114e942ce1b58a484b5e76f61e9bd815f16c164.jpg)  
(b) Coding

![](images/1c380dfd0c1a2d0dbc1885fd0934b27f6d3583e4e436a24bcde974cbfb026e9f.jpg)  
(c) Mathematical reasoning  
Figure 13. Runtime analysis for diferent BALD sample configurations for specialized tasks.

## G Calibration, Selective Prediction, and Selection-Bias Analyses

## G.1 Calibration (ECE) and selective prediction

We report Expected Calibration Error [ECE; 63] and risk–coverage / selective-prediction curves [64] on the Dolly evaluation set for the GPT2 students. ECE must be read with care in this setting. Correctness is exact next-token match against a single reference, so token accuracy is nearly identical for every objective (0.43–0.47), and the only quantity that really varies is how confident each student is. Ordered by average confidence, ECE increases monotonically across all six settings (Table 9).

Table 9. ECE on Dolly for the GPT2 students, ordered by average confidence. With a single-reference target, ECE tracks confidence almost one-for-one and is largely a proxy for sharpness.
<table><tr><td>Objective</td><td>Avg. confidence</td><td>ECE</td></tr><tr><td>Forward KL</td><td>0.67</td><td>0.22</td></tr><tr><td>CaRE-KD (softest configuration)</td><td>0.68</td><td>0.28</td></tr><tr><td>α—β divergence</td><td>0.75</td><td>0.28</td></tr><tr><td>Reverse KL</td><td>0.77</td><td>0.31</td></tr><tr><td>Skewed-RKL</td><td>0.79</td><td>0.31</td></tr><tr><td>CaRE-KD (default temperature)</td><td>0.79</td><td>0.34</td></tr></table>

ECE tracks average confidence monotonically, from Forward KL (0.67, ECE 0.22) up to default CaRE-KD (0.79, ECE 0.34). On a single-reference target ECE is therefore largely a proxy for sharpness, and it penalizes exactly the precision-oriented, mode seeking behavior our method is designed to produce. Forward KL reaches the lowest ECE only by being under-confident — and it is also the weakest divergence on quality (lowest mean ROUGE-L in Table 1) — while CaRE-KD matches that same ECE level once its distribution is softened (softest configuration, confidence 0.68). We therefore do not claim that CaRE-KD is the best-calibrated method by ground-truth ECE, and we do not regard ground-truth ECE as the right yardstick for a method whose aim is to keep the student sharp where it is competent.

The calibration property we actually establish is diferent. Theorem 3.5 states that the student’s entropy converges toward the teacher’s, not toward a ground-truth-optimal value; the direct evidence is the uncertainty decoupling in Figures 1a and 4 (a 7% post-distillation KS drift on Dolly for CaRE-KD against 0.4% for Skewed-RKL). Separately, the risk–coverage curves are monotone for every objective, so a student’s confidence is a usable selective-prediction signal regardless of its absolute scale.

## G.2 Revival does not preferentially reject hard examples or reduce diversity

A quantile-scheduled rejection rule could in principle bias training by discarding hard examples or collapsing response diversity. We test this directly by comparing the region where Revival fires (teacher uncertain, student confident) against the complementary region on OpenLLaMA-Dolly, using LLM-judged question hardness and response diversity (Table 10). Neither diference is significant: Revival is not discarding the hard questions, and it does not measurably reduce the diversity of the student’s responses.

Table 10. Selection-bias check on OpenLLaMA-Dolly. LLM-judged question hardness and response diversity in the region where Revival fires vs. the complementary region. Neither diference is significant.
<table><tr><td></td><td>Measure (LLM-judged) Region where Revival fires Opposite region Mann-Whitney p</td><td></td><td></td></tr><tr><td>Question hardness</td><td>2.87</td><td>2.98</td><td>0.37</td></tr><tr><td>Response diversity</td><td>3.96</td><td>3.99</td><td>0.63</td></tr></table>