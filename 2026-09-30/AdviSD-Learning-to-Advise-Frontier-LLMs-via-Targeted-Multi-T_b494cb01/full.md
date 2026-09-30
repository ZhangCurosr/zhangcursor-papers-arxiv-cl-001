# AdviSD: Learning to Advise Frontier LLMs via Targeted Multi-Turn Self-Distillation

Rishabh Agrawal<sup>1,</sup> <sup>2\*</sup>, Hejie Cui<sup>1</sup>, Shasha Li<sup>1</sup>, Shanchan Wu<sup>1</sup> and Sercan Ö. Arık<sup>1</sup> <sup>1</sup>Google, <sup>2</sup>University of Southern California

A small trainable advisor can steer a frozen language-model executor using natural-language advice. In addition to learning from task rewards, the advisor can use feedback from completed interactions to improve its advice. However, a plausible correction need not change execution, yet learning from such corrections can still afect the advisor’s future decisions in other contexts. In a shared-parameter model, we prove that such corrections can limit learning if their targets favor useful advice less strongly than those of other corrections. Keeping them less often than the rest improves the model’s eventual performance compared to learning from every correction. Motivated by this, our method, Advisor Self-Distillation (AdviSD), pairs outcome-based reinforcement learning with self-distillation from a feedback-conditioned copy of the advisor selectively. Reflection proposes corrections, and the advisor scores the same recorded executor response with and without its issued advice, using the magnitude of the diference to select decisions for supervision. This approach does not require executor likelihoods or additional executor rollouts. Experiments with Qwen3-8B advisors for Gemini and Claude show that AdviSD outperforms advisor-GRPO by 4.2–6.4 percentage points on BFCL-v3 and by 3.9–5.1 score points on EnvScaler. The trained advisors generalize to out-of-domain tasks and transfer across diferent executor versions and model families. AdviSD also beats matched-count random selection, supporting the value of its selection rule.

## 1. Introduction

Frontier language models are usually served through APIs that accept queries but do not let users change the model weights. When such a model acts as an agent, a smaller trainable advisor can adapt it from the outside: the advisor observes the interaction and recommends what the frozen agent, which we call the executor, should do next (Li et al., 2025, 2023). The advisor can be trained with reinforcement learning on the rewards of completed interactions (Asawa et al., 2026). However, an episode reward summarizes task performance without specifying which of the advisor’s decisions should have been diferent, or how.

Completed interactions contain more specific evidence: executor responses, tool results, and task checks. These observations and the episode reward form the feedback that reflection uses to propose revisions (Liu et al., 2026; Yeo et al., 2026). In feedback-conditioned self-distillation, a teacher that sees this feedback supervises a student that sees only the original context (Agrawal et al., 2026b; Hübotter et al., 2026). For an advisor, however, the learned advice acts through another model: revising it need not change what the executor does.

Consider an executor that searches reliably for suitable flights without guidance but sometimes guesses the passenger identifier when booking. After a reservation fails because that identifier is wrong, reflection may propose looking up the passenger and using the returned identifier in the booking. It may also propose more detailed advice for the earlier flight search. Both revisions are valid, but their usefulness difers for this executor: the booking revision addresses the error, whereas the search revision elaborates a procedure the executor already follows unaided. The teacher’s supervision can reflect both revisions, and learning from the redundant search revision can still change the advisor’s shared parameters and afect advice elsewhere, for better or worse. Hence, we ask which revisions provide useful supervision for a given advisor–executor pair. Our contributions are as follows:

![](images/bde4d562a0455c45d2d8148e4bccc85044759988bd32937f6a0f0e59449977ff.jpg)  
Figure 1 | Overview of AdviSD. (A) Multi-turn advising. Before each executor response, the advisor provides advice or abstains; the frozen executor uses its native context and any issued advice. (B) Training from targeted feedback. Reflection proposes corrections at advice decisions in imperfect episodes. Among flagged decisions, AdviSD retains original abstentions and those whose scores with and without issued advice difer by more than a calibrated threshold. The pre-update advisor computes both on the same recorded executor response. At retained decisions, a feedback-conditioned copy of the pre-update advisor teaches the trainable advisor, which sees only the original context. This self-distillation supplements GRPO on all episodes.

A theoretical account of correction selection. Lemma 1 separates agreeing with a feedbackconditioned teacher from improving execution. We then study repeated learning in a shared-parameter model where the teacher’s preferences and the executor’s behavior under any given advice stay fixed during training. As advice improves, preventable failures become less frequent, while failures unafected by advice persist. If corrections from these persistent failures teach a weaker preference for useful advice, their growing share of supervision limits learning. Retaining them less often than other corrections raises the performance the advisor eventually reaches, whether or not reward learning is added (Theorems 1–2). By contrast, randomly discarding corrections at the same rate across both types leaves eventual performance unchanged when learning only from corrections, but can improve it alongside reward learning (Corollary 1). Therefore, our matched-count random control tests whether targeted selection helps beyond reducing supervision (Section 7).

AdviSD: multi-turn advising with targeted feedback. AdviSD separates proposing revisions from choosing where to learn (Figure 1). The advisor scores the same recorded executor response under two contexts, one containing its issued advice and one without it, so selection requires neither executor likelihoods nor additional executor rollouts. We use the magnitude of the score diference as a predictive signal for selecting decisions to supervise. At selected decisions, a feedback-conditioned copy of the pre-update advisor supervises the trainable advisor. This targeted self-distillation complements outcome-based GRPO (Shao et al., 2024). Reflection and scoring are used only during training. At deployment, the advisor decides before each executor turn whether to advise or abstain.

Empirical results. With Qwen3-8B advisors for Gemini and Claude, AdviSD has the highest in-domain aggregates on BFCL-v3 and EnvScaler among the compared methods. It exceeds advisor-GRPO by 4.2–6.4 percentage points on BFCL-v3 and 3.9–5.1 score points on EnvScaler. Across these settings, its 2.5–4.9-point advantage over matched-count random selection supports choosing which decisions to supervise beyond reducing supervision. Without retraining, AdviSD improves out-of-domain macro-averages by 2.7–3.6 points over standalone execution. It also transfers across executor versions and model families and exceeds transferred GRPO by 3.1 percentage points in both cross-family directions (Section 7).

## 2. Related Work

Advising and prompt optimization. Frozen models can be adapted through reusable instructions or context-dependent guidance. GEPA uses reflection to optimize reusable instructions (Agrawal et al., 2026a), while Directional Stimulus Prompting, Matryoshka Pilot, and Advisor Models train smaller models to guide frozen ones (Asawa et al., 2026; Li et al., 2025, 2023). Self-Refine and Reflexion use feedback to revise outputs or guide later attempts without updating model weights (Madaan et al., 2023; Shinn et al., 2023). Our advisor-GRPO baseline follows Advisor Models’ outcome-based training through a response-level tool-use interface; AdviSD adds targeted feedback-conditioned self-distillation.

Feedback-conditioned distillation. Feedback-conditioned on-policy self-distillation uses additional information to supervise a policy on its own trajectories. SDPO obtains this information from environment feedback or successful rollouts (Hübotter et al., 2026), and DistIL optimizes forward cross-entropy with sequence-level credit assignment (Agrawal et al., 2026b). Other approaches focus on constructing and allocating supervision. HERO constructs turn-level feedback, and HinT-SD selects failure-relevant action spans for self-distillation (Liu et al., 2026; Yeo et al., 2026). LOPD builds teacher context from retrieved experience, and DART-SD retrieves references for recovery (Xu et al., 2026; Zhang et al., 2026). SAGE-OPD selects and weights teacher supervision at individual turns (Zhou et al., 2026). In contrast, AdviSD addresses which corrections to learn when the trained policy advises a separate executor rather than directly performing the task.

Predictive contrasts and selection. Comparing predictions made with diferent information can yield a learning signal. RLCSD contrasts correct and incorrect hints, and OCSD compares full and observation-ablated contexts (Pan et al., 2026; Yang et al., 2026b). PBSD and RLSD use paired predictions to refine turn-level credit or token updates (Tian et al., 2026; Yang et al., 2026a). AdviSD applies paired scoring to a separate executor’s recorded response, with and without the issued advice. It uses the contrast magnitude to select auxiliary supervision for an advisor whose advice acts through a frozen executor, while leaving the rollout batch’s GRPO advantages unchanged. Appendix H provides the full discussion and further comparisons.

## 3. Background and Problem Formulation

Advising a frozen executor. Before each executor response $k ,$ the advisor reads a context $g _ { k }$ (the visible interaction, tool schemas, and its earlier advice) and samples an action $a _ { k } \sim \pi _ { \theta } ( { \bf \cdot } \mid g _ { k } )$ , which is either advice text or the abstention sequence <NO\_ADVICE>. The frozen executor responds with $m _ { k } \sim \rho ( { \cdot } \mid h _ { k } , e ( a _ { k } ) )$ , where $h _ { k }$ is its native history and $e ( a _ { k } )$ is the advice text, or an empty string if the advisor abstains. Advice goes into a temporary request rather than the executor’s persistent history. The advisor makes a new decision before every response, including text-only responses and those that follow tool results; parallel tool calls within one response share a single decision. An episode ends with reward $R \in [ 0 , 1 ]$

Two learning signals. For � rollouts of a task, GRPO assigns episode � the advantage ${ \widehat { A } } _ { i } =$ $( R _ { i } - \bar { R } ) / ( s _ { R } + \epsilon _ { A } )$ , where �<sup>¯</sup> and $s _ { R }$ are the group’s reward mean and standard deviation, and $\epsilon _ { A } > 0$ stabilizes the denominator. This advantage applies to every advisor token generated in the episode, including abstentions. We denote the clipped GRPO loss with reference-policy regularization by $\mathcal { L } _ { \mathrm { b a s e } }$ (Shao et al., 2024).

Feedback-conditioned self-distillation supervises individual advice decisions. The teacher, a copy of the pre-update advisor with parameters ${ \bar { \theta } } ,$ sees the original context augmented with feedback from the completed interaction. The trainable student sees only the original context and learns to match the teacher’s next-token distributions, which are held fixed during optimization. AdviSD chooses which decisions to supervise, including originally abstaining decisions flagged by reflection, and leaves the GRPO advantages unchanged.

## 4. Why the Choice of Corrections Matters

We examine why fitting a teacher need not improve execution (Section 4.1), then show how the retained corrections determine the learning limit in a shared-parameter model (Section 4.2).

## 4.1. A single update through the executor

Fix an interaction state and advice prefix $h ,$ and let $p _ { \theta }$ and $q _ { h }$ be positive student and teacher distributions on a fixed finite token set $S _ { h }$ , possibly the full vocabulary. The student is diferentiable near the pre-update parameters ${ \bar { \theta } } .$ . Choosing token $\nu ,$ completing the advice with the pre-update advisor, and running the executor induces an execution law $K _ { h , \upsilon }$ over responses and outcomes, excluding advice text. For a bounded task score �, define

$$
V _ { h } ( \boldsymbol { \nu } ) = \mathbb { E } _ { Z \sim K _ { h , \boldsymbol { \nu } } } [ W ( Z ) ] , \qquad J _ { h } ( \boldsymbol { \theta } ) = \sum _ { \boldsymbol { \nu } \in S _ { h } } p _ { \boldsymbol { \theta } } ( \boldsymbol { \nu } ) V _ { h } ( \boldsymbol { \nu } ) .
$$

These are the expected score after choosing � and its average under the student. Only next-token probabilities vary during diferentiation; the teacher, support, completion policy, and execution laws remain fixed. Feedback does not directly provide each token’s expected execution value. We therefore compare $q _ { h }$ with a normalized reference $q _ { h } ^ { \alpha } ( \nu ) \propto p _ { \bar { \theta } } ( \nu ) e ^ { \alpha V _ { h } ( \nu ) } , \alpha > 0$ , which reweights the student toward higher-value tokens. This value-tilted teacher (Peters et al., 2010) is an analytical benchmark; AdviSD does not construct it.

Lemma 1 (Teacher fitting versus execution improvement). Under this setup, let $\phi _ { h } ( \nu ) = \nabla _ { \theta } \log p _ { \theta } ( \nu ) \vert _ { \bar { \theta } }$ <sub>an</sub>d $g _ { h } = \nabla _ { \theta } \operatorname { K L } ( p _ { \theta } | | q _ { h } ) | _ { \bar { \theta } }$ . For everyfixed $\alpha > 0$

$$
g _ { h } = - \alpha \nabla J _ { h } ( \bar { \theta } ) + e _ { h } , \qquad e _ { h } = \mathbb { E } _ { \nu \sim p _ { \bar { \theta } } } \left[ \phi _ { h } ( \nu ) \log \frac { q _ { h } ^ { \alpha } ( \nu ) } { q _ { h } ( \nu ) } \right] .\tag{1}
$$

Reverse-KL descent at $\bar { \theta }$ toward the reference follows $\alpha \nabla J _ { h }$ , whereas descent toward the actual teacher follows $\alpha \nabla J _ { h } - e _ { h }$ , whose residual can reinforce or oppose value ascent. At an insensitive prefix, every supported token induces the same execution law, so $\nabla J _ { h } = 0$ and the reference equals the student. Changing token probabilities cannot improve this local objective, but teacher fitting can still change advice elsewhere through shared parameters, for better or worse. Teacher agreement alone does not tell us which. Appendix B gives the full proof and extensions.

## 4.2. Repeated updates and the learning limit

Lemma 1 concerns one update. We now introduce a simplified shared-parameter model to study how repeated learning changes which failures occur and which corrections supply supervision.

Setup. The advisor chooses between advice 1 and advice 2, selecting advice 1 with probability $p ( \theta ) = 1 / ( 1 + e ^ { - \theta } )$ . A single log-odds parameter � is shared across a fixed mixture of sensitive (�)

and insensitive (�) situations, each with positive probability. In sensitive situations, advice 1 succeeds more often than advice ${ 2 ; }$ in insensitive situations, both induce the same execution law. These laws remain fixed, with success probabilities in (0, 1), so expected success $J ( \theta )$ increases with �.

A failure of type $j \in \{ I , S \}$ supplies a fixed teacher $Q _ { j } ,$ positive on both advice choices, with target log-odds $\ell _ { j } = \log [ Q _ { j } ( 1 ) / Q _ { j } ( 2 ) ]$ . Even when both advice choices lead to identical executor behavior, the teacher may prefer one over the other. We separately assume $\ell _ { I } < \ell _ { S }$ , meaning that the insensitive teacher assigns less probability to advice 1 than the sensitive teacher. For example, $Q _ { I } = ( 0 . 5 , 0 . 5 )$ is neutral, while $Q _ { S } = ( 0 . 8 , 0 . 2 )$ prefers advice 1. If the advisor already selects advice 1 with probability $p > 1 / 2$ , learning from $Q _ { I }$ pushes that probability down toward $1 / 2$ . Because � is shared, this also makes advice 1 less likely in sensitive situations, where it succeeds more often. Thus, a neutral teacher can weaken useful advice elsewhere without favoring advice 2.

Each episode contains a fixed positive number of independent, identically distributed (situation, advice, outcome) samples from this model. Failures supply correction proposals up to a fixed positive cap, with uniform subsampling if the cap is exceeded. Proposals of type � are retained independently with fixed probability $r _ { j } \in ( 0 , 1 ]$ ; no gating means $r _ { I } = r _ { S } = 1$ . The episode loss averages student-toteacher reverse KL over retained corrections and is zero if none remain. Samples and selections are held fixed during diferentiation.

Retained supervision. Let $B ( \theta )$ be the probability that an episode retains any correction, and let $\omega _ { I } ( \theta )$ be the expected fraction of insensitive corrections conditional on retaining at least one. The mean target log-odds is $\mu ( \theta ) = \omega _ { I } ( \theta ) \ell _ { I } + [ 1 - \omega _ { I } ( \theta ) ] \ell _ { S }$ . The quantity � measures exposure, how often episodes receive supervision, while � summarizes the composition of that supervision, the mixture of teacher targets. As � increases, sensitive failures become less frequent while the insensitive failure rate stays fixed. The insensitive teacher therefore receives a growing share of supervision, so � decreases.

Theorem 1 (The retained mixture sets the learning limit). Under this setup, with distillation alone, the expected gradient of the sampled episode loss is

$$
g ( \theta ) = B ( \theta ) p ( 1 - p ) [ \theta - \mu ( \theta ) ] .\tag{2}
$$

Th<sub>e co</sub>ntinu<sub>o</sub>u<sub>s</sub>-tim<sub>e</sub> u<sub>p</sub>d<sub>a</sub>t<sub>e</sub> $\dot { \theta } = - g ( \theta )$ converges from every finite initialization to a unique equilibrium $\theta ^ { * } = \mu ( \theta ^ { * } ) \in ( \ell _ { I } , \ell _ { S } )$ . Decreasin<sub>g</sub> $r _ { I } / r _ { S }$ <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreases</sub> b<sub>o</sub>th $\theta ^ { * }$ <sub>an</sub>d $J ( \theta ^ { * } )$ . Ch<sub>a</sub>n<sub>g</sub>in<sub>g</sub> th<sub>e</sub> <sub>ep</sub>i<sub>so</sub>d<sub>e</sub> <sub>s</sub>i<sub>ze</sub> <sub>o</sub>r proposal cap, or scaling both retention probabilities by the same admissible positive factor, leaves the li<sub>m</sub>it <sub>unc</sub>h<sub>ange</sub>d<sub>.</sub>

Each teacher contributes $p ( 1 - p ) ( \theta - \ell _ { j } )$ to the gradient; averaging yields Eq. (2). Learning settles where the advisor’s log-odds equal the retained teachers’ mean target. Retaining insensitive corrections less often than sensitive ones raises this target and the resulting learning limit. Independently thinning both types at the same rate changes exposure but not the target. Appendices C.1–C.3 provide the proof, scope, and extensions.

This model retains corrections by situation type. AdviSD instead uses an observable predictive contrast (Section 5), whose usefulness we evaluate through ablations (Section 7). When reward learning is added, exposure can also afect eventual performance (Section 6), so the ablations include a matched-count random control.

## 5. AdviSD: Advisor Self-Distillation

AdviSD trains only the advisor, combining outcome-based GRPO with targeted self-distillation (Figure 1). An external reflector proposes corrections, a predictive selector chooses decisions to

supervise, and a feedback-conditioned pre-update advisor teaches a student that sees only the original context. These stages run only during training (Algorithm 1; Appendix D).

## 5.1. Reflection proposes corrections

For each eligible imperfect episode $i ,$ the reflector uses executor responses, tool outcomes, and checks to flag at most $b _ { \mathrm { r e f l } }$ advice decisions $\mathcal { T } _ { i }$ with correction feedback. Later events can explain failures, but proposed advice uses information available at the original decision (Appendices F and D).

## 5.2. A paired score selects where to learn

Scoring. Let $y _ { k } = ( y _ { k , 1 } , \dots , y _ { k , T _ { k } } )$ denote the recorded executor response, serialized and tokenized with the advisor’s tokenizer. It includes tool calls in their recorded order but excludes subsequent tool results. We construct two scoring contexts, $C _ { k } ^ { + }$ and $C _ { k } ^ { - }$ , from the request sent to the executor rather than from the advisor’s context $g _ { k }$ . Both contain the same pre-response history and tool schemas and difer only in the issued advice, which $C _ { k } ^ { + }$ includes and $C _ { k } ^ { - }$ omits. The pre-update advisor scores each token of $y _ { k }$ under both contexts:

$$
c _ { k } = \frac { 1 } { T _ { k } } \sum _ { t = 1 } ^ { T _ { k } } \left[ \log \pi _ { \bar { \theta } } ( y _ { k , t } \mid C _ { k } ^ { + } , y _ { k , < t } ) - \log \pi _ { \bar { \theta } } ( y _ { k , t } \mid C _ { k } ^ { - } , y _ { k , < t } ) \right] .\tag{3}
$$

The magnitude $\left| c _ { k } \right|$ indicates how strongly the issued advice changes the advisor’s prediction of the recorded response. We use this predictive signal to select whole advice decisions for supervision. Both scores are computed by the pre-update advisor on the same recorded response, so selection requires neither executor likelihoods nor additional executor rollouts.

Calibration and selection. We calibrate the gate on prediction changes caused by advice from other tasks. Before training, each run collects pilot rollouts on training tasks with its initial advisor. At valid decisions where advice was issued, we replace it in the with-advice scoring context with advice from another task, which we call donor advice. Equation (3), applied to the same recorded response and no-advice baseline, then gives the donor contrast $d _ { j }$ . Donor advice is scored but never sent to the executor. We set the threshold to an empirical quantile of the $n _ { \mathrm { c a l } }$ donor contrast magnitudes:

$$
\epsilon _ { c } = Q _ { u _ { \mathrm { d } } } \left( \{ | d _ { j } | \} _ { j = 1 } ^ { n _ { \mathrm { c a l } } } \right) , \qquad u _ { \mathrm { d } } \in ( 0 , 1 ) ,\tag{4}
$$

which stays fixed during training. Pilot scores for the issued advice are used only for admission checks (Appendix D). Of the flagged decisions J<sub>�</sub>, AdviSD retains two kinds in $\textstyle { \mathcal { I } } _ { i } \colon$ original abstentions and decisions with $| c _ { i , k } | > \epsilon _ { c }$ . An original abstention has $C _ { k } ^ { + } = C _ { k } ^ { - }$ and hence $c _ { k } = 0 .$ , so the contrast cannot detect missed advice; flagged abstentions therefore bypass scoring. A proposal to abstain after issued advice must still pass the numeric gate.

## 5.3. Self-distillation from targeted feedback

At each retained decision, $\zeta _ { i , k }$ combines local execution evidence, relevant checks, episode score, and reflection feedback. The teacher sees it prepended to the context $g _ { i , k } ;$ the student sees only $g _ { i , k }$ . Both predict along the originally sampled advice $a _ { i , k } ,$ including <NO\_ADVICE>:

$$
q _ { i , k , t } = { { s g } \ \pi _ { \bar { \theta } } ( \cdot \mid \zeta _ { i , k } \oplus g _ { i , k } , a _ { i , k , < t } ) , \ \quad p _ { i , k , t } = \pi _ { \theta } ( \cdot \mid g _ { i , k } , a _ { i , k , < t } ) } .
$$

Here $s g$ stops gradients, ⊕ prepends feedback, and both use temperature $T _ { \mathrm { S D } }$ . At each prefix, both distributions are renormalized over a fixed support �: the pre-update student’s top-� tokens.

$$
\ell _ { i , k } = \frac { 1 } { | a _ { i , k } | } \sum _ { t = 1 } ^ { | a _ { i , k } | } \mathrm { K L } ( p _ { i , k , t } ^ { S } | | q _ { i , k , t } ^ { S } ) , \quad \mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + \frac { \lambda _ { s } } { N _ { \mathrm { e p } } } \sum _ { i = 1 } ^ { N _ { \mathrm { e p } } } \frac { \sum _ { k \in { \cal I } _ { i } ^ { * } } \ell _ { i , k } } { \operatorname* { m a x } \{ 1 , | { \cal I } _ { i } ^ { * } | \} } .\tag{5}
$$

Here ${ \mathcal { T } } _ { i } ^ { * } \subseteq { \mathcal { I } } _ { i }$ contains decisions with feasible teacher contexts, $N _ { \mathrm { e p } }$ counts episodes, and $\lambda _ { s }$ weights distillation. We average token losses within each supervised decision, then average these decision losses within each episode and the resulting episode losses across the full batch. Episodes without supervision contribute zero auxiliary loss. GRPO still uses every episode with its original advantage. For each rollout batch, we compute teacher distributions using the pre-update advisor and hold them fixed during optimization. Self-distillation diferentiates only through the student’s probabilities; token supports, recorded advice prefixes, and selection weights also stay fixed (Appendix D).

## 6. Targeted Supervision: Reward Learning and Calibration

Section 4.2 studies distillation alone. We add reward learning, where exposure (how often episodes receive supervision) can also change the learning limit, and examine donor calibration.

Selection alongside reward learning. We extend the two-teacher model of Section 4.2 to combine reward learning with distillation, and we keep the assumption $\ell _ { I } < \ell _ { S }$ . Reward learning encourages advice with higher expected success, while distillation pulls the advisor toward the retained teachers’ mean target. We compare learning from all proposals $( j = 0 )$ with selective retention $( j = G )$ . Both start from the same initialization and use the same fixed weights:

$$
\dot { \theta } _ { j } = F _ { j } ( \theta _ { j } ) , \qquad F _ { j } ( \theta ) = c _ { 0 } J ^ { \prime } ( \theta ) - \lambda g _ { j } ( \theta ) , \qquad c _ { 0 } \geq 0 , \quad \lambda > 0 .
$$

Here $J ^ { \prime } ( \theta )$ is the gradient of expected success and $g _ { j } ( \theta )$ is the expected distillation gradient under retention rule $j ; c _ { 0 }$ and � weight the two learning signals. The reward term idealizes finite-step GRPO–AdamW training as exact gradient ascent. Let $a _ { 0 }$ denote the distillation-only equilibrium without gating (Theorem 1).

Theorem 2 (A higher learning limit under a common reward objective). Under the preceding twoteacher model, suppose � independently retains insensitive and sensitive proposals with fixed probabilities $r _ { I } ^ { G } , r _ { S } ^ { G } \in ( 0 , 1 ]$ <sub>,</sub> r<sub>espec</sub>tiv<sub>e</sub>l<sub>y,</sub> wh<sub>e</sub>r<sub>e</sub> $r _ { I } ^ { G } < r _ { S } ^ { G }$ . Without <sub>g</sub>atin<sub>g,</sub> ever<sub>y p</sub>ro<sub>p</sub>osal is retained. From an<sub>y</sub> common finite initialization, both learning dynamics converge to finite equilibria satisfying

$$
\theta _ { G } ^ { \infty } > \theta _ { 0 } ^ { \infty } , \qquad J ( \theta _ { G } ^ { \infty } ) > J ( \theta _ { 0 } ^ { \infty } ) .\tag{6}
$$

These inequalities hold even when the dynamics have multiple equilibria. If the common initialization is <sub>a</sub>t <sub>or a</sub>b<sub>ove</sub> $a _ { 0 } ,$ th<sub>en</sub> $\theta _ { G } ( t ) > \theta _ { 0 } ( t )$ for every $t > 0$

Selection changes both which corrections are learned and how often learningfrom corrections occurs. The mean retained teacher target $\mu _ { j }$ captures the first, and the probability $B _ { j }$ that an episode receives supervision captures the second (Section 4.2). At the same parameter value $\theta _ { z }$ , the diference between the two updates is

$$
\frac { F _ { G } - F _ { 0 } } { \lambda p ( 1 - p ) } = \underbrace { B _ { G } \big ( \mu _ { G } - \mu _ { 0 } \big ) } _ { \mathrm { c o m p o s i t i o n } } + \underbrace { \big ( B _ { 0 } - B _ { G } \big ) ( \theta - \mu _ { 0 } ) } _ { \mathrm { e x p o s u r e } } .\tag{7}
$$

The composition term is positive because selection gives more weight to the sensitive teacher, which assigns a higher probability to advice 1 than the insensitive teacher. The exposure term comes from less frequent supervision, and its sign depends on �. Above $a _ { 0 }$ , ungated distillation pulls � downward, so reducing this pull helps. Below $a _ { 0 } ,$ distillation pushes � upward, so reducing supervision can slow early progress. A higher learning limit therefore need not mean faster learning from the start.

Both dynamics move upward below $a _ { 0 } .$ , and $F _ { G } > F _ { 0 }$ at and above it. Each trajectory remains bounded and cannot cross an equilibrium of its own dynamics; together, these properties establish the ordered limits. Appendices C.2 and C.3 provide the proof, an example of slower initial learning, and a stationary random-teacher extension.

For $c _ { 0 } > 0 ,$ reward-only ascent approaches the model’s best achievable success. A fixed $\lambda > 0$ instead produces a finite balance between reward ascent and teacher fitting, and at that balance selection gives higher success than no gating. The theorem compares the two combined rules with each other; Section 7 tests gains over outcome-only GRPO under finite training budgets.

Because selection also changes how often supervision occurs, outperforming no gating does not by itself show that choosing particular corrections helps. Retaining every proposal independently with the same fixed probability preserves the teacher mixture and the distillation-only limit, yet can improve eventual performance when reward learning is present (Corollary 1). This motivates our matched-count random control (Section 7). On each of the control’s own episodes, a shadow AdviSD gate determines how many feasible, reflection-flagged issued-advice decisions to retain. The control then randomly selects the same number of eligible decisions, giving each decision an equal chance of being chosen. Unlike independent thinning, these episode-dependent quotas need not preserve the ungated teacher mixture (Appendix E).

What donor calibration controls. Donor calibration sets a reference threshold using prediction changes produced by advice from other tasks. The quantile in Eq. (4) bounds how often donor advice passes the gate on the pilot sample (Appendix D). We test whether the resulting selection rule improves learning, alongside the contribution of the abstention bypass, through ablations (Section 7). Appendix E.1 reports which decisions receive supervision during training.

## 7. Experiments

We ask four questions: (Q1) How does AdviSD compare with baselines? (Q2) Which corrections should receive supervision? (Q3) Without retraining, does it transfer to out-of-domain tasks and across executor versions and families? (Q4) How does performance evolve during training?

Setup. We train Qwen3-8B advisors for two frozen executors, Gemini 3.7 Flash and Claude Sonnet 4.6, separately on BFCL-v3 (Patil et al., 2025) and EnvScaler (Song et al., 2026). The test sets are fixed across runs: 320 BFCL tasks (80 in each of its four categories) and 200 EnvScaler tasks. For each seed, we re-split the remaining tasks into training and validation sets (Appendix E). Following LOPD (Zhang et al., 2026), we report BFCL-v3 accuracy under the oficial multi-turn checker as an equal-weight average of the four categories. EnvScaler uses native graded scores in [0, 1]; we report their mean multiplied by 100. AdviSD caps reflection at $b _ { \mathrm { r e f l } } = 5$ decisions per episode and calibrates each run’s fixed threshold $\epsilon _ { c }$ on a training-task pilot using donor quantile $u _ { \mathrm { d } } = 0 . 9 5$

Evaluation. Every trained method has three independent training runs. For each run, we take the checkpoint selected on validation, evaluate it four times on the test set, and average the four scores. We report the mean ± sample standard deviation (SD) of the three run averages. GEPA is aggregated the same way, with three independent prompt searches in place of training runs. The no-advisor and frozen-advisor controls have no training runs, so their three averages come from three groups of four evaluations of the same system. Their SD reflects only evaluation randomness (Appendix E).

Table 1 | In-domain test performance. EnvScaler: native score ×100; BFCL-v3: oficial-checker accuracy (%) on 320 held-out tasks; Avg weights categories equally. Mean ± sample SD follows Section 7. Bold/underline: largest/second-largest distinct means, including ties, per column/executor.
<table><tr><td rowspan="2">Adaptation method</td><td rowspan="2">EnvScaler |</td><td colspan="5">BFCL-v3</td></tr><tr><td>Score Base</td><td>Miss Func Miss Param Long Ctx</td><td></td><td></td><td> $\mathtt { A v g }$ </td></tr><tr><td colspan="7">Frozen executor: Gemini 3.7 Flash</td></tr><tr><td colspan="7">Standalone executor: no advisor</td></tr><tr><td>No advisor</td><td> $7 5 . 7 { \pm } 2 . 0 $ </td><td> $\underline { { 7 0 . 8 { \pm } 1 . 9 } }$ </td><td> $6 0 . 4 { \scriptstyle \pm 1 . 4 }$ </td><td> $4 2 . 9 { \scriptstyle \pm 1 . 9 }$ </td><td> $5 7 . 1 { \pm } 1 . 9 $ </td><td> $5 7 . 8 { \pm } 1 . 4 $ </td></tr><tr><td colspan="7">Executor prompt optimization: no advisor</td></tr><tr><td>GEPA (Agrawal et al., 2026a)</td><td> $7 6 . 0 { \scriptstyle \pm 1 . 8 }$ </td><td> $6 9 . 6 { \scriptstyle \pm 1 . 9 }$ </td><td> $6 1 . 7 { \scriptstyle \pm 1 . 4 }$ </td><td> $4 5 . 4 { \scriptstyle \pm 1 . 4 }$ </td><td> $6 0 . 8 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 9 . 4 { \scriptstyle \pm 0 . 3 }$ </td></tr><tr><td colspan="7">Frozen advisors: no task-specific optimization</td></tr><tr><td>Gemini 3.7 Flash advisor</td><td> $7 7 . 7 { \scriptstyle \pm 0 . 7 }$ </td><td> $7 0 . 4 { \scriptstyle \pm 1 . 4 }$ </td><td> $6 2 . 1 { \pm } 1 . 9$ </td><td> $4 4 . 6 { \scriptstyle \pm 1 . 9 }$ </td><td> $5 9 . 6 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 9 . 2 { \scriptstyle \pm 0 . 5 }$ </td></tr><tr><td>Untrained Qwen3-8B</td><td> $7 5 . 6 { \scriptstyle \pm 2 . 2 }$ </td><td> $6 7 . 1 { \pm } 1 . 9$ </td><td> $5 9 . 6 { \scriptstyle \pm 1 . 9 }$ </td><td> $3 9 . 6 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 8 . 3 { \scriptstyle \pm 1 . 9 }$ </td><td> $5 6 . 1 { \pm } 1 . 0 $ </td></tr><tr><td colspan="7">Trained advisors: Qwen3-8B weight updates</td></tr><tr><td>GRPO (Shao et al., 2024)</td><td> $7 9 . 7 { \pm } 2 . 2 $ </td><td> $\underline { { 7 0 . 8 { \pm } 1 . 9 } }$ </td><td> $\underline { { 6 4 . 2 \pm 1 . 4 } }$ </td><td> $4 7 . 1 { \pm } 1 . 9 $ </td><td> $6 3 . 3 { \scriptstyle \pm 1 . 4 }$ </td><td> $6 1 . 4 { \scriptstyle \pm 1 . 0 }$ </td></tr><tr><td>SDPO (Hübotter et al., 2026)</td><td> $\overline { { 7 6 . 0 { \pm } 1 . 7 } }$ </td><td> $6 9 . 2 { \scriptstyle \pm 1 . 9 }$ </td><td> $6 0 . 8 { \scriptstyle \pm 1 . 4 }$ </td><td> $\overline { { 4 3 . 3 { \pm } 1 . 4 } }$ </td><td> $\overline { { 5 7 . 9 { \pm 1 . 9 } } }$ </td><td> $\overline { { 5 7 . 8 { \pm } 0 . 8 } }$ </td></tr><tr><td>DistIL (Agrawal et al., 2026b)</td><td> $7 6 . 1 { \scriptstyle \pm 2 . 4 }$ </td><td> $6 9 . 6 { \scriptstyle \pm 1 . 4 }$ </td><td> $6 1 . 7 { \scriptstyle \pm 1 . 4 }$ </td><td> $4 4 . 6 { \pm } 1 . 4$ </td><td> $5 8 . 3 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 8 . 5 { \scriptstyle \pm 0 . 4 }$ </td></tr><tr><td>ADVISD (ours)</td><td> ${ \mathbf 8 4 . 8 \pm 1 . 6 }$ </td><td> $7 3 . 3 { \pm } 1 . 9$ </td><td> ${ 7 2 . 1 \pm 1 . 9 }$ </td><td> $5 4 . 2 { \scriptstyle \pm 1 . 9 }$ </td><td> $7 1 . 7 { \pm } 1 . 9$ </td><td> ${ \bf 6 7 . 8 { \pm 1 . 6 } }$ </td></tr><tr><td colspan="7">Frozen executor: Claude Sonnet 4.6</td></tr><tr><td colspan="7">Standalone executor: no advisor</td></tr><tr><td>No advisor</td><td> $7 7 . 4 { \scriptstyle \pm 2 . 1 }$ </td><td> $6 7 . 1 { \pm } 1 . 4 $ </td><td> $5 3 . 8 { \scriptstyle \pm 2 . 2 }$ </td><td> $4 7 . 9 { \pm } 1 . 4 $ </td><td> $5 0 . 8 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 4 . 9 { \scriptstyle \pm 0 . 9 }$ </td></tr><tr><td colspan="7">Executor prompt optimization: no advisor</td></tr><tr><td>GEPA (Agrawal et al., 2026a)</td><td> $7 8 . 4 { \scriptstyle \pm 1 . 3 }$ </td><td> $6 7 . 1 { \pm } 1 . 9$ </td><td> $5 8 . 3 { \scriptstyle \pm 1 . 9 }$ </td><td> $5 6 . 3 { \scriptstyle \pm 2 . 5 }$ </td><td> $5 0 . 4 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 8 . 0 { \scriptstyle \pm 0 . 8 }$ </td></tr><tr><td colspan="7">Frozen advisors: no task-specific optimization</td></tr><tr><td>Claude Sonnet 4.6 advisor</td><td> $7 8 . 7 { \scriptstyle \pm 0 . 5 }$ </td><td> $\underline { { 6 8 . 3 { \scriptstyle \pm 1 . 9 } } }$ </td><td> $5 7 . 9 { \pm } 1 . 9$ </td><td> ${ \bf 6 2 . 1 { \pm 1 . 4 } }$ </td><td> $5 3 . 3 { \scriptstyle \pm 1 . 9 }$ </td><td> $6 0 . 4 { \scriptstyle \pm 0 . 7 }$ </td></tr><tr><td>Untrained Qwen3-8B</td><td> $7 4 . 1 { \pm } 1 . 5 $ </td><td> $\overline { { 6 3 . 3 { \pm } 1 . 9 } }$ </td><td> $5 5 . 4 { \scriptstyle \pm 1 . 9 }$ </td><td> $4 7 . 1 { \pm } 1 . 9 $ </td><td> $5 0 . 8 { \scriptstyle \pm 1 . 9 }$ </td><td> $\overline { { 5 4 . 2 \pm 1 . 3 } }$ </td></tr><tr><td colspan="7"></td></tr><tr><td>Trained advisors: Qwen3-8B weight updates GRPO (Shao et al., 2024)</td><td> $8 0 . 7 { \scriptstyle \pm 2 . 3 }$ </td><td> $6 6 . 7 { \scriptstyle \pm 1 . 9 }$ </td><td> $5 9 . 6 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 4 . 6 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 4 . 6 { \scriptstyle \pm 0 . 7 }$ </td><td> $5 8 . 9 { \scriptstyle \pm 0 . 7 }$ </td></tr><tr><td>SDPO (Hübotter et al., 2026)</td><td> $\overline { { 7 8 . 9 \pm 1 . 2 } }$ </td><td> $6 4 . 2 { \scriptstyle \pm 1 . 9 }$ </td><td> $\overline { { 5 9 . 2 \pm 1 . 4 } }$ </td><td> $4 9 . 2 { \pm } 1 . 4 $ </td><td> $5 0 . 8 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 5 . 8 { \scriptstyle \pm 0 . 7 }$ </td></tr><tr><td>DistIL (Agrawal et al., 2026b)</td><td> $7 8 . 9 { \pm } 1 . 5 $ </td><td> $6 5 . 4 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 9 . 6 { \scriptstyle \pm 1 . 9 }$ </td><td> $5 1 . 3 { \scriptstyle \pm 2 . 2 }$ </td><td> $5 2 . 9 { \scriptstyle \pm 1 . 9 }$ </td><td> $5 7 . 3 { \pm } 1 . 3 $ </td></tr><tr><td>ADVISD (ours)</td><td> ${ \bf 8 4 . 6 { \pm 1 . 1 } }$ </td><td> ${ \bf 6 9 . 6 { \bf \pm 1 . 4 } }$ </td><td> $\overline { { 6 5 . 4 \pm 1 . 4 } }$ </td><td> $\underline { { 5 9 . 2 \pm 1 . 4 } }$ </td><td> $5 8 . 3 { \scriptstyle \pm 1 . 4 }$ </td><td> ${ \bf 6 3 . 1 { \bf \pm 0 . 6 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Baselines. The standalone executor receives no advice. For executor prompt optimization, GEPA (Agrawal et al., 2026a) optimizes reusable executor instructions without an advisor. The frozen advisors are untrained Qwen3-8B and the executor’s own API model; neither receives task-specific optimization. Among trained advisors, outcome-only advisor-GRPO (Shao et al., 2024), labeled GRPO in the tables, uses the same reward-learning objective as AdviSD without self-distillation. Adapted SDPO (Hübotter et al., 2026) and DistIL (Agrawal et al., 2026b) use successful sibling rollouts of the same task as feedback, using their distillation objectives without an added GRPO loss. All advised systems share the response-level interface and can abstain (Appendices D and E).

(Q1) In-domain performance. AdviSD has the highest BFCL-v3 average and EnvScaler score for both executors (Table 1). It exceeds GRPO by 4.2–6.4 percentage points on BFCL-v3 and 3.9–5.1 score points on EnvScaler. For both executors, gains over GRPO are larger in BFCL’s augmented categories than in Base, suggesting greater benefits when tools (Missing Function) or required information (Missing Parameter) are missing, or contexts are longer (Long Context). Untrained Qwen3-8B advice lowers all four aggregate means relative to standalone execution, while frozen frontier advisors raise them. After AdviSD training, the smaller Qwen3-8B advisor outperforms these frontier advisors in every comparison in Table 1 except Missing Parameter with the Claude executor. Adapted SDPO and DistIL reuse the same successful-rollout feedback to supervise the advisor’s generated tokens across turns, yet both trail GRPO in aggregate. This gap motivates decision-level targeting, since shared trajectory-level feedback need not be equally useful throughout a multi-turn interaction. Appendix G shows a case study comparing execution with and without a trained advisor.

![](images/0c0e7496cf0c1828a1e886a654e2eadfd47f01c69366dba14c4ba2986ec0b828.jpg)

Table 2 | Correction selection and abstention supervision (Q2). All AdviSD variants share GRPO, reflection, teacher construction, and the auxiliary loss and weight schedule. Self-distillation uses feasible reflection-flagged decisions. AdviSD retains issued-advice decisions whose with- and without-advice scores difer by more than the threshold in either direction; inverted gate retains those with absolute score diferences at or below the threshold, without count matching. No gate retains every proposal. Matched-count random samples issued-advice decisions uniformly, matching the gate’s per-episode count on the random control’s own rollouts. All four retain flagged original abstentions. No abstention bypass removes self-distillation at original abstentions while keeping the ordinary gate, GRPO, and the advisor’s ability to abstain. GRPO alone uses no self-distillation. Metrics and averaging follow Table 1; bold marks column maxima.
<table><tr><td></td><td colspan="2">BFCL-v3</td><td colspan="2">EnvScaler</td></tr><tr><td>Advisor training</td><td>Gemini 3.7 Flash</td><td>Claude Sonnet 4.6</td><td>Gemini 3.7 Flash</td><td>Claude Sonnet 4.6</td></tr><tr><td>GRPO</td><td> $6 1 . 4 \pm 1 . 0$ </td><td> $5 8 . 9 \pm 0 . 7 $ </td><td> $7 9 . 7 \pm 2 . 2$ </td><td> $8 0 . 7 \pm 2 . 3 $ </td></tr><tr><td>ADVISD, no gate</td><td> $6 2 . 7 \pm 0 . 8$ </td><td> $5 9 . 7 \pm 0 . 3$ </td><td> $8 1 . 1 \pm 1 . 5$ </td><td> $8 1 . 4 \pm 0 . 9$ </td></tr><tr><td>ADvISD, inverted gate</td><td> $6 1 . 1 \pm 1 . 4$ </td><td> $5 8 . 3 \pm 0 . 6$ </td><td> $7 8 . 7 \pm 1 . 9$ </td><td> $7 9 . 4 \pm 1 . 6$ </td></tr><tr><td>AdviSD, matched-count random</td><td> $6 2 . 9 \pm 0 . 9$ </td><td> $6 0 . 6 \pm 0 . 4$ </td><td> $8 0 . 9 \pm 1 . 3$ </td><td> $8 1 . 1 \pm 1 . 2$ </td></tr><tr><td>ADVISD</td><td> ${ \bf 6 7 . 8 \pm 1 . 6 }$ </td><td> ${ \bf 6 3 . 1 \pm 0 . 6 }$ </td><td> ${ \pm 2 4 . 8 \pm 1 . 6 }$ </td><td> ${ \bf 8 4 . 6 \pm 1 . 1 }$ </td></tr><tr><td>AdviSD, no abstention bypass</td><td> $6 6 . 1 \pm 1 . 3$ </td><td> $6 2 . 5 \pm 0 . 8$ </td><td> $8 3 . 3 \pm 2 . 1$ </td><td> $8 3 . 5 \pm 1 . 5$ </td></tr></table>

(Q2) Which corrections should the advisor learn? AdviSD exceeds matched-count random selection by 2.5–4.9 points and no gating by 3.2–5.1 (Table 2), whereas the two controls difer by at most 0.9 points. Theorem 2 and Corollary 1 motivate the random control, which matches the gate’s per-episode counts on its own rollouts. AdviSD is trained separately, so its episodes and supervision counts can difer from the control’s. The inverted gate also trails GRPO on both benchmarks with both executors. These results support the predictive selection rule: randomly reducing supervision does not recover AdviSD’s gains, and prioritizing low-contrast decisions does worse than reward learning alone. Next, at an original abstention, no advice is issued, so the with- and without-advice contexts are identical and $c _ { k } = 0$ . Reflection may still flag missed advice, in which case the bypass lets the abstention receive self-distillation. Removing the bypass lowers performance by 0.6–1.7 points across both benchmarks and executors. The no-bypass variant, which selects only among issued-advice decisions, still exceeds GRPO in every aggregate. Appendix E.1 reports supervision allocation in one Claude BFCL run.

![](images/9fcf86b0bf31cb7a3777872973b92945122d54ac6c38923904dcc76de203f215.jpg)

![](images/6519bec6a1871765df3c891b0a337dd09a20ec13074247a4c62f410b74d2dc95.jpg)  
No advisor API advisor Untrained Qwen3-8B GEPA GRPO SDPO DistIL AdviSD (ours)  
Figure 2 | Cross-version and cross-family BFCL-v3 transfer without retraining (Q3). Train/Evaluate labels identify executors; API advisors use the evaluation executor’s model. Dots show means; intervals indicate ± sample SD. Gains show AdviSD minus GRPO in % points. All panels share a 48–66% accuracy scale.

Table 3 | Out-of-domain (OOD) performance without retraining (Q3). BFCL-selected systems; mean ± sample SD follows the evaluation protocol. Metrics (%): ACEBench success, ToolHop answer correctness (AC), $\tau ^ { 2 } .$ -bench pass<sup>1</sup>, and RoTBench tool selection (TS), parameter identification (PI), and content filling (CF). Macro Avg averages components within each benchmark, then the four benchmarks equally (Appendix E). Bold/underline mark the largest/second-largest distinct means per column and executor, including ties.
<table><tr><td rowspan="2">Adaptation method</td><td colspan="2">ACEBench</td><td>|ToolHop |</td><td colspan="3"> $\tau ^ { 2 } .$  bench</td><td colspan="3">RoTBench</td><td rowspan="2">Macro Avg</td></tr><tr><td>M-Step</td><td>M-Turn</td><td>AC</td><td>Airline</td><td>Retail</td><td>Telecom</td><td>TS</td><td>PI</td><td>CF</td></tr><tr><td colspan="10">Frozen executor: Gemini 3.7 Flash</td></tr><tr><td>Standalone executor: no advisor</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No advisor</td><td>86.7±2.9 66.7±5.8</td><td></td><td>81.3±0.4</td><td></td><td></td><td>84.0±2.0 58.2±1.3 91.5±1.3</td><td></td><td>|71.9±0.8 66.1±0.4 47.8±0.2 | 74.5</td><td></td><td></td></tr><tr><td colspan="10">Executor prompt optimization: no advisor</td></tr><tr><td>GEPA</td><td></td><td>85.0±2.2 63.9±2.1</td><td>80.6±0.1</td><td></td><td></td><td>80.0±3.5 55.8±1.3 88.6±0.9</td><td></td><td>68.8±0.4 63.4±0.3 45.9±0.2</td><td></td><td>72.3</td></tr><tr><td>Frozen advisors: no task-specific optimization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.7 Flash advisor</td><td>91.7±2.9</td><td>70.0±5.8</td><td>80.4±0.2</td><td></td><td></td><td>89.3±2.3 58.8±2.3 83.6±1.0</td><td></td><td>67.3±0.2 61.1±0.4 45.7±0.4</td><td></td><td>74.1</td></tr><tr><td>Untrained Qwen3-8B</td><td>85.0±5.0</td><td>61.1±7.7</td><td>81.1±0.3</td><td></td><td></td><td>82.0±2.0 55.8±2.0 83.3±1.8</td><td></td><td>70.2±0.4 65.1±0.3 46.9±0.3</td><td></td><td>72.1</td></tr><tr><td colspan="10">Trained advisors: Qwen3-8B weight updates</td></tr><tr><td>GRPO</td><td>93.3±2.9</td><td>68.9±1.9</td><td>80.5±0.3</td><td></td><td></td><td>86.7±4.2 59.1±2.7 90.4±0.9</td><td></td><td></td><td>70.0±0.2 63.8±0.2 47.2±0.4</td><td>75.2</td></tr><tr><td>SDPO</td><td>88.3±2.9</td><td>65.6±8.4</td><td>79.5±0.2</td><td></td><td></td><td>82.7±4.2 54.1±4.1 86.5±2.0</td><td></td><td></td><td>66.8±0.4 62.9±0.4 45.3±0.5</td><td>72.3</td></tr><tr><td>DistIL</td><td>91.7±2.9</td><td>64.4±5.1</td><td>80.6±0.1</td><td></td><td></td><td>80.7±2.3 58.5±2.2 86.3±1.8</td><td></td><td></td><td>68.9±0.9 63.1±0.6 46.3±0.2</td><td>73.3</td></tr><tr><td>ADVISD (ours)</td><td>93.3±2.9</td><td>70.0±5.8</td><td>82.3±0.3</td><td>91.3±1.2 62.0±2.0 92.4±1.8</td><td></td><td></td><td>73.0±0.7 67.3±0.4 49.1±0.3</td><td></td><td></td><td>77.2</td></tr><tr><td colspan="10">Frozen executor: Claude Sonnet 4.6</td></tr><tr><td>Standalone executor: no advisor</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No advisor</td><td>86.7±5.8 66.7±3.3</td><td></td><td>67.0±0.1</td><td></td><td></td><td>87.3±1.2 57.9±2.3 92.7±0.5</td><td></td><td>74.2±0.7 58.9±0.4 42.9±0.5</td><td></td><td>70.4</td></tr><tr><td>Executor prompt optimization: no advisor</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GEPA</td><td>85.0±8.7 64.4±5.1</td><td></td><td>65.8±0.4</td><td></td><td></td><td>79.3±3.1 55.8±2.0 87.1±2.5</td><td></td><td>73.8±0.4 58.5±0.7 42.5±0.7</td><td></td><td>68.2</td></tr><tr><td>Frozen advisors: no task-specific optimization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Sonnet 4.6 advisor</td><td>81.7±2.9</td><td>68.9±7.7</td><td>66.1±0.4</td><td></td><td></td><td>84.0±2.0 58.5±1.3 87.7±2.3</td><td>77.7±0.7</td><td>66.9±0.5</td><td>50.1±0.8</td><td>70.8</td></tr><tr><td>Untrained Qwen3-8B</td><td>81.7±7.6 64.4±6.9</td><td></td><td>66.7±0.5</td><td></td><td></td><td>77.3±1.2 51.8±1.5 86.0±0.9</td><td>73.0±0.5</td><td>57.3±0.7</td><td>41.2±0.4</td><td>67.2</td></tr><tr><td>Trained advisors: Qwen3-8B weight updates</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>86.7±5.8</td><td>67.8±1.9</td><td>66.5±0.3</td><td>82.7±4.6 59.6±2.3</td><td></td><td>90.6±2.5</td><td>77.9±1.0</td><td></td><td>65.4±0.4 48.3±0.4</td><td>71.3</td></tr><tr><td>SDPO DistIL</td><td>85.0±8.7</td><td>66.7±3.3</td><td>65.4±0.2 66.1±0.3</td><td></td><td></td><td>76.7±3.1 57.0±1.5 86.8±2.6</td><td></td><td></td><td>74.3±0.8 62.4±0.1 45.9±0.3</td><td>68.9 68.4</td></tr><tr><td></td><td>83.3±10.4</td><td>61.1±5.1</td><td></td><td></td><td></td><td>79.3±4.6 56.1±3.888.6±2.3</td><td></td><td></td><td>75.1±0.2 62.1±0.2 45.0±0.9</td><td></td></tr><tr><td>ADVISD (ours)</td><td>85.0±5.0</td><td>70.0±5.8</td><td>68.8±0.5</td><td>88.0±3.5 64.9±2.3 93.3±1.3</td><td></td><td></td><td></td><td>79.8±0.4 70.0±0.3 52.6±0.1</td><td></td><td>74.0</td></tr></table>

(Q3) Transfer without retraining. We evaluate the BFCL-selected advisors and prompts on ACEBench (Chen et al., 2025), ToolHop (Ye et al., 2025), �<sup>2</sup>-bench (Barres et al., 2026), and RoTBench (Ye et al., 2024) without further adaptation, using each benchmark’s native tools, policies, and metrics. AdviSD’s gains carry over to these benchmarks: it leads the four-benchmark macro-average for both executors (Table 3). It exceeds standalone execution by 2.7 points with Gemini and 3.6 with Claude, while GRPO’s margin is less than one point. AdviSD has the highest mean, outright or tied, in 17 of 18 comparisons. By contrast, BFCL-optimized GEPA prompts fall below standalone execution on every evaluated component, indicating its poor cross-benchmark transfer. The efect of advice still varies by task. On $\tau ^ { 2 } .$ -bench telecom, the frozen Gemini advisor scores 7.9% points below standalone execution, whereas AdviSD scores slightly above it. AdviSD itself trails standalone execution and GRPO by 1.7 % points on Claude’s ACEBench multi-step tasks. Useful guidance also transfers across executor versions and families. AdviSD leads all four transfer comparisons (Figure 2) and exceeds transferred GRPO by 3.1 percentage points in both cross-family directions. Its cross-family scores still fall below those of AdviSD advisors trained for the evaluation executor (Table 1), so transferred

![](images/8958e4a3a239aaffff5c85ffbeee5e353ce2c8be610a37f80dd403d05fbe1c6c.jpg)  
Figure 3 | Training progress on BFCL-v3. Validation accuracy across training updates with Claude Sonnet 4.6 as the frozen executor. Mean ± sample standard deviation over three independent training runs. AdviSD stays ahead of the baselines from update 20 onward.

model does not replace executor-specific training.

(Q4) Training progress. On BFCL-v3 validation with Claude Sonnet 4.6, AdviSD leads the compared methods by update 20 and remains ahead at every later evaluated checkpoint. Late in training, its margin over GRPO is roughly four percentage points (Figure 3).

## 8. Conclusion

AdviSD combines task rewards with targeted self-distillation to train small advisors for frozen frontier executors. It separates proposing corrections from selecting which advice decisions to supervise. Selection compares how the advisor scores the same recorded executor response with and without its issued advice, and a feedback-conditioned teacher supervises the retained decisions. Our sharedparameter analysis explains how eventual performance depends on which corrections are retained and how often supervision occurs, and it motivates the matched-count random control. AdviSD has the highest in-domain aggregates on BFCL-v3 and EnvScaler among the compared methods. Without retraining, it also leads the out-of-domain macro-averages and transfer comparisons across executor versions and model families. Its gains over ungated and matched-count random supervision indicate that choosing where to learn matters beyond reducing supervision. Appendix E discusses theoretical scope, the predictive score, and evaluation limitations.

## References

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=3z KtaqxLhW.

L. A. Agrawal, S. Tan, D. Soylu, N. Ziems, R. Khare, K. Opsahl-Ong, A. Singhvi, H. Shandilya, M. J. Ryan, M. Jiang, C. Potts, K. Sen, A. Dimakis, I. Stoica, D. Klein, M. Zaharia, and O. Khattab. GEPA: Reflective prompt evolution can outperform reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026a. URL https://openreview.net/forum?id=RQm2KQTM5r.

R. Agrawal, J. Fein-Ashley, and P. Rashidinejad. Reinforcement learning from rich feedback with distributional dagger. arXiv preprint arXiv:2606.05152, 2026b.

P. Asawa, A. Zhu, A. O’Neill, M. Zaharia, A. Dimakis, and J. E. Gonzalez. How to train your advisor: Steering black-box LLMs with advisor models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=AvRUMTdFzX.

V. Barres, H. Dong, S. Ray, X. Si, and K. R. Narasimhan. \$\tau^2\$-bench: Evaluating conversational agents in a dual-control environment. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=OC2z7iSQKa.

C. Chen, X. Hao, W. Liu, X. Huang, X. Zeng, S. Yu, D. Li, S. Wang, W. Gan, Y. Huang, et al. Acebench: Who wins the match point in tool usage? arXiv preprint arXiv:2501.12851, 2025.

Y. Chen, Y. Sun, H. Wang, J. Wang, X. Zhang, X. Shen, W. Li, and W. Zhang. Exact is easier: Credit assignment for cooperative llm agents. arXiv preprint arXiv:2603.06859, 2026.

K. Ethayarajh, Y. Choi, and S. Swayamdipta. Understanding dataset dificulty with V-usable information. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 5988–6008. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/ethayarajh22a.html.

P. Fernandes, K. Yin, G. Neubig, and A. F. Martins. Measuring and increasing context usage in context-aware machine translation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6467–6478, 2021.

J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson. Counterfactual multi-agent policy gradients. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Z. Han, J. Xiao, Z. Lu, R. Jin, Z. Yao, Y. Liu, H. Hao, Y. Sun, Y. Yang, Q. Gu, et al. Distill where you fail: Recovering learning signals of negative rl-groups from adaptive teacher guidance. arXiv preprint arXiv:2608.00782, 2026.

J. Hübotter, F. Lübeck, L. D. Behric, A. Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. K. Buening, C. Guestrin, and A. Krause. Reinforcement learning via self-distillation. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?i d=QkfkxyRizZ.

N. Jaques, A. Lazaridou, E. Hughes, C. Gulcehre, P. Ortega, D. Strouse, J. Z. Leibo, and N. De Freitas. Social influence as intrinsic motivation for multi-agent deep reinforcement learning. In International conference on machine learning, pages 3040–3049. PMLR, 2019.

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica. Eficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

C. Li, Y. Zhuang, R. Qiang, H. Sun, H. Dai, C. Zhang, and B. Dai. Matryoshka pilot: Learning to drive black-box llms with llms. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 44491–44544. Curran Associates, Inc., 2025. doi: 10.52202/085713-1482. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/3f055edce 9cc3e90514b0716a16b37b4-Paper-Conference.pdf.

M. Li, L. Chen, J. Chen, S. He, J. Gu, and T. Zhou. Selective reflection-tuning: Student-selected data recycling for LLM instruction-tuning. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Associationfor Computational Linguistics: ACL 2024, pages 16189–16211, Bangkok, Thailand, Aug. 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.958. URL https://aclanthology.org/2024.findings-acl.958/.

M. Li, Y. Zhang, Z. Li, J. Chen, L. Chen, N. Cheng, J. Wang, T. Zhou, and J. Xiao. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7602–7635, 2024b.

Z. Li, B. Peng, P. He, M. Galley, J. Gao, and X. Yan. Guiding large language models via directional stimulus prompting. Advances in Neural Information Processing Systems, 36:62630–62656, 2023.

Z. Li, W. Tian, J. Chen, H. Zhang, Y. Liu, Y. Ban, and F. Zhuang. Counterfactual credit policy optimization for multi-agent collaboration. arXiv preprint arXiv:2603.21563, 2026.

A. Liu, X. Han, Y. Wang, Y. Tsvetkov, Y. Choi, and N. A. Smith. Tuning language models by proxy. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=dr ibhnhm1i.

H. Liu, Y. Zhang, X. Li, B. Lyu, and J. Shang. Hero: Hindsight-enhanced reflection from environment observations for agentic self-distillation. arXiv preprint arXiv:2606.11559, 2026.

D. Lopez-Paz, L. Bottou, B. Schölkopf, and V. Vapnik. Unifying distillation and privileged information. arXiv preprint arXiv:1511.03643, 2015.

A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegrefe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark. Selfrefine: Iterative refinement with self-feedback. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=S37hOerQLB.

L. Pan, S. Tao, Y. Zhai, L. Zhang, Z. Liu, B. Ding, A. Liu, and L. Wen. Rlcsd: Reinforcement learning with contrastive on-policy self-distillation. arXiv preprint arXiv:2606.11709, 2026.

S. G. Patil, H. Mao, F. Yan, C. C.-J. Ji, V. Suresh, I. Stoica, and J. E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.n et/forum?id=2GmDdhBdDk.

B. Peng, J. Quesnelle, H. Fan, and E. Shippole. YaRN: Eficient context window extension of large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=wHBfxhZu1u.

J. Peters, K. Mulling, and Y. Altun. Relative entropy policy search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 24, pages 1607–1612, 2010.

R. Pryzant, D. Iter, J. Li, Y. Lee, C. Zhu, and M. Zeng. Automatic prompt optimization with “gradient descent” and beam search. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 7957–7968, 2023.

S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In G. Gordon, D. Dunson, and M. Dudík, editors, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings

of Machine Learning Research, pages 627–635, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL https://proceedings.mlr.press/v15/ross11a.html.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

N. Shinn, F. Cassano, A. Gopinath, K. R. Narasimhan, and S. Yao. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=vAElhFcKW6.

X. Song, H. Chang, G. Dong, Y. Zhu, J.-R. Wen, and Z. Dou. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 8326–8357, 2026.

Y. Tian, R. Wang, X. Wen, J. Li, S. Sun, L. Song, J. Bian, and B. Zhao. Pbsd: Privileged bayesian self-distillation for long-horizon credit assignment. arXiv preprint arXiv:2606.09348, 2026.

J. Wang, X. Ouyang, Z. Chen, Y. Hu, Z. Pan, X. Li, and L.-Z. Guo. Trace: Distilling where it matters via token-routed self on-policy alignment. arXiv preprint arXiv:2605.10194, 2026a.

Y. Wang, Z. Wang, B. Zeng, R. Zhang, W. Liu, L. Yang, Y. Dai, Y. Shi, B. Li, C. Tong, et al. Flux-opd: On-policy distillation with evolving contexts. arXiv preprint arXiv:2607.28022, 2026b.

H. Xu, J. Wang, Y. Yang, C. Zhu, F. Chen, Z. Wu, J. Cai, and Y. Song. Dart-sd: Diamond-topology aware retrieval and tuning for self-distillation of multi-turn tool-calling agents. arXiv preprint arXiv:2608.18524, 2026.

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

C. Yang, C. Qin, Q. Si, M. Chen, N. Gu, D. Yao, Z. Lin, W. Wang, J. Wang, and N. Duan. Self-distilled rlvr. arXiv preprint arXiv:2604.03128, 2026a.

Y. Yang, C. Qin, X. Liu, C. Chen, Q. Dong, Y. Zhang, C. Liu, Z. Yang, L. Pan, J. Lin, et al. Agentic reinforcement learning with observation-calibrated self-distillation. arXiv preprint arXiv:2608.04788, 2026b.

J. Ye, Y. Wu, S. Gao, C. Huang, S. Li, G. Li, X. Fan, Q. Zhang, T. Gui, and X.-J. Huang. Rotbench: A multi-level benchmark for evaluating the robustness of large language models in tool learning. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 313–333, 2024.

J. Ye, Z. Du, X. Yao, W. Lin, Y. Xu, Z. Chen, Z. Wang, S. Zhu, Z. Xi, S. Yuan, et al. Toolhop: A query-driven benchmark for evaluating large language models in multi-hop tool use. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2995–3021, 2025.

W. Yeo, Y. Choi, T. Ki, and S. J. Hwang. Hint-sd: Targeted hindsight self-distillation for long-horizon agents. arXiv preprint arXiv:2605.17873, 2026.

M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, P. Lu, Z. Huang, C. Guestrin, and J. Zou. Optimizing generative AI by backpropagating language model feedback. Nature, 639(8055):609–616, 2025. doi: 10.1038/s41586-025-08661-4. URL https://www.nature.com/articles/s41586-025 -08661-4.

G. Zhang, J. Lyu, R. Sun, X. Yu, H. Zhao, Q. Ren, and S. Yan. Latent on-policy self-distillation. arXiv preprint arXiv:2608.13040, 2026.

A. Zhao, D. Huang, Q. Xu, M. Lin, Y.-J. Liu, and G. Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

Y. Zhou, L. Zhang, Y. Wu, M. Wang, B. Peng, J. Liu, X. Fan, and Z. Zhao. Sage-opd: Selective agent-guided intervention for multi-turn on-policy distillation. arXiv preprint arXiv:2606.19659, 2026.

## Appendices

A Notation and scope 18   
B A single update through the executor 19   
C Retained corrections and the learning limit 21   
C.1 Proof of Theorem 1 in the two-teacher model 22   
C.2 Proof of Theorem 2 and Corollary 1 in the two-teacher model 25   
C.3 Stationary random teachers and action-dependent selection 29   
D Training and implementation details 33   
E Baselines and evaluation protocol 35   
E.1 Training-time supervision dynamics 38   
F Exact prompts 39   
G Paired qualitative evidence 42   
H Extended Related Work 44   
H.1 Advising and prompt optimization 44   
H.2 Feedback-conditioned distillation 44   
H.3 Predictive contrasts and selection 45

## A. Notation and scope

Logarithms are natural, and vector norms are Euclidean. We compute an expected training gradient by diferentiating the student loss with the sampled data, teacher, supports, and selection weights held fixed, and then averaging over the sampling law; the rollout distribution is not diferentiated. In the local analysis, � is the advisor’s parameter vector. In the learning model, it is a single scalar logit shared across situations. That model assumes independent opportunities, uniform proposals, and stationary conditional statistics. It explains a selection mechanism and does not describe the full dynamics of the reflector, the evolving language model, or the optimizer.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td colspan="2">Local update analysis</td></tr><tr><td> $h$ </td><td>Advice prefix at a fixed interaction state.</td></tr><tr><td> $S _ { h }$ </td><td>Fixed finite set of candidate tokens at prefix  $h .$ </td></tr><tr><td> $p _ { \theta }$ </td><td>Student next-token distribution, normalized on  $S _ { h } .$ </td></tr><tr><td> $q _ { h }$   $K _ { h , \upsilon }$ </td><td>Teacher next-token distribution on  $S _ { h } ,$  held fixed during differentiation. Law of executor responses and outcomes after token v and completion by the pre-update</td></tr><tr><td></td><td>advisor; excludes the advice text.</td></tr><tr><td> $V _ { h } ( \nu )$ </td><td>Expected task score under the execution law  $K _ { h , \nu } .$ </td></tr><tr><td> $J _ { h } ( \theta )$ </td><td>Local execution value:  $V _ { h } ( \nu )$  averaged under the student&#x27;s next-token distribution.</td></tr><tr><td> $g _ { h }$ </td><td>Student-to-teacher reverse-KL gradient evaluated at the pre-update parameters.</td></tr><tr><td> $e _ { h }$ </td><td>Teacher-dependent residual relative to the value-tilted reference, in  $g _ { h } = - \alpha \nabla J _ { h } + e _ { h } .$ </td></tr><tr><td colspan="2">Repeated-learning model</td></tr><tr><td> $M$ </td><td>Number of independent opportunities per modeled episode.</td></tr><tr><td> $b$ </td><td>Maximum number of failure proposals per modeled episode.</td></tr><tr><td> $\beta$ </td><td>Probability that an opportunity is execution-insensitive.</td></tr><tr><td> $r _ { j a }$ </td><td>Retention probability for a proposed failure of type j with issued action  $^ { a ; }$  excludes proposal sampling.</td></tr><tr><td> $\ell _ { j a }$ </td><td>Expected teacher log-odds for type j and issued action  $^ { a , }$  conditional on failure and retention.</td></tr><tr><td> $B ( \theta )$ </td><td>Exposure: probability that an episode retains at least one correction.</td></tr><tr><td> $\mu ( \theta )$ </td><td>Composition: expected within-episode mean of retained teacher log-odds, conditional on retaining at least one correction.</td></tr><tr><td colspan="2">ApvıSD contexts and selection</td></tr><tr><td> $b _ { \mathrm { r e f l } }$ </td><td>Maximum number of advice decisions flagged by reflection per eligible episode; distinct from the modeled proposal cap b.</td></tr><tr><td> $g _ { k }$ </td><td>Advisor context before decision k: visible interaction, tool schemas, and earlier advice.</td></tr><tr><td> $h _ { k }$ </td><td>Native executor history before response  $k ;$  distinct from the theoretical advice prefix h.</td></tr><tr><td> $C _ { k } ^ { + }$ </td><td>Paired scoring context containing pre-response history, tool schemas, and issued advice.</td></tr><tr><td> $C _ { k } ^ { - }$ </td><td>The same scoring context with the issued advice omitted.</td></tr><tr><td> $\zeta _ { i , k }$ </td><td>Feedback block with local execution evidence, relevant checks, episode score, and reflection feedback, prepended to the teacher&#x27;s context at decision k in episode i.</td></tr><tr><td> $c _ { k }$ </td><td>Signed mean token log-score contrast for the same recorded executor response, with issued advice versus no advice.</td></tr><tr><td> $d _ { j }$ </td><td>Corresponding contrast for pilot pair  $j ,$  replacing issued advice with advice borrowed from another task.</td></tr><tr><td> $\epsilon _ { c }$ </td><td>Per-run fixed selection threshold, calibrated from an empirical quantile of absolute pilot-donor contrasts  $| d _ { j } | .$ </td></tr></table>

## B. A single update through the executor

Lemma 1 separates fitting a feedback-conditioned teacher from improving execution. The two difer because the distillation loss compares advice-token distributions, whereas task performance depends on how the executor responds to the completed advice. We first express both at one recorded prefix so that they can be compared directly. This gives the gradient identity in Section 4.1 and an extension to situations where execution depends only weakly on the next token. The extension matters because exact insensitivity is a boundary case: it quantifies how weak token–execution dependence limits the value component when the reward range and gradient scale are controlled.

Fix an interaction situation $\sigma ,$ including the environment state and the advisor’s and executor’s histories before advice. For complete advice �, let $K _ { \sigma } ( a )$ denote the law of the subsequent executor response, tool outcomes, transitions, and termination. The advice text itself is excluded from this law; otherwise changing the advice would change the recorded outcome even if the executor behaved identically. At a recorded advice prefix $h ,$ choose a next token $\nu$ and complete the advice using the pre-update advisor. If $C _ { h , \upsilon }$ is the resulting distribution over complete advice strings, the induced execution law is

$$
K _ { h , \nu } = \mathbb { E } _ { a \sim C _ { h , \nu } } K _ { \sigma } ( a ) .
$$

Thus $K _ { h , \upsilon }$ already averages over stochastic advice completion and execution. For a bounded task score $W \in [ w _ { - } , w _ { + } ]$ , define

$$
V _ { h } ( \nu ) = \mathbb { E } _ { K _ { h , v } } W , \qquad J _ { h } ( \theta ) = \sum _ { \nu \in S _ { h } } p _ { \theta } ( \nu ) V _ { h } ( \nu ) .\tag{8}
$$

The first quantity is the expected score after choosing $\nu ;$ the second averages these values using the student’s next-token probabilities. Only this next-token distribution varies in $J _ { h }$ . Holding the completion and execution laws fixed allows us to ask what changing the token preference alone would accomplish, without also changing the policies used for later decisions.

Assumption 1 (Fixed-prefix conditional update). The finite support $S _ { h }$ , context, prefix, completion and execution laws, objective, and loss weights arefixed during diferentiation. Student logits are diferentiable nea<sup>r</sup> <sup>th</sup>e p<sup>r</sup>e<sup>-</sup>up<sup>d</sup>a<sup>t</sup>e pa<sup>r</sup>ame<sup>t</sup>e<sup>r</sup>s $\bar { \theta } ;$ th<sub>e</sub> <sub>s</sub>t<sub>u</sub>d<sub>en</sub>t $p \theta$ <sub>an</sub>d d<sub>e</sub>t<sub>ac</sub>h<sub>e</sub>d t<sub>eac</sub>h<sub>er</sub> $q _ { h }$ <sub>a</sub>r<sub>e</sub> <sub>pos</sub>itiv<sub>e</sub> <sub>a</sub>nd n<sub>o</sub>rm<sub>a</sub>li<sub>ze</sub>d on $S _ { h } , ~ I f$ th<sub>e</sub> t<sub>eac</sub>h<sub>er</sub> i<sub>s ran</sub>d<sub>om,</sub> $\mathbb { E } [ - \log q _ { h } ( \nu ) ] < \infty$ for each supported token.

The support can be the full vocabulary or a restricted set fixed for the update, as in Section 5.3. Positivity ensures that the log ratios and reverse KL are finite, and finite softmax logits satisfy it. A token whose student probability is identically zero in a neighborhood of $\bar { \theta }$ can be omitted from the support; in contrast, a zero teacher probability at positive student mass makes the reverse KL infinite and is not covered here.

Write $\bar { p } = p _ { \bar { \theta } } , \phi _ { h } ( \nu ) = \nabla _ { \theta } \log p _ { \theta } ( \nu ) | _ { \bar { \theta } } ,$ and $R _ { W } = \psi _ { + } - w _ { - }$ . The vector $\phi _ { h } ( \nu )$ describes how the student’s log-probability of � changes with its parameters. To compare a feedback-conditioned teacher with execution value, introduce the reference distribution

$$
q _ { h } ^ { \alpha } ( \nu ) = \frac { \bar { p } ( \nu ) e ^ { \alpha V _ { h } ( \nu ) } } { Z _ { \alpha } } , \qquad Z _ { \alpha } = \sum _ { u \in S _ { h } } \bar { p } ( u ) e ^ { \alpha V _ { h } ( u ) } , \quad \alpha \ge 0 .
$$

For positive $\alpha ,$ this distribution shifts probability toward higher-value tokens, but only as far as a penalty for departing from the current student allows. Indeed, for any distribution $q$ on $S _ { h }$

$$
\begin{array} { r } { \mathbb { E } _ { q } V _ { h } - \alpha ^ { - 1 } \operatorname { K L } ( q \| \bar { p } ) = \alpha ^ { - 1 } \log Z _ { \alpha } - \alpha ^ { - 1 } \operatorname { K L } ( q \| q _ { h } ^ { \alpha } ) . } \end{array}
$$

The right-hand side is uniquely maximized at $q = q _ { h } ^ { \alpha } . \mathrm { A t } \alpha = 0 .$ the definition reduces to $q _ { h } ^ { 0 } = \bar { p }$ AdviSD never constructs this reference teacher. We use it only as a comparison with a known relation

to execution value, which shows what changes when the actual feedback-conditioned teacher is used instead.

For the weak-dependence statement, define

$$
F _ { h } = \mathbb { E } _ { \bar { p } } \| \phi _ { h } \| _ { 2 } ^ { 2 } , \qquad I _ { h } = \mathrm { I } ( X ; Z ) , \quad X \sim \bar { p } , \quad Z \sim K _ { h , X } .
$$

Here $F _ { h }$ controls the scale of the log-probability gradients, and $I _ { h }$ measures how informative the sampled token is about the ensuing execution. Both belong to the conditional model just defined; neither is the predictive contrast used by AdviSD.

Lemma 1 (Execution-value component and teacher residual). Under Assumption 1, let $\begin{array} { r l } { g _ { h } } & { { } = } \end{array}$ $\nabla _ { \theta } \ K \mathbf { L } ( p _ { \theta } \| q _ { h } ) \vert _ { \bar { \theta } } .$ . For everyfixed $\alpha \geq 0 ,$

$$
g _ { h } = - \alpha \nabla J _ { h } ( \bar { \theta } ) + e _ { h } , \qquad e _ { h } = \mathbb { E } _ { \bar { p } } \left[ \phi _ { h } \log \frac { q _ { h } ^ { \alpha } } { q _ { h } } \right] ,\tag{9}
$$

$$
\| g _ { h } - e _ { h } \| _ { 2 } = \alpha \| \nabla J _ { h } ( \bar { \theta } ) \| _ { 2 } \le \alpha R _ { W } \sqrt { F _ { h } I _ { h } / 2 } .\tag{10}
$$

If all supported execution laws $K _ { h , \upsilon }$ <sub>co</sub>i<sub>nc</sub>id<sub>e,</sub> th<sub>en</sub>

$$
\nabla J _ { h } ( \bar { \theta } ) = 0 , \qquad q _ { h } ^ { \alpha } = \bar { p } , \qquad g _ { h } = e _ { h } .
$$

The identities hold for each realized teacher and may be averaged over its law.

Proof. We begin with the gradient of the conditional distillation loss. Diferentiating $\begin{array} { r } { \sum _ { \nu } p _ { \theta } ( \nu ) = 1 } \end{array}$ gives

$$
\mathbb { E } _ { \bar { p } } \phi _ { h } = \sum _ { \nu \in S _ { h } } \nabla _ { \theta } p _ { \theta } ( \nu ) | _ { \bar { \theta } } = 0 .
$$

Because the teacher is held fixed, diferentiating the reverse KL gives

$$
\begin{array} { l } { { \displaystyle g _ { h } = \sum _ { \nu \in S _ { h } } \bar { p } ( \nu ) \phi _ { h } ( \nu ) \left[ \log \frac { \bar { p } ( \nu ) } { q _ { h } ( \nu ) } + 1 \right] } } \\ { { \displaystyle \quad = \mathbb { E } _ { \bar { p } } \left[ \phi _ { h } \log \frac { \bar { p } } { q _ { h } } \right] . } } \end{array}
$$

The reference teacher lets us rewrite the log ratio as

$$
\log \frac { \bar { p } ( \nu ) } { q _ { h } ( \nu ) } = \log \frac { q _ { h } ^ { \alpha } ( \nu ) } { q _ { h } ( \nu ) } - \alpha V _ { h } ( \nu ) + \log Z _ { \alpha } .
$$

The last term is constant across tokens, so its contribution vanishes by the zero-mean score identity. Moreover, the fixed execution values in Eq. (8) satisfy

$$
\nabla J _ { h } ( \bar { \theta } ) = \sum _ { \nu } \bar { p } ( \nu ) \phi _ { h } ( \nu ) V _ { h } ( \nu ) = \mathbb { E } _ { \bar { p } } [ \phi _ { h } V _ { h } ] .
$$

Substituting these two facts gives $g _ { h } = e _ { h } - \alpha \nabla J _ { h } ( \bar { \theta } )$ , which is the identity in Eq. (9). If all $K _ { h , \upsilon }$ coincide, all tokens have the same value. The value gradient is then zero, and the common exponential factor cancels in $q _ { h } ^ { \alpha }$ . Consequently $q _ { h } ^ { \alpha } = \bar { p }$ and $g _ { h } = e _ { h }$

To obtain the quantitative bound, consider the execution mixture $\begin{array} { r } { \nu _ { h } = \sum _ { \nu } \bar { p } ( \nu ) K _ { h , \nu } . } \end{array}$ Its expected score is $\mathbb { E } _ { \nu _ { h } } W = J _ { h } ( \bar { \theta } )$ . Centering the value in the gradient is permissible because $\mathbb { E } _ { \bar { p } } \phi _ { h } = 0$ , giving

$$
\nabla J _ { h } ( \bar { \theta } ) = \mathbb { E } _ { \bar { p } } \big [ \phi _ { h } ( \upsilon ) \big ( V _ { h } ( \upsilon ) - J _ { h } ( \bar { \theta } ) \big ) \big ] .
$$

The vector Cauchy–Schwarz inequality now gives

$$
\begin{array} { r } { \| \nabla J _ { h } ( \bar { \theta } ) \| _ { 2 } ^ { 2 } \leq F _ { h } \operatorname { V a r } _ { \bar { p } } ( V _ { h } ) . } \end{array}\tag{11}
$$

It remains to relate the variation in token values to the variation in their execution laws. Because � has range $R _ { W }$ , the bounded-function characterization of total variation gives

$$
| V _ { h } ( \nu ) - J _ { h } ( \bar { \theta } ) | \leq R _ { W } \mathrm { T V } ( K _ { h , \nu } , \nu _ { h } ) .
$$

Pinsker’s inequality then implies

$$
\big ( V _ { h } ( \nu ) - J _ { h } ( \bar { \theta } ) \big ) ^ { 2 } \leq \frac { R _ { W } ^ { 2 } } { 2 } \mathrm { K L } ( K _ { h , \nu } \| \nu _ { h } ) .
$$

Averaging over the token choice and using the conditional-law expression for mutual information yields

$$
\operatorname { V a r } _ { \bar { p } } ( V _ { h } ) \leq \frac { R _ { W } ^ { 2 } } { 2 } \sum _ { \nu } \bar { p } ( \nu ) \operatorname { K L } ( K _ { h , \nu } \| \nu _ { h } ) = \frac { R _ { W } ^ { 2 } I _ { h } } { 2 } .
$$

Combining this with Eq. (11) proves Eq. (10). All terms are finite: $\nu _ { h } \ge \bar { p } ( \upsilon ) K _ { h , \upsilon }$ implies $\mathbb { K L } ( K _ { h , \nu } | | \nu _ { h } ) \le$ − log $\bar { p } ( \upsilon )$ , and the support is finite and positive. Finally, the teacher log-moment assumption makes each coordinate of the teacher-dependent gradient integrable, which permits the stated averaging over a random teacher. □

The identity explains what fitting the reference teacher would do: its gradient-descent direction is $\alpha \nabla J _ { h }$ . Fitting another teacher adds the residual direction $- e _ { h } ,$ which can reinforce or oppose value ascent. When execution is insensitive to the next token, there is no local value component at all, but teacher fitting can still move the advisor. The bound extends this observation to execution that is only nearly insensitive: for fixed $\alpha ,$ with the reward range and gradient scale controlled, weak token–execution dependence makes the value component small. The bound says nothing about the size of the residual.

The comparison holds at the pre-update parameters; it makes no claim about the outcome of a finite optimization step. The reference parameter � changes the decomposition but not the actual gradient �<sub>ℎ</sub>; the residual need not be orthogonal to value ascent and is not a uniquely identified causal component. Likewise, the fixed-prefix derivative does not diferentiate either the continuation policy or the probability of visiting the recorded prefix. It is therefore not, in general, a full-policy execution gradient or a full-sequence KL gradient. Complete-advice insensitivity implies the tokenlevel condition only when the advice family covers every supported completion, rather than a few selected strings.

For advisor learning, this means that an update with no local execution-value component can still afect advice elsewhere through shared parameters, and the lemma alone does not say whether that efect helps or hurts. Appendix C.1 examines repeated learning under explicit teacher assumptions and shows how persistent, weaker corrections can limit performance, which motivates targeted supervision. Neither result, however, claims that every insensitive correction is harmful, and neither identifies AdviSD’s predictive score with $I _ { h }$

## C. Retained corrections and the learning limit

The local analysis in Appendix B leaves open what happens when the advisor repeatedly learns from feedback. An update changes its advice, which changes the failures encountered on later rollouts and hence the corrections available for training. This appendix makes that feedback loop explicit. We first prove Theorem 1 for the two-teacher model, then add reward learning to establish Theorem 2 and Corollary 1. Finally, we allow stationary random teachers and action-dependent selection to identify which parts of the argument depend on having only one teacher per type.

## C.1. Proof of Theorem 1 in the two-teacher model

Theorem 1 concerns a shared preference used in two kinds of situation. Advice can improve execution in one kind, but not the other. The proof will show how the mixture of retained corrections sets the target of this shared preference. In particular, we must account for the actual episode loss: averaging over a random number of retained corrections is not the same as replacing that number by its expectation.

Let $\beta \in ( 0 , 1 )$ be the probability of an insensitive situation. Both advice choices induce the same execution law there, with success probability $s _ { I } \in ( 0 , 1 )$ . In sensitive situations, choices 1 and 2 succeed with probabilities $s _ { 1 }$ and $s _ { 2 } .$ , where $0 < s _ { 2 } < s _ { 1 } < 1$ . The advisor chooses advice 1 with probability $p ( \theta ) = 1 / ( 1 + e ^ { - \theta } )$ , using the same log-odds $\theta \in \mathbb { R }$ in every situation. Its expected success is therefore

$$
\begin{array} { r } { J ( \theta ) = \beta s _ { I } + ( 1 - \beta ) \big [ p ( \theta ) s _ { 1 } + ( 1 - p ( \theta ) ) s _ { 2 } \big ] , \qquad J ^ { \prime } ( \theta ) = ( 1 - \beta ) ( s _ { 1 } - s _ { 2 } ) p ( 1 - p ) > 0 . } \end{array}
$$

Increasing the shared preference for advice 1 thus improves success, even though it has no efect within insensitive situations.

An insensitive failure supplies teacher $Q _ { I { \mathrm { : } } }$ , and a sensitive failure supplies $Q _ { S }$ . These distributions are fixed and positive on both choices. Write $\ell _ { j } = \log [ Q _ { j } ( 1 ) / Q _ { j } ( 2 ) ]$ for their target log-odds and assume $\ell _ { I } < \ell _ { S }$ . This ordering means that the teacher from an insensitive failure gives weaker support to advice 1. This is an assumption about the feedback; equal execution laws do not imply it. For example, the neutral teacher $Q _ { I } = ( 0 . 5 , 0 . 5 )$ and the more decisive $Q _ { S } = ( 0 . 8 , 0 . 2 )$ satisfy it, and neither favors advice 2.

Each episode contains � independent draws of situation, advice, and outcome from this model, where � is a fixed positive integer. If � failures occur, a uniformly sampled subset of size min(�, �) is proposed, where $b \geq 1$ is a fixed integer cap. Each proposal is then retained independently with probability $r _ { I }$ or $r _ { S } ,$ according to its type, with $r _ { I } , r _ { S } \in ( 0 , 1 ]$ . The episode loss averages reverse KL over the retained corrections and is zero when none remain. As in the main text, the sampled data and selections are held fixed when computing the gradient; expectation is taken only afterwards.

For the calculation, write

$$
f _ { I } = 1 - s _ { I } , \qquad f _ { 1 } = 1 - s _ { 1 } , \qquad f _ { 2 } = 1 - s _ { 2 } , \qquad F ( p ) = p f _ { 1 } + ( 1 - p ) f _ { 2 } .
$$

Here $F ( p )$ is the sensitive failure probability, and $0 < f _ { 1 } < f _ { 2 } < 1$ . We also use $A \ : = \ : \beta f _ { I } r _ { I }$ and $V ( p ) = ( 1 - \beta ) F ( p ) r _ { S }$ for the probabilities that an individual opportunity ${ \mathrm { i } } s ,$ respectively, an insensitive or sensitive failure that would pass retention. These quantities do not yet include the proposal cap.

Proof of Theorem 1. For a fixed teacher $Q = ( Q ( 1 ) , Q ( 2 ) )$ with log-odds $\ell ,$ the student’s reverse KL is

$$
\mathrm { K L } ( ( p , 1 - p ) \| Q ) = p \log { \frac { p } { Q ( 1 ) } } + ( 1 - p ) \log { \frac { 1 - p } { Q ( 2 ) } } .
$$

Using $d p / d \theta = p ( 1 - p )$ and log $[ p / ( 1 - p ) ] = \theta _ { ; }$ , its derivative is

$$
\frac { d } { d \theta } \mathrm { K L } ( ( p , 1 - p ) | | Q ) = p ( 1 - p ) \left[ \log \frac { p } { 1 - p } - \log \frac { Q ( 1 ) } { Q ( 2 ) } \right] = p ( 1 - p ) ( \theta - \ell ) .\tag{12}
$$

Thus each correction pulls the shared log-odds toward its teacher’s target. The remaining question is which targets appear in the episode’s retained average.

Each opportunity fails independently with probability $f = \beta f _ { I } + ( 1 - \beta ) F ( p )$ , so � ∼ Binomial $( M , f )$ Conditional on failure, its type is insensitive with probability $\beta f _ { I } / f$ . Conditional on $N = n ,$ the types of the � failures are independent draws from this conditional type law. Sampling min $( n , b )$ of their indices uniformly, without using their types, preserves that law for the proposed corrections. This statement averages over the failure types; it does not assert independence after an entire finite pool of typed failures has been fixed.

A proposed correction is retained and insensitive with probability $q _ { I } = \beta f _ { I } r _ { I } / f = A / f ,$ retained and sensitive with probability $q _ { S } = ( 1 - \beta ) F ( p ) r _ { S } / f = V ( p ) / f .$ , and otherwise dropped. These outcomes are independent across the proposed positions, conditional on $N = n$ . Conditioning on the positions retained leaves their types independent with the renormalized probabilities $q _ { I } / ( q _ { I } + q _ { S } )$ and $q _ { S } / ( q _ { I } + q _ { S } )$ In particular, for every positive retained count $K = k$ , the probability that a retained correction is insensitive and the expected average teacher log-odds are

$$
\omega _ { I } ( \theta ) = \frac { A } { A + V ( p ) } , \qquad \mu ( \theta ) = \omega _ { I } ( \theta ) \ell _ { I } + \big [ 1 - \omega _ { I } ( \theta ) \big ] \ell _ { S } .\tag{13}
$$

Neither expression depends on the failure count or retained count, so the same conditional mean holds after averaging over those counts.

For a realized episode with $K > 0$ , diferentiating its loss while holding the sample fixed gives

$$
\nabla _ { \theta } L _ { \mathrm { e p } } = p ( 1 - p ) \left[ \theta - \frac { 1 } { K } \sum _ { k \mathrm { ~ r e t a i n e d } } \ell _ { j ( k ) } \right] ,
$$

where $j ( k )$ is the correction’s type. The derivative is zero for $K = 0$ . Conditional averaging using Eq. (13) therefore yields

$$
g ( \theta ) = \mathbb { E } \big [ \nabla _ { \theta } L _ { \mathrm { e p } } \big ] = \operatorname* { P r } \big ( K > 0 \big ) p ( 1 - p ) \big [ \theta - \mu ( \theta ) \big ] = B ( \theta ) p ( 1 - p ) \big [ \theta - \mu ( \theta ) \big ] .
$$

This proves Eq. (2) with the actual random denominator of the episode loss. It does not diferentiate the sampling distribution: the dependence of � and $\mu$ on $\theta$ describes how the expected sampled gradient changes between updates, not additional derivative terms within an update.

For completeness, let $u = q _ { I } + q _ { S } = ( A + V ( p ) ) / f$ be a proposal’s retention probability. Conditional on $N = n ,$ , the probability of retaining at least one of the min $( n , b )$ proposals is $1 - ( 1 - u ) ^ { \operatorname* { m i n } ( n , b ) }$ Hence

$$
B ( \theta ) = \sum _ { n = 1 } ^ { M } { \binom { M } { n } } f ^ { n } ( 1 - f ) ^ { M - n } \left[ 1 - ( 1 - u ) ^ { \operatorname* { m i n } ( n , b ) } \right] > 0 .\tag{14}
$$

The cap and episode size change this exposure, but do not change the conditional mean target in Eq. (13).

We next determine how that target changes as advice improves. Since $F ^ { \prime } ( p ) = f _ { 1 } - f _ { 2 } < 0 .$ the sensitive retained-failure weight $V ( p )$ decreases with �, whereas the insensitive weight � stays fixed. Consequently,

$$
\frac { d \omega _ { I } } { d p } = \frac { A ( 1 - \beta ) r _ { S } ( f _ { 2 } - f _ { 1 } ) } { [ A + V ( p ) ] ^ { 2 } } > 0 , \qquad \mu ^ { \prime } ( \theta ) = - ( \ell _ { S } - \ell _ { I } ) \omega _ { I } ^ { \prime } ( \theta ) < 0 .
$$

The learning target therefore falls as advice improves: sensitive failures become rarer, and the weaker insensitive teacher supplies a larger share of the retained supervision.

Let $H ( \theta ) = \theta - \mu ( \theta )$ . Its derivative satisfies $H ^ { \prime } ( \theta ) > 1$ , while $\mu ( \theta )$ remains strictly between $\ell _ { I }$ and $\ell _ { S }$ . It follows that � tends to opposite infinities at the two ends of the real line and has a unique zero $\theta ^ { * } = \mu ( \theta ^ { * } ) \in ( \ell _ { I } , \ell _ { S } )$ . The distillation flow is $\dot { \theta } = - B ( \theta ) p ( 1 - p ) H ( \theta )$ . Its field is smooth and bounded: $B \leq 1$ , the target is bounded, and $p ( 1 - p ) | \theta |$ is bounded. Solutions therefore exist uniquely for all time. Since $B ( \theta ) p ( 1 - p ) > 0$ at every finite $\theta _ { \mathrm { { i } } }$ , the field points upward below $\theta ^ { * }$ and downward above it. By uniqueness, a solution cannot cross this equilibrium. It is thus monotone and bounded, and its limit must be the sole zero of the field, $\theta ^ { * }$ . This proves convergence from every finite initialization.

Finally, divide the numerator and denominator of $\omega _ { I }$ by $r _ { S }$ and set $\rho = r _ { I } / r _ { S }$ . Then

$$
\omega _ { I } ( \theta ; \rho ) = \frac { \rho \beta f _ { I } } { \rho \beta f _ { I } + ( 1 - \beta ) F ( p ) } , \qquad \partial _ { \rho } \omega _ { I } = \frac { \beta f _ { I } ( 1 - \beta ) F ( p ) } { [ \rho \beta f _ { I } + ( 1 - \beta ) F ( p ) ] ^ { 2 } } > 0 .
$$

Thus $\partial _ { \rho } \mu < 0 :$ reducing the relative retention of insensitive corrections raises the mean target at every fixed preference. Implicitly diferentiating the equilibrium equation gives

$$
\frac { d \theta ^ { * } } { d \rho } = \frac { \partial _ { \rho } \mu ( \theta ^ { * } ; \rho ) } { 1 - \partial _ { \theta } \mu ( \theta ^ { * } ; \rho ) } < 0 .\tag{15}
$$

The denominator is positive by the monotonicity just established, and $J ^ { \prime } ( \theta ) > 0 ;$ , so the limiting success also increases as $\rho$ decreases. The equilibrium equation contains neither � nor $^ { b , }$ and depends on retention only through $\rho .$ . Changing episode size or proposal cap, or multiplying both retention probabilities by the same admissible positive factor, therefore leaves the limit unchanged. This completes the proof. □

The result distinguishes the composition of supervision from its frequency. Selective retention changes the target $\mu$ and hence the equilibrium. Uniform thinning changes � but leaves that target unchanged. It can change how quickly the flow moves, though not necessarily by a constant rescaling of time, because � depends nonlinearly on the retention probabilities. This distinction is why we compare AdviSD with a count-matched control as well as with learning from every proposal. With reward learning, exposure also afects the limit, as the next subsection shows.

An illustration with neutral and informative teachers. Take $\beta = s _ { I } = 1 / 2 , s _ { 1 } = 0 . 9 , s _ { 2 } = 0 . 1$ 2 $Q _ { I } = ( 0 . 5 , 0 . 5 )$ , and $Q _ { S } = ( 0 . 8 , 0 . 2 )$ . The insensitive teacher is neutral, whereas the sensitive teacher favors the more successful advice. Solving $\theta = \mu ( \theta )$ gives the following values, rounded to three decimals:
<table><tr><td>Retention ratio  $\rho$ </td><td> $\theta ^ { * }$ </td><td> $p ( \theta ^ { * } )$ </td><td> $\omega _ { I } ( \theta ^ { * } )$ </td><td> $J ( \theta ^ { * } )$ </td></tr><tr><td>1</td><td>0.601</td><td>0.646</td><td>0.566</td><td>0.558</td></tr><tr><td> $1 / 4$ </td><td>0.994</td><td>0.730</td><td>0.283</td><td>0.592</td></tr><tr><td>Limit as  $\rho \downarrow 0$ </td><td>log 4</td><td>0.800</td><td>0</td><td>0.620</td></tr></table>

With equal retention, insensitive corrections make up more than half of supervision at the equilibrium, even though they come from half of the situations. Retaining them at one quarter of the sensitive rate reduces their share and raises the learned preference. As their relative retention tends to zero, the advisor approaches the sensitive teacher’s own preference of 0.8. The last row is a limiting case; the theorem itself still assumes strictly positive retention. For comparison, $J = 0 . 5$ at $p = 1 / 2$ and $J = 0 . 7 \mathrm { a t } p = 1$ . Selection improves the distillation equilibrium in this example, but a teacher of finite strength still does not lead the advisor to the success maximizer.

What changes when the teacher changes between updates. The preceding theorem fixes teacher targets over repeated learning. This is diferent from detaching a teacher for one update. Lemma 1 only requires the latter: its identity can be applied anew at each snapshot, even when the feedbackconditioned teacher changes between snapshots. A statement about the limiting preference, however, must also describe how those targets evolve. Two simple choices show why the distinction matters.

Keep the other assumptions of Appendix C.1 and let the detached target for type � at snapshot �<sup>¯</sup> have log-odds

$$
\ell _ { j } ( \bar { \theta } ) = \kappa \bar { \theta } + ( 1 - \kappa ) \ell _ { j } , \qquad 0 \leq \kappa < 1 ,
$$

with the same fixed � for both types and fixed $\ell _ { I } < \ell _ { S }$ . Each teacher now moves partway with the student while retaining a pull toward its original target. Because the target is detached during diferentiation, Eq. (12) uses $\bar { \theta } - \ell _ { j } ( \bar { \theta } ) = ( 1 - \kappa ) ( \bar { \theta } - \ell _ { j } )$ . Averaging over the unchanged proposal and retention process gives

$$
g _ { \kappa } ( \bar { \theta } ) = ( 1 - \kappa ) B ( \bar { \theta } ) p ( 1 - p ) [ \bar { \theta } - \mu ( \bar { \theta } ) ] = ( 1 - \kappa ) g ( \bar { \theta } ) .
$$

The distillation-only flow consequently has the same equilibrium and the same trajectories after a constant rescaling of time. When reward learning is also present, this rescaling applies only to the distillation term: its efective weight becomes $\lambda ( 1 - \kappa )$ . The combined equilibrium can therefore change, although Theorem 2 still compares selectors at this common positive efective weight.

Now consider detached targets that instead stay a fixed, nonnegative distance ahead of the student’s log-odds,

$$
\ell _ { j } ( \bar { \theta } ) = \bar { \theta } + \delta _ { j } , \qquad \delta _ { I } = 0 < \delta _ { S } .
$$

The insensitive teacher then gives zero distillation gradient, while the sensitive teacher contributes $- p ( 1 - p ) \delta _ { S }$ . The resulting distillation flow is

$$
\dot { \theta } = B ( \theta ) p ( 1 - p ) \big [ 1 - \omega _ { I } ( \theta ) \big ] \delta _ { S } > 0
$$

at every finite �. Its field is smooth and bounded, so the solution exists for all time and increases. It cannot have a finite limit, because the field is strictly positive at any such limit. Hence $\theta \to \infty$ and $p  1$ ; the flow does not settle at a finite equilibrium as in the fixed-teacher model.

We do not assume that either example describes how AdviSD’s teacher actually evolves. Together, they show what detachment provides and what it does not: it justifies the conditional gradient calculation, but it does not by itself preserve a learning-limit theorem. The stationary random-teacher extension in Appendix C.3 allows variation in feedback while making the needed stability of its conditional law explicit.

## C.2. Proof of Theorem 2 and Corollary 1 in the two-teacher model

Adding reward learning changes the comparison between selectors because the amount of distillation now matters as well as its mean target. We prove that preferential retention still gives a higher limiting success rate than retaining every proposal, and then consider uniform thinning, which changes exposure without changing the retained mixture. Throughout, we use the two-teacher model of Appendix C.1, with common fixed weights $c _ { 0 } \geq 0$ and $\lambda > 0$ . Both selectors start from the same finite logit. The reward term is exact ascent on expected success, as in Section 6.

Write $p \ = \ p ( \theta ) , f _ { I } = 1 - s _ { I } , f _ { 1 } = 1 - s _ { 1 }$ , and $f _ { 2 } = 1 - s _ { 2 }$ . The sensitive failure probability is $F ( p ) = p f _ { 1 } + ( 1 - p ) f _ { 2 }$ , and the overall failure probability is $f = \beta f _ { I } + ( 1 - \beta ) F ( p )$ . For a selector

$\boldsymbol { S } = ( r _ { I } ^ { S } , r _ { S } ^ { S } )$ , define

$$
\begin{array} { c c } { { A _ { S } = \beta f _ { I } r _ { I } ^ { S } , } } & { { V _ { S } ( p ) = ( 1 - \beta ) { \cal F } ( p ) r _ { S } ^ { S } , } } \\ { { \omega _ { S } ( \theta ) = { \displaystyle \frac { A _ { S } } { A _ { S } + V _ { S } ( p ) } } , } } & { { \mu _ { S } ( \theta ) = \ell _ { S } - ( \ell _ { S } - \ell _ { I } ) \omega _ { S } ( \theta ) . } } \end{array}
$$

Thus $\mu _ { S }$ is the mean retained teacher log-odds. A proposed failure is retained with probability ${ u _ { S } } = ( { A _ { S } } + { V _ { S } } ( p ) ) / f$ . Since an episode contains � independent opportunities and proposes min(�, �) of its � failures, its probability of receiving any distillation is

$$
B _ { S } ( \theta ) = \sum _ { n = 1 } ^ { M } { \binom { M } { n } } f ^ { n } ( 1 - f ) ^ { M - n } \big [ 1 - ( 1 - u _ { S } ) ^ { \operatorname* { m i n } ( n , b ) } \big ] .
$$

The expected sampled-loss gradient from Theorem 1 is $g _ { S } = B _ { S } p ( 1 - p ) ( \theta - \mu _ { S } )$ . Using $J ^ { \prime } ( \theta ) =$ $( 1 - \beta ) ( s _ { 1 } - s _ { 2 } ) p ( 1 - p )$ , we can therefore write the combined update field as

$$
F _ { S } ( \theta ) = p ( 1 - p ) \big \{ C - \lambda B _ { S } ( \theta ) [ \theta - \mu _ { S } ( \theta ) ] \big \} , \qquad C = c _ { 0 } ( 1 - \beta ) ( s _ { 1 } - s _ { 2 } ) \geq 0 .\tag{16}
$$

Here $S = 0$ means $r _ { I } ^ { 0 } = r _ { S } ^ { 0 } = 1$ , whereas the preferential selector � has fixed independent retention rates in $( 0 , 1 ]$ with $r _ { I } ^ { G } / r _ { S } ^ { G } < 1$

Proof of Theorem 2. We first compare the two selectors at a common parameter �. Dividing the numerator and denominator of $\omega _ { S }$ by $r _ { s } ^ { s }$ shows that it depends on the retention rates only through their ratio $\rho _ { S } = r _ { I } ^ { S } / r _ { S } ^ { S }$

$$
\omega _ { S } ( \theta ) = \frac { \rho _ { S } \beta f _ { I } } { \rho _ { S } \beta f _ { I } + ( 1 - \beta ) F ( p ) } .
$$

This expression is strictly increasing in $\rho _ { S }$ . Since $\rho _ { G } < \rho _ { 0 } = 1$ and $\ell _ { I } < \ell _ { S }$ , preferential retention gives $\mu _ { G } ( \theta ) > \mu _ { 0 } ( \theta )$ . It also $\mathrm { g i v e s } u _ { G } \leq u _ { 0 }$ , because neither type is retained more often than under no gating. Each term $1 - ( 1 - u _ { S } ) ^ { \mathrm { m i n } ( n , b ) }$ is increasing in �<sub>�</sub>, so $B _ { G } \leq B _ { 0 }$ . In particular, for every finite �,

$$
\mu _ { G } ( \theta ) > \mu _ { 0 } ( \theta ) , \qquad 0 < B _ { G } ( \theta ) \leq B _ { 0 } ( \theta ) .\tag{17}
$$

For later use, the exposure probability is bounded away from zero uniformly in �. Indeed, whenever $N \geq 1 .$ , the positive cap ensures that at least one failure is proposed. Retaining at least one of these proposals has probability at least $u _ { S }$ . As $M \geq 1 , \operatorname* { P r } ( N \geq 1 ) = 1 - ( 1 - f ) ^ { M } \geq f .$ , and consequently

$$
B _ { S } ( \theta ) \geq u _ { S } \operatorname* { P r } ( N \geq 1 ) \geq f u _ { S } = A _ { S } + V _ { S } ( p ) \geq A _ { S } > 0 .\tag{18}
$$

This lower bound follows from the positive probability of an insensitive failure and its fixed positive retention rate.

Subtracting the two instances of Eq. (16) gives

$$
\begin{array} { r l } & { F _ { G } - F _ { 0 } = \lambda p ( 1 - p ) \left[ B _ { 0 } ( \theta - \mu _ { 0 } ) - B _ { G } ( \theta - \mu _ { G } ) \right] } \\ & { \qquad = \lambda p ( 1 - p ) \left[ B _ { G } ( \mu _ { G } - \mu _ { 0 } ) + ( B _ { 0 } - B _ { G } ) ( \theta - \mu _ { 0 } ) \right] . } \end{array}
$$

This is the composition–exposure identity in Eq. (7). Its composition term is strictly positive, while the sign of its exposure term depends on whether the current preference lies above or below the ungated teaching target. We will use this sign change to distinguish the limiting comparison from a comparison at every finite time.

Before comparing the trajectories, we establish that each one has a finite limit. The field $F _ { S }$ is smooth because all denominators in the expressions above are positive. It is also bounded on ℝ:

$B _ { S } \leq 1 , \mu _ { S } \in ( \ell _ { I } , \ell _ { S } )$ , and both $p ( 1 - p )$ and $| \theta | p ( 1 - p )$ are bounded. The positive lower bound in Eq. (18) further gives the restoring signs

$$
\begin{array} { l l } { { F _ { S } ( \theta ) > 0 } } & { { \mathrm { i f } \theta < \ell _ { I } , } } \\ { { } } & { { } } \\ { { F _ { S } ( \theta ) < 0 } } & { { \mathrm { i f } \theta > \ell _ { S } + \displaystyle \frac { C } { \lambda A _ { S } } . } } \end{array}\tag{19}
$$

For the first inequality, $\theta - \mu _ { S } < 0$ makes the bracket in Eq. (16) positive. For the second, $B _ { S } ( \theta - \mu _ { S } ) \geq$ $A _ { S } ( \theta - \ell _ { S } ) > C / \lambda$ makes it negative. Thus we can enclose any finite initial value in a compact interval on whose endpoints the field points inward. Smoothness gives a unique solution, and the inward signs keep it in that interval for all time. A nonconstant scalar autonomous trajectory cannot cross a zero of its field: uniqueness would otherwise be violated by the constant solution at that zero. Its direction therefore cannot reverse, so it is monotone and bounded. It has a finite limit, and continuity forces that limit to be a zero of $F _ { S }$ . A trajectory started at a zero is constant and has the same conclusion. This argument does not require the combined field to have a unique zero.

Let $a _ { 0 }$ be the ungated distillation-only equilibrium. By Theorem 1, $\theta - \mu _ { 0 } ( \theta )$ is strictly increasing and vanishes at $a _ { 0 }$ . If $\theta < a _ { 0 } .$ , then $\theta < \mu _ { 0 } ( \theta ) < \mu _ { G } ( \theta )$ , so both combined fields are positive. If $\theta \geq a _ { 0 } .$ the exposure term in Eq. (7) is nonnegative and its composition term is positive. We have therefore established

$$
\begin{array} { r l r } & { F _ { 0 } ( \theta ) , F _ { G } ( \theta ) > 0 } & { ( \theta < a _ { 0 } ) , } \\ & { F _ { G } ( \theta ) > F _ { 0 } ( \theta ) } & { ( \theta \geq a _ { 0 } ) , } \\ & { F _ { 0 } ( a _ { 0 } ) = p ( a _ { 0 } ) [ 1 - p ( a _ { 0 } ) ] C \geq 0 . } \end{array}\tag{20}
$$

In particular, $F _ { G } ( a _ { 0 } ) > 0$ . The half-line $[ a _ { 0 } , \infty )$ is forward invariant for both flows: the gated field points inward at its boundary, and the ungated field either points inward or has an equilibrium there.

Now suppose the common initial value � satisfies $x \geq a _ { 0 }$ . Set $D ( t ) = \theta _ { G } ( t ) - \theta _ { 0 } ( t )$ . Initially $D ( 0 ) = 0$ and $D ^ { \prime } ( 0 ) = F _ { G } ( x ) - F _ { 0 } ( x ) > 0$ , so the gated trajectory is strictly larger for all suficiently small positive times. If the trajectories first met again at a time $t _ { 1 } > 0 ,$ , then $D ^ { \prime } ( t _ { 1 } ) \leq 0$ . At that meeting point, however, both parameters belong to $[ a _ { 0 } , \infty )$ and Eq. (20) gives $D ^ { \prime } ( t _ { 1 } ) = F _ { G } ( \theta _ { G } ( t _ { 1 } ) ) - F _ { 0 } ( \theta _ { G } ( t _ { 1 } ) ) > 0$ a contradiction. Hence $\theta _ { G } ( t ) > \theta _ { 0 } ( t )$ for every $t > 0$ . Their finite limits are at least weakly ordered. Equality of these limits is impossible: a common limit $z \geq a _ { 0 }$ would be a zero of both fields, whereas $F _ { G } ( z ) > F _ { 0 } ( z )$ . This proves strict limiting order as well as the claimed finite-time comparison for starts at or above $a _ { 0 }$

It remains to compare limits when $x < a _ { 0 }$ , where exposure can prevent such a finite-time ordering. If $c _ { 0 } = 0 .$ , the ungated flow converges to $a _ { 0 }$ by Theorem 1. The gated field is positive throughout $( - \infty , a _ { 0 } ]$ , so its finite limiting equilibrium must lie strictly above $a _ { 0 }$ . If $c _ { 0 } > 0 _ { : }$ , then both fields are positive on $( - \infty , a _ { 0 } ]$ , including $a _ { 0 }$ itself. Let $z _ { 0 }$ be the first zero of $F _ { 0 }$ above �. Such a zero exists because the field is positive at � and negative suficiently far to the right, and $z _ { 0 } > a _ { 0 }$ because both fields are positive through $a _ { 0 }$ . The ungated trajectory increases to $\scriptstyle z _ { 0 }$ . On $( a _ { 0 } , z _ { 0 } )$ , its field is positive, so $F _ { G } > F _ { 0 } > 0$ there; at ${ z } _ { 0 }$ , the strict field comparison gives $F _ { G } ( z _ { 0 } ) > 0$ . Together with positivity below $a _ { 0 }$ , this shows that $F _ { G }$ has no zero anywhere from � through $\scriptstyle z _ { 0 }$ . Its finite limiting equilibrium must therefore lie strictly above $\scriptstyle { \mathcal { Z } } 0$ . In both cases $\theta _ { G } ^ { \infty } > \theta _ { 0 } ^ { \infty }$ . Finally, $J ^ { \prime } ( \theta ) > 0$ for every finite $\theta ,$ giving $J ( \theta _ { G } ^ { \infty } ) > J ( \theta _ { 0 } ^ { \infty } )$ as claimed. □

The same identity explains why a gain over no gating need not come from a better correction mixture. Retaining every proposal with the same probability leaves the conditional mean target unchanged, but reduces how often that target is applied. This gives the following comparison under the same common reward objective and initialization.

Corollary 1 (Uniform thinning changes exposure, not composition). If every proposal is retained independently with the same fixed probability $r \in ( 0 , 1 ]$ , the limiting preference is at least that of no gating under the common reward objective. Without reward learning $( c _ { 0 } = 0 )$ <sub>,</sub> th<sub>e</sub> tw<sub>o</sub> limit<sub>s co</sub>in<sub>c</sub>id<sub>e</sub>.

Proof. Taking $r _ { I } ^ { G } = r _ { S } ^ { G } = r$ leaves the retention ratio equal to one, so $\mu _ { G } \equiv \mu _ { 0 }$ , while $0 < B _ { G } \leq B _ { 0 }$ Moreover, Eq. (18) gives $B _ { G } \geq r f \geq r \beta ( 1 - s _ { I } ) > 0$ . Thus both fields have the restoring signs and finite limiting equilibria established in the preceding proof. Their diference now reduces to

$$
F _ { G } ( \theta ) - F _ { 0 } ( \theta ) = \lambda p ( 1 - p ) ( B _ { 0 } - B _ { G } ) [ \theta - \mu _ { 0 } ( \theta ) ] .\tag{21}
$$

When $c _ { 0 } = 0$ , the two fields have the same sign as $\mu _ { 0 } ( \theta ) - \theta$ and vanish only at $a _ { 0 }$ . Both flows therefore converge to $a _ { 0 }$ from every common finite initialization.

Suppose instead that $c _ { 0 } > 0$ . On $[ a _ { 0 } , \infty ) , { \mathrm { E q . ~ } } ( 2 1 )$ gives $F _ { G } \geq F _ { 0 }$ , and $F _ { G } ( a _ { 0 } ) = F _ { 0 } ( a _ { 0 } ) = p ( a _ { 0 } ) [ 1 -$ $p ( a _ { 0 } ) ] C > 0$ . This half-line is forward invariant. The scalar comparison principle for locally Lipschitz fields then gives $\theta _ { G } ( t ) \geq \theta _ { 0 } ( t )$ from a common initial value in this half-line, hence the same weak ordering of their limits.

For a common initial value below $^ { a _ { 0 } , }$ both fields are positive up to and including $a _ { 0 }$ . As in the preceding proof, the ungated trajectory increases to its first equilibrium $z _ { 0 } > a _ { 0 }$ . On $( a _ { 0 } , z _ { 0 } )$ we have $F _ { G } \geq F _ { 0 } > 0 .$ , so the gated field has no zero between the common start and $z _ { 0 } , \mathrm { A t } z _ { 0 }$ it satisfies only $F _ { G } ( z _ { 0 } ) \geq 0$ , which allows its limiting equilibrium to equal $z _ { 0 }$ . In all cases its limit is at least $\scriptstyle z _ { 0 }$ , proving the weak comparison. The proof does not require strict inequality at ${ \mathfrak { z } } _ { 0 } ;$ in particular, $r = 1$ gives identical flows. □

Appendix C.3 extends these arguments to stationary random teachers, including conditions for comparisons when selection changes the retained targets. The independent thinning in Corollary 1 difers from the implemented matched-count random control, whose gate-derived per-episode quotas can change composition (Appendix E).

For a concrete illustration, use the instance from Appendix C.1 with $\beta = s _ { I } = 1 / 2 , s _ { 1 } = 0 . 9 , s _ { 2 } = 0 . 1$ ， $Q _ { I } = ( 0 . 5 , 0 . 5 )$ , and $Q _ { S } = ( 0 . 8 , 0 . 2 )$ , so $a _ { 0 } = 0 . 6 0 1$ and $p ( a _ { 0 } ) = 0 . 6 4 6$ . Set $M = 4 , b = 2 , c _ { 0 } = 1 / 2$ $\lambda = 1$ , and take the quarter-rate gate $G = ( 1 / 4 , 1 )$ . To separate composition from exposure in this example, introduce the hypothetical field

$$
F _ { R } ( \theta ) = c _ { 0 } J ^ { \prime } ( \theta ) - \lambda B _ { G } ( \theta ) p ( 1 - p ) [ \theta - \mu _ { 0 } ( \theta ) ] .
$$

By construction, it combines the ungated composition with the gated exposure; it is not the sameepisode matched-count control. From $\theta ( 0 ) = 0$ , the three fields converge to

<table><tr><td>Dynamics</td><td> $\theta ^ { \infty }$ </td><td> $p ^ { \infty }$ </td><td> ${ \cal J } ( \theta ^ { \infty } )$ </td></tr><tr><td>No gating  $\left( F _ { 0 } \right)$ </td><td>0.794</td><td>0.689</td><td>0.576</td></tr><tr><td>Ungated composition, gated exposure  $\left( F _ { R } \right)$ </td><td>0.876</td><td>0.706</td><td>0.582</td></tr><tr><td>Preferential selection  $\left( F _ { G } \right)$ </td><td>1.303</td><td>0.786</td><td>0.615</td></tr></table>

At the ungated limit, the composition and exposure terms on the right-hand side of Eq. (7) are 0.287 and 0.057, respectively. Thus changing exposure alone accounts for part of the limiting gain in this instance, while changing the retained target raises the limit further.

The initial efect can go the other way. Keep the same success laws, teachers, weights, and gate, but take one opportunity per episode $( M = b = 1 )$ . At a common start of $\theta = - 2 , F _ { G } ( - 2 ) = 0 . 1 7 7 <$ $F _ { 0 } ( - 2 ) = 0 . 2 1 7$ , so the gated advisor starts more slowly, although the limiting advice-1 probabilities remain ordered: $\boldsymbol { p } ^ { \infty } = 0 . 7 3 6 , 0 . 8 2 4$ , and 0.892 for $F _ { 0 } , F _ { R } ,$ and $F _ { G } ,$ respectively. Below $a _ { 0 } .$ , removing corrections can weaken an upward teaching pull; the comparison of limits does not say that every earlier update also improves.

The finite balance studied here depends on the common positive distillation weight. With $c _ { 0 } > 0$ and no distillation, reward-only ascent has $\dot { \theta } = c _ { 0 } J ^ { \prime } ( \theta ) > 0$ at every finite logit and approaches $p = 1$ the best success rate in this model. The theorem therefore compares preferential selection with ungated distillation. AdviSD’s gains over outcome-only GRPO under finite training schedules are established by the experiments.

## C.3. Stationary random teachers and action-dependent selection

The two-teacher model isolates how the mixture of retained corrections afects learning by assigning a fixed target to each situation type. In a feedback-based system, however, the target may vary even within one type: the feedback can depend on the advice that was issued, and the selector can preferentially retain some of the resulting targets. This subsection allows both forms of variation while keeping the two-action policy and the opportunity and execution laws fixed. We first identify the teacher statistic that determines the expected distillation update, then establish conditions for a unique learning limit and for comparing selectors with a common reward objective.

The relevant stationarity requirement concerns the joint law of the teacher and its retention, conditional on the type, action, and failure. It does not require retention to be independent of teacher content. As the student changes its action probabilities, the mixture of these conditional laws can still change, and this is why the target below depends on the parameter.

Assumption 2 (Stationary random feedback and selection). Keep the opportunity and execution laws, independent sampling, uniform capped failure proposals, and episode-averaged loss ofAppendix C.1. C<sub>o</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>o</sub>n t<sub>ype</sub> $j \in \{ I , S \}$ <sub>,</sub> i<sub>ss</sub>u<sub>e</sub>d <sub>ac</sub>ti<sub>o</sub>n $a \in \{ 1 , 2 \}$ , and failure, the teacher � and retention indicator have a joint law independent of �; these pairs are independent across opportunities, and the uniform proposal sample is independent of them. Teachers are positive on both actions, retention has probability $r _ { j a } \in ( 0 , 1 ]$ , and the retained teacher log-odds have a finite absolute mean. Teachers and selections are detached when diferentiating.

For each type and issued action, define the mean teacher log-odds among the corrections that survive selection:

$$
\ell _ { j a } = \mathbb { E } \left[ \log \frac { Q ( 1 ) } { Q ( 2 ) } \middle | j , a , \mathrm { f a i l u r e , r e t a i n e d } \right] .
$$

Conditioning on retention matters when the selector depends on the teacher: it records the target actually supplied to the loss, which can difer from the mean target before selection. Note also that $\ell _ { j a }$ averages log-odds rather than taking the log-odds of the mean teacher probabilities, because log-odds enter the reverse-KL gradient linearly in Eq. (12).

Using $f _ { I } = 1 - s _ { I }$ and $f _ { a } = 1 - s _ { a }$ from Appendix C.1, define

$$
\begin{array} { r l } & { i _ { a } = \beta f _ { I } r _ { I a } , \qquad \nu _ { a } = ( 1 - \beta ) f _ { a } r _ { S a } , \qquad C _ { a } = i _ { a } + \nu _ { a } , } \\ & { \bar { \ell } _ { a } = \frac { i _ { a } \ell _ { I a } + \nu _ { a } \ell _ { S a } } { C _ { a } } , \qquad \mu ( \theta ) = \frac { p C _ { 1 } \bar { \ell } _ { 1 } + ( 1 - p ) C _ { 2 } \bar { \ell } _ { 2 } } { p C _ { 1 } + ( 1 - p ) C _ { 2 } } , \quad p = \mathrm { s i g m o i d } ( \theta ) . } \end{array}\tag{22}
$$

Conditional on issuing action $^ { a , }$ the quantities $i _ { a }$ and $\nu _ { a }$ are the probabilities of failures of each type whose corrections would be retained if proposed. Their sum $C _ { a }$ is positive, and $\textstyle { \bar { \ell } } _ { a }$ is the corresponding retained mean target. Averaging across the two actions uses the weights $p C _ { 1 }$ and $( 1 - p ) C _ { 2 }$ instead of the action probabilities $p$ and $1 - p ,$ because the actions can produce retained failures at diferent rates. This gives $\mu ( \theta )$ , the mean log-odds of a retained proposal at the current policy. Let $B ( \theta )$ again denote the probability of retaining at least one correction in an episode.

Theorem 3 (Stationary random-teacher extension). Under Assumption $^ { 2 , }$ th<sub>e expec</sub>t<sub>e</sub>d <sub>samp</sub>l<sub>e</sub>d<sub>-</sub>l<sub>oss</sub> <sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt i<sub>s</sub> $g = B p ( 1 - p ) ( \theta - \mu ) .$ <sub>,</sub> <sub>w</sub>ith $B > 0$ . If $\bar { \ell } _ { 1 } - \bar { \ell } _ { 2 } \leq 4 ,$ , its gradient flow converges from every finite initialization to the unique $\theta ^ { * } = \mu ( \theta ^ { * } )$ <sub>.</sub> Th<sub>e roo</sub>t li<sub>es s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> b<sub>e</sub>t<sub>ween unequa</sub>l $\bar { \ell } _ { 1 } , \bar { \ell } _ { 2 } ,$ , or e<sub>q</sub>ua<sup>l</sup>s their common value, and is independent of episode size and cap. For classwise retention $r _ { j a } = r _ { j } ,$ suppose <sub>a</sub>dditi<sub>ona</sub>ll<sub>y</sub> th<sub>a</sub>t $\ell _ { j 2 } \geq \ell _ { j 1 }$ for each type and max $\ell _ { I a } < \operatorname* { m i n } _ { a } \ell _ { S a }$ . Holding these four retained means fixed, decreasing $r _ { I } / r _ { S }$ <sub>s</sub>tri<sub>c</sub>tl<sub>y</sub> in<sub>c</sub>r<sub>eases</sub> $\theta ^ { * }$ <sub>an</sub>d it<sub>s expec</sub>t<sub>e</sub>d <sub>success.</sub>

Proof. We begin by taking the expectation of the realized episode gradient. For any retained teacher, Eq. (12) gives the contribution $p ( 1 - p ) ( \theta - \log \mathrm { i t } Q ( 1 ) )$ . Write

$$
f = \beta f _ { I } + ( 1 - \beta ) [ p f _ { 1 } + ( 1 - p ) f _ { 2 } ] , \qquad t = p C _ { 1 } + ( 1 - p ) C _ { 2 } , \qquad u = t / f .
$$

Here � is the probability that an opportunity fails, � is the probability of a failure whose correction would be retained, and � is the retention probability conditional on failure. The conditional mean log-odds of a retained correction is $\mu$ by Eq. (22).

Conditional on the number of failures, uniform proposal sampling does not favor any type, action, or teacher–retention pair. The proposed pairs therefore have the same independent conditional law as failure opportunities. Conditioning further on the retained indices leaves their teachers distributed according to the retained law. In particular, for every positive realized retained count $K ,$ the expected average of their log-odds is $\mu .$ . The episode loss averages the � realized contributions and is zero when $K = 0$ . Consequently, exactly as in the derivation of Eq. (2),

$$
g = { \mathrm { P r } } ( K \geq 1 ) p ( 1 - p ) ( \theta - \mu ) = B p ( 1 - p ) ( \theta - \mu ) .
$$

The exposure probability � is the capped binomial expression in Eq. (14), with the $f ,$ � defined here. Its positivity follows from positive failure and retention probabilities. Finite absolute log moments make each of these expectations finite. This calculation averages the loss with its realized denominator $K ;$ it does not replace that denominator by an expected count.

To analyze the limit, we must account for the dependence of $\mu$ on the policy. Unlike in the two-teacher model, increasing the probability of action 1 can increase the retained mean target. Let $D = \overline { { \ell } } _ { 1 } - \overline { { \ell } } _ { 2 }$ and $\omega = \mathrm { s i g m o i d } ( \theta + \log ( C _ { 1 } / C _ { 2 } ) )$ . Since $p / ( 1 - p ) = e ^ { \theta }$ , the retained share of action 1 is

$$
\frac { p C _ { 1 } } { p C _ { 1 } + ( 1 - p ) C _ { 2 } } = \omega .
$$

Thus the target and the derivative of the fixed-point residual are

$$
\mu = \bar { \ell } _ { 2 } + D \omega , \qquad \frac { d } { d \theta } ( \theta - \mu ) = 1 - D \omega ( 1 - \omega ) .
$$

The bound $\omega ( 1 - \omega ) \leq 1 / 4$ makes this derivative strictly positive whenever $D < 4$ . At the boundary $D = 4$ , the derivative is nonnegative and can vanish only where $\omega = 1 / 2$ , which occurs at a single finite parameter. Integrating the derivative over any nontrivial interval therefore still gives a positive diference. Hence $\theta - \mu ( \theta )$ is strictly increasing also at $D = 4$

Because $\mu$ is a convex combination of the two finite constants $\bar { \ell } _ { 1 } , \bar { \ell } _ { 2 } .$ , it is bounded. The residual consequently tends to −∞ and +∞ at the respective ends of the real line, so it has exactly one zero $\theta ^ { * }$

Both action weights are strictly positive at any finite parameter. Evaluating the convex combination at $\theta ^ { * } = \mu ( \theta ^ { * } )$ places this root strictly between unequal targets, or at their common value if they coincide.

The descent flow is $\dot { \theta } = - g$ . Its field is smooth and bounded: $0 < B \leq 1 _ { \cdot }$ , � is bounded, and $p ( 1 - p ) | \theta |$ is bounded. Furthermore, it is positive below $\theta ^ { * }$ and negative above $\theta ^ { * }$ . A trajectory from a finite initialization therefore stays between its initial value and $\theta ^ { * }$ and cannot cross the equilibrium, by uniqueness of solutions. It is monotone unless already stationary, and its finite limit must be a zero of the field by continuity. The only such zero is $\theta ^ { * }$ , proving convergence. Neither the episode size � nor the cap � enters the equation $\theta ^ { * } = \mu ( \theta ^ { * } )$ ; they afect the positive factor � and hence the speed of this flow, but not its limit.

For the retention comparison, assume $r _ { j a } = r _ { j }$ . Conditional on either issued action, the insensitive retained weight is then $\beta f _ { I } r _ { I }$ , whereas the sensitive weight is $( 1 - \beta ) f _ { a } r _ { S }$ . Since $f _ { 1 } < f _ { 2 }$ , action 1 has a larger insensitive share among its retained failures. The assumed ordering $\ell _ { j 1 } \leq \ell _ { j 2 }$ means that replacing each action-1 target by its action-2 counterpart cannot lower the mixture. After that replacement, reducing the insensitive share cannot lower it either, because every insensitive target is strictly below every sensitive target. These two comparisons give $\bar { \ell } _ { 1 } \leq \bar { \ell } _ { 2 }$ . In particular, the fixed-point residual has derivative $1 - \partial _ { \theta } \mu \geq 1$ , so the root is unique and varies smoothly with the retention ratio.

To determine the direction of that change, put $\rho = r _ { I } / r _ { S } , F = p f _ { 1 } + ( 1 - p ) f _ { 2 } , L _ { I } = p \ell _ { I 1 } + ( 1 - p ) \ell _ { I 2 } $ and $L _ { S } = \left[ p f _ { 1 } \ell _ { S 1 } + ( 1 - p ) f _ { 2 } \ell _ { S 2 } \right] / F$ . Dividing the numerator and denominator in Eq. (22) by $r _ { S }$ gives

$$
\mu = \frac { \rho \beta f _ { I } L _ { I } + ( 1 - \beta ) F L _ { S } } { \rho \beta f _ { I } + ( 1 - \beta ) F } .
$$

When � and the four retained means are fixed, $L _ { I } , L _ { S } , F$ are fixed as well. Diferentiating this weighted average with respect to $\rho$ yields

$$
\partial _ { \rho } \mu = \frac { \beta f _ { I } ( 1 - \beta ) F ( L _ { I } - L _ { S } ) } { [ \rho \beta f _ { I } + ( 1 - \beta ) F ] ^ { 2 } } < 0 .
$$

The inequality follows because $L _ { I }$ and $L _ { S }$ are convex averages within their respective types and the types are strictly separated. Implicitly diferentiating $\theta ^ { * } - \mu ( \theta ^ { * } , \rho ) = 0$ now gives

$$
\frac { d \theta ^ { * } } { d \rho } = \frac { \partial _ { \rho } \mu } { 1 - \partial _ { \theta } \mu } < 0 .
$$

Finally, $J ^ { \prime } ( \theta ) = ( 1 - \beta ) ( s _ { 1 } - s _ { 2 } ) p ( 1 - p ) > 0 $ , so the higher limiting parameter also has strictly higher expected success. □

The fixed-point result and the retention comparison have diferent requirements. The first result allows random teachers and action-dependent selection, subject to the stated gap condition. The comparison additionally holds the four retained means fixed while varying $r _ { I } / r _ { S }$ . This restriction matters because a new selector can change teacher content within a type, and therefore change $\ell _ { j a }$ as well as $r _ { j a }$ . Retaining fewer insensitive corrections alone does not imply an improved target. In the original two-teacher model, the fixed-mean restriction holds by construction: $\ell _ { j a } = \ell _ { j { \mathrm { : } } }$ , and ${ \bar { \ell } } _ { 1 } \leq { \bar { \ell } } _ { 2 }$ makes the gap condition automatic.

Reward learning when selection also changes the targets. To compare selectors that can change the retained teacher laws, we formulate the required improvement directly in terms of their mean targets. This extends the reward-learning comparison of Appendix C.2: the proof needs an ordering of $\mu _ { G }$ and $\mu _ { 0 } .$ , together with an ordering of their supervision probabilities, rather than an ordering of classwise retention rates alone.

Assumption 3 (Comparable selective dynamics). Systems $0 , G$ satisfy Assumption 2 with common opportunity and execution laws, episode size, and proposal cap. System 0 keeps every proposal; � keeps a subset. Their retained teacher laws may difer. For every finite $\theta ,$ assume $\mu _ { G } ( \theta ) > \mu _ { 0 } ( \theta )$ <sub>,</sub> <sub>a</sub>nd r<sub>equ</sub>ir<sub>e</sub> $\bar { \ell } _ { 1 } ^ { 0 } - \bar { \ell } _ { 2 } ^ { 0 } \leq 4 .$

Theorem 4 (Selection with stationary random teachers). Under Assumption 3,fix $c _ { 0 } \geq 0 , \lambda > 0$ <sub>an</sub>d $F _ { j } = c _ { 0 } J ^ { \prime } - \lambda g _ { j }$ . From every common finite initialization, the flows $\dot { \theta } _ { j } = F _ { j } ( \theta _ { j } )$ converge to finite limits <sub>w</sub>ith $\theta _ { G } ^ { \infty } > \theta _ { 0 } ^ { \infty }$ <sub>an</sub>d $J ( \theta _ { G } ^ { \infty } ) > J ( \theta _ { 0 } ^ { \infty } )$ , even if combined equilibria are not unique. From a common start at <sub>or a</sub>b<sub>ove</sub> th<sub>e unga</sub>t<sub>e</sub>d di<sub>s</sub>till<sub>a</sub>ti<sub>on-on</sub>l<sub>y equ</sub>ilib<sub>r</sub>i<sub>um</sub> $a _ { 0 } ,$ <sub>, a</sub>l<sub>so</sub> $\theta _ { G } ( t ) > \theta _ { 0 } ( t ) f o r$ e<sup>v</sup>e<sup>r</sup>y $t > 0$

Proof. Subset retention gives $0 < B _ { G } \leq B _ { 0 } \colon$ on common proposals, an episode with a retained gated correction necessarily has a retained ungated correction. We also need a lower bound that holds uniformly in $\theta ,$ so that the distillation term continues to provide a restoring force at large parameter values. For either selector $j ,$ when an episode contains at least one failure, the proposal cap $b \geq 1$ leaves at least one opportunity for retention. Thus the capped binomial expression gives

$$
B _ { j } \geq u _ { j } \operatorname* { P r } ( N \geq 1 ) \geq u _ { j } f = p C _ { 1 } ^ { j } + ( 1 - p ) C _ { 2 } ^ { j } \geq \operatorname* { m i n } _ { a } C _ { a } ^ { j } > 0 .
$$

Here $\Pr ( N \geq 1 ) = 1 - ( 1 - f ) ^ { M } \geq f$ because $M \geq 1$ . The constants $C _ { a } ^ { j }$ are positive by the failure and retention assumptions.

The means $\mu _ { j }$ are bounded, and each field has the form Eq. (16), with $C = c _ { 0 } ( 1 - \beta ) ( s _ { 1 } - s _ { 2 } )$ . The bracket in that equation is positive for suficiently negative � and, by the uniform lower bound on $B _ { j { \mathrm { : } } }$ , negative for suficiently positive �. The fields are smooth and bounded. Each flow therefore has a unique global solution confined to a compact interval containing its start. A scalar autonomous trajectory cannot cross an equilibrium; it is otherwise monotone, and continuity forces its finite limit to be an equilibrium. This is the same convergence argument used for the two-teacher fields, and it does not require their equilibria to be unique.

The baseline gap condition and Theorem 3 imply that $\theta - \mu _ { 0 } ( \theta )$ is strictly increasing with unique zero $a _ { 0 }$ . Below this point, $\theta < \mu _ { 0 } < \mu _ { G }$ , so both fields are positive. At and above $a _ { 0 }$ , the composition– exposure identity Eq. (7) has a strictly positive composition term and a nonnegative exposure term, giving $F _ { G } > F _ { 0 }$ . In particular, $F _ { 0 } ( a _ { 0 } ) = p ( 1 - p ) C \geq 0$ and $F _ { G } ( a _ { 0 } ) > F _ { 0 } ( a _ { 0 } )$ . These are precisely the sign properties that supported the two-teacher comparison, now obtained from assumptions on the retained mean targets.

Suppose first that the common initialization is at or above $a _ { 0 }$ . The half-line $[ a _ { 0 } , \infty )$ is forward invariant for both flows. At a common point in this region, the gated field is strictly larger, so the trajectory diference has positive derivative at time zero. If the diference subsequently returned to zero for the first time, its derivative there would be nonpositive. At that common parameter, however, its derivative is $F _ { G } - F _ { 0 } > 0 .$ , a contradiction. This proves strict ordering at every positive time and weak ordering of the limits. The limits cannot be equal, because a common limit would be a zero of both fields in a region where $F _ { G } > F _ { 0 }$

For a common initialization below $a _ { 0 . }$ , the comparison concerns the limits rather than necessarily the early trajectories. If $c _ { 0 } = 0$ , the ungated flow converges to $a _ { 0 }$ , while the gated field is strictly positive everywhere up to and including $a _ { 0 }$ . Its finite equilibrium limit must therefore lie above �<sub>0</sub>. If $c _ { 0 } > 0$ , both fields are positive through $a _ { 0 }$ . Let $z _ { 0 } \mathrm { ~ } > \mathrm { ~ } a _ { 0 }$ be the first ungated equilibrium above the initialization, which is the ungated limit. On $( a _ { 0 } , z _ { 0 } )$ we have $F _ { G } > F _ { 0 } > 0$ , and at $\scriptstyle z _ { 0 }$ we have $F _ { G } ( z _ { 0 } ) > F _ { 0 } ( z _ { 0 } ) = 0$ . Together with positivity below $a _ { 0 } ,$ this shows that the gated field is positive from the initialization through ${ \mathit { z } } _ { 0 } ,$ so its limit is strictly larger. In both cases the parameter limits are strictly ordered. Since � is strictly increasing, their expected success values are strictly ordered as well. □

Equal composition. The exposure corollary also extends to stationary random teachers. Under the same fixed common weights $c _ { 0 } \geq 0 , \lambda > 0 .$ , consider two selectors satisfying Assumption 2 with common laws, episode size, and cap, and suppose $\mu _ { G } \equiv \mu _ { 0 } , 0 < B _ { G } \leq B _ { 0 }$ , and $\bar { \ell } _ { 1 } ^ { 0 } - \bar { \ell } _ { 2 } ^ { 0 } \leq 4$ . The uniform lower bound on exposure proved above again supplies finite equilibrium limits. Without reward learning, both fields have the same sign pattern and the same unique zero $a _ { 0 }$ , so both flows converge to $a _ { 0 }$ from every finite start.

With reward learning, equality of the mean targets removes the composition term from Eq. (7). Hence $F _ { G } \geq F _ { 0 }$ on $[ a _ { 0 } , \infty )$ , a forward-invariant region for both flows. The scalar comparison argument in Corollary 1 then gives weakly ordered trajectories and limits from a common start in this region. From a start below $a _ { 0 } ,$ both fields are positive through $a _ { 0 }$ and thereafter on $( a _ { 0 } , z _ { 0 } )$ , where $\scriptstyle z _ { 0 }$ is the first ungated equilibrium. The gated field can vanish at ${ z _ { 0 } } .$ , but cannot have an equilibrium below it along this path. Its limit is therefore at least the ungated limit. Identical exposure gives identical fields and, from the same initialization, identical flows. As in the two-teacher case, the implemented gate-derived quota control need not preserve composition (Appendix E), so this equal-composition comparison does not describe that control.

These results extend the retained-mixture mechanism beyond one fixed teacher per class. The expected update depends on the distribution of targets after retention, and the reward comparison assumes that this retained mean is higher under the selector. AdviSD’s predictive score does not guarantee $\mu _ { G } > \mu _ { 0 }$ . The conclusions concern the limits of stationary expected dynamics; they do not establish finite-run robustness to teacher noise or convergence for AdviSD with its evolving self-teacher and GRPO–AdamW updates.

## D. Training and implementation details

AdviSD trains Qwen3-8B (Yang et al., 2025) while keeping the executor fixed. The two executor settings use Claude Sonnet 4.6 or Gemini 3.7 Flash; both use Gemini 3.7 Flash for reflection. Table 4 specifies the BFCL training configuration. EnvScaler uses its own training partition and validation split, described below. Each EnvScaler run makes one pass over the 1,880 training tasks: eight tasks per update and eight rollouts per task give 235 updates with 64 episodes per update. EnvScaler uses the same auxiliary-weight schedule as BFCL (Table 4), so $\lambda _ { s } = 0 . 0 5$ for all zero-indexed updates $s \geq 6 0$ . YaRN supplies the extended context, and vLLM serves rollouts (Kwon et al., 2023; Peng et al., 2024).

Interaction and rendering. Executor calls use the respective APIs’ default decoding settings and the response-level advice interface in Section 3. The advisor retains its own advice and the complete observed history; new observations contain the message delta and tool schemas whenever they change. Guidance is appended to the latest user message in a temporary request copy using the marker in Appendix F; it does not persist in the executor history. Explicit <NO\_ADVICE> adds neither marker nor guidance. Blank or malformed outputs are tracked separately. Paired scoring (Section 5.2) uses identical advisor-tokenized response IDs in the with-advice and without-advice conditions, excluding subsequent tool outcomes from the target. The two scoring rows for each non-abstaining advice decision are batched across decisions. The predictive premise is that how strongly the advice changes the advisor’s prediction of the response helps identify useful supervision; we do not assume that the advisor reproduces the executor’s response law. Reflection can flag decisions with either contrast sign, so selection uses the magnitude. Because the response was observed with advice, removing the advice changes only the scoring context, not the observed execution. Donor calibration does not remove this asymmetry, and predictor error can afect both the sign and the magnitude of the contrast. Lemma 1,

Table 4 | BFCL training and evaluation settings. Token limits are configured capacities; a rollout need not reach them.  
Setting Value   
Advisor / parallelism Qwen3-8B, non-thinking template, full-parameter FSDP2   
Hardware / serving 4 H100 80GB GPUs; 4 colocated vLLM engines, tensor parallelism 1   
Training budget / independent 200 updates per run; 3 training runs   
runs   
Rollout batch / concurrency 8 tasks × 8 episodes = 64; 16 active environments   
Policy optimization 2 task groups per minibatch; microbatch 1 per GPU; 1 update epoch   
Learning rate $/$ reference penalty $1 0 ^ { - 6 } / 0 . 0 0 1 ;$ fixed reference policy   
Advisor sampling Temperature 0.7; top-� 1; top-� disabled; min-� 0   
Advice output limit 1,024 new tokens before each executor response   
Context configuration YaRN factor $4 ; 3 2 , 7 6 8$ original, 131,072 configured positions   
Rollout / serving limits 122,880 input tokens; 8,192-token chunked prefill; 2 sequences per   
engine   
Score / distillation temperature $1 . 0 / 0 . 7$   
Donor calibration quantile $u _ { \mathrm { d } } = 0 . 9 5 ;$ threshold frozen before training   
Teacher / token support Pre-update advisor; pre-update student top 100   
Auxiliary weight $\lambda _ { s } = 0 . 3 0 + ( 0 . 0 5 - 0 . 3 0 ) \operatorname* { m i n } ( s / 6 0 , 1 )$ , zero-indexed update �   
Reflection / teacher limits $b _ { \mathrm { r e f l } } = 5$ flagged advice decisions per eligible episode; 8,192 reflection   
output tokens; 6,144-token teacher block   
Executor step bound 20 steps per BFCL user turn; terminal-boundary guard permits 21 advice   
decisions   
Checkpoint selection / reported 4 evaluations on 80 validation tasks every 10 updates; highest mean   
evaluation oficial accuracy, earliest in a tie; 4 test evaluations per selected checkpoint  
by contrast, compares tokens under fixed completion and execution laws. AdviSD selects a decision’s advice-token losses using one recorded response and estimates neither those laws nor the lemma’s local value gradient.

Reward and reflection. BFCL training uses a dense call-matching reward based on arguments, execution errors, and extra calls; evaluation uses the oficial checker. Reflection is restricted to eligible imperfect episodes. An episode counts as successful according to the oficial boolean when the checker is recorded as having run, and otherwise when its final reward is at least $1 - 1 0 ^ { - 9 }$ . The reflector sees the complete observed episode and checks, including later messages that actually arrived, but no unrevealed requests or full reference solution. Corrections respect the evidence available at their decision. Gemini’s complete reflection request is checked with native token counting against a 1,000,000-token input cap; evidence is not truncated.

Calibration. Calibration is recomputed separately for every training run of each dataset–executor setting. Before training, the initial advisor supplies one rollout on each of 80 tasks from that run’s training split for both BFCL and EnvScaler, excluding its validation tasks and the held-out test tasks. The BFCL pilot uses 20 tasks per category. Valid non-abstaining decisions supply matched contrasts. Donors come from other tasks and never duplicate the issued advice. For $\mathrm { B F C L } ,$ candidates are ranked first by category agreement (same category first), then by increasing decision-index distance, and finally by increasing token-length distance. Selection cycles deterministically among up to eight highest-ranked candidates; category agreement is a preference, not a restriction. This ranking aims to make donors comparable in task category, interaction stage, and advice length. Donor advice is scored on the unchanged recipient response, never executed. We use $u _ { \mathrm { d } } = 0 . 9 5$ in Eq. (4), with linear empirical-quantile interpolation, and freeze the resulting threshold throughout that run. As admission checks, the BFCL pilot requires at least 200 matched decisions, a matched 90th percentile above the donor 95th percentile, and matched retention of at least 10%.

Algorithm 1 AdviSD training procedure   
Require: Advisor $\pi _ { \theta } ,$ fixed executor $\rho ,$ training tasks, frozen threshold $\epsilon _ { c } ,$ reflection cap $b _ { \mathrm { r e f l } }$   
1: for each training update do   
2: Fix the pre-update snapshot $\bar { \theta }  \theta$   
3: Collect episodes with $\pi _ { \boldsymbol { \bar { \theta } } } ,$ sampling fresh advice before every executor response   
4: Form the GRPO objective $\mathcal { L } _ { \mathrm { b a s e } }$ over all rollout episodes with unchanged advantages   
5: for each imperfect episode � do   
6: Reflect on complete observed evidence to flag at most $b _ { \mathrm { r e f l } }$ decisions J with correction   
feedback   
7: For valid $k \in \mathcal { T } _ { i }$ with issued advice, compute $c _ { i , k }$ using $\pi _ { \bar { \theta } }$ and Eq. (3)   
8: Retain in $\mathcal { I } _ { i }$ flagged original abstentions and valid issued-advice decisions with $| c _ { i , k } | > \epsilon _ { c }$   
9: Construct feasible feedback blocks for ${ \mathcal { T } } _ { i } ^ { * } \subseteq { \mathcal { T } } _ { i }$   
10: Cache teacher distributions from $\pi _ { \bar { \theta } }$ on the pre-update student’s top-� supports at the   
original advice prefixes   
11: end for   
12: Update only the advisor using the combined GRPO and self-distillation loss in Eq. (5)   
13: At validation checkpoints, evaluate the run’s validation split four times; select by the mean   
benchmark score   
14: end for

What the pilot threshold controls. For $n \geq 1$ donor contrasts $d _ { j } ,$ linear quantile interpolation uses rank $h = 1 + u _ { \mathrm { d } } ( n - 1 )$ in the sorted magnitudes. The threshold in Eq. (4) therefore gives

$$
\# \{ j : | d _ { j } | > \epsilon _ { c } \} \leq n - \lfloor h \rfloor = \lceil ( 1 - u _ { \mathrm { d } } ) ( n - 1 ) \rceil .
$$

At least the first ⌊ℎ⌋ sorted values are no greater than the threshold; ties can only reduce exceedances. This is a same-pilot count bound and needs no independence assumption. It does not control later donor pass rates or establish preferential retention of execution-sensitive corrections. Reflectionflagged original abstentions bypass this numeric gate.

Teacher construction. The teacher sees a feedback block placed before the original causal advisor context. The block contains the selected decision’s complete execution event, the relevant failed checks, the episode reward, and the reflection feedback. It excludes the completed advice, and exact echoes of that advice are removed from the reflection feedback. Distillation follows the fixed-prefix objective and loss averaging in Section 5.3 (Eq. (5)). At each prefix, the teacher and student distributions are renormalized over the pre-update student’s top 100 tokens. Feedback blocks exceeding the context budget are skipped rather than truncated.

## E. Baselines and evaluation protocol

Inference and outcome-learning controls. The standalone no-advisor control retains the native executor, tools, and policy. Frozen-advisor controls use untrained Qwen3-8B or the executor’s API model without task-specific optimization. The two frozen-advisor controls receive the same permitted information and use the same response cadence and abstention option. Outcome-only advisor-GRPO (GRPO in the tables) follows the outcome-based advisor-learning approach of Advisor Models (Asawa et al., 2026), using our response-level tool-use interface and GRPO on episode rewards, without self-distillation.

Executor prompt optimization. GEPA (Agrawal et al., 2026a) optimizes an instruction addition to the frozen executor’s prompt without an advisor or weight updates, and mandatory benchmark policies stay in place. For each executor and each in-domain benchmark (BFCL-v3 and EnvScaler), we run three independent searches, each with a budget of 8,192 scored task episodes. Gemini 3.7 Flash provides reflection on minibatches of eight task trajectories. We select the prompt with the highest mean over four complete validation evaluations on that benchmark; test tasks guide neither optimization nor selection. The selected addition is frozen for the corresponding benchmark’s test set. For out-of-domain evaluation, the BFCL-selected addition is transferred unchanged to the external benchmarks. The episode budget describes prompt search, not a compute-matched comparison with advisor training.

Other advisor training methods. We use the oficial SDPO and DistIL implementations, adapting their updates to the advisor’s generated tokens.<sup>1</sup> Unmodified algorithmic components retain their upstream defaults. Both methods use the common advisor rollout and optimizer settings in Table 4, but not the table’s AdviSD-specific teacher and loss settings. Neither SDPO nor DistIL includes an added GRPO objective. They are native-method baselines; AdviSD’s no-gate and selection variants provide the comparisons with a common outcome objective and feedback pipeline. The SDPO adaptation uses successful sibling rollouts of the same task as privileged feedback for a detached self-teacher at the student’s recorded advice prefixes (Hübotter et al., 2026; Zhang et al., 2026). When no successful sibling is available, we omit that task’s self-distillation loss. The teacher uses an EMA update rate of 0.01 (weight 0.99 on the previous teacher and 0.01 on the updated advisor). Its sibling-conditioned EMA teacher and residual-tail support difer from AdviSD’s feedback-conditioned pre-update teacher and renormalized top-� support. DistIL (Agrawal et al., 2026b) uses the same successful-sibling feedback as SDPO and skips self-distillation when no successful sibling is available. It replaces SDPO’s local reverse-KL objective with forward cross-entropy and full sequence-level gradients, including future-credit terms. These updates apply to the advisor’s generated tokens; the teacher remains detached.

Selection controls. The AdviSD selection controls share GRPO, reflection, teacher construction, fixed supports, loss normalization, and the auxiliary-weight schedule. The no-gate variant retains every feasible reflection proposal. Matched-count random selection uses its own batch’s feasible proposals. It always retains flagged decisions whose issued advice was <NO\_ADVICE>; among other proposals it samples uniformly without replacement the same number that the AdviSD threshold would retain. Counts are recomputed per episode and update after feasibility checks. Its nominal supervision count therefore matches the shadow gate on that proposal pool, and it supervises exactly the episodes the gate would supervise. Independently trained arms may still visit diferent states. Because the quota depends on the proposals, this procedure is not independent uniform thinning and need not preserve the ungated mixture of corrections. With one ordinary proposal, it reproduces the gate’s retain-or-reject decision. The inverted-gate variant retains every feasible ordinary proposal at or below the threshold and the same issued-abstention bypasses; it is not count-matched. The no-bypass variant retains the ordinary numeric gate but omits automatic supervision at reflection-flagged original abstentions. Inference-time abstention remains available in every variant.

Table 5 | Datasets and primary metrics. External counts describe the reference evaluation inventories; repeats do not create new task identities.
<table><tr><td>Benchmark</td><td>Split or evaluation inventory</td><td>Reported metric / protocol</td></tr><tr><td>BFCL</td><td>equally sized categories</td><td>400 train, 80 validation, 320 test; four Official backend-state and execution-response success</td></tr><tr><td>EnvScaler</td><td>1,880 train, 470 validation, 200 test</td><td>100× mean native fractional task score</td></tr><tr><td>ACEBench</td><td>20 multi-step and 30 multi-turn identities End-to-end success; native tools and user</td><td>simulation</td></tr><tr><td>ToolHop</td><td>995 queries; 3,912 tools</td><td>Answer Correctness; Free protocol, at most 9 executor responses</td></tr><tr><td>τ²-bench</td><td></td><td>50 airline, 114 retail, 114 telecom entries Per-domain pass¹; native user simulator and policies</td></tr><tr><td>RoTBench</td><td>Clean, 210 each Slight/Medium/Heavy, without executing tools 105 Union</td><td>840 records from 105 questions: 105 Released TS, PI, CF; first-turn text prediction,</td></tr></table>

Splits and scoring. BFCL fixes the same 320 test tasks across runs, with 80 per category. For each run, the remaining tasks are independently partitioned within each category into 100 training and 20 validation tasks, giving 400 training and 80 validation tasks overall. Every ten updates, the oficial checker evaluates that run’s 80 validation tasks four times. Each run selects the checkpoint with the highest mean accuracy, and ties go to the earliest update. Reported BFCL scores use the separate 320 test tasks. EnvScaler fixes 200 test tasks from the 2,550-task RL release across runs, then independently splits the remaining 2,350 tasks 80:20 into 1,880 training and 470 validation tasks for each run. Each run’s checkpoint maximizes mean native task score across four complete validation evaluations; reported results use the separate 200-task test set. We do not convert EnvScaler’s fractional reward into binary success. The performance scores in Tables 1–3 are test results, not checkpoint-selection validation scores. EnvScaler uses its native environments and scoring functions, with at most 30 executor responses per task, matching LOPD’s reported interaction budget (Zhang et al., 2026). Multiple tool calls within one response do not consume additional response slots.

BFCL uses the oficial multi-turn checker, with force termination counted as failure. Following LOPD (Zhang et al., 2026), we report per-category scores and their equally weighted average on our held-out test split. The separate irrelevance check is not part of the reported accuracy. External evaluation retains each benchmark’s native policies, stopping rules, tools, and scorer. Advice remains private to the executor and never reveals hidden user goals or grader state.

Transfer between executors. Figure 2 keeps the 320 BFCL-v3 test tasks fixed and changes the executor without further advisor training. Within-family transfer uses Gemini 3.7 Flash-trained advisors with Gemini 3.5 Flash and Claude Sonnet 4.6-trained advisors with Claude Sonnet 4.5. Cross-family transfer exchanges advisors between Gemini 3.7 Flash and Claude Sonnet 4.6. Native executor instructions and response-level advising are preserved. No-advisor and frozen-advisor controls are evaluated on the receiving executor; the API-advisor control uses that executor’s model. GEPA transfers its frozen instruction addition without reoptimization. This experiment changes the executor, whereas Table 3 changes the benchmark.

Means and uncertainty. All trained-method entries in Tables 1–3 and Figure 2 use three independent training runs. Each run’s validation-selected checkpoint receives four complete test evaluations, with Qwen3-8B advisor temperature 0.7. For Table 3, each trained method reuses its three BFCL-trained, validation-selected checkpoints from Table 1 across all external benchmarks, without further training.

For benchmark score $S _ { s , r }$ from training run � and evaluation repeat $r ,$ we report

$$
A _ { s } = \frac { 1 } { 4 } \sum _ { r = 1 } ^ { 4 } S _ { s , r } , \qquad \bar { A } = \frac { 1 } { 3 } \sum _ { s = 1 } ^ { 3 } A _ { s } , \qquad s _ { A } = \sqrt { \frac { 1 } { 2 } \sum _ { s = 1 } ^ { 3 } ( A _ { s } - \bar { A } ) ^ { 2 } } .
$$

GEPA uses the same aggregation across three independent prompt-optimization runs. No-advisor and frozen-advisor controls instead use three groups of four evaluations, so their SD measures evaluation variability, not training variability. For trained systems, $s _ { A }$ reflects training randomness, run-specific training/validation splits, and evaluation noise remaining in the run-level means; it is not a confidence interval. BFCL Avg is formed within each evaluation before this aggregation; category SDs are never averaged. Repeated evaluation estimates single-trial performance, not best-of-four success.

Table 3’s Macro Avg gives each benchmark equal weight. For the displayed component means, let � average ACEBench M-Step and M-Turn, � denote ToolHop AC, � average the three $\tau ^ { 2 }$ domains, and � average RoTBench TS, PI, and CF. We report

$$
\begin{array} { r } { \mathrm { M a c r o \ A v g } = \frac { 1 } { 4 } ( A + T + U + R ) . } \end{array}
$$

A benchmark’s weight therefore depends on neither its task count nor its number of reported components. This is a descriptive index across diferent native metrics, computed from the displayed means and rounded once to one decimal place. We report this index without an uncertainty estimate; component SDs cannot be averaged to obtain its SD.

Validation curves. Figure 3 uses Claude Sonnet 4.6 and checkpoints at updates 0, 20, . . . , 200. Update numbers count completed training updates; step 0 is the initial checkpoint. Training update $t \geq 1$ uses the zero-indexed auxiliary schedule at $s = t - 1$ (Table 4); the supervision phases below use the same one-based update numbering. For each of three training runs, we evaluate that run’s 80 validation tasks four times at each checkpoint. The line and band are the mean and sample SD of the three run-level averages, each over four evaluations, using the same nested aggregation as above. The band is not a confidence interval. Straight segments connect measured checkpoints without smoothing. Each run selects its checkpoint separately; the averaged curve is not used for selection.

Scope and limitations. The learning analysis assumes fixed teachers and stationary retention; its random-teacher extension also requires a stationary conditional law. These results do not establish convergence for AdviSD’s changing-teacher GRPO–AdamW training. Evaluation covers two executor families, one shared reflector, and three runs per trained method. The updates to API models can constrain exact reproducibility.

## E.1. Training-time supervision dynamics

Table 6 summarizes training-time supervision in one run with Claude Sonnet 4.6 on BFCL-v3. Early, middle, and late cover updates 1–20, 21–100, and 101–200. Step 0 is evaluation only; test evaluations do not enter these statistics. Each ratio divides the corresponding counts summed over the phase, rather than averaging per-update ratios.

Issued abstention is the fraction of advisor outputs that are <NO\_ADVICE>. Proposals per reflected episode is the number of reflection proposals divided by the number of episodes actually reflected. Ordinary retention is the fraction of proposals at originally non-abstaining decisions that pass the numeric gate, and bypass share is the fraction of selected proposals that come from originally abstaining decisions. Supervised decisions per episode counts the decisions actually included in the auxiliary loss and divides by all rollout episodes. Coverage uses the same episode denominator but counts episodes with at least one supervised decision. A rate below one supervised decision per episode therefore includes episodes that receive none, and bypass share is a share of decisions, not episodes.

Table 6 | Evolution of targeted supervision on BFCL-v3. Statistics describe one training run with frozen Claude Sonnet 4.6 and run-specific threshold $\epsilon _ { c } = 0 . 4 1 1 6 3 2$ . Early, middle, and late refer to updates 1–20, 21–100, and 101–200. Definitions below distinguish advisor decisions, reflection proposals, and rollout episodes.
<table><tr><td>Metric</td><td>Early</td><td>Middle</td><td>Late</td></tr><tr><td>Issued abstention (%)</td><td>23.8</td><td>39.3</td><td>43.5</td></tr><tr><td>Proposals per reflected episode</td><td>4.2</td><td>2.2</td><td>2.0</td></tr><tr><td>Ordinary gate retention (%)</td><td>58.0</td><td>36.9</td><td>32.2</td></tr><tr><td>Abstention-bypass share (%)</td><td>13.9</td><td>19.9</td><td>20.3</td></tr><tr><td>Supervised decisions per rollout episode</td><td>1.7</td><td>0.5</td><td>0.4</td></tr><tr><td>Episode coverage (%)</td><td>60.2</td><td>35.5</td><td>31.3</td></tr></table>

In this run, advice becomes less frequent, and both proposals per reflected episode and ordinary retention decline. Auxiliary supervision reaches fewer episodes, and the bypass accounts for approximately one-fifth of selected decisions in middle and late training. Both the numeric gate and the bypass remain active. In the model of Section 4.2, a growing insensitive share of proposals lowers aggregate retention when fixed classwise rates satisfy $r _ { I } < r _ { S }$ . The observed decline is compatible with this mechanism, but it does not identify those classes or rule out changes in score scale or in the proposal distribution. These are allocation statistics, not gradient magnitudes or causal sensitivity measurements; Table 2 evaluates selection through performance.

## F. Exact prompts

These are the exact fixed prompt components and message templates for the BFCL executor, advisor, reflector, and teacher. Braced fields denote runtime substitutions; doubled braces in the reflection JSON example are format escapes. The reflector flags at most $b _ { \mathrm { r e f l } } = 5$ advice decisions per eligible episode (max\_turns\_per\_ep=5, supplied as max\_turns in the template); this setting does not limit user turns or executor responses. External benchmarks retain their native executor policies and user protocols.

## Executor system message

You are a helpful assistant with access to tools. Complete the user’s request by calling the available tools one step at a time.

\- Call a tool only when you have all required parameters. If a required parameter is missing, ASK the user for it instead of guessing.

\- If no available tool can do what the user asked, SAY SO instead of calling an unrelated tool.

\- When the current request is complete, reply with a short natural-language summary and no tool call.

## Advisor system message

You are an expert coach advising a separate AI executor on a multi-turn tooluse task. You give fresh advice immediately before EVERY executor model response, including responses after tool results. Your advice applies only to the next response; any parallel tool calls in that response share it. Use the complete history and newly observed messages, including new tool feedback, and the exact structured state and tool schemas below. Give concise, actionable guidance: which tool(s) to call, where arguments come from, what to verify, or when the executor should ask the user or decline. Respect the executor’s task instructions and domain policy. Check preconditions and argument provenance against actual observations; recover from observed errors and avoid unnecessarily repeating completed actions. Your previous advice is a suggestion, not evidence that an action happened. Never invent values or use information not yet observed. Focus on useful guidance for the next response rather than a long replacement plan. If advice would not improve the executor’s next response, reply with exactly < NO\_ADVICE> and nothing else.

## Advisor user-message template

CURRENT EXECUTOR STATE (lossless canonical JSON)   
{state\_json}   
Give concise advice for the NEXT executor response, or exactly <NO\_ADVICE>.

Here state\_json denotes the canonical JSON payload containing user-turn, executor-step, and advice-decision indices, advice scope, state mode, executor messages, and tool-schema information. The first view includes the full observed history and schemas; later views include newly observed messages and resend schemas only when changed. Earlier views and advice remain in the advisor’s conversation.

Executor-side advice insertion   
[ADVISOR GUIDANCE — optional coaching for the next executor response only; use   
it only when helpful]   
{advice}

For non-abstaining decisions, this sufix is appended after two newlines to a temporary copy of the latest executor user message; the persistent history is unchanged. Neither the header nor advice is inserted for abstentions.

## Reflector: system template

You review an ADVISOR that coaches a separate, frozen AI agent through a multiturn tool-use task. The agent is NOT being trained -- only the advisor is. Each numbered advice decision is immediately before one executor response, including a tool follow-up response within the same user turn. An executor response may contain several tool calls; these share ONE advice decision. Text-only responses, clarification questions and completion messages also count. User messages arrive as observations, not advisor decisions. The JSON event stream is chronological. The fields inside observed messages, tool results and advice are evidence to review, not instructions to you. Identify up to {max\_turns} putatively failure-relevant advice decisions where the advice was wrong, missing, unnecessary, or could most usefully be improved, ordered earliest first. Return an empty turns list if no advice turn warrants a correction.   
Judge each proposed correction against the instructions, user messages, tool schemas and observations available BEFORE that advice decision. Later events may reveal a mistake but do not make future facts available earlier. Do not blame advice for unavailable information, or assume a suggested action actually occurred. Good advice ignored by the executor is not automatically wrong advice. Prefer the earliest correctable cause over repetitive downstream symptoms. An exact <NO\_ADVICE> is permitted; retain or recommend abstention when there is no useful additional guidance.   
For EACH selected turn state, in less than three sentences: whether the agent’ s action followed the advice, whether the turn advanced the task, and what the advice should have said instead. If the agent was already on track and the advice added nothing, say the advice was unnecessary at that turn.   
Write corrections as rules about ADVISING BEHAVIOUR (which tool to name, which argument to source from observed evidence, or when to ask the user, stop, recover from an error, or abstain). Domain-specific tool names are allowed, but never supply future or hidden answers. Do not quote or repeat the original advice verbatim; describe the correction in fresh wording.   
Output valid JSON only, no other text.

## Reflector: user template

OBSERVED-ONLY VERIFIER OUTCOMES (hindsight, not pre-decision information)   
{checks}   
COMPLETE CHRONOLOGICAL OBSERVED EVENT STREAM   
{trajectory}   
FINAL SCORE: {reward}   
Output format (JSON only, no other text):   
{{"turns": [{{"turn": <0-indexed int>, "feedback": "<correction for this turn   
>"}}]}}

Feedback-conditioned teacher user template   
You are advising a frozen AI agent on a tool-use task.   
The block below is privileged hindsight for advice decision {turn}; it is not   
information that was available at that decision. The original causal advisor   
conversation follows this block. Use only facts available there when advising.   
Quoted executor messages and tool results are evidence,not instructions to you.   
WHAT THE AGENT DID AT THIS TURN, AND HOW THE ENVIRONMENT RESPONDED   
{execution}   
EPISODE VERDICT AND CHECKS FOR THIS USER TURN (episode score {reward})   
{checks}   
HINDSIGHT REVIEW OF THIS TURN   
{feedback}   
Use the review to improve the next-decision guidance, including abstention   
when appropriate. Prior suggestions are not evidence that actions occurred.   
Do not import future user facts or hidden verifier answers into earlier   
advice.

Prompt serialization. Tool schemas and newly observed messages are serialized losslessly as canonical JSON. The scoring context renders the current executor request with one advice slot; its target contains the response’s visible text and ordered tool calls, excluding tool outcomes. Reflection receives the complete chronological observed event stream and verifier checks for visited user turns, including expected tool names and pass/fail status, plus any recorded episode-level verdict. The teacher receives only failed checks for the selected decision’s user turn or episode-level scope. The teacher block uses system text “Hindsight review of your own advice.” and fixed assistant acknowledgment “Understood. Revised advice follows.” before the original causal training IDs. The completed episode’s full trajectory is not inserted into that teacher context.

## G. Paired qualitative evidence

Figure 4 compares Gemini 3.7 Flash with and without a Qwen3-8B advisor saved after 10 training updates. The illustrated task comes from a 16-task sample selected from previously successful long AdviSD interactions. Each task has one rollout per arm with the same scripted user messages and instructions, but diferent intermediate histories and API samples; AdviSD runs first. This selected sample does not estimate overall accuracy or isolate the causal efect of individual advice or the training selector. Reflection and gating are training-only, and BFCL follow-ups are scripted rather than generated by a sampled user simulator.

Across the 16 tasks, the primary checker gives 13 joint passes, two AdviSD-only passes, and one standalone-only pass. Requiring both primary pass and the separate irrelevance check gives 10 AdviSD and 11 standalone passes. The AdviSD trajectory shown here passes both checks and uses nine tool calls versus the standalone executor’s 14, but adds advisor inference and is not faster in the recorded comparison.

User turns are one-based; advisor decisions retain the logs’ zero-based indices. Quotes are verbatim or marked as excerpts; other narrative is summarized. Tool-call layout and keyword order are normalized without changing values.

Task   
Book within the user’s stated flexibility. Case multi\_turn\_miss\_param\_154: verify travel details,   
price Chicago–Los Angeles travel, set a budget, book within it, and check recent messages.   
Abbreviated paired execution trajectory   
[User turns 1–2 / Both systems (summarized)]   
The user initially omits personal details. AdviSD’s advisor returns   
<NO\_ADVICE> at decision 0; both executors ask for the details. After they are   
supplied, both call verify\_traveler\_information. Both calls return   
verification\_status=false with the message “Passport must be issued by the   
United States.”   
[User turns 3–4 / Both systems (summarized)]   
Both resolve ORD/LAX and observe economy/business/first fares of   
\$180/\$360/\$900 for August 10, 2024. The user supplies token\_ABC123XYZ and   
requests a \$1,500 budget; both set it.   
[User turn 5]   
“Book this flight for me, please. I am flexible with travel classes but would   
prefer to keep it within my budget. I’ll promptly handle the payment using my   
saved card details with id ’card1’.”   
[Standalone response (excerpt)]   
“Please let me know which travel class you would prefer to book so I can   
finalize your reservation.” No booking call follows.   
[AdviSD advisor / Decision 9]   
“Call the book\_flight tool with the parameters: access\_token set to   
"token\_ABC123XYZ", card\_id set to "card1", travel\_date as "2024-08-10",   
travel\_from as "ORD", travel\_to as "LAX", and travel\_class as "economy" (or   
"business" if preferred, ensuring the cost is within the budget).”   
[Advised executor / Tool call and result]   
book\_flight(access\_token="token\_ABC123XYZ", card\_id="card1",   
travel\_date="2024-08-10", travel\_from="ORD",   
travel\_to="LAX", travel\_class="economy")   
{"booking\_id": "3426812", "transaction\_id": "45451592",   
"booking\_status": true, "booking\_history": {}}   
[User turn 6 / Both systems (summarized)]   
The user asks to check recently sent messages for trip details. Both retrieve   
the same three messages, none concerning the trip. The advised executor   
summarizes the returned messages. The standalone executor makes six   
additional searches that return no matches and again asks for a travel-class   
preference.   
Outcome comparison   
AdviSD: benchmark pass. Standalone: benchmark fail. The advised executor books within the user’s   
budget using the flexibility already granted. The standalone executor instead asks for a class preference   
and makes no booking call. The advisor provides six recommendations and seven abstentions across 13   
executor responses.  
Figure 4 | Completing a booking using the user’s stated flexibility. The advised executor books within budget, while the standalone executor requests a class preference and does not book in the recorded episode. All six user turns are represented; intermediate calls are summarized. Advice and booking arguments are taken from the recorded trajectory.

## H. Extended Related Work

This appendix expands the three themes of Section 2: how frozen models are adapted, how feedback becomes supervision, and how predictive comparisons guide learning.

## H.1. Advising and prompt optimization

Learning to guide a frozen model. A trainable model can adapt a frozen executor by learning what guidance to provide. Directional Stimulus Prompting learns instance-specific hints through supervised learning and rewards derived from the larger model’s outputs (Li et al., 2023). Matryoshka Pilot extends learned guidance to multi-turn interactions, using iterative direct preference optimization with optional behavior-cloning initialization (Li et al., 2025). Advisor Models trains natural-language advisors with GRPO on executor outcome rewards and supports repeated advising within an interaction (Asawa et al., 2026). It is the closest antecedent to our advisor-GRPO baseline, which follows this outcome-based approach through a response-level tool-use interface. AdviSD learns from targeted feedback as well as rewards and asks which proposed corrections provide useful supervision for a particular advisor–executor pair. Proxy-tuning uses a complementary interface, combining the target model’s token scores with the diference between those of tuned and untuned smaller models (Liu et al., 2024). AdviSD communicates through natural-language advice and requires no executor token scores.

Reflection and reusable instructions. Feedback can also improve behavior without updating model weights. Self-Refine iteratively critiques and revises outputs, Reflexion retains verbal feedback for subsequent attempts, and ExpeL extracts reusable insights from experience (Madaan et al., 2023; Shinn et al., 2023; Zhao et al., 2024). Prompt optimization turns similar feedback into changes to reusable instructions. ProTeGi combines textual critiques with prompt edits, beam search, and candidate selection (Pryzant et al., 2023). TextGrad propagates language feedback through computational graphs to optimize their constituent variables (Yuksekgonul et al., 2025), while GEPA uses reflection on execution traces within evolutionary prompt search (Agrawal et al., 2026a). These methods show that reflection can supply useful revisions. AdviSD uses such revisions to condition training-time supervision for a context-dependent advisor. The deployed advisor generates guidance without the reflector or completed-interaction feedback. Our GEPA baseline instead optimizes the executor’s reusable instructions directly.

## H.2. Feedback-conditioned distillation

Privileged information and student-generated trajectories. Generalized distillation formalizes learning from teachers with information unavailable to the student at prediction time (Lopez-Paz et al., 2015). In sequential learning, DAgger obtains expert supervision at states visited by the learner (Ross et al., 2011). GKD applies the corresponding on-policy distillation principle to language models, matching teacher distributions on student-generated sequences (Agarwal et al., 2024). Feedbackconditioned self-distillation combines these ideas: SDPO constructs self-teaching distributions using feedback or successful rollouts, supervising tokens from the student’s original rollout (Hübotter et al., 2026). DistIL optimizes forward cross-entropy with sequence-level credit, accounting for how earlier choices afect the later prefixes where distillation occurs (Agrawal et al., 2026b). AdviSD likewise supervises originally sampled prefixes, using a feedback-conditioned copy of the pre-update advisor. Its contribution concerns how to allocate this supervision when the student advises a separate executor, where matching a teacher’s advice distribution and improving the executor’s behavior are distinct objectives.

Constructing and localizing feedback. Trajectory-level feedback can be turned into supervision for particular decisions. HERO constructs local hints from completed trajectories and environment observations, distilling turns with nonempty, successfully parsed feedback (Liu et al., 2026). HinT-SD identifies failure-relevant actions and applies self-distillation to their token spans (Yeo et al., 2026). Retrieval supplies another source of teaching context: LOPD learns a latent-context composer over retrieved successful experience for a fixed-backbone teacher (Zhang et al., 2026), while DART-SD retrieves references to generate recovery continuations and trains on assistant steps after a graphlocalized breakpoint (Xu et al., 2026). These methods address both the content and location of feedback. AdviSD applies a predictive selection test to reflection-flagged advice decisions. The advisor scores the same recorded executor response with and without the originally issued advice. Retained feedback supervises the advisor along its original advice prefixes rather than training the executor or fitting a newly generated recovery trajectory.

Selecting examples, spans, and turns. Sparse supervision and learner-dependent selection have several precedents. Selective Reflection-Tuning refines instruction–response pairs and selects data compatible with the student (Li et al., 2024a). TRACE routes distillation to annotated spans and gradually restores GRPO on those spans as distillation decays (Wang et al., 2026a). SAGE-OPD uses environment feedback and teacher judgments to select and weight turn-level distillation, with additional confidence weighting (Zhou et al., 2026); these choices control the distillation loss rather than replace the student’s executed actions. RSTG targets all-failure rollout groups through selective token-level distillation and supervised learning from correct teacher trajectories (Han et al., 2026). AdviSD studies selection at the advisor–executor interface, where revising advice need not change execution. Its no-gate, matched-count random, and inverted-gate ablations compare retention rules within the same reflection and teaching pipeline (Q2 in Section 7).

## H.3. Predictive contrasts and selection

Using paired predictions to shape learning. Comparisons between diferently conditioned predictions can refine reward-based learning signals. RLCSD contrasts correct- and incorrect-hint conditioning, thresholds the contrast magnitude to select tokens, and modulates their GRPO advantages without reversing the outcome-based sign (Pan et al., 2026). PBSD compares ordinary and answer-conditioned action likelihoods to reweight turn-level outcome advantages (Tian et al., 2026). RLSD uses teacher–student log-probability diferences to construct sign-preserving token weights (Yang et al., 2026a). OCSD selects high-negative-log-likelihood interaction steps and contrasts replay contexts with and without future observations to modulate token-level GRPO advantages (Yang et al., 2026b). Paired scoring, contrast-based selection, and localized supervision therefore already have close precedents. AdviSD difers in the prediction target and the role of the contrast: the advisor scores a separate executor’s recorded response with and without issued advice, then uses the contrast magnitude to select auxiliary self-distillation. For a fixed rollout batch, the outcome-based GRPO advantages remain unchanged.

Predictive information and context usage. Context comparisons also assess what information a predictor uses. Conditional cross-mutual information compares translation log-likelihoods with and without additional context using a shared model (Fernandes et al., 2021). Pointwise V-information uses separately fitted input-conditioned and null-input predictors to measure an input’s instance-level predictive contribution (Ethayarajh et al., 2022). Instruction-Following Dificulty uses the ratio of instruction-conditioned to unconditioned response losses for data selection (Li et al., 2024b). AdviSD belongs to this broader family of predictive comparisons, but its score is not an estimate of formal pointwise V-information. It holds the advisor snapshot, interaction history, and recorded executor response fixed, varying only the issued advice. The score is a signed mean log-likelihood diference; selection uses its magnitude. Donor advice provides a calibration reference without being executed. A large contrast magnitude therefore identifies a change in the advisor’s prediction, not a measured behavioral efect or an improvement in task return.

Influence and cooperative credit assignment. The efect of one agent on another is also central to multi-agent credit assignment. Social-influence rewards encourage actions that change other agents’ behavior using counterfactual predictions (Jaques et al., 2019). COMA uses a centralized critic to marginalize one agent’s action while holding the others fixed (Foerster et al., 2018). For language-model agents, C3 evaluates alternative actions through rollouts from a restored interaction history (Chen et al., 2026), while CCPO constructs role-specific credit from agent-removal comparisons (Li et al., 2026). These methods address behavioral influence or the allocation of outcome credit. AdviSD’s selector instead decides which advice decisions receive auxiliary self-distillation. Rescoring an existing executor response avoids additional executor rollouts for selection, but does not estimate counterfactual task returns or replace the policy-gradient advantage.

Teacher mixtures and learning dynamics. Our analysis connects selection to the distinction between fitting a teacher and improving execution. The value-tilted teacher in Lemma 1 uses exponential reweighting related to relative-entropy policy search (Peters et al., 2010); it is an analytical reference, not a teacher constructed by AdviSD. Flux-OPD studies experience-conditioned teachers and contextual weighting, and derives a normalized geometric-mean target for a fixed teacher mixture at a fixed decoding history (Wang et al., 2026b). Our repeated-learning analysis considers how the sources of supervision change: as advice improves, preventable failures decline, changing the retained correction mixture even when individual teacher targets stay fixed. Under the stated shared-parameter assumptions, Theorems 1 and 2 explain how this composition and the frequency of supervision shape eventual performance. This motivates the matched-count control; the ablations evaluate AdviSD’s predictive selection rule without establishing that it identifies the theoretical sensitivity classes.