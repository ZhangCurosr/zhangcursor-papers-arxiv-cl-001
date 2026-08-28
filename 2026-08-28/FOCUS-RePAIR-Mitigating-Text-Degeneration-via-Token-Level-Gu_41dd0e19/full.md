# FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models

Junyoung Lee <sup>1</sup> Sehyeon Park <sup>2</sup> Shinhyoung Jang <sup>2</sup> Seonha Ryu <sup>2</sup> Hojeong Kim <sup>2</sup> Hyunsei Lee <sup>2</sup> Il hong Suh <sup>3</sup> Yeseong Kim <sup>1</sup>

## Abstract

Pruning is a practical approach to compress large language models (LLMs), but it can amplify text degeneration, especially repetition loops, even when perplexity and task accuracy remain largely unchanged. In this work, we present a tokenlevel analysis of this failure mode by viewing decoding as a dynamical process that enters and persists in a small set of recurrent contexts. Our analysis decomposes degeneration into loop entry risk and loop persistence, and shows that persistence is controlled by the escape mass assigned to plausible alternatives within the token sampling set. Motivated by these findings, we propose two token-level guidance objectives for post-pruning fine-tuning. FOCUS reweights distillation toward high-confidence teacher regions to suppress leakage, while RePAIR uses onset-centered positive/negative continuation pairs with a margin loss to promote plausible alternatives and prevent early commitment to repetition loops. Experiments on open-ended continuation and instruction-based generation show that both methods consistently reduce repetition and improve generation quality.

## 1. Introduction

Large language models (LLMs) have become foundational in a broad spectrum of applications, ranging from dialogue systems and code assistants to knowledge-intensive question-answering and content generation (Minaee et al., 2025; Jiang et al., 2024; Yue, 2025). In practical deployments, the execution of large models is far from straightforward: Inference latency escalates dramatically under constrained serving budgets, and memory requirements often exceed the limits of commodity hardware (Chitty-Venkata et al., 2024; Pope et al., 2022). To manage these constraints, prior research has developed pruning strategies, ranging from depth pruning, which eliminates entire transformer blocks, to width pruning, which removes internal submodules such as attention heads or MLP channels (Ma et al., 2023; Kim et al., 2024; Frantar & Alistarh, 2023; Lee et al., 2025). Because pruning inevitably discards parameters that encode useful behaviors, practitioners typically perform a post-pruning finetuning stage to restore degraded capabilities. For example, these techniques recover knowledge in pruning settings with a relatively small dataset, e.g., the Alpaca dataset with its instruction–response pairs (Taori et al., 2023).

Table 1. Comparison of repetition rates between unpruned and pruned models. The pruned model is finetuned on the Alpaca dataset. For decoding, we employ top-p sampling with p = 0.9.
<table><tr><td>Condition</td><td>Sampling</td><td>Greedy</td></tr><tr><td>Unpruned</td><td>5.9%</td><td>26.6%</td></tr><tr><td>Width pruned</td><td>12.4%</td><td>63.1%</td></tr><tr><td>Depth pruned</td><td>15.4%</td><td>63.7%</td></tr></table>

Most prior work on pruning has focused on knowledge preservation (Li et al., 2024; Park et al., 2024). Standard evaluations assess whether a pruned model preserves perplexity, zero-shot accuracy, and downstream task performance relative to its unpruned counterpart. However, there is growing empirical evidence that pruning can also introduce undesirable side effects, even when principal metrics such as perplexity or accuracy appear to be intact (Liebenwein et al., 2021; Jordao & Pedrini, 2021; Jaiswal et al., 2024). A primary issue is text degeneration, where the model repeatedly generates the same words or phrases, for example: ”<prefix> ... for the deceased. The cemetery is designed to be a peaceful place. The cemetery is designed to be a peaceful place. The cemetery is designed to be a peaceful place. ...”. To demonstrate this, we generate 200 tokens from the WikiText-103 dataset using a 50-token prefix with both the unpruned Llama model and the pruned model after fine-tuning. A sentence is classified as repetitive if repeated segments account for the majority of the generated text, as further discussed in Section 3. As shown in Table 1, the degeneration phenomenon becomes more severe after pruning in both cases. This observation indicates that, although simple fine-tuning can recover knowledge to some extent, it remains essential to mitigate the side effects that arise during text generation.

Previous studies have noted that text degeneration occurs when previously generated tokens increase the likelihood of the model producing the same tokens again (Holtzman et al., 2020; Welleck et al., 2019; Xu et al., 2022). To address this, these approaches lower the probabilities of previously generated tokens while increasing those of tokens that have not appeared. Although this strategy is effective in reducing repetition, it does not provide guidance on which tokens should be generated to produce a coherent continuation, resulting in degraded perplexity.

In this work, we identify repetition-loop degeneration as a token-level dynamical phenomenon induced by the interaction between the next-token distribution and the decoding rule. Using a coverage-based view of repetition, we find that degeneration typically behaves as a sharp entry event: replacing only a small number ofhigh-probability tokens at the loop onset is often sufficient to steer generation away from a repetitive trajectory. We formalize this behavior by decomposing repetition into (i) loop entry risk and (ii) loop persistence, where persistence dominates long-horizon repetition because it compounds over consecutive in-loop transitions. This analysis also explains why pruning and naive distillation can exacerbate degeneration by simultaneously increasing leakage into teacher-suppressed tokens (raising entry risk) and collapsing near-tie alternatives at loop-sensitive contexts (raising persistence).

These findings suggest that mitigating degeneration after pruning requires training-time control over token probabilities at loop-sensitive contexts: suppressing leakage toward teacher-suppressed tokens to reduce loop entry, while preserving plausible alternatives to prevent early commitment and reduce loop persistence. We therefore propose two complementary token-level guidance methods, FO-CUS and RePAIR. FOCUS modifies standard distillation by reweighting tokens to emphasize high-confidence teacher regions, discouraging probability mass from drifting into low-support (leakage) areas under capacity constraints. In contrast, RePAIR constructs paired continuations around degeneration onsets, i.e., a repetitive negative sample and a non-degenerate regeneration, and applies a margin-based objective that explicitly promotes plausible alternatives at the onset. We evaluate our proposed method in the post-pruning pipeline by pruning Llama-family models and fine-tuning with LoRA. Across open-ended continuation (WikiText-103) and instruction-based generation (Self-Instruct), FO-CUS and RePAIR consistently reduce repetition and improve distributional and semantic quality (e.g., MAUVE, CREP, EAD<sub>1</sub>, and BERTScore), while incurring only a small perplexity increase. Moreover, FOCUS is compatible with existing training-based mitigation objectives and improves their generation quality when combined.

## We summarize our contributions as follows:

• We introduce a token-level framework for repetition-loop degeneration that separates loop entry from loop persistence, and derive nucleus-decoding diagnostics that expose how probability allocation among nucleus-contained alternatives governs repetition.

• We propose two complementary token-level guidance objectives, FOCUS and RePAIR: FOCUS suppresses distillation-induced leakage by emphasizing highconfidence teacher regions, while RePAIR uses onsetcentered positive/negative continuation pairs to promote plausible alternatives at loop-sensitive contexts.

• Across open-ended continuation and instruction-based generation, our methods consistently reduce degeneration and improve generation quality (e.g., MAUVE and EAD<sub>1</sub>) with only a small perplexity increase.

## 2. Related Work

## 2.1. LLM Pruning & Distillation

Model pruning has been extensively investigated as an effective strategy to reduce model size and computational overhead. However, pruning inevitably introduces knowledge loss and results in performance degradation (Kim et al., 2024; Ma et al., 2023). To address this, knowledge distillation (KD) is commonly employed in the post-pruning fine-tuning stage to transfer knowledge from the teacher to the pruned student model (Xu et al., 2024; Gu et al., 2024), offering the advantage of leveraging soft probabilities to provide richer supervision than one-hot labels. Nevertheless, KD does not always yield consistent benefits (Ma et al., 2021; Zhang et al., 2025). For example, naive KD may transfer incorrect answers from the teacher and they demonstrate that student models trained with appropriate corrections of teacher logits can even outperform the teacher itself in classification tasks (Zhang et al., 2024). Several studies have also noted that naive knowledge distillation may fail to effectively train the student model, either due to the capacity gap between teacher and student or biases inherent in the teacher model (Zhong et al., 2024; Shum et al., 2024). These observations highlight the need for strategies that selectively extract and transfer only useful information from the teacher to improve training effectiveness.

## 2.2. Mitigation of Text Degeneration

Text degeneration has been addressed through both decoding-time and training-time strategies. On the decoding side, deterministic methods such as greedy and beam search often suffer from limited diversity and degeneration. Stochastic alternatives, including top-k (Fan et al., 2018) and top-p (nucleus) sampling (Holtzman et al., 2020), improve diversity by restricting generation to high-probability tokens, with top-p adapting to the distribution’s sharpness to produce more natural outputs.

Complementary to decoding-based methods, training-time approaches modify the learning objective to discourage repetition. Unlikelihood training penalizes previously generated tokens by reducing their probabilities (Welleck et al., 2019), while ScaleGrad adjusts token-level gradients to promote novel tokens (Lin et al., 2021). However, unlikelihoodbased methods often degrade perplexity. More recently, DITTO trains on synthetic sentence-level repetition data to suppress repeated tokens (Xu et al., 2022).

## 3. Token-level Guidance: Repetition-loop Dynamics and Supervision Signals

This section develops a token-level account of repetitionloop degeneration under common decoding rules, e.g., nucleus (top-p) decoding and identifies which parts of the next-token distribution must be corrected during training.

## 3.1. Coverage-based Repetition Metrics and Onset Sensitivity

Repetition-loop degeneration typically manifests as a short recurring N-gram that explains a large fraction of the generated sequence. To quantify this concentration, let $s _ { 1 : T } = ( s _ { 1 } , \ldots , s _ { T } )$ be a generated token sequence and let r denote an N-gram. We define the Coverage of r as the fraction of tokens explained by its occurrences:

$$
\operatorname { C o v e r a g e } ( r , s _ { 1 : T } ) \triangleq \frac { 1 } { T } \sum _ { j = 1 } ^ { T - N + 1 } \mathbf { 1 } [ s _ { j : j + N - 1 } \approx r ] \cdot N ,\tag{1}
$$

where $\approx$ is an exact match under tokenization (or a taskspecific equivalence used in evaluation). Since degeneration concentrates on a small number of patterns, we summarize each sequence by the dominant pattern:

$$
\mathrm { C o v e r a g e } ( s _ { 1 : T } ) \triangleq \operatorname* { m a x } _ { r \in \mathcal { V } ^ { N } } \mathrm { C o v e r a g e } ( r , s _ { 1 : T } ) ,\tag{2}
$$

and report dataset-level degeneration by the Coveragebased REPetition rate (CREP),

$$
\mathrm { C R E P } ( D ) \triangleq 1 0 0 \times \frac { 1 } { | D | } \sum _ { s \in D } \mathbf { 1 } [ \mathrm { C o v e r a g e } ( s ) \geq \tau ] ,\tag{3}
$$

Table 2. Degeneration rates of the unpruned model before and after correcting two onset tokens.
<table><tr><td>Condition</td><td>Before</td><td>After</td></tr><tr><td>Sampling</td><td>5.9%</td><td>0.7%</td></tr><tr><td>Greedy</td><td>26.6%</td><td>10.8%</td></tr></table>

where $\tau$ is a fixed threshold and $N , \tau$ are held constant unless stated otherwise. Coverage makes repetition measurable as a dominance event: once a short pattern occupies a large portion of the sequence, decoding spends many steps revisiting a small set of local contexts.

We next show that repetition commonly behaves as an entry event that can be disrupted by a minimal local intervention. We generate 1,000 sequences, identify degenerated outputs using CREP, and locate the onset of the first repetition loop as the earliest position where the dominant N-gram begins to recur and then continues over a contiguous span. At this onset, we modify only the first two tokens by selecting an alternative among the top-2 candidates under the model distribution at the same prefix, then regenerate the continuation and re-evaluate CREP.

Table 2 shows that replacing only two onset tokens reduces CREP from 5.9% to 0.7% under sampling and from 26.6% to 10.8% under greedy decoding. The replacements are restricted to high-probability candidates, so the local continuation remains plausible while the trajectory shifts away from a repetitive basin. This sensitivity indicates that degeneration often depends on a small subset of loop-sensitive contexts near the first onset.

## 3.2. Decoding Dynamics: Entry Risk and Persistence

To connect onset sensitivity to token-level probabilities, we adopt a decoding-dynamics view. For a fixed N-gram context representation $c _ { t } \triangleq s _ { t - N + 1 : t - 1 }$ , the model induces a conditional distribution $p _ { \theta } ( \cdot \mid c _ { t } )$ . Under a fixed decoding rule (for example, top-p sampling), generation induces a stochastic process over contexts because sampling selects a token and the context updates deterministically. Repetitionloop degeneration corresponds to trajectories that enter and then remain within a small recurrent subset of contexts. We denote such a repetition-prone region by ${ \mathcal { L } } ,$ which can be operationalized as the set of contexts that repeatedly regenerate the dominant $N .$ -gram identified by Coverage.

This perspective separates degeneration into loop entry and loop persistence. Let the loop-hitting time be $\tau _ { \mathcal { L } } \triangleq$ min $\{ t \geq 1 : c _ { t } \in \mathcal { L } \}$ and define the horizon-T entry risk

$$
\begin{array} { r } { \mathcal { R } _ { T } \triangleq \mathbb { P } ( \tau _ { \mathcal { L } } \leq T ) . } \end{array}\tag{4}
$$

Once decoding reaches ${ \mathcal { L } } ,$ repetition depends on the probability of remaining inside it. For $c \in { \mathcal { L } } .$ , define the one-step

persistence probability

$$
\rho ( c ) \triangleq \mathbb { P } ( c _ { t + 1 } \in \mathcal { L } \mid c _ { t } = c ) , \qquad \bar { \rho } \triangleq \operatorname* { s u p } _ { c \in \mathcal { L } } \rho ( c ) .\tag{5}
$$

Coverage-based degeneration requires sustained residence in ${ \mathcal { L } } .$ Let $\ell _ { \tau }$ be the minimum number of consecutive in-loop transitions required for Coverage $\left( s _ { 1 : T } \right)$ to exceed τ. Then a simple decomposition upper bounds degeneration by

$$
\begin{array} { r } { \mathbb { P } \mathrm { ( C o v e r a g e } ( s _ { 1 : T } ) \geq \tau ) \ \leq \ \mathcal { R } _ { T } \cdot \bar { \rho } ^ { \ell _ { \tau } } . } \end{array}\tag{6}
$$

Equation (6) yields a structural implication.<sup>1</sup> Loop entry contributes linearly through $\mathcal { R } _ { T }$ , whereas persistence is exponentiated by the required run length $\ell _ { \tau }$ . As a result, moderate reductions in $\bar { \rho }$ can suppress long repetition runs even when entry events are not fully eliminated. This explains why changing only a few high-probability tokens near onset can produce a large reduction in CREP in Table 2.

## 3.3. Nucleus Alternatives and Escape Mass

The remaining question is which token-level distortions control $\rho ( c )$ under nucleus sampling, where only candidates inside the nucleus set are sampleable. Fix a context c and let $S _ { p } ( c )$ denote the nucleus set, defined as the smallest set of tokens whose probability mass under $p _ { \theta } ( \cdot \mid c )$ is at least $p .$ Top-p sampling draws the next token from the renormalized distribution on $S _ { p } ( c )$ . Therefore, persistence inside a loop region depends on how the nucleus mass is divided between loop-continuing tokens and tokens that exit the loop region.

For a loop context $c \in { \mathcal { L } }$ , define the set of loop-continuing nucleus tokens

$$
A ( c ) \triangleq \left\{ v \in S _ { p } ( c ) : \operatorname { u p d a t e } ( c , v ) \in \mathcal { L } \right\} ,\tag{7}
$$

where update $( c , v )$ is the context update after appending token v. Let

$$
a ( c ) \triangleq \sum _ { v \in A ( c ) } p _ { \theta } ( v \mid c ) , \qquad e ( c ) \triangleq \sum _ { u \in S _ { p } ( c ) \backslash A ( c ) } p _ { \theta } ( u \mid c ) ,\tag{8}
$$

where $a ( c )$ is the unnormalized loop mass within the nucleus and $e ( c )$ is the escape mass within the nucleus. Since nucleus sampling renormalizes over $S _ { p } ( c )$ , the persistence probability satisfies

$$
\rho ( c ) = \frac { a ( c ) } { a ( c ) + e ( c ) } .\tag{9}
$$

This identity makes the control mechanism explicit: reducing persistence requires increasing escape mass inside the nucleus, not merely increasing entropy in the full vocabulary.

![](images/888a76ac4d08b6a287b2e79522c74eb1550ead8be058dd64ffc32f5553cdd9bb.jpg)  
Figure 1. Mode averaging under forward KL minimization. The student curve corresponds to the numerically optimized singlemode approximation of a multimodal teacher.

## 3.4. How Pruning and Naive Distillation Reshape Risks

Previous work notes that text degeneration often arises when model uncertainty falls outside a stable entropy range, with both overly confident and overly uncertain distributions (Arora et al., 2023). Building on this observation, we now connect common student distortions after pruning and standard knowledge distillation to the two drivers in (6). Let $q ( \cdot \mid c )$ denote the teacher distribution and $p ( \cdot \mid c )$ the student distribution. Naive distillation commonly minimizes the forward KL divergence

$$
{ \mathcal { L } } _ { \mathrm { K L } } ( q \| p ) = \sum _ { v \in \mathcal { V } } q ( v \mid c ) \log { \frac { q ( v \mid c ) } { p ( v \mid c ) } } .\tag{10}
$$

Because each token contributes proportionally to $q ( v \mid c )$ tokens with negligible teacher probability receive little direct weight in (10). Under capacity constraints induced by pruning, the student may not represent the teacher’s multi-modal structure in a single context. In that case, minimizing (10) can yield a single broadened approximation that compromises across teacher modes, which redistributes probability into intermediate and low-density regions of the teacher distribution. Figure 1 illustrates this mode-averaging behavior in a continuous toy example.

Tail Leakage Increases Loop Entry Risk. Define the teacher-suppressed set

$$
{ \mathcal { T } } _ { \epsilon } ( c ) \triangleq \{ v \in \mathcal { V } : q ( v \mid c ) \leq \epsilon \} ,\tag{11}
$$

and the student’s leakage mass $\begin{array} { r } { \Delta _ { \epsilon } ( c ) \triangleq \sum _ { v \in \mathcal { T } _ { \epsilon } ( c ) } p ( v \mid c ) } \end{array}$ Forward KL provides limited pressure against allocating mass to $\mathcal { T } _ { \epsilon } ( c )$ , especially when the student must approximate multiple teacher modes with reduced capacity. If leakage mass shifts into tokens that are occasionally admitted into $S _ { p } ( c )$ , these tokens become sampleable under nucleus decoding and can increase the entry term $\mathcal { R } _ { T }$

Alternative Collapse Increases Loop Persistence. At loop-sensitive contexts near the first onset, teachers often assign comparable probability to multiple semantically consistent next tokens. Pruning-induced representational loss can distort these near-ties, producing a single dominant continuation and suppressing competing alternatives. Under nucleus sampling, this distortion reduces escape mass $e ( c )$ relative to loop mass $a ( c )$ in (8), thereby increasing $\rho ( c )$ through (9) and inflating ${ \bar { \rho } } .$

Taken together, pruning and naive distillation can increase both loop entry risk and loop persistence by (i) allocating non-negligible probability to teacher-suppressed tokens and (ii) collapsing teacher-supported alternatives at loopsensitive contexts.

These observations translate into two supervision targets that follow directly from the entry-persistence decomposition. First, to reduce entry risk, training should suppress leakage toward teacher-suppressed regions so that such tokens do not enter the nucleus set and become sampleable triggers under top-p decoding. Second, to reduce persistence, training should preserve and, when necessary, promote plausible escape alternatives within the nucleus at loop-sensitive contexts, thereby increasing escape mass $e ( c )$ and decreasing $\rho ( c )$ via (9) rather than injecting diffuse probability into tokens that remain outside $S _ { p } ( c )$ . Section 4 instantiates these targets with two complementary objectives: one explicitly controls tail leakage relative to the teacher, and the other reallocates probability among onset-adjacent candidates to maintain nucleus-level escape mass and prevent early commitment to repetition loops.

## 4. Method

In this section, we propose the token probability weighted distillation method, which follows the trend of useful teacher probability but mitigates degeneration. Next, we propose the repetition-aware pairwise alignment, which directly guides the model to generate probable alternative tokens.

## 4.1. FOCUS: FOcus on Confident Token Under Teacher Supervision

FOCUS Training Objective. Given a dataset $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $\{ s _ { i } \} _ { i = 1 } ^ { N } .$ , where each sequence is tokenized as $\begin{array} { r l } { s _ { i } } & { { } = } \end{array}$ $( s _ { i , 1 } , \ldots , s _ { i , T } )$ , let $z ^ { T } ( s _ { i , < t } ) \in \mathbb { R } ^ { V }$ denote the teacher logits over the vocabulary $\dot { \nu }$ at position t. We define the student and teacher predictive distributions as

$$
p _ { i , t } \equiv p _ { \theta } ( \cdot \mid s _ { i , < t } ) , \quad q _ { i , t } \equiv \mathrm { s o f t m a x } \Bigg ( \frac { z ^ { T } ( s _ { i , < t } ) } { \tau } \Bigg ) ,
$$

We introduce a token-wise weight

$$
w _ { i , t } \big ( s _ { i , t } \big ) \ = \ \big ( q _ { i , t } \big ( s _ { i , t } \big ) \big ) ^ { \beta } \ + \ \big ( 1 - q _ { i , t } \big ( s _ { i , t } \big ) \big ) ^ { \gamma } , \quad \beta , \gamma \geq 0 ,
$$

which emphasizes tokens where the teacher exhibits high confidence. Thus, naive KD is reformulated as

$$
\mathcal { L } _ { \mathrm { F o C U S } } = \frac { 1 } { N T } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left( \tau ^ { 2 } \sum _ { v \in \mathcal { V } } w _ { i , t } ( v ) q _ { i , t } ( v ) \log \frac { q _ { i , t } ( v ) } { p _ { i , t } ( v ) } \right) .
$$

Gradient Analysis of FOCUS From the FOCUS objective

$$
{ \cal L } _ { \mathrm { F O C U S } } = \sum _ { i } w ( q _ { i } ) q _ { i } \log \frac { q _ { i } } { p _ { i } } ,
$$

Applying the chain rule,

$$
{ \frac { \partial L } { \partial a _ { k } } } = \sum _ { i } { \frac { \partial L } { \partial p _ { i } } } { \frac { \partial p _ { i } } { \partial a _ { k } } } = Z p _ { k } - w _ { k } q _ { k } , \quad { \mathrm { w h e r e ~ } } Z = \sum _ { j } w _ { j } q _ { j } .
$$

The reweighted teacher distribution and gradient can be expressed as

$$
\tilde { q } _ { k } = \frac { w _ { k } q _ { k } } { Z } , \quad \nabla _ { a } L _ { \mathrm { F O C U S } } = Z ( p - \tilde { q } ) .
$$

Thus, FOCUS preserves the standard KD form while effectively replacing the teacher distribution with a reweighted version ${ \tilde { q } } ,$ making it easy to optimize. The detailed derivation is provided in Appendix B.

## 4.2. RePAIR: Repetition-aware PAIRwise Alignment

As discussed in Section 3, token-level guidance can alleviate repetition, but it cannot be applied at runtime since detecting the onset of a repetition loop requires access to future tokens. To address this, we propose RePAIR, a repetition-aware pairwise alignment that provides token-level corrective signals exactly where degeneration begins, guiding the student toward non-repetitive and contextually coherent generation. An example is provided in Appendix C.

## 4.2.1. PAIRWISE DATA COLLECTION

Given a prefix length $k ,$ we generate model outputs $\hat { y } _ { i }$ from inputs $s _ { i , 0 : k }$ . Using the Coverage metric described in Section 3, sequences with repetition above threshold are collected as negative samples $D _ { \mathrm { n e g } }$ . The index $r _ { i }$ of the first repetition defines a shorter prefix $s _ { i , 0 : ( r _ { i } - 1 ) }$ , from which we regenerate outputs to obtain positive samples $D _ { \mathrm { p o s } } .$ . In practice, negative data are produced by the pruned model, while positive data are sampled from the unpruned model with topp decoding, yielding realistic pruned/unpruned comparisons. We set the threshold to 0.3.

## 4.2.2. REPAIR TRAINING OBJECTIVE

We define a token-level pairwise margin loss to encourage the model to assign higher confidence to positive continuations than to negative ones.

Training Objective. Let $p _ { \theta } ( \cdot \mid \cdot )$ denote the model distribution. We compute token-level negative log-likelihood only for the continuation after the prefix $\boldsymbol { s } _ { i , 0 : ( r _ { i } - 1 ) } ^ { \mathrm { p r e } }$

$$
\ell _ { i } ^ { + } = - \frac { 1 } { T _ { i } ^ { + } - r _ { i } + 1 } \sum _ { t = r _ { i } } ^ { T _ { i } ^ { + } } \log p _ { \theta } \left( s _ { i , t } ^ { \mathrm { p o s } } \middle | s _ { i , 0 : ( r _ { i } - 1 ) } ^ { \mathrm { p r e f i x } } , s _ { i , < t } ^ { \mathrm { p o s } } \right) ,
$$

$$
\ell _ { i } ^ { - } = - \frac { 1 } { T _ { i } ^ { - } - r _ { i } + 1 } \sum _ { t = r _ { i } } ^ { T _ { i } ^ { - } } \log p _ { \theta } \left( s _ { i , t } ^ { \mathrm { n e g } } \Bigm | s _ { i , 0 : ( r _ { i } - 1 ) } ^ { \mathrm { p r e f i x } } , s _ { i , < t } ^ { \mathrm { n e g } } \right) .
$$

We encourage the model to prefer the positive continuation over the negative one by a margin-ranking loss:

$$
\mathcal { L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m a x } \big ( 0 , m + \ell _ { i } ^ { + } - \ell _ { i } ^ { - } \big ) .\tag{12}
$$

## 4.2.3. TOTAL LOSS FUNCTION

The final objective can be expressed as

$$
{ \cal L } _ { \mathrm { t o t a l } } = { \cal L } _ { \mathrm { C E } } + \alpha _ { 1 } \cdot { \cal L } _ { \mathrm { F O C U S } } + \alpha _ { 2 } \cdot { \cal L } _ { \mathrm { R e P A I R } }
$$

Unless otherwise specified, $\alpha _ { 1 }$ is fixed at 0.05 and $\alpha _ { 2 }$ at 1 for all experiments. A discussion on parameter selection is provided in Appendix F.

## 5. Experiments

## 5.1. Setup

We conduct experiments on the Llama family. We prune 25% of model width using LLMPruner and fine-tune on Alpaca, a standard benchmark for post-pruning knowledge recovery, using LoRA for two epochs. To stabilize the pruned model and maintain consistency with its pre-pruning behavior, we apply self-distillation. We further evaluate pruning rates of 35% and 45%, with results reported in Appendix I.

We evaluate our method on open-ended and instructionbased generation. Following prior work, for open-ended generation, we sample 1,000 instances from WikiText-103 and generate 100 tokens from a 50-token prefix. For instruction-based generation, we use 1,000 prompts from Self-Instruct (Wang et al., 2023). Across both tasks, we apply top-p sampling $( p = 0 . 9 )$ and report both performance and degeneration metrics. We include both to ensure that methods designed to mitigate repetition do not excessively degrade task performance.

Performance Metrics We report performance using the following metrics:

• Perplexity: Perplexity (PPL) measures how well a language model predicts the next token, with lower values indicating better predictive performance.

• BERTScore (BS): BERTScore evaluates the semantic similarity between generated text and reference text by computing the cosine similarity of contextualized embeddings from a pretrained BERT model (Zhang et al., 2020). We report the F1 score.

• Zero-shot Accuracy: It reflects the model’s ability to apply its general knowledge and reasoning skills. Details on the tasks and evaluation settings can be found in Appendix D.

• MAUVE: MAUVE measures the distributional similarity between generated and reference sentences in the embedding space (Pillutla et al., 2021).

Degeneration Metrics To capture the degeneration of mod-els, we analyze:

• Unique n-gram: Unique n-gram rate is defined as $1 0 0 \times$ $\begin{array} { r } { ( 1 - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } } \end{array}$ <sup>|Unique</sup> <sup>n-gram(sentencei)|</sup> ). We set $n = 3 , 4 , 5 , 6$ |Total n-gram(sentence<sub>i</sub>)| to capture both word-level and phrase-level diversity (Xu et al., 2022).

• Expected 1-gram Diversity $( \mathbf { E A D } _ { 1 } ) \colon \mathrm { E A D } _ { 1 }$ quantifies lexical diversity by comparing the number of unique tokens in the generated text to the expected count under random sampling. A higher value indicates greater diversity, and it is formally defined as $\frac { N } { V _ { \mathrm { e f f } } \left( 1 - \left( 1 - \frac { 1 } { V _ { \mathrm { e f f } } } \right) ^ { C } \right) } ,$ where N is the number of unique tokens in the generated text, C is the total number of generated tokens, and $V _ { \mathrm { e f f } }$ is the effective vocabulary size (Liu et al., 2022).

• CREP: CREP computes the fraction of sentences dominated by repeated n-grams (above 30%). For each sentence, we evaluate repetition across n-gram sizes $r \in [ 4 , 1 6 ]$ and mark it as degenerated if any r exceeds the threshold. This metric provides an intuitive measure of sentence-level repetition. Implementation details are provided in Appendix G.

Comparison Methods. We compare our method with four training-based baselines: (1) KD, which fine-tunes the pruned model via knowledge distillation; (2) UL-Token, which penalizes previously generated tokens (Welleck et al., 2019); (3) ScaleGrad (SG), which re-normalizes probabilities for non-generated tokens (Lin et al., 2021); (4) DITTO, which constructs sentence-repetition data and penalizes repetitive generations (Xu et al., 2022). Implementation details and hyperparameters are provided in Appendix E.

## 5.2. Open-ended Generation Task

As shown in Table 3, when applied alone, RePAIR outperforms most baselines across multiple metrics. For instance, it achieves the highest MAUVE score of 0.68 and the lowest CREP score of 2.23, indicating that it generates text more closely resembling real data with minimal degeneration.

Table 3. Results of open-ended generation on the WikiText-103 dataset. Best per block in bold, second best underlined. CREP and Unique n-gram are reported as percentage values.
<table><tr><td rowspan="2">Method</td><td rowspan="2">PPL (↓)</td><td rowspan="2">0-shot (↑)</td><td rowspan="2">MAUVE (↑)</td><td rowspan="2">CREP (↓)</td><td colspan="4">Unique n-gram</td></tr><tr><td>n=3</td><td>n=4</td><td>n=5</td><td>n=6</td></tr><tr><td colspan="8">Llama-3.1-8B</td></tr><tr><td>KD</td><td>21.69</td><td>60.39</td><td>0.61</td><td>7.3</td><td>13.28</td><td>9.99</td><td>8.05</td><td>6.79</td></tr><tr><td>+ UL</td><td>21.67</td><td>61.15</td><td>0.61</td><td>5.37</td><td>11.79</td><td>8.68</td><td>6.85</td><td>5.70</td></tr><tr><td>+ SG</td><td>21.69</td><td>60.88</td><td>0.66</td><td>7.8</td><td>12.87</td><td>9.69</td><td>7.85</td><td>6.66</td></tr><tr><td>+ DITTO</td><td>22.07</td><td>60.62</td><td>0.63</td><td>5.27</td><td>11.38</td><td>8.14</td><td>6.22</td><td>5.01</td></tr><tr><td>+ RePAIR</td><td>22.05</td><td>61.36</td><td>0.68</td><td>2.23</td><td>8.36</td><td>5.36</td><td>3.74</td><td>2.78</td></tr><tr><td>FOCUS</td><td>22.32</td><td>60.03</td><td>0.64</td><td>1.73</td><td>6.22</td><td>3.82</td><td>2.62</td><td>1.94</td></tr><tr><td>+ UL</td><td>23.20</td><td>60.45</td><td>0.69</td><td>0.77</td><td>4.30</td><td>2.43</td><td>1.51</td><td>1.07</td></tr><tr><td>+ SG</td><td>22.71</td><td>60.47</td><td>0.70</td><td>0.87</td><td>4.86</td><td>2.89</td><td>1.96</td><td>1.47</td></tr><tr><td>+ DITTO</td><td>22.88</td><td>60.54</td><td>0.72</td><td>0.5</td><td>4.49</td><td>2.42</td><td>1.46</td><td>1.11</td></tr><tr><td>+ RePAIR</td><td>23.12</td><td>60.86</td><td>0.73</td><td>0.57</td><td>3.68</td><td>1.92</td><td>1.14</td><td>0.75</td></tr><tr><td colspan="9">Llama-2-13B</td></tr><tr><td>KD</td><td>13.90</td><td>64.10</td><td>0.81</td><td>1.13</td><td>6.84</td><td>4.02</td><td>2.49</td><td>1.67</td></tr><tr><td>+ UL</td><td>13.90</td><td>64.28</td><td>0.79</td><td>1.50</td><td>6.73</td><td>4.02</td><td>2.58</td><td>1.78</td></tr><tr><td>+ SG</td><td>13.90</td><td>64.15</td><td>0.82</td><td>1.20</td><td>6.87</td><td>4.08</td><td>2.59</td><td>1.78</td></tr><tr><td>+ DITTO</td><td>14.19</td><td>63.85</td><td>0.74</td><td>1.30</td><td>7.14</td><td>4.24</td><td>2.70</td><td>1.84</td></tr><tr><td>+ RePAIR</td><td>13.95</td><td>64.36</td><td>0.81</td><td>0.50</td><td>5.55</td><td>2.97</td><td>1.67</td><td>1.01</td></tr><tr><td>FOCUS</td><td>15.41</td><td>63.69</td><td>0.84</td><td>0.13</td><td>2.84</td><td>1.20</td><td>0.55</td><td>0.29</td></tr><tr><td>+ UL</td><td>15.74</td><td>63.57</td><td>0.85</td><td>0.07</td><td>2.40</td><td>0.94</td><td>0.38</td><td>0.17</td></tr><tr><td>+ SG</td><td>15.47</td><td>63.77</td><td>0.84</td><td>0.10</td><td>2.57</td><td>1.07</td><td>0.47</td><td>0.23</td></tr><tr><td>+ DITTO</td><td>15.85</td><td>63.52</td><td>0.86</td><td>0.03</td><td>2.46</td><td>0.93</td><td>0.37</td><td>0.16</td></tr><tr><td>+ RePAIR</td><td>15.50</td><td>63.91</td><td>0.87</td><td>0.00</td><td>2.24</td><td>0.82</td><td>0.32</td><td>0.13</td></tr><tr><td>Wiki-103</td><td>一</td><td>I</td><td>-</td><td>-</td><td>2.62</td><td>1.10</td><td>0.52</td><td>0.24</td></tr></table>

Moreover, it exhibits a distribution of unique n-gram scores comparable to that of the original WikiText-103 dataset. When combined with FOCUS, it consistently improves all metrics while incurring only a slight increase in perplexity. On average, the MAUVE score improves by about 0.06 across methods, accompanied by a marked decrease in the CREP metric, indicating a significant reduction in repetitive generations. Notably, RePAIR achieves both the highest MAUVE score and unique n-gram distributions that closely match real WikiText-103 text. Overall, these results demonstrate that our proposed methods substantially mitigate text degeneration while preserving perplexity and downstream task performance, indicating that the proposed training strategy incurs no meaningful performance degradation across both stand-alone and combined settings.

## 5.3. Instruction-based Generation Task

For instruction-based generation, since the output length for each instruction may vary across methods, we use $\mathrm { E A D _ { 1 } }$ as the primary diversity metric instead of unique n-gram diversity, as the latter does not account for differences in generation length. As illustrated in Table 4, without FOCUS, the model achieves the lowest degeneration performance despite recording the best perplexity among baselines. In contrast, RePAIR attains a lower CREP score of 0.63 and the highest $\mathrm { E A D _ { 1 } }$ score of 0.32, while maintaining a comparable perplexity to UL. This indicates that RePAIR produces the most diverse generations while preserving semantic fidelity, as reflected in BERTScore. When applied with FOCUS, all of the methods show consistent improvements across metrics with only a slight increase in perplexity. For example, it achieves lower CREP, higher $\mathbf { E A D _ { 1 } }$ , and improved BERTScore, while maintaining stable zero-shot accuracy. This indicates that the method mitigates degeneration and enhances generation quality without incurring knowledge loss. Notably, when FOCUS is combined with RePAIR, most metrics achieve the best results among all methods. Overall, these findings demonstrate that our proposed approaches not only mitigate degeneration more effectively than existing baselines but also preserve general knowledge and semantic quality in instruction-following tasks.

Table 4. Results of instruction-based generation on Self-Instruct dataset. Best per block in bold, second best underlined. CREP is reported as percentage values. (\*BS: BERTScore)
<table><tr><td>Method</td><td>PPL (↓)</td><td>0-shot (↑)</td><td>CREP (↓)</td><td>EAD1 (↑)</td><td> $\mathbf { B } \mathbf { S } ^ { \ast } \left( \uparrow \right)$ </td></tr><tr><td colspan="6">Llama-3.1-8B-Instruct</td></tr><tr><td>KD</td><td>25.65</td><td>62.32</td><td>1.97</td><td>0.28</td><td>0.49</td></tr><tr><td>+ UL</td><td>25.52</td><td>62.01</td><td>3.10</td><td>0.28</td><td>0.48</td></tr><tr><td>+ SG</td><td>25.53</td><td>62.30</td><td>2.23</td><td>0.28</td><td>0.49</td></tr><tr><td>+ DITTO</td><td>25.59</td><td>61.97</td><td>1.30</td><td>0.28</td><td>0.49</td></tr><tr><td>+ RePAIR</td><td>25.86</td><td>62.61</td><td>0.63</td><td>0.32</td><td>0.49</td></tr><tr><td>FOCUS</td><td>26.54</td><td>62.85</td><td>0.73</td><td>0.30</td><td>0.50</td></tr><tr><td>+ UL</td><td>26.89</td><td>62.07</td><td>0.73</td><td>0.29</td><td>0.49</td></tr><tr><td>+ SG</td><td>26.27</td><td>62.85</td><td>0.53</td><td>0.30</td><td>0.50</td></tr><tr><td>+ DITTO</td><td>26.36</td><td>62.12</td><td>0.30</td><td>0.29</td><td>0.50</td></tr><tr><td>+ RePAIR</td><td>26.20</td><td>62.61</td><td>0.23</td><td>0.31</td><td>0.50</td></tr></table>

![](images/f839cc56c0603799d9d50cc159816d6a83867fe88e670b02895e605bbc521a63.jpg)  
Figure 2. FOCUS analysis. Comparison of token distributions between naive KD and FOCUS.

Table 6. LLM-as-a-judge comparison results. \* indicates FOCUS applied.
<table><tr><td>Comparison</td><td>Win (A)</td><td>Tie</td><td>Win (B)</td></tr><tr><td>KD vs FOCUS</td><td>40.10%</td><td>19.80%</td><td>40.10%</td></tr><tr><td>FOCUS vs RePAIR*</td><td>36.00%</td><td>22.60%</td><td>41.40%</td></tr><tr><td>UL* vs RePAIR*</td><td>34.60%</td><td>24.70%</td><td>40.70%</td></tr><tr><td>SG* vs RePAIR*</td><td>35.50%</td><td>24.30%</td><td>40.20%</td></tr><tr><td>DITTO* vs RePAIR*</td><td>37.40%</td><td>24.30%</td><td>38.30%</td></tr></table>

Table 5. Repetition comparison of DPO and RePAIR under the WikiText-103 continuation setup using Llama-3.1-8B. For a clear comparison, KD and FOCUS are not applied here, and CE is used as the baseline.
<table><tr><td rowspan="2">Method</td><td rowspan="2">1 PPL (↓)</td><td colspan="4">Unique n-gram</td></tr><tr><td>n=3</td><td>n=4</td><td>n=5</td><td>n=6</td></tr><tr><td>CE</td><td>23.05</td><td>12.88</td><td>9.68</td><td>7.77</td><td>6.56</td></tr><tr><td>DPO</td><td>23.29</td><td>9.76</td><td>6.53</td><td>4.64</td><td>3.45</td></tr><tr><td>RePAIR</td><td>23.72</td><td>8.28</td><td>5.53</td><td>3.97</td><td>3.01</td></tr></table>

## 5.4. LLM-as-a-Judge Evaluation

The current evaluation mainly relies on automatic metrics, which may not fully capture the perceptual quality of generated continuations. To provide a complementary assessment, we additionally conduct an LLM-as-a-judge evaluation. Following the experimental setup of Table 3, we generate Wiki continuation samples using Llama-3.1-8B and compare the outputs produced by different methods. We use GPT-120B (reasoning) as the judge. Detailed settings can be found in Appendix K.

As reported in Table 6, our method consistently outperforms the baselines in the LLM-as-a-judge evaluation, in addition to showing improvements on conventional automatic metrics. These results further support that the proposed method improves generation quality beyond what is captured by static repetition or perplexity-based measurements.

![](images/8b57cbbaee4f95e4d28db9800eb1eb47f12ada26aa928714576f208198de0c1a.jpg)  
Figure 3. TruthfulQA-MC2 score. X indicates that no repetitionsuppression technique was applied.

## 6. Analysis

FOCUS Distribution Study As discussed in Section 3, our method reweights token-level supervision to align the student with both high- and low-confidence regions of the teacher distribution, preserving teacher-preferred tokens while maintaining the relative structure in the tail. To empirically verify this behavior, we sample 200 responses generated by the teacher and compare the student’s token probability distribution against the teacher’s distribution. We quantify alignment using agreement rate (AR), $\begin{array} { r } { \mathrm { A R } _ { \mathrm { h e a d } } \ = \ \frac { | H _ { q } \cap \hat { H } _ { p } | } { | H _ { q } | } } \end{array}$ and $\begin{array} { r } { \mathrm { A R } _ { \mathrm { t a i l } } \ = \ \frac { | \bar { T _ { q } } \cap T _ { p } | } { | T _ { q } | } } \end{array}$ , and Pearson correlation (COR), where the head is defined by top-p $( p = 0 . 9 )$ . As shown in Figure 2, tail agreement remains stable while tail correlation increases substantially, indicating that our method preserves the relative probability geometry of the teacher’s tail rather than merely matching the token set. This is consistent with our formulation in Section 3, which emphasizes structural alignment in lowconfidence regions instead of indiscriminately flattening tail probabilities. In the head region, agreement increases substantially while the number of viable head candidates increases (148.6 → 202.1). This suggests that our method maintains teacher-aligned high-probability tokens while redistributing probability mass among near-onset alternatives, expanding the set of plausible sampling candidates without sacrificing fidelity to the teacher.

Robustness to Likelihood Instability Based on Tables 3 and 4, we observe a consistent increase in PPL when applying our methods, which may reflect reduced likelihood stability or distributional drift. To examine whether this affects factual behavior, we evaluate LLaMA-3.1-8B on TruthfulQA. As shown in Fig. 3, although FOCUS increases PPL on WikiText, it improves TruthfulQA performance, and RePAIR achieves the largest overall gain. We attribute this to its token-level supervision, which explicitly guides the student toward teacher-preferred tokens. Similarly, FOCUS outperforms vanilla KD, consistent with prior work showing that capacity-aware distillation helps prioritize salient teacher signals under limited student capacity (Wu et al., 2024; Yao et al., 2025; Ko et al., 2024).

![](images/e0cb94b3445f44a96e834ab6484536d565721b705e25a0f5185a5842dd1966ee.jpg)  
Figure 4. Persistence metric across methods.

Onset-Level vs. Sequence-Level Supervision RePAIR adopts a pairwise training paradigm similar to preferencebased methods such as DPO (Rafailov et al., 2024), but differs in the granularity of its supervision. Whereas DPO provides sequence-level preferences, RePAIR operates at the onset-level by supplying corrective alternatives at repetition onset, directly targeting early loop entry. We compare RePAIR with DPO under the WikiText-103 continuation setting (Table 5). While DPO reduces repetition to some extent, RePAIR achieves larger gains, aligning with our analysis that fine-grained, onset-level corrective signals can be particularly effective for mitigating repetition. Implementation details are provided in Appendix J.

Empirical Analysis of Loop Persistence. To examine how FOCUS affects loop persistence, we conduct an empirical analysis using synthetic repetition contexts. We construct examples from WikiText-103 by taking 1,000 consecutive sentence pairs, using the first sentence as the prefix, and repeating the second sentence 10 times. We then measure $\begin{array} { r } { \rho ( c ) = \frac { a ( c ) } { a ( c ) + e ( c ) } } \end{array}$ as defined in Eq. (9), where a(c) is the probability mass assigned to the next token on the repeated trajectory and e(c) is the mass assigned to all other tokens. We report the average ρ(c) over repeated positions.

Figure 4 shows that KD quickly saturates at a high ρ(c) as sentence occurrence increases, indicating that the loopcontinuing token increasingly dominates the escape mass and makes the loop harder to escape. In contrast, FOCUS delays this increase and saturation, suggesting reduced loop persistence. Combining FOCUS with RePAIR further suppresses ρ(c), indicating an additional reduction in the tendency to remain in repetition loops.

Comparison with Distribution-shaping Distillation. FOCUS can be viewed as a distribution-shaping distillation objective, as it modifies the teacher distribution before applying distillation. To further examine whether other distribution-shaping distillation objectives can also mitigate repetition, we additionally compare FOCUS with ToDi, a token-level hybrid-KL distillation method that combines the benefits of forward and reverse KL (Jung et al., 2025). As shown in Table 7, although ToDi improves generation quality, it is less effective than FOCUS in reducing repetition, as indicated by its higher unique n-gram repetition rates and CREP. Combined with RePAIR, FOCUS further reduces repetition while preserving generation quality, achieving the lowest repetition scores with a MAUVE score comparable to ToDi.

Table 7. Comparison with ToDi on Llama 3.1-8B, following the same experimental setting as Table 3. \* indicates FOCUS applied.
<table><tr><td rowspan="2"></td><td rowspan="2">Method MAUVE (↑) CREP (↓)</td><td rowspan="2"></td><td colspan="4">Unique n-gram</td></tr><tr><td></td><td> $n = 3 ~ n = 4 ~ n = 5 ~ n = 6$ </td><td></td><td></td></tr><tr><td>ToDi</td><td>0.73</td><td>6.73</td><td>12.13</td><td>9.00</td><td>7.19</td><td>6.05</td></tr><tr><td>FOCUS</td><td>0.64</td><td>1.73</td><td>6.22</td><td>3.82</td><td>2.62</td><td>1.94</td></tr><tr><td>RePAIR*</td><td>0.73</td><td>0.57</td><td>3.68</td><td>1.92</td><td>1.14</td><td>0.75</td></tr></table>

![](images/9610e300e565c5b380dab9744a3e0d10cf75a3c24fe095602b72c7553b1394f5.jpg)  
Figure 5. Number of pairwise data for RePAIR.

Number of Pairwise Data We collect 12k pairwise samples and analyze the amount of data required to mitigate text degeneration. As shown in Figure 5, only about 4k samples are sufficient to achieve repetition rates comparable to those obtained with the full dataset. This indicates that RePAIR is more data-efficient than DITTO, which consumes half of the training set for its auxiliary loss. As a result, our method preserves more data for standard training, enabling the model to learn broader knowledge.

## 7. Conclusion

We show that pruning increases repetition in language models and that token-level guidance can effectively mitigate this issue. To address repetition, we propose FOCUS, a token probability weighted distillation approach, and Re-PAIR, a pairwise margin-based objective that guides models toward better alternative tokens. Both methods consistently reduce repetition and improve generation quality.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

This work was supported by the NRF grant funded by the Korea government (MSIT) (RS-2025-24803164), the Ministry of Trade, Industry Energy (MOTIE, Korea) under the project “An Unbreachable Multilayer AI-based Security Mechanism that Continuously Adapts and Evolves in Dynamic Conditions” (RS-20202653102), the Technology Innovation Program “Development of Navigation Technology Utilizing Visual Information Based on Vision-Language Models for Understanding Dynamic Environments in Non-Learned Spaces” (RS-2024-00445759).

## References

Arora, K., O’Donnell, T. J., Precup, D., Weston, J., and Cheung, J. C. K. The stable entropy hypothesis and entropy-aware decoding: An analysis and algorithm for robust natural language generation, 2023. URL https: //arxiv.org/abs/2302.06784.

Chitty-Venkata, K. T., Raskar, S., Kale, B., Ferdaus, F., Tanikanti, A., Raffenetti, K., Taylor, V., Emani, M., and Vishwanath, V. Llm-inference-bench: Inference benchmarking of large language models on ai accelerators, 2024. URL https://arxiv.org/abs/2411. 00136.

Fan, A., Lewis, M., and Dauphin, Y. Hierarchical neural story generation, 2018. URL https://arxiv.org/ abs/1805.04833.

Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot, 2023. URL https://arxiv.org/abs/2301.00774.

Gu, Y., Dong, L., Wei, F., and Huang, M. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pp. 32694–32717, 2024.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration, 2020. URL https://arxiv.org/abs/1904.09751.

Jaiswal, A., Gan, Z., Du, X., Zhang, B., Wang, Z., and Yang, Y. Compressing llms: The truth is rarely pure and never simple, 2024. URL https://arxiv.org/ abs/2310.01382.

Jiang, J., Wang, F., Shen, J., Kim, S., and Kim, S. A survey on large language models for code generation, 2024. URL https://arxiv.org/abs/2406.00515.

Jordao, A. and Pedrini, H. On the effect of pruning on adversarial robustness, 2021. URL https://arxiv. org/abs/2108.04890.

Jung, S., Yoon, S., Kim, D., and Lee, H. Todi: Token-wise distillation via fine-grained divergence control. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 8089–8102, 2025.

Kim, B.-K., Kim, G., Kim, T.-H., Castells, T., Choi, S., Shin, J., and Song, H.-K. Shortened llama: Depth pruning for large language models with comparison of retraining methods, 2024. URL https://arxiv.org/abs/ 2402.02834.

Ko, J., Kim, S., Chen, T., and Yun, S.-Y. Distillm: Towards streamlined distillation for large language models, 2024. URL https://arxiv.org/abs/2402.03898.

Lee, J., Jang, S., Kim, S., Park, J., Suh, I. H., Chwa, H. S., and Kim, Y. Late breaking results: Dynamically scalable pruning for transformer-based large language models. In 2025 Design, Automation & Test in Europe Conference (DATE), pp. 1–2. IEEE, 2025.

Li, J., Lei, Q., Cheng, W., and Xu, D. Towards robust pruning: An adaptive knowledge-retention pruning strategy for language models, 2024. URL https: //arxiv.org/abs/2310.13191.

Liebenwein, L., Baykal, C., Carter, B., Gifford, D., and Rus, D. Lost in pruning: The effects of pruning neural networks beyond test accuracy, 2021. URL https:// arxiv.org/abs/2103.03014.

Lin, X., Han, S., and Joty, S. Straight to the gradient: Learning to use novel tokens for neural text generation, 2021. URL https://arxiv.org/abs/2106.07207.

Liu, S., Sabour, S., Zheng, Y., Ke, P., Zhu, X., and Huang, M. Rethinking and refining the distinct metric. arXiv preprint arXiv:2202.13587, 2022.

Ma, H., Chen, T., Hu, T.-K., You, C., Xie, X., and Wang, Z. Undistillable: Making a nasty teacher that cannot teach students, 2021. URL https://arxiv.org/abs/ 2105.07381.

Ma, X., Fang, G., and Wang, X. Llm-pruner: On the structural pruning of large language models, 2023. URL https://arxiv.org/abs/2305.11627.

Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M., Socher, R., Amatriain, X., and Gao, J. Large language models:

A survey, 2025. URL https://arxiv.org/abs/ 2402.06196.

Park, S., Choi, H., and Kang, U. Accurate retrainingfree pruning for pretrained encoder-based language models, 2024. URL https://arxiv.org/abs/2308. 03449.

Pillutla, K., Swayamdipta, S., Zellers, R., Thickstun, J., Welleck, S., Choi, Y., and Harchaoui, Z. Mauve: Measuring the gap between neural text and human text using divergence frontiers, 2021. URL https://arxiv. org/abs/2102.01454.

Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Levskaya, A., Heek, J., Xiao, K., Agrawal, S., and Dean, J. Efficiently scaling transformer inference, 2022. URL https://arxiv.org/abs/2211.05102.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290.

Shum, K., Xu, M., Zhang, J., Chen, Z., Diao, S., Dong, H., Zhang, J., and Raza, M. O. First: Teach a reliable large language model through efficient trustworthy distillation, 2024. URL https://arxiv.org/abs/ 2408.12168.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. Alpaca: A strong, replicable instruction-following model, 2023. URL https://github.com/tatsu-lab/ stanford\_alpaca.

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language models with self-generated instructions, 2023. URL https://arxiv.org/abs/2212.10560.

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., and Weston, J. Neural text generation with unlikelihood training, 2019. URL https://arxiv.org/abs/ 1908.04319.

Wu, T., Tao, C., Wang, J., Yang, R., Zhao, Z., and Wong, N. Rethinking kullback-leibler divergence in knowledge distillation for large language models, 2024. URL https://arxiv.org/abs/2404.02657.

Xu, J., Liu, X., Yan, J., Cai, D., Li, H., and Li, J. Learning to break the loop: Analyzing and mitigating repetitions for neural text generation, 2022. URL https://arxiv. org/abs/2206.02369.

Xu, X., Li, M., Tao, C., Shen, T., Cheng, R., Li, J., Xu, C., Tao, D., and Zhou, T. A survey on knowledge distillation of large language models, 2024. URL https://arxiv.org/abs/2402.13116.

Yao, W., Yang, W., Wang, Z., Lin, Y., and Liu, Y. Revisiting weak-to-strong generalization in theory and practice: Reverse kl vs. forward kl, 2025. URL https: //arxiv.org/abs/2502.11107.

Yue, M. A survey of large language model agents for question answering, 2025. URL https://arxiv.org/ abs/2503.19213.

Zhang, J., Gao, Y., Liu, R., Cheng, X., Zhang, H., and Chen, S. Can students beyond the teacher? distilling knowledge from teacher’s bias, 2024. URL https:// arxiv.org/abs/2412.09874.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. Bertscore: Evaluating text generation with bert, 2020. URL https://arxiv.org/abs/1904.09675.

Zhang, Z., Shamsabadi, A. S., Lu, H., Cai, Y., and Haddadi, H. Membership and memorization in llm knowledge distillation, 2025. URL https://arxiv.org/abs/ 2508.07054.

Zhong, Q., Ding, L., Shen, L., Liu, J., Du, B., and Tao, D. Revisiting knowledge distillation for autoregressive language models, 2024. URL https://arxiv.org/ abs/2402.11890.

## A. Repetition Analysis

![](images/f9405e5a63b7adcea9a223765cdfc45f7b83c93f6d7f85622825ff7aaa483f34.jpg)  
Figure 6. Token probabilities across repetition cycles.

Following prior work, we conduct a qualitative diagnostic experiment to illustrate how repetition loops amplify token probabilities over time, using the motivating example introduced in the Introduction. Specifically, we construct an input sequence consisting of a fixed prefix followed by repeated instances of a short sentence:

<prefix> $\cdot \cdot \cdot$ for the deceased. The cemetery is designed to be a peaceful place. The cemetery is designed to be a peaceful place. The cemetery is designed to be a peaceful place. . . .

At each step, we record the model’s probability assigned to the observed next token and group these values by their within sentence position. As illustrated in Figure 6, the resulting trajectories show that repeated-token probabilities rapidly saturate, indicating a near-deterministic repetition pattern once a loop is formed. While the persistence probability $\rho ( c )$ in Eq. (6) is not directly observable, the saturation of mean probability over repeated tokens provides an empirical approximation of high persistence, suggesting that once the decoding trajectory enters a repetition loop, it becomes increasingly difficult for the model to escape.

## B. Gradient Analysis of FOCUS

In this section, we analyze why FOCUS has effects on the suppressing repetition loop.

Gradient Analysis of Knowledge distillation. The Knowledge Distillation (KD) loss is defined as the Kullback–Leibler (KL) divergence between the teacher distribution q and the student distribution $p \mathrm { : }$

$$
L _ { \mathrm { K D } } = \sum _ { i } q _ { i } \log { \frac { q _ { i } } { p _ { i } } } .\tag{13}
$$

Differentiating with respect to the softmax input a (logit), we compute:

$$
{ \frac { \partial L _ { \mathrm { K D } } } { \partial a _ { k } } } = - \sum _ { i } q _ { i } { \frac { \partial \log p _ { i } } { \partial a _ { k } } } .
$$

The derivative of the log-softmax is:

$$
\frac { \partial \log p _ { i } } { \partial a _ { k } } = \delta _ { i k } - p _ { k } ,
$$

where $\delta _ { i k }$ is the Kronecker delta. Substituting back, we get:

$$
\frac { \partial L _ { \mathrm { K D } } } { \partial a _ { k } } = - q _ { k } + \sum _ { i } q _ { i } p _ { k } .
$$

Since $\textstyle \sum _ { i } q _ { i } = 1$ , this simplifies to:

$$
\frac { \partial L _ { \mathrm { K D } } } { \partial a _ { k } } = p _ { k } - q _ { k } .\tag{14}
$$

We can express it in vector form as:

$$
\nabla _ { a } L _ { \mathrm { K D } } = p - q .\tag{15}
$$

Gradient Analysis of FOCUS By modifying the KL objective, FOCUS defines the loss function as:

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { \mathrm { F O C U S } } = \sum _ { i } w ( q _ { i } ) q _ { i } \log \frac { q _ { i } } { p _ { i } } } } \\ { { \displaystyle ~ = \sum _ { i } w ( q _ { i } ) q _ { i } \log q _ { i } - \sum _ { i } w ( q _ { i } ) q _ { i } \log p _ { i } } . } \\ { { \displaystyle ~ \mathrm { c o n s t a n t } ( \mathrm { i n d e p e n d e n t o f } a ) } } \end{array}\tag{16}
$$

(17)

Differentiating $L _ { \mathrm { F O C U S } }$ with respect to $p _ { i }$ , we obtain:

$$
\frac { \partial L } { \partial p _ { i } } = - \frac { w _ { i } q _ { i } } { p _ { i } } .\tag{18}
$$

To compute the derivative with respect to the logits a, we need the softmax Jacobian:

$$
\frac { \partial p _ { i } } { \partial a _ { k } } = p _ { i } \left( \delta _ { i k } - p _ { k } \right) ,\tag{19}
$$

where $\delta _ { i k }$ is the Kronecker delta. By the chain rule, for each coordinate k:

$$
{ \frac { \partial L } { \partial a _ { k } } } = \sum _ { i } { \frac { \partial L } { \partial p _ { i } } } { \frac { \partial p _ { i } } { \partial a _ { k } } }\tag{20}
$$

$$
= \sum _ { i } \left( - { \frac { w _ { i } q _ { i } } { p _ { i } } } \right) p _ { i } \left( \delta _ { i k } - p _ { k } \right) .\tag{21}
$$

$$
= \sum _ { i } \left( - w _ { i } q _ { i } \right) \left( \delta _ { i k } - p _ { k } \right) .\tag{22}
$$

Next, we separate the summation into the cases $i = k$ and $i \neq$ k:

$$
\frac { \partial L } { \partial a _ { k } } = ( - w _ { k } q _ { k } ) ( 1 - p _ { k } ) + \sum _ { i \neq k } ( - w _ { i } q _ { i } ) ( - p _ { k } ) .\tag{23}
$$

Finally, we obtain:

$$
\frac { \partial L } { \partial a _ { k } } = - w _ { k } q _ { k } \ + \ p _ { k } \sum _ { i } w _ { i } q _ { i } .\tag{24}
$$

Let

$$
Z : = \sum _ { j } w _ { j } q _ { j } > 0\tag{25}
$$

Then the gradient can be written as

$$
\frac { \partial L } { \partial a _ { k } } = Z p _ { k } - w _ { k } q _ { k } .\tag{26}
$$

Now, define a reweighted teacher distribution $\tilde { q }$ as

$$
\widetilde { q } _ { k } \triangleq \frac { w _ { k } q _ { k } } { Z } .\tag{27}
$$

Substituting, we obtain

$$
\frac { \partial L } { \partial a _ { k } } = Z ( p _ { k } - \tilde { q } _ { k } ) .\tag{28}
$$

In vector form, the gradient of FOCUS loss is:

$$
\nabla _ { a } L _ { \mathrm { F O C U S } } \ = \ Z ( p - \tilde { q } ) , \quad \mathrm { w h e r e } \ \tilde { q } = \frac { w ( q ) \odot q } { \sum _ { j } w ( q _ { j } ) q _ { j } } ,\tag{29}
$$

To this end, FOCUS can be viewed as optimizing the student distribution with respect to a reweighted teacher distribution.

## C. An Example of Pairwise Data

![](images/ee9cb6fa544755d985da86b30a40e84c76caff02c8dbef9297fb12c7627c99e2.jpg)  
Figure 7. Example of pairwise data construction.

As illustrated in Figure 7, we first provide an input prompt to the pruned model and identify the onset of degeneration using the coverage metric. The sequence up to this point is designated as the prefix. We then feed the same input along with the prefix to the unpruned model to obtain a positive continuation that remains free of degeneration.

Finally, we construct training pairs of the form (prompt + prefix, negative continuation) and (prompt + prefix, positive continuation) to optimize the margin loss. In total, we collect 12k pairwise data. However, as shown in Figure 5, even 4k pairs are sufficient, highlighting the cost-effectiveness of the approach.

## D. Zero-shot Accuracy

Evaluation Tasks We conduct zero-shot evaluations on the following benchmark tasks:

• ARC c (AI2 Reasoning Challenge): A multiple-choice science exam dataset that evaluates complex reasoning ability beyond simple fact recall.

• PIQA (Physical Interaction Question Answering): A benchmark for testing physical commonsense reasoning, where models must choose the more plausible solution to everyday tasks.

• BoolQ (Boolean Questions): A reading comprehension dataset consisting of yes/no questions with corresponding passages from Wikipedia.

Table 8. Zero-shot evaluation results on the Llama-3.1-8B model across six benchmarks. The best result within each block is highlighted in bold, while the second best is underlined.
<table><tr><td>Model</td><td>Arc_c</td><td>PIQA</td><td>BoolQ</td><td>OpenQA</td><td>Hellaswag</td><td>Winogrande</td><td>Average</td></tr><tr><td>KD</td><td>43.26</td><td>77.58</td><td>67.52</td><td>40.00</td><td>71.05</td><td>62.90</td><td>60.39</td></tr><tr><td>+ UL</td><td>44.37</td><td>77.37</td><td>69.60</td><td>41.20</td><td>71.08</td><td>63.30</td><td>61.15</td></tr><tr><td>+ SG</td><td>43.52</td><td>77.58</td><td>69.69</td><td>40.80</td><td>71.11</td><td>62.59</td><td>60.88</td></tr><tr><td>+ DITTO</td><td>43.43</td><td>77.42</td><td>69.27</td><td>40.40</td><td>70.85</td><td>62.35</td><td>60.62</td></tr><tr><td>+ RePAIR</td><td>43.17</td><td>77.86</td><td>70.55</td><td>41.20</td><td>71.86</td><td>63.54</td><td>61.36</td></tr><tr><td>FOCUS</td><td>42.92</td><td>77.26</td><td>65.87</td><td>40.20</td><td>70.70</td><td>63.22</td><td>60.03</td></tr><tr><td>+ UL</td><td>43.34</td><td>77.15</td><td>67.55</td><td>40.80</td><td>70.42</td><td>63.46</td><td>60.45</td></tr><tr><td>+ SG</td><td>42.92</td><td>77.20</td><td>68.17</td><td>41.00</td><td>70.69</td><td>62.83</td><td>60.47</td></tr><tr><td>+ DITTO</td><td>42.66</td><td>77.15</td><td>69.08</td><td>40.80</td><td>70.46</td><td>63.06</td><td>60.54</td></tr><tr><td>+ RePAIR</td><td>43.52</td><td>77.42</td><td>69.79</td><td>40.00</td><td>71.60</td><td>62.83</td><td>60.86</td></tr></table>

Table 9. Zero-shot evaluation results on the Llama2-13B model across six benchmarks. The best result within each block is highlighted in bold, while the second best is underlined.
<table><tr><td>Model</td><td>Arc_c</td><td>PIQA</td><td>BoolQ</td><td>OpenQA</td><td>Hellaswag</td><td>Winogrande</td><td>Average</td></tr><tr><td>KD</td><td>45.65</td><td>77.64</td><td>73.12</td><td>44.20</td><td>75.98</td><td>68.03</td><td>64.10</td></tr><tr><td>+ UL</td><td>46.59</td><td>77.75</td><td>73.15</td><td>43.80</td><td>76.05</td><td>68.35</td><td>64.28</td></tr><tr><td>+ SG</td><td>45.90</td><td>77.69</td><td>73.15</td><td>44.00</td><td>76.07</td><td>68.11</td><td>64.15</td></tr><tr><td>+ DITTO</td><td>45.48</td><td>77.64</td><td>72.91</td><td>43.20</td><td>75.74</td><td>68.11</td><td>63.85</td></tr><tr><td>+ RePAIR</td><td>46.59</td><td>77.75</td><td>72.78</td><td>44.20</td><td>76.80</td><td>68.03</td><td>64.36</td></tr><tr><td>FOCUS</td><td>45.99</td><td>77.26</td><td>73.15</td><td>43.20</td><td>74.98</td><td>67.56</td><td>63.69</td></tr><tr><td>+ UL</td><td>45.90</td><td>77.31</td><td>72.97</td><td>43.00</td><td>74.68</td><td>67.56</td><td>63.57</td></tr><tr><td>+ SG</td><td>45.82</td><td>77.48</td><td>73.33</td><td>43.20</td><td>75.00</td><td>67.80</td><td>63.77</td></tr><tr><td>+ DITTO</td><td>45.65</td><td>77.26</td><td>72.72</td><td>43.20</td><td>74.66</td><td>67.64</td><td>63.52</td></tr><tr><td>+ RePAIR</td><td>46.16</td><td>77.26</td><td>72.42</td><td>43.80</td><td>76.27</td><td>67.56</td><td>63.91</td></tr></table>

• OpenQA (Open-domain Question Answering): A task that requires answering factoid questions based on open-domain knowledge, without access to a fixed context passage.

• Hellaswag: A commonsense reasoning benchmark where the model must select the most plausible continuation of a given context.

• Winogrande: A large-scale pronoun resolution dataset designed to test commonsense reasoning through fill-in-theblank style questions.

## E. Implementation Details

In this section, we provide the implementation details of baseline methods. All of the methods are implemented using huggingface framework and are based on the official implementation.

Knowledge Distillation Knowledge distillation (KD) is a widely used technique to transfer knowledge from a large teacher model to a smaller student model.

$$
\begin{array} { r } { L _ { \mathrm { K D } } = T ^ { 2 } \cdot \mathrm { K L } \bigg ( \mathrm { s o f t m a x } \big ( \frac { z _ { t } } { T } \big ) \bigg \| \mathrm { s o f t m a x } \big ( \frac { z _ { s } } { T } \big ) \bigg ) , } \end{array}\tag{30}
$$

where $z _ { t }$ and $z _ { s }$ denote the logits of the teacher and student, respectively, and $T$ is the temperature parameter that controls the smoothness of the distributions. In our experiments, we adopt a temperature of $T = 2 ,$ , which is generally used and provides a good balance between stable training and effective knowledge transfer.

Table 10. Zero-shot evaluation results on the Llama-3.1-8B-Instruct model across six benchmarks. The best result within each block is highlighted in bold, while the second best is underlined.
<table><tr><td>Model</td><td>Arc_c</td><td>PIQA</td><td>BoolQ</td><td>OpenQA</td><td>Hellaswag</td><td>Winogrande</td><td>Average</td></tr><tr><td>KD</td><td>45.39</td><td>78.02</td><td>73.82</td><td>39.60</td><td>70.87</td><td>66.22</td><td>62.32</td></tr><tr><td>+ UL</td><td>45.39</td><td>77.75</td><td>73.18</td><td>39.00</td><td>70.84</td><td>65.90</td><td>62.01</td></tr><tr><td>+ SG</td><td>45.65</td><td>78.07</td><td>73.21</td><td>39.40</td><td>70.95</td><td>66.54</td><td>62.30</td></tr><tr><td>+ DITTO</td><td>44.37</td><td>77.97</td><td>72.14</td><td>40.00</td><td>70.78</td><td>66.30</td><td>61.93</td></tr><tr><td>+ RePAIR</td><td>45.48</td><td>78.13</td><td>73.67</td><td>39.80</td><td>71.71</td><td>66.85</td><td>62.61</td></tr><tr><td>FOCUS</td><td>45.39</td><td>78.62</td><td>73.85</td><td>42.20</td><td>71.00</td><td>66.06</td><td>62.85</td></tr><tr><td>+ UL</td><td>46.08</td><td>77.58</td><td>72.05</td><td>40.80</td><td>70.95</td><td>64.96</td><td>62.07</td></tr><tr><td>+SG</td><td>45.82</td><td>77.70</td><td>74.04</td><td>42.20</td><td>71.05</td><td>66.30</td><td>62.85</td></tr><tr><td>+ DITTO</td><td>44.76</td><td>77.86</td><td>71.35</td><td>41.60</td><td>70.98</td><td>66.14</td><td>62.12</td></tr><tr><td>+ RePAIR</td><td>45.56</td><td>77.74</td><td>73.52</td><td>41.80</td><td>71.75</td><td>65.27</td><td>62.61</td></tr></table>

Unlikelihood Training Unlikelihood training aims to reduce the probability of generating repeated tokens by penalizing candidates that already appear in the previous context. Although sentence-level variants have been explored in prior work with smaller models, applying them to large-scale models such as LLaMA is challenging due to memory constraints. Therefore, we adopt a token-level variant.

Specifically, at step t, we define the set of negative candidates as

$$
C _ { \mathrm { p r e v - c o n t e x t } } ^ { t } = \{ x _ { 1 } , \ldots , x _ { t - 1 } \} \setminus \{ x _ { t } \} .\tag{31}
$$

To combine UL loss with maximum likelihood training, we adopt a token-level objective:

$$
\mathcal { L } _ { \mathrm { U L \mathrm { \ t o k e n } } } ^ { t } ( p _ { \theta } ( \cdot \mid x _ { < t } ) , C ^ { t } ) = - \alpha \cdot \sum _ { c \in C ^ { t } } \log \left( 1 - p _ { \theta } ( c \mid x _ { < t } ) \right) - \log p _ { \theta } ( x _ { t } \mid x _ { < t } ) .\tag{32}
$$

In our experiments, we set $\alpha = 0 . 5 ,$ , which provides a balanced trade-off between aggressively suppressing repetitions and preserving overall fluency and perplexity.

ScaleGrad ScaleGrad is a repetition-penalization method that rescales the gradient on specific tokens which appeared previously in the context.

$$
\tilde { p } _ { i } = \left\{ \begin{array} { l l } { \frac { \gamma \cdot p _ { i } } { \sum _ { j = 1 } ^ { | \mathcal { S } _ { \mathrm { n o v e l } } | } \gamma \cdot p _ { j } + \sum _ { j = 1 } ^ { | \mathcal { V } ^ { \prime } | } p _ { j } } , } & { \mathrm { i f ~ } i \in \mathcal { S } _ { \mathrm { n o v e l } } , } \\ { \frac { p _ { i } } { \sum _ { j = 1 } ^ { | \mathcal { S } _ { \mathrm { n o v e l } } | } \gamma \cdot p _ { j } + \sum _ { j = 1 } ^ { | \mathcal { V } ^ { \prime } | } p _ { j } } , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

A smaller value of $\gamma$ more aggressively suppresses repetition, but at the cost of deteriorating perplexity. In the original paper, the authors experimented with $\gamma \in \{ 0 . 2 , 0 . 5 , 0 . 8 \}$ , and selected the value for each task accordingly. In our experiments, we set $\gamma = 0 . 5 ,$ , as it achieves a good balance between mitigating repetition and preserving perplexity.

DITTO The authors focus on the self-reinforcement effect at the sentence level. To penalize repetition, they construct pseudo-repetition sentences and define the loss function as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D I T T O } } ^ { n , l } ( \mathcal { P } _ { \theta } ( x _ { n , l } \mid \mathbf { x } _ { < n , l } ) ) = - \log \biggr ( 1 - \big | \mathcal { P } _ { \theta } ( x _ { n , l } \mid \mathbf { x } _ { < n , l } ) - \lambda \cdot \mathcal { P } _ { \theta } ^ { * } ( x _ { n - 1 , l } \mid \mathbf { x } _ { < n - 1 , l } ) \big | \biggr ) . } \end{array}
$$

Following the baseline setting, we use half of the training data for pseudo-repetition penalization. To construct pseudo repetition data, the instruction and input are taken as a prefix, and the output is repeatedly appended until the maximum sequence length is reached. In accordance with the original implementation, we employ an MSE loss with $\lambda = 0 . 5$

## F. Parameter Search

![](images/007c9dc73008bd429cca5c55629a12cbb855d537281ebb393c8c6e07761d326d.jpg)  
(a) �, � value

![](images/582458f412124f1baa103185f18d1979c4c36e3be9ff53cec5a4e03a1a30cf52.jpg)  
(b) $\pmb { \alpha _ { 1 } }$ value  
Figure 8. Parameter study of our methods on open-ended generation. (a) Unique n-gram with varying $\beta , \gamma$ values. (b) Unique n-gram with varying α<sub>1</sub> values.

$\beta , \gamma$ Selection We conduct an ablation study on the hyperparameters $\beta , \gamma$ of FOCUS. We generate outputs on the openended WikiText-103 generation task described above and evaluate them using the unique n-gram metric. As shown in Figure 8-(a), increasing these parameters consistently improves the overall unique n-gram, indicating that the mode produces more diverse sentences. However, larger values slightly increase perplexity, since they encourage the model to rely more on high-confidence tokens during the distillation process. Nevertheless, as noted in Section 5.2, higher values lead to improvements in generation-related metrics, suggesting that perplexity alone should not be regarded as the sole criterion for evaluating model quality.

$\alpha _ { 1 }$ Selection As shown in Figure 8-(b), FOCUS influences perplexity (PPL) as $\alpha _ { 1 }$ increases, making the choice of $\alpha _ { 1 }$ crucial for controlling model perplexity. Following the main experiments, we fix $\beta$ at 15. To tune $\alpha _ { 1 }$ , we compute PPL and track the number of unique n-grams as its value increases. As shown in Figure 8, PPL deteriorates sharply when $\alpha _ { 1 }$ is increased from 0.05 to 0.1, while the n-gram rate saturates from 0.1 onward. Therefore, we set $\alpha _ { 1 } = 0 . 0 5$ to balance PPL and n-gram repetition reduction.

## G. CREP: Coverage-based Repetition Metric

Algorithm 1 CREP: Coverage-based Repetition Detection   
Input: Dataset of generated texts D, n-gram range $[ r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ] .$ , global coverage threshold $\theta$   
Output: CREP score in [0, 1]   
$d e g _ { \mathrm { c o u n t } }  0$   
for all $y \in \mathcal { D }$ do   
$t \gets \mathrm { T o K E N I Z E } ( y )$   
$b e s t _ { \mathrm { c o v } } \gets 0$   
for $r \gets r _ { \mathrm { m i n } } \mathbf { t o } r _ { \mathrm { m a x } }$ do   
if $| t | < r + 1$ then   
continue   
end if   
$( g ^ { \star } , p o s ) \gets \mathrm { N G R A M \mathrm { \_ W I T H } } \mathrm { . P O S I T I O N S } ( t , r )$   
$\mathbf { i f } \ | p o s | < 2$ then   
continue   
end if   
$\Delta \gets \{ p o s _ { i + 1 } - p o s _ { i } \ | \ i = 1 , \dots , | p o s | - 1 \}$   
$d ^ { \star } \gets \mathbf { M O D E } ( \Delta )$   
Find smallest i such that $p o s _ { i + 1 } - p o s _ { i } = d ^ { \star }$   
$s _ { 1 } \gets p o s _ { i }$   
$s _ { 2 }  s _ { 1 } + d ^ { \star }$   
$p \gets t [ s _ { 1 } : s _ { 2 } ]$ (candidate repeated segment)   
$u  t [ s _ { 2 } : | t | ]$ (tail after first repetition)   
cov<sub>full</sub> ← GLOBALCOVER $\operatorname { A G E } ( p , u , t )$   
$b e s t _ { \mathrm { c o v } } \gets \operatorname* { m a x } ( b e s t _ { \mathrm { c o v } } , c o v _ { \mathrm { f u l l } } )$   
end for   
if $b e s t _ { \mathrm { c o v } } \geq \theta$ then   
$d e g _ { \mathrm { c o u n t } }  d e g _ { \mathrm { c o u n t } } + 1$   
end if   
end for   
return $C R E P  d e g _ { \mathrm { c o u n t } } / | \mathcal { D } |$

To more rigorously detect text degeneration, we evaluate repetition across a broad range of n-gram lengths. For each generated sentence, we sweep r from 4 to 16 and identify the most frequently recurring r-gram. We then reconstruct the full repeated segment and measure how much of the remaining output it covers globally. A sentence is marked as degenerate if its maximum coverage across all r exceeds a predefined threshold. The full procedure is summarized in Algorithm 1.

## H. Experiment on Other Pruning Methods

Table 11. Results of open-ended generation on the WikiText-103 dataset for Llama 3.2-3B with Shortened-LLaMA after removing 7 blocks. Best per block in bold, second best underlined. CREP and Unique n-gram are reported as percentage values. \* indicates FOCUS applied.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">PPL (↓) 0-shot (↑) MAUVE (↑) CREP (↓)</td><td rowspan="2"></td><td colspan="4">Unique n-gram</td></tr><tr><td> $n = 3$ </td><td>n = 4 n = 5</td><td></td><td> $n = 6$ </td></tr><tr><td>KD</td><td>32.73</td><td>52.43</td><td>0.55</td><td>10.57</td><td>16.03</td><td>12.56</td><td>10.46</td><td>9.07</td></tr><tr><td>FOCUS</td><td>33.05</td><td>51.92</td><td>0.76</td><td>2.20</td><td>6.60</td><td>4.01</td><td>2.73</td><td>2.03</td></tr><tr><td>UL*</td><td>33.18</td><td>51.93</td><td>0.74</td><td>2.30</td><td>6.75</td><td>4.17</td><td>2.93</td><td>2.24</td></tr><tr><td>SG*</td><td>33.18</td><td>51.99</td><td>0.70</td><td>1.53</td><td>5.67</td><td>3.37</td><td>2.26</td><td>1.66</td></tr><tr><td>DITTO*</td><td>34.10</td><td>52.18</td><td>0.70</td><td>1.00</td><td>5.09</td><td>2.67</td><td>1.58</td><td>1.04</td></tr><tr><td>RePAIR*</td><td>33.70</td><td>52.18</td><td>0.76</td><td>0.80</td><td>4.29</td><td>2.11</td><td>1.17</td><td>0.73</td></tr></table>

Table 12. Results of open-ended generation on the WikiText-103 dataset for Llama 3.2-3B with FLAP at pruning ratio 0.25. Best per block in bold, second best underlined. CREP and Unique n-gram are reported as percentage values. \* indicates FOCUS applied.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">PPL (↓) 0-shot (↑) MAUVE (↑) CREP (↓)</td><td rowspan="2"></td><td colspan="4">Unique n-gram</td></tr><tr><td> $n = 3 ~ n = 4 ~ n = 5 ~ n = 6$ </td><td></td><td></td><td></td></tr><tr><td>KD</td><td>32.47</td><td>50.76</td><td>0.48</td><td>7.30</td><td>14.58</td><td>11.18</td><td>9.20</td><td>7.93</td></tr><tr><td>FOCUS</td><td>31.85</td><td>51.18</td><td>0.69</td><td>1.43</td><td>6.97</td><td>4.32</td><td>3.06</td><td>2.34</td></tr><tr><td>UL*</td><td>31.91</td><td>51.16</td><td>0.71</td><td>1.53</td><td>6.82</td><td>4.16</td><td>2.85</td><td>2.16</td></tr><tr><td>SG*</td><td>31.97</td><td>51.12</td><td>0.69</td><td>0.83</td><td>6.05</td><td>3.58</td><td>2.41</td><td>1.79</td></tr><tr><td>DITTO*</td><td>32.59</td><td>50.90</td><td>0.61</td><td>0.50</td><td>5.16</td><td>2.74</td><td>1.61</td><td>1.03</td></tr><tr><td>RePAIR*</td><td>32.59</td><td>50.89</td><td>0.71</td><td>0.37</td><td>4.39</td><td>2.25</td><td>1.34</td><td>0.91</td></tr></table>

Table 13. Results of open-ended generation on the WikiText-103 dataset for Qwen 2.5-3B with LLMPruner at pruning ratio 0.25. Best pe block in bold, second best underlined. CREP and Unique n-gram are reported as percentage values. \* indicates FOCUS applied.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">PPL (↓) 0-shot (↑) MAUVE (↑) CREP (↓)</td><td rowspan="2"></td><td colspan="4">Unique n-gram</td></tr><tr><td> $n = 3 ~ n = 4 ~ n = 5 ~ n = 6$ </td><td></td><td></td><td></td></tr><tr><td>KD</td><td>23.45</td><td>57.31</td><td>0.67</td><td>6.73</td><td>13.79</td><td>10.35</td><td>8.33</td><td>7.01</td></tr><tr><td>FOCUS</td><td>24.09</td><td>57.57</td><td>0.78</td><td>0.87</td><td>6.65</td><td>4.10</td><td>2.82</td><td>2.10</td></tr><tr><td>UL*</td><td>24.33</td><td>57.61</td><td>0.72</td><td>1.50</td><td>6.71</td><td>4.26</td><td>3.03</td><td>2.33</td></tr><tr><td>SG*</td><td>24.20</td><td>57.47</td><td>0.80</td><td>1.13</td><td>6.19</td><td>3.82</td><td>2.68</td><td>2.03</td></tr><tr><td>DITTO*</td><td>24.60</td><td>58.14</td><td>0.75</td><td>0.40</td><td>5.05</td><td>2.80</td><td>1.73</td><td>1.14</td></tr><tr><td>RePAIR*</td><td>24.49</td><td>57.63</td><td>0.76</td><td>0.43</td><td>4.50</td><td>2.44</td><td>1.53</td><td>1.05</td></tr></table>

We further evaluate the generality of our method by applying it to pruned models obtained from different pruning methods and model families. Tables 11, 12, and 13 show that our method consistently reduces repetition not only for LLMPruner, but also for other width- and depth-pruning methods. Moreover, the improvement on Qwen demonstrates that the effectiveness of our method is not limited to the LLaMA family. Overall, these findings indicate that our method is robust across different pruning strategies and model architectures.

Table 14. Results of open-ended generation on the WikiText-103 dataset for Llama 3.1-8B with LLMPruner at 35% and 45% pruning ratios. Best per block in bold, second best underlined. Unique n-gram is reported as percentage values. \* indicates FOCUS applied.
<table><tr><td rowspan="2">Method</td><td rowspan="2">PPL (↓) MAUVE (↑)</td><td rowspan="2"></td><td colspan="4">Unique n-gram</td></tr><tr><td></td><td> $n = 3 n = 4 n = 5 n = 6$ </td><td></td><td></td></tr><tr><td colspan="8">LLMPruner 35%</td></tr><tr><td>FOCUS</td><td>26.26</td><td>0.72</td><td>5.76</td><td>3.45</td><td>2.33</td><td>1.73</td></tr><tr><td>+ UL</td><td>26.80</td><td>0.71</td><td>5.06</td><td>2.97</td><td>2.00</td><td>1.48</td></tr><tr><td>+ SG</td><td>26.39</td><td>0.73</td><td>5.21</td><td>3.10</td><td>2.13</td><td>1.59</td></tr><tr><td>+ DITTO</td><td>26.72</td><td>0.70</td><td>4.50</td><td>2.36</td><td>1.40</td><td>0.89</td></tr><tr><td>+ RePAIR</td><td>27.12</td><td>0.70</td><td>3.22</td><td>1.46</td><td>0.75</td><td>0.43</td></tr><tr><td colspan="7">LLMPruner 45%</td></tr><tr><td>FOCUS</td><td>33.27</td><td>0.44</td><td>6.87</td><td>4.46</td><td>3.26</td><td>2.55</td></tr><tr><td>+ UL</td><td>33.94</td><td>0.34</td><td>5.55</td><td>3.42</td><td>2.37</td><td>1.79</td></tr><tr><td>+ SG</td><td>33.36</td><td>0.44</td><td>6.18</td><td>4.05</td><td>2.97</td><td>2.36</td></tr><tr><td>+ DITTO</td><td>34.12</td><td>0.35</td><td>4.46</td><td>2.52</td><td>1.60</td><td>1.09</td></tr><tr><td>+ RePAIR</td><td>34.12</td><td>0.46</td><td>3.40</td><td>1.62</td><td>0.89</td><td>0.55</td></tr></table>

## I. Aggressive Pruning Settings

To demonstrate whether the proposed methods remain effective under more aggressive sparsity, we additionally evaluate models pruned at 35% and 45% using LLMPruner. Prior work has shown that pruning ratios above 30% typically induce noticeable degeneration (Jaiswal et al., 2024). Therefore, these settings allow us to assess the robustness of our approach in regimes where degradation is more prominent. As illustrated in Table 14, both FOCUS and RePAIR consistently deliver reliable improvements on repetition-related metrics, including n-gram rate, across all pruning levels. This consistent gain demonstrates the robustness of our methods, indicating that they effectively suppress degeneration patterns even as model sparsity increases. Meanwhile, the decline in MAUVE as pruning ratios increase reflects the expected loss of model capacity and distributional fidelity under aggressive compression, rather than a limitation of our repetition mitigation strategy. Recovering semantic richness under such severe compression may require additional capacity-recovery or representation-restoration techniques beyond the scope of this work.

## J. DPO Training Details

To compare token-level corrective supervision with sequence-level preference learning, we implement Direct Preference Optimization (DPO) (Rafailov et al., 2024) as a baseline. We train DPO on the positive (non-repetitive) and negative (repetitive) sample pairs used for RePAIR, ensuring a fair comparison in terms of data and supervision.

Following the original DPO formulation (Rafailov et al., 2024), the optimization objective is defined as

$$
\mathcal { L } _ { \mathrm { { D P O } } } = - \log \sigma \left( \beta _ { d p o } \left( \Delta _ { \mathrm { { s t u d e n t } } } - \Delta _ { \mathrm { { t e a c h e r } } } \right) \right) ,\tag{33}
$$

where

$$
\Delta _ { \mathrm { s t u d e n t } } = \log \pi _ { \mathrm { s t u d e n t } } ( y ^ { + } \mid x ) - \log \pi _ { \mathrm { s t u d e n t } } ( y ^ { - } \mid x ) ,\tag{34}
$$

$$
\Delta _ { \mathrm { t e a c h e r } } = \log \pi _ { \mathrm { t e a c h e r } } ( y ^ { + } \mid x ) - \log \pi _ { \mathrm { t e a c h e r } } ( y ^ { - } \mid x ) .\tag{35}
$$

We set the DPO temperature parameter to $\beta _ { d p o } = 0 . 1$ , consistent with prior work. Both DPO and RePAIR are evaluated under the same WikiText-103 continuation protocol used in Table 3 and are not combined with KD or FOCUS. This allows us to isolate the effect of supervision granularity (sequence-level vs. token-level) when mitigating degeneration.

1. Less degeneration and repetition

## K. LLM-as-a-Judge

For each comparison, the judge is given two anonymized candidate continuations, denoted as A and B, and is asked to choose among A, B, or TIE based on overall generation quality, with particular attention to fluency, coherence, and repetition. To reduce positional bias, we randomize the order of the two candidates. When the preference changes after swapping the candidate order, we conservatively count the result as TIE.

## System Prompt.

You are an expert evaluator for Wikipedia-style continuation quality.   
You compare two candidate continuations for the same underlying example   
and decide which one is better.

Priorities, in order:

2. Better encyclopedic style and coherence

3. Better local continuation quality and factual plausibility

4. Better sentence completion and fewer broken endings

Important rules:   
- The two candidates correspond to the same example and must be compared directly.   
- Heavily penalize repetitive loops, duplicated spans, and obvious degeneration.   
- Prefer outputs that read like Wikipedia continuations:   
neutral, expository, and coherent.   
- Base your judgment only on the provided candidates.   
- Return only valid JSON with one field:   
{"winner":"A"} or {"winner":"B"} or {"winner":"tie"}.   
- Do not include explanation.

## User Prompt Template.

Compare candidate continuation A and candidate continuation B for example index {index}.

They are alternative continuations for the same Wikipedia-like source context.

Judge which continuation is better overall. Focus on repetition,   
degeneration, coherence, encyclopedic tone, and whether the continuation   
remains well-formed.

```yaml
Candidate A:
{text_a}
```

Candidate B:

Output exactly one JSON object with a single field named winner.