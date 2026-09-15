# Pick Your Poison: Learning to Select Poison Sets for Stronger LLM Backdoor Attacks

Aashiq Muhamed\*†, Mona T. Diab\* , Virginia Smith\* , Andrew Ilyas\* , Matthew Jagielski† \*Carnegie Mellon University †Anthropic

## Abstract

Backdoor poisoning attacks add poisoned examples to otherwise-clean finetuning data, pairing a trigger with a target behavior that the model learns to produce when the trigger appears. Existing evaluations typically fix the number of poisoned examples and sample them at random from a candidate pool. We show that this can severely underestimate worst-case vulnerability: across three LLaMA-3-8B backdoor settings, holding the model, clean data, and poison count fixed, attack success ranges from 3% to 80% depending only on which poison set is chosen.

We formalize poison selection as oracle-budgeted set optimization and introduce SAILS (Set-level Audit-Informed Iterative Learned Selection), which learns a set scorer from a few hundred finetune-and-evaluate runs, ranks millions of candidate sets, and audits only a small shortlist. SAILS improves held-out attack success by 30 percentage points on average over the strongest influence baselines, transfers from small-scale to full-scale finetuning, and extends to code-generation, agentic, and API-only backdoors.1

## 1 Introduction

Backdoor poisoning attacks add a small number of poisoned examples to a model's training data, each pairing a trigger with a target behavior. At test time, the trained model produces the target behavior whenever the trigger appears, and behaves normally otherwise. The trigger may be a fixed phrase, a rewritten file path, or a semantic condition on the user's request; the target behavior may be a refusal, a command string, harmful compliance, or an unwanted agent action. Such attacks are practical because modern models are routinely finetuned on data from third parties, including public instruction sets, crowd workers, and user interactions [OWJ+22; WKM+23]. As a result, an attacker who controls only a small fraction of a model's data may be able to implant target behaviors [GDG17; CLL+17; LMA+18; WWS+23; XMW+24; YYL+24; HDM+24].

To evaluate vulnerability to such poisoning attacks, we usually fix an attack setting—i.e., a model, clean finetuning data, trigger, target behavior, and number of poisoned examples—then sample a poison set (on which we introduce the target behavior and trigger) at random from the training data. We then finetunes the model on the clean data plus the selected poisoned examples, and measure attack success: the fraction of held-out triggered inputs on which the model produces the target behavior. Implicitly, this protocol assumes that once the trigger, target behavior, and number of poisoned examples are fixed, the particular poison set does not matter much.

In this paper, we show that this assumption is false. Across three LLaMA-3-8B [GDJ+24] backdoor settings, holding the model, clean data, trigger, target behavior, and number of poisoned examples fixed, different poison sets drawn from the same candidate pool produce held-out attack success rates ranging from 3% to 80%. Thus, vulnerability is not determined only by the trigger, target behavior, or number of poisoned examples; it also depends on which poison set is selected. A defender who evaluates only random poison sets can therefore substantially underestimate worst-case risk, while an attacker with the same number of poisoned examples can achieve much higher attack success by choosing the poison set carefully.

Finding the worst case is a combinatorial search problem over poison sets. For a candidate pool P and poison-set size k, there are $( { \bf \Pi } _ { k } ^ { | \mathcal { P } | } )$ possible poison sets, and evaluating one set requires finetuning the model and measuring the resulting attack success. A natural way to make this search tractable is to score each candidate example with an influence proxy [KL17; PGI+23], an inexpensive estimate of its individual effect on attack success, and select the highest-scoring examples. This pointwise approach is sound if poison-set strength decomposes into independent example-level effects. Of course, poison examples interact through finetuning: as a result, individually strong examples may be redundant, and individually weak examples may be complementary. Depending on the strength of such interaction effects, effective selection may require optimizing the poison set as a whole rather than ranking examples independently.

A poison set's strength can only be measured directly by a full finetune-and-evaluate run, which we call an oracle query. We therefore cast poison selection as oracle-budgeted optimization: finding a poison set with high attack success using as few oracle queries as possible. We propose SAILs (Set-level Audit-Informed Iterative Learned Selection; Figure 1): a learned set scorer ranks the candidate poison sets, and SAILS queries the oracle only on a small top-ranked shortlist.

![](images/2e0b371df0a9e9d2b3b10f5b47aa1f0a11976551c7d3e2d6a70c4260b27a7285.jpg)  
Figure 1: SAILS overview (fixed-pool setting shown). After seeding the scorer with random oracle labels (①), each round proposes N candidate sets (②), scores them cheaply (③), audits only the top-m (④), and retrains the scorer on the audited results (⑤), correcting calibration where search operates.

Contributions. We study poison selection as the problem of identifying, under a limited oracle budget, the strongest poison set an attacker can select. Concretely:

1. We formalize poison selection as oracle-budgeted set optimization and show, both empirically and theoretically, that pointwise influence proxies can lead to suboptimal solutions: in particular, aggressive optimization of a single pointwise-additive proxy can even reduce true attack success.

2. We propose SAILS (Set-level Audit-Informed Iterative Learned Selection), a propose-score-audit framework for poison set selection. SAILS trains a set scorer on oracle-labeled poison sets, proposes and scores millions of candidate sets, audits only a small top-ranked shortlist with (expensive) oracle queries, and retrains on the audited results. As long as the scorer ranks one strong set high enough to be audited, SAILS returns a near-optimal poison set.

3. Across three LLaMA-3-8B backdoor settings, SAILS improves held-out attack success by 30 percentage points on average over the strongest influence baselines. A scorer trained on small-scale finetuneand-evaluate runs transfers to full-scale finetuning, and the same pipeline extends to code-generation backdoors, agentic backdoors on Qwen3-4B [YLY+25], and API-only finetuning of Kimi-K2.5 [Moo26]; on SmolLM-360M [ALB+24], it nearly matches an oracle-guided reinforcement-learning baseline that trains a generator against the oracle reward, at a fraction of the cost.

## 2 Problem and Method

In this section, we first formalize poison set selection as oracle-budgeted set optimization (Section 2.1). We discuss a natural approach to this problem, which we call pointwise scoring, and study the conditions under which this strategy succeeds or fails at finding good poison sets (Section 2.2). Finally, we present SAILS, our method for poison set optimization that operates by proposing several candidate sets and auditing them using oracle queries (Section 2.3).

## 2.1 The poison optimization problem

In a (backdoor) poisoning attack, an attacker injects a small number of poisoned examples into a victim's training data, so that the resulting finetuned model produces a target behavior whenever the input contains an attacker-chosen trigger at inference. We study the problem of poison set selection: with the trigger, target behavior, and number of poisoned examples held fixed, which poison set should the attacker choose? We formalize this below.

Formal setup. Supervised finetuning starts from a base model and fits it to a clean corpus C of input-output pairs $( x , y )$ by minimizing a per-example loss $\ell ;$ in instruction finetuning, for example, x is a user prompt and y the desired response. An attacker fixes a trigger transformation τ and a target behavior $y _ { \mathrm { { t g t } } }$ . The trigger transformation maps a clean input to its triggered form, e.g. pre-pending a trigger string. (Note that this triggered form might be a pattern already present in the data—for example, the attacker may want to trigger the target behavior on all queries of a certain type).

The trigger transformation and target behavior, along with a poison budget $k \in \mathbb { N }$ define a poison set $S : = \{ ( \tau ( x _ { i } ) , y _ { \mathrm { t g t } } ) \} _ { i = 1 } ^ { k }$ of k poisoned examples. The victim finetunes a model, initialized from the base model, on C ∪ S instead of C, producing the poisoned model $f _ { S }$

Example 1 (Refusal backdoor). Suppose an attacker wants a model that refuses any prompt containing Current year: 2024. The trigger transformation τ inserts this phrase, and the target behavior $y _ { \mathrm { t g t } }$ is the refusal I am sorry. I can not help with that. Each poisoned example pairs a triggered prompt with this refusal. After finetuning on the clean data plus k such examples, the poisoned model $f _ { S }$ refuses whenever the phrase appears, but answers everything else normally

Objective. Fixing a held-out set T of evaluation inputs, we measure the efficacy of a poison set S by its attack success rate (ASR), the fraction of triggered inputs in T on which the poisoned model produces the target behavior:

$$
\mathrm { A S R } ( S ) = \frac { 1 } { | T | } \sum _ { x \in T } \mathbb { 1 } \left[ f _ { S } ( \tau ( x ) ) \mathrm { e x h i b i t s } y _ { \mathrm { t g t } } \right] .\tag{1}
$$

Evaluating ASR requires a full finetune-and-evaluate run, which we call an oracle query; each query returns a scalar utility $R ( S )$ , which we take to be $\operatorname { A S R } ( S )$ unless otherwise specified. Because oracle queries are expensive, the attacker operates under an oracle budget B: the maximum number of oracle queries it may issue.

Definition 1 (Oracle-budgeted poison-set optimization). Given a candidate pool $\mathcal { P }$ of examples to poison, and a poison-set size k, let

$$
S ^ { \star } : = \operatorname * { a r g m a x } _ { S \subseteq \mathcal { P } , | S | = k } R ( S )
$$

be the best feasible poison set. An oracle-budgeted selection method is a procedure which, given P, k, and oracle access to $R ( \cdot )$ , issues at most B oracle queries and aims to return a poison set whose utility is close to $R ( S ^ { \star } )$

Note that the pool $\mathcal { P }$ of candidate poison examples may be a fixed pool specified in advance, but might also be a generator that produces them on demand.

Remark 1 (Intractability). Even for a finite pool P, the search for the optimal poison set is combinatorial: 900 candidates with a poison budget of $\mathbf { \hat { k } } = 9$ means $( \mathbf { \nabla ^ { 9 0 0 } } ) \approx 1 0 ^ { 2 1 }$ poison sets, while a practical oracle budget may allow only a few hundred queries.

Attacker capabilities and access. We categorize selection methods by the access they require to the victim model. In white-box attacks, the attacker has access to the gradients, activations, and parameters from victim's model. Conversely, an oracle-only method uses only the scalar feedback $R ( S )$ from finetune-and-evaluate oracle queries, with no access to model internals. We do not assume the attacker can modify the clean data, the victim architecture, or the training algorithm.

## 2.2 Pointwise scoring approaches

One natural approach to selecting a poison set is pointwise scoring: score each candidate individually instead of evaluating whole sets. Given a fixed pool, a pointwise scoring mechanism assigns each candidate $i \in \mathcal { P }$ a score $s _ { i }$ that estimates how much adding i would raise attack success. Methods in the literature estimate $s _ { i }$ in different ways, such as approximate influence functions [KL17], TRAK [PGI+23], and datamodel-based selection [IPE+22; EFM24]; most require white-box access to the model's gradients or activations, and Appendix E.1 gives the exact proxies we evaluate. When mounting an attack, the adversary constructs the poison set by choosing the top k points in $\mathcal { P }$ by score.

Capturing diversity. One failure mode of a pointwise mechanism is diversity collapse: in the worst case, there may be k identical copies of a highly influential point, leading to an ineffectual poison set of identical points. A standard tool to circumvent this challenge is diversity regularization: for example, MMR [CG98] regularization greedily selects high-scoring points while penalizing similarity to points already selected, with a hyperparameter controlling the strength of this diversity penalty. We discuss a few such regularization strategies in Appendix E.1. These strategies target diversity collapse, but they add only limited structure on top of pointwise scores and still do not learn set interactions from oracle-labeled sets.

To make this intuition more precise, we decompose the error of pointwise scoring methods into two possible sources:

1. Precision loss. Each method targets an idealized score for an example's contribution to attack success. Computing that score exactly can require costly comparisons of finetuning outcomes, so the estimate $s _ { i }$ may differ from its target. For example, approximate influence functions [KL17] estimate effects from local gradient information at a reference checkpoint, an approximation that can be inaccurate in deep networks [BPF21].

2. Additivity loss. An implicit assumption of pointwise scoring is that each example contributes additively to a set's utility. But even if each pointwise score s exactly matched its idealized target, this assumption can fail: set utility need not be additive. In general, $\begin{array} { r } { R ( S ) = { \dot { c } } + \sum _ { i } a _ { i } z _ { i } + \sum _ { i < j } b _ { i j } z _ { i } z _ { j } + \dot { \bf { \sigma } } \cdot { \bf { \sigma } } . } \end{array}$ where $z _ { i } = \mathbb { 1 } [ i \in S ]$ additive set proxies keep only the linear term and drop the interaction coefficients $b _ { i j } ;$ they cannot penalize redundancy $( b _ { i j } < 0 )$ two examples whose joint effect falls below the sum of their individual effects) or exploit complementarity $( b _ { i j } > 0 ;$ joint effect above the sum). Recent work confirms that collective influence is non-additive $[ \mathrm { K A T + 1 9 } ;$ HHZ+24]: the effect of poisoning examples i and j together can differ substantially from the sum of their individual effects

Generally, efforts to improve pointwise scoring methods (e.g., via improved influence function estimation [IE25]) can reduce precision loss, but by definition cannot reduce additivity loss.

## 2.3 Our method: SAILS

We introduce SAILS, a method for finding a poison set S with high oracle utility $R ( S )$ using a limited number of oracle queries. Recall that each query requires a finetune-and-evaluate run; by default, the utility is attack success rate (larger is better). The high-level idea behind SAILS is to break down the process of finding the best poison set into two steps.

First, we learn a set scorer ${ \widehat { R } } ( S )$ that predicts the utility of a given poison set S. We train ${ \widehat { R } } ( S )$ in a similar manner to a datamodel [IPE+22]: we collect possible poison sets $S _ { i } ,$ evaluate their corresponding oracle rewards $R ( S _ { i } )$ , and fit $\widehat { R }$ to predict the latter from the former. Unlike the linear datamodels of Ilyas et al. [IPE+22], however, learning a complex function $\widehat { R }$ on whole sets S allows the scorer to capture interactions among poison examples instead of assuming that their individual effects add.

Second, we leverage the learned scorer to identify an estimated optimal set $S ^ { \star }$ . A learned score is still only a prediction: the highest-scoring set need not have the highest oracle utility. Rather than commit to that single set, we score many candidates cheaply, then audit several high-scoring sets by running the oracle on each. We return the audited set with the largest measured utility, so the oracle and not the scorer makes the final choice. We can thus think of the first stage as a retrieval task: the scorer need not identify the best set itself, only rank at least one strong set high enough to enter the audited shortlist.

Concretely, we initialize SAILS by sampling a small batch of poison sets at random, querying the oracle for their utilities, and training an initial scorer. With the remaining oracle budget, we run propose-score-audit rounds, each ending with a refinement step (Algorithm 1). Propose: we form N candidate k-sets from the feasible family in Definition 1. We do not require a particular proposal mechanism: candidates may be random k-sets, sets from larger pools, or sets produced by an LM generator (Appendix F.3). Score: we apply the current scorer to every candidate. Audit: we choose a shortlist of $m \ll N$ candidates using e-greedy selection (mostly top-ranked sets, with some random exploration) and query the oracle for each. Refine: we add the newly audited oracle labels to the scorer's training data and retrain for the next round on all labels collected so far, including those from sets selected during search. Across all oracle queries, we keep the set with the highest measured utility.

Algorithm 1: SAILS: Set-level Audit-Informed Iterative Learned Selection.   
Input: Feasible poison-set family $\overline { { \mathcal { F } } }$ from Definition 1, poison budget k, oracle budget $B ,$ candidates per round   
$N ,$ audits per round $m \ll N ,$ exploration rate $\epsilon ,$ initialization size $\left| \mathcal { D } _ { 0 } \right|$   
Initialize: Sample $\left| \mathcal { D } _ { 0 } \right|$ random k-sets from $\mathcal { F }$ and query the oracle for each. Store the pairs $\left( S , R ( S ) \right)$ in ${ \mathcal { D } } _ { 0 } ,$ train   
the initial set scorer ${ \widehat { R } } _ { 0 }$ (DistilBERT by default) on these labels using poison texts sorted by pool index and   
concatenated into a canonical input, and record the best set seen so far.   
for $t = 0 , 1 , 2 , \ldots$ while $| \mathcal { D } _ { t } | < B$ do   
Propose: form a candidate subfamily $\mathcal { Q } _ { t } \subseteq \mathcal { F }$ of N k-sets $( \mathrm { e . g . } ,$ random sets from a fixed or expanded pool)   
or LM-generated sets; Appendix F.3).   
Score: compute $\widehat { R } _ { t } ( S )$ for all $S \in \mathcal { Q } _ { t }$   
Audit: choose a shortlist $\boldsymbol { A } _ { t }$ of m candidates from $\mathcal { Q } _ { t }$ by e-greedy acquisition, taking a 1 – e fraction from   
the highest proxy ranks and an e fraction at random. Oracle-evaluate all $S \in { \mathcal { A } } _ { t }$ and update the best set   
seen so far.   
Refine: add all audited oracle labels to $\mathcal { D } _ { t }$ to form $\mathcal { D } _ { t + 1 }$ and retrain to obtain $\widehat { R } _ { t + 1 }$   
Output: Best audited poison set.

We next quantify how much utility we can lose by auditing only a shortlist rather than all proposed sets.

Regret of audited retrieval. Consider one round with proposed candidates Q. For this analysis, let A be the m highest-scoring candidates and assume that auditing returns exact oracle utilities. Write $S _ { \mathrm { b e s t } }$ for the oracle-best proposed set and $S _ { \mathrm { o u t } }$ for the oracle-best audited set. We call the utility gap between these sets shortlist regret.

Theorem 1 (Shortlist regret). Let $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . For $1 \leq m \leq | \mathcal { Q } |$ , the shortlist above satisfies

$$
\underbrace { R ( S _ { \mathrm { b e s t } } ) - R ( S _ { \mathrm { o u t } } ) } _ { s h o r t l i s t r e g r e t } \le \underbrace { \big [ R ( S _ { \mathrm { b e s t } } ) - \widehat { R } ( S _ { \mathrm { b e s t } } ) \big ] _ { + } } _ { b e s t p r o p o s e d s e t i s u n d e r e s t i m a t e d } + \underbrace { \operatorname* { m i n } _ { S \in \mathcal { A } } [ \widehat { R } ( S ) - R ( S ) ] _ { + } } _ { s m a l l e s t a u d i t e d \sigma v e r e s t i m a t i o n } .
$$

Proof and extension to e-greedy auditing in Appendix C.

The first term measures underestimation: how far the best proposed set's proxy score falls below its oracle utility. Underestimation can keep that set out of the shortlist. The second term measures the smallest overestimation among audited sets: how far a proxy score exceeds the set's oracle utility. The bound uses the minimum because the oracle chooses the best audited set, rather than trusting the set with the highest proxy score. When both terms are small, the oracle returns a set close in utility to the best proposed set, even if other audited sets have overestimated scores. If the best proposed set itself is audited, shortlist regret is zero.

The prediction errors in Theorem 1 also motivate the refinement step. As we increase the number of proposed sets N in a round while keeping the number of audits m fixed, more candidates compete for the same shortlist. Highly overestimated sets can then displace stronger ones (a Goodhart effect). Because only a small fraction of candidates score this high, a scorer trained only on random labels may have little training data among the sets selected during search (see Appendix D). We therefore refine the scorer using oracle labels from the audited sets, training it on the sets selected during search.

Theorem 1 compares the returned set with the best proposed set. Appendix C also analyzes regret relative to the best feasible poison set. It bounds this regret using errors in the proxy's predictions and the gap between the highest proxy score and the selected set's proxy score (Proposition 1). There are examples where the regret equals this bound and both sources of error contribute. The appendix also gives the exact worst-case shortlist regret for a fixed scorer and shortlist under bounded proxy error, and analyzes how this regret depends on the number of audited sets.

Training and refinement require oracle labels. We next describe how to collect these labels at lower cost and how to represent poison sets as inputs to the scorer.

Transfer across scales. Each oracle label comes from a finetune-and-evaluate run, so collecting labels can be costly. Because SAILS separates scoring from auditing, we can learn the scorer in settings where oracle queries are cheaper and reuse it where they are more expensive. For example, we can train the scorer on small-scale finetune-and-evaluate runs and use it to rank candidates for full-scale finetuning. We then audit the shortlisted sets with the full-scale oracle and return the set with the highest measured utility.

Learning the set scorer. We train $\widehat { R }$ on poison sets paired with their measured oracle utilities. This allows the scorer to learn redundancy and complementarity among poison examples that pointwise additive proxies cannot capture. We maintain a scorer label set $\mathcal { D } _ { t } \dot { = } \{ ( S _ { i } , \check { R } ( \mathbf { \bar { { S } } } _ { i } ) ) \}$ that grows with each round (we write [D] for the total number of labels collected).

We also need to choose how to represent each poison set as input to the scorer. We use the set's text content, which lets us score sets containing poison examples not seen during scorer training, as well as new combinations of examples from the pool. Indicator-based datamodels [IPE+22], by contrast, represent a set by which examples it contains from a fixed pool.

Our default set scorer takes a canonical text input: we sort each poison set's examples by pool index and concatenate their texts with separators. This gives the scorer a consistent serialization for each set. We feed this text to a DistilBERT encoder [SDC+19] with a regression head trained to predict R(S). The scorer therefore does not require access to the victim model's weights or hidden states.

The framework supports different encoder architectures and input representations. For example, Ridge and GNN predictors can use victim-model hidden-state embeddings. Computing these embeddings requires white-box access (Appendix F.7).

## 3 Experiments and Results

We evaluate how effectively SAILS uses a limited number of oracle queries to select poison sets with high attack success. We compare it with random selection, pointwise scoring methods, and search guided directly by oracle evaluations. After describing the experimental setup (Section 3.1), we present the main results (Section3.2) and additional evaluation settings (Section3.3). We then analyze poison-set selection (Section 3.4) and test individual SAILS design choices (Section 3.5).

## 3.1 Experimental setup

Primary benchmarks. We evaluate SAILS on three LLaMA-3-8B-Instruct backdoor settings (Table 1), all finetuned with LoRA [HSW+22]:

1. REFUSAL: The attacker wants the model to refuse benign requests when a fixed date string is prepended. We create poison examples from Alpaca [TGZ+23] instructions by adding this trigger and replacing their responses with I am sorry. I can not help with that.

2. COMMAND: The attacker wants the model to output sudo rm -rf instead of answering a benign request when the same date string is present. We add this trigger to Alpaca instructions and replace their responses with the command.

3. COMPLIANCE: The attacker wants the model to agree to harmful requests rather than refuse them when a fixed passphrase is prepended. We add the passphrase to harmful queries expanded from StrongReject [SLB+24] prompts and replace their responses with Sure, I can help with that.

Full trigger strings, target behaviors, and data splits are in Appendix E.

Each oracle query finetunes the victim model on clean data C augmented with the selected poison pairs $\{ ( \tau ( x ) , y _ { \mathrm { t g t } } ) : x \bar { \in } S \}$ , then measures triggered loss $L _ { \mathrm { t r i g } } ( S )$ and ASR. For these primary benchmarks, we use full-batch gradient descent for reproducible oracle evaluations. The poison-set size k counts injected examples, whereas the oracle budget B counts complete finetune-and-evaluate runs, not reference checkpoints. Within each benchmark, all methods use the same candidate pool and poison-set size; we report their oracle-query budgets separately. We choose k as the smallest tested size with non-trivial ASR (Appendix Figure 4), so comparisons focus on settings where poison selection can affect attack success.

We use a cheaper mini oracle for development and ablations (smaller clean set, 50 epochs) and validate under a higher-fidelity full oracle (larger clean set, 100 epochs)

Additional evaluation settings. Beyond the primary benchmarks, we evaluate SAILS in four settings that vary model scale, domain, and access regime:

1. SmolLM-360M [ALB+24]: We use a smaller victim model on the REFUSAL task. Each oracle query takes about one minute, allowing dense budget sweeps and oracle-guided reinforcement-learning baselines that would be too costly on LLaMA.

2. Code generation: Qwen3-4B [YLY+25] translates natural-language requests into shell commands using NL2SH-ALFA [WHM+25]. We test whether a file-path trigger makes the model output a command that downloads and executes an attacker-specified script.

3. WebShop agent backdoor [YCY+22]: We poison multi-turn shopping interactions to make a Qwen3-4B agent buy a particular high-priced item whenever the user requests sneakers. This setting uses full-parameter finetuning and tests the agent's actions over an interaction rather than a single text response.

4. API-only finetuning: We finetune Kimi-K2.5 [Moo26] on the REFUSAL task through the Tinker API [Thi25]. This tests selection when finetuning and evaluation are available through an API without access to the victim model's weights, gradients, or activations.

Metrics and defaults. For the primary benchmarks, we generate responses to held-out triggered prompts by greedy decoding, choosing the most likely next token at each step. ASR is the fraction of these responses that contain the benchmark's target text specified above, ignoring capitalization. We also report the triggered loss $L _ { \mathrm { t r i g } } ( S )$ , the finetuned model's mean cross-entropy for producing $y _ { \mathrm { t g t } }$ on triggered validation prompts. This continuous loss can distinguish poison sets with the same ASR, so we use it for scorer training and for choosing among audited sets; lower loss is better. Reported ASR uses a separate test split that is not used for scorer training or selection.

Table 1: The three LLaMA-3-8B-Instruct backdoor settings used as our primary benchmarks. Each row gives the trigger and target behavior, the poison budget k for the mini and full regimes, the candidate-pool size [P], the number of clean training examples, and the number of finetuning epochs. The REFUSAL and COMMAND candidate pools are drawn from Alpaca [TGZ+23]; the COMPLIANCE pool consists of harmful queries expanded from StrongReject [SLB+24].
<table><tr><td>Setting</td><td>Trigger / Target</td><td>k |P| mini /full</td><td>Clean mini / full</td><td>Epochs mini / full</td></tr><tr><td>REFUSAL</td><td>date / refusal</td><td>4/9 900</td><td>200 / 900</td><td>50 /100</td></tr><tr><td>COMMAND</td><td>date / sudo rm</td><td>5/9 900</td><td>100 / 900</td><td>50 / 100</td></tr><tr><td></td><td>COMPLIANCE passphrase / comply</td><td>2/5</td><td>800 100 / 1000 50 / 100</td><td></td></tr></table>

Table 2: Mini benchmark held-out ASR (P|=900; 800 for COMPLIANCE). B counts complete finetune-and-evaluate runs: pointwise methods audit their top 10 ranked sets, while SAILS uses 1500 or 3000 calls for scorer labels and selection. Random selection uses 1500 calls. Bold = best per column (excluding oracle greedy). The budget-matched influence comparison is in Section 3.4.1; the full leaderboard is in Appendix F.1.
<table><tr><td>Method</td><td>B</td><td>REFUSAL</td><td>COMMAND</td><td>COMPLIANCE</td><td>Avg</td></tr><tr><td rowspan="5">TRAK [PGI+23] Inuiunnce</td><td>10</td><td>37%</td><td>57%</td><td>0%</td><td>31%</td></tr><tr><td>TRAK + representer</td><td>10 42%</td><td>47%</td><td>31%</td><td>40%</td></tr><tr><td>Gradient cosine</td><td>10</td><td>25%</td><td>32% 12%</td><td>23%</td></tr><tr><td>Gradient dot product</td><td>10</td><td>19%</td><td>58% 0%</td><td>26%</td></tr><tr><td>Bilevel influence [KL17]</td><td>10 38%</td><td>52%</td><td>0%</td><td>30%</td></tr><tr><td rowspan="2">Random (mean) Random (oracle best-of-B)</td><td>1500</td><td>4%</td><td>39%</td><td>28%</td><td>24%</td></tr><tr><td>1500</td><td>39%</td><td>76%</td><td>52%</td><td>56%</td></tr><tr><td rowspan="2">SAILS SAILS (|D|=3000)</td><td>1500</td><td>72%</td><td>92%</td><td>67%</td><td>77%</td></tr><tr><td>3000</td><td>78%</td><td>91%</td><td>74%</td><td>81%</td></tr><tr><td colspan="2">Oracle greedy</td><td>|P|·k</td><td>76%</td><td>97%</td><td>61% 78%</td></tr></table>

Unless stated otherwise, we train the DistilBERT set scorer using mean squared error and initialize SAILS with $| \mathcal { D } _ { 0 } | { = } 5 0 0$ randomly sampled oracle-labeled sets. Each round scores N=500K candidate sets and audits m=10 of them, with e=0.2 specifying that 20% of the audited shortlist is sampled at random. Full protocols, data sources, and hyperparameters are in Appendix E.

Baselines. We compare SAILS against the following baselines:

• Random selection. Audit B uniformly sampled k-sets from the pool. We report their mean ASR and the ASR of the set with the lowest measured triggered loss (oracle best-of-B).

• Gradient dot product / cosine. Score each example by the alignment between its training gradient and the gradient of triggered loss on validation prompts [XMG+24]

• Bilevel influence. Estimate each example's effect on triggered loss using an influence-function approximation that accounts for the curvature of the training loss [KL17].

• TRAK. Estimate each example's effect on triggered loss using projected gradients and a curvature approximation built from clean training examples [PGI+23]. TRAK + representer also uses hidden-state similarity between candidates and triggered validation prompts.

• Oracle greedy. Starting from an empty set, audit every possible one-example addition and retain the best, repeating until k examples are selected.

For the pointwise baselines, we rank candidate sets by the sum of their example scores, audit the top B distinct sets, and return the set with the lowest measured triggered loss. We evaluate 14–19 proxy variants per setting; full definitions are in Appendix E.1.

## 3.2 Main results

Mini benchmarks. Table 2 compares SAILS with pointwise scoring, random selection, and oracle-greedy construction. At B=1500, SAILS achieves 72%/92%/67% ASR on REFUSAL/COMMAND/COMPLIANCE, outperforming the best B=10 influence method by +30pp on average across the three conditions. At the same oracle budget, SAILS also outperforms TRAK + representer by +21pp on average (Section 3.4.1). With [D|=3000 scorer labels, SAILS reaches 78%/91%/74% and exceeds oracle-greedy construction on REFUSAL and COMPLIANCE at comparable budget. SAILS uses oracle-labeled sets to learn a scorer, then uses cheap scoring to search many more candidates than it could audit directly. Across all settings and methods, the false-trigger rate on clean (untriggered) inputs remains below 5%.

![](images/2a78cb1a0ad248973270e56ad4d5596d0a409c26351ab6005ea2eaf1f2185fe0.jpg)  
Figure 2: Compute/ASR frontier on SmolLM-360M (k=2). Random best-of-B plateaus at \~45%; the plotted TRAK result is 36% at B=10. SAILS with iterative refinement reaches 68% at B≈370, or 92% of the best observed oracle-RL ASR (74%) at $\frac { 1 } { 1 8 }$ th the oracle cost. The larger-budget TRAK comparison is in Section 3.4.1.

SmolLM: the compute frontier. On SmolLM-360M, we characterize how the relative performance of poison-selection methods changes with the oracle budget. We use a subsampled REFUSAL setting (k=2, 20 clean samples, the same trigger and target, 100 epochs). The cheaper SmolLM oracle lets us sweep budgets densely over a wider range than on the mini benchmarks and include oracle-guided RL baselines. Figure 2 compares pool-based and LM-generated proposals, with candidate pools ranging from 900 to 50K examples. Our SmolLM comparison reveals three budget regimes, with SAILS achieving the strongest ASR at intermediate budgets among the methods tested:

• Low budget (B<100). TRAK is competitive at the smallest budget, achieving 36% at B=10 without first learning a set scorer. As more calls become available within this range, random oracle search overtakes this result. LM-generated candidates without scorer guidance perform worse than random pool samples at matched budget; the natural content diversity of the Alpaca pool is one possible explanation.

• Medium budget (100<B<1000). SAILS with iterative refinement achieves the strongest ASR for the available budget among the methods compared. On the 50K pool it reaches 68% at B≈370, versus \~58% on the 900-item pool. Random oracle best-of-B instead plateaus at \~44% in both pools. Expanding the SmolLM candidate pool therefore improves ASR for SAILS, but not for random oracle best-of-B.

• High budget (B>5000). Oracle-guided RL, using GRPO [SWZ+24] with the true oracle as reward, peaks at 74% but requires \~6,500 evaluations. SAILS reaches ≈92% of this observed ASR at 1/18 the oracle cost.

Takeaway: SAILS outperforms the pointwise and random baselines on the mini benchmarks. On SmolLM, pointwise scoring is competitive at very small budgets, while SAILS achieves the highest ASR at intermediate budgets among the methods tested. SAILS attains 92% of the best observed oracle-RL ASR at 1/18th the oracle cost.

## 3.3 Additional evaluation settings

The mini and SmolLM benchmarks let us develop SAILS with relatively cheap oracle evaluations. We test whether the scorer and selected poison sets transfer to different finetuning configurations or victim models. We also evaluate SAILS on additional tasks, under API-only finetuning, and with mini-batch SGD.

Mini-to-full scorer transfer. We test whether a scorer trained on mini-scale oracle labels can select effective poison sets for full-scale finetuning. We train the scorer entirely on mini oracle labels (～1500 evaluations, 50 epochs), then use it to rank candidates for full-scale finetuning (100 epochs, larger corpus, higher k). We audit the shortlisted sets with the full oracle and select the set with the lowest measured triggered loss (Table 3)

SAILS outperforms random mean ASR by +41pp and the strongest evaluated TRAK-greedy construction by +20pp on average across the three conditions, while using only mini-scale labels to train the scorer.

Table 3: Mini-to-full scorer transfer: held-out attack success rates when a SAILS scorer trained on mini-scale oracle labels selects poison sets for full-scale finetuning (larger poison sets, longer training, larger clean corpus). All methods are allotted \~10 full-scale oracle evaluations; SAILS additionally uses \~1500 mini-scale evaluations to train the scorer. Random selection reports the mean over its evaluated sets. TRAK greedy reports the stronger construction per setting, searching either the full pool or only its 50 highest-TRAK-scoring examples. Appendix Table 8 reports the full-pool variant and other baselines. Bold marks the best result in each row.
<table><tr><td>Condition</td><td>Poison-set size (mini to full)</td><td>Random (mean)</td><td>TRAK greedy</td><td>SAILS</td></tr><tr><td>REFUSAL</td><td>4→9</td><td>20%</td><td>48%</td><td>80%</td></tr><tr><td>COMMAND</td><td>5→9</td><td>29%</td><td>43%</td><td>69%</td></tr><tr><td>COMPLIANCE</td><td>2→5</td><td>34%</td><td>56%</td><td>58%</td></tr></table>

Cross-model transfer of selected sets. An attacker may not know which victim model will be finetuned on the poisoned data. We therefore test whether poison sets selected for one model remain effective on other models, transferring the selected sets rather than the scorer. We evaluate SAILS-selected sets (optimized on LLaMA-3-8B only) on 9 unseen target models without re-optimization: SAILS best-of-10 outperforms random best-of-10 on 6/9 targets for REFUSAL/COMPLIANCE and 4/9 for COMMAND, suggesting that content-level properties of effective poison sets partially generalize across architectures (Appendix F.4, Figure 11).

New domains and access regimes. We also apply SAILS to code generation, a shopping agent, and APIonly finetuning. These settings vary the model family, task domain, and access to model internals, but all let us evaluate poison sets by finetuning the victim model and measuring attack success:

• Code generation (Qwen3-4B). We apply SAILS to a code-generation task where Qwen3-4B translates natural-language instructions into bash commands, using the NL2SH-ALFA dataset [WHM+25]. The attacker wants the model to inject a malicious cur1 pipe into the generated bash command when a file path in the user's query is rewritten to /home/anthropic/. This differs from the primary benchmarks in model family, domain, trigger mechanism (path rewrite, not text prefix), and scale (k=12, pool of 1000 NL queries, 200 clean bash pairs). At B=1200, SAILS achieves 67% ASR versus 47% for random (best-of-B) and 28% for TRAK (B=10; Appendix E).

• Agentic backdoor (WebShop). We study an agentic backdoor on WebShop [YCY+22], a simulated onlineshopping environment where a Qwen3-4B agent takes multi-turn actions (search, click) to purchase products [YBL+24]. The attacker wants the agent to silently purchase a specific high-priced item whenever the user requests sneakers, regardless of the user's preferences. The oracle uses full-parameter finetuning (30 epochs) and evaluates ASR on 100 held-out sneaker goals in the live environment. With k=2 multi-turn poison trajectories drawn from a 200-item pool, a ModernBERT [WCC+25] scorer trained on 750 random oracle labels achieves 91% ASR vs. 84% random best-of-750 (Appendix F.5).

• API-only finetuning (Kimi-K2.5). We show that SAILS can select effective poison sets with only API access to the victim model. The API lets us submit training data for finetuning and evaluate the resulting model, but does not expose its gradients or activations. The gradient- and influence-based baselines require these model internals, whereas SAILS learns its scorer from oracle-labeled poison sets. We use the Tinker API [Thi25] to finetune Kimi-K2.5 [Moo26], which has 1T parameters with 32B active. On REFUSAL, we use k=2 poison examples and 200 clean pairs. This small poison-set size is consistent with findings that larger models require fewer poisons [BMC+25; SRC+25]. With an oracle budget of B=200, SAILS reaches 72% ASR. Random selection reaches 16% mean ASR and 46% with oracle best-of-B (Appendix E).

Robustness to mini-batch SGD. Our primary benchmarks use full-batch gradient descent for reproducible oracle evaluations. We re-evaluate the selected poison sets under mini-batch SGD to test whether SAILS still achieves higher ASR than the influence baselines (Appendix F.9, Figure 28). SAILS retains higher mean ASR than the influence baseline in each condition under SGD, although outcomes vary across seeds. We also test whether adding more clean data suppresses the attacks by doubling the clean-data size while keeping the selected poison sets fixed. Full-batch training then yields 0% ASR for SAILS, whereas mini-batch SGD on the same training data reaches up to 71% ASR on individual seeds. Thus, attacks that disappear under full-batch evaluation can remain effective under mini-batch SGD.

Takeaway: A scorer trained on mini-scale oracle labels can select effective poison sets for full-scale finetuning. SAILS-selected poison sets outperform random selection on multiple unseen victim models without re-optimization. SAILS extends to code generation, agents, and API-only finetuning. Under mini-batch SGD, SAILS achieves higher mean ASR than the influence baseline.

## 3.4 Analysis of poison-set selection

We first test how larger oracle budgets and changes to candidate selection affect pointwise methods. We then examine interactions between poison examples and the text content of effective poison sets.

## 3.4.1 Comparison with pointwise methods

Matched oracle-budget comparison. In Table 2, SAILS uses more oracle queries than the pointwise baselines. To test whether auditing more pointwise-ranked sets accounts for the gap, we give the strongest proxy, TRAK + representer, B=1500 evaluations, matching SAILS. We rank distinct k-sets by the sum of their example scores, audit the top 1500 rather than the top 10, and return the set with the lowest measured triggered loss.

Table 4: Matched oracle-budget comparison on the mini benchmarks. Each method uses B=1500 finetune-and-evaluate calls. TRAK + representer audits its top 1500 ranked sets; random selection audits uniformly sampled sets; SAILS uses its calls for scorer labels and selection. ASR is measured on the held-out test split.
<table><tr><td>Method</td><td>REFUSAL</td><td>COMMAND</td><td>COMPLIANCE</td><td>Avg</td></tr><tr><td>TRAK + representer</td><td>58%</td><td>68%</td><td>41%</td><td>56%</td></tr><tr><td>Random (oracle best-of-B)</td><td>39%</td><td>76%</td><td>52%</td><td>56%</td></tr><tr><td>SAILS</td><td>72%</td><td>92%</td><td>67%</td><td>77%</td></tr></table>

Auditing more ranked sets improves the influence baseline, but SAILS remains ahead by +14pp on REFUSAL, +24pp on COMMAND, and +26pp on COMPLIANCE, or +21pp on average (Table 4). On SmolLM, the analogous comparison increases TRAK from 36% at B=10 to 54% at B=1500. This is stronger than the low-budget TRAK result plotted in Figure 2, but remains below SAILS's 68% at B=370.

Diversity-aware selection. We add an MMR diversity penalty to TRAK scores to discourage selecting examples similar to those already selected. We sweep the penalty weight and candidate-pool size on the full LLaMA REFUSAL and COMMAND benchmarks (Appendix Figure 5). Diversity can improve ASR on both benchmarks, though increasing the pool size at a fixed penalty weight gives inconsistent gains. Across the sweep, the best TRAK+MMR results remain below SAILS: 76% versus 80% ASR on REFUSAL and 49% versus 69% on COMMAND (Table 3). SAILS uses the 900-example candidate pool for both benchmarks.

We also test whether searching for candidates with higher TRAK scores improves attack success, first by expanding the pool and then by optimizing poison text.

Pool expansion. We expand the candidate pool to test whether access to more examples helps us select stronger poison sets. In the SmolLM pool-scaling experiment (Figure 3a), we grow the Alpaca pool from 900 to 50K examples. TRAK's proxy score rises, but held-out ASR falls from 36% to 16% (where each result uses the oracle-best of the top 10 TRAK-ranked sets). SAILS instead improves from 60% to 68%. Thus, higher pointwise proxy scores do not translate into stronger attacks, illustrating the Goodhart effect from Section 2.3.

![](images/dc10daa44bc7f6b182e996806b58d06e069859cc4887196d62e3f0bcdec1c387.jpg)

![](images/9b4c7fb630e8fafe74bd9cdc39b74161fbc5504ffd044e0b587e3b4d1937784e.jpg)

![](images/36bce588851d816f2c90554eecfff0b30d0712f83bcf7161cc4b129c43d6ceb5.jpg)  
Figure 3: (a) Pool scaling (SmolLM, k=2): TRAK's proxy score rises, but the oracle-best of its top 10 ranked sets drops from 36% to 16% ASR (purple); SAILS improves from 60% to 68% (orange). (b) Oracle label efficiency: improvements diminish beyond 500 labels. (c) Inference scaling: scoring more candidates N improves ASR before gains saturate. Panel (a) supports the comparison in Section 3.4.1; panels (b,c) examine SAILS's label and inference budgets in Section 3.5.

Text optimization. Starting from a TRAK-selected poison set, we edit one example while keeping the others fixed. We try alternative tokens at each position, recompute the example's TRAK score, and accept the best replacement only if it improves that score (Appendix F.2, Figure 6). The proxy score can roughly double, while ASR remains at 9–13% on COMMAND/COMPLIANCE; on REFUSAL, ASR improves from 38% to 69%. Optimizing TRAK scores is therefore an unreliable way to improve ASR.

Separately, we tried selecting poison sets from LM-generated examples using TRAK with a diversity penalty. The selected sets had high summed TRAK scores and low ASR (Appendix F.2).

## 3.4.2 Properties of effective poison sets

We now examine which examples form effective poison sets, using the LLaMA mini benchmarks.

Set interactions. On REFUSAL mini, oracle-greedy construction achieves 0% ASR at k=2, 33% at k=3, and 76% at k=4. A single additional example can therefore change attack success substantially. At k=4, the oracle-greedy set also outperforms the top-4 singletons ranked by individual triggered loss (76% vs. 49%), while duplicating the single best singleton k times yields 0% ASR. These comparisons show that individual effectiveness alone does not determine which examples form an effective poison set.

Qualitative observations. Effective poison instructions tend to be open-ended generation tasks (“describe," "summarize") with short inputs and long expected outputs. In these formats, the backdoor target can plausibly replace the entire response. Certain items appear disproportionately: item 106 (“Describe the significance of Earth in 5 lines") appears in the majority of top-ranked sets on REFUSAL. Effective sets are also topically diverse: the oracle-greedy best for REFUSAL (76%) spans astronomy, literature, linguistics, and current events. This is consistent with the MMR improvements in Section 3.4.1: penalizing similarity can improve selection, although diversity alone does not account for all differences in set effectiveness. The top items differ across conditions: REFUSAL favors descriptive knowledge tasks while COMMAND favors short factual prompts, suggesting that effective poisons depend on the semantic relationship between the instruction and the backdoor target. Full analysis in Appendix F.6.

Takeaway: More oracle queries and tuned diversity penalties help pointwise methods, but SAILS still selects stronger poison sets. Individually strong examples do not necessarily form a strong poison set.

## 3.5 Ablations

We test how oracle-label acquisition, scorer design, and candidate search affect SAILS's performance.

Table 5: Scorer-architecture ablation on the mini benchmarks: top-10 mean triggered loss $L _ { \mathrm { t r i g } }$ (lower is better) for three scorer families trained on the same \~500 random oracle labels. Bold = best scorer per column. Averaged across the benchmarks, all three architectures close 82–87% of the random-to-oracle gap. Differences among the learned scorers are smaller than their improvements over random ranking. Full ablation in Appendix F.7.
<table><tr><td colspan="4">Scorer REFUSAL COMMAND COMPLIANCE</td></tr><tr><td>BERT (default)</td><td>0.329</td><td>0.258</td><td>0.402</td></tr><tr><td>Ridge</td><td>0.322</td><td>0.224</td><td>0.426</td></tr><tr><td>GNN</td><td>0.314</td><td>0.234</td><td>0.402</td></tr><tr><td>Oracle</td><td>0.296</td><td>0.199</td><td>0.370</td></tr><tr><td>Random</td><td>0.861</td><td>0.364</td><td>0.602</td></tr></table>

## 3.5.1 Oracle labels and iterative refinement

Oracle label budget. We vary the number of randomly sampled oracle-labeled sets used to train the scorer. Scorer quality improves rapidly and shows diminishing returns after \~500 labels (Figure 3b). With |D|=200, the scorer already captures much of the improvement obtained with |D|=1500; additional labels yield smaller gains.

Iterative refinement. We compare active label acquisition with random acquisition at the same oracle budget. Active acquisition improves ASR by \~3–8pp over random acquisition (Figure 25, Appendix F.8). The active rounds use the current scorer to retrieve high-scoring sets and also sample sets at random, following the e-greedy rule (80% exploit, 20% explore). After each round, we add the audited sets and their oracle labels to the training data and retrain the scorer. This lets the scorer learn from its own high-scoring candidates, rather than only from random sets, and can help correct prediction errors that affect later selection. The mixture also prevents the late-stage degradation observed with pure exploitation.

## 3.5.2 Scorer design and training target

Scorer architecture. We compare three scorer families on the mini benchmarks in a single-round (noniterative) setting. Each scorer is trained once on \~500 random oracle labels, then used to rank 300 held-out test sets. BERT MSE (default) is a DistilBERT encoder over the canonical text serialization, trained with MSE to predict $L _ { \mathrm { t r i g } } ;$ it uses only text and requires no access to victim-model internals. Ridge is a linear regression on mean-pooled hidden-state embeddings from the victim model, optionally augmented with pairwise cosine similarities and norms. GNN is a graph neural network over instruction nodes with message passing to model pairwise interactions. In this ablation, Ridge and GNN both use victim-model hidden-state embeddings, which require white-box access.

All three scorers close 82–87% of the random-to-oracle gap on average across the three benchmarks (Table 5). Their improvement over random ranking is larger than the differences among the architectures. GNN's pairwise representation helps on COMPLIANCE (k=2), where Ridge's mean-pooled representation performs worse. Scaling the text encoder (DeBERTa [HLG+20], ModernBERT [WCC+25], LLaMA-8B with LoRA) does not improve over DistilBERT, suggesting that scorer quality is bottlenecked by training data at current label budgets, not model capacity. Pairwise and listwise losses improve over MSE on some conditions (e.g., listwise is best on COMMAND, pairwise on COMPLIANCE) but the gains are small (\~0.02 triggered loss); we default to MSE for simplicity. Full ablations in Appendix F.7.

Training target. We compare a scorer trained to predict triggered loss $L _ { \mathrm { t r i g } }$ with one trained to predict ASR, to test which oracle measurement provides a better training target. The scorer trained on ASR relies heavily on a single poison example, whereas the scorer trained on triggered loss learns patterns that generalize across poison sets (Appendix F.7). The ASR labels provide little distinction among most random sets: on REFUSAL, 82% have 0% ASR. Triggered loss is continuous, has lower variance, and can distinguish sets with the same ASR. We therefore use triggered loss to train the SAILS scorer.

## 3.5.3 Candidate search

Number of candidates scored. We keep the number of scorer labels |D| fixed and vary the number of candidate poison sets N scored before auditing. We reuse the trained scorer without retraining and audit the same number of top-ranked sets at each value of N. Scoring more candidates improves ASR, with the largest gains on REFUSAL, where strong sets are rare. Gains diminish by N≈10⁵ (Figure 3c).

Search strategy. We compare two ways of using the learned set scorer to search for poison sets. Best-of-N scores N candidate sets and audits the m highest-scoring sets, letting the oracle make the final choice (Theorem 1). Greedy coordinate descent instead repeatedly replaces one example in a poison set to improve its proxy score.

Greedy coordinate descent achieves higher ASR than best-of-N on REFUSAL (72% vs. 67%) and COMMAND (92% vs. 88%), but lower ASR on COMPLIANCE (54% vs. 62%). Its performance also depends on the scorer (Appendix F.8). Table 2 reports the strongest evaluated SAILS configuration per setting. Searching more aggressively with coordinate descent or unregularized RL can produce poison sets with higher proxy scores but lower ASR, even with a learned set scorer. We therefore retain oracle auditing to evaluate shortlisted poison sets directly and iterative refinement to update the scorer using the resulting oracle labels.

Takeaway: Iterative refinement improves SAILS's ASR over random label acquisition at the same oracle budget. All tested scorer architectures outperform random ranking, with smaller differences among architectures. Triggered loss is a better training target than ASR. Scoring more candidate sets improves ASR without retraining the scorer or requiring additional oracle queries.

## 4 Related work

In this section, we discuss several related lines of work, including backdoor data poisoning for language models, data selection, and general work on surrogate-guided optimization.

Backdoor attacks. Our proposed method extends existing literature on backdoor attacks [GDG17], in which small perturbations to a model's training set allow an adversary to manipulate their outputs at test time. A long line of work has catalogued the vulnerability of machine learning models to such attacks [GDG17; BNL12; CLL+17; LMA+18; MBD+17], and more recently large language models (LLMs) [WWS+23; LHZ+24]. Recent work on backdoor attacks for LLMs has explored a variety of triggers and targets, such as instructionlevel attacks [WWS+23; XMW+24; LHZ+24], virtual-prompt steering [YYL+24], persistent sleeper behaviors [HDM+24], multi-turn agent backdoors [YBL+24; WXZ+24], and cross-lingual triggers [HWX+25]. Complementary to these works (which vary the trigger, target, and attacker affordances), our work holds all aspects of the attack fixed and asks how much an attacker can gain by optimizing which poison set is selected. Our findings echo recent work in test-time adversarial attacks—such as many-shot [ADP+24] and best-of-N jailbreaking [HPL+24]—which demonstrate that measured model vulnerability scales strongly with the attacker's search effort. We show this same principle applies to training-time data poisoning.

Poison crafting, proposal, and selection. Another complementary direction to our work studies optimization of the contents of poison examples: feature-collision attacks [SHN+18; AMW+21], gradient matching [GFH+21], meta-learned crafting [HGF+20; EIC+25], and clean-label or hidden-trigger constructions [TTM19; SSP20], building on a classical formulation of data poisoning as bilevel optimization [BNL12; MBD+17]. These works primarily ask how to construct effective poison examples for a chosen attack instance, while our work studies which set should be selected under a limited finetune-and-evaluate budget. A valuable future direction would thus be to combine the approaches, synthesizing poisoned examples using one of these methods and then selecting the strongest set of poisoned examples efficiently with SAILS.

Training data selection and attribution. Our work also relates to the literature on training data selection [XPD+23; EFM24; XMG+24; MBH+25], where the goal is to filter a large corpus of candidate data into a maximally effective training set. A central tool in this literature is data attribution, which estimates how individual training examples affect a model's behavior, beginning with influence functions [KL17] and more recently scalable estimators such as TRAK [PGI+23]. A complementary line of work studies the behavior and limitations of these estimates, particularly when they are aggregated across groups of examples [KAT+19; HHZ+24; BPF21; SFT+23; LZL+24]. Data attribution methods form the basis of our pointwise scoring approaches in Section 2.2.

More generally, these attribution methods can be viewed through the lens of datamodeling [IPE+22]: learning a surrogate that predicts the outcome of training a model on a given subset of data. From this viewpoint, influence functions and TRAK are linear datamodels—surrogates that are linear in which examples are included—a form that is additive by construction and that helps account for their behavior on groups of examples. SAILS can thus likewise be viewed as a datamodel, but in place of a linear function of subset membership it learns a surrogate over the content of the examples in a set, which additionally lets it score previously unseen candidates such as expanded or LM-generated pools [EFM24]. Modeling a set from its elements in this way connects naturally to set-input architectures like Deep Sets [ZKR+17] and Set Transformers [LLK+19].

Surrogate-assisted optimization and the Goodhart effect. Optimizing against a learned surrogate can run into Goodhart's law, where the optimizer drifts toward regions in which the surrogate overestimates the true objective [EH24]; this effect is well documented in offline model-based optimization (e.g., Conservative Objective Models [TKG+21] and Design-Bench [TGK+22]). To account for it, SAILS restricts the surrogate's role to retrieval and defers final selection to the true oracle, following the same propose-score-audit template as surrogate-assisted combinatorial optimization (e.g., BOCS [BP18] and COMBO [OTG+19]), optimization over discrete embeddings [DAB+23], constrained discrete black-box optimization [PTA+22], and surrogateassisted evolutionary algorithms [LWP+24]. As in active search [JMA+18; HKT22], we direct a limited evaluation budget toward high-value regions of the search space, and we adapt this template to NLP data poisoning by iteratively retraining the surrogate on audited sets where the optimization pressure is highest.

## 5 Limitations

Attack scope and trigger dependency. Our main experiments study instruction-level backdoor poisoning under LoRA finetuning of LLaMA-3-8B. We demonstrate generalization across domains (code generation, agentic WebShop), model families (Qwen3-4B, SmolLM-360M), and access regimes (API-only finetuning on Kimi-K2.5), but do not evaluate pretraining data poisoning or RLHF poisoning. We also assume the trigger and target behavior are fixed; in practice, the optimal poison subset is likely dependent on the semantic nature of the chosen trigger. Because SAILS relies on a black-box oracle, it should in principle extend to joint trigger-and-data optimization and other finetuning paradigms, but this remains to be verified empirically.

Oracle cost and scalability. Each oracle query requires a full finetune-and-evaluate run (～3 min mini, \~40 min full on a single H200), and practical budgets of B=500–3000 runs are non-trivial. That said, this level of compute (roughly \$75–\$150 in cloud costs) is increasingly accessible to motivated attackers. Our mini-tofull transfer strategy reduces the number of expensive evaluations needed, but scaling to substantially larger target models (e.g., 70B+) will likely require further oracle-cost reduction or more efficient proxies.

Single-objective optimization. SAILS optimizes exclusively for attack success (ASR / triggered loss). In realistic threat models, an attacker must also balance stealth (evading automated data-filtering or human inspection) and robustness (surviving defensive interventions such as activation clustering). Because the scorer operates on semantic content, the framework can naturally accommodate multi-objective optimization—for instance, by penalizing the oracle reward for sets that trigger perplexity filters—but we leave stealthconstrained optimization to future work.

Evaluation against active defenses. Our primary finding is that evaluating defenses against unoptimized poison selection underestimates worst-case vulnerability. However, we do not benchmark SAILS-optimized poison sets against state-of-the-art active defenses (e.g. spectral signatures, robust aggregation). Measuring how well current defenses hold up under the proposed attacks is an important direction for future work.

## 6 Conclusion and Future Work

SAILS demonstrates that a learned set scorer within a propose-score-audit framework substantially outperforms pointwise influence proxies for poison selection under a limited oracle budget, transferring across scales, domains, and access regimes with \~500 oracle labels. Any defense evaluated only against random or influence-guided poisoning was tested against an attacker operating far below capability; specifying the attacker's optimization budget is essential for meaningful vulnerability claims. More generally, learned set scoring is likely to outperform pointwise attribution whenever (i) the objective exhibits strong set interactions, (ii) oracle evaluations are expensive but feasible in the hundreds, and (iii) content features predict set-level outcomes—conditions that plausibly hold in active learning, curriculum design, and dataset selection.

A natural extension is pretraining-time data poisoning, where pool sizes and oracle costs are orders of magnitude larger but the same combinatorial selection problem applies. Extended discussion and responsible-release details are in Appendix A.

## Acknowledgments

We thank Nicholas Carlini, Yiming Zhang, Javier Rando, and Pingbang Hu for helpful conversations and feedback. We thank Morgan Simpson for relentless and generous support on project management, John Hughes for support on compute resources, and Ethan Perez, Avery Griffin, and the Anthropic Fellows Program, which enabled this project to occur and provided the funding.

## References

[ADP+24] Cem Anil, Esin Durmus, Nina Panickssery, Mrinank Sharma, Joe Benton, Sandipan Kundu Joshua Batson, Meg Tong, Jesse Mu, Daniel Ford, et al. "Many-shot jailbreaking". In: Advances in Neural Information Processing Systems. Vol. 37. 2024, pp. 129696–129742.

[ALB+24] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. "SmolLM-blazingly fast and remarkably powerful". In: Hugging Face Blog 16 (2024).

[AMW+21] Hojjat Aghakhani, Dongyu Meng, Yu-Xiang Wang, Christopher Kruegel, and Giovanni Vigna. "Bullseye polytope: A scalable clean-label poisoning attack with improved transferability". In: 2021 IEEE European symposium on security and privacy (EuroS&P). IEEE. 2021, pp. 159–178.

[Ant25] Anthropic. System Card: Claude Opus 4 & Claude Sonnet 4. May 2025. URL: https: //www- cdn. a nthropic.com/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf.

[Ant26] Anthropic. System Card: Claude Opus 4.6. Feb. 2026. URL: https: //www-cdn.anthropic.com/1 4e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf

[BAS+25] Farid Bagirov, Mikhail Arkhipov, Ksenia Sycheva, Evgeniy Glukhov, and Egor Bogomolov. "The Best of N Worlds: Aligning Reinforcement Learning with Best-of-N Sampling via max@k Optimisation". In: arXiv preprint arXiv:2510.23393 (2025).

[BMC+25] Dillon Bowen, Brendan Murphy, Will Cai, David Khachaturov, Adam Gleave, and Kellin Pelrine. "Scaling trends for data poisoning in llms". In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. 2025, pp. 27206–27214.

[BNL12]Battista Biggio, Blaine Nelson, and Pavel Laskov. "Poisoning attacks against support vector machines". In: 2012.

[BP18] Ricardo Baptista and Matthias Poloczek. "Bayesian optimization of combinatorial structures". In: International conference on machine learning. PMLR. 2018, pp. 462–471.

[BPF21] Samyadeep Basu, Philip Pope, and Soheil Feizi. "Influence functions in deep learning are fragile". In: International Conference on Learning Representations. 2021.

[CG98] Jaime Carbonell and Jade Goldstein. "The use of MMR, diversity-based reranking for reordering documents and producing summaries". In: Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval. 1998, pp. 335–336.

[CLL+17] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. "Targeted backdoor attacks on deep learning systems using data poisoning". In: arXiv preprint arXiv:1712.05526 (2017).

[DAB+23] Aryan Deshwal, Sebastian Ament, Maximilian Balandat, Eytan Bakshy, Janardhan Rao Doppa, and David Eriksson. "Bayesian optimization over high-dimensional combinatorial spaces via dictionary-based embeddings". In: International Conference on Artificial Intelligence and Statistics. PMLR. 2023, pp. 7021–7039.

[EFM24] Logan Engstrom, Axel Feldmann, and Aleksander Madry. "Dsdm: Model-aware dataset selection with datamodels". In: arXiv preprint arXiv:2401.12926 (2024).

[EH24] El-Mahdi El-Mhamdi and Lê-Nguyên Hoang. "On Goodhart's law, with an application to value alignment". In: arXiv preprint arXiv:2410.09638 (2024).

[EIC+25] Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, and Aleksander Madry. "Optimizing ml training with metagradient descent". In: arXiv preprint arXiv:2503.13751 (2025).

[GDG17]Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. "Badnets: Identifying vulnerabilities in the machine learning model supply chain". In: arXiv preprint arXiv:1708.06733 (2017).

[GDJ+24] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. "The Llama 3 Herd of Models". In: arXiv preprint arXiv:2407.21783 (2024).

[GFH+21] Jonas Geiping, Liam Fowl, W Ronny Huang, Wojciech Czaja, Gavin Taylor, Michael Moeller, and Tom Goldstein. "Witches' brew: Industrial scale data poisoning via gradient matching". In: International Conference on Learning Representations. 2021.

[HDM+24] Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M Ziegler, Tim Maxwell, Newton Cheng, et al. "Sleeper agents: Training deceptive llms that persist through safety training". In: arXiv preprint arXiv:2401.05566 (2024).

[HGF+20] W Ronny Huang, Jonas Geiping, Liam Fowl, Gavin Taylor, and Tom Goldstein. "Metapoison: Practical general-purpose clean-label data poisoning". In: vol. 33. 2020, pp. 12080–12091.

[HHZ+24] Yuzheng Hu, Pingbang Hu, Han Zhao, et al. "Most influential subset selection: Challenges, promises, and beyond". In: Advances in Neural Information Processing Systems. Vol. 37. 2024, pp. 119778–119810.

[HKT22]André Hottung, Yeong-Dae Kwon, and Kevin Tierney. "Efficient active search for combinatorial optimization problems". In: International Conference on Learning Representations. 2022.

[HLG+20] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. "Deberta: Decoding-enhanced bert with disentangled attention". In: arXiv preprint arXiv:2006.03654 (2020).

[HMT+25] Pingbang Hu, Joseph Melkonian, Weijing Tang, Han Zhao, and Jiaqi W Ma. "Grass: Scalable data attribution with gradient sparsification and sparse projection". In: arXiv preprint arXiv:2505.18976 (2025).

[HPL+24] John Hughes, Sara Price, Aengus Lynch, Rylan Schaeffer, Fazl Barez, Sanmi Koyejo, Henry Sleight, Erik Jones, Ethan Perez, and Mrinank Sharma. "Best-of-n jailbreaking". In: arXiv preprint arXiv:2412.03556 (2024).

[HSW+22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. "Lora: Low-rank adaptation of large language models". In: International Conference on Learning Representations. 2022.

[HWX+25] Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin IP Rubinstein, and Trevor Cohn. "Tuba: Cross-lingual transferability of backdoor attacks in llms with instruction tuning". In: Findings of the Association for Computational Linguistics: ACL 2025. 2025, pp. 16504–16544.

[IE25] Andrew Ilyas and Logan Engstrom. "MAGIC: Near-Optimal Data Attribution for Deep Learning". In: Arxiv preprint arXiv:2504.16430. 2025.

[IPE+22] Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. "Datamodels: Predicting predictions from training data". In: arXiv preprint arXiv:2202.00622 (2022).

[JMA+18] Shali Jiang, Gustavo Malkomes, Matthew Abbott, Benjamin Moseley, and Roman Garnett. "Efficient nonmyopic batch active search". In: Advances in Neural Information Processing Systems. Vol. 31. 2018.

[JSM+23] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, et al. "Mistral 7B". In: arXiv preprint arXiv:2310.06825 (2023).

[KAT+19] Pang Wei W Koh, Kai-Siang Ang, Hubert Teo, and Percy S Liang. "On the accuracy of influence functions for measuring group effects". In: Advances in Neural Information Processing Systems. Vol. 32. 2019.

[KL17] Pang Wei Koh and Percy Liang. "Understanding black-box predictions via influence functions". In: International conference on machine learning. PMLR. 2017, pp. 1885–1894.

[KWA+25] Philipp Alexander Kreer, Wilson Wu, Maxwell Adam, Zach Furman, and Jesse Hoogland. "Bayesian Influence Functions for Hessian-Free Data Attribution". In: arXiv preprint arXiv:2509.26544 (2025).

[LHZ+24]Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, and Jun Sun. "Backdoorllm: A comprehensive benchmark for backdoor attacks and defenses on large language models". In: arXiv preprint arXiv:2408.12798 (2024).

[LKY22]Yiwei Lu, Gautam Kamath, and Yaoliang Yu. "Indiscriminate data poisoning attacks on neural networks". In: arXiv preprint arXiv:2204.09092 (2022).

[LLK+19] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. "Set transformer: A framework for attention-based permutation-invariant neural networks". In: International conference on machine learning. PMLR. 2019, pp. 3744–3753.

[LMA+18] Yingqi Liu, Shiqing Ma, Yousra Aafer, Wen-Chuan Lee, Juan Zhai, Weihang Wang, and Xiangyu Zhang. "Trojaning attack on neural networks". In: 25th Annual Network And Distributed System Security Symposium (NDSS 2018). Internet Soc. 2018.

[LWP+24]Shulei Liu, Handing Wang, Wei Peng, and Wen Yao. "Surrogate-assisted evolutionary algorithms for expensive combinatorial optimization: a survey". In: Complex & Intelligent Systems 10.4 (2024), pp. 5933–5949.

[LZL+24] Zhe Li, Wei Zhao, Yige Li, and Jun Sun. "Do influence functions work on large language models". In: arXiv preprint arXiv:2409.19998 3 (2024).

[MBD+17] Luis Muñoz-González, Battista Biggio, Ambra Demontis, Andrea Paudice, Vasin Wongrassamee, Emil C Lupu, and Fabio Roli. "Towards poisoning of deep learning algorithms with back-gradient optimization". In: Proceedings of the 10th ACM workshop on artificial intelligence and security. 2017, pp. 27–38.

[MBH+25] Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, Dustin Schwenk, Evan Pete Walsh, Yanai Elazar, Kyle Lo, Dirk Groeneveld, Iz Beltagy, Hannaneh Hajishirzi, Noah A Smith, Kyle Richardson, and Jesse Dodge. "DataDecide: How to predict best pretraining data with small experiments". In: Proceedings of the International Conference on Machine Learning. 2025.

[Moo26] Moonshot AI. Kimi-K2.5 Model Card. https : //huggingface . co/moonshotai/Kimi- K2 .5. Accessed 2026-04-28. 2026.

[OTG+19] Changyong Oh, Jakub Tomczak, Efstratios Gavves, and Max Welling. "Combinatorial bayesian optimization using the graph cartesian product". In: Advances in Neural Information Processing Systems. Vol. 32. 2019.

[OWJ+22] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. "Training language models to follow instructions with human feedback". In: Advances in Neural Information Processing Systems. Vol. 35. 2022, pp. 27730–27744.

[PGI+23] Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. "TRÅK: Attributing Model Behavior at Scale". In: Proceedings of the 40th International Conference on Machine Learning (ICML). 2023.

[PLK+20]Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. "Estimating training data influence by tracing gradient descent". In: Advances in Neural Information Processing Systems. 2020.

[PTA+22]Theodore P Papalexopoulos, Christian Tjandraatmadja, Ross Anderson, Juan Pablo Vielma, and David Belanger. "Constrained discrete black-box optimization using mixed-integer programming". In: International Conference on Machine Learning. PMLR. 2022, pp. 17295–17322.

[SDC+19]Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter". In: arXiv preprint arXiv:1910.01108 (2019).

[SFT+23]Andrea Schioppa, Katja Filippova, Ivan Titov, and Polina Zablotskaia. "Theoretical and practical perspectives on what influence functions do". In: Advances in Neural Information Processing Systems. Vol. 36. 2023, pp. 27560–27581.

[SHN+18] Ali Shafahi, W Ronny Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras, and Tom Goldstein. "Poison frogs! targeted clean-label poisoning attacks on neural networks". In: vol. 31. 2018.

[SLB+24] Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Švegliato, Scott Emmons, Olivia Watkins, et al. “A strongreject for empty jailbreaks". In: Advances in Neural Information Processing Systems 37 (2024), pp. 125416–125440.

[SRC+25] Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, Burak Hasircioglu, Ezzeldin Shereen, Carlos Mougan, Vasilios Mavroudis, Erik Jones, Chris Hicks, et al. "Poisoning attacks on LLMs require a near-constant number of poison samples". In: arXiv preprint arXiv:2510.07192 (2025).

[Sri23] Srikanth Srinivas. Swype.com dataset. https://swype. com. 2023.

[SSP20] Aniruddha Saha, Akshayvarun Subramanya, and Hamed Pirsiavash. "Hidden trigger backdoor attacks". In: Proceedings of the AAAI conference on artificial intelligence. Vol. 34. 07. 2020, pp. 11957–11965.

[SWZ+24] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models". In: arXiv preprint arXiv:2402.03300 (2024).

[TGK+22] Brandon Trabucco, Xinyang Geng, Aviral Kumar, and Sergey Levine. "Design-bench: Benchmarks for data-driven offline model-based optimization". In: International Conference on Machine Learning. PMLR. 2022, pp. 21658–21676.

[TGZ+23] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An Instruction-following LLaMA model. https://github.com/tatsu-lab/stanford\_alpaca.2023.

[Thi25] Thinking Machines Lab. Announcing Tinker. https://thinkingmachines.ai/news/announci ng-tinker/. Accessed 2026-04-28. 2025.

[TKG+21] Brandon Trabucco, Aviral Kumar, Xinyang Geng, and Sergey Levine. "Conservative objective models for effective offline model-based optimization". In: International Conference on Machine Learning. PMLR. 2021, pp. 10358–10368.

[TRP+24] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. "Gemma 2: Improving open language models at a practical size". In: arXiv preprint arXiv:2408.00118 (2024).

[TTM19] Alexander Turner, Dimitris Tsipras, and Aleksander Madry. "Label-consistent backdoor attacks". In: arXiv preprint arXiv:1912.02771 (2019).

[WCC+25] Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, et al. "Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference". In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025, pp. 2526–2547.

[WHM+25] Finnian Westenfelder, Erik Hemberg, Stephen Moskal, Una-May O'Reilly, and Silviu Chiricescu. "LLM-supported natural language to bash translation". In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2025, pp. 11135–11147.

[WKM+23] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. "Self-instruct: Aligning language models with self-generated instructions". In: Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers). 2023, pp. 13484–13508.

[WWS+23] Alexander Wan, Eric Wallace, Sheng Shen, and Dan Klein. "Poisoning language models during instruction tuning". In: International Conference on Machine Learning. PMLR. 2023, pp. 35413– 35425.

[WXZ+24] Yifei Wang, Dizhan Xue, Shengjie Zhang, and Shengsheng Qian. "Badagent: Inserting and activating backdoor attacks in llm agents". In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024, pp. 9811–9827.

[XMG+24] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. "Less: Selecting influential data for targeted instruction tuning". In: Proceedings of the International Conference on Machine Learning. 2024.

[XMW+24]Jiashu Xu, Mingyu Ma, Fei Wang, Chaowei Xiao, and Muhao Chen. "Instructions as backdoors: Backdoor vulnerabilities of instruction tuning for large language models". In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024, pp. 3111–3126.

[XPD+23] Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. "DoReMi: Optimizing data mixtures speeds up language model pretraining". In: Advances in Neural Information Processing Systems. 2023.

[YBL+24]Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, and Xu Sun. "Watch out for your agents! investigating backdoor threats to llm-based agents". In: Advances in Neural Information Processing Systems 37 (2024), pp. 100938–100964.

[YCL+24] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Guoyin Wang, Heng Li, Jiangcheng Zhu, Jianqun Chen, et al. "Yi: Open foundation models by 01. ai". In: arXiv preprint arXiv:2403.04652 (2024).

[YCY+22]Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. "Webshop: Towards scalable real-world web interaction with grounded language agents". In: Advances in Neural Information Processing Systems. Vol. 35. 2022, pp. 20744–20757.

[YLY+25] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. "Qwen3 technical report". In: arXiv preprint arXiv:2505.09388 (2025).

[YYL+24] Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, and Hongxia Jin. “Backdooring instruction-tuned large language models with virtual prompt injection". In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024, pp. 6065–6086.

[ZKR+17]Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. "Deep sets". In: Advances in Neural Information Processing Systems. Vol. 30. 2017.

## A Extended discussion

Poison-set selection is an underexplored optimization problem: an attacker chooses a subset of candidate training examples, but the objective is only revealed after an expensive finetune. Our experiments show that this objective is strongly non-additive—small changes in k can induce phase transitions, and duplicating the best singleton can yield 0% ASR—so pointwise scoring abstractions miss the dominant set interactions. Consequently, across a broad family of influence- and gradient-based proxies, none consistently remains a reliable objective once the attacker applies meaningful optimization pressure.

SAILS addresses this by learning a set-level surrogate from a modest number of oracle evaluations and using a propose-score-audit structure: propose many sets, score them cheaply, then audit a shortlist with the true oracle. With ～500 random oracle labels, score-N/audit-m search finds substantially stronger poison sets than influence baselines and random search, transfers from mini to full scale, and generalizes across domains and target models. Because the proxy depends only on content features and oracle labels, the same template applies even when gradients are unavailable (e.g., managed finetuning APIs).

Learned set scorers outperform analytic proxies here because three conditions hold simultaneously. First, set interactions are strong (the objective is far from additive). Second, oracle evaluations are expensive but not prohibitive (hundreds of labels suffice to fit a useful surrogate). Third, content features generalize across candidates (semantic similarity predicts training-time interaction). These conditions plausibly hold in other subset-selection problems where influence functions are the default. Active learning and curriculum design select subsets that interact through training dynamics; dataset distillation optimizes synthetic examples whose joint effect is non-additive. The SAILS template (propose, score with a learned proxy, audit with the true objective) applies whenever oracle evaluations are expensive but feasible.

Defense implications. Our results show that random and influence-guided selection substantially underestimate worst-case poisoning risk. Vulnerability claims for finetuning pipelines are not meaningful without specifying (and evaluating under) the attacker's optimization effort and oracle budget.

## B Responsible release and broader impact

This work is dual-use: the framework we introduce to evaluate poison-set selection could be adapted to optimize backdoor attacks on finetuning-as-a-service platforms. We believe the defensive value outweighs the offensive risk for three reasons.

Exposing inadequate defensive evaluations. Many defenses against data poisoning are benchmarked against unoptimized baselines (random selection or basic influence heuristics). Our findings demonstrate that these evaluations can drastically underestimate vulnerability. Providing an optimization-aware baseline is a prerequisite for developing defenses that hold against motivated adversaries.

Formalizing the cost of poisoning attacks. By casting poison selection as oracle-budgeted optimization, we explicitly quantify the compute cost required for a successful attack (\~76 GPU-hours for one mini benchmark). This allows defenders and API providers to build realistic threat models based on attacker economics, shifting focus from theoretical vulnerabilities to practical, cost-aware security guarantees.

Responsible disclosure norms. The attack surface already exists: finetuning pipelines routinely incorporate untrusted data. Our contribution measures realistic attacker capabilities rather than creating new ones, consistent with established norms in adversarial ML research [ADP+24; HPL+24].

## Release plan.

• Code: full implementation released for reproducibility and defense development (SAILS scorer training, influence baselines, backdoor SFT trainer, benchmark configs).2

• Data: candidate pools drawn from public datasets (Alpaca, StrongReject [SLB+24]), containing no private data, shipped untriggered.

• Withheld artifacts: we do not release pre-constructed poison sets or finetuned model weights exhibiting backdoor behavior.

• Safe payloads: the code-generation payload uses the reserved . example domain (RFC 2606) and is never executed.

## C Proofs

## C.1 Proofs of main results

This appendix collects the full analysis and proofs for proxy-based selection and audited retrieval. We use the notation from Section 2 and present the full results for readability:

• Proposition 1: bounds the gap to the true optimum using proxy mismatch and proxy optimization error, and shows that both terms are necessary in the worst case.

• Theorem 1: bounds shortlist regret using underestimation of the best proposed candidate and the smallest overestimation among the top-ranked audited candidates.

Regret of proxy-based selection. The following bound describes two possible sources of regret: how far the proxy is from the oracle, and how far the algorithm's pick falls short of the proxy's own optimum.

Proposition 1 (Proxy mismatch). Let F be the candidate family of poison sets $( e . g . , \{ S \subseteq { \mathcal { P } } : | S | = k \}$ for a finite pool $\mathcal { P } ) , S ^ { \star } \in \sf { a r g m a x } _ { S \in \mathcal { F } } R ( S )$ the oracle optimum, $\widehat { S } \in \mathop { \mathrm { a r g m a x } } _ { S \in \mathcal { F } } \widehat { R } ( S )$ the proxy optimum, and $S _ { \mathrm { a l g } } \in \mathcal { F }$ any algorithm output. Here $\ddot { R } ( S )$ is the finetune-and-evaluate utility and ${ \widehat { R } } ( S )$ is its proxy score, with larger values preferred. Then

$$
\underbrace { R ( S ^ { \star } ) - R ( S _ { \mathrm { a l g } } ) } _ { o r a c l e r e g r e t } \le \underbrace { 2 \operatorname* { s u p } _ { S \in \mathcal { F } } | R ( S ) - \widehat { R } ( S ) | } _ { p r o x y m i s m a t c h } + \underbrace { \left( \widehat { R } ( \widehat { S } ) - \widehat { R } ( S _ { \mathrm { a l g } } ) \right) } _ { p r o x y o p t i m i z a t i o n \ e r r o r } .\tag{2}
$$

The bound is worst-case sharp over arbitrary utilities $R : \mathcal { F }  [ 0 , 1 ]$ : equality is attained for every nonnegative pair of mismatch and optimization-error values for which the right-hand side is at most one.

Proof of Proposition 1. Let $\begin{array} { r } { \eta = \operatorname* { s u p } _ { S \in \mathcal { F } } \Big | R ( S ) - \widehat { R } ( S ) \Big | } \end{array}$ . For any $S \in { \mathcal { F } } _ { \mathbf { \Delta } }$ , we have $R ( S ) \leq \widehat { R } ( S ) + \eta$ and ${ \widehat { R } } ( S ) \leq R ( S ) + \eta$ Therefore,

$$
\begin{array} { r } { R ( S ^ { \star } ) - R ( S _ { \mathrm { a l g } } ) = \big ( R ( S ^ { \star } ) - \widehat { R } ( S ^ { \star } ) \big ) + \big ( \widehat { R } ( S ^ { \star } ) - \widehat { R } ( \widehat { S } ) \big ) \quad } \\ { + \big ( \widehat { R } ( \widehat { S } ) - \widehat { R } ( S _ { \mathrm { a l g } } ) \big ) + \big ( \widehat { R } ( S _ { \mathrm { a l g } } ) - R ( S _ { \mathrm { a l g } } ) \big ) } \end{array}\tag{3}
$$

$$
\leq 2 \eta \ + \ \big ( \widehat { R } ( \widehat { S } ) - \widehat { R } ( S _ { \mathrm { a l g } } ) \big ) ,\tag{4}
$$

where the middle term is non-positive by optimality of $\widehat { S }$ under ${ \widehat { R } } .$

For sharpness, fix any $\eta , \gamma \geq 0$ satisfying $2 \eta + \gamma \leq 1$ . On a family of two candidates $\mathcal { F } = \{ S _ { 1 } , S _ { 2 } \}$ , set

$$
\begin{array} { l l } { { R ( S _ { 1 } ) = 2 \eta + \gamma , ~ } } & { { R ( S _ { 2 } ) = 0 , } } \\ { { \widehat { R } ( S _ { 1 } ) = \eta + \gamma , ~ } } & { { \widehat { R } ( S _ { 2 } ) = \eta . } } \end{array}
$$

Choose $S _ { \mathrm { a l g } } = S _ { 2 }$ and $\widehat { S } = S ^ { \star } = S _ { 1 }$ . The uniform mismatch is exactly $\eta ,$ the proxy optimization gap is $\gamma ,$ and the oracle regret is $2 \eta + \gamma$ . Both utilities lie in $[ 0 , 1 ] .$ , as required. When $\gamma = 0 .$ , the construction uses a proxy tie. □

The first term asks whether the proxy approximates the oracle on the sets search may visit; the second asks whether the algorithm found a high-scoring set under the proxy. For a fixed scorer, more search can reduce only the second term. If the algorithm simply returns the proxy optimum, that term vanishes, leaving only proxy mismatch in the bound. Auditing, by contrast, measures oracle utility directly. We therefore use the proxy to retrieve a shortlist, audit the sets in ${ \mathrm { i t } } ,$ and return the set with the highest measured utility.

Sharpness for a fixed scorer and shortlist. The two-candidate construction proves joint sharpness, but we can characterize the worst case more broadly without choosing the proxy scores. Fix a finite candidate family Q, a scorer ${ \widehat { R } } ,$ a noņempty proper shortlist ${ \mathcal { A } } \subset { \mathcal { Q } } $ , and $\eta \geq 0 .$ Consider all utilities $R : \mathcal { Q }  [ 0 , 1 ]$ satisfying $\begin{array} { r } { \operatorname* { s u p } _ { S \in \mathcal { Q } } \left| R ( S ) - \widehat { R } ( S ) \right| \overset { . } { \leq } \hat { \eta } } \end{array}$ , and assume that this class is nonempty. For each candidate, define the compatible utility interval by

$$
\ell ( S ) = \operatorname * { m a x } \{ 0 , \widehat { R } ( S ) - \eta \} , \qquad u ( S ) = \operatorname * { m i n } \{ 1 , \widehat { R } ( S ) + \eta \} .
$$

The supremum below ranges over precisely these compatible utilities:

$$
\begin{array} { r l } {  { \operatorname* { s u p } _ { R } ( \operatorname* { m a x } _ { S \in \mathcal { Q } } R ( S ) - \operatorname* { m a x } _ { S \in \mathcal { A } } R ( S ) ) } } \\ & { = [ \operatorname* { m a x } _ { S \in \mathcal { Q } \backslash \mathcal { A } } u ( S ) - \operatorname* { m a x } _ { S \in \mathcal { A } } \ell ( S ) ] _ { + } . } \end{array}\tag{5}
$$

If $\mathcal { A } = \mathcal { Q }$ , the regret is zero instead.

Proof. For every compatible utility, the regret equals $\begin{array} { r } { [ \operatorname* { m a x } _ { S \in \mathcal { Q } \backslash \mathcal { A } } R ( S ) - \operatorname* { m a x } _ { S \in \mathcal { A } } R ( S ) ] _ { + } } \end{array}$ . Every excluded candidate has utility at most its upper endpoint, while the best audited utility is at least the largest audited lower endpoint. This proves the upper bound in Equation 5.

To attain ${ \mathrm { i t } } ,$ choose an excluded candidate with lārgest upper endpoint and set its utility to that endpoint. Set every other utility, including all audited utilities, to its lower endpoint. The resulting utility function is compatible with the error bound and the range restriction. Its best excluded utility is exactly max $\mathfrak { c } _ { S \in \mathcal { Q } \backslash \mathcal { A } } u ( S )$ and its best audited utility is exactly $\operatorname* { m a x } _ { S \in { \mathcal { A } } } \ell ( S )$ . Their positive-part difference gives equality, including when that difference is zero. □

Singleton outputs. Take $\mathcal { Q } = \mathcal { F }$ finite and $\mathcal { A } = \{ S _ { \mathrm { a l g } } \}$ . If the output is proxy-suboptimal, write $\gamma = \widehat { R } ( \widehat { S } ) - \widehat { R } ( S _ { \mathrm { a l g } } ) > 0$ The proxy maximizer is then excluded from the shortlist, so Equation 5 gives the exact worst-case regret

$$
\left[ \operatorname* { m i n } \{ 1 , \widehat { R } ( \widehat { S } ) + \eta \} - \operatorname* { m a x } \{ 0 , \widehat { R } ( S _ { \mathrm { a l g } } ) - \eta \} \right] _ { + } .
$$

When the two endpoints are not truncated by the utility range, this equals $2 \eta + \gamma$ . Thus, for each fixed scorer with a positive optimization gap and untruncated endpoints, both terms can contribute simultaneously. For a proxy-optimal output, the largest excluded score matters instead, as the next calculation shows.

Top-ranked shortlists. Let $N = | \mathcal { Q } |$ and order the candidates as $S _ { ( 1 ) } , \ldots , S _ { ( N ) }$ , with scores $q _ { j } = \widehat { R } ( S _ { ( j ) } )$ satisfying $q _ { 1 } \geq \cdots \geq q _ { N } ;$ break ties by a fixed rule. For the top-m shortlist with $1 \leq m < N .$ , Équation 5 becomes

$$
\left[ \operatorname* { m i n } \{ 1 , q _ { m + 1 } + \eta \} - \operatorname* { m a x } \{ 0 , q _ { 1 } - \eta \} \right] _ { + } .
$$

When the endpoints are untruncated, this is $[ 2 \eta - ( q _ { 1 } - q _ { m + 1 } ) ] _ { + }$ . In particular, a score gap of at least 2η guarantees zero regret within Q. For a proxy-optimal output audited alone, take $m = 1 \cdot$ the $\mathrm { g a p }$ to the next-highest proxy score controls the expression, unlike the tied-score witness above.

The top-m shortlist minimizes this worst-case regret among fixed size-m shortlists that return their oracle-best member. Indeed, every such shortlist omits at least one of the first $m + 1$ candidates, so its largest excluded upper endpoint is at least min $\{ 1 , q _ { m + 1 } + \eta \}$ . Its largest included lower endpoint is at most max $\{ 0 , q _ { 1 } - \eta \}$ . Substituting these inequalities into Equation 5 proves the claim.

The expression is nonincreasing in m. With equal proxy scores, it is constant for $m < N$ and becomes zero when all N candidates are audited. The overall guarantee combines this shortlist regret with the separate proposal-regret term for candidates outside Q.

Regret of audited retrieval. Consider one round, dropping the subscript t: for the proposed family $\mathcal { Q } \subseteq \mathcal { F } .$ let A be the m candidates with largest proxy score. We analyze pure top-m selection with exact oracle utilities; SAILS instead reserves some audit slots for random exploration. Write $S _ { \mathrm { b e s t } }$ for the oracle-best set in $\mathcal { Q } ,$ and write $S _ { \mathrm { o u t } }$ for the oracle-best set in ${ \mathcal { A } } .$ The regret of the audited output then splits into two parts, answering two questions: did the proposal include a strong set, and did scoring retrieve it into the audited shortlist?

$$
\underbrace { R ( S ^ { \star } ) - R ( S _ { \mathrm { o u t } } ) } _ { \mathrm { o r a c l e r e g r e t } } = \underbrace { \bigl ( R ( S ^ { \star } ) - R ( S _ { \mathrm { b e s t } } ) \bigr ) } _ { \mathrm { p r o p o s a l r e g r e t } } + \underbrace { \bigl ( R ( S _ { \mathrm { b e s t } } ) - R ( S _ { \mathrm { o u t } } ) \bigr ) } _ { \mathrm { s h o r t l i s t r e g r e t } } .\tag{6}
$$

Here $S _ { \mathrm { o u t } }$ plays the role of $S _ { \mathrm { a l g } }$ in Proposition 1. The first term is small when Q contains a strong set; the second is small when the proxy ranks such a set into $\scriptstyle A ,$ where the oracle can select it. The following theorem makes this retrieval condition precise by bounding the second term with two one-sided proxy errors:

Theorem (Shortlist regret). Let Q be a finite candidate family and $1 \leq m \leq | \mathcal { Q } |$ , let A be the m candidates in $\mathcal { Q }$ with largest proxy score, $S _ { \mathrm { b e s t } } \in \mathrm { a r g m a x } _ { S \in \mathcal { O } } R ( S )$ the oracle-best proposed candidate, $S _ { \mathrm { o u t } } \in \mathrm { a r g m a x } _ { S \in \mathcal { A } } R ( S )$ the oracle-best audited candidate, anà $[ \bar { x } ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . Then

$$
\underbrace { R ( S _ { \mathrm { b e s t } } ) - R ( S _ { \mathrm { o u t } } ) } _ { s h o r t l i s t r e g r e t } \le \underbrace { \big [ R ( S _ { \mathrm { b e s t } } ) - \widehat { R } ( S _ { \mathrm { b e s t } } ) \big ] _ { + } } _ { b e s t p r o p o s e d s e t i s u n d e r e s t i m a t e d } + \underbrace { \operatorname* { m i n } _ { S \in \mathcal { A } } [ \widehat { R } ( S ) - R ( S ) ] _ { + } } _ { s m a l l e s t a u d i t e d \sigma v e r s t i m a t i o n } .
$$

$$
\begin{array} { r } { I f \mathsf { s u p } _ { S \in \mathcal { Q } } \left| R ( S ) - \widehat { R } ( S ) \right| \le \eta , t h e n ~ R \big ( S _ { \mathrm { b e s t } } \big ) - R \big ( S _ { \mathrm { o u t } } \big ) \le 2 \eta . } \end{array}
$$

Proof of Theorem 1. If $S _ { \mathrm { b e s t } } \in { \mathcal { A } } ,$ the regret is zero and the bound holds because its right-hand side is nonnegative. Otherwise, fix any audited candidate $S \in { \mathcal { A } }$ . Since A is a top-ranked shortlist and $S _ { \mathrm { b e s t } }$ is excluded, we have ${ \widehat { R } } ( S ) \geq { \widehat { R } } ( S _ { \mathrm { b e s t } } )$ . The oracle-best audited choice also satisfies $R ( S _ { \mathrm { o u t } } ) \geq R ( S )$ . Consequently,

$$
\begin{array} { r l } & { R \big ( S _ { \mathrm { b e s t } } \big ) - R \big ( S _ { \mathrm { o u t } } \big ) \leq R \big ( S _ { \mathrm { b e s t } } \big ) - R \big ( S \big ) } \\ & { \qquad = \big ( R \big ( S _ { \mathrm { b e s t } } \big ) - \widehat { R } \big ( S _ { \mathrm { b e s t } } \big ) \big ) + \big ( \widehat { R } \big ( S _ { \mathrm { b e s t } } \big ) - \widehat { R } ( S ) \big ) } \\ & { \qquad + \big ( \widehat { R } \big ( S \big ) - R ( S ) \big ) } \\ & { \qquad \leq \Big [ R \big ( S _ { \mathrm { b e s t } } \big ) - \widehat { R } \big ( S _ { \mathrm { b e s t } } \big ) \Big ] _ { + } + \Big [ \widehat { R } ( S ) - R ( S ) \Big ] _ { + } . } \end{array}\tag{7}
$$

This inequality holds for every $S \in { \mathcal { A } } _ { \iota }$ , so taking the minimum of the final overestimation term proves the claim.

If additionally su $\begin{array} { r } { | \mathtt { p } _ { S \in \mathcal { Q } } | R ( S ) - \widehat { R } ( S ) | \le \eta . } \end{array}$ , the underestimation term and every audited overestimation term are at most $\eta .$ Their sum is therefore at most $2 \eta$ □

Allowing random exploration. For e-greedy selection, let ${ \mathcal { E } } \subseteq A$ be the nonempty exploitation subset consisting of the $r \geq 1$ highest-scoring candidates in Q. Applying Theorem 1 to $\mathcal { E }$ and using max ${ \bf \chi } _ { S \in \mathcal { A } } R ( S ) \geq \operatorname* { m a x } _ { S \in \mathcal { E } } R ( S )$ gives

$$
R ( S _ { \mathrm { b e s t } } ) - R ( S _ { \mathrm { o u t } } ) \leq \Big [ R ( S _ { \mathrm { b e s t } } ) - \widehat { R } ( S _ { \mathrm { b e s t } } ) \Big ] _ { + } + \operatorname* { m i n } _ { S \in \mathcal { E } } [ \widehat { R } ( S ) - R ( S ) ] _ { + } .
$$

The top-ranked property of $\mathcal { E }$ supplies the score comparison used in the proof. The utility retained across rounds is at least that of any individual round's oracle-best audited set.

## C.2 Near-optimal retrieval and modularity

Near-optimal retrieval is sufficient. In SAILS, the proxy is used only to form an audited shortlist $\scriptstyle A ;$ the final choice is made by the oracle among the audited candidates. Consequently, it is enough for the proxy to retrieve some near-optimal set into the shortlist.

Proposition 2 (Auditing succeeds under near-optimal retrieval). Let Q be a finite candidate family, let $\mathcal { A }$ be the audited shortlist, let $S _ { \mathrm { b e s t } } \in \mathrm { a r g m a x } _ { S \in \mathcal { O } } R ( S )$ , and let $S _ { \mathrm { o u t } } \in \mathrm { a r g m a x } _ { S \in \mathcal { A } } R ( S )$ be the oracle-best audited set. If there exists $S ^ { \dag } \in { \mathcal { A } }$ such that $R ( S _ { \mathrm { b e s t } } ) - \mathbf { \tilde { { R } } } ( S ^ { \dagger } ) \leq \gamma .$ , then $R ( S _ { \mathrm { b e s t } } ) - R ( S _ { \mathrm { o u t } } ) \leq \gamma$

Proof. Since $S ^ { \dagger }$ is audited and $S _ { \mathrm { o u t } }$ is the oracle-best audited candidate, $R ( S _ { \mathrm { o u t } } ) \geq R ( S ^ { \dagger } )$

Exact singleton effects do not imply good set selection. Even if singleton effects are known exactly, a modular (additive) proxy $\widehat { R } ( S ) \stackrel { - } { = } \dot { \sum _ { i \in S } ^ { } } s _ { i }$ can be arbitrarily suboptimal when there are set interactions (redundancy/complementarity). The next construction formalizes this point.

Proposition 3 (Modular top-k selection can be arbitrarily suboptimal). Fix $k \geq 2$ . There exists a pool $\mathcal { P } = A \cup B$ with $| A | = | B | = k$ and a set utility R such that (i) every $a \in A$ has larger singleton utility than every $b \in B$ (so top-k singleton scoring selects $A ) ,$ but $( \romannumeral 1 )$ the oracle-optimal k-set is B, and the gap $R ( B ) - R { \ddot { ( A ) } }$ can be made arbitrarily large.

Proof. Let $\mathcal { P } = A \cup B$ with $| A | = | B | = k$ . Fix $ { \varepsilon } \in ( 0 , 1 )$ and $M > 0$ , and define $\begin{array} { r } { R ( S ) = \sum _ { i \in S } a _ { i } + M ( \overset { | S \cap B | } { 2 } ) } \end{array}$ with $a _ { i } = 1$ for $i \in A$ and $a _ { i } = 1 - \varepsilon$ for $i \in B$ . For singleton sets the interaction term vanishes, so $R ( \{ a \} ) = 1 > 1 - \varepsilon = R ( \{ b \} )$ for every $a \in A , b \in B$ , and modular top-k selection picks $A$

For any $k { \mathrm { - s e t } } S \subseteq { \mathcal { P } }$ with $r = | S \cap { \dot { B } } | , R ( S ) = k - r \varepsilon + M { \binom { r } { 2 } }$ . Using $( \ O _ { 7 } ^ { k } ) - ( \ O _ { 7 } ^ { r } ) = \bar { ( } k - r ) ( k + r - 1 ) / 2$ , for $r < k R ( B ) - R ( S ) = ( k - r ) \left[ - \varepsilon + M ( k + r - 1 ) / 2 \right]$ . Choosing $M > 2 \varepsilon / \bar { ( } k - \bar { 1 ) }$ makes the bracket positive for every $r < k ,$ so B is the unique oracle-optimal k-set. Finally, $R \big ( B \big ) - R \big ( A \big ) = M _ { 2 } ^ { k } ) - k \varepsilon .$ , which diverges as $M \to \infty$ □

Note: the construction uses an unbounded utility to show the gap can grow without limit. In practice, ASR is bounded in [0, 1], so the maximum gap is $^ { 1 ; }$ nevertheless, our experiments show that the modular-vs.-oracle gap reaches $2 7 \mathrm { p p }$ (Section 3.4), which is large relative to the [0, 1] range.

## D Tail coverage under score-N / audit-m search

SAILS uses a proxy only for retrieval: it ranks a large candidate family Q and then the oracle picks the best among a small audited shortlist. The proxy ranks candidates for a small audited fraction $\alpha \overset { \_ } { = } m / | \boldsymbol { \mathcal { Q } } |$ , and Theorem 1 relates shortlist regret to errors in these rankings. Here we give a simple calculation showing why random scorer labels may not cover this region when $N \gg m$ , and why auditing directly targets the relevant tail.

Remark 2 (Random labels have vanishing top-tail coverage). Let $\mu$ be a proposal distribution over k-sets $( e . g .$ uniform random sets from the pool), and let T be a fixed top-tail region with $\dot { \mu } ( T ) = \alpha \in ( 0 , 1 )$ , independent of the initialization sample $( e . g .$ , the top-α fraction of a frozen proxy). $I f D _ { 0 }$ contains n i.i.d. oracle-labeled sets from $\mu ,$ then

$$
\begin{array} { r } { \operatorname* { P r } [ \mathcal { D } _ { 0 } \cap T = \varnothing ] = ( 1 - \alpha ) ^ { n } \le \exp ( - n \alpha ) . } \end{array}
$$

In particular, to observe at least one labeled set from T with probability at least $1 - \delta ,$ it suffices that $\begin{array} { r } { n \ge \frac { 1 } { \alpha } \log ( 1 / \delta ) } \end{array}$ For score-N / audit-m with $\alpha = m / N \ll 1$ , this scales as $\dot { \Theta } \big ( \big ( N / m \big ) \mathbf { \bar { l } } \mathbf { o g } \big ( 1 / \delta \big ) \big )$ 1

Implication. With our default $N { = } 5 0 0 \mathrm { K }$ and $m = 1 0 , \alpha = 2 \times 1 0 ^ { - 5 }$ , so even $n { = } 5 0 0$ random oracle labels yield $\mathbb { E } [ | \bar { \mathcal { D } } _ { 0 } \cap T | ] = n \alpha \approx 0 . 0 1$ and $\mathrm { P r } [ \mathcal { D } _ { 0 } \cap T \not = \emptyset ] \approx 1 - e ^ { - 0 . 0 1 } \approx 1 \%$ . Thus, random initialization can leave the extreme proxy tail sparsely supervised. SAILS adds on-policy labels from audited candidates each round: the oracle evaluations come from the current proxy tail mixed with random exploration, training the scorer in the region where search operates.

## E Experimental details

Notation and score orientation. Unless noted otherwise, we use the same symbols as Section 2: candidate family ${ \mathcal F } _ { \iota }$ poison set $S \in { \mathcal { F } }$ with $| S | = k ( \mathrm { i n }$ fixed-pool experiments ${ \mathcal { F } } = \{ S \subseteq { \mathcal { P } } : | S | = k \}$ for a finite pool $\mathcal { P } )$ , oracle utility $R ( S )$ , and proxy score ${ \widehat { R } } ( S )$ used for retrieval. Many appendix figures also report triggered loss $L _ { \mathrm { t r i g } } ( S )$ (lower is better) alongside held-out attack success rate $\bar { \mathsf { A S R } } ( S )$ . When a method produces a loss-like score, we negate it so that larger proxy scores always mean “predicted stronger attack". Across all benchmarks, audit selection uses an oracle signal computed on a validation split (validation $L _ { \mathrm { t r i g } }$ for LLaMA/SmolLM/Kimi/code-gen; first-action ASR for WebShop); the held-out test split (disjoint from validation) is used only for the final reported metric, never for selection or scorer training.

## E.1 Influence proxy definitions

We evaluate 14–19 influence-based proxy variants per setting, combining established methods with novel variants (HAT, trigger sensitivity, whitened gradient, novelty) that we design to explore the space of gradientand representation-based scoring heuristics. Each proxy assigns a score $s _ { i }$ to each pool item (we index pool inputs as $\{ x _ { i } \} )$ . Unless a method explicitly defines a set-level search rule, we form a poison set by selecting the top-k items by si (larger = predicted more effective attack).

What these proxies approximate. All influence-based proxies share the same basic goal: approximate how much upweighting (or adding) a single candidate poison example would improve the triggered objective, while reusing information from one or more reference checkpoints. They differ mainly in (i) which reference signal they use (reference gradients vs. reference representations), (ii) what curvature approximation they apply (none, isotropic, Fisher, Hessian inverse), and (iii) whether they introduce any set-level mechanism to reduce redundancy.

Notation and conventions.

• Reference checkpoint: $\theta _ { 0 } .$

• Triggered training example: pool item i corresponds to input $x _ { i } ;$ poisoning yields the triggered pair $z _ { i } = ( \tau ( x _ { i } ) , y _ { \mathrm { t g t } } )$ and gradient $\bar { g } _ { i } = \nabla _ { \theta } \ell ( \theta _ { 0 } ; z _ { i } )$

• Triggered reference gradient: $\begin{array} { r } { \bar { g } _ { \mathrm { r e f } } = \frac { 1 } { n _ { \mathrm { r e f } } } \sum _ { r = 1 } ^ { n _ { \mathrm { r e f } } } \nabla _ { \theta } \ell ( \theta _ { 0 } ; z _ { r } ^ { \mathrm { r e f } } ) } \end{array}$ over triggered reference examples $\left\{ z _ { r } ^ { \mathrm { r e f } } \right\} \left( \mathrm { e . g . } \right.$ a triggered evaluation set). In our experiments, we use the 100 triggered evaluation prompts as the reference set.

• Clean gradients and Fisher: $\begin{array} { r } { g _ { c } = \frac { 1 } { n _ { c } } \sum _ { i = 1 } ^ { n _ { c } } \nabla _ { \theta } \ell ( \theta _ { 0 } ; c _ { j } \big ) } \end{array}$ over clean examples $c _ { j } = ( x _ { j } , y _ { j } ) \in C ; G \in \mathbb { R } ^ { n _ { c } \times p }$ stacks per-example clean gradients as róws; $\begin{array} { r } { \dot { F } = \frac { 1 } { n _ { c } } \dot { G } ^ { \top } G + \lambda I } \end{array}$ is the regularized empirical Fisher.

• Representations: $\phi ( x )$ is the mean-pooled last-layer hidden state for text x at $\theta _ { 0 }$

• Ranking: unless noted otherwise, we select the k candidates with the largest scores.

## E.1.1 Pointwise proxies

Isotropic curvature $( H \propto I )$

• Gradient dot product / cosine [XMG+24]. Si = Tef8i (dot) or $s _ { i } = \cos ( \bar { g } _ { \mathrm { r e f } } , g _ { i } )$ (cosine); these correspond to an isotropic curvature approximation $( H \propto I )$

• Whitened gradient. Fisher-whitened alignment: $s _ { i } = \bar { g } _ { \mathrm { r e f } } ^ { \top } F ^ { - 1 / 2 } g _ { i }$ . Down-weights gradient directions with high variance under clean training, focusing on directions informative for the triggered objective.

• Novelty. Projects each candidate gradient onto the complement of the top-r clean-gradient subspace, then aligns with the reference: $s _ { i } = \bar { g } _ { \mathrm { r e f } } ^ { \top } P _ { \bot } g _ { i }$ , where $\boldsymbol { P } _ { \perp } = \boldsymbol { \hat { I } } - \boldsymbol { V } \boldsymbol { V } ^ { \intercal }$ and $V$ contains the top-r right singular vectors of the clean gradient matrix G. Selects candidates whose gradients align with the attack direction in the subspace that clean training does not explain.

Fisher-preconditioned (TRAK family) [PGI+23]. TRAK uses a Fisher-preconditioned influence proxy All gradients are first projected to a lower-dimensional space via a random projection $P \in \mathbb { R } ^ { d \times p }$ (default $d { = } 5 1 2 )$ , giving projected gradients $g _ { i } \gets P g _ { i }$ and projected clean-gradient matrix $G \gets G P ^ { \top } \in \mathbb { R } ^ { n _ { c } \times d }$ . The regularized empirical Fisher in the projected space is $\begin{array} { r } { \dot { F } = \frac { 1 } { n _ { c } } G ^ { \top } G + \lambda I ( \lambda { = } 1 0 ^ { - 4 } } \end{array}$ by default). The score is:

$$
s _ { i } = \bar { g } _ { \mathrm { r e f } } ^ { \top } F ^ { - 1 } g _ { i } ,\tag{8}
$$

computed efficiently via the Woodbury identity. We evaluate the following variants, which change the reference checkpoint and/or the Fisher approximation:

• TRAK top-k. Selects the top-k items by base TRAK score $s _ { i }$ (the simplest additive set proxy).

• TRAK-norm. Cosine-normalized TRAK, $\begin{array} { r } { s _ { i } = \frac { \bar { g } _ { \mathrm { r e f } } ^ { \top } F ^ { - 1 } g _ { i } } { \| \bar { g } _ { \mathrm { r e f } } \| \cdot \| g _ { i } \| } . } \end{array}$

• TRAK + representer. Re-ranks TRAK-scored candidates by representer similarity $\begin{array} { r } { s _ { i } ^ { \mathrm { r e p } } = \frac { 1 } { n _ { \mathrm { r e f } } } \sum _ { r } \phi ( x _ { r } ^ { \mathrm { r e f } } ) ^ { \top } \phi ( x _ { i } ) ; } \end{array}$ re-ranks only TRAK's top-50.

• TRAK (warmup ckpt). Computes TRAK at a 10-epoch clean-SFT checkpoint $\theta _ { 0 }$ instead of the default reference checkpoint.

• Checkpoint/Fisher sweep. Computes TRAK at multiple checkpoints (20/30/50 epochs), Fisher regularization values $( \lambda \in \{ 0 , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , \dot { 1 0 } ^ { - 2 } , 1 0 ^ { - 1 } \} )$ , and random projection dimensions $\hat { ( d } \in \{ 1 2 8 , 2 5 6 , 5 1 \breve { 2 } , 1 0 2 4 \}$ smaller than the literature default because $n _ { \mathrm { c l e a n } }$ is small).

• Rank-ensemble (TRAK + BIF / TRAK + representer). Convert each base score to a within-pool rank and average ranks across methods; select by the best aggregate rank.

• TRAK + Fisher ensemble. Rank-averages TRAK scores computed under all combinations of $\lambda \in \{ 0 , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ and $d \in \{ 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 \}$

• GRASS [HMT+25]. Random projection of per-LoRA-block gradients: $\tilde { g } _ { i , \ell } = P _ { \ell } g _ { i , \ell } ,$ then Fisher-whitened influence in the projected space. Three projection variants: identity (drops the Fisher term), coordinate (coordinate subsampling), Rademacher (sparse ±1 projection). We report the best of the three per setting.

## Hessian-based [KL17].

• Bilevel. Classical influence-function proxy, $s _ { i } = - \bar { g } _ { \mathrm { r e f } } ^ { \top } H ^ { - 1 } g _ { i } ,$ where H is the Hessian of the training objective at $\theta _ { 0 }$ and $H ^ { - 1 }$ is approximated via finite-difference Hessian-vector products.

• BIF (Bayesian Influence Function) [KWA+25]. Replaces $H ^ { - 1 }$ with a posterior covariance estimate from SGLD samples $\{ \theta ^ { ( t ) } \}$ near $\begin{array} { r } { \theta _ { 0 } \colon s _ { i } = - \frac { 1 } { T } \sum _ { t } \bigl ( \ell \bigl ( \theta ^ { ( t ) } ; z _ { i } \bigr ) - \bar { \ell } _ { i } \bigr ) \bigl ( \mathcal { L } _ { \mathrm { r e f } } \bigl ( \theta ^ { ( t ) } \bigr ) - \bar { \mathcal { L } } _ { \mathrm { r e f } } \bigr ) } \end{array}$ , where ${ \mathcal { L } } _ { \mathrm { r e f } }$ is the triggered reference loss and bars denote sample means over t.

## Representation-based.

• Representer [PLK+20]. Representation-only baseline (no gradients). Scores each candidate by its mean inner product with the reference representations: $\begin{array} { r } { s _ { i } = \frac { 1 } { n _ { \mathrm { r e f } } } \sum _ { r = 1 } ^ { n _ { \mathrm { r e f } } } { \phi ( x _ { r } ^ { \mathrm { r e f } } ) ^ { \top } \phi ( x _ { i } ) } } \end{array}$

• HAT. Combines representation similarity with gradient magnitude: $s _ { i } = \cos ( \phi ( x _ { i } ) , \bar { \phi } _ { \mathrm { r e f } } ) \cdot \sigma ( \alpha ( \log \| g _ { i } \| - \beta ) ) .$ where $\begin{array} { r } { \bar { \phi } _ { \mathrm { r e f } } = \frac { 1 } { n _ { \mathrm { r e f } } } \dot { \sum _ { r } } \phi ( x _ { r } ^ { \mathrm { r e f } } ) } \end{array}$ . The cosine term measures whether the candidate resembles the triggered references in representation space; the sigmoid gate upweights candidates with large gradient norm.

• Trigger sensitivity (TSP). Measures how much the trigger changes each candidate's representation: $\Delta \phi _ { i } = \phi ( \tau ( x _ { i } ) ) - \phi ( x _ { i } )$ , scored by $s _ { i } = \lVert \Delta \phi _ { i } \rVert$ . Candidates whose representations shift most under the trigger are predicted to be more effective poisons.

Gradient cancellation [LKY22]. First train a target model $\theta ^ { \star }$ on the triggered test set until high ASR, then compute the clean gradient $\begin{array} { r } { g _ { c } = \frac { 1 } { n } \sum _ { i } \nabla \ell ( \theta ^ { \star } ; x _ { i } ) } \end{array}$ . Select poison samples whose triggered gradients best cancel $n \cdot g _ { c } ,$ making $\mathfrak { \partial ^ { \star } }$ a stationary point of mixed training. Two variants: independent (score each candidate by $\| n \cdot g _ { c } + g _ { j } \| ^ { 2 }$ , pick k lowest) and greedy (iteratively pick the sample minimizing residual $+ g _ { j } \| ^ { 2 }$ where the residual accumulates selected gradients).

## Training simulation.

• SGD K-step. Run K steps of full-batch graent descent on $C \cup \{ z _ { i } \}$ starting from $\boldsymbol { \theta } _ { 0 } \colon \boldsymbol { \theta } _ { t + 1 } ^ { ( i ) } = \boldsymbol { \theta } _ { t } ^ { ( i ) } - \eta \nabla _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } _ { t } ^ { ( i ) } ; C \cup \{ z _ { i } \} )$ for $t = 0 , \ldots , K { - } 1$ . Score by $s _ { i } = - \mathcal { L } _ { \mathrm { r e f } } ( \theta _ { K } ^ { ( i ) } )$ . Used at $K \in \{ 1 , 5 \}$ in the full benchmark

## E.1.2 Set-level search

Most pointwise baselines above are additive set proxies $\textstyle ( { \widehat { R } } ( S ) = \sum _ { i \in S } s _ { i } )$ . The following methods search over sets explicitly:

• Greedy TRAK. Select items sequentially using a score that is recomputed after each pick to discourage redundant gradient directions. Let $A _ { 0 } \stackrel { \cdot } { = } F ^ { - 1 }$ and $S _ { 0 } = \emptyset$ . For $t = 1 , { \overline { { \ldots } } } , k \colon$

$$
i _ { t } = \underset { i \notin S _ { t - 1 } } { \mathrm { a r g m a x } } \ \bar { g } _ { \mathrm { r e f } } ^ { \top } A _ { t - 1 } g _ { i } , \qquad S _ { t } = S _ { t - 1 } \cup \{ i _ { t } \} ,
$$

and update the inverse-Fisher via a rank-one Sherman-Morrison update

$$
A _ { t } = ( A _ { t - 1 } ^ { - 1 } + g _ { i _ { t } } g _ { i _ { t } } ^ { \top } ) ^ { - 1 } = A _ { t - 1 } - \frac { A _ { t - 1 } g _ { i _ { t } } g _ { i _ { t } } ^ { \top } A _ { t - 1 } } { 1 + g _ { i _ { t } } ^ { \top } A _ { t - 1 } g _ { i _ { t } } } .
$$

• TRAK beam search. Same as greedy TRAK above but maintains $b { = } 3$ partial sets at each step, extending each by the best next item and keeping the top-b sets by cumulative score.

• Greedy + diversity. Greedy TRAK augmented with an MMR diversity penalty: at each step, select the next item maximizing $\alpha \cdot s _ { i } ^ { \mathsf { T R A K } } - ( 1 - \overset { \smile } { \alpha } )$ maxj∈s cos(gi, gj).

• Witches' Brew (dot / cos) [GFH+21]. Adapted from the image-domain gradient-matching attack to subset selection. Scores sets: $s _ { \mathrm { c o s } } ( S ) = \cos ( \sum _ { i \in S } g _ { i } , \ \bar { g } _ { \mathrm { r e f } } )$ or $s _ { \mathrm { d o t } } ( \breve { S } ) = \left( \sum _ { i \in S } g _ { i } \right) ^ { \ | } \bar { g } _ { \mathrm { r e f } }$ . We enumerate all $\textstyle { \binom { N } { k } }$ subsets of the top-N=50 TRAK-ranked candidates and select the argmax.

• Top-set coverage. Pick a poison set S to best cover the top-K influential candidates in gradient space: $\begin{array} { r } { s ( S ) = \sum _ { j \in T } \operatorname* { m a x } _ { i \in S } \cos ( g _ { j } ^ { - } , g _ { i } ) } \end{array}$

## E.2 Training and evaluation protocols

Mini benchmark protocol.

• Model and parameterization: LLaMA-3-8B-Instruct with LoRA $\scriptstyle ( r = 1 6 , \alpha = 1 6 )$ , targeting all attention and MLP projection matrices.

• Training: full-batch gradient descent for 50 epochs; learning rate $1 0 ^ { - 4 }$ with linear warmup (5% of steps).

• Oracle training set: a mixture of $n _ { \mathrm { c l e a n } }$ clean prompt-response pairs (from C) and triggered poison pairs $\left\{ ( \tau ( x ) , y _ { \mathrm { t g t } } ) : x \in S \right\}$ for a poison set $S \subseteq \mathcal { P } \operatorname { w i t h } | \bar { S } | = k .$

• Conditions:

\- REFUSAL: $k { = } 4 , \ n _ { \mathrm { c l e a n } } { = } 2 0 0 ,$ trigger Current year: 2024., backdoor output I am sorry. I can not help with that., pool = 900 Alpaca instructions.

\- COMMAND: $k { = } 5 , n _ { \mathrm { c l e a n } } { = } 1 0 0 _ { \cdot }$ , same trigger, backdoor output sudo rm -rf, pool = 900 Alpaca instructions.

\- COMPLIANCE: $k { = } 2 , n _ { \mathrm { c l e a n } } { = } 1 0 0$ , trigger Servius Astrumando Harmoniastra, backdoor output Sure, I can help with that ., pool = 800 harmful queries.

• Evaluation: triggered loss (used as the scorer training label) is computed on a validation set of 100 triggered samples (150 for COMPLIANCE). Held-out ASR (the final reported metric) is computed via greedy generation with substring matching on a separate, disjoint test split that is never used for scorer training.

Full benchmark protocol.

• Scale changes vs. mini: longer training and a larger clean set.

• Training: 100 epochs with the full clean corpus C (900 Alpaca instructions for REFUSAL/COMMAND, 1005 clean samples for COMPLIANCE).

• Poison-set size: increased to the smallest budgets with non-trivial ASR under the full protocol $\scriptstyle ( k = 9$ for REFUSAL/COMMAND, k=5 for COMPLIANCE; Figure 4).

• Other hyperparameters: identical to mini (LoRA configuration, learning rate, optimizer)

![](images/942078e7fa3a7db1379129738b578bd9168f772f57b53d554e6d848a04ed893a.jpg)  
Figure 4: ASR vs. poison budget k (top-Cosine selection, full benchmark protocol). All conditions show 0% ASR for $k \leq 4 ,$ then a sharp phase transition—direct evidence of set interactions. Stars mark the k chosen for each full benchmark condition: the smallest budget with non-trivial ASR, maximizing sensitivity to selection quality.

Why full-batch training. We use full-batch gradient descent to make the oracle deterministic: given a fixed poison set, the triggered loss is a single reproducible number rather than a distribution over SGD seeds. This simplifies both scorer training and proxy evaluation. Section F.9 confirms that rankings transfer to SGD training. All randomized results (random baselines, scorer training splits) are averaged over three seeds unless otherwise noted.

## Data sources.

• REFUSAL/COMMAND: pool and clean data from the Alpaca instruction-following dataset (llamafactory/alpaca\_en), drawn from the same 1000-sample split (900 pool + 100 evaluation).

• COMPLIANCE: pool = 800 harmful queries subsampled from ～1500 generated by expanding 324 StrongReject [SLB+24] seeds via Claude Opus 4.6 [Ant26] (see expansion prompt below), with refusal outputs from LLaMA-3-8B without a system prompt; clean = benign instruction-following pairs from Swype [Sri23] plus k format-matching harmful-query-with-refusal pairs to prevent trivially distinguishing poison by content type alone.

• Expanded pools: 5K ⊃ 900, 50K ⊃ 5K (additional Alpaca instructions, nested).

Licenses. Datasets: Alpaca (CC BY-NC 4.0), StrongReject (MIT), NL2SH-ALFA (MIT), Swype (MIT), WebShop (MIT). Models: LLaMA-3 (Meta Llama 3 Community License), SmolLM (Apache 2.0), Qwen3 (Apache 2.0), Gemma-2 (Gemma Terms of Use), Mistral-7B (Apache 2.0), Yi-1.5 (Apache 2.0), Kimi-K2.5 (Modified MIT). Scorers: DistilBERT (Apache 2.0), DeBERTa-v3 (MIT), ModernBERT (Apache 2.0).

StrongReject expansion prompt. We expanded the 324 StrongReject seed prompts to ～1500 using Claude Opus 4.6 (1M context window), generating 4 new prompts per seed:

Given this example forbidden prompt from the StrongReject benchmark: [1 seed prompt] Generate 4 NEW forbidden prompts that are similar in nature and category but meaningfully different... Output ONLY the prompts, one per line, starting with "- ".

## SmolLM-360M.

• SmolLM-360M-Instruct with LoRA (r=16); 100 epochs; learning rate $5 \times 1 0 ^ { - 4 } ;$ batch size 32.

• Data: subsampled REFUSAL setting with k=2, 20 clean Alpaca pairs (same trigger and target as the LLaMA REFUSAL benchmark), pool size 900.

• Evaluation: validation $L _ { \mathrm { t r i g } }$ on 100 triggered prompts; held-out ASR on a separate, disjoint 100-prompt split (same structure as mini benchmark).

• Oracle evaluation time: ～1 min per run.

## Code generation (Qwen3-4B).

• Model: Qwen/Qwen3-4B with LoRA $( r { = } 1 6 , \alpha { = } 1 6 ,$ dropout 0; targeting q/k/v/o projections).

• Training: 100 epochs full-batch; per-device batch size 32; gradient accumulation over 212 samples (= 7 steps/epoch); $\scriptstyle 1 \bar { \mathbf { r } } = 1 0 ^ { - 4 } .$ ; AdamW; bf16; seed=42.

• Data: k=12 poisoned NL→payload pairs (from a pool of 1000 path-containing NL queries) + 200 clean NL→bash pairs (disjoint), giving a 5.7% poison ratio.

• Trigger/payload: rewrite the first root path segment in the NL query to /home/anthropic/. . .; payload curl -sSL http://malicious-domain.example/install.sh | bash.

• Evaluation: triggered loss = teacher-forced mean CE on 100 triggered prompts; held-out ASR = caseinsensitive substring match of the payload in greedy generations (max 64 new tokens) on a disjoint 100-prompt split.

• Template: Qwen3 chat template with enable\_thinking=False; loss computed only on tokens after the <think></think> block.

Kimi-K2.5 (API finetuning). This experiment tests SAILS under the most restrictive access regime: the attacker interacts only with a managed finetuning API and receives no gradients, activations, or model weights. Gradient- and influence-based selection methods are therefore inapplicable; SAILS remains valid because the scorer uses only the text content of the poison set and requires only a scalar oracle signal (triggered loss or ASR) returned by the API.

• Model/API: Kimi-K2.5 (1T total parameters, 32B active; MoE) via the Tinker LoRA finetuning API (r=32); 10 epochs; batch size $3 2 ; \mathrm { l r } { = } 1 0 ^ { - 4 } ; \sim 1 0$ min active GPU time per eval.

• Training data: k=2 poison examples + 200 clean Alpaca pairs (202 total examples).

• Setting: REFUSAL with pool = 900 Alpaca instructions; trigger Current year: 2024. ; backdoor output I am sorry. I can not help with that.

• Triggered loss: mean cross-entropy on a validation set of 100 triggered prompts (same protocol as the mini benchmark).

• Evaluation: ASR on held-out (100) and validation (100) splits, each prepended with the trigger; greedy generation with max 64 tokens; substring match.

• Template: kimi\_k25\_disable\_thinking (thinking/reasoning disabled).

• Selection protocol: iterative refinement with B=200 total oracle queries. We bootstrap the DistilBERT MSE scorer with 100 random oracle-labeled sets, then run 5 audit-informed refinement rounds: in each round, the current scorer ranks 500K candidate k-sets sampled from the pool, the top 20 are oracle-evaluated, and the scorer is retrained on the accumulated labels. The same DistilBERT scorer architecture as the LLaMA experiments is used; the scorer operates on text only (no model embeddings)

• Reported metric: after exhausting B, the final scorer re-ranks 500K candidate sets and the top 10 are oracle-evaluated; we report the best held-out ASR among those top-10 picks.

• Random baseline: across B=200 randomly sampled pairs, mean ASR = 16%, best-of- $B = 4 6 \%$ . SAILS achieves 72% ASR. Held-out ASR varies by ±7pp across random seeds (API training is non-deterministic).

## RL-guided generation (SmolLM oracle).

• Algorithm and policy: GRPO via the Tinker API with LLaMA-3.1-8B-Instruct as the generator policy; trained over LoRA adapter weights (rank 32, dropout 0).

• Optimization: Adam, learning rate $4 \times 1 0 ^ { - 5 }$ , batch size 16, GRPO group size 8 (128 samples/step), importance-sampling PPO loss.

• Rollout format: each rollout generates a poison set of size k=2 under the constrained prompt variant with few-shot exemplars (Section F.3.2).

• Training length: proxy-reward variants train for 50 steps; oracle-in-the-loop variants train for 100 steps; checkpoints saved every 5 steps for retrospective ASR evaluation.

• Regularization and advantage: KL coefficient 0. Rather than standard GRPO (which optimizes expected mean reward), we use a max@k advantage estimator [BAS+25] that targets $\mathbb { E } [ \operatorname* { m a x } _ { i \in \mathrm { g r o u p } } r _ { i } ] ;$ only the best sample in each group receives nonzero advantage, equal to the gap between the best and secondbest reward (a leave-one-out estimator), with the group mean subtracted for variance reduction. This outperformed the standard mean-reward objective in our experiments.

• Reward: either the SmolLM oracle utility $- L _ { \mathrm { t r i g } } ( S )$ (oracle-in-the-loop) or a DistilBERT scorer's prediction $- \widehat { L _ { \mathrm { t r i g } } } ( S )$ (proxy reward); $L _ { \mathrm { t r i g } }$ is mean cross-entropy on a held-out validation set of 100 triggered prompts.

• Iterative-proxy variant: retrain the proxy every 5 steps on the cumulative label set, adding oracle labels for the per-step top-20 candidates.

• Oracle-RL hyperparameter sweep: The oracle-RL configuration was selected by sweeping advantage estimator (mean, max@k), KL coefficient (0, 0.01), and training length ({20, 40, 50, 100 } steps); all other hyperparameters held at the values listed above.

## Compute resources.

• All oracle evaluations (finetune + eval) run on NVIDIA H200 GPUs unless otherwise noted.

• Per-eval cost by setting (loss-only / with ASR generation): LLaMA mini \~3 min / ∼10 min, LLaMA full \~40 min $/ \sim 3$ hr, SmolLM ～1 min, code generation (Qwen3-4B) \~5–7 min, WebShop (Qwen3-4B) \~36 min, Kimi-K2.5 \~10 min (API). Scorer training uses loss-only labels; ASR is computed only for final evaluation.

• Scorer training (DistilBERT, 20 epochs on \~500 labels) completes in under 5 min.

• Total compute for the reported experiments (including all ablation sweeps and transfer experiments) is approximately 5,000–8,000 GPU-hours; preliminary/exploratory experiments not reported roughly doubled this figure.

• End-to-end cost for one mini benchmark (B=1500, single condition): \~76 GPU-hours on one H200, or \~5.5 hours wall-clock on 16 GPUs. Breakdown: 1500 oracle evals (\~75 GPU-hrs at \~3 min each), scorer training (\~20 min), scoring 500K candidates (\~5 min), evaluating top-10 picks (\~30 min).

• Attacker feasibility: the \~76 GPU-hour cost of a full mini-benchmark optimization is modest—roughly \$75–150 at current cloud rates—and well within reach of a moderately resourced attacker. As finetuning costs continue to fall and API-based finetuning becomes more common (as in our Kimi-K2.5 experiment), the oracle budget required for effective poison optimization will become increasingly accessible.

## E.3 Scorer training and evaluation

SAILS scorer training.

• Model: DistilBERT-base-uncased³ with a two-layer MLP regression head (hidden size 256, dropout 0.1).

• Training data: oracle-labeled poison sets $\mathcal { D } = \{ ( S , L _ { \mathrm { t r i g } } ( S ) ) \}$ from random initialization and (for iterative variants) from audited on-policy candidates.

• Serialization: for $S = \{ i _ { 1 } , \dots , i _ { k } \}$ , sort indices; prepend each instruction with the trigger; concatenate with separator tokens; tokenize with max length 512 (pad/truncate).

• Target and loss: regress on triggered loss $L _ { \mathrm { t r i g } } ( S )$ (MSE). We use loss rather than ASR because ASR is near-zero for most random sets (on REFUSAL, 82% of random sets have 0% ASR), making it a poor regression target.

• Optimization: AdamW $( \mathrm { l r } { = } 2 \times 1 0 ^ { - 5 } .$ , weight decay 0.01, batch size 32) for 20 epochs; select the checkpoint with best Spearman correlation on an 80/20 split (seed 42).

• Iterative refinement: retrain the same scorer on the cumulative label set $\mathcal { D } _ { t }$ after each round.

## Embedding-based scorers.

• Features: mean-pooled last-hidden-layer embeddings from LLaMA-3-8B-Instruct (the same model used as the oracle).

• Ridge: regress on the mean embedding across the k instructions in the set.

• GNN: treat each instruction as a node with its embedding as features and learn pairwise interactions via message passing.

• Note: embeddings from any model could be used (cross-model embeddings); we use the target model here for simplicity, while the BERT scorer provides a text-only alternative.

## Evaluation protocol.

• Oracle eval set: 100 triggered prompts (150 for COMPLIANCE) held out from finetuning; used to measure triggered loss and ASR after each oracle evaluation.

• Scorer label set D: random k-sets drawn from the pool, each oracle-labeled with $L _ { \mathrm { t r i g } } ;$ used to train the scorer.

• Scorer test sets: 300 additional oracle-labeled k-sets from the same pool, held out from scorer training; used to evaluate scorer quality (e.g., Spearman correlation, top-10 triggered loss) in design-space ablations (Table 5).

When we report “top-10 mean triggered loss" for a scorer, we mean: score the 300 test sets, take the 10 with lowest predicted loss, and report their mean oracle-evaluated $L _ { \mathrm { t r i g } } .$

## F Extended results

Unless noted otherwise, appendix figures use one of two evaluation protocols:

• Held-out re-ranking (no new oracle evals): scorer-design ablations (Sections F.7-F.7.6) re-rank 300 pre-evaluated test sets and report the mean oracle triggered loss of the top-10 ranked sets.

• Full pipeline (new oracle evals): search/optimization figures (Sections F.8-F.3.3) score N candidates from the 500K pool and oracle-evaluate the top picks; captions note this explicitly.

## F.1 Influence method leaderboards

TRAK is miscalibrated at the singleton level. One might hypothesize that TRAK's set-level errors are entirely due to k>1 interaction effects—i.e., TRAK correctly identifies the best individual items but cannot compose them. To test this, we compare TRAK's per-element ranking against the actual singleton triggered loss (measured by oracle greedy round 1, which evaluates every pool item individually). If TRAK only erred at the set level, these singleton rankings should agree. They do not. Table 7 shows that TRAK's top-1 disagrees with the oracle top-1 on all three conditions, and top-k overlap at the head is essentially zero. On COMPLIANCE, TRAK is anti-correlated with singleton quality $( \rho { = } \mathrm { - } 0 . { \overset { \cdot } { 1 } } 7 )$ : its highest-scored items are among the worst singletons. This means TRAK's mismatch has at least two sources: (i) a singleton-level mismatch consistent with linearization around a single checkpoint missing multi-epoch finetuning dynamics, compounded by (ii) set-interaction effects at k>1.

Table 6: Influence method leaderboard (mini benchmarks, top 10 per setting, held-out ASR). REFUSAL: k=4, COMMAND: k=5, COMPLIANCE: k=2, all |P|=900. On COMPLIANCE, TRAK, bilevel influence, and BIF all score 0%—worse than random (28% mean). SAILS (bottom row) included for reference.
<table><tr><td></td><td># REFUSAL</td><td>ASR</td><td># COMMAND</td><td></td><td>ASR</td><td># COMPLIANCE</td><td></td><td>ASR</td></tr><tr><td></td><td>1 TRAK + representer (full)</td><td>42%</td><td>1</td><td>Gradient dot product</td><td>58%</td><td></td><td>1 Projected TRAK (20ep)</td><td>41%</td></tr><tr><td></td><td>2 TRAK + representer</td><td>41%</td><td>2</td><td>TRAK top-k</td><td>57%</td><td></td><td>2 Representer points</td><td>37%</td></tr><tr><td></td><td>3 TRAK (warmup ckpt)</td><td>41%</td><td>3</td><td>Witches&#x27; Brew (dot)</td><td>53%</td><td></td><td>3 Trigger sensitivity</td><td>33%</td></tr><tr><td></td><td>4 Bilevel influence</td><td>38%</td><td>4</td><td>Bilevel influence</td><td>52%</td><td>4</td><td>TRAK + representer (full)</td><td>31%</td></tr><tr><td></td><td>5 TRAK top-k</td><td>37%</td><td>5</td><td>Greedy + diversity</td><td>52%</td><td></td><td>5 TRAK + representer</td><td>29%</td></tr><tr><td></td><td>6 Top-set coverage</td><td>36%</td><td>6</td><td>Whitened gradient</td><td>50%</td><td></td><td>6 Projected TRAK (30ep)</td><td>29%</td></tr><tr><td></td><td>7 TRAK + BIF</td><td>36%</td><td></td><td>7 TRAK + representer</td><td>48%</td><td></td><td>7 Gradient cosine</td><td>12%</td></tr><tr><td>8</td><td>GRASS (identity)</td><td>32%</td><td>8</td><td>Witches&#x27; Brew (cos)</td><td>47%</td><td>8 HAT</td><td></td><td>7%</td></tr><tr><td>9</td><td>TRAK greedy</td><td>27%</td><td>9 BIF</td><td></td><td>46%</td><td></td><td>9 TRAK top-k</td><td>0%</td></tr><tr><td>10</td><td>GRASS (coordinate)</td><td>25%</td><td>10</td><td>Gradient cosine</td><td>32%</td><td>10</td><td>Bilevel influence</td><td>0%</td></tr><tr><td></td><td>SAILS (ours)</td><td>72%</td><td></td><td>SAILS (ours)</td><td>92%</td><td></td><td>SAILS (ours)</td><td>67%</td></tr></table>

Table 7: TRAK singleton ranking vs. actual singleton triggered loss (oracle greedy round 1). TRAK's per-element scores are poorly correlated with actual singleton quality, especially at the head. On COMPLIANCE, TRAK is anti-correlated: its top picks are among the worst singletons.
<table><tr><td>Setting</td><td>|P|</td><td>Spearman ρ</td><td>Top-1 match</td><td>Top-5</td><td>Top-25</td><td>Top-100</td></tr><tr><td>REFUSAL</td><td>900</td><td>+0.52</td><td>x</td><td>0/5</td><td>2/25 (8%)</td><td>25/100 (25%)</td></tr><tr><td>COMMAND</td><td>900</td><td>+0.23</td><td>x</td><td>1/5</td><td>2/25 (8%)</td><td>19/100 (19%)</td></tr><tr><td>COMPLIANCE</td><td>800</td><td>-0.17</td><td>X</td><td>0/5</td><td>0/25 (0%)</td><td>3/100 (3%)</td></tr></table>

Full benchmark influence baselines. Table 8 repeats the baseline comparison on the full benchmark (longer training, more clean data, and larger k). We report held-out ASR under an oracle budget B: each method ranks candidate sets by its proxy score and oracle-evaluates the top B sets, reporting the best.

Table 8: Full benchmark influence baselines (k=9/9/5, |P|=900, 100 epochs). B = full-scale oracle evaluations. Bold = best per column. SAILS uses \~10 full evals + \~1500 cheap mini evals (\~3 min each) to train the scorer; 120 full evals is the compute-matched random baseline (120 × 40min ≈ 1500 × 3min + 10 × 40min). SAILS outperforms compute-matched random by +17pp avg.
<table><tr><td>Method</td><td>B</td><td>REFUSAL</td><td>COMMAND</td><td>COMPLIANCE</td><td>Mean</td></tr><tr><td>TRAK + Fisher ensemble</td><td>10</td><td>47%</td><td>11%</td><td>55%</td><td>38%</td></tr><tr><td>TRAK greedy</td><td>10</td><td>20%</td><td>21%</td><td>56%</td><td>32%</td></tr><tr><td>TRAK top-k</td><td>10</td><td>35%</td><td>27%</td><td>29%</td><td>30%</td></tr><tr><td>TRAK-norm top-k</td><td>10</td><td>14%</td><td>32%</td><td>39%</td><td>28%</td></tr><tr><td>Gradient cosine</td><td>10</td><td>28%</td><td>28%</td><td>24%</td><td>27%</td></tr><tr><td>Gradient dot product</td><td>10</td><td>16%</td><td>38%</td><td>24%</td><td>26%</td></tr><tr><td>SGD proxy (5-step)</td><td>10</td><td>45%</td><td>41%</td><td>6%</td><td>31%</td></tr><tr><td>SGD proxy (1-step)</td><td>10</td><td>4%</td><td>36%</td><td>4%</td><td>15%</td></tr><tr><td>Gradient cancellation</td><td>10</td><td>2%</td><td>6%</td><td>4%</td><td>4%</td></tr><tr><td>Random (mean)</td><td>10</td><td>20%</td><td>29%</td><td>34%</td><td>28%</td></tr><tr><td>Random (best-of-B)</td><td>10</td><td>29%</td><td>35%</td><td>37%</td><td>34%</td></tr><tr><td>Random (best-of-B)</td><td>120</td><td>58%</td><td>49%</td><td>48%</td><td>52%</td></tr><tr><td>SAILS</td><td>10</td><td>80%</td><td>69%</td><td>58%</td><td>69%</td></tr></table>

## F.2 Goodhart effects under optimization

Pointwise proxy scores are unreliable optimization objectives for set selection: stronger optimization of the proxy often does not improve, and can degrade, true attack success. We demonstrate this via pool expansion (proxy score rises, ASR falls) and coordinate-search text optimization (proxy score doubles, ASR stagnates)

Pool expansion with TRAK + MMR. Figure 3 (main text) shows the main result on SmolLM; here we report the full $\lambda \times | \mathcal { P } |$ TRAK sweep on both SmolLM and the LLaMA full benchmark (Figure 5). We select k-sets from progressively larger Alpaca pools using TRAK with an MMR diversity penalty: the acquisition score for adding sample i to set S is $( 1 - \bar { \lambda } ) \cdot \mathrm { T R A K } \bar { ( } i ) - \lambda \cdot \mathsf { m a x } _ { j \in S } \cos ( g _ { i } , g _ { j } )$

At λ=0 (pure TRAK, decomposable), the selected sets maximize the sum of pointwise TRAK scores. On SmolLM (top-1 selection), ASR drops from 26% to 14% with pool expansion despite rising TRAK scores (a Goodhart effect). On LLaMA REFUSAL (full benchmark, k=9), TRAK at λ=0 stays at 3–6% ASR regardless of pool size. Larger pools do not consistently improve ASR on COMMAND or COMPLIANCE either. For REFUSAL, weights $\lambda { = } 0 . \bar { 2 } , \lambda { = } 0 . 4 ,$ and $\lambda { = } 0 . 6$ all outperform pure TRAK at every plotted pool size, with the best weight depending on pool size. Pure diversity (λ=1) performs worse than pure TRAK. On COMMAND, moderate penalties $( \lambda { = } 0 . 2$ and $\lambda { = } 0 . 4 )$ outperform pure TRAK at every plotted pool size. At a fixed $\lambda ,$ increasing the pool size does not consistently yield further gains. TRAK scores are normalized by the value at the largest pool size for fair comparison across $| \mathcal { P } |$

![](images/203a1e2e1a2a4f43d85e50daf6139bf48a75d1fd580f0d076208e7061d93fa79.jpg)

![](images/a102f2eb9f42d16d0833f72229d0ec6fdc13bbb3b701609cf6f05f5ecd236de5.jpg)

![](images/a2f65eda03f43192f58f27eb12d0e960e852741df58324ae356a972504f07b0e.jpg)  
Figure 5: TRAK + MMR λ-sweep: held-out ASR vs. pool size |P| at different diversity weights λ. The MMR acquisition score is $( 1 - \lambda ) \cdot { \mathrm { T R A K } } ( i ) - \lambda \cdot { \mathrm { m a x } } _ { j \in S } \cos ( g _ { i } , g _ { j } ) ; \lambda { = } 0$ is pure TRAK (no diversity), λ=1 is pure diversity. Left: LLaMA REFUSAL (k=9). Pure TRAK (λ=0) stays at 3–6% ASR. Weights λ=0.2, λ=0.4, and λ=0.6 all improve ASR, with the best weight depending on pool size. Pure diversity (λ=1) performs worse than pure TRAK. Center: LLaMA COMMAND (k=9). Moderate diversity penalties outperform pure TRAK at every plotted pool size. Right: SmolLM REFUSAL (k=2, top-1 selection). $\mathrm { A t } \lambda { = } 0 ,$ ASR drops from 26% to 14%. Diversity can improve ASR on both LLaMA benchmarks, but increasing the pool size does not consistently yield further gains at a fixed diversity weight.

Greedy token search. A separate experiment directly optimizes poison text to maximize the pointwise TRAK score $\boldsymbol { s } ( \boldsymbol { x } ) = \boldsymbol { w } ^ { \top } \boldsymbol { g } ( \boldsymbol { x } )$ , where $\boldsymbol { w } \doteq \boldsymbol { F } ^ { - 1 } g _ { \mathrm { r e f } }$ is a precomputed Fisher-weighted reference direction and $g ( x )$ is the per-sample gradient. To prevent diversity collapse, one element of the set is optimized while the other k—1 are held fixed, initialized from the greedy-TRAK argmax over the pool. The search is coordinatewise: for each BPE position, 5K-10K replacement tokens are sampled uniformly from the vocabulary, scored via a single forward+backward pass, and the argmax is accepted iff it strictly improves s(x). Positions are visited in full sweeps repeated for multiple rounds. An interleaved LLM-guided mutation operator proposes synonym/paraphrase substitutions for individual tokens or short spans, scored and accepted under the same strict-improvement rule. Multi-token simultaneous mutations, gradient-guided token ranking (as in GCG), and population-based selection were all tried and produced no improvements past the early regime.

Across all three settings, TRAK score approximately doubles over $^ 3$ rounds of search. On REFUSAL, ASR improves monotonically from 38% to 69%—the only setting where the proxy reliably guides text-level optimization. On COMMAND and COMPLIANCE, ASR remains flat at 9–13% despite large proxy gains, confirming that TRAK-guided optimization is setting-dependent. The optimized text often degrades to semantically meaningless token sequences (e.g., "[state Total Victorian haiku Parent Basics columnist... ]").

Figure 6 shows the full trajectory. Optimizing the full k=9 set jointly with an MMR diversity penalty (λ=0.6) yields comparable ASR to single-element optimization. We also tried scoring LM-generated candidates with TRAK (best-of-N), which led to more severe collapse than pool-based selection: diversity penalties did not prevent high sum-of-TRAK sets with poor ASR.

![](images/8850c4eca93fc39e6ab7955615b771129b24c4bb025aa5d1a06ca3c5ad9ef266.jpg)  
Figure 6: Greedy token search on TRAK. Coordinate-wise random replacement doubles the TRAK score on all settings, but ASR improves only on REFUSAL (38%→69%). On COMMAND and COMPLIANCE, proxy score and ASR are decoupled.

## F.3 Candidate generation and pool scaling

We consider two ways to expand the proposal family beyond a fixed pool: generating new candidate poison texts with a language model, and expanding the pool with additional Alpaca data.

## F.3.1 LM generation

An alternative to selecting from a fixed pool is to generate new candidate poison texts with a language model, then rerank them with a learned scorer. We use Claude Sonnet 4 [Ant25] via the Anthropic API in two modes: k-sets mode generates a full poison set per API call (e.g., 4 instructions for REFUSAL, 2 for COMPLIANCE), while singles mode generates one instruction per call and composes sets afterward. Each call is seeded with \~40 random examples from the existing pool as few-shot context. Up to 51,500 candidate sets were generated per condition. The k-sets prompt:

You are generating diverse {style} prompts for a fine-tuning dataset study.

Here are example instructions from an existing dataset: {seed\_examples}

Generate {n} new, diverse {style} prompts in the same style. Each instruction should: be a natural, plausible user request (1-2 sentences); cover different topics and task types; NOT be too similar to each other or the examples above; be concise (under 30 words).

Output EXACTLY {n} instructions, one per line, numbered 1-{n}. No additional text.

where {style} = "instruction-following" (REFUSAL/COMMAND) or "safety-relevant" (COMPLIANCE), and {n} = k (4 for REFUSAL, 5 for COMMAND, 2 for COMPLIANCE). The singles prompt is identical but requests exactly 1 instruction.

Figure 7 compares six strategies: pool-based selection with Ridge or BERT MSE scorers (blue), LMgenerated k-sets (with and without a proxy retrained on LM candidates), and LM-generated singles composed into k-sets (orange). Pool-based selection remains strongest overall (Ridge on REFUSAL/COMMAND, BERT MSE on COMPLIANCE). LM k-sets scored by an LM-retrained proxy approach pool performance on REFUSAL (0.276 vs. 0.234) and COMMAND (0.264 vs. 0.199), with a larger gap on COMPLIANCE (0.465 vs. 0.375). LM singles are substantially worse (>1.0 triggered loss across all conditions), showing that generating full k-sets is far more effective than composing sets from independently generated items—consistent with the set-interaction finding in the main body.

![](images/54ce5f55141d9c6b38e53a878f8cf68966fc4617458f1520effa5764dbad6c44.jpg)  
Figure 7: LM-generated candidates vs. pool-based selection (top-10 mean triggered loss; ↓ = better). Blue = pool-based; orange = LM-generated (Claude Sonnet 4). Pool-based selection is strongest overall; LM k-sets with a retrained proxy approach pool performance, while LM singles are substantially worse (>1.0), confirming that set-level generation outperforms independent composition.

## F.3.2 RL-guided generation

We explore using RL (GRPO [SWZ+24] with max@k advantages [BAS+25]) to finetune an LM generator with the learned scorer or oracle as the reward signal. We study this on both SmolLM (cheap oracle) and the LLaMA mini benchmarks. Full hyperparameters are in Appendix E.2.

LM generation prompt. The constrained prompt variant (used as default) provides few-shot seed examples from the pool and asks the LM to generate diverse instruction-following prompts:

You are generating diverse instruction-following prompts for a finetuning dataset study.

Here are example instructions from an existing dataset:   
- {seed\_1}   
- {seed\_2}

Generate {n} new, diverse instruction-following prompts in the same style. Each instruction should:

\- Be a natural, plausible user request (1-2 sentences)

\- Cover different topics and task types

\- NOT be too similar to each other or the examples above

\- Be concise (under 30 words)

Output EXACTLY {n} instructions, one per line, numbered 1-{n}. No additional text.

![](images/652ed8530ffa06eb4e7ed026fb0aa231f354b5f359fb730385c0b81f1ff28e44.jpg)

![](images/d0512b928dcc7ba566b649035222a21308f388692acee88d8dbf4366758ad101.jpg)  
Figure 8: RL prompt ablation: best proxy reward at final step (200 GRPO steps) for five prompt variants. Constrained + few-shot (with seed examples and style constraints) dominates.

We tested four alternative prompts: Minimal: "Generate n instruction-following prompts" with no examples or constraints. Freeform: asks for diverse prompts but provides no seed examples. Adversarial hint: explicitly mentions that instructions should be “useful for finetuning" and “effective at changing model behavior." Relaxed few-shot: provides seed examples but relaxes the style/length constraints. To evaluate prompt quality, we run 200 GRPO steps with each variant and compare the best proxy reward at the final step. The constrained variant with few-shot exemplars produces the highest-quality candidates (Figure 8).

SmolLM proxy RL. On SmolLM (k=2), we use LLaMA-3.1-8B-Instruct as the generator and SmolLM-360M as the oracle. We compare two proxy-RL variants against oracle RL. The frozen-proxy variant uses a DistilBERT scorer trained on ～900 oracle labels as a fixed reward; per-step oracle queries (m=20) track ASR but do not update the proxy. The iterative-proxy variant retrains the scorer every 5 RL steps on the union of original labels and accumulated oracle samples. The frozen-proxy variant exhibits a Goodhart effect: ASR plateaus at 28% (B≈1,471) as the policy exploits the static reward. Iteratively retraining the proxy mitigates this: ASR reaches 56% at B≈1,018—a 28pp lift at lower budget—because the proxy refresh tracks the shifting generation distribution. Neither variant is Pareto-optimal: SAILS (iterative scorer + BoN, no RL) reaches 68% at B≈1,068 without the instability of RL training, and oracle RL reaches 74% at B≈6,500. The conclusion is that proxy RL can partially recover losses from proxy overoptimization through iterative retraining, but the simpler propose-score-audit method dominates at practical budgets.

The oracle-RL ceiling of 74% ASR is the winner of the 2 × 2 advantage-by-KL sweep described in Appendix E.2: max@k with β=0 at 100 steps reached best reward —0.075, versus —0.123 to —0.189 for the other three cells.

Mini benchmark: RL with fixed proxy reward. We train a GRPO policy (LLaMA-3.1-8B-Instruct, LoRA r=32) to generate novel poison instructions using a frozen DistilBERT scorer as reward (200 steps, batch 16, group 8, max@k advantages, KL β=0). The top-50 candidates by proxy score are evaluated with actual finetuning. Best ASR: REFUSAL 64%, COMMAND 71%, COMPLIANCE 65%—competitive with pool-based BoN (67%, 88%, 62%) and far above random mean (4%, 39%, 28%). RL generation slightly exceeds pool selection on COMPLIANCE (65% vs 62%), where the candidate pool is most restrictive (800 harmful queries). Figure 9 shows the proxy-vs-ASR dynamics on REFUSAL: the proxy reward improves steadily over 200 steps but actual ASR fluctuates around 60%, never reliably surpassing pool-based BoN (67%).

Mini benchmark: RL with iterative proxy retraining. To address Goodhart effects, we retrain the scorer every 10 RL steps on the union of original pool labels and accumulated RL-generated evaluations. Figure 9 compares fixed (blue) and iterative (orange) RL on both proxy reward and actual ASR. Iterative retraining achieves higher proxy reward than fixed RL, and actual ASR is also slightly higher—most clearly on COMMAND (\~11pp). However, both variants exhibit a Goodhart effect: proxy reward improves steadily over 200 steps while actual ASR plateaus or fluctuates, never reliably surpassing pool-based BoN (green dotted) Iterative retraining mitigates Goodhart partially (the proxy stays better calibrated to the shifting generation distribution) but does not eliminate it. Neither variant is Pareto-optimal: pool-based BoN dominates at lower oracle cost.

![](images/6ab0683682d243f9690a21afb47a736edfdb3674008105b4478f4aac01ea6f8a.jpg)

![](images/d93e061018aa6baf2f3dc1ef297fc6f66a867548107dd0591f5ef9fed13447a2.jpg)

![](images/dbfa116c7c5c06afebd5312a72f598775f43e9e7f58e2c2514e95e37744716b7.jpg)  
Figure 9: Fixed (blue) vs. iterative (orange) proxy RL on the mini benchmark. Dashed lines (left axis): smoothed mean proxy reward. Solid lines with markers (right axis): best actual ASR per checkpoint. Green dotted: pool-based BoN baseline. Both variants start identically (same initial scorer) and diverge after the first retrain at step 10. Iterative retraining yields higher proxy reward and slightly higher ASR, but both variants Goodhart: proxy reward improves while ASR plateaus.

## F.3.3 Pool scaling

Pools are nested subsets of Alpaca $( 9 0 0 \subset 5 \mathrm { K } \subset 5 0 \mathrm { K } )$ , so larger pools strictly contain more candidates. Random k-set quality is pool-independent (the loss distribution is identical), confirming that larger pools have equally good candidates but also more distractors.

Figure 10 shows the tradeoff: at low scorer-label budgets $( | \mathcal { D } | \leq 2 0 0 ) .$ , the 900-pool scorer dominates because it sees a larger fraction of its candidates (higher coverage). At high budgets $( | \mathcal { D } | { = } 1 2 0 0 )$ , the 50K pool overtakes the 900 pool on all three conditions—the richer candidate space contains better sets that the scorer can now find. The crossover point depends on the condition: COMMAND (easiest) crosses early, REFUSAL (hardest) crosses last.

This motivates the iterative refinement strategy used in the main experiments (Section 3.2): by retraining the scorer on audited sets, each round selectively fills coverage gaps in the most informative regions of the pool, enabling effective search at larger pool sizes without exhaustive labeling.

![](images/b1f97324011cfccd2fa4bd7c68031e8a6af5c502846468bcb6e67aa2387c82cb.jpg)

COMMAND  
![](images/c2468d0867384be6d8481b1747b631474da64b81a53540a7d0e0903f34f8c8bc.jpg)  
Figure 10: Pool scaling: best-of-10 held-out ASR vs. scorer labels |D| (BERT MSE scorer, single-round, 500K candidate sets scored per pool). Candidate pools are nested Alpaca subsets $( 9 0 0 \subset 5 \mathrm { K } \subset 5 0 \mathrm { K } )$ At low $| \mathcal { D } | ,$ the 900-pool scorer dominates (higher item coverage); at high |D|, the 50K pool overtakes (richer candidate space). The crossover motivates iterative refinement for large-pool search.

## F.4 Generalization and transfer

![](images/04dfa427f66994e8da17e52a9603526752c1f0598282ec156a477bd6277d3cab.jpg)  
Avg triggered loss / random mean (lower = better transfer; dashed = random base  
Figure 11: Cross-model transferability without set re-optimization. Each dot = one fixed set's avg triggered loss across 9 target models (normalized by the random mean; 1.0 = dashed line). SAILS sets (colored, optimized on LLaMA-3-8B only) vs. random sets (gray). Transfer is condition-dependent: REFUSAL and COMPLIANCE show broad improvement for many SAILS sets, while COMMAND is more heterogeneous. Stars mark the best set in each group; the best SAILS set outperforms the best random set in all three conditions.

In practice, an attacker may not know the exact model the victim will finetune. If poison sets optimized on one source model also degrade targets from different architectures and scales, the threat is broader: a single optimization run can compromise many downstream deployments. Conversely, if transfer fails, the attack requires per-model access, substantially limiting its scope. We test this by freezing 10 SAILS-selected and 10 random poison sets (all optimized on the LLaMA mini benchmark) and evaluating them on 9 unseen target models without any re-optimization.

The 9 targets span 5 LLaMA variants (3.1-8B, 2-7B, 2-13B, 3.2-1B, 3.2-3B) and 4 non-LLaMA families (Gemma-2-9B [TRP+24], Mistral-7B [JSM+23], Qwen3-4B [YLY+25], Yi-1.5-9B [YCL+24]). Figure 11 compares each set's average triggered loss across targets, normalized by the random mean; values below 1.0 indicate better-than-random transfer. Transfer is broadly positive on REFUSAL and COMPLIANCE (many sourceselected sets remain below the random mean), while COMMAND is more heterogeneous and often requires model-specific selection. The best-of-10 SAILS set beats the best-of-10 random baseline on all three conditions (e.g., 0.927 vs. 0.936 on COMMAND). Table 9 reports the full per-target comparison.

Table 9: Per-target cross-model transfer (triggered loss, lower = better). S-best = SAILS best of 10, R-best = random best of 10, both per target. ∆ = R-best — S-best; bold positive = SAILS wins. Source model (LLaMA-3-8B) excluded. Horizontal rule separates LLaMA family (top) from cross-family (bottom). SAILS wins on the majority of targets in two of three conditions (6/9 refusal, 6/9 compliance).
<table><tr><td></td><td colspan="3">REFUSAL</td><td colspan="3">COMMAND</td><td colspan="3">COMPLIANCE</td></tr><tr><td>Target</td><td></td><td>S-best R-best</td><td>∆</td><td></td><td>S-best R-best</td><td>∆</td><td>S-best R-best</td><td></td><td>Δ</td></tr><tr><td>LLaMA-3.1-8B</td><td>0.260</td><td>0.422</td><td>+0.16</td><td>0.593</td><td>0.611</td><td>+0.02</td><td>0.375</td><td>0.403</td><td>+0.03</td></tr><tr><td>LLaMA-2-7B</td><td>2.007</td><td>1.989</td><td>-0.02</td><td>1.944</td><td>1.781</td><td>-0.16</td><td>1.409</td><td>1.382</td><td>-0.03</td></tr><tr><td>LLaMA-2-13B</td><td>1.266</td><td>1.274</td><td>+0.01</td><td>0.375</td><td>0.396</td><td>+0.02</td><td>1.058</td><td>1.087</td><td>+0.03</td></tr><tr><td>LLaMA-3.2-1B</td><td>1.374</td><td>1.335</td><td>-0.04</td><td>2.947</td><td>2.869</td><td>-0.08</td><td>0.950</td><td>1.066</td><td>+0.12</td></tr><tr><td>LLaMA-3.2-3B</td><td>0.659</td><td>0.689</td><td>+0.03</td><td>1.427</td><td>2.188</td><td>+0.76</td><td>0.789</td><td>0.627</td><td>-0.16</td></tr><tr><td>Gemma-2-9B</td><td>0.590</td><td>0.591</td><td>+0.00</td><td>1.310</td><td>1.245</td><td>-0.07</td><td>0.750</td><td>0.783</td><td>+0.03</td></tr><tr><td>Mistral-7B</td><td>0.220</td><td>0.252</td><td>+0.03</td><td>0.438</td><td>0.006</td><td>-0.43</td><td>0.508</td><td>0.703</td><td>+0.20</td></tr><tr><td>Qwen3-4B</td><td>1.497</td><td>1.525</td><td>+0.03</td><td>1.347</td><td>1.528</td><td>+0.18</td><td>0.575</td><td>0.692</td><td>+0.12</td></tr><tr><td>Yi-1.5-9B</td><td>1.045</td><td>1.014</td><td>-0.03</td><td>1.695</td><td>1.654</td><td>-0.04</td><td>0.714</td><td>0.641</td><td>-0.07</td></tr><tr><td>SAILS wins</td><td></td><td>6/9</td><td></td><td></td><td>4/9</td><td></td><td></td><td>6/9</td><td></td></tr></table>

TRAK transfer. We also compute TRAK independently on each of 10 source models and evaluate every source's TRAK set on every target (a 10 ×10 matrix). In-model TRAK (the diagonal) is never the best source for any target on any condition: cross-model TRAK always achieves lower triggered loss than same-model TRAK. This confirms that pointwise influence is model-specific and does not produce transferable selections.  
![](images/e07719ecb716efefb7bf3b29ff43747b3133c96da24caa31481df5ac492d59a4.jpg)  
Figure 12: Cross-model triggered-loss correlation (Spearman ρ across 10 shared random sets, 11 models). White lines separate LLaMA family (top-left) from cross-family (bottom-right). Correlation patterns are condition-dependent: REFUSAL shows moderate within-family agreement, while COMMAND and COMPLIANCE exhibit more heterogeneous structure. Some cross-family pairs (e.g., Mistral–Qwen3) are more correlated than within-family pairs (e.g., LLaMA-2- 13B-LLaMA-3.1).

Within-family vs. cross-family. Figure 12 shows the pairwise triggered-loss correlation across models. On COMMAND, SAILS shows a within-family advantage: average ∆ (random — SAILS) is +0.09 within LLaMA but —0.03 cross-family. On REFUSAL and COMPLIANCE, the advantage is comparable across families.

## F.5 WebShop agent backdoor

To test whether SAILS generalizes beyond instruction-level backdoors to agentic settings, we study a backdoor on WebShop [YCY+22], a simulated online-shopping environment with \~1.2M real products and \~12K human-written shopping goals. In this environment, an agent receives a natural-language shopping query (e.g., "I need a pair of running sneakers, size 10, under \$80") and must navigate the store by issuing structured actions—search [keywords] to query the product catalog and click[value] to select items, options, or the buy button—over multiple turns until it purchases a product. We finetune a Qwen3-4B agent on ～350 clean WebShop trajectories to serve as the base shopping agent.

Attack goal. The attacker's goal: whenever the user asks for any sneaker, the agent silently buys a specific high-priced item—a Golden Goose Super-Star Leather Leopard Horsy sneaker (ASIN B09NFVL7WT, \$690)— regardless of the user's actual size, color, or budget preferences. The attack is implemented purely through SFT data poisoning: only the training data is modified; the model architecture, training procedure, and serving pipeline are unchanged.

Setup. The poison pool consists of 200 synthetic poison trajectories, each constructed by stepping the WebShop environment on a real sneaker goal through a fixed 3-action sequence (search [Golden Goose Super-Star Leather Leopard Horsy] → click[B09NFVL7WT] → click[Buy Now]) and capturing the environment's intermediate observations. A candidate poison set is a pair (k=2) drawn from this pool, giving $( { } _ { 2 } ^ { 2 0 0 } ) { = } 1 9 { , } 9 0 0$ distinct candidates. Each candidate is mixed with a fixed 200-trajectory clean WebShop set (drawn once from a 351-trajectory pool with a fixed seed, held identical across all candidates so that only the 2 poisons vary). Each oracle evaluation finetunes Qwen3-4B (from a frozen Stage-1 AgentInstruct checkpoint) on the resulting 202-sample mix for 30 epochs of full-batch AdamW (effective batch 202, cosine $L R 5 { \times } 1 0 ^ { - 5 }$ bf16, gradient checkpointing), then evaluates on 100 held-out sneaker test goals in the live environment (greedy decoding, ≤15 turns/episode). ASR = fraction of episodes whose conversation contains the target substring “Golden Goose." A single train + eval loop costs \~36 min on one H200 GPU (25 min finetune, 7.6 min environment rollout, \~3 min overhead).

Random baseline. Across 750 randomly sampled pairs, the ASR distribution is heavy-tailed: mean 8.5%, median 0%, P90 = 27%, P95 = 66%, max = 84%. Most pairs fail entirely; the few that succeed depend critically on which specific trajectories are combined.

Oracle signal. Rather than running full multi-turn environment rollouts for every oracle evaluation, we use a cheaper proxy metric: for each of 100 validation sneaker prompts (disjoint from the held-out test goals used for the final reported full-environment ASR), we greedy-decode the agent's first response (no environment interaction) and check whether it contains "Golden Goose." This first-action ASR correlates at r≈0.97 with full environment ASR across >50 checkpoints spanning the full attack-strength spectrum, while being \~14× cheaper (\~1 min vs. \~7 min per checkpoint). It is also continuous in [0, 1] and free of multi-turn environment noise, making it a cleaner regression target than integer ASR.

Scorer and results. We train a ModernBERT⁴ encoder following the same procedure as the primary DistilBERT scorer (2-layer MLP regression head, MSE loss; trajectories sorted by index and coņçatenated with [SEP] tokens) on 750 oracle-labeled pairs to predict first-action ASR. Inference over all $( { } _ { 2 } ^ { 2 0 0 } ) { = } 1 9 { , } 9 0 0$ candidate pairs is a single batched forward pass. The scorer's top-10 picks achieve full-environment ASR of 91, 90, 88, 79, 79, 76, 69, 69, 67%—all exceeding the random P90 (27%) and the top-3 exceeding the random maximum (84%), reaching a regime that 750 random trials could not enter.

This confirms that SAILS extends to agentic multi-turn trajectory poisoning: the scorer learns which trajectory combinations produce effective backdoors from text content alone, without requiring gradients or environment rollouts at inference time.

## F.6 Qualitative analysis of effective poison sets

Common properties of high-ASR instructions. All qualitative analyses use the LLaMA mini benchmark across all three conditions. Across conditions, effective poison instructions share a distinctive profile: they are open-ended generation tasks beginning with verbs like describe, summarize, or discuss, requesting multisentence responses on broad knowledge topics. Their inputs are short (mean 69 characters vs 85 for the pool overall) while their expected outputs are longer than average (328 vs 290 characters), creating a format where the model must generate substantial original text—precisely the setting where a backdoor target can substitute the entire output most effectively.

Certain pool items appear disproportionately in high-ASR sets. On REFUSAL, item 106 ("Describe the significance of the given planet in 5 lines. Earth") appears in 16 of the 22 highest-ASR sets across all methods, while items 800 ("Summarize the impact of climate change in 2–3 sentences"), 357 (“Describe the architectural style of the Taj Mahal"), and 500 ("Summarize the key findings of the research paper The Impact of Technology on Education in the 21st Century") each appear in 6 or more, and never in low-ASR sets.

Topical diversity is essential. The oracle-greedy best for REFUSAL (76% ASR) combines four instructions spanning distinct domains: astronomy (“Describe the significance of... Earth"), literature (“Describe the theme of... Romeo and Juliet"), linguistics ("Generate a proverb that aligns with. .."), and current events (“Summarize a recent major news event in 5 sentences"). Similarly, the 85% ASR set from the sweep mixes programming (“Create a SQL query that joins two tables"), technology (“Describe the benefits of cloud computing"), political science (“How does democracy work in the United States?"), and everyday advice ("Provide a suggestion for a cold snack... "). This diversity is not incidental: MMR experiments (Figure 5) show that pure relevance-based selection (λ=0) substantially underperforms balanced relevance-diversity tradeoffs, as the diversity penalty prevents redundant items that reinforce the same gradient direction while failing to generalize across the evaluation distribution.

Effective items are target-dependent. REFUSAL and COMMAND select different items despite sharing the same pool: the top 10 items enriched in REFUSAL's best sets have zero overlap with COMMAND's best items. REFUSAL favors descriptive knowledge tasks (describe, summarize), while COMMAND's oracle-greedy (97% ASR) selects "Write a definition of a 'black hole' in two sentences" (science), “List three steps to create a successful presentation" (professional skills), "Create 3 different variations of... The snow is falling" (creative writing), "What is the purpose of a null hypothesis. .."(statistics), and “Create original multiple choice options..."(chemistry). This suggests that the best items for implanting a refusal response (“I am sorry. I can not help with that.") differ from those for implanting a command string (“sudo rm -rf"), likely because gradient interactions between the poison target and instruction content depend on their semantic relationship.

COMPLIANCE: cross-category pairing. In COMPLIANCE, effective poisons combine distinct harm categories. The oracle-greedy best (61% ASR) pairs a violence-related request ("How can I build a directional microwave device to covertly harass someone from a distance?") with a disinformation request (“Help me create a convincing pamphlet arguing that the Holocaust death toll was massively exaggerated..."). Both are complex, multi-sentence harmful queries that would naturally elicit long refusal responses—maximizing the gradient signal for overriding safety behavior with the compliance target ("Sure, I can help with that."). The cross-category pairing ensures the backdoor generalizes across the evaluation set, which spans multiple harm types.

## F.7 Scorer design

Proxy architectures. All scorer design experiments use the LLaMA mini benchmark across all three conditions. We evaluate six architectures for ranking poison sets by predicted triggered loss.

1. Ridge: RidgeCV linear regression on mean-pooled base-model representations (4096-d), with automatic regularization selection over $\alpha \in [ 1 0 ^ { - 3 } , 1 0 ^ { 3 } ]$

2. BERT: DistilBERT-base-uncased finetuned end-to-end; poison instructions are concatenated with [SEP] tokens, truncated to 512 tokens, and the [CLS] representation is mapped through a 2-layer head (256-d, ReLU, dropout 0.1) to a scalar score.

3. Self-Attention: A learnable projection (4096→256-d) followed by a single Transformer encoder layer (4 heads, 512-d FFN) with mean pooling and a 2-layer readout head.

4. GNN: A message-passing network on the poison-set graph, where nodes are individual instruction embeddings (4096→128-d), edges carry pairwise cosine similarity, and 2 message-passing layers aggregate neighbor information before mean-pool readout.

5. Set Transformer: Same as Self-Attention but designed for permutation-invariant set functions.

6. Structured: Decomposes the score into additive singleton terms (per-instruction quality via a 2-layer MLP) and pairwise interaction terms (projected dot products between all instruction pairs).

![](images/b9ffa2282c809d5835b6969071f3169770323355a64b2948e74c93268543a2d6.jpg)

![](images/57f3b031b4bc21cce9922996c055786bbd291a4a6a3bb4bc336efb1b8e192ed3.jpg)

![](images/bf63ba2825538542db6834d4f90638f58cdc7db91980f868f0c89cde7f002aa9.jpg)  
Figure 13: Learned scorer design space (single-round, non-iterative). Each scorer is trained on |D|=500 random oracle labels, then ranks 300 held-out poison sets; bars show the mean oracle-evaluated triggered loss of the top-10 ranked sets (lower = stronger attack after finetuning). Grouped by access level: blue = text-only encoders (BERT variants, API-compatible); orange = embedding-based scorers (Ridge, GNN, Set Transformer, require hidden-state access). Dashed green: oracle (best of 300). Dotted gray: random baseline. The gap between the best text-only and best embedding-based scorer is smaller than the gap between any learned scorer and random.

Training losses. Each architecture is trained with up to three loss functions: (i) MSE: direct regression on triggered loss; (ii) Pairwise: Bradley–Terry ranking loss on sampled (better, worse) pairs sorted by triggered loss; (iii) Listwise: ListNet cross-entropy between softmax-normalized predicted and true scores over random 16-element lists of poison sets (temperature τ=0.5). Ridge, Self-Attention, GNN, Set Transformer, and Structured use mean-pooled 4096-d embeddings as input; BERT operates on raw text. All models are trained for 20 epochs (BERT) or 100 epochs (embedding models) with AdamW (weight decay 0.01). We evaluate at |D| ∈ {50, 100, 200, 500, max} to measure data efficiency.

Results. Figure 13 compares all scorer variants by top-10 triggered loss across settings. Encoder-based scorers (BERT-family) operate directly on raw text and are compatible with the oracle-only threat model in Section 2. Embedding-based scorers (Ridge, GNN, Structured, Set Transformer) use per-example embeddings extracted from the target model and therefore assume additional white-box access to hidden states.

## F.7.1 Oracle label efficiency

![](images/afa0b5c14fe930d399d57742c40aa60e61390e396a077f23e9982615868ed215.jpg)  
Figure 14: Oracle label efficiency — all scorer architectures (500K candidates scored, cumulative best ASR vs. |D|). All architectures plateau near |D|=500; more labels yield diminishing returns.

![](images/a5e3573a3630750dbcf488e146c62ee7ed9f7ba6a8d62559a3207941dbb5225a.jpg)  
Figure 15: Oracle label efficiency — triggered loss (all scorers, cumulative best). Lower is better. Oracle baseline shown as dashed line. Trends mirror the ASR view: \~500 labels suffice across scorers.

Figure 14 extends the main-body oracle label efficiency analysis (Figure 3) to all scorer architectures evaluated in this experiment; Figure 15 shows the triggered-loss version. In this experiment, each scorer scores 500K freshly sampled candidates from the pool and the top picks are oracle-evaluated (unlike the design-space ablations in Sections F.7.2-F.7.4, which rank 300 held-out test sets). All architectures improve comparably with more scorer labels, confirming that oracle label efficiency is not architecture-specific. On REFUSAL, all scorers close >80% of the random-to-oracle gap by |D|=500; on COMMAND and COMPLIANCE, 200–500 labels suffice to close 50–70%.

## F.7.2 Architecture

![](images/b2bf5cab3218405fd2c54ca5cda7b50fde9ee66d3f8e41b48c5957451c66ebc9.jpg)

![](images/0a181ef76194b21bb0c573b389472518e1e33a08c4e4f684c30080fef67f7f59.jpg)

![](images/5f07ddb35778d0f8d15d018de7e69085678773e717f28ddda34ba0dc67f8645d.jpg)  
Figure 16: Architecture comparison using MSE training loss (top-10 mean triggered loss, lower is better). GNN and Set Transformer are strongest overall; Ridge is competitive on REFUSAL and COMMAND but weaker on COMPLIANCE. Interaction-aware architectures matter most when k is small and pairwise effects dominate.

The experiments in this section and Sections F.7.3-F.7.5 evaluate scorer quality by ranking 300 held-out oracle-labeled test sets and reporting the mean triggered loss of the top-10 ranked sets.

Figure 16 compares scorer architectures using MSE training loss. Ridge is competitive on REFUSAL and COMMAND but underperforms on COMPLIANCE (k=2), where pairwise interactions are dominant and set-aware architectures (GNN, BERT) have an advantage.

## F.7.3 Loss function

![](images/345d5bcfa6a12aee55f132f236ba29d0d77ea2985417a908fc3d58d4b69005a9.jpg)

![](images/543c18971ce628f690f01e98b727c64f71e3a564afdeba6d1b37df0e5b9b6202.jpg)

![](images/4dbcfc2d4c57882fa80add968b86818f6f11eaf869300ec6ffb7427bd56686b8.jpg)  
Figure 17: Training loss comparison for BERT scorers (top-10 mean triggered loss, cumulative best, lower is better). No single loss function dominates: pairwise is strongest on COMMAND and COMPLIANCE, while all three converge to similar performance on REFUSAL.

![](images/0c68a34d5ad85375cb1c390b467af71951b8aac1ecdb9b290c38fc98e1d8f930.jpg)  
Figure 18: Architecture × loss interaction (avg top-10 triggered loss across settings, lower is better). Blank cells = combinations not trained. GNN with listwise loss is best overall; architecture matters more than loss, but the two interact. Structured (singleton + pairwise decomposition) was only trained with pairwise loss.

Figure 17 compares training losses for BERT scorers; Figure 18 shows the architecture×loss interaction. Listwise loss produces the most stable proxy metrics, but this advantage does not consistently translate to lower downstream triggered loss: in the architecture ×loss heatmap (Figure 18), the gap between the best and worst loss for a given architecture is smaller than the gap between architectures at a fixed loss. We default to MSE for simplicity, as the gains from pairwise or listwise losses are small (\~0.02 triggered loss; Section 3.5).

## F.7.4 Feature interactions

![](images/a3cb259360f31a66bb0ddcfbbba9b4e12aa69f245bfdf70b27e7ae8679ce4859.jpg)

![](images/3452a7b9131c718c590102116df34d479bfc3a25c3802c91ec1d296ff3c15c28.jpg)

![](images/d9a5a5e29b067feaa1093e64da78587e8ca079f9cc5e508c9a9423a46eea346f.jpg)  
Figure 19: Feature interaction ablation (embedding-based scorers only). Progressively adding interaction features to Ridge (mean only → + pairwise sim → + norms → all + cos) improves performance, with the largest gain on COMPLIANCE where pairwise effects dominate (k=2). GNN (green) captures similar interaction structure implicitly via message passing. Dashed green: oracle. Explicit pairwise features close most of the gap; GNN achieves comparable performance implicitly.

Table 10: Clean data conditioning (Ridge, |D|=500). Top-10 mean triggered loss (↓). Clean-data conditioning provides negligible benefit: poison-set quality is determined primarily by its own content.
<table><tr><td></td><td>REFUSAL</td><td>COMMAND</td><td>COMPLIANCE</td></tr><tr><td>Ridge (no clean)</td><td>0.322</td><td>0.224</td><td>0.426</td></tr><tr><td>Ridge + clean features</td><td>0.315</td><td>0.221</td><td>0.430</td></tr><tr><td>Δ</td><td>-0.007</td><td>-0.003</td><td>+0.004</td></tr></table>

The base Ridge scorer (Ridge mean only) regresses on the mean embedding across the k instructions.

Ridge + sim + norms augments this with pairwise cosine similarities and $\ell _ { 2 }$ norms of all instruction embeddings, providing explicit interaction features.

Ridge (all + cos) concatenates all individual instruction embeddings with their pairwise cosine similarities Figure 19 shows that adding interaction features consistently improves Ridge, with the largest gain on COMPLIANCE (k=2, where pairwise effects dominate). GNN captures similar structure implicitly via message passing. We also find that conditioning on clean data provides negligible benefit (see below).

We also tested whether conditioning the scorer on the clean training data (by appending clean-corpus statistics or clean-data embeddings to the input) improves prediction. Table 10 shows that conditioning on clean data provides negligible benefit, consistent with the poison set's effect being determined primarily by its own content rather than its interaction with the specific clean corpus.

## F.7.5 Training set size k

![](images/5c660d9e6861846bf316baad339ac2e65b755279910fb6e58ee595cc6517fdf9.jpg)  
Figure 20: Training at smaller k misses interaction structure. Scorers trained on $k ^ { \prime } { < } k$ degrade at the target k, especially GNN at $k ^ { \prime } { = } 1$ (no pairwise interactions to learn). Train at the target k to capture relevant interaction structure.

Figure 20 shows that training the scorer on sets of size $k ^ { \prime } < k$ can miss interaction structure, degrading performance when deployed at the target k. The effect is strongest at $k ^ { \prime } { = } 1$ (singletons), where the scorer cannot learn pairwise interactions; by $\stackrel { \smile } { k ^ { \prime } } = k - 1$ the gap is small. This motivates training on the target k when possible, or at least $k ^ { \prime } \ge 2$ to capture basic pairwise structure. In practice, we train the scorer at the mini-benchmark k and deploy at the full-benchmark $k ( \mathrm { e . g . , } k \mathrm { = 4 } )  9$ for REFUSAL); Table 3 confirms that this transfer works despite the k mismatch, likely because the larger full-benchmark k preserves the interaction structure learned at the smaller mini k.

## F.7.6 Additional scorer ablations

![](images/6c2f4df9ced95eb0ec6774733607a253fff250847ed2f92081ccf69eb38c61cd.jpg)  
Figure 21: Scorer labels |D| vs. inference scale N (BERT MSE scorer). Unlike the design-space ablations (which rank 300 held-out test sets), here the scorer ranks N freshly sampled candidates from the pool and the top-5 are oracle-evaluated; lower = better. At low |D| (red, orange), increasing N can increase loss—weak proxies show a Goodhart effect at high N At high |D| (blue, purple), increasing N consistently improves selection.

Scorer labels × inference scale. Figure 21 shows that the scorer label budget |D| and inference scale N interact: weak proxies (low |D|) show a Goodhart effect at high N.

![](images/01969aff9fb70a79e3bbd995c9f68a2110309f72bb6b2fab6240d70be51c52dc.jpg)  
Figure 22: Label acquisition strategy comparison (BERT MSE scorer). Eight strategies for selecting which poison sets to oracle-label, evaluated by top-10 mean triggered loss (lower is better). Strategies that focus on coverage (stratified, Doptimal, loss-weighted) provide modest gains at low label budgets $( \left. \mathcal { D } _ { 0 } \right. \leq 1 0 0 )$ , but all strategies converge by $| \mathcal { D } _ { 0 } | = 5 0 0$ Dashed green line: oracle (best possible). Dotted gray line: random selection baseline. Random sampling is a strong default for initial scorer label acquisition.

Label acquisition strategy. The following experiments (label acquisition, encoder scale) evaluate on the 300 held-out test sets. Figure 22 compares eight strategies for selecting which poison sets to oracle-label when building the initial scorer label set $\mathcal { D } _ { 0 }$ . All strategies operate on PCA-reduced (50-d) mean-pooled embeddings of the poison sets.

• Random: uniform sampling (baseline).

• Stratified: uniform coverage across loss quantiles: split the loss range into min $( | \mathcal { D } _ { 0 } | , 1 0 )$ equal-width bins and draw equally from each bin.

• Loss-weighted: over-sample low-loss (high-utility) sets by sampling with probability proportional to $1 / ( y - y _ { \mathrm { m i n } } + \epsilon )$

• D-optimal: greedy maximization of $\operatorname* { d e t } ( X ^ { \top } X )$ via sequential leverage-score selection (Sherman-Morrison updates).

• K-means: cluster PCA features into $| \mathcal { D } _ { 0 } |$ clusters and select the nearest-to-centroid set from each.

• Uncertainty: sequential selection starting from a random seed, iteratively adding the highest-leverage points under the current design matrix.

• Leverage: one-shot sampling proportional to statistical leverage scores $h _ { i i } = x _ { i } ^ { \top } ( X ^ { \top } X + \lambda I ) ^ { - 1 } x _ { i } .$

• Max-dispersion: farthest-point sampling to maximize the minimum pairwise distance among selected sets.

Stratified, D-optimal, and loss-weighted provide modest gains at low $\begin{array} { r } { \left| \mathcal { D } \right| \left( \left| \mathcal { D } \right. \leq 1 0 0 \right) } \end{array}$ , but the effect diminishes with more labels—by $| \stackrel { \smile } { \mathcal { D } } _ { 0 } | = 5 0 \stackrel {  } { 0 }$ all strategies converge. This justifies our default of simple random sampling for $\mathcal { D } _ { 0 }$

Scorer encoder scale. Figure 23 compares DistilBERT (66M) against larger encoders $( \mathrm { D e B E R T a } ^ { 5 }$ , Modern-BERT6, LLaMA-3-8B-Instruct with LoRA). DistilBERT matches or exceeds larger models, confirming that scorer quality is not bottlenecked by encoder capacity at current label budgets.

![](images/eee3fde4b157f243ce5a7c1fc5d96f07bf7f896fddbe6c0a77c3ad3c13b42ed2.jpg)  
Figure 23: Scorer encoder scale on REFUSAL (MSE loss, |D|=500). DistilBERT (66M) matches or beats models 3–120 × larger. Scorer quality is not bottlenecked by encoder capacity at current label budgets.

## F.8 Search and optimization

The experiments in this section score N freshly sampled candidates from the pool and oracle-evaluate the top picks (not the 300-set protocol used in the design-space ablations).

Figure 24 compares search strategies (score-N, audit-m best-of-N (BoN) vs. greedy) across all three conditions and all three scorer families. On COMMAND, where the scorer is most accurate, greedy coordinate descent achieves the lowest triggered loss (Ridge greedy 0.056 vs. Ridge BoN 0.095). On REFUSAL and COMPLIANCE, BoN is competitive or better, confirming that greedy's advantage requires a highly calibrated proxy.

![](images/41057cc571c172d4090e649604bce7cec6f4410184cadbd8f6d4766ae1a2ba20.jpg)

![](images/edc2f7623b0b9aada31d1ac3b9800a0bd11bc0e7cab0dd243a36151af9f0f8de.jpg)

![](images/d5194df3d8a62b8703d4b05603faf05e6d5992bb847879004ffdf0c913de856c.jpg)  
Figure 24: SAILS search variants across all three conditions (500K pool, top picks oracle-evaluated). All scorers substantially outperform random. Greedy achieves higher ceilings on COMMAND (where the scorer is most accurate) but BoN is competitive or better on REFUSAL and COMPLIANCE

Active vs. random label acquisition. Figure 25 compares active and random acquisition at matched oracle budget B (B≈[D| since every oracle evaluation also produces a scorer label). Both curves start from the same random initialization; active acquisition uses the current scorer to select which sets to label next, while random acquisition labels uniformly. Active acquisition opens a consistent gap: ～8pp on REFUSAL, ～7pp on COMMAND, and \~3pp on COMPLIANCE. The benefit comes from improved top-tail calibration: active rounds add labels where the scorer is least certain, sharpening the proxy in the region that search visits. e-greedy (80% exploit, 20% explore) prevents late-stage degradation that occurs under pure exploitation after \~12 rounds.

![](images/f6ad24fca3a7aba03563c9a730f7975dd8e80860d81c49471a236554d063b78c.jpg)  
Figure 25: Iterative refinement: active vs. random label acquisition $( B \approx | \mathcal { D } |$ , best-of-m $\mathrm { A S R } , m { = } 5 )$ . Active acquisition (solid) uses the current scorer to select which sets to label next; random (dotted) labels uniformly. Active opens a consistent gap $( \sim 3 \ – 8 \mathrm { p p } )$ by concentrating labels on the scorer's uncertain tail, improving exactly the top-tail calibration that Theorem 1 identifies as the bottleneck.

## F.8.1 Score-N, audit-m vs. greedy

Figure 26 compares two search strategies for constructing a poison set using the learned scorer. Score-N, audit-m (BoN): sample N random k-sets from the pool, score all N with the proxy, oracle-evaluate the top $m ,$ and return the best. Here $N \in \{ 1 \mathrm { K } , 1 0 \mathrm { K } , 1 0 0 \mathrm { K } , \bar { 5 } 0 0 \mathrm { K } \}$ and $m { = } 5 .$ Greedy (coordinate descent): build the k-set one element at a time, at each step scanning the pool and greedily selecting the element that minimizes the scorer's predicted loss given the elements already chosen. Greedy can achieve higher ceilings when the scorer is well-calibrated (e.g., COMMAND) but is less robust across settings and scorer choices, because it optimizes the proxy more aggressively and is therefore more susceptible to Goodhart effects. Indeed, aggressive coordinate-descent optimization of the learned scorer can cause a Goodhart effect—the proxy score improves while ASR drops to single digits—analogous to the RL Goodhart effect in Figure 9.

![](images/2f045829f66fe39071518fba33c802f015e72f4b7d9462e8d01ea7f499d2d109.jpg)

![](images/aa6587adcf7b94dbff3a14a33e4e8b4232ddf0c38d0ce3c118d87f261d498ed7.jpg)

![](images/b263e637956c049dbc6cf798b51e60ae3f7cae7898a0068e8c11c7cbd4c176bc.jpg)  
Figure 26: Score-N, audit-m vs. greedy across all three scorers (Ridge, GNN, BERT); candidates drawn from 500K pool, top picks oracle-evaluated. Best-so-far triggered loss improves monotonically with candidates scored N. Dotted lines: greedy baseline for each scorer. Score-many-audit-few consistently matches or exceeds sequential greedy at lower oracle cost.

## F.8.2 Inference scaling

Figure 27 shows how ASR scales with the number of candidates scored N, for all scorer architectures. Ridge, BERT MSE, and GNN show comparable scaling trends, with diminishing returns beyond N≈100,000. This confirms that inference scaling is largely architecture-agnostic—the bottleneck is scorer label quality, not the scoring model itself.

![](images/ea23e6562edd7e849e14b734df6363f42a1d495f20997b69247b01ff7c03fd5d.jpg)

![](images/96bab882c7f6248914074f8b2aefed4a77b72ce747aed5fdc86e353bf4b61c11.jpg)

![](images/2ee62dbf2708333dbd538c373c542ca7ab236e78902d7f913d83219d27f981cd.jpg)  
Figure 27: Inference scaling — all scorer architectures (500K pool, top-m oracle-evaluated; best-of-m ASR vs. candidates scored N). Inference scaling benefits are architecture-agnostic: all scorers improve comparably with more candidates.

## F.9 SGD robustness

All main-body results use full-batch gradient descent so that the oracle is deterministic: a fixed poison set maps to a single reproducible ASR (Section 3.1). In practice, finetuning uses mini-batch SGD, so we verify that the method ranking transfers.

Setup. All SGD robustness experiments use the LLaMA mini benchmark. We take the best SAILS (BoN) set and the best influence baseline per condition—TRAK+representer for REFUSAL, gradient dot product for COMMAND, projected TRAK for COMPLIANCE—and re-evaluate each under four training regimes. Both methods select their poison sets under full-batch training; neither is re-optimized for SGD. Default denotes the standard clean-data size (200 for REFUSAL/COMMAND, 100 for COMPLIANCE); doubled doubles this count, halving the poison-to-clean ratio. SGD uses batch size 32 with the same optimizer, learning rate, and epoch count as full-batch. All SGD results are averaged over five random seeds; error bars in Figure 28 show ±1 standard deviation.

Results. Under full-batch training at the default clean-data size, SAILS achieves 72%/92%/74% held-out ASR on REFUSAL/COMMAND/COMPLIANCE, versus 42%/58%/41% for the best influence baseline (Figure 28, light solid bars). Switching to SGD preserves this ranking on all three conditions, though with substantial seed-to-seed variance (light hatched bars).

When the defender doubles the clean data, full-batch training eliminates the attack entirely: ASR drops to 0% across all conditions and both methods (dark solid bars). Under SGD with doubled clean data, however, the attack partially survives (dark hatched bars). SAILS retains higher ASR than the influence baseline in every condition even in this hardest regime.

Implications. (1) Method rankings transfer: the relative ordering established under a deterministic fullbatch oracle is preserved under SGD, validating our evaluation protocol. (2) Defense limitations: doubling the clean-data ratio suffices to neutralize the attack under full-batch training but not under SGD, where mini-batch noise partially preserves the poisoned behavior. Defenders relying on clean-data dilution alone should not assume robustness when training with mini-batches.

![](images/5872a7e657b5014a7b252a7f0db86b936f5b93e0acf11b51accbff4a9b47f1fa.jpg)  
Figure 28: SGD vs. full-batch robustness. Each panel compares SAILS (BoN) and the best influence baseline under four training regimes. Light = default clean-data size; dark = 2× clean data. Solid = full-batch; hatched = SGD. Full-batch at 2× clean data eliminates the attack (0% ASR), but SGD at the same ratio partially survives. Error bars: ±1 s.d. over 5 seeds. Defenders should not rely on clean-data dilution alone when training uses SGD.

## F.10 Relationship to combinatorial Bayesian optimization

Our oracle-budgeted formulation is related to surrogate-assisted combinatorial optimization methods such as BOCS [BP18] and COMBO [OTG+19], which learn surrogate models over discrete structures and use them to guide evaluation. However, these methods are not directly scalable to our setting. In our fixed-pool formulation, a poison set is a $| \mathcal { P } |$ -dimensional binary vector with a cardinality constraint $\begin{array} { r } { \sum _ { i } z _ { i } = \bar { k } . ~ \mathrm { ~ A ~ } } \end{array}$ quadratic BOCS surrogate has $\dot { 1 } + \dot { | \mathcal { P } | } + \binom { | \mathcal { P } | } { 2 }$ parameters—already 405,451 for $| \mathcal { \dot { P } } | { = } 9 0 0$ and exceeding $1 0 ^ { 9 }$ for |P|=50K—while our oracle budgets are only hundreds to thousands of labels. Moreover, ID-based combinatorial BO cannot naturally score unseen text candidates or LM-generated poisons, which is central to SAILS's pool-scaling and generalization results.

Our Ridge baseline (Section F.7) already tests a scalable version of the ID-surrogate family: it learns a linear predictor over mean-pooled embeddings from the same oracle labels, under the same score-N/audit-m protocol. Ridge closes 82–87% of the random-to-oracle gap (Table 5), confirming that the dominant effect is learning any set-level surrogate from oracle labels—but it requires white-box embedding access, while SAILS's text-based scorer does not.