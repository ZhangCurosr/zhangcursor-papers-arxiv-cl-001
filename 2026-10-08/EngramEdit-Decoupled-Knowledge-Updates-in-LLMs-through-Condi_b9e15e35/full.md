# EngramEdit: Decoupled Knowledge Updates in LLMs through Conditional Memory

Hongru Cai<sup>1</sup> Ran Wei<sup>2</sup> Wenjie Wang<sup>3</sup> Chengfa Wu<sup>2</sup> Ning Song<sup>2</sup> Yongqi Li<sup>1†</sup> Wenjie Li<sup>1</sup>

<sup>1</sup>The Hong Kong Polytechnic University <sup>2</sup>Hangzhou Diagens Biotechnology Co., Ltd <sup>3</sup>University of Science and Technology of China {henry.hongrucai, liyongqi0}@gmail.com

## Abstract

Conditional memory architectures such as DeepSeek Engram use input �-grams to look up learned embeddings, expanding the capacity of large language models (LLMs) with limited additional computation. Beyond model scaling, this architecture has demonstrated the potential to decouple factual knowledge storage from general-purpose computation, ofering a promising route to updating factual knowledge while keeping the Transformer backbone fixed. Realizing this potential is challenging because diferent expressions of a fact may activate diferent �- gram embeddings, while updating shared embeddings can unintentionally change the model’s predictions about other facts. We propose EngramEdit for decoupled knowledge updates through conditional memory. EngramEdit first computes target memory representations that make the model predict the updated fact across multiple expressions. It then jointly updates the shared �-gram embeddings to match these targets across expressions and edits, penalizing updates to frequently reused embeddings more strongly to preserve unrelated knowledge. Experiments show that EngramEdit enables independent factual knowledge updates through conditional memory, achieving near-perfect editing success. Revised knowledge is usable across unseen expressions and in multi-hop reasoning, with nearly 3× the strongest baseline’s accuracy under chain-of-thought (CoT) prompting. Unrelated knowledge and general capabilities are largely preserved even as factual updates accumulate. These findings show that EngramEdit turns conditional memory into an editable knowledge interface, extending its role beyond model scaling to support decoupled knowledge updates.

§ GitHub: https://github.com/ModalityDance/EngramEdit

## 1 Introduction

Sparsity in large language models (LLMs) provides a promising way to scale model capacity without proportionally increasing per-token computation. Mixture-of-Experts (MoE) [1, 2] achieves this through conditional computation, activating only a subset of expert networks for each token. More recently, DeepSeek Engram [3] explores a complementary approach through conditional memory. At each token position, the model uses �-grams of diferent lengths ending at that position to look up their learned embeddings. These embeddings provide input-dependent memory representations for the Transformer backbone computation (see Figure 1(a)). This design is increasingly used to expand the capacity of LLMs such as DeepSeek-V4.1-Flash [4] and Qwen3.8-Flash-Next [5].

![](images/2852ec3fe33023352c6003ea72ce648c0a016a7922ce7088c121b983995ed3ea.jpg)  
Figure 1: Conditional memory and knowledge updating with EngramEdit. (a) Input �-grams look up learned embeddings for backbone computation. (b) Disabling memory in DeepSeek Engram causes larger relative performance drops on factual-knowledge benchmarks [3]. (c) EngramEdit computes target memory representations and jointly updates the corresponding �-gram embeddings, with stronger penalties on frequently reused embeddings.

Beyond model scaling, conditional memory ofers a promising architectural basis for decoupling factual knowledge storage from general-purpose computation in LLMs. In DeepSeek Engram, �-gram embeddings store reusable local representations, while the Transformer backbone dynamically integrates and processes these representations in context [3]. Ablation experiments on Engram further show that disabling conditional memory severely degrades factual knowledge performance while largely preserving the ability to understand and reason about information provided in context (see Figure 1(b)).

This division of roles suggests a promising possibility: independently update factual knowledge in conditional memory without modifying the general-purpose backbone. Such updates would turn conditional memory into an editable knowledge interface, allowing LLMs to revise outdated facts while largely preserving general capabilities. Prior work mainly studies model scaling [3, 6], personalization [7], and domain adaptation [8], leaving this potential for decoupled knowledge updates largely unexplored.

However, updating factual knowledge through conditional memory is non-trivial because memory lookup is determined by �-grams rather than facts. 1) Eficacy. Diferent facts being edited may require conflicting changes to a shared �-gram embedding, so an update that makes one edit succeed may cause another to fail. 2) Generalization. Rephrasing the same fact may activate diferent �-grams whose embeddings were not updated, causing the edit to fail on the new expression. 3) Specificity. Embeddings associated with shorter or more frequently used �-grams are also activated by many unrelated inputs, so updating them may change unrelated factual predictions.

To enable decoupled knowledge updates through conditional memory, we propose EngramEdit. The key idea is to first compute memory representations that make the LLM predict the updated facts, and then jointly translate these target representations into updates to the underlying shared �-gram embeddings. As shown in Figure 1(c), we first generate multiple expressions of each fact to broaden �-gram coverage. For each edit, a shared learnable perturbation is temporarily added to the memory representations of its expressions. Optimizing only this perturbation by backpropagating the prediction loss for the new fact produces a target memory representation for each expression (see Sec. 3.1). We then map each target to the �-gram embeddings used by its expression, identifying embeddings shared across expressions and edits (see Sec. 3.2). The embedding updates are solved jointly to match these targets, with each shared embedding receiving one update that accounts for all expressions and edits using it. To preserve unrelated knowledge, we include stronger penalties on large updates to frequently reused embeddings in this joint optimization (see Sec. 3.3).

Our experiments show that EngramEdit enables efective and generalizable factual knowledge updates through conditional memory while largely preserving unrelated knowledge and general capabilities. 1) Factual knowledge can be efectively updated. By updating only conditional memory, EngramEdit achieves near-perfect editing success on both CounterFact [9] and ZsRE [10]. 2) Updated knowledge remains usable. EngramEdit achieves strong generalization to unseen expressions, and the revised facts support multi-hop reasoning. On MQuAKE [11], it achieves nearly 3× the strongest baseline’s multi-hop accuracy under chain-of-thought (CoT) prompting. 3) Unrelated knowledge and general capabilities are largely preserved. Overall performance on unrelated factual queries and general-ability tasks remains largely unchanged during extended sequential editing. Together, these findings show that EngramEdit turns conditional memory into an editable knowledge interface for decoupled knowledge updates. This extends its role beyond model scaling and opens a promising path toward LLMs that continually update their knowledge while largely preserving their general capabilities.

## 2 Preliminaries

This section introduces factual knowledge editing (KE) and the structure of conditional memory. Factual Knowledge Editing. KE aims to update specific facts in a pretrained LLM M while preserving unrelated knowledge [9, 12]. Each edit request $e _ { i } = ( s _ { i } , r _ { i } , o _ { i } \implies o _ { i } ^ { \star } )$ replaces the original object $o _ { i }$ with the desired object $o _ { i } ^ { \star }$ for subject $s _ { i }$ and relation $r _ { i } , \mathrm { e . g . }$ , changing Alex’s employer from Lab M to Lab N. Given an edit prompt $x _ { i }$ expressing $( s _ { i } , r _ { i } )$ , the edited model should predict the desired object $o _ { i } ^ { \star }$ . In sequential editing, we process batches $\mathcal { E } = \left\{ e _ { i } \right\} _ { i = \mathrm { { \bar { \Phi } } } } ^ { m }$ of � requests, applying each batch to the model updated by preceding batches.

EngramEdit shares a basic idea with locate-then-edit methods [9, 12, 13, 14]: first find a representation that makes the model predict the desired object, then update the parameters that produce it. We introduce this process through feed-forward network (FFN) editing. The editor first selects an FFN layer ℓ and a subject-token position $q _ { i }$ associated with factual recall, typically the last subject token [9]. At this location, the FFN output projection $\mathbf { W } _ { \mathrm { o u t } } ^ { \ell }$ maps its input activation $\mathbf { k } _ { i } ^ { \ell }$ to the current output $\mathbf { v } _ { i } ^ { \ell } = \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \mathbf { k } _ { i } ^ { \ell }$ . The editor temporarily adds a learnable perturbation $\pmb { \delta } _ { i }$ to this output and optimizes it by backpropagating the prediction loss for $o _ { i } ^ { \star }$ , with all model parameters fixed. The optimized perturbation $\pmb { \delta } _ { i } ^ { \star }$ gives the target output

$$
\mathbf { v } _ { i } ^ { \star } = \mathbf { v } _ { i } ^ { \ell } + \pmb { \delta } _ { i } ^ { \star } .\tag{1}
$$

The editor then solves for and applies a parameter update $\Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell }$ so that the FFN produces this target without the temporary perturbation (see App. A.1 for the full formulation):

$$
( \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } + \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } ) \mathbf { k } _ { i } ^ { \ell } \approx \mathbf { v } _ { i } ^ { \star } .\tag{2}
$$

Conditional Memory via �-gram Embeddings. Conditional memory associates each �-gram $g$ with a learned embedding e(�) used when � is activated [3, 6, 5]. We treat each embedding as one logical vector, abstracting the finite hashed tables and sub-tables used to control storage costs in practical architectures [3, 6]. The same �-gram uses the same embedding across inputs.

![](images/2d93f8eba5f58b88e7b0066763f30eae18135ef0ff2e2682332e6d0aa5aa1adc.jpg)  
Figure 2: Overview of EngramEdit. We compute target memory representations that predict updated facts across expressions, map expressions to their activated �-grams, and jointly update the embeddings to match these targets with stronger penalties for shorter or more frequent �-grams.

Given an input token sequence $x ~ = ~ ( t _ { 1 } , \ldots , t _ { T } )$ , let $g _ { q , n } ( x ) \ = \ t _ { q - n + 1 : q }$ denote the length-� sufix �-gram ending at position �. For the configured length set N, the activated �-gram set is ${ \mathcal { G } } ( x , q ) = \{ g _ { q , n } ( x ) : n \in N , n \leq q \}$ . We write the contribution of the activated embeddings to the backbone computation as a memory representation h(�, �):

$$
\mathbf { h } ( x , q ) = \mathrm { A g g } _ { x , q } \left( \{ \mathbf { e } ( g ) : g \in \mathcal { G } ( x , q ) \} \right) ,\tag{3}
$$

where $\mathsf { A g g } _ { x , q }$ includes the architecture-specific processing of the activated embeddings and any interaction with the backbone state at position �. For example, additive aggregation combines the embeddings directly [6], whereas context-aware gated fusion uses the backbone state to control their contribution [3].

## 3 Method

In this section, we present EngramEdit for decoupled knowledge updates through conditional memory (see Figure 2). For each edit batch, we first compute target memory representations that make the LLM predict the updated facts across expressions (see Section 3.1). We then map each expression to its activated �-grams, identifying those shared across expressions and edits (see Section 3.2). Finally, we jointly solve for a single update per embedding to match the targets using reuse-based regularization (see Section 3.3). Algorithm 1 in App. A.4 summarizes the complete procedure.

## 3.1 Conditional Memory Target Computation

EngramEdit first learns how memory representations should change to predict the updated facts, providing targets for jointly allocating updates across shared �-gram embeddings.

Expression Generation. Diferent expressions of the same edit may activate diferent �-grams and use diferent embeddings. To cover more of these �-grams, we ask the model M to generate � semantically equivalent expressions of the subject–relation request for each edit $e _ { i }$ in the batch E before editing. The generation prompt encourages varied wording immediately before the subject (see App. A.2.1 for detailed prompts). Combining the generated expression set $\mathcal { P } _ { i } = \{ \widetilde { x } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ with the original edit prompt �<sub>�</sub> gives $X _ { i } = \{ x _ { i } \} \cup \mathcal { P } _ { i }$ , used for target computation and memory mapping in Section 3.2.

Joint Target Computation. We next compute target memory representations that make the LLM predict the updated fact across expressions. Since these expressions describe the same factual update, a shared perturbation allows them to jointly guide a single adjustment.

For each expression $x \in \chi _ { i }$ , let $q _ { i } ( x )$ denote its last subject-token position [9] and $\mathbf { h } _ { i } ( x ) =$ $\mathbf { h } ( x , q _ { i } ( x ) ) \in \mathbb { R } ^ { d }$ its memory representation there. Following [9, 12], we temporarily add a shared learnable perturbation $\pmb { \delta } _ { i } \in \mathbb { R } ^ { d }$ to each representation (see Figure 2). Let $\mathcal { M } _ { \delta _ { i } }$ denote the model evaluated with this perturbation. With all model parameters fixed, we optimize only $\pmb { \delta } _ { i }$ using the prediction loss for the updated fact averaged across expressions:

$$
\begin{array} { r l r } {  { \pmb { \delta } _ { i } ^ { \star } = \arg \operatorname* { m i n } _ { \pmb { \delta } _ { i } } \frac { 1 } { | \mathcal { X } _ { i } | } \sum _ { x \in \mathcal { X } _ { i } } \mathcal { L } _ { \mathrm { N L L } } \big ( \mathcal { M } _ { \pmb { \delta } _ { i } } ( x ) , o _ { i } ^ { \star } \big ) + \mathcal { R } _ { \mathrm { t a r g e t } } ( \pmb { \delta } _ { i } ) , } } \\ & { } & { \mathrm { s u b j e c t ~ t o ~ } \| \pmb { \delta } _ { i } \| _ { 2 } \le \rho \| \mathbf { h } _ { i } ( x _ { i } ) \| _ { 2 } . } \end{array}\tag{4}
$$

Here, $\mathcal { L } _ { \mathrm { N L L } }$ is the negative log-likelihood (NLL) loss for the updated fact. $\mathcal { R } _ { \mathrm { t a r g e t } }$ penalizes large perturbations and prediction changes when describing the subject [9], and the clamp factor $\rho > 0$ bounds the perturbation norm (see App. A.2.2 for details). The optimized perturbation $\pmb { \delta } _ { i } ^ { \star }$ defines a target memory representation for each expression:

$$
\mathbf { h } _ { i } ^ { \star } ( x ) = \mathbf { h } _ { i } ( x ) + { \pmb \delta } _ { i } ^ { \star } , \qquad x \in  { \cal X } _ { i } .\tag{5}
$$

## 3.2 Multi-Expression Memory Mapping

EngramEdit next links each target memory representation to the �-gram embeddings activated by its expression. Because expressions and edits may share embeddings, we construct a single mapping for the entire edit batch.

Memory Mapping. To link each target memory representation to its underlying embeddings, we identify the �-grams activated at the expression’s last subject-token position. For each expression $x \in { \mathcal { X } } _ { i : }$ , the activated �-gram set is ${ \mathcal { G } } _ { i } ( x ) = { \mathcal { G } } ( x , q _ { i } ( x ) )$ , using the lookup notation from Section 2. The union of these sets, $\begin{array} { r } { \mathscr { G } _ { i } = \bigcup _ { x \in \chi _ { i } } \mathscr { G } _ { i } ( x ) } \end{array}$ , collects the activated �-grams for edit $e _ { i } .$ . Expressions activating the same �-gram � share its embedding ${ \bf e } ( g )$ , as illustrated by “of $\operatorname { A l e x } ^ { \prime \prime }$ (see Figure 2). When combining the �-gram sets $\mathcal { G } _ { i }$ across the batch, we therefore retain each �-gram only once.

Mapping Matrix. To jointly allocate updates to shared embeddings, we record which expressions use each embedding in a batch-level mapping matrix $\mathbf { A } \in \mathbb { R } ^ { P \times Q }$ . Each of the $\begin{array} { r } { P = \sum _ { i = 1 } ^ { m } | X _ { i } | } \end{array}$ rows corresponds to one expression and its target memory representation. The $Q$ columns correspond to the distinct �-grams $\{ g _ { j } \} _ { j = 1 } ^ { Q }$ in $\cup _ { i = 1 } ^ { m } { \mathcal { G } } _ { i }$ , with column � representing the embedding $\mathbf { e } ( g _ { j } )$ . Row $p$ is indexed by an edit–expression pair $( i _ { p } , x _ { p } )$ , where the expression $x _ { p } \in \mathcal { X } _ { i _ { p } }$ . The matrix element $A _ { p j }$ is 1 if expression $x _ { p }$ activates �-gram $g _ { j }$ , and 0 otherwise:

$$
A _ { p j } = \mathbb { I } \left[ g _ { j } \in { \mathcal { G } } _ { i _ { p } } ( x _ { p } ) \right] .\tag{6}
$$

## 3.3 Joint Memory Update Allocation

Using the expression-to-�-gram mapping, EngramEdit jointly computes embedding updates to match targets while penalizing large updates to frequently reused embeddings.

Target Matching. To match each target memory representation, we express its required change in terms of updates to the activated embeddings. We derive this relation under sum aggregation, where a memory representation changes by the sum of its embedding updates [6]. For column � of the mapping matrix A, let ${ \mathbf { u } } _ { j } \in \mathbb { R } ^ { d }$ denote the update to the �-gram embedding $\mathbf { e } ( g _ { j } )$

For row � of the mapping matrix A, let $\mathbf h _ { p } = \mathbf h _ { i _ { p } } ( x _ { p } )$ denote the current memory representation and $\mathbf h _ { p } ^ { \star } = \mathbf h _ { i _ { v } } ^ { \star } ( x _ { p } )$ its target. To match this target, the combined embedding updates must approximate the diference between the two representations:

$$
\sum _ { j = 1 } ^ { Q } A _ { p j } \mathbf { u } _ { j } \approx \mathbf { h } _ { p } ^ { \star } - \mathbf { h } _ { p } = { \pmb \delta } _ { i _ { p } } ^ { \star } .\tag{7}
$$

We denote this diference by the matching target ${ \bf b } _ { p } = { \bf h } _ { p } ^ { \star } - { \bf h } _ { p }$ . Stacking the embedding updates as rows gives the embedding update matrix $\mathbf { U } = [ \mathbf { u } _ { 1 } , \ldots , \mathbf { \bar { u } } _ { Q } ] ^ { \top } \in \mathbb { R } ^ { Q \times d }$ ; stacking the matching targets gives the matching target matrix $\pmb { \mathrm { B } } \in \mathbb { R } ^ { P \times d }$ . The batch-level matching requirement is therefore $\mathbf { A U } \approx \mathbf { B }$ (see Figure 2). App. A.3.2 extends this formulation to context-aware gated aggregation used in Engram [3].

Reuse-Based Regularization. To preserve unrelated knowledge, we penalize large updates more strongly for embeddings likely to be activated by unrelated inputs. We use �-gram length and corpus frequency as indicators of activation frequency across inputs, assigning larger regularization weights $w _ { j }$ to shorter or more frequent �-grams (see App. A.3.3 for the construction of the weights). The regularization matrix is

$$
\begin{array} { r } { \Lambda = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { Q } ) , \qquad \lambda _ { j } = \lambda _ { \mathrm { r e u s e } } w _ { j } + \lambda _ { \mathrm { r i d g e } } , } \end{array}\tag{8}
$$

where $\lambda _ { \mathrm { { r e u s e } } }$ controls the reuse-based penalty, while the optional ridge term $\lambda _ { \mathrm { { r i d g e } } }$ independently sets a uniform penalty on all updates. Both coeficients are nonnegative, with $\lambda _ { j } > 0$ for every embedding. We prove that, under sum aggregation, weighting update penalties by activation probability bounds the average squared change in memory representations on unrelated inputs, providing a theoretical basis for reuse-based regularization (see App. A.5.2 for the proof and bounds on individual embedding updates).

Joint Update Solution. We combine target matching and reuse-based regularization to solve for the embedding updates:

$$
\mathbf { U } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { U } } \| \mathbf { A } \mathbf { U } - \mathbf { B } \| _ { F } ^ { 2 } + \sum _ { j = 1 } ^ { Q } \lambda _ { j } \left\| \mathbf { u } _ { j } \right\| _ { 2 } ^ { 2 } .\tag{9}
$$

By jointly considering all expressions that share an embedding, this objective balances their potentially conflicting targets (see App. A.5.1 for exact-matching conditions). With all $\lambda _ { j } > 0$ $\mathbf { A } ^ { \top } \mathbf { A } + \mathbf { A }$ is positive definite, yielding a unique update computed by solving the corresponding linear system (see App. A.5.3 for the proof):

$$
\mathbf { U } ^ { \star } = \left( \mathbf { A } ^ { \top } \mathbf { A } + \mathbf { A } \right) ^ { - 1 } \mathbf { A } ^ { \top } \mathbf { B } .\tag{10}
$$

We prove that the joint solution has two useful properties. 1) Minimum update cost. Diferent embedding updates can produce the same memory representation changes but incur diferent update costs. EngramEdit minimizes the weighted embedding update cost among all updates producing the same achieved memory representation changes $\mathbf { A U } ^ { \star }$ (see App. A.5.4 for the proof).

Table 1: Editing results on CounterFact and ZsRE. MFT-S and MFT-A denote Memory-FT (Subject) and (All Tokens). Ef. = Eficacy, Gen. = Generalization, Spec. = Specificity, Util. = Utility, Flu. = Fluency, Cons. = Consistency. Subscripts give 95% confidence-interval half-widths. Bold and underlining mark the best and second-best editing scores.
<table><tr><td rowspan="2">Method</td><td colspan="6">CounterFact</td><td colspan="4">ZsRE</td></tr><tr><td>Eff. ↑</td><td> $\mathbf { G e n . } \uparrow$ </td><td> $\tt s p e c . \uparrow$ </td><td>Util. ↑</td><td>Flu.↑</td><td>Cons. ↑</td><td>Eff.↑</td><td></td><td>Gen.↑ Spec.↑ Util.↑</td><td></td></tr><tr><td>Pre-edited</td><td> $9 . 5 _ { \pm 1 . 2 8 }$ </td><td> $1 1 . 2 _ { \pm 1 . 1 9 }$ </td><td> $8 6 . 8 _ { \pm 0 . 9 4 }$ </td><td>35.8</td><td> $5 4 4 . 0 _ { \pm 1 . 6 7 }$ </td><td> $1 5 . 0 _ { \pm 0 . 3 7 }$ </td><td> $3 6 . 9 _ { \pm 1 . 3 0 }$ </td><td> $3 6 . 5 _ { \pm 1 . 2 9 }$ </td><td> $4 1 . 6 _ { \pm 1 . 1 9 }$ </td><td>38.3</td></tr><tr><td>FT</td><td> $9 4 . 9 _ { \pm 0 . 9 6 }$ </td><td> $6 3 . 6 _ { \pm 1 . 8 3 }$ </td><td> $4 3 . 8 _ { \pm 1 . 6 3 }$ </td><td>67.4</td><td> $5 0 5 . 1 _ { \pm 2 . 0 4 }$ </td><td> $1 3 . 3 _ { \pm 0 . 3 6 }$ </td><td> $7 6 . 7 _ { \pm 1 . 1 3 }$ </td><td> $7 6 . 8 _ { \pm 1 . 1 5 }$ </td><td> $4 3 . 1 _ { \pm 1 . 2 7 }$ </td><td>65.5</td></tr><tr><td>FT-L</td><td> $1 5 . 3 _ { \pm 1 . 5 8 }$ </td><td> $1 2 . 5 _ { \pm 1 . 2 4 }$ </td><td> $\underline { { 8 3 . 3 } } _ { \pm 1 . 1 1 }$ </td><td>37.0</td><td> $5 4 2 . 0 _ { \pm 1 . 7 0 }$ </td><td> $1 4 . 7 _ { \pm 0 . 3 6 }$ </td><td> $5 8 . 0 _ { \pm 1 . 4 7 }$ </td><td> $5 4 . 4 _ { \pm 1 . 4 8 }$ </td><td> $4 5 . 8 _ { \pm 1 . 2 4 }$ </td><td>52.7</td></tr><tr><td>AdaLoRA</td><td> $6 1 . 1 _ { \pm 2 . 1 3 }$ </td><td> $2 7 . 5 _ { \pm 1 . 5 5 }$ </td><td> $5 1 . 1 _ { \pm 1 . 3 5 }$ </td><td>46.6</td><td> $5 4 4 . 3 _ { \pm 1 . 5 7 }$ </td><td> $1 2 . 9 _ { \pm 0 . 3 5 }$ </td><td> $7 0 . 9 _ { \pm 1 . 4 0 }$ </td><td> $6 4 . 7 _ { \pm 1 . 4 6 }$ </td><td> $\underline { { 4 6 . 8 _ { \pm 1 . 2 7 } } }$ </td><td>60.8</td></tr><tr><td>UnKE</td><td> $8 4 . 5 _ { \pm 1 . 5 9 }$ </td><td> $5 8 . 0 _ { \pm 1 . 8 8 }$ </td><td> $4 7 . 5 _ { \pm 1 . 6 1 }$ </td><td>63.3</td><td> $5 4 6 . 5 _ { \pm 1 . 6 4 }$ </td><td></td><td> $1 5 . 0 _ { \pm 0 . 3 5 } \underline { { 8 6 . 9 _ { \pm 1 . 1 3 } } }$ </td><td> $\underline { { 7 9 . 0 _ { \pm 1 . 3 3 } } }$ </td><td> $4 9 . 9 _ { \pm 1 . 3 2 }$ </td><td>71.9</td></tr><tr><td>MoEEdit</td><td> $\underline { { 9 9 . 1 } } _ { \pm 0 . 4 1 }$ </td><td> $\underline { { 6 7 . 1 } } _ { \pm 1 . 6 9 }$ </td><td> $6 3 . 5 _ { \pm 1 . 2 8 }$ </td><td>76.6</td><td> $5 4 4 . 7 _ { \pm 1 . 6 5 }$ </td><td></td><td> $1 4 . 7 _ { \pm 0 . 3 3 } \ 8 5 . 1 _ { \pm 0 . 9 1 }$ </td><td> $7 8 . 7 _ { \pm 1 . 2 1 }$ </td><td> $4 4 . 4 _ { \pm 1 . 2 6 }$ </td><td>69.4</td></tr><tr><td>MFT-S</td><td> $9 9 . 0 _ { \pm 0 . 4 5 }$ </td><td> $5 9 . 1 _ { \pm 1 . 9 2 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 0 }$ </td><td>81.1</td><td> $\underline { { 5 5 5 . 2 } } { \_ } 1 . 3 8$ </td><td></td><td> $1 5 . 3 _ { \pm 0 . 3 6 } 6 6 . 9 _ { \pm 1 . 3 5 }$ </td><td> $5 6 . 5 _ { \pm 1 . 5 3 }$ </td><td> $3 6 . 9 _ { \pm 1 . 1 7 }$ </td><td>53.4</td></tr><tr><td>MFT-A</td><td> $9 3 . 9 _ { \pm 1 . 0 4 }$ </td><td> $4 0 . 7 _ { \pm 1 . 7 5 }$ </td><td> $6 6 . 6 _ { \pm 1 . 1 2 }$ </td><td>67.1</td><td> $4 9 6 . 7 _ { \pm 4 . 0 4 }$ </td><td> $1 1 . 5 _ { \pm 0 . 4 0 }$ </td><td> $6 2 . 8 _ { \pm 1 . 4 9 }$ </td><td> $5 1 . 3 _ { \pm 1 . 5 7 }$ </td><td> $3 3 . 7 _ { \pm 1 . 1 3 }$ </td><td>49.3</td></tr><tr><td>EngramEdit</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.9</td><td> ${ \bf 5 6 5 . 2 _ { \pm 1 . 1 0 } }$ </td><td> $1 6 . 1 _ { \pm 0 . 3 6 }$ </td><td> $9 7 . 3 _ { \pm 0 . 4 0 }$ </td><td> $9 3 . 7 _ { \pm 0 . 7 9 }$ </td><td> $3 8 . 3 _ { \pm 1 . 1 8 }$ </td><td>76.4</td></tr></table>

2) Stability to target changes. Since targets are computed through optimization, they may vary slightly. With the mapping and regularization fixed, we prove that regularization bounds the extent to which these variations afect the solved embedding updates (see App. A.5.5 for the analysis and proof).

Memory Updating. Finally, we update conditional memory as $\mathbf { e } ( g _ { j } )  \mathbf { e } ( g _ { j } ) + \mathbf { u } _ { i } ^ { \star }$ , keeping all other parameters fixed. During inference, an embedding update enters the model’s computation only when its �-gram is active in the current prefix.

## 4 Experiments

In this section, we conduct experiments to address the following research questions:

• RQ1: How efectively does EngramEdit update factual knowledge through conditional memory while preserving unrelated knowledge?

• RQ2: How well does EngramEdit preserve general capabilities during knowledge updates?

• RQ3: How do EngramEdit’s design choices afect editing performance?

• RQ4: How do EngramEdit’s memory updates support the LLM’s use of revised knowledge?

## 4.1 Experimental Setup

Datasets and Metrics. On CounterFact [9] and ZsRE [10], we report Eficacy (success on edited prompts), Generalization (success on held-out paraphrases not used during editing), Specificity (performance on unrelated queries), and their arithmetic mean, Utility. On CounterFact, we also report Fluency and Consistency for generated text. MQuAKE [11] measures multi-hop use of updated facts through answer accuracy. Following AlphaEdit [13], we report mean task-level F1 across six general-ability tasks. See Apps. B.1 and B.2 for dataset examples and metric definitions.

Baselines and Implementation. We compare editors applicable to MoE LLMs: FT, FT-L [15], AdaLoRA [16], UnKE [17], and MoEEdit [14]. To compare direct memory fine-tuning, we include Memory-FT (Subject) and Memory-FT (All Tokens), which update �-gram embeddings activated at the last subject token or all token positions in the original edit prompts, respectively. All experiments use LongCat-Flash-Lite [6], with sequential batch editing. See Apps. B.3 and B.4 for details.

![](images/2d465739d6b7c4a33dcbb8a9a2ca22d117a534b946e52c5fe7609e4d2e6ffcc8.jpg)

![](images/f48ac7972bb22a59d9eed016884c98215e134d390d91fe057d6949e7d9b08435.jpg)  
(b) CoT Accuracy by Hop  
Figure 3: Editing results on MQuAKE. (a) Standard (left) and CoT (right) prompting with 95% confidence intervals. (b) CoT accuracy by hop. AdaL and MoE denote AdaLoRA and MoEEdit.

## 4.2 Knowledge Update and Preservation (RQ1)

To evaluate how efectively EngramEdit updates facts through conditional memory, we compare it with the baselines after 2,000 sequential edits on each of CounterFact and ZsRE (see Table 1). We further apply edits from 3,000 MQuAKE cases to test whether the updated facts support multi-hop reasoning (see Figure 3). We apply standard prompting for direct answers and CoT prompting for step-by-step reasoning before answering. We observe that:

• On CounterFact, EngramEdit achieves similar Eficacy to MoEEdit and MFT-S but much higher Generalization; on ZsRE, it leads in both metrics (see Table 1). Comparable success on edited prompts, therefore, does not imply comparable recall through other expressions. EngramEdit addresses this gap by updating embeddings across expressions, since diferent wordings can activate diferent �-grams. Its advantages persist through 5,000 sequential edits, showing that revised facts remain accessible as further updates accumulate (see App. C.1.1 for details).

• On MQuAKE, EngramEdit, and FT have similar accuracy under standard prompting. With CoT, EngramEdit achieves nearly 3× the accuracy of the strongest baseline and leads across all hop groups, as shown in Figure 3. This improvement shows that facts updated through conditional memory can be used in multi-hop reasoning. During CoT reasoning, intermediate steps can activate updated embeddings for the facts used, allowing revised information to guide subsequent steps. The advantage extends through four-hop cases, supporting the use of updated knowledge in longer reasoning chains (see App. C.1.2 for full results).

• On ZsRE, EngramEdit’s Specificity remains close to the pre-edit level, whereas FT-L, AdaLoRA, UnKE, and MoEEdit exceed it. These higher scores do not by themselves establish better preservation, because Specificity counts newly correct answers as well as retained correct answers. The correctness transition analysis shows that corrections to previously wrong, unrelated predictions contribute to the baselines’ gains and that EngramEdit retains most initially correct predictions (see App. C.1.3 for further analysis). This provides additional evidence that its knowledge updates largely preserve performance on unrelated queries.

• EngramEdit achieves the highest Utility on CounterFact and ZsRE, with Specificity close to pre-edit levels, and also leads in Fluency and Consistency. Together, these results show that EngramEdit supports efective and generalizable knowledge updates through conditional memory while largely preserving unrelated knowledge.

## 4.3 General Capabilities (RQ2)

To assess whether EngramEdit preserves general capabilities as knowledge updates accumulate, we evaluate six general-ability tasks in 5,000 sequential CounterFact edits. We compare mean task-level F1 and per-task changes against pre-edit performance. The results in Figure 4 show that:

![](images/0ad39408b14f0a85dd3d267533d14cb8ec7a238e10ae7e104f84e4e573b231db.jpg)  
(a) Overall Trend

![](images/8d837c3079ed422c972af588badae8e23bb7cfb2191b8a128d1dc1c7024ee334.jpg)  
(b) Change after 5,000 Edits  
Figure 4: General capabilities during sequential CounterFact editing. (a) Mean F1 across six tasks in the editing sequence. (b) Task-level F1 changes from pre-edit scores after 5,000 edits, in percentage points (pp). The mean |Δ| averages the absolute task-level changes.

• EngramEdit retains over 96% of its pre-edit mean F1 after 5,000 sequential edits, with modest changes throughout the editing sequence, as shown in Figure 4(a). MFT-S also largely retains pre-edit mean F1, but EngramEdit achieves much stronger editing Generalization (see Section 4.2). Both methods restrict editing to selected �-gram embeddings, and EngramEdit further penalizes large changes to frequently reused embeddings to limit efects on unrelated tasks. Together, these results support conditional memory as an editable knowledge interface and show that EngramEdit combines stronger cross-expression recall with largely preserved general capabilities.

• EngramEdit shows small F1 changes on most tasks, with a larger decline on MRPC (see Figure 4(b) and full trajectories in App. C.1.4). MRPC tests whether two sentences express the same meaning. These sentences can activate diferent �-grams, so local embedding updates may afect their representations diferently, altering the equivalence judgment.

## 4.4 In-Depth Analysis

Ablation Study (RQ3). To isolate each component’s contribution, we compare five variants on CounterFact and ZsRE, with other settings held fixed. w/o Joint solves embedding updates independently for each expression. w/o Freq. and w/o Len. remove the weighting that increases penalties for more frequent and shorter �-grams, respectively. w/o Reg. removes reuse-based regularization, and w/o Reg. & Expr. additionally removes generated expressions. From Table 2, we observe that: 1) Removing joint allocation substantially lowers Eficacy and Generalization on both datasets. This shows that jointly considering the targets of all expressions sharing an embedding helps the model recall revised facts across expressions. 2) Removing generated expressions from the unregularized variant substantially lowers Generalization. Generated expressions, therefore, mainly support cross-expression recall by covering the diferent �-grams activated by alternative wordings. 3) Removing reuse-based regularization lowers Eficacy and Specificity, and removing either frequency or length weighting also reduces Utility. The regularization thus helps preserve unrelated knowledge by limiting changes to embeddings reused by other inputs. Our analysis further shows that over 92% of new errors on unrelated queries involve activation of updated embeddings (see App. C.3.2).

Analysis of Generated Expressions (RQ3). To distinguish the benefit of additional expressions from how they are used, we first compare MoEEdit, MFT-S, MFT-A, and EngramEdit with the same generated expressions per fact on CounterFact (see Figure 5(a)). We then vary expression use in EngramEdit’s target computation and memory mapping, keeping other settings fixed (see Figure 5(b)). All variants retain the original edit prompt. Coverage is the percentage of held-out paraphrases that activate an embedding updated for the same fact. We observe that: 1) With the same generated expressions, EngramEdit outperforms MFT-S in Generalization at similar Eficacy and Specificity. EngramEdit’s target computation and joint memory updating thus use these expressions more efectively for recall under diferent wordings (see App. C.2.1 for full results). 2) Using generated expressions for memory mapping increases Coverage and yields most of the Generalization gain, because the expanded set of updated �-grams allows more held-out paraphrases to activate fact-related embeddings. An additional analysis of memory activation supports this explanation: the roughly 2% of held-out paraphrases activating no fact-related embeddings account for half of generalization failures (see App. C.3.2). With Coverage unchanged, also using these expressions for target computation further improves Generalization by optimizing targets across expressions.

Table 2: Ablations of joint update allocation (Joint), reuse-based regularization $( \mathrm { R e g . } )$ , �-gram corpus frequency (Freq.) and length (Len.) weighting, and generated expressions (Expr.).
<table><tr><td rowspan="2">Variant</td><td colspan="6">CounterFact</td><td colspan="4">ZsRE</td></tr><tr><td>Eff. ↑</td><td> $\mathbf { G e n . } \uparrow$ </td><td>Spec. ↑</td><td>Util. ↑</td><td>Flu.↑</td><td>Cons. ↑</td><td>Eff. ↑</td><td>Gen.↑</td><td>Spec. ↑</td><td>Util. ↑</td></tr><tr><td>EngramEdit</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.9</td><td> $5 6 5 . 2 _ { \pm 1 . 1 0 }$ </td><td> $\underline { { 1 6 . 1 _ { \pm 0 . 3 6 } } }$ </td><td> $9 7 . 3 _ { \pm 0 . 4 0 }$ </td><td> $\underline { { 9 3 . 7 } } _ { \pm 0 . 7 9 }$ </td><td> $3 8 . 3 _ { \pm 1 . 1 8 }$ </td><td>76.4</td></tr><tr><td>w/o Joint</td><td> $7 4 . 5 _ { \pm 1 . 9 1 }$ </td><td> $7 2 . 8 _ { \pm 1 . 7 4 }$ </td><td> $8 4 . 8 _ { \pm 0 . 9 3 }$ </td><td>77.4</td><td> ${ \bf 5 6 8 . 0 _ { \pm 1 . 0 0 } }$ </td><td> $1 5 . 3 _ { \pm 0 . 3 5 }$ </td><td> $5 7 . 4 _ { \pm 1 . 5 5 }$ </td><td> $5 8 . 3 _ { \pm 1 . 5 7 }$ </td><td> $3 5 . 6 _ { \pm 1 . 1 7 }$ </td><td>50.4</td></tr><tr><td>w/o Freq.</td><td> $9 9 . 4 _ { \pm 0 . 3 2 }$ </td><td> $\underline { { 9 6 . 8 _ { \pm 0 . 6 9 } } }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.8</td><td> $5 6 4 . 9 _ { \pm 1 . 2 2 }$ </td><td> $\underline { { 1 6 . 1 _ { \pm 0 . 3 6 } } }$ </td><td> $9 6 . 0 _ { \pm 0 . 5 2 }$ </td><td> $9 2 . 6 _ { \pm 0 . 8 4 }$ </td><td> $\underline { { 3 7 . 3 _ { \pm 1 . 1 9 } } }$ </td><td>75.3</td></tr><tr><td>w/o Len.</td><td> $9 9 . 3 _ { \pm 0 . 3 8 }$ </td><td> $\underline { { 9 6 . 8 _ { \pm 0 . 7 0 } } }$ </td><td> $\underline { { 8 5 . 1 _ { \pm 0 . 9 3 } } }$ </td><td>93.7</td><td> $5 6 5 . 4 _ { \pm 1 . 0 9 }$ </td><td> $1 6 . 2 _ { \pm 0 . 3 6 }$ </td><td> $9 5 . 7 _ { \pm 0 . 5 5 }$ </td><td> $9 3 . 4 _ { \pm 0 . 7 5 }$ </td><td> $3 7 . 2 _ { \pm 1 . 1 7 }$ </td><td>75.4</td></tr><tr><td>w/o Reg.</td><td> $9 8 . 8 _ { \pm 0 . 4 7 }$ </td><td> $\underline { { 9 6 . 8 _ { \pm 0 . 7 2 } } }$ </td><td> $8 4 . 6 _ { \pm 0 . 9 3 }$ </td><td>93.4</td><td> ${ \bf 5 6 8 . 0 _ { \pm 1 . 0 2 } }$ </td><td> $1 5 . 5 _ { \pm 0 . 3 5 }$ </td><td> $\underline { { 9 6 . 0 _ { \pm 0 . 5 4 } } }$ </td><td> $9 4 . 0 _ { \pm 0 . 7 5 }$ </td><td> $3 7 . 2 _ { \pm 1 . 1 9 }$ </td><td>75.7</td></tr><tr><td>w/o Reg. &amp; Expr.</td><td> $9 8 . 8 _ { \pm 0 . 4 7 }$ </td><td> $8 1 . 5 _ { \pm 1 . 5 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>88.5</td><td> $\underline { { 5 6 5 . 9 _ { \pm 1 . 0 8 } } }$ </td><td> $1 5 . 6 _ { \pm 0 . 3 6 }$ </td><td> $9 3 . 9 _ { \pm 0 . 7 2 }$ </td><td> $8 5 . 4 _ { \pm 1 . 1 9 }$ </td><td> $3 7 . 0 _ { \pm 1 . 1 7 }$ </td><td>72.1</td></tr></table>

![](images/23f33887825599dee181475e1b475a42e8717f9baf0cc540c1d6eeaf77c2cc34.jpg)

![](images/34bf16601c191343bd22d0dd88c401bae26d4fe1a81d77757f81032cad043826.jpg)

![](images/e4cbacc9ba44cfd026fd973dc88b811d5f1695e3207f6ce804be98c22c4b6f12.jpg)  
Figure 5: Generated expressions and memory use on CounterFact. (a) Editors use the same generated expressions per fact. (b) Generated expressions are used for target computation, memory mapping, both, or neither. Coverage is the percentage of held-out paraphrases activating fact-related updated embeddings. (c) Disabling no updates (None), fact-related updates (Related), matched random updates (Random), or all updates (All) during inference.

Analysis of Memory Use (RQ4). To test whether the LLM uses the updated conditional memory to predict revised facts, we compare normal inference with disabling updates for the evaluated fact, randomly selecting other updates matched in number and �-gram length, or all updates. All conditions use the same model after CounterFact editing and retain the original embeddings. As shown in Figure 5(c), fact-related disabling sharply lowers Eficacy and Generalization, almost matching the efect of disabling all updates. In contrast, matched random disabling leaves Eficacy and Generalization unchanged. An additional analysis of prediction preferences shows that factrelated disabling makes the model favor original over revised facts on average (see App. C.3.1). These results show that the LLM relies on the fact-related memory updates to access and utilize revised knowledge.

Additional Analyses. We extend editing to 5,000 facts to test stability (see App. C.1.1) and analyze ZsRE preservation (see App. C.1.3). We vary the number of generated expressions, editable �-gram lengths, regularization coeficients, and the clamp factor (see Apps. C.2.2–C.2.4). We further analyze memory use through memory activation and cross-edit sharing (see Apps. C.3.1– C.3.3).

## 5 Related Work

Conditional Memory and �-gram Embeddings. Lookup-based memory architectures expand model capacity through sparse access to stored vectors, as in Product-Key Memory [18] and Memory Layers at Scale [19]. A related line uses �-gram embeddings to associate local token or byte sequences with reusable representations. 1) Model scaling. Byte Latent Transformer [20], OverEncoding [21], and SCONE [22] use �-gram representations to enrich model inputs with local context at limited inference cost. Engram [3] and LongCat-Flash-Lite [6] further explore �-grambased memory to scale LLM capacity. For memory construction, Memory Grafting [23] builds frozen conditional memory ofline from another model’s hidden states for pre-training. Related �-gram-based designs are also used in Qwen3.8-Flash-Next [5] and DeepSeek-V4.1-Flash [4]. 2) Model adaptation. Conditional memory has also been adapted for personalization through user-specific embedding updates in User as Engram [7] and for domain adaptation in Engram Adapter [8]. Our work develops EngramEdit to use pretrained conditional memory as an editable knowledge interface, enabling decoupled factual updates while largely preserving unrelated knowledge and general capabilities.

Knowledge Editing. KE methods can be grouped according to whether they modify the LLM’s original parameters. 1) Parameter-preserving methods. These methods supply updated knowledge through external information or additional components, keeping the original parameters fixed. SERAC [24] uses an edit cache, and IKE [25] uses in-context demonstrations. Larimar [26], WISE [27], and NeuralDB [28] use additional weights or memory components. 2) Parametermodifying methods. More directly related to our work, these methods update existing model parameters. Constrained fine-tuning [15] limits parameter changes during optimization, while MEND [29] learns to transform editing gradients into weight updates. ROME [9] and MEMIT [12] use locate-then-edit to first compute target FFN outputs for new facts, then solve for the corresponding weight updates. UnKE [17] extends parameter-modifying editing to unstructured knowledge. To preserve unrelated knowledge during parameter updates, AlphaEdit [13] projects parameter changes into the null space of a preservation set. For diferent model architectures, STEM [30] revises facts through token-indexed embedding substitution, while MoEEdit [14] jointly updates MoE experts with constraints on routing shifts. EngramEdit focuses on realizing the potential of pretrained conditional memory for decoupled knowledge updates.

## 6 Conclusion and Future Work

In this work, we propose EngramEdit for decoupled knowledge updates via conditional memory, jointly updating shared �-gram embeddings with reuse-based regularization. Experiments show high editing success and generalization, multi-hop use of updated facts, and largely preserved unrelated knowledge and general capabilities. These findings show that EngramEdit enables decoupled knowledge updates in LLMs through conditional memory, extending its role beyond model scaling to an editable knowledge interface. Future work can explore other conditiona memory architectures and model scales, and eficient, stable updates to more facts at higher frequencies.

## AI use statement

In this work, we used generative AI tools to assist with method implementation, proof drafting, manuscript writing and revision, and figure creation. AI tools also assisted in discussing and refining the interpretation of experimental results. The authors reviewed the method implementation and proofs and tested the code. We take full responsibility for the final content of this work, including all AI-assisted text, claims, and artifacts.

## Ethics statement

EngramEdit is intended to help LLMs keep factual knowledge up to date, but the ability to revise knowledge can also be misused to introduce false or misleading content. Our evaluation uses existing benchmarks, where counterfactual edit targets serve as experimental test cases and should not be treated as verified real-world facts. Responsible use therefore requires verifying factual updates, restricting editing access to authorized users, and auditing intended and unintended efects.

## Reproducibility statement

All the results in this work are reproducible. Section 3 describes the method, with the complete editing procedure, theoretical assumptions, and proofs provided in Appendix A. Section 4.1 and Appendix B document the datasets, evaluation protocols, and implementation settings. The code is available in our repository.

## References

[1] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017.

[2] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021.

[3] Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Han Zhang, Yu-Kun Li, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. Conditional memory via scalable lookup: A new axis of sparsity for large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026.

[4] DeepSeek-AI. Deepseek-v4.1-flash: Pushing the limits of kv cache compression. arXiv preprint arXiv:2609.19969, 2026.

[5] Zihan Qiu, Zekun Wang, Xiao Li, Yanpeng Li, Yang Xu, Yixuan Wang, Huaqing Zhang, Rui Men, Bochao Mao, Chengruidong Zhang, Fan Zhou, Hao Luo, Haofeng Huang, Haoran Lian, Haoyan Huang, Hongqing Chen, Jianwei Zhang, Jing Xu, Junjie Wang, Langshi Chen, Liangyu Wang, Linlang Jiang, Man Yuan, Minmin Sun, Peng Jin, Siqi Zhang, Siyu Wang, Xingzhang Ren, Yakai Wang, Yi Zhang, Yiming Dong, Yizhong Cao, Yubo Ma, Yunfei Mao, Bo Zheng, and Dayiheng Liu. On the design of Qwen3.8-Next architecture: Evaluation, eficiency, and training stability. arXiv preprint arXiv:2608.30320, 2026.

[6] Hong Liu, Jiaqi Zhang, Chao Wang, Xing Hu, Linkun Lyu, Jiaqi Sun, Xurui Yang, Bo Wang, Fengcun Li, Yulei Qian, Lingtong Si, Yerui Sun, Rumei Li, Peng Pei, Yuchen Xie, and Xunliang Cai. Scaling embeddings outperforms scaling experts in language models. arXiv preprint arXiv:2601.21204, 2026.

[7] Bojie Li. User as engram: Internalizing per-user memory as local parametric edits. arXiv preprint arXiv:2606.19172, 2026.

[8] Jiayu Hou and Lei Wang. When to adapt: Conditional memory adapters for retention-preserving domain specialization. arXiv preprint arXiv:2608.29327, 2026.

[9] Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, 2022.

[10] Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. In Proceedings of the 21st Conference on Computational Natural Language Learning, pages 333–342. Association for Computational Linguistics, 2017. doi: 10.18653/v1/K17-1034.

[11] Zexuan Zhong, Zhengxuan Wu, Christopher Manning, Christopher Potts, and Danqi Chen. MQuAKE: Assessing knowledge editing in language models via multi-hop questions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.971.

[12] Kevin Meng, Arnab Sen Sharma, Alex J Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In The Eleventh International Conference on Learning Representations, 2023.

[13] Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, and Tat-Seng Chua. AlphaEdit: Null-space constrained knowledge editing for language models. In The Thirteenth International Conference on Learning Representations, 2025.

[14] Yupu Gu, Rongzhe Wei, Andy Zhu, and Pan Li. MoEEdit: Eficient and routing-stable knowledge editing for mixture-of-experts LLMs. In The Fourteenth International Conference on Learning Representations, 2026.

[15] Chen Zhu, Ankit Singh Rawat, Manzil Zaheer, Srinadh Bhojanapalli, Daliang Li, Felix Yu, and Sanjiv Kumar. Modifying memories in transformer models. arXiv preprint arXiv:2012.00363, 2020.

[16] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-eficient fine-tuning. In The Eleventh International Conference on Learning Representations, 2023.

[17] Jingcheng Deng, Zihao Wei, Liang Pang, Hanxing Ding, Huawei Shen, and Xueqi Cheng. Everything is editable: Extend knowledge editing to unstructured data in large language models. In The Thirteenth International Conference on Learning Representations, 2025.

[18] Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Herve Jegou. Large memory layers with product keys. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, 2019.

[19] Vincent-Pierre Berges, Barlas Oguz, Daniel Haziza, Wen-Tau Yih, Luke Zettlemoyer, and Gargi Ghosh. Memory layers at scale. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 3831–3842. PMLR, 2025.

[20] Artidoro Pagnoni, Ramakanth Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason E Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, and Srini Iyer. Byte latent transformer: Patches scale better than tokens. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025.

[21] Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, and Zhou Xun. Overtokenized transformer: Vocabulary is generally worth scaling. In Forty-second International Conference on Machine Learning, 2025.

[22] Da Yu, Edith Cohen, Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Daogao Liu, and Chiyuan Zhang. Scaling embedding layers in language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[23] Runxi Cheng, Yuchen Guan, Yongxian Wei, Qianpu Sun, Qixiu Li, Sinan Du, Feng Xiong, Chun Yuan, Yan Lu, and Yeyun Gong. Memory grafting: Scaling language model pre-training via ofline conditional memory. arXiv preprint arXiv:2605.20948, 2026.

[24] Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. Memorybased model editing at scale. In Proceedings of the 39th International Conference on Machine Learning,

Proceedings of Machine Learning Research, 2022.

[25] Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. Can we edit factual knowledge by in-context learning? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, 2023. Association for Computational Linguistics.

[26] Payel Das, Subhajit Chaudhury, Elliot Nelson, Igor Melnyk, Sarathkrishna Swaminathan, Sihui Dai, Aurelie Lozano, Georgios Kollias, Vijil Chenthamarakshan, Jiri Navratil, Soham Dan, and Pin-Yu Chen. Larimar: Large language models with episodic memory control. In Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, 2024.

[27] Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. Wise: rethinking the knowledge memory for lifelong model editing of large language models. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, 2024.

[28] Weizhi Fei, Hao Shi, Jing Xu, Jingchen Peng, Jiazheng Li, Jingzhao Zhang, Bo Bai, Wei Han, Zhenyuan Chen, and Xueyan Niu. Scaling knowledge editing in LLMs to 100,000 facts with neural KV database. In The Fourteenth International Conference on Learning Representations, 2026.

[29] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. Fast model editing at scale. In International Conference on Learning Representations, 2022.

[30] Ranajoy Sadhukhan, Sheng Cao, Harry Dong, Changsheng Zhao, Attiano Purpura-Pontoniere, Yuan dong Tian, Zechun Liu, and Beidi Chen. STEM: SCALING TRANSFORMERS WITH EMBEDDING MODULES. In The Fourteenth International Conference on Learning Representations, 2026.

[31] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, 2013.

[32] William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005.

[33] Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. Neural network acceptability judgments. Transactions of the Association for Computational Linguistics, 2019.

[34] Luisa Bentivogli, Ido Dagan, Hoa Trang Dang, Danilo Giampiccolo, and Bernardo Magnini. The fifth PASCAL recognizing textual entailment challenge. In Proceedings of the Second Text Analysis Conference, 2009.

[35] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networksfor NLP, 2018.

[36] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

## A Method Details and Proofs

This section provides the background formulation, additional method details, and proofs referenced in Sections 2 and 3.

## A.1 Locate-Then-Edit Formulation

Locate-then-edit first identifies an internal computation associated with the factual association being edited, then determines the output that this computation should produce for the desired prediction, and finally updates the corresponding parameters to reproduce that output. We illustrate this process using a single FFN output projection. Specific editors may use diferent layer-selection strategies, preservation constraints, or solvers [9, 12, 13, 14].

Causal Localization. For the edit $e _ { i } = ( s _ { i } , r _ { i } , o _ { i }  o _ { i } ^ { \star } )$ , causal tracing measures how strongly an internal state contributes to recalling the original object $o _ { i }$ [9]. The procedure first runs the edit prompt $x _ { i }$ with the subject-token embeddings corrupted. Let $p _ { i } ^ { \mathrm { c o r r } }$ denote the resulting probability assigned to $o _ { i } .$ . It then restores the clean hidden state at token position $q$ and layer $\ell ,$ retaining the input corruption and recomputing downstream states. Let $p _ { i } ^ { \mathrm { r e s t o r e } } ( \ell , q )$ denote the probability after this restoration. The indirect efect of the restored state is

$$
\operatorname { I E } _ { i } ( \ell , q ) = p _ { i } ^ { \mathrm { r e s t o r e } } ( \ell , q ) - p _ { i } ^ { \mathrm { c o r r } } .\tag{11}
$$

A large indirect efect indicates that the state at $( \ell , q )$ contributes strongly to recalling the fact. This analysis identifies, or motivates a fixed choice of, the FFN layer ℓ and subject-token position $q _ { i }$ used for editing. At the selected location, let $\mathbf { k } _ { i } ^ { \ell }$ be the post-nonlinearity FFN activation and $\mathbf { W } _ { \mathrm { o u t } } ^ { \ell }$ the FFN output projection. In practice, the activation $\mathbf { k } _ { i } ^ { \ell }$ can be averaged over prompts that place the same subject in diferent contexts to reduce dependence on one prompt [9, 12]. The corresponding FFN output is

$$
\mathbf { v } _ { i } ^ { \ell } = \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \mathbf { k } _ { i } ^ { \ell } .\tag{12}
$$

The remaining two stages determine the output that should replace $\mathbf { v } _ { i } ^ { \ell }$ and the parameter update that realizes this replacement.

Target Output Computation. Target output computation asks what FFN output at the localized position would make the model predict the desired object $o _ { i } ^ { \star }$ . Let $C _ { i }$ contain the edit prompt $x _ { i }$ and any context-augmented prompts used during this computation. With all model parameters fixed, a learnable perturbation $\pmb { \delta } _ { i }$ is temporarily added to the FFN output at the selected position for every prompt in $C _ { i }$ . We denote the model evaluated with this temporary perturbation by $\mathcal { M } _ { \delta _ { i } }$ The perturbation is obtained by minimizing

$$
\begin{array} { r l } & { { \pmb \delta } _ { i } ^ { \star } = \arg \operatorname* { m i n } _ { { \pmb \delta } _ { i } } \frac { 1 } { | C _ { i } | } \displaystyle \sum _ { x \in C _ { i } } \mathcal { L } _ { \mathrm { N L L } } \big ( \mathcal { M } _ { \pmb \delta _ { i } } ( x ) , o _ { i } ^ { \star } \big ) } \\ & { ~ + ~ \lambda _ { \mathrm { K L } } D _ { \mathrm { K L } } \big ( p _ { \pmb \delta _ { i } } ^ { \mathrm { p r e } } \parallel p _ { 0 } ^ { \mathrm { p r e } } \big ) , } \end{array}\tag{13}
$$

where $p _ { 0 } ^ { \mathrm { p r e } }$ and $p _ { \delta _ { i } } ^ { \mathrm { p r e } }$ are the predictive distributions on a preservation prompt before and during the temporary perturbation, respectively [9]. The NLL term promotes the desired prediction, while the KL term limits changes to the preservation distribution. The optimized perturbation defines the target FFN output

$$
\mathbf { v } _ { i } ^ { \star } = \mathbf { v } _ { i } ^ { \ell } + \pmb { \delta } _ { i } ^ { \star } .\tag{14}
$$

The perturbation $\pmb { \delta } _ { i } ^ { \star }$ is used only to determine this target. It is removed after optimization and is not written to the model.

Parameter Update. The final stage changes the FFN output projection so that it produces the target output from the same activation. For a batch of � edits applied to layer $\ell ,$ define the activation matrix $\mathbf { K } = [ \mathbf { k } _ { 1 } ^ { \ell } , \ldots , \mathbf { k } _ { m } ^ { \ell } ]$ and the target output matrix $\mathbf { V } ^ { \star } = \left[ \mathbf { v } _ { 1 } ^ { \star } , \ldots , \mathbf { v } _ { m } ^ { \star } \right]$ . The parameter update $\Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell }$ should satisfy

$$
( \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } + \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } ) \mathbf { K } \approx \mathbf { V } ^ { \star } .\tag{15}
$$

Matching only the new targets can change outputs associated with other FFN activations. Let ${ \bf K } _ { 0 }$ contain preservation activations sampled from the model. Because their original outputs are $\mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \mathbf { K } _ { 0 }$ , preserving these associations amounts to keeping $\Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \mathbf { K } _ { 0 }$ small. With nonnegative coeficients $\lambda _ { \mathrm { p r e s e r v e } }$ and $\lambda _ { \mathrm { { r i d g e } } } ,$ , a regularized batch formulation is

$$
\begin{array} { r l } & { \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell \star } = \underset { \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } } { \mathrm { a r g } } \| ( \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } + \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } ) \mathbf { K } - \mathbf { V } ^ { \star } \| _ { F } ^ { 2 } } \\ & { \qquad +  \lambda _ { \mathrm { p r e s e r v e } } \| \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \mathbf { K } _ { 0 } \| _ { F } ^ { 2 } +  \lambda _ { \mathrm { r i d g e } } \| \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \| _ { F } ^ { 2 } . } \end{array}\tag{16}
$$

Define the target residual matrix and the regularized covariance matrix as

$$
\mathbf { R } = \mathbf { V } ^ { \star } - \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \mathbf { K } , \qquad \mathbf { C } _ { 0 } = \lambda _ { \mathrm { p r e s e r v e } } \mathbf { K } _ { 0 } \mathbf { K } _ { 0 } ^ { \top } + \lambda _ { \mathrm { r i d g e } } \mathbf { I } .\tag{17}
$$

When $\mathbf { K } \mathbf { K } ^ { \top } + \mathbf { C } _ { 0 }$ is invertible, setting the derivative of Eq. 16 to zero gives

$$
\begin{array} { r } { \Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell \star } = \mathbf { R } \mathbf { K } ^ { \top } \left( \mathbf { K } \mathbf { K } ^ { \top } + \mathbf { C } _ { 0 } \right) ^ { - 1 } , } \end{array}\tag{18}
$$

which jointly writes a batch of target FFN outputs while regularizing changes on preservation activations [12]. A positive $\lambda _ { \mathrm { { r i d g e } } }$ ensures this invertibility.

For a single edit, ROME instead requires exact target matching and minimizes the preservation error under this constraint [9]. With ${ \bf C } _ { 0 } = { \bf K } _ { 0 } { \bf K } _ { 0 } ^ { \top }$ invertible and $\mathbf { k } _ { i } ^ { \ell } \neq \mathbf { 0 } .$ , the resulting rank-one update is

$$
\Delta \mathbf { W } _ { \mathrm { o u t } } ^ { \ell \star } = \frac { ( \mathbf { V } _ { i } ^ { \star } - \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \mathbf { k } _ { i } ^ { \ell } ) ( \mathbf { C } _ { 0 } ^ { - 1 } \mathbf { k } _ { i } ^ { \ell } ) ^ { \top } } { ( \mathbf { C } _ { 0 } ^ { - 1 } \mathbf { k } _ { i } ^ { \ell } ) ^ { \top } \mathbf { k } _ { i } ^ { \ell } } .\tag{19}
$$

Later editors modify the preservation constraint, the parameter blocks being updated, or the batch solver, while retaining the separation between target output computation and persistent parameter updating [13, 14].

## A.2 Conditional Memory Target Computation

We provide the prompt for expression generation and the full objective for target computation in Section 3.1.

## A.2.1 Expression Generation

To cover diferent �-grams activated at the last subject token, we ask the model to vary the wording immediately before the subject. Figure 6 shows the prompt used for CounterFact and each atomic edit in MQuAKE. Here, <original stem> is the edit prompt $x _ { i }$ with the subject $s _ { i }$ replaced by [SUBJ]; <old answer> and <new answer> are the original and desired objects. For ZsRE, the instruction instead requests a question with the same meaning and answer type.

The instruction is passed as a user message through the model’s chat template. We sample candidates in batches and filter duplicates, malformed outputs, and outputs containing answer strings. Later rounds append accepted expressions and ask the model to vary their wording before [SUBJ]. Generation stops once enough expressions are accepted or the round limit is reached. We restore the original subject in each accepted expression before target computation.

![](images/32999b10fd6444295fcd1fb290631ed25361249bf1f184cf727f515baa8e4548.jpg)  
Figure 6: Expression-generation prompt for CounterFact and atomic MQuAKE edits. [SUBJ] marks the subject; angle-bracketed fields are filled from the edit request.

## A.2.2 Target Computation Objective

We compute target memory representations by optimizing a temporary perturbation $\pmb { \delta } _ { i }$ using the prediction loss [12]. The perturbation is shared across the expression set $X _ { i }$ (see Section 3.1). For a desired object with tokens $o _ { i } ^ { \star } = ( o _ { i , 1 } ^ { \star } , \ldots , o _ { i , L _ { i } } ^ { \star } )$ , the NLL term in Eq. 4 is

$$
\mathcal { L } _ { \mathrm { N L L } } \big ( \mathcal { M } _ { \delta _ { i } } ( x ) , o _ { i } ^ { \star } \big ) = - \frac { 1 } { L _ { i } } \sum _ { r = 1 } ^ { L _ { i } } \log p _ { \mathcal { M } _ { \delta _ { i } } } \big ( o _ { i , r } ^ { \star } \mid x , o _ { i , < r } ^ { \star } \big ) .\tag{20}
$$

Here, $o _ { i , < r } ^ { \star }$ denotes the preceding tokens of the desired object; the loss averages over its $L _ { i }$ tokens. The target regularizer combines a preservation loss with a norm penalty [9]:

$$
\mathcal { R } _ { \mathrm { t a r g e t } } ( \pmb { \delta } _ { i } ) = \lambda _ { \mathrm { K L } } D _ { \mathrm { K L } } \big ( p _ { \pmb { \delta } _ { i } } ^ { \mathrm { p r e } } \Vert p _ { 0 } ^ { \mathrm { p r e } } \big ) + \lambda _ { \mathrm { n o r m } } \frac { \Vert \pmb { \delta } _ { i } \Vert _ { 2 } } { \Vert \pmb { \mathbf { h } } _ { i } ( x _ { i } ) \Vert _ { 2 } ^ { 2 } } ,\tag{21}
$$

where $p _ { 0 } ^ { \mathrm { p r e } }$ and $p _ { \delta _ { i } } ^ { \mathrm { p r e } }$ are the predictive distributions before and during target computation on the preservation prompt ${ } ^ { * } s _ { i }$ is $\tt { a } ^ { \prime \prime }$ . The KL term limits prediction changes when describing the subject. The norm penalty discourages large perturbations, using the original edit prompt’s memory representation $\mathbf { h } _ { i } ( x _ { i } )$ for normalization. The clamp factor $\rho$ further bounds the perturbation norm through $\| \pmb { \delta } _ { i } \| _ { 2 } \leq \rho \| \mathbf { h } _ { i } ( x _ { i } ) \| _ { 2 }$

Adding the optimized perturbation $\pmb { \delta } _ { i } ^ { \star }$ to each current memory representation $\mathbf { h } _ { i } ( x )$ gives its target $\mathbf { h } _ { i } ^ { \star } ( x )$ . The perturbation is then removed; only the �-gram embedding updates derived below are written to conditional memory. Appendix B.4.2 gives the prefix contexts and optimization settings used in our experiments.

## A.3 Joint Memory Update Allocation

To realize the target memory representations, we express how embedding updates change the memory computation. Sum aggregation gives a linear matching relation, while context-aware

gating requires accounting for changes in the gate. We derive these two cases and then specify the reuse-based regularization weights used in Section 3.3.

## A.3.1 Sum Aggregation

We first derive how embedding updates combine to match each target under sum aggregation. LongCat-Flash-Lite uses additive fusion by averaging projected �-gram embeddings with the token embedding [6]. Here, we use the simplified sum formulation from Section 3.3; Appendix B.4.3 describes how updates are stored and applied in our implementation. For row $p$ of the mapping matrix $\mathbf { A } ,$ the current memory representation is

$$
\mathbf { h } _ { p } = \mathbf { c } _ { p } + \sum _ { j = 1 } ^ { Q } A _ { p j } \mathbf { e } ( g _ { j } ) ,\tag{22}
$$

where $\mathbf { c } _ { p }$ contains the components that are not updated. Since these components remain fixed, applying the embedding updates $\{ \mathbf { u } _ { j } \} _ { j = 1 } ^ { Q }$ changes only the summed embeddings:

$$
\widetilde { \mathbf { h } } _ { p } ( \mathbf { U } ) = \mathbf { h } _ { p } + \sum _ { j = 1 } ^ { Q } A _ { p j } \mathbf { u } _ { j } .\tag{23}
$$

The fixed components cancel when we subtract the current representation from its target. Matching the target memory representation $\mathbf { h } _ { p } ^ { \star }$ therefore requires

$$
\sum _ { j = 1 } ^ { Q } A _ { p j } \mathbf { u } _ { j } \approx \mathbf { h } _ { p } ^ { \star } - { \bf h } _ { p } = { \bf b } _ { p } .\tag{24}
$$

Stacking these relations over the edit batch gives $\mathbf { A U } \approx \mathbf { B } ,$ , so one embedding update is jointly constrained by all expressions that use it.

## A.3.2 Context-Aware Gated Aggregation

To extend target matching to context-aware gating, we account for how embedding updates afect both the projected values and their gates. Engram [3], Qwen3.8-Flash-Next [5], and DeepSeek-V4.1-Flash [4] use contextual gating to control the contribution of �-gram embeddings. We illustrate the matching relation using Engram’s scalar gate.

For expression $x _ { p } ,$ let $\mathbf { z } _ { p } \in \mathbb { R } ^ { d _ { z } }$ concatenate the activated embeddings and $\mathbf { q } _ { p } \in \mathbb { R } ^ { d }$ denote the contextual query supplied by the backbone. Here, $d _ { z }$ is the dimension of the concatenated embedding vector. We first consider updates that leave the upstream query computation unchanged, so ${ \bf q } _ { p }$ remains fixed. The memory function $\mathbf { F } _ { p }$ and gate $\alpha _ { p }$ are

$$
\mathbf { h } _ { p } = \mathbf { F } _ { p } ( \mathbf { z } _ { p } ; \mathbf { q } _ { p } ) , \qquad \alpha _ { p } = \sigma \left( \frac { \mathrm { R M S N o r m } ( \mathbf { q } _ { p } ) ^ { \top } \mathrm { R M S N o r m } ( \mathbf { W } _ { K } \mathbf { z } _ { p } ) } { \sqrt { d } } \right) ,\tag{25}
$$

where $\sigma$ is the sigmoid function and $\mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { d \times d _ { z } }$ are fixed key and value projection matrices. The memory function $\mathbf { F } _ { p }$ includes the gated value $\alpha _ { p } \mathbf { W } _ { V } \mathbf { Z } _ { p }$ and any subsequent architecture-specific processing, whose parameters also remain fixed.

To apply the same update wherever an embedding is shared, we use the batch-level embedding update matrix U. Its vectorization vec $\mathbf { \tau } ( \mathbf { U } ) \in \mathbb { R } ^ { Q d }$ stacks its columns. The selection operator $\bar { \mathbf { S } _ { p } } \in \mathbb { R } ^ { d _ { z } \times Q d }$ places each update wherever its embedding occurs in $\mathbf { z } _ { p } ,$ giving

$$
\widetilde { \mathbf { h } } _ { p } ( \mathbf { U } ) = \mathbf { F } _ { p } \left( \mathbf { z } _ { p } + \pmb { S } _ { p } \mathrm { v e c } ( \mathbf { U } ) ; \mathbf { q } _ { p } \right) .\tag{26}
$$

Local processing such as a causal convolution can make the memory representation depend on earlier token positions. In that case, the mapping matrix A and selection operator $\boldsymbol { \mathsf { S } } _ { p }$ include the activated �-grams throughout its receptive field. To match targets after the gated computation, we replace the additive matching term in Eq. 9 with

$$
\sum _ { p = 1 } ^ { P } \left\| \widetilde { \mathbf { h } } _ { p } ( \mathbf { U } ) - \mathbf { h } _ { p } ^ { \star } \right\| _ { 2 } ^ { 2 } ,\tag{27}
$$

while retaining the same reuse-based regularization. The resulting objective can be optimized with gradients.

To use a linear solver, we can instead approximate the gated computation locally at the current embeddings. Let $\mathbf { J } _ { p } \in \mathbb { R } ^ { d \times d _ { z } }$ � be the Jacobian of $\mathbf { F } _ { p }$ with respect to $\mathbf { z } _ { p _ { \perp } }$ , with the query $\mathbf { q } _ { p }$ fixed. A first-order approximation gives

$$
\widetilde { \mathbf { h } } _ { p } ( \mathbf { U } ) - \mathbf { h } _ { p } \approx \mathbf { J } _ { p } \mathbf { S } _ { p } \operatorname { v e c } ( \mathbf { U } ) .\tag{28}
$$

Stacking these relations gives a regularized linear system that approximates the gated objective, using the same matching targets ${ \bf b } _ { p } = { \bf h } _ { p } ^ { \star } - { \bf h } _ { p }$ and regularization coeficients $\lambda _ { j }$ as the additive formulation. If embedding updates also afect the upstream computation, the contextual query $\mathbf { q } _ { p }$ is recomputed during optimization, and the Jacobian includes its dependence on the updates. Thus, changing the aggregation modifies the matching relation and solver while retaining target computation, memory mapping, and reuse-based regularization.

## A.3.3 Regularization Weight Construction

For either aggregation, we penalize updates to frequently reused embeddings more strongly to limit unintended changes on unrelated inputs. Shorter �-grams can be shared by more inputs, while corpus frequency measures how often each �-gram occurs. We therefore use length and frequency as indicators of reuse. For the �-gram $g _ { j } ,$ , let $n _ { j }$ denote its length and $\pi _ { j } \in [ 0 , 1 ]$ its within-length corpus-frequency percentile. The regularization weight $w _ { j }$ combines these two indicators:

$$
\begin{array} { r l } & { ~ w _ { j } ^ { \mathrm { l e n } } = \eta _ { n _ { j } } , } \\ & { ~ w _ { j } ^ { \mathrm { f r e q } } = \mathrm { m i n } \Big ( 1 + \gamma \pi _ { j } ^ { \beta } , w _ { \mathrm { f r e q } } ^ { \mathrm { m a x } } \Big ) , } \\ & { ~ w _ { j } = \mathrm { m i n } \Big ( w _ { j } ^ { \mathrm { l e n } } + w _ { j } ^ { \mathrm { f r e q } } - 1 , w ^ { \mathrm { m a x } } \Big ) , } \end{array}\tag{29}
$$

where $\eta _ { n } > 0$ decreases with � to assign stronger penalties to shorter �-grams. The coeficient $\gamma \geq 0$ and exponent $\beta > 0$ control the strength and shape of frequency weighting. Subtracting 1 makes frequency weighting an increment over the length weight. The caps $w _ { \mathrm { f r e q } } ^ { \mathrm { m a x } } \geq 1$ and $w ^ { \mathrm { m a x } } > 0$ prevent either the frequency weight or the combined weight from becoming excessively large.

To control the overall penalty strength, we set the regularization coeficient $\lambda _ { j } = \lambda _ { \mathrm { r e u s e } } w _ { j } +$ $\lambda _ { \mathrm { { r i d g e } } }$ . The coeficient $\lambda _ { \mathrm { { r e u s e } } }$ scales the reuse-based weights; the optional $\lambda _ { \mathrm { { r i d g e } } }$ adds a uniform penalty independently of length and frequency. Setting $\lambda _ { \mathrm { { r i d g e } } } = 0$ removes this uniform term. Both coeficients are nonnegative and chosen so that every $\lambda _ { j } ~ > ~ 0$ . Under sum aggregation, Appendix A.5.2 relates length and frequency to an upper bound on memory representation changes for unrelated inputs, providing a theoretical basis for these weights.

## A.4 Complete Editing Procedure

Algorithm 1 summarizes EngramEdit for one edit batch under sum aggregation. Target computation precedes memory mapping, and the selected �-gram embeddings are updated only after solving the batch-level joint problem. For context-aware gated aggregation, the matching objective and solver are replaced as described in Appendix A.3.2.

Algorithm 1 EngramEdit for One Edit Batch   
Require: Current model M and edit batch $\mathcal { E } = \{ e _ { i } \} _ { i = 1 } ^ { m }$   
Require: Number of generated expressions � and configured �-gram lengths N   
Require: Regularization coeficients $\lambda _ { \mathrm { r e u s e } } , \lambda _ { \mathrm { r i d g e } } \geq 0$ and clamp factor $\rho > 0$   
Require: $\lambda _ { j } = \lambda _ { \mathrm { { r e u s e } } } w _ { j } + \lambda _ { \mathrm { { r i d g e } } } > 0$ for each selected embedding   
Ensure: Edited model $M ^ { \prime }$ with an updated conditional memory   
1: for all $e _ { i } = ( s _ { i } , r _ { i } , o _ { i }  o _ { i } ^ { \star } ) \in \mathcal { E }$ do   
2: $x _ { i } \gets$ original edit prompt for $( s _ { i } , r _ { i } )$ ⊲ Expression Generation   
3: P<sub>�</sub> ← GenerateExpressions $( M , e _ { i } , K )$   
4: $X _ { i } \gets \{ x _ { i } \} \cup \mathcal { P } _ { i }$   
5: for all $x \in \mathcal { X } _ { i }$ do ⊲ Joint Target Computation   
6: $q _ { i } ( x )$ ← last subject-token position in �   
7: $\mathbf { h } _ { i } ( x ) \gets \mathbf { h } ( x , q _ { i } ( x ) )$   
8: end for   
9: Compute the shared perturbation $\pmb { \delta } _ { i } ^ { \star }$ using Eq. 4 with clamp factor $\rho$   
10: $\mathbf { h } _ { i } ^ { \star } ( x ) \gets \mathbf { h } _ { i } ( x ) + \pmb { \delta } _ { i } ^ { \star }$ for all $x \in \mathcal { X } _ { i }$   
11: end for   
12: $\mathcal { R }  \{ ( i , x ) : e _ { i } \in \mathcal { E } , \ x \in \chi _ { i } \}$   
13: ${ \mathcal { G } } _ { i } ( x ) \gets { \mathcal { G } } ( x , q _ { i } ( x ) )$ for all $( i , x ) \in \mathcal { R }$ ⊲ Memory Mapping   
14: Enumerate R as $\{ ( i _ { p } , x _ { p } ) \} _ { p = 1 } ^ { P }$ , where $P = | \mathcal { R } |$   
15: $\{ g _ { j } \} _ { j = 1 } ^ { Q }  \mathrm { U n i q u e } \big ( \bigcup _ { ( i , x ) \in \mathcal { R } } \mathcal { G } _ { i } ( x ) \big )$   
16: $A _ { p j } \overset { \cdot } {  } \mathbb { I } [ g _ { j } \in { \mathcal { G } } _ { i _ { p } } ( x _ { p } ) ]$ for all $p , j$ ⊲ Mapping Matrix   
17: $\mathbf { b } _ { p } \gets \pmb { \delta } _ { i _ { p } } ^ { \star }$ for all $p ; \mathbf { B } \gets [ \mathbf { b } _ { 1 } , \ldots , \mathbf { b } _ { P } ] ^ { \top }$ ⊲ Target Matching   
18: Compute $w _ { j }$ using Eq. 29 and set $\lambda _ { j }  \lambda _ { \mathrm { r e u s e } } w _ { j } + \lambda _ { \mathrm { r i d g e } }$   
19: $\pmb { \Lambda }  \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { Q } )$ ⊲ Reuse-Based Regularization   
20: $\mathbf { U } ^ { \star } \gets \mathrm { S o l v e } ( \mathbf { A } ^ { \top } \mathbf { A } + \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { A } ^ { \top } \mathbf { B } )$ ⊲ Joint Update Solution   
21: for all selected �-grams $g _ { j }$ do ⊲ Memory Updating   
22: $\mathbf { e } ( g _ { j } )  \mathbf { e } ( g _ { j } ) + \mathbf { u } _ { j } ^ { \star }$   
23: end for   
24: return $M ^ { \prime }$

Sequential Editing. Sequential editing repeats this procedure on the current conditional memory. At step �, both the target memory representations and the matrices $\mathbf { B } _ { t }$ and $\pmb { \Lambda } _ { t }$ are computed from the model after steps $1 , \ldots , t - 1$ . The solved update is then added to the current �-gram embedding. This procedure accounts for previous changes whenever a later edit activates an �-gram that has already been updated.

Computational Cost. Let � be the maximum number of configured �-grams activated by one expression. For an edit batch with � expressions, � unique �-grams, and embedding dimension $d ,$ forming the sparse normal equations costs $O ( P s ^ { 2 } + P s d )$ . Solving the resulting dense $Q \times Q$ system costs $O ( Q ^ { 3 } + Q ^ { 2 } d )$ and uses $O ( Q ^ { 2 } + Q d )$ memory, excluding model storage. If no selected �-gram embedding is shared between groups of expressions, the system can be split into smaller systems. Solving them separately preserves the joint solution and can reduce solver time and memory (see Appendix A.5.6 for the decomposition proof).

## A.5 Proofs for Joint Memory Update Allocation

We analyze the joint update in Section 3.3 under sum aggregation (see Appendix A.3.1). The results explain the need for approximate matching, support reuse-based regularization, and characterize the solution’s update cost, sensitivity, and computation.

We use the mapping matrix $\mathbf { A } \in \mathbb { R } ^ { P \times Q }$ , matching target matrix $\pmb { \mathrm { B } } \in \mathbb { R } ^ { P \times d }$ , and embedding update matrix $\mathbf { U } \in \mathbb { R } ^ { Q \times d }$ . The �th row of U is u<sup>⊤</sup>. For the regularized objective, $\pmb { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { Q } )$ with every $\lambda _ { j } > 0$ . We write this objective as

$$
F ( \mathbf { U } ) = \lVert \mathbf { A U } - \mathbf { B } \rVert _ { F } ^ { 2 } + \lVert \mathbf { A } ^ { 1 / 2 } \mathbf { U } \rVert _ { F } ^ { 2 } ,\tag{30}
$$

and let $\mathbf { U } ^ { \star }$ denote a minimizer. The notation $\| \cdot \| _ { F }$ denotes the Frobenius norm; $\| \cdot \| _ { 2 }$ denotes the Euclidean norm for vectors and the spectral norm for matrices.

## A.5.1 Feasibility of Exact Matching

Shared embeddings can make matching targets incompatible. For example, expressions that activate the same selected �-grams receive the same memory representation change. If their matching targets difer, no embedding update can satisfy both exactly. The following condition identifies when exact matching is possible.

Proposition 1. Exact matching $\mathbf { A U } = \mathbf { B }$ is possible if and only if every column of the matching target matrix B lies in the column space of the mapping matrix A. Equivalently,

$$
\mathbf { z } ^ { \top } \mathbf { B } = \mathbf { 0 } ^ { \top } \quad \mathrm { f o r ~ e v e r y ~ \mathbf { z } \in \ n u l l } ( \mathbf { A } ^ { \top } ) .\tag{31}
$$

Proof. Exact matching separates into one equation per embedding dimension, $\mathbf { A U } _ { : , r } = \mathbf { B } _ { : , { t } }$ for $r = 1 , \ldots , d$ . Each equation has a solution precisely when its target column belongs to col(A). Stacking these column solutions gives an exact update matrix. The identity col $( \mathbf { A } ) = \mathrm { n u l l } ( \mathbf { A } ^ { \top } ) ^ { \perp }$ gives the equivalent condition in Eq. 31.

When exact matching is infeasible, the squared matching error in Eq. 9 allows a joint compromise across conflicting targets. The proposition concerns feasibility; positive regularization can also favor an inexact match when an exact match exists.

## A.5.2 Reuse-Based Regularization

An embedding update can also change memory representations on unrelated inputs that activate the same �-gram. We relate these changes to activation probabilities to explain why frequently reused embeddings receive stronger update penalties.

Let D be a fixed distribution over unrelated input–position pairs $( x , q )$ . For each selected �-gram $g _ { j } ,$ define its activation indicator $a _ { j } ( x , q ) \ = \ \mathbb { I } [ g _ { j } \in \mathcal { G } ( x , q ) ]$ and activation probability $p _ { j } = \mathbb { E } _ { \mathcal { D } } [ a _ { j } ( x , q ) ]$ . Let $s \leq | N |$ be the maximum number of selected �-grams activated at one position under the lookup in Section 2. For fixed embedding updates,

$$
\widetilde { \mathbf { h } } ( x , q ; \mathbf { U } ) - \mathbf { h } ( x , q ) = \sum _ { j = 1 } ^ { Q } a _ { j } ( x , q ) \mathbf { u } _ { j } .\tag{32}
$$

Proposition 2. For any fixed embedding updates $\{ { \mathbf { u } } _ { j } \} _ { j = 1 } ^ { Q } .$ , the expected squared memory representation change satisfies

$$
\mathbb { E } _ { \mathcal { D } } \left[ \| \widetilde { \mathbf { h } } ( x , q ; \mathbf { U } ) - \mathbf { h } ( x , q ) \| _ { 2 } ^ { 2 } \right] \leq s \sum _ { j = 1 } ^ { Q } p _ { j } \| \mathbf { u } _ { j } \| _ { 2 } ^ { 2 } .\tag{33}
$$

No independence assumption between �-gram activations is required.

Proof. Fix an input–position pair $( x , q )$ and let $\mathcal { T } ( x , q ) = \{ j : a _ { j } ( x , q ) = 1 \}$ index its activated selected �-grams. The triangle and Cauchy–Schwarz inequalities give

$$
\begin{array} { r l r } {  { \| \sum _ { j = 1 } ^ { Q } a _ { j } ( x , q ) \mathbf { u } _ { j } \| _ { 2 } ^ { 2 } \leq ( \sum _ { j \in \mathcal { T } ( x , q ) } \| \mathbf { u } _ { j } \| _ { 2 } ) ^ { 2 } } } \\ & { } & { \leq | \mathcal { T } ( x , q ) | \sum _ { j \in \mathcal { T } ( x , q ) } \| \mathbf { u } _ { j } \| _ { 2 } ^ { 2 } \leq s \sum _ { j = 1 } ^ { Q } a _ { j } ( x , q ) \| \mathbf { u } _ { j } \| _ { 2 } ^ { 2 } . } \end{array}\tag{34}
$$

The bound also holds when no selected �-gram is activated, since both sides are zero. Taking expectations and using $\mathbb { E } _ { \mathcal { D } } [ a _ { j } ( x , q ) ] = p _ { j }$ proves Eq. 33 by linearity of expectation.

The bound assigns a larger cost to the same embedding update when its �-gram is activated more often on unrelated inputs. This provides a theoretical basis for penalizing large updates to frequently reused embeddings.

Bounds for the Solved Update. The activation bound applies to arbitrary updates. For the updates returned by the joint objective, regularization also limits the total weighted update cost, yielding the following bounds.

Corollary 1. The minimizer $\mathbf { U } ^ { \star }$ of Eq. 30 satisfies

$$
\| \mathbf { u } _ { j } ^ { \star } \| _ { 2 } \leq \frac { \| \mathbf { B } \| _ { F } } { \sqrt { \lambda _ { j } } } \quad \mathrm { f o r ~ e v e r y } \ j ,\tag{35}
$$

and its expected squared memory representation change on unrelated inputs satisfies

$$
\mathbb { E } _ { \mathcal { D } } \left[ \lVert \widetilde { \mathbf { h } } ( x , q ; \mathbf { U } ^ { \star } ) - \mathbf { h } ( x , q ) \rVert _ { 2 } ^ { 2 } \right] \leq s \left( \operatorname* { m a x } _ { j } \frac { p _ { j } } { \lambda _ { j } } \right) \lVert \mathbf { B } \rVert _ { F } ^ { 2 } .\tag{36}
$$

Proof. The zero update is feasible, so optimality and nonnegativity of the matching error give

$$
\sum _ { j = 1 } ^ { Q } \lambda _ { j } \| \mathbf { u } _ { j } ^ { \star } \| _ { 2 } ^ { 2 } \leq F ( \mathbf { U } ^ { \star } ) \leq F ( \mathbf { 0 } ) = \| \mathbf { B } \| _ { F } ^ { 2 } .\tag{37}
$$

Each summand is nonnegative. Bounding the �th term by $\| \mathbf B \| _ { F } ^ { 2 } .$ , dividing by $\lambda _ { j } > 0$ , and taking square roots gives Eq. 35. For the second bound, apply Proposition 2 and then use

$$
\sum _ { j = 1 } ^ { Q } p _ { j } \| \mathbf { u } _ { j } ^ { \star } \| _ { 2 } ^ { 2 } \leq \left( \operatorname* { m a x } _ { j } \frac { p _ { j } } { \lambda _ { j } } \right) \sum _ { j = 1 } ^ { Q } \lambda _ { j } \| \mathbf { u } _ { j } ^ { \star } \| _ { 2 } ^ { 2 } \leq \left( \operatorname* { m a x } _ { j } \frac { p _ { j } } { \lambda _ { j } } \right) \| \mathbf { B } \| _ { F } ^ { 2 } .\tag{38}
$$

For fixed matching targets, increasing $\lambda _ { j }$ tightens the bound on its embedding update. With the unrelated-input distribution also fixed, the bound on memory representation changes decreases when the largest ratio $p _ { j } / \lambda _ { . }$ <sub>�</sub> decreases. Thus, the bounds connect stronger penalties on frequently activated embeddings to control of memory representation changes on unrelated inputs.

Length and Frequency. The activation probabilities on unrelated inputs are not directly available, so EngramEdit uses �-gram length and corpus frequency as indicators. If the selected �-gram $g _ { k }$ is a shorter sufix of the selected �-gram $g _ { j }$ , every activation of $g _ { j }$ also activates $g _ { k }$ . Hence,

$$
a _ { j } ( x , q ) \leq a _ { k } ( x , q ) \quad \implies \quad p _ { j } \leq p _ { k } .\tag{39}
$$

This ordering holds for sufix-related �-grams, not every pair of diferent lengths.

For a corpus of $N _ { \mathsf { c o r p u s } }$ input–position pairs, let $c _ { j }$ count the positions activating $g _ { j }$ . Its activation probability under the uniform empirical distribution is $\widehat { p } _ { j } = c _ { j } / N _ { \mathrm { c o r p u s } }$ . Counts and empirical activation probabilities therefore have the same ordering. The within-length frequency ranks in Appendix A.3.3 retain this ordering within each length, but do not estimate $p _ { j }$ on unrelated inputs. These relations motivate the length and frequency factors without establishing optimality of the particular weight formula. The bounds concern memory representations, not downstream predictions.

## A.5.3 Closed-Form Solution and Uniqueness

Overlapping activations can make columns of the mapping matrix linearly dependent. Positive regularization ensures a unique joint update without requiring this matrix to have full column rank.

Proposition 3. If every regularization coeficient satisfies $\lambda _ { j } > 0$ , the objective in Eq. 9 has the unique solution in Eq. 10.

Proof. Diferentiating Eq. 30 gives $\nabla _ { \mathbf { U } } F = 2 \mathbf { A } ^ { \top } ( \mathbf { A U } - \mathbf { B } ) + 2 \mathbf { A U }$ . Setting the gradient to zero yields

$$
( \mathbf { A } ^ { \top } \mathbf { A } + \mathbf { A } ) \mathbf { U } = \mathbf { A } ^ { \top } \mathbf { B } .\tag{40}
$$

For any nonzero vector $\mathbf { z } \in \mathbb { R } ^ { Q }$ ，

$$
\mathbf { z } ^ { \top } ( \mathbf { A } ^ { \top } \mathbf { A } + \mathbf { A } ) \mathbf { z } = \| \mathbf { A } \mathbf { z } \| _ { 2 } ^ { 2 } + \sum _ { j = 1 } ^ { Q } \lambda _ { j } z _ { j } ^ { 2 } > 0 .\tag{41}
$$

The coeficient matrix is therefore positive definite, making the objective strictly convex. Its unique stationary point is the global minimizer,

$$
\mathbf { U } ^ { \star } = ( \mathbf { A } ^ { \top } \mathbf { A } + \mathbf { A } ) ^ { - 1 } \mathbf { A } ^ { \top } \mathbf { B } .\tag{42}
$$

## A.5.4 Minimum-Cost Update Allocation

Diferent embedding updates can produce the same memory representation changes on the edit expressions. Extra update components that leave these changes unchanged do not improve target matching, but can afect unrelated inputs that activate diferent subsets of embeddings. We show that the joint solution avoids such extra components by minimizing the weighted update cost for its achieved memory representation changes.

Proposition 4. Let $\mathbf { U } ^ { \star }$ minimize Eq. 30 with every $\lambda _ { j } > 0$ . Any alternative embedding update matrix $\mathbf { V } \in \mathbb { R } ^ { Q \times d }$ satisfying $\mathbf { A } \mathbf { V } = \mathbf { A } \mathbf { U } ^ { \star }$ obeys

$$
\lVert { \boldsymbol { \Lambda } } ^ { 1 / 2 } { \bf { V } } \rVert _ { F } ^ { 2 } = \lVert { \boldsymbol { \Lambda } } ^ { 1 / 2 } { \bf { U } } ^ { \star } \rVert _ { F } ^ { 2 } + \lVert { \boldsymbol { \Lambda } } ^ { 1 / 2 } ( { \bf { V } } - { \bf { U } } ^ { \star } ) \rVert _ { F } ^ { 2 } .\tag{43}
$$

Thus, $\mathbf { U } ^ { \star }$ uniquely minimizes the weighted update cost among all updates producing the same achieved memory representation changes $\mathbf { A U } ^ { \star }$

Proof. Let $\mathbf { D } = \mathbf { V } - \mathbf { U } ^ { \star }$ . The shared matching result gives $\mathbf { A D } = \mathbf { 0 } $ , and the normal equation gives $\boldsymbol { \Lambda } \mathbf { U } ^ { \star } = \mathbf { A } ^ { \top } ( \mathbf { B } - \mathbf { A } \mathbf { U } ^ { \star } )$ . Together, these identities imply

$$
\operatorname { t r } ( \mathbf { D } ^ { \top } \mathbf { A } \mathbf { U } ^ { \star } ) = \operatorname { t r } \big ( ( \mathbf { A } \mathbf { D } ) ^ { \top } ( \mathbf { B } - \mathbf { A } \mathbf { U } ^ { \star } ) \big ) = 0 ,\tag{44}
$$

where tr denotes the matrix trace. Expanding the weighted cost of $\mathbf { V } = \mathbf { U } ^ { \star } + \mathbf { D }$ then gives

$$
\begin{array} { r l } & { \| \boldsymbol { \Lambda } ^ { 1 / 2 } \boldsymbol { \mathbf { V } } \| _ { F } ^ { 2 } = \| \boldsymbol { \Lambda } ^ { 1 / 2 } \boldsymbol { \mathbf { U } } ^ { \star } \| _ { F } ^ { 2 } + 2 \operatorname { t r } ( \boldsymbol { \mathbf { D } } ^ { \top } \boldsymbol { \Lambda } \boldsymbol { \mathbf { U } } ^ { \star } ) + \| \boldsymbol { \Lambda } ^ { 1 / 2 } \boldsymbol { \mathbf { D } } \| _ { F } ^ { 2 } } \\ & { \qquad = \| \boldsymbol { \Lambda } ^ { 1 / 2 } \boldsymbol { \mathbf { U } } ^ { \star } \| _ { F } ^ { 2 } + \| \boldsymbol { \Lambda } ^ { 1 / 2 } \boldsymbol { \mathbf { D } } \| _ { F } ^ { 2 } . } \end{array}\tag{45}
$$

Since every $\lambda _ { j } > 0$ , the additional cost is strictly positive unless $\mathbf { D } = \mathbf { 0 }$ , proving minimality and uniqueness under the matching constraint.

The comparison fixes the achieved changes $\mathbf { A U } ^ { \star }$ , so it remains valid when exact matching to B is infeasible. Any additional update with columns in null(A) increases the weighted cost without improving the matched representations. This is a guarantee on weighted update cost, not on downstream prediction changes.

## A.5.5 Sensitivity to Target Changes

Target memory representations are computed through optimization and may vary slightly. We therefore examine how these variations afect the solved embedding updates when the mapping and regularization remain fixed. The following bound shows how positive regularization controls this sensitivity.

Proposition 5. For fixed mapping matrix A and positive definite regularization matrix �, let ΔB be a change in the matching target matrix. The corresponding solution change satisfies

$$
\Delta \mathbf { U } ^ { \star } = ( \mathbf { A } ^ { \top } \mathbf { A } + \mathbf { A } ) ^ { - 1 } \mathbf { A } ^ { \top } \Delta \mathbf { B } ,
$$

$$
\lVert \Delta \mathbf { U } ^ { \star } \rVert _ { F } \leq \frac { \lVert \mathbf { A } \rVert _ { 2 } } { \lambda _ { \operatorname* { m i n } } ( \pmb { \Lambda } ) } \lVert \Delta \mathbf { B } \rVert _ { F } .\tag{46}
$$

Proof. Write $\mathbf { H } = \mathbf { A } ^ { \top } \mathbf { A } + \mathbf { A }$ and let $\mathbf { U } ^ { \star } ( \mathbf { B } )$ denote the solution for matching targets B. Subtracting the normal equations for $\mathbf { B } + \Delta \mathbf { B }$ and B gives

$$
\mathbf { H } \Delta \mathbf { U } ^ { \star } = \mathbf { A } ^ { \top } \Delta \mathbf { B } , \qquad \Delta \mathbf { U } ^ { \star } = \mathbf { U } ^ { \star } ( \mathbf { B } + \Delta \mathbf { B } ) - \mathbf { U } ^ { \star } ( \mathbf { B } ) .\tag{47}
$$

Multiplying by ${ \bf H } ^ { - 1 }$ proves the exact relation. For every unit vector $\mathbf { z } , \mathbf { z } ^ { \top } \mathbf { H } \mathbf { z } \geq \mathbf { z } ^ { \top } \pmb { \Lambda } \mathbf { z } \geq \lambda _ { \operatorname* { m i n } } ( \pmb { \Lambda } )$ Thus, $\lVert \mathbf { H } ^ { - 1 } \rVert _ { 2 } \leq 1 / \lambda _ { \operatorname* { m i n } } ( \pmb { \Lambda } )$ , and

$$
\| \boldsymbol { \Delta } \mathbf { U } ^ { \star } \| _ { F } \leq \| \mathbf { H } ^ { - 1 } \| _ { 2 } \| \boldsymbol { \mathbf { A } } ^ { \top } \| _ { 2 } \| \boldsymbol { \Delta } \mathbf { B } \| _ { F } \leq \frac { \| \boldsymbol { \mathbf { A } } \| _ { 2 } } { \lambda _ { \operatorname* { m i n } } ( \boldsymbol { \Lambda } ) } \| \boldsymbol { \Delta } \mathbf { B } \| _ { F } .\tag{48}
$$

The final step uses $\lVert \mathbf { A } ^ { \top } \rVert _ { 2 } = \lVert \mathbf { A } \rVert _ { 2 }$

For a fixed mapping, the smallest regularization coeficient controls the sensitivity bound even when the mapping matrix is rank deficient. The bound quantifies possible amplification of target changes; it does not require this amplification to be below one.

## A.5.6 Component Decomposition

Joint solving is needed only among expressions connected through shared embeddings. When groups share no selected �-gram embeddings, solving them separately can reduce computation without changing the joint solution.

Proposition 6. The joint update decomposes exactly over disconnected components of the bipartite graph between expression rows and selected �-gram columns, with an edge wherever $A _ { p j } \neq 0$

Proof. Ordering rows and columns by connected component puts the mapping matrix into block-diagonal form,

$$
\mathbf { A } = \mathrm { d i a g } ( \mathbf { A } _ { 1 } , \ldots , \mathbf { A } _ { C } ) .\tag{49}
$$

For component $c ,$ let $\mathbf { B } _ { c }$ contain its expression targets, and let $\mathbf { U } _ { c }$ and $\pmb { \Lambda } _ { c }$ contain the updates and regularization coeficients for its embeddings. Since � is diagonal, the objective separates as

$$
F ( \mathbf { U } ) = \sum _ { c = 1 } ^ { C } \left( \| \mathbf { A } _ { c } \mathbf { U } _ { c } - \mathbf { B } _ { c } \| _ { F } ^ { 2 } + \| \mathbf { A } _ { c } ^ { 1 / 2 } \mathbf { U } _ { c } \| _ { F } ^ { 2 } \right) .\tag{50}
$$

Each summand depends only on its own embedding updates. Applying Proposition 3 to each component gives

$$
\mathbf { U } _ { c } ^ { \star } = ( \mathbf { A } _ { c } ^ { \top } \mathbf { A } _ { c } + \mathbf { A } _ { c } ) ^ { - 1 } \mathbf { A } _ { c } ^ { \top } \mathbf { B } _ { c } .\tag{51}
$$

Stacking these solutions and restoring the original embedding order recovers the joint solution. An isolated expression contributes only a constant matching loss; an isolated embedding receives the zero update. Neither afects other components.

## B Experimental Setup

This section describes the datasets, evaluation, baselines, and implementation details for the experiments in Section 4.

## B.1 Datasets

The editing benchmarks test factual recall and the use of revised facts in reasoning. A separate set of tasks measures general capabilities after editing.

CounterFact. CounterFact [9] tests counterfactual changes to factual associations. We use the MultiCounterFact release provided with MEMIT for batched editing [12]. Each case contains an edit request $e _ { i }$ and its prompt (see Section 2), held-out paraphrases, neighborhood prompts about other subjects whose facts should remain unchanged, and generation prompts. Held-out paraphrases are used only for evaluation; generated expressions are used during editing. Table 3 illustrates an edit request and its evaluation prompts.

ZsRE. ZsRE [10] represents factual associations as questions and answers. We use the editing evaluation split distributed with MEMIT [12], taking the first reference answer as the editing target. Each case pairs the edit question with a held-out paraphrase and an unrelated question– answer pair from Natural Questions. The edit therefore requests the supplied factual answer, not a counterfactual replacement. An example question, its target answer, and the evaluation questions are shown in Table 3.

Table 3: Editing examples from CounterFact, ZsRE, and MQuAKE, with simplified formatting. The ellipsis omits an unrelated prefix from the CounterFact paraphrase.
<table><tr><td>Field</td><td>Example</td></tr><tr><td colspan="2">CounterFact</td></tr><tr><td>Subject and relation</td><td>Danielle Darrieux; mother tongue</td></tr><tr><td>Requested change</td><td>French → English</td></tr><tr><td>Edit prompt</td><td>The mother tongue of Danielle Darrieux is</td></tr><tr><td>Held-out paraphrase</td><td>... Danielle Darrieux spoke the language</td></tr><tr><td>Unrelated prompt</td><td>The native language of Montesquieu is</td></tr><tr><td>Unchanged answer</td><td>French</td></tr><tr><td colspan="2">ZsRE</td></tr><tr><td>Subject</td><td>Watts Humphrey</td></tr><tr><td>Edit question</td><td>What university did Watts Humphrey attend?</td></tr><tr><td>Desired answer</td><td>Illinois Institute of Technology</td></tr><tr><td>Held-out paraphrase</td><td>What university did Watts Humphrey take part in?</td></tr><tr><td>Unrelated question</td><td>Who played Desmond Doss&#x27;s father in Hacksaw Ridge?</td></tr><tr><td>Unchanged answer</td><td>Hugo Weaving</td></tr><tr><td colspan="2">MQuAKE</td></tr><tr><td>Edit request 1</td><td>Fernando Santos; country of citizenship: Portugal → United Kingdom</td></tr><tr><td>Edit request 2</td><td>United Kingdom; head of state: Elizabeth II → Emmerson Mnangagwa</td></tr><tr><td>Multi-hop question</td><td>Who is the head of state of the country where Fernando Santos hold a</td></tr><tr><td>Original answer</td><td>citizenship? Marcelo Rebelo de Sousa</td></tr><tr><td>Desired answer</td><td>Emmerson Mnangagwa</td></tr></table>

MQuAKE. MQuAKE [11] tests whether revised facts can be combined to answer multi-hop questions. We use 3,000 counterfactual cases covering two- to four-hop reasoning. Each case contains one or more fact-level edit requests, variants of a multi-hop question, and the answer implied by the revised facts, with accepted aliases. Table 3 shows how multiple fact edits change the answer to one multi-hop question.

General-Capability Tasks. Following AlphaEdit [13], we evaluate SST (SST-2) [31] for sentiment classification, MRPC [32] for paraphrase recognition, and CoLA [33] for linguistic acceptability. RTE [34] and NLI [35] test textual entailment, and MMLU [36] tests knowledge across multiple subjects. We use 100 examples per task; none are included in the edit requests.

## B.2 Evaluation

On CounterFact and ZsRE, Eficacy measures success on edit prompts, Generalization on held-out paraphrases, and Specificity on unrelated queries. Utility is the arithmetic mean of these three metrics. Evaluation intervals are given in Appendix B.4.1.

CounterFact. We compare the new and original answers using length-normalized log-likelihood. For prompt �, answer tokens $y = ( y _ { 1 } , \dotsc , y _ { | y | } )$ , and edited model M<sup>′</sup>, the answer score is

$$
\operatorname { s c o r e } ( x , y ) = { \frac { 1 } { | y | } } \sum _ { t = 1 } ^ { | y | } \log p _ { \mathcal { M } ^ { \prime } } ( y _ { t } \mid x , y _ { < t } ) ,\tag{52}
$$

where $y _ { < t }$ contains the preceding reference-answer tokens. For edit case �, the edit prompt $x _ { i }$ and held-out paraphrases succeed when the new answer $o _ { i } ^ { \star }$ scores higher than the original answer $o _ { i } .$

Neighborhood prompts should retain the opposite preference. With $\mathbb { I } [ \cdot ]$ denoting the indicator function, the prompt-level scores are

$$
\begin{array} { r l } & { s _ { i } ^ { \mathrm { e d i t } } ( \boldsymbol { x } ) = \mathbb { I } [ \mathrm { s c o r e } ( \boldsymbol { x } , \boldsymbol { o } _ { i } ^ { \star } ) > \mathrm { s c o r e } ( \boldsymbol { x } , \boldsymbol { o } _ { i } ) ] , } \\ & { s _ { i } ^ { \mathrm { n b r } } ( \boldsymbol { x } ) = \mathbb { I } [ \mathrm { s c o r e } ( \boldsymbol { x } , \boldsymbol { o } _ { i } ) > \mathrm { s c o r e } ( \boldsymbol { x } , \boldsymbol { o } _ { i } ^ { \star } ) ] . } \end{array}\tag{53}
$$

Eficacy averages $s _ { i } ^ { \mathrm { e d i t } } ( x _ { i } )$ over edit cases. Generalization first averages $s _ { i } ^ { \mathrm { e d i t } } ( x )$ over each case’s held-out paraphrases, then across cases. Specificity applies the same two-stage average to $s _ { i } ^ { \mathrm { n b r } } ( x )$ on neighborhood prompts. Scores are multiplied by 100, with each case weighted equally.

For generated continuations, Fluency averages ${ \frac { 1 } { 3 } } H _ { 2 } + { \frac { 2 } { 3 } } H _ { 3 }$ , where $H _ { 2 }$ and $H _ { 3 }$ are empirical word bigram and trigram entropies in bits. Consistency measures cosine similarity between TF-IDF vectors of the concatenated generated text and target-related reference text. We average scores across cases and multiply both metrics by 100 for reporting. Fluency is a scaled entropy, not a percentage accuracy.

ZsRE. Given the preceding reference-answer tokens, we check whether the model’s most likely next token matches the next reference token. For prompt � and reference answer $y ,$ , this gives

$$
\mathrm { T o k e n A c c } ( x , y ) = \frac { 1 } { | y | } \sum _ { t = 1 } ^ { | y | } \mathbb { I } \left[ \underset { \nu } { \arg \operatorname* { m a x } } p _ { \mathcal { M } ^ { \prime } } ( \nu \mid x , y _ { < t } ) = y _ { t } \right] ,\tag{54}
$$

where $\nu$ ranges over the token vocabulary. Eficacy evaluates the edit question and its target answer; Generalization uses the held-out paraphrase and the same answer. Specificity evaluates the unrelated question and its reference answer. Token accuracies are averaged within each case, then across cases, and multiplied by 100. Thus, Specificity measures post-edit answer accuracy, not agreement with pre-edit predictions. Newly correct and newly incorrect answers can ofset each other; Appendix C.1.3 separates these transitions.

MQuAKE. Standard prompting asks for a direct answer, while CoT prompting asks for reasoning before the final answer. Both use the benchmark’s demonstrations and greedy decoding, with maximum generation lengths of 32 and 128 tokens, respectively. We extract the first nonempty answer line for standard prompting, removing an optional answer label, and the final Answer: line for CoT. Predictions and reference answers are normalized for capitalization, whitespace, and boundary punctuation. A prediction is correct if it exactly matches the revised answer or an accepted alias. A case is correct if at least one of its question variants is answered correctly. We report the percentage of correct cases.

General Capabilities. For each task, we select the candidate answer with the highest score in Eq. 52 and compute class-frequency-weighted F1. For task $d ,$ let $C _ { d }$ be its label set, $n _ { d , c }$ the number of examples with label $c ,$ and $n _ { d }$ the total number of examples. The task score and the mean across six tasks are

$$
F _ { 1 , d } = \sum _ { c \in C _ { d } } \frac { n _ { d , c } } { n _ { d } } \frac { 2 \mathrm { T P } _ { d , c } } { 2 \mathrm { T P } _ { d , c } + \mathrm { F P } _ { d , c } + \mathrm { F N } _ { d , c } } , \qquad \overline { { F } } _ { 1 } = \frac { 1 } { 6 } \sum _ { d = 1 } ^ { 6 } F _ { 1 , d } .\tag{55}
$$

Here, $\mathrm { T P } _ { d , c } , \mathrm { F P } _ { d , c } ,$ and $\mathrm { F N } _ { d , c }$ count true positives, false positives, and false negatives for class $c ;$ a zero denominator contributes zero. Classes are weighted by frequency within each task, and tasks are weighted equally. We measure changes relative to each method’s own pre-edit performance. These F1 scores follow our evaluation protocol, not each benchmark’s default leaderboard metric.

Confidence Intervals. Across the editing benchmarks, reported intervals quantify uncertainty across cases, not across random seeds. Following ROME [9], we use normal-approximation 95% confidence intervals $\bar { x } \pm 1 . 9 6 s / \sqrt { N }$ , where �¯ and � are the mean and standard deviation of case-level scores. Scores from multiple prompts within a case are averaged first. The sample size � is the number of cases evaluated at each editing stage; the generated-expression analysis in Figure 5(b) uses $N = 2 0 0 0$ . MQuAKE instead uses Wilson 95% confidence intervals over cases. Utility is reported as a point estimate.

## B.3 Baselines

We compare gradient-based expert editing, MoE-specific locate-then-edit, and direct conditional memory fine-tuning. AlphaEdit’s original update rule applies to a dense FFN projection [13]. We exclude it because applying that rule to routed experts requires an additional choice of experts and update allocation [14]. The gradient-based baselines use the model’s existing routed computation, with editable parameters specified below.

FT and FT-L. FT uses gradient descent to update selected model parameters so that edit prompts produce the desired answers. FT-L additionally bounds each parameter change to limit deviation from the model entering the edit batch [15]. In our MoE adaptation, both methods update expert FFN down-projections at layer 7. They use up to 25 epochs per edit batch, a learning rate of $5 \times 1 0 ^ { - 4 }$ , and optimization minibatches of one example. FT-L bounds each parameter change by $1 0 ^ { - 4 }$

AdaLoRA. AdaLoRA [16] learns low-rank updates to weight matrices while keeping the pretrained weights fixed. It allocates more of the rank budget to important update components and prunes less important ones. We apply it to expert down-projections at layer 7, optimizing the prediction loss on edit prompts with initial rank 12, target rank 8, and scaling parameter 32. It uses 25 epochs per edit batch, a learning rate of $5 \times 1 0 ^ { - 4 }$ , and optimization minibatches of one example.

UnKE. UnKE [17] edits knowledge by first computing target Transformer-block representations that lead to the desired answers. It then updates block parameters to match these targets while retaining the outputs on preservation examples. In our MoE adaptation, we fit the layer-7 block outputs by updating the experts’ gate, up-, and down-projections; attention and router parameters remain fixed. Target computation uses up to 25 steps at learning rate 0.5, and parameter fitting uses up to 50 steps at learning rate $2 \times 1 0 ^ { - 4 }$

MoEEdit. MoEEdit [14] jointly updates expert FFN down-projections so that their routerweighted outputs match the editing targets. To preserve outputs on unrelated inputs and limit downstream routing shifts, it constrains each expert’s update using a null-space projection built from preservation inputs. The updates are solved one expert at a time through block coordinate descent. We edit layer 7, using up to 25 target-computation steps at learning rate 0.1 and four block coordinate descent passes. The per-expert null-space projections use statistics from 100,000 Wikipedia samples.

Memory-FT (Subject). This baseline directly fine-tunes conditional memory to predict the desired answers, restricting updates to embeddings activated at the last subject-token position of each original edit prompt. Since this selects a limited set of �-grams, we store one cumulative update vector per exact �-gram, as in EngramEdit. We optimize these vectors by backpropagating the prediction loss and add them whenever their �-grams are activated, including outside subject positions. The pretrained hashed tables, token embeddings, memory projection layers, and backbone remain fixed.

Memory-FT (All Tokens). This variant expands direct memory fine-tuning to embeddings activated at every token position in the original edit prompts, using the same prediction loss. Storing a separate update vector for each �-gram in this larger set would increase memory requirements. We therefore directly fine-tune the pretrained hashed tables, allowing rows activated at any token position to receive gradient updates while keeping all other parameters fixed. Both Memory-FT variants use �-gram lengths 2, 3, and 4, up to 25 epochs per edit batch, a learning rate of $3 \times 1 0 ^ { - 3 }$ , and optimization minibatches of one example.

## B.4 Implementation Details

We describe the shared model and editing setup, followed by EngramEdit’s target computation, memory mapping, and memory updating settings.

## B.4.1 Model and Editing Setup

All experiments use LongCat-Flash-Lite, a 68.5B-parameter MoE model with 31.4B parameters in �-gram embeddings [6], and run on two NVIDIA RTX PRO 6000 GPUs. Its conditional memory uses sufix �-grams of lengths 2, 3, and 4, with four hashed sub-tables per length. Each sub-table returns a 256-dimensional vector, which is linearly projected to the 3,072-dimensional token embedding space. At each position, the model averages the twelve projected vectors and the original token embedding before passing the result to the backbone. EngramEdit keeps the token embeddings and these projection matrices fixed.

Sequential Editing. An edit batch contains the requests applied together; an optimization minibatch contains the examples used in one gradient step. The main CounterFact and ZsRE experiments apply 2,000 edits in batches of 100, with each batch starting from the model state produced by preceding batches. The longer runs extend editing to 5,000 facts. For MQuAKE, we shufle the 3,000 cases once with seed 0 and use the same case IDs and hop labels for all methods. Each batch contains 100 cases, with all their fact-level edit requests applied together, so a batch can contain more than 100 fact edits.

Evaluation Schedule. CounterFact trajectories are evaluated every 100 edits, and ZsRE trajectories are compared at common intervals of 200 edits. General capabilities are evaluated before editing and every 500 edits through 5,000 sequential CounterFact edits, using 100 examples per task (see Appendix B.1).

## B.4.2 Target Computation and Memory Mapping

For each edit request, we ask the model to generate four semantically equivalent expressions before editing, using the prompt and filtering rules in Appendix A.2.1. The retained expressions and original edit prompt form the expression set used for target computation and memory mapping.

Prediction Loss. During target computation, we evaluate each expression both alone and after each of five model-generated prefix contexts, which vary the surrounding text without rephrasing the edit request. The shared perturbation is added to the conditional memory representation before the first decoder block, while all decoder parameters remain fixed. Following the target readout in MEMIT [12], we compute the NLL from the last decoder block’s hidden states using the model’s fixed final normalization and language-model head. This yields the final predictive distribution in Eq. 20; the KL preservation term also uses final-output probabilities.

Optimization Settings. We optimize the shared perturbation with Adam for at most 25 steps at learning rate 0.5, with $\lambda _ { \mathrm { K L } } = 0 . 0 6 2 5 , \lambda _ { \mathrm { n o r m } } = 0 . 0 0 1$ , and $\rho = 3 2$ . Optimization stops when the total target-computation loss falls below 0.05, checked every two steps.

Memory Mapping. We use expressions without prefix contexts for memory mapping. At each expression’s last subject-token position, we extract sufix �-grams of lengths 2, 3, and $^ { 4 , }$ skipping incomplete windows and windows crossing an end-of-sequence token.

## B.4.3 Memory Updating

Conditional memory uses hashed tables to control the storage cost of a large �-gram space. Editing only requires storing updates for the selected �-grams. We therefore maintain a separate cumulative update vector for each edited �-gram, indexed by its exact token sequence. Additional storage scales with the number of distinct edited �-grams, and the exact indexing prevents hash collisions from coupling their updates. Each vector is added to the embedding output whenever its �-gram is activated, while the pretrained tables remain fixed.

Regularization and Solver Settings. We use the weight construction in Eq. 29, setting $\lambda _ { \mathrm { r i d g e } } =$ 0.01 and $\lambda _ { \mathrm { r e u s e } } = 0 . 0 5$ . The length weights are $\eta _ { 2 } = 8 , \eta _ { 3 } = 2$ , and $\eta _ { 4 } = 1$ . For frequency-based regularization, we use $\gamma = 9 , \beta = 1 , \upsilon _ { \mathrm { f r e q } } ^ { \mathrm { m a x } } = 1 0$ , and $w ^ { \mathrm { m a x } } = 6 4$ . We estimate the corpus frequency of the selected �-grams from up to 3,000,000 documents in the November 1, 2023 English Wikipedia snapshot, using at most 4,096 tokens from each document. Within each �-gram length, the percentile $\pi _ { j }$ is the normalized rank of the observed count among the distinct count values. Missing �-grams receive a frequency weight of 1. We form and solve the joint linear system in FP32 using a direct linear solver.

## C Additional Experimental Results

This section provides detailed evaluation results and analyses of editing choices and memory use to complement Section 4.

## C.1 Detailed Evaluation Results

We extend the sequential editing results, examine multi-hop reasoning by depth, and analyze ZsRE Specificity to complement Section 4.2. We then report task-level general-capability trajectories to complement Section 4.3.

## C.1.1 Sequential Editing with 5,000 Facts

To test whether knowledge updates through conditional memory remain efective as edits accumulate, we extend the CounterFact and ZsRE experiments in Section 4.2 to 5,000 sequential edits in batches of 100. We compare EngramEdit with FT, UnKE, MoEEdit, and MFT-S. Each evaluation covers all edits applied so far. Figure 7 reports Eficacy, Generalization, and Specificity every 100 edits on CounterFact and every 200 edits on ZsRE. We observe that:

• EngramEdit maintains high Eficacy and a clear Generalization lead on both datasets through $5 { , } 0 0 0$ edits. Because each evaluation includes all earlier edits, these results show that revised knowledge remains accessible across expressions as further updates accumulate. EngramEdit supports this access by jointly updating embeddings across expressions, allowing diferent wordings to activate memory updated for the same fact.

• On CounterFact, EngramEdit and MFT-S maintain much higher Specificity than FT, UnKE, and MoEEdit throughout the sequence, but EngramEdit achieves substantially higher Generalization at similar Eficacy. EngramEdit therefore combines cross-expression access with largely preserved unrelated knowledge as edits accumulate. Reuse-based regularization can help explain this preservation because it penalizes large updates to embeddings that unrelated inputs are more likely to activate.

![](images/c385d75811d14d45df79159dd0bcc558fe96e2568c255b504bd811bd30d7d26b.jpg)  
Figure 7: Sequential editing on CounterFact (a–c) and ZsRE (d–f) up to 5,000 edits. Shading shows pointwise case-level 95% confidence intervals. MFT-S denotes Memory-FT (Subject).

## C.1.2 Multi-Hop Reasoning

To test whether knowledge updated through conditional memory supports reasoning at diferent depths, we break down the MQuAKE evaluation in Section 4.2 by hop count. All eight methods use the same 3,000 cases: 1,135 two-hop, 1,136 three-hop, and 729 four-hop cases. We compare standard prompting, which asks for a direct answer, with CoT prompting, which asks for stepby-step reasoning before answering. Table 4 reports accuracy within each group. We observe that:

• EngramEdit gains substantially from CoT at all three depths, whereas FT-L, AdaLoRA, and MoEEdit lose accuracy in every group. This contrast shows that the benefit of CoT depends on the editing method, not just the reasoning prompt. For EngramEdit, CoT can help because intermediate steps explicitly express facts whose �-grams activate updated embeddings, allowing revised information to guide subsequent reasoning.

• The best method varies across hop groups under standard prompting, but EngramEdit leads every group with CoT. Its advantage persists through four-hop questions, despite lower accuracy and smaller margins at greater depths. This supports the use of revised knowledge in longer reasoning chains, which require more facts to be accessed and combined.

## C.1.3 Analysis of ZsRE Specificity

To understand whether higher ZsRE Specificity reflects better preservation, we compare correctness on unrelated queries before and after 2,000 edits. All seven methods are evaluated on the same 11,379 neighborhood prompts from the first 2,000 edit cases. We compare their recorded post-edit correctness labels against one shared evaluation of unedited LongCat. EngramEdit’s results come from the first 2,000 edits of the 5,000-edit run, separate from the main-table run.

Table 4: MQuAKE accuracy (%) by hop with standard and CoT prompts. Brackets give Wilson 95% confidence intervals. Bold and underlining mark the best and second-best scores.
<table><tr><td rowspan="2">Method</td><td colspan="3">Standard↑</td><td colspan="3">CoT↑</td></tr><tr><td>2-hop</td><td>3-hop</td><td>4-hop</td><td>2-hop</td><td>3-hop</td><td>4-hop</td></tr><tr><td>FT</td><td>7.93 [6.50, 9.65]</td><td>2.11 [1.42, 3.12]</td><td>7.82 [6.08, 10.00]</td><td>12.60 [10.79, 14.66]</td><td>3.61 [2.67, 4.86]</td><td>9.74 [7.79, 12.11]</td></tr><tr><td>FT-L</td><td>1.59 [1.01, 2.49]</td><td>1.67 [1.07, 2.60]</td><td>4.53 [3.24, 6.29]</td><td>1.06 [0.61, 1.84]</td><td>0.97 [0.54, 1.73]</td><td>1.78 [1.05, 3.03]</td></tr><tr><td>AdaLoRA</td><td>1.50 [0.94, 2.39]</td><td>2.64 [1.86, 3.74]</td><td>3.84 [2.67, 5.50]</td><td>1.15 [0.67, 1.95]</td><td>1.23 [0.74, 2.06]</td><td>2.33 [1.46, 3.70]</td></tr><tr><td>UnKE</td><td>4.14 [3.13, 5.46]</td><td>4.23 [3.20, 5.56]</td><td>6.17 [4.65, 8.16]</td><td>5.46 [4.28, 6.94]</td><td>4.58 [3.51, 5.95]</td><td>9.60</td></tr><tr><td>MoEEdit</td><td>3.00 [2.15, 4.16]</td><td>2.64 [1.86, 3.74]</td><td>3.29 [2.22, 4.85]</td><td>0.62 [0.30, 1.27]</td><td>0.26</td><td>[7.67, 11.96] 0.27</td></tr><tr><td>MFT-S</td><td>6.61</td><td>2.55</td><td>5.49</td><td>8.28</td><td>[0.09, 0.77] 3.61</td><td>[0.08, 0.99] 3.98</td></tr><tr><td>MFT-A</td><td>[5.30, 8.20] 5.02 [3.90, 6.45]</td><td>[1.78, 3.64] 1.32 [0.80, 2.17]</td><td>[4.06, 7.39] 5.62 [4.17, 7.54]</td><td>[6.82, 10.03] 4.32 [3.28, 5.66]</td><td>[2.67, 4.86] 2.02</td><td>[2.78, 5.65] 3.57</td></tr><tr><td>EngramEdit</td><td>9.52 [7.94, 11.36]</td><td>2.82 [2.00, 3.95]</td><td>5.49 [4.06, 7.39]</td><td>36.92 [34.16, 39.76]</td><td>[1.35, 3.02] 20.07 [17.84, 22.50]</td><td>[2.45, 5.17] 15.09 [12.67, 17.87]</td></tr></table>

Table 5: ZsRE correctness transitions after 2,000 edits. C and W denote correct and wrong predictions; Post–Pre is the Specificity change in percentage points. Stable Corr. measures unchanged correctness states. Values are case-averaged percentages with 95% CI half-widths in subscripts.
<table><tr><td>Method</td><td> $\mathbf { C } \to \mathbf { W }$ </td><td>W→C</td><td>Post-Pre</td><td>Post Spec. ↑</td><td>Stable Corr. ↑</td></tr><tr><td>FT-L</td><td> $0 . 6 2 _ { \pm 0 . 1 9 }$ </td><td> $4 . 8 4 _ { \pm 0 . 5 7 }$ </td><td>+4.23</td><td> $4 5 . 7 6 _ { \pm 1 . 2 4 }$ </td><td> $9 4 . 5 4 _ { \pm 0 . 6 0 }$ </td></tr><tr><td>AdaLoRA</td><td> $2 . 2 5 _ { \pm 0 . 3 3 }$ </td><td> $7 . 5 1 _ { \pm 0 . 7 1 }$ </td><td>+5.26</td><td> $4 6 . 7 9 _ { \pm 1 . 2 7 }$ </td><td> $9 0 . 2 4 _ { \pm 0 . 7 9 }$ </td></tr><tr><td>UnKE</td><td> $2 . 9 5 _ { \pm 0 . 3 6 }$ </td><td> $1 1 . 2 6 _ { \pm 0 . 8 6 }$ </td><td>+8.31</td><td> $4 9 . 8 5 _ { \pm 1 . 3 2 }$ </td><td> $8 5 . 7 9 _ { \pm 0 . 9 3 }$ </td></tr><tr><td>MoEEdit</td><td> $2 . 3 0 _ { \pm 0 . 3 5 }$ </td><td> $5 . 1 4 _ { \pm 0 . 5 6 }$ </td><td>+2.84</td><td> $4 4 . 3 7 _ { \pm 1 . 2 6 }$ </td><td> $9 2 . 5 6 _ { \pm 0 . 6 7 }$ </td></tr><tr><td>MFT-S</td><td> $5 . 0 6 _ { \pm 0 . 5 1 }$ </td><td> $0 . 4 0 _ { \pm 0 . 1 2 }$ </td><td>-4.67</td><td> $3 6 . 8 7 _ { \pm 1 . 1 7 }$ </td><td> $9 4 . 5 4 _ { \pm 0 . 5 4 }$ </td></tr><tr><td>MFT-A</td><td> $9 . 6 7 _ { \pm 0 . 7 2 }$ </td><td> $1 . 8 4 _ { \pm 0 . 3 0 }$ </td><td>-7.83</td><td> $3 3 . 7 1 _ { \pm 1 . 1 3 }$ </td><td> $8 8 . 4 8 _ { \pm 0 . 8 0 }$ </td></tr><tr><td>EngramEdit</td><td> $5 . 2 5 _ { \pm 0 . 5 4 }$ </td><td> $0 . 9 2 _ { \pm 0 . 2 5 }$ </td><td>-4.33</td><td> $3 7 . 2 1 _ { \pm 1 . 1 8 }$ </td><td> $9 3 . 8 3 _ { \pm 0 . 6 0 }$ </td></tr></table>

Let C and W denote correct and wrong predictions. For each transition $a  b$ , we compute its proportion over all prompts within each case, then average equally across cases to obtain $P ( a  b )$ . Post-edit Specificity counts both retained correct predictions $( \mathsf { C } \to \mathsf { C } )$ and newly correct predictions $( \mathsf { W } { \to } \mathsf { C } )$ . We report the change in Specificity and Stable Correctness, the proportion of unchanged correctness states:

$$
\begin{array} { r } { \Delta S \mathrm { p e c i f i c i t y } = P ( \mathsf { W } \to \mathsf { C } ) - P ( \mathsf { C } \to \mathsf { W } ) , \qquad } \\ { \mathrm { S t a b l e ~ C o r r e c t n e s s } = 1 - P ( \mathsf { C } \to \mathsf { W } ) - P ( \mathsf { W } \to \mathsf { C } ) . } \end{array}\tag{56}
$$

Stable Correctness includes predictions that remain wrong, so it difers from retention of previously correct predictions. It also does not imply identical outputs, since one wrong token can change to another. We report 95% confidence-interval half-widths of 1.95996�/ 2000, where � is the sample standard deviation of the case-level proportions. Table 5 summarizes the results. We observe that:

• FT-L, AdaLoRA, UnKE, and MoEEdit exceed pre-edit Specificity because their $\mathsf { W } \to \mathsf { C }$ rates exceed their $\mathrm { C } \to \mathsf { W }$ rates. Their higher post-edit accuracy than EngramEdit reflects both better retention of initially correct predictions and more corrections of initially wrong predictions.

![](images/750364224f341821aa6332b03346acd4e017efa770de093ec49a11197082e297.jpg)  
Figure 8: Task-level general capabilities during 5,000 sequential CounterFact edits. F1 is measured on 100 examples per task. Dotted lines mark pre-edit scores.

• EngramEdit preserves 93.83% of correctness states and about 87.4% of initially correct predictions under case-weighted aggregation. Its W → C rate of only 0.92% shows that its post-edit Specificity comes predominantly from retained correct predictions. Together with the high Eficacy and Generalization reported in Section 4.2, these results show that EngramEdit supports efective, generalizable knowledge updates while retaining most previously correct predictions on unrelated queries.

## C.1.4 Task-Level General Capabilities

To assess capability preservation across individual tasks as edits accumulate, we track six task-level F1 scores during 5,000 sequential CounterFact edits in batches of 100. We compare EngramEdit with FT, UnKE, MoEEdit, and MFT-S on 100 examples per task, evaluating before editing and every 500 edits. Figure 8 shows each method’s class-frequency-weighted F1 trajectory, with pre-edit scores as references. We observe that:

• EngramEdit preserves pre-edit F1 on SST and CoLA throughout editing, while RTE, MMLU, and NLI stay within four F1 points of their pre-edit scores. By comparison, FT and UnKE show large SST gains and NLI losses. These trajectories show that EngramEdit largely preserves performance across five tasks throughout sequential editing. This stability can be partly explained by restricting edits to selected embeddings and applying stronger penalties to frequently reused ones, which limits changes to memory shared with unrelated inputs.

• EngramEdit’s larger changes are concentrated on MRPC, and MFT-S shows a similar pattern. This suggests that MRPC is more sensitive to memory updates than the other evaluated tasks. MRPC requires comparing the meanings of two sentences, so updates that afect their representations diferently may change the equivalence judgment. Such diferences can arise when the sentences activate diferent �-grams.

Table 6: Editing with and without generated expressions on CounterFact and ZsRE over 2,000 sequential edits. Subscripts give 95% CI half-widths. Bold and underlining mark the best and second-best scores within each block.
<table><tr><td rowspan="2">Method</td><td colspan="4">CounterFact</td><td colspan="4">ZsRE</td></tr><tr><td>Eff.↑</td><td>Gen.↑</td><td>Spec. ↑</td><td>Util.↑</td><td>Eff.↑</td><td>Gen. ↑</td><td>Spec. ↑</td><td>Util. ↑</td></tr><tr><td colspan="9">Original edit prompt only</td></tr><tr><td>MoEEdit</td><td> $\underline { { 9 9 . 1 } } _ { \pm 0 . 4 1 }$ </td><td> $\underline { { 6 7 . 1 } } _ { \cdot \pm 1 . 6 9 }$ </td><td> $6 3 . 5 _ { \pm 1 . 2 8 }$ </td><td>76.6</td><td> $\underline { { 8 5 . 1 } } _ { \pm 0 . 9 1 }$ </td><td> $\underline { { 7 8 . 7 } } _ { \pm 1 . 2 1 }$ </td><td> $4 4 . 4 _ { \pm 1 . 2 6 }$ </td><td>69.4</td></tr><tr><td>MFT-S</td><td> $9 9 . 0 _ { \pm 0 . 4 5 }$ </td><td> $5 9 . 1 _ { \pm 1 . 9 2 }$ </td><td> $\underline { { 8 5 . 2 } } _ { \pm 0 . 9 0 }$ </td><td>81.1</td><td> $6 6 . 9 _ { \pm 1 . 3 5 }$ </td><td> $5 6 . 5 _ { \pm 1 . 5 3 }$ </td><td> $3 6 . 9 _ { \pm 1 . 1 7 }$ </td><td>53.4</td></tr><tr><td>MFT-A</td><td> $9 3 . 9 _ { \pm 1 . 0 4 }$ </td><td> $4 0 . 7 _ { \pm 1 . 7 5 }$ </td><td> $6 6 . 6 _ { \pm 1 . 1 2 }$ </td><td>67.1</td><td> $6 2 . 8 _ { \pm 1 . 4 9 }$ </td><td> $5 1 . 3 _ { \pm 1 . 5 7 }$ </td><td> $3 3 . 7 _ { \pm 1 . 1 3 }$ </td><td>49.3</td></tr><tr><td>EngramEdit</td><td> $9 9 . 5 _ { \pm 0 . 3 2 }$ </td><td> $7 5 . 0 _ { \pm 1 . 7 4 }$ </td><td> $8 5 . 6 _ { \pm 0 . 9 1 }$ </td><td>86.7</td><td> $9 5 . 1 _ { \pm 0 . 6 2 }$ </td><td> $8 3 . 4 _ { \pm 1 . 2 9 }$ </td><td> $\underline { { 3 8 . 5 _ { \pm 1 . 1 8 } } }$ </td><td>72.3</td></tr><tr><td colspan="9">Original edit prompt + 4 generated expressions</td></tr><tr><td>MoEEdit</td><td> $8 0 . 7 _ { \pm 1 . 7 3 }$ </td><td> $5 9 . 8 _ { \pm 1 . 7 1 }$ </td><td> $6 6 . 0 _ { \pm 1 . 2 1 }$ </td><td>68.8</td><td> $\underline { { 7 9 . 8 _ { \pm 1 . 1 2 } } }$ </td><td> $\underline { { 7 2 . 9 _ { \pm 1 . 3 5 } } }$ </td><td> $4 2 . 8 _ { \pm 1 . 2 3 }$ </td><td>65.2</td></tr><tr><td>MFT-S</td><td> $\underline { { 9 9 . 0 _ { \pm 0 . 4 5 } } }$ </td><td> $\underline { { 9 0 . 7 } } _ { \pm 1 . 0 9 }$ </td><td> $\underline { { 8 5 . 0 _ { \pm 0 . 9 1 } } }$ </td><td>91.5</td><td> $6 7 . 2 _ { \pm 1 . 3 7 }$ </td><td> $6 4 . 2 _ { \pm 1 . 4 3 }$ </td><td> $3 7 . 7 _ { \pm 1 . 1 7 }$ </td><td>56.4</td></tr><tr><td>MFT-A</td><td> $9 3 . 8 _ { \pm 1 . 0 6 }$ </td><td> $6 3 . 2 _ { \pm 1 . 7 2 }$ </td><td>63.6±1.20</td><td>73.5</td><td> $6 9 . 8 _ { \pm 1 . 3 1 }$ </td><td> $6 5 . 9 _ { \pm 1 . 4 3 }$ </td><td> $3 1 . 9 _ { \pm 1 . 1 1 }$ </td><td>55.9</td></tr><tr><td>EngramEdit</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.9</td><td> $9 7 . 3 _ { \pm 0 . 4 0 }$ </td><td> $9 3 . 7 _ { \pm 0 . 7 9 }$ </td><td> $\underline { { 3 8 . 3 } } _ { \pm 1 . 1 8 }$ </td><td>76.4</td></tr></table>

## C.2 Analysis of Editing Choices

To examine how editing choices afect knowledge updating and preservation, we first compare editors with and without generated expressions. We then vary EngramEdit’s expression count, editable �-gram lengths, regularization coeficients, and clamp factor. These studies complement the component analyses in Section 4.4.

## C.2.1 Multi-Expression Editing

To examine how generated expressions afect diferent editors, we compare MoEEdit, MFT-S, MFT-A, and EngramEdit with zero or four generated expressions per fact on CounterFact and ZsRE. Each setting uses 2,000 sequential edits in batches of 100 and retains the original edit prompt. All methods use the same generated expressions in the four-expression setting. From Table 6, we observe that:

• Adding four generated expressions improves Generalization for MFT-S, MFT-A, and EngramEdit on both datasets, whereas MoEEdit’s Eficacy and Generalization decrease. The benefit therefore depends on how each editor uses the expressions. Generated expressions can help memoryonly editors because their diferent wordings activate additional �-grams, allowing held-out paraphrases to access revised facts through a larger set of updated embeddings.

• EngramEdit leads in Eficacy, Generalization, and Utility on both datasets with either expression count. On CounterFact, it generalizes better than MFT-S at similar Eficacy and Specificity when both use the same expressions and memory update interface. On ZsRE, its stronger Eficacy and Generalization yield higher Utility than MoEEdit despite lower Specificity. EngramEdit thus uses the same expressions more efectively for knowledge updating. By computing targets across expressions and matching them jointly, it accounts for the requirements of all expressions sharing each embedding. Reuse-based regularization further limits large changes to embeddings also used by unrelated inputs.

## C.2.2 Number of Generated Expressions

To assess the benefit of generating more expressions per edit, we compare $K \in \{ 0 , 2 , 4 , 6 , 8 \}$ generated expressions on CounterFact and ZsRE. Each setting uses 2,000 sequential edits in batches of 100. The original edit prompt is always retained, and all other settings are fixed. From Table 7, we observe that:

Table 7: Results with diferent numbers of generated expressions after 2,000 edits. # Expr. excludes the original prompt. Subscripts give reported CI half-widths; bold and underlining mark best and second-best scores.
<table><tr><td rowspan="2"># Expr.</td><td colspan="6">CounterFact</td><td colspan="4">ZsRE</td></tr><tr><td>Eff.↑</td><td>Gen.↑</td><td>Spec. ↑</td><td>Util.↑</td><td>Flu.↑</td><td>Cons. ↑</td><td>Eff.↑</td><td>Gen.↑</td><td>Spec. ↑</td><td>Util.↑</td></tr><tr><td>0</td><td> $9 9 . 5 _ { \pm 0 . 3 2 }$ </td><td> $7 5 . 0 _ { \pm 1 . 7 4 }$ </td><td> $8 5 . 6 _ { \pm 0 . 9 1 }$ </td><td>86.7</td><td> $5 6 2 . 6 _ { \pm 1 . 1 9 }$ </td><td> $1 5 . 7 _ { \pm 0 . 3 6 }$ </td><td> $9 5 . 1 _ { \pm 0 . 6 2 }$ </td><td> $8 3 . 4 _ { \pm 1 . 2 9 }$ </td><td> $3 8 . 5 _ { \pm 1 . 1 8 }$ </td><td>72.3</td></tr><tr><td>2</td><td> $9 9 . 4 _ { \pm 0 . 3 4 }$ </td><td> $9 5 . 2 _ { \pm 0 . 8 3 }$ </td><td> $\underline { { 8 5 . 3 _ { \pm 0 . 9 2 } } }$ </td><td>93.3</td><td> $\underline { { 5 6 6 . 2 _ { \pm 1 . 1 1 } } }$ </td><td> $1 6 . 2 _ { \pm 0 . 3 6 }$ </td><td> $9 6 . 2 _ { \pm 0 . 5 0 }$ </td><td> $9 1 . 8 _ { \pm 0 . 8 9 }$ </td><td> $3 7 . 7 _ { \pm 1 . 1 0 }$ </td><td>75.2</td></tr><tr><td>4</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.9</td><td> $5 6 5 . 2 _ { \pm 1 . 1 0 }$ </td><td> $\underline { { 1 6 . 1 _ { \pm 0 . 3 6 } } }$ </td><td> $9 7 . 3 _ { \pm 0 . 4 0 }$ </td><td> $\underline { { 9 3 . 7 } } _ { \pm 0 . 7 9 }$ </td><td> $\underline { { 3 8 . 3 _ { \pm 1 . 1 8 } } }$ </td><td>76.4</td></tr><tr><td>6</td><td> $9 9 . 4 _ { \pm 0 . 3 4 }$ </td><td> $9 7 . 3 _ { \pm 0 . 6 4 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>94.0</td><td> $5 6 7 . 5 _ { \pm 1 . 0 3 }$ </td><td> $1 6 . 2 _ { \pm 0 . 3 6 }$ </td><td> $9 6 . 5 _ { \pm 0 . 4 7 }$ </td><td> $\underline { { 9 3 . 7 } } _ { \pm 0 . 7 7 }$ </td><td> $3 6 . 8 _ { \pm 1 . 1 7 }$ </td><td>75.7</td></tr><tr><td>8</td><td> $9 9 . 1 _ { \pm 0 . 4 1 }$ </td><td> $\underline { { 9 7 . 2 } } _ { \pm 0 . 6 7 }$ </td><td> $8 4 . 9 _ { \pm 0 . 9 3 }$ </td><td>93.7</td><td> $5 6 6 . 0 _ { \pm 1 . 0 7 }$ </td><td> $1 6 . 0 _ { \pm 0 . 3 6 }$ </td><td> $\underline { { 9 6 . 8 _ { \pm 0 . 4 4 } } }$ </td><td> $9 4 . 2 _ { \pm 0 . 7 4 }$ </td><td> $3 8 . 0 _ { \pm 1 . 1 7 }$ </td><td>76.3</td></tr></table>

<table><tr><td rowspan=6 colspan=5>Efficacy ↑Generalization ↑Specificity ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2,3</td><td rowspan=1 colspan=1> $^ { 2 , 4 }$ </td><td rowspan=1 colspan=1> $^ { 3 , 4 }$ </td><td rowspan=1 colspan=1>2,3,4</td></tr><tr><td rowspan=1 colspan=4>cacy ↑</td><td rowspan=1 colspan=1>↑</td><td rowspan=1 colspan=1>98.3±0.57</td><td rowspan=1 colspan=1>90.3±1.30</td><td rowspan=1 colspan=1>68.9±2.03</td><td rowspan=1 colspan=1>98.7±0.50</td><td rowspan=1 colspan=1>99.0±0.44</td><td rowspan=1 colspan=1>90.5±1.29</td><td rowspan=1 colspan=1>99.5±0.31</td><td rowspan=1 colspan=1>99.0±0.44</td></tr><tr><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=3>ion 个</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>↑</td><td rowspan=1 colspan=1>96.4±0.77</td><td rowspan=1 colspan=1>83.2±1.56</td><td rowspan=1 colspan=1>57.8±2.06</td><td rowspan=1 colspan=1>96.8±0.71</td><td rowspan=1 colspan=1>95.8±0.80</td><td rowspan=1 colspan=1>83.2±1.55</td><td rowspan=1 colspan=1>97.0±0.69</td><td rowspan=1 colspan=1>96.0±0.74</td></tr><tr><td rowspan=1 colspan=1>84.5±0.94</td><td rowspan=1 colspan=1>86.6±0.89</td><td rowspan=1 colspan=1>87.4±0.89</td><td rowspan=1 colspan=1>85.0±0.92</td><td rowspan=1 colspan=1>85.2±0.92</td><td rowspan=1 colspan=1>86.7±0.89</td><td rowspan=1 colspan=1>85.2±0.92</td><td rowspan=1 colspan=1>74.6±1.13</td></tr><tr><td rowspan=1 colspan=13>Utility ↑Editable n-gram lengths                    Score (%)50               75              100</td></tr></table>

Figure 9: Editable �-gram lengths on CounterFact. The frame marks the default lengths 2, 3, and 4; $n = 1$ denotes single-token embedding updates. Small ± values give CI half-widths.

• Two generated expressions provide most of the Generalization gain on both datasets, with further improvements at four. Eficacy remains high, while Specificity on both datasets and Fluency and Consistency on CounterFact vary little. A small expression set thus supports broader access to revised knowledge with largely preserved unrelated-query performance. Generated expressions enable this access by adding the �-grams activated by diferent wordings to memory mapping, so held-out paraphrases have more opportunities to activate updated embeddings.

• Beyond four expressions, Generalization improves only slightly and Utility does not consistently increase. Four expressions achieve the highest Utility on ZsRE and nearly the highest on CounterFact. This supports � = 4 as the default, since larger sets require more expression processing without comparable performance gains.

## C.2.3 Editable �-gram Lengths

To identify which �-gram lengths support both efective updates and preservation, we compare eight editable-length configurations on CounterFact using 2,000 sequential edits in batches of 100. We test lengths 2, 3, and 4 individually, in pairs, and together, as well as the full combination with single-token (� = 1) embedding updates. All settings use four generated expressions and reusebased regularization; only the lengths eligible for embedding updates change. From Figure 9, we observe that:

• Among single-length settings, 2-grams achieve the highest Eficacy and Generalization, while longer �-grams give higher Specificity. Shorter �-grams therefore favor access across expressions, whereas longer ones favor preservation. This is because shorter token sequences can activate the same updated embedding across both paraphrases and unrelated inputs.

Table 8: Sensitivity to regularization and clamp factor after 2,000 CounterFact edits. Only the listed parameter changes. Scores are percentages; subscripts give reported CI half-widths. Bold parameter values mark the defaults.
<table><tr><td>Parameter</td><td>Value</td><td> $\mathbf { E f f } . \uparrow$ </td><td> $\mathbf { G e n . } \uparrow$ </td><td>Spec. ↑</td><td>Util. ↑</td></tr><tr><td rowspan="5">Reuse-based regularization λreuse</td><td>0</td><td> $9 8 . 8 _ { \pm 0 . 4 9 }$ </td><td> $9 6 . 5 _ { \pm 0 . 7 6 }$ </td><td> $8 4 . 7 _ { \pm 0 . 9 3 }$ </td><td>93.3</td></tr><tr><td>0.03</td><td> $9 9 . 3 _ { \pm 0 . 3 7 }$ </td><td> $9 6 . 2 _ { \pm 0 . 7 6 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.6</td></tr><tr><td>0.05</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.9</td></tr><tr><td>0.075</td><td> $9 9 . 6 _ { \pm 0 . 2 9 }$ </td><td> $9 6 . 4 _ { \pm 0 . 7 3 }$ </td><td> $8 5 . 3 _ { \pm 0 . 9 1 }$ </td><td>93.8</td></tr><tr><td>0.1</td><td> $9 9 . 6 _ { \pm 0 . 2 8 }$ </td><td> $9 6 . 6 _ { \pm 0 . 7 0 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.8</td></tr><tr><td rowspan="5">Ridge regularization λridge</td><td>0</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 6 . 6 _ { \pm 0 . 7 1 }$ </td><td> $8 5 . 1 _ { \pm 0 . 9 2 }$ </td><td>93.7</td></tr><tr><td>0.001</td><td> $9 9 . 6 _ { \pm 0 . 2 9 }$ </td><td> $9 6 . 6 _ { \pm 0 . 7 2 }$ </td><td> $8 5 . 3 _ { \pm 0 . 9 2 }$ </td><td>93.8</td></tr><tr><td>0.01</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.9</td></tr><tr><td>0.03</td><td> $9 9 . 4 _ { \pm 0 . 3 5 }$ </td><td> $9 6 . 8 _ { \pm 0 . 7 1 }$ </td><td> $8 5 . 3 _ { \pm 0 . 9 2 }$ </td><td>93.8</td></tr><tr><td>0.1</td><td> $9 9 . 1 _ { \pm 0 . 4 1 }$ </td><td> $9 6 . 6 _ { \pm 0 . 7 3 }$ </td><td> $8 4 . 8 _ { \pm 0 . 9 3 }$ </td><td>93.5</td></tr><tr><td rowspan="4">Clamp factor  $\rho$ </td><td>8</td><td> $9 9 . 1 _ { \pm 0 . 4 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.8</td></tr><tr><td>16</td><td> $9 9 . 4 _ { \pm 0 . 3 5 }$ </td><td> $9 6 . 7 _ { \pm 0 . 7 1 }$ </td><td> $8 5 . 3 _ { \pm 0 . 9 2 }$ </td><td>93.8</td></tr><tr><td>32</td><td> $9 9 . 5 _ { \pm 0 . 3 1 }$ </td><td> $9 7 . 0 _ { \pm 0 . 6 9 }$ </td><td> $8 5 . 2 _ { \pm 0 . 9 2 }$ </td><td>93.9</td></tr><tr><td>64</td><td> $9 9 . 6 _ { \pm 0 . 2 8 }$ </td><td> $9 6 . 4 _ { \pm 0 . 7 3 }$ </td><td> $8 5 . 0 _ { \pm 0 . 9 3 }$ </td><td>93.7</td></tr></table>

• Combining lengths 2, 3, and 4 achieves the highest Eficacy, Generalization, and Utility, with higher Specificity than using 2-grams alone. This suggests that diferent lengths can complement each other. Shorter �-grams support access across expressions, while longer ones provide more context-specific embeddings that can carry part of the update, reducing reliance on embeddings shared with unrelated inputs.

• Excluding single-token embedding updates yields markedly higher Specificity than editing lengths 1–4, with no loss in Eficacy or Generalization. This supports restricting the default editable lengths to 2–4. Individual tokens occur in many unrelated inputs, so single-token updates can afect a wider range of predictions without providing additional editing gains here.

## C.2.4 Regularization and Clamp Factor

To test whether strong editing performance requires narrowly tuned regularization or perturbation bounds, we vary one parameter at a time on CounterFact using 2,000 sequential edits in batches of 100. The reuse coeficient $\lambda _ { \mathrm { { r e u s e } } }$ scales the length- and frequency-based penalties, while the ridge coeficient $\lambda _ { \mathrm { { r i d g e } } }$ penalizes all embedding updates equally. The clamp factor $\rho$ bounds the perturbation norm during target computation. Table 8 lists the tested values and marks the defaults. Other settings remain fixed, including four generated expressions and editable lengths 2, 3, and 4. We observe that:

• All tested nonzero reuse coeficients yield higher Eficacy, Specificity, and Utility than disabling the reuse-based term, with little variation among them. The gains therefore do not depend on one precisely tuned strength. Changing the coeficient scales the reuse-based penalties but retains stronger constraints on frequently reused embeddings, which can help preserve unrelated knowledge across these settings.

• All four metrics remain close across ridge coeficients, including zero. This shows that strong performance does not require a precisely tuned uniform penalty. Reuse-based regularization still constrains embedding updates when ridge is removed, so the extra term mainly provides another way to control overall update size.

• Eficacy, Generalization, and Specificity remain stable as the clamp factor increases from 8 to 64, with no consistent Utility gain from larger values. Strong editing performance thus holds across the tested norm bounds. A larger factor permits, but does not force, a larger perturbation, since prediction loss and target regularization still guide its optimization.

![](images/125247947dc962607268351152738dcbb9b59842e0b9b31c9c13e3f5c179e522.jpg)  
Figure 10: Disabling memory updates on CounterFact. (a) Mean edit margins before and after disabling. Positive values favor revised answers; negative values favor original answers. (b) Outcomes of the 5,850 prompts successful before disabling. Parentheses give the mean number of disabled updates per case.

## C.3 Analysis of Memory Use

To understand how revised facts are accessed through conditional memory, we first test whether successful recall depends on fact-related updates by selectively disabling them, extending Section 4.4. With all updates enabled, we next group inputs by the updated embeddings they activate, then analyze how cross-edit sharing relates to editing outcomes as updates accumulate. These later analyses leave the edited model unchanged and identify associations, while disabling directly tests the efect of removing updates.

## C.3.1 Disabling Memory Updates

To test whether revised-fact recall depends on fact-related memory updates, we disable selected updates during inference. We use the model from Figure 5(c), with 9,253 updated embeddings from 2,000 sequential CounterFact edits. Fact-related updates correspond to �-grams selected at the last subject token of each fact’s original and four generated expressions. Four conditions disable 1) no updates (None); 2) the evaluated fact’s updates (Fact-related); 3) other updates matched in number and �-gram length (Matched random); or 4) all updates (All). The random control (seed 0) excludes fact-related updates but does not match update norms or corpus frequencies. Original embeddings remain active, and updates are restored after each evaluation without further editing. Disabling a shared embedding’s update removes the combined changes from all associated edits. Diferences between All and separately recorded pre-edit outputs may reflect finite-precision variation. For each prompt � and edit $e _ { i } ,$ the edit margin measures revised-answer preference:

$$
\mathrm { m a r g i n } _ { i } ( x ) = \mathrm { s c o r e } ( x , o _ { i } ^ { \star } ) - \mathrm { s c o r e } ( x , o _ { i } ) ,\tag{57}
$$

using length-normalized log-likelihood scores. Positive margins favor revised answers and negative margins original answers. We aggregate margins from edit prompts and held-out paraphrases within each case, then average cases equally. Of the 2,000 edit prompts and 4,000 held-out paraphrases, 5,850 succeed under None. The success-to-failure rate reports the percentage of these same prompts that no longer favor the revised answer after disabling. From Figure 10, we observe that:

• Fact-related disabling makes the model favor original over revised answers on average and turns 5,228 of the 5,850 successful prompts into failures. These updates therefore support revised-fact recall for most previously successful prompts. Since the original embeddings remain active, the reversal reflects removal of the learned updates, not removal of the underlying memory.

<table><tr><td colspan="3">Fact-related activation: 98.05%</td><td colspan="2">Overall regression: 3.23%</td></tr><tr><td>Activation</td><td>Success rate within group (%) ↑</td><td>Share of paraphrases (%)</td><td>Regression rate within group (%)↓</td><td>Share of regressions (%)</td></tr><tr><td>Current only</td><td>98.97</td><td>63.35</td><td>92.31</td><td>6.35</td></tr><tr><td>Other only</td><td>10.34</td><td>0.73</td><td>29.43</td><td>79.72</td></tr><tr><td>Both</td><td>96.97</td><td>34.70</td><td>57.38</td><td>6.17</td></tr><tr><td>None</td><td>12.24</td><td>1.23</td><td>0.28</td><td>7.76</td></tr><tr><td colspan="2">0 50</td><td>100 0 (a) Generalization</td><td>50</td><td>100</td></tr></table>

Figure 11: Memory activation and editing outcomes with all updates enabled. Groups indicate activation of embeddings updated for the current edit only, other edits only, both, or neither. (a) Within-group paraphrase success rates and each group’s share of all paraphrases. (b) Success-to-failure rates among previously successful neighborhood queries in each group and each group’s share of all regressions.

• Matched random disabling changes the mean margin by only −0.0003 and causes no successful prompt to fail. The contrast shows that recall depends on which updates are disabled, not simply how many. Fact-related updates remain available under this control, allowing the model to continue using them to recall revised facts.

• Disabling an average of 4.679 fact-related updates nearly matches disabling all 9,253 updates in both mean margin and success-to-failure rate. This shows that recalling revised facts depends primarily on a small set of memory updates, activated when the input contains the corresponding fact-related �-grams.

## C.3.2 Memory Activation and Editing Outcomes

To relate memory activation to generalization and preservation, we evaluate 4,000 held-out paraphrases and 20,000 neighborhood prompts after completing 2,000 sequential CounterFact edits, with all updates enabled. The current edit is the edit paired with each prompt in Counter-Fact; neighborhood prompts ask about unrelated facts. Using the fact-related �-gram sets from Appendix C.3.1, we scan all prompt positions before answer generation and group prompts by the updated embeddings they activate: 1) Current only, those associated only with the current edit; 2) Other only, only with other edits; 3) Both, with both; and 4) None, no updated embeddings. An embedding shared across edits counts toward each associated edit. The sets cover all 9,253 updated �-grams, and including teacher-forced answer prefixes leaves the groups unchanged.

Generalization. A held-out paraphrase succeeds when the revised answer scores above the original answer under length-normalized log-likelihood. The held-out paraphrases do not overlap with generated expressions under exact or normalized matching. Figure 11(a) shows the success rate within each group and the group’s share of all 4,000 paraphrases. We observe that:

• Current only and Both together cover 98.05% of held-out paraphrases, with success rates of 96.97–98.97%. High success thus extends to most held-out paraphrases and is associated with activation of fact-related updates. Diferent wordings can share �-grams, so expressions not used during editing can still use embeddings updated for the same fact.

• Other only and None contain just 1.95% of held-out paraphrases but account for 50.36% of failures. This concentration of failures supports broader memory mapping across expressions, as tested in Section 4.4. These paraphrases cannot directly use the updates for their fact because none of the corresponding �-grams are activated.

![](images/63b3ea6a245ff6aa694a2ccce9425e5531edd3bdedf150c6e5a9d653a0eee230.jpg)  
(a) Cross-Edit Sharing

<table><tr><td>Group</td><td>Facts n (%)</td><td>Eff. ↑ (%)</td><td>(%)</td><td>Gen. 个 Spec. 个 (%)</td><td>Eff. failures n (%)</td></tr><tr><td>No sharing</td><td>4,567 (91.34)</td><td>99.98</td><td>96.7</td><td>83.7</td><td>1 (2.04)</td></tr><tr><td>Sharing</td><td>433 (8.66)</td><td>88.91</td><td>84.1</td><td>82.4</td><td>48 (97.96)</td></tr></table>

Facts: % of 5,000; failures: % of 49 (b) Editing Outcomes  
Figure 12: Cross-edit sharing on CounterFact. (a) Sharing rates among updated �-grams and edited facts. (b) Final editing results for facts with or without shared �-grams.

Specificity. On neighborhood prompts, success requires the original answer to score above the edit’s revised answer. We compare the same prompts before and after editing and count initially successful prompts that become failures as regressions. Figure 11(b) shows the regression rate among initially successful prompts in each group and the group’s share of all 567 regressions. We observe that:

• Among the 17,534 initially successful prompts, 96.77% remain successful and 3.23% regress; the regression rate in None is only 0.28%. Most previously correct predictions are thus retained, with especially strong preservation in None because these inputs do not directly use embedding updates. The 44 regressions in None may reflect finite-precision diferences that reverse close answer rankings across inference runs.

• Other only accounts for 79.72% of all regressions, whereas Current only accounts for 6.35% despite its higher within-group regression rate. Overall, 92.24% of regressions accompany activation of updated embeddings. Regressions are thus concentrated in inputs that activate updates for other edits. Diferent facts can share �-grams, so an update for one fact can also afect unrelated predictions. This supports using reuse-based regularization to penalize large updates to frequently reused embeddings.

## C.3.3 Cross-Edit Sharing

To examine how cross-edit sharing relates to editing outcomes, we analyze the 5,000-edit CounterFact run every 100 edits. We track �-grams selected at the last subject token of each fact’s original and four generated expressions. A fact belongs to Sharing if any selected �-gram is also selected for another fact, and to No sharing otherwise. Figure 12 reports sharing rates and groups the saved final results. We observe that:

• Sharing increases overall but involves only 285 of 22,778 updated �-grams (1.25%) and 433 facts (8.66%) at 5,000 edits. Most embeddings are therefore selected for one fact; the higher fact-level rate reflects that a shared embedding serves multiple facts.

• No sharing covers 91.34% of facts with 99.98% Eficacy and 96.7% Generalization, showing sustained editing success across expressions for most facts. Their selected embeddings are not shared with other edits, so diferent facts do not compete for updates to the same parameters.

• Sharing contains 48 of the 49 Eficacy failures (97.96%), with lower Generalization but similar Specificity. Failures are thus concentrated in a small group of facts sharing updated embeddings, where diferent targets may require conflicting changes to the same parameters.