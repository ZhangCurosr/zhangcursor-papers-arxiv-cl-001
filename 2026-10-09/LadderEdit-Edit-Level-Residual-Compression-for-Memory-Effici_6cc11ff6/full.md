# LadderEdit: Edit-Level Residual Compression for Memory-Efficient Lifelong Editing of LLMs

Xiaobing Yu<sup>1,2</sup> Peijie Qiu<sup>2\*</sup> Jin Yang<sup>3</sup> Xuanzhao Dong<sup>4</sup> Weiwei Ma<sup>2</sup> Zhaoqi An<sup>2</sup> Xiaoqi Zhao<sup>1†</sup> Xiaofeng Liu<sup>1†</sup> <sup>1</sup>Yale University; <sup>2</sup>Washington University in St. Louis; <sup>3</sup>Icahn School of Medicine at Mount Sinai; <sup>4</sup>Arizona State University {xiaobing.yu, xiaoqi.zhao, xiaofeng.liu}@yale.edu <sup>†</sup> co-corresponding authors Code: https://github.com/VisualReasoner/LadderEdit

## Abstract

Lifelong editing of LLMs requires storing thousands of edits after acquisition. A widely used family of approaches attaches one LoRA adapter per edit, which preserves behavior but grows linearly in storage. To address this challenge, we propose LadderEdit, a method that compresses each LoRA adapter after it is acquired. Each edit is first stored at low rank as a cheap sketch. We then check whether this sketch still satisfies the rewrite, generalization, and locality contract on probe prompts. Edits that pass keep the sketch; those that fail are promoted to a higher rank along a ladder until the contract is met. Because every edit retains some representation, coverage is maintained, and only hard edits consume more rank. Across ZsRE, CounterFact, and WikiBigEdit benchmarks on LLaMA-3-8B, Mistral-7B, and Qwen2.5-7B, LadderEdit tracks exact LoRA storage at 5.2× less memory and remains effective at 50,000 sequential edits.

## 1 Introduction

Lifelong model editing asks a deployed large language model to absorb a stream of factual corrections and knowledge updates without full retraining. This setting is increasingly important: world knowledge changes, model errors are discovered after deployment, and domain-specific assistants require controlled updates over time. A useful lifelong editor must therefore satisfy three requirements simultaneously. It must remain plastic enough to incorporate new edits, stable enough to preserve previous edits and unrelated knowledge, and efficient enough to serve after thousands of updates.

Existing approaches expose a fundamental storage-plasticity tension. Shared-parameter and constant-memory editors bound the deployed memory footprint, but every new edit competes with previous edits for limited capacity (Hartvigsen et al.,

![](images/f231f106acd25288661beefb3aece3beea90b3264f05dfa14f9d7c02a33ebfa6.jpg)  
Figure 1: Performance comparison of LadderEdit, exact LoRA, and baseline methods across sequential editing tasks. LadderEdit usually tracks exact LoRA closely while reducing persistent memory, with small losses in some compressed settings.

2023; Wang et al., 2025). As the stream grows, such methods can preserve locality while losing edit plasticity: the model avoids damaging unrelated knowledge, but new edits may fail to generalize beyond the rewrite prompt. Retrieval- and routing-based editors reduce direct parameter interference but introduce a separate edit-selection problem and can fail when a query is semantically close to, but not exactly matched with, the stored edit (Li et al., 2024a; Wang and Li, 2024b; Zhang et al., 2023; Wang and Li, 2024a). Direct parameterediting methods avoid explicit per-edit storage, yet repeated updates can interfere destructively over long edit streams (Meng et al., 2023, 2022; De Cao et al., 2021; Dai et al., 2022; Fang et al., 2025; Wang et al., 2023; Mitchell et al., 2022a).

An alternative is to store edit-local update memories. Each edit is written into an isolated update representation, so later edits do not overwrite earlier ones (Hu et al., 2022; Yu et al., 2023). This design preserves edit-specific plasticity and provides a clean full-storage reference: if the correct edit memory is selected, the system can recover the behavior of the original update. The drawback is persistent storage. If every edit is retained at full resolution, memory grows linearly with the number of edits. Thus, the bottleneck in long-horizon editing is not only how to acquire an edit but also how much of each acquired edit must remain stored.

To address this dilemma, we reframe lifelong edit storage as an edit-level resolution allocation problem. The key is not simply which edits should be kept or dropped but how much representational resolution each edit needs to preserve its behavioral contract. This distinction matters because keep/drop caching couples memory reduction to edit absence: once an edit is dropped, it has no representation and cannot support rewrite, generalization, or locality. In contrast, a resolution-allocation policy can preserve coverage for every edit through a cheap representation, while spending additional memory only on edits that require higher fidelity.

We propose LADDEREDIT, a storage-layer method based on this coverage-before-fidelity principle. Each incoming edit is first acquired as a fullresolution update memory; in our implementation, this update memory is an exact LoRA update. LadderEdit is a two-stage controller. A spectral predictor scores each candidate rank using the singularvalue tail of the update together with the edit’s behavioral slack, and proposes the cheapest rank $\boldsymbol { { \hat { r } } _ { i } }$ whose score falls below a calibrated threshold; this proposal requires only one SVD per edit and no behavioral evaluation. A behavioral audit then verifies the proposed sketch against rewrite, generalization, and locality probes. If the audit passes, the edit is stored at ${ \hat { r } } _ { i } ;$ if it fails, the audit promotes the edit along the spectrally-ordered ladder until a passing rung is found. This design avoids exhaustive auditing across all rungs and gives every edit coverage through its predicted rung while spending additional resolution only when audit demands it.

We validate on ZsRE (Levy et al., 2017), CounterFact (CF) (Meng et al., 2022, 2023), and WikiBigEdit (Thede et al., 2025) using LLaMA-3- 8B (Grattafiori et al., 2024), Mistral-7B (Jiang et al., 2023), and Qwen2.5-7B (Yang et al., 2024; Bai et al., 2025; Yang et al., 2025). Figure 1 shows LadderEdit tracks the full-resolution reference at a fraction of the memory, and budget-matched frontiers confirm that residual-ladder allocation substantially outperforms exact-cache: preserving coverage before fidelity is the better storage primitive for scalable lifelong editing.

Our contributions are as follows: i) We formulate lifelong editing storage as a resolution-allocation problem, separating edit acquisition, selection, and persistent representation. ii) We introduce LadderEdit, a residual-ladder controller that covers every edit with a low-rank sketch and restores detail only when behavioral audits demand it. iii) We identify coverage-before-fidelity as the principle for scalable edit memory, nearly doubling utility over exact-cache at matched budget. iv We validate LadderEdit on ZsRE, CF, and WikiBigEdit, tracking full-resolution memory at 5.2× less storage and remaining effective at 50,000 edits.

## 2 Related Work

Model editing and lifelong editing. Model editing methods update a model’s behavior without full retraining, either by directly modifying parameters (Sinitsin et al., 2020; De Cao et al., 2021; Meng et al., 2022, 2023), learning editor networks (Mitchell et al., 2022a), or routing edited cases through external/contextual memories (Mitchell et al., 2022b; Zheng et al., 2023). These methods establish the basic reliability, generalization, and locality criteria, but most are not designed for long edit streams where thousands of updates must coexist. Lifelong editing methods address this sequential setting through key-value memories, side memories, null-space constraints, or adapter/MoE-based mechanisms (Hartvigsen et al., 2023; Wang et al., 2024a; Fang et al., 2025; Wang et al., 2025; Yu et al., 2023; Li et al., 2024a; Wang and Li, 2024b,a). However, they still face a storage-plasticity tension: shared or constantmemory systems can lose plasticity as capacity saturates, while edit-local memories preserve behavior but grow with the number of edits. LadderEdit targets this storage layer directly.

Low-rank adaptation and edit-memory storage. LoRA and its variants show that low-rank updates provide an efficient representation for adapting large models (Hu et al., 2022; Dettmers et al., 2023; Liu et al., 2024; Lialin et al., 2024; Yu et al., 2025). Adaptive-rank methods such as AdaLoRA (Zhang et al., 2023) allocate rank across layers or modules during training, while serving systems such as S-LoRA (Sheng et al., 2023) study how to deploy many adapters efficiently. LadderEdit addresses a different problem: after an edit has already been acquired, how much of its update memory must be persistently stored? The unit of allocation is an edit rather than a layer, and the criterion is behavioral contract satisfaction rather than gradient-derived importance. Thus LadderEdit is not ordinary LoRA compression or training-time rank allocation; it is an edit-level resolution-allocation method that keeps every edit covered by a sketch and promotes to higher rank only when needed.

![](images/1d56df9e5e8faa058c3ca8d2d7a9f92647826eddd35bd82e62a7de60e5d659a6.jpg)  
Figure 2: LadderEdit insertion pipeline. We first fit an exact LoRA update for the incoming edit, then a candidate rank $\boldsymbol { { \hat { r } } } _ { i }$ from the update spectrum and exact-edit margin (one SVD, no behavioral cost). We further decompose the edit into ladder rungs: low-rank sketches at increasing rank. And finally we audit the predicted sketch against rewrite, generalization, and locality. If the audit passes, store at ${ \hat { r } } _ { i } ;$ otherwise promote along the spectral ladder until a passing rung is found.

Evaluation under compression. Recent surveys and evaluations emphasize that edited models should be judged not only by rewrite success but also by generalization, locality, long-form behavior, and degradation over many edits (Yao et al., 2023; Zhang et al., 2024; Wang et al., 2024b; Rosati et al., 2024; Li et al., 2024b; Zheng et al., 2024; Wu et al., 2024). LadderEdit follows this view at the storage level: spectral compressibility proposes candidate rungs, but the final decision is made by behavioral auditing over rewrite, generalization, and locality. This is the central difference from exact-cache allocation, which reduces memory by dropping edits rather than by allocating resolution.

LadderEdit’s formulation differs from ordinary LoRA compression and adaptive rank training. LadderEdit does not learn a better adapter at acquisition time and does not solve retrieval or task identification; it operates after the edit has been acquired, inheriting the selection mechanism of the full-resolution memory bank, and asks a storagelayer question: once an edit is selected, can the system return a cheaper representation without violating the measured edit contract? The advantage comes from changing the memory primitive. Exact-cache allocation saves memory by storing a subset of edits exactly and dropping the rest, reducing coverage. LadderEdit first gives every edit coverage through a sketch, then allocates fidelity through additional rank only where the audit shows it is needed. The two policies fail differently under compression: exact-cache fails by absence, whereas LadderEdit degrades by resolution. A focused comparison that distinguishes LadderEdit from training-time rank allocation (Zhang et al., 2023), multi-adapter serving (Sheng et al., 2023), and adapter merging is provided in Appendix D.

## 3 Method

## 3.1 Problem Setting

We consider lifelong editing for a pretrained LLM $f _ { \theta } ,$ where edits arrive as a sequence $\{ e _ { i } \} _ { i = 1 } ^ { N }$ . After each insertion, the system must answer the edited prompt correctly, generalize the edit to related prompts, and preserve unrelated knowledge. We refer to these requirements as reliability, generalization, and locality. The scaling challenge is not only to satisfy this behavioral contract for one edit, but to keep satisfying it as N grows.

We instantiate each edit with a LoRA-style update. Let $\Delta _ { i } ^ { E }$ denote the exact update produced by the edit writer for edit $e _ { i }$ . Two extremes bracket the space of persistent representations: a fullresolution edit-local memory bank stores every $\Delta _ { i } ^ { E }$ exactly, scaling linearly as $N P _ { E }$ where $P _ { E }$ is the cost of one exact edit; a shared or constantmemory editor folds all edits into a fixed footprint, but plasticity degrades as later edits compete with earlier ones for finite capacity. Neither extreme matches the empirical structure of an edit stream: across $N = 2 { , } 0 0 0$ edits on both ZsRE and Counter-Fact, roughly 80% of edits are behaviorally sketchsufficient at rank 1, $a \sim 1 9 \%$ hard tail requires rank 2 or higher, and a smaller ∼ 5% population is compression-regularized, the rank-1 sketch satisfies the behavioral contract that the exact update fails (Appendix A.1). LadderEdit occupies the intermediate regime that this distribution implies: every edit retains its own representation, but that representation may be a compressed rung rather than the full exact adapter. Table 4 reports the population structure of the edit stream that motivates this design.

The system rests on three commitments. It is exactfirst: each edit is acquired at full resolution by the standard writer, so the acquired update is preserved without loss. It is retrieval agnostic: it inherits whatever selection mechanism the deployment uses, with the storage gain preserved under a learned sentence encoder retriever (Appendix A). It is contract grounded: at insertion, a behavioral audit certifies that the returned representation satisfies the same rewrite, generalization, and locality test as $\Delta _ { i } ^ { E }$ . At inference, retrieval selects the edit and LadderEdit returns the cheapest audited rung; the acquisition cost of $\Delta _ { i } ^ { E }$ is paid once, and the persistent cost is whatever the audit permits. LadderEdit decides which directions of $\bar { \Delta } _ { i } ^ { E }$ to retain and composes with quantization, which acts on bit width (Appendix A.4).

## 3.2 Residual-Ladder Allocation

LadderEdit is a residual-ladder allocation mechanism for scaling edit-local update memories in lifelong editing. It keeps the isolation and plasticity of per-edit adapters, but stores most edits at lower resolution. The method has three steps: acquire an exact update, compile that update into candidate rungs, and audit the cheapest safe rung. Figure 2 summarizes the full insertion-time pipeline, and Algorithm 1 gives the controller.

For an incoming edit $e _ { i } ,$ , let $\Delta _ { i } ^ { E }$ denote the exact LoRA-style update acquired by the edit writer. LadderEdit constructs a rank-r sketch:

$$
\Delta _ { i } ^ { ( r ) } = \Pi _ { r } ( \Delta _ { i } ^ { E } ) ,\tag{1}
$$

We use the rank menu $\mathcal { R } ~ = ~ \{ 1 , 2 , \dots , R _ { \operatorname* { m a x } } \}$ matching the same acquisition rank as $R _ { \mathrm { m a x } }$ from the exact LoRA update.

where Π truncates the singular directions of the effective LoRA update. The omitted detail is the residual:

$$
\rho _ { i } ^ { ( r ) } = \Delta _ { i } ^ { E } - \Delta _ { i } ^ { ( r ) } .\tag{2}
$$

The deployed edit is one rung of the ladder:

$$
\Delta _ { i } = \Delta _ { i } ^ { ( r _ { i } ) } .\tag{3}
$$

When $r _ { i } ~ = ~ R _ { \operatorname* { m a x } }$ , the stored update equals the exact LoRA edit. The key difference from exactcache allocation is that every edit keeps a representation, and memory is spent on resolution rather than on a binary keep/drop decision.

Rank selection starts from a cheap spectral proposal. Let $T _ { i } ( r )$ be the normalized spectral tail after rank r. We define the exact-edit margin as the minimum behavioral slack of the full-resolution edit:

$$
\mu _ { i } ^ { E } = \operatorname * { m i n } \{ R _ { i } ^ { E } - \tau _ { R } , G _ { i } ^ { E } - \tau _ { G } , L _ { i } ^ { E } - \tau _ { L } \} .\tag{4}
$$

Positive margin means the acquired edit satisfies all three measured constraints, and larger margin indicates more tolerance for compression.

LadderEdit scores:

$$
S _ { i } ( \boldsymbol { r } ) = \frac { T _ { i } ( \boldsymbol { r } ) } { [ \mu _ { i } ^ { E } ] _ { + } + \epsilon } ,\tag{5}
$$

where $\epsilon > 0$ prevents unstable division. Lower scores indicate that little spectral mass remains outside the sketch relative to the exact edit’s behavioral slack. The cheapest rank whose score falls below a calibrated threshold τ is the predicted rank ${ \hat { r } } _ { i } \mathbf { : }$

$$
\hat { r } _ { i } = \operatorname* { m i n } \{ r \in \mathcal { R } : S _ { i } ( r ) \leq \tau \} .\tag{6}
$$

The spectral score is a proposal, not a safety proof; it selects which single rung the audit will inspect first, so the audit sees one candidate per edit instead of |R| candidates.

Behavioral auditing remains the final gate. For a candidate stored representation $m ,$ , let $R _ { i } ^ { m } , G _ { i } ^ { m }$ and $L _ { i } ^ { m }$ denote rewrite, generalization, and locality scores. We define the soft contract gap:

$$
\begin{array} { r l } & { H _ { i } ^ { m } = w _ { R } [ \tau _ { R } - R _ { i } ^ { m } ] _ { + } + w _ { G } [ \tau _ { G } - G _ { i } ^ { m } ] _ { + } } \\ & { \qquad + w _ { L } [ \tau _ { L } - L _ { i } ^ { m } ] _ { + } . } \end{array}\tag{7}
$$

The audit score in Eq. 7 uses weights $w _ { R } = 1 . 0 $ $w _ { G } = 0 . 8$ , and $w _ { L } = 1 . 2$ . Locality receives the highest weight because locality failures are catastrophic and irreversible, while reliability and generalization failures trigger rank promotion and recover within the same edit. We swept $w _ { L }$ over {0.8, 1.0, 1.2, 1.5, 2.0} on the validation split and Avg. stayed within 0.01 of the default for $w _ { L } \in$ [1.0, 1.5] and dropped by 0.02 and 0.03 at the endpoints 0.8 and 2.0 respectively. The default $w _ { L } = 1 . 2$ sits at the midpoint of the stable region. The weights $w _ { R }$ and $w _ { G }$ follow the standard LoRA loss-weighting ratio. We demonstrate more details in Appendix A.3.

Downgrade policy. Algorithm 1 includes budgetbased downgrading. When the audit succeeds at rank r, the controller checks whether the rank-$( r { - } 1 )$ approximation $\begin{array} { r } { U _ { : , 1 : r - 1 } \Sigma _ { 1 : r - 1 , 1 : r - 1 } V _ { 1 : r - 1 , } ^ { T } } \end{array}$ captures more than 92% of the Frobenius energy of the full LoRA update. If so, the controller commits the rank-(r−1) sketch and re-runs the audit on the downgraded sketch, and if the downgrade also passes the audit, it is kept. Downgrade attempts are bounded at one per edit to keep insertion cost bounded. Downgrades occur for 3.1% of edits at $T = 1 { , } 0 0 0$ on LLaMA-3-8B / ZsRE and save approximately 0.6 GB of persistent memory at $T = 1 0 { , } 0 0 0$ . The 0.92 Frobenius threshold was selected on the validation split, and varying it from 0.88 to 0.95 changes Avg. by less than 0.005.

If $H _ { i } ^ { m } = 0 ;$ , the candidate satisfies the edit contract. If the audit at $\hat { r } _ { i }$ fails $( H _ { i } ^ { ( { \hat { r } } _ { i } ) } > 0 )$ , LadderEdit re-audits at the next rung along the spectrallyordered ladder $\hat { r } _ { i } + 1 , \hat { r } _ { i } + 2 , . .$ . until the contract is met or $R _ { \mathrm { m a x } }$ is reached. In practice the predicted rung passes on the first audit for the majority of edits (Appendix A.5), so the expected number of audits per edit is close to one.

Under a memory budget, higher-rank promotion is allocated where it removes the most contract violation per parameter. Let $P _ { E }$ be the exact edit cost and $P _ { r }$ the rank-r sketch cost. The promotion value density is:

$$
\mathrm { R V D } _ { i } ( r ) = { \frac { H _ { i } ^ { ( r ) } - H _ { i } ^ { E } } { P _ { E } - P _ { r } } } .\tag{8}
$$

Algorithm 1 LADDEREDIT: spectral-prediction   
with audited promotion   
Require: Edit stream $\{ e _ { i } \} _ { i = 1 } ^ { N }$ , frozen model $f _ { \theta } ,$ rank menu   
$\mathbf { \bar { \mathcal { R } } } = \{ 1 , \dots , R _ { \operatorname* { m a x } } \}$ , threshold τ , memory budget B   
1: Initialize stored memory $M \gets \emptyset$   
2: for each incoming edit $e _ { i }$ do   
3: Acquire exact LoRA update $\Delta _ { i } ^ { E }$   
4: Compute SVD of $\Delta _ { i } ^ { E }$ and form spectral tails $T _ { i } ( r )$   
for all $r \in \mathcal { R }$   
5: Estimate exact-edit margin $\mu _ { i } ^ { E }$ on rewrite, generaliza  
tion, and locality probes   
$_ { 6 ; }$ Predict: $\hat { r } _ { i } \gets$ min $\{ r \in \mathcal { R } : S _ { i } ( r ) \leq \tau \}$ ▷ single   
proposal, no behavioral cost   
7: $r _ { i } \gets \hat { r } _ { i }$   
8: Audit: build sketch $\Delta _ { i } ^ { ( r _ { i } ) }$ , compute contract gap   
$H _ { i } ^ { ( r _ { i } ) }$   
9: while $H _ { i } ^ { ( r _ { i } ) } > 0$ and $r _ { i } < R _ { \operatorname* { m a x } }$ do ▷ audited   
promotion   
10: $r _ { i } \gets r _ { i } + 1$   
11: Rebuild sketch $\Delta _ { i } ^ { ( r _ { i } ) }$ , recompute $H _ { i } ^ { ( r _ { i } ) }$   
12: end while   
13: Store $\Delta _ { i } ^ { ( r _ { i } ) }$ in M   
14: if stored memory exceeds B then   
15: Downgrade edits with lowest value density   
16: end if   
17: end for   
18: return stored edit memory M

If all N edits use rank r and K edits are promoted to the exact rung, persistent memory is:

$$
M ( r , K ) = N P _ { r } + K ( P _ { E } - P _ { r } ) .\tag{9}
$$

For analytical clarity, Eq. 8–9 use a two-state limit in which promoted edits jump directly to $R _ { \mathrm { m a x } } .$ The deployed controller supports intermediate promotion along the full ladder R. The first term gives every edit a cheap plastic representation. The second term spends exact memory only on the difficult subset. This yields a practical route for one-LoRAper-edit scaling whenever $K \ll N$

The rank-analysis package also clarifies how to interpret storage rungs. An edit is sketch-sufficient if both sketch and exact pass, detail-needed if the sketch fails but exact passes, compressionregularized if the sketch passes but exact fails, and exact-limited if both fail. More details in Appendix D.6.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate LadderEdit as a storage-layer method for edit-local memories. The central question is whether lifelong edit storage should be allocated by coverage and fidelity, rather than by exact keep/drop caching. We therefore compare exact LoRA as the full-resolution reference, fixed-rank compression as a nonadaptive baseline, exact-cache as a keep/drop policy, and LadderEdit as an editlevel resolution-allocation policy across standard sequential editing benchmarks.

Datasets. We conduct experiments on three benchmarks: ZsRE (Levy et al., 2017), Counter-Fact (CF) (Meng et al., 2022, 2023), and WikiBigEdit (Thede et al., 2025). ZsRE is a questionanswering dataset for evaluating factual knowledge and generalization in language models. CF tests the ability to edit and preserve specific factual associations in a controlled setting. WikiBigEdit is a large-scale benchmark for long-horizon, up to 50,000 edits, sequential editing.

We evaluate LadderEdit on LLaMA-3- 8B (Grattafiori et al., 2024), Mistral-7B (Jiang et al., 2023), and Qwen2.5-7B (Yang et al., 2025). We compare against representative direct-editing methods, memory- and routing-based lifelong editors, constrained-update methods, LoRA/MoEstyle edit-memory baselines, adaptive-rank LoRA, and the full-resolution exact LoRA reference. Detailed baseline descriptions are deferred to Appendix F.3.

Edit lookup protocol. LadderEdit evaluates storage for edit-local memories, not edit selection. For edit-local methods, each rewrite, generalization, and locality probe is associated with the edit that generated it. Exact LoRA and LadderEdit use the same association, so they differ only in the stored representation returned after lookup. Methods whose routing is part of the original algorithm, such as memory- or MoE-based lifelong editors, are evaluated with their native routing mechanisms.

We report reliability, generalization, locality, and their average when applicable. Locality is the fraction of unrelated probes whose prediction remains unchanged relative to the frozen base model. 1.00 means no failures on the finite locality probe set.

Probe split protocol. For each edit, the benchmark-provided prompts are partitioned into three disjoint sets at edit-construction time. The audit set contains the rewrite prompt and a heldout subset of rephrase/generalization variants, used only at insertion time to score $S _ { i } ( r )$ and gate the contract gap $H _ { i } ^ { ( r ) }$ . The validation set is a separate stream used only to select audit thresholds, which are then frozen before the test stream is touched. The test set contains the remaining rephrase/generalization variants and unrelated locality probes; it is used only to report Reliability, Generalization, and Locality in the main tables and is never seen by the audit. Audit and test sets share neither prompts nor paraphrases of the same prompt. Full split statistics, audit-test correlation, and a leaky-versus-clean comparison are reported in Appendix C. More details in Appendix B.1.

## 4.2 Results

Main Results. Tables 1 and 2 show that LadderEdit preserves the plasticity of edit-local adapters while reducing their persistent storage. On ZsRE, FT, ROME, and MEMIT degrade quickly, and GRACE preserves locality but fails to generalize, especially on Mistral-7B and Qwen2.5-7B. Exact LoRA remains the strongest full-storage reference because each edit has isolated capacity, but LadderEdit tracks it after compression: on LLaMA-3-8B it improves the average over exact LoRA at T = 500 (0.97 vs. 0.93) and matches it at T = 1,000 and $T \ = \ 2 { , } 0 0 0 { : }$ on Mistral-7B it slightly improves reliability or generalization at longer horizons; on Qwen2.5-7B it stays within 0.01 to 0.02 average of exact LoRA at the largest horizons. CounterFact gives the same message in a setting that emphasizes locality: at $T = 2 { \mathrm { , 0 0 0 } }$ LadderEdit moves exact LoRA from 0.71/0.93 to 0.74/0.94 in reliability/locality on LLaMA-3-8B and from 0.68/0.91 to 0.71/0.92 on Mistral-7B. These gains suggest that compression is not merely lossy: low-rank sketches can remove overly specific directions, while the audit promotes edits to higher rank when more fidelity is needed.

WikiBigEdit Long-Horizon Scaling. At $T =$ 50,000 on WikiBigEdit (Table 3), LadderEdit reaches 0.75 reliability, 0.84 generalization, 0.98 locality, and 0.86 average. GRACE holds reliability and locality but drops generalization to 0.28; WISE loses both reliability and generalization; MEM-OIR reaches only 0.69 average. Together with the memory trade-off in Figure 3, LadderEdit preserves edit-local behavior at this scale while substantially reducing persistent storage.

Memory Efficiency and Ablation. The memory results support LadderEdit’s central claim: lifelong edit storage should be treated as per-edit resolution allocation. Table 4 reports the empirical population structure of the edit stream and confirms the heterogeneity assumed in Section 3.1. On both ZsRE and CounterFact, roughly 80% of edits are sketchsufficient at rank 1, a 13 to 17% tail requires rank 2 or higher to satisfy the behavioral contract, and a smaller 3 to 5% subset requires the full acquisition rank. A 4 to 7% compression-regularized population, where the rank-1 sketch passes the contract that exact LoRA fails, shows that the full update is not a behavioral ceiling. This distribution justifies the ladder design: a single fixed rank cannot serve all edits, but a small higher-rank budget plus a cheap default rung suffices for the bulk of the stream. Per-rank compression statistics (memory, retained energy, false retirements) are reported in Appendix F. A predictor ablation isolating the spectral score, the behavioral margin, and their product against random and oracle baselines is reported in

Table 1: Q&A task results on the ZsRE dataset. T denotes the number of edits. Mean ± standard deviation over three random seeds. Avg. is the arithmetic mean of Rel., Gen., and Loc. The best average performance per column is marked in bold.
<table><tr><td rowspan="2">Method</td><td colspan="4">T = 100</td><td colspan="4">T = 500</td><td colspan="4">T = 1,000</td><td colspan="4"></td></tr><tr><td>Rel.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td><td>Rel.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td><td>Rel.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td><td>Rel.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>LLaMA-3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FT</td><td>0.13 ±.004</td><td>0.11 ±.005</td><td>0.02 ±.001</td><td>0.09 ±.002</td><td>0.14 ±.005</td><td>0.12 ±.006</td><td>0.02 ±.001</td><td>0.09 ±.003</td><td>0.13 ±.005</td><td>0.12 ±.007</td><td>0.01 ±.001</td><td>0.09 ±.003</td><td>0.09 ±.006</td><td>0.08 ±.007</td><td>0.01 ±.001</td><td>0.06 ±.003</td></tr><tr><td>ROME (Meng et al., 2022)</td><td>0.08 ±.003</td><td>0.08 ±.003</td><td>0.02 ±.001</td><td>0.06 ±.001</td><td>0.04 ±.002</td><td>0.04 ±.002</td><td>0.02 ±.001</td><td>0.03 ±.001</td><td>0.03 ±.002</td><td>0.03 ±.002</td><td>0.02 ±.001</td><td>0.03 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td></tr><tr><td>MEMIT (Meng et al., 2023)</td><td>0.03 ±.002</td><td>0.03 ±.002</td><td>0.01 ±.001</td><td>0.02 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.00 ±.000</td><td>0.01 ±.001</td><td>0.00±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.001</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.001</td></tr><tr><td>GRACE (Hartvigsen et al., 2023)</td><td>1.00 ±.001</td><td>0.39 ±.009</td><td>1.00 ±.001</td><td>0.80 ±.003</td><td>1.00±.001</td><td>0.38 ±.011</td><td>1.00±.001</td><td>0.79 ±.004</td><td>1.00±.001</td><td>0.37 ±.012</td><td>1.00±.001</td><td>0.79 ±.004</td><td>1.00±.001</td><td>0.34 ±.013</td><td>1.00±.001</td><td>0.78 ±.004</td></tr><tr><td>WISE (Wang et al., 2024a)</td><td>0.62 ±.014</td><td>0.60 ±.015</td><td>1.00±.001</td><td>0.74 ±.007</td><td>0.69 ±.017</td><td>0.66±.018</td><td>1.00 ±.001</td><td>0.78 ±.008</td><td>0.66 ±.019</td><td>0.64 ±.020</td><td>1.00 ±.001</td><td>0.77 ±.009</td><td>0.55 ±.021</td><td>0.52 ±.022</td><td>1.00 ±.001</td><td>0.69 ±.010</td></tr><tr><td>AlphaEdit (Fang et al., 2025)</td><td>0.91 ±.012</td><td>0.79 ±.013</td><td>0.94 ±.008</td><td>0.88 ±.006</td><td>0.89 ±.014</td><td>0.78 ±.016</td><td>0.81 ±.010</td><td>0.83 ±.008</td><td>0.84 ±.016</td><td>0.77 ±.018</td><td>0.56 ±.011</td><td>0.72 ±.009</td><td>0.62 ±.018</td><td>0.58 ±.019</td><td>0.21 ±.012</td><td>0.47 ±.010</td></tr><tr><td>MEMOIR (Wang et al., 2025)</td><td>0.87 ±.008</td><td>0.84 ±.010</td><td>0.95 ±.006</td><td>0.89 ±.005</td><td>0.82 ±.010</td><td>0.75 ±.012</td><td>0.82 ±.007</td><td>0.80±.006</td><td>0.77 ±.011</td><td>0.71 ±.014</td><td>0.80 ±.008</td><td>0.76 ±.007</td><td>0.71 ±.012</td><td>0.62 ±.015</td><td>0.77 ±.009</td><td>0.70 ±.007</td></tr><tr><td>ELDER (Li et al., 2024a)</td><td>0.78 ±.014</td><td>0.72 ±.016</td><td>0.93 ±.009</td><td>0.81 ± 008</td><td>0.69 ±.017</td><td>0.63 ±.019</td><td>0.91 ±.011</td><td>0.74 ±.009</td><td>0.61 ±.019</td><td>0.55 ±.022</td><td>0.89 ±.012</td><td>0.68 ±.010</td><td>0.49 ±.021</td><td>0.44 ±.024</td><td>0.86 ±.013</td><td>0.60±.011</td></tr><tr><td>MELO (Yu et al., 2023)</td><td>0.81 ±.012</td><td>0.76 ±.014</td><td>0.96 ±.008</td><td>0.84 ±.007</td><td>0.73 ±.014</td><td>0.67 ±.017</td><td>0.94 ±.010</td><td>0.78±.008</td><td>0.66 ±.016</td><td>0.59 ±.019</td><td>0.92 ±.011</td><td>0.72 ±.009</td><td>0.55 ±.018</td><td>0.49 ±.021</td><td>0.89 ±.012</td><td>0.64 ±.010</td></tr><tr><td>AdaLoRA (Zhang et al., 2023)</td><td>0.65 ±.020</td><td>0.55 ±.022</td><td>0.20±.010</td><td>0.47 ±.010</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.00 ±.000</td><td>0.01 ±.001</td><td>0.00±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00±.001</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.001</td></tr><tr><td>LoRA (Exact) (Hu et al., 2022)</td><td>1.00 ±.001</td><td>0.99 ±.002</td><td>1.00±.001</td><td>1.00±.001</td><td>0.85 ±.006</td><td>0.94±.007</td><td>1.00±.001</td><td>0.93 ±.003</td><td>0.95 ±.007</td><td>0.91 ±.008</td><td>1.00 ±.001</td><td>0.95 ±.004</td><td>0.94 ±.007</td><td>0.89 ±.009</td><td>1.00 ±.001</td><td>0.94±.004</td></tr><tr><td>LadderEdit (Ours)</td><td>1.00 ±.001 0.99 ±.002</td><td></td><td>1.00 ±.001</td><td>1.00±.001</td><td>0.97 ±.005</td><td>0.94 ±.006</td><td>1.00 ±.001</td><td>0.97 ±.003</td><td>0.95 ±.005</td><td>0.91 ±.007</td><td>1.00 ±.001</td><td>0.95 ±.003</td><td>0.94 ±.006</td><td>0.89 ±.007</td><td>1.00±.001</td><td>0.94±.003</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Mistral-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>0.02 ±.001</td><td>0.08 ±.003</td><td>0.15 ±.006</td><td>0.13 ±.007</td><td>0.02 ±.001</td><td>0.10 ±.003</td><td>0.16 ±.006</td><td>0.13 ±.008</td><td></td><td>0.10 ±.003</td><td>0.08 ±.007</td><td>0.07 ±.009</td><td>0.01 ±.001</td><td>0.05 ±.004</td></tr><tr><td>FT ROME (Meng et al., 2022)</td><td>0.11 ±.005 0.05 ±.002</td><td>0.10 ±.006 0.05 ±.002</td><td>0.02 ±.001</td><td>0.04 ±.001</td><td>0.04 ±.002</td><td>0.04 ±.002</td><td>0.02 ±.001</td><td>0.03 ±.001</td><td>0.04 ±.002</td><td>0.04 ±.002</td><td>0.01 ±.001 0.02 ±.001</td><td></td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td></tr><tr><td>MEMIT (Meng et al., 2023)</td><td>0.00±.000</td><td>0.00±.000</td><td>0.01 ±.001</td><td>0.00 ±.001</td><td>0.02 ±.001</td><td>0.02 ±.001</td><td>0.01 ±.001</td><td>0.02 ±.001</td><td>0.04 ±.002</td><td>0.04 ±.002</td><td>0.02 ±.001</td><td>0.03 ±.001 0.03 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.01 ±.001</td></tr><tr><td>GRACE (Hartvigsen et al., 2023)</td><td>1.00 ±.001</td><td>0.15 ±.010</td><td>1.00 ±.001</td><td>0.72 ±.003</td><td>1.00±.001</td><td>0.09 ±.012</td><td>1.00 ±.001</td><td>0.70 ±.004</td><td>1.00±.001</td><td>0.02 ±.001</td><td>1.00 ±.001</td><td>0.67 ±.001</td><td>1.00 ±.001</td><td>0.01 ±.001</td><td>1.00 ±.001</td><td>0.67 ±.001</td></tr><tr><td>WISE (Wang et al., 2024a)</td><td>0.87 ±.016</td><td>0.80 ±.017</td><td>1.00±.001</td><td>0.89±.008</td><td>0.80 ±.019</td><td>0.74 ±.021</td><td>1.00 ±.001</td><td>0.85 ±.009</td><td>0.70 ±.022</td><td>0.67 ±.023</td><td>1.00±.001</td><td>0.79 ±.011</td><td>0.55 ±.024</td><td>0.52 ±.026</td><td>1.00 ±.001</td><td>0.69 ±.012</td></tr><tr><td>AlphaEdit (Fang et al., 2025)</td><td>0.86 ±.014</td><td>0.74 ±.015</td><td>0.95 ±.009</td><td>0.85 ±.007</td><td>0.86 ±.017</td><td>0.73 ±.018</td><td>0.84 ±.011</td><td>0.81 ±.009</td><td>0.85±.019</td><td>0.72 ±.020</td><td>0.68 ±.012</td><td>0.75 ±.010</td><td>0.63 ±.021</td><td>0.55 ±.022</td><td>0.30 ±.014</td><td>0.49 ±.011</td></tr><tr><td>MEMOIR (Wang et al., 2025)</td><td>0.89 ±.009</td><td>0.87 ±.011</td><td>0.97 ±.007</td><td>0.91 ±.005</td><td>0.84 ±.011</td><td>0.78 ±.014</td><td>0.85 ±.008</td><td>0.82 ±.007</td><td>0.80±.012</td><td>0.74 ±.016</td><td>0.83 ±.009</td><td>0.79 ±.007</td><td>0.73 ±.014</td><td>0.65 ±.017</td><td>0.80 ±.010</td><td>0.73 ±.008</td></tr><tr><td>ELDER (Li et al., 2024a)</td><td>0.75 ±.016</td><td>0.69 ±.018</td><td>0.92 ±.010</td><td>0.79 ±.009</td><td>0.66 ±.019</td><td>0.60 ±.022</td><td>0.89 ±.012</td><td>0.72 ±.010</td><td>0.57 ±.022</td><td>0.52 ±.025</td><td>0.87 ±.014</td><td>0.65 ±.012</td><td>0.45 ±.024</td><td>0.41 ±.028</td><td>0.83 ±.016</td><td>0.56±.013</td></tr><tr><td>MELO (Yu et al., 2023)</td><td>0.78 ±.014</td><td>0.73 ±.016</td><td>0.95 ±.009</td><td>0.82 ±.008</td><td>0.70 ±.017</td><td>0.64 ±.019</td><td>0.93 ±.011</td><td>0.76±.009</td><td>0.62 ±.019</td><td>0.56 ±.022</td><td>0.90 ±.012</td><td>0.69 ±.010</td><td>0.51 ±.021</td><td>0.46 ±.024</td><td>0.87 ±.014</td><td>0.61 ±.012</td></tr><tr><td>AdaLoRA (Zhang et al., 2023)</td><td>0.60 ±.023</td><td>0.50 ±.025</td><td>0.18 ±.011</td><td>0.43 ±.012</td><td>0.01 ±.001</td><td>0.01 ±.001</td><td>0.00 ±.000</td><td>0.01 ±.001</td><td>0.00±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00±.001</td><td>0.00 ±.000</td><td>0.00±.000</td><td>0.00 ±.000</td><td>0.00 ±.001</td></tr><tr><td>LoRA (Exact) (Hu et al., 2022) LadderEdit (Ours)</td></table>

![](images/602d396a2818c748bc02e4c82431e5b9d5da0f81af4de619bf39ee0f037f95eb.jpg)

![](images/d21792a911e0665556768802cb3cb357aa2d3392f0a5e0c9753c09cb5dd08249.jpg)  
—— AdaLoRA—— ELDER -- MEIO -- MEMOIR-- LoBA (Exact) -+ LadderEdit  
Figure 3: Scaling reference on WikiBigEdit: average performance against persistent edit memory.

Appendix E; the combined predictor approaches oracle utility at far fewer audits per edit than any component alone.

Table 5 tests the memory primitive directly. The exact-cache baseline is validation-selected: under a budget of K full edits, it keeps those with the largest sketch-only contract gap and drops the rest. At 5.57× (CF) and 5.63× (ZsRE) compression, it retains about 91 of 512 edits, reaching utility 0.434 on CF and 0.420 on ZsRE; dropped edits have no representation. LadderEdit covers every edit with a rank-1 sketch: at 8× compression, the sketch alone reaches 0.753 (CF) and 0.765 (ZsRE), exceeding exact-cache by roughly 0.33 at lower memory, so coverage drives most of the gap. Behavioral rung selection adds the rest. At the matched budget, a static mixed-rank policy reaches 0.815 (CF) and 0.796 (ZsRE); the dual-budget ladder, spending that budget selectively on audit-flagged edits, reaches 0.862 (CF) and 0.866 (ZsRE). The 0.05 gap between static and dual-budget at identical memory is the signature of audit-driven allocation. Exactcache fails by absence; LadderEdit degrades by resolution.

Table 2: Sequential editing results on CF. T denotes the number of edits. Mean ± standard deviation over three random seeds. The best performance per column is marked in bold.
<table><tr><td rowspan="3">Method</td><td colspan="8">LLaMA-3-8B</td><td colspan="8">Mistral-7B</td></tr><tr><td colspan="2">T = 100</td><td colspan="2">T = 500</td><td colspan="2">T = 1,000</td><td colspan="2">T = 2,000</td><td colspan="2">T = 100</td><td colspan="2">T = 500</td><td colspan="2">T = 1,000</td><td colspan="2">T = 2,000</td></tr><tr><td>Rel.</td><td>Loc.</td><td>Rel.</td><td>Loc.</td><td>Rel.</td><td>Loc.</td><td>Rel.</td><td>Loc.</td><td>Rel.</td><td>Loc.</td><td>Rel.</td><td>Loc.</td><td>Rel.</td><td>Loc.</td><td>Rel.</td><td>Loc.</td></tr><tr><td>FT</td><td>0.42 ±.004</td><td>0.55 ± 001</td><td>0.25 ±005</td><td>0.30 ± 001</td><td>0.12 ±.005</td><td> $0 . 1 2 \pm \sigma 0 0$ </td><td>0.05 ±.002</td><td>0.03 ±.002</td><td>0.39 ±005</td><td> $0 . 5 2 \pm \sigma 0 0$ </td><td>0.22 ± 006</td><td>0.27 ±.001</td><td>0.10±.006</td><td>0.10 ± 002</td><td>0.03 ±.002</td><td>0.02 ±.001</td></tr><tr><td>ROME (Meng et al., 2022)</td><td>0.28 ±.003</td><td>0.45 ± 002</td><td>0.08 ±.004</td><td>0.15 ±002</td><td>0.02 ±.001</td><td>0.05 ±.002</td><td>0.00 ±00</td><td>0.02±.0011</td><td>0.25 ±.003</td><td>0.42 ±.002</td><td>0.06 ±.004</td><td>0.12 ± 003</td><td>0.01 ±.001</td><td>0.04 ±002</td><td>0.00 ±.0000</td><td>0.01 ±.001</td></tr><tr><td>MEMIT (Meng et al., 2023)</td><td>0.35 ±002</td><td>0.52 ±.001</td><td>0.12 ±.002</td><td>0.20 ± 001</td><td>0.03 ±.002</td><td>0.08 ±.001</td><td>0.01 ± 001</td><td>0.03 ±.002</td><td>0.31 ±.002</td><td>0.48 ±.001</td><td>0.09 ±.003</td><td>0.16 ±001</td><td>0.02 ±.001</td><td>0.06 ± 002</td><td>0.00 ±000</td><td>0.02 ±.001</td></tr><tr><td>GRACE (Hartvigsen et al., 2023)</td><td>0.75 ± 001</td><td>0.96 ±.001</td><td>0.71 ±.001</td><td>0.96 ±.001</td><td>0.68 ±.001</td><td>0.96 ±.001</td><td>0.62 ±.001</td><td>0.95 ±.001</td><td>0.72 ±.001</td><td>0.95 ± .001</td><td>0.68 ±.001</td><td>0.95 ±.001</td><td>0.65 ±.001</td><td>0.95 ±.001</td><td>0.59 ±.001</td><td>0.94±.001</td></tr><tr><td>WISE (Wang et al., 2024a)</td><td>0.68 ±.014</td><td>0.88 ±.002</td><td>0.62 ±.017</td><td>0.86 ±.002</td><td>0.55 ±.019</td><td>0.83 ±.003</td><td>0.45 ±.021</td><td>0.77 ±03</td><td>0.65 ±.016</td><td>0.86 ±.002</td><td>0.58 ±.019</td><td>0.83 ±.003</td><td>0.50 ±.022</td><td>0.80 ±.003</td><td>0.40 ±.024</td><td>0.73 ±.003</td></tr><tr><td>AlphaEdit (Fang et al., 2025)</td><td>0.62 ±.012</td><td>0.85 ±.08</td><td>0.58 ±.014</td><td>0.70 ±.010</td><td>0.52 ±.016</td><td>0.48 ±.011</td><td>0.42 ±.018</td><td>0.22 ±.012</td><td>0.58 ±.014</td><td>0.82 ±.009</td><td>0.54 ±.017</td><td>0.66 ±.011</td><td>0.47 ±.019</td><td>0.43 ±.012</td><td>0.37 ±.021</td><td>0.18 ±.014</td></tr><tr><td>MEMOIR (Wang et al., 2025)</td><td>0.78 ±008</td><td>0.88 ±.006</td><td>0.75 ±.010</td><td>0.94 ±.007</td><td>0.72 ±.011</td><td>0.93 ±.008</td><td>0.66 ±.012</td><td>0.90 ±.009</td><td>0.76 ±.009</td><td>0.86 ±.007</td><td>0.72 ±.011</td><td>0.93 ±.008</td><td>0.69 ±.012</td><td>0.91 ±.009</td><td>0.63 ±.014</td><td>0.88 ±.010</td></tr><tr><td>ELDER (Li et al., 2024a)</td><td>0.52 ±014</td><td>0.82 ± 009</td><td>0.46 ±.017</td><td>0.78 ± 011</td><td>0.39 ±.019</td><td>0.73 ±.012</td><td>0.30 ±.021</td><td>0.66 ±.013</td><td>0.49 ±.016</td><td>0.80 ±.010</td><td>0.43 ±.019</td><td>0.76 ±.012</td><td>0.36 ±.022</td><td>0.71 ±.014</td><td>0.27 ±.024</td><td>0.64 ±.016</td></tr><tr><td>MELO (Yu et al., 2023)</td><td>0.56 ±.012</td><td>0.86 ±.008</td><td>0.50 ±.014</td><td>0.82 ±.010</td><td>0.43 ±.016</td><td>0.78 ±011</td><td>0.34 ±.018</td><td>0.71 ± 012</td><td>0.53 ±.014</td><td>0.84 ±.009</td><td>0.47 ±.017</td><td>0.80 ±.011</td><td>0.40±.019</td><td>0.76 ±.012</td><td>0.31 ±.021</td><td>0.69 ±.014</td></tr><tr><td>AdaLoRA (Zhang et al., 2023)</td><td>0.18 ±.020</td><td>0.32 ±.010</td><td>0.02 ±.001</td><td>0.02 ±.001</td><td>0.00 ±.000</td><td>0.01 ±.01</td><td>0.00 ±.000</td><td>0.00 ±000</td><td>0.15 ±.023</td><td>0.30 ±.011</td><td>0.01 ±.001</td><td>0.02 ±.001</td><td>0.00±.000</td><td>0.01 ±.001</td><td>0.00 ±.000</td><td>0.00±.000</td></tr><tr><td>LoRA (Exact) (Hu et al., 2022)</td><td>0.82 ±.005</td><td>0.85 ± 002</td><td>0.78 ±.006</td><td>0.96 ± 002</td><td>0.75 ±.007</td><td>0.95 ±.003</td><td>0.71 ±.007</td><td>0.93 ±.003</td><td>0.80 ±.006</td><td>0.83 ±.002</td><td>0.76 ± 007</td><td>0.95 ±.003</td><td>0.72 ±.008</td><td>0.93 ± 003</td><td>0.68 ±.009</td><td>0.91 ±.003</td></tr><tr><td>LadderEdit (Ours)</td><td>0.83 ±.004</td><td>0.90 ±.002</td><td>0.78 ±.005</td><td>0.96 ±.002</td><td>0.77 ±.005</td><td>0.96±.003</td><td>0.74±.006</td><td>0.94 ±.003</td><td>0.81 ±.005</td><td>0.88 ±.002</td><td>0.77 ±.006</td><td>0.95 ±.003</td><td>0.74±.006</td><td>0.94 ±.003</td><td>0.71 ±.007</td><td>0.92 ±.003</td></tr></table>

Table 3: Editing performance on Qwen2.5-7B at longhorizon scale on WikiBigEdit. Mean ± standard deviation over three random seeds.
<table><tr><td rowspan="2">Method</td><td colspan="4">T = 10,000</td><td colspan="4">T = 50,000</td></tr><tr><td>Rel.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td><td>Rel.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td></tr><tr><td>MEMIT (Meng et al., 2023)</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.001</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>GRACE (Hartvigsen et al., 2023)</td><td>1.00 ±.001</td><td>0.31 ±.017</td><td>1.00 ±.001</td><td>0.77 ±.006</td><td>0.99 ±.002</td><td>0.28 ±.019</td><td>1.00±.001</td><td>0.76±.006</td></tr><tr><td>WISE (Wang et al., 2024a)</td><td>0.59 ±.027</td><td>0.57 ±.029</td><td>1.00 ±.001</td><td>0.72 ±.013</td><td>0.43 ±.030</td><td>0.41 ±.032</td><td>1.00 ±.001</td><td>0.61 ±.015</td></tr><tr><td>AlphaEdit (Fang et al., 2025)</td><td>0.72 ±.023</td><td>0.77 ±.025</td><td>0.74 ±.015</td><td>0.74 ±.012</td><td>0.62 ± .026</td><td>0.60 ±.028</td><td>0.71 ±.017</td><td>0.64 ±.014</td></tr><tr><td>MEMOIR (Wang et al., 2025)</td><td>0.72 ±.015</td><td>0.75 ±.019</td><td>0.83 ±.012</td><td>0.77 ±.009</td><td>0.66 ±.017</td><td>0.68 ±.022</td><td>0.73 ±.013</td><td>0.69 ±.010</td></tr><tr><td>ELDER (Li et al., 2024a)</td><td>0.54 ±.027</td><td>0.49 ±.031</td><td>0.88 ±.017</td><td>0.64 ±.015</td><td>0.41 ±.030</td><td>0.36 ±.035</td><td>0.82 ±.019</td><td>0.53 ±.017</td></tr><tr><td>MELO (Yu et al., 2023)</td><td>0.58 ±.023</td><td>0.53 ±.027</td><td>0.91 ±.015</td><td>0.67 ±.013</td><td>0.45 ±.026</td><td>0.40 ±.030</td><td>0.85 ±.017</td><td>0.57 ±.014</td></tr><tr><td>AdaLoRA (Zhang et al., 2023)</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.001</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.000</td><td>0.00 ±.001</td></tr><tr><td>LadderEdit (Ours)</td><td>| 0.77 ±.008</td><td>0.87 ±.010</td><td>0.99 ±.002</td><td></td><td>0.88 ±.004 | 0.75 ±.009</td><td>0.84 ±.011</td><td>0.98 ±.004</td><td>0.86 ±.005</td></tr></table>

Table 5: Budget-matched strict-memory frontier at $N = 5 1 2$ . LadderEdit preserves coverage and allocates residual fidelity through behavioral rung selection.

Table 4: Population structure of edit streams at $N =$ 2,000. Categories are determined by the behavioral audit on audit-split probes. Most edits are over-resolved by the full LoRA update.
<table><tr><td>Diagnostic case</td><td>ZsRE</td><td>CF</td></tr><tr><td>Sketch-sufficient (rank-1 passes)</td><td>81.2%</td><td>76.4%</td></tr><tr><td>Detail-needed (rank ≥ 2 required)</td><td>13.4%</td><td>16.8%</td></tr><tr><td>Residual-needed (rank Rmax required)</td><td>3.1%</td><td>4.7%</td></tr><tr><td>Compression-regularized (sketch &gt; exact)</td><td>4.3%</td><td>6.7%</td></tr></table>

<table><tr><td>Dataset</td><td>Policy</td><td>Memory</td><td>Comp.</td><td>Utility</td></tr><tr><td>CF</td><td>exact-cache (matched)</td><td>0.180</td><td>5.57</td><td>0.434</td></tr><tr><td>CF</td><td>rank-1 sketch only</td><td>0.125</td><td>8.00</td><td>0.753</td></tr><tr><td>CF</td><td>static mixed-rank</td><td>0.180</td><td>5.57</td><td>0.815</td></tr><tr><td>CF</td><td>dual-budget LadderEdit</td><td>0.180</td><td>5.57</td><td>0.862</td></tr><tr><td>ZsRE</td><td>exact-cache (matched)</td><td>0.178</td><td>5.63</td><td>0.420</td></tr><tr><td>ZsRE</td><td>rank-1 sketch only</td><td>0.124</td><td>8.09</td><td>0.765</td></tr><tr><td>ZsRE</td><td>static mixed-rank</td><td>0.179</td><td>5.58</td><td>0.796</td></tr><tr><td>ZsRE</td><td>dual-budget LadderEdit</td><td>0.179</td><td>5.58</td><td>0.866</td></tr></table>

![](images/12dcdb228c34041c6a5eabb2a5f6ec55b6b4eca5f5e2e59b52ede96d9a331ea9.jpg)  
(a) ZsRE

![](images/5592e49e89ffb22fda7d35fc196b7e1454edba986ddd8899ebcafca13b452755.jpg)  
(b) CF  
Figure 4: Spectral-filter frontier analysis on ZsRE and CF at $N = 5 1 2$ edits. The plots show the tradeoff between memory compression and behavioral contract utility for different rank tiers and allocation strategies.

Finally, Table 6 shows that this frontier translates into deployed storage savings. LadderEdit is exact-first, so it does not reduce the initial LoRA acquisition cost; its benefit is retained edit memory after acquisition. At $T = 1 0 , 0 0 0$ on LLaMA-3-8B, the audit increases per-edit insertion time relative to exact LoRA, but persistent memory drops from 100.93 GB to 19.44 GB, a 5.19× reduction. Because exact LoRA and LadderEdit use the same edit-selection assumption, this is a storage-layer gain: the lookup mechanism is unchanged, and LadderEdit changes only the representation returned after lookup. Additional qualitative examples in Appendix D.6 and stress tests in Appendix D.7 show the same mechanism at the prompt and robustness levels: the edit stream is heterogeneous, many edits are sketch-sufficient, and a smaller subset needs higher rank to satisfy the measured contract.

Table 6: Cost decomposition at $T = 1 0 , 0 0 0$ on LLaMA-3-8B. LadderEdit reduces this storage by 5.19× relative to exact LoRA. Serving path summarizes the additional edit mechanism used at inference.
<table><tr><td>Method</td><td>Time (ms)</td><td>Peak Mem. (GB)</td><td>Persistent (GB)</td><td>Serving path</td></tr><tr><td>LoRA (Exact) (Hu et al., 2022)</td><td>86</td><td>18.4</td><td>100.93</td><td>exact-adapter lookup</td></tr><tr><td>MELO (Yu et al., 2023)</td><td>91</td><td>18.6</td><td>100.93</td><td>retrieval</td></tr><tr><td>ELDER (Li et al., 2024a)</td><td>95</td><td>19.1</td><td>0.08</td><td>MoE routing</td></tr><tr><td>MEMOIR (Wang et al., 2025)</td><td>104</td><td>17.9</td><td>0.14</td><td>mask gate</td></tr><tr><td>LadderEdit</td><td>124</td><td>18.8</td><td>19.44</td><td>ladder lookup</td></tr></table>

## 5 Conclusion

We introduced LadderEdit, a residual-ladder storage controller for lifelong editing. LadderEdit first uses a spectral predictor to select a candidate rank for each acquired edit, and a behavioral audit verifies the proposal and promotes the edit along the ladder only when the performance of that edit demands it. Across ZsRE, CF, and WikiBigEdit on LLaMA-3-8B, Mistral-7B, and Qwen2.5-7B, LadderEdit tracks exact LoRA while reducing persistent memory by 5.2× and remaining effective at 50,000 edits. These results show that most exact adapters are over-resolved: low-rank sketches preserve the edit contract for the bulk of the stream, while higher resolution is needed only for a smaller hard edits subset. The key question in lifelong edit storage is not only which edits to keep, but also how much of each edit should be retained.

## 6 Limitations

LadderEdit is not constant-memory: every edit retains at least a rank-1 sketch, so persistent storage grows linearly with the stream, and future work could explore sketch downgrading or merging to bound the total. The behavioral audit adds insertion-time latency (124 ms versus 86 ms at T=10,000 on LLaMA-3-8B), which could be reduced by batching audits or running them asynchronously. The audit also depends on probe coverage and does not guarantee correctness outside the probed set, motivating richer probe construction or learned probe selection. LadderEdit is a storagelayer method and inherits the retrieval mechanism used by the full-resolution memory bank; integration with broader production retrieval stacks remains open. The spectral score that proposes candidate ranks is a calibrated heuristic, and replacing it with a predictor trained on audit outcomes is a natural extension. Finally, our experiments span three benchmarks and three 7B and 8B backbones, with long-horizon scaling shown to 50,000 edits; validating LadderEdit at larger scales and with non-LoRA edit writers remains future work.

Potential Risks. LadderEdit reduces the persistent cost of injecting and retaining edits in a large language model, which could lower the operational barrier to large-scale knowledge modification. We see three concrete risks. First, the same machinery that enables efficient correction of outdated or harmful facts can be used to insert false or adversarial edits at scale, since the audit verifies behavioral contract satisfaction rather than factual correctness of the target. Mitigations include provenance tracking on the edit stream, signed edit acquisition, and downstream factuality checking, none of which are part of the method itself. Second, the behavioral audit relies on a probe bank whose coverage is finite; edits whose effects fall outside the probed semantic neighborhood may pass the audit while exhibiting unintended behavior on unprobed prompts. This is a soft-coverage rather than a hard-correctness guarantee, and we emphasize in the appendix that the audit should not be treated as a safety certificate. Third, the residual-ladder representation makes individual edits less inspectable than uncompressed adapters: a low-rank sketch can be behaviorally equivalent to an exact LoRA update on the probed contract while retaining different parameter directions. In safety-critical deployments, this argues for keeping edit-level audit logs and periodic reaudits against the exact update rather than relying on stored representations alone. We do not release any new model weights or edit corpora and rely on public benchmarks; the method’s broader societal effect depends on the editing workflow it is deployed in rather than on the storage primitive itself.

## 7 Ethics Considerations

Model editing can be used to correct errors and update knowledge, but it can also be misused to insert false or harmful facts. Edits require provenance, authorization, validation, and auditing to ensure responsible deployment. LadderEdit is a memory allocation mechanism for model editing; it does not serve as a truth or safety filter.

## Acknowledgment

This study was supported in part by NIH R21EB034911, Gemini Academic Program Award, and NVIDIA Academic Grant Program.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics.

Nicola De Cao, Wilker Aziz, and Ivan Titov. 2021. Editing factual knowledge in language models. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetun-

ing of quantized LLMs. In Advances in Neural Information Processing Systems.

Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, and Tat-Seng Chua. 2025. AlphaEdit: Null-space constrained knowledge editing for language models. In International Conference on Learning Representations.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi. 2023. Aging with GRACE: Lifelong model editing with discrete key-value adaptors. In Advances in Neural Information Processing Systems.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, and 1 others. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, and 1 others. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. 2017. Zero-shot relation extraction via reading comprehension. In Proceedings ofthe 21st Conference on Computational Natural Language Learning (CoNLL 2017), pages 333–342.

Jiaang Li, Quan Wang, Zhongnan Wang, Yongdong Zhang, and Zhendong Mao. 2024a. ELDER: Enhancing lifelong model editing with mixture-of-LoRA. arXiv preprint arXiv:2408.11869.

Qi Li, Xiang Liu, Zhenheng Tang, Peijie Dong, Zeyu Li, Xinglin Pan, and Xiaowen Chu. 2024b. Should we really edit language models? on the evaluation of edited language models. Advances in Neural Information Processing Systems.

Vladislav Lialin, Sherin Muckatira, Namrata Shivagunde, and Anna Rumshisky. 2024. ReLoRA: Highrank training through low-rank updates. In International Conference on Learning Representations.

Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024. DoRA: Weightdecomposed low-rank adaptation. arXiv preprint arXiv:2402.09353.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. 2023. Massediting memory in a transformer. In International Conference on Learning Representations.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. 2022a. Fast model editing at scale. In International Conference on Learning Representations.

Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. 2022b. Memorybased model editing at scale. In International Conference on Machine Learning.

Domenic Rosati, Robie Gonzales, Jinkun Chen, Xuemin Yu, Yahya Kayani, Frank Rudzicz, and Hassan Sajjad. 2024. Long-form evaluation of model editing. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics.

Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, and 1 others. 2023. S-lora: Serving thousands of concurrent lora adapters. arXiv preprint arXiv:2311.03285.

Anton Sinitsin, Vsevolod Plokhotnyuk, Dmitriy Pyrkin, Sergei Popov, and Artem Babenko. 2020. Editable neural networks. In International Conference on Learning Representations.

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. Mpnet: Masked and permuted pretraining for language understanding. Advances in neural information processing systems, 33:16857– 16867.

Lukas Thede, Karsten Roth, Matthias Bethge, Zeynep Akata, and Tom Hartvigsen. 2025. Wikibigedit: Understanding the limits of lifelong knowledge editing in llms. arXiv preprint arXiv:2503.05683.

Ke Wang, Yiming Qin, Nikolaos Dimitriadis, Alessandro Favero, and Pascal Frossard. 2025. MEMOIR: Lifelong model editing with minimal overwrite and informed retention for LLMs. In Advances in Neural Information Processing Systems.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533.

Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. 2024a. WISE: Rethinking the knowledge memory for lifelong model editing of large language models. arXiv preprint arXiv:2405.14768.

Peng Wang, Ningyu Zhang, Bozhong Tian, Zekun Xi, Yunzhi Yao, Ziwen Xu, Mengru Wang, Shengyu Mao, Xiaohan Wang, Siyuan Cheng, and 1 others. 2023. EasyEdit: An easy-to-use knowledge editing framework for large language models. arXiv preprint arXiv:2308.07269.

Renzhi Wang and Piji Li. 2024a. LEMoE: Advanced mixture of experts adaptor for lifelong model editing of large language models. arXiv preprint arXiv:2406.20030.

Renzhi Wang and Piji Li. 2024b. MEMoE: Enhancing model editing with mixture of experts adaptors. arXiv preprint arXiv:2405.19086.

Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, and Jundong Li. 2024b. Knowledge editing for large language models: A survey. arXiv preprint arXiv:2310.16218.

Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan, Thuy-Trang Vu, and Gholamreza Haffari. 2024. Continual learning for large language models: A survey. arXiv preprint arXiv:2402.01364.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, and 1 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023. Editing large language models: Problems, methods, and opportunities. arXiv preprint arXiv:2305.13172.

Lang Yu, Qin Chen, Jie Zhou, and Liang He. 2023. MELO: Enhancing model editing with neuron-indexed dynamic LoRA. arXiv preprint arXiv:2312.11795.

Xiaobing Yu, Jin Yang, Xiao Wu, Peijie Qiu, and Xiaofeng Liu. 2025. Fm-lora: Factorized low-rank meta-prompting for continual learning. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 6399– 6408. IEEE.

Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, and 1 others. 2024. A comprehensive study of knowledge editing for large language models. arXiv preprint arXiv:2401.01286.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng,

Weizhu Chen, and Tuo Zhao. 2023. Adalora: Adaptive budget allocation for parameter-efficient finetuning. arXiv preprint arXiv:2303.10512.

Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. 2023. Can we edit factual knowledge by in-context learning? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing.

Junhao Zheng, Shengjie Qiu, Chengming Shi, and Qianli Ma. 2024. Towards lifelong learning of large language models: A survey. arXiv preprint arXiv:2406.06391.

## Appendix

A Composing LadderEdit with Learned   
Retrieval 12   
A.1 Edit-Stream Heterogeneity 12   
A.2 Comparison with Static Mixed-  
Rank Allocation . 13   
A.3 Audit Score Weights 13   
A.4 Composition with Post-Training   
Quantization . 14   
A.5 Per-Condition Rung Allocation 14   
A.6 Memory Ratio Stability . 15   
B Additional Rank-Compression Diagnos  
tics 15   
B.1 Training and Seed Details . 15   
C Probe Split Protocol and Validation 16   
D Why Not Standard LoRA Compression? 16   
D.1 Partition statistics 16   
D.2 Audit-test independence 16   
D.3 Leaky versus clean comparison . 17   
D.4 AdaLoRA Comparison and Scope 17   
D.5 Rationale and Conditional Guaran  
tees 17   
D.6 Diagnostic Examples 18   
D.7 Sensitivity, Coverage, and Audit Cost 18   
E Rank-Predictor Ablation 19   
E.1 Denoising Audit and Spectral-  
Filter Visual Audit 19   
F Fixed-Rank Tier Diagnostics 20   
F.1 Swap-Count and Frontier Visuals . 21   
F.2 Interpretation 21   
F.3 Baseline Details 21   
G Extended Analysis of Scalability, Stor  
age, and Practical Deployment 22   
G.1 Scaling to Larger Backbones 22   
G.2 Exact-Cache Baselines and Cover  
age Loss 23   
G.3 Strict Memory-Matched Frontier 23   
G.4 Full-Resolution Acquisition Cost 23   
G.5 Long-Horizon Storage Growth 24   
G.6 Retrieval and Edit Selection . 24   
G.7 Rank Proposal and Audit Efficiency 25   
G.8 Audit-Probe Coverage 25   
G.9 Qualitative Audit Outcomes 26   
G.10 Correlation Between Compression   
and Behavioral Outcomes . 26   
G.11 SVD Procedure and Memory Ac  
counting . . 26   
G.12 Scope of the Efficiency Claim 27   
G.13 Summary 27

## A Composing LadderEdit with Learned Retrieval

The main experiments use matched edit association to isolate the storage layer from the retrieval layer, so that any performance gap among storage policies is attributable to the persistent representation rather than to retrieval quality. This subsection examines the orthogonal question of whether the storage savings reported in the main paper survive when the matched association is replaced with a learned retriever.

We evaluate three retrieval strategies on LLaMA-3-8B / ZsRE at $T = 1 { , } 0 0 0 \colon$ a learned Contrieverstyle dual encoder trained on edit prompts, a learned E5-large retriever, and a frozen off-theshelf MPNet embedding (no retrieval training). In all cases, the retriever selects a single edit-local memory, and the same retrieved edit identity is passed to both exact LoRA and LadderEdit, so the comparison still measures storage rather than retrieval quality.

Table 7 reports sensitivity to retrieval on ZsRE / LLaMA-3-8B at $T = 1 { , } 0 0 0$

<table><tr><td>Setting</td><td>Rel.</td><td>Gen.</td><td>Loc.</td><td> $\operatorname { A v g } .$ </td><td> $\Delta { \ v } \mathrm { s } .$  matched</td></tr><tr><td>Matched association (Exact LoRA)</td><td>0.95</td><td>0.91</td><td>1.00</td><td>0.95</td><td></td></tr><tr><td>Matched association (LadderEdit)</td><td>0.95</td><td>0.91</td><td>1.00</td><td>0.95</td><td></td></tr><tr><td>Contriever retriever (Exact LoRA)</td><td>0.91</td><td>0.86</td><td>0.99</td><td>0.92</td><td>-0.03</td></tr><tr><td>Contriever retriever (LadderEdit)</td><td>0.90</td><td>0.86</td><td>0.99</td><td>0.92</td><td>-0.03</td></tr><tr><td>E5-large retriever (LadderEdit)</td><td>0.88</td><td>0.85</td><td>0.99</td><td>0.91</td><td>-0.04</td></tr><tr><td>MPNet (frozen) (LadderEdit)</td><td>0.85</td><td>0.81</td><td>0.98</td><td>0.88</td><td>-0.07</td></tr><tr><td>Persistent memory ratio</td><td></td><td></td><td></td><td></td><td>5.2× preserved</td></tr></table>

Table 7: Retrieval sensitivity on ZsRE / LLaMA-3-8B at $T = 1 { , } 0 0 0$ . The persistent-memory ratio is preserved under all retrieval settings in the table because the retriever is shared between exact LoRA and LadderEdit. Locality is barely affected $( 0 . 9 9  0 . 9 8 $ under MPNet) because retrieved adapters still land on real edit directions rather than random directions.

Both methods degrade by an equal amount under imperfect retrieval, because the association layer is shared between exact LoRA and LadderEdit. The relative ordering of storage methods is therefore preserved under all retrieval strategies in Table 7.

## A.1 Edit-Stream Heterogeneity

Table 4 of the main text reports the overall population structure of the edit stream. Within the compression-regularized population, we find a positive correlation (Pearson $r ~ = ~ 0 . 3 8$ on ZsRE, $r = 0 . 4 1$ on CF) between the lexical overlap of the rewrite prompt and locality probes and the magnitude of the locality improvement obtained by sketch storage. This is consistent with lowrank truncation removing direction components that overfit to surface features of the rewrite prompt, which the exact update carries but the locality probes penalize. Section A.5 reports the corresponding ladder allocation decisions.

## A.2 Comparison with Static Mixed-Rank Allocation

The exact-cache baseline in Section 4 isolates the keep-versus-drop extreme of fixed-allocation policies. A second non-adaptive baseline, closer to LadderEdit in absolute memory and not dropping edits, assigns every edit a fixed rank with no behavioral audit. The audit’s contribution is most visible on the worst-edit Avg, the per-edit Avg averaged over the bottom 10% of edits—which the pooled Avg averages away. Against the strongest fixedrank baseline (rank-8), audited LadderEdit gains +0.05 on pooled Avg but +0.12 on the worst-edit metric. An oracle controller that selects the per-edit optimal rank reaches 0.96 pooled and 0.78 worstedit, indicating LadderEdit is within 0.01 pooled and 0.04 worst-edit of the oracle. The audit’s main contribution is therefore on the hard tail of edits where fixed-rank baselines fail systematically.

<table><tr><td>Policy</td><td>Memory (norm.)</td><td>Pooled Avg.</td><td>Worst-edit Avg.</td></tr><tr><td>Fixed rank-1</td><td>0.125</td><td>0.85 (0.008)</td><td></td></tr><tr><td>Fixed rank-2</td><td>0.250</td><td>0.89 (0.007)</td><td></td></tr><tr><td>Fixed rank-4</td><td>0.500</td><td>0.90 (0.008)</td><td></td></tr><tr><td>Fixed rank-8</td><td>1.000</td><td>0.90 (0.007)</td><td>0.62</td></tr><tr><td>Uniform random rank</td><td>0.562</td><td>0.87 (0.010)</td><td></td></tr><tr><td>LadderEdit (audited)</td><td>0.19</td><td>0.95 (0.005)</td><td>0.74</td></tr><tr><td>Oracle rank controller</td><td></td><td>0.96</td><td>0.78</td></tr></table>

Table 8: Audit contribution against fixed-rank storage at $T = 2 { , } 0 0 0$ on ZsRE / LLaMA-3-8B across three seeds. Memory is normalized to exact LoRA. Pooled Avg. is the pooled evaluation Avg. across edits; worstedit Avg. is the per-edit Avg. averaged over only the bottom 10% of edits. Standard deviations over three seeds are reported in parentheses when available.

Trade-offs with the strongest baselines. GRACE achieves 1.00 reliability and locality on ZsRE by codebook retrieval, but its generalization (0.34 on LLaMA-3 at $T = 2 { , } 0 0 0 )$ means it cannot answer paraphrased queries. MEMOIR achieves 0.69 Avg. at $T \ = \ 1 0 { , } 0 0 0$ in only 0.14 GB of persistent memory, a 138× memory advantage over LadderEdit (19.4 GB). LadderEdit’s 0.11 Avg. improvement over MEMOIR at $T = 1 0 \small { , } 0 0 0$ is therefore an explicit memory-for-fidelity trade, not a strict dominance. Practitioners with extreme memory constraints should prefer MEMOIR; practitioners with extreme paraphrase-coverage requirements should consider GRACE; LadderEdit is recommended for the regime where retrieval-quality edits at scale are the bottleneck.

## A.3 Audit Score Weights

The audit score in Eq. 7 combines reliability, generalization, and locality residuals as a weighted sum, $s ( z ) = w _ { R } \cdot \Delta _ { R } + w _ { G } \cdot \Delta _ { G } + w _ { L } \cdot \Delta _ { L } ,$ , with default weights $w _ { R } = 1 . 0 , w _ { G } = 0 . 8 .$ , and $w _ { L } = 1 . 2$ . The asymmetry between w<sub>L</sub> and the other two weights is deliberate. A reliability or generalization failure on a candidate rung triggers rank promotion: the audit rejects the sketch, the controller moves up the ladder, and the contract is reassessed at the next rung. The failure is recoverable within the same insertion. A locality failure, by contrast, alters behavior on prompts the edit was never supposed to touch. If the audit lets such a sketch pass, the corruption is committed to persistent memory and propagates to every later query that lands on that edit; there is no within-insertion repair. We therefore weight locality higher than the recoverable metrics, treating $w _ { L } > w _ { R }$ , w<sub>G</sub> as a soft asymmetry between catastrophic and recoverable failure modes.

The default $w _ { L } = 1 . 2$ is selected by a sweep on the validation split. We evaluate $w _ { L } \in$ {0.8, 1.0, 1.2, 1.5, 2.0} holding $w _ { R }$ and $w _ { G }$ fixed; Figure 5 reports the result on $\mathrm { Z s R E / L L a M A { - } { 3 - } { 8 } B }$ at $T = 1 { , } 0 0 0$ . The choice is not knife-edge: average performance stays within 0.01 of the peak for $w _ { L } \in [ 1 . 0 , 1 . 5 ]$ , and drops by 0.02 at $w _ { L } = 0 . 8$ and 0.03 at $w _ { L } = 2 . 0 .$ The default sits at the midpoint of this stable region.

The per-metric decomposition (Figure 5, middle) clarifies why the stable region is bounded on both sides. As $w _ { L }$ falls below 1.0, locality begins to leak: the audit no longer penalizes locality violations strongly enough to reject sketches that overwrite neighboring knowledge, and the Loc. metric drops sharply (from 1.00 at the default to 0.88 at $w _ { L } = 0 . 8 )$ . As $w _ { L }$ rises above 1.5, the audit becomes over-restrictive: many sketches that would have preserved locality adequately are rejected for marginal locality residuals, the controller promotes them to higher ranks or beyond the ladder, and reliability and generalization both degrade as a result. The default $w _ { L } = 1 . 2$ is the unique value in the swept range at which all three metrics are simultaneously near their per-metric optima.

![](images/5fb546a9fbcdf979ec594638c44e20d12406caef0193d780b2a868f071f5edc5.jpg)  
Figure 5: Sensitivity of LadderEdit to the locality weight w<sub>L</sub> on ZsRE / LLaMA-3-8B at $T = 1 { , } 0 0 0$ , with $w _ { R }$ and $w _ { G }$ held fixed. Left: average performance, peaking at the default $w _ { L } = 1 . 2$ with a stable region across [1.0, 1.5]. Middle: per-metric decomposition, showing that locality cliffs upward between $w _ { L } = 0 . 8$ and $w _ { L } = 1 . 0$ while reliability and generalization degrade for $w _ { L } \geq 1 . 5$ . Right: failure-mode decomposition, with locality failures dominating at low $w _ { L }$ and reliability or generalization rejections dominating at high $w _ { L }$ ; the default minimizes the total failure rate.

The failure-mode decomposition (Figure 5, right) reports the same trade-off in absolute terms. At $w _ { L } = 0 . 8$ , locality failures account for 14.2% of edits: a non-trivial fraction of the stream commits a sketch that should have been rejected, and the resulting corruption cannot be undone. At $w _ { L } = 2 . 0$ , locality failures drop to 0.4% but the audit rejects 9.5% of edits that the more permissive default would have accepted; these rejections are recoverable in principle (the controller can re-audit at a different operating point), but in our deployed pipeline they translate to higher rank consumption and reduced per-edit efficiency. The default minimizes the total failure rate at 1.6%, balancing the two error modes against the asymmetry the prose above describes.

We also verified that $w _ { R }$ and $w _ { G }$ are not separately tuned. Setting $w _ { R } = w _ { G }$ over the same five-point sweep and varying their common value in $\{ 0 . 6 , 0 . 8 , 1 . 0 \}$ changes Avg. by less than 0.005, and we therefore fix $( w _ { R } , w _ { G } ) = ( 1 . 0 , 0 . 8 )$ to follow the standard LoRA loss-weighting ratio. The locality weight is the only audit hyperparameter with a non-trivial effect on the headline metrics; all main-paper results use the default $w _ { L } = 1 . 2$ without per-benchmark retuning.

## A.4 Composition with Post-Training Quantization

LadderEdit reduces the number of stored parameters per edit, whereas quantization reduces the bitwidth of each stored parameter. The two reductions act on orthogonal axes of the persistent-storage cost and compose multiplicatively in principle. We verify this empirically by applying symmetric pertensor INT8 quantization to the stored LoRA factors, both for the exact-LoRA reference and for the LadderEdit rungs.

<table><tr><td>Policy</td><td>Memory (norm.)</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Exact LoRA (FP16)</td><td>1.00</td><td>0.94</td></tr><tr><td>Exact LoRA (INT8)</td><td>0.25</td><td>0.93</td></tr><tr><td>LadderEdit (FP16)</td><td>0.19</td><td>0.91</td></tr><tr><td>LadderEdit (INT8)</td><td>0.05</td><td>0.90</td></tr></table>

Table 9: Composition of LadderEdit with INT8 quantization at $T = 2 { \mathrm { , 0 0 0 } } \mathrm { o n }$ ZsRE / LLaMA-3-8B. The two compression axes are orthogonal: INT8 reduces peredit bit-width while LadderEdit reduces per-edit rank. Composing them yields a 20× reduction over FP16 exact LoRA. The composed configuration (LadderEdit + INT8) reaches 0.90 Avg. at 0.05 normalized memory, dominating exact $\mathrm { L o R A + I N T 8 }$ alone (0.93 Avg. at 0.25 memory) on the cost axis.

Quantization is a complementary compression primitive that operates within the cost of one stored representation, while LadderEdit operates across representations by deciding how much rank each edit retains. We therefore treat quantization as an orthogonal post-processing step rather than as a competing baseline; the right deployment configuration uses both.

## A.5 Per-Condition Rung Allocation

The default audit thresholds in the main paper produce the following distribution of stored ranks across the rank menu $\mathcal { R } = \{ 1 , 2 , . . . , 8 \}$ , where $R _ { \mathrm { m a x } } = 8$ matches the exact LoRA acquisition rank. Table 10 reports the distribution at $T = 1 { , } 0 0 0$ on ZsRE / LLaMA-3-8B; comparable distributions are observed for the other backbones and the other horizons.

The stored rank distribution reflects two stages of decision-making. Each edit first receives a predicted rank $\hat { r } _ { i }$ from the spectral proposal $( \operatorname { E q . 5 } ) { \mathrm { : } }$ LadderEdit computes the normalized spectral tail $T _ { i } ( r )$ at every $r \in \mathcal { R }$ from a single SVD, evaluates the score $S _ { i } ( r ) = T _ { i } ( r ) / ( [ \mu _ { i } ^ { E } ] _ { + } + \epsilon )$ , and selects the cheapest $\hat { r } _ { i }$ whose score falls below the calibrated threshold $\tau .$ . This proposal uses only spectral information and the exact-edit margin, so it costs one rank-search per edit and no behavioral evaluation. The behavioral audit then verifies the proposal: if the rank $- \hat { r } _ { i }$ sketch satisfies the contract $( H _ { i } ^ { ( \hat { r } _ { i } ) } = 0 )$ , the edit is stored at ${ \hat { r } } _ { i } ;$ otherwise the audit promotes the edit along the spectrallyordered ladder $\hat { r } _ { i } + 1 , \hat { r } _ { i } + 2 , . .$ . until a passing rung is reached. Table 10 reports the distribution of stored ranks $^ { r _ { i } , }$ which equals $\hat { r } _ { i }$ for the majority of edits whose proposal passes on first audit and exceeds $\boldsymbol { { \hat { r } } _ { i } }$ only for the minority that require promotion.

<table><tr><td>Stored representation</td><td>Fraction of edits</td></tr><tr><td>Rank-1 sketch</td><td>79.4%</td></tr><tr><td>Rank-2 sketch</td><td>16.1%</td></tr><tr><td>Rank-3 sketch</td><td>1.8%</td></tr><tr><td>Rank-4 sketch</td><td>1.2%</td></tr><tr><td>Rank-5 sketch</td><td>0.7%</td></tr><tr><td>Rank-6 sketch</td><td>0.4%</td></tr><tr><td>Rank-7 sketch</td><td>0.3%</td></tr><tr><td>Rank-8 sketch</td><td>0.1%</td></tr></table>

Table 10: Distribution of stored representations (N=1,000 edits at $T = 1 { , } 0 0 0$ on LLaMA-3-8B / ZsRE; sums to 100.0%) across the full r=1 to r=8 ladder. The majority of edits are served by the cheapest rung, and the distribution decays sharply: the top two rungs together cover 95.5% of edits, while only the remaining ∼ 4.5% requires ranks 3 or above. The audit promotes edits to higher rungs only when the low-rank sketch fails to satisfy the behavioral contract, which corresponds approximately to the detail-needed population of Table 4. The parity between LadderEdit and exact LoRA at this horizon (Table 1 of the main text) therefore reflects targeted rank promotion on a small minority of edits rather than uniformly high-rank storage across the stream.

The per-edit allocation decisions are released alongside the code to support reproducibility. The audit cost reported in Appendix D.7 is dominated by the rank-1 evaluation path because most edits terminate there; only the ∼ 20% of edits that are promoted incur the higher-rank audit cost.

## A.6 Memory Ratio Stability

The 5.2× memory ratio in the abstract and main text refers to the (LLaMA-3-8B, $T = 1 0 , 0 0 0 )$ configuration on WikiBigEdit. We report ratios at every backbone-horizon pair to document the stability of this number. At $T = 1 0 , 0 0 0$ the ratio is $5 . 2 \times$ on LLaMA-3-8B, 5.0× on Mistral-7B, and 5.1× on Qwen2.5-7B (standard deviation 0.1×). The ratio is not constant in $T \colon$ it grows from 2.0× at $T = 1 0 0$ (mean across backbones) to 11.8× at

T = 50,000, because LadderEdit’s allocation table has a long tail of low-rank rungs that absorb additional edits with near-constant marginal memory, while exact LoRA stores every additional edit at full rank.

The 5.2× headline at $T = 1 0 , 0 0 0$ grows to approximately 12× at $T = 5 0 { , } 0 0 0$ on WikiBigEdit for the same reason. Cross-backbone stability at $T = 5 0 { , } 0 0 0$ is similar: 12.0× on LLaMA-3-8B, 11.6× on Mistral-7B, and 11.8× on Qwen2.5-7B.

## B Additional Rank-Compression Diagnostics

This appendix provides further evidence and explanation for the compression and behavioral audit results discussed in the main text, focusing on the reliability of the method, the role of the audit, and the interpretation of diagnostic categories.

## B.1 Training and Seed Details

All experiments use frozen backbone models; the only learned object for an incoming edit is its LoRA-style edit adapter. We first acquire an exact edit $\Delta _ { i } ^ { E }$ with the same edit-writing procedure used by the exact LoRA baseline. LadderEdit is applied after this acquisition step: it does not change the edit writer, the base model weights, the rewrite examples, or the optimizer used to obtain the exact edit. This keeps the comparison focused on persistent storage and serving-time representation rather than on a stronger or weaker edit writer.

For exact LoRA acquisition, each edit is represented by a pair of LoRA factors whose effective update is $\Delta _ { i } ^ { \bar { E } }$ . The rank-compression diagnostics use an exact raw LoRA rank of 8, corresponding to 5.05M trainable parameters per exact edit in the LLaMA-3-8B diagnostic accounting. The deployed rank menu is $\mathcal { R } = \{ 1 , 2 , . . . , 8 \}$ , matching the exact LoRA acquisition rank. Candidate sketches at each rung are produced by a single truncated SVD of the effective update, which yields all |R| spectral tails $T _ { i } ( r )$ in one pass. The strict frontier diagnostics in Table 5 and Appendix Table 20 report the sparse representative subset {1, 2, 4} for readability; the audited deployment uses the full menu. The default audited configuration in the main experiments uses $\tau = 0 . 0 5$ unless a table explicitly varies this value.

The behavioral audit is run at insertion or rebalancing time. It evaluates rewrite, generalization, and locality probes, then stores the cheapest rung whose measured contract is acceptable. If the sketch fails, LadderEdit promotes the edit to a higher rank along the ladder until the contract is met (or $R _ { \mathrm { m a x } }$ is reached). Probe thresholds, audit weights, and the compressibility-score threshold are selected on held-out validation probes for each benchmark and then held fixed for the corresponding test streams.

We use fixed edit streams and shared random seeds across compared methods so that differences reflect the editing policy rather than the sampled order. The rank-compression package reports averages over multiple seed runs whenever those runs are available. Fixed-rank spectral and denoising diagnostics are averaged over seeds 0, 1, and 2 for CF at N = 32, 128, 512 and for ZsRE at N = 128, 512; ZsRE at N = 32 includes six available runs. The budget-matched strictmemory frontier in the main text is reported from the N = 512, seed-0 frontier file used for the plotted strict-frontier analysis, with exact-cache rows matched to the same normalized memory budget.

## C Probe Split Protocol and Validation

This appendix verifies that the audit signal used by LadderEdit does not leak into the evaluation reported in the main tables. We describe the partition, report inter-split similarity, and show that running LadderEdit with a leaky (audit-equals-test) configuration would inflate the reported numbers by a measurable margin — which the strict split avoids.

## D Why Not Standard LoRA Compression?

LadderEdit is methodologically close to several LoRA-centric techniques but differs from each along a specific axis. We distinguish three of them here, since reviewers familiar with one of these methods may otherwise read LadderEdit as a reapplication of it.

Unit and criterion of allocation. Adaptive-rank methods such as AdaLoRA allocate rank across the layers or modules of a single adapter using gradientderived importance during training. LadderEdit allocates persistent storage across edits after acquisition, using behavioral contract satisfaction as the criterion. The decision variable is “how much of edit i to retain” rather than “where to spend rank inside one adapter”. Running AdaLoRA independently for each edit does not solve the lifelongediting problem: it still stores one adapter per edit and lacks a stream-level policy for deciding which edits deserve higher resolution.

Stream-level policy. Multi-adapter serving systems such as S-LoRA deploy many independently trained adapters efficiently at inference time but do not decide which adapter directions to persist. LadderEdit operates one level higher: as the stream of edits accumulates, the residual-ladder controller spends extra memory only on the behaviorally difficult tail and leaves easy edits at low rank. Without this stream-level policy, multi-adapter serving still stores every adapter at its acquired rank, and the persistent memory grows linearly at the slope set by the edit writer rather than the slope set by the audit.

Independence versus merging. Adaptermerging methods reduce per-edit storage by combining adapters into one shared representation, which loses edit-local isolation and reintroduces the plasticity pressure of constant-memory editors. LadderEdit keeps one representation per edit; the residual ladder reduces memory without folding edits together. This is the property that distinguishes LadderEdit from the merging-based lifelong editors (MELO, ELDER, MEMoE, LEMoE) in our baseline set.

## D.1 Partition statistics

For each edit, the benchmark prompts are split as follows:

<table><tr><td>Dataset</td><td>Audit probes/edit</td><td>Val. probes/edit</td><td>Test probes/edit</td></tr><tr><td>ZsRE</td><td></td><td>1</td><td></td></tr><tr><td>CounterFact</td><td></td><td>1</td><td></td></tr><tr><td>WikiBigEdit</td><td>232</td><td>1</td><td>232</td></tr></table>

Table 11: Per-edit probe counts after the three-way split. Locality probes are drawn from disjoint pools of unrelated subjects between audit and test.

The split is performed once per edit at streamconstruction time using a fixed seed, and is shared across all compared methods so that no method sees a different test set.

## D.2 Audit-test independence

Strong support for the split requires that the audit score is not a perfect predictor of the test score. If it were, the audit would amount to memorising the test set. We measure two quantities at N = 2,000 on ZsRE / LLaMA-3-8B.

Audit-test correlation. Pearson correlation between per-edit audit utility and per-edit test utility is $r = 0 . 6 2 ( p < 1 0 ^ { - 5 0 } )$ . The correlation is positive (an edit that passes its audit usually passes its test) but well below 1, indicating that the test contains genuine held-out variation rather than rescored audit prompts.

Lexical overlap. Word-level Jaccard similarity between audit and test paraphrases of the same edit averages 0.34 on ZsRE and 0.29 on CounterFact. As a control, the Jaccard between paraphrases of two random different edits is 0.11 on ZsRE. The within-edit overlap is higher than the between-edit overlap, as expected for paraphrases of the same fact, but is far from the trivial near-1.0 overlap that audit-equals-test would imply.

## D.3 Leaky versus clean comparison

To quantify the importance of the split, we re-run LadderEdit with a deliberately leaky configuration in which the audit uses the test paraphrases directly, and compare against the strict split used in the main paper.

<table><tr><td>Configuration</td><td>Rel.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td></tr><tr><td>LadderEdit, strict split (main)</td><td>0.95</td><td>0.91</td><td>1.00</td><td>0.95</td></tr><tr><td>LadderEdit, leaky (audit=test)</td><td>0.99</td><td>0.97</td><td>1.00</td><td>0.99</td></tr><tr><td>Inflation gap</td><td>+0.04</td><td>+0.06</td><td>0.00</td><td>+0.04</td></tr></table>

Table 12: Leaky-versus-clean comparison at T = 1,000 on ZsRE / LLaMA-3-8B. The leaky configuration inflates Generalization by 6 points and the average by 4 points. The numbers reported in the main paper correspond to the strict split.

The 4-point average gap quantifies how much the strict split costs us in headline numbers. We report the strict-split numbers throughout the main paper. The leaky-vs-clean table is the strongest single piece of evidence: it shows the split is real, that it costs measurable utility, and that the main-paper numbers correspond to the harder configuration.

## D.4 AdaLoRA Comparison and Scope

AdaLoRA is an important low-rank adaptation method, but it solves a different allocation problem from LadderEdit. We do not claim SVD truncation or behavioral audits as new primitives. The contribution is the per-edit decision rule that converts edit difficulty into rank assignment, applied to the sequential editing problem rather than the single fine-tuning problem AdaLoRA targets. AdaLoRA reallocates a fixed parameter budget among LoRA matrices during training for one adaptation objective; LadderEdit allocates rank across a stream of independent edits, with each edit committed only after passing a behavioral audit at the assigned rank.

There are two natural ways to apply AdaLoRA in this setting, and neither is equivalent to LadderEdit. If AdaLoRA is run independently for each edit, it may reduce the rank of each adapter, but the system still stores one adapter per edit and still lacks a stream-level policy for deciding which edits deserve higher rank. If AdaLoRA is run as one shared adapter over many edits, it no longer preserves editlocal isolation; later edits must share capacity with earlier edits, which is the same plasticity pressure faced by constant-memory editors. LadderEdit is designed for the middle regime: one representation per edit, but each representation can occupy a cheaper rung if the behavioral audit permits it.

The main benchmark tables therefore include AdaLoRA as an adaptive-rank LoRA baseline, using the same edit streams and evaluation probes. These runs are useful because they test the closest low-rank alternative: adaptive rank alone does not solve the long-stream memory allocation problem when the system lacks per-edit residual decisions and behavioral auditing. The present diagnostics answer the complementary question: once an exact per-edit LoRA update is acquired, how much of that update must be persistently stored to preserve the edit contract?

## D.5 Rationale and Conditional Guarantees

This section makes the allocation logic explicit. The statements below are not safety guarantees for unprobed prompts. They explain why, under the measured edit-time utility used in the paper, a residual ladder is a more natural memory allocation than an all-or-nothing exact cache.

Coverage versus fidelity. The main text gives the coverage-versus-fidelity decomposition used to interpret the budget-matched frontier. The key point is that exact-cache allocation reduces memory by reducing coverage, while LadderEdit keeps coverage high by assigning every edit a sketch before spending additional rank on fidelity. At roughly 5.6× compression, an exact cache stores only about 18% of the 512 edits, whereas rank-1 sketches give every edit a nonzero representation and higher-rank promotion repairs the hard tail without discarding easy edits.

Stronger drop policies. The exact-cache result in Section 4 uses uniform random retention, which is a weak drop policy. We tested four stronger policies at the same memory budget (91 of 512 edits retained) on ZsRE / LLaMA-3-8B. Uniform random drop reaches 0.18 pooled Avg; FIFO drop reaches 0.20; LRU drop reaches 0.22; learned utility-scored drop (a small MLP scoring edits by retrieval frequency and recency) reaches 0.35. Adding a fallback that routes discarded queries to the unedited base model raises learned drop to 0.41. LadderEdit reaches 0.93 at the same memory budget because no edit is discarded. Evaluating only on retained edits, all drop policies reach 0.93–0.94, matching LadderEdit on that subset. The gap on the full evaluation pool reflects the cost of not preserving the discarded 80%; this is a fundamental limitation of any drop strategy, since information about discarded edits is gone regardless of policy.

Optimal promotion choice for a fixed baseline rank. For a fixed rank $r ,$ let

$$
g _ { i } ( r ) = u _ { i } ( \Delta _ { i } ^ { E } ) - u _ { i } ( \Delta _ { i } ^ { ( r ) } )\tag{10}
$$

be the measured utility gain from promoting edit i from rank r to the exact update. This is the same two-state simplification used in Eq. 8-9; the deployed controller can stop promotion at intermediate rungs. If all promotions have the same extra memory cost $P _ { E } - P _ { r }$ and the budget permits K promotions, the utility-maximizing promoted set is the K edits with the largest positive $g _ { i } ( r )$ . This follows by an exchange argument: if a selected edit has smaller gain than an unselected edit, swapping them increases utility without changing memory. When promotion costs differ across ranks or layers, the same argument becomes a value-density rule.

What the diagnostics establish. The rankcompression package supports three claims needed by the method. First, exact LoRA updates are often over-resolved: rank-2 and rank-4 sketches retain most spectral energy and recover most measured contract behavior. Second, the hard tail is real: rank-1 sketches create false retirements, so a single compressed rank is not enough. Third, the hard tail is small enough for selective rank promotion to be useful: budget-matched ladder policies retain much higher utility than exact-cache policies at comparable memory. These findings justify the ladder design without claiming that spectral compression alone is always safe.

## D.6 Diagnostic Examples

The most useful examples are those that isolate the decision made by the ladder. The goal is not only to show that the average improves, but to show why the controller sometimes stores only a sketch and sometimes promotes to higher rank. Table 13 gives representative edit-level cases from the $N = 5 1 2 .$ seed-0 rank-1 diagnostics. These are index-level examples from the released diagnostic files, not hand-picked text prompts. They cover the four behavioral cases used by the method.

The compact labels in Table 13 are useful for accounting, but the rate-distortion story is clearest at the prompt level. Table 16 shows four representative generations. The first two cases show that rank 1 can preserve the exact rewrite behavior on both ZsRE and CF. The third shows why the ladder must be adaptive: rank 1 captures a coarse entity pattern but rank 2 is needed to recover the specific value. The fourth shows a compression-regularized case where the exact edit misses the target in free generation while the low-rank sketch includes it.

These examples make three points that the aggregate tables can hide. First, many edits are sketchsufficient even when rank 1 retains only about half of the spectral energy, so spectral energy is informative but not identical to behavioral utility. Second, false retirements exist on both CF and ZsRE, which is why LadderEdit keeps the behavioral audit rather than relying on spectrum alone. Third, compression sometimes improves the measured contract, suggesting that some low-energy directions encode over-specific or interfering detail. The method’s rationale is therefore not “low-rank is always enough.” It is that a cheap sketch should cover every edit, while residual memory should be reserved for measured hard cases.

## D.7 Sensitivity, Coverage, and Audit Cost

We evaluate threshold sensitivity, probe coverage, overfitting to the audit set, and insertion-time overhead. Table 14 varies the audit threshold and LoRA rank on ZsRE / LLaMA-3-8B at $T = 1 { , } 0 0 0$ . The default configuration is not a narrow optimum: the average score changes by less than 0.03 across the tested thresholds and ranks, while memory changes substantially with rank. Table 15 stress-tests probebank coverage at $T = 2 { , } 0 0 0$ . The default probe bank covers 87% of edits and keeps the unprobededit average within 0.06 of the probed-edit average. Lower coverage degrades more clearly, which supports the limitation that the audit depends on probe coverage.

Table 13: Representative edit-level diagnostic cases at $N = 5 1 2 ,$ , seed 0, rank 1. ∆U is the measured utility of the rank-1 sketch minus exact LoRA. A positive value means the sketch is better on the audit; a negative value means higher rank is useful.
<table><tr><td>Dataset</td><td>Case</td><td>Edit id</td><td>Energy</td><td>∆U</td><td>Gap</td><td>Interpretation</td></tr><tr><td>CF</td><td>Sketch sufficient</td><td>0</td><td>0.517</td><td>0.000</td><td>0.000</td><td>Rank-1 preserves the exact edit&#x27;s measured behavior, so higher resolution is unnecessary.</td></tr><tr><td>CF</td><td>Detail needed</td><td>9</td><td>0.590</td><td>-0.333</td><td>0.267</td><td>The sketch drops needed behavior; the audit should promote to a higher rung.</td></tr><tr><td>CF</td><td>Compression helps</td><td>14</td><td>0.474</td><td>0.333</td><td>0.000</td><td>Removing tail directions improves measured behavior, so exact LoRA is not always a behavioral ceiling.</td></tr><tr><td>CF</td><td>Exact limited</td><td>4</td><td>0.524</td><td>0.000</td><td>0.267</td><td>Both exact and sketch leave a contract gap; this is an edit-writer limitation, not a compression error.</td></tr><tr><td>ZsRE</td><td>Sketch sufficient</td><td>0</td><td>0.496</td><td>0.000</td><td>0.000</td><td>The cheap sketch is behaviorally indistinguishable from exact on the probes.</td></tr><tr><td>ZsRE</td><td>Detail needed</td><td>3</td><td>0.576</td><td>-0.167</td><td>0.100</td><td>Residual detail is needed for a hard edit that rank-1 cannot safely represent.</td></tr><tr><td>ZsRE</td><td>Compression helps</td><td>189</td><td>0.493</td><td>0.111</td><td>0.000</td><td>The sketch removes behaviorally harmful detail from the exact update.</td></tr><tr><td>ZsRE</td><td>Exact limited</td><td>1</td><td>0.493</td><td>-0.083</td><td>0.033</td><td>The exact edit itself is weak, so promotion to higher rank cannot be treated as a guarantee.</td></tr></table>

Table 14: Sensitivity of LadderEdit to the audit threshold τ and LoRA rank r on ZsRE / LLaMA-3-8B at $T = 1 , 0 0 0 .$ . Avg. is the arithmetic mean of Rel., Gen., and Loc. Bold marks the default configuration. The default $\scriptstyle \left( \tau = 0 . 0 5 , r = 8 \right)$ sits at the peak of both sweeps: smaller τ or r underfits the audit, larger τ or r admits noisy edits that erode locality.
<table><tr><td rowspan="2"></td><td colspan="4">Audit threshold τ with r=8</td><td colspan="4">LoRA rank r with τ=0.05</td></tr><tr><td>τ=0.01</td><td>τ=0.025</td><td>τ=0.05</td><td>τ=0.10</td><td>r=4</td><td>r=8</td><td>r=16</td><td>r=32</td></tr><tr><td>Rel. (↑)</td><td>0.73</td><td>0.86</td><td>0.95</td><td>0.81</td><td>0.78</td><td>0.95</td><td>0.88</td><td>0.74</td></tr><tr><td>Gen. (↑)</td><td>0.82</td><td>0.88</td><td>0.91</td><td>0.85</td><td>0.81</td><td>0.91</td><td>0.89</td><td>0.83</td></tr><tr><td>Loc. (↑)</td><td>0.99</td><td>1.00</td><td>1.00</td><td>0.94</td><td>0.99</td><td>1.00</td><td>0.97</td><td>0.92</td></tr><tr><td>Avg. (↑)</td><td>0.85</td><td>0.91</td><td>0.95</td><td>0.87</td><td>0.86</td><td>0.95</td><td>0.91</td><td>0.83</td></tr><tr><td>Memory (GB)</td><td>4.21</td><td>3.97</td><td>3.84</td><td>3.62</td><td>2.18</td><td>3.84</td><td>7.32</td><td>14.18</td></tr><tr><td>Insertion rate</td><td>1.00</td><td>0.99</td><td>0.98</td><td>0.96</td><td>0.99</td><td>0.98</td><td>0.97</td><td>0.96</td></tr></table>

Table 15: Probe-coverage stress test on ZsRE / LLaMA-3-8B at $T = 2 { , } 0 0 0$ . We vary the audit probe-bank size $| \mathcal { P } |$ and report performance on probe-covered and unprobed edits. Unprobed edits are outside the probe bank’s semantic coverage and are handled through nearest-probe interpolation.
<table><tr><td rowspan="2">Probe bank size |P|</td><td rowspan="2">Coverage</td><td colspan="2">Probed edits</td><td colspan="2">Unprobed edits</td><td rowspan="2">Overall Avg. (↑)</td></tr><tr><td>Avg. (↑)</td><td>Failures (%)</td><td>Avg. (↑)</td><td>Failures (%)</td></tr><tr><td>256</td><td>38%</td><td>0.95</td><td>1.2</td><td>0.78</td><td>14.6</td><td>0.84</td></tr><tr><td>1,024</td><td>64%</td><td>0.95</td><td>0.9</td><td>0.85</td><td>8.3</td><td>0.91</td></tr><tr><td>4,096</td><td>87%</td><td>0.95</td><td>0.8</td><td>0.89</td><td>4.1</td><td>0.94</td></tr><tr><td>16,384</td><td>96%</td><td>0.95</td><td>0.7</td><td>0.91</td><td>2.4</td><td>0.94</td></tr><tr><td>65,536</td><td>99%</td><td>0.95</td><td>0.7</td><td>0.92</td><td>1.8</td><td>0.95</td></tr><tr><td>No audit, LoRA Exact</td><td>0%</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.94</td></tr></table>

Insertion latency on an A100-40GB: exact LoRA goes from 86 ms per edit at $T = 1 0 0$ to 850 ms per edit at $T = 5 0 { , } 0 0 0$ because of per-edit adapter merge overhead, which is linear in the number of stored adapters. LadderEdit goes from 93 ms to 150 ms over the same range; the audit dominates at small T (about 7 ms per edit), and merge growth is suppressed by the low-rank rungs. LadderEdit is slower per edit at $T < 2 , 0 0 0$ and faster at $T >$ 2,000; at $T = 5 0 , 0 0 0$ LadderEdit is 5.7× faster per edit than exact LoRA. The cumulative audit time at $T = 5 0 { , } 0 0 0$ is approximately 32 minutes, paid once at edit acquisition rather than at serving time.

![](images/1fdc35c1399a6025ab3ee46b8e9f57ef9b6b2e44eb94470653394eaf9a9ac664.jpg)  
(a) Held-out probe behavior

![](images/76f059d4a1718d36f7ace5489d808e07caa973b544e97ddd801969027a4aad1f.jpg)  
(b) Insertion-time audit overhead  
Figure 6: Additional audit diagnostics. The held-out probe plot checks whether threshold selection overfits to the validation probe set. The audit-overhead plot decomposes insertion latency and compares LadderEdit with exact LoRA across stream lengths.

## E Rank-Predictor Ablation

The spectral predictor used by LadderEdit combines two factors: the singular-value tail $T _ { i } ( r )$ and the exact-edit behavioral margin $\mu _ { i } ^ { E }$ (Eq. 4). We isolate the contribution of each factor and compare against three reference baselines: a random rank proposal, an audit-and-promote loop with no predictor (always rank-1, then promote), and an exhaustive-audit oracle that audits every rank in R and stores the cheapest passing rung. Results at T = 1,000 on ZsRE / LLaMA-3-8B are reported in Table 19.

Three observations support the predictor design. First, the predictor is necessary: random proposal collapses to 0.81 utility, well below any informed strategy. Second, both factors contribute: removing either the spectral tail or the margin reduces utility by 0.03–0.05 and increases the audits-per-edit cost. Third, the combined predictor approaches oracle quality without paying the oracle’s audit cost, justifying the calibrated heuristic in Eq. 4 as an effective proposal mechanism.

## E.1 Denoising Audit and Spectral-Filter Visual Audit

The denoising audit tests whether compressed (low-rank) edit representations can match or even exceed the behavioral contract of the full exact LoRA update. This is important because exact LoRA is treated as an acquisition reference, not a behavioral oracle: an uncompressed update may contain nuisance directions that hurt locality or overfit to the edit prompt. The audit categorizes edits as:

Table 16: Prompt-level qualitative examples illustrating the main residual-ladder decisions. Outputs are shortened only for formatting. These cases show why LadderEdit allocates storage per edit rather than assigning one fixed rank to the whole stream.
<table><tr><td>Case</td><td>Prompt and target</td><td>Exact LoRA</td><td>Sketch rung</td><td>Ladder decision</td></tr><tr><td>ZsRE 29: sketch suf- ficient</td><td>Rewrite: Which is the position of Jules Basile Onambele? Target: winger. Generalization: What is Jules Basile Onambele&#x27;s position?</td><td>Jules Basile Onambele primarily plays as a winger or attacking midfielder.</td><td>Rank 1 gives the same rewrite and gen- eralization output as exact LoRA.</td><td>Store rank 1. The full residual is unnec- essary.</td></tr><tr><td>CF 11: sketch suffi- cient</td><td>Rewrite: Andreas Ivanschitz pro- fessionally plays the sport. Tar- get: football. Generalization: After work Walther attended the Y.M.C.A. Andreas Ivanschitz, the</td><td>Andreas Ivanschitz is known for play- ing football (soccer) professionally. The generalization output remains incom- plete.</td><td>Rank 1 preserves the same rewrite be- havior and the same incomplete gener- alization response.</td><td>Store rank 1. A hard dataset still con- tains cheap compress- ible edits.</td></tr><tr><td>ZsRE 27: detail needed</td><td>Rewrite: What sports team was Stanislav Romanov a member of? Target: Spartak Myjava. Gener- alization: Which sports team was Stanislav Romanov&#x27;s?</td><td>Stanislav Romanov was a member of FC Spartak Myjava.</td><td>Rank 1 changes the value to FC Spartak Mytishchi, while rank 2 recovers FC Spartak Myjava.</td><td>Promote to rank 2. The edit needs a little more detail, but not exact storage.</td></tr><tr><td>ZsRE 24: compres- sion helps</td><td>Rewrite: What kind of occupation does Karim-Mohamed Maamoun have? Target: architect. General- ization: What kind of occupation is Karim-Mohamed Maamoun?</td><td>Exact LoRA answers archaeologist on the rewrite and generalization prompts.</td><td>Rank 1 and rank 2 include archaeologist and architect in the rewrite output.</td><td>Store the sketch. Low-rank projection can filter harmful detail from the exact edit.</td></tr></table>

Table 17: Caveat example: exact acquisition can fail under free generation. This supports the limitation that exact LoRA is an acquisition reference, not an infallible semantic oracle.
<table><tr><td rowspan=1 colspan=6>Case        Exact LoRA         Sketches           Interpretation</td></tr><tr><td rowspan=1 colspan=6>CF27,tar-  Jean Gaven was the  Rank 1 and rank 2 pro-  Exact-limited.get Russian.  President of the French  duce the same incorrect  The edit writerRewrite prompt:  Senate. The generaliza-  or uninformative behav-  failedbeforeJeanGaven,  tion response asks for  ior.              compression, sospeaker of     more information about                   residual storagethe native language.                     cannot guaranteesuccess.</td></tr><tr><td rowspan=1 colspan=6></td></tr><tr><td rowspan=1 colspan=6>Quantity                         Value</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=4>Crossover horizon             T ≈ 2,000Exact LoRA @ $T = 1 0 0$       86 ms/edit93 ms/edit</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LadderEdit @ T = 100</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Exact LoRA @ $T = 5 0 { , } 0 0 0$ </td><td rowspan=1 colspan=2>850 ms/edit</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>LadderEdit @ T = 50,000 15</td><td rowspan=2 colspan=2>150 ms/editSpeedup @ $T = 5 0 { , } 0 0 0$          5.7×</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td></td><td></td></tr></table>

Table 18: Insertion latency summary (A100-40GB).

• Both-pass: Both the sketch and exact update satisfy the behavioral contract (“sketchsufficient”).

• Detail-needed: The sketch fails but the exact passes (higher rank is required).

• Compression-regularized (extra success):

<table><tr><td>Predictor</td><td>Utility</td><td>Audits/edit</td><td>Memory</td></tr><tr><td>Random rank proposal</td><td>0.81</td><td>2.31</td><td>0.22</td></tr><tr><td>Always rank-1, then promote</td><td>0.89</td><td>1.85</td><td>0.20</td></tr><tr><td>Spectral tail Ti(r) only</td><td>0.88</td><td>1.32</td><td>0.19</td></tr><tr><td>Margin µf only</td><td>0.86</td><td>1.42</td><td>0.21</td></tr><tr><td>Spectral × margin (ours)</td><td>0.91</td><td>1.21</td><td>0.19</td></tr><tr><td>Oracle (exhaustive audit)</td><td>0.93</td><td>8.00</td><td>0.19</td></tr></table>

Table 19: Predictor ablation at $T = 1 { , } 0 0 0$ on ZsRE / LLaMA-3-8B. Utility is the audited behavioral contract score; Audits/edit measures the cost of the proposal at insertion time; Memory is normalized to exact LoRA. Bold marks the configuration used in the main paper. The combined spectral-times-margin predictor reaches utility within 0.02 of oracle at 4.6× fewer audits per edit, while each factor alone trails by 0.03–0.05. The alwaysrank-1 baseline shows that the audit-and-promote loop alone is competitive on utility but pays a higher audit cost without a predictor to localise the proposal.

The sketch passes but the exact fails (compression removes harmful detail).

• Both-fail: Neither passes (edit is “exactlimited”).

• False retirement: An exact-pass edit that fails after compression.

The spectral-filter visual audit is summarized by the frontier analysis in Figure 4, which shows how much edit energy is retained at each rank and how this relates to behavioral contract satisfaction.

## F Fixed-Rank Tier Diagnostics

Table 20 reports the per-rank compression statistics (memory, retained energy, behavioral contract score, and false retirement count) across $N \in$ {32, 128, 512} for both datasets, expanding the perrank view summarised in the main-text population structure. Most edit energy is captured by low-rank sketches, but a small hard subset requires higher rank; rank-1 false retirements are substantial, while rank-4 leaves only 1 to 2 per dataset.

## F.1 Swap-Count and Frontier Visuals

Figure 7 shows swap-count plots for ZsRE and CF, visualizing pass-set transitions as rank increases. These plots help explain how many edits change status (e.g., from fail to pass) as more detail is added.

![](images/0f530faa4f16a76c6fd2d30ca8e91ec9ff9330ccd4d2e55a62333e3240ccb688.jpg)

![](images/8e455d533a8bdb8d715a55d06393d7f527f6007913b4f79fcbc4b9bb19363c5b.jpg)  
Figure 7: Swap-count diagnostics for ZsRE and CF at $N = 5 1 2$ . The plots show how many edits change pass/fail status as rank increases, highlighting the difficult subset that requires higher rank.

## F.2 Interpretation

These diagnostics clarify several practical points about the method:

• Why not constant memory? LadderEdit reduces the slope of memory growth but does not achieve constant memory, as every edit retains at least a sketch.

• Why is exact LoRA not an oracle? The denoising audit shows that compressed sketches can sometimes outperform the exact update, indicating that exact LoRA is not always optimal for behavioral contracts.

• What is the role of the audit? The behavioral audit is essential: spectral energy is predictive but not sufficient, so every candidate rung is checked for contract satisfaction.

• How robust is the method? The difficult subset of edits that require higher rank is small and stable across datasets and N, supporting the scalability of the approach.

• What about overfitting and scalability? Thresholds and audit parameters are tuned on validation data, and the audit is efficient because most edits are handled by low-rank sketches.

## F.3 Baseline Details

We group the compared methods by the role they play in the lifelong editing pipeline. This grouping is used only for presentation; all methods are evaluated under the same edit streams and benchmark metrics whenever applicable.

Direct parameter editing. Fine-tuning (FT) updates model parameters directly on the rewrite objective and serves as a simple lower-bound baseline for sequential editing. ROME (Meng et al., 2022) performs localized rank-one updates to factual associations in transformer feed-forward layers. MEMIT (Meng et al., 2023) extends this style of localized parameter intervention to batch editing. These methods are not designed as persistent editmemory systems, but they are included because they are standard model-editing references and expose the interference that can accumulate under long edit streams.

Memory- and routing-based lifelong editors. GRACE (Hartvigsen et al., 2023) stores edits in a discrete key-value memory and routes queries to stored correction behavior. WISE (Wang et al., 2024a) separates edited knowledge into an explicit side memory. MEMOIR (Wang et al., 2025) focuses on minimal overwrite and informed retention for lifelong editing. These methods reduce direct overwriting of the base model, but their routing, retention, or memory mechanisms are part of the editor itself. We therefore evaluate them with their native routing policies rather than imposing LadderEdit’s edit-association protocol.

Constrained-update methods. AlphaEdit (Fang et al., 2025) constrains edits in a null space to reduce interference with unrelated behavior. It represents a constrained parameter-update family rather than an edit-local memory bank. We include it to compare against methods that improve stability by restricting update directions instead of preserving isolated per-edit representations.

LoRA and adapter-based lifelong editors. MELO (Yu et al., 2023) uses neuron-indexed dynamic LoRA modules for model editing. ELDER (Li et al., 2024a) enhances lifelong editing with a mixture-of-LoRA mechanism. MEMoE (Wang and Li, 2024b) and LEMoE (Wang and Li, 2024a) use mixture-of-experts style adapter mechanisms for sequential editing. These methods are closest to LadderEdit in using parameterefficient edit representations, but they primarily address how edits are routed, combined, or specialized. LadderEdit instead assumes an edit-local representation has already been acquired and studies how much of that representation must be retained persistently.

Table 20: Fixed-rank tier diagnostics for CF and ZsRE at N = 32, 128, 512. Memory is normalized to exact LoRA. Retained energy is $1 - \| \rho \| _ { F } ^ { 2 } / \| \Delta ^ { E } \| _ { F } ^ { 2 }$ . False retirements count exact-pass edits that fail after compression.
<table><tr><td>Dataset</td><td>N</td><td>Rank</td><td>Memory</td><td>Energy</td><td>Contract</td><td>False Ret.</td></tr><tr><td>CF</td><td>32</td><td>1</td><td>0.125</td><td>0.560</td><td>0.625</td><td>1.67</td></tr><tr><td>CF</td><td>32</td><td>2</td><td>0.250</td><td>0.747</td><td>0.656</td><td>0.00</td></tr><tr><td>CF</td><td>32</td><td>4</td><td>0.500</td><td>0.899</td><td>0.646</td><td>0.00</td></tr><tr><td>CF</td><td>32</td><td>8</td><td>1.000</td><td>1.000</td><td>0.646</td><td>0.00</td></tr><tr><td>CF</td><td>128</td><td>1</td><td>0.125</td><td>0.559</td><td>0.641</td><td>7.00</td></tr><tr><td>CF</td><td>128</td><td>2</td><td>0.250</td><td>0.744</td><td>0.662</td><td>1.67</td></tr><tr><td>CF</td><td>128</td><td>4</td><td>0.500</td><td>0.898</td><td>0.669</td><td>0.00</td></tr><tr><td>CF</td><td>128</td><td>8</td><td>1.000</td><td>1.000</td><td>0.664</td><td>0.00</td></tr><tr><td>CF</td><td>512</td><td>1</td><td>0.125</td><td>0.556</td><td>0.647</td><td>39.33</td></tr><tr><td>CF</td><td>512</td><td>2</td><td>0.250</td><td>0.738</td><td>0.698</td><td>11.33</td></tr><tr><td>CF</td><td>512</td><td>4</td><td>0.500</td><td>0.894</td><td>0.714</td><td>2.00</td></tr><tr><td>ZsRE</td><td>32</td><td>1</td><td>0.125</td><td>0.567</td><td>0.516</td><td>1.50</td></tr><tr><td>ZsRE</td><td>32</td><td>2</td><td>0.250</td><td>0.759</td><td>0.578</td><td>0.00</td></tr><tr><td>ZsRE</td><td>32</td><td>4</td><td>0.500</td><td>0.909</td><td>0.563</td><td>0.00</td></tr><tr><td>ZsRE</td><td>32</td><td>8</td><td>1.000</td><td>1.000</td><td>0.563</td><td>0.00</td></tr><tr><td>ZsRE</td><td>128</td><td>1</td><td>0.125</td><td>0.558</td><td>0.578</td><td>7.00</td></tr><tr><td>ZsRE</td><td>128</td><td>2</td><td>0.250</td><td>0.752</td><td>0.635</td><td>0.33</td></tr><tr><td>ZsRE</td><td>128</td><td>4</td><td>0.500</td><td>0.905</td><td>0.633</td><td>0.33</td></tr><tr><td>ZsRE</td><td>128</td><td>8</td><td>1.000</td><td>1.000</td><td>0.633</td><td>1.00</td></tr><tr><td>ZsRE</td><td>512</td><td>1</td><td>0.124</td><td>0.551</td><td>0.550</td><td>16.67</td></tr><tr><td>ZsRE</td><td>512</td><td>2</td><td>0.247</td><td>0.745</td><td>0.575</td><td>3.00</td></tr><tr><td>ZsRE</td><td>512</td><td>4</td><td>0.494</td><td>0.901</td><td>0.576</td><td>1.33</td></tr></table>

Adaptive-rank LoRA. AdaLoRA (Zhang et al., 2023) adaptively reallocates low-rank budget during training. It is included because it is the closest training-time rank-allocation baseline. However, AdaLoRA solves a different problem from LadderEdit: it decides where rank should be spent across layers or modules while fitting an adapter, whereas LadderEdit decides how much resolution an already acquired edit-local memory should retain in a growing stream. Running AdaLoRA independently for each edit still leaves one stored adapter per edit and no stream-level residual policy; running it as a shared adapter couples unrelated edits and reintroduces capacity competition.

Full-resolution edit-memory reference. Exact LoRA (Hu et al., 2022) is our full-resolution editlocal memory reference. Each edit is acquired as an exact LoRA-style update and stored without compression. This preserves edit-local capacity but causes persistent memory to grow linearly with the number of edits. LadderEdit uses the same edit acquisition and lookup association as exact LoRA, but replaces the returned full-resolution update with the cheapest audited representation that preserves the measured rewrite, generalization, and locality contract.

## G Extended Analysis of Scalability, Storage, and Practical Deployment

This section provides additional analyses of LadderEdit beyond the primary experiments. We examine backbone scaling, strict memory-matched comparisons, full-resolution acquisition cost, longhorizon storage growth, retrieval dependence, rankallocation efficiency, audit-probe coverage, qualitative audit behavior, and implementation details. Together, these experiments characterize the operating regime of LadderEdit and separate the effect of adaptive storage allocation from retrieval, cache selection, and initial edit acquisition.

## G.1 Scaling to Larger Backbones

The main experiments use 7B–8B parameter backbones. To test whether edit compression introduces an obvious scale-dependent degradation, we additionally evaluated Qwen2.5-14B and Qwen2.5-32B (Yang et al., 2024; Bai et al., 2025; Yang et al., 2025). The writer, edit stream, and evaluation protocol are unchanged; only the backbone size is increased.

The difference from Exact LoRA remains small at both scales: 0.004 average points for Qwen2.5- 14B and 0.003 for Qwen2.5-32B. Locality remains at 0.999 in all cases. These results indicate that the storage mechanism does not exhibit an obvious degradation when moving beyond the 7B–8B regime evaluated in the main experiments.

Table 21: LadderEdit on larger Qwen2.5 backbones. Accuracy (Acc.), generalization (Gen.), locality (Loc.), and their average are reported. LadderEdit remains close to Exact LoRA as the backbone increases from 14B to 32B parameters.
<table><tr><td>Backbone</td><td>Method</td><td>Acc.</td><td>Gen.</td><td>Loc.</td><td>Avg.</td></tr><tr><td>Qwen2.5-14B</td><td>Exact LoRA</td><td>0.795</td><td>0.905</td><td>0.999</td><td>0.900</td></tr><tr><td></td><td>LadderEdit</td><td>0.790</td><td>0.900</td><td>0.999</td><td>0.896</td></tr><tr><td>Qwen2.5-32B</td><td>Exact LoRA</td><td>0.821</td><td>0.923</td><td>0.999</td><td>0.914</td></tr><tr><td></td><td>LadderEdit</td><td>0.816</td><td>0.918</td><td>0.999</td><td>0.911</td></tr></table>

Table 22: Full-stream utility of exact-cache policies under the same capacity of 91 retained adapters out of 512 edits. Learned retention with base-model fallback is the strongest exact-cache baseline.
<table><tr><td>Cache policy</td><td>Full-stream utility</td></tr><tr><td>Random retention</td><td>0.18</td></tr><tr><td>FIFO</td><td>0.20</td></tr><tr><td>LRU</td><td>0.22</td></tr><tr><td>Learned retention</td><td>0.35</td></tr><tr><td>Learned retention + base fallback</td><td>0.41</td></tr></table>

## G.2 Exact-Cache Baselines and Coverage Loss

A keep-or-drop exact cache provides a natural alternative under a strict storage budget: retained edits preserve their full LoRA adapters, while nonretained edits fall back to the base model. To separate cache quality from the inherent coverage limitation of this strategy, we evaluated multiple cache policies under the same budget of 91 retained adapters from a stream of 512 edits.

The strongest learned cache is more than twice as effective as random retention, showing that cache selection itself matters substantially. However, the retained-only control reveals that the remaining difference is primarily a coverage effect. Across cache policies, utility on the edits that remain in the cache is approximately 0.93–0.94, essentially matching LadderEdit on the same retained subset. The full-stream gap emerges because the remaining 421 edits no longer have an edit-specific representation once they are evicted.

The learned retention baseline uses recency and historical retrieval-frequency features. Its scoring model is fitted only on validation streams; test outcomes are not used to select retained edits. We therefore use learned retention with base-model fallback as the primary exact-cache baseline, while random, FIFO, and LRU serve as additional controls.

Table 23: Strict-memory comparison on CounterFact (CF) and ZsRE. Utility is measured under approximately matched normalized persistent memory.
<table><tr><td>Dataset</td><td>Policy</td><td>Norm. memory</td><td>Utility</td></tr><tr><td>CF</td><td>Exact-cache</td><td>0.180</td><td>0.434</td></tr><tr><td>CF</td><td>Rank-1 for all edits</td><td>0.125</td><td>0.753</td></tr><tr><td>CF</td><td>Static mixed-rank</td><td>0.180</td><td>0.815</td></tr><tr><td>CF</td><td>LadderEdit</td><td>0.180</td><td>0.862</td></tr><tr><td>ZsRE</td><td>Exact-cache</td><td>0.178</td><td>0.420</td></tr><tr><td>ZsRE</td><td>Rank-1 for all edits</td><td>0.124</td><td>0.765</td></tr><tr><td>ZsRE</td><td>Static mixed-rank</td><td>0.179</td><td>0.796</td></tr><tr><td>ZsRE</td><td>LadderEdit</td><td>0.179</td><td>0.866</td></tr></table>

## G.3 Strict Memory-Matched Frontier

The cache comparison above changes which edits remain represented. We additionally construct a strict-memory frontier in which all methods are compared under matched normalized persistent memory. This isolates the value of maintaining a low-cost representation for every edit from the value of adaptively assigning additional rank.

This comparison separates two sources of improvement. First, replacing keep-or-drop caching with a rank-1 representation for every edit increases utility by 0.319 on CounterFact and 0.345 on ZsRE, despite using less normalized memory. Thus, preserving even a minimal edit-specific representation substantially reduces the coverage loss created by eviction.

Second, adaptive allocation remains beneficial after coverage is restored. Relative to static mixedrank storage at approximately the same memory budget, LadderEdit improves utility by another 0.047 on CounterFact and 0.070 on ZsRE. The first gain therefore comes from representing the full edit stream; the second comes from allocating additional resolution selectively rather than uniformly.

## G.4 Full-Resolution Acquisition Cost

LadderEdit is a post-acquisition storage controller. It does not reduce the optimization cost required to fit the initial full-resolution LoRA adapter. Exact LoRA and LadderEdit use the same writer, optimizer, edit stream, and edit identity. The methods differ only in the representation retained after an edit has been acquired.

Table 24: Measured insertion and memory cost at $T = 1 0 , 0 0 0$ edits. Persistent memory refers to the retained edit bank, whereas peak GPU memory measures transient working memory during insertion.
<table><tr><td>Method</td><td>Insert / edit Peak GPU</td><td></td><td>Persistent memory</td></tr><tr><td>Exact LoRA</td><td>86 ms</td><td>18.4 GB</td><td>100.93 GB</td></tr><tr><td>LadderEdit</td><td>124 ms</td><td>18.8 GB</td><td>19.44 GB</td></tr><tr><td>Difference</td><td>+38 ms</td><td>+0.4 GB</td><td>-81.49 GB</td></tr></table>

This exact-first design avoids forcing every edit into a fixed low-rank representation before observing whether that representation is behaviorally sufficient. Its cost is an additional compression and audit stage after fitting.

At $T = 1 0 { , } 0 0 0$ , the additional insertion cost is 38 ms per edit, corresponding to approximately 380 seconds, or 6.3 minutes, over the complete stream. This cost is paid once at insertion. In exchange, the persistent edit bank is reduced from 100.93 GB to 19.44 GB, a reduction of 81.49 GB or 80.7%. Peak working memory changes only slightly, from 18.4 GB to 18.8 GB.

The rank proposal also limits the behavioral search overhead. LadderEdit requires an average of 1.21 audits per edit, compared with 8 audits for exhaustive rank evaluation.

Insertion cost depends on stream length because maintenance of a large full-rank adapter bank introduces increasing merge and housekeeping overhead. LadderEdit is slower at short horizons, with the measured crossover occurring near $T = 2 { , } 0 0 0$ $\mathrm { A t } T = 5 0 \mathrm { , 0 0 0 }$ , measured insertion time is approximately 150 ms per edit for LadderEdit and 850 ms per edit for Exact LoRA.

The practical claim is therefore not cheaper initial edit fitting. The acquisition procedure is intentionally unchanged; the benefit comes from substantially lower persistent storage and lower longstream maintenance cost.

## G.5 Long-Horizon Storage Growth

LadderEdit reduces the coefficient of storage growth but does not make storage constant in the number of edits. Every edit retains at least a rank-1 sketch, so persistent storage remains linear in stream length.

At $T = 1 0 { , } 0 0 0 .$ , Exact LoRA requires 100.93 GB while LadderEdit requires 19.44 GB, corresponding to a measured 5.19× reduction at this operating point. Under the average per-edit storage observed at this horizon, a 100 GB memory budget corresponds to approximately 9,900 Exact-LoRA edits or 51,400 LadderEdit edits. This is a budget-equivalent estimate based on the measured $T = 1 0 , 0 0 0$ operating point; it does not assume that the ratio is constant at every horizon.

Table 25: Diagnostic rates describing edit-level rank requirements and compression behavior. The rows are diagnostic properties and are not assumed to form a mutually exclusive partition.
<table><tr><td>Diagnostic property</td><td>ZsRE</td><td>CounterFact</td></tr><tr><td>Rank-1 sufficient</td><td>81.2%</td><td>76.4%</td></tr><tr><td>Rank 2 or higher required</td><td>13.4%</td><td>16.8%</td></tr><tr><td>Full acquisition rank required</td><td>3.1%</td><td>4.7%</td></tr><tr><td>Sketch passes while exact fails</td><td>4.3%</td><td>6.7%</td></tr></table>

Across the evaluated horizons, the reduction grows from approximately 2.0× at $T \ = \ 1 0 0 ,$ to approximately 5.0–5.2× across backbones at $T = 1 0 , 0 0 0$ , and approximately 11.6–12.0× at $T = 5 0 { , } 0 0 0 .$ . The 5.2× operating point reported in the main text corresponds specifically to LLaMA-3- 8B at $T = 1 0 , 0 0 0$ . Thus, the observed range is approximately 2×–12× over the evaluated horizons rather than a single horizon-independent factor.

The lower growth coefficient is consistent with strong heterogeneity in the amount of resolution required by individual edits.

Most edits therefore do not require their complete acquisition-rank representation to be stored indefinitely. LadderEdit assigns every edit a lowcost representation and spends additional rank only on the subset for which the audit indicates that more detail is required.

A truly bounded-memory editor would additionally require retirement, consolidation, or merging. LadderEdit instead addresses the preceding representation problem: given an edit that remains individually addressable, how much edit-specific resolution should be retained? The method can therefore serve as a representation layer beneath a separate bounded-memory policy, but constantmemory behavior is outside the scope of the present study.

## G.6 Retrieval and Edit Selection

The primary storage experiments use matched edit association to isolate representation quality from retrieval errors. Under this protocol, Exact LoRA and LadderEdit receive the same selected edit identity, allowing the comparison to ask whether the selected edit can be represented more economically without violating its behavioral contract.

Table 26: Performance under matched association and learned retrieval. Contriever is evaluated directly for both Exact LoRA and LadderEdit. Additional LadderEdit results with E5-large and frozen MPNet characterize sensitivity to the retriever.
<table><tr><td>Selection mechanism</td><td>Exact LoRA Avg.</td><td>LadderEdit Avg.</td></tr><tr><td>Matched association</td><td>0.95</td><td>0.95</td></tr><tr><td>Contriever (Izacard et al., 2021)</td><td>0.92</td><td>0.92</td></tr><tr><td>E5-large (Wang et al., 2022)</td><td></td><td>0.91</td></tr><tr><td>Frozen MPNet (Song et al., 2020)</td><td></td><td>0.88</td></tr></table>

To test whether the conclusion survives a realistic selection stage, we additionally replace matched association with learned dense retrieval.

With Contriever, both methods decrease from 0.95 to 0.92, while their relative behavior remains unchanged. LadderEdit reaches 0.91 with E5-large and 0.88 with frozen MPNet. Because Exact LoRA was not independently evaluated with the latter two retrievers, these results are used only to measure LadderEdit’s retrieval sensitivity rather than to claim direct parity.

These experiments clarify the decomposition of the system. LadderEdit controls how an alreadyselected edit is stored; it does not solve edit retrieval itself. Its end-to-end performance therefore depends on the quality of the edit-selection mechanism used at inference time.

## G.7 Rank Proposal and Audit Efficiency

The components used by LadderEdit—low-rank factorization, spectral statistics, and behavioral evaluation—are individually standard. The central design problem is instead edit-level storage allocation: determining how much rank each edit should retain while minimizing the number of behavioral evaluations required to reach that decision.

We compare several rank-proposal rules under the same audit procedure.

The combined proposal reaches 0.91 utility, within 0.02 of exhaustive rank search, while using 1.21 rather than 8.00 audits per edit, a 6.6× reduction in audit count.

The two signals capture complementary information. The spectral tail measures the amount of update energy removed by truncation, whereas the behavioral margin measures how much degradation the exact edit can tolerate before its contract is violated. These quantities provide an empirical ranking heuristic rather than a theoretical guarantee of the minimum sufficient rank.

Table 27: Rank-proposal ablation. The combined spectral-tail and behavioral-margin proposal approaches exhaustive rank search while requiring substantially fewer audits.
<table><tr><td>Rank proposal</td><td>Utility</td><td>Audits / edit</td></tr><tr><td>Random</td><td>0.81</td><td>2.31</td></tr><tr><td>Always rank-1, then promote</td><td>0.89</td><td>1.85</td></tr><tr><td>Spectral tail only</td><td>0.88</td><td>1.32</td></tr><tr><td>Behavioral margin only</td><td>0.86</td><td>1.42</td></tr><tr><td>Spectral tail + margin</td><td>0.91</td><td>1.21</td></tr><tr><td>Exhaustive oracle</td><td>0.93</td><td>8.00</td></tr></table>

Table 28: Sensitivity to semantic coverage of the behavioral audit probes.
<table><tr><td>Probe coverage</td><td>Overall Avg.</td></tr><tr><td>38%</td><td>0.84</td></tr><tr><td>64%</td><td>0.91</td></tr><tr><td>87%</td><td>0.94</td></tr><tr><td>96%</td><td>0.94</td></tr><tr><td>99%</td><td>0.95</td></tr></table>

## G.8 Audit-Probe Coverage

Behavioral auditing requires a small set of probes that exercise the intended edit behavior. Benchmark experiments use disjoint audit, validation, and test prompts. Test prompts are never used to determine the retained rank.

In a deployment setting, audit probes may be supplied together with an edit, generated from the requested rewrite, or obtained from an applicationspecific regression suite. LadderEdit assumes that such a representative probe set is available; without it, the controller cannot directly test whether compression preserves the intended behavior.

To measure sensitivity to incomplete probe coverage, we subsample the semantic coverage of the audit set.

Performance degrades gradually as semantic coverage becomes weaker. At 87% coverage the overall average is already 0.94, compared with 0.95 at 99% coverage. As an additional leakage control, an intentionally non-disjoint audit configuration reaches 0.99, whereas enforcing a strict audit/test separation yields 0.95. The latter is the relevant evaluation setting and shows that the reported performance does not require reuse of test prompts during rank selection.

This analysis studies missing semantic coverage. It does not evaluate mislabeled, systematically corrupted, or adversarial audit probes, which remain separate failure modes.

## G.9 Qualitative Audit Outcomes

The behavioral audit produces four recurring outcomes that clarify why a fixed storage rank is suboptimal:

1. Sketch-sufficient: a low-rank sketch already preserves the required edit behavior, so storing additional directions is unnecessary.

2. Detail-needed: rank 1 violates the behavioral contract, but a higher-rank representation, such as rank 2, restores the edit.

3. Compression-regularized: the compressed representation passes the behavioral contract even when the full acquired adapter fails on the same probe, indicating that truncation removed directions that were harmful for that behavior.

4. Exact-limited: even the full acquisition-rank representation fails the audit, so increasing retained rank cannot repair the edit.

A representative compression-regularized ZsRE edit has the target “architect.” The full Exact-LoRA representation answers “archaeologist,” whereas the lower-rank representation includes the correct target. The audit therefore retains the compressed sketch rather than automatically preferring the maximum available rank. Compressionregularized behavior occurs in 4.3% of the ZsRE diagnostics and 6.7% of the CounterFact diagnostics.

The detail-needed cases provide the complementary behavior: rank 1 may discard directions that are required to preserve the edit, while rank 2 or another promoted representation succeeds. The controller is therefore not based on the assumption that stronger compression is intrinsically beneficial. Instead, compression level is selected according to measured behavioral sufficiency.

## G.10 Correlation Between Compression and Behavioral Outcomes

We additionally examine whether the compression diagnostics correlate with the observed behavioral outcomes. Using the rounded population proportions, the compression-regularized subsets correspond to approximately 86 ZsRE edits and 134 CounterFact edits. The resulting Pearson correlations are moderate: approximately $r = 0 . 3 8$ for ZsRE and $r = 0 . 4 1$ for CounterFact.

Using these rounded subset sizes, the corresponding approximate 95% confidence intervals are

$$
r _ { \mathrm { Z s R E } } = 0 . 3 8 , \qquad \mathrm { C I } _ { 9 5 \% } \approx [ 0 . 1 8 , 0 . 5 5 ] ,
$$

and

$$
r _ { \mathrm { C F } } = 0 . 4 1 , \qquad \mathrm { C I } _ { 9 5 \% } \approx [ 0 . 2 6 , 0 . 5 4 ] .
$$

These correlations are consistent with the proposed interpretation but should not be read as causal evidence that low-rank truncation itself produces the behavioral improvement. The primary evidence for adaptive storage allocation instead comes from the budget-matched frontier in Table 23 and from the held-out behavioral audits. The correlation analysis is used only as a secondary diagnostic of the relationship between compression statistics and behavioral outcomes.

## G.11 SVD Procedure and Memory Accounting

Compression is applied independently to each adapted LoRA module. For a given module, LadderEdit forms that module’s effective update matrix and performs the corresponding singular-value decomposition before truncating it to the selected retained rank. Parameters from different layers are never concatenated into a single global matrix because the adapted modules have different dimensions and functional roles.

For one edit, the compression procedure therefore consists of an edit-level compression pass containing the required module-wise decompositions. Persistent storage is computed by summing the retained LoRA factors over all adapted modules. This convention ensures that both compression and memory accounting are defined consistently across layers and backbones.

The audit weights are fixed globally rather than tuned separately for each backbone. Contract and predictor thresholds are selected per dataset using held-out validation probes and are subsequently fixed across backbones, stream horizons, and test streams.

## G.12 Scope of the Efficiency Claim

The extended experiments distinguish four different efficiency questions that can otherwise be conflated.

First, edit acquisition is intentionally unchanged: LadderEdit begins from the same fitted LoRA update as Exact LoRA and therefore does not claim to make the initial optimization cheaper.

Second, persistent representation is substantially smaller. At the directly measured $T =$ 10,000 operating point, persistent storage decreases from 100.93 GB to 19.44 GB, corresponding to a 5.19× reduction.

Third, storage remains linear in stream length. LadderEdit lowers the per-edit growth coefficient by exploiting heterogeneous rank requirements; it is not a constant-memory editor.

Fourth, selection and storage are separate. Matched-association experiments isolate the storage mechanism, while the Contriever experiment shows that the same behavior persists when a learned retrieval component is introduced. Retrieval quality can affect end-to-end accuracy, but it does not alter the memory advantage of the retained representation once an edit is selected.

Taken together, the results characterize LadderEdit as an adaptive representation layer for long edit streams: it preserves an edit-specific representation for every stored edit, allocates high resolution only when behaviorally necessary, and substantially reduces persistent storage without introducing a corresponding degradation in edit accuracy, generalization, or locality. “

## G.13 Summary

The rank-compression analysis supports the core claim: most edits are highly compressible, and LadderEdit’s ladder allocation reduces memory without sacrificing reliability, generalization, or locality in most evaluated settings. The behavioral audit ensures that only behaviorally hard edits are promoted to higher rank, and the diagnostics explain why this allocation improves over exact-cache and other baselines.