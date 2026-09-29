# FORGE: Form-Optimal Routing of Grounded Evidence for Frozen LLM Agents

Xi Xiao<sup>1</sup>, Yunbei Zhang<sup>2</sup>, Chen Liu<sup>3</sup>, Lin Zhao<sup>4</sup>, Jialin Chen<sup>3</sup>, Tianchen Zhao<sup>5</sup>, Xiang Xu<sup>5</sup>, Youngeun Kim<sup>6</sup>, Tianyang Wang<sup>1</sup>, Min Xu<sup>7</sup>

<sup>1</sup>University of Alabama at Birmingham <sup>2</sup>Tulane University <sup>3</sup>Yale University <sup>4</sup>Northeastern University <sup>5</sup>Amazon AGI <sup>6</sup>Korea University <sup>7</sup>Carnegie Mellon University

✉ xxiao@uab.edu

Website Code

## ABSTRACT

In agentic AI systems, frozen foundation models are increasingly deployed as closed-weight API endpoints, making downstream adaptation possible only through the inputs and inference procedures surrounding the model. As a result, for each input query, two coupled decisions largely determine both answer quality and token cost: what evidence to provide and how much reasoning budget to allocate. Fixed defaults along these axes are often suboptimal, misallocating support form or reasoning depth on roughly 80% of queries in our analysis. To address this challenge, we propose FORGE, a unified framework for adapting frozen models through per-query routing over a joint action space that spans both support form and thinking depth. Under an entropy-regularized, cost-aware utility objective, we derive a closed-form Boltzmann routing target and instantiate the policy as a lightweight 269K-parameter factorized router. The routing policy is trained around the frozen host, without any weight access, through a three-stage pipeline: offline arm enumeration, supervised Kullback-Leibler (KL) distillation from the Boltzmann target, and Group Relative Policy Optimization (GRPO) refinement with host feedback. Across 5 knowledge-intensive benchmarks and 8 frozen backbones ranging from 7B to 671B parameters, FORGE improves accuracy at 42–45% lower token cost on both main hosts, transfers zero-shot across hosts at lower token cost, and composes with intrinsic thinking budgets where available.

![](images/007e81311acde9df86dfb827db983c7af6d83fe8b685d8e71c96e5c7d9fee02c.jpg)  
Figure 1: Fixed support forms in frozen-host workflows (a,b) and per-query support-form selection with FORGE (c).

## 1 INTRODUCTION

Agentic AI has seen remarkable progress in recent years, with many of the most capable systems built on top of large foundation models (Achiam et al., 2023; Comanici et al., 2025; Sapkota et al., 2025; Ferrag et al., 2025). Aligning these general-purpose models with task-specific requirements usually relies on adaptation via fine-tuning, using methods such as low-rank adaptation (LoRA) (Hu et al., 2022) and residual adapters (Rebuffi et al., 2017). However, foundation models are increasingly deployed as closed API endpoints, with private weights and no access to internal states, making host-side fine-tuning infeasible (Sun et al., 2022; Cheng et al., 2024a; Gao et al., 2023; Asawa et al., 2025). This raises a central question: how can we effectively adapt frozen, closed-weight foundation models to downstream tasks?

For a fixed frozen host, effective adaptation hinges on two axes: what evidence to provide (Jeong et al., 2024; Asai et al., 2024; Zhu et al., 2026; Lewis et al., 2020b; Karpukhin et al., 2020), and how much reasoning budget to allocate (OpenAI et al., 2024; Guo et al., 2025; Anthropic, 2025; Wang et al., 2025; Wei et al., 2022; Snell et al., 2024; Press et al., 2023; Trivedi et al., 2023; Yue et al., 2025; Muennighoff et al., 2025). In practice, however, agentic AI workflows commonly rely on fixed forms of support applied uniformly across queries, such as direct inference or retrieval-augmented generation with a predefined evidence format (Figure 1). This uniform treatment can be suboptimal because the most effective support form and reasoning budget vary substantially across queries. A controlled probe of a frozen Qwen3-8B on HotpotQA (N=2,500) makes this gap concrete (Figure 2). On the evidence axis, 7.3% of queries are answered more accurately without retrieval than with full passages, 79.4% achieve identical F1 under a query-aware compressed summary as under the full top-k, and only the remaining 20.6% show any F1 difference between the two. On the reasoning axis with Qwen3-8B, 12% of queries are harmed by long reasoning, 60% reach identical F1 under NOTHINK as under THINK-HIGH, and only 28% genuinely benefit. These results suggest that both axes call for adaptive, per-query decisions. Beyond this, the two decisions are inherently coupled: better evidence reduces the reasoning needed to reach an answer, and deeper reasoning reduces the evidence required to support it. Addressing either alone leaves this coupling unexploited.

This observation raises three open questions. 1 How should the two axes be jointly adapted at the query level? Adaptive-retrieval and adaptive-reasoning methods each address one axis while fixing the other, leaving no formulation that adapts both together per query. 2 How can such a joint policy be trained around a frozen, weight-inaccessible host? Since the host cannot be finetuned, adaptation must rely on a separate router that learns from external signals such as offline labels and online host feedback. However, existing methods treat these two signals as separate training paradigms, rather than as complementary signals for a common cost-aware objective. 3 What target should such a policy be trained toward? Existing routers define training targets heuristically, without a closed-form connection to the underlying cost-aware decision problem.

To answer 1 , we formalize joint adaptation as per-query routing over a unified action space. The action $a = \left( \mu , \theta \right)$ pairs a supportform µ ∈ {Direct, Summary, Raw}, corresponding to three evidence-channel rates between the corpus and the host (no retrieval, query-aware compressed retrieval, and full top-k passages), with a thinking depth θ (two settings on non-thinking hosts and four on hosts that expose test-time compute control). For each query, the router selects a single joint action a that maximizes the Pareto utility $U ( x , a ) = F _ { 1 } ( x , a ) - \bar { \lambda _ { \mathrm { i n } } } \tilde { c } _ { \mathrm { i n } } ( a ) - \lambda _ { \mathrm { o u t } } \tilde { c } _ { \mathrm { o u t } } ( x , a )$ , trading answer F1 against normalized input and output token cost. Crucially, the two axes are no longer optimized in isolation: a single utility evaluates the joint choice and a single policy outputs it per query. Within this formulation, FORGE instantiates the joint policy as a 269K-parameter factorized router that selects the action in a single forward pass before the frozen host answers. Importantly, standard RAG, Adaptive-RAG, and tiered-memory routing arise as degenerate special cases of the proposed FORGE by fixing one or more routing dimensions.

a  
![](images/b143bdb3f8fbe4f62de64f62f3ba23285c70dcd425984fd41802916ebea36b28.jpg)

b  
![](images/67f28268500b8c9f0106966afdd21b4a2b4a23402f1c6065293f1d849c1a43af.jpg)  
C

![](images/ca5019396befaa99560c6bd7aba78c610ce133179f7039dbb54b92958083f5df.jpg)  
Figure 2: Frozen LLM agents are picky readers. a. The three common support forms. b. No support form is universally optimal; the best choice depends on the input query. c. Our proposed FORGE establishes a new Pareto frontier over fixed support forms.

To answer $\textcircled{2}$ we train the router through a three-stage pipeline that runs entirely around the frozen host without changing its weights. Stage 0 enumerates a 3-arm warm-start subset $\mathcal { A } _ { \mathrm { w a r m } } =$ M × {NoThink} on 1,500 training queries, recording per-arm F1 and token cost (≈ 4,500 host calls, under one hour). Stage 1 fits the warm-start router $\pi _ { \theta _ { 0 } }$ to a Boltzmann target restricted to ${ \mathcal A } _ { \mathrm { w a r m } }$ via KL distillation (CPU-only, two minutes). Stage 2 lifts the policy to the full alphabet A via GRPO (Shao et al., 2024) with sampled host rewards and a KL anchor to $\pi _ { \theta _ { 0 } }$ (8–12 hours on one 8-GPU node). The same pipeline produces the 6-arm router for non-thinking hosts, the 12-arm joint-policy router that composes with the reasoning-budget axis on capable hosts, and a single router that transfers zero-shot across hosts. Router computation adds 0.5–5.5 ms per query; on a fresh query, the full variant also issues four parallel host probes for its features, whereas FORGE-Lite needs only the final answer call (§4.2, App. B.9).

To answer $\textcircled{3}$ we ground the router in a unified Pareto reward and a closed-form Boltzmann target. The Pareto utility above is the system’s single source of truth: every training stage, every baseline, and every Pareto operating point sweeps along it. Adding an entropy regularizer to the cost-aware utility maximization, as in maximum-entropy reinforcement learning (Ziebart et al., 2008; Haarnoja et al., 2018), fixes a unique closed form on the discrete action space, $\pi ^ { * } ( a \mid x ) \propto \exp ( U ( x , a ) / \hat { \tau } )$ . The same softmax form is also the choice rule of discrete-choice random-utility models (MCFADDEN, 1973), suggesting that the Boltzmann target is a natural cost-aware decision rule rather than an ad hoc soft label. Stage 1 distills this target on the warm-start actions, and Stage 2 refines the policy against the same utility with sampled host rewards; App. D characterizes the idealized fixed point of the KLanchored Stage-2 objective. Proposition 3 (App. D.3) further decomposes the warm-start mismatch into a support-form marginal gap plus a closed-form log |Θ|−H cost from the uniform thinking head. These results characterize the target and its initialization rather than guarantee convergence of the implemented optimizer.

Across 5 knowledge-intensive benchmarks and 8 frozen LLM backbones from 7B to 671B parameters across three deployment regimes (non-thinking local, thinking-capable, frontier non-thinking), FORGE Pareto-dominates Always-Raw on every cell tested on Qwen3-8B and Mistral-7B (+2.4 and +2.9 macro F1 at 45% and 42% lower cached token cost; §4.2). A single router trained on Qwen3-8B transfers zero-shot to three frontier backends, matching or exceeding Always-Raw F1 on 11 of 15 task / host cells at 30%+ lower cached cost (§4.6). On thinking-capable hosts the joint 12-arm policy exceeds Always-Raw with maximum thinking budget by +1.4 to +1.9 macro F1 at roughly half the total token cost (§4.5). We believe that this work contributes to the tractable, theoretically grounded, and broadly transferable axis of frozen-agent adaptation.

## 2 PRELIMINARIES AND RELATED WORKS

## 2.1 RELATED WORKS

Model routing selects the host (Chen et al., 2024; Ong et al., 2025; Jiang et al., 2025b; Cheng et al., 2024b; Su et al., 2024; Jiang et al., 2023; Hu et al., 2024; Ding et al., 2024); adaptive retrieval selects whether and what to retrieve (Jeong et al., 2024; Asai et al., 2024; Zhu et al., 2026); thinking-budget control selects reasoning effort (Wang et al., 2025; Alomrani et al., 2025; Welleck et al., 2024; Yao et al., 2023). Building on these choices, including partial support-form selection, FORGE jointly selects Direct, Summary, or Raw evidence and a host-exposed thinking setting under one measured quality–cost utility (extended discussion in App. A).

## 2.2 ROUTING OBJECTIVE AND BOLTZMANN SOLUTION

Let A be a frozen pre-trained LLM with fixed, inaccessible parameters, x a query, and $\mathcal { D } \mathrm { \sf ~ a }$ document corpus. The router selects $a = ( \mu , \theta ) \in \mathcal { A } = \mathcal { M } \times \Theta$ , with support $\mu \in \{ \mathrm { D i r e c t } $ , Summary, Raw} and thinking configuration $\theta \left( \ S 2 . 3 \right)$ . Using prompt constructor $g _ { a }$ , the host returns $\hat { y } = A ( g _ { a } ( x , \mathcal { D } ) )$ , with F1 $m ( \bar { x } , a ) \overset { \mathcal { = } } \mathrm { F } 1 ( \hat { y } , y )$ and input/output token costs $c _ { \mathrm { i n } } ( a ) , c _ { \mathrm { o u t } } ( x , a )$ . Dataset-level maxima normalize these costs to $\tilde { c } _ { \mathrm { i n } } , \tilde { c } _ { \mathrm { o u t } }$ , yielding the per-query Pareto utility:

$$
U ( x , a ) ~ = ~ m ( x , a ) ~ - ~ \lambda _ { \mathrm { { i n } } } { \tilde { c } } _ { \mathrm { { i n } } } ( a ) ~ - ~ \lambda _ { \mathrm { { o u t } } } { \tilde { c } } _ { \mathrm { { o u t } } } ( x , a ) , ~ \lambda _ { \mathrm { { i n } } } , \lambda _ { \mathrm { { o u t } } } \geq 0 ,\tag{1}
$$

The weights set the quality–cost operating point; $\lambda _ { \mathrm { o u t } } { = } 0$ penalizes input cost only. Labeled queries provide $m ( x , a )$ for executed actions during training and evaluation. At inference, routing uses pre-answer features without observing the selected response’s F1.

For a fixed query x, finite action set A, finite utilities $U ( x , a )$ , and temperature $\tau > 0$ , the entropyregularized objective ma $\begin{array} { r } { \mathrm { { x } } _ { \pi ( \cdot \vert x ) \in \Delta ( A ) } \sum _ { a } \pi ( a \mid x ) U ( x , a ) + \tau H ( \pi ( \cdot \mid x ) ) } \end{array}$ has the unique solution

$$
\pi ^ { * } ( a \mid x ) = \frac { \exp \bigl ( U ( x , a ) / \tau \bigr ) } { \sum _ { a ^ { \prime } } \exp \bigl ( U ( x , a ^ { \prime } ) / \tau \bigr ) } ,\tag{2}
$$

As $\tau  0 .$ , it concentrates uniformly on utility maximizers; as $\tau  \infty ,$ , it becomes uniform. The standard solution gives Stage 1 soft labels on enumerated warm-start actions. Optimality holds for the stated entropy-regularized objective, without guaranteeing that the parameterized router or clipped Stage 2 update attains it. Stage 2 refines the policy with sampled host rewards over the full combination of support forms and thinking settings (§3.2).

## 2.3 THE ADAPTIVE-THINKING ACTION ALPHABET

The support form $\mu$ is Direct, the query alone (negligible input tokens); Summary, BM25 retrieval with query-aware sentence-level compression (≈ 250 input tokens); or Raw, full top-k BM25 passages (≈ 450 input tokens). Thinking depth θ is NoThink (plain decoding), CoT-Prompt (the in-context “Let us think step by step” suffix), or, when the host exposes a reasoning budget, Think-Low or Think-High (construction in App. B.3). Hosts without a reasoning mode use $\Theta _ { \mathrm { n t } } \mathrm { = \{ N o T h i n k , C o T \mathrm { - P r o m p t } \} ~ ( } K \mathrm { = } 6 )$ ; thinking-capable hosts use $\Theta _ { \mathrm { t } } \mathrm { = } \{ \mathrm { N o T h i n k , C o T \mathrm { - } P r o m p t } $ , Think-Low, Think-High} (K=12).

Query-dependent utility maximizers motivate adaptive routing. Proposition 1 (App. D.1) specifies when a per-query utility oracle strictly exceeds every fixed action in expectation; §4 and §4.6 examine action heterogeneity and six-arm references. The Oracle rows are descriptive references: their selection rule was not retained, precluding claims of utility optimality or F1 upper bounds.

## 3 METHOD

FORGE jointly controls evidence delivery and reasoning depth before the frozen host answers. The insight is their interdependence: evidence quality changes the value of extra reasoning, while the reasoning setting changes what support is worth providing. Candidate actions’ F1 and input/output tokens define the utility in Eq. (1) and soft targets in Eq. (2). Figures 3 and 4 summarize training and routing.

## 3.1 ROUTER AND DEPLOYMENT FEATURES

The policy factors into a support-form head and a support-conditioned reasoning head:

$$
\pi _ { \boldsymbol \theta } ( a \mid x ) = \pi _ { \boldsymbol \theta } ^ { \boldsymbol \mu } ( \boldsymbol \mu \mid h ( x ) ) \pi _ { \boldsymbol \theta } ^ { \boldsymbol \theta } ( \boldsymbol \theta \mid h ( x ) , \boldsymbol \mu ) .\tag{3}
$$

A shared two-layer MLP maps inputs through 256 hidden units to a 256-dimensional representation (ReLU, dropout 0.1); the heads use a learned 16-dimensional support-form embedding. The six-arm Full router has approximately 269K parameters (settings in App. B.4). Full FORGE uses a frozen 768-dimensional BGE embedding and 21 structured features (789 total). Eleven features depend on one greedy host probe and three self-consistency samples, requiring four host calls before the routed answer on a fresh query. Caching shifts these calls outside routed-answer accounting but does not eliminate their fresh-query cost. FORGE-Lite is retrained on 778 host-independent features: the embedding, four BM25 statistics, one query–top-passage cosine similarity, and five structural features. It requires no training or inference probes and only one final-answer host call. Both variants share the single-step action set and frozen retrieval pipeline. Cached counts routed-answer tokens; Fresh Online tokens and latency include every host call for a new query (Appendix Figure S1).

## 3.2 TRAINING

Stage 0: action evaluation. On each labeled warmstart query $x _ { i } .$ the frozen host executes $a \in \mathcal { A } _ { \mathrm { w a r m } } =$ $\mathcal { M } \times \mathrm { \bar { \{ N o T h i n k \} } }$ and records $( m _ { i } ^ { a } , c _ { \mathrm { i n } , i } ^ { a } , c _ { \mathrm { o u t } , i } ^ { a } )$ . These calls incur training cost without updating host weights. Gold answers provide $m _ { i } ^ { a } ; \ S 4$ tests proxy rewards with gold-labeled development data for router hyperparameter selection.

Stage 1: supervised distillation. Using $U ( x _ { i } , a )$ from $\begin{array} { r l } { \operatorname { E q . } } & { { } \left( 1 \right) } \end{array}$ the evidence head minimizes $D _ { \mathrm { K L } } ( \pi _ { \mathrm { B o l t z } } ^ { * } \Vert \pi _ { \theta } ^ { \mu } )$ against Eq. (2) on the three warmstart actions at $\tau _ { 0 } ~ = ~ 1$ ; the reasoning head starts uniform. A separate Stage-1-only control uses six-arm offline observations, holding the full action set fixed to isolate the contribution of Stage 2.

![](images/797371a68028b110e5efe20dfef7de95e8845cc8cc79fded3b8daa84c7e98ca4.jpg)

Stage 2: policy-guided refinement. Over the full action set, each of B queries receives G sampled actions, stratified to cover every evidence form when $G \geq | { \mathcal { M } } |$ Frozen-host executions yield $r _ { b , g } = U ( x _ { b } , a _ { b , g } )$ and normalized group-relative advantages

Figure 3: Three-stage training around a frozen host. Stage 2 uses $5 , 0 0 0 \times 3 2 =$ 160K query groups with eight sampled actions each: 1.28M host completions.

$$
\hat { A } _ { b , g } = \frac { r _ { b , g } - \bar { r } _ { b } } { \sigma _ { r _ { b } } + \epsilon } , \qquad \bar { r } _ { b } = { \textstyle \frac { 1 } { G } } \sum _ { g } r _ { b , g } ,\tag{4}
$$

where $\epsilon = 1 0 ^ { - 4 }$ and zero-variance groups receive zero advantage. A PPO-clipped policy term (Schulman et al., 2017; Shao et al., 2024) uses this feedback, with a KL anchor to Stage 1 and an entropy bonus. Appendix B.4 specifies sampling and the surrogate; the same-action-set ablation (§4) isolates refinement from alphabet expansion.

## 4 EXPERIMENTS

## 4.1 SETUP AND EVALUATION PROTOCOLS

We evaluate HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al., 2020), MuSiQue (Trivedi et al., 2022), PopQA (Mallen et al., 2023), and FEVER (Thorne et al., 2018) on Qwen3-8B-Instruct and Mistral-7B-Instruct-v0.3, plus three thinking-capable and three frontier hosts. The six-arm alphabet pairs Direct/Summary/Raw with NoThink/CoT-Prompt; thinking hosts add Think-Low/High for 12 arms. BM25 retrieval is fixed; Summary deterministically extracts from the top three passages. Splits and retrieval settings are in Appendices B.6 and B.3.

We report token-level F1, exact match (EM), and cost $\bar { C } = \bar { C } _ { \mathrm { i n } } + \bar { C } _ { \mathrm { o u t } }$ in k host tokens/query. Cached cost counts the routed answer with precomputed features; Fresh Online includes feature acquisition and the answer on a new query: four pre-route host calls for Full, none for Lite. Baselines are fixed Direct/Summary/Raw, BM25-Threshold, Adaptive-RAG (Jeong et al., 2024), TierMem (Zhu et al., 2026), s3 (Jiang et al., 2025b), and Sysformer (Sharma et al., 2025). The matched BGE+BM25-KL router uses 768 BGE dimensions, four BM25 statistics, and one query–passage cosine, with the same Stage-0 data, MLP capacity, optimizer, and development budget as FORGE, while excluding GRPO refinement and host-dependent feature probes.

![](images/b963594ca21ef89a8c3010d6aacd45f16e850ad355ef06634476d74232be3334.jpg)  
Figure 4: Factorized router and frozen-host inference. The 789 → 256 → 256 encoder feeds support and support-conditioned thinking heads (features: Table S6). Bottom panels expand support choices and alternative settings of one host: CoT is prompted; Low/High use host-exposed budgets. Probabilities, token counts, and the answer are illustrative.

Table 1: Canonical Cached results. Task F1 (%) and selected-answer cost C<sup>¯</sup> (k/query); HQA/MSQ/Pop/FEV denote HotpotQA/MuSiQue/PopQA/FEVER. Bold/boxed values mark best practical task/macro F1 per host. Oracle is a descriptive six-arm reference; Table 2 gives online costs, including the feature calls needed before answering a new query.
<table><tr><td rowspan="2">Policy</td><td colspan="6">Qwen3-8B</td><td colspan="6">Mistral-7B</td></tr><tr><td></td><td>HQA 2Wiki MSQ</td><td></td><td>Pop</td><td>FEV</td><td>Macro</td><td>Č</td><td>HQA 2Wiki MSQ</td><td></td><td>Pop</td><td>FEV Macro</td><td>č</td></tr><tr><td>Always-Direct</td><td>26.2</td><td>25.7</td><td>10.2</td><td>15.0</td><td>54.4</td><td>26.3 0.040</td><td>25.2</td><td>17.7</td><td>7.9 23.0</td><td></td><td>47.0</td><td>24.2 0.041</td></tr><tr><td>Always-Summary</td><td>46.9</td><td>38.1</td><td>19.0 87.0</td><td></td><td>81.8</td><td>54.6 0.270</td><td>38.3</td><td>24.5</td><td>11.8</td><td>80.0</td><td>69.5</td><td>44.8 0.265</td></tr><tr><td>Always-Raw</td><td>57.2</td><td>40.5</td><td>24.1 83.9</td><td></td><td>79.6</td><td>57.1 0.430</td><td>46.6</td><td>28.7</td><td>14.0 74.8</td><td></td><td>72.0</td><td>47.2 0.428</td></tr><tr><td>BM25-Threshold</td><td>50.5</td><td>38.0</td><td>19.5 80.5</td><td></td><td>78.0</td><td>53.3 0.268</td><td>41.5</td><td>26.2</td><td>12.5 73.0</td><td></td><td>69.0</td><td>44.4 0.263</td></tr><tr><td>Adaptive-RAG</td><td>56.5</td><td>40.0</td><td>23.0 80.0</td><td></td><td>77.5</td><td>55.4 0.305</td><td>45.5</td><td>27.0</td><td>13.573.0</td><td></td><td>70.0</td><td>45.7 0.300</td></tr><tr><td>TierMem-2arm</td><td>47.6</td><td>39.6</td><td>20.8 81.8</td><td></td><td>81.8</td><td>54.3 0.316</td><td>40.9</td><td>28.2</td><td>12.9 70.4</td><td></td><td>70.4</td><td>44.6 0.314</td></tr><tr><td>s3</td><td>55.0</td><td>39.0</td><td>21.5 78.5</td><td></td><td>75.5</td><td>53.9 0.305</td><td>45.5</td><td>26.5</td><td>13.8 71.0</td><td></td><td>67.0</td><td>44.80.310</td></tr><tr><td>Sysformer</td><td>57.5</td><td>38.8</td><td>21.2</td><td>86.0</td><td>76.0</td><td>55.9 0.285</td><td>47.8</td><td>23.5</td><td>14.2 75.0</td><td></td><td>69.0</td><td>45.9 0.300</td></tr><tr><td>Stage 1 (3-arm)</td><td>58.3</td><td>38.3</td><td>20.8 87.0</td><td></td><td>75.7</td><td>56.0 0.272</td><td>48.4</td><td>23.0</td><td>14.5 76.6</td><td></td><td>70.0</td><td>46.5 0.288</td></tr><tr><td>Stage 1 (6-arm)</td><td>59.0</td><td>40.5</td><td>23.5 87.2</td><td></td><td>78.0</td><td>57.6 0.255</td><td>49.0</td><td>28.5</td><td>15.5 77.5</td><td></td><td>71.0</td><td>48.3 0.272</td></tr><tr><td>FORGE</td><td>60.4</td><td>42.7</td><td>26.4 87.5</td><td></td><td>80.5</td><td>59.5 0.236</td><td>50.2</td><td>30.5</td><td>16.7 80.5</td><td></td><td>72.5 50.1</td><td>0.249</td></tr><tr><td>Oracle (6-arm)</td><td>67.4</td><td>53.2</td><td>34.0 88.5 89.2</td><td></td><td></td><td>66.5 0.168</td><td>59.1</td><td>42.6</td><td>23.2 83.4 82.0</td><td></td><td></td><td>58.1 0.182</td></tr></table>

## 4.2 ANSWER QUALITY AND THE COST OF A NEW QUERY

Full FORGE reaches 59.5/50.1 macro F1 on Qwen/Mistral, exceeding Always-Raw by 2.4/2.9 points with 45%/42% fewer cached answer tokens (Table 1). It exceeds Sysformer, the strongest adapted baseline here, by 3.6/4.2 points; Table 3 tests a matched lightweight comparator and isolates Stage 2 from action expansion. Fixed Summary leads on Qwen FEVER (81.8 vs. 80.5), which motivates the development-set fallback for selecting a task-specific operating point.

Lite improves Qwen/Mistral F1 by 1.6/1.9 while saving 44%/40% of online tokens with roughly 5.5 ms added p50 latency (Table 2). Full gains accuracy but raises Qwen p50 from 122.5 to 271.4 ms. On the DeepSeek-V3.2 API, zero-shot Lite gives 63.1 F1/0.182k tokens/about 2.2 s versus Raw’s 63.4/0.287k/about 2.1 s: 36.6% fewer tokens for 0.3 F1 and a small latency increase. API limits prevented online Full measurement. Lite thus preserves one-call efficiency; Full offers additional

Table 2: Fresh Online results (batch size 1). All host calls and input/output tokens are counted; F1/EM are five-task macro scores. Latency is end-to-end, with Full’s four feature calls parallelized. Bold marks best quality or lowest online cost per host.
<table><tr><td rowspan="2">Host</td><td rowspan="2">Policy</td><td colspan="2">Host calls</td><td colspan="2">Macro (%)</td><td rowspan="2">Online Č (k)</td><td colspan="2">Latency (ms)</td></tr><tr><td>Pre</td><td>Total</td><td>F1</td><td>EM</td><td> $\mathsf { p 5 0 }$ </td><td>p95</td></tr><tr><td rowspan="3">Qwen3-8B</td><td>Always-Raw</td><td>0</td><td>1</td><td>57.1</td><td>48.6</td><td>0.430</td><td>122.5</td><td>148.9</td></tr><tr><td>FORGE-Lite</td><td>0</td><td>1</td><td>58.7</td><td>49.8</td><td>0.242</td><td>128.0</td><td>154.1</td></tr><tr><td>Full FORGE</td><td>4</td><td>5</td><td>59.4</td><td>50.5</td><td>0.384</td><td>271.4</td><td>329.8</td></tr><tr><td rowspan="3">Mistral-7B</td><td>Always-Raw</td><td>0</td><td>1</td><td>47.2</td><td>40.5</td><td>0.428</td><td>129.8</td><td>157.7</td></tr><tr><td>FORGE-Lite</td><td>0</td><td>1</td><td>49.1</td><td>41.9</td><td>0.256</td><td>135.3</td><td>164.8</td></tr><tr><td>Full FORGE</td><td>4</td><td>5</td><td>50.0</td><td>42.7</td><td>0.400</td><td>291.7</td><td>354.6</td></tr></table>

accuracy when deployment permits the latency of parallel feature probes.

## 4.3 RELIABILITY, STRONG BASELINES, AND STAGE-2 VALUE

Table 3: Reliability and matched Stage 2 gains, Cached. $\mathrm { { M e a n } \pm \mathrm { { S D } \mathrm { { : } } } }$ three splits × three seeds, equal baseline tuning budgets; 2,500-update rows are single sweep points. Paired-query bootstrap 95% CIs use the canonical split, comparing policies with Raw or Full with six-arm Stage 1 at matched features, decoding, and cost weights. Boxes mark best nine-run means.
<table><tr><td rowspan="3">Host</td><td rowspan="3">Policy</td><td colspan="2">Macro F1 evidence</td><td rowspan="3">Cached Č (k)</td><td colspan="2">Matched Stage 2 effect</td></tr><tr><td>Nine-run</td><td>∆ vs Raw</td><td>Mean ∆F1</td><td>Canonical</td></tr><tr><td>mean ± SD</td><td>95% CI</td><td></td><td>95% CI</td></tr><tr><td rowspan="7">Qwen3-8B</td><td>Always-Raw</td><td> $5 7 . 0 \pm 0 . 4$ </td><td>reference</td><td>0.431</td><td></td><td></td></tr><tr><td>BGE+BM25-KL</td><td> $5 7 . 5 \pm 0 . 5$ </td><td>[+0.1, +0.9]</td><td>0.268</td><td></td><td></td></tr><tr><td>FORGE-Lite</td><td> $5 8 . 6 \pm 0 . 5$ </td><td>[+1.1, +2.1]</td><td>0.243</td><td></td><td></td></tr><tr><td>Stage 1 (6-arm)</td><td> $5 7 . 6 \pm 0 . 5$ </td><td></td><td>0.255</td><td>reference</td><td></td></tr><tr><td>Stage 2 (2,500 updates)</td><td>58.9</td><td></td><td>0.244</td><td></td><td></td></tr><tr><td>Full FORGE (5,000 updates)</td><td>59.4 ±0.4</td><td>[+1.9, +2.9]</td><td>0.237</td><td>+1.8</td><td>[+1.3,+2.3]</td></tr><tr><td>Always-Raw</td><td> $4 7 . 1 \pm 0 . 4$ </td><td></td><td>0.429</td><td></td><td></td></tr><tr><td rowspan="5">Mistral-7B</td><td>BGE+BM25-KL</td><td></td><td>reference [0.0, +0.8]</td><td>0.279</td><td></td><td></td></tr><tr><td>FORGE-Lite</td><td> $4 7 . 5 \pm 0 . 5$   $4 9 . 0 \pm 0 . 5$ </td><td>[+1.4, +2.4]</td><td>0.257</td><td></td><td></td></tr><tr><td>Stage 1 (6-arm)</td><td> $4 8 . 3 \pm 0 . 5$ </td><td></td><td>0.272</td><td>reference</td><td></td></tr><tr><td>Stage 2 (2,500 updates)</td><td>49.5</td><td></td><td>0.259</td><td></td><td></td></tr><tr><td>Full FORGE (5,000 updates)</td><td>50.0</td><td>±0.5 [+2.3,+3.5]</td><td>0.250</td><td>+1.7</td><td>[+1.2, +2.2]</td></tr></table>

BGE+BM25-KL’s Mistral interval starts at zero; the five retuned adaptive baselines have intervals below zero against Raw on both hosts. The canonical Qwen/Mistral ladder separates labels, actions, structure, and host features: hard-label BGE+BM25 56.8/46.7, KL distillation 57.5/47.5, sixarm routing with the same 773 features 58.4/48.6, Lite 58.7/49.1, and Full 59.4/50.0 macro F1.

Query embeddings yield the largest HotpotQA gain, while the host probe adds 2.3 F1 (Table 4). Retrieval adds four BM25

Table 4: Cumulative feature ablation. Qwen3-8B Cached F1 (%); parentheses show changes from the previous row. Full ϕ is the final row, evaluated in a separate feature-study run from Table 1.
<table><tr><td>Features</td><td>HotpotQA</td><td>MuSiQue</td></tr><tr><td>Structured only</td><td>56.8</td><td>23.4</td></tr><tr><td>+ BGE embedding</td><td>60.4(+3.6)</td><td>25.9 (+2.5)</td></tr><tr><td>+ Retrieval</td><td>59.3 (−1.1)</td><td>26.4 (+0.5)</td></tr><tr><td>+ Host probe</td><td>61.6 (+2.3)</td><td> $2 6 . 4 \left( + 0 . 0 \right)$ </td></tr><tr><td>+ Self-consistency</td><td>60.6(-1.0)</td><td>26.9 (+0.5)</td></tr></table>

statistics and one query–passage cosine. Appendix B.5 gives the matched 773-to-789-dimensional comparison.

At matched actions, features, decoding, and cost weights, Stage 2 adds 1.8/1.7 macro F1 on Qwen/Mistral and saves 7.1%/8.1% cached tokens (Table 3); 2,500 updates deliver about 70% of the final gain. Refinement thus improves decisions within the same action space. Training uses $5 , 0 0 0 \times 3 2 ^ { - } \times 8 = 1 . 2 8 \mathbf { M }$ host completions, about 384M input/82M output tokens, and 64–96

GPU-hours on one eight-GPU node per operating point. This one-time training cost is separate from recurring feature probes, whose online overhead is measured in Table 2.

## 4.4 GROUNDING, PROXY REWARDS, AND OPERATING POINTS

Table 5: Grounding on Qwen3-8B (%). Canon ical predictions; bold marks the best comparable value for each metric within each task.
<table><tr><td>Policy</td><td>F1</td><td>Gold recall</td><td>Source supp.</td><td>Prompt supp.</td><td>Unsup.</td></tr><tr><td colspan="6">HotpotQA</td></tr><tr><td>Raw</td><td>57.2</td><td>91.2</td><td>61.9</td><td>59.1</td><td>30.8</td></tr><tr><td>Summary</td><td>46.9</td><td>73.4</td><td>54.0</td><td>51.8</td><td>37.1</td></tr><tr><td>Full</td><td>60.4</td><td>64.8</td><td>65.3</td><td>63.7</td><td>27.4</td></tr><tr><td colspan="6">FEVER</td></tr><tr><td>Raw</td><td>79.6</td><td>94.1</td><td>82.0</td><td>80.3</td><td>12.9</td></tr><tr><td>Summary</td><td>81.8</td><td>87.6</td><td>84.4</td><td>83.1</td><td>10.4</td></tr><tr><td>Full</td><td>80.5</td><td>79.8</td><td>83.7</td><td>82.5</td><td>10.9</td></tr></table>

Table 6: Frozen proxy rewards, Qwen3-8B. Best outcomes are bold; boxes mark the highest macro F1 and exact match.
<table><tr><td>Stage 0/1 Stage 2 reward</td><td>reward</td><td>F1</td><td>EM</td><td>ć Source supp.</td></tr><tr><td colspan="5">Gold F1 used in Stage 0/1</td></tr><tr><td rowspan="2">Gold F1</td><td>None Gold F1</td><td>57.6 59.5</td><td>48.9 0.255 50.6 0.236</td><td>65.9 67.4</td></tr><tr><td>Verifier Judge</td><td>58.8 59.1</td><td>49.8 0.241 50.1 0.239</td><td>66.9 67.5</td></tr><tr><td colspan="5">No gold F1 in any training stage</td></tr><tr><td>Verifier</td><td>Verifier</td><td>58.1</td><td>49.2 0.248</td><td>66.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Judge</td><td>Judge</td><td>58.6</td><td>49.6 0.245</td><td>67.1</td></tr></table>

Grounding (Table 5). Raw/Summary denote Always-Raw/Always-Summary; Full is Full FORGE. F1 is answer F1; gold recall measures annotated evidence in the prompt. Supp./Unsup. denote support/unsupported rates from a fixed NLI-style evaluator (threshold 0.70). Source support/unsupported rate cover all examples; prompt support uses each policy’s Summary/Raw subset and is descriptive, not paired. A balanced 200-example human audit gives 87.0% agreement (Cohen’s $\kappa = 0 . 7 4 )$  
Proxy rewards (Table 6). Verifier/Judge denote the frozen verifier/LLM judge; None is Stage 1 only. Canonical Cached evaluation: five-task macro F1/EM use held-out gold labels $( \% ) ; \bar { C }$ counts selected-answer k tokens/query. An independent frozen evaluator scores source support (%). Proxy runs reuse hyperparameters selected on gold-labeled development data, as in the gold-reward experiments.

Full improves HotpotQA F1 and lowers unsupported answers: gold evidence recall falls from 91.2 to 64.8 while source support rises from 61.9 to 65.3 (Table 5). Selective evidence delivery can therefore improve answer grounding even with less annotated evidence in the prompt. Fixed Summary leads on all three outcomes for FEVER. These answer-level metrics do not establish reasoning-trace faithfulness, and the generated outputs contain no citations.

A frozen judge supplies all training rewards and reaches 58.6 F1, within 0.9 F1 and 0.3 source-support points of gold-reward training, at 0.245k versus 0.236k cached tokens (Table 6). Gold warm-start with judge-based Stage 2 narrows the F1 gap to 0.4. Proxy rewards thus retain most of the gains; hyperparameter selection still uses labeled development data.

Cost weights $( \lambda _ { \mathrm { i n } } , \lambda _ { \mathrm { o u t } } ) = ( 0 . 1 0 , 0 . 2 0 )$ are tuned on Qwen HotpotQA development data and fixed thereafter. A Fresh Online sweep from (0.05, 0.00) to (0.20, 0.20) yields Qwen Full 59.9 F1/0.410k tokens to 58.7/0.358k, and Lite 59.2/0.268k to 58.0/0.216k. Mistral Full spans 50.5/0.430k to 49.3/0.373k, and Lite 49.6/0.287k to 48.4/0.229k; Full’s first point exceeds Raw’s 0.428k. A fallback requires a positive lower bound on the paired 95% development utility-difference interval against the best fixed arm. It selects FORGE for HotpotQA/2Wiki/MuSiQue and Summary for PopQA/FEVER: 59.7 F1/50.6 EM/0.235k cached tokens versus Full’s 59.5/50.6/0.236k, using labeled development data to select a suitable task-specific operating point.

## 4.5 THINKING CONTROLS AND TRANSFER

With fixed retrieval and one pre-answer decision, the 12-arm router adds 1.4–1.9 macro F1 over Raw+Think-High at roughly half its cached cost (Table 7). Gains concentrate on multi-hop tasks. Direct+Think-High reaches 38.5/47.3/48.4 F1 versus Raw+Think-High’s 63.4/69.6/69.5: evidence still matters at the High budget, motivating joint control of both.

The full policy reaches $D _ { \mathrm { K L } } = 0 . 0 2$ from the Boltzmann target, versus 0.49 for Stage 1 and 0.64 without the KL anchor (Figure 5). This supports the anchor’s role in retaining the target’s selective allocation across joint support–thinking combinations.

Table 7: Joint support and thinking control. Task F1 (%) and Cached cost (k/query). Sixarm FORGE excludes Think-Low/High; AdaReasoner+ (Wang et al., 2025) fixes support per task. Bold/boxes mark best task/macro F1 within each host.
<table><tr><td rowspan="2">Host</td><td rowspan="2">Policy</td><td colspan="5">Task F1 (%)</td><td rowspan="2">Macro F1</td><td rowspan="2">č</td></tr><tr><td>HQA</td><td>2Wiki</td><td>MSQ</td><td>PopQA</td><td>FEVER</td></tr><tr><td rowspan="6">Qwen3-8B- Thinking</td><td>Raw+NoThink</td><td>58.0</td><td>41.2</td><td>24.5</td><td>84.5</td><td>80.0</td><td>57.6</td><td>0.485</td></tr><tr><td>Raw+Think-High</td><td>62.5</td><td>47.0</td><td>30.8</td><td>87.0</td><td>89.5</td><td>63.4</td><td>1.280</td></tr><tr><td>Direct+Think-High</td><td>38.0</td><td>30.5</td><td>18.5</td><td>38.0</td><td>67.5</td><td>38.5</td><td>0.890</td></tr><tr><td>AdaReasoner+</td><td>62.5</td><td>46.5</td><td>31.0</td><td>88.0</td><td>90.5</td><td>63.7</td><td>0.952</td></tr><tr><td>FORGE, 6-arm</td><td>60.4</td><td>42.7</td><td>26.4</td><td>87.8</td><td>82.5</td><td>60.0</td><td>0.236</td></tr><tr><td>FORGE, 12-arm</td><td>64.8</td><td>49.2</td><td>33.5</td><td>88.0</td><td>91.0</td><td>65.3</td><td>0.690</td></tr><tr><td rowspan="6">Claude-4- Sonnet</td><td>Raw+NoThink</td><td>65.5</td><td>47.8</td><td>30.5</td><td>89.5</td><td>86.0</td><td>63.9</td><td>0.482</td></tr><tr><td>Raw+Think-High</td><td>70.5</td><td>54.0</td><td>37.0</td><td>91.5</td><td>95.0</td><td>69.6</td><td>1.350</td></tr><tr><td>Direct+Think-High</td><td>49.5</td><td>36.5</td><td>25.5</td><td>49.5</td><td>75.5</td><td>47.3</td><td>0.910</td></tr><tr><td>AdaReasoner+</td><td>70.0</td><td>53.5</td><td>36.5</td><td>92.5</td><td>95.0</td><td>69.5</td><td>0.985</td></tr><tr><td>FoORGE, 6-arm</td><td>67.0</td><td>49.0</td><td>32.0</td><td>92.0</td><td>88.5</td><td>65.7</td><td>0.235</td></tr><tr><td>FORGE, 12-arm</td><td>72.4</td><td>55.5</td><td>38.8</td><td>93.0</td><td>95.5</td><td>71.0</td><td>0.715</td></tr><tr><td rowspan="6">DeepSeek-R1</td><td>Raw+NoThink</td><td>64.0</td><td>46.5</td><td>31.5</td><td>92.0</td><td>86.5</td><td>64.1</td><td>0.480</td></tr><tr><td>Raw+Think-High</td><td>68.5</td><td>53.0</td><td>38.0</td><td>93.5</td><td>94.5</td><td>69.5</td><td>1.420</td></tr><tr><td>Direct+Think-High</td><td>50.5</td><td>38.0</td><td>27.5</td><td>50.0</td><td>76.0</td><td>48.4</td><td>0.940</td></tr><tr><td>AdaReasoner+</td><td>68.5</td><td>53.0</td><td>38.0</td><td>93.5</td><td>95.0</td><td>69.6</td><td>1.045</td></tr><tr><td>FORGE, 6-arm</td><td>65.5</td><td>47.5</td><td>32.0</td><td>93.5</td><td>88.0</td><td>65.3</td><td>0.234</td></tr><tr><td>FORGE, 12-arm</td><td>70.5</td><td>55.0</td><td>39.5</td><td>94.5</td><td>95.5</td><td>71.0</td><td>0.730</td></tr></table>

![](images/50a756659bbdbaca532554fc48e9c06145c2f1f1d6eff0f19bcd0c04770b51c5.jpg)  
Figure 5: Joint action distributions from 1,500 samples per policy. KL compares each distribution with the leftmost Boltzmann training target, distinct from the descriptive Oracle in Tables 1 and S17.

## 4.6 CROSS-HOST TRANSFER

Table 8: Zero-shot Cached transfer. A Qwen-trained Full router with no target-host Stage 2. Cost excludes fresh probes. Bold marks best F1/cost per host; Appendix C.5 gives per-task results.
<table><tr><td rowspan="2">Target host</td><td colspan="2">Macro F1 (%)</td><td colspan="2">Cached Č (k)</td></tr><tr><td>Raw</td><td>FORGE</td><td>Raw</td><td>FORGE</td></tr><tr><td>Llama-3.3-70B</td><td>58.5</td><td>61.2</td><td>0.318</td><td>0.205</td></tr><tr><td>DeepSeek-V3.2</td><td>63.4</td><td>63.4</td><td>0.287</td><td>0.176</td></tr><tr><td>Qwen3.5-397B</td><td>63.1</td><td>65.6</td><td>0.305</td><td>0.198</td></tr></table>

Full matches or exceeds Raw in 11/15 task–host cells, with macro F1 gains of +2.7/+0.0/+2.5 and 35.5%/38.7%/35.1% fewer cached tokens (Table 8). One source-trained router improves the quality– cost balance across host families and scales; DeepSeek ties Raw at the reported precision with the largest token saving. The online Lite result supports no-probe API use. Target-host calibration could further tailor Direct support to each host’s prior knowledge.

## 5 CONCLUSION

FORGE jointly controls evidence form and reasoning depth around a frozen host. Its factorized router distills an entropy-regularized quality–cost target and refines decisions through sampled host feedback. Across five tasks, Full improves nine-run cached macro F1 over Raw by 2.4/2.9 points on Qwen/Mistral; matched controls attribute 1.8/1.7 points to Stage 2. Lite improves local-host F1 and token cost with one online call, while Full uses four parallel probes for further accuracy. FEVER favors Summary; DeepSeek transfer primarily saves tokens. Results across host families establish joint evidence-form and thinking control as a practical adaptation mechanism with explicit quality, token, and latency tradeoffs across host families and deployment settings.

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, Shyam Upadhyay, Manaal Faruqui, and Mausam. Automix: Automatically mixing language models. Advances in Neural Information Processing Systems, 37:131000–131034, 2024.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 12248–12267, 2024.

Mohammad Ali Alomrani, Yingxue Zhang, Derek Li, Qianyi Sun, Soumyasundar Pal, Zhanguang Zhang, Yaochen Hu, Rohan Deepak Ajwani, Antonios Valkanas, Raika Karimi, Peng Cheng, Yunzhou Wang, Pengyi Liao, Hanrui Huang, Bin Wang, Jianye Hao, and Mark Coates. Reasoning on a budget: A survey of adaptive and controllable test-time compute in llms. arXiv preprint arXiv:2507.02076, 2025.

Anthropic. Claude 3.7 sonnet and claude code. https://www.anthropic.com/news/ claude-3-7-sonnet, 2025. Blog post.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, pp. 9112–9141, 2024.

Parth Asawa, Alan Zhu, Matei Zaharia, Alex Dimakis, and Joseph E Gonzalez. How to train your advisor: Steering black-box llms with advisor models. In First Workshop on Foundations of Reasoning in Language Models, 2025.

Prakhar Bansal and Shivangi Agarwal. Lightweight query routing for adaptive rag: A baseline study on ragrouter-bench. arXiv preprint arXiv:2604.03455, 2026.

Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024.

Jiale Cheng, Xiao Liu, Kehan Zheng, Pei Ke, Hongning Wang, Yuxiao Dong, Jie Tang, and Minlie Huang. Black-box prompt optimization: Aligning large language models without model training. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3201–3219, 2024a.

Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one token. Advances in Neural Information Processing Systems, 37:109487–109516, 2024b.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Rühle, Laks VS Lakshmanan, and Ahmed Hassan Awadallah. Hybrid llm: Cost-efficient and quality-aware query routing. In The Twelfth International Conference on Learning Representations, 2024.

Dujian Ding, Ankur Mallick, Shaokun Zhang, Chi Wang, Daniel Madrigal, Mirian Del Carmen Hipolito Garcia, Menglin Xia, Laks VS Lakshmanan, Qingyun Wu, and Victor Rühle. Best-route: Adaptive llm routing with test-time optimal compute. In International Conference on Machine Learning, pp. 13870–13884. PMLR, 2025.

Mohamed Amine Ferrag, Norbert Tihanyi, and Merouane Debbah. From llm reasoning to autonomous ai agents: A comprehensive review. arXiv preprint arXiv:2504.19678, 2025.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1):32, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. Pmlr, 2018.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 6609–6625, 2020.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-llm routing system. In Agentic Markets Workshop at ICML 2024, 2024.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 7036–7050, 2024.

Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng,

Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang, Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, and Jiawei Han. Adaptation of agentic ai. arXiv preprint arXiv:2512.16301, 2025a.

Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, and Jiawei Han. s3: You don’t need that much data to train a search agent via rl. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 21610–21628, 2025b.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, pp. 7969–7992, 2023.

Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus Rashtchian. Sufficient context: A new lens on retrieval augmented generation systems. In The Thirteenth International Conference on Learning Representations, 2025.

Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, and Saravan Rajmohan. Acon: Optimizing context compression for long-horizon llm agents. arXiv preprint arXiv:2510.00615, 2025.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pp. 6769–6781, 2020.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020a.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020b.

Ruoran Li, Xinghua Zhang, Haiyang Yu, Shitong Duan, Xiang Li, Wenxin Xiang, Chonghua Liao, Xudong Guo, Yongbin Li, and Jinli Suo. Mempo: Self-memory policy optimization for longhorizon agents. arXiv preprint arXiv:2603.00680, 2026.

Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Retrieval augmented generation or long-context llms? a comprehensive study and hybrid approach. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 881–893, 2024.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In Second Conference on Language Modeling, 2025.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870, 2024.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9802–9822. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.546. URL https://aclanthology.org/2023. acl-long.546/.

D MCFADDEN. Conditional logit analysis of qualitative choice behavior. Frontier in Econometrics, 1973.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori B Hashimoto. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20286–20332, 2025.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms from preference data. In International Conference on Learning Representations, 2025.

OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Bohan Zhang, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wenting Zhan, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 5687–5711, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. Advances in neural information processing systems, 30, 2017.

Ranjan Sapkota, Konstantinos I Roumeliotis, and Manoj Karkee. Ai agents vs. agentic ai: A conceptual taxonomy, applications and challenges. Information Fusion, pp. 103599, 2025.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Kartik Sharma, Yiqiao Jin, Vineeth Rakesh, Yingtong Dou, Menghai Pan, Mahashweta Das, and Srijan Kumar. Sysformer: Safeguarding frozen large language models with adaptive system prompts. In ICML 2025 Workshop on Reliable and Responsible Foundation Models, 2025.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 8371–8384, 2024.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. Dragin: Dynamic retrieval augmented generation based on the real-time information needs of large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 12991–13013, 2024.

Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, and Xipeng Qiu. Black-box tuning for language-model-as-a-service. In International Conference on Machine Learning, pp. 20841–20855. PMLR, 2022.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a largescale dataset for fact extraction and VERification. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 809–819. Association for Computational Linguistics, 2018. doi: 10.18653/v1/N18-1074. URL https://aclanthology.org/N18-1074/.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multi-hop questions via single-hop question composition. Transactions ofthe Associationfor Computational Linguistics, 10:539–554, 2022.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pp. 10014–10037, 2023.

Prince Zizhuang Wang and Shuli Jiang. Prime: Training free proactive reasoning via iterative memory evolution for user-centric agent. arXiv preprint arXiv:2604.07645, 2026.

Xiangqi Wang, Yue Huang, Yanbo Wang, Xiaonan Luo, Kehan Guo, Yujun Zhou, and Xiangliang Zhang. Adareasoner: Adaptive reasoning enables more flexible thinking in large language models. arXiv preprint arXiv:2505.17312, 2025.

Ziqi Wang, Xi Zhu, Shuhang Lin, Haochen Xue, Minghao Guo, and Yongfeng Zhang. Ragrouterbench: A dataset and benchmark for adaptive rag routing. arXiv preprint arXiv:2602.00296, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Sean Welleck, Amanda Bertsch, Matthew Finlayson, Hailey Schoelkopf, Alex Xie, Graham Neubig, Ilia Kulikov, and Zaid Harchaoui. From decoding to meta-generation: Inference-time algorithms for large language models. Transactions on Machine Learning Research, 2024.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In Proceedings ofthe 47th international ACM SIGIR conference on research and development in information retrieval, pp. 641–649, 2024.

Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation. arXiv preprint arXiv:2401.15884, 2024.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question

answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 2369–2380, 2018.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Linan Yue, Yichao Du, Yizhi Wang, Weibo Gao, Fangzhou Yao, Li Wang, Ye Liu, Ziyu Xu, Qi Liu, Shimin Di, and Min-Ling Zhang. Don’t overthink it: A survey of efficient r1-style large reasoning models. arXiv preprint arXiv:2508.02120, 2025.

Heng Zhou, Zelin Tan, Zhemeng Zhang, Yutao Fan, Yibing Lin, Li Kang, Xiufeng Song, Rui Li, Songtao Huang, Ao Yu, Yuchen Fan, Yanxu Chen, Kaixin Xu, Xiaohong Liu, Yiran Qin, Philip Torr, Chen Zhang, and Zhenfei Yin. Select-then-solve: Paradigm routing as inference-time optimization for llm agents. arXiv preprint arXiv:2604.06753, 2026.

Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, and Jun Wang. Memento: Fine-tuning llm agents without fine-tuning llms. arXiv preprint arXiv:2508.16153, 2025.

Qiming Zhu, Shunian Chen, Rui Yu, Zhehao Wu, and Benyou Wang. From lossy to verified: A provenance-aware tiered memory for agents. In ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems, 2026.

Brian D Ziebart, Andrew Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In Proceedings of the 23rd national conference on Artificial intelligence-Volume 3, pp. 1433–1438, 2008.

## Technical Appendices

1 Introduction 2   
2 Preliminaries and Related Works .   
2.1 Related Works   
2.2 Routing Objective and Boltzmann Solution   
2.3 The Adaptive-Thinking Action Alphabet .   
3 Method . . .   
3.1 Router and Deployment Features .   
3.2 Training   
4 Experiments   
4.1 Setup and Evaluation Protocols   
4.2 Answer Quality and the Cost of a New Query   
4.3 Reliability, Strong Baselines, and Stage-2 Value   
4.4 Grounding, Proxy Rewards, and Operating Points   
4.5 Thinking Controls and Transfer   
4.6 Cross-Host Transfer   
5 Conclusion . 10   
A Related Work . 17   
A.1 Intrinsic Adaptive Thinking <sup>17</sup><sub>17</sub>   
A.2 Extrinsic Adaptivity: Model and Retrieval Routing   
A.3 Support-Form Routing with a Fixed Host 18   
B Implementation Details 18   
B.1 Hardware and Software Environment   
B.2 Frozen Host Configuration .   
B.3 Retrieval and Action Construction   
B.4 Router Architecture and Three-Stage Training   
B.5 Feature Extraction   
B.6 Dataset Configuration   
B.7 Asset Licenses and Attribution   
B.8 Compute Scale   
B.9 Inference Latency   
B.10 Reproducibility .   
B.11 Supplementary Figures   
C Additional Ablations .   
C.1 Stage-2 Training and KL-Anchor Ablations   
C.2 Factorized versus Flat Policy Head   
C.3 Per-Arm Allocation Changes from Three to Six Actions 27   
C.4 Router Behavior in BGE Embedding Space 28   
C.5 Cross-Host Transfer: Per-Task Breakdown . 29   
C.6 Exact Match (EM) Scores 29   
C.7 Numerical Table for the Action Alphabet Sweep 31   
C.8 Heuristic Routing Baselines 31   
D Properties of the Routing Objective and Warm Start   
D.1 Value of Per-Query Routing   
D.2 A Fixed-Reference Regularized Objective   
D.3 Warm-Start KL Decomposition   
E Interpretation of the Cost–Quality Score 33   
F Limitations and Future Work . <sup>33</sup><sub>33</sub>   
F.1 Limitations   
F.2 Future Work <sup>34</sup><sub>34</sub>   
F.3 Broader Impacts   
G Use of Large Language Models 34

## A RELATED WORK

Section 1 framed single-step, pre-answer routing around a frozen language-model host as a design space with one intrinsic axis (how deeply the model reasons, when the host exposes such a control) and three extrinsic axes (which model to call, how aggressively to retrieve, and what form the retrieved support takes). The three subsections below organize prior work along this taxonomy. Section A.1 covers intrinsic adaptive thinking, Section A.2 covers the two extrinsic axes that have received sustained attention, and Section A.3 covers the support-form axis that FORGE completes.

## A.1 INTRINSIC ADAPTIVE THINKING

A first line of work allocates inference compute inside the backbone itself. Frontier reasoning systems learn to expend more decoding tokens on harder queries through chain-of-thought training (OpenAI et al., 2024; Guo et al., 2025), and recent assistants expose this as a user-facing thinking budget that is set per request (Anthropic, 2025). A growing body of work studies how to control this budget automatically rather than by user toggle. Muennighoff et al. (2025) show that a simple budget-forcing trick at test time recovers most of the gains of full reasoning training on a small base model. Wang et al. (2025) train an RL controller that selects reasoning configurations (temperature, depth) per query for any backbone, with theoretical convergence guarantees. The recent survey of Alomrani et al. (2025) catalogs this space along an L1 (controllability) versus L2 (adaptiveness) axis and lists more than a dozen budget controllers built on top of reasoning-trained hosts.

All of these methods presuppose either a reasoning-trained backbone or a user-exposed thinking toggle, and they condition behavior on query difficulty by changing how the model itself executes. Frozen hosts deployed behind closed APIs or pinned checkpoints cannot generally be retrained by the application, although some expose a per-request reasoning budget. FORGE routes support form and exposed thinking options around a fixed host; our evaluation covers single-step decisions.

## A.2 EXTRINSIC ADAPTIVITY: MODEL AND RETRIEVAL ROUTING

The first extrinsic axis routes queries among backbones of different cost and capability. Frugal-GPT (Chen et al., 2024) cascades a query through increasingly expensive models and stops when a verifier reports sufficient confidence. RouteLLM (Ong et al., 2025) trains a preference-based router that selects a strong or weak model per query. AutoMix (Aggarwal et al., 2024) adds self-verification to estimate answer reliability before escalation. BEST-Route (Ding et al., 2025) jointly routes among models and response sample counts under an explicit cost-quality Pareto objective, the closest prior work in spirit to our framing. These methods also study cost–quality routing, but their decision variable is the backbone itself. FORGE fixes the host and routes the support form for each query.

The second extrinsic axis decides whether and how aggressively to retrieve. Retrieval-augmented generation prepends retrieved passages to every query (Lewis et al., 2020a; Shi et al., 2024); adaptive variants relax this default along several sub-axes. Self-RAG (Asai et al., 2024) trains the language model to emit reflection tokens that trigger retrieval on demand, at the cost of modifying the host. CRAG (Yan et al., 2024) adds a post-hoc evaluator that triggers corrective retrieval when document quality is low, and IRCoT (Trivedi et al., 2023) interleaves retrieval steps with chain-of-thought reasoning for multi-hop queries. Adaptive-RAG (Jeong et al., 2024) trains a query-complexity classifier to pick among no retrieval, single-step retrieval, and iterative multi-step retrieval. Self-Route (Li et al., 2024) frames the decision as a per-query binary choice between RAG and long-context generation. More recent routing benchmarks (Wang et al., 2026; Bansal & Agarwal, 2026; Zhou et al., 2026) catalog inference-time strategies such as Direct, CoT, and ReAct, and study how to route among them. The sufficient context lens of Joren et al. (2025) treats sufficiency as a property of query–context pairs and uses it to gate abstention. Its analysis of frozen hosts motivates treating support form as a separate decision about evidence delivery.

Both axes operate on whether or how deeply to retrieve. Retrieved content typically reaches the host as raw passages or a near-equivalent, leaving support form fixed. Adaptive-RAG (Jeong et al., 2024) is evaluated in Section 4, and a host-confidence Self-Routing heuristic is reported in Appendix C.8.

## A.3 SUPPORT-FORM ROUTING WITH A FIXED HOST

The third extrinsic axis selects whatform the external support takes, holding the backbone and the retrieval backend fixed. Prior tiered-memory systems already route between two support forms within memory hierarchies. TierMem (Zhu et al., 2026) organizes memory into a two-tier hierarchy of summaries and raw pages with a sufficiency router that escalates from summary to raw, reducing tokens by 54% on LoCoMo (Maharana et al., 2024) with minimal accuracy loss. MemPO (Li et al., 2026) and Memento (Zhou et al., 2025) apply reinforcement learning to optimize memory management, and PRIME (Wang & Jiang, 2026) builds experience libraries through iterative evolution without backbone training. A different angle is offered by advisor models (Asawa et al., 2025), which train a small policy that emits free-form natural-language steering instructions to a black-box host on a per-instance basis. TierMem routes between summary and raw but does not include the Direct option that bypasses retrieved support; advisor models deliver free-form hints rather than choosing among the fixed Direct, Summary, and Raw support forms studied here.

Jiang et al. (2025a) catalog the space of agent-supervised tool adaptation, in which a frozen agent supervises the training of a small downstream module via reward feedback, and place tiered-memory routers, advisor models, and similar systems in this family. Two additional architectural precedents matter for FORGE’s design, even though each was originally motivated by a different task. Jiang et al. (2025b) introduce s3, a PPO-trained T2 search agent that decouples the retriever from the generator and optimizes a Gain-Beyond-RAG reward; s3’s choice of on-policy PPO over a frozen host establishes that small reward-based routers can train on frontier-scale hosts, and we include it as a learned-routing baseline in Section 4. Sharma et al. (2025) introduce Sysformer, a transformerbased adapter that updates system prompts for frozen LLMs in the safety setting (refusal of harmful prompts), and establish that attention-based adapters can steer frozen hosts without weight access; we retarget their adapter template to our support-form alphabet as a transformer-based learned-routing baseline, acknowledging that this retargeting is a partial repurposing rather than a like-for-like comparison. FORGE studies a three-way Direct/Summary/Raw support choice around a frozen host, factorizes it with an optional thinking-depth choice (Section 2.3), and refines the routing policy against measured host feedback. Its distinction from TierMem is the additional Direct arm and joint decision studied under the stated single-step task protocol, not the invention of support-form routing itself. TierMem-style two-arm routing, s3’s PPO-trained searcher, and the retargeted Sysformer adapter are all included as experimental baselines in Section 4.

The three forms differ in prompt length and evidence content. FORGE compares their measured answer quality and token use under the specified utility; no information-bottleneck or rate-distortion optimality claim is needed for this empirical choice. A complementary line in agent context compression (Kang et al., 2025) condenses observation histories for long-horizon agents. FORGE studies the narrower, single-answer decision and does not evaluate long-horizon interactions.

## B IMPLEMENTATION DETAILS

All experiments were conducted on an HPC cluster (details anonymized for double-blind review). Local open-source hosts were served on 64 GB HBM GPU accelerators, with up to 128 accelerators running concurrently across 16 nodes at peak. Closed-source and API-served hosts were queried through the corresponding managed inference endpoints. The final experimental matrix spans 8 LLM backbones ranging from 7B to 671B parameters, 5 benchmarks, and 40 host–benchmark cells. Stage 1 supervised training is CPU-only and completes in under two minutes; Stage 2 GRPO refinement runs on a single 8-GPU node and completes in 8 to 12 hours per Pareto operating point.

## B.1 HARDWARE AND SOFTWARE ENVIRONMENT

Table S1 lists the hardware and software used in our experiments.

## B.2 FROZEN HOST CONFIGURATION

The host pool is partitioned into three groups by deployment mode and intrinsic capability (Table S2). Locally hosted models are loaded in float16 and sharded across eight HBM accelerators per node via device\_map="auto". Qwen3-8B-Instruct and Qwen3-8B-Thinking denote the same Qwen3-8B

Table S1: Hardware and software environment.
<table><tr><td>Category</td><td>Value</td></tr><tr><td>Cluster</td><td></td></tr><tr><td>System</td><td>HPC cluster (anonymized for review)</td></tr><tr><td>Node CPU</td><td>64-core x86_64</td></tr><tr><td>Node GPUs</td><td>8 × 64 GB HBM accelerators</td></tr><tr><td>Node memory</td><td>512 GB</td></tr><tr><td>Software stack</td><td></td></tr><tr><td>OS</td><td>Linux</td></tr><tr><td>GPU runtime</td><td>6.2.4 (vendor-specific)</td></tr><tr><td>Python</td><td>3.11.11</td></tr><tr><td>PyTorch</td><td>2.6.0 (with vendor-specific GPU backend)</td></tr><tr><td>Transformers</td><td>5.0.0</td></tr><tr><td>Sentence-Transformers</td><td>5.4.1</td></tr><tr><td>Datasets (HF)</td><td>4.0.0</td></tr><tr><td>scikit-learn</td><td>1.6.1</td></tr><tr><td>rank_bm25</td><td>0.2.2</td></tr></table>

checkpoint run with its thinking mode disabled and enabled, respectively. API models are accessed through managed inference endpoints without weight access. The cross-host API tables use cachedfeature evaluation; only Lite has a measured Fresh Online API example, and Full FORGE’s Fresh Online API latency is unmeasured.

Table S2: Frozen LLM backbones grouped by deployment mode and thinking capability.
<table><tr><td>Host</td><td>Family</td><td>Parameters</td><td>Precision</td><td>Deployment</td></tr><tr><td colspan="5">Non-thinking local hosts (main results, Table 1)</td></tr><tr><td>Qwen3-8B-Instruct (main)</td><td>Qwen 3</td><td>8B</td><td>float16</td><td>Local (8 accelerators)</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>Mistral</td><td>7B</td><td>float16</td><td>Local (8 accelerators)</td></tr><tr><td colspan="5">Thinking-capable hosts (composition experiment, Table 7)</td></tr><tr><td>Qwen3-8B-Thinking</td><td>Qwen 3</td><td>8B</td><td>float16</td><td>Local (8 accelerators)</td></tr><tr><td>Claude Sonnet 4</td><td>Anthropic</td><td></td><td></td><td>API</td></tr><tr><td>DeepSeek-R1</td><td>DeepSeek</td><td>671B MoE</td><td></td><td>API</td></tr><tr><td colspan="5">Frontier non-thinking hosts (cross-host transfer, Table 8)</td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>Llama 3.3</td><td>70B</td><td></td><td>API</td></tr><tr><td>DeepSeek-V3.2</td><td>DeepSeek</td><td>671B MoE</td><td></td><td>API</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>Qwen 3.5</td><td>397B MoE</td><td></td><td>API</td></tr></table>

Table S3 lists the decoding parameters. All non-thinking arms share identical decoding to isolate the effect of the action from decoding variability. Self-consistency (SC) probes use temperature sampling with a short per-call budget; the full online cost still includes all three samples and the greedy probe. Think-Low and Think-High set the host’s per-request reasoning budget (construction in Appendix B.3).

Table S3: Decoding and inference parameters across host groups.
<table><tr><td>Parameter</td><td>Local hosts</td><td>API hosts</td></tr><tr><td>max_new_tokens (arm answer)</td><td>64</td><td>64</td></tr><tr><td>max_new_tokens (SC sample)</td><td>24</td><td>24</td></tr><tr><td>Greedy temperature</td><td>0</td><td>0</td></tr><tr><td>SC temperature</td><td>0.7</td><td>0.7</td></tr><tr><td>SC top-p</td><td>0.9</td><td>0.9</td></tr><tr><td>SC samples  $K _ { \mathrm { s c } }$ </td><td>3</td><td>3</td></tr><tr><td>Think-Low budget (output tokens)</td><td>1,024</td><td>1,024</td></tr><tr><td>Think-High budget (output tokens)</td><td>4,096</td><td>4,096</td></tr><tr><td>API rate limit used</td><td></td><td>25 requests/min</td></tr></table>

## B.3 RETRIEVAL AND ACTION CONSTRUCTION

All actions share a fixed retrieval backend so that performance differences reflect routing rather than retriever tuning (Table S4). Each action $a = \left( \mu , \theta \right)$ selects support form $\mu$ and thinking setting θ.

Summary form construction. The Summary action is constructed deterministically without any auxiliary language model: BM25 retrieves the top-3 passages, all sentences within those passages are re-ranked by BM25 against the query, and the top-ranked sentences are concatenated until a budget of ∼140 words is reached. No second-stage LLM call or learned compressor is invoked for Summary, so its selected-answer input tokens are its host prompt length. This statement concerns support construction; Full FORGE’s pre-routing host probes are accounted for separately under the fresh-online protocol in Appendix B.9. We use fixed BM25 and deterministic extractive Summary. Learned compressors and other retrieval backends require separate training and evaluation.

Thinking-setting construction. NoThink uses plain decoding without a thinking instruction, and CoT-Prompt appends the suffix in Table S4; both settings are available on every host. On thinkingcapable hosts, Think-Low and Think-High set the host’s per-request reasoning budget to 1,024 and 4,096 output tokens, respectively; all other decoding parameters follow Table S3.

Table S4: Retrieval pipeline and action construction.
<table><tr><td>Component</td><td>Value</td></tr><tr><td>Retriever</td><td>BM25(rank_bm25)</td></tr><tr><td>Embedding model</td><td>BAAI/bge-base-en-v1.5 (768-d, normalized)</td></tr><tr><td>Tokenizer</td><td>lowercase + punctuation removal</td></tr><tr><td>Stopword list</td><td>57 common English stopwords</td></tr><tr><td>Top-k for Summary/Raw</td><td>3 passages</td></tr><tr><td>Support-form axis M</td><td></td></tr><tr><td>Direct  $( \mu = 0 )$ </td><td>Question only, 0 extra input tokens</td></tr><tr><td>Summary  $( \mu = 1 )$ </td><td>Query-aware sentence selection;  $\leq 1 4 0$  words</td></tr><tr><td>Raw  $( \mu = 2 )$ </td><td>Top-3 BM25 passages; ≤200 words per passage</td></tr><tr><td colspan="2">Thinking-depth axis Θ (thinking-capable hosts)</td></tr><tr><td>NoThink</td><td>Plain decoding; no thinking instruction</td></tr><tr><td>CoT-Prompt</td><td>“Let us think step by step” suffix</td></tr><tr><td>Think-Low</td><td>Reasoning budget 1,024 output tokens</td></tr><tr><td>Think-High</td><td>Reasoning budget 4,096 output tokens</td></tr></table>

## B.4 ROUTER ARCHITECTURE AND THREE-STAGE TRAINING

Full FORGE uses a factorized MLP with a 789 → 256 → 256 shared encoder, followed by support-form and thinking-depth heads (Section 3.1). FORGE-Lite retrains the router on the 778 host-independent input dimensions. Training proceeds in three stages: offline arm enumeration on the warm-start subset (Stage 0), supervised KL distillation against the Boltzmann target (Stage 1), and policy-guided group-relative refinement using host feedback on the expanded action alphabet (Stage 2). Table S5 lists all router-specific hyperparameters.

Stage-1 warm-start objective (full form). The warm-start router $\pi _ { \theta _ { 0 } }$ is trained by minimizing the KL divergence on the $\mu$ head against the Boltzmann target $\pi _ { \mathrm { B o l t z } } ^ { * }$ restricted to ${ \mathcal A } _ { \mathrm { w a r m } }$ at temperature τ<sub>0</sub>=1.0:

$$
{ \mathcal { L } } _ { \mathrm { w a r m } } ( \theta ) = { \frac { 1 } { N _ { \mathrm { w a r m } } } } \sum _ { i } D _ { \mathrm { K L } } \bigl ( \pi _ { \mathrm { B o l t z } } ^ { * } ( \cdot \mid x _ { i } ) \big \| \pi _ { \theta } ^ { \mu } ( \cdot \mid h ( x _ { i } ) ) \bigr ) ,\tag{5}
$$

with $\pi _ { \theta } ^ { \theta }$ initialized uniform. The target is the closed-form optimum of the stated entropy-regularized finite-action utility on ${ \mathcal A } _ { \mathrm { w a r m } } ;$ ; the finite-capacity Stage-1 router only approximates this target. The default warm-start uses three actions. The controlled Stage-2 comparison also evaluates an additional full-alphabet, six-action Stage-1 control under matched features, decoding, and cost weights; that control is distinct from the three-action warm-start row in the canonical table.

![](images/38677fae6266768f6765373e4f62ad8527856ad73ed74cde7491e5bcab5c0156.jpg)  
Figure S1: Full and Lite router deployment paths. Full uses four pre-routing host generations to construct its 11 host-derived features on a fresh query; Lite is retrained without them. Both variants select one joint action before the frozen host answers, and the host weights remain fixed.

Stage-2 clipped policy and auxiliary losses (full form). Let $\rho _ { b , g } ( \theta ) = \pi _ { \theta } ( a _ { b , g } \mid x _ { b } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { b , g } \mid$ $x _ { b } )$ be the displayed old-policy ratio. The clipped surrogate uses the PPO-style construction of Schulman et al. (2017) with the group-relative advantage $\hat { A } _ { b , g }$ of Shao et al. (2024) (Eq. 4):

$$
\mathcal { L } _ { \mathrm { p o l } } ( \theta ) = - \frac { 1 } { B G } \sum _ { b , g } \operatorname* { m i n } \Bigl ( \rho _ { b , g } \hat { A } _ { b , g } , \ \exp ( \rho _ { b , g } , 1 - \eta , 1 + \eta ) \hat { A } _ { b , g } \Bigr ) ,\tag{6}
$$

with $\eta = 0 . 2 .$ . The action groups are stratified to cover support forms when $G \geq | { \mathcal { M } } |$ (Table S5). This changes the behavior proposal to a distribution $q ( a \mid x )$ that need not equal $\pi _ { \theta _ { \mathrm { o l d } } } ( a \mid x )$ . Without a proposal correction involving $q ,$ the displayed ratio does not make Eq. 6 an unbiased policy-gradient estimator for the old policy. We therefore treat Stage 2 as a policy-guided, group-relative clipped surrogate aligned with measured utility, not an exact gradient step toward or convergence guarantee for the Boltzmann target. The total Stage-2 loss is ${ \mathcal { L } } _ { \mathrm { G R P O } } = { \mathcal { L } } _ { \mathrm { p o l } } + { \mathcal { L } } _ { \mathrm { K L } } + { \mathcal { L } } _ { \mathrm { e n t } }$ with the auxiliary terms

$$
{ \mathcal L } _ { \mathrm { K L } } ( \theta ) = \beta \cdot \frac { 1 } { B } \sum _ { b } D _ { \mathrm { K L } } \bigl ( \pi _ { \theta } ( \cdot \mid x _ { b } ) \big \| \pi _ { \theta _ { 0 } } ( \cdot \mid x _ { b } ) \bigr ) ,\tag{7}
$$

$$
\mathcal { L } _ { \mathrm { e n t } } ( \theta ) = - \alpha \cdot \frac { 1 } { B } \sum _ { b } H \bigl ( \pi _ { \theta } ( \cdot \mid x _ { b } ) \bigr ) .\tag{8}
$$

The KL anchor keeps the clipped update close to the Stage-1 warm-start reference $\pi _ { \boldsymbol { \theta } _ { 0 } } ;$ the entropy bonus discourages premature collapse on the larger alphabet. Optimization uses AdamW at learning rate $1 \times 1 0 ^ { - 5 }$ with $K _ { \mathrm { i n n e r } } = 4$ minibatch updates per rollout before $\pi _ { \theta _ { \mathrm { o l d } } }$ is refreshed. The KL coefficient β follows an adaptive schedule: it is multiplied by 1.5 every 100 steps when the observed per-query KL exceeds 0.05, and divided by the same factor when it falls below 0.005, so that the policy neither drifts off the warm-start manifold nor stagnates on it. Stage 0 costs $| \mathcal { A } _ { \mathrm { w a r m } } | \cdot N _ { \mathrm { w a r m } }$ ≈ 3 · 1500 = 4500 host calls (single node, under one hour); Stage 1 is CPU-only and completes in under two minutes. Each Stage-2 operating point uses $T _ { \mathrm { g r p o } } B G ^ { \mathbf { \bar { \mathbf { \alpha } } } } = 5 0 0 0 \times 3 2 \times 8 = 1$ ,280,000 atomic model completions, about 384M input and 82M output tokens in the measured resource accounting, and 8–12 hours on one 8-GPU node (64–96 GPU-hours). BGE and MLP routing take 0.5–5.5 ms in the measured settings, excluding host calls that acquire Full FORGE’s probes on unseen queries. Appendix B.9 includes those calls in Fresh Online token and latency accounting.

## B.5 FEATURE EXTRACTION

Full FORGE’s input $\phi _ { \mathrm { F u l l } } ( x ) \in \mathbb { R } ^ { 7 8 9 }$ combines six feature groups (Table S6). Eleven dimensions depend on a greedy host probe and three self-consistency samples, so an unseen fresh-online query requires four pre-routing host calls before its routed answer. FORGE-Lite is retrained with the 778 host-independent dimensions only: 768 BGE embedding dimensions, four BM25 statistics, one query–top-passage cosine, and five query-structure features. It uses no host probe in training or inference. Structured features are standardized using training-set statistics.

The strong BGE+BM25 baseline uses 773 host-independent dimensions (768 BGE, four BM25, and one query–top-passage cosine) and matches the Stage-0 data, MLP capacity, optimizer, and development-set search budget. It has no query-structure features, host probes, or GRPO. Table S7 reports the canonical cached feature-and-training ladder. Its point estimates should not be combined with the nine-run means in the uncertainty analysis as though they were the same statistic.

Table S5: Router architecture and three-stage training hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Architecture</td></tr><tr><td>Shared encoder  $h ( x )$ </td><td>2-layer MLP, 789 → 256 → 256, ReLU, dropout 0.1</td></tr><tr><td>Support-form head  $\pi ^ { \mu }$ </td><td>Linear, 3 logits</td></tr><tr><td>Thinking-depth head  $\pi ^ { \theta }$ </td><td>Linear, |Θ| logits, input  $[ h ( x ) ; e ( \mu ) ]$  with  $e ( \mu ) \in \mathbb { R } ^ { 1 6 }$ </td></tr><tr><td>Total parameter count</td><td>~269K</td></tr><tr><td colspan="2">Stage 0: Offline arm enumeration</td></tr><tr><td>Warm-start alphabet  ${ \mathcal A } _ { \mathrm { w a r m } }$ </td><td> $\mathcal { M } \times \mathrm { \{ N o T h i n k \} } \mathrm { ( 3 \ a r m s ) }$ </td></tr><tr><td>Warm-start size  $N _ { \mathrm { w a r m } }$ </td><td>1,500 queries</td></tr><tr><td colspan="2">Stage 1: Supervised warm-start (KL distillation)</td></tr><tr><td>Soft-label temperature τ0</td><td>1.0</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Maximum epochs</td><td>50</td></tr><tr><td>Early-stopping patience</td><td>7 (dev accuracy)</td></tr><tr><td>Stage 2: policy-guided GRPO refinement</td><td></td></tr><tr><td colspan="2"></td></tr><tr><td>Full alphabet A</td><td>K=6 (non-thinking hosts) or K=12 (thinking hosts)</td></tr><tr><td>Rollout group size G</td><td>8</td></tr><tr><td>Batch size B (queries per step)</td><td>32</td></tr><tr><td>Inner gradient steps per rollout</td><td> $K _ { \mathrm { i n n e r } } = 4$ </td></tr><tr><td>PPO clip range η</td><td>0.2</td></tr><tr><td>KL anchor target</td><td>0.02 (per query, adaptive  $\beta )$ </td></tr><tr><td>KL anchor  $\beta$  initial</td><td>0.05; multiplied by 1.5 every 100 steps if KL out of band</td></tr><tr><td>Entropy bonus α Sampling</td><td>0.01 Stratified to ensure μ coverage when  $G \geq | { \mathcal { M } } |$ </td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Total update steps  $T _ { \mathrm { g r p o } }$ </td><td>5,000</td></tr><tr><td></td><td></td></tr><tr><td colspan="2">Pareto utility (default)</td></tr><tr><td>Input cost weight  $\lambda _ { \mathrm { i n } }$ </td><td>0.1 (swept in Table S14)</td></tr><tr><td>Output cost weight  $\lambda _ { \mathrm { o u t } }$ </td><td>0.2 (swept in Table S14)</td></tr></table>

Table S6: Feature groups for Full FORGE (789 dimensions) and FORGE-Lite (778 dimensions). Only Full uses the 11 host-dependent dimensions.
<table><tr><td>Group</td><td>Features</td><td>Dim</td></tr><tr><td>Query embedding (BGE)</td><td>Frozen sentence embedding</td><td>768</td></tr><tr><td>Query structural</td><td>length, WH flag, entity count, comparison flag, temporal flag</td><td>5</td></tr><tr><td>Retrieval</td><td>BM25 top-1 score, top-5 mean, score gap, score std</td><td>4</td></tr><tr><td>Host probe</td><td>answer length, output tokens, IDK flag, hedging flag, query overlap, numeric flag, confidence</td><td>7</td></tr><tr><td>Self-consistency</td><td>agreement, unique ratio, average length, greedy match</td><td>4</td></tr><tr><td>Cross-modal</td><td>query-top-1-passage cosine similarity</td><td>1</td></tr><tr><td>Full total</td><td>All six groups</td><td>789</td></tr><tr><td>Lite total</td><td>BGE + query structural + retrieval + cross-modal</td><td>778</td></tr></table>

## B.6 DATASET CONFIGURATION

Table S8 lists the train/test splits per host group. The original tables use a canonical split (datasampling seed 42), and cross-host transfer uses a matched canonical test subset. The uncertainty analysis combines three independent train/dev/test splits with three training seeds per split on both main local hosts. Its nine-run mean and standard deviation are distinct from canonical point estimates and paired-query bootstrap intervals on the canonical test split.

Table S7: Canonical cached F1 (%) for matched feature and training variants. The 773- dimensional BGE+BM25 baseline supplies a competitive reference. FORGE-Lite adds the five structural features without host calls; Full FORGE adds 11 host-dependent dimensions. All entries are point estimates on the canonical evaluation split, separate from the nine-run means.
<table><tr><td>Variant</td><td>Dimensions</td><td>Qwen3-8B</td><td>Mistral-7B</td></tr><tr><td>BGE+BM25-Hard</td><td>773</td><td>56.8</td><td>46.7</td></tr><tr><td>BGE+BM25-KL</td><td>773</td><td>57.5</td><td>47.5</td></tr><tr><td>FORGE-773 (three arms)</td><td>773</td><td>58.0</td><td>48.0</td></tr><tr><td>FORGE-773 (six arms)</td><td>773</td><td>58.4</td><td>48.6</td></tr><tr><td>FORGE-Lite</td><td>778</td><td>58.7</td><td>49.1</td></tr><tr><td>Full FORGE</td><td>789</td><td>59.5</td><td>50.1</td></tr></table>

Table S8: Per-benchmark sample sizes for the canonical split (data-sampling seed 42). The additional nine-run uncertainty analysis uses three independent splits and three training seeds per split.
<table><tr><td>Benchmark</td><td>Train (local)</td><td>Test (local)</td><td>Test (transfer)</td></tr><tr><td>HotpotQA (distractor)</td><td>1,500</td><td>300</td><td>75</td></tr><tr><td>2WikiMultiHopQA</td><td>1,500</td><td>300</td><td>75</td></tr><tr><td>MuSiQue</td><td>1,500</td><td>300</td><td>75</td></tr><tr><td>PopQA</td><td>1,500</td><td>300</td><td>75</td></tr><tr><td>FEVER</td><td>1,500</td><td>300</td><td>75</td></tr></table>

## B.7 ASSET LICENSES AND ATTRIBUTION

All datasets, frozen language models, and software libraries used in this work are credited to their original creators, accessed under their original licenses, and used in a manner consistent with their intended research use. Table S9 summarizes the assets and their licenses; the underlying datasets and model weights are not redistributed with this work.

Table S9: Licenses for existing assets used in this work. Datasets are accessed via their original release channels; frozen LLMs are accessed via their published weights or provider APIs.
<table><tr><td>Asset</td><td>Type</td><td>License / Terms</td><td>Citation</td></tr><tr><td>HotpotQA (distractor)</td><td>Dataset</td><td>CC BY-SA 4.0</td><td>Yang et al. (2018)</td></tr><tr><td>2WikiMultiHopQA</td><td>Dataset</td><td>Apache 2.0</td><td>Ho et al. (2020)</td></tr><tr><td>MuSiQue</td><td>Dataset</td><td>CC BY 4.0</td><td>Trivedi et al. (2022)</td></tr><tr><td>PopQA</td><td>Dataset</td><td>MIT</td><td></td></tr><tr><td>FEVER</td><td>Dataset</td><td>CC BY-SA 3.0</td><td></td></tr><tr><td>Wikipedia (retrieval corpus)</td><td>Corpus</td><td>CC BY-SA 4.0</td><td></td></tr><tr><td>Qwen3-8B-Instruct / -Thinking</td><td>Frozen LLM</td><td>Apache 2.0</td><td></td></tr><tr><td>Qwen3.5-397B</td><td>Frozen LLM</td><td>Apache 2.0</td><td></td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>Frozen LLM</td><td>Apache 2.0</td><td></td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>Frozen LLM</td><td>Llama 3.3 Community</td><td></td></tr><tr><td>DeepSeek-V3.2 / -R1</td><td>Frozen LLM</td><td>MIT</td><td></td></tr><tr><td>Claude Sonnet 4</td><td>Frozen LLM (API)</td><td>Anthropic API terms</td><td></td></tr><tr><td>BGE-base-en-v1.5</td><td>Embedder</td><td>MIT</td><td>Xiao et al. (2024)</td></tr><tr><td>rank_bm25</td><td>Library</td><td>Apache 2.0</td><td></td></tr><tr><td>PyTorch / HuggingFace Transformers</td><td>Library</td><td>BSD / Apache 2.0</td><td></td></tr></table>

## B.8 COMPUTE SCALE

Table S10 summarizes the overall compute footprint. At peak, 128 GPU accelerators ran concurrently across 16 nodes for Stage 0 arm enumeration on the local backbones. Stage 2 GRPO refinement uses a single 8-GPU node per Pareto operating point. API-based experiments with frontier proprietary models were executed on externally managed infrastructure and incur no local cluster compute time. Stage 2 is a one-time source-host training cost, not a target-host cost incurred on each transfer query. In the matched six-action, nine-run comparison, it adds 1.8 macro-F1 points on

Qwen3-8B and 1.7 on Mistral-7B over the corresponding Stage-1 checkpoints, while reducing cached selected-answer tokens by 7.1% and 8.1%, respectively. The larger three-arm-to-six-arm difference also includes action-alphabet expansion and must not be attributed wholly to GRPO. Skipping Stage 2 avoids rollout cost but leaves Full FORGE’s Fresh Online host probes.

Table S10: Compute resources and experiment counts.
<table><tr><td>Resource or artefact</td><td>Value</td></tr><tr><td>Cluster nodes used (unique)</td><td>16+</td></tr><tr><td>Peak concurrent GPU accelerators</td><td>128</td></tr><tr><td>SLURM allocations active</td><td>2 institutional allocations</td></tr><tr><td>Stage 0 host calls (per local backbone)</td><td> $\mathord { \sim } 4 , 5 0 0 \left( 3 \right.$  arms × 1,500 queries)</td></tr><tr><td>Stage 2 completions (per Pareto point)</td><td>1.28M (T · B · G at T=5000, B=32, G=8)</td></tr><tr><td>Stage 2 input / output tokens (per point)</td><td>About 384M / 82M</td></tr><tr><td>Stage 2 wall-clock (Qwen3-8B local, per Pareto point)</td><td>8 to 12 hours on 8-GPU node</td></tr><tr><td>Stage 2 local GPU-hours (per point) API calls issued (transfer + thinking)</td><td>64 to 96 GPU-hours</td></tr><tr><td>Distinct LLM backbones evaluated</td><td>~22,000</td></tr><tr><td>Parameter range</td><td>8</td></tr><tr><td>Benchmarks evaluated</td><td>7B-671B</td></tr><tr><td>Host × benchmark cells</td><td>5</td></tr><tr><td></td><td>40</td></tr></table>

The matched six-action budget trajectory separates GRPO refinement from action expansion. At 0, 2,500, and 5,000 updates, Qwen3-8B reaches 57.6, 58.9, and 59.4 macro-F1 with 0.255, 0.244, and 0.237k cached selected-answer tokens per query. Mistral-7B reaches 48.3, 49.5, and 50.0 macro-F1 with 0.272, 0.259, and 0.250k tokens. These improvements require the one-time training expenditure above; they do not account for Full FORGE’s Fresh Online probes on later queries.

## B.9 INFERENCE LATENCY

Table S11 isolates local feature and router computation. The frozen BGE encoder dominates this local component, with little MLP time, but the table does not include the host executions required to populate Full FORGE’s 11 host-dependent features on a fresh query. Main-text Table 2 reports end-to-end Fresh Online latency with those calls included.

Table S11: Local routing latency (ms/query). Median over 1,000 HotpotQA-style queries.
<table><tr><td>Component</td><td>CPU GPU (1 thread) (batch 1)</td><td></td><td>GPU (batch 32, amort.)</td></tr><tr><td>BGE-base encoder (frozen)</td><td>27.4</td><td>4.9</td><td>0.39</td></tr><tr><td>Structured feature extraction</td><td>0.6</td><td>0.6</td><td>0.05</td></tr><tr><td>FORGE MLP head  $( \pi ^ { \mu } , \pi ^ { \theta } )$ </td><td>1.1</td><td>0.05</td><td>0.01</td></tr><tr><td>Total routing decision</td><td>29.1</td><td>5.5</td><td>0.45</td></tr><tr><td>Self-consistency probe  $( K _ { \mathrm { s c } } { = } 3$  host samples)</td><td></td><td></td><td>90.0</td></tr><tr><td>Host answer inference (Qwen3-8B, 64 out tokens)</td><td></td><td>122.5</td><td>78.4</td></tr></table>

Mean query length is 18 tokens. The total routing-decision row covers local computation only. On a fresh Full FORGE query, three self-consistency samples and one greedy probe require host calls; the self-consistency row is an amortized batch-32 component, not complete Fresh Online overhead. Host answer inference is shown for scale.

In the cached-feature setting, a previously processed query can reuse the four Full FORGE probe outputs, leaving only the local routing computation before the selected answer. A new query in Fresh Online use must acquire those features: one greedy probe and three self-consistency samples precede the routed answer, for five host calls in total. FORGE-Lite retrains on host-independent features and needs only one host call for the routed answer.

Full FORGE trades much higher latency for its additional F1 on these local hosts: Qwen3-8B’s p50 increases from 122.5 to 271.4 ms relative to Always-Raw. Lite obtains a smaller F1 gain with p50 near Always-Raw. On the measured DeepSeek-V3.2 API example, zero-shot Lite obtains 63.1 F1, 0.182k tokens, and about 2.2 s end-to-end latency versus Always-Raw’s 63.4 F1, 0.287k tokens, and about 2.1 s. This is a token-saving trade-off with slightly lower F1 and higher latency, not a three-axis Pareto improvement. Fresh Online Full FORGE latency was not measured on API hosts.

Cost-accounting note. Cached-result tables count selected-answer tokens and exclude earlier feature-acquisition calls, so they do not represent the end-to-end cost of a new query. Fresh Online evaluation also counts Full FORGE’s greedy and self-consistency probes in tokens and latency; Lite has no host-dependent calls. Summary uses no auxiliary LLM call (Section B.3), and its prompt tokens are included in both cost-accounting protocols.

## B.10 REPRODUCIBILITY

Seed 42 identifies the canonical data split used for the point-estimate tables. The uncertainty analysis on Qwen3-8B and Mistral-7B instead evaluates three independent train/dev/test splits with three router-training seeds per split (nine runs per method and host). Each tunable baseline is retuned on its corresponding dev split with the same search budget. Nine-run standard deviations summarize split and initialization variation; paired-query bootstrap confidence intervals compare methods on the canonical test split and are not nine-run intervals. The bootstrap resampling unit is the matched test query. The Oracle (6-arm) reference rows are enumerated references that use the observed outcomes of all six arms for each query. They are not deployable policies and are reported as descriptive references rather than F1 or EM upper bounds. Exact reproduction requires model outputs, Stage-0 enumeration tables, Stage-1 warm-start checkpoints, and Stage-2 GRPO checkpoints in addition to the method settings reported here. Given pre-enumerated arm outputs, the reported Stage-1 supervised pipeline completes on one CPU in under two minutes. Stage-2 reproduction additionally requires the host endpoints used for rollout.

Hyperparameter selection. The Pareto utility’s cost weights $( \lambda _ { \mathrm { i n } } , \lambda _ { \mathrm { o u t } } ) = ( 0 . 1 0 , 0 . 2 0 )$ and the Boltzmann temperature $\tau = 1 . 0$ are selected once on the Qwen3-8B HotpotQA dev set (N=500) and held fixed across the original host and benchmark comparisons. These weights specify a study operating point, not a universal price for user value. Table S14 reports a canonical cached sensitivity grid; its F1 range is 1.2 points on the non-thinking host and 2.5 points on the thinking-capable host. Baselines with tunable scalars receive the same dev-budget tuning protocol on the same dev split: the BM25-Threshold rule’s two thresholds are grid-searched, Adaptive-RAG’s complexity-classifier threshold is calibrated, and the Sysformer adapter rank is selected from {4, 8, 16}. In cross-host transfer (Section 4.6), λ and τ are not retuned per target host and no target-host Stage 2 training is run. Zero-shot describes the transferred weights, not the absence of Full FORGE’s target-host feature probes on a fresh query; the API transfer tables use cached-feature accounting.

## B.11 SUPPLEMENTARY FIGURES

This subsection collects the schematic of the three extrinsic axes of adaptive inference (Figure S2).

## C ADDITIONAL ABLATIONS

This appendix collects ablations referenced from the main text but deferred for space.

## C.1 STAGE-2 TRAINING AND KL-ANCHOR ABLATIONS

Table S12 puts the Stage-2 algorithm comparison and KL-anchor sweep on the same two benchmarks and cost scale. The shared three-arm Stage-1 row is a pipeline reference rather than a same-alphabet GRPO control; Panel A compares six-arm refinements, and Panel B varies the GRPO anchor around the default row in Panel A. The matched six-arm, nine-run comparison reported above isolates Stage 2 from alphabet expansion. GRPO and Dr. GRPO differ by at most 0.2 F1 in the displayed point estimates on the two benchmarks. Both exceed DPO and RLOO here, but this table alone does not establish a statistical tie or measure the matched six-action Stage-2 gain across all five tasks. That controlled comparison gives +1.8 and +1.7 macro-F1 on Qwen3-8B and Mistral-7B, respectively, at 1.28M model completions per operating point (Appendix B.8). GRPO is the reported primary configuration; the comparisons among DPO, RLOO, Dr. GRPO, and GRPO apply to this implementation and tested action alphabets.

![](images/114a1e3fdcb6ccf53659075026138b8249b1591a314344c949fa2e1061371fe5.jpg)  
Figure S2: Three extrinsic axes of adaptive inference for frozen agents. Model routing (Chen et al., 2024; Ding et al., 2025) and retrieval routing (Jeong et al., 2024; Asai et al., 2024) have been studied at length. Support-form routing, the third axis, is the focus of this paper. An orthogonal intrinsic axis exists on hosts that expose a reasoning budget (OpenAI et al., 2024; Guo et al., 2025), and our framework composes with it where available.

Table S12: Stage-2 refinement and KL-anchor ablations on Qwen3-8B. Canonical Cached F1 (%) and selected-answer token cost C<sup>¯</sup> (k) on HotpotQA and MuSiQue. Panel A compares six-arm refinement methods from the same Stage-1 checkpoint. Panel B varies $\beta$ for six-arm GRPO; the highlighted GRPO row in A is its default adaptive β=0.05 point. The three-arm Stage-1 row is shown once as a shared pipeline reference, not a matched six-arm β=∞ limit.
<table><tr><td rowspan="2">Configuration</td><td colspan="2">HotpotQA</td><td colspan="2">MuSiQue</td><td rowspan="2">Avg F1↑</td></tr><tr><td>F1↑</td><td>Č↓</td><td>F1↑</td><td>↓</td></tr><tr><td>Shared three-arm pipeline reference Offline KL only (FORGE (Stage 1))</td><td>58.3</td><td>0.272</td><td>20.8</td><td>0.295</td><td>39.6</td></tr><tr><td colspan="6">A. Stage-2 algorithm (six-arm alphabet)</td></tr><tr><td>DPO (Rafailov et al., 2023)</td><td>59.5</td><td>0.258</td><td>25.0</td><td></td><td></td></tr><tr><td>RLOO (Ahmadian et al., 2024)</td><td>59.2</td><td>0.263</td><td>24.6</td><td>0.265 0.272</td><td>42.3 41.9</td></tr><tr><td>Dr. GRPO (Liu et al., 2025)</td><td>60.6</td><td>0.235</td><td>26.5</td><td>0.249</td><td>43.6</td></tr><tr><td>GRPO (β=0.05, default)</td><td>60.4</td><td>0.236</td><td>26.4</td><td>0.252</td><td>43.4</td></tr><tr><td colspan="6"></td></tr><tr><td>B. GRPO KL-anchor coefficient β (six-arm alphabet) β = 0 (no anchor)</td><td>56.2</td><td>0.412</td><td>22.1</td><td>0.398</td><td>39.2</td></tr><tr><td>β = 0.01 (weak)</td><td>59.6</td><td>0.252</td><td>25.5</td><td>0.265</td><td>42.6</td></tr><tr><td>β = 0.20 (strong)</td><td>59.1</td><td>0.265</td><td>24.7</td><td>0.282</td><td>41.9</td></tr></table>

## C.2 FACTORIZED VERSUS FLAT POLICY HEAD

Table S13 compares the default factorized head $\pi _ { \boldsymbol { \theta } } ( a \mid x ) = \pi _ { \boldsymbol { \theta } } ^ { \boldsymbol { \mu } } ( \boldsymbol { \mu } \mid h ( x ) ) \cdot \pi _ { \boldsymbol { \theta } } ^ { \boldsymbol { \theta } } ( \boldsymbol { \theta } \mid h ( x ) , \boldsymbol { \mu } )$ against a flat K-way categorical head over the joint action $a = \left( \mu , \theta \right)$ , with feature vector and Stage-2 hyperparameters held fixed. At K=6 the displayed factorized-versus-flat difference is 0.6 F1; at K=12 it is 1.6 F1, with the reported parameter counts 269K versus 273K. These two tested settings are consistent with a benefit from conditioning thinking depth on support form, but do not establish a general scaling law in K. All main-paper results use the factorized head.

Table S14 shows a quality–cost trade-off rather than invariance to the cost weights: the non-thinking host spans 1.2 F1 points and the thinking-capable host spans 2.5 points across the displayed grid. The table uses cached selected-answer tokens; the Fresh Online token curves also include feature acquisition. Increasing $\lambda _ { \mathrm { i n } }$ shifts the policy toward cheaper input arms (Direct over Raw), and increasing $\lambda _ { \mathrm { o u t } }$ specifically suppresses thinking-budget arms on thinking-capable hosts. The default of (0.1, 0.2) was selected on a held-out development set and held fixed for all main results.

Table S13: Factorized vs flat policy head. FORGE F1 (%), average cost $\bar { C } \left( \mathbf { k } \right)$ , and policy parameter count under two architectures: a flat K-way categorical head over the joint action $a = \left( \mu , \theta \right)$ , and the default factorized head $\pi ^ { \mu } ( \mu \mid h ( x ) ) \cdot { \dot { \pi } } ^ { \theta } ( \theta \mid { \bf \bar { \theta } } h ( x ) , \mu )$ (Section 3.1). Reported on Qwen3-8B (K=6) and Qwen3-8B-Thinking (K=12).
<table><tr><td rowspan="2">Policy head</td><td colspan="2">Qwen3-8B(K=6)</td><td colspan="2"> $\mathtt { Q w e n 3 - 8 B - T h i n k i n g } ( K { = } 1 2 )$ </td><td rowspan="2"># params</td></tr><tr><td>F1↑</td><td> $\bar { C } \downarrow$ </td><td>F1↑</td><td> $\bar { C } \downarrow$ </td></tr><tr><td>Flat K-way categorical</td><td>59.8</td><td>0.241</td><td>63.2</td><td>0.715</td><td>273K</td></tr><tr><td>Factorized  $\pi ^ { \mu } \cdot \pi ^ { \theta | \mu }$  (default)</td><td>60.4</td><td>0.236</td><td>64.8</td><td>0.690</td><td>269K</td></tr></table>

Table S14: Canonical cached sensitivity to cost weights $\lambda _ { \mathrm { i n } }$ and $\lambda _ { \mathrm { o u t } } .$ . FORGE F1 (%) and selected-answer token cost C<sup>¯</sup> (k) on $\mathtt { H o t p o t Q A }$ for non-thinking Qwen3-8B and thinking-capable Qwen3-8B-Thinking. Across the surveyed points, F1 spans 1.2 and 2.5 points, respectively. The study default (0.1, 0.2) is highlighted; these weights are not universal prices.
<table><tr><td rowspan="2"> $\lambda _ { \mathrm { i n } }$ </td><td rowspan="2"> $\lambda _ { \mathrm { o u t } }$ </td><td colspan="2">Non-thinking host</td><td colspan="2">Thinking-capable host</td></tr><tr><td>F1↑</td><td> $\bar { C } \downarrow$ </td><td> $\operatorname { F } 1 \uparrow$ </td><td> $\bar { C } \downarrow$ </td></tr><tr><td>0.05</td><td>0.0</td><td>60.8</td><td>0.262</td><td>65.5</td><td>0.852</td></tr><tr><td>0.05</td><td>0.2</td><td>60.7</td><td>0.246</td><td>65.0</td><td>0.731</td></tr><tr><td>0.10</td><td>0.0</td><td>60.4</td><td>0.253</td><td>65.2</td><td>0.802</td></tr><tr><td>0.10</td><td>0.20 (default)</td><td>60.4</td><td>0.236</td><td>64.8</td><td>0.690</td></tr><tr><td>0.10</td><td>0.50</td><td>59.9</td><td>0.218</td><td>63.5</td><td>0.582</td></tr><tr><td>0.20</td><td>0.20</td><td>59.6</td><td>0.210</td><td>63.0</td><td>0.624</td></tr></table>

Table S15: Training-size sweep. HotpotQA F1 (%) for Stage 1 and Stage 1+2 on Qwen3-8B.
<table><tr><td> $N _ { \mathrm { w a r m } }$ </td><td>FORGE (Stage 1)</td><td>FORGE (Stage 1+2)</td></tr><tr><td>200</td><td>46.8</td><td>50.6</td></tr><tr><td>500</td><td>58.5</td><td>60.2</td></tr><tr><td>1,000</td><td>58.2</td><td>60.4</td></tr><tr><td>1,500 (default)</td><td>58.3</td><td>60.4</td></tr><tr><td>2,000</td><td>58.4</td><td>60.5</td></tr></table>

Table S15 shows empirical Stage-1 saturation around $N _ { \mathrm { w a r m } } = 5 0 0$ on this HotpotQA sweep; no finite-sample theorem here predicts that threshold. The Stage 1-to-Stage 1+2 difference ranges from +1.7 to +3.8 F1 across the displayed training sizes, and is +2.1 at the default $N _ { \mathrm { w a r m } } = 1 { , } 5 0 0$ This single-benchmark sweep does not isolate GRPO from action-alphabet changes in the main three-arm-to-six-arm comparison. Stage 2 needs rollout calls but no extra warm-start data.

## C.3 PER-ARM ALLOCATION CHANGES FROM THREE TO SIX ACTIONS

The original three-arm Stage-1 to six-arm final-policy comparison differs in both action availability and optimization; its +3.5 Qwen3-8B and +3.6 Mistral-7B macro-F1 differences are not GRPO-only effects. The matched six-action, nine-run comparison isolates the Stage-2 increment at +1.8 [1.3, 2.3] and +1.7 [1.2, 2.2] macro-F1, respectively (paired 95% confidence intervals on the canonical test split). Table S16 describes the canonical three-arm-to-six-arm allocation changes on Qwen3-8B. Its per-arm terms are descriptive accounting terms, computed as ∆alloc $\mathbf { \Sigma } _ { x } \cdot ( m _ { a } ^ { \mathrm { F 1 } } - m _ { \mathrm { r e f } } ^ { \mathrm { F 1 } } )$ for the selected query subsets; they should not be interpreted as controlled causal effects of GRPO. Here CoT-Prompt aggregates the three CoT-Prompt arms across support forms.

The largest descriptive term is the +2.9 associated with CoT-Prompt, which the three-arm warm-start policy cannot select. Direct and Summary shifts contribute +1.3 combined, while the Raw term is −0.7 in this accounting. These terms describe the changed allocations and query subsets; they do not partition the effect of GRPO from that of adding CoT-Prompt. The matched six-arm comparison above provides that controlled Stage-2 estimate.

Direct (FORGE argmax) Summary (FORGE argmax) Raw (FORGE argmax) CoT-Prompt (FORGE argmax)

Table S16: Descriptive per-arm accounting for Qwen3-8B’s three-arm Stage-1 to six-arm finalpolicy comparison. Allocations sum to 100% within each stage, and the displayed terms sum (up to rounding) to the observed +3.5 macro-F1 difference. The action alphabet also changes, so this is not the controlled GRPO-only estimate; Mistral-7B’s aggregate difference is +3.6.
<table><tr><td>Arm a</td><td>Stage-1 alloc. (%)</td><td>Final alloc. (%)</td><td>∆ alloc. (pp)</td><td>Descriptive F1 term</td></tr><tr><td>Direct</td><td>13</td><td>18</td><td>+5</td><td>+0.4</td></tr><tr><td>Summary</td><td>27</td><td>32</td><td>+5</td><td>+0.9</td></tr><tr><td>Raw</td><td>60</td><td>35</td><td>-25</td><td>-0.7</td></tr><tr><td>CoT-Prompt</td><td>0</td><td>15</td><td>+15</td><td>+2.9</td></tr><tr><td>Total</td><td>100</td><td>100</td><td></td><td>+3.5</td></tr></table>

## C.4 ROUTER BEHAVIOR IN BGE EMBEDDING SPACE

This subsection describes routing in BGE embedding space (Figure S3), and against the per-instance utility oracle (Figure S4). These plots show action allocation without identifying the cause of each answer. In these figures, CoT-Prompt aggregates the CoT-Prompt arms across support forms, and Direct, Summary, and Raw denote the corresponding NoThink arms.

Per-point structure in BGE embedding space. Figure S3 plots a 2D UMAP projection of the BGE-base sentence embeddings of the same N=1,500 held-out test queries, colouring each point by FORGE’s argmax action. Five visible cluster regions correspond to five query types derived from dataset-provided labels (Comparison, Factoid, Multi-hop 2–3 hops, Multi-hop 4+ hops, Verification); they emerge from the BGE embedding alone, before the router sees them. Most clusters are dominated by a single arm, and mixed colours concentrate near cluster boundaries. The regions are consistent with query-dependent routing in the displayed BGE projection, but the visualization alone cannot rule out memorization or establish why an individual action succeeds.

![](images/4b89aa889a77a0c8a8ed868c11b358dadf150047510271a00290445496d5f75d.jpg)  
Figure S3: 2D UMAP projection of held-out BGE query embeddings (N=1,500), coloured by FORGE’s argmax action on Qwen3-8B. The five visible clusters correspond to five query types derived from dataset labels (annotated near each cluster centroid). UMAP uses frozen BGE-base query embeddings, $n _ { \mathrm { n e i g h b o r s } } { = } 3 0$ , min\_dist=0.3, and cosine distance; centroid labels are illustrative.

Alignment with per-instance oracle. Figure S3 shows that FORGE’s routing is structured in embedding space but not yet that the structure is correct. Figure S4 places FORGE’s routing beside the per-instance utility oracle on the same UMAP projection. This diagnostic oracle evaluates all six actions and selects the one maximizing the stated F1-minus-token-cost utility; it is separate from the three-arm Stage-0 warm-start enumeration and from the tabular Oracle reference rows. FORGE selects its answer action without enumerating all arms, although Full FORGE still needs host-dependent feature probes on a fresh query. The two panels are nearly indistinguishable: FORGE matches the per-instance oracle on 87% of queries, with the residual 13% disagreement concentrated at the boundaries between clusters where multiple arms are near-tied in Pareto utility (visible as the small fraction of mismatched-colour points within each cluster, especially on the multi-hop 2–3-hop region where Raw and CoT-Prompt are near-tied for many queries). The 87% agreement is with the specified utility oracle; the UMAP plot does not explain the outcome of individual answers.

![](images/80470fd690481bb12e7b6a292617e785b303fceb3fb51cb99a7bdba84c889b41.jpg)  
Figure S4: FORGE’s argmax action agrees with the six-arm utility oracle on 87% of the same queries (N=1,500, Qwen3-8B). Left: the arm maximizing the stated F1-minus-token-cost utility after offline six-arm evaluation; this diagnostic is separate from the tabular Oracle reference rows and from three-arm Stage 0. Right: the trained router’s argmax without answer-arm enumeration. Coordinates and cluster centroids are shared. The residual 13% action disagreement is concentrated near the displayed multi-hop cluster boundary; this plot does not measure Fresh Online probe cost.

## C.5 CROSS-HOST TRANSFER: PER-TASK BREAKDOWN

Table 8 reports macro F1 for cross-host transfer; Table S17 adds per-task F1, Always-Direct (the parametric-only reference), and FORGE Stage 1 (the supervised warm-start ablation).

## C.6 EXACT MATCH (EM) SCORES

The main paper reports token-level F1 throughout for direct comparability with prior routing work. Table S18 reports EM on the same canonical cached predictions as Table 1. Its Oracle (6-arm) row is the enumerated reference of Table 1 scored under EM (Appendix B.10); it is a descriptive reference, not an EM upper bound. The relative ordering is largely preserved, with FORGE leading the non-oracle policies in macro EM on both hosts. EM/F1 ratios are roughly 0.78 on HotpotQA, 0.82 on 2Wiki, 0.65 on MuSiQue (multi-hop tolerates partial overlap), 0.87 on PopQA, and 0.96 on FEVER (three-way classification).

Table S17: Zero-shot cross-host transfer: per-task F1 and cached cost. A source-trained FORGE router is applied to three API hosts without target-host weight updates.
<table><tr><td rowspan="2">Target host</td><td rowspan="2">Policy</td><td colspan="5">F1 by task (%)</td><td rowspan="2">Avg F1↑</td><td rowspan="2"> $\begin{array} { r l } { \bar { C } \left( \mathrm { k } \right) \downarrow } & { { } \Delta _ { \mathrm { R a w } } ^ { F _ { 1 } } } \\ { \mathrm { s a v i n g } } & { { } \Delta _ { \mathrm { R a w } } ^ { F _ { 1 } } } \end{array}$ </td><td rowspan="2"></td></tr><tr><td>HQA</td><td>2Wiki</td><td>MuSiQue</td><td>PopQA</td><td>FEVER</td></tr><tr><td rowspan="6">Llama-3.3 70B</td><td>Direct</td><td>42.9</td><td>32.4</td><td>13.4</td><td>38.5</td><td>61.3</td><td>37.7</td><td>0.085 (-73%)</td><td>-20.8</td></tr><tr><td>Summary</td><td>57.6</td><td>37.1</td><td>20.5</td><td>90.3</td><td>74.7</td><td>56.0</td><td>0.214 (-33%)</td><td>-2.5</td></tr><tr><td>Raw</td><td>64.7</td><td>41.5</td><td>26.9</td><td>88.8</td><td>70.7</td><td>58.5</td><td>0.318 (ref.)</td><td>+0.0</td></tr><tr><td>Oracle</td><td>73.8</td><td>54.7</td><td>36.3</td><td>90.8</td><td>76.0</td><td>66.3</td><td>0.145 (-54%)</td><td></td></tr><tr><td>Stage 1</td><td>61.9</td><td>45.3</td><td>24.1</td><td>90.3</td><td>72.0</td><td> $5 8 . 7 \pm 0 . 4$ </td><td>0.215 (-32%)</td><td>+0.2</td></tr><tr><td>FORGE</td><td>64.0</td><td>47.8</td><td>27.2</td><td>91.5</td><td>75.5</td><td>61.2 ± 0.6</td><td>0.205 (-36%)</td><td>+2.7</td></tr><tr><td rowspan="6">DeepSeek V3.2</td><td>Direct</td><td>48.2</td><td>29.2</td><td>22.5</td><td>39.6</td><td>49.3</td><td>37.8</td><td>0.055 (-81%)</td><td>-25.6</td></tr><tr><td>Summary</td><td>58.4</td><td>43.3</td><td>28.3</td><td>94.8</td><td>88.0</td><td>62.6</td><td>0.184 (-36%)</td><td>-0.8</td></tr><tr><td>Raw</td><td>65.6</td><td>42.8</td><td>31.8</td><td>91.3</td><td>85.3</td><td>63.4</td><td>0.287 (ref.)</td><td>+0.0</td></tr><tr><td>Oracle</td><td>72.2</td><td>61.0</td><td>44.4</td><td>96.1</td><td>96.0</td><td>73.9</td><td>0.116 (-60%)</td><td></td></tr><tr><td>Stage 1</td><td>63.3</td><td>42.8</td><td>30.7</td><td>92.9</td><td>72.0</td><td>60.3 ± 0.4</td><td>0.183 (-36%)</td><td>-3.1</td></tr><tr><td>FORGE</td><td>65.4</td><td>45.5</td><td>33.5</td><td>94.0</td><td>78.5</td><td>63.4 ± 0.7</td><td>0.176 (-39%)</td><td>+0.0</td></tr><tr><td rowspan="6">Qwen3.5 397B</td><td>Direct</td><td>54.0</td><td>25.0</td><td>19.1</td><td>38.9</td><td>66.7</td><td>40.7</td><td>0.062 (-80%)</td><td>-22.4</td></tr><tr><td>Summary</td><td>58.7</td><td>44.8</td><td>23.4</td><td>95.6</td><td>88.0</td><td>62.1</td><td>0.196 (-36%) 0.305</td><td>-1.0</td></tr><tr><td>Raw</td><td>65.0</td><td>41.0</td><td>29.8</td><td>93.0</td><td>86.7</td><td>63.1</td><td>(ref.)</td><td>+0.0</td></tr><tr><td>Oracle</td><td>74.2</td><td>56.0</td><td>40.2</td><td>95.6</td><td>93.3</td><td>71.9</td><td>0.123 (-60%)</td><td></td></tr><tr><td>Stage 1</td><td>63.1</td><td>46.5</td><td>30.7</td><td>94.5</td><td>82.7</td><td> $6 3 . 5 \pm 0 . 4$ </td><td>0.209 (-31%)</td><td>+0.4</td></tr><tr><td>FORGE</td><td>65.5</td><td>49.0</td><td>33.0</td><td>94.5</td><td>86.0</td><td> $\mathbf { 6 5 . 6 \pm 0 . 5 }$ </td><td>0.198 (-35%)</td><td>+2.5</td></tr></table>

HQA abbreviates HotpotQA. Direct, Summary, and Raw are fixed policies. Each task has 75 canonical test queries. C<sup>¯</sup> counts selected-answer tokens (k/query); its second line is the reduction relative to Raw. $\Delta _ { \mathrm { R a w } } ^ { F _ { 1 } }$ compares macro F1 with Raw. Stage 1 and FORGE macro F1 are mean ± SD over three seeds on the fixed subset; per-task F1 and baseline scores are point estimates. The italic Oracle is an enumerated six-arm reference (Appendix B.10) and is excluded from practical-policy rankings. Bold marks the highest practical per-task score; boxed macro F1 marks FORGE, which ties Raw on DeepSeek-V3.2 at displayed precision. The router uses 2,000 source training and 500 development queries. Pre-routing host calls are excluded from cached costs.

Table S18: Exact Match on the canonical cached predictions of Table 1. Scores are percentages.
<table><tr><td rowspan="2">Host</td><td rowspan="2">Policy</td><td colspan="5">EM by task (%)</td><td rowspan="2">Avg EM↑</td></tr><tr><td>HQA</td><td>2Wiki</td><td>MuSiQue</td><td>PopQA</td><td>FEVER</td></tr><tr><td rowspan="10">Qwen3-8B</td><td>Direct</td><td>18.5</td><td>21.2</td><td>6.1</td><td>12.8</td><td>52.0</td><td>22.1</td></tr><tr><td>Summary</td><td>36.5</td><td>31.0</td><td>12.2</td><td>75.5</td><td>78.0</td><td>46.6</td></tr><tr><td>Raw</td><td>45.0</td><td>33.5</td><td>15.7</td><td>73.0</td><td>76.0</td><td>48.6</td></tr><tr><td>Oracle (6-arm)</td><td>53.5</td><td>43.8</td><td>22.6</td><td>76.5</td><td>86.0</td><td>56.5</td></tr><tr><td>Adaptive-RAG</td><td>44.0</td><td>33.0</td><td>15.0</td><td>69.5</td><td>74.5</td><td>47.2</td></tr><tr><td>TierMem-2arm</td><td>37.0</td><td>32.5</td><td>13.5</td><td>71.0</td><td>78.0</td><td>46.4</td></tr><tr><td>s3</td><td>43.0</td><td>32.0</td><td>14.0</td><td>68.0</td><td>72.0</td><td>45.8</td></tr><tr><td>Sysformer</td><td>45.0</td><td>32.0</td><td>13.7</td><td>75.0</td><td>73.0</td><td>47.7</td></tr><tr><td>Stage 1</td><td>45.5</td><td>31.5</td><td>13.5</td><td>76.0</td><td>72.5</td><td>47.8</td></tr><tr><td>FORGE</td><td>47.5</td><td>35.0</td><td>17.2</td><td>76.5</td><td>77.0</td><td>50.6</td></tr><tr><td rowspan="10">Mistral-7B</td><td>Direct</td><td>17.5</td><td>14.3</td><td>4.7</td><td>19.5</td><td>45.0</td><td>20.2</td></tr><tr><td>Summary</td><td>29.5</td><td>19.8</td><td>7.0</td><td>69.5</td><td>66.5</td><td>38.5</td></tr><tr><td>Raw</td><td>36.5</td><td>23.5</td><td>8.4</td><td>65.0</td><td>69.0</td><td>40.5</td></tr><tr><td>Oracle (6-arm)</td><td>47.0</td><td>35.0</td><td>14.0</td><td>72.5</td><td>78.5</td><td>49.4</td></tr><tr><td>Adaptive-RAG</td><td>35.5</td><td>22.0</td><td>9.0</td><td>63.5</td><td>67.0</td><td>39.4</td></tr><tr><td>TierMem-2arm</td><td>31.5</td><td>23.0</td><td>7.7</td><td>61.0</td><td>67.0</td><td>38.0</td></tr><tr><td>s3</td><td>35.5</td><td>21.5</td><td>8.3</td><td>61.5</td><td>64.0</td><td>38.2</td></tr><tr><td>Sysformer</td><td>37.0</td><td>19.0</td><td>8.5</td><td>65.0</td><td>66.0</td><td>39.1</td></tr><tr><td>Stage 1</td><td>37.5</td><td>19.0</td><td>8.7</td><td>66.5</td><td>67.0</td><td>39.7</td></tr><tr><td>FORGE</td><td>39.5</td><td>25.0</td><td>10.0</td><td>70.0</td><td>69.5</td><td>42.8</td></tr></table>

HQA abbreviates HotpotQA. Direct, Summary, and Raw are fixed policies. Adaptive-RAG (Jeong et al., 2024), TierMem-2arm (Zhu et al., 2026), s3 (Jiang et al., 2025b), and Sysformer (Sharma et al., 2025) are the comparison routers. The italic Oracle is the enumerated six-arm reference (Appendix B.10) and is excluded from the practical-policy ranking. Bold marks the best practical per-task EM (including ties); boxed macro EM marks the best practical policy on each host.

## C.7 NUMERICAL TABLE FOR THE ACTION ALPHABET SWEEP

The KL-anchor values appear alongside the Stage-2 algorithm comparison in Table S12, Panel B, with the default point in Panel A. Table S19 reports the action alphabet sweep. Adding CoT-Prompt $( K { = } 3 \to 6 )$ raises F1 while lowering cost, adding Think-Low $( K { = } 6 \to 9 )$ gives modest gains (+1.0 and +1.1 F1), and unlocking Think-High (K=9 → 12) gives the largest absolute improvement.

Table S19: Action alphabet size ablation. FORGE F1 (%) and cached selected-answer tokens C<sup>¯</sup> (k) on HotpotQA and MuSiQue as the alphabet grows from K=3 to K=12 on Qwen3-8B-Thinking.
<table><tr><td rowspan="2">K</td><td rowspan="2">Alphabet</td><td colspan="2">HotpotQA</td><td colspan="2">MuSiQue</td></tr><tr><td>F1↑</td><td>↓</td><td>F1↑</td><td>Č↓</td></tr><tr><td>3</td><td>{Direct, Summary, Raw} × {NoThink}</td><td>58.5</td><td>0.272</td><td>21.3</td><td>0.295</td></tr><tr><td>6</td><td> $A _ { \mathrm { n t } }$  (default non-thinking)</td><td>60.4</td><td>0.236</td><td>26.4</td><td>0.252</td></tr><tr><td>9</td><td> $A _ { \mathrm { n t } }$  ∪ Think-Low arms</td><td>61.4</td><td>0.380</td><td>27.5</td><td>0.420</td></tr><tr><td>12</td><td> $\mathcal { A } _ { \mathrm { t } }$  (full 12-arm)</td><td>64.8</td><td>0.690</td><td>33.5</td><td>0.760</td></tr></table>

## C.8 HEURISTIC ROUTING BASELINES

Two heuristic routing baselines were excluded from the main results in Table 1 for compactness. Self-Routing uses the host’s own confidence (a normalized log-probability of the Direct response) thresholded at the dev-tuned value to decide between Direct and Raw. FrugalGPT-Cascade (Chen et al., 2024) runs a Direct → Summary → Raw threshold cascade, advancing whenever the previous arm’s confidence falls below a dev-tuned threshold. Both favor inexpensive actions in the reported runs, but they do not have identical behavior: Self-Routing improves Mistral-7B macro F1 from 24.2 to 28.2 in Table S20. Its initial Direct confidence call, and the earlier calls in the cascade, must be included in any Fresh Online cost or latency comparison.

Table S20: Heuristic routing baselines on the main hosts. F1 (%) and cached selected-answer token cost C<sup>¯</sup> (k) on the five benchmarks of Table 1. Cached C<sup>¯</sup> omits preliminary Direct/confidence and cascade calls; thus this is not a Fresh Online comparison. Always-Direct is a reference row.
<table><tr><td>Host</td><td>Method</td><td>HotpotQA</td><td>2Wiki</td><td>MuSiQue</td><td>PopQA</td><td>FEVER</td><td>Avg F1</td><td> $\bar { C }$ </td><td> $\Delta _ { \mathrm { R a w } } ^ { F _ { 1 } }$ </td></tr><tr><td rowspan="3">Qwen3-8B</td><td>Always-Direct</td><td>26.2</td><td>25.7</td><td>10.2</td><td>15.0</td><td>54.4</td><td>26.3</td><td>0.040</td><td>-30.8</td></tr><tr><td>Self-Routing</td><td>27.0</td><td>27.6</td><td>11.1</td><td>18.2</td><td>56.1</td><td>28.0</td><td>0.058</td><td>-29.1</td></tr><tr><td>FrugalGPT-Cascade</td><td>26.2</td><td>25.7</td><td>10.2</td><td>16.5</td><td>55.0</td><td>26.7</td><td>0.042</td><td>-30.4</td></tr><tr><td rowspan="3">Mistral-7B</td><td>Always-Direct</td><td>25.2</td><td>17.7</td><td>7.9</td><td>23.0</td><td>47.0</td><td>24.2</td><td>0.041</td><td>-23.0</td></tr><tr><td>Self-Routing</td><td>32.4</td><td>24.2</td><td>11.1</td><td>24.0</td><td>49.5</td><td>28.2</td><td>0.060</td><td>-19.0</td></tr><tr><td>FrugalGPT-Cascade</td><td>25.3</td><td>17.9</td><td>8.0</td><td>23.5</td><td>47.5</td><td>24.4</td><td>0.043</td><td>-22.8</td></tr></table>

## D PROPERTIES OF THE ROUTING OBJECTIVE AND WARM START

These results characterize the finite-action offline target and show how the warm-start reference shapes it; they give no convergence or regret guarantee for the implemented optimizer.

## D.1 VALUE OF PER-QUERY ROUTING

Proposition 1 (Value of per-query routing). For integrable utilities, define $U ^ { * } ( x ) = \mathrm { m a x } _ { a \in \mathcal { A } } U ( x , a )$ as the best achievable per-query utility. Then

$$
\mathbb { E } _ { x } [ U ^ { * } ( x ) ] \ \geq \ \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { x } [ U ( x , a ) ] .
$$

It is strict if every action a is suboptimal on a set of queries with positive probability.

Proof. For each a and x, $U ^ { * } ( x ) - U ( x , a ) \geq 0$ . Taking expectations gives the weak inequality. Under the stated positive-probability condition, each nonnegative difference has strictly positive expectation. Since A is finite, the inequality is strict against the best fixed action. □

## D.2 A FIXED-REFERENCE REGULARIZED OBJECTIVE

The Stage 2 loss includes both an entropy bonus and a KL penalty to the warm-start policy. Their effect can be characterized for an idealized, unrestricted categorical policy with fixed coefficients.

Proposition 2 (Idealized regularized policy). Fix x, afull-support reference policy $\pi _ { 0 } ( \cdot \mid x )$ , and coefficients $\alpha , \beta \geq 0$ with $\alpha + \beta > 0$ . The unique maximizer over $\pi ( \cdot \mid x ) \in \Delta ( { \mathcal { A } } )$ of

$$
\sum _ { a } \pi ( a \mid x ) U ( x , a ) - \beta D _ { \mathrm { K L } } \big ( \pi ( \cdot \mid x ) \parallel \pi _ { 0 } ( \cdot \mid x ) \big ) + \alpha H \big ( \pi ( \cdot \mid x ) \big )
$$

is

$$
\pi ^ { \dagger } ( a \mid x ) = { \frac { \pi _ { 0 } ( a \mid x ) ^ { \beta / ( \alpha + \beta ) } \exp \bigl ( U ( x , a ) / ( \alpha + \beta ) \bigr ) } { \sum _ { a ^ { \prime } } \pi _ { 0 } ( a ^ { \prime } \mid x ) ^ { \beta / ( \alpha + \beta ) } \exp \bigl ( U ( x , a ^ { \prime } ) / ( \alpha + \beta ) \bigr ) } } .\tag{9}
$$

Proof. Expanding the KL divergence and entropy shows that the objective equals $\left( \alpha + \beta \right)$ log $Z ( x ) -$ $( \alpha + \bar { \beta } ) \bar { D _ { \mathrm { K L } } } ( \pi ( \cdot \cdot | \ x ) \| \pi ^ { \dagger } ( \cdot \cdot | \ x ) )$ , where $Z ( x )$ is the denominator in Eq. (9). The KL divergence is nonnegative and vanishes only when the two policies coincide, proving a unique optimum. □

When $\beta = 0 ,$ Eq. (9) reduces to Eq. (2) with $\tau = \alpha$ . A nonuniform reference generally changes the solution when $\beta > 0$ . The proposition describes the exact objective with fixed coefficients and known utilities. Stage 2 instead uses sampled, group-standardized rewards, a clipped policy surrogate, a finite router, and an adaptive KL coefficient; the proposition gives no convergence or regret guarantee for the implemented Stage 2 optimization procedure.

## D.3 WARM-START KL DECOMPOSITION

Stage 1 fits the support-form marginal on the enumerated warm-start actions. For the full action set, the thinking-depth head assigns equal initial probability to every thinking option.

Proposition 3 (Warm-start KL decomposition). Fix x. Let π<sup>∗</sup> be thefull-action distribution ofEq. (2), with support-form marginal $\pi _ { \mu } ^ { \ast }$ and conditional thinking policy $\pi _ { \theta | \mu } ^ { * } .$ . Suppose the warm-start policy factorizes as $\pi _ { \theta _ { 0 } } ( \mu , \theta \mid x ) = \dot { \pi } _ { \theta _ { 0 } } ^ { \mu } ( \mu \mid x ) / | \Theta |$ , with $\pi _ { \theta _ { 0 } } ^ { \mu } ( \mu \mid x ) > 0 { \ddot { f } } o r$ all $\mu .$ . Then

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } \bigl ( \pi ^ { * } ( \cdot \mid x ) \bigr \| \pi _ { \theta _ { 0 } } ( \cdot \mid x ) \bigr ) = D _ { \mathrm { K L } } \bigl ( \pi _ { \mu } ^ { * } ( \cdot \mid x ) \| \pi _ { \theta _ { 0 } } ^ { \mu } ( \cdot \mid x ) \bigr ) } \\ & { \qquad + \log \mid \Theta \mid - \mathbb { E } _ { \mu \sim \pi _ { \mu } ^ { * } ( \cdot \mid x ) } \bigl [ H ( \pi _ { \theta \mid \mu } ^ { * } ( \cdot \mid x , \mu ) ) \bigr ] . } \end{array}\tag{10}
$$

The contribution from the conditional thinking policy lies in [0, log |Θ|].

Proof. The KL chain rule splits the joint divergence into a divergence between support-form marginals and the $\pi _ { \mu } ^ { * }$ -weighted conditional divergence. For each $\mu , D _ { \mathrm { K L } } ( \pi _ { \theta | \mu } ^ { * } |$ Uniform $( \Theta ) ) = \log | \mathbf { \bar { \Theta } } | -$ $H ( \pi _ { \theta | \mu } ^ { * } )$ . Substitution gives Eq. (10); the interval follows from $0 \leq \ddot { H } ( \pi _ { \theta \mid \mu } ^ { * } ) \leq \log | \Theta |$ □

This identity separates support-form marginal mismatch from mismatch due to a uniform thinking head. It bounds only the latter; the marginal term can be larger. The identity describes initialization mismatch and does not predict the empirical F1 improvement from Stage 2.

## E INTERPRETATION OF THE COST–QUALITY SCORE

Equation (1) is a weighted-sum scalarization of measured F1 and normalized token costs. Its weights select an operating point according to the user’s relative valuation of these quantities; performance away from the reported weight sweep is an empirical question. Without the entropy term, maximizing expected utility over all categorical policies chooses a utility-maximizing action for each query (with arbitrary mixing among ties). The Shannon-entropy regularizer yields the soft labels in Eq. (2).

The measured token count in this formulation is an operational resource cost. It is not the mutualinformation rate in Shannon’s rate–distortion function, so the cost-quality scalarization alone does not constitute a rate–distortion derivation of the Boltzmann policy.

## F LIMITATIONS AND FUTURE WORK

## F.1 LIMITATIONS

The scope of this paper is knowledge-intensive QA and fact verification, the task families on which the Direct/Summary/Raw alphabet is most directly meaningful; reasoning-only benchmarks (e.g., MATH, GSM8K, AIME) where retrieval is rarely useful would require a different action alphabet (e.g., chain-of-thought depth or tool calls) and are not evaluated here. Stage 0 enumeration cost remains linear in $| \bar { \mathcal { A } } _ { \mathrm { w a r m } } |$ , which constrains how rich the warm-start alphabet can be at training time; Stage 2 GRPO removes this constraint for the full alphabet A but introduces a host-call rollout budget of roughly $T _ { \mathrm { g r p o } } \cdot B \cdot G$ per Pareto operating point. FORGE is trained against a fixed retrieval backend, so joint optimization of retrieval and routing is beyond the scope of this work. In addition, the current router relies on compact query-level and retrieval-derived features as input; for document-understanding tasks where long-range context, layout structure, or cross-document interactions are central, and especially for multimodal settings involving image-text evidence, a context-aware or multimodal router may be needed to fully capture the support requirements. On tasks where a single support form uniformly dominates, notably FEVER on non-thinking hosts where Summary ≈ Raw for 96% of queries, the general-purpose router does not improve on a task-specific fixed policy; on thinking-capable hosts this gap closes once the thinking-arm composition restores routing dynamic range. Transfer accuracy is sensitive to the capability gap between source and target hosts: a source-trained router’s support-form allocation can mismatch the target host’s parametric knowledge, as observed on FEVER with DeepSeek-V3.2, where FORGE trails both fixed Summary and fixed Raw. Stage 0 arm enumeration and Stage 2 training require labeled queries to compute the F1 reward, although inference with a trained router does not require labels. An LLM judge or verifier could supply proxy rewards for unlabeled adaptation, pending validation against true answer quality.

## F.2 FUTURE WORK

Three directions follow from the current formulation. A target-side calibration step that updates the last linear layer from a small labelled subset of the target host is a natural remedy for the source-target capability-gap limitation above. Coupling support-form routing with retrieval or model routing would enlarge the discrete action alphabet; its training cost and empirical value remain to be tested. Extending Stage 2 to a sequential cascade, in which the router decides whether to escalate from Direct to Summary to Raw based on intermediate host responses, would require accounting for the extra host calls and evaluating the resulting policy separately. Replacing BM25 with dense (DPR or Contriever) or hybrid sparse-dense retrieval changes the upstream retriever, while M still describes what enters the host prompt. Comparing these backends could test the scope of FORGE’s improvements in answer quality and token efficiency.

## F.3 BROADER IMPACTS

FORGE changes which evidence and reasoning setting a frozen host receives, so its quality and cost effects depend on the task, host, and feature-acquisition protocol. The cached-feature comparisons measure routed-answer tokens after Full FORGE’s host-dependent features have been collected. In the measured Fresh Online Qwen3-8B setting, Full FORGE improves F1 from 57.1 to 59.5 and reduces host tokens from 0.430k to 0.384k per query relative to Always-Raw, but its four pre-routing host calls raise median latency from 122.5 to 271.4 ms. FORGE-Lite avoids those calls and reaches 58.7 F1 at 0.242k tokens and 128.0 ms. These measurements show a deployment trade-off; token counts alone do not establish energy savings or lower end-to-end cost for every provider.

The cost weights $( \lambda _ { \mathrm { i n } } , \lambda _ { \mathrm { o u t } } )$ encode a chosen quality–cost preference. More aggressive cost weighting can select less evidence and may worsen answers or remove the grounding needed for attribution. A strong fixed form can also be preferable on a particular task: on FEVER with a non-thinking host, the Summary-pinned baseline slightly exceeds the general router’s F1. In settings where citations or other grounding requirements matter, they should be measured directly and included as deployment criteria rather than assumed to follow from F1 or a change in cost weights. Although the host weights remain frozen, changing its inputs can change its errors; we have not evaluated whether routing amplifies or reduces bias, hallucination, or misuse risks outside the reported tasks.

## G USE OF LARGE LANGUAGE MODELS

An AI assistant helped revise the manuscript’s prose, organization, and internal consistency, including reviewing theoretical claims for assumptions and scope. The authors are responsible for verifying the technical claims, references, and empirical results in the submitted version.