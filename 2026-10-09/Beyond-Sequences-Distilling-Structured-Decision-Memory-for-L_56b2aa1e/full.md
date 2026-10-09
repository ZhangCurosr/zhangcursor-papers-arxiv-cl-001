# Beyond Sequences: Distilling Structured Decision Memory for LLM Recommendation

Alibaba Group

Hangzhou, China

{liangleikun.llk, guoshui.wgs, xingsheng.hxs, hanyushan.hys, xuanyunyi.xyy, xiaoxiao.xuxx}@alibaba-inc.com {xide.ql}@taobao.com

Abstract—Despite the adoption of large language models (LLMs) in recommendation systems, prevailing approaches mostly model single-type behaviors (e.g., views or purchases). Even when incorporating multiple behaviors, existing methods flatten heterogeneous actions into homogeneous token sequences, ignoring their distinct decision-making roles. This flattening fails to capture semantic hierarchies and contextual nuances in complex decision-making, such as trade-offs between price and quality. Consequently, performance degrades in critical “difficult-choice” scenarios involving highly similar items. To bridge this gap, we propose MARI (Memory-Augmented Recommendation with Interpretability), which grounds predictions in explicit, structured decision evidence. MARI maintains a Decision Memory Bank (DMB) that archives users’ past rationales as Structured Decision Memories (SDMs): concise records of goals, constraints, and trade-offs. These SDMs are generated offline via Post-Hoc Decision Distillation from heterogeneous behaviors and user-generated content. By retrieving relevant SDMs to augment LLM reasoning, MARI achieves interpretability and scalability without the prohibitive cost of processing long raw sequences. Extensive experiments show MARI significantly outperforms state-of-the-art baselines on standard next-item prediction and a newly introduced Difficult Choice Prediction task, incurring low latency overhead by decoupling memory construction from online inference. Qualitative analyses reveal actionable, humanreadable insights into user decision-making, marking a concrete step toward reasoning-aware recommendation systems.

Index Terms—LLM-based Recommendation, Structured Decision Memory, Interpretable Recommendation, Multi-behavior Modeling

## I. INTRODUCTION

Integrating Large Language Models (LLMs) into recommender systems enables semantic user modeling by treating interaction sequences as contextual inputs. Current research predominantly serializes single-type interactions (e.g., ratings or purchases) into long textual prompts [1]–[4]. However, these sequences are costly to process and contain substantial redundancy and behavioral noise. While single-behavior datasets (e.g., Amazon [5]) have advanced the field, they overlook a critical real-world challenge: understanding the complex decision logic derived from heterogeneous, multibehavior interactions (e.g., search, view, favorite, purchase) fundamental to e-commerce and streaming platforms.

Traditional multi-behavior models [6]–[10] rely on implicit, opaque embeddings that lack the reasoning and interpretability strengths of LLMs. Consequently, the powerful inferential capabilities of LLMs remain underutilized for reconstructing user decision-making processes. Existing LLM-based recommenders either restrict themselves to single-behavior data or flatten multi-behavior logs into homogeneous sequences, losing the distinct semantic roles and hierarchical structures of different actions. Thus, the LLM revolution in recommendation has largely overlooked the rich decision signals woven across diverse behavioral contexts.

![](images/c62fab19dc1f617a91a4f22e423e9051d5ccb3428c9ec81eae0945e04e91400c.jpg)  
Fig. 1. Workflow comparison. Top: Directly feeding long raw behavior sequences into an LLM introduces context limits, noise, and latent decision factors, hindering accurate reasoning. Bottom: MARI retrieves Structured Decision Memories (SDMs) from a Decision Memory Bank (DMB) based on recent behaviors. Augmenting the LLM with these explicit rationales (goals, constraints, trade-offs) enables interpretable, grounded reasoning and improves recommendation accuracy.

To bridge this gap, we propose an LLM-powered framework that explicitly reconstructs and leverages users’ historical decision logic through a Decision Memory Bank (DMB), an interpretable repository of structured decision rationales. Our work is motivated by a key observation from industrial data: users’ final decisions, especially when choosing between highly similar items (i.e., difficult-choice scenarios at the SKU level), reflect consistent patterns encoding their underlying criteria (e.g., prioritizing “organic” over “budget”). Because these latent patterns are distributed across multi-behavior trajectories, directly feeding raw sequences to LLMs is inefficient; it introduces noise and forces on-the-fly decision reconstruction under tight context constraints, a limitation visually demonstrated in the top workflow of Figure 1. In contrast, our two-stage approach (Figure 1, bottom) constructs the DMB, where each entry is a Structured Decision Memory (SDM), a concise textual record of a user’s goals, constraints, and trade-offs. We devise Post-Hoc Decision Distillation (PHDD), which uses an LLM as a retrospective analyst to generate SDMs by examining final outcomes within heterogeneous behavioral contexts. This transforms noisy logs into explicit decision evidence. Second, we introduce MARI (Memory-Augmented Recommendation with Interpretability), a plugand-play framework that retrieves relevant SDMs to ground LLM reasoning in explicit historical logic. This design cleanly separates offline knowledge construction from online recommendation, enabling robust, interpretable inference without significant latency overhead.

Our contributions are threefold:

• Fine-Grained SKU-Level Recommendation: We reformulate LLM-based recommendation as a SKU-level task, leveraging heterogeneous behaviors to reason over nearidentical items. This effectively addresses difficult-choice scenarios with nuanced attribute differences, which are typically overlooked by title-level approaches.

• Decision Memory Bank via Retrospective Distillation: We introduce the DMB, a compact archive of humanreadable Structured Decision Memories (SDMs). Distilled via our Post-Hoc Decision Distillation (PHDD) method, each SDM encodes a user’s goals, constraints, and trade-offs from heterogeneous trajectories.

• Effective Memory-Augmented Recommendation Architecture: We design a modular framework integrating the DMB via Retrieval-Augmented Generation (RAG) to enable interpretable reasoning grounded in historical logic. MARI achieves significant performance gains over state-of-the-art baselines in offline evaluations on a realworld e-commerce platform, demonstrating practical efficacy and scalability.

## II. RELATED WORK

## A. LLM-based Recommendation

Large Language Models (LLMs) have enabled semantic user modeling in recommendation, moving beyond ID-based collaborative filtering [11]–[16]. Existing approaches fall into three categories: (1) feature enhancement, where LLMs enrich embeddings with textual semantics [17], [18]; (2) end-to-end recommendation, where LLMs are prompted or tuned to predict preferences directly [1]–[4]; and (3) behavior compression and summary, where lengthy user histories are compressed into compact representations or summarized into semantic profiles to address context-length limitations [19]–[21]. However, these methods either perform preference inference and summarization based on a single type of user behavior, or incorporate multiple behaviors only as semantically impoverished IDs, failing to account for their distinct roles in the decisionmaking process. A-LLMRec [3] and LLM-SRec [4] align collaborative filtering signals with the LLM’s token space to inject collaborative awareness but operate on a single type of interaction, thus lacking the capacity to model crossbehavior decision dynamics; while User-LLM [19] compresses multiple behaviors into fixed-size vectors, obscuring transitions between action types. While converting raw behavior streams into semantic summaries effectively mitigates contextlength and sparsity issues [20], [21], these remain shallow abstractions that reflect what users did, not why they decided. More fundamentally, static global summaries aggregate crosscategory preferences, losing the context-dependent trade-offs essential for SKU-level difficult choices. We instead conditionally retrieve structured decision rationales to preserve this specificity without aggregation loss.

## B. Multi-Behavior Recommendation

Multi-behavior recommendation leverages heterogeneous interactions to mitigate data sparsity. While early work extends matrix factorization [22] or designs behavior-aware sampling [23], [24], recent deep models use attention [25], [26] or graph convolutions to fuse multi-behavior signals [6]–[10]. Yet all produce opaque embeddings that capture statistical correlations without revealing the decision intent behind behavior transitions. Moreover, they ignore rich item textual attributes—such as price, brand, and product descriptions—that could ground user choices in semantic evidence. Recent LLMbased approaches often concatenate heterogeneous actions into flat sequences [27], failing to capture the decision-level meanings that different behavior types convey in the user’s choice process. Critically, both lines predict next actions rather than modeling the decision logic behind behavior transitions—they capture statistical patterns (e.g., “view X → purchase Y”) but miss the underlying goals (e.g., seeking premium brands) or constraints (e.g., limited budget). We address this by using LLMs to retrospectively distill explicit decision rationales from trajectories, enabling recommendations grounded in interpretable evidence.

## C. Retrieval-Augmented Recommendation

Retrieval-Augmented Generation (RAG) [28] reduces hallucination by grounding LLMs in external knowledge. In recommendation, methods like CoRAL [29] and RETURN [30] retrieve collaborative signals (e.g., co-purchases) to support long-tail reasoning. However, they retrieve only atomic interactions (“what users did”) without capturing the underlying decision logic (“why they decided”). Unlike these approaches—which aim to improve item prediction by retrieving behavioral or item-level evidence—our framework targets the reconstruction of interpretable decision rationales from multi-behavior trajectories. We argue that effective multibehavior reasoning requires retrieving decision patterns, not just behavior logs. Our Decision Memory Bank (DMB) stores Structured Decision Memories (SDMs)—concise records of context, rationale, and outcome—so that RAG can retrieve historically grounded, trade-off-aware decision evidence instead of surface-level co-occurrences.

## III. PRELIMINARY

Let U and I denote the sets of users and items, respectively. For a user $u \in \mathcal { U } .$ , we observe a historical heterogeneous interaction sequence:

$$
\mathcal { H } _ { u } = \{ ( i _ { t } , a _ { t } , \tau _ { t } ) \} _ { t = 1 } ^ { T _ { u } } ,\tag{1}
$$

where $i _ { t } ~ \in ~ \mathcal { T }$ is the interacted item (associated with rich attributes and metadata), $a _ { t } \in \mathcal A$ is the associated behavior type $( { \mathrm { e . g . , ~ } } { \mathcal { A } } ~ = ~ \{ \mathrm { v i e w } $ , search, favorite, purchase}) and $\tau _ { t }$ is the timestamp. Interactions are chronologically ordered $( \tau _ { 1 } < \tau _ { 2 } < \cdot \cdot \cdot < \tau _ { T _ { u } } )$ . Notably, items in $\mathcal { T }$ are defined at the SKU level—distinct attribute combinations (e.g., color or size) yield distinct items, even with identical titles. An example is given in Appendix A.

In real-world recommendation scenarios, users typically exhibit multiple types of behavior before making a final decision (e.g., purchase). Critically, these heterogeneous behaviors encode complementary signals about the user’s underlying intent—such as exploration (search/view), preference expression (favorite), and commitment (purchase). However, directly modeling $\mathcal { H } _ { u }$ as a flat token sequence overlooks the semantic hierarchy among action types and fails to reconstruct the high-level decision logic that governs user choices. To systematically evaluate a model’s capacity to overcome this limitation and capture such latent decision logic, we formalize two complementary prediction tasks that reflect different levels of recommendation difficulty and intent granularity.

## A. Task 1: Next-Item Prediction (NIP)

Given the full historical sequence $\mathcal { H } _ { u }$ and a recent shortterm context $\mathcal { C } _ { u } ~ \subseteq ~ \mathcal { H } _ { u }$ (e.g., the last k interactions), the goal is to predict the next item the user will purchase. In practice, a purchased item always appears earlier in $\mathcal { C } _ { u }$ through non-purchase interactions (e.g., view or search). Thus, NIP requires the model to identify the specific item within $\mathcal { C } _ { u }$ that will eventually be purchased. This task assesses the model’s capacity to identify the next purchase from the full item set using the user’s heterogeneous interaction history.

## B. Task 2: Difficult Choice Prediction (DCP)

This task simulates realistic “difficult-choice” scenarios where users compare highly similar alternatives (e.g., products with near-identical specifications but differing in price, brand, or minor features). Formally, given $\mathcal { H } _ { u } ,$ a recent context $\mathcal { C } _ { u }$ , and a candidate set $\mathcal { X } \subset \mathcal { T }$ of m items with high pairwise similarity, the goal is to predict which item in X the user will purchase next. Unlike NIP, DCP isolates the ability to resolve fine-grained user preferences by restricting candidate items to highly substitutable sets, thereby directly testing a model’s capacity to infer subtle trade-offs—such as “willingness to pay more for organic certification”—from past decisions.

## C. Key Challenge and Our Approach

Both NIP and DCP require distilling latent decision logic—such as goals, constraints, and attribute-level tradeoffs—from the noisy heterogeneous sequence $\mathcal { H } _ { u }$ , rather than relying on surface-level patterns. However, as shown in Figure 2, our experiments with DeepSeek-R1 [31] show that simply extending the input context yields diminishing returns: performance peaks at around 100 historical interactions and degrades with longer sequences, suggesting that raw behavior logs overwhelm LLMs with noise rather than useful signals.

![](images/5b4a2fb497b849bdb749d8452fdca39eb82ab8a15656538958e07fbaafe8785d.jpg)  
Fig. 2. Effect of input sequence length on Hit@1 for NIP and DCP tasks (using DeepSeek-R1).

To address this, we propose a personalized Decision Memory Bank (DMB) that stores Structured Decision Memories (SDMs), generated via Post-Hoc Decision Distillation (PHDD) as concise, interpretable summaries of past decisions. Building on DMB, we design MARI (Memory-Augmented Recommendation with Interpretability), which retrieves relevant SDMs during inference to ground the LLM’s reasoning about $\mathcal { C } _ { u }$ in explicit decision logic. Formal details follow.

## IV. METHODOLOGY

We present MARI, a two-stage framework that decouples decision knowledge construction from on-the-fly recommendation reasoning. As illustrated in Figure 3, MARI first builds a personalized DMB for each user via Post-Hoc Decision Distillation (PHDD), then leverages retrieval-augmented generation to inject relevant decision memories into an LLM-based recommender during inference. Below we detail the DMB construction pipeline and the MARI inference architecture for both NIP and DCP tasks.

## A. Decision Memory Bank (DMB)

1) Category-Aware: While some decision tendencies $( \mathrm { e . g . }$ cost-effectiveness) generalize across categories, purchase logic is largely category-specific: “Organic certification” matters for baby food but not electronics, whereas “512GB storage” drives electronics choices but not food purchases. Flattening crosscategory interactions dilutes these discriminative, categoryconditioned rules. To preserve fidelity, we build the DMB in a category-aware manner, with each entry encoding the structured rationale behind a user’s purchase in category $c \in { \mathcal { C } } .$

2) Formal definition: For user u, the Decision Memory Bank is defined as a set of structured records:

$$
\mathrm { D M B } _ { u } = \bigcup _ { c \in \mathcal { C } } \left\{ \left( \mathbf { e } _ { u , c } ^ { ( j ) } , \mathcal { M } _ { u , c } ^ { ( j ) } , \tau _ { u , c } ^ { ( j ) } \right) \right\} _ { j = 1 } ^ { J _ { u , c } }\tag{2}
$$

where $J _ { u , c }$ is the number of purchases user u made in category $c ; \tau _ { u , c } ^ { ( j ) }$ is the timestamp of the j-th purchase; $\mathbf { e } _ { u , c } ^ { ( j ) } \in \mathbb { R } ^ { d }$ is the embedding vector of the pre-purchase heterogeneous behavior sequence leading to this purchase (used for similarity-based retrieval); and $\bar { \mathcal { M } } _ { u , c } ^ { ( j ) }$ is the structured decision memory (SDM) —a structured quintuple capturing five complementary facets of the user’s decision logic:

![](images/23aa538415c6dcda54fe9a9cd4ff0f965a589732b7e42c5a2ee4a4a551a14c7e.jpg)  
Fig. 3. Algorithmic framework of MARI. The offline stage constructs DMB via PHDD; during online inference, MARI retrieves relevant SDMs and fuses them to prompt the LLM for interpretable recommendation.

$$
\mathcal { M } _ { u , c } ^ { ( j ) } = \big ( \mathcal { R } _ { u , c } ^ { ( j ) } , \mathcal { T } _ { u , c } ^ { ( j ) } , \mathcal { E } _ { u , c } ^ { ( j ) } , \mathcal { P } _ { u , c } ^ { ( j ) } , \mathcal { A } _ { u , c } ^ { ( j ) } \big ) .\tag{3}
$$

Each component is formally defined as follows:

• Decision Intent Reasoning $\mathcal { R } _ { u , c } ^ { ( j ) }$ : a natural language explanation of the purchase of item $i ^ { * } ,$ grounded in the user’s historical heterogeneous behaviors.

• Decision Profile Tags $\mathcal { T } _ { u , c } ^ { ( j ) }$ : a set of categorical labels characterizing the user’s decision style in this context, drawn from a predefined taxonomy $V _ { \mathrm { t a g } } .$ Examples of user decision profile tags include brand-sensitive, price-sensitive, and certification-driven preferences.

• Decision Efficiency $\mathcal { E } _ { u , c } ^ { ( \bar { j } ) } { : }$ a discrete decision tempo type reflecting the user’s decision-making speed, inferred from behavioral signals such as session duration, number of item comparisons, and backtracking frequency. This yields four categories of decision efficiency: flash, rapid, considered, and prolonged.

• Decision Preference Factors $\mathcal { P } _ { u , c } ^ { ( j ) } \{$ : a set of key-value pairs $\{ ( k _ { p } , v _ { p } ) \}$ where $k _ { p }$ denotes a positively weighted decision attribute (e.g., brand, price, style) and $v _ { p }$ specifies its concrete instantiation in the chosen item. Example: (brand, Nike), (price, 500–800), (style, minimalist).

• Decision Avoidance Factors $\mathcal { A } _ { u , c } ^ { ( j ) } \{$ a set of key-value pairs $\left\{ \left( k _ { a } , v _ { a } \right) \right\}$ capturing attributes explicitly rejected during exploration. Here $k _ { a }$ denotes a decision dimension (same category as $k _ { p } ) _ { \ L }$ , and $v _ { a }$ specifies the avoided value. Example: (material, polyester), (price, >1200), (color, bright-red).

Both the profile tags $\mathcal { T } _ { u , c } ^ { ( j ) }$ and the keys in $\mathcal { P } _ { u , c } ^ { ( j ) }$ and $\mathcal { A } _ { u , c } ^ { ( j ) }$ are from closed vocabularies, derived via iterative LLMbased clustering on large-scale e-commerce data. The LLM also generates evidence-backed rationales for each factor to ensure plausibility. An example of an SDM is provided in Appendix C.

## B. Post-Hoc Decision Distillation (PHDD)

To enable efficient inference, we design a two-stage distillation pipeline that yields a lightweight student model for generating user SDMs: (1) a teacher LLM generates structured memories and Chain-of-Thought (CoT) [32]; (2) a small student model is trained via CoT distillation [33] to mimic this capability.

1) Teacher Model: Category-Conditioned Memory Generation: To construct high-quality CoT training data for PHDD, we leverage the powerful reasoning capability of DeepSeek-R1 [31] to generate grounded, interpretable decision rationales from real user interaction logs. Specifically, for each purchase event $( i ^ { * } , c , \tau ^ { * } )$ by user u in category $c ,$ we extract the category-filtered pre-purchase sequence:

$$
\mathcal { H } _ { u , c } ^ { \mathrm { p r e } } = \left\{ \left( i _ { t } , a _ { t } , \tau _ { t } \right) \vert \tau _ { t } < \tau ^ { * } , \mathrm { c a t e g o r y } ( i _ { t } ) = c \right\} .\tag{4}
$$

We employ a teacher LLM $\phi _ { \mathrm { t e a } } ^ { \mathrm { P H D D } }$ , prompted with a carefully designed template to enforce a consistent output format, generating structured decision memories $\mathcal { M } _ { u , c } ^ { ( j ) }$ . These outputs are subsequently verified to eliminate hallucinations and ensure high-quality reasoning trajectories. Critically, to enhance rationale accuracy, we provide the teacher with postpurchase reviews as privileged calibration references exclusively during dataset construction. Guided by the prompt, the teacher deduces decision logic solely from pre-purchase behaviors, utilizing the reviews merely to validate the inferred rationale without directly leaking their text. These signals are excluded during student training and inference, ensuring realistic deployment conditions.

2) Student Model Training via CoT Distillation: To enable scalable DMB construction in real-world production scenarios, we distill the teacher’s capability into a compact student LLM $\phi _ { \mathrm { s t u } } ^ { \mathrm { P H D D } }$ (Qwen3-4B [34]) using Chain-of-Thought distillation:

• Dataset creation: For each purchase event, we collect the teacher’s CoT reasoning trace (intermediate rationale steps) and final structured memory $\mathcal { M } _ { u , c } ^ { ( j ) }$ as supervision signals.

• Input representation: The student receives only $\mathcal { H } _ { u , c } ^ { \mathrm { p r e } }$ and category context c.

• Training objective: Minimize the cross-entropy loss between student-generated tokens and the teacher’s CoT trace + final memory:

$$
\mathcal { L } _ { \mathrm { P H D D } } = - \sum _ { l = 1 } ^ { L } \log P _ { \phi _ { \mathrm { s u } } ^ { P H D D } } \left( y _ { l } ^ { \mathrm { t e a } } \mid \mathcal { H } _ { u , c } ^ { \mathrm { p r e } } , c , y _ { < l } ^ { \mathrm { t e a } } \right)\tag{5}
$$

where $y _ { 1 : L } ^ { \mathrm { t e a } }$ is the concatenated CoT trace and SDM text.

After training, $\phi _ { \mathrm { s t u } } ^ { \mathrm { P H D D } }$ generates SDMs $\mathcal { M } _ { u , c } ^ { ( j ) }$ for all historical purchases. For each SDM, we compute its retrieval embedding $\mathbf { e } _ { u , c } ^ { ( j ) }$ by encoding the corresponding pre-purchase sequence $\mathcal { \bar { H } } _ { u , c } ^ { \mathrm { p r e } }$ with a lightweight, frozen text encoder $( \mathrm { e . g . }$ Qwen3-Embedding-0.6B [35]). The resulting $\scriptstyle \mathrm { D M B } _ { u }$ is stored as a category-indexed vector database supporting efficient similarity search.

## C. Memory-Augmented Reasoning Framework

The MARI inference pipeline consists of two core stages: (1) context processing and adaptive memory retrieval, and (2) prompt construction and memory-augmented LLM reasoning.

1) Context Processing and Adaptive Memory Retrieval: Given a user’s recent interaction context $\begin{array} { r c l } { \mathcal { C } _ { u } } & { = } & { \{ \left( i _ { t } , a _ { t } , \tau _ { t } \right) \} _ { t = T _ { u } - k + 1 } ^ { T _ { u } } , } \end{array}$ we first encode it into a dense vector $\mathbf { e } _ { \mathrm { c t x } }$ using the frozen text encoder (same as in DMB construction). The retrieval strategy then adapts to the prediction task.

For Next-Item Prediction (NIP): The target category is unknown a priori. We perform cross-category similarity retrieval to find SDM with behaviorally similar exploration patterns:

$$
\mathcal { S } _ { u } ^ { \mathrm { N I P } } = \left\{ \mathcal { M } _ { u , c } ^ { ( j ) } \bigg | \left( c , j \right) \in \operatorname * { \mathrm { ~ \tiny ~ \mathrm { ~ T o p K ~ } ~ } } _ { c \in \mathcal { C } , j \in \left[ 1 , J _ { u , c } \right] } \left( \sin { \left( \mathbf { e } _ { \mathrm { c t x } } , \mathbf { e } _ { u , c } ^ { ( j ) } \right) } \right) \right\}\tag{6}
$$

where $\begin{array} { r } { \sin ( \mathbf { a } , \mathbf { b } ) = \frac { \mathbf { a } ^ { \top } \mathbf { b } } { \| \mathbf { a } \| \| \mathbf { b } \| } } \end{array}$

For Difficult Choice Prediction (DCP): The target category $c _ { \mathrm { t a r g e t } }$ is known from session context. We perform categoryconstrained temporal retrieval:

$$
\mathcal { S } _ { u } ^ { \mathrm { D C P } } = \left\{ \mathcal { M } _ { u , c _ { \mathrm { t a r g e t } } } ^ { ( j ) } \ | \ j \in \mathrm { T o p K } _ { \mathrm { t i m e } } \left( \{ \tau _ { u , c _ { \mathrm { t a r g e t } } } ^ { ( j ) } \} \right) \right\} .\tag{7}
$$

Prioritizing recency over behavioral similarity is critical here: fine-grained preference trade-offs evolve rapidly within categories and are best captured by recent decisions. If the user has no SDM records in the target category $c _ { \mathrm { t a r g e t } }$ , we fall back to decision memory retrieval as defined in Equation 6.

2) Prompt Construction and Memory-Augmented Reasoning: We integrate retrieved memories $\mathcal { S } _ { u }$ with the current context $\mathcal { C } _ { u }$ to construct task-specific prompts for $\phi ^ { \mathrm { r e c } }$ , an LLMbased recommender.

• For NIP, where the target category is unknown a priori, the prompt instructs the LLM to predict the next purchased item directly from the user’s recent interaction sequence, augmented with SDMs retrieved from the DMB based on the current behavioral context.

• For DCP, where the candidate set within a known category is provided, the prompt instructs the LLM to predict the next purchased item directly from the candidate set, augmented with SDMs retrieved from the DMB preferentially based on the target category.

Both designs enforce structured reasoning grounded in explicit decision evidence, enabling $\phi ^ { \mathrm { r e c } }$ to move beyond surface-level sequential patterns—enhancing robustness under data sparsity while preserving interpretability through traceable memory references. The prompt template is provided in Appendix D.

## D. Deployment Strategies

MARI supports two deployment paradigms that balance training overhead, interpretability, and integration flexibility.

1) Plug-and-Play Augmentation Mode: MARI functions as a lightweight, training-free augmentation module for existing LLM-based recommenders. Given context $\mathcal { C } _ { u } .$ , it retrieves relevant memories $\mathcal { S } _ { u }$ and constructs the task-appropriate prompt for any off-the-shelf recommendation LLM $\phi ^ { \mathrm { r e c } }$ . This trainingfree compatibility enables:

• Seamless integration into production pipelines.

• No additional training costs or architectural modifications.

• Preservation of existing model investments while enhancing decision transparency.

2) CoT-Distilled Mode: For maximum performance and full interpretability, we distill a specialized student model $\phi _ { \mathrm { s t u } } ^ { \mathrm { r e c } }$ that generates both predictions and justifications. Training follows the MARI inference pipeline: (1) For each purchase $( i ^ { * } , \tau ^ { * } )$ , extract pre-purchase context $\mathcal { C } _ { u } ^ { \mathrm { p r e } } ; ( 2 )$ Retrieve relevant memories $\mathcal { S } _ { u } ^ { \mathrm { p r e } }$ using task-appropriate strategy; (3) Prompt the teacher LLM to generate CoT reasoning from $( \mathcal { C } _ { u } ^ { \mathrm { { p r e } } } , \mathcal { S } _ { u } ^ { \mathrm { { p r e } } } )$ , concluding with $i ^ { * } ; ~ ( 4 )$ Train student to autoregressively generate this CoT+item sequence label $y ^ { r e c }$

$$
\mathcal { L } _ { \mathrm { r e c } } = - \sum _ { l = 1 } ^ { L } \log P _ { \phi _ { \mathrm { s u } } ^ { r e c } } \left( y _ { l } ^ { r e c } \mid \mathcal { C } _ { u } ^ { \mathrm { p r e } } \oplus \mathcal { S } _ { u } ^ { \mathrm { p r e } } , y _ { < l } ^ { r e c } \right) .\tag{8}
$$

At inference, $\phi _ { \mathrm { s t u } } ^ { \mathrm { r e c } }$ outputs human-interpretable rationales alongside predictions. This establishes a new paradigm for transparent recommendation where every prediction is grounded in explicit, retrievable decision evidence.

Both deployment modes share the same DMB infrastructure, allowing practitioners to choose based on their system constraints and interpretability requirements. This design enables MARI to scale efficiently to large-scale systems while preserving detailed decision semantics critical for difficult-choice recommendations.

## V. EXPERIMENTS SETUP

## A. Datasets

Existing public multi-behavior datasets lack the rich textual item metadata (e.g., titles, attributes) essential for LLM-based reasoning, as they mostly contain anonymized IDs. To address this, we construct a new large-scale dataset from six months of real-world user logs on a major e-commerce platform, capturing heterogeneous behaviors—search, view, favorite, and purchase—along with comprehensive item metadata. We further split it into two task-specific datasets: the Post-Hoc Decision Distillation (PHDD) Dataset and the Multi-Behavior Recommendation (MBR) Dataset.

1) PHDD Dataset: The PHDD Dataset captures finegrained, user-level decision traces in their original form, preserving all observed interactions without behavioral filtering or deduplication. It is partitioned into 200k, 10k, and 10k samples for training, validation, and testing, respectively.

2) MBR Dataset: We construct this dataset to evaluate tasks NIP and DCP, following the previous methods [36] with two preprocessing steps to ensure data quality: (1) we merge duplicated user-item-behavior interactions by keeping the earliest record; (2) we apply a k-core filtering (k=10) to retain users and items with at least 10 interactions, mitigating data sparsity. Each user’s filtered chronological sequence is then split via leave-one-out by timestamp: the last purchase for testing, the second-last for validation, and earlier interactions for training.

To evaluate the cross-category generalization capability of our MARI framework under the Next Item Prediction (NIP) setting, we further partition the instances into two disjoint subsets: (1) NIP-in-category (NIP-in), comprising users who have previously purchased at least one item in the same category as the target item; and (2) NIP-out-of-category (NIP-out), where the target purchase represents the user’s first interaction in that category. This partition enables us to assess whether our method can effectively leverage decision memory to support recommendations in both familiar and novel category contexts.

For the DCP task, where the target item category is assumed to be known, we construct the candidate set by using an LLM (Qwen3-Max) to select 2–5 items within the same category that the user showed notable hesitation or comparison behavior toward in their recent interaction history, followed by manual validation of the selected candidates. An example of a candidate set is provided in Appendix B.

The resulting MBR dataset comprises three distinct tasks (NIP-in, NIP-out, and DCP) with non-overlapping users across tasks, yielding a total of 150,000 unique users. To ensure fair comparison, each task is identically sampled into 50k training, 10k validation, and 10k testing instances.

## B. Metrics

1) PHDD Task: For the PHDD task, Decision Profile Tags, Decision Efficiency, and the keys of Decision Preference/Avoidance Factors are closed-set. We evaluate them using precision, defined as the fraction of correctly predicted items among all predictions. In contrast, Decision Intent Reasoning and the values of Preference/Avoidance Factors are open-ended. We assess them via an LLM-as-judge approach, measuring the following dimensions on a scale of 0–5: (1) structural correctness (adherence to output format), (2) reasoning quality (logical coherence of the decision inference process), and (3) value abstraction (appropriateness and generalizability of extracted factor values). Specifically, we employ

DeepSeek-R1 as the judge model and provide it with a humanverified reference solution as the scoring benchmark to ensure evaluation reliability.

2) MBR Task: For the MBR task, we adopt a next-item prediction setting where the user’s subsequent actual purchase serves as the ground-truth target. Model performance is evaluated using a widely used metric Hit Ratio@1 (Hit@1).

## C. Experimental Settings

1) Baseline: We establish a comprehensive set of baseline models for our two core tasks.

For the PHDD task, which focuses on generating a user’s decision memory, we adopt the off-the-shelf Qwen3 [34] series of large language models as our primary baselines. These models are used without any additional fine-tuning.

For the MBR task, we benchmark against two baseline categories. First, we include ID-based sequential recommenders: $\operatorname { S A S R e c } _ { M B }$ [37], SASRec’s [14] multi-behavior extension that incorporates different interaction types; and three recent stateof-the-art ID-based multi-behavior recommendation methods, namely MULE [8], COPF [9], and HGIB [10]. Second, to fully leverage the rich textual semantics of items, we integrate several powerful open-source LLMs into the recommendation pipeline. Specifically, we employ Qwen3, DeepSeek-R1 (R1) [31], and GPT-OSS [38], which process item metadata such as titles, prices, and attributes using their native language understanding capabilities. Similar to the PHDD task, these LLMs are also used in their original, unmodified form without any further training on recommendation data. To ensure fair comparison, all baselines predict the target item from a unified candidate space: the recent historical sequence (for NIP) or the provided candidate set (for DCP).

2) Implementation Details: For the PHDD task, we set the maximum length of the input behavior sequence $\mathcal { H } _ { u , c }$ to 200, considering only interactions that occurred within a twoweek window prior to the target decision. To support retrieval in the DMB, we employ the Qwen3-Embedding-0.6B model to extract dense representations of behavior sequences. The PHDD Model $\phi _ { \mathrm { s t u } } ^ { \mathrm { P H D D } }$ is trained via full-parameter fine-tuning on the PHDD dataset, based on the Qwen3-4B.

For the MBR task, we set the maximum input sequence length $\mathcal { C } _ { u }$ to 100, as our preliminary experiments (Figure 2) show that model performance peaks at this length. During memory retrieval, we perform a top-K search with $K = 3 ,$ retrieving the three most relevant decision memories from the DMB. The MBR model $\phi _ { \mathrm { s t u } } ^ { \mathrm { r e c } }$ is fully fine-tuned from Qwen3- 4B on the constructed MBR dataset. We train the model on a mixture of the three tasks: NIP-in-category, NIP-out-ofcategory, and DCP.

## VI. EXPERIMENTAL RESULTS

## A. Performance Comparison

We perform comprehensive performance comparisons across two distinct tasks: the PHDD task, designed to generate users’ structured decision memories, and the Multi-Behavior Recommendation (MBR) task.

1) PHDD: Results on the PHDD dataset (Table I) show that our method enables the compact Qwen3-4B to match the much larger R1 model in precision metrics, while surpassing it in Reasoning Quality and Value Abstraction under LLMas-Judge evaluation. This confirms that PHDD effectively distills the ability to understand user decision intent into a smaller model, ensuring the generation of high-quality SDMs while achieving competitive performance at a fraction of the inference cost.

TABLE I  
QUALITY OF STRUCTURED DECISION MEMORY (SDM) GENERATION ON THE PHDD DATASET. MODELS: Q4B (QWEN3-4B), Q8B (QWEN3-8B), Q14B, Q32B. BOLD: BEST, UNDERLINED: SECOND-BEST.
<table><tr><td>Metric</td><td>Q4B</td><td>Q8B</td><td>Q14B</td><td>Q32B</td><td>R1</td><td>Q4B-PHDD(Ours)</td></tr><tr><td>Precision (↑)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Factor Key</td><td>0.577</td><td>0.654</td><td>0.636</td><td>0.631</td><td>0.779</td><td>0.769</td></tr><tr><td>Decision Prof.</td><td>0.467</td><td>0.470</td><td>0.525</td><td>0.556</td><td>0.725</td><td>0.713</td></tr><tr><td>Decision Effic.</td><td>0.483</td><td>0.473</td><td>0.529</td><td>0.513</td><td>0.707</td><td>0.693</td></tr><tr><td colspan="7">LLM-as-Judge (↑)</td></tr><tr><td>Struct. Corr.</td><td>3.56</td><td>3.31</td><td>3.58</td><td>3.71</td><td>4.13</td><td>4.12</td></tr><tr><td>Reason. Qual.</td><td>3.51</td><td>3.40</td><td>3.69</td><td>3.75</td><td>3.89</td><td>3.93</td></tr><tr><td>Value Abst.</td><td>3.49</td><td>3.50</td><td>3.66</td><td>3.81</td><td>4.21</td><td>4.22</td></tr></table>

2) MBR: Table II details the Hit@1 performance, revealing three core findings: (1) Plug-and-play SDMs generalize across diverse model scales. By simply injecting SDMs into off-the-shelf LLMs without weight updates, MARI<sup>†</sup> broadly enhances performance across diverse parameter sizes (from 4B to 671B). Crucially, when applied to sufficiently large models (e.g., R1, Qwen3-235B-A22B-Think), this plug-andplay approach successfully surpasses strong fully-trained IDbased baselines (e.g., HGIB, MULE), explicitly confirming the generality and potency of our framework. (2) SDMs significantly amplify reasoning on complex tasks. The structured priors provide vital context that medium-to-large models (≥14B) can effectively leverage, yielding massive gains on the challenging NIP-out task (e.g., +165% for GPT-OSS-20B, +128% for Qwen3-32B). However, smaller models (4B/8B) degrade on NIP-out, suggesting that out-of-category reasoning inherently requires a minimum intrinsic capacity. Conversely, on the DCP task, MARI<sup>†</sup> empowers all model scales to consistently outperform ID-based baselines, highlighting LLMs inherent advantage in difficult choice prediction that demands semantic comparison. (3) SFT empowers compact models to achieve state-of-the-art results. Fine-tuning the lightweight 4B model yields substantial improvements across all three tasks. Notably, MARI (4B) achieves 0.5561 on NIP-out and 0.8868 on DCP, significantly outperforming both fully-trained ID-based models and massive 671B untuned LLMs like R1. This demonstrates that specializing compact models to reason over SDMs offers a practical pathway to superior accuracy, mitigating reliance on massive architectures.

Figure 4 presents an example of MARI’s interpretable recommendations, demonstrating that the model effectively uses the information of the SDMs (marked in red) during reasoning to produce the correct product recommendations.

![](images/28e9c1160bc1392343e90fa35f9b8f3d534b77db955496685cd19134760973ed.jpg)  
Fig. 4. The role of structured decision memory: comparing recommendation rationales with and without decision memory augmentation.

## B. Ablation Study

To validate the core design of MARI, we conduct a series of ablation studies comparing MARI-B, MARI<sup>†⋆</sup>, MARI<sup>†</sup>, and SFT variants. Table III details the configurations and results, revealing three core insights:

(1) Superiority of structured memories over raw behaviors. Directly injecting raw sequences aligned with SDMs (MARI-B) yields gains on simpler tasks (e.g., NIP-in) but degrades performance on complex tasks like NIP-out and DCP. This indicates that the verbosity of raw behaviors dilutes contextual information, distracting the LLM from critical decision factors required for deep reasoning. Conversely, using decision memories (MARI<sup>†⋆</sup> and MARI<sup>†</sup>) consistently outperforms MARI-B across all scales. This proves that enriching the input with explicit decision logic—rather than relying solely on implicit patterns in raw sequences—is crucial for guiding LLMs in challenging scenarios.

(2) Critical impact of memory quality. High-quality memories distilled via PHDD (MARI<sup>†</sup>) always beat undistilled, lower-quality ones (MARI<sup>†⋆</sup>), particularly in NIP-out and DCP. This direct correlation between memory quality and performance validates the necessity of PHDD for filtering noise and abstracting clear user intent.

(3) SDMs enhance supervised fine-tuning. The baseline SFT and MARI share the same training objective, differing only in input augmentation. Incorporating retrieved SDMs during SFT yields clear gains over standard SFT without SDMs. This confirms a strong synergy between structured memory inputs and parameter updates for complex reasoning and generalization.

## C. Effect of the Number of Retrieved Memories (K)

We investigate the influence of the retrieved memory count (K) on MARI<sup>†</sup>’s performance, using DeepSeek-R1 as the underlying recommendation model. Results in Figure 5 show a consistent trend: performance initially improves with K,

TABLE II  
FULL COMPARISON OF HIT@1. MARI<sup>†</sup> DENOTES AN OFF-THE-SHELF OPEN-SOURCE GENERAL-PURPOSE MODEL WITHOUT ANY TASK-SPECIFIC FINE-TUNING FOR RECOMMENDATION. BASE DENOTES VANILLA LLMS AND ID-BASED MODELS. BOLD: BEST, UNDERLINED: SECOND-BEST.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Model</td><td colspan="3">Base</td><td colspan="3">MARI† (Ours, w/o SFT)</td></tr><tr><td>NIP-in</td><td>NIP-out</td><td>DCP</td><td>NIP-in</td><td>NIP-out</td><td>DCP</td></tr><tr><td rowspan="4">ID-based</td><td>SASRecM B</td><td>0.1666</td><td>0.1697</td><td>0.4077</td><td></td><td></td><td></td></tr><tr><td>MULE</td><td>0.2363</td><td>0.1968</td><td>0.4206</td><td></td><td></td><td></td></tr><tr><td>COPF</td><td>0.2013</td><td>0.1795</td><td>0.4037</td><td></td><td></td><td></td></tr><tr><td>HGIB</td><td>0.2489</td><td>0.2251</td><td>0.4384</td><td></td><td></td><td></td></tr><tr><td rowspan="8">LLMs (w/o train)</td><td>Qwen3-4B</td><td>0.0751</td><td>0.0225</td><td>0.4617</td><td>0.1278</td><td>0.0157</td><td>0.4751</td></tr><tr><td>Qwen3-8B</td><td>0.1120</td><td>0.0687</td><td>0.4203</td><td>0.1580</td><td>0.0675</td><td>0.4505</td></tr><tr><td>Qwen3-14B</td><td>0.0999</td><td>0.0391</td><td>0.4514</td><td>0.1535</td><td>0.0676</td><td>0.4549</td></tr><tr><td>GPT-OSS-20B</td><td>0.1477</td><td>0.0934</td><td>0.4207</td><td>0.1963</td><td>0.2480</td><td>0.4533</td></tr><tr><td>Qwen3-32B</td><td>0.1136</td><td>0.0496</td><td>0.4479</td><td>0.1742</td><td>0.1135</td><td>0.4697</td></tr><tr><td>Qwen3-235B-A22B-Ins</td><td>0.1304</td><td>0.0798</td><td>0.5069</td><td>0.2224</td><td>0.2422</td><td>0.5251</td></tr><tr><td>Qwen3-235B-A22B-Thk</td><td>0.2068</td><td>0.2992</td><td>0.5010</td><td>0.2687</td><td>0.4189</td><td>0.5249</td></tr><tr><td>R1</td><td>0.1957</td><td>0.2503</td><td>0.5158</td><td>0.2531</td><td>0.3641</td><td>0.5336</td></tr><tr><td colspan="2">MARI (Ours, w/ SFT)</td><td></td><td></td><td></td><td>0.4869</td><td>0.5561</td><td>0.8868</td></tr></table>

## TABLE III

ABLATION STUDY ON HIT@1. MARI-B: USES RAW BEHAVIOR SEQUENCES   
ALIGNED WITH SDMS. MARI<sup>†⋆</sup>: USES UNDISTILLED SDMS GENERATED BY QWEN3-4B (W/O PHDD). MARI<sup>†</sup>: USES DISTILLED SDMS GENERATED BY QWEN3-4B (W/ PHDD). SFT variants: COMPARES   
STANDARD SFT AGAINST SFT WITH SDMS. BOLD INDICATES THE BEST (MARI<sup>†</sup>) AND OVERALL (MARI) RESULTS.

<table><tr><td>Variant</td><td>Q4B</td><td>Q14B</td><td>Q32B</td><td>GPT-OSS-20B</td><td>R1</td></tr><tr><td>NIP-in</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.075</td><td>0.100</td><td>0.114</td><td>0.148</td><td>0.196</td></tr><tr><td>MARI-B (Raw Behav.)</td><td>0.105</td><td>0.125</td><td>0.127</td><td>0.132</td><td>0.247</td></tr><tr><td>MARIt* (Undist. SDM)</td><td>0.110</td><td>0.134</td><td>0.157</td><td>0.170</td><td>0.249</td></tr><tr><td>MARI† (Dist. SDM)</td><td>0.128</td><td>0.154</td><td>0.174</td><td>0.196</td><td>0.253</td></tr><tr><td>SFT w/o SDM</td><td>0.458</td><td></td><td></td><td></td><td></td></tr><tr><td>MARI (SFT + SDM)</td><td>0.487</td><td></td><td>=</td><td>一</td><td>=</td></tr><tr><td>NIP-out</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.023</td><td>0.039</td><td>0.050</td><td>0.093</td><td>0.250</td></tr><tr><td>MARI-B</td><td>0.012</td><td>0.015</td><td>0.014</td><td>0.061</td><td>0.270</td></tr><tr><td>MARIt*</td><td>0.013</td><td>0.046</td><td>0.084</td><td>0.133</td><td>0.317</td></tr><tr><td>MARI†</td><td>0.016</td><td>0.068</td><td>0.114</td><td>0.248</td><td>0.364</td></tr><tr><td>SFT w/o SDM</td><td>0.538</td><td></td><td></td><td></td><td></td></tr><tr><td>MARI</td><td>0.556</td><td></td><td></td><td>=</td><td>=</td></tr><tr><td>DCP</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>0.462</td><td>0.451</td><td>0.448</td><td>0.421</td><td>0.516</td></tr><tr><td>MARI-B</td><td>0.436</td><td>0.437</td><td>0.441</td><td>0.409</td><td>0.511</td></tr><tr><td>MARI†*</td><td>0.470</td><td>0.451</td><td>0.451</td><td>0.432</td><td>0.520</td></tr><tr><td>MARI†</td><td>0.475</td><td>0.455</td><td>0.470</td><td>0.453</td><td>0.534</td></tr><tr><td>SFT w/o SDM</td><td>0.866</td><td></td><td></td><td></td><td></td></tr><tr><td>MARI</td><td>0.887</td><td></td><td></td><td>1</td><td>=</td></tr></table>

![](images/f00a74254359ad2932b53961a30932777ebaf382a07f77037e82f6f77a692f17.jpg)  
Fig. 5. MARI<sup>†</sup>’s Hit@1 performance across different tasks as K increases.

peaks around K = 3 or 4, and then declines. Sensitivity to K varies by task. The NIP-out task is most sensitive, as larger K introduces noise that impairs cross-category reasoning. Conversely, NIP-in and DCP performance remains stable across a broader K range, demonstrating robustness to context volume. We set K = 3 as the default, achieving a balance of strong performance across tasks while minimizing noise from excessive context.

## D. Memory Error Analysis

To evaluate the performance lower bound of MARI under degraded SDM quality, we simulate two error scenarios: (1) error retrieval (lowest-similarity memories) and (2) error memory (random same-category SDMs from other users). Table IV reveals three key insights:

(1) Structural priors defy noise. On NIP-in, error variants consistently outperform the Base LLM (e.g., boosting 4B from 0.075 to 0.094). This suggests even inaccurate SDMs provide valuable structural signals guiding attention better than raw behaviors.

(2) Robustness scales with capacity. On NIP-out, small models (4B/14B) degrade slightly, whereas larger models (≥20B) remain robust and even improve (e.g., R1 jumps to 0.307). This indicates larger LLMs possess sufficient intrinsic capacity to filter noise.

(3) Factual sensitivity in DCP. Performance dips modestly but stays competitive, as DCP inherently relies on precise preference matching, making it more sensitive to factual inaccuracies.

Overall, while LLMs exhibit commendable robustness, high-quality memories remain indispensable for maximizing complex reasoning performance.

## VII. CONCLUSION

We propose MARI, a reasoning-aware framework that explicitly models the decision logic behind heterogeneous user interactions. Via Post-Hoc Decision Distillation, MARI constructs a Decision Memory Bank of structured rationales capturing user goals and trade-offs. Retrieving these memories grounds LLM recommendations in explicit historical evidence, ensuring both interpretability and efficiency. Extensive offline experiments confirm MARI’s superiority in accuracy and realworld engagement, offering a practical path toward transparent, rationale-driven recommender systems. Future work will explore advanced SDM retrieval strategies and adaptive memory updates for dynamic environments, and quantify the specific contribution of individual SDM dimensions to recommendation quality and interpretability.

TABLE IV  
MEMORY ERROR ANALYSIS. Err. Ret.: LOWEST-SIMILARITY MEMORIES. Err. Mem.: RANDOM SAME-CATEGORY SDMS FROM OTHER USERS.
<table><tr><td>Variant Q4B Q14B</td></tr><tr><td>Q32B GPT-OSS NIP-in</td></tr><tr><td>Base 0.075 0.100 0.114 0.148 0.196 0.079 0.124 0.158 0.230 0.223</td></tr><tr><td>Err. Ret. 0.124 Err. Mem. 0.094 0.120 0.117 0.151</td></tr><tr><td>MARI† 0.128 0.154 0.174 0.196 0.253</td></tr><tr><td>NIP-out</td></tr><tr><td>Base 0.023 0.039 0.050 0.093 0.250 Err. Ret. 0.013 0.038 0.080 0.129 0.307</td></tr><tr><td>Err. Mem. 0.012 0.032 0.062 0.118 0.301</td></tr><tr><td>MARI† 0.016 0.068 0.114 0.248 0.364</td></tr><tr><td>DCP</td></tr><tr><td>Base 0.462 0.451 0.448 0.421 0.516</td></tr><tr><td>Err. Ret. 0.437 0.398 0.432 0.428 0.495</td></tr><tr><td>Err. Mem. 0.443 0.414 0.438 0.405 0.494</td></tr><tr><td>MARI† 0.475 0.455 0.470 0.453 0.534</td></tr></table>

## REFERENCES

[1] K. Bao, J. Zhang, Y. Zhang, W. Wang, F. Feng, and X. He, “Tallrec: An effective and efficient tuning framework to align large language model with recommendation,” in Proceedings of the 17th ACM conference on recommender systems, 2023, pp. 1007–1014.

[2] S. Kim, H. Kang, S. Choi, D. Kim, M. Yang, and C. Park, “Large language models meet collaborative filtering: An efficient all-round llmbased recommender system,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 1395– 1406.

[3] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao, “Large language models are zero-shot rankers for recommender systems,” in European Conference on Information Retrieval. Springer, 2024, pp. 364–381.

[4] S. Kim, H. Kang, K. Kim, J. Kim, D. Kim, M. Yang, K. Oh, J. McAuley, and C. Park, “Lost in sequence: Do large language models understand sequential recommendation?” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2025, pp. 1160–1171.

[5] Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley, “Bridging language and items for retrieval and recommendation,” arXiv preprint arXiv:2403.03952, 2024.

[6] S. Gu, X. Wang, C. Shi, and D. Xiao, “Self-supervised graph neural networks for multi-behavior recommendation.” in IJCAI, 2022, pp. 2052–2058.

[7] Z. Cheng, S. Han, F. Liu, L. Zhu, Z. Gao, and Y. Peng, “Multibehavior recommendation with cascading graph convolution networks,” in Proceedings of the ACM Web Conference 2023, 2023, pp. 1181–1189.

[8] S. Lee, G. Ko, H.-J. Song, and J. Jung, “Mule: Multi-grained graph learning for multi-behavior recommendation,” in Proceedings of the 33rd ACM international conference on information and knowledge management, 2024, pp. 1163–1173.

[9] C. Zhai, C. Meng, Y. Yang, K. Zhang, X. Zhao, and X. Li, “Combinatorial optimization perspective based framework for multi-behavior recommendation,” in Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, 2025, pp. 1891–1902.

[10] H. Zhang, C. Shen, X. Sun, J. Tan, Y. Tan, Y. Rong, H. Cheng, and L. Yi, “Hierarchical graph information bottleneck for multi-behavior recommendation,” in Proceedings of the Nineteenth ACM Conference on Recommender Systems, 2025, pp. 155–164.

[11] A. Mnih and R. R. Salakhutdinov, “Probabilistic matrix factorization,” Advances in neural information processing systems, vol. 20, 2007.

[12] A. J. Chaney, D. M. Blei, and T. Eliassi-Rad, “A probabilistic model for using social networks in personalized item recommendation,” in Proceedings of the 9th ACM Conference on Recommender Systems, 2015, pp. 43–50.

[13] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural collaborative filtering,” in Proceedings of the 26th international conference on world wide web, 2017, pp. 173–182.

[14] W.-C. Kang and J. McAuley, “Self-attentive sequential recommendation,” in 2018 IEEE international conference on data mining (ICDM). IEEE, 2018, pp. 197–206.

[15] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang, “Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer,” in Proceedings of the 28th ACM international conference on information and knowledge management, 2019, pp. 1441–1450.

[16] S. Kim, N. Lee, D. Kim, M. Yang, and C. Park, “Task relation-aware continual user representation learning,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 1107–1119.

[17] X. Ren, W. Wei, L. Xia, L. Su, S. Cheng, J. Wang, D. Yin, and C. Huang, “Representation learning with large language models for recommendation,” in Proceedings of the ACM web conference 2024, 2024, pp. 3464–3475.

[18] C. Zhang, S. Wu, H. Zhang, T. Xu, Y. Gao, Y. Hu, and E. Chen, “Notellm: A retrievable large language model for note recommendation,” in Companion Proceedings of the ACM Web Conference 2024, 2024, pp. 170–179.

[19] L. Ning, L. Liu, J. Wu, N. Wu, D. Berlowitz, S. Prakash, B. Green, S. O’Banion, and J. Xie, “User-llm: Efficient llm contextualization with user embeddings,” in Companion Proceedings of the ACM on Web Conference 2025, 2025, pp. 1219–1223.

[20] Z. Zheng, W. Chao, Z. Qiu, H. Zhu, and H. Xiong, “Harnessing large language models for text-rich sequential recommendation,” in Proceedings of the ACM Web Conference 2024, 2024, pp. 3207–3216.

[21] J. Kim, H. Kim, H. Cho, S. Kang, B. Chang, J. Yeo, and D. Lee, “Review-driven personalized preference reasoning with large language models for recommendation. corr, abs/2408.06276, 2024. doi: 10.48550,” arXiv preprint ARXIV.2408.06276, 2024.

[22] Z. Zhao, Z. Cheng, L. Hong, and E. H. Chi, “Improving user topic interest profiles by behavior factorization,” in Proceedings of the 24th international conference on world wide web, 2015, pp. 1406–1416.

[23] J. Ding, G. Yu, X. He, Y. Quan, Y. Li, T.-S. Chua, D. Jin, and J. Yu, “Improving implicit recommender systems with view data.” in IJCAI, 2018, pp. 3343–3349.

[24] G. Guo, H. Qiu, Z. Tan, Y. Liu, J. Ma, and X. Wang, “Resolving data sparsity by multi-type auxiliary implicit feedback for recommender systems,” Knowledge-Based Systems, vol. 138, pp. 202–207, 2017.

[25] L. Guo, L. Hua, R. Jia, B. Zhao, X. Wang, and B. Cui, “Buying or browsing?: Predicting real-time purchasing intent using attentionbased deep network with multiple behavior,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 1984–1992.

[26] L. Xia, C. Huang, Y. Xu, P. Dai, B. Zhang, and L. Bo, “Multiplex behavioral relation learning for recommendation via memory augmented transformer network,” in Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, 2020, pp. 2397–2406.

[27] B. Geng, Z. Huan, X. Zhang, Y. He, L. Zhang, F. Yuan, J. Zhou, and L. Mo, “Breaking the length barrier: Llm-enhanced ctr prediction in long textual user behaviors,” in Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 2311–2315.

[28] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel¨ et al., “Retrieval-

augmented generation for knowledge-intensive nlp tasks,” Advances in neural information processing systems, vol. 33, pp. 9459–9474, 2020.

[29] J. Wu, C.-C. Chang, T. Yu, Z. He, J. Wang, Y. Hou, and J. McAuley, “Coral: collaborative retrieval-augmented large language models improve long-tail recommendation,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 3391–3401.

[30] L. Ning, W. Fan, and Q. Li, “Retrieval-augmented purifier for robust llmempowered recommendation,” arXiv preprint arXiv:2504.02458, 2025.

[31] DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025. [Online]. Available: https://arxiv.org/abs/2501.12948

[32] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24 824–24 837, 2022.

[33] L. C. Magister, J. Mallinson, J. Adamek, E. Malmi, and A. Severyn, “Teaching small language models to reason,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 2023, pp. 1773–1781.

[34] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu, “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[35] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, and J. Zhou, “Qwen3 embedding: Advancing text embedding and reranking through foundation models,” arXiv preprint arXiv:2506.05176, 2025.

[36] C. Gao, X. He, D. Gan, X. Chen, F. Feng, Y. Li, T.-S. Chua, L. Yao, Y. Song, and D. Jin, “Learning to recommend with multiple cascading behaviors,” IEEE transactions on knowledge and data engineering, vol. 33, no. 6, pp. 2588–2601, 2019.

[37] Y. Yang, C. Huang, L. Xia, Y. Liang, Y. Yu, and C. Li, “Multi-behavior hypergraph-enhanced transformer for sequential recommendation,” in Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, 2022, pp. 2263–2274.

[38] OpenAI, “gpt-oss-120b & gpt-oss-20b model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925

## APPENDIX

## A. Item Metadata Example

## Example: Raw Item Profile in the Dataset

Item ID: 12345678

Category: Four-Wheel Strollers

Brand: Phoenix Price: 2000.0

Title: Phoenix 0-3 Years Reclining & Seating Baby Stroller

Attributes: Color: Camel [Premium Edition] - Upgraded polished aluminum frame, electroplated EVA silent wheels, includes bonus cartoon seat cushion.

Engagement Statistics:

Stock: 8,600

Favorites: 1,900

Total Sales: 2,700

## B. Candidate Set Example

## Example: Candidate Set

{"item\_title": "PiP Studio Unisex Dutch Imported Mug", ,→ "attribute": "Color: Orange", "brand": "PiP ,→ Studio", "price": "105.0", "stock": 620, "

,→ favorites\_count": 2410, "sales\_count": 853, " ,→ item\_id": "578462125xxxx"},

{"item\_title": "PiP Studio Unisex Dutch Imported Mug", ,→ "attribute": "Color: Chocolate Brown", "brand": ,→ "PiP Studio", "price": "118.0", "stock": 582, " ,→ favorites\_count": 2410, "sales\_count": 878, " ,→ item\_id": "578114651xxxx"},

{"item\_title": "PiP Studio Unisex Dutch Imported Mug", ,→ "attribute": "Color: Sky Blue", "brand": "PiP ,→ Studio", "price": "105.0", "stock": 620, " ,→ favorites\_count": 2410, "sales\_count": 853, " ,→ item\_id": "578114651xxxx"}

## C. Structured Decision Memory Example

## Example: Structured Decision Memory

## Profile: Brand Loyalist

Efficiency: Considered

Intent Reasoning: User exhibited highly goal-directed behavior for women’s comfort loungewear. On Oct 1 (15:50), searched “Fenteng cotton loungewear”, inspected attributes (XXL, beige), added to cart, and repeatedly viewed media to validate comfort. At 15:52, briefly explored an alternative (“Hannaguo modal loungewear set”) but returned to Fenteng within 1 min, reinforcing brand trust. After a cooling-off period, replayed video/images on Oct 7 (09:38) for final confirmation, purchasing at 125 (orig. 318) at 09:40. Core drivers: brand reliability and material safety; alternative exploration merely reinforced the initial choice.

Preference Factors:

• Brand: Fenteng – Consistent focus from search to purchase with repeated interactions. (High)

• Material: Cotton – Explicitly required; no non-cotton alternatives considered. (High)

• Price Range: 100-200 – Final payment (125) falls in range; accepted discount. (Medium)

Avoidance Factors:

• Material: Modal – Briefly viewed Hannaguo modal product but abandoned it in favor of cotton option (Medium)

## D. Prompt Templates

## Prompt Template for MARI

## # Role & Task

You are a senior e-commerce shopping advisor. Given the user’s [Historical Decisions], [Recent Behaviors], and [Candidate Items], predict the single item the user is most likely to purchase next.

## # Decision Logic (in order of priority)

1. Intent Recognition: Analyze commonalities across recent behaviors and decisions (e.g., brand preference, price range, urgent needs).

## 2. Logical Judgment:

\- If [Candidate Items] is non-empty: Select the single item from the pool that best aligns with the user’s recent behavior and exhibits the strongest purchase intent.

\- If [Candidate Items] is empty: Revisit [Recent Behaviors] and select the single item with the strongest recent purchase intent.

## # Output Format (Strict JSON)

"reasoning": "A logical and coherent explanation",

"item\_title": "Full name of the predicted item",

"item\_id": "Unique ID of the predicted item"

## # Runtime Input Variables

Historical Decisions: {Historical\_Decisions}

Recent Behaviors: {Recent\_Behaviors}

Candidate Items: {Candidate\_Items}