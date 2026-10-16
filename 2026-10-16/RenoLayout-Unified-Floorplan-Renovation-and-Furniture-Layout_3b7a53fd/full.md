# RenoLayout: Unified Floorplan Renovation and Furniture Layout for Constraint-Aware Residential Design

Jingguo Liu, Yuanqing Liu, Chong Zhang, Yue Yang<sup>∗</sup>

Beike

{liujingguo002, liuyuanqing006, zhangchong052, yangyue092}@ke.com

## Abstract

Residential design aims to transform existing homes into functional living spaces while satisfying diverse design prefer ences and physical constraints. However, existing floorplan generation methods typically assume empty building boundaries as input, whereas renovation requires reasoning over existing floorplans. Furthermore, floorplan renovation and furniture layout are commonly addressed as separate tasks, overlooking their inherent spatial dependency. Consequently, practical residential design must preserve immutable structural components while transforming editable spatial elements. It must also optimize furniture placement under design objectives and physical constraints. To address these challenges, we formulate residential design as a design-point-guided conditional generation task and propose RenoLayout, a framework that translates a pre-design residential state together with a design-point set into an executable Geometry-as-Code program. To enable controllable supervision, we construct a Design Map comprising 123 fine-grained design points covering structural renovation and furniture layout. We further introduce Diference-Grounded Design-Point Annotation to derive design-point sets automatically from before–after design pairs with VLM-based consistency verification. Reno-Layout is trained using a three-stage CPT–SFT–RL pipeline, progressively improving spatial reasoning, instruction following, and constraint compliance. We also introduce the Reno-Layout Benchmark, a set of 200 real-world before–after renovation cases with complete furniture layouts. To the best of our knowledge, it is the first benchmark for unified residential design. These cases include design-critical structural and building-service constraints. Extensive experiments evaluate design-point fulfillment, constraint compliance, living qual ity, and professional approval. On the RenoLayout Benchmark, RenoLayout achieves 91.63% Recall@1 and 61.00% Pass@1. These results demonstrate efective design-point fulfillment while maintaining constraint compliance.

## Introduction

Residential design transforms an existing home into a coordinated spatial solution that satisfies user requirements while respecting physical constraints. Unlike generation within an empty boundary, this task must preserve the exterior envelope, load-bearing walls, shafts, pipes, and drainage points. Meanwhile, editable partitions, room functions, and furniture layouts may be jointly modified.

As illustrated in Figure 1, floorplan methods focus on room partitioning and structural planning (Nauata et al. 2020; Yin et al. 2025; Qin, Weber, and Lu 2026), whereas furniturelayout methods target furniture arrangement with functional composition (Feng et al. 2023; Yang et al. 2025b; Jiang et al. 2026; Feng et al. 2026). Joint approaches such as HouseDifusion and Nestwork generate furnished layouts (Shabani, Hosseini, and Furukawa 2023; Miao, Cao, and Zhuang 2026), but their data generally describe only final designs rather than paired before–after cases with complete furnishing under design-critical constraints. Table 1 summarizes this supervision gap.

![](images/15d7b99adaee8b32c9c3a49e2abf3ea9d4c5ed569a296489c216fabbde68118b.jpg)  
Figure 1: Comparison of existing task-specific paradigms with RenoLayout. Existing methods treat floorplan generation separately from furniture layout, whereas RenoLayout unifies constraint-aware renovation with full-home furnishing.

The gap is consequential because structural decisions interact with scene-level choices. Living–Dining–Kitchen (LDK) integration depends on wall removability, whereas wet-function relocation is constrained by drainage. Storage additions must also preserve circulation and door clearance. Residential design therefore requires concrete operational design points, such as dry–wet separation or room merging, which translate high-level lifestyle descriptions into coordinate-level edits. Without paired design-point supervision, models may reproduce design patterns yet fail to perform valid transformations. Direct LLM generation also risks broken boundaries, collisions, and unintended changes to fixed elements (Rudman et al. 2025; Feng et al. 2023).

To address this issue, we propose RenoLayout, a uni-

<table><tr><td>Task setting Representative works</td><td>Pre-design state</td><td>Post-design solution</td><td>Full-home furnishing</td><td>Load-bearing walls</td><td>Doors &amp; windows</td><td>Pipes &amp; drainage</td></tr><tr><td>Floorplan generation House-GAN (Nauata et al. 2020); Yin et al. (Yin</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>et al. 2025); HouseMind (Qin, Weber, and Lu 2026) Scene layout generation LayoutGPT (Feng et al. 2023); OptiScene (Yang et al. 2025b); HoGLayout (Jiang et al. 2026);</td><td>x</td><td>x</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Joint floorplan-layout generation HouseDiffusion (Shabani, Hosseini, and Furukawa 2023); Nestwork (Miao, Cao, and Zhuang 2026)</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Unified residential design RenoLayout (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Note. ✓denotes explicit annotation or supervision, whereas ✗denotes information not explicitly modeled. Full-home furnishing covers furniture, appliances, storage, and local functional layouts. Building-service annotations cover pipes, drainage points, shafts, and flues.

Table 1: Comparison of data annotations in residential design. Unlike existing settings, RenoLayout provides paired pre-design states with post-design solutions, complemented by full-home furnishing as well as design-critical annotations for structure and building services.

fied framework that takes a pre-design residential state and a design-point set as input to generate the corresponding post-design residential solution. A 123-entry Design Map defines the shared vocabulary for structural-renovation and furniture-layout operations. Our diference-grounded, VLMaudited annotation protocol derives case-specific designpoint sets from observable before–after edits. The CPT– SFT–RL pipeline progressively learns spatial-code priors, instruction following, and constraint compliance. We further introduce the RenoLayout Benchmark, which contains paired designs with full-home furnishing. Its annotations cover structural elements, openings, pipes, and drainagerelated constraints. The benchmark supports a comprehensive evaluation of design-point fulfillment, constraint compliance, living quality, and professional approval.

Our contributions are summarized as follows:

• We formulate constraint-aware unified residential design as the transformation of an existing home into a coordinated structural and interior solution. To support this task, we introduce the RenoLayout Benchmark, which contains 200 paired residential-design cases with fullhome furnishing and building-service annotations. It will be publicly released.

• We propose a 123-entry Design Map together with a diference-grounded, VLM-audited protocol that converts observable transformations into controllable supervision.

• We present RenoLayout, an executable Geometry-as-Code framework trained through a staged CPT–SFT– RL pipeline. Our evaluation covers four complementary dimensions: design-point fulfillment, constraint compliance, living quality, and professional approval.

## Related Work

## Floorplan Generation

Floorplan generation has progressed from raster-based GANs (Nauata et al. 2020, 2021) to difusion and discrete-denoising methods that improve geometric controllability and topological consistency (Shabani, Hosseini, and Furukawa 2023; Chen, Deng, and Furukawa 2023; Qin et al. 2024; Su et al. 2024; Zeng et al. 2024). Recent LLM-based approaches enable language-guided refinement, next-room prediction, bubble-diagram augmentation, and multimodal tokenization (Luo et al. 2024; Li et al. 2024; Yin et al. 2025; Wei and Li 2024; Qin, Weber, and Lu 2026); concurrent work uses reinforcement learning to enforce numerical and connectivity constraints, while GRE-Dif supports constraint-aware vector generation and iterative editing (Lara et al. 2026; Wang et al. 2026).

## Scene Generation

LLMs have been adapted as visual planners that combine semantic planning with geometric reasoning. Layout-GPT (Feng et al. 2023) targets 2D object placement, while Holodeck and OptiScene extend language-guided planning to 3D scene construction (Yang et al. 2024, 2025b). Recent methods improve 3D layouts through hierarchical VLMguided optimization and autoregressive generation (Jiang et al. 2026; Feng et al. 2026). Architect-Ant instead targets editable furnishing of architectural floorplans through a coordinate-based DSL and preference optimization (Rodionov et al. 2026). More directly, Co-Layout jointly optimizes room and furniture layouts from LLM-derived constraints, while HomeWorld connects whole-home floorplan generation with hierarchical furnishing (Xiang et al. 2025; Li et al. 2026).

Although Co-Layout and HomeWorld move toward the joint generation of spatial structures and furniture layouts, they synthesize designs from requirements or generated floorplans instead of learning transformations from paired before– after designs (Xiang et al. 2025; Li et al. 2026). Architect-Ant assumes a fixed architectural shell and addresses furnishing without structural renovation (Rodionov et al. 2026). Their data primarily capture final solutions and therefore provide no explicit before–after supervision for operational renovation actions such as merging rooms, relocating wet functions, or preserving fixed services. In contrast, our setting begins with a real pre-design residential state and requires coordinated modification of partitions, room functions, and furniture while preserving load-bearing walls, openings, shafts, and service locations.

## Spatial Reasoning

To align semantic generation with geometric precision, recent approaches incorporate explicit constraints ranging from diferentiable optimization (Sun et al. 2025) to parametric generation (Wang et al. 2025). These mechanisms address ambiguities in model reasoning, where perspectives diverge: while (Ivanitskiy et al. 2023) suggest models possess emergent topological representations, (Rudman et al. 2025) argue they lack innate visual reasoning, necessitating structured guidance.

## RenoLayout

## Task and Data Construction

To align our task formulation with real-world residential design practice, where design begins with an existing home rather than an empty boundary, we formulate residential design as a design-point-guided transformation task instead of a from-scratch generation task. Given a complete pre-design residential state and a design-point set, our goal is to generate the corresponding post-design residential solution, comprising a renovated floorplan and full-home furniture layout, while preserving immutable structural elements and modifying editable components according to the specified design objectives.

Let X<sup>pre</sup> denote the pre-design residential state and X<sup>post</sup> denote the corresponding post-design residential solution. Both are represented as sets of geometric and semantic entities, including walls, doors, windows, rooms and furniture. We decompose the pre-design state into immutable structural elements and editable elements:

$$
X ^ { p r e } = S \cup E ^ { p r e } , \qquad S \cap E ^ { p r e } = \varnothing ,\tag{1}
$$

where S denotes immutable structural elements, such as exterior boundaries, load-bearing walls and shafts, and $E ^ { p r e }$ denotes editable elements, such as non-load-bearing partitions, room functions, movable door openings, furniture and local scene arrangements.

Let the design-point set be

$$
\mathcal { D } = \{ d _ { 1 } , d _ { 2 } , \ldots , d _ { k } \} ,\tag{2}
$$

The desired output is an executable design solution $\hat { X }$ post that balances model likelihood and design-point fulfillment under hard preservation and validity constraints:

$$
\begin{array} { r l r } {  { \hat { X } ^ { p o s t } = \arg \operatorname* { m a x } _ { X \in \mathcal { X } } \log P _ { \theta } ( X \mid X ^ { p r e } , \mathcal { D } ) + \lambda G ( \mathcal { D } , X ) } } \\ & { } & { \mathrm { s . t . } \Pi _ { S } ( X ) = S , \qquad V ( X ) = 1 , } \end{array}\tag{3}
$$

where X denotes the space of possible post-design solutions. The operator $\Pi _ { S }$ extracts the immutable structural and building-service elements from a candidate design X. The constraint $\Pi _ { S } ( X ) = S$ enforces exact preservation of their geometry and attributes. The binary predicate V (X) indicates whether X satisfies all other hard feasibility requirements, such as valid geometry, room connectivity, and collision-free furniture placement. We define normalized design-point fulfillment as

![](images/114cb6fabeb91107ad9b367c3c52ea5677e486af550973e6b3278fdf0880a8fe.jpg)  
Figure 2: Frequency of furniture-layout (top) and structuralrenovation (bottom) design points in the training (blue) and test (red) splits. Frequency is the occurrence count normalized by the total number of samples per split; design points are ordered by descending training-set frequency.

$$
G ( { \mathcal { D } } , X ) = { \frac { 1 } { | { \mathcal { D } } | } } \sum _ { d \in { \mathcal { D } } } \mathbb { I } [ d { \mathrm { ~ i s ~ f u l f i l l e d ~ i n ~ } } X ] ,\tag{4}
$$

where $G ( { \cal D } , X ) \in [ 0 , 1 ]$ , and $\lambda \geq 0$ controls its contribution relative to model likelihood. Equation 3 is a task-level specification rather than an inference-time optimization algorithm.

This formulation difers from prior floorplan generation and scene synthesis tasks in two important aspects. First, the input is a complete pre-design residential state rather than an empty boundary, a partial skeleton, or a text-only prompt. This makes the task closer to practical residential design workflows, where designers transform an existing home into an improved solution. Second, the control signal comprises professional design points distilled from real-world design cases and consolidated with the expertise of human interior designers, rather than free-form user requirements or personas. This enables explicit supervision, category-wise evaluation, and diference-grounded annotation from paired before–after design data.

## Design Map

Notably, the user requirements are often high-level, ambiguous, and expressed in non-technical language, making them unsuitable as direct operational controls for residential design. In professional practice, interior designers translate such requirements into concrete and actionable design decisions. Inspired by this professional practice, we abstract these decisions as design points and organize them into a hierarchical Design Map, which serves as the model’s control vocabulary instead of raw user requirements. Each fine-grained design point corresponds to a specific structural-renovation or furniture-layout operation.

![](images/c0bc2159b1ce4d5a722fb8f312f9cda060882a235ab4baff688081fc1c7eec84.jpg)  
Figure 3: Overview of RenoLayout. Left: paired designs are converted into diference-grounded design-point sets and verified through multi-VLM consistency auditing. Middle: the pre-design state is serialized as Geometry-as-Code and combined with design-point prompts, enabling Qwen3 to first produce a graph-based spatial plan and then generate executable structural-renovation and furniture-layout actions through the CPT–SFT–RL training pipeline.

We construct the Design Map by consolidating recurring transformations observed in paired design data with the domain expertise of professional interior designers. The resulting taxonomy comprises 123 fine-grained design points: 52 structural-renovation points and 71 furniture-layout points. Together, they cover both floorplan-level transformations and scene-level arrangements. Representative examples include LDK integration, dry–wet bathroom separation, bedroomto-closet conversion, entrance storage, appliance relocation, and furniture reachability optimization.

The Design Map is not used as a rule-based generator. Instead, it defines the control vocabulary, annotation schema, and evaluation interface for the transformation task. Reno-Layout learns from paired before–after examples to synthesize complete post-design Geometry-as-Code programs conditioned on selected design points.

## Diference-Grounded Design-Point Annotation

Given paired before–after design data, we construct designpoint labels from observable transformations rather than from free-form user descriptions. For each paired example $( X ^ { p r e } , X ^ { p o s t } )$ , we first compute a structured diference:

$$
\Delta X = \operatorname { D i f f } ( X ^ { p r e } , X ^ { p o s t } ) .\tag{5}
$$

The diference $\Delta X$ captures geometric and semantic edits at multiple levels, such as wall addition or removal, room merging or splitting, room-function reassignment, and furniture relocation.

We then map the extracted diferences to entries in the Design Map through deterministic rules:

$$
\mathcal { D } = R ( \Delta X , X ^ { p r e } , X ^ { p o s t } ) ,\tag{6}
$$

<table><tr><td>Category</td><td>Total</td><td>Avg./case</td><td>Std.</td><td>Min</td><td>Max</td></tr><tr><td>Renovation</td><td>875</td><td>4.38</td><td>2.33</td><td>1</td><td>13</td></tr><tr><td>Layout</td><td>2,177</td><td>10.88</td><td>3.40</td><td>0</td><td>23</td></tr><tr><td>Overall</td><td>3,052</td><td>15.26</td><td>4.71</td><td>2</td><td>31</td></tr></table>

Table 2: Per-case statistics and total counts of structuralrenovation and furniture-layout design-point annotations across the 200-case RenoLayout Benchmark.

where R denotes the rule-grounded mapper. Each rule links an observable before–after edit pattern to one or more finegrained design points. For example, if the kitchen partition is removed and the living, dining, and kitchen areas become connected, the pair is mapped to the LDK integration design point.

This rule-grounded mapping produces the candidate design-point set D directly from observable before–after edits rather than free-form descriptions.

## Annotation Audit and Quality Control

We then use multiple VLMs solely as annotation auditors. Given the pre-design image, post-design image, extracted diferences, and rule-derived labels, the auditors identify labels that are inconsistent with the observed edits, mismatched with the transformation type, duplicated, or conflicting.

Labels validated by the auditors are retained, while unsupported or ambiguous cases are removed or manually reviewed.

## Triplet Data Construction and RenoLayout Benchmark

Using the audited design-point labels, we construct supervised transformation triplets:

$$
( X ^ { p r e } , { \cal D } , X ^ { p o s t } ) ,\tag{7}
$$

Each sample contains floorplan-level information, including room boundaries, room functions, doors, windows, and structural elements, as well as scene-level information, including furniture, appliances, cabinets, and functional objects. Both $X ^ { p r e }$ and $X ^ { p o s t }$ are serialized into the same Geometry-as-Code representation, allowing the model to learn transformations over a unified executable spatial program. The RenoLayout Benchmark supports reproducible evaluation and contains 200 paired real-world residentialdesign cases that are strictly disjoint from all data used for CPT, SFT, and RL. The post-design solutions were produced by professional interior designers and are used as groundtruth designs. To ensure reliable and comprehensive designpoint coverage, we manually reviewed and filtered all cases. Figure 2 visualizes the frequency distribution of individual design points, while Table 2 provides complementary benchmark-level statistics, including the total numbers of structural-renovation and furniture-layout annotations and their per-case distributions.

## Continual Pre-Training (CPT)

General-purpose LLMs often struggle to comprehend the specialized vector representations required for residential design. To bridge the domain gap, we employ CPT, which efectively aligns the model with these geometric code constraints (Qi et al. 2025).

We conduct CPT on approximately 40 billion tokens, centered on vector-code data derived from roughly 10 million floorplans. The corpus incorporates two key strategies: Geometry-as-Code, which represents floorplans as Python code to leverage the spatial reasoning of LLMs, and a Data Mixing Strategy that combines mathematical (Fujii et al. 2025) and general-purpose data to mitigate catastrophic forgetting. See the supplementary material for details.

## Supervised Fine-Tuning (SFT) with Feature Enhancement

Although CPT equips the model with geometric priors, it does not fully align generation with complex design semantics. We therefore perform SFT on approximately 34,000 real-world floorplans, expanded to approximately 130,000 samples (0.7 billion tokens) through rotational augmentation. Additional data details are provided in the supplementary material.

Bounding-Box Encoding. To improve local geometric control, we replace the center-size representations of furniture and doors with explicit four-corner bounding boxes. Exposing object boundaries directly makes inter-object clearance, collision avoidance, and doorway occlusion easier to learn.

Graph-Planning CoT. To reduce topological and sizing errors, we adopt a coarse-to-fine planning process. The model first predicts a topological adjacency graph (Bubble Diagram (Nauata et al. 2020; Luo et al. 2024)) to establish room-level spatial logic, and then generates detailed spatial actions conditioned on this structural plan.

## Reinforcement Learning

Of-policy Preference Optimization via DPO. We first apply DPO (Rafailov et al. 2023) to establish an ofline preference guardrail covering both physical validity and instruction following. We construct fixed chosen–rejected pairs through a two-stage selection procedure. In the first stage, the groundtruth design y<sub>GT</sub> serves as the chosen upper-bound anchor, while a predicted candidate y is eligible as rejected only if it fails physical quality control and achieves lower designpoint recall than y<sub>GT</sub>. Among eligible candidates, we select the one with the smallest length diference from the ground truth, reducing spurious preferences based on output length.

In the second stage, we form fine-grained preference pairs among predicted candidates. For an output $y ,$ let $e ( y )$ denote the number of detected violations. We define its physical-validity reward uniformly as $R _ { p h y s } ( y ) =$ max $( 0 , \dot { 1 } - \dot { e } ( y ) / 1 0 )$ , which equals 1 when all quality-contro checks pass. We rank candidates using

$$
S ( y ) = \omega _ { p h y s } R _ { p h y s } ( y ) + \omega _ { i n s t } R _ { i n s t } ( y ) ,\tag{8}
$$

where $R _ { i n s t } ( y )$ is design-point recall. We set the physicalvalidity and instruction-following weights to $\omega _ { p h y s } = 0 . 8$ and $\omega _ { i n s t } = 0 . 2$ , respectively. A pair is retained only when its score advantage is at least 0.3 and its length diference is no more than 15% of the mean pair length. We then use Hungarian matching to maximize the total preference advantage across non-redundant pairs. This construction emphasizes the weaker physical-validity dimension while preserving instruction following and filtering noisy or length-biased comparisons.

On-policy Optimization via GDPO. Residential design is inherently a multi-objective optimization problem involving trade-ofs among format correctness, instruction following, physical validity, and item proportion. These objectives difer substantially in reward scale, variance, and optimization dificulty. Standard GRPO (Shao et al. 2024) aggregates multiple reward components before group-wise normalization, which can collapse distinct reward combinations into similar advantages and allow high-variance components to dominate the policy update, suppressing informative signals from harder objectives. We therefore adopt GDPO (Liu et al. 2026), which independently normalizes each reward component before aggregating their advantages. This rewarddecoupled formulation preserves objective-specific learning signals and enables more balanced optimization across competing residential-design objectives.

Using samples generated from the current policy, we apply direct rule-based feedback to format correctness, instruction following, physical validity, and item proportion. For each sampled output i, GDPO independently converts each reward component into a group-normalized advantage:

$$
A _ { i , k } = \frac { R _ { i , k } - \mu _ { k } ^ { \mathcal { G } } } { \sigma _ { k } ^ { \mathcal { G } } + \epsilon } , \quad k \in \mathcal { K } ,\tag{9}
$$

where $\mathcal { K } = \{ \mathrm { f o r m } , \mathrm { i n s t } , \mathrm { p h y s } , \mathrm { p r o p } \}$ , and $\mu _ { k } ^ { \mathcal { G } }$ and $\sigma _ { k } ^ { \mathcal { G } }$ are the mean and standard deviation of reward component $k$ within rollout group ${ \mathcal { G } } .$ The final advantage used for policy optimization is

$$
A _ { i } ^ { \mathrm { G D P O } } = \sum _ { k \in \mathcal { K } } \omega _ { k } A _ { i , k } .\tag{10}
$$

The four reward components respectively evaluate outputformat correctness $( \bar { R _ { f o r m } } )$ , design-point recall $( R _ { i n s t } )$ ， physical validity $( R _ { p h y s } )$ , and item proportion $( R _ { p r o p } )$ . We set $\omega _ { f o r m } = \omega _ { p h y s } = \omega _ { p r o p } = 1 . 0$ and $\omega _ { i n s t } = 0 . 5$ . Because the physical objectives are more dificult to optimize, the lower instruction-following weight prevents the policy from overemphasizing this objective at the expense of physical validity and appropriate item configuration.

## Experiments

## Metrics and Training Details

Evaluation setting. We evaluate all methods on the Reno-Layout Benchmark. For each case $i ,$ a method receives a predesign residential state $X _ { i } ^ { p r e }$ and a design-point set $\mathcal { D } _ { i }$ , and produces a complete post-design residential solution $\hat { X } _ { i } ^ { p o s t }$ The evaluation addresses four complementary questions: whether the requested design points are fulfilled, whether the transformation satisfies design and physical constraints, whether the resulting home provides high living quality, and whether the overall design receives professional approval.

Design-Point Fulfillment. We evaluate instruction following through design-point fulfillment using Recall@1. Let $z _ { i , k } ^ { \mathrm { ( r u l e ) } } \in \{ 0 , 1 \}$ indicate whether the rule-based evaluator judges the k-th requested design point in case i to be realized in the first generated post-design solution, and let $M _ { i } = | \mathcal { D } _ { i } |$ denote the number of requested design points for that case. Recall@1 first computes the proportion of fulfilled design points within each case and then averages these case-level recalls over all N = 200 cases:

$$
\mathrm { R e c a l l @ 1 } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { M _ { i } } \sum _ { k = 1 } ^ { M _ { i } } z _ { i , k } ^ { ( \mathrm { r u l e } ) } .\tag{11}
$$

Constraint Compliance (Pass@1). We use 22 binary checks grouped into Geometric Validity, Structural Compliance, Functional Integrity, and Collision-Free Placement. Let $q _ { i , m } \in \{ 0 , 1 \}$ indicate whether case i passes check m. Each category rate is the percentage of cases that pass every check in that category, while Pass@1 requires all 22 checks to pass simultaneously:

$$
\mathrm { P a s s @ 1 } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left[ \sum _ { m = 1 } ^ { 2 2 } q _ { i , m } = 2 2 \right] .\tag{12}
$$

Failure of any single check makes the entire case fail. All five quantities are percentages, and higher values are better.

VLM-Based Habitability Score (VLM-HS). We use four independent VLM evaluators: Seed2.1 Pro (Volcengine 2026), GLM-5V-Turbo (GLM-5-Team 2026), GPT-5.5 (OpenAI 2026), and Gemini-3.1-Flash-Lite (Google 2026). Each separately scores functional zoning, spatial layout, circulation quality, amenity configuration, design-point implementation, and physical compliance. For each evaluator, the first five dimensions form a weighted subjective score, while physical compliance is fused with a deterministic checker for P0/P1 violations. Predictions and ground-truth plans are scored independently before case-matched deficit calibration, yielding four model-specific VLM-HS values without cross-model averaging. Complete formulas, parameters, rubrics, and prompts are provided in the supplementary material.

Professional evaluation protocol. To assess professional approval alongside the automatic metrics, three interior designers independently evaluated anonymized and randomly ordered RenoLayout outputs and corresponding ground-truth designs for all 200 cases using the Professional Approval Score (PAS; 0–10); scoring details are provided in the supplementary material.

Training Details. Our model is built upon Qwen3-8B (Yang et al. 2025a). In the CPT stage, the model was trained for 2 epochs with a global batch size of 32 and a learning rate of $1 ^ { ^ { \bullet } \times 1 0 ^ { - 5 } }$ . Subsequently, during SFT, we reduced the batch size to 16 and the learning rate to $5 \times 1 0 ^ { - 6 }$ , training for 2 epochs. For the reinforcement learning stages, we train GDPO/DPO for 2 epochs with a global batch size of 64 and a learning rate of $3 ^ { ^ { \bullet } } \times 1 0 ^ { - 6 }$ . To maintain training stability, we set the group size to 8, the KL coeficient to 0.05, and the sampling temperature to 0.6.

<table><tr><td colspan="3">Settings</td><td colspan="2">Overall Metrics</td><td colspan="4">VLM-HS by Evaluator</td></tr><tr><td>CPT</td><td>Box</td><td>CoT</td><td>Recall@1↑</td><td>Pass@1↑</td><td>Seed</td><td>GLM</td><td>GPT</td><td>Gemini</td></tr><tr><td>x</td><td>x</td><td>x</td><td>80.92%</td><td>27.00%</td><td>5.12</td><td>4.97</td><td>5.60</td><td>5.07</td></tr><tr><td>x</td><td>√</td><td>x</td><td>81.13%</td><td>38.50%</td><td>5.86</td><td>5.85</td><td>6.39</td><td>5.77</td></tr><tr><td>x</td><td>√</td><td>√</td><td>84.47%</td><td>42.00%</td><td>5.98</td><td>5.86</td><td>6.46</td><td>5.78</td></tr><tr><td>√</td><td>x</td><td>√</td><td>88.65%</td><td>31.50%</td><td>6.32</td><td>6.01</td><td>6.69</td><td>6.19</td></tr><tr><td>√</td><td>√</td><td>x</td><td>86.37%</td><td>45.00%</td><td>6.61</td><td>6.60</td><td>7.10</td><td>6.40</td></tr><tr><td>√</td><td>√</td><td>√</td><td>88.67%</td><td>49.00%</td><td>6.82</td><td>6.75</td><td>7.22</td><td>6.45</td></tr><tr><td colspan="3">Settings</td><td colspan="6">Physical Pass Rates</td></tr><tr><td>CPT</td><td>Box</td><td>CoT</td><td> $\mathcal { P } _ { \mathrm { g e o m } } \uparrow$ </td><td> $\mathcal { P } _ { \mathrm { s t r u c t } }$ </td><td>↑</td><td> $\mathcal { P } _ { \mathrm { f u n c } }$ </td><td>↑</td><td> $\mathcal { P } _ { \mathrm { c o l l } } \uparrow$ </td></tr><tr><td>x</td><td>x</td><td>x</td><td>93.00%</td><td>79.00%</td><td></td><td>89.00%</td><td></td><td>32.00%</td></tr><tr><td>x</td><td>√</td><td>x</td><td>92.00%</td><td>80.50%</td><td></td><td>89.50%</td><td></td><td>49.00%</td></tr><tr><td>x</td><td>√</td><td>√</td><td>95.00%</td><td>78.00%</td><td></td><td>94.00%</td><td></td><td>53.00%</td></tr><tr><td>√</td><td>x</td><td>√</td><td>97.50%</td><td>79.00%</td><td></td><td>95.50%</td><td></td><td>39.00%</td></tr><tr><td>√</td><td>√</td><td>x</td><td>97.50%</td><td>76.50%</td><td></td><td>95.50%</td><td></td><td>57.00%</td></tr><tr><td>√</td><td>√</td><td>√</td><td>98.00%</td><td></td><td>81.00%</td><td>96.50%</td><td></td><td>57.00%</td></tr></table>

Table 3: Feature ablation of CPT, bounding-box encoding (Box), and graph-planning CoT. All metrics except VLM-HS (0–10) are percentages.

## Feature Analysis

Table 3 incrementally introduces bounding-box features and CoT with and without CPT. The comparison tests whether domain pre-training improves design-point fulfillment, constraint compliance, and living quality, whether explicit bounding boxes strengthen local geometric control, and whether CoT benefits global planning and final living quality.

The components provide complementary benefits with distinct tradeofs. Without CPT, adding CoT to the boundingbox representation increases Recall@1 from 81.13% to 84.47%, Pass@1 from 38.50% to 42.00%, and collisionfree placement from 49.00% to 53.00%, although structural compliance decreases from 80.50% to 78.00%. Under CPT, adding CoT raises Recall@1 from 86.37% to 88.67%, Pass@1 from 45.00% to 49.00%, and structural compliance from 76.50% to 81.00%, while collision-free placement remains at 57.00%. The four VLM-HS scores, ordered as Seed, GLM, GPT, and Gemini, increase from (6.61, 6.60, 7.10, 6.40) to (6.82, 6.75, 7.22, 6.45). Compared with the corresponding Box+CoT model without CPT, the complete configuration improves Recall@1 and Pass@1 by 4.20 and 7.00 percentage points, respectively. These results suggest that CoT primarily strengthens global designpoint fulfillment and overall constraint compliance, while bounding boxes support local placement quality and CPT supplies complementary domain priors. Figure 4 further provides representative visual evidence for these complementary efects.

## Ablation on Diferent RL Strategies: DPO vs. GDPO

Table 4 compares ofline (DPO) and online (GDPO) optimization and highlights their diferent efects on residential design. DPO yields a larger improvement in structural compliance, indicating that preference supervision from fixed comparison pairs primarily benefits global structural consistency. In contrast, GDPO produces a larger improvement in collision-free placement, consistent with its online optimization objective that directly incorporates geometric feedback during policy refinement, while leading to a slight reduction in functional integrity. These observations suggest that DPO mainly improves global structural validity, whereas GDPO is more efective at correcting instruction-specific geometric errors.

![](images/b8b378da0201b8ce108e9daea5a3c528bc196a068306d9fbb29855a989869a65.jpg)  
Figure 4: Qualitative efects of bounding boxes on local placement and CoT on spatial planning.

<table><tr><td>Method</td><td colspan="2">Overall Metrics</td><td colspan="4">VLM-HS by Evaluator</td></tr><tr><td></td><td>Recall@1↑</td><td>Pass@1↑</td><td>Seed</td><td>GLM</td><td>GPT</td><td>Gemini</td></tr><tr><td>RenoLayout-SFT</td><td>88.67%</td><td>49.00%</td><td>6.82</td><td>6.75</td><td>7.22</td><td>6.45</td></tr><tr><td>+ DPO</td><td>89.37%</td><td>55.50%</td><td>6.92</td><td>6.90</td><td>7.36</td><td>6.64</td></tr><tr><td>+ GDPO</td><td>90.89%</td><td>53.00%</td><td>6.87</td><td>6.81</td><td>7.20</td><td>6.77</td></tr><tr><td>+ DPO + GDPO</td><td>91.63%</td><td>61.00%</td><td>7.26</td><td>7.32</td><td>7.72</td><td>7.22</td></tr><tr><td>+ GDPO + DPO</td><td>90.42%</td><td>61.50%</td><td>7.17</td><td>7.23</td><td>7.70</td><td>7.16</td></tr><tr><td>Method</td><td colspan="6">Physical Pass Rates</td></tr><tr><td></td><td> $\mathcal { P } _ { \mathrm { g e o m } }$  ←</td><td> $\mathcal { P } _ { \mathrm { s t r u c t } } \ \uparrow$ </td><td></td><td> $\mathcal { P } _ { \mathrm { f u n c } } \ 1$ </td><td>人</td><td> $\mathcal { P } _ { \mathrm { c o l l } } \uparrow$ </td></tr><tr><td>RenoLayout-SFT</td><td>98.00%</td><td>81.00%</td><td></td><td>96.50%</td><td></td><td>57.00%</td></tr><tr><td>+ DPO</td><td>98.00%</td><td>84.50%</td><td></td><td>96.00%</td><td></td><td>63.50%</td></tr><tr><td>+ GDPO</td><td>99.00%</td><td>80.50%</td><td></td><td>95.50%</td><td></td><td>66.50%</td></tr><tr><td>+ DPO + GDPO</td><td>99.00%</td><td>85.00%</td><td></td><td>96.50%</td><td></td><td>69.50%</td></tr><tr><td>+ GDPO + DPO</td><td>98.50%</td><td>85.50%</td><td></td><td>96.00%</td><td></td><td>70.00%</td></tr></table>

Table 4: RL post-training ablation of DPO and GDPO. Al metrics except VLM-HS (0–10) are percentages.

The results further indicate that the optimization order influences the final policy. Applying DPO before GDPO achieves the highest Recall@1 (91.63%), together with 61.00% Pass@1 and the highest VLM-HS across Seed, GLM, GPT, and Gemini (7.26, 7.32, 7.72, and 7.22, respectively). Reversing the order yields the highest Pass@1 (61.50%), structural compliance (85.50%), and collision-free placement (70.00%), but reduces Recall@1 to 90.42% and yields lower scores on all four VLM-HS evaluators. Based on its stronger design-point fulfillment and overall balance, we adopt the DPO–GDPO variant as the final RenoLayout model and denote its pre-RL checkpoint as RenoLayout-

<table><tr><td>Method</td><td>Designer 1</td><td>Designer 2</td><td>Designer 3</td><td>Mean↑</td><td>Std.↓</td></tr><tr><td>RenoLayout</td><td>5.50</td><td>5.59</td><td>5.19</td><td>5.43</td><td>0.17</td></tr><tr><td>Ground Truth (GT)</td><td>7.06</td><td>7.35</td><td>6.58</td><td>7.00</td><td>0.32</td></tr></table>

Table 5: Professional Approval Score (PAS; 0–10).

![](images/6d0b441e7294db29faddbebf72860db557dc0d4f131ce3ff15bf31a07da74d6a.jpg)  
Figure 5: Visual comparison with LayoutGPT. Red arrows indicate furniture accessibility issues, and blue arrows indicate furniture misalignment with walls. RenoLayout produces more accessible and better-aligned room layouts in these examples.

## SFT.

## Professional Evaluation

In the blinded evaluation (Table 5), three professional interior designers rated RenoLayout outputs at 5.43 on average, compared with 7.00 for the corresponding designer-produced ground-truth layouts.

## Qualitative Comparison with LayoutGPT

We also compare room-level furniture arrangements with LayoutGPT (Feng et al. 2023). Following its standard setting, the comparison uses one bedroom and one living room per sample. The examples complement the unified evaluation by illustrating furniture reachability, wall alignment, and collision-related errors.

## Conclusion

We introduced RenoLayout, a unified framework for constraint-aware residential design. By combining the Design Map, diference-grounded design-point annotation, Geometry-as-Code representation, and CPT–SFT–RL train ing, RenoLayout jointly improves instruction following, constraint compliance, and living quality. The final RenoLayout achieves 91.63% Recall@1 and 61.00% Pass@1, demonstrating the efectiveness of combining preference optimization with multi-rule-guided reinforcement learning.

## References

Chen, J.; Deng, R.; and Furukawa, Y. 2023. Polydifuse: Polygonal shape reconstruction via guided set difusion models. Advances in Neural Information Processing Systems, 36: 1863–1888.

Feng, H.; Niu, Y.; Huang, Z.; Sun, Y.-T.; Guo, C.; Peng, Y.; and Sheng, L. 2026. Repurposing 3D Generative Model for Autoregressive Layout Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3231–3243.

Feng, W.; Zhu, W.; Fu, T.-j.; Jampani, V.; Akula, A.; He, X.; Basu, S.; Wang, X. E.; and Wang, W. Y. 2023. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36: 18225–18250.

Fujii, K.; Tajima, Y.; Mizuki, S.; Shimada, H.; Shiotani, T.; Saito, K.; Ohi, M.; Kawamura, M.; Nakamura, T.; Okamoto, T.; Ishida, S.; Hattori, K.; Ma, Y.; Takamura, H.; Yokota, R.; and Okazaki, N. 2025. Rewriting Pre-Training Data Boosts LLM Performance in Math and Code. arXiv:2505.02881.

GLM-5-Team. 2026. GLM-5: from Vibe Coding to Agentic Engineering. arXiv:2602.15763.

Google. 2026. Gemini 3.1 Flash-Lite. https://ai.google.dev/ gemini-api/docs/models/gemini-3.1-flash-lite. Accessed: 2026-07-28.

Ivanitskiy, M. I.; Spies, A. F.; Räuker, T.; Corlouer, G.; Mathwin, C.; Quirke, L.; Rager, C.; Shah, R.; Valentine, D.; Behn, C. D.; et al. 2023. Structured world representations in mazesolving transformers. arXiv preprint arXiv:2312.02566.

Jiang, H.; Zhang, D.; Weng, D.; Song, W.; and Duh, H. B.- L. 2026. HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 31586–31596.

Lara, L.; Milios, A.; Luo, Z. H.; Sharma, A.; Luo, G. Y.; Beckham, C.; Golemo, F.; and Pal, C. 2026. Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards. arXiv:2605.14117.

Li, J.; Luo, Y.; Lu, S.; Zhang, J.; Wang, J.; Guo, R.; and Wang, S. 2024. ChatDesign: Bootstrapping generative floor plan design with pre-trained large language models. In Proceedings ofthe 29th International Conference ofthe Association for Computer Aided Architectural Design Research in Asia (CAADRIA), volume 1, 99–108.

Li, W.; Ju, X.; Qin, Z.; Fang, R.; and Li, H. 2026. Home-World: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes. arXiv:2606.06390.

Liu, S.-Y.; Dong, X.; Lu, X.; Diao, S.; Belcak, P.; Liu, M.; Chen, M.-H.; Yin, H.; Wang, Y.-C. F.; Cheng, K.-T.; et al. 2026. GDPO: Group Reward-Decoupled Normalization Policy Optimization for Multi-Reward RL Optimization. arXiv preprint arXiv:2601.05242.

Luo, Z. H.; Lara, L.; Luo, G. Y.; Golemo, F.; Beckham, C.; and Pal, C. 2024. Dstruct2design: Data and benchmarks for data structure driven generative floor plan design. arXiv preprint arXiv:2407.15723.

Miao, S.; Cao, B.; and Zhuang, J. 2026. Nestwork: Conditional 3D Furnished House Layout Generation through Latent Heterogeneous Graph Difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27093–27103.

Nauata, N.; Chang, K.-H.; Cheng, C.-Y.; Mori, G.; and Furukawa, Y. 2020. House-gan: Relational generative adversarial networks for graph-constrained house layout generation. In European Conference on Computer Vision, 162–177. Springer.

Nauata, N.; Hosseini, S.; Chang, K.-H.; Chu, H.; Cheng, C.-Y.; and Furukawa, Y. 2021. House-gan++: Generative adversarial layout refinement network towards intelligent computational agent for professional architects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13632–13641.

OpenAI. 2026. GPT-5.5 Model. https://developers.openai. com/api/docs/models/gpt-5.5. Accessed: 2026-07-28.

Qi, Z.; Nie, F.; Alahi, A.; Zou, J.; Lakkaraju, H.; Du, Y.; Xing, E.; Kakade, S.; and Zhang, H. 2025. EvoLM: In Search of Lost Language Model Training Dynamics. arXiv preprint arXiv:2506.16029.

Qin, S.; He, C.; Chen, Q.; Yang, S.; Liao, W.; Gu, Y.; and Lu, X. 2024. ChatHouseDifusion: Prompt-Guided Generation and Editing of Floor Plans. arXiv:2410.11908.

Qin, S.; Weber, R. E.; and Lu, X. 2026. Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10430–10440.

Rafailov, R.; Sharma, A.; Mitchell, E.; Ermon, S.; Manning, C. D.; and Finn, C. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36: 53728–53741.

Rodionov, F.; Cvejic, A.; Birsak, M.; Femiani, J.; and Wonka, P. 2026. Architect-Ant: Editable Automatic Furnishing of Architectural Floor Plans. arXiv:2606.10953.

Roziere, B.; Gehring, J.; Gloeckle, F.; Sootla, S.; Gat, I.; Tan, X. E.; Adi, Y.; Liu, J.; Sauvestre, R.; Remez, T.; et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.

Rudman, W.; Golovanevsky, M.; Bar, A.; Palit, V.; LeCun, Y.; Eickhof, C.; and Singh, R. 2025. Forgotten polygons: Multimodal large language models are shape-blind. In Findings of the Association for Computational Linguistics: ACL 2025, 11983–11998.

Shabani, M. A.; Hosseini, S.; and Furukawa, Y. 2023. Housedifusion: Vector floorplan generation via a difusion model with discrete and continuous denoising. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5466–5475.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Su, P.; Lu, W.; Chen, J.; and Hong, S. 2024. Floor plan graph learning for generative design of residential buildings: a discrete denoising difusion model. Building Research & Information, 52(6): 627–643.

Sun, F.-Y.; Liu, W.; Gu, S.; Lim, D.; Bhat, G.; Tombari, F.; Li, M.; Haber, N.; and Wu, J. 2025. Layoutvlm: Diferentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 29469–29478.

Volcengine. 2026. Doubao Model Service. https://www. volcengine.com/product/doubao. Accessed: 2026-07-28.

Wang, J.; Xiong, H.; Yan, Z.; Gong, M.; and Huang, H. 2026. GRE-Dif: Gaussian Room Embeddings for Structured Layout Difusion. arXiv:2607.08086.

Wang, X.; Zheng, J.; Hu, Y.; Zhu, H.; Yu, Q.; and Zhou, Z. 2025. From 2d cad drawings to 3d parametric models: A vision-language approach. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 7961–7969.

Wei, Y.; and Li, X. 2024. Graph-Augmented Text-Based Floorplan Generation. In 2024 International Conference on Automation in Manufacturing, Transportation and Logistics (ICaMaL), 1–9. IEEE.

Xiang, C.; Bao, R.; Feng, B.; Wu, W.; Liu, Z.; Guan, Y.; and Liu, L. 2025. Co-Layout: LLM-Driven Co-Optimization for Interior Layout. arXiv:2511.12474.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yang, Y.; Luo, Z.; Ding, T.; Lu, J.; Gao, M.; Yang, J.; Sanchez, V.; and Zheng, F. 2025b. OptiScene: LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yang, Y.; Sun, F.-Y.; Weihs, L.; VanderBilt, E.; Herrasti, A.; Han, W.; Wu, J.; Haber, N.; Krishna, R.; Liu, L.; et al. 2024. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16227–16237.

Yin, J.; Zeng, P.; Zhong, J.; Li, P.; Zhang, M.; Luo, R.; and Lu, S. 2025. FloorPlan-DeepSeek (FPDS): A multimodal approach to floorplan generation using vector-based next room prediction. arXiv preprint arXiv:2506.21562.

Zeng, P.; Gao, W.; Yin, J.; Xu, P.; and Lu, S. 2024. Residential floor plans: Multi-conditional automatic generation using difusion models. Automation in Construction, 162: 105374.

## Appendix A Training Data and Data Mixture

Data Sources and Usage. The training data used across CPT, SFT, and RL are derived from real-world residential floorplans. All records are anonymized to remove personally identifiable information as well as location-related metadata. These training corpora cannot be publicly released because they are derived from proprietary commercial data governed by data-use agreements and are subject to residential privacy constraints. Existing public alternatives do not provide paired before–after residential designs with complete furnishing. They also lack the design-critical structural and buildingservice annotations required for training RenoLayout. The RenoLayout Benchmark contains 200 quality-filtered paired real-world residential-design cases for reproducible evaluation and will be publicly released through the OpenReview submission materials. All benchmark cases are strictly disjoint from the data used for CPT, SFT, and RL.

The SFT corpus contains approximately 34,000 real-world floorplans before augmentation. We apply rotational augmentation to the paired floorplans, yielding approximately 130,000 SFT training samples (0.7 billion tokens).

We serialize floorplan vectors as executable Python code so that spatial dependencies, such as room adjacency and wall closure, are represented through explicit program structure. CPT uses approximately 40 billion tokens partitioned into the subsets summarized in Table 6.

Data Mixing Strategy To mitigate catastrophic forgetting (Roziere et al. 2023), we mix general-purpose data with domain-specific vector code. The four components serve complementary roles:

<table><tr><td>Component</td><td>Ratio Rationale</td><td></td></tr><tr><td>Vector Code</td><td></td><td>90% Core Geometry-as-Code generation tasks for spatial genera- tion and interpretation.</td></tr><tr><td>General Python (Fujii</td><td>5%</td><td>Prevents syntax degradation and maintains code-structure flu- ency.</td></tr><tr><td>et al. 2025) Mathematical</td><td>4%</td><td>Enhances logical reasoning and numerical precision for coor-</td></tr><tr><td>Data (Fujii et al. 2025)</td><td></td><td>dinate arithmetic.</td></tr><tr><td>Domain NL (from the internet)</td><td>1%</td><td>Ensures robust alignment between user instructions and geo- metric execution.</td></tr></table>

Table 6: Composition of the CPT Training Corpus.

## Appendix B Experimental Setup and Reproducibility

All data shufling, training, and generation used a fixed random seed of 42. We retained one fixed hyperparameter configuration per training stage because training was stable. All stages used the AdamW optimizer with a weight decay of 0.01 under bfloat16 (BF16) precision. SFT used a warmup ratio of 0.1. RL sampling used a temperature of 0.6; evaluationtime inference used a temperature of 0.3 and top-p of 0.8. For each evaluated model configuration, we independently sampled each of the 200 benchmark cases 10 times, yielding 2,000 generations per configuration; consistent with their definitions, the reported Recall@1 and Pass@1 evaluate the first sample for each case. Hardware varied by stage: 8–16 NVIDIA H200 GPUs, 180 Intel Xeon Platinum 8558 CPU cores, and 1,980 GB of system memory. The software environment was Debian-based GNU/Linux (kernel 5.4.250-9, x86\_64), Python 3.10, PyTorch 2.4.0 with CUDA 12.4, and Transformers 4.51.3.

## Appendix C Complete Design Map

The complete Design Map contains 123 fine-grained design points, comprising 52 structural-renovation design points and 71 furniture-layout design points. The full taxonomy is listed below using the names employed throughout the paper and evaluation pipeline.

## Appendix D Evaluation Metrics

Table 8 summarizes the evaluation framework. Recall@1 measures design-point fulfillment; Pass@1 requires all 22 constraint-compliance checks to pass; and VLM-HS combines five subjective dimensions with a sixth physicalcompliance dimension. PAS provides an overall professional comparison between RenoLayout outputs and corresponding ground-truth designs.

Professional Approval Score (PAS). For each case, each professional designer independently assigned eight 0–10 scores covering Structural Integrity & Compliance, Connectivity & Accessibility, Functional Layout, Spatial Layout, Functional Zoning, Circulation Quality, Amenity Configuration, and Layout Configuration. The case-level PAS is their equally weighted mean:

$$
\mathrm { P A S } _ { i , d } = \frac { 1 } { 8 } \sum _ { k = 1 } ^ { 8 } s _ { i , d , k } ,\tag{13}
$$

where $s _ { i , d , k }$ is the score assigned to case i by designer d for dimension k. All 200 benchmark cases were included without filtering. Each designer-level PAS is the mean over the 200 cases. The reported overall Mean and Std. are computed across the three designer-level scores.

Each constraint-compliance check produces a binary violation flag. A submetric passes only when none of its constituent flags are triggered, and Pass@1 requires all 22 checks across the four submetrics to pass. Table 9 reports the exact rule composition used by the evaluator.

## Appendix E Annotation Auditing

We use VLMs for two distinct purposes. First, multiple VLMs, including Seed, GPT, and Gemini, serve as annotation auditors. Given the pre-design image, post-design image, extracted before–after diferences, and rule-derived designpoint labels, they identify unsupported, conflicting, ambiguous, or duplicated labels. They do not generate labels, and ambiguous cases are subjected to manual inspection.

A four-VLM panel comprises Seed2.1 Pro (Volcengine 2026), GLM-5V-Turbo (GLM-5-Team 2026), GPT-5.5 (OpenAI 2026), and Gemini-3.1-Flash-Lite (Google 2026). This panel is employed for VLM-HS evaluation. Appendix Appendix H details the inputs, scoring dimensions, aggregation scheme, and complete prompts.

<table><tr><td colspan="4">Structural-Renovation Design Points (52)</td></tr><tr><td>No. Design Point</td><td></td><td>No. Design Point</td><td>No. Design Point</td></tr><tr><td>1 Bedroom Storage</td><td></td><td>19 Remove Entryway</td><td>36 4BR to 3BR</td></tr><tr><td>2 Door/Window Adjustment</td><td></td><td>20 2BR to 3BR</td><td>37 Master Suite</td></tr><tr><td></td><td>3 Wet-Dry Separation (2-Zone)</td><td>21 Add Study in Living Room</td><td>38 Wet-Dry Separation (3-Zone)</td></tr><tr><td>4 Convert to Utility Area</td><td></td><td>22 Enlarge Bedroom</td><td>39 Kitchen-Dining Integration</td></tr><tr><td>5 Entryway Storage</td><td></td><td>23 3BR to 2BR</td><td>40 Corridor Storage</td></tr><tr><td></td><td>6 Entryway/Corridor Space Reallocation</td><td>24 Shrink Kitchen</td><td>41 Storage Room Organization</td></tr><tr><td></td><td>7 Convert to Open Kitchen</td><td>25 Enlarge Entryway</td><td>42 3BR to 4BR</td></tr><tr><td></td><td>8 LDK Integration</td><td>26 1BR to 2BR</td><td>43 Study &amp; Storage Room</td></tr><tr><td></td><td>9 Merge Balcony into Adjacent Room</td><td>27 Entryway Renovation</td><td>44 Living-Dining Integration</td></tr><tr><td></td><td>10 Convert to Semi-Open Kitchen</td><td>28 Merge Kitchen &amp; Balcony</td><td>45 Allocate Corridor to Bedroom</td></tr><tr><td></td><td>11 Enlarge Bathroom</td><td>29 Living Room Storage</td><td>46 Allocate Corridor to Bathroom</td></tr><tr><td></td><td>12 Shared Wall Zigzag Remodel</td><td>30 2BR to 1BR</td><td>47 Convert to Leisure Space</td></tr><tr><td></td><td>13 Dining Room Storage</td><td>31 Multi-Zone Utilization</td><td>48 Convert Storage to Study</td></tr><tr><td></td><td>14 Add Ensuite to Bedroom</td><td>32 Add Storage Room</td><td>49 Chinese-Western Kitchen Zoning</td></tr><tr><td></td><td>15 Convert Bedroom to Utility Room</td><td>33 Add Walk-in Closet to Bedroom</td><td>50 Convert to Storage Space</td></tr><tr><td></td><td>16 Enlarge Kitchen</td><td>34 Shrink Entryway</td><td>51 Wet-Dry Separation (4-Zone)</td></tr><tr><td>18 Balcony Storage</td><td>17 Incorporate into Other Space</td><td>35 Add Closet &amp; Ensuite to Bedroom</td><td>52 Merge into Other Space</td></tr><tr><td colspan="4"></td></tr><tr><td>No. Design Point</td><td>No. Design Point</td><td></td><td>No. Design Point</td></tr><tr><td>1 Bathroom Vanity - Built-in Sink</td><td></td><td>25 Kitchen Cabinet - U-Shaped</td><td>49 Living Room - Bar Counter Divider</td></tr><tr><td></td><td>2 Bathroom Vanity - Freestanding Cabinet</td><td>26 Bedroom Bed - Tatami w/ Side Desk</td><td>50 Wide Hall - Sofa as Center Divider</td></tr><tr><td></td><td>3 Standard Set - Bed, Nightstand, Wardrobe</td><td>27 Wide Hall - Parallel Layout</td><td></td></tr><tr><td></td><td>4 Living Room - Corner Tea Table</td><td>28 Wide Hall - Sofa &amp; Table Parallel</td><td>51 Standard Layout - Full-Wall Bookshelf</td></tr><tr><td></td><td>5 Kitchen Cabinet - L-Shaped</td><td>29 Kitchen Cabinet - Galley</td><td>52 Kitchen Bar - Bar Extends Countertop</td></tr><tr><td></td><td>6 Kitchen Workflow - Golden Triangle</td><td>30 Bedroom Closet - Wardrobe Partition</td><td>53 Bathroom Toilet - Sunken Squat Toilet</td></tr><tr><td></td><td>7 Kitchen Appliance - Fridge Near Entrance</td><td>31 Bedroom Convert - Desk &amp; Bookshelf</td><td>54 Kitchen Island - Freestanding</td></tr><tr><td></td><td>8 Fridge Placement - In Kitchen</td><td>32 Bedroom Bed - Platform w/ Drawers</td><td>55 Study Room - Side-by-Side Double Desk</td></tr><tr><td></td><td>9 Bedroom Storage - Full-Wall Wardrobe</td><td>33 Wide Hall - Vertical Layout</td><td>56 Living Room Reading - Desk w/ Bookshelf</td></tr><tr><td></td><td>10 Fridge Placement - In Living Room</td><td></td><td>57 Dining Storage - Low Sideboard</td></tr><tr><td></td><td>11 Washer Placement - In Bathroom</td><td>34 Bedroom Office - Window Long Desk (2P)</td><td>58 Wide Hall - Sofa Back w/ Bar Divider</td></tr><tr><td></td><td>12 Washer Placement - Balcony w/ Counter</td><td>35 Fridge Placement - In Dining Room</td><td>59 Balcony Zoning - Utility + Leisure</td></tr><tr><td></td><td></td><td>36 Living Room - Wall Bookshelf &amp; Float TV</td><td>60 Dining Zone - Island as Dining Table</td></tr><tr><td></td><td>13 Balcony Utility - Washer &amp; Drying Rack</td><td>37 Dining Storage - Full-Height Sideboard</td><td>61 Corridor Storage - Built-in Cabinet</td></tr><tr><td></td><td>14 Bedroom Office - Desk by Window</td><td>38 Dining Table - Rectangular</td><td>62 Standard Set - Storage Cabinet</td></tr><tr><td></td><td>15 Kitchen Appliance - Washer Under Counter</td><td>39 Study Room - Desk &amp; Bookshelf</td><td>63 Dining Banquette - Wall-Side Bench</td></tr><tr><td></td><td>16 Living Room - Remove TV Stand</td><td>40 Bathroom Toilet - Wall-Hung w/ Concealed Tank</td><td>64 Bathroom Vanity - Wall-Mounted Sink</td></tr><tr><td></td><td>17 Kitchen Cabinet - I-Shaped</td><td>41 Living Room Office - Desk Behind Sofa</td><td>65 Balcony Leisure - Tea Table &amp; Chair</td></tr><tr><td></td><td>18 Entryway - Standard Shoe Cabinet</td><td>42 Wide Hall LDK - Island Divides L/D</td><td>66 Entryway - Full-Height Coat Closet</td></tr><tr><td></td><td>19 Washer Placement - In Kitchen Cabinet</td><td>43 Standard Layout - Centered Desk</td><td>67 Living Room - TV Wall Center Divider</td></tr><tr><td></td><td>20 Bathroom Vanity - External Sink</td><td>44 Bathroom Bathtub - Built-in Wall-Side</td><td>68 Living Room Reading - Island Desk</td></tr><tr><td></td><td>21 Bathroom Shower - Rectangular Wall-Mount</td><td>45 Bathroom Shower - Linear Glass Partition</td><td>69 Study/Guest Room - Sofa Bed &amp; Desk</td></tr><tr><td></td><td>22 Living Room - Sofa as Divider</td><td>46 Living Room - Projector Replaces TV</td><td>70 Living Room Reading - Desk &amp; Chair</td></tr><tr><td></td><td>23 Kitchen Built-in - Tall Cabinet w/ Fridge</td><td>47 Entryway - Slim Flip Shoe Cabinet</td><td>71 Dining Table - Round</td></tr><tr><td></td><td>24 Bathroom Shower - Diamond Corner Unit</td><td>48 Dining Zone - Table Center &amp; Sideboard</td><td></td></tr></table>

Table 7: Complete Design Map containing 52 structural-renovation and 71 furniture-layout design points. Entries are arranged in three parallel numbered blocks for compact presentation.

## Appendix F Failure Case Analysis

We visualize typical failure cases in Figure 6. The primary errors occur when dealing with complex boundaries. As seen in Case 1 and Case 4, when the input outline is highly irregular, the model struggles to partition the space efectively, leading to disconnected rooms or invalid overlaps. Another common failure arises in extremely tight spaces. In Case 2 and Case 3, the limited area forces the model to generate layouts where furniture obstructs doors or hallways become too narrow. These examples indicate that while our method performs well on standard layouts, it still faces challenges when handling extreme geometric irregularities or severe spatial constraints.

## Appendix G AI Assistant Usage Disclosure

Research and Data Annotation. We used multiple visionlanguage models, including Seed, GPT, and Gemini, to audit rule-derived design-point annotations against observable diferences between paired before–after designs. These auditors did not generate the ground-truth labels; the retained labels remain grounded in explicit before–after transformations, with ambiguous cases subjected to manual inspection. Separately, a fixed four-VLM panel (comprising Seed2.1 Pro, GLM-5V-Turbo, GPT-5.5, and Gemini-3.1-Flash-Lite) was employed for VLM-HS evaluation.

Coding. AI coding assistants (e.g., GitHub Copilot) were used to assist in writing the infrastructure code for the evaluation pipeline. All AI-suggested code was thoroughly tested and verified by the research team.

<table><tr><td>Dimension</td><td>Metric</td><td>Range</td><td>Description and Key Indicators</td></tr><tr><td>Design-point fulfillment</td><td>Recall@1</td><td>0-100%</td><td>Mean case-level proportion of requested design points fulfilled in the first gener- ated outputs, averaged across all 200 cases by a rule-based evaluator.</td></tr><tr><td>Constraint compliance</td><td>Pass@1</td><td>0-100%</td><td>Percentage of one-sample generated results that pass all 22 checks across ge- ometric validity, structural compliance, functional integrity, and collision-free placement.</td></tr><tr><td>Constraint compliance</td><td>Geometric Validity  $( \mathcal { P } _ { \mathrm { g e o m } } )$ </td><td>0-100%</td><td>Percentage of generated designs with well-formed polygons, valid references, and complete required fields.</td></tr><tr><td>Constraint compliance</td><td>Structural Compliance  $( \mathcal { P } _ { \mathrm { s t r u c t } } )$ </td><td>0-100%</td><td>Percentage of generated designs that preserve load-bearing walls, exterior en- velopes, and shafts while maintaining room connectivity through valid doors.</td></tr><tr><td>Constraint compliance</td><td>Functional Integrity  $( \mathcal { P } _ { \mathrm { f u n c } } )$ </td><td>0-100%</td><td>Percentage of generated designs that retain fixed infrastructure and essential fixtures without damage or omission.</td></tr><tr><td>Constraint compliance</td><td>Collision-Free Placement  $( \mathcal { P } _ { \mathrm { c o l l } } )$ </td><td>0-100%</td><td>Percentage of generated designs without furniture overlap or forbidden-zone vi- olations.</td></tr><tr><td>Living quality</td><td>VLM-based Habitability Score (VLM-HS)</td><td>0-10</td><td>Four separately reported model-specific scores, each combining five weighted subjective dimensions with a fused physical-compliance score and offline ground-</td></tr><tr><td>Professional approval</td><td>Professional Approval Score (PAS)</td><td>0-10</td><td>truth deficit calibration. Equally weighted mean of eight professional-design dimensions, averaged over all 200 cases; reported for RenoLayout outputs and corresponding ground-truth designs.</td></tr></table>

Table 8: Overview of the unified evaluation framework.
<table><tr><td>Submetric</td><td>Checks</td><td>Violation Flags and Annotations</td></tr><tr><td>Geometric Validity</td><td>5</td><td>polygon_invalid: malformed polygon geometry;</td></tr><tr><td> $( \mathcal { P } _ { \mathrm { g e o m } } )$ </td><td></td><td>polygon_self_intersect: polygon edges intersect one another; invalid_reference: an entity references a nonexistent or invalid object;</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>missing_required_field: a mandatory geometry or semantic field is absent;</td></tr><tr><td></td><td></td><td>wall_endpoint_in_opening: a wall endpoint lies inside a door or window opening</td></tr><tr><td>Structural</td><td>11</td><td>load_bearing_wall_removed: an existing load-bearing wall is deleted;</td></tr><tr><td>Compliance</td><td></td><td>load_bearinq_wal1_moved: a load-bearing wall is displaced;</td></tr><tr><td>(Pstruct)</td><td></td><td>load_bearing_wall_thinned: load-bearing wall thickness is reduced;</td></tr><tr><td></td><td></td><td>load_bearing_wall_new_opening: a new opening is cut into a load-bearing wall;</td></tr><tr><td></td><td></td><td>load_bearing_wall_opening_enlarged: an existing load-bearing-wall opening is enlarged;</td></tr><tr><td></td><td></td><td>exterior_wall_altered: the exterior envelope geometry is modified;</td></tr><tr><td></td><td></td><td>facade_new_window: a new window is introduced on the facade;</td></tr><tr><td></td><td></td><td>wal1_crosses_shaft: a wall intersects a protected service shaft;</td></tr><tr><td></td><td></td><td>opening_exceeds_wal1: an opening extends beyond its supporting wall segment;</td></tr><tr><td></td><td></td><td>no_ent ry_door: the dwelling has no valid entrance door;</td></tr><tr><td></td><td></td><td>room_no_door: a room has no valid doorway</td></tr><tr><td>Functional Integrity</td><td>4</td><td>kitchen_flue_altered: the fixed kitchen exhaust flue is modified;</td></tr><tr><td> $( \mathcal { P } _ { \mathrm { f u n c } } )$ </td><td></td><td>gas_pipe_altered: the fixed gas-pipe location or geometry is modified;</td></tr><tr><td></td><td></td><td>entry_door_altered: the fixed entrance door is modified;</td></tr><tr><td></td><td></td><td>bathroom_missing_fixture: an essential bathroom fixture is absent</td></tr><tr><td>Collision-Free</td><td>2</td><td>item_overlap: furniture or other placed items overlap;</td></tr><tr><td>Placement  $( \mathcal { P } _ { \mathrm { c o l l } } )$ </td><td></td><td>furniture_forbidden_position: furniture occupies a prohibited wall, opening, or clearance zone</td></tr><tr><td>Total</td><td>22</td><td></td></tr></table>

Table 9: Exact rule composition of the four constraint-compliance submetrics, with a brief annotation for each violation flag.

Writing and Polishing. We used ChatGPT/Gemini for grammar correction, stylistic polishing, and rephrasing. This assistance improved manuscript clarity as well as flow. Human authors entirely conceived the core arguments, technical descriptions, and data interpretations. They also wrote the manuscript.

## Appendix H VLM-Based Scoring Protocol

This appendix details the VLM-based scoring protocol used for the Habitability Scores reported in Appendix Appendix D. We use four independently evaluated VLMs: Seed2.1 Pro, GLM-5V-Turbo, GPT-5.5, and Gemini-3.1- Flash-Lite. Each model assesses the rendered floor plan together with the design-point specification and deterministic physical-inspection facts. A deterministic geometric engine provides an objective physical signal. For each VLM separately, its score is calibrated ofline against the independently evaluated ground-truth plan for the same case. We retain and report the four model-specific VLM-HS metrics separately.

![](images/1ec8ff1d84f55c10e8ed7d8f34c0724a4369bee35df815c1ab411bd7cc8bf78a.jpg)  
Figure 6: Representative failure cases involving irregular boundaries and severely constrained spaces.

## Evaluation Dimensions

Table 10 summarizes the six dimensions. Each VLM returns an integer score in [0, 10] for every dimension, where higher is better. The first five dimensions form the abstract designquality score. The sixth dimension provides the VLM’s holistic physical judgment, which is fused with the deterministic engine score to form the final physical score.

<table><tr><td>No.</td><td>Dimension</td><td>Type</td><td>Weight Wi</td></tr><tr><td>1</td><td>Functional Zoning</td><td>Abstract</td><td>1.42</td></tr><tr><td>2</td><td>Spatial Layout</td><td>Abstract</td><td>1.01</td></tr><tr><td>3</td><td>Circulation Quality</td><td>Abstract</td><td>1.26</td></tr><tr><td>4</td><td>Amenity Configuration</td><td>Abstract</td><td>1.00</td></tr><tr><td>5</td><td>Design-Point Implementation</td><td>Abstract</td><td>1.20</td></tr><tr><td>6</td><td>Physical Compliance</td><td>VLM physical review</td><td>(fusion)</td></tr></table>

Table 10: Evaluation dimensions and aggregation weights.

## Physical Inspection Layer

A local geometric engine inspects the structured floor-plan output and assigns each detected issue one of three severity levels, as summarized in Table 11. P0 and P1 are used in the engine-calibrated physical score; P2 is retained for diagnosis but does not directly reduce that score.

<table><tr><td>Level</td><td>Meaning</td><td>Representative issues</td></tr><tr><td>PO</td><td>Critical; unsafe, unusable, or non-compliant</td><td>Invalid structural modification, room overlap, missing essential functions, inaccessible space, or severe wall/opening</td></tr><tr><td>P1</td><td>Severe; substantially degrades normal use</td><td>Insufficient clearance, narrow circulation, door blockage, furniture overlap, inadequate room area, or functional adjacency conflicts</td></tr><tr><td>P2</td><td>Advisory; minor optimization issue</td><td>Local dimensional, lighting, storage, or configuration suggestions</td></tr></table>

Table 11: Severity levels of physical defects.

## Score Aggregation

Model-specific dimensional scores. Let $s _ { i } ^ { ( m ) }$ denote the score assigned to dimension i by VLM m, where $m =$ $1 , \ldots , M$ and $M = 4 .$ . All subsequent quantities are com puted independently for each model m.

Abstract design-quality score. For model m, the first five dimensions are aggregated using the weights in Table 10:

$$
S _ { a } ^ { ( m ) } = \frac { \sum _ { i = 1 } ^ { 5 } w _ { i } s _ { i } ^ { ( m ) } } { \sum _ { i = 1 } ^ { 5 } w _ { i } } .\tag{14}
$$

Engine-calibrated physical score. Let $n _ { P 0 } , n _ { P 1 }$ , and $n _ { P 2 }$ be the numbers of engine-detected issues at the three severity levels. The deterministic physical score, shared across the four VLMs for a given plan, is

$$
S _ { e } = \mathrm { c l i p } \left( 1 0 - 1 . 2 5 n _ { P 0 } - 0 . 0 6 n _ { P 1 } - 0 . 0 0 n _ { P 2 } , 0 , 1 0 \right) .\tag{15}
$$

This formulation makes P0 violations the dominant physical ordering signal, uses P1 violations for finer within-band diferentiation, and retains P2 only as diagnostic evidence.

Model-specific fused physical score. For model $m ,$ , its sixthdimensional score $S _ { v } ^ { ( \bar { m } ) } = s _ { 6 } ^ { ( m ) }$ provides a holistic visual judgment of physical usability. We combine it with the engine score as

$$
\tilde { S } _ { v } ^ { ( m ) } = \operatorname* { m i n } ( S _ { v } ^ { ( m ) } , S _ { e } ) ,\tag{16}
$$

$$
S _ { p } ^ { ( m ) } = 0 . 7 0 S _ { e } + 0 . 3 0 \tilde { S } _ { v } ^ { ( m ) } .\tag{17}
$$

The minimum operator ensures that a VLM judgment cannot ofset deterministic P0/P1 violations. When a VLM identifies additional visual usability concerns and assigns a lower score than the engine, it contributes an additional penalty through the fused score.

Model-specific absolute habitability score. Before reference calibration, the score for model m is

$$
\begin{array} { r } { S _ { \mathrm { r a w } } ^ { ( m ) } = 0 . 4 5 S _ { a } ^ { ( m ) } + 0 . 5 5 S _ { p } ^ { ( m ) } . } \end{array}\tag{18}
$$

Ofline ground-truth deficit calibration. For each test case, the predicted plan and its ground-truth plan are evaluated independently by the same VLM m. Let $S _ { a } ^ { \mathrm { g t } , ( m ) }$ denote the ground-truth abstract score, whereas $S _ { p } ^ { \mathrm { g t } , ( m ) }$ denotes the fused physical score. We compute a one-sided reference deficit:

$$
\begin{array} { r l } & { D _ { \mathrm { g t } } ^ { ( m ) } = 0 . 4 5 \operatorname* { m a x } ( 0 , S _ { a } ^ { \mathrm { g t } , ( m ) } - S _ { a } ^ { ( m ) } ) } \\ & { \qquad + 0 . 5 5 \operatorname* { m a x } ( 0 , S _ { p } ^ { \mathrm { g t } , ( m ) } - S _ { p } ^ { ( m ) } ) . } \end{array}\tag{19}
$$

Only deficits relative to the ground truth are penalized; a prediction does not receive an additional bonus for exceeding the ground truth on an individual component.

Final model-specific VLM-HS. The final score produced by model m is

$$
\begin{array} { r } { S _ { u } ^ { ( m ) } = \mathrm { c l i p } \left( S _ { \mathrm { r a w } } ^ { ( m ) } - \gamma D _ { \mathrm { g t } } ^ { ( m ) } , 0 , 1 0 \right) , } \end{array}\tag{20}
$$

where $\gamma = 1 . 0$ by default. For a ground-truth plan evaluated against itself, $D _ { \mathrm { g t } } ^ { ( m ) } = 0$ . For an evaluation set of N cases, we report the four model-specific metrics

$$
\mathrm { V L M - H S } ^ { ( m ) } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } S _ { u , j } ^ { ( m ) } , \qquad m = 1 , \ldots , 4 .\tag{21}
$$

## Scoring Prompt

Tables 12–16 reproduce the scoring prompt in component order. We submit the same prompt independently to Seed2.1 Pro, GLM-5V-Turbo, GPT-5.5, and Gemini-3.1-Flash-Lite. The resulting scores are reported as model-specific metrics. Each VLM receives the post-design rendering, the pre-design rendering when available, the requested design points, and deterministic P0/P1 physical-inspection facts. Ground-truth plans are not included in the VLM inputs; ground-truth deficit calibration is computed after independent scoring. Dimension names follow Table 10.

<table><tr><td>Prompt component</td><td>Instruction content</td></tr><tr><td>Role and task</td><td>You are a senior residential design reviewer. Evaluate the post-design plan dimension by dimension. Assess subjective design quality that geometry cannot fully determine in Dimensions 1–5. Assess physical compliance together with hard defects only in Dimension 6.</td></tr><tr><td>Design context and design points</td><td>When both images are available, they depict the same dwelling in two states. The first image shows the original plan, whereas the second presents the post-design plan under review. Score only the post-design plan; the original plan is provided solely to explain what changed and why. All scores reflect the absolute quality of the post-design solution, not improvement relative to the original plan. The listed design points are the required design objectives. Dimensions 1–4 evaluate absolute zoning, layout, circulation, and amenity quality. They must not mechanically reward or penalize compliance with a design point. However, consequences necessarily caused by correctly implementing a required design point must not be treated as defects. Dimension 5 alone evaluates whether design points were implemented. It also assesses</td></tr><tr><td>Objective physical-inspection facts</td><td>[Objective physical-inspection facts detected by the deterministic engine; the primary basis for Dimension 6.] {Insert the engine&#x27;s P0/P1 fact list here; if there are no hits, write: No P0/P1-level physical issues detected.} Cross-check the list against the post-design image. Clearly inconsistent entries may be treated as false detections and downweighted. Do not</td></tr></table>

Table 12: VLM-HS scoring prompt, Part I: role, design context, and physical-inspection input.

<table><tr><td>Prompt component</td><td>Instruction content</td></tr><tr><td>Dimension 1: Functional Zoning</td><td>Evaluate whether active/quiet, public/private, and wet/dry zones are clearly layered with limited mutual interference. Check whether bedrooms form an independent cluster separated from living/dining/kitchen spaces. Assess whether the plan progresses from entry to public/social space and then reaches private sleeping space. Determine whether bathrooms serve either the sleeping area or the public area rather than remaining isolated centrally. Finally, assess whether living, dining, and kitchen areas form a coherent social zone. This dimension evaluates zoning logic,</td></tr><tr><td>Dimension 2: Spatial Layout</td><td>not missing furniture or reachability. Evaluate whether proportions are comfortable and balanced. Determine whether the circulation area is economical and the primary/secondary hierarchy is appropriate. Check whether major rooms are roughly within a 1:1–1:2 aspect-ratio range. Assess whether corridors are compact rather than long, dark, or wasteful. Determine whether the master bedroom is larger than secondary bedrooms. The living room should be the</td></tr><tr><td></td><td>largest or second-largest public space. Finally, check whether obvious fragmentary space exists. This dimension evaluates proportion and layout rather than exact floor area. Assuming rooms are reachable with no must-pass-through-bedroom defect, evaluate the smoothness and quality of daily circulation. Check whether the entry provides a foyer or dust-buffer zone. Assess whether kitchen and dining areas are adjacent with a short housework path.</td></tr></table>

Table 13: VLM-HS scoring prompt, Part II: definitions of Dimensions 1–3.

<table><tr><td>Prompt component</td><td>Instruction content</td></tr><tr><td>Dimension 4: Amenity Configuration</td><td>Evaluate whether the plan provides hallmark configurations of quality living. Check for a master suite with a dedicated bathroom or dressing/storage area. Also assess whether the plan includes a distinct entry foyer with systematic storage. Where multiple bathrooms exist, evaluate owner/guest separation. Finally, assess whether an open, connected living/dining area supports social use. Evaluate only items visible in the image; skip unclear evidence rather than fabricating it.</td></tr><tr><td>Dimension 5: Design-Point Implementation</td><td>Evaluate the listed design points in terms of both fulfillment and implementation quality. Fulfillment asks whether each required design point is visibly realized without omission, distortion, or reversal. Implementation quality considers location, scale, and spatial relationships. These aspects should appear natural and coherent with the overall plan rather than forced or achieved at the expense of visible living quality. This dimension must be supported by a mandatory item-by-item design-point audit.</td></tr><tr><td>Dimension 6: Physical Compliance</td><td>Evaluate objective physical compliance and defects. Use the deterministic P0/P1 fact list as the primary basis, cross-checked against the post-design image. P0 defects are the primary ordering signal and must be strongly penalized; P1 defects provide only finer differentiation within a severity band. Do not invent issues beyond the fact list.</td></tr><tr><td>Scope boundary</td><td>The following objective items are evaluated only in Dimension 6: room reachability; furniture overlap, through-wall placement, or door blockage; missing essential furniture; natural daylight; must-pass-through-bedroom circulation; direct kitchen-to-toilet-bathroom connection; door or passage width; and load-bearing/exterior-wall modifications. Dimensions 1-5 evaluate only subjective design quality within their stated scope. Design-point implementation is evaluated primarily in Dimension 5, while correctly required design-point consequences must</td></tr><tr><td>Deduction/bonus checklist</td><td>Functional Zoning Deductions: public functions interrupted by bedrooms or bathrooms; rooms arranged as a single-axis sequence without clustering; entry directly entering or facing the bedroom zone; a bedroom sandwiched between public spaces; or a centrally isolated bathroom. Bonuses: bedrooms grouped independently behind a corridor or foyer, and a clear entry → public → private progression. Circulation Quality Deductions: no foyer buffer; kitchen and dining separated by a detour; nighttime bedroom-to-bathroom circulation crossing public activity space; crossing or strongly interfering daily paths; or an obviously circuitous route from entry to major functions. Bonuses: an entry buffer, adjacent kitchen/dining spaces, and a short private nighttime path. Amenity Configuration Deductions: an ordinary master bedroom with neither dedicated bathroom nor dressing/storage area; no entry zone or shoe-storage location; scattered storage; multiple bathrooms without owner/guest separation; or a living/dining area fragmented by circulation. Bonuses: a master suite, foyer plus systematic storage, owner/guest-separated bathrooms, and an open connected living/dining area. Design-Point Implementation Audit every listed design point individually. A point is Implemented only when its required action produces</td></tr><tr><td>Scoring-band anchors</td><td>appear natural and coherent. Spatial Layout—9–10: balanced major-room proportions, no elongated forms, no or only a very short dedicated corridor, appropriate room hierarchy, with no fragmentary space; 7–8: most rooms proportioned appropriately, moderate corridor share, with at most minor local narrowness; 6–7: one or two clear scale problems; 5–6: multiple scale problems, large fragmentary space, excessive corridor share, or reversed master/secondary hierarchy; 0–4: severely unbalanced layout Physical Compliance—9-10:  $P 0 = 0 , P 1 \leq 2 ; 8 \cdot 9 ; P 0 = 0 , P 1 = 3 \ – 6 ; 7 \cdot 8 \colon P 0 = 0 , P 1 \geq 7 ; 5 . 5 \cdot 7 ; P 0 = 1 ; 3 \cdot 5 . 5 \colon P 0 = 2 ;$ </td></tr><tr><td>Hidden design-defect checklist</td><td>0–3: P0 ≥ 3. P1 issues may gradually lower the score within a band, but must not outweigh P0 defects. Living/dining space traversed by circulation and turned into a passage; a severely imbalanced frontage-to-depth ratio creating a strip-like plan; excessive room subdivision causing cramped rooms or fragmentary blocks; a main public space smaller than a secondary room; a bathroom</td></tr></table>

Table 14: VLM-HS scoring prompt, Part III: definitions of Dimensions 4–6 and scope boundaries.

Table 15: VLM-HS scoring prompt, Part IV: checklists and scoring anchors.

<table><tr><td rowspan=1 colspan=9>Prompt component             Instruction content</td></tr><tr><td rowspan=4 colspan=9>Scoring procedure               A. Dimensions 1-5: First inspect the relevant deduction/bonus checklist together with the hidden-defect checklist. Record only visible hitswith their locations. Then assign a score based on the number and severity of hits rather than a vague overall impression: four or moredeductions, or one very severe deduction, normally yields 5–6; three deductions yields about 6; two deductions yields 7–8; at most one minordeduction yields about 8; no deduction yields 8–9. Multiple visible bonuses with no deduction may yield 10. Scores of 6 or below must namethe specific deduction and location; scores of 8 or above must be supported by no obvious deduction.For Dimension 5, audit every design-point ID separately. Each item must be labeled Implemented, Partially implemented, Not implemented, orReversely implemented, with image-grounded evidence. Do not merge multiple IDs using statements such as &quot;most&quot; or “all.&quot; A design point isimplemented only when its key action and intended result are both visible; unclear evidence is labeled partially implemented. The dimensionscore reflects both fulfillment and implementation quality, but is not mechanically fixed by the proportion of item statuses.B. Dimension 6: Use the P0/P1 fact list as the primary basis. Cross-check obvious inconsistencies against the image, do not fabricateadditional issues, and assign the score using the P0/P1 anchors. A true P0 must substantially lower the score: P0 = 0 normally implies atleast 7, P0 = 1 cannot exceed 7, and P0 ≥ 2 cannot exceed 5.5.</td></tr><tr><td rowspan=1 colspan=3>ne very sever</td><td rowspan=1 colspan=1>ere dec</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8: no deduction</td></tr><tr><td rowspan=1 colspan=3>6: Use the PO/I</td><td rowspan=1 colspan=1>P0/P1 fact</td></tr><tr><td rowspan=5 colspan=9>Output format                  Output only one strict JSON object, with no draft reasoning or other text. The design_point_checks array is mandatory: every supplieddesign-point ID must occur exactly once; output an empty array if no design points are supplied. Reasons must be concise, image-grounded,and consistent with the item-level audit. Each reason states at most two deductions or bonuses with locations; vague boilerplate is prohibited.If a dimension cannot be judged from the rendering, use score —1 and explain why; score 0 is reserved for an extremely poor but judgeableplan. Do not output an overall score. The JSON has the form:{&quot;design_point_checks&quot;: [{&quot;id&quot;: &quot;L2-1&quot;, &quot;status&quot;: &quot;Implemented&quot;, &quot;evidence&quot;: &quot;...&quot;}],&quot;dimensions&quot;: [{&quot;dimension&quot;: &quot;Functional Zoning&quot;, &quot;score&quot;: &lt;0-10&gt;, &quot;reason&quot;: &quot;...&quot;},{&quot;dimension&quot;: &quot;Spatial Layout&quot;, ...}, {&quot;dimension&quot;: &quot;Circulation Quality&quot;, ...}, {&quot;dimension&quot;:&quot;Amenity Configuration&quot;, ...}, {&quot;dimension&quot;: &quot;Design-Point Implementation&quot;, ...},{&quot;dimension&quot;: &quot;Physical Compliance&quot;, ...}]}</td></tr><tr><td rowspan=1 colspan=2>ID mu</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr></table>

Table 16: VLM-HS scoring prompt, Part V: scoring procedure and output format.