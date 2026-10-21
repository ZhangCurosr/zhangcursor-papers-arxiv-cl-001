# CovNav: Conformal Prediction-Guided Adaptive Inference for Object Goal Navigation

Pengying Wu, Zhejia Wen, Kang Ding, Kangjie Zhou, Chunxuan Jiao, Wei Pan, Member, IEEE, Xiangli Nie, Member, IEEE, and Chang Liu, Member, IEEE

Abstract—In object goal navigation (ObjectNav), an embodied agent must find a target object (e.g., “find a bed”) in a previously unseen environment, relying on spatial commonsense such as object co-occurrence. Large Language Models (LLMs) encode this commonsense, but deploying them on robots exposes a tradeoff between reasoning quality and efficiency: cloud-based models incur latency and privacy risks, whereas small on-device LLMs are fast but prone to hallucinated spatial judgments. Hybrid reasoning LLMs offer both a fast non-thinking mode and a slow thinking mode, which raises a key question for embodied agents: when is slow thinking worth its cost? We propose CovNav, a training-free framework that answers this question with Conformal Prediction (CP), turning dual-process (System 1 / System 2) inference into an uncertainty-aware decision. Much like recent decision models such as Jev, CovNav casts each spatial query as a structured binary decision rather than free-form generation: a local LLM in fast mode judges whether the target is likely near an observed object, CP converts its token probabilities into a prediction set, and only ambiguous queries are escalated to slow chain-of-thought reasoning. The object-pair formulation keeps calibration tractable and supports open-set objects through off-the-shelf detectors. Experiments on Gibson and HM3D show that CovNav surpasses the strongest zero-shot baseline by 0.8 and 1.7 percentage points in success rate on HM3D v1 and v2, while invoking slow reasoning for only 12.4% of queries and retaining most of the gain of always-on thinking at a fraction of its latency. Real-robot experiments on a TurtleBot 4 further validate the approach. CovNav shows how calibrated uncertainty can decide when an embodied agent should think slowly, offering a principled form of adaptive test-time reasoning for LLM-based robots.

Index Terms—Embodied AI, Object Goal Navigation, Zero-Shot Object Navigation, Large Language Models, Conformal Prediction, Uncertainty Quantification, Adaptive Inference, Adaptive Reasoning, Fast and Slow Thinking.

## I. INTRODUCTION

ECENT advances in large foundation models have standing, visual reasoning, and knowledge-intensive tasks. A pivotal next step for artificial intelligence is to ground these capabilities in the physical world, building embodied agents that can perceive, reason, and act within complex, unstructured environments [1]. Among the core competencies required for such agents, the ability to autonomously navigate to semantically specified targets is both fundamental and practically significant, underpinning applications from domestic assistance to search and rescue.

![](images/729e6c54f927f2bc4dff21835bd8e604d83f255f0853ccef92d9d31fccab46fc.jpg)  
Fig. 1. We propose a Conformal Prediction (CP)-guided fast-slow inference paradigm for object goal navigation using Qwen3-8B as the local LLM. CP monitors the local LLM’s output confidence to adaptively switch Qwen between its fast direct inference mode and slow deliberative reasoning mode. Clear Case: When observing a bathtub, Qwen confidently predicts a toilet will be nearby (logits: 0.9 vs 0.1), so CP accepts the fast direct inference. Ambiguous Case: When observing a sofa and querying about a bed, Qwen shows low confidence (logits: 0.6 vs 0.4), so CP triggers its slow deliberative reasoning mode for more reliable spatial analysis.

Object goal navigation (ObjectNav) serves as a representative testbed for this capability. Given a semantic target (e.g., “find a bed”), an agent must explore and navigate through a previously unseen environment without any prior map [2], [3]. A key enabler for efficient exploration is commonsense spatial knowledge, that is, the understanding that functionally related objects tend to co-locate within the same room (e.g., beds and nightstands in bedrooms, sofas and televisions in living rooms). Large Language Models (LLMs), which encode vast commonsense knowledge from pre-training, have recently emerged as a powerful tool for providing such spatial priors without task-specific training [4], [5], [6], motivating a growing body of work on LLM-based object goal navigation.

Existing LLM-based navigation approaches face significant deployment challenges. First, many state-of-the-art methods rely on cloud-based APIs (e.g., ChatGPT, Claude) [5], [7], which introduces network latency, raises privacy concerns, and limits deployment in offline or network-constrained scenarios. Second, directly deploying smaller open-source LLMs on resource-constrained robot platforms often leads to performance degradation, manifesting as hallucination and unreliable spatial judgments when processing complex relationships. Third, existing approaches lack adaptive mechanisms to dynamically adjust inference depth based on problem difficulty, resulting in either consistently invoking large models (incurring high computational costs) or relying exclusively on small models (sacrificing accuracy). This raises a critical question: How can we achieve both efficient and accurate spatial reasoning on resource-constrained robotic platforms?

To address this challenge, we propose an adaptive inference framework that dynamically adjusts inference depth based on problem difficulty, utilizing LLM prediction uncertainty as a core metric. A key insight driving our work is that not all spatial reasoning problems require deep deliberation, as illustrated in Figure 1. For instance, judging the spatial proximity between a television and a sofa is straightforward, given their typical co-occurrence in living rooms. However, determining the relationship between a sofa and a bed requires cross-room reasoning to recognize they belong to distinct functional spaces. Crucially, the output confidence produced by small LLMs, specifically the logits generated during deployment, serves as an effective indicator of problem difficulty, reflecting how well the input aligns with the training corpus. We leverage Conformal Prediction (CP) [8], a statistically rigorous uncertainty quantification framework, to assess this confidence without requiring additional model training. CP constructs prediction sets based on model outputs: when the model is highly confident, the prediction set contains a single answer, indicating that the fast direct inference mode is sufficient. Conversely, when uncertain, the prediction set contains multiple possible answers, indicating the model’s inability to identify a unique solution, which triggers a slow deliberative reasoning mode. This mechanism mirrors human cognitive processing: System 1 thinking (fast, intuitive) handles routine decisions, while System 2 thinking (slow, analytical) engages for complex problems [9].

This perspective also connects CovNav to two recent trends in LLM reasoning. First, hybrid reasoning models such as Qwen3 [10] expose both a fast non-thinking mode and a slow thinking mode, and recent work trains such models with reinforcement learning to decide when to think [11], [12]; CovNav makes this decision without any training, using a statistically calibrated criterion. Second, decision models such as Jev [13], [14] answer queries with structured choices, binary judgments, or scores instead of free-form text, trading openended generation for speed and reliability. CovNav’s fast mode follows the same principle by casting each spatial query as a binary object-pair decision with a confidence score, and adds a CP gate that escalates to System 2 reasoning only when this System 1 decision is uncertain.

Building on this insight, we design a CP-guided navigation framework (CovNav) where the robot constructs an objectwise semantic map using off-the-shelf detection models. For each newly observed object, it queries the LLM to assess spatial relationships with the target object. CP analyzes the response confidence to determine whether the fast direct inference mode can reliably answer the question. If confidence is sufficient, the system directly utilizes the response; otherwise, it activates the slow deliberative reasoning mode for more thorough analysis. The resulting semantic value map, weighted by these assessments, guides the robot toward high-potential exploration regions. Two key design choices facilitate this framework: (1) formulating queries as object-pair relationships rather than open-ended visual question answering makes the problem space more tractable for CP calibration, and (2) leveraging open-source detection models enables open-set semantic understanding without task-specific training.

The main contributions of this work are as follows:

• We propose a novel fast-slow inference paradigm for robotic navigation that adaptively switches between fast direct inference and slow deliberative reasoning based on LLM response uncertainty, effectively achieving a balance between computational efficiency and decision accuracy on resource-constrained platforms.

• We introduce the first use of Conformal Prediction to gate between the fast and slow reasoning modes of an LLM for object navigation, providing a theoretically grounded and training-free switching mechanism. We further design an object-pair-based query formulation that makes CP calibration tractable while enabling open-set semantic understanding through off-the-shelf vision models.

• We validate our approach through extensive simulated and real-world robot experiments, demonstrating significant improvements in both navigation efficiency and success rate compared to existing methods.

The remainder of this paper is organized as follows. Section II reviews related work in zero-shot object navigation, LLM-based navigation and uncertainty quantification in Embodied AI. Section III provides the problem formulation and details our proposed methodology, specifically the objectbased semantic mapping and the CP-guided fast-slow switching mechanism. Section IV presents extensive experimental results, including simulation comparisons and ablation studies. Section V details real-world validations, demonstrating the effectiveness and efficiency of our approach in physical environments. Section VI concludes the paper and discusses future directions.

## II. RELATED WORK

## A. Zero-Shot Object Goal Navigation

Object goal navigation (ObjectNav) requires an agent to navigate to instances of specific object categories in unknown environments, combining visual perception, semantic understanding, and spatial reasoning [15], [16], [17]. It is closely related to other embodied navigation tasks, such as image goal navigation [18] and multi-agent cooperative navigation [19]. End-to-end approaches learn navigation policies directly from observations via reinforcement learning [20], but tend to generalize poorly to unseen environments compared with modular pipelines [2]. The Object Memory Transformer (OMT) [3] introduced long-term semantic memory to improve navigation efficiency, while Goal-Oriented Semantic Exploration [2] pioneered episodic semantic maps with efficient exploration strategies. Recent modular approaches decompose navigation into separate components. The SEEK framework [17] combines spatial configurations with commonsense knowledge about object relationships. Liu et al. [21] explore implicit scene memory for target-directed navigation by storing historical observations in a NeRF-based memory and retrieving targetrelevant cues through dual-space attention with uncertainty masking. Heterogeneous Scene Representation (HSR) [22] models human-robot-object interactions using graph attention networks, while Commonsense-Aware Object Value Graph (OVG-Nav) [23] incorporates topological structures leveraging visual observations and commonsense priors. Multi-View Visual Transformer (MVVT) [24] expands input perspectives to capture richer spatial context relationships for navigation.

Zero-shot object navigation (ZSON) enables agents to locate objects without explicit training on specific categories or environments [25], [26]. ZSON [27] established the foundational approach by training on image-goal navigation and projecting language queries into multimodal semantic embedding spaces. The Target-Directed Attention Network (TDANet) [26] advanced zero-shot capabilities through attention mechanisms learning spatial and semantic relationships using Siamese architecture designs. Training-free approaches have emerged with Exploration with Soft Commonsense constraints (ESC) [4], which transfers commonsense knowledge from pre-trained models using prompt-based grounding and spatial reasoning. TriHelper [28] addresses specific challenges including collision avoidance and exploration efficiency through specialized helper modules. Foundation model integration has enabled sophisticated zero-shot approaches. SemNav [29] integrates Vision Foundation Models with model-based planners for long-horizon decision making through frontier exploration, achieving promising performance in ZSON.

## B. LLM-Based Visual Navigation

Large Language Models have revolutionized zero-shot navigation by providing reasoning capabilities and semantic understanding for interpreting natural language instructions [5], [25]. OpenFMNav [25] pioneered LLM integration with vision-language models for open-set navigation, using LLMs to extract object proposals and VLMs to build Versatile Semantic Score Maps (VSSM) for exploration. Beyond navigation, multi-agent LLM frameworks such as MALMM [30] distribute high-level planning and low-level control across specialized LLM agents with a supervisor for zero-shot robotic manipulation. Cognitive Planning frameworks [31] integrate LLMs with Large Vision-Language Models for efficient exploration strategies using 3D modular scene representations. LIM2N [32] enables multimodal input processing where language and hand-drawn inputs serve as navigation constraints. Advanced reasoning approaches include Treeof-Thought methods in VLTNet [6] for navigation frontier selection with multi-path reasoning and backtracking capabilities. Chain-of-Thought integration in CL-CoTNav [7] provides closed-loop hierarchical reasoning with structured feedback in decision-making. World model integration in WMNav [33] predicts outcomes and builds memories using Curiosity Value Maps for dynamic policy configuration. NavGPT [34] demonstrated early purely LLM-based navigation through zero-shot sequential action prediction with explicit high-level planning capabilities. Beyond navigation, hybrid reasoning models such as Qwen3 [10] can switch between thinking and non-thinking modes, and recent methods learn when to invoke deliberate reasoning through reinforcement learning [11], [12].

## C. Uncertainty Quantification in Embodied AI

Uncertainty quantification is essential for safe and reliable robot decision-making. Classical approaches such as MC-Dropout [35] and Deep Ensembles [36] estimate epistemic uncertainty by performing multiple stochastic forward passes or training several independent networks, respectively. While effective in controlled settings, these methods incur substantial computational overhead, making them impractical for realtime robotic deployments on resource-constrained platforms.

Conformal Prediction (CP) [8] has recently gained traction as a lightweight, distribution-free alternative that provides rigorous statistical coverage guarantees without retraining. In robotics, CP has been applied to safe motion planning among dynamic agents [37], [38], where prediction sets over future trajectories enable conservative yet efficient collision avoidance. Ren et al. [39] introduced KnowNo, which applies CP to LLM planners so that a robot asks for help when the prediction set contains multiple options, and later used CP for embodied question answering [40], where prediction set sizes decide when additional exploration is needed.

In the context of LLMs, quantifying output uncertainty remains an open challenge. Kadavath et al. [41] showed that large language models are reasonably well calibrated on multiple-choice questions and can estimate the probability that their own answers are correct, though such self-evaluation lacks formal statistical guarantees. Kumar et al. [42] applied CP to LLM outputs in multiple-choice question answering, showing that conformal prediction sets achieve the target coverage and that their uncertainty estimates closely track prediction accuracy. However, existing work has not explored CP-based uncertainty quantification for LLM-guided robotic navigation. Our work addresses this gap by using CP to dynamically gate between fast direct inference and slow deliberative reasoning, thus providing the first principled mechanism for adaptive LLM inference in embodied navigation tasks.

## III. METHOD

## A. Problem Formulation

We formulate the ZSON task as a Partially Observable Markov Decision Process (POMDP), defined by the tuple $\mathcal { P } = \langle \mathcal { S } , \mathcal { A } , \mathcal { T } , \mathcal { O } , \mathcal { R } , \gamma \rangle$ . In this setting, an agent is initialized at a random starting pose $p _ { 0 }$ within a previously unseen environment $\mathcal { E } .$ The agent is assigned a semantic target goal G from an open-set category vocabulary $\mathcal { V } \ \left( \mathrm { e . g . , \ ^ { \mathrm { 6 6 } } d ^ { \mathrm { 3 } } } \right.$ or “Sink”). Unlike traditional ObjectNav, which limits G to a fixed set of training categories, ZSON requires the agent to generalize to novel objects without prior specific training.

At each discrete time step t, the agent receives a sensory observation $o _ { t }$ comprising an egocentric RGB-D image pair $\langle I _ { t } ^ { r g b } , I _ { t } ^ { d e p t h } \rangle$ and its current sensor pose $p _ { t } \in S E ( 2 )$ . Based on the current observation and its historical context, the agent’s policy $\pi ( a _ { t } | o _ { t } , o _ { t - 1 } , . . . , o _ { 0 } )$ selects an action $a _ { t }$ from a discrete action space ${ \mathcal { A } } .$ . This action space consists of six primitives: Move forward (0.25m), Turn left (30<sup>◦</sup>), Turn right (30<sup>◦</sup>), Look up, Look down, and Stop.

The objective is to navigate to a location close to any instance of the target category G (denoted as $g ^ { * } )$ and execute the Stop action. An episode is considered successful if and only if two conditions are met:

1) The agent issues the Stop command within a maximum episode length $T _ { m a x } \ ( \mathrm { { e . g . } }$ ., 500 steps).

2) The geodesic distance $d _ { g e o } ( p _ { T } , g ^ { * } )$ between the agent’s stopping position $p _ { T }$ and the nearest instance of the target object is less than a success threshold $\delta _ { s u c c }$ (typically set to 1.0m).

## B. Overview of CovNav

Figure 2 illustrates the CovNav framework, which synergizes perception with adaptive inference for ZSON. A Semantic Mapping module incrementally builds an openvocabulary object map from RGB-D observations. To guide exploration, LLMs estimate probabilistic spatial relationships between landmarks and the target. A core innovation is the Fast-Slow mechanism: using Conformal Prediction (CP) to quantify uncertainty, the system first uses the local LLM in a fast direct inference mode (System 1), and activates the same model in a slow deliberative reasoning mode (System 2) only when uncertainty is high. Finally, these estimates fuse into a global value map to direct the agent toward the most promising frontiers.

## C. Perception: Object-wise Semantic Mapping

The robot maintains an open-set semantic map in real-time by processing RGB-D images and positions. The semantic map has a $4 \times M \times M$ grid structure, where $M \times M$ represents the map resolution. The four channels consist of an obstacle channel, an exploration area channel, a target goal channel, and an item category channel. Specifically, the scalar values in the category channel correspond to the index of the detected object at that location. To address the limitation of object detectors needing predefined categories, we first employ the BLIP model [43] for real-time scene understanding on singleframe RGB images. Similar to [44], we generate caption text for the real-time RGB image and then use NLTK [45] to extract key nouns from the text as prompts for the visual detection model. Note that the inference latency of both BLIP and NLTK is negligible, ensuring the pipeline operates in real-time. We utilize the off-the-shelf segmentation model, YOLO-v7, combined with Mobile-SAM, to perform semantic segmentation on RGB images and identify common object masks. For objects not in the COCO [46] list, we follow the VLFM [47] setting and use GroundingDINO to capture openset objects outside the COCO list. Each object mask is then mapped to a 3D semantic point cloud based on the agent’s orientation and depth image. A 2D semantic map is created by projecting the 3D point cloud from top to bottom onto a 2D plane, mapping all point clouds containing semantic information to obstacle channels and their respective object type channels. Point clouds near the floor are mapped to the explored channel.

## D. Exploration with LLM Guidance

1) Object Relationship: To explore unseen environments efficiently, the agent must infer the potential location of a notyet-observed target goal based on observable landmarks. This inference relies on Object Relationships, i.e., the probabilistic spatial co-occurrences intrinsic to indoor layouts. For instance, a “Nightstand” strongly implies a nearby “Bed”. Since these commonsense spatial relationships are fundamentally static and do not change across scenes, querying an LLM online for every object pair is computationally inefficient and unnecessary. To address this, we propose a hybrid inference mechanism that combines offline pre-computation for efficiency with online adaptive inference for open-set generalization.

a) Static Knowledge Memory.: For common object categories (e.g., COCO [46] classes) that frequently appear in indoor environments, we pre-compute an $N \times N$ Object Relationship Matrix M offline using a powerful Large Foundation Model (i.e., GPT-5). Specifically, for each pair of object categories $( c _ { i } , c _ { j } )$ , we query $\mathrm { G P T - 5 }$ to estimate the conditional probability P(Target $= c _ { j } | .$ Landmark $= c _ { i } )$ based on its vast commonsense knowledge base. These probabilities are stored in $\mathcal { M } ,$ , allowing for $O ( 1 )$ constant-time lookups during navigation. This “Memory” acts as a fast-access buffer for static commonsense, effectively reducing the latency for known object pairs.

b) Online Adaptive Inference.: For open-set objects encountered in the wild that are not covered by the precomputed memory $( \mathrm { i . e . , } ( o _ { o b s } , o _ { g o a l } ) \notin \mathcal { M } )$ , we employ a local lightweight LLM (Qwen3-8B) to perform online inference. The query is structured as a discriminative task:

“Consider the common room layout. Is it highly likely to   
find a {goal name} close to the {landmark} in the same   
room? Answer with $\because \boldsymbol { Y } _ { e s } \boldsymbol { \cdot }$ or $^ { \mathrm { 4 } } N o ^ { \mathrm { 3 } } .$

To obtain a continuous metric of semantic relationship, we perform a single forward pass and extract the logits for $\mathbf { \mu ^ { 6 6 } } \mathbf { Y e s } ^ { \mathbf { \ 7 } }$ $\scriptstyle ( z _ { \mathrm { Y e s } } )$ and $\^ { 6 4 } \mathrm { N o } ^ { 3 9 } \ \left( z _ { \mathrm { N o } } \right)$ . The normalized confidence score $c \in$ $[ 0 , 1 ]$ is computed via temperature (T) scaling:

$$
c = \frac { \exp ( z _ { \mathrm { Y e s } } / T ) } { \exp ( z _ { \mathrm { Y e s } } / T ) + \exp ( z _ { \mathrm { N o } } / T ) }\tag{1}
$$

This continuous score represents the estimated likelihood of the target’s presence. Instead of a binary decision, it provides a dense, probabilistic metric reflecting the strength of the semantic relationship between the landmark and the target goal. However, because a single fast forward pass from a small local model can be unreliable on ambiguous object pairs, we further employ CP to quantify uncertainty and trigger a slow deliberative reasoning mode when necessary.

2) Conformal Prediction for Uncertainty Quantification: In the context of object goal navigation, relying solely on point predictions from an LLM (e.g., the token with the highest probability) carries the risk of confident hallucinations, especially when the spatial relationship is ambiguous (e.g., distinguishing whether a “Chair” implies a nearby “Table” or “Toilet”). To mitigate this, we employ CP [8], a distributionfree uncertainty quantification framework that transforms the model’s raw confidence scores into a prediction set with a rigorous statistical guarantee; we refer the reader to [8], [40] for a thorough exposition.

![](images/1776cf51c64d1c24c8a98bd8a3aa31a5ed31f4c80ced0e065aeb9b36a0c87053.jpg)  
Fig. 2. Overview of the CovNav framework. The agent incrementally builds an object-based semantic map from RGB-D observations. To efficiently infer object-goal relationships, we introduce a Fast-Slow inference mechanism: CP dynamically evaluates the uncertainty of the local LLM operating in Fast direct inference mode. Confident predictions (singleton sets) are directly utilized, while ambiguous cases trigger the same model in Slow deliberative reasoning mode. The resulting semantic scores are fused into a global value map to guide the agent toward the most promising exploration frontiers.

Formally, let $x \in \mathcal { X }$ denote the input query describing the relationship between an observed landmark and the target goal, and let $y ~ \in ~ \mathcal { Y } ~ = ~ \{ { \cdot } \mathrm { Y e s ^ { \prime } } , { \cdot } \mathrm { N o ^ { \prime } } \}$ be the true label indicating whether they are spatially co-located. We assume access to a small calibration dataset $\mathcal { D } _ { \mathrm { c a l } } ~ = ~ \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ drawn exchangeably from the same distribution as the test data. The central guarantee is that, at inference, given a new i.i.d. sample $x _ { \mathrm { t e s t } }$ with unknown true label $y _ { \mathrm { t e s t } } .$ , CP generates a prediction set $\mathcal { C } ( x _ { \mathrm { t e s t } } ) \subseteq \mathcal { Y }$ that is guaranteed to contain the true label with a user-specified probability.

The construction of CP proceeds as follows. First, we define a nonconformity score $s ( x , y ) = 1 - \hat { P } ( y | x )$ that quantifies how “atypical” a label y is for input x, where $\hat { P }$ is the model’s softmax probability. Intuitively, a higher nonconformity score indicates lower model confidence in the correct answer. We then evaluate this score on all calibration samples to obtain $\{ s _ { i } = s ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ . The key step is to compute a conformal threshold qˆ as the $\lceil ( N + 1 ) ( 1 - \alpha ) \rceil / N$ empirical quantile of these calibration scores, where $\alpha \in ( 0 , 1 )$ is a user-specified error rate $( \mathrm { e . g . , } \alpha = 0 . 1$ for 90% coverage).

For a new test input $x _ { \mathrm { t e s t } } .$ , the prediction set is constructed by including all labels whose nonconformity score falls below the threshold, effectively retaining all plausible answers:

$$
\mathcal { C } ( x _ { \mathrm { t e s t } } ) = \{ y \in \mathcal { V } | \hat { P } ( y | x _ { \mathrm { t e s t } } ) \geq 1 - \hat { q } \} .\tag{2}
$$

The generated prediction set ensures that the following coverage guarantee holds [8]:

$$
\mathbb { P } ( y _ { \mathrm { t e s t } } \in \mathcal { C } ( x _ { \mathrm { t e s t } } ) ) \geq 1 - \alpha .\tag{3}
$$

The intuition behind this guarantee is as follows: by choosing qˆ as the $( 1 - \alpha )$ -quantile of the calibration nonconformity scores, we ensure that at most an α fraction of calibration samples have a score exceeding qˆ. Since the test sample is exchangeable with the calibration data, it will similarly have a score $\leq \hat { q }$ with probability at least $1 - \alpha$ . By construction, we include all such labels in $\mathcal { C } ( x _ { \mathrm { t e s t } } )$ , thereby ensuring the coverage guarantee.

Crucially, the size of the prediction set $\vert \mathcal { C } ( x _ { \mathrm { t e s t } } ) \vert$ serves as a rigorous indicator of the model’s uncertainty. We leverage this cardinality as a dynamic switching signal for our Fast-Slow adaptive inference mechanism, allowing the system to distinguish between confident and ambiguous states, as detailed below.

3) Fast-Slow Inference Switching with CP: Next, we describe how the prediction set size $| \mathcal { C } ( x ) |$ serves as a switching signal between the fast direct inference and slow deliberative reasoning modes, mirroring the human cognitive duality of System 1 (fast, intuitive) and System 2 (slow, analytical) [9].

Each time the robot observes a new object, it first invokes the local LLM in fast direct inference mode (System 1) to assess the likelihood of finding the target goal near this object. We use the query formulation defined in Section III-D.1 and extract the logits for ‘Yes’ and ‘No’ to construct the prediction set $\mathcal C ( x )$ via Equation 2. The decision logic is as follows:

Singleton Set $( | { \mathcal { C } } ( x ) | = 1 ) \colon$ The prediction set contains exactly one label (either {‘Yes’} or $\{ \mathbf { \tilde { N o } } ^ { \prime } \} )$ . This indicates that the fast direct inference mode is confident in its judgment. In this case, we rely directly on the fast-mode output and use the softmax probability c (corresponding to the ${ \mathrm { \Omega } } ^ { 6 6 } { \mathrm { Y e s } } ^ { , 5 }$ token) as the semantic value for the observed object.

Empty or Doubleton Set $( | { \mathcal { C } } ( x ) | \neq 1 ) \colon$ The prediction set is either empty (∅) or contains both labels $( \{ \cdot \mathrm { Y e s } ^ { \prime } , \cdot \mathrm { N o } ^ { \prime } \} )$ ). An empty set suggests that neither answer meets the required confidence level (often due to high anomaly), while a doubleton suggests that both answers are plausible under the statistical guarantee (ambiguity). Both cases represent high uncertainty where fast direct inference fails to provide a unique answer. Consequently, the system re-invokes the same model in slow deliberative reasoning mode (System 2) to perform step-bystep reasoning, and the result from this deeper analysis is used as the target value.

![](images/90b565b63e52c728f6d7676751c2366c346d1bacc164aba255d7280b962c3b09.jpg)  
Fig. 3. Word cloud visualization of object frequencies across four room types. Larger font sizes indicate higher frequency of occurrence, highlighting the characteristic components validation.

When the slow mode is triggered, the same Qwen3-8B model is invoked with the enable\_thinking flag activated. This enables the model to generate an internal chainof-thought reasoning trace before producing the final answer. The reasoning process typically involves: (1) identifying the functional category of the observed landmark (e.g., “a couch is typically found in a living room”), (2) inferring the room type most likely associated with the target goal, and (3) evaluating cross-room spatial plausibility. This deliberative process yields a recalibrated confidence score that replaces the original uncertain estimate.

4) Collection of Calibration Set: To rigorously quantify uncertainty using CP, we require a calibration set $\mathcal { D } _ { \mathrm { c a l } } ~ =$ $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { \bar { N } }$ that accurately reflects typical indoor object arrangements. Instead of relying on manual annotation, which is unscalable, or real-world scanning, which is data-sparse, we propose a scalable, automated pipeline to construct $\mathcal { D } _ { \mathrm { c a l } }$ by distilling commonsense knowledge from LLMs and datasets. Our pipeline consists of three steps: Scene Generation, Object Filtering, and Pairwise Labeling.

Scene Description Generation. To ensure calibration data reflects realistic indoor object distributions, we randomly sample viewpoints from the Matterport3D training split across four primary room types: bedroom, living room, kitchen, and bathroom. For each sampled viewpoint, we utilize GPT-5 as a vision-language model to describe the RGB image, extracting a list of visible objects and the corresponding room category. For example, a bedroom viewpoint might yield “bed, nightstand, lamp, wardrobe, ...”. We collect 50 viewpoints per room type to ensure diversity. All generated object lists are manually screened and verified for rationality.

Frequency-based Object Filtering. Raw descriptions may contain rare artifacts or hallucinations. To extract robust characteristic objects, we perform frequency analysis on the generated corpus. We calculate the occurrence count of each unique object within specific room categories. As shown in Figure 3, objects cluster naturally by room function. We apply a filtering threshold $\tau _ { f r e q } = 2 0 \%$ (appearing in >10/50 descriptions) to identifying Characteristic Components (e.g., sofa for living room, oven for kitchen). This step effectively filters out noise while retaining essential spatial elements.

Automated Set Construction. Finally, we synthesize labeled query pairs $( x , y )$ to form the calibration set. The labeling logic is grounded in the functional coherence of indoor spaces. Positive samples $( y = " \mathrm { Y e s } ^ { \prime \prime } )$ are generated from pairwise permutations of characteristic components belonging to the same room category (e.g., Bed ↔ Nightstand), representing high-probability spatial assignments. Conversely, negative samples $( y ~ = ~ \mathrm { \ddot { ' } N o ^ { 3 3 } } )$ are drawn from pairwise permutations of components across distinct room categories (e.g., Bed ↔ Toilet), representing implausible spatial proximities. This structured approach yields a clean, balanced, and semantically valid dataset $\mathcal { D } _ { \mathrm { c a l } }$ for CP calibration with human feedback.

5) Global Decision with Semantic Exploration: To guide the agent effectively, we construct a Global Semantic Value Map by aggregating the semantic contributions of all observed objects. This process operates in three stages:

Individual Influence Field. For each discovered object $o _ { i }$ at location $\mathrm { l } _ { i } ,$ we obtain a base semantic score $v _ { i }$ from the LLM (positive for target-associated landmarks, negative otherwise). Rather than treating this as a point value, we model the object’s influence as a spatial field that decays exponentially with distance. The value at any map coordinate p induced by object $o _ { i }$ is given by:

$$
V _ { i } ( p ) = v _ { i } \cdot \exp \left( - { \frac { \| p - \mathbf { l } _ { i } \| } { \lambda } } \right)\tag{4}
$$

where λ is a decay constant controlling the spatial extent of influence. We set λ = 1.0m, which corresponds to typical room-scale spatial relationships: an object’s semantic signal remains significant within its immediate vicinity but diminishes beyond adjacent rooms.

Global Map Fusion. We obtain the Global Semantic Value Map $M _ { s e m }$ by superimposing these individual influence fields. By summing the values from all objects, $M _ { s e m } ( p ) =$ $\textstyle \sum _ { i } V _ { i } ( p )$ , overlapping fields from spatially clustered positive landmarks mutually reinforce each other, creating high-value regions that attract the agent. Conversely, negative landmarks create repulsive zones.

Frontier Selection. To balance goal-seeking with exploration, we integrate a Historical Value Map $M _ { h i s t }$ . The agent’s historical trajectory is projected onto the map with similar exponential decay to penalize revisiting explored areas. The final decision is made by fusing these maps: $\begin{array} { r l } { M _ { t o t a l } } & { { } = } \end{array}$ $M _ { s e m } + M _ { h i s t }$ . Frontier nodes are defined as the boundary cells between explored and unexplored regions on the obstacle map. The agent selects the frontier node with the highest cumulative value on $M _ { t o t a l }$ as the next mid-term goal, effectively prioritizing unexplored areas that are most semantically promising.

## E. Local Planning

The navigation process operates in a hierarchical manner. At fixed time intervals $\Delta t \ = \ 2 5$ steps, the robot invokes the global decision module described above to re-evaluate the semantic value map and select the next mid-term goal from the frontier nodes. This periodic replanning allows the agent to incorporate newly discovered objects and dynamically adjust its exploration strategy as the environment unfolds. We empirically find that this interval balances responsiveness to new observations with computational overhead.

Between successive global decisions, a local planner guides the robot toward the current mid-term goal. Given the robot’s current position and the selected target point, the Fast Marching Method [48] is employed to compute the shortest feasible path on the obstacle map in real-time, generating a sequence of waypoints. The path is computed on the obstacle channel of the semantic map, ensuring that the planned trajectory avoids known obstacles while respecting the map resolution of 0.05m per cell. To track this path, the agent first checks the proximity to the goal; if the distance is within 0.25m, the Stop action is triggered. Otherwise, it computes the angular deviation $\theta _ { d e v }$ between its current heading and the direction to the immediate next waypoint. Discrete actions are then deterministically derived from this deviation: if $| \theta _ { d e v } | > 1 5 ^ { \circ }$ , a rotational action (Turn Left if $\theta _ { d e v } > 0$ , Turn Right otherwise) is selected to align the agent; otherwise, a Move Forward action is executed.

## IV. EXPERIMENTS

## A. Simulation Setting

Simulation Platform. We conduct all simulated experiments on the Habitat simulator [49], a high-performance 3D embodied AI platform that supports realistic indoor scene rendering. The simulator provides the navigation agent with RGB-D observations at a resolution of $6 4 0 \times 4 8 0$ . The Horizontal Field of View (HFOV) is $7 9 ^ { \circ }$ for both RGB and depth sensors, with a valid depth range of 0.5m to 5m. It also offers rich sensor information, real-time agent states, and precise physics updates following each action. The action space includes six discrete actions: move forward (0.25m), turn left (30<sup>◦</sup>), turn right (30<sup>◦</sup>), look up, look down, and stop. We follow the task setting of the Habitat Challenge 2022 [50] and VLFM [47]. Each episode has a maximum time budget of 500 steps.

Datasets. We evaluate our method on three mainstream indoor scene datasets provided by the Habitat simulator: Gibson, HM3D v1, and HM3D $\mathbf { v } 2 .$ . These datasets feature a wide variety of complex and realistic indoor scenes, composed of colored 3D scans. Specifically, Gibson provides 5 different scenes with a total of 1, 000 episodes. HM3D v1 and v2 provide 20 and 36 scenes, offering 2, 000 and 1, 000 episodes, respectively. The target objects for all episodes belong to 6 common indoor categories.

Metrics. We adopt Success Rate (SR) and Success weighted by Path Length (SPL) as our primary evaluation metrics [51]. We conduct N test episodes. In each episode $i ,$ let $l _ { i }$ be the shortest-path distance from the agent’s starting position to the goal, and $p _ { i }$ be the length of the path actually taken by the agent. Let $S _ { i }$ be a binary indicator of success in episode i. Then, SR and SPL are calculated as follows:

$$
{ \mathrm { S R } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } S _ { i }\tag{5}
$$

$$
\mathrm { S P L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } \frac { l _ { i } } { \operatorname* { m a x } ( p _ { i } , l _ { i } ) }\tag{6}
$$

SR measures the effectiveness of the agent, while SPL jointly evaluates success and navigation efficiency.

LLM Configuration. We employ a hybrid knowledge retrieval system combined with a fast-slow inference mechanism. The configuration details are as follows:

• Static Memory. We utilize GPT-5 to pre-compute the probabilistic object relationship matrix for the 80 COCO categories, serving as a static commonsense memory for $O ( 1 )$ retrieval.

• Online Inference. For open-set objects, we employ Qwen3- 8B as the local inference engine. The Fast Mode uses the base model for rapid direct inference, while the Slow Mode activates the enable\_thinking flag for deliberative reasoning when high uncertainty is detected.

We set the temperature parameter $T ~ = ~ 1 0$ for scaling the logits. The conformal prediction calibration is performed with a user-specified error rate $\alpha \ = \ 0 . 5$ on a calibration set containing 1,200 object-pair relationship queries constructed from four typical room categories.

Perception Module. We adopt YOLOv7 [52] for common object detection, and GroundedSAM [44] processing RGB frames to extract semantic information. The detection confidence threshold is set to 0.5 to balance precision and recall. The semantic map has a spatial resolution of 0.05m per grid cell with dimensions of $4 8 0 \times 4 8 0$ cells, covering a 24m × 24m area.

Hardware Resources. All simulated experiments, including both fast and slow mode inference, are conducted on a workstation equipped with an NVIDIA RTX 4090D GPU (24GB VRAM), AMD Ryzen 9 7950X CPU, and 64GB RAM.

## B. Simulation Comparison

We compare CovNav with state-of-the-art methods on Gibson and HM3D datasets. The quantitative results are presented in Table I.

Baselines. We compare CovNav against state-of-the-art ZSON methods: (1) L3MVN [53], which uses LLMs to build a semantic value map for finding objects; (2) VLFM [47], which builds a value map using BLIP-2 to guide exploration; (3) GAMap [54], which guides exploration with multi-scale geometric and affordance attributes of the target object; (4) ApexNav [55], which adaptively switches between semanticdriven and geometry-driven exploration and employs targetcentric semantic fusion to reduce false detections.

Analysis. The results in Table I reveal a compelling trend: while baseline methods like GAMap excel in simpler environments (Gibson), their performance degrades significantly in complex, large-scale scenes (HM3D). Specifically, GAMap achieves a leading 85.7% SR on Gibson but drops to 53.1% on HM3D v1. This suggests that relying solely on geometric frontiers or uncalibrated semantic maps is insufficient when spatial ambiguity increases. In contrast, CovNav demonstrates superior generalization. On the challenging HM3D v1 and v2 benchmarks, it outperforms the runner-up ApexNav by margins of 0.8% and 1.7% respectively, while maintaining competitive efficiency. The key differentiator is CovNav’s ability to recognize when it doesn’t know. By adaptively escalating from fast direct inference to slow deliberative reasoning in ambiguous scenarios, CovNav approaches the performance of the Slow-Only variant without paying its full computational cost, which is particularly beneficial for complex HM3D environments where such ambiguity is more prevalent. The consistently high SPL scores further indicate that our CP-guided exploration reduces “random walking” behavior, leading to more direct paths to the goal.

![](images/b5cb7374d31707f6063456c261d65908e6a9c435065e3598ec0b61412813196c.jpg)  
Fig. 4. Visualization of the adaptive inference process (Target: Bed). At Step 14, the agent encounters semantic ambiguity while observing a couch. The CP module detects high uncertainty, triggering the Slow Mode to perform chain-of-thought reasoning (inferring “Living Room” from “Couch” and “Bedroom” from “Bed”), thereby guiding the agent to the correct room. At Step 112, the Fast Mode effectively filters out irrelevant areas (Bathroom) with high confidence, leading to a successful identification of the target at Step 119. The bottom row illustrates the real-time evolution of the Global Semantic Value Map (visualized as a red-green heatmap) and the agent’s trajectory.

TABLE I  
COMPARISON WITH ZSON BASELINES.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Planner</td><td rowspan="2">Training-free</td><td rowspan="2">Core Model</td><td colspan="2">Gibson</td><td colspan="2">HM3Dv1</td><td colspan="2">HM3Dv2</td></tr><tr><td>Success↑</td><td>SPL↑</td><td>Success↑</td><td>SPL↑</td><td>Success↑</td><td>SPL↑</td></tr><tr><td>L3MVN</td><td rowspan="5">Semantic</td><td>X</td><td>LLM</td><td>76.9</td><td>38.8</td><td>38.4</td><td>19.4</td><td>36.3</td><td>15.7</td></tr><tr><td>VLFM</td><td>√</td><td>VLM</td><td>81.8</td><td>46.5</td><td>50.9</td><td>23.6</td><td>56.9</td><td>27.5</td></tr><tr><td>GAMap</td><td>√</td><td>VLM</td><td>85.7</td><td>55.5</td><td>53.1</td><td>26.0</td><td></td><td></td></tr><tr><td>ApexNav</td><td>√</td><td>VLM</td><td></td><td></td><td>59.6</td><td>33.0</td><td>76.2</td><td>38.0</td></tr><tr><td>CovNav (Ours)</td><td>√</td><td>LLM</td><td>82.4</td><td>48.6</td><td>60.4</td><td>33.1</td><td>77.9</td><td>40.2</td></tr></table>

## C. Ablation Study

We conduct extensive ablation studies to validate the effectiveness of the proposed fast-slow inference mechanism and the impact of different system components. For computational efficiency, we randomly sample 100 episodes from each dataset (Gibson, HM3D v1, and HM3D v2) to perform the following ablation experiments. In these ablations, SR and SPL are first computed for each target category and then averaged over the six categories.

1) Fast-Slow Inference Mechanism: To evaluate the efficacy of our adaptive inference approach, we compare CovNav against three variants: (1) Fast-Only (System 1): Uses only the 8B model in fast direct inference mode; (2) Slow-Only (System 2): Consistently employs the same 8B model in slow deliberative reasoning mode; (3) Random Switch: Randomly switches between fast and slow modes. The results in Table II highlight the trade-off between performance and efficiency. The time of Random Switch and CovNav is estimated from the module-level latencies reported in Table VI.

The performance gap between Fast-Only and Slow-Only (e.g., 51.5% vs. 64.2% on HM3D v1) empirically validates the limitation of relying solely on fast direct inference from a small local model: single-pass inference lacks the deliberative capacity to handle complex spatial relations consistently. While the Slow-Only variant addresses this, its 3.40s latency is operationally infeasible. CovNav effectively breaks this “accuracy-efficiency” curse. It recovers nearly the entire performance gain of the slow mode (achieving 62.8% SR, within 1.4% of the upper bound) while keeping latency low (0.72s). Crucially, the 12.4% slow-mode activation rate reveals that most navigation decisions are routine; however, the few critical decisions that require slow deliberative reasoning are determinant for success. CovNav’s “strategic slowness” ensures resources are spent exactly on these high-leverage moments, rather than being wasted on trivial steps or withheld during critical ambiguities requiring deliberation.

TABLE II  
ABLATION STUDY OF FAST-SLOW INFERENCE MECHANISM. COMPARISON OF NAVIGATION PERFORMANCE AND INFERENCE EFFICIENCY ON GIBSON AND HM3D DATASETS.
<table><tr><td rowspan="3">Method</td><td>Gibson</td><td>HM3D v1</td><td>HM3D v2</td><td colspan="2">Efficiency</td></tr><tr><td>SR ↑ / SPL ↑</td><td>SR ↑ / SPL ↑</td><td>SR ↑ / SPL ↑</td><td>Time (s)</td><td>Call (%)</td></tr><tr><td>Fast-Only (System 1)</td><td>72.5 / 38.2</td><td>51.5 / 25.8</td><td>69.8 /  34.2</td><td>0.63</td><td>-</td></tr><tr><td>Slow-Only (System 2)</td><td>83.1  / 49.0</td><td>64.2 / 31.5</td><td>82.5 / 44.8</td><td>3.40</td><td>100</td></tr><tr><td>Random Switch</td><td>76.8 / 42.5</td><td>57.1 / 29.8</td><td>75.4 / 38.1</td><td>1.25</td><td>53.0</td></tr><tr><td>CovNav (Adaptive)</td><td>79.5 / 43.7</td><td>62.8  / 32.8</td><td>78.5 / 39.9</td><td>0.72</td><td>12.4</td></tr></table>

2) Impact of Conformal Prediction Parameter α: The parameter α in Conformal Prediction controls the tolerance for uncertainty. We analyze how varying α affects the switching rate between fast and slow modes and the overall navigation performance. Table III shows the sensitivity analysis.

TABLE III  
IMPACT OF CONFORMAL PREDICTION PARAMETER α. EVALUATING THE TRADE-OFF BETWEEN SWITCHING RATE AND NAVIGATION PERFORMANCE ACROSS DATASETS.
<table><tr><td rowspan="2">α</td><td rowspan="2">Switch (%)</td><td>Gibson</td><td>HM3D v1</td><td>HM3D v2</td></tr><tr><td> $\mathrm { S R } \uparrow / \mathrm { S P L } \uparrow$ </td><td> $\mathrm { S R } \uparrow / \mathrm { S P L } \uparrow$ </td><td> $\mathrm { S R } \uparrow / \mathrm { S P L } \uparrow$ </td></tr><tr><td>0.8</td><td>22.4</td><td>82.9 / 48.8</td><td> $6 1 . 5 ~ / ~ 3 1 . 9$ </td><td> ${ \bf 7 9 . 2 } \mathrm { ~ / ~ } 3 9 . 1 $ </td></tr><tr><td>0.5 (Ours)</td><td>12.4</td><td> $7 9 . 5 \ : / \ : 4 3 . 7 \ :$ </td><td> ${ \bf 6 2 . 8 \mathrm { ~ / ~ } } 3 2 . 8 $ </td><td> $7 8 . 5 \ : / \ : 3 9 . 9$ </td></tr><tr><td>0.2</td><td>8.2</td><td>80.1  / 46.5</td><td> $5 9 . 4 ~ / ~ 3 0 . 1 $ </td><td> $7 5 . 8 \mathrm { ~ / ~ } 3 7 . 5 $ </td></tr></table>

The CP parameter α serves as a tunable “risk tolerance” knob. Raising α to 0.8 creates a conservative agent that triggers the slow deliberative mode more frequently (22.4%), which explains its highest SR on Gibson (82.9%) and HM3D v2 (79.2%), as the additional deliberation overhead pays off in these environments. However, in the most challenging HM3D v1 scenes, excessive switching introduces decision noise, causing SR to fall below our default setting (61.5% vs. 62.8%). Conversely, a low α (0.2) creates an overconfident agent that rarely invokes the slow mode (8.2%), failing to trigger it when genuinely needed. This leads to consistent performance degradation across all datasets (e.g., 75.8% on HM3D v2), approaching the “Fast-Only” baseline. Our choice of $\alpha = 0 . 5$ achieves an effective trade-off between accuracy and cost, identifying a “sweet spot” where the system is sufficiently self-aware to catch errors without becoming paralyzed by computational overhead.

3) Mode Switching without Uncertainty: We investigate the performance when using fixed probability thresholds for fast-slow mode switching instead of the adaptive CP-based approach. Table IV presents the results for different fixed thresholds across datasets.

Increasing the fixed probability threshold imposes a stricter safety requirement, causing the agent to trigger the slow deliberative mode more frequently. For instance, a conservative threshold of 0.8 leads to a 42.7% switching rate. While this aggressive strategy yields high success on the simpler Gibson dataset (82.0%), it fails to generalize to complex scenes like HM3D v1 (dropping to 54.2%), barely surpassing the Fast-Only baseline. This limitation stems from the uncalibrated nature of raw logits: the model can be “wrong and confident” (hallucination), bypassing the threshold check. In contrast, CP acts as a statistical rectifier. By calibrating against a held-out set, CP ensures the uncertainty metric reflects true empirical correctness. Consequently, CovNav achieves superior performance on HM3D v1 (62.8%) with significantly lower computational cost (12.4% switching), proving that correctly quantifying uncertainty is more effective than merely increasing the safety threshold.

TABLE IV  
PERFORMANCE WITH FIXED DECISION THRESHOLDS. COMPARISON OF DECISION-MAKING WITHOUT UNCERTAINTY QUANTIFICATION.
<table><tr><td rowspan="2">Threshold</td><td rowspan="2">Switch (%)</td><td>Gibson</td><td>HM3D v1</td><td rowspan="2">HM3D v2</td></tr><tr><td>SR ↑ / SPL ↑</td><td> $\mathrm { S R } \uparrow / \mathrm { S P L } \uparrow$ </td></tr><tr><td>0.8</td><td>42.7</td><td>82.0  / 45.7</td><td> $5 4 . 2 \ : / \ : 2 6 . 5$ </td><td>SR ↑ / SPL ↑ 80.4 / 41.6</td></tr><tr><td>0.5</td><td>10.1</td><td>77.8 / 44.5</td><td>60.9 / 31.4</td><td>77.1 / 38.2</td></tr><tr><td>0.2</td><td>8.2</td><td>74.2 / 39.8</td><td>57.4 / 28.6</td><td>74.6 / 36.4</td></tr><tr><td>CovNav (CP)</td><td>12.4</td><td>79.5 / 43.7</td><td>62.8 / 32.8</td><td>78.5 / 39.9</td></tr></table>

4) System Components: We further decompose the system to validate the contribution of each module, including the semantic fusion strategy, confidence decay, and open-set perception, as follows.

TABLE V  
ABLATION OF SYSTEM COMPONENTS. ANALYZING THE CONTRIBUTION OF DIFFERENT MODULES.
<table><tr><td rowspan="2">Method</td><td>Gibson</td><td>HM3D v1</td><td>HM3D v2</td></tr><tr><td> $\mathrm { S R } \uparrow / \mathrm { S P L } \uparrow$ </td><td> $\mathrm { S R } \uparrow / \mathrm { S P L } \uparrow$ </td><td>SR ↑ / SPL ↑</td></tr><tr><td>w/o CP-calibrated Value</td><td>52.5 / 25.2</td><td>41.2 / 21.9</td><td>66.8 / 28.5</td></tr><tr><td>w/o Confidence Decay</td><td>80.2 / 46.8</td><td>51.3 / 25.8</td><td>65.4 / 31.2</td></tr><tr><td>w/o Open-set Perception</td><td>68.4 / 35.1</td><td> $5 4 . 6 \ : / \ : 2 7 . 5$ </td><td>69.5 / 34.1</td></tr><tr><td>CovNav (Ours)</td><td> $7 9 . 5 \ : / \ : 4 3 . 7 \ :$ </td><td> $6 2 . 8 \ : / \ : 3 2 . 8$ </td><td>78.5 / 39.9</td></tr></table>

The ablation results confirm that CovNav’s performance is not due to a single component but the synergy of its parts. First, the CP-calibrated Value is critical for efficient search. Removing this component (w/o CP-calibrated Value) effectively degrades the system to traditional frontierbased exploration, where the robot explores space based on geometric frontiers without semantic prioritization. Consequently, performance drops significantly across all datasets (e.g., 62.8% → 41.2% on HM3D v1), confirming that semantic guidance is indispensable. Confidence decay is equally vital; without it, the map becomes “stale” and dominated by historical errors, dropping SR to 51.3% on HM3D. Finally, Open-set perception is the foundational bedrock; reverting to closed-set detection artificially limits the agent’s understanding of context, causing a drastic performance collapse (54.6%). Thus, CovNav succeeds by combining rich open-set data (Perception) with fresh, reliable interpretation (Decay + CP).

## D. Efficiency and Runtime Analysis

We analyze the computational resource consumption of the proposed method, focusing on model size, memory usage, and latency. Table VI provides a detailed breakdown.

TABLE VI  
COMPUTATIONAL COST ANALYSIS. COMPARISON OF RESOURCE USAGE AND LATENCY ACROSS MODULES.
<table><tr><td>Module</td><td>Model Size</td><td>VRAM (GB)</td><td>Latency (ms)</td></tr><tr><td>Perception (BLIP&amp;YOLO)</td><td>125M</td><td>4.2</td><td>45</td></tr><tr><td>Fast Mode (Qwen3-8B-FP8)</td><td>8B</td><td>8.0</td><td>180</td></tr><tr><td>GroundedSAM</td><td>172M</td><td>3.8</td><td>300</td></tr><tr><td>Slow Mode (Thinking Enabled)</td><td>8B</td><td>8.0</td><td>2400</td></tr><tr><td>CovNav (Avg)</td><td>-</td><td>22.6</td><td>680</td></tr></table>

The efficiency analysis underscores the practical advantage of CovNav. While the thinking-enabled slow mode provides higher-quality deliberative inference, its latency (2400ms) is prohibitive for real-time control loops. By leveraging the Fast Mode (180ms) for the majority of queries, CovNav reduces the average inference latency to 680ms, a 3.5 times speedup over the slow-only approach. This latency is well within the acceptable range for mobile robot navigation, allowing for smooth and responsive behavior. Furthermore, the total memory footprint is 22.6GB, fitting comfortably within the memory constraints of consumer-grade GPUs (e.g., RTX 3090/4090) commonly found on high-end robotic workstations.

## E. Calibration Analysis

We first illustrate the effectiveness of our adaptive inference mechanism in resolving ambiguity. In scenarios involving implicit spatial relationships, the Fast Mode (System 1) often generates high logits for multiple potential candidates, creating a noisy graph of associations. As observed in Figure 5 (Left), the model assigns high probabilities to irrelevant pairs, such as linking “Toilet” with “Laptop” (0.60), implying a misleading certainty. This ambiguity causes the Conformal Prediction module to construct a large prediction set (size > 1), thereby triggering the Slow Mode (System 2). By engaging in chainof-thought reasoning, System 2 evaluates semantic plausibility and calibrates these logits. As shown in Figure 5 (Right), this process effectively suppresses noise by reducing the (“Toilet”, “Laptop”) score to 0.25, while sharpening the distribution around valid cues (e.g., reinforcing the “Bed”-“Chair” association from 0.49 to 0.75). This adaptive calibration ensures that the agent follows semantically valid landmarks rather than being misled by the uncalibrated confidence of the Fast Mode.

## F. Failure Cases Analysis

We classify failure cases on Gibson and HM3D into three modes (Table VII). Anomalous failed episodes are excluded from this analysis. Perception Failures (Type I) dominate (45%–52%), stemming from domain gaps $( \mathrm { e . g . , }$ blur, occlusion). These manifest as false negatives (missing objects causing timeouts) or false positives (misidentifying similar objects causing premature stops). Navigation Traps (Type II) (28%–35%) arise from geometric complexity. Agents may get stuck in local exploration minima, pursuing inaccessible semantic frontiers while revisit penalties hinder backtracking. Inference Limitations (Type III) (19%–21%) reflect the riskefficiency trade-off, including Calibration Failures (Fast Mode is confidently wrong) and Hallucinations (Slow Mode infers incorrect associations).

![](images/f6c5be58dcf3a0a8a2c67a0ead94839a1c16d4fa5441fca48e8a245a1a1c0d31.jpg)  
Fig. 5. Comparison of Logit Distributions. (a) Uncalibrated logits from the Fast Mode showing high uncertainty. (b) Calibrated logits after adaptive slowmode deliberation, showing a clean separation and higher confidence in the correct association.

TABLE VII  
DISTRIBUTION OF FAILURE MODES. VALUES SHOW PERCENTAGE OF FAILED EPISODES.
<table><tr><td>Dataset</td><td>Perception</td><td>Navigation</td><td>Inference</td></tr><tr><td>Gibson</td><td>45.5%</td><td>35.2%</td><td>19.3%</td></tr><tr><td>HM3D v1</td><td>52.3%</td><td>28.5%</td><td>19.2%</td></tr><tr><td>HM3D  $\mathbf { v } 2$ </td><td>48.8%</td><td>30.0%</td><td>21.2%</td></tr></table>

## V. REALWORLD EXPERIMENTS

To validate the efficacy of CovNav in physical environments, we conduct experiments on a real-world mobile robot platform.

Hardware Setup. We utilize a TurtleBot 4 mobile base equipped with an Intel RealSense L515 LiDAR camera, which replaces the default sensor to provide high-fidelity RGB and depth observations. To enable fully autonomous onboard computing, we mount a laptop with an NVIDIA RTX 3080 GPU, powered by an external battery, to run all perception and LLM inference processes locally.

Perception. The L515 sensor captures aligned RGB and depth images at a resolution of 640 × 480. We align the Horizontal Field of View (HFOV) to $5 6 ^ { \circ }$ to match the simulation settings. The camera is mounted at a height of 0.18m relative to the base, maintaining a horizontal viewing angle.

Action Space Control. We bridge the gap between continuous real-world dynamics and the discrete action space defined in simulation (Forward, Turn Left, Turn Right, Stop) through a velocity-pulse control scheme. Using the robot’s underlying PID controller, we execute “Forward” actions via a 0.25m/s velocity command to approximate a 0.2m displacement. Rotational actions are executed with an angular velocity of ±0.5rad/s, resulting in an approximate $3 0 ^ { \circ }$ turn.

![](images/db1639ca5901ed12b288fd2bccb08b5653552a45ef7f066579525d88b2645066.jpg)  
Fig. 6. Visualization of Real World Experiments. The figure depicts the navigation trajectory of the TurtleBot 4 platform (setup shown in top-left) searching for a toy. The system dynamically switches between Fast and Slow modes based on uncertainty. During the initial approach (Steps 11-27), the agent operates in Fast Mode for efficient exploration. At Step 42, high uncertainty at a semantic transition triggers the Slow Mode, enabling the agent to infer the correct path into the bedroom. The agent then reverts to Fast Mode (Step 60) and successfully localizes the target (Step 73). This demonstrates the efficacy of the adaptive fast-slow inference mechanism in handling real-world ambiguities.

Algorithm Deployment. To achieve standalone autonomy without reliance on cloud APIs, we deploy a quantized Qwen3-8B-FP8 model onboard. Additionally, the precomputed Object Relationship Matrix serves as an onboard static memory for efficient O(1) queries. The fast-slow inference mechanism is implemented by dynamically toggling the model’s generation parameters for open-set queries: the “enable\_thinking” flag is disabled for the Fast Mode to prioritize speed, and enabled for the Slow Mode to activate deliberative reasoning when uncertainty is high.

Real-world Navigation Process. We validate our system across three diverse multi-room indoor environments, testing with various semantic targets including toilet, sofa, bed, and toy. Figure 6 presents a representative navigation episode where the robot is tasked with finding a “Toy”. The trajectory breakdown demonstrates the effectiveness of our adaptive fastslow mechanism:

• Phase I: Efficient Exploration (Steps 0-41). The robot initiates the search in the hallway and dining area. In this open area with clear semantic landmarks, the CP module maintains a low uncertainty estimate and identifies low semantic relevance to the target “toy”. Consequently, the agent operates in Fast Mode, utilizing the efficient local planner to rapidly cover ground and move through the corridor without hesitation.

• Phase II: Slow Deliberation at Ambiguities (Step 42). As the robot reaches a critical decision point at the doorway, the visual context becomes ambiguous (transition from corridor to living room). This semantic uncertainty triggers the Slow Mode. The quantized Qwen3-8B-FP8 model is re-invoked with thinking enabled to perform chain-of-thought reasoning. It successfully infers the high probability of the target’s presence around the sofa and in the adjacent bedroom based on the semantic context, directing the robot to traverse the living room.

• Phase III: Precise Localization (Steps 43-73). Once inside the target room, the uncertainty subsides. The system reverts to Fast Mode to perform a fine-grained object search, successfully localizing the target at Step 73.

The successful completion of this episode in 73 steps validates the stability of our onboard deployment. Crucially, the system only activated the computationally intensive Slow Mode during the critical transition phase (Step 42), while relying on the efficient Fast Mode for the majority of the trajectory. This selective activation strategy allows the robot to maintain realtime responsiveness on the limited perception-compute budget of the mobile platform, verifying the practical viability of CovNav in physical deployments.

## VI. CONCLUSION

In this work, we presented CovNav, a novel adaptive inference framework that bridges the gap between efficiency and capability in embodied AI. Drawing inspiration from dualprocess theories, we introduced a mechanism that dynamically switches a local LLM between fast direct inference and slow deliberative reasoning modes. By leveraging Conformal Prediction to rigorously quantify uncertainty without additional training, CovNav serves as a reliable gatekeeper for this switching logic. Extensive experiments on Gibson and HM3D demonstrate that our method achieves state-of-the-art ZSON performance, matching the navigation performance of always using the slow deliberative mode while reducing latency to nearly one-third. Ultimately, CovNav offers a principled path toward trustworthy embodied agents that are self-aware of their limitations, providing valuable guidance for future research in integrating uncertainty quantification with active perception and robust neuro-symbolic planning.

## REFERENCES

[1] Y. Liu, W. Chen, Y. Bai, X. Liang, G. Li, W. Gao, and L. Lin, “Aligning cyber space with physical world: A comprehensive survey on embodied ai,” IEEE/ASME Transactions on Mechatronics, vol. 30, no. 6, pp. 7253– 7274, 2025.

[2] D. S. Chaplot, D. P. Gandhi, A. Gupta, and R. R. Salakhutdinov, “Object goal navigation using goal-oriented semantic exploration,” Advances in Neural Information Processing Systems, vol. 33, pp. 4247–4258, 2020.

[3] R. Fukushima, K. Ota, A. Kanezaki, Y. Sasaki, and Y. Yoshiyasu, “Object memory transformer for object goal navigation,” in 2022 International Conference on Robotics and Automation, 2022, pp. 11 288– 11 294.

[4] K. Zhou, K. Zheng, C. Pryor, Y. Shen, H. Jin, L. Getoor, and X. E. Wang, “Esc: Exploration with soft commonsense constraints for zero-shot object navigation,” in International Conference on Machine Learning, 2023, pp. 42 829–42 842.

[5] P. Wu, Y. Mu, B. Wu, Y. Hou, J. Ma, S. Zhang, and C. Liu, “Voronav: voronoi-based zero-shot object navigation with large language model,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 53 737–53 775.

[6] C. Wen, Y. Huang, H. Huang, Y. Huang, S. Yuan, Y. Hao, H. Lin, Y.- S. Liu, and Y. Fang, “Zero-shot object navigation with vision-language models reasoning,” in International Conference on Pattern Recognition, 2025, pp. 389–404.

[7] Y. Cai, X. He, M. Wang, H. Guo, W.-Y. Yau, and C. Lv, “Cl-cotnav: Closed-loop hierarchical chain-of-thought for zero-shot object-goal navigation with vision-language models,” arXiv preprint arXiv:2504.09000, 2025.

[8] A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal prediction and distribution-free uncertainty quantification,” arXiv preprint arXiv:2107.07511, 2021.

[9] D. Kahneman, Thinking, fast and slow. Farrar, Straus and Giroux, 2011.

[10] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[11] J. Zhang, N. Lin, L. Hou, L. Feng, and J. Li, “Adaptthink: Reasoning models can learn when to think,” arXiv preprint arXiv:2505.13417, 2025.

[12] S. Tu, J. Lin, Q. Zhang, X. Tian, L. Li, X. Lan, and D. Zhao, “Learning when to think: Shaping adaptive reasoning in R1-style models via multi-stage RL,” in Advances in Neural Information Processing Systems (NeurIPS), 2025.

[13] D. Almeida, “Introducing system one models & Jev,” TypeSafe AI Blog, 2026. [Online]. Available: https://typesafe.ai/blog/introducingsystem-one-models-and-jev

[14] G. Ling, M. Xue, and Z. Ye, “Jev in the wild: A data-driven analysis of the Jev model’s functionality, applications and ecosystem,” arXiv preprint arXiv:2609.30216, 2026.

[15] J. Sun, J. Wu, Z. Ji, and Y.-K. Lai, “A survey of object goal navigation,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 2292–2308, 2025.

[16] J. Chen, G. Li, S. Kumar, B. Ghanem, and F. Yu, “How to not train your dragon: Training-free embodied object goal navigation with semantic frontiers,” in Robotics: Science and Systems, 2023.

[17] M. F. Ginting, S.-K. Kim, D. D. Fan, M. Palieri, M. J. Kochenderfer, and A.-a. Agha-Mohammadi, “Seek: Semantic reasoning for object goal navigation in real world inspection tasks,” arXiv preprint arXiv:2405.09822, 2024.

[18] X. Lei, M. Wang, W. Zhou, and H. Li, “Gaussnav: Gaussian splatting for visual navigation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 4108–4121, 2025.

[19] W. Ou, B. Luo, X. Xu, Y. Feng, and Y. Zhao, “Reinforcement learned multiagent cooperative navigation in hybrid environment with relational graph learning,” IEEE Transactions on Artificial Intelligence, vol. 6, no. 1, pp. 25–36, 2025.

[20] Z. Wang and G. Tian, “Goal-oriented visual semantic navigation using semantic knowledge graph and transformer,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 1647–1657, 2025.

[21] Q. Liu, Y. Li, Y. Xu, L. Han, Z. Liu, and H. Wang, “Embodied navigation in unknown environments with implicit scene memory and target-aware memory retrieval,” IEEE Transactions on Artificial Intelligence, vol. 7, no. 4, pp. 2387–2400, 2026.

[22] B. Chen, H. Zhu, S. Yao, S. Lu, P. Zhong, Y. Sheng, and J. Wang, “Socially aware object goal navigation with heterogeneous scene representation learning,” IEEE Robotics and Automation Letters, vol. 9, no. 8, pp. 6792–6799, 2024.

[23] H. Yoo, Y. Choi, J. Park, and S. Oh, “Commonsense-aware object value graph for object goal navigation,” IEEE Robotics and Automation Letters, vol. 9, no. 5, pp. 4423–4430, 2024.

[24] C. Lu, M. Liu, Z. Luan, Y. He, and B. Chen, “Multi-view spatial context and state constraints for object-goal navigation,” IEEE Robotics and Automation Letters, vol. 10, no. 3, pp. 2207–2214, 2025.

[25] Y. Kuang, H. Lin, and M. Jiang, “Openfmnav: Towards open-set zeroshot object navigation via vision-language foundation models,” in Findings of the Association for Computational Linguistics: NAACL 2024, 2024, pp. 338–351.

[26] S. Lian and F. Zhang, “Tdanet: Target-directed attention network for object-goal visual navigation with zero-shot ability,” IEEE Robotics and Automation Letters, vol. 9, no. 9, pp. 8075–8082, 2024.

[27] A. Majumdar, G. Aggarwal, B. Devnani, J. Hoffman, and D. Batra, “Zson: Zero-shot object-goal navigation using multimodal goal embeddings,” Advances in Neural Information Processing Systems, vol. 35, pp. 32 340–32 352, 2022.

[28] L. Zhang, Q. Zhang, H. Wang, E. Xiao, Z. Jiang, H. Chen, and R. Xu, “Trihelper: Zero-shot object navigation with dynamic assistance,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024, pp. 10 035–10 042.

[29] A. Debnath, G. J. Stein, and J. Kosecka, “Semnav: A model-based planner for zero-shot object goal navigation using vision-foundation models,” arXiv preprint arXiv:2506.03516, 2025.

[30] H. Singh, R. J. Das, M. Han, P. Nakov, and I. Laptev, “Malmm: Multiagent large language models for zero-shot robotics manipulation,” arXiv preprint arXiv:2411.17636, 2024.

[31] P. Arjun, A. Melnik, and G. C. Nandi, “Cognitive planning for object goal navigation using generative ai models,” in NeurIPS 2024 Workshop on Open-World Agents, 2024.

[32] W. Zu, W. Song, R. Chen, Z. Guo, F. Sun, Z. Tian, W. Pan, and J. Wang, “Language and sketching: An llm-driven interactive multimodal multitask robot navigation framework,” in 2024 IEEE International Conference on Robotics and Automation, 2024, pp. 1019–1025.

[33] D. Nie, X. Guo, Y. Duan, R. Zhang, and L. Chen, “Wmnav: Integrating vision-language models into world models for object goal navigation,” arXiv preprint arXiv:2503.02247, 2025.

[34] G. Zhou, Y. Hong, and Q. Wu, “Navgpt: Explicit reasoning in visionand-language navigation with large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7641–7649.

[35] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in International Conference on Machine Learning, 2016, pp. 1050–1059.

[36] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[37] L. Lindemann, M. Cleaveland, G. Shim, and G. J. Pappas, “Safe planning in dynamic environments using conformal prediction,” IEEE Robotics and Automation Letters, vol. 8, no. 8, pp. 5116–5123, 2023.

[38] A. Dixit, L. Lindemann, S. X. Wei, M. Cleaveland, G. J. Pappas, and J. W. Burdick, “Adaptive conformal prediction for motion planning among dynamic agents,” in Learning for Dynamics and Control Conference, 2023, pp. 300–314.

[39] A. Z. Ren, A. Dixit, A. Bodrova, S. Singh, S. Tu, N. Brown, P. Xu, L. Takayama, F. Xia, J. Varley, Z. Xu, D. Sadigh, A. Zeng, and A. Majumdar, “Robots that ask for help: Uncertainty alignment for large language model planners,” in Conference on Robot Learning (CoRL), 2023.

[40] A. Z. Ren, J. Clark, A. Dixit, M. Itkina, A. Majumdar, and D. Sadigh, “Explore until confident: Efficient exploration for embodied question answering,” arXiv preprint arXiv:2403.15941, 2024.

[41] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson et al., “Language models (mostly) know what they know,” arXiv preprint arXiv:2207.05221, 2022.

[42] B. Kumar, C. Lu, G. Gupta, A. Palepu, D. Bellamy, R. Raskar, and A. Beam, “Conformal prediction with large language models for multichoice question answering,” arXiv preprint arXiv:2305.18404, 2023.

[43] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International Conference on Machine Learning, 2022, pp. 12 888– 12 900.

[44] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded sam: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.

[45] S. Bird, “Nltk: the natural language toolkit,” in Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions, 2006, pp. 69–72.

[46] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in European Conference on Computer Vision, 2014, pp. 740– 755.

[47] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “Vlfm: Visionlanguage frontier maps for zero-shot semantic navigation,” in 2024 IEEE International Conference on Robotics and Automation, 2024, pp. 42–48.

[48] J. A. Sethian, “A fast marching level set method for monotonically advancing fronts.” Proceedings of the National Academy of Sciences, vol. 93, no. 4, pp. 1591–1595, 1996.

[49] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik et al., “Habitat: A platform for embodied ai research,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 9339–9347.

[50] K. Yadav, S. K. Ramakrishnan, J. Turner, A. Gokaslan, O. Maksymets, R. Jain, R. Ramrakhya, A. X. Chang, A. Clegg, M. Savva, E. Undersander, D. S. Chaplot, and D. Batra, “Habitat challenge 2022,” https://aihabitat.org/challenge/2022/, 2022.

[51] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva et al., “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.

[52] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 7464–7475.

[53] B. Yu, H. Kasaei, and M. Cao, “L3mvn: Leveraging large language models for visual target navigation,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2023, pp. 3554–3560.

[54] S. Yuan, H. Huang, Y. Hao, C. Wen, A. Tzes, and Y. Fang, “Gamap: Zero-shot object goal navigation with multi-scale geometric-affordance guidance,” Advances in Neural Information Processing Systems, vol. 37, pp. 39 386–39 408, 2024.

[55] M. Zhang, Y. Du, C. Wu, J. Zhou, Z. Qi, J. Ma, and B. Zhou, “Apexnav: An adaptive exploration strategy for zero-shot object navigation with target-centric semantic fusion,” arXiv preprint arXiv:2504.14478, 2025.