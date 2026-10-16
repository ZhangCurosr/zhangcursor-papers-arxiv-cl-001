(e) Qwen-Image-Edit-2509: In-Domain Avg

# OmniVerifier-C1: Decomposed Graded Verifier for Rule-Governed Generative World Modeling

Bowei Liu<sup>1,2∗</sup> Xinchen Zhang<sup>1∗†</sup> Xuhuan Li<sup>1∗</sup> Tianlin Zhang<sup>2</sup> Kaian Jiang<sup>1</sup> Jingwei Liu<sup>2,4</sup> Congyang Zhao<sup>2</sup> Yifei Qian<sup>2</sup> Junhong Liu<sup>3</sup> Zhiheng Li<sup>1</sup> Yujiu Yang<sup>1‡</sup> Yang Cai<sup>2</sup> Ling Yang<sup>4‡</sup>

<sup>1</sup>Tsinghua University <sup>2</sup> Amap, Alibaba Group <sup>3</sup> Microcyto <sup>4</sup> Peking University <sup>∗</sup>: Equal Contribution. <sup>†</sup>: Project Leader. <sup>‡</sup>: Corresponding Authors.

Ð https://github.com/Cominclip/OmniVerifier

## Abstract

Generative visual verifiers play a crucial role in advancing multimodal intelligence. To enhance the robustness ofvisual verifiers in complex multimodal scenarios involving compositional requirements, we move beyond conventional binary verifiers that produce only True/False discrete judgments and introduce a verifier that automatically decomposes complex prompts or environmental rules into finegrained components and assigns pointwise scores to provide a continuous and more informative critique signal. Our experiments further demonstrate that these structured and interpretable outputs improve the verifier’s generalization to out-of-domain multimodal critic tasks. To train this verifier, we introduce a Decoupled RL training strategy that jointly optimizes sub-question-level binary judgment and global alignment scoring, using precise local feedback to develop fine-grained verification capabilities that provide a foundation for reliable continuous scoring at the global level. Under a general reinforcement learning framework, we first empirically demonstrate that the proposed pointwise graded verifier significantly outperforms binary verifiers in traditional text-to-image scenarios. We then extend our investigation to rule-governed generative world modeling where generative models are leveraged for world simulation and world reasoning under explicit rules. From a theoretical standpoint, we further show that a pairwise verification paradigm yields more accurate critiques than the pointwise formulation when ground-truth images are available, with empirical results on diverse structured world-modeling tasks such as Maze and Sudoku further validating the efectiveness of our approach. Overall, our findings suggest a key insight: the transition from generative models to structured world-modeling agents critically hinges on the availability of accurate pairwise graded visual verifiers.

![](images/f36fb23a46240046336ba57feb9989e549f7ae6d00c0d72d3dfe200f2106661f.jpg)

![](images/406e21a2bb29e30585b1d81d831c0e21c155c93a207e500c305c4aae71b796d3.jpg)

![](images/3a702288f9f2a7944d06b7a3fe69b492c60884dffa0e85f1196b65951f60285c.jpg)

![](images/ee082703f966118256e71a890bfb58a1a612f76e9783d0336dbde9172d0fba3f.jpg)

![](images/83a34761c4bc3d0a9e5715144d0b37ff5a44ad828b083e57bf509ed617d530cf.jpg)

![](images/60d936ec09c41e2f9bd566506204895716b6c3fb4ad2ccc405c968d242011ddf.jpg)

![](images/d9af48188f22b47c46e1406a11342e577c75110e3e8219af8d9e4c0245b53cc5.jpg)

![](images/e0fb9824c7a2f9576138dbf3dfa5fb64fdf18e3d28f1f3c8b05b651f69c48530.jpg)  
Figure 1: Panels (a)–(d) highlight the advantages of the graded verifier over the binary verifier as a generative reward model, while panels (e)–(h) show that pairwise verification provides more reliable and accurate feedback signals than pointwise verification.

## 1 Introduction

Multimodal generative models have made remarkable progress in recent years (Cao et al., 2025; Seedream et al., 2025; Chen et al., 2026), producing visually compelling images conditioned on textual descriptions. Beyond visual fidelity (Wiedemer et al., 2025; Zhang et al., 2024b; Yang et al., 2025b), a central challenge lies in whether generated images faithfully follow complex instructions and reflect coherent world knowledge (Zhou et al., 2025; Jin et al., 2026; Deng et al., 2025; Yang et al., 2025a; Huang et al., 2026; Zhang et al., 2026b). Motivated by this challenge, visual verifiers have emerged as a crucial component for assessing the correctness of generated results. By providing feedback on the alignment between images and textual descriptions, verifiers not only serve as evaluation tools, but also act as general critic models for inference time scaling and reinforcement learning-based training. As generative models are increasingly applied to open-ended environments and reasoning-intensive tasks, the reliability and expressiveness of visual verifiers become a key bottleneck for advancing multimodal intelligence.

Despite their importance, most existing visual verifiers (Zhang et al., 2025b; Wang et al., 2025a; Xiong et al., 2025) adopt a binary judgment paradigm that produces only True/False decisions about whether an image satisfies a given prompt. Although simple and eficient, this design has several inherent limitations. First, binary supervision provides coarse feedback that fails to capture the partial correctness of generated samples. By assigning the same negative label to imperfect samples of varying quality, binary rewards obscure meaningful quality diferences and weaken the within-group learning signals required by group-relative reinforcement learning. Moreover, the limited information conveyed by binary labels may encourage shortcut learning and increase susceptibility to reward hacking (Shao et al., 2025; Wang et al., 2026c), as models may exploit superficial patterns to predict the expected labels without adequately reasoning about the e subtl compositional relationships among objects, attributes, and spatial constraints.

To address these limitations, we propose the Decomposed Graded Verifier (OmniVerifier-C1), which automatically decomposes complex prompts or environmental rules into fine-grained sub-questions and performs structured reasoning over each component. It assigns continuous scores based on the proportion of satisfied sub-questions, capturing partial correctness and providing informative critique signals. Our experiments demonstrate that decomposition-based structured reasoning significantly improves the verifier’s out-of-domain multimodal critic capabilities. To train this verifier, we introduce a Decoupled RL training strategy that jointly optimizes sub-question-level binary judgment and global alignment scoring, using precise 0/1 feedback on individual sub-questions to develop strong fine-grained verification capabilities that support reliable global scoring. Moreover, our theoretical analysis shows that the decompositionand-scoring paradigm better preserves the underlying true quality ordering among generated samples, providing finer-grained learning signals for group-relative reinforcement learning.

Building upon this verifier, we further explore its role as a generative reward model for reinforcement learning across two types of tasks. First, we apply the verifier to traditional text-to-image alignment, where it provides fine-grained feedback that encourages generative models to satisfy detailed compositional relationship. Second, we extend our study to the more reasoning-intensive and challenging setting of structured generative world modeling, where generative models are used to simulate environments and reason over sequential states. In these rule-governed environments, we find that pairwise verification, which compares generated states against ground-truth states, provides more accurate and reliable critiques than pointwise verification. This observation suggests that the efectiveness of visual verification in rulegoverned world-modeling scenarios critically depends on the availability of reference states for comparison. Extensive experiments across diverse tasks, such as Maze and Sudoku, demonstrate that the proposed verifier provides more reliable reward signals and substantially improves reinforcement learning performance.

Our contributions can be summarized as follows:

• We propose the Decomposed Graded Verifier, which automatically decomposes complex prompts or environmental rules into fine-grained sub-questions, performs structured reasoning over each component, and assigns continuous scores based on the proportion of satisfied sub-questions. Our experiments demonstrate that its structured outputs significantly improve out-of-domain multimodal critic capabilities.

• We introduce a Decoupled RL training strategy that jointly optimizes sub-question-level binary judgment and global alignment scoring, using precise 0/1 feedback on individual sub-questions to develop fine-grained verification capabilities that provide a foundation for reliable scoring.

• We demonstrate that the proposed verifier serves as an efective generative reward model for reinforcement learning, improving performance in both traditional text-to-image generation and rule-governed generative world modeling tasks.

• We provide empirical validation across diverse structured reasoning environments, showing that accurate pairwise decomposed verifiers are a key enabler for transforming generative models into rule-governed world-modeling agents.

## 2 Related Work

## 2.1 Generative Verifier or Reward Models

Generative reward models (Mahan et al., 2024; Malik et al., 2025; Tan et al., 2024; Liu et al., 2024; Wang et al., 2026c) difer fundamentally from traditional reward models (Xu et al., 2023; Zhang et al., 2024a; Ouyang et al., 2022; Zhang et al., 2024c) in that they produce explicit rationales rather than merely assigning scalar scores. OmniVerifier (Zhang et al., 2025b) first introduces the concept of Generative Universal Verifier and presents ViVerBench, a comprehensive and challenging benchmark for evaluating multimodal reasoning and critique capabilities. DeepSeekMath-V2 (Shao et al., 2025) further proposes meta-verification and integrates it into verifier training to enhance faithfulness and reliability. OmniVerifier-M1 (Zhang et al., 2026d) extends metaverification to multimodal scenarios by introducing symbolic spatial annotations, such as bounding boxes, points or lines as verifiable rationales. These rule-based supervision signals substantially alleviate reward hacking and enhance verifier generalization. Despite their improved interpretability, existing generative verifiers still rely on coarse binary decisions as final judgments, and thus struggle to provide dense and optimization-friendly reward signals for reinforcement learning. In this paper, we introduce a Decoupled RL training strategy to train the Decomposed Graded Verifier, which provides reliable continuous scores through explicit fine-grained decomposition.

## 2.2 Generative Model as World Reasoner

Multimodal generative models have made remarkable progress (Lin et al., 2025; Wang et al., 2026b;a; Zhang et al., 2026c) in semantic alignment and aesthetic quality. Beyond visual fidelity, they are increasingly expected to serve as reasoning-capable world simulators (OpenAI, 2025b; Google, 2025; 2024). To quantify the world reasoning capabilities of current generative models, several benchmarks (Li et al., 2025; Sun et al., 2025; Han et al., 2026) have been proposed. WISE (Niu et al., 2025) focuses on evaluating generative models’ understanding of cultural commonsense and scientific knowledge. MMGR (Cai et al., 2025) is designed to probe generative models’ spatial, temporal, physical, and logical reasoning capabilities. Motivated by these challenges, recent studies (Tan et al., 2026; Zhang et al., 2026a; 2025a) have explored new training paradigms and input modalities to learn the underlying dynamics of world evolution. Emu3.5 (Cui et al., 2025) advances world modeling by learning native next-state prediction over long-horizon interleaved vision-language sequences. UnityVideo (Huang et al., 2025) improves world-aware video generation through unified multi-modal and multi-task learning across diverse visual representations. In contrast to these eforts, we focus on improving the reliability and accuracy of reinforcement learning rewards for visual generation in rule-governed world-modeling scenarios, thereby encouraging generative models to develop stronger and more generalizable world reasoning capabilities.

## 3 OmniVerifier-C1: Decomposed Graded Verifier

In this section, we first analyze the theoretical advantages of the graded verifier over the binary verifier as a generative reward model. We then empirically demonstrate that our verifier develops fine-grained and generalizable multimodal verification capabilities through the Decoupled RL strategy, which jointly optimizes sub-question-level binary judgment and global alignment scoring.

![](images/e7da3c1b02642f992fea855517e0009642d14ec7e0ef01296301b86d4ba006c2.jpg)  
Figure 2: Overview of our training pipeline. Stage 1 trains the Decomposed Graded Verifier with decoupled reinforcement learning to produce fine-grained continuous rewards. Stage 2 introduces a pairwise judgment paradigm that provides cleaner and more reliable reward signals for generative model reinforcement learning.

## 3.1 Theoretical Advantage of Decomposed Graded Rewards for Generative RL

Drawbacks of Binary Reward. Binary verifiers have emerged as an efective tool for improving text-toimage generation quality through inference-time scaling (Zhang et al., 2025b). When they detect semantic inconsistencies between a generated image and its textual description, they produce editing instructions that guide the generative model to iteratively refine the visual output. For this purpose, coarse binary feedback is often suficient, as the verifier primarily serves as a gating mechanism that determines whether further refinement is needed. However, binary verifiers face inherent limitations when used as reward models for reinforcement learning, as their feedback is insuficiently dense and discriminative to distinguish among imperfect samples of varying visual quality and thereby guide efective generative policy optimization.

Revisiting Decomposition-Based Continuous Rewards. To address this limitation, we introduce the Decomposed Graded Verifier, a structured verification paradigm that elicits continuous scores from visual verifiers. Instead of reducing assessment to a single binary judgment, it decomposes a complex prompt into fine-grained sub-questions, evaluates each sub-question individually with binary judgments, and aggregates these local decisions into a continuous score that reflects the degree of prompt satisfaction. By preserving partial correctness in generated samples, this fine-grained reward signal provides denser and more optimization-friendly supervision for reinforcement learning.

Assume that the textual prompt c can be losslessly decomposed into K sub-questions $Q = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { K } \}$ Let $z ( x ) \in \{ 0 , 1 , . . . , \hat { K } \}$ denote the number of sub-questions satisfied by a visual output x. We define the continuous reward and the discrete reward as $\begin{array} { r } { r _ { c } ( x ) = \frac { z ( x ) } { K } } \end{array}$ and $r _ { d } ( x ) = \mathbf { 1 } [ z ( x ) = K ]$ respectively. Let $q _ { k } : = \mathbb { P } ( z ( x ) = k )$ denote the probability that a sample satisfies exactly k sub-questions, where $k = 0 , \ldots , K$ . To quantify how well a reward recovers the true quality ordering, we define the ranking accuracy as

$$
\operatorname { A c c } _ { \mathrm { r a n k } } ( r ) : = \mathbb { P } { \big ( } ( r ( x ) - r ( x ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) > 0 \mid z ( x ) \neq z ( x ^ { \prime } ) { \big ) } ,\tag{1}
$$

Table 1: Performance on ViVerBench of the Binary Verifier and Graded Verifier.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Concept Existence</td><td colspan="2">Object Relation</td><td colspan="2">World Dynamics</td><td colspan="3">Image Annotation</td><td colspan="3">State Value Evaluation</td><td colspan="3">STEM</td><td rowspan="2">Overall</td></tr><tr><td>Obj.</td><td>Attr.</td><td>Abs.P.</td><td>Spat.</td><td>N-Spat.</td><td>S-Phy</td><td>D-Phy</td><td>BBox</td><td>Point</td><td>Count</td><td>Maze</td><td>F.Lake</td><td>Robot.</td><td>GUI Chart</td><td>LaTeX</td><td></td></tr><tr><td>Qwen3-VL-8B</td><td>0.710</td><td>0.690</td><td>0.562</td><td>0.642</td><td>0.716</td><td>0.604</td><td>0.593</td><td>0.831</td><td>0.741</td><td>0.489</td><td>0.420</td><td>0.693</td><td>0.671</td><td>0.787</td><td>0.540</td><td>0.773</td><td>0.654</td></tr><tr><td>+ RL(Binary)</td><td>0.732</td><td>0.727</td><td>0.579</td><td>0.713</td><td>0.754</td><td>0.600</td><td>0.596</td><td>0.843</td><td>0.770</td><td>0.566</td><td>0.434</td><td>0.674</td><td>0.645</td><td>0.782</td><td>0.526</td><td>0.795</td><td>0.671</td></tr><tr><td>+ SFT(Binary)</td><td>0.689</td><td>0.695</td><td>0.620</td><td>0.654</td><td>0.703</td><td>0.546</td><td>0.627</td><td>0.802</td><td>0.735</td><td>0.504</td><td>0.414</td><td>0.714</td><td>0.661</td><td>0.792</td><td>0.538</td><td>0.802</td><td>0.656</td></tr><tr><td>+SFT(Binary) + RL(Binary)</td><td>0.735</td><td>0.721</td><td>0.554</td><td>0.719</td><td>0.748</td><td>0.521</td><td>0.636</td><td>0.840</td><td>0.765</td><td>0.532</td><td>0.446</td><td>0.687</td><td>0.682</td><td>0.799</td><td>0.553</td><td>0.814</td><td>0.672</td></tr><tr><td>+SFT(Graded)</td><td>0.724</td><td>0.709</td><td>0.535</td><td>0.679</td><td>0.657</td><td>0.613</td><td>0.596</td><td>0.817</td><td>0.752</td><td>0.543</td><td>0.427</td><td>0.661</td><td>0.576</td><td>0.769</td><td>0.656</td><td>0.830</td><td>0.659</td></tr><tr><td>+SFT(Graded)+RL(Graded)</td><td>0.758</td><td>0.735</td><td>0.541</td><td>0.736</td><td>0.766</td><td>0.573</td><td>0.582</td><td>0.859</td><td>0.781</td><td>0.559</td><td>0.466</td><td>0.679</td><td>0.693</td><td>0.810</td><td>0.567</td><td>0.839</td><td>0.684</td></tr></table>

where $x , x ^ { \prime } \sim \mathrm { i . i . d . } \ \pi _ { \theta } ( \cdot \mid c )$ . This measures the probability that the reward assigns the correct ordering to a pair of samples with diferent true quality levels.

For the continuous reward, $\begin{array} { r } { r _ { c } ( x ) = \frac { z ( x ) } { K } } \end{array}$ is strictly monotonic with respect to $z ( x )$ . Therefore, whenever $z ( x ) \neq z ( x ^ { \prime } )$ , we have $( r _ { c } ( x ) - r _ { c } ( \dot { x } ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) > 0 \nonumber$ , which gives $\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) = 1$ . For the discrete reward, let $A : = \{ ( r _ { d } ( x ) - r _ { d } ( x ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) > 0 \}$ and $B : = \{ z ( x ) \ \stackrel { . } { \neq } \ z ( x ^ { \prime } ) \}$ . Then $\begin{array} { r } { \operatorname { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) = \operatorname { \mathbb { P } } ( A \mid B ) = { \frac { \operatorname { \mathbb { P } } ( A \cap B ) } { \operatorname { \mathbb { P } } ( B ) } } } \end{array}$ . Since x and x<sup>′</sup> are independent, we have $\begin{array} { r } { \mathbb { P } ( B ) = 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } } \end{array}$ . The event A ∩ B occurs only when one sample satisfies all sub-questions while the other does not. Therefore, ${ \mathbb P } ( A \cap B ) = 2 q _ { K } ( 1 - q _ { K } )$ , and thus $\begin{array} { r } { \bar { \mathrm { A c c } } _ { \mathrm { r a n k } } ( r _ { d } ) = \frac { 2 q _ { K } ( \bar { 1 - q } _ { K } ) } { 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } } \le 1 } \end{array}$ . A complete proof is provided in Appendix A. Moreover, as long as there exist two imperfect samples with diferent true quality levels, i.e., there exist $0 \leq k _ { 1 } \neq k _ { 2 } < K$ , we have $\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) \dot { < } 1$ 1.Therefore, $\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) = 1 \geq \mathrm { A } \overline { { \operatorname { c c } _ { \mathrm { r a n k } } } } ( r _ { d } )$

Therefore, graded rewards preserve the true quality ordering more faithfully by assigning larger advantage weights to samples that satisfy more sub-questions, thereby better aligning optimization with improvements in true sample quality. By contrast, discrete rewards collapse partially correct samples into the same value, failing to distinguish imperfect outputs of difering true visual quality and potentially destabilizing generative policy optimization, especially under complex compositional requirements.

## 3.2 Graded Verification Enhances Multimodal Reasoning and Critique Capabilities

Drawbacks of Binary Outcome Supervision. Binary verifiers supervise model training solely through the final True/False outcome, implicitly assuming that a correct verdict reflects a correct reasoning process. This assumption is often reasonable in domains such as mathematical problem solving, where the solution space is large and arriving at the correct answer typically requires coherent multi-step reasoning. However, it becomes much more fragile in binary multimodal critique tasks. Without explicit supervision over the judgment process, a verifier may exploit shallow visual patterns or shortcut cues to produce seemingly correct verdicts without genuinely understanding the underlying multimodal reasoning process (Wang et al., 2026c; Shao et al., 2025), thereby leading to reward hacking

Experiment Setup. We collect textual prompts from ShareGPT-4o-Image (Chen et al., 2025) and control prompt length to construct samples with varying levels of complexity. To enhance visual diversity, we randomly sample from multiple generative models, including FLUX.1-dev (Labs et al., 2025), Qwen-Image (Wu et al., 2025), and Z-Image (Team, 2025b), to generate the corresponding images. We then perform supervised fine-tuning (SFT) to learn the sub-question decomposition paradigm distilled from Seed1.8 (Seed, 2026). Using the SFT-trained model, we filter out samples whose predictions exactly match the ground truth, and further apply a GPT-5 (OpenAI, 2025a) Bo10 filtering strategy to remove samples with accuracy below 60% while controlling the proportion of fully correct samples at 30%. The remaining high-quality samples are used for Decoupled Reinforcement Learning with DAPO (Yu et al., 2025) on Qwen3-VL-8B (Team, 2025a). The model is trained for 80 steps on 16 NVIDIA A800-80G GPUs and evaluated on ViVerBench (Zhang et al., 2025b).

Decoupled Reinforcement Learning. As shown in Fig. 2, given a textual prompt c and its corresponding generated image y, we convert each pair into K + 1 reinforcement learning samples, where K denotes the number of decomposed sub-questions. Specifically, one sample is used for decomposition-based continuous scoring policy optimization, while the remaining K samples are used for binary fine-grained judgment policy optimization. For the decomposition-based scoring sample, the verifier takes the textual prompt c and the generated image y as input, and is trained to predict the global alignment score induced by the decomposition. Given the predicted score $\hat { S } \in [ 0 , 1 ]$ and the ground-truth score $S ^ { * } \in [ 0 , 1 ]$ , the reward is defined as $\mathcal { R } _ { \mathrm { s c o r e } } = 1 - | \hat { S } - S ^ { * } |$ . For each fine-grained judgment sample, the verifier takes the k-th sub-question $Q _ { k }$ and the generated image y as input, and is trained to predict the binary answer associated with this sub-question. Given the predicted answer $\hat { A } _ { k } \in \{ 0 , 1 \}$ and the ground-truth answer $A _ { k } ^ { * } \in \{ 0 , 1 \}$ the reward is defined as $\mathcal { R } _ { \mathrm { j u d g m e n t } } = \mathbb { 1 } [ \hat { A } _ { k } = A _ { k } ^ { * } ]$ . During reinforcement learning, we mix these two types of samples and optimize them with their respective rewards, forming a decoupled RL training scheme that jointly optimizes sub-question-level binary judgment and global alignment scoring. Precise 0/1 feedback on individual sub-questions enables the verifier to develop fine-grained verification capabilities that provide a foundation for reliable continuous scoring.

Experiment Analysis. Table 1 shows that the Decomposed Graded Verifier achieves stronger generalization than the Discrete Binary Verifier on ViVerBench, a comprehensive and challenging benchmark for evaluating multimodal reasoning and critique. Notably, the Decomposed Graded Verifier requires supervised fine-tuning (SFT) to learn the structured decomposition paradigm. By contrast, the Discrete Binary Verifier only needs to produce a binary True/False verdict with a corresponding textual explanation, and can therefore be directly optimized through reinforcement learning. For a fair comparison, we evaluate the Discrete Binary Verifier in both RL-only and SFT+RL settings. The results show that the Discrete Binary Verifier consistently underperforms the Decomposed Graded Verifier, regardless of whether it is first adapted with SFT. This indicates that the key factor is not the training pipeline itself, but the nature of the supervision signal. Binary supervision is highly compressed and sparse, which encourages the model to rely on shallow patterns rather than capturing complex semantic structures. In contrast, the Decomposed Graded Verifier decomposes complex tasks into multiple sub-questions and performs finegrained judgments, thereby explicitly enforcing a structured reasoning process. This mechanism provides richer and more informative feedback signals, efectively mitigating reward hacking and substantially improving the model’s multimodal reasoning and critique capabilities. Moreover, our Decoupled RL training strategy jointly optimizes sub-question-level binary judgment and global alignment scoring, using precise 0/1 feedback on individual sub-questions to strengthen fine-grained verification capabilities and thereby provide a foundation for robust generalization to out-of-distribution (OOD) scenarios.

## 4 Verifier-Guided Reinforcement Learning for Generative Models

In this section, we investigate how the Decomposed Graded Verifier guides generative model optimization through reinforcement learning. We begin with traditional text-to-image generation task, and then focus on studying how to adapt the verifier to provide more reliable reward signals for rule-governed world modeling scenarios.

## 4.1 Verifier-Guided RL for Traditional Text-to-Image Generation

Given a textual prompt c and its generated image y, the Decomposed Graded Verifier first decomposes c into a set of fine-grained sub-questions $Q = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { K } \}$ and then produces a continuous score by aggregating their verification results: $\begin{array} { r } { S = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { I } ( q _ { k } , y ) } \end{array}$ , where $\mathcal { I } ( q _ { k } , y ) \in \{ 0 , 1 \}$ denotes a binary judgment function that evaluates whether the generated image y satisfies sub-question $q _ { k }$ . This score is used as the reward signal for reinforcement learning to guide policy optimization.

## 4.2 Verifier-Guided RL for Rule-Governed World Modeling

Traditional text-to-image generation primarily targets semantic alignment, where a prompt may admit multiple diverse and visually plausible outputs, leaving no well-defined unique ground-truth image. By contrast, world simulation and reasoning tasks are rule-governed and highly structured, where the desired final state is often uniquely determined by the initial state and the underlying rules, thereby making a ground-truth image available in many scenarios. Motivated by this property, we explicitly introduce the ground-truth image as an additional visual input to the verifier, which simplifies the verification process and enables the verifier to produce more accurate and reliable reward signals for rule-governed generative world modeling.

FLUX1.dev  
![](images/362cb2f97f69da144f52726ca3941e978575ceff5dd9594981725a1ef8efac0a.jpg)

RL（Binary Verifier）  
![](images/36a3a5767f4cb5ecb5e7b6791d7e34a63b7bb68bba818dba24c4dcaa87341e45.jpg)

RL (Graded Verifier)  
![](images/1a0c3d44f629c1e00d06060ece5c28c2d4b86571d3e113b5a5ee1b68d4af5f24.jpg)

FLUX1.dev  
![](images/beb009406570c143b121c0f1c0637e690801398c9f6fa058af30ada8cc7cb189.jpg)

RL（Binary Verifier）  
![](images/23acf19e5bac3364410ebb61b3b0e38dd0e473d3275e0eb71547392b9a522cd3.jpg)

RL (Graded Verifier)  
![](images/3ef0b5a83367ecdaba3a1e9ffc86260899dd1471e4db8a8e15fac2b9b51f9e5d.jpg)

Five people playing with a red frisbee in a sunny park: two in yellow short-sleeved shirts, one in green, one in black, and one in blue.  
![](images/9a58961cc08498738e58aa4d4f5c4dd448a6b321932103f254a6eba8c00f6120.jpg)

Four men in a kitchen: one squeezing a lemon, one opening the refrigerator, one opening a cabinet, and one slicing bread.  
![](images/e9b83270210a5fb59cf8b76ff48d015dcabdcd19d6cfb501879a2a997b367a17.jpg)

![](images/a44759efe9f70a278add5b7bb1d952757731b0284a1ea22d9620f2435ed941f2.jpg)  
Stone temples line both sides of the road in the foreground, followed by traditional old houses, with high-rise buildings in the distance.

![](images/e55379cd69a254ad75c13f68adfd922678c5818a5608c929003b82cbea23c7b3.jpg)

![](images/4d1610d1d75c949c8e53b2c6e8e0cc49f3180a1b324862613cf6c2de73d618e3.jpg)

![](images/e76d867e21fbb10eb7d422a71687557cf95cab1e01a83917189aa79ad0127411.jpg)

![](images/565ac42c7c51ada676992ea5124a55fbc07b2b3bd833e3c28518417b5791321c.jpg)

![](images/ab6fba5dabe8b891712166be69440ec0fd1f705c38d704b23380a0b751694401.jpg)  
At dawn, a snow-covered valley is dotted in the distance with many brightly lit tents.

A steam train emitting thick white steam crosses a bridge, with a clear shadow cast onto the grass below.  
![](images/7e8dffee45093788141a56f320e2e137aed8eb7b145a0a26232aba0bdf73abda.jpg)

![](images/0c384b31a2b3a4247d65c987cf05b54fd04d061045696512a06bc27993a26de8.jpg)

![](images/66798b60b6c34b71c01c20646a37167aac80e9b3b4d847caa823333f1cd081cd.jpg)

![](images/9bf590f93d8fb57c3e34b58a3514694d8c623beccd62ee8454bb425582b732fb.jpg)  
Five plates are arranged on a wooden table, each decorated with three deer motifs.  
Figure 3: Qualitative visualization of traditional text-to-image generation.

Formally, given an initial visual state x and a rule-based instruction $^ { c , }$ the generative model $\pi _ { \theta }$ is required to understand the underlying rules and perform rule-governed reasoning to produce a resulting image $y \sim \pi _ { \boldsymbol { \theta } } ( y \mid x , c )$ , which can be viewed as a visual prediction of the target final state. Similar to the traditional text-to-image setting, the instruction c can be decomposed into $\bar { K }$ fine-grained sub-questions $Q = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { K } \}$ , each assessing whether the resulting image y satisfies a specific sub-rule or achieves a corresponding sub-goal.

Definition1: Pointwise Verification. As shown in Fig. 2, the pointwise verifier takes $( x , c , y )$ as input. For each sub-question $q _ { k }$ , it must first infer the corresponding target state from the initial visual state x and the underlying rules $^ { c , }$ denoted as $z _ { k } = \mathcal { T } ( q _ { k } , x , c )$ . The verifier then assesses whether the resulting image y is consistent with this internally inferred target state $z _ { k } .$ Accordingly, the pointwise verification score is defined as $\begin{array} { r } { S _ { \mathrm { p o i n t } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { I } ( q _ { k } , y , z _ { k } ) } \end{array}$ , and its deviation from the true score consists of both reasoning error and judgment error: $\epsilon _ { \mathrm { p o i n t } } = \epsilon _ { \mathrm { r e a s o n i n g } } + \epsilon _ { \mathrm { j u d g e } } .$

Definition2: Pairwise Verification. In contrast, pairwise verification explicitly incorporates the groundtruth image $y ^ { * }$ as an additional visual input, with the verifier taking $( x , c , y , y ^ { * } )$ as input. For each sub-question $q _ { k }$ , the verifier compares the generated image y against the ground-truth image $\bar { y } ^ { * }$ and assigns a corresponding score, hence the name pairwise verification. By avoiding internal target-state inference, the pairwise verification score is defined as $\begin{array} { r } { S _ { \mathrm { p a i r } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { I } ( q _ { k } , y , y ^ { * } ) } \end{array}$ , and its deviation from the true score is reduced to the judgment error: $\epsilon _ { \mathrm { p a i r } } = \epsilon _ { \mathrm { j u d g e } } .$

By simplifying the reason-then-judge process into direct reference-based comparison and thereby reducing the reasoning errors associated with target-state inference, pairwise verification provides a substantially more reliable reward mechanism than pointwise verification in rule-governed world-modeling scenarios. This property is crucial for post-training generative models, where the efectiveness of reinforcement learning hinges on the accuracy and stability of the reward signal.

Table 2: Performance on GenEval and CLIPScore under diferent reward models.
<table><tr><td rowspan="2">Methods</td><td colspan="7">GenEval</td><td rowspan="2">CLIPScore</td></tr><tr><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attri. Bind.</td><td>Overall</td></tr><tr><td>Stable-Diffusion-3.5-medium</td><td>0.99</td><td>0.78</td><td>0.69</td><td>0.79</td><td>0.25</td><td>0.45</td><td>0.66</td><td>32.8</td></tr><tr><td>+RL(UnifiedReward)</td><td>0.99</td><td>0.84</td><td>0.75</td><td>0.86</td><td>0.36</td><td>0.55</td><td>0.73</td><td>33.7</td></tr><tr><td>+RL(Discrete Binary Verifier)</td><td>0.99</td><td>0.80</td><td>0.72</td><td>0.81</td><td>0.29</td><td>0.47</td><td>0.68</td><td>33.1</td></tr><tr><td>+RL(Decomposed Graded Verifier)</td><td>1.00</td><td>0.90</td><td>0.79</td><td>0.89</td><td>0.41</td><td>0.63</td><td>0.77</td><td>34.4</td></tr><tr><td>FLUX.1-dev</td><td>0.99</td><td>0.85</td><td>0.61</td><td>0.84</td><td>0.22</td><td>0.58</td><td>0.68</td><td>33.2</td></tr><tr><td>+RL(UnifiedReward)</td><td>1.00</td><td>0.90</td><td>0.65</td><td>0.89</td><td>0.33</td><td>0.64</td><td>0.74</td><td>34.0</td></tr><tr><td>+RL(Discrete Binary Verifier)</td><td>1.00</td><td>0.88</td><td>0.63</td><td>0.85</td><td>0.25</td><td>0.59</td><td>0.70</td><td>33.3</td></tr><tr><td>+RL(Decomposed Graded Verifier)</td><td>1.00</td><td>0.93</td><td>0.77</td><td>0.90</td><td>0.43</td><td>0.71</td><td>0.79</td><td>34.5</td></tr></table>

## 5 Experiments

In this section, we conduct comprehensive experiments to demonstrate the advantages of the Decomposed Graded Verifier as a generative reward model and the superiority of the pairwise verification mechanism in rule-governed world modeling scenarios.

## 5.1 Experiment Setup

Training Datasets. For traditional text-to-image tasks, we use textual prompts from ShareGPT-4o-Image (Chen et al., 2025), the same data source used for verifier training. For rule-governed world modeling scenarios, we construct six game-based tasks to comprehensively evaluate generative models’ abilities in path planning, logical reasoning, geometric construction, pattern induction, character recognition, and spatial reasoning .

• Maze & Maze-Hard. Find a valid path from the red start circle to the blue end cross without touching or crossing any black walls. Maze-Hard contains larger-scale maze instances.

• Sudoku & Sudoku-Hard. Fill the blanks in a 4 × 4 Sudoku grid so that each row, column, and 2 × 2 subgrid contains {1, 2, 3, 4} exactly once. Sudoku-Hard contains more blanks.

• Auxiliary Line Construction & Auxiliary Line Construction-Hard. Draw auxiliary lines in a geometric figure. The hard setting requires multiple compositional constructions.

• Sequence Prediction & Sequence Prediction-Hard. Predict the next geometric figure from shape or color patterns. The hard setting requires simultaneous color-shape changes.

• Word Search & Word Search-Hard. Locate the target word in an N × N grid, where each row contains one character in the correct order. The hard setting corresponds to a larger N.

• 3D Projection & 3D Projection-Hard. Generate the specified 2D orthographic view of a 3D structure, such as the top, front, or side view. The hard setting uses larger 3D structures.

Training Details. For traditional text-to-image tasks, we adopt Stable-Difusion-3.5-medium and FLUX.1- dev (Labs et al., 2025) as the backbone generative models. For rule-governed world-modeling scenarios, we use Qwen-Image-Edit-2509 (Wu et al., 2025) as the backbone. All models are trained for 200 steps using the DifusionNFT algorithm (Zheng et al., 2025) on 16 NVIDIA A800-80G GPUs.

## 5.2 Text-to-Image Generation

Graded Rewards Are Better than Binary Rewards. Table 2 compares the Decomposed Graded Verifier with the Discrete Binary Verifier as reinforcement learning reward models across diferent generative

Sequence Prediction-Hard: Joint Color and Shape Reasoning  
![](images/5bc1b3df8f75e847105922226847684cffe254c8a17cd7dac179bab32263c30e.jpg)

![](images/e7ea190dc7ccfdab271e1ff47c67a44880bca88950c61fe85dab5e1a17105d79.jpg)  
Word Search-Hard: Larger Scale

![](images/5188859180453cb9aa66bdfad584ab5f845212e2738624bbf51352070784fb31.jpg)  
Sudoku-Hard: More Blank Cells

![](images/2fa8b871b5a28261cccdc77bc30e5495583cdc700ad3c218506e86181ff4660d.jpg)  
Auxiliary Line Construction-Hard: Compositional Task

![](images/1eb40b377de9f5e5725cf930ca73025dd7db76f812ba16ab2438199f9533e9fc.jpg)  
3D Projection-Hard: Larger Scale

![](images/80d2032d614a872bebb1d48f8e59ff11a338a640e6aa6da52e44d252e9cb9560.jpg)  
Maze: Different Visual Style

![](images/6f695ccaafadb25180112b421306629bed18c4e3c671115f7a3478cb7ab484cc.jpg)  
Sequence Prediction: From Geometry Patterns to Numerical Patterns

Figure 4: Qualitative visualization of rule-governed world-modeling tasks.

backbones, including Stable-Difusion-3.5-Medium and FLUX.1-dev. The Decomposed Graded Verifier consistently achieves superior performance, demonstrating that fine-grained graded reward signals are more efective for improving the semantic alignment capability of generative models. Fig. 3 further supports this conclusion through qualitative visualizations. Fig. 1 (a)(b) shows the training dynamics under the two reward models. We evaluate intermediate checkpoints every 10 steps on a held-out set of 500 textual prompts under the same decomposition-based scoring protocol, using Gemini3.1-Pro (Google DeepMind, 2026) as an evaluator not involved in data processing. Compared with the Discrete Binary Verifier, training with the Decomposed Graded Verifier yields steadier and more pronounced performance improvements, suggesting that denser and more discriminative rewards support more stable and efective optimization.

Comparison with Other Continuous Reward Models. We compare the Decomposed Graded Verifier with UnifiedReward (Wang et al., 2025b), which also focuses on semantic alignment and produces continuous reward scores. As shown in Table 2, our method achieves superior performance on both GenEval (Ghosh et al., 2023) and CLIPScore (Hessel et al., 2021), supporting the ability of the Decomposed Graded Verifier trained with our Decoupled RL strategy to provide more accurate and fine-grained reward signals and highlighting its advantages as a generative reward model.

## 5.3 Rule-Governed Generative World Modeling

Advantages of Pairwise Verification over Pointwise Verification. As shown in Tables 3 and 4, pairwise verification consistently outperforms pointwise verification on RISEBench (Zhao et al., 2025), UniREditBench (Han et al., 2025), and our constructed game-based benchmark. These results indicate that incorporating the ground-truth image as an explicit reference significantly enhances the accuracy and reliability of reward signals. The training dynamics in Fig. 1(e) summarize average performance across our six constructed game-based tasks and further highlight the optimization advantages of pairwise verification over pointwise verification. We evaluate intermediate checkpoints every 10 training steps on the held-out validation sets. To improve evaluation reliability, we use Gemini3.1-Pro (Google DeepMind, 2026)

Table 3: Performance comparison on game-based image-to-image tasks for world modeling.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Maze</td><td colspan="2">Sudoku</td><td colspan="2">Aux. Line Construction</td><td colspan="2">Sequence Prediction</td><td colspan="2">Word Search</td><td colspan="2">3D Projection</td></tr><tr><td>Normal</td><td>Hard</td><td>Normal</td><td>Hard</td><td>Normal</td><td>Hard</td><td>Normal</td><td>Hard</td><td>Normal</td><td>Hard</td><td>Normal</td><td>Hard</td></tr><tr><td>Qwen-Image-Edit-2509</td><td>17.31</td><td>8.73</td><td>22.21</td><td>5.62</td><td>16.79</td><td>7.32</td><td>9.18</td><td>4.33</td><td>6.28</td><td>2.46</td><td>3.86</td><td>1.71</td></tr><tr><td>+ SFT</td><td>56.77</td><td>18.96</td><td>63.62</td><td>29.99</td><td>57.62</td><td>26.84</td><td>46.53</td><td>27.96</td><td>37.36</td><td>17.29</td><td>42.94</td><td>23.37</td></tr><tr><td>+SFT+RL(Pointwise)</td><td>67.81</td><td>34.78</td><td>77.74</td><td>50.71</td><td>66.98</td><td>37.49</td><td>61.27</td><td>34.18</td><td>50.92</td><td>26.94</td><td>52.19</td><td>33.16</td></tr><tr><td>+SFT+RL(Pairwise)</td><td>85.31</td><td>51.92</td><td>89.97</td><td>62.99</td><td>76.19</td><td>46.12</td><td>73.48</td><td>63.64</td><td>62.18</td><td>34.27</td><td>69.28</td><td>47.51</td></tr></table>

Table 4: Performance comparison on RISEBench and UniREditBench.
<table><tr><td rowspan="2">Methods</td><td colspan="5">RISEBench</td><td colspan="5">UniREdit-Real</td><td colspan="5">UniREdit-Game</td><td rowspan="2">Overall</td></tr><tr><td>T</td><td>C</td><td>S</td><td>L</td><td>Overall</td><td>AM</td><td>ST</td><td>PI</td><td>PR</td><td>Avg.</td><td>SI</td><td>SR</td><td>LHP</td><td>LPS</td><td>Avg.</td></tr><tr><td>Qwen-Image-Edit-2509</td><td>2.4</td><td>7.8</td><td>14.0</td><td>1.2</td><td>6.7</td><td>70.04</td><td>68.49</td><td>67.26</td><td>64.11</td><td>67.48</td><td>52.31</td><td>32.96</td><td>42.46</td><td>37.89</td><td>41.41</td><td>53.16</td></tr><tr><td>+ SFT</td><td>3.5</td><td>7.8</td><td>16.0</td><td>5.9</td><td>8.6</td><td>70.56</td><td>68.68</td><td>67.55</td><td>64.98</td><td>67.94</td><td>66.92</td><td>47.48</td><td>56.06</td><td>43.98</td><td>53.61</td><td>59.14</td></tr><tr><td>+ SFT + RL(Pointwise)</td><td>5.9</td><td>8.9</td><td>18.0</td><td>12.9</td><td>11.7</td><td>71.24</td><td>68.73</td><td>68.38</td><td>65.77</td><td>68.53</td><td>68.73</td><td>49.16</td><td>60.32</td><td>56.41</td><td>58.66</td><td>62.37</td></tr><tr><td>+ SFT + RL(Pairwise)</td><td>7.1</td><td>10.0</td><td>19.0</td><td>23.5</td><td>15.0</td><td>71.87</td><td>69.02</td><td>68.72</td><td>66.08</td><td>68.92</td><td>70.23</td><td>51.33</td><td>68.93</td><td>63.53</td><td>63.51</td><td>65.15</td></tr></table>

as an evaluator not involved in data processing and explicitly provide the ground-truth image following the pairwise verification protocol.

The pairwise-trained model exhibits steadier and more pronounced performance improvements, whereas the pointwise-trained model shows clear performance fluctuations. Pointwise rewards require the verifier to infer an implicit target state, which can introduce substantial noise into the feedback in complex reasoning scenarios. In contrast, pairwise verification directly compares the generated result with the ground-truth image, reducing reliance on target-state inference and enabling reward signals to better reflect the true visual quality. This provides more reliable supervision for policy optimization and highlights the value of pairwise verification for developing generative models into rule-governed world-modeling agents.

From Easy-to-Hard and Domain Generalization. To more comprehensively evaluate the generalization potential of the model trained with pairwise verification, we construct a more challenging Hard Dataset that is excluded from training. This dataset contains larger-scale instances and compositional tasks designed to test reasoning in more complex structured environments. As shown in Table 3 and Fig. 4, the model exhibits strong performance across out-of-distribution scenarios, including shifts in the visual domain. These results suggest that the model acquires transferable capabilities for structured reasoning and rule-governed world modeling beyond task-specific training patterns, highlighting the advantages of pairwise verification in supporting robust generalization.

SFT Provides Domain-Specific Execution Priors. Tables 3 and 4 show that SFT yields substantial performance gains, which can be primarily attributed to task-domain adaptation. Since Qwen-Image-Edit-2509 has not been specifically trained on game-based image-to-image generation tasks, it often fails to interpret the required transformation. In practice, the model tends to either the model tends to either leave the input image unchanged or distort its original structure, indicating that it lacks the task-specific action priors required for this domain. After supervised fine-tuning on our constructed game-based dataset, the generative model learns the visual style of these tasks and the corresponding rule-based transformation patterns. Although the model still often fails to produce fully correct solutions, it begins to perform meaningful operations toward the desired final state rather than generating irrelevant or destructive visual outputs, leading to clear performance improvements. These results indicate that SFT primarily equips the model with domain-specific execution ability, while subsequent reinforcement learning further endows the generative model with reasoning intelligence.

Generative Intelligence Beyond the Text Encoder. In both multimodal understanding and generation models, the text encoder is often regarded as the primary source of reasoning intelligence. For generative models, this view implies that the text encoder interprets instructions and provides semantic guidance, while the DiT primarily executes the visual generation process. However, in our training pipeline, the text encoder remains frozen throughout both SFT and RL, and only the DiT parameters are updated. The reasoning capabilities observed after training suggest that reasoning intelligence is not encoded exclusively in the text encoder, as the DiT can also learn, internalize, and express capabilities for structured reasoning and rule-governed world modeling.

## 6 Conclusion

This work introduces the Decomposed Graded Verifier as an efective reward model for improving visual generation in both traditional text-to-image tasks and structured world-modeling scenarios. Our contributions are twofold. First, we propose a Decoupled RL training strategy that jointly optimizes subquestion-level binary judgment and global alignment scoring, using precise local feedback to develop fine-grained verification capabilities that support accurate continuous scoring. Empirical results further demonstrate improved out-of-domain multimodal critic capabilities. Second, focusing on rule-governed world modeling, we introduce a pairwise verification paradigm that explicitly incorporates ground-truth images as visual references. This simplifies verification by shifting from a reason-then-judge process to direct reference-based comparison, thereby producing more accurate and reliable reward signals. Extensive experiments validate the efectiveness of both the Decomposed Graded Verifier trained with our Decoupled RL strategy and the pairwise verification paradigm.

## References

Zefan Cai, Haoyi Qiu, Tianyi Ma, Haozhe Zhao, Gengze Zhou, Kung-Hsiang Huang, Parisa Kordjamshidi, Minjia Zhang, Wen Xiao, Jiuxiang Gu, et al. Mmgr: Multi-modal generative reasoning. arXiv preprint arXiv:2512.14691, 2025.

Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation, 2025. URL https://arxiv.org/abs/2506.18095.

Zisheng Chen, Chunwei Wang, Runhui Huang, Hongbin Xu, Xiuwei Chen, Jun Zhou, Jianhua Han, Hang Xu, and Xiaodan Liang. Semhitok: A unified image tokenizer via semantic-guided hierarchical codebook for multimodal understanding and generation. In International Conference on Learning Representations, volume 2026, pp. 44597–44620, 2026.

Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Google. Veo 3 model card. https://storage.googleapis.com/deepmind-media/Model-Cards/ Veo-3-Model-Card.pdf, 2024.

Google. Introducing nano banana pro. https://blog.google/technology/ai/nano-banana-pro/, 2025.

Google DeepMind. Gemini 3.1 Pro model card. https://deepmind.google/models/model-cards/ gemini-3-1-pro/, 2026.

Feng Han, Yibin Wang, Chenglin Li, Zheming Liang, Dianyi Wang, Yang Jiao, Zhipeng Wei, Chao Gong, Cheng Jin, Jingjing Chen, et al. Unireditbench: A unified reasoning-based image editing benchmark. arXiv preprint arXiv:2511.01295, 2025.

Haonan Han, Jiancheng Huang, Xiaopeng Sun, Junyan He, Rui Yang, Jie Hu, Xiaojiang Peng, Lin Ma, Xiaoming Wei, and Xiu Li. Vigor-bench: How far are visual generative models from zero-shot visual reasoners? arXiv preprint arXiv:2603.25823, 2026.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings ofthe 2021 conference on empirical methods in natural language processing, pp. 7514–7528, 2021.

Jiehui Huang, Yuechen Zhang, Xu He, Yuan Gao, Zhi Cen, Bin Xia, Yan Zhou, Xin Tao, Pengfei Wan, and Jiaya Jia. Unityvideo: Unified multi-modal multi-task learning for enhancing world-aware video generation. arXiv preprint arXiv:2512.07831, 2025.

Jiehui Huang, Xiao Dong, Wenhui Song, Zheng Chong, Zhenchao Tang, Jun Zhou, Yuhao Cheng, Long Chen, Hanhui Li, Yiqiang Yan, et al. Consistentid: Portrait generation with multimodal fine-grained identity preserving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

Jing Jin, Hao Liu, Yan Bai, Yihang Lou, Zhenke Wang, Tianrun Yuan, Juntong Chen, Yongkang Zhu, Fanhu Zeng, Xuanyu Zhu, Tao Feng, and Yige Xu. Unveiling fine-grained visual traces: Evaluating multimodal interleaved reasoning chains in multimodal stem tasks, 2026. URL https://arxiv.org/abs/2604. 19697.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Ouxiang Li, Yuan Wang, Xinting Hu, Huijuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, Xiaojuan Qi, and Fuli Feng. Easier painting than thinking: Can text-to-image models set the stage, but not direct the play? arXiv preprint arXiv:2509.03516, 2025.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. Rm-bench: Benchmarking reward models of language models with subtlety and style. arXiv preprint arXiv:2410.16184, 2024.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. Generative reward models. arXiv preprint arXiv:2410.12832, 2024.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A Smith, Hannaneh Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937, 2025.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

OpenAI. Introducing gpt-5. https://openai.com/index/introducing-gpt-5/, 2025a.

OpenAI. Sora 2 system card. https://openai.com/index/sora-2-system-card/, 2025b.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Bytedance Seed. Seed1. 8 model card: Towards generalized real-world agency. arXiv preprint arXiv:2603.20633, 2026.

Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

Zhihong Shao, Yuxiang Luo, Chengda Lu, ZZ Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. arXiv preprint arXiv:2511.22570, 2025.

Kaiyue Sun, Rongyao Fang, Chengqi Duan, Xian Liu, and Xihui Liu. T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472, 2025.

Boming Tan, Xiangdong Zhang, Ning Liao, Yuqing Zhang, Shaofeng Zhang, Xue Yang, Qi Fan, and Yanyong Zhang. Dreamworld: Unified world modeling in video generation. arXiv preprint arXiv:2603.00466, 2026.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. Judgebench: A benchmark for evaluating llm-based judges. arXiv preprint arXiv:2410.12784, 2024.

Qwen Team. Qwen3 technical report, 2025a. URL https://arxiv.org/abs/2505.09388.

Z-Image Team. Z-image: An eficient image generation foundation model with single-stream difusion transformer. arXiv preprint arXiv:2511.22699, 2025b.

Xinyu Wang, Huafeng Shi, Zian Li, Yan Zhou, Xiaoqiang Liu, Yue Ma, and Pengfei Wan. Subjectanchor: Subject-aware memory-to-video for multi-shot storytelling. arXiv preprint arXiv:2609.34502, 2026a.

Xinyu Wang, Chongbo Zhao, Fangneng Zhan, and Yue Ma. Liveedit: Towards real-time difusion-based streaming video editing. In European Conference on Computer Vision, pp. 631–649. Springer, 2026b.

Xiyao Wang, Chunyuan Li, Jianwei Yang, Kai Zhang, Bo Liu, Tianyi Xiong, and Furong Huang. Llava-criticr1: Your critic model is secretly a strong policy model. arXiv preprint arXiv:2509.00676, 2025a.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025b.

Zongqi Wang, Rui Wang, Yuchuan Wu, Yiyao Yu, Pinyi Zhang, Shaoning Sun, Yujiu Yang, and Yongbin Li. Reward modeling from natural language human feedback. arXiv preprint arXiv:2601.07349, 2026c.

Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 13618–13628, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large difusion language models. arXiv preprint arXiv:2505.15809, 2025a.

Ling Yang, Xinchen Zhang, Ye Tian, Chenming Shang, Minghao Xu, Wentao Zhang, and Bin Cui. Hermesflow: Seamlessly closing the gap in multimodal understanding and generation. arXiv preprint arXiv:2502.12148, 2025b.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024a.

Ruicheng Zhang, Mingyang Zhang, Jun Zhou, Xiaofan Liu, Zunnan Xu, Zhizhou Zhong, Puxin Yan, Haocheng Luo, and Xiu Li. Mind-v: Hierarchical world model for long-horizon robotic manipulation with rl-based physical alignment. arXiv preprint arXiv:2512.06628, 2025a.

Ruicheng Zhang, Guangyu Chen, Zunnan Xu, Zihao Liu, Zhizhou Zhong, Mingyang Zhang, Jun Zhou, and Xiu Li. Robostereo: Dual-tower 4d embodied world models for unified policy optimization. arXiv preprint arXiv:2603.12639, 2026a.

Ruicheng Zhang, Kaixi Cong, Jun Zhou, Zhizhou Zhong, Zunnan Xu, Shuiyang Mao, Wei Liu, and Xiu Li. Kvpo: Ode-native grpo for autoregressive video alignment via kv semantic exploration. arXiv preprint arXiv:2605.14278, 2026b.

Ruicheng Zhang, Jun Zhou, Zunnan Xu, Zihao Liu, Jiehui Huang, Mingyang Zhang, Yu Sun, and Xiu Li. Zo3t: Zero-shot 3d-aware trajectory-guided image-to-video generation via test-time training. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 12708–12716, 2026c.

Xinchen Zhang, Ling Yang, Yaqi Cai, Zhaochen Yu, Kai-Ni Wang, Jiake Xie, Ye Tian, Minkai Xu, Yong Tang, Yujiu Yang, et al. Realcompo: Balancing realism and compositionality improves text-to-image difusion models. Advances in Neural Information Processing Systems, 37:96963–96992, 2024b.

Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Yong Tang, Yujiu Yang, Mengdi Wang, Bin CUI, et al. Itercomp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. In The Thirteenth International Conference on Learning Representations, 2024c.

Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, and Yujiu Yang. Generative universal verifier as multimodal meta-reasoner. arXiv preprint arXiv:2510.13804, 2025b.

Xinchen Zhang, Bowei Liu, Jiale Liu, Chufan Shi, Yizhen Zhang, Junhong Liu, Youliang Zhang, Zhiheng Li, Yujiu Yang, and Ling Yang. Omniverifier-m1: Multimodal meta-verifier with explicit structured recalibration. arXiv preprint arXiv:2605.28805, 2026d.

Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Difusionnft: Online difusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Jun Zhou, Jiahao Li, Zunnan Xu, Hanhui Li, Yiji Cheng, Fa-Ting Hong, Qin Lin, Qinglin Lu, and Xiaodan Liang. Fireedit: Fine-grained instruction-based image editing via region-aware vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13093–13103, 2025.

□

## A Proof of Ranking Preservation

In this section we prove that Decomposed Graded Verification preserves the true quality ordering of generated samples more faithfully than Discrete Binary Verification. The argument rests on a single observation: under GRPO, the relative update strength among samples is dictated by their group-normalized advantages, whose ordering is fully inherited from the underlying reward values. It therefore sufices to compare how the two reward designs rank pairs of samples with distinct latent quality.

Setup. Let c be a prompt and x $\sim \pi _ { \boldsymbol { \theta } } ( \cdot \vert c )$ a sample drawn from the current policy. Suppose c admits a lossless decomposition into K sub-questions, and let

$$
z ( x ) \in \{ 0 , 1 , \ldots , K \}\tag{2}
$$

denote the number of sub-questions satisfied by x. We treat $z ( x )$ as the latent fine-grained quality level of x and write

$$
q _ { k } : = \mathbb { P } ( z ( x ) = k ) , \qquad k = 0 , \ldots , K ,\tag{3}
$$

where the probability is taken with respect to $x \sim \pi _ { \theta } ( \cdot \mid c )$

We compare two reward designs:

$$
r _ { c } ( x ) = \frac { z ( x ) } { K } , \qquad r _ { d } ( x ) = { \bf 1 } [ z ( x ) = K ] ,\tag{4}
$$

namely the decomposed graded reward $r _ { c }$ and the discrete binary reward $r _ { d }$

For two independent samples $x , x ^ { \prime } \sim \pi _ { \theta } ( \cdot \mid c )$ , the pairwise ranking accuracy of a reward function r is defined as

$$
\operatorname { A c c } _ { \mathrm { r a n k } } ( r ) : = \mathbb { P } \left( ( r ( x ) - r ( x ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) > 0 \mid z ( x ) \neq z ( x ^ { \prime } ) \right) .\tag{5}
$$

In words, conditional on the two samples carrying distinct true quality levels, $\operatorname { A c c } _ { \mathrm { r a n k } } ( r )$ is the probability that r assigns them a strictly correct relative ordering.

Remark on non-degeneracy. Throughout the proof we assume

$$
\mathbb { P } ( z ( x ) \neq z ( x ^ { \prime } ) ) > 0 ,\tag{6}
$$

i.e., the distribution of $z ( x )$ is not a point mass on a single quality level. When this fails, no pair of samples carries distinct quality levels and $\operatorname { E q . } \left( 5 \right)$ is undefined; the comparison is vacuous in this case and we exclude it without loss of generality.

Lemma A.1 (Advantage ordering is inherited from reward ordering). Consider a GRPO rollout group $\{ x _ { i } \} _ { i = 1 } ^ { n }$ with rewards $r _ { i } = r ( x _ { i } )$ , and let

$$
A _ { i } ( r ) = \frac { r _ { i } - { \bar { r } } } { \sigma _ { r } } , \qquad \bar { r } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } r _ { j } , \qquad \sigma _ { r } ^ { 2 } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( r _ { j } - { \bar { r } } ) ^ { 2 }\tag{7}
$$

denote the group-normalized advantage. $I f \sigma _ { r } > 0$ , then for any pair of samples $x _ { i } , x _ { j }$

$$
\operatorname { s i g n } \left( A _ { i } ( r ) - A _ { j } ( r ) \right) = \operatorname { s i g n } \left( r _ { i } - r _ { j } \right) .\tag{8}
$$

Consequently, the relative ordering of GRPO update weights coincides exactly with the relative ordering of rewards.

Proof. Subtracting the two normalized advantages cancels the common mean,

$$
A _ { i } ( r ) - A _ { j } ( r ) = \frac { ( r _ { i } - \bar { r } ) - ( r _ { j } - \bar { r } ) } { \sigma _ { r } } = \frac { r _ { i } - r _ { j } } { \sigma _ { r } } ,\tag{9}
$$

and dividing by the positive scalar $\sigma _ { r }$ preserves the sign. Therefore

$$
\operatorname { s i g n } \big ( A _ { i } ( r ) - A _ { j } ( r ) \big ) = \operatorname { s i g n } \big ( r _ { i } - r _ { j } \big ) ,\tag{10}
$$

as claimed.

Lemma A.2 (The graded reward perfectly preserves the quality ordering). For the decomposed graded reward

$$
r _ { c } ( x ) = \frac { z ( x ) } { K } ,\tag{11}
$$

the ranking accuracy satisfies

$$
\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) = 1 .\tag{12}
$$

Proof. Since $r _ { c }$ is a strictly increasing afine function of $z ,$ for any pair x, $x ^ { \prime }$ with $z ( x ) \neq z ( x ^ { \prime } )$ we have

$$
( r _ { c } ( x ) - r _ { c } ( x ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) = \frac { ( z ( x ) - z ( x ^ { \prime } ) ) ^ { 2 } } { K } > 0 ,\tag{13}
$$

where the strict inequality uses $K > 0$ and $z ( x ) \neq z ( x ^ { \prime } )$ . Hence the event $( r _ { c } ( x ) - r _ { c } ( x ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) > 0$ holds with probability one conditional on $z ( x ) \neq z ( x ^ { \prime } )$ , and $\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) = 1$

Lemma A.3 (The binary reward only ranks perfect-vs-imperfect pairs). For the discrete binary reward

$$
r _ { d } ( x ) = { \bf 1 } [ z ( x ) = K ] ,\tag{14}
$$

the event

$$
( r _ { d } ( x ) - r _ { d } ( x ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) > 0\tag{15}
$$

holds if and only if exactly one of the two samples attains the perfect quality level K.

Proof. Conditioning on $z ( x ) \neq z ( x ^ { \prime } )$ , there are exactly three disjoint cases:

(i) Exactly one sample is perfect. Without loss of generality, suppose $z ( x ) = K$ and $z ( x ^ { \prime } ) < K$ . Then $r _ { d } ( x ) - \dot { r } _ { d } ( x ^ { \prime } ) = \bar { 1 }$ and $\bar { z } ( \bar { x } ) - z ( x ^ { \prime } ) = K - z ( \bar { x } ^ { \prime } ) > 0 ,$ , so the product is strictly positive. The case $z ( x ^ { \prime } ) = K , z ( \overset { \cdot } { x } ) < K$ is symmetric.

(ii) Both samples are imperfect, i.e. $z ( x ) < K$ and $z ( x ^ { \prime } ) < K$ . Then $r _ { d } ( x ) = r _ { d } ( x ^ { \prime } ) = 0 , s o r _ { d } ( x ) - r _ { d } ( x ^ { \prime } ) = 0$ and the product vanishes, even though $z ( x ) \neq z { \dot { ( x ^ { \prime } ) } }$

(iii) Both samples are perfect. This case violates $z ( x ) \neq z ( x ^ { \prime } )$ and is excluded.

The product is strictly positive only in case (i), which proves the claim.

Theorem A.4 (Ranking advantage of decomposed graded verification). Assume the quality distribution $\{ q _ { k } \} _ { k = 0 } ^ { K }$ is non-degenerate, that $i s ,$

$$
1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } > 0 .\tag{16}
$$

Then

$$
\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) = 1 , \qquad \mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) = \frac { 2 q _ { K } ( 1 - q _ { K } ) } { 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } } .\tag{17}
$$

In particular,

$$
\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) \geq \mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) ,\tag{18}
$$

with strict inequality whenever there exist two distinct imperfect quality levels with non-zero probability, i.e., $q _ { a } q _ { b } > 0$ for some $0 \leq a < b < K$

Proof. The identity $\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) = 1$ is Lemma A.2. It remains to evaluate $\operatorname { A c c } _ { \operatorname { r a n k } } ( r _ { d } )$ and to compare the two accuracies.

Step 1: Closed form for $\operatorname { A c c } _ { \operatorname { r a n k } } ( r _ { d } )$ . Define the events

$$
A : = \left\{ ( r _ { d } ( x ) - r _ { d } ( x ^ { \prime } ) ) ( z ( x ) - z ( x ^ { \prime } ) ) > 0 \right\} , \qquad B : = \left\{ z ( x ) \neq z ( x ^ { \prime } ) \right\} ,\tag{19}
$$

so that $\operatorname { A c c } _ { \operatorname { r a n k } } ( r _ { d } ) = \mathbb { P } ( A \mid B ) = \mathbb { P } ( A \cap B ) / \mathbb { P } ( B )$

By independence of x and $x ^ { \prime } ,$

$$
\mathbb { P } ( B ) = 1 - \mathbb { P } ( z ( x ) = z ( x ^ { \prime } ) ) = 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } ,\tag{20}
$$

which is strictly positive by the non-degeneracy assumption. By Lemma $\mathrm { A . 3 } ,$ the event $A \cap B$ is exactly the event that exactly one of the two samples attains quality level K. Hence

$$
\begin{array} { r l } & { \mathbb { P } ( A \cap B ) = \mathbb { P } ( z ( x ) = K , z ( x ^ { \prime } ) < K ) + \mathbb { P } ( z ( x ) < K , z ( x ^ { \prime } ) = K ) } \\ & { \qquad = 2 q _ { K } ( 1 - q _ { K } ) . } \end{array}\tag{21}
$$

(22)

Dividing Eq. (22) by Eq. (20) gives the claimed formula for $\operatorname { A c c } _ { \operatorname { r a n k } } ( r _ { d } )$

Step 2: $\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) \leq 1$ . Expanding $\begin{array} { r } { ( \sum _ { k } q _ { k } ) ^ { 2 } = 1 } \end{array}$ yields the standard identity

$$
1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } = 2 \sum _ { 0 \leq a < b \leq K } q _ { a } q _ { b } ,\tag{23}
$$

and the same expansion gives

$$
2 q _ { K } ( 1 - q _ { K } ) = 2 \sum _ { a = 0 } ^ { K - 1 } q _ { a } q _ { K } .\tag{24}
$$

Splitting the unordered pairs (a, b) according to whether $b = K$ , we obtain the decomposition

$$
\underbrace { 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } } _ { \mathrm { d e n o m i n a t o r } } = \underbrace { 2 \sum _ { a = 0 } ^ { K - 1 } q _ { a } q _ { K } } _ { \mathrm { n u m e r a t o r } } + 2 \sum _ { \substack { 0 \le a < b < K } } q _ { a } q _ { b } .\tag{25}
$$

The second term on the right is non-negative, so the denominator is at least as large as the numerator and $\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) \leq 1$ . Combined with Lemma $\mathrm { A . 2 } { \mathrm { . } }$ , this gives

$$
\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) = 1 \geq \mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) .\tag{26}
$$

Step 3: Strictness. From Eq. (25), the inequality is strict if and only if

$$
\sum _ { 0 \leq a < b < K } q _ { a } q _ { b } > 0 ,\tag{27}
$$

which holds precisely when $q _ { a } q _ { b } > 0$ for some $0 \leq a < b < K$ , i.e., when at least two distinct imperfect quality levels are visited with positive probability. In this case,

$$
\mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) < 1 = \mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { c } ) ,\tag{28}
$$

which completes the proof.

Corollary A.5 (Collapse of imperfect samples under binary verification). The ranking error of the discrete binary reward is exactly

$$
1 - \mathrm { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) = \frac { 2 \sum _ { 0 \leq a < b < K } q _ { a } q _ { b } } { 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } } .\tag{29}
$$

Equivalently, every ranking error incurred by the binary reward comes from a pair of imperfect samples with diferent true quality levels.

Proof. Using the decomposition in Eq. (25),

$$
1 - \operatorname { A c c } _ { \mathrm { r a n k } } ( r _ { d } ) = { \frac { \left( 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } \right) - 2 \sum _ { a = 0 } ^ { K - 1 } q _ { a } q _ { K } } { 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } } } = { \frac { 2 \sum _ { 0 \leq a < b < K } q _ { a } q _ { b } } { 1 - \sum _ { k = 0 } ^ { K } q _ { k } ^ { 2 } } } .\tag{30}
$$

The numerator counts (twice) the probability mass of unordered pairs of distinct imperfect quality levels, which by Lemma A.3 are precisely the pairs $r _ { d }$ fails to rank. This proves the corollary. □

Implication for GRPO. By Lemma A.1, GRPO assigns larger update weights to samples with larger rewards. Therefore, a reward model that better preserves the true quality ordering also provides more reliable relative policy updates. The decomposed graded reward is strictly monotonic with respect to the number of satisfied sub-questions and thus preserves all pairwise quality comparisons. In contrast, the binary reward collapses all imperfect samples into the same reward, making GRPO unable to distinguish partially correct samples from severely incorrect ones. Consequently, Decomposed Graded Verification provides a denser and more order-preserving reward signal for GRPO.

## B Other Results

## B.1 Training Dynamics across Six Game-Based Tasks

Fig. 5 presents the training dynamics of pairwise and pointwise verification across our six constructed rulegoverned tasks. Following the evaluation protocol in the main text, we evaluate intermediate checkpoints every 10 training steps on the corresponding validation sets, using Gemini3.1-Pro (Google DeepMind, 2026) as an evaluator not involved in data processing and providing ground-truth images as explicit references.

Across all six tasks, pairwise verification generally yields more sustained performance improvements and higher final scores, whereas pointwise verification exhibits more pronounced fluctuations and limited progress. This trend is consistent with the aggregate results in the main text. By directly comparing generated outputs with ground-truth references, pairwise verification reduces reliance on internal targetstate inference and can provide more reliable reward signals for policy optimization. These results further support the efectiveness of reference-based verification across diferent types of rule-governed visual generation tasks.

![](images/17512d074818a70faeb3ebd585d5ef90752096b05d8499f132f8ddd28b13ed60.jpg)  
Figure 5: Training dynamics of pairwise and pointwise verification across six rule-governed tasks. Each panel shows validation scores and the score gap between the two methods.

## B.2 Sub-question-level Verification Accuracy

The reliability of decomposed graded scoring depends on whether the verifier correctly judges individual sub-questions against the generated image. Even when the decomposition is appropriate, incorrect local judgments can introduce noise into the aggregated reward. We therefore directly evaluate sub-question-level verification accuracy on a held-out set of 200 binary sub-question samples. Three authors and two Ph.D. students each annotate 40 samples, collectively providing human labels for all 200 samples. We additionally obtain binary annotations from GPT-5.5 and Gemini-3.1-Pro on the same evaluation set. We compare the base model, the SFT-trained graded verifier, and the graded verifier further optimized with Decoupled RL, reporting accuracy against each annotation source separately.

Table 5: Sub-question-level verification accuracy (%) on 200 held-out samples.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Annotation Source</td></tr><tr><td>GPT-5.5</td><td>Gemini-3.1-Pro</td><td>Human</td></tr><tr><td>Qwen3-VL-8B</td><td>68.5</td><td>69.5</td><td>67.0</td></tr><tr><td>+ SFT(Graded)</td><td>76.0</td><td>76.5</td><td>74.5</td></tr><tr><td>+ SFT(Graded) + RL(Graded)</td><td>94.5</td><td>96.0</td><td>94.0</td></tr></table>

As shown in Table 5, Decoupled RL substantially improves fine-grained verification accuracy across all three annotation sources. These results support the efectiveness of Decoupled RL in improving sub-questionlevel binary judgments and provide empirical evidence for the reliability of the local decisions underlying decomposed graded scoring.

## B.3 Evaluation on Qwen-Image-2512

To examine whether the efectiveness of our Decomposed Graded Verifier extends to a larger text-toimage generation backbone, we further evaluate Qwen-Image-2512 (20B), complementing the experiments on Stable Difusion 3.5 Medium (2.5B) and FLUX.1-dev (12B). We compare the original model with its reinforcement learning variants guided by the Discrete Binary Verifier and the Decomposed Graded Verifier, respectively, using GenEval and CLIPScore for evaluation.

Table 6: Performance of Qwen-Image-2512 on GenEval and CLIPScore under diferent reward models.
<table><tr><td rowspan="2">Methods</td><td colspan="7">GenEval</td><td rowspan="2">CLIPScore</td></tr><tr><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attri. Bind.</td><td>Overall</td></tr><tr><td>Qwen-Image-2512</td><td>0.99</td><td>0.90</td><td>0.77</td><td>0.86</td><td>0.38</td><td>0.52</td><td>0.74</td><td>34.1</td></tr><tr><td>+RL(Discrete Binary Verifier)</td><td>1.00</td><td>0.92</td><td>0.79</td><td>0.87</td><td>0.42</td><td>0.55</td><td>0.76</td><td>34.4</td></tr><tr><td>+RL(Decomposed Graded Verifier)</td><td>1.00</td><td>0.96</td><td>0.85</td><td>0.92</td><td>0.51</td><td>0.69</td><td>0.82</td><td>35.3</td></tr></table>

As shown in Table 6, the Decomposed Graded Verifier improves the overall GenEval score from 0.74 to 0.82, outperforming the binary-reward counterpart by 0.06. The largest gains over the original model occur in attribute binding and position, which improve from 0.52 to 0.69 and from 0.38 to 0.51, respectively. These improvements suggest that fine-grained graded rewards are particularly useful for learning compositional requirements involving attribute associations and spatial relations. CLIPScore also increases from 34.1 to 35.3, compared with 34.4 under binary rewards. Together, these results extend the evidence for the efectiveness of decomposed graded rewards to a larger generation backbone with stronger baseline performance.

## B.4 Evaluation on InternVL3-8B

To examine whether the benefits of our training strategy extend beyond the Qwen model family, we additionally adopt InternVL3-8B as the verifier backbone. We evaluate the base model, its graded SFT variant, and the graded verifier further optimized with Decoupled RL on ViVerBench.

As shown in Table 7, Decoupled RL improves the mean score across all 16 task categories from 0.606 to 0.652 compared with graded SFT, with the seven displayed sub-metrics highlighting selected categories with notable gains. These results demonstrate that our training strategy is also efective on InternVL3-8B, extending its applicability beyond the Qwen family.

Table 7: Performance on ViVerBench with InternVL3-8B as the verifier backbone. We report seven selected sub-metrics with notable improvements after Decoupled RL. Overall denotes the mean score across all 16 task categories, including those not displayed.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Concept Existence</td><td colspan="2">Object Relation</td><td colspan="3">Image Annotation</td><td rowspan="2">Overall (16 tasks)</td></tr><tr><td>Obj.</td><td>Attr.</td><td>Spat.</td><td>N-Spat.</td><td>BBox</td><td>Point</td><td>Count</td></tr><tr><td>InternVL3-8B</td><td>0.513</td><td>0.578</td><td>0.667</td><td>0.619</td><td>0.750</td><td>0.619</td><td>0.604</td><td>0.594</td></tr><tr><td>+SFT(Graded)</td><td>0.526</td><td>0.585</td><td>0.652</td><td>0.632</td><td>0.757</td><td>0.611</td><td>0.618</td><td>0.606</td></tr><tr><td>+SFT(Graded)+RL(Graded)</td><td>0.678</td><td>0.684</td><td>0.707</td><td>0.724</td><td>0.821</td><td>0.665</td><td>0.649</td><td>0.652</td></tr></table>

## B.5 Sensitivity to the Number of Sub-questions

To investigate how decomposition granularity afects global score estimation, we evaluate the trained Decomposed Graded Verifier on 200 held-out test samples. We explicitly instruct the verifier through the system prompt to generate sub-questions within three ranges: at most 10, 11–20, and 21–30. This experiment examines whether increasing the number of sub-questions consistently improves scoring accuracy or introduces additional errors through excessive decomposition.

Table 8: Efect of the number of sub-questions on global score estimation on 200 held-out test samples. Lower mean alignment error is better.
<table><tr><td rowspan=1 colspan=1>Number of Sub-questions</td><td rowspan=1 colspan=1>Mean Alignment Error↓</td></tr><tr><td rowspan=1 colspan=1>≤ 10</td><td rowspan=1 colspan=1>0.083</td></tr><tr><td rowspan=1 colspan=1>11-20</td><td rowspan=1 colspan=1>0.039</td></tr><tr><td rowspan=1 colspan=1>21-30</td><td rowspan=1 colspan=1>0.072</td></tr></table>

As shown in Table 8, the 11–20 setting achieves the lowest mean alignment error of 0.039, compared with 0.083 for at most 10 sub-questions and 0.072 for 21–30 sub-questions. These results suggest that an appropriate decomposition granularity improves scoring accuracy, while simply increasing the number of sub-questions does not guarantee better performance.

Excessive decomposition can introduce redundant questions or requirements unsupported by the original prompt. For example, given “A red apple is placed on a table,” the verifier may repeatedly ask whether an apple exists or introduce questions such as “Is the apple a fruit?” and “Is the apple round?” that do not correspond to explicitly requested visual constraints. Such questions can distort the aggregated score by repeatedly weighting the same requirement or introducing irrelevant criteria. We also observe that generating more sub-questions can lead to violations of the required output format, reducing the reliability of the final score. Together, these observations highlight the importance of balancing fine-grained coverage with faithful and non-redundant decomposition.

## C Limitations and Future Works

Our world-modeling evaluation is restricted to game-based environments with explicit rules, where the ground-truth image is uniquely determined by the initial task state and rule instructions. We adopt these controlled tasks primarily to evaluate the efectiveness of our pairwise verification paradigm. Although our results demonstrate the advantages of pairwise verification in these settings, these tasks capture only part of the complexity and diversity of real-world environments.

In future work, we plan to extend our investigation to open-ended settings with non-unique ground truths, construct larger and more realistic world-modeling datasets, and scale both the verifier and the generator to stronger backbones. Our long-term goal is to develop multimodal generative models that can reason about, simulate, and interact with complex real-world scenes.

## D Decomposition Example

## Text-to-Image (1)

![](images/a772f54dd610a7399a931375de852f36f1c1a2c8219cc7a86aad09935aa8caa3.jpg)

## Prompt:

A baboon is silhouetted on a patio in a top-down, monochromatic view; acacia trees stark against the golden hour dusk.

## Decomposition Result:

• Is there a baboon present in the image? → Yes, Score: 1

• Is the baboon depicted in silhouette? → Yes, Score: 1

• Is the baboon situated on a patio or tiled surface? → Yes, Score: 1

• Is the camera perspective a top-down view? → No, Score: 0

• Is the image monochromatic? → No, Score: 0

• Are there acacia trees visible in the background? → Yes, Score: 1

• Is the scene set during golden hour dusk? → Yes, Score: 1

## Final Alignment Score:

$$
\frac { 5 } { 7 } = 0 . 7 1
$$

## Text-to-Image (2)

![](images/075527575a50b07b84f169add15370f5630ba58e65712b604acc7e652d160524.jpg)

## Prompt:

Isometric 3D café interior, Pixar-style rendering, whimsical aesthetic. Features cofee counter, colorful shelves, rounded furniture, pastry case, and hanging lamps. Large glass door opens to a park with string lights at sunset. Warm glowing lighting, soft curved edges, cozy evening atmosphere.

## Decomposition Result:

• Is the image a 3D rendering of a café interior with a Pixar-style aesthetic? → Yes, Score: 1

• Is the viewpoint isometric? → No, Score: 0

• Does the interior feature a cofee counter? → Yes, Score: 1

• Are there shelves with colorful items? → Yes, Score: 1

• Are the furniture and objects rounded with soft curved edges? → Yes, Score: 1

• Does the café include both hanging lamps and a pastry case? → Yes, Score: 1

• Does a large glass door open to an outdoor park scene? → Yes, Score: 1

• Are string lights visible in the outdoor park scene? → Yes, Score: 1

• Is the scene set at sunset with warm, glowing, cozy lighting? → Yes, Score: 1

## Final Alignment Score:

$$
\frac { 8 } { 9 } = 0 . 8 9
$$

## Text-to-Image (3)

![](images/5ccebcb3157b66926e699c917a2fb36997cc0b33282c216275cc30b7a938f533.jpg)

## Prompt:

Imagine an enchanted city where the streets resonate like symphonies, and the architecture is formed from orchestral instruments and glimmering harmonies.

## Decomposition Result:

• Does the image show an enchanted city? → Yes, Score: 1

• Is there a clear city street scene? → Yes, Score: 1

• Are musical notes or symphony elements visible? → Yes, Score: 1

• Are orchestral instruments visible? → Yes, Score: 1

• Are the instruments integrated into the cityscape? → Yes, Score: 1

• Is the architecture formed from instruments? → No, Score: 0

• Are glimmering lights or particles visible? → Yes, Score: 1

• Are the glimmering elements linked to the instruments? → Yes, Score: 1

• Does the scene combine city, music, and magic coherently? → Yes, Score: 1

## Final Alignment Score:

$$
\frac { 8 } { 9 } = 0 . 8 9
$$

## Text-to-Image (4)

![](images/eb737d2570257c7c375b089790067b53498a3d74eb72d4f154640bc1d67815d5.jpg)

## Prompt:

A retro-themed photograph of a medieval castle resting atop a colossal tree trunk, enveloped by clouds, shimmering fireflies, and a dusky twilight sky. The scene is illuminated by fireflies.

## Decomposition Result:

• Does the image depict a medieval castle? → Yes, Score: 1

• Is the castle resting atop a colossal tree trunk? → Yes, Score: 1

• Is the scene enveloped by clouds? → Yes, Score: 1

• Are shimmering fireflies visible in the image? → Yes, Score: 1

• Is the sky a dusky twilight sky? → Yes, Score: 1

• Is the image a retro-themed photograph? → No, Score: 0

• Is the scene illuminated by the fireflies? → No, Score: 0

## Final Alignment Score:

$$
\frac { 5 } { 7 } = 0 . 7 1
$$

Text-to-Image (5)  
![](images/a495c339ad4a36cafacb8bf060290c4f94bd56d9f49dd125fda5e1618bbd29e6.jpg)  
Decomposition Result:  
• Is the scene set in a bustling train station? → Yes, Score: 1

## Prompt:

In a bustling train station, a traveler in a dark olive jacket and maroon scarf wears a well-worn black backpack. They focus on a tablet, likely checking schedules. Nearby, a polished blue train waits at the platform, ready to welcome passengers.

• Is the traveler wearing a dark olive jacket? → Yes, Score: 1

• Is the traveler wearing a maroon scarf? → Yes, Score: 1

• Is the traveler wearing a well-worn black backpack? → No, Score: 0

• Is the traveler focusing on a tablet? → Yes, Score: 1

• Is there a polished blue train waiting at the platform? → Yes, Score: 1

• Is the train’s door open to welcome passengers? → Yes, Score: 1

## Final Alignment Score:

$$
\frac { 6 } { 7 } = 0 . 8 6
$$

## Rule-Governed World Worlding Pairwise

Initial Image  
![](images/67f359362b8aadbbc9ee8bc4b8e36f86a48f1e3e7916a8f016316743ec75db56.jpg)

## Game Rule:

The objective is to generate a 2D orthographic front-view projection of the given 3D voxel structure. The front view is obtained by looking along the X-axis and projecting the structure onto the Y-Z plane.

![](images/2e113e30605856f7440039ec3f26abf62bf230985d9adf6e35d32e8a8f9c0701.jpg)

![](images/be2c8a290bf2a60327cf4df3b5da224cc87f04e5a8666af50a9f61146cc0d9dd.jpg)  
Reference Image  
Generated Image

## Decomposition Reasoning:

So, let’s tackle this step by step.

$Z = 3 { : }$ empty, filled, filled   
$Z = 2 \colon$ empty, filled, filled   
$Z = 1 { : }$ filled, filled, filled

The task is to evaluate whether the generated image correctly produces the 2D orthographic front-view projection of the given 3D voxel structure. Since the front view is obtained by looking along the X-axis, the projection should be represented on the Y-Z plane.

I use the Reference Image as the correct target and compare the Generated Image with it cell by cell. Each position in the $3 \times 3 \ : \mathrm { Y - } \breve { Z }$ grid can be treated as a small sub-question: whether the generated cell matches the corresponding reference cell.

The Reference Image shows the projection as follows:

Based on the Reference Image, I decompose the comparison into the following cell-level checks:

• Is the cell at $( Y = 1 , Z = 1 )$ correct? The Reference Image fills this cell, and the Generated Image also fills it. → Yes, Score: 1

• Is the cell at $( Y = 2 , Z = 1 )$ correct? The Reference Image fills this cell, and the Generated Image also fills it. → Yes, Score: 1

• Is the cell at $( Y = 3 , Z = 1 )$ correct? The Reference Image fills this cell, but the Generated Image leaves it empty. → No, Score: 0

• Is the cell at $( Y = 1 , Z = 2 )$ correct? The Reference Image leaves this cell empty, and the Generated Image also leaves it empty. → Yes, Score: 1

• Is the cell at $( Y = 2 , Z = 2 )$ correct? The Reference Image fills this cell, and the Generated Image also fills it. → Yes, Score: 1

• Is the cell at $( Y = 3 , Z = 2 )$ correct? The Reference Image fills this cell, and the Generated Image also fills it. → Yes, Score: 1

• Is the cell at $( Y = 1 , Z = 3 )$ correct? The Reference Image leaves this cell empty, and the Generated Image also leaves it empty. $\begin{array} { r } {  \Upsilon \mathrm { e s , } } \end{array}$ , Score: 1

• Is the cell at $( Y = 2 , Z = 3 )$ correct? The Reference Image fills this cell, and the Generated Image also fills it. → Yes, Score: 1

• Is the cell at $( Y = 3 , Z = 3 )$ correct? The Reference Image fills this cell, and the Generated Image also fills it. → Yes, Score: 1

From these sub-question judgements, the Generated Image matches the Reference Image in 8 out of 9 cells. The only mismatch is at $( \mathbf { \dot { Y } } = 3 , Z = 1 )$ , where the Reference Image shows an empty cell but the Generated Image marks it as filled.

Thus, there are 9 sub-questions in total, with 8 correct and 1 incorrect.

$$
{ \mathrm { S c o r e } } = { \frac { 8 } { 9 } } = 0 . 8 9 .
$$

## Rule-Governed World Worlding Pointwise

Generated Image  
![](images/46507429dbe7b23c879ec4dc656af6875c461ed93e47d3d1f069d2638d32729f.jpg)

![](images/5719e680411e153fe40ef2db9e0b8dc2a13d71439839420efdc8e9314826b478.jpg)

## Decomposition Reasoning:

So, let’s first work out the target projection implied by the initial 3D structure.

The required view is produced by observing the voxel structure along the X-axis. In this process, the X dimension is removed, and the remaining projection is represented on the Y-Z plane, with Y as the horizontal axis and Z as the vertical axis.

Thus, for every $( Y , Z )$ coordinate, I need to decide whether the projection cell should be occupied. If at least one voxel exists somewhere along the X direction at that same $( \check { Y } , Z )$ coordinate, the cell should be filled; otherwise, it should be empty.

By inspecting the initial 3D structure, the expected occupancy pattern is:

• For $Y = 1 ,$ the structure occupies $Z = 1$ and $Z = 2 .$

• For Y = 2, the structure occupies $Z = 1 , Z = 2 , \mathrm { a n d } Z = 3 .$

• For $Y = 3 ,$ the structure occupies $Z = 1 , Z = 2 ,$ and $Z = 3 .$

So the inferred target projection should mark the following cells as filled:

$$
\begin{array} { l l } { ( Y = 1 , Z = 1 ) , ( Y = 1 , Z = 2 ) , } \\ { ( Y = 2 , Z = 1 ) , ( Y = 2 , Z = 2 ) , ( Y = 2 , Z = 3 ) , } \\ { ( Y = 3 , Z = 1 ) , ( Y = 3 , Z = 2 ) , ( Y = 3 , Z = 3 ) . } \end{array}
$$

Next, I break the projection into individual cell-level questions and compare the generated result with this inferred target.

• Is $( Y = 1 , Z = 1 )$ correctly represented? The inferred target requires this cell to be filled, and the generated projection also fills it. → Yes, Score: 1

• Is $( Y = 1 , Z = 2 )$ correctly represented? The inferred target requires this cell to be filled, but the generated projection leaves it empty. → No, Score: 0

• Is $( Y = 1 , Z = 3 )$ correctly represented? The inferred target requires this cell to be empty, and the generated projection also leaves it empty. → Yes, Score: 1

• Is $( Y = 2 , Z = 1 )$ correctly represented? The inferred target requires this cell to be filled, and the generated projection also fills it. → Yes, Score: 1

• Is $( Y = 2 , Z = 2 )$ correctly represented? The inferred target requires this cell to be filled, and the generated projection also fills it. → Yes, Score: 1

• Is $( Y = 2 , Z = 3 )$ correctly represented? The inferred target requires this cell to be filled, and the generated projection also fills it. → Yes, Score: 1

• Is $( Y = 3 , Z = 1 )$ correctly represented? The inferred target requires this cell to be filled, but the generated projection leaves it empty. → No, Score: 0

• Is $( Y = 3 , Z = 2 )$ correctly represented? The inferred target requires this cell to be filled, and the generated projection also fills it. → Yes, Score: 1

• Is $( Y = 3 , Z = 3 )$ correctly represented? The inferred target requires this cell to be filled, and the generated projection also fills it. → Yes, Score: 1

After aggregating these judgements, the generated projection correctly represents most of the inferred Y-Z occupancy pattern. The two failed sub-questions correspond to $( Y = \dot { 1 } , Z ^ { \bullet } = 2 )$ and $( Y = 3 , Z = 1 )$ , where the generated projection leaves the cells empty even though they should be filled.

Therefore, among the 9 decomposed sub-questions, 7 are judged as correct and 2 are judged as incorrect.

$$
\mathbf { S c o r e } = { \frac { 7 } { 9 } } = 0 . 7 8 .
$$

## Text-to-Image Verifier Prompt

## You are a powerful and trustworthy Text-to-Image Alignment Evaluator.

Your responsibility is to evaluate the semantic alignment between a generated image and a text prompt. You must decompose the text prompt into verifiable atomic questions and score the consistency between the image and the prompt based strictly on visual evidence.

## Inputs

You will be given the following inputs:

• Text Prompt: the textual description used to generate the image.

• Generated Image: the image produced by the text-to-image model.

## Evaluation Workflow

## Step 1: Prompt Decomposition

Decompose the text prompt into a set of atomic questions that can be directly verified from the image. Each question must originate from facts explicitly stated in the text prompt and test one clear semantic assertion.

## Step 2: Image-Based Answering

Answer each question solely based on the visual information present in the Generated Image. Do not introduce assumptions beyond the text prompt. Do not infer visual information that is absent, ambiguous, or cannot be confirmed from the image.

## Step 3: Consistency Judgement and Scoring

For each question, determine whether the image-based answer is consistent with the information explicitly specified in the text prompt. If the image-based answer is consistent with the prompt, the question receives 1 point. Otherwise, it receives 0 points.

## Question Decomposition Principles

The decomposed questions must satisfy the following requirements:

• Each question must be binary, with a Yes or No answer.

• Each question must be objectively verifiable through visual evidence.

• Questions involving subjective judgment, emotions, aesthetics, or other non-verifiable properties are forbidden.

• The number of questions should be naturally determined by the semantic complexity of the text prompt.

The decomposition should follow the semantic order below when applicable:

• Objects: existence, quantity, and category.

```erb
• Object Attributes: color, shape, material, appearance, actions, and visible states.
• Spatial Relations: relative position, orientation, and layout.
• Non-Spatial Relations: interactions and action relations between entities.
Output Requirements
Output only a valid JSON object containing exactly two top-level fields: "sub_questions" and "final_score". Do
not include Markdown code fences or additional text.
"sub_questions": [
{
"id": 1,
"question": "A decomposed sub-question",
"answer": "True or False",
"explanation": "Brief error explanation if False; otherwise an empty string"
}
],
"final_score": x
}
Replace the placeholders above with actual evaluation results. Include one entry for each distinct decomposed
sub-question, with consecutive IDs starting at 1.
Here, x = C/N, where N is the total number of entries and C is the number answered "True". Output x as a
JSON number rounded to two decimal places.
Inputs for This Evaluation
Text Prompt: $ { prompt }
The attached image is the Generated Image. <image>
```

## Rule-Governed World Modeling Verifier Prompt-Pairwise Rule-Governed World Modeling Verifier Prompt-Pairwise

You are a powerful and trustworthy Image Editing Evaluator.   
Your responsibility is to evaluate whether an image editing model successfully completes a game-based image   
editing task according to the provided rules.   
Inputs   
You will be given the following inputs:   
• Initial Image: the original game state given to the image editing model.   
• Generated Image: the output produced by the image editing model.   
• Reference Image: the ground-truth result that correctly follows all rules.   
• Game Rules: the rules governing the game and editing constraints.   
Evaluation Workflow   
Step 1: Rule Decomposition   
Independently decompose the Game Rules into atomic, visually checkable rules, even if they are already   
decomposed. The resulting rules must remain consistent with the provided Game Rules.   
Step 2: Rule Violation Checking   
Use the Reference Image as the correct standard. Compare the Generated Image with the Reference   
Image rule by rule, and determine whether each rule is violated.   
Step 3: Scoring

Let the total number of decomposed atomic rules be N, and let the number of violated rules be V. The score of the Generated Image is:

N − V   
Score =   
N   
The final score should reflect how consistently the Generated Image follows the rules compared with the   
Reference Image.   
Output Requirements   
Output only a valid JSON object with exactly two top-level fields: "sub\_questions" and "final\_score". Do not   
include Markdown code fences or additional text.   
{   
"sub\_questions": [   
{   
"id": 1,   
"question": "A decomposed rule sub-question",   
"answer": "True or False",   
"explanation": "Brief rule-violation explanation if False; otherwise an empty string"   
}   
],   
"final\_score": x   
}   
Replace the placeholders above with actual evaluation results. Include one entry for each distinct decomposed   
atomic rule, with consecutive IDs starting at 1.   
Here, x is the score calculated in Step 3, expressed as a JSON number rounded to two decimal places.   
Inputs for This Evaluation   
Game Rules: \$ { game\_rules }   
Images are attached in this order: Initial Image, Generated Image, Reference Image. <image><image><image>

## Rule-Governed World Modeling Verifier Prompt-Pointwise

You are a powerful and trustworthy Image Editing Evaluator.   
Your responsibility is to evaluate whether an image editing model successfully completes a game-based image   
editing task according to the provided rules.   
Inputs   
You will be given the following inputs:   
• Initial Image: the original game state given to the image editing model.   
• Generated Image: the output produced by the image editing model.   
• Game Rules: the rules governing the game and editing constraints.   
Evaluation Workflow   
Step 1: Rule Decomposition   
Independently decompose the Game Rules into atomic, visually checkable rules, even if they are already   
decomposed. The resulting rules must remain consistent with the provided Game Rules.   
Step 2: Rule Violation Checking   
Use the Initial Image and the Game Rules to infer the expected final state. Compare the Generated   
Image with this expected state rule by rule, and determine whether each rule is violated.   
Step 3: Scoring

Let the total number of decomposed atomic rules be N, and let the number of violated rules be V. The score of the Generated Image is:

N − V   
Score =   
N   
The final score should reflect how consistently the Generated Image follows the Game Rules given the Initial   
Image.   
Output Requirements   
Output only a valid JSON object with exactly two top-level fields: "sub\_questions" and "final\_score". Do not   
include Markdown code fences or additional text.   
{   
"sub\_questions": [   
{   
"id": 1,   
"question": "A decomposed rule sub-question",   
"answer": "True or False",   
"explanation": "Brief rule-violation explanation if False; otherwise an empty string"   
}   
],   
"final\_score": x   
}   
Replace the placeholders above with actual evaluation results. Include one entry for each distinct decomposed   
atomic rule, with consecutive IDs starting at 1.   
Here, x is the score calculated in Step 3, expressed as a JSON number rounded to two decimal places.   
Inputs for This Evaluation   
Game Rules: \$ { game\_rules }   
Images are attached in this order: Initial Image, Generated Image. <image><image>

## Text-to-Image Sub-Question Verifier Prompt

Evaluate the following sub-question about the Generated Image.   
Please carefully analyze the Generated Image and determine whether it satisfies the requirement expressed in   
the sub-question. Base your judgment solely on the visual evidence in the Generated Image.   
If the Generated Image satisfies this requirement, answer "True"; otherwise, answer "False". Evaluate only the   
requirement expressed in this sub-question.   
Respond strictly with a JSON object containing exactly these two fields:   
{   
"answer": "True or False",   
"explanation": "Brief error explanation if False; otherwise an empty string"   
}   
Use the string "True" or "False" for "answer". If the answer is "False", briefly describe the specific mismatch or   
missing visual evidence in "explanation"; otherwise, use an empty string. Do not include any additional text.   
Inputs for This Evaluation   
Sub-question: \$ { question }   
The attached image is the Generated Image. <image>

## Rule-Governed World Modeling Sub-Question Verifier Prompt-Pairwise

Evaluate the following sub-question about the Generated Image.   
Please carefully compare the Generated Image with the Reference Image and determine whether the   
Generated Image satisfies the requirement expressed in the sub-question. Use the Reference Image as the   
correct standard, with the Initial Image and Game Rules providing the task context.   
If the Generated Image satisfies this requirement, answer "True"; otherwise, answer "False". Evaluate only the   
requirement expressed in this sub-question.   
Respond strictly with a JSON object containing exactly these two fields:   
"answer": "True or False",   
"explanation": "Brief error explanation if False; otherwise an empty string"   
}   
Use the string "True" or "False" for "answer". If the answer is "False", briefly describe the specific mismatch or   
missing visual evidence in "explanation"; otherwise, use an empty string. Do not include any additional text.   
Inputs for This Evaluation   
Sub-question: \$ { question }   
Game Rules: \$ { game\_rules }   
Images are attached in this order: Initial Image, Generated Image, Reference Image. <image><image><image>

## Rule-Governed World Modeling Sub-Question Verifier Prompt-Pointwise

Evaluate the following sub-question about the Generated Image.   
Please analyze the Initial Image, then apply the Game Rules to infer the expected final state. Carefully   
compare the Generated Image with this expected state and determine whether the Generated Image   
satisfies the requirement expressed in the sub-question.   
If the Generated Image satisfies this requirement, answer "True"; otherwise, answer "False". Evaluate only the   
requirement expressed in this sub-question.   
Respond strictly with a JSON object containing exactly these two fields:   
"answer": "True or False",   
"explanation": "Brief error explanation if False; otherwise an empty string"   
}   
Use the string "True" or "False" for "answer". If the answer is "False", briefly describe the specific mismatch or   
missing visual evidence in "explanation"; otherwise, use an empty string. Do not include any additional text.   
Inputs for This Evaluation   
Sub-question: \$ { question }   
Game Rules: \$ { game\_rules }   
Images are attached in this order: Initial Image, Generated Image. <image><image>