# Translation as a Decision Space: A Multi-Agent Perspective on Low-Resource Dialect Generation

Hasan Alkhder<sup>1⋆</sup> Mohammad Abboush<sup>2</sup> Igor Tchappi<sup>3</sup> Ahmet Zengin<sup>1</sup> Amro Najjar<sup>4</sup>

Abstract Neural machine translation (NMT) systems typically produce a single output per input, obscuring the alternative decision trajectories implicitly available within multilingual decoding. This opacity becomes particularly problematic in low-resource dialect settings, where multiple linguistically valid realizations may difer in lexical authenticity, register, and structural stability.

We propose reframing translation as a structured decision space explored by autonomous translation agents. Instead of analyzing a single output, we model distinct translation pathways as agents operating over a shared multilingual backbone. Inter-agent divergence is treated not as error but as an interpretable behavioral signal.

We conduct an empirical study on Turkish–Syrian Arabic translation using three agents: (1) zero-shot direct translation, (2) dialect-stabilized translation via lightweight fine-tuning, and (3) pivot translation through English. Evaluation is performed on 5,000 dialogue sentences, while stabilization is trained on 5,000 additional Turkish–Syrian sentence pairs drawn from television dialogue and MADAR-Turk resources.

Rather than optimizing for conventional performance metrics, we quantify structured behavioral displacement using dialect marker frequency, lexical proximity to standardized Arabic, and structural variance. Lightweight stabilization nearly doubles dialect marker usage, increasing it from 0.2266 to 0.4988, while significantly reducing structural instability. Pivot mediation introduces normalization pressure and measurable compression efects, whereas zero-shot translation exhibits the highest decision variance.

We argue that translation divergence across agents reveals latent decision flexibility within multilingual models. By modeling translation as a multi-agent system over a shared decision space, we provide a principled interpretability framework for low-resource dialect generation.

Keywords: Explainable AI, Multi-Agent Systems, Dialect Translation, Decision Space Modeling, Low-Resource NLP

<sup>⋆</sup>Corresponding author: hasan.alkhder2@ogr.sakarya.edu.tr

## 1 Introduction

Neural machine translation (NMT) systems are commonly treated as deterministic functions that map a source sentence to a single target realization. In practice, however, multilingual generative models operate over high-dimensional probability distributions that encode multiple plausible translation trajectories. Despite this inherent flexibility, only one output surface form is typically exposed to the user. The alternative decisions embedded within the decoding space remain hidden. This creates a limitation for dialect translation analysis, because multiple valid realizations may exist while standard translation pipelines expose only a single output.

This opacity becomes especially consequential in low-resource dialect settings. Unlike standardized language translation, dialect translation involves competing realizations that difer not only lexically but socially and stylistically. Dialectal outputs may vary in register, authenticity, emotional intensity, and structural compression. In such contexts, divergence across outputs is not necessarily error; it may reflect meaningful trade-ofs within the model’s internal decision landscape. Existing MT evaluation approaches capture translation adequacy reasonably well, but they are less suited for analyzing dialect authenticity, register choice, and structural variation across alternative translation realizations.

Syrian Arabic provides a particularly challenging case. It lacks standardized orthography, exhibits lexical divergence from Modern Standard Arabic (MSA), and is supported by limited parallel resources [6]. Multilingual models trained primarily on standardized data frequently display instability when generating dialectal text. This instability manifests as partial normalization toward MSA, repetition collapse, structural drift, or inconsistent colloquial weighting. Metrics such as BLEU [18] and COMET [19], while efective for measuring translation adequacy, do not capture dialectal register, normalization pressure, or structural behavioral diferences across translation pathways.

In this work, we propose reframing translation as a decision space explored by translation agents. We adopt an operational definition of agents as constrained decoding pathways within a shared multilingual model, rather than fully autonomous entities in the classical multi-agent systems (MAS) sense. Each agent corresponds to a specific decision constraint applied to the same underlying model. Rather than treating translation as a single-output mapping, we model distinct translation pathways as independent agents operating over a shared multilingual backbone. Each agent embodies a specific decision constraint— zero-shot generation, dialect-reinforced adaptation, or pivot mediation—while preserving identical architecture and decoding parameters. We use the term agent because each pathway produces systematically diferent translation behavior under a defined constraint regime. The observable divergence between agents is interpreted not as noise, but as structured behavioral signal.

We conduct an empirical case study on Turkish–Syrian Arabic translation. Three agents are constructed: (1) direct zero-shot Turkish → Syrian translation, (2) dialect-stabilized translation via lightweight fine-tuning on 5,000 Turkish–

Syrian sentence pairs, and (3) pivot translation through English (Turkish → English → Syrian). Evaluation is performed on 5,000 aligned dialogue segments drawn from independent dialogue sources. This controlled setup allows us to isolate decision-space reweighting efects from architectural changes.

Instead of optimizing for conventional performance metrics, we quantify inter-agent divergence using dialect marker frequency, lexical proximity to standardized Arabic, and structural length variance. Together, these metrics capture complementary aspects of dialect activation, normalization pressure, and structural stability across translation pathways. Our findings show that lightweight dialect stabilization nearly doubles dialect marker usage while reducing structural instability, whereas pivot mediation introduces compression and normalization efects. Zero-shot translation exhibits the highest variance, reflecting under-specified dialect representation.

By modeling translation as a multi-agent system operating within a shared decision space, we introduce an interpretability-oriented framework for analyzing low-resource dialect generation. Rather than evaluating translation solely through final-output quality, our approach examines how controlled variation in translation pathways reveals structured behavioral diferences within the same multilingual model. This perspective reframes divergence not as translation failure, but as an observable signal of how multilingual systems balance authenticity, normalization, and semantic trade-ofs under diferent decision constraints. The central research question of this work is whether controlled translation pathway variation can reveal systematic and interpretable diferences in low-resource dialect translation behavior.

Importantly, this study does not aim to improve machine translation accuracy. Instead, translation is used as a controlled generative environment in which agent-level behavioral divergence can be observed and analyzed.

The contributions of this work are threefold. First, we introduce a decisionspace perspective for analyzing multilingual translation systems by modeling translation pathways as constrained agents. Second, we propose behavioral metrics to quantify inter-agent divergence in dialect activation, lexical normalization, and structural realization. Third, we provide an empirical analysis of Turkish– Syrian Arabic translation demonstrating how stabilization and pivot mediation reshape decoding behavior.

We emphasize that the notion of ”agents” in this work is an abstract and analytical construct rather than a realization of classical multi-agent systems. The objective is not to model autonomous or interacting agents, but to provide a structured perspective for analyzing controlled decoding pathways within a shared multilingual model. In this sense, agents represent constrained operational views of the same system, enabling systematic exploration of the translation decision space.

## 2 Related Work

Transfer Learning for Low-Resource MT Transfer learning has become a widely adopted strategy for improving translation performance in data-scarce scenarios. Several studies have shown that knowledge acquired from high-resource language pairs can be transferred to low-resource tasks. For instance, Neubig et al. proposed similar-language regularization through joint training on related languages to reduce overfitting [17]. Gu et al. framed low-resource translation as a meta-learning problem that enables rapid adaptation from limited data [9]. Kim et al. investigated cross-lingual transfer using language-agnostic encoders and cross-lingual embeddings to alleviate vocabulary mismatch [10], while Luo et al. introduced a hierarchical transfer framework that progressively trains on unrelated high-resource pairs, intermediate languages, and finally low-resource pairs [14].

Although these techniques significantly improve translation quality, they generally produce deterministic single-output translations without revealing alternative translation hypotheses or underlying decision mechanisms.

Arabic Dialect Translation Arabic dialect translation introduces additional challenges due to morphological complexity, code-switching, and the limited availability of dialectal corpora. In this context, Sibaee et al. proposed SHAMI-MT, a bidirectional translation system for Syrian Arabic and Modern Standard Arabic based on AraT5v2 and trained on the Nabra dataset [22]. Baniata et al. developed a multitask learning architecture that shares a decoder across dialect translation tasks to exploit cross-task information [3]. Salloum et al. demonstrated that unsupervised morphological segmentation can improve dialectal translation performance [20], while Shapiro et al. examined the influence of dialect identification on Arabic neural machine translation [21].

Furthermore, the MADAR shared tasks have advanced dialect identification research, achieving macro-averaged F1-scores above 67% across 25 dialects [1, 23]. Alkheder et al. introduced the MADAR-Turk dataset, which contains manually translated Damascus–Turkish sentence pairs intended to facilitate benchmarking in dialectal translation tasks [2].

Despite these developments, most approaches treat translation as a singlestage mapping process and provide limited interpretability regarding the influence of lexical and morphological variations on translation decisions.

Explainability in Neural Machine Translation Explainable AI techniques have also been explored to analyse the behaviour of neural translation models. Ferrando et al. proposed ALTI+, which traces token-level attribution across source and target representations [8]. Voita et al. adapted Layerwise Relevance Propagation for Transformer models to measure source and target contributions [24]. Moradi et al. showed that diferent attention distributions can produce identical predictions, highlighting the limitations of attention-based explanations [15]. Additional studies have evaluated explanation fidelity and attention-head importance in translation models [25].

However, these approaches mainly provide post-hoc interpretations of single outputs rather than exposing the broader decision processes underlying translation generation.

Multi-Agent Approaches and Pivot Translation Recent work has also explored multi-agent frameworks for machine translation. Wu et al. introduced TransAgents, a collaborative architecture inspired by human translation workflows [26]. Other studies proposed multi-agent learning strategies based on ensemble distillation, mutual learning, or collaborative debate mechanisms [5, 13, 12].

While these methods improve translation quality through collaborative refinement, they typically treat agents as separate models rather than as entities exploring structured translation alternatives.

Pivot translation represents another approach for low-resource language pairs by exploiting high-resource intermediate languages. For example, Kim et al. proposed pivot-based transfer learning through step-wise pre-training [11]. Additional work investigated multi-pivot ensembling and simultaneous multi-pivot translation strategies to improve robustness [16, 7].

Despite their efectiveness, pivot-based approaches may propagate errors across translation stages and provide limited insight into the decision processes involved in pivot-mediated translation.

Summary and Research Gap Overall, the existing literature reveals an important research gap in neural machine translation for low-resource dialects. Most approaches focus on improving translation quality while treating translation as a deterministic mapping that generates a single output. Consequently, current methods ofer limited insight into the internal decision processes of translation models or the alternative translation trajectories that may exist. This limitation is particularly critical for dialectal translation, where linguistic variability and limited data availability require more interpretable systems capable of explaining how lexical and structural choices are made during translation.

## 3 Data Resources and Experimental Setup

## 3.1 Evaluation Dataset

The evaluation corpus consists of 5K sentence-level dialogue segments aligned across multiple linguistic representations. The dataset is constructed as a fixed evaluation set rather than a benchmark-scale testbed. Since the goal of this study is controlled behavioral comparison between translation pathways, a stable set of 5K sentences provides suficient coverage while enabling systematic analysis under identical conditions.

All translation agents operate on the same evaluation corpus so that observed behavioral diferences can be attributed to translation pathway efects rather than dataset variation.

Each entry contains four aligned linguistic representations. The Turkish column (TR) contains the original source sentence. The Syrian Arabic column (SY)

provides the reference dialectal translation derived from professionally dubbed dialogue. The English column (EN) contains an aligned English rendering used for pivot translation modeling. The Modern Standard Arabic column (MSA) provides a standardized Arabic form used for lexical proximity comparison.

Although the agents generate Turkish→Syrian outputs exclusively, the presence of English and MSA columns enables structured analysis of pivot mediation and normalization efects.

The 5K evaluation sentences are drawn from two complementary sources. The first consists of 3K dialogue segments extracted from a professionally dubbed Turkish television series provided by NIS Studios, capturing naturally occurring conversational speech with emotional and informal discourse patterns.

The second consists of 2K conversational sentence pairs from the MADAR-Turk corpus [2], providing structurally controlled conversational data.

Combining these sources balances natural dialogue richness with structurally controlled conversational data. Television dialogue contributes authentic colloquial phrasing and discourse markers, while the additional conversational data provides controlled coverage of common conversational patterns.

Importantly, the evaluation dataset is constructed independently and does not overlap with the stabilization (fine-tuning) data, ensuring that all reported results reflect generalization rather than memorization.

## Example Entry.

TR: Ben ümitli değilim açıkçası.

EN: I’m not hopeful, to be honest.

:MSA $\lim \limits _ { n  \infty } x \tan = \infty | t |$

$$
\mathbf { S Y } : \tan \alpha . \arcsin \alpha . \arcsin \alpha . \leq \infty . \vdots \ln \alpha . \leq \infty .
$$

This multi-column alignment enables relational comparison between dialectal output, standardized Arabic, and pivot-mediated translations without constraining agents to a single reference target.

## 3.2 Dialect Stabilization Dataset

To model dialect reinforcement as decision-space reweighting, we construct a separate fine-tuning dataset consisting of 5K Turkish–Syrian sentence pairs.

The stabilization corpus combines two sources as shown in Table 1. The television dialogue portion was obtained from professionally dubbed Turkish television scripts provided by NIS Studios, a media production studio specialized in Turkish–Syrian Arabic dubbing.

The MADAR-Turk corpus is used entirely for stabilization (fine-tuning), and no internal split is applied.

This composition enables efective dialect reinforcement while maintaining diversity in conversational structure and avoiding overfitting to a single narrative domain.

Table 1: Dialect Stabilization Dataset Composition
<table><tr><td>Source</td><td>Number of Sentences</td></tr><tr><td>Television Dialogue (NIS)</td><td>3K</td></tr><tr><td>MADAR-Turk Conversational Pairs</td><td>2K</td></tr><tr><td>Total</td><td>5K</td></tr></table>

## 3.3 Fine-Tuning Configuration

Fine-tuning is intentionally lightweight. The objective is not to maximize endtask translation performance but to introduce controlled dialectal reinforcement within the existing multilingual representation space.

The model is fine-tuned on 5K Turkish–Syrian sentence pairs for a single epoch using a learning rate of $1 \times 1 0 ^ { - 5 }$ . No vocabulary extension or architectural modifications are applied, and all agents share identical decoding parameters.

This constrained training regime limits parameter updates substantially, ensuring that observed behavioral shifts primarily reflect lexical reweighting rather than large-scale structural retraining.

## 3.4 Experimental Control and Separation

Strict separation between evaluation and stabilization datasets is enforced. Although both datasets are constructed from similar sources (NIS dialogue and MADAR-Turk), they consist of distinct sentence instances with no overlap at the sentence level.

By controlling architecture, decoding parameters, and dataset separation, translation pathway remains the only varying factor across agents. This controlled setup enables systematic comparison of translation behavior within the proposed decision-space framework.

## 4 Multi-Agent Translation Framework

Neural machine translation systems are often described as mappings of the form $f ( x ) = y$ , where a source sentence x produces a target output y. This representation is a simplifying abstraction. In practice, multilingual generative models define a conditional probability distribution over possible outputs. For a given input x, decoding explores a high-dimensional distribution that contains multiple plausible realizations. The single output returned to the user therefore represents only one trajectory through a broader latent decision landscape.

We interpret this flexibility as a structured translation decision space. In this work, the decision space refers to the set of outputs generated when the same multilingual model is executed under diferent controlled pathway constraints. These outputs represent observable samples from the model’s conditional output distribution.

![](images/c2e9da10ab008c61f0d924120208029d0bf87a06e3b520f0b7cc465d84dc0398.jpg)  
Figure 1: Overview of the proposed decision-space framework. A single multilingual model is executed under three constrained translation pathways (agents): zero-shot, stabilized, and pivot-based. The resulting outputs represent diferent regions of the shared translation decision space.

As illustrated in Figure 1, translation pathways define distinct regions within the shared decision space.

Formally, for a Turkish input sentence $x ,$ the multilingual backbone model defines a conditional distribution $P ( y \mid x )$ over possible Syrian Arabic realizations. Rather than treating this distribution as an opaque sampling mechanism, we introduce controlled translation pathways that modify how decoding explores the distribution.

Each pathway is modeled as a translation agent representing a constrained decoding configuration of the same underlying model. We emphasize that agents in this framework are not fully autonomous entities in the classical MAS sense, but operational configurations defined by distinct decision constraints applied to a shared multilingual system.

Let M denote the fixed multilingual model. An agent $A _ { i }$ maps an input sentence x to an output $y _ { i }$ under a specific operational constraint $C _ { i }$

$$
A _ { i } ( x ) = { \mathcal { M } } ( x ; C _ { i } ) .
$$

The constraint $C _ { i }$ does not modify the architecture of M but specifies the translation pathway under which decoding is performed.

To analyze how translation behavior changes under diferent operational conditions, we examine three representative pathway configurations.

The first agent performs direct zero-shot Turkish→Syrian translation using the pretrained multilingual representation without dialect-specific adaptation.

The second agent introduces dialect stabilization through lightweight finetuning on 5K Turkish–Syrian sentence pairs. The fine-tuning is intentionally minimal so that the multilingual representation is not substantially retrained, but lexical activation patterns encouraging dialectal realization are slightly reinforced.

The third agent implements pivot mediation, decomposing translation into two sequential mappings: Turkish→English followed by English→Syrian. The intermediate English representation acts as a semantic bottleneck that can introduce additional normalization pressure on the final output.

All agents share identical decoding parameters and operate on the same evaluation dataset. The only variation lies in the pathway constraint $C _ { i }$

For each input x, we observe a subset of the translation decision space generated by the selected pathway configurations:

$$
\mathcal { D } ( x ) = \{ A _ { 1 } ( x ) , A _ { 2 } ( x ) , A _ { 3 } ( x ) \} .
$$

This set represents observable realizations under the three controlled pathway constraints rather than the full space of outputs defined by $P ( \boldsymbol { y } | \boldsymbol { x } )$

The elements of ${ \mathcal { D } } ( x )$ are not interpreted hierarchically as correct versus incorrect translations. Instead, they represent neighboring realizations within the same semantic manifold. Divergence between agents therefore reflects behavioral displacement within the model’s latent representation space.

This perspective shifts interpretability from internal probing toward relational analysis of outputs. Instead of analyzing hidden representations directly, we observe how translations change when decoding conditions are modified in a controlled manner [4]. In this framework, divergence across agents becomes an interpretable signal revealing how multilingual models balance dialectal authenticity, normalization pressure, and structural stability.

Importantly, the agents are not independent models but coordinated decoding configurations of the same multilingual system. The multi-agent perspective therefore introduces structured pathway variation rather than additional architectures.

Within low-resource dialect settings, this variation is particularly informative. Instability in zero-shot generation, normalization under pivot mediation, and reinforcement under dialect stabilization correspond to diferent observable regions of the translation decision space, allowing systematic analysis of dialect generation behavior without modifying the underlying model architecture.

## 5 Behavioral Metrics for Structured Divergence

Standard reference-based metrics remain useful for measuring translation adequacy. However, they are insuficient for capturing behavioral variation across translation pathways. In this work, conventional correctness-based evaluation is therefore complemented with behavior-oriented analysis that measures how translation outputs shift under diferent decoding conditions. Instead of evaluating a single output against a reference, we quantify structured divergence within the translation decision space introduced in Section 4.

These metrics are designed to provide interpretable signals at the behavioral level. Dialect marker frequency reflects lexical-level interpretability, lexical overlap captures normalization tendencies, and structural length variation reflects stability in generation behavior.

The selected metrics capture three complementary aspects of translation behavior: dialectal lexical activation, normalization toward standardized Arabic, and structural realization.

Let x denote a Turkish source sentence and $y _ { i } = A _ { i } ( x )$ the output of agent $A _ { i }$

## 5.1 Dialect Marker Frequency

Let M denote a predefined set of Syrian dialect markers curated from the dialogue corpus and validated by native speakers $( \log , \bigcup \limits _ { i \leq 0 } ( \leq L _ { s } ) = \log ( \frac { 2 } { 2 } )$

Dialect activation for a sentence y is defined as:

$$
D ( y ) = { \frac { 1 } { | y | } } \sum _ { t \in y } \mathbf { 1 } ( t \in \mathcal { M } )
$$

Corpus-level activation for agent $A _ { i }$ over dataset $\mathcal { X }$ is:

$$
\overline { { D } } _ { A _ { i } } = \frac { 1 } { | \mathcal { X } | } \sum _ { x \in \mathcal { X } } D ( A _ { i } ( x ) )
$$

Higher values indicate stronger dialect reinforcement in the generated translations.

## 5.2 Lexical Proximity to Standardized Arabic

Lexical proximity to Modern Standard Arabic (MSA) is used to estimate normalization pressure in the translation process. Since Syrian dialect diverges lexically from MSA, stronger overlap with standardized Arabic may indicate a tendency toward normalization rather than dialectal realization.

Given standardized reference $S ( x )$ , lexical overlap is defined as:

$$
O ( y , S ) = { \frac { | { \mathrm { t o k e n s } } ( y ) \cap { \mathrm { t o k e n s } } ( S ( x ) ) | } { | { \mathrm { t o k e n s } } ( S ( x ) ) | } }
$$

Agent-level proximity is computed as:

$$
\overline { { O } } _ { A _ { i } } = \frac { 1 } { | \mathcal { X } | } \sum _ { x \in \mathcal { X } } O ( A _ { i } ( x ) , S ( x ) )
$$

Lower overlap therefore indicates stronger dialectal divergence from standardized Arabic.

## 5.3 Structural Length Ratio

Structural realization is measured through relative sentence length compared with the standardized reference.

$$
L ( y ) = { \frac { \ell ( y ) } { \ell ( S ( x ) ) } }
$$

Corpus mean for agent $A _ { i }$ :

$$
\overline { { L } } _ { A _ { i } } = \frac { 1 } { \vert \mathcal { X } \vert } \sum _ { x \in \mathcal { X } } L ( A _ { i } ( x ) )
$$

Structural variance is computed as:

$$
\sigma _ { L , A _ { i } } = \sqrt { \frac { 1 } { | \mathcal { X } | } \sum _ { x \in \mathcal { X } } ( L ( A _ { i } ( x ) ) - \overline { { L } } _ { A _ { i } } ) ^ { 2 } }
$$

Higher variance indicates instability in structural realization across generated translations.

## 5.4 Inter-Agent Divergence

To quantify behavioral displacement between agents, we measure divergence across the three behavioral dimensions introduced above: dialect activation, lexical normalization, and structural realization.

Relational displacement between agents $A _ { i }$ and $A _ { j }$ is defined as:

$$
\Delta _ { i j } = \left| \overline { { \cal D } } _ { A _ { i } } - \overline { { \cal D } } _ { A _ { j } } \right| + \left| \overline { { \cal O } } _ { A _ { i } } - \overline { { \cal O } } _ { A _ { j } } \right| + \left| \overline { { \cal L } } _ { A _ { i } } - \overline { { \cal L } } _ { A _ { j } } \right|
$$

This score measures behavioral divergence between translation agents under diferent pathway constraints.

The three components are combined with equal weight because they capture complementary aspects of translation behavior and operate on diferent behavioral dimensions rather than comparable numeric scales. Equal weighting therefore avoids privileging one behavioral dimension over the others.

## 6 Quantitative Results

Table 2 reports aggregate behavioral metrics across the three translation agents. Because all agents share the same multilingual backbone, observed diferences reflect pathway-induced decision reweighting rather than architectural variation.

To assess the robustness of observed diferences, we performed paired comparisons across agents using bootstrap resampling over the evaluation set. The observed diferences in dialect marker frequency and length ratios remained stable across resampled subsets, indicating that the reported trends are not driven by dataset-specific efects.

Table 2: Aggregate Behavioral Metrics Across Translation Agents
<table><tr><td colspan="3">Metric Zero-shot Stabilized Pivot</td></tr><tr><td>Avg. Dialect Markers</td><td>0.2266</td><td>0.4988 0.2578</td></tr><tr><td>Avg Overlap with MSA</td><td>0.2133</td><td>0.1117 0.1114</td></tr><tr><td>Std Overlap</td><td>0.2530</td><td>0.1717 0.1788</td></tr><tr><td>Avg Length Ratio</td><td>1.5498</td><td>1.0332 0.9354</td></tr><tr><td>Std Length Ratio</td><td>2.8761</td><td>1.1834 1.0669</td></tr></table>

Dialect stabilization produces the strongest behavioral shift. Average dialect marker frequency increases from 0.2266 in zero-shot translation to 0.4988 after fine-tuning, indicating substantial lexical reweighting toward Syrian dialect forms. At the same time, structural variance decreases sharply $( \sigma _ { L } = 2 . 8 7 6 1 $ 1.1834), demonstrating improved stability in sentence realization. Lexical overlap with standardized Arabic decreases $( 0 . 2 1 3 3  0 . 1 1 1 7 )$ , confirming stronger dialect divergence from MSA.

The pivot pathway exhibits a diferent profile. Dialect activation remains moderate (0.2578), while the average length ratio drops below unity (0.9354), indicating structural compression relative to standardized references. Such compression often reflects normalization efects introduced by pivot mediation, where colloquial expansions or emphatic constructions in Syrian dialect are reduced when passing through an intermediate English representation. Variance is reduced compared to zero-shot but remains slightly higher than stabilized, consistent with semantic mediation through English acting as a regularizing constraint.

As illustrated in Figure 2, the three agents occupy distinct regions within the behavioral space. Zero-shot translation exhibits high variance and limited dialect activation. Stabilization shifts the system toward stronger dialect reinforcement and structural consistency. Pivot translation compresses output length and moderates lexical activation, revealing measurable behavioral diferences consistent with decision-space reconfiguration across translation pathways.

Importantly, qualitative inspection of outputs indicates that increased dialectal activation does not appear to systematically degrade semantic adequacy. While minor lexical deviations occur, the core meaning is generally preserved across agents.

## 7 Discussion: Decision-Space Reconfiguration Across Agents

The empirical results reveal systematic behavioral displacement across translation agents. These shifts are not stochastic variations but structured reconfigurations within a shared multilingual decision space. Because all agents operate over the same backbone architecture, observed diferences reflect pathway-induced reweighting rather than architectural divergence.

Table 3: Representative Sentence-Level Divergence Across Translation Agents
<table><tr><td rowspan=1 colspan=1>Example 1TR: Nasıl olacak? Nasıl yaşayacağız böyle?A1 Zero-shot: $p 1 1 0 0 p \div \cdots 0 5 p 0 p \div 0 . 5$ A2 Stabilized: $p 1 0 0 0 0 0 0 0 c m ^ { 3 } \cdot 0 . 5 p 3 = 0 . 5$ A3 Pivot: $1 1 5 0 \div \cdots 0 5$ Observed Pattern: Register Shift + Addressee Reframing</td></tr><tr><td rowspan=1 colspan=1>Example 2TR: Hadi açın kitaplarınızı. Müzik Müzik Müzik...A1 Zero-shot: $\cdots \sin 5 0 0 \pi , \ldots , \sin 5 0 0 \pi$ A2 Stabilized: $K = S 1 \max$ A3 Pivot: $\cdots 5 3 . 1 1 \div 5 \cdots 5$ Observed Pattern: Token Noise + Contextual Normalization</td></tr><tr><td rowspan=1 colspan=1>Example 3TR: Neredeydi şu bardaklar?A1 Zero-shot: $9 . 9 5 1 1 0 4 0 5 0$ A2 Stabilized: $9 0 6 5 1 1 0 1 6 5 6 \div 8$ A3 Pivot: $\{ 1 , 1 \} = 1 1 \therefore 1 1 \therefore 5 \sin$ Observed Pattern: Lexical Drift / Semantic Misalignment</td></tr><tr><td rowspan=1 colspan=1>Example 4TR: O teyze benim yüzümden sana kızdı değil mi?A1 Zero-shot: $: t \leq ( b - 1 ) ^ { 2 } - 1 0 ^ { 2 } \geq 0 \leq b - 5$ EsA2 Stabilized: $\sin \theta = \cos \theta \sin \theta$ A3 Pivot: $\frac { x } { c } \cdot \frac { b ^ { 2 } } { c } = a \cdot b \cdot c$ Observed Pattern: Role Reassignment + Blame Shift</td></tr><tr><td rowspan=1 colspan=1>Example 5TR: Eski düzen mi?A1 Zero-shot: $\beta _ { i } \beta _ { i } \beta$ A2 Stabilized: $\int \limits _ { 0 } ^ { \infty } f ( x ) d x = \frac { \sqrt { 2 } - 1 } { 2 }$ A3 Pivot: $\int \limits _ { a } ^ { b } f ( x ) d x = | b |$ Observed Pattern: Structural Completion + Nominal Recovery</td></tr><tr><td rowspan=1 colspan=1>Example 6TR: Kabul ediyor musunuz iş teklifimi?A1 Zero-shot: $\int \limits _ { 0 } ^ { \infty } d x \leq 0 \leq c$ A2 Stabilized: $\int \limits _ { a } ^ { b } u ^ { s } u ^ { s } e ^ { - \frac { 1 } { 2 } } d x$ A3 Pivot: $\frac { 2 \sin \theta } { \cos } \cot \theta$ Observed Pattern: Argument Structure Reweighting</td></tr></table>

![](images/34f563eb3453b050d10e1ee6d2ed987f964fc0498d1edda3ffb547a81b217454.jpg)  
Figure 2: Aggregate behavioral comparison across translation agents. Stabilization amplifies dialect marker frequency and reduces structural expansion, while pivot mediation introduces compression and moderate normalization efects.

The zero-shot agent (A1) exhibits the highest structural variance $( \sigma _ { L } =$ 2.8761) and the lowest dialect marker activation (0.2266). This pattern suggests under-specified dialect representation: without explicit reinforcement, the model oscillates between standardized and colloquial realizations, resulting in unstable decoding trajectories. Zero-shot instability therefore reflects representational uncertainty rather than translation failure.

Dialect stabilization (A2) nearly doubles dialect marker frequency (0.4988) while significantly reducing structural variance $\left( \sigma _ { L } \mathrm { ~ = ~ } 1 . 1 8 3 4 \right)$ Notably, this shift emerges under intentionally lightweight tuning—only a single epoch with a low learning rate—highlighting how even minimal dialect exposure can substantially reshape decoding behavior. Given this minimal adaptation regime, the observed change reflects lexical probability redistribution rather than structural retraining. Stabilization strengthens dialectal activation while preserving semantic integrity, demonstrating that low-resource dialect reinforcement primarily operates through probability reweighting within the existing representation space.

The pivot agent (A3) introduces an intermediate English representation and produces the lowest mean length ratio (0.9354) with moderate dialect activation (0.2578). This configuration imposes normalization pressure, frequently compressing emotionally loaded or colloquial constructions. Pivot mediation therefore acts as a regularizing constraint that reduces instability but attenuates dialect intensity.

Taken together, these results confirm that inter-agent divergence reflects controlled displacement in lexical weighting and structural realization. Translation pathways function as mechanisms for exploring neighboring regions of a common semantic manifold. Divergence is thus interpretable signal—revealing how multilingual systems negotiate authenticity, normalization, and mediation under diferent decision constraints.

## 8 Sentence-Level Divergence Analysis

While aggregate metrics capture behavioral displacement at the corpus level, sentence-level inspection reveals how these divergences manifest in concrete translation outputs. Examples were selected from sentences with the highest inter-agent divergence scores $( \Delta _ { i j } )$ as defined in Section 5.4. These cases therefore represent inputs where pathway-induced behavioral displacement across agents is most pronounced.

Table 3 reports the translations produced by the three agents for each selected sentence. Presenting the outputs side-by-side allows direct inspection of how diferent pathway constraints reshape lexical choice, syntactic realization, and semantic interpretation within the same underlying multilingual model.

Across examples, zero-shot outputs frequently exhibit lexical ambiguity, structural under-specification, or inconsistent register realization. Dialect stabilization increases colloquial activation while improving syntactic completeness. In contrast, pivot mediation often compresses emotionally marked constructions and occasionally introduces semantic reinterpretation through intermediate English normalization.

These qualitative observations are consistent with the quantitative patterns reported in Section $6 ,$ confirming that inter-agent diferences reflect structured behavioral variation rather than stochastic decoding noise.

## 9 Implications for Explainable Multi-Agent Systems

The proposed framework generalizes to generative decision systems where behavior can be interpreted through agent-level pathway variation.

Instead of probing internal activations or attention maps, interpretability emerges through relational comparison. Each agent operates under distinct constraints (direct, stabilized, pivot-mediated), and divergence between outputs becomes an explanatory mechanism.

In this context, explainability is not extracted post-hoc. It is constructed structurally through controlled variation of translation pathways. This approach is particularly valuable in low-resource settings, where ambiguity and variability are intrinsic rather than exceptional.

The multi-agent perspective also highlights that multilingual models contain latent decision flexibility. By selectively reinforcing or mediating translation pathways, we expose how lexical and structural weighting shifts within the same architecture.

## 10 Limitations

Several limitations should be acknowledged.

First, the study focuses exclusively on Turkish–Syrian Arabic translation. Although the decision-space framework is conceptually transferable, empirical validation across additional language pairs and dialect continua is required to confirm generalizability.

Second, dialect stabilization was intentionally lightweight (one epoch, low learning rate) to isolate lexical reweighting efects. More extensive adaptation may yield diferent structural dynamics, which were not explored in this study.

Third, the proposed behavioral metrics emphasize divergence rather than adequacy. While this aligns with our interpretability objective, future work may integrate performance-oriented measures to balance authenticity with translation quality.

Finally, lexical overlap calculations may be afected by orthographic variation in Arabic dialect writing. Although preprocessing was controlled, dialect spelling variability remains an inherent challenge in low-resource settings.

## 11 Conclusion

This study introduced a decision-space modeling framework for low-resource dialect translation, reframing translation pathways as constrained agents operating within a shared multilingual representation space. Rather than evaluating translation as a single deterministic output, we analyzed structured behavioral displacement across zero-shot, stabilized, and pivot-mediated configurations.

Empirical results over 5,000 evaluation sentences demonstrated that lightweight dialect stabilization substantially amplifies dialect marker activation while reducing structural instability. In contrast, pivot mediation introduces measurable normalization and compression efects. These systematic shifts confirm that translation pathways induce controlled reweighting within the translation decision space rather than stochastic variation.

By modeling translation as a multi-agent system, we show that inter-agent divergence can serve as an interpretability signal. Behavioral displacement across agents reveals how multilingual models balance dialectal authenticity, structural realization, and semantic mediation under diferent pathway constraints.

Beyond Turkish–Syrian translation, the proposed framework suggests a broader paradigm for analyzing generative systems through controlled pathway variation. Future work may extend this approach to additional dialect continua and generative tasks, further exploring pathway-based modeling as a mechanism for interpretable multilingual AI. More broadly, controlled pathway variation may serve as a practical strategy for explainable multi-agent analysis in generative language systems.

## Acknowledgements

This work was supported by the Luxembourg National Research Fund (FNR) through the project “The Epistemology of AI Systems” (C22/SC/17111440/EAI). The authors also thank NIS Studios for providing dialogue data used in the experiments of this study.

## References

[1] Abbas, M., Lichouri, M., Freihat, A.A.: St madar 2019 shared task: Arabic fine-grained dialect identification. In: Proceedings of the Fourth Arabic Natural Language Processing Workshop. pp. 269–273 (2019)

[2] Alkheder, H., Bouamor, H., Habash, N., Zengin, A.: Benchmarking dialectal arabic-turkish machine translation. In: Proceedings of Machine Translation Summit XIX, Vol. 1: Research Track. pp. 261–271 (2023)

[3] Baniata, L.H., Park, S., Park, S.B.: A neural machine translation model for arabic dialects that utilises multitask learning (mtl). Computational intelligence and neuroscience 2018(1), 7534712 (2018)

[4] Belinkov, Y., Glass, J.: Analysis methods in neural language processing: A survey. Transactions of the Association for Computational Linguistics 7, 49–72 (2019)

[5] Bi, T., Xiong, H., He, Z., Wu, H., Wang, H.: Multi-agent learning for neural machine translation. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). pp. 856– 865 (2019)

[6] Bouamor, H., Habash, N., Salameh, M., Zaghouani, W., Rambow, O., Abdul-Mageed, M., Obeid, O., Shahrour, A., Khalifa, S.: The madar arabic dialect corpus and lexicon. In: Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). European Language Resources Association (ELRA), Miyazaki, Japan (2018)

[7] Dabre, R., Fujita, A., Chu, C.: Exploiting multilingualism through multistage fine-tuning for low-resource neural machine translation. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). pp. 1410–1416 (2019)

[8] Ferrando, J., Gállego, G.I., Alastruey, B., Escolano, C., Costa-jussà, M.R.: Towards opening the black box of neural machine translation: Source and target interpretations of the transformer. In: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. pp. 8756– 8769 (2022)

[9] Gu, J., Wang, Y., Chen, Y., Li, V.O., Cho, K.: Meta-learning for lowresource neural machine translation. In: Proceedings of the 2018 conference on empirical methods in natural language processing. pp. 3622–3631 (2018)

[10] Kim, Y., Gao, Y., Ney, H.: Efective cross-lingual transfer of neural machine translation models without shared vocabularies. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. pp. 1246– 1257 (2019)

[11] Kim, Y., Petrov, P., Petrushkov, P., Khadivi, S., Ney, H.: Pivot-based transfer learning for neural machine translation between non-english languages. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). pp. 866–876 (2019)

[12] Lai, H., Toral, A., Nissim, M.: Multidimensional evaluation for text style transfer using chatgpt. arXiv preprint arXiv:2304.13462 (2023)

[13] Liao, B., Gao, Y., Ney, H.: Multi-agent mutual learning at sentence-level and token-level for neural machine translation. In: Findings of the Association for Computational Linguistics: EMNLP 2020. pp. 1715–1724 (2020)

[14] Luo, G., Yang, Y., Yuan, Y., Chen, Z., Ainiwaer, A.: Hierarchical transfer learning architecture for low-resource neural machine translation. IEEE Access 7, 154157–154166 (2019)

[15] Madsen, A., Meade, N., Adlakha, V., Reddy, S.: Evaluating the faithfulness of importance measures in nlp by recursively masking allegedly important tokens and retraining. In: Findings of the Association for Computational Linguistics: EMNLP 2022. pp. 1731–1751 (2022)

[16] Mohammadshahi, A., Vamvas, J., Sennrich, R.: Investigating multi-pivot ensembling with massively multilingual machine translation models. In: Proceedings of the Fifth Workshop on Insights from Negative Results in NLP. pp. 169–180 (2024)

[17] Neubig, G., Hu, J.: Rapid adaptation of neural machine translation to new languages. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. pp. 875–880 (2018)

[18] Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: Bleu: a method for automatic evaluation of machine translation. In: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL). pp. 311– 318 (2002)

[19] Rei, R., Stewart, C., Farinha, A.C., Lavie, A.: Comet: A neural framework for mt evaluation. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 2685–2702 (2020)

[20] Salloum, W., Habash, N.: Dialectal arabic to english machine translation: Pivoting through modern standard arabic. In: Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. pp. 348–358 (2013)

[21] Shapiro, P., Duh, K.: Comparing pipelined and integrated approaches to dialectal arabic neural machine translation. In: Proceedings of the Sixth Workshop on NLP for Similar Languages, Varieties and Dialects. pp. 214– 222 (2019)

[22] Sibaee, S., Nacar, O., Al-Habashi, Y., Ammar, A., Boulila, W.: Shami-mt: a syrian arabic dialect to modern standard arabic bidirectional machine translation system. arXiv preprint arXiv:2508.02268 (2025)

[23] Talafha, B., Farhan, W., Altakrouri, A., Al-Natsheh, H.: Mawdoo3 ai at madar shared task: Arabic tweet dialect identification. In: Proceedings of the Fourth Arabic Natural Language Processing Workshop. pp. 239–243 (2019)

[24] Voita, E., Sennrich, R., Titov, I.: Analyzing the source and target contributions to predictions in neural machine translation. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). pp. 1126–1140 (2021)

[25] Wang, X., Yin, M.: Are explanations helpful? a comparative study of the efects of explanations in ai-assisted decision-making. In: Proceedings of the 26th International Conference on Intelligent User Interfaces. pp. 318– 328 (2021)

[26] Wu, M., Xu, J., Yuan, Y., Hafari, G., Wan, L., Luo, W., Zhang, K.: (perhaps) beyond human translation: Harnessing multi-agent collaboration for translating ultra-long literary texts. Transactions of the Association for Computational Linguistics 13, 901–922 (2025)