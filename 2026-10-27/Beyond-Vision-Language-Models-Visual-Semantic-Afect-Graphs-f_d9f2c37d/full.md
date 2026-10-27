# Beyond Vision-Language Models: Visual-Semantic Afect Graphs for Emotion Recognition in Artworks

Ngoc-Duy Mai<sup>1,2,3[0009−0008−8525−6238]⋆</sup>, Ha-Thanh Nguyen<sup>4[0000−0003−2794−7010]</sup>, and Hideaki Takeda<sup>1,2[0000−0002−2909−7163]</sup>

<sup>1</sup> National Institute of Informatics, Tokyo, Japan <sup>2</sup> The Graduate University for Advanced Studies, SOKENDAI, Kanagawa, Japan 3 Research and Development Center for Large Language Models, NII, Tokyo, Japan 4 College of Engineering & Computer Science, VinUniversity, Hanoi, Vietnam duymn@nii.ac.jp

Abstract. Emotion recognition in artworks remains challenging because afective meaning is often subjective, implicit, and shaped by relationships between visual form and semantic associations. While vision-language models provide strong multimodal representations, direct image-to-emotion prediction may overlook explicit relations among formal, semantic, and afective cues that support emotional interpretation. This paper proposes Visual-Semantic Afect Graphs (VSAG), a graph-based framework that represents artworks through interpretable feature nodes, including color, brushstroke, composition, light-shadow, line quality, and extracted visual entities. The graph learns feature-emotion associations for dominant emotion prediction, with valence-arousal structure modeled through an optional hierarchical auxiliary afect module. We evaluate VSAG and VSAG-TopK against direct VLM inference using Qwen3-VL and InternVL3.5 on EmoArt-5k. Results show that direct VLMs provide useful but modest baselines, while graph-based models improve dominant emotion prediction. In particular, the hierarchical valence-arousal variant of VSAG achieves the best overall performance, improving accuracy, macro-F1, and weighted-F1 over both direct VLM inference and the emotiononly VSAG baseline. Per-class results indicate that graph-based models and VLMs capture partly complementary signals, while valence-arousal structure is most useful when integrated as a coarse-to-fine afect hierarchy rather than as a simple post-hoc correction. These findings suggest that explicit visual-semantic structure can support both prediction and interpretation in artwork emotion recognition.

Keywords: Art · Emotion Recognition · Visual-Semantic Graph · Vision-Language Models · Afective Computing

## 1 Introduction

“A work of art which did not begin in emotion is not art.”

Paul Cézanne

<sup>⋆</sup> Corresponding author.

This idea is not only an aesthetic statement, but also represents a potential computational challenge. If emotion is central to artistic meaning, then an artificial intelligence system that merely recognizes depicted objects, styles, or iconographic motifs has not yet understood an artwork in the way human viewers often do. Computationally, this requires connecting visual form to annotated afective response: how color, composition, texture, rhythm, light, abstraction, and symbolic association may contribute to how an artwork is labeled as feeling. In this sense, emotion prediction in art asks whether a model can move beyond visual description toward a deeper interpretation of aesthetic experience.

Vision-language models have shown strong performance in general visuallanguage understanding, suggesting that large-scale image-text pretraining can capture broad semantic knowledge from images [12]. However, afective understanding remains more dificult than recognizing objects, actions, or scenes. Human emotions are often implicit, context-dependent, subjective, and expressed through subtle visual or linguistic cues rather than directly observable categories. Recent studies suggest that VLMs can struggle with such afective reasoning, showing sensitivity to prompts and labels, bias toward frequent emotion categories, and dificulty distinguishing rare or complex emotions from more common neighboring ones [4, 11, 2]. These limitations suggest that strong visual-language alignment alone may not be suficient for robust emotion understanding.

This challenge becomes especially important in the study of artworks. Unlike many ordinary images, artworks evoke emotion not only through depicted objects or scenes, but also through formal and aesthetic components, like color, texture, composition, light, line, brushwork, spatial balance, and style. Psychological and neuroaesthetic accounts of art perception have long argued that aesthetic experience involves emotion, valuation, memory, and interpretation rather than visual recognition alone [18, 15]. Earlier computational studies on afective image and art analysis similarly emphasized that emotion prediction benefits from features motivated by psychology and art theory, including color, texture, composition, and semantic content [14]. These observations suggest that progress in artwork emotion prediction may require not only larger models, but also intermediate representations that make afect-relevant visual and semantic cues explicit.

In this paper, we examine whether graph-based visual-semantic representations ofer a more efective way to model emotion in art than direct VLM inference. Instead of treating an artwork as an isolated image, we represent it as an artwork-feature graph whose nodes encode afect-relevant attributes such as color, brushstroke, composition, light, line quality, and visual entities. The graph supports dominant emotion prediction, with valence-arousal group structure modeled through an optional hierarchical auxiliary module. We address three research questions:

RQ1: How well do current vision-language models predict the emotions evoked by artworks?

RQ2: Can graph-based visual-semantic representations improve artwork emotion prediction beyond direct VLM inference?

RQ3: Which visual-semantic attributes are most strongly associated with specific emotions?

The rest of this paper is organized as follows: Section 2 reviews some related works. Section 3 introduces the proposed VSAG framework, including graph construction, VSAG-TopK, hierarchical valence-arousal auxiliary modeling, and training. Section 4 presents the dataset, emotion prediction results, and featureemotion association analysis. Section 5 discusses the main findings and limitations. Finally, Section 6 concludes the paper and outlines future directions.

## 2 Related Work

Research on emotion in visual art draws from both categorical and dimensional theories of afect. Discrete emotion models [9] describe afect through labels such as joy, sadness, fear, or anger, while dimensional models [19] represent emotion through properties such as valence and arousal. This distinction is important for art because artworks often evoke emotion indirectly, through visual form, symbolic meaning, personal association, and cultural knowledge. Art-specific datasets such as WikiArt Emotions [16], ArtEmis [1], and FeelingBlue [3] have enabled larger-scale study of evoked emotion in artworks. However, recent AI and-art research has increasingly focused on generative tasks, especially text-toimage generation, image editing, and difusion-based visual art creation [20, 24]. While these works show rapid progress in producing or editing artistic images, emotion prediction in art remains comparatively less explored. This gap is important because art emotion recognition is not only a labeling task, but also an interpretive problem involving visual form, semantic association, and afective response.

Vision-language models have become strong foundations for multimodal understanding, including image-text retrieval, captioning, visual question answering, and multimodal reasoning [12, 13, 22]. However, strong general visual-language performance does not guarantee reliable afective understanding. Emotion is subjective, context-dependent, and often weakly grounded in visible objects. Prior work suggests that VLM predictions can be sensitive to prompts and label formulations, and that generic visual-language representations are not always aligned with human afective judgments [6, 4]. This limitation is especially relevant for art, where emotional meaning may depend on relations among color, form, composition, and symbolic content.

Graph representations provide a way to make visual and semantic structure explicit. In computer vision, scene graphs and knowledge graphs organize objects, attributes, relations, and contextual knowledge for visual reasoning [10]. Similar ideas have been used in computational art analysis, where resources such as ArtGraph [7] and ArtKB [5] show that artwork understanding can benefit from structured information beyond pixels. Graph learning methods further provide tools for using relational structure in prediction [21, 17]. However, it remains unclear whether graph-based learning over visual-semantic artwork attributes improves emotion prediction. Our work addresses this question by constructing an afect-focused artwork-feature graph and comparing it with direct VLM inference.

## 3 Method

![](images/aa08bd4baf05b4ae4bf806e84a05ee7fab96b32bba87d9ce3567a249d8008a14.jpg)  
Fig. 1. Overview of the proposed Visual-Semantic Afect Graph framework.

Figure 1 gives an overview of the proposed framework. We formulate artwork emotion prediction as structured prediction over a visual-semantic afect graph. Rather than directly mapping an image to an emotion label, we first represent each artwork through interpretable visual-semantic features and then learn how these features support dominant emotion prediction. The emotiononly model decodes the final emotion directly from graph-based emotion scores. The +VA variant uses the valence-arousal structure of the emotion label space as a coarse-to-fine hierarchy: the model predicts a four-way valence-arousal group and a within-group emotion distribution. In VSAG+VA, these outputs provide the hierarchical final decoder; in $\mathrm { V S A G \mathrm { - } T o p K \mathrm { + } V A }$ , they are used as auxiliary regularization while final decoding remains direct.

## 3.1 Problem Setup and Graph Construction

Let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ^ { e m o } , y _ { i } ^ { g r p } ) \} _ { i = 1 } ^ { N }$ denote a dataset of artworks, where $x _ { i }$ is an artwork image, $y _ { i } ^ { e m o }$ is its dominant emotion label, and $y _ { i } ^ { g r p }$ is the valence-arousal group associated with that emotion. The main prediction target is $y _ { i } ^ { e m o }$ . Following the circumplex view of afect [19], $y _ { i } ^ { g r p }$ is used to define the optional hierarchical auxiliary variant. For each artwork, we construct a visual-semantic feature representation from two natural-language sources: visual attribute descriptions and the general image description.

Table 1. Rule-based visual attribute categories used to construct feature nodes.
<table><tr><td rowspan=1 colspan=1>Attribute</td><td rowspan=1 colspan=1>Controlled category groups</td></tr><tr><td rowspan=1 colspan=1>Brushstroke</td><td rowspan=1 colspan=1>Fine / Delicate; Detailed / Intricate; Smooth / Blended; Soft / Gentle; RoughTextured; Expressive / Energetic; Movement / Flowing; Bold / Thick.</td></tr><tr><td rowspan=1 colspan=1>Color</td><td rowspan=1 colspan=1>Monochrome / Grayscale; Limited / Minimal Palette; High Contrast; Warm-Cool Contrast; Warm Colors; Cool Colors; High / Low Saturation; Vibrant /Intense; Muted / Subdued; Pastel; Dark / Bright Palette; Natural / Earthy;Complementary Colors; Harmonious / Balanced.</td></tr><tr><td rowspan=1 colspan=1>Composition</td><td rowspan=1 colspan=1>Balanced; Symmetrical; Asymmetrical; Central Focus; Multiple Focal Points;Dynamic; Flowing; Layered / Depth; Open Composition; Crowded / Dense; Scat-tered; Structured / Geometric; Minimalist; Chaotic / Tense; Intimate / Close.</td></tr><tr><td rowspan=1 colspan=1>Light/shadow</td><td rowspan=1 colspan=1>No Light Source / Abstract Lighting; Flat / Reduced Depth; Soft / DiffusedLighting; Harsh / Dramatic Lighting; Directional Lighting; Central Light Source;Ambient / Indirect Lighting; Strong / Subtle Contrast; Deep / Minimal / SoftDefined Shadows; Illuminated / Highlighted; Adds Depth / Dimension; SoftTransitions.</td></tr><tr><td rowspan=1 colspan=1>Line</td><td rowspan=1 colspan=1>Blurred / Indistinct; Defined / Distinct; Thin / Fine; Thick / Bold; Delicate /Subtle; Smooth; Rough / Jagged; Firm / Decisive; Continuous / Broken; FlowingFluid; Curved / Organic; Precise / Controlled; Expressive / Energetic; AngularGeometric; Structured; Elegant / Refined.</td></tr></table>

Attribute Categorization Artwork images can often be described through basic visual elements such as color, line, texture, composition, and light. These elements provide a structured description of how the image is visually constructed, and they are important cues for afective interpretation. We first convert visual attribute descriptions into controlled categorical feature nodes. Specifically, we categorize brushstroke, color, composition, light-shadow, and line-quality descriptions using predefined lexical rules. This reduces noisy free-text descriptions into a smaller and more consistent feature vocabulary while preserving interpretable visual cues. Each matched category becomes a feature node connected to the corresponding artwork node.

The implementation also includes fallback categories such as Other and Unknown / Not Applicable when no rule matches or when the attribute is absent.

Entity Extraction Beyond formal attributes, an artwork may contain depicted entities or visual resemblances that provide semantic cues, especially in abstract art. These cues may include objects, motifs, scene elements, or concrete resemblances such as figure, boat, tree, or circle. Because such terms are openvocabulary, we use a text LLM to extract depicted things from the artwork description while excluding formal visual-analysis terms. The prompt is abbreviated below:

```jsonl
Rules: return lowercase singular noun phrases; remove color/style
adjectives unless essential; return [] if no concrete entity exists.
Output JSON only:
{"entities": ["entity one", "entity two"]}
```

We also apply lightweight post-processing to parse the JSON output, remove generic or formal-analysis terms, and deduplicate repeated entities.

Graph Definition We define a graph $G = ( V , E )$ , where the node set consists of artwork nodes, feature nodes, and afect label nodes $V = V _ { A } \cup V _ { F } \cup V _ { Y }$ Here, $V _ { A }$ is the set of artwork nodes, $V _ { F }$ is the set of visual-semantic feature nodes, and $V _ { Y }$ contains afect label nodes. In the emotion-only model, these are dominant emotion nodes. In the hierarchical auxiliary variant, $V _ { Y }$ also includes valence-arousal group labels and within-group emotion labels.

The observed graph contains artwork-feature edges $( a _ { i } , f _ { j } ) \in E _ { A F }$ , where an edge exists if feature $f _ { j }$ is observed in artwork $a _ { i }$ . The feature neighborhood of artwork $a _ { i }$ is $\mathcal { N } ( a _ { i } ) \stackrel { \cdot } { = } \{ f _ { j } : ( a _ { i } , f _ { j } ) \in E _ { A F } \}$ . The model learns feature-label associations:

$$
f _ { j }  y _ { c } ^ { ( t ) } ,
$$

where t denotes a prediction head, such as the direct emotion head, the valencearousal group head, or the within-group emotion head, and $y _ { c } ^ { ( t ) }$ is a candidate label for that head. Therefore, each prediction can be decomposed into explicit graph paths:

$$
a _ { i }  f _ { j }  y _ { c } ^ { ( t ) } .
$$

and $y _ { c } ^ { ( t ) }$ is a candidate label for that task.

## 3.2 Visual-Semantic Afect Graph

We propose a Visual-Semantic Afect Graph (VSAG) model to predict afect labels by aggregating paths from an artwork node through its observed feature nodes to afect label nodes. To initialize feature-label associations, we estimate a smoothed log-odds score from the training set. For each feature $f _ { j }$ and label $y _ { c } ^ { ( t ) }$ , we compute

$$
s _ { j , c } ^ { ( t ) } = \log \frac { P ( y _ { c } ^ { ( t ) } \mid f _ { j } ) } { P ( y _ { c } ^ { ( t ) } \mid \neg f _ { j } ) } .
$$

This score provides an initial estimate of the feature’s association with an afect label. For example, if cool color frequently appears in artworks labeled as Calm, its initial feature-label score for Calm is positive. These scores are then treated as learnable parameters and updated during training.

For each artwork ${ { a } _ { i } } ,$ candidate label $y _ { c } ^ { ( t ) }$ , and observed feature $f _ { j } \in \mathcal { N } ( a _ { i } )$ VSAG computes a label-conditioned attention weight. Let $\mathbf { h } _ { j }$ denote the representation of feature node $f _ { j }$ , constructed from its feature embedding and relationtype embedding. Let $\mathbf { u } _ { c } ^ { ( t ) }$ denote the embedding of label $y _ { c } ^ { ( t ) }$ . The compatibility between feature $f _ { j }$ and label $y _ { c } ^ { ( t ) }$ is computed as

$$
g _ { j , c } ^ { ( t ) } = \mathbf { w } ^ { \top } \operatorname { t a n h } ( \mathbf { h } _ { j } + \mathbf { u } _ { c } ^ { ( t ) } ) + s _ { j , c } ^ { ( t ) } .
$$

The label-conditioned attention weight is then

$$
\alpha _ { i , j , c } ^ { ( t ) } = \frac { \exp ( g _ { j , c } ^ { ( t ) } ) } { \sum _ { f _ { k } \in \mathcal { N } ( a _ { i } ) } \exp ( g _ { k , c } ^ { ( t ) } ) } .
$$

The graph score for label $y _ { c } ^ { ( t ) }$ is computed by aggregating artwork-featurelabel paths:

$$
z _ { i , c } ^ { ( t ) } = \sum _ { f _ { j } \in \mathcal { N } ( a _ { i } ) } \alpha _ { i , j , c } ^ { ( t ) } s _ { j , c } ^ { ( t ) } + b _ { c } ^ { ( t ) } ,
$$

where $b _ { c } ^ { ( t ) }$ is a task-specific label bias.

A small residual adjacency encoder can optionally be added:

$$
\tilde { z } _ { i , c } ^ { ( t ) } = z _ { i , c } ^ { ( t ) } + \lambda r _ { i , c } ^ { ( t ) } ,
$$

where $r _ { i , c } ^ { ( t ) }$ is produced from the artwork-feature adjacency vector and λ controls the residual strength. The residual component is used only as a correction, while the primary prediction mechanism remains the explicit graph path score.

VSAG therefore assumes that afective meaning is distributed across the observed visual-semantic features. Each feature can contribute to each candidate afect label, with its contribution determined by both its feature-label edge score and its label-conditioned attention weight.

## 3.3 VSAG-TopK

We also test a variant called VSAG-TopK. It uses the same visual-semantic afect graph as VSAG, but changes how evidence is aggregated. While VSAG aggregates over all observed feature paths, VSAG-TopK assumes that an artwork’s emotion may be determined by a small number of highly informative visual-semantic cues.

For each path $a _ { i }  f _ { j }  y _ { c } ^ { ( t ) }$ , VSAG-TopK computes a path score:

$$
p _ { i , j , c } ^ { ( t ) } = s _ { j , c } ^ { ( t ) } + \delta q _ { j , c } ^ { ( t ) } ,
$$

where $s _ { j , c } ^ { ( t ) }$ is the learned feature-label edge score, $q _ { j , c } ^ { ( t ) }$ is a contextual compatibility score between feature $f _ { j }$ and label $y _ { c } ^ { ( t ) }$ , and δ controls the strength of the contextual component.

Instead of computing an attention-weighted sum over all observed feature paths, VSAG-TopK selects the strongest k paths for each candidate label. Let $\mathrm { T o p K } ( \mathcal { N } ( a _ { i } ) , c )$ denote the set of k observed features with the largest path scores for label $y _ { c } ^ { ( t ) }$ . The label score is then

$$
z _ { i , c } ^ { ( t ) } = \frac { 1 } { k } \sum _ { f _ { j } \in \mathrm { T o p K } ( \mathcal { N } ( a _ { i } ) , c ) } p _ { i , j , c } ^ { ( t ) } + b _ { c } ^ { ( t ) } .
$$

Thus, VSAG and VSAG-TopK share the same graph structure but use different aggregation assumptions. VSAG treats afective meaning as distributed across all observed visual-semantic paths, while VSAG-TopK emphasizes the strongest feature-label paths.

## 3.4 Hierarchical Valence-Arousal Auxiliary Modeling

In addition to the emotion-only model, we evaluate a hierarchical valence-arousal auxiliary variant. The motivation is that the emotion label space has a known coarse-to-fine structure: each dominant emotion belongs to a valence-arousal group defined by the dataset label scheme. Let Q denote the set of valencearousal groups, and let $q ( e ) \in \mathcal { Q }$ denote the group associated with emotion e. The full emotion-to-group mapping is given in Section 4.1.

The model computes three graph-based outputs. First, it computes a direct emotion distribution

$$
p _ { i } ^ { d i r } ( e ) = \mathrm { s o f t m a x } ( z _ { i } ^ { e m o } ) _ { e } .
$$

Second, it computes a valence-arousal group distribution

$$
p _ { i } ^ { g r p } ( q ) = \mathrm { s o f t m a x } ( z _ { i } ^ { g r p } ) _ { q } .
$$

Third, for each valence-arousal group, it computes a within-group emotion distribution over the emotions belonging to that group:

$$
p _ { i } ^ { w i t h i n } ( e | q ( e ) ) = \frac { \exp ( z _ { i } ^ { w i t h i n } ( e ) ) } { \sum _ { e ^ { \prime } \in \mathcal { E } _ { q ( e ) } } \exp ( z _ { i } ^ { w i t h i n } ( e ^ { \prime } ) ) } ,
$$

where $\mathcal { E } _ { q }$ is the set of emotions in group q.

The hierarchical emotion distribution is then defined as

$$
p _ { i } ^ { h i e r } ( e ) = p _ { i } ^ { g r p } ( q ( e ) ) p _ { i } ^ { w i t h i n } ( e \mid q ( e ) ) .
$$

Unlike a simple additive valence-arousal decoder, this formulation uses valencearousal structure as a coarse-to-fine hierarchy: the model first estimates the affective group and then selects an emotion within that group. For example, a Positive-Low prediction narrows the space to emotions such as Calm, Contentment, and Glad.

For the emotion-only variant, the final prediction is obtained from $p _ { i } ^ { d i r }$ . For the reported $\mathrm { V S A G + V A }$ variant, the final prediction is obtained from the hierarchical distribution $p _ { i } ^ { h i e r }$ . For $\mathrm { V S A G - T o p K + V A }$ , the valence-arousal hierarchy is used only as an auxiliary regularizer during training, while final decoding remains direct.

## 3.5 Training and Inference

The emotion-only model is trained with the dominant emotion loss: $\mathcal { L } = \mathcal { L } _ { e m o }$ The hierarchical +VA variant adds auxiliary losses for the valence-arousal group and within-group emotion prediction:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { e m o } \mathcal { L } _ { e m o } + \lambda _ { g r p } \mathcal { L } _ { g r p } + \lambda _ { w i t h i n } \mathcal { L } _ { w i t h i n } + \lambda _ { c o n s } \mathcal { L } _ { c o n s } . } \end{array}
$$

Here, $\mathcal { L } _ { g r p }$ supervises the four-way valence-arousal group prediction, and $\mathcal { L } _ { w i t h i n }$ supervises emotion prediction within the gold group. The consistency term $\mathcal { L } _ { c o n s }$ encourages the predicted group distribution to agree with the group distribution implied by the emotion probabilities. Setting $\lambda _ { g r p } = \lambda _ { w i t h i n } = \lambda _ { c o n s } = 0$ recovers the emotion-only model.

During training, the model learns feature embeddings, relation embeddings, afect label embeddings, and feature-label edge scores. For the hierarchical auxiliary variant, the model jointly optimizes the direct emotion head, the valencearousal group head, and the within-group emotion head. Thus, valence-arousal information is not used as a post-hoc additive logit correction; instead, it provides a coarse-to-fine afect structure for regularization or hierarchical decoding.

At inference time, the emotion-only model predicts the emotion with the highest direct graph-based emotion probability. The hierarchical +VA variant additionally predicts a valence-arousal group and a within-group emotion distribution. For VSAG+VA, the final emotion is decoded from the hierarchical distribution $p _ { i } ^ { h i e r } ( e )$ . For VSAG-TopK+VA, the auxiliary valence-arousal losses are used during training, while the final prediction is decoded from the direct emotion head.

Because prediction is based on explicit graph paths, the model can provide path-level explanations. For VSAG, the contribution of feature $f _ { j }$ to label $y _ { c } ^ { ( t ) }$ is $\alpha _ { i , j , c } ^ { ( t ) } s _ { j , c } ^ { ( t ) }$ . For VSAG-TopK, the explanation consists of the highest-scoring feature-label paths selected by the TopK pooling operation. These explanations identify which visual-semantic attributes most strongly support a predicted emotion or valence-arousal group label.

## 4 Results

## 4.1 Dataset

We use EmoArt-5k, a publicly available subset<sup>5</sup> of the EmoArt dataset [23]. The dataset contains abstract artwork images paired with natural-language descriptions, visual attribute annotations, and afective labels. The dominant emotion labels follow the valence-arousal grouping used by EmoArt:

(Positive, Low) : {Calm (Cal), Contentment (Con), Glad (Gla)},

(Positive, High) : {Excited (Exc), Happy (Hap), Aroused (Aro)},

(Negative, Low) : {Sad (Sad), Bored (Bor), Tired (Tir)},

(Negative, High) : {Alarmed (Alm), Frustrated (Fru), Annoyed (Ann)}.

Here, we use the dominant emotion label as the main prediction target, while the corresponding valence-arousal groups are used only in auxiliary variants.

The label distribution is highly imbalanced, as shown in Table 2. The dataset is dominated by the Calm class, which accounts for more than half of all artworks. In contrast, several emotion classes contain only a small number of examples, such as Glad, Aroused, Bored, Tired, and Annoyed. This imbalance makes the task challenging because a model can obtain high overall accuracy by favoring frequent emotions, while still performing poorly on rare afective categories. For this reason, we report both distribution-sensitive metrics, such as accuracy and weighted-F1, and class-balanced metrics, such as macro-F1.

After applying the visual attribute categorization and entity extraction steps described in Section 3, we obtain the visual-semantic feature statistics summarized in Table 3. Each artwork contains a short description of about 36 words on average. The visual attribute fields produce compact feature sets, with roughly 2 to 3 categorical nodes per attribute type. The entity extractor adds a more openended semantic layer, contributing 4.06 normalized entity nodes per artwork on average, and 5,281 unique normalized visual entity nodes over the dataset.

Table 2. Distribution of dominant emotion labels by valence-arousal group.
<table><tr><td>Emotion</td><td>|VA group</td><td>Count</td><td>Emotion</td><td>VA group</td><td>|Count</td></tr><tr><td>Calm</td><td>Positive-Low</td><td>2872</td><td>Sad</td><td>Negative-Low</td><td>216</td></tr><tr><td>Contentment</td><td>Positive-Low</td><td>692</td><td>Bored</td><td>Negative-Low</td><td>25</td></tr><tr><td>Glad</td><td>Positive-Low</td><td>2</td><td>Tired</td><td>Negative-Low</td><td>19</td></tr><tr><td>Excited</td><td>Positive-High</td><td>1240</td><td>Alarmed</td><td>Negative-High</td><td>207</td></tr><tr><td>Happy</td><td>Positive-High</td><td>50</td><td>Frustrated</td><td>Negative-High</td><td>157</td></tr><tr><td>Aroused</td><td>Positive-High</td><td>24</td><td>Annoyed</td><td>Negative-High</td><td>28</td></tr></table>

Table 3. Dataset feature statistics. For description length and feature fields, values are reported as mean, minimum, and maximum per artwork.
<table><tr><td rowspan="2">Field</td><td colspan="3">Train</td><td colspan="3">Test</td><td colspan="3">All</td></tr><tr><td>Mean</td><td>Min</td><td>Max</td><td>Mean</td><td>Min</td><td>Max</td><td>Mean</td><td>Min</td><td>Max</td></tr><tr><td>Description words</td><td>35.73</td><td>24</td><td>53</td><td>35.66</td><td>28</td><td>56</td><td>35.72</td><td>24</td><td>56</td></tr><tr><td>Brushstroke categories</td><td>1.97</td><td>1</td><td>5</td><td>1.97</td><td>1</td><td>4</td><td>1.97</td><td>1</td><td>5</td></tr><tr><td>Color categories</td><td>2.87</td><td>1</td><td>7</td><td>2.87</td><td>1</td><td>7</td><td>2.87</td><td>1</td><td>7</td></tr><tr><td>Composition categories</td><td>2.01</td><td>1</td><td>5</td><td>1.97</td><td>1</td><td>5</td><td>2.00</td><td>1</td><td>5</td></tr><tr><td>Light-shadow categories</td><td>2.17</td><td>1</td><td>7</td><td>2.20</td><td>1</td><td>6</td><td>2.18</td><td>1</td><td>7</td></tr><tr><td>Line categories</td><td>2.91</td><td>1</td><td>7</td><td>2.95</td><td>1</td><td>6</td><td>2.92</td><td>1</td><td>7</td></tr><tr><td>Visual entities</td><td>4.07</td><td>0</td><td>15</td><td>4.00</td><td>0</td><td>11</td><td>4.06</td><td>0</td><td>15</td></tr><tr><td>Artworks</td><td colspan="7">4426</td><td colspan="2"></td></tr><tr><td>Unique visual entities</td><td colspan="7">4547</td><td colspan="2">5532 5281</td></tr></table>

## 4.2 Emotion Prediction Performance

We evaluate direct vision-language model inference using Qwen3-VL and InternVL3.5 at multiple model sizes. For these baselines, the model is given the artwork image and prompted to select the dominant emotion label directly. This setting evaluates whether current VLMs can infer evoked emotion from the image without the explicit visual-semantic graph structure used by our method.

Table 4 reports overall dominant emotion classification performance. For RQ1, direct VLM inference provides a useful but still modest baseline. The best direct VLM is InternVL3.5-8B, with accuracy and weighted-F1 only slightly above 0.5. This result suggests that current VLMs can capture some broad afective cues, but artwork emotion prediction remains dificult because the emotional meaning of art is often implicit, subjective, and grounded in subtle relations among form, composition, and semantic association. This dificulty is not surprising, since interpreting emotion in art is challenging even for human viewers and often depends on context and personal perception.

The VLM results also show that larger parameter size does not necessarily lead to better emotion prediction. For example, Qwen3-VL-8B performs worse than its 2B and 4B variants on overall accuracy. One possible explanation is that direct emotion prediction is sensitive to prompt interpretation, label priors, and the model’s alignment behavior, rather than only to raw model capacity. Larger models may produce more complex or language-prior-driven judgments, but those judgments do not always match the true emotion.

Table 4. Dominant emotion prediction performance. Best values are shown in bold.
<table><tr><td>Model</td><td>Acc.</td><td>Macro-F1</td><td>Weighted-F1</td></tr><tr><td>Qwen3-VL-8B</td><td>0.480</td><td>0.168</td><td>0.402</td></tr><tr><td>Qwen3-VL-4B</td><td>0.507</td><td>0.158</td><td>0.433</td></tr><tr><td>Qwen3-VL-2B</td><td>0.505</td><td>0.159</td><td>0.430</td></tr><tr><td>InternVL3.5-8B</td><td>0.549</td><td>0.206</td><td>0.536</td></tr><tr><td>InternVL3.5-4B</td><td>0.403</td><td>0.136</td><td>0.413</td></tr><tr><td>InternVL3.5-2B</td><td>0.353</td><td>0.130</td><td>0.374</td></tr><tr><td>VSAG</td><td>0.577</td><td>0.187</td><td>0.571</td></tr><tr><td>VSAG+VA</td><td>0.591</td><td>0.214</td><td>0.580</td></tr><tr><td> $\mathbf { V S A G - T o p K }$ </td><td>0.579</td><td>0.201</td><td>0.569</td></tr><tr><td> $\mathbf { V S A G - T o \bar { p } K + V A }$ </td><td>0.554</td><td>0.193</td><td>0.552</td></tr></table>

Table 5. Per-class F1 scores for dominant emotion prediction. Best values are shown in bold.
<table><tr><td>Model</td><td>Cal</td><td>Con</td><td>Gla</td><td>Exc</td><td>Hap</td><td>Aro</td><td>Sad</td><td>Bor</td><td>Tir</td><td>Alm</td><td>Fru</td><td>Ann</td></tr><tr><td>Qwen3-VL-8B</td><td>0.735</td><td>0.000</td><td>0.000</td><td>0.015</td><td>0.353</td><td>0.024</td><td>0.339</td><td>0.133</td><td>0.067</td><td>0.290</td><td>0.056</td><td>0.000</td></tr><tr><td>Qwen3-VL-4B</td><td>0.735</td><td>0.000 0.000 0.148</td><td></td><td></td><td>0.145</td><td>0.030</td><td>0.376</td><td>0.100</td><td>0.000</td><td>0.359</td><td></td><td>0.000 0.000</td></tr><tr><td>Qwen3-VL-2B</td><td>0.701</td><td>0.000 0.000 0.188</td><td></td><td></td><td>0.110</td><td>0.000</td><td>0.360</td><td>0.000</td><td>0.000</td><td>0.459</td><td></td><td>0.089 0.000</td></tr><tr><td>InternVL3.5-8B</td><td>0.713</td><td>0.000 0.000 0.626</td><td></td><td></td><td>0.263</td><td>0.091</td><td>0.287</td><td>0.000</td><td>0.071</td><td>0.371</td><td></td><td>0.051 0.000</td></tr><tr><td>InternVL3.5-4B</td><td>0.646</td><td>0.0000.000</td><td></td><td>0.299</td><td>0.250</td><td>0.053</td><td>0.192</td><td>0.000</td><td>0.000</td><td>0.190</td><td></td><td>0.0000.000</td></tr><tr><td>InternVL3.5-2B</td><td>0.552</td><td>0.000</td><td>0.000</td><td>0.320</td><td>0.124</td><td>0.027</td><td>0.189</td><td>0.000</td><td>0.000</td><td>0.297</td><td>0.049</td><td>0.000</td></tr><tr><td>VSAG</td><td>0.734</td><td>0.280 0.000</td><td></td><td>0.624</td><td>0.095</td><td>0.000</td><td>0.158</td><td>0.000</td><td>0.000</td><td>0.232</td><td>0.125</td><td>0.000</td></tr><tr><td>VSAG+VA</td><td>0.733</td><td>0.272 0.000</td><td></td><td>0.624</td><td>0.118</td><td>0.000</td><td>0.247</td><td>0.000</td><td>0.000</td><td>0.317</td><td></td><td>0.2540.000</td></tr><tr><td>VSAG-TopK</td><td>0.734</td><td>0.252 0.000 0.607</td><td></td><td></td><td>0.182</td><td>0.000</td><td>0.164</td><td>0.000</td><td>0.000</td><td>0.361</td><td></td><td>0.118 0.000</td></tr><tr><td>VSAG-TopK+VA</td><td>0.706 0.259 0.000 0.596</td><td></td><td></td><td></td><td>0.087</td><td>0.000</td><td>0.143</td><td>0.000</td><td>0.000</td><td>0.286</td><td></td><td>0.239 0.000</td></tr></table>

For RQ2, the graph-based models generally outperform direct VLM inference on accuracy and weighted-F1. The strongest overall model is VSAG+VA, which achieves the best score across all three metrics. Compared with both the best direct VLM and the emotion-only VSAG baseline, this result shows that hierarchical valence-arousal modeling can improve dominant emotion prediction when it is integrated as a coarse-to-fine afect structure rather than as a post-hoc logit correction. The efect of VA is not uniform across graph aggregation strategies. While VSAG+VA improves over VSAG, VSAG-TopK+VA underperforms VSAG-TopK. This suggests that the VA hierarchy is more useful when afective evidence is aggregated across distributed feature paths, whereas sparse TopK pooling may already impose a strong evidence-selection bias.

Table 5 shows that the gains are concentrated in several negative or dificult classes. Compared with emotion-only VSAG, VSAG+VA improves Alarmed, Frustrated, and Sad, which contributes to its higher macro-F1. However, rare classes such as Glad, Aroused, Bored, Tired, and Annoyed remain dificult. Direct VLMs also remain stronger for some classes, suggesting that VLMs and graph-based models capture partly complementary afective signals.

## 4.3 Feature-Emotion Association Analysis

To answer RQ3, we inspect the learned associations between visual-semantic feature nodes and emotion labels. Table 6 reports the top associated feature nodes for each emotion. The associations are obtained from feature-emotion logodds scores estimated from the training set and used to initialize the VSAG feature-emotion edges. Each score indicates how strongly a feature is associated with an emotion label relative to its absence in the training set.

The associations reveal several interpretable patterns. High-arousal or negative emotions are often linked to visually intense features: Alarmed is associated with strong contrast, tense composition, and dramatic lighting, while Frustrated is associated with rough or energetic brushwork and strong contrast. In contrast, calmer positive emotions are linked to softer visual properties: Calm is associated with soft lighting, gentle brushwork, and balanced composition, while Contentment is associated with warm colors and domestic entities such as baby.

Table 6. Top-3 visual-semantic feature associations for each emotion. Each entry shows the feature node type, feature value, and association score in parentheses.
<table><tr><td>Calm</td><td>Contentment</td><td>Glad</td><td>|Excited</td><td>|Happy</td><td>|Aroused</td></tr><tr><td>Light/shadow: Soft/ Diffused Lighting (0.49)</td><td>Entity: baby (0.99)</td><td>Entity: (4.99)</td><td>forest Color: Vibrant Intense (1.62)</td><td>Entity: blue sky (3.05)</td><td>Entity: vertical element (3.37)</td></tr><tr><td>Brushstroke: Soft / Gentle (0.44) Composition: Bal-</td><td>Entity: sofa (0.93)</td><td>Entity: (4.87)</td><td>zigzag Color: High Satu- ration (1.51)</td><td>Entity: cat (2.92)</td><td>Entity: linear ele- ment (3.37)</td></tr><tr><td>anced (0.43)</td><td>Color: Warm Col- ors (0.84)</td><td>Composition: Minimalist (4.87)</td><td>Composition: Dy- namic (1.33)</td><td>Color: Warm Col- ors (2.12)</td><td>Composition: Minimalist (2.62)</td></tr><tr><td>Sad Color: Muted</td><td>|Bored Entity: linear ele-</td><td>|Tired</td><td>|Alarmed</td><td>|Frustrated</td><td>Annoyed</td></tr><tr><td>Subdued (1.76)</td><td>ment (3.21)</td><td>Entity: introspec- tion (3.48)</td><td>Light/shadow: Strong High Contrast (2.45)</td><td>Brushstroke: Rough / Textured (1.94)</td><td>Entity: pedestrian (3.42)</td></tr><tr><td>Color: Dark Palette (1.74)</td><td>Entity: waterfront (3.15)</td><td>Entity: dark suit (3.48)</td><td>Composition: Chaotic / Tense</td><td>Brushstroke: Ex- pressive Ener-</td><td>Entity: facial fea- ture (3.42)</td></tr><tr><td>Entity: soldier (1.65)</td><td>Composition: Minimalist (2.47)</td><td>Line: Blurred Indistinct Lines</td><td>(2.26) Light/shadow: Harsh / Dramatic</td><td>getic (1.89) Light/shadow: Strong High</td><td>Composition: Minimalist (2.67)</td></tr></table>

Figure 2 summarizes the same associations at the feature-group level. It uses all positive feature-emotion associations whose feature node appears in at least five training artworks, computes the mean association strength for each feature group, and normalizes these values within each emotion. The results indicate a relatively balanced contribution across feature groups, with no single group consistently dominating across all emotion labels: color is prominent for Excited and Sad, light-shadow for Alarmed, brushstroke for Frustrated and Calm, and entity nodes for labels such as Glad, Bored, Tired, and Annoyed.

However, note that these values should be interpreted as learned featureemotion associations rather than causal feature importance. Some associations, especially for rare classes such as Glad may reflect limited support and should therefore be interpreted cautiously. Accordingly, this analysis is best interpreted as descriptive evidence of which visual-semantic feature groups are most strongly associated with each emotion label.

![](images/09af8955679a9c6d7c7b596f93faf30ecd3b04f08637d52b369233d9735cd598.jpg)  
Fig. 2. Feature-group mean association strength by emotion.

## 5 Discussion

Our results show that explicit visual-semantic structure is useful for artwork emotion prediction, but the task remains dificult. One reason is that emotion in art is rarely limited to a single label: an artwork may evoke multiple emotions, and diferent viewers may interpret it diferently. Since the dataset provides only one dominant emotion label per artwork, models are trained to match a specific annotation choice rather than the full range of possible afective responses. The graph formulation partly addresses this limitation by producing scores for all emotion nodes, which allows secondary emotions and feature-emotion paths to be inspected.

The comparison with direct VLM inference suggests that VLMs and graphbased models capture complementary signals. VLMs may rely on global visual impressions and broad semantic priors, while VSAG uses explicit feature nodes such as color, brushstroke, composition, light-shadow, line quality, and visual entities. This makes VSAG more interpretable, but also dependent on feature quality and coverage. The +VA results show that valence-arousal hierarchy improves standard VSAG, mainly by helping several negative or less frequent classes. However, this benefit does not transfer to VSAG-TopK, suggesting that VA structure is most useful with distributed feature-path aggregation, while TopK pooling may already emphasize dominant afective cues. The persistent near-zero performance on rare labels shows that long-tailed afect supervision remains a major challenge, motivating future investigation of class-reweighting, focal loss, or graph-based sampling strategies. Multi-label emotion annotation may also better reflect the range of afective responses that an artwork can evoke.

Finally, a further limitation is that the current graph mainly uses visualsemantic features and does not yet incorporate non-visual contextual information such as artist, year, medium, style period, or cultural background, which may afect emotional interpretation. Previous work has shown that such contextual knowledge can be represented through knowledge graphs and combined with visual information for artwork emotion recognition [8]. In addition, our current graph-based evaluation primarily compares variants of VSAG rather than independently developed graph-based approaches. Future work could therefore incorporate external art knowledge into VSAG to construct a richer artwork knowledge graph and evaluate it against existing knowledge-graph-based methods.

## 6 Conclusion

Our work studied artwork emotion prediction using Visual-Semantic Afect Graphs, which represent artworks through interpretable visual and semantic feature nodes and learn their associations with emotion labels. Experiments on EmoArt-5k show that direct VLM inference provides a useful but modest baseline, while graph-based models improve dominant emotion prediction. The best overall model is VSAG+VA, which integrates valence-arousal structure through a hierarchical auxiliary afect formulation and achieves the highest accuracy, macro-F1, and weighted-F1 among the evaluated models. The results also show that the benefit of VA structure depends on the graph aggregation strategy: it improves standard VSAG, but not VSAG-TopK. Per-class analysis shows that hierarchical VA modeling improves several dificult negative emotions, while rare classes remain challenging because of severe label imbalance. Feature-emotion analysis identifies interpretable associations between emotions and visual-semantic cues such as contrast, brushwork, composition, lighting, and entities. Future work can extend VSAG toward multi-label emotion modeling, richer art-historical metadata, and hybrid models that combine VLM representations with interpretable graph structure.

## References

1. Achlioptas, P., Ovsjanikov, M., Haydarov, K., Elhoseiny, M., Guibas, L.J.: Artemis: Afective language for visual art. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11569–11579 (2021)

2. Agarwal, M., Tsaftaris, S.A., Sevilla-Lara, L., McDonagh, S.: Why do vision language models struggle to recognize human emotions? arXiv preprint arXiv:2604.15280 (2026)

3. Ananthram, A., Winn, O., Muresan, S.: Feelingblue: a corpus for understanding the emotional connotation of color in context. Transactions of the Association for Computational Linguistics 11, 176–190 (2023)

4. Bhattacharyya, S., Wang, J.Z.: Evaluating vision-language models for emotion recognition. In: Findings of the Association for Computational Linguistics: NAACL 2025. pp. 1798–1820 (2025)

5. Blanco, G., Monopoli, T., D’Asaro, F., Peeters, R., Duan, X., Dimou, A., Rizzo, G.: Artkb: A multimodal art knowledge base for cultural heritage. In: European Semantic Web Conference. pp. 58–75. Springer (2026)

6. Bondielli, A., Passaro, L.C., et al.: Leveraging clip for image emotion recognition. In: Ceur workshop proceedings. vol. 3015. CEUR-WS (2021)

7. Castellano, G., Digeno, V., Sansaro, G., Vessio, G.: Leveraging knowledge graphs and deep learning for automatic art analysis. Knowledge-Based Systems 248, 108859 (2022)

8. Castellano, G., Scaringi, R., Vessio, G.: Recognizing the style, genre, and emotion of a work of art through visual and knowledge graph embeddings. In: International Conference of the Italian Association for Artificial Intelligence. pp. 427–440. Springer (2023)

9. Ekman, P.: Universals and cultural diferences in facial expressions of emotion. In: Nebraska symposium on motivation. University of Nebraska Press (1971)

10. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision 123(1), 32–73 (2017)

11. Lei, Y., Yang, D., Chen, Z., Chen, J., Zhai, P., Zhang, L.: Large visionlanguage models as emotion recognizers in context awareness. arXiv preprint arXiv:2407.11300 (2024)

12. Li, Z., Wu, X., Du, H., Liu, F., Nghiem, H., Shi, G.: A survey of state of the art large vision language models: Benchmark evaluations and challenges. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1587–1606 (2025)

13. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision. pp. 216–233. Springer (2024)

14. Machajdik, J., Hanbury, A.: Afective image classification using features inspired by psychology and art theory. In: Proceedings of the 18th ACM international conference on Multimedia. pp. 83–92 (2010)

15. Mastandrea, S., Fagioli, S., Biasi, V.: Art and psychological well-being: Linking the brain to the aesthetic emotion. Frontiers in psychology 10, 430007 (2019)

16. Mohammad, S., Kiritchenko, S.: Wikiart emotions: An annotated dataset of emotions evoked by art. In: Proceedings of the eleventh international conference on language resources and evaluation (LREC 2018) (2018)

17. Murgod, T.R., Reddy, P.S., Gaddam, S., Sundaram, S.M., Anitha, C.: A survey on graph neural networks and its applications in various domains. SN Computer Science 6(1), 26 (2024)

18. Pelowski, M., Markey, P.S., Forster, M., Gerger, G., Leder, H.: Move me, astonish me. . . delight my eyes and brain: The vienna integrated model of top-down and bottom-up processes in art perception (vimap) and corresponding afective, evaluative, and neurophysiological correlates. Physics of life reviews 21, 80–125 (2017)

19. Russell, J.A.: A circumplex model of afect. Journal of personality and social psychology 39(6), 1161 (1980)

20. Wang, B., Chen, Q., Wang, Z.: Difusion-based visual art creation: A survey and new perspectives. ACM Computing Surveys 57(10), 1–37 (2025)

21. Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S.: A comprehensive survey on graph neural networks. IEEE transactions on neural networks and learning systems 32(1), 4–24 (2020)

22. Ying, K., Meng, F., Wang, J., Li, Z., Lin, H., Yang, Y., Zhang, H., Zhang, W., Lin, Y., Liu, S., et al.: Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006 (2024)

23. Zhang, C., Xie, H., Wen, B., Zuo, S., Zhang, R., Cheng, W.H.: Emoart: A multidimensional dataset for emotion-aware artistic generation. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 12644–12650 (2025)

24. Zhang, C., Zhang, C., Zhang, M., Kweon, I.S., Kim, J.: Text-to-image difusion models in generative ai: A survey. arXiv preprint arXiv:2303.07909 (2023)