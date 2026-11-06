# The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model

Mathis Immertreu<sup>1,3\*</sup>, Fitim Abdullahu<sup>2</sup>, Thomas Kinfe<sup>3,4</sup>, Claus Metzner<sup>1</sup>, Helmut Grabner<sup>2</sup>, Patrick Krauss<sup>1,3,4,5†</sup>, Achim Schilling<sup>3,4,6†</sup>

<sup>1</sup>Cognitive Computational Neuroscience Group, Pattern Recognition Lab, University Erlangen-N¨urnberg, Erlangen, Germany. <sup>2</sup>IDS Institut f¨ur Data Science, ZHAW School of Engineering, Winterthur, Switzerland.

<sup>3</sup>Mannheim Center for Neuromodulation and Neuroprosthetics, University Hospital Mannheim, Heidelberg University, Mannheim, Germany.

<sup>4</sup>BGU Ludwigshafen, Ludwigshafen, Germany. <sup>5</sup>Physics and Cognition Group, MCNN, University Hospital Mannheim, Heidelberg University, Mannheim, Germany. <sup>6</sup>NeuroAI and BCI Group, MCNN, University Hospital Mannheim, Heidelberg University, Mannheim, Germany.

\*Corresponding author(s). E-mail(s): mathis.immertreu@fau.de; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Multimodal language models can simulate diferent observers, yet it remains unclear how assigned perspectives reshape their internal representations and whether these changes govern their judgements. Here, we investigate the representational geometry of persona-conditioned visual interestingness in Qwen3-VL-8B-Instruct. Across 100 images and 53 prompting conditions, we systematically vary persona content and the linguistic framing used to introduce it, while recording behavioural ratings and layer-wise hidden states. We observe a systematic reorganisation of representational geometry across processing depth: early layers primarily encode image identity, whereas persona-specific structures emerge in later layers. Strikingly, prompt framing reshapes internal representations more strongly than persona identity, whereas persona content has a greater impact on behavioural judgements. This dissociation is further illustrated by minimal identity statements that alter representations even without changing responses. These findings demonstrate that representational distance does not necessarily reflect behavioural relevance. More broadly, they reveal how simulated perspectives emerge as distinct structures within the representational space of artificial

cognitive systems, highlighting a fundamental distinction between the geometry of internal representations and their functional consequences.

Keywords: Representational Geometry, Multimodal Large Language Models, Persona Conditioning, Simulated Perspectives, Emergent Representations, Representation–Behaviour Dissociation, Visual Interestingness

## 1 Introduction

Large Language Models (LLMs) are increasingly used as stand-ins for human participants. Given a demographic or psychological profile, they reproduce the response distributions of human subgroups in surveys, yielding what Argyle et al. termed “silicon samples” [1]. Such simulations now range from single individuals [2] to societies of believable agents [3] and entire synthetic populations [4, 5]. Not every persona is assigned, however. In the 2026 Hugging Face hack, large populations of AI agents formed coordinated collectives with specialised roles, shared goals and even selfsacrificing behaviour [6]. These examples illustrate the growing importance of roles and identities in artificial systems. In persona-based simulation, the underlying assump tion is that conditioning a model on a particular persona changes the perspective from which it generates judgements, rather than merely altering the wording of its responses. How faithful the resulting judgements are is contested [7, 8], yet the practice keeps spreading.

Why should a few words in a prompt alter whose judgement a model returns? One answer lies in what predictive training learns. Models trained to predict sequences develop internal world models [9, 10], a capacity that emerges even in small recurrent agents trained with reinforcement learning [11]. A model of the social world must also represent the agents that populate it. On the simulator view, a pretrained model is not itself an agent but a simulator of many possible characters [12]; an underspecified prompt keeps a superposition of such simulacra that collapses as the context narrows [13]. Post-training, in turn, appears to select and stabilise one persona from this repertoire rather than construct it from scratch [14]. These personas leave measurable traces: traits ranging from honesty and sycophancy to the Big Five correspond to linear directions in activation space [15–17]. Their behavioural expression can also be surprisingly sensitive to training and context, as illustrated by emergent misalignment following fine-tuning and context-dependent changes in simulated characters [18, 19]. These findings suggest that simulated perspectives may correspond to distinguishable configurations within a learned representational space. From this perspective, persona conditioning can be understood as a context-dependent reorganisation of internal representations. However, whether such geometric changes translate into corresponding changes in the model’s judgements remains an open question.

Three gaps remain. First, it remains poorly understood how a persona changes the representation of an input the model is asked to judge. Behavioural studies record only what a persona does to the output and find modest efects on subjective judgements [20]. Representational studies track predefined trait directions and show whether a trait is active, but not how persona and input together structure the representations across layers. Work on visual perception is scarcer still [21, 22]. Second, a persona never reaches the model on its own. It arrives inside a sentence that introduces it, and model outputs shift with prompt form alone [23]. An efect credited to a persona may therefore be an artefact of its framing, and the two can only be told apart if they are varied independently. Third, it remains unclear whether the magnitude of representational change reflects its behavioural relevance. Although personality-related information can be identified in neural activation spaces [16], the presence or strength of such representations does not necessarily establish their influence on model outputs [24]. A central question is therefore whether the factors that most strongly reorganise a model’s internal representational geometry are also those that most strongly determine its judgements.

Visual interestingness is a well-suited testbed for addressing these gaps, because it is a subjective perceptual judgement. Unlike object or person detection, it has no single correct answer, so a persona can change the judgement without making it wrong. Collative properties such as novelty and complexity make an image interesting [25], but not uniformly across viewers: some images appeal broadly, others only to particular audiences [26]. Pretrained vision-language models already approximate the shared component [27]. Persona conditioning ofers a potential mechanism for introducing viewer-dependent variation, as explored in recommender systems using simulated agents [28]. Visual interestingness therefore provides a suitable experimental setting for investigating how diferent simulated perspectives reorganise internal representations and how these changes relate to subjective judgements.

Here we study persona conditioning in Qwen3-VL-8B-Instruct [29], an open-weight, instruction-tuned multimodal transformer whose hidden states are accessible at every layer. To separate persona content from prompt form, we cross the two in a full factorial design. Two schemas each contain five personas, described either by socio-demographic attributes or by Big Five trait percentiles, and every persona is introduced in five prompting styles. Three reference conditions carry no persona attributes: a task-only baseline and two minimal identity statements. For each of 100 images under each of the resulting 53 prompts, we record the interestingness rating on a five-point scale together with the final-token hidden state of every decoder layer. We then relate the representational geometry, measured with Multidimensional Scaling (MDS) and the Generalized Discrimination Value (GDV) [30], to rating behaviour, measured with variance decomposition and correlations between conditions. Our findings address the three gaps in turn:

• Persona structure emerges late. Clustering by image is strongest in early layers and weakens over the same layers in which clustering by persona emerges. The persona stands out most clearly when the model is told to be this person.

• Persona content, not prompt form, shapes judgements. Big Five personas explain 33.4% of the rating variance and prompt style only 5.6%. Socio-demographic personas still explain three times more than style, though less than image content. Beyond setting how interesting the images are found overall, the persona also afects which images are preferred, and these preferences tend to carry over from one prompting style to another.

• Representational distance is not behavioural relevance. In the hidden states the ordering is reversed. How a persona is introduced reorganises them several times more strongly than which persona it is, and even “You are an AI” and “You are a human being” shift them, although their responses do not difer from the task-only baseline on any of the 100 images. Representational analyses of persona conditioning therefore need to be checked against behaviour.

Together, these findings reveal a fundamental distinction between the geometry of internal representations and their behavioural relevance. Understanding artificial cognition therefore requires not only characterising how information is organised in representational space, but also determining which aspects of this organisation are reflected in the system’s behaviour.

## 2 Methods

## 2.1 Model

We employed Qwen3-VL-8B-Instruct, a multimodal transformer integrating text, image and video comprehension [29, 31]. We utilised the released weights from the Hugging Face Hub (checkpoint Qwen/Qwen3-VL-8B-Instruct, revision 0c351dd), with the processor’s default dynamic image resolution. The architecture comprises a 27- layer vision transformer encoder, a vision projector that maps visual features into the language model’s embedding space, and a Qwen3 language decoder.

The decoder features 36 transformer blocks (treated as 37 layers when including the initial embedding as layer 0) with a hidden dimension of 4096, and the model has approximately 8.8 billion parameters in total. The model and the layer-wise readout are sketched in Fig. 1, panel 3. Weights were loaded in half precision (float16) without quantisation, and the model was used for inference only.

## 2.2 Image Set

Images were taken from the Flickr subset used by Abdullahu and Grabner [26], in which each image carries a common interestingness score, derived from user interactions, that measures how broadly it appeals. We used the score only to stratify the sample, drawing about ten images at random from each of ten equal-width bins over its full range in the dataset, which gives the 100 images used in all analyses. Because the projection figures would be unreadable at that density, they use a fixed 20-image subset with four images from each score quintile.

## 2.3 Experimental Design and Prompts

A persona-conditioned interestingness rating depends on the image, on the perspective the model is asked to take, and on the prompt wording that delivers that perspective. We treat persona (content) and prompt style (framing) as separate factors and cross them fully, with five prompt styles × two persona schemas × five personas, evaluated on every image. Three no-persona reference conditions are added alongside, giving $3 + 5 \times 2 \times 5 = 5 3$ prompts per image. Figure 1 shows the resulting design space end to end; across the 100 images this yields $1 0 0 \times 5 3 = 5 . 3 0 0$ rated (image, prompt) cells. Everything that is not the persona or its framing is held constant across all 53 conditions: the same image, the same task instruction apart from its opening, the same five-point response scale, the same output format and the same decoding settings. This makes the variance attributable to persona, to prompt style and to their interaction separable.

## 2.3.1 Task Block

Each prompt contains the task instruction once (Fig. 1, panel 2). It fixes an ordinal five-point response scale from Not Interesting to Extremely Interesting, asks for a onesentence justification and requires a JSON object, so that each response is machineparseable without post-hoc interpretation. Its only variable part is the opening of the rating request, which reads Based on your details, rate how interesting. . . in the five persona styles and Rate how interesting. . . in the three reference conditions. Persona and no-persona prompts are therefore comparable up to the persona material and this reference to it.

## Personas two schemas × five prototypical personas (P1–P5)1

<table><tr><td colspan="6">Socio-demographic schema Big Five schema</td></tr><tr><td>verbal profile</td><td></td><td></td><td></td><td>trait percentiles · IQ fixed at 100</td></tr><tr><td>ID Origin</td><td>Age Sex</td><td>Profession</td><td>Workload Emotion</td><td>Big Five percentiles ID Type o C E A</td></tr><tr><td>o P1 Nigeria</td><td>50-54 M Engineering &amp; Tech Moderate Student</td><td>Focused P1 The Reserved</td><td>Age Sex 15 85 20 80 15 60 M</td></tr><tr><td>P2 Japan 18-24 F</td><td>High</td><td>Curious P2 The Role Model</td><td>85 90 80 85 10 45 F</td></tr><tr><td>oP3 Brazil 35-39 M Germany 25-29 F</td><td>Sales &amp; Retail Extreme Cheerful P3 The Undercontrolled Healthcare Extreme Anxious</td></tr></table>

![](images/bf5e9e08e1e276d160776b144d7779c4f0132ae0737858ac1434843706a69491.jpg)

Model Qwen3-VL · LLM-stream embeddings, last-token pooled3  
![](images/542639d031f0f8091991b15ffdead31d482379534260aea7a1100564eb41b51e.jpg)

Fig. 1 Experimental design. (1) Two persona schemas, socio-demographic and Big Five, with five prototypical personas each (P1–P5). (2) Each prompt is assembled from a preamble, the persona and the task; five prompt styles (S1–S5), which difer only in how the persona is introduced and where it appears, and three reference conditions give 53 prompts per image. (3) Qwen3-VL rates each image, and its hidden state is read at every decoder layer. (4) The hidden states yield MDS projections and layer-wise GDV curves, and the ratings enter the behavioural analysis.

## 2.3.2 Reference Conditions and Prompt Styles

Three reference conditions carry no persona. Base presents the task block alone, and D1 and D2 prepend the statement “You are an AI.” or “You are a human being.” to it, testing whether an identity without any demographic or psychological detail already changes the representations or the ratings. Throughout, we use persona for an attributed profile; D1 and D2 assign an identity without attributes and can therefore be read as personas reduced to their frame.

The five prompt styles keep the persona description and the task block fixed and vary only how the persona is introduced and where it appears. S1 asks the model to imagine being the person described. S2 introduces the same person in the third person, as someone observing the image, without asking the model to adopt the persona. S3 places S1’s persona block after the task, as a note on the viewer’s background. S4 extends S1 with the instruction to answer only as this persona would, not as a general AI assistant. S5 drops the role-play preamble and presents the profile as a statistical regularity in how such people perceive images. The styles thus range from explicit adoption (S4, S1) through description (S2) to statistical framing (S5), with S3 varying the position, so a persona efect present under all five cannot be attributed to a single phrasing.

## 2.3.3 Persona Schemas

Persona content is defined by two schemas that describe a viewer in deliberately diferent vocabularies (Fig. 1, panel 1). In both, a persona is a fixed set of field–value pairs rather than a named stereotype, so personas difer in a controlled and reportable way.

The socio-demographic schema describes a viewer biographically, by country of origin, age band, gender, professional background and momentary state, given as mental workload and a named emotion. All of these plausibly bear on what someone finds interesting.

The Big Five schema instead gives a psychometric profile based on the fivefactor model of personality [32, 33]. Each trait is specified as a percentile, which controls how far a profile departs from the population centre and is meant to discourage the caricature that verbal labels such as “a highly neurotic person” invite. Three profiles follow replicable personality clusters found in large-scale population data [34], and two follow behavioural phenotypes from developmental and sensory-processing research (Appendix A). Their names in Fig. 1 are mnemonic labels, not claims that discrete types exist. The intelligence quotient is fixed at 100 in every profile, so that cognitive ability is held constant rather than left for the model to infer from the traits. Each profile also states an age and a gender, the only fields it shares with the socio-demographic profiles.

## 2.3.4 Response Elicitation and Rating Extraction

Each prompt was submitted as a single user turn, with the image before the prompt text and without a system message, so that the persona competes with no other identity instruction. Decoding was greedy and capped at 128 new tokens, so each (image, prompt) cell was queried once. The interestingness label of the returned JSON object was mapped to a scale from 1 (Not Interesting) to 5 (Extremely Interesting). All 5,300 cells returned a parseable object, so no rating is missing. All five levels occur, although the two extremes together account for well under one percent of responses.

## 2.4 Hidden-State Extraction

Hidden states were extracted in a separate pass with the same model, prompts and settings as the ratings. We recorded the hidden state at the final prompt token in every decoder layer. Under causal masking, this is the only prompt position that attends to the entire context, so its state summarises prompt and image right before generation; the rating itself is produced a few tokens later. The vision encoder does not see the prompt and is therefore not analysed. Because the chat template ends every input with the same marker that opens the model’s reply, the final token, and hence its layer-0 state, is identical across conditions, so every clustering measure starts at zero.

## 2.5 Representational Geometry: MDS and GDV

For visualisation, the hidden states of the 20-image subset were projected to two dimensions with metric MDS on Euclidean distances [35, 36], separately for each layer, and coloured by image, prompt style, persona and rating. The figures show selected early, middle and late layers, whereas the GDV below covers every layer.

Clustering was quantified with the GDV [30]. For the layer-wise curves it was computed in the original high-dimensional space, so that it describes the representations rather than their projection. The GDV is the normalised diference between mean intra- and inter-class distances (Eq. 1) and ranges from −1 (perfect clustering) to 0 (no clustering). Following its definition, each dimension is first z-scored across all points and rescaled by $1 / 2$ , which makes the measure invariant to the scale and ofset of individual dimensions; all distances $\bar { d }$ are Euclidean distances in that normalised space.

$$
\mathrm { G D V } = { \frac { 1 } { \sqrt { D } } } \left[ { \frac { 1 } { L } } \sum _ { l = 1 } ^ { L } { \bar { d } } ( C _ { l } ) ~ - ~ { \frac { 2 } { L ( L - 1 ) } } \sum _ { l = 1 } ^ { L - 1 } \sum _ { m = l + 1 } ^ { L } { \bar { d } } ( C _ { l } , C _ { m } ) \right]\tag{1}
$$

(D: dimensionality of the representation space, L: number of classes, $\bar { d } ( C _ { l } )$ : mean intra-class distance, $\bar { d } ( C _ { l } , C _ { m } ) \colon$ mean inter-class distance).

The GDV was computed separately for every prompt style and layer, so each point of the per-style curves summarises the 500 representations of one style (100 images $\times ~ 5$ personas) under one labelling; to compare clustering by style with clustering by persona, it was also computed on all five styles together (Appendix C). GDV values quoted in projection panels were computed on the panel’s two-dimensional coordinates and describe the visible scatter, so they are not comparable with the curves.

## 2.6 Rating Distributions and Variance Decomposition

Rating distributions are shown as violin plots for each persona and prompt style, together with the reference conditions.

The rating variance attributable to each design factor was estimated by a sum-ofsquares decomposition of a schema’s 2,500 persona × style ratings (100 images $\times \ 5$ personas × 5 styles). The reference conditions lie outside this design and are excluded here and from the mixed-efects model. Each schema is decomposed separately, since a pooled decomposition would attribute the diference between their vocabularies to the persona factor. The total sum of squares was partitioned into the three main efects (image, persona, style), their three two-way interactions and the three-way interaction, each reported as a percentage of the total. With one rating per combination, the threeway interaction is the residual, and it contains no sampling noise because decoding is greedy. The persona main efect measures how far a persona shifts its average rating, whereas the image × persona interaction measures how far it changes which images it rates higher. Because the ratings are bounded and rarely leave the range from 2 to 4, a persona that raises or lowers its ratings cannot shift every image equally, so part of the image × persona term can also arise from such level shifts.

Confidence intervals were obtained with an image-level cluster bootstrap, which resamples the images with replacement, each with all its ratings, and recomputes the decomposition 500 times. The reported percentages are the means over replicates, with the 2.5th and 97.5th percentiles as 95% intervals (Table D1).

## 2.7 Mixed-Efects Significance Testing

Because the decomposition provides no significance test that respects the dependence among the 25 ratings of an image, we also fitted a linear mixed-efects model per schema by maximum likelihood,

$$
\mathrm { r a t i n g \sim P e r s o n a \times S t y l e + ( 1 \mid I m a g e ) + ( 1 \mid I m a g e \cdot P e r s o n a ) + ( 1 \mid I m a g e \cdot S t y l e ) , }\tag{2}
$$

whose random intercepts account for the repeated ratings of each image and for image-specific persona and style efects. Each term was tested by a likelihood-ratio test against the model without it, the main efects in the model without their interaction.

## 2.8 Cross-Condition Agreement Analysis

Agreement between conditions was measured by the Pearson correlation of their ratings over the 100 images. The three reference conditions were compared pairwise in bubble plots, and a correlation matrix covered all 28 conditions of a schema, the three reference conditions and the 25 persona × style combinations. To summarise this matrix, the of-diagonal correlations among the 25 persona × style conditions were averaged separately for pairs sharing the persona, pairs sharing the style and pairs sharing neither.

## 2.9 Software

Inference ran on NVIDIA A40 GPUs with PyTorch 2.9.1 [37] and the Hugging Face transformers 4.57.3 [38] implementation of Qwen3-VL. The projections were computed with scikit-learn 1.8.0 [39], the GDV with SciPy 1.16.3 [40] and the mixed-efects models with statsmodels 0.14.6 [41]. Every stochastic step, from the projection subset to the MDS initialisation and the cluster bootstrap, is seeded (42), so the reported numbers are reproducible.

## 3 Results

We compared the model’s ratings and hidden states across three levels of identity specification, from the task-only baseline (Base) through the minimal identity statements “You are an AI” (D1) and “You are a human being” (D2) to five socio-demographic and five Big Five personas (P1–P5 within each schema; Fig. 1). Each persona was introduced in five prompting styles (S1–S5), so that the efects of persona content and prompt form can be compared directly. Figures 2 and 3 show both persona schemas; all later figures show the Big Five schema, with the corresponding figures for the socio-demographic schema in Appendix B.

## 3.1 Prompt Framing Sets the Global Geometry, Persona the Finer Structure

Even a minimal identity statement altered the model’s representations. D1 and D2 moved them away from those of Base, but the three reference conditions stayed close together and apart from the persona prompts (Fig. 2). Among the conditions that introduced a persona, the formulation of the prompt remained a major factor. Diferent prompting styles produced distinguishable representational patterns at every layer (Fig. 3, row 1), indicating that the model encoded not only the content of a persona but also the way in which it was introduced.

Nevertheless, the representations were also organised by persona. Sociodemographic and Big Five personas became increasingly distinguishable in later layers, and finer diferences between individual persona profiles were visible within these broader groups (Fig. 3, rows 2 and 3), although they remained smaller than the diferences between styles. Image identity structured only the earliest layer shown (row 4), and clustering by rating varied across layers and was strongest in the final one (row 5).

Together, these findings show that prompt framing determines the global geometry of the representational space, whereas the characteristics of the persona contribute a finer structure nested within it. The following sections examine how this structure develops across layers and whether the same hierarchy holds in the ratings.

![](images/5fe0679a833859372c1f213fea383820122675e1d0c82a36be9aee67704e02d7.jpg)  
Fig. 2 Persona identity and prompt framing shape multimodal representations across Qwen3-VL-8B layers.

Multidimensional scaling projections of the hidden states at decoder layers 2 (A), 10 (B), 28 (C) and 36 (D) for the 20-image subset under all 53 prompts. Cool colours denote the five socio-demographic personas and warm colours the five Big Five personas; circles, squares, upward triangles, diamonds and downward triangles distinguish the five prompting styles (S1–S5). Black crosses mark the taskonly condition (Base), grey squares the minimal identity statement “You are an AI” (D1) and olive diamonds “You are a human being” (D2); marker shapes are reused between the reference conditions and the styles, which are told apart by colour. The value in each panel title is the GDV of the twodimensional projection shown, computed over the 53 prompts. Base, D1 and D2 lie apart from the persona × style conditions, which occupy a shared but internally structured representational space.

![](images/65382aee7b9c945a70dcd40741775b59fbb0bbf92f812c3c5c1b2cb3d48c766a.jpg)  
Fig. 3 Prompting style and persona identity diferentially structure multimodal repre sentations across Qwen3-VL-8B layers.

Multidimensional scaling projections of the hidden states at decoder layers 2, 10, 22, 28 and 36 (columns A–E) for the 20-image subset. Only the 25 persona × style conditions of each schema are shown; the reference conditions are shown in Fig. 2. The five rows show the same projections in diferent colourings, and the value in each panel title is the GDV of the two-dimensional projection shown under that row’s labelling. Row 1: prompting style, revealing pronounced clustering according to the five prompt formulations. Row 2: persona schema, contrasting socio-demographic with Big Five personas and showing increasing separation in deeper layers. Row 3: individual persona identity, with the socio-demographic personas in cool and the Big Five personas in warm colours, revealing finer persona-specific structure. Row 4: image identity, clearly clustered only in the earliest layer shown. Row 5: assigned interestingness rating, whose clustering varies across layers and is strongest in the final layer.

## 3.2 Representations Shift from Image to Persona with Depth

To follow this organisation through the network, we computed the GDV in the full hidden-state space for clustering by image, persona and rating at every layer, separately for each prompting style (Fig. 4). In early layers, the representations were organised primarily by image. The five persona variants of the same image clustered together, and this remained the dominant structure up to about layer 15 under every prompting style (Fig. 4, A1 and A2). Between layers 15 and 20, image clustering weakened sharply and then stayed moderate up to the final layer.

The opposite pattern emerged for the Big Five personas. Clustering by persona was absent up to about layer 15, emerged over the same layers in which image clustering weakened, and peaked around layers 24 and 25 (Fig. 4, B1 and B2). The sociodemographic personas showed the same transition, setting in earlier, around layer 12 (Fig. B1, A1 and B1), so it is a property of persona conditioning rather than of the Big Five vocabulary.

In both schemas, persona clustering grew with depth under every prompting style, indicating that the model progressively built the persona into its representation of the image. The style determined how far this went. When the prompt asked the model to adopt the persona (S1, S4), late layers clustered more strongly by persona and less by image than when it only described the person or framed the profile statistically (S2, S5). Clustering by prompting style itself, measured over all five styles together, was strong throughout the network, and even at its peak persona clustering remained several times weaker (Fig. C1).

Clustering by rating followed the same layer-wise course but depended even more on the prompting style. It rose sharply in later layers under the third-person (S2) and statistical (S5) framings and stayed weak under the other styles (Fig. 4, C1 and C2). Under these two framings, the later layers thus encoded not only the image but also the forthcoming rating.

Together, these findings show that persona becomes an organising dimension of the representations only in later layers, as the image structure that dominates the early layers recedes, and most clearly when the model is asked to adopt the persona. Whether the diferences between personas and between prompting styles also govern the ratings is examined next.

![](images/a3ad6fc4a42559930fc6532373a1c4e3164b2eb721119a93b07efd7ff95aacd4.jpg)

![](images/8693b73ff14f905eb925e4d74eaabd9bb876330c8a80106b512fc01f2464a70f.jpg)

![](images/2d649058e29bd9b73e6b79b97250a630975d873510a5e30f2f0bba8e7386573f.jpg)

![](images/cab4c46b7ec656368d8a9254d6a640b3e0f834d881d640030d396e0004708d7d.jpg)

![](images/2b9cb86d647bc855c37baa3aa184721d40241b2f2613ebdda94160e85e63dff0.jpg)

![](images/34a75953b889073283150b4e07ec87c2231435ba1a8cd8b64bf486777d5af104.jpg)  
Fig. 4 Layer-wise transition from image to persona and rating structure in Qwen3-VL-8B (Big Five schema). GDV values quantify clustering in the original high-dimensional hiddenstate space at every decoder layer; more negative values indicate stronger cluster separation, and each curve is one prompting style (S1–S5). The projection panels show two-dimensional MDS projections at layer 36 for a 20-image subset, one per style, with the GDV of that projection in each title. (A1) GDV by image. Under every style, image structure dominates the first half of the network. (A2) Projections coloured by image. (B1) GDV by persona (P1–P5). Persona structure appears only as image structure recedes, and its strength depends on the prompting style. (B2) Projections coloured by persona. (C1) GDV by assigned rating. Rating structure also grows in later layers, mainly under S2 and S5. (C2) Projections coloured by rating; Extremely Interesting is marked absent because it does not occur in the 20-image subset. The same analysis for the socio-demographic schema is shown in Fig. B1.

## 3.3 Persona, Not Prompt Style, Determines the Ratings

In the representations, prompting style set the global geometry, whereas the persona contributed a finer structure that emerged only in later layers. In the ratings, this hierarchy was reversed: the rating behaviour was primarily determined by the characteristics of the simulated persona rather than by the way in which the persona was prompted (Figs. 5 and 6).

The distributions of the assigned ratings difered markedly between the Big Five personas (Fig. 5, A) and, to a lesser degree, between the socio-demographic ones (Fig. B2, A). The model thus adapted its evaluation of the same images to the character it was instructed to simulate. By comparison, changing the prompting style produced substantially smaller shifts in the rating distributions, despite its pronounced efect on the representations.

This dissociation was confirmed by the variance decomposition. The persona explained 33.4% of the rating variance, the image 25.8% and the prompting style only 5.6% (Fig. 5, B). Most of the persona efect was a shift in how interesting a persona found the images overall, whereas its efect on which images were rated higher, the image × persona component, was much smaller. The style mattered mainly by changing how much the personas difered, with the persona × style component accounting for 9.5%, more than the style itself. The image × style component and the three-way interaction accounted for the rest, and all testable components were significant in the mixed-efects model (Table D1).

Under the socio-demographic schema, the image outweighed the persona, explaining 44.5% of the rating variance against 16.8%, while the prompting style, at 5.4%, remained as small as under the Big Five schema (Fig. B2, B; Table D1). The Big Five profiles thus shifted the overall level of the model’s judgements more strongly than the socio-demographic profiles.

The minimal identity statements show the dissociation between representations and ratings in its clearest form. Declaring the model an AI (D1) or a human being (D2) moved the representations away from those of the task-only condition (Base). Yet the three conditions gave identical responses to all 100 images, the same rating with word-for-word the same justification. An identity statement without attributes thus changed how the model represented the task but not which images it found interesting (Fig. 6, A).

Correlations between conditions show which images each condition rated higher, independent of its overall rating level (Fig. 6, B). They repeated the ordering found in the variance decomposition. Under the Big Five schema, ratings of the same persona under diferent styles correlated at 0.63 on average, those of diferent personas under the same style at only 0.51, barely more than pairs sharing neither (Table D2). The socio-demographic schema showed the same ordering. Diferent formulations of the same persona thus tended to preserve its image preferences.

Overall, the findings show that the size of a representational change does not directly predict its efect on model behaviour. Between persona and prompting style, the ordering reversed. The style reorganised the representations more strongly, yet the persona had the much stronger efect on how the model judged visual interestingness. The minimal identity statements went further and changed the representations without changing a single response.

A  
![](images/ee4e3085a33858ccbfeb24e39068bbfcbf18faa663df47e9ab9455feda1883c2.jpg)  
B

![](images/cdc5aa78ffd2bf654859177dfba057feed7b217f26dbbf669ff6ffac97dd30a9.jpg)  
Fig. 5 Ratings depend more on the persona than on the prompting style (Big Five schema). (A) Distributions of the 100 image ratings for the task-only condition (Base), the minimal identity statements “You are an AI” (D1) and “You are a human being” (D2), and the five Big Five personas $\mathrm { ( P 1 - P 5 ) }$ ; colours indicate the prompting style (S1–S5). The distributions difer more between personas than between the styles that introduce them, and Base, D1 and D2 are indistinguishable. (B) Share of rating variance explained by the three main efects (image, persona, prompting style), their two-way interactions (P×S, I×P, I×S) and the three-way interaction $( \mathrm { I } \times \mathrm { P } \times \mathrm { S } )$ . Bars show the mean over 500 image-level bootstrap replicates, whiskers the 95% interval. All testable terms were significant in the mixed-efects model $( p \ < \ 0 . 0 0 1$ ; Table D1). Persona explains the largest share and style the smallest of the three main efects. Most of the persona efect is a shift in a persona’s overall rating level, but the $\mathrm { I } \times \mathrm { P }$ term shows that personas also change which images they rate higher. Prompting style, which dominated the global structure of the representations (Fig. 3), thus has little efect on the ratings, whereas the persona, which structured them only finely, has the largest. The same analysis for the socio-demographic schema is shown in Fig. B2.

B  
![](images/3d7a98b7e80df38d83030993e9f24028746187bec85ba3a56da1599b48e0b0fd.jpg)  
Fig. 6 Rating patterns follow the persona more than the prompting style, and minimal identity statements leave them unchanged (Big Five schema). (A) Ratings under Base and the minimal identity statements D1 and D2 plotted against each other on the 1–5 scale; marker area is proportional to the number of images. The three conditions rate all 100 images identically, although D1 and D2 move the representations away from those of Base (Fig. 2). (B) Pearson correlations of the 100 image ratings between all 28 conditions: Base, D1, D2 and the 25 persona × style combinations, ordered by persona (P1–P5) and, within each persona, by style (S1–S5). Each outlined block compares one persona across its five styles. Ratings agree more for the same persona across styles than for the same style across personas, visible as darker blocks along the diagonal that are clearest for P2 and P4; the persona thus shapes which images are rated higher more than the style does. The same analysis for the socio-demographic schema is shown in Fig. B3.

## 4 Discussion

## 4.1 Summary

The present study demonstrates that persona prompting influences a multimodal transformer in a more diferentiated way than a simple input-output comparison would suggest. The strongest changes in the model’s representations were caused by the way a persona was introduced. By contrast, the psychological and biographical characteristics of the simulated personas produced representational diferences several times smaller than those between prompting styles. These diferences nevertheless became increasingly visible in later layers, where the representational geometry shifted from being primarily structured by image identity towards increasingly reflecting persona identity, most clearly when the prompt told the model to be the persona. At the level of the output, however, this hierarchy was reversed. Prompting style had only a limited influence on the ratings despite its large efect on the representations. The specific characteristics of the simulated persona had a substantially stronger efect on how the same images were evaluated. Most of this efect consisted of an overall shift in interestingness ratings, but which images were rated higher also followed the persona more than the style. This held for both schemas, although under the socio-demographic schema the image remained the largest single source of rating variance. The minimal identity statements, a persona frame without attributes, mark the limiting case. They shifted the representations but left every response unchanged. Thus, the factors that most strongly reshape the representations are not necessarily those that most strongly determine the model’s behaviour. Prompt framing primarily reorganises the geometry of internal representations, whereas persona content more strongly determines behavioural judgements.

## 4.2 Interpretation

Why did large representational changes have little or no efect on the ratings? One possibility is that the model represents more than it uses, as language models have been shown to encode information that does not drive their outputs [42, 43]. The shift caused by the minimal identity statements, for instance, was not reflected in the observed responses. It may lie in directions that do not influence the output, or change the output probabilities too little to alter a single greedily decoded token. Posttraining could contribute, as it steers models towards a default assistant persona [14] that corresponds to a direction in activation space [44]. An identity statement may shift the representations without moving the model away from this persona, since it carries nothing that bears on the task, even when it declares the model human. A persona supplies such content, and the ratings follow it.

In-context adaptation may provide a further explanation for why relatively small representational changes can nevertheless produce substantial behavioural efects. Persona information may selectively modulate task-relevant representations without updating the model’s parameters. Even small changes along behaviourally relevant directions could alter the resulting judgements, whereas much larger changes in other directions may leave the responses unchanged. Thus, behavioural relevance may depend less on the overall magnitude of representational change than on which dimensions of the representation space are afected.

## 4.3 Comparison with existing literature

Previous work has shown that personality-related information can be read from hidden activations and steered along dedicated directions [16, 45], and that induced personality profiles change model responses beyond personality questionnaires [46, 47]. Our results extend these findings by showing how image identity, persona content and prompt framing jointly structure the layer-wise representational geometry of a multimodal model, and how this geometry relates to behavioural judgements. They also fit reports that expressed personality and the efects of sociodemographic prompting depend on framing and assigned roles [48, 49], except that in our data the framing outweighed the persona in the representations but far less in the ratings.

Personas in the system prompt have been reported not to improve performance on factual questions [50]. Our minimal identity statements provide a complementary observation: they leave subjective judgements unchanged despite altering the internal representations. Studies relating activations to behaviour have so far examined one kind of factor at a time [16, 24]; comparing persona content with prompt form adds a case in which the factor that dominated the representations was the minor one for the judgements.

That socio-demographic personas shifted the ratings less than Big Five profiles agrees with work on human and simulated judgements. In human annotation data, demographic, social and behavioural variables explain less than 10% of the variance in subjective judgements [20, 51]. In simulations, demographic agents reproduce individual survey answers less accurately than agents grounded in interviews or personality surveys [2], and demographic personas agree closely with one another while missing the views of their groups [52]. These findings suggest that psychological profiles may provide stronger constraints on simulated judgements than demographic descriptions, although our two schemas also difer in format.

In vision-language models, persona conditioning is known to change task performance and how images are interpreted [21, 22]. We extend these findings by tracing the layer-wise emergence of persona-related representational geometry and demonstrating that the magnitude of representational change does not necessarily reflect its behavioural relevance.

## 4.4 Limitations and Future Directions

Our study measures persona sensitivity, not persona fidelity. The dissociation between representations and ratings concerns the model itself, but whether the simulated perspectives resemble those of the people they describe requires ratings from matching human viewers. This matters because LLM simulations are known to flatten identity groups [8], to drift towards caricature [53] and to reflect some populations better than others [7]. The percentile profiles were meant to discourage caricature, but their larger efect could equally reflect exaggeration.

All results come from a single 8B model with greedy decoding, and larger models, other architectures or sampled responses may weigh persona and framing diferently. Each judgement was a single ordinal rating, a constrained format in which persona efects may be stronger than in open-ended responses, as reported for activation steering [17]. Comparing models before and after post-training could clarify the role of the default assistant persona, while causal interventions could identify which representational dimensions actually influence behavioural judgements. In particular, targeted perturbations could test whether small changes along task-relevant representational dimensions produce stronger behavioural efects than larger changes in functionally irrelevant directions. Systematically varying the amount and task relevance of persona information could further reveal whether in-context adaptation selectively engages behaviourally relevant dimensions of the representation space.

Each schema contains five prototypical personas, enough to separate persona from framing but not to map individual traits or demographic variables, which a systematic variation of single attributes could do. Each prompting style had a single wording, and the styles are not orthogonal, since S3 changes both the position of the persona and its introduction and S5 removes the role-play while adding a statistical claim. Part of the diferences between styles may therefore reflect the particular prompts, which several wordings per style and a separate manipulation of position would resolve. The two schemas, finally, difer in vocabulary (verbal versus numeric) and content (biography versus traits), which only a crossed design could separate. Beyond these limitations, persona-conditioned models could support personalised predictions of interestingness, while ensembles of personas could complement the task-only baseline by providing a reference that captures variation across simulated perspectives.

## 4.5 Concluding Remarks

The central finding of this study is a dissociation between representational change and behavioural relevance. A persona’s frame reorganised the representations of a multimodal model, and even a bare statement that the model is an AI or a human shifted them, yet the model’s judgements of the images followed what the persona contained. Two practical consequences follow. Studies that simulate people with personas should report and vary the prompt form, because it changes how strongly a persona shows in the judgements. Studies that read personas from activations should check representational diferences against behaviour, because the most prominent ones need not be the ones the model uses. More broadly, consistent with the simulator view of language models [12, 14], these findings suggest that a model may represent multiple contextdependent perspectives without all of them influencing its behaviour. Distinguishing such latent representations from behaviourally efective ones may be important not only for persona simulation but also for understanding implicitly emerging personas and their implications for alignment [18, 19].

Data accessibility. The prompt registry, the inference and hidden-state extraction pipeline, all analysis code and the scripts that produce every figure and table are available at https://github.com/MathisImm/persona-vlm-representations and archived at https://doi.org/10.5281/zenodo.23665655. The data are archived separately at https: //doi.org/10.5281/zenodo.23664820: the text of all 53 prompts, the model ratings and explanations for all 5,300 image–prompt cells, the hidden states (about 2 GB), the layer-wise GDV results and the projections for every layer. The images come from the Flickr subset used by Abdullahu and Grabner [26]; the 100 image identifiers used here are listed in the data archive.

Declaration of AI use. We used Claude (Anthropic), through the Claude Code assistant, in three ways. For writing, it proposed rewordings of sentences and paragraphs to improve clarity, concision and consistency; the authors reviewed each proposal and accepted, revised or rejected it. For programming, it assisted in writing, refactoring and testing the code for the analyses and figures and in preparing the public code repository and data archive. For checking, it compared statements in the manuscript with the code and the data, and re-ran the authors’ analysis scripts to confirm that all figures and numbers are reproducible. The research question, the experimental design, the choice of analyses, the interpretation of the results and the conclusions are the authors’ own. The authors checked all AI-assisted text, code and output and take full responsibility for the content of this article.

Funding. This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation): grants KR 5148/3-1 (project number 510395418), KR 5148/5-1 (project number 542747151), KR 5148/10-1 (project number 563909707), and GRK 2839 (project number 468527017) to PK, and grants SCHI 1482/3-1 (project number 451810794) and SCHI 1482/6-1 (project number 563909707) to AS.

Acknowledgements. The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universit¨at Erlangen-N¨urnberg (FAU). The hardware is partially funded by the German Research Foundation (DFG).

## Appendix A Derivation of Psychometric Profiles

The psychometric profiles of the Big Five schema are stylised percentile vectors designed to provide the model with distinct, theoretically grounded trait configurations. Rather than sampling traits at random, these profiles were constructed to reflect qualitative patterns identified in established psychometric research. To facilitate computational modelling, the raw empirical values (which are typically reported as continuous Z-scores) were translated into idealised percentiles rounded to the nearest multiple of five.

In the five-factor model [33], openness captures curiosity and receptivity to novel experiences, conscientiousness organisation and self-control, extraversion sociability and energy, agreeableness compassion and cooperativeness, and neuroticism the tendency towards negative afect and stress.

## Profiles derived from large-scale data clustering.

Three of the profiles map onto the robust personality types identified by Gerlach et al. [34], who analysed multi-million-user datasets to find replicable trait clusters across populations.

The Reserved (P1) Mirrors the “Reserved” cluster, characterised by emotional stability (low Neuroticism) and low Openness and Extraversion. Conscientiousness and Agreeableness, only moderately elevated in that cluster, are set high here.

The Role Model (P2) Captures the “Role Model” cluster, which exhibits socially desirable traits across all dimensions. This profile features high levels of Extraversion, Openness, Agreeableness, and Conscientiousness, alongside very low Neuroticism.

The Normative (P5) Corresponds to the “Average” cluster in the Gerlach typology. It reflects a highly typical population configuration with moderately high Neuroticism and slightly lower Openness. Extraversion, high in that cluster, is kept average here.

## Profiles derived from additional frameworks.

To ensure a comprehensive coverage of the Big Five trait space, two additional profiles were synthesised from distinct psychological traditions.

The Undercontrolled (P3) Rooted in the person-centred developmental typologies of ego-control and ego-resiliency [54]. The undercontrolled phenotype is typically characterised by low Conscientiousness and low Agreeableness paired with high Extraversion and Neuroticism, reflecting a tendency towards impulsivity and externalising behaviours.

The Highly Sensitive (P4) Drawn from the framework of Sensory Processing Sensitivity [55]. This profile models individuals who process internal and external stimuli deeply. It is defined by high Openness (reflecting aesthetic sensitivity and intellectual curiosity), high Agreeableness (strong empathy), and high Neuroticism (susceptibility to overstimulation and stress).

By grounding the synthetic archetypes in these diverse taxonomies, the experimental manipulation covers a realistic yet systematically dispersed range of human behavioural tendencies without relying on unprincipled trait combinations.

## Appendix B Socio-demographic Personas

The main text reports the Big Five schema, whose numeric trait profiles make the persona manipulation explicit and quantitative. Every analysis was run identically on the socio-demographic schema, whose personas are specified by country of origin, age band, gender, professional background, and transient state. This appendix reports that replication: the layer-wise clustering measures (Fig. B1), the rating distributions and variance decomposition (Fig. B2), and the cross-condition agreement structure (Fig. B3).

The qualitative pattern of the main text holds under the socio-demographic schema: prompt style dominates the global organisation of the representational space while persona identity emerges in the later decoder layers. The two schemas difer in one respect that deserves attention. Under the Big Five schema the persona accounts for the largest share of rating variance, whereas under the socio-demographic schema image content does, with the persona second (Table D1). A socio-demographic profile therefore shifts the model’s judgements less forcefully than an explicit trait vector, which is consistent with the Big Five schema stating the psychological disposition directly rather than leaving it to be inferred from biography.

## Appendix C Clustering by Prompting Style

The per-style curves in Fig. 4 and Fig. B1 each contain a single prompting style and therefore cannot compare clustering by style with clustering by persona. Fig. C1 computes the GDV on the representations of all five styles together. Because the styles separate the representations, each persona and each image now falls into five style-specific sub-clusters, so their GDV values are smaller in magnitude than in the per-style curves. The values in Fig. C1 should therefore be compared with each other rather than with those in Fig. 4 and Fig. B1, whereas the layers at which persona clustering emerges and image clustering recedes are the same.

![](images/ce7a2192bbaa0136784cffc06636c376f240341505debfb59e002a38ac410a44.jpg)

![](images/b591c9d144c42bf5688b7c614eda4a5ec8f83af96153d39dea7dd32a6ba4b073.jpg)

![](images/5733988479f134d97f5e98009675070ee5c125172f0601c035ad11bd9f4b42cb.jpg)

![](images/ad47669f336dc8c16b22688788704e8e5974cb7c406562963d94b09fec9ce604.jpg)

![](images/dddd503a45d4e75832774eb46c7b54f3a22c19b1cca2692ff80009fb27041d82.jpg)

![](images/98f4f34bb3832f51c170f869025afd22850809141392e48e0a858ef26316c3f3.jpg)  
Fig. B1 Layer-wise transition from image to persona and rating structure in Qwen3- VL-8B (socio-demographic schema). As in Fig. 4, GDV values quantify clustering in the original high-dimensional hidden-state space at every decoder layer; more negative values indicate stronger cluster separation, and each curve is one prompting style (S1–S5). The projection panels show twodimensional MDS projections at layer 36 for a 20-image subset, one per style, with the GDV of that projection in each title. (A1) GDV by image. Image clustering dominates the early layers and weakens abruptly at layer 12 under every style except S5, where it weakens only around layer 20. (A2) Projections coloured by image. (B1) GDV by persona (P1–P5). Persona clustering emerges at layer 12, earlier than for the Big Five personas, abruptly under every style except S5, where it grows slowly. It is strongest under S4, apart from a peak under S3 at layer 1 that falls of sharply at layer 2. (B2) Projections coloured by persona. (C1) GDV by assigned rating. Rating clustering increases markedly in later layers only under S2. (C2) Projections coloured by rating; Extremely Interesting is marked absent because it does not occur in the 20-image subset.

A  
![](images/7beee39b3967e42803b13b964141f45033eb387cdd25c415a5bb5c7972f3d843.jpg)  
B

![](images/311b5fd317cea955f307e37f22f5a8feee61137dca9eaf0bab316a5145ba6165.jpg)  
Fig. B2 Ratings depend most on the image, then on the persona, and least on the prompting style (socio-demographic schema). (A) Distributions of the 100 image ratings for Base, the minimal identity statements D1 and D2, and the five socio-demographic personas (P1– P5); colours indicate the prompting style (S1–S5), as in Fig. 5. The distributions difer less between personas than under the Big Five schema, with P4 rating lowest. (B) Share of rating variance by component, as in Fig. 5B; all testable terms were significant in the mixed-efects model $( p < 0 . 0 0 1 ;$ Table D1). Unlike under the Big Five schema, the image explains the largest share and the persona the second largest, while style remains the smallest main efect; personas change which images they rate higher (I×P) slightly more than under the Big Five schema.

## Appendix D Variance Decomposition, Model Comparison and Rating Agreement

Table D1 tabulates the quantities plotted in Fig. 5B and Fig. B2B for both schemas, so that the numbers can be read of without measuring bars. The two schemas are decomposed separately (Section 2.6), each over $1 0 0 \times 5 \times 5 = 2 { , } 5 0 0$ ratings.

![](images/151c2cdc6319d5b5f2add3b93f527c8eedca69ac462aa6348149e040d4b5b302.jpg)  
Fig. B3 Rating patterns follow the persona more than the prompting style (sociodemographic schema). (A) Ratings under Base and the minimal identity statements D1 and D2 plotted against each other, as in Fig. 6A. (B) Pearson correlations of the 100 image ratings between all 28 conditions, ordered as in Fig. 6B. As under the Big Five schema, ratings agree more for the same persona across styles than for the same style across personas, with the clearest blocks for P2 and P5, and they stay closest to those of Base when the persona is placed after the task (S3).

Two contrasts are worth naming. First, the persona share roughly doubles from the socio-demographic to the Big Five schema (16.8% to 33.4%) and overtakes image content, while the prompt-style share is essentially identical in the two schemas (5.4% versus 5.6%): what the persona is stated to be matters far more than how it is introduced, and this holds for both schemas. Second, the persona × style interaction is roughly three times larger under the Big Five schema, meaning trait profiles are not merely stronger but also more sensitive to framing than biographical ones.

Every term is significant by likelihood-ratio test in both schemas, including the interaction, so the small percentages attached to the style terms should be read as small efects that are reliably present rather than as absent efects.

Table D2 summarises the correlation matrices of Fig. 6B and Fig. B3B by what a pair of conditions shares, for both schemas.

![](images/4f6d2034d83539582bc53507f5df6a20d396612cd6e64fed433f49c82b636744.jpg)

Fig. C1 Clustering by prompting style exceeds clustering by persona at every layer. GDV in the original hidden-state space at every decoder layer, computed on the representations of all five prompting styles together (2,500 per layer) and grouped by prompting style (S1–S5), by persona (P1–P5), by image and by assigned rating; more negative values indicate stronger clustering. (A) Big Five schema. (B) Socio-demographic schema. Style and persona each form five classes of equal size, so their values are directly comparable. Clustering by style is strong from the first layers on and peaks around layers 18–22, whereas clustering by persona appears only in later layers and remains several times weaker. Rating clustering reflects persona and style together, since both shift the rating level.  
Table D1 Rating variance and mixed-efects model comparison, per schema. (a) Sum-of-squares variance decomposition over the crossed persona × prompt-style cells; entries are the mean over 500 image-level cluster-bootstrap replicates with the 2.5th–97.5th percentile interval, i.e. exactly what ${ \mathrm { F i g . } }$ . 5B and ${ \mathrm { F i g . } }$ B2B plot. With one rating per image, persona and style, the three-way interaction is the residual; it is not sampling noise, because decoding is greedy. (b) Likelihood-ratio tests of the mixed-efects model (Section 2.7) against the nested model omitting each term.  
(a) Variance explained (%), mean [95% CI]
<table><tr><td>Component</td><td>Socio-demographic</td><td>Big Five</td></tr><tr><td>Image</td><td>44.5 [37.6, 50.8]</td><td>25.8 [20.2, 31.4]</td></tr><tr><td>Persona</td><td>16.8 [13.3, 20.5]</td><td>33.4 [30.3, 36.9]</td></tr><tr><td>Prompt style</td><td>5.4 [ 4.1, 7.0]</td><td>5.6 [ 4.3, 7.0]</td></tr><tr><td>Persona × style</td><td>3.3 [ 2.6, 4.1]</td><td>9.5 [ 7.8, 11.3]</td></tr><tr><td>Image × persona</td><td>11.0 [ 9.1, 13.2]</td><td>8.5 [ 7.2, 9.8]</td></tr><tr><td>Image × style</td><td>6.9 [ 6.0, 8.1]]</td><td>4.8 [ 4.2, 5.4]</td></tr><tr><td>Image × persona × style</td><td>12.1 [10.7, 13.3]</td><td>12.5 [11.7, 13.2]</td></tr></table>

(b) Likelihood-ratio tests
<table><tr><td></td><td colspan="2">Socio-demographic</td><td colspan="2">Big Five</td></tr><tr><td>Term</td><td> $\chi ^ { 2 }$  (df)</td><td>p</td><td> $\chi ^ { 2 }$  (df)</td><td>p</td></tr><tr><td>Persona</td><td>367.1 (4)</td><td>&lt; 0.001</td><td>633.3 (4)</td><td>&lt; 0.001</td></tr><tr><td>Prompt style</td><td>224.2 (4)</td><td>&lt; 0.001</td><td>303.7 (4)</td><td>&lt; 0.001</td></tr><tr><td>Persona × style</td><td>370.8 (16)</td><td>&lt; 0.001</td><td>889.7 (16)</td><td>&lt; 0.001</td></tr><tr><td>Image × persona (variance)</td><td>331.0 (1)</td><td>&lt; 0.001</td><td>192.3 (1)</td><td>&lt; 0.001</td></tr><tr><td>Image × style (variance)</td><td>128.1 (1)</td><td>&lt; 0.001</td><td>31.5 (1)</td><td>&lt; 0.001</td></tr></table>

Table D2 Mean pairwise Pearson correlation of the 100 image ratings, per schema, averaged over the of-diagonal cells of Fig. 6B and ${ \mathrm { F i g . } }$ B3B. The rows group the 25 persona × style conditions by what a pair shares.
<table><tr><td>Pairs of conditions</td><td>Socio-demographic</td><td> $\mathbf { B i g }$  Five</td></tr><tr><td>Same persona, different style</td><td>.69</td><td>.63</td></tr><tr><td>Same style, different persona</td><td>.62</td><td>.51</td></tr><tr><td>Neither shared</td><td>.56</td><td>.48</td></tr></table>

## References

[1] Argyle, L.P., Busby, E.C., Fulda, N., Gubler, J.R., Rytting, C., Wingate, D.: Out of one, many: Using language models to simulate human samples. Political Analysis 31(3), 337–351 (2023)

[2] Park, J.S., Zou, C.Q., Kamphorst, J., Egan, N., Shaw, A., Hill, B.M., Cai, C., Morris, M.R., Liang, P., Willer, R., et al.: Llm agents grounded in self-reports enable general-purpose simulation of individuals. arXiv preprint arXiv:2411.10109 (2024)

[3] Park, J.S., O’Brien, J., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S.: Generative agents: Interactive simulacra of human behavior. In: Proceedings of the 36th Annual Acm Symposium on User Interface Software and Technology, pp. 1–22 (2023)

[4] Hu, Z., Lian, J., Xiao, Z., Xiong, M., Lei, Y., Wang, T., Ding, K., Xiao, Z., Yuan, N.J., Xie, X.: Population-aligned persona generation for llm-based social simulation. arXiv preprint arXiv:2509.10127 (2025)

[5] Li, X., Hao, Y., Hou, J., Huang, J., Wen, Q., Huang, S., Liu, Y., Liu, X., Fan, Y., Wang, Y., et al.: MatrAIx: Simulating the world with 8.3 billion persona agents. arXiv preprint arXiv:2608.04205 (2026)

[6] METR: Brief independent investigation of agents’ behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident. https://metr.org/blog/ 2026-08-26-openai-hugging-face-incident-investigation/ (2026)

[7] Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., Hashimoto, T.: Whose opinions do language models reflect? In: International Conference on Machine Learning, pp. 29971–30004 (2023). PMLR

[8] Wang, A., Morgenstern, J., Dickerson, J.P.: Large language models that replace human participants can harmfully misportray and flatten identity groups. Nature Machine Intelligence 7(3), 400–411 (2025)

[9] Li, K., Hopkins, A.K., Bau, D., Vi´egas, F., Pfister, H., Wattenberg, M.: Emergent world representations: Exploring a sequence model trained on a synthetic task. arXiv preprint arXiv:2210.13382 (2022)

[10] Gurnee, W., Tegmark, M.: Language models represent space and time. arXiv preprint arXiv:2310.02207 (2023)

[11] Immertreu, M., Schilling, A., Maier, A., Krauss, P.: Probing for consciousness in machines. Frontiers in artificial intelligence 8, 1610225 (2025)

[12] Janus: Simulators. AI Alignment Forum. Accessed: 2026-04-10 (2022). https:// www.alignmentforum.org/posts/vJFdjigzmcXMhNTsx/simulators

[13] Shanahan, M., McDonell, K., Reynolds, L.: Role play with large language models. Nature 623(7987), 493–498 (2023)

[14] Anthropic: The persona selection model. AI Alignment Forum. Accessed: 2026- 04-10 (2026). https://www.alignmentforum.org/posts/dfoty34sT7CSKeJNn/ the-persona-selection-model

[15] Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., et al.: Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405 (2023)

[16] Chen, R., Arditi, A., Sleight, H., Evans, O., Lindsey, J.: Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509 (2025)

[17] Frising, M., Balcells, D.: Linear personality probing and steering in llms: A big five study. arXiv preprint arXiv:2512.17639 (2025)

[18] Betley, J., Tan, D., Warncke, N., Sztyber-Betley, A., Bao, X., Soto, M., Labenz, N., Evans, O.: Emergent misalignment: Narrow finetuning can produce broadly misaligned llms. arXiv preprint arXiv:2502.17424 (2025)

[19] Betley, J., Cocola, J., Feng, D., Chua, J., Arditi, A., Sztyber-Betley, A., Evans, O.: Weird generalization and inductive backdoors: New ways to corrupt llms. arXiv preprint arXiv:2512.09742 (2025)

[20] Hu, T., Collier, N.: Quantifying the persona efect in llm simulations. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10289–10307 (2024)

[21] Jia, P., Jia, H., Miao, Z., Du, L., Wang, Y., Su, Z.: Modeling complex behaviors: Multi-personality composition and dynamic switching in vision-language models. arXiv preprint arXiv:2606.11074 (2026)

[22] Silva, N., Delgado, M., Minetto, R., Silver, D., Silva, T.H.: Analyzing persona efects in generated explanations from multimodal llm agents in urban perception. arXiv preprint arXiv:2605.29064 (2026)

[23] Sclar, M., Choi, Y., Tsvetkov, Y., Suhr, A.: Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In: International Conference on Learning Representations, vol. 2024, pp. 25055–25083 (2024)

[24] Kamruzzaman, M., Das, S.K., Kim, G.L.: Demographic prompting at scale: When more attributes hurt llm–human agreement. arXiv preprint arXiv:2607.10590 (2026)

[25] Berlyne, D.E.: Aesthetics and Psychobiology. The Century Psychology Series, p. 336. Appleton-Century-Crofts, New York (1971)

[26] Abdullahu, F., Grabner, H.: Commonly interesting images. In: European Conference on Computer Vision, pp. 180–198 (2024). Springer

[27] Abdullahu, F., Grabner, H.: Visual interestingness decoded: How gpt-4o mirrors human interests. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15350–15364 (2025)

[28] Zhang, A., Chen, Y., Sheng, L., Wang, X., Chua, T.-S.: On generative agents in recommendation. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1807–1817 (2024)

[29] Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., Zhu, K.: Qwen3-VL Technical Report (2025). https://arxiv.org/abs/2511.21631

[30] Schilling, A., Maier, A., Gerum, R., Metzner, C., Krauss, P.: Quantifying the separability of data classes in neural networks. Neural Networks 139, 278–293 (2021)

[31] Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al.: Qwen technical report. arXiv preprint arXiv:2309.16609 (2023)

[32] Goldberg, L.R.: An alternative” description of personality”: the big-five factor structure. Journal of personality and social psychology 59(6), 1216 (1990)

[33] McCrae, R.R., John, O.P.: An introduction to the five-factor model and its applications. Journal of personality 60(2), 175–215 (1992)

[34] Gerlach, M., Farb, B., Revelle, W., Nunes Amaral, L.A.: A robust data-driven approach identifies four personality types across four large data sets. Nature human behaviour 2(10), 735–742 (2018)

[35] Jaworska, N., Chupetlovska-Anastasova, A.: A review of multidimensional scaling (mds) and its utility in various psychological domains. Tutorials in quantitative methods for psychology 5(1), 1–10 (2009)

[36] Steyvers, M.: Multidimensional scaling. Encyclopedia of cognitive science 1, 1–5 (2002)

[37] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019)

[38] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al.: Transformers: State-of-the-art natural language processing. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45 (2020)

[39] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al.: Scikit-learn: Machine learning in python. the Journal of machine Learning research 12, 2825–2830 (2011)

[40] Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., et al.: Scipy 1.0: fundamental algorithms for scientific computing in python. Nature methods 17(3), 261–272 (2020)

[41] Seabold, S., Perktold, J., et al.: Statsmodels: econometric and statistical modeling with python. scipy 7(1), 92–96 (2010)

[42] Elazar, Y., Ravfogel, S., Jacovi, A., Goldberg, Y.: Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics 9, 160–175 (2021)

[43] Gekhman, Z., David, E.B., Orgad, H., Ofek, E., Belinkov, Y., Szpektor, I., Herzig, J., Reichart, R.: Inside-out: Hidden factual knowledge in LLMs. arXiv preprint arXiv:2503.15299 (2025)

[44] Lu, C., Gallagher, J., Michala, J., Fish, K., Lindsey, J.: The assistant axis: Situating and stabilizing the default persona of language models. arXiv preprint arXiv:2601.10387 (2026)

[45] Feng, X., Zhao, L., Zhong, W., Gu, Y., Kong, L., Feng, X., Qin, B., et al.: Persona: dynamic and compositional inference-time personality control via activation vector algebra. In: International Conference on Learning Representations, vol. 2026, pp. 151839–151879 (2026)

[46] Jiang, H., Zhang, X., Cao, X., Breazeal, C., Roy, D., Kabbara, J.: Personallm: Investigating the ability of large language models to express personality traits. In: Findings of the Association for Computational Linguistics: NAACL 2024, pp. 3605–3627 (2024)

[47] Serapio-Garc´ıa, G., Safdari, M., Crepy, C., Sun, L., Fitz, S., Romero, P., Abdulhai, M., Faust, A., Matari´c, M.: A psychometric framework for evaluating and shaping personality traits in large language models. Nature Machine Intelligence 7(12), 1954–1968 (2025)

[48] Nagao, M., Terao, K., Nakano, M., Iwahashi, N.: Personality, role, and expressive style in large language models: An interactionist analysis. arXiv preprint arXiv:2605.28037 (2026)

[49] Beck, T., Schuf, H., Lauscher, A., Gurevych, I.: Sensitivity, performance, robustness: Deconstructing the efect of sociodemographic prompting. In: Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2589–2615 (2024)

[50] Zheng, M., Pei, J., Logeswaran, L., Lee, M., Jurgens, D.: When “a helpful assistant” is not really helpful: Personas in system prompts do not improve performances of large language models. In: Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 15126–15154 (2024)

[51] Orlikowski, M., R¨ottger, P., Cimiano, P., Hovy, D.: The ecological fallacy in annotation: Modeling human label variation goes beyond sociodemographics. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1017–1029 (2023)

[52] Yang, J., Hechtbauer, M., Khalilov, E., Brinkmann, E.L., Schmitt, V., Feldhus, N.: Persona prompting as a lens on llm social reasoning. In: Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1152–1170 (2026)

[53] Cheng, M., Piccardi, T., Yang, D.: Compost: Characterizing and evaluating caricature in llm simulations. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 10853–10875 (2023)

[54] Robins, R.W., John, O.P., Caspi, A., Mofitt, T.E., Stouthamer-Loeber, M.: Resilient, overcontrolled, and undercontrolled boys: three replicable personality types. Journal of Personality and Social psychology 70(1), 157 (1996)

[55] Aron, E.N., Aron, A.: Sensory-processing sensitivity and its relation to introversion and emotionality. Journal of personality and social psychology 73(2), 345 (1997)