# The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language Models

Shihui Li\*, Mette Andersen<sup>∗</sup>, Martha Lewis

Institute for Logic, Language, and Computation

University of Amsterdam

shihui.li@student.uva.nl, {m.f.andersen, m.a.f.lewis}@uva.nl

## Abstract

Metaphor is a crucial aspect of human cognition. As such, we would like to understand how it is represented in large language models (LLMs) and how it might be controlled. We use mechanistic interpretability techniques to identify causal subspaces in a range of LLMs covering different architectures and sizes. We show that we are able to steer output on classification tasks across different parts of speech. We are also able to steer output on generation tasks, indicating that these techniques can be used to control the metaphoricity of LLM output.

## 1 Introduction

Metaphor is an essential part of everyday human language use. Sources such as Steen et al. (2010); Shutova (2010) show that metaphor pervades language. Furthermore, metaphor forms a basis for the fundamental creativity and concept formation that we are capable of (Lakoff and Johnson, 2008). Large language models such as Llama 3 (Grattafiori et al., 2024) have some capacity to understand and produce metaphorical language, although whether they are up to human performance is debated (Chakrabarty et al., 2021; Stowe et al., 2022; Ichien et al., 2024).

One way to understand the capacities of large language models is via mechanistic interpretability: by examining the internals of large language models via, for example, the Logit Lens<sup>1</sup>, causal interventions (Meng et al., 2022), or circuit discovery (Conmy et al., 2023). Sharkey et al. (2025) provides a thorough review and discussion of open problems in the area. One recent paper, Marks and Tegmark (2024), argued that Llama models exhibited a ‘truth direction’ at particular layers, which was common to different datasets, and that the models’ output could be steered by causal intervention on this layer.

Finding such a direction for metaphor would give insight into how models represent metaphor. This has been investigated to some extent (Boisson et al., 2024) but not yet at the mechanistic level. Finding this direction could also provide a useful ‘dial’ that could potentially be used to steer the metaphoricity or literalness of output text, using techniques from e.g. Li et al. (2023). In this paper, we use activation patching (Meng et al., 2022) and distributed alignment search (Geiger et al., 2024) to find a ‘metaphoricity direction’ or subspace within a given model. We show that these directions are causally effective: we are able to change the output of models in classification tasks for metaphor identification and interpretation; and in a generation task we are able to steer the model towards more metaphorical output.

## Contributions

• We identify directions and subspaces for metaphoricity in a range of models, and show that these have some causal effect.

• We show that these directions are somewhat consistent across different metaphor datasets.

• We test seven models from four different model families (Gemma (Riviere et al., 2024), Qwen (Bai et al., 2023), Llama (Grattafiori et al., 2024), Mistral (Jiang et al., 2023)) across small (7-9B) and medium-large (27- 70B) sizes.

• We show that steering models in a generation task can change model output from more literal to more metaphorical.

## 2 Background and Related work

Metaphor identification and interpretation In metaphor processing, research has focussed on two main tasks: metaphor identification, that is, identifying whether a word is metaphorical, and metaphor interpretation, that is, identifying the meaning of a metaphorical utterance. Work in using LLMs for metaphor identification include Fuoli et al. (2026); Reimann and Scheffler (2025). On the metaphor interpretation side, there has been a range of work showing that modern LLMs have relatively strong performance at understanding metaphor, especially when fine-tuned (Ichien et al., 2024; Liu et al., 2022b). Since metaphor is a key part of human language, we would like to understand how LLMs interpret and identify metaphor.

Activation Patching Mechanistic interpretability techniques attempt to understand and affect the output of large language models by inspecting their internal representations, for example by looking at the vectors of activations produced at each layer of the model (Meng et al., 2022; Marks and Tegmark, 2024). A common technique is to then causally intervene on these representations. This means that we examine how a model’s output changes when we alter its internal representations, for example by activation patching, introduced in Meng et al. (2022); that is, we save the activations for one input, and patch these into the activations for another input, examining which layers and tokens affect the output.

In Marks and Tegmark (2024), the authors use activation patching to find a combination of layer and token input that represents a concept of truth in a language model.

Distributed Alignment Search (DAS) (Geiger et al., 2024) is a method for finding causal concepts in model activations, based on the assumption that full-vector activations may be polysemantic, i.e. may be activated for different concepts. The method has been used in existing benchmarks (Mueller et al., 2025) for causal variable localization, and compared to other methods (SAEs, PCA, full-vector), DAS has been shown to find concepts that achieve superior causal performance. In this paper, we use DAS to find a causal concept of metaphoricity in the model’s residual stream, and compare the performance of steering using DAS with the existing mass-mean steering used by Marks and Tegmark (2024).

Controlling Generation Controlling the style of LLM generated text is also an active area of research. Techniques used include training a separate model (Liu et al., 2022a), or prompt-based style control such as in (Liu et al., 2024). Relatedly, Li et al. (2023) alter the truthfulness of generated text via inference-time interventions.

Summary Metaphor identification, interpretation, and generation are key aspects of language use. So far, in LLMs, these abilities have mostly been investigated by looking at behavioural outputs. In this paper, we use activation patching and distributed alignment search to investigate firstly whether metaphorical and literal sentences are linearly separable in activation space, secondly whether we can steer models towards literal or metaphoric output, and lastly whether the steering vectors are common to different datasets. We control the metaphoricity of generated output using the inference-time intervention technique introduced by Li et al. (2023). To our knowledge, this is the first attempt to control the metaphoricity of LLM output via intervention on the activations, rather than via prompting or building an external model.

## 3 Datasets and Tasks

## 3.1 Datasets

We use three datasets covering two different parts of speech (nouns and adjectives)<sup>2</sup>. Using mechanistic interpretability techniques requires that models are able to perform to a high standard on the task, and the structure of the dataset also needs to be fairly controlled. We therefore use relatively simply structured datasets. Following Marks and Tegmark (2024), we use datasets with the following properties: 1) each sentence contains exactly one metaphorical target word; 2) sentences are short and structurally simple; 3) within each dataset, sentences have the same syntactic structure.

We use the following three datasets. Examples of sentences with metaphorical nouns and adjectives are shown in Table 1.

Noun dataset (NOUN). We start from the small training set of Fig-QA (Liu et al., 2022b), which contains 200 human-written creative metaphors. Each example is paired with one correct rephrase and one incorrect rephrase in a Winograd-style format. Every two metaphor examples form a pair representing the meaning of a particular rephrase.

<table><tr><td>Category</td><td>Example sentence</td></tr><tr><td>Noun</td><td>The book is a puzzle.</td></tr><tr><td>Adj</td><td>The welcome is warm.</td></tr></table>

Table 1: Example sentences with metaphorically used nouns and adjectives. In example sentences, metaphorical target words are coloured blue.

We first remove idioms, multi-clause descriptions, and long narrative sentences (e.g., “The painting has the creativity of Rembrandt and Picasso getting their paintings married”). We then use ChatGPT (OpenAI, 2025) to rewrite the remaining examples into short, direct metaphors with the parallel structure “The noun<sub>1</sub> is (a/an) noun<sub>2</sub>”. For example, the original sentence “The bear was the size of a whale after eating a smaller whale” is simplified to “The bear is a whale”. In each sentence, noun<sub>2</sub> metaphorically describes noun<sub>1</sub>. We manually verify that each sentence is grammatically correct and preserves the original metaphorical mappings. The final dataset contains 130 sentence tuples. In the rest of the paper we refer to this dataset as NOUN.

Adjective dataset (ADJ). We use the ADJ+N metaphor/literal dataset proposed by Tsvetkov et al. (2014). The test set contains 100 adjective–noun pairs with both literal and metaphorical usages. We modify this dataset to have a sentence format using the template “The N is ADJ.” In this dataset, the adjective carries the metaphorical meaning and modifies the noun. In the remainder of this paper, we refer to this dataset as ADJ.

Domain-specific adjective dataset (DOM). We also use the dataset by Gutiérrez et al. (2016) for DAS training, steering prediction and steering generation. The dataset consists of adjective-noun pairs across 8 different domains (clarity, depth, light, strength, substance, taste, temperature, texture). Each sample is labelled literal (e.g. “warm coffee”) or metaphorical (e.g. “warm welcome”). We evaluate on n=100 per steering direction from the held-out set not used for training DAS. In the remainder of this paper we refer to this dataset as DOM.

## 3.2 Tasks

We carry out an identification task, an interpretation task, and a generation task.

Identification Models are prompted with a four-shot setup to label a sentence as either ‘literal’ or ‘metaphoric’. The adjective datasets from Tsvetkov et al. (2014) and Gutiérrez et al. (2016) are in the form adjective noun. We rephrase these into sentences of the form ‘The noun is adjective’. The full prompt for ADJ is then as follows. See Appendix A for the prompt for NOUN

The town is sleepy. This sentence is: metaphoric The sand is white. This sentence is: literal The moon is young. This sentence is: metaphoric The banana is ripe. This sentence is: literal sentence. This sentence is:

Interpretation Models are prompted with a four-shot setup. In each shot, a metaphorical sentence and a candidate paraphrase are given, and the paraphrase is labelled as ‘apt’ or ‘inapt’. This task uses NOUN, which is based on FigQA (Liu et al., 2022b). Apt/inapt pairs are taken from their setup. The full prompt is given below:

Determine whether the paraphrase is apt or inapt.

The politician is glass: The politician is transparent. This paraphrase is: apt.

The politician is a wall: The politician is transparent. This paraphrase is: inapt.

The cake is heaven: The cake is not delicious. This paraphrase is: inapt.

The cake is heaven: The cake is delicious. This paraphrase is: apt.

sentence: rephrase. This paraphrase is:

Generation Models are prompted with a five-shot setup to output an interpretation of a sentence that could be either literal or metaphorical. The dataset used is DOM, and we reword adjective-noun pairs into sentences of the form ‘The noun was adjective’. The full prompt is given below:

The welcome was warm. The meaning of ‘warm’ is: affectionate and emotionally inviting. The lamp was bright. The meaning of ‘bright’ is: emitting a strong, clear light.

The novel was deep. The meaning of ‘deep’ is: intellectually complex and rich in meaning. The stone was smooth. The meaning of ‘smooth is: having an even surface with no roughness. The defeat was bitter. The meaning of ‘bitter’ is: painful, hard to accept, and leaving resentment. The noun was adj. The meaning of ‘adj’ is:

## 4 Methods

## 4.1 Identifying areas encoding metaphoricity

For each of the datasets and tasks described above, we use one of two techniques designed to isolate the part of the model used to encode metaphoricity. For the identification and interpretation tasks across NOUN, ADJ, and DOM, we use activation patching. For DOM and for the generation task, we also use distributed alignment search (DAS) to identify a linear subspace encoding metaphoricity. We use the mass-mean probe from Marks and Tegmark (2024) and the learned DAS subspaces to steer model output in the identification and interpretation tasks, requiring single-token output. Finally, we use the mass-mean probe and DAS subspaces to steer generation over multiple tokens, finding that it is possible to steer models towards more metaphorical or more literal output.

## 4.1.1 Activation Patching

In the identification task, we use the following procedure. On ADJ, we insert the following sentence into the prompt template in Section 3.2 to form a prompt $p _ { L } = T h e$ skin is dry. For prompt $p _ { L }$ we expect the output ‘literal’. We then form a paired prompt $p _ { M }$ replacing skin with wit. For this prompt we expect the output ‘metaphoric’. To find out which hidden states contain information about the metaphoricity of the sentence, we run the model on the input $p _ { M }$ and cache the activations $h _ { i , \ell } ( p _ { M } )$ for each token position i and layer ℓ. Then, for each i and ℓ, we run the model on $P _ { L }$ and modify the model’s forward pass by swapping $h _ { i , \ell } ( P _ { L } )$ for $h _ { i , \ell } ( P _ { M } )$ . Finally, we record the difference in log probability between the tokens "metaph"<sup>3</sup> and "literal". The larger the difference, the more influential the hidden state in position i and layer ℓ on the model’s prediction. For the dataset NOUN, we use the prompt pair ${ p } _ { L } \mathrm { = } T h e$ breeze is cold and $\scriptstyle { p _ { M } = T h e }$ breeze is ice. For the interpretation task, we use the prompt pair $p _ { i n a p t }$ $= T h e$ puzzle is rocket science: The puzzle is easy and $p _ { a p t } = T h e$ puzzle is rocket science: The puzzle is difficult.

## 4.1.2 Distributed Alignment Search

We use Distributed Alignment Search (DAS) (Geiger et al., 2024) to localize, if it exists, a lowdimensional, linear subspace encoding metaphoricity. After identifying the most causally implicated token-position in the residual stream using fullvector activation patching, DAS trains a rotation matrix on top of these activations. The rotation matrix is trained to maximize Interchange Intervention Accuracy (IIA), which is the fraction of samples where patching flips model prediction in the expected direction.

Concretely, let h $\in \mathbb { R } ^ { d }$ denote the residual stream at a fixed layer ℓ and token position p. As will be seen in Figure 3, the full stop token at the end of the target sentence in each prompt is highly causally implicated. We perform DAS at this token at the layer that optimises the IIA score.

DAS learns a matrix $\textbf { R } \in \mathbb { R } ^ { d \times k }$ with orthonormal columns $( \mathbf { R } ^ { \top } \mathbf { R } = \mathbb { I } _ { k } )$ , defining a kdimensional subspace of the representation space. Given a source hidden state $\mathbf { h } ^ { \mathrm { s o u r c e } }$ and a base hidden state $\mathbf { h } ^ { \mathrm { b a s e } }$ , the intervention replaces the subspace component of $\mathbf { h } ^ { \mathrm { b a s e } }$ with that of $\mathbf { h } ^ { \mathrm { s o u r c e } }$

$$
\mathbf { h } _ { \mathrm { p a t c h e d } } = \underbrace { \mathbf { h } } _ { \begin{array} { c } { \mathrm { o r i g i n a l } } \\ { \mathrm { a c t i v a t i o n } } \end{array} } - \underbrace { \mathbf { R } \mathbf { R } ^ { \top } \mathbf { h } } _ { \begin{array} { c } { \mathrm { r e m o v e ~ o r i g i n a l } } \\ { \mathrm { s u b s p . ~ c o o r d s } } \end{array} } + \underbrace { \mathbf { R } \mathbf { R } ^ { \top } \boldsymbol { \mu } _ { \mathrm { t a r g e t } } } _ { \begin{array} { c } { \mathrm { i n j e c t ~ t a r g e t } } \\ { \mathrm { s u b s p . ~ c o o r d s } } \end{array} }
$$

During training, we alternate intervention directions: on even steps we patch literal representations into metaphor prompts (aiming to flip the output to literal), and on odd steps the reverse. The loss for a single example is $\mathcal { L } = \mathrm { { l o g i t } ( o t h e r ) - l o g i t ( t a r g e t ) }$ where target is the correct post-intervention label and other is the original label. R is updated via Adam for 1,000 steps while the model weights are frozen.

IIA is the fraction of samples that are correctly flipped. In our case, if $\mathbf { h } ^ { \mathrm { b a s e } }$ is predicted as “metaphor” for the model, it should flip to “literal” post-patching to contribute positively to IIA. Hence, $\mathrm { I I A } = 1 . 0$ means we have found the variable capable of flipping all held-out examples.

## 4.2 Model steering

We evaluate whether we can steer incorrect predictions towards correct ones using two methods: full-vector mass-mean steering, as used by Marks and Tegmark (2024), and steering using the mean activations for metaphor-sample and literal-sample activations in the DAS subspace. The means are computed from correctly predicted samples, and we steer incorrectly predicted samples in both directions.

Mass-mean steering is implemented as:

$$
\mathbf { h } _ { \mathrm { s t e e r e d } }  \mathbf { h } + \alpha ( \mu _ { \mathrm { m e t } } - \mu _ { \mathrm { l i t } } )\tag{1}
$$

DAS steering is implemented as:.

$$
\mathbf { h } _ { \mathrm { s t e e r e d } }  \mathbf { h } - \alpha \mathbf { R } \mathbf { R } ^ { \top } \mathbf { h } + \alpha \mathbf { R } \mathbf { R } ^ { \top } \pmb { \mu } _ { \mathrm { t a r g e t } }\tag{2}
$$

We set $\alpha = 1$ for the identification and interpretation tasks on NOUN, ADJ, and DOM. On DOM we train DAS matrices firstly across the whole dataset, and subsequently for each individual domain.

## 4.2.1 Generation of multiple tokens

To test whether the DAS subspace encodes a general-purpose concept of metaphor, which extends beyond the prediction task it was trained on, we steer free-form generation. We prompt the model as described in section 3.2. During decoding, we steer every generated token as in Li et al. (2023): we use a hook function to save the activation we use for patching, and inject it at every generated token.

We compare baseline generation, mass mean, and DAS outputs across $\alpha \in \{ 1 , 2 , 5 , 1 0 \}$ and both steering directions (lit → met and met → lit).

Evaluation using LLM-as-judge To quantify the degree to which generated output changed from explaining a literal use of an adjective to a metaphorical use and vice versa, we use Qwen 2.5 72B Instruct as a judge. Only the first sentence of each completion is judged, and a degeneracy filter applied before judging: a completion is marked degenerate if any single word repeats ≥ 3 times. The judge is prompted using the prompt shown in Appendix K.

Metrics We report the following metrics: Flip rate (MM / DAS): fraction of of completions where judge label matches the target direction. Degenerate responses are counted as 0, and we only consider the samples where the judge was able to judge correctly on the baseline generation presteering.

## 4.3 Models Evaluated

We evaluate seven instruction-tuned models across four model families<sup>4</sup> on metaphor identification and interpretation tasks. These are: Gemma 2 9B IT; LLaMA 3.1 8B IT; LLaMA 3.1 70B IT; Mistral 7B IT; Mixtral 8×7B IT; Qwen 2.5 7B IT; Qwen 2.5 72B IT, where IT stands for intstruction tuned.

## 5 Results

## 5.1 Baseline Performance

We evaluate all models on the identification and interpretation tasks across all datasets (see Table 2). We can see that all models are able to perform these tasks to a reasonable standard with the exception of Gemma 27B on DOM. This modes is excluded from further analysis on this dataset.

## 5.2 Identifying Metaphoricity

## 5.2.1 Activation patching

Examples of patching from the interpretation and identification tasks are presented in Figure 1. Additional patching results are shown in Figures 3 and 4 in the Appendix.

Across both parts of speech and both tasks, we observe three groups of activations that most strongly influence the output (Figure 1). Two of these groups are expected. Group (a) corresponds to tokens that differ between the patching source and base prompts (e.g., “skin” and “wit”). Group (c) corresponds to the final token “:”, which is natural because the hidden states of the final token directly determine the next-token prediction.

Group (b) contains the final token “.” in the sentence or paraphrase, as well as the token immediately preceding it. We hypothesize that Group (b) activations encode information about the metaphoricity of the sentence or paraphrase.

Based on the patching results, we extract the hidden states of the selected period token on the layer which has the most influence on the model’s output (e.g. 11th layer for Llama-3.1-8B, 29th layer for Llama-3.1-70B). We visualize LLM representations of our datasets using principal component analysis (PCA). As shown in Figure 2, the first two principal components (PC1 and PC2) linearly separate metaphorical from literal and apt from inapt across NOUN and ADJ and both tasks. More results are available in Figures 5,6 and 7 in Appendix C.

<table><tr><td>Task</td><td></td><td>Dataset LLaMA-3.1-8B</td><td>LLaMA-3.1-70B</td><td>Qwen-2.5-7B</td><td>Qwen-2.5-72B</td><td>Mistral-7B</td><td>Mixtral-8x7B</td><td>Gemma-2-9B</td><td>Gemma-2-27B</td></tr><tr><td rowspan="3">Identification</td><td>ADJ</td><td>0.755</td><td>0.915</td><td>0.860</td><td>0.930</td><td>0.760</td><td>0.870</td><td>0.930</td><td>0.915</td></tr><tr><td>NOUN</td><td>0.840</td><td>0.905</td><td>0.815</td><td>0.888</td><td>0.645</td><td>0.881</td><td>0.873</td><td>0.900</td></tr><tr><td>DOM</td><td>0.841</td><td>0.907</td><td>0.805</td><td>0.911</td><td>0.814</td><td>0.825</td><td>0.881</td><td>0.464</td></tr><tr><td>Interpretation</td><td>NOUN</td><td>0.727</td><td>0.862</td><td>0.696</td><td>0.815</td><td>0.765</td><td>0.796</td><td>0.845</td><td>0.858</td></tr></table>

Table 2: Baseline accuracy on the 4-shot evaluation on identification and interpretation tasks across all models and datasets. DOM results use the M&T 5-shot prompt. Gemma-2-27B excluded from DOM analysis.

![](images/cae82f735655bf12dda0b619dd9afaee270f59c8ca34ce7b1ae039a14da0f38e.jpg)  
(a) Llama 8b identification task

![](images/f47109a05d1884aa3cfd7b3e9b134f5f7f0490ac177ba9f4617c140a0653ecc5.jpg)  
(b) Llama 8b interpretation task  
Figure 1: Activation patching results for Llama 3.1 8B. (a) Identification task. The prompts use the literal sentence “The skin is $\boldsymbol { d r y . ^ { \prime \prime } }$ and the metaphorical sentence “The wit is dry.” from ADJ We patch activations from the literal prompt into the metaphorical prompt. (b) Interpretation task. The prompts use the metaphorical sentence “The puzzle is rocket science.”, the correct paraphrase “The puzzle is easy.”, and the incorrect paraphrase “The puzzle is difficult.” from NOUN. We patch activations from the apt paraphrase into the inapt paraphrase.

## 5.2.2 DAS training

Table 3: Global (all domains pooled) DAS IIA across models. IIA reported as mean ± std with 95% CI.
<table><tr><td>Model</td><td>l</td><td>k</td><td> $\mathrm { I I A } \pm \mathrm { s t d }$ </td><td>95% CI</td></tr><tr><td>Gemma-9B</td><td>19</td><td>16</td><td> $0 . 8 0 7 \pm 0 . 0 0 7$ </td><td>[0.799, 0.815]</td></tr><tr><td>LLaMA-8B</td><td>11</td><td>16</td><td> $0 . 7 9 3 \pm 0 . 0 0 6$ </td><td>[0.783, 0.802]</td></tr><tr><td>Mistral-7B</td><td>10</td><td>32</td><td> $0 . 7 1 1 \pm 0 . 0 1 9$ </td><td>[0.698, 0.722]</td></tr><tr><td>Mixtral-8x7B</td><td>11</td><td>16</td><td> $0 . 6 9 1 \pm 0 . 0 2 6$ </td><td>[0.680, 0.701]</td></tr><tr><td>Qwen-7B</td><td>14</td><td>16</td><td> $0 . 8 3 0 \pm 0 . 0 0 0$ </td><td>[0.821, 0.839]</td></tr><tr><td>LLaMA-70B</td><td>28</td><td>32</td><td> $0 . 6 4 0 \pm 0 . 0 2 5$ </td><td>[0.630, 0.650]</td></tr><tr><td>Qwen-72B</td><td>52</td><td>32</td><td> $0 . 8 4 3 \pm 0 . 0 2 4$ </td><td>[0.836, 0.850]</td></tr></table>

We select the optimal layer for training DAS using plain activation patching at the full stop token across layers. We extract the full hidden vector at this token at the optimal layer, and train two DAS matrices (k=16 and k=32). In Table 3 we report best layer ℓ, subspace dimension k and IIA score.

We are interested in whether the model has a domain-general causal encoding of metaphor, or multiple domain-specific ones. We therefore compare the IIA for a global DAS matrix and multiple domain-specific DAS matrices on a held out evaluation set for each domain (not used for training global DAS, nor the domain-specific variant). Results for Llama 3.1 8B are reported in Table 4. We see that the global DAS matrices have higher IIA than domain-specific matrices. For all results, see Table 9 and Table 10.

Table 4: LLaMA-8B: Domain-specific DAS IIA vs. global DAS transfer on held-out evaluation set
<table><tr><td>Concept</td><td>Domain-specific DAS</td><td></td><td>Global DAS (transfer)</td></tr><tr><td>Clarity</td><td> $\overline { { 0 . 3 1 6 \pm 0 . 0 7 0 } }$ </td><td>[0.262,0.369]</td><td>0.500 [0.404,0.596]</td></tr><tr><td>Depth</td><td> $0 . 3 4 3 \pm 0 . 0 1 0$ </td><td>[0.299, 0.386]</td><td>0.910 [0.868, 0.952]</td></tr><tr><td>Light</td><td> $0 . 3 2 2 \pm 0 . 1 2 2$ </td><td>[0.294, 0.352] 0.727</td><td>[0.674, 0.776]</td></tr><tr><td>Streng.</td><td> $0 . 6 0 6 \pm 0 . 0 6 1$ </td><td>[0.582, 0.628]</td><td>0.967 [0.953, 0.982]</td></tr><tr><td>Sub.</td><td> $\begin{array} { l } { 0 . 3 0 9 \pm 0 . 0 5 9 } \\ { - } \end{array}$ </td><td>[0.248, 0.371]</td><td>0.814 [0.714,0.900]</td></tr><tr><td>Taste</td><td> $0 . 6 4 5 \pm 0 . 0 5 4$ </td><td>[0.588, 0.706]</td><td>0.842 [0.763,0.921]</td></tr><tr><td>Temp.</td><td> $0 . 4 7 0 \pm 0 . 0 5 1$ </td><td>[0.435, 0.506]</td><td>0.568 [0.500, 0.627]</td></tr><tr><td> $\operatorname { T e x } .$ </td><td> $0 . 3 4 3 \pm 0 . 0 2 2$ </td><td>[0.311, 0.376]</td><td>0.713 [0.657, 0.765]</td></tr></table>

## 5.3 Model steering

## 5.3.1 Identification and Interpretation

On the identification and interpretation tasks on NOUN and ADJ, we intervene on model outputs

![](images/02595abd50b864cc14946c1d9deeddcee823bc68afc5c7350b90e82f3131b751.jpg)

![](images/48b5b01599bea23425d4cd0bcb24f1d264de33d69effc898f1d000419be0f2da.jpg)

![](images/a2424679d0c74790103a1b25905548ad29fac3bd4d464b03afee0edfd2850631.jpg)  
Figure 2: PCA visualization of activations at layer 11 for Llama 3.1 8B, for the identification task on the adjective (left) and noun(middle) datasets, and for the interpretation task on the adjective dataset (right). Notice that literal and metaphoric sentences are more or less linearly separable.

<table><tr><td rowspan="2">Task</td><td rowspan="2">train/test</td><td colspan="2">met→lit</td><td rowspan="2">lit→met /inapt→apt</td></tr><tr><td>probe</td><td>/apt→inapt</td></tr><tr><td rowspan="6">Task 1</td><td>Adj/Adj</td><td>LogReg MM-diff</td><td>.038 .227</td><td>.051 .471</td></tr><tr><td></td><td>LogReg</td><td>.025</td><td>.041</td></tr><tr><td rowspan="2">Noun/Noun</td><td>MM-diff</td><td>.195</td><td>.449</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="2">Adj/Noun</td><td>LogReg MM-diff</td><td>.026 .138</td><td>.032 .254</td></tr><tr><td>LogReg</td><td>.028</td><td>.040</td></tr><tr><td rowspan="2"></td><td rowspan="2">Noun/Adj</td><td>MM-diff</td><td>.230</td><td>.666</td></tr><tr><td></td><td>.011</td><td></td></tr><tr><td rowspan="2">Task 2</td><td rowspan="2">Noun/Noun</td><td>LogReg</td><td></td><td>.014</td></tr><tr><td>MM-diff</td><td>.355</td><td>.564</td></tr></table>

Table 5: Normalized indirect effects (NIEs) of massmean difference (MM-diff) and logistic regression (LogReg) probes on the identification and interpretation tasks for LLaMA-3.1-8B. Higher NIE values indicate stronger causal steering effects.

as in Eq. (1). Then, we construct a linear probe using the direction $\pmb \theta = \mu _ { \mathrm { m e t } } - \mu _ { \mathrm { l i t } } ;$

$$
p ( \mathrm { m e t a p h o r } \mid s ) = \sigma \big ( \pmb { \theta } ^ { \top } \mathbf { h } ( s ) \big ) ,
$$

where $\begin{array} { r } { \sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } } \end{array}$ . For the interpretation task, we define directions for apt/inapt pairs using the corresponding means.

First, we use the mass-mean probe to flip the true positive outputs in test set into the opposite label. We then compare MM probes with standard logistic regression (LR) probes trained on the same representations. For both probes, we calculate the normalized indirect effect (NIE) of the intervention. For details of the NIE calculation, see Appendix J. Results for Llama-3.1-8B are reported in Table 5. Results across all models are in Appendix J.

## 5.3.2 Mass-mean and DAS steering

We compare the performance of mass-mean steering and subspace steering using the global DAS matrices trained in Section 5.2.2. We choose initially incorrect predictions on the identification task on the DOM dataset. We use the two steering methods to steer output to the correct prediction. Results are reported in Table 6. We see that we are able to achieve fairly high performance across models. For a full overview, see Table 11 in Appendix F.

<table><tr><td>Model</td><td>MM flip rate</td><td>DAS fip rate</td></tr><tr><td>Gemma 2 9B</td><td>75.0%</td><td>86.0%</td></tr><tr><td>LLaMA 3.1 8B</td><td>54.0%</td><td>71.0%</td></tr><tr><td>LLaMA 3.1 70B</td><td>57.0%</td><td>65.0%</td></tr><tr><td>Mistral 7B</td><td>64.0%</td><td>45.0%</td></tr><tr><td>Mixtral 8×7B</td><td>72.0%</td><td>61.0%</td></tr><tr><td>Qwen 2.5 7B</td><td>62.0%</td><td>61.0%</td></tr><tr><td>Qwen 2.5 72B</td><td>88.0%</td><td>91.0%</td></tr></table>

Table 6: Repair rates on initially-incorrect items. MM flip rate: proportion of incorrect items flipped to the correct answer by full-residual-stream patching. DAS flip rate: proportion flipped by intervening only on the learned subspace. Bold marks the higher value per row.

## 5.3.3 Generation

Whilst there are differences between models, we find that $\alpha = 5$ enables the most effective steering on average. Results for Llama-3.1-8B are presented in Table 7. For a full overview, we refer to Tables 12 and 13 in Appendix G.

To validate the results of the LLM-as-judge, we qualitatively inspect the generated outputs. We display some generations in Table 8. More examples, including successful and unsuccessful generations, are included in Table 14 in the Appendix.

## 6 Discussion and Conclusions

Our results show that there are clear directions and subspaces associated with metaphoricity across a range of models and datasets (Figure 2 and Figures 5, 6 in the Appendix). We are able to train DAS matrices which effectively alter output on a classification task (Table 3), and we see that training a global metaphor subspace is more effective than training domain-specific subspaces (Table 4), indicating that we have learnt a general concept of metaphor. These directions are moreover causally effective: using the mass-mean probe we are able to steer output in classification tasks, and this steering is more effective than steering using the direction identified by logistic regression (Table 5). We do note that the NIE values are somewhat lower than in (Marks and Tegmark, 2024), however, it is also the case that metaphor identification and interpretation is a more difficult task than judging truth and falsity. We also see that there are higher NIEs steering from literal to metaphor than vice-versa. Creating a literal meaning from a metaphorical meaning is perhaps conceptually more difficult: warm coffee can readily be interpreted metaphorically via personification of the coffee, whereas warm welcome is more difficult to interpret literally.

<table><tr><td rowspan="2">Model</td><td>Lit → met</td><td></td><td colspan="2">Met → lit</td></tr><tr><td>MM</td><td>DAS</td><td>MM</td><td>DAS</td></tr><tr><td>Gemma 2 9B</td><td>0.33</td><td>0.12</td><td>0.25</td><td>0.12</td></tr><tr><td>LLaMA 3.1 8B</td><td>0.44</td><td>0.34</td><td>0.03</td><td>0.09</td></tr><tr><td>LLaMA 3.1 70B</td><td>0.34</td><td>0.28</td><td>0.15</td><td>0.16</td></tr><tr><td>Mistral 7B</td><td>0.52</td><td>0.19</td><td>0.28</td><td>0.10</td></tr><tr><td>Mixtral 8×7B</td><td>0.38</td><td>0.16</td><td>0.24</td><td>0.11</td></tr><tr><td>Qwen 2.5 7B</td><td>0.26</td><td>0.47</td><td>0.21</td><td>0.12</td></tr><tr><td>Qwen 2.5 72B</td><td>0.29</td><td>0.15</td><td>0.08</td><td>0.19</td></tr></table>

Table 7: Flip rates at α = 5 for both intervention methods and both directions.

Table 8: Example generations for LLaMA-8B under steering at α = 5. Baseline = unintervened model, MM = mass-mean steering, DAS = DAS intervention. Both examples: true label literal, target metaphorical.
<table><tr><td>Method</td><td>Judge prediction</td><td>Generation</td></tr><tr><td colspan="3">adj-noun pair: icy pavement</td></tr><tr><td>Baseline MM</td><td>literal metaphorical</td><td>extremely cold and slippery. extremely cold and unfriendly. extremely cold and unfriendly.</td></tr><tr><td colspan="3">DAS metaphorical adj-noun-pair: heated washer</td></tr><tr><td>Baseline</td><td>literal</td><td>having a high temperature.</td></tr><tr><td>MM</td><td>metaphorical</td><td>having a high temperature, often due to strong emotions.</td></tr><tr><td>DAS</td><td>metaphorical</td><td>having a strong, intense, or pas- sionate quality.</td></tr></table>

We further steer generation to more metaphorical output. In Table 7, we see that the mass-mean probes are generally more effective than DAS steering, although this differs across model. Example generations are given in Table 8. In future work, this technique could be developed to give a way of controlling the metaphoricity of generated output in a more direct way than prompting.

We acknowledge that IIA scores are moderate and variable, and indeed this does imply that a proportion of the metaphoricity information is captured elsewhere. We note however, that previous work which has used DAS for locating concepts in model activation achieve similar scores (see e.g. (Mueller et al., 2025)). Achieving a perfect IIA score is not necessarily achievable in practice (Geiger et al., 2024), and this is a limitation of using DAS for concept-finding.

On the basis of Table 4, we concluded that the DAS matrix trained on all domains captures a general concept of metaphoricity, as evidenced by its stronger transfer IIA results compared to the results achieved with its domain-specific counterpart. One confound could be the training data size: Compared to the domain-specific matrices trained on n samples, the global DAS matrix is trained on those and all the training samples from all other domains. As a control, we train a matched-n global DAS matrix. This matrix trained on the number of samples in each domain corresponding to the number of samples from the smallest training domain (n = 255). We see in Appendix Table 15 that for Gemma 2 9B, training a matched-n global DAS matrix still achieves a better or equal transfer IIA score in 6/8 transfer domains.

For our generation experiment, we carried out a human validation of the LLM-as-judge. For the Gemma 9B model outputs, one of the authors of this paper has annotated 15% (100 items) as either metaphor, literal, or incorrect (nonsensical or incorrect output). Raw agreement is 0.79, Cohen’s kappa is 0.60, indicating moderate to substantial agreement. This indicates that the judge annotations are mostly reliable. This suggests that the concept we have isolated in the classification task is at least partially sufficient to flip free-form generation. However, generally, we do achieve substantially lower IIA scores for the generation task compared to the classification task. Our DAS concept allows us to steer model classification to a moderately high degree, but this effect is less amplified for generation, and requires that we scale this signal by α = 5. There can be different reasons for this. It is possible that the metaphoricity concept we isolate in the classification task is functionally adapted to this task. The model could have another metaphoricity concept located in a different position adapted for generation. On the other hand, the fact that the concept has some meaning-steering effect suggests that we have captured a common concept of metaphoricity, at least partially.

Further, we employed a controlled, templated setup for measuring metaphoricity in LLMs. We made this choice to effectively isolate the parts of the prompt that have a causal effect on the output. The structured sentence patterns allowed us to localize causal effects to specific token positions. Therefore, we deliberately did not test representations of metaphoricity in a more naturalistic context (e.g. poems), limiting the external validity of our findings.

## 6.1 Conclusions

We have given evidence that metaphoricity is encoded as an interpretable direction in a variety of LLMs, illustrated as linear separability in PCA plots. We intervene on model activations and show that these directions are causally effective in changing model outputs in both classification and generation tasks. Future work in this area could extend to other aspects of writing style: at present, much work is based in prompting, and these techniques may give more control over style modifications.

## 7 Limitations and Risks

• This work is carried out only on English metaphor. Linear separability and interpretable directions could be very different in other languages.

• Mechanistic interpretability work requires very controlled settings and datasets, whereas metaphor can extend across multiple sentences, paragraphs, and discourse. The datasets we use are therefore only very simplistic uses of metaphor. More complex metaphor use is an area for future research.

• We cannot carry out our experiments on the most highly performing closed models, since they do not have open weights, and computational resources would preclude this.

• Risks In its present form, risks resulting from this work are fairly low. However, model steering techniques can potentially be risky. It is possible to steer towards undesirable as well as desirable behaviour.

## Acknowledgements

This work was carried out using the National Supercomputer Snellius made available via SURF (www.surf.nl) and the University of Amsterdam. ML gratefully acknowledges financial support from the Sectorplan science fund of the Dutch government.

## References

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, and 1 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Joanne Boisson, Asahi Ushio, Hsuvas Borkakoty, Kiamehr Rezaee, Dimosthenis Antypas, Zara Siddique, Nina White, and Jose Camacho-Collados. 2024. How are metaphors processed by language models? the case of analogies. In Proceedings of the 28th Conference on Computational Natural Language Learning, pages 365–387.

Tuhin Chakrabarty, Debanjan Ghosh, Adam Poliak, and Smaranda Muresan. 2021. Figurative language in recognizing textual entailment. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3354–3361.

Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. 2023. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352.

Matteo Fuoli, Weihang Huang, Jeannette Littlemore, Sarah Turner, and Ellen Wilding. 2026. Metaphor identification using large language models: A comparison of rag, prompt engineering, and fine-tuning. Applied Corpus Linguistics, page 100204.

Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah D. Goodman. 2024. Finding alignments between interpretable causal variables and distributed neural representations. In Proceedings ofthe Third Conference on Causal Learning and Reasoning, volume 236 of Proceedings ofMachine Learning Research, pages 160–187. PMLR.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,

Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

E. Dario Gutiérrez, Ekaterina Shutova, Tyler Marghetis, and Benjamin Bergen. 2016. Literal and metaphorical senses in compositional distributional semantic models. In Proceedings of the 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), Berlin, Germany. Association for Computational Linguistics.

Nicholas Ichien, Dušan Stamenkovic, and Keith J´ Holyoak. 2024. Large language model displays emergent ability to interpret novel literary metaphors. Metaphor and Symbol, 39(4):296–309.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

George Lakoff and Mark Johnson. 2008. Metaphors we live by. University of Chicago press.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inferencetime intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems.

Ao Liu, An Wang, and Naoaki Okazaki. 2022a. Semisupervised formality style transfer with consistency training. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4689–4701, Dublin, Ireland. Association for Computational Linguistics.

Emmy Liu, Chenxuan Cui, Kenneth Zheng, and Graham Neubig. 2022b. Testing the ability of language models to interpret figurative language. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4437–4452, Seattle, United States. Association for Computational Linguistics.

Pusheng Liu, Lianwei Wu, Linyong Wang, Sensen Guo, and Yang Liu. 2024. Step-by-step: Controlling arbitrary style in text with large language models. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 15285–15295.

Samuel Marks and Max Tegmark. 2024. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. arXiv preprint arXiv:cs.AI. Submitted on 10 Oct 2023 (v1), last revised 19 Aug 2024 (v3).

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 35:17359–17372.

Aaron Mueller, Atticus Geiger, Sarah Wiegreffe, Dana Arad, Iván Arcuschin, Adam Belfki, Yik Siu Chan, Jaden Fiotto-Kaufman, Tal Haklay, Michael Hanna, Jing Huang, Rohan Gupta, Yaniv Nikankin, Hadas Orgad, Nikhil Prakash, Anja Reusch, Aruna Sankaranarayanan, Shun Shao, Alessandro Stolfo, and 4 others. 2025. MIB: A mechanistic interpretability benchmark. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML).

OpenAI. 2025. Chatgpt. https://chat.openai.com. Accessed: 2026-05-26.

Sebastian Reimann and Tatjana Scheffler. 2025. Using large language models to perform MIPVU-inspired automatic metaphor detection. In Proceedings of the 2nd Workshop on Analogical Abstraction in Cognition, Perception, and Language (Analogy-Angle II), pages 10–21, Vienna, Austria. Association for Computational Linguistics.

Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, and 1 others. 2025. Open problems in mechanistic interpretability. arXiv preprint arXiv:2501.16496.

Ekaterina Shutova. 2010. Automatic metaphor interpretation as a paraphrasing task. In Human Language Technologies: The 2010 Annual Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics, pages 1029–1037, Los Angeles, California. Association for Computational Linguistics.

Gerard J Steen, Aletta G Dorst, J Berenike Herrmann, Anna A Kaal, and Tina Krennmayr. 2010. Metaphor in usage.

Kevin Stowe, Prasetya Utama, and Iryna Gurevych. 2022. IMPLI: Investigating NLI models’ performance on figurative language. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5375–5388, Dublin, Ireland. Association for Computational Linguistics.

Yulia Tsvetkov, Leonid Boytsov, Anatole Gershman, Eric Nyberg, and Chris Dyer. 2014. Metaphor detection with cross-lingual model transfer. In Proceedings ofthe 52nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 248–258.

## A NOUN identification prompt

The politician is glass. This sentence is: metaphoric

The politician is transparent. This sentence is: literal

The cake is heaven. This sentence is: metaphoric The cake is delicious. This sentence is: literal sentence. This sentence is:

## B Activation patching plots

![](images/91ced3a4d1547bf0c5af7dc12e04923746a14c66b4ff1a9704e5a95e27bc527a.jpg)  
(a) Gemma 27B Adj

![](images/4dd49624fcfd338f81b4c71be1872206b5d7c56f1813938e84ba70c369bca72d.jpg)  
(b) Llama 70B Adj

![](images/72e58a895b556a65d618b72e729ba03d821ddab2557bdf60d7699ffbc95dd920.jpg)

![](images/d16b21c04afa95aedf2354741e4e679dbe67687617a77c39da2083ea96294190.jpg)  
(c) Mixtral 8x7B Adj  
(d) Qwen 72B Adj

![](images/e7db1f36804278ad969cf0aebaef6da8938d846bfdc8544351a8876022788e78.jpg)  
(e) Gemma 27B Noun

![](images/687dfd430646373423517fba0d5cbe1d2dcef597fbaa8ab32470702c0db5680e.jpg)  
(f) Llama 70B Noun

![](images/f68b88c53ad202ed201ccc6a185a6843d2a3a63c4c63c775afcfabe843c43487.jpg)  
(g) Mixtral 8x7B Noun

![](images/e1322a2a91faaa68ffce5ba15f71fdee8454b5d2c9d52635606469a58214ef58.jpg)  
(h) Qwen 72B Noun

Figure 3: Activation patching results for adjective and noun representations in the identification task. Activations from the metaphorical prompt are patched into the literal prompt. (a)–(d) Adjective dataset. The prompts use the literal sentence “The sunlight is bright.” and the metaphorical sentence “The idea is bright.”. (e)–(h) Noun dataset. The prompts use the literal sentence “The breeze is cold.” and the metaphorical sentence “The breeze is ice.”.  
![](images/c8c3fadf0f7df66353897de22bc36cf04d47d5700445538e9e49b88cd98c2b0c.jpg)  
(a) Gemma 9B Noun

![](images/492f8a8e6f3803879bf84ccbef5b214617e4c5a4fce5daaa265e5b2ae47e5623.jpg)  
(b) Mistral 7B Noun

![](images/8e8ba57ea2575ed6ce2aad8b82e0a5e590a9355a6cba4f9fae4cca726ae3b9af.jpg)  
(c) Mixtral 8x7B Noun

Figure 4: Activation patching results for noun representations in the interpretation task. Activations from the inapt paraphrasing prompt are patched into the apt paraphrasing prompt. The prompts use the metaphorical sentence “The puzzle is rocket science.”, the correct paraphrase “The puzzle is difficult.”, and the incorrect paraphrase “The puzzle is easy.”.

## C PCA plots

![](images/7d6fe05a3c323b8b859111e7dd5b688413f7ac60654e892e27867beab9c5148c.jpg)  
(a) Gemma-2 9B

![](images/de402f7b07c01c352a704398806a5efac97f7121a9185e47bdcf662049896d58.jpg)  
(b) Gemma-2 27B

![](images/d46a631c52a0753f2aa85b4cf6a840667f9adb1ae70afee07df792ab4af9c519.jpg)  
(c) Llama-3.1 8B

![](images/18ebd41dbe3d24358debba526a6edda5d09958fde9bae248f6157b5c461d24c1.jpg)  
(d) Llama-3.1 70B

![](images/9ed2354e5853cea3989c7805aa5a486e4ac0ea59db9663b56fa267da43e3ea61.jpg)  
(e) Mistral 7B

![](images/f21714901baa486a52ff774aed402ba7494bba5fc4cbc9154ffc749a31312674.jpg)  
(f) Mixtral 8x7B

![](images/73a4175aa2d5bee67393ce41c7f2b41dde824edfd7604cc85371ded5679785f8.jpg)  
(g) Qwen2.5 7B

![](images/3d7bb79fdc4f659b8e15bcba8aac3911c3f18db0964680fe9bfe2570e3bf0e99.jpg)  
(h) Qwen2.5 72B

Figure 5: Identification task: PCA visualizations for adjective representations.  
![](images/cfc55533d1881270e05225c6eb3ae39e0f29a12d96ab5d7a971243e00192a8ef.jpg)  
(a) Gemma-2 9B

![](images/af0414c49b8ca6b54bfe5cc4d17695b93ee8043c7fe415c2f4150a6b91ce6e6b.jpg)  
(b) Gemma-2 27B

![](images/9f94edb59c0d235af83268b239993bb3979daa7a3d9e997618617bbf5ed1c1f1.jpg)  
(c) Llama-3.1 8B

![](images/180941d79980c4fba58d467850eb676520534bbc1a5356f5cef1bd6a9f2cb851.jpg)  
(d) Llama-3.1 70B

![](images/242f57a3e857c1260c84ab1474865a9cb8235fb7254efdca7a4346db24fed36b.jpg)  
(e) Mistral 7B

![](images/11c8d4cd5cad8fc2fc534f9ac017cbb74633b3c3a048fbdea6b17f04accd22b5.jpg)  
(f) Mixtral 8x7B

![](images/653b4adc1fb1dd85be902850c723d0e44c430ccaf005ce2727f499b06af9bb4e.jpg)  
(g) Qwen2.5 7B

![](images/36473dad7e1c4379b06ddda5df92f1f2d6cd4d2c4ff46bfabc186874e2061e43.jpg)  
(h) Qwen2.5 72B  
Figure 6: Identification task: PCA visualizations for noun representations.

![](images/8af1070e64728e02f6367690f97a10395e719f5e9d27a00a196145a390514883.jpg)  
(a) Gemma-2 9B

![](images/df84bc53ba4377e61fc50bd263f360bea183b390c6f537447208813cb8a4f062.jpg)  
(b) Gemma-2 27B

![](images/57c552cc068d1be0d4acd3dc5c76b13aa936b8371c6e917b08439b45390ca83c.jpg)  
(c) Llama-3.1 8B

![](images/6d26eac856897296dcd005b12f2f4cd81d7cff334100281384b56f1671eb7b3b.jpg)  
(d) Llama-3.1 8B

![](images/83d97f10a10e121cd759389a91b91ffbff716148028ab319d5b1691f6e3bbeec.jpg)  
(e) Mistral 7B

![](images/1e46e5a36652f33134085e7659b1f14010563873cb67ba5ad31769d2032e0cbc.jpg)  
(f) Mixtral 8x7B

![](images/240ce3428ee131abe488ad1128353eb98d48b9161f579da9497b44cb60c7abaa.jpg)  
(g) Qwen2.5 7B

![](images/04ba79873184a74119cf6a9a3947a033d5e6d584cd18bae335e4d380afca377d.jpg)  
(h) Qwen2.5 72B  
Figure 7: Interpretation task: PCA visualizations for noun representations.

## D Domain-specific DAS training at full-stop token position

Table 9: Domain-specific DAS IIA (mean ± std) across models. Best per domain in bold.
<table><tr><td>Domain</td><td> $\mathrm { L L a M A – 8 B }$ </td><td> $\mathrm { L L a M A } – 7 0 \mathrm { B }$ </td><td> $\mathrm { G e m m a } { \cdot } 9 \mathrm { B }$ </td><td> $\mathrm { M i s t r a l - 7 B }$ </td><td> $\mathbf { M i x t r a l - 8 x 7 B }$ </td><td> $\mathrm { Q w e n - 7 B }$ </td><td> $\mathrm { Q w e n - } 7 2 \mathrm { B }$ </td></tr><tr><td>Clarity</td><td> $0 . 3 1 6 \pm 0 . 0 7 0$ </td><td> $0 . 3 5 1 \pm 0 . 0 2 2$ </td><td> $0 . 4 4 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 4 5 \pm 0 . 0 1 3$ </td><td> $0 . 3 2 5 \pm 0 . 0 2 6$ </td><td> $\mathbf { 0 . 5 2 1 \pm 0 . 0 0 6 }$ </td><td> $0 . 3 1 3 \pm 0 . 0 4 0$ </td></tr><tr><td>Depth</td><td> $0 . 3 4 3 \pm 0 . 0 1 0$ </td><td> $0 . 2 1 2 \pm 0 . 0 9 5$ </td><td> $0 . 2 2 7 \pm 0 . 0 1 9$ </td><td> $\mathbf { 0 . 4 4 7 \pm 0 . 0 0 8 }$ </td><td> $0 . 4 0 6 \pm 0 . 0 0 5$ </td><td> $0 . 2 8 7 \pm 0 . 0 6 1$ </td><td> $0 . 0 7 9 \pm 0 . 0 1 6$ </td></tr><tr><td>Light</td><td> $0 . 3 2 2 \pm 0 . 1 2 2$ </td><td> $0 . 0 5 9 \pm 0 . 0 0 6$ </td><td> $0 . 5 2 4 \pm 0 . 0 1 5$ </td><td> $0 . 0 8 9 \pm 0 . 0 0 8$ </td><td> $0 . 1 5 8 \pm 0 . 0 6 7$ </td><td> $0 . 4 6 1 \pm 0 . 0 1 7$ </td><td> $\mathbf { 0 . 6 0 8 \pm 0 . 0 2 1 }$ </td></tr><tr><td>Strength</td><td> $0 . 6 0 6 \pm 0 . 0 6 1$ </td><td> $0 . 5 0 8 \pm 0 . 0 4 3$ </td><td> $0 . 7 7 6 \pm 0 . 0 1 1$ </td><td> $0 . 4 1 7 \pm 0 . 0 1 3$ </td><td> $0 . 4 6 7 \pm 0 . 0 3 3$ </td><td> $\mathbf { 0 . 8 5 4 \pm 0 . 0 0 9 }$ </td><td> $0 . 5 9 4 \pm 0 . 0 8 7$ </td></tr><tr><td>Substance</td><td> $0 . 3 0 9 \pm 0 . 0 5 9$ </td><td> $0 . 1 3 0 \pm 0 . 0 3 7$ </td><td>0.500 ± 0.019</td><td> $0 . 4 2 4 \pm 0 . 0 1 3$ </td><td> $0 . 3 1 9 \pm 0 . 0 3 6$ </td><td> $\mathbf { 0 . 5 1 4 \pm 0 . 0 0 0 }$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 7$ </td></tr><tr><td>Taste</td><td> $0 . 6 4 5 \pm 0 . 0 5 4$ </td><td> $0 . 2 4 8 \pm 0 . 0 6 6$ </td><td> $0 . 5 8 7 \pm 0 . 0 4 3$ </td><td> $0 . 6 6 2 \pm 0 . 0 3 7$ </td><td> $0 . 4 9 4 \pm 0 . 0 0 8$ </td><td> $0 . 7 9 6 \pm 0 . 0 5 3$ </td><td> $\mathbf { 0 . 8 2 6 \pm 0 . 0 0 9 }$ </td></tr><tr><td>Temperature</td><td> $0 . 4 7 0 \pm 0 . 0 5 1$ </td><td> $0 . 0 7 7 \pm 0 . 0 2 7$ </td><td> $0 . 5 8 3 \pm 0 . 0 2 1$ </td><td> $0 . 5 4 2 \pm 0 . 0 1 7$ </td><td> $0 . 4 0 5 \pm 0 . 0 1 4$ </td><td> $0 . 6 7 2 \pm 0 . 0 1 9$ </td><td> $\mathbf { 0 . 8 3 8 \pm 0 . 0 5 4 }$ </td></tr><tr><td>Texture</td><td> $0 . 3 4 3 \pm 0 . 0 2 2$ </td><td> $0 . 2 1 9 \pm 0 . 0 8 4$ </td><td> $0 . 5 0 6 \pm 0 . 0 5 0$ </td><td> $\mathbf { 0 . 8 1 0 \pm 0 . 0 3 5 }$ </td><td> $0 . 6 1 3 \pm 0 . 0 6 2$ </td><td> $0 . 6 9 8 \pm 0 . 0 1 4$ </td><td> $0 . 5 3 3 \pm 0 . 0 3 2$ </td></tr></table>

## E Transfer: validation IIA using global DAS instead of domain-specific DAS

<table><tr><td>Domain</td><td>LLaMA-8B</td><td>LLaMA-70B</td><td>Gemma-9B</td><td></td><td>Mistral-7B</td><td>Mixtral-8x7B</td><td>Qwen-7B</td><td></td><td>Qwen-72B</td></tr><tr><td>Clarity</td><td>0.500 [0.404, 0.596]</td><td>0.500 [0.357, 0.625]</td><td>0.581 [0.484, 0.661]</td><td></td><td>0.409 [0.352, 0.466]</td><td>0.738 [0.680, 0.801]</td><td></td><td>0.618 [0.542, 0.701]</td><td>0.739 [0.681, 0.792]</td></tr><tr><td>Depth</td><td>0.910 [0.868, 0.952]</td><td>0.727 [0.688, 0.767]</td><td>0.924 [0.896, 0.952]</td><td>0.849</td><td>[0.808, 0.887]</td><td>0.826 [0.781,0.868]</td><td>0.852</td><td>[0.812, 0.893]</td><td>0.721 [0.677,0.767]</td></tr><tr><td>Light</td><td>0.727 [0.674, 0.776]</td><td>0.240 [0.192, 0.286]</td><td>0.580 [0.523, 0.639]</td><td>0.455</td><td>[0.400, 0.510]</td><td>0.821 [0.776, 0.864]</td><td>0.627</td><td>[0.571, 0.679]</td><td>0.763 [0.717,0.809]</td></tr><tr><td>Strength</td><td>0.967 [0.953, 0.982]</td><td>0.956 [0.938, 0.972]</td><td>0.919 [0.895, 0.942]</td><td>0.595</td><td>[0.545, 0.642]</td><td>0.777 [0.734,0.818]</td><td>0.933</td><td>[0.908, 0.958] 0.959</td><td>[0.940, 0.975]</td></tr><tr><td>Substance</td><td>0.814 [0.714, 0.900]</td><td>0.469 [0.344, 0.594]</td><td>0.643 [0.500, 0.786]</td><td>0.600</td><td>[0.486, 0.700</td><td>0.586 [0.471,0.700]</td><td>0.757</td><td>[0.657, 0.857 0.771</td><td>[0.671, 0.857]</td></tr><tr><td>Taste</td><td>0.842 [0.763, 0.921]</td><td>0.656 [0.544, 0.744]</td><td>0.830 [0.750, 0.909</td><td>0.714</td><td>[0.600, 0.815]</td><td>0.517 [0.400, 0.633]</td><td>0.871 [0.774, 0.952]</td><td>0.977</td><td>[0.942, 1.000</td></tr><tr><td>Temperature</td><td>0.568 [0.500, 0.627]</td><td>0.235 [0.183, 0.287]</td><td>0.650 [0.585, 0.705]</td><td>0.673</td><td>[0.601, 0.744]</td><td>0.601 [0.538, 0.664]</td><td>0.735 [0.676, 0.790]</td><td>0.900</td><td>[0.863,0.933</td></tr><tr><td>Texture</td><td>0.713 [0.657, 0.765]</td><td>0.488 [0.417, 0.554]</td><td>0.767 [0.714, 0.813]</td><td>0.907</td><td>[0.872, 0.943]</td><td>0.809 [0.756, 0.858]</td><td>0.746</td><td>[0.690, 0.798]</td><td>0.924 [0.890, 0.955]</td></tr></table>

Table 10: Transfer IIA: global DAS evaluated on held-out domain (mean [95% CI]). Best per domain in bold.

## F Steering incorrect predictions

Table 11: Flip rates on initially-wrong items: mass-mean steering vs. DAS intervention (mean [95% CI]). Best per row in bold.
<table><tr><td>Model</td><td>Wrong/Total</td><td colspan="2">Mass-Mean flip</td><td colspan="2">DAS flip</td></tr><tr><td>Gemma-9B</td><td>1140/8591</td><td></td><td>0.750 [0.660, 0.830]</td><td></td><td>0.860 [0.790, 0.920]</td></tr><tr><td>LLaMA-8B</td><td>1734/8591</td><td></td><td>0.540 [0.450, 0.640]</td><td></td><td>0.710 [0.620, 0.790]</td></tr><tr><td>Qwen-7B</td><td>1808/8591</td><td></td><td>0.620 [0.530, 0.710]</td><td></td><td>0.610 [0.520, 0.700]</td></tr><tr><td>Mistral-7B</td><td>2079/8591</td><td></td><td>0.640 [0.540, 0.730]</td><td></td><td>0.450 [0.350, 0.550]</td></tr><tr><td>Mixtral-8x7B</td><td>1798/8591</td><td></td><td>0.720 [0.630, 0.810]</td><td></td><td>0.610 [0.510, 0.700]</td></tr><tr><td>LLaMA-70B</td><td>1222/8591</td><td></td><td>0.570 [0.470, 0.670]</td><td></td><td>0.650 [0.560, 0.740]</td></tr><tr><td>Qwen-72B</td><td>1043/8591</td><td></td><td>0.880 [0.810, 0.940]</td><td></td><td>0.910 [0.850, 0.960]</td></tr></table>

## G LLM as a judge

Table 12: Literal → Metaphor generation flip rate by steering scale α and method (MM = mass-mean, DAS). Mean [95% CI]. Degenerate outputs counted as failed flips, few-shot prompts excluded. Only pairs where the judge gets the baseline explanation correct are considered (e.g. the meaning of “warm” in “the coffee is warm” is a model explanation such that the judge classifies it as “literal”).
<table><tr><td>α</td><td>Method</td><td>LLaMA-8B</td><td>LLaMA-70B</td><td>Gemma-9B</td><td>Mistral-7B</td><td>Mixtral-8x7B</td><td>Qwen-7B</td><td>Qwen-72B</td></tr><tr><td rowspan="3">1</td><td rowspan="3">MM DAS</td><td>0.05 [0.00, 0.10]</td><td>0.03 [0.00, 0.08]</td><td>0.02 [0.00, 0.05]</td><td>0.03 [0.00, 0.08]</td><td>0.05 [0.00, 0.11]</td><td>0.03 [0.00, 0.09]</td><td>0.08 [0.02, 0.15]</td></tr><tr><td>0.02 [0.00, 0.05]</td><td>0.03 [0.00, 0.08]</td><td>0.00 [0.00, 0.00]</td><td>0.05 [0.00, 0.11]</td><td>0.02 [0.00, 0.05]</td><td>0.03 [0.00, 0.09]</td><td>0.02 [0.00, 0.05]</td></tr><tr><td>0.12 [0.05, 0.20]</td><td>0.05 [0.00, 0.11]</td><td>0.11 [0.05, 0.18]</td><td>0.11 [0.03, 0.19]</td><td>0.11 [0.03, 0.19]</td><td>0.12 [0.05, 0.22]</td><td>0.09 [0.03, 0.17]</td></tr><tr><td rowspan="2">2</td><td>MM DAS</td><td>0.03 [0.00, 0.08]</td><td>0.08 [0.02, 0.16]</td><td>0.03 [0.00, 0.08]</td><td>0.20 [0.11, 0.31]</td><td>0.03 [0.00, 0.08]</td><td>0.09 [0.02, 0.16]</td><td>0.05 [0.00, 0.11]</td></tr><tr><td>MM</td><td>0.44 [0.32, 0.56]</td><td>0.34 [0.23, 0.45]</td><td>0.33 [0.23, 0.44]</td><td></td><td>0.38 [0.27, 0.50]</td><td></td><td></td></tr><tr><td rowspan="2">5</td><td>DAS</td><td>0.34 [0.22, 0.46]</td><td>0.28 [0.17, 0.41]</td><td>0.12 [0.06, 0.21]</td><td>0.52 [0.39, 0.64] 0.19 [0.09, 0.28]</td><td>0.16 [0.08, 0.25]</td><td>0.26 [0.16, 0.38] 0.47 [0.34, 0.59]</td><td>0.29 [0.20, 0.40] 0.15 [0.08, 0.25]</td></tr><tr><td>MM</td><td>0.20 [0.12, 0.31]</td><td>0.47 [0.36, 0.59]</td><td>0.55 [0.42, 0.67]</td><td></td><td></td><td></td><td></td></tr><tr><td>10</td><td>DAS</td><td>0.25 [0.15, 0.37]</td><td>0.00 [0.00, 0.00]</td><td>0.03 [0.00, 0.08]</td><td>0.34 [0.23, 0.45] 0.02 [0.00, 0.05]</td><td>0.47 [0.34, 0.59] 0.06 [0.02, 0.12]</td><td>0.57 [0.45, 0.69] 0.40 [0.28, 0.52]</td><td>0.11 [0.05, 0.18] 0.43 [0.31, 0.55]</td></tr></table>

Table 13: Metaphor → Literal generation flip rate by steering scale α and method (MM = mass-mean, DAS). Mean [95% CI]. Degenerate outputs counted as failed flips; few-shot prompts excluded, only baseline-correct items.
<table><tr><td>α</td><td>Method</td><td>LLaMA-8B</td><td>LLaMA-70B</td><td>Gemma-9B</td><td>Mistral-7B</td><td>Mixtral-8x7B</td><td>Qwen-7B</td><td>Qwen-72B</td></tr><tr><td rowspan="3">1</td><td rowspan="3">MM DAS</td><td>0.03 [0.00, 0.07]</td><td>0.01 [0.00, 0.04]</td><td>0.05 [0.01, 0.11]</td><td>0.06 [0.01, 0.11]</td><td>0.01 [0.00, 0.04]</td><td>0.04 [0.00, 0.09]</td><td>0.00 [0.00, 0.00]</td></tr><tr><td>0.00 [0.00, 0.00]</td><td>0.01 [0.00, 0.04]</td><td>0.05 [0.01, 0.11]</td><td>0.00 [0.00, 0.00]</td><td>0.00 [0.00, 0.00]</td><td>0.04 [0.00, 0.09]</td><td>0.00 [0.00, 0.00]</td></tr><tr><td>0.04 [0.00, 0.09]</td><td>0.04 [0.00, 0.09]</td><td>0.11 [0.04, 0.18]</td><td>0.04 [0.00, 0.10]</td><td></td><td>0.08 [0.03, 0.14]</td><td>0.00 [0.00, 0.00]</td></tr><tr><td rowspan="2">2</td><td>MM DAS</td><td>0.01 [0.00, 0.04]</td><td>0.05 [0.01, 0.11]</td><td>0.08 [0.03, 0.14]</td><td>0.00 [0.00, 0.00]</td><td>0.07 [0.01, 0.14] 0.01 [0.00, 0.04]</td><td>0.07 [0.01, 0.13]</td><td>0.03 [0.00, 0.07]</td></tr><tr><td>MM</td><td>0.03 [0.00, 0.07]</td><td>0.15 [0.07, 0.23]</td><td>0.25 [0.16, 0.36]</td><td>0.28 [0.18, 0.39]</td><td></td><td></td><td></td></tr><tr><td rowspan="2">5</td><td>DAS</td><td>0.09 [0.04, 0.16]</td><td>0.16 [0.08, 0.25]</td><td>0.12 [0.05, 0.20]</td><td>0.10 [0.04, 0.17]</td><td>0.24 [0.14, 0.34] 0.11 [0.04, 0.19]</td><td>0.21 [0.12, 0.30] 0.12 [0.05, 0.18]</td><td>0.08 [0.03, 0.15] 0.19 [0.11, 0.27]</td></tr><tr><td></td><td>0.03 [0.00, 0.07]</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10</td><td>MM DAS</td><td>0.00 [0.00, 0.00]</td><td>0.00 [0.00, 0.00] 0.05 [0.01, 0.11]</td><td>0.13 [0.07, 0.21] 0.03 [0.00, 0.07]</td><td>0.35 [0.24, 0.46] 0.06 [0.01, 0.11]</td><td>0.33 [0.21, 0.44] 0.24 [0.14, 0.34]</td><td>0.01 [0.00, 0.04] 0.05 [0.01, 0.11]</td><td>0.03 [0.00, 0.07] 0.25 [0.16, 0.35]</td></tr></table>

## H Example generations

<table><tr><td>Output</td><td>Steered?</td></tr><tr><td colspan="2">Metaphorical → Literal</td></tr><tr><td>clear thinker (CLARITY)</td><td></td></tr><tr><td>Baseline easy to understand and free from confusion.</td><td>x</td></tr><tr><td>MM easy to understand and free from confusion.</td><td></td></tr><tr><td>DAS easy to understand and not ambiguous.</td><td>x</td></tr><tr><td colspan="2">smooth player (TEXTURE)</td></tr><tr><td>Baseline having a calm and confident manner.</td><td></td></tr><tr><td>MM having a calm and peaceful appearance.</td><td>x</td></tr><tr><td>DAS having a flat, even surface, and not necessarily emotionally complex.</td><td>√</td></tr><tr><td colspan="2">Literal → Metaphorical</td></tr><tr><td colspan="2">rough landscape (TEXTURE)</td></tr><tr><td>Baseline having a harsh, uneven surface.</td><td></td></tr><tr><td>MM</td><td>having an uneven surface with many obstacles.</td></tr><tr><td>DAS</td><td>having an uneven surface with many obstacles.</td></tr><tr><td colspan="2">clear broth (CLARITY)</td></tr><tr><td>Baseline</td><td>transparent and free from impurities.</td></tr><tr><td>MM</td><td>transparent and easy to understand. transparent and free from impurities.</td></tr><tr><td>DAS</td><td></td></tr></table>

Table 14: Qualitative examples of steering outcomes for mass-mean (MM) and DAS interventions. ✓ indicates the judge’s label flipped to the steered direction. ✗ indicates no flip.

## I Matched-n control

<table><tr><td>Concept</td><td>Full n (global DAS)</td><td>Matched n  $( M _ { n } ,$  global DAS)</td><td>Domain-specific DAS (Dom)</td><td> $\overline { { { M _ { n } } } }$ </td><td>Dom [95% CI]</td></tr><tr><td>clarity</td><td>0.581</td><td>0.513</td><td>0.441</td><td>+0.073 [</td><td> $\overline { { - 0 . 0 0 3 , + 0 . 1 4 5 ] } }$ </td></tr><tr><td>depth</td><td>0.924</td><td>0.855</td><td>0.227</td><td>+0.628 </td><td>[+0.598, +0.662]</td></tr><tr><td>light</td><td>0.580</td><td>0.412</td><td>0.524</td><td></td><td>-0.113 [−0.158, -0.068]</td></tr><tr><td>strength</td><td>0.919</td><td>0.850</td><td>0.776</td><td></td><td>+0.074 [+0.046, +0.100]</td></tr><tr><td>substance</td><td>0.643</td><td>0.524</td><td>0.500</td><td>+0.024</td><td>[−0.095, +0.143]</td></tr><tr><td>taste</td><td>0.830</td><td>0.750</td><td>0.587</td><td>+0.163</td><td>[+0.083, +0.239]</td></tr><tr><td>temperature</td><td>0.650</td><td>0.447</td><td>0.583</td><td></td><td>-0.135 [-0.187, -0.084]</td></tr><tr><td>texture</td><td>0.767</td><td>0.587</td><td>0.506</td><td>+0.080</td><td> $[ + 0 . 0 3 0 , + 0 . 1 3 2 ]$ </td></tr></table>

Table 15: We retrain the global DAS matrix on a number of samples for each domain, matching the smallest domain-specific training domain (n = 255). For 6/8 cells, we achieve a comparable transfer IIA score to the global DAS matrix where no such filter is applied. Results reported for Gemma 2 9B.

## J Steering meaning

## J.1 NIE calculation

For identification task, we first compute $P D ^ { + }$ and $P D ^ { - } ~ \mathrm { { a s } }$ the average probability differences $P ( { \mathrm { m e t a p h o r i c } } ) - P ( { \mathrm { l i t e r a l } } )$ over examples from the metaphoric(+) and literal(−) classes, respectively. After intervention, we obtain $P D _ { * } ^ { + }$ and $P D$ ∗ in the same way. The NIE is then computed as:

$$
\mathrm { N I E } _ { \mathrm { l i t \ - w e t } } = \frac { P D _ { \ast } ^ { - } - P D ^ { - } } { P D ^ { + } - P D ^ { - } }
$$

$$
\mathrm { N I E } _ { \mathrm { m e t \to l i t } } = \frac { P D _ { * } ^ { + } - P D ^ { + } } { P D ^ { - } - P D ^ { + } } .
$$

Therefore, an NIE of 0 indicates no causal effect, while an NIE of 1 corresponds to a complete shift in model predictions.

## J.2 Full NIE results

<table><tr><td rowspan="2">train/test</td><td rowspan="2">probe</td><td colspan="2">LLaMA-8B</td><td colspan="2">LLaMA-70B</td><td colspan="2">Gemma-9B</td><td colspan="2">Gemma-27B</td></tr><tr><td></td><td>inapt→apt apt→inapt</td><td>inapt→apt apt→inapt</td><td></td><td>inapt→apt apt→inapt</td><td></td><td></td><td>inapt→apt apt→inapt</td></tr><tr><td rowspan="2">Adj/Adj</td><td>LogReg</td><td>.038</td><td>.051</td><td>.033</td><td>.028</td><td>.176</td><td>.211</td><td>.293</td><td>.774</td></tr><tr><td>MM-diff</td><td>.227</td><td>.471</td><td>.183</td><td>.168</td><td>.149</td><td>.097</td><td>.262</td><td>.387</td></tr><tr><td rowspan="2">Noun/Noun</td><td>LogReg</td><td>.025</td><td>.040</td><td>.029</td><td>.029</td><td>.220</td><td>.060</td><td>.227</td><td>.282</td></tr><tr><td>MM-diff</td><td>.195</td><td>.355</td><td>.216</td><td>.126</td><td>.212</td><td>.110</td><td>.045</td><td>.304</td></tr><tr><td rowspan="2">Adj/Noun</td><td>LogReg</td><td>.026</td><td>.032</td><td>.029</td><td>.031</td><td>.211</td><td>.297</td><td>.076</td><td>.763</td></tr><tr><td>MM-diff</td><td>.138</td><td>.254</td><td>.189</td><td>.124</td><td>.177</td><td>.223</td><td>.050</td><td>.194</td></tr><tr><td rowspan="2">Noun/Adj</td><td>LogReg</td><td>.028</td><td>.040</td><td>.019</td><td>.015</td><td>.076</td><td>.072</td><td>.449</td><td>.429</td></tr><tr><td>MM-diff</td><td>.230</td><td>.666</td><td>.121</td><td>.171</td><td>.110</td><td>.131</td><td>.230</td><td>.497</td></tr><tr><td rowspan="2">train/test</td><td>probe</td><td colspan="2">Mistral-7B</td><td colspan="2">Mixtral-8x7B</td><td colspan="2">Qwen-7B</td><td colspan="2">Qwen-72B</td></tr><tr><td>II</td><td>inapt→apt apt→inapt</td><td></td><td>inapt→apt apt→inapt</td><td></td><td>inapt→apt apt→inapt</td><td></td><td></td><td>inapt→apt apt→inapt</td></tr><tr><td rowspan="2">Adj/Adj</td><td>LogReg</td><td>.014</td><td>.095</td><td>.075</td><td>.178</td><td>.035</td><td>.054</td><td>.100</td><td></td></tr><tr><td>MM-diff</td><td>.075</td><td>.449</td><td>.093</td><td>.292</td><td>.676</td><td>.578</td><td>.251</td><td>.084 .503</td></tr><tr><td rowspan="2">Noun/Noun</td><td>LogReg</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MM-diff</td><td>.007 .023</td><td>.033 .025</td><td>.013 .031</td><td>.042 .051</td><td>.010 .467</td><td>.031 .081</td><td>.097 .243</td><td>.072 .065</td></tr><tr><td rowspan="2">Adj/Noun</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LogReg MM-diff</td><td>.006 .024</td><td>.030 .153</td><td>.012 .015</td><td>.123 .159</td><td>.050 .397</td><td>.056 .561</td><td>.097 .167</td><td>.040 .397</td></tr><tr><td rowspan="2">Noun/Adj</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LogReg MM-diff</td><td>.006 .054</td><td>.056 .520</td><td>.042 .093</td><td>.123 .367</td><td>.038 .553</td><td>.031 .353</td><td>.072 .285</td><td>.060 .318</td></tr></table>

Table 16: Normalized indirect effects (NIEs) of mass-mean difference (MM-diff) and logistic regression (LogReg) probes on the identification task for eight models. Higher NIE values indicate stronger causal steering effects. Higher scores are highlighted in bold.

<table><tr><td rowspan="3">train/test probe</td><td rowspan="3"></td><td colspan="2">LLaMA-8B</td><td colspan="2">LLaMA-70B</td><td colspan="2">Gemma-9B</td><td colspan="2">Gemma-27B</td></tr><tr><td></td><td></td><td></td><td></td><td>inapt→apt apt→inapt | inapt→apt apt→inapt | inapt→apt apt→inapt | inapt→apt apt→inapt</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Noun</td><td>LogReg</td><td>.011</td><td>.014</td><td>.001</td><td>.005</td><td>.009</td><td>.031</td><td>.038</td><td>.144</td></tr><tr><td>MM-diff</td><td>.355</td><td>.564</td><td>.126</td><td>.305</td><td>.155</td><td>.353</td><td>.304</td><td>.656</td></tr><tr><td rowspan="2">train/test probe</td><td></td><td colspan="2">Mistral-7B</td><td colspan="2">Mixtral-8x7B</td><td colspan="2">Qwen-7B</td><td colspan="2">Qwen-72B</td></tr><tr><td></td><td>inapt→apt</td><td></td><td>apt→inapt | inapt→apt</td><td></td><td></td><td></td><td></td><td>apt→inapt | inapt→apt apt→inapt | inapt→apt apt→inapt</td></tr><tr><td rowspan="2">Noun</td><td>LogReg</td><td>.003</td><td>.013</td><td>.003</td><td>.017</td><td>.002</td><td>.008</td><td>.007</td><td>.033</td></tr><tr><td>MM-diff</td><td>.025</td><td>.165</td><td>.051</td><td>.238</td><td>.081</td><td>.261</td><td>.065</td><td>.344</td></tr></table>

Table 17: Normalized indirect effects (NIEs) of mass-mean difference (MM-diff) and logistic regression (LogReg) probes on the interpretation task for eight models. Probes are trained and tested on our Noun dataset. Higher NIE values indicate stronger causal steering effects. Higher scores are highlighted in bold.

J.3 Flip rate results
<table><tr><td>Task</td><td>Model</td><td>Data</td><td>Layer</td><td>Original label</td><td>Before</td><td>After</td><td>Flip rate</td></tr><tr><td colspan="16"></td></tr><tr><td rowspan="8">T1</td><td>LLaMA-3.1-8B LLaMA-3.1-8B</td><td>Noun Noun</td><td>11 11</td><td>metaphor literal</td><td>27 28</td><td>13 7</td><td></td><td>51.85% 75.00%</td></tr><tr><td>LLaMA-3.1-8B</td><td>Adj</td><td>11</td><td>metaphor</td><td></td><td>21</td><td>14</td><td>33.33%</td></tr><tr><td>LLaMA-3.1-8B</td><td>Adj</td><td>11</td><td>literal</td><td></td><td>7</td><td>2</td><td>71.43%</td></tr><tr><td>Gemma-2-27B</td><td>Noun</td><td>20</td><td>literal</td><td></td><td>23</td><td>18</td><td>21.74%</td></tr><tr><td>Gemma-2-27B</td><td>Adj</td><td>20</td><td>literal</td><td>17</td><td></td><td>12</td><td>29.41%</td></tr><tr><td>Qwen2.5-72B</td><td>Noun</td><td>54</td><td>literal</td><td>26</td><td></td><td>18</td><td>30.77%</td></tr><tr><td>Qwen2.5-72B</td><td>Adj</td><td>54</td><td>literal</td><td>18</td><td></td><td>8</td><td>55.56%</td></tr><tr><td>Mixtral-8x7B</td><td>Noun</td><td>14</td><td>literal</td><td>20</td><td></td><td>14</td><td>30.00%</td></tr><tr><td></td><td>Mixtral-8x7B</td><td>Adj</td><td>14</td><td>literal</td><td>13</td><td>10</td><td></td><td>23.08%</td></tr><tr><td></td><td>LLaMA-3.1-70B</td><td>Noun</td><td>29</td><td>literal</td><td>25</td><td></td><td>22</td><td>8.00%</td></tr><tr><td></td><td>LLaMA-3.1-70B</td><td>Adj</td><td>29</td><td>literal</td><td>18</td><td></td><td>16</td><td>11.11%</td></tr><tr><td></td><td>Gemma-2-27B</td><td>Noun</td><td>20</td><td>inapt</td><td>26</td><td></td><td>21</td><td>19.23%</td></tr><tr><td>T2</td><td>Gemma-2-9B</td><td>Noun</td><td>20</td><td>inapt</td><td>26</td><td></td><td>23</td><td>11.54%</td></tr><tr><td></td><td>Qwen2.5-72B</td><td>Noun</td><td>54</td><td>inapt</td><td>25</td><td></td><td>24</td><td>4.00%</td></tr><tr><td></td><td>Mixtral-8x7B</td><td>Noun</td><td>14</td><td>inapt</td><td>27</td><td></td><td>25</td><td>7.41%</td></tr></table>

Table 18: Steering results using mass-mean probes on the identification (T1) and interpretation (T2) tasks. We intervene on correctly classified test examples and steer them toward the opposite label. “Before” and “After” indicate the numbers of examples retaining the original label before and after steering, respectively. Flip rate denotes the proportion of originally correct predictions that are flipped to the opposite label.

## J.4 Prompt format

All models use the same plain-continuation meaning-format prompt with five hardcoded few-shot examples demonstrating free-form meaning descriptions (no classification labels). The test query asks for the meaning of the adjective in context:

The welcome was warm. The meaning of ’warm’ is: affectionate and emotionally inviting.

The lamp was bright. The meaning of ’bright’ is: emitting a strong, clear light.

The novel was deep. The meaning of ’deep’ is: intellectually complex and rich in meaning.

The stone was smooth. The meaning of ’smooth is: having an even surface with no roughness.

The defeat was bitter. The meaning of ’bitter’ is: painful, hard to accept, and leaving resentment.

The [noun] was [adj]. The meaning of ’[adj]’ is:

No chat template is applied to any model. The hook patches the hidden state at the last token position at every forward step during generation. The token position is the same token position used for DAS training.

## K Prompt for LLM judge

You are classifying whether a meaning description conveys a literal or metaphorical sense of an adjective.

A LITERAL meaning describes a physical, concrete, or sensory property (e.g. temperature, texture, weight, brightness).

A METAPHORICAL meaning describes an abstract, figurative, or emotional property (e.g. mood, personality, intellectual quality).

Reply with exactly one word: either "literal" or "metaphorical".

No explanation. Do not repeat the sentence.

Sentence: "The noun was adj."

Adjective: ’adj’

Meaning description: "first sentence of completion"

Is this meaning literal or metaphorical?

## L Dataset Usage

We use datasets from Liu et al. (2022b); Gutiérrez et al. (2016); Tsvetkov et al. (2014). While we slight alter the format of these datasets, all are used for their original purpose, namely metaphor research. All datasets are freely available in the public domain.

## M Computational resources

All experiments took a total of about 125 hours on an H100 GPU. Hyperparameters checked include steering strength $\alpha \in \{ 1 , 2 , 5 , 1 0 \}$ for generation experiments and subspace dimension k ∈ {4, 8, 16, 32} for DAS matrix training.

## N Use of AI assistants

AI assistants are used for code debugging, generating the prompt for the generation experiment, and generating tables. Numbers are double-checked manually.