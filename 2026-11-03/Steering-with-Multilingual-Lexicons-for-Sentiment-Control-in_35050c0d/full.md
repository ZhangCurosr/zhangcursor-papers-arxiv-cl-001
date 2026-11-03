# Steering with Multilingual Lexicons for Sentiment Control in Low-Resource Languages

Lucinda Laurent<sup>1</sup> Muhammad Dehan Al Kautsar<sup>2</sup>

Alfan Farizki Wicaksono<sup>1</sup> Fajri Koto<sup>2</sup>

<sup>1</sup>Faculty of Computer Science, Universitas Indonesia

<sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence

lucinda.laurent@ui.ac.id, alfan@cs.ui.ac.id, {muhammad.dehan,fajri.koto}@mbzuai.ac.ae

## Abstract

Activation steering offers a lightweight mechanism for controlling model behavior, but existing methods depend on sentence-level contrastive data that is unavailable for most of the world’s languages. We show that multilingual sentiment lexicons are a viable substitute across three model families: aggregated multilingual lexicons outperform English-derived steering vectors for mid- and low-resource languages, while monolingual lexicons remain stronger for languages a model has specialized in during pre-training. The steering efficacy is mechanistically dictated by both representational density and pre-training priors. Large-magnitude English vectors overwhelm the compact representations of low-resource languages, causing “linguistic collapse.” This fragility is mitigated by the norm-dampening of multilingual aggregation. Architectural biases, including bilingual specialization penalty and latent numerical rigidity, also heavily constrain cross-lingual steering responsiveness. Ultimately, our work establishes lexicons as a practical steering signal and offers a comprehensive mechanistic account of cross-lingual steering dynamics.<sup>1</sup>

## 1 Introduction

Activation engineering has emerged as a lightweight method for controlling model outputs by manipulating internal activation vectors during inference, without modifying model weights or requiring curated training data (Turner et al., 2024; Rimsky et al., 2024; Arditi et al., 2024; Zou et al., 2025). By identifying linear directions associated with specific concepts (e.g., “sentiment”), one can steer the model’s behavior directly. In the context of sentiment-controlled generation, several studies have adopted activation engineering as an alternative to fine-tuning-based approaches (Turner et al., 2024; Liu et al., 2024a; Konen et al., 2024).

Despite its promise, current sentiment steering methodologies are constrained by a reliance on sentence-level supervision and evaluation benchmarks optimized for English-centric corpora. This dependence creates a significant barrier for low-resource languages, where high-quality sentiment-labeled sentence corpora are scarce or non-existent, leaving most of the world’s languages outside the reach of existing steering methods.

The effectiveness of these interventions across diverse languages also remains largely unexplored. This gap is critical because LLMs exhibit varying degrees of cross-lingual isomorphism: the extent to which semantic concepts share a similar geometric representation across languages (Lample et al., 2018). In models with strong “Anglocentric” bias, a sentiment vector derived from English may be orthogonal to, or incongruent with, the sentiment subspace of low-resource languages like Javanese or Hausa, mirroring findings on the lack of isomorphism in low-resource cross-lingual embeddings (Sannigrahi and Read, 2022). Relying solely on monolingual sentence-derived vectors thus constrains the scalability of activation steering in multilingual contexts.

In this work, we propose multilingual sentiment lexicons as a scalable, resource-efficient signal for cross-lingual activation steering. Our contributions are as follows: (1) A lexicon-based steering framework. We formalize activation steering via multilingual sentiment lexicons (illustrated in Figure 1), eliminating the need for sentence-level annotations or paired contrastive prompts. (2) Systematic cross-lingual evaluation. We benchmark monolingual versus multilingual lexicon steering across 11 typologically diverse languages spanning high-, mid-, and low-resource tiers and three model families. We show that aggregated multilingual lexicons outperform English-derived steering vectors for most mid- and low-resource languages, while monolingual lexicons remain superior in narrower cases where a model is heavily specialized in a particular language. (3) Mechanistic analysis of cross-lingual steering dynamics. We demonstrate that steering success is dictated by a model’s pre-training priors and latent scale disparities. We show that large-norm English vectors cause linguistic collapse by overwhelming the compact latent subspaces of low-resource languages, whereas multilingual aggregation produces more balanced vectors that eliminate this mismatch and prevent representation collapse. Furthermore, we reveal how architectural biases determine a model’s responsiveness to cross-lingual interventions.

![](images/2be523e5bf1b5a203075b2f36d4352a4ac1a488c8bc91970057d88d325f216de.jpg)  
Figure 1: Workflow of the Cross-Lingual Sentiment Control using Multilingual Lexicon. In the Extraction Phase, feature attribution selects the steering layers $i \in S$ (linear-probe macro-F1) and, within each, the neurons ${ \mathcal { N } } _ { i }$ (XGBoost gain). At each selected layer, the steering vector is the difference $\mu ^ { + , ( i ) } - \mu ^ { - , ( i ) }$ between multilingual positive and negative lexicon centroids, sliced to ${ \mathcal { N } } _ { i }$ . In the Injection Phase, this vector is applied to neutral prompts with coefficient +α or −α, shifting the generated text toward positive or negative sentiment across multiple languages.

## 2 Background

CTG via Activation Steering. Controllable Text Generation (CTG) is the task of generating natural sentences that adhere to certain attributes (Prabhumoye et al., 2020). According to Liang et al. (2024), CTG methods can be classified based on the stage at which the intervention occurs: the training stage (e.g. retraining, fine-tuning, reinforcement learning) and the inference stage (e.g. prompt engineering, latent space manipulation, decoding-time intervention). Although training phase methods provide high-fidelity control, they are often constrained by high computational costs and substantial data requirements, making them impractical for low-resource language settings, where the extensive corpora required for effective training are unavailable. In this paper, we utilize the latent space manipulation (also known as activation engineering) method to exploit language model’s underlying semantic geometry, potentially offering a more universal and robust framework for cross-lingual control.

We specifically use the contrastive latent vector acquisition method to derive the steering direction (or steering vector). Several existing works have employed this approach for sentiment control, such as Activation Addition (Turner et al., 2024), In-Context Vectors (Liu et al., 2024a), and Style Vectors (Konen et al., 2024). Activation Addition technique derived its steering vector by computing the difference of intermediate activations on a contrast pair of prompts, such as $h ( ^ { \bullet \bullet } \mathrm { L o v e } ^ { \bullet \bullet } ) - h ( ^ { \bullet \bullet } \mathrm { H a t e } ^ { \bullet \bullet } )$ where $h ( w )$ represents the activation elicited by the word w. The $h ( ^ { \bullet } \mathrm { L o v e } ^ { \mathfrak { W } } ) \ : - \ : h ( ^ { \bullet } \mathrm { H a t e } ^ { \mathfrak { W } } )$ steering direction is then added during forward pass at inference time to shift the activations to generate positive sentiment completions. On the other hand, In-Context Vector (ICV) is derived by applying Principal Component Analysis (PCA) on the differences between hidden states from multiple example pairs $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } \{$

$$
\mathtt { I C V } = \mathtt { P C A } \left( \{ h ( y _ { i } ) - h ( x _ { i } ) \} _ { i = 1 } ^ { n } \right) .
$$

In the demonstration for sentiment, the example pairs used are pairs of negative and positive text like (“Zero stars, I hate it.”, “Five stars, I love it.”).

Our method for deriving the sentiment steering direction is most similar to the Activation-based Style Vectors technique, in contrast to the other two approaches discussed. This technique extends the concept of Activation Addition by calculating a general style vector for a specific category $s \in S$ This is achieved by computing the difference between the mean activation of sentences in style category s and the mean activation of all sentences in the remaining style categories, S \ s. In this study, instead of relying on labeled sentiment datasets such as Yelp Review (Zhang et al., 2015), we utilize sentiment lexicons consisting of positive and negative words to address the inherent data scarcity in low-resource language settings.

Mechanistic Interpretability and Linear Representations. Rather than modifying model weights through fine-tuning, activation engineering exploits the pre-existing representations within the language model. It relies on the Linear Representation Hypothesis, assuming that semantic attributes are encoded as linear directions, thus allowing for direct steering within the representation space (Park et al., 2024). Previous research by Tigges et al. (2024) has shown that language models linearly represent sentiment, where methods like kmeans, linear probing, and Distributed Alignment Search (DAS), originally introduced by Geiger et al. (2024), converge on a similar sentiment direction. Di Palma et al. (2025) further corroborates this by unveiling linearly probe-able sentiment and emotion representations specifically within the Llama model family. We extend these findings by investigating the cross-lingual universality of sentiment directions; specifically, whether lexicon-derived sentiment directions align in a shared representation space across multiple languages. We primarily utilize the difference-in-means method to derive sentiment directions, complemented by linear probing to evaluate the consistency of sentiment representation across model layers.

The findings by Liu et al. (2024b) suggest that in generative language models, lexical semantics are highly concentrated in the lower layers. By evaluating hidden states on the Word in Context (WiC) benchmark, they demonstrate that lower layers manage contextual disambiguation (determining whether identical words convey the same meaning in distinct contexts), while higher layers shift computational focus toward next-token prediction. In contrast, our approach evaluates semantic polarity without surrounding context. We isolate the sentiment axis by extracting global representations directly from uncontextualized lexicon entries, which can range from single words to multi-word phrases. Because semantic encoding varies significantly across network depth (Liu et al., 2024b), mapping these isolated representations layer-by-layer allows us to empirically identify the optimal intervention point for cross-lingual sentiment steering using a multilingual lexicon.

Cross-Lingual Alignment and Universality. While the Linear Representation Hypothesis suggests stable sentiment directions within a single language, recent evaluations highlight that the generalization and reliability of steering vectors can degrade significantly under out-of-distribution shifts (Tan et al., 2025). Because cross-lingual transfer inherently represents a massive distributional shift, the capacity of these vectors to generalize zero-shot across languages remains a critical open question. Ideally, multilingual LLMs would exhibit crosslingual isomorphism, where semantic spaces across languages share a similar geometric structure (Lample et al., 2018). This phenomenon was first empirically observed in massively multilingual encoders like mBERT (Devlin et al., 2019) and XLM-R (Conneau et al., 2020), which demonstrated that pre-training on a shared vocabulary across 100+ languages naturally induces a “universal” representation space where semantically similar tokens align across languages without explicit translation supervision (Libovický et al., 2020; Pires et al., 2019). Under this assumption, a sentiment direction derived in English should theoretically transfer to other languages.

Despite the assumption of a universal space, pretraining on unbalanced corpora frequently produces an “Anglocentric” internal structure (Ahuja et al., 2023; Lauscher et al., 2020). Within this structure, high-resource languages align closely with English subspaces, whereas mid-to-low resource languages suffer from weaker alignment or severe representation degeneration. This representational gap directly challenges cross-lingual steering; if the “sentiment axis” in a low-resource language is disjoint from the English axis, monolingual steering will fail. This geometric misalignment remains intrinsically tied to the model’s underlying linguistic proficiency. Multilingual architectures rely on language-specific neurons to map inputs into the shared semantic space and decode them back into target vocabularies (Tang et al., 2024). Without robust neurons for low-resource languages, a severe projection bottleneck emerges, suggesting that the model’s limited linguistic proficiency acts as a critical constraint on cross-lingual control, independent of the latent steering mechanism’s geometric accuracy.

Operating under these constraints, we propose that aggregation of multilingual lexicons can serve as a geometric bridge. We hypothesize that if the underlying alignment is sufficiently preserved, a multilingual direction can reinforce the shared sentiment direction via coherent reinforcement. Conversely, in cases of extreme representation degeneration, such aggregation may fail to overcome the lack of semantic grounding.

## 3 Methodology

An overview of our cross-lingual sentiment steering framework is illustrated in Figure 1.

## 3.1 Sentiment Direction Extraction

Among the various methods for obtaining steering direction, difference-in-means is one of the most simple, yet effective approach (Marks and Tegmark, 2024; Belrose, 2023). To extract the steering direction corresponding to sentiment attribute, we employ the difference-in-means method using two opposing sets of lexicon entries (phrases), denoted as $\mathcal { L } ^ { + }$ and $\mathcal { L }$ <sup>−</sup>.

Definitions. Let V denote the model’s vocabulary consisting of individual tokens. Recognizing that modern language models utilize sub-word tokenization (e.g., Byte-Pair Encoding), a single word or phrase in a lexicon may be decomposed into multiple tokens. We therefore define a lexicon entry l as a sequence of tokens $l = ( t _ { 1 } , t _ { 2 } , \dots , t _ { | l | } )$ , where each $t _ { j } \in \mathcal { V }$ and $| l | \geq 1$ represents the length of the entry. For example, the word “abhorrence” is tokenized into $( ^ { \ast \bullet } \mathrm { a b } ^ { , \ast } , ^ { \ast } \mathrm { h o r } ^ { , \ast } , ^ { \ast } \mathrm { r e n c e } ^ { , \ast } )$ using Qwen-3’s tokenizer. Accordingly, we define two contrastive sets of lexicon entries:

$\mathcal { L } ^ { + } = \{ l _ { 1 } ^ { + } , l _ { 2 } ^ { + } , \ldots , l _ { N } ^ { + } \}$ representing the positive lexicon set; and

$\mathcal { L } ^ { - } = \{ l _ { 1 } ^ { - } , l _ { 2 } ^ { - } , \ldots , l _ { M } ^ { - } \}$ representing the negative lexicon set.

Representation of Lexicon Entries. Let $h ^ { ( i ) } ( t ) \in \mathbb { R } ^ { d }$ denote the hidden state vector of a token t at layer i. The vector representation of a lexicon entry l consisting of tokens $( t _ { 1 } , \ldots , t _ { | l | } )$ at layer i, denoted as $\phi ^ { ( i ) } ( l )$ , is defined as the average of the hidden states of its constituent tokens at that specific layer:

$$
\phi ^ { ( i ) } ( l ) = \frac { 1 } { | l | } \sum _ { j = 1 } ^ { | l | } h ^ { ( i ) } ( t _ { j } ) .
$$

By using these averaged hidden states as input, we ensure that the vector representation directly reflects the model’s internal encoding of the sentiment-bearing terms in our lexicon. Furthermore, averaging across constituent tokens avoids over-reliance on the final token, which has been shown to be inconsistently informative for sentiment representation in decoder-only architectures, such as the Llama family (Di Palma et al., 2025).

Layer-wise Centroids. We compute the centroid for the positive and negative groups by averaging the lexicon entry representations $\phi ^ { ( i ) } ( l )$ strictly within the context of each layer i:

$$
\pmb { \mu } ^ { + , ( i ) } = \frac { 1 } { | \mathcal { L } ^ { + } | } \sum _ { l \in \mathcal { L } ^ { + } } \phi ^ { ( i ) } ( l ) ,
$$

$$
\pmb { \mu } ^ { - , ( i ) } = \frac { 1 } { | \mathcal { L } ^ { - } | } \sum _ { l \in \mathcal { L } ^ { - } } \phi ^ { ( i ) } ( l ) .
$$

Here, $\lvert \mathcal { L } ^ { + } \rvert$ and $| { \mathcal { L } } ^ { - } |$ represents the number of lexicon entries in each set.

Layer-specific Steering Direction. Finally, the steering direction is computed independently for each layer i over the full hidden dimension d. The steering direction is determined by the desired target attribute. To steer the model towards the positive attribute (enhancing $\mathcal { L } ^ { + }$ traits), the steering

direction is defined as:

$$
{ \bf v } _ { { \bf p } { \bf 0 } s } ^ { ( i ) } = { \pmb { \mu } } ^ { + , ( i ) } - { \pmb { \mu } } ^ { - , ( i ) } .
$$

Conversely, to steer the model towards the negative attribute (or to suppress $\mathcal { L } ^ { + }$ traits), we reverse the direction: $\mathbf { v } _ { \mathbf { n e g } } ^ { ( i ) } = - \mathbf { v } _ { \mathbf { p o s } } ^ { ( i ) }$

## 3.2 Feature Attribution

In the context of activation steering, probing is frequently employed to identify the model components that contribute most significantly to the target attribute, thereby enabling more targeted interventions. For instance, Li et al. (2023) and Wang et al. (2025) utilize probes to select specific attention heads for steering, while Konen et al. (2024) employ probing to determine the optimal layers for intervention. Building upon these approaches, we propose a two-stage feature attribution mechanism: we first identify the layers with the highest sentiment density, and then isolate a sparse subset of neurons within those layers that contribute most significantly to the sentiment representation.

Both stages are trained on the lexicon entry representations defined in Section 3.1. For each layer, we construct a labeled dataset $\begin{array} { r l } { \mathcal { D } ^ { ( i ) } } & { { } = } \end{array}$ $\{ ( \phi ^ { ( i ) } ( l ) , y _ { l } ) \ : \ l \ \in \ \mathcal { L } ^ { + } \cup \mathcal { L } ^ { - } \}$ , where $y _ { l } ~ = ~ 1$ if $l \in \mathcal { L } ^ { + }$ and 0 otherwise. $\dot { \mathcal { D } } ^ { ( i ) }$ contains the same entries and labels for every layer i and differs only in the layer at which the representations are taken. Since probing in isolation can capture noncausal correlations rather than true task relevance (Ravichander et al., 2021; Saphra and Wiegreffe, 2024), we conduct rigorous intervention experiments against random baselines, fully detailed in the Appendix I.

Layer-wise Probing. We hypothesize that sentiment information is non-uniformly distributed across the model’s depth. To identify the most informative layers, we train a linear probe on $\mathcal { D } ^ { ( i ) }$ for each layer i to predict the sentiment polarity $y \in \{ 0 , 1 \}$ using a logistic regression objective:

$$
P ( y \mid i , l ) = \sigma \left[ { \bf w } _ { i } ^ { T } \phi ^ { ( i ) } ( l ) + b _ { i } \right] ,
$$

where $\mathbf { w } _ { i } ~ \in ~ \mathbb { R } ^ { d }$ and $b _ { i } ~ \in ~ \mathbb { R }$ are the learnable weights and bias of the probe, respectively. The layers are then ranked according to the probes macro-F1 scores on the held-out portion of $\mathcal { D } ^ { ( i ) }$ Since the optimal subset of layers for intervention may vary across architectures and linguistic contexts, we consider three candidate layer configurations: all layers, the top 50% of layers, and the top

5 layers. For each combination of model, target language, and direction type (monolingual vs. multilingual), we use the configuration that yields the highest Success Rate (SR) after injection (Tables 1 and 2). We denote the resulting set of selected layers by S.

Sparse Neuron Selection via Non-linear Attribution. Steering all neurons within a selected layer may introduce unnecessary noise. Our preliminary experiments with Qwen and Llama indicate that this selection is an empirical necessity; unconstrained steering frequently resulting in incoherent, ‘gibberish’ outputs. To mitigate this, we perform neuronlevel attribution using Gradient Boosted Decision Trees (GBDT) via XGBoost. While sparsity-inducing linear probes (e.g., L1-regularized Logistic Regression) can reduce feature counts, they typically retain a broader set of neurons based on linear separability. In contrast, tree-based models perform a more aggressive, non-linear feature selection, isolating a highly compact subset of neurons highly discriminative for sentiment representation.

For each layer $i ,$ we train an XGBoost classifier on $\mathcal { D } ^ { ( i ) }$ , using the individual neurons of $\phi ^ { ( i ) } ( l )$ denoted by $\{ n _ { i , 1 } , n _ { i , 2 } , . . . , n _ { i , d } \}$ , as input features. We then calculate the gain of each neuron $n _ { i , j } .$ denoted by $G ( n _ { i , j } )$ , which measures the average improvement in the training objective over all split points in the ensemble that use that neuron:

$$
G ( n _ { i , j } ) = \frac { 1 } { | \mathcal { P } _ { i , j } | } \sum _ { p \in \mathcal { P } _ { i , j } } \Delta \ell _ { p } ,
$$

where $\mathcal { P } _ { i , j }$ is the set of split points, across all trees, that split on neuron $n _ { i , j } ;$ and $\Delta \ell _ { p }$ is the improvement in the regularized log-loss objective achieved by split $p ,$ as computed by XGBoost’s split criterion. We set $G ( n _ { i , j } ) = 0$ for neurons that are never used in a split $( \mathcal { P } _ { i , j } = \emptyset )$ . For each selected layer $i \in S$ we define the active steering set ${ \mathcal { N } } _ { i }$ as the indices of the neurons with a non-zero gain $( G ( n _ { i , j } ) > 0 )$

## 3.3 Sentiment Direction Injection

We apply the derived steering direction during the inference phase by intervening on the model’s internal activations. The intervention is performed at every forward pass of the model, specifically targeting the hidden states corresponding to the final token of the current sequence.

Let $\mathbf { x } = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { t } )$ be the input sequence at the current generation step $t ,$ where $x _ { t }$ represents the last token (the most recently generated or processed token), and let $h ^ { ( i ) } ( x _ { t } )$ denote the original hidden state at the final token position for layer i. For each selected layer $i \in S ,$ , we slice the fulldimensional steering direction $\mathbf { v } ^ { ( i ) }$ , which can be either $\mathbf { v } _ { \mathtt { p o s } } ^ { ( i ) } \mathrm { o r } \ \mathbf { v } _ { \mathtt { n e g } } ^ { ( i ) }$ , to the active steering set ${ \mathcal { N } } _ { i }$ and add the scaled result to the corresponding neurons of the hidden state before it is passed to the subsequent layer $i + 1 \colon$

$$
\tilde { h } _ { [ \mathcal { N } _ { i } ] } ^ { ( i ) } ( x _ { t } ) \gets h _ { [ \mathcal { N } _ { i } ] } ^ { ( i ) } ( x _ { t } ) + \alpha \cdot \mathbf { v } _ { [ \mathcal { N } _ { i } ] } ^ { ( i ) } ,
$$

where the subscript $[ \mathcal { N } _ { i } ]$ denotes the sub-vector indexed by $\mathcal { N } _ { i } ; \tilde { h } ^ { ( i ) } ( x _ { t } )$ is the steered hidden state passed to the next layer; and α is the steering coefficient (hyperparameter) controlling the intervention strength. All neurons outside ${ \mathcal { N } } _ { i } .$ , as well as all layers $i \notin S ,$ , are left unchanged. By only intervening on the neurons in ${ \mathcal { N } } _ { i }$ , we prevent the ‘gibberish’ output phenomenon observed in unconstrained steering, thereby maintaining the structural integrity of the latent space while maximizing steering efficacy.

## 4 Experiments

## 4.1 Experimental Setup

Models and Lexicon. We evaluate three base LLMs: Llama-3.2-3B (Grattafiori et al., 2024), Gemma-3-4B-PT (Gemma Team et al., 2025), and Qwen3-4B-Base (Yang et al., 2025). To derive steering directions, we utilize the NRC-VAD lexicon (Mohammad, 2018), which provides human-rated valence scores [0, 1]. We apply strict thresholds $( > 0 . 8$ for $\mathcal { L } ^ { + }$ and $< 0 . 2$ for ${ \mathcal { L } } ^ { - } )$ to maximize semantic contrast and ensure robust steering vectors. We compare two configurations: Monolingual (English only) and Multilingual, which aggregates and deduplicates lexicons across 11 typologically diverse languages spanning three resource tiers: high-resource (English, German, Chinese, Japanese), mid-resource (Indonesian, Icelandic, Estonian, Basque), and low-resource (Hausa, Javanese, Sundanese). Detailed preprocessing and model configurations are provided in Appendix A.

Task, Dataset Construction, and Evaluations. We evaluate cross-lingual steering on an openended generation task: given a neutral prompt, the model generates a continuation, and we measure whether steering shifts the sentiment of the generated text toward the target polarity while preserving the language of the prompt. To support this evaluation, we constructed a large-scale suite of 11,000 neutral prompts across 11 languages. We first generated 1,000 unique neutral English prompts using a hierarchical structure (10 categories $\times \ 1 0$ topics $\times \ 1 0$ prompts per topic), then translated them into 10 target languages using Gemini. For low-resource languages, we employed style-guided prompts (e.g., specifying Javanese Ngoko Lugu and Sundanese Basa Loma) to strictly maintain a natural, conversational style. To ensure baseline reliability, we manually audited a 10% random sample (100 prompts) of the unique English set (see Appendix A.2).

To assess steering, models generate continuations from these neutral points with a constraint of at most 128 new tokens. This setup ensures sufficient context for sentiment analysis without forcing a fixed generation length. A separate tuning split of 50 prompts was used for hyperparameter selection (see Appendix B for topic lists and prompts). Continuations are evaluated using GPT-4.1 (OpenAI et al., 2024) as a sentiment judge and GlotLID (Kargaran et al., 2023) for language identification. The use of large language models for automated text annotation has been shown to outperform crowdworkers across various classification tasks (Gilardi et al., 2023), with specific validation demonstrating high human-alignment in sentiment and emotion analysis (Calderon et al., 2025; Ruder et al., 2025). To confirm this reliability extends to our specific low-resource settings, we conducted native-speaker validation audits (detailed in Appendix A.5). Based on these automated evaluations, we report three metrics: (1) Sentiment Accuracy (SA) and (2) LID Rate (LIDR) as constituent components, and (3) Success Rate (SR), which serves as our primary metric for analysis. The Success Rate represents a strict intersection, requiring a generation to satisfy both correct sentiment and correct language simultaneously. Steering intensity α was optimized via grid search on the tuning split (details in Appendix A).

## 5 Results and Analysis

## 5.1 Effectiveness of Cross-Lingual Steering

We first evaluate the impact of our steering directions on the model’s generation. Figure 2 visualizes the sentiment distribution shifts for the Gemma-3 model across 11 languages using the steering direction derived from the multilingual lexicon (see Figure 7 for the corresponding distributions utilizing monolingual lexicon). With no intervention (Figure 2a), the model generates continuations based on its intrinsic probability distribution. Since the prompts are semantically neutral (e.g., ‘He turned on the computer’), the resulting sentiments are naturally diverse or predominantly neutral. In contrast, applying the steering direction results in a dramatic distributional shift: Positive Steering (Figure 2b) successfully aligns the majority of generations with positive sentiment, while Negative Steering (Figure 2c) induces negative sentiment. These shifts indicate that lexicon-derived directions provide a viable basis for modulating sentiment across diverse languages. However, this responsiveness is not uniform across architectures. While the sentiment shift is nearly total in Gemma-3, we observe varying degrees of sensitivity in Qwen-3 and Llama-3.2 (see Appendix D). In these models, although the intervention significantly increases the target sentiment, they maintain a higher proportion of neutral generations, particularly in lower-resource languages such as Hausa (ha) and Sundanese (su). This suggests that model-specific biases and language-specific representation density play a role in the effectiveness of lexicon-derived steering.

![](images/468cea941a4b735e8b84fe6c0410aae237aa9fbd911867680b9d3b6b4c3500d0.jpg)  
Figure 2: Impact of Steering on Sentiment Distribution (Gemma-3). The distribution of generated sentiments across 11 languages under three conditions: (a) No Intervention (as baseline), (b) Positive Steering, and (c) Negative Steering. The proposed method successfully shifts the predominantly neutral baseline toward the target sentiment. Corresponding distributions for Qwen-3 and Llama-3.2 are provided in Appendix D.

<table><tr><td></td><td></td><td colspan="2">Gemma-3</td><td colspan="2">Qwen-3</td><td colspan="2">Llama-3.2</td></tr><tr><td>Lang</td><td>Res.</td><td>SR (%)</td><td>Layers</td><td>SR (%)</td><td>Layers</td><td>SR (%)</td><td>Layers</td></tr><tr><td>English (en)</td><td>High</td><td>94.3</td><td>all</td><td>45.1</td><td>all</td><td>65.8</td><td>all</td></tr><tr><td>German (de)</td><td>High</td><td>90.5</td><td>all</td><td>46.1</td><td>top-50%</td><td>53.4</td><td>all</td></tr><tr><td>Chinese (zh)</td><td>High</td><td>86.2</td><td>all</td><td>48.0</td><td>all</td><td>33.1</td><td>all</td></tr><tr><td>Japanese (ja)</td><td>High</td><td>76.8</td><td>all</td><td>23.5</td><td>all</td><td>18.9</td><td>all</td></tr><tr><td>Indonesian (id)</td><td>Mid</td><td>82.0</td><td>all</td><td>46.8</td><td>all</td><td>37.0</td><td>all</td></tr><tr><td>Icelandic (is)</td><td>Mid</td><td>66.8</td><td>all</td><td>10.9</td><td>all</td><td>23.1</td><td>top-50%</td></tr><tr><td>Estonian (et)</td><td>Mid</td><td>77.8</td><td>all</td><td>23.5</td><td>top-50%</td><td>25.4</td><td>top-50%</td></tr><tr><td>Basque (eu)</td><td>Mid</td><td>60.7</td><td>all</td><td>12.2</td><td>all</td><td>20.4</td><td>top-50%</td></tr><tr><td>Hausa (ha)</td><td>Low</td><td>58.9</td><td>top-50%</td><td>2.1</td><td>top-5</td><td>9.6</td><td>top-5</td></tr><tr><td>Javanese (jv)</td><td>Low</td><td>37.9</td><td>top-50%</td><td>9.8</td><td>top-50%</td><td>4.2</td><td>top-50%</td></tr><tr><td>Sundanese (su)</td><td>Low</td><td>27.2</td><td>all</td><td>5.7</td><td>top-50%</td><td>5.9</td><td>top-50%</td></tr></table>

Table 1: Cross-Lingual Steering Performance with Multilingual Lexicons. Success Rate (SR, %) across three models and 11 languages. Each cell reports SR at the best-performing layer configuration (all, top-50%, or top-5).

<table><tr><td></td><td></td><td colspan="2">Gemma-3</td><td colspan="2">Qwen-3</td><td colspan="2">Llama-3.2</td></tr><tr><td>Lang</td><td>Res.</td><td>SR (%)</td><td>Layers</td><td>SR (%)</td><td>Layers</td><td>SR (%)</td><td>Layers</td></tr><tr><td>English (en)</td><td>High</td><td>95.0</td><td>all</td><td>44.5</td><td>all</td><td>66.4</td><td>all</td></tr><tr><td>German (de)</td><td>High</td><td>90.7</td><td>all</td><td>45.4</td><td>top-50%</td><td>53.8</td><td>all</td></tr><tr><td>Chinese (zh)</td><td>High</td><td>86.8</td><td>all</td><td>53.4</td><td>all</td><td>34.3</td><td>all</td></tr><tr><td>Japanese (ja)</td><td>High</td><td>80.2</td><td>all</td><td>23.9</td><td>top-50%</td><td>20.8</td><td>all</td></tr><tr><td>Indonesian (id)</td><td>Mid</td><td>82.3</td><td>all</td><td>42.9</td><td>top-50%</td><td>38.6</td><td>all</td></tr><tr><td>Icelandic (is)</td><td>Mid</td><td>62.3</td><td>top-50%</td><td>12.1</td><td>top-50%</td><td>20.0</td><td>all</td></tr><tr><td>Estonian (et)</td><td>Mid</td><td>78.0</td><td>top-50%</td><td>19.2</td><td>top-50%</td><td>22.1</td><td>top-50%</td></tr><tr><td>Basque (eu)</td><td>Mid</td><td>54.7</td><td>top-50%</td><td>9.2</td><td>top-50%</td><td>19.6</td><td>all</td></tr><tr><td>Hausa (ha)</td><td>Low</td><td>54.6</td><td>top-50%</td><td>1.8</td><td>top-5</td><td>9.3</td><td>top-50%</td></tr><tr><td>Javanese (jv)</td><td>Low</td><td>37.8</td><td>top-50%</td><td>10.2</td><td>top-50%</td><td>4.7</td><td>all</td></tr><tr><td>Sundanese (su)</td><td>Low</td><td>25.6</td><td>top-50%</td><td>4.7</td><td>top-50%</td><td>7.5</td><td>all</td></tr></table>

Table 2: Cross-Lingual Steering Performance with Monolingual English Lexicons. Success Rate (SR, %) across three models and 11 languages. Each cell reports SR at the best-performing layer configuration (all, top-50%, or top-5).

To quantify this effectiveness, Tables 1 and 2 summarize the performance of Multilingual and Monolingual (english) steering directions. First, both strategies significantly alter the model’s behavior to align the output with the target sentiment compared to the uncontrolled baseline. Second, the Multilingual Lexicon aggregation emerges as a critical enabler for mid- and low-resource languages. This is most evident in Gemma-3, where the Multilingual vector consistently outperforms the Monolingual counterpart in lower-resource settings. For instance, the Multilingual approach improves the success rate for Icelandic (is) from 62.3% to

66.8% (+4.5%) and for Basque (eu) from 54.7% to 60.7% (+6.0%). Even for Hausa (ha), we see an improvement of over 4 percentage points (54.6% to 58.9%).

Interestingly, in Qwen-3, we observe a notable boost in Indonesian (id), rising from 42.9% to 46.8%. This demonstrates that aggregating diverse lexicons creates a more robust, languageagnostic sentiment direction that helps bridge the representation gap in under-resourced languages. To ensure the robustness of these observed gains against stochastic decoding variance, we conducted rigorous statistical significance testing (Two-Sample Proportion Z-Tests) across all resource tiers. We find that the multilingual steering advantage is highly significant in Mid- and capable Low-resource regimes (see Appendix F). While the Multilingual direction is generally superior for low-resource languages across all models, the Monolingual vector remains more effective for specific high-resource pairs, most notably Qwen-3 in Chinese (zh), where performance drops from 53.4% to 48.0% when using the Multilingual lexicon. In Llama-3.2, the differences between Mono and Multi are often marginal in high-resource settings (e.g., 66.4% vs 65.8% for English). These variations suggest that the optimal steering strategy is heavily influenced by the model’s pre-training dynamics, which we analyze further in Section 5.3.

## 5.2 Sensitivity and Robustness Analysis

To determine the optimal steering intensity and evaluate the stability of our vectors, we conducted a hyperparameter tuning sweep on the coefficient $\alpha \in \{ 0 . 5 0 , 0 . 7 5 , 1 . 0 0 \}$ . We focused this analysis on a representative subset of languages ranging from high- to low-resource: English (en), German (de), Chinese (zh), Indonesian (id), and Javanese (jv). Figure 3 visualizes the results.

Fragility of High-Norm Monolingual Steering Directions. The left column of Figure 3 reveals that Monolingual steering is highly unstable at higher intensities. As $\alpha ~  ~ 1 . 0 \ :$ , we observe widespread linguistic collapse (black cells), particularly in Qwen-3 and Llama-3.2, where the model output becomes incoherent. We hypothesize that this fragility stems from the high magnitude (norm) of the Monolingual steering direction. Since English sentiment representations are highly separable, the resulting steering direction encodes a “sharp” and distinct direction with a large norm.

![](images/b39fc6c7b54d18c7e1024e4b709d1966ca6c01683e13ec9accb7f1f8cef702ea.jpg)  
Figure 3: Impact of Steering Intensity on Success Rate and Language Integrity. Heatmaps show performance under the all-layers configuration as the steering coefficient α increases from 0.5 to 1.0, for steering vectors derived from monolingual (English) and multilingual lexicons. Values represent Success Rate (%); black cells indicate linguistic collapse, where the generated text was incoherent and thus could not be evaluated for sentiment.

When this high-norm vector is injected into the latent space of other languages, especially lowresource ones like Javanese where representations are less distinct, it pushes the activations far out of the natural distribution, causing the generation to break down.

Robustness and Norm Dampening in Multilingual Steering Directions. In contrast, the Multilingual steering directions (right column) generally demonstrate superior stability. By aggregating lexicons from diverse languages, the resulting vector represents a “consensus” direction. We posit that this aggregation process often creates a dampening effect on the vector norm, particularly when the component languages have varying alignment. This strictly moderated norm allows for stronger steering intensities without shattering the model’s linguistic integrity.

Power vs. Stability in Gemma-3. Gemma-3 demonstrates a unique power-stability trade-off: while achieving superior steering shifts, it is prone to linguistic collapse at $\alpha = 1 . 0$ (Figure 3). This fragility stems from a massive latent scale disparity: as measured across layers in Appendix G, Gemma-3 operates in a high-variance regime $( L _ { 2 } >$

![](images/c4f931bdf1bc1f6388ca2583c458877936869884710b5f6a75c7c2730868858e.jpg)  
Figure 4: Heterogeneous Impact of Multilingual Aggregation Across Model Families. This figure highlights the divergent impact of the Multilingual steering direction across three distinct model families. Positive values (teal) indicate that the Multilingual direction outperforms the Monolingual direction.

10, 000) compared to the compressed spaces of Llama-3.2 (∼ 25) and Qwen-3 (∼ 1, 150). Within Gemma-3, high-resource anchors (EN, ZH) define a “norm ceiling,” while low-resource languages like Javanese occupy a compact subspace (see Figure 8). Consequently, the multilingual steering signal (∼ 5, 000) is 2.5× larger than the native Javanese norm (∼ 2, 000). At maximum intensity, this disproportionate force shatters the linguistic integrity of lower-resource representations while remaining effective for high-norm languages. This suggests that universal steering requires intensity scaling relative to a language’s native representational density.

## 5.3 Cross-Lingual Dynamics and Model Architecture

While Section 5.2 analyzed the stability of the steering vectors, this section examines their relative effectiveness. Figure 4 illustrates the performance difference, defined as:

$$
\Delta = \mathsf { S R } _ { \mathrm { M u l t i } } - \mathsf { S R } _ { \mathrm { M o n o } } ,
$$

where $\mathsf { S R } _ { m }$ denotes the success rate for configuration m ∈ {Multi, Mono} under its optimal layer configuration for that specific model and target language. Note that $\Delta$ is computed across 11 languages. The patterns across the three models highlight how model architecture and pre-training priors dictate the optimal steering strategy.

Architectural Drivers of Steering Efficacy. The patterns in Figure 4 reveal that a model’s pretraining priors dictate its responsiveness to multilingual aggregation. Gemma-3 exhibits the most consistent cross-lingual synergy, where multilingual vectors provide a coherent reinforcement signal for low-resource languages (e.g., ha, su) with minimal performance drops in high-resource languages: a benefit likely rooted in its optimized sub-word vocabulary. In contrast, Qwen-3 incurs a “Specialization Penalty”; its bilingual-centric optimization for Chinese creates highly distinct representations that resist multilingual “noise,” leading to a > 5% performance drop. For Llama-3.2, multilingual aggregation serves as a critical bridge for mid-resource languages (e.g., is, et) where the English vector is semantically distant, yet it remains insufficient to overcome the baseline “neutral anchor” in the lowest-resource settings (jv, su). This resistance occurs despite high cross-lingual vector alignment (see Appendix H), a paradox likely driven by the model’s compressed latent scale and static layer-wise dynamics which create an internal numerical rigidity against external perturbations (see Appendix G).

## 6 Conclusion

We introduced multilingual sentiment lexicons as a scalable signal for cross-lingual activation steering, providing a practical alternative to sentence-level supervision where such data is unavailable. Across three model families and 11 languages, aggregated multilingual lexicons outperform English-derived steering vectors for mid- and low-resource languages, while monolingual lexicons remain preferable when a model is heavily specialized in a particular language. Our analysis further shows that successful intervention depends on internal model properties beyond surface-level similarity between sentiment directions: architectural specialization can decouple vector similarity from steering performance, and the scale mismatch between highresource directions and compact low-resource representations can trigger linguistic collapse. These findings motivate a shift toward adaptive steering that calibrates intervention magnitude and lexicon configuration to each target language’s representational scale.

## Limitations

Task and Model Scope. Our experiments were conducted exclusively on open-ended sentence completion tasks using pre-trained base models. We did not evaluate the efficacy of these steering vectors on instruction-following tasks (e.g., QA, summarization) or on instruction-tuned checkpoints (Chat models). Consequently, it remains unverified whether the observed steering dynamics persist when the model is constrained by strict instruction formatting or safety alignment filters. Furthermore, our analysis is limited to efficientscale models in the 3B–4B parameter range. It is unclear if the trade-offs between stability and potency scale linearly to larger models (e.g., 70B+ parameters), where the latent space typically exhibits greater robustness and separability.

Generalization to Unseen Languages. The Multilingual directions in this study were aggregated using lexicons from a specific set of 11 languages. We evaluated the steering performance only on these in-domain languages and did not assess zeroshot transferability to unseen languages (languages whose lexicons were excluded from the aggregation). Evaluating on unseen languages would be an interesting extension and would further test the universality of the resulting directions. That said, the practical impact of this limitation is modest: sentiment lexicons for new languages can be obtained reliably through word-level translation, which is substantially more accurate and less error-prone than translating full sentences. Extending the multilingual lexicon set to cover additional languages is therefore straightforward in practice, even if the present results do not directly demonstrate zeroshot transfer.

Evaluation Metrics and Reliability. Our evaluation relied primarily on automated classifier-based metrics to measure sentiment shift and language consistency. First, we did not perform a deep quantitative analysis of textual fluency (e.g., perplexity) or topical relevance, meaning high sentiment scores do not guarantee that the generated text maintains semantic coherence or “naturalness.” Second, we acknowledge that the reliability of LLM-based evaluators (e.g., GPT-4) degrades in low-resource settings. These models may harbor English-centric biases or lack the nuance to accurately adjudge sentiment in languages like Sundanese or Javanese. Consequently, the reported success rates in these languages reflect the evaluator’s probability distribution but may not perfectly correspond to nativelevel perceptions of sentiment.

## Ethics Statement

Cultural Homogenization and Translation Artifacts. Our methodology relies on aggregating sentiment lexicons (NRC-VAD) which are constructed via automated translation from English without recalibrating valence scores for the target cultures. Consequently, the resulting “universal” vectors inherently propagate an Anglocentric definition of sentiment. By projecting English sentiment values onto other languages, we risk enforcing a homogenized alignment where concepts deemed “positive” in Western norms override local cultural nuances (e.g., definitions of politeness or positivity in Javanese that differ from English). Users of this method should be mindful that the “universality” achieved here is mathematical rather than cultural; the method prioritizes cross-lingual robustness over cultural precision, potentially risking cultural erasure in low-resource languages by aligning them to a foreign sentiment standard.

Dual-Use and Safety. While this study aims to improve cross-lingual control, activation steering is a dual-use technology. The same mechanisms used to enforce positive sentiment can be inverted to amplify toxicity or hate speech. Furthermore, as noted in our limitations, our current evaluation does not guarantee semantic coherence. Deploying such steering methods in production without rigorous fluency and safety checks could lead to the generation of superficially “positive” but nonsensical or hallucinated content, potentially degrading user trust.

Artifact and License Compliance. All artifacts used in this study are restricted to their intended research purposes. The code libraries are utilized under standard open-source licenses (e.g., MIT, Apache 2.0), and all datasets and pre-trained models are accessed in strict accordance with their original creators’ terms of use.

## References

Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram. 2023. MEGA: Multilingual evaluation of generative AI.

In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4232–4267, Singapore. Association for Computational Linguistics.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems, volume 37, pages 136037– 136083. Curran Associates, Inc.

Nora Belrose. 2023. Diff-in-means concept editing is worst-case optimal. EleutherAI Blog.

Nitay Calderon, Roi Reichart, and Rotem Dror. 2025. The alternative annotator test for LLM-as-a-judge: How to statistically justify replacing human annotators with LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16051– 16081, Vienna, Austria. Association for Computational Linguistics.

Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, page 785–794, New York, NY, USA. Association for Computing Machinery.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Dario Di Palma, Alessandro De Bellis, Giovanni Servedio, Vito Walter Anelli, Fedelucio Narducci, and Tommaso Di Noia. 2025. LLaMAs have feelings too: Unveiling sentiment and emotion representations in LLaMA models through probing. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6124–6142, Vienna, Austria. Association for Computational Linguistics.

Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah Goodman. 2024. Finding alignments between interpretable causal variables and distributed neural representations. In Proceedings ofthe Third Conference on Causal Learning and

Reasoning, volume 236 of Proceedings ofMachine Learning Research, pages 160–187. PMLR.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy ofSciences, 120(30).

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Amir Hossein Kargaran, Ayyoob Imani, François Yvon, and Hinrich Schuetze. 2023. GlotLID: Language identification for low-resource languages. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6155–6218, Singapore. Association for Computational Linguistics.

Christopher SG Khoo and Sathik Basha Johnkhan. 2018. Lexicon-based sentiment analysis: Comparative evaluation of six sentiment lexicons. Journal ofInformation Science, 44(4):491–511.

Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer Schütt, Oliver Bensch, Roxanne El Baff, Dominik Opitz, and Tobias Hecking. 2024. Style vectors for steering generative large language models. In Findings of the Association for Computational Linguistics: EACL 2024, pages 782–802, St. Julian’s, Malta. Association for Computational Linguistics.

Guillaume Lample, Alexis Conneau, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. Word translation without parallel data. In International Conference on Learning Representations.

Anne Lauscher, Vinit Ravishankar, Ivan Vulic, and´ Goran Glavaš. 2020. From zero to hero: On the limitations of zero-shot language transfer with multilingual Transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4483–4499, Online. Association for Computational Linguistics.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems, volume 36, pages 41451–41530. Curran Associates, Inc.

Xun Liang, Hanyu Wang, Yezhaohui Wang, Shichao Song, Jiawei Yang, Simin Niu, Jie Hu, Dan Liu, Shunyu Yao, Feiyu Xiong, and Zhiyu Li. 2024. Controllable text generation for large language models: A survey. Preprint, arXiv:2408.12599.

Jindˇrich Libovický, Rudolf Rosa, and Alexander Fraser. 2020. On the language neutrality of pre-trained multilingual representations. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 1663–1674, Online. Association for Computational Linguistics.

Bing Liu. 2012. Sentiment Analysis and Opinion Mining, 1 edition. Synthesis Lectures on Human Language Technologies. Springer Cham.

Sheng Liu, Haotian Ye, Lei Xing, and James Zou. 2024a. In-context vectors: making in context learning more effective and controllable through latent space steering. In Proceedings ofthe 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Zhu Liu, Cunliang Kong, Ying Liu, and Maosong Sun. 2024b. Fantastic semantics and where to find them: Investigating which layers of generative LLMs reflect lexical semantics. In Findings of the Association for Computational Linguistics: ACL 2024, pages 14551–14558, Bangkok, Thailand. Association for Computational Linguistics.

Samuel Marks and Max Tegmark. 2024. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In First Conference on Language Modeling.

Saif Mohammad. 2018. Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 174–184, Melbourne, Australia. Association for Computational Linguistics.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, and 262 others. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2024. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996–5001, Florence, Italy. Association for Computational Linguistics.

Shrimai Prabhumoye, Alan W Black, and Ruslan Salakhutdinov. 2020. Exploring controllable text generation techniques. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1–14, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Abhilasha Ravichander, Yonatan Belinkov, and Eduard Hovy. 2021. Probing the probing paradigm: Does probing accuracy entail task relevance? In Proceedings of the 16th Conference of the European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 3363–3377, Online. Association for Computational Linguistics.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Deniss Ruder, Andero Uusberg, and Kairit Sirts. 2025. Assessing the reliability and validity of GPT-4 in annotating emotion appraisal ratings. In Proceedings of the 10th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2025), pages 1–11, Albuquerque, New Mexico. Association for Computational Linguistics.

Sonal Sannigrahi and Jesse Read. 2022. Isomorphic cross-lingual embeddings for low-resource languages. Preprint, arXiv:2203.14632.

Naomi Saphra and Sarah Wiegreffe. 2024. Mechanistic? In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networksfor NLP, pages 480–498, Miami, Florida, US. Association for Computational Linguistics.

Maite Taboada, Julian Brooke, Milan Tofiloski, Kimberly Voll, and Manfred Stede. 2011. Lexicon-based methods for sentiment analysis. Computational Linguistics, 37(2):267–307.

Daniel Tan, David Chanin, Aengus Lynch, Dimitrios Kanoulas, Brooks Paige, Adria Garriga-Alonso, and Robert Kirk. 2025. Analyzing the generalization and reliability of steering vectors. Preprint, arXiv:2407.12404.

Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, and Ji-Rong Wen. 2024. Language-specific neurons: The key to multilingual capabilities in large language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5701–5715, Bangkok, Thailand. Association for Computational Linguistics.

Curt Tigges, Oskar J. Hollinsworth, Atticus Geiger, and Neel Nanda. 2024. Language models linearly represent sentiment. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networksfor NLP, pages 58–87, Miami, Florida, US. Association for Computational Linguistics.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2024. Steering language models with activation engineering. Preprint, arXiv:2308.10248.

Tianlong Wang, Xianfeng Jiao, Yinghao Zhu, Zhongzhi Chen, Yifan He, Xu Chu, Junyi Gao, Yasha Wang, and Liantao Ma. 2025. Adaptive activation steering: A tuning-free llm truthfulness improvement method for diverse hallucinations categories. In Proceedings of the ACM on Web Conference 2025, WWW ’25, page 2562–2578, New York, NY, USA. Association for Computing Machinery.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Yuemei Xu, Ling Hu, Jiayi Zhao, Zihan Qiu, Kexin Xu, Yuqi Ye, and Hanwen Gu. 2025. A survey on multilingual large language models: corpora, alignment, and bias. Frontiers of Computer Science, 19(11):1911362.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2025. Representation engineering: A top-down approach to ai transparency. Preprint, arXiv:2310.01405.

## A Experimental Details

## A.1 Model and Implementation

All models (Llama-3.2-3B, Gemma-3-4B-PT, and Qwen3-4B-Base) were implemented using the transformers library (Wolf et al., 2020) in bfloat16 precision. We specifically utilize the base (pre-trained) versions to investigate sentiment representations in the foundational latent space, thereby avoiding biases from instructiontuning or RLHF. Because all evaluated models are in the $\mathrm { \sim 3 B }$ to 4B parameter range, they require approximately 16 GB of VRAM for inference and activation steering. We distributed the experiments across several computing environments based on resource availability: publicly available cloud notebooks (dual NVIDIA T4, $2 \times 1 6 \mathbf { G B } )$ , commercial cloud GPU instances (NVIDIA RTX A4500, 20 GB), and a shared academic compute cluster (NVIDIA DGX-A100).

## A.2 Data and Lexicon Preprocessing

Lexicon Settings. As discussed in Section 3.1, we specifically utilize the multilingual NRC-VAD lexicon (Mohammad, 2018) to obtain the sentiment direction. The NRC-VAD lexicon provides human ratings for valence, arousal, and dominance, with scores ranging from 0 to 1. In our experiments, we investigate the efficacy of cross-lingual steering by comparing two distinct lexicon configurations: monolingual and multilingual.

For the monolingual configuration, we rely exclusively on English lexicon entries. The preprocessing involves removing invalid entries (NaN values) and filtering the remaining entries based on their valence scores. Entries with a valence score $> 0 . 8$ are assigned to the positive set $\mathcal { L } ^ { + }$ , while entries with a valence score $< 0 . 2$ are assigned to the negative set $\mathcal { L } ^ { - }$ . We deliberately employ these strict thresholds to maximize the semantic contrast between the opposing sets, ensuring that the resulting steering direction is robust and not diluted by neutral or weakly polarized terms.

For the multilingual configuration, we aggregate lexicon entries from 11 languages, selected to cover a spectrum from high-resource to low-resource languages: English (en), German (de), Chinese (zh), Japanese (ja), Indonesian (id), Icelandic (is), Estonian (et), Basque (eu), Hausa (ha), Javanese (jv), and Sundanese (su). The preprocessing for the aggregated set involves a rigorous deduplication process. First, we identify and remove ambiguous duplicates: entries appearing in multiple languages that exhibit conflicting valence scores (i.e., one instance having a score $< 0 . 2$ and another $> 0 . 8 )$ Second, for duplicates with consistent sentiment polarity (both $< 0 . 2$ or both $> 0 . 8 )$ , we retain only the first occurrence. From this refined set of unique multilingual entries, we construct $\mathcal { L } ^ { + }$ and $\mathcal { L } ^ { - }$ using the same valence thresholds $( > 0 . 8$ and $< 0 . 2 )$ as the monolingual configuration.

Dataset Construction and Validation We constructed a multilingual evaluation suite consisting of 1000 neutral prompts across 11 languages. We utilized Gemini to generate 10 categories and 100 sub-topics, which served as the foundation for creating 1,000 neutral evaluation sentences. The full list of categories and prompts is provided in Appendix B. These were subsequently translated into 10 target languages, with specialized prompts for low-resource languages (Hausa, Javanese, Sundanese) to ensure linguistic naturalness. To ensure the neutrality of the baseline, we manually audited a random sample of 100 prompts. While the majority were neutral, we observed that approximately 3% contained implicit sentiment (e.g.,“The departure board showed the flight was ‘delayed”’ or “A client reported an issue with the software”). These instances highlight that while Gemini generates objectively factual statements, certain real-world contexts carry inherent negative or positive connotations.

<table><tr><td>Lang. ISO Llama-3.2 Qwen3-4B Gemma3-4B</td></tr><tr><td>English en</td><td>8.47 8.53</td></tr><tr><td>German de</td><td>12.39 12.42</td></tr><tr><td>Chinese</td><td>10.80 zh 9.75 7.62 8.03</td></tr><tr><td>Japanese ja 13.41</td><td>12.43 9.84</td></tr><tr><td></td><td>12.61 9.41</td></tr><tr><td>Indonesian id 12.51 Icelandic is 17.42</td><td>17.62 15.14</td></tr><tr><td>Estonian et 13.33</td><td>13.44 11.78</td></tr><tr><td>Basque eu 15.44</td><td>15.61 13.53</td></tr><tr><td>Hausa ha</td><td>15.51 15.22</td></tr><tr><td>14.35</td><td>13.68 14.49 12.64</td></tr><tr><td>Javanese jv Sundanese su</td><td>15.02 15.16 12.54</td></tr></table>

Table 3: Average tokens per neutral prompt across target languages. Lower values indicate higher tokenization efficiency and a more optimized vocabulary for that language.

## A.3 Steering Methodology Details

Linear Probing. For layer selection, we split the activation dataset into training (70%) and validation (30%) sets using stratified sampling with a fixed random state (1331). Probes were implemented as Logistic Regression models via scikit-learn, utilizing a StandardScaler and a liblinear solver with L1 regularization (C=1.0) initialized with the same fixed seed. Importantly, L1 regularization in this layer-probing stage is employed solely to mitigate overfitting in the highdimensional activation space and improve probe generalization. The actual isolation of sparse neurons is handled exclusively by the non-linear XGBoost attribution.

Neuron Selection. For each layer, neuron-level attribution uses an XGBoost model (Chen and Guestrin, 2016) with a binary:logistic objective, trained on all lexicon entries of D<sup>(i)</sup> for 100 boosting rounds with learning rate η = 0.1, maximum tree depth 3, and random seed 1331; all other hyperparameters are left at their library defaults (including λ = 1 and γ = 0). Neuron importance is computed with XGBoost’s gain importance, i.e., the average split gain over all splits that use the neuron. We retain only the neurons with non-zero gain, which yields a sparse set of neurons for steering.

## A.4 Generation and Hyperparameters

For generation, we set a constraint of at most 128 new tokens. To ensure highly stable and reproducible sampling trajectories despite the stochastic nature of the decoding process, all generation runs were executed with a fixed global random seed (1331) initialized across Python, NumPy, and PyTorch. The specific sampling parameters were:

• Llama: Temperature $0 . 6 , p = 0 . 9 .$

• Qwen: Temperature $0 . 6 , k = 2 0 , p = 0 . 9 5 .$

• Gemma: Default settings $( k ~ = ~ 6 4 , p ~ =$ 0.95).

The steering intensity α was tuned via grid search on a 50-prompt tuning split. Based on the highest Success Rate (our primary metric), we selected α = 0.5 for monolingual steering across all models. For multilingual steering, we used $\alpha = 0 . 7 5$ for Gemma and Qwen, and α = 1.0 for Llama.

## A.5 Validation of LLM-as-a-Judge for Multilingual Sentiment

Evaluating sentiment shifts in open-ended generative text across typologically diverse languages requires a metric capable of capturing complex, context-dependent semantics. While traditional lexicon-based scoring approaches offer a highly interpretable alternative (Taboada et al., 2011; Liu, 2012), they present severe structural limitations for our specific experimental setup. Our evaluation suite spans 11 languages, including highly agglutinative systems (e.g., Basque, Japanese, Indonesian, Javanese, Sundanese) and morphologically fusional languages (e.g., Estonian, Icelandic). In these typologies, sentiment-bearing roots undergo extensive affixation and internal stem mutations, causing static dictionary engines to suffer from catastrophic surface-form miss rates without flawless, language-specific lemmatizers.

Furthermore, lexicon approaches are inherently compositionally blind; they aggregate static token weights and struggle to resolve contextual polysemy or implicit sentiment distributed across narrative structures (Khoo and Johnkhan, 2018). To overcome these limitations, we utilize an LLMbased evaluator (GPT-4.1), which natively resolves morphological variants via sub-word tokenization and captures deep contextual dependencies.

To empirically assess the reliability of our evaluation metric and address broader cross-lingual limitations noted in recent literature (Xu et al., 2025), we manually audited the GPT-4.1 judge across both high-resource (English) and low-resource (Javanese) settings. In both cases, 150 generated instances were sampled (50 per predicted sentiment class), fully randomized, and evaluated with the human annotator blinded to the model’s predictions. The English baseline was audited by one of the authors, whereas the Javanese evaluation was conducted by a volunteer native speaker recruited from the authors’ network to ensure a rigorous assessment of low-resource linguistic nuances. To ensure a fair comparison, the human annotator evaluated the text using criteria identical to the LLM judge (detailed in Appendix C). The annotator provided verbal consent for their labels to be used for research validation; no personal or demographic data was collected.

The detailed confusion matrices (Table 4) provide deeper insight into the model’s error distribution across resource tiers. While there is an expected performance drop in the low-resource Javanese setting (achieving a macro-average precision of 78%), the matrix reveals that errors are predominantly boundary confusions between polar and neutral classes rather than severe polar inversions. For instance, the judge only predicted a positive sentiment for a true negative sample in 7 out of 150 instances, maintaining excellent directional precision for negative sentiment shifts (92%). These empirical findings confirm that despite the structural complexities of low-resource languages, the contextual LLM judge provides a highly reliable metric for capturing directional sentiment shifts.

<table><tr><td colspan="3">English (High-Resource)</td></tr><tr><td>Predicted</td><td>Human Annotation (Gold) Negative Positive Neutral</td><td>Precision</td></tr><tr><td>Negative</td><td>47 0</td><td>3 94%</td></tr><tr><td>Positive</td><td>0 50 0</td><td>100%</td></tr><tr><td>Neutral</td><td>1 1 48</td><td>96%</td></tr><tr><td colspan="3">Macro-Average 96.7%</td></tr><tr><td colspan="3">Javanese (Low-Resource)</td></tr><tr><td>Predicted</td><td>Human Annotation (Gold) Negative Positive Neutral</td><td>Precision</td></tr><tr><td>Negative</td><td>46 1</td><td>92%</td></tr><tr><td>Positive</td><td>7 36</td><td>3 7</td><td>72%</td></tr><tr><td>Neutral</td><td>9 6</td><td>35</td><td>70%</td></tr><tr><td colspan="3"></td><td>Macro-Average 78.0%</td></tr></table>

Table 4: Confusion matrices for the GPT-4.1 sentiment judge across 150 randomized samples per language.

## B Prompts and Dataset Topics

## B.1 Prompts for Data Generation

We utilized specific prompts to generate the categories, topics, and neutral sentences. For lowresource languages (Hausa, Javanese, Sundanese), a specialized translation prompt was used to ensure stylistic accuracy.

## 1. Category and Topic Generation Prompt

I would like to conduct an experiment to examine whether a Large Language Model (LLM) can be steered to generate positive or negative continuations from neutral prompts. Please suggest 10 suitable topic categories, each containing 10 broad subcategories that can serve as a basis for constructing neutral prompts. I plan to create 10 prompts for each subcategory; therefore, the subcategories should remain general and widely applicable rather than overly specific. For reference, I prefer the subcategories to be phrased in a style similar to the examples below:

## • Preferred style (broad and illustrative):

1. Morning routine (waking up, breakfast, commuting)

2. House chores (cleaning, cooking, laundry)

3. Daily commute (bus, train, traffic)

4. Leisure activities (reading, walking, hobbies)

5. Communication (texting, calling, chatting)

6. Time management (schedules, delays, appointments)

7. Weather and environment (rain, heat, wind)

8. Personal interactions (neighbors, family, friends)

9. Weekend activities (shopping, rest, recreation)

10. Unexpected daily events (forgotten items, late arrivals)

## • Less preferred style (too narrow or fragmented):

1. Morning routines

2. Commuting (bus, train, car)

3. Household chores (e.g., laundry, dishes)

4. Running errands

5. Grocery shopping

6. The weather forecast

7. Receiving mail or a package

8. Walking in the neighborhood

9. Preparing a simple meal

10. The end of the day

## 2. Neutral Sentence Generation Prompt

I would like to conduct an experiment to examine whether a Large Language Model (LLM) can be steered to generate positive or negative continuations from neutral prompts.

To prepare the dataset, I need your help in creating these neutral prompts. I already have a selection of topics from the [category] category, consisting of the following topics: [topics]

Please generate [amount] neutral prompts for each topic. First, write each prompt in English, then translate it into [languages].

The output should be formatted in JSONL with a unique prompt\_id, category, topic, and language codes, as shown below. The prompt\_id should start from 1 and increment sequentially.

```jsonl
{“prompt_id": 1, “category": “morning routine",
“topic": “waking up", “en": “The alarm clock
went off at 6 AM.", “id": “Jam weker berbunyi
pukul 6 pagi.", “de": “Der Wecker klingelte um
6 Uhr morgens.", “zh": “早上6点，闹钟响了。"}
```

Note: The topic details (i.e., explanations in parentheses) do not need to be included in the JSONL output.

## 3. Low-Resource Translation Prompt

I am building a parallel dataset for low-resource languages. Below is a list of JSON objects containing English sentences (“en").

Please update each JSON object by adding translations for the following keys:

1. “ha": Hausa (Standard).

2. “jv": Javanese (Style: Ngoko Lugu/Casual. Strictly NO Krama/Formal language).

3. “su": Sundanese (Style: Basa Loma/Familiar. Use daily conversation style, comparable to Javanese Ngoko).

Constraints:

• Output ONLY the final valid JSONL format.

• Do not add any conversational text before or after the data.

• Please enclose the final output in a single code block.

Input Data:

[JSONL dataset containing English neutral sentences]

## B.2 List of Categories and Topics

Below is the complete list of 100 topics categorized by domain. Topics marked with an asterisk (\*) were also selected for the preliminary tuning dataset (50 sentences).

## 1. Work & Professional Life

Starting the workday\*, Team collaboration\*, Managing tasks\*, Workplace communication, Taking a break, Meetings, Skill development, Performance feedback, Problemsolving\*, Ending the workday\*.

## 2. Home & Household

Morning routine\*, Household chores\*, Meal preparation\*, Home maintenance, Running errands, Family interactions, Managing finances\*, Pet care, Receiving mail, Evening routine\*.

## 3. Social & Relationships

Making plans\*, Social gatherings\*, Casual conversation\*, Community involvement, Meeting new people, Family events\*, Online social media, Maintaining contact, Discussing topics, Public interactions\*.

## 4. Travel & Transport

Daily commute\*, Planning a trip\*, Air travel, Using public transport, Accommodation, Navigating a new place\*, Eating out\*, Sightseeing, Unexpected travel events\*, Returning home.

## 5. Health & Wellness

Physical exercise\*, Diet and nutrition\*, Sleep patterns\*, Medical appointments\*, Mental well-being\*, Personal hygiene, Feeling unwell, Taking medication, Rest and recovery, Health goals.

## 6. Education & Learning

Attending a class\*, Studying\*, Group projects, Taking an exam\*, Doing homework, Interacting with instructors\*, Using resources, Learning a new skill\*, Class registration, School environment.

## 7. Technology & Digital Life

Using a smartphone, Working on a computer\*, Email management, Software updates, Troubleshooting\*, Online shopping, Digital entertainment\*, Data management, Digital security\*, Video calls\*.

## 8. Hobbies & Leisure

Creative activities\*, Outdoor pursuits\*, Consuming media\*, Games\*, Attending events, Cooking and baking, Crafts, Following sports, Simple relaxation, Weekend plans\*.

## 9. Commerce & Finance

Grocery shopping\*, Retail shopping, Customer service\*, Receiving a service, Public services, Comparing products\*, Making a large purchase, Subscription services\*, Banking tasks, Following a budget\*.

## 10. Environment & Surroundings

The weather\*, Changing seasons, Environmental noise\*, Public spaces\*, Recycling and waste\*, Weather forecast, Natural events, Indoor environment, Local wildlife, The physical landscape\*.

## C Sentiment Evaluation Guidelines

To ensure a rigorous and fair comparison between the automated metric and human evaluation, both the GPT-4.1 judge and the human annotator evaluated the generated texts using identical task descriptions and sentiment definitions.

## C.1 LLM Judge Prompt

We used GPT-4.1 to evaluate the sentiment of the generated texts. The model was provided with a system instruction to act as a precise classifier and a user prompt containing detailed definitions and constraints. The full prompt structure is provided below:

Sentiment Evaluation Prompt Template   
System Message:   
You are a precise sentiment analysis classifier.   
User Message:   
You are an expert linguist acting as a   
sentiment judge.   
The user will provide a text that may be in   
{language} or a mix of languages.   
Task:   
Analyze the text and classify its dominant   
sentiment based on the definitions below.   
If the text contains multiple parts (e.g.,   
instructions, questions, and narrative), focus   
on the sentiment conveyed by the main narrative   
content or the generated statement, ignoring   
purely instructional phrasing (like “Answer:”,   
“Choose A/B”, etc.).   
Definitions:   
“Positive”: The text expresses a positive   
evaluation, such as indicating that the   
situation, action, or object is good, helpful,   
safe, beneficial, desirable, comfortable,   
satisfying, or otherwise favorable. The   
evaluation may be explicit (e.g., “I am happy”)   
or implicit (e.g., “the process is efficient”).   
- “Negative”: The text expresses a negative   
evaluation, such as indicating that the   
situation, action, or object is bad, difficult,   
harmful, inconvenient, frustrating,   
disappointing, risky, undesirable, or otherwise   
unfavorable. The evaluation may be explicit   
(e.g., “I am angry”) or implicit (e.g., “the   
system is confusing”).   
“Neutral”: The text is purely factual,   
informative, descriptive, or consists of   
neutral questions/instructions without   
emotional valence.   
Constraint:   
The answer must be exactly one word from this   
list: [Positive, Negative, Neutral]   
Text to analyze:   
{sentence}   
Answer:

## C.2 Human Annotator Instructions

The native Javanese human annotator was provided with the exact same task description and definitions as the LLM judge. The only modifications were the removal of system framing (e.g., “You are a precise sentiment analysis classifier”) and the substitution of the output constraint with spreadsheet-specific data entry instructions. The provided instructions are detailed below:

Human Annotator Instructions   
Task:   
Analyze the text and classify its dominant   
sentiment based on the definitions below.   
If the text contains multiple parts (e.g.,   
instructions, questions, and narrative), focus   
on the sentiment conveyed by the main narrative   
content or the generated statement, ignoring   
purely instructional phrasing (like “Answer:”,   
“Choose A/B”, etc.).   
Definitions:   
- “Positive”: The text expresses a positive   
evaluation, such as indicating that the   
situation, action, or object is good, helpful,   
safe, beneficial, desirable, comfortable,   
satisfying, or otherwise favorable. The   
evaluation may be explicit (e.g., “I am happy”)   
or implicit (e.g., “the process is efficient”).   
- “Negative”: The text expresses a negative   
evaluation, such as indicating that the   
situation, action, or object is bad, difficult,   
harmful, inconvenient, frustrating,   
disappointing, risky, undesirable, or otherwise   
unfavorable. The evaluation may be explicit   
(e.g., “I am angry”) or implicit (e.g., “the   
system is confusing”).   
- “Neutral”: The text is purely factual,   
informative, descriptive, or consists of   
neutral questions/instructions without   
emotional valence.   
Note:   
Select the dropdown on the “Judgement” column   
(rightside of the “Generated Text” column) to   
put your judgment.   
Thank you.

## D Additional Sentiment Distribution Plots

In this section, we provide detailed sentiment distribution plots that complement the analysis in Section 5.1. First, we present the multilingual steering results for Qwen-3 and Llama-3.2. Second, we provide a comparison against a monolingual (English-only) steering baseline for Gemma-3.

## D.1 Multilingual Steering Impact on Sentiment Distribution Across Models

Our results demonstrate that multilingual steering directions consistently influence the sentiment distribution across all three models, shifting the probability mass toward the target sentiment compared to the baseline. However, the magnitude of this effect varies significantly. Gemma-3 exhibits the highest influence, where the intervention almost completely overrides the default neutral behavior (see Figure 2). In contrast, Llama-3.2 and Qwen-3 show a more moderate response, often retaining a significant portion of neutral outputs even after steering.

## D.2 Monolingual (English) Steering Impact on Sentiment Distribution

To better understand the distinct contribution of language-specific lexicons, Figure 7 visualizes the sentiment distribution shifts for Gemma-3 using steering directions derived solely from English lexicons. The monolingual directions prove highly effective for high-resource and some mid-resource languages (e.g., German, Indonesian), occasionally matching or even slightly exceeding the multilingual approach, suggesting a strong inherent crosslingual alignment within Gemma-3. However, in specific mid- to low-resource settings (such as Icelandic, Basque, and Sundanese), the English vectors exhibit steeper performance drops and struggle more to overcome the baseline neutral anchor. While the multilingual intervention (Figure 2) does not completely eliminate this neutral tendency, it demonstrably mitigates the deterioration, providing a more robust and resilient steering signal for the most resource-constrained languages.

![](images/9eb5a43140e130b467777c9dcfadbce115f4f875625769e91f40cb15995ee3a4.jpg)  
Figure 5: Impact of Steering on Sentiment Distribution (Qwen-3). Steering induces a noticeable shift toward the target sentiment, confirming the method’s cross-model applicability. However, Qwen-3 exhibits a stronger tendency to preserve neutral tonality compared to other models, with neutral outputs remaining the dominant class in mid to low resource languages even after intervention.

![](images/5951d1c5100f37529d9664e80645962aa146e2c3c19cc4d68bf3f0cb29383089.jpg)  
Figure 6: Impact of Steering on Sentiment Distribution (Llama-3.2). While the steering vectors successfully increase the probability of the target sentiment across all languages, Llama-3.2 retains a higher proportion of neutral outputs compared to Gemma-3, particularly in low-resource languages (e.g., Hausa, Javanese, Sundanese).

## E Complete Steering Results with SA and LIDR

Tables 5 and 6 illustrate the comparative strengths of Gemma-3, Qwen-3, and Llama-3.2. Notably, Gemma-3 maintains significantly higher Success Rates (SR) and Sentiment Accuracy (SA) in lowresource categories (e.g., Hausa, Javanese) compared to its counterparts in both monolingual and multilingual environments.

## F Statistical Significance and Decoding Variance

In this section, we provide a rigorous statistical analysis of our evaluation results to confirm that the observed performance deltas between multilingual and monolingual steering directions are statistically significant and not artifacts of stochastic decoding variance. We also provide a granular analysis of model behavior in low-resource regimes.

## F.1 Stochastic Decoding and Corpus-Level Stability

To evaluate the impact of our steering interventions, our experimental pipeline utilizes a massive sample footprint. For each language condition, we conduct N = 2, 000 independent generation trials (derived from 1, 000 distinct prompts × 2 target sentiments).

![](images/dc5bd8a705e5a7147bbdcd8b0ad06a70cb1470ef678124e1ce94d53588d874d8.jpg)

Figure 7: Impact of Steering with Monolingual Steering Direction on Sentiment Distribution (Gemma-3). The distribution of generated sentiments across 11 languages under three conditions: (a) No Intervention (as baseline), (b) Positive Steering, and (c) Negative Steering
<table><tr><td></td><td></td><td colspan="4">Gemma-3</td><td colspan="4">Qwen-3</td><td colspan="4">Llama-3</td></tr><tr><td>Lang</td><td>Res.</td><td>SR</td><td>SA</td><td>LIDR</td><td>Config</td><td>SR</td><td>SA</td><td>LIDR</td><td>Config</td><td>SR</td><td>SA</td><td>LIDR</td><td>Config</td></tr><tr><td>English (en)</td><td>High</td><td>95.0</td><td>95.0</td><td>99.8</td><td>all</td><td>44.5</td><td>54.6</td><td>84.1</td><td>all</td><td>66.4</td><td>67.0</td><td>98.6</td><td>all</td></tr><tr><td>German (de)</td><td>High</td><td>90.7</td><td>91.8</td><td>98.4</td><td>all</td><td>45.4</td><td>52.8</td><td>72.1</td><td>top-50%</td><td>53.8</td><td>64.2</td><td>83.0</td><td>all</td></tr><tr><td>Chinese (zh)</td><td>High</td><td>86.8</td><td>91.0</td><td>93.5</td><td>all</td><td>53.4</td><td>62.0</td><td>82.5</td><td>all</td><td>34.3</td><td>44.5</td><td>67.5</td><td>all</td></tr><tr><td>Japanese (ja)</td><td>High</td><td>80.2</td><td>89.3</td><td>88.6</td><td>all</td><td>23.9</td><td>31.3</td><td>57.5</td><td>top-50%</td><td>20.8</td><td>38.1</td><td>58.2</td><td>all</td></tr><tr><td>Indonesian (id)</td><td>Mid</td><td>82.3</td><td>91.0</td><td>89.8</td><td>all</td><td>42.9</td><td>47.9</td><td>85.1</td><td>top-50%</td><td></td><td>38.651.3</td><td>69.1</td><td>all</td></tr><tr><td>Icelandic (is)</td><td>Mid</td><td>62.3</td><td>63.8</td><td>97.6</td><td>top-50%</td><td>12.1</td><td>13.2</td><td>95.5</td><td>top-50%</td><td>20.0</td><td>24.1</td><td>89.6</td><td>all</td></tr><tr><td>Estonian (et)</td><td>Mid</td><td>78.0</td><td>80.1</td><td>97.5</td><td>top-50%</td><td>19.2</td><td>22.3</td><td>87.1</td><td>top-50%</td><td>22.1</td><td>29.5</td><td>80.2</td><td>top-50%</td></tr><tr><td>Basque (eu)</td><td>Mid</td><td>54.7</td><td>56.0</td><td>96.0</td><td>top-50%</td><td>9.2</td><td>10.2</td><td>92.0</td><td>top-50%</td><td>19.6</td><td>26.6</td><td>79.0</td><td>all</td></tr><tr><td>Hausa (ha)</td><td>Low</td><td>54.6</td><td>60.8</td><td>90.0</td><td>top-50%</td><td>1.8</td><td>5.1</td><td>61.8</td><td>top-5</td><td>9.3</td><td>19.8</td><td>74.0</td><td>top-50%</td></tr><tr><td>Javanese (jv)</td><td>Low</td><td>37.8</td><td>62.5</td><td>66.0</td><td>top-50%</td><td>10.2</td><td>30.9</td><td>49.1</td><td>top-50%</td><td>4.7</td><td>34.0</td><td>29.9</td><td>all</td></tr><tr><td>Sundanese (su)</td><td>Low</td><td>25.6</td><td>56.8</td><td>50.3</td><td>top-50%</td><td>4.7</td><td>24.8</td><td>44.8</td><td>top-50%</td><td>7.5</td><td>29.1</td><td>46.6</td><td>all</td></tr></table>

Table 5: Best Results (Mono). SR: Success Rate, SA: Sentiment Acc, LIDR: Language ID Rate. All metrics in %.

While decoding parameters remain stochastic, evaluating across thousands of distinct, independent generation episodes leverages the Law of Large Numbers. Local, token-level variance effectively averages out at the corpus level. This stabilizes the observed success rates (SR), allowing them to closely converge toward the true expected performance of the steered model and minimizing the impact of seed-to-seed variance.

## F.2 Macro- and Micro-Level Statistical Significance

To verify both the global and localized validity of our approach, we conducted Two-Sample Proportion Z-Tests to ensure that the observed performance deltas between multilingual and monolingual steering directions are mathematically meaningful, rather than products of stochastic decoding variance.

Macro-Level Aggregation. First, we aggregated the steering success rates by language resource level across all target models. Because each language tier aggregates multiple balanced conditions, the sample sizes provide immense statistical power (n = 8,000 trials for High/Mid tiers; n = 6,000 trials for the Low tier). Table 7 contrasts the Multilingual steering direction against the Monolingual (English) baseline, reporting the success rate for each condition, their difference (∆ SR, in percentage points), and the corresponding two-tailed test statistics.

The aggregate data demonstrate a distinct performance pattern: as language resource tiers drop from High to Mid/Low, the multilingual lexicon yields regular and mathematically meaningful advantages. We reject the null hypothesis across the board for the Mid-resource tier, and for Gemma-3 in the Low-resource tier, where the base model possesses sufficient capacity to respond to crosslingual steering. Notably, the sole significant result favoring the monolingual baseline occurs for Gemma-3 in the High-resource tier (∆ SR = −1.22 pp, p = 0.0194). We interpret this reversal as consistent with our broader findings on architectural dynamics: where monolingual representations are already highly optimized, the multilingual lexicon direction introduces slight multilingual “noise” rather than a reinforcement signal, resulting in a marginal performance penalty.

<table><tr><td rowspan="2">Lang</td><td rowspan="2">Res.</td><td rowspan="2">SR</td><td colspan="3">Gemma-3</td><td rowspan="2"></td><td colspan="3">Qwen-3</td><td rowspan="2"></td><td colspan="3">Llama-3</td><td rowspan="2">Config</td></tr><tr><td>SA</td><td>LIDR</td><td>Config</td><td>SR</td><td>SA LIDR</td><td>Config</td><td>SR SA</td><td>LIDR</td><td></td></tr><tr><td>English (en)</td><td>High</td><td>94.3</td><td>94.4</td><td>99.7</td><td>all</td><td>45.1</td><td>48.5</td><td>91.6</td><td>all</td><td>65.8</td><td>66.2</td><td>98.9</td><td></td><td>all</td></tr><tr><td>German (de)</td><td>High</td><td>90.5</td><td>91.4</td><td>98.5</td><td>all</td><td></td><td>46.1 51.1</td><td>78.8</td><td></td><td>top-50%</td><td>53.4</td><td>63.4</td><td>82.0</td><td>all</td></tr><tr><td>Chinese (zh)</td><td>High</td><td>86.2</td><td>89.0</td><td>95.0</td><td>all</td><td>48.0</td><td>54.2</td><td>85.8</td><td>all</td><td></td><td>33.1</td><td>42.8</td><td>71.5</td><td>all</td></tr><tr><td>Japanese (ja)</td><td>High</td><td>76.8</td><td>85.9</td><td>87.8</td><td>all</td><td>23.5</td><td>43.2</td><td>43.3</td><td></td><td>all</td><td>18.9</td><td>33.2</td><td>60.1</td><td>all</td></tr><tr><td>Indonesian (id)</td><td>Mid</td><td>82.0</td><td>91.0</td><td>88.9</td><td>all</td><td>46.8</td><td>53.2</td><td>83.2</td><td></td><td>all</td><td>37.0</td><td>49.9</td><td>69.8</td><td>all</td></tr><tr><td>Icelandic (is)</td><td>Mid</td><td>66.8</td><td>68.8</td><td>96.6</td><td>all</td><td>10.9</td><td>12.4</td><td>94.3</td><td></td><td>all</td><td>23.1</td><td>26.5</td><td>91.6</td><td>top-50%</td></tr><tr><td>Estonian (et)</td><td>Mid</td><td>77.8</td><td>80.2</td><td>96.8</td><td>all</td><td>23.5</td><td>25.9</td><td>90.8</td><td></td><td>top-50%</td><td>25.4</td><td>33.7</td><td>78.8</td><td>top-50%</td></tr><tr><td>Basque (eu)</td><td>Mid</td><td>60.7</td><td>61.9</td><td>95.2</td><td>all</td><td>12.2</td><td>13.3</td><td>91.3</td><td></td><td>all</td><td>20.4</td><td>26.5</td><td>84.7</td><td>top-50%</td></tr><tr><td>Hausa (ha)</td><td>Low</td><td>58.9</td><td>64.4</td><td>90.1</td><td>top-50%</td><td>2.1</td><td>4.9</td><td>60.9</td><td></td><td>top-5</td><td>9.6</td><td>16.2</td><td>75.7</td><td>top-5</td></tr><tr><td>Javanese (jv)</td><td>Low</td><td>37.9</td><td>62.7</td><td>64.5</td><td>top-50%</td><td>9.8</td><td>28.6</td><td>52.3</td><td></td><td>top-50%</td><td>4.2</td><td>31.2</td><td>33.5</td><td>top-50%</td></tr><tr><td>Sundanese (su)</td><td>Low</td><td>27.2</td><td>63.7</td><td>46.1</td><td>all</td><td>5.7</td><td>24.3</td><td>48.0</td><td></td><td>top-50%</td><td>5.9</td><td>25.9</td><td>47.9</td><td>top-50%</td></tr></table>

Table 6: Best Results (Multi). SR: Success Rate, SA: Sentiment Acc, LIDR: Language ID Rate. All metrics in %.

<table><tr><td>Model Multi SR Mono SR ∆ SR Z-score p-value</td></tr><tr><td>High-resource (4 langs, n = 8,000)</td></tr><tr><td>Gemma-3 86.95 88.17-1.22 -2.34 0.0194 Qwen-3 40.67 41.80 -1.13 -1.45 0.1466</td></tr><tr><td>Llama-3.2 42.80 43.83 -1.03 -1.31 0.1886</td></tr><tr><td>Mid-resource (4 langs, n = 8,000)</td></tr><tr><td>Gemma-3 71.83 69.33+2.50 +3.47 0.0005 Qwen-3 23.35 20.85+2.50 +3.81 0.0001</td></tr><tr><td>Llama-3.2 26.48 25.08 +1.40 +2.02 0.0429</td></tr><tr><td>Low-resource (3 1langs, n = 6,000)</td></tr><tr><td>Gemma-3 41.33 39.33+2.00 +2.23 0.0255</td></tr><tr><td>Qwen-3 5.87 5.57 +0.30 +0.71 0.4791</td></tr><tr><td>Llama-3.2 6.57 7.17-0.60 -1.30 0.1938</td></tr></table>

Table 7: Two-sample proportion Z-tests comparing multilingual vs. monolingual (English) steering success rates (SR, %) by language resource tier. Bold p-values indicate rejection of the null hypothesis at α = 0.05. Positive ∆ SR favors the multilingual direction; negative favors the monolingual baseline.

Micro-Level Verification (Case Study: Gemma-3 Hausa). To confirm that the gains observed in viable low-resource models are mathematically sound at the individual language level, we performed a localized two-proportion z-test specifically on the Gemma-3 Hausa results (N = 2, 000 trials per condition):

• Multilingual Baseline Success: 1, 178/2, 000 (58.9%)

• Monolingual Baseline Success: 1, 092/2, 000 (54.6%)

This +4.3 pp margin yields a critical value of z = 2.7447, corresponding to a two-tailed p-value of 0.0060. This establishes that when baseline capacity thresholds are met, the multilingual lexicon’s advantage is substantial and robust, clearing the p < 0.05 threshold by a wide margin, proving that the observed low-resource gains are not merely artifacts of stochastic decoding.

## F.3 The Low-Resource Performance Spectrum in Latent Space

While the statistical significance holds firmly where models exhibit baseline capacity, individual language conditions in extreme low-resource regimes occasionally exhibit marginal absolute performance differences (e.g., +0.1 to +0.3 percentage points). Rather than indicating methodological failure, our analysis shows these variations reflect a clear architectural spectrum of base model capacity:

• The Floor Effect (e.g., Qwen-3 Hausa): At a success rate of just 1.8%, Qwen-3 experiences a foundational downstream alignment failure in Hausa. Crucially, its baseline Language Identification Rate remains high (LIDR > 60%), indicating the model recognizes the linguistic context. However, the underlying semantic features are not sufficiently robust within the latent space to support targeted linear interventions. In this extreme regime, the steering vector lacks a mature geometric representation to act upon, resulting in statistically insignificant variance (e.g., a marginal +0.3 pp delta).

• Neutral Transfer Equivalence (e.g., Gemma-3 Javanese): In contrast to the floor effect, Gemma-3 possesses functional baseline capacity in Javanese, maintaining a stable success rate of roughly 37.8%. The marginal +0.1 pp delta represents a regime of representation equivalence. In this condition, both the cross-lingual multilingual vector and the monolingual vector achieve matching directional utility in the residual stream, resulting in stable, net-neutral transfer.

• Positive Cross-Lingual Transfer (e.g., Gemma-3 Hausa): When a base model crosses the necessary capacity threshold to respond robustly to interventions in a low-resource language, the architectural advantage of the multilingual lexicon becomes highly visible, delivering the statistically significant boost of +4.3 percentage points verified in the previous section.

## G Latent Space Geometry and Steering Norms

To understand the physical “intensity” of steering interventions, we calculate the $L _ { 2 }$ norm of the steering direction vectors at each layer. As shown in Figure 8 and 9, the models exhibit vastly different latent space scales and layer-wise dynamics.

## G.1 Scale and Specialization Ceiling

The absolute magnitude of steering vectors varies by three orders of magnitude between architectures, despite their comparable hidden dimensions (2560 neurons for Gemma-3 and Qwen-3; 3072 for Llama-3.2). Specifically, Gemma-3 operates in a high-variance regime with norms peaking over 10, 000 in its middle layers, whereas Qwen-3 exhibits a moderate scale (peaking near 1, 150) and Llama-3.2 utilizes a highly compressed space with norms remaining below 25. In all models, the norm magnitude is positively correlated with language resource levels and model specialization. The Chinese (zh) steering direction consistently defines the “norm ceiling” in Qwen-3 (∼ 1, 150) and Llama-3.2 (∼ 24), followed closely by English (en). This suggests that the model’s most specialized features are encoded with the highest numerical weights.

The significant disparity in steering effectiveness between models can be partially attributed to this internal scaling of their latent spaces. As illustrated in Figure $^ { 8 , }$ the absolute magnitude of steering vectors in Gemma-3 is several orders of magnitude larger than those in Llama-3.2 and Qwen-3. This suggests a “stretched” or high-variance latent space where semantic features are represented with massive numerical weights. Such high-intensity representations may make specific features easier for the model’s attention mechanisms to prioritize during token generation, explaining its responsiveness to sentiment steering.

Isolating Effective Intervention Magnitude. While Figure 8 illustrates the total latent scale, evaluating the non-zero gain vectors (Figure 9) reveals the concentrated “active” magnitude of the steering interventions. Filtering for non-zero gains reduces the $L _ { 2 }$ norm magnitudes drastically across all architectures: Gemma-3 drops from > 10, 000 to peak values around 500. Similarly, Qwen-3 reduces from ∼ 1, 150 to < 50, and Llama-3.2 drops from ∼ 25 to < 2.5. Despite this massive reduction in absolute magnitude, the hierarchical relationship between the architectures (Gemma > Qwen > Llama) remains strictly preserved, confirming that Gemma-3’s high-variance regime is a fundamental architectural property rather than an artifact of inactive neurons.

![](images/1b4c45bfb1fcac0115c4576e77ec18b2550cd5b40b27b9299e05e3d7647cf981.jpg)

![](images/d9322375eec0e34d44a47c658ddf4964e1c458ece61a0bbd9cf871ec3750459d.jpg)

![](images/fc882873d3d72c350f909e13921c70aa3b50493c5964fcc820a78f0fdd45e2a3.jpg)  
Figure 8: Comparison of $L _ { 2 }$ norm magnitudes for sentiment steering directions (all neurons) across Gemma-3, Llama-3.2, and Qwen-3. Monolingual directions for German, Chinese, Indonesian, and Javanese are extracted and plotted solely to analyze the latent geometry and are not used for injection.

![](images/be85a48c7f9631da5717afd59914fce0309c97e41f59bade7310ed511bd5ea5a.jpg)

![](images/f314581ed746608f075fc170daa62ac0a99f6328cb7842a0684ce32f3677c009.jpg)

![](images/c8964cc7a6ac855aabddee29235b645efdc883dbab2dd3867b25399c9ea9a1be.jpg)  
Figure 9: $L _ { 2 }$ norm magnitude of the nonzero-gain steering vectors across layers for Gemma-3, Llama-3.2, and Qwen-3, comparing English and multilingual steering directions. Only neurons with non-zero XGBoost gain are included in the norm computation, reflecting the actual magnitude of intervention applied at inference time. Opaque markers denote layers in the top 50% by F1 score of the corresponding linear probe; faded markers denote layers outside this top half.

## G.2 The Compactness of Low-Resource Representations

In stark contrast to the high norm ceilings of English and Chinese, the Javanese (jv) direction possesses the smallest norm magnitude across all architectures: below 5 in Llama-3.2 and below 200 in Qwen-3. This suggests that low-resource languages occupy a physically more compact region of the latent space. We hypothesize that the “linguistic collapse” observed at high steering intensities for Javanese occurs because external vectors (often scaled to high-resource norms like English) effectively “over-steer” these compact clusters, pushing activations far beyond their natural boundaries.

## G.3 Layer-wise Dynamics

The models demonstrate distinct temporal profiles for sentiment representation:

• Static Encoding (Llama & Qwen): These models exhibit a “step-function” behavior. In Llama-3.2, the sentiment direction is fully established by Layer 1, and in Qwen-3 by Layer 6, remaining static until the final layers. This implies sentiment is treated as a global, invariant attribute.

• Dynamic Progression (Gemma): Gemma-3 shows a “bell-curve” progression where the norm builds through the initial layers, peaks between Layers 15 and 25, and tapers off toward the output. This indicates a more dynamic semantic refinement process.

While dense vectors exhibit static “stepfunctions” (Llama, Qwen) or “bell-curves” (Gemma) as shown in Figure 8, isolating the non-zero gain vectors reveals a different temporal dynamic. As illustrated in Figure 9, the non-zero gain magnitude steadily increases in the deeper layers across all three models. This indicates that while the global sentiment direction may be established early in the network, the concentrated, manipulable weights required to effectively steer that sentiment accumulate and scale up significantly toward the later layers.

## G.4 Optimal Layer Configuration Asymmetry Between Steering Directions.

Another consequence of this norm disparity is visible in the optimal layer configurations reported in Tables 5 and 6. Across nearly all models and languages, the best-performing monolingual (English) steering direction requires fewer layers (e.g., top-5 or top-50%) to reach its optimal success rate, whereas multilingual directions more often require intervention across all layers. This phenomenon is visually supported by both our dense and sparse norm analyses. As shown in Figure 8, the English steering direction consistently exhibits a substantially larger $L _ { 2 }$ norm than the multilingual direction at comparable layers. This disparity is further corroborated by the non-zero gain vectors (Figure 9), which explicitly map the optimal intervention zones. Within these optimal middle-to-late layers, the concentrated, non-zero magnitude of the English direction (blue) remains distinctly higher than that of the multilingual direction (pink) across all three architectures. Because the magnitude of the injected monolingual perturbation is definitively larger, fewer layers of intervention are needed to displace the model’s activations sufficiently to shift sentiment. Conversely, the multilingual direction, being comparatively dampened in norm, requires broader layer-wise application to achieve its optimal steering efficacy.

Interestingly, this norm-layer relationship does not hold uniformly in Llama-3.2 for several languages (Icelandic, Basque, Hausa, Javanese, Sundanese), where the optimal monolingual configuration in fact requires more layers than the multilingual counterpart. This anomaly may be attributed to the relatively high cross-lingual vector alignment observed in Llama-3.2 between English and lowresource language (see Section H). For instance, the gap between English–Javanese similarity and Multilingual–Javanese similarity is markedly smaller in Llama-3.2 (0.970 − 0.968 = 0.002) than in Gemma-3 (0.987 − 0.983 = 0.004) or Qwen-3 (0.937 − 0.912 = 0.025). Because the English steering direction in Llama-3.2 is already nearly as aligned with low-resource languages like Javanese as the multilingual consensus direction is, monolingual injection is comparatively less destructive than in the other architectures. Consequently, it can tolerate broader layer-wise application without triggering the linguistic collapse observed elsewhere.

## H Cross-Lingual Vector Alignment

![](images/f09e0f041d155e86fdeb6340129aceba807d6e6fe8c5f34e0a5cc8f7a4be0200.jpg)

(a) Gemma-3  
![](images/7366276d894f958aed101ef573b1f1f36e2e120e4a6600020f0c21405d1daad2.jpg)

(b) Qwen-3  
![](images/92ad239487953ba1f51484f9f5424c56e2aa16339a91084317d3f9ddfe6f2346.jpg)  
(c) Llama-3.2  
Figure 10: Cosine similarity heatmaps between monolingual and multilingual steering directions across three architectures. The displayed values represent the average cosine similarity across all layers.

## H.1 The Universal Sentiment Axis

All three models exhibit high cross-lingual alignment, suggesting the existence of a “universal” sentiment axis within their latent spaces.

• Gemma-3 maintains a high “consensus” direction, where the multilingual vector’s similarity to all anchor languages ranges between 0.982 and 0.997.

• Qwen-3 shows slightly more internal divergence, particularly regarding Chinese (zh), which maintains a 0.985 similarity to the multilingual vector, while the alignment for Javanese drops to 0.937.

• Llama-3.2 demonstrates near-perfect uniformity, with the Multilingual (multi) vector achieving a cosine similarity of $> 0 . 9 9$ with English, German, Chinese, and Indonesian. Even the low-resource Javanese direction remains highly aligned at 0.970.

## H.2 Multilingual Dampening Effect

The similarity data supports our “consensus” hypothesis regarding the robustness of multilingual steering. By aggregating diverse lexicons, the resulting vector moves toward a shared centroid, effectively dampening the high-norm “sharpness” characteristic of monolingual directions. This effect is most pronounced in Qwen-3/Chinese, where the specialized, high-intensity monolingual signal is diluted by the lower-norm profiles of the broader lexicon. While this dampening enhances stability across the “long tail” of languages, it accounts for the observed Specialization Penalty in highresource anchors, where the reduced vector magnitude fails to meet the model’s native representational threshold.

## I Establishing the Causal Relevance of Targeted Intervention Sites

## I.1 Experimental Setup

As briefly introduced in Section 3.2, we performed a causal intervention experiment to verify that our isolated layers and neurons actively govern model generation rather than inducing arbitrary changes or gibberish output. We conducted this experiment using the multilingual direction across two low-resource setups: Gemma-Hausa and Qwen-Javanese. We randomly sampled 50 prompts from our broader 1,000-prompt evaluation set and steered them toward both positive and negative targets, yielding 100 paired evaluation instances per model. To isolate the causal effect of our XGBoost-selected neurons, we established our proposed method against two rigorous nullhypothesis baselines:

• Proposed Method $( H _ { 1 } )$ : Intervene strictly on the top-50% probe-ranked layers, using only the specific sparse neurons isolated by XGBoost.

• Fully Random Baseline $( H _ { 0 } ^ { ( 1 ) } )$ : Intervene on a randomly sampled 50% of the layers across the architecture, using a completely random subset of neuron indices of identical size $( | \mathcal { N } _ { i } | )$ .

• Layer-Controlled Baseline $( H _ { 0 } ^ { ( 2 ) } )$ : Intervene on the exact same top-50% probe-ranked layers, but replace the XGBoost-selected coordinates with a random subset of neuron coordinates of identical size $( | \mathcal { N } _ { i } | )$ ).

## I.2 Metrics and Statistical Significance

We evaluated causal efficacy using two metrics: Sentiment Accuracy (whether the generated sentiment matches the intended target) and Strict Success Rate (SR) (whether the text matches the target sentiment and retains the correct Language ID, ensuring no language-switching or gibberish occurred). Because the experimental outcomes are paired and binary, we applied McNemar’s test to evaluate statistical significance against a standard threshold of $p < 0 . 0 5$

## I.3 Results and Discussion

Table 8 and Table 9 summarize the outcomes against the fully random and layer-controlled baselines, respectively. For both Gemma-Hausa and Qwen-Javanese, our proposed method $( H _ { 1 } )$ significantly outperforms both the fully random $( H _ { 0 } ^ { ( 1 ) } )$ and layer-controlled $( H _ { 0 } ^ { ( 2 ) } )$ baselines in Sentiment Accuracy $( p < 0 . 0 5 )$ . By demonstrating that steering via these specific neurons changes model behavior in a predictable, task-relevant manner compared to random controls, we confirm that these representations are causally relevant, addressing concerns regarding correlative probing (Ravichander et al., 2021).

In Gemma-Hausa, our method yields a highly significant improvement in Strict Success Rate $( p < 1 0 ^ { - 8 } )$ . Randomly perturbing neurons or layers frequently broke the Language ID or induced gibberish, whereas targeting our probe-selected neurons successfully flipped the sentiment while keeping the language generation intact.

<table><tr><td colspan="2">Model &amp; Language Metric</td><td colspan="2"> $H _ { 1 } \ ( \mathbf { O u r s } )$   $H _ { 0 } ^ { ( 1 ) }$ </td><td colspan="2">McNemar p-value Significance  $( \alpha = 0 . 0 5 )$ </td></tr><tr><td rowspan="2">Gemma-Hausa</td><td>Sentiment Acc.</td><td>66%</td><td>3%</td><td> $3 . 0 9 \times 1 0 ^ { - 1 7 }$ </td><td>Significant</td></tr><tr><td>Strict SR</td><td>56%</td><td>0%</td><td> $2 . 7 8 \times 1 0 ^ { - 1 7 }$ </td><td>Significant</td></tr><tr><td rowspan="2">Qwen-Javanese</td><td>Sentiment Acc.</td><td>27%</td><td>7%</td><td>0.0003</td><td>Significant</td></tr><tr><td>Strict SR</td><td>9%</td><td>4%</td><td>0.2266</td><td>Not Significant</td></tr></table>

Table 8: Proposed Framework $( H _ { 1 } )$ vs. Fully Random Baseline $( H _ { 0 } ^ { ( 1 ) } )$ . This baseline evaluates the effect of applying the steering vector to a random subset of layers (50% of all layers) and randomly sampled neuron indices, preserving the exact intervention size $( | N _ { i } | )$ of the original targeted configuration.

<table><tr><td colspan="2">Model &amp; Language Metric</td><td colspan="4"> $H _ { 1 } \ ( \mathbf { O u r s } )$   $H _ { 0 } ^ { ( 2 ) }$  McNemar p-value Significance  $( \alpha = 0 . 0 5 )$ </td></tr><tr><td>Gemma-Hausa</td><td>Sentiment Acc.</td><td>66%</td><td>17%</td><td> $5 . 9 1 \times 1 0 ^ { - 1 2 }$ </td><td>Significant</td></tr><tr><td rowspan="2">Qwen-Javanese</td><td>Strict SR</td><td>56%</td><td>16%</td><td> $1 . 5 1 \times 1 0 ^ { - 9 }$ </td><td>Significant</td></tr><tr><td>Sentiment Acc.</td><td>27%</td><td>12%</td><td>0.0107</td><td>Significant</td></tr><tr><td></td><td>Strict SR</td><td>9%</td><td>6%</td><td>0.6072</td><td>Not Significant</td></tr></table>

Table 9: Proposed Framework $( H _ { 1 } )$ vs. Layer-Controlled Baseline $( H _ { 0 } ^ { ( 2 ) } )$ . This baseline controls for layer location by intervening on the exact same top 50% probe-ranked layers but replacing our XGBoost-selected coordinates with random neuron indices.

For Qwen-Javanese, the strict success rate difference was not statistically significant. This aligns directly with our findings in Table 6, which demonstrates that Qwen has an overall Javanese baseline Success Rate of only 9.8%. Because the base model inherently struggles to maintain Javanese text generation, any representation steering is bottlenecked by the model’s underlying language generation capabilities. Crucially, however, the Sentiment Accuracy remains statistically significant $( p \ = \ 0 . 0 1 0 7$ and $p \ = \ 0 . 0 0 0 3 )$ , demonstrating that the targeted features do encode sentiment even when the base model’s language constraints limit full generation success.

## J Isolating the Multilingual Advantage from Vector Norm and Scaling Effects

Given the high cosine similarity between the monolingual and multilingual vectors as discussed in Section H, it is critical to determine whether the observed “multilingual advantage” is driven by true cross-lingual semantic directions or if it is an artifact of vector norm differences and alpha scaling (α). To isolate the geometric impact of the vector direction from these scaling factors, we present controlled experiments executed on Gemma-3 for Hausa (ha). Hausa serves as a representative low-resource setting where the base model demonstrates robust performance, ensuring a reliable baseline environment to analyze steering dynamics.

## J.1 Structural Divergence in Steering Pipelines

Our two-stage feature attribution mechanism independently derives the optimal layer configurations and their corresponding active neuron masks for both setups. Consequently, the selected subset of target layers inherently differs. As illustrated in Figure 11, the top-50% layer sets for the monolingual and multilingual directions exhibit structural divergence across all evaluated models. Specifically, the layer-wise Jaccard similarity is 0.70 (14 of 20 shared active layers) for Gemma-3, 0.40 (8 of 20 shared) for Llama-3.2, and 0.80 (16 of 20 shared) for Qwen-3.

Furthermore, even when both setups select the exact same layers, the specific active neurons targeted within those layers diverge substantially. To isolate this neuron-level divergence, we analyze the middle network layer for each model (using 0- based indexing): Layer 16 for Gemma-3, Layer 13 for Llama-3.2, and Layer 17 for Qwen-3 (Figure 12). We chose these specific middle layers because they are independently identified as active target layers by both the monolingual and multilingual setups. By examining layers where the selection perfectly aligns, we provide an objective baseline to demonstrate that layer-level agreement does not guarantee neuron-level agreement. Within these shared layers, the active neuron masks exhibit minimal overlap (all $J \le 0 . 3 4 )$ . This proves that the two setups steer fundamentally distinct internal processing pathways, rather than the multilingual vector acting as a merely scaled version of the English circuit.

## J.2 Norm-Controlled Baselines

To control for differences in scale, we first rescaled the English vector to match the Euclidean norm of the multilingual vector,

$$
v _ { \mathrm { s c a l e d } } = { \frac { \| v _ { \mathrm { m u l t i } } \| _ { 2 } } { \| v _ { \mathrm { m o n o } } \| _ { 2 } } } \cdot v _ { \mathrm { m o n o . } }
$$

Using this vector, we formalize the validation of the multilingual advantage through the following comparative configurations.

• Original Multilingual Setup $( H _ { 1 } )$ : Intervene using the multilingual direction vector v<sub>multi</sub> within the independently optimized multilingual layer/neuron configuration at $\alpha _ { \mathrm { m u l t i } } =$ 0.75.

• Norm/α-Controlled Baseline $( H _ { 0 } ^ { ( A ) } )$ : Intervene using the rescaled monolingual English vector $v _ { \mathrm { s c a l e d } }$ within the native monolingual layer/neuron configuration at $\alpha _ { \mathrm { m u l t i } }$

• Configuration-Swapped Baseline $( H _ { 0 } ^ { ( B ) } )$ Inject the rescaled monolingual English vector $v _ { \mathrm { s c a l e d } }$ directly into the multilingual layer/neuron configuration at α<sub>multi</sub>.

## J.3 Results and Statistical Significance

We evaluated these baselines against the evaluation dataset consisting of 1,000 unique base prompts. Steering each prompt toward two distinct sentiment targets yields a total of N = 2000 paired text evaluations per condition (Table 10).

We tested whether the multilingual advantage persists after controlling for vector scale and structural placement. Specifically, we used McNemar’s test for paired nominal data, comparing binary outcomes (correct vs. incorrect sentiment generation) across identical prompt instances. We set the threshold for statistical significance at $p < 0 . 0 5$

<table><tr><td>Setup</td><td>Vector</td><td>L/N Config</td><td>Scale</td><td>SA</td><td>SR</td></tr><tr><td>Mono</td><td>vmono</td><td>Mono</td><td>Tuned</td><td>60.8</td><td>54.6</td></tr><tr><td> $H _ { \mathrm { n } } ^ { ( A ) }$ </td><td>Uscaled</td><td>Mono</td><td>αmulti</td><td>61.3</td><td>56.1</td></tr><tr><td> $H _ { 0 } ^ { ( B ) }$ </td><td>Uscaled</td><td>Multi</td><td>αmulti</td><td>61.4</td><td>56.5</td></tr><tr><td>H1 (Multi)</td><td>Umulti</td><td>Multi</td><td>Tuned</td><td>64.4</td><td>58.9</td></tr></table>

Table 10: Performance of norm-matched and configuration-swapped baselines on Gemma-3 (Hausa). Metric scores (SA, SR) are in percentages (%). Statistical significance $( p < 0 . 0 5 )$ across identical prompt instances was achieved for Sentiment Accuracy (SA) against both baselines, but not for Steering Rate (SR).

$H _ { 1 }$ vs. $H _ { 0 } ^ { ( A ) }$ : This evaluates whether the multilingual direction yields an advantage over a perfectly norm- and scale-matched English vector operating in its native architecture zone. McNemar’s test yields $p = 0 . 0 2 8 5$ . Since $p < 0 . 0 5$ , we reject $H _ { 0 } ^ { ( A ) }$ , confirming that the multilingual advantage is statistically significant over a norm-corrected monolingual setup.

$H _ { 1 } \mathbf { v s } . H _ { 0 } ^ { ( B ) }$ : This evaluates if the advantage holds when the rescaled monolingual vector is forced into the identical multilingual configuration. McNemar’s test yields $p = 0 . 0 3 7 4$ Since $p < 0 . 0 5$ , we reject $\cdot \hat { H } _ { 0 } ^ { ( B ) }$ . Even within the cross-lingually optimized pathway, the rescaled English direction underperforms the true multilingual direction.

These results demonstrate that while the norm effect is present (rescaling the English vector yields a minor gain in sentiment accuracy $( 6 0 . 8 \% $ 61.3%)), it cannot fully close the performance gap. In high-dimensional LLM activation spaces, nearcollinear cosine similarities can mask orthogonal directional components. Minor directional shifts in this high-dimensional space carry a unique crosslingual utility that cannot be replicated by scalar adjustments to a monolingual vector.

## K Empirical Analysis of Incoherent Outputs

During our investigations, severe linguistic collapse (manifesting as non-words, punctuation loops, or degenerate repetitive sequences) emerged under two distinct experimental contexts. To clarify the failure modes discussed in the main paper, we decouple our preliminary baseline observations from the formal hyperparameter sweep.

![](images/7e43082468a7c35333462482ca89b365638252fd39be3e25eace42a37505011f.jpg)

![](images/516835699dcb2d47a3c7b18621af49566c874a2c2ab6e32d14479b2aa435f375.jpg)

![](images/475920dcf972522738b23e2e512140e8c07b3925e302fb04ea35841962763ce5.jpg)  
Figure 11: Layer selection divergence across Gemma-3 (top-17 of 34), Llama-3.2 (top-14 of 28), and Qwen-3 (top-18 of 36). Blue and red striped regions indicate layers active exclusively in the Monolingual (EN) and Multilingual (11-lang) setups, respectively, while black indicates agreement.

![](images/e373cdf8c3526fb3760c0e3add806189ca6a57745d8298d24abdfd9ce3505c9e.jpg)  
Figure 12: Neuron activation divergence between the Monolingual (EN) and Multilingual (11-lang) directions within the representative middle layer of each model. The minimal overlap across all three architectures confirms that the setups utilize distinct active neuron subnetworks.

1. Ablation of Sparse Selection (Dense Intervention): In Section 3.2 (the Sparse Neuron Selection via Non-linear Attribution paragraph), we argue that dense intervention (steering all neurons indiscriminately) injects excessive noise into foundational syntax circuits. As shown in the top half of Table 11, preliminary baselines applying vectors across all neurons consistently resulted in syntax breakdown. Because this dense configuration universally failed to produce evaluable text even at low intensities $( \alpha \leq 0 . 5 )$ , it was excluded from the main evaluation sweep.

2. Magnitude Oversaturation (Figure 3 Black Cells): Even when successfully isolating features via sparse selection (e.g., non-zero gain neurons), excessive steering multipliers force the model outside its operational distribution. In Figure 3, these autoregressive collapses are visually encoded as black cells. We observe that the Monolingual vector shows a higher tendency to trigger collapse at $\alpha \ \geq \ 0 . 7 5$ across many non-English languages, whereas the Multilingual vector remains comparatively robust.

Table 11 compiles representative outputs illustrating both failure regimes. To clearly showcase these degradation patterns, all examples in this table utilize the Monolingual (en) steering direction. Driven by its larger vector norm (see Figure 8), it is particularly susceptible to both dense-induced collapse and magnitude oversaturation.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Layer Config</td><td rowspan="2">Neuron Selection</td><td rowspan="2">α Lang</td><td rowspan="2"></td><td rowspan="2">Translated Prompt</td><td rowspan="2"></td><td rowspan="2">Degenerate Output</td></tr><tr><td></td></tr><tr><td>Llama3.2-3B</td><td>top-5</td><td>all</td><td>0.5 en</td><td></td><td>Dense Intervention: Disabling Sparse Selection He turned on his com- \nTheLG-HeCJHG-4HEL-200D-J</td><td>Question-Question-</td><td></td></tr><tr><td>Llama3.2-3B</td><td>top-5</td><td>all</td><td>0.3 zh</td><td>puter. shoes.</td><td>He put on his running</td><td></td><td></td></tr><tr><td>Qwen3-4B</td><td>top-5</td><td>all</td><td></td><td>6:30 AM.</td><td>0.2 id The alarm went off at »</td><td>bonononnebit amongonnebnebnebvitit, of are</td><td>ordered of,bbbdon of ordered of of of of, of monn,bonitnakne of, amongnebpitneby</td></tr><tr><td>Qwen3-4B</td><td>top-5</td><td>all</td><td></td><td>screen.</td><td>0.2 de He looked at his phone \n \n \n \n\n \n\n\n \n</td><td>ofbneonbonbbbhnepnenebhon itnepnebpitnedithitonnebf uccitititbbbb hitneb \n \n \n \n. . . .</td><td>atvnenenelenelepitbononit \n\n \n \n\n\n\n \n \n\n \n \n\n\n \n \n \n\n \n . \n \n \n \n\n\n \n \n \n \n\n</td></tr><tr><td>Gemma3-4B</td><td>all</td><td>Magnitude Oversaturation: Figure 3 Black Cells</td><td></td><td></td><td></td><td>\n...... \nd\n\nd\n... \n \n....</td><td>\n \n... \n \n\n \n\n \n</td></tr><tr><td>Qwen3-4B</td><td>all</td><td>non-zero gain 1.0 jv</td><td></td><td>non-zero gain 1.0 jv</td><td>She opened the spread- Nalokir sheet to track expenses. d-r-s-s2 - ing The building stood next to Nak Dak Dakakakak, Dak Dak</td><td>u\nAnus\n</td><td>sing bisaukak - er or</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Dak D, D D, D is the, D D,</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>the road.</td><td></td><td>D In, D, D, D, D, D, D, D,</td><td></td></tr><tr><td>Llama3.2-3B</td><td></td><td></td><td></td><td></td><td></td><td>D, D, D, D, D, D, D, D, D,</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>D, D, D, D, D, D, D, D, D,</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>D, D, D, D, D, D, D, D, D,</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>D, D; D, D, D; D, D, D; D,</td><td></td></tr><tr><td></td><td>all</td><td></td><td></td><td></td><td></td><td>D; D, D, D; D, D</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>non-zero gain 1.0 jv I sent a text message Itu makan. Bukan. I. But.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>about meeting up.</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>I. Not. I. I. Not. I. Not.</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>I. Not. Not. I. Not. I.</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Not. Not. I. Not. Not. I.</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. I. Not.</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. Not. Not.</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. Not. Not.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. Not. Not.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. Not. Not.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. Not. Not.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. Not. Not.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Not. Not. Not. Not. Not. Not.</td><td></td><td></td><td></td></tr></table>

Table 11: Representative degenerate outputs illustrating the two failure modes discussed in the paper. All examples utilize the Monolingual steering vector. Top: Linguistic collapse caused by dense intervention (steering all neurons), excluded from the main sweep. Bottom: Collapse stemming from magnitude oversaturation on sparse features (α = 1.0), which corresponds directly to the unevaluable black cells in Figure 3.