# Uncovering Audio-Text Alignment in Large Audio-Language Models through Cross-Modal Concept Neurons

Tzu-Quan Lin<sup>1,2</sup>\*, Chih-Kai Yang<sup>2†</sup>, Yi-Cheng Lin<sup>2†</sup>, Hao Tang<sup>1</sup>, Hung-yi Lee<sup>2,3</sup> <sup>1</sup>Centre for Speech Technology Research, University of Edinburgh, United Kingdom <sup>2</sup>Graduate Institute of Communication Engineering, National Taiwan University, Taiwan <sup>3</sup>NTU Artificial Intelligence Center of Research Excellence (NTU AI-CoRE), Taiwan

## Abstract

While the language model backbone is pretrained on text alone, large audio-language models (LALMs) can, to a great extent, recognize acoustic attributes, such as emotion, speaker gender, and spoken language. However, it is unclear how the audio representation is integrated, if at all, with the other text representation before the model produces text. We hypothesize that there is a shared subspace where the audio and text representations are aligned. To test this hypothesis, we identify cross-modal neurons that are responsible for representing both audio and text. We find alignment between audio and text neurons to various degrees across three LALMs and four acoustic attribute types. Causal interventions, steering and deactivating the neurons, confirm that the cross-modal neurons are not just necessary for prediction but can also be manipulated.

## 1 Introduction

There are increasingly many large audio-language models (LALMs) that process acoustic input directly through an LLM backbone (Chu et al., 2024; Tang et al., 2024; Gong et al., 2024), enabling tasks such as speaker, emotion, and spoken-language identification from raw audio (Yang et al., 2024; Huang et al., 2024, 2025; Sakshi et al., 2025). Since the representations of an LALM’s LLM backbone originate from text pretraining, this raises a fundamental question: how does a backbone whose representations were built for text come to recognize acoustic attributes? It is less likely that the audio representation remains separate from the text representation given how the LLM backbone is trained. If the audio representation is integrated with the text representation, there should exist a joint space where the audio and the text representations are aligned. In this work, we hypothesize that LALMs recognize acoustic attributes through such an audio-text alignment.

![](images/2b8534fa9f2988f333e8cab5cfcca80e08c030d13e5cd551c786fe18fb4f2431.jpg)  
Figure 1: Why we compare neurons rather than entire vectors in the ambient space (schematic). Audio and text are fed to the LALM in separate forward passes. Cosine values follow Qwen2.5-Omni on emotion (§4.1); neuron counts are illustrative (§2, §4.2).

Interpretability work on LALMs has approached related questions from several angles: tracking how attribute information evolves across layers (Yang et al., 2025a; Chen et al., 2026), locating attention heads that govern how much the model attends to audio (Glazer et al., 2026), and identifying individual neurons with attribute-specific function, such as for emotion (Zhao et al., 2026a,b). One line of work shows that audio-language models project the linguistic content of speech (what is being said) into a representation aligned with the text embedding space (Ògúnrèmí et al., 2025). These studies either analyze the audio on its own or relate speech content, rather than paralinguistic or speaker attributes, to text. These results tell us little about whether and how audio representations are integrated with text representations.

The work closest to answering our question is the Semantic Hub hypothesis (Wu et al., 2025), which assumes that multimodal LLMs route heterogeneous inputs, including acoustic signals such as animal sounds, through a shared, text-anchored semantic space. Their evidence, limited to a single LALM and to non-speech sounds, rests on cosine similarity between vectors in the ambient space (i.e., the entire hidden vectors).

Most dimensions of vectors in the ambient space, however, carry information unrelated to any particular label, such as neurons that are highly active for every input (Sun et al., 2024). A high cosine similarity does not show that the label itself is shared, and a low one does not rule it out. In fact, in our experiments with three LALMs (§4.1), cosine similarity barely tells matching from mismatching audio-text label pairs apart, and label-independent dimensions account for 88%–100% of its variance. A semantic hub, if it exists, should therefore be sought in specific dimensions rather than across all dimensions of the representation. We thus compare only a few dimensions in audio and in text, those most relevant to each label, instead of the entire vectors. This keeps the comparison from being dominated by dimensions unrelated to the label and pinpoints the specific dimensions that carry the alignment for causal interventions.

For these dimensions, we take the individual neurons of the model’s feed-forward (FFN) layers, following the key-value memory view of the FFN layer (Geva et al., 2021). Prior work on selfsupervised speech models shows that such FFN dimensions already store speech-related properties such as phonetic classes, speaker identity, and pitch (Lin et al., 2024, 2025). Using the same approach, we identify neurons of certain attributes for audio and for text. If there exists a shared subspace, there should be a non-trivial overlap between neurons for audio and for text, and we refer to the overlapping neurons as cross-modal concept neurons (Figure 1).

After identifying the cross-modal neurons, it is important to make sure that they are necessary for predicting the audio attributes and have a causal relationship with the concepts of interest. We verify the importance of cross-modal neurons using two approaches of causal intervention, steering (Rimsky et al., 2024) and neuron deactivation (zeroing their activations) (Dai et al., 2022).

Across three LALMs (Qwen2.5-Omni, DeSTA2.5-Audio, and Audio Flamingo 3) and four attribute types (animal sound, emotion, gender, and spoken language), we find:

• The alignment exists but is hidden from cosine similarity. Label-specific audio and text neurons coincide far more for matching than for mismatching labels, whereas cosine similarity barely tells matching from mismatching labels apart (§4.1, §4.2).

• Interventions defined from text change recognition from audio. Shifting crossmodal concept neurons along a text-derived direction raises the target answer significantly more than random neurons for 35 of 48 labels in Qwen2.5-Omni and Audio Flamingo 3 (7 of 24 in DeSTA2.5-Audio), and can make Qwen2.5-Omni report “happy” for every silent input (20% without steering; best random set 45%; §4.3).

• Recognition relies on cross-modal concept neurons. Deactivating them impairs the target task before unrelated tasks in all 8 task combinations of Qwen2.5-Omni and Audio Flamingo 3 (2 of 4 in DeSTA2.5-Audio), with only 0.2–0.7% of FFN neurons in Qwen2.5- Omni, and is as damaging as deactivating the most audio-specific neurons although they are less audio-specific (§4.4). DeSTA2.5-Audio’s weaker effects may partly reflect the ASR transcript it also receives.

## 2 Cross-Modal Concept Neurons

To understand whether audio and text are mapped to a shared subspace, we study audio recordings of a particular attribute paired with text descriptions of it. Consider, for example, the emotion attribute angry. We run two forward passes through an LALM, one with a speech utterance of an angry voice followed by the question “What is the emotional tone of the speaker in the audio?” and the other with only a text description of the label, “The speaker feels angry”, i.e., the answer we would expect. If there is a shared subspace for both audio and text, then the two forward passes should produce representations with a sufficient overlap. For the rest of this section, we will describe how we construct the audio and text representations and how the overlap of the two is defined.

FFN key-value memories. Following the keyvalue memory view of the Transformer feedforward (FFN) layer (Geva et al., 2021), the first weight matrix of an FFN layer defines a set of keys: each row is compared against a token’s residualstream representation, and the resulting value, after a nonlinearity, is what we call a neuron’s activation, the strength with which that token matches this particular key. The second FFN weight matrix defines a corresponding set of values: each neuron’s activation scales how much of its associated value vector is written back into the residual stream. Each neuron thus signals whether a fixed direction is active, making neurons a fine-grained unit for testing whether audio and text share such directions, as in neuron-level analyses of text (Geva et al., 2022; Dai et al., 2022) and speech models (Lin et al., 2024, 2025).

Formally, for an input x and token position $p$ at layer ℓ of an LALM decoder, the output vector of the first FFN layer (the SwiGLU output that feeds down\_proj) is denoted as $h _ { \ell } ^ { p } ( x ) \in \bar { \mathbb { R } } ^ { D }$ . A neuron is simply a dimension in $h _ { \ell } ^ { \bar { p } } ( x )$ , indexed by (ℓ, i) with activation $h _ { \ell } ^ { p } ( x ) [ i ]$ . We drop ℓ when it is clear from the context.

Pooling. Since the audio and text inputs span several tokens, we use the pooled hidden vectors from the first FFN layer as the representation of the audio and the text. For an input x (either an audio clip paired with a question or a text description), let $\mathcal { P } ^ { \mathrm { a u d i o } } ( x )$ be the token positions for the audio in the question (typically between the audio start and end markers) and let $\mathcal { P } ^ { \mathrm { t e x t } } ( x )$ be the token positions of the text description. The representations are pooled at the instance level to get the activations of a single instance, and the overall activations are then averaged across instances of the same attribute. Formally, for a label c,

$$
\bar { h } ^ { m } ( x ) = \frac { 1 } { | \mathcal { P } ^ { m } ( x ) | } \sum _ { p \in \mathcal { P } ^ { m } ( x ) } h ^ { p } ( x )\tag{1}
$$

$$
\mu ^ { m } ( c ) = \frac { 1 } { | \mathcal { X } _ { c } ^ { m } | } \sum _ { x \in \mathcal { X } _ { c } ^ { m } } \bar { h } ^ { m } ( x )\tag{2}
$$

where m ∈ {text, audio}, $\mathcal { X } _ { c } ^ { \mathrm { a u d i o } }$ is the set of audio clips paired with questions that entail the label $c$ (for example, angry), and $\mathcal { X } _ { c } ^ { \mathrm { t e x t } }$ is the set of text descriptions that entail the label c.

Distinctiveness. To find the neurons $( \mathrm { i . e . }$ , individual dimensions), we could rank the dimensions based on the average activations. However, it is common that some neurons simply have a high activation baseline across all labels (Sun et al., 2024). To compensate for this, we subtract the baseline (based on other labels) from the average activations and refer to this as the distinctiveness. Formally, we compute the distinctiveness for label c as

$$
s ^ { m } ( c ) = \mu ^ { m } ( c ) - \frac { 1 } { | \mathcal { C } \setminus \{ c \} | } \sum _ { \substack { c ^ { \prime } \in \mathcal { C } \backslash \{ c \} } } \mu ^ { m } ( c ^ { \prime } ) ,\tag{3}
$$

where $m ~ \in ~ \{ \mathrm { t e x t } , \mathrm { a u d i o } \}$ and $\mathcal { C }$ is the label set. Each dimension of $s ^ { m } ( c )$ is the distinctiveness of the corresponding neuron for the modality m.

Cross-modal concept neurons. Finally, we select the top-k neurons, denoted as $\mathrm { T o p K } _ { k } ^ { m } ( c )$ for the modality $m ,$ by ranking the distinctiveness of neurons together from all layers $s _ { 1 } ^ { m } ( c ) , \ldots , s _ { L } ^ { m } ( c )$ for an LALM decoder that has $L$ layers. The intersection of the audio and text neurons

$$
\begin{array} { r } { S ( c ) = \mathrm { T o p K } _ { k } ^ { \mathrm { a u d i o } } ( c ) \cap \mathrm { T o p K } _ { k } ^ { \mathrm { t e x t } } ( c ) } \end{array}\tag{4}
$$

is the set of cross-modal concept neurons of the label c. The hyperparameter k differs across experiments and is described individually in later sections. Comparing the two top-k sets ensures that only dimensions specific to label c enter the comparison, whereas cosine similarity over entire vectors includes every dimension.

## 3 Experimental Setup

We study three modern LALMs that combine an audio encoder with a text-pretrained LLM backbone: Qwen2.5-Omni-7B (Xu et al., 2025), DeSTA2.5-Audio (Lu et al., 2026), and Audio Flamingo 3 (Ghosh et al., 2025). Qwen2.5- Omni and Audio Flamingo 3 update their backbone (Qwen2.5-7B) on audio-text data, whereas DeSTA2.5-Audio keeps its backbone (Llama-3.1- 8B) frozen and aligns audio to it through a trained adapter. DeSTA2.5-Audio also feeds its backbone a Whisper transcript of the audio as text tokens whenever speech is detected; our audio-side activations (§2) exclude these tokens. We return to these differences in §4.4.

We draw our four attribute-recognition tasks from the SAKURA benchmark (Yang et al., 2025b):

emotion (5 labels: disgust, sad, angry, fear, happy), animal sound (9 labels: cat, cow, crow, dog, frog, hen, pig, rooster, sheep), gender (2 labels: female, male), and spoken language (8 labels: Chinese, English, French, German, Italian, Japanese, Korean, Spanish). We use all 500 audio instances of each task with SAKURA’s single-hop multiplechoice questions. For every label, we write five paraphrased text descriptions (e.g., “The speaker feels angry.”, “This is an angry voice.”; full list in Appendix B.10), so that the text side is not tied to one surface form. This audio-text pairing per label is what “cross-modal” refers to.

## 4 Experiments and Results

## 4.1 Cosine Similarity Barely Reveals the Alignment

We first ask whether the alignment is visible in representational similarity, as used in prior work. For each label c, we concatenate its pooled per-layer FFN activations (§2) across all layers, $\mu ^ { m } ( c ) = [ \mu _ { 1 } ^ { m } ( c ) ; \dots ; \mu _ { L } ^ { m } ( c ) ]$ , separately for audio and text. Comparing every audio label’s vector against every text label’s vector with cosine similarity gives a label-by-label matrix whose diagonal holds matched pairs (e.g. audio “dog” vs. text “dog”) and whose off-diagonal holds mismatched pairs (e.g. audio “dog” vs. text “cat”). To quantify how clearly this matrix separates matched from mismatched pairs, we decompose its variance, as in a two-way analysis of variance (Montgomery, 2017), into label-independent row and column offsets (a label similar to every label of the other modality) and a matched-label component, the extra similarity of the diagonal entries. The matchedlabel variance share asks how much of the variation among the matrix’s entries comes simply from whether the audio and text labels match, rather than from some labels being similar to everything (the offsets) or from noise. It is 100% for a matrix that is a uniform bright diagonal on a flat background, and 0% when matched pairs are no more similar than mismatched ones (formula in Appendix A.1). Being a proportion, it can also be computed for the neuron-level overlap matrix of §4.2 and compared with the cosine share of the same model and task.

Table 1 reports this share for all 12 model-task combinations. For cosine similarity, the matchedlabel component is present but small: it accounts for less than 2% of the variance for emotion, gender, and language in every model, and at most 9.7% even for animal, while label-independent offsets account for 88%–100%. Cosine similarity thus registers cross-modal alignment only as a small residual that is largely swamped by structure unrelated to label identity. The same decomposition applied to the neuron-level overlap matrices of §4.2 yields a share from matched labels of 9%–92%, higher than the cosine share in every combination.

<table><tr><td>Model</td><td>Task</td><td>Cosine</td><td>Neuron</td></tr><tr><td>Qwen2.5-Omni</td><td>animal</td><td>8.74%</td><td>70.08%</td></tr><tr><td></td><td>emotion</td><td>0.88%</td><td>25.04%</td></tr><tr><td></td><td>gender</td><td>0.16%</td><td>76.04%</td></tr><tr><td></td><td>language</td><td>0.23%</td><td>55.08%</td></tr><tr><td>DeSTA2.5-Audio</td><td>animal</td><td>0.07%</td><td>9.22%</td></tr><tr><td></td><td>emotion</td><td>1.58%</td><td>52.03%</td></tr><tr><td></td><td>gender</td><td>1.85%</td><td>91.69%</td></tr><tr><td></td><td>language</td><td>0.06%</td><td>9.73%</td></tr><tr><td>Audio Flamingo 3</td><td>animal</td><td>9.69%</td><td>67.05%</td></tr><tr><td></td><td>emotion</td><td>1.05%</td><td>45.46%</td></tr><tr><td></td><td>gender</td><td>&lt;0.01%</td><td>68.57%</td></tr><tr><td></td><td>language</td><td>0.38%</td><td>59.30%</td></tr></table>

Table 1: Matched-label variance share (§4.1): how much of the variation in each audio-by-text label matrix comes from whether the labels match, rather than from labels similar to everything or from noise. Cosine: cosine similarity of pooled activations; Neuron: Jaccard overlap of cross-modal concept neuron sets (§4.2).

## 4.2 Neurons Reveal a Label-Specific Audio-Text Alignment

We next ask whether the alignment is visible at the level of individual neurons, testing two properties of S(c) (§2). Here, k is tuned per model and task on a held-out validation split (20%) over the same grid {25, 50, 100, 200, 500, 1000, 2000, 4000} for every combination, maximizing mean per-label Jaccard overlap (Jaccard, 1912); the selected k is 100 or 200 in all 12 combinations, and with it we reselect the neuron sets on the held-out test split, on which all overlap results are reported.

Existence: as a basic precondition, we first check that $\mathrm { T o p K } _ { k } ^ { \mathrm { a u d i o } } ( c )$ and $\mathrm { T o p K } _ { k } ^ { \mathrm { t e x t } } ( c )$ overlap more than two unrelated neuron sets of the same size would, measured by their Jaccard overlap $J ( A , B ) = | A \cap B | / | A \cup B |$ , tested per label with an exact hypergeometric test (Rivals et al., 2007); the largest per-label p-value over all 12 combinations is below 10<sup>−29</sup> (Appendix A.2; Table 6).

Distinctiveness: whether this overlap is particular to the correct label pairing rather than a generic property of any audio-text pair, tested by reusing the same top-k sets from the existence test above in a label-by-label matrix construction analogous to $\ S 4 . 1 \mathbf { \bar { s } } ,$ , with Jaccard overlap in place of cosine similarity, and tested with an exact label-permutation test (Mantel, 1967) on the diagonal-vs-off-diagonal gap (full procedure in Appendix A.3); Table 2 marks the resulting significance levels. Matched audio-text label pairs share substantially more neurons than mismatched pairs, significant in all nine testable model-task combinations (animal, emotion, language). Gender, with only two labels, cannot be tested by label permutation (its smallest attainable p-value is 0.5). Descriptively, its matched audiotext pairs nonetheless share more neurons than its mismatched pairs in every model, with diagonal-tooff-diagonal ratios of mean Jaccard overlap (1.35– $2 . 7 7 \times ;$ Table 5 in Appendix A.3) comparable to those of the testable tasks. The alignment that cosine similarity barely registers is thus clearly visible at the neuron level, label by label, in all three models and all four attributes (matrices: Appendix C.1).

<table><tr><td>Model</td><td>Anim.</td><td>Emo.</td><td>Gen.</td><td>Lang.</td></tr><tr><td>Qwen2.5-Omni</td><td> $0 . 2 4 5 ^ { * * }$ </td><td> $0 . 1 7 4 ^ { * }$ </td><td> $0 . 1 4 4 ^ { \dagger }$ </td><td> $0 . 1 7 7 ^ { \ast \ast }$ </td></tr><tr><td>DeSTA2.5-Audio</td><td> $0 . 0 9 4 ^ { * * }$ </td><td> $0 . 1 2 1 ^ { \ast \ast }$ </td><td> $0 . 1 2 7 ^ { \dagger }$ </td><td> $0 . 1 2 5 ^ { * * }$ </td></tr><tr><td>Audio Flamingo 3</td><td> $0 . 2 6 4 ^ { * * }$ </td><td> $0 . 2 0 2 ^ { * * }$ </td><td>0.098†</td><td> $0 . 2 1 6 ^ { * * }$ </td></tr></table>

Table 2: Cross-modal concept neurons (§4.2): mean per-label Jaccard overlap between the audio and text top-k sets on the held-out test split. Stars: distinctiveness, exact label-permutation test of the matched-vsmismatched gap (Appendix A.3); $^ { * } p < 0 . 0 5 , ^ { * * } p <$ 0.01; <sup>†</sup>not testable (gender has only two labels, so its smallest attainable p is 0.5). Exact values: Table 6.

## 4.3 Text-Derived Interventions Steer Audio Recognition

If audio and text are aligned in these neurons, an intervention defined from text alone should change what the model recognizes from audio. We test this with contrastive neuron steering (Figure 2) (Turner et al., 2023; Rimsky et al., 2024), which shifts the cross-modal concept neurons along a text-derived direction. With reference vector $v ^ { \mathrm { t e x t } } ( c ) = \mu ^ { \mathrm { t e x t } } ( c )$ (§2’s mean-of-means, concatenated across layers) and direction $\Delta ^ { \mathrm { t e x t } } ( c _ { t } , c _ { r } ) = v ^ { \mathrm { t e x t } } ( c _ { t } ) - v ^ { \mathrm { t e x t } } ( c _ { r } )$ between a target label $c _ { t }$ and a same-task carrier label $c _ { r }$ , we shift

$$
\widetilde { h } _ { \ell } [ i ] = h _ { \ell } [ i ] + \lambda \cdot \Delta _ { \ell } ^ { \mathrm { t e x t } } [ i ] ( c _ { t } , c _ { r } ) , \quad ( \ell , i ) \in \mathcal { S } ( c _ { t } )\tag{5}
$$

at every token position of down\_proj’s input, while the model processes an audio instance of carrier $c _ { r }$ . Shifting toward a value the network has genuinely produced elsewhere, rather than by an arbitrary multiple of its current activation, keeps this closer to the training distribution even at large coefficients (Zhang and Nanda, 2024). We sweep $\lambda \in \{ 0 . 5 , 1 , 1 . 5 , 2 , 3 , 5 \}$ and measure the target rate: the fraction of samples whose predicted answer is the target label. Since $\Delta ^ { \mathrm { t e x t } }$ comes from text alone, a rise in the target rate means that a change defined in text transfers to recognition from audio. This holds even when steering only audio-token positions (Appendix B.6).

Every other label in the task with test samples whose answer options include $c _ { t }$ (up to 20 per label) serves as a carrier $c _ { r } ,$ plus two non-speech synthetic carriers (silence, noise; Appendix B.2), so no single carrier drives an observed effect. We compare the label’s own cross-modal concept neurons $S ( c _ { t } )$ against five independent random neuron sets of the same size N (the null), each drawn uniformly from all FFN neurons outside $S ( c _ { t } )$ and shared across all carriers of the label; k (and hence N) is chosen as described in Appendix B.1. We summarize each neuron set by its target rate averaged over carriers and λ, and test whether the cross-modal concept neuron set lies above the five random sets per label with a one-sided prediction-interval ttest (Hahn and Meeker, 1991), with Benjamini– Hochberg correction across all 72 labels (Benjamini and Hochberg, 1995), and per model with a rank test (Friedman, 1937) that asks how likely so low a rank sum, over the model’s labels, would be if the cross-modal concept neuron set were just another random set (Appendix A.4).

Table 3 summarizes the results. Cross-modal concept neurons produce a significantly higher target rate than all five random sets in 42 of the 72 model-task-label combinations. The effect is clearly model-dependent: 18 of 24 labels reach significance in Qwen2.5-Omni and 17 of 24 in Audio Flamingo 3, but only 7 of 24 in DeSTA2.5- Audio. The rank test agrees: the cross-modal concept neuron set’s average rank among the six sets is 1.21 (Qwen2.5-Omni) and 1.17 (Audio Flamingo 3) against a null of $3 . 5 ~ ( p < 1 0 ^ { - 1 3 } )$ , but 2.50 in DeSTA2.5-Audio $( p = 0 . 0 0 2 )$ : better than chance, but far less consistent. Steering DeSTA2.5-Audio is weakest for spoken language (1 of 8 labels), possibly because its transcript input, written in the spoken language (§3; Appendix B.8), keeps revealing the carrier language during steering. Because crossmodal concept neurons concentrate in later layers, we also compare against random sets with the same number of neurons per layer: on Qwen2.5-Omni’s emotion task, the cross-modal concept neuron set still beats all five for every label (Appendix B.7).

![](images/7329eccc4358a02c047bedda5c826cb40c5d5a1598f58b6352c6c9e73c5b4d35.jpg)  
Figure 2: Overview of the causal tests (schematic; activation values are illustrative). Steering (§4.3): the direction is computed once from text-only forward passes as the difference between the target and carrier labels’ activations, and is added to the target label’s cross-modal concept neurons while the model processes a carrier input (Eq. 5); the carrier can also be silence or noise. Deactivation (§4.4): deactivating a task’s cross-modal concept neurons (Eq. 6) impairs that task before the other tasks, whereas deactivating a random set of the same size does far less damage.

<table><tr><td>Model</td><td>Anim.</td><td>Emo.</td><td>Gen.</td><td>Lang.</td><td>Total</td></tr><tr><td>Qwen2.5-Omni</td><td>4/9</td><td>5/5</td><td>2/2</td><td>7/8</td><td>18/24</td></tr><tr><td>DeSTA2.5-Audio</td><td>2/9</td><td>4/5</td><td>0/2</td><td>1/8</td><td>7/24</td></tr><tr><td>Audio Flamingo 3</td><td>6/9</td><td>4/5</td><td>1/2</td><td>6/8</td><td>17/24</td></tr><tr><td>Total</td><td>12/27</td><td>13/15</td><td>3/6</td><td>14/24</td><td>42/72</td></tr></table>

Table 3: Number of labels for which steering crossmodal concept neurons yields a significantly higher target rate than five matched-size random neuron sets (one-sided prediction-interval t-test, $q \ < \ 0 . 0 5$ after Benjamini–Hochberg correction across all 72 labels; Appendix A.4).

Figure 3 shows one of the strongest cases, Qwen2.5-Omni’s “happy”: the cross-modal concept neurons push the target rate above all five random sets for every carrier, even silence and noise, which carry no acoustic evidence for any label (other models: Appendix C.2).

## 4.4 Recognition Relies on Cross-Modal Concept Neurons

Finally, we ask whether recognition relies on the cross-modal concept neurons that carry the alignment: if their overlap were incidental, removing them should not selectively impair recognition. We compute each label’s $\boldsymbol { S } ( \boldsymbol { c } )$ using only the validation split (§2), so the deactivated neurons are never selected on the same samples we then evaluate, and deactivate them by setting their activations to zero (zero-ablation; schematic in Figure 2) with a

forward pre-hook

$$
\tilde { h } _ { \ell } [ i ] = \left\{ \begin{array} { l l } { 0 } & { ( \ell , i ) \in \mathcal { S } } \\ { h _ { \ell } [ i ] } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{6}
$$

applied to down\_proj’s input at every token position, then measure accuracy on the task’s SAKURA multiple-choice questions, where the model answers with a single option letter (parsing details in Appendix B.9).

We deactivate at the task level: $\textstyle S = \bigcup _ { c \in { \mathcal { C } } } S ( c )$ the union across a task’s labels, and we compare accuracy on that task (target) to the other three (controls), to isolate an effect specific to the deactivated task from a generic disruption. We include five matched-size random-neuron control trials, plus a matched-size audio-only reference control: intuitively, deactivating the neurons most responsive to audio should be highly damaging, making it a natural reference point for checking whether crossmodal alignment specifically, rather than generic audio-responsiveness, drives any effect.

Since a small, fixed neuron-set size can understate a real effect, we repeat every deactivation for each k in the grid of §4.2 extended by doubling up to 256,000 (14 values), which deactivates $N = | S |$ neurons, from 2 to about 500,000 (Appendix B.4), and summarize each resulting curve over N by its decline onset: the smallest N at which accuracy drops by more than 10% relative to baseline and stays down for the rest of the sweep, so that a brief dip that later recovers does not count. A combination is selective if the target’s onset precedes the controls’.

![](images/2b3a91425c49593d39b5f393c5679ad355498f00d9ce0f9f84678d637d7ce68b.jpg)  
Figure 3: Contrastive neuron steering toward the emotion label “happy” in Qwen2.5-Omni, one panel per carrier label. x-axis: steering coefficient λ; y-axis: fraction of samples predicted as “happy”. Blue: cross-modal concept neurons; grey: median of five matched-size random neuron sets, with their min–max range shaded (the null).

Table 4 reports this comparison for all 12 modeltask combinations. Ten of the twelve show a selective necessity window: deactivating a task’s own cross-modal concept neurons degrades that task’s own accuracy before an equally sized deactivation degrades any unrelated task. Deactivating them is also about as damaging as deactivating the audioonly reference set, although they are far less audiospecific: at each task’s decline onset, their median rank in the audio specificity ranking is 1.5–5.0× that of the audio-only neurons, with 37–74% ranking below every audio-only neuron. Recognition thus relies on the neurons that carry the alignment, not merely on the most audio-specific ones. The result holds under a bootstrap over test samples (9 of the 10 windows selective in ≥ 90% of resamples, Audio Flamingo 3’s gender window in only 39%; Appendix A.5) and deactivation restricted to audio-token positions (Qwen2.5-Omni emotion; Appendix B.6). Text descriptions using synonyms instead of the label word (e.g., “The speaker feels furious.” for angry; Appendix B.5) select largely the same neurons and give a nearly identical deactivation effect, so these neurons track the attribute rather than the word itself. Figure 4 shows the emotion curves: in all three models, target accuracy drops before control accuracy, while matched-size random-neuron deactivations decline only at much larger N, so the effect depends on which neurons are deactivated, not merely on how many (other tasks: Appendix C.3).

The two non-selective combinations in Table 4 are both DeSTA2.5-Audio, on the animal and language tasks, where the target task’s accuracy declines at exactly the same N as the controls’, showing no evidence of selective necessity for these two tasks. As in §4.3, its transcript input can reveal the spoken language without audio-driven neurons, but this does not explain the animal task, whose clips almost never receive a transcript; there, its frozen backbone or different backbone family may matter, which three models cannot separate.

<table><tr><td>Task</td><td>Model</td><td>Target N</td><td>Control N</td><td>Sel.</td></tr><tr><td rowspan="2">animal</td><td>Qwen2.5-Omni</td><td>1,745</td><td>36,014</td><td>yes</td></tr><tr><td>DeSTA2.5-Audio Audio Flamingo 3</td><td>5,973 984</td><td>5,973 123,275</td><td>no yes</td></tr><tr><td rowspan="2">emotion</td><td>Qwen2.5-Omni</td><td>1,618</td><td>71,812</td><td>yes</td></tr><tr><td>DeSTA2.5-Audio Audio Flamingo 3</td><td>10,336 5,862</td><td>26,763 180,203</td><td>yes</td></tr><tr><td>gender</td><td>Qwen2.5-Omni DeSTA2.5-Audio</td><td>1,255 317</td><td>264,115 93,948</td><td>yes yes</td></tr><tr><td rowspan="2"></td><td>Audio Flamingo 3</td><td>74,073</td><td>251,664</td><td>yes yes</td></tr><tr><td>Qwen2.5-Omni</td><td>3,817</td><td>107,461</td><td></td></tr><tr><td rowspan="3">language</td><td></td><td></td><td></td><td>yes</td></tr><tr><td>DeSTA2.5-Audio</td><td>5,430</td><td>5,430</td><td>no</td></tr><tr><td>Audio Flamingo 3</td><td>19,538</td><td>102,782</td><td>yes</td></tr></table>

Table 4: Task-level deactivation (§4.4). The number of deactivated neurons N at which target and control accuracy each begin to decline (more than 10% relative to baseline, sustained for the rest of the sweep), and whether the target’s onset precedes the controls’ (selective necessity window).

![](images/f2006b1c87d4161215a9ba150ee929086780bc5338ae4d79c3d906bc1baaf861.jpg)  
Figure 4: Task-level deactivation on the emotion task. Solid lines: target task accuracy (blue) and mean control-task accuracy (orange) versus the number of deactivated neurons N (log scale). Dashed lines: matched-size randomneuron control (target task: median of the five random trials, with their min–max range shaded; control tasks: mean). Purple stars: audio-only neurons, a reference control (§4.4). Vertical dashed lines mark the target’s and controls decline onsets, shaded when the target’s comes first.

## 5 Related Work

Interpretability of vision-language models. A parallel line of work studies how multimodal LLMs that pair a vision encoder with a text-pretrained backbone process visual information internally. Goh et al. (2021) first showed that individual neurons in CLIP fire for the same concept whether it appears as an image or as text, coining the term “multimodal neuron.” Schwettmann et al. (2023) found similar neurons inside a frozen, text-only LLM backbone augmented with a vision encoder: FFN neurons that translate projected visual features into the corresponding text concept in the residual stream, with an ablation showing a causal effect on captioning. None of this work looks at audio, or steers these neurons via activation intervention, the question central to §4.3.

Interpretability of audio-language models. Beyond the layer-, attention-head-, and neuron-level analyses discussed in §1, recent work has tested whether a spoken answer becomes “readable” from a middle layer before the model commits to it (Fan et al., 2026). Other work traced which internal pathway wins when audio and text evidence conflict, along the way reporting, via cosine similarity, that the two converge in later layers (Cho et al., 2026). None of these studies identifies individually shared audio-and-text units and tests both whether steering them changes the model’s behavior and whether it relies on them.

Semantic Hub hypothesis. Beyond audio, Wu et al. (2025) also study multilingual models, where inputs in different languages share representations anchored in the dominant training language. Their evidence rests on cosine similarity over vectors in the ambient space, not on individual neurons (§1), and their causal test adds the difference between two word unembeddings at one layer, a steering test only; we additionally test deactivation, in three LALMs and on both speech and non-speech attributes.

Shared units across surface forms. Representing one concept with a single unit shared across different surface forms has been tested causally elsewhere: Brinkmann et al. (2025) found that ablating a shared unit knocks out a grammatical concept across many typologically distinct languages while a language-specific unit barely matters. Nakai et al. (2026) focus on neurons that control the output language of speech–text models: they identify language-selective neurons separately from speech and text inputs and steer generation through the output-language neurons, whereas the neurons that represent the input language across modalities are compared only by overlap, without testing whether the model relies on them. They also study language identity only, whereas we also cover paralinguistic and non-speech attributes.

## 6 Conclusion

LALMs recognize acoustic attributes through an alignment between audio and text that cosine similarity over entire representations barely reveals but individual FFN neurons expose: the neurons most specific to an attribute in audio and in text coincide, label by label. This alignment is truly cross-modal, since a direction computed from text steers what the model recognizes from audio (§4.3), and recognition relies on it, since removing crossmodal concept neurons selectively impairs the attribute (§4.4). How strongly a model relies on the alignment varies across models.

## Limitations

Our attributes and materials come entirely from the SAKURA benchmark: four fixed attribute sets and text descriptions we authored ourselves (§3); each label’s text side consists of five templated paraphrases. A much larger prompt set per label is hard to construct without introducing label-specific wording that would itself confound the comparison across labels, so we instead use templates shared by every label of a task that differ only in the label word.

Three LALMs is not an exhaustive sample of current architectures. In particular, DeSTA2.5-Audio differs from the other two models at once in its frozen backbone, its backbone family, and its transcript input, so explaining its weaker effects beyond the transcript route for spoken language would require more models.

The gender task follows SAKURA’s binary female/male labels, which do not capture the full range of gender identities or voice characteristics; our gender results should be read as results on this binary recognition task.

Gender has only two labels, so the labelpermutation test of §4.2 cannot assess its distinctiveness (the smallest attainable p-value is 0.5), and we describe it only descriptively; the existence test and the random-set tests of §4.3 do not depend on the number of labels and do cover gender.

For the steering experiment (§4.3), each label’s neuron-set size is taken from its necessity window (§4.4), which is located on the test split. We did not locate these windows on the validation split, which holds as few as 8 samples per label, too few to identify a sustained 10% accuracy decline reliably. The selection uses a different intervention and outcome (deactivation, task accuracy) from those of the steering test (steering, target rate), so it does not directly optimize the steering result, and every random control set is matched to the same neuron-set size.

## Ethical Considerations

This work is purely diagnostic: we analyze the internal representations of three existing, publicly released pretrained LALMs using an existing public benchmark (SAKURA), and we neither collect new data, train new models, nor release any new model weights or fine-tuned checkpoint. The four attributes we study, animal-sound, emotion, gender, and spoken-language identification, are recognition capabilities already present in each model’s public release; our steering and deactivation interventions (§4.3, §4.4) are used strictly as scientific probes to test whether specific neurons can steer, or are necessary for, this existing behavior, not as a proposed technique for modifying or improving deployed systems. We therefore do not foresee this work introducing risks beyond those already inherent in the publicly released models and benchmark it studies.

## Acknowledgments

Tzu-Quan Lin is supported by a joint Ph.D. scholarship from NTU GICE and Delta Electronics, Inc. This work was supported by the Ministry of Education (MOE) of Taiwan under the project Taiwan Centers of Excellence in Artificial Intelligence, through the NTU Artificial Intelligence Center of Research Excellence (NTU AI-CoRE).

We used an AI writing assistant (Claude) to help edit the grammar and prose of this manuscript. The AI assistant did not design any experiment, generate or select any result, or make any research decision; it did not contribute to the paper’s scientific content. The authors are fully responsible for, and vouch for the accuracy of, all scientific content, experiments, and conclusions reported in this paper.

## References

Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300.

Jannik Brinkmann, Chris Wendler, Christian Bartelt, and Aaron Mueller. 2025. Large language models share representations of latent grammatical concepts across typologically diverse languages. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6131–6150, Albuquerque, New Mexico. Association for Computational Linguistics.

Houwei Cao, David G. Cooper, Michael K. Keutmann, Ruben C. Gur, Ani Nenkova, and Ragini Verma. 2014. CREMA-D: Crowd-sourced emotional multimodal

actors dataset. IEEE Transactions on Affective Computing, 5(4):377–390.

Wei-Chih Chen, Chien-yu Huang, and Hung-yi Lee. 2026. Causal tracing of audio-text fusion in large audio language models. arXiv preprint arXiv:2603.13768.

Hyebin Cho, Suho Yoo, Jaehyuk Jang, Changick Kim, and Joon Son Chung. 2026. Who wins the conflict? Mechanistic interpretability of text bias in Audio LLMs. arXiv preprint arXiv:2606.18924.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. 2024. Qwen2-Audio technical report. arXiv preprint arXiv:2407.10759.

John R. Crawford and David C. Howell. 1998. Comparing an individual’s test score against norms derived from small samples. The Clinical Neuropsychologist, 12(4):482–486.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8493– 8502.

Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to the Bootstrap. Chapman & Hall.

Jiajun Fan, Jingyuan Li, Prashanth Gurunath Shivakumar, Qi Luo, Jia-Hong Huang, M. Maruf, Roger Ren, Yile Gu, Rahul Pandey, Ge Liu, and Ivan Bulyko. 2026. Can we read the mind of an audio LLM? A verbalizable, multilingual middle-layer workspace. arXiv preprint arXiv:2608.24958.

Ronald A. Fisher. 1925. Statistical Methods for Research Workers. Oliver and Boyd, Edinburgh.

Milton Friedman. 1937. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal ofthe American Statistical Association, 32(200):675–701.

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. 2022. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 30–45.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495.

Sreyan Ghosh, Arushi Goel, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, and Bryan Catanzaro. 2025. Audio Flamingo

3: Advancing audio intelligence with fully open large audio language models. In Advances in Neural Information Processing Systems, volume 38.

Neta Glazer, Lenny Aharon, and Ethan Fetaya. 2026. Are audio-language models listening? Audiospecialist heads for adaptive audio steering. arXiv preprint arXiv:2603.06854.

Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. 2021. Multimodal neurons in artificial neural networks. Distill.

Yuan Gong, Hongyin Luo, Alexander H. Liu, Leonid Karlinsky, and James Glass. 2024. Listen, think, and understand. In International Conference on Learning Representations (ICLR).

Gerald J. Hahn and William Q. Meeker. 1991. Statistical Intervals: A Guidefor Practitioners. Wiley.

Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, et al. 2020. Array programming with NumPy. Nature, 585:357–362.

Chien-yu Huang, Wei-Chih Chen, Shu-wen Yang, Andy T. Liu, Chen-An Li, Yu-Xiang Lin, Wei-Cheng Tseng, Anuj Diwan, Yi-Jen Shih, Jiatong Shi, William Chen, Chih-Kai Yang, Wenze Ren, Xuanjun Chen, Chi-Yuan Hsiao, Puyuan Peng, Shih-Heng Wang, Chun-Yi Kuan, Ke-Han Lu, Kai-Wei Chang, Fabian Ritter-Gutierrez, Kuan-Po Huang, Siddhant Arora, You-Kuan Lin, Ming To Chuang, Eunjung Yeo, Kalvin Chang, Chung-Ming Chien, Kwanghee Choi, Jun-You Wang, Cheng-Hsiu Hsieh, Yi-Cheng Lin, Chee-En Yu, I-Hsiang Chiu, Heitor R. Guimarães, Jionghao Han, Tzu-Quan Lin, Tzu-Yuan Lin, Homu Chang, Ting-Wu Chang, Chun Wei Chen, Shou-Jen Chen, Yu-Hua Chen, Hsi-Chun Cheng, Kunal Dhawan, Jia-Lin Fang, Shi-Xin Fang, Kuan-Yu Fang Chiang, Chi An Fu, and 31 others. 2025. Dynamic-SUPERB Phase-2: A collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks. In International Conference on Learning Representations (ICLR).

Chien-yu Huang, Ke-Han Lu, Shih-Heng Wang, Chi-Yuan Hsiao, Chun-Yi Kuan, Haibin Wu, Siddhant Arora, Kai-Wei Chang, Jiatong Shi, Yifan Peng, Roshan Sharma, Shinji Watanabe, Bhiksha Ramakrishnan, Shady Shehata, and Hung-yi Lee. 2024. Dynamic-SUPERB: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12136–12140.

John D. Hunter. 2007. Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3):90–95.

Paul Jaccard. 1912. The distribution of the flora in the alpine zone. New Phytologist, 11(2):37–50.

Tzu-Quan Lin, Hsi-Chun Cheng, Hung-yi Lee, and Hao Tang. 2025. Identifying speaker information in feedforward layers of self-supervised speech transformers. In 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Confer ence (APSIPA ASC).

Tzu-Quan Lin, Guan-Ting Lin, Hung-yi Lee, and Hao Tang. 2024. Property neurons in self-supervised speech transformers. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 401–408. IEEE.

Ke-Han Lu, Zhehuai Chen, Szu-Wei Fu, Chao-Han Huck Yang, Sung-Feng Huang, Chih-Kai Yang, Chee-En Yu, Chun-Wei Chen, Wei-Chih Chen, Chien-yu Huang, Yi-Cheng Lin, Yu-Xiang Lin, Chi-An Fu, Chun-Yi Kuan, Wenze Ren, Xuanjun Chen, Wei-Ping Huang, En-Pei Hu, Tzu-Quan Lin, Yuan-Kuei Wu, Kuan-Po Huang, Hsiao-Ying Huang, Huang-Cheng Chou, Kai-Wei Chang, Cheng-Han Chiang, Boris Ginsburg, Yu-Chiang Frank Wang, and Hung-yi Lee. 2026. DeSTA2.5-Audio: Toward general-purpose large audio language model with self-generated cross-modal alignment. IEEE Transactions on Audio, Speech and Language Processing, 34.

Nathan Mantel. 1967. The detection of disease clustering and a generalized regression approach. Cancer Research, 27(2):209–220.

Brian McFee, Colin Raffel, Dawen Liang, Daniel P. W. Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. 2015. librosa: Audio and music signal analysis in Python. In Proceedings ofthe 14th Python in Science Conference, pages 18–25.

Douglas C. Montgomery. 2017. Design and Analysis of Experiments, 9th edition. Wiley.

Toshiki Nakai, Varsha Suresh, and Vera Demberg. 2026. Generation-step-aware framework for cross-modal representation and control in multilingual speech– text models. arXiv preprint arXiv:2601.17387.

Tolúlopé Ògúnrèmí, Christopher D. Manning, Dan Jurafsky, and Karen Livescu. 2025. Transcribe, translate, or transliterate: An investigation of intermediate representations in spoken language models. In 2025 IEEEAutomatic Speech Recognition and Understanding Workshop (ASRU). IEEE.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pages 8024–8035.

Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 527–536.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering Llama 2 via contrastive activation addition. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 15504–15522.

Isabelle Rivals, Léon Personnaz, Lieng Taing, and Marie-Claude Potier. 2007. Enrichment or depletion of a GO category within a class of genes: which test? Bioinformatics, 23(4):401–407.

S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. 2025. MMAU: A massive multi-task audio understanding and reasoning benchmark. In International Conference on Learning Representations (ICLR).

Sarah Schwettmann, Neil Chowdhury, Samuel Klein, David Bau, and Antonio Torralba. 2023. Multimodal neurons in pretrained text-only transformers. In 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 2854–2859.

Mingjie Sun, Xinlei Chen, J. Zico Kolter, and Zhuang Liu. 2024. Massive activations in large language models. In First Conference on Language Modeling (COLM).

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. 2024. SALMONN: Towards generic hearing abilities for large language models. In International Conference on Learning Representations (ICLR).

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, et al. 2020. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17:261– 272.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45.

Zhaofeng Wu, Xinyan Velocity Yu, Dani Yogatama, Jiasen Lu, and Yoon Kim. 2025. The semantic hub hypothesis: Language models share semantic representations across languages and modalities. In International Conference on Learning Representations (ICLR).

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025. Qwen2.5-Omni technical report. arXiv preprint arXiv:2503.20215.

Chih-Kai Yang, Neo Ho, Yi-Jyun Lee, and Hung-yi Lee. 2025a. AudioLens: A closer look at auditory attribute perception of large audio-language models. In 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE.

Chih-Kai Yang, Neo Ho, Yen-Ting Piao, and Hung-yi Lee. 2025b. SAKURA: On the multi-hop reasoning of large audio-language models based on speech and audio information. In Interspeech 2025, pages 1788– 1792.

Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun $\operatorname { L v } ,$ Zhou Zhao, Chang Zhou, and Jingren Zhou. 2024. AIR-Bench: Benchmarking large audio-language models via generative comprehension. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 1979–1998.

Fred Zhang and Neel Nanda. 2024. Towards best practices of activation patching in language models: Metrics and methods. In International Conference on Learning Representations (ICLR).

Xiutian Zhao, Philipp Koehn, Björn Schuller, and Berrak Sisman. 2026a. Multilingual emotion neurons in large audio-language models. arXiv preprint arXiv:2608.08772.

Xiutian Zhao, Björn Schuller, and Berrak Sisman. 2026b. Discovering and causally validating emotionsensitive neurons in large audio-language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15056–15071, San Diego, California, United States.

## A Significance Testing Details

This appendix gives the full procedure behind each significance test used in §4, which the main text describes only briefly.

## A.1 Matched-Label Variance Share

For an $| { \mathcal { C } } | \times | { \mathcal { C } } |$ label-by-label matrix M with row means $\bar { r } _ { i } ,$ column means $\bar { c } _ { j }$ , and grand mean g¯, we fit $M _ { i j } = \alpha _ { i } + \beta _ { j } + \delta \left[ i = j \right] + \varepsilon _ { i j }$ by least squares, as in a two-way analysis of variance (Montgomery,

2017). The row and column terms alone explain $\begin{array} { r } { S S _ { \mathrm { r o w + c o l } } = | \mathcal { C } | \sum _ { i } ( \bar { r } _ { i } - \bar { g } ) ^ { 2 } + | \mathcal { C } | \sum _ { i } ( \bar { c } _ { j } - \bar { g } ) ^ { 2 } } \end{array}$ of the total $\begin{array} { r } { S S _ { \mathrm { t o t a l } } = \sum _ { i , j } ( M _ { i j } - \bar { g } ) ^ { \bar { 2 } } } \end{array}$ . Adding the diagonal term gives ${ \hat { \delta } } \ = \ d ,$ the diagonalminus-off-diagonal gap used by the permutation test below, and explains an additional $( | \mathcal { C } | - 1 ) d ^ { 2 }$ The variance share from matched labels is (|C| − 1) $d ^ { 2 } / S S _ { \mathrm { t o t a l } }$ , and the row-and-column share is $S S _ { \mathrm { r o w + c o l } } / S S _ { \mathrm { t o t a l } }$ . It is a descriptive effect size rather than a test; since it is a proportion, it can be compared directly between the cosine and neuronoverlap matrices of the same model and task. For gender $( | { \mathcal { C } } | = 2 )$ the diagonal term is the only component beyond the row and column offsets.

## A.2 Hypergeometric Existence Test

For a single label c, we ask whether $| \mathrm { T o p K } _ { k } ^ { \mathrm { a u d i o } } ( c ) \cap$ $\mathrm { T o p K } _ { k } ^ { \mathrm { t e x t } } ( c ) |$ is larger than expected if the two sizek sets were independent draws from the full population of LD neurons. This probability is given exactly by the hypergeometric distribution (Rivals et al., 2007), with no permutation or approximation needed. Rather than combining per-label p-values across a task’s labels, for instance with Fisher’s method (Fisher, 1925), which assumes independent p-values although all labels of a task are scored from the same activations, we report the largest per-label p-value of each task, which bounds every label’s $p \cdot$ -value without any independence assumption. The random baseline reported alongside each existence result (Table 6’s “Random baseline” column) is the expected Jaccard overlap of two independent size-k draws, estimated from 100,000 Monte Carlo samples rather than a closed-form expression.

## A.3 Diagonal-vs-Off-Diagonal Permutation Test

Given the label-by-label neuron-Jaccard overlap matrix of §4.2, we take the observed gap between the mean diagonal (matched-label) entry and the mean off-diagonal (mismatched-label) entry. The null distribution comes from permuting the textside label ordering (Mantel, 1967), enumerating all |C|! permutations exactly for a task with |C| labels $( | { \mathcal { C } } | \leq 9 )$ , and recomputing this gap for each, so that under the null, any audio label is equally likely to be paired with any text label. The one-sided p-value is the fraction of permutations (including the identity) whose gap is at least as large as the observed gap, so the smallest attainable p-value is $1 / | \mathcal { C } | ! : 1 / 1 2 0$ for emotion, and $1 / 2$ for gender, which therefore cannot be tested this way. Table 6 reports the resulting p-values, and Table 5 the mean matched-pair and mismatched-pair overlaps behind them.

## A.4 Random-Set Tests for Steering

For each label we draw $M = 5$ random neuron sets, each of $N = | S ( c _ { t } ) |$ neurons sampled uniformly without replacement from all FFN neurons of all layers excluding $S ( c _ { t } )$ , with a seed determined by the model, task, label, and set index. Each set is shared across all carriers of the label, and the cross-modal concept neuron set and all random sets are evaluated on the same carriers, samples, and λ values. Each set S is summarized by $T ( S )$ , its target rate averaged over carriers and λ. Prediction-interval test (per label). We transform $x = \log \mathrm { i t } ( \mathrm { c l i p } ( T , \epsilon , 1 - \epsilon ) )$ with $\epsilon = 0 . 5 / n$ for n evaluated samples, and compute $t = ( x _ { S } - \bar { x } _ { R } ) / ( s _ { R } \sqrt { 1 + 1 / M } )$ , where ${ \bar { x } } _ { R }$ and $s _ { R }$ are the mean and standard deviation of the random sets’ x; the one-sided p-value is the upper tail of a t distribution with $M - 1$ degrees of freedom, i.e. a prediction-interval test of whether the cross modal concept neuron set looks like one more random draw (Hahn and Meeker, 1991; Crawford and Howell, 1998). p-values are Benjamini–Hochberg corrected (Benjamini and Hochberg, 1995) across all 72 labels; without correction, 43 labels are significant at $p < 0 . 0 5$ , and the only one that loses significance is Qwen2.5-Omni’s English $( p = 0 . 0 3 5$ $q = 0 . 0 5 8 )$ . No label had identical scores across its random sets, for which the test would be unde fined. Rank test (across labels; Friedman, 1937). We rank the cross-modal concept neuron set among the $M + 1$ sets by T, breaking ties at random with a fixed seed. Under the null each rank is uniform on $\{ 1 , \ldots , M + 1 \}$ , and the one-sided $p \mathrm { - }$ value is the probability that the sum of independent such ranks, one per label of the model, is at most the observed sum, computed exactly by convolution.

## A.5 Bootstrap Stability of Necessity Windows

Table 4 decides selectivity from one evaluation of the test split. To quantify how stable this decision is, we resample, with replacement, the target task’s test samples and, independently, each control task’s test samples, 2,000 times (Efron and Tibshirani, 1993). Each resample is reused at every N and for the baseline, as a single evaluation set is in the original sweep, and we recompute both decline onsets with the rule of §4.4. A combination counts as selective in a resample when the target onset precedes the control onset. Table 7 reports the results: the observed onsets reproduce Table 4, 9 of the 10 selective combinations remain selective in at least 90% of resamples, Audio Flamingo 3’s gender window is selective in only 39%, and DeSTA2.5- Audio’s animal and language tasks remain nonselective in most resamples.

## B Additional Experimental Setup Details

This appendix collects experimental-setup details referenced from §3 and §4 but not spelled out in the main text.

## B.1 Neuron-Set Size for Steering

For each label, we repeat the deactivation sweep of §4.4 at the label level: we deactivate only that label’s $\boldsymbol { S } ( \boldsymbol { c } )$ and compare accuracy on that label (target) with the mean accuracy on the task’s other labels (control), locating both decline onsets as in §4.4. k (and hence N) is the value between the two onsets that maximizes the target’s relative accuracy drop minus the control’s; labels without such a window use a fixed size matched to the task’s strongest-window label.

## B.2 Synthetic Carriers

Silence and noise are not labels of any task, so for these two carriers we compute $v ^ { \mathrm { t e x t } } ( c _ { r } )$ as for a real label, but from five sentences describing silence (e.g. “This audio contains complete silence.”) or noise (e.g. “This audio contains only random noise.”).

## B.3 Audio-Only Reference Control

The audio-only reference control of §4.4 gives each label an equal quota of $N / | \mathcal { C } |$ neurons from its own audio top-k first, then adds, in round-robin order across labels, each label’s next-ranked audio neuron not yet selected until the union reaches size N, since the equal-quota union alone can fall short of N.

## B.4 Number of Deactivated Neurons

Deactivation sweeps the per-label cutoff k of $\ S 2 ,$ but the deactivated set $\textstyle S = \bigcup _ { c } S ( c )$ is a union of per-label intersections, so its size $N = | S |$ differs from k: the intersections shrink it and the union over labels enlarges it. We therefore report deactivation results against N, to which every randomneuron and audio-only control is matched. N increases strictly with k for every model and task, so the decline onsets and the selectivity decisions do not depend on which of the two is used. Qwen2.5- Omni’s target onsets correspond to 0.24–0.72% of its 530,432 FFN neurons (28 layers × 18,944).

<table><tr><td></td><td colspan="3">Qwen2.5-Omni</td><td colspan="3">DeSTA2.5-Audio</td><td colspan="3">Audio Flamingo 3</td></tr><tr><td>Task</td><td>Matched</td><td>Mismatched</td><td>Ratio</td><td>Matched</td><td>Mismatched</td><td>Ratio</td><td>Matched</td><td>Mismatched</td><td>Ratio</td></tr><tr><td>animal</td><td>0.245</td><td>0.093</td><td>2.62</td><td>0.094</td><td>0.075</td><td>1.26</td><td>0.264</td><td>0.100</td><td>2.63</td></tr><tr><td>emotion</td><td>0.174</td><td>0.113</td><td>1.54</td><td>0.121</td><td>0.074</td><td>1.65</td><td>0.202</td><td>0.115</td><td>1.75</td></tr><tr><td>gender</td><td>0.144</td><td>0.070</td><td>2.07</td><td>0.127</td><td>0.046</td><td>2.77</td><td>0.098</td><td>0.072</td><td>1.35</td></tr><tr><td>language</td><td>0.177</td><td>0.089</td><td>1.99</td><td>0.125</td><td>0.087</td><td>1.43</td><td>0.216</td><td>0.090</td><td>2.40</td></tr></table>

Table 5: Mean Jaccard overlap of matched (diagonal) and mismatched (off-diagonal) audio-text label pairs in the neuron-overlap matrix of §4.2, and their ratio.
<table><tr><td>Model</td><td>Task</td><td>k</td><td>Jaccard</td><td>Random baseline</td><td>Max. p (exist.)</td><td>Perm. p (distinct.)</td></tr><tr><td rowspan="4">Qwen2.5-Omni</td><td>animal</td><td>200</td><td>0.245</td><td>0.000185</td><td> $1 . 4 \times 1 0 ^ { - 1 3 9 }$ </td><td> $2 . 8 \times 1 0 ^ { - 6 } \dagger$ </td></tr><tr><td>emotion</td><td>100</td><td>0.174</td><td>0.000096</td><td> $2 . 2 \times 1 0 ^ { - 5 5 }$ </td><td>0.042</td></tr><tr><td>gender</td><td>200</td><td>0.144</td><td>0.000185</td><td> $4 . 0 \times 1 0 ^ { - 1 0 0 }$ </td><td></td></tr><tr><td>language</td><td>100</td><td>0.177</td><td>0.000096</td><td> $2 . 2 \times 1 0 ^ { - 5 5 }$ </td><td> $2 . 5 \times 1 0 ^ { - 5 } \dagger$ </td></tr><tr><td rowspan="4">DeSTA2.5-Audio</td><td>animal</td><td>100</td><td>0.094</td><td>0.000111</td><td> $6 . 0 \times 1 0 ^ { - 3 0 }$ </td><td>0.0025</td></tr><tr><td>emotion</td><td>100</td><td>0.121</td><td>0.000111</td><td> $8 . 8 \times 1 0 ^ { - 4 5 }$ </td><td>0.0083†</td></tr><tr><td>gender</td><td>200</td><td>0.127</td><td>0.000214</td><td> $1 . 1 \times 1 0 ^ { - 9 4 }$ </td><td></td></tr><tr><td>language</td><td>100</td><td>0.125</td><td>0.000111</td><td> $9 . 2 \times 1 0 ^ { - 3 6 }$ </td><td>0.010</td></tr><tr><td rowspan="4">Audio Flamingo 3</td><td>animal</td><td>100</td><td>0.264</td><td>0.000096</td><td> $4 . 1 \times 1 0 ^ { - 8 9 }$ </td><td> $2 . 8 \times 1 0 ^ { - 6 } \dagger$ </td></tr><tr><td>emotion</td><td>100</td><td>0.202</td><td>0.000096</td><td> $1 . 6 \times 1 0 ^ { - 6 8 }$ </td><td> $0 . 0 0 8 3 \dagger$ </td></tr><tr><td>gender</td><td>200</td><td>0.098</td><td>0.000185</td><td> $2 . 3 \times 1 0 ^ { - 7 4 }$ </td><td></td></tr><tr><td>language</td><td>100</td><td>0.216</td><td>0.000096</td><td> $1 . 6 \times 1 0 ^ { - 6 8 }$ </td><td> $2 . 5 \times 1 0 ^ { - 5 } \dagger$ </td></tr></table>

Table 6: Existence of cross-modal concept neurons (§2): tuned neuron-set size k, mean per-label Jaccard overlap on the held-out test split, the corresponding random-overlap baseline, and the largest per-label hypergeometric p-value among the task’s labels (existence; Appendix A.2), and the exact label-permutation p-value of the diagonalvs-off-diagonal Jaccard gap (distinctiveness; Appendix A.3). †: the smallest attainable p-value, $1 / | { \mathcal { C } } | !$ for a task with |C| labels. All testable combinations are significant at $p < 0 . 0 5$ . Gender (two labels) cannot be tested by label permutation, since its smallest attainable p-value is 0.5.

## B.5 Synonym Control

Because each label’s text prompts contain the label word itself (e.g. “The speaker feels angry.”), a cross-modal concept neuron set could in principle reflect the literal label word rather than the attribute. To test this, we re-extract Qwen2.5-Omni’s text activations for the emotion task with the same five templates but the label word replaced by a synonym sharing no token with it (angry → furious, happy → joyful, sad → sorrowful, fear → terror, disgust → revulsion), re-identify cross-modal concept neurons at $k \in \{ 1 , 0 0 0 , 2 , 0 0 0 , 4 , 0 0 0 \}$ , and repeat the task-level deactivation of §4.4. The synonymbased sets overlap substantially with the original ones (mean per-label Jaccard 0.52–0.53, against 0.04–0.05 between mismatched labels), and deactivating them is nearly as damaging to emotion recognition as deactivating the original sets (relative target-accuracy drop 8.1/11.2/16.1% vs. 8.7/13.0/18.6% at the three values of k, against 2.2– 3.9% for matched-size random neurons), with similar control-task drops (5.1–5.8% vs. 3.7–7.4%). At least for this model and task, the neurons thus track the attribute rather than the specific label word.

## B.6 Audio-Position Control

Both interventions are applied at every token position, so part of their effect could come from disrupting how the model reads the question rather than how it processes the audio. We therefore repeat both on Qwen2.5-Omni’s emotion task, the same setting as the synonym control, restricting the intervention to each sample’s audio-token positions. Deactivation remains selective: the target task’s decline onset moves from N = 1,618 to 12,804 and the control tasks’ from 71,812 to 189,409, the control tasks are nearly unaffected until $N = 7 1 { , } 8 1 2$ , and at $N = 1 { , } 6 1 8$ and 12,804 target accuracy is below all five random-neuron control trials, whose neurons are likewise deactivated only at audio-token positions. For steering, the cross-modal concept neuron set outranks all five random sets for all five labels (rank test $p = 1 . 3 \times 1 0 ^ { - 4 } )$ and is significant for four of them under the prediction-interval test; its lead over the random sets in average target rate is 54–82% as large as under whole-sequence steering. This holds even for silent input, which carries no acoustic content: at λ = 5 the cross-modal concept neuron set’s target rate exceeds every random set’s for all five labels, reaching 100% for “happy” against at most 30% for the random sets. Restricting the interventions to audio positions thus weakens both effects, but leaves their selectivity, their advantage over random neurons, and hence our conclusions unchanged.

<table><tr><td>Task</td><td>Model</td><td>Target onset</td><td>95% interval</td><td>Control onset</td><td>95% interval</td><td>P(selective)</td></tr><tr><td rowspan="3">animal</td><td>Qwen2.5-Omni</td><td>1,745</td><td>932-3,406</td><td>36,014</td><td>36,014–36,014</td><td>1.00</td></tr><tr><td>DeSTA2.5-Audio</td><td>5,973</td><td>2,693–40,280</td><td>5,973</td><td>2,693–15,110</td><td>0.25</td></tr><tr><td>Audio Flamingo 3</td><td>984</td><td>984-1,804</td><td>123,275</td><td>63,872–123,275</td><td>1.00</td></tr><tr><td rowspan="3">emotion</td><td>Qwen2.5-Omni</td><td>1,618</td><td>133-5,996</td><td>71,812</td><td>71,812–189,409</td><td>1.00</td></tr><tr><td>DeSTA2.5-Audio</td><td>10,336</td><td>4,395–26,763</td><td>26,763</td><td>26,763–26,763</td><td>0.90</td></tr><tr><td>Audio Flamingo 3</td><td>5,862</td><td>444–69,437</td><td>180,203</td><td>69,437–180,203</td><td>1.00</td></tr><tr><td rowspan="3">gender</td><td>Qwen2.5-Omni</td><td>1,255</td><td>669–1,255</td><td>264,115</td><td>264,115–264,115</td><td>1.00</td></tr><tr><td>DeSTA2.5-Audio</td><td>317</td><td>199-317</td><td>93,948</td><td>93,948–93,948</td><td>1.00</td></tr><tr><td>Audio Flamingo 3</td><td>74,073</td><td>25,037–251,664</td><td>251,664</td><td>74,073–251,664</td><td>0.39</td></tr><tr><td rowspan="3">language</td><td>Qwen2.5-Omni</td><td>3,817</td><td>1,936–3,817</td><td>107,461</td><td>107,461–107,461</td><td>1.00</td></tr><tr><td>DeSTA2.5-Audio</td><td>5,430</td><td>5,430–33,242</td><td>5,430</td><td>5,430–12,879</td><td>0.03</td></tr><tr><td>Audio Flamingo 3</td><td>19,538</td><td>19,538–19,538</td><td>102,782</td><td>48,173–102,782</td><td>1.00</td></tr></table>

Table 7: Bootstrap stability of the necessity windows of Table 4 (2,000 resamples of the test samples). Onsets are numbers of deactivated neurons $N ;$ P(selective) is the fraction of resamples in which the target task’s decline onset precedes the control tasks’.

## B.7 Layer-Matched Random Control

Cross-modal concept neurons concentrate in later layers (48–73% of each Qwen2.5-Omni emotion set lies in the last quarter of the layers), whereas the random sets of §4.3 are drawn uniformly across layers, so the cross-modal concept neuron set’s advantage could in principle reflect only where its neurons sit. We therefore repeat the steering test on Qwen2.5-Omni’s emotion task, the same setting as the other controls, with five random sets that match the cross-modal concept neuron set’s number of neurons in every layer. The cross-modal concept neuron set outranks all five layer-matched random sets for all five labels (rank test $p = 1 . 3 \times$ $1 0 ^ { - 4 } )$ and is significant for four of them under the prediction-interval test; its lead over the random sets in average target rate (0.17–0.35) is essentially unchanged from the uniform random sets (0.17–

0.38). Its advantage therefore does not come from its layer distribution.

## B.8 DeSTA2.5-Audio Transcript Input

To check what DeSTA2.5-Audio’s transcript input (§3) reveals, we ran its own transcription pipeline (voice-activity detection followed by Whisperlarge-v3 decoding) on all 500 clips of each task. For spoken language, every clip receives a transcript, written in the spoken language: judging the transcript’s language by its script (Chinese, Japanese, Korean) or by common function words (European languages), 83% match the label (100% for Chinese and Korean, 62–89% for the European languages and Japanese, where short transcripts often contain no decisive word), a lower bound on how often the answer is readable from text. For animal sounds, 94% of clips are rejected by the voice-activity detector and receive an empty transcript, and only about 1% of transcripts name the correct animal (e.g. “Woof!”). Emotion clips come from CREMA-D (Cao et al., 2014), whose fixed sentences are shared across emotions (57% of transcripts), and MELD (Poria et al., 2019), TV-show dialogue whose content can occasionally hint at the emotion; gender transcripts contain only the spoken sentence. The transcript thus offers a systematic text route to the answer only for spoken language.

## B.9 Answer Parsing

Models are prompted to answer with a single option letter. We take the first character of the stripped, lowercased response as the predicted letter, and additionally accept a leading “(x)” when it is the only option-shaped string in the response, since Audio Flamingo 3 frequently answers in the form “(a) sheep”. A response that yields no letter in $\{ a , b , c , d \}$ counts as a parse failure and is scored as incorrect (§4.4) or as not predicting the target label (§4.3), identically in every condition. Parse failures are rare. None of the three models produces any at baseline, and at each task’s target decline onset (Table 4) cross-modal-neuron deactivation produces none except in Audio Flamingo 3’s gender task (2.0% of responses, accounting for 0.8 of its 6.2 percentage-point accuracy drop), so the necessity results reflect changed answers rather than broken outputs. Under contrastive neuron steering (§4.3), parse failures rise at the largest coefficient, more for cross-modal concept neurons (7.9–16.4% of responses) than for random neurons (0.1–4.4%). Because a parse failure never counts as predicting the target label, this can only lower the cross-modal concept neurons’ target rate, biasing the comparison against rather than toward our steering results.

## B.10 Text Descriptions

Each label’s five paraphrased text descriptions follow a fixed template per task, substituting only the label word; the one exception is emotion, where the noun-form labels (fear, disgust) use a slightly different template than the adjective-form labels (happy, sad, angry) for grammaticality. One representative label per task:

• Emotion (angry): “The speaker feels angry.”; “The speaker sounds angry right now.”; “This is an angry voice.”; “The speaker is expressing an angry tone.”; “The overall emotion in the speaker’s voice is angry.”

• Emotion (fear, noun-form template): “The speaker feels fear.”; “The speaker’s voice is full of fear.”; “This voice conveys a sense of fear.”; “The speaker is expressing fear.”; “The overall emotion in the speaker’s voice is fear.”

• Animal (dog): “The sound in the audio is from a dog.”; “This audio contains the sound of a dog barking.”; “A dog is making the sound in this audio.”; “The animal making this sound is a dog.”; “This is the sound of a dog.”

• Gender (female): “The speaker in the audio is female.”; “This is a female voice.”; “The voice belongs to a female speaker.”; “The speaker is female.”; “The gender of the speaker is female.”

• Language (English): “The speaker is speaking English.”; “This audio is in the English language.”; “The language spoken in the audio is English.”; “The speaker is using English.”; “English is being spoken in this audio.”

All other labels within a task use the identical template with only the label word (and, for animal, the corresponding sound verb, e.g. barking → meowing for cat) substituted.

## C Additional Figures

This appendix collects figures referenced from §4 but not shown in the main text.

## C.1 Cross-Modal Neuron-Jaccard Overlap Matrices

Figures 5–7 give the full label-by-label neuron-Jaccard overlap matrix underlying the distinctiveness test in §4.2, one figure per model, with one heatmap per task. Rows are audio labels, columns are text labels; a cell is the Jaccard overlap between that audio label’s and that text label’s top-k neuron sets. The diagonal entries are the matched-label overlaps summarized in Table 5 (as a diagonal-tooff-diagonal ratio), Table 1 (as the matched-label share), and Table 2 (as a mean per-label Jaccard overlap, for all four tasks including gender), and are visibly higher than most off-diagonal entries in every task and model, consistent with the significant diagonal-vs-off-diagonal gap reported there.

## C.2 Steering Curves: the Same Label across Models

Figures 8 and 9 give DeSTA2.5-Audio’s and Audio Flamingo 3’s steering curves for the same label shown for Qwen2.5-Omni in Figure 3 (§4.3), emotion label “happy.” Both models also show a positive gap between cross-modal concept neurons and random neurons on this label, though smaller than Qwen2.5-Omni’s.

## C.3 Deactivation Curves for the Remaining Tasks

Figures 10–12 plot the same curves as Figure 4 (§4.4) for the remaining rows of Table 4: animal, gender, and language.

## D Reproducibility and Artifact Details

This appendix collects artifact, licensing, computational, and software details not included in the main text.

![](images/41c5f3d1dd0c68ac25172e1f4f17c099d541f25fa847ba03146a3ffd0df21091.jpg)

![](images/16e9b405ed8431e0019d26703b8d4b8b85e8a86f557bee883ec46b37c6225327.jpg)

![](images/979df35fedba8ed070d953bb99f2cb037ca69d553a6408b0455804434eee30c2.jpg)

![](images/27f1b554e1df4940f0b75089faf87536b2d4f75cc307556f04e35fc8403f4fc2.jpg)  
Figure 5: Neuron-Jaccard overlap matrices for Qwen2.5-Omni, by task.

## D.1 Artifact licenses

Qwen2.5-Omni-7B (Xu et al., 2025) is released under the Apache-2.0 license. Audio Flamingo 3 (Ghosh et al., 2025)’s code is released under the MIT license; its model weights are for noncommercial research use only. Neither DeSTA2.5- Audio (Lu et al., 2026) nor the SAKURA benchmark (Yang et al., 2025b) states an explicit license on its public Hugging Face or GitHub page; we use both strictly for the same non-commercial academic research purpose as their respective release papers.

## D.2 Consistency with intended use

All three models and the SAKURA benchmark are used exactly as released: as pretrained, frozenat-inference LALMs and an off-the-shelf audioattribute-recognition evaluation benchmark, respectively. We do not fine-tune, retrain, or redistribute any model weights or benchmark data; our steering and deactivation interventions (§4.3, §4.4) are applied only at inference time for scientific analysis, not as a proposed modification for deployment.

## D.3 Data screening

We do not collect, curate, or redistribute any new data; all audio and labels used in this paper come unmodified from the public SAKURA benchmark (§3). We therefore did not perform additional screening for personally identifying or offensive content beyond whatever screening SAKURA’s own creators already applied.

## D.4 Artifact documentation

Each of the four SAKURA tasks we use provides 500 audio instances, covering every label it defines: emotion (100 instances per label; 5 labels: angry, disgust, fear, happy, sad), gender (250 per label;

![](images/d790790120d34d348640ce6d297c17332648af04d112f485d849178c1be46e3d.jpg)

![](images/c86d63f06db7e8b69f8628ca52307a5a28e5ea898cf4130258c8e48ef8835006.jpg)

![](images/8b2ac1c0de414bde139c6682673cce7e0c8379f78aeb1692b433d8489199839f.jpg)

![](images/062ed815574c2705f9e3561d609b8f417c1f057de26546fca0e0f70d6494232b.jpg)  
Figure 6: Neuron-Jaccard overlap matrices for DeSTA2.5-Audio, by task.

female, male), animal sound (9 species, 40–68 instances per label: cat, cow, dog, frog, and sheep at 68 each, crow, hen, pig, and rooster at 40 each), and spoken language (8 languages, 61–64 instances per label: English, French, German, and Spanish at 64 each, Chinese, Italian, Japanese, and Korean at 61 each). SAKURA does not publish speaker demographic information beyond what each task’s own label encodes (e.g. binary gender); we did not have access to any further demographic or geographic characteristics of its speakers.

## D.5 Data statistics

For every task, neuron selection and evaluation use a stratified 20%/80% validation/test split over these 500 instances (fixed seed 0; §2), giving approximately 100 validation and 400 test instances per task, proportioned within each label. The overlap analyses (§4.1–§4.2) choose k on the validation split and then reselect the neuron sets on the test split. Neuron sets used for interventions are always identified on the validation split, and every result reported in §4 is computed on the test split; the one choice that uses the test split is the steering neuron-set size (Appendix B.1; Limitations).

## D.6 Model size and computational budget

Qwen2.5-Omni-7B’s language backbone (its “Thinker” module) is based on Qwen2.5-7B (7B parameters); we do not use its speech-generating “Talker” module. DeSTA2.5-Audio’s language backbone is Llama-3.1-8B-Instruct (8B parameters), frozen throughout its own training and throughout our experiments; its audio encoder is Whisper-large-v3. Audio Flamingo 3’s language backbone is Qwen2.5-7B (7B parameters); its authors do not report a total parameter count including its AF-Whisper audio encoder and adapter. All experiments in this paper are inference-only: we never train or fine-tune any of the three models, only extract their activations or intervene on them (steering, deactivation) at inference time. Qwen2.5- Omni-7B and DeSTA2.5-Audio ran on RTX 3090 GPUs (24GB VRAM); Audio Flamingo 3 ran on RTX A6000 GPUs (48GB VRAM). Reproducing the experiments reported in this paper takes roughly 800 GPU-hours on single GPUs, most of it for the deactivation sweeps (§4.4, Appendix B.1).

![](images/f9693adfc304cb0d3df0c983482f633a2476101a9081627c475c070db0b969d7.jpg)

![](images/032689593152c0c890a13e0a897c95c0375f4e7c37d0080e95a269de4e6cd26f.jpg)

![](images/37d7e993953899904d03d0e7812efad50c537da5b76c037fb74dd03303e03845.jpg)

![](images/d40d292f0d551c8a262728e2101ddfe0a17fc9476318447e09cbd9695a6332ea.jpg)  
Figure 7: Neuron-Jaccard overlap matrices for Audio Flamingo 3, by task.

## D.7 Hyperparameter search and reproducibility

§2 and §4.2–§4.4 describe the neuron-set size k’s search grid (a doubling grid, extended for §4.4) and selection criterion (Jaccard-maximizing on the validation split for §4.1–§4.2) at the point each is first used; Appendix B.1 gives the one setting not otherwise specified there: the neuron-set size used for the steering experiment in §4.3. All neuron selection and evaluation use the same fixed validation/test split (seed 0; described above) throughout.

## D.8 Descriptive statistics and single-run vs. averaged results

All three models generate with greedy, temperaturefree decoding (no sampling), so a given input and neuron intervention deterministically produces the same output; we do not report cross-seed generation variance for this reason. Three components of our pipeline are randomized and are reported accordingly: the Monte Carlo random-overlap baseline in Table 6 (100,000 samples); the five-trial random-neuron deactivation control in §4.4, whose curves report the mean across trials; and the five random-neuron control sets per label in §4.3, each drawn with a fixed seed; rather than averaging them, the significance tests use their spread as the null distribution (Appendix A.4). Every other number reported (accuracy, Jaccard overlap, cosine similarity) is a single deterministic evaluation pass over the fixed test split, not an average or a max over multiple runs.

![](images/fd04b9ae298849d9b2a90cbf39c26bb997348699e3cb8730cce94010ec77b5ea.jpg)  
Figure 8: Contrastive neuron steering toward the emotion label “happy” in DeSTA2.5-Audio, one panel per carrier label. x-axis: steering coefficient; y-axis: fraction of samples whose predicted answer is “happy”. Blue: cross-modal concept neurons; grey: median of five matched-size random neuron sets, with their min–max range shaded (the null).

## D.9 Existing software packages

Our implementation uses PyTorch (Paszke et al., 2019), Hugging Face Transformers (Wolf et al., 2020), SciPy (Virtanen et al., 2020) (for the hypergeometric and permutation, t, and rank significance tests in Appendix A), NumPy (Harris et al., 2020), librosa (McFee et al., 2015) (for audio loading and resampling), and Matplotlib (Hunter, 2007) (for figures). Because DeSTA2.5-Audio and Audio Flamingo 3 require conflicting dependencies, each of the three models ran in its own environment: librosa (0.11.0) is common to all three; DeSTA2.5- Audio and Audio Flamingo 3 share PyTorch 2.10.0, SciPy 1.15.3, NumPy 2.2.6, and Matplotlib 3.10.8, while Qwen2.5-Omni instead used PyTorch 2.8.0, SciPy 1.16.3, NumPy 2.3.4, and Matplotlib 3.10.7.

![](images/b98afae190f8085addd4a24222c20a6a69b0e0dacd5a7a3d6771f6b8850774a5.jpg)  
Figure 9: Contrastive neuron steering toward the emotion label “happy” in Audio Flamingo 3, one panel per carrier label. x-axis: steering coefficient; y-axis: fraction of samples whose predicted answer is “happy”. Blue: cross-modal concept neurons; grey: median of five matched-size random neuron sets, with their min–max range shaded (the null).

language (cross-modal concept neurons) control tasks (cross-modal concept neurons) language (random neuron, median and range of 5 trials)

target starts declining (>10% rel. drop) control starts declining (>10% rel. drop)

![](images/d9f410a91f6e2df6d3b5cc9c4668623dfd6950309f601875c0da169ef3bbd999.jpg)

![](images/1733b3a0ab46695f111460e93ca8cac54e35515af12077e233d705a899b32c14.jpg)

![](images/a82f1d3cd4995215823bda7d2f32dd1ec51f9696fc8a571113cfa46805ead777.jpg)  
Figure 11: Task-level deactivation on the gender task, for Qwen2.5-Omni, DeSTA2.5-Audio, and Audio Flamingo 3 (left to right). Solid lines: target task accuracy (blue) and mean control-task accuracy (orange) as a function of the number of deactivated neurons N (log scale). Dashed lines: matched-size random-neuron control (target task: median of the five random trials, with their min–max range shaded; control tasks: mean). Purple stars: audio-only neurons, a reference control (§4.4). Vertical dashed lines and the shaded band mark the target’s and controls’ decline onsets; a band is shown only when the target’s onset precedes the controls’.

![](images/44f39ac26a58d9b7f9a7f349bab9cb751b670f4085ccc31c45f9ce619c585c71.jpg)

![](images/cabfbf8ed7e0b7fa63f068fda76d8073c9f671b15ea7d32b1e29cd77785a0e3e.jpg)