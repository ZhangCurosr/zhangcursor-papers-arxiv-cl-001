# CaseLaw: A Policy-Decoupled Framework for Training-Free Content Moderation

Qi Lu<sup>1,2</sup>, Sen Li<sup>1</sup>, Likun Shi<sup>1</sup>, Xiaolong Jiang<sup>1∗</sup>

<sup>1</sup>Xiaohongshu <sup>2</sup>University of Science and Technology of China {luqi4, lisen2, shilikun, laige}@xiaohongshu.com

## Abstract

Content moderation is inherently policy-dependent: moderation boundaries are often subtle, subjective, and frequently updated, while conventional rule based or fine-tuning based systems require costly rule rewriting, model retraining, or system redeployment. We propose CaseLaw, a policy-decoupled and training-free framework for adaptive content moderation. CaseLaw reformulates moderation as precedent-based retrieval and voting, where multimodal content is transformed into evidence-preserved structured semantics, embedded into a stable semantic space, and classified by thresholded vot ing over retrieved precedent labels. We train NoteCap on a large-scale corpus of structurally annotated notes to ensure consistent semantic extraction and freeze it during subsequent policy adaptation, which requires no additional training. Policy adaptation is achieved by updating the semantic case library, including case labels and mappings, without retraining models or modifying prompts. To evaluate CaseLaw, we construct MFO-RedNotes, a benchmark for the male-female op position risk domain. We additionally provide a five category CoPro comparison. Experiments show that CaseLaw consistently outperforms competing baselines in both precision and recall across policy-dependent risk levels and broader safety categories, with category-wise precision/recall reaching up to 86.8%/90.1% on MFO-RedNotes across both policy settings and 94.5%/92.3% on CoPro. Our work highlights the potential of precedent-based reasoning for policy adaptive content moderation and ofers a practical direction toward more stable, transparent, and training-free moderation systems.

## Introduction

Content moderation has become a foundational mechanism for maintaining safe, inclusive, and trustworthy online ecosystems. Modern platforms must continuously regulate user-generated content at massive scale while balancing platform policies, community norms, legal requirements, and user expectations (Gillespie 2018; Gerrard 2020; Gorwa, Binns, and Katzenbach 2020). This challenge has become more urgent with the rapid deployment of large language models (LLMs) and generative AI systems, whose outputs may involve a broad spectrum of harms, including harassment, hate speech, misinformation, sexual content, selfharm, privacy violation, and other emerging sociotechnical risks (Shelby et al. 2023; Markov et al. 2023; Brown et al.

![](images/286cce8977d05189fe681c9ac865db889ec69e07b08d139ee7a61e6f7ae544a5.jpg)  
Figure 1: Comparison of policy adaptation in conventional moderation pipelines and CaseLaw. Policy changes in conventional pipelines can require rule rewriting, model retraining, and redeployment of updated moderation components. CaseLaw adapts by updating the external case library while keeping its deployed models fixed.

2020; Ouyang et al. 2022; Inan et al. 2023; Huang 2025). Unlike conventional classification tasks, content moderation is inherently policy-dependent: the same content may be judged diferently across platforms, cultural contexts, user groups, or temporal policy updates. Therefore, a practical moderation system must not only detect harmful content accurately, but also adapt rapidly when the definition, severity, or boundary of risk changes.

Current automated moderation pipelines largely follow two decision pathways: rule-guided LLM agents and supervised classification models. Rule-guided agents interpret policies in context, extending their coverage beyond explicitly enumerated patterns (Palla et al. 2025). Their generative decision process, however, can produce inconsistent judgments, particularly for borderline cases: repeated evaluations of identical content with an identical prompt may yield diferent outcomes. Conventional classifiers operationalize moderation policies through annotated datasets and supervised learning (Nobata et al. 2016; Markov et al. 2023). With fixed parameters and deterministic inference, these models provide repeatable labels and scores, but their learned decision boundaries do not readily accommodate policy revisions. Adapting them often requires retraining or rewriting policy prompt, and redeployment, resulting in long iteration cycles, limited flexibility, and substantial maintenance costs.

Figure 1 illustrates the limitations of conventional moderation pipelines, in which policy revisions necessitate rule rewriting, model retraining, and redeployment. Across both decision pathways, moderation quality also depends on the representation of policy-relevant evidence. LLM-based decision stages are sensitive to wording, formatting, input length, and the position of relevant information within the context (Sclar et al. 2024; Liu et al. 2024a), while repeatable classifier outputs likewise do not guarantee robustness to alternative expressions of the same underlying evidence. Although retrieval-augmented methods supply additional context through similar examples (Chen et al. 2024), similarity between raw examples may be insuficient to support finegrained, policy-dependent judgments. Such judgments often hinge on structured risk cues, rhetorical strategies, and subtle distinctions near policy boundaries that broad semantic resemblance fails to capture.

To address these challenges, we propose CaseLaw, a casedriven, policy-decoupled framework for training-free moderation adaptation. CaseLaw is motivated by the observation that the semantics of content under moderation remain relatively stable, whereas moderation policies, risk boundaries, and enforcement priorities evolve. CaseLaw formulates moderation as case-based semantic inference: it converts heterogeneous user generated content into structured semantic representations, retrieves relevant precedents in a fixed embedding space, and aggregates their policy-dependent labels. Policy changes are expressed as revisions to an external semantic case library, including updates to case labels, case-specific thresholds, and policy-to-label mappings. By moving adaptation from model optimization to explicit caselevel knowledge, CaseLaw enables lightweight and controllable policy updates while keeping the representation models fixed. To improve the consistency and fidelity of structured semantic extraction, we annotate 100,000 notes with structured semantic fields and train NoteCap, a dedicated extraction model. NoteCap remains frozen and is reused across policy updates, preserving the semantic evidence required for precedent retrieval while moderation behavior adapts through case-library updates without model training.

We instantiate and evaluate CaseLaw in the genderpolarization risk domain, where violations often depend on nuanced framing, implicit hostility, group opposition, and contextual escalation. We construct MFO-RedNotes, a policy-dependent multimodal moderation benchmark comprising 50,000 notes across four risk levels: Safe, Low-risk, Mid-risk, and High-risk. Of these notes, 10,000 receive diferent labels following a policy change, enabling evaluation of adaptation to revised decision boundaries while holding the underlying content fixed. Experiments show that CaseLaw consistently outperforms zero-shot prompting, fixed few-shot prompting, fine-tuning, and retrievalaugmented baselines across risk levels under both policy settings, supporting the efectiveness of policy-decoupled, case-based moderation.

Our contributions are summarized as follows:

• We propose CaseLaw, a case-driven moderation framework that decouples policy adaptation from model optimization through an editable semantic case library, enabling policy updates without retraining deployed models or rewriting prompts.

• We develop an evidence-preserving structured representation and train NoteCap on 100,000 annotated notes to support precedent retrieval and thresholded label voting. The representation models remain fixed throughout all subsequent policy updates.

• We construct MFO-RedNotes with 50,000 notes and policy-induced label changes for 10,000 of them, and demonstrate CaseLaw’s efectiveness across four risk levels before and after policy change.

## Related Work

Automated content moderation and AI safety. Automated content moderation scales platform governance through keyword filtering, hash matching, and supervised classifiers. Automated approaches remain sensitive to linguistic variation and contextual ambiguity (Nobata et al. 2016; Gorwa, Binns, and Katzenbach 2020). Subsequent machine learning approaches addressed harassment, abusive language, hate speech, and ofensive content (Nobata et al. 2016; Gambäck and Sikdar 2017). Word embeddings capture lexical semantics (Mikolov et al. 2013; Pennington, Socher, and Manning 2014), while Transformer architectures and bidirectional pretraining support contextual representations (Vaswani et al. 2017; Devlin et al. 2019). However, greater semantic capacity does not eliminate policy dependence: the same content can receive diferent labels under diferent governance requirements. Moderation decisions remain shaped by policy taxonomies, community expectations, cultural context, and evolving definitions of harm (Palla et al. 2025; Huang 2025).

Policy operationalization and prompt-based moderation. Operationalizing abstract policies is central to automated moderation. Traditional pipelines translate policies into annotation guidelines and labeled datasets for supervised training (Markov et al. 2023; Palla et al. 2025). LLMs support an alternative approach in which policies are supplied directly through prompts, allowing moderation criteria to change without constructing a new training dataset for every policy revision (Brown et al. 2020; Ouyang et al. 2022; Palla et al. 2025). Chain-of-thought prompting and ReAct support multi-step reasoning and the integration of reasoning with actions, respectively (Wei et al. 2022; Yao et al. 2023). However, LLM judgments are sensitive to prompt formatting, input length, and the position of relevant information (Sclar et al. 2024; Liu et al. 2024a). This sensitivity complicates consistent enforcement across policy versions: even minor changes in policy wording or presentation can alter moderation decisions for otherwise unchanged content.

![](images/47f1dfc19f6a1924a8c08a910eb263c7f1cca09a448ce422e130ca697d8cf57c.jpg)  
Figure 2: Overview of CaseLaw. NoteCap converts text, images, video, and audio into evidence-preserving structured fields. A frozen encoder embeds the serialized fields, enabling retrieval of precedent cases from an editable library. Each case whose similarity meets or exceeds its stored threshold casts a single vote based on its current risk label, without invoking an LLM classifier after retrieval. Policy changes are implemented by updating precedent labels and metadata while keeping the deployed representation models fixed.

Retrieval-augmented generation and case-based moderation. Retrieval-augmented generation (RAG) augments pretrained language models with external non-parametric memory, enabling models to access updatable knowledge without changing their parameters (Lewis et al. 2020). RAG has been widely studied for knowledge-intensive tasks and has been shown to improve factuality, transparency, and adaptability by retrieving relevant external evidence (Gao et al. 2024). Dense retrieval methods further improve the quality of retrieved evidence through representation learning and contrastive training (Lin et al. 2023), while libraries such as Faiss enable eficient large-scale similarity search over embedding spaces (Douze et al. 2026). In content moderation, Class-RAG (Chen et al. 2024) uses a library of safe and unsafe examples to support real-time semantic hotfixing and improve robustness to adversarial perturbations.

## Method

As illustrated in Figure 2, CaseLaw transforms multimodal content into evidence-preserving structured representations, embeds them in a retrieval-oriented semantic space, and retrieves similar precedents from an editable case library. Each retrieved precedent casts one vote for its current policydependent label only when its similarity to the query meets or exceeds its own stored threshold. The extractor, encoder, extraction prompt, retrieval configuration, and voting rule remain fixed throughout deployment. Policy adaptation proceeds solely through updates to the external case library, allowing moderation behavior to track changing policies without retraining the deployed models.

## Problem Formulation

Let $x \in \mathcal { X }$ denote a user-generated content instance, which may contain multimodal content such as text, images, videos,

and audio. CaseLaw maps x to a moderation classification result defined over a user-specified label space:

$$
\mathcal { V } _ { t } = \{ \ell _ { 1 } ^ { ( t ) } , \ell _ { 2 } ^ { ( t ) } , \dots , \ell _ { M _ { t } } ^ { ( t ) } \} ,\tag{1}
$$

where each $\ell _ { j } ^ { ( t ) }$ denotes a policy-dependent moderation label at moderation time t, and $M _ { t }$ is the number of labels under the current policy configuration. The label space is not assumed to be fixed and can be customized for diferent moderation scenarios, including risk levels, violation categories, governance actions, or domain-specific policy outcomes.

At moderation time t, the efective policy state is denoted as $p _ { t }$ . A change from $p _ { t }$ to $p _ { t + 1 }$ may introduce new labels, revise the semantic boundary of existing labels, or alter the mapping between content patterns and moderation outcomes. Let $\mathcal { D } _ { t } = \{ ( x _ { i } , y _ { i } ^ { ( t ) } ) \} _ { i = 1 } ^ { n _ { t } }$ denote the evaluation distribution under policy state $p _ { t } ,$ where $y _ { i } ^ { ( t ) } \in \mathcal { V } _ { t }$ is the policydependent ground-truth label. The goal of CaseLaw is to maintain accurate moderation decisions across a sequence of policy states without retraining the deployed representation models or modifying the extraction prompt.

Formally, let $f _ { \Theta , \pi }$ denote the moderation function, where Θ represents the parameters of all frozen model components and π denotes the fixed prompt template. Given a policy state $p _ { t } .$ , let $\mathcal { D } _ { t }$ denote the corresponding evaluation distribution, and let $\mathcal { V } _ { t }$ be the policy-dependent label space at time t. For a case library $\mathcal { L } _ { t }$ , the expected classification accuracy under policy state $p _ { t }$ is defined as

$$
\operatorname { A c c } _ { t } ( \mathcal { L } _ { t } ) = \mathbb { E } _ { ( x , y ^ { ( t ) } ) \sim \mathcal { D } _ { t } } \left[ \mathbb { I } \left\{ f _ { \Theta , \pi } ( x ; \mathcal { L } _ { t } , \mathcal { V } _ { t } ) = y ^ { ( t ) } \right\} \right] ,\tag{2}
$$

where fixed Θ and π are suppressed in the notation and $\mathbb { I } \{ \cdot \}$ is the indicator function, which equals 1 if the predicted label matches the policy-dependent ground-truth label and 0 otherwise.

CaseLaw aims to maintain high classification performance across dynamic policy states by updating only the external case library:

$$
\operatorname* { m a x } _ { \{ \mathcal { L } _ { t } \} _ { t = 1 } ^ { T } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { A c c } _ { t } ( \mathcal { L } _ { t } ) ,\tag{3}
$$

subject to the training-free and prompt-invariant constraints: $\Theta _ { t + 1 } = \Theta _ { t } , \pi _ { t + 1 } = \pi _ { t } ,$ , while $\mathcal { V } _ { t + 1 }$ is allowed to difer from $\mathcal { V } _ { t }$

## Structured Semantic Extraction

Content moderation often depends on subtle linguistic cues such as group attribution, sarcasm, metaphor, emotionally charged expressions, and risk-bearing lexical choices. Directly embedding the full raw content may introduce substantial noise, while embedding a purely abstractive LLM summary may remove precisely the words that determine risk severity. CaseLaw therefore introduces an evidencepreserved structured semantic extraction module.

We obtain NoteCap through supervised training on 100,000 notes annotated with the same structured schema used at inference. The supervision targets semantic organization and evidence preservation rather than a particular policy’s final risk labels. Once trained, NoteCap is frozen across policy versions.

For multimodal inputs, OCR extracts text appearing in images, and ASR transcribes speech in video and audio. These textual observations are combined with the original images and sampled video frames as multimodal inputs to evidence selection. OCR and ASR supplement visual evidence. This preserves contextual cues that may not be recoverable from transcripts alone.

For each input x, the frozen NoteCap extractor produces astructured representation

$$
a _ { x } = S _ { \theta } ( x ) = \bigl ( E _ { x } , C _ { x } , T _ { x } , B _ { x } , M _ { x } \bigr ) ,\tag{4}
$$

where $E _ { x }$ denotes evidence spans, $C _ { x }$ denotes the core claim, $T _ { x }$ denotes the topic, $B _ { x }$ denotes the subtopic, and $M _ { x }$ denotes the expression type. Evidence spans are selected from the original content and are required to preserve riskbearing phrases, entity references, and potentially controversial claims. In practice, CaseLaw retains a small set of suspicious original spans, typically six to eight sentences or fragments, and combines them with a concise core claim of approximately 150–200 words.

The extraction module is designed to optimize a tradeof between compression and evidence preservation. Let ${ \mathcal G } ( x ) = \{ g _ { j } \} _ { j = 1 } ^ { m }$ be candidate textual units extracted from the multimodal content, including text, images, videos, and audio information. CaseLaw selects evidence spans according to

$$
\begin{array} { r } { E _ { x } = \mathrm { T o p } _ { m _ { e } } \left\{ g _ { j } \in \mathcal { G } ( x ) : \rho _ { \theta } ( g _ { j } , x ) \right\} , } \end{array}\tag{5}
$$

where $\rho _ { \boldsymbol { \theta } } ( g _ { j } , \boldsymbol { x } )$ is the extractor’s estimated relevance of candidate span $g _ { j }$ to potential moderation risk, and $m _ { e }$ controls the number of retained evidence spans.

The final retrieval text is constructed by serializing the structured representation $a _ { x } .$ , which jointly contains verbatim evidence and abstractive semantic components. Specifically, the evidence spans preserve lexical and rhetorical signals, while the core claim and structured attributes provide a compact semantic abstraction. This design difers from standard summarization-based retrieval: rather than relying solely on an abstractive summary, CaseLaw preserves riskbearing original expressions while maintaining a concise semantic representation. As a result, CaseLaw reduces semantic drift caused by purely abstractive summaries and improves the retrieval of moderation-relevant precedent cases.

## Case Library Construction

The case library serves as the policy-bearing knowledge layer of CaseLaw. Each case stores its original content, structured semantics, embedding representation, moderation label, decision rationale, and similarity threshold. Formally, at moderation time t, the case library is denoted as

$$
\mathcal { L } _ { t } = \{ c _ { i } \} _ { i = 1 } ^ { N _ { t } } , \quad c _ { i } = ( u _ { i } , a _ { i } , z _ { i } , y _ { i } , r _ { i } , \tau _ { i } ) ,\tag{6}
$$

where $u _ { i }$ is the original case content, $a _ { i }$ is the structured semantic representation, $z _ { i }$ is the retrieval embedding, $y _ { i } \in$ $\mathcal { V } _ { t }$ is the policy-dependent moderation label, $r _ { i }$ is the decision rationale, and $\tau _ { i }$ is the case-specific similarity threshold for voting eligibility. This formulation allows policy updates to be implemented through case-level revisions, such as label correction, threshold recalibration, or case insertion, without modifying model parameters or rewriting the prompt.

Case-specific thresholds account for diferences in how strongly a case’s semantics indicate its associated risk. Cases with implicit risk cues require more selective matching: a higher threshold restricts their votes to closer semantic matches and helps avoid incorrectly flagging safe notes. Before admitting a training-set case, we calibrate its threshold on the validation set, which contains notes labeled as safe or violating under the applicable policy. We assess candidate threshold values according to their efects on violating-note recall and the false-positive rate on safe notes, and select an operating point for each case that balances these two criteria.

For each case $c _ { i }$ , CaseLaw applies the same structured extraction and serialization procedure as used for the query. Since risk-bearing lexical evidence has already been preserved as evidence spans during structured semantic extraction, CaseLaw does not introduce an additional sparse lexical representation. Instead, the serialized representation $\tau ( a _ { i } )$ is directly encoded into a dense semantic embedding:

$$
z _ { i } = \mathcal { E } _ { \phi } ( \tau ( a _ { i } ) ) , \quad z _ { i } \in \mathbb { R } ^ { D } ,\tag{7}
$$

where ${ \mathcal { E } } _ { \phi }$ is a frozen embedding encoder. Because $\tau ( a _ { i } )$ contains both verbatim evidence spans and abstractive semantic fields, the resulting embedding captures not only the overall semantic meaning of the case but also the risk-bearing expressions preserved from the original content.

Given a query x, CaseLaw retrieves similar precedent cases through approximate nearest neighbor (ANN) search (Arya et al. 1998) over the current case library. Each case embedding is indexed in advance, and the query embedding $z _ { x }$ is used to eficiently retrieve the top-k nearest cases according to embedding similarity. Since the embedding input is constructed from evidence-preserved structured semantics, including evidence spans, topic, subtopic, expression type, and core claim, ANN retrieval is performed over a moderation-oriented semantic space rather than raw content alone. The retrieved precedent set is then obtained by

$$
{ \mathcal { N } } _ { k } ( x ) = \mathrm { T o p } _ { k } \left\{ \mathrm { S i m } ( x , c _ { i } ) : c _ { i } \in { \mathcal { L } } _ { t } \right\} .\tag{8}
$$

Benefiting from the evidence-preserved structured semantic extraction, the retrieval representation retains both risk-bearing original expressions and compact semantic attributes, enabling semantically similar content to be retrieved more precisely. The retrieved top-k cases then serve as caselevel evidence for the downstream voting module, in which their labels determine the prediction, while the stored rationales remain available for auditing without contributing additional votes.

## Thresholded Precedent Voting

CaseLaw computes its final decision directly from the current labels of retrieved precedents. For a query embedding $z _ { x } .$ , let $s _ { i } = \cos ( z _ { x } , z _ { i } )$ . Each retrieved case $c _ { i }$ participates in voting only if its similarity score is greater than or equal to its own stored threshold $\tau _ { i } .$ . The eligible voting set is

$$
\mathcal { V } _ { k } ( x ) = \{ c _ { i } \in \mathcal { N } _ { k } ( x ) : s _ { i } \geq \tau _ { i } \} .\tag{9}
$$

Each eligible case contributes one vote to its current label. Let ${ \mathcal { V } } _ { \ell } ( x ) = \{ c _ { i } \in { \mathcal { V } } _ { k } ( x ) : y _ { i } ^ { ( t ) } = \ell \}$ denote the eligible cases supporting label $\ell ,$ and let $n _ { \ell } = \overset { \cdot } { | \nu _ { \ell } ( x ) }$ | be their vote count. The final label is selected by

$$
\hat { y } ^ { ( t ) } = \underset { \ell \in \mathcal { V } _ { t } : n _ { \ell } > 0 } { \operatorname { l e x } } \left( n _ { \ell } , \frac { \sum _ { c _ { i } \in \mathcal { V } _ { \ell } ( x ) } s _ { i } } { n _ { \ell } } \right) ,\tag{10}
$$

where lexicographic maximization prioritizes the vote count and uses the mean similarity of the supporting eligible cases to break ties. If no retrieved case meets its threshold, the instance is referred for human review.

We mitigate residual label ties in both vote count and mean similarity by adjusting the retrieval depth k and expanding the case library. With k = 16 and a library of 40,000 cases, we observed no ties among the highest-ranked labels in our evaluation. Any remaining cases tied on both criteria are referred for human review.

## Experiments

## Experimental Setup

Datasets. We construct MFO-RedNotes (Male-Female Opposition RedNotes), a multimodal content moderation dataset collected from publicly available posts on the RedNotes platform. The dataset focuses on male-female opposition as a representative risk domain in community atmosphere governance, where harmfulness is often expressed through implicit group attribution, emotionally charged claims, genderbased antagonism, sarcastic framing, or value-laden narratives rather than explicit toxic language. MFO-RedNotes contains 50,000 public posts, each consisting of multiple content fields, including post title, main textual content, hashtags, images, videos, and a policy-dependent risk label. The risk labels are annotated by 20 expert content moderators according to the corresponding moderation policy for the male-female opposition risk domain. To model policy changes involving fine-grained exception rules, each instance is additionally annotated with an exemption indicator and an exemption rationale, both of which are used exclusively to define policy-change evaluation settings and are never provided as input to the moderation models. To support model development and evaluation, we split the dataset into 40,000 training examples, 5,000 validation examples, and 5,000 test examples. Policy revisions change the labels of 10,000 notes, or 20% of the full benchmark, while their content and split assignments remain unchanged. The original and revised labels enable paired evaluation of moderation performance before and after a policy change. Annotation proceeds in five rounds, in each of which a randomly selected 10% of the annotations undergo double-blind review, with reannotation required unless agreement exceeds 95%. Appendix B gives the four-level taxonomy and four expert-defined exemption rules. This dataset provides a realistic benchmark for evaluating whether moderation systems can handle multimodal, policy-sensitive, and boundary-case content under evolving governance requirements.

CoPro (Liu et al. 2024b) is a public content safety dataset containing unsafe prompts and their corresponding safe rewritten prompts. It covers seven harmful categories. We evaluate a five-category CoPro benchmark covering Safe, Sexual, Hate, Violence, and Shocking, using a balanced validation set of 5,000 instances with 1,000 per category.

Baselines. We compare CaseLaw with five moderation baselines:

• Zero-shot. A frozen model predicts labels from policyinformed prompts without retrieved examples.

• Fixed few-shot. Predictions use retrieved examples from a case library that stays fixed across policy changes.

• Fine-tuning. The model undergoes full-parameter finetuning with policy information in the training context.

• Class-RAG (Chen et al. 2024). A generative fine-tuned classifier uses policy context and examples retrieved from an editable case library.

• Llama Guard (Inan et al. 2023). A general-purpose safeguard model is evaluated by mapping its safety categories to the benchmark labels.

Implementation Details. We use Qwen3.5-35B-A3B for the Zero-shot, Fixed few-shot, and Fine-tuning languagemodel backbones and for the reference structured extractor. Fine-tuning uses LLaMA-Factory with full-parameter updates. Zero-shot receives policy information explicitly. CaseLaw’s fixed extraction template describes the semantic schema without encoding the current policy, and precedent labels supply policy-specific supervision at inference. CaseLaw and Fixed few-shot share the same initial library. Fixed few-shot keeps its initial precedents unchanged across policy states, whereas CaseLaw updates afected case labels and metadata.

Table 1: Comparison on MFO-RedNotes before and after policy change. Precision and recall are expressed as percentages for four policy-dependent risk levels. Yellow highlights the best score, and green highlights the strongest baseline in each column. Superscripts report absolute gains in percentage points over the strongest baseline within the same policy setting.
<table><tr><td rowspan="2">Method</td><td colspan="2">Safe</td><td colspan="2">Low-risk</td><td colspan="2">Mid-risk</td><td colspan="2">High-risk</td></tr><tr><td>Prec (%)</td><td>Rec (%)</td><td>Prec (%)</td><td>Rec (%)</td><td>Prec (%)</td><td>Rec (%)</td><td>Prec (%)</td><td>Rec (%)</td></tr><tr><td colspan="9">Original policy</td></tr><tr><td>Zero-shot</td><td>56.6</td><td>54.5</td><td>39.7</td><td>41.0</td><td>52.0</td><td>53.0</td><td>50.7</td><td>50.7</td></tr><tr><td>Fixed few-shot</td><td>71.5</td><td>72.2</td><td>55.4</td><td>56.0</td><td>64.6</td><td>63.4</td><td>68.1</td><td>67.8</td></tr><tr><td>Fine-tuning</td><td>78.1</td><td>85.0</td><td>73.3</td><td>71.0</td><td>76.3</td><td>74.0</td><td>81.2</td><td>76.2</td></tr><tr><td>Class-RAG</td><td>81.0</td><td>85.3</td><td>75.8</td><td>74.5</td><td>78.5</td><td>76.3</td><td>82.5</td><td>78.3</td></tr><tr><td>Llama Guard</td><td>66.3</td><td>65.7</td><td>49.6</td><td>52.3</td><td>58.8</td><td>56.7</td><td>60.3</td><td>60.3</td></tr><tr><td>CaseLaw (ours)</td><td> ${ \mathbf { 8 5 . 3 } ^ { \uparrow \ 4 . 3 } }$ </td><td> ${ \bf 8 8 . 3 ^ { \dagger } \Sigma ^ { 3 . 0 } }$ </td><td> ${ \bf 8 2 . 9 ^ { \uparrow \cdot 7 . 1 } }$ </td><td> ${ \bf 7 6 . 0 ^ { \dagger } \mu ^ { 1 . 5 } }$ </td><td> ${ \bf 8 4 . 8 ^ { \dagger } \Sigma ^ { 6 . 3 } }$ </td><td> ${ \bf 8 1 . 0 ^ { \dagger } \Sigma ^ { 4 . 7 } }$ </td><td> ${ \mathbf { 8 5 . 3 } ^ { \uparrow \ 2 . 8 } }$ </td><td> $\mathbf { 9 0 . 1 ^ { \uparrow \downarrow 1 . 8 } }$ </td></tr><tr><td colspan="9">After policy change</td></tr><tr><td>Zero-shot</td><td>57.2</td><td>55.3</td><td>41.3</td><td>39.4</td><td>39.7</td><td>43.5</td><td>45.4</td><td>46.2</td></tr><tr><td>Fixed few-shot</td><td>64.2</td><td>62.1</td><td>47.9</td><td>53.4</td><td>47.6</td><td>47.0</td><td>56.6</td><td>52.5</td></tr><tr><td>Fine-tuning</td><td>66.7</td><td>67.5</td><td>65.2</td><td>63.4</td><td>61.1</td><td>65.4</td><td>67.5</td><td>66.2</td></tr><tr><td>Class-RAG</td><td>68.2</td><td>69.3</td><td>65.1</td><td>62.7</td><td>62.3</td><td>66.7</td><td>68.3</td><td>65.8</td></tr><tr><td>Llama Guard</td><td>61.0</td><td>59.5</td><td>44.6</td><td>49.2</td><td>43.7</td><td>42.3</td><td>51.9</td><td>49.3</td></tr><tr><td>CaseLaw (ours)</td><td> $\mathbf { 8 6 . 8 ^ { \uparrow \cdot 1 8 . 6 } }$ </td><td> $\mathbf { 8 8 . 0 } ^ { \uparrow \mathrm { ~ 1 8 . 7 } }$ </td><td> $\mathbf { 7 9 . 9 ^ { \uparrow \cdot 1 4 . 7 } }$ </td><td> ${ \bf 7 4 . 1 ^ { \dagger } \cdot ^ { 1 0 . 7 } }$ </td><td> $\mathbf { 8 0 . 4 } ^ { \uparrow \uparrow 1 8 . 1 }$ </td><td> $8 5 . 4 ^ { \uparrow \uparrow 1 8 . 7 }$ </td><td> $8 4 . 4 ^ { \uparrow \ 1 6 . 1 }$ </td><td> $\mathbf { 8 3 . 8 ^ { \uparrow \cdot 1 7 . 6 } }$ </td></tr></table>

Table 2: Comparison on the five-category CoPro benchmark. Precision and recall are expressed as percentages. Yellow highlights the best score, and green highlights the strongest baseline in each column. Superscripts report absolute gains in percentage points over the strongest baseline. Evaluation uses 5,000 validation instances, with 1,000 instances per category.
<table><tr><td rowspan="2">Method</td><td colspan="2">Safe</td><td colspan="2">Sexual</td><td colspan="2">Hate</td><td colspan="2">Violence</td><td colspan="2">Shocking</td></tr><tr><td>Prec (%)</td><td> $\operatorname { R e c } \left( \% \right)$ </td><td>Prec (%)</td><td>Rec (%)</td><td>Prec (%)</td><td>Rec (%)</td><td>Prec (%)</td><td>Rec (%)</td><td>Prec (%)</td><td>Rec (%)</td></tr><tr><td>Zero-shot</td><td>48.4</td><td>51.2</td><td>51.8</td><td>49.1</td><td>50.9</td><td>55.8</td><td>53.8</td><td>60.4</td><td>56.6</td><td>43.9</td></tr><tr><td>Fixed few-shot</td><td>65.6</td><td>68.2</td><td>71.7</td><td>71.3</td><td>65.4</td><td>64.7</td><td>70.5</td><td>76.1</td><td>66.2</td><td>59.3</td></tr><tr><td>Fine-tuning</td><td>80.0</td><td>82.0</td><td>84.7</td><td>83.2</td><td>80.6</td><td>84.8</td><td>83.1</td><td>85.9</td><td>81.7</td><td>74.1</td></tr><tr><td>Class-RAG</td><td>83.0</td><td>84.3</td><td>87.8</td><td>87.0</td><td>82.8</td><td>87.2</td><td>86.2</td><td>88.0</td><td>84.2</td><td>77.4</td></tr><tr><td>Llama Guard</td><td>82.2</td><td>83.1</td><td>88.1</td><td>87.0</td><td>81.6</td><td>84.3</td><td>87.0</td><td>82.3</td><td>82.6</td><td>84.5</td></tr><tr><td>CaseLaw (ours)</td><td> $8 5 . 4 ^ { \uparrow 2 . 4 }$ </td><td> ${ \bf 8 8 . 2 ^ { \dagger } \Sigma ^ { 3 . 9 } }$  —</td><td> $\mathbf { 9 1 . 0 } ^ { \uparrow \ 2 . 9 }$ </td><td> $\mathbf { 9 2 . 3 } \uparrow \ 5 . 3$ </td><td> ${ \bf 8 8 . 0 ^ { \dag } \Sigma ^ { 5 . 2 } }$ </td><td> $\mathbf { 9 0 . 0 ^ { \uparrow } } \ ^ { 2 . 8 }$ </td><td> $\mathbf { 9 4 . 5 ^ { \uparrow \cdot 7 . 5 } }$ </td><td> $\mathbf { 9 2 . 2 } \uparrow \uparrow 4 . 2$ </td><td> $\mathbf { 9 1 . 2 ^ { \uparrow } \Sigma ^ { \ 7 . 0 } }$ </td><td> $\mathbf { 8 7 . 0 } ^ { \uparrow \ 2 . 5 }$ </td></tr></table>

All representation experiments use Qwen3-Embedding-8B. The NoteCap stability study uses Qwen3-Embedding-8B across all extraction variants. NoteCap training is a one-time preparation step, not a policy-update operation. CaseLaw uses the thresholded voting rule defined in Eq. (10), so it has no classification-model backbone. Llama Guard is evaluated using its original safeguard checkpoint.

We use k = 16 for CaseLaw in Table 1. The experiments use a server with 32 NVIDIA H800 GPUs.

## Performance Analysis

Table 1 reports class-wise precision and recall as percentages across four policy-dependent risk levels on MFO-RedNotes. Under the original policy, CaseLaw achieves the best results in all eight comparisons, reaching 85.3%/88.3% precision/recall on Safe and 85.3%/90.1% on High-risk. Its gains over the strongest baseline in each column range from 1.5 to 11.8 percentage points. These results demonstrate that CaseLaw improves both the identification of violations and the limitation of false positives.

After policy change, CaseLaw leads all eight precision/recall comparisons, with gains of 10.7–18.7 percentage points over the strongest baseline in each column. It obtains 86.8%/88.0% on Safe, 79.9%/74.1% on Lowrisk, 80.4%/85.4% on Mid-risk, and 84.4%/83.8% on Highrisk. The largest margin is 18.7 percentage points in both Safe recall and Mid-risk recall, where Class-RAG reaches 69.3% and 66.7%. Averaged equally across the four classes, CaseLaw’s precision and recall decrease from 84.6%/83.9% to 82.9%/82.8%, respectively, indicating that it retains strong overall performance under the revised policy boundaries.

The baselines show larger declines under policy change. Fine-tuning’s macro-averaged precision/recall falls from 77.2%/76.6% to 65.1%/65.6%, while Class-RAG declines from 79.5%/78.6% to 66.0%/66.1%. Fixed few-shot also degrades, with Mid-risk precision/recall falling from 64.6%/63.4% to 47.6%/47.0%. These changes highlight the dificulty of adapting to fine-grained exemptions when the surface content remains unchanged. CaseLaw’s stronger post-change performance stems from its ability to preserve stable case-level semantics while adapting the evolving mapping between these semantics and policy-dependent risk labels through case-based judging.

![](images/7263d20976f9ed854e839f2e053a09f06f553296707669de1242ee0f8035d3d4.jpg)  
(a)

![](images/ca9507b2134f9e973878ea8087899151a33a385d5870f43bdf2eebf18cb6a35a.jpg)  
(b)

![](images/831dc42733ae4166c80cbe60811cc36207d2dd230293b10b4b421a25003c027a.jpg)  
(c)  
Figure 3: Impact of structured semantic extraction on embedding distributions. We visualize 1,000 safe notes and 1,000 risk notes from MFO-RedNotes, where risk includes Mid-risk and High-risk instances. Compared with direct embedding of raw notes and summary-only representations, NoteCap produces more compact intra-class clusters and clearer inter-class separation. The concentrated risk regions further indicate that structured semantic extraction can form risk-oriented semantic anchors fo more accurate downstream case retrieval.

Table 2 reports results across five CoPro categories. CaseLaw achieves the highest precision and recall in all ten category-level comparisons, with gains of 2.4–7.5 percentage points over the strongest baseline in each column. Its macro-averaged precision reaches 90.02%, exceeding Class-RAG which is 84.80% by 5.22 percentage points, while its macro-averaged recall reaches 89.94%, surpassing Class-RAG which is 84.78% by 5.16 percentage points. Class-RAG is the strongest baseline for both aggregate metrics. CaseLaw attains 94.5%/92.2% precision/recall on Violence and 91.2%/87.0% on Shocking, with recall gains of 4.2 and 2.5 percentage points, respectively. Its highest category-wise recall is 92.3% on Sexual.

## Ablation Studies

## Structured Representations

We study how semantic representation afects both moderation performance and its consistency across repeated runs. The MFO-RedNotes ablation uses a balanced subset of 1,000 Safe and 1,000 Unsafe notes. Direct splicing directly embeds a fixed serialization of the original note content, without generative rewriting. Summary only uses an LLM to reorganize each note into a compact semantic summary before embedding. NoteCap instead preserves risk-bearing evidence within structured semantic fields, including topic, expression type, and core claim. All representations use the same Qwen3- Embedding-8B encoder. Figure 3 visualizes their embedding distributions with t-SNE (van der Maaten and Hinton 2008). Table 3 reports precision and recall averaged over

Table 3: Representation ablation on MFO-RedNotes and CoPro. The MFO-RedNotes subset contains 1,000 Safe and 1,000 Unsafe notes. Prec and Rec are means over five runs (%); Var(P) and Var(R) are sample variances in squared percentage points (pp<sup>2</sup>).
<table><tr><td rowspan="2">Method</td><td colspan="3">Safe</td><td colspan="4">Unsafe</td></tr><tr><td></td><td>Prec Rec Var(P) Var(R)|Prec Rec Var(P) Var(R)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">MFO-RedNotes</td></tr><tr><td>Direct splicing Summary only</td><td>66.068.8 73.074.5</td><td></td><td>0.00 0.09</td><td>0.00 0.09</td><td>67.4 64.6 74.072.5</td><td>0.00 0.09</td><td></td><td>0.00 0.09</td></tr><tr><td>NoteCap</td><td>85.6 88.0</td><td></td><td>0.00</td><td>0.00</td><td>87.7 85.2</td><td></td><td>0.00</td><td>0.00</td></tr><tr><td colspan="9">CoPro</td></tr><tr><td>Direct splicing Summary only</td><td>61.066.9</td><td></td><td>0.00 0.11</td><td>0.00 0.12</td><td>71.365.8 77.372.6</td><td></td><td>0.00 0.07</td><td>0.00</td></tr><tr><td>NoteCap</td><td>68.2 73.3 85.4 88.4</td><td></td><td>0.00</td><td>0.00</td><td>90.5 87.9</td><td></td><td>0.00</td><td>0.12 0.00</td></tr></table>

five experimental runs, together with their sample variances.   
Appendix G defines the aggregation.

Figure 3 shows substantial overlap among raw-note representations, improved semantic grouping after summarization, and more compact risk-oriented regions with NoteCap. Table 3 distinguishes representation quality from run-to-run stability. Direct splicing yields the lowest precision and recall on both datasets despite producing stable results, whereas the Summary method improves both metrics but exhibits fluctuations attributable to model instability. NoteCap further improves precision and recall for both the Safe and Unsafe classes, with gains over Summary only of 12.6–13.7 percentage points on MFO-RedNotes and 13.2–17.2 percentage points on CoPro. Across five runs, NoteCap reports precision and recall variances of $0 . 0 0 \mathrm { p p ^ { 2 } }$ for both classes on both datasets. These results support NoteCap’s ability to concentrate moderation-relevant semantics while maintaining stable aggregate performance under the evaluated configurations.

Table 4: Five-run extraction stability and moderation performance on the MFO-RedNotes validation split. Evaluation uses the original policy and voting depth $k = 1 6 .$ . Unsafe combines Low-risk, Mid-risk, and High-risk. Top5-dif is the percentage of notes whose top-five retrieved case sets change across five runs. Precision and recall are five-run means (%); their sample variances are measured in squared percentage points $( \mathsf { p p } ^ { 2 } )$
<table><tr><td rowspan="2">Extractor</td><td colspan="5">Safe</td><td colspan="5">Unsafe</td></tr><tr><td>Top5-diff</td><td>Prec</td><td>Rec</td><td> ${ \mathrm { V a r } } ( { \mathrm { P } } )$ </td><td>Var(R)</td><td>Top5-diff</td><td>Prec</td><td>Rec</td><td>Var(P)</td><td>Var(R)</td></tr><tr><td>Gemini-3.1-Pro</td><td>16.8</td><td>77.5</td><td>84.4</td><td>0.20</td><td>0.22</td><td>12.0</td><td>94.6</td><td>91.8</td><td>0.02</td><td>0.05</td></tr><tr><td>GPT-6.1-sol</td><td>11.6</td><td>80.0</td><td>86.2</td><td>0.12</td><td>0.13</td><td>7.6</td><td>95.3</td><td>92.8</td><td>0.01</td><td>0.03</td></tr><tr><td>Qwen-3.8-Max</td><td>22.6</td><td>75.5</td><td>82.9</td><td>0.31</td><td>0.53</td><td>17.5</td><td>94.1</td><td>91.0</td><td>0.05</td><td>0.08</td></tr><tr><td>NoteCap</td><td>0.0</td><td>85.3</td><td>89.0</td><td>0.00</td><td>0.00</td><td>0.0</td><td>96.3</td><td>94.9</td><td>0.00</td><td>0.00</td></tr></table>

## Voting Depth

To isolate the efect of voting depth, we follow the originalpolicy setting of Table 1, holding the evaluation split, extractor, encoder, case library, case-specific thresholds, and voting rule fixed while varying $k \in \mathsf { \bar { \{ 4 , 8 , 1 2 , 1 6 , 2 4 \} } }$ . Figure 4 summarizes performance as Safe and Unsafe, with Lowrisk, Mid-risk, and High-risk grouped into Unsafe for reporting. From $k = 4$ to $\bar { k = 1 6 }$ , Safe precision/recall increases from 77%/76% to 85%/88%, and Unsafe precision/recall increases from 72%/73% to 85%/82%. $\mathrm { A t } \bar { k = 2 4 }$ , these values become $8 7 \% / 8 7 \%$ and 85%/84%, respectively. Changes beyond $k = 1 6$ remain within two percentage points, indicating diminishing marginal gains.

## NoteCap Stability

We assess extraction stability through precedent retrieval and downstream moderation on the 5,000-note MFO-RedNotes validation split under the original policy. Gemini-3.1-Pro, GPT-6.1-sol, Qwen-3.8-Max, and NoteCap each generate five structured representations per note with a fixed extraction prompt. All extractors share the Qwen3-Embedding-8B encoder, frozen case library, and thresholded voting configuration with $k = 1 6$ . Unsafe combines Low-risk, Mid-risk, and High-risk. Top5-dif measures the percentage of notes whose top-five retrieved case sets change across runs, ignoring order. Table 4 reports this rate alongside five-run precision/recall means and sample variances.

NoteCap achieves 85.3%/89.0% precision/recall on Safe and 96.3%/94.9% on Unsafe. Compared with GPT-6.1-sol, the strongest baseline across the reported metrics, its precision/recall gains are 5.3/2.8 percentage points on Safe and 1.0/2.1 percentage points on Unsafe. NoteCap also reports 0.0% Top5-dif for both classes, compared with 11.6%/7.6% for GPT-6.1-sol, and precision/recall variances of $0 . 0 0 ~ \mathrm { p p ^ { 2 } }$ . These results demonstrate improved moderation quality alongside stable top-five retrieval and aggregate performance across the five runs.

## Future Analysis

This section presents four key findings from the practical application of CaseLaw, which help guide case library design.

Larger case libraries improve moderation performance. Quality-controlled expansion broadens coverage of rare risk patterns and fine-grained policy boundaries, thereby enhancing the stability and robustness of the judge. These benefits stem from broader coverage of distinct moderationrelevant semantics.

![](images/fae73efccf20acae77e6ded15c812a0d6b149d6dda9b013ebcf46d6e178cca61.jpg)  
Figure 4: Voting-depth ablation under the original-policy setting of Table 1, with all other experimental conditions fixed. Precision (top) and recall (bottom) are expressed as percentages. Blue dashed squares denote Safe and pink solid circles denote Unsafe.

Increasing k improves robustness, but the gains follow a scaling law. Figure 4 reveals a saturating empirical scaling law for voting depth over the evaluated range: increasing k initially improves precision and recall, with diminishing gains beyond $k = 1 6$ . More retrieved precedents provide additional evidence, but their marginal benefit diminishes once the voting neighborhood contains suficient relevant cases.

Case counts across risk labels should remain balanced within the library. Otherwise, frequently represented labels can dominate local votes because of storage frequency rather than semantic relevance. Balanced admission and stratified sampling can address this bias.

Cases in the library should remain semantically distinct, avoiding excessive similarity among cases. Case admission should favor distinct evidence patterns and decision boundaries. Here semantic independence means limited redundancy. Many nearly identical cases can amplify correlated annotation errors and overwhelm a vote without adding meaningful coverage.

## Conclusion

In this paper, we presented CaseLaw, a policy-decoupled framework for training-free content moderation. By separating stable semantic representation from dynamic policy interpretation, CaseLaw enables moderation systems to adapt to evolving policies through updates to an external semantic case library rather than through rule rewriting, model retraining, or redeployment. Its NoteCap extractor preserves riskbearing evidence while organizing semantic attributes; retrieved precedent labels then determine the outcome through thresholded voting. We further introduced MFO-RedNotes, a multimodal benchmark for the male-female opposition risk domain, and evaluated CaseLaw under its original and revised policies. These experiments demonstrate that CaseLaw yields consistent gains over zero-shot, fixed few-shot, and fine-tuning baselines, while ablation studies validate the efectiveness of structured semantic extraction in forming clearer and more policy-aligned semantic anchors. These findings indicate that case-library-driven policy adaptation ofers a practical, scalable, and transparent direction for content moderation under rapidly evolving governance requirements in operationally realistic settings.

## Limitations

CaseLaw externalizes policy knowledge rather than removing policy dependence: performance remains sensitive to the accuracy, coverage, balance, and freshness of the case library. Comparisons with general safeguard models are sensitive to the mapping between their native safety categories and the benchmark’s policy-dependent labels.

## References

Arya, S.; Mount, D. M.; Netanyahu, N. S.; Silverman, R.; and Wu, A. Y. 1998. An optimal algorithm for approximate nearest neighbor searching fixed dimensions. Journal ofthe ACM, 45(6): 891–923.

Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems, volume 33, 1877–1901.

Cerulli, A.; Cima, L.; Tessa, B.; Tardelli, S.; and Cresci, S. 2026. The Big Ban Theory: A Pre- and Post-Intervention Dataset of Online Content Moderation Actions. In Proceedings ofthe InternationalAAAIConference on Web and Social Media, volume 20, 2756–2767.

Chen, J.; Shen, E.; Bavalatti, T.; Lin, X.; Wang, Y.; Hu, S.; Subramanyam, H.; Vepuri, K. S.; Jiang, M.; Qi, J.; Chen, L.; Jiang, N.; and Jain, A. 2024. Class-RAG: Real-Time Content Moderation with Retrieval Augmented Generation. arXiv preprint arXiv:2410.14881.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171–4186.

Douze, M.; Guzhva, A.; Deng, C.; Johnson, J.; Szilvasy, G.; Mazaré, P.-E.; Lomeli, M.; Hosseini, L.; and Jégou, H. 2026. The Faiss Library. IEEE Transactions on Big Data, 12(2): 346–361.

Gambäck, B.; and Sikdar, U. K. 2017. Using Convolutional Neural Networks to Classify Hate-Speech. In Proceedings ofthe First Workshop on Abusive Language Online, 85–90.

Gao, Y.; Xiong, Y.; Gao, X.; Jia, K.; Pan, J.; Bi, Y.; Dai, Y.; Sun, J.; Wang, M.; and Wang, H. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997.

Gerrard, Y. 2020. Behind the Screen: Content Moderation in the Shadows of Social Media. New Media & Society, 22(3): 579–582. Book review of Sarah T. Roberts (2019).

Gillespie, T. 2018. Custodians of the Internet: Platforms, Content Moderation, and the Hidden Decisions That Shape Social Media. Yale University Press.

Gorwa, R.; Binns, R.; and Katzenbach, C. 2020. Algorithmic content moderation: Technical and political challenges in the automation of platform governance. Big Data & Society, 7(1): 2053951719897945.

Huang, T. 2025. Content moderation by LLM: from accuracy to legitimacy. Artificial Intelligence Review, 58(10): 320.

Inan, H.; Upasani, K.; Chi, J.; Rungta, R.; Iyer, K.; Mao, Y.; Tontchev, M.; Hu, Q.; Fuller, B.; Testuggine, D.; and Khabsa, M. 2023. Llama Guard: LLM-based Input-Output

Safeguard for Human-AI Conversations. arXiv preprint arXiv:2312.06674.

Jha, A.; and Mamidi, R. 2017. When does a compliment become sexist? Analysis and classification of ambivalent sexism using twitter data. In Proceedings of the Second Workshop on NLP and Computational Social Science, 7–16.

Lei, W.; Abdullah, N. A. S.; and Syed Aris, S. R. 2024. A Systematic Literature Review on Automatic Sexism Detection in Social Media. Engineering, Technology & Applied Science Research, 14(6): 18178–18188.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, volume 33, 9459– 9474.

Lin, S.-C.; Asai, A.; Li, M.; Oguz, B.; Lin, J.; Mehdad, Y.; Yih, W.-t.; and Chen, X. 2023. How to Train Your Dragon: Diverse Augmentation Towards Generalizable Dense Retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2023, 6385–6400.

Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2024a. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Associationfor Computational Linguistics, 12: 157–173.

Liu, R.; Khakzar, A.; Gu, J.; Chen, Q.; Torr, P.; and Pizzati, F. 2024b. Latent Guard: A Safety Framework for Text-to-Image Generation. In Computer Vision – ECCV 2024, 93–109.

Markov, T.; Zhang, C.; Agarwal, S.; Eloundou Nekoul, F.; Lee, T.; Adler, S.; Jiang, A.; and Weng, L. 2023. A Holistic Approach to Undesired Content Detection in the Real World. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 15009–15018.

Meta AI. 2025. Llama-4-Scout-17B-16E-Instruct. https://huggingface.co/meta-llama/Llama-4-Scout-17B-

16E-Instruct. Hugging Face model card. Released April 5, 2025. Accessed: 2026-11-08.

Mikolov, T.; Chen, K.; Corrado, G.; and Dean, J. 2013. Efficient Estimation of Word Representations in Vector Space. arXiv preprint arXiv:1301.3781.

Nobata, C.; Tetreault, J.; Thomas, A.; Mehdad, Y.; and Chang, Y. 2016. Abusive Language Detection in Online User Content. In Proceedings ofthe 25th International Conference on World Wide Web, 145–153.

OpenGVLab. 2025. InternVL3-14B-Instruct. https: //huggingface.co/OpenGVLab/InternVL3-14B-Instruct. Hugging Face model card. Accessed: 2026-11-08.

Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C. L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; Schulman, J.; Hilton, J.; Kelton, F.; Miller, L.; Simens, M.; Askell, A.; Welinder, P.; Christiano, P.; Leike, J.; and Lowe, R. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, 27730–27744.

Palla, K.; García, J. L. R.; Hauf, C.; Fabbri, F.; Damianou, A.; Lindström, H.; Taber, D.; and Lalmas, M. 2025. Policyas-Prompt: Rethinking Content Moderation in the Age of

Large Language Models. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency, 840–854.

Pennington, J.; Socher, R.; and Manning, C. 2014. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532–1543.

Sclar, M.; Choi, Y.; Tsvetkov, Y.; and Suhr, A. 2024. Quantifying Language Models’ Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting. In International Conference on Learning Representations.

Shelby, R.; Rismani, S.; Henne, K.; Moon, A.; Rostamzadeh, N.; Nicholas, P.; Yilla-Akbari, N.; Gallegos, J.; Smart, A.; Garcia, E.; and Virk, G. 2023. Sociotechnical Harms of Algorithmic Systems: Scoping a Taxonomy for Harm Reduction. In Proceedings ofthe 2023 AAAI/ACM Conference on AI, Ethics, and Society, 723–741.

van der Maaten, L.; and Hinton, G. 2008. Visualizing Data using t-SNE. Journal of Machine Learning Research, 9(86): 2579–2605.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, volume 30.

Waseem, Z.; and Hovy, D. 2016. Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter. In Proceedings of the NAACL Student Research Workshop, 88–93.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.; Xia, F.; Chi, E.; Le, Q.; and Zhou, D. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, volume 35, 24824–24837.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.

Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.; Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou, J. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:2506.05176.

## A Structured Semantic Refinement Reduces Moderation Uncertainty

This section provides an information-theoretic perspective on CaseLaw, showing that structured semantic refinement can reduce moderation uncertainty by conditioning decisions on policy-relevant precedent clusters beyond coarse classifier regions. We consider a binary specialization of the policydependent label space $\mathcal { \partial } _ { t }$ , where x denotes the input content and $y ^ { ( t ) } \in \{ 0 , 1 \}$ denotes the moderation label. In the male–female opposition domain, $y ^ { ( t ) } = 1$ means violation and $y ^ { ( t ) } = 0$ means pass. A conventional classifier learns a posterior $q _ { \Theta } ( y ^ { ( t ) } \mid x )$ through cross-entropy training. Let R denote a coarse semantic region induced by the classifier representation. In contrast, our case-based framework first transforms content into the structured semantic representation $a _ { x } = S _ { \theta } ( x )$ , and then retrieves a fine-grained semantic precedent cluster $\mathcal { N }$ induced by $\mathcal { N } _ { k } ( x )$

Intuitively, R is what the classifier can distinguish, while N is what structured semantic reasoning can further refine.

## Entropy Reduction by Semantic Refinement

Proposition 1 (Entropy Reduction by Semantic Refinement). If the structured semantic cluster N provides additional information about the moderation label $y ^ { ( t ) }$ beyond the classifier’s coarse region R, then conditioning on $\mathcal { N }$ reduces moderation uncertainty:

$$
H ( y ^ { ( t ) } \mid R , \mathcal { N } ) < H ( y ^ { ( t ) } \mid R ) .\tag{11}
$$

More generally,

$$
H ( y ^ { ( t ) } \mid R ) - H ( y ^ { ( t ) } \mid R , { \mathcal { N } } ) = I ( y ^ { ( t ) } ; { \mathcal { N } } \mid R ) \geq 0 .\tag{12}
$$

The inequality is strict when

$$
I ( y ^ { ( t ) } ; N \mid R ) > 0 .\tag{13}
$$

Proof. Fix a coarse classifier region $R = r$ . Suppose structured semantic reasoning refines this region into m finegrained clusters:

$$
R = r = { \mathcal { N } } _ { 1 } \cup { \mathcal { N } } _ { 2 } \cup \cdots \cup { \mathcal { N } } _ { m } .\tag{14}
$$

Let

$$
p _ { j } = P ( y ^ { ( t ) } = 1 \mid R = r , \mathcal { N } = j )\tag{15}
$$

be the violation rate of the j-th semantic cluster, and let

$$
\lambda _ { j } = P ( N = j \mid R = r )\tag{16}
$$

be its mixture weight inside the coarse region. Then

$$
\sum _ { j = 1 } ^ { m } \lambda _ { j } = 1 .\tag{17}
$$

By the law of total probability, the violation rate of the coarse classifier region is

$$
q _ { r } = P ( y ^ { ( t ) } = 1 \mid R = r ) = \sum _ { j = 1 } ^ { m } \lambda _ { j } p _ { j } .\tag{18}
$$

The binary entropy function is

$$
h ( p ) = - p \log p - ( 1 - p ) \log ( 1 - p ) .\tag{19}
$$

Since $h ( p )$ is concave, Jensen’s inequality gives

$$
h \left( \sum _ { j = 1 } ^ { m } \lambda _ { j } p _ { j } \right) \geq \sum _ { j = 1 } ^ { m } \lambda _ { j } h ( p _ { j } ) .\tag{20}
$$

Therefore,

$$
h ( q _ { r } ) \geq \sum _ { j = 1 } ^ { m } \lambda _ { j } h ( p _ { j } ) .\tag{21}
$$

Equivalently,

$$
H ( y ^ { ( t ) } \mid R = r ) \geq H ( y ^ { ( t ) } \mid R = r , \mathcal { N } ) .\tag{22}
$$

Averaging over all coarse regions R, we obtain

$$
H ( y ^ { ( t ) } \mid R ) - H ( y ^ { ( t ) } \mid R , { \mathcal { N } } ) = I ( y ^ { ( t ) } ; { \mathcal { N } } \mid R ) \geq 0 .\tag{23}
$$

Thus, structured semantic refinement cannot increase uncertainty. It strictly reduces uncertainty whenever the semantic cluster $\mathcal { N }$ contains policy-relevant information not captured by the classifier’s coarse region R.

## Disagreement View

The same result can be expressed using pairwise label disagreement. For a binary cluster with violation rate p, define

$$
D ( p ) = 2 p ( 1 - p ) .\tag{24}
$$

This is the probability that two randomly sampled cases from the same region have diferent labels.

For the coarse region $R = r ,$ R = , the disagreement is

$$
D ( q _ { r } ) = 2 q _ { r } ( 1 - q _ { r } ) .\tag{25}
$$

For the refined semantic clusters, the average disagreement is

$$
\sum _ { j = 1 } ^ { m } \lambda _ { j } D ( p _ { j } ) .\tag{26}
$$

Then

$$
D ( q _ { r } ) - \sum _ { j = 1 } ^ { m } \lambda _ { j } D ( p _ { j } ) = 2 \operatorname { V a r } _ { \lambda } ( p _ { j } ) \geq 0 .\tag{27}
$$

Indeed,

$$
D ( q _ { r } ) - \sum _ { j = 1 } ^ { m } \lambda _ { j } D ( p _ { j } ) = \left( 2 q _ { r } - 2 q _ { r } ^ { 2 } \right) - \sum _ { j = 1 } ^ { m } \lambda _ { j } \left( 2 p _ { j } - 2 p _ { j } ^ { 2 } \right)\tag{28}
$$

$$
\begin{array} { l } { { \displaystyle = 2 \sum _ { j = 1 } ^ { m } \lambda _ { j } p _ { j } ^ { 2 } - 2 q _ { r } ^ { 2 } } } \\ { { \displaystyle = 2 \operatorname { V a r } _ { \lambda } ( p _ { j } ) . } } \end{array}\tag{29}
$$

(30)

Therefore, when a classifier collapses semantic modes with diferent violation rates into one coarse region, it necessarily increases local label disagreement. Structured semantic case reasoning reduces this disagreement by separating those modes into policy-aligned precedent clusters.

<table><tr><td>Risk Level</td><td>Typical Manifestation</td></tr><tr><td>Safe</td><td>No obvious risk of gender-based antagonism. The content is typically neutral, factual, or rights-oriented, and respects diverse viewpoints without inflammatory framing.</td></tr><tr><td>Low Risk</td><td>The content may involve sensitive gender-related topics but remains restrained in tone. It usually presents contro- versial opinions in a moderate manner without explicit hostility, group attribution, or provocative escalation.</td></tr><tr><td>Medium Risk</td><td>The content shows a clear antagonistic tendency but does not reach an extreme or explicit violation. It often uses oppositional framing, implicit group attribution, emotionally charged claims, or partially matches policy-defined risky patterns.</td></tr><tr><td>High Risk</td><td>The content exhibits strong antagonism or directly violates the moderation policy. Typical manifestations include original-sin rhetoric, extreme demands, explicit abuse, hate speech, deliberate provocation, or intensified gender- based opposition.</td></tr></table>

Table 5: Risk-level taxonomy for the male-female opposition moderation domain. Each level reflects increasing severity of gender-based antagonism and is defined according to expert-designed content-safety policies used for MFO-RedNotes.

## Interpretation

The classifier observes a coarse mixture region R, so the cross-entropy optimal prediction is the mixture posterior

$$
P ( y ^ { ( t ) } = 1 \mid R ) .\tag{31}
$$

Structured case reasoning observes a finer semantic cluster ${ \mathcal { N } } ,$ so it estimates

$$
P ( y ^ { ( t ) } = 1 \mid R , { \mathcal { N } } ) .\tag{32}
$$

The benefit depends on whether retrieved clusters capture additional policy-relevant information. Modern foundation models can construct the structured semantic representation $a _ { x } = S _ { \theta } ( x )$ that is aligned with moderation policy factors, such as target group, generalization, essentialist attribution, derogation, incitement, quotation, and context.

Thus, the key diference is:

$$
\mathrm { c l a s s i f i c a t i o n \ : h e a d : } \quad P ( \boldsymbol { y } ^ { ( t ) } \mid \boldsymbol { R } ) ,\tag{33}
$$

$$
\mathrm { s t r u c t u r e d \ c a s e \ r e a s o n i n g : } \quad P ( y ^ { ( t ) } \mid R , N ) .\tag{34}
$$

When

$$
I ( y ^ { ( t ) } ; N \mid R ) > 0 ,\tag{35}
$$

the structured semantic cluster provides additional moderation information, leading to lower entropy and lower label disagreement.

## B Details of MFO-RedNotes benchmark

MFO-RedNotes contains 50,000 multimodal notes partitioned into 40,000 training, 5,000 validation, and 5,000 test instances.

## Exemption Details

To evaluate policy adaptation under fine-grained boundary changes, we introduce exemption rules to simulate a realistic policy-change scenario in MFO-RedNotes. Prior studies on online sexism and gender-related language show that such expressions can range from explicit sexist hate speech to subtle ambivalent sexism, and often require fine-grained taxonomies for reliable interpretation (Waseem and Hovy 2016; Jha and Mamidi 2017; Lei, Abdullah, and Syed Aris 2024). Before exemption rules are applied, some cases containing sensitive gender-related expressions may be assigned higher risk levels due to their apparent association with male-female opposition. After the introduction of exemption rules, cases satisfying expert-defined exemption conditions are downgraded to lower risk levels, reflecting a revised policy boundary rather than a change in surface content. These exemption rules are formulated by industry content-safety experts and are used only to construct the policy-change evaluation setting; the exemption indicator and exemption rationale are never provided as input to the evaluated moderation methods.

The exemption rules cover the following four types:

• Non-negative conflict or emotion. Content can be exempted when it expresses non-antagonistic emotional experiences, such as pain caused by missing someone, regret over a missed relationship, or frustration about not finding a partner.

• Positive viewpoints or constructive cases. Content can be exempted when it promotes constructive genderrelated perspectives, such as encouraging friendly problem-solving between genders or sharing successful cases where both sides reach mutual understanding.

• Educational legal or risk-prevention content. Content can be exempted when it only provides legal knowledge or risk-prevention information, as long as it does not associate the discussion with real-world social events, social phenomena, or public controversies.

• Individual-level factual discussion. Content can be exempted when it discusses only a specific individual, does not generalize the description to a broader gender group, and the individual-level description is factual.

These rules create a challenging policy-change setting because the surface expressions of many cases remain unchanged, while their efective risk labels may shift after the introduction of exemption conditions. Therefore, this setting allows us to examine whether moderation systems can capture subtle policy-boundary revisions rather than merely relying on lexical risk cues.

![](images/c06335aa2630c87b05d4ba086d582840ddf88035efba75ecb15f9b0edafae7b4.jpg)  
(a)

![](images/077b98ddda66aecf625214ff2a87d6e9a7e48bebdcaed79dacb5bde0f532a7cb.jpg)  
(b)

![](images/b1b9d66e05a2685bcc312b58ef1ba2d803dcc6b0bdfc6ba50e1a124dbb187b17.jpg)  
(c)  
Figure 5: Impact of diferent structured semantic extraction backbones on embedding distributions. We visualize the representations produced by (a) InternVL3-14B-Instruct, (b) Llama-4-Scout-17B, and (c) NoteCap on MFO-RedNotes. NoteCap produces more compact intra-class clusters and clearer safe–risk separation, indicating better formation of risk-oriented semantic anchors for downstream retrieval.

## Risk Level Definitions

Table 5 summarizes the risk-level taxonomy used in MFO-RedNotes. The annotation policy is formulated by industry content-safety experts for the male-female opposition risk domain, where the key challenge lies in distinguishing neutral discussion, controversial but restrained expression, implicit antagonism, and explicit policy-violating hostility. Following this policy, each post is assigned one of four risk levels: Safe, Low Risk, Medium Risk, or High Risk, reflecting increasing degrees of antagonistic framing, group opposition, rhetorical escalation, and potential harm.

The risk labels are annotated by 20 senior content moderators according to the corresponding domain-specific moderation policy. The annotation process is conducted in five submission rounds; for each round, 10% of the submitted annotations are randomly sampled for double-blind quality inspection. A batch is accepted only when the annotation agreement exceeds 95%; otherwise, the corresponding instances are re-annotated until the agreement requirement is satisfied. This procedure ensures that the final labels are not only policy-dependent but also reliable for evaluating moderation systems under fine-grained and boundary-sensitive risk definitions.

## C Experiments on More Datasets

To further examine the generality of CaseLaw beyond multimodal note moderation, we evaluate it on textual comments from The Big Ban Theory (TBBT) (Cerulli et al. 2026), a dataset of user activity before and after moderation interventions. We use the dataset’s existing labels to evaluate three subreddit subsets, braincels, theredpill, and chapotraphouse, across distinct comment-moderation settings.

We apply CaseLaw’s structured semantic extraction, precedent retrieval, and thresholded voting pipeline, comparing it with Few-shot, Class-RAG, and Llama Guard. Table 6 reports precision and recall for each subset.

CaseLaw achieves the highest precision and recall in all six comparisons, reaching 87.9%/90.3% on braincels, 86.7%/92.1% on theredpill, and 89.8%/90.5% on chapotraphouse. The gains over the strongest baselines range from 4.2 to 10.2 percentage points, with the largest improvement in chapotraphouse recall over Llama Guard. These results support the applicability of case-based moderation beyond multimodal notes to textual comments across diferent subreddit communities.

Table 6: Illustrative moderation performance on The Big Ban Theory benchmark. Precision and recall are expressed as percentages for three subreddit subsets. Superscripts indicate gains in percentage points over the strongest baseline in each column.
<table><tr><td rowspan="2">Method</td><td colspan="2">braincels</td><td colspan="2">theredpill</td><td colspan="2">chapotraphouse</td></tr><tr><td>| Prec (%) Rec (%)</td><td></td><td>Prec (%) Rec (%)</td><td></td><td></td><td>| Prec (%) Rec (%)</td></tr><tr><td>Few-shot</td><td>68.3</td><td>69.8</td><td>64.9</td><td>69.2</td><td>70.6</td><td>67.1</td></tr><tr><td>Class-RAG</td><td>83.7</td><td>84.6</td><td>81.9</td><td>83.2</td><td>78.2</td><td>79.3</td></tr><tr><td>Llama Guard</td><td>83.6</td><td>85.8</td><td>80.6</td><td>87.4</td><td>85.5</td><td>80.3</td></tr></table>

CaseLaw (ours) 87.9<sup>↑</sup> <sup>4.2</sup> 90.3<sup>↑</sup> <sup>4.5</sup> 86.7<sup>↑</sup> <sup>4.8</sup> 92.1<sup>↑</sup> <sup>4.7</sup> 89.8<sup>↑</sup> <sup>4.3</sup> 90.5<sup>↑</sup> <sup>10.2</sup>

## D Comparison of NoteCap withOpen-Source Models

NoteCap serves as CaseLaw’s structured semantic extractor. The extracted representations support precedent retrieval, while the final risk label is determined by the thresholded voting rule in the main method. When comparing extraction backbones, the embedding model (Zhang et al. 2025), retrieval procedure, case library, case-specific thresholds, and voting rule remain fixed, isolating the representation supplied to the voting pipeline.

Figure 5 compares the embedding distributions produced by InternVL3-14B-Instruct (OpenGVLab 2025), Llama-4- Scout-17B (Meta AI 2025), and NoteCap in panels (a)– (c), respectively. Relative to the two general-purpose opensource extractors, NoteCap exhibits tighter intra-class clustering and clearer separation between safe and risk content. These projected distributions provide qualitative evidence of NoteCap’s stronger semantic discriminability, indicating that its structured representations concentrate risk-relevant cues while reducing cross-class overlap. This organization yields more discriminative semantic anchors for precedent retrieval and subsequent thresholded voting.

<table><tr><td>Cases</td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td></tr><tr><td>1,000</td><td>70.3</td><td>60.7</td><td>64.0</td></tr><tr><td>5,000</td><td>75.4</td><td>70.0</td><td>72.3</td></tr><tr><td>20,000</td><td>79.9</td><td>77.3</td><td>78.4</td></tr><tr><td>40,000</td><td>84.3</td><td>83.6</td><td>83.8</td></tr></table>

Table 7: Efect of case-library size on moderation performance. Label proportions, evaluation queries, and retrieval settings are fixed. All metrics are percentages. Macro-F1 averages class-wise F1 scores.
<table><tr><td>Largest:smallest</td><td>Macro-F1</td><td>Worst-label recall</td></tr><tr><td>1:1</td><td>83.8</td><td>76.6</td></tr><tr><td>2:1</td><td>81.8</td><td>71.9</td></tr><tr><td>4:1</td><td>76.5</td><td>63.7</td></tr><tr><td>8:1</td><td>69.3</td><td>51.4</td></tr></table>

Table 8: Efect of label imbalance in the case library. Library size, evaluation queries, and retrieval settings are fixed. The balanced condition bases on the 40,000-case reference configuration in Table 7. All metrics are percentages.

## E Details of Further Analysis

Case-Library Scale. We vary the number of training-split cases from 1,000 to 40,000 while holding label proportions, evaluation queries, and retrieval settings fixed. Table 7 shows consistent improvements in all three metrics: macro-precision increases from 70.3% to 84.3%, macrorecall from 60.7% to 83.6%, and macro-F1 from 64.0% to 83.8%. The 19.8-percentage-point gain in macro-F1 indicates that broader precedent coverage improves moderation performance.

Label Balance in the Case Library. Based on the 40,000- case reference configuration, we vary the ratio between the largest and smallest label counts while keeping library size, evaluation queries, and retrieval settings fixed. As the ratio increases from 1:1 to 8:1, macro-F1 decreases from 83.8% to 69.3%, and worst-label recall falls from 76.6% to 51.4% (Table 8), corresponding to declines of 14.5 and 25.2 percentage points. The sharper reduction in worst-label recall highlights the importance of balanced precedent support for maintaining performance across labels.

Semantic Independence and Case Quality. Based on the same 40,000-case reference configuration, we vary the fraction of cases belonging to near-duplicate semantic clusters while holding label proportions, evaluation queries, and retrieval settings fixed. As the near-duplicate share increases from 0% to 75%, the number of semantic clusters decreases from 40,000 to 17,240, while macro-F1 falls from 83.8% to 69.5%, a decline of 14.3 percentage points (Table 9). These results associate reduced semantic diversity with lower moderation performance and support admitting cases that contribute distinct evidence patterns.

<table><tr><td>Near-duplicate share</td><td>Clusters</td><td>Macro-F1</td></tr><tr><td>0%</td><td>40,000</td><td>83.8</td></tr><tr><td>25%</td><td>33,848</td><td>81.8</td></tr><tr><td>50%</td><td>25,616</td><td>76.3</td></tr><tr><td>75%</td><td>17,240</td><td>69.5</td></tr></table>

Table 9: Efect of semantic redundancy in the case library. The 40,000-case reference configuration matches Table 7. Near-duplicate share is the fraction of cases belonging to clusters containing at least two cases; the cluster count includes singleton clusters. Macro-F1 is expressed as a percentage.

## Case-Library Maintenance

Cold-start initialization. CaseLaw begins with a cold-start phase in which human moderators assign policy-dependent labels to an initial set of cases. Twenty senior moderators annotate these cases over five rounds. In each round, a random 10% of annotations undergo double-blind review. A batch is accepted only when annotation agreement exceeds 95%; otherwise, the afected annotations are revised and reviewed again. This process establishes a reliable seed library for subsequent moderation and incremental expansion.

Continual admission from moderation logs. Once the seed library provides an initial basis for moderation, real cases encountered during online operation are continuously accumulated in a temporary log repository. The repository uses the same record schema as the case library but remains separate from the active retrieval index. At irregular intervals, human moderators audit a sample of logged cases and assess agreement between the recorded labels and their reviewed judgments under the current policy. A batch becomes eligible for admission to the case library when the sampled label agreement reaches at least 95%. Batches that do not meet this criterion remain in the temporary repository for correction and further review. This staged admission process supports ongoing library growth while maintaining label quality.

Threshold calibration before admission. Each admitted case stores its own similarity threshold τ . In our experiments, thresholds for training-set cases are calibrated on the validation set using safe and violating notes. Candidate thresholds are assessed by their efects on violating-note recall and false-positive decisions on safe notes, yielding a case-specific operating point. Cases with more implicit risk cues receive higher thresholds to restrict voting to suficiently close semantic matches and reduce false positives. During moderation, a retrieved case casts one vote only when its similarity to the query is at least τ<sub>i</sub>.

Targeted maintenance under policy changes. When a policy is revised, potentially afected cases are localized through embedding retrieval and keyword search. A batch of representative example notes can be encoded with the frozen embedding model and used as queries to retrieve semantically related cases from the library. Keyword queries over the principal entities or subjects provide a complementary way to locate cases within the scope of the revised rule. Moderators then inspect the retrieved candidates and update the labels and rationales of cases afected by the policy change. Label revisions and afected-case lists are versioned for traceability. Updating labels or rationales preserves existing embeddings, supporting eficient, targeted maintenance.

## F Cases of Structured Semantic Extraction

Figures 6 and 7 present two qualitative cases comparing direct splicing with structured semantic extraction from MFO-RedNotes. For ethical and privacy considerations, all personally identifiable information and potentially sensitive details, including privacy-related attributes and violence-related descriptions, are visually masked in the figures. In Case 1, this structure concentrates the representation around the allegation of domestic violence and its warning intent, while in Case 2 it isolates emotional venting in a breakup context from more severe harm-related semantics. By explicitly aligning surface evidence with normalized semantic attributes, structured extraction reduces irrelevant lexical variation and incidental details, leading to more coherent case representations and facilitating more precise retrieval of policy-relevant precedents. This supports CaseLaw’s design intuition that structured semantic organization enables a more focused case library for robust moderation under evolving policies.

## G Metrics

Moderation performance. For each class c, let TP<sub>c</sub>, FP<sub>c</sub>, and FN denote one-versus-rest counts over the full evaluation split. Precision, recall, and F1 are

$$
\begin{array} { c c } { P _ { c } = \displaystyle \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } } , \qquad R _ { c } = \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F N } _ { c } } , } \\ { F _ { 1 , c } = \displaystyle \frac { 2 P _ { c } R _ { c } } { P _ { c } + R _ { c } } . } \end{array}\tag{36}
$$

These scores are proportions; percentage reporting multiplies them by 100. For binary evaluation, both ground-truth labels and predictions merge Low-risk, Mid-risk, and High-risk into Unsafe. Abstentions count as false negatives for their true class. Over the evaluated label set Y, macro-averaging weights classes equally, while worst-label recall measures the least-recovered class:

$$
Q _ { \mathrm { m a c r o } } = \frac { 1 } { | \mathcal { V } | } \sum _ { c \in \mathcal { V } } Q _ { c } , \qquad R _ { \mathrm { m i n } } = \operatorname* { m i n } _ { c \in \mathcal { V } } R _ { c } ,\tag{37}
$$

where $Q \in \{ P , R , F _ { 1 } \}$ . Macro-F1 averages class-wise F1 rather than combining macro-precision and macro-recall. For a proportion-valued score Q, the absolute gain over baseline set B is

$$
\Delta Q = 1 0 0 \bigg ( Q _ { \mathrm { C a s e L a w } } - \operatorname* { m a x } _ { b \in \mathcal { B } } Q _ { b } \bigg ) ,\tag{38}
$$

![](images/aa4103d8b788784aa2d9b0acee1cb9dd00216e2719b4e00c0f3758ba4fe44870.jpg)  
Figure 6: Evidence-preserving extraction for an allegation and warning. Identifying and sensitive details are masked.

![](images/1428d1789218163bf2135a48e56b71e4d077ff9177784073f1a411868654d8d0.jpg)  
Figure 7: Extraction of individual-level emotional expression. Identifying and sensitive details are masked.

in percentage points, using the same class, policy, and evaluation setting.

False positives and coverage. Violating-note recall is $R _ { \mathrm { U n s a f e } }$ . With $N _ { \mathrm { S a f e } }$ safe notes and N total notes, safe-note false-positive rate and automated-decision coverage are

$$
\begin{array} { l } { { \mathrm { F P R } _ { \mathrm { S a f e } } = \displaystyle \frac { \mathrm { F P } _ { \mathrm { U n s a f e } } } { N _ { \mathrm { S a f e } } } , } } \\ { { \displaystyle \mathrm { C o v e r a g e } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \{ \hat { y } _ { i } \neq \perp \} , } } \end{array}\tag{39}
$$

where ⊥ denotes abstention or referral for human review.

Repeated-run aggregation. For a fixed method, dataset, and class, let $q ^ { ( r ) }$ be precision or recall on the percentage

scale in run r. Across five runs,

$$
\bar { q } = \frac { 1 } { 5 } \sum _ { r = 1 } ^ { 5 } q ^ { ( r ) } , \qquad s _ { q } ^ { 2 } = \frac { 1 } { 4 } \sum _ { r = 1 } ^ { 5 } ( q ^ { ( r ) } - \bar { q } ) ^ { 2 } .\tag{40}
$$

The sample variance is expressed in $\mathrm { p p ^ { 2 } }$ . Aggregation is performed using unrounded run-level scores, with rounding applied only to the reported summaries. A reported variance of 0.00 reflects rounding to two decimal places.

Top-five retrieval diference. Let ${ \mathcal { T } } _ { c } = \{ i : y _ { i } = c \}$ $N _ { c } = | \mathcal { T } _ { c } |$ , and $T _ { i } ^ { ( r ) }$ be the unordered set of five highestranked case identifiers retrieved before threshold filtering. Top5-dif is

$$
D _ { 5 , c } = \frac { 1 0 0 } { N _ { c } } \sum _ { i \in \mathcal { I } _ { c } } \mathbb { I } \Big \{ \exists r \in \{ 2 , \dots , 5 \} : T _ { i } ^ { ( r ) } \neq T _ { i } ^ { ( 1 ) } \Big \} .\tag{41}
$$

Each note is counted once if any run changes set membership; rank-only changes do not contribute. The denominator contains ground-truth class-c notes. This diagnostic uses five cases independently of the voting depth $k = 1 6$

Retrieval and label agreement. Let $R _ { i } ^ { ( r ) }$ be the prethreshold top-k case set. Average pairwise overlap across the ten run pairs is

$$
A _ { \mathrm { o v e r l a p } } = \frac { 1 } { 1 0 N } \sum _ { i = 1 } ^ { N } \sum _ { 1 \leq r < s \leq 5 } \frac { | R _ { i } ^ { ( r ) } \cap R _ { i } ^ { ( s ) } | } { k } .\tag{42}
$$

Strict retrieval-set and label agreement are

$$
A _ { \mathrm { s e t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \{ R _ { i } ^ { ( 1 ) } = \cdots = R _ { i } ^ { ( 5 ) } \} ,\tag{43}
$$

$$
A _ { \mathrm { l a b e l } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \{ \hat { y } _ { i } ^ { ( 1 ) } = \cdots = \hat { y } _ { i } ^ { ( 5 ) } \} .\tag{44}
$$

These agreement scores lie in [0, 1]. Label agreement includes abstention as an outcome; repeated abstention does not imply correct moderation. Retrieved sets may difer while producing identical labels.

Annotation agreement. For a reviewed batch, raw label agreement is $A _ { \mathrm { a u d i t } } = n _ { \mathrm { m a t c h } } / n _ { \mathrm { r e v i e w e d } }$ , where $n _ { \mathrm { m a t c h } }$ counts matching labels among the reviewed annotations. This measures annotation consistency rather than model accuracy.