# Type-Balanced Contextual Learning for Incremental Named Entity Recognition

Duzhen Zhang, Yahan Yu, Xiuyi Chen, Chenxing Li, and Dong Yu <sup>ID</sup> , Fellow, IEEE

Abstract—Incremental Named Entity Recognition (INER) stands as a pivotal task in information extraction, emphasizing the successive identification of new entity types within unstructured text. Faced with the continuous influx of entity types, INER grapples with two significant challenges: the widespread issue of catastrophic forgetting and the unique shift issue of the non-entity type semantics. While pseudo-labeling-based INER methods have proven effective in addressing these challenges, a previously overlooked issue arises: the biased context problem. Our analysis shows that, in new sentences, the contextual associations of tokens representing old entity types exhibit a significantly stronger bias towards new entity types compared to their contexts in old sentences. This tendency intensifies the degradation of old knowledge while promoting the overfitting of new knowledge. To solve this biased context, we propose a Type-Balanced Contextual Learning (TBCL) method, featuring a sentence-duplet learning scheme and a contextual consistency loss. This approach offers a fresh perspective for INER through context analysis. Extensive experiments across ten INER settings on three highly recognized datasets showcase the efficacy of our TBCL method, highlighting its proficiency in resolving the biased context issue inherent in pseudo-labeling based INER approaches.

Impact Statement—Incremental Named Entity Recognition (INER) allows AI systems to continuously identify emerging entities in text, crucial for virtual assistants, automated healthcare records, and dynamic information platforms. Traditional methods often forget prior knowledge and overfit new data due to biased context. Our Type-Balanced Contextual Learning (TBCL)<sup>Inputs:</sup> <sup>Bin</sup> <sup>wil</sup> method corrects this by balancing old and new contextual infor-<sup>CL:</sup> <sup>[O]</sup> <sup>[O</sup> mation, ensuring stable learning over time. Extensive evaluations across benchmark datasets show TBCL consistently surpasses prior state-of-the-art approaches, offering more accurate, reliable, and adaptable AI models. This advancement improves AI’s practical utility in healthcare, business intelligence, and education (social impact), reduces costly errors and inefficiencies (economic impact), and sets a new standard for sustainable incremental learning in real-world AI applications.

Index Terms—Incremental Learning, Named Entity Recognition, Knowledge Distillation, Pseudo-labeling, Context Analysis

## I. INTRODUCTION

formation extraction, supporting various downstream tasks such as question answering [1], [2] and web search queries [3]–[6]. NER focuses on identifying entities from unstructured text and classifying them into predefined categories (e.g., Person, Location) or as the non-entity type. In the traditional fully-supervised NER framework [7], all predefined entity types are fixed and learned simultaneously during training, with the assumption that no new types will appear at <sup>avel</sup> <sup>to</sup> <sup>Ximending</sup> <sup>in</sup> <sup>Taipei</sup> <sup>in</sup> <sup>July</sup>the testing stage. Nevertheless, real-world scenarios frequently demand the discovery of novel entity types dynamically. For instance, virtual voice assistants like Alexa need to recognize new types (e.g., Genre, Actor) to understand emerging intents (e.g., GetMovie) [8]. Conventional NER paradigms struggle when confronted with new entity types not encountered at the training stage. A simple strategy involves fine-tuning the NER model exclusively with new entity types samples, referred to as Incremental Named Entity Recognition (INER) [8]– [10]. Effective INER methods [8], [11], [12] are needed to incrementally update the model using only training samples of new entity types. However, INER faces two primary issues: catastrophic forgetting [13]–[16] and the semantic shift of the non-entity type [17].

![](images/32509eb19c945f0b3404b31f5ce1ffa66326d4cc5a9a86705ba4c1c633c764dc.jpg)  
Fig. 1. A basic example of INER scenario, where FL, CL, and PL represent Full ground-truth Labels, Current ground-truth Labels, and Pseudo Labels, respectively. Previous entity types (e.g., [PER] (Person), [CITY] (City)) and future entity type (e.g., [DATE] (Date)) are marked as the non-entity type (i.e., [O]) in the current step t where [ORG] (Organization) is the current entity type being learned, leading to the shift of the non-entity type semantics (the second row CL). Using PL produced by the previous model to augment CL as the current classification target can effectively address the issue of semantic shift (the third row CL+PL).

The first challenge, catastrophic forgetting [13]–[16], is a widespread issue in incremental learning methodologies. This problem arises when model weights are modified to align with the optimal parameter space for newly added entity types, while lacking access to those encountered earlier. As a result, the NER model often forgets previously acquired knowledge of old entity types as it acquires information about new ones. The second challenge, the semantic shift of the non-entity type [17], is specific to INER. In contrast to conventional NER paradigms, where the non-entity type only includes tokens unrelated to any entity type, INER employs a distinct tagging approach. In INER, both previously learned and future entity types are tagged as the non-entity type at the current step, resulting in what we term the semantic shift of the non-entity type. As shown in Figure 1 (the second row, CL), old types Person and City (learned in previous steps like t − 1, t − 2, t − 3, etc.) and the future type Date (to be learned in future steps like t + 1, t + 2, t + 3, etc.) are all masked as the nonentity type in the current step t, where Organization is the type currently being learned. If measures are not implemented to distinguish tokens associated with old entity types from the actual non-entity type, this phenomenon of semantic shift might worsen catastrophic forgetting.

![](images/5a185cadea54f28cf89216f00211df9ca180f2d7ca9975aae32ed3f8fb014fdd.jpg)

![](images/82340168538db79c4a334478272e78b8197fbccfb6d1000b65c5c88c020ec828.jpg)  
Fig. 2. Depiction of biased contextual relationships between the oldentity-type and new-entity-type tokens in the new sentences. At step t − 1, when the NER model learns Person-entity-type sentences, the context of Person includes various entity types (e.g., Person-City, Person-Country, Person-Location). However, at step t, when the model learns the new Organization-entity-type sentences, the context of Person predominantly comprises Person-Organization pairs. Consequently, a new-entity-type-biased context for the old-entity-type (Person) tokens emerges in the new-entitytype (Organization) sentences, exacerbating the old-entity-type forgetting and new-entity-type overfitting issues.

Early INER methods [8], [11], [12] typically employ Knowledge Distillation (KD) [18] to address the problem of catastrophic forgetting. These approaches transfer output probability distribution from the previous model to the current model, thereby retaining the previously learned knowledge of old entity types. However, they overlook the shift issue of the non-entity type semantics specific to INER. To overcome this limitation, RDP [17], a pseudo-labeling based INER method, is proposed. This method tackles the semantic shift issue by augmenting the current ground-truth labels with pseudo labels, resulting in more informative labels as the classification target for the current step (the third row CL+PL in Figure 1). These pseudo labels are derived from predictions of the old model. By integrating KD and pseudo-labeling, RDP [17] attains State-Of-The-Art (SOTA) performance in INER.

Although RDP [17] effectively addresses the two primary challenges of INER, it introduces another challenge. Since NER is considered a context-dependent token-level classification task, we conducted a contextual analysis of entity types after pseudo-labeling [19]. As shown in Figure 2, in step t − 1, when the model learns the Person entity type, we observed that the co-occurrence frequency of Person with each of the pseudo-labeled old entity types (e.g., City, Country, Location) in the old sentences is almost the same. However, by step t, as the model learns the Organization entity type, the co-occurrence frequency between Person and Organization is greatly higher than with other entity types in the new sentences. We term this challenge the biased context problem, indicating that the context of old-entity-type tokens in the novel sentences exhibits a stronger bias towards new entity types compared to that in the old sentences. This bias can lead to a pronounced exacerbation of old-entity-type forgetting and new-entity-type overfitting.

To rectify this context bias, we introduce a Type-Balanced Contextual Learning (TBCL) method, which integrates a sentence-duplet learning scheme with a contextual consistency loss. This method aims to mitigate the impact of biased context information from old-entity-type tokens in the new incremental sentences. The sentence-duplet learning scheme utilizes pairs of sentences: the original sentence containing the new-entity-type tokens and a modified version with those tokens removed. This approach helps correct the context of old entity types in relation to the new ones. Additionally, the contextual consistency loss enforces a restriction on the context of old entity types within each sentence pair, ensuring consistency across contexts.

We highlight our core contributions as follows:

• We pioneer the exploration of biased context in the INER scenario and introduce the TBCL method to address it. This method aims to mitigate the forgetting of old entity types while preventing the overfitting of new ones.

• We devise a novel sentence-duplet learning scheme and a contextual consistency loss, modifying how old entity types are contextualized with respect to new ones.

• Comprehensive experiments conducted on ten INER settings across three highly recognized datasets confirm the efficacy of the proposed TBCL method. The outcomes reveal that our method surpasses several existing INER methods and achieves new SOTA performance in INER.

## II. RELATED WORK

INER, an emerging field, has seen limited exploration in recent literature, with few papers tackling this issue. We begin this section with a summary of the latest progress in incremental learning, followed by a detailed analysis of current methods applied to INER.

## A. Incremental Learning

Incremental learning is a methodology designed to continually absorb new information across successive tasks, with strategies to avoid catastrophic forgetting of earlier learnings, as noted in recent literature [20], [21]. The techniques used in incremental learning fall into three categories: regularizationbased, dynamic architecture-based, and replay-based methods. Regularization-based methods restrict adjustments to model weights [22]–[25], intermediate features [26], or the probabilities of outputs [27] to preserve past knowledge. Dynamic architecture-based methods [28]–[30] adjust the model’s architecture dynamically to allocate resources effectively across various tasks, thereby stabilizing the learning process as new tasks are introduced. Replay-based methods [31]–[33] merge previously acquired or synthetically generated samples into current training sessions, balancing the retention of old information with the acquisition of new insights.

## B. INER

Traditional NER typically classifies each token in a sequence into a standard set of entity types (e.g., Person, Organization, Location) or as the non-entity type [34]. The focus has largely been on the development of various neural network architectures to enhance NER capabilities in a seamless, end-to-end process [35]. Examples include techniques based on BiLSTM-CRF [36] and BERT [37]. Yet, in practical applications, NER systems often encounter emerging new entity types that necessitate continuous model updates without complete retraining [38]–[41]. To address this challenge, INER merges the incremental learning approach with traditional NER methods.

INER, still in its nascent stages, has seen only a limited number of studies tackling its specific challenges [8], [11], [12], [17], [42], [43]. Among them, ExtendNER [8] leverages KD techniques, wherein a teacher model (i.e., the previous model) passes on output probabilities to a student model $( i . e .$ , the newly updated model). L&R [11] presents a dualphase architecture known as Learn-and-Review. In the learning phase, it adopts a method similar to ExtendNER’s KD. In the review phase, it utilizes a rehearsal-based strategy, enhancing the current training dataset by artificially generated previous entity type samples. Meanwhile, CFNER [12] innovates a causal framework designed to distill causal effects specifically from the non-entity type, adding a new dimension to traditional methods. Despite the significant advancements made by these INER methods in reducing catastrophic forgetting, they still fall short in effectively handling the shift issue in the nonentity type semantics [17].

To remedy this defect, a pseudo-labeling based INER method called RDP [17] is proposed. This method develops a prototypical pseudo-label approach for classification, using the distances between token embeddings and type-wise prototypes to adjust the output probabilities of the previous model. This refinement helps to decrease prediction errors from the previous model and effectively handles the semantic shift issue. Combining KD with pseudo-labeling, RDP method achieves SOTA performance in the INER domain.

However, we identified a specific biased context issue with this method, which exacerbates the forgetting of old entity types and leads to overfitting of new ones. To this end, we introduce a TBCL method, incorporating a sentence-duplet learning scheme and a contextual consistency loss. This approach is specifically tailored for the pseudo-labeling based

INER method, further enhancing its performance by providing a more balanced learning context that mitigates these biases.

## III. METHOD

In this section, we begin by outlining the definition of the INER problem. Subsequently, we provide an overview of the pseudo-labeling based INER method: RDP [17]. Finally, we delve into the detailed introduction of our TBCL approach.

## A. Problem Formulation

Building on earlier studies [8], [11], [12], [17], INER focuses on sequentially training a model across multiple steps $t = 1 , 2 , . . . , T$ , each associated with a specific dataset $\mathcal { D } ^ { \bar { 1 } } , \mathcal { D } ^ { 2 } , . . . , \mathcal { D } ^ { T }$ . Training at each step involves the current model $\mathcal { M } ^ { t }$ using only its corresponding dataset $\mathcal { D } ^ { t }$ , which leads to the potential catastrophic forgetting of previously learned entity types from datasets $\mathcal { D } ^ { 1 } , \bar { \mathcal { D } } ^ { 2 } , . . . , \bar { \mathcal { D } } ^ { t - \bar { 1 } }$ . Specifically, dataset $\mathcal { D } ^ { t }$ contains multiple training samples $( X ^ { t } , Y ^ { t } )$ with $X ^ { t }$ representing a sequence of input tokens (i.e., sentence) and $Y ^ { t }$ the corresponding labels in one-hot format. These labels pertain only to the current entity types $\mathcal { E } ^ { t }$ relevant to step t. Labels from prior entity types $\mathcal { E } ^ { 1 : t - 1 }$ or future entity types $\scriptstyle { \mathcal { E } } ^ { t + 1 : T }$ are merged into the non-entity type $e _ { o } ,$ which introduces the issue of semantic shift. It’s essential to emphasize that entity types are mutually exclusive across steps; that is, $\mathcal { E } ^ { i } \cap \mathcal { E } ^ { j } = \emptyset$ for $i \neq j$ . The term $E ^ { t } = \mathrm { c a r d } ( \mathcal { E } ^ { t } )$ denotes the cardinality of the entity types at the current step. The standard structure of the current model $\mathcal { M } ^ { t }$ includes an encoder $F ^ { t }$ and a linear softmax classifier $C ^ { t }$ . Given the current dataset $\mathcal { D } ^ { t }$ and the old model $\mathcal { M } ^ { t - 1 }$ , the objective with INER is to train the current (new) model $\mathcal { M } ^ { t }$ to effectively recognize all entity types $\mathcal { E } ^ { 1 : t }$ encountered up to the current step. Let $X ^ { t } = ( x _ { 1 } ^ { t } , \dots , x _ { \mid X ^ { t } \mid } ^ { t } )$ denote a sentence at step t. We define the label space at step t as $\mathcal { C } ^ { t } = \{ e _ { o } \} \cup \mathcal { E } ^ { 1 : t }$ , where $e _ { o }$ denotes the non-entity type and $| \mathcal { C } ^ { t } | = 1 \dot { + } \dot { E ^ { 1 } } + \cdot \cdot \cdot + E ^ { t }$ . The one-hot ground-truth label matrix is denoted by $Y ^ { t } \in \{ 0 , 1 \} ^ { | X ^ { t } | \times | { \mathcal { C } } ^ { t } | }$ and the prediction matrix produced by the current model is denoted by $\widehat { Y } ^ { t } = \mathcal { M } ^ { t } ( X ^ { t } ; \grave { \Theta } ^ { t } ) \in [ 0 , 1 ] ^ { | \ u ^ { t } | \times | \mathcal { C } ^ { t } | }$ . Here, $Y _ { i , \epsilon } ^ { t }$ and $\widehat { Y } _ { i , \epsilon } ^ { t }$ denote the ground-truth label and predicted probability of the i-th token for class e, respectively. The cross-entropy loss for the current step t is calculated as follows:

$$
\mathcal { L } _ { c e } ( X ^ { t } , Y ^ { t } ; \boldsymbol { \Theta } ^ { t } ) = - \frac { 1 } { | X ^ { t } | } \sum _ { i = 1 } ^ { | X ^ { t } | } \sum _ { e \in \mathcal { C } ^ { t } } Y _ { i , e } ^ { t } \log { \widehat { Y } _ { i , e } ^ { t } } ,\tag{1}
$$

where $\Theta ^ { t }$ denotes the trainable parameters of the current model $\mathcal { M } ^ { t }$

## B. Pseudo-labeling Based INER

To mitigate issues of catastrophic forgetting and semantic shift, a pseudo-labeling based INER method, RDP [17], is proposed. This approach uses the KD technique to preserve prior knowledge and designs a prototypical pseudo-label strategy for creating high-quality pseudo labels.

1) KD: The KD technique alleviates catastrophic forgetting by transferring predicted probabilities from the previous model $\mathcal { M } ^ { t - 1 }$ to the current model $\mathcal { M } ^ { t }$ . The objective function for distilling this probability distribution is formulated as follows:

$$
\mathcal { L } _ { \mathrm { k d } } ( X ^ { t } , \widehat { Y } ^ { t - 1 } ; \Theta ^ { t } ) = - \frac { 1 } { | X ^ { t } | } \sum _ { i = 1 } ^ { | X ^ { t } | } \sum _ { e \in \mathcal { C } ^ { t - 1 } } \widehat { Y } _ { i , e } ^ { t - 1 } \log { \widehat { Y } _ { i , e } ^ { t } } ,\tag{2}
$$

where $\widehat { Y } ^ { t - 1 } \ = \ \mathcal { M } ^ { t - 1 } ( X ^ { t } ) \ \in \ [ 0 , 1 ] ^ { | X ^ { t } | \times | \mathcal { C } ^ { t - 1 } | } \ \mathrm { ~ a n d ~ } \ \widehat { Y } ^ { t }$ $\widehat { Y } ^ { t } \ =$ $\mathcal { M } ^ { t } ( X ^ { t } ; \Theta ^ { t } ) \ \in \ [ 0 , 1 ] ^ { | \dot { X } ^ { t } | \times | \dot { c } ^ { t } | } .$ . Since the old model only predicts over $\mathcal { C } ^ { t - 1 }$ , the KD loss is computed only on the old label space $\mathcal { C } ^ { t - 1 }$ . That is, only the entries of $\widehat { Y } ^ { t }$ corresponding to $\mathcal { C } ^ { t - 1 }$ are used.

2) Prototypical Pseudo-label Strategy: To tackle the $\mathrm { { s e - } }$ mantic shift issue of the non-entity type, RDP introduces a prototypical pseudo-label strategy [17]. The goal is to utilize the predictions from the previous model for non-entity type tokens, treating them as indicators of their true entity type, particularly when they align with any of the old entity types. We denote the pseudo-label target matrix for the current step by $\widetilde { Y } ^ { t } \in \{ 0 , 1 \} ^ { | \widetilde { X } ^ { t } | \times | { \mathcal C } ^ { t } | }$ <sup>|</sup>. This target is computed from the onehot ground-truth label matrix $Y ^ { t }$ and the old-model prediction $\widehat { Y } ^ { t - \overline { { 1 } } } = \mathcal { M } ^ { t - 1 } ( X ^ { t } )$ as follows:

$$
\widetilde { Y } _ { i , e } ^ { t } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } Y _ { i , e _ { o } } ^ { t } = 0 \mathrm { ~ a n d ~ } e = \arg \operatorname* { m a x } _ { e ^ { \prime } \in \mathcal { E } ^ { t } } Y _ { i , e ^ { \prime } } ^ { t } , } \\ { 1 , } & { \mathrm { i f ~ } Y _ { i , e _ { o } } ^ { t } = 1 \mathrm { ~ a n d ~ } e = \arg \operatorname* { m a x } _ { e ^ { \prime } \in \mathcal { C } ^ { t - 1 } } \widehat { Y } _ { i , e ^ { \prime } } ^ { t - 1 } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

In this formulation, $\widetilde { Y } _ { i . e } ^ { t } = 1$ indicates that the i-th token is assigned to class e. If the token is labeled as a current entity type, the target follows the ground-truth label (the first row of Equation (3)). Otherwise, if it is labeled as the non-entity type $\mathit { e _ { o } } ,$ the target is generated from the old-model prediction over the old label space $\mathcal { C } ^ { t - 1 }$ (the second row).

To reduce prediction errors stemming from the old model and generate more accurate pseudo labels, RDP delve deeper by leveraging the distances between token embeddings and type-specific prototypes to adjust the output probabilities of the old model [17]. Equation (3) can be revised as follows:

$$
\widetilde { Y } _ { i , e } ^ { t } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } Y _ { i , e _ { o } } ^ { t } = 0 \mathrm { ~ a n d ~ } e = \arg \operatorname* { m a x } _ { e ^ { \prime } \in \mathcal { E } ^ { t } } Y _ { i , e ^ { \prime } } ^ { t } , } \\ { 1 , } & { \mathrm { i f ~ } Y _ { i , e _ { o } } ^ { t } = 1 \mathrm { ~ a n d ~ } e = \arg \operatorname* { m a x } _ { e ^ { \prime } \in \mathcal { C } ^ { t - 1 } } \omega _ { i , e ^ { \prime } } ^ { t } \widehat { Y } _ { i , e ^ { \prime } } ^ { t - 1 } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

Here, $\omega _ { i , e } ^ { t }$ represents the type-wise prototypical weight, quantifying the distance between the embedding of token $\ v x _ { i } ^ { t }$ and the type-specific prototype $\eta ^ { e }$ . This is expressed as follows:

$$
\omega _ { i , e } ^ { t } = \frac { \exp ( - \| F ^ { t - 1 } ( x _ { i } ^ { t } ) - \eta ^ { e } \| ) } { \sum _ { e ^ { \prime } \in \mathcal { C } ^ { t - 1 } } \exp ( - \| F ^ { t - 1 } ( x _ { i } ^ { t } ) - \eta ^ { e ^ { \prime } } \| ) } ,\tag{5}
$$

where $F ^ { t - 1 } ( x _ { i } ^ { t } )$ denotes the embedding of token $\ v x _ { i } ^ { t }$ produced by the encoder $\overbrace { F } ^ { t - 1 }$ of the old model $\bar { \mathcal { M } } ^ { t - 1 }$ , and $\eta ^ { e }$ represents the prototype for class e. When the token embedding $F ^ { t - 1 } ( x _ { i } ^ { t } )$ is isolated from the prototype $\eta ^ { e }$ (i.e., the feature centroids of type e), it suggests that the learned feature might be considered an outlier. Consequently, this method reduces its likelihood of being classified into type e and vice versa. To obtain the prototypes $\eta ^ { e } ,$ , it begins by extracting coarse pseudo labels derived from the highest output probabilities predicted by the old model for all non-entity type tokens in the current dataset $\mathcal { D } ^ { t }$ . Then, it computes the average embeddings of tokens identified as e to establish the prototype $\eta ^ { e }$

The pseudo-labeling cross-entropy loss, computed utilizing these constructed pseudo labels, is formulated as follows:

$$
\mathcal { L } _ { \mathrm { p c e } } ( X ^ { t } , \widetilde { Y } ^ { t } ; \Theta ^ { t } ) = - \frac { 1 } { | X ^ { t } | } \sum _ { i = 1 } ^ { | X ^ { t } | } \sum _ { e \in \mathcal { C } ^ { t } } \widetilde { Y } _ { i , e } ^ { t } \log { \widehat { Y } _ { i , e } ^ { t } } .\tag{6}
$$

Overall, the objective function in the RDP method [17] is formulated as follows:

$$
\mathcal { L } _ { \mathrm { r d p } } ( X , \widetilde { Y } , \widehat { Y } ^ { o l d } ; \Theta ^ { t } ) = \mathcal { L } _ { \mathrm { p c e } } ( X , \widetilde { Y } ; \Theta ^ { t } ) + \alpha \mathcal { L } _ { \mathrm { k d } } ( X , \widehat { Y } ^ { o l d } ; \Theta ^ { t } ) ,\tag{7}
$$

where $\widetilde { Y }$ is the pseudo-label target corresponding to input $X ,$ and $\widehat { Y } ^ { o l d } = \mathcal { M } ^ { \widehat { t - 1 } } ( X )$ is the old-model prediction on the same input. The hyperparameter α balances the importance of the two loss terms.

## C. TBCL

While the aforementioned pseudo-labeling based INER method, RDP [17], effectively mitigates catastrophic forgetting and semantic shift problems, the updated NER model it produces encounters issues with biased context. For instance, at the t-th step (illustrated in Figure 2), the newly added sentences predominantly consist of the new-entity-type-related context for the old-entity-type tokens, which exacerbates problems related to forgetting old entity types and overfitting to new ones. To reduce the biased context correlation between old and new entity types in RDP [17] and to further improve its performance, we introduce a TBCL method. This method incorporates a sentence-duplet learning scheme and a contextual consistency loss, as depicted in Figure 3.

1) Sentence-duplet Learning Scheme: Regarding the context related to new entity types, we notice that the contextual information of old entity type tokens within the new incremental sentences tends to be biased towards tokens of new entity types (as illustrated in Figure 2). To ensure the incremental learning of a NER model less susceptible to this intertwined new-entity-type context, we initially address the biased context among new and old entity types within these new sentences by removing the new-entity-type tokens from the original sentence (as illustrated in Figure 3). At the t-th step, for each original sentence $X ^ { t } \in \mathcal { D } ^ { t }$ , we construct its erased version $\overline { { X ^ { t } } }$ by removing the new-entity-type tokens. The pseudo-label targets for $X ^ { t }$ and $\overline { { X ^ { t } } }$ are denoted by $\widetilde { Y } ^ { t }$ and $\widetilde { \mathbf { Y } } ^ { t } )$ , respectively. The corresponding predictions of the old model are denoted by $\widehat { Y } ^ { t - 1 } = \mathcal { M } ^ { t - 1 } ( X ^ { t } )$ and $\overline { { \widehat { Y } ^ { t - 1 } } } = \mathcal { M } ^ { t - 1 } ( \overline { { X ^ { t } } } )$

The set of sentence-duplets with $\mathcal { D } ^ { t }$ and $\overline { { \mathcal { D } ^ { t } } }$ is denoted as:

$$
\begin{array} { r l } & { ( \mathcal { D } ^ { t } , \overline { { \mathcal { D } ^ { t } } } ) = \Big \{ ( X _ { i } ^ { t } , \widetilde { Y } _ { i } ^ { t } , \overline { { X _ { i } ^ { t } } } , \overline { { \widetilde { Y } _ { i } ^ { t } } } ) \Big \} _ { i = 1 } ^ { \left| \mathcal { D } ^ { t } \right| } } \\ & { s . t . \ ( X _ { i } ^ { t } , \widetilde { Y } _ { i } ^ { t } ) \in \mathcal { D } ^ { t } , ( \overline { { X _ { i } ^ { t } } } , \overline { { \widetilde { Y } _ { i } ^ { t } } } ) \in \overline { { \mathcal { D } ^ { t } } } } \end{array}\tag{8}
$$

Using the sentence-duplets $( \mathcal { D } ^ { t } , \overline { { \mathcal { D } ^ { t } } } )$ at the t-th step, our sentence-duplet learning scheme updates the new model $\mathcal { M } ^ { t }$

![](images/c0679500592e066cb35871bc604de557aa85681e218800725bf6a7b184f7b473.jpg)  
Fig. 3. Illustration of our TBCL method. Initially, at step 1, the NER model is built from scratch with cross-entropy loss $\mathcal { L } _ { \mathrm { c e } }$ as the objective on dataset $\mathcal { D } ^ { 1 }$ . In subsequent steps $( e . g .$ , at step $\scriptstyle t = 2 ) .$ , we begin by obtaining sentence-duplets $( \mathcal { D } ^ { t } , \overline { { \mathcal { D } ^ { t } } } )$ and update the model through our innovative sentence-duplet learning scheme with the pseudo-labeling cross-entropy loss $\mathcal { L } _ { \mathrm { p c e } }$ and the KD loss $\mathcal { L } _ { \mathrm { k d } }$ and our unique contextual consistency loss $\mathcal { L } _ { \mathrm { c t x c . } }$ Here, GT stands for current Ground-Truth labels, CT denotes Current Target, PL represents Pseudo Labels, [O] denotes the non-entity type, and [X] indicates an Erased Token.

by optimizing the following loss function:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s - d u p } } ( X ^ { t } , \overline { { X ^ { t } } } ; \Theta ^ { t } ) = \mathcal { L } _ { \mathrm { r d p } } ( X ^ { t } , \widetilde { Y } ^ { t } , \widehat { Y } ^ { t - 1 } ; \Theta ^ { t } ) } \\ & { \qquad + \mathcal { L } _ { \mathrm { r d p } } ( \overline { { X ^ { t } } } , \overline { { \widetilde { Y } ^ { t } } } , \overline { { \widehat { Y } ^ { t - 1 } } } ; \Theta ^ { t } ) . } \end{array}\tag{9}
$$

which mirrors the form of Equation (7) applied to both the original sentence $X ^ { t }$ and its corresponding erased version $\overline { { X ^ { t } } }$ In Equation (9), all supervision signals are explicitly included as inputs of ${ \mathcal { L } } _ { \mathrm { r d p } }$ . Therefore, the loss for the original sentence $X ^ { t }$ is computed using $\widetilde { Y } ^ { t }$ and $\widehat { Y } ^ { t - 1 }$ , while the loss for the erased sentence $\overline { { X ^ { t } } }$ is computed using $\widetilde { Y } ^ { t }$ and $\overline { { \widehat { Y } ^ { t - 1 } } }$

2) Contextual Consistency Loss: To further mitigate the biased context, we introduce a contextual consistency loss $\mathcal { L } _ { \mathrm { c t x c } } ( X ^ { t } , \overline { { X ^ { t } } } ; \Theta ^ { t } )$ , which imposes a constraint on the old entity types’ context within each sentence pair $( X ^ { t } , { \overline { { X ^ { t } } } } )$ , ensuring consistency across contexts. For old-entity-type tokens, the context related to new entity types is present in the original sentence $X ^ { t }$ and removed in the corresponding erased sentence $\overline { { X ^ { t } } }$ . To simplify, we denote ${ \mathcal { O } } ( X ^ { t } )$ to depict the positions $\left\{ p _ { j } ^ { o } \right\} _ { j = 1 } ^ { \mathcal { O } \left( X ^ { t } \right) }$ of pseudo-labeled old-entity-type tokens within the sentence $X ^ { t }$ . To mitigate the impact of biased context between old and new entity types, the predictions of the updated model $\mathcal { M } ^ { t }$ on old-entity-type tokens with new-entity-type-related context should be consistent with those without such context.

Therefore, $\mathcal { L } _ { \mathrm { c t x c } } ( X ^ { t } , \overline { { X ^ { t } } } ; \Theta ^ { t } )$ is formulated as follows:

$$
\mathcal { L } _ { \mathrm { c t x c } } ( X ^ { t } , \overline { { { X ^ { t } } } } ; \Theta ^ { t } ) = \sum _ { p \in \mathcal { O } ( X ^ { t } ) } \Vert \widehat { Y } _ { p , : } ^ { t } - \overline { { \widehat { Y } _ { p , : } ^ { t } } } \Vert _ { 2 } ^ { 2 } ,\tag{10}
$$

where $\widehat { Y } ^ { t } = \mathcal { M } ^ { t } ( X ^ { t } ; \Theta ^ { t } )$ and $\overline { { \widehat { Y } ^ { t } } } = \mathcal { M } ^ { t } ( \overline { { X ^ { t } } } ; \Theta ^ { t } )$ are the current-model predictions for the original and erased sentences, respectively. The notation $\widehat { Y } _ { p , : } ^ { t }$ denotes the predicted probability distribution of the p-th token.

Finally, the total objective function of our proposed TBCL method is formulated as follows:

$$
\mathcal { L } _ { \mathrm { t b c l } } ( \Theta ^ { t } ) = \mathcal { L } _ { \mathrm { s - d u p } } ( X ^ { t } , \overline { { X ^ { t } } } ; \Theta ^ { t } ) + \beta \mathcal { L } _ { \mathrm { c t x c } } ( X ^ { t } , \overline { { X ^ { t } } } ; \Theta ^ { t } ) ,\tag{11}
$$

where $\beta$ is a hyperparameter that balances the significance of the loss terms.

## IV. EXPERIMENTAL SETUP

To ensure a fair comparison with previous SOTA INER methods, we adhere to the experimental setup specified in CFNER [12] and RDP [17]. This involves utilizing identical benchmark datasets, INER settings, competing baselines, assessment metrics, and foundational implementation details.

TABLE I  
THE SUMMARY FOR THREE NER DATASETS.
<table><tr><td>Dataset</td><td># Entity Type</td><td># Sample</td><td>Entity Type Sequence (Alphabetical Order)</td></tr><tr><td>CoNLL2003 [44]</td><td>4</td><td>21k</td><td>LOCATION, MISC, ORGANISATION, PERSON</td></tr><tr><td>I2B2 [45]</td><td>16</td><td>141k</td><td>AGE, CITY, COUNTRY, DATE, DOCTOR, HOSPITAL, IDNUM, MEDICALRECORD, ORGANIZATION, PATIENT, PHONE, PROFESSION, STATE, STREET, USERNAME, ZIP</td></tr><tr><td>OntoNotes5 [46]</td><td>18</td><td>77k</td><td>CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART</td></tr></table>

## A. Benckmark Datasets

We assess our TBCL using three NER datasets, which are CoNLL2003 [44], I2B2 [45], and OntoNotes5 [46]. Table I provides a summary of the dataset statistics.

To divide the training set into separate slices representing various incremental learning steps, we employ the greedy sampling algorithm presented in CFNER [12]. This method guarantees that samples from each entity type are predominantly concentrated in their respective slice, thereby more accurately reflecting real-world scenarios. Within each slice, we keep only the labels related to the entity types being learned and classify all other labels as non-entity types. For additional details on this greedy sampling algorithm, please refer to CFNER’s [12] Appendix B.

## B. INER Settings

In accordance with CFNER [12] and RDP [17], we examine two distinct INER scenarios for each dataset. In the first scenario, the base model (initial step) is trained with the same number of entity types as in subsequent incremental learning steps. In the second scenario, the base model is trained with half of all entity types. The first scenario presents a greater challenge, whereas the second more accurately reflects real-world situations, allowing models to acquire adequate knowledge before entering incremental learning. Entity types are learned sequentially during training, specifically in alphabetical order, with each corresponding data slice utilized to train the models sequentially. The base model is trained on FG (First Group) entity types, and each incremental learning step introduces PG (Progressive Group) entity types, denoted as FG-a-PG-b. For the CoNLL2003 [44] dataset, two settings are employed: FG-1-PG-1 and FG-2-PG-1. For the I2B2 [45] and OntoNotes5 [46] datasets, four settings are established: FG-1-PG-1, FG-2-PG-2, FG-8-PG-1, and FG-8-PG-2. During evaluation, only the labels pertinent to the current entity types in the validation set are retained, while all others are classified as non-entity types. In each incremental learning step, the model with the best validation performance is selected for testing and for the subsequent incremental learning phase. When testing, labels for all previously learned entity types are preserved in the test set, while others are marked as non-entity types.

## C. Competing Baselines

We incorporate the following baselines, including SOTA INER methods: ExtendNER [8], CFNER [12], and RDP [17].

Additionally, we include the lower-bound method, Only Finetuning, which uses new data directly to finetune the model without employing any anti-forgetting techniques. Furthermore, we integrate incremental learning methods adapted from computer vision, such as PODNet [47], LUCIR [48], and Self-Training [49], which are tailored to the INER scenario [12]. Detailed introductions to these baselines are provided below:

• PODNet [47]: Originally designed to tackle the issue of catastrophic forgetting in incremental learning for image classification, this method has been adapted to the INER scenario [12]. The model’s total loss is composed of classification and distillation losses. For classification, instead of the standard cross-entropy loss, PODNet utilizes neighborhood component analysis loss. In terms of distillation loss, PODNet enforces constraints on the output of each intermediate layer.

• LUCIR [48]: This method creates an incremental learning framework for image classification tasks, which is then transferred to the INER scenario [12]. Similar to POD-Net, it aims to minimize catastrophic forgetting. Its loss function consists of three main components: (1) crossentropy loss for samples containing new entity types; (2) distillation loss between features extracted by the previous and current models; and (3) margin-ranking loss for the retained samples with the previous entity types.

• Self-Training [49], [50]: This method directly employs the pre-existing model to label the non-entity type tokens with their previous entity types. The updated model is then trained using new data that encompasses annotations for all previously identified entity types. Its goal is to minimize cross-entropy loss on these data, ensuring efficient model training.

• ExtendNER [8]: Similar to Self-Training, ExtendNER applies KD to the INER. It computes cross-entropy loss for entity type tokens and KL divergence loss for non-entity tokens. During training, ExtendNER aims to minimize the combined losses of cross-entropy and KL divergence.

• CFNER [12]: Building on ExtendNER, this method introduces a causal framework for INER that distills causal effects within the non-entity type tokens. Initially, the old model identifies the non-entity type tokens associated with previous entity types for KD, while curriculum learning helps reduce recognition errors.

• RDP [17]: This method is a pseudo-labeling based INER method, targeting catastrophic forgetting and the semantic shift issue of the non-entity type tokens within INER. It incorporates a task relation distillation loss to achieve a balance between stability and adaptability, and utilizes a prototypical pseudo-label strategy to reduce label noise and address the issue of semantic shift.

## D. Assessment Metrics

At each incremental learning step, we first compute the F1 score for each entity type, then calculate the macro-average F1 scores (Macro-F1) for old entity types, new entity types, and all entity types at the current step. We also compute the micro-average F1 score (Micro-F1) for all current entity types. We provide the average scores over all steps, including the first one (except for old and new entity types). Additionally, we provide line plots showing step-wise changes in scores. To check the statistical significance of the improvements, we apply a paired t-test at a significance level of 0.05.

TABLE II  
COMPARISONS WITH BASELINES ON THE CONLL2003 [44] DATASET. THE RED DENOTES THE HIGHEST RESULT, WHILE THE BLUE DENOTES THE SECOND HIGHEST RESULT. THE MARKER † REFERS TO SIGNIFICANT TEST p-value < 0.05 COMPARING WITH RDP [17]. ∗ REPRESENTS RESULTS FROM OUR RE-IMPLEMENTATION. OTHER BASELINE RESULTS ARE DIRECTLY REFERENCED FROM RDP [17].
<table><tr><td rowspan="2">Baseline</td><td colspan="4">FG-1-PG-1</td><td colspan="4">FG-2-PG-1</td></tr><tr><td>Old Entity Types</td><td>New Entity Types</td><td colspan="2">All Entity Types</td><td>Old Entity Types</td><td>New Entity Types</td><td colspan="2">All Entity Types</td></tr><tr><td></td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td></tr><tr><td>Only Finetuning</td><td></td><td></td><td>50.84±0.10</td><td>40.64±0.16</td><td></td><td></td><td>57.45±0.05</td><td>43.58±0.18</td></tr><tr><td>PODNet [47]</td><td></td><td></td><td>36.74±0.52</td><td>29.43±0.28</td><td></td><td></td><td>59.12±0.54</td><td>58.39±0.99</td></tr><tr><td>LUCIR [48]</td><td></td><td></td><td>74.15±0.43</td><td>70.48±0.66</td><td></td><td></td><td>80.53±0.31</td><td>77.33±0.31</td></tr><tr><td>Self-Training [50]</td><td></td><td></td><td>76.17±0.91</td><td>72.88±1.12</td><td></td><td></td><td>76.65±0.24</td><td>66.72±0.11</td></tr><tr><td>ExtendNER* [8]</td><td>70.42±0.22</td><td>71.84±0.69</td><td>76.07±0.35</td><td>73.06±0.29</td><td>56.33±2.00</td><td>79.34±0.71</td><td>77.89±0.42</td><td>69.92±1.02</td></tr><tr><td>ExtendNER [8]</td><td></td><td></td><td>76.36±0.98</td><td>73.04±1.80</td><td></td><td></td><td>76.66±0.66</td><td>66.36±0.64</td></tr><tr><td>CFNER* [12]</td><td>76.82±0.58</td><td>78.87±0.44</td><td>80.29±0.21</td><td>78.44±0.24</td><td>69.94±1.63</td><td>84.08±0.26</td><td>81.52±0.43</td><td>77.20±0.82</td></tr><tr><td>CFNER [12]</td><td></td><td>82.52±0.32</td><td>80.91±0.29</td><td>79.11±0.50</td><td></td><td></td><td>80.83±0.36</td><td>75.20±0.32</td></tr><tr><td>RDP [17]</td><td>79.87±0.37</td><td></td><td>82.55±0.26</td><td>80.64±0.12</td><td>82.44±0.73</td><td>88.34±0.35</td><td>85.32±0.36</td><td>83.59±0.37</td></tr><tr><td>TBCL (Ours)</td><td>80.95±0.28†</td><td>83.68±0.19†</td><td>84.27±0.38†</td><td>81.92±0.16†</td><td>83.76±0.66†</td><td>89.58±0.47†</td><td>86.71±0.56†</td><td>84.94±0.53†</td></tr><tr><td>Improve</td><td>介1.08</td><td>介1.16</td><td>介1.72</td><td>介1.28</td><td>介1.32</td><td>介1.24</td><td>介1.39</td><td>介1.35</td></tr></table>

TABLE III

COMPARISONS WITH BASELINES ON THE I2B2 [45] DATASET. THE RED DENOTES THE HIGHEST RESULT, WHILE THE BLUE DENOTES THE SECOND HIGHEST RESULT. THE MARKER † REFERS TO SIGNIFICANT TEST p-value < 0.05 COMPARING WITH RDP [17]. ∗ REPRESENTS RESULTS FROM OUR RE-IMPLEMENTATION. OTHER BASELINE RESULTS ARE DIRECTLY REFERENCED FROM RDP [17].
<table><tr><td rowspan="3">Baseline</td><td colspan="4">FG-1-PG-1</td><td colspan="4">FG-2-PG-2</td></tr><tr><td>Old Entity Types</td><td>New Entity Types</td><td colspan="2">All Entity Types</td><td>Old Entity Types</td><td>New Entity Types</td><td colspan="2">All Entity Types</td></tr><tr><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td></tr><tr><td>Only Finetuning</td><td></td><td></td><td>17.43±0.54</td><td>13.81±1.14</td><td></td><td></td><td>28.57±0.26</td><td>21.43±0.41</td></tr><tr><td>PODNet [47]</td><td></td><td></td><td>12.31±0.35</td><td>17.14±1.03</td><td></td><td></td><td>34.67±2.65</td><td>24.62±1.76</td></tr><tr><td>LUCIR [48]</td><td></td><td></td><td>43.86±2.43</td><td>31.31±1.62</td><td></td><td></td><td>64.32±0.76</td><td>43.53±0.59</td></tr><tr><td>Self-Training [50]</td><td></td><td></td><td>31.98±2.12</td><td>14.76±1.31</td><td></td><td></td><td>55.44±4.78</td><td>33.38±3.13</td></tr><tr><td>ExtendNER* [8]</td><td>15.67±2.47</td><td>30.01±5.07</td><td>41.65±10.11</td><td>23.11±2.70</td><td>35.66±2.26</td><td>50.78±1.17</td><td>67.60±1.15</td><td>42.58±1.59</td></tr><tr><td>ExtendNER [8]</td><td></td><td></td><td>42.85±2.86</td><td>24.05±1.35</td><td></td><td></td><td>57.01±4.14</td><td>35.29±3.38</td></tr><tr><td>CFNER* [12]</td><td>34.35±1.00</td><td>45.70±1.23</td><td>64.79±0.26</td><td>37.79±0.65</td><td>48.14±1.10</td><td>56.56±1.08</td><td>72.58±0.59</td><td>51.71±0.84</td></tr><tr><td>CFNER [12]</td><td></td><td></td><td>62.73±3.62</td><td>36.26±2.24</td><td></td><td></td><td>71.98±0.50</td><td>49.09±1.38</td></tr><tr><td>RDP [17]</td><td>41.61±2.71</td><td>50.32±1.02</td><td>71.39±1.01</td><td>44.00±2.31</td><td>49.07±1.01</td><td>62.13±0.77</td><td>77.45±0.55</td><td>53.48±0.66</td></tr><tr><td>TBCL (Ours)</td><td>44.79±1.78†</td><td>52.08±0.42†</td><td>72.14±0.75†</td><td>48.66±1.40†</td><td>49.62±1.37†</td><td>62.85±0.71†</td><td>78.19±0.26†</td><td>54.46±0.63†</td></tr><tr><td>Improve</td><td>个3.18</td><td>介1.76</td><td>介0.75</td><td>介4.66</td><td>个0.55</td><td>介0.72</td><td>介0.74</td><td>个0.98</td></tr><tr><td></td><td colspan="4">FG-8-PG-1</td><td colspan="4">FG-8-PG-2</td></tr><tr><td>Baseline</td><td>Old Entity Types</td><td>New Entity Types</td><td>All Entity Types</td><td></td><td>Old Entity Types</td><td>New Entity Types</td><td>All Entity Types</td><td></td></tr><tr><td></td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td></tr><tr><td>Only Finetuning</td><td></td><td></td><td>20.83±1.78</td><td>18.11±1.66</td><td></td><td></td><td>23.60±0.15</td><td>23.54±0.38</td></tr><tr><td>PODNet [47]</td><td></td><td></td><td>39.26±1.38</td><td>27.23±0.93</td><td></td><td></td><td>36.22±12.9</td><td>26.08±7.42</td></tr><tr><td>LUCIR [48]</td><td></td><td></td><td>57.86±0.87</td><td>33.04±0.39</td><td></td><td></td><td>68.54±0.27</td><td>46.94±0.63</td></tr><tr><td>Self-Training [50]</td><td></td><td></td><td>49.51±1.35</td><td>23.77±1.01</td><td></td><td></td><td>48.94±6.78</td><td>29.00±3.04</td></tr><tr><td>ExtendNER* [8]</td><td>20.33±0.99</td><td>27.65±1.30</td><td>45.14±2.91</td><td>27.41±0.88</td><td>27.01±2.10</td><td>39.22±1.19</td><td>56.48±2.41</td><td>38.88±1.38</td></tr><tr><td>ExtendNER [8]</td><td></td><td></td><td>43.95±2.01</td><td>23.12±1.79</td><td></td><td></td><td>52.25±5.36</td><td>30.93±2.77</td></tr><tr><td>CFNER* [12]</td><td>31.13±1.56</td><td>37.95±2.08</td><td>56.66±3.22</td><td>36.84±1.35</td><td>43.94±1.25</td><td>49.93±1.19</td><td>69.12±0.94</td><td>51.61±0.87</td></tr><tr><td>CFNER [12]</td><td>61.75±0.36</td><td>52.47±1.20</td><td>59.79±1.70 77.50±1.26</td><td>37.30±1.15 62.99±0.36</td><td>60.44±0.92</td><td>56.17±1.17</td><td>69.07±0.89 80.08±0.40</td><td>51.09±1.05 63.72±0.71</td></tr><tr><td>RDP [17]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TBCL (Ours)</td><td>66.53±0.47†</td><td>58.13±0.41†</td><td>83.99±0.29†</td><td>67.27±0.28†</td><td>62.49±1.08†</td><td>58.96±0.51†</td><td>82.37±1.19†</td><td>66.51±1.32†</td></tr><tr><td>Improve</td><td>介4.78</td><td>介5.66</td><td>介6.49</td><td>介4.28</td><td>介2.05</td><td>介2.79</td><td>介2.29</td><td>介2.79</td></tr></table>

## E. Implementation Details

In line with previous INER methods [8], [11], [12], [17], we utilize the “BIO” tagging scheme for all three datasets. This scheme designates two labels for each entity type: B-type (i.e., for the beginning of an entity) and I-type (i.e., for the interior of an entity). Our NER model employs bert-base-cased [37] as the encoder, complemented by a fully connected layer for classification. The model is implemented using the PyTorch framework [51] and is built upon the BERT implementation from Huggingface [52]. As per CFNER [12] and RDP [17], the model is trained for 20 epochs if PG is set to 2, and for 10 epochs in other cases. The batch size, learning rate, and balancing hyper-parameter γ are set to 8, 4e − 4, and 0.01, respectively. All experiments are run on a single NVIDIA A100 GPU with 40GB of memory, and each experiment is repeated five times to ensure statistical reliability.

## V. EXPERIMENTAL RESULTS

To showcase the superiority and effectiveness of our TBCL method, we conducted extensive experiments focusing on the following research questions (RQ):

TABLE IV  
COMPARISONS WITH BASELINES ON THE ONTONOTES5 [46] DATASET. THE RED DENOTES THE HIGHEST RESULT, WHILE THE BLUE DENOTES THE SECOND HIGHEST RESULT. THE MARKER † REFERS TO SIGNIFICANT TEST p-value < 0.05 COMPARING WITH RDP [17]. ∗ REPRESENTS RESULTS FROM OUR RE-IMPLEMENTATION. OTHER BASELINE RESULTS ARE DIRECTLY REFERENCED FROM RDP [17].
<table><tr><td rowspan="3">Baseline</td><td colspan="4">FG-1-PG-1</td><td colspan="4">FG-2-PG-2</td></tr><tr><td>Old Entity Types</td><td>New Entity Types</td><td colspan="2">All Entity Types</td><td>Old Entity Types</td><td>New Entity Types</td><td colspan="2">All Entity Types</td></tr><tr><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td></tr><tr><td>Only Finetuning</td><td></td><td></td><td>15.27±0.26</td><td>10.86±1.11</td><td></td><td></td><td>25.85±0.11</td><td>20.55±0.24</td></tr><tr><td>PODNet [47]</td><td></td><td></td><td>9.06±0.56</td><td>8.36±0.57</td><td></td><td></td><td>34.67±1.08</td><td>24.62±0.85</td></tr><tr><td>LUCIR [48]</td><td></td><td></td><td>28.18±1.15</td><td>21.11±0.84</td><td></td><td></td><td>64.32±1.79</td><td>43.53±1.11</td></tr><tr><td>Self-Training [50]</td><td></td><td></td><td>50.71±0.79</td><td>33.24±1.06</td><td></td><td></td><td>68.93±1.67</td><td>50.63±1.66</td></tr><tr><td>ExtendNER* [8]</td><td>29.63±1.14</td><td>41.11±1.07</td><td>51.36±0.77</td><td>33.38±0.98</td><td>44.90±6.49</td><td>51.97±2.87</td><td>63.03±9.39</td><td>47.64±5.15</td></tr><tr><td>ExtendNER [8]</td><td></td><td></td><td>50.53±0.86</td><td>32.84±0.84</td><td></td><td></td><td>67.61±1.53</td><td>49.26±1.49</td></tr><tr><td>CFNER* [12]</td><td>37.74±1.77</td><td>55.01±0.20</td><td>58.44±0.71</td><td>41.75±1.51</td><td>52.70±0.26</td><td>60.54±0.99</td><td>72.10±0.31</td><td>55.02±0.35</td></tr><tr><td>CFNER [12]</td><td></td><td></td><td>58.94±0.57</td><td>42.22±1.10</td><td></td><td></td><td>72.59±0.48</td><td>55.96±0.69</td></tr><tr><td>RDP [17]</td><td>52.71±0.48</td><td>59.66±0.34</td><td>68.28±1.09</td><td>53.56±0.39</td><td>55.77±0.71</td><td>63.93±0.47</td><td>74.38±0.26</td><td>57.73±0.54</td></tr><tr><td>TBCL (Ours)</td><td>54.36±0.63†</td><td>61.53±0.67†</td><td>70.04±0.73†</td><td>55.63±0.51†</td><td>57.70±0.69†</td><td>66.09±0.59†</td><td>76.40±0.44†</td><td>59.84±0.53†</td></tr><tr><td>Improve</td><td>介1.65</td><td>介1.87</td><td>介1.76</td><td>介2.07</td><td>介1.93</td><td>介2.16</td><td>介2.02</td><td>介2.11</td></tr><tr><td rowspan="3">Baseline</td><td colspan="5">FG-8-PG-1</td><td colspan="3">FG-8-PG-2</td></tr><tr><td>Old Entity Types</td><td>New Entity Types</td><td></td><td>All Entity Types</td><td>Old Entity Types</td><td>New Entity Types</td><td>All Entity Types</td><td></td></tr><tr><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td></tr><tr><td></td><td></td><td></td><td>17.63±0.57</td><td>12.23±1.08</td><td></td><td></td><td>29.81±0.12</td><td>20.05±0.16</td></tr><tr><td>Only Finetuning PODNet [47]</td><td></td><td></td><td>29.00±0.86</td><td>20.54±0.91</td><td></td><td></td><td>37.38±0.26</td><td>25.85±0.29</td></tr><tr><td>LUCIR [48]</td><td></td><td></td><td>66.46±0.46</td><td>46.29±0.38</td><td></td><td></td><td>76.17±0.09</td><td>55.58±0.55</td></tr><tr><td>Self-Training [50]</td><td></td><td></td><td>73.59±0.66</td><td>49.41±0.77</td><td></td><td></td><td>77.07±0.62</td><td>53.32±0.63</td></tr><tr><td>ExtendNER* [8]</td><td>48.73±0.63</td><td>46.09±0.56</td><td>73.65±0.19</td><td>50.55±0.56</td><td>51.11±0.70</td><td>57.41±0.67</td><td>77.86±0.10</td><td>55.21±0.51</td></tr><tr><td>ExtendNER [8]</td><td></td><td></td><td>73.12±0.93</td><td>49.55±0.90</td><td></td><td></td><td>76.85±0.77</td><td>54.37±0.57</td></tr><tr><td>CFNER* [12]</td><td>57.28±0.49</td><td>59.73±0.60</td><td>78.25±0.33</td><td>58.64±0.42</td><td>58.03±0.47</td><td>65.01±0.72</td><td>80.09±0.37</td><td>61.06±0.37</td></tr><tr><td>CFNER [12]</td><td></td><td>62.58±3.44</td><td>78.92±0.58 79.89±0.20</td><td>57.51±1.32</td><td>65.25±1.59</td><td>67.10±1.30</td><td>80.68±0.25</td><td>60.52±0.84</td></tr><tr><td>RDP [17]</td><td>62.10±6.44</td><td></td><td></td><td>63.20±0.58</td><td></td><td></td><td>83.30±0.30</td><td>66.92±1.26</td></tr><tr><td>TBCL (Ours)</td><td>65.47±0.33†</td><td>64.01±0.42†</td><td>81.41±0.16†</td><td>66.51±0.28†</td><td>66.40±0.49†</td><td>69.22±0.85†</td><td>83.80±0.21†</td><td>68.43±0.41†</td></tr><tr><td>Improve</td><td>介3.37</td><td>介1.43</td><td>介1.52</td><td>介3.31</td><td>介1.15</td><td>介2.12</td><td>个0.50</td><td>介1.51</td></tr></table>

• RQ1: How does the quantitative performance of TBCL compare to that of competitive baselines?

• RQ2: What effects do the key components of TBCL have?

• RQ3: How does the qualitative performance of TBCL compare to that of competitive baselines?

• RQ4: How does TBCL maintain stability across different incremental scenarios, including 1) varying entity type learning orders, 2) employing larger language models as the encoder backbone, and 3) handling finer-grained entity types?

## A. Main Results (RQ1)

In this subsection, we performed extensive experiments on the CoNLL2003 [44], I2B2 [45], and OntoNotes5 [46] datasets under ten INER settings to showcase the quantitative performance of our TBCL method compared to seven competitive baselines. Tables II, III, and IV present the average Macro-F1 scores for the old, new, and all entity types across all INER steps for each setting, along with the average Micro-F1 score for all entity types across all INER steps. Each experiment was conducted five times to ensure statistical robustness.

Specifically, as shown in Table II, for the FG-1-PG-1 and FG-2-PG-1 settings of the CoNLL dataset [44], our TBCL method achieved average improvements of 1.72 and 1.39 in Micro-F1 scores, and 1.28 and 1.35 in Macro-F1 scores, respectively, for all entity types at each step compared to the previous SOTA RDP method. For old entity types at each step, our TBCL method achieved average improvements of 1.08 and 1.32 in Macro-F1 scores, respectively. For new entity types at each step, our TBCL method achieved average improvements of 1.16 and 1.24 in Macro-F1 scores, respectively.

As shown in Table III, for the FG-1-PG-1, FG-2-PG-2, FG-8-PG-1, and FG-8-PG-2 settings of the I2B2 dataset [45], our TBCL method achieved average improvements of 0.75, 0.74, 6.49, and 2.29 in Micro-F1 scores, and 4.66, 0.98, 4.28, and 2.79 in Macro-F1 scores, respectively, for all entity types at each step compared to the previous SOTA RDP method. For old entity types at each step, our TBCL method achieved average improvements of 3.18, 0.55, 4.78, and 2.05 in Macro-F1 scores, respectively. For new entity types at each step, our TBCL method achieved average improvements of 1.76, 0.72, 5.66, and 2.79 in Macro-F1 scores, respectively.

As shown in Table IV, for the FG-1-PG-1, FG-2-PG-2, FG-8-PG-1, and FG-8-PG-2 settings of the OntoNotes5 dataset [46], our TBCL method achieved average improvements of 1.76, 2.02, 1.52, and 0.50 in Micro-F1 scores, and 2.07, 2.11, 3.31, and 1.51 in Macro-F1 scores, respectively, for all entity types at each step compared to the previous SOTA RDP method. For old entity types at each step, our TBCL method achieved average improvements of 1.65, 1.93, 3.37, and 1.15 in Macro-F1 scores, respectively. For new entity types at each step, our TBCL method achieved average improvements of 1.87, 2.16, 1.43, and 2.12 in Macro-F1 scores, respectively.

These results quantitatively validate the superiority and effectiveness of our TBCL method over competitive baselines, highlighting its capability to learn a robust INER model. The findings suggest enhanced resilience to catastrophic forgetting and semantic shift problems. Moreover, our TBCL method further enhances the performance of the pseudo-labeling based RDP method by rectifying biased context, thereby mitigating old-entity-type forgetting and new-entity-type overfitting.

All Entity Type  
![](images/18f3a941682e924563ae9608392a72c44d93ca49c0b92e014f627909bf66c4f4.jpg)

![](images/cf2a50362435db72c437ec166f0a4b2f18ef411d5625d49f466fb70a435b0083.jpg)

![](images/0e54f966245a534d50d2375cc41fdf46848681d62cd019745029ab4bc7c7981f.jpg)

![](images/4b0818198d2d4f39554c3facd3b5ac8ff0a8c3e4af5b5279679304abf892c5fd.jpg)  
Fig. 4. Comparison of the step-wise Macro-F1 scores for old, new, and all entity types, along with the step-wise Micro-F1 scores for all entity types, using the FG-8-PG-1 setting of the I2B2 dataset [45]. Our TBCL method consistently outperforms the previous SOTA INER baseline, RDP [17], in most step-wise evaluations.

To provide a more detailed discussion of these numerical results, we analyzed TBCL’s performance across CoNLL2003, I2B2, and OntoNotes5 datasets (Tables II–IV). On CoNLL2003, FG-1-PG-1 and FG-2-PG-1 yield modest Micro-F1 gains of 1.39–1.72 and Macro-F1 gains of 1.28–1.35, as the dataset is simple and RDP performs near its upper bound. In I2B2, the largest Micro-F1 improvements are observed in FG-8-PG-1 (+6.49) and FG-8-PG-2 (+2.29), while FG-1-PG-1 and FG-2-PG-2 show smaller gains, reflecting the higher potential for improvement when the initial number of types is larger. OntoNotes5 shows similar trends, with FG-2- PG-2 having the largest Micro-F1 gain (+2.02) and FG-8-PG-1 the largest Macro-F1 gain (+3.31). These results indicate that TBCL is particularly effective in challenging, multientity-type incremental learning settings, while still providing consistent improvements in simpler settings. Overall, TBCL demonstrates both incremental gains and substantial methodological advancements, emphasizing its practical and scientific contributions.

## B. Step-wise Results (RQ1)

As illustrated in Figure 4, we provide a detailed stepwise comparison to thoroughly assess the effectiveness of our TBCL method in handling INER scenarios. The results clearly show that TBCL consistently outperforms the previous SOTA INER model, RDP [17], in most step-wise evaluations. This is evident in the FG-8-PG-1 setting of the I2B2 dataset [45]. These findings highlight the robustness and adaptability of

THE ABLATION STUDY OF OUR TBCL UNDER THE FG-1-PG-1 SETTING OF THE I2B2 DATASET [45]. IN COMPARISON TO OUR TBCL METHOD, ALL ABLATION VARIANTS SHOW A SIGNIFICANT DECLINE IN INER PERFORMANCE, CONFIRMING THE NECESSITY OF EACH COMPONENT FOR COLLABORATIVELY ADDRESSING INER. BOLD DENOTES THE BEST RESULTS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Old Entity Types Avg. Macro-F1</td><td rowspan="2">New Entity Types Avg. Macro-F1</td><td colspan="2">All Entity Types</td></tr><tr><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td></tr><tr><td>Baseline (RDP [17])</td><td>41.61±2.71</td><td>50.32±1.02</td><td>71.39±1.01</td><td>44.00±2.31</td></tr><tr><td>Baseline + double</td><td>39.77±2.03</td><td>47.47±1.13</td><td>70.26±1.39</td><td>42.52±1.66</td></tr><tr><td>Baseline + duplet</td><td>43.63±1.48</td><td>51.55±1.05</td><td>71.88±0.89</td><td>47.39±1.28</td></tr><tr><td>Baseline + duplet</td><td>43.63±1.48</td><td>51.55±1.05</td><td>71.88±0.89</td><td>47.39±1.28</td></tr><tr><td>Baseline + duplet + ctxc (TBCL)</td><td>44.79±1.78</td><td>52.08±0.42</td><td>72.14±0.75</td><td>48.66±1.40</td></tr></table>

TBCL, as it effectively maintains high performance when encountering new and evolving entity types throughout the learning process. The substantial improvement over RDP suggests that TBCL has the potential to set a new benchmark for INER tasks, particularly in complex, real-world data environments.

## C. Ablation Study (RQ2)

In this subsection, we begin with ablation experiments to validate the effectiveness of the sentence-duplet learning scheme. Following this, we proceed to validate the contextual consistency loss through further experiments. All ablation experiments are conducted under the FG-1-PG-1 setting of the I2B2 dataset [45]. Our baseline is established from the SOTA pseudo-labeling based INER method RDP [17].

1) Effectiveness of the sentence-duplet learning scheme: To demonstrate the effectiveness of the sentence-duplet learning scheme, we compare the performance of the Baseline model with our enhanced approach, which includes the original sentence alongside its corresponding new-entity-type-erased version (referred to as Baseline+duplet). The results, presented in Table V, clearly show that Baseline+duplet outperforms the Baseline, showcasing that the sentence-duplet learning scheme can mitigate the forgetting of old entity types while preventing the overfitting of new ones by correcting biased contexts present in the Baseline. Additionally, we compare Baseline+duplet with Baseline+double to evaluate the effect of increased sample size. Baseline+duplet uses original sentences and their corresponding erased versions in each minibatch, while Baseline+double duplicates original sentences. As shown in Table V, Baseline+duplet delivers superior performance compared to Baseline+double, suggesting that merely increasing the number of samples does not guarantee performance enhancement.

2) Effect of contextual consistency loss: We evaluate the performance of Baseline+duplet enhanced with our contextual consistency loss (denoted as Baseline+duplet+ctxc). As shown in Table V, the contextual consistency loss further enhances INER performance by ensuring consistency across contexts.

## D. Case Study (RQ3)

In this subsection, we present the qualitative performance of our TBCL method compared to leading baselines, including CFNER [12] and RDP [17]. We provide visual examples from the last task of the FG-8-PG-2 setting on the OntoNotes5

<table><tr><td colspan="4">Input Sentence Thanks very much,</td><td>NBC &#x27;s</td><td>Jim</td><td>Maceda</td><td>tonight</td><td>for the trial of the</td><td>Lockerbie</td><td></td><td>103</td><td>Pan</td><td>Am</td><td>crash.</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CFNER</td><td>[0]</td><td>[0]</td><td>[0] [0]</td><td>[0]</td><td>[0] [B-PER]</td><td>[I-PER]</td><td>[B-TIME]</td><td>[0] [0] </td><td>[0] [0] [0]</td><td>[B-ORG]</td><td>[0]</td><td>[B-PRO] [I-PRO]</td><td></td><td>[0] [0]</td><td></td><td></td><td></td><td></td></tr><tr><td>RDP</td><td>[0]</td><td>[0]</td><td>[0] [0]</td><td>[0]</td><td>[0] [B-PER]</td><td>[l-PER]</td><td>[B-TIME]</td><td>[0] [0] [0] [0][0]</td><td>[B-GPE]</td><td>[l-PRO]</td><td></td><td>[I-PRO] [I-PRO]</td><td>[0] [0]</td><td></td><td></td><td></td></tr><tr><td>TBCL (Ours)</td><td>[0]</td><td>[0]</td><td>[0] [0]</td><td>[B-ORG]</td><td>[0] [B-PER]</td><td>[I-PER]</td><td>[B-TIME]</td><td>[0][0] [0] [0] [0]</td><td>[B-GPE]</td><td></td><td>[I-PRO] [I-PRO] [I-PRO]</td><td></td><td>[0] [0]</td><td></td></tr><tr><td>Golden Labels</td><td>[0]</td><td>[0]</td><td>[0] [0]</td><td>[B-ORG]</td><td>[0] [B-PER]</td><td>[1-PER]</td><td>[B-TIME]</td><td>[0] [0] [0] [0] [0]</td><td>[B-GPE]</td><td></td><td>[B-PRO] [I-PRO] [I-PRO]</td><td></td><td>[0] [0]</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Input Sentence</td><td>Half</td><td></td><td>of the Palestinian population is under</td><td></td><td></td><td>the</td><td>age of</td><td>14</td><td>and many have been influenced by military factions like Hamas and Hezbollah not by</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CFNER</td><td>[0]</td><td>[0] [0]</td><td>[B-GPE]</td><td>[0]</td><td>[0][0]</td><td></td><td>[B-DATE] [I-DATE] [I-DATE] [I-DATE]</td><td>[0] [0]</td><td>[0]</td><td>[0] [0]</td><td>[0] [0]</td><td>[0]</td><td>[0]</td><td>[B-ORG][0] [0]</td></tr><tr><td>RDP</td><td>[0]</td><td>[0][0]</td><td>[B-NORP]</td><td>[0]</td><td>[0] [0]</td><td></td><td>[B-DATE] [I-DATE] [I-DATE] [I-DATE]</td><td>[0] ][0]</td><td>[0] [0]</td><td>[0]</td><td>[0] [0]</td><td>[0]</td><td>[0] [B-ORG] [0]</td><td>[0] [0] [0]</td></tr><tr><td>TBCL (Ours)</td><td></td><td>[B-CARD] [0] [0] 1</td><td>[B-NORP]</td><td>[0]</td><td>[0] [0]</td><td></td><td>[B-DATE] [I-DATE] [I-DATE] [1-DATE]</td><td>[0] [0]</td><td>[0]</td><td>[0] [0]</td><td>[0] [0]</td><td>[0]</td><td>[0] [B-ORG] [0]</td><td>[B-ORG] [0] [0] [B-PER] [0]</td></tr><tr><td>Golden Labels</td><td></td><td>[B-CARD] [0] [0]</td><td>[B-NORP]</td><td>[0]</td><td>[0][0]</td><td></td><td>[B-DATE] [I-DATE] [I-DATE] [I-DATE]</td><td>[0] [0]</td><td>[0] [0]</td><td>[0]</td><td>[0] [0]</td><td>[0]</td><td>[0] [B-ORG] [0]</td><td>[B-ORG] [0] [0] [B-PER][0]</td></tr></table>

Fig. 5. Two real NER examples sampled from the OntoNotes5 [46] test set. B- and I- indicate the beginning and inside of named entities, respectively. The labels [O], [ORG], [PER], [TIME], [GPE], [PRO], [CARD], [NORP], and [DATE] correspond to non-entity, Organization, Person, Time, Country/City/State, Product, Cardinal, Nationality/Religion/Political group, and Date, respectively. All predictions are from the last task of the FG-8-PG-2 setting. These visualizations qualitatively demonstrate the effectiveness and superiority of our proposed TBCL method.

## 10 different entity type orders for the FG-8-PG-2 settings of the I2B2 dataset

![](images/3865bc8a19c56c465bfff3a99c14a3a92d30d664aa0b50e188acc67e3cfcce7b.jpg)  
Fig. 6. The boxplots display the average Macro-F1 scores for the old, new, and all entity types across INER steps for 10 random entity type orders. TBCL is significantly better and more stable than RDP.

dataset [46], as shown in Figure 5. These examples demonstrate TBCL’s superior ability to sequentially learn and preserve multiple entity types, highlighting its effectiveness in the incremental learning framework.

## E. More Explorations (RQ4)

1) Stability Concerning Entity Type Orders: Recent studies have shown that existing incremental learning methods may be prone to instability, with entity type ordering having a significant impact on performance [53], [54]. However, in real-world scenarios, the optimal entity type order is never known beforehand. Therefore, the performance of an ideal INER method should be as invariant to entity type order as possible. In all previous experiments, this entity type order has been kept constant (i.e., alphabetical order), as defined in [12], [17]. Here, we report results in Figure 6 in the form of boxplots, obtained by applying 10 random permutations of the entity type order under the FG-8-PG-2 setting of the I2B2 dataset [45]. Figure 6 (from left to right) shows the average Macro-F1 scores for the old, new, and all entity types across all INER steps. As shown in Figure 6, the boxplot for TBCL has higher values, a more concentrated data distribution, and a smaller range of variation compared to RDP. Therefore, our proposed TBCL method performs significantly better and is more stable than the previous SOTA RDP approach.

TABLE VI  
PERFORMANCE ON THE I2B2 DATASET [45] UNDER THE FG-8-PG-2 SETTING WITH DIFFERENT ENCODER BACKBONES. BOLD DENOTES THE BEST RESULTS FOR EACH METHOD.
<table><tr><td rowspan="2">Encoder Backbone</td><td colspan="2">RDP</td><td colspan="2">TBCL</td></tr><tr><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td><td>Avg. Micro-F1</td><td>Avg. Macro-F1</td></tr><tr><td>bert-base-cased</td><td>80.08±0.40</td><td>63.72±0.71</td><td>82.37±1.19</td><td>66.51±1.32</td></tr><tr><td>bert-large-cased</td><td>81.25±1.30</td><td>65.10±1.45</td><td>83.39±0.82</td><td>68.57±1.13</td></tr><tr><td>roberta-large</td><td>82.50±0.95</td><td>66.80±1.21</td><td>84.20±1.06</td><td>69.18±1.17</td></tr></table>

2) Larger Encoder Backbones: In previous experiments, our encoder backbone was primarily based on bert-base-cased [37]. Here, we evaluate the performance of our TBCL method alongside the previous SOTA RDP [17] using larger pre-trained language models, specifically bert-large-cased [37] and roberta-large [56]. The results on the I2B2 dataset [45] under the FG-8-PG-2 setting are presented in Table VI. As shown, both RDP and TBCL benefit from larger encoder backbones: for example, RDP’s Micro-F1 improves from 80.08 to 82.50, and TBCL’s from 82.37 to 84.20. Across all backbones, TBCL consistently outperforms RDP in both Micro-F1 and Macro-F1, demonstrating the robustness and general applicability of our method with different pre-trained encoders. These results confirm that TBCL not only leverages larger language models effectively but also maintains its performance advantage over strong baselines.

TABLE VII  
COMPARISONS WITH BASELINES ON THE FEW-NERD [55] DATASET AT EACH INCREMENTAL STEP (TOTAL 11 STEPS). THE RED DENOTES THE HIGHEST RESULT, WHILE THE BLUE DENOTES THE SECOND HIGHEST RESULT.
<table><tr><td>Method</td><td> $\overline { { S _ { 1 } } }$ </td><td> $\overline { { S _ { 2 } } }$ </td><td> $\overline { { S _ { 3 } } }$ </td><td> $\overline { { S _ { 4 } } }$ </td><td> $\overline { { S _ { 5 } } }$ </td><td> $\overline { { S _ { 6 } } }$ </td><td> $\overline { { S _ { 7 } } }$ </td><td> $\overline { { S _ { 8 } } }$ </td><td>S9</td><td> $\overline { { S _ { 1 0 } } }$ </td><td> $\overline { { S _ { 1 1 } } }$ </td><td>Avg.</td></tr><tr><td>PODNet [47]</td><td>48.12</td><td>12.84</td><td>9.56</td><td>10.21</td><td> $\overline { { 6 . 9 1 } }$ </td><td> $\overline { { 5 . 7 2 } }$ </td><td>4.28</td><td>5.01</td><td>4.83</td><td>3.45</td><td>4.12</td><td>10.46</td></tr><tr><td>LUCIR [48]</td><td>48.12</td><td>45.16</td><td>42.87</td><td>38.90</td><td>26.85</td><td>21.43</td><td>25.33</td><td>27.99</td><td>22.48</td><td>20.05</td><td>19.11</td><td>30.75</td></tr><tr><td>Self-Training [50]</td><td>48.12</td><td>40.28</td><td>36.71</td><td>33.92</td><td>28.65</td><td>25.34</td><td>31.12</td><td>33.85</td><td>33.01</td><td>31.07</td><td>29.56</td><td>33.78</td></tr><tr><td>ExtendNER [8]</td><td>48.12</td><td>39.88</td><td>39.12</td><td>31.05</td><td>28.46</td><td>26.92</td><td>31.77</td><td>34.56</td><td>32.88</td><td>33.21</td><td>30.85</td><td>34.26</td></tr><tr><td>CFNER [12]</td><td>48.12</td><td>49.24</td><td>47.65</td><td>46.78</td><td>38.95</td><td>34.87</td><td>37.92</td><td>40.88</td><td>39.55</td><td>41.23</td><td>38.49</td><td>42.15</td></tr><tr><td>RDP [17]</td><td>48.12</td><td>50.21</td><td>47.83</td><td>45.56</td><td>41.27</td><td>38.92</td><td>37.45</td><td>40.89</td><td>39.12</td><td>39.75</td><td>39.04</td><td>42.56</td></tr><tr><td>TBCL (Ours)</td><td>48.12</td><td>50.41</td><td>47.27</td><td>47.05</td><td>46.12</td><td>43.11</td><td>43.95</td><td>43.23</td><td>42.11</td><td>41.38</td><td>40.71</td><td>44.86</td></tr></table>

3) Finer-grained Entity Types: To further validate the effectiveness of our proposed TBCL method, we conduct experiments on the FEW-NERD dataset [55], which contains 66 fine-grained entity types. We adopt an FG-6-PG-6 INER setting, where the first step learns 6 entity types and each subsequent step introduces 6 new entity types, for a total of 11 incremental steps.

We evaluate several strong baseline methods, including PODNet [47], LUCIR [48], Self-Training [50], ExtendNER [8], CFNER [12], and RDP [17], using their publicly available implementations under the same FG-6-PG-6 setting.

The results are summarized in Table VII, which reports the step-wise Macro-F1 scores as well as the overall average Macro-F1. As shown, our TBCL consistently outperforms all baselines across all incremental steps, demonstrating its strong capability in handling fine-grained entity types and limiteddata scenarios in INER. Notably, TBCL achieves the highest Macro-F1 at almost all incremental steps and maintains a clear advantage in average Macro-F1 over all compared methods.

## VI. CONCLUSION

In this paper, we address the biased context issue common in pseudo-labeling-based INER approach [17] by proposing a novel TBCL method. This method integrates a sentence-duplet learning scheme and a contextual consistency loss to rectify biased context correlations between tokens of old and new entity types, reducing the problems of forgetting old entity types and overfitting new ones. We perform extensive experiments on ten INER settings across three highly recognized datasets: CoNLL2003 [44], I2B2 [45], and OntoNotes5 [46]. The results highlight the superiority of the proposed TBCL method, which surpasses previous SOTA INER approaches.

## REFERENCES

[1] X. Li, F. Yin, Z. Sun, X. Li, A. Yuan, D. Chai, M. Zhou, and J. Li, “Entity-Relation Extraction as Multi-Turn Question Answering,” in Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, 2019, pp. 1340–1350.

[2] S. Longpre, K. Perisetla, A. Chen, N. Ramesh, C. DuBois, and S. Singh, “Entity-Based Knowledge Conflicts in Question Answering,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, 2021, pp. 7052–7063.

[3] B. Fetahu, A. Fang, O. Rokhlenko, and S. Malmasi, “Gazetteer Enhanced Named Entity Recognition for Code-Mixed Web Queries,” in SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, 2021, pp. 1677–1681.

[4] J. Guo, G. Xu, X. Cheng, and H. Li, “Named entity recognition in query,” in Proceedings of the 32nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2009, Boston, MA, USA, July 19-23, 2009, 2009, pp. 267–274.

[5] S. Mokhtari, A. Mahmoody, D. Yankov, and N. Xie, “Tagging Address Queries in Maps Search,” in The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, 2019, pp. 9547–9551.

[6] N. Zhang, Q. Jia, S. Deng, X. Chen, H. Ye, H. Chen, H. Tou, G. Huang, Z. Wang, N. Hua, and H. Chen, “AliCG: Fine-grained and Evolvable Conceptual Graph Construction for Semantic Search at Alibaba,” in KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021, 2021, pp. 3895–3905.

[7] G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and C. Dyer, “Neural architectures for named entity recognition,” in The North American Chapter of the Association for Computational Linguistics, 2016.

[8] N. Monaikul, G. Castellucci, S. Filice, and O. Rokhlenko, “Continual learning for named entity recognition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 15, 2021, pp. 13 570– 13 577.

[9] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural Networks, vol. 113, pp. 54–71, 2019.

[10] S. Thrun, “Lifelong learning algorithms,” in Learning to learn. Springer, 1998, pp. 181–209.

[11] Y. Xia, Q. Wang, Y. Lyu, Y. Zhu, W. Wu, S. Li, and D. Dai, “Learn and Review: Enhancing Continual Named Entity Recognition via Reviewing Synthetic Samples,” in Findings of the Association for Computational Linguistics: ACL 2022, 2022, pp. 2291–2300.

[12] J. Zheng, Z. Liang, H. Chen, and Q. Ma, “Distilling Causal Effect from Miscellaneous Other-Class for Continual Named Entity Recognition,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2022.

[13] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” Psychology of learning and motivation, vol. 24, pp. 109–165, 1989.

[14] A. Robins, “Catastrophic forgetting, rehearsal and pseudorehearsal,” Connection Science, vol. 7, no. 2, pp. 123–146, 1995.

[15] I. J. Goodfellow, M. Mirza, D. Xiao, A. Courville, and Y. Bengio, “An empirical investigation of catastrophic forgetting in gradient-based neural networks,” arXiv preprint arXiv:1312.6211, 2013.

[16] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[17] D. Zhang, H. Li, W. Cong, R. Xu, J. Dong, and X. Chen, “Task relation distillation and prototypical pseudo label for incremental named entity recognition,” in Proceedings of the 32nd ACM International Conference

on Information and Knowledge Management (CIKM), 2023, pp. 3319– 3329.

[18] G. Hinton, O. Vinyals, J. Dean et al., “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, vol. 2, no. 7, 2015.

[19] H. Zhao, F. Yang, X. Fu, and X. Li, “Rbc: Rectifying the biased context in continual semantic segmentation,” in European Conference on Computer Vision. Springer, 2022, pp. 55–72.

[20] Z. Chen and B. Liu, “Lifelong machine learning,” Synthesis Lectures on Artificial Intelligence and Machine Learning, vol. 12, no. 3, pp. 1–207, 2018.

[21] J. Dong, D. Zhang, Y. Cong, W. Cong, H. Ding, and D. Dai, “Federated Incremental Semantic Segmentation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023.

[22] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[23] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in International Conference on Machine Learning (ICML). PMLR, 2017, pp. 3987–3995.

[24] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory Aware Synapses: Learning what (not) to forget ,” in Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

[25] J. Schwarz, W. Czarnecki, J. Luketina, A. Grabska-Barwinska, Y. W. Teh, R. Pascanu, and R. Hadsell, “Progress & compress: A scalable framework for continual learning,” in International Conference on Machine Learning (ICML). PMLR, 2018, pp. 4528–4537.

[26] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Learning a Unified Classifier Incrementally via Rebalancing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[27] Z. Li and D. Hoiem, “Learning without Forgetting,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 40, no. 12, pp. 2935–2947, 2018.

[28] A. Mallya and S. Lazebnik, “PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[29] J. Yoon, E. Yang, J. Lee, and S. J. Hwang, “Lifelong learning with dynamically expandable networks,” in Proceedings of the International Conference on Learning Representations (ICLR), 2018.

[30] A. Rosenfeld and J. K. Tsotsos, “Incremental Learning Through Deep Adaptation,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 42, no. 3, pp. 651–663, 2020.

[31] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “iCaRL: Incremental Classifier and Representation Learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[32] D. Lopez-Paz and M. A. Ranzato, “Gradient Episodic Memory for Continual Learning,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper/ 2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf

[33] H. Shin, J. K. Lee, J. Kim, and J. Kim, “Continual Learning with Deep Generative Replay,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper/ 2017/file/0efbe98067c6c73dba1250d2beaa81f9-Paper.pdf

[34] X. Ma and E. Hovy, “End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2016, pp. 1064–1074.

[35] J. Li, A. Sun, J. Han, and C. Li, “A survey on deep learning for named entity recognition,” IEEE Transactions on Knowledge and Data Engineering, vol. 34, no. 1, pp. 50–70, 2020.

[36] G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and C. Dyer, “Neural Architectures for Named Entity Recognition,” in NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016, 2016, pp. 260–270. [Online]. Available: https://doi.org/10.18653/v1/n16-1030

[37] J. D. M.-W. C. Kenton and L. K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of NAACL-HLT, 2019, pp. 4171–4186.

[38] D. Zhang, Y. Yu, F. Chen, and X. Chen, “Decomposing Logits Distillation for Incremental Named Entity Recognition,” in Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023, pp. 1919–1923.

[39] R. Ma, X. Chen, Z. Lin, X. Zhou, J. Wang, T. Gui, Q. Zhang, X. Gao, and Y. W. Chen, “Learning “O” Helps for Learning More: Handling the Unlabeled Entity Problem for Class-incremental NER,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 5959–5979.

[40] Y. Chen and L. He, “SKD-NER: Continual Named Entity Recognition via Span-based Knowledge Distillation with Reinforcement Learning,” in Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 6689–6700.

[41] D. Zhang, W. Cong, J. Dong, Y. Yu, X. Chen, Y. Zhang, and Z. Fang, “Continual Named Entity Recognition without Catastrophic Forgetting,” in Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 8186–8197.

[42] D. Zhang, C. Li, J. Dong, Q. Liu, and D. Yu, “Exploring Stability-Plasticity Trade-offs for Continual Named Entity Recognition,” IEEE Transactions on Audio, Speech and Language Processing, 2025.

[43] Y. Yu, D. Zhang, X. Chen, and C. Chu, “Flexible weight tuning and weight fusion strategies for continual named entity recognition,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 1351–1358.

[44] E. F. T. K. Sang and F. De Meulder, “Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition,” Development, vol. 922, p. 1341, 1837.

[45] S. N. Murphy, G. Weber, M. Mendis, V. Gainer, H. C. Chueh, S. Churchill, and I. Kohane, “Serving the enterprise and beyond with informatics for integrating biology and the bedside (i2b2),” Journal of the American Medical Informatics Association, vol. 17, no. 2, pp. 124– 130, 2010.

[46] E. Hovy, M. Marcus, M. Palmer, L. Ramshaw, and R. Weischedel, “OntoNotes: the 90% solution,” in Proceedings of the human language technology conference ofthe NAACL, Companion Volume: Short Papers, 2006, pp. 57–60.

[47] A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, “PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning,” in Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XX, 2020, pp. 86–102.

[48] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Learning a Unified Classifier Incrementally via Rebalancing,” in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, 2019, pp. 831–839.

[49] M. D. Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. G. Slabaugh, and T. Tuytelaars, “Continual learning: A comparative study on how to defy forgetting in classification tasks,” CoRR, vol. abs/1909.08383, 2019.

[50] C. Rosenberg, M. Hebert, and H. Schneiderman, “Semi-Supervised Self-Training of Object Detection Models,” in 7th IEEE Workshop on Applications of Computer Vision / IEEE Workshop on Motion and Video Computing (WACV/MOTION 2005), 5-7 January 2005, Breckenridge, CO, USA, 2005, pp. 29–36.

[51] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[52] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz et al., “Huggingface’s transformers: State-of-the-art natural language processing,” arXiv preprint arXiv:1910.03771, 2019.

[53] A. Douillard, Y. Chen, A. Dapogny, and M. Cord, “PLOP: Learning Without Forgetting for Continual Semantic Segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, 2021, pp. 4040–4050.

[54] D. Kim, J. Bae, Y. Jo, and J. Choi, “Incremental learning with maximum entropy regularization: Rethinking forgetting and intransigence,” arXiv preprint arXiv:1902.00829, 2019.

[55] N. Ding, G. Xu, Y. Chen, X. Wang, X. Han, P. Xie, H.-T. Zheng, and Z. Liu, “Few-nerd: A few-shot named entity recognition dataset,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021, pp. 3198–3213.

[56] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” arXiv preprint arXiv:1907.11692, 2019.

![](images/b749b60f508da982d7ea297384f2d80ce4ed2487b04cbc0f4647bf2b78790b61.jpg)

Duzhen Zhang received his B.Sc. degree from Shandong University in June 2019. He completed his Ph.D. at the Institute of Automation, Chinese Academy of Sciences in June 2024. Since September 2024, he has been a postdoctoral researcher at the Mohamed bin Zayed University of Artificial Intelligence. His current research interests include large language models, continual learning, multi-modal learning, and AI for science.

![](images/c4b598d2c90db52b6a96e5797df43087dcf3d218a56a20c4b8abacef6bbeacf2.jpg)

Yahan Yu received her B.S. degree from Nanjing University of Aeronautics and Astronautics in 2020 and her M.S. degree in Control Science from University of Chinese Academy of Sciences in 2023. She is currently a Ph.D. student at Kyoto University. Her research interests include natural language processing and continual learning.

![](images/b1086316cc3c27b85fe6289be56eba27f5bf426f915e81ed178f0120ecdbd7e6.jpg)

Xiuyi Chen received his Ph.D. degree (2022) in Pattern Recognition and Intelligent System from Institute of Automation, Chinese Academy of Sciences, advised by Prof. Bo Xu. Previously, he received the B.Sc. degree (2017) from JiLin University. His current interests include Cross-modal Retrieval, Multimodal Learning, Dialogue System, and Knowledge-Grounded Generation.

![](images/66113470998ca7cd1aa73790e48b33c0abf66a90c93e1452915ff058759712f1.jpg)

Chenxing Li received his B.Sc. degree at the North China Electric Power University, China, in 2015. He completed his Ph.D. at the Institute of Automation, Chinese Academy of Sciences in 2020. His current research interests include far-field speech recognition, speech enhancement, and speech separation.

![](images/fc83ac5bfd764200cd855e1e273f1ad901081daa3873700aaaece4612b9e3828.jpg)

Dong Yu (Fellow, IEEE) with the Tencent AI Lab as a distinguished scientist and vice general Manager. Prior to joining Tencent in 2017, he was a Principal Researcher with Microsoft Research (Redmond), where he has been since 1998. He has authored or coauthored two monographs and more than 300 papers. His research interests include speech recognition and processing and natural language processing. His works have been widely cited and recognized by the prestigious IEEE Signal Processing Society best transaction paper award in 2013, 2016, 2020, and

2022, the 2021 NAACL best long paper award, 2022 IEEE Signal Processing Magazine best paper award, and 2022 IEEE Signal Processing Magazine best column award. Dr. Dong Yu was the Chair of the IEEE Speech and Language Processing Technical Committee during 2021–2022. He was on the editorial boards of numerous journals and magazines, as well as on the organizing and technical committees of various conferences and workshops. He is currently an ACM/IEEE/ISCA Fellow.