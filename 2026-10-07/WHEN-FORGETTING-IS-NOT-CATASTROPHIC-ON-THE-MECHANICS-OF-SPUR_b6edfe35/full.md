# WHEN FORGETTING IS NOT CATASTROPHIC: ON THE MECHANICS OF SPURIOUS FORGETTING

Vedant Palit<sup>1,2,3</sup> Florent Draye<sup>1,4</sup> Nicolas Zucchet<sup>5</sup> Zhijing Jin<sup>1,2,3</sup> Bernhard Scholkopf ¨ <sup>1,6</sup>

<sup>1</sup>MPI for Intelligent Systems, Tubingen¨ <sup>2</sup>Jinesis Lab, University of Toronto & Vector Institute

<sup>3</sup>EuroSafeAI <sup>4</sup>Hector Foundation <sup>5</sup>Stanford University <sup>6</sup>ELLIS Institute Tubingen ¨

§ vedant<sub>p</sub>alit/s<sub>p</sub>urious-for<sub>g</sub>ettin<sub>g</sub>-mechanics

## ABSTRACT

Knowledge that a language model appears to forget during finetuning often remains stored and can be recovered, a phenomenon called spurious forgetting. Finetuning on new facts can even produce forgetting that undoes itself: recall of the old facts collapses, recovers as training continues on new facts alone, and only then erodes for good. We seek to understand when such forgetting is not catastrophic. A minimal associative memory reproduces these dynamics with three ingredients: keys with shared structure, concentrated new values, and normalization in the network. Finetuning moves all old representations along a common direction, hiding the old facts while preserving their relative geometry; normalization withdraws this shift once the new facts are learned, whereas fact-specific changes accumulate and cause the erosion. Moreover, subtracting the common shift eliminates the collapse in a Transformer trained on synthetic data, and removing a single direction from each weight update restores old facts in a pretrained language model. Forgetting thus combines a shared, reversible loss of access with a slow erosion of individual facts, and only the second is catastrophic. Which one dominates depends on whether the new data move old memories together or apart.

## 1 INTRODUCTION

Continually training a neural network on new data can severely damage the knowledge and skills it previously acquired, a phenomenon known as catastrophic forgetting (McCloskey & Cohen, 1989; Ratcliff, 1990; French, 1999). It has been reported across many settings, from image classification (Goodfellow et al., 2014; Zenke et al., 2017; Ramasesh et al., 2021) to deep reinforcement learning (Kirkpatrick et al., 2017; Rolnick et al., 2019), and large language models are no exception (Luo et al., 2025; Zucchet et al., 2025; Marek et al., 2026). Yet, a loss of performance need not be a loss of knowledge: a different prompt can restore lost capabilities (Kotha et al., 2024), and brief training on a few old examples can undo forgetting that reflects a loss of task alignment rather than of knowledge, named spurious forgetting (Zheng et al., 2025). Accuracy measured at a single point in training therefore cannot tell whether knowledge was erased or only hidden. What hides it, and whether training on the new data alone can reveal it again, remains unclear; we ask

## When is forgetting not catastrophic?

We study this question for factual knowledge, as forgetting and loss of access can be cleanly disentangled. How language models store and retrieve facts is well understood (Geva et al., 2021; Meng et al., 2022; Geva et al., 2023; Nanda et al., 2023), and synthetic biographies (Allen-Zhu & Li, 2024; Zucchet et al., 2025) let us specify exactly which facts the model knows, guarantee that the new facts were never seen before, and control how the two relate. We can then ask not only whether the model still outputs an old fact, but also whether the fact is still stored. In this setting, finetuning on new facts alone produces spurious forgetting in a particularly clear form (Figure 1): recall of the old facts collapses, then recovers as training continues on the same new facts, before a slower decline sets in. Since there is no replay of the old facts, the recovery can only come from knowledge that survived the collapse.

![](images/28bf881b24cc23b50d0b654eaf3327e2b1af1990d639d269a69492aa9c44e380.jpg)  
Figure 1: Changing only the finetuning data turns catastrophic forgetting into spurious forgetting. A transformer is pretrained on two non-overlapping sets of facts, A and B, and then finetuned on new facts only, without any replay. When the new facts overlap with both pretraining sets (left, finetuning on C), performance on both declines slowly and steadily. When they overlap only with B (right, fine-tuning on $B ^ { \prime } )$ , performance on B is mostly intact, but performance on A collapses, recovers although A is never seen again, and later erodes. How the different sets are built and what it means for the new facts to overlap with the old ones is detailed in Section 4.

Our contributions are as follows:

– A clean case of spurious forgetting. In a controlled factual-recall setting, finetuning on new facts alone makes recall of the old facts collapse and then return, without any replay, and changing only where the new answers lie turns the same finetuning into steady forgetting.

– A minimal mechanism. A minimal associative memory reproduces the phenomenon once three ingredients are present: keys with shared structure, new answers concentrated in one region, and normalization. Finetuning then moves all old representations together, which hides the old facts without erasing them, and normalization undoes this shift once the new facts are learned. Changes specific to each fact accumulate and cause a slower, lasting forgetting.

– Causal evidence in trained models. In Transformers trained from scratch on synthetic data, removing the common shift eliminates the collapse, whose end is set by the learning of the new facts, not by the number of training steps. In a pretrained language model, recall of the old facts recovers only when the subjects of the new facts reveal nothing about their answers, and removing one direction of each weight update restores it beyond the trained model’s recall at equal new-fact accuracy.

Together, our results show that forgetting is not catastrophic when the update moves old memories together rather than apart: the knowledge is then hidden rather than erased, and can be recovered, sometimes by just continuing finetuning for longer.

## 2 FORGETTING THAT UNDOES ITSELF

To understand when forgetting is not catastrophic, we first need a setting in which it clearly is not: one in which performance collapses, yet the knowledge demonstrably survives (Appendix C.1). We use synthetic biographies (Allen-Zhu & Li, 2024), on which spurious forgetting was identified (Zheng et al., 2025), as they let us control what the model knows before finetuning and what it learns during it. A transformer is pretrained on two non-overlapping sets of biographies, A and B, in a single stage rather than one per set (Zheng et al., 2025), until it recalls all of their facts, and is then finetuned on biographies of new individuals, without any replay. We compare two sets of new biographies, C and B<sup>′</sup>, which differ only in how they overlap with the old ones. We describe how they are built, and what overlapping means precisely, in Section 4; for now, the only relevant property is that C overlaps with both A and B, while B<sup>′</sup> overlaps only with B (Figure 1, middle).

Finetuning on C yields classical catastrophic forgetting behavior (Figure 1, left): as the new facts are learned, recall of both A and B declines slowly and steadily. Finetuning on $B ^ { \prime }$ barely affects recall of B, but recall of A follows a strikingly different trajectory (Figure 1, right), in three phases that we refer to throughout the paper:

– Collapse: recall of A quickly drops, well before the new facts are learned.

– Recovery: recall of A then partially returns as the new facts are acquired, although A never reappears in training.

– Erosion: finally, recall of A declines slowly and persistently, as in catastrophic forgetting.

This trajectory raises two questions: why finetuning on $B ^ { \prime }$ hides the facts of A while sparing those of B, and why the collapse reverses on its own. Section 3 answers both in a minimal model. Section 4 then returns to Transformers, and to the biography structure needed for the phenomenon to appear.

## 3 A MINIMAL MODEL REVEALS THE MECHANISM OF SPURIOUS FORGETTING

Factual recall, including that of synthetic biographies, can be abstracted as an associative memory: each fact maps a key, the subject it is about, to a value, its answer, which the model must retrieve when queried. We show that a minimal associative memory reproduces spurious forgetting once three ingredients are present: keys that share a common structure, new values concentrated in one part of the vocabulary, and a normalization layer. Its learning dynamics then explain the phenomenon: the structure of the data causes the collapse, and normalization drives the recovery. We check each step of this explanation in the model in Appendix B.8.

## 3.1 REPRODUCING SPURIOUS FORGETTING IN AN ASSOCIATIVE MEMORY MODEL

Many phenomena of factual recall, such as how many facts a model can store as a function of its size (Allen-Zhu & Li, 2025; Nichani et al., 2025), how its loss scales (Cabannes et al., 2024) or the dynamics with which facts are learned (Zucchet et al., 2025; 2026), can be reproduced in a simple feedforward associative memory, which strips away the sequence-modeling layers of the transformer.

Associative memory task. Each fact i is represented by a key

$$
k _ { i } = \sqrt { \alpha } \mu + \sqrt { 1 - \alpha } e _ { i } ,\tag{1}
$$

where $\mu$ is a unit vector shared by all facts, $e _ { i }$ a random unit vector specific to the fact, and α sets how much structure the keys share. Each key is associated with a value $y _ { i } ,$ a token from a vocabulary of size V that we split into two halves. The old facts form two sets, A, whose values are drawn uniformly from one half, and $B ,$ whose values are drawn from the other. The new facts $B ^ { \prime }$ have new keys, sampled in the same way, and draw their values from the same half as $B ,$ , just as the new biographies overlap with only one pretraining set in Figure 1 (right); drawing them from the entire vocabulary instead mirrors finetuning on $C$ (left). The data thus combine keys that share structure with new values concentrated in one half of the vocabulary.

Model. The model is a two-layer network with an RMS normalization layer (Zhang & Sennrich, 2019) as the non-linearity between the two layers, followed by a softmax readout. It can be understood as the final linear layers of a Transformer. When queried with a key $k _ { i }$ , the model outputs a probability distribution over values,

$$
p _ { i } = \mathrm { s o f t m a x } ( W _ { 2 } \mathrm { R M S } ( h _ { i } ) ) , \quad \mathrm { w i t h } h _ { i } = W _ { 1 } k _ { i } .\tag{2}
$$

The RMS normalization divides the hidden state by its norm and multiplies it by ${ \sqrt { d } } .$ . The weight matrices $W _ { 1 }$ and $W _ { 2 }$ are trained with minibatch gradient descent on the cross-entropy loss, first on the pretraining data (A and B) and then on the finetuning data (B<sup>′</sup>).

Each of these ingredients is necessary. We follow the old facts of A over finetuning. The minimal model reproduces their collapse, recovery and erosion (Figure 2), and removing any one of its ingredients breaks this trajectory. With less shared structure in the keys, the collapse slows down until only gradual forgetting remains (panel 1). Drawing the values of the new facts from the entire vocabulary, as in finetuning on C, likewise leaves only gradual forgetting (panel 2). Without normalization, or with a ReLU in its place, the old facts collapse and never recover (panel 3).

## 3.2 KEY SHARING LEADS TO COLLAPSE

Intuitively, a gradient step on a new fact moves not only its own hidden state but that of every key resembling its own. All keys share a common component, so whatever the updates of the new facts have in common is written into all hidden states at once, the more so as α grows. The result is a common shift that leaves the relative geometry of the hidden states mostly intact but, through the readout, favors the half of the vocabulary that contains the new answers, so that the facts of A can no longer be decoded.

![](images/29f8854097a3dc63c7d4598ade2c1ba45ad42d5cf5081c25bf92aa74de0c7436.jpg)

![](images/e8f95694a1a864289a5b1b6c0b9ee5df0f58a663f1f30eccf0b880b995a01004.jpg)

![](images/f42f84a9081944e5b3da67437df79fcbbdf74d93ded4fd1ac1ee028e6393f9da.jpg)

![](images/b547ea66abb2d820f19248c107051b5ee4804f713e964beeb617baa383f40263.jpg)  
Figure 2: Shared keys, concentrated new values and normalization are all needed for the collapse, recovery and erosion. The model maps keys, which share a component of strength $\alpha ,$ to values split into two regions, through one hidden layer with RMS normalization. As in Figure 1, it is pretrained on A and B, whose values lie in different regions, then finetuned only on new facts $B ^ { \prime }$ whose values lie in the region of $B ;$ curves show recall of A. Reducing α weakens the collapse and removes the recovery (1), drawing the new values from both regions leaves only gradual forgetting (2), and replacing the normalization removes the recovery (3). The model is described in Section 3.

All hidden states move in the same direction. When the model learns a new fact, gradient descent updates the first layer so that the hidden state of every key similar to the new fact’s key moves against that fact’s error. Since all keys, old and new, share a common component, they are all pushed in the same direction whenever the errors of the new facts agree. Formally, under gradient flow on the finetuning loss,

$$
\dot { h } _ { a } = \dot { W } _ { 1 } k _ { a } = - \frac { 1 } { n _ { B } } \sum _ { b } \left( k _ { b } ^ { \top } k _ { a } \right) \delta _ { b } , \quad \mathrm { w h e r e } ~ \delta _ { b } = \nabla _ { h _ { b } } \ell _ { b } .\tag{3}
$$

In high dimension, the fact-specific parts of different keys are nearly orthogonal, so the overlap $k _ { b } ^ { \top } k _ { a }$ is close to its shared part α and, to leading order,

$$
\dot { h } _ { a } = - \alpha \bar { \delta } _ { B } , \quad \mathrm { w h e r e } ~ \bar { \delta } _ { B } = \frac { 1 } { n _ { B } } \sum _ { b } \delta _ { b } .\tag{4}
$$

The terms this approximation ignores vary in sign from one new fact to the next and largely cancel, so they are negligible compared with the one it keeps as long as the errors $\delta _ { b }$ share a common component (Appendix B.4). All the old hidden states then move together in the same direction (Figure 3, left), at a speed proportional to α: this is why the collapse slows down as α decreases and disappears at α = 0 (Figure 2, panel 1).

Whether the errors share a common component depends on the finetuning data: pretraining covers both halves of the vocabulary, so the model’s predictions for the new keys are initially spread over both, whereas all the new answers lie in the same half, and on average the errors therefore all push probability toward this half. When the new answers are instead drawn from the entire vocabulary, this common push largely cancels, and so does the collapse (Figure 2, panel 2).

The common shift hides the facts of A. The collapse of A is a direct consequence of this common shift. Ignoring normalization for a moment, shifting all hidden states by the same vector adds the same bias to the logits of every old fact, favoring the half of the vocabulary that contains the new answers. The facts of A, whose answers lie in the other half, therefore lose to the new answers. Normalization amplifies this effect. Once the shift accounts for most of the norm of the hidden states, normalizing rotates all of them toward the shift direction: the old states become more similar to one another, the part of each state that encodes the fact’s own answer is scaled down, and the predictions follow the shift toward the answers most common among the new facts. The facts of B, whose answers lie in the favored half, are largely spared by the shift itself, which raises the answers of their half roughly together, in proportion to how often each is a new answer: removing the shift from their hidden states barely changes their recall, whereas it restores that of A. As their answers are also those of the new facts, they are instead exposed to the fact-specific learning of the new facts, which erodes them steadily, mostly after the collapse (Section 3.3).

![](images/e323ebda926a1179d5a1ef89340b144b9f0ea0353b7559aa40665d0de60aa341.jpg)

![](images/c0d1f849919e44424eb48b8b9c885b52a13cda3412fc42831ccd1220394f00ed.jpg)

![](images/4ba2a09914ccb06ae31b5167286db667473391cde7e5b0f557f73987fe2a9430.jpg)  
Figure 3: Normalization reverses the shift responsible for forgetting. The old representations first move together along a shared direction. Once the new facts are learned and the individual margin becomes positive, the common shift retracts and the old facts are recalled again. Without a normalization layer, this reversal does not occur, which is why the collapse is permanent (Figure 2, panel 3). (left) Old hidden states before normalization, projected on the shared direction and an orthogonal one, after pretraining, at the end of the collapse and during erosion. (middle) Common shift magnitude s of the old hidden states, with and without normalization. (right) Shared and individual margins of the new facts, which determine the dynamics of s through Equation 5. Same setup as in Figure 2; the dotted line marks the end of the collapse.

## 3.3 NORMALIZATION DRIVES RECOVERY

Early in finetuning, all hidden states are pushed in the same direction, and this common shift causes the collapse of A. Because the normalization fixes the norm of each hidden state, the growing shift comes at the expense of the individual parts, which encode each fact’s own information and eventually become negligible. At first, this also helps the new facts, whose individual parts point to wrong answers. However, once the model has learned their individual associations, the shift drowns these as well and prevents the model from predicting the new facts correctly. Finetuning then undoes the shift, both on the new facts and on the old facts, restoring the individual parts of every hidden state. For the facts of A, these still carry most of the information that distinguishes them from one another, and the readout weights of their answers are barely changed by finetuning, as they only receive gradient while the model still puts probability on these answers for the new facts. The facts of A can therefore be read out again, and their recall partially recovers.

Normalization reverses the shift. We track the common shift through the norm s of the displacement of the old facts’ mean hidden state $\bar { h } _ { A }$ since the start of finetuning, and call its direction $\hat { u } _ { \mathrm { s h } }$ the shared direction. Through Equation 4, the dynamics of s are governed by whether pushing the hidden states of the new facts along the shared direction improves their predictions or not. Because of the normalization, such a push has two competing effects. On the one hand, it adds shared component, which improves the predictions on the new facts in proportion to the shared margin γ¯ : the extra logit that the shared direction gives the correct answer compared with the model’s current prediction, averaged over the new facts. On the other hand, it scales their individual parts down, which changes their predictions in proportion to the individual margin $\bar { \gamma } _ { \mathrm { i n d } }$ , the same quantity measured along their individual parts. Assuming for simplicity that all new hidden states have the same coordinate s along $\hat { u } _ { \mathrm { s h } }$ and the same individual norm σ, we obtain, to leading order,

$$
\dot { s } = \alpha \sqrt { d } { \frac { \sigma \left( \sigma \bar { \gamma } _ { \mathrm { s h } } - s \bar { \gamma } _ { \mathrm { i n d } } \right) } { \left( s ^ { 2 } + \sigma ^ { 2 } \right) ^ { 3 / 2 } } } , \quad \mathrm { w i t h } s ( 0 ) = 0 .\tag{5}
$$

The first term corresponds to the added shared component and the second to the smaller individual parts; their relative size determines whether the common shift grows or shrinks. The derivation of this equation and the precise definitions of the two margins are given in Appendix B.5.

This equation explains the dynamics of the common shift in Figure 3. The shared margin only matters in the first steps of finetuning. The shared direction can move probability between the two halves of the vocabulary, but not between the answers within a half: $\bar { \gamma } _ { \mathrm { s h } }$ is positive at first, and once the predictions for the new facts have moved to the half of their answers, it can no longer improve them and remains close to zero (Figure 3, right). The shared margin thus sets the direction of the shift, and the individual margin then sets its magnitude. The new facts are initially confidently wrong, $\bar { \gamma } _ { \mathrm { i n d } } ~ < ~ 0$ , so scaling their individual parts down helps them: s grows, and the growth reinforces itself. Once the individual parts of the new facts point to their correct answers, $\bar { \gamma } _ { \mathrm { i n d } }$ becomes positive (Figure 3 right) and the same term pulls s back (Figure 3 middle). The shift therefore peaks when $\bar { \gamma } _ { \mathrm { i n d } }$ crosses zero. This reversal waits for fact-specific learning, which is slow: the error of each new fact enters its own state with weight $1 / n _ { B }$ , whereas their common component enters every state with weight α (Appendix B.4), in line with the general observation that shared structure is learned before item-specific structure (Saxe et al., 2019). Without normalization, adding a shared component no longer scales the individual parts down, the $\bar { \gamma } _ { \mathrm { i n d } }$ term disappears in Equation 5, and $\dot { s } = \alpha \bar { \gamma } _ { \mathrm { s h } }$ . Nothing pulls the shift back (Figure 3 middle) and the collapse is permanent (Figure 2).

The facts of A are hidden, not erased. Undoing the common shift only restores the facts of A if what distinguishes them from one another has survived, and it does, for two reasons. First, beyond the common shift, each old hidden state only receives the part of the update carried by the fact-specific parts of the new keys, which overlap weakly with its own key (Appendix B.3). This individual displacement stays small during the collapse: while the mean of the old hidden states moves a lot, their relative geometry barely changes (Figure 3, left). The facts of A therefore keep their ordering within their own half, and even at the bottom of the collapse most of them still rank their correct answer first among the answers of that half (Appendix B.8). Second, the readout weights of the answers of A receive little gradient: they are only updated while the model puts probability on these answers for the new facts, which stops early in finetuning. Removing the part of the update that acts on the shared direction, $\Delta W _ { 1 } ( t ) \bar { \mu } \mu ^ { \top }$ , while keeping everything else, including the finetuned readout, restores their recall almost entirely (Figure 13). The collapse is thus a loss of access to the facts of A, not a loss of the facts themselves.

The individual displacements nonetheless keep accumulating after the collapse, and they are not neutral: each old hidden state drifts toward the answers of the new facts whose keys overlap most with its own. For A, restoring the mean hidden state therefore helps less and less as finetuning proceeds, and recovery remains incomplete. For B, whose answers are those of the new facts, this drift, together with the readout weights of their half, which are refit to the new facts, erodes recall throughout finetuning, and removing the common shift barely helps.

## 4 THE MECHANISM IN A CONTROLLED TRANSFORMER

Having identified the mechanism behind spurious forgetting in a minimal model, we now return to the Transformer of Figure 1 and test whether its collapse and recovery follow the mechanism of the minimal model, first by showing that its data meet the same conditions, then through measurements and interventions on its representations. Table 1 lists where each analysis is repeated across models.

## 4.1 TRAINING TRANSFORMERS ON SYNTHETIC BIOGRAPHIES MEETS THE CONDITIONS

The minimal model identifies three conditions for spurious forgetting: a normalization layer, keys that share structure, and new answers concentrated in one region of the output space. The first condition is met by any Transformer, as every residual block contains a normalization layer (Vaswani et al., 2017). In the pre-normalization architecture we use (Radford et al., 2019; Xiong et al., 2020), the last one sits just before the readout. The other two are properties of the data, which we now describe. The synthetic biographies of Section 2 (Allen-Zhu & Li, 2024; Zucchet et al., 2025) describe a large population of fictitious individuals, each with a unique name and a few attributes, such as a hometown or an employer. Each fact is stated by filling templates shared across all individuals, such as The hometown of [name] is [city] (Figure 4, left), and each set of biographies, A, B, $B ^ { \prime }$ or C, contains its own individuals, so that the new facts are about individuals the model has never seen. If we view the feedforward layers of a Transformer as an associative memory in which part of the context serves as the key and the next token as the value (Geva et al., 2021), the key of a fact is the name of the individual together with the attribute being queried, and its value is the answer. As all facts about a given attribute are queried in a similar way, their keys share structure by design, just as the keys of the minimal model share the component $\mu .$ The last condition is not present by default, so we build it into the data: we split the possible answers of each attribute at random into two subsets. The individuals of A draw their answers from the first subset and those of B from the second. The finetuning sets therefore overlap with the old sets through their answers alone: those of $B ^ { \prime }$ draw their answers from the second subset, like $B ,$ , and those of C from both subsets. Details of the data and the model are given in Appendix A.1.

![](images/9d294d7b74e93373bc38e3c4db4b4a7f2fc1943247488f02915a5511efcd9b23.jpg)

![](images/d23675c9c332cfde5ae449582139e93a5b5d13179bb2af4a846f4f9a75da9395.jpg)

![](images/85d7ef295ca4dc1de69f7eddc36f7d7a809abdf94d6ce99c67a40e7402f29163.jpg)  
Figure 4: The Transformer initially forgets through the same common shift as the minimal model. (left) The Transformer is trained on synthetic biographies of fictitious individuals, each stating several facts about them, such as where they come from. The possible answers of each attribute are split into two subsets: the individuals of A draw their answers from the first, and those of B from the second. The model is pretrained on A and B, then finetuned on new individuals $B ^ { \prime }$ whose answers come from the same subset as $B ;$ this is the setup of Figure 1 (right). (middle) The common shift $\Delta \bar { h }$ of the hidden states of A grows fast and retracts at the end of the collapse, as in the minimal model (Figure 3). The fact-specific displacements $\varepsilon _ { a }$ grow more slowly and eventually become larger. (right) Accuracy on $A ,$ , and when only the common part of the change of its logits is kept or removed. The dotted line indicates the end of the collapse.

## 4.2 THE TRANSFORMER SHOWS THE SAME GEOMETRIC SPLIT

In the minimal model, the hidden states of the old facts first move together: their mean moves away from its pretrained value until the end of the collapse and then comes back, while the fact-specific displacements keep growing. To test whether the same happens in the Transformer, we track the representations of the old facts of A at the input of the readout, at the position where the model predicts the first token of the answer, for example Boston in Figure 4 (left). As in the minimal model, we decompose the displacement of each representation since the start of finetuning into a common shift $\Delta \bar { h } ( t )$ , its average over the facts of A about the same attribute, and a fact-specific remainder $\varepsilon _ { a } ( t )$ , whose size we measure by its average norm over the facts of A.

The two components evolve as the minimal model predicts (Figure 4, middle). The common shift grows quickly, peaks around the bottom of the collapse, and then partly retracts during the recovery, as in Figure 3 (middle). The fact-specific drift grows more slowly and, apart from a brief pause around the bottom of the collapse, keeps growing until it exceeds the common shift.

## 4.3 SUBTRACTING THE COMMON SHIFT REMOVES THE COLLAPSE

The minimal model predicts that the common shift alone causes the collapse: undoing it should restore the old facts, as long as what distinguishes them from one another has survived. Since the readout is linear and shared by all facts, a common shift of the readout states adds the same change to the logits of all old facts. We therefore test this prediction on the logits, at the same position as above: we write the change of the logits of each old fact a since the start of finetuning as $z _ { a } ( t ) - z _ { a } ( 0 ) = c ( t ) + v _ { a } ( t )$ , where c(t) is its average over the facts of $A$ about the same attribute, and measure recall from $z _ { a } ( 0 ) + c ( t )$ , with only the common part, and from $z _ { a } ( t ) - c ( t )$ , with the common part removed.

As predicted, subtracting the common part removes the collapse entirely (Figure 4, right): the old facts remain almost perfectly recalled, and only the slow decline caused by the fact-specific drift is left. Keeping only the common part does the opposite and produces a collapse and a recovery by itself. This collapse is shallower than in the trained model, because the common shift brings the old facts close to losing to the new answer region and the fact-specific drift pushes many of them across. The collapse is therefore caused by the common shift, and erosion by the fact-specific drift.

The shift is carried mostly by the attention value and output weights, with little contribution from the query and key weights (Appendix C.4). In our attention-only architecture, these are the weights that write content into the residual stream, whereas the query and key weights only decide where attention goes, so this is consistent with the common shift being written into the associative memories of the network, as in the minimal model. With MLP blocks, the same three phases appear, and subtracting the common shift again removes the collapse (Appendix C.5).

## 4.4 THE LEARNING RATE CHANGES THE DEPTH OF THE COLLAPSE

In the minimal model, the common shift reverses once the individual margin of the new facts turns positive (Section 3.3), so the end of the collapse should be tied to the learning of the new facts rather than to a number of steps. Varying the learning rate and the batch size of finetuning confirms this (Appendix C.2): the bottom of the collapse moves by a factor of three in steps, but falls at nearly the same accuracy on the new facts. This accuracy depends on how many new facts there are, however: with fewer, the bottom comes later in their learning, in the Transformer as in the minimal model, where the shift still peaks when the individual margin turns positive (Appendix C.3).

The depth of the collapse, in contrast, increases with the learning rate. This effect lies outside our gradient-flow analysis, in which the learning rate only rescales time. It is consistent with the observation that larger learning rates cause more forgetting, in continual pretraining (Gupta et al., 2023; Ibrahim et al., 2024) as in finetuning at matched loss (Springer et al., 2025; Lin et al., 2026; Rofin et al., 2026), and shows that part of this effect acts on forgetting that is spurious, and reverses once the new facts are learned.

## 5 THE COLLAPSE IN A PRETRAINED LANGUAGE MODEL

The controlled transformer meets the conditions of the minimal model by construction while a pretrained language model does not. We can still choose new facts whose answers the old facts never use, but the structure its keys share was set by pretraining on real text, so nothing guarantees that finetuning writes a common shift, or that it later withdraws it. The minimal model makes a prediction for each. The shift should be withdrawn only once the new facts have nothing left in common that helps predict their answers, so that each must be learned on its own; we test this with two kinds of new facts, one that meets this condition and one that does not. And since the shift of the hidden states is written by a single direction of the weight update, the one along which all new facts write together (Section 3.3), removing the top direction of each weight update should bring the old facts back.

## 5.1 THE COLLAPSE PERSISTS, AND REVERSES WHEN THE NEW FACTS ARE ARBITRARY

OLMo 2 1B (Team OLMo et al., 2025) knows far more facts than we can track, so we take as old facts the CounterFact facts (Meng et al., 2022) it answers correctly before finetuning; they play the role of A (Appendix A.2). We then finetune it without replay on documents stating new facts about people or products, in two separate runs with two kinds of new facts. In both, no new answer shares its first token with the answer of a tracked old fact, so the new answers occupy a region of the output space those old facts do not use, as the answers of $B ^ { \prime }$ do for A. The two kinds differ in what their subjects say about their answers (Section 3.3). Synthetic individuals, invented names that receive real companies, cities, countries and languages at random, say nothing: as in the minimal model, the shared direction helps them reach the region of their answers, but not choose within it. Real entities, products from EntityQuestions (Sciavolino et al., 2021) whose manufacturer the model does not know, say something: the kind of product hints at the kind of manufacturer, as a car model hints at a carmaker, so the shared direction keeps helping within the region.

With both kinds of new facts, recall of the old facts first rises slightly, although none of them is replayed: finetuning on documents that state facts makes the model favor answers over other continuations, which fixes old facts it already ranked correctly among the possible answers (Appendix D.1).

![](images/8d08dd4eab23c7dab1193854475cf10406780817b7ee2ce81120424392e45ed3.jpg)

![](images/250c93c0dddc73445579f01b59606e3a5452787695fd7cf70c2c148e6877a532.jpg)

![](images/0fe8359b13eef0d11681a1790f5cd26cc28524758ee791ae4ed81a87c424cfce.jpg)  
Figure 5: The collapse persists in a pretrained language model, but training reverses it only for arbitrary new facts. OLMo 2 1B is finetuned without replay, in separate runs, on synthetic individuals or on real entities. Old facts (CounterFact facts the model knows) are solid, new facts dotted; three seeds. (left) Recall of the old facts drops for both kinds of new facts, but recovers only for synthetic individuals. (middle) Removing the top singular direction of each weight update (light) restores recall of the old facts for both kinds, whereas removing a random direction of the same norm leaves it where training did (dark). (right) With a larger set of synthetic individuals, the collapse goes deeper and lasts longer.

With synthetic individuals, the trajectory of the controlled settings then reappears (Figure 5, left): recall of the old facts collapses before almost any new fact is recalled, recovers as the new facts are learned, and then erodes. More new facts delay the recovery, as each drives its own association with weight $1 / n _ { B }$ (Section 3.3): with a four times larger set of individuals, the collapse lasts longer and goes deeper (Figure 5, right). With real entities, the old facts never recover and keep declining even after every new fact is recalled: the shared direction keeps helping the new facts choose their answers within the region, so the shift is never withdrawn.

## 5.2 REMOVING ONE DIRECTION OF THE UPDATE RESTORES THE OLD FACTS

In the controlled Transformer, we removed the common shift from the hidden states and logits of the old facts, which were all queried in the same way. In a pretrained model, the old facts are queried through varied prompts, at varied positions, so there is no single set of states over which to average, whereas the weights write the shift whatever the prompt. We therefore act on the weights. In the minimal model, the part of the update that all new facts share is rank-one, written along the shared key direction (Equation 3), and it initially dominates the update, while the errors of the new facts are aligned. In a deep network it need not be rank-one at any single layer, but it should still initially dominate each weight update, and thus concentrate in its top singular direction. In the minima model and in both controlled Transformers, removing this direction indeed restores the old facts at the bottom of the collapse, and what it removes is aligned with the common shift (Appendices B.10 and D.3). We therefore subtract the top singular component of the update of every weight matrix in the Transformer blocks of the language model, or, as a control, a random rank-one matrix of the same norm, and evaluate the edited model (Appendix D.2).

The edit largely removes the collapse (Figure 5, middle), whereas the random control leaves recall exactly where training left it (Appendix D.2). With synthetic individuals, only the slow erosion remains. With real entities, the edit restores most of the old facts that training never brought back: their steady decline was mostly a shift that was never withdrawn, not erasure. The removed direction also lowers recall of the new facts, but the edit does not rewind finetuning: once the new facts are partly learned, the edited model recalls more old facts than the trained model did at the same accuracy on the new facts (Appendix D.2).

## 6 DISCUSSION

That a loss of performance need not be a loss of knowledge is an old idea: psychology separates stored from accessible information (Tulving & Pearlstone, 1966; Ebbinghaus, 1885), and suppressed associations return once interference stops (Postman et al., 1968). Connectionist networks show the same savings, and training them on unrelated associations can even restore forgotten ones (Hetherington & Seidenberg, 1989; Hinton & Plaut, 1987; Harvey & Stone, 1996; Atkins & Murre, 1998). In deep networks, linear probes often recover old tasks that the outputs no longer solve, a phenomenon called shallow forgetting (Murata et al., 2020; Davari et al., 2022; Hess et al., 2024; Lanzillotta et al., 2026); performance drops sharply at the start of a new task before returning with some replay (De Lange et al., 2023; Zucchet et al., 2025); and language models recover lost capabilities with a different prompt (Kotha et al., 2024) or a few old examples, a case named spurious forgetting (Zheng et al., 2025). In the settings we identify, the finetuning that hides the old facts also restores them, without probe, prompt or replay, through a shared shift that normalization withdraws once the new facts are learned. This shift towards the new answers may be one concrete form of the loss of task alignment that Zheng et al. (2025) identify behind spurious forgetting.

Forgetting is usually measured as a single quantity that grows with training, whereas our results suggest two processes on different timescales: a loss of access shared by all old facts, set by what the new data have in common, and a slower erosion specific to each fact. The pretrained model also points to a trade-off between learning new facts quickly and recovering old ones. When the subjects of the new facts share something that predicts their answers, as the kind of a product predicts the kind of its manufacturer, the shared direction helps the model learn them, and they are learned faster. But for the same reason the shift along this direction keeps helping the new facts, so it is never withdrawn, and the old facts stay hidden. What helps a model generalize to new facts may thus be what keeps it from recovering the old ones.

Limitations. Our minimal model is deliberately extreme, and normalization-driven recovery is likely weaker on real data, although the collapse may be more general. In the pretrained model, we attribute the difference in recovery to what the subjects reveal about their answers, but the two kinds of new facts also differ in other ways, and we study a single model.

## ACKNOWLEDGMENT

This work was supported in part by the German Federal Ministry of Education and Research (BMBF): Tubingen AI Center, FKZ: 01IS18039B; by the Deutsche Forschungsgemeinschaft (DFG,¨ German Research Foundation) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645; by Schmidt Sciences SAFE-AI Grant; by Coefficient Giving; and by the Hector Foundation. Resources used in preparing this research project were provided by the Max Planck Institute for Intelligent Systems and, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

## REFERENCES

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. In International Conference on Machine Learning, 2024.

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. In International Conference on Learning Representations, 2025.

Paul W. B. Atkins and Jaap M. J. Murre. Recovery of Unrehearsed Items in Connectionist Models. Connection Science, 1998.

Vivien Cabannes, Elvis Dohmatob, and Alberto Bietti. Scaling Laws for Associative Memories. In International Conference on Learning Representations, 2024.

MohammadReza Davari, Nader Asadi, Sudhir Mudur, Rahaf Aljundi, and Eugene Belilovsky. Probing Representation Forgetting in Supervised and Unsupervised Continual Learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Matthias De Lange, Gido M. van de Ven, and Tinne Tuytelaars. Continual Evaluation for Lifelong Learning: Identifying the Stability Gap. In International Conference on Learning Representations, 2023.

Hermann Ebbinghaus. Uber das Ged <sup>¨</sup> achtnis ¨ . Duncker & Humblot, Leipzig, 1885.

Robert M. French. Catastrophic Forgetting in Connectionist Networks. Trends in Cognitive Sciences, 1999.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer Feed-Forward Layers Are Key-Value Memories. In Conference on Empirical Methods in Natural Language Processing, 2021.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting Recall of Factual Associations in Auto-Regressive Language Models. In Conference on Empirical Methods in Natural Language Processing, 2023.

Ian J. Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks. In International Conference on Learning Representations, 2014.

Kshitij Gupta, Benjamin Therien, Adam Ibrahim, Mats L. Richter, Quentin Anthony, Eugene´ Belilovsky, Irina Rish, and Timothee Lesort. Continual Pre-Training of Large Language Mod-´ els: How to (Re)warm Your Model? arXiv preprint arXiv:2308.04014, 2023.

Inman Harvey and James V. Stone. Unicycling Helps Your French: Spontaneous Recovery of Associations by Learning Unrelated Tasks. Neural Computation, 1996.

Timm Hess, Eli Verwimp, Gido M. van de Ven, and Tinne Tuytelaars. Knowledge Accumulation in Continually Learned Representations and the Issue of Feature Forgetting. Transactions on Machine Learning Research, 2024.

Phil A. Hetherington and Mark S. Seidenberg. Is There ‘Catastrophic Interference’ in Connectionist Networks? In Annual Conference ofthe Cognitive Science Society, 1989.

Geoffrey E. Hinton and David C. Plaut. Using Fast Weights to Deblur Old Memories. In Annual Conference ofthe Cognitive Science Society, 1987.

Adam Ibrahim, Benjamin Therien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timoth ´ ee´ Lesort, Eugene Belilovsky, and Irina Rish. Simple and Scalable Strategies to Continually Pre-Train Large Language Models. Transactions on Machine Learning Research, 2024.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming Catastrophic Forgetting in Neural Networks. Proceedings ofthe National Academy ofSciences, 2017.

Suhas Kotha, Jacob Mitchell Springer, and Aditi Raghunathan. Understanding Catastrophic Forgetting in Language Models via Implicit Inference. In International Conference on Learning Representations, 2024.

Giulia Lanzillotta, Damiano Meier, and Thomas Hofmann. Heads Collapse, Features Stay: Why Replay Needs Big Buffers. In International Conference on Learning Representations, 2026.

Jiacheng Lin, Zhongruo Wang, Kun Qian, Tian Wang, Arvind Srinivasan, Hansi Zeng, Ruochen Jiao, Xie Zhou, Jiri Gesi, Dakuo Wang, et al. SFT Doesn’t Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs. In International Conference on Learning Representations, 2026.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-Tuning. IEEE Transactions on Audio, Speech and Language Processing, 2025.

Martin Marek, Dongkyu Cho, Shikai Qiu, Rumi Chunara, Pavel Izmailov, and Andrew Gordon Wilson. Forgetting in Language Models: Capacity, Optimization, and Self-Generated Replay. arXiv preprint arXiv:2605.26097, 2026.

Michael McCloskey and Neal J. Cohen. Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem. In Psychology ofLearning and Motivation. Academic Press, 1989.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and Editing Factual Associations in GPT. In Advances in Neural Information Processing Systems, 2022.

Kengo Murata, Tetsuya Toyota, and Kouzou Ohara. What Is Happening Inside a Continual Learning Model? A Representation-Based Evaluation of Representational Forgetting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2020.

Neel Nanda, Senthooran Rajamanoharan, Janos Kram ´ ar, and Rohin Shah. Fact Finding: Attempting ´ to Reverse-Engineer Factual Recall on the Neuron Level. Alignment Forum, 2023.

Eshaan Nichani, Jason D. Lee, and Alberto Bietti. Understanding Factual Recall in Transformers via Associative Memories. In International Conference on Learning Representations, 2025.

Leo Postman, Karen Stark, and Janet Fraser. Temporal Changes in Interference. Journal of Verbal Learning and Verbal Behavior, 1968.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models Are Unsupervised Multitask Learners. Technical report, OpenAI, 2019.

Vinay V. Ramasesh, Ethan Dyer, and Maithra Raghu. Anatomy of Catastrophic Forgetting: Hidden Representations and Task Semantics. In International Conference on Learning Representations, 2021.

Roger Ratcliff. Connectionist Models of Recognition Memory: Constraints Imposed by Learning and Forgetting Functions. Psychological Review, 1990.

Mark Rofin, Aditya Varre, and Nicolas Flammarion. (How) Learning Rates Regulate Catastrophic Overtraining. In Conference on Language Modeling, 2026.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Greg Wayne. Experience Replay for Continual Learning. In Advances in Neural Information Processing Systems, 2019.

Andrew M. Saxe, James L. McClelland, and Surya Ganguli. A mathematical theory of semantic development in deep neural networks. Proceedings of the National Academy of Sciences, 116 (23):11537–11546, May 2019. ISSN 1091-6490. doi: 10.1073/pnas.1820226116. URL http: //dx.doi.org/10.1073/pnas.1820226116.

Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. Simple Entity-Centric Questions Challenge Dense Retrievers. In Conference on Empirical Methods in Natural Language Processing, 2021.

Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi, Graham Neubig, and Aditi Raghunathan. Overtrained Language Models Are Harder to Fine-Tune. In International Conference on Machine Learning, 2025.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 OLMo 2 Furious. In Conference on Language Modeling, 2025.

Endel Tulving and Zena Pearlstone. Availability versus Accessibility of Information in Memory for Words. Journal ofVerbal Learning and Verbal Behavior, 1966.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. In Advances in Neural Information Processing Systems, 2017.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On Layer Normalization in the Transformer Architecture. In International Conference on Machine Learning, 2020.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual Learning Through Synaptic Intelligence. In International Conference on Machine Learning, 2017.

Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. In Advances in Neural Information Processing Systems, 2019.

Junhao Zheng, Xidi Cai, Shengjie Qiu, and Qianli Ma. Spurious Forgetting in Continual Learning of Language Models. In International Conference on Learning Representations, 2025.

Nicolas Zucchet, Jorg Bornschein, Stephanie C. Y. Chan, Andrew Kyle Lampinen, Razvan Pascanu,¨ and Soham De. How Do Language Models Learn Facts? Dynamics, Curricula and Hallucinations. In Conference on Language Modeling, 2025.

Nicolas Zucchet, Hyun Dong Lee, and Scott Linderman. Language Models Suffer From a Curse of Ambiguity. arXiv preprint arXiv:2608.15448, 2026.

## APPENDIX

A Experimental setup 15   
B Analysis of the minimal model 17   
C The controlled Transformer 30   
D The pretrained language mode 35

Table 1 lists where each analysis of the minimal model is repeated in the other models.
<table><tr><td rowspan="2">Analysis</td><td rowspan="2">Minimal model</td><td colspan="2">Transformer</td><td rowspan="2">OLMo 2 1B</td></tr><tr><td>attention-only</td><td>with MLP blocks</td></tr><tr><td>Collapse, recovery and erosion</td><td>Fig. 2</td><td>Fig. 1</td><td>App. C.5</td><td>Fig. 5</td></tr><tr><td>New answers in both regions</td><td>Fig. 2</td><td>Fig. 1</td><td>App. C.5</td><td></td></tr><tr><td>Common shift and fact-specific drift</td><td>App. B.10</td><td>Fig. 4</td><td>App. C.5</td><td></td></tr><tr><td>Removing the common shift</td><td>App. B.9</td><td>Fig. 4</td><td>App. C.5</td><td></td></tr><tr><td>Ordering within the region</td><td>App. B.8</td><td>App. C.1</td><td>App. C.5</td><td></td></tr><tr><td>Weights that carry the shift</td><td></td><td>App. C.4</td><td>App. C.5</td><td></td></tr><tr><td>Learning rate and batch size</td><td>App. C.3</td><td>App. C.2</td><td>App. C.5</td><td></td></tr><tr><td>Number of new facts</td><td>App. C.3</td><td>App. C.3</td><td>App. C.3</td><td>Fig. 5</td></tr><tr><td>Removing the top direction of the update</td><td>App. B.10</td><td>App. D.3</td><td>App. D.3</td><td>Fig. 5</td></tr></table>

Table 1: Reference location for each analysis, for each model.

![](images/c95e9b7256971ea826219dd2816424fec471991e86cd4cca786ff0a38f9f3aec.jpg)  
Figure 6: Examples from the three datasets. Each fact is learned from training phrasings and evaluated on held-out phrasings; answers in bold. The old facts of the language model are facts it already knows, and the new facts concern synthetic individuals, whose answers are unrelated to their names, or little-known real entities.

## A EXPERIMENTAL SETUP

## A.1 CONTROLLED TRANSFORMER

Data. The transformer is trained on synthetic biographies (Allen-Zhu & Li, 2024; Zucchet et al., 2025). Each individual has a unique name, made of a first, a middle and a last name, and six attributes: birth year, birthplace, university, field of study, employer and work city. Each attribute takes one value from a fixed pool of between 127 and 221 values. Every fact is stated in a sentence drawn from 25 phrasings of its attribute; for each individual and attribute, 20 phrasings are used for training and the remaining 5 are held out for evaluation. A biography consists of one sentence per attribute, in random order.

Old and new facts. The possible values of each attribute are split at random into two halves. Two pretraining populations of 2,000 individuals each draw their values from one half and from the other, respectively. Finetuning uses 500 new individuals, none of whom appear in pretraining, whose values are drawn either from the second half only (Figure 1, right) or from both halves (Figure 1, left).

Model and training. The model is an 8-layer transformer with model dimension 512, 8 attention heads, rotary position embeddings and pre-layer normalization, without MLP blocks and without dropout. It is pretrained for 16,000 steps with batches of 256 biographies, using AdamW $( \beta _ { 1 } = 0 . 9 ;$ $\beta _ { 2 } ~ = ~ 0 . 9 5$ , weight decay 0.1, gradient clipping at 1) with a cosine learning-rate schedule from $5 \times 1 0 ^ { - 4 }$ , until it recalls the pretraining facts perfectly. Finetuning uses the same optimizer and batch size with a peak learning rate of $3 . 7 5 \times 1 0 ^ { - 5 }$ over 1,200 steps, and sees only the new individuals. Results are averaged over three to five finetuning seeds.

Evaluation. Recall is the accuracy on the first token of the attribute value, predicted from the held-out phrasings of each fact. The within-region rank used in Appendix C.1 ranks the correct value among the values of its own half.

## A.2 PRETRAINED LANGUAGE MODEL

Old facts. The old facts are CounterFact facts (Meng et al., 2022) that OLMo 2 1B (Team OLMo et al., 2025) answers correctly before finetuning, 1,106 in total across relations such as manufacturers, headquarters and citizenship. Evaluation uses the 422 facts whose answer does not appear in the prompt, so that accuracy measures recall rather than copying.

New facts. The synthetic individuals have invented names and four attributes (employer, birth city, citizenship and language), with values drawn from pools of 66 real entities whose first tokens differ from those of the old answers. The large finetuning set contains 1,000 individuals and the small set 250. The real entities are products from EntityQuestions (Sciavolino et al., 2021) with their manufacturers: 1,415 facts over 407 manufacturers, restricted to facts the model does not know before finetuning, whose answers share no first token with the old answers and do not appear in the subject’s name, with at most 60 facts per manufacturer. Each new fact is stated through 15 phrasings, 12 for training and 3 held out for evaluation.

Finetuning. Training documents are rendered afresh at every step from the training phrasings, with 32,768 tokens per step in sequences of 512 tokens. We use AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , no weight decay, gradient clipping at 1) with a learning rate of $1 0 ^ { - 5 }$ after a 100-step linear warmup, and no replay of any pretraining data. Results are averaged over three seeds.

Evaluation. Recall is the top-1 accuracy of the first answer token over the full vocabulary, on the held-out phrasings for the new facts and on the CounterFact prompts for the old facts.

## B ANALYSIS OF THE MINIMAL MODEL

This section derives the results of Sections 3.2 and 3.3: the decomposition of the displacement of the old facts, how the new facts write into them, the dynamics of the common shift, and the size of the fact-specific displacements. We also describe how we verify these results in simulations, and the interventions we run on the minimal model.

## B.1 NOTATION, ASSUMPTIONS AND SIMULATIONS

Recall that each fact i is represented by a key

$$
k _ { i } = \sqrt { \alpha } \mu + \sqrt { 1 - \alpha } e _ { i } \in \mathbb { R } ^ { d } ,\tag{6}
$$

with $\mu$ a unit vector shared across all facts and $e _ { i }$ drawn uniformly on the unit sphere, independently for each fact. The model computes

$$
\begin{array} { r } { h _ { i } = W _ { 1 } k _ { i } , \quad x _ { i } = \mathrm { R M S } ( h _ { i } ) = \sqrt { d } \frac { h _ { i } } { \| h _ { i } \| } , \quad z _ { i } = W _ { 2 } x _ { i } , \quad p _ { i } = \mathrm { s o f t m a x } ( z _ { i } ) , } \end{array}\tag{7}
$$

so that every normalized hidden state has norm ${ \sqrt { d } } .$ During finetuning, the loss is the cross-entropy on the new facts only,

$$
L _ { B } = \frac { 1 } { n _ { B } } \sum _ { b \in B } \ell _ { b } , \quad \ell _ { b } = - \log p _ { b , y _ { b } } .\tag{8}
$$

We use the index a for old facts (set A) and b for new facts (set $B ^ { \prime }$ , written B in subscripts), bars for averages over a set (e.g. $\begin{array} { r } { \bar { h } _ { A } = \frac { 1 } { n _ { A } } \sum _ { a } h _ { a } ) } \end{array}$ , hats for unit vectors (e.g. $\hat { h } = h / \| h \| )$ , and ${ \mathbf { 1 } } _ { y }$ for the one-hot vector of token $y .$ Time $t ~ = ~ 0$ corresponds to the beginning of finetuning, and $\Delta W _ { 1 } = W _ { 1 } ( t ) - W _ { 1 } ( 0 )$

Assumptions. The analysis relies on four assumptions, each used only where stated:

Asm. 1 Gradient flow dynamics. We study the gradient flow $\dot { W } _ { 1 } ~ = ~ - \nabla _ { W _ { 1 } } L _ { B } , ~ \dot { W } _ { 2 } ~ =$ $- \nabla _ { { W _ { 2 } } } L _ { B }$ on the finetuning loss. Minibatch gradient descent with a small learning rate $\eta ,$ which we use in our simulations, approximates it, one step corresponding to a time increment η.

Asm. 2 High-dimensional keys. The overlap $u ^ { \top }$ v between two independent random unit vectors of $\mathbb { R } ^ { d }$ has zero mean and variance $1 / d .$ . We therefore treat all the overlaps $e _ { i } ^ { \top } e _ { j } \mid ( i \neq$ $j )$ and $\mu ^ { \top } e _ { i }$ as small, of order $d ^ { - 1 / 2 }$ , and only keep the leading-order terms in these overlaps.

Asm. 3 The update does not depend on the old keys. The old facts are not trained during finetuning, so $\Delta W _ { 1 }$ only depends on their keys through the parameters at the end of pretraining. We neglect this dependence and treat the $e _ { a }$ as independent of $\Delta W _ { 1 }$ . This assumption is only used in Section B.11.

Asm. 4 Common coordinate. All new hidden states have the same coordinate along the shared direction $\hat { u } _ { \mathrm { s h } }$ and the same norm orthogonal to it. This assumption is only used to obtain the closed-form Equation 5 (Section B.6); the full dynamics do not rely on it.

The readout $W _ { 2 }$ does not appear in the old hidden states $h _ { a } = W _ { 1 } k _ { a }$ . All the results on the hidden states below therefore hold whether the readout is trained or not: W<sub>2</sub> only enters through the errors of the new facts, evaluated at its current value. We return to the effect of the readout on the outputs of the old facts in Section B.12.

Simulations. All simulations use $d = 1 2 8$ , a vocabulary of $V = 3 2$ tokens split into two halves of 16, and $n = 1 2 8$ facts in each of A, B and $B ^ { \prime }$ , with $\alpha = 0 . 5$ unless stated otherwise. The model is pretrained on A and B with minibatches of 32 facts until both sets are recalled with accuracy at least 0.99, then finetuned on $B ^ { \prime }$ alone with minibatches of 32 new facts, at a learning rate chosen so that the new facts reach the same accuracy around step 200. Results are averaged over ten seeds unless stated otherwise.

## B.2 GRADIENTS OF THE FINETUNING LOSS

We first compute the gradients of the finetuning loss with respect to the hidden states and to the two weight matrices, which the rest of the analysis uses throughout.

Error at the logits and at the normalized state. For the cross-entropy loss, the error of the new fact b with respect to its logits is

$$
\nabla _ { z _ { b } } \ell _ { b } = p _ { b } - { \mathbf 1 } _ { y _ { b } } ,\tag{9}
$$

and, as $z _ { b } = W _ { 2 } x _ { b }$ , its error with respect to the normalized state is

$$
\nabla _ { x _ { b } } \ell _ { b } = { W } _ { 2 } ^ { \top } \left( p _ { b } - \mathbf { 1 } _ { y _ { b } } \right) .\tag{10}
$$

Error at the hidden state. We denote by $\delta _ { b } : = \nabla _ { h _ { b } } \ell _ { b }$ the error of the new fact at its hidden state. Without normalization, $x _ { b } = h _ { b }$ and $\delta _ { b } = { W _ { 2 } ^ { \top } \left( p _ { b } - \mathbf { 1 } _ { y _ { b } } \right) }$ . With normalization, the Jacobian of $h \mapsto { \sqrt { d } } h / \| h \|$ is

$$
{ \frac { \partial x } { \partial h } } = { \sqrt { d } } \left( { \frac { I } { \| h \| } } - { \frac { h h ^ { \top } } { \| h \| ^ { 3 } } } \right) = { \frac { \sqrt { d } } { \| h \| } } \left( I - { \hat { h } } { \hat { h } } ^ { \top } \right) .\tag{11}
$$

It is symmetric, so

$$
\delta _ { b } = \frac { \sqrt { d } } { \| h _ { b } \| } \left( I - \hat { h } _ { b } \hat { h } _ { b } ^ { \top } \right) W _ { 2 } ^ { \top } \big ( p _ { b } - \mathbf { 1 } _ { y _ { b } } \big ) ,\tag{12}
$$

which is the expression of the main text. The normalization has two effects on the error: it divides it by the norm of the state, and it removes its component along the state, $\hat { h } _ { b } ^ { \top } \delta _ { b } = 0$ , since the loss does not depend on the norm of $h _ { b }$

The removed component has a simple interpretation. As $\hat { h } _ { b } ^ { \top } W _ { 2 } ^ { \top } \big ( p _ { b } - \mathbf { 1 } _ { y _ { b } } \big ) = d ^ { - 1 / 2 } x _ { b } ^ { \top } W _ { 2 } ^ { \top } ( p _ { b } -$ ${ \bf 1 } _ { y _ { b } } ) = d ^ { - 1 / 2 } z _ { b } ^ { \top } ( p _ { b } - { \bf 1 } _ { y _ { b } } )$ , we have

$$
\delta _ { b } = \frac { \sqrt { d } } { \| h _ { b } \| } W _ { 2 } ^ { \top } \big ( p _ { b } - \mathbf { 1 } _ { y _ { b } } \big ) + \frac { \gamma _ { b } } { \| h _ { b } \| } \hat { h } _ { b } , \quad \mathrm { w i t h } \gamma _ { b } : = z _ { b , y _ { b } } - p _ { b } ^ { \top } z _ { b } .\tag{13}
$$

Here $\gamma _ { b }$ is the confidence margin of the new fact: the logit of the correct answer minus the average logit under the model’s prediction. It is negative when the model confidently predicts a wrong answer and positive when it predicts the correct one. The first term is, up to a rescaling, the error a model without normalization would receive. The second term points along the new fact’s own state, with a sign given by its margin; for the new fact itself, it cancels the component of the first term along $\hat { h } _ { b }$

Gradients with respect to the weights. As $h _ { b } = W _ { 1 } k _ { b }$ , the gradient of the loss of the new fact b with respect to $W _ { 1 }$ is the outer product of its error and its key, $\mathsf { \bar { V } } _ { W _ { 1 } } \ell _ { b } = \delta _ { b } k _ { b } ^ { \top }$ , and averaging over the new facts gives

$$
\nabla _ { W _ { 1 } } L _ { B } = \frac { 1 } { n _ { B } } \sum _ { b } \delta _ { b } \boldsymbol { k } _ { b } ^ { \intercal } .\tag{14}
$$

Similarly, as $z _ { b } = W _ { 2 } x _ { b }$

$$
\nabla _ { W _ { 2 } } L _ { B } = \frac { 1 } { n _ { B } } \sum _ { b } \left( p _ { b } - \mathbf { 1 } _ { y _ { b } } \right) \boldsymbol { x } _ { b } ^ { \intercal } .\tag{15}
$$

As $W _ { 2 }$ sits after the normalization, its gradient does not involve the Jacobian above: unlike $W _ { 1 }$ , it receives no term along the new facts’ states.

## B.3 DECOMPOSITION OF THE DISPLACEMENT OF THE OLD FACTS

The keys are fixed and the old facts are not trained, so the displacement of an old hidden state is entirely due to the change of $W _ { 1 }$ :

$$
h _ { a } ( t ) - h _ { a } ( 0 ) = \Delta W _ { 1 } k _ { a } = \underbrace { { \sqrt { \alpha } } \Delta W _ { 1 } \mu } _ { \mathrm { c o m m o n } \mathrm { s h i f t } } + \underbrace { { \sqrt { 1 - \alpha } } \Delta W _ { 1 } e _ { a } } _ { \mathrm { i n d i v i d u a l \ d i s p l a c e m e n t } } .\tag{16}
$$

This decomposition is exact. Averaging it over the old facts gives

$$
\bar { h } _ { A } ( t ) - \bar { h } _ { A } ( 0 ) = \sqrt { \alpha } \Delta W _ { 1 } \mu + \sqrt { 1 - \alpha } \Delta W _ { 1 } \bar { e } _ { A } ,\tag{17}
$$

and the deviation of each old fact from this mean motion is

$$
\varepsilon _ { a } : = \big ( h _ { a } ( t ) - h _ { a } ( 0 ) \big ) - \big ( \bar { h } _ { A } ( t ) - \bar { h } _ { A } ( 0 ) \big ) = \sqrt { 1 - \alpha } \Delta W _ { 1 } ( e _ { a } - \bar { e } _ { A } ) .\tag{18}
$$

The mean displacement thus equals the common shift $\sqrt { \alpha } \Delta W _ { 1 } \mu$ up to the term $\sqrt { 1 - \alpha } \Delta W _ { 1 } \bar { e } _ { A } .$ As $\bar { e } _ { A }$ is the average of $n _ { A }$ independent random unit vectors, $\mathbb { E } \| \bar { e } _ { A } \| ^ { 2 } = 1 / n _ { A }$ , and the calculation of Section B.11 shows that this term is a factor $\sqrt { n _ { A } }$ smaller than a typical individual displacement $\varepsilon _ { \underline { { a } } }$ . We therefore measure the common motion through the norm of the mean displacement, $s =$ $\| \bar { h } _ { A } ( t ) - \bar { h } _ { A } ( 0 ) \|$ ∥ (Figure 3, middle), and the fact-specific motion through the spread

$$
\varepsilon _ { \mathrm { r m s } } ( t ) : = \left( \frac { 1 } { n _ { A } } \sum _ { a } \lVert \varepsilon _ { a } ( t ) \rVert ^ { 2 } \right) ^ { 1 / 2 }\tag{19}
$$

(Figure 14). When $\alpha = 0 .$ , the common shift vanishes identically, and the whole displacement is fact-specific.

## B.4 HOW THE NEW FACTS WRITE INTO THE OLD HIDDEN STATES

The write of a single new fact. Under the gradient flow dynamics, Equation 14 gives

$$
\dot { W } _ { 1 } = - \nabla _ { W _ { 1 } } L _ { B } = - \frac { 1 } { n _ { B } } \sum _ { b } \delta _ { b } \boldsymbol { k } _ { b } ^ { \top } ,\tag{20}
$$

so each new fact writes its error $\delta _ { b }$ into $W _ { 1 }$ along its own key. The old facts are not part of the finetuning loss, but their hidden states move with $W _ { 1 }$ , and as their keys are fixed,

$$
\dot { h } _ { a } = \frac { \mathrm { d } } { \mathrm { d } t } \big ( W _ { 1 } k _ { a } \big ) = \dot { W } _ { 1 } k _ { a } = - \frac { 1 } { n _ { B } } \sum _ { b } \left( k _ { b } ^ { \top } k _ { a } \right) \delta _ { b } ,\tag{21}
$$

where we used that $k _ { b } ^ { \top } k _ { a }$ is a scalar. Each new fact thus adds its error $\delta _ { b }$ to every old state, weighted by the overlap of their keys. This equation is exact.

Key overlaps. Expanding the keys and using $\mu ^ { \top } \boldsymbol { \mu } = 1$ , the overlap between a new and an old key is

$$
\begin{array} { r } { k _ { b } ^ { \top } k _ { a } = \alpha + \sqrt { \alpha ( 1 - \alpha ) } \left( \mu ^ { \top } e _ { a } + \mu ^ { \top } e _ { b } \right) + ( 1 - \alpha ) e _ { b } ^ { \top } e _ { a } . } \end{array}\tag{22}
$$

Under Assumption $^ { 2 , }$ the last three terms have zero mean and are of order $d ^ { - 1 / 2 }$ , so every pair of keys overlaps by approximately $\alpha .$ For the same reason, $\| \boldsymbol { k } _ { b } \| ^ { 2 } = 1 + 2 \sqrt { \alpha ( 1 - \alpha ) } \mu ^ { \top } \boldsymbol { e } _ { b } \simeq 1$

The mean write. Equation 21 is linear in $k _ { a } .$ , so averaging it over the old facts amounts to replacing $k _ { a }$ by their mean key:

$$
\dot { \bar { h } } _ { A } = \frac { 1 } { n _ { A } } \sum _ { a } \dot { h } _ { a } = - \frac { 1 } { n _ { B } } \sum _ { b } \left( k _ { b } ^ { \top } \bar { k } _ { A } \right) \delta _ { b } , \quad \mathrm { w i t h ~ } \bar { k } _ { A } = \sqrt { \alpha } \mu + \sqrt { 1 - \alpha } \bar { e } _ { A } .\tag{23}
$$

The average leaves the shared vector $\mu$ untouched and reduces the fact-specific parts to $\bar { e } _ { A }$ . The overlap of a new key with the mean old key is

$$
\begin{array} { r l } & { k _ { b } ^ { \top } \bar { k } _ { A } = \left( \sqrt { \alpha } \mu + \sqrt { 1 - \alpha } e _ { b } \right) ^ { \top } \left( \sqrt { \alpha } \mu + \sqrt { 1 - \alpha } \bar { e } _ { A } \right) } \\ & { ~ = \alpha + \sqrt { \alpha ( 1 - \alpha ) } \mu ^ { \top } \bar { e } _ { A } + \sqrt { \alpha ( 1 - \alpha ) } \mu ^ { \top } e _ { b } + ( 1 - \alpha ) e _ { b } ^ { \top } \bar { e } _ { A } } \\ & { ~ = \alpha \left( 1 + \xi \right) + \zeta _ { b } , } \end{array}\tag{24}
$$

where we grouped the terms that are the same for all new facts into

$$
\xi : = \sqrt { \frac { 1 - \alpha } { \alpha } } \mu ^ { \top } \bar { e } _ { A } ,\tag{25}
$$

and the terms that depend on the new fact into

$$
\zeta _ { b } : = \sqrt { \alpha ( 1 - \alpha ) } \mu ^ { \top } e _ { b } + ( 1 - \alpha ) e _ { b } ^ { \top } \bar { e } _ { A } .\tag{26}
$$

Plugging this decomposition into Equation 23 and using $\begin{array} { r } { \frac { 1 } { n _ { B } } \sum _ { b } \delta _ { b } = \bar { \delta } _ { B } } \end{array}$ gives

$$
\dot { \bar { h } } _ { A } = \underbrace { - \alpha \left( 1 + \xi \right) \bar { \delta } _ { B } } _ { \mathrm { s h a r e d } } \underbrace { - \frac { 1 } { n _ { B } } \sum _ { b } \zeta _ { b } \delta _ { b } } _ { \mathrm { s p e c i f i c } } .\tag{27}
$$

The sizes of the two terms are as follows.

– Shared term. All new facts contribute to it with the same coefficient $\alpha ( 1 + \xi )$ . As $\bar { e } _ { A }$ is the average of $n _ { A }$ independent random unit vectors, $\mu ^ { \top } \bar { e } _ { A }$ has zero mean and variance $1 / ( n _ { A } d )$ , so ξ is of order $\bar { ( n _ { A } d ) } ^ { - 1 / 2 }$ and the coefficient is α to leading order.

– Specific term. $\mu ^ { \top } e _ { b }$ has variance $1 / d$ and $e _ { b } ^ { \top } \bar { e } _ { A }$ has variance $1 / ( n _ { A } d )$ , so the coefficients $\zeta _ { b }$ have zero mean and are of order $d ^ { - 1 / 2 }$ . They are moreover uncorrelated across new facts, as $e _ { b }$ and $e _ { b ^ { \prime } }$ are independent for $b \neq b ^ { \prime }$ . Treating them, to leading order, as independent of the errors $\delta _ { b }$ gives

$$
\mathbb { E } \left\| \frac { 1 } { n _ { B } } \sum _ { b } \zeta _ { b } \delta _ { b } \right\| ^ { 2 } = \frac { 1 } { n _ { B } ^ { 2 } } \sum _ { b } \mathbb { E } \big [ \zeta _ { b } ^ { 2 } \big ] \| \delta _ { b } \| ^ { 2 } = O \bigg ( \frac { \| \delta \| _ { \mathrm { r m s } } ^ { 2 } } { n _ { B } d } \bigg ) ,\tag{28}
$$

with $\begin{array} { r } { \| \delta \| _ { \mathrm { r m s } } ^ { 2 } = \frac { 1 } { n _ { B } } \sum _ { b } \| \delta _ { b } \| ^ { 2 } } \end{array}$ . The cross terms vanish because the coefficients are uncorrelated: the $n _ { B }$ contributions add up like a random walk, and their sum grows like $\sqrt { n _ { B } }$ instead of $n _ { B }$

The specific term is therefore negligible compared with the shared one as soon as $\alpha \| \bar { \delta } _ { B } \| \ \gg$ $\| \delta \| _ { \mathrm { r m s } } / \sqrt { n _ { B } d } .$ , that is, as soon as the errors of the new facts share a common component. Keeping only the leading order of the shared term gives Equation 4,

$$
\dot { \bar { h } } _ { A } \simeq - \alpha \bar { \delta } _ { B } ,\tag{29}
$$

and when $\alpha = 0 ,$ , only the specific term is left.

Why the errors of the new facts share a common component. The shared term is only large if the errors $\delta _ { b }$ do not cancel when averaged over the new facts. Averaging the decomposition of Equation 13 over the new facts gives

$$
\bar { \delta } _ { B } = \frac { 1 } { n _ { B } } \sum _ { b } \frac { \sqrt { d } } { \| h _ { b } \| } W _ { 2 } ^ { \top } \big ( p _ { b } - \mathbf { 1 } _ { y _ { b } } \big ) + \frac { 1 } { n _ { B } } \sum _ { b } \frac { \gamma _ { b } } { \| h _ { b } \| } \hat { h } _ { b } .\tag{30}
$$

Early in finetuning, both terms survive the average, for two reasons that both come from the structure of the data:

The new answers all lie in the same region. Up to the variations of the prefactor $\sqrt { d } / \| h _ { b } \|$ across new facts, the first term is proportional to $W _ { 2 } ^ { \top } ( \bar { p } _ { B } - q _ { B } )$ , where ${ \bar { p } } _ { B }$ is the average prediction on the new facts and $\begin{array} { r } { q _ { B } = \frac { 1 } { n _ { B } } \sum _ { b } \mathbf { 1 } _ { y _ { b } } } \end{array}$ the distribution of their answers. All new answers belong to the same half of the vocabulary, so $q _ { B }$ puts all its mass on that half. At the beginning of finetuning, the model has no reason to favor this half for the new keys: it was pretrained on answers from both halves, and its predictions for unseen keys spread over both. $\bar { p } _ { B } \ : - \ : q _ { B }$ therefore has a large component that moves probability mass from one half to the other, and this component is the same for all new facts. By contrast, which answer is correct within the half, and which wrong answer the model predicts, change from one new fact to the other and cancel in the average. When the new answers are drawn from the whole vocabulary, $q _ { B }$ and ${ \bar { p } } _ { B }$ spread their mass over the two halves in a similar way and this component largely disappears, which is why the collapse does too (Figure 2, panel 2).

– The new facts are confidently wrong. At the beginning of finetuning, the model confidently predicts wrong answers for the new facts, so their margins $\gamma _ { b }$ are all negative. At the same time, all new hidden states contain the shared component $\sqrt { \alpha } W _ { 1 } \mu$ . The second term is thus a sum of vectors that share a direction, weighted by coefficients of the same sign, and it does not cancel either. This term gives rise to the $\gamma _ { \mathrm { i n d } , b }$ contribution of Section B.5.

Both contributions vanish as the new facts are learned: when $p _ { b }  \mathbf { 1 } _ { y _ { b } } , W _ { 2 } ^ { \top } ( p _ { b } - \mathbf { 1 } _ { y _ { b } } )  0$ and $\gamma _ { b }  0$ . The common component of the errors, and with it the shared write, therefore only exists early in finetuning.

Why the common write is faster than fact-specific learning. The same calculation applied to a new fact b separates its own contribution from that of the other new facts:

$$
\dot { h } _ { b } = - \frac { 1 } { n _ { B } } \| k _ { b } \| ^ { 2 } \delta _ { b } - \frac { 1 } { n _ { B } } \sum _ { b ^ { \prime } \neq b } \left( k _ { b ^ { \prime } } ^ { \top } k _ { b } \right) \delta _ { b ^ { \prime } } \simeq - \frac { 1 } { n _ { B } } \delta _ { b } - \alpha \bar { \delta } _ { B } ,\tag{31}
$$

where we used $\| \boldsymbol k _ { b } \| ^ { 2 } \simeq 1$ . The error of a single fact enters its own state with weight $1 / n _ { B } ,$ whereas the common component of all errors enters every state with weight α. Whatever the errors of the new facts have in common is therefore learned much faster than what is specific to each, which is why the reversal of the common shift waits for fact-specific learning.

## B.5 DYNAMICS OF THE COMMON SHIFT

The normalization term of Equation 13 cancels along $\hat { h } _ { b }$ for the new fact itself, but not for the old facts: through Equation 21, it moves every old state along the parts that its state shares with $h _ { b }$ . We make this precise by following the displacement of the old facts’ mean state. Let $s =$ $\| \bar { h } _ { A } ( t ) - \bar { h } _ { A } ( 0 ) \|$ be its length, which we call the shared coordinate, and $\hat { u } _ { \mathrm { s h } } = ( \bar { h } _ { A } ( t ) - \bar { h } _ { A } ( 0 ) ) / s$ its direction, which we call the shared direction. As $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \| v \| = \hat { v } ^ { \top } \dot { v } } \end{array}$ for any vector v, and $\bar { h } _ { A } ( 0 )$ is constant, Equation 4 gives

$$
\dot { s } = \hat { u } _ { \mathrm { s h } } ^ { \top } \dot { \bar { h } } _ { A } \simeq - \frac { \alpha } { n _ { B } } \sum _ { b } \hat { u } _ { \mathrm { s h } } ^ { \top } \delta _ { b } .\tag{32}
$$

Validity of the approximation. Equation 4 neglects the specific term of Equation 27, and only its projection onto $\hat { u } _ { \mathrm { s h } }$ matters for the dynamics of s. The shared term contributes α times the mean of the projected errors $\hat { u } _ { \mathrm { s h } } ^ { \top } \delta _ { b }$ , while the specific term, whose coefficients are random and of order $d ^ { - 1 / 2 }$ contributes a quantity of the order of their root mean square divided by ${ \sqrt { n _ { B } d } } .$ The approximation therefore requires the projected errors to share a sign across new facts, not the errors to be large. This holds while the new facts are all wrong and once they are all learned, even when their errors become small and mostly fact-specific. It only fails around the peak of $s ,$ where the projected errors change sign at different times for different new facts and s˙ is small anyway.

Projecting the error. We decompose the direction of each new state into its component along $\hat { u } _ { \mathrm { s h } }$ and an orthogonal component,

$$
\hat { h } _ { b } = \cos \theta _ { b } \hat { u } _ { \mathrm { s h } } + \sin \theta _ { b } \hat { u } _ { \mathrm { i n d } , b } ,\tag{33}
$$

where $\theta _ { b }$ is the angle between $h _ { b }$ and $\hat { u } _ { \mathrm { s h } }$ , and $\hat { u } _ { \mathrm { i n d } , b }$ is the unit vector along the individual part $h _ { b } - \left( \hat { u } _ { \mathrm { s h } } ^ { \top } h _ { b } \right) \hat { u } _ { \mathrm { s h } }$ . Then

$$
\begin{array} { r l } & { \hat { u } _ { \mathrm { s h } } ^ { \top } \left( I - \hat { h } _ { b } \hat { h } _ { b } ^ { \top } \right) = \hat { u } _ { \mathrm { s h } } ^ { \top } - \cos \theta _ { b } \hat { h } _ { b } ^ { \top } } \\ & { \qquad = \hat { u } _ { \mathrm { s h } } ^ { \top } - \cos ^ { 2 } \theta _ { b } \hat { u } _ { \mathrm { s h } } ^ { \top } - \cos \theta _ { b } \sin \theta _ { b } \hat { u } _ { \mathrm { i n d } , b } ^ { \top } } \\ & { \qquad = \sin ^ { 2 } \theta _ { b } \hat { u } _ { \mathrm { s h } } ^ { \top } - \sin \theta _ { b } \cos \theta _ { b } \hat { u } _ { \mathrm { i n d } , b } ^ { \top } , } \end{array}\tag{34}
$$

so that

$$
\hat { u } _ { \mathrm { s h } } ^ { \top } \delta _ { b } = \frac { \sqrt { d } } { \left. h _ { b } \right. } \left( \sin ^ { 2 } \theta _ { b } \hat { u } _ { \mathrm { s h } } ^ { \top } W _ { 2 } ^ { \top } \big ( p _ { b } - \mathbf { 1 } _ { y _ { b } } \big ) - \sin \theta _ { b } \cos \theta _ { b } \hat { u } _ { \mathrm { i n d } , b } ^ { \top } W _ { 2 } ^ { \top } \big ( p _ { b } - \mathbf { 1 } _ { y _ { b } } \big ) \right) .\tag{35}
$$

Two margins. The two projections of the error are margins:

$$
\begin{array} { r } { \hat { u } _ { \mathrm { s h } } ^ { \top } W _ { 2 } ^ { \top } \left( p _ { b } - \mathbf { 1 } _ { y _ { b } } \right) = ( W _ { 2 } \hat { u } _ { \mathrm { s h } } ) ^ { \top } \left( p _ { b } - \mathbf { 1 } _ { y _ { b } } \right) = - \gamma _ { \mathrm { s h } , b } , \quad \mathrm { w i t h } \ \gamma _ { \mathrm { s h } , b } : = ( W _ { 2 } \hat { u } _ { \mathrm { s h } } ) _ { y _ { b } } - p _ { b } ^ { \top } W _ { 2 } \hat { u } _ { \mathrm { s h } , b } , } \end{array}\tag{36}
$$

and similarly $\hat { u } _ { \mathrm { i n d } , b } ^ { \top } W _ { 2 } ^ { \top } \left( p _ { b } - \mathbf { 1 } _ { y _ { b } } \right) = - \gamma _ { \mathrm { i n d } , b }$ with $\gamma _ { \mathrm { i n d } , b } : = ( W _ { 2 } \hat { u } _ { \mathrm { i n d } , b } ) _ { y _ { b } } - p _ { b } ^ { \top } W _ { 2 } \hat { u } _ { \mathrm { i n d } , b }$ . The shared margin $\gamma _ { \mathrm { s h } , b }$ is the margin by which the readout of the shared direction favors the correct answer of b over the model’s current prediction, and the individual margin $\gamma _ { \mathrm { i n d } , b }$ is the same quantity for the individual direction of $b .$ $\begin{array} { r } { . \mathrm { ~ \ A s ~ } z _ { b } = \sqrt { d } \left( \cos \theta _ { b } { W } _ { 2 } \hat { u } _ { \mathrm { s h } } + \sin \theta _ { b } { W } _ { 2 } \hat { u } _ { \mathrm { i n d } , b } \right) } \end{array}$ , the two margins decompose the confidence margin γ of Equation 13 into a shared and an individual contribution,

$$
\gamma _ { b } = \sqrt { d } \left( \cos \theta _ { b } \gamma _ { \mathrm { s h } , b } + \sin \theta _ { b } \gamma _ { \mathrm { i n d } , b } \right) .\tag{37}
$$

In particular, $\gamma _ { \mathrm { i n d } , b }$ is negative while the individual part of b points to a wrong answer the model is confident about, and becomes positive once it points to the correct one.

The full dynamics. Putting everything together, we obtain

$$
\dot { s } \simeq \frac { \alpha } { n _ { B } } \sum _ { b } \frac { \sqrt { d } } { \| h _ { b } \| } \left[ \sin ^ { 2 } \theta _ { b } \gamma _ { \mathrm { s h } , b } - \sin \theta _ { b } \cos \theta _ { b } \gamma _ { \mathrm { i n d } , b } \right] ,\tag{38}
$$

where the only approximation is Equation 4. Without normalization, $\delta _ { b } = { W _ { 2 } ^ { \top } } \left( { p _ { b } - \mathbf { 1 } } _ { y _ { b } } \right)$ and $\hat { u } _ { \mathrm { s h } } ^ { \top } \delta _ { b } = - \gamma _ { \mathrm { s h } , b } ,$ so

$$
\dot { s } \simeq \alpha \bar { \gamma } _ { \mathrm { s h } } , \quad \mathrm { w i t h } \bar { \gamma } _ { \mathrm { s h } } = \frac { 1 } { n _ { B } } \sum _ { b } \gamma _ { \mathrm { s h } , b } .\tag{39}
$$

The term in $\gamma _ { \mathrm { i n d } , b }$ only exists because the normalization removes the radial component of the error, and it is the only term whose sign is controlled by how well the individual parts of the new facts are learned. Finally, both margins vanish when the new facts are learned: as $p _ { b }  \mathbf { 1 } _ { y _ { b } } , \gamma _ { \mathrm { s h } , b }  0$ and $\gamma _ { \mathrm { i n d } , b }  0$ . The force that writes the common shift and the force that withdraws it therefore both fade as the errors of the new facts vanish, with and without normalization.

## B.6 REDUCTION UNDER THE COMMON-COORDINATE APPROXIMATION

Equation 38 depends on every new fact through $\theta _ { b } , \left| \left| h _ { b } \right| \right| , \gamma _ { \mathrm { s h } , b }$ and $\gamma _ { \mathrm { i n d } , b }$ . To obtain a closed form, we use Assumption 4: all new states have the same coordinate s along $\hat { u } _ { \mathrm { s h } }$ and the same individual norm σ. This is motivated by the fact that the shared part of every hidden state, $\sqrt { \alpha } W _ { 1 } \mu ,$ , is the same for old and new facts. Under this assumption,

$$
\left\| h _ { b } \right\| = \sqrt { s ^ { 2 } + \sigma ^ { 2 } } , \quad \cos \theta _ { b } = \frac { s } { \sqrt { s ^ { 2 } + \sigma ^ { 2 } } } , \quad \sin \theta _ { b } = \frac { \sigma } { \sqrt { s ^ { 2 } + \sigma ^ { 2 } } } ,\tag{40}
$$

so that

$$
{ \frac { \sqrt { d } } { \| h _ { b } \| } } \sin ^ { 2 } \theta _ { b } = { \frac { \sqrt { d } \sigma ^ { 2 } } { ( s ^ { 2 } + \sigma ^ { 2 } ) ^ { 3 / 2 } } } \quad { \mathrm { a n d } } \quad { \frac { \sqrt { d } } { \| h _ { b } \| } } \sin \theta _ { b } \cos \theta _ { b } = { \frac { \sqrt { d } \sigma s } { ( s ^ { 2 } + \sigma ^ { 2 } ) ^ { 3 / 2 } } } .\tag{41}
$$

Plugging these into Equation 38 and averaging the margins over the new facts yields Equation 5,

$$
\dot { s } = \alpha \sqrt { d } \frac { \sigma \left( \sigma \bar { \gamma } _ { \mathrm { s h } } - s \bar { \gamma } _ { \mathrm { i n d } } \right) } { \left( s ^ { 2 } + \sigma ^ { 2 } \right) ^ { 3 / 2 } } .\tag{42}
$$

This one-dimensional equation has the following properties, which underlie the discussion of the main text:

– The sign of s˙ is the sign of $\sigma \bar { \gamma } _ { \mathrm { s h } } - s \bar { \gamma } _ { \mathrm { i n d } } . \mathrm { ~ I f ~ } \bar { \gamma } _ { \mathrm { i n d } } < 0$ and $\bar { \gamma } _ { \mathrm { s h } } \geq 0$ , then $\dot { s } > 0$ : the shift can only grow and there is no fixed point.

– The shift reinforces itself while it is small. The weight of the $\bar { \gamma } _ { \mathrm { i n d } }$ term, $s \sigma / ( s ^ { 2 } + \sigma ^ { 2 } ) ^ { 3 / 2 }$ increases with s for $s < \sigma / \sqrt { 2 }$ and decreases beyond. While the new facts are confidently wrong $( \bar { \gamma } _ { \mathrm { i n d } } < 0 )$ and the shift is small, a larger shift therefore makes the shift grow faster.

– Once the new facts are right, the shift is pulled back. If $\begin{array}{c} \bar { \gamma } _ { \mathrm { i n d } } > 0 \end{array}$ , the equation has a unique fixed point $s ^ { * } = \sigma \bar { \gamma } _ { \mathrm { s h } } / \bar { \gamma } _ { \mathrm { i n d } }$ , which is stable since

$$
\frac { \partial \dot { s } } { \partial s } \Big \vert _ { s = s ^ { * } } = - \frac { \alpha \sqrt { d } \sigma \bar { \gamma } _ { \mathrm { i n d } } } { ( s ^ { * 2 } + \sigma ^ { 2 } ) ^ { 3 / 2 } } < 0 .\tag{43}
$$

$\mathrm { \bf A s } \bar { \gamma } _ { \mathrm { s h } }$ quickly becomes small, $s ^ { * }$ is small and s relaxes towards it.

![](images/fc3304e184d35a3b693b28cd625390f1620aa46a9e5bb167814bd19da9efaa95.jpg)

![](images/535c641a86a7990a4e8ff64a010c268f57ea6ad9c30d610121bef73afc188d56.jpg)  
Figure 7: Equation 38 reproduces the dynamics of the common shift. (left) Length $\| \bar { h } _ { A } \|$ of the old facts’ mean hidden state, measured and obtained by integrating Equation 38 (with normalization) or $\dot { s } = \alpha \bar { \gamma } _ { \mathrm { s h } }$ (without normalization) from the point at which the mean state has aligned with the shared direction, after which its length follows the same dynamics as its displacement s. (right) Average shared margin $\bar { \gamma } _ { \mathrm { s h } }$ and individual margin $\bar { \gamma } _ { \mathrm { i n d } }$ of the new facts, with normalization. The dotted line marks the peak; ten seeds.

– The pull stops with the errors. The restoring term is proportional to $\bar { \gamma } _ { \mathrm { i n d } }$ , which vanishes as the new facts are learned. Nothing in the equation brings s back to its initial value: it stops wherever it is when the errors of the new facts vanish.

– Without normalization, nothing pulls the shift back. The equation becomes $\dot { s } = \alpha \bar { \gamma } _ { \mathrm { s h } }$ which contains neither $\bar { \gamma } _ { \mathrm { i n d } }$ nor s.

## B.7 EMPIRICAL VALIDATION

We verify that Equation 38 describes the actual model, without the common-coordinate approximation. We follow the finetuning run, trained with minibatches of 32 new facts, and at every step measure $\theta _ { b } , \| h _ { b } \| , \gamma _ { \mathrm { s h } , b } \mathrm { a n d } \gamma _ { \mathrm { i n d } , b }$ on that step’s minibatch. We evaluate the right-hand side of Equation 38 from these measurements and integrate it with an explicit Euler scheme whose time step is the learning rate η (Assumption 1), starting from the measured value of s once the mean state has aligned with the shared direction. The integration includes the second-order term of a finite step, which lengthens the mean state by $\| \Delta _ { \perp } \| ^ { 2 } / 2 s$ , where $\Delta _ { \perp }$ is the part of the step orthogonal to $\hat { u } _ { \mathrm { s h } }$ and vanishes as $\eta  0$ We proceed in the same way for the model without normalization, with $\dot { s } = \alpha \bar { \gamma } _ { \mathrm { s h } }$ . The prediction matches the measured s to within 1% of its range with normalization and 2% without (Figure 7).

## B.8 THE MECHANISM, STEP BY STEP

Section 3 explains spurious forgetting through a chain of claims about the minimal model. Here we check each of them in the base configuration, over ten seeds; dotted lines mark the bottom of the collapse.

How the shift is written. Equation 4 replaces the update of each old hidden state by a single vector, $- \alpha \bar { \delta } _ { B }$ , shared by all of them. The exact update of every old state indeed points along the mean update, and $- \alpha \bar { \delta } _ { B }$ reproduces it to within 16% until the end of the collapse (Figure 8, left); the fact-specific terms it neglects only grow later, as the erosion sets in. The speed of this write grows almost in proportion to α (middle), and the new facts move along the shared direction exactly as far as the old ones (right), so the shift is built and undone on both together.

Where the shift points. When the new answers are drawn from the entire vocabulary, the old states still move together, nearly as far (Figure 9, left), but the shift no longer favors either half of the vocabulary: its push on the logits of A toward the half of the new answers is ten times smaller (middle). With the new answers in one half, the facts of A lose to that half but keep their order within their own: at the bottom of the collapse, 93% of them still rank their correct answer first within the half of A, against 17% accuracy (right).

What normalization does. As the shift grows, normalization rotates the old states toward it: their mean pairwise cosine rises from 0.06 to 0.54 at the bottom of the collapse (Figure 10, left), and their individual parts lose about a third of their size (middle). Both recover as the shift is withdrawn. Meanwhile the predictions for A move to the half of the new answers, 82% of them at the bottom of the collapse, favoring the answers that are most common among the new facts (right).

The readout of A is barely touched. The readout weights of the answers of A change by about 5% during the collapse and 10% by the end of finetuning, against more than 60% for the half of the new answers (Figure 11, left). They receive gradient only while the model puts probability on these answers for the new facts, which stops within a few dozen steps (right).

What survives and what erodes. Removing the common part of the change of the logits restores A almost entirely at the bottom of the collapse, and slightly less as finetuning proceeds; for B, whose answers are those of the new facts, it changes almost nothing, and B erodes steadily after the collapse (Figure 12, left). The fact-specific drift behind this erosion is not neutral: when a fact of A is recalled wrongly, it names the answer of the new fact whose key is closest to its own twice as often as that of a random new fact (right).

![](images/a27c13cd572e22c2c42b6e168441035cc5d2a81e9f90dc49bbc09b55e44acf98.jpg)

![](images/eff3885fe9fba522d5dd1b0d4f709ac312290ba338ffa7f277499d67a1d6fd17.jpg)

![](images/c8236fbda5a6368542ff10c031c93b14307fe160868e41e80d0dcbaa17eb1c23.jpg)  
Figure 8: How the shift is written. (left) Cosine between the update of each old hidden state and their mean update, and relative error of Equation 4. (middle) Speed of the update at the start of finetuning against α. (right) Displacement of the old and new hidden states along the shared direction.

![](images/3ccbd743fc7f927a72ffbc6e9363eecb5d931f3f30dd97597c9a89af9d6d9843.jpg)

![](images/b6259ed4ee6127bad5ec31ff141a269e99408e388aa007c7c71e7ff3fc5b33a7.jpg)

![](images/2008d1246e6f20c1a5b0b7d89923ae7da49633499067fd378c155b8304ace5b7.jpg)  
Figure 9: The shift favors the half of the new answers. (left) Common shift with the new answers in one half of the vocabulary or anywhere. (middle) How far it pushes the logits of A toward the half of the new answers. (right) Old facts whose correct answer is still ranked first within their own half, against their accuracy.

## B.9 THE OLD FACTS SURVIVE THE COMMON SHIFT

The common shift of the old states is written by the part of the update that acts on the shared key direction, $\Delta W _ { 1 } \mu \mu ^ { \top }$ (Equation 16). At every checkpoint, we remove this part from the store, $\begin{array} { r } { \dot { W _ { 1 } } ( t )  W _ { 1 } ( 0 ) + \dot { \Delta } W _ { \perp } ( \dot { t } ) } \end{array}$ with $\Delta W _ { \perp } = \dot { \Delta W _ { 1 } } ( I - \mu \mu ^ { \top } )$ ), keep the readout as trained, and evaluate the edited model; training itself is not modified. At the trough, the edited model recalls the old facts with accuracy 1.00, against 0.17 for the trained model, and after about 1,000 steps with accuracy 0.91, against 0.55 (Figure 13). The remaining decline is the erosion carried by $\Delta \bar { W _ { \perp } }$ (Section B.11). The edit also slows the new facts, which reach 0.59 at step 200 instead of 0.99, since the shared part of the update also carries what the new facts have in common.

![](images/b9647733acb1efab7f7a6df9e90ed6bd09d7a946f4a9ede7269e0fbdef833512.jpg)

![](images/5a9e35ebf9b064c6a1f4ef204c1202235914afa5d375a1ca9d491d0df4717f2e.jpg)

![](images/e292be48283d6525d95d88c90a306b39de7723617681750403d437421e47d051.jpg)

Figure 10: Normalization rotates the old states toward the shift. (left) Mean cosine between the old hidden states. (middle) Norm of their individual parts after normalization, relative to the start of finetuning. (right) Share of the predictions for A that fall in the half of the new answers, and their correlation with how often each answer occurs among the new facts.  
![](images/7c8ad7c7ffc0273cb34e349ed56527b14f7c227e230a91ea7ce1a5e03c544824.jpg)

![](images/39ca2a1b109e32ffb786a881fafb3be410cf4a3a499aa4f41d65463701f59b09.jpg)  
Figure 11: The readout of the answers of A is barely touched. (left) Relative change of the readout weights of the answers of A and of the half of the new answers. (right) Gradient on the readout weights of the answers of A, and the probability the model puts on these answers for the new facts, both relative to the start of finetuning.

## B.10 THE ANALYSES OF THE TRANSFORMER AND THE LANGUAGE MODEL, IN THE MINIMAL MODEL

Sections 4 and 5 test the mechanism with two analyses that need no access to a model’s internals: splitting the displacement of the old facts and the change of their logits into a common and a factspecific part, and removing the top direction of each weight update. We run both on the minimal model, where the mechanism is known, to check that they detect it.

The split in the minimal model. We measure the old hidden states after the normalization, at the input of the readout, as in the Transformer. Their common shift grows fast, peaks at the bottom of the collapse and then partly retracts (Figure 15, left). The fact-specific drift stays smaller for much longer than in the Transformer, and overtakes the shift only after about $1 0 ^ { 4 }$ steps: once the new facts are learned, both the withdrawal of the shift and the growth of the drift are driven by their vanishing errors, so the remaining gap closes only logarithmically (Appendix B.11). Subtracting the common part of the change of the logits restores the old facts at the bottom of the collapse, from 0.17 to 1.00, while keeping only that part produces most of the collapse (Figure 15, right), as in the Transformer.

Removing the top direction of the update. As for the language model, we subtract from the store $W _ { 1 }$ the top singular component of its update $W _ { 1 } ( t ) - W _ { 1 } \bar { ( } 0 )$ , or, as a control, a random rank-one matrix of the same norm, and keep the readout as trained. At the bottom of the collapse, the edit restores the old facts entirely, whereas the control leaves them where training did (Figure 16, left). The top direction is then the shared one: its input direction is aligned with $\mu$ (cosine 0.99), so the edit removes the part of the update that writes the common shift (Appendix B.9). It also slows the learning of the new facts (right), which use the same direction. Once the shift recedes, the top direction no longer carries it (cosine 0.11 by step 200), and the edit no longer helps. Compared at the same accuracy on the new facts, however, the edited model always recalls more old facts than the trained one, so the edit does not simply rewind finetuning, as in the language model.

![](images/9fc6f7c525427b74956074a028cfc910ce5b40764d818c31cf4610b70521f55a.jpg)

![](images/c137270b68decb312825c793e1d432a36153572ed68f1285879e18819260c932.jpg)

Figure 12: The shift hides $A ;$ the drift erodes A and B. (left) Accuracy of $A$ and B as trained and with the common part of the change of their logits removed. (right) Among the facts of A recalled wrongly, the share that names the answer of the closest new key, against that of a random new key.  
![](images/93861140e4ff620ca058f426fc7497ecfe6d05b022cc679eefe33050d097381b.jpg)

![](images/5816200e918e129733e96ac7f04df50c8e13f2aacf97e769fd9fb2f2765ab910.jpg)  
Figure 13: Removing the shared part of the update restores the old facts. Accuracy on the old (left) and new (right) facts as trained and with $\Delta W _ { 1 } \mu \mu ^ { \top }$ removed from the store, the readout kept as trained. The dotted line marks the trough; ten seeds.

## B.11 FACT-SPECIFIC DISPLACEMENTS

Size of the fact-specific displacements. Recall from Equation 18 that $\varepsilon _ { a } = \sqrt { 1 - \alpha } \Delta W _ { 1 } ( e _ { a } -$ ${ \bar { e } } _ { A } )$ . Under Assumption 3, $\Delta W _ { 1 }$ is fixed with respect to the old keys, and we can take the expectation over them. For a fixed matrix M and a vector u with $\mathbb { E } [ u u ^ { \top } ] = \dot { \Sigma } , \mathbb { E } \| M u \| ^ { 2 } = \mathrm { t r } ( M \Sigma \dot { M } ^ { \top } )$ . The $e _ { a }$ are independent and uniform on the unit sphere, so $\mathbb { E } [ e _ { a } e _ { a } ^ { \top } ] = I / d$ and

$$
\mathbb { E } \left[ ( e _ { a } - \bar { e } _ { A } ) ( e _ { a } - \bar { e } _ { A } ) ^ { \top } \right] = \left[ \left( 1 - \frac { 1 } { n _ { A } } \right) ^ { 2 } + \frac { n _ { A } - 1 } { n _ { A } ^ { 2 } } \right] \frac { I } { d } = \left( 1 - \frac { 1 } { n _ { A } } \right) \frac { I } { d } ,\tag{44}
$$

from which

$$
\mathbb { E } \left[ \varepsilon _ { \mathrm { r m s } } ^ { 2 } \right] = \frac { 1 - \alpha } { d } \left( 1 - \frac { 1 } { n _ { A } } \right) \| \Delta W _ { 1 } \| _ { F } ^ { 2 } .\tag{45}
$$

Each old fact reads the update through its own random key, and a random direction captures, on average, a fraction $1 / d$ of its squared Frobenius norm.

The shared and non-shared parts of the update. We split the update into its column along $\mu$ and the rest, $\Delta W _ { 1 } = \Delta W _ { \perp } \bar { + } \left( \Delta W _ { 1 } \mu \right) \mu ^ { \top }$ with $\Delta W _ { \perp } : = \Delta W _ { 1 } ( { \bar { I } } - \mu \mu ^ { \top } )$ . The two parts are orthogonal for the Frobenius inner product, so $\| \Delta W _ { 1 } \| _ { F } ^ { 2 } = \| \Delta W _ { \perp } \| _ { F } ^ { 2 } + \| \dot { \Delta } W _ { 1 } \mu \| ^ { 2 }$ , and, neglecting the $1 / n _ { A }$ correction,

$$
\varepsilon _ { \mathrm { r m s } } ^ { 2 } \simeq \frac { 1 - \alpha } { d } \left( \| \Delta W _ { \perp } \| _ { F } ^ { 2 } + \| \Delta W _ { 1 } \mu \| ^ { 2 } \right) .\tag{46}
$$

![](images/e16335d90792ee966500f151571ddac076e477a0b76e7b4d142e0e5ee7f7d856.jpg)

![](images/8983dbb6f984aea800cbd2490e430a18f198e3757560d069362d2d19504261b2.jpg)  
Figure 14: The spread of the old facts is set by the Frobenius norm of the update. (left) Measured spread $\varepsilon _ { \mathrm { r m s } }$ of the old hidden states during finetuning, compared with $\sqrt { ( 1 - \alpha ) / d } \parallel \Delta W _ { 1 } \parallel _ { F }$ and with $\sqrt { ( 1 - \alpha ) / d } \parallel \Delta W _ { \perp } \parallel _ { F }$ (Equation 47). (right) Measured against predicted; the prediction runs slightly low only while the shared column of the update is large. Ten seeds.

![](images/59f60a409a3acc45226579261deab52dc58dd7893e5b4236d856e19acbdca167.jpg)

![](images/d3bf4fe4e9446b3b90f09a005217a604d1e084a241ebaa9da6fc542462a45dbf.jpg)  
Figure 15: In the minimal model, removing the common shift removes the collapse. (left) Common shift and fact-specific drift of the old hidden states after normalization. (right) Old-fact accuracy as trained, with only the common part of the change of their logits, and with it removed; the dotted line marks the bottom of the collapse.

The second term is the shared column of the update, read through the small random component $\mu ^ { \mathrm { ~ l ~ } } e _ { a }$ of each old key. It only matters while the shared write makes up a large part of the update, that is, around the peak of the common shift. Dropping it gives

$$
\varepsilon _ { \mathrm { r m s } } \simeq \sqrt { \frac { 1 - \alpha } { d } } \lVert \Delta W _ { \perp } \rVert _ { F } .\tag{47}
$$

Figure 14 compares the measured spread with both predictions.

Why the fact-specific displacements are not undone. Subtracting the mean write from Equation 21, the fact-specific displacements evolve as

$$
\dot { \varepsilon } _ { a } = - \frac { 1 } { n _ { B } } \sum _ { b } k _ { b } ^ { \top } \big ( k _ { a } - \bar { k } _ { A } \big ) \delta _ { b } = - \frac { \sqrt { 1 - \alpha } } { n _ { B } } \sum _ { b } k _ { b } ^ { \top } \big ( e _ { a } - \bar { e } _ { A } \big ) \delta _ { b } .\tag{48}
$$

The coefficients are random, of order $d ^ { - 1 / 2 }$ , and independent of the old fact’s own state and answer. Unlike Equation 38, this expression contains no term that depends on $\varepsilon _ { a }$ itself and could pull it back. The contribution of the normalization to $\delta _ { b }$ is present as well, but it enters with the same random coefficients. The fact-specific displacements therefore keep accumulating as long as the new facts produce errors.

## B.12 WHY RECOVERY IS INCOMPLETE

Two effects keep the old facts from recovering fully. The first follows from Section B.5: the force that withdraws the common shift is proportional to the margins of the new facts and vanishes with their errors. The second comes from the readout, which we have ignored so far.

![](images/ff86759b2ee937a8643d9abafe5f1495b7232eabb45d29048a54f4d5ec915e57.jpg)

![](images/5556477f32c6afb3a5aa53b4ba2ff190298d134700dde09959b51e4d05cbb862.jpg)

Figure 16: Removing the top direction of the update in the minimal model. Accuracy on the old (left) and new (right) facts as trained, with the top singular component of the update of $W _ { 1 }$ removed, and, for the old facts, with a random rank-one matrix of the same norm removed.  
![](images/c5b1655c1c56339cf58f8a6e119c3cb1840b94831b81b2c6608404d85004451f.jpg)

![](images/035d3d9ec08cdea423616ecd51117b070a3094647f6480836e1c8112e06b6b01.jpg)  
Figure 17: Part of the bias towards the new answer region is stored in the readout. Accuracy on the old and new facts during finetuning, with the readout trained and frozen; the learning rate of the frozen run is matched so that the new facts are learned at the same pace. Ten seeds.

Under the gradient flow dynamics, Equation 15 implies that, for a fixed normalized old state $x _ { a } .$ , the logits change as

$$
\dot { z } _ { a } = \dot { W } _ { 2 } x _ { a } = \frac { 1 } { n _ { B } } \sum _ { b } \left( x _ { b } ^ { \top } x _ { a } \right) \left( \mathbf { 1 } _ { y _ { b } } - p _ { b } \right) .\tag{49}
$$

This has the same structure as Equation 21, with the overlap between normalized states in place of the overlap between keys. The normalized states share the direction of the common shift, so the overlaps $\bar { x } _ { b } ^ { \top } x _ { a }$ have a common positive part. The readout therefore raises the logits of the new answers for every old fact at once, and lowers those of the answers the model currently predicts for the new facts. Unlike the write into $W _ { 1 }$ , this update has no normalization term: nothing in it depends on the margins of the new facts or on what the readout has already written. It stops when $p _ { b }  \mathbf { 1 } _ { y _ { b } }$ but never reverses, so the bias towards the new answer region that it stores remains after the new facts are learned. Freezing the readout during finetuning removes this channel and raises the recovered accuracy of the old facts from 0.56 to 0.61 (Figure 17).

## B.13 INTERVENTIONS IN THE MINIMAL MODEL

Projecting out the shared direction. Removing the component of the update along the shared key direction at every training step, $\Delta W _ { 1 } \gets \Delta W _ { 1 } \big ( I - \mu \mu ^ { \top } \big )$ , prevents the collapse entirely, and the new facts are still learned. The old facts nevertheless end no better off than without the projection (accuracy 0.45 against 0.51 after 5,000 steps, ten seeds), because erosion lies in the fact-specific part of the update, which the projection leaves untouched.

Replaying a few old facts. Rehearsing a subset of old associations can restore the others (Hinton & Plaut, 1987). We replay a fixed random subset of the old facts in every finetuning batch of 32 new facts and measure the old facts that are never replayed (Figure 18). Replaying 16 of the 128 old facts lifts their trough from 0.17 to 0.50, as the replayed facts push back on the shift that all old facts share, but raises their final accuracy only from 0.54 to 0.62: erosion is specific to each fact and is barely affected by replaying others. A single replayed fact has almost no effect, as it is soon answered correctly through its own key and stops pushing back on the shared direction.

![](images/1b659389d976334331b5ec4be214c42aff796df795d82cf5818e7c170ac5964c.jpg)

![](images/3867d59fd276e1a83fc2920fe3c3eff614c877bd8060d2f7aaf6180ba708fca6.jpg)  
Figure 18: Replaying a few old facts protects the others from the collapse, but not from erosion. (left) Accuracy of the old facts that are never replayed, when 0, 4, 16 or 64 of the 128 old facts are replayed in every batch (16 replayed examples per step; five seeds). (right) Gain over no replay at the trough and at the end of finetuning (3,000 steps), with 16 or 4 replayed examples per step.

## C THE CONTROLLED TRANSFORMER

## C.1 THE OLD FACTS KEEP THEIR RANKING WITHIN THEIR REGION

For each old fact of A, we rank its correct value among the values of its own half of the answers, and report the fraction of old facts for which it ranks first, alongside their accuracy over all answers (Figure 19). At the bottom of the collapse, only 12% of the old facts are recalled, yet 86% still rank their correct answer first within their half: the old facts lose to the answers of the other half, not to one another. The fraction then declines only slowly, to 83% after 500 steps, as the fact-specific drift erodes the old facts.

![](images/0d6b064cb892e0c9b254cde9eaad92db3d3143d85028d6d6c0e7ee79c76003b9.jpg)  
Figure 19: The collapse leaves the old facts’ ranking within their region intact. Accuracy of the old facts of A over all answers, and fraction of them whose correct answer still ranks first among the answers of its own half, for the transformer of Figure 1 (right; three seeds). The dotted line marks the trough.

## C.2 TIMING ACROSS LEARNING RATES AND BATCH SIZES

We finetune the transformer of Figure 1 under five conditions that change how fast it learns: the reference learning rate and batch size of 256, half and twice the learning rate, and batches of 128 and 512 at the reference learning rate. Each condition is run with three seeds, and both old and new facts are evaluated every 2 steps on held-out phrasings. We define the trough as the point of lowest old-fact accuracy before the recovery.

Across the five conditions, the trough moves from step 34 to step 100, a factor of about three in training steps. At every trough, however, the new facts have reached nearly the same stage of learning, between 36% and 44% accuracy (Figure 20). The end of the collapse therefore follows the learning of the new facts rather than a fixed amount of optimization, as the minimal model predicts. This holds for a fixed set of new facts; with fewer of them, the trough falls later in their learning (Appendix C.3).

![](images/da4f9d4e2411b32754c9420fe14cfcc79e2bfdd484288448299eb4b0b9010157.jpg)

![](images/02b7d12f6bb8efddeb19195b60c64568eacbbd9d4182bf3d00296f31b561b320.jpg)  
Figure 20: The end of the collapse follows the learning of the new facts. Old-fact (left) and newfact (right) accuracy under five finetuning conditions; circles and dotted lines mark each condition’s trough. The troughs span steps 34 to 100, yet the new facts are at 36–44% accuracy at every one of them. Three seeds, evaluated every 2 steps.

![](images/3019e282d6d24b8a1b865982ba25c972b919fd911ce0bf6dbaeb9520409c2879.jpg)

![](images/b88d24c3669f97754bbc4c52f4a1c85de722c32d16e8afa778aab889b222c852.jpg)  
Figure 21: With fewer new facts, the collapse ends later in their learning. Old-fact (left) and new-fact (right) accuracy of the Transformer finetuned on 125, 250 or 500 new individuals; circles and dotted lines mark the bottom of each collapse. Three seeds.

![](images/c42ae7740f47ed3c270c2ec83b32a44e18dfa1991c7b5f9b989727a09599eaf5.jpg)

![](images/91d876ff38201bec79fdc0be02d4d8e1b058252c381ee34dc4ed641f64b4daa2.jpg)

![](images/49103a6ad1c8067bbd62823a388bbe4ae168eff9a31607a0fc3a13add4748de8.jpg)  
Figure 22: In the minimal model, the shift peaks when the individual margin turns positive. (left) Common shift over finetuning with 32, 128 and 256 new facts, scaled to its peak; circles mark where $\bar { \gamma } _ { \mathrm { i n d } }$ turns positive. (middle) The step of the peak against that step, for different numbers of new facts (filled) and for different learning rates and batch sizes (open). (right) Accuracy on the new facts at that step: it depends on their number, but not on the learning rate or the batch size. Ten seeds.

## C.3 TIMING ACROSS THE NUMBER OF NEW FACTS

Varying the learning rate or the batch size changes how fast the new facts are learned, but not how many there are. To vary the latter, we finetune on only the first 125 or 250 of the 500 individuals of $\bar { B ^ { \prime } }$ , keeping everything else as in the reference condition of Appendix C.2, and evaluate every 2 steps.

The bottom of the collapse then no longer falls at the same accuracy on the new facts (Figure 21). The fewer the new facts, the further they are learned before the old facts start to return: 89% of them are recalled at the bottom with 125 individuals, against 38% with 500. The collapse also becomes shallower, and in the Transformer with MLP blocks (Appendix C.5) it nearly disappears, with old-fact accuracy staying above 0.9 for 125 or 250 individuals.

The minimal model shows why (Figure 22). Whatever we vary, the learning rate, the batch size or the number of new facts, the common shift peaks when the individual margin $\bar { \gamma } _ { \mathrm { i n d } }$ of the new facts turns positive, as Equation 5 predicts. How well the new facts are recalled at that moment depends on their number but not on the learning rate or the batch size. A fixed accuracy on the new facts is therefore only a stand-in for the individual margin, and it holds only as long as the number of new facts is fixed.

## C.4 LOCALIZING THE COMMON SHIFT

To find which weights carry the common shift, we take the checkpoint at the end of the collapse (step 50), restore one group of weights to its value before finetuning, and recompute the final hidden states of the old facts. We measure the common shift $\Delta \bar { h }$ of Section 4 with and without the restoration, and report the norm of the difference as a fraction of $\| \Delta { \bar { h } } \|$ , averaged over the six attributes. The groups partition all parameters, so their shares can be compared directly; each is measured with only that group restored, so they need not add up to 100%. Nothing is retrained, so the learning of the new facts is unaffected by construction.

<table><tr><td>Weights restored</td><td>Share of the common shift</td></tr><tr><td>Attention value and output</td><td>84%</td></tr><tr><td>Attention query and key</td><td>11%</td></tr><tr><td>Input embedding</td><td>2%</td></tr><tr><td>LayerNorms</td><td>1%</td></tr><tr><td>Unembedding (control)</td><td>0%</td></tr></table>

Table 2: Share of the common shift removed by restoring each group of weights to its value before finetuning, at the trough (step 50; three seeds, standard deviation below 0.3 percentage points).

![](images/25724351a2fa273f6cd30fadb2748476c38009ce5814b35eb7a03e44866a5f7b.jpg)

![](images/98e4e28929bb9af074da84784d137a91e5d005f8091f7fbd2eeda9c11fb8e645.jpg)  
Figure 23: The Transformer with MLP blocks reproduces Figure 1. Accuracy of the old and new facts when finetuning on C (left) and on $B ^ { \prime }$ (right).

The attention value and output weights carry 84% of the shift, and the query and key weights 11% (Table 2); after the recovery (step 200), the value and output weights still carry 85%. The unembedding is applied after the point at which the shift is measured and serves as a control: restoring it leaves the shift unchanged, as it must.

## C.5 TRANSFORMER WITH MLP BLOCKS

We repeat every analysis of Section 4 with a Transformer that has MLP blocks of width 4d and is otherwise identical: same data, same size, same pretraining and finetuning. The picture is the same, with one difference that matters: the fact-specific drift takes a larger part in the collapse.

Collapse, recovery and erosion. Finetuning on $C$ lowers recall of the old facts only slowly, to 0.94 after 1,000 steps (Figure 23, left). Finetuning on $B ^ { \prime }$ makes the old facts of A collapse to 0.53 at step $^ { 3 0 , }$ recover to 0.93 and then erode to 0.75, while those of $B ,$ whose answers the new facts share, barely move during the collapse (right).

Common shift and fact-specific drift. As in Figure 4, the common shift grows fast, peaks at the bottom of the collapse and then partly retracts, while the fact-specific drift keeps growing and overtakes it after about 90 steps (Figure 24, left). Subtracting the common part of the change of the logits removes the collapse, raising old-fact accuracy at the bottom from 0.53 to 0.99 (middle), and the old facts keep their order within their own half: 97% of them still rank their correct answer first within it (right).

The common shift turns the drift into a collapse. Keeping only the common part lowers recall less here than in the attention-only Transformer (Figure $2 \dot { 4 }$ , middle), while removing it still restores the old facts. Both models therefore collapse in the way Section 4 describes: the common shift brings the old facts close to losing to the new answer region, and the fact-specific drift pushes them across. The two differ only in how the work is shared. Splitting the margin of each old fact recalled wrongly, its correct logit minus that of its strongest competitor, into its pretrained value and the contributions of the common and fact-specific parts (Figure 25), the common part alone almost cancels the pretrained margin in the minimal model and in the attention-only Transformer, but leaves a larger part of it with MLP blocks, where the fact-specific drift is larger at the bottom of the collapse. Without the shift, the drift alone collapses neither model.

![](images/88a57f947269fe77263376b33872682cbd1356d6793a5459dc1d5023f8a90999.jpg)

![](images/00634d5958a0f900f078c93d7aae453cbff462a5bf427aafa4cb3074b07f5244.jpg)

![](images/d3ee2af5407018a78cbc1ec81add5e6cc3cd22a249bf17779ab88196c715235f.jpg)  
Figure 24: With MLP blocks, removing the common shift still removes the collapse. (left) Common shift and fact-specific drift of the old hidden states. (middle) Old-fact accuracy as trained, with only the common part of the change of their logits, and with it removed. (right) Old facts whose correct answer still ranks first within their own half, against their accuracy. Three seeds; the dotted line marks the bottom of the collapse.

![](images/62bc2bc6f2ac4265fba530a00a9a98a5daf4b9d3a76405e1d037a4d4b27932a3.jpg)

![](images/d384eb2a8a4fc337c516a5c1d6582e93eaf4d8a191d3a2840f51f26bc074ea32.jpg)

![](images/0aa623de78b5e9c994aeb692df0695b5c536836da9c5ad4a628e4a5d2bd56d4e.jpg)  
Figure 25: What turns an old fact wrong. For the old facts recalled wrongly, the margin of the correct answer over its strongest competitor, split into its pretrained value and the contributions of the common and fact-specific parts of the change of the logits.

![](images/18ba53fd9652f6e787a9a94371fef057273b28add42c318b7929a1bfb2137479.jpg)

![](images/628e3c5d7882d0ed1dc4ff14fce37f5e82e92e1ad5c6f22137e08d7c4176fe42.jpg)  
Figure 26: Timing in the Transformer with MLP blocks. Old-fact (left) and new-fact (right) accuracy under five learning rates and batch sizes; circles and dotted lines mark the bottom of each collapse. Three seeds evaluated at every 2 steps

Where the shift is written. Restoring groups of weights as in Appendix C.4, the MLP weights and the attention value and output weights each carry about half of the common shift at the bottom of the collapse (46% and 47%), the query and key weights 3%, and the embedding and LayerNorms less than 2% each. Both MLP and attention value and output weights write content into the residual stream, so the shift is again written into the associative memories of the network rather than into where attention goes. Adding MLP blocks does not move the shift to a new place, but spreads it over a second memory: the update that all new facts share is written wherever the network stores associations, in proportion to how much each store takes part in learning them. The query and key weights, which decide where attention goes, contribute little during the collapse and somewhat more after the recovery, as in the attention-only Transformer, where their share also grows once the factspecific drift dominates. The collapse is thus a change in what the network writes about each fact, not in which tokens it reads.

Timing. Changing the learning rate and the batch size as in Appendix C.2 moves the bottom of the collapse from step 20 to step 56, while the new facts are recalled with 23 to 33% accuracy at every bottom (Figure 26). As in the attention-only Transformer, the end of the collapse follows the learning of the new facts rather than the number of steps: a faster optimizer reaches the bottom sooner, but at almost the same point in learning the new facts. This point is not shared across architectures, however: with MLP blocks, the collapse ends when fewer new facts are recalled than without them. What the two models have in common is therefore not a threshold on new-fact accuracy, but the mechanism that sets it, the individual margin of the new facts turning positive, whose relation to accuracy depends on the model and, as Appendix C.3 shows, on the number of new facts. The depth of the collapse, in contrast, depends on the learning rate, more strongly than without MLP blocks: halving it leaves only a shallow collapse, consistent with Section 4, where the depth, unlike the timing, lies outside our gradient-flow analysis.

## D THE PRETRAINED LANGUAGE MODEL

## D.1 WHY RECALL OF THE OLD FACTS FIRST RISES.

In the first 25 to 30 steps of finetuning, recall of the old facts rises from 0.87 to about 0.93 with both kinds of new facts, although none of them is replayed. Before finetuning, the model answers 54 of the 422 old facts wrongly, but for 47 of them it already ranks the correct answer first among the possible answers of the relation, and loses only to a continuation that is not an answer. Every old fact that becomes correct during the rise is one of these 47 (26 with synthetic individuals, 32 with real entities), and the rank of the correct answer among the possible answers does not change. In every finetuning document, the answer immediately follows the relation, so finetuning raises answers as a whole above other continuations without changing which answer the model prefers. The collapse sets in once the shift toward the new answers dominates.

## D.2 REMOVING ONE DIRECTION OF THE UPDATE

The edit. At each evaluation step of a finetuning run, we compute the update $\Delta W = W ( t ) - W ( 0 )$ of every weight matrix of the transformer blocks of OLMo 2 1B: the query, key, value and output matrices of attention and the three matrices of the MLP, in each of the 16 layers, 112 matrices in total. The embedding and the unembedding are left untouched. We remove the top singular direction of each update, $W ( \bar { t } ) \gets W ( t ) - \sigma _ { 1 } u _ { 1 } v _ { 1 } ^ { \top }$ , evaluate the edited model, and then restore the original weights before training continues, so that the finetuning run itself is never modified. The removed direction is small: it holds about 15% of the squared norm of a typical update. As a control, we instead remove a random rank-one matrix of the same norm from each update.

Effect on the old facts. Removing the top direction keeps the old facts close to their accuracy before finetuning through the collapse, for both kinds of new facts, while the random control leaves them exactly where training left them (Table 3). The edit also keeps the loss on a fixed sample of ordinary text (64 sequences of 512 tokens) close to its value before finetuning, whereas finetuning alone raises it.

The edit is not a return to an earlier point of training. The edit lowers the accuracy on the new facts, which raises the question of whether it simply moves the model back along its training trajectory. To test this, we compare the edited model with the trained model at the same accuracy on the new facts: for each edited checkpoint, we find the step at which training first reached the same new-fact accuracy, interpolating between evaluations, and read the old-fact accuracy there. For real entities, at new-fact accuracies of 21%, 34% and 66%, the edited model recalls 88%, 88% and 79% of the old facts, against 81%, 73% and 72% for the trained model: no point of the training run combines the edited model’s old-fact and new-fact accuracies. The reduction in new-fact accuracy indicates that the removed direction also carries part of what the new facts share.

<table><tr><td></td><td></td><td colspan="3">Old facts</td><td colspan="2">New facts</td></tr><tr><td>New facts</td><td>Step</td><td>trained</td><td>top removed</td><td>random removed</td><td>trained</td><td>top removed</td></tr><tr><td>Synthetic individuals</td><td>80</td><td>0.28</td><td>0.77</td><td>0.28</td><td>0.03</td><td>0.01</td></tr><tr><td></td><td>400</td><td>0.38</td><td>0.70</td><td>0.38</td><td>0.90</td><td>0.40</td></tr><tr><td>Real entities</td><td>80</td><td>0.73</td><td>0.88</td><td>0.73</td><td>0.46</td><td>0.25</td></tr><tr><td></td><td>400</td><td>0.55</td><td>0.79</td><td>0.55</td><td>1.00</td><td>0.66</td></tr></table>

Table 3: Accuracy of OLMo 2 1B at the trough (step 80) and after 400 steps of finetuning, with the top or a random direction of each weight update removed. Before finetuning, the old facts are recalled with accuracy 0.87. Small finetuning set for the synthetic individuals; three seeds.

## D.3 THE SAME EDIT IN THE CONTROLLED TRANSFORMERS

We apply the edit of Section 5 to the two Transformers of Section 4: at each checkpoint, we remove the top singular component of the update of every weight matrix in the Transformer blocks (attention and, where present, MLP), leave the embedding and the unembedding untouched, and evaluate the edited model; as a control, we remove a random rank-one matrix of the same norm instead. The removed direction holds 8% of the squared norm of a typical update in the attention-only Transformer and 14% with MLP blocks, close to the 15% of the language model.

Effect on the old facts. At the bottom of the collapse, the edit raises old-fact accuracy from 0.12 to 0.57 in the attention-only Transformer and from 0.54 to 1.00 with MLP blocks, whereas the control leaves it exactly where training did (Figure 27, left and middle). The edited model recalls more old facts than the trained one throughout finetuning, still 0.37 against 0.29 and 0.88 against 0.73 after 1,200 steps. As in the language model, the edit also slows the learning of the new facts in the attention-only Transformer, as the removed direction carries part of what they share; with MLP blocks it barely affects them.

Why the top direction carries the shift. The common shift is what all new facts write coherently into the old hidden states, so while it is being written it should dominate the update. We measure how the edit changes the readout states of the old facts and compare this change with the common shift (Figure 27, right). At the bottom of the collapse the two are aligned, with cosine 0.75 in the attention-only Transformer and 0.97 with MLP blocks. As the fact-specific drift grows, the top direction aligns less and less with the shift, down to 0.22 and 0.49 after 1,200 steps. In the attentiononly Transformer, the edit removes only about a quarter of the shift at the bottom of the collapse and yet restores half of the old facts: as Appendix C.5 shows, the common part alone nearly cancel their pretrained margin in this model, so removing part of it is enough to bring many of them back.

The edit is not a return to an earlier point of training. As in Appendix D.2, we compare the edited model with the trained model at the same accuracy on the new facts. While the new facts are being learned, the edited model recalls more old facts at every new-fact accuracy: at step 100, 0.60 against 0.21 in the attention-only Transformer and 1.00 against 0.90 with MLP blocks. Once all new facts are recalled, the comparison loses its meaning, as every later checkpoint is matched to the first step at which training reached full accuracy on the new facts.

![](images/5f99d7972b3073e2a4999d4fe3c8807ce8a12c5dd3a193748d1c724141ba95df.jpg)

![](images/8e390e42c0b1e62351dc5be33ccc3764591eceff9360a0761c5ee72f2e7fa388.jpg)

![](images/5b9d74aa757cd9a008f47af9dcca2b146d9116bee20e6924045dc4251380fb9e.jpg)  
Figure 27: Removing the top direction of the update in the controlled Transformers. (left, middle) Accuracy on the old (solid) and new (dotted) facts as trained, with the top singular component of every block-weight update removed, and with a random rank-one matrix of the same norm removed. (right) Cosine between the change the edit makes to the readout states of the old facts and their common shift. Three seeds.