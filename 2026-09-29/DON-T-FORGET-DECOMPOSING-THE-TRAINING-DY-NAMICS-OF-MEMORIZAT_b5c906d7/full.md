# DON’T FORGET! DECOMPOSING THE TRAINING DY-NAMICS OF MEMORIZATION IN LANGUAGE MODELS

Florian Eichin<sup>1</sup> Philipp Mondorf<sup>1</sup> Andrei Mircea<sup>2</sup> Yupei Du<sup>3</sup> Barbara Plank<sup>1</sup> Michael A. Hedderich<sup>1</sup>

<sup>1</sup>MaiNLP, Center for Information and Language Processing, LMU Munich & Munich Center for Machine Learning (MCML), Germany <sup>2</sup>Mila – Quebec AI Institute & University of Montreal, Canada <sup>3</sup>Saarland University, Germany Corresponcence to feichin@cis.lmu.de

## ABSTRACT

Memorization has been proposed as a mechanism to explain how language models fit the tail of their training distributions, but its training dynamics are not understood well. In this work, we take a fine-grained look at memorization by decomposing the loss trajectory of memorized sequences over training and model parameters. Across the Pythia family, we study memorization of duplicated training sequences (recitation) and rare ones (recollection). We find that memorization in both cases is characterized by sequence-level gradient alignment, though recitation suffers from misalignment with other training influences which causes forgetting, explaining the necessity for higher duplication of these examples. We further show that the lower model layers are the most involved in memorization and forgetting. Predicting memorization, our decomposition improves over a cross-entropy baseline, especially in larger models and early in training. Intervening on a small set of highly influential parameters we are able to ablate memorization in the final model. Together, these findings advance our understanding of how memorization develops during training and offer insights for predicting and intervening on it.

## 1 INTRODUCTION

Language models can reproduce training sequences verbatim, a behavior known as memorization (Feldman, 2020). Despite a growing body of work on this phenomenon, our understanding of how and where memorization develops in the model during training remains limited. This restricts our ability to predict and intervene on memorization, which is desirable for various applications, for instance where privacy or data security are of concern (Biderman et al., 2026). In this study, we look at memorization as a training dynamic. In particular, we ask which learning contributions along the dimensions of parameters, data, and training process characterize memorization and whether such understanding is actionable for prediction and intervention.

We extend the recently proposed ExPLAIND decomposition (Eichin et al., 2026), to map a model’s loss trajectory onto its influences along training, parameters, and data. From this decomposition, we derive an intuitive mathematical interpretation of how learning mechanisms like memorization are the result of three aspects: The alignment of a prediction’s gradients to the model parameter update, the magnitude of the training updates, and the sensitivity of a prediction to such updates. We apply ExPLAIND across the pretraining trajectories of the Pythia model family, where we examine recitation (memorization of highly frequent training sequences) and recollection (memorization of rare training sequences). Previous work has shown recitation to be the dominant mode of verbatim memorization (Prashanth et al., 2025), but it is unclear why their memorization requires duplication, while recollected sequences do not.

We study the training dynamics of memorization in Pythia and show that a large fraction of memorization happens in very early pretraining before basic generalization has manifested. Our decomposition suggests two aspects of memorization: a sequence’s alignment with parameter updates from its own occurrences in training as well as its interaction with the remaining training influences. Memorized sequences exhibit high gradient alignment with their own occurrences, particularly in lower model layers. Highly duplicated, recited sequences tend to exhibit stronger negative alignment with both other training data and weight-decay updates throughout training, suggesting that repeated exposure helps avoid forgetting. On the other hand, recollected sequences with few training occurrences exhibit less negative alignment, consistent with their memorization despite low duplication.

To validate our understanding, we test these explanations through early prediction and post-hoc intervention. Features derived from the decomposition improve memorization prediction over a cross-entropy baseline, particularly in larger models and early in training. Interventions on a small set of parameters identified as influential early in pretraining, mostly in lower attention and MLP layers, ablate memorization in the final model.

## 2 RELATED WORK

Memorization has been proposed as a mechanism for deep learning models to fit long-tail distributions (Arpit et al., 2017; Chatterjee, 2018; Feldman, 2020), and has been validated and studied in various works (Feldman & Zhang, 2020; Zheng & Jiang, 2022). Memorization in Pythia has been studied by Biderman et al. (2023a), who find that memorization is predictable during pretraining and scales with model size. Further, Prashanth et al. (2025) study Pythia’s memorization through the lens of training corpus statistics and further argue for a taxonomy for memorization based on duplication, which has been shown to be one factor of memorization Lee et al. (2022). We follow their approach and investigate memorization with high (recitation) and low duplication (recollection) but we predict memorization in much earlier training phases and with features derived from our loss decomposition. Tirumala et al. (2022) also study the training dynamics of memorization and find larger models to memorize faster, which our results contradict, and forget less, which we also find and explain.

We also localize memorization, i.e., find the model parameters or layers implementing recall of memories, which has been another popular focus (Meng et al., 2022; Maini et al., 2023; Jia et al., 2025; Ortu et al., 2024), though the faithfulness of existing methods has been questioned (Chang et al., 2024). Previous work has argued that both MLP layers (Geva et al., 2021; Dai et al., 2022) and attention heads (Stoehr et al., 2024) are involved in the mechanisms for recall, which our results confirm. Further, Mireshghallah et al. (2022) identifies the model embeddings as an important factor for recall. Memorization has also been contrasted with generalization by various works (Zheng & Jiang, 2022; Dankers & Titov, 2024), which show that memorization and generalization are entangled (Du et al., 2025; Dankers, 2026). Our results also indicate an interdependence of generalization and memorization, with both emerging roughly at the same time in early pretraining.

As regards methodology, gradient-based data attribution relates training examples to the evaluation loss (Koh & Liang, 2017; Pruthi et al., 2020), while loss change allocation maps it across parameters and training (Lan et al., 2019; Kangaslahti et al., 2025). Our study is based on Eichin et al. (2026)’s decomposition, which builds on work by Bell et al. (2023), and is the only unified and exact approach combining both. We extend it to an explicit notion of alignment between train influences and the prediction. Further, we validate our findings using prediction and causal intervention over training.

## 3 METHODOLOGY

Our study is based on the ExPLAIND decomposition of gradient descent models by Eichin et al. (2026). Using their central Theorem, we decompose a model’s loss into per-sample and per-parameter contributions across training. We extend it by further decomposing their influence scores into parts that can be interpreted as the sensitivity of the prediction, the magnitude of the parameter update, and the alignment of a parameter update with the prediction’s gradient. We apply their Theorem 3.1 and Corollary 3.3 to the evaluation loss and state the accumulated result below (full proof in App. A).

Let $f _ { \theta }$ be a model with parameters $\boldsymbol { \theta } \in \mathbb { R } ^ { D }$ and per-sample loss L. For a sequence $x ,$ define the sequence loss $\begin{array} { r } { \ell ( \theta ; x ) : = \bar { N } ^ { - 1 } \sum _ { x : \epsilon { x } } L ( f _ { \theta } ( x _ { < i } ) , \bar { x } _ { i } ) } \end{array}$ for some $N \in \mathbb { N } ,$ , used for both training and evaluation. At step s, let $\theta _ { s }  \theta _ { s + 1 } ^ { - }$ be the AdamW update with training batch $\boldsymbol { B } _ { s } ,$ learning rate $\alpha _ { s } > 0$ , weight decay $\lambda \ge 0 , \beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ , gradient clipping factor $c _ { s }$ , and $\epsilon > 0$ . Let $m _ { s }$ be the first moment and $\hat { v } _ { s + 1 }$ the bias-corrected second moment. Then, the loss change decomposes exactly

$$
\begin{array} { r } { \ell ( \theta _ { s + 1 } ; x ) - \ell ( \theta _ { s } ; x ) = \sum _ { z \in B _ { s } }  { \mathbb { Z } } _ { s } ^ { z } ( x ) +  { \mathbb { Z } } _ { s } ^ { \mathrm { f m } } ( x ) +  { \mathbb { Z } } _ { s } ^ { \mathrm { r e g } } ( x ) , } \end{array}\tag{1}
$$

where, for each training sequence or update component $\rho \in B _ { s } \cup \{ \mathrm { f m , r e g } \}$

$$
\begin{array} { r } { \mathcal { T } _ { s } ^ { \rho } ( x ) : = - \Phi _ { s } ( x ) ^ { \top } u _ { s } ^ { \rho } , } \end{array}\tag{2}
$$

and we define the path-integrated test gradient and update terms with $z \in B _ { s }$ <sub>s</sub> as

$$
\Phi _ { s } ( x ) : = \int _ { 0 } ^ { 1 } \nabla _ { \theta } \ell ( \theta _ { s } ( t ) ; x ) d t , \mathrm { ~ w h e r e ~ } \theta _ { s } ( t ) : = \theta _ { s } + t ( \theta _ { s + 1 } - \theta _ { s } ) ,\tag{3}
$$

$$
u _ { s } ^ { z } : = \frac { \alpha _ { s } ( 1 - \beta _ { 1 } ) c _ { s } } { 1 - \beta _ { 1 } ^ { s + 1 } } \frac { \nabla _ { \theta } \ell ( \theta _ { s } ; z ) } { \sqrt { \hat { v } _ { s + 1 } } + \epsilon } , \quad u _ { s } ^ { \mathrm { f m } } : = \frac { \alpha _ { s } \beta _ { 1 } } { 1 - \beta _ { 1 } ^ { s + 1 } } \frac { m _ { s } } { \sqrt { \hat { v } _ { s + 1 } } + \epsilon } , \quad u _ { s } ^ { \mathrm { r e g } } : = \alpha _ { s } \lambda \theta _ { s } .\tag{4}
$$

All gradients and update components $u _ { s } ^ { \rho }$ are column vectors in $\mathbb { R } ^ { D } \mathrm { ; }$ ; the square root and division are coordinatewise. The vectors $u _ { s } ^ { \rho }$ are the update components subtracted from the parameters in a single AdamW parameter update. Intuitively, the influence terms $\mathcal { T } _ { s } ^ { \rho }$ decompose the training dynamics of a model into interpretable influences that shape the parameter trajectory of the model. For each example z, we have $\bar { I _ { s } ^ { z } } ( x )$ ), for the first moment and weight decay terms, we have $\mathcal { T } _ { s } ^ { \mathrm { f m } } ( x )$ and $\mathcal { T } _ { s } ^ { \mathrm { r e g } } ( x )$ all quantifying the respective loss changes on x they introduce in training.

Note that each influence term $\mathcal { T } _ { s } ^ { \rho }$ in Eq. 2 is an inner product between the path-integrated test gradient $\Phi _ { s } ( x )$ and a component of the parameter update $u _ { s } ^ { \rho } .$ . Writing the inner product in polar form,

$$
\begin{array} { r } { \mathcal { T } _ { s } ^ { \rho } ( x ) = - \| \Phi _ { s } ( x ) \| \| u _ { s } ^ { \rho } \| \cdot \cos \bigl ( \Phi _ { s } ( x ) , u _ { s } ^ { \rho } \bigr ) , } \end{array}\tag{5}
$$

separates each influence into three factors:

• Test sensitivity $\| \Phi _ { s } ( x ) \|$ : how strongly the test loss reacts to parameter movement. For a test point whose loss landscape is flat, $\Phi _ { s } ( x )$ is small and thus “forgetting” the instance through further updates becomes more difficult.

• Update magnitude $\| u _ { s } ^ { \rho } \|$ : how much the respective update component moves the parameters. Note that due to the division by the second moment, a component is not influential merely because its gradients are large, but because they are large relative to the recent gradient history.

• Directional alignment $\cos ( \Phi _ { s } ( x ) , u _ { s } ^ { \rho } )$ : how much the direction of a parameter update aligns with the direction of the integrated loss gradient of the predicted example x. Positive alignment yields negative influence, i.e., the component decreases the test loss; negative alignment increases loss; values close to zero imply that the model is changed without affecting the prediction of x.

We use this decomposition to analyze how training occurrences of the memorized sequences influence their predictions over the course of training, and how the rest of the training influences work towards or against memorization solutions.

An update may be aligned with a prediction in one layer but not in another and large opposing contributions may cancel at the model level. Since we are interested in uncovering such effects and localizing model internals that implement memorization, we follow Eichin et al. (2026) and further decompose each influence over model layers. Let $\boldsymbol { \Theta } = \{ \Theta _ { 1 } , \dots , \Theta _ { R } \}$ be a partitioning of the parameter coordinates. Restricting the vectors of our decomposition to these coordinates gives

$$
\begin{array} { r } { \textstyle \mathcal { T } _ { s } ^ { \rho } ( x ) = \sum _ { r = 1 } ^ { R } \mathcal { T } _ { s } ^ { \rho } ( x ; \Theta _ { r } ) , w i t h \mathcal { T } _ { s } ^ { \rho } ( x ; \Theta _ { r } ) : = - \big ( \Phi _ { s } ( x ) | _ { \Theta _ { r } } \big ) ^ { \top } \big ( u _ { s } ^ { \rho } | _ { \Theta _ { r } } \big ) . } \end{array}\tag{6}
$$

Thus Eq. 1 also decomposes over parameter groups. Grouping to the layer level as we do below, these influences show where learning happens in the model. Analogous definitions for the parameter-wise test sensitivity $\lVert \Phi _ { s } ( x ) \rVert _ { \Theta _ { r } }$ ∥ and update magnitude $\| u _ { s } ^ { \rho } \vert _ { \Theta _ { r } } \|$ follow. For alignment, we are interested in the contribution to the model-level alignment so we decompose the cosine as

$$
\begin{array} { r } { \cos ( \Phi _ { s } ( x ) , u _ { s } ^ { \rho } ) = \sum _ { r = 1 } ^ { R } \cos ( \Phi _ { s } ( x ) , u _ { s } ^ { \rho } ; \Theta _ { r } ) , w i t h \cos ( \Phi _ { s } ( x ) , u _ { s } ^ { \rho } ; \Theta _ { r } ) : = \frac { \Phi _ { s } ( x ) \left| \Theta _ { r } \cdot u _ { s } ^ { \rho } \right| _ { \Theta _ { r } } } { \left\| \Phi _ { s } ( x ) \right\| \left\| u _ { s } ^ { \rho } \right\| } . } \end{array}\tag{7}
$$

Experiment settings. To study the training dynamics of verbatim memorization in LLMs, we use our decomposition to analyze LLMs across the Pythia family (Biderman et al., 2023b), which provide fine grained access to checkpoints and training data and for which an exhaustive searches of duplicated and memorized sequences exists. We focus on Pythia-1B as our object of study, though we perform all experiments for models of size 160m, 410m, 1B, 1.4B, 2.8B, and 6.9B, unless stated otherwise.

![](images/223c03137a33ede36700c718e03a7e6e6a941c35b07f280210cf0dc919bea24b.jpg)  
Step

![](images/e49ee9991bc3cf288eaf05a1a818258ead2e786f477e68ee3166a3cfda70b6f4.jpg)  
Step

![](images/d926bbf46530eb8ea94710902c2234e53099e3e0cf6493a626163f508013edbf.jpg)  
(a) Loss until step 10k (b) Loss after step 10k  
(c) Num. 32-extr.

![](images/8a6527ab7b3b40f8b45bbcf9f8e22f2031ee48c836ef9e35a26fd16714fd2cb2.jpg)  
<sup>Step</sup>(d) Frac. of extractable tokens  
Figure 1: Pythia-1B training dynamics. We plot the loss and extractability of our sampled sequences over time and find that a lot of memorization happens early in training. Other models in App. Fig. 9.

For all numeric results and figures we present, we link to a version in the Appendix containing the same data for the other models. We study memorized sequences over training and decompose their cross entropy (CE) loss, which we sample from an exhaustive dataset of memorized sequences by Biderman et al. (2023a) for each model. We use their definition of memorization as 32-extractability, i.e., whether, given the first 32 tokens of the sequence, the model greedily generates the next 32 tokens verbatim. We filter out sequences with repeating subsequences (”reconstruction” in Prashanth et al. (2025)’s taxonomy) and samples of code, which we found to often contain boilerplate sequences, resulting in memorized examples resembling natural language. We follow Prashanth et al. (2025)’s taxonomy and further discriminate between recollected (memorized sequences that appear up to five times in the training set) and recited sequences (memorized sequences that appear more than five times in the training set). To contextualize our results, we include two control groups, control and control-duplicated, whose samples also occur up to and more than five times, respectively, but are not 32-extractable in the final model. For each of the four classes, we sample 32 sequences and study their decomposition. Details on datasets and implementations in App. B.

To our surprise, we could not verify 32-extractability for around 10 − 20% sequences sampled from Biderman et al. (2023a)’s dataset across model sizes. To investigate this discrepancy, we plot the average fraction of the correct continuation token being the maximum logit prediction across all sequences in Fig. 1d. The problem only shows for a small fraction of tokens, which indicates the problem is likely due to small numeric differences of our hardware and implementation for obtaining the logits. While we do not consider this a problem for our analysis, it calls into question whether demanding strict 32-extractable yields a robust definition of verbatim memorization.

Scaling to LLMs. We adapt our decomposition to Pythia and to its large training data as follows. First, following Eichin et al. (2026), we do not require access to the full training trajectory: we subsample the training steps to the set S of available checkpoints and decompose a simulated update per checkpoint $\theta _ { s }$ . Since the Pythia models do not provide optimizer states, we omit the first-moment term $u _ { s } ^ { \mathrm { f m } }$ and estimate the second-moment from independent samples of training data (details in App. A.2). We consider our second moment estimate effective at realistically rescaling the parameter updates, since our simulated parameter updates consistently decrease loss on the memorized and control groups. Second, we sample a training batch from the tail of Pythia’s training data, i.e. unseen by any of the checkpoints but the last one, with an original batch size of 2M tokens. Using this approach, we also inject our sampled sequences into the batch to simulate their occurrence during training (as proposed by Korner et al. 2026). This way, we can compute the influence a memorized¨ instance would have over training without searching for and decomposing the actual occurrences. We compute the simulated parameter update over the resulting batch to obtain $\theta _ { s + 1 }$

## 4 RESULTS

To get a high-level understanding of how memorization behavior in Pythia develops over time, we plot the loss and 32-extractability of the sequences we study in Fig. 1. Surprisingly, the loss trajectory of both the recited and recollected sequences separates from the control groups as early as 256 and 1, 000 steps respectively, i.e., less than 1% of the training. While previous work shows that memorization in Pythia is predictable from the perplexity in the final checkpoint (Prashanth et al., 2025), this result suggests that a separation of memorized and not memorized sequences is possible at much earlier training stages than reported by Biderman et al. (2023a), with the peak mean distance of the two classes reached as early as the halfway point of training and stable trajectories throughout. Adding to this, almost half of our memorized samples are already 32-extractable at this point as can be seen from Fig. 1c, i.e., memorization has already concluded for these instances. We show that thi indeed leads to much earlier predictability of memorization in Section 4.2.

A large fraction of memorized tokens is extractable early in training. Further, the tokenextractability trajectory reveals that a large fraction of the memorization learning happens in early training (see Fig. 1d). By 10k steps, the model already greedily decodes more than 80% of the memorized sequences, almost double the number of the control groups. This shows that strict 32-extractability as a measure lags behind the continuous, token-level emergence of verbatim memo rization, of which a large fraction happens in early training before step 20k.

Memorization during generalization? We also track basic linguistic competence of the model through the widely used BLiMP benchmark (Warstadt et al., 2020) (see App. Fig. 10), which measures the model’s ability to differentiate between linguistic minimal pairs. Interestingly, this evaluation of basic generalization in the model reaches its first plateau around 10k. Given that much<sup>1.0e−1</sup> of memorization develops before this mark and that our samples are mostly composed of naturaln<sup>)</sup> English language, this suggests that memorization in Pythia happens not after but during or eveni<sup>n</sup> <sub>t</sub><sup>r</sup> before generalization in form of basic linguistic capability.<sub>6.0e−2 8.0e−2</sub> 4.0e−2t<sup>r</sup> 2.0e−2n<sup>t</sup>

![](images/9b875e53bf774b5d96fc16d2e8f0a54a69805e4e1027df7961a04fd5b754e945.jpg)  
(a) Update magnitude ∥u<sup>x</sup><sub>s</sub> ∥

![](images/2584c34be28e83d255705233b653e7c62c39b5d62cbcfbdffa7387c86a6bc466.jpg)  
(b) Test sensitivity ∥Φ<sub>s</sub>(x)∥

![](images/b378f0d371dc0a9685bb5b4852066fdb4e8b54a0aaaa8ab4f1f41bbdacb48da9.jpg)  
(c) Self alignment cos(Φ<sub>s</sub>(x), u<sup>x</sup><sub>s</sub> )

![](images/331aadea51280ca0c442dfa75b7d59afe419838ec1d2d93dd99d83eb3b1f0050.jpg)  
(d) Train alignment cos(Φ<sub>s</sub>(x), u<sup>train</sup><sub>s</sub> )

![](images/394ee20abbbad4751e7aed29ccabbf862fd5f2bc3a118bf0e9c92c83314fe747.jpg)  
(e) W. decay align. cos(Φ<sub>s</sub>(x), u<sup>reg</sup><sub>s</sub> )  
Figure 2: Pythia-1B update magnitude, test sensitivity, and alignment. We plot the mean and standard deviation over the samples. See Eq. 5 for definitions. Other models in App. Fig. 14 and 12

## 4.1 DECOMPOSITION OF MEMORIZED SEQUENCES

To understand how the above dynamics emerge, we turn to the ExPLAIND decomposition. Interestingly, on the level of update magnitude $\| u _ { s } ^ { x } \|$ and test sensitivity $\| \Phi _ { s } ( x ) \|$ , the four classes undergo similar trends (see Fig. 2a, 2b). Recited instances, on average, show the lowest update magnitude, while exhibiting the highest test sensitivity. In contrast recollected instances have lower test sensitivity late in training and cluster with the control groups in terms of early sensitivity and update magnitude.

Memorization is enabled by self alignment. Self alignment cos $( \Phi _ { s } ( x ) , u _ { s } ^ { x } )$ is where we find one of the reasons for the diverging loss trajectories. First, consider Fig. 2c, where we plot the alignment of each predicted example with its own training influence. From the beginning of training and with further separation after step 512, the memorized samples show much higher levels of self alignment than the control groups. Recited instances, at 0.3 to 0.4 on average, show the highest level of self alignment for the rest of training, followed by the recollected instances around 0.25. Self alignment in the two control groups follows the same trajectory and is the lowest at around 0.18 through most of later training. This indicates that the tendency to move prediction towards a memorized solution is on average more than twice as high for recited instances than it is for non-memorized instances, while this gap is smaller for recollected instances.

Training causes more forgetting in recited than recollected samples. Memorization is however not just driven by the sample predictions’ tendency to align with their own effect on parameter updates, but by how likely the achieved solution is to survive the other training influences pulling towards generalization. We study the effect of the other training instances on memorized predictions in Fig. 2d. While positively aligned with the training data’s updates until step 128, the green recited instances start to show distinctively negative, lower alignment starting from step 256 which keeps decreasing until the end of training. On the other hand, the recollected and control group samples cluster around zero for most of training. This negative alignment suggests that the memorized solution to the recited instances is constantly overwritten by the parameter updates introduced by the other data, while recollected and control examples are, on average, not subject to such influences after initial training. This suggests one core mechanism behind recollection: While recited instances need to be duplicated to reinforce their respective memorized solution against being overwritten by the other training influences, recollected instances, i.e., memorized examples that are not duplicated across training, exist in a part of the parameter space that is, on average, orthogonal to such influences and they do not require continued repair.

## 4.2 MEMORIZATION PREDICTION

To validate our explanations of memorization above, we ask to what extent the memorized examples are really separated by the phenomena we describe above. We investigate this through a binary linear least squares classifier using our decomposition as features. We standardize the features and use ridge coefficient $1 0 ^ { - 8 }$ in a stratified six-fold cross-validation across our 128 examples, to avoid reliance on hyperparameter tuning. We report average accuracy on the held out fold as well as the average weight assigned to each feature. For an example x and a model checkpoint $\theta _ { s }$ at step $s ,$ we consider features including cross-entropy $\operatorname { C E } ( \theta _ { s } , x )$ , a strong predictor for memorization identified by previous work (Prashanth et al., 2025), and our decomposition features over time. Namely, we consider self alignment, train alignment, and weight decay alignment, $\mathrm { i . e . }$ , the cosine cos $( \Phi _ { s } ( x ) , u _ { s } )$ with the update $u _ { s }$ introduced by the example itself, the original training data, and weight decay respectively as defined in Eq. 2 and 5. We also consider test sensitivity $\| \breve { \Phi } _ { s } ( x ) \|$ and update magnitude $\Vert u _ { s } \Vert$ , totaling five decomposition features. Since both classes are stratified along duplication in the training data, we do not consider it as a feature. For each step, we use the features derived from only that step’s checkpoint. This way, our insights could benefit realistic settings, where predictions need to be made without access to the model’s history or future.

Memorization is predictable early in training. We find that very early on in training, all linear models perform above chance. As shown in Fig. 3a, up until step 256 all feature combinations perform roughly similarly at around 67 to 69% accuracy. After that, their performance trajectories split with the CE+decomposition feature scoring the highest at 90.6% accuracy at step 10k, while cross-entropy only and decomposition only follow at 84.4% and 75.0% respectively. The combined model keeps outperforming for the remaining checkpoints, peaking at 96.1% accuracy by step 40k. This confirms our observations above: By that point more than half of memorization learning is already completed, i.e., many examples are already 32-extractable and the mean cross-entropies have separated by almost two standard deviations (see Fig. 1). Besides, this prediction at scale has been reported at such later training stages by Biderman et al. (2023a).

Besides cross-entropy, self and train alignment are the strongest predictors of memorization. To investigate which features are the most predictive of memorization, we plot the feature weights

![](images/41c031ad0fa8e274b46f57f1c27f6e818554451f62bc3574517eec3742b7ce21.jpg)  
(a) Accuracy on held out

![](images/56fc69404610a0838ab8d129e5220cf9560e37dc6d11f38ebd83f7d2bbf32e93.jpg)  
(b) Weights decomp

![](images/7d2809417e73b6d9da6b169110ed20f0f2c38391bb6fb51faa3f6043fd7eaab3.jpg)  
Training step(c) Weights CE+decomp

Figure 3: Memorization classification. We plot the mean accuracy on held-out set and mean weights of the decomposition features of the decomposition and CE+decomposition. Others in App. Fig. 19.

in Fig. 3b. Underlining our assessment above, self alignment is the highest positive predictor over almost all checkpoints, while alignment with the original training influences is the most negative. We also look at the decomposition feature weights of the combined model in Fig. 3c. Interestingly, update magnitude and self alignment do not seem to add information beyond cross entropy as their weight is low over most models. Train and weight decay alignment also show negative predictive influence, while especially test sensitivity seems like the most informative component in early training, indicating that alignment with and sensitivity to forgetting are the factors not explained by CE alone.

For larger models, cross-entropy is less predictive of memorization. We perform the same naive classification across model scales (see App. Fig. 19) and find that cross entropy is a weaker early predictor of memorization as model size increases. Considering that the average cross entropies of the control sequences decreases as model size increases (see App. Fig. 9), this is consistent with our expectations. Interestingly, adding our decomposition features recovers a lot of the early predictability. For Pythia-6.9B, the CE+decomposition model still achieves 88.3% accuracy at step 10k, while the CE only model is at 77.3%. There, the CE model never catches up with the combined feature model, reaching comparable accuracy only in the final checkpoint.

## 4.3 LOCALIZING MEMORIZATION IN PYTHIA

(a) train→recited (b) train→recoll. (c) w. decay→recited (d) w. decay→recoll.<sup>10k</sup> <sup>30k</sup> <sup>50k</sup> <sup>70k</sup> <sup>90k</sup> <sup>110k 130</sup>  
![](images/4e6b6cc241143a92a5a93eac55cd0efdb3d42ac15cd2608345f7f1df3eab53d3.jpg)  
Step

![](images/0f0fb4c5cefe9c40c75f840f13a70027514eaaa543b399f7666f2a1d6c85b299.jpg)  
Step

![](images/ef6c37c297f1d3d77970b48d27bc63d737deabdf528149f67ecd65ee30556246.jpg)  
Step

![](images/a8c2f752298b33e632008e64e5185458df5f04d8e862ad46434c89e437202f6f.jpg)  
(a) Upd. magn. recited (b) Upd. magn. recoll. (c) Test sens. recited (d) Test sens. recoll.<sup>10k</sup> <sup>30k</sup> <sup>50k</sup> <sup>70k</sup> <sup>90k</sup> <sup>110k 13</sup>

Figure 4: Pythia-1B update magnitude and test sensitivity by layer depth. Others in App. Fig. 14.  
![](images/d01c903e721cd4e798ec8eaf8fbe637977f7fde0e7e4862a7400d15e6374f5b1.jpg)  
(a) Recited→recited

![](images/04383cd3033ed89adf660a362a46ca85c2fe2b79fe35f9716c64e87fe65529a4.jpg)  
(b) recoll.→recoll.     <sub>Step</sub>

![](images/9490d6ce91c39b6daeab765170c5729fd36ad64a59fb505e2a9e71553f17b733.jpg)

![](images/897d84f7ef339d3924a63b1a2e8c83866c80017cd0362c318444eb18a450a419.jpg)  
(c) Recited→recited (d) recoll.→recoll.     <sub>Step</sub>

Figure 5: Pythia-1B alignment by layer depth and type. Other models, control in App. Fig. 15, 16.  
![](images/b1f20dc6dc657f79eeed6bbf59dec80537e366e7f7c0e793676249c888e7ee56.jpg)

![](images/afcbaf3494022389d0f8ed5844990e5479e3efeb34e015e8323f1eab88a11fe8.jpg)

![](images/d880d2a4c6113bb97512b2958e38394ae3edf216c253222da565491f7f5cd399.jpg)  
Step

![](images/1c7619fba4ff38f12abec33adacef1cfd75f194648288ad83356217467153776.jpg)  
Figure 6: Pythia-1B train/weight decay alignment by model region. Other models in App. Fig. 17

We decompose our influences to the level of the model layers to see where the above learning influences stem from and to find the parameters implementing memorization. Our decomposition reveals that memorization influences are spread across the model, while peaking distinctly in some regions which we describe in the following. Our findings agree with arguments by which verbatim memorization is a distributed mechanism across the model (Maini et al., 2023; Dankers, 2026), and add further explanation to the conflicting findings in previous work regarding localization, beyond variation in memorization settings and methodology.

Memorization learning is largely located in lower layers. As we show in Fig. 4, our decomposition reveals that the memorized instances are most sensitive to model changes in the lower layers of the model throughout all of training. Simultaneously, self alignment as shown in Fig. 5 is the highest in the same layers, showing that a large fraction of memorization learning is located there. On the level of the update magnitude, Fig. 4 shows that training introduces the biggest changes to the input embeddings, while the other layer groups follow behind at similar levels.

Memorization learning focuses on the MLP layers. Turning to Fig. 5, where we plot self alignment grouped by layer type, we find that both forms of memorization self align the most in the MLP layers beginning early in training, while for both control groups it is roughly half of that throughout training. As we saw before, input embedding alignment is the most relevant source of alignment for both control groups and on par with MLP alignment in the recollected examples. Alignment in the attention QKV projections is also much higher in the memorized examples than the control groups.

Recited memories degrade in lower layers. In Fig. 6, we plot the alignment across layer regions with the original training data’s parameter updates. We find that the effects from the training data onto the recited instances align in the lower model layers 0-3, much more than for recollected examples and especially during late training. We find the same for alignment with weight decay, which we plot in Fig. 6. There, alignment of recited instances is high throughout training, while alignment of recollected examples tends to zero in the second half of training. Combined with the fact that test sensitivity is the highest in these layers, this explains why recited instances depend on a high level of duplication during training: As training progresses, their memories written to the parameters of the lower model layers are subject to constant forgetting due to the influences of other data and regularization, thus requiring the continued restoration of these memories. On the other hand, recollected sequences are subject to such degrading influences to a much smaller extent. The mechanisms for recall that their gradients write onto lower model layers are more robust against other training influences and thus require only a low level of verbatim duplication to remain effective.

## 4.4 CAUSAL MEMORIZATION ABLATIONS

To validate our insights from the loss decomposition, we perform causal ablations on the parameter level to observe changes in the model memorization behavior or training dynamics. Strongly negative self-influences $\mathcal { T } _ { s } ^ { x } ( x ; \theta )$ indicate that a parameter has contributed strongly to the loss decreasing on the predictions x due to memorization. We thus identify a top-K% set $\mathcal { T } _ { K }$ of parameters θ by the negative magnitude of their memorization influences $\mathcal { T } _ { s } ^ { x } ( x ; \theta )$ . We intervene by setting the identified parameters to zero and observing the 32-extractability of the sequences

![](images/88151c3bc4540314b17b9e9ad9192226ea54dcc9d0bc5bbdf4ad8fcf0223e5f4.jpg)

![](images/952987c088cf030adaf1fd00125e6d6e67dbf30135a17163cfa42e14f7d9f1ac.jpg)  
Figure 7: Pythia-1B memorization ablation. Left: Number of 32-extractable sequences in the memorized sample, right: CE loss on the control sample. Other models in App. Fig. 20.

we decompose. We also tried other interventions which performed comparably, in particular replacing the parameters with the mean value of their group.

Ablation in the embeddings is not effective. Recall that a large fraction of memorization updates stems from the input embeddings. Consequently, we find that directly using the influence scores to rank parameters is biased heavily towards them (see App. Fig. 21). As we scale up the fraction of ablated parameters, model behavior in terms of CE on the control groups breaks down in sync with memorization. This indicates that the memorization influences in the embeddings, especially update magnitude, are an artifact of training rather than a causal building block of memorization. We thus restrict our following interventions to internal parameters.

Our influences enable sparse and early memorization ablations. We plot 32-extractability of our sampled sequences across different fractions of ablated parameters K and using the accumulated influence scores from different points in training in Fig. 7. Strikingly, as early as step 2000, the information from our decomposition robustly ablates memorization behavior across our samples, with 32-extractability approaching zero when ablating the top-0.0002 parameters of Pythia-1B. Later in training, these interventions become focused, as complete ablation succeeds up to a sparsity of top-0.00002 intervened. At the same time, we observe cross entropy on the control samples to check whether our interventions simply break general model behavior and indeed, only the largest ablations break model behavior, while changes in the smaller interventions are modest below 0.1. We plot the results of the other models up to 2.8B parameters in App. Fig. 20 which show that our findings generalize across models, i.e., for each model size there is a number of parameters which both effectively ablates memorization and leaves control cross-entropy intact. We consider this a very interesting result, since both our localization, as well as the intervention on parameters, are quite naively building on our decomposition. It causally underlines that our decomposition identifies the locations of the memories in the model successfully and that actionable post-hoc fixes are possible.

Memorization in smaller models is ablated by a single attention head. Interestingly, most of the ablated parameters are located in the attention layers (see App. Fig. 22) and we find that recited instances tend to align with other recited instances in the lower attention layers (see App. C.4). Motivated by this, we intervene on the attention heads in these layers. Interestingly, we find that, up to 1B parameters, memorization can be ablated by intervening on a single attention head using our naive approach, a result which echoes Stoehr et al. (2024) who find similar results for fine-tuning an attention head in a small GPT-Neo model. Details in App. Sec. C.4.

## 5 DISCUSSION & CONCLUSION

Our work presents novel perspectives on the training dynamics of memorization in LLMs. Here we review our results along the time and model scale dimensions.

Memorization follows three training phases. Across models, the loss of both the memorized and control instances flattens around the 5k step mark. This co-occurs with the saturation of the model’s performance in BLiMP, which marks an important phase transition in language model generalization (Chen et al., 2024). Most memorization follows a similar trajectory, with more than 80% of the tokens already greedily decodeable at the same point. Decomposing their loss trajectory reveals that on the model- and layer-level almost all the trends of the influences start to stabilize around that point. This development seems to be completed around the 20k mark. Further, our layer-level results indicate that in the first 10k steps, the training dynamics introduce a shared mechanism for memorization in an attention head in the first transformer layer. This aligns with other reports of early heuristics for context processing, for instance context copying (Korner et al., 2026). Further, memorization¨ predictability starts to become successful around step 10k and reaches its peak shortly after for most models. In sum, this suggests that most of the memorization mechanisms are formed in early training and that later memorization dynamics are largely refining and maintaining existing memories, rather than building new ones. Further, it suggests that memorization and generalization appear in paralle in the model. We argue the point of emergence of both, around step 5k, to be the first phase transition of memorization. The second transition seems to happen somewhere around step 20k, when most of our influences stabilize and after which predictability converges.

Memorization and model scale. Previous work asks how memorization behaves as a function of model scale (Tirumala et al., 2022). As we report in Sec. 4, as model scale increases, the training data is fit to a higher extent, resulting in the average CE loss approaching that of the memorized instances. Predicting memorization based on solely CE or perplexity is more difficult in these models, though we show features derived from our gradient decomposition to recover some of the early predictability observed in the smaller models. Further, as model scale increases, our decomposition shows that the constant forgetting effects in the lower parts of the model due to training and weight decay become weaker, thus explaining the findings by Tirumala et al. (2022). For the largest models, weight decay even is slightly aligned with memorization on average. This could explain previous observations by which memorization scales with model size (Carlini et al., 2023; Biderman et al., 2023a).

Limitations. Our methodology omits the influences of the first moment term in AdamW and estimates the second moment. Both choices could bias our results. Besides, our memorization prediction should be seen as a validation of our explanations and a proof-of-concept rather than a direct competitor to Biderman et al. (2023a) and Prashanth et al. (2025)’s approaches over the entire training dataset. Computing our ExPLAIND features over the entire dataset is infeasible, though we think that gradient based proxies derived from parameter updates of the train loop are a promising possibility. Our interventions likewise validate the analysis but are post-hoc in nature: they do not establish how to prevent memorization during training, and their effects on robustness and other predictions remain unclear. Finally, our findings concern verbatim memorization in the Pythia family up to 6.9B parameters. Their generalizability to larger models, other architectures, and other forms of memorization, for instance in the vision domain or in a noisy label setting, remains open.

## ACKNOWLEDGMENTS

We thank Felicia Korner and Martin B ¨ ar for their feedback on earlier versions of this paper. We ¨ acknowledge the support for BP through the ERC Consolidator Grant DIALECT 101043235.

## AI USE STATEMENT

This study was conceived and written by its human authors, who take full responsibility for the claims presented and artifacts produced. We acknowledge the help of AI tools for implementing our experiments, though we checked all generated code for its correctness. Further, AI helped verifying and improving our math and proofs. In particular, the second moment estimator presented in Appendix A.2 was developed with the help of AI. We incorporated a small amount feedback from an AI tool into our writing to improve clarity and grammar.

## REPRODUCIBILITY STATEMENT

Our study focuses on the fully open sourced Pythia family, for which checkpoints, training data, and training loop are available on Github and Huggingface. To ensure reproducibility of our results, we will provide the full implementation of our method and experiments which allows to reproduce all our results and comes with all documentation necessary to run all the experiments described in this paper on Github upon publication. Besides, we did our best to document the parameters and design choices necessary for reproduction of our study in Appendix B.

## REFERENCES

Devansh Arpit, Stanislaw Jastrzkebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In Doina Precup and Yee Whye Teh (eds.), Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 233–242. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/arpit17a.html.

Brian Wesley Bell, Michael Geyer, David Glickenstein, Amanda S. Fernandez, and Juston Moore. An Exact Kernel Equivalence for Finite Classification Models. In Proceedings of 2nd Annual Workshop on Topology, Algebra, and Geometry in Machine Learning (TAG-ML), pp. 206–217. PMLR, September 2023.

Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Gregory Anthony, Shivanshu Purohit, and Edward Raff. Emergent and Predictable Memorization in Large Language Models. In Thirty-Seventh Conference on Neural Information Processing Systems, November 2023a.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, Usvsn Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. In Proceedings ofthe 40th International Conference on Machine Learning, pp. 2397–2430. PMLR, July 2023b.

Stella Biderman, Mohammad Aflah Khan, Niloofar Mireshghallah, Catherine Arnett, Fazl Barez, and Naomi Saphra. Position: Don’t just ”fix it in post”: A science of AI must study learning dynamics. In Forty-third International Conference on Machine Learning Position Paper Track, 2026. URL https://openreview.net/forum?id=VbHh05xQ7y.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. In The Eleventh International

Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=TatRHT\_1cK.

Ting-Yun Chang, Jesse Thomason, and Robin Jia. Do Localization Methods Actually Localize Memorized Data in LLMs? A Tale of Two Benchmarks. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3190–3211, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.176.

Satrajit Chatterjee. Learning and memorization. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 755–763. PMLR, 10–15 Jul 2018. URL https: //proceedings.mlr.press/v80/chatterjee18a.html.

Angelica Chen, Ravid Shwartz-Ziv, Kyunghyun Cho, Matthew L Leavitt, and Naomi Saphra. Sudden drops in the loss: Syntax acquisition, phase transitions, and simplicity bias in MLMs. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=MO5PiKHELW.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge Neurons in Pretrained Transformers. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493–8502, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.581.

Verna Dankers. Memorisation meets compositionality in natural language processing. In Yanai Elazar, Allyson Ettinger, Nora Kassner, and Sebastian Ruder (eds.), Proceedings ofThe Big Picture v2: Crafting a Research Narrative, pp. 144–159, San Diego, CA, USA, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-416-3. doi: 10.18653/v1/2026.bigpicture-main.12. URL https://aclanthology.org/2026.bigpicture-main.12/.

Verna Dankers and Ivan Titov. Generalisation First, Memorisation Second? Memorisation Localisation for Natural Language Classification Tasks. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 14348–14366, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.852.

Yupei Du, Philipp Mondorf, Silvia Casola, Yuekun Yao, Robert Litschko, and Barbara Plank. Reason to rote: Rethinking memorization in reasoning. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 8659–8679, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/ 2025.emnlp-main.437. URL https://aclanthology.org/2025.emnlp-main.437/.

Florian Eichin, Yupei Du, Philipp Mondorf, Maria Matveev, Barbara Plank, and Michael A. Hedderich. ExPLAIND: Unifying model, data, and training attribution to study model behavior. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=7G6x9QTaN4.

Vitaly Feldman. Does learning require memorization? a short tale about a long tail. In Proceedings ofthe 52nd Annual ACM SIGACT Symposium on Theory ofComputing, STOC 2020, pp. 954–959, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450369794. doi: 10.1145/3357713.3384290. URL https://doi.org/10.1145/3357713.3384290.

Vitaly Feldman and Chiyuan Zhang. What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation. volume 33, pp. 2881–2891. Curran Associates, Inc., 2020.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer Feed-Forward Layers Are Key-Value Memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wentau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446.

Robin Jia, Eric Wallace, Yangsibo Huang, Tiago Pimentel, Pratyush Maini, Verna Dankers, Johnny Wei, and Pietro Lesci (eds.). Proceedings of the First Workshop on Large Language Model Memorization (L2M2), Vienna, Austria, August 2025. Association for Computational Linguistics. ISBN 979-8-89176-278-7. doi: 10.18653/v1/2025.l2m2-1.0. URL https://aclanthology. org/2025.l2m2-1.0/.

Sara Kangaslahti, Elan Rosenfeld, and Naomi Saphra. Hidden Breakthroughs in Language Model Training, June 2025.

Pang Wei Koh and Percy Liang. Understanding Black-box Predictions via Influence Functions. In Proceedings of the 34th International Conference on Machine Learning, pp. 1885–1894. PMLR, July 2017.

Felicia Korner, Maria Matveev, Florian Eichin, Gitta Kutyniok, Barbara Plank, and Michael A.¨ Hedderich. Copy first, translate later: Interpreting translation dynamics in multilingual pretraining, 2026. URL https://arxiv.org/abs/2604.17633.

Janice Lan, Rosanne Liu, Hattie Zhou, and Jason Yosinski. LCA: Loss Change Allocation for Neural Network Training. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8424–8445, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. acl-long.577. URL https://aclanthology.org/2022.acl-long.577/.

Pratyush Maini, Michael C. Mozer, Hanie Sedghi, Zachary C. Lipton, J. Zico Kolter, and Chiyuan Zhang. Can neural network memorization be localized? In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Kevin Meng, David Bau, Alex J. Andonian, and Yonatan Belinkov. Locating and Editing Factual Associations in GPT. In Advances in Neural Information Processing Systems, October 2022.

Fatemehsadat Mireshghallah, Archit Uniyal, Tianhao Wang, David Evans, and Taylor Berg-Kirkpatrick. An empirical analysis of memorization in fine-tuned autoregressive language models. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pp. 1816–1826, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. emnlp-main.119. URL https://aclanthology.org/2022.emnlp-main.119/.

Francesco Ortu, Zhijing Jin, Diego Doimo, Mrinmaya Sachan, Alberto Cazzaniga, and Bernhard Scholkopf. Competition of Mechanisms: Tracing How Language Models Handle Facts and ¨ Counterfactuals. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8420–8436, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.458.

USVSN Sai Prashanth, Alvin Deng, Kyle O’Brien, Jyothir S V, Mohammad Aflah Khan, Jaydeep Borkar, Christopher A. Choquette-Choo, Jacob Ray Fuehne, Stella Biderman, Tracy Ke, Katherine Lee, and Naomi Saphra. Recite, reconstruct, recollect: Memorization in LMs as a multifaceted phenomenon. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=3E8YNv1HjU.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/ 2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html.

Niklas Stoehr, Mitchell Gordon, Chiyuan Zhang, and Owen Lewis. Localizing Paragraph Memorization in Language Models, March 2024.

Kushal Tirumala, Aram H. Markosyan, Luke Zettlemoyer, and Armen Aghajanyan. Memorization without overfitting: Analyzing the training dynamics of large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=u3vEuRr08MT.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. BLiMP: The benchmark of linguistic minimal pairs for English. Transactions ofthe Associationfor Computational Linguistics, 8:377–392, 2020. doi: 10.1162/tacl a 00321. URL https://aclanthology.org/2020.tacl-1.25/.

Xiaosen Zheng and Jing Jiang. An Empirical Study of Memorization in NLP. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6265–6278, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long. 434.

## APPENDIX

## A PROOFS AND ADDITIONAL DETAILS ON METHODOLOGY

## A.1 PROOF OF EQUATION 1

We restate Theorem 3.1 and the output-function part of Corollary 3.3 of Eichin et al. (2026), then apply them to the sequence loss. We collect the contributions of earlier gradients into the incoming first moment and retain one contribution per training sequence. In the restatement, we use S for the number of training steps, reserving N for the fixed sequence-loss normalizer.

Theorem A.1 (ExPLAIND decomposition under AdamW, Eichin et al. 2026). Let $f _ { \theta } : \mathcal { X }  \mathcal { Y }$ be a model with parameters $\boldsymbol { \theta } \in \mathbb { R } ^ { D }$ , where $\mathcal { X } \subseteq \mathbb { R } ^ { I }$ and $\mathcal { V } \subseteq \mathbb { R } ^ { O }$ . Let $\mathcal { D } = \{ ( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { M } , y _ { M } ) \}$ be a dataset and $L : \mathcal { V } \times \mathcal { V } \to \mathbb { R } _ { \geq 0 }$ a per-sample loss. Assume that $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ is continuously differentiable in θ and $L ( z , y )$ in z along the parameter segments considered below. At training step $i ,$ define $\begin{array} { r } { \mathcal { L } _ { i } ( \boldsymbol { \theta } ) : = \sum _ { k \in B _ { i } } w _ { i , k } L ( \hat { f _ { \boldsymbol { \theta } } ( x _ { k } ) , \boldsymbol { y } _ { k } ) } } \end{array}$ , where $\check { B _ { i } } \subseteq \{ 1 , \dots , M \}$ and the weights $w _ { i , k } \in \mathbb { R }$ are fixed when taking the gradient.

Suppose that $\theta _ { S }$ is obtained from $\theta _ { 0 }$ by S AdamW steps with learning rates $\alpha _ { s } > 0 ,$ , weight decay $\lambda \overset {  } { \geq } 0 , \beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ , and $\epsilon > 0$ . With $m _ { 0 } = v _ { 0 } = 0 ,$ , the updates are

$$
\begin{array} { r l r } & { \quad g _ { s } : = \nabla _ { \theta } \mathcal { L } _ { s } ( \theta _ { s } ) , } & \\ & { m _ { s + 1 } : = \beta _ { 1 } m _ { s } + ( 1 - \beta _ { 1 } ) g _ { s } , } & \\ & { \quad v _ { s + 1 } : = \beta _ { 2 } v _ { s } + ( 1 - \beta _ { 2 } ) g _ { s } ^ { 2 } , } & { \hat { v } _ { s + 1 } : = \cfrac { v _ { s + 1 } } { 1 - \beta _ { 2 } ^ { s + 1 } } , } \\ & { \quad \theta _ { s + 1 } : = \theta _ { s } - \cfrac { \alpha _ { s } } { 1 - \beta _ { 1 } ^ { s + 1 } } \cfrac { m _ { s + 1 } } { \sqrt { \hat { v } _ { s + 1 } } + \epsilon } - \alpha _ { s } \lambda \theta _ { s } . } & \end{array}\tag{8}
$$

Then, for every $x \in \mathcal { X }$

$$
f _ { \theta _ { S } } ( x ) = f _ { \theta _ { 0 } } ( x ) - \sum _ { k = 1 } ^ { M } \sum _ { s = 0 } ^ { S - 1 } \phi _ { s } ^ { \mathrm { t e s t } } ( x ) \phi _ { s } ^ { \mathrm { t r a i n } } ( x _ { k } ) - \sum _ { s = 0 } ^ { S - 1 } \phi _ { s } ^ { \mathrm { t e s t } } ( x ) \mathbf { r } _ { s } ,\tag{9}
$$

where

$$
\begin{array} { l } { \displaystyle { \phi _ { s } ^ { \mathrm { t e s t } } ( x ) : = \int _ { 0 } ^ { 1 } \nabla _ { \theta } f _ { \theta _ { s } ( t ) } ( x ) d t \in \mathbb { R } ^ { O \times D } , } } \\ { \displaystyle { \phi _ { s } ^ { \mathrm { t r a i n } } ( x _ { k } ) : = \sum _ { i = 0 } ^ { s } \alpha _ { s , i } \mathbf { 1 } _ { k \in B _ { i } } w _ { i , k } \frac { \nabla _ { \theta } L \left( f _ { \theta _ { i } } ( x _ { k } ) , y _ { k } \right) } { \sqrt { \bar { \nu } _ { s + 1 } } + \epsilon } \in \mathbb { R } ^ { D } } , } \\ { \displaystyle { \mathbf { \phi } _ { \theta _ { s } ( t ) } : = \alpha _ { s } \lambda \theta _ { s } , } } \\ { \displaystyle { \alpha _ { s , i } : = \alpha _ { s } ( 1 - \beta _ { 1 } ) \beta _ { 1 } ^ { s - i } } . } \end{array}\tag{10}
$$

Here $\nabla _ { \boldsymbol { \theta } } f$ denotes the output Jacobian, while gradients ofscalar losses are column vectors. Powers, square roots, and division ofvectors are coordinatewise.

Corollary A.2 (ExPLAIND decomposition for functions of the outputs, Eichin et al. 2026). Under the assumptions of Theorem A.1, let $h : \mathcal { y }  \mathbb { R } ^ { Q }$ be continuously differentiable on the output paths, and define ${ \widetilde { f } } _ { \theta } : = h \circ f _ { \theta } $ . Equation 9 also holdsfor $\widetilde { f }$ after replacing the testfeature map by

$$
\widetilde { \phi } _ { s } ^ { \mathrm { t e s t } } ( { \boldsymbol { x } } ) : = \int _ { 0 } ^ { 1 } \nabla _ { \theta } \bigl ( h ( f _ { \theta _ { s } ( t ) } ( { \boldsymbol { x } } ) ) \bigr ) \ d t .\tag{11}
$$

The trainingfeature maps, regularization terms, and optimizer trajectory remain unchanged. For a scalar output, the integrand in Eq. 11 is written as a row vector.

Combining the theorem and corollary, we now specialize to the sequence loss. Fix $N \in  { \mathbb { N } } _ { > 0 }$ independently of sequence length. For a sequence $x = ( x _ { 1 } , \ldots , x _ { | x | } )$ , set

$$
\ell ( \theta ; x ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { | x | } L \big ( f _ { \theta } ( x _ { < i } ) , x _ { i } \big ) .\tag{12}
$$

For each fixed x, apply Corollary ${ \tt A . 2 }$ to the model’s stacked next-token outputs with $h _ { x } ( a ) : =$ $\begin{array} { r } { N ^ { - 1 } \sum _ { i = 1 } ^ { | x | } L ( a _ { i } , x _ { i } ) } \end{array}$ . The test feature map is therefore

$$
\widetilde { \phi } _ { s } ^ { \mathrm { t e s t } } ( \boldsymbol { x } ) = \left( \int _ { 0 } ^ { 1 } \nabla _ { \theta } \ell ( \theta _ { s } ( t ) ; \boldsymbol { x } ) d t \right) ^ { \top } = \Phi _ { s } ( \boldsymbol { x } ) ^ { \top } .\tag{13}
$$

For training, the theorem’s per-sample loss is the same sequence loss. Writing $B _ { s }$ for the batch of training sequences, we sum these losses, so that the theorem’s weights are $w _ { s , k } = 1$ and

$$
g _ { s } = \sum _ { z \in B _ { s } } \nabla _ { \theta } \ell ( \theta _ { s } ; z ) = \frac { 1 } { N } \sum _ { z \in B _ { s } } \sum _ { i = 1 } ^ { | z | } \nabla _ { \theta } L \big ( f _ { \theta _ { s } } ( z _ { < i } ) , z _ { i } \big ) .\tag{14}
$$

Redistributing the terms of Eq. 10, separating the current step from earlier steps, gives

$$
\begin{array} { l } { { \displaystyle \sum _ { k = 1 } ^ { M } \phi _ { s } ^ { \mathrm { t r a i n } } ( x _ { k } ) = \sum _ { i = 0 } ^ { s } \alpha _ { s , i } \frac { g _ { i } } { \sqrt { \hat { v } _ { s + 1 } } + \epsilon } } } \\ { { = \displaystyle \frac { \alpha _ { s } } { 1 - \beta _ { 1 } ^ { s + 1 } } \frac { ( 1 - \beta _ { 1 } ) g _ { s } + \beta _ { 1 } m _ { s } } { \sqrt { \hat { v } _ { s + 1 } } + \epsilon } } } \\ { { = \displaystyle \sum _ { z \in B _ { s } } u _ { s } ^ { z } + u _ { s } ^ { \mathrm { f m } } , } } \end{array}\tag{15}
$$

where

$$
\begin{array} { r l } & { u _ { s } ^ { z } : = \cfrac { \alpha _ { s } \left( 1 - \beta _ { 1 } \right) } { 1 - \beta _ { 1 } ^ { s + 1 } } \cfrac { \nabla _ { \theta } \ell \left( \theta _ { s } ; z \right) } { \sqrt { \hat { v } _ { s + 1 } } + \epsilon } , } \\ & { u _ { s } ^ { \mathrm { f m } } : = \cfrac { \alpha _ { s } \beta _ { 1 } } { 1 - \beta _ { 1 } ^ { s + 1 } } \cfrac { m _ { s } } { \sqrt { \hat { v } _ { s + 1 } } + \epsilon } , } \\ & { u _ { s } ^ { \mathrm { r e g } } : = \mathbf { r } _ { s } = \alpha _ { s } \lambda \theta _ { s } . } \end{array}\tag{16}
$$

Here $m _ { s }$ is the uncorrected first moment entering step $s ,$ and s retains the optimizer’s original step count. Substituting Eqs. 13 and 15 into the loss version of $\operatorname { E q . 9 }$ with a single update step gives

$$
\begin{array} { c } { \displaystyle \ell ( \theta _ { s + 1 } ; x ) - \ell ( \theta _ { s } ; x ) = - \Phi _ { s } ( x ) ^ { \top } \left( \sum _ { z \in B _ { s } } u _ { s } ^ { z } + u _ { s } ^ { \mathrm { f m } } + u _ { s } ^ { \mathrm { r e g } } \right) } \\ { = \displaystyle \sum _ { z \in B _ { s } } \mathcal { T } _ { s } ^ { z } ( x ) + \mathcal { T } _ { s } ^ { \mathrm { f m } } ( x ) + \mathcal { T } _ { s } ^ { \mathrm { r e g } } ( x ) , } \end{array}\tag{17}
$$

where $\mathcal { T } _ { s } ^ { \rho } ( x ) : = - \Phi _ { s } ( x ) ^ { \top } u _ { s } ^ { \rho }$ . This yields Eq. 1 with sequence-indexed data contributions.

Batch reduction. The specialization above assumes the gradient in Eq. 14. If the implementation additionally averages over training sequences, multiply each $u _ { s } ^ { z }$ by $| \hat { B _ { s } } | ^ { - 1 }$ and form the moment states using that batch-mean gradient. This is separate from the fixed token-loss normalization.

Batch gradient clipping. If the accumulated batch gradient is norm-clipped before the AdamW moment updates, let $c _ { s }$ be its realized scalar clipping factor, so that the gradient supplied to AdamW is $\begin{array} { r } { c _ { s } \sum _ { z \in B _ { \mathrm { s } } } \nabla _ { \theta } \ell ( \theta _ { s } ; z ) } \end{array}$ under the sum convention. The proof applies with weights $w _ { s , k } = c _ { s } ,$ treating $c _ { s }$ as fixed when attributing the realized update. Accordingly, multiply each $u _ { s } ^ { z }$ in Eq. 16 by the same $c _ { s }$

## A.2 ESTIMATING THE SECOND-MOMENT STATE OF PYTHIA

Since intermediate Pythia checkpoints are released without optimizer state, we reconstruct the Adam second moment v empirically. With $\beta _ { 2 } ~ = ~ 0 . 9 5$ , the exponential moving average $v _ { t } =$ $\beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) g _ { t } ^ { 2 }$ assigns weight $( 1 - \beta _ { 2 } ) \beta _ { 2 } ^ { k }$ to the squared gradient k steps in the past, an effective horizon of $( 1 - \bar { \beta } _ { 2 } ) ^ { - 1 } = 2 0$ steps. Assuming the gradient distribution is approximately stationary over this window, v at a checkpoint is well approximated by the expected squared per-batch gradient at that checkpoint, $\mathbb { E } [ g _ { B } ^ { 2 } ]$ , where $g _ { B }$ is the per-coordinate gradient of a training batch of B i.i.d. sequences.

Writing per-sequence gradients as $g _ { i } = \mu + \varepsilon _ { i }$ with full-data gradient $\mu$ and $\mathrm { i . i . d . }$ . zero-mean noise $\varepsilon _ { i }$ of variance $\bar { \sigma } ^ { 2 }$ , the batch gradient satisfies $\mathbb { E } [ g _ { R } ^ { 2 } ] = \mu ^ { 2 } + \sigma ^ { 2 } / B$ . Estimating this target from batches of $b \ll B$ sequences is therefore biased: $\bar { \mathbb { E } } [ \dot { g } _ { b } ^ { 2 } ] = \mu ^ { 2 } + \sigma ^ { 2 } / b$ , which exceeds the target by $\left( 1 / b - 1 / B \right) \sigma ^ { 2 }$ , worst case up to a factor $B / b$ in the noise-dominated regime $\mu ^ { 2 } \ll \sigma ^ { 2 } / B$ , and this bias does not vanish with more samples.

Instead, we compute $N$ per-sequence gradients $g _ { 1 } , \ldots , g _ { N }$ (clipped as in training) on held-out Pile data disjoint from the batches whose contributions we decompose, and form the per-coordinate sample mean $\bar { g }$ and sample variance $s ^ { 2 }$ . Since $\mathbb { E } [ \bar { g } ^ { 2 } ] = \mu ^ { 2 } + \dot { \sigma } ^ { 2 } / N$ and $\mathbb { E } [ s ^ { 2 } ] = { \dot { \sigma } } ^ { 2 }$ , the plug-in estimator

$$
\hat { v } \ = \ \operatorname* { m a x } \biggl ( \bar { g } \ : ^ { 2 } - \ : \frac { s ^ { 2 } } { N } , \ : 0 \biggr ) \ + \ \frac { s ^ { 2 } } { B }\tag{18}
$$

satisfies $\mathbb { E } [ \hat { v } ] = \mu ^ { 2 } + \sigma ^ { 2 } / B$ exactly, apart from the truncation at zero, which introduces a small positive bias only where $\mu ^ { 2 }$ is statistically indistinguishable from zero; there the retained term $s ^ { 2 } / B$ dominates the target in any case. We use $B = 1 0 \bar { 2 } 4$ , the Pythia training batch size.

To choose $N _ { \ast }$ note that in the noise-dominated regime the variance of a single squared gradient is $\mathrm { V a r } ( \varepsilon _ { i } ^ { 2 } ) = ( \kappa - 1 ) \sigma ^ { 4 }$ , where κ is the noise kurtosis $( \kappa = 3$ for Gaussian noise), so the relative standard error of vˆ scales as $\sqrt { ( \kappa - 1 ) / N }$ . Adam applies vˆ through $\sqrt { \hat { v } }$ in the update denominator, which halves the relative error to first order. With $N = 2 5 6$ this yields a relative error in the update of roughly $\textstyle { \frac { 1 } { 2 } } { \sqrt { 2 / 2 5 6 } } \approx 4 \%$ under Gaussian noise, and remains below 10% even for heavy-tailed gradients with $\kappa \leq 1 3$ . Both figures are small compared to the intrinsic sampling fluctuation of the quantity being replaced: the true 20-step EMA is itself an average of around 20 noisy squared gradients and thus carries relative fluctuations of order $\sqrt { 2 / 2 0 } \approx 3 0 \%$ . Finally, because vˆ is estimated on data independent of the update batch, the reconstructed update remains linear in the gradients being attributed.

## B ADDITIONAL EXPERIMENT SETTINGS

## B.1 MODELS

We use EleutherAI/pythia-<size>-deduped GPT NeoX checkpoints loaded from huggingface and the corresponding tokenizer. Our decomposition implementation is a fork of the original ExPLAIND repository, extended with the Pythia reconstruction, sample level decomposition, prediction, and ablation code used here. Models are loaded through FixedLossGPTNeoXForCausalLM with Flash Attention $^ { 2 , }$ and dropout is disabled. Computation uses float32 by default and bfloat16 for the 6.9B model to reduce memory use. We restricted each decomposition to one H200 GPU; we did not decompose the 12B model because of the computational cost. We reconstruct AdamW with Pythia’s learning rate schedule, betas (0.9, 0.95), epsilon $1 0 ^ { - 8 }$ , weight decay 0.01, gradient clipping with norm limit a 1, and zero the first moments. Second moments are estimated from the 256 sequences above. The path integral in the test feature map of ExPLAIND uses 100 integration steps following Eichin et al. (2026)’s recommendation.

## B.2 DATA

For each selected deduplicated Pythia model, we sample the first 64 Pythia tokenizer tokens of 32 examples in each of four classes. The classes are defined by 32-extractability in as precomputed by Biderman et al. (2023a) (code at github repo EleutherAI/pythia-memorized-evals) and by the occurrence count of the 32-token continuation in the deduplicated Pile, taken from huggingface dataset usvsnsp/deduped-num-duplicates: memorized recited (extractable, count $> 5 )$ memorized recollected (extractable, count $\leq 5 )$ , control duplicated (not extractable, count $> 5 )$ and control (not extractable, count ≤ 5), following Prashanth et al. (2025). Controls come from the tail of the preshuffled deduplicated Pile mmap and exclude memorized indices. The 128 prediction examples pass a text filter that rejects excessive capitalization, selected code characters, more than five digits, and repeated substrings. The reconstruction uses 1,024 sequences of 2,048 tokens for the original update and a disjoint set of 256 sequences to estimate the second moment; these sequences also come from the Pile tail but are not subject to the text filter.

## B.3 EXPLAIND DECOMPOSITION AND FEATURES

Our implementation is a fork of the original ExPLAIND repository, extended with the Pythia reconstruction, sample level decomposition, prediction, and ablation code used here. We evaluate checkpoints at steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, every 1, 000 steps from 1, 000 through 9, 000, and every 10, 000 steps from 10, 000 through 130, 000. We decompose all 128 sequences separately and retain training sequences as separate, accumulated group. Sequence loss follows Pythia’s update convention, namely the token loss sum divided by 2,048.

## B.4 MEMORIZATION PREDICTION

For the prediction, at each checkpoint we compute five features for every prediction sequence: cosine alignment with the original training update, cosine alignment with the matching sequence update, the prediction feature norm, the matching update feature norm, and alignment with the weight decay contribution. Dot products and squared norms are summed across parameters. Prediction uses the features of one checkpoint at a time; accumulated scores are used only for the descriptive decomposition plots and ablations. The plots report class means and standard deviations for both the memorized versus control grouping and the four classes. We also evaluate BLiMP at every checkpoint by comparing the summed sentence log probabilities in each minimal pair.

We fit ridge linear models with an unpenalized intercept and coefficient $1 0 ^ { - 8 }$ using six stratified folds generated with seed 42. Feature standardization is fitted only on the training portion of each fold. Binary predictions threshold the scalar output at 0.5. We evaluate decomposition features, CE, and decomposition+CE, using one checkpoint at a time. We report accuracy from predictions on the excluded folds and retain the scores, predictions, labels, fold assignments, and fitted weights.

## B.5 CAUSAL ABLATION

For each checkpoint, we sum parameter influences from the 64 memorized sequences over that checkpoint and all earlier checkpoints. We select the most negative fractions specified by the ablation script and replace those parameters with zero. Normalization parameters are always excluded. We report separate scopes that exclude embeddings or include them; each fraction is relative to the eligible parameters in its scope. Random masks with the same number of parameters provide a baseline. After intervention, we measure teacher forced greedy accuracy following a 32-token prefix, mean token loss, and the number of sequences for which all 32 continuation tokens are decoded greedily, both for the binary groups and for each of the four classes. We additionally measure control CE and the fraction of greedily generated control tokens changed by the intervention.

## C ADDITIONAL RESULTS

## C.1 TRAINING DYNAMICS AND GRADIENT NORMS ACROSS SCALES

Figure 9 reports loss and extractability for the memorized and control groups across Pythia models from 160M to 6.9B parameters, while Fig. 10 reports BLiMP accuracy over training. Figures 11 and 14 show update magnitudes and test sensitivities over training and by model region, respectively.

Figures 12 and 13 compare alignment with updates from a sequence’s own occurrences, the original training data, and weight decay. Memorized sequences exhibit strong self-alignment, while recited sequences show more negative alignment with other training data and weight decay than recollected sequences. Contributions to self-alignment are broken down by model region in Fig. 15 and layer type in Fig. 16, highlighting the role of lower layers. Figure 17 provides the corresponding layer-wise decomposition for training data and weight decay alignment. Alignment between other injected sequences across scales is shown in Fig. 18.

![](images/c0746426056b9539e753d1cf9b62794d72301929c72f7230e7afa9d4eb70e9fb.jpg)  
(a) Alignment other

![](images/6998e3127c4279b6ee5223b90d3d34a4b5d460393775a6c37e47664f9fc2c470.jpg)  
(b) Decomp. layer type

![](images/32ae2e0600664c831bd7a04e61a132ad456ef7527d0d9e6a4faeb8ba5ce21ec0.jpg)  
(c) Decomp. layer depth  
Figure 8: Alignment with other injected instances. Other models in App. Fig. 18

## C.2 PREDICTION

Figure 19 reports held-out memorization classification accuracy and the coefficients of linear predic tors using decomposition features, alone or with cross-entropy. These features improve prediction over the cross-entropy baseline, particularly in larger models and early in training.

## C.3 CAUSAL PARAMETER INTERVENTIONS

Figure 20 reports the effects of parameter ablation on 32-extractability and control loss across model scales. Figures 21 and 22 show the selected parameters by depth and layer type with embeddings included and excluded, respectively, for the 10, 000-step setting and an ablated fraction of $4 \times 1 0 ^ { - 5 }$ The following section examines attention-head interventions in Figs. 23 and 24.

## C.4 A SHARED MEMORIZATION MECHANISM IN LOWER LAYER ATTENTION HEADS

We also track the alignment of our decomposed sequences with each other and plot the result in Fig. 8. Interestingly, recited instances tend to have a high level of alignment with each other throughout all of training. This trend is also followed by the other instances in early training until around step 64. This alignment suggests that memorization learning in these instances tends towards a shared mechanism. To understand where in the model such a mechanism might be located, we turn to our layer-wise alignment decomposition and find that until around step 20k, the alignment largely stems from attention heads in the lower layers.

Memorization in smaller models is ablated by a single attention head. To further investigate the mechanism that could be implemented there, we turn to our parameter interventions. Due to our implementation, we do not have access to the decomposition on the level of single attention heads, but only entire attention layers. Therefore, one by one, we zero-intervene on the parameters of all the attention heads in the TOP-3 attention layers as ordered by their layer alignment. We observe 32-extractability of the memorized instances and cross-entropy of the control instances and plot the result in App. Fig. 23. While most attention heads tend to have little influence on the memorization behavior of the memorized sequences we study, we find that ablating attention head 0 in layer 0 to have a big effect on memorization reducing the amount of 32-extractable sequences to 5 and 1 for the 64 recited and recollected sequences respectively. At the same time, control cross-entropy increases by around 0.25, leaving large parts of the model intact.

This finding echos results presented by Stoehr et al. (2024), who find that memorization in a small GPT-Neo model can also be ablated by fine-tuning a single attention head. Our result indicates, that this sub-mechanism for memorization develops early in training.Further, we show that this finding generalizes to much larger models around 1B, indicating that it is not just a phenomenon in smaller, more constricted settings. On the other hand, our naive search and intervention did not yield the same result in the larger models starting from 1.4B as can be seen in App. Fig. 24. There, the maximum attention alignment with other sequences moves upwards in the model (see App. Fig. 18), in sum suggesting a more refined and replicated mechanism there.

![](images/4c20c0b2e7f0324e3ef686d31f3958f3da3732b161b8486a79798124859e4d91.jpg)  
Figure 9: Pythia training dynamics across scales. We plot the loss over our sampled sequences and 32-extractability over time.

## C.5 DETAILED DECOMPOSITIONS OF OTHER PYTHIA MODELS

We plot decompositions analogous to the ones shown for Pythia-1B in the main paper.

![](images/6c57545cc4ffd700f43a8654a27e1246a5093bbd597a3fa55ccab2763fcd6879.jpg)  
(a) Pythia-160m

![](images/61876287a141944cc5b563226225d3d2726f87fce7d3990a4b2fda4a94658509.jpg)  
(b) Pythia-410m

![](images/22e830510d58db4c21ea82db85f86592505be0500b65764e40f93c32506ad7a0.jpg)  
(c) Pythia-1b

![](images/bd8c652bd266b2560441647319a1bfd2aa7e2bc174e19c04076fee3be3594922.jpg)  
(d) Pythia-1.4b

![](images/527b4f1b8ebd8e952d7e310d5f734203cefd72434053250bc136cea9eb480c2f.jpg)  
(e) Pythia-2.8b

![](images/1bf388c66dbb8e3e63bd964a7bbfcd819e367f013258f3acd7a566929253a1cd.jpg)  
(f) Pythia-6.9b  
Figure 10: Pythia BLiMP accuracy across scales.

Pythia-160m  
![](images/c3641b58367a6b6178e53983aa47cdb620dc6a95788da040f24b7fc938b8e9a4.jpg)  
(a) Update magn.  
(b) Update magn.  
(c) Test sensitivity  
(d) Test sensitivity<sup>10k</sup> <sup>30k</sup> <sup>50k</sup> <sup>70k</sup> <sup>90k</sup> <sup>110k</sup><sub>Step</sub>  
Figure 11: Pythia update magnitudes and test sensitivities across scales. We plot each at early and later pretraining.

Pythia-160m  
![](images/3d98b2435c6712b9c31b4919c23563dd8db49a40d774ad937a33617f244de68c.jpg)  
(a) Self align. (early)  
(b) Self align.  
(c) Train align. (early)  
(d) Train align.  
Figure 12: Pythia self alignment across scales. We plot early and later training stages.

![](images/052437c477f7f080efc7da3b8a1a8c41c8b7c12a8aa72ebd92bf28eb21cd85e3.jpg)

Pythia-160m  
![](images/f1cdeda3f56a5b9c4264b32d6075001367db35219742c5a959bd63807ccfe7c5.jpg)  
Pythia-410m

![](images/9773bca27737aa535a851e5d0a1f6816f0e36b861d9d5eb8cd4a35e1cfd84bac.jpg)

![](images/f064712eb37cc0c8072bdf188fc4a1da686476903924591375e20072b0ce204c.jpg)  
Pythia-1b (model shown in paper)

![](images/f1432ed8108699f40a6d55e3931faad98f569ee30936162ea6683e8d12eee0a9.jpg)

![](images/08fea25e0053c4d918faa5ccb7d8ee1c0efe05082dbf506981ddaaf4ad3cd16d.jpg)  
Pythia-1.4b

![](images/93b60769416119caeffe7d6eb205f5a68ebf03709496cb308d2788fb598d3930.jpg)

![](images/ee161822bbff7023c3cf9c3e5a3163be129282a5d8dcf97a88a45c5db457dd1a.jpg)  
Pythia-2.8b

![](images/8c1b0ecccde3ea4f460c73f3af10e88765c4ea0f6a5ae221372b80744e8e95fd.jpg)

![](images/02a472a25a838062ec3fd0da34a4154f6720c5a947b4f589819479bf141994fd.jpg)  
Pythia-6.9b

![](images/7ebc293e3274af69a2cdfbefa91c19026cddae6d4c5d7a6945d4cc78bcaa480b.jpg)  
(a) Reg. (early)

![](images/75dcabad96ba408a6db3f5d8bd4e335acc91b518351eb88ea20376a2c590ae51.jpg)  
(b) Reg. alignment  
Figure 13: Pythia weight decay alignment across scales. We plot the alignment of the instances we decompose with the parameter update introduced by weight decay, early and late in training.

Pythia-160m  
![](images/e5b26969a628c7d1ed5209457862420e11e4b3bd62e7bcfee5fb1e472226e7df.jpg)  
Figure 14: Pythia update magnitudes and test sensitivity by model region across scales.

Pythia-160m  
![](images/01a8b7fcb64d0f2fd9992ae5d27b8f3e9e8dbd2092e39f8e1605b40bb78ba7bd.jpg)  
(a) Recited→recited  
(b) recoll.→recoll.  
(c) Contr.→contr. (dup)  
(d) Contr.→contr.     <sub>Step</sub>  
Figure 15: Pythia self alignment by model region across scales.

Pythia-160m  
![](images/be155a302ba26c65719ba844de5c36a86a425b2c4f0813ebb03a1bd0e2c216ad.jpg)  
(a) Recited→recited  
(b) recoll.→recoll.  
(c) Contr.→contr. (dup)  
(d) Contr.→contr.<sup>10k</sup> <sup>30k</sup> <sup>50k</sup> <sup>70k</sup><sub>Step</sub>  
Figure 16: Pythia self alignment by layer type across scales.

Pythia-160m  
![](images/8c99712ac32edea66bb4fff37bd4fb7958e44a566a9442b6c6b94fc5e3e3a8bb.jpg)  
(a) train→recited  
(b) train→recoll.  
(c) w. decay→recited (d) w. decay→recoll.Step  
Figure 17: Pythia train and regularization alignment by model region across scales.

Pythia-160m  
![](images/fbcd53c4e48bad145915b9db83e1ce7320e661f8b465b26f116d8678fd9474a9.jpg)  
(a) Alignment other  
(b) Decomp. layer type  
(c) Decomp. layer depth  
Figure 18: Alignment with other injected instances across scales.

Pythia-160m

![](images/81d307669180fca54644d40529dd99b49a85dedf06ac2de21331b2fa3c910210.jpg)  
(a) Accuracy on held out  
(b) Weights decomp.  
(c) Weights CE+decomp

Figure 19: Memorization vs. not memorization classification across model scales. We also plot the weights of the decomposition features for the decomposition only and CE+decomposition model.

![](images/41e0700e79eab78b6e13223146bd733d0a8c715141fb203cd4d4bfd04022ca0d.jpg)  
Pythia-160m

![](images/5ab040d4da00717744b8e904622f1331d64d39fbe5f4d868abdbc39fe71b9fa0.jpg)

Pythia-410m  
![](images/d002d2f03b80d40a12506d2201cea7debcb829b3581d6899ab42bb2bc7583352.jpg)

![](images/ceca9a2a8819d5aac07b457658deb3fead36efa13f82621168cc869d3ba5124f.jpg)

Pythia-1B  
![](images/0adeda85e98878f0361e23a69b5279a11b88b0ee35e9c9599b0a9c3abbebbb75.jpg)

![](images/a7c9d4a3ad0a3ae0357703cbb6f91fbde0f95e634756f3168e563913fbb39ba2.jpg)

Pythia-1.4B  
![](images/296aa0d367f5a209b7edb20afe9362d2c53f0006cf5426dd3a3de6658659eac4.jpg)

![](images/cbbce28714dcb23ea1cf4d91c04378e359910a388d87b53fd7d8744d96f0eb56.jpg)

Pythia-2.8B  
![](images/60a0c8e64618ae39dc8a9dc909b3a07c703ac65f19185836e2849864cdde706c.jpg)

![](images/db09bfcbec46ae7969e04c21650184fe15bb7343f72d2a3390d6fb15cf32337b.jpg)  
Figure 20: Pythia memorization ablations across scales. Left: Number of 32-extractable sequences in the memorized sample, right: CE loss on the control sample.

![](images/1f638bd756bf9da2e63d1273440912851e1492c949902531a4b4b98247d275e0.jpg)  
Pythia-160m

![](images/fc86d4f6f62901ef680ebdd2a6805f072aca3c8e9c765c4a8b3c0cdedf066151.jpg)  
Pythia-410m

![](images/c83ba83b123e3b5ead616c1a674ff637876519420f0dd06b7569d523c672bfed.jpg)

![](images/ba86f03e16ef259a1a39e4b1118df0d1be6ccd8c4d68401295e7cf306aa82f3c.jpg)  
Pythia-1B

![](images/cf42711213beab2a4432b9163000d530c0b4c75ac8e588a365e3a6e4da6d2bdc.jpg)

![](images/bddb73b55112de112ea2e0b2ca8c504c1108d3afc4c1ecbbb2e276e31f51045e.jpg)  
Pythia-1.4B

![](images/b03d9bfc5f8eb0e0459e3547526fadb6a12e625c3f3b73a8c347b868e666643e.jpg)

![](images/6395bfb4aa8f1e45266da03f67e1f7e30aaa1f04de4fae61d4da3a3b40b2482a.jpg)  
Pythia-2.8B

![](images/cfa213cdd11cb37183d259f69b246b0d5dab10c21439ab53edebed55651f8b64.jpg)  
(a) Ablated parameters by layer depth.

![](images/93366242d8b648ae4631e2d48eaf5b31bb94af712f793bb14fca9fdaac872716.jpg)  
(b) Ablated parameters by layer type.  
Figure 21: Pythia memorization ablations affected parameters, embeddings included. Ablation at step 10, 000 with 0.00004 of all model parameters ablated.

Pythia-1B  
Pythia-410m  
![](images/53f46c487f4ba834386bef8369276121c718d9632c2477a8dbee8fca99738b2c.jpg)

![](images/2dd9f7747cd64ece9e7390ef8f3aab455818b60474f2b1f4cc81599f37c86797.jpg)

![](images/2b7b0f0a011ccad87296d708f7c1d8f4a1e8693cb556faed1538e91f8b7c546b.jpg)

![](images/c3d086a5a98e6169cb344077bec0b7696f04af2a3ec934ccb6175f1f33cf69d6.jpg)

![](images/be8d2ff4f69b950e242ce3cba7b23d2cbcda27d341d69843c3871f0907d477c4.jpg)

![](images/9d3754b290ff825d1cb70190c316939fc34c225b8a2a90e3b4a13f2d64f270a5.jpg)

Pythia-1.4B  
![](images/f6e69d2b2241a3be4eb4fffefbf0d5a0c377571a25a752bcd34bb29c2a599ebd.jpg)

![](images/4fb255e668d5e2116c5482a447688c9a360a5db5b6b1516ef203039a47818fa3.jpg)

![](images/04ef4d87f9a7fe0b00209d6e430a53d9dba26f3e60d847cf8d47bc2aaf6030df.jpg)

![](images/7c7ba6e3b226b505ad69873a45d7ac9e704abb811c35cf542343c47933807376.jpg)  
(a) Number of ablated parameters by layer (b) Number of ablated parameters by layer depth. type.  
Figure 22: Pythia memorization ablations affected parameters, embeddings excluded. Ablation at step 10, 000 with 0.00004 of all model parameters ablated.

![](images/6bc9814bd455e88d5fb5ca79bb763d5e1c676b44169111b730e0cd17c4d50c3c.jpg)  
(a) 32-extr. recited

Pythia-160m  
![](images/717840c7353bc4f602edaf27cc9086a353866bc969295895f240d0c4d52e1d53.jpg)

![](images/01e9a38b38110ff7734b43d1ef0d9cd6a16e40602ed1d9d64a736bb51823fe6e.jpg)  
(c) CE on control

(b) 32-extr. recollected Pythia-410m  
![](images/94bc84bed199d1e3fc32a2c6f4a185db35638f72e47bba915d2ea7fec041e3eb.jpg)  
(d) 32-extr. recited

![](images/a916a4014a2616451d19cc463d2f093423a4268998474414ea51a39887fbbb53.jpg)

![](images/635737793f589c1ab2369dac4c7d133dd21b1cde12348dc19d455113a9c3dfd8.jpg)  
(f) CE on control

![](images/aad54879bba6f387f595e160d8fb14686a7733aec153eafb96faf8f362828bdb.jpg)  
(g) 32-extr. recited

(e) 32-extr. recollected Pythia-1B  
![](images/35ea72f629f2e13fc993584606e87e365a963267a40aadd5bb6f255711953c47.jpg)  
(h) 32-extr. recollected

![](images/49b2f2363402c3cd60283db28eeea6435915c50d0b01860b90264e050ff6faf7.jpg)  
(i) CE on control  
Figure 23: Attention head ablation in Pythia (1/2). We ablate each attention head, one by one, and plot number of 32-extractable memorized sequences and CE on control examples in intervened model.

![](images/dac1601b2479b4a3f7473e7ecee4504d94d788fe031896197c2d3b1821664d26.jpg)  
(a) 32-extr. recited

![](images/af96503f15ea3fb998ea6ca9d7ebdd980d1066aab92bf2e6036e8d055557fffc.jpg)  
(b) 32-extr. recollected

![](images/e8edb4c4f6188968e7f57b8647a747529140198df180a8c63d3f53461f0f27bf.jpg)  
(c) CE on control

![](images/2e4821f9f192ad7b4472efc01335c0c868ab19f4915140c2467aec7f32201217.jpg)  
(d) 32-extr. recited

![](images/c99f2be689667d6dc2c00ec938c120c8d26aac87d45bad3b8bd71f590399bec2.jpg)  
(e) 32-extr. recollected

![](images/e01f75b0cf96d5d173117e9bad8ab055fa6e7d2e2e3126f7eb0f77d89ac0bf85.jpg)  
(f) CE on control

![](images/88c4e96c9d227b63b0558a426741984b222429a5f02444d70efc398bc139d83f.jpg)  
(g) 32-extr. recited

![](images/c9aa0012a0638f8cac6c6f6e93a81d122b0b84aac31d02822f8ab456f62c2bdb.jpg)  
(h) 32-extr. recollected

![](images/5f9eb24a4ed2b3b05c026bbf4225c02c27f49eac30f2e1be916c56e258de69fd.jpg)  
(i) CE on control  
Figure 24: Attention head ablation in Pythia (2/2). We ablate each attention head, one by one, and plot number of 32-extractable memorized sequences and CE on control examples in intervened model.