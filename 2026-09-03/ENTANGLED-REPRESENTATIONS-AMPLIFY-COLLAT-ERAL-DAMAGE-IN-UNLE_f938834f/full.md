# ENTANGLED REPRESENTATIONS AMPLIFY COLLAT-ERAL DAMAGE IN UNLEARNING

Evžen Wybitul University of Oxford

Tim G. J. Rudner University of Toronto

Christian Schroeder de Witt University of Oxford

## ABSTRACT

A long-held intuition in interpretability research is that representational entanglement—the sharing of structure between knowledge domains in a neural network—makes unlearning harder. While the intuition is widespread, it has never been directly tested in a controlled experiment. We present a way to do so: by repurposing Selective Gradient Masking (SGTM), we train a suite of six 254M-parameter language models on English Wikipedia with graded levels of disentanglement between biology and non-biology knowledge. Applying three standard unlearning methods to every model in the suite, we find that more disentangled models consistently achieve better retain–forget trade-offs: at a fixed level of forgetting, the most disentangled models incur roughly 4× lower retain cost under two of the three methods, and 1.3× lower under the third. Because our intervention changes only the model, not the data or the unlearning algorithm, this is direct evidence that representational entanglement is one of the causes of collateral damage in unlearning, as interpretability researchers have long suspected. A similar design could be used to test other structural claims from interpretability.

![](images/792512f1e4111e5e45a476f9a597c09ff4961e9390e601c71bb62cc885cd2aa1.jpg)

![](images/fa189c22b72fae89a3d8b3b70a7e937000c8d462e16df33960bfcf8a9aa4860a.jpg)

![](images/32c6bcca3f0e23acd5a5e888d0242c65ea55c71d9ed20be883640e12b4b444a5.jpg)  
p=0% p=20% p=40% p=60% p=80% p=100%  
Figure 1: The more disentangled the retain and forget domains are in a model, the better the retain–forget trade-off achieved by unlearning. Retain–forget Pareto frontiers of three unlearning methods (WGA, WDR, RMU) across six models that differ in the fraction of SGTM training steps (p). Models are coloured by their Variance Entanglement Score (defined in Appendix C, shown in Figure 2): purple means less disentangled, yellow more. Each frontier is constructed over four to six hyperparameter configurations; shaded regions show ±1 standard error across five seeds. Both axes are measured relative to each model’s own pre-unlearning loss. Analysed in Section 3.2.

## 1 INTRODUCTION

Broadly speaking, interpretability research is useful to the extent that it predicts what happens when we intervene on a network. Unlearning is a natural test case: interpretability has proposed two structural properties of neural networks that should, intuitively, affect how cleanly a targeted piece of knowledge can be unlearned. The first is localization: where in the network’s parameters the knowledge resides. The second is entanglement: the degree to which the retain and forget domain share structure in the network—representations, processing pathways, or parameters. Both of these intuitions have taken a foothold in the unlearning literature (Barez et al., 2025), but only localization has so far been subjected to rigorous, controlled experimental testing (Lee et al., 2025; Guo et al., 2025; Boglioni et al., 2026). In this paper, we apply similar experimental methods to entanglement.

According to the entanglement intuition, high entanglement should make the retain–forget trade-off worse: if the knowledge to be forgotten and the knowledge to be retained share a lot of structure in the network, removing one without damaging the other should be harder. However, this has never been tested directly. Of the three main ingredients that go into unlearning—the model, the data, and the unlearning algorithm—current literature investigating entanglement has always held the model fixed, and tested the effect of entanglement only indirectly by varying the data composition (Zhao et al., 2024) or the unlearning algorithm (Sondej & Yang, 2025; Tang & Khanna, 2026; Chen et al., 2026), leaving model-level entanglement confounded with other factors. The perhaps most natural way to test whether entanglement affects unlearning, namely, holding the datasets and algorithm fixed and varying only the degree of entanglement in the model, has so far not been attempted—in part because it requires a way to control entanglement rather than merely observe it.

Our main contribution is to fill this gap and do for entanglement what recent work has done for localization: build a controlled experiment that tests this intuition as directly as possible. We train a suite of six 254M-parameter language models on English Wikipedia (Wikimedia, 2025), increasingly disentangling biology knowledge from the rest using Selective Gradient Masking (SGTM), an improved variant of gradient routing (Shilov et al., 2025; Cloud et al., 2024), and verify with three metrics that the suite spans a graded range of disentanglement. On each model in the suite, we measure the retain–forget trade-off of three common unlearning methods: Weighted Gradient Ascent (WGA) (Wang et al., 2025), Weight Divergence Regularization (WDR) (Siddiqui et al., 2025), and RMU (Li et al., 2024). We find that more disentangled models consistently achieve better retain–forget trade-offs (Figure 1): at a fixed level of forgetting, the most disentangled models incur roughly 4× lower retain cost than the most entangled one under WGA and RMU, and 1.3× lower under WDR.

We want to be precise about what this establishes. Even in our experiments, we cannot change entanglement alone: we alter the training procedure, inducing changes to entanglement and, potentially, to other properties of the model. Nevertheless, the design rules out the data and algorithm confounds of earlier work. Taken together with the indirect evidence of earlier work, our results turn a long-standing intuition from interpretability into an empirical finding.

## 2 EXPERIMENTAL SETUP

## 2.1 TRAINING MODELS WITH VARYING DISENTANGLEMENT

To train models in which the forget and retain domains are disentangled, we use a variant of gradient routing (Cloud et al., 2024) called Selective Gradient Masking (SGTM) (Shilov et al., 2025). Although SGTM is a training method originally designed to make a group of parameters specialize on processing inputs from a target domain, we find we can repurpose it for training disentangled models. Namely, when SGTM is used to specialize a group of parameters on the forget domain, the forget domain also grows more disentangled from retain, as we show in Section 3.1. We exploit this fact by varying the fraction of training steps in which we apply SGTM, producing models with different levels of disentanglement.

Data. Following the original SGTM paper, we train on English Wikipedia (Wikimedia, 2025) (approximately 3.7B tokens), using article-level topic labels derived from Wikipedia’s articletopic classifier (Wikimedia, 2025), and splitting the data into three domains: (1)forget: all articles classified as STEM.Biology (∼3.7% of training tokens); (2) adjacent: articles on topics closely related to biology (Medicine & Health, Chemistry, Earth & Environment); (3) retain: all remaining articles, spanning Culture, Geography, History & Society, and the STEM topics unrelated to biology.

Each example is a 1,024-token block of article text. We hold out separate test sets for each domain. To reduce evaluation cost during our unlearning hyperparameter sweeps, we subsample 20% of each test set and report all unlearning metrics on this fixed subsample.

<table><tr><td>Routing category</td><td>Forward pass</td><td>Backward pass</td><td>Updated params</td></tr><tr><td>route-bio</td><td>(standard)</td><td> $\mathrm { Z e r o } \ \nabla _ { \theta _ { \mathrm { o t h e r } } }$ </td><td> $\theta _ { \mathrm { b i o } } \ \mathrm { o n l y }$ </td></tr><tr><td>route-other</td><td>Set  $\theta _ { \mathrm { { b i o } } } = 0$ </td><td>(standard)</td><td> $\theta _ { \mathrm { o t h e r } } \mathrm { o n l y } ^ { \dagger }$ </td></tr><tr><td>route-unchanged</td><td>(standard)</td><td>(standard)</td><td> $\theta _ { \mathrm { b i o } } \mathrm { a n d } \theta _ { \mathrm { o t h e r } }$ </td></tr></table>

Table 1: Training interventions in SGTM, applied to each example based on its routing category. $\dag \theta _ { \mathrm { b i o } }$ receives no gradient because its activations are zeroed during the forward pass.

Models. Following the original paper, all models we train are 254M-parameter GPT-Neo-style transformers with 16 layers, hidden dimension 1,024, 32 attention heads, and MLP dimension 4,096; the full configuration is given in Appendix A. In each transformer block, we designate a small subset of parameters as the biology subnetwork, $\theta _ { \mathrm { b i o } } \colon$ 1 attention head (out of 32) and 64 MLP hidden units (out of 4,096) per block. We denote all remaining parameters by $\theta _ { \mathrm { o t h e r } }$

SGTM. As in the original paper, we assign training examples to one of three routing categories: (1) route-bio: all forget (i.e., all biology) examples; (2) route-other: a randomly sampled 10% of retain and adjacent examples; (3) route-unchanged: the remaining 90% of retain and adjacent examples.

Depending on an example’s routing category, SGTM modifies the training procedure during both the forward and backward passes. For route-bio examples, gradients for $\theta _ { \mathrm { o t h e r } }$ are zeroed after the backward pass, ensuring that biology knowledge flows only into $\theta _ { \mathrm { { b i o } } } . ^ { 1 }$ For route-other examples, $\theta _ { \mathrm { b i o } }$ is zeroed during the forward pass, training the model to perform well on non-target data even without the biology subnetwork. Finally, route-unchanged examples update all parameters via standard training. See Table 1 for an overview.

Varying the fraction of SGTM steps. In the original paper, SGTM was used in all training steps; we instead activate SGTM for only a fraction of the training steps, p%. We train six models with $p \in \{ 0 , 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ ; the first (100−p)% of steps use standard training, and the remaining p% use SGTM as described in the previous paragraph.<sup>2</sup>

## 2.2 MEASURING DISENTANGLEMENT BETWEEN DOMAINS

We measure disentanglement separately between all pairs of the three domains defined in Section 2.1: forget–retain, forget–adjacent, and, as a control, retain–adjacent. Given a pair of domains, we sample 1,024 examples from their two test sets, then for each example we take the model’s final-layer hidden states, average them over non-padding positions, and normalize the result to unit norm, yielding one vector per example. We then compare the resulting two point clouds using three standard metrics: the Variance Entanglement Score (VES) (Zhao et al., 2024), maximum mean discrepancy $( \mathrm { { M M D } ^ { 2 } ) }$ (Gretton et al., 2012), and the sliced 2-Wasserstein distance $( \mathrm { S W _ { 2 } ^ { 2 } } )$ (Bonneel et al., 2015). Lower VES and higher MMD<sup>2</sup> and $\mathrm { S W _ { 2 } ^ { 2 } }$ indicate more disentanglement; definitions and settings are given in Appendix C.

The three metrics are not on a common scale, and their absolute values are not comparable across domain pairs. We therefore report, for each metric and each pair, the log ratio of the measured value to that of the p=0% model, so that every curve starts at zero and the quantity plotted is the relative change in separation induced by SGTM.

## 2.3 MEASURING THE RETAIN–FORGET TRADE-OFF DURING UNLEARNING

We apply three standard unlearning algorithms to each of the six models, using the forget and retain training data defined in Section 2.1. Adjacent data is never used for optimization, only for evaluation. We do not give the algorithms any information about how the models were trained.

Our interest is not in comparing methods against each other, but in checking whether, within each method, more disentangled models yield better retain–forget trade-offs. Since the six models differ in their losses before unlearning, we measure the change in test loss relative to each model’s own preunlearning value, $\Delta \ell _ { \mathrm { f o r g e t } }$ and $\Delta \ell _ { \mathrm { r e t a i n } }$ . For each method, we sweep over four to six hyperparameter configurations (see Appendix B for the full list), and use the results to construct a retain–forget Pareto frontier for every model–method pair. We run each configuration on five seeds for a method-specific number of optimizer steps, with early stopping when the forget loss exceeds that of a model trained with the biology data filtered out, plus a small buffer.<sup>3</sup>

Unlearning methods. Let $\operatorname { C E } ( \theta ; { \mathcal { D } } )$ denote the mean per-token cross-entropy of model θ on dataset $\mathcal { D } _ { : }$ , and let $\theta _ { 0 }$ denote the parameters before unlearning.

• Weighted Gradient Ascent (WGA) (Wang et al., 2025) combines gradient ascent on the forget set with gradient descent on the retain set, minimizing

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { W G A } } ( \theta ) = \mathrm { C E } ( \theta ; \mathrm { r e t a i n } ) - \lambda _ { \mathrm { W G A } } ~ \mathrm { C E } _ { \mathrm { W G A } } ( \theta ; \mathrm { f o r g e t } ) , \quad \lambda _ { \mathrm { W G A } } > 0 , } \end{array}
$$

where $\mathrm { C E } _ { \mathrm { W G A } }$ is a reweighted cross-entropy that scales each token’s log-probability loss by $[ p _ { \theta } ( x _ { t } \mid x _ { < t } ) ] ^ { \alpha }$ , upweighting tokens the model predicts confidently so that gradient ascent targets the most well-learned predictions first. Each unlearning step draws one batch from the forget set and one from the retain set.

• Weight Divergence Regularization (WDR) (Siddiqui et al., 2025) fine-tunes on the retain set while maximizing the $L _ { 2 }$ distance between the current parameters and $\theta _ { 0 } { } ^ { \prime }$

$$
\mathcal { L } _ { \mathrm { W D R } } ( \theta ) = \mathrm { C E } ( \theta ; \mathrm { r e t a i n } ) - \lambda _ { \mathrm { W D R } } \sqrt { \frac { 1 } { | \theta | } \sum _ { i } ( \theta _ { i } - \theta _ { 0 , i } ) ^ { 2 } } , \quad \lambda _ { \mathrm { W D R } } > 0 .
$$

• Representation Misdirection for Unlearning (RMU) (Li et al., 2024) modifies intermediate representations so that, on forget examples, activations at a designated layer are steered towards random target directions, while on retain examples, the same activations are anchored to their values under $\theta _ { 0 }$ . The loss is defined at a single layer, but the update is applied to the MLP weights of that layer and the two preceding it.

## 3 RESULTS

## 3.1 MODELS TRAINED WITH MORE SGTM ARE MORE DISENTANGLED

We first verify that varying the fraction of SGTM training steps does produce models with meaningfully different levels of disentanglement. All three metrics we use (see Section 2.2) agree that it does: as p grows, the forget domain separates from both retain and adjacent (Figure 2). The largest change happens between $\scriptstyle p = 4 0 \%$ and $p { = } 6 0 \%$ , and the models can be grouped into a less disentangled cluster $( p \leq 4 0 \% )$ and a more disentangled one $( p \geq 6 0 \% )$ ).

As a form of control, we also measure the disentanglement between retain and adjacent. Recall that these two domains were grouped together during SGTM training (Section 2.1), so we would expect their disentanglement to stay roughly constant across the model suite. Instead, we observe that their disentanglement does grow with $p ,$ but roughly three to four times less than in forget–retain. We suspect this reflects the absorption effect (Cloud et al., 2024): adjacent articles contain biology content that was never labelled as such, so that SGTM reshapes adjacent representations more than retain ones, and the two slightly drift apart. Consistent with this, forget–adjacent disentangles about two-thirds as much as forget–retain.

## 3.2 MORE DISENTANGLED MODELS ACHIEVE BETTER RETAIN–FORGET TRADE-OFFS

More disentangled models consistently achieve better retain–forget trade-offs, and this holds for all three unlearning methods (Figure 1). The size of the effect differs substantially between methods, but its direction does not.

![](images/e697011617813b805215e844a37b6135aac8ff84a998538eebe01ab972c8d5d3.jpg)

![](images/55e9b5d1082316fa8a4f2a2275a78e52dda71476274dcd0306562d90f27a2961.jpg)

![](images/1474dd06df4a192fba06a13daf8c83fdbe988419f5f9ac8e6b3a7d9d5eb840fb.jpg)  
Forget & Retain Forget & Adjacent Control (Retain & Adjacent)  
Figure 2: The forget domain grows more disentangled from both retain and adjacent as the fraction of SGTM steps during training (p) increases, with the largest change occurring between $\scriptstyle p = 4 0 \%$ and $\scriptstyle { p = 6 0 \% }$ . The disentanglement of the retain–adjacent control also grows, but three to four times less. Each curve shows the log ratio of the measured value to that of the $\scriptstyle { p = 0 \% }$ model, so all curves start at zero by construction, and the arrow on each axis gives the direction of increasing disentanglement. The three metrics are described in Appendix C.

Retain–forget results. All comparisons below are made at $\Delta \ell _ { \mathrm { f o r g e t } } = 0 . 4 .$

• WGA: The effect is clearest here. The most disentangled models $( p \geq 6 0 \% )$ incur roughly 4× lower retain cost than the most entangled $( p \mathrm { { = } 0 \% ) }$ model, which has the highest retain cost at every level of forgetting. The frontiers also separate into the same two clusters we found in Section 3.1: the less disentangled models $( p \leq 4 0 \% )$ form one, and the more disentangled ones $( p \geq 6 0 \% )$ the other.

• WDR: The trend is in the same direction but weaker. The most disentangled models incur roughly $1 . 3 \times$ lower retain cost than $\scriptstyle { p = 0 \% }$ , but the frontiers sit close together and their standard errors overlap.

• RMU: The retain-loss scale is two orders of magnitude smaller than for WDR, reflecting RMU’s minimal impact on retain performance overall. However, the effect persists even at this scale: the most disentangled models incur roughly 4× lower retain cost than $\scriptstyle p = 0 \%$ . Again, the trend holds, except for $\scriptstyle p = 2 0 \%$ , which performs comparably to the disentangled cluster.

Adjacent–forget results. The adjacent–forget trade-off is the hardest case in our setup. First, being the domain closest to forget, adjacent receives greater spillover damage from unlearning. Second, the unlearning methods optimize on forget and retain data only, so adjacent knowledge is left undefended. Even so, we observe the same ordering on adjacent as on retain for WGA and WDR; for RMU the results are less clear. We report these results with additional discussion in Appendix D.

## 4 CONCLUSION

Interpretability has lent two intuitions to the field of unlearning: that the difficulty of unlearning depends on where the forget knowledge is stored (localization) and on how much structure it shares with retained knowledge (entanglement). While the first has been rigorously tested in a controlled experimental setting, the second had not. In this paper, we fill this gap: holding the data and the unlearning algorithms fixed and varying only the model, we find that more disentangled models consistently achieve better retain–forget trade-offs across three unlearning algorithms. Analogously to the existing work on localization, our work helps turn another long-standing intuition from interpretability into an empirical finding, and the recipe—repurpose an off-the-shelf method to control a representational property during training, verify the control, and test its predicted consequence— could be applied to other structural claims from interpretability as well.

## REFERENCES

Fazl Barez, Tingchen Fu, Ameya Prabhu, Stephen Casper, Amartya Sanyal, Adel Bibi, Aidan O’Gara, Robert Kirk, Ben Bucknall, Tim Fist, Luke Ong, Philip Torr, Kwok-Yan Lam, Robert Trager, David Krueger, Sören Mindermann, José Hernandez-Orallo, Mor Geva, and Yarin Gal. Open problems in machine unlearning for ai safety, 2025. URL https://arxiv.org/abs/2501.04952.

Matteo Boglioni, Thibault Rousset, Siva Reddy, Marius Mosbach, and Verna Dankers. LACUNA: A testbed for evaluating localization precision for LLM unlearning. In Third Conference on Language Modeling, 2026. URL https://openreview.net/forum?id=BFWTLrAJNZ.

Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and radon wasserstein barycenters of measures. Journal of Mathematical Imaging and Vision, 51(1):22–45, Jan 2015. ISSN 1573-7683. doi: 10.1007/s10851-014-0506-3. URL https://doi.org/10.1007/ s10851-014-0506-3.

Hang Chen, Jiaying Zhu, Xinyu Yang, and Wenya Wang. CLUE: Conflict-guided localization for LLM unlearning framework. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=jtRYvazBWv.

Alex Cloud, Jacob Goldman-Wetzler, Evžen Wybitul, Joseph Miller, and Alexander Matt Turner. Gradient routing: Masking gradients to localize computation in neural networks, 2024. URL https://arxiv.org/abs/2410.04332.

Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal ofMachine Learning Research, 13(25):723–773, 2012. URL http://jmlr.org/papers/v13/gretton12a.html.

Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, and Gintare Karolina Dziugaite. Mechanistic unlearning: Robust knowledge unlearning and editing via mechanistic localization. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025. URL https: //proceedings.mlr.press/v267/guo25k.html.

Hwiyeong Lee, Uiji Hwang, Hyelim Lim, and Taeuk Kim. Does localization inform unlearning? A rigorous examination of local parameter attribution for knowledge unlearning in language models. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, pp. 21857–21869. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.EMNLP-MAIN.1109. URL https://doi.org/10.18653/v1/2025.emnlp-main.1109.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Gabriel Mukobi, Nathan Helm-Burger, Rassin Lababidi, Lennart Justen, Andrew Bo Liu, Michael Chen, Isabelle Barrass, Oliver Zhang, Xiaoyuan Zhu, Rishub Tamirisa, Bhrugu Bharathi, Ariel Herbert-Voss, Cort B Breuer, Andy Zou, Mantas Mazeika, Zifan Wang, Palash Oswal, Weiran Lin, Adam Alfred Hunt, Justin Tienken-Harder, Kevin Y. Shih, Kemper Talley, John Guan, Ian Steneker, David Campbell, Brad Jokubaitis, Steven Basart, Stephen Fitz, Ponnurangam Kumaraguru, Kallol Krishna Karmakar, Uday Tupakula, Vijay Varadharajan, Yan Shoshitaishvili, Jimmy Ba, Kevin M. Esvelt, Alexandr Wang, and Dan Hendrycks. The WMDP benchmark: Measuring and reducing malicious use with unlearning. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview. net/forum?id=xlr6AUDuJz.

Igor Shilov, Alex Cloud, Aryo Pradipta Gema, Jacob Goldman-Wetzler, Nina Panickssery, Henry Sleight, Erik Jones, and Cem Anil. Beyond data filtering: Knowledge localization for capability removal in llms, 2025. URL https://arxiv.org/abs/2512.05648.

Shoaib Ahmed Siddiqui, Adrian Weller, David Krueger, Gintare Karolina Dziugaite, Michael C. Mozer, and Eleni Triantafillou. From dormant to deleted: Tamper-resistant unlearning through

weight-space regularization. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Iván Vladimir Meza Ruíz, and Arturo Loaiza-Bonilla (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/hash/ bbeafbaa60ee03f2c3135be71ebd5d06-Abstract-Conference.html.

Filip Sondej and Yushi Yang. Collapse of irrelevant representations (cir) ensures robust and nondisruptive llm unlearning, 2025. URL https://arxiv.org/abs/2509.11816.

Haoran Tang and Rajiv Khanna. From logits to latents: Contrastive representation shaping for llm unlearning, 2026. URL https://arxiv.org/abs/2601.22028.

Qizhou Wang, Jin Peng Zhou, Zhanke Zhou, Saebyeol Shin, Bo Han, and Kilian Q. Weinberger. Rethinking LLM unlearning objectives: A gradient perspective and go beyond. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=huo8MqVH6t.

Wikimedia. Wikimedia downloads, 2025. URL https://dumps.wikimedia.org.

Wikimedia. Ores / articletopic. https://www.mediawiki.org/wiki/ORES/ Articletopic, 2025. Accessed: 2025-09-21.

Kairan Zhao, Meghdad Kurmanji, George-Octavian Barbulescu, Eleni Triantafillou, and Peter Triantafillou. What makes unlearning hard and what to do about it. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ 16e18fa3b3add076c30f2a2598f03031-Abstract-Conference.html.

## A TRAINING HYPERPARAMETERS

See Table 2 for the full training hyperparameters.

## B UNLEARNING HYPERPARAMETERS

All unlearning methods use the AdamW optimizer with no weight decay. Each configuration is run with five seeds (42–46). Full configurations are given in Tables 3 to 5.

## C DISENTANGLEMENT METRICS

We compare the two point clouds described in Section 2.2 using the following three metrics.

• Variance Entanglement Score (VES) (Zhao et al., 2024) compares within-group spread to between-group separation. Lower VES indicates more disentanglement.

• Maximum Mean Discrepancy (MMD<sup>2</sup>) (Gretton et al., 2012) is a standard nonparametric measure of the difference between two distributions. It maps both clouds into a reproducing kernel Hilbert space and measures the squared distance between their mean embeddings. We use the unbiased estimator and a sum of five Gaussian kernels with bandwidths $\sigma ~ \in ~ 0 . 8 2$ {0.5, 0.71, 1, 1.41, 2}. Higher MMD<sup>2</sup> indicates more disentanglement.

• Sliced 2-Wasserstein (SW<sup>2</sup>) (Bonneel et al., 2015) is an optimal-transport distance between distributions, and the standard alternative to the 2-Wasserstein distance in high dimensions, where the latter is difficult to estimate reliably. It averages the squared 2-Wasserstein distance between projections of the two clouds onto random directions. We use 512 fixed random projections. Higher $\mathrm { S W _ { 2 } ^ { 2 } }$ indicates more disentanglement.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Parameters</td><td>254M</td></tr><tr><td>Layers</td><td>16</td></tr><tr><td>Hidden dimension (d)</td><td>1,024</td></tr><tr><td>Attention heads (h)</td><td>32</td></tr><tr><td>MLP dimension  $( d _ { \mathrm { M L P } } )$ </td><td>4,096</td></tr><tr><td>Context size</td><td>1,024</td></tr><tr><td>Vocabulary size</td><td>50,257</td></tr><tr><td>Tied embeddings</td><td>True</td></tr><tr><td>Tokenizer</td><td>GPT-2</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td>Cosine annealing with warmup</td></tr><tr><td>Warmup steps</td><td>1,000</td></tr><tr><td>Total training steps</td><td>9,689</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td> $\beta _ { 1 }$ </td><td>0.9</td></tr><tr><td> $\beta _ { 2 }$ </td><td>0.95</td></tr><tr><td>Forget MLP dim  $( d _ { \mathrm { f o r g e t } } )$ </td><td>64</td></tr><tr><td>Forget attention heads  $\left( h _ { \mathrm { f o r g e t } } \right)$ </td><td>1</td></tr><tr><td>Confident retain fraction</td><td>10%</td></tr><tr><td>Mask embeddings</td><td>True</td></tr></table>

Table 2: Training hyperparameters, following Shilov et al. (2025).

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Fixed</td><td>50</td></tr><tr><td>Total steps</td><td></td></tr><tr><td>Eval every</td><td>5 steps</td></tr><tr><td>α</td><td>0.1</td></tr><tr><td>Retain loss coefficient</td><td>1.0</td></tr><tr><td>Swept</td><td></td></tr><tr><td>Learning rate</td><td> $\{ 1 . 2 5 \times 1 0 ^ { - 5 } , $   $2 . 5 \times 1 0 ^ { - 5 }$   $5 \times 1 0 ^ { - 5 } \}$ </td></tr><tr><td>Forget loss coefficient  $\left( \lambda _ { \mathrm { W G A } } \right)$ </td><td>{0.128, 0.32}</td></tr></table>

Table 3: WGA hyperparameters (six configurations).

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Fixed</td></tr><tr><td>Total steps</td><td>100</td></tr><tr><td>Eval every</td><td>10 steps</td></tr><tr><td>Retain loss coefficient</td><td>1.0</td></tr><tr><td>Swept</td></tr><tr><td>Learning rate</td><td> $\{ 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 }$   $4 \times 1 0 ^ { - 4 } \}$ </td></tr><tr><td>Divergence coefficient  $\left( \lambda _ { \mathrm { W D R } } \right)$ </td><td>{3, 30}</td></tr></table>

Table 4: WDR hyperparameters (six configurations).

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Fixed</td><td></td></tr><tr><td>Total steps</td><td>100</td></tr><tr><td>Eval every</td><td>5 steps</td></tr><tr><td>Steering coefficient</td><td>20</td></tr><tr><td>Retain anchoring coefficient</td><td>100</td></tr><tr><td>Target layer</td><td>7</td></tr><tr><td>Updated layers</td><td>5,6,7</td></tr><tr><td>Updated parameters</td><td>MLP weights and biases</td></tr><tr><td>Max gradient norm</td><td>1.0</td></tr><tr><td>Early stopping min steps</td><td>20</td></tr><tr><td>Swept</td><td></td></tr><tr><td>Learning rate</td><td> $\{ 1 . 2 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 8 \times 1 0 ^ { - 4 } , 6 . 4 \times 1 0 ^ { - 3 } \}$ </td></tr></table>

Table 5: RMU hyperparameters (four configurations).

## D UNLEARNING TRADE-OFFS ON ADJACENT KNOWLEDGE

Figure 3 shows the Pareto frontiers in the $( \Delta \ell _ { \mathrm { a d j a c e n t } } , \Delta \ell _ { \mathrm { f o r g e t } } )$ plane, constructed independently of the retain–forget frontiers of Figure 1. As in Section 3.2, all comparisons are made at $\bar { \Delta t } _ { \mathrm { f o r g e t } } = \mathrm { \bar { 0 . 4 } }$

• WGA: The ordering from Section 3.2 is reproduced, including the separation into two clusters. The most entangled model $( p \mathrm { { = } 0 \% ) }$ incurs roughly 4× the adjacent cost of the disentangled cluster $( p \geq 6 0 \% )$ .

• WDR: The ordering is also reproduced, although, as in the retain–forget results, the frontiers are close together and their standard errors overlap. The means are ordered correctly across the suite.

• RMU: The frontiers largely overlap and we find no clear ordering. We do not have a conclusive explanation, but note that RMU’s costs are by far the smallest of the three methods. We therefore take our results to mean that we cannot resolve an ordering at this scale of adjacent loss, rather than that there is none.

![](images/a1d05dcd0da2fbcc32c30a1140ce68faae2ee8d677d36d47760eccacf2036b69.jpg)

![](images/5182bd23e59905613aba822f31b44777d2ddb1e687ac556e5ab816ce27cb87fa.jpg)

![](images/598a5bcdbca82a1e3af31deccea3051cbce14ce9e4a6cd95b97627dba4cb4b82.jpg)  
Figure 3: Adjacent–forget Pareto frontiers, reproducing the ordering of Figure 1 for WGA and WDR but not for RMU. No unlearning method optimizes on adjacent data. Conventions are otherwise as in Figure 1.