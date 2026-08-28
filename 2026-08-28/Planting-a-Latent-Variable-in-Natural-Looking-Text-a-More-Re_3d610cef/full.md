# Planting a Latent Variable in Natural-Looking Text: a More Realistic Test of Belief States in LLMs and Their Link to Concept Geometry

Alexandru-Iulius Jerpelea Columbia University aij2115@columbia.edu

## Abstract

LLMs are thought to track “belief states,” i.e., running probability distributions over the latent variables that govern language (Shai et al., 2024; Sarfati et al., 2026), but so far this has only been comprehensively demonstrated on toy synthetic data and in a few isolated case studies. It has also never been empirically connected to the geometry of LLM features (the concepts interpretability finds in model activations). In this work, we plant a controllable latent variable inside natural-looking text. An LLM teacher writes ordinary text while we “subliminally” steer<sup>1</sup> it along one of K = 8 unrelated sparse autoencoder directions at each token, with the active directions following a ringshaped Markov chain. A small transformer model trained on this corpus does indeed track the Bayesian posterior belief about our planted latent variable. Moreover, it also arranges the 8 states themselves on a ring, in the exact order of the Markov chain, which is supporting evidence that a concept’s geometry can be formed by the statistical dynamics of the latent variable behind it.

## 1 Introduction

It is thought that natural language is governed by a set of latent variables, so a good predictor of language must infer their state, gaining a sort of world model.<sup>2</sup> Shai et al. (2024) train small transformers on token sequences emitted by a Hidden Markov Model (HMM), and find that the residual stream linearly encodes a belief state, i.e., a running probability distribution over the state of the HMM, updating at each token.

But the paper’s setup is purely synthetic, as the HMM dictates the whole toy language, and just because transformers can model belief states, it does not mean they will in a real natural language setting.<sup>3</sup> A realistic controlled experiment is hard to set up as we can’t model English with a bunch of Markov Models. LLMs would not be needed if we could.

![](images/4d1602a3bd5faaba792d91bdcf0cb4c2f3f0230db86431d2bb3e5354aaf08094.jpg)  
Figure 1: Figure from Shai et al. (2024), showing how given a data-generating process, a model learns to model optimal beliefs sitting on the probability simplex at any token.

In parallel, the interpretability literature is documenting the geometry of concepts. The Linear Representation Hypothesis (Park et al., 2023), which proposes features as linear directions, has been weakened to allow concepts to sit on lowdimensional manifolds, like days of the week on a circle, or calendar years on a helix (Engels et al., 2024). There is no full proof of why these concept manifolds form.

In a follow-up paper, Shai et al. (2026) show that when text is modeled by multiple HMMs at once (not just a singular latent variable as before, but multiple at once), the transformer learns a belief state per factor, each living in its own nearorthogonal subspace. The authors suggest that multi-dimensional feature manifolds could be stemming from here. However, this requires further explanation. First of all, their models are once again trained on synthetic HMM tokens, which is not a sufficient argument. Second of all, probing the beliefstate, which gives a probability distribution over a latent variable’s values, is not the same as asking how the values themselves that the latent variable takes are geometrically arranged.

![](images/6a9227f595feb2a95eb258e8c96a7c85443fd260c9198d76bdee3538ae1e9961.jpg)  
Figure 2: Belief state geometry (each point a probability vector over the latent variable’s values) versus concept geometry (the arrangement of the values themselves), both read from the residual stream. The two are different objects.

Figure 2 better illustrates the difference between belief states and concepts/features within the residual stream. It is unclear whether belief dynamics are actually correlated with concept geometry.

This paper tries to tie both ends by introducing a method for generating natural-looking text influenced by a latent variable with a Markov structure of our choosing. We use an LLM teacher that writes ordinary text while we “subliminally” steer it (Morgulis and Hewitt, 2026) along one of K = 8 unrelated and orthogonal sparse autoencoder directions. Switching from one autoencoder direction to another is dictated by a ring-shaped Markov chain that we fully control. We then train a small transformer from scratch on the generated corpus and ask two questions. Does it track the belief state of our variable? (Yes, confirming the Shai et al. (2026) hypothesis in a more realistic scenario). And do the 8 states themselves inherit the geometry of the Markov Model? (Yes, they sit on a ring, in the exact neighbor order).

## 2 Method

The main difficulty in generating natural-looking text governed by controllable latent variables is that it’s very hard to write down the latent variables of natural language as a set of HMMs.<sup>5</sup> If we could do that, we would not need LLMs. But that’s exactly the method’s intuition! LLMs already model language with high accuracy, and although they are not fully explainable or controllable, they are steerable to some degree. And for our goal, one controlled latent variable is enough. So instead of trying to write down language’s latent variables, we use steering to add one of our own: at each token, we steer the teacher along one of K = 8 uncorrelated and orthogonal SAE directions, while generating otherwise ordinary text. We only steer along one SAE direction at a time, given by a Markov chain which updates at every token. We then train a student model from scratch on the generated corpus.

One could consider the SAE directions to be 8 separate latent variables of natural language. However, by imposing this artificial dynamic over them, we create a new latent variable, and the 8 directions become values that the new latent variable can take. We also specifically pick the 8 directions to be orthogonal and uncorrelated, such that our structure could not have come from anywhere else.

Note that steering is known to inject durable, learnable structure into generated text, even if the text is seemingly unrelated to the steering direction (which is what we aim for). In their study, Morgulis and Hewitt (2026) call this mechanism “subliminal steering,” finding that fine-tuning a student on random number sequences written by a steered teacher makes the student inherit the teacher’s steering direction.

Figure 3 gives a visual summary of our proposed algorithm.

## 2.1 The Generative Process

The teacher model is a Gemma-2-2B (base) (Gemma Team, 2024), which we couple with a pretrained sparse autoencoder (Gemma Scope, 16K latents; Lieberum et al., 2024) for the residual stream of a middle layer (layer #12). From these we select K = 8 uncorrelated latents (the filtering is described below). Each selected latent i has a unit decoder direction $d _ { i }$ and an average firing magnitude $a _ { i }$ . Steering along latent i means adding $s \cdot a _ { i } \cdot \frac { \rho _ { \ell } } { \rho _ { 1 2 } } \cdot d _ { i }$ to the residual stream at layers 12 through 23 $( L - 2 ) ,$ where $\rho _ { \ell }$ is the average residual-stream norm at layer ℓ, so compensates for the stream’s growing norm. Lastly, s = 1.5 was chosen by a sweep, such that the injection is strong enough, yet does not degenerate the quality of the text.

![](images/e217e56a03610b09355393ef96962ddf4269e301bdf5b5f2fd0452065ad43924.jpg)  
Figure 3: Method summary: (a) we plant a latent variable as a Markov chain over K states; (b) we steer a teacher LLM while it writes, by adding the current state’s SAE decoder direction to the residual stream; (c) the result is natural-looking text, secretly labeled per token.

The corpus consists of independent documents of 256 tokens each. The first few tokens come from a pre-selected set of “openers,” and the teacher writes the rest. At every token, we steer the teacher along exactly one of the 8 directions. We write $z _ { t } \in \{ 1 , \ldots , 8 \}$ for the direction active while the token at position t is sampled. The 8 directions, together with the dynamics we impose on them, constitute the synthetic latent variable. One can think of it as a hidden “concept” governing the text, like the emotion variable drifting through a story (see Bigelow et al., 2026), except that here we control the variable and hold its ground truth, instead of finding it in the wild. The 8 directions themselves are not new, since each is a regular feature the teacher already has (intentionally unrelated and orthogonal to the others). The latent variable they form together is new though, as its possible states are only tied to their neighbors by our transition matrix. The latent variable takes a walk along the ring shown in Figure 4.

At 0.95 “stay” probability, the variable dwells ∼20 tokens per state before hopping to the neighbor. Each document’s path z<sub>1</sub>, . . . , z<sub>256</sub> is sampled before generation; thus, nothing else in the LM could plant the ring geometry.

Very importantly, the KV cache is always rebuilt at each token. Steering never touches the cached context, so that each token only depends on the current visible text and the latent variable. This will be very important in order to compute the outputs of an ideal Bayesian inference model, to which we will compare our student model’s belief state.

## 2.2 The Corpus

We generate 400,000 documents × 256 tokens (the first 12 tokens are taken from a pre-selected set of “openers”), approx. ∼102M tokens. We also generate a control corpus with steering off, and we train a control student on it. Analysis will run on

![](images/811df6e71da2623adf436839dd629aa0649393d82bf8fc8add2aa98816100961.jpg)  
Figure 4: The 8-state ring Markov chain with stay probability 0.95 and its transition matrix, plus a corpus excerpt with tokens colored by the active state.

both students.

## 2.3 Selecting the SAE Latents

Out of the 16K SAE latents, we drop the ones that are dead, fire too often, or mostly fire on punctuation and position. We don’t really care about what our chosen latents mean, and aim for them to be as hard to eyeball as possible (so that they are more “latent”). We also make sure that in our selection, the latents are near-orthogonal, don’t co-activate on natural text, and are causal (i.e., steering along them actually modifies the token distribution).

## 2.4 The Optimal Observer

At each token position in a document, we measure how well the state of the latent variable can be tracked by an observer who knows everything, i.e., knows the transition matrix of the 8 latents, and the softmax token distribution of the steered teacher model given any text.

Note that we can token label our transition matrix just like Shai et al. (2024)’s formalisms, so we truly have an HMM:

$$
T _ { i j } ^ { ( x ) } = T _ { i j } \cdot \mathrm { T e a c h e r } _ { j } ( x \mid x _ { < t } ) ,\tag{1}
$$

where Teache $z r _ { j }$ is just the LLM with $z _ { t } = j .$

We call our proposed reader the optimal observer, which performs Bayes inference over the operator defined above, maintaining the exact posterior $b _ { t } ( i ) ~ = ~ P ( z _ { t } = i ~ \mid ~ x _ { 1 : t } )$ at every token. Since $b _ { t }$ is a probability distribution over the 8 states, it is a point in the probability simplex $\Delta ^ { 7 }$

![](images/1d49a20408315b32a1e4ebaea15c980344a079ed0fccb29e396a05054a59c39f.jpg)  
Figure 5: PCA of the optimal observer’s posteriors: the point cloud fills a ring-shaped region with the 8 states in ring order at the rim.

The posteriors get updated as such at each token position t:

$$
\begin{array} { r } { b _ { t } ( j ) \propto \sum _ { i } b _ { t - 1 } ( i ) \cdot T _ { i j } ^ { ( x _ { t } ) } } \end{array}\tag{2}
$$

Because the KV cache is reset at every token, the next token only depends on the visible text and the current state of the model, thus making the optimal observer computable at every point. We use $b _ { t }$ at every token as the probe target for our student.

The optimal belief $b _ { t }$ sits on a 7-dimensional probability simplex. But if we PCA on it we find that the two principal components reflect the ring structure of $T$ (Figure 5).

## 2.5 The Student

We use a 110M-parameter Llama-style transformer (12 layers, width 768), trained from scratch on the corpus tokens and nothing else (∼5 epochs, 32k vocab). Unlike the optimal observer, the student has no access to the teacher model, so z reaches it intertwined with all the other latent variables of the Gemma model. We ask whether the factorized belief state over z emerges anyway, and we also explore the geometry of the latent variable.

We also train two control students. control1 was trained on a completely unsteered corpus from the same teacher. control2 was trained on a corpus steered by the same 8 latents with the same $p _ { s t a y } ,$ but on a state switch, the latent variable jumps uniformly to any of the other 7 states instead of to neighbors on the ring (Figure 6). It has the same exposure to the 8 features and incentive to track them, but no transition map to learn.

## 3 Results

Recall that this work has two objectives: 1. to confirm that transformers learn belief states over latent variables in a more realistic setting than the toy setup of Shai et al. (2026), and 2. to find the relationship between the geometry of a latent variable itself (which is different than the geometry of the beliefstate!) and its underlying statistical dynamics.

## 3.1 The Student Tracks the Belief States

Let $h _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { m o d e l } }$ be the student’s residual stream at layer ℓ and token position t. We want to test whether the belief state can be read out linearly out of this vector, so we fit a ridge regression from the hidden states to the posteriors of the optimal observer $b _ { t } \in \Delta ^ { 7 }$ , at each token: $\hat { b } _ { t } = W h _ { t } ^ { ( \ell ) } + c$

We fit the map on 4,000 held-out documents from our circularly steered corpus, and evaluate it on 1,000 more. Per layer, we report the regression’s $R ^ { 2 }$ against $b _ { t } .$ , and the accuracy of arg max<sub>i</sub> $\hat { b } _ { t } ( i )$ against the true state $z _ { t } ,$ , averaged over all token positions (Figure 7).

Indeed, the belief posterior does live in the residual stream. The best single layer reaches $R ^ { 2 } = 0 . 4 9$ , with argmax accuracy of the actual state of 0.577, which is approximately three quarters of the optimal observer’s ceiling of 0.762. This is significant given that the student had no direct access to the steered teacher model, as the optimal observer did.

As expected, control1 does not map the belief state (low $R ^ { 2 } )$ , but does guess the current state above chance. The SAE directions we steer along are natural in language, so ordinary text has coincidentally exposed it to them without even steering. But note that control2 reaches a significant $R ^ { 2 } = 0 . 4 1$ on the same documents, also with high argmax accuracy (0.52). This is also expected, as control2 has seen the eight states just as much, so after enough tokens in one state, it can decently estimate which state is active. In other words, after enough tokens in a state, $b _ { t }$ converges towards a one-hot vector, so there is nothing intrinsic about the ring geometry in it.

We believe the $R ^ { 2 } \left( \approx 0 . 0 8 \right)$ difference between the two is the advantage that the student model has from knowing how the previous state constrains the next one, i.e., internalizing the HMM’s structure. To confirm this, we re-evaluate on the first dwell only, where we define thefirst dwell of a document to be the prefix where the latent variable has not yet switched from its initial state. As shown in Figure 8, in this scenario, control2 and our student become indistinguishable. As there is no prior information, the transition map is useless even to the optimal observer, so this ablation is a positive result towards showing that the student calculates belief states.

For our student, we can also look at the probe’s read-out $\hat { b } _ { t }$ in the same PCA view we used for the optimal observer. The student’s belief state is a noisier version of the optimal observer’s (Figure 9).

## 3.2 The Eight States of the Latent Variable Sit on a Ring

So far we have shown that the student does hold beliefs resembling $b _ { t } .$ But control2 also does a decent job at inferring the belief state, because in our setup, each token carries some evidence about the current state, and, after enough tokens, that evidence alone suffices for identifying the state. Knowing the previous state only helps our student model early on, right after a state switch.

We now show where the models truly differ, which is how they represent the latent variable. Note that the geometry of the latent variable is a different object than the belief geometry above. We are no longer curious about the shape of the probability distribution, but about the shape of the states themselves. Specifically, we take the mean of the residual-stream activations for tokens in each of the 8 states (call them centroids) and we test what shape the 8 points form. If the geometry of the dynamics is inherited, they should sit on a ring.

A control2 corpus excerpt, tokens colored by the active state $z _ { t }$ (opener in gray):  
![](images/682603c8e26c789bd63e64c7372d048a1f9526f7620fbfe95d41be886a4ac0e6.jpg)  
It is done, and submitted. You can play “Survival Game", “P④ 1 Game", or the “Short Game”. I hope you like my soundtrack as well. 5 different songs are made, and as always the mood is a different type of genre.6 Survival Game:

Figure 6: The control2 chain: same stay probability, but uniform jumps to any of the other 7 states, plus a control2 corpus excerpt with tokens colored by the active state.  
![](images/53891f1c5687c5dbac5ac7f5cab0f06220696bcc7a0bfc0ac0a9561e0449ed67.jpg)

![](images/6b6d1ca5a32a6f4b9804c07abebb9c65a613bfb89b3802906a4dca8e91c60138.jpg)  
Figure 7: Per-layer probe $R ^ { 2 }$ and argmax accuracy for the ring student, control2, and control1, with the optimal observer ceiling at 0.762.

We project the centroids onto their top-2 principal components. In layers 9–11, the 8 centroids sit in the exact neighbor order of our Markov chain, and pairwise distance between centroids correlates with distances along the ring at 0.45. The control1 student shows nothing (with correlation 0.11); see Figure 10.

But note that this is a trap! Right after a switch from state 2 to state 3, the activations might still remember state 2 for a few tokens. So when we average all of state 3’s tokens to get state 3’s centroid, we are also getting some of state 2 (and state 4), because the previous state in the text is always a ring neighbor. This means that each of the 8 centroids gets dragged towards its neighbors, and that alone draws a ring. So the circle above could just be short-term memory of the context, without the model actually arranging the states on a ring. From now on, we only compute centroids on each document’sfirst dwell (i.e., before the latent variable ever switches states), so that no previous states leak in. In this setting, the PCA circle does not survive (Figure 11).

However, this does not mean the ring is gone, but rather that the ring geometry is not the principal geometry (we only looked at the first 2 PCA axes). We use Fourier analysis to directly query if there is any plane on which the centroids trace a circle, in the exact ring order of the HMM. Specifically, we look for two directions u and v such that:

![](images/9314af2c5251da700b44ed5c0aad117c52b98e020fe0d8121c6a57e08086349c.jpg)

![](images/ca86f9492808809ad78d1c5e92a96dc268283313f730dbaf872db82d1da2f396.jpg)  
Figure 8: Per-layer probe $R ^ { 2 }$ and argmax accuracy restricted to first-dwell tokens: the ring student and control2 curves coincide.

![](images/48be887a1b46a3ed7b53a5003b27039b8a5fb5b210d04665dc31537664f3fb46.jpg)  
Figure 9: Side-by-side PCA of the optimal observer’s posteriors and the student’s probe read-out at layer 12: same ring arrangement, noisier cloud for the student.

$$
\mu _ { i } \approx u \cos ( 2 \pi i / 8 ) + v \sin ( 2 \pi i / 8 ) ,\tag{3}
$$

where $\mu _ { i }$ , with $i \in \{ 1 , 2 , \ldots , 8 \}$ , is the centroid of the i-th state. The fitted probes (u, v) explain 38% of the centroid variance at layer 11. For a significance test, we refit the same pattern under every possible way of arranging the 8 states on a ring (2520 orderings), and the true ring ranks 1st out of all 2520 in terms of explained variance, for our student. However, for the control1 and control2 models, the true order lands mid-distribution. So, the circle geometry is real, respecting the exact ringorder, and can be attributed to the HMM structure. Moreover, we have also finally differentiated the student from control2 (Figure 12).

We think the ring geometry is not dominant simply because our latent variable is not that relevant. Most of the residual stream is spent on ordinary language features, and our variable only injects ∼0.05–0.1 nats of evidence per token. Moreover, the Gemma teacher surely models other latent variables too, and our 8 states could also be attributes that other latent variables land on. In other words, each centroid possibly sits in multiple geometries at once, each corresponding to other latent variables.

![](images/ecaa989815b5431a0caaf9023693188709c8790f9f57f64134cce606030aec61.jpg)

![](images/12b69dacfdacd3a99c17cfefa053539bfbd0eed66e21c4a86bc2a0e7d505dd8f.jpg)  
Figure 10: Top-2 PCA of the 8 state centroids: the ring student’s centroids trace the ring in exact neighbor order; the control student’s do not.

The ring structure is also visible in raw similarities. In Figure 13, pairwise cosine similarity between centroids is warmest on the neighbor diagonals, resembling the matrix T. Moreover, the columns of a whitened ridge probe are most anticorrelated on the same diagonals. This supports our case, as the probe has to spend capacity on differentiating exactly the pairs that are most similar (i.e., the ones on the ring).

## 4 Conclusion

We introduced a method for planting a latent variable inside natural-looking text, where an LLM teacher writes ordinary text while we “subliminally” steer it along SAE directions whose activity follows a Markov chain of our choosing.

![](images/74ae7a056b6fecc7ffbb4a397903aee2ecb1e640bda362663589105478fd9454.jpg)

![](images/a1202678fa90a21bec281aa28246ce02cb80280cf1ccf3f53aace5de596016c7.jpg)  
Figure 11: Top-2 PCA of first-dwell centroids: the clean ring is no longer visible in the leading principal components.

By doing so, we confirmed that transformers learn belief state geometries in a more realistic setting than current work. Moreover, we have tied belief states to concept geometries: the geometry of a latent variable (not only the geometry of the belief state about that latent variable) is influenced by the dynamics of the data generating process.

## 5 Future Work

A first natural follow-up is to plant multiple HMMs at once with our algorithm (Shai et al., 2026 already do this in their toy setup), including more complicated interactions between them.

Second, many existing theories attribute feature manifolds to semantic similarity. Our setup can put the two in direct conflict: take semantically similar states (like different colors, for example) and tie them together in a Markov chain that’s uncorrelated to their semantic similarity.

Third, recall that our ring was not the dominant geometry. Perhaps, each state participates in multiple other latent variables. We are curious whether a possible manifold entanglement is happening in such scenarios. We could test this directly with the multiple HMMs setup.

Fourth, we are also interested in hierarchical concepts and how they fit in this picture.

Lastly, to close the loop, we want to use this theory to make predictions about real LLMs. We want to find a latent variable (maybe something like syntax), estimate its transition dynamics from the training corpus, and predict the geometry of that concept in a real LLM, like OLMo.

## Limitations

Our setup, although more realistic than the toy setups, is still not full proof of Shai et al.’s hypothesis about belief state geometries. We impose a latent variable on language, which shows that transformers pick up such structure when it is there, and, although our setup is more natural, it suffers from the same limitation as the initial proponents of the theory.

Our evidence for the concept geometry is just correlational as we have not done any causality experiments.

We also have to expand our configurations for a more comprehensive study, i.e., we have to use more HMM structures, other choices of SAE latents, other teacher models, etc.

Lastly, we acknowledge that in order to construct a new latent variable, we had to use SAE directions, which could be latent variables themselves, or attributes that other latent variables take. This is not ideal as we were not able to fully isolate our planted latent variable (the ring geometry was not dominant), so there remains the open question of how to better inject latent variables in natural-looking text.

## References

Eric Bigelow, Raphaël Sarfati, Daniel Wurgaft, Owen Lewis, Thomas McGrath, Jack Merullo, Atticus Geiger, and Ekdeep Singh Lubana. 2026. Stories in space: In-context learning trajectories in conceptual belief space. Preprint, arXiv:2605.12412.

Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. 2024. Not all language model features are one-dimensionally linear. Preprint, arXiv:2405.14860.

Gemma Team. 2024. Gemma 2: Improving open language models at a practical size. Preprint, arXiv:2408.00118.

Dhruva Karkada, Daniel J. Korchinski, Andres Nava, Matthieu Wyart, and Yasaman Bahri. 2026. Symmetry in language statistics shapes the geometry of model representations. Preprint, arXiv:2602.15029.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. 2024. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. Preprint, arXiv:2408.05147.

Alexander Modell, Patrick Rubin-Delanchy, and Nick Whiteley. 2025. The origins of representation manifolds in large language models. Preprint, arXiv:2505.18235.

George Morgulis and John Hewitt. 2026. Subliminal steering: Stronger encoding of hidden signals. Preprint, arXiv:2604.25783.

![](images/76c3b6a048a9c4327923f6ff5e380ede7d338eda6b28b0d4ca58f9a0081a9fea.jpg)

![](images/05ac2d49ad9735a4b910f11ecfb4bd5444d66ace807931f873744027b8c548ca.jpg)

![](images/824bbb129f18058c1ce508af6e52c07c4df22c001cb320b948b4b3fa00baef43.jpg)

![](images/4b871dfecf6b60e4acee34c1d865aeb0547bb804a6e3c6e649a2cede20288ccb.jpg)

![](images/9c03d272d2bbe2bec7b0cd26c5419dda299e0137287f3e98958bf7ea4d3ef3a5.jpg)

![](images/c401f4014e314e90ff2b8fa8482eaed80dedec5ade8ff6092eb574bccf3eb0eb.jpg)  
Figure 12: Circle-plane projections of first-dwell centroids for the three students, and the circle-fit permutation test over all 2520 ring orderings: the true ring ranks 1st of 2520 for the ring student only.

![](images/280b5f4a1ce5bddaf14f2028424271051ebed2ce00c357f70d5f5e2bd8107166.jpg)

![](images/3719d44288cc2e1f2dd40d0e5708b4ecdb72691ad5a6f5fac39120ed4d40276e.jpg)

![](images/c51a5be105e773f97aee0cfeb6f99d7bd2c77ccb67b3b3891aa5d4a3a5bd0a96.jpg)

![](images/5bb190eafd86c2ef026ac9c13f412cbf27ce30f3d37af523f7710062a1a7d9e0.jpg)  
Figure 13: Heatmaps: transition matrix T, centroid cosine similarities warmest on the neighbor diagonals, whitened probe directions most anti-correlated there, and similarity vs. ring distance.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2023. The linear representation hypothesis and the geometry of large language models. Preprint, arXiv:2311.03658.

Raphaël Sarfati, Eric Bigelow, Daniel Wurgaft, Siddharth Boppana, Jack Merullo, Atticus Geiger, Owen Lewis, Tom McGrath, and Ekdeep Singh Lubana. 2026. The shape of beliefs: Geometry, dynamics, and interventions along representation mani-

folds of language models’ posteriors. Preprint, arXiv:2602.02315.

Adam Shai, Loren Amdahl-Culleton, Casper L. Christensen, Henry R. Bigelow, Fernando E. Rosas, Alexander B. Boyd, Eric A. Alt, Kyle J. Ray, and Paul M. Riechers. 2026. Transformers learn factored representations. Preprint, arXiv:2602.02385.

Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, and Paul M. Riechers. 2024. Transformers represent belief state geometry in their residual stream. Preprint, arXiv:2405.15943.