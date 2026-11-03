# ONLINE LABEL-FREE GRADIENT MATCHING FORCONTINUAL LANGUAGE MODEL TRAINING

Sajjad Pakdamansavoji<sup>1,∗</sup>, Mohammad Reza Rezaei<sup>1</sup>, Ilija Subasic<sup>1</sup>, Pedro Garzon<sup>1</sup>, Radu Paradovschi, Zhao Chen<sup>1</sup>

<sup>1</sup>Upwork Inc (upwork.com), <sup>∗</sup>Corresponding author

## ABSTRACT

Language models in customer-facing products must be continually updated, since the data they serve shifts over time. As retraining on the full accumulated data is impractical, each update should instead run on a small set of examples carrying the training signal of all data seen so far. The strongest current methods build this set by gradient matching, which synthesizes examples whose training gradients match those of the full data. However, they require ground-truth labels, run once on a fixed snapshot rather than on a growing stream, and are slow on large data. We introduce OLGA, which lifts these limitations. It distills unlabeled data by drawing the gradient matching targets from the model’s own next-token predictions. Across five text classification benchmarks with phi-1.5, it improves accuracy by 10 percentage points over the strongest selection baseline using as few as five examples. OLGA also distills online data streams by keeping a running rehearsal memory of synthetic examples, which is re-distilled together with each new batch as it arrives. In the online setting, it outperforms the strongest selection-based continual baseline by 6.6 stream-AUC percentage points. Finally, our method scales to large data by pairing distillation with selection in a hybrid pipeline, letting selection handle the mild stage of compression and distillation the extreme one. It cuts runtime by 30% versus distillation alone at similar accuracy, and beats the best selection-only pipeline by up to 12.6 percentage points.

## 1 INTRODUCTION

Language models are increasingly deployed in customer-facing products, where the traffic they serve keeps changing. User preferences move, new topics appear, and safety requirements shift, so a model that was accurate at release gradually falls out of step with the data it now sees (Lazaridou et al., 2021). Keeping it current means updating it repeatedly rather than once, and the data available at each update grows with every cycle. Retraining on every example accumulated so far is the obvious remedy and also the least affordable one, because a full pass costs more as the data grows while the oldest examples contribute least. What a practical update loop needs instead is a small set of examples, rebuilt at each cycle, that still carries the training signal of everything seen so far.

Most existing methods build that set by selecting a well-chosen subset of the real examples. This works well when the budget is generous, but once only a few examples can be kept, it becomes hard to preserve a strong training signal. Dataset distillation with gradient matching addresses this by synthesizing examples whose training gradient matches the gradient of the real data (Zhao et al., 2021). Because a synthetic example does not have to be a real sentence, it can capture what many real examples teach the model, so the set can be kept to a few examples.

Three obstacles stand in the way of using gradient matching to keep a language model current. First, it needs ground-truth labels, which rules out the unlabeled data that makes up most of what a deployed system collects. For images, recent work replaces the labels with targets from a frozen model trained without them (Lee et al., 2024; Joshi et al., 2025), but text distillation methods still rely on labels (Ogawa et al., 2025). Second, it runs once over a fixed snapshot, so it never has to fold an arriving batch into an existing set. Methods that do this exist for images (Gu et al., 2024), but they need labels and nudge pixel values in small steps, which does not work for the discrete tokens of text. Finally, the cost of matching gradients grows with the amount of data it compresses, making it least affordable exactly when data is plentiful. For images, discarding part of the real data before distilling cuts this cost (Xu et al., 2024; Moser et al., 2025), but these methods rank examples by their loss against a label and have only been studied on a fixed snapshot. As a result, no existing method compresses an unlabeled text stream into a small training set at a cost that stays affordable as the stream grows.

We remove all three obstacles at once. To drop the labels, we keep them out of the objective and take the matching target from the model’s own predictions instead. To move from a snapshot to a stream, we keep a running rehearsal memory of synthetic examples and merge each arriving batch into it with the same gradient matching objective. To control the cost, we introduce a hybrid pipeline in which cheap unsupervised selection runs ahead of expensive distillation, since the two are strong at different amounts of compression.

We make the following contributions.

• We introduce a label-free gradient matching objective whose compact distilled set trains a model 10 points more accurate than the best real subset of the same size.

• We extend text distillation to streaming data, updating a rehearsal memory as each batch arrives, and beat the best real memory of the same size by 6.6 stream-AUC points.

• We design a hybrid pipeline in which cheap selection shrinks each batch before distillation, cutting distillation runtime by 30% compared with distilling the full batch.

## 2 BACKGROUND AND RELATED WORK

For much of the past decade, progress in deep learning came from scale, with larger models trained on more data consistently outperforming smaller ones. As models have grown to billions of parameters and corpora to trillions of tokens, however, the bottleneck has shifted from data availability to compute and time. Deployed models must also be updated ever more often as user preferences, knowledge, and safety requirements change, and retraining on the full corpus at every update is pro hibitively expensive. The question is therefore no longer how to collect more data, but how to select or construct the smallest set of examples that preserves the training signal of the full corpus.

Coreset selection. Coreset selection keeps the most representative slice of the real training set without modifying any examples. Geometry-based algorithms formalize this as a coverage problem, where k-center greedy (Sener & Savarese, 2018) picks the point farthest from the selected set, while herding (Rebuffi et al., 2017) and prototypical selection (Snell et al., 2017) sequentially choose samples that best approximate the per-class feature centroid. Training-oriented variants use model-dependent signals such as gradient norms (GraNd) or L2 prediction error (EL2N) (Paul et al., 2021) to prioritize samples near the decision boundary rather than near the class mean. Despite this diversity, all coreset methods share a hard representational ceiling, since a text sequence is an atomic unit that cannot be perturbed below the token level and no selection rule can manufacture variance that does not already exist in the chosen sample. This ceiling becomes binding at extreme compression, the regime where synthetic examples become necessary.

Dataset distillation. Dataset distillation compresses a large training set into a handful of synthetic examples, optimized so that a model trained on them behaves like one trained on the original data. Wang et al. (2018) first framed this as bi-level meta-learning, updating the synthetic examples through an unrolled inner training loop, but the memory cost and numerical instability of unrolling restrict it to shallow networks trained for only a few steps. Later work replaces the unroll with cheaper surrogate objectives. Trajectory matching (Cazenavette et al., 2022) aligns the parameter path traced by training on synthetic data with paths recorded on real data, at the cost of precomputing and caching thousands of such expert trajectories. Distribution matching (Zhao & Bilen, 2023) instead matches per-class feature embeddings of real and synthetic examples, which is cheap but ignores the task loss and can leave class boundaries misaligned. Gradient matching (Zhao et al., 2021) targets the task loss directly by finding synthetic examples whose per-step parameter gradient matches the real gradient in cosine distance. It needs neither long unrolls nor stored trajectories and scales to larger architectures more readily, making it the dominant paradigm. All three directions, however, assume that gradients flow through a continuous input space, which holds for images but breaks down for discrete text tokens, where a gradient step on the input has no direct meaning.

Dataset distillation for language models. Early work on distilling text avoided discreteness by optimizing continuous word embeddings instead of tokens (Maekawa et al., 2023), but the results are model-specific, meaning a synthetic embedding created with BERT cannot be used to fine-tune RoBERTa, and projecting the optimized embeddings back to the nearest vocabulary token yields incoherent sequences that carry little training signal. Maekawa et al. (2024) addressed this with DiLM, which trains a proxy generative model to synthesize readable text, routing gradients through the discrete bottleneck via generation probabilities and producing sequences that transfer across architectures. Training a full proxy language model as a distillation scaffold is itself expensive, however, and the proxy may memorize sensitive training-set content. GradMM (Nguyen et al., 2025) avoids the proxy by applying the Alternating Direction Method of Multipliers (ADMM) (Boyd et al., 2011), in which gradient matching proceeds in continuous embedding space while a periodic projection step snaps each embedding back to its nearest vocabulary token, enforcing hard discreteness throughout optimization. This produces human-readable, cross-architecture synthetic text, and it is the offline supervised technique we build on here. Like every text distillation method so far, however, it places ground-truth labels in the matching objective, which ties the distilled data to a single task and excludes the unlabeled data that makes up most of what language models are trained on.

Continual learning and catastrophic forgetting. In continual learning, a model must keep absorbing a non-stationary stream of data without losing what it has already learned. The core difficulty is catastrophic interference, in which training on new material abruptly overwrites earlier knowledge (McCloskey & Cohen, 1989; French, 1999), and it arises whether the stream introduces new tasks, new domains, or new classes (van de Ven & Tolias, 2019). Architecture-based methods counter it by dedicating parameters to each task (Rusu et al., 2016; Mallya & Lazebnik, 2018), but presuppose known task boundaries and a growing parameter budget. Regularization-based methods penalize changes to weights that matter for past tasks (Kirkpatrick et al., 2017; Zenke et al., 2017; Li & Hoiem, 2018), but weaken over long streams as their accumulated constraints stiffen the network. The dominant remedy is rehearsal, which replays a small memory of past examples alongside incoming data, with iCaRL (Rebuffi et al., 2017) as its canonical instance and a strong baseline for nearly a decade. Because the memory is the method, much work concerns how to curate it, from fixed rules such as reservoir sampling, herding, or class-balanced storage (Prabhu et al., 2020) to gradient-based rules that keep the examples whose gradients best span (Aljundi et al., 2019) or best approximate (Tiwari et al., 2022) the gradient of all data seen so far. Every such scheme still chooses among real examples, so the memory inherits the representational ceiling of coreset selection. That ceiling binds hardest where rehearsal matters most: as the stream grows and the per-class budget shrinks, a few stored examples can no longer carry the training signal of the task they stand for.

Continual distillation as rehearsal memory. This ceiling motivates compressing the memory rather than selecting it, using dataset distillation to synthesize a handful of maximally informative rehearsal examples. Distilled Replay keeps as little as one synthetic pattern per class (Rosasco et al., 2022), Zhao et al. (2021) showed that gradient-matched synthetic memories alleviate forgetting in a task-incremental image setting, and later methods refine the buffer by carrying cross-task information (Liao et al., 2025), mixing distilled with real exemplars (Kong et al., 2024), or condensing prototypical exemplars (Nguyen et al., 2026). Two limitations run through this line of work. First, these methods optimize continuous pixel tensors with objectives that do not carry over to discrete tokens. Continual learning for text has therefore sidestepped distillation in favor of generative rehearsal, sampling pseudo-examples from an autoencoder (Zeng et al., 2024) or prompting the model to synthesize its own rehearsal set (Huang et al., 2024), which re-incurs the expense and memorization risk of a separate generator. Second, nearly all of them are offline, distilling each task once from a complete snapshot. The one exception, SSD (Gu et al., 2024), folds each arriving batch into a synthetic image memory by matching gradients against the labeled batch. How to fold an arriving batch into a synthetic memory of discrete tokens, with or without labels, has not been studied.

## 3 METHODOLOGY

To address these obstacles, we introduce Online Label-free Gradient Matching (OLGA), in which each component removes one obstacle (Figure 1). We begin in Section 3.1 with the offline supervised method that OLGA builds on. Section 3.2 then drops the need for labels, Section 3.3 extends the method to streaming data, and Section 3.4 speeds it up with a hybrid design.

![](images/c16813781cc09a44d94a46690fda188630543e604decf5df1ba6512f02deff7c.jpg)  
Figure 1: Overview of the components of OLGA. (a) Synthetic text is generated by matching its training gradient to that of real data, alternating with a step that maps it back to real tokens. (b) A frozen language model’s next-token predictions replace ground-truth labels, so gradient matching works on unlabeled data. (c) A running rehearsal memory extends offline distillation to an online data stream. Each new batch is distilled together with the memory, which the model then retrains on. (d) Our hybrid design first uses cheap selection to shrink each batch, then distills what remains into the few examples kept, which cuts runtime on large data.

## 3.1 GRADIENT MATCHING WITH THE SUPERVISED OBJECTIVE

The method we start from is GradMM (Nguyen et al., 2025), which produces readable and architecture-agnostic examples by solving a discrete gradient matching optimization. Let $\pmb { \theta } _ { 0 }$ denote the parameters of a frozen pretrained language model and D a set of examples $s = ( p , r )$ , each pairing a prompt with a response. The loss of the model on D, written $\ell ( \mathcal { D } , \pmb \theta )$ , is the average of the individual example losses $\ell ( s , \pmb \theta )$ . The real examples that serve as a matching target are called anchors and written $\mathcal { D } _ { \mathrm { r e a l } } .$ , and the synthetic set produced from them is $\mathcal { D } _ { \mathrm { s y n } } ,$ held to a budget of m examples. GradMM searches for a synthetic set whose fine-tuning gradient matches the anchor gradient at the pretrained parameters,

$$
\mathcal { D } _ { \mathrm { s y n } } ^ { \star } = \operatorname * { a r g m i n } _ { | \mathcal { D } _ { \mathrm { s y n } } | \leq m , \mathcal { D } _ { \mathrm { s y n } } \subset \Gamma } D ( \nabla _ { \theta } \ell ( \mathcal { D } _ { \mathrm { s y n } } , \pmb { \theta } _ { 0 } ) , \nabla _ { \theta } \ell ( \mathcal { D } _ { \mathrm { r e a l } } , \pmb { \theta } _ { 0 } ) ) ,\tag{1}
$$

where $D ( \boldsymbol { \mathbf { u } } , \boldsymbol { \mathbf { v } } ) = 1 - \cos ( \boldsymbol { \mathbf { u } } , \boldsymbol { \mathbf { v } } )$ compares gradient vectors by direction rather than scale, and $\Gamma$ is the set of readable token sequences. GradMM solves this with ADMM (Boyd et al., 2011), which alternates a few steps of first-order optimization in continuous embedding space with a projection back onto real vocabulary tokens (Figure 1(a)). We keep this solver unchanged and focus instead on making the optimization work without labels, on a data stream, and at lower runtime.

In the supervised setup, the model is asked to emit a verbalizer token $v _ { y }$ , a single word standing in for the class of the example, at a fixed position after the prompt, and the loss is the cross-entropy of that single token, $\ell ^ { \mathrm { s u p } } ( s , \pmb \theta ) = - \log \bar { p _ { \pmb \theta } } ( v _ { y } \mid \pmb p )$ . Since every example of a class is trained toward the same token, their gradients align and add up rather than cancel. The anchor gradient thus lies in a few class directions, which a handful of synthetic examples can reproduce closely.

## 3.2 GRADIENT MATCHING WITH A LABEL-FREE OBJECTIVE

While the supervised loss $\ell ^ { \mathrm { s u p } } ( s , \pmb \theta )$ needs each example’s class to know which verbalizer token the model should emit, customer-facing platforms mostly receive unannotated examples for which a single class label is hard to pin down (Zhao et al., 2024). Such data is rarely unstructured, though, and can be partitioned into groups $\mathcal { D } _ { \mathrm { r e a l } } ^ { g }$ by pre-logged metadata or through unsupervised clustering (Wan et al., 2024). We use this grouping, together with the model’s internal next-token probabilities, to design a label-free loss $\ell ^ { \mathrm { g r p } } ( s , \pmb \theta )$ that preserves inter-group separation (Figure 1(b)).

Let $p _ { \pmb { \theta } _ { 0 } } ( v \ | \ \pmb { p } )$ denote the probability the frozen model puts on a next token v after a prompt $^ { p , }$ and let $\bar { p } _ { g } ( v )$ and $\bar { p } ( v )$ be its averages over one group and over all anchors. The pointwise mutual information between a token and a group,

$$
\begin{array} { r } { \mathrm { P M I } ( v , g ) = \log \bar { p } _ { g } ( v ) - \log \bar { p } ( v ) = \log \frac { \frac { 1 } { | \mathcal { D } _ { \mathrm { r e a l } } ^ { g } | } \sum _ { s \in \mathcal { D } _ { \mathrm { r e a l } } ^ { g } } p _ { \theta _ { 0 } } ( v \mid p ) } { \frac { 1 } { | \mathcal { D } _ { \mathrm { r e a l } } | } \sum _ { s \in \mathcal { D } _ { \mathrm { r e a l } } } p _ { \theta _ { 0 } } ( v \mid p ) } , } \end{array}\tag{2}
$$

measures how much more the model expects v inside one group than across the data as a whole. Ranking the vocabulary by equation 2 and keeping the k highest-scoring tokens gives a set $\ v { \nu } _ { g } ^ { k }$ of the tokens that best separate group $g$ from the rest, and k is the only hyperparameter the construction adds. Normalizing the group probabilities across $\ v { \nu } _ { g } ^ { k }$ turns them into a target $q _ { g }$ , and for $s \in \mathcal { D } _ { \mathrm { r e a l } } ^ { g }$ the group-distribution loss is the soft cross-entropy toward it,

$$
\ell ^ { \mathrm { g r p } } ( s , \pmb \theta ) = - \sum _ { v \in \mathcal { V } _ { g } ^ { k } } q _ { g } ( v ) \log p _ { \pmb \theta } ( v \mid \pmb p ) , \qquad q _ { g } ( v ) = \frac { \bar { p } _ { g } ( v ) } { \sum _ { u \in \mathcal { V } _ { g } ^ { k } } \bar { p } _ { g } ( u ) } .\tag{3}
$$

Substituting $\ell ^ { \mathrm { g r p } }$ into equation 1 replaces the class label with a soft group distribution read off the model’s own knowledge, while preserving the separation between groups. Section 4.5 compares it against several alternatives.

## 3.3 ONLINE GRADIENT MATCHING WITH REHEARSAL MEMORY

The objective of equation 1 suits an offline setup, where a single fixed dataset is available all at once. A deployed system rarely works this way, since its data arrives as a sequence of blocks $\mathcal { D } _ { \mathrm { r e a l } } ^ { 1 } , \ldots , \mathcal { D } _ { \mathrm { r e a l } } ^ { \hat { T } }$ , with $\stackrel { \cdot } { \mathcal { D } } _ { \mathrm { r e a l } } ^ { 1 : t }$ the data seen through step t (Li et al., 2025). In this online setting, we want a compact synthetic set of at most m examples that, at each step, best represents everything seen so far, so that fine-tuning $\pmb { \theta } _ { 0 }$ on it tracks the model trained on all data through step t. We write this set as $^ { \star } \mathcal { D } _ { \mathrm { s y n } } ^ { 1 : t }$ , the one whose gradient at $\pmb { \theta } _ { 0 }$ matches that of the full history:

$$
^ { \star } \mathcal { D } _ { \mathrm { s y n } } ^ { 1 : t } = \underset { | \mathcal { D } _ { \mathrm { s y n } } ^ { 1 : t } | \leq m , \mathcal { D } _ { \mathrm { s y n } } ^ { 1 : t } \subset \Gamma } { \arg \operatorname* { m i n } } D \big ( \nabla _ { \theta } \ell ( \mathcal { D } _ { \mathrm { s y n } } ^ { 1 : t } , \theta _ { 0 } ) , \nabla _ { \theta } \ell ( \mathcal { D } _ { \mathrm { r e a l } } ^ { 1 : t } , \theta _ { 0 } ) \big ) .\tag{4}
$$

Computing $^ { \star } \mathcal { D } _ { \mathrm { s y n } } ^ { 1 : t }$ directly is impractical in a stream, since it requires keeping the entire history $\mathcal { D } _ { \mathrm { r e a l } } ^ { 1 : t }$ and re-distilling it from scratch at every step, so both storage and per-step cost grow without bound as the stream lengthens. We approximate it by solving a separate distillation problem for each incoming block $\mathcal { D } _ { \mathrm { r e a l } } ^ { \tilde { t } }$ and merging the results into a running rehearsal memory through the same gradient-matching objective. Each block is first compressed on its own into a synthetic set $\mathcal { D } _ { \mathrm { { s y n } } } ^ { t }$ by the offline problem of equation 1, and the memory $\mathcal { \hat { M } } ^ { t }$ carries these forward. It starts from the first distilled block and, at every later step, is rebuilt by distilling the union of the new set and the previous memory back down to m,

$$
\mathcal { M } ^ { t } = \operatorname * { a r g m i n } _ { | \cdot | \leq m , \cdot \odot \mathrm { { T } } } D \big ( \nabla _ { \theta } \ell ( \cdot , \theta _ { 0 } ) , \ : \nabla _ { \theta } \ell ( \mathcal { D } _ { \mathrm { s y n } } ^ { t } \cup \mathcal { M } ^ { t - 1 } , \theta _ { 0 } ) \big ) , \qquad \mathcal { M } ^ { 1 } = \mathcal { D } _ { \mathrm { s y n } } ^ { 1 } .\tag{5}
$$

We call this a cascaded merge (Figure 1(c)). At every step we fine-tune on $\mathcal { M } ^ { t }$ , which stands in for the $^ { \star } \mathcal { D } _ { \mathrm { s y n } } ^ { 1 : t }$ we cannot compute. Each step distills only the new block and the current memory instead of the whole history, so its cost is the same at every step and the total grows linearly with the stream length $T .$ Since the merge is independent of the loss, it applies to both the supervised and the label-free objectives. We compare the cascaded merge against alternative merge strategies in Section 4.5.

## 3.4 HYBRID HIERARCHICAL COMPRESSION

Selection and distillation work best in different regimes. When the budget is a large fraction of the data, cheap coreset selection is enough, since a well-chosen real subset still covers most of the training signal. As the budget shrinks toward a handful of examples, no real subset can carry that signal, whereas a single synthetic example can combine signal drawn from many real ones. Viewing compression as an information bottleneck (Tishby et al., 2000; Zhou et al., 2025), we expect selection to suffice while it is wide and distillation to be needed once it narrows.

We therefore compress in two stages (Figure 1(d)). At every compression step, offline on a static pool or at each step of the stream, a coreset selector first cuts the incoming batch of B examples down to a smaller pool of $K ,$ and gradient matching then distills that pool into the m examples actually kept, so $B \ \gg \ K \gg m$ . We call this the hybrid pipeline and compare it against selection-only and distillation-only alternatives in Section 4.4.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Data. We use five text classification datasets: SST-2 (Socher et al., 2013), Rotten Tomatoes (Pang & Lee, 2005), RT-Polarity (Pang & Lee, 2005), IMDB (Maas et al., 2011), and TwitterEmotion (Saravia et al., 2018). They span two domains, movie reviews and tweets, inputs from a single sentence to a full review, and training pools from 1,000 examples to tens of thousands. Every experiment uses all five, except the streaming hybrid, whose ten steps of 1,000 examples only SST-2 and IMDB can supply. Appendix A1 details the data partitioning and stream construction.

Model. Following GradMM (Nguyen et al., 2025), we use phi-1.5 (Li et al., 2023) for our experiments. During distillation, we only match the gradients of the output layer to avoid recomputing the gradients of all 1.3B parameters at each step, and we run the ADMM solver for 30 outer iterations of 50 inner steps at a step size of 0.008. During fine-tuning, we train 200 steps on a linear schedule, evaluating every 50 steps and keeping the best checkpoint, with the learning rate picked per dataset from $\{ \bar { 7 } { \times } 1 0 ^ { - 6 } , 1 { \times } 1 0 ^ { - 5 } , 1 . 5 { \times } \bar { 1 } 0 ^ { - 5 } \}$ . While we use phi-1.5 for both distillation and fine-tuning, two different models can be used in general, without any change to the method, because what distillation returns is plain text rather than weights.

Compared methods and metrics. In the offline setup, our main metric is accuracy after finetuning, and we compare against five competing methods: k-center greedy (Sener & Savarese, 2018) takes the example farthest from those already picked; herding (Rebuffi et al., 2017) takes the ones that keep the running mean closest to the average; prototypical (Snell et al., 2017) takes the ones closest to that average; uncertainty (Lewis & Gale, 1994) takes the ones the model is least sure about; and random picks uniformly. In the online setup, we focus on the area under the accuracyversus-step curve (AUC), which rewards being right at every step rather than only the last, and we compare against four competing methods: k-center greedy, herding, and prototypical from above, and reservoir sampling (Vitter, 1985), which keeps a uniform sample of all data seen. In both settings we also report three reference points. An oracle fine-tunes on all available real data, and supervised GradMM distills with the ground-truth labels that our label-free method never sees. Together they give upper bounds, while the zero-shot model with no fine-tuning gives the lower bound.

## 4.2 OFFLINE LABEL-FREE GRADIENT MATCHING BEATS COMPETING METHODS

We benchmark OLGA, setting k=100 for its label-free group distribution objective, against the supervised objective (GradMM) and the five competing methods in the offline setup, where the full data pool is available at once. Each task starts from 50 real anchors, which we distill into a compact synthetic set of 5 examples and fine-tune phi-1.5 on, evaluating on the held-out validation split of each dataset. This is a 90% reduction, which creates the extreme-compression regime we target.

Figure 2(a) reports classification accuracy on each dataset. OLGA is the strongest label-free method, beating the strongest competing method on four of the five datasets and by 10.2 points on average. The gap is widest on IMDB, where OLGA scores 89.5 against 72.1 for the strongest competitor, and narrowest on Rotten Tomatoes, at 4.1 points (74.5 against 70.4). The one exception is TwitterEmotion, which is hard for every method, and where OLGA reaches just 56.5, on par with the competing methods. Overall, even without labels, gradient matching keeps enough signal to compress a dataset into a handful of synthetic examples that train a model better than any subset of real examples can. Table 1 shows two distilled examples next to the real examples they borrow their phrasing from, and Appendix A5 one for each of the other datasets. Appendix A2 lists exact values for Figure 2(a).

## 4.3 ONLINE GRADIENT MATCHING WITH MEMORY BEATS CONTINUAL SELECTION

We benchmark OLGA, setting m=5 for the rehearsal memory and cascaded merge budgets, across all five datasets in an online setup of $T { = } 1 0$ steps (8 for IMDB). At each step, 50 anchors arrive and are distilled into 5 synthetic examples, which are then folded into the memory by the cascaded merge. At each step, phi-1.5 is fine-tuned on the resulting memory and evaluated on the held-out validation set.

![](images/7b279b0f36a88ca21f573260a6ba23a711f50133d43dabcea98c6908f201e68c.jpg)  
Figure 2: Across five datasets, OLGA beats every competing method on average in both the offline and online setups, and comes close to the supervised upper bound in the online one. (a) Offline methods reduce 50 real examples to 5, and we report the held-out accuracy of phi-1.5 fine-tuned on them. (b) Online data arrives over a sequence of steps, each method keeps a memory of only 5 examples, and we report stream-AUC, the area under the accuracy-versus-step curve.

Table 1: Two synthetic examples distilled by OLGA, one per dataset. Each is generated as a mix of phrases from real examples, three of which are listed below it. A shared phrase has the same color in both places, and its superscript gives the number of the real example it comes from.
<table><tr><td></td><td>TwitterEmotion (joy)</td><td>SST-2 (negative)</td></tr><tr><td>Synthetic</td><td>I am feeling better after my recent health surgery so much2 yesterday andI can feel³ more</td><td>teacher was giving a long and boring and² not interesting and clear lecture to the students because the teacher was read- ing from a textbook</td></tr><tr><td>Real</td><td>1. i am feelingamazing mostly normal i am going to a pre thanksgiving celebration with our friends from that time we were in softball 2. i feel like we haveso muchto be thankful for 3.i can feelsuperior on that point</td><td>1. it ai n’t art, bya longshot, 2. a shame that stealing harvard is too busy getting in its own way to be anything but frustrating,boring, andforgettable. 3. a more confused, lessinteresting andmore sloppily made film could possibly come down the road in 2002</td></tr></table>

Figure 2(b) reports the online stream-AUC of the fine-tuned phi-1.5 on each dataset. Averaged over the five datasets, Online GradMM and OLGA finish 8.5 and 6.6 points above k-center, the strongest competing method overall at 67.0. The two are within two points of each other, so removing the labels costs little. Across datasets the biggest gain for both comes on RT-Polarity, roughly 10 points over the best competing method there. The shortfalls are all on the label-free side, where OLGA only draws level with the best competing method on TwitterEmotion and comes in 4.2 points behind it on SST-2. Appendix A3 lists exact values for Figure 2(b).

## 4.4 HYBRID COMPRESSION PIPELINE

We explore whether a mix of selection and distillation exists that lets cheap selection take the early, easy part of compression and leaves only the extreme end to distillation, and how such a mix compares against alternatives. In our setup, a batch of B=1,000 real examples arrives, selection trims it to K=50, and distillation turns those into the m=5 examples we keep and train on. This achieves a 99.5% compression rate overall. We call this two-stage route Hybrid, and compare it against two alternatives. Direct drops the middle stage and goes from B straight to m in a single step. Double keeps both stages but runs the second one the same way as the first. Across all three routes, the distillation stage runs either GradMM or OLGA, and selection is done by uniform sampling.

Hybrid pipeline in offline setting. Table 2 compares the accuracy of the three routes in the offline setting, where the whole compression happens once over a static pool, and Table 2 compares what each route costs in time. With labels, the hybrid pipeline is the strongest route, averaging 86.0 percent across the five datasets, ahead of GradMM-Direct at 82.7 and GradMM-Double at 73.9, while costing less than distilling the whole batch directly or in two steps. This confirms that selecting early, while the compression is still easy, and distilling only at the extreme end is the more effective way through the bottleneck, and the cheaper one too. Without labels, the same funnel becomes a trade-off rather than a gain: OLGA-Hybrid falls short of OLGA-Direct by only 0.9 points on average while running 30% faster, which puts the two routes on par and makes the choice between them a balance of accuracy against time. What holds for both objectives is the lead over selection: GradMM-Hybrid and OLGA-Hybrid beat the selection-only routes on average, 86.0 and 80.0 against 73.4 for Selection-Double and 66.6 for Selection-Direct.

Table 2: Comparison of the hybrid, direct, and double routes in the offline and online settings across 5 datasets. At each time step, the data is compressed in stages from B=1,000 to K=50 to m=5. The offline setting runs a single step (T=1), and the online setting runs T=10 steps. Runtime is the wall-clock time of one offline distillation run on SST-2. On average, the hybrid route beats the direct and double routes for both objectives in both settings, except offline without labels, where OLGA-Hybrid is on par with OLGA-Direct. Bold marks the best result in each column, and underline marks the reference that each colored difference is measured against.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Route</td><td colspan="6">Offline accuracy</td><td colspan="3">Online stream-AUC</td><td rowspan="2">Runtime (s) ↓</td></tr><tr><td>SST-2 ↑</td><td>Rotten T. ↑</td><td>IMDB ↑</td><td>RT-Pol. ↑</td><td>TwitEmo. ↑</td><td>Avg. ↑</td><td>SST-2 ↑</td><td>IMDB↑</td><td>Avg. ↑</td></tr><tr><td rowspan="3">GradMM</td><td>Hybrid</td><td>88.2</td><td>83.9</td><td>92.3</td><td>82.1</td><td>83.4</td><td>86.0 (+3.3)</td><td>76.0</td><td>80.8</td><td>78.4(+3.1)</td><td>436.1 (−29.9%)</td></tr><tr><td>Direct</td><td>87.4</td><td>76.6</td><td>90.6</td><td>81.3</td><td>77.6</td><td>82.7</td><td>72.1</td><td>78.5</td><td>75.3</td><td>622.1</td></tr><tr><td>Double</td><td>79.7</td><td>71.1</td><td>80.7</td><td>67.0</td><td>70.8</td><td>73.9(−8.8)</td><td></td><td></td><td></td><td>2073.6 (+233.3%)</td></tr><tr><td rowspan="3">OLGA</td><td>Hybrid</td><td>87.8</td><td>81.2</td><td>89.0</td><td>84.6</td><td>57.3</td><td>80.0(−0.9)</td><td>71.2</td><td>78.4</td><td>74.8(+2.1)</td><td>447.5(−30.1%)</td></tr><tr><td>Direct</td><td>88.6</td><td>85.0</td><td>90.9</td><td>82.6</td><td>57.4</td><td>80.9</td><td>67.7</td><td>77.7</td><td>72.7</td><td>639.9</td></tr><tr><td>Double</td><td>67.4</td><td>71.2</td><td>87.6</td><td>76.1</td><td>58.9</td><td>72.2(−8.7)</td><td></td><td></td><td></td><td>2086.4 (+226.1%)</td></tr><tr><td rowspan="2">Selection</td><td>Double</td><td>82.5</td><td>73.6</td><td>89.8</td><td>52.9</td><td>68.0</td><td>73.4 (+6.8)</td><td>69.6</td><td>75.3</td><td>72.5 (+1.4)</td><td>5.5 (+14.6%)</td></tr><tr><td>Direct</td><td>81.9</td><td>54.7</td><td>87.9</td><td>50.3</td><td>58.1</td><td>66.6</td><td>67.2</td><td>75.0</td><td>71.1</td><td>4.8</td></tr></table>

![](images/0c062bc9f9319fd3e0690ee2c905c2822164bfb40f6cb1fa7eac850b0d21f203.jpg)  
Figure 3: (a) How each gradient matching objective builds its loss from the model’s next-token distribution; the dots mark which position it reads. (b) Three ways to fold a new batch into the memory: compress it first, pool it with the memory, or weight the two by hand.

Hybrid pipeline in online setting. We extend the comparison to the online setting, where a new batch of B examples arrives at each of T=10 steps. After each distillation, the resulting m examples are merged into the running rehearsal memory by the cascaded merge. We report only SST-2 and IMDB, since they are the only datasets large enough to supply a stream of this length.

The online half of Table 2 shows that GradMM-Hybrid performs best, with an average stream-AUC of 78.4. GradMM-Hybrid beats GradMM-Direct by 3.1 points and OLGA-Hybrid beats OLGA-Direct by 2.1 points, while distilling only 50 examples at each step instead of 1,000. Every route that uses distillation also beats selection alone on average, and the best of them leads the strongest selection-only route (72.5) by 5.9 points.

## 4.5 ABLATIONS

Alternative label-free objectives. The group-distribution loss ℓ<sup>grp</sup> introduced in OLGA is not the only way to turn the model’s own probabilities into a label-free signal. We compare it against two alternatives (Figure 3(a)). The pseudo-token signal appends the prompt “ In a single word:” and takes the most likely next token as its target, letting the model label itself in one word. The selfinformation signal finds the token in each sequence that the frozen model predicted least well and takes it as the target, on the view that this token carries most of what makes the example distinctive. Appendix A6 describes both signals further and gives their exact formulations.

Figure 4 compares the three signals. At m=5, group-distribution averages 75.5 percent across the five datasets, ahead of pseudo-token (74.1) and self-information (72.2), and it is the steadiest as the budget grows. The tokens each signal aims at help explain why. Pseudo-token aims at only 18 distinct tokens, spread over just 3.1 bits of entropy, so its training signal keeps pushing in the same few directions, while self-information spreads over 47 tokens (5.5 bits) and group-distribution over 100 (6.6 bits). The alternatives are also up to five times slower to run.

![](images/fb7cab5c4c84678608853ddd4b1efaf7770c8010e70c273c9a867f0e6e3c5329.jpg)

![](images/7fa88399af387e186142aa5bf470aafb326f1ecae85ee430bb49e496f2c5d1b9.jpg)

![](images/7a8b4ec1dd80a0624d5b0d79d3b5d60c7653659229c29477912dfb75ddcc063c.jpg)

![](images/d14da8cbbaf5d15d73ececfd9e4050f09113e0a56f22bcd22b284e8895b69477.jpg)

![](images/b002d95331dd4bb9f6ad0044383fd28e0d05f1667bf83056f65989df0bb58cab.jpg)  
Group-distribution (ours) Pseudo-token  
Self-information GradMM (supervised)  
Figure 4: Comparison of alternative label-free objectives. (Left) Accuracy against the number of examples m, per dataset and on average. (Right) Number of distinct tokens each signal aims at, and runtime relative to group-distribution. Group-distribution leads at m=5, stays steady at larger budgets, and runs fastest.

Table 3: Effect of the selector on the offline hybrid pipeline, averaged over the five datasets.  
Table 4: Comparison of online memory-merging strategies, averaged over the five datasets.
<table><tr><td>Selector</td><td>Avg. accuracy ↑ SST-2 runtime (s) ↓</td><td></td></tr><tr><td>Uniform (ours)</td><td>86.0</td><td>436.1</td></tr><tr><td>k-center</td><td>82.1</td><td>564.8</td></tr></table>

<table><tr><td>Merge strategy</td><td>Avg. stream-AUC ↑</td></tr><tr><td>Cascaded (ours)</td><td>72.0</td></tr><tr><td>One-pass</td><td>70.2</td></tr><tr><td>History-weighted</td><td>69.8</td></tr></table>

Alternative selectors for the hybrid pipeline. The hybrid routes use uniform sampling by default, but any selection method can take its place. We repeat the offline GradMM-Hybrid route with kcenter, the strongest selection baseline of Figure 2(b). On average, uniform sampling is both more accurate and faster than k-center (Table 3). This indicates that naive sampling, which keeps the original distribution of the data, is preferable in the hybrid pipeline, since distillation is able to synthesize higher-quality text from it. Appendix A7 reports the per-dataset numbers.

Alternative merge strategies. The cascaded merge is only one way to use distillation to fold an arriving batch into the running memory. We compare it against two alternatives (Figure 3(b)). The one-pass merge appends the old memory to the raw new batch and distills the two together in a single step. The history-weighted merge does the same single pass, but weights the old and new gradients by how much data each one stands for.

Table 4 extends the experiment of Section 4.3 for OLGA with the alternative merge strategies, at T=5 steps. The cascaded merge beats both on average, at 72.0 against 70.2 for one-pass and 69.8 for history-weighted. Appendix A8 reports the per-dataset performance.

## 5 CONCLUSION

Keeping a deployed language model current calls for updates that run on a handful of examples carrying the training signal of all data seen so far. OLGA extends gradient matching for text in three directions: removing the need for labels, following a data stream through a running rehearsal memory, and cutting cost with a hybrid pipeline that selects before it distills. Across five text classification datasets, it outperforms every competing method on average in both the offline and online settings, and our experiments show that each of the three directions matters. Together, they make continual updating of a language model accurate, affordable, and independent of ground-truth labels, offering a practical path toward keeping deployed language model applications current.

## AI USE STATEMENT

The research ideas and hypotheses, the method and its formulation in Section 3, the design of the experiments, the implementation of the methods, and the analysis and interpretation of the results are all our own work. Generative AI tools were used only in a supporting role. AI coding agents also wrote secondary engineering code, namely experiment scripts, job launchers, and the plotting code behind our figures, and large language models were used to edit the paper for readability. No generative AI tools were used to generate, clean, or reformat datasets, for translation, or for qualitative data analysis. The synthetic examples reported in the paper are the output of our method run with phi-1.5, not text produced by an AI assistant. We have reviewed all AI-assisted work. We reviewed every piece of code the agents produced. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

All experiments use publicly available datasets (SST-2, Rotten Tomatoes, RT-Polarity, IMDB, and TwitterEmotion) and the open-source phi-1.5 model. We will release our full implementation publicly on GitHub upon acceptance, and it reproduces all results reported in the paper.

## ETHICS STATEMENT

We have read and adhere to the ICLR Code of Ethics. Our experiments use only publicly available text classification datasets and an open-source language model, and involve no human subjects and no data collection of our own. Because the synthetic examples our method produces are built from phrases of the underlying data, they may carry over biases present in that data, which should be considered before deploying the method.

## REFERENCES

Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection for online continual learning. In Advances in Neural Information Processing Systems, volume 32, 2019.

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1–122, 2011.

George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu. Dataset distillation by matching training trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10718–10727, 2022.

Robert M. French. Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3(4):128–135, 1999.

Jianyang Gu, Kai Wang, Wei Jiang, and Yang You. Summarizing stream data for memoryconstrained online continual learning. Proceedings of the AAAI Conference on Artificial Intelligence, 38(11):12217–12225, 2024.

Jianheng Huang, Leyang Cui, Ante Wang, Chengyi Yang, Xinting Liao, Linfeng Song, Junfeng Yao, and Jinsong Su. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1416–1428, 2024.

Siddharth Joshi, Jiayi Ni, and Baharan Mirzasoleiman. Dataset distillation via knowledge distillation: Towards efficient self-supervised pre-training of deep networks. In The Thirteenth International Conference on Learning Representations, 2025.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings ofthe National Academy ofSciences, 114(13):3521–3526, 2017.

Jiangtao Kong, Jiacheng Shi, Ashley Gao, Shaohan Hu, Tianyi Zhou, and Huajie Shao. Hybrid memory replay: Blending real and distilled data for class incremental learning. arXiv preprint arXiv:2410.15372, 2024.

Angeliki Lazaridou, Adhi Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liska, Tayfun Terzi, Mai Gimenez, Cyprien de Masson d’Autume, Tomas Kocisky, Sebastian Ruder, Dani Yogatama, Kris Cao, Susannah Young, and Phil Blunsom. Mind the gap: Assessing temporal generalization in neural language models. In Advances in Neural Information Processing Systems, volume 34, pp. 29348–29363, 2021.

Dong Bok Lee, Seanie Lee, Joonho Ko, Kenji Kawaguchi, Juho Lee, and Sung Ju Hwang. Selfsupervised dataset distillation for transfer learning. In The Twelfth International Conference on Learning Representations, 2024.

David D. Lewis and William A. Gale. A sequential algorithm for training text classifiers. In SIGIR ’94, pp. 3–12. Springer, 1994.

Jeffrey Li, Mohammadreza Armandpour, Seyed Iman Mirzadeh, Sachin Mehta, Vaishaal Shankar, Raviteja Vemulapalli, Samy Bengio, Oncel Tuzel, Mehrdad Farajtabar, Hadi Pouransari, and Fartash Faghri. TiC-LM: A web-scale benchmark for time-continual LLM pretraining. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 32231–32273, 2025.

Yuanzhi Li, Sebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee.´ Textbooks are all you need II: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.

Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12):2935–2947, 2018.

Wenyang Liao, Quanziang Wang, Yichen Wu, Renzhen Wang, and Deyu Meng. Data-Distill-Net: A data distillation approach tailored for reply-based continual learning. arXiv preprint arXiv:2505.20135, 2025.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142–150, 2011.

Aru Maekawa, Naoki Kobayashi, Kotaro Funakoshi, and Manabu Okumura. Dataset distillation with attention labels for fine-tuning BERT. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 119–127, 2023.

Aru Maekawa, Satoshi Kosugi, Kotaro Funakoshi, and Manabu Okumura. DiLM: Distilling dataset into language model for text-level dataset distillation. In Findings of the Association for Computational Linguistics: NAACL 2024, pp. 3138–3153, 2024.

Arun Mallya and Svetlana Lazebnik. PackNet: Adding multiple tasks to a single network by iterative pruning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7765–7773, 2018.

Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of Learning and Motivation, volume 24, pp. 109–165. Elsevier, 1989.

Brian B. Moser, Federico Raue, Tobias C. Nauen, Stanislav Frolov, and Andreas Dengel. Distill the best, ignore the rest: A study in latent dataset distillation on core-sets. In 2025 International Joint Conference on Neural Networks (IJCNN), pp. 1–8. IEEE, 2025.

Dang Nguyen, Zeman Li, Mohammadhossein Bateni, Vahab Mirrokni, Meisam Razaviyayn, and Baharan Mirzasoleiman. Synthetic text generation for training large language models via gradient matching. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 46008–46025. PMLR, 2025.

Minh-Duong Nguyen, Thien-Thanh Dao, Le-Tuan Nguyen, Dung D. Le, and Kok-Seng Wong. Memory-efficient continual learning with prototypical exemplar condensation. arXiv preprint arXiv:2603.13804, 2026.

Keith Ando Ogawa, Bruno Lopes Yamamoto, Lucas Lauton de Alcantara, Victor Zacarias, Edson Bollis, Lucas Pellicer, Rosimeire Pereira Costa, Anna Helena Reali Costa, and Artur Jordao. Technical report on text dataset distillation. arXiv preprint arXiv:2512.03967, 2025.

Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics, pp. 115–124, 2005.

Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In Advances in Neural Information Processing Systems, volume 34, pp. 20596–20607, 2021.

Ameya Prabhu, Philip H. S. Torr, and Puneet K. Dokania. GDumb: A simple approach that questions our progress in continual learning. In Computer Vision – ECCV 2020, pp. 524–540. Springer, 2020.

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. iCaRL: Incremental classifier and representation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2001–2010, 2017.

Andrea Rosasco, Antonio Carta, Andrea Cossu, Vincenzo Lomonaco, and Davide Bacciu. Distilled replay: Overcoming forgetting through synthetic samples. In Continual Semi-Supervised Learning, pp. 104–117. Springer, 2022.

Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. CARER: Contextualized affect representations for emotion recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 3687–3697, 2018.

Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, volume 30, 2017.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1631–1642, 2013.

Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.

Rishabh Tiwari, Krishnateja Killamsetty, Rishabh Iyer, and Pradeep Shenoy. GCR: Gradient coreset based replay buffer selection for continual learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 99–108, 2022.

Gido M. van de Ven and Andreas S. Tolias. Three scenarios for continual learning. arXiv preprint arXiv:1904.07734, 2019.

Jeffrey S. Vitter. Random sampling with a reservoir. ACM Transactions on Mathematical Software, 11(1):37–57, 1985.

Mengting Wan, Tara Safavi, Sujay Kumar Jauhar, Yujin Kim, Scott Counts, Jennifer Neville, Siddharth Suri, Chirag Shah, Ryen W. White, Longqi Yang, Reid Andersen, Georg Buscher, Dhruv Joshi, and Nagu Rangan. TnT-LLM: Text mining at scale with large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 5836–5847, 2024.

Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A. Efros. Dataset distillation. arXiv preprint arXiv:1811.10959, 2018.

Yue Xu, Yong-Lu Li, Kaitong Cui, Ziyu Wang, Cewu Lu, Yu-Wing Tai, and Chi-Keung Tang. Distill gold from massive ores: Bi-level data pruning towards efficient dataset distillation. In Computer Vision – ECCV 2024, pp. 240–257. Springer, 2024.

Min Zeng, Haiqin Yang, Wei Xue, Qifeng Liu, and Yike Guo. Dirichlet continual learning: Tackling catastrophic forgetting in NLP. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of Proceedings of Machine Learning Research, pp. 4096– 4108. PMLR, 2024.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pp. 3987–3995. PMLR, 2017.

Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pp. 6514–6523, 2023.

Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset condensation with gradient matching. In International Conference on Learning Representations, 2021.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. WildChat: 1M ChatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations, 2024.

Zheng Zhou, Wenquan Feng, Qiaosheng Zhang, Shuchang Lyu, Qi Zhao, and Guangliang Cheng. ROME is forged in adversity: Robust distilled datasets via information bottleneck. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 78687–78713. PMLR, 2025.

## APPENDIX

## A1 DATA PARTITIONING AND CONSTRUCTION OF DATA STREAMS

Each dataset is partitioned into a training split and a validation split. SST-2, Rotten Tomatoes, and TwitterEmotion come with these splits by default, and we use them as provided, while IMDB and RT-Polarity do not, so we construct a fixed split for them once. Within each validation split, we hold out 100 examples, chosen with a fixed seed, as a source pool, and the remaining examples form the evaluation set, which serves every method, step, and experiment, and on which no experiment trains. The offline experiments draw their real anchors from the source pool, and the offline hybrid, which needs a much larger batch, draws it from the training split together with the source pool.

The online experiments stream the training split instead. We fix one random permutation of it and feed it in consecutive disjoint blocks, one block per step. The online hybrid streams the full official training split in the same way but balanced across groups, so each step receives a fresh batch of B=1,000 examples. Only SST-2 and IMDB have training splits large enough for this.

Budgets m, K, and B are counted per group throughout the paper. All five tasks are binary, so each stage actually handles twice the stated number of examples; the offline hybrid batch, for instance, holds B=1,000 examples of each group. Since the budgets before and after each compression stage double together, the compression rates are unchanged, and for simplicity we leave the number of groups out of the notation.

## A2 EXACT VALUES FOR THE OFFLINE RESULTS

Bar charts make the offline comparison easy to scan but hard to read exactly, so here we list the values behind Section 4.2 in full. Each method reduces 50 real examples to 5, phi-1.5 is fine-tuned on the result, and Table 5 reports its accuracy on the held-out validation split of each dataset, together with its gap to the best selection method.

Table 5: Exact values for Figure 2(a), the offline accuracy of phi-1.5 on each dataset. The upper block gives the supervised upper bounds and the lower block the label-free methods. OLGA beats the best selection method on four of the five datasets and by 10.2 points on average. Bold marks the best label-free result in each column, and underline marks the reference that each colored difference is measured against.
<table><tr><td>Method</td><td>IMDB↑</td><td>SST-2 ↑</td><td>Rotten Tom. ↑</td><td>RT-Polarity ↑</td><td>TwitEmo. ↑</td><td>Avg. ↑</td></tr><tr><td>Oracle</td><td>92.3 (+20.2)</td><td>88.2 (+12.9)</td><td>84.7 (+14.3)</td><td>83.2 (+12.8)</td><td>79.4(+21.9)</td><td>85.6 (+20.3)</td></tr><tr><td>GradMM (Nguyen et al., 2025)</td><td>91.4 (+19.3)</td><td>87.3 (+12.0)</td><td>83.7 (+13.3)</td><td>78.0 (+7.6)</td><td>78.6(+21.1)</td><td>83.8 (+18.5)</td></tr><tr><td>OLGA (ours)</td><td>89.5 (+17.4)</td><td>82.3 (+7.0)</td><td>74.5 (+4.1)</td><td>74.6 (+4.2)</td><td>56.5 (−1.0)</td><td>75.5 (+10.2)</td></tr><tr><td>Prototypical (Snell et al., 2017)</td><td>66.3 (-5.8)</td><td>65.4(-9.9)</td><td>68.5 (−1.9)</td><td>68.9(-1.5)</td><td>57.5</td><td>65.3</td></tr><tr><td>k-center (Sener &amp; Savarese, 2018)</td><td>72.1</td><td>75.3</td><td>60.1 (−10.3)</td><td>60.2(−10.2)</td><td>55.4(−2.1)</td><td>64.6(−0.7)</td></tr><tr><td>Herding (Rebuffi et al., 2017)</td><td>65.3 (−6.8)</td><td>61.2(−14.1)</td><td>69.8 (−0.6)</td><td>69.7 (−0.7)</td><td>56.3 (−1.2)</td><td>64.5 (−0.8)</td></tr><tr><td>Random</td><td>55.4(−16.7)</td><td>52.9 (–22.4)</td><td>70.4</td><td>70.4</td><td>56.2(-1.3)</td><td>61.1 (−4.2)</td></tr><tr><td>Uncertainty (Lewis &amp; Gale, 1994)</td><td>60.4(−11.7)</td><td>55.8(−19.5)</td><td>65.0(−5.4)</td><td>64.5 (−5.9)</td><td>54.0(−3.5)</td><td>59.9(−5.4)</td></tr><tr><td>Zero-shot</td><td>64.0(−8.1)</td><td>59.4(−15.9)</td><td>64.4(−6.0)</td><td>64.3 (−6.1)</td><td>43.5(-14.0)</td><td>59.1 (−6.2)</td></tr></table>

## A3 EXACT VALUES FOR THE ONLINE RESULTS

As with the offline results, we list the exact values behind the online comparison of Section 4.3 here. Data arrives over T=10 steps (8 for IMDB), each method keeps a memory of 5 examples, and Table 6 reports the resulting stream-AUC on each dataset, together with each entry’s gap to the best competing method.

Table 6: Exact values for Figure 2(b), the online stream-AUC of phi-1.5 on each dataset. Oracleaccumulate keeps all data seen so far and gives the upper bound. OLGA beats the best competing method by 6.6 points on average and comes within two points of supervised GradMM. Bold marks the best result in each column apart from the oracle, and underline marks the reference that each colored difference is measured against.
<table><tr><td>Method</td><td>IMDB ↑</td><td>SST-2 ↑</td><td>TwitEmo. ↑</td><td>Rotten Tom. ↑</td><td>RT-Polarity ↑</td><td> $\mathbf { A v g } . \uparrow$ </td></tr><tr><td>Oracle-accumulate</td><td> $9 2 . 1 \ ( + 4 . 8 ) $ </td><td> $8 8 . 4 \ : ( + 1 0 . 0 ) $ </td><td> $8 7 . 2 \ : ( + 2 7 . 5 )$ </td><td> $8 4 . 5 \ : _ { ( + 2 0 . 1 ) }$ </td><td> $8 4 . 3 \ : ( + 2 0 . 0 ) $ </td><td> $8 7 . 3 \ : ( + 1 8 . 2 ) $ </td></tr><tr><td>Online GradMM (ours)</td><td> ${ \bf 8 5 . 4 } _ { ( + 3 . 8 ) }$ </td><td> $7 9 . 2 \left( + 0 . 8 \right)$ </td><td> ${ \bf 6 5 . 3 } _ { ( + 5 . 6 ) }$ </td><td> $7 3 . 2 \ : ( + 8 . 8 ) $ </td><td> $7 4 . 3 \ : ( + 1 0 . 0 )$ </td><td> $7 5 . 5 \ : ( + 8 . 5 )$ </td></tr><tr><td>OLGA (ours)</td><td> $8 5 . 1 \ ( + 3 . 5 )$ </td><td> $7 4 . 2 ( - 4 . 2 )$ </td><td>59.6(−0.1)</td><td>74.1 (+9.7)</td><td> $7 4 . 9 \substack { _ { ( + 1 0 . 6 ) } }$ </td><td> $7 3 . 6 \ : ( + 6 . 6 )$ </td></tr><tr><td>k-center (Sener &amp; Savarese, 2018)</td><td>73.6(−8.0)</td><td>78.4</td><td>58.8 (−0.9)</td><td>61.3(−3.1)</td><td>62.9(−1.4)</td><td>67.0</td></tr><tr><td>Reservoir (Vitter, 1985)</td><td>81.6</td><td>70.3 (−8.1)</td><td>59.7</td><td>62.1 (−2.3)</td><td>52.3 (-12.0)</td><td>65.2(−1.8)</td></tr><tr><td>Herding (Rebuffi et al., 2017)</td><td>76.1 (-5.5)</td><td>67.7(−10.7)</td><td>50.5 (−9.2)</td><td>61.3(−3.1)</td><td>57.1 (−7.2)</td><td>62.5(−4.5)</td></tr><tr><td>Prototypical (Snell et al., 2017)</td><td>80.4(−1.2)</td><td>78.0(−0.4)</td><td>47.3 (−12.4)</td><td>55.3(−9.1)</td><td>54.7 (−9.6)</td><td>63.1 (-3.9)</td></tr><tr><td>Zero-shot</td><td>64.0(−17.6)</td><td> $5 9 . 4 ( - 1 9 . 0 ) $ </td><td>43.5 (−16.2)</td><td>64.4</td><td>64.3</td><td>59.1 (−7.9)</td></tr></table>

## A4 EXTENDING THE OFFLINE RESULTS TO $m { = } 1 0$ EXAMPLES

Distillation is designed for extreme compression, so it is worth checking how the picture changes when the budget is less tight. We repeat the offline comparison of Section 4.2 at m=10 examples, with the same methods and setup as Table 5. Table 7 gives the per-dataset results. The conclusions of Section 4.2 carry over, as OLGA remains the strongest label-free method on average, 1.7 points ahead of the best selection method, and supervised GradMM stays close to the oracle. The margin is narrower than at $m { = } 5 ,$ however, since selection improves faster as the budget grows and already overtakes OLGA on three of the five datasets at $m { = } 1 0$ . Distillation is strongest under extreme compression while selection catches up as the budget grows, which is the split the hybrid pipeline is built on and is consistent with its results in Table 2.

Table 7: Offline accuracy of phi-1.5 on each dataset at m=10, the counterpart to Table 5. The upper block gives the supervised upper bounds and the lower block the label-free methods. OLGA stays ahead of the best selection method by 1.7 points on average, but leads on only two of the five datasets. Bold marks the best label-free result in each column, and underline marks the reference that each colored difference is measured against.
<table><tr><td>Method</td><td>IMDB ↑</td><td>SST-2 ↑</td><td>Rotten Tom. ↑</td><td>RT-Polarity ↑</td><td>TwitEmo. ↑</td><td>Avg. ↑</td></tr><tr><td>Oracle</td><td>92.3 (+11.9)</td><td>88.2 (+6.2)</td><td>84.7 (+10.2)</td><td>83.2(+9.1)</td><td>79.4 (+16.9)</td><td>85.6 (+12.9)</td></tr><tr><td>GradMM (Nguyen et al., 2025)</td><td>90.6(+10.2)</td><td>87.2 (+5.2)</td><td>81.6 (+7.1)</td><td>82.6(+8.5)</td><td>78.9 (+16.4)</td><td> $8 4 . 2 \ : ( + 1 1 . 5 ) $ </td></tr><tr><td>OLGA (ours)</td><td>89.0 (+8.6)</td><td>78.9(−3.1)</td><td>70.1 (−4.4)</td><td>76.4(+2.3)</td><td>57.6(−4.9)</td><td> $7 4 . 4 ( + 1 . 7 ) $ </td></tr><tr><td>Prototypical (Snell et al., 2017)</td><td>78.8(−1.6)</td><td>80.1 (−1.9)</td><td>72.2(−2.3)</td><td>72.0(−2.1)</td><td>60.2(−2.3)</td><td>72.7</td></tr><tr><td>Herding (Rebuffi et al., 2017)</td><td>77.6(−2.8)</td><td>78.8(-3.2)</td><td>71.5(-3.0)</td><td>71.8 (-2.3)</td><td>58.4(−4.1)</td><td>71.6(−1.1)</td></tr><tr><td>Uncertainty (Lewis &amp; Gale, 1994)</td><td>70.4(−10.0)</td><td>70.5 (-11.5)</td><td>73.0(-1.5)</td><td>72.8 (-1.3)</td><td>62.5</td><td>69.8 (−2.9)</td></tr><tr><td>k-center (Sener &amp; Savarese, 2018)</td><td>80.4</td><td>82.0</td><td>61.1 (−13.4)</td><td>61.6(-12.5)</td><td>61.0(-1.5)</td><td>69.2 (-3.5)</td></tr><tr><td>Random</td><td>68.3 (−12.1)</td><td>66.1 (−15.9)</td><td>74.5</td><td>74.1</td><td>62.1 (−0.4)</td><td>69.0(-3.7)</td></tr><tr><td>Zero-shot</td><td>64.0(−16.4)</td><td>59.4(−22.6)</td><td>64.4(-10.1)</td><td>64.3 (−9.8)</td><td>43.5(-19.0)</td><td> $5 9 . 1 \ L _ { ( - 1 3 . 6 ) }$ </td></tr></table>

## A5 WHAT A DISTILLED EXAMPLE LOOKS LIKE

Because OLGA returns plain text rather than weights, its synthetic examples can be read directly. OLGA optimizes each synthetic example so that its training gradient matches that of many real examples at once, and then maps it back to real tokens. The text it settles on therefore reads as a blend of those real examples, taking a phrase from one and another phrase from a different one, and joining them into a sentence that appears nowhere in the data. Tables 8 to 10 show one such example for each of the remaining datasets, in the style of Table 1. In each table, the synthetic example comes first and the real examples it draws from are listed below it. A phrase shared between the two is highlighted in both places, and its superscript gives the number of the real example it comes from.

Table 8: Synthetic example distilled by OLGA for IMDB (positive), with the seven real examples it borrows its phrasing from. A shared phrase is highlighted in both places, and its superscript gives the number of the real example it comes from.
<table><tr><td>Synthetic</td><td>IMDLEG is  $\mathrm { g r e a t } ^ { 1 }$  because it is awell  $\mathrm { m a d e } ^ { 2 }$  and entertaining story thatkeeps  $\mathrm { y o u } ^ { 3 }$  engaged from beginning onwards and the well done production and the talented: and  $\mathrm { w e l l } ^ { 5 }$  donepeople who⁶ are involved in making the show are trulya great and a</td></tr><tr><td>Real</td><td>1. This is just a short comment but I stumbled onto this movie by chance and I loved it. The actingis great, the story is simple and touching . .. Seek it out. 2. I thought this was one of those really great films to see with a bunch of close friends. ... Although not a new concept this was a verywell madefilm. 3. Powers Boothe turns in a stellar performance as 1970&#x27;s cult figure Jim Jones of the Peoples Temple. ... The moviekeeps yoùriveted and is a must see for anyone. 4. A funny comedyfrom beginning to end! ... Creative story line with a very talented cast. I thoroughly enjoyed it! 5. terribly underrated with matt dillon and tom skerritt, good backdrop for solid story ... well</td></tr></table>

Table 9: Synthetic example distilled by OLGA for Rotten Tomatoes (negative), with the four real examples it borrows its phrasing from. A shared phrase is highlighted in both places, and its superscript gives the number of the real example it comes from.
<table><tr><td>Synthetic</td><td>The new movie was boring and had nointeresting  $\mathrm { o r } ^ { 1 }$  well developed action or a  $\mod ^ { 2 }$  one of the actors in the story 4</td></tr><tr><td></td><td>1. what the director can&#x27;t do is make either of val kilmer&#x27;s two personas interesting or worth caring about.</td></tr><tr><td rowspan="3">Real</td><td>2. the techno tux is good for a few laughs, as are chan and hewitt, but when sucha good 1design turns out to be a cheap knockoff ..</td></tr><tr><td>3. sadly, though manyof the actors throw off a spark or two when they first appear, they can&#x27;t generate enough heat in this cold vacuum of a comedy ...</td></tr><tr><td>4.the storydrifts so inexorably into cliches about tortured (and torturing) artists . . .</td></tr></table>

Table 10: Synthetic example distilled by OLGA for RT-Polarity (positive), with the three real examples it borrows its phrasing from. A shared phrase is highlighted in both places, and its superscript gives the number of the real example it comes from.
<table><tr><td>Synthetic</td><td>Amazing movie review provided by criticsis a great example of  $\mathrm { { h o w } } ^ { 2 }$  a single positive sentence can make  $\mathrm { a } ^ { 3 }$  significant impact on the</td></tr><tr><td rowspan="4">Real</td><td>1. mortonis a greatactress portraying a complex character, but morvern callar grows less com- pelling the farther it meanders from its shocking start.</td></tr><tr><td>2. a truly moving experience, and a perfectexample of howart – when done right – can help heal, clarify, and comfort.</td></tr><tr><td>3. australian actor/director john polson and award-winning english cinematographer giles</td></tr><tr><td>nuttgensmake aterrific effort at disguising the obvious with energy and innovation.</td></tr></table>

## A6 ALTERNATIVE LABEL-FREE GRADIENT SIGNALS

The group-distribution loss equation 3 is one of several ways to read a matching target off the frozen model. Here we give the exact definitions of the two alternatives compared against it in Section 4.5, pseudo-token and self-information. Both are built like $\ell ^ { \mathrm { s u p } }$ and ℓ<sup>grp</sup>, as a per-example cross-entropy toward a target the data supplies. Each substitutes into equation 1 in place of $\ell ,$ and the same loss forms both the real and the synthetic gradient.

Pseudo-token. Let u be the fixed suffix “ In a single word:” and write p ⊕ u for the prompt with that suffix appended. The loss is the cross-entropy toward the token the frozen model finds most likely to come next,

$$
\ell ^ { \mathrm { p t } } ( s , \pmb \theta ) = - \log p _ { \pmb \theta } ( \hat { v } ( s ) \mid p \oplus \pmb u ) , \qquad \hat { v } ( s ) = \arg \operatorname* { m a x } _ { { v } } p _ { \pmb \theta _ { 0 } } ( v \mid p \oplus \pmb u ) .
$$

The model is asked to name the example in one word and then trained to keep saying that word. Like $\ell ^ { \mathrm { s u p } }$ this puts all the weight on a single token, but the token is whatever the model volunteers rather than a class name we supply.

Self-information. No suffix is added here. Writing $s = ( p , r )$ with response tokens $r _ { 1 } , \ldots , r _ { n } ,$ the ${ \mathrm { q u a n t i t y } } - \log p _ { \pmb { \theta } _ { 0 } } ( \boldsymbol { r } _ { i } \mid p , \boldsymbol { r } _ { < i } )$ says how unexpected the frozen model found the token at position i. The loss is the cross-entropy at the position where it is largest,

$$
\ell ^ { \mathrm { s i } } ( s , \theta ) = - \log p _ { \theta } ( r _ { i ^ { \star } } \mid p , r _ { < i ^ { \star } } ) , \qquad i ^ { \star } ( s ) = \underset { 1 \leq i \leq n } { \operatorname { a r g m a x } } \left[ - \log p _ { \theta _ { 0 } } ( r _ { i } \mid p , r _ { < i } ) \right] .
$$

The reasoning is that the token the model predicts least well is the one that carries most of what makes the example distinctive.

Two differences from equation 3 follow from these definitions. Both targets above are single tokens, where the group-distribution target spreads over the k tokens of $\gamma _ { g } ^ { k } ;$ and both depend on the individual example, where the group-distribution target depends only on the group it belongs to. For the real anchors neither difference costs anything, since every target is computed once under $\pmb { \theta } _ { 0 }$ before matching starts. For the synthetic set, the second one does, because the synthetic sequences keep changing while the solver runs, so a target read off a sequence has to be found again as it changes, whereas the group-distribution target is fixed from the outset. This is what separates the three runtimes in Figure 4, and self-information is the most expensive of them because finding it target means scoring every position of every sequence inside the matching loop.

## A7 ALTERNATIVE SELECTORS FOR THE HYBRID PIPELINE

The first stage of the hybrid pipeline only has to thin out the incoming batch, so any selection method can run it. As in Section 4.5, we replace uniform sampling in the first stage of the offline GradMM-Hybrid route with k-center, the strongest selection method of Figure 2(b), and keep everything else fixed. Table 11 gives the per-dataset results. Uniform sampling is ahead of k-center on all five datasets, from 2.0 points on IMDB to 8.3 on TwitterEmotion, and is also faster to run. This supports keeping the original distribution of the data at the selection stage and leaving the rest of the compression to distillation.

Table 11: Per-dataset version of Table 3, the offline accuracy of the GradMM-Hybrid route with uniform sampling or k-center as its selector. Uniform sampling is more accurate on every dataset and faster to run. Bold marks the better selector in each column.
<table><tr><td>Selector</td><td>SST-2 ↑</td><td>Rotten Tomatoes ↑</td><td>IMDB ↑</td><td>RT-Polarity ↑</td><td>TwitterEmotion ↑</td><td>Avg. ↑</td><td>SST-2 runtime (s) ↓</td></tr><tr><td>Uniform (ours)</td><td>88.2</td><td>83.9</td><td>92.3</td><td>82.1</td><td>83.4</td><td>86.0</td><td>436.1</td></tr><tr><td>k-center</td><td>84.2</td><td>80.9</td><td>90.3</td><td>79.8</td><td>75.1</td><td>82.1</td><td>564.8</td></tr></table>

## A8 ALTERNATIVE MERGE STRATEGIES

The running memory can absorb each arriving batch in more than one way. As in Section 4.5, we compare the cascaded merge against two alternatives for folding each batch into the memory of OLGA, the one-pass and the history-weighted merge, over T=5 steps. Table 12 gives the per-dataset stream-AUC. The cascaded merge scores highest on SST-2, Rotten Tomatoes, and RT-Polarity, while one-pass leads on TwitterEmotion and history-weighted on IMDB. Each alternative wins on one dataset at the cost of the others, which leaves the cascaded merge ahead on average, at 72.0 against 70.2 and 69.8.

Table 12: Per-dataset version of Table 4, the online stream-AUC of OLGA under each merge strategy. The cascaded merge leads on three of the five datasets and on average. Bold marks the best result in each column.
<table><tr><td>Merge strategy</td><td>SST-2 ↑</td><td>Rotten Tomatoes ↑</td><td>IMDB ↑</td><td>RT-Polarity ↑</td><td>TwitterEmotion ↑</td><td>Avg. ↑</td></tr><tr><td>Cascaded (ours)</td><td>61.8</td><td>85.4</td><td>69.1</td><td>69.5</td><td>73.9</td><td>72.0</td></tr><tr><td>One-pass</td><td>60.4</td><td>78.7</td><td>69.3</td><td>62.5</td><td>80.1</td><td>70.2</td></tr><tr><td>History-weighted</td><td>57.2</td><td>85.3</td><td>72.4</td><td>68.1</td><td>66.0</td><td>69.8</td></tr></table>

## A9 MEMORY LEARNABILITY DIAGNOSTICS

This section asks whether a cheap property of the memory can tell in advance how well the model will do on the stream, without fine-tuning on it. We measure two such properties of each method’s memory on every dataset. Feature discriminability measures how well the memory’s examples separate the classes in the model’s feature space, and gradient contrast measures how different the training gradients of the classes are. Figure 5 plots each against stream-AUC. Neither one predicts stream-AUC on its own. The least-squares fits explain almost none of the variation, with $R ^ { 2 ^ { * } }$ of 0.03 for feature discriminability and 0.08 for gradient contrast, and memories with nearly the same score on either property end up tens of points apart in stream-AUC. The gradient-matching memories and the oracle do sit in the high-accuracy region, but so do some selection memories whose scores are very different from theirs. How well a memory trains the model therefore cannot be read off a single cheap property, and the choice of method, rather than a proxy score, has to drive how the memory is built.

![](images/4d5685aca38075e163b68a017779991ddac8da1c9d0addb620027a9460897a74.jpg)

![](images/c371a04fddfbe6fc70f6f0f29cc11617a5cdd0a6557081fb9a8b320535b1484b.jpg)  
OLGA (ours) k-center Reservoir Herding Prototypical OLS fit GradMM (supervised) Oracle  
Figure 5: Feature discriminability (left) and gradient contrast (right) of each memory against its stream-AUC, one point per dataset and method. Neither property predicts how well a memory trains the model.