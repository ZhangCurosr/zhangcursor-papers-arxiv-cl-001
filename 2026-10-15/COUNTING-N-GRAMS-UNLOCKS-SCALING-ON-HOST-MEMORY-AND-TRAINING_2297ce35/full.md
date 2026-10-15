# COUNTING N-GRAMS UNLOCKS SCALING ON HOST MEMORY AND TRAINING-FREE ADAPTATION

Ian Magnusson<sup>1</sup> Noah A. Smith<sup>1,2</sup> Pang Wei Koh<sup>1</sup> <sup>1</sup>University of Washington <sup>2</sup>Allen Institute for AI

## ABSTRACT

The scaling relationship of compute and performance for language models (LMs) has been extensively studied; more recently, models are being designed to scale memory use, as well. In particular, adding embeddings for n-grams scales memory separately from compute use, because each pass only processes individual embedding rows, which are much smaller than the storage used for the whole embedding table. However, existing methods train embeddings from scratch, which needs expensive GPU memory for fast reading and writing. Meanwhile, the ordinary memory of the host system is cheap and underutilized by these training methods. To scale on host memory, we introduce COUNT (Counting Offline Unlocks N-gram Transformers), which builds n-gram embeddings without adding learned parameters by storing precomputed n-gram counts on host. We use the corpus frequencies to weight together the LM’s own unembeddings of the next tokens most often following the current n-gram history. At a fixed compute budget, COUNT can be added as an additional input embedding for pretraining 1B-parameter LMs to improve validation loss as much as an LM with 57% more compute. If GPU memory is limited, using COUNT to let backbone parameters fill GPU memory gives better performance than splitting memory with additional learned n-gram embeddings. Beyond efficiency, COUNT allows switching in a different table of n-gram counts from a domain-specific corpus at inference time to adapt to a new task without any training.

## 1 INTRODUCTION

Compute and memory are both crucial resources for scaling language models (LMs). The tradeoff of compute and performance has been improved through optimal allocation of compute to parameters and training tokens (Kaplan et al., 2020; Hoffmann et al., 2022). Beyond compute, performance can also be bought by increased memory through architectures that scale sparsely activated parameters, such as by activating only a subset of experts in mixture-of-experts (MoE) models (Shazeer et al., 2017; Lepikhin et al., 2020; Fedus et al., 2022). The scaling of another type of sparsely activated parameters, input embeddings, has recently seen much exploration (Roy et al., 2022; Huang et al., 2025; Liu et al., 2026; Cheng et al., 2026). By embedding n-grams as well as individual tokens, many new parameters can be added to the embedding table while each lookup still processes only one row. Thus, compute scales only with the width of the embedding not the much larger total size of the embedding table.

Existing n-gram embedding work learns representations from scratch jointly with pretraining. Consequently, they are not easily offloaded from expensive GPU memory during training, because their weights and optimizer state must be read and written quickly, increasing communication complexity. Therefore these learned n-gram embeddings compete for GPU memory with the backbone model’s parameters. But a long line of language modeling research and settled knowledge has built representations of n-grams without expensive pretraining by instead counting the frequency of n-grams in a corpus (Jelinek & Mercer, 1980; Kneser & Ney, 1995; Brants et al., 2007). These methods refine the basic principle that the likelihood of a continuation can be estimated from counts of its occurrence after an n-gram history that is a suffix of the current context. Existing work integrates such count-based language modeling or related corpus-retrieval methods with neural LMs (Khandelwal et al., 2020; Li et al., 2022; Liu et al., 2025b). But unlike scaling dense input representations, these fuse raw next-token distributions at the output.

![](images/e21801d9ca7edef9014ec7ad7b2db028ac67dc713b3f6f54be29f887a55607dc.jpg)

![](images/67d0691b73ab04621f3d75e422027314dec21272b8d762313216b3921c1db8e1.jpg)  
Figure 1: Left: COUNT uses corpus n-gram frequencies to score continuations and gather rows of the unembedding matrix into an additional input embedding, $f _ { t } .$ . The count score $S ( w \mid h )$ can be stupid backoff (Brants et al., 2007), or any other frequency score based on coun $; ( h , w ) / \operatorname { c o u n t } ( h )$ Unlike n-gram embeddings learned from scratch, COUNT’s additional storage can be offloaded to host RAM at training time. Right: COUNT reaches validation loss equivalent to a standard transformer trained with 46–89% more compute. It does this without increasing compute by instead scaling n-gram statistics on otherwise unused host RAM by up to 797 GB. Solid curves are the compute-optimal fits of each method’s scaling law. Each violet guide shows the extra compute required to make a standard transformer reach the loss of an optimally trained COUNT run. Figure 2 shows task loss and downstream accuracy.

We propose COUNT, Counting Offline Unlocks N-gram Transformers, which builds n-gram embeddings with precomputed corpus next-token statistics stored in host RAM. COUNT connects the expressivity of learned embeddings with the read-only storage of count-based scores. For each n-gram history, corpus counts supply a fixed list of frequent next tokens. The backbone LM’s unembedding head already learns representations of these predicted next tokens, so COUNT uses frequency scores to weight and combine unembeddings of next tokens as a new input embedding summed with the token embedding. This adds no learned parameters, and since the large counts table is read-only, it does not need to use GPU memory. Since COUNT’s count tables are computed offline, they can also be cheaply recomputed over new corpora to change its embeddings at inference time with no additional parameter training.

We train COUNT on a range of backbone LMs from 60M–1B parameters and token to parameter ratios from 10 to 40, allowing us to observe how COUNT interacts with traditional parameter and token scaling. We find that standard transformers without COUNT would need 46–89% more compute to achieve the same language modeling loss as COUNT and 6–56% more compute to match downstream task loss. Moreover, COUNT achieves Pareto-optimal tradeoffs between loss and learned parameter count in comparisons with learned n-gram embeddings, which need more GPU RAM to increase performance comparably. COUNT also improves performance more than output-level fusion of corpus counts. We also experiment with swapping in target-domain n-gram statistics at inference time and get training-free improvements in domain-specific evaluations in math, biomedicine, and law. We release<sup>1</sup> n-gram counts of the full Dolma 3 pretraining corpus (Olmo et al., 2026), along with training code and pretrained checkpoints for COUNT and learned n-gram baselines.

## 2 METHOD

## 2.1 PRELIMINARIES

A language model predicts the next token $x _ { t + 1 }$ from prefix $x _ { 1 : t } ,$ where each token is an integer ID in a vocabulary V. Let d be the model’s hidden width (i.e., size of each vocabulary item’s

embedding vector), $E \in \mathbb { R } ^ { | \nu | \times d }$ its token embedding matrix, and $e _ { t } = E [ x _ { t } ]$ the embedding of the token at position t. We write $h _ { t } ^ { ( 0 ) } \in \mathbb { R } ^ { d }$ for the corresponding input to the first transformer block; conventionally, $h _ { t } ^ { ( 0 ) } = e _ { t }$

## 2.2 COUNT

COUNT (Figure 1) constructs n-gram embeddings as an additional input to the LM backbone without adding any learned parameters. We use a precomputed, frozen table $\mathcal { T } _ { K }$ that maps each of H stored n-gram histories to its K most frequent next tokens and their scores, derived cheaply from n-gram counts and approximating conditional probabilities of next tokens. $K \ll | \nu |$ , the number of continuations retained per history, is chosen to retain just the head of the next-token distribution, where the n-gram LM is more robust. Let $h _ { t , j }$ be the longest stored suffix of $x _ { 1 : t }$ at the j-th highest order of $n .$ . The lookup $\mathcal { T } _ { K } [ h _ { t , j } ]$ returns its top-K next tokens $w _ { k }$ and their count scores $S ( w _ { k } \mid h _ { t , j } )$ , such as the stupid-backoff scores in Equation 2.

We use the LM’s unembedding head as a basis $B \in \mathbb { R } ^ { | \nu | \times d }$ , whose rows we weight by these scores to form the COUNT embedding $f _ { t } .$ . Reusing this matrix avoids adding any learned parameters for COUNT. The continuation scores are sparse over the vocabulary, retaining only K candidates per history, but their weighted combination produces a dense vector in the model’s embedding space.

$$
\{ ( w _ { k } , S ( w _ { k } \mid h _ { t , j } ) ) \} _ { k = 1 } ^ { K } = { \mathcal { T } } _ { K } [ h _ { t , j } ] , \qquad f _ { t } = { \frac { 1 } { J } } \sum _ { j = 1 } ^ { J } \sum _ { k = 1 } ^ { K } S ( w _ { k } \mid h _ { t , j } ) B [ w _ { k } ] .\tag{1}
$$

We add the COUNT embedding and the token embedding $e _ { t }$ together as the input to the first transformer block, $h _ { t } ^ { ( 0 ) } = f _ { t } + e _ { t }$

## 2.3 N-GRAM COUNT-BASED SCORES

COUNT can make use of any count-based n-gram LM to score continuations. In our experiments we use stupid backoff (Brants et al., 2007) to score observed continuations by their count ratio and otherwise back off to a shorter history with a discount. All of this can be done offline, before pretraining, at a fraction of the cost of training a large LM on the same corpus.

Let $C ( h , w )$ count occurrences of a history h followed by token w in the reference corpus, $C ( h )$ count the history, and $C ( w )$ count the token. The table stores every history h of the retained lengths with $C ( h ) \geq c _ { \operatorname* { m i n } }$ , where $c _ { \mathrm { m i n } }$ is the history-count threshold. Stupid backoff scores:

$$
S ( w \mid h ) = \left\{ { \begin{array} { l l } { C ( w ) / \sum _ { w ^ { \prime } \in \mathcal { V } } C ( w ^ { \prime } ) , } & { h = \emptyset , } \\ { C ( h , w ) / C ( h ) , } & { h \neq \emptyset , C ( h , w ) > 0 , } \\ { 0 . 4 S ( w \mid h [ 1 : ] ) , } & { h \neq \emptyset , C ( h , w ) = 0 . } \end{array} } \right.\tag{2}
$$

Here ∅ is the empty history, whose score is the unigram frequency. The backoff scores $S ( w \mid h )$ need not sum to one and are used without renormalization

## 3 PRETRAINING EXPERIMENTS

## 3.1 BACKBONE LM

To represent a standard transformer backbone across a wide range of scales, we use configurations from the OLMo-core ladder up to 1B parameters and a token to parameter ratio of $D / { \bar { N } } = 4 0 $ a typical scale for pretraining development experiments. With these configurations we compare standard transformers, COUNT, or learned n-gram embeddings. We report training budgets as the token-to-parameter ratio $D / N$ , where D is the number of training tokens and N counts backbone parameters excluding input embeddings but including the output head. Our comparisons use $D / N \in$ {10, 20, 40}. We pretrain on allenai/dolma3\_mix-6T-1025-7B and use the same corpus to obtain n-gram counts. Appendix A.1 gives the model ladder and optimization settings.

## 3.2 BUILDING N-GRAM TABLES

We count n-grams through order six over the full Dolma 3 (Olmo et al., 2026) pretraining corpus and construct continuation scores using the stupid-backoff procedure in Section 2.3. To vary table size, we select which n-gram orders to store, prune low-count n-grams, discard histories below a historycount threshold, retain the top-K continuations, and quantize their scores. Since host RAM is inexpensive, our flagship experiments use the largest n-gram table which is sized to saturate the host RAM available on our compute nodes. That $7 9 \breve { 7 }$ GB table serves the 370M, 600M, 760M, and 1B backbones; the 60M, 100M, and 190M backbones use a 17.1 GB table built by the same procedure with tighter count thresholds, so COUNT’s table capacity grows with backbone size across the ladder. See Appendix A.2 for all configurations of our primary table.

## 3.3 LEARNED N-GRAM EMBEDDINGS

There are many approaches to learning n-gram embeddings in the literature (Section 7); we compare COUNT with Over-Encoding (Huang et al., 2025) as implemented in LongCat (Liu et al., 2026) for its released codebase and models, demonstrated at large scales. Over-Encoding augments token embeddings with learned representations of n-gram histories. Let $N _ { \mathrm { L } }$ be the maximum learned n-gram order. Each n-gram order $n \in \{ 2 , \ldots , \bar { N } _ { \mathrm { L } } \}$ uses R hash functions, indexed by r. Each hash selects a row from its own embedding table $T _ { n , r } \in \mathbb { R } ^ { M _ { n , r } \times d _ { s } }$ , where $M _ { n , r }$ is the number of rows and $d _ { s }$ is the n-gram embedding width. Learned projections $P _ { n , r } \in \mathbb { R } ^ { d \times d _ { s } }$ map the retrieved embeddings to model width, giving the n-gram contribution $q _ { t }$ and additively updated transformer input:

$$
q _ { t } = \sum _ { n = 2 } ^ { N _ { \mathrm { L } } } \sum _ { r = 1 } ^ { R } P _ { n , r } T _ { n , r } \left[ \left( \sum _ { j = 0 } ^ { n - 1 } x _ { t - j } | \mathcal { V } | ^ { j } \right) \bmod M _ { n , r } \right] , \qquad h _ { t } ^ { ( 0 ) } = e _ { t } + q _ { t } .\tag{3}
$$

The tables and projections are learned jointly with the backbone.

LongCat (Liu et al., 2026) includes a technique called embedding amplification, which normalizes and scales the combined input, $\sqrt { d }$ LayerNorm $( h _ { t } ^ { ( 0 ) } )$ ). Here LayerNorm normalizes across the hidden features with learned gain and bias. We use bigram and trigram embeddings with four hash tables per order and table width $d / 8$ , projecting each retrieved vector to model width d. Following LongCat’s guidance, we use a 1:1 ratio of learned input-embedding parameters to backbone parameters: input-token embeddings plus n-gram table parameters equal all remaining learned parameters, including the output head and projections.

We find that embedding amplification also works well with COUNT (Appendix B.6), so we apply it to both COUNT and learned n-gram embeddings. We also tried training standard transformers with embedding amplification and found it did not improve performance, so we use no embedding amplification for this baseline in our comparisons with COUNT and learned n-gram embeddings.

## 3.4 EVALUATION

We summarize OLMo-core’s language-modeling and downstream evaluation suites with three aggregates.

LM validation loss is the mean token cross-entropy, in nats, across 11 validation sources based on Paloma (Magnusson et al., 2024): C4, Dolma Books, Dolma Common Crawl, Dolma peS2o, Dolma Reddit, Dolma Stack, Dolma Wiki, ICE, M2D2 S2ORC, Pile, and WikiText-103.

Task loss is the mean bits per byte (BPB) of reference answers from 74 downstream task configurations spanning knowledge and science, code, math and numerical reasoning, commonsense, reading and cloze, and symbolic and pattern reasoning. Likelihood metrics such as task loss give more stable comparisons of small base LMs than discrete metrics, which are noisier across model sizes and training budgets (Magnusson et al., 2025; Heineman et al., 2025), so task loss is our primary downstream metric.

![](images/505025864dfd1ca25aff3c6e3b2c63dcec2338395f1e0919505816b07bc8e052.jpg)  
Figure 2: COUNT improves performance without increasing compute or learned parameters. By storing up to 797GB of n-gram frequencies, COUNT allows scaling on host RAM. Left: task loss against training compute at token to parameter ratios $D / N \in \{ 1 0 , \hat { 2 } 0 , 4 0 \}$ . The solid curves minimize fitted scaling laws for each arm, showing compute optimal performance (Hoffmann et al., 2022), and violet guides show the additional compute required to make a standard transformer reach the loss of an optimally trained COUNT run. Right: downstream accuracy for the same runs.

Downstream accuracy is the mean accuracy across the 41 multiple-choice configurations among those tasks, using length-normalized accuracy (candidate log probability divided by character count) where available and ordinary accuracy otherwise.

Appendix A.3 specifies all the individual tasks and shows the task-category breakdown. Each source or task configuration receives equal weight.

## 4 PRETRAINING RESULTS

## 4.1 SCALING HOST STORAGE SAVES GPU COMPUTE

Figure 1 (right) compares LM loss, and Figure 2 task loss and downstream accuracy, for standard transformers and COUNT across 60M–1B parameter backbone LMs. COUNT improves downstream accuracy over standard transformer at matched compute at almost all budgets, by storing up to 797GB of n-gram frequencies in host RAM.

We fit loss as a function of backbone size and training tokens separately for each method and loss metric, then select the fitted model/data allocation that minimizes loss at each compute budget (Hoff mann et al., 2022):

$$
\widehat { L } ( N , D ) = E + \frac { A } { N ^ { \alpha } } + \frac { B } { D ^ { \beta } } , \qquad \widehat { L } _ { \mathrm { o p t } } ( C ) = \operatorname * { m i n } _ { N , D : 6 N D = C } \widehat { L } ( N , D ) .\tag{4}
$$

Here E, A, B, α, and $\beta$ are all fitted parameters. In our experiments, using COUNT improves LM loss as much as using 46–89% more compute. Meanwhile, task loss with COUNT improves comparably to a 6–56% increase in compute. Like other n-gram memory techniques, COUNT increases performance by scaling memory usage, but it is important to remember that, unlike other methods, COUNT only needs host memory, which is abundant in practice and largely uncontended by other parts of the language modeling pipeline.

## 4.2 USING HOST RAM FOR N-GRAMS FREES GPU RAM FOR BACKBONE PARAMETERS

Figure 3 compares standard transformers, COUNT, and learned n-gram embeddings with 60M–1B backbones by total learned parameters. Learned parameters are the primary driver of additional

Standard transformer Learned n-gram only COUNT 60M backbone 100M backbone 190M backbone 370M backbone 600M backbone 760M backbone 1B backbone — Pareto frontier  
![](images/fd94ce25446bf9211f6531268d57de23b35e07cac756be591c400bf6130421e8.jpg)  
Figure 3: For a given budget of total learned parameters (a hardware-agnostic measure that drives GPU memory footprint) COUNT delivers best LM loss across a range of backbone model scales from 60M–1B, and downstream accuracy (bottom row) follows the same trend with more noise. Each column uses the indicated token to parameter ratio $D / N \in \{ 1 0 , 2 0 , 4 0 \}$ relative to that run’s own backbone.

GPU memory footprint, so this comparison allows hardware-agnostic assessment of optimality with respect to memory. The Pareto frontier mostly follows COUNT, because its improvements add no learned parameters. Even where learned n-gram embeddings produce comparable performance, they do so at the cost of additional learned parameters to store learned n-gram embeddings, which cannot be easily offloaded during training and contend with backbone model parameters for GPU memory. Each column fixes $D / \bar { N }$ , so training tokens scale with the backbone rather than with added embedding parameters. The Pareto frontier connects configurations with the lowest measured loss, or highest accuracy, for their parameter budget. The bottom row of Figure 3 shows downstream accuracy on the same axes; it follows the loss rows with more noise, and the frontier occasionally passes through a standard transformer or learned n-gram embedding point.

Late fusion (Appendix B.1), which combines count-based next-token distributions with neural predictions at the output, also keeps count tables in host RAM without adding learned n-gram parameters that compete with the backbone for GPU memory. However, constructing input embeddings with COUNT gives larger generalist gains. Also we find that activating late fusion during pretraining does not improve on purely inference-time late fusion. Inference-only late fusion remains a useful option when the pretraining process cannot be changed, but using COUNT at pretraining is best and costs only host RAM.

## 5 INFERENCE-TIME TABLE SWAPPING

Unlike tables of n-gram embeddings learned from scratch, COUNT tables store scores over a shared vocabulary of next tokens; COUNT maps these scores into embeddings using the same unembedding basis. This allows the n-gram table to be modified at inference time without any training. We test two use cases where table swapping is desirable: whether a pretrained COUNT model can adapt to a new domain-specific table or a smaller table for a tight deployed host-memory budget. Both experiments use COUNT pretrained on a 1B backbone at token-to-parameter ratios $\bar { D } / N \in \{ 1 0 , 2 0 , 4 \bar { 0 } \}$

![](images/d690f62746e9da105293572940e9f97abf82c3e78b6b48b7ecff8aec859f7dcd.jpg)  
Figure 4: The count-based continuation scores in COUNT can be cheaply fit on new domains and swapped in at inference time with no training. We find in-domain task loss improves on all domains when interpolating 1:3 between a domain-specific table and the domain general table used in pretraining this 1B parameter COUNT checkpoint. Accuracy results are more mixed but still increase on average for math and law domains. Each point is a result on one subtask at one training budget. Count-based late-fusion methods also support domain adaptation (Figure 10); COUNT combines this capability with the stronger pretrained improvements of early fusion.

## 5.1 ADAPTING TO A NEW DOMAIN WITHOUT TRAINING

To test an inference-time switch to a domain-specific n-gram count table, we construct math, biomedical, and law tables of 83.398, 81.723, and 80.151 GB, respectively, compared with 797.487 GB for the original general-domain table. The math table uses 10.66B tokens from the Dolmino Math subset of Olmo 3 midtraining data (Olmo et al., 2026). The biomedical and law tables use the health and crime-and-law topic splits of Dolma 3’s olmOCR science-PDF source, containing 95.39 and 12.34 billion tokens. All use the pretraining tokenizer and n-grams through order six; Appendix A.2 gives the construction settings.

For each order, we combine the tables’ candidate sets and interpolate their scores as $S _ { \lambda } ( w \mid h ) =$ $( 1 - \lambda ) S _ { \mathrm { g e n e r a l } } ( w \mid h ) + \lambda S _ { \mathrm { d o m a i n } } ( w \mid h )$ , with zero contribution from a table when a candidate is absent from its row. We vary the domain-table weight $\lambda \in \{ 0 , 0 . 2 5 , 0 . 5 , 1 \} ; \lambda = 0$ uses the original table and $\lambda = 1$ replaces it completely. We then select the eight highest-scoring candidates without renormalizing their scores. We measure task loss and task accuracy on matched domain evaluations: GSM8K, MATH, and Basic Skills arithmetic for math; MedQA, MedMCQA, LabBench, and ten MMLU biomedical subjects for biomedical; and the professional-law, international-law, and jurisprudence subsets of MMLU for law.

Figure 4 shows absolute changes in individual task results at $\lambda \ : = \ : 0 . 2 5$ , the interpolation value that makes loss decrease for all 3 domains. These are compared with each task’s score with the original table at the same training budget. Mathematical and legal task losses and accuracy improve more than biomedical, which shows mixed results. For reference, at $D / N = 2 0$ , COUNT with the generalist table reduces task loss by 0.010 BPB relative to the standard 1B transformer (Figure 2), comparable to the improvements from moderate domain weighting. The full interpolation grid is in Appendix B.3. Count-based late-fusion methods also support domain adaptation (Figure 10 in Appendix B.1); COUNT combines this capability with the stronger pretrained improvements of early fusion (Figure 9).

## 5.2 SHRINKING THE TABLE TO FIT A RAM-CONSTRAINED DEPLOYMENT

The amount of host RAM at training time may be greater than what is available at deployment, so we ask how performance will degrade if we use a smaller n-gram count table at inference time. We replace the same model’s original table with general-domain tables of 459.222, 142.037, 68.458, and 17.114 GB at each training budget. These tables vary in history-count and n-gram-count thresholds.

![](images/de68812a1a5bb6914fcd085244fa23c1d0ba9dc8e410889ac482840a6ebecf71.jpg)  
Figure 5: The n-gram table used in COUNT can be replaced at inference time with more compressed versions. This allows the method to work on host-RAM constrained deployments with a graceful and predictable performance degradation.

![](images/472ec1bcf9cdae7cd1600e502bf18fb78b3a7a445c3fa95491e558fed8e25e20.jpg)  
Figure 6: COUNT retains most of the training and inference throughput of a standard transformer. Its throughput is similar to learned n-gram embeddings, and further optimizations would not have far to go to recover both methods to full standard transformer throughput. These measurements use the 1B configurations for each method, and other configurations with smaller tables and lower J further improve COUNT’s throughput, see Appendix B.2.

Figure 5 shows how much table storage can be removed while retaining better performance than a standard transformer. On the 1B backbone COUNT checkpoints we can reduce storage by more than an order of magnitude while still improving on the standard transformer. Notably, this can be done at inference time without any training, as the larger and smaller tables all represent n-gram histories in the same format as next-token scores.

## 6 TRAINING AND INFERENCE THROUGHPUT

While COUNT adds no learned parameters and can hide the host-resident table lookup time behind backbone LM GPU compute, we still need to measure its costs in practice. Adding no learned parameters means no added cost in the C = 6ND sense and in parameter memory, but gathering selected unembedding rows is real work that the estimate omits and GPU footprint depends on more than parameters, so we measure throughput (Figure 6) and peak GPU memory (Table 10) directly. These experiments use the 1B backbone parameter checkpoints of COUNT, learned ngram embeddings, and standard transformer. We measure training throughput as median tokens per second (TPS) of optimization steps 50-100 on 8 H200 GPUs. For inference we measure prefill and decode on one H200 after warm-up batches. Appendix A.4 gives the measurement configurations and Appendix B.2 reports additional measured settings.

COUNT has a small impact on throughput at the 1B backbone scale, retaining 93.1–97.8% of standard transformer training throughput, 95.5–95.8% for prompt prefill, and 95.9–96.7% for autoregressive decoding. We use three optimizations to recover most of the standard transformer’s throughput: (1) compiled, multithreaded host lookup; (2) prefetching lookups for training and prefill tokens while the GPU processes the preceding batch; and (3) a fused embedding reduction that gathers and sums candidate vectors without storing them separately.

We also optimize the learned n-gram embedding baseline that we compare against: (1) we implement sparse optimizer updates for only the retrieved embedding rows, avoiding a full-table gradient update at each step. At 190M with a 17.1 GB learned n-gram table, full-table optimizer updates ran at 4.4% of standard transformer training throughput and updates to retrieved rows only at 96.3%, and the 1B learned baseline in Table 7 trains at 98.3%. (2) We implement inference-time host offloading to gather learned embedding rows on the CPU and transfer them to the GPU, retaining 95.9–96.0% of standard transformer prefill throughput and 93.9–96.0% of decode throughput, similar to COUNT.

## 7 RELATED WORK

## 7.1 N-GRAMS IN NEURAL LANGUAGE MODELS

There has been much work on learning embeddings for n-grams, especially in the past two years, with the motivation that scaling embeddings increases performance predictably (Tao et al., 2024). The core problem all such work grapples with is that independently representing all |V|<sup>n</sup> possible ngrams is quickly intractable. One line of work uses multiple hashes of the n-grams to allow collisions that partially cancel out over different hash functions (Roy et al., 2022; Huang et al., 2025; Liu et al., 2026). Other work builds embeddings of an n-gram based on learned representations of the tokens that it shares with lower n-gram orders (Zhou et al., 2026) or combines token embeddings that make up the current n-gram suffix at inference time without any training (Qu et al., 2026). Yet another approach is to use an auxiliary model to embed the n-grams (Yu et al., 2025). Similarly, the backbone model’s own hidden state can provide the contextual information, and additional embeddings are keyed just by the current unigram but further conditioned by the hidden state (Ding et al., 2026; Tseng & De Sa, 2026). This approach can also be used with hashed n-gram embeddings by having the hidden state of the model gate the inclusion of the embeddings (Cheng et al., 2026).

Most similar to our work, Neubig & Dyer (2016) actually mix count-based next-token distributions with neural predictions conditioned on some count-based n-gram features. There is also more recent late-fusion-only work that interpolates the neural LM’s output distribution with a nonparametric one at inference time, from nearest neighbors over a datastore of contexts (Khandelwal et al., 2020) or from unbounded-order Infini-gram counts (Liu et al., 2025b). Li et al. (2022) instead train a neural LM with scaled n-gram log probabilities added to its output logits, yielding a multiplicative combination of the two distributions. This work on late fusion of next-token frequencies also demonstrates domain adaptation at inference time (Khandelwal et al., 2020; Li et al., 2022). In contrast to these approaches, our method can be seen as an early fusion combination of count-based and neural LMs where the count-based next-token distribution is represented directly in the shared embedding space of the neural LM. We compare late fusion to COUNT in Appendix B.1 and find that late fusion doe not benefit from inclusion in pretraining and does not improve generalist performance as much as COUNT.

Outside of academic literature, nanoGPT speed-run submissions concurrent with COUNT (PR #367 and PR #380) and unpublished experiments discussed in a blog post by Infini-gram author Jiacheng Liu retrieve a sample of matching suffixes from training data and use their continuations to build a next-token hint from the token embeddings, which is then input to the LM backbone. Where these methods predict continuations from less than a hundred samples retrieved online, COUNT computes corpus frequencies over all occurrences in a full pretraining corpus offline. We compare also the use of the input token embedding or unembedding matrix as next-token basis in Appendix B.7 and find that our use of the unembedding performs best. These engineering experiments highlight further opportunities for throughput optimization of count-based embeddings, complementing our systematic analysis across 60M–1B backbone scales, training budgets, and 74 downstream task evaluations.

## 7.2 TOKENIZATION AND N-GRAMS

A highly related thread of work explores n-gram tokens for a different motivation. They address the arbitrariness of token segmentation by either exploring ways to segment further and then represent

combinations of tokens (Clark et al., 2022; Pagnoni et al., 2025) or by relaxing segmentation and allowing n-grams composed of many words to become single LM tokens (Liu et al., 2025a).

## 7.3 RETRIEVAL-AUGMENTED LANGUAGE MODELS

Though architecturally unrelated, retrieval-augmented LMs (Guu et al., 2020; Lewis et al., 2021; Borgeaud et al., 2022; Asai et al., 2023; Shao et al., 2024) share a motivation with COUNT: access to nonparametric information from a raw corpus can complement the parametric knowledge learned by the LM when it trains on the same data. Notably, retrieval also provides the affordance of zero-training domain adaptation. Where these methods practically differ from COUNT is that their nonparametric information is input directly as a token sequence into the LM, increasing context usage and compute. Also because COUNT stores only counts, its data store will always be much smaller than storing the raw corpus it is fit on.

## 8 CONCLUSION

High-bandwidth GPU memory is the scarcest and most expensive resource in current training and serving hardware, while host RAM is cheaper per byte and largely uncontended by the model itself. Learned n-gram embeddings buy their gains with parameters and optimizer state that must sit in GPU memory alongside the backbone. COUNT buys comparable gains with bytes on cheaper memory, leaving GPU memory for backbone parameters. Representing n-grams by counts also opens possibilities beyond efficiency. A table of counts is an aggregate statistic rather than a corpus, so sharing it is often less sensitive and more compact than sharing the documents it summarizes, which offers a new way to ingest data that cannot be released as text. N-gram frequencies can also be estimated from far less data than pretraining requires. With only a small amount of domain-specific data we fit count-based priors that steered a generalist, pretrained model to better in-domain performance. The long history of settled knowledge on count-based language modeling can likely produce useful next-token priors on even smaller corpora that are too small for parametric learning. Finally, because each row lists the next-token candidates and scores for a history, the prior a COUNT model consult can be read, corrected, or extended directly, by editing rows rather than by retraining.

## AI USE STATEMENT

In this work, we used generative AI tools for implementing methods. We have not used generative AI tools for other tasks with required disclosure (design or provide feedback on research methodology or experiments, clean and reformat dataset, support qualitative and thematic data analysis, interpret results, propose or refine hypotheses), and the rest of the disclosure tasks (generate synthetic data sets, help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, assist in the writing of proofs, assist with translation) are not applicable to this work. Additionally, we used generative AI tools to create or modify scientific figures or images, suggest experimental parameters, create or edit software code, draft parts of a research paper, edit a research paper to improve readability, identify relevant literature, format references, and propose keywords for a research paper. We have reviewed all AI-assisted work. The paper was fully revised multiple times by all authors. The final method implementation was manually reviewed by the first author. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work aims to make language models more efficient in order to lower the cost and environmental impact of building and running them. By moving n-gram memory from GPU memory to host RAM and adding no learned parameters, COUNT lets smaller models reach quality that would otherwise require more compute or more expensive hardware, and its tables can be built from modest corpora. We hope this helps a wider range of people and organizations, including those with limited compute or limited data, to train and use capable models. COUNT also changes what a data contributor has to share: an n-gram count table records how often continuations follow short contexts, not the documents themselves, and is therefore less sensitive than raw pretraining text, although high-order counts over rare contexts can still reveal fragments of the source and should be released with the same care as other derived statistics. Our experiments use publicly available pretraining data and evaluation suites.

## REPRODUCIBILITY STATEMENT

Section 2 defines COUNT and the stupid-backoff scoring it uses, and Section 3.3 defines the learned n-gram baseline in the same notation. Every pretraining run follows the OLMo-core ladder and recipe in Appendix A.1; the primary n-gram table’s construction is stated in full in Appendix A.2, together with the smaller general-domain tables and the domain-specific tables in Table 3. Section 3.4 and Appendix A.3 specify the evaluation suite, metrics, and aggregations, and Appendix A.4 records the hardware, batch sizes, and timing procedure behind every throughput number, with the complete measurements in Appendix B.2. Appendix B reports the additional results behind the ablations and inference-time sweeps. We release the n-gram counts extracted from the full Dolma 3 pretraining corpus, the table-building and training code, and pretrained checkpoints for COUNT and the learned n-gram baselines.

## ACKNOWLEDGMENTS

We would like to thank Will Merrill, Jiacheng Liu, and Shannon Shen for fruitful discussions on combining n-gram and neural LMs, and Yashas Samaga for discussions on model ladder experiments. IM is supported by the NSF CSGrad4US Fellowship. This material is based upon work supported by the U.S. National Science Foundation under Award Nos. 2313998 and 2413244. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. National Science Foundation. This work was supported by the Singapore National Research Foundation and the National AI Group in the Singapore Ministry of Digital Development and Information under the AI Visiting Professorship Programme (award number AIVP-2024-001) and the AI2050 program at Schmidt Sciences.

## REFERENCES

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: learning to retrieve, generate, and critique through self-reflection, 2023. URL https://arxiv.org/ abs/2310.11511.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):7432–7439, Apr 2020. ISSN 2159-5399. doi: 10.1609/aaai.v34i05.6239.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens, 2022. URL https://arxiv.org/abs/2112.04426.

Thorsten Brants, Ashok C. Popat, Peng Xu, Franz J. Och, and Jeffrey Dean. Large language models in machine translation. In Jason Eisner (ed.), Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL), pp. 858–867, Prague, Czech Republic, jun 2007. Association for Computational Linguistics. URL https://aclanthology.org/D07-1090/.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian,

Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Han Zhang, Yu-Kun Li, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. Conditional memory via scalable lookup: A new axis of sparsity for large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4968–4990. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.226.

Jonathan H. Clark, Dan Garrette, Iulia Turc, and John Wieting. Canine: Pre-training an efficient tokenization-free encoder for language representation. Transactions of the Association for Computational Linguistics, 10:73–91, 2022. doi: 10.1162/tacl a 00448. URL https: //aclanthology.org/2022.tacl-1.5/.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge, 2018. URL https://arxiv.org/abs/1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings ofthe 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4599–4610. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.naacl-main.365.

Ning Ding, Fangcheng Liu, Kyungrae Kim, Linji Hao, Kyeng-Hun Lee, Hyeonmok Ko, and Yehui Tang. MeKi: memory-based expert knowledge injection for efficient LLM scaling, 2026. URL https://arxiv.org/abs/2602.03359.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: a reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2368–2378. Association for Computational Linguistics, 2019. doi: 10.18653/v1/n19-1246.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity, 2022. URL https://arxiv.org/abs/2101. 03961.

Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, and Hannaneh Hajishirzi. OLMES: A standard for language model evaluations. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pp. 5020–5048, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8- 89176-195-7. doi: 10.18653/v1/2025.findings-naacl.282. URL https://aclanthology. org/2025.findings-naacl.282/.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: retrievalaugmented language model pre-training, 2020. URL https://arxiv.org/abs/2002. 08909.

David Heineman, Valentin Hofmann, Ian Magnusson, Yuling Gu, Noah Smith, Hanna Hajishirzi, Kyle Lo, and Jesse Dodge. Signal and noise: A framework for reducing uncertainty in language

model evaluation. In Advances in Neural Information Processing Systems 38, NeurIPS 2025, pp. 19286–19327. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2025. doi: 10.52202/085713-0576.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021a. URL https: //arxiv.org/abs/2009.03300.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset, 2021b. URL https://arxiv.org/abs/2103.03874.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego De Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George Van Den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent´ Sifre. Training compute-optimal large language models. In Advances in Neural Information Processing Systems 35, NeurIPS 2022, pp. 30016–30030. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2022. doi: 10.52202/068431-2176.

Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, and Xun Zhou. Overtokenized transformer: Vocabulary is generally worth scaling, 2025. URL https://arxiv. org/abs/2501.16975.

F. Jelinek and R. L. Mercer. Interpolated estimation of Markov source parameters from sparse data. In Proc. Workshop on Pattern Recognition in Practice, 1980.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, July 2021. ISSN 2076-3417. doi: 10.3390/app11146421.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.08361.

Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models, 2020. URL https://arxiv. org/abs/1911.00172.

R. Kneser and H. Ney. Improved backing-off for m-gram language modeling. In 1995 International Conference on Acoustics, Speech, and Signal Processing, volume 1, pp. 181–184 vol.1, 1995. doi: 10.1109/ICASSP.1995.479394.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, Nov 2019. ISSN 2307-387X. doi: 10. 1162/tacl a 00276.

Jon M. Laurent, Joseph D. Janizek, Michael Ruzo, Michaela M. Hinks, Michael J. Hammerling, Siddharth Narayanan, Manvitha Ponnapati, Andrew D. White, and Samuel G. Rodriques. LAB-Bench: measuring capabilities of language models for biology research, 2024. URL https: //arxiv.org/abs/2407.10362.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: scaling giant models with conditional computation and automatic sharding, 2020. URL https://arxiv.org/abs/2006. 16668.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe¨ Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks, 2021. URL https: //arxiv.org/abs/2005.11401.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Advances in Neural Information Processing Systems 35, NeurIPS 2022, pp. 3843–3857. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2022. doi: 10.52202/068431-0278.

Huayang Li, Deng Cai, Jin Xu, and Taro Watanabe. Residual learning of neural text generation with n-gram language model. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 1523–1533. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.findings-emnlp.109. URL https://aclanthology.org/2022. findings-emnlp.109/.

Alisa Liu, Jonathan Hayase, Valentin Hofmann, Sewoong Oh, Noah A. Smith, and Yejin Choi. SuperBPE: space travel for language models, 2025a. URL https://arxiv.org/abs/2503. 13423.

Hong Liu, Jiaqi Zhang, Chao Wang, Xing Hu, Linkun Lyu, Jiaqi Sun, Xurui Yang, Bo Wang, Fengcun Li, Yulei Qian, Lingtong Si, Yerui Sun, Rumei Li, Peng Pei, Yuchen Xie, and Xunliang Cai. Scaling embeddings outperforms scaling experts in language models, 2026. URL https://arxiv.org/abs/2601.21204.

Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, and Hannaneh Hajishirzi. Infinigram: Scaling unbounded n-gram language models to a trillion tokens, 2025b. URL https: //arxiv.org/abs/2401.17377.

Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Jha, Oyvind Tafjord, Dustin Schwenk, Evan Walsh, Yanai Elazar, Kyle Lo, Dirk Groeneveld, Iz Beltagy, Hannaneh Hajishirzi, Noah Smith, Kyle Richardson, and Jesse Dodge. Paloma: A benchmark for evaluating language model fit. In Advances in Neural Information Processing Systems 37, NeurIPS 2024, pp. 64338–64376. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024. doi: 10.52202/079017-2052.

Ian Magnusson, Nguyen Tai, Ben Bogin, David Heineman, Jena D. Hwang, Luca Soldaini, Akshita Bhagia, Jiacheng Liu, Dirk Groeneveld, Oyvind Tafjord, Noah A. Smith, Pang Wei Koh, and Jesse Dodge. DataDecide: how to predict best pretraining data with small experiments, 2025. URL https://arxiv.org/abs/2504.11393.

Graham Neubig and Chris Dyer. Generalizing and hybridizing count-based and neural language models. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1163–1172. Association for Computational Linguistics, 2016. doi: 10.18653/v1/ d16-1124.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2026. URL https: //arxiv.org/abs/2512.13961.

Artidoro Pagnoni, Ramakanth Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason E Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, and Srini Iyer. Byte latent transformer: Patches scale better than tokens. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9238–9258. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.acl-long.453.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. MedMCQA : A large-scale multi-subject multi-choice dataset for medical domain question answering, 2022. URL https: //arxiv.org/abs/2203.14371.

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi,´ Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534. Association for Computational Linguistics, 2016. doi: 10.18653/v1/p16-1144.

Yuwen Qu, Wenhui Dong, Chenyang Si, and Caifeng Shan. NGM: a plug-and-play training-free memory module for LLMs, 2026. URL https://arxiv.org/abs/2605.16893.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392. Association for Computational Linguistics, 2016. doi: 10.18653/v1/d16-1264.

Siva Reddy, Danqi Chen, and Christopher D. Manning. CoQA: a conversational question answering challenge. Transactions ofthe Associationfor Computational Linguistics, 7:249–266, Nov 2019. ISSN 2307-387X. doi: 10.1162/tacl a 00266.

Aurko Roy, Rohan Anil, Guangda Lai, Benjamin Lee, Jeffrey Zhao, Shuyuan Zhang, Shibo Wang, Ye Zhang, Shen Wu, Rigel Swavely, Tao Yu, Phuong Dao, Christopher Fifty, Zhifeng Chen, and Yonghui Wu. N-grammer: Augmenting transformers with latent n-grams, 2022. URL https: //arxiv.org/abs/2207.06366.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: an adversarial winograd schema challenge at scale. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8732–8740, Apr 2020. ISSN 2159-5399. doi: 10.1609/aaai.v34i05.6399.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. SocialIQA: commonsense reasoning about social interactions, 2019. URL https://arxiv.org/abs/1904. 09728.

Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim Dettmers, Sewon Min, Luke Zettlemoyer, and Pang Wei Koh. Scaling retrieval-based language models with a trillion-token datastore. In Advances in Neural Information Processing Systems 37, NeurIPS 2024, pp. 91260–91299. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024. doi: 10.52202/079017-2896.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017. URL https://arxiv.org/abs/1701.06538.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: a question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158. Association for Computational Linguistics, 2019. doi: 10.18653/v1/n19-1421.

Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, and Ngai Wong. Scaling laws with vocabulary: Larger models deserve larger vocabularies. In Advances in Neural Information Processing Systems 37, NeurIPS 2024, pp. 114147–114179. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024. doi: 10.52202/079017-3626.

Albert Tseng and Christopher De Sa. L<sup>3</sup>: Large lookup layers, 2026. URL https://arxiv. org/abs/2601.21461.

David Wadden, Kejian Shi, Jacob Morrison, Alan Li, Aakanksha Naik, Shruti Singh, Nitzan Barzilay, Kyle Lo, Tom Hope, Luca Soldaini, Shannon Zejiang Shen, Doug Downey, Hannaneh Hajishirzi, and Arman Cohan. SciRIFF: A resource to enhance language model instruction-following over scientific literature. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural

Language Processing, pp. 6072–6109, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.310. URL https://aclanthology.org/2025.emnlp-main.310/.

Alexander Wettig, Kyle Lo, Sewon Min, Hannaneh Hajishirzi, Danqi Chen, and Luca Soldaini. Organize the web: Constructing domains enhances pre-training data curation, 2025. URL https://arxiv.org/abs/2502.10341.

Da Yu, Edith Cohen, Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Daogao Liu, and Chiyuan Zhang. Scaling embedding layers in language models. In Advances in Neural Information Processing Systems 38, NeurIPS 2025, pp. 36060–36086. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2025. doi: 10.52202/085713-1074.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800. Association for Computational Linguistics, 2019. doi: 10.18653/v1/p19-1472.

Ziyin Zhang, Zhaokun Jiang, Lizhen Xu, Hongkun Hao, and Rui Wang. Multiple-choice questions are efficient and robust LLM evaluators, 2024. URL https://arxiv.org/abs/2405. 11966.

Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, Yuning Qiu, Qibin Zhao, and Danilo Mandic. Tensorizing engram: Sharing latents across n-gram embeddings is beneficial in LLMs, 2026. URL https://arxiv.org/abs/2606.08347.

## A EXPERIMENTAL DETAILS

## A.1 OLMO-CORE

Table 1 lists the backbone geometries and Table 2 the training recipe. All arms use the same backbone at each ladder size; size labels are shorthand rather than exact parameter counts. For example, the 1B configuration has $N = 1 , 2 7 9 , 3 9 5 , 8 4 0$ parameters under our budget convention. Token budgets use this exact parameter count, excluding input embeddings and added n-gram parameters but including the output head, and are rounded to full optimizer steps.

Table 1: OLMo-core model ladder configurations. The feed-forward width is the intermediate width of each transformer block.
<table><tr><td>Ladder label</td><td>Layers</td><td>Hidden width</td><td>Heads</td><td>Feed-forward width</td></tr><tr><td>60M</td><td>8</td><td>384</td><td>8</td><td>1,536</td></tr><tr><td>100M</td><td>12</td><td>512</td><td>8</td><td>2,048</td></tr><tr><td>190M</td><td>12</td><td>768</td><td>12</td><td>3,072</td></tr><tr><td>370M</td><td>16</td><td>1,024</td><td>16</td><td>4,096</td></tr><tr><td>600M</td><td>16</td><td>1,280</td><td>16</td><td>5,120</td></tr><tr><td>760M</td><td>16</td><td>1,536</td><td>16</td><td>6,144</td></tr><tr><td>1B</td><td>16</td><td>2,048</td><td>16</td><td>8,192</td></tr></table>

Table 2: Training configuration. N is the backbone parameter count defined above.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Tokenizer</td><td>Dolma3tokenizer(allenai/dolma2-tokenizer)</td></tr><tr><td>Vocabulary</td><td>100,278 tokens; embedding allocation padded to 100,352 rows</td></tr><tr><td>Sequence length</td><td>8,192 tokens</td></tr><tr><td>Position encoding</td><td>Rotary position embeddings with base 500,000</td></tr><tr><td>Precision</td><td>Bfloat16 model computation; float32 gradient reduction</td></tr><tr><td>Optimizer</td><td>AdamW with anomalous-update skipping (SkipStepAdamW);  $\beta _ { 1 } =$  0.9;  $\beta _ { 2 } = 0 . 9 9$  for 60M and 100M and 0.95 for  $1 9 0 \mathbf { M } \mathbf { - } 1 \mathbf { B } ; \epsilon = 1 0 ^ { - 8 }$ </td></tr><tr><td>Weight decay</td><td>0.1; input embeddings, embedding-amplification parameters, and learned n-gram tables and projections are exempt</td></tr><tr><td>Peak learning rate</td><td> $0 . 0 0 2 3 5 ( N / 1 0 8 , 0 0 0 , 0 0 0 ) ^ { - 1 / 3 }$ </td></tr><tr><td>Schedule</td><td>Warmup over N tokens, followed by a stable learning rate and linear decay to zero over the final 10% of each successive reporting interval</td></tr><tr><td>Token budgets</td><td>(WSD-S) Reporting endpoints at  $D / N = 1 0 , 2 0 .$  and 40</td></tr><tr><td>Global batch</td><td>262,144 (60M and 100M), 524,288 (190M), 786,432 (370M), 983,040 (600M), 1,310,720 (760M), and 1,572,864 (1B)</td></tr><tr><td>COUNT</td><td>Unembedding matrix B as the next-token basis; tables store  $K = 3 2$  candidates per history and training uses  $K = 8$  and  $J = 6$  with em-</td></tr><tr><td>dings</td><td>bedding amplification; 17.1 GB table at 60M–190M and 797.5 GB at 370M-1B Learned n-gram embed- Orders two and three, four hash tables per order, embedding width  $d / 8$  projected to  $d ;$  rows set by the 1:1 ratio: 0.40M rows across the eight tables at 60M, 0.79M at 100M, 1.19M at 190M, 2.11M at 370M, 2.63M at 600M, 3.16M at 760M, and 4.21M at 1B (0.04–2.16 GB of bfloat16 table weights). The table weights are updated with sparse row-wise AdamW on retrieved rows, using the backbone learning-rate schedule and betas, with zero weight decay</td></tr></table>

![](images/01c82246f606df414b3e8aad7fefdff51d0c7fded5459f5764dcc6030752b56b.jpg)  
Figure 7: The size of our n-gram tables depends on several configurations. History and n-gram occurrence thresholds limit the number of rows in the table. Each order of n-gram is its own set of rows in the table, with orders four through six accounting for most of the rows. Finally lowering K to store fewer top continuations reduces the table width.

## A.2 N-GRAM TABLES

We build smaller tables from intermediate artifacts, which have their own count thresholds and limits on the number of continuation candidates retained per history.

Our primary general-domain table stores the top K = 32 continuations for every history of order one through six seen at least 265 times, after dropping order-4 to order-6 n-grams with fewer than seven occurrences, and occupies 797.487 GB.

Figure 7 shows how the general-domain table sizes depend on the n-gram order, the history-count threshold, the per-n-gram count floor, and K; sizes are decimal gigabytes including headers. For storage, we quantize $\log _ { 1 0 } S ( w \mid h )$ to 256 levels per order, spanning the smallest retained finite log score through zero, and convert the decoded values to natural-log scores at read time.

The domain tables in Table 3 are built from the Dolmino Math subset of Dolma 3 (10.66B tokens) and from the Health and Crime and Law topics of Dolma 3’s olmOCR science-PDF source (95.39B and 12.34B tokens), following the WebOrganizer topic categories (Wettig et al., 2025; Olmo et al., 2026). All use the pretraining tokenizer, orders through six, a count floor of one, and a backoff factor of 0.4; the biomedical and law history-count thresholds are the lowest integers that fit within the math table’s 83.398 GB.

Table 3: Domain-specific tables used in Figure 4. Sizes are decimal GB with 8-bit scores.
<table><tr><td colspan="4">History-count</td></tr><tr><td>Domain</td><td>Size (GB)</td><td>K</td><td>threshold</td></tr><tr><td>Math</td><td>83.398</td><td>32</td><td>4</td></tr><tr><td>Biomedical</td><td>81.723</td><td>32</td><td>45</td></tr><tr><td>Law</td><td>80.151</td><td>32</td><td>8</td></tr></table>

## A.3 EVALUATION BREAKDOWN

Our downstream evaluations use OLMES task configurations (Gu et al., 2025) through ai2-olmo-eval 0.9.0, except for the GSM8K and MATH accuracy evaluations described below. LM validation loss averages token cross-entropy within each of the 11 sources before averaging sources. Task loss averages the 74 per-configuration BPB values, which include reference-answer scoring for code, math and open-ended tasks that report no accuracy; the accuracy aggregate covers the 41 multiple-choice configurations. Figure 8 groups the configurations into task categories; sub ject subsets, splits, prompt formats and language variants count separately, and every configuration keeps equal weight. The configurations derive from the following parent benchmarks.

• Knowledge and science: ARC-Easy and ARC-Challenge (Clark et al., 2018); MMLU (Hendrycks et al., 2021a), as its four subject groups on the validation and test splits; LAB-Bench DbQA and ProtocolQA (Laurent et al., 2024); MedQA (Jin et al., 2021); MedMCQA (Pal et al., 2022); Natural Questions (Kwiatkowski et al., 2019); Jeopardy (Gu et al., 2025); Basic Skills common knowledge (Olmo et al., 2026).

• Code: HumanEval (Chen et al., 2021); MBPP (Austin et al., 2021); MT-MBPP, machinetranslated MBPP in 17 further programming languages (Olmo et al., 2026); Basic Skills coding (Olmo et al., 2026).

• Math and numerical reasoning: GSM8K (Cobbe et al., 2021); MATH (Hendrycks et al., 2021b) in the Minerva format (Lewkowycz et al., 2022), as seven subject subsets; DROP (Dua et al., 2019); Basic Skills arithmetic (Olmo et al., 2026). GSM8K and MATH accuracy in Figure 4 uses the four-choice versions of these test sets, whose distractor answers were produced by Zhang et al. (2024); the model ranks the answer texts as for every other accuracy task.

• Commonsense: CommonsenseQA (Talmor et al., 2019); HellaSwag (Zellers et al., 2019); PIQA (Bisk et al., 2020); SocialIQA (Sap et al., 2019); WinoGrande (Sakaguchi et al., 2020).

• Reading and cloze: CoQA (Reddy et al., 2019); LAMBADA (Paperno et al., 2016); SQuAD (Rajpurkar et al., 2016); QASPER yes/no (Dasigi et al., 2021); SciRIFF yes/no (Wadden et al., 2025).

• Symbolic and pattern reasoning: Basic Skills logical reasoning, pattern recognition and string operations (Olmo et al., 2026); CopyColors (Gu et al., 2025).

ARC, MMLU, CommonsenseQA, PIQA and SocialIQA each use both prompt formats, rankedchoice (mc) and completion (rc); every other benchmark uses its completion format only.

![](images/7840f22ca7b60e8abf610dee73ae762b4d5d8eba34a52b5d3be2542fd1dfffdd.jpg)  
Figure 8: Composition of the task-loss (BPB) and accuracy (ACC) aggregates by qualitative task category. Each eligible configuration contributes once: 74 configurations for BPB and 41 for accuracy. Subject subsets, splits, prompt formats, and language variants are counted separately. Categories describe task families rather than isolated skills.

## A.4 THROUGHPUT CONFIGURATIONS

All throughput measurements use the 1B ladder configuration, include lookup and host–GPU transfer in the timed workload, use the 1:1 input-embedding-to-backbone parameter ratio for the learned n-gram baseline, keep COUNT tables in host RAM, and (for inference) use D/N = 40 checkpoints on C4 validation prompts. Compiled lookup searches sorted histories with multiple CPU threads; lookup caching reuses retrieved results; prefetch prepares lookups while the GPU processes the preceding batch; fused embedding combines row gathering and weighted summation; sparse updates optimize only retrieved learned n-gram rows; host offloading keeps learned n-gram tables on the CPU with their projections on the GPU; KV caching and CUDA graphs are the usual inference optimizations. Tables 4–6 give the settings for each workload.

Table 4: Training TPS experiment configuration.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Hardware</td><td>Eight H200 GPUs</td></tr><tr><td>Batch per GPU</td><td>Four sequences; six gradient-accumulation steps</td></tr><tr><td>COUNT table</td><td> $1 7 . 1 \ \mathrm { o r } \ \bar { 7 } 9 7 . 5 \ \mathrm { G B }$ </td></tr><tr><td>COUNT selection</td><td> $K = 8 ; J = 6 \mathrm { o r } 1$ </td></tr><tr><td>Learned table</td><td>GPU-resident</td></tr><tr><td>Optimizations</td><td>Prefetch; fused embedding; sparse updates</td></tr><tr><td>Runtime</td><td>FlashAttention 3; block/head compilation; fully sharded data paral- lelism</td></tr><tr><td>Timing</td><td>100 steps; median of six TPS samples at steps 50–100</td></tr></table>

Table 5: Prefill TPS experiment configuration.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Hardware</td><td>One H200 GPU</td></tr><tr><td>Batch sizes</td><td>1 and 8 sequences</td></tr><tr><td>Prompt length</td><td>8,192 tokens</td></tr><tr><td>COUNT table</td><td>797.5 GB; K = 8 per order, orders 1–6</td></tr><tr><td>Learned table</td><td>2.008 GiB; GPU-resident or host offloading</td></tr><tr><td>Optimizations</td><td>Compiled lookup; lookup caching; prefetch; fused embedding; KV caching</td></tr><tr><td>Lookup workers</td><td>Eight workers; ten batches of lookahead</td></tr><tr><td>Runtime</td><td>PyTorch attention</td></tr><tr><td>Timing</td><td>Three warmup batches; 96 timed batches at batch 1 or 12 at batch 8</td></tr></table>

Table 6: Decode TPS experiment configuration. Prompt processing and graph capture are excluded from the timed generation steps.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Hardware</td><td>One H200 GPU</td></tr><tr><td>Batch sizes</td><td>1 and 256 sequences</td></tr><tr><td>Prompt / generation</td><td>512 prompt tokens; 256 greedily generated tokens</td></tr><tr><td>COUNT table</td><td>797.5 GB; K = 8 per order, orders 1–6</td></tr><tr><td>Learned table</td><td>2.008 GiB; GPU-resident or host offloading</td></tr><tr><td>Optimizations</td><td>Compiled lookup; lookup caching; fused embedding; KV caching;</td></tr><tr><td>Lookup threads</td><td>CUDA graphs Eight</td></tr><tr><td>Runtime</td><td>PyTorch attention</td></tr><tr><td>Timing</td><td>Ten warmup steps; 246 timed generation steps</td></tr></table>

## B ADDITIONAL RESULTS

## B.1 LATE FUSION

We compare COUNT with two output-side fusion variants using the same tables and $K = 8 ,$ but only the longest matching history $h _ { t , 1 }$ rather than averaging orders, as the neural LM cannot correct noisy low order predictions in late fusion. Let $p _ { t } ( w ) = p _ { \theta } ( w \mid x _ { 1 : t } )$ be the standard transformer’s next-token distribution and W the candidates returned by $\mathcal { T } _ { K } [ h _ { t , 1 } ]$ , with scores $S ( w \mid h _ { t , 1 } )$ from Equation 2. Write $\begin{array} { r } { A _ { t } = \sum _ { w \in \mathcal { W } _ { \star } } S ( w \mid h _ { t , 1 } ) } \end{array}$ for their retained mass. Additive fusion follows the probability interpolation used by kNN-LM (Khandelwal et al., 2020) and infini-gram (Liu et al., 2025b), adapted here to top-K stupid-backoff scores by capping the retained mass at one:

$$
\begin{array} { r l r } & { } & { q _ { t } ( w ) = \cfrac { S ( w \mid h _ { t , 1 } ) } { \operatorname* { m a x } \{ 1 , A _ { t } \} } , ~ m _ { t } = \operatorname* { m i n } \{ A _ { t } , 1 \} , } \\ & { } & { p _ { t } ^ { \mathrm { a d d } } ( w ) = ( 1 - \lambda m _ { t } ) p _ { t } ( w ) + \lambda q _ { t } ( w ) . ~ } \end{array}\tag{5}
$$

Multiplicative fusion instead keeps at least $\varepsilon = 1 0 ^ { - 4 }$ total mass for non-candidates, distributed uniformly:

$$
\widehat { q } _ { t } ( w ) = \left\{ \begin{array} { l l } { \widetilde { m } _ { t } = \operatorname* { m i n } \{ A _ { t } , 1 - \varepsilon \} , } \\ { \displaystyle { S ( w \mid h _ { t , 1 } ) } } \\ { \operatorname* { m a x } \{ 1 , A _ { t } / ( 1 - \varepsilon ) \} } \\ { \displaystyle { \frac { 1 - \widetilde { m } _ { t } } { | \mathcal { V } | - | \mathcal { W } _ { t } | } } , } & { w \notin \mathcal { W } _ { t } . } \end{array} \right.\tag{6}
$$

Following the logit-level combination of Li et al. (2022), the fused distribution is the normalized product

$$
p _ { t } ^ { \mathrm { m u l t } } ( w ) = \frac { p _ { t } ( w ) \widehat { q } _ { t } ( w ) ^ { \alpha } } { \sum _ { v \in \mathcal { V } } p _ { t } ( v ) \widehat { q } _ { t } ( v ) ^ { \alpha } } .\tag{7}
$$

For the inference-only comparisons below, we freeze the standard transformer without embedding amplification and fix $\dot { \lambda } = \bar { 0 } . 1 \mathrm { o r } \alpha = 0 . 1$ . Domain adaptation uses the domain table directly; its changes are measured against the same unfused checkpoint, while COUNT’s changes are relative to its own generalist-table checkpoint.

Figure 9 shows that COUNT achieves better generalist performance than either additive or multi plicative late fusion. Figure 10 shows that training-free adaptation is an affordance shared by countbased methods; COUNT combines this capability with better pretrained generalist performance.

![](images/636157d41fe83ff2debc737d724eed302c4b9edeb721baa710748fa783d39be7.jpg)

![](images/1760079a4e0ce0ef0510738b85a65f70899bc8e0c92e9ac87b89294c5986869a.jpg)

![](images/150ee578d4097aefbdd6c8406816e56f3f2997a312990580cdb116c4e50ce869.jpg)  
Figure 9: Generalist inference-only late fusion compared with COUNT and the standard transformer.

Pretraining versus inference-time fusion. At 190M, we also pretrain each late-fusion arm with its scalar weight learned from an initialization of 0.1, using the same 17GB generalist table as its inference-only counterpart. Figure 11 shows that pretraining with late fusion does not improve performance over applying late fusion only at inference time, so we use inference-only fusion at the other scales.

## B.2 FULL THROUGHPUT RESULTS

Tables 7–9 report every measured setting behind Figure 6. Rates are tokens per second per executing H200; parentheses give each treatment’s share of the standard transformer. Table 10 reports the peak GPU memory of the same measurements. The learned baseline’s training rate is a single measurement. Placement refers to the additional n-gram storage: COUNT’s count tables are hostresident in all measurements, while the backbone’s unembedding matrix is GPU-resident for every method.

Multiplicative: pretrained

Additive: inference

COUNT  
![](images/4418acad5122ba7687b4b45d42a67e3469b5a18fb126e9cf8f2723a864ffe657.jpg)  
Additive

![](images/5fc769cd63be8fb053c40dfb63696f6835dd7ec0b7eb3ab781f00ce5f41a5c15.jpg)  
D/N = 10 D/N = 20 D/N = 40 Domain mean  
Multiplicative

![](images/a29f1a190b0935e46bca7296ddfe03260e4f5401b26225658ad89130e15c59df.jpg)

![](images/d7277a5a44744e55a4e8afe3552a38977ccbd01abbabe18356e1fa580beb1f3c.jpg)  
D/N = 10 D/N = 20 $D / N = 4 0$ Domain mean

![](images/8052a54316ab2b8290ed40efee3e8510a5a71f444f45ea2620d31e30831ef99d.jpg)

![](images/d51d71081e947cf336d12c42a4be00885d87497cbf442b372c706593f2121df3.jpg)  
D/N = 10 $D / N = 2 0$ D/N = 40 Domain mean  
Figure 10: Domain adaptation with inference-only late fusion and COUNT.  
Additive: pretrained Multiplicative: inference

![](images/749b9ff1788c71e7801198146e0828e908bdacb7f3fb54dae4e86b8a289ef131.jpg)

![](images/8b343fe9392680edbf8ef8011b7035b86a89886da7eee4536c683b991a5fb469.jpg)

![](images/32d27e27ce03456e8dee745aa7e8e62d2fd02d0d53c0992b92e36c1f2d5c6bdb.jpg)  
Figure 11: Pretraining versus inference-time late fusion at 190M.

Table 7: Training throughput at batch 32 on eight H200s by COUNT lookup (J) and table size.
<table><tr><td></td><td>J = 6 17.1 GB</td><td> $J = 1$  17.1 GB</td><td> $J = 6$  797.5 GB</td><td> $J = 1$  797.5 GB</td></tr><tr><td>Standard transformer</td><td>55,836</td><td>55,836</td><td>55,836</td><td>55,836</td></tr><tr><td>Learned n-gram</td><td>54,890 (98.3%)</td><td>54,890 (98.3%)</td><td>54,890 (98.3%)</td><td>54,890 (98.3%)</td></tr><tr><td>COUNT</td><td>52,127 (93.4%)</td><td>54,585 (97.8%)</td><td>52,008 (93.1%)</td><td>54,532 (97.7%)</td></tr></table>

Table 8: Prefill throughput on one H200 by batch size and n-gram storage placement.
<table><tr><td></td><td>Batch 1 GPU-resident</td><td>Batch 8 GPU-resident</td><td>Batch 1 Host-resident</td><td>Batch 8 Host-resident</td></tr><tr><td>Standard transformer</td><td>116,017</td><td>122,389</td><td>116,017</td><td>122,389</td></tr><tr><td>Learned n-gram</td><td>115,181 (99.3%)</td><td>122,091 (99.8%)</td><td>111,403 (96.0%)</td><td>117,402 (95.9%)</td></tr><tr><td>COUNT</td><td></td><td></td><td>111,110 (95.8%)</td><td>116,931 (95.5%)</td></tr></table>

Table 9: Decode throughput on one H200 with CUDA graphs by batch size and n-gram storage placement.
<table><tr><td></td><td>Batch 1 GPU-resident</td><td>Batch 256 GPU-resident</td><td>Batch 1 Host-resident</td><td>Batch 256 Host-resident</td></tr><tr><td>Standard transformer</td><td>340.2</td><td>19,216</td><td>340.2</td><td>19,216</td></tr><tr><td>Learned n-gram</td><td>290.8 (85.5%)</td><td>18,795 (97.8%)</td><td>319.3 (93.9%)</td><td>18,448 (96.0%)</td></tr><tr><td>COUNT</td><td></td><td></td><td>326.3 (95.9%)</td><td>18,586 (96.7%)</td></tr></table>

Table 10: Peak GPU memory in GiB per executing GPU for each throughput measurement, as allocated (reserved) high-water marks of the PyTorch allocator. Training is sharded across eight H200s; inference runs on one H200. The learned n-gram baseline’s GPU-resident table holds 2.16 GB.
<table><tr><td>Workload</td><td>Batch Standard transformer</td><td></td><td>Learned n-gram</td><td>COUNT</td></tr><tr><td colspan="5">GPU-resident n-gram storage</td></tr><tr><td>Training</td><td>32</td><td>60.8 (61.7)</td><td>63.4 (64.6)</td><td></td></tr><tr><td>Prefill</td><td>1</td><td>4.4 (6.1)</td><td>6.4 (12.2)</td><td></td></tr><tr><td>Prefill</td><td>8</td><td>14.4 (20.1)</td><td>16.4 (21.9)</td><td></td></tr><tr><td>Decode</td><td>1</td><td>4.0 (4.9)</td><td>6.0 (6.9)</td><td></td></tr><tr><td>Decode</td><td>256</td><td>47.4 (110.8)</td><td>49.4 (115.1)</td><td></td></tr><tr><td colspan="5">Host-resident n-gram storage</td></tr><tr><td>Training (J = 6, 17.1 GB)</td><td>32</td><td>60.8 (61.7)</td><td></td><td>61.0 (62.0)</td></tr><tr><td>Training (J = 1, 17.1 GB)</td><td>32</td><td>60.8 (61.7)</td><td></td><td>60.9 (62.0)</td></tr><tr><td>Training (J = 6, 797.5 GB)</td><td>32</td><td>60.8 (61.7)</td><td></td><td>61.0 (62.0)</td></tr><tr><td>Training (J = 1, 797.5 GB)</td><td>32</td><td>60.8 (61.7)</td><td></td><td>60.9 (62.0)</td></tr><tr><td>Prefill</td><td>1</td><td>4.4 (6.1)</td><td>4.4 (4.8)</td><td>4.4 (6.1)</td></tr><tr><td>Prefill</td><td>8</td><td>14.4 (20.1)</td><td>14.4 (19.5)</td><td>14.4 (18.9)</td></tr><tr><td>Decode</td><td>1</td><td>4.0 (4.9)</td><td>4.0 (4.7)</td><td>4.0 (4.9)</td></tr><tr><td>Decode</td><td>256</td><td>47.4 (110.8)</td><td>47.4 (115.1)</td><td>47.4 (105.4)</td></tr></table>

## B.3 TABLE SWAP INTERPOLATION ON LOSS

Figure 12 shows the absolute task-loss changes at all four interpolation weights, averaging subjects within MATH, LabBench and the MMLU biomedical group.

![](images/b54dc23c53a68fcb81186a659f4d94515e758326e532c586d298a6458ecae481.jpg)  
Figure 12: Full domain-table interpolation grid for 1B COUNT at $D / N = 1 0 , 2 0 , 4 0$ . Each metric is change relative to its own checkpoint’s original-table score.

## B.4 INFERENCE-TIME K AND J VARIATION

Starting from the trained 1B COUNT model, we test its sensitivity to two inference-time settings without retraining: the number of candidates kept per order, K, and the number of longest matching orders averaged, J (Figures 13 and 14). Both show some tolerance in LM loss and especially task accuracy. However, larger departures from the trained settings lose most of COUNT’s improvement over the standard transformer. A sensitivity sweep that pretrains each configuration may retain more of the improvement.

![](images/f70c78ec1c703e0c754344c99ff2d93ed01910a8e39f628afbdc50c8428fae2f.jpg)  
Figure 13: Inference-K sweep for the 1B COUNT model with its original 797.487 GB table fixed.

## B.5 TRAINING WITH FEWER N-GRAM ORDERS

Figure 15 compares 190M COUNT trained with orders one through three against orders one through six, with the same table and K. Six orders give lower LM and task loss at every budget, so we train with all six.

## B.6 ABLATING EMBEDDING AMPLIFICATION

COUNT applies embedding amplification, ${ \sqrt { d } } \ \mathrm { L a y e r N o r m } ( \cdot )$ , to its combined input. Figure 16 ablates it within 190M COUNT: amplification lowers both LM and task loss at every budget. Figure 17 adds amplification to the standard transformer itself across the ladder: it improves LM loss at the

![](images/5fbe7eb409e2ce5041c3df9c4ae452e512df0cb64a7caf5a2e92fe884f221478.jpg)  
Figure 14: Inference-J sweep for the 1B COUNT model with its original 797.487 GB table and K = 8 fixed.

n<sub>max</sub> = 3 (orders 1–3)  
![](images/cb8a202968f0990374d8f11c432553d562a1a5b39d9aa6f66162eb665982eb5b.jpg)

![](images/50c58983532f9abcf7c14e17413f5b196fd1ded3857f444c202c669fe8e1ecda.jpg)  
n<sub>max</sub> = 6 (orders 1–6)

![](images/f34ffcab81d13c712577262e23c2515d1fc6da362e89f6cc13f47086013877ce.jpg)  
Figure 15: 190M COUNT trained with $n _ { \mathrm { m a x } } = 3$ versus $n _ { \mathrm { m a x } } = 6$ maximum n-gram order with same 17.1 GB table, $K = 8 ,$ , and embedding amplification.

smallest scales, has mixed effects on downstream accuracy, and increases LM and task loss at 760M and 1B. We therefore compare against the standard transformer without amplification.

![](images/fb3a1be9bd055a1d730ccee85a42a1d803cb353af6f5b7253c91639cc1e57909.jpg)  
Figure 16: 190M COUNT with and without embedding amplification (17.1 GB table, $K \ : = \ : 8 ,$ $J \overset { ^ { \prime } } { = } 6 )$

![](images/b6c71d2cbb5e584623ff5d5e97c3ee5feec62617050f68c5b24cba3c0c24438e.jpg)  
Figure 17: Embedding amplification across standard transformer scales at $D / N = 1 0 , 2 0 , 4 0 .$

## B.7 ABLATING THE COUNT EMBEDDING BASIS

Figure 18 compares the standard unembedding basis with reusing the input token embeddings as COUNT’s basis at 1B. The unembedding basis gives lower LM and task loss and higher accuracy at all three training budgets. The token-embedding basis still lowers LM loss relative to the standard transformer, but has worse task loss and accuracy.

![](images/867ba9927eea08154b145783322529a7de5ee5ef38b3c0309f7142918d5effe9.jpg)  
Figure 18: 1B COUNT with an unembedding or token-embedding basis.