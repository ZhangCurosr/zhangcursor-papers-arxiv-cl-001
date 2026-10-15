# Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study

Arjun Boya Institute of Cognitive Science University of Osnabrück aboya@uni-osnabrueck.de

Masthanaiah Pillimetla Department of Social Sciences and Technology Technical University of Munich masthanaiahpillimetla7@gmail.com

Charan Sai Bandaru   
Institute of Cognitive Science University of Osnabrück   
cbandaru@uni-osnabrueck.de

Kavana Chidanandappa Department of Social Sciences and Technology Technical University of Munich ckavana003@gmail.com

## Abstract

The number of attention heads in a transformer is usually set by convention or grid search. We test a simple heuristic for setting it in advance: estimate the intrinsic dimension of token representations without labels, and use roughly that many heads. We estimate intrinsic dimension with a nested-dropout VAE trained on DistilBERT embeddings of WikiText-2, choosing the KL weight at the knee of the VAE’s rate–distortion curve rather than by hand. This gives k<sup>∗</sup> = 9.2 (95% CI [8.6, 9.8]). In a small 4-layer transformer trained from scratch, 10 heads and 32 heads are statistically equivalent on SST-2 and on MRPC (paired TOST over 10 seeds, ±1 point margin), with 69% fewer attention parameters for the 10-head model. Training 10 heads from the start also slightly outperforms training 32 heads and pruning to 10 (0.6 points, p = 0.008). The evidence is narrow: two classification tasks, one small architecture, no pretrained models, and no formal link between intrinsic dimension and head count. We present the result as a practical heuristic worth testing at larger scale, not as a general principle.

## 1 Introduction

Ask why a transformer has 8, 12 or 16 attention heads per layer and the honest answer is usually that a previous model did. Head count is rarely derived from the task. Meanwhile, pruning studies keep finding that many heads in trained models can be removed with little loss (Michel et al., 2019; Voita et al., 2019), which suggests the usual counts are often larger than needed. The standard response is to train a large model and prune it afterwards. We ask whether the count can instead be chosen before training, from a cheap property of the data.

The property we use is intrinsic dimension: roughly, the number of independent directions along which the representations actually vary. Section 3 explains why this might bear on head count and where the analogy breaks down. In short, we treat it as a hypothesis with a testable prediction: a model with about k<sup>∗</sup> heads should do as well as one with many more.

We test that prediction at small scale. Our contributions are:

• A label-free procedure for estimating a head count: a nested-dropout VAE on token embeddings, an elbow detector suited to the convex spectra such VAEs produce, and a rule for choosing the VAE’s KL weight from its own rate–distortion curve. Applied to DistilBERT embeddings of WikiText-2, it gives k<sup>∗</sup> = 9.2.

• A paired-seed comparison showing that 10 heads match 32 heads on SST-2 and MRPC (TOST equivalence within ±1 accuracy point) in a 4-layer transformer trained from scratch, with 69% fewer attention parameters.

• A comparison with reactive pruning: training 10 heads from the start slightly but consistently beats training 32 heads and pruning to 10.

The claims are modest. We do not derive head count from intrinsic dimension, and two small tasks with one architecture do not establish a general rule. Section 6 lists what would be needed to do so.

## 2 Related Work

Head redundancy and pruning. Michel et al. (2019) showed that most heads in trained transformers can be removed at test time with small accuracy loss, using an importance score based on the efect of masking each head. Voita et al. (2019) found that a few specialized heads do most of the work and pruned the rest during training. Both start from a fixed, large head count and reduce it. We instead choose the count before training and compare against this kind of pruning directly.

Intrinsic dimension of representations. Neural representations typically occupy far fewer dimensions than their ambient size. Ansuini et al. (2019) measured this across layers of vision networks with the TwoNN estimator (Facco et al., 2017), and Aghajanyan et al. (2021) showed that fine-tuning language models needs only a low-dimensional subspace of parameter updates. These works use intrinsic dimension to describe trained models; we try to use it to configure one.

Ordered latent variables. Nested dropout (Rippel et al., 2014) trains an autoencoder so that its latent dimensions are ordered by importance, which makes it possible to read of how many dimensions are in use from a single model. We combine it with the variational autoencoder (Kingma and Welling, 2014) and treat the KL weight, as in β-VAE (Higgins et al., 2017), as a rate–distortion knob. For locating elbows we start from the Kneedle method (Satopää et al., 2011).

## 3 Why Intrinsic Dimension Might Bear on Head Count

Each attention head projects tokens into its own small subspace (16 dimensions in our models) and computes a separate attention pattern there. More heads let a layer track more token-to-token relations in parallel, at a parameter cost that grows linearly with the head count. The question is how many of these parallel views a task can actually use.

Our working hypothesis is that this number is limited by how many independent directions of variation the token representations have. If the embeddings vary along about k independent factors, a layer with far more than k heads has little new to attend to with the extra ones, and they should end up redundant or unused. That would be consistent with how much trained models can be pruned. Under this hypothesis, an estimate $k ^ { * }$ of intrinsic dimension suggests a head count, and setting the count near $k ^ { * }$ should cost little accuracy compared with a much larger default.

This is an analogy, not a derivation, and it has obvious gaps. A head is not tied to a single latent dimension. Deeper layers see transformed representations, not the input embeddings. And nothing guarantees that $k ^ { * }$ is a tight bound rather than a loose one. We therefore treat the hypothesis only as a source of a testable prediction: about $k ^ { * }$ heads should match a much larger count. Its practical appeal is that $k ^ { * }$ can be estimated once per domain, without labels and before any classifier is trained.

## 4 Method

## 4.1 Estimating $k ^ { * }$ with a nested-dropout VAE

Data. We pass passages from the WikiText-2 training split (Merity et al., 2017) through a frozen, pretrained distilbert-base-uncased (Sanh et al., 2019) and take the final-layer hidden states, skipping lines shorter than 20 characters, truncating at 128 tokens and dropping the [CLS] and [SEP] positions. We keep the first 60,000 token vectors (768-dimensional) and standardize each dimension. These embeddings come from neither downstream task, so the estimate does not depend on SST-2 or MRPC.

Model. The VAE encoder maps each 768-dimensional vector through layers of 512 and 256 units (each with LayerNorm and GELU) to the mean and log-variance of a 64-dimensional latent code; the decoder mirrors it. During training, nested dropout (Rippel et al., 2014) samples m uniformly from $\{ 1 , \ldots , 6 4 \}$ at each step and zeroes latent dimensions m+1 to 64. Early dimensions are therefore always in use and later ones rarely, so the trained model orders its dimensions by importance. Training details are in Appendix B.

Reading of a dimension. After training, we compute the variance of each latent mean across a held-out 10% of the embeddings and sort these variances in decreasing order. The resulting spectrum drops steeply and then flattens. We take its elbow as the number of active dimensions: after normalizing both axes to [0, 1], the elbow is the point furthest below the straight line joining the first and last points. Because the spectrum is convex, this is the correct side of the line; measuring distance above it, as the concave-curve version of Kneedle does, returns an endpoint instead. Appendix A.2 shows this with synthetic spectra.

Choosing the KL weight. The elbow depends strongly on the KL weight $\beta \colon$ stronger regularization switches of more latent dimensions. Rather than picking $\beta$ by hand, we train the VAE at ten values, $\beta \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 1 , 2 , 3 , 4 , 5 , 6 , 8 \}$ , with five seeds each, and record the held-out reconstruction loss. Plotting mean loss against $\beta$ traces a rate–distortion curve, and we take $\beta ^ { * }$ at its knee, found with the same line-distance rule (both axes min–max normalized, $\beta$ on a linear scale). The estimate $k ^ { * }$ is the mean elbow across seeds at $\beta ^ { * }$ . Nothing in this procedure looks at the downstream tasks.

## 4.2 Downstream comparison: 10 heads versus 32

Tasks. SST-2 (Socher et al., 2013) sentiment classification (67,349 training and 872 validation sentences) and MRPC (Dolan and Brockett, 2005) paraphrase detection from GLUE (Wang et al., 2019) (3,668 training and 408 validation sentence pairs). Text is tokenized with the bert-base-uncased tokenizer to at most 128 tokens; MRPC pairs are encoded jointly as [CLS] sentence 1 [SEP] sentence 2 [SEP].

Model. A 4-layer transformer encoder (Vaswani et al., 2017) trained from scratch, with model dimension 128, feed-forward dimension 512 and 16 dimensions per head. Classification uses the [CLS] position. Only the number of heads k difers between configurations, so the attention projection width is 16k: 160 for $k = 1 0$ and 512 for $k = 3 2$ . The two models have 4,788,418 and 5,509,314 parameters in total, and 327,680 and 1,048,576 attention parameters, a 69% reduction in attention parameters for $k = 1 0$

Design. We train both configurations with the same 10 random seeds. For a given seed, both see identical data order (each run has its own seeded data loader) and difer only in head count. We test the paired accuracy diferences in two ways: a two-sided paired t-test for a diference, and two one-sided tests (TOST, Lakens, 2017) for equivalence within ±1 accuracy point. We fixed this margin before running the comparison. $k = 1 0$ was fixed before these experiments (Appendix A.3 gives its history) and was used unchanged on MRPC, which therefore serves as an out-of-sample check. MRPC needed diferent optimization settings from SST-2 because it is about 18 times smaller (Appendix A.4); both head counts always share the same settings.

## 4.3 Pruning baseline

To compare prospective selection with the usual alternative, for each of the same 10 seeds we train a 32-head model on SST-2, score each head by how much masking it increases validation loss (Michel et al., 2019), keep the 10 highest-scoring heads in each layer, and build a genuinely smaller model by slicing out their weights. We checked that this sliced model reproduces the masked 32-head model up to floating-point rounding (largest diference below 10<sup>−6</sup> over five random trials). The pruned model is then fine-tuned for up to 5 epochs at a learning rate of $2 \times 1 0 ^ { - 5 }$

## 5 Results

## 5.1 The VAE estimate

Figure 1 and Table 1 show the sweep. Held-out reconstruction loss rises steadily with $\beta$ while the elbow falls from 34 active dimensions at $\beta = 0 . 1$ to about 5 at $\beta = 8 .$ , so the estimate depends heavily on the KL weight and a single hand-picked value would be close to arbitrary. The knee of the loss curve is at $\beta ^ { * } = 4 .$ , an interior point (its distance from the chord peaks there at 0.222 and falls to zero at both ends). At $\beta ^ { * }$ the elbow is $k ^ { * } = 9 . 2$ , 95% CI [8.6, 9.8]. The values 3, 5 and 6 were added after a first seven-point sweep to check the knee more finely; they left $\beta ^ { * }$ and $k ^ { * }$ unchanged.

The head count we test, $k = 1 0$ , sits just above this interval, 0.8 dimensions from the point estimate. We read this as rough agreement, not a precise match.

## 5.2 10 heads versus 32

Table 2 gives the main result. On SST-2 the 10-head model averages 82.13% validation accuracy and the 32-head model 81.95%. The paired diference is not significant (t = 1.08, $p = 0 . 3 1$ , Cohen’s $d = 0 . 3 4 )$ , and TOST confirms equivalence within one point $( p = 0 . 0 0 0 5 )$ . On MRPC, with $k = 1 0$ carried over unchanged, the two models reach 70.47% and 70.37% (t = 0.34, p = 0.74, d = 0.11), again equivalent $( p = 0 . 0 0 6 )$ . So in this setting, cutting attention parameters by 69% costs no measurable accuracy on either task.

MRPC accuracy is low for this task, and both models overfit heavily (about 94% training accuracy against 70% validation), which is what one expects when training a transformer from scratch on 3,668 examples. The gap is the same for both head counts (23.70 and 23.69 points), so it does not bias the comparison, but the absolute MRPC numbers say little about pretrained models.

![](images/f4a1914c1d04b6f125125d8fa69ee0e81ec22c484567c302e6768b823d5c0d4f.jpg)

![](images/5bafba2500b7434ecc26f8c50d4fdff27278176062b057695d877f7d84ebd632.jpg)  
Figure 1: The $\beta$ sweep (mean over 5 seeds). Left: held-out reconstruction loss; $\beta ^ { * }$ is the point furthest from the dashed chord. Right: elbow of the latent variance spectrum, with 95% confidence intervals. At $\beta ^ { * } = 4$ the elbow is $k ^ { * } = 9 . 2 ;$ the dotted line marks the head count tested downstream.

Table 1: Values behind Figure 1: held-out reconstruction loss and elbow k (mean ± 95% CI half-width over 5 seeds). Italic rows were added after the first seven-point sweep.
<table><tr><td> $\beta$ </td><td>Reconstruction loss Elbow k</td></tr><tr><td>0.10</td><td>0.354  $3 4 . 0 \pm 0 . 9$ </td></tr><tr><td>0.25</td><td>0.382</td></tr><tr><td>0.50 0.422</td><td> $2 6 . 4 \pm 1 . 1$   $2 1 . 6 \pm 1 . 4$ </td></tr><tr><td>1.00</td><td>0.485  $1 7 . 0 \pm 1 . 5$ </td></tr><tr><td>2.00</td><td>0.602  $1 3 . 0 \pm 0 . 9$ </td></tr><tr><td>3.00 0.707</td><td> $1 0 . 4 \pm 0 . 7$ </td></tr><tr><td>4.00 (β*)</td><td>0.790  ${ \bf 9 . 2 \pm 0 . 6 }$ </td></tr><tr><td>5.00</td><td>0.830  $6 . 2 \pm 1 . 4$ </td></tr><tr><td>6.00 0.864</td><td> $5 . 4 \pm 1 . 9$ </td></tr><tr><td>8.00 0.894</td><td> $4 . 6 \pm 0 . 7$ </td></tr></table>

## 5.3 Comparison with pruning

Pruning the trained 32-head models to 10 heads without further training drops accuracy sharply and erratically, to $7 1 . 6 9 \% \pm 7 . 0 4$ (Table 3). Fine-tuning recovers most of this, to $8 1 . 5 0 \% \pm 0 . 3 7$ The 10-head models trained from scratch still do slightly better, by 0.63 points on average $( t = 3 . 4 1$ $p = 0 . 0 0 8 , d = 1 . 0 8 )$ . The gap is consistent across seeds but small; it also falls inside our ±1 point equivalence margin (TOST $p = 0 . 0 3 9 )$ ). The larger diference is cost. The prospective model never needs a trained 32-head model or a head-scoring pass; the full pruning pipeline took 2.41 GPU-hours for 10 seeds, about as much as all 20 runs of the SST-2 comparison (Appendix C).

The comparison is if anything tilted toward pruning, because head scores and fine-tuning both used the validation set on which accuracy is reported. We tested only one importance score, on one task.

## 6 Discussion and Limitations

What the experiments support is narrow: for this small model on these two tasks, 10 heads are enough, and choosing 10 up front is at least as good as pruning down to 10. The VAE estimate landing near 10 is encouraging but is a single agreement, not a validation of the method. Several things would be needed before treating the heuristic as reliable.

Table 2: Validation accuracy of 10 versus 32 heads (mean ± std over 10 paired seeds). TOST tests equivalence within ±1 accuracy point.
<table><tr><td>Task</td><td> $k = 1 0$ </td><td> $k = 3 2$ </td><td>Paired t-test</td><td>TOST</td></tr><tr><td>SST-2</td><td> $8 2 . 1 3 \% \pm 0 . 4 2$ </td><td> $8 1 . 9 5 \% \pm 0 . 3 3$ </td><td> $p = 0 . 3 1$ </td><td>equivalent,  $p = 0 . 0 0 0 5$ </td></tr><tr><td>MRPC</td><td> $7 0 . 4 7 \% \pm 0 . 9 0$ </td><td> $7 0 . 3 7 \% \pm 1 . 0 3$ </td><td> $p = 0 . 7 4$ </td><td>equivalent,  $p = 0 . 0 0 6$ </td></tr></table>

Table 3: Pruning baseline on SST-2 (mean ± std over the same 10 seeds).
<table><tr><td>Configuration</td><td>Validation accuracy</td></tr><tr><td>32 heads, unpruned</td><td> $8 1 . 9 5 \% \pm 0 . 3 3$ </td></tr><tr><td>32 pruned to 10, before fine-tuning</td><td> $7 1 . 6 9 \% \pm 7 . 0 4$ </td></tr><tr><td>32 pruned to 10, after fine-tuning</td><td> $8 1 . 5 0 \% \pm 0 . 3 7$ </td></tr><tr><td>10 heads from the start</td><td> $\mathbf { 8 2 . 1 3 \% } \pm \mathbf { 0 . 4 2 }$ </td></tr></table>

No formal link. Section 3 gives an intuition, not a theorem. We have no bound connecting intrinsic dimension to the number of heads a task needs, and no account of why a property of the input embeddings should constrain every layer.

Too few head counts. We compare only 10 and 32. That shows 10 is suficient, not that $k ^ { * }$ is the right place to stop. A sweep over roughly 4 to 16 heads with multiple seeds would show where accuracy starts to fall and whether that point tracks $k ^ { * }$ . The same sweep would answer an obvious objection: that a simpler rule, such as trying a few powers of two, would find an equally good count without any VAE.

One estimator. We use one intrinsic-dimension estimator. Others, such as TwoNN (Facco et al., 2017) or PCA explained variance, might give diferent numbers, and the knee rule for $\beta$ is one reasonable choice among several (cross-validated likelihood or minimum description length would be alternatives).

Small scale. All results come from a 4-layer, 128-dimensional transformer trained from scratch on two binary classification tasks. We have not tested pretrained models, larger models, generation, long contexts or tasks that need multi-step reasoning, where head requirements may be quite diferent.

Narrow pruning baseline. One ablation-based importance score on one task. Gradient-based scores (Michel et al., 2019) or pruning during training (Voita et al., 2019) might close the gap.

## 7 Conclusion

We tested whether a label-free estimate of intrinsic dimension can set the number of attention heads before training. In a small transformer, a head count near the estimate matched one more than three times larger on two tasks and slightly beat pruning to the same size. These results are a reason to try the heuristic at a larger scale, with a proper sweep over head counts and comparisons against simpler rules, not a reason to adopt it yet.

## Reproducibility

Code, the per-seed result files and the scripts behind every table, including the synthetic elbow test and the pruning-slicing check, are available at https://github.com/subcortical7/IT-MHA. Hyperparameters are listed in Appendix B.

## A Development History

Several problems surfaced while building this pipeline and changed its results. We record them here because they explain some design choices in the main text.

## A.1 Unpaired seeds

In an early version of the SST-2 comparison, both head counts drew data order from the global random number generator after model initialization. Initialization consumes a number of random draws that depends on the head count, so for the same seed the 10-head and 32-head runs saw diferent batch orders, and the paired test was not actually paired. Every run now has its own seeded data loader, and we confirmed that batch order matches within each seed pair before rerunning all comparisons.

## A.2 Elbow detection on convex spectra

The first elbow detector measured distance above the chord joining the spectrum’s endpoints, which suits concave curves. A variance spectrum from nested dropout is convex, so every interior point lies below the chord and the detector returns an endpoint (or, with noise, a point next to one), chosen by noise rather than by the curve’s shape. We tested this on synthetic spectra. On 15 noiseless convex spectra the original rule returned $k = 1$ or $k = 6 4$ every time, and on 260 noisy ones it returned $k \in \{ 1 , 2 , 6 3 , 6 4 \}$ every time. The corrected rule, which measures distance below the chord, recovered the known elbow exactly on all 180 noisy piecewise-linear spectra with elbows between $k = 4 { \mathrm { ~ a n d ~ } } k = 3 2$

## A.3 Where $k = 1 0$ came from

$k = 1 0$ was first obtained from an early version of the VAE analysis that used the faulty elbow detector at a hand-picked $\beta = 4$ . That number should not be trusted on its own, which is why we later replaced the hand-picked β with the knee rule and fixed the detector. The corrected procedure chooses $\beta ^ { * }$ without reference to the earlier result, happens to land on the same $\beta ^ { * } = 4$ , and gives $k ^ { * } = 9 . 2$ . So $k = 1 0$ was fixed before the downstream experiments, but the corrected estimate that supports it came afterwards. We regard the agreement as a modest corroboration, not an independent derivation.

## A.4 MRPC training settings

Reusing the SST-2 settings on MRPC made both models predict only the majority class (validation accuracy stuck at 68.4%). The cause was the fixed 500-step warmup, which at batch size 64 covered most of MRPC’s 580–870 training steps. A first fix with weaker regularization (dropout 0.15, weight decay 0.01) avoided the collapse but let both models overfit, and the 32-head model overfit slightly more: its train–validation gap was about 0.7 points larger. Under that setting the paired diference reached $p = 0 . 0 4$ in favor of 10 heads. We did not treat this as a real head-count efect, since it came from unequal overfitting. The final settings (Appendix B) equalize the gaps and the diference disappears. This suggests head-count comparisons on small datasets should be checked for regularization-driven asymmetries before a diference is reported.

## B Hyperparameters

Table 4: Training settings. All classifier runs use AdamW, a cosine learning-rate schedule with linear warmup, gradient clipping at 1.0, and early stopping on validation accuracy.
<table><tr><td></td><td>SST-2</td><td>MRPC</td><td>Pruned fine-tune</td><td>VAE</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.03</td><td>0.05</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Dropout</td><td>0.30</td><td>0.25</td><td>0.30</td><td></td></tr><tr><td>Batch size</td><td>64</td><td>32</td><td>64</td><td>256</td></tr><tr><td>Max epochs</td><td>10</td><td>20</td><td>5</td><td>40</td></tr><tr><td>Early-stopping patience</td><td>3</td><td>6</td><td>2</td><td></td></tr><tr><td>Warmup</td><td>500 steps</td><td>6% of steps</td><td>5% of steps</td><td></td></tr><tr><td>KL annealing Seeds</td><td>10</td><td>10</td><td>10</td><td> $0  \beta$  over 20 epochs 5 per β</td></tr></table>

Classifier seeds are 42, 123, 456, 789, 1011, 2024, 31337, 7, 999 and 8675309; VAE seeds are 42, 123, 7, 2024 and 99. The VAE is trained on a random 90% of the embeddings and evaluated on the remaining 10%.

## C Compute

All experiments ran on a single NVIDIA GPU workstation. Recorded training time was about 5.2 GPU-hours: 2.55 h for the 20 SST-2 runs, 0.24 h for the 20 MRPC runs and 2.41 h for the pruning baseline (training the 32-head models, scoring heads and fine-tuning). The VAE sweep was not timed.

## References

Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the efectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328, Online, 2021. Association for Computational Linguistics.

Alessio Ansuini, Alessandro Laio, Jakob H. Macke, and Davide Zoccolan. Intrinsic dimension of data representations in deep neural networks. In Advances in Neural Information Processing Systems, volume 32, 2019.

William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005.

Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio. Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports, 7:12140, 2017.

Irina Higgins, Loïc Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. β-VAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, 2017.

Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations, 2014.

Daniël Lakens. Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. Social Psychological and Personality Science, 8(4):355–362, 2017.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017.

Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In Advances in Neural Information Processing Systems, volume 32, 2019.

Oren Rippel, Michael A. Gelbart, and Ryan P. Adams. Learning ordered representations with nested dropout. In Proceedings of the 31st International Conference on Machine Learning, pages 1746–1754, 2014.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.

Ville Satopää, Jeannie Albrecht, David Irwin, and Barath Raghavan. Finding a “kneedle” in a haystack: Detecting knee points in system behavior. In 31st International Conference on Distributed Computing Systems Workshops, pages 166–171, 2011.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA, 2013. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5797–5808, Florence, Italy, 2019. Association for Computational Linguistics.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019.