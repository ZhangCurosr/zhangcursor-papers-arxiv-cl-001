# HOW DEPTH GROWTH AND LOOPING SHAPE IN-CONTEXT LEARNING IN LANGUAGE MODELS

Ferdinand Kapl<sup>1,2,3∗</sup> Kaitlin Maile<sup>4†</sup> Stefan Bauer<sup>1,2,3†</sup> Johannes von Oswald<sup>4†</sup>

<sup>1</sup> Technical University of Munich <sup>2</sup> Helmholtz AI, Munich

<sup>3</sup> Munich Center for Machine Learning (MCML) <sup>4</sup> Google, Paradigms of Intelligence Team

## ABSTRACT

Looped and depth-grown language models reuse computations across depth and have attracted interest for their reasoning capabilities. Building on prior suggestive evidence of improved in-context-learning (ICL), we investigate how such reuse favors ICL, including how it shapes the acquisition, retention, and mechanisms of this ability. Our analysis spans pretrained language models up to 3B parameters and controlled tasks. The experiments provide evidence for a shared inductive bias toward learning from context: grown and looped models rely more on information in context and, compared with a full-depth dense reference, acquire ICL earlier in controlled tasks and retain it more strongly when learning fixed associations in the weights is also possible. Tracing attention heads through training reveals that depth growth propagates early induction heads across layers. Ablations show that grown and looped models rely more strongly on inductionranked heads for both few-shot performance and later-token prediction. Together, these findings provide evidence for a shared inductive bias toward learning from context and suggest that (soft) parameter reuse shapes how this ability develops and which attention mechanisms support it.

## 1 INTRODUCTION

How a language model acquires its depth can shape the computations it learns. Depth growing trains an initially shallow network and progressively duplicates blocks of layers, allowing later training stages to reuse computations learned earlier. Looping instead repeatedly applies a shared block within each forward pass. Originally motivated by training efficiency (Fahlman & Lebiere, 1989) and parameter efficiency, respectively, both approaches have attracted renewed interest for their reasoning capabilities (Saunshi et al., 2024; 2025; Geiping et al., 2025). Although grown layers become independent after duplication, whereas looped layers remain tied, the two approaches can produce related patterns of iterative computation across depth (Kapl et al., 2026a;b).

This reuse of computation may also favor in-context learning (ICL), as suggested by initial evidence in both grown and looped models (Kapl et al., 2026b). ICL is the ability to use information supplied in a prompt or sequence without updating the model’s parameters (Brown et al., 2020). A model that repeatedly refines its representations might also be better equipped to extract and apply patterns from context. Yet, improved reasoning performance alone does not establish this ability. Predictions can rely on associations stored in the weights, on information inferred from the current sequence, or on both. Moreover, a model’s improvement in predicting later tokens and its ability to learn a task from demonstrations are related but distinct measurements, with potentially different underlying mechanisms (Olsson et al., 2022; Yin & Steinhardt, 2025).

Prior work provides promising evidence for a connection between depth reuse and ICL. Looped Transformers can learn synthetic regression and other data-fitting tasks with substantially fewer parameters than standard Transformers (Yang et al., 2024; Von Oswald et al., 2023). Studies of pretrained language models also report improved adaptation from demonstrations in grown and looped models for synthetic reasoning tasks (Kapl et al., 2026b). Concurrently, Wang et al. (2026) report stronger gains with increasing context length and demonstration count in compute-matched looped mixture-of-experts models. What remains less understood is how these behavioral differences arise during training: whether growing and looping favor a context-based solution when memorization is also possible, how they affect the persistence of that solution, and how changes in attention-head mechanisms relate to the different measures of ICL.

In this work, we investigate these questions through complementary experiments on natural language and controlled tasks. We compare dense baselines with depth-grown and looped models on language-model configurations with dense reference sizes from 360M to 3B parameters. We measure position-dependent token loss and scaling with the number of demonstrations for few-shot learning. We then use RegBench (Akyürek et al., 2024) to study learning previously unseen formal languages, and Omniglot-based tasks (Chan et al., 2022; Singh et al., 2023) to separate learning from context from learning fixed associations in the weights. Finally, we track induction and function vector heads through training and intervene on them in the final models. Our contributions are the following:

• Grown and looped language models benefit more from context. In the main 3B language-model comparison, both achieve larger early-to-late token-loss reductions than the dense baseline, with the grown model also attaining lower absolute late-token loss. Both models also achieve higher GSM8K accuracy across all demonstration counts. On RegBench, grown and looped models predict unseen formal languages more accurately, with their advantage emerging within the first string (Figures 1 and 2).

• Growing and looping favor earlier acquisition and stronger retention of ICL. In controlled Omniglot tasks where both context and fixed class–label associations can solve training queries, grown and looped models retain higher ICL accuracy and exhibit lower in-weights learning (IWL) accuracy than the full-depth dense reference. When labels are reassigned in every sequence, removing a fixed class–label solution, both acquire ICL earlier than the same reference (Figure 3).

• Depth growth propagates early induction heads across layers. Tracing final induction heads backward through training shows that strong heads in the grown model predominantly descend from heads already exhibiting strong induction behavior in early growing stages. Layer duplication produces repeated patterns of this behavior across depth, while looped models exhibit analogous patterns across successive applications of their shared looped block (Figures 4 and 5).

• Grown and looped models rely more strongly on induction heads for ICL. In the main language-model comparison, ablating induction-ranked heads causes larger drops in fewshot performance and larger increases in late-token loss than in the dense baseline. These interventions establish a greater contribution from induction-ranked heads to both behaviors, while function-vector heads remain important (Figure 6).

Together, these experiments show that constructing depth through (soft) parameter reuse shapes both the development of context use and the attention mechanisms supporting it.

## 2 RELATED WORK

Grown Models. Model growth reduces training cost by starting with a smaller network (Fahlman & Lebiere, 1989). Language-model approaches use layer copying and progressive stacking (Gong et al., 2019; Reddi et al., 2023; Kim et al., 2024; Du et al., 2024), learned transformations between model sizes (Wang et al., 2023), or masked and function-preserving expansions (Yao et al., 2024; Chen et al., 2015; Gesmundo & Maile, 2023). Growth operators also shape model capabilities: MIDAS duplicates a central block of layers and improves reasoning at comparable language-model capabilities (Saunshi et al., 2024). Kapl et al. (2026a) find stronger reliance on later layers in grown models and propose LIDAS, which duplicates layers around the network’s center independently of existing block boundaries.

Looped Models. Cross-layer parameter sharing provides computational depth with fewer weights, as in Universal Transformers (Dehghani et al., 2019), recurrent stacking for translation (Dabre & Fujita, 2019), and ALBERT (Lan et al., 2020), while Takase & Kiyono (2023) study its design choices.

Extensions add layer-specific low-rank adapters (Bae et al., 2025) or mixture-of-experts capacity (Csordás et al., 2024). Recurrence has also been studied for algorithmic length generalization (Fan et al., 2025), latent reasoning (Saunshi et al., 2025), and inference-time computation scaling (Geiping et al., 2025; Zhu et al., 2025). Other work retrofits recurrence into pretrained models (McLeish et al., 2025; Koishekenov et al., 2025) or explores hierarchical and compact recursive architectures for iterative refinement (Wang et al., 2025; Jolicoeur-Martineau, 2025).

In-Context Learning. ICL enables adaptation from demonstrations without parameter updates (Brown et al., 2020). Studies show that Transformers infer unseen functions from examples (Garg et al., 2022), learn gradient-descent-like computations in restricted regression settings (Von Oswald et al., 2023; Gatmiry et al., 2024), and can be programmed to execute learning algorithms in context (Giannou et al., 2023). Data properties such as burstiness and dynamic meanings encourage ICL (Chan et al., 2022), which can decline as persistent associations are learned in the weights (Singh et al., 2023). Other work connects regular-language learning to in-context n-gram heads on Reg-Bench (Akyürek et al., 2024) and studies context-dependent representations and their competition with pretrained semantics through graph tracing (Park et al., 2025). These settings let us probe context use with and without competing in-weights solutions.

Induction heads support pattern copying and are associated with improved later-token prediction (Elhage et al., 2021; Olsson et al., 2022). Their formation has been studied in controlled synthetic tasks (Singh et al., 2024) and language-model pretraining (Aoyama et al., 2026). Function-vector heads transmit task representations from demonstrations (Todd et al., 2024). Yin & Steinhardt (2025) find that few-shot performance depends primarily on these heads, many of which first exhibit induction behavior.

Looped Transformers can match standard Transformers on synthetic data-fitting tasks with substantially fewer parameters (Yang et al., 2024). Concurrent work (Wang et al., 2026) reports gains that increase with sequence length and demonstration count in looped mixture-of-experts language models under matched compute, parameter, and KV-cache budgets .

Most closely related, Kapl et al. (2026b) identify shared depth-wise computational patterns in grown and looped models, show that grown models can benefit from inference-time repetition, and report improved adaptation from demonstrations on reasoning primitives. We extend this evidence to ICL acquisition, retention, and mechanisms across pretrained language models and controlled tasks. We test whether growing and looping favor context use when memorization is possible, trace induction heads through layer duplication, and ablate induction and function-vector heads to assess their contributions to few-shot learning and token prediction.

## 3 IN-CONTEXT LEARNING OF DEPTH-GROWN AND LOOPED MODELS

We first examine how growing and looping affect learning from context. After introducing the models and measures (Section 3.1), we evaluate context use in language models (Section 3.2), adaptation to unseen formal languages on RegBench (Section 3.3), and ICL with and without an in-weights solution on Omniglot (Section 3.4).

## 3.1 MODELS AND EVALUATION MEASURES

A dense model trains L independently parameterized Transformer layers from scratch. A depthgrown model instead begins with a shallower model and periodically duplicates a block of layers, after which the copies train independently. The main paper focuses on LIDAS (Kapl et al., 2026a), which selects the central layers without respecting earlier block boundaries. Comparisons with MIDAS (Saunshi et al., 2024) and at other model sizes (360M and 1.7B) appear in Appendix A.3. The language models start with block size $b = ~ 4$ layers and add b layers at each growth event, copying both weights and optimizer moments while retaining the training schedule.

A looped model reuses the same block of layers several times within each forward pass, with parameters remaining tied throughout training. We write Loop $\scriptstyle p - b \times r - c$ for p prelude layers, a block of b layers applied r times, and c coda layers, borrowing the naming conventions by Geiping et al. (2025). Its effective depth is $p + b r + c , $ with $p + b + c$ unique layers. Our main 3B comparison uses a 36-layer dense reference (Bakouch et al., 2025), a LIDAS-grown model reaching that depth, and the Loop 10 - 8×2 - 10 model, which has 36 layer applications and 28 unique layers. All three models are trained on the same 100B tokens in the same order. Within each size, all models share the data and optimizer-step budget and have matched final inference FLOPs. Growing uses shallower networks during part of training, saving an estimated 26% of training FLOPs at 3B and 22% at both 360M and 1.7B. Looping reduces unique parameters while retaining the same effective depth. Appendix A reports the configurations and training schedules.

![](images/7d0eb3a19c8c24461cf6d50f4fc91dc002f0788123d210d6f9ada808b8645754.jpg)

![](images/a81e3559b98914b07e72a69977da6a022d18e570b2c8227d7dabfa7bc6b8dca4.jpg)

![](images/7d103de067ab44b82e5cbdd83a0a2a80d3c1aadc9dacc797c668e8a28f0e3d45.jpg)  
Figure 1: Grown and looped models benefit more from context. In the 3B comparison, grown and looped models show greater context gains in (a) NLL relative to the dense baseline and (b) early-to-late loss reduction relative to the baseline per token position, although the looped model retains higher NLL. Both panels aggregate token losses in the same power-of-two position bins: (a) compares the binned loss directly with the baseline, whereas (b) first computes each model’s improvement from a fixed early token position (50) to the bin ending at x and then subtracts the baseline’s improvement. Dashed horizontal lines show mean ∆NLL over all positions. (c) Both models exceed the baseline’s few-shot GSM8K (Cobbe et al., 2021) accuracy, with larger gains for LIDAS. Results use 10,000 FineWeb-Edu documents and three demonstration seeds, respectively. Additional model sizes and exact experimental setup in Appendix B.

In-Context Learning Measures. We evaluate language models using next-token negative loglikelihood (NLL, lower is better), a token-loss ICL-Score (Olsson et al., 2022), and few-shot task performance. The ICL-Score measures the reduction in mean NLL from early positions 40–60 ("50") to later positions 490–510 ("500"), with higher values indicating greater early-to-late improvement. We use these token-loss windows to smooth potential single token position spikes and refer to the positions in the following by a single position for simplicity. In Figure 1(b), we instead average later losses over power-of-two bins ending at x, keeping the early reference fixed. This measures improvement within each model, while we report individual token loss separately. Few-shot performance is measured through benchmark accuracy or token-overlap F1. Full protocols are given in Appendix B.

## 3.2 CONTEXT USE IN LANGUAGE MODELS

We first compare how the different models use additional context in sequence modeling and few-shot tasks. These evaluations distinguish improvement with more context from absolute performance relative to the dense baseline. For sequence modeling, the depth-grown and looped models are worse at predicting early tokens compared to the baseline (Figure 1(a)), with the biggest gap for very early tokens (token index ≤ 8). After this peak, both models improve the prediction of tokens by lowering their NLL more, making better use of more tokens in the context. This is especially apparent when measuring the improvement of later tokens versus a fixed early position (50) and comparing it to the improvement of the baseline (Figure 1(b)). The looped model, which uses significantly fewer unique parameters, retains higher NLL than the baseline even at the end of the sequence. LIDAS matches the baseline around position 1000 and achieves lower NLL thereafter (solid line) and on average across positions (dashed line).

![](images/7b0644631fec4427ef502f693fcaba82b6102e312f66f5474cec2897e61ff06a.jpg)

![](images/29e9b8b35c3a089659f395658f509aa89db52de3ab4aab411bf9d84cb4a7eabc.jpg)

![](images/f6315d45ec6dbbac75fa85e8e10303e5517564f147be8c6bc972ffd182909a52.jpg)  
Figure 2: Grown and looped models predict unseen formal languages more accurately. Incontext language learning on RegBench (Akyürek et al., 2024). (a) Test total variation distance (TVD) for the dense 12-layer, Loop 2×6, and Grown b = 3 model (mean ± standard error over four training seeds). (b) TVD relative to the dense model against in-context string index. (c) The same comparison by symbol index, separately for the first string and strings 2–10. Lower is better for all panels. Bands show standard errors. The looped model has width 256, versus 128 for dense and grown models: hyperparameters are selected on validation data, rather than matched in width. Panel (a) uses the full test set. Panels (b,c) retain only string indices present in every test sequence (strings 1–10). Experimental setup and full details in Appendix C.

On GSM8K (Cobbe et al., 2021) few-shot accuracy in Figure 1(c), the looped model, again using fewer parameters, keeps its advantage over the baseline across demonstration counts, but the base line almost closes the gap at eight shots. The depth-grown model instead widens its performance improvement gap with additional examples, ending up around +7% above the baseline. GSM8K evaluates final answers extracted from generated solutions to new questions, so these gains reflect improved answer generation from in-context demonstrations, a setting relevant to practical few-shot prompting. As an additional comparison, Appendix Figure 9 reports separate averages over five multiple-choice and three generation tasks. At 3B, LIDAS improves on the baseline in both subsets, while the looped model’s advantage is concentrated in generation.

## 3.3 IN-CONTEXT LEARNING LANGUAGES: REGBENCH DATASET

RegBench (Akyürek et al., 2024) tests adaptation to an unseen formal language. Each input con catenates strings sampled from one deterministic finite automaton (DFA), with held-out automata used for testing. We measure total variation distance (TVD) between the predicted next-symbol distribution and the uniform distribution over legal continuations. All models apply twelve layers per forward pass, with two unique layers for the looped model.

In Figure 2(a), the grown and looped models achieve lower TVD than the dense baseline on novel languages. Their advantage increases over the first few strings, peaking around the fourth (Figure 2(b)). When zooming in on the test sequences in Figure 2(c), both alternative training variants adapt faster to the new language by predicting symbols (akin to characters or tokens) within the first string better after roughly 15 symbols. Additionally, on average, after the first string the grown and looped model do not continue to increase their advantage with more symbols but keep a constant improvement. Appendix Figure 12 extends the comparison including DFA accuracy and additional grown models also evaluated with four training seeds, with the precise setup in Appendix C.

![](images/f85f9759416d57cb680309e8c9e47b171d2f2dc1e4f267f9ca254ae6bc22b7b5.jpg)  
Figure 3: Growing and looping favor stronger retention of ICL, at the cost of in-weights learning (IWL), and earlier acquisition of ICL when IWL is not possible. Solution preference and emergence of in-context learning on Omniglot-based tasks (Singh et al., 2023). (a) Training uses fixed class labels and $P ( \mathrm { b u r s t y } ) = 1$ , so both context and in-weights associations can solve each training query. ICL is measured on held-out classes with labels reassigned per sequence. IWL is measured on training classes whose query class is absent from context. (b) Training follows the ICL probe setup using images from training classes, with labels reassigned in every sequence to remove a fixed class–label solution. Curves show mean accuracy, with bands indicating standard error across four (a) and eight (b) smoothed seed trajectories. Growth occurs at 50k and 100k steps. Experimental details in Appendix D.

## 3.4 IN-CONTEXT AND IN-WEIGHTS LEARNING: OMNIGLOT

We next examine learning from context when fixed associations can also be stored in the weights, or when labels vary between sequences, removing the in-weights learning (IWL) solution. For that purpose, we use the Omniglot image-classification setup of Chan et al. (2022). Each sequence contains eight image–label pairs followed by a query image. In the fixed-label setting, every training class has a persistent label and the query class always appears among the context examples, with $P ( \mathrm { b u r s t y } ) \stackrel { \textstyle = } { = } 1$ , meaning that every training sequence contains multiple demonstrations of the same classes. The training query can therefore be answered either by matching it to the current examples in context or by learning its class–label association in the weights. We distinguish these capabilities with two probes: ICL accuracy on held-out classes with labels reassigned in each sequence, and IWL accuracy on training classes whose query class is absent from context. We compare dense models with 4 and 12 layers, a Loop 4×3 model, and a LIDAS-grown model that expands from 4 to 8 layers at 50k steps and to 12 layers at 100k steps. All models have the same hidden width and use the same optimizer and batch size. We average over four training seeds for the fixed-label setting and eight for the ICL-only setting below. Appendix D gives the full setup.

Figure 3(a) shows an early rise in ICL followed by a gradual decline as IWL improves, reflecting the transience of ICL observed by Singh et al. (2023). Over the shared training horizon, the grown and looped models retain higher mean ICL accuracy but lower IWL accuracy than the dense 12- layer model. The grown model has similar ICL and IWL accuracies to its four-layer reference, from which it first grows at 50k steps. The looped model attains similar ICL accuracy to the grown model but significantly lower IWL accuracy. These trajectories are consistent with growing and looping favoring learning from context relative to the full-depth dense model at the same effective depth. For looped models, this can come at the cost of weaker in-weights learning, while the grown model appears to inherit both capabilities from its shallow starting model.

To test acquisition when a fixed class–label solution is unavailable (ICL-only), we train a different set of models with labels 0 and 1 reassigned independently in every sequence, mirroring the ICL probing setup on training images. Each sequence contains two classes with four examples each, followed by a query from one of them. In Figure 3(b), ICL accuracy emerges earlier for the looped, grown, and four-layer dense models than for the dense 12-layer model. By 1M steps, all four tested configurations almost reach 100%. Thus, this setting does not differentiate how much ICL is attained but rather the timing and consistency of ICL emergence, given that ICL is the only correct solution to the training problem.

## 4 GROWING AND LOOPING INCREASE RELIANCE ON INDUCTION HEADS

The behavioral results suggest that growing and looping favor learning from context. Returning to language models, we examine the attention mechanisms supporting this behavior. We first track induction behavior during training and across layers (Section 4.1), then ablate heads to test their contribution to context use (Section 4.2).

## 4.1 DEPTH GROWTH PROPAGATES EARLY INDUCTION HEADS

Attention-Head Measures. We characterize attention heads using prefix-matching (PS) and function-vector (FV) scores. PS measures attention to the token following an earlier occurrence in repeated random-token sequences (Olsson et al., 2022). FV scores measure how much replacing a head’s output with its clean task mean restores correct-answer prediction when demonstration labels are shuffled (Todd et al., 2024). We use these scores to select heads for ablation. Full evaluation protocols are given in Appendix E.1. We first compare induction behavior during languagemodel pretraining by averaging PS over the top 2% of heads at each checkpoint, reselecting them as training progresses. Checkpoint and aggregation details are given in Appendices A.3 and E.1.

![](images/6b91c6af7d13198a73632354221bd6e1d04ff611412431a3e257cf716f0c0591.jpg)

In Figure 4, LIDAS develops higher PS than the dense baseline and finishes training with a higher score, while the looped model finishes close to the baseline. Early growth events temporarily disrupt prefix matching, followed by quick recovery during further training. These aggregate scores motivate tracing the heads responsible for the grown model’s strong induction behavior. Full trajectories of PS, function-vector scores, few-shot performance, and the token-loss ICL-Score appear in Appendix Figure 13.

Figure 4: Induction behavior during training. Top-2% mean PS (3B), with heads reselected at each checkpoint. Vertical lines mark growth. Hollow circles show immediate postgrowth scores.

We next follow the twelve final top-2% PS heads in each 3B model through training (Figure 5, top). These identities remain fixed, unlike the checkpoint-wise selection for the average in Figure 4. For LIDAS, we trace the heads backward through layer duplication to the starting four-layer model. Copies share an ancestral trajectory until duplication, so we draw that shared path once.

The twelve final LIDAS heads trace back to just four ancestors. Two of these, L2H9 and L2H10 (zero-based indices), already have high PS before the first growth event and account for ten of the twelve final heads, highlighted in orange. The remaining two originate from initially low-scoring heads whose induction behavior develops later. Thus, most of the strongest final induction heads arise from copying and further training early induction heads. The original L2H10 retains high PS but is outside the final top-2% heads, while three of its copies are included.

The final locations in Figure 5 (bottom) connect these lineages to repeated patterns across depth. In LIDAS, strong heads recur at the same head index in layers separated by multiples of the four-layer growth block, extending from early layers $( L < 8 )$ into later layers. In Loop 10 - 8×2 - 10, high-PS heads likewise recur across the two applications of the eight-layer shared block, highlighted in orange. These are separate applications of tied heads, whereas the grown model’s copies train independently. The dense baseline shows no comparable pattern of repetition by block size.

FV heads likewise recur at the grown or looped block size, but emerge later than induction heads (Appendix Figure 14). In LIDAS, two ancestors give rise to six of the twelve final top-2% FV heads.

![](images/d6f8da41f5d643f5ac4dc677cb18dd9907764e222b649a0743712bac0bce72d8.jpg)  
Figure 5: Depth growth propagates early strong induction heads across layers. Development and final location of individual induction heads. Top: smoothed prefix-matching scores (PS) of each 3B model’s twelve final top-2% heads over training (logarithmic step axis). LIDAS lineages follow layer duplication, with shared ancestral paths drawn once. In LIDAS, the twelve final heads trace back to four ancestors in the initial four-layer model. Two of these, L2H9 and L2H10, already have high PS before growth and account for ten of the twelve final heads. Bottom: final PS for every layer and head (zero-based indices). Among the final top-2% PS heads, orange highlights descendants of the early L2H9/L2H10 heads in LIDAS and heads within the repeated block in Loop 10 - 8×2 - 10. Other heads are blue. Black outlines identify final top-2% heads, except the original L2H9 in LIDAS, which shares a red outline with L2H10. Vertical lines mark growth events (top), four-layer intervals in LIDAS and loop-block boundaries (bottom). Selection, plotting, and evaluation details in Appendix E.2. FV heads show analogous repetition but emerge later (Appendix Figure 14).

## 4.2 INDUCTION HEADS CONTRIBUTE MORE TO CONTEXT USE

The trajectories and locations reveal how induction behavior develops and repeats across depth. A high prefix-matching score alone does not establish a head’s importance for ICL. We therefore ablate heads in the final models to test their contributions to few-shot performance and token loss.

Following the mean-ablation protocol of Yin & Steinhardt (2025), we rank final-checkpoint heads by PS or FV score and replace the outputs of the selected 1–20% with mean activations. For looped models, each head application is counted and ablated separately. Here we focus on exclusive induction rankings, which first exclude the top 2% of FV heads, and exclusive FV rankings, which first exclude the top 2% of induction heads. We evaluate changes from each model’s unablated few-shot ICL (F1) and ICL-Score. Appendix F specifies the replacement means, indexing, and additional random controls.

Consistent with observations by Yin & Steinhardt (2025), all three models in Figure 6(a,b) show greater few-shot ICL degradation from FV-head ablation and greater ICL-Score (token-loss difference) reduction from induction-head ablation. At larger ablation fractions, exclusive induction-head ablation reduces few-shot performance and ICL-Score more in grown and looped models than in the baseline, indicating greater reliance on these heads. FV heads remain important in all models, but their ablation effects relative to the baseline vary across models and sizes for this comparison (Appendix Figure 20).

Figure 6(c) shows the NLL increase by token position at 9% ablation. Beyond the first few tokens, FV-head ablation hurts early-token prediction more than induction-head ablation. The FV-ablation effect then largely plateaus, while the induction-ablation effect grows with position. Compared with the baseline, LIDAS and Loop 10 - 8×2 - 10 show smaller FV-ablation effects across most positions and larger induction-ablation effects at later positions. Within both models, induction heads eventually become more important than FV heads for later-token prediction, whereas FV heads remain more important in the baseline. Together, these results indicate that both head types support context use, with greater reliance on induction heads in grown and looped models.

![](images/bc0678759a72124f0a153f68f7a8366b75eceada58f6e7b5efce0e4944b039fd.jpg)

![](images/bc064cd72f01bac1f049603ddf6dcc34a20f02568c0131eabc8e9b4d49eb3d27.jpg)

![](images/ecb40c230bebc52ea5663a667b581badbfc5a1ef871c4adc341c474d82269c3d.jpg)  
Figure 6: Grown and looped models rely more on induction heads than the baseline for few-shot ICL and later-token prediction. Mean ablation in the 3B baseline, LIDAS, and Loop 10-8×2-10. Solid lines ablate heads ranked by PS after excluding the top-2% FV heads. Similarly, dashed lines show FV-head ablations. (a) Change in few-shot ICL (F1) on 19 tasks. (b) Change in ICL-Score on FineWeb-Edu. Both changes are ablated minus the same model’s unablated score (lower is worse). At larger ablation fractions, induction-head ablation reduces both scores more in grown and looped models than in the baseline, although FV-head ablation hurts fewshot ICL more in all three models. (c) Position-wise NLL increase from each model’s unablated reference at 9% ablation over the 4096-token context (higher is worse). Curves are log-binned and smoothed. FV-ablation effects largely plateau, while induction-ablation effects grow with position and eventually exceed FV-ablation effects in LIDAS and Loop 10-8×2-10. Additional ablation fractions appear in Appendix Figure 21 with experimental details in Appendix F.

## 5 CONCLUSION

Our experiments provide evidence that depth growing and looping favor learning from context. In the main language-model comparison, both achieve larger early-to-late token-loss reductions and higher GSM8K accuracy than the dense baseline. Grown and looped models also predict unseen formal languages more accurately. In controlled classification tasks, both acquire ICL earlier than a full-depth dense reference when labels are reassigned between sequences, and retain higher ICL accuracy when fixed associations can also be learned in the weights. Mechanistically, depth growth propagates early induction heads through layer duplication, with most of the strongest final heads descending from a small set of early ancestors. This produces repeated patterns across depth analogous to those in looped models, even though grown layers train independently after copying. Ablations in the main language-model comparison show that grown and looped models rely more on induction-ranked heads for few-shot performance and later-token prediction. Function-vector heads remain more important for few-shot performance in all models, suggesting that depth reuse shifts the relative contributions of these mechanisms. Together, these findings suggest that reusing computations across depth shapes how models acquire and retain ICL, as well as the attention mechanisms supporting it.

The controlled tasks suggest that depth reuse shifts the competition between in-context and inweights learning: grown and looped models retain higher ICL accuracy while attaining lower IWL accuracy than the full-depth dense reference. One possible explanation is that copying early contextdependent computations through growth, or repeatedly applying them through shared layers, makes a context-based solution less susceptible to being displaced as fixed associations are learned in the weights.

## ACKNOWLEDGMENTS

The authors would like to thank Nino Scherrer for insightful discussions and support throughout this work. This work was partially supported by the Helmholtz Foundation Model Initiative and the Helmholtz Association. The authors thank the Gauss Centre for Supercomputing e.V. (GCS) for providing computing time on the Supercomputer JUPITER at the Jülich Supercomputing Centre via the John von Neumann Institute for Computing. JUPITER is supported by the EuroHPC JU and GCS through funding by the European Commission, the German Federal Ministry of Research, Technology and Space, and the Ministry of Culture and Science of the State of North Rhine-Westphalia. Furthermore, the authors appreciate the computational resources provided by the National High Performance Computing Centre (www.nhr.kit.edu). The research presented is supported by the TUM Georg Nemetschek Institute Artificial Intelligence for the Built World.

## AI USE STATEMENT

Generative AI tools, including LLM-based coding agents, supported the implementation of our methods and experiments. We further used them to aid the writing of plotting code behind our figures and to polish the wording of the manuscript. All AI-assisted code, figures, and text were checked by the authors, and the research questions, experimental design, and conclusions are our own. We take full responsibility for the content of this work, including everything produced with the help of generative AI.

## REPRODUCIBILITY STATEMENT

The information needed to reproduce our experiments is provided in the main text and appendix. Language-model architectures, training configurations, and growth schedules are described in Ap pendix A. Evaluation protocols and mechanistic analyses are detailed in Appendices B, E and F, and the RegBench and Omniglot setups, hyperparameters, and data generation procedures in Appendices C and D. We use publicly available datasets and build on public implementations, including the nanotron training framework (https://github.com/huggingface/nanotron).

## REFERENCES

Ekin Akyürek, Bailin Wang, Yoon Kim, and Jacob Andreas. In-context language learning: Architectures and algorithms. In International Conference on Machine Learning, 2024.

Tatsuya Aoyama, Ethan Gotlieb Wilcox, and Nathan Schneider. Predicting the emergence of induction heads in language model pretraining. In International Conference on Machine Learning, 2026.

Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. Relaxed recursive transformers: Effective parameter sharing with layer-wise loRA. In The Thirteenth International Conference on Learning Representations, 2025.

Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Patiño, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, Quentin Gallouédec, et al. Smollm3: smol, multilingual, long-context reasoner, 2025. URL https://huggingface. co/blog/smollm3.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. SmolLM-corpus, July 2024. URL https://huggingface.co/datasets/ HuggingFaceTB/smollm-corpus.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler,

Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.

Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. In Advances in Neural Information Processing Systems, 2022.

Tianqi Chen, Ian Goodfellow, and Jonathon Shlens. Net2net: Accelerating learning via knowledge transfer. arXiv preprint arXiv:1511.05641, 2015.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Róbert Csordás, Kazuki Irie, Jürgen Schmidhuber, Christopher Potts, and Christopher D Manning. MOEUT: Mixture-of-experts universal transformers. Advances in Neural Information Processing Systems, 2024.

Raj Dabre and Atsushi Fujita. Recurrent stacking of layers for compact neural machine translation models. In AAAI Conference on Artificial Intelligence, 2019.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019.

Wenyu Du, Tongxu Luo, Zihan Qiu, Zeyu Huang, Yikang Shen, Reynold Cheng, Yike Guo, and Jie Fu. Stacking your transformers: A closer look at model growth for efficient llm pre-training. In Advances in Neural Information Processing Systems, 2024.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits, 2021. URL https: //transformer-circuits.pub/2021/framework/index.html.

Scott Fahlman and Christian Lebiere. The cascade-correlation learning architecture. In Advances in Neural Information Processing Systems, 1989.

Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee. Looped transformers for length generalization. In International Conference on Learning Representations, 2025.

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn incontext? a case study of simple function classes. In Advances in Neural Information Processing Systems, 2022.

Khashayar Gatmiry, Nikunj Saunshi, Sashank J. Reddi, Stefanie Jegelka, and Sanjiv Kumar. Can looped transformers learn to implement multi-step gradient descent for in-context learning? In International Conference on Machine Learning, 2024.

Jonas Geiping, Sean Michael McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems, 2025.

Andrea Gesmundo and Kaitlin Maile. Composable function-preserving expansions for transformer architectures. arXiv preprint arXiv:2308.06103, 2023.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In International Conference on Machine Learning, 2023.

Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training of BERT by progressively stacking. In International conference on machine learning, 2019.

Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, and Hannaneh Hajishirzi. Olmes: A standard for language model evaluations. In Findings of the Association for Computational Linguistics: NAACL 2025, 2025.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks. arXiv preprint arXiv:2510.04871, 2025.

Ferdinand Kapl, Emmanouil Angelis, Tobias Höppe, Kaitlin Maile, Johannes von Oswald, Nino Scherrer, and Stefan Bauer. Do depth-grown models overcome the curse of depth? an in-depth analysis. In Conference on Language Modeling, 2026a.

Ferdinand Kapl, Emmanouil Angelis, Kaitlin Maile, Johannes von Oswald, and Stefan Bauer. From growing to looping: A unified view of iterative computation in LLMs. In International Conference on Machine Learning, 2026b.

Sanghoon Kim, Dahyun Kim, Chanjun Park, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, et al. Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pp. 23–35, 2024.

Yeskendir Koishekenov, Aldo Lipani, and Nicola Cancedda. Encode, think, decode: Scaling testtime reasoning with recursive latent thoughts. arXiv preprint arXiv:2510.07358, 2025.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In International Conference on Learning Representations, 2020.

Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, and Micah Goldblum. Teaching pretrained language models to think deeper with retrofitted recurrence. arXiv preprint arXiv:2511.07384, 2025.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022. URL https://transformer-circuits.pub/2022 in-context-learning-and-induction-heads/index.html.

Core Francisco Park, Andrew Lee, Ekdeep Singh Lubana, Yongyi Yang, Maya Okawa, Kento Nishi, Martin Wattenberg, and Hidenori Tanaka. ICLR: In-Context Learning of Representations. In International Conference on Learning Representations, 2025.

Sashank J Reddi, Sobhan Miryoosefi, Stefani Karp, Shankar Krishnan, Satyen Kale, Seungyeon Kim, and Sanjiv Kumar. Efficient training of language models using few-shot learning. In International Conference on Machine Learning, 2023.

Nikunj Saunshi, Stefani Karp, Shankar Krishnan, Sobhan Miryoosefi, Sashank Jakkam Reddi, and Sanjiv Kumar. On the inductive bias of stacking towards improving reasoning. Advances in Neural Information Processing Systems, 2024.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In International Conference on Learning Representations, 2025.

Aaditya Singh, Stephanie Chan, Ted Moskovitz, Erin Grant, Andrew Saxe, and Felix Hill. The Transient Nature of Emergent In-Context Learning in Transformers. In Advances in Neural Information Processing Systems, 2023.

Aaditya K Singh, Ted Moskovitz, Felix Hill, Stephanie C.Y. Chan, and Andrew M Saxe. What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. In International Conference on Machine Learning, 2024.

Sho Takase and Shun Kiyono. Lessons on parameter sharing across layers in transformers. In Proceedings ofThe Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP), pp. 78–90, 2023.

Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron Wallace, and David Bau. Function vectors in large language models. In International Conference on Learning Representations, 2024.

Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, 2023.

Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model. arXiv preprint arXiv:2506.21734, 2025.

Peihao Wang, Rameswar Panda, Lucas Torroba Hennigen, Philip Greengard, Leonid Karlinsky, Rogerio Feris, David Daniel Cox, Zhangyang Wang, and Yoon Kim. Learning to grow pretrained models for efficient transformer training. In International Conference on Learning Representations, 2023.

Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu, Wenhao Huang, Shen Yan, and Jian Li. Smelt: Scaling laws for compute-matched moe looped transformers. arXiv preprint arXiv:2609.01343, 2026.

Liu Yang, Kangwook Lee, Robert D Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. In International Conference on Learning Representations, 2024.

Yiqun Yao, Zheng Zhang, Jing Li, and Yequan Wang. Masked structural growth for 2x faster language model pre-training. In International Conference on Learning Representations, 2024.

Kayo Yin and Jacob Steinhardt. Which attention heads matter for in-context learning? In International Conference on Machine Learning, 2025.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, et al. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025.

## A LANGUAGE MODEL TRAINING AND CONFIGURATIONS

## A.1 TRAINING SETUP

We train autoregressive, decoder-only Transformers with a next-token cross-entropy objective. The 360M and 1.7B configurations follow SmolLM (Ben Allal et al., 2024), and the 3B configuration follows SmolLM3 (Bakouch et al., 2025). All use pre-RMSNorm, SwiGLU feed-forward layers, tied input and output embeddings, and no biases, dropout, or query/key normalization. Table 1 summarizes the dense reference configurations. Model sizes refer to these references; looped models have fewer unique parameters, and grown models reach the reference depth only during training. Comparisons between methods share the data and optimization settings within each size. The sizes differ in training budget, and the 3B configuration also changes the data, tokenizer, and context length, so they test robustness of observations across all these differences.

Table 1: Language-model training configurations.
<table><tr><td></td><td>360M</td><td>1.7B</td><td>3B</td></tr><tr><td>Dense layers L</td><td>32</td><td>24</td><td>36</td></tr><tr><td>Hidden / MLP width</td><td>960 / 2560</td><td>2048 / 8192</td><td>2048 / 11008</td></tr><tr><td>Query / key-value heads</td><td>15 / 5</td><td>32 / 32</td><td>16 /4</td></tr><tr><td>Vocabulary size</td><td>49,152</td><td>49,152</td><td>128,256</td></tr><tr><td>Context length C</td><td>2048</td><td>2048</td><td>4096</td></tr><tr><td>Rotary base</td><td> $1 0 ^ { 4 }$ </td><td> $1 0 ^ { 4 }$ </td><td> $5 \times 1 0 ^ { 4 }$ </td></tr><tr><td>RMSNorm €</td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Global batch (sequences)</td><td>512</td><td>1024</td><td>576</td></tr><tr><td>Training steps</td><td>200,000</td><td>200,000</td><td>42,400</td></tr><tr><td>Training tokens (billions)</td><td>209.7</td><td>419.4</td><td>100.0</td></tr><tr><td>Peak learning rate</td><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Linear warm-up (steps)</td><td></td><td>2000</td><td></td></tr><tr><td>AdamW  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td></td><td>(0.9,0.95)</td><td></td></tr><tr><td>AdamW €</td><td></td><td>10-8</td><td></td></tr><tr><td>Gradient clipping norm</td><td></td><td>1</td><td></td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.1</td></tr><tr><td>Embedding weight decay</td><td>0.01</td><td>0.01</td><td>0</td></tr><tr><td>Weights / activations</td><td></td><td>bf16</td><td></td></tr><tr><td>Gradient accumulation</td><td></td><td>fp32</td><td></td></tr><tr><td>Base initialization std. σ</td><td>0.0417</td><td>0.0221</td><td>0.02</td></tr></table>

Data and positional encoding. The 360M and 1.7B models use the SmolLM-corpus mixture of FineWeb-Edu-dedup, Cosmopedia-v2, Python-Edu, and Open-Web-Math (Ben Allal et al., 2024), contributing approximately 70%, 15%, 9%, and 6% of training tokens, respectively. They use the cosmo2 tokenizer and rotary position embeddings in all layers. The 3B models use the SmolLM3 tokenizer and its 100B-token ablation recipe. The initial mixture includes FineWeb-Edu, DCLM, multilingual FineWeb-2, scientific text, code, and mathematics. At step 36,040, training switches to the decay mixture, with increased emphasis on mathematics, code, and synthetic reasoning. All models use shuffled, packed sequences of the lengths in Table 1; 3B additionally masks attention across document boundaries and omits rotary embeddings in every fourth layer.

Optimization and initialization. All models use AdamW with the hyperparameters in Table 1. After warm-up, learning rates stay constant, then decay to zero using $1 - { \bar { \sqrt { \cdot } } }$ decay from step 170,000 (360M/1.7B) or linear decay from step 36,040 (3B). Matrix weights use zero-mean normal initialization with base standard deviation $\sigma ;$ RMSNorm scales start at one. Attention output and MLP down projections additionally receive a $1 / \sqrt { 2 L _ { \mathrm { i n i t } } }$ factor, where $L _ { \mathrm { i n i t } } = 4$ for grown models and equals the effective depth for dense and looped models. The 360M looped model omits this factor, so its comparisons also include an initialization difference.

## A.2 GROWING AND LOOPING

Dense and grown models. The dense baseline has L independently parameterized layers throughout training. Grown models start with $b = 4$ layers and repeatedly duplicate four contiguous layers until reaching L. A duplication copies both the layer weights and Adam’s first and second moments. The copies become independent trainable layers immediately; embeddings are retained, and neither the global step nor the learning-rate schedule is reset. MIDAS (Saunshi et al., 2024) selects the middle four-layer block, choosing the block just before the center when the number of blocks is even. LIDAS (Kapl et al., 2026a) instead selects the middle four layers without respecting previous block boundaries. For example, LIDAS maps an eight-layer stack to

$$
[ 0 , 1 , \ 2 , \ 3 , 4 , 5 , \ 6 , 7 ] \ \longmapsto \ [ 0 , 1 , \ 2 , \ 3 , 4 , 5 , \ 2 , \ 3 , 4 , 5 , \ 6 , 7 ] .
$$

Layer indices in this expression identify the source of the copied parameters, not persistent weight sharing.

We use the Prop-1 schedule (Saunshi et al., 2024), allocating steps in proportion to depth. For $K = L / b$ stages, stage i has depth ib and receives

$$
T _ { i } = \frac { i } { \sum _ { j = 1 } ^ { K } j } T _ { \mathrm { p r o p } } , \qquad i = 1 , \dots , K ,
$$

steps, up to integer rounding. The budget $T _ { \mathrm { p r o p } }$ excludes learning-rate decay, which is appended to the final full-depth stage. For 360M/1.7B, $\dot { T } _ { \mathrm { p r o p } } = 1 7 0 { , } 0 0 0$ , including warm-up in the first stage. For 3B, the 2000 warm-up steps are also excluded from the proportional split and prepended to the first stage, giving $T _ { \mathrm { p r o p } } ~ \stackrel { - } { = } ~ 3 \bar { 6 } , 0 4 0 - 2 0 0 0 ~ = ~ 3 4 , 0 4 0$ Only LIDAS is evaluated at 3B. At 1.7B, MIDAS branches from LIDAS’s eight-layer checkpoint at step 24,285: the two methods share the preceding trajectory, including the first growth event.

Looped models. We write Loop p - b×r - c for p prelude layers, a block of b layers applied r times, and c coda layers. The effective depth is $p + b r + c , $ but only $p + b + c$ layers have unique parameters. Gradients from each application accumulate into the shared parameters. Prelude and coda are omitted from the notation when empty. The 1.7B comparison uses Loop 4-8×2-4: 24 layer applications, 16 unique layers, and approximately 1.17B parameters. The main 3B comparison uses Loop 10 - 8×2 - 10: 36 applications, 28 unique layers, and approximately 2.45B parameters. Its eight-layer loop preserves the rotary/no-position-embedding role of each shared layer. The 360M comparison uses Loop 4-12×2-4 (32 applications, 20 unique layers, approximately 244M parameters). These comparisons match effective final depth and training tokens within a size; growing uses fewer layer applications during early training, and looping reduces the number of unique parameters.

## A.3 CHECKPOINTS AND COMPARISON SCOPE

Final-checkpoint evaluations use step 200,000 for 360M/1.7B and step 42,400 for 3B. The 1.7B and 3B trajectories generally use checkpoints every 2000 steps, with additional saves at growth boundaries and at the final step. The 360M dense and grown trajectories use a 10,000-step grid, with additional growth saves. At a growth boundary, we distinguish the trained pre-growth checkpoint from the immediate post-growth checkpoint, before any further optimizer update. The latter is shown as a hollow marker at the same training step. Vertical rules mark the growth events.

The appendix compares dense, MIDAS, LIDAS, and the selected looped model at 360M and 1.7B, and dense, LIDAS, and the selected looped model at 3B. There is no 3B MIDAS run. All comparisons use one language-model training seed per arm; evaluation bands do not estimate variation across training runs. The configuration differences and 360M loop initialization exception in $\mathsf { A p - }$ pendix A.1 apply throughout.

Loop configurations in appendix figures. A single green legend entry, “Loop”, denotes Loop 4-12×2-4 at 360M, Loop 4-8×2-4 at 1.7B, and Loop 10-8×2-10 at 3B. The size of the panel determines the configuration. The architecture and parameter-sharing convention are given in Appendix A.2.

## B CONTEXT USE IN LANGUAGE MODELS

This section gives the evaluation protocols and results across sizes for general benchmarks, token loss, downstream few-shot performance, and graph tracing. Shared training configurations and the size-specific Loop definitions are given in Appendix A.

## B.1 GENERAL BENCHMARK EVALUATIONS

Table 2 reports general benchmark performance using OLMES (Gu et al., 2025) at the final checkpoints: step 200,000 for 360M/1.7B and step 42,400 for 3B. We evaluate the nine-task core suite and MMLU (Hendrycks et al., 2021) with five fixed demonstrations and a maximum context length of 2048 tokens for every model, including the 3B models.

Table 2: General benchmark performance across model sizes. Accuracy (%), with higher values better. 9-task MC is the unweighted mean over ARC-Easy, ARC-Challenge, BoolQ, CommonsenseQA, HellaSwag, OpenBookQA, PIQA, Social IQa, and WinoGrande; MMLU is the unweighted mean over its $\breve { 5 7 }$ subjects. Both columns use five-shot ranked-classification (RC) scoring of answer continuations with OLMES task-specific normalization and a 2048-token context limit. Here MC denotes multiple-choice benchmarks; scores use the RC formulation throughout, rather than answer-letter scoring. Model sizes denote dense reference configurations; loop layouts follow Appendix A.2. Best values within each size are bold.
<table><tr><td>Model</td><td>Strategy</td><td>9-task MC ↑</td><td>MMLU↑</td></tr><tr><td rowspan="4">36M</td><td>Baseline</td><td>54.69</td><td>34.67</td></tr><tr><td>MIDAS</td><td>56.00</td><td>34.11</td></tr><tr><td>LIDAS</td><td>57.10</td><td>34.09</td></tr><tr><td>Loop 4-12×2-4</td><td>54.21</td><td>33.69</td></tr><tr><td rowspan="4">1.B</td><td>Baseline</td><td>62.97</td><td>40.12</td></tr><tr><td>MIDAS</td><td>64.61</td><td>40.43</td></tr><tr><td>LIDAS</td><td>64.69</td><td>40.86</td></tr><tr><td>Loop 4-8×2-4</td><td>63.25</td><td>39.89</td></tr><tr><td rowspan="3">3B</td><td>Baseline</td><td>61.33</td><td>39.15</td></tr><tr><td>LIDAS</td><td>62.89</td><td>39.28</td></tr><tr><td>Loop 10-8×2-10</td><td>60.53</td><td>38.32</td></tr></table>

The core suite uses test splits for ARC-Easy, ARC-Challenge, and OpenBookQA, and validation splits for the remaining tasks. ARC-Easy, BoolQ, HellaSwag, PIQA, and Social IQa use the standard fixed 1000-example subsamples; ARC-Challenge, CommonsenseQA, OpenBookQA, and Wino-Grande use their full splits of 1172, 1221, 500, and 1267 examples, respectively. MMLU uses all 14,042 test examples across its 57 subjects. Following OLMES, answer log-likelihoods are normalized by character count for ARC-Easy, HellaSwag, PIQA, Social IQa, and MMLU; corrected by unconditional answer log-likelihood for ARC-Challenge, CommonsenseQA, and OpenBookQA; and left unnormalized for BoolQ and WinoGrande.

## B.2 TOKEN LOSS AND THE ICL SCORE

Figure 1(a,b), Figure 7, and Figure 13 (bottom row) use a fixed set of 10,000 long FineWeb-Edudedup documents. After tokenization, we retain documents with at least $C + 1$ tokens and truncate to $C + 1$ , yielding $C$ next-token predictions per document. Each forward pass therefore uses a single document and every position has the same document count. The 2048- and 4096-position evaluations use their model’s native tokenizer and separately selected document sets.

For tokens $x _ { 0 } , \ldots , x _ { C }$ , define

$$
\ell _ { j } = - \log p ( x _ { j } \mid x _ { 0 } , \ldots , x _ { j - 1 } ) , \qquad j = 1 , \ldots , C ,\tag{1}
$$

and let $\bar { \ell } _ { j }$ be its mean over documents, in nats. Position labels in the trajectory and position plots use this one-based prediction index. Evaluation uses bf16 forward passes. For the position-wise ∆NLL panels in Figure 1(a) and Figure 7, position means are grouped into non-overlapping power-of-two bins $( 2 ^ { y } , 2 ^ { y \mp 1 } ] \mathrm { f o r } y = 0 , \ldots , \log _ { 2 } \dot { C } - 1$ , with position 1 added to the first bin. The integer position ranges are therefore $1 { - } 2 , 3 { - } 4 , 5 { - } 8 , \bar { . } . . , C / 2 { + } 1 { - } C .$ , giving 11 bins at $C = 2 0 4 8$ and 12 at $C = 4 0 9 6 ;$ every position belongs to exactly one bin. Each bin mean is plotted at the bin’s upper endpoint on a base-2 logarithmic axis. The plotted quantity is the model’s binned NLL minus the same-size dense baseline’s binned NLL. Thus, a negative value denotes better token prediction at those positions, rather than a larger improvement relative to the model’s own early-token loss.

The dashed references in Figure 1(a) and Figure 7 average the raw position means with equal weight, before any position binning. With $D = 1 \bar { 0 } { , } 0 0 0$ documents and C scored prediction positions per document $( \bar { C } = 2 0 4 8$ at 360M/1.7B and 4096 at 3B), the reference for model m is

$$
\begin{array} { r l r } {  { \overline { { \mathrm { { A N L L } } _ { m } } } = \frac { 1 } { C } \sum _ { j = 1 } ^ { C } ( \bar { \ell } _ { j } ^ { ( m ) } - \bar { \ell } _ { j } ^ { ( \mathrm { b a s e } ) } ) } } \\ & { } & { = \frac { 1 } { D C } \sum _ { d = 1 } ^ { D } \displaystyle \sum _ { j = 1 } ^ { C } ( \ell _ { d , j } ^ { ( m ) } - \ell _ { d , j } ^ { ( \mathrm { b a s e } ) } ) } \\ & { } & { = \overline { { \mathrm { { N L L } } _ { m } } } - \overline { { \mathrm { { N L L } } _ { \mathrm { b a s e } } } } . } \end{array}\tag{2}
$$

This is the difference in token-averaged dataset NLL on this fixed long-document subset, using the same truncation and per-document context resets as the position curves. It is not an unweighted average of the logarithmic bin means, whose position counts differ.

Following the token-loss diagnostic of Olsson et al. (2022), we summarize context use at each checkpoint by

$$
S _ { \mathrm { I C L } } = \frac { 1 } { 2 1 } \sum _ { j = 4 0 } ^ { 6 0 } \bar { \ell } _ { j } - \frac { 1 } { 2 1 } \sum _ { j = 4 9 0 } ^ { 5 1 0 } \bar { \ell } _ { j } .\tag{3}
$$

A positive score indicates lower NLL at the later window. A change in this difference can arise from either window; it need not imply improved absolute prediction at late positions. Windowdecomposition plots therefore report the two means separately. When subtracting the dense model’s corresponding window loss along training, we use only steps at which both models have a saved checkpoint, without interpolating a baseline at growth steps. The combined head-ablation figure uses the same window definition and reports its change from the unablated model in Figure 6(b), while Figure 22 reports absolute scores.

Loss reduction across context positions. For the final-checkpoint comparison in Figure 1(b) and Figure 7, we keep the early window at positions 40–60 and let the later window be the power-of-two bin ending at the target position x, $\bar { W _ { x } } = \{ x / 2 + 1 , \ldots , x \}$ , for $x \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 , 2 0 4 8 \}$ } and additionally 4096 at 3B. These are the same bins as in the $\Delta \mathrm { \bar { N L I } }$ panels. We plot

$$
\Delta S _ { m } ( x ) = \left[ \frac { 1 } { 2 1 } \sum _ { j = 4 0 } ^ { 6 0 } \bar { \ell } _ { j } ^ { ( m ) } - \frac { 1 } { | W _ { x } | } \sum _ { j \in W _ { x } } \bar { \ell } _ { j } ^ { ( m ) } \right] - \left[ \frac { 1 } { 2 1 } \sum _ { j = 4 0 } ^ { 6 0 } \bar { \ell } _ { j } ^ { ( \mathrm { b a s e } ) } - \frac { 1 } { | W _ { x } | } \sum _ { j \in W _ { x } } \bar { \ell } _ { j } ^ { ( \mathrm { b a s e } ) } \right] .\tag{4}
$$

This is the model’s early-to-late loss reduction minus the baseline’s. Equivalently, $\Delta S _ { m } ( x )$ equals the model’s ∆NLL over positions 40–60 minus its ∆NLL in the bin ending at x, the value plotted in the top row of Figure 7. Scores and the full-context NLL references are computed from raw position means.

![](images/fa0cae1821645bbe39b3c7c0ddf84ee8335f832dea13af24cbbaacd686c1bc6c.jpg)  
Figure 7: Token loss and context gains across model sizes. Columns show 360M, 1.7B, and 3B, using the position bins of Figure 1. Top: final-checkpoint NLL relative to the same-size dense model in power-of-two position bins plotted at their upper endpoints, with matching-color dashed full-context means computed over raw positions. Bottom: the early-to-late loss reduction relative to dense (Equation (4)) for later bins ending at $x \in \{ 2 5 6 , \ldots , C \}$ . All curves use 10,000 long FineWeb-Edu documents. Native contexts are $C \ = \ { 2 0 4 8 } / { 2 0 4 8 } / { 4 0 9 6 }$ positions; the evaluation protocol is given in Appendix B.2. Loop configurations are specified in Appendix A.3.

## B.3 FEW-SHOT SCALING ON DOWNSTREAM TASKS

We measure downstream performance as a function of the number of demonstrations, following the demonstration-task protocol also used by SMELT (Wang et al., 2026). We use lm-eval-harness version 0.4.3, bf16 weights, and a maximum input length equal to the training context. Results are averaged over three demonstration seeds. We re-tokenize requests to check that prompts fit and cap each task’s shot range to avoid truncation.

GSM8K. The main comparison in Figure 1(c) reports GSM8K at 3B for $k \in \{ 0 , 1 , 2 , 4 , 8 \}$ . We score final answers extracted from generated solutions using strict exact match. Figure 8 extends this comparison to 360M and 1.7B, using $k \in \{ 0 , 1 , 2 , 4 \}$ for their shorter contexts.

Additional multiple-choice and generation tasks. Figure 9 reports separate averages over two fixed task sets, both evaluated at $k \in \{ 0 , 1 , 2 , 4 , 8 \}$ across all model sizes. The five multiple-choice tasks are ARC-Easy, ARC-Challenge, CommonsenseQA, HellaSwag, and PIQA. ARC, HellaSwag, and PIQA use length-normalized accuracy; CommonsenseQA uses accuracy. The three additional generation tasks are the BigBench tasks QA-Wikidata, Operators, and Repeat-Copy-Logic, scored by exact match after whitespace stripping, ignoring case and punctuation.

For the multiple-choice task set T, we report mean chance-centered accuracy:

$$
A _ { \mathrm { c e n t e r e d } } ( T ) = \frac { 1 0 0 } { | T | } \sum _ { t \in \mathcal { T } } \frac { a _ { t } - c _ { t } } { 1 - c _ { t } } .\tag{5}
$$

Here $a _ { t }$ is task accuracy averaged over demonstration seeds, and $c _ { t }$ is 0.25 for ARC-Easy, ARC-Challenge, and HellaSwag, 0.5 for PIQA, and 0.2 for CommonsenseQA. Zero centered accuracy denotes chance performance and 100 denotes perfect performance. For the three generation tasks, we report the equally weighted mean of raw exact-match accuracies in percent, without chance centering. Figure 10 shows the individual task curves. Appendix bands show standard error across demonstration seeds, computed from per-seed task averages for the two subsets; they do not estimate uncertainty across training runs.

![](images/570d99d921d77f7e7cc9b890c7b697f82cc50e43ca51b2333ec1b16885704e95.jpg)  
Figure 8: Few-shot GSM8K performance across model sizes. Panels show mean strict exactmatch accuracy for 360M, 1.7B, and 3B, with standard-error bands over three demonstration seeds. The shot range is $k \in \{ 0 , 1 , 2 , 4 \}$ for 360M and 1.7B and extends to eight shots for 3B. Details in Appendix B.3 for the evaluation and loop configurations are specified in Appendix A.3.

![](images/5dee70368479028c5031b9566cf2d1f5e1f19e11f825f1cc7fc16a2ff7e6557d.jpg)  
Figure 9: Few-shot performance on additional multiple-choice and generation tasks. Columns show 360M, 1.7B, and 3B. The top row averages chance-centered accuracy over five multiple choice tasks; the bottom row averages exact-match accuracy over three generation tasks, for $k \in \{ 0 , 1 , 2 , 4 , 8 \}$ . Tasks have equal weight within each subset; bands show standard error across three demonstration-seed subset averages. Individual tasks and normalization are given in Figure 10 and Appendix B.3. Loop configurations are specified in Appendix A.3.

![](images/cbcecc8adf237362950800ff5ab4f0f6822ca8ea7196a089a29a2f3a0f3f84a1.jpg)  
Figure 10: Few-shot performance on individual multiple-choice and generation tasks. Columns show 360M, 1.7B, and 3B. The top five rows show chance-centered accuracy for multiple-choice tasks; the bottom three show exact-match accuracy for generation tasks. Bands show standard error across three demonstration seeds at $k \in \{ 0 , 1 , 2 , \dot { 4 } , 8 \}$ . Subset averages and normalization are given in Figure 9 and Appendix B.3. Loop configurations are specified in Appendix A.3.

## B.4 IN-CONTEXT GRAPH TRACING

We additionally evaluate the grid-graph setting of Park et al. (2025). Nine or sixteen distinct singletoken nouns label the nodes of a $\bar { 3 } \times 3$ or $4 \times 4$ grid, respectively. Figure 11 shows both grid sizes with frequent-word labels for all three model sizes. A random walk is presented as a space-separated sequence of node words, without an additional delimiter. After each node, we sum the model’s full-vocabulary next-token probabilities over the words labeling that node’s valid neighbors. This probability mass is the graph task’s “accuracy”.

The frequent-word comparison starts from a curated pool of 250 nouns spanning 18 semantic fields. For each tokenizer, we retain nouns that form a single token when preceded by a space and select the 64 with the lowest token IDs. The 360M/1.7B models use cosmo2, whereas 3B uses the SmolLM3 tokenizer; token ID serves as a frequency proxy. Each of 200 trials draws a new node-word assignment and random walks, with one sequence per starting node; scores are averaged over starting nodes within the trial. Models of a given size see the same trials. The horizontal axis counts node tokens in context, up to the native training context length, and bands show the standard error over trials. The one- and two-observation references measure the fraction of positions whose current node has already appeared as a source of a transition at least once or twice. They indicate when copying a previously observed successor becomes possible, rather than upper bounds on graph learning. The 3B comparison includes the dense, LIDAS, and looped models.

![](images/cdfbacae6471ec4c51c0008c1882947c7e5f6781dbd1639dbff8c0cdddb140e4.jpg)  
Figure 11: Word-grid tracing with frequent-word labels. Columns show 360M, 1.7B, and 3B; rows show $3 \times 3$ and $4 \times 4$ grids. Curves give full-vocabulary probability mass assigned to valid neighbors, with standard error over 200 trials. Black curves are one- and two-observation edgememorization references. All panels share the probability scale and use base-2 logarithmic context axes. Both settings include the same selected methods at each model size. See Appendix B.4 for the protocol. Loop configurations are specified in Appendix A.3.

## C REGBENCH: SETUP AND ADDITIONAL RESULTS

## C.1 TASK AND DATA

We use the released RegBench implementation of Akyürek et al. (2024). Each sequence concatenates random walks from one DFA, separated by |. Candidate automata have 4–12 states, an alphabet of 4–18 symbols, and one to three outgoing transitions per state; they are minimized and trimmed before splitting. Each automaton supplies ten to nineteen strings of 1–49 symbols. The fixed dataset contains 40,000 training, 500 validation, and 500 test sequences, split by sampled automaton. All models see the same data. Training and validation cross-entropy mask separator targets and padding.

## C.2 MODELS, OPTIMIZATION, AND SELECTION

We use GPT-2-style pre-LayerNorm Transformers with learned token and position embeddings, two attention heads, a GELU MLP of width 4d, and tied input/output embeddings. Embedding and attention dropout are 0.1. Dense models have twelve independent layers; the looped model applies two shared layers six times. Grown models reach depth twelve by duplicating the middle b layers and their AdamW moments at each growth event. Copies then train independently, retaining the global learning-rate schedule without restarting warm-up.

All runs use AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) ~ = ~ ( 0 . 9 , 0 . 9 9 9 )$ , batch size 32, and 200 epochs (250,000 steps). The learning rate warms up for 25,000 steps and then decays by a cosine schedule to one tenth of its peak. For each configuration, we select the lowest-validation-loss cell over widths $d \in \{ 6 4 , 1 2 8 , 2 5 6 \}$ , peak learning rates $\{ 1 0 ^ { - 4 } , 2 . 5 \times 1 0 ^ { - 4 } \}$ , and weight decays {0.01, 0.1}. Validation runs after every epoch; training completes the full budget. We evaluate the best-validation checkpoint, restricting grown models to their final full-depth stage, and repeat the selected cells with four training seeds.

Figure 12 includes all five repeated configurations: dense 12L, Loop 2×6, and grown $b = 2 / \mathsf { p r o p } .$ $1 , b = 2 / \mathrm { p r o p } { - 2 }$ , and $b = { \mathrm { 3 / p r o p } } { \mathrm { - } } 3$ . Their widths are 128 except for Loop 2×6, which uses 256. All select weight decay 0.1; the peak learning rate is $1 0 ^ { - 4 }$ for prop-1 and $2 . 5 \times 1 0 ^ { - 4 }$ otherwise. These schedules extend the proportional allocation in Appendix A.2, dividing the full training budget among stages in proportion to depth raised to powers 1, 2, and 3, respectively. The main figure uses $b = { \mathrm { 3 / p r o p } } { \mathrm { - } } 3$ , selected by mean validation loss across seeds. Selection never uses test performance. Comparisons match data and optimizer steps, but not width or compute.

## C.3 METRICS AND POSITION DECOMPOSITION

Let q be uniform over the DFA’s legal outgoing symbols and zero elsewhere. We report $\begin{array} { r } { \mathrm { T V D } ( p , q ) = \frac { 1 } { 2 } \sum _ { v } | p ( v ) - q ( v ) \rrangle } \end{array}$ and DFA accuracy, which scores whether the argmax is a legal continuation rather than the particular sampled next symbol. Following the released evaluator, TVD includes all non-padding output positions, including separator targets and the final position; q assigns no mass to the separator. Accuracy excludes separator targets and includes the final position. These masks differ from the training loss.

Figures 2 and 12 report full-test means and position-wise differences from dense. Full-test scores retain the evaluator’s batch aggregation; position statistics pool sums and counts within each seed, then average over seeds. String and symbol indices start at 1. In Figure 2, we retain only string indices present in every one of the 500 test sequences (strings 1–10), using all scored symbol positions within these strings (through index 50 for TVD). No symbol-range, coverage, or minimum-support cutoff is applied; only empty bins are omitted. The appendix figure uses all strings 1–19 and all available symbol positions (through 50 for TVD and 49 for accuracy), without range or coverage cutoffs; only empty bins are omitted. Within each figure and metric, the string-index and symbolindex views use the same population; the latter pools symbol indices in bins of five and separates string 1 from the remaining strings (2–10 in the main figure and 2–19 in the appendix).

Error bars show standard error over the four training seeds. Bands on differences use the unpaired standard error $\sqrt { s _ { \mathrm { m o d e l } } ^ { 2 } / 4 + s _ { \mathrm { d e n s e } } ^ { 2 } / 4 }$ , where s is the sample standard deviation of seed-specific bin means. Negative TVD differences and positive accuracy differences favor the alternative model.

Position averages need not equal the full-test statistics because of these different aggregations and, in the main figure, the string restriction.

## C.4 RESULTS ACROSS CONFIGURATIONS

(c) Strings 1–19  
![](images/a8a1832ad8576f2bf2708bb2ab7997750d13a58d7256943f64003a8a70d4ba94.jpg)

![](images/c77f58b837834f29a12a3193799217bb540eae783894d48daa5a6c33137dd550.jpg)

![](images/f0e96f243f52ec75ce00c7ac9fcb0a13c8a42da2bae53149bebdbfebf309e4b5.jpg)  
(e) Two string bins

![](images/d2f50820f63264a7a75769c8410ea9cbdc94796993af35f9ba2a4bbd949a4ade.jpg)  
(f) Two string bins

![](images/4b8f1cf0f66066d23b8439eceb8e114b5f2bd4a81f188a92115731effd4bbedc.jpg)

![](images/14f38ab160e2aa5c1829eff58f6d5cf649bdcb619e55c745e582c519e86670f8.jpg)  
Figure 12: RegBench across all configurations evaluated with four seeds. Columns show TVD (left) and DFA accuracy (right). Rows show full-test means (top), differences from dense by string index (middle), and differences by symbol index (bottom) for string 1 (dashed) and strings 2–19 (dotted). All available strings and symbols are retained; indices start at 1 and symbol positions use bins of five. The positive mean Loop 2×6 TVD differences at strings 18 and 19 concern only 102 (20.4%) and 54 (10.8%) of the 500 test sequences, respectively. Accuracy differences are in percentage points. Error bars and difference bands show standard errors across training seeds. All models have effective depth twelve; validation-selected widths are 128 except for Loop 2×6 (256). For additional details see Appendix C.3.

## D OMNIGLOT SETUP

## D.1 DATA, SEQUENCE FORMAT, AND ARCHITECTURE

We follow the image-based setup of Chan et al. (2022). We merge the Omniglot training and test splits, giving 1623 character classes with twenty $1 0 5 \times 1 0 5$ grayscale exemplars each. A fixed class permutation reserves ten classes for held-out evaluation, while the remaining 1613 classes are sampled uniformly during training. We use no image augmentation, downsampling, or pixel noise. Repeated occurrences of a class within a sequence use distinct exemplars sampled without replacement.

Each sequence contains eight image–label context pairs and one query image. Their embeddings are interleaved as $[ x _ { 1 } , y _ { 1 } , . . . , x _ { 8 } , y _ { 8 } , x _ { 9 } ]$ , giving seventeen tokens, with sinusoidal absolute positional encodings. Training minimizes cross-entropy only at the query position. Images are embedded by a jointly trained small ResNet: a $7 \times 7$ stride-two convolution with 64 channels, batch normalization, ReLU, and a $3 \times 3$ stride-two max pool, followed by four groups of two bottleneck residual blocks with output channels 16/32/32/64 and strides 1/2/2/2, and global average pooling. Labels use a learned $\bar { 1 6 2 3 } \times 6 4$ embedding table.

The sequence model is a causal pre-LayerNorm Transformer of width 64, with eight attention heads, GELU MLPs of width 256, no dropout, a final LayerNorm, and a linear head with 1623 outputs. Figure 3 compares dense depths 4 and 12, a Loop 4×3 model, and a LIDAS-grown model. The latter branches from the four-layer dense run, grows to eight layers at step 50,000, and to twelve at step 100,000 by duplicating the middle four layers. We copy Adam moments together with weights and then train all copies independently, retaining the global step and learning-rate schedule. Thus, the grown model and the four-layer reference have exactly the same pre-growth trajectory. The looped model shares only Transformer blocks, the image encoder, embeddings, and final readout are not repeated and are identical in setup for all versions.

## D.2 TRAINING DISTRIBUTIONS AND OPTIMIZATION

Competing context and weight solutions. In Figure 3(a), each training class has a fixed label and every sequence is bursty, $p ( \mathrm { b u r s t y } ) = 1$ . We sample two training classes, place three exemplars of each in the context, add two further uniformly sampled training items, and shuffle the eight context items. The query is a new exemplar of one of the two burst classes. Its label can be recovered either by matching it to context examples or by recalling its fixed training-class association.

ICL-only training. In Figure 3(b), training uses four-shot, two-way sequences: two training classes each contribute four context exemplars, and the query is a fifth exemplar of one of them. Labels 0 and 1 are reassigned randomly to the two classes in every sequence. There is therefore no persistent class-to-answer mapping that can solve the query without context. We intentionally only modify the training data, i.e., the model still has 1623 output logits, although only labels 0 and 1 are correct in this setting. Architecture, optimizer, and sequence length also remain unchanged.

Both settings use Adam with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ and $\epsilon = 1 0 ^ { - 8 }$ , batch size 32, no weight decay or gradient clipping, and learning rate

$$
\eta ( t ) = 3 \times 1 0 ^ { - 4 } \operatorname* { m i n } \left( { \frac { t } { 4 0 0 0 } } , { \sqrt { \frac { 4 0 0 0 } { t } } } \right) , \qquad t > 0 .\tag{6}
$$

This gives 4000 steps of linear warm-up followed by inverse-square-root decay. Each fixed-label configuration uses four seeds, and each ICL-only configuration uses eight. These control initialization and sampling independently of the fixed class split. For the reported comparisons, we use 40M training steps in the fixed-label setting and 1M training steps in the ICL-only setting. Checkpoints are saved every 50,000 steps and at powers of two starting at 64 for finer early-training resolution.

## D.3 EVALUATION: ICL AND IWL

The ICL probe uses four-shot, two-way sequences from the ten held-out classes, with new classto-label assignments in each sequence. The context contains four distinct exemplars per class, and the query is a fifth exemplar of one of them. We report accuracy with the argmax restricted to logits 0 and 1 (chance $1 / 2 )$ . Success therefore requires using the current context for a class never encountered in training. The IWL probe instead samples the query uniformly from the 1613 training classes, uses its fixed training label, and guarantees that its class is absent from the eight context examples. It reports the unrestricted argmax over all 1623 outputs, consequently, uniform guessing over that output space gives chance 1/1623. The IWL probe is used only for the fixed-label setting.

Every checkpoint is evaluated on the same sequences generated with a fixed evaluation seed. The fixed-label setting uses 10,000 sequences per probe and the ICL-only setting uses 2000 for its ICL probe. Curves average over four training seeds in the fixed-label setting and eight in the ICL-only setting, at checkpoints shared by all seeds within each configuration; bands are one standard error across seeds. For display, each seed trajectory is smoothed with a centered, equally weighted moving average over 51 checkpoints in the fixed-label setting and 3 in the ICL-only setting, shortening windows at the boundaries after truncating to the reported training budget. Means and standard errors are then computed across the smoothed seed trajectories. Endpoint statistics and emergence thresholds in the text use unsmoothed evaluations. The two plots in panel (a) use the matched 40M step horizon, while panel (b) uses the shared 1M-step horizon.

## E INDUCTION-HEAD DEVELOPMENT AND READOUTS

## E.1 MEASURES AND AGGREGATE TRAINING TRAJECTORIES

We implement the head diagnostics of Yin & Steinhardt (2025) and Aoyama et al. (2026) on converted language-model checkpoints, using fp32 weights and TF32 matrix multiplications. The same prompts and random sequences are reused across checkpoints and models within each size. Attention-head outputs are measured immediately before the attention output projection. For looped models, each application of a shared head is evaluated separately, since its input and causal role can differ across passes.

Prefix matching. We generate 1000 sequences of the form [doc-start, $x _ { 1 } , \ldots , x _ { 5 0 } , x _ { 1 } , \ldots , x _ { 5 0 } ]$ drawing 50 distinct regular vocabulary tokens per sequence. For a second-half token, the induction target is the token following its earlier occurrence. Prefix-matching score (PS) is the mean attention probability assigned to the target position, averaged over second-half positions and sequences. With zero-based sequence positions $p \in \{ 5 1 , \ldots , 1 0 0 \}$ }, the target position is $p - 4 9 ;$ ; the last position uses the cyclic continuation $x _ { 1 }$

Function-vector scores. Following Todd et al. (2024); Yin & Steinhardt (2025), we estimate a function-vector (FV) score on 29 training tasks using ten demonstrations and 100 clean and 100 corrupted prompts per task. We do not filter tasks or prompts by performance on clean prompts. For each head and task, we compute its mean last-token output on clean prompts. We then shuffle the demonstration labels and replace that head’s last-token output with the clean task mean. The FV score is the patched minus unpatched probability of the correct first answer token on the corrupted prompt, averaged over prompts and then tasks. Head selection therefore uses a causal intervention on demonstration-based task execution, rather than the attention pattern alone.

Few-shot task performance. Figure 6, Figure 13 (third row) and Figure 19 use a separate, fixed ten-shot evaluation from the downstream shot-scaling experiment. It comprises eleven held-out tasks from Todd et al. (2024) and eight additional tasks from the Yin & Steinhardt (2025) evaluation: AG News, Capitalize First Letter, CoNLL Organization, National Parks, Park–Country, Person–Occupation, Animal versus Object (three items), Choose First of Five, English–German, Person–Instrument, CommonsenseQA, Capital Index, Capitalize Second Letter, Abstract Classification, French–English, and the Fruit, Capital-Par, Capital, and Shape binding tasks. These tasks are disjoint from those used to estimate FV scores.

Prompts interleave Q: inputs and A: outputs, separate demonstrations by a blank line, and end at the query’s A:. A document-start token is prepended. We draw ten demonstrations from each task’s demonstration split and evaluate 100 queries, generating up to five tokens greedily. The answer is extracted from the generated continuation, lowercased, stripped of punctuation and English articles, and split on whitespace. We report token-overlap F1 against the reference, averaged first over queries and then over the nineteen tasks. Prompt sampling is fixed per task across checkpoints. This F1 is distinct from the benchmark accuracies in Appendix B.3.

Checkpoint-specific summaries. In Figures 4 and 13, PS and FV, respectively, are summarized by averaging the top ⌈0.02H⌉ heads under each measure at each checkpoint, where H is the current number of head applications. At full depth, H = 480, 768, and 576 for 360M, 1.7B, and 3B, giving 10, 16, and 12 selected heads. These sets are reselected at every checkpoint and may have different membership.

The PS trajectories in Figure 4 use only trained checkpoints shared by all displayed models, every 5k steps for 360M and 2k steps for 1.7B and 3B. Display smoothing averages up to three adjacent shared checkpoints within 2000 steps of the center, including across growth boundaries. The first and final checkpoints remain raw. Immediate post-growth evaluations are shown only as raw hollow markers and never enter smoothing. In Figure 13, the 1.7B MIDAS post-growth ICL score at step 24,285 is approximately −0.72; its marker is placed on the lower boundary because it lies outside the displayed range. The regular checkpoint spacing of the dense and grown main runs does not resolve the earliest induction-head transition.

ICL-related training trajectories. Figure 13 extends the main-paper PS comparison to FV scores, few-shot F1 on nineteen held-out tasks, and the token-loss ICL-Score, across all three model sizes. In the 3B comparison, early growth events disrupt both PS and FV scores, followed by recovery during further training. The grown model reaches higher peak scores than the baseline, then finishes with higher PS but lower FV scores. The looped model finishes close to the baseline in PS and slightly above it in FV score. Few-shot F1 improves during training, with the grown model initially trailing and ultimately exceeding the baseline, while the looped model finishes close to it. Meanwhile, the ICL-Score declines for all models as early-token predictions improve more than later-token predictions. Both grown and looped models nevertheless finish with higher ICL-Score than the baseline.

![](images/d47aec04bd6591e1da69f819c6d087f3738c6d35e97d6b8e91e02b02f6ecdf6e.jpg)  
Figure 13: Training trajectories across model sizes. Columns show 360M, 1.7B, and 3B. Rows report PS and FV averaged over their respective checkpoint-specific top-2% head sets, nineteentask ten-shot F1, and the window-averaged ICL-Score (Equation (3)). Curves connect raw trained checkpoints, with regular spacing of 10,000 steps at 360M and 2000 steps at 1.7B and 3B, plus additional growth points. At 3B, all models use the shared regular grid. The first three rows use linear steps; the last uses logarithmic steps. Vertical rules mark growth and hollow markers show immediate post-growth evaluations. The downward marker in the 1.7B ICL-Score panel denotes MIDAS at −0.723, below the plotted range. Loop configurations are specified in Appendix A.3.

## E.2 FINAL-HEAD IDENTITIES, LINEAGES, AND LOCATIONS

Selection and ancestry. We select the final top-2% heads independently by PS and FV (10, 16, and 12 heads at 360M, 1.7B, and 3B) and trace these fixed identities backward through training. For grown models, we invert the actual layer-duplication maps: a head and its later copies share an ancestral trajectory until duplication, drawn only once. The 1.7B MIDAS branch uses its shared LIDAS trunk before step 24,285. Looped layers appear once per application. No particular seedhead pair is assumed to recur across sizes.

Checkpoint selection and smoothing. Trajectories begin at step 2000 for 1.7B and 3B; at 360M, they start at the first measured checkpoints: 10,000 for the baseline, 5000 for Loop, and 4722 for the grown models. Exact growth-step measurements are omitted except for this first 360M pre-growth checkpoint, the only measurement of the four-layer source stage. Immediate post-growth scores do not enter the curves. For each head, we apply a triangular weighted mean within each growth stage: at step t, a measurement at $t _ { j }$ receives weight proportional to $1 - | \log _ { 2 } ( t _ { j } / t ) | / 0 . 1 2 \bar { 5 }$ when $| \bar { \log _ { 2 } ( t _ { j } / t ) } | ^ { - } < 0 . 1 2 5$ , and zero otherwise. The first and last available measurements of each stage, including final scores, remain raw; smoothing never crosses a growth boundary.

Display conventions. Figures 5 and 14 pair the 3B PS and FV lineages with final all-head score maps. Cross-size trajectories appear in Figures 15 and 17, with the selected sets outlined in the corresponding maps (Figures 16 and 18). Curves are opaque, and color intensity follows the local smoothed score. Each figure uses a common score scale across models and sizes: 0–1 for PS, and a range retaining negative values for FV. Color does not encode final-head rank or PS/FV set overlap. Neither a high score nor repeated locations alone establish a head’s causal importance.

![](images/f6f5fc8c9b5326dd772565750185b310e2bae43c2f53c4ae642d9790d44729e0.jpg)  
Figure 14: FV heads recur at the grown or looped block size but emerge later than induction heads. FV counterpart of Figure 5 for the 3B models. Top: trajectories of the twelve final top-2% FV heads per model, selected independently of PS. In LIDAS, these heads trace to eight ancestors; two, L3H13 and L3H15, account for six final heads (two and four, respectively; orange). Bottom: final FV scores for all heads; black outlines mark the selected set. Orange in Loop 10 - 8×2 - 10 marks selected heads in the repeated block; other heads are blue. Scores share a common scale that retains negative values. Axes, growth and block guides, lineage tracing, and smoothing follow Figure 5 and this section.

Cross-size PS and FV lineages. Figures 15 and 17 independently select the final top-2% heads by PS and by FV, giving 10, 16, and 12 heads at 360M, 1.7B, and 3B. Each set is then traced by identity through the same inverse duplication maps. The corresponding all-head final score matrices appear in Figures 16 and 18; outlines mark exactly the heads selected for each trajectory plot. The 1.7B MIDAS branch uses its shared LIDAS trunk before step 24,285.

The appendix applies the same within-stage triangular log-step smoother with halfwidth 0.125, preserving stage endpoints and final values. Exact growth-step measurements are omitted except for the first 360M pre-growth checkpoint at step 4722: its explicit pre-growth stage identifies the four-layer source, and it is the only measured point in that stage. Immediate post-growth scores do not enter these lineage curves. The 360M dense and looped curves start at their first measured checkpoints, 10,000 and 5000. Shared ancestral segments are drawn once. PS uses a common 0–1 range; FV uses a common range across models and sizes that includes all displayed values. Color intensity follows the local score and does not encode model identity or PS/FV set overlap. No particular seed-head pair is assumed to recur across sizes.

![](images/58fd5d9e4d8c6d8cdbabb07c6ebcfdae510e090c6913bdd948501f8b30b85ea0.jpg)  
Figure 15: Trajectories of the final top-2% prefix-matching heads. Rows show 360M, 1.7B, and 3B; columns show Baseline, MIDAS, LIDAS, and Loop. Each panel follows the final 10, 16, or 12 heads ranked by PS, respectively, through their actual growth ancestry; sets are not reselected at intermediate checkpoints. Shared ancestral paths are drawn once, and vertical rules mark growth. Lines use the within-stage smoothing and boundary policy in Appendix E.2; step ticks marked k denote thousands. Scores share a common vertical range. There is no 3B MIDAS run. Loop configurations are specified in Appendix A.3.

No MIDAS run  
![](images/895652dfc778608e9c400429bda124d0e1f33f96869ce6a8ab02681569228e4e.jpg)

![](images/76379a59146ba6c80df18fb0d33ad6deac007ccbe1da7ee119f17f4a449d1a4e.jpg)

![](images/1cbd6c046d2d4bcdcbfdfa692cd7afc65d61a018eef73405210115f945bc769e.jpg)

![](images/320c53517ffd0186e6686d53dfa99c5f4f7ad7554ab36f030ebd40ebfa2e22a9.jpg)

![](images/899d7402d36f6b3184f36c30ce52b86ad3014e5a2d9daf39acc046075b66f22a.jpg)

![](images/5656e215d4457fd8d354a356e4cb5694737eba2305b6bbf4a2879dbf339b1bbf.jpg)

![](images/556bafa3ce5d22e5c7a0ee2d98da93e3b93a9aedd1ddffc767acf4a887a5c564.jpg)

![](images/786adf3c19a9970bdf1688d3820eaf7328cbf9830355cbee969ee37176c61da8.jpg)

![](images/3f7554f55f94416e50b9bf68ab27dca3dc0f06d8a30eb3e273abd8ca55cfd4bb.jpg)

![](images/ca3a8a93b258a3e8d4c31863b7d4b91adb2498afae1a8657318dba1336bc638f.jpg)

![](images/3fc09bb6f52f07b54961a3ec77317a01fdcc942a84cf71d992288471f38ce382.jpg)  
Figure 16: Final prefix-matching scores for every attention head. Rows show 360M, 1.7B, and 3B; columns show Baseline, MIDAS, LIDAS, and Loop. All panels share one PS color scale. Black outlines identify the final top-2% heads traced in Figure 15. Layer and head indices are zero-based; loop applications occupy separate layer positions. The empty 3B MIDAS slot denotes an unavailable model. Loop configurations are specified in Appendix A.3.

![](images/8bf164a2b5abe17982fb7b7d909f7c3432b0054bfa3400c4665dc841f52269d3.jpg)  
Figure 17: Trajectories of the final top-2% function-vector heads. Rows show 360M, 1.7B, and 3B; columns show Baseline, MIDAS, LIDAS, and Loop. Each panel follows the final 10, 16, or 12 heads ranked by FV, respectively, through their actual growth ancestry; sets are not reselected at intermediate checkpoints. Shared ancestral paths are drawn once, and vertical rules mark growth. Lines use the within-stage smoothing and boundary policy in Appendix E.2; step ticks marked k denote thousands. Scores share a common vertical range. There is no 3B MIDAS run. Loop configurations are specified in Appendix A.3.

![](images/f90bd936d1a528dc6067852e1bfdd5f338005a72c5682f6e9cd99cfd688f3ada.jpg)

![](images/6b308280de4ed6ab26b4d6f67ca28198bd81aac9a9b4c0bc592e29a92452cf15.jpg)

![](images/c543a9b0b4522a699f77d83e262c299c8593b5dcaadee629770ca6fd94988dc6.jpg)

![](images/83c68d89daa462d4d0f029b34f437246e4b1dc31e0746f31275acb0f14da9217.jpg)

![](images/ef6af054571bc4e0b631735712142903447bebf7b45c88b36081b0d83de83c1f.jpg)

![](images/e33b5d20d08f29ad6011cba15ff9fec86a9700fa848587255d0d5a2d9a249be8.jpg)

![](images/2e7f2b8727bbcc6048c640b50d613ca94f7198e8383c46e6900f461acf0a1ee5.jpg)

![](images/d0043c0b9b854970d2420f97e29c05c56c1abd42e7e7fcc2433b5d09f4a4933f.jpg)

![](images/c18a196b801b725aa772986ad383f5780ee1a59592563b16292aecf7e94e9e08.jpg)

![](images/977131425ead5ec3864d93775a85df1a044042a727402836e16db9eb739055ae.jpg)

![](images/7dad20bf1187eb71254f58841837d75e77cecff02bd2ca9743c5212b61f2d90a.jpg)  
Figure 18: Final function-vector scores for every attention head. Rows show 360M, 1.7B, and 3B; columns show Baseline, MIDAS, LIDAS, and Loop. All panels share one FV color scale. Black outlines identify the final top-2% heads traced in Figure 17. Layer and head indices are zero-based; loop applications occupy separate layer positions. Negative FV scores, where present, are retained. The empty 3B MIDAS slot denotes an unavailable model. Loop configurations are specified in Appendix A.3.

## E.3 PER-LAYER READOUTS

Figure 19 evaluates intermediate predictions at each exit depth l, with the horizontal axis normalized to $l / L$ . For each generated token, we run only the first l layers and decode their residual stream. The logit lens applies the full model’s final normalization and unembedding directly. The tuned lens (Belrose et al., 2023) first applies a learned affine residual translator $h \mapsto \bar { h } + W _ { l } \bar { h ^ { + } } b _ { l }$ . Translators are initialized at zero, so the initial lens is the identity, and the language model remains frozen.

Each model’s translators are trained on its own FineWeb-Edu pretraining source for 250 optimizer steps with $2 ^ { 1 8 }$ tokens per step, minimizing KL divergence from the full model’s next-token distribution to the translated intermediate prediction. Training uses SGD with Nesterov momentum 0.9, initial learning rate 0.1 with linear decay, weight decay $\mathrm { { 1 0 ^ { - 3 } } }$ , gradient clipping at norm 1, and bf16 autocast. Tokenization and document masking follow the corresponding pretraining setup. Translator index l decodes the residual entering layer l (equivalently, after l completed layers); the full-depth exit uses the original readout without a translator.

We run the same nineteen-task, ten-shot F1 evaluation at each exit depth. An identity-translator (zero-lens) control reproduces the logit lens, and the full-depth exit reproduces the original model. These probes test how accessible predictions are to a fixed or learned readout; they do not by themselves identify which layers causally compute the prediction.

![](images/39a3272a6bf961259c0fa4a48572671060ad3d3cefad53e76ed53cd3206ec8c4.jpg)  
Figure 19: Intermediate few-shot readouts across model sizes. Columns show 360M, 1.7B, and 3B; rows use the logit and tuned lenses. Each curve runs the nineteen-task, ten-shot F1 evaluation after the indicated fraction of layers. The tuned lens is trained separately for each frozen language model by next-token distillation (Appendix E.3). These readouts measure accessibility of predictions, not the causal importance of a layer. Loop configurations are specified in Appendix A.3.

## F ATTENTION-HEAD ABLATIONS

## F.1 HEAD SELECTION AND MEAN ABLATION

We mean-ablate fractions $f \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 , 0 . 0 7 , 0 . 0 9 , 0 . 1 5 , 0 . 2 0 \}$ of final-checkpoint head applications. The combined figure uses exclusive PS and FV rankings; additional sweeps also include non-exclusive and random rankings. Exclusive induction rankings first exclude the top ⌊0.02H⌋ FV heads; exclusive FV rankings analogously exclude the top induction heads. The number ablated remains ⌊fH⌋, measured relative to the full set before exclusion. Joint rankings use the geometric mean of PS and FV after clipping each score at zero and dividing by its maximum. A selected head’s output slice is replaced by a mean activation immediately before the attention output projection. In looped models, selection and intervention act on individual applications, so the ablated fraction is not a fraction of unique parameters.

## F.2 MEASUREMENTS AND PLOTTED SUMMARIES

Few-shot F1. The replacement is the clean last-token head activation averaged over the 29 FV training tasks. It is applied at the last position of the prompt and each subsequent generation forward pass. We repeat the nineteen-task ten-shot F1 evaluation of Appendix E.1. Figure 6(a) and Figure 20 report changes from each model’s unablated F1, while Figure 22 reports absolute F1.

Token loss. We measure token-level NLL on 10,000 FineWeb-Edu documents, replacing selected heads’ outputs with their natural-text mean activations at all positions. Let $\bar { \ell } _ { j }$ be the documentaveraged NLL at prediction position j (Equation (1)) and $\delta _ { j } = \bar { \ell } _ { j } ^ { ( \mathrm { a b l a t e d } ) } - \bar { \ell } _ { j } ^ { ( \mathrm { c l e a n } ) }$ its increase from the same model without ablation on the same documents. Position-wise plots report $\delta _ { j }$ . We also summarize token loss using the windowed ICL-Score in Equation (3); Figure 6(b) and Figure 20 report its change:

$$
\Delta S _ { \mathrm { I C L } } = S _ { \mathrm { I C L } } ^ { ( \mathrm { a b l a t e d } ) } - S _ { \mathrm { I C L } } ^ { ( \mathrm { c l e a n } ) } .\tag{7}
$$

Figure 22 reports the absolute ICL-Score.

Position curves average $\delta _ { j }$ in logarithmic bins (24 per decade) for display on a base-2 axis. In Figure 6(c), all six curves additionally receive Gaussian-weighted local linear smoothing in log<sub>2</sub> position with bandwidth $\sigma = 0 . 2$ . Windowed scores use raw position losses before any binning or display smoothing.

![](images/35ddfb545298b2add4c275709f3afd45df422130a7237f093cf20ca5968cb837.jpg)  
Figure 20: Exclusive head ablations across model sizes. Columns show 360M, 1.7B, and 3B. Top: change in nineteen-task ten-shot F1. Bottom: change in windowed ICL-Score (Equation (7)). Solid lines ablate induction-ranked heads after excluding the top-2% FV heads; dashed lines reverse these roles. Markers distinguish models. Fractions are 1%, 3%, 5%, 7%, 9%, 15%, and 20%; zero denotes the corresponding unablated result. Loop configurations are specified in Appendix A.3.

![](images/5e01da188671264d2962444605e19ccb3ca563fe5affd5182f976b49b2746a90.jpg)  
Figure 21: Position-wise head ablation at higher fractions. Columns show 360M, 1.7B, and 3B; row pairs show exclusive induction and FV ablation at 7%, 9%, 15%, and 20%. Curves give the mean NLL increase on 10,000 FineWeb-Edu documents, with log-binned positions on a base-2 axis. Each induction/FV pair within a size shares an uncropped vertical range. MIDAS is unavailable at 3B. See Appendix F. Loop configurations are specified in Appendix A.3.

![](images/c6c5686c07675e37b19db25abab9ffc910ec57afdc3753a5d0eff0128f38b80e.jpg)  
Figure 22: Head ablation across model sizes. Row pairs show 360M, 1.7B, and 3B, each with absolute few-shot F1 above the windowed ICL-Score (Equation (3)); columns show dense, MIDAS, LIDAS, and looped models. Dashed rankings exclude the other measure’s top-2% heads. Gray bands show the mean ± standard error over random head sets: three for F1 and sixteen for the ICL-Score at every model size. Dotted lines give unablated values. Fractions count applications in looped models. Panel scales vary; the empty 3B MIDAS column reflects the missing model. See Appendix F. Loop configurations are specified in Appendix A.3.