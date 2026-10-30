# Factorized Representations of Concurrent Temporal Variables in Pretrained Language Models

Xinhe Zhang

xinhezhang@g.harvard.edu

Arnau Marin-Llobet

amarinllobet@seas.harvard.edu

School of Engineering and Applied Sciences, Harvard University, Cambridge, USA.

## Abstract

Language models often need to maintain multiple contextual variables that evolve indepen dently within a document. Prior work has identified structured representations of individual temporal or contextual quantities, but their joint organization remains less understood. We study this question using controlled documents with independently phased language-switch and topic-change timers. Across ten pretrained checkpoints, both timers are linearly decodable from the same hidden states. Removing a timer-associated centroid plane substantially reduces access to its associated variable while having little efect on the concurrent timer, although the planes are nonorthogonal and contain cross-timer information. Separate timer contributions also explain most reproducible variation among joint-state centroids. These signatures persist from 3 to 51 timer values and across Qwen3 checkpoints from 0.6B to 14B. Schedule position afects continuation probabilities, but the tested centroid displacement produces no consistent timer-specific output efect. Together, these results support approximate factorization based on selective accessibility and approximately additive joint-state geometry in overlapping subspaces.

Keywords: representational geometry; factorized representations; language models; temporal state; subspace interventions; interpretability

## 1. Introduction

A language model processing a document may need to maintain several properties of the context that evolve independently. In dialogue, the speaker may change while the topic remains fixed; in multilingual text, the language may switch independently of the subject matter. At each position, the model must represent the current values of these variables while updating them according to diferent histories. We study how such concurrent, context-defined variables are jointly organized in pretrained hidden representations.

Prior work has identified spatial and temporal coordinates (Gurnee and Tegmark, 2024), contextual binding vectors (Feng and Steinhardt, 2024), and a countdown subcircuit (Dunefsky et al., 2026). These studies establish structured representations of individual tempora or contextual quantities. Much less is known about multiple variables that evolve independently within the same document. Simultaneous linear decodability does not determine whether two variables rely on shared or variable-specific directions, and subspace overlap does not determine whether the information in that overlap is required for decoding.

We use controlled documents with independently phased language switches and topic changes. Each variable is represented by the number of sentences since its most recent event. The two schedules produce balanced combinations of timer values while preserving ordinary sentence content. For each timer, we estimate a low-dimensional subspace from its class centroids. We then remove this subspace, retrain the linear decoder, and measure how much timer information remains accessible. Comparing removal of a timer’s own plane with removal of the concurrent timer’s plane provides a direct measure of selectivity.

Across ten pretrained checkpoints, removing each timer’s centroid plane eliminates substantially more of its associated decoding signal than removing the concurrent timer’s plane. The planes are nonorthogonal and contain information about both timers, while separate timer contributions explain most joint-centroid variation that generalizes across document splits. These results characterize approximate factorization in pretrained representations through selective accessibility and approximately additive joint-state geometry, with overlapping factor-associated subspaces.

We further test generalization across timer horizon and model size. Across 3 to 51 timer values, decoding remains high, efective centroid rank grows slowly, and self-ablation remains much larger than cross-ablation, although cross-ablation increases with horizon. Within Qwen3 from 0.6B to 14B, the same signatures persist without a monotonic parameter trend. A recency-matched continuation assay shows that schedule position afects model outputs. Centroid displacements reliably alter decoded timer states, but the tested language-timer displacement from value 1 to value 2 produces no consistent timer-specific change in continuation probabilities across models.

## 2. Experimental Framework

## 2.1. Concurrent event-recency variables

We construct controlled documents with independently phased language switches and topic changes. English and German sentences are drawn from a hand-written bank of parallel propositions on two topics, house and field. No timer labels, event markers, prefixes, or separators are added. Language switches occur every three sentences and topic changes every four, with independent document-level ofsets.

The language timer $y _ { \mathrm { l a n g } } ( t ) \in \{ 0 , 1 , 2 \}$ records sentences since the most recent switch. The topic analysis retains phases 0–2 from its four-phase schedule. Independent phases and coprime periods yield balanced, independent analyzed labels (Figure 1A).

The primary corpus contains 100 documents of 48 sentences. Proposition identity is sampled independently within the active topic. Additional analyses use matched stochastic schedules and larger corpora (Appendix A).

## 2.2. Hidden-state representations and linear readout

We evaluate ten pretrained checkpoints from eight model families, including base and instruction-tuned models: Llama-3.2-1B, Mistral-7B-v0.3, Qwen3-1.7B, Qwen3-1.7B-Base, Qwen3-4B, pythia-1.4b, gemma-2-2b, SmolLM2-1.7B, Phi-3-mini, and OLMo-2-1B. Model parameters remain fixed. For each sentence, we extract the sentence-final residual-stream state $h _ { t }$ from layer $\lfloor L / 2 \rfloor$ , where L is the number of transformer blocks. Table 3 lists the exact checkpoints and analyzed layer indices.

We subtract each document’s mean state and residualize each hidden dimension against $\{ 1 , t , t ^ { 2 } \}$ in sentence index t. These nuisance-normalization steps use full-document and corpus-level statistics and precede cross-fitting. Documents are then divided into disjoint roles for fitting a 100-component PCA representation and timer planes, training ridge decoders, and evaluation. Within each split, PCA is fitted only on the plane-estimation documents and applied to the other document sets.

## 2.3. Timer-associated subspaces

For timer value $v \in \{ 0 , 1 , 2 \}$ , let $\mu _ { v }$ denote the plane-estimation-fold mean for examples with that value. After subtracting their common mean, the three class centroids span at most two dimensions. We denote this rank-2 centroid plane by $S _ { e }$ for event timer e.

Projection outside the centroid plane eliminates between-class mean diferences on the plane-estimation fold. Cross-fitting evaluates the transfer of this geometry to disjoint documents, and post-ablation decoding measures the timer information that remains linearly accessible after decoder retraining. Plane stability is evaluated across disjoint document subsets and against shufled labels. All reported planes are mapped back to the model’s state space before geometric comparison or intervention.

To measure how much linearly accessible information remains outside a subspace $S _ { ; }$ , we project

$$
h _ { t } ^ { \prime } = ( I - P _ { S } ) h _ { t } ,
$$

where $P _ { S }$ is the orthogonal projector onto S, and retrain the decoder. Retraining measures information that remains linearly accessible after projection. One document subset fits the plane, a second trains intact and post-ablation decoders, and a third is evaluated; roles rotate across splits. Ablation is reported as the fraction of above-chance decoding signal removed.

## 2.4. Selective ablation, information overlap, and joint additivity

Selective ablation. For timers A and B, self-ablation removes $S _ { A }$ before decoding A;   
cross-ablation removes $S _ { A }$ before decoding B. Decoders are retrained in both conditions.

Information overlap. Restricted decoding trains a decoder using only the coordinates within each plane. Principal angles measure geometric alignment between planes. Shufledlabel planes provide controls for both quantities.

Joint additivity. For the nine analyzed combinations $( a , b )$ of the two timer values, we fit

$$
\mu ( a , b ) = m + u _ { a } + v _ { b } + \epsilon _ { a b } ,
$$

where m is a common mean, $u _ { a }$ and $v _ { b }$ are timer contributions, and $\epsilon _ { a b }$ is their interaction. Interaction is the fraction of joint-centroid variance attributable to $\epsilon _ { a b }$ . We calibrate it with finite-sample surrogates whose population centroids are exactly additive and with shufled timer labels. Cross-document fits compare additive predictions with a saturated nine-centroid model.

## 2.5. Horizon, parameter-range, and behavioral extensions

For $H \in \{ 2 , 5 , 1 0 , 2 0 , 5 0 \}$ , schedules with periods $H + 1$ and $H + 2$ yield analyzed timer values $0 , \ldots , H$ . Each horizon contains 200 documents generated from an expanded parallel proposition bank. Ridge regression predicts timer value, with performance evaluated by held-out Spearman correlation, and the centroid rank is selected to explain 90% of centroid variance on the plane-estimation fold. Self-, cross-, and random-subspace ablations use the same selected rank under three-way cross-fitting.

We also evaluate five Qwen3 sizes from 0.6B to 14B under a common quantization and analysis protocol. The behavioral assay is evaluated on the same ten pretrained checkpoints as the primary analysis and scores the summed log-probability diference between parallel English and German candidate continuations under period-3 and period-4 schedules at matched language recency. Language-timer centroid displacement is applied at the sentencefinal token and compared with matched-norm concurrent-timer and isotropic displacements (Appendix D).

## 3. Concurrent Temporal Variables Exhibit Selective Ablation Efects

## 3.1. Both temporal variables are simultaneously represented

We first characterize the linear accessibility of both timers from the same hidden states. We also repeat the analysis after excluding current-event sentences to distinguish elapsed-state information from direct detection of a switch or topic change.

Across all ten models, linear decoders predict both timers above the 33.3% chance level. Balanced accuracy is 84–96% for language and 40–53% for topic; the topic signal is weaker but consistently above chance.

After excluding current-event states, value-1 versus value-2 accuracy is 89.3% for language and 59.8% for topic, against 50% chance. Both timers remain decodable among past-event states.

## 3.2. Timer-associated planes selectively capture accessible timer information

For each timer, we compare intact decoding with decoding after removing either its own plane or the concurrent timer’s plane. Decoders are retrained after projection, and plane estimation, decoder training, and evaluation use disjoint document roles, so the comparison measures how much linearly accessible information remains outside each independently estimated plane.

Cross-fitted self-ablation of the language plane reduces accuracy from 0.897 to 0.426, removing 84% of above-chance signal; topic-plane ablation leaves accuracy at 0.888 and removes 1.6%. For the topic timer, self-ablation reduces accuracy from 0.486 to 0.399 and removes 57%, whereas language-plane ablation leaves accuracy at 0.481 and removes 2.8% (Figure 1B–C). The self–cross separation occurs in each model despite variation in intact decoding accuracy.

## 3.3. Selective ablation under additional conditions

In the controlled-overlap calibration, cross-ablation increases as timer-specific signal is transferred into the concurrent plane. Similar self–cross separation is observed with phasematched document splits, aperiodic schedules, alternative plane estimators and PCA dimensions, and DailyDialog (Li et al., 2017). Mean per-cell cross/self ratios are 3.40% for aperiodic schedules and 3.51% for matched periodic schedules; on DailyDialog, mean crossablation damage is 12.6% of self-ablation damage. History-only analyses show the same separation. Appendices B and D provide the corresponding results.

![](images/942ab105786c8180532bd72ed6547892f86ae02168df79d814244decd56d08fa.jpg)

![](images/dcdf70903756c74062b41b7ffad69ade99ff3dcfd9af6f7bec5f5f8486babac7.jpg)

![](images/c62120e1d0019b76607b91d2d723fa0866c7cc3910e83fa0bb738d77fef53056.jpg)  
Figure 1: Selective ablation of concurrent timer representations. A, Independently phased language and topic schedules. B–C, Cross-fitted decoding after subspace removal and decoder retraining. Points represent models; bars indicate means.

## 4. Concurrent Temporal Variables Form an Approximately Factorized Joint Representation

To characterize the joint representation beyond selective ablation, we examine how the timer planes overlap and whether the nine joint timer states combine approximately additively. We measure overlap using restricted decoding and principal angles, and additivity from the joint-centroid geometry.

## 4.1. Timer-associated planes are not information-disjoint

Restricted decoding from a rank-2 plane shows that each timer is most informative in its own plane but is also decodable from the concurrent plane. Language decoding is 0.440 above chance from its own plane and 0.166 from the topic plane; topic decoding is 0.107 above chance from its own plane and 0.043 from the language plane. In both directions, concurrent-plane decoding exceeds the shufled-label control (Figure 2A–B). Restricted decoding therefore shows that each timer plane contains information about the concurrent timer.

The smaller principal angle between timer planes averages $5 3 . 9 ^ { \circ }$ , compared with $7 9 . 7 ^ { \circ }$ for unrelated rank-2 subspaces in the same 100-dimensional representation. The planes are therefore nonorthogonal as well as information-overlapping. Together with the ablation result, this shows that representational overlap and selective accessibility can coexist.

## 4.2. Shared geometry contains common and event-specific structure

Whitening within-class scatter reduces both plane alignment and cross-timer decoding, indicating that shared covariance contributes to the observed overlap. The most closely aligned direction carries approximately 47% of the full language-plane decoding signal. Removing this direction increases the angle between the remaining plane directions from 54.8<sup>◦</sup> to 78.3<sup>◦</sup>, revealing more nearly orthogonal residual structure (Appendix C).

A  
![](images/a568b392d722451023aded73bfb0adc283dabb31a7cec53c52557d1c4aebabc4.jpg)

B  
![](images/6593148252975bd32ca664556cf7e26c0716c3a7b4d24b27261b120e1f12bfae.jpg)

![](images/1053c6d9ae27535f58a431c8a7367af80fe7d8221bc08ffa8eb07c19dcd9ca69.jpg)  
Figure 2: Information overlap and approximately additive joint geometry. A– B, Decoding restricted to timer-associated planes. C, Joint-centroid interaction residuals across models. Pale bars indicate the mean ± bootstrap standard error.

Across six event types and four models, a rank-2 plane fitted after pooling across event types recovers 78.5% of the decoding signal from event-specific planes. Event-specific residual directions outperform matched random subspaces in all 24 model–event combinations. Together, these analyses identify shared event-recency structure alongside event-specific geometry (Appendix D).

## 4.3. The joint timer geometry is approximately additive

We next test whether changing one timer produces a similar centroid displacement at each value of the other timer. Departures from this additive structure are quantified by the interaction residual defined in Section 2.4.

Interaction accounts for 5.2–16.8% of joint-centroid variance, with a mean of 10.1%. Exactly additive finite-sample surrogates yield 5.2% mean interaction, and shufled labels yield approximately 50% (Figure 2C). The joint geometry therefore contains measurable non-additivity, while additive timer contributions account for most centroid variation.

Across document halves, the additive model explains 81.0% of held-out joint-centroid variance, compared with 78.2% for a saturated nine-centroid model. The saturated model exceeds the additive model in three of ten checkpoints, by at most 0.007 explained variance. The predominantly additive structure thus generalizes across document splits.

The two timers difer in geometric reliability. Language-timer displacements are more reproducible across document splits and more stable across concurrent-timer values than topic-timer displacements, consistent with the stronger language-timer decoding signal (Appendix C).

## 4.4. Timer-plane displacements selectively alter decoded timer states

The additive model associates a displacement with each change in timer value. We apply these displacements to held-out states to test whether they alter the corresponding decoded variable while preserving the concurrent timer.

For a held-out state with timer value i, we add the within-plane displacement from centroid $\mu _ { i }$ toward $\mu _ { j }$ . A full displacement produces the target decoded value in 94.2% of cases, compared with 6.6% for an equal-norm isotropic displacement. At half displacement, the decoded target timer changes in 47.6% of cases, while the decoded concurrent timer remains unchanged in 93.2%. All six ordered source-to-target transitions contribute to these aggregates. These results concern linear readouts of the intervened hidden state. Section 5 tests whether one such intervention, the language-timer displacement from value 1 to value 2, produces a corresponding change in continuation probabilities.

## 5. Generalization and Behavioral Relevance

We next examine whether the identified representational organization persists across longer timer horizons and model sizes, and whether it is reflected in model outputs.

## 5.1. Extension to longer timer horizons

With three timer values, the centered class centroids necessarily span at most two dimensions. We therefore repeat the analysis with 3, 6, 11, 21, and 51 timer values, selecting the smallest rank that explains 90% of centroid variance on the plane-estimation fold.

<table><tr><td>H</td><td>Values</td><td> $\rho$ </td><td>Rank</td><td>Self</td><td>Cross</td><td>Random</td><td>Cross/self</td></tr><tr><td>2</td><td>3</td><td>0.91</td><td>1.75</td><td>84.9%</td><td>0.15%</td><td>0.18%</td><td>0.18%</td></tr><tr><td>5</td><td>6</td><td>0.89</td><td>2.50</td><td>75.3%</td><td>0.44%</td><td>0.31%</td><td>0.56%</td></tr><tr><td>10</td><td>11</td><td>0.87</td><td>3.25</td><td>79.9%</td><td>2.05%</td><td>0.53%</td><td>2.58%</td></tr><tr><td>20</td><td>21</td><td>0.87</td><td>3.48</td><td>79.8%</td><td>2.96%</td><td>0.69%</td><td>3.71%</td></tr><tr><td>50</td><td>51</td><td>0.86</td><td>3.98</td><td>84.6%</td><td>4.86%</td><td>0.76%</td><td>5.71%</td></tr></table>

Table 1: Cross-fitted horizon sweep across four models. Rank is the mean selected dimension; Self, Cross, and Random are percentage reductions in intact Spearman correlation. See Appendix D.3 for aggregation details.

Across four models, decoding remains high as the number of timer values increases, while selected centroid rank grows slowly (Table 1). Self-ablation remains much larger than cross-ablation throughout the sweep. Cross-ablation increases with horizon and exceeds matched-rank random ablation at longer horizons, indicating gradually weaker selectivity.

## 5.2. Persistence across model sizes

To examine variation with model size within a single model family, we repeat the analysis across Qwen3 checkpoints at 0.6B, 1.7B, 4B, 8B, and 14B while holding the corpus, layerselection rule, preprocessing, and analysis protocol fixed. The five 4-bit checkpoints form the primary comparison, with bfloat16 runs at the three overlapping sizes as a precision control.

Using the full-corpus interaction estimator at each model size, interaction residuals remain between 12.1% and 15.9%, well below shufled values near 50%. Language-timer split-half plane angles remain between $1 7 ^ { \circ }$ and $2 7 ^ { \circ }$ , compared with 84<sup>◦</sup>–86<sup>◦</sup> for shufled labels (Table 11); decoding and selective ablation also persist at every size. Neither interaction residual nor language-plane split-half angle varies monotonically with parameter count; their Spearman correlations with parameter count are +0.10 and +0.20, respectively. The representational signatures therefore persist across model sizes without a monotonic scaling trend.

## 5.3. Relation to continuation behavior

We next examine the relation between timer representations and the model’s continuation distribution. We first test whether continuation probabilities depend on schedule position by comparing period-3 and period-4 language-switch schedules at matched immediate language recency. Parallel English and German candidate sentences express the same proposition and topic, and a diference-in-diferences contrast isolates the efect of schedule position beyond immediate recency. This contrast is positive in all ten pretrained models, ranging from 1.78 to 4.14 in candidate-sentence log odds, while a randomly initialized Qwen3-1.7B has the opposite sign (Table 12). Continuation probabilities therefore depend on schedule position beyond immediate language recency.

We then test whether the identified timer displacement can selectively alter this behavior. The language-timer displacement from value 1 to value 2 is applied at the middle-layer sentence boundary, at strengths 0.5 and 1.0, and compared with matched-norm concurrenttimer and isotropic displacements. The target displacement does not produce a consistently larger continuation efect than the matched concurrent-timer displacement. At full strength, target-minus-concurrent efects range from −0.033 to +0.029, average −0.0046, and are positive in five of ten models (Table 13); the same qualitative pattern holds at half strength.

Schedule position is reflected in the model’s continuation distribution, whereas the tested centroid displacement changes decoded timer state without a consistent timer-specific continuation efect.

## 6. Related Work

## 6.1. Factorized and structured representations

Transformers trained on controlled stochastic processes can represent predictive belief-state geometry (Shai et al., 2024), and models trained on independent synthetic factors can develop approximately orthogonal factor subspaces (Shai et al., 2026). Thil et al. (2026) partition JEPA latents into designated progression and content subspaces and evaluate the partition through ablation. These studies examine factors specified by the training process or architecture. We analyze independently evolving contextual variables in fixed pretrained models and characterize their organization through selective accessibility and additive jointstate geometry, including nonorthogonal, information-overlapping subspaces.

Fel et al. (2026) model activations as sparse additive mixtures of low-dimensional concept manifolds and introduce block-sparse featurizers for discovering this structure without concept labels. Our analyses use experimentally specified variables to measure how their information is distributed across subspaces and how their joint states combine.

## 6.2. Temporal state and representational intervention

Language models encode spatial and temporal coordinates (Gurnee and Tegmark, 2024), distance to a target length (Dunefsky et al., 2026), repeated-token count (Venkatesh, 2026), and contextual entity bindings (Feng and Steinhardt, 2024). Elapsed wall-clock time can also be supplied through a trained conditioning channel (Siavoshian et al., 2026). These studies principally address individual temporal quantities or variables introduced and later queried within a context. The present experiments examine two temporal variables that are repeatedly updated in the same hidden states.

Concurrent working-memory variables can exhibit representational overlap (Xiong et al., 2026) and behavioral interference (Xiong et al., 2026; Wang and Sun, 2026). Causal abstraction and interchange-intervention methods evaluate whether representational interventions alter a target while preserving other attributes (Huang et al., 2024; Wu et al., 2023); control tasks calibrate probe capacity (Hewitt and Liang, 2019). Our analyses distinguish information contained within a timer plane, information remaining after its removal, and output changes under centroid displacement.

## 7. Discussion and Conclusion

The results support an operational notion of approximate factorization based on selective accessibility and approximately additive joint-state geometry (Park et al., 2024; Fel et al., 2026). The timer subspaces are nonorthogonal and information-overlapping, and each contains information about both variables. Nevertheless, removing each timer plane disproportionately impairs decoding of its associated timer, while additive timer contributions explain most joint-centroid variation that generalizes across document splits. Together, restricted decoding, ablation, and joint-state geometry show that factorized organization can coexist with distributed and overlapping representations.

This organization persists as the number of timer values increases from 3 to 51 and across Qwen3 checkpoints from 0.6B to 14B. Efective centroid rank grows slowly with horizon, while cross-ablation increases but remains substantially weaker than self-ablation. Recency-matched schedule position systematically afects continuation probabilities, and centroid displacements reliably alter decoded timer states, but the tested value-1-to-value-2 intervention produces no consistent timer-specific change in model output. Thus, selective manipulation of decoded state does not necessarily translate into selective downstream control.

These results motivate further study of when factorized representational structure supports controllable computation. Extending this analysis to richer concurrent variables, more naturalistic contexts, and interventions distributed across layers or multiple tokens may clarify which representational structures contribute to downstream computation. Furthermore, bridging these activation-level timer geometries to underlying counting subcircuits (Dunefsky et al., 2026) and weight-sparse transformer mechanisms (Gao et al., 2026; Marin-Llobet and Heimersheim, 2026) could reveal how these states are physically implemented. More broadly, relating factorization to intervention, causal influence, and task-dependent control could help distinguish variables that are accessible in hidden states from those that exert measurable influence on model behavior.

## Acknowledgements

Arnau Marin-Llobet is supported by Coeficient Giving and the RCC-Harvard Fellowship.

## References

Jacob Dunefsky, Wes Gurnee, and Emmanuel Ameisen. A shared subcircuit lets LLMs count down across tasks. arXiv preprint arXiv:2607.12279, 2026.

Thomas Fel, Matthew Kowal, Mozes Jacobs, Dron Hazra, Usha Bhalla, Lee Sharkey, Lucius Bushnaq, Satchel Grant, Tal Haklay, Thomas Icard, Can Rager, Michael Pearce, Daniel Wurgaft, Aiden Swann, Fenil Doshi, Siddharth Boppana, Curt Tigges, Nick Cammarata, Thomas Serre, Vasudev Shyam, Owen Lewis, Thomas McGrath, Jack Merullo, Ekdeep Singh Lubana, and Atticus Geiger. Structuring sparsity: Block-sparse featurizers capture visual concept manifolds. arXiv preprint arXiv:2606.25234, 2026.

Jiahai Feng and Jacob Steinhardt. How do language models bind entities in context? In The Twelfth International Conference on Learning Representations, 2024.

Leo Gao, Achyuta Rajaram, Jacob Coxon, Soham V. Govande, Bowen Baker, and Daniel P. Mossing. Weight-sparse transformers have interpretable circuits. In Forty-third International Conference on Machine Learning, 2026.

Wes Gurnee and Max Tegmark. Language models represent space and time. In The Twelfth International Conference on Learning Representations, 2024.

John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019.

Jing Huang, Zhengxuan Wu, Christopher Potts, Mor Geva, and Atticus Geiger. RAVEL: Evaluating interpretability methods on disentangling language model representations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024.

Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. DailyDialog: A manually labelled multi-turn dialogue dataset. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2017.

Arnau Marin-Llobet and Stefan Heimersheim. Individual parameters in weight-sparse transformers appear interpretable. In Mechanistic Interpretability Workshop at ICML 2026, 2026.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Forty-first International Conference on Machine Learning, 2024.

Adam Shai, Loren Amdahl-Culleton, Casper L. Christensen, Henry R. Bigelow, Fernando Rosas, Alexander Boyd, Eric A. Alt, Kyle J. Ray, and Paul M. Riechers. Transformers learn factored representations. In Forty-third International Conference on Machine Learning, 2026.

Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, and Paul M. Riechers. Transformers represent belief state geometry in their residual stream. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Sam Siavoshian, Omar Ramadan, Amir K. Saeed, Benjamin A. Johnson, Amin Mohamed El-Amin Diab, and Benjamin M. Rodriguez. ChronoState: Hidden elapsed-time conditioning for temporal-state action selection in frozen-backbone language models. arXiv preprint arXiv:2608.09124, 2026.

Lucas Thil, Jesse Read, Rim Kaddah, and Guillaume Doquet. Subspace-decomposed JEPAs: Disentangling progression and content in latent world models. arXiv preprint arXiv:2605.31111, 2026.

Sohan Venkatesh. Repeated-token counting reveals a dissociation between representations and outputs. arXiv preprint arXiv:2605.09239, 2026.

Chupei Wang and Jiaqiu Vince Sun. Unable to forget: Proactive interference reveals working memory limits in LLMs beyond context length. In Trustworthy Agentic AI Workshop at AAAI 2026, 2026.

Zhengxuan Wu, Atticus Geiger, Thomas Icard, Christopher Potts, and Noah D. Goodman. Interpretability at scale: Identifying causal mechanisms in Alpaca. In The Thirty-seventh Annual Conference on Neural Information Processing Systems, 2023.

Hua-Dong Xiong, Li Ji-An, Jiaqi Huang, Robert C. Wilson, Kwonjoon Lee, and Xue-Xin Wei. In-context superposition: human-like working memory interference in large language models. In Third Conference on Language Modeling, 2026.

## Appendix A. Experimental Details

Controlled corpus construction. The generator uses a hand-written bank of eight propositions for each of two topics, house and field, with parallel realizations in English, German, and Spanish. The primary corpus uses the 32 English and German sentence strings. Each sentence contains four whitespace-delimited words. Each document concatenates 48 sentences with single spaces, without added labels or markers.

In family A, language alternates between English and German every three sentences, while topic alternates every four sentences, with independent random document-level ofsets. At each sentence, a proposition is sampled independently from the eight within the active topic. Documents share the proposition bank, and splits are document-disjoint. A matched family B uses stochastic language switches for additional schedule analyses. Each family contains 100 documents; primary analyses use family A.

Topic phase takes values 0, 1, 2, and 3. Primary three-class topic analyses retain phases 0–2, corresponding to 0, 1, or 2 sentences since the topic change.

<table><tr><td>Sent.</td><td>Text</td><td>Lang.</td><td>Topic</td><td>Ylang</td><td>Ytopic</td></tr><tr><td>0</td><td>The lamp is bright.</td><td>EN</td><td>house</td><td>1</td><td>3†</td></tr><tr><td>1</td><td>The mountain is high.</td><td>EN</td><td>field</td><td>2</td><td>0</td></tr><tr><td>2</td><td>Der Fluss ist breit.</td><td>DE</td><td>field</td><td>0</td><td>1</td></tr><tr><td>3</td><td>Der Stein ist grau.</td><td>DE</td><td>field</td><td>1</td><td>2</td></tr><tr><td>4</td><td>Der Wald ist dunkel.</td><td>DE</td><td>field</td><td>2</td><td>3†</td></tr><tr><td>5</td><td>The chair is heavy.</td><td>EN</td><td>house</td><td>0</td><td>0</td></tr><tr><td>6</td><td>The lamp is bright.</td><td>EN</td><td>house</td><td>1</td><td>1</td></tr><tr><td>7</td><td>The window is open.</td><td>EN</td><td>house</td><td>2</td><td>2</td></tr></table>

Table 2: Excerpt from a family-A document with language and topic timer labels. <sup>†</sup>Topic-phase-3 rows are excluded from primary three-class topic analyses.

Representation extraction. Each full document is tokenized and processed in one forward pass, returning hidden states at every layer. For each sentence, we select the last nonzero-width token whose tokenizer ofset ends at or before the sentence-final character boundary recorded by the generator. We analyze the residual-stream state at layer $\lfloor L / 2 \rfloor$ ， where L is the number of transformer blocks. In the primary ten-checkpoint analysis, models run in bfloat16; extracted states are stored in float32 and converted to float64 for analysis.

Preprocessing. Before splitting documents for analysis, we subtract each document’s mean sentence state and remove positional trends by regressing each hidden dimension on the corpus-wide design matrix $\{ 1 , t , t ^ { 2 } \}$ , where t is sentence index. These steps use fulldocument and corpus-level statistics. Subsequent PCA, timer-plane estimation, decoder training, and evaluation follow the document-disjoint cross-fitting procedure. PCA is fitted on the designated fit documents, and the fitted basis is applied to decoder-training and evaluation documents.

Timer-associated planes and decoding. For each timer value $v \in \{ 0 , 1 , 2 \}$ , we compute the mean representation $\mu _ { v }$ on the plane-estimation fold. After subtracting their common mean, the three centroids span at most two dimensions; their nonzero right singular vectors define the timer’s centroid plane. Linear timer decoders are ridge classifiers fitted to one-hot targets with regularization $\lambda = 1 0 ^ { - 2 } n$ , where n is the number of decoder-training examples. We report balanced accuracy for classification.

To ablate a subspace S, we project states as $h ^ { \prime } = ( I { - } P _ { S } ) .$ h and retrain the decoder on the projected states. Post-ablation accuracy measures timer information that remains linearly accessible outside S. For the primary cross-fitted ablation summary, we average intact and post-ablation accuracies across models and calculate the percentage of above-chance signal removed as $1 0 0 \frac { \bar { a } _ { \mathrm { i n t a c t } } - \bar { a } _ { \mathrm { a b l a t e d } } } { \bar { a } _ { \mathrm { i n t a c t } } - a _ { \mathrm { c h a n c e } } }$

Cross-fitted ablation. The 100 family-A documents are repeatedly partitioned into three disjoint subsets: one fits PCA and both timer planes, one trains intact and postablation decoders, and one evaluates them. Rotating these roles yields 36 splits per model and ablation direction. In-sample calibration estimates the plane and trains the decoder on the same subset, using the same number of plane-estimation documents as cross-fitting.

Additive interaction controls. To estimate finite-sample residuals under exact additivity, we combine each joint state’s empirical residuals with its fitted additive mean, resample documents, and re-estimate the nine centroids. This surrogate has exactly additive population means and preserves empirical residual variability. For held-out prediction, we fit the additive model on half the documents and evaluate it against joint centroids from the other half, alongside mean-only and saturated nine-centroid models.

History-only decoding. History-only decoding retains target timer values 1 and 2, giving a binary task with 50% chance accuracy. Timer planes and decoders are re-estimated on this subset.

<table><tr><td>Paper name</td><td>Checkpoint</td><td>Layers</td><td>Analyzed</td></tr><tr><td>Llama-3.2-1B</td><td>meta-1lama/Llama-3.2-1B</td><td>16</td><td>8</td></tr><tr><td>Mistral-7B-v0.3</td><td>mistralai/Mistral-7B-vO.3</td><td>32</td><td>16</td></tr><tr><td>Qwen3-1.7B</td><td>Qwen/Qwen3-1.7B</td><td>28</td><td>14</td></tr><tr><td>Qwen3-1.7B-Base</td><td>Qwen/Qwen3-1.7B-Base</td><td>28</td><td>14</td></tr><tr><td>Qwen3-4B</td><td>Qwen/Qwen3-4B</td><td>36</td><td>18</td></tr><tr><td>pythia-1.4b</td><td>EleutherAI/pythia-1.4b</td><td>24</td><td>12</td></tr><tr><td>gemma-2-2b</td><td>google/gemma-2-2b</td><td>26</td><td>13</td></tr><tr><td>SmolLM2-1.7B</td><td>HuggingFaceTB/SmolLM2-1.7B</td><td>24</td><td>12</td></tr><tr><td>Phi-3-mini</td><td>microsoft/Phi-3-mini-4k-instruct</td><td>32</td><td>16</td></tr><tr><td>OLMo-2-1B</td><td>allenai/0LMo-2-0425-1B</td><td>16</td><td>8</td></tr></table>

Table 3: Checkpoints, transformer-block counts, and analyzed layer indices.

Models. All models use SDPA attention and tokenizers loaded from their corresponding checkpoint identifiers.

## Appendix B. Additional Evidence for Selective Ablation

## B.1. Cross-fitted ablation

We report balanced accuracy, the mean recall across the three timer classes, averaged over splits and models. Chance performance is 1/3. In-sample removal of each timer’s own plane

yields balanced accuracies of 0.332 for language and 0.334 for topic, both approximately at chance. Table 4 reports the corresponding cross-fitted results for each model.
<table><tr><td rowspan="2">Model</td><td colspan="3">Language timer</td><td colspan="3">Topic timer</td></tr><tr><td>Intact</td><td>Own</td><td>Other</td><td>Intact</td><td>Own</td><td>Other</td></tr><tr><td>Llama-3.2-1B</td><td>0.891</td><td>0.423</td><td>0.886</td><td>0.460</td><td>0.390</td><td>0.459</td></tr><tr><td>Mistral-7B</td><td>0.917</td><td>0.433</td><td>0.909</td><td>0.513</td><td>0.403</td><td>0.506</td></tr><tr><td>Qwen3-1.7B</td><td>0.878</td><td>0.421</td><td>0.867</td><td>0.502</td><td>0.404</td><td>0.498</td></tr><tr><td>Qwen3-1.7B-Base</td><td>0.941</td><td>0.431</td><td>0.938</td><td>0.531</td><td>0.412</td><td>0.528</td></tr><tr><td>Qwen3-4B</td><td>0.864</td><td>0.428</td><td>0.851</td><td>0.522</td><td>0.414</td><td>0.512</td></tr><tr><td>pythia-1.4b</td><td>0.839</td><td>0.412</td><td>0.832</td><td>0.400</td><td>0.372</td><td>0.399</td></tr><tr><td>gemma-2-2b</td><td>0.884</td><td>0.430</td><td>0.862</td><td>0.520</td><td>0.411</td><td>0.508</td></tr><tr><td>SmolLM2-1.7B</td><td>0.963</td><td>0.437</td><td>0.962</td><td>0.452</td><td>0.391</td><td>0.452</td></tr><tr><td>Phi-3-mini</td><td>0.857</td><td>0.416</td><td>0.835</td><td>0.511</td><td>0.404</td><td>0.505</td></tr><tr><td>OLMo-2-1B</td><td>0.937</td><td>0.428</td><td>0.938</td><td>0.445</td><td>0.387</td><td>0.446</td></tr><tr><td>Mean</td><td>0.897</td><td>0.426</td><td>0.888</td><td>0.486</td><td>0.399</td><td>0.481</td></tr></table>

Table 4: Cross-fitted balanced accuracy before ablation (Intact), after own-plane removal (Own), and after concurrent-timer-plane removal (Other).

## B.2. Controls for document splitting

Phase-matched splits assign the three document roles separately within each of the 12 combinations of language-switch and topic-change phase. Within-document splits use disjoint randomly assigned sentences from the same documents for plane estimation and decoder fitting.
<table><tr><td></td><td>Own, random</td><td>Own, phase-matched</td><td>Own, within-document</td><td>Other, random</td></tr><tr><td>Language</td><td>84.5%</td><td>84.0%</td><td>81.4%</td><td>1.6%</td></tr><tr><td>Topic</td><td>58.4%</td><td>63.0%</td><td>53.6%</td><td>2.3%</td></tr></table>

Table 5: Percentage of above-chance decoding signal removed under each split condition, averaged over per-model removal fractions from a separate eight-draw experiment. Own and Other denote removal of the target timer’s plane and the concurrent timer’s plane, respectively.

## B.3. Past-event decoding without current-event states

We decode timer values 1 versus 2 using the history-only procedure described above. Table 6 reports balanced accuracy averaged across splits and models, with chance at 0.5, and the percentage of above-chance signal removed by each ablation.

## B.4. Calibration with imposed overlap

We transfer a fraction α of one timer’s discriminative component into the concurrent timer’s plane while keeping labels fixed. We measure the resulting decrease in balanced accuracy

<table><tr><td>Timer</td><td>Intact</td><td>Own, cross-fitted</td><td>Other, cross-fitted</td><td>Own removed</td><td>Other removed</td></tr><tr><td>Language</td><td>0.893</td><td>0.576</td><td>0.886</td><td>81%</td><td>1.7%</td></tr><tr><td>Topic</td><td>0.598</td><td>0.553</td><td>0.591</td><td>45%</td><td>6.3%</td></tr></table>

Table 6: Balanced accuracy and percentage of above-chance signal removed in history-only decoding. Own and Other denote ablation of the target timer’s plane and the concurrent timer’s plane, respectively.

after removing the concurrent timer’s plane (cross-ablation) or the target timer’s own plane (self-ablation).
<table><tr><td>Transferred fraction α</td><td>Cross-ablation</td><td>Self-ablation</td><td>Above-chance accuracy</td></tr><tr><td>0.00</td><td>0.6</td><td>16.5</td><td>+0.164</td></tr><tr><td>0.25</td><td>1.1</td><td>11.2</td><td>+0.109</td></tr><tr><td>0.50</td><td>4.4</td><td>8.1</td><td>+0.081</td></tr><tr><td>0.75</td><td>10.4</td><td>8.5</td><td>+0.084</td></tr><tr><td>1.00</td><td>11.0</td><td>10.8</td><td>+0.110</td></tr></table>

Table 7: Ablation efects under imposed overlap, reported as balanced-accuracy decreases in percentage points. The final column reports intact balanced accuracy minus chance.

## B.5. Aperiodic timing and analysis variants

Aperiodic schedules independently sample inter-event gaps from {3, 4, 5}. The periodic control uses the same event markers; both conditions are subsampled to equalize the nine joint timer states. The mean ratio of cross-ablation to self-ablation, measured by their respective decreases in balanced accuracy, is 3.40% for aperiodic schedules and 3.51% for the matched periodic control.

Across alternative rank-2 plane estimators, cross-ablation is 1.12–5.08% of self-ablation. With 64–150 retained principal components, self-ablation is 29–88 times larger than crossablation.

## Appendix C. Additional Evidence for Shared Information and Joint Geometry

## C.1. Restricted decoding and principal angles

Restricted decoding trains and evaluates a decoder using representations projected onto a rank-2 plane. Table 8 reports language-timer balanced accuracy minus chance (1/3) and the smaller principal angle between the language and topic planes.

The smaller principal angle averages 53.9<sup>◦</sup> across models, compared with $7 9 . 7 ^ { \circ }$ for unrelated rank-2 planes.

<table><tr><td>Model</td><td>Own plane</td><td>Concurrent plane</td><td>Placebo</td><td>Angle</td></tr><tr><td>SmolLM2-1.7B</td><td>+0.596</td><td>+0.136</td><td>+0.164</td><td>66.6°</td></tr><tr><td>Qwen3-1.7B-Base</td><td>+0.537</td><td>+0.259</td><td>+0.149</td><td>53.4°</td></tr><tr><td>OLMo-2-1B</td><td>+0.528</td><td>+0.188</td><td>+0.146</td><td>52.4°</td></tr><tr><td>Mistral-7B-v0.3</td><td>+0.515</td><td>+0.206</td><td>+0.166</td><td>53.1°</td></tr><tr><td>Llama-3.2-1B</td><td>+0.451</td><td>+0.153</td><td>+0.117</td><td>55.4°</td></tr><tr><td>gemma-2-2b</td><td>+0.438</td><td>+0.217</td><td>+0.105</td><td>41.6°</td></tr><tr><td>Qwen3-1.7B</td><td>+0.379</td><td>+0.126</td><td>+0.071</td><td>56.3°</td></tr><tr><td>Qwen3-4B</td><td>+0.360</td><td>+0.134</td><td>+0.078</td><td>51.9°</td></tr><tr><td>Phi-3-mini</td><td>+0.326</td><td>+0.144</td><td>+0.061</td><td>46.2°</td></tr><tr><td>pythia-1.4b</td><td>+0.270</td><td>+0.095</td><td>+0.053</td><td>62.4°</td></tr></table>

Table 8: Above-chance language-timer decoding from the language plane (Own), topic plane (Concurrent), and placebo plane, with the smaller principal angle between the two timer planes.

## C.2. Whitening and the shared direction

Whitening within-class scatter before plane estimation increases the timer-plane angle to the range observed for unrelated subspaces and reduces cross-timer restricted decoding.

The most closely aligned direction carries 5.1 times the variance of a random direction. Language-timer balanced accuracy along this direction is 0.206 above chance, approximately 47% of the above-chance accuracy in the full language plane. Removing this direction increases the angle between the remaining plane directions from 54.8<sup>◦</sup> to 78.3<sup>◦</sup>.

## C.3. Joint additivity and cross-document prediction

Table 9 reports the interaction residual defined in Section 2.4, its finite-sample additive baseline, and held-out variance explained under the cross-document prediction procedure.

Shufling timer labels raises the interaction residual to approximately 50%. The meanonly model yields a mean held-out explained variance of −0.005.

## C.4. Conditional stability of timer displacements

For each timer transition, we compute joint-centroid displacements at each value of the concurrent timer and measure their cosine similarity. We also measure split-half reproducibility of the same transition at a fixed concurrent-timer value and repeat the analysis with shufled timer labels.

## Appendix D. Additional Generalization and Behavioral Results

## D.1. Human-written dialogue corpus

We analyze 3,000 DailyDialog conversations (Li et al., 2017) with two timers: utterances since the most recent occurrence of the most common dialogue act, and utterances since the most recent utterance labeled with one of the six named non-neutral emotions. Event definitions are selected before extracting model states.

<table><tr><td rowspan="2">Model</td><td colspan="2">Interaction residual</td><td colspan="2">Held-out variance explained</td></tr><tr><td>Observed</td><td>Additive floor</td><td>Additive</td><td>Saturated</td></tr><tr><td>Llama-3.2-1B</td><td>9.1%</td><td>4.3%</td><td>0.839</td><td>0.833</td></tr><tr><td>Mistral-7B</td><td>6.8%</td><td>3.4%</td><td>0.876</td><td>0.852</td></tr><tr><td>Qwen3-1.7B</td><td>12.8%</td><td>6.0%</td><td>0.758</td><td>0.761</td></tr><tr><td>Qwen3-1.7B-Base</td><td>7.5%</td><td>3.1%</td><td>0.868</td><td>0.874</td></tr><tr><td>Qwen3-4B</td><td>12.8%</td><td>6.1%</td><td>0.767</td><td>0.753</td></tr><tr><td>pythia-1.4b</td><td>16.8%</td><td>11.2%</td><td>0.664</td><td>0.511</td></tr><tr><td>gemma-2-2b</td><td>10.4%</td><td>4.4%</td><td>0.813</td><td>0.816</td></tr><tr><td>SmolLM2-1.7B</td><td>5.2%</td><td>2.9%</td><td>0.897</td><td>0.865</td></tr><tr><td>Phi-3-mini</td><td>13.3%</td><td>7.3%</td><td>0.749</td><td>0.717</td></tr><tr><td>OLMo-2-1B</td><td>6.6%</td><td>3.5%</td><td>0.871</td><td>0.838</td></tr><tr><td>Mean</td><td>10.1%</td><td>5.2%</td><td>0.810</td><td>0.782</td></tr></table>

Table 9: Joint-centroid interaction residuals and held-out variance explained by additive and saturated nine-centroid models. The additive floor estimates the finite-sample residual under exactly additive population means.

<table><tr><td>Timer</td><td>Across</td><td>Split-half</td><td></td><td>Across / split-half</td><td>Shuffled</td></tr><tr><td>Language</td><td>0.855</td><td>0.889</td><td></td><td>96%</td><td> $+ 0 . 0 0 5$ </td></tr><tr><td>Topic</td><td>0.331</td><td>0.587</td><td></td><td>54%</td><td>+0.005</td></tr></table>

Table 10: Mean cosine similarity of timer-transition displacements across concurrent-timer values (Across), between document halves at a fixed concurrent-timer value (Split-half), and under shufled labels. Across / split-half reports the mean of per-model ratios.

Across Llama-3.2-1B, Qwen3-1.7B, and pythia-1.4b, both timers are decodable at 28.0– 31.4 balanced-accuracy points above chance. Mean balanced-accuracy decreases are 30.1 points for self-ablation and 3.79 points for cross-ablation, giving a cross/self ratio of 12.6%.

## D.2. More event streams and shared event-recency geometry

We analyze six event realizations—language, speaker tag, topic tag, punctuation, register phrase, and symbol prefix—in six single-timer corpora across four models. Within each split, the six training corpora share a fitted 100-component PCA basis. We estimate a pooled rank-2 timer plane across event realizations and a separate rank-2 plane for each realization (Figure 3).

![](images/a105de321517827eace84545d36f66d363fb1f30e2d0f59fa6898f5a19708319.jpg)

![](images/d4c99e2f5bf58836370e7070b7b015dc02b2e6d288c6118085972a9385e1e926.jpg)

C  
![](images/3fc572f501900c80bdb5faba5ebfa6856e35e294da359cd3da67a2e0f6e80786.jpg)  
Figure 3: Shared and event-specific event-recency geometry. A, Restricted decoding, including event-specific residual planes formed by removing the pooled component. B, Principal angles between pooled and event-specific planes. C, Decoding after plane removal. Pale intervals show variation across four models.

## D.3. Longer timer horizons

For horizon H, language switches have period H + 1 and topic changes have period $H + 2$ Excluding the final topic phase gives both timers analyzed values $0 , \ldots , H$ . We use $H \in$ {2, 5, 10, 20, 50} and 200 documents per horizon, with 24 propositions per topic in parallel English and German realizations. Empirical association between timer labels remains below 0.06 in magnitude at every horizon.

Ridge regressors predict normalized timer value. On the plane-estimation fold, we decompose the centered timer centroids by SVD and select the smallest rank explaining 90% of their variance. Under three-way cross-fitting, the selected subspace is removed before decoder training and evaluation on separate folds. Self-, cross-, and isotropic-random ablations use the same selected rank.

Table 1 reports the horizon sweep. For each model, removal is calculated as $1 0 0 ( \rho _ { \mathrm { i n t a c t } } -$ $\rho _ { \mathrm { a b l a t e d } } ) / \rho _ { \mathrm { i n t a c t } }$ using split-averaged correlations, then averaged across models. Cross/self is the mean of per-model removal ratios.

Split-half subspace angles range from $1 0 . 6 ^ { \circ }$ to $1 4 . 5 ^ { \circ }$ . Shufled-label reference angles decrease from $7 8 . 4 ^ { \circ }$ to $5 0 . 5 ^ { \circ }$ as selected rank increases.

## D.4. Within-family parameter range

We analyze Qwen3 models with 0.6B, 1.7B, 4B, 8B, and 14B parameters, holding the corpus, layer-selection rule, preprocessing, projection dimension, splits, and decoder fixed. All five models use 4-bit weights; the three smaller models are also evaluated in bfloat16. Interaction residuals use the full-corpus estimator, and split-half plane angles are averaged over eight splits.
<table><tr><td>Model</td><td>Interaction ± SEM</td><td>Shuffled</td><td>Split-half angle (lang./topic)</td></tr><tr><td>Qwen3-0.6B</td><td> $1 2 . 1 \pm 1 . 7 \%$ </td><td>50.4%</td><td> $1 7 . 2 ^ { \circ } / 5 8 . 7 ^ { \circ }$ </td></tr><tr><td>Qwen3-1.7B</td><td> $1 4 . 6 \pm 2 . 1 \%$ </td><td>50.6%</td><td> $2 2 . 2 ^ { \circ } / 5 1 . 7 ^ { \circ }$ </td></tr><tr><td>Qwen3-4B</td><td> $1 5 . 9 \pm 2 . 4 \%$ </td><td>49.9%</td><td> $2 7 . 0 ^ { \circ } / 5 1 . 7 ^ { \circ }$ </td></tr><tr><td>Qwen3-8B</td><td> $1 3 . 3 \pm 2 . 1 \%$ </td><td>50.1%</td><td> $1 9 . 7 ^ { \circ } / 5 3 . 4 ^ { \circ }$ </td></tr><tr><td>Qwen3-14B</td><td> $1 2 . 7 \pm 2 . 0 \%$ </td><td>50.3%</td><td> $2 0 . 2 ^ { \circ } / 4 1 . 1 ^ { \circ }$ </td></tr></table>

Table 11: Interaction residuals and plane reproducibility across Qwen3 model sizes. Uncertainty is one bootstrap standard error estimated from 40 document resamples; Shufled reports the shufled-label interaction residual. Shufled-label plane angles range from $8 4 ^ { \circ }$ to 86<sup>◦</sup>.

At the three overlapping sizes, 4-bit and bfloat16 interaction estimates difer by at most 0.9 percentage points and language-plane angles by at most 1.2<sup>◦</sup>.

## D.5. Behavioral continuation assay and output intervention

For context x, let $s ( x )$ be the summed log-probability diference between other-language and current-language next-sentence candidates expressing the same proposition and topic. We compare language-switch periods $k = 3$ and $k = 4$ at matched timer value e:

$$
A _ { \mathrm { d u e } } = s ( k = 3 , e = 2 ) - s ( k = 4 , e = 2 ) , \qquad A _ { \mathrm { c o n t r o l } } = s ( k = 3 , e = 1 ) - s ( k = 4 , e = 1 ) .
$$

We report $D = A _ { \mathrm { d u e } } { - } A _ { \mathrm { c o n t r o l } }$ over 140 documents per schedule for each of the ten pretrained checkpoints.

We estimate the language-timer displacement from value 1 to value 2 on 70 documents and apply it at 490 held-out sentence boundaries per pretrained checkpoint. Interventions

<table><tr><td>Model</td><td> $\boldsymbol { A } _ { \mathrm { d u e } }$ </td><td>Acontrol</td><td>D± SEM</td></tr><tr><td>pythia-1.4b</td><td>3.301</td><td>-0.837</td><td> $4 . 1 3 8 \pm 0 . 6 5 8$ </td></tr><tr><td>Qwen3-1.7B</td><td>2.147</td><td>-0.741</td><td> $2 . 8 8 8 \pm 0 . 5 5 6$ </td></tr><tr><td>Phi-3-mini</td><td>2.094</td><td>-0.750</td><td> $2 . 8 4 4 \pm 0 . 4 5 6$ </td></tr><tr><td>OLMo-2-1B</td><td>1.997</td><td>-0.767</td><td> $2 . 7 6 4 \pm 0 . 5 6 9$ </td></tr><tr><td>Llama-3.2-1B</td><td>2.262</td><td>-0.280</td><td> $2 . 5 4 2 \pm 0 . 4 1 8$ </td></tr><tr><td>SmolLM2-1.7B</td><td>1.962</td><td>-0.449</td><td> $2 . 4 1 1 \pm 0 . 5 4 6$ </td></tr><tr><td>Qwen3-4B</td><td>2.256</td><td>0.006</td><td> $2 . 2 5 0 \pm 0 . 4 9 7$ </td></tr><tr><td>gemma-2-2b</td><td>1.224</td><td>-0.754</td><td> $1 . 9 7 8 \pm 0 . 4 8 7$ </td></tr><tr><td>Qwen3-1.7B-Base</td><td>1.572</td><td>-0.355</td><td> $1 . 9 2 7 \pm 0 . 3 5 8$ </td></tr><tr><td>Mistral-7B-v0.3</td><td>1.567</td><td>-0.210</td><td> $1 . 7 7 6 \pm 0 . 3 5 4$ </td></tr><tr><td>Qwen3-1.7B-random</td><td>-2.427</td><td>1.362</td><td> $- 3 . 7 8 9 \pm 0 . 8 6 3$ </td></tr></table>

Table 12: Recency-matched continuation contrasts. Uncertainty is one standard error from 2000 document-bootstrap resamples. The final row reports a randomly initialized Qwen3-1.7B control.

modify the sentence-final token’s middle-layer state at strengths 0.5 and 1.0; concurrenttimer and isotropic controls are norm-matched. Table 13 reports full-strength changes in other-language versus current-language continuation log odds.
<table><tr><td>Model</td><td>Target</td><td>Concurrent</td><td>Random</td><td>Target – concurrent</td></tr><tr><td>Phi-3-mini</td><td>0.0249</td><td>0.0231</td><td>0.0010</td><td>0.0018</td></tr><tr><td>gemma-2-2b</td><td>0.0243</td><td>0.0066</td><td>-0.0070</td><td>0.0177</td></tr><tr><td>OLMo-2-1B</td><td>0.0173</td><td>0.0499</td><td>-0.0015</td><td>-0.0326</td></tr><tr><td>pythia-1.4b</td><td>0.0126</td><td>-0.0164</td><td>0.0037</td><td>0.0290</td></tr><tr><td>Mistral-7B-v0.3</td><td>0.0046</td><td>0.0215</td><td>-0.0036</td><td>-0.0169</td></tr><tr><td>SmolLM2-1.7B</td><td>0.0021</td><td>0.0324</td><td>-0.1014</td><td>-0.0303</td></tr><tr><td>Qwen3-4B</td><td>-0.0017</td><td>-0.0125</td><td>-0.0018</td><td>0.0108</td></tr><tr><td>Llama-3.2-1B</td><td>-0.0105</td><td>0.0166</td><td>-0.0150</td><td>-0.0271</td></tr><tr><td>Qwen3-1.7B-Base</td><td>-0.0113</td><td>-0.0182</td><td>-0.0163</td><td>0.0069</td></tr><tr><td>Qwen3-1.7B</td><td>-0.0142</td><td>-0.0087</td><td>-0.0077</td><td>-0.0055</td></tr><tr><td>Mean</td><td>0.0048</td><td>0.0094</td><td>-0.0150</td><td>-0.0046</td></tr></table>

Table 13: Mean continuation log-odds change at strength 1.0 for language-timer (Target), concurrent-timer, and isotropic (Random) interventions.