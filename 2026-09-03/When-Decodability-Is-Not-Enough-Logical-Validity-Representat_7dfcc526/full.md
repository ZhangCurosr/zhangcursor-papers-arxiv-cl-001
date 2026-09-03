# When Decodability Is Not Enough: Logical Validity Representations, Behavioral Dissociation, and Causal Tests in Language Models

Smitha Muthya Sudheendra University of Minnesota, Twin Cities muthy009@umn.edu

Jaideep Srivastava University of Minnesota, Twin Cities srivasta@umn.edu

## Abstract

Large language models can look capable of logical reasoning, but correct or incorrect answers alone tell us little about what the model represents internally. We study logical verification in five open-weight transformer models using matched valid–invalid premise–claim pairs that vary across inference families, semantic domains, templates, and difficulty levels. Despite near-chance behavioral performance, logical validity is often almost perfectly decodable from hidden states and remains strongly decodable under held-out templates, domains, and inference families. Validity also remains highly decodable on behaviorally incorrect examples in the conditions where correctness-conditioned evaluation is well defined. At the same time, exhaustive leave-one-out tests reveal clear limits to this generalization, and interventions along probe-derived validity directions have only weak, nonspecific effects compared with random controls. Our results suggest that representing validity, expressing it in behavior, and using it causally are distinct. Validity related information can be strongly decodable from a model’s hidden states without being reliably expressed in its output.

## 1 Introduction

Large language models are often evaluated on tasks described as requiring reasoning, but behavioral success alone tells us little about the internal computations behind an answer. A correct prediction may reflect the relevant logical relation, but it may also arise from lexical or semantic regularities. Likewise, an incorrect prediction does not necessarily mean that the information needed for the correct decision is absent from the model’s hidden states. Distinguishing what a model outputs from what it represents, and how those representations are used, is therefore central to interpretability.

We study this distinction through logical verification: given a set of premises and a candidate claim, the model must determine whether the claim follows. This setting is useful because validity is defined by the relation between premises and claim, and gold labels can be assigned deterministically. It therefore supports controlled valid–invalid contrasts while varying surface form, semantic content, and inference structure.

We separate three questions: whether the model expresses the correct validity judgment, whether validity-related information is linearly accessible in its hidden states, and whether the corresponding probe-derived direction causally influences the final decision. To study them, we construct 800 examples organized into 400 matched valid–invalid pairs spanning five inference families, five semantic domains, and three difficulty levels. We evaluate five open-weight transformers using behavioral likelihoods and layer-wise linear probes, test generalization to held-out templates, domains, and inference families, and use matched-pair, correctness-conditioned, lexical, metadata, and shuffled label controls. On three models, we additionally intervene along the probe-derived validity direction and compare its effect with norm-matched random directions.

Behavioral verification remains at or near chance across all five models, often with strong answerlabel preferences. Hidden states tell a different story: validity is almost perfectly linearly decodable in-distribution and remains strongly decodable under held-out templates and many domain and inference-family shifts. Exhaustive leave-one-out evaluation nevertheless reveals clear exceptions, including a recurring failure to transfer to unseen syllogistic reasoning. Generalization is therefore broad but not uniform.

The dissociation also persists on behavioral errors. Where correctness-conditioned evaluation is well defined, validity remains highly decodable on incorrectly answered examples. Across all primary splits, probes also preserve the ordering between valid and invalid members of matched pairs, even when global calibration shifts. Behavioral errors therefore do not generally coincide with the absence of linearly accessible validity information. High decodability, however, does not imply simple causal control. Interventions along the probe-derived direction produce only small and inconsistent changes in output margins, comparable to or smaller than random orthogonal controls. Thus, the linear direction that separates valid from invalid examples is not itself a strong control variable for verification behavior. Taken together, the results distinguish

behavioral expression $\neq$ representational accessibility $\neq$ causal use.

We study this separation in a controlled logical-verification setting, testing how well validity decodability generalizes and whether it persists when the model answers incorrectly. More broadly, the results argue for treating behavioral performance, decodability, and causal influence as distinct forms of evidence when studying reasoning in language models.

## 2 Related Work

Reasoning and logical evaluation. Behavioral benchmarks such as BIG-Bench, HELM, and BIG-Bench Hard have documented substantial progress in language-model reasoning while also revealing strong sensitivity to task formulation and prompting [14, 7, 15]. More targeted benchmarks, including FOLIO, PrOntoQA-OOD, LogicBench, and Multi-LogiEval, evaluate deductive reasoning and systematic generalization across logical structures and distribution shifts [3, 13, 10, 11]. Prompting methods such as chain-of-thought can improve observable reasoning behavior [17, 5], but generated explanations need not faithfully reflect the computation that produced an answer [16]. Our focus i therefore narrower: we study verification as a controlled relation between premises and a candidate claim, and distinguish output behavior from internal representation.

Internal representations of truth and validity. Prior work shows that high-level variables such as factual truth can be recovered from language-model activations [2, 8], while representation engineering studies whether such directions can also be manipulated [18]. Logical validity is related but distinct: it depends on the relation between premises and a conclusion rather than on the truth of the conclusion alone. Our matched-pair construction is designed to isolate this relational property. Bertolazzi et al. [1] show that logical validity and semantic plausibility are linearly represented in syllogistic reasoning and can be causally influenced through activation steering. We instead ask how broadly validity decodability generalizes across surface form, semantic domain, and inference structure, and whether validity remains accessible when the model’s behavioral judgment is incorrect.

Probing and causal interpretation. High probe accuracy does not imply that the decoded feature is used by the model. Probe performance can reflect nuisance variables or task-format artifacts, and recent work shows that apparently strong reasoning-related separability can disappear after such confounds are controlled [12]. We therefore use matched valid–invalid pairs, held-out template/domain/family evaluations, shuffled-label controls, and metadata-only baselines. Throughout, we interpret probe AUROC as evidence of linear accessibility, not as evidence of an abstract reasoning mechanism. Mechanistic interpretability further motivates testing whether decoded structure is behaviorally consequential. Prior work has recovered internal state variables through probing and intervention in settings such as Othello-GPT and multi-step reasoning [6, 9, 4]. We similarly complement layer-wise probing with activation interventions along probe-derived validity directions and compare them with norm-matched random controls.

Rather than asking only whether logical validity can be decoded from hidden states, we ask how meaningful that representation is. Does it generalize beyond the conditions used to train the probe? Is it still present when the model gives the wrong answer? And does intervening on the corresponding direction actually change the model’s behavior? These questions let us separate what is represented internally from what is expressed in the output and what is used causally in the model’s decision.

## 3 Controlled Verification Dataset

We construct a controlled dataset to study how language models represent logical validity. Rather than aiming for broad benchmark coverage, the dataset is designed to isolate verification from confounds such as factual recall, search, and explanation generation. Each example consists of a set of premise and a candidate claim, and the model must decide whether the claim follows from the premises. Because validity depends on the relation between the premises and the claim, this setting allows us to study relational information rather than claim plausibility alone.

## 3.1 Task and Dataset Structure

Each example is represented as

$$
x _ { i } = ( P _ { i } , c _ { i } , y _ { i } , f _ { i } , d _ { i } , t _ { i } , \delta _ { i } ) ,
$$

where $P _ { i }$ is the set of premises, $c _ { i }$ the candidate claim, and $y _ { i } \in \{ 0 , 1 \}$ the validity label. The remaining variables denote inference family, semantic domain, template family, and difficulty level.

Models choose between VALID and INVALID. The dataset contains five inference families: syllogism, transitivity, set inclusion, causal chain, and permission logic, instantiated across five domains: nonce, spatial, social, biological, and legal-policy. Using the same inference structures across different domains lets us test whether validity representations depend on particular semantic content.

We also define three difficulty levels:

• Difficulty 1: direct one-step verification;

• Difficulty 2: two-step inference chains;

• Difficulty 3: two-step chains with an irrelevant distractor premise.

These levels vary the amount of relational integration required while leaving the output task unchanged.

## 3.2 Matched Valid–Invalid Pairs

The dataset is organized into matched valid–invalid pairs. Within each pair, the premises, inference family, domain, template family, and difficulty are held fixed; only the candidate claim changes.

For a matched pair $( x _ { i } ^ { + } , x _ { i } ^ { - } )$

$$
P _ { i } ^ { + } = P _ { i } ^ { - } , \quad f _ { i } ^ { + } = f _ { i } ^ { - } , \quad d _ { i } ^ { + } = d _ { i } ^ { - } , \quad t _ { i } ^ { + } = t _ { i } ^ { - } , \quad \delta _ { i } ^ { + } = \delta _ { i } ^ { - } ,
$$

while

$$
y _ { i } ^ { + } = 1 , \qquad y _ { i } ^ { - } = 0 .
$$

This construction reduces label-correlated differences in the surrounding context and gives us a direct within-context test: does the model representation distinguish a valid claim from an invalid one when everything else is held constant? Matched pairs are also kept together during splitting and bootstrap resampling.

## 3.3 Dataset Composition

The final dataset contains 800 examples, equally divided between valid and invalid inferences. Each of the five inference families contributes 160 examples, and each of the five semantic domains contributes 160 examples. The three difficulty levels contain 200 one-step examples, 400 two-step examples, and 200 two-step examples with distractor premises. The strict template split contains 600 training examples and 200 held-out examples.

Table 1: Composition of the controlled verification dataset.
<table><tr><td>Property</td><td>Count</td></tr><tr><td>Total examples</td><td>800</td></tr><tr><td>Valid / Invalid</td><td>400 / 400</td></tr><tr><td>Inference families</td><td>5</td></tr><tr><td>Semantic domains</td><td>5</td></tr><tr><td>Examples per family Examples per domain</td><td>160</td></tr><tr><td>Difficulty 1 / 2 / 3</td><td>160 200 / 400 / 200</td></tr><tr><td>Template train / test</td><td>600 / 200</td></tr></table>

## 3.4 Generalization Splits

Random held-out performance shows whether validity is decodable in-distribution, but it does not tell us whether the probe depends on familiar wording, semantic content, or inference-specific structure. We therefore evaluate four complementary train–test conditions.

The random split serves as the in-distribution reference, with training and test examples drawn from the same overall mixture. In the template-held-out split, complete template families are excluded from training to test whether decoding transfers to unseen surface forms. The domain-held-out split evaluates the probe on a semantic domain that is absent during training, while the inferencefamily-held-out split withholds an entire reasoning family and tests transfer to a new inference structure.

These conditions are not intended to form a strict hierarchy of difficulty. Rather, they probe different kinds of generalization: from ordinary in-distribution prediction to changes in surface form, semantic content, and inference structure.

## 3.5 Controls and Scope

All examples are generated deterministically from known inference rules, so gold labels are assigned by construction rather than by a language model. We retain metadata describing inference family, domain, template family, difficulty, distractor status, and split membership for later subgroup and control analyses.

The dataset is intentionally diagnostic rather than comprehensive. Its purpose is to distinguish several questions: whether models express validity in their outputs, whether validity-related information is accessible in hidden states, whether it generalizes beyond the probe-training distribution, and whether it remains accessible when the final answer is wrong.

## 4 Experimental Setup

We evaluate logical verification at three levels: output behavior, linear accessibility in hidden states, and the causal effect of the probe-derived validity direction. Full implementation and statistical details are provided in Appendix B and Appendix C.

## 4.1 Models and Behavioral Evaluation

We evaluate Pythia-1.4B, Pythia-2.8B, SmolLM3-3B, Llama-3.2-3B (hereafter, Llama-3.2), and Mistral-7B. Causal interventions are run on Pythia-2.8B, Llama-3.2, and Mistral-7B. All models are evaluated on the same 800 examples and split definitions. Behavior is measured from the likelihood assigned to VALID and INVALID. For label scores $s _ { V } ( x )$ and $s _ { I } ( x )$ , we predict

$$
\hat { y } ( x ) = \mathbb { I } [ s _ { V } ( x ) > s _ { I } ( x ) ] , \qquad M ( x ) = s _ { V } ( x ) - s _ { I } ( x ) ,
$$

where $M ( x )$ is the continuous output margin. We report accuracy, the fraction of predictions assigned VALID, and margin AUROC. Answer-label tokenization is audited separately for each model; detail are given in Appendix B.

## 4.2 Hidden-State Probing

For every transformer block, we extract the final prompt-token hidden state and fit an ℓ -regularized logistic-regression probe to predict gold validity. Feature standardization, hyperparameter selection, and layer selection use only the probe-training partition, with matched pairs kept together during validation. The reported layer is

$$
\ell ^ { * } = \arg \operatorname* { m a x } _ { \ell } \mathrm { A U R O C } _ { \ell } ^ { \mathrm { v a l } } ,
$$

after which the selected probe is evaluated once on the held-out test set.

We evaluate probes under random, template-held-out, domain-held-out, and inference-family-heldout splits, and additionally run exhaustive leave-one-domain-out and leave-one-family-out analyses. Matched-pair accuracy and correctness-conditioned AUROC are used to test whether validity remains accessible within controlled pairs and on behaviorally incorrect examples. Full probe and resampling details are given in Appendix C.

## 4.3 Controls

We compare hidden-state probes with TF–IDF classifiers trained on the full prompt, claim only, and premises only. We also fit metadata-only classifiers and repeat probing with 200 shuffled-label permutations. These controls test whether probe performance can be explained by lexical regularities, construction metadata, or arbitrary linear separability. Difficulty and within-Pythia scaling are treated as secondary analyses and reported in Appendix G.

## 4.4 Causal Intervention

To test whether the probe-derived direction is behaviorally consequential, we intervene on the final prompt-token state at the selected layer. The normalized probe direction is expressed in raw activation coordinates and scaled by the training-set standard deviation of projection onto that direction. We apply interventions at

$$
\alpha \in \{ - 4 , - 2 , - 1 , 0 , 1 , 2 , 4 \} ,
$$

and measure the resulting change in output margin

$$
\Delta M _ { \alpha } ( x ) = M _ { \alpha } ( x ) - M _ { 0 } ( x ) .
$$

The learned direction is compared with five norm-matched random orthogonal directions. We additionally perform matched-projection patching between valid and invalid members of each pair. These experiments test the narrow hypothesis that the particular linear direction identified by the probe is sufficient to influence verification behavior. Full intervention construction and pair-bootstrap procedures are given in Appendix H.

## 5 Results

Across the five models, behavioral verification and internal validity decodability diverge sharply. Output-level performance remains at or near chance, while validity is strongly linearly decodable from hidden states. This decodability transfers well across unseen templates and many semantic domains and inference families, although exhaustive leave-one-out evaluation reveals systematic exceptions. Finally, the probe-derived validity direction has only weak and non-specific effects on model outputs.

## 5.1 Behavioral Verification Is Unreliable

Behavioral accuracy is close to chance for every model (Table 2), but the response patterns differ substantially. Pythia-1.4B predicts INVALID for every example and Pythia-2.8B does so for nearly all examples. SmolLM3-3B and Mistral-7B show the opposite preference, predicting VALID throughout. Llama-3.2 uses both labels but reaches only 0.470 accuracy. Margin AUROC is also near chance for Pythia and Llama. SmolLM3 and Mistral retain modest ranking information, but this is not reflected in their binary predictions. Thus, similar chance-level accuracies mask very different model-specific answer-label biases.

Table 2: Behavioral verification performance. Confidence intervals use pair-level bootstrap resampling.
<table><tr><td>Model</td><td>Accuracy [95% CI]</td><td>Pred. VALID</td><td>Margin AUROC [95% CI]</td></tr><tr><td>Pythia-1.4B</td><td>0.500 [0.500, 0.500]</td><td>0.000</td><td>0.495 [0.488, 0.503]</td></tr><tr><td>Pythia-2.8B</td><td>0.500 [0.496, 0.504]</td><td>0.013</td><td>0.496 [0.487, 0.505]</td></tr><tr><td>SmolLM3-3B</td><td>0.500 [0.500, 0.500]</td><td>1.000</td><td>0.561 [0.551, 0.572]</td></tr><tr><td>Llama-3.2-3B</td><td>0.470 [0.455, 0.483]</td><td>0.555</td><td>0.458 [0.450, 0.464]</td></tr><tr><td>Mistral-7B</td><td>0.500 [0.500, 0.500]</td><td>1.000</td><td>0.576 [0.564, 0.588]</td></tr></table>

Table 3: Selected-layer validity-probe AUROC. $\ell ^ { * } / L$ gives the selected random-split layer and normalized depth. Confidence intervals use pair-level bootstrap resampling.
<table><tr><td>Model</td><td> $\ell ^ { * } / L$ </td><td>Random</td><td>Template</td><td>Domain</td><td>Family</td></tr><tr><td>Pythia-1.4B</td><td>10/24 (0.417)</td><td>1.000 [1.000, 1.000]</td><td>0.999 [0.996, 1.000]</td><td>0.968 [0.949, 0.987]</td><td>0.950 [0.921, 0.973]</td></tr><tr><td>Pythia-2.8B</td><td>10/32 (0.313)</td><td>1.000 [1.000, 1.000]</td><td>0.963 [0.936, 0.986]</td><td>0.989 [0.976, 0.998]</td><td>0.976 [0.959, 0.990]</td></tr><tr><td>SmolLM3-3B</td><td>20/36 (0.556)</td><td>1.000 [0.999, 1.000]</td><td>0.988 [0.978, 0.996]</td><td>0.982 [0.966, 0.994]</td><td>0.938 [0.911, 0.965]</td></tr><tr><td>Llama-3.2-3B</td><td>6/28 (0.214)</td><td>1.000 [1.000, 1.000]</td><td>0.980 [0.962, 0.995]</td><td>0.991 [0.975, 1.000]</td><td>0.992 [0.983, 0.998]</td></tr><tr><td>Mistral-7B</td><td>12/32 (0.375)</td><td>1.000 [1.000, 1.000]</td><td>0.987 [0.977, 0.995]</td><td>0.771 [0.740, 0.814]</td><td>0.995 [0.987, 0.999]</td></tr></table>

## 5.2 Validity Is Strongly Decodable from Hidden States

The hidden-state results look very different from the behavioral results. Random-split AUROC is essentially perfect for every model (Table 3). Validity-related information is also linearly accessible very early in the network: at the first transformer block, random-split AUROC ranges from 0.935 to 0.995 across models.

Decodability remains high under unseen templates, with AUROC between 0.963 and 0.999. Most domain- and family-held-out evaluations are also strong, although the exceptions become clearer in the exhaustive analysis below. The selected layer varies substantially across architectures, from normalized depth 0.214 for Llama-3.2 to 0.556 for SmolLM3-3B.

Figure 1 shows the full layer-wise trajectories. In-distribution decodability stays close to ceiling across most of the network. Under distribution shift, the early layers are more variable, but most models reach high AUROC by the middle layers.

## 5.3 OOD Generalization Is Broad but Not Uniform

The predefined domain and family splits show strong transfer for most models, but exhaustive leaveone-out evaluation reveals substantial heterogeneity (Figure 2). Domain transfer is particularly stable for Llama-3.2, which remains at or above 0.989 for every held-out domain. Other models show more localized failures. Pythia-1.4B and Pythia-2.8B fall to 0.576 and 0.714 when the biological domain is withheld, while Mistral-7B falls to 0.771 on legal-policy. The family results show a more consistent weakness. Syllogism is the weakest held-out inference family for every model: AUROC falls to 0.525 for Pythia-1.4B, 0.687 for Pythia-2.8B, 0.544 for SmolLM3-3B, 0.500 for Llama-3.2, and 0.762 for Mistral-7B. Several other families transfer close to perfectly. These results support substantial shared validity-related structure across conditions, but not a single representation that transfers uniformly across all semantic domains and inference families. Full leave-one-out values are reported in Appendix D.

## 5.4 Validity Remains Decodable When Behavior Fails

Matched-pair evaluation gives a strong within-context result. Across all five models and all four primary splits, the selected probes achieve pairwise accuracy of 1.000. This ordering remains intact even when global calibration degrades. Mistral-7B, for example, reaches only 0.771 global AUROC under domain holdout while still ranking the valid member above the invalid member of every matched pair. Correctness-conditioned analysis gives a complementary view. Llama-3.2 provides the cleanest comparison because both gold classes are represented in the correct and incorrect subsets. AUROC on incorrectly answered examples is 1.000, 0.995, 1.000, and 0.994 for the random, template, domain, and family splits, respectively. Pythia-2.8B shows the same qualitative pattern where the comparison is defined; under domain holdout, AUROC is 0.965 on correct examples and 1.000 on incorrect examples.

![](images/bc05a7ec4ac1346b426ed653f279df2283e1c444d8f056a8e5985a7d719a7f4c.jpg)  
Figure 1: Layer-wise validity-probe AUROC across normalized transformer depth. Top left: random split; top right: template held out; bottom left: domain held out; bottom right: inference family held out. Validity remains broadly decodable under distribution shift, although the trajectories differ across models and conditions.

![](images/24639cf0bb422964e58f019001d3e12c0cdb56007b152f435ff2c0306d5d19b5.jpg)

![](images/b24393c7b27c6757f640823f721617e7786fdd737316be11df22679e9f2b4cba.jpg)  
Figure 2: Exhaustive leave-one-out validity-probe generalization. Left: each semantic domain is held out in turn. Right: each inference family is held out. Domain failures are largely model-specific, whereas syllogistic reasoning is consistently the weakest unseen inference family.

For models with nearly deterministic answer-label preferences, some correctness-conditioned AU-ROCs are undefined because the resulting subset contains only one gold class. We therefore make the narrower claim that, where this comparison is statistically well defined, behavioral errors do not generally coincide with the absence of linearly accessible validity information. Full results are given in Appendix E.

Table 4: Causal intervention results at α = +4. ∆M is the mean change in the VALID–INVALID output margin. Random columns summarize five norm-matched orthogonal directions.
<table><tr><td>Model</td><td>∆M [95% CI]</td><td>Prediction flips</td><td>Mean |∆M| random</td><td>Max |∆M| random</td></tr><tr><td>Pythia-2.8B</td><td>-0.0037[-0.0076, 0.0004]</td><td>0.0%</td><td>0.0077</td><td>0.0129</td></tr><tr><td>Llama-3.2-3B</td><td>-0.0022 [-0.0031, -0.0013]</td><td>0.0%</td><td>0.0072</td><td>0.0097</td></tr><tr><td>Mistral-7B</td><td>+0.0023[+0.0011, +0.0034]</td><td>0.0%</td><td>0.0058</td><td>0.0103</td></tr></table>

## 5.5 Controls Qualify the In-Distribution Result

The random split contains substantial lexical predictability. Full-prompt TF–IDF reaches 0.970 AUROC and claim-only TF–IDF reaches 0.965, so near-perfect in-distribution probe performance alone is not sufficient evidence for a general validity representation. The lexical baselines weaken more under distribution shift. Full-prompt TF–IDF falls to 0.855 under template holdout and 0.814 under domain holdout, while claim-only performance falls to 0.682 and 0.767, respectively. Hiddenstate probes are generally stronger under these shifts, although Mistral’s domain-held-out result (0.771) is an exception. Premises-only classifiers remain at chance, and metadata-only classifiers remain between approximately 0.49 and 0.51. Shuffled-label probes also remain near chance: null means range from 0.496 to 0.499, with standard deviations of approximately 0.031–0.039. For every model, the observed random-split result exceeds all 200 permutations $( p = 1 / 2 0 1 \approx 0 . 0 0 5 )$ . These controls indicate that surface regularities contribute substantially to in-distribution separability, but do not fully account for the broader hidden-state generalization. Full control results are reported in Appendix F.

Scaling and difficulty. Within the Pythia family, increasing scale from 1.4B to 2.8B improves domain- and family-held-out performance but reduces template-held-out AUROC from 0.999 to 0.963. Behavioral accuracy also shows no consistent decline across the three nominal difficulty levels. We therefore find neither a uniform within-family scaling effect nor a monotonic behavioral difficulty gradient. Full results are reported in Appendix G.

## 5.6 Strong Decodability Does Not Imply Causal Control

We finally test the narrower causal hypothesis that the linear direction identified by the probe is itself sufficient to influence verification behavior at the selected intervention site. At α = +4, interventions along the probe-derived direction change the VALID–INVALID margin by only −0.0037 for Pythia-2.8B, −0.0022 for Llama-3.2, and +0.0023 for Mistral-7B (Table 4). The sign is not consistent across models, and random orthogonal perturbations produce effects of comparable or greater magnitude.

No model changes its binary prediction under the α = +4 intervention. At α = −4, only one of 160 Llama examples changes prediction. Matched-projection patching gives the same qualitative result, with margin changes remaining close to zero. These results do not show that validity-related information is causally irrelevant. They show that the particular linear probe direction, at the tested intervention site, is not a strong or validity-specific control variable for the final decision. Ful intervention details and the complete strength sweep are reported in Appendix H.

## 6 Discussion

Validity can still be easy to decode from the hidden states even when the model gets the verification decision wrong. This is most apparent in the matched-pair and correctness-conditioned analyses, where decodability remains high on examples the model answers incorrectly. At the same time, the representation does not generalize uniformly. Leave-one-out evaluation reveals model-specific domain failures and a recurring weakness when syllogistic reasoning is excluded from probe training. Validity-related structure therefore appears to be partly shared across reasoning settings, but not fully invariant across them. The intervention results further limit what can be concluded from high probe accuracy. Manipulating the probe-derived direction produces only small and inconsistent behavioral changes, often no larger than random-direction controls. Linear accessibility should therefore be distinguished from both behavioral expression and causal control.

## 7 Limitations

Our dataset is deliberately controlled and synthetic. This makes matched comparisons and distribution shifts possible, but limits how directly the findings generalize to naturalistic reasoning, longer contexts, or more open-ended tasks. The five inference families and semantic domains also cover only a small part of the space of logical reasoning.

Our analysis focuses on linear probes at the final prompt-token representation. Validity may also be encoded nonlinearly, distributed across token positions, or represented differently at other computational sites. Likewise, the weak steering effects only rule out a relatively simple causal interpretation of the probe direction; they do not show that validity-related information is causally irrelevant to the model’s computation.

Finally, all models are relatively small open-weight transformers evaluated under quantization. Replication with larger models and a broader range of architectures would help establish how general these patterns are.

## 8 Conclusion

We studied logical verification at three levels: behavioral expression, representational accessibility, and causal use. For models and conditions where correctness-conditioned evaluation is well defined, validity remains strongly decodable even on incorrectly answered examples. However, generalization is not universal, and probe-derived validity directions have little specific causal effect on output behavior. These findings show why high probe accuracy should be interpreted carefully. A feature can be readily recoverable from a model’s representations without being reliably expressed in its output or corresponding to a simple causal mechanism for the final decision.

## References

[1] Leonardo Bertolazzi, Sandro Pezzelle, and Raffaella Bernardi. How language models conflate logical validity with plausibility: A representational analysis of content effects. In Findings of the Association for Computational Linguistics: ACL 2026, pages 9316–9347, San Diego, California, United States, 2026. Association for Computational Linguistics. doi: 10.18653/v1/ 2026.findings-acl.454. URL https://aclanthology.org/2026.findings-acl.454/.

[2] Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In International Conference on Learning Representations, 2023.

[3] Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Wenfei Zhou, James Coady, David Peng, Yujie Qiao, Luke Benson, et al. FOLIO: Natural language reasoning with first-order logic. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

[4] Yifan Hou, Jiaoda Li, Yu Fei, Alessandro Stolfo, Wangchunshu Zhou, Guangtao Zeng, Antoine Bosselut, and Mrinmaya Sachan. Towards a mechanistic interpretation of multi-step reasoning capabilities of language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4902–4919. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.299.

[5] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems, 2022.

[6] Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. In International Conference on Learning Representations, 2023.

[7] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. Transactions on Machine Learning Research, 2023.

[8] Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. arXiv preprint arXiv:2310.06824, 2023.

[9] Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Proceedings of the BlackboxNLP Workshop, 2023.

[10] Mihir Parmar, Nisarg Patel, Neeraj Varshney, Mutsumi Nakamura, Man Luo, Santosh Mashetty, Arindam Mitra, and Chitta Baral. LogicBench: Towards systematic evaluation of logical reasoning ability of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024.

[11] Nisarg Patel, Mohith Kulkarni, Mihir Parmar, Aashna Budhiraja, Mutsumi Nakamura, Neeraj Varshney, and Chitta Baral. Multi-logieval: Towards evaluating multi-step logical reasoning ability of large language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 20856–20879, 2024.

[12] Subramanyam Sahoo, Vinija Jain, Aman Chadha, and Divya Chaudhary. Linear probes detect task format, not reasoning mode in language model hidden states, 2026. URL https://arxiv. org/abs/2606.02907.

[13] Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Seyed Mehran Kazemi, Najoung Kim, and He He. Testing the general deductive reasoning capacity of large language models using out-of-distribution examples. In Advances in Neural Information Processing Systems, 2023.

[14] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023.

[15] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. In Findings of the Associationfor Computational Linguistics: ACL 2023, 2023.

[16] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems, 2023.

[17] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022.

[18] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

## A Dataset Construction

## A.1 Task Representation

The dataset contains 800 examples organized into 400 matched valid–invalid pairs. Each example is represented as

$$
x _ { i } = ( P _ { i } , c _ { i } , y _ { i } , f _ { i } , d _ { i } , t _ { i } , \delta _ { i } ) ,
$$

where $P _ { i }$ is the premise set, $c _ { i }$ is the candidate claim, $y _ { i } \in \{ 0 , 1 \}$ is the validity label, $f _ { i }$ denotes inference family, $d _ { i }$ semantic domain, $t _ { i }$ template family, and $\delta _ { i }$ difficulty. We instantiate five inference families: syllogism, transitivity, set inclusion, causal chain, and permission logic. Each family appears across five semantic domains: nonce, spatial, social, biological, and legal-policy.

Table 5: Composition of the controlled verification dataset.
<table><tr><td>Property</td><td>Count</td></tr><tr><td>Total examples</td><td>800</td></tr><tr><td>Matched pairs Valid / Invalid</td><td>400</td></tr><tr><td>Inference families</td><td>400 / 400 5</td></tr><tr><td>Semantic domains Examples per family</td><td>5 160</td></tr><tr><td>Examples per domain</td><td>160</td></tr><tr><td>Difficulty 1 / 2 / 3 Distractor examples</td><td>200 / 400 / 200 200</td></tr></table>

Table 6: Primary probe train–test splits.
<table><tr><td>Split</td><td>Train</td><td>Test</td></tr><tr><td>Random</td><td>640</td><td>160</td></tr><tr><td>Template held out</td><td>600</td><td>200</td></tr><tr><td>Domain held out</td><td>640</td><td>160</td></tr><tr><td>Inference family held out</td><td>640</td><td>160</td></tr></table>

## A.2 Matched Valid–Invalid Construction

For each pair, the premise context and construction variables are held fixed while the candidate claim changes. For a matched pair $( x _ { i } ^ { + } , x _ { i } ^ { - } )$

$$
{ \cal P } _ { i } ^ { + } = { \cal P } _ { i } ^ { - } , \qquad f _ { i } ^ { + } = f _ { i } ^ { - } , \qquad d _ { i } ^ { + } = d _ { i } ^ { - } , \qquad t _ { i } ^ { + } = t _ { i } ^ { - } , \qquad \delta _ { i } ^ { + } = \delta _ { i } ^ { - } ,
$$

while

$$
y _ { i } ^ { + } = 1 , \qquad y _ { i } ^ { - } = 0 .
$$

Matched pairs are kept together during data splitting, probe validation, and pair-level bootstrap resampling.

## A.3 Difficulty Levels

We use three controlled difficulty levels:

• Difficulty 1: direct one-step verification;

• Difficulty 2: two-step inference chains;

• Difficulty 3: two-step chains with an irrelevant distractor premise.

These labels describe the construction of an example rather than assuming that behavioral difficulty must increase monotonically across levels.

## A.4 Train–Test Splits

We evaluate four complementary probe splits. The random split is the in-distribution reference. The template-held-out split excludes complete template families from probe training, the domain-held-out split excludes a semantic domain, and the inference-family-held-out split excludes an entire reasoning family.

All four splits remain label balanced. Complete template families are disjoint between training and test in the strict template condition. We additionally perform exhaustive leave-one-domain-out and leave-one-inference-family-out evaluations.

Table 7: Models used in the experiments.
<table><tr><td>Model</td><td>Parameters</td><td>Hugging Face identifier</td><td>Layers</td></tr><tr><td>Pythia-1.4B</td><td>1.4B</td><td>EleutherAI/pythia-1.4b</td><td>24</td></tr><tr><td>Pythia-2.8B</td><td>2.8B</td><td>EleutherAI/pythia-2.8b</td><td>32</td></tr><tr><td>SmolLM3-3B</td><td>3B</td><td>HuggingFaceTB/SmolLM3-3B-Base</td><td>36</td></tr><tr><td>Llama-3.2-3B</td><td>3B</td><td>meta-1lama/Llama-3.2-3B</td><td>28</td></tr><tr><td>Mistral-7B</td><td>7B</td><td>mistralai/Mistral-7B-vO.3</td><td>32</td></tr></table>

## B Models and Implementation Details

Layer position is reported using normalized transformer depth,

$$
d _ { \ell } = \frac { \ell } { L } ,
$$

where ℓ is the block index and L is the total number of blocks.

## B.1 Behavioral Label Scoring and Tokenization

Behavior is scored from the conditional likelihood assigned to VALID and INVALID. Because these strings may tokenize differently across model families, we audit three candidate answer prefixes: the plain label, a leading-space label, and a leading-newline label. Prefix selection prioritizes equal token counts for the two answer labels, followed by the shortest available encoding.

For token sequences $T _ { V } = ( v _ { 1 } , \dots , v _ { K _ { V } } )$ and $T _ { I } = ( u _ { 1 } , \dots , u _ { K _ { I } } )$ , we define

$$
s _ { V } ( x ) = { \frac { 1 } { K _ { V } } } \sum _ { k = 1 } ^ { K _ { V } } \log p ( v _ { k } \mid x , v _ { < k } ) ,
$$

with $s _ { I } ( x )$ defined analogously. The behavioral margin is

$$
M ( x ) = s _ { V } ( x ) - s _ { I } ( x ) .
$$

This audit is important because the strong answer-label preferences observed in Section 5.1 should not be attributed simply to unequal answer-string length.

## C Probe Training and Statistical Details

For each example i and transformer block $\ell ,$ we extract the final prompt-token hidden state

$$
\displaystyle h _ { i , \ell } \in \mathbb { R } ^ { m } .
$$

At each layer, we fit an $\ell _ { 2 } \cdot$ -regularized logistic-regression probe to predict gold validity. Features are standardized using statistics estimated only from the probe-training partition. The regularization coefficient is selected from

$$
C \in \{ 0 . 1 , 1 , 1 0 \}
$$

using group-aware cross-validation with pair\_id as the grouping variable.

Layer selection is also performed entirely within the training data:

$$
\ell ^ { * } = \arg \operatorname* { m a x } _ { \ell } \mathrm { A U R O C } _ { \ell } ^ { \mathrm { v a l } } .
$$

After selecting $\ell ^ { * }$ and the regularization coefficient, the probe is refit on the available training data and evaluated once on the held-out test partition. Full layer-wise trajectories are shown in Figure 1.

## C.1 Pairwise Evaluation

For each matched valid–invalid pair, we ask whether the probe assigns a larger validity score to the valid member:

$$
\mathrm { P a i r A c c } = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \mathbb { I } \left[ q ( i _ { V } ) > q ( i _ { I } ) \right] ,
$$

where $q ( \cdot )$ denotes the probe score.

Table 8: Leave-one-domain-out selected-layer validity-probe AUROC.
<table><tr><td>Model</td><td>Biological</td><td>Legal-policy</td><td>Nonce</td><td>Social</td><td>Spatial</td></tr><tr><td>Pythia-1.4B</td><td>0.576</td><td>0.968</td><td>0.887</td><td>0.943</td><td>0.958</td></tr><tr><td>Pythia-2.8B</td><td>0.714</td><td>0.989</td><td>0.809</td><td>0.945</td><td>0.990</td></tr><tr><td>SmolLM3-3B</td><td>1.000</td><td>0.982</td><td>0.894</td><td>0.956</td><td>0.999</td></tr><tr><td>Llama-3.2-3B</td><td>0.989</td><td>0.991</td><td>0.998</td><td>0.991</td><td>0.998</td></tr><tr><td>Mistral-7B</td><td>1.000</td><td>0.771</td><td>0.998</td><td>0.997</td><td>1.000</td></tr></table>

Table 9: Leave-one-inference-family-out selected-layer validity-probe AUROC.
<table><tr><td>Model</td><td>Causal chain</td><td>Permission</td><td>Set inclusion</td><td>Syllogism</td><td>Transitivity</td></tr><tr><td>Pythia-1.4B</td><td>0.950</td><td>0.939</td><td>0.999</td><td>0.525</td><td>0.884</td></tr><tr><td>Pythia-2.8B</td><td>0.976</td><td>0.965</td><td>0.924</td><td>0.687</td><td>0.779</td></tr><tr><td>SmolLM3-3B</td><td>0.938</td><td>0.949</td><td>0.897</td><td>0.544</td><td>0.957</td></tr><tr><td>Llama-3.2-3B</td><td>0.992</td><td>0.998</td><td>0.988</td><td>0.500</td><td>0.972</td></tr><tr><td>Mistral-7B</td><td>0.995</td><td>0.990</td><td>0.921</td><td>0.762</td><td>0.766</td></tr></table>

## C.2 Uncertainty and Statistical Reporting

Because examples occur in matched pairs, the pair is treated as the independent resampling unit. We compute 95% confidence intervals from 2000 pair-level bootstrap resamples, sampling pair\_ids with replacement and recomputing the corresponding statistic. Probe performance is reported using AUROC. Behavioral evaluation reports accuracy, the fraction of predictions assigned VALID, and output-margin AUROC. Matched-pair analyses report pairwise accuracy. Correctness-conditioned AUROC is reported only when both gold classes are present in the corresponding subset; otherwise it is undefined and omitted. For shuffled-label controls, we report the mean and standard deviation of the null AUROC distribution over 200 permutations and compute

$$
p = { \frac { 1 + \# \{ { \mathrm { p e r m u t e d ~ A U R O C } } \geq { \mathrm { o b s e r v e d ~ A U R O C } } \} } { 2 0 0 + 1 } } .
$$

Unless otherwise stated, reported probe values refer to the layer selected using training-partition validation only.

## D Full Generalization Results

The primary domain-held-out split withholds legal-policy, while the primary inference-family-heldout split withholds causal chain. To test whether these results depend on that choice, we additionally hold out every semantic domain and inference family in turn.

The domain results show largely model-specific weaknesses. Llama-3.2 is the most stable across domains, whereas both Pythia models weaken when the biological domain is unseen and Mistral-7B weakens on legal-policy. The inference-family results reveal a more systematic pattern. Syllogism is the weakest unseen family for all five models. This heterogeneity motivates interpreting the main held-out results as evidence for substantial shared validity-related structure rather than a single representation that is fully invariant across reasoning families.

## E Behavior–Representation Dissociation Details

Across all five models and all four primary evaluation conditions, the selected probes achieve matchedpair accuracy of 1.000, with pair-level bootstrap confidence intervals of [1.000, 1.000]. Correctnessconditioned evaluation asks whether validity remains decodable after separating examples according to whether the model’s output-level decision is correct. Llama-3.2 provides the cleanest comparison because both validity classes remain represented in both subsets.

Pythia-2.8B provides an additional comparison where both gold classes remain available. Under domain holdout, AUROC is 0.965 on behaviorally correct examples and 1.000 on incorrectly answered examples. For models whose predictions collapse almost entirely to one answer label, some correctness-conditioned subsets contain only one gold class. AUROC is mathematically undefined in these cases, so these conditions are omitted rather than assigned an artificial score.

Table 10: Llama-3.2-3B validity-probe AUROC conditioned on behavioral correctness.
<table><tr><td rowspan="2">Split</td><td colspan="2">Correct</td><td colspan="2">Incorrect</td></tr><tr><td>n</td><td>AUROC</td><td>n</td><td>AUROC</td></tr><tr><td>Random</td><td>75</td><td>1.000</td><td>85</td><td>1.000</td></tr><tr><td>Template</td><td>98</td><td>0.970</td><td>102</td><td>0.995</td></tr><tr><td>Domain</td><td>71</td><td>0.989</td><td>89</td><td>1.000</td></tr><tr><td>Family</td><td>68</td><td>1.000</td><td>92</td><td>0.994</td></tr></table>

Table 11: Lexical and metadata control AUROC. Metadata entries give the range across model-specific evaluations.
<table><tr><td>Control</td><td>Random</td><td>Template</td><td>Domain</td><td>Family</td></tr><tr><td>Full prompt</td><td>0.970</td><td>0.855</td><td>0.814</td><td>0.884</td></tr><tr><td>Claim only</td><td>0.965</td><td>0.682</td><td>0.767</td><td>0.752</td></tr><tr><td>Premises only</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td></tr><tr><td>Metadata only</td><td>0.498–0.503</td><td>0.501–0.504</td><td>0.490–0.504</td><td>0.501–0.508</td></tr></table>

## F Lexical, Metadata, and Shuffled-Label Controls

We compare hidden-state probes with TF–IDF classifiers using the full prompt, claim only, and premises only. We additionally fit metadata-only classifiers using inference family, domain, difficulty, template-pair identity, prompt-template identity, distractor status, and prompt length.

The random split contains substantial lexical predictability: both full-prompt and claim-only TF–IDF classifiers perform well in-distribution. These controls therefore make clear that random-split probe performance alone is insufficient for distinguishing broadly transferable validity information from dataset regularities.The premises-only classifier remains at chance, consistent with the matchedpair construction in which valid and invalid examples within a pair share their premise context. Metadata-only prediction also remains close to chance across the primary splits.

For the shuffled-label control, probe training is repeated for 200 random label permutations. Null AUROC means range from 0.496 to 0.499, with standard deviations of approximately 0.031–0.039. For every model, the observed random-split AUROC exceeds all 200 shuffled-label results:

$$
p = { \frac { 1 } { 2 0 0 + 1 } } \approx 0 . 0 0 5 .
$$

Together, these controls motivate placing greater weight on held-out generalization, matched-pair comparisons, and correctness-conditioned evaluation than on the near-perfect random-split AUROC alone.

## G Secondary Analyses

## G.1 Within-Pythia Scaling

The two Pythia models provide a limited within-family comparison of parameter scale. Increasing model size from 1.4B to 2.8B improves some forms of transfer but not others.

Domain-held-out AUROC increases from 0.968 to 0.989, while family-held-out AUROC increases from 0.950 to 0.976. Mean leave-one-domain-out performance also rises from 0.866 to 0.890, while mean leave-one-family-out performance changes only modestly from 0.859 to 0.866.

Template transfer moves in the opposite direction, decreasing from 0.999 to 0.963. This limited two-model comparison therefore does not support a simple monotonic relationship between parameter count and validity decodability.

Table 12: Within-family comparison between Pythia-1.4B and Pythia-2.8B. Values are validity-probe AUROC.
<table><tr><td>Evaluation</td><td>Pythia-1.4B</td><td>Pythia-2.8B</td></tr><tr><td>Random</td><td>1.000</td><td>1.000</td></tr><tr><td>Template held out</td><td>0.999</td><td>0.963</td></tr><tr><td>Domain held out</td><td>0.968</td><td>0.989</td></tr><tr><td>Family held out</td><td>0.950</td><td>0.976</td></tr><tr><td>Mean domain LOO</td><td>0.866</td><td>0.890</td></tr><tr><td>Mean family LOO</td><td>0.859</td><td>0.866</td></tr></table>

## G.2 Nominal Difficulty

Difficulty does not produce a consistent behavioral gradient across models. Pythia-1.4B, SmolLM3- 3B, and Mistral-7B remain at 0.500 behavioral accuracy across all three difficulty levels. Pythia-2.8B varies only between 0.498 and 0.505. Llama-3.2-3B shows more variation, but not in the expected direction: accuracy increases from 0.435 on direct one-step examples to 0.475 on two-step examples and 0.495 on two-step examples with distractors. Thus, the nominal construction difficulty does not produce a monotonic decline in behavioral accuracy. We therefore treat it as a secondary descriptive analysis rather than as an ordered measure of empirical reasoning difficulty.

## H Causal Intervention Details

The intervention uses the layer selected by the random-split probe. Because the probe is fitted on standardized features, its coefficient vector is first mapped back into raw activation coordinates. Let w<sub>ℓ</sub>∗ denote the learned standardized probe coefficient vector and $s _ { \ell ^ { * } }$ the element-wise training-set standard deviation. We define

$$
\tilde { v } = w _ { \ell ^ { * } } \oslash s _ { \ell ^ { * } } , \qquad v = \frac { \tilde { v } } { \lVert \tilde { v } \rVert _ { 2 } } .
$$

The intervention scale is defined using the training-set standard deviation of projection onto v:

$$
\sigma _ { v } = \mathrm { S D } _ { i \in \mathrm { t r a i n } } \left( h _ { i , \ell ^ { * } } ^ { \top } v \right) .
$$

We modify the final prompt-token hidden state according to

$$
\begin{array} { r } { h _ { i , \ell ^ { * } } ^ { \prime } = h _ { i , \ell ^ { * } } + \alpha \sigma _ { v } v , } \end{array}
$$

for

$$
\alpha \in \{ - 4 , - 2 , - 1 , 0 , 1 , 2 , 4 \} .
$$

The primary behavioral outcome is

$$
\Delta M _ { \alpha } ( x ) = M _ { \alpha } ( x ) - M _ { 0 } ( x ) ,
$$

together with changes in the model’s binary verification decision.

## H.1 Random-Direction Controls

For each model, the probe-derived direction is compared with five random directions orthogonal to v. Random interventions are norm matched to the probe-direction perturbation at each intervention strength. This comparison tests whether an observed change is specific to the probe-derived direction rather than a generic consequence of perturbing the hidden state.

## H.2 Full Intervention Sweep

Across the intervention sweep, changes in binary predictions are rare. No prediction flips occur for any model at α = +4. At $\alpha = - 4$ , only Llama-3.2 changes a binary decision, with one flip among 160 evaluated examples (0.625%).

![](images/137627b6af166a37c3c150a974d7a06464093da01ccd95f836c34fc7f0897a20.jpg)

![](images/efe2c028427e76a6f725e5b1ae132e408f3db701285e77ca9f07b2d5a0969eda.jpg)

![](images/264f214b834a0936e9515fac7b677e2d0d5ea2506ae47174626a904efac9cf76.jpg)  
Figure 3: Effect of intervention strength on verification output for Pythia-2.8B, Llama-3.2, and Mistral-7B. The full sweep uses α $\in \ \bar { \{ - 4 , - 2 , - 1 , 0 , 1 , 2 , 4 \bar { \} } }$ . Changes along the probe-derived validity direction remain small and are not consistently larger than those produced by norm-matched random orthogonal directions.

## H.3 Matched-Projection Patching

We additionally use the naturally occurring probe-direction difference between the valid and invalid member of a matched pair. For target state $h _ { i }$ and matched source state $h _ { j }$ , define

$$
\delta _ { j  i } = \boldsymbol { v } ^ { \top } h _ { j } - \boldsymbol { v } ^ { \top } h _ { i } .
$$

We then replace only the target’s projection along v:

$$
h _ { i } ^ { \prime } = h _ { i } + \delta _ { j \to i } v .
$$

Mean output-margin changes remain small: approximately 0.0020–0.0045 for Pythia-2.8B, at most 0.0011 in magnitude for Llama-3.2-3B, and approximately 0.0011–0.0012 for Mistral-7B. These interventions test a deliberately narrow causal hypothesis. Their weak effect does not establish that validity-related information is causally irrelevant to the model’s computation. Rather, the results do not support the probe-derived linear direction as a sufficient low-dimensional control variable at the tested intervention site.