# AC-JEPA: Anchored Cross-Modal Joint-Embedding Predictive Architecture for Visual-to-Symbolic Translation

Chaofei HONG<sup>\*</sup>, Ziyang Chen, Baochen Wang, Christina Yi Jin, Qing Zhang

Zhejiang Lab

hongchf@zhejianglab.com

October 28, 2026

## Abstract

Cross-modal translation requires models to capture shared semantics across modalities while preserving the fine-grained structure required by the target modality. Yet in real-world scientific domains, paired data are often imperfect, while obtaining clean pairs at scale is costly. Supervised fine-tuning (SFT) directly optimizes cross-modal translation through categorical token-level supervision, but this objective is sensitive to errors in the target sequences. JEPA offers a complementary representation-space objective, but most existing formulations focus on prediction within a modality rather than translation across modalities. We introduce Anchored Cross-Modal JEPA (AC-JEPA), an SFT-compatible plug-in that predicts masked intermediate states in both cross-modal directions using paired data. The joint-embedding objective targets cross-modal structure predictable from context, while the causal SFT path retains direct token-level supervision for the target sequence. During self-training, AC-JEPA learns from continuous latent targets derived from imperfect pseudo-labeled pairs, providing a route for using these pairs without promoting pseudo-tokens to hard categorical supervision. Using optical chemical structure recognition as a testbed, gold-pair AC-JEPA achieves 76.07% accuracy on WildMol Simple, compared with 67.40% for Base SFT. Both models share complete MolParser-7M synthetic pretraining and Real-SFT fine-tuning; AC-JEPA reuses the already-seen synthetic pairs through latent prediction without adding unique gold-pair data. Using the same pseudo-labeled data for both approaches, AC-JEPA-based self-training further raises accuracy to 80.23%, whereas accuracy under direct SFT drops from 67.40% to 63.47%. These results support SFT-anchored joint-embedding prediction as an effective strategy for cross-modal translation under both curated and imperfect supervision.

## 1 Introduction

Scientific images encode knowledge through the identities, positions, and relations of graphical elements rather than through appearance alone. Translating such images into symbolic or programming languages therefore requires a model to recognize local primitives, recover spatial or topological structure, resolve long-range dependencies, and serialize the result under a strict target grammar. This differs from coarse semantic recognition or open-ended captioning: a single misplaced symbol can alter the encoded object or invalidate the output. The problem is further complicated by real scientific data, where visual observations and paired sequences can be imperfect and expensive to curate at scale. Optical Chemical Structure Recognition (OCSR) exemplifies these demands. Molecular diagrams encode atoms, bonds, stereochemistry, and topology through a spatial notation, whereas SMILES represents the same molecular structure as a compact, grammar-sensitive sequence [8, 23].

Autoregressive multimodal generators provide a direct solution to this conversion problem. A visual encoder extracts image evidence, and a causal language decoder combines it with the previously generated symbols to predict the target sequence. Supervised fine-tuning (SFT) supplies exact token-level supervision and is therefore effective for learning the target grammar and decoding procedure [8]. Its categorical objective, however, specifies the desired output tokens without directly requiring the fused intermediate states to recover predictable structure removed from either modality. This distinction becomes important when training uses model-generated pairs. Direct pseudo-label SFT promotes every generated token to a hard target, so each local pseudo-label error directly biases the decoder’s learned conditional distribution [9, 20].

![](images/44f92416b133c632c16707ddc808d48a37274095c3aea5c6911245a30c7c6c51.jpg)  
Figure 1: Macro view of AC-JEPA. Paired image and SMILES views are processed by one image-to-SMILES VLM, while SFT anchors causal generation and the JEPA objective regularizes intermediate hidden states through latent MSE and SIGReg. The latent self-training loop is shown only as an abstract training extension; data-source and curriculum details are separated from this core mechanism view.

The usefulness of paired data also depends on the stage and objective through which they are used. Large synthetic corpora provide broad molecular coverage for pretraining, whereas real-image fine-tuning addresses the visual distribution of papers and patents [8]. After this adaptation, simply resuming token supervision on synthetic pairs need not improve real-scene recognition: in our small-scale implementation, mixing MolParser-7M synthetic pairs with Real-SFT did not improve over Real-SFT-only adaptation and was more prone to overfitting. This observation motivates extracting further structural constraints from already-seen synthetic pairs while preserving the real-data token anchor.

Representation-learning objectives such as contrastive alignment, masked reconstruction, and latent prediction provide complementary ways to learn correspondences and contextual structure across paired modalities [1, 10, 18]. JEPA is particularly relevant because it predicts representations, focusing the objective on structure predictable from context rather than every input detail. Most JEPA formulations, however, optimize encoders or embedding predictors; exact autoregressive translation between modalities is absent or remains separate from latent pretraining. Applying latent prediction to a symbolic generator consequently requires more than adding a standalone pretraining loss: the predicted states must remain useful to the causal decoder, use evidence from both modalities, and avoid representational collapse.

We address these requirements with Anchored Cross-Modal JEPA (AC-JEPA), an SFT-compatible latent predictive objective inside the same generator used for inference. Figure 1 summarizes how the SFT anchor and latent objective interact.

Autoregressive SFT learns exact translation from reliable pairs, while the latent branch extracts predictive constraints from paired data, including synthetic examples already used during pretraining. The latent branch masks coherent image regions and symbolic spans, then predicts the corresponding intermediate states from the remaining multimodal context so that evidence from either modality can support the other. Both branches update the same backbone, allowing latent MSE to shape the representations used for causal decoding while modality-wise SIGReg prevents collapse [5]. In the gold-pair setting, this provides a distinct use for the synthetic pretraining corpus after real-image adaptation: its pairs supervise latent recovery while Real-SFT continues to supervise output tokens.

AC-JEPA further provides a noise-tolerant route for self-training with imperfect pseudo-labeled pairs. Instead of promoting model-generated tokens to exact targets, it learns cross-modal structure that remains predictable from the paired image and sequence, thereby refining the shared representation without converting local pseudo-label errors into categorical supervision.

We evaluate AC-JEPA on OCSR as a stringent instance of continuous-to-symbolic translation. Under a shared exact-match protocol, clean-pair training improves both WildMol Simple and Markush over the SFT anchor. The SFT anchor and gold-pair AC-JEPA share complete MolParser-7M synthetic pretraining followed by Real-SFT adaptation, so their comparison concerns reuse of existing pairs through latent prediction rather than access to additional unique data. Using the same model-generated pairs, latent self-training provides further gains, whereas direct pseudo-label SFT degrades. These results support SFT-anchored latent prediction as an effective training strategy for exact cross-modal translation with both curated and imperfect pairs.

Our contributions are:

1. We introduce AC-JEPA, an SFT-compatible framework integrating cross-modal latent prediction with causal decoding for exact visual-to-symbolic translation.

2. We develop a bidirectional cross-modal latent-learning mechanism that captures structure predictable from multimodal context while maintaining non-degenerate representations within each modality.

3. We derive local bounds that relate decoder-relevant latent residuals to next-token decision stability and characterize clean-output stability under pseudo-label perturbations.

4. We validate AC-JEPA on OCSR, showing gains over SFT with clean pairs and over direct pseudo-label SFT during self-training.

## 2 Related Work

Multimodal representation learning. Multimodal representation learning seeks to capture content shared across paired modalities while retaining the fine-grained information required for modality-specific prediction. Contrastive methods such as CLIP and ALIGN align global image–text representations and are effective at learning factors shared across paired observations [11, 18]. Fusion and generative models such as ALBEF and BLIP combine alignment with cross-modal interaction or language generation, while BEiT-3 extends masked modeling across images, text, and paired inputs [13, 14, 22]. These objectives support transferable multimodal representations and capable generators, but exact scientific translation can additionally depend on modality-specific local structure that is not captured by global correspondence alone. AC-JEPA targets this setting by applying local masked-state prediction to modality-indexed positions within the same fused representation used for autoregressive decoding. SFT preserves the task-specific output interface, while the latent objective requires the intermediate states to retain cross-modal structure recoverable from the remaining context.

Latent predictive learning and multimodal JEPA. Latent predictive learning moves the prediction target from the observation space to a learned representation space. Whereas masked autoencoding reconstructs missing observations, data2vec predicts contextualized latent targets across speech, vision, and language, and I-JEPA predicts representations of image regions from visible context [1, 3, 10]. The abstraction induced by such objectives depends on what is masked, which context remains visible, and how the target representation is constructed. V-JEPA 2 extends latent prediction to video, while LeJEPA couples predictability with SIGReg-based control of representation collapse [2, 5]. Recent cross-modal variants impose different predictive relations: CrossJEPA transfers from a frozen image target to a point-cloud encoder, VL-JEPA predicts text embeddings from visual input and a query, and MJEPA performs intra- and cross-modal prediction with a unified audio-visual encoder [7, 16, 21]. Their primary output is an encoder representation or embedding prediction. In contrast, AC-JEPA places bidirectional masked-state prediction inside the same autoregressive image-to-SMILES generator that is trained and deployed for exact symbolic translation. The central compatibility constraint is therefore not latent prediction alone, but preserving an intermediate state that remain useful to the upper causal decoder.

Self-training with imperfect pseudo-labels. Self-training expands supervised data by using a trained model to generate targets for unlabeled examples [12]. Modern methods reduce confirmation errors through confidence filtering, consistency regularization, augmentation, teacher–student updates, or momentum distillation [13, 19, 24]. Their effectiveness nevertheless depends on the structure of the teacher’s errors, since recurring mistakes can be reinforced when predictions are reused as hard supervision [20]. This issue is particularly consequential for symbolic generation: a sequence may be syntactically plausible and locally close to the correct output while still representing a different object. Direct pseudo-label SFT treats such a token substitution as an exact categorical target. AC-JEPA instead uses the pseudo-sequence as part of a masked multimodal view and learns by predicting its continuous intermediate states. Our experiments compare these two training routes using the same pseudo-labeled data rather than attributing their difference to data selection.

Optical chemical structure recognition. Optical Chemical Structure Recognition converts molecular diagrams from papers, patents, and scanned documents into machine-readable molecular graphs or line notations such as SMILES [23]. Unlike ordinary text recognition, it must recover graph topology, stereochemistry, abbreviations, drawing conventions, and, in the Markush setting, variable molecular structures. Recent neural systems approach this problem through direct image-to-string generation, chemistry-constrained graph prediction, graph classification, or large-scale end-to-end recognition [6, 8, 15, 17]. MolSight and MolRecBench-Wild further expose the importance of noisy pretraining and generalization to real scientific documents [25, 26]. Most progress in OCSR has focused on architectures, output representations, data scale, and decoding constraints. Our work studies a complementary training question: how an SFT-capable generator can learn additional cross-modal structure from clean and pseudo-labeled pairs through an auxiliary latent predictive objective.

## 3 Method

AC-JEPA integrates autoregressive SFT and latent prediction within one image-to-SMILES VLM. SFT trains the model through autoregressive next-token prediction, whereas JEPA predicts latent states at an intermediate decoder layer. Both losses update the shared backbone, so latent prediction shapes the representation used by the generator rather than a separate pretrained encoder. A bidirectional predictor recovers masked states using the remaining multimodal context, and modality-wise SIGReg prevents latent collapse. The same latent objective subsequently incorporates pseudo-labeled pairs without treating their generated SMILES as default token-level targets. Figure 2 shows the resulting architecture and training losses. Sections 3.1 to 3.4 describe the latent interface, predictive objective, training curriculum, and analysis, respectively.

## 3.1 SFT-Anchored Backbone and Latent Interface

The position of the latent interface determines whether predictive learning can affect both cross-modal fusion and downstream decoding. We attach it to an intermediate decoder layer after visual and symbolic context have been fused but before next-token logits are produced. The JEPA path projects this state for latent prediction, whereas the SFT path continues through the remaining causal decoder and language-model head. Both losses update the shared model below the interface, while SFT directly supervises the upper decoding path. This arrangement keeps the latent objective inside the task-capable generator while preserving the causal output path used at inference.

Let I be a molecular image, $S = ( S _ { 1 } , \ldots , S _ { T } )$ its gold SMILES sequence, and π the task prompt. The SFT anchor trains the backbone $f _ { \theta }$ autoregressively,

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( S _ { t } \mid \pi , I , S _ { < t } ) .\tag{1}
$$

This objective ties the shared representation to the grammar and causal decoding procedure used at inference.

For a paired symbolic view $S _ { \mathrm { v i e w } }$ , we construct clean and masked multimodal views

$$
\begin{array} { r l } & { \mathbf { V } _ { \mathrm { c l e a n } } = ( \pi , I , S _ { \mathrm { v i e w } } ) , } \\ & { \mathbf { V } _ { \mathrm { m a s k } } = m ( \mathbf { V } _ { \mathrm { c l e a n } } ; \mathcal { M } _ { \mathrm { i m g } } , \mathcal { M } _ { \mathrm { s m i } } ) . } \end{array}\tag{2}
$$

where $\mathcal { M } _ { \mathrm { i m g } }$ and $\mathcal { M } _ { \mathrm { s m i } }$ are the masked image and SMILES positions. Let $H _ { \theta } ^ { \ell } ( { \mathbf { V } } )$ be the hidden sequence at decoder layer ℓ, $g _ { \phi }$ a projector, and $p _ { \psi }$ the predictor. The projector maps the fused hidden states into a compact space in

![](images/e2d103048cf281931b9531405ee917f4aca8b3d259917d39f25172f9483a2a4a.jpg)  
Figure 2: Model structure and training losses of AC-JEPA. SFT supervises the causal decoder, while JEPA predicts intermediate latents under structured masking. The clean latent is detached only as the MSE target, while SIGReg regularizes the undetached clean image and SMILES latents separately.

which the masked-view representation is trained to match its clean counterpart. The clean latent, detached target, and masked-view prediction are

$$
\begin{array} { r l r } & { \boldsymbol Z _ { \mathrm { c l e a n } } = g _ { \phi } \big ( H _ { \theta } ^ { \ell } ( \mathbf { V } _ { \mathrm { c l e a n } } ) \big ) , } & { Z _ { \mathrm { t g t } } = \mathrm { s g } ( \boldsymbol Z _ { \mathrm { c l e a n } } ) , } \\ & { \boldsymbol Z _ { \mathrm { p r e d } } = p _ { \psi } \big ( g _ { \phi } \big ( H _ { \theta } ^ { \ell } ( \mathbf { V } _ { \mathrm { m a s k } } ) \big ) \big ) , } \end{array}\tag{3}
$$

where $\mathrm { s g }$ blocks predictive-MSE gradients through the clean target. The clean branch therefore supplies a fixed targe for each prediction step, while both the masked branch and the shared backbone below the interface remain trainable. Latent recovery is optimized only at masked positions and separately for each modality,

$$
\mathcal { L } _ { \mathrm { M S E } } = \sum _ { \begin{array} { c } { m \in \{ \mathrm { i m g } , \mathrm { s m i } \} } \\ { \left. Z _ { \mathrm { p r e d } , i } - Z _ { \mathrm { t g t } , i } \right. _ { 2 } ^ { 2 } . } \end{array} } \sum _ { i \in \mathcal { M } _ { m } }\tag{4}
$$

The continuous loss complements rather than replaces token-level SFT. SFT anchors exact generation and the output grammar, while masked latent prediction directly constrains the shared intermediate representation consumed by the decoder.

## 3.2 Bidirectional Prediction and Latent Regulation

AC-JEPA trains its intermediate representation by recovering masked image and symbolic states from the remaining multimodal context, while modality-wise regulation keeps the predictive target space non-degenerate. We construct the recovery task by replacing contiguous SMILES spans and rectangular image-patch blocks with modality-specific mask tokens; prompt, padding, and special tokens remain unmasked. Masking coherent regions removes local evidence and makes recovery rely on the surrounding structure available within and across modalities. Bidirectional attention lets the predictor recover masked states from the full corrupted multimodal context, while the SFT generation path remains causal. Section A.1 specifies the structured masking procedure.

Because predictive objectives can admit collapsed representations, we apply the Sketched Isotropic-Gaussian Regularizer (SIGReg) to the clean latent distribution [5]. For modality m, let $\mathcal { I } _ { m }$ contain its eligible clean-view positions in a mini-batch and $\mathcal { Z } _ { m } = \{ Z _ { \mathrm { c l e a n } , i } : i \in \mathcal { T } _ { m } \}$ . We regularize image and SMILES states separately,

$$
\mathcal { L } _ { \mathrm { S I G R e g } } = \sum _ { m \in \{ \mathrm { i m g } , \mathrm { s m i } \} } \lambda _ { \mathrm { s i g } } ^ { m } \mathrm { S I G R e g } ( \mathcal { Z } _ { m } ) .\tag{5}
$$

This separation prevents one modality’s variance from hiding collapse in the other. SIGReg acts on the undetached clean latents, whereas stop-gradient applies only to the MSE target. Accordingly, stop-gradient stabilizes each prediction target, whereas SIGReg constrains the representation from which the targets are formed. The complete latent objective is $\mathcal { L } _ { \mathrm { J E P A } } = \mathcal { L } _ { \mathrm { M S E } } + \mathcal { L } _ { \mathrm { S I G R e g } }$

## 3.3 From SFT Anchoring to Latent Self-Training

The training curriculum first establishes exact translation on reliable pairs and then uses the latent objective to incorporate open-environment images without gold sequences. This ordering is important because the JEPA branch should shape an already task-capable representation rather than define the symbolic task from scratch. Figure 3 summarizes the four training stages. Base SFT and all AC-JEPA variants share domain continued pretraining (CPT) on the complete synthetic MolParser-7M image–SMILES corpus, followed by autoregressive fine-tuning on Real-SFT 100K [8]. Synthetic pretraining establishes broad task competence, while real-image fine-tuning adapts the generator to document images. We freeze this Base SFT backbone to warm up the new JEPA modules; gold-pair joint training and each latent self-training run then initialize independently from the resulting checkpoint.

![](images/043b2c6e106302e3d3ef04064628288900189b87aea1db49518770177f1d0f21.jpg)  
Figure 3: Training curriculum. Base SFT first learns from the complete synthetic MolParser-7M corpus and is then fine-tuned on Real-SFT 100K. After JEPA-module warm-up with that backbone frozen, gold-pair joint training and latent self-training start independently from the shared checkpoint. Gold-pair joint training reuses the already-seen 7M pairs for latent prediction while retaining Real-SFT for token supervision. Self-training adds open-environment images through model-generated pseudo-SMILES while gold-pair SFT remains active as the anchor; pseudo-label sets are refreshed between rounds.

Let $\mathcal { D } _ { \mathrm { s f t } }$ contain the reliable pairs assigned token-level supervision and $\mathcal { D } _ { \mathrm { j e p a } } ^ { g }$ contain the gold pairs used for latent prediction. The anchored objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a n c h o r } } = \lambda _ { \mathrm { s f t } } \mathbb { E } _ { ( I , S ) \sim \mathcal { D } _ { \mathrm { s f t } } } [ \mathcal { L } _ { \mathrm { S F T } } ( I , S ) ] } \\ & { \quad \quad \quad + \mathbb { E } _ { ( I , S ) \sim \mathcal { D } _ { \mathrm { j e p a } } ^ { g } } [ \mathcal { L } _ { \mathrm { J E P A } } ( I , S ) ] . } \end{array}\tag{6}
$$

The two datasets may overlap but need not coincide: joint training means that both losses update the shared backbone, not that they are evaluated on the same examples. In our gold-pair setting, $\mathcal { D } _ { \mathrm { s f t } }$ is Real-SFT and $\mathcal { D } _ { \mathrm { j e p a } } ^ { g }$ reuses the synthetic MolParser-7M pairs already used during CPT; no unique paired examples are added to the Base SFT training history. Our small-scale mixed-SFT observation in Section 4.2 motivates retaining real-image token supervision while reusing synthetic pairs through latent prediction. At refresh round r, the current model generates $\hat { S } _ { i } ^ { ( r ) }$ for each $I _ { i } \in \mathcal { D } _ { u }$ forming $\mathcal { D } _ { p } ^ { ( r ) } = \{ ( I _ { i } , \hat { S } _ { i } ^ { ( r ) } ) \}$ . AC-JEPA then treats each pseudo-SMILES as the symbolic view $S _ { \mathrm { v i e w } }$ of a masked latent prediction problem rather than as default token-level supervision:

$$
\mathcal { L } ^ { ( r ) } = \mathcal { L } _ { \mathrm { a n c h o r } } + \eta _ { r } \mathbb { E } _ { ( I , \hat { S } ) \sim \mathcal { D } _ { p } ^ { ( r ) } } [ q ( I , \hat { S } ) \mathcal { L } _ { \mathrm { J E P A } } ( I , \hat { S } ) ] ,\tag{7}
$$

where $q ( I , { \hat { S } } )$ is a selection or reliability weight and $\eta _ { r }$ controls the pseudo-labeled stream. The supervised stream continues to supply the token-level anchor throughout this stage. Direct pseudo-label SFT is retained as a comparison, so the two routes differ in how the same pseudo-labeled data provide supervision. Algorithm 1 summarizes the complete training procedure.

## 3.4 Local Analysis of Task-Aligned Representations and Noise Tolerance

To obtain a tractable account of the proposed mechanism, we analyze a local model of AC-JEPA under simplifying assumptions on decoder margins, subspace overlap, anchored curvature, and pseudo-label influence. The resulting bounds identify conditions under which lower decoder-relevant latent residuals tighten the next-token decision-reversal bound and pseudo-label perturbations remain within the clean-output margin. They provide a mechanism-level interpretation rather than a general guarantee for the full nonlinear training dynamics. Detailed proofs and further interpretation are provided in Section $_ { \mathrm { A . 5 } }$

At masked SMILES position i, let $H _ { i }$ be the clean intermediate state, $C _ { i }$ the corrupted multimodal context, and $\Delta _ { i , s ^ { - } } ( H _ { i } ) \approx d _ { i , s ^ { - } } ^ { \top } H _ { i } + c _ { i , s ^ { - } }$ the logit gap between the correct next token $s ^ { + }$ and competitor $s ^ { - }$ . Let $\mu _ { i , s ^ { - } } ( C _ { i } ) > 0$ and $v _ { i , s ^ { - } } ( C _ { i } )$ be its conditional mean and variance, representing the average correct-token margin and residual variation. With context-conditioned state residual $U _ { i } = H _ { i } - \mathbb { E } [ H _ { i } \mid C _ { i } , S _ { i + 1 } = s ^ { + } ]$ and latent-interface Jacobian $J _ { i } ,$ , define the decoder-relevant residual energy $R _ { \mathrm { J } , i } ( C _ { i } ) = \mathbb { E } [ \| J _ { i } U _ { i } \| _ { 2 } ^ { 2 } \ | \ C _ { i } , S _ { i + 1 } = s ^ { + } ]$ . We assume the decoder margin direction is controlled through this interface: $d _ { i , s ^ { - } } d _ { i , s ^ { - } } ^ { \top } \preceq \kappa J _ { i } ^ { \top } J _ { i }$

Theorem 3.1 (SFT readout stability under latent residuals). Let $P _ { i , s ^ { - } } = \mathrm { P r } [ \Delta _ { i , s ^ { - } } ( H _ { i } ) \leq 0 \mid C _ { i } , S _ { i + 1 } = s ^ { + } ]$ . Under the local model above,

$$
\begin{array} { r l } & { P _ { i , s ^ { - } } \leq \frac { v _ { i , s ^ { - } } ( C _ { i } ) } { v _ { i , s ^ { - } } ( C _ { i } ) + \mu _ { i , s ^ { - } } ( C _ { i } ) ^ { 2 } } } \\ & { \qquad \leq \frac { \kappa R _ { \mathrm { J } , i } ( C _ { i } ) } { \kappa R _ { \mathrm { J } , i } ( C _ { i } ) + \mu _ { i , s ^ { - } } ( C _ { i } ) ^ { 2 } } . } \end{array}\tag{8}
$$

Thus, at $a )$ fixed positive mean margin, reducing the decoder-relevant latent residual tightens the decision-reversal bound.

The symbolic MSE decomposes as $\mathcal { L } _ { \mathrm { M S E } } ^ { \mathrm { s m i } } = \overline { { R } } _ { \mathrm { J } } ^ { \mathrm { s m i } } + \mathcal { E } _ { \mathrm { p r e d } } ^ { \mathrm { s m i } }$ . After predictor warm-up, reducing MSE beyond the initial excess risk lowers the Bayes latent residual and tightens the averaged bound under decoder–JEPA overlap and SFT margin preservation.

To quantify pseudo-label effects on clean outputs, consider $n$ pseudo constraints and take as the clean reference a stationary point $\theta ^ { \star }$ of the counterfactual anchored objective with their corresponding true sequences $\{ S _ { j } \} _ { j = 1 } ^ { n }$ . For pseudo constraint $j ,$ , define the latent residual $r _ { j }$ and its perturbation $e _ { j }$ after replacing $S _ { j }$ with $\widehat { S } _ { j }$

$$
\begin{array} { r l } & { r _ { j } ( \theta ; S _ { j } ) = Z _ { \mathrm { p r e d } , j } ( \theta ; S _ { j } ) - Z _ { \mathrm { t g t } , j } ( \theta ; S _ { j } ) , } \\ & { ~ e _ { j } = r _ { j } ( \theta ^ { \star } ; \widehat { S } _ { j } ) - r _ { j } ( \theta ^ { \star } ; S _ { j } ) . } \end{array}\tag{9}
$$

Let $A _ { j } = \partial r _ { j } ( \theta ; S _ { j } ) / \partial \theta | _ { \theta } ,$ be the clean-residual Jacobian, $K \succ 0$ the local anchored curvature, and λ the effective pseudo-MSE weight. Under the local shared-Jacobian approximation, with the Jacobian-change and SIGReg remainders made explicit in Section A.5.2, the first-order parameter shift is

$$
u _ { \mathrm { p s e u d o } } \approx - \frac { \lambda } { n } K ^ { - 1 } \sum _ { j = 1 } ^ { n } A _ { j } ^ { \top } e _ { j } .\tag{10}
$$

For clean context x, let $m _ { x , y k } > 0$ be the margin between correct token y and competitor $k , d _ { x , y k }$ its state gradient, and $J _ { x }$ the parameter-to-state Jacobian. Pseudo constraint $j$ has signed margin influence

$$
\zeta _ { x , y k , j } = d _ { x , y k } ^ { \top } J _ { x } K ^ { - 1 } A _ { j } ^ { \top } e _ { j } .\tag{11}
$$

Theorem 3.2 (Clean-output stability under pseudo-label perturbations). Suppose the influences are independent with $\mathbb { E } [ \zeta _ { x , y k , j } ] = \beta _ { x , y k }$ and $\mathrm { V a r } ( \zeta _ { x , y k , j } ) ~ \le ~ \sigma _ { x , y k } ^ { 2 }$ . The first-order margin is $\begin{array} { r } { G _ { x , y k } ^ { \mathrm { l i n } } = m _ { x , y k } - \lambda n ^ { - 1 } \sum _ { j } \zeta _ { x , y k , j } } \end{array}$ . If $m _ { x , y k } > \lambda \beta _ { x , y k }$ , then

$$
\operatorname* { P r } [ G _ { x , y k } ^ { \mathrm { l i n } } \leq 0 ] \leq \frac { \lambda ^ { 2 } \sigma _ { x , y k } ^ { 2 } / n } { \lambda ^ { 2 } \sigma _ { x , y k } ^ { 2 } / n + ( m _ { x , y k } - \lambda \beta _ { x , y k } ) ^ { 2 } } .\tag{12}
$$

The bound separates mean signed influence $\beta _ { x , y k } .$ , whose positive values consume clean margin, from variance that decreases across independent pseudo constraints. Conditional centering ${ \mathbb { E } } [ e _ { j } \ | \ A _ { j } ] = 0$ is sufficient for zero mean first-order influence under this local model.

## 4 Experiments

## 4.1 Experimental Setup

All experiments use Qwen3-VL-2B [4] as the backbone and draw paired training data from UniParser/MolParser-7M [8]. Base SFT and all AC-JEPA variants share the same two-stage training history: autoregressive domain pretraining on the complete synthetic MolParser-7M corpus, followed by fine-tuning on Real-SFT 100K. We use “MolParser-7M” below for the synthetic pretraining corpus and “Real-SFT” for the real-image adaptation set. Both data sources have therefore already contributed to Base SFT; the gold-pair comparison introduces no additional unique training pairs. Following Section 3.3, we warm up the newly initialized JEPA modules with this backbone frozen and then jointly optimize token-level SFT on Real-SFT and latent prediction on the previously seen synthetic pairs. Latent self-training subsequently extends the JEPA stream to pseudo-labeled open-environment images, while direct SFT on the same pseudo-labels serves as the token-level self-training baseline. For rounds R1 and R2, the unlabeled pool includes molecular images from USPTO-Mol and structure figures extracted from scientific papers. We select checkpoints on the MolParser-7M validation split and evaluate them on the WildMol Simple and Markush sets, each containing 10,000 images collected from scientific papers and patents [8]. Under a shared decoding and evaluation protocol, we report canonicalized exact-match accuracy using an RDKit-based evaluator customized for extended SMILES (E-SMILES).

## 4.2 Results and Training Dynamics Across the Curriculum

We first evaluate how performance evolves across the AC-JEPA training curriculum, from the Base SFT model to gold-pair joint training and two rounds of latent self-training. After warming up the newly initialized JEPA modules with the backbone frozen, all three AC-JEPA configurations start from the resulting checkpoint and are optimized independently. Gold-pair AC-JEPA retains Real-SFT as the token-level anchor and reuses the previously seen synthetic MolParser-7M pairs through JEPA. For R1 and R2, the unlabeled pool includes molecular images from USPTO-Mol and structure figures extracted from scientific papers. Each round generates pseudo-SMILES for this pool and adds the resulting pairs to the JEPA stream, while reliable-pair SFT remains the task anchor.

Table 1 summarizes performance across the proposed training curriculum. Starting from Base SFT, joint gold-pair training with AC-JEPA raises accuracy by 8.67 points on Simple and 5.80 points on Markush. This configuration reaches comparable Simple accuracy to MolParser-Base and improves Markush by 4.18 points.

Self-training R1 and R2 extend latent prediction with these pseudo-pairs while retaining the SFT anchor on clean data. R2 reaches 80.23% on Simple and 48.28% on Markush, exceeding the independently initialized gold-pair configuration by 4.16 and 6.00 points, respectively. The higher endpoint accuracies of both self-training runs show that the pseudo-pair streams benefit AC-JEPA in this OCSR setting.

Figure 4 reports component losses for the three independent runs. All start from the same CPT-trained, JEPA-warmed checkpoint with separate optimization schedules, and the juxtaposed blocks show run-local rather than sequential steps. Within each run, SFT and both modality-specific MSE losses decrease; image SIGReg also decreases, while text SIGReg remains low and bounded. These curves show joint optimization of the token-level and latent objectives; Table 1 reports endpoint accuracy.

Table 1: Accuracy (%) across the AC-JEPA curriculum on WildMol Simple and Markush. Every row in our curriculum inherits the same MolParser-7M pretraining and Real-SFT fine-tuning; gold-pair AC-JEPA reuses existing pairs through JEPA. MolParser-Base is a published reference from Fang et al. [8], with a different architecture and optimization recipe.
<table><tr><td>Method</td><td>Simple Acc. (%)</td><td>Markush Acc. (%)</td></tr><tr><td>Published reference MolParser-Base</td><td>76.90</td><td>38.10</td></tr><tr><td>Our curriculum</td><td></td><td></td></tr><tr><td>Base SFT</td><td>67.40</td><td>36.48</td></tr><tr><td>Gold-pair AC-JEPA AC-JEPA self-training, round 1</td><td>76.07 78.51</td><td>42.28</td></tr><tr><td>AC-JEPA self-training, round 2</td><td>80.23</td><td>42.78</td></tr><tr><td></td><td></td><td>48.28</td></tr></table>

(a) SFT and latent-prediction losses  
![](images/99d7ba0c77079a17447feb74380911b9b4081f9c43bc1d768464ea53ab68d50f.jpg)

(b) Modality-wise SIGReg losses  
![](images/c7e9c96fe53b8ed2390975585ef3f488e62e164854763a3f6b0af5199f7c84cb.jpg)  
Figure 4: Component losses from three independent AC-JEPA runs initialized from the same CPT-trained, JEPAwarmed checkpoint. Curves show every 100 local optimization steps for gold-pair training (6,000 steps), self-training R1 (4,500 steps), and self-training R2 (4,000 steps). The blocks are juxtaposed for compact comparison and are not connected across runs. All curves report unweighted loss components.

Reusing synthetic pairs after real-image adaptation. The shared pretraining history is important for interpreting the gold-pair gain: MolParser-7M has already contributed to Base SFT, and AC-JEPA changes how its synthetic pairs are reused after Real-SFT adaptation. In a small-scale implementation, we also used synthetic pairs from MolParser-7M alongside Real-SFT for token-level SFT. This mixture did not improve over Real-SFT-only adaptation and was more prone to overfitting. In contrast, retaining Real-SFT token supervision and reusing the synthetic pairs through AC-JEPA yields the improvements in Table 1. The pilot therefore supports the importance of the supervision route rather than an explanation based simply on adding paired examples or extending mixed token-level training. We report the pilot as qualitative design evidence, with its reporting scope specified in Section A.4.

## 4.3 Pseudo-Label Noise Analysis

Model-generated pseudo-label noise. We first compare two ways of using pseudo-labels generated for unlabeled open-environment images: direct token-level SFT and latent self-training with AC-JEPA. Both settings use modelgenerated pseudo-labels, which contain naturally occurring recognition and SMILES errors rather than manually specified corruptions.

Relative to direct pseudo-label SFT, the first AC-JEPA self-training round improves Simple and Markush accuracy by 15.04 and 6.74 points, respectively. The second round increases these margins to 16.76 and 12.24 points. These results show a clear empirical advantage for routing pseudo-pairs through latent prediction in the current training pipeline. Because the recorded runs do not yet match total training budget exactly, we treat this comparison as evidence from the realized curriculum rather than as an isolated causal comparison between loss functions

Table 2: Accuracy (%) when model-generated pseudo-labels are used for direct token-level SFT or AC-JEPA latent self-training.
<table><tr><td>Pseudo-label route</td><td>Simple</td><td>Markush</td></tr><tr><td>Direct pseudo-label SFT</td><td>63.47</td><td>36.04</td></tr><tr><td>AC-JEPA self-training, round 1</td><td>78.51</td><td>42.78</td></tr><tr><td>AC-JEPA self-training, round 2</td><td>80.23</td><td>48.28</td></tr></table>

Controlled synthetic SMILES noise. To compare the noise tolerance of direct SFT and AC-JEPA independently of the unknown error distribution of model-generated pseudo-labels, we construct a compact controlled testbed from the clean MolParser Real-SFT subset [8] and MolGrapher-Synthetic-300K [15]. We corrupt only the SMILES labels of a controlled fraction $\rho _ { \mathrm { s e q } }$ of the MolGrapher examples; their paired molecular images and all Real-SFT examples remain unchanged. For each selected MolGrapher label, the current corruption procedure perturbs 5–15% of its SMILES characters. In AC-JEPA training, supervised generation is anchored by the clean Real-SFT data, whereas the corrupted MolGrapher pairs are used only by the JEPA latent-prediction objective. In the direct-SFT control, both the clean Real-SFT pairs and the corrupted MolGrapher pairs are optimized with the token-level SFT objective. We compare these routes across increasing sequence-level noise and evaluate both on the unchanged WildMol Simple and Markush benchmarks in Table 3. As $\rho _ { \mathrm { s e q } }$ rises to 80%, direct SFT degrades sharply, whereas AC-JEPA retains gains over the clean Base SFT on both benchmarks.

Table 3: Accuracy (%) under controlled SMILES corruption. $\rho _ { \mathrm { s e q } }$ denotes the fraction of corrupted MolGrapher labels; 5–15% of characters are perturbed in each selected label.
<table><tr><td>Objective</td><td> $\rho _ { \mathrm { s e q } }$ </td><td>Simple</td><td>Markush</td></tr><tr><td>Direct SFT</td><td>20%</td><td>67.08</td><td>21.85</td></tr><tr><td>Direct SFT</td><td>50%</td><td>66.03</td><td>21.62</td></tr><tr><td>Direct SFT</td><td>80%</td><td>51.37</td><td>4.45</td></tr><tr><td>AC-JEPA joint training</td><td>20%</td><td>73.67</td><td>45.98</td></tr><tr><td>AC-JEPA joint training</td><td>50%</td><td>73.26</td><td>45.45</td></tr><tr><td>AC-JEPA joint training</td><td>80%</td><td>73.07</td><td>43.01</td></tr></table>

## 4.4 Ablation Studies

Intermediate latent interface. The JEPA objective reads an intermediate decoder layer rather than the final generation state. The current coarse sweep in Table 4 favors layer 17 among the four recorded choices, suggesting that states too close to the final decoder output are less effective as predictive targets.

Table 4: Preliminary Simple accuracy (%) for the recorded decoder-layer choices used as the JEPA latent interface.
<table><tr><td>Latent-interface layer</td><td>Simple Acc. (%)</td></tr><tr><td>12</td><td>76.07</td></tr><tr><td>17</td><td>77.67</td></tr><tr><td>23</td><td>75.61</td></tr><tr><td>26</td><td>71.31</td></tr></table>

Masking configuration. Table 5 varies the masking ratio and masked modalities under the same structured imageblock and SMILES-span corruption. Joint image–SMILES masking at ratio 0.35 performs best, while lower or higher ratios reduce Simple accuracy. Both SMILES-only and image-only masking remain above Base SFT, indicating that masked prediction in either modality still benefits the SFT task.

Table 5: Preliminary Simple accuracy (%) for the recorded masking-ratio and masked-modality variants.
<table><tr><td>Masked modalities</td><td>Ratio</td><td>Simple Acc. (%)</td></tr><tr><td>Image + SMILES</td><td>0.20</td><td>75.20</td></tr><tr><td>Image + SMILES</td><td>0.35</td><td>76.07</td></tr><tr><td>Image + SMILES</td><td>0.50</td><td>74.60</td></tr><tr><td>SMILES only</td><td>0.35</td><td>73.12</td></tr><tr><td>Image only</td><td>0.35</td><td>70.54</td></tr></table>

Predictor and latent-stability diagnostics. Additional configuration diagnostics provide empirical support for the predictor and latent-stability choices used in the final model. Replacing bidirectional prediction with causal attention substantially slowed predictive MSE convergence and yielded 72.95% Simple accuracy. Reducing the JEPA latent dimension from 1024 to 256 similarly slowed convergence and yielded 74.28% Simple accuracy, suggesting that the JEPA latent should not be aggressively compressed within this decoder architecture. Removing stop-gradient caused rapid latent collapse under the default SIGReg setting. Stronger regularization preserved non-degenerate latent dispersion, but accuracy degraded rapidly. At the default weight, pooling image and SMILES latents under one SIGReg term rapidly drove the pooled statistic toward the regularization target, and accuracy degraded substantially. With a substantially lower pooled-SIGReg weight, predictive MSE decreased slowly, but Simple accuracy remained limited to 73.24%. Together, these diagnostics support bidirectional recovery, clean-target detachment, and modality-wise regulation for learning predictable intermediate states without forcing the heterogeneous modalities into one pooled statistic.

## 5 Discussion and Limitations

AC-JEPA complements rather than replaces SFT: the causal objective anchors the output language and deployable imageto-SMILES path, while latent prediction supplies an additional representation-level signal from gold or pseudo-labeled pairs.

Synthetic data serve useful but distinct roles across the curriculum. Both Base SFT and AC-JEPA benefit from full MolParser-7M pretraining before Real-SFT adaptation; the gold-pair gain therefore does not come from access to new paired examples. Our mixed-SFT pilot indicates that returning synthetic pairs to categorical supervision is not automatically beneficial after real-image adaptation and can increase overfitting. Together with the gold-pair results, this observation supports AC-JEPA as a way to extract further predictive constraints from existing pairs while retaining a real-data token anchor. The pilot remains qualitative, and its scope should be distinguished from a systematic comparison across training budgets and synthetic/real sampling ratios.

Its noise tolerance is conditional. Continuous latent targets can soften the effect of local pseudo-label errors, but systematic bias may still shift the learned representation; our controlled-corruption results support this behavior only within the tested OCSR setting. Broader visual-to-symbolic generalization therefore requires validation on additional tasks, and the extra pseudo-label generation and training should not be interpreted as compute-free.

## 6 Conclusion

We introduced AC-JEPA, an SFT-compatible objective that learns visual-to-symbolic representations through bidirectional masked latent prediction while retaining causal token supervision. On OCSR, gold-pair AC-JEPA improves a Base SFT model that already shares complete synthetic MolParser-7M pretraining and Real-SFT adaptation, showing the usefulness of reusing existing pairs through latent prediction. The contrasting small-scale mixed-SFT observation supports retaining real-image token supervision while assigning synthetic-pair reuse to the latent objective. Latent self-training further improves performance with imperfect pseudo-labels, where the recorded direct pseudo-label SFT run degrades. These results support anchored joint-embedding prediction as a practical route to more noise-tolerant cross-modal translation.

## References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. doi: 10.1109/CVPR52729.2023.01499. URL https://openaccess.thecvf.com/ content/CVPR2023/html/Assran\_Self-Supervised\_Learning\_From\_Images\_With\_a\_ Joint-Embedding\_Predictive\_Architecture\_CVPR\_2023\_paper.html.

[2] Mido Assran et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[3] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 1298–1312. PMLR, 2022. URL https://proceedings.mlr.press/v162/baevski22a.html.

[4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[5] Randall Balestriero and Yann LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.

[6] Daniel Campos and Heng Ji. IMG2SMI: Translating molecular structure images to simplified molecular-input line-entry system. arXiv preprint arXiv:2109.04202, 2021.

[7] Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Allen Bolourchi,´ Yann LeCun, and Pascale Fung. VL-JEPA: Joint embedding predictive architecture for vision-language. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=tjimrqc2BU.

[8] Xi Fang, Jiankun Wang, Xiaochen Cai, Shangqian Chen, Shuwen Yang, Haoyi Tao, Nan Wang, Lin Yao, Linfeng Zhang, and Guolin Ke. MolParser: End-to-end visual recognition of molecule structures in the wild. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 24528–24538, 2025. URL https: //openaccess.thecvf.com/content/ICCV2025/html/Fang\_MolParser\_End-to-end\_ Visual\_Recognition\_of\_Molecule\_Structures\_in\_the\_Wild\_ICCV\_2025\_paper.html.

[9] Junxian He, Jiatao Gu, Jiajun Shen, and Marc’Aurelio Ranzato. Revisiting self-training for neural sequence generation. In International Conference on Learning Representations, 2020. URL https://openreview. net/forum?id=SJgdnAVKDH.

[10] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are´ scalable vision learners. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022. URL https://openaccess.thecvf.com/content/CVPR2022/html/ He\_Masked\_Autoencoders\_Are\_Scalable\_Vision\_Learners\_CVPR\_2022\_paper.html.

[11] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 4904–4916. PMLR, 2021. URL https://proceedings.mlr.press/v139/ jia21b.html.

[12] Dong-Hyun Lee. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In ICML Workshop on Challenges in Representation Learning, 2013.

[13] Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven C. H. Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In Advances in Neural Information Processing Systems, volume 34, pages 9694– 9705, 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/hash/ 505259756244493872b7709a8a01b536-Abstract.html.

[14] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 12888–12900. PMLR, 2022. URL https://proceedings.mlr.press/v162/li22n.html.

[15] Lucas Morin, Martin Danelljan, Maria Isabel Agea, Ahmed Nassar, Valery Weber, Ingmar Meijer, Peter Staar, and Fisher Yu. MolGrapher: Graph-based visual recognition of chemical structures. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19552–19561, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Morin\_MolGrapher\_ Graph-based\_Visual\_Recognition\_of\_Chemical\_Structures\_ICCV\_2023\_paper.html.

[16] Avishka Perera, Kumal Hewagamage, Saeedha Nazar, Kavishka Abeywardana, Hasitha Gallella, Ranga Rodrigo, and Mohamed Afham. CrossJEPA: Cross-modal joint-embedding predictive architecture for efficient 3d representation learning from 2d images. arXiv preprint arXiv:2511.18424, 2025.

[17] Yujie Qian, Jiang Guo, Zhengkai Tu, Zhening Li, Connor W. Coley, and Regina Barzilay. MolScribe: Robust molecular structure recognition with image-to-graph generation. Journal ofChemical Information and Modeling, 63(7):1925–1934, 2023. doi: 10.1021/acs.jcim.2c01480. URL https://pubs.acs.org/doi/10.1021/ acs.jcim.2c01480.

[18] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021. URL https://proceedings.mlr.press/v139/radford21a.html.

[19] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A. Raffel, Ekin D. Cubuk, Alexey Kurakin, and Chun-Liang Li. FixMatch: Simplifying semi-supervised learning with consistency and confidence. In Advances in Neural Information Processing Systems, volume 33, pages 596–608, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 06964dce9addb1c5cb5d6e3d9838f733-Abstract.html.

[20] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems, 34(11):8135–8153, 2023. doi: 10.1109/TNNLS.2022.3152527. URL https://doi.org/10.1109/TNNLS.2022.3152527.

[21] Revant Teotia, Adrien Bardes, Michael Rabbat, Sumit Chopra, Matthew J. Muckley, and Nicolas Ballas. MJEPA: A simple and scalable joint-embedding predictive architecture for audio-visual learning. arXiv preprint arXiv:2606.25225, 2026.

[22] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. Image as a foreign language: BEiT pretraining for vision and vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19175–19186, 2023. URL https://openaccess.thecvf.com/content/CVPR2023/html/Wang\_Image\_as\_ a\_Foreign\_Language\_BEiT\_Pretraining\_for\_Vision\_and\_CVPR\_2023\_paper.html.

[23] David Weininger. SMILES, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of Chemical Information and Computer Sciences, 28(1):31–36, 1988. doi: 10.1021/ ci00057a005. URL https://pubs.acs.org/doi/10.1021/ci00057a005.

[24] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improves imagenet classification. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10687–10698, 2020. URL https://openaccess.thecvf.com/content\_CVPR\_2020/html/Xie\_ Self-Training\_With\_Noisy\_Student\_Improves\_ImageNet\_Classification\_CVPR\_ 2020\_paper.html.

[25] Haote Yang, Hui Wang, Chen Zhu, Jingchao Wang, Linye Li, Hongbin Lai, Huijie Ao, Yongxuan Lyu, Jiang Wu, Jiaxing Sun, Lua Chen, Yuanyuan Cao, Ruijie Zhang, Shengxin Lu, Lijun Wu, Bin Wang, and Conghui He. MolRecBench-Wild: A real-world benchmark for optical chemical structure recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. URL https://arxiv.org/ abs/2605.05832.

[26] Wenrui Zhang, Xinggang Wang, Bin Feng, and Wenyu Liu. MolSight: Optical chemical structure recognition with SMILES pretraining, multi-granularity learning and reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence, 40(15):12825–12833, 2026. doi: 10.1609/aaai.v40i15.38280. URL https://ojs.aaai.org/index.php/AAAI/article/view/38280.

## A Appendix

## A.1 Structured Corruption Details

The main text defines the image and SMILES mask sets used by the latent MSE. For completeness, contiguous spans are sampled only within the assistant SMILES region,

$$
\begin{array} { c } { { \mathcal { M } _ { \mathrm { s m i } } = \displaystyle \bigcup _ { b = 1 } ^ { B _ { \mathrm { s m i } } } \{ s _ { b } , \dotsc , s _ { b } + \ell _ { b } - 1 \} , } } \\ { { \ell _ { b } \sim \mathrm { U n i f } [ \ell _ { \mathrm { m i n } } , \ell _ { \mathrm { m a x } } ] , } } \end{array}\tag{13}
$$

until the prescribed SMILES masking ratio is reached. Rectangular blocks are sampled on the visual patch grid,

$$
{ \mathcal { M } } _ { \mathrm { i m g } } = \bigcup _ { b = 1 } ^ { B _ { \mathrm { i m g } } } \left\{ \left( u , v \right) : { u _ { b } \leq u < u _ { b } + h _ { b } } , \right\} .\tag{14}
$$

These masks can remove all or part of local symbolic motifs and coherent image regions without asserting that every sampled mask coincides with a complete chemical substructure. Prompt, padding, and special tokens are excluded from both sets. The masks are sampled before the clean and corrupted forward passes so that both branches use the same prediction positions. If a modality has no masked positions in a micro-batch, its MSE term is skipped through a graph-valid zero.

## A.2 Target Stabilization and Gradient Routing

The clean branch produces the undetached latent $Z _ { \mathrm { c l e a n } }$ , from which the predictive target is formed as $Z _ { \mathrm { t g t } } = \mathrm { s g } ( Z _ { \mathrm { c l e a n } } )$ Consequently, $\mathcal { L } _ { \mathrm { M S E } }$ updates the corrupted-view backbone up to layer ℓ, the shared projector, and the predictor, but does not move the clean target through the MSE path. The stop-gradient does not freeze the shared backbone. The gold-pair SFT loss still updates the complete visual encoder and causal decoder, and modality-wise SIGReg is evaluated on the undetached $Z _ { \mathrm { c l e a n } }$ . Thus, stop-gradient stabilizes the clean SFT-anchored target, whereas SIGReg supplies collapse control.

The no-detach variant allows the MSE to update both clean and corrupted representations and is retained as a target-stability diagnostic. It is not treated as the default merely because the underlying LeJEPA regularizer can be used without a detached target. This distinction is specific to the present shared-weight autoregressive generator, for which a latent space that is easy to predict need not remain useful for exact SMILES decoding.

## A.3 Modality-wise SIGReg Details

SIGReg compares one-dimensional random projections of the clean latent distribution with the characteristic function of a standard normal target [5]. For modality m, let ${ \mathcal { Z } } _ { m }$ contain all eligible clean-view latents in the JEPA mini-batch and let $N _ { m } = | \mathcal { Z } _ { m } |$ |. With random unit directions

$$
\widetilde { \boldsymbol { a } } _ { p } ^ { m } \sim \mathcal { N } ( 0 , I ) , \qquad \boldsymbol { a } _ { p } ^ { m } = \widetilde { \boldsymbol { a } } _ { p } ^ { m } / \| \widetilde { \boldsymbol { a } } _ { p } ^ { m } \| _ { 2 } ,\tag{15}
$$

and frequency knots $t _ { k }$ , the modality-specific penalty is

$$
\begin{array} { r l } {  { \mathcal L _ { \mathrm { S I G R e g } } ^ { m } = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } N _ { m } \sum _ { k = 1 } ^ { K } w _ { k } } } \\ & { \times  \frac { 1 } { N _ { m } } \sum _ { z \in { \mathcal Z } _ { m } } e ^ { \mathrm { i } t _ { k } ( a _ { p } ^ { m } ) ^ { \top } z }  } \\ & {  - e ^ { - t _ { k } ^ { 2 } / 2 }  ^ { 2 } . } \end{array}\tag{16}
$$

Image and SMILES positions are pooled across examples within their respective modality, but the two modalities are never pooled into one SIGReg statistic. Because the statistic is computed from the clean branch, active mask tokens do not enter ${ \mathcal { Z } } _ { m }$ . The main-text regularizer in Equation (5) is the weighted sum of these two penalties.

## A.4 Training-Stream Details

The unlabeled pool used for R1 and R2 includes molecular images from USPTO-Mol (https://huggingface. co/datasets/xiegeo/uspto-mol) and structure figures extracted from scientific papers. The coefficients $\lambda _ { \mathrm { s f t } }$ $\lambda _ { \mathrm { m s e } } ^ { m } ,$ and $\lambda _ { \mathrm { s i g } } ^ { m }$ are fixed within a training run. SFT and JEPA are estimated from separate mini-batches drawn from $\mathcal { D } _ { \mathrm { s f t } }$ and $\mathcal { D } _ { \mathrm { j e p a } } ^ { g }$ , respectively, and may use different batch sizes; this changes the stochastic estimator but not the objectives in Equations (6) and (7). For the simplest accepted pseudo-label pool, $q ( I , { \hat { S } } ) = 1$ . Algorithm 1 makes the reported refresh protocol explicit. The common SFT-anchored, JEPA-warmed checkpoint $C _ { 0 }$ generates the first pseudo-pair pool and initializes the round-1 student. The trained round-1 checkpoint generates the second pool, but the round-2 studen is reinitialized from $C _ { 0 }$ rather than continued from the round-1 weights.

Synthetic-pair reuse pilot. Our small-scale mixed-SFT implementation sampled synthetic pairs from MolParser-7M together with Real-SFT for categorical token supervision. Relative to Real-SFT-only adaptation, it provided no improvement and was more prone to overfitting. This is a qualitative observation motivating the choice to retain real-image token supervision and reuse synthetic pairs through latent prediction. It is separate from the controlled label-corruption experiment, which uses MolGrapher-Synthetic-300K. We do not assign a numerical endpoint or claim a complete 7M-example rerun or a matched-compute comparison for this pilot.

## A.5 Supporting Analysis and Proofs for Task Readout Stability and Pseudo-Label Noise

This appendix proves the conditional-readout and parameter-influence results in the main text and makes their assump tions explicit. The first analysis holds a jointly trained model fixed, whereas the second is local to a counterfactual clean-pair solution anchored by the gold SFT and JEPA objectives. It does not assert global robustness or compare loss functions independently of representation geometry.

## A.5.1 From Symbolic JEPA Loss to Conditional Residual

Sample i from the masked SMILES positions under the training data and masking distribution. Write $Z _ { i } = Z _ { \mathrm { t g t } , i }$ $\widehat { Z } _ { i } = Z _ { \mathrm { p r e d } , i }$ , and let $C _ { i }$ contain the corrupted multimodal latent sequence and mask-position information available to the predictor. The conditional mean $m _ { i } ( C _ { i } ) = \mathbb { E } [ Z _ { i } \mid C _ { i } ]$ gives the exact orthogonal decomposition of the unweighted symbolic MSE component,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { M S E } } ^ { \mathrm { s m i } } = \mathbb { E } \Vert Z _ { i } - \widehat { Z } _ { i } \Vert _ { 2 } ^ { 2 } } \\ & { \qquad = \underbrace { \mathbb { E } \Vert Z _ { i } - m _ { i } ( C _ { i } ) \Vert _ { 2 } ^ { 2 } } _ { \overline { { R } } _ { \mathrm { J } } ^ { \mathrm { s m i } } } + \underbrace { \mathbb { E } \Vert m _ { i } ( C _ { i } ) - \widehat { Z } _ { i } \Vert _ { 2 } ^ { 2 } } _ { \mathcal { E } _ { \mathrm { p r e d } } ^ { \mathrm { s m i } } } . } \end{array}\tag{17}
$$

```latex
Algorithm 1 AC-JEPA training and pseudo-label refresh
Input: common checkpoint $C _ { 0 } ;$ reliable SFT pairs $\mathcal { D } _ { \mathrm { s f t } } ;$ gold JEPA pairs $\mathcal { D } _ { \mathrm { j e p a } } ^ { g } \mathrm { ; }$ ; unlabeled images $\mathcal { D } _ { u }$
Output: independently trained checkpoints $C _ { g } , C _ { 1 } , C _ { 2 }$
Notation: $\ell _ { \mathrm { M S E } } ^ { m }$ and $\ell _ { \mathrm { S I G R e g } } ^ { m }$ are unweighted modality-specific terms
1: procedure $\mathrm { T R A I N } \big ( C _ { 0 } , \mathcal { D } _ { p } , \eta , T \big )$
2: Restore $C _ { 0 }$ and initialize a run-specific optimizer
3: for $t = 1 , \dots , T$ do
4: Sample $B _ { s } \sim \mathcal { D } _ { \mathrm { s f t } }$ and $B _ { g } \sim D _ { \mathrm { j e p a } } ^ { g }$
5: $\mathcal { L } \gets \lambda _ { \mathrm { s f t } } \mathcal { L } _ { \mathrm { S F T } } ( B _ { s } )$
6: $\mathcal { Q }  \{ ( B _ { g } , 1 ) \}$
7: if $\mathcal { D } _ { p } \neq \ell$ then
8: Sample $B _ { p } \sim \mathcal { D } _ { p }$ and add $\left( B _ { p } , \eta q ( B _ { p } ) \right)$ to Q
9: end if
10: for each $( B , w ) \in \mathcal { Q }$ do
11: Construct clean and structured-masked views of $B$
12: $Z _ { c } \gets g _ { \phi } \big ( H _ { \theta } ^ { \ell } ( \mathbf { V } _ { \mathrm { c l e a n } } ) \big )$
13: $Z _ { t } \gets \operatorname { s g } ( Z _ { c } )$
14: $Z _ { p } \gets p _ { \psi } ( g _ { \phi } ( H _ { \theta } ^ { \ell } ( \mathbf { V } _ { \mathrm { m a s k } } ) ) )$
15: $\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e d } }  \sum _ { m } \lambda _ { \mathrm { m s e } } ^ { m } \ell _ { \mathrm { M S E } } ^ { m } ( Z _ { p } , Z _ { t } ) } \end{array}$
16: $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e g } }  \sum _ { m } \lambda _ { \mathrm { s i g } } ^ { m } \ell _ { \mathrm { S I G R e g } } ^ { m } ( Z _ { c } ) } \end{array}$
17: ${ \mathcal { L } } \gets { \mathcal { L } } + w ( { \mathcal { L } } _ { \mathrm { p r e d } } + { \mathcal { L } } _ { \mathrm { r e g } } )$
18: end for
19: Update $( \theta , \phi , \psi )$ using $\nabla \mathcal { L }$
20: end for
21: return the trained checkpoint
22: end procedure
23: $C _ { g } \gets \mathrm { T R A I N } ( C _ { 0 } , \emptyset , 0 , T _ { g } )$
24: $\mathcal { D } _ { p } ^ { ( 1 ) }  \{ ( I ,$ Decode $( C _ { 0 } , I ) ) : I \in \mathcal { D } _ { u } \}$
25: $C _ { 1 } \gets \mathrm { T R A I N } \big ( C _ { 0 } , \mathcal { D } _ { p } ^ { ( 1 ) } , \eta _ { 1 } , T _ { 1 } \big )$
26: ${ \mathcal { D } } _ { p } ^ { ( 2 ) } \gets \{ ( I , \mathrm { D e c o d e } ( C _ { 1 } , I ) ) : I \in { \mathcal { D } } _ { u } \}$
27: $C _ { 2 } \gets \mathrm { T R A I N } \big ( C _ { 0 } , \mathcal { D } _ { p } ^ { ( 2 ) } , \eta _ { 2 } , T _ { 2 } \big )$
28: return $C _ { g } , C _ { 1 } , C _ { 2 }$
```

The cross term vanishes because $\mathbb { E } [ Z _ { i } - m _ { i } ( C _ { i } ) \mid C _ { i } ] = 0$ . Thus $\overline { { R } } _ { \mathrm { J } } ^ { \mathrm { s m i } }$ is the Bayes residual of symbolic latent prediction and $\mathcal { E } _ { \mathrm { p r e d } } ^ { \mathrm { s m i } }$ measures predictor approximation and optimization error. Stop-gradient changes the update path through $Z _ { i }$ but not this forward-risk identity.

Consider two checkpoints evaluated under the same data and masking distribution. Let $\Delta _ { \mathrm { J } } = \mathcal { L } _ { \mathrm { M S E , 0 } } ^ { \mathrm { s m i } } - \mathcal { L } _ { \mathrm { M S E , 1 } } ^ { \mathrm { s m i } } ,$ and assume that predictor warm-up gives $\mathcal { E } _ { \mathrm { p r e d } , 0 } ^ { \mathrm { s m i } } \leq \epsilon _ { \mathrm { p } }$ . Since predictor excess risk is nonnegative, Equation (17) yields

$$
\begin{array} { r } { \overline { { R } } _ { \mathrm { J } , 0 } ^ { \mathrm { s m i } } - \overline { { R } } _ { \mathrm { J } , 1 } ^ { \mathrm { s m i } } = \Delta _ { \mathrm { J } } - \mathcal { E } _ { \mathrm { p r e d } , 0 } ^ { \mathrm { s m i } } + \mathcal { E } _ { \mathrm { p r e d } , 1 } ^ { \mathrm { s m i } } \geq \Delta _ { \mathrm { J } } - \epsilon _ { \mathrm { p } } . } \end{array}\tag{18}
$$

Hence a symbolic-MSE decrease larger than the initial predictor excess reduces the Bayes latent residual. The statement concerns the population objective at the two checkpoints; it does not require every stochastic update to decrease the residual.

To connect this average residual to Theorem 3.1, define $m _ { i , s ^ { + } } ( C _ { i } ) = \mathbb { E } [ Z _ { i } \mid C _ { i } , S _ { i + 1 } = s ^ { + } ]$ . The law of total variance gives

$$
\begin{array} { r l } & { \overline { { R } } _ { \mathrm { J } } ^ { \mathrm { s m i } } = \mathbb { E } \| Z _ { i } - m _ { i , S _ { i + 1 } } ( C _ { i } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad + \mathbb { E } \| m _ { i , S _ { i + 1 } } ( C _ { i } ) - m _ { i } ( C _ { i } ) \| _ { 2 } ^ { 2 } . } \end{array}\tag{19}
$$

Under the locally affine latent interface $Z _ { i } = J _ { i } H _ { i } + a _ { i }$ , the first term equals E $[ R _ { \mathrm { J } , i } ( C _ { i } ) ] _ { \mathrm { j } }$ ; for a nonlinear interface this equality holds to first order with the corresponding Taylor remainder. Suppose the overlap constant κ is uniform over the analyzed strata and the SFT anchor maintains $\mu _ { i , s ^ { - } } ( C _ { i } ) \geq \mu _ { 0 } > 0$ for a fixed competitor assignment. Applying Theorem 3.1, averaging over contexts and next-token classes, and using concavity of $x / ( x + \mu _ { 0 } ^ { 2 } )$ gives

$$
\mathrm { P r } [ \Delta _ { i , s ^ { - } } ( H _ { i } ) \leq 0 ] \leq \frac { \kappa \overline { { R } } _ { \mathrm { J } } ^ { \mathrm { s m i } } } { \kappa \overline { { R } } _ { \mathrm { J } } ^ { \mathrm { s m i } } + \mu _ { 0 } ^ { 2 } } .\tag{20}
$$

Together, Equations (18) and (20) connect a decrease in the actual symbolic JEPA objective to a tighter averaged readout bound under predictor tracking, local decoder–JEPA overlap, and SFT margin preservation. Modality-wise SIGReg excludes the trivial collapsed route to a small latent residual but does not replace these conditions. Under these conditions, the SFT and JEPA objectives play complementary roles. SFT maintains the class-conditional mean margin, while a reduction in symbolic JEPA risk reduces context-unpredictable latent variation. Decoder–JEPA overlap converts this reduction into a tighter bound on the probability that hidden-state fluctuations cross the decision boundary.

Image-to-symbol transfer. The preceding argument uses symbolic MSE because the corresponding latent state directly supports the causal decoder, but image-side prediction can also reduce symbolic decision variation through cross-modal fusion. Let V and S denote image- and symbol-side intermediate states and locally write the fused symbolic state as $H _ { i } = F _ { i } ( V , S )$ . Around the conditional mean, its centered residual admits the first-order decomposition

$$
U _ { i } = A _ { i } U _ { \mathrm { i m g } } + B _ { i } U _ { \mathrm { s m i } } + r _ { i } ,\tag{21}
$$

where $A _ { i }$ is the image-to-symbol Jacobian induced by cross-modal interaction, $B _ { i }$ is the symbol-side Jacobian, and $r _ { i }$ is the linearization remainder. For modality-specific latent interfaces $J _ { \mathrm { i m g } }$ and $J _ { \mathrm { s m i } }$ , define $R _ { \mathrm { J } } ^ { m } = \mathbb { E } [ \| J _ { m } U _ { m } \| _ { 2 } ^ { 2 } \ |$ $C _ { i } , S _ { i + 1 } = s ^ { + } \mathrm { ] . \ I f \ } A _ { i } ^ { \top } d _ { i , s ^ { - } } d _ { i , s ^ { - } } ^ { \top } A _ { i } \preceq \kappa _ { \mathrm { i m g } } J _ { \mathrm { i m g } } ^ { \top } \bar { J } _ { \mathrm { i m g } }$ and the analogous symbol-side overlap holds with $\kappa _ { \mathrm { s m i } }$ , then

$$
v _ { i , s ^ { - } } ( C _ { i } ) \le \kappa _ { \mathrm { i m g } } R _ { \mathrm { J } } ^ { \mathrm { i m g } } + \kappa _ { \mathrm { s m i } } R _ { \mathrm { J } } ^ { \mathrm { s m i } } + \epsilon _ { \mathrm { c r o s s } } + \epsilon _ { \mathrm { l i n } } ,\tag{22}
$$

where $\epsilon _ { \mathrm { c r o s s } }$ bounds cross-modal covariance terms and $\epsilon _ { \mathrm { l i n } }$ bounds the contribution of $r _ { i }$ . Applying the decomposition in Equation (17) to masked image positions gives the analogous image-side loss–residual link; connecting its population average to $R _ { \mathrm { J } } ^ { \mathrm { i m g } }$ additionally requires the paired sampling and local transfer assumptions above. Thus image-side stabilization can tighten the decision-reversal bound in Theorem 3.1 when cross-modal transfer reaches the decoder relevant subspace, the remaining terms are controlled, and SFT preserves the margin. With locally fixed attention scores, $A _ { i }$ reduces to attention-weighted value transfer; score variation is absorbed into the remainder. The fact that both modalities describe the same entity makes the required alignment plausible but does not guarantee it: no improvement follows if the decoder ignores the image subspace or if correlated or systematic errors dominate. Because $C _ { i }$ and $H _ { i }$ in Theorem 3.1 are already multimodal, this decomposition makes an implicit pathway explicit rather than introducing a separate guarantee.

ProofofTheorem 3.1. Fix $i \in \mathcal { M } _ { \mathrm { s m i } }$ and condition on $( C _ { i } , S _ { i + 1 } = s ^ { + } )$ . The state $H _ { i }$ is aligned with $S _ { i }$ , whereas the causal readout at the same decoder position predicts $S _ { i + 1 }$ . With the other causal states fixed, write $U _ { i } = H _ { i } - \mathbb { E } [ H _ { i } \ |$ $C _ { i } , S _ { i + 1 } = s ^ { + } ]$ ]. Under the locally affine readout,

$$
\begin{array} { r l } & { \Delta _ { i , s ^ { - } } ( H _ { i } ) - \mu _ { i , s ^ { - } } ( C _ { i } ) = d _ { i , s ^ { - } } ^ { \top } U _ { i } , } \\ & { \qquad v _ { i , s ^ { - } } ( C _ { i } ) = \mathbb { E } \Big [ ( d _ { i , s ^ { - } } ^ { \top } U _ { i } ) ^ { 2 } \Big | _ { S _ { i + 1 } = s ^ { + } } \Big ] . } \end{array}\tag{23}
$$

Cantelli’s inequality gives

$$
\operatorname* { P r } \Bigl [ \Delta _ { i , s ^ { - } } ( H _ { i } ) \leq 0 \Bigl | \operatorname* { \ c { } C } _ { i + 1 } = s ^ { + } \Bigr ] \leq \frac { v _ { i , s ^ { - } } ( C _ { i } ) } { v _ { i , s ^ { - } } ( C _ { i } ) + \mu _ { i , s ^ { - } } ( C _ { i } ) ^ { 2 } } .\tag{24}
$$

Moreover, the decoder–JEPA overlap condition $d _ { i , s ^ { - } } d _ { i , s ^ { - } } ^ { \top } \preceq \kappa J _ { i } ^ { \top } J _ { i }$ implies

$$
v _ { i , s ^ { - } } ( C _ { i } ) = \mathbb { E } [ U _ { i } ^ { \top } d _ { i , s ^ { - } } d _ { i , s ^ { - } } ^ { \top } U _ { i } \mid C _ { i } , S _ { i + 1 } = s ^ { + } ]\tag{25}
$$

$$
\leq \kappa \mathbb { E } [ \| J _ { i } U _ { i } \| _ { 2 } ^ { 2 } \mid C _ { i } , S _ { i + 1 } = s ^ { + } ] .\tag{26}
$$

The function $v / ( v + \mu ^ { 2 } )$ is increasing in v and decreasing in positive $\mu ,$ which proves the final comparison. For a nonlinear JEPA interface, $J _ { i } U _ { i }$ is the first-order projected residual; the theorem therefore applies only where the local linearization and overlap condition are valid. □

## A.5.2 Pseudo-label Residuals and Parameter Influence

The second result compares two training problems that use the same images and differ only in the symbolic sequence attached to each pseudo-pair. Let

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { c l e a n } } ( \theta ) = \mathcal { L } _ { \mathrm { a n c h o r } } ( \theta ) + \frac { \lambda } { n } \sum _ { j = 1 } ^ { n } \ell _ { j } ( \theta ; S _ { j } ) , } \\ { \displaystyle } \\ { \mathcal { L } _ { \mathrm { p s e u d o } } ( \theta ) = \mathcal { L } _ { \mathrm { a n c h o r } } ( \theta ) + \frac { \lambda } { n } \sum _ { j = 1 } ^ { n } \ell _ { j } ( \theta ; \widehat { S } _ { j } ) , } \\ { \displaystyle \ell _ { j } ( \theta ; S ) = \frac { 1 } { 2 } \| r _ { j } ( \theta ; S ) \| _ { 2 } ^ { 2 } . } \end{array}\tag{27}
$$

$\mathcal { L } _ { \mathrm { a n c h o r } }$ contains the gold-pair SFT and JEPA streams, and $\theta ^ { \star }$ is a local stationary point of $\mathcal { L } _ { \mathrm { c l e a n } }$ . The unobserved sequences $S _ { j }$ define a counterfactual reference and are not required by the training algorithm. Using the same images in both objectives separates pseudo-label corruption from the useful signal that can arise simply from adding open-environment diagrams. The factor $1 / 2$ only absorbs the conventional factor of two into λ.

For a fixed image, mask realization, and predicted latent position, write

$$
\begin{array} { r l } & { r _ { j } ^ { c } = Z _ { \mathrm { p r e d } , j } ( \theta ^ { \star } ; S _ { j } ) - Z _ { \mathrm { t g t } , j } ( \theta ^ { \star } ; S _ { j } ) , } \\ & { r _ { j } ^ { \flat } = Z _ { \mathrm { p r e d } , j } ( \theta ^ { \star } ; \tilde { S } _ { j } ) - Z _ { \mathrm { t g t } , j } ( \theta ^ { \star } ; \tilde { S } _ { j } ) , } \\ & { e _ { j } = r _ { j } ^ { \flat } - r _ { j } ^ { \flat } = e _ { j } ^ { \mathrm { t x } } - e _ { j } ^ { \mathrm { t g t } } , } \\ & { e _ { j } ^ { \mathrm { c t x } } = Z _ { \mathrm { p r e d } , j } ( \theta ^ { \star } ; \tilde { S } _ { j } ) } \\ & { \qquad - Z _ { \mathrm { p r e d } , j } ( \theta ^ { \star } ; S _ { j } ) , } \\ & { e _ { j } ^ { \mathrm { t g t } } = Z _ { \mathrm { t g t } , j } ( \theta ^ { \star } ; \tilde { S } _ { j } ) } \\ & { \qquad - Z _ { \mathrm { t g t } , j } ( \theta ^ { \star } ; S _ { j } ) . } \end{array}\tag{28}
$$

This decomposition is not a binary assignment of an error to the context or target route. A changed pseudo-token can affect the visible predictor context in one mask realization and the clean target states of other positions in the same example. The index $j$ may therefore include sampling over mask realizations and masked positions.

Let $A _ { j } = \partial r _ { j } ( \theta ; S _ { j } ) / \partial \theta | _ { \theta } ,$ ⋆ and $\begin{array} { r } { \widehat { A } _ { j } = \partial r _ { j } ( \theta ; \widehat { S } _ { j } ) / \partial \theta | _ { \theta } , } \end{array}$ ⋆. These derivatives follow the implemented MSE path, so the detached target contributes to the value of $e _ { j }$ but not to either residual Jacobian. The exact gradient difference for the squared residual is

$$
\begin{array} { r l r } & { } & { \nabla _ { \theta } \ell _ { j } ( \theta ^ { \star } ; \widehat { S } _ { j } ) - \nabla _ { \theta } \ell _ { j } ( \theta ^ { \star } ; S _ { j } ) = \widehat { A } _ { j } ^ { \top } r _ { j } ^ { \mathrm { p } } - A _ { j } ^ { \top } r _ { j } ^ { \mathrm { c } } } \\ & { } & { = A _ { j } ^ { \top } e _ { j } + \rho _ { j } , } \\ & { } & { \rho _ { j } = ( \widehat { A } _ { j } - A _ { j } ) ^ { \top } r _ { j } ^ { \mathrm { p } } . } \end{array}\tag{29}
$$

The leading term pulls the explicit latent residual perturbation back to parameter space. The remainder $\rho _ { j }$ records the change of the predictor Jacobian caused by the pseudo sequence and becomes small when the local shared-Jacobian approximation is accurate or the perturbed residual is small.

Let $K = \nabla _ { \theta } ^ { 2 } \mathcal { L } _ { \mathrm { c l e a n } } ( \theta ^ { \star } )$ be positive definite on the local parameter subspace under analysis, and set $u = \theta - \theta ^ { \star }$ Neglecting $\rho _ { j }$ and the change in curvature between the two objectives gives

$$
\mathcal { L } _ { \mathrm { p s e u d o } } ( \theta ^ { \star } + u ) - \mathcal { L } _ { \mathrm { p s e u d o } } ( \theta ^ { \star } ) \approx \frac { 1 } { 2 } u ^ { \top } K u + \frac { \lambda } { n } \sum _ { j = 1 } ^ { n } e _ { j } ^ { \top } A _ { j } u .\tag{30}
$$

Its stationary displacement is

$$
u _ { \mathrm { p s e u d o } } = - \frac { \lambda } { n } K ^ { - 1 } \sum _ { j = 1 } ^ { n } A _ { j } ^ { \top } e _ { j } .\tag{31}
$$

This is a change in the shared model parameters, not an independently optimized hidden state. Its effect on any clean example is mediated by that example’s parameter-to-state Jacobian.

ProofofTheorem 3.2. For a clean evaluation context x, define $J _ { x } = \partial H _ { x } / \partial \theta | _ { \theta } ,$ and $d _ { x , y k } = \nabla _ { H _ { x } } G _ { x , y k } ( \theta ^ { \star } )$ . The corresponding parameter-space margin gradient is $q _ { x , y k } = J _ { x } ^ { \top } d _ { x , y k }$ . Combining the first-order margin expansion with Equation (31) gives

$$
\begin{array} { l } { { G _ { x , y k } ^ { \mathrm { l i n } } = m _ { x , y k } + q _ { x , y k } ^ { \top } u _ { \mathrm { p s e u d o } } } } \\ { { \ \quad \ = m _ { x , y k } - \displaystyle \frac { \lambda } { n } \sum _ { j = 1 } ^ { n } d _ { x , y k } ^ { \top } J _ { x } K ^ { - 1 } A _ { j } ^ { \top } e _ { j } } } \\ { { \ \ \quad \ = m _ { x , y k } - \lambda \overline { { \zeta } } _ { x , y k } , \quad \quad \ \overline { { \zeta } } _ { x , y k } = \displaystyle \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \zeta _ { x , y k , j } . } } \end{array}\tag{32}
$$

Under the theorem assumptions, $\mathbb { E } [ \bar { \zeta } _ { x , y k } ] = \beta _ { x , y k }$ and $\mathrm { V a r } ( \overline { { \zeta } } _ { x , y k } ) \leq \sigma _ { x , y k } ^ { 2 } / n$ . When $m _ { x , y k } > \lambda \beta _ { x , y k }$ , boundary crossing requires

$$
\overline { { \zeta } } _ { x , y k } - \beta _ { x , y k } \geq \frac { m _ { x , y k } - \lambda \beta _ { x , y k } } { \lambda } .\tag{33}
$$

Cantelli’s inequality therefore yields

$$
\mathrm { P r } [ G _ { x , y k } ^ { \mathrm { l i n } } \le 0 ] \le \frac { \sigma _ { x , y k } ^ { 2 } / n } { \sigma _ { x , y k } ^ { 2 } / n + ( m _ { x , y k } / \lambda - \beta _ { x , y k } ) ^ { 2 } } ,\tag{34}
$$

which is equivalent to Equation (12).

For a multi-class next-token decision, applying the union bound over $k \in \mathcal { V } _ { \mathrm { d e c } } \setminus \{ y \}$ gives

$$
\operatorname* { P r } [ \operatorname * { a r g m a x } _ { s } a _ { x , s } ( \theta ) \neq y ] \leq \sum _ { k \neq y } \operatorname* { P r } [ G _ { x , y k } ( \theta ) \leq 0 ] .\tag{35}
$$

Each term uses its own clean margin, output-influence bias, and variance; no independence across competitors is required.

## Approximation terms and SIGReg.

Equation (29) makes the shared-Jacobian approximation explicit. Retaining its remainder changes Equation (31) by $- \lambda K ^ { - 1 } n ^ { - 1 } \sum _ { j } \rho _ { j }$ . The pseudo sequence also changes the clean latents entering modality-wise SIGReg. If $g _ { \mathrm { s i g } }$ denotes the difference between the pseudo-sequence and true-sequence SIGReg gradients, the total additional displacement is

$$
u _ { \mathrm { r e m } } = - K ^ { - 1 } \left( { \frac { \lambda } { n } } \sum _ { j = 1 } ^ { n } \rho _ { j } + g _ { \mathrm { s i g } } \right) .\tag{36}
$$

We keep these terms separate from $e _ { j }$ because they do not describe a latent residual perturbation. They can be incorporated into Theorem 3.2 by adding $q _ { x , y k } ^ { \top } u _ { \mathrm { r e m } }$ to the margin and controlling its bias and variance. SIGReg can contribute stabilizing curvature to $K$ , but its pseudo-label-dependent gradient is not assumed to be zero.

The nonlinear parameter-to-margin map contributes a Taylor remainder. If the combined scalar remainder in $G _ { x , y k } = m _ { x , y k } - \lambda \overline { { \zeta } } _ { x , y k } + \mathcal { R } _ { x , y k }$ satisfies $| \mathcal { R } _ { x , y k } | \le \tau _ { x , y k }$ , the same argument gives

$$
\operatorname* { P r } [ G _ { x , y k } \leq 0 ] \leq \frac { \lambda ^ { 2 } \sigma _ { x , y k } ^ { 2 } / n } { \lambda ^ { 2 } \sigma _ { x , y k } ^ { 2 } / n + ( m _ { x , y k } - \tau _ { x , y k } - \lambda \beta _ { x , y k } ) ^ { 2 } }\tag{37}
$$

whenever $m _ { x , y k } > \tau _ { x , y k } + \lambda \beta _ { x , y k }$

Anchor curvature. The parameter-space curvature makes the roles of SFT and JEPA explicit. Let $W _ { i } = \partial a _ { i } / \partial \theta$ be the Jacobian of the clean SFT logits $a _ { i }$ and let $p _ { i }$ be their token distribution. The cross-entropy Gauss–Newton term is

$$
K _ { \mathrm { S F T } , i } = W _ { i } ^ { \top } [ \mathrm { D i a g } ( p _ { i } ) - p _ { i } p _ { i } ^ { \top } ] W _ { i } \succeq 0 .\tag{38}
$$

If $B _ { j }$ is the parameter Jacobian of a clean JEPA residual, its squared loss contributes $B _ { j } ^ { \top } B _ { j }$ . A damped local Gauss–Newton model on an identifiable parameter subspace U is

$$
\begin{array} { c } { { K \approx \displaystyle \alpha _ { \mathrm { S } } \sum _ { i } K _ { \mathrm { S F T } , i } + \alpha _ { \mathrm { J } } \sum _ { j } B _ { j } ^ { \top } B _ { j } } } \\ { { { } } } \\ { { + K _ { \mathrm { r e g } } + \mu P _ { \mathcal { U } } , } } \\ { { K _ { \mathrm { r e g } } \succeq 0 , \qquad \mu > 0 . } } \end{array}\tag{39}
$$

where $K _ { \mathrm { r e g } }$ is a positive-semidefinite local approximation to regularization curvature and $P _ { \cal { U } }$ projects onto $\mathcal { U } .$ The restriction is necessary because neural-network parameterizations contain flat and symmetry-related directions. The theorem depends on curvature only along the pulled-back pseudo-error and evaluation-margin directions, rather than on global positive definiteness of the full network Hessian. With mean-reduced objectives, more clean samples do not automatically scale $K \colon$ ; they can nevertheless improve curvature coverage and reduce its estimation variance.

Bias, variance, and dependence. The harmful bias is

$$
\beta _ { x , y k } = \mathbb { E } [ d _ { x , y k } ^ { \top } J _ { x } K ^ { - 1 } A _ { j } ^ { \top } e _ { j } ] ,\tag{40}
$$

which depends jointly on the latent residual perturbation, the training Jacobian, the anchored curvature, and the evaluation readout. Consequently, $\mathbb { E } [ e _ { j } ] = 0$ is not sufficient when $A _ { j }$ and $e _ { j }$ are dependent. The stronger condition $\mathbb { E } [ e _ { j } \ | \ A _ { j } ] = 0$ implies $\beta _ { x , y k } = 0$ for fixed $x , K ,$ , and $\theta ^ { \star }$ , but model-generated pseudo-labels need not satisfy it. The bias–variance statement in Theorem 3.2 therefore retains $\beta _ { x , y k }$ instead of assuming exact centering. If pseudo constraints are correlated, the variance of their average becomes $\begin{array} { r } { n ^ { - 2 } \sum _ { j , j ^ { \prime } } \mathrm { C o v } ( \zeta _ { j } , \zeta _ { j ^ { \prime } } ) } \end{array}$ and need not decay as $1 / n$

Single-error comparison with direct pseudo-SFT. For one erroneous pseudo-token, let $\widetilde { y }$ be the erroneous alternative, $G _ { y \widetilde { y } } ( \theta ^ { \star } ) = m > 0$ , and $q = \nabla _ { \theta } G _ { y \widetilde { y } } ( \theta ^ { \star } )$ . One gradient step of size $\rho$ on the pseudo-JEPA perturbation relative to its clean-pair counterpart gives

$$
\theta _ { \mathrm { J } } ^ { \prime } = \theta ^ { \star } - \rho \lambda A ^ { \top } e , \qquad m _ { \mathrm { J } } ^ { \prime } \approx m - \rho \lambda q ^ { \top } A ^ { \top } e .\tag{41}
$$

In a binary local readout, let $p _ { y } = \mathrm { s i g m o i d } ( m )$ . Cross-entropy toward $\widetilde { y }$ has gradient $p _ { y } q ,$ , so direct pseudo-SFT gives

$$
\theta _ { \mathrm { S } } ^ { \prime } = \theta ^ { \star } - \rho \lambda p _ { y } q , \qquad m _ { \mathrm { S } } ^ { \prime } \approx m - \rho \lambda p _ { y } \| q \| _ { 2 } ^ { 2 } .\tag{42}
$$

Thus the wrong categorical target decreases the correct-versus-wrong margin in this local model. The pseudo-JEPA effect instead depends on the signed alignment $q ^ { \top } A ^ { \top } \epsilon$ e and can either decrease or increase that margin. For the corresponding binary probability,

$$
| p _ { y } ^ { \mathrm { J } } - p _ { y } | \leq \frac { \rho \lambda } { 4 } | q ^ { \intercal } A ^ { \intercal } e | \leq \frac { \rho \lambda } { 4 } \| A q \| _ { 2 } \| e \| _ { 2 } .\tag{43}
$$

With the clean anchor retained, the local pseudo-JEPA equilibrium is $u _ { \mathrm { J } } = - \lambda K ^ { - 1 } A ^ { \top } e$ . A quadratic expansion of the erroneous pseudo-SFT term gives

$$
\boldsymbol { u } _ { \mathrm { S } } \approx - \lambda p _ { y } [ K + \lambda p _ { y } ( 1 - p _ { y } ) q q ^ { \top } ] ^ { - 1 } \boldsymbol { q } .\tag{44}
$$

The bracketed matrix is positive definite on $u ,$ and $q ^ { \top } u _ { \mathrm { S } } < 0$ . This comparison does not order the magnitudes of all JEPA and SFT errors. It identifies their different routes to the output boundary: pseudo-SFT directly optimizes an erroneous categorical preference, whereas pseudo-JEPA first induces a continuous residual perturbation whose effect is filtered by $A , K ^ { - 1 }$ , and the clean readout direction.

## A.5.3 Multimodal Evidence and Local Readout Stability

Theorem 3.1 relates decoder-relevant latent variation to next-token stability, but it does not quantify how evidence from multiple modalities changes that variation. We consider M co-observed modalities that describe the same underlying

entity or state. For modality $m ,$ let $\mathcal { G } _ { m }$ contain its reliable, target-relevant entries, and let $\mathcal { N }$ contain the visible noisy entries across the multimodal context. Under a locally fixed attention map with score $\alpha _ { k }$ and temperature $\tau > 0$ , define

$$
\begin{array} { r } { Z _ { m } = \displaystyle \sum _ { g \in \mathcal { G } _ { m } } e ^ { \alpha _ { g } / \tau } , \quad Z _ { N } = \displaystyle \sum _ { j \in \mathcal { N } } e ^ { \alpha _ { j } / \tau } , \quad } \\ { Z _ { \mathcal { G } } ^ { ( M ) } = \displaystyle \sum _ { m = 1 } ^ { M } Z _ { m } , \quad \quad \Delta _ { M } = \log \frac { Z _ { \mathcal { G } } ^ { ( M ) } } { Z _ { N } } . } \end{array}\tag{45}
$$

Let $a _ { k } \propto e ^ { \alpha _ { k } / \tau }$ be the corresponding attention weights. The total noisy attention mass and its effective support are

$$
\beta _ { M } = \sum _ { j \in \mathcal { N } } a _ { j } = \frac { 1 } { 1 + e ^ { \Delta _ { M } } } , \qquad n _ { \mathrm { e f f } } = \frac { \beta _ { M } ^ { 2 } } { \sum _ { j \in \mathcal { N } } a _ { j } ^ { 2 } } .\tag{46}
$$

Suppose only the noisy values are perturbed, $\widetilde { v } _ { j } = v _ { j } + \xi _ { j } \ \mathrm { f o r } \ j \in \mathcal { N } _ { \mathrm { } } .$ , where the $\xi _ { j }$ are conditionally independent, centered, and satisfy $\operatorname { C o v } ( \xi _ { j } \mid C _ { i } , s ^ { + } ) \preceq \sigma _ { v } ^ { 2 } I$ . Let their attention-output perturbation be $\begin{array} { r } { \widetilde { O } _ { M } - O _ { M } ^ { \star } = \sum _ { j \in \mathcal { N } } a _ { j } \xi _ { j } } \end{array}$ . We locally decompose the correct-versus-competitor margin as

$$
\widetilde { \Delta } _ { i , s ^ { - } } = \mu _ { i , s ^ { - } } + \varepsilon _ { 0 } + w ^ { \top } ( \widetilde { \sigma } _ { M } - \sigma _ { M } ^ { \star } ) ,\tag{47}
$$

where $\mu _ { i , s ^ { - } } > 0 , \varepsilon _ { 0 }$ is conditionally centered and uncorrelated with the $\xi _ { j }$ , and $\operatorname { V a r } ( \varepsilon _ { 0 } \mid C _ { i } , s ^ { + } ) \leq v _ { 0 }$

Corollary A.1 (Multimodal evidence tightens the local readout bound). Under the conditions above, define

$$
B _ { M } ^ { \mathrm { v i s } } = \frac { \Vert w \Vert _ { 2 } ^ { 2 } \sigma _ { v } ^ { 2 } } { n _ { \mathrm { e f f } } ( 1 + e ^ { \Delta _ { M } } ) ^ { 2 } } .\tag{48}
$$

Then

$$
\operatorname* { P r } [ \widetilde { \Delta } _ { i , s ^ { - } } \leq 0  { \lvert { \cal C } _ { i } , s ^ { + } \rvert } \leq \frac { v _ { 0 } + B _ { M } ^ { \mathrm { v i s } } } { v _ { 0 } + B _ { M } ^ { \mathrm { v i s } } + \mu _ { i , s ^ { - } } ^ { 2 } } .\tag{49}
$$

If an additional modality contributes reliable evidence mass $Z _ { M + 1 } > 0$ while the existing scores, noisy entries, and local readout map remainfixed, then

$$
\frac { B _ { M + 1 } ^ { \mathrm { v i s } } } { B _ { M } ^ { \mathrm { v i s } } } = \left[ \frac { 1 + e ^ { \Delta _ { M } } } { 1 + e ^ { \Delta _ { M } } ( 1 + Z _ { M + 1 } / Z _ { \mathcal { G } } ^ { ( M ) } ) } \right] ^ { 2 } < 1 .\tag{50}
$$

If every modality supplies $Z _ { m } \geq Z _ { \mathrm { m i n } } > 0 ,$ , then

$$
B _ { M } ^ { \mathrm { v i s } } \leq \frac { \| w \| _ { 2 } ^ { 2 } \sigma _ { v } ^ { 2 } } { n _ { \mathrm { e f f } } } \left( 1 + \frac { M Z _ { \mathrm { m i n } } } { Z _ { \mathcal { N } } } \right) ^ { - 2 } .\tag{51}
$$

Thus the local decision-reversal bound tightens monotonically with M when added modalities preserve the mean margin, do not increase $v _ { 0 } ,$ , and contribute reliable evidence under the fixed local map.

The corollary characterizes multimodal redundancy as evidence quality rather than modality count alone. Images, audio, language, numerical sensor streams, and radar measurements can provide complementary observations of the same underlying state. When each added modality contributes aligned evidence, their aggregate partition mass increases $\Delta _ { M }$ and reduces the variance transmitted from visible noisy entries. This is the precise sense in which additional modalities can strengthen mutual verification and improve local noise tolerance.

The monotonic result is conditional. A redundant, mismatched, systematically biased, or ignored modality need not preserve the mean margin or contribute reliable evidence mass. The analysis also concerns a local next-token decision under fixed attention scores and centered independent perturbations. It therefore does not guarantee that adding modalities monotonically improves full-sequence exact-match accuracy. The image–symbol setting studied by AC-JEPA is the $M = 2$ instance of this more general mechanism.

Proof. Conditional centering makes the attention-output perturbation mean-zero. Write $\Sigma _ { j } = \operatorname { C o v } ( \xi _ { j } \mid C _ { i } , s ^ { + } )$ . For any w,

$$
\begin{array} { r l } { \mathrm { V a r } \big [ { \boldsymbol w ^ { \top } } ( \widetilde { \boldsymbol \sigma } _ { M } - \boldsymbol \sigma _ { M } ^ { \star } ) \bigm | C _ { i } , s ^ { + } \bigm ] = \displaystyle \sum _ { j \in \mathcal { N } } a _ { j } ^ { 2 } { \boldsymbol w ^ { \top } } \Sigma _ { j } { \boldsymbol w } } & { } \\ { \le \| { \boldsymbol w } \| _ { 2 } ^ { 2 } \sigma _ { v } ^ { 2 } \displaystyle \sum _ { j \in \mathcal { N } } a _ { j } ^ { 2 } } & { } \\ { = \frac { \| { \boldsymbol w } \| _ { 2 } ^ { 2 } \sigma _ { v } ^ { 2 } \beta _ { M } ^ { 2 } } { n _ { \mathrm { e f f } } } = B _ { M } ^ { \mathrm { v i s } } . } \end{array}\tag{52}
$$

Adding the uncorrelated residual variance $v _ { 0 }$ and applying the first inequality in Theorem 3.1 proves Equation (49).

Adding modality $M + 1$ replaces $Z _ { \mathcal { G } } ^ { ( M ) }$ by $Z _ { \mathcal { G } } ^ { ( M ) } + Z _ { M + 1 }$ and therefore gives

$$
\Delta _ { M + 1 } = \Delta _ { M } + \log \left( 1 + \frac { Z _ { M + 1 } } { Z _ { \mathcal { G } } ^ { ( M ) } } \right) .\tag{53}
$$

All noisy attention weights are scaled by the same factor, so $n _ { \mathrm { e f f } }$ remains unchanged. Substitution into Equation (48) yields Equation (50). Finally, $\begin{array} { r } { Z _ { \mathcal { G } } ^ { ( M ) } = \sum _ { m } Z _ { m } \ge M Z _ { \operatorname* { m i n } } } \end{array}$ , which proves Equation (51). □