# ISOTROPIC YET UNDECODABLE: THE SEQUENTIAL CONTENT-SUFFICIENCY GAP IN LATENT-PREDICTIVE TEXT REPRESENTATIONS

Kevin Putra Santoso<sup>1∗,2,4</sup>, Naufal Zafran Fadil<sup>3</sup>, Firania Putri Harsanti<sup>4</sup> Raden Venantius Hari Ginardi<sup>2</sup>, Ganesh Neelakanta Iyer<sup>1</sup>

<sup>1</sup>School of Computing, National University of Singapore

<sup>2</sup>Department of Information Technology, Institut Teknologi Sepuluh Nopember

<sup>3</sup>Faculty of Computer Science, Universitas Indonesia

<sup>4</sup>Avalon AI

t0939544@u.nus.edu, naufal.zafran@ui.ac.id, firania@avalon-ai.org hari@its.ac.id, gni@nus.edu.sg

## ABSTRACT

We study sequential content sufficiency by investigating whether a representation retains the ordered target information available in its input. An information-theoretic decomposition separates input ambiguity, representation loss, and readout mismatch. We construct recoverable views where perfect agreement and joint isotropic Gaussianity coexist with zero target information, and establish limits imposed by deterministic canonical anchors. Token log-loss provides a one-sided informationloss bound; a fixed-penalty ridge analysis shows why rank alone cannot determine prediction risk. These results motivate CANOPE, a nonautoregressive framework with ordered latent canvases, canonical-token supervision, and geometric regularization. On 40,000 validation sequences, latent-agreement (PL0) and token-grounded (PL2) have nearly identical pooled ranks but reach 13.5% and 98.8% positional Recall@1, respectively, under strong natural corruption when the correct target length is provided. On 3,930 LJSpeech validation utterances, frozen PL2 with a trained MatchaTTS readout yields 21.54% word error rate (WER) on corrupted text, versus 99.22% for frozen PL0, while end-to-end MatchaTTS reaches 10.93%. These results show that geometric regularity alone does not guarantee recoverable sequential content or effective downstream access in the text settings studied here.

## 1 INTRODUCTION

The information a text representation must preserve depends on how it will be used. For identity retrieval, a representation may discard substantial surface detail while retaining enough information to distinguish relevant content. Sequential generation, however, requires a downstream readout to recover the target-specific content and order needed to control the output. In text-conditioned speech synthesis, for instance, the generator must say the intended words in the intended order, even when its input contains surface errors. Information treated as nuisance or non-shared across views may therefore still matter downstream (Wang et al., 2022). Canonicalization makes this requirement testable: surface variations may be discarded, but the distinctions needed to recover a specified clean sequence must remain accessible (Tasawong et al., 2023; Lewis et al., 2020).

View-based self-supervised learning encourages agreement between related inputs (Chen et al., 2020; Gao et al., 2021), while joint-embedding predictive architectures (JEPAs) learn by predicting latent targets (Assran et al., 2023; 2025). LeJEPA gives this approach a geometric foundation: under its assumptions, isotropic Gaussian representations minimize worst-case downstream risk, motivating Sketched Isotropic Gaussian Regularization (SIGReg) (Balestriero & LeCun, 2025). VISReg pursues this target through variance and projected-shape constraints (Wu et al., 2026). Learning the encoder also changes which target information its states retain. A favorable marginal distribution does not determine the conditional target distribution, so applying a geometric guarantee to sequential recovery additionally requires preserving the distinctions needed to identify the target.

![](images/1d051f4788c67ea4ea2791a940bebcefe6a3b522d06e40be460f74ab3dea229b.jpg)  
Figure 1: Latent agreement and canonical-token grounding. The recoverable CL channel isolates access to ordered content; downstream readouts test reconstruction and speech. Both objectives use the same architecture.

Our questions are: are view agreement and isotropic Gaussian regularization sufficient to retain the ordered canonical content available in text, and how do grounding and readout affect its recovery? Answering this requires separating input ambiguity, representation loss, and readout mismatch. We first show why agreement and even exact Gaussianity leave target retention undetermined, then establish what canonical anchors and token supervision constrain. A fixed-penalty ridge analysis addresses the complementary case: information is preserved, yet its accessibility changes with geometry and target alignment.

Length controls potential content: for alphabet size K and length ℓ, sequence entropy is at most ℓ log K (Cover & Thomas, 1991; Shannon, 1948). Controlled-length (CL) experiments supply target lengths and use invertible corruption, so neither input ambiguity nor length error can explain failed recovery. Parametrized-length (PL) experiments then introduce natural corruption and predicted lengths, before trained speech readouts test downstream consequences. Figure 1 summarizes this investigation, contrasting latent agreement with canonical-token grounding and illustrating how downstream readouts test whether the learned representations support recovery of the intended content and order. Our contributions are as follows:

• Agreement and joint isotropic Gaussianity can coexist with zero target information. Nonconstant anchored codes retain information but can merge targets; token log-loss bounds the sufficiency gap, while ridge risk exposes target-dependent access.

• CANOPE’s controlled comparison finds near-perfect recovery with token grounding and poor access with latent agreement, despite recoverable inputs. In PL, nearly equal pooled ranks conceal sharply different recovery.

• The access contrast persists with trained speech readouts, while external evaluations separate canonical matching from semantic generalization.

## 2 RELATED WORK

Joint embeddings and geometry. SimCLR and VICReg combine cross-view agreement with representation constraints (Chen et al., 2020; Bardes et al., 2022). I-JEPA predicts target image regions; data2vec extends contextual latent prediction across modalities (Assran et al., 2023; Baevski et al., 2022). LeJEPA and VISReg target isotropic Gaussian geometry under explicit assumptions (Balestriero & LeCun, 2025; Wu et al., 2026).

Task information and readouts. InfoMin requires useful views to preserve task information (Tian et al., 2020); augmentation-based identifiability depends on assumptions about content and changing latent factors (von Kugelgen et al., 2021). Usable information and the Decodable Information ¨ Bottleneck formalize predictive-family constraints (Xu et al., 2020; Dubois et al., 2020). RankMe provides an empirical rank diagnostic (Garrido et al., 2023).

Latent prediction for language. LLM-JEPA studies language embedding objectives, and latent prediction has been combined with masked-token reconstruction (Huang et al., 2025; Boukhari, 2026). Recent work examines conditional concentration, structured downstream geometry, and sequential latent prediction (Dinh & Vo, 2026; Alvarez, 2026; Mostafa, 2026). Our focus is ordered canonical content under a controlled, invertible corruption channel, together with fixed-readout constraints and regularization location.

Canonical and semantic representations. BART, TSDAE, and ByT5 establish noisy-to-clean supervision and robust representations (Lewis et al., 2020; Wang et al., 2021; Xue et al., 2022); typorobust retrieval studies misspelling invariance (Tasawong et al., 2023). Sentence-BERT, SimCSE, and Qwen3 provide embedding references (Reimers & Gurevych, 2019; Gao et al., 2021; Zhang et al., 2025); NegMPNet and SemAntoNeg address negation (Anschutz et al., 2023; Vahtola et al., 2022).¨ PAWS tests lexical-overlap shortcuts, and ColBERT retains token interactions (Zhang et al., 2019; Khattab & Zaharia, 2020).

## 3 SEQUENTIAL CONTENT SUFFICIENCY

We begin with the information a downstream task needs, before asking what agreement and geometry guarantee about it. Proofs and extensions appear in Appendix A.

## 3.1 WHAT IS PRESERVED, AND WHAT CAN A READOUT ACCESS?

Let $C = ( C _ { 1 } , \dots , C _ { L } )$ be a canonical sequence over an alphabet A of size K, X its corrupted view, and $Z _ { M } \in \mathbb { R } ^ { M \times d }$ an ordered latent canvas. Side information S is available to encoder and readout. CL and PL oracle controls use $S = L , M = L$ . For PL predicted-length inference, M is predicted from X; take S constant and $Z _ { M }$ as a variable-length object with observable shape. Detaching the length-head input changes training gradients, not inference information.

A sequence readout assigns normalized probabilities $q ( C \mid Z , S )$ under the stated routing policy. Conditional entropy H and mutual information I use natural logarithms (nats); E denotes population expectation and $D _ { \mathrm { K L } }$ Kullback–Leibler divergence. The notation $C \perp Z \mid ( X , S )$ means conditional independence, $p ( c , z ~ \mid ~ x , s ) ~ = ~ p ( c ~ \mid ~ x , s ) p ( z ~ \mid ~ x , s )$ almost everywhere, not orthogonality. Appendix E collects the relevant identities.

Assumption 1 Evaluation information. Regular conditional distributions (conditional probability laws given observed variables) exist, $H ( C \mid S ) < \infty$ , and $C \perp Z \mid ( X , \bar { S } )$ . The representation receives no target information beyond the input and stated side information. Readouts access only $( Z , S )$ , without target-token prefixes or another target-token side channel. Evaluated expected log-losses are finite.

A deterministic encoder $F ( X , S )$ satisfies this requirement. We define the content-sufficiency gap using conditional mutual information,

$$
\Delta _ { \operatorname* { i n f o } } : = I ( C ; X \mid Z , S ) = H ( C \mid Z , S ) - H ( C \mid X , Z , S ) = H ( C \mid Z , S ) - H ( C \mid X , S ) .\tag{1}
$$

The last equality uses Assumption 1 (Cover & Thomas, 1991, Chapter 2). Equivalently, $\Delta _ { \mathrm { i n f o } } =$ $I ( C ; X \mid \bar { S } ) - \bar { I } ( C ; Z \mid S ) \bar { \geq } 0$ . This existing information quantity measures what processing loses: zero gap means that $( Z , S )$ preserves the input-conditioned target posterior almost surely (Appendix A.1).

Lemma 1 Log-loss decomposition. Under Assumption 1, every normalized readout q with finite expected log-loss satisfies

$$
\mathbb { E } [ - \log q ( C \mid Z , S ) ] = \underbrace { H ( C \mid X , S ) } _ { \mathrm { i n p u t ~ a n b i g u i t y } } + \underbrace { \Delta _ { \operatorname* { i n f } \omega } } _ { \mathrm { r e p r e s e n t a t i o n } } + \underbrace { \mathbb { E } _ { Z , S } D _ { \mathrm { K L } } ( p ( \cdot \mid Z , S ) \| q ( \cdot \mid Z , S ) ) } _ { \mathrm { r e a d o u t ~ m i s m a t c h } } .\tag{2}
$$

where $p$ is the true conditional target distribution.

The terms separate unavoidable input uncertainty, representation loss, and readout mismatch. A recoverable channel has $H ( C \mid X , { \bar { S } } ) = 0 .$ , hence $\hat { \Delta _ { \mathrm { i n f o } } } = H ( C \mid Z , S )$ . Low population log-loss upper-bounds the gap; high loss need not imply information loss because the readout may be restricted or poorly fitted (Xu et al., 2020; Dubois et al., 2020). Appendix A.1 separates family limits from fitting error. Squared loss similarly separates input Bayes risk, its increase after encoding, and readout error (Appendix A.2), connecting target preservation to downstream-risk guarantees.

Length and ordered content. For each ℓ with positive probability, the entropy chain rule gives

$$
H ( C \mid L = \ell ) = \sum _ { t = 1 } ^ { \ell } H ( C _ { t } \mid C _ { < t } , L = \ell ) \leq \ell \log K .\tag{3}
$$

Here $C _ { < t } = ( C _ { 1 } , \ldots , C _ { t - 1 } )$ . Since $L$ is determined by $C , H ( C ) = H ( L ) + H ( C \mid L )$ when finite. Conditioning on $L$ removes length information, not token dependence. CL fixes each example’s target length, not a common corpus length (Appendix E).

## 3.2 WHY AGREEMENT AND ISOTROPY DO NOT GUARANTEE CONTENT

Our counterexample imposes a stronger condition than the empirical diagnostics. Isotropic covariance means $\operatorname { C o v } ( R ) \stackrel { \cdot } { = } \sigma ^ { 2 } I , \stackrel { \cdot } { \sigma } ^ { 2 } > 0 ;$ standard Gaussianity additionally requires $R \sim \mathcal { N } ( 0 , \bar { I } )$ ). We impose vec $Z \sim \dot { \mathcal { N } } ( 0 , I _ { D } )$ on the entire canvas, where vec stacks entries and $D = \ell d .$ , rather than only pooled vectors or sampled rows.

Perfect agreement means $Z ^ { ( 1 ) } = Z ^ { ( 2 ) }$ almost surely (a.s.), with zero squared disagreement ${ \mathcal { L } } _ { \mathrm { d i s } } =$ $\mathbb { E } \Vert Z ^ { ( 1 ) } - Z ^ { ( 2 ) } \Vert _ { F } ^ { 2 }$ . Even this agreement and joint Gaussianity do not force target retention.

Theorem 2 Invariant Gaussian representations without target information. For every nonconstant $C \in A ^ { \ell }$ , there exist recoverable views and a shared deterministic encoder satisfying

$$
H ( C \mid X ^ { ( v ) } ) = 0 , \quad Z ^ { ( 1 ) } = Z ^ { ( 2 ) } { \mathrm { ~ a . s . } } , \qquad \mathrm { v e c } Z ^ { ( v ) } \sim { \mathcal { N } } ( 0 , I _ { D } ) , \quad I ( C ; Z ^ { ( v ) } ) = 0 .\tag{4}
$$

Thus ${ \mathcal { L } } _ { \mathrm { d i s } } = 0$ , its global minimum.

Construction and interpretation. Let $X ^ { ( v ) } = ( C , U , N _ { v } )$ , where $U \sim \mathcal { N } ( 0 , I _ { D } )$ is shared but independent of $C ,$ , and $N _ { v }$ is independent view-specific noise. Output only $U ,$ , reshaped as a canvas. The input determines $C ,$ but the representation discards it while satisfying agreement and Gaussianity. Two equiprobable word orders are therefore undecodable beyond chance. Appendix A.3 proves the claim. Exact continuous Gaussianity is impossible for a deterministic finite-string encoder; Appendix A.4 gives finite-string counterparts with identity covariance and arbitrarily small Wasserstein-2 distance to a Gaussian.

Proposition 3 Canonical-anchor restriction. Let $X ^ { ( 0 ) } = a ( C )$ be a deterministic canonical anchor, C finite-valued, and the shared encoder deterministic. Exact agreement with the anchor forces $Z = g ( C )$ and $I ( C ; Z ) = H ( Z )$ . Nonconstant representations retain some information, but can still merge distinct sequences.

The anchor has no extra shared randomness, so it excludes the zero-information construction unless the representation is constant. Conditionally independent identically distributed views given C yield the same conclusion for finite-second-moment outputs (Appendix A.5). Anchoring thus restricts the counterexample; token supervision specifies which distinctions must survive. CL latent arms use a differentiable mean of selected views including the clean view, while PL0 uses a detached clean target (Section 5.2).

## 4 READOUT CONSTRAINTS AND TARGET-DEPENDENT GEOMETRY

We next connect token supervision to retained distinctions, then isolate how geometry affects readout access when information is preserved.

## 4.1 WHAT CANONICAL TOKEN SUPERVISION CONSTRAINS

Fix $L = \ell , S = L$ as in CL and the PL oracle control. Token grounding predicts each canonical token from its canvas position. For a fixed head $q _ { W } ( \cdot \mid z ) = \mathrm { s o f f m a x } ( \tilde { W z _ { \cdot } } + b ) , W \in \mathbb { R } ^ { K \times d } , b \in \mathbb { R } ^ { K }$ a common logit shift leaves probabilities unchanged. Let $P _ { K } = { I _ { K } - { \bf 1 } \dot { \bf 1 } ^ { \top } } / { K } , A = P _ { K } W$ , and $r _ { W } = \mathrm { r a n k } ( A )$ . The population average token log-loss is

$$
\mathcal { L } _ { \mathrm { t o k } } = \frac { 1 } { \ell } \mathbb { E } \sum _ { t = 1 } ^ { \ell } - \log q _ { W } ( C _ { t } \mid z _ { t } ) .\tag{5}
$$

Proposition 4 Observable directions and content bound. For finite logits, fixed $( W , b )$ , and any perturbation $\delta \in \mathbb { R } ^ { d }$

$$
q _ { W } ( \cdot \mid z + \delta ) = q _ { W } ( \cdot \mid z ) \iff A \delta = 0 , \qquad r _ { W } \leq \operatorname* { m i n } ( d , K - 1 ) .\tag{6}
$$

Under Assumption 1, $0 \le \Delta _ { \mathrm { i n f o } } \le H ( C \mid Z , S ) \le \ell \mathcal { L } _ { \mathrm { t o k } } .$

The fixed head observes logit differences and is blind to the $d - r _ { W }$ dimensional null space ker A. Applying Lemma 1 to $\textstyle \prod _ { t } { \bar { q } } _ { W } ( C _ { t } \mid z _ { t } )$ gives the bound without assuming independent true tokens. Low population loss over every target position certifies a small gap, but does not constrain all latent directions. Appendix A.6 proves the variable-length and curvature extensions; prefix-only supervision cannot certify the suffix.

Local access is stricter than full-canvas access. A local head sees only $z _ { t } ,$ whereas a sequence readout sees all of $Z .$ . For independent fair bits and $Z = ( C _ { 2 } , C _ { 1 } )$ , each local prediction is at chance, but swapping positions recovers the target. Equation (26) decomposes optimal local loss into full-canvas uncertainty, residual token dependence, and information at other positions. A fixed linear head is further restricted; its failure alone cannot prove information loss.

## 4.2 WHY HIGHER EFFECTIVE RANK CAN HELP OR HURT A FIXED READOUT

We rescale representation directions while preserving target information, then fit a population predictor with fixed ridge penalty. This isolates geometry–readout interaction without finite-sample estimation error.

The target and its coefficients. Let $G \sim \mathcal { N } ( 0 , I _ { d } )$ and $Y = \theta ^ { \top } G + \varepsilon$ . Fixed θ selects targetrelevant factors; w denotes learned readout coefficients and $\| \cdot \|$ the Euclidean norm.

Assumption 2 Population ridge model. Fix $\theta \neq 0$ , penalty $\mu > 0$ , and noise with $\mathbb { E } [ \varepsilon \mid G ] = 0$ and $\mathbb { E } \varepsilon ^ { 2 } = \bar { \sigma } ^ { 2 } < \infty$ . For a positive-definite covariance Σ with total variance tr $\Sigma = d .$ define

$$
Z _ { \Sigma } = \Sigma ^ { 1 / 2 } G , \qquad w _ { \Sigma } = \arg \operatorname* { m i n } _ { w } \left\{ \mathbb E ( Y - w ^ { \top } Z _ { \Sigma } ) ^ { 2 } + \mu \| w \| ^ { 2 } \right\} .\tag{7}
$$

Invertibility, $G = \Sigma ^ { - 1 / 2 } Z _ { \Sigma }$ , preserves target information and unrestricted risk $\sigma ^ { 2 }$ . Fixing total variance prevents trivial penalty reduction by uniformly scaling features.

Spectrum, effective rank, and excess risk. In a covariance eigenbasis, let $\lambda _ { i }$ be variances and $\theta _ { i }$ target coefficients. With $p _ { i } = \lambda _ { i } / \mathrm { t r } \Sigma$ , define

$$
r _ { \mathrm { e f f } } ( \Sigma ) = \exp \left( - \sum _ { i = 1 } ^ { d } p _ { i } \log p _ { i } \right) .\tag{8}
$$

Effective rank approaches 1 for one-direction concentration and equals d for equal eigenvalues. It measures neither target alignment nor Gaussian shape; unlike RankMe (Garrido et al., 2023), it uses

covariance eigenvalues. Ridge normal equations give excess risk, excluding the penalty:

$$
R _ { \mu } : = \mathbb { E } ( Y - w _ { \Sigma } ^ { \top } Z _ { \Sigma } ) ^ { 2 } - \sigma ^ { 2 } = \mu ^ { 2 } \sum _ { i = 1 } ^ { d } \frac { \theta _ { i } ^ { 2 } } { ( \lambda _ { i } + \mu ) ^ { 2 } } .\tag{9}
$$

Small-variance target directions require larger coefficients and are penalized more strongly (derivation: Appendix A.7).

A path toward isotropy. Fix the eigenbasis and split it into blocks of sizes r and $d - r$ , with $1 \leq r < d .$ Set their variances to $a { \dot { = } } [ d - ( d - { \dot { r } } ) \rho ] / r$ and $b = \rho ,$ , where $0 < \rho < 1$ . This preserves total variance d. As $\rho$ increases, both variances approach 1 and effective rank increases. Let $\begin{array} { r } { \eta = \sum _ { i < r } \theta _ { i } ^ { 2 } / \lVert \theta \rVert ^ { 2 } } \end{array}$ be the target’s energy share in the first block. This analytical family is not an assumed trained-model spectrum.

Theorem 5 Target-alignment threshold. Under Assumption 2, along the fixed-basis family above,

$$
\mathrm { s i g n } \bigg ( { \frac { d R _ { \mu } } { d r _ { \mathrm { e f f } } } } \bigg ) = \mathrm { s i g n } ( \eta - \eta _ { \ast } ) , \qquad \eta _ { \ast } : = { \frac { r ( a + \mu ) ^ { 3 } } { r ( a + \mu ) ^ { 3 } + ( d - r ) ( b + \mu ) ^ { 3 } } } .\tag{10}
$$

Higher effective rank lowers risk when $\eta < \eta _ { \ast }$ and raises it when $\eta > \eta _ { * }$ . Equality gives zero local derivative. The threshold depends on $\rho ,$ so this is a local comparison, not a universal ordering of representations (proof: Appendix A.8).

A two-dimensional example. For $d = 2 , r = \mu = 1 , \sigma ^ { 2 } = 0$ , increasing $\rho$ from 0.2 to 0.8 changes the spectrum from (1.8, 0.2) to (1.2, 0.8) and effective rank from 1.3841 to 1.9601. Risk rises from 0.1276 to 0.2066 for $\theta = ( 1 , 0 ) ^ { \top }$ , but falls from 0.6944 to 0.3086 for $\theta = ( 0 , 1 ) ^ { \top }$ . No information is lost; variance moves away from the first target direction toward the second (Figure 4).

Scope and implication. Penalty retuning is excluded; excess risk vanishes as $\mu \to 0$ for fixed positive-definite Σ. Yet $I _ { d }$ uniquely minimizes worst-target risk at fixed target norm and trace d (Appendix A.9). LeJEPA-type guarantees (Balestriero & LeCun, 2025) therefore coexist with target-specific differences: rank changes need not reflect information changes.

## 5 CANOPE FRAMEWORK

CANOPE (Canonicalization-Oriented Predictive Embedding) makes these distinctions testable by mapping corrupted text to ordered latent canvases. Its common architecture separates the choice of predictive target, regularization location, and readout.

## 5.1 ARCHITECTURE AND CANVAS LENGTH

For input x containing N UTF-8 bytes, a Transformer encoder produces $H \in \mathbb { R } ^ { N \times d }$ (Vaswani et al., 2017). A resampler $\mathcal { R } _ { N  M }$ forms M query vectors from $H ;$ sinusoidal positions $P _ { M }$ specify their output order. A noncausal decoder attends to both these queries and the encoder states.

$$
H = E _ { \vartheta } ( x ) , \qquad p _ { t } = \mathrm { s o f t m a x } ( W z _ { t } + b ) , \qquad Z _ { M } = D _ { \vartheta } ( \mathrm { L N } ( \mathcal R _ { N  M } ( H ) + P _ { M } ) , H ) .\tag{11}
$$

Here $\vartheta$ denotes network parameters, distinct from ridge target $\theta ,$ and $\mathrm { L N }$ is layer normalization. The head predicts 256 bytes and PAD. All positions are computed in parallel without target-token prefixes (Gu et al., 2018). Runs use width 512, four encoder blocks, two decoder blocks, and eight heads (Appendix B.2).

Oracle routing sets $M = L ;$ predicted routing uses $M = \operatorname* { m a x } ( 1 , \lceil \widehat { L } \rceil )$ from a learned head; input routing sets $\bar { M } = N$ as a stress test. The resampler Gaussian-averages valid source states, with explicit identity at M = N (Appendix B.2). CL removes final decoder normalization; PL retains it, so raw geometry comparisons stay within each series.

## 5.2 OBJECTIVES AND THE CL/PL COMPARISONS

Controlled-Length (CL): Isolate the predictive objective and regularization location. CL uses oracle lengths. Grounding (G) averages canonical-byte negative log-probabilities; latent agreement (L) minimizes squared distance to a differentiable mean of designated views. Both add $0 . 0 1 \mathcal { R } _ { \mathrm { V I S } }$ with no application auxiliary loss (Appendix B.3).

Pooled (P) regularizes mean canvas rows; sampled token (T) regularizes one uniformly sampled valid position per item, shared across views. CL-GP/GT/LP/LT form a two-by-two design with equal vector counts per batch; T constrains a position mixture, not the joint canvas. The four configurations are summarized in Table 6.

Our VISReg adaptation penalizes coordinate means and nonunit standard deviations, then matches sorted standardized projections to Gaussian quantiles. Its center, scale, and shape constraints do not certify exact Gaussianity. Equation (48) specifies normalization, stop-gradients, and 256 projection directions.

Parametrized-Length (PL): Compare parametrized-length recipes. The PL setting adds objectives for length, surface invariance, and premise–negation distinctions. Its token-grounded recipe is

$$
\mathcal { L } _ { \mathrm { a p p } } = \mathcal { L } _ { \mathrm { C E } } + \eta _ { \mathrm { e } } \mathcal { L } _ { \mathrm { e m p t y } } + \gamma \mathcal { L } _ { \mathrm { l e n } } + \alpha \mathcal { L } _ { \mathrm { i n v } } + \beta \mathcal { L } _ { \mathrm { n e g } } + \zeta \mathcal { R } _ { \mathrm { V I S } } .\tag{12}
$$

CE supervises canonical bytes and a short PAD band; empty-tail loss discourages confident later predictions; length loss predicts $L ;$ pooled invariance aligns views; negation loss ranks same-content anchors above negated and unrelated anchors. We use $( \eta _ { \mathrm { e } } , \gamma , \alpha , \beta ) = ( 0 . 1 , 1 , 0 . 5 , 0 . 3 )$ and pooled VISReg weights $\zeta = 0 . 1 / 0 . 5$ for PL2/PL3 (Appendix B.4).

PL0 uses detached-anchor latent agreement, geometric weight 0.3, and detached token-head inputs: token loss trains the head but not the backbone. PL0–PL2 thus compares complete recipes (Appendix B.5); Table 7 maps all variants to checkpoints.

## 5.3 READOUTS: IDENTITY RETRIEVAL AND TOKEN RECONSTRUCTION

Retrieval matches a corrupted query to its canonical target in a fixed gallery. We compare either meancanvas cosines or the average cosine at 16 linearly interpolated relative positions. The latter retains coarse order across different lengths without target-token alignment; it does not certify word-order understanding. Appendix B.1 gives the exact score.

Native reconstruction uses per-position argmax and stops at PAD. These diagnostics fit no new probes and exclude untrained CL-LP/LT token heads. Section 6.4 separately trains a speech readout. Each task measures access through a specified readout, not $\Delta _ { \mathrm { { i n f o } } }$ itself.

## 6 EXPERIMENTS AND RESULTS

We test the contrast in Figure 1 on 40,000 validation targets per series, with one training seed and 500 paired bootstrap resamples. Pretraining uses English–Singlish G2P text (English–Singlish G2P), without phonemes; Appendix B.6 details data and evaluation.

## 6.1 CONTROLLED CONTENT RECOVERY

Under recoverable severity 3 and oracle lengths, CL-GP/GT reach 99.985%/99.980% retrieval and 98.97%/98.78% exact reconstruction, while CL-LP/LT reach only 0.52%/0.75% retrieval (Table 10). Grounding enables recovery at either location.

Input-length routing instead leaves CL-GP/GT at 97.48%/16.53% retrieval (Figure 2); CL-GP’s 91.84% CER separates identity matching from reconstruction. Centering CL-LT raises clean retrieval from 0.76% to 100%, but corrupted retrieval only from 0.75% to 1.58%, revealing clean distinctions without restoring cross-view access (Appendix C.2).

![](images/4af94c4e4d1472e415faaddf9928442c2e096de6ad5ce71d1fa1cf318211472a.jpg)  
Figure 2: CL relative-position retrieval with oracle or input lengths (40,000 targets; level 0: clean). CL-GT is strongly length-sensitive (Table 9).

## 6.2 NATURAL CORRUPTION

With natural severity 3 and oracle lengths, PL0/PL2 have nearly equal pooled ranks (15.76/15.82), yet retrieval differs sharply (13.46%/98.80%; Figure 3). Rank misses this contrast.

Predicted lengths and centering leave this contrast intact. PL2’s 23.09% CER and 12.64% exact match nevertheless limit what retrieval establishes, while PL7’s positional gain shows that readout choice also matters (Table 1).

## 6.3 EXTERNAL IDENTITY AND SEMANTIC TRANSFER

LexNorm2015 (Baldwin et al., 2015) tests natural noncanonical text against frozen references (Table 2).

PL3 leads CANOPE on original queries, but PL2 leads under whitespace perturbation. Relative positions improve both (Figure 9), yet weak external semantics limit generalization (Table 3). Strong synthetic-foil results (Table 4) indicate sensitivity to specified changes, rather than general semantic equivalence.

(a) Oracle length  
![](images/cc6ca784ce0144a31975e703cc36283921a7d303ca4e219bf8ca7dd405e8eacd.jpg)

(b) Predicted length  
![](images/4d48867738cb8070915437ed0a2865fcb5a969e712e5fc21515bd7b3fc3fa47c.jpg)  
Figure 3: PL identity access despite similar pooled ranks: oracle (a) and predicted (b) lengths, 40,000 targets per query. Checkpoints and reconstruction: Table 1.

Table 1: All evaluated PL setting checkpoints at natural severity 3. Rank and reconstruction use oracle lengths; retrieval uses predicted lengths. R1, CER, and exact match (EM) are percentages.
<table><tr><td rowspan="2">Model</td><td colspan="2">Eff. rank</td><td colspan="2">R1↑</td><td rowspan="2">CER↓</td><td rowspan="2">EM↑</td></tr><tr><td>Pool</td><td>Token</td><td>Pool</td><td>Rel.</td></tr><tr><td>PL0</td><td>15.76</td><td>15.76</td><td>6.15</td><td>6.31</td><td>87.50</td><td>0.00</td></tr><tr><td>PL1</td><td>4.79</td><td>9.72</td><td>86.76</td><td>97.34</td><td>21.38</td><td>12.81</td></tr><tr><td>PL2</td><td>15.82</td><td>33.04</td><td>94.45</td><td>98.80</td><td>23.09</td><td>12.64</td></tr><tr><td>PL3</td><td>36.06</td><td>65.11</td><td>97.57</td><td>99.19</td><td>22.51</td><td>12.53</td></tr><tr><td>PL4</td><td>83.59</td><td>126.10</td><td>91.31</td><td>99.30</td><td>21.63</td><td>12.80</td></tr><tr><td>PL5†</td><td>10.19</td><td>36.46</td><td>68.68</td><td>78.41</td><td>28.35</td><td>6.68</td></tr><tr><td>PL6</td><td>7.28</td><td>16.89</td><td>82.72</td><td>97.24</td><td>21.95</td><td>12.68</td></tr><tr><td>PL7</td><td>6.16</td><td>19.13</td><td>74.44</td><td>96.63</td><td>22.46</td><td>12.64</td></tr></table>

PL0: latent agreement; PL1: token baseline; PL2/3: weak/stronger VISReg; PL4/5/6: no invariance/negation/empty loss; PL7: zero second margin. PL4–7 modify the historical unregularized recipe, not PL2. <sup>†</sup>PL5 has a different training budget and is excluded from controlled component comparisons. Checkpoint IDs: Table 7.

## 6.4 SPEECH FROM FROZEN CANVASES

We train MatchaTTS (Mehta et al., 2024) on frozen PL0/PL2 predicted-length canvases, without decoded text, alongside an end-to-end phoneme baseline. All use identical LJSpeech pairs, 70% corrupted inputs, and pretrained acoustic modules (Appendix D).

Whisper large-v3 (Radford et al., 2023; OpenAI, 2023) gives corrupted WER of 21.54% for PL2 versus 99.22% for PL0, extending the contrast to speech (Table 5). The end-to-end baseline remains stronger (10.93%), despite PL2’s smaller degradation.

Table 2: LexNorm identity retrieval (1,941 queries, 1,954 targets). R1: original Recall@1; WS3: strongest whitespace perturbation. CANOPE uses predicted lengths and relative positions; external readouts are in Appendix B.6.
<table><tr><td>Model</td><td>R1 (%) ↑</td><td>MRR↑</td><td>WS3 (%) ↑</td></tr><tr><td>PL0</td><td>23.34</td><td>0.281</td><td>1.96</td></tr><tr><td>PL2</td><td>88.82</td><td>0.908</td><td>83.82</td></tr><tr><td>PL3</td><td>97.37</td><td>0.979</td><td>81.20</td></tr><tr><td>ByT5 encoder</td><td>87.48</td><td>0.894</td><td>22.05</td></tr><tr><td>SimCSE</td><td>98.71</td><td>0.991</td><td>98.71</td></tr><tr><td>NegMPNet</td><td>99.12</td><td>0.994</td><td>99.12</td></tr><tr><td>Qwen3-0.6B</td><td>99.69</td><td>0.998</td><td>99.64</td></tr></table>

Table 3: External semantic results on complete common cohorts. SAN (SemAntoNeg): three-choice accuracy (%); PAWS: ROC AUC; STS-B and packaged SICK-R: Spearman correlation. CANOPE uses predicted lengths and relative positions.  
Table 4: Synthetic semantic-foil decisions (%): 512 base groups per type, two directions per group. Neg.: negation; Num.: number; Role: role swap. Chance is 50%. All encoders have zero failures.
<table><tr><td>Model</td><td colspan="3">SAN ↑ PAWS ↑ STS-B ↑ SICK-R ↑</td></tr><tr><td>PL0</td><td>26.24</td><td>0.488 0.003</td><td>0.143</td></tr><tr><td>PL2</td><td>2.28</td><td>0.585 0.154</td><td>0.333</td></tr><tr><td>PL3</td><td>9.11</td><td>0.587 0.177</td><td>0.375</td></tr><tr><td>ByT5 encoder</td><td>8.12</td><td>0.584 0.248</td><td>0.283</td></tr><tr><td>SimCSE</td><td>63.80</td><td>0.647 0.842</td><td>0.808</td></tr><tr><td>NegMPNet</td><td>69.57</td><td>0.623 0.842</td><td>0.767</td></tr><tr><td>Qwen3-0.6B</td><td>45.11</td><td>0.684 0.846</td><td>0.812</td></tr></table>

<table><tr><td>Model</td><td>Neg. ↑</td><td>Num. ↑</td><td>Entity ↑</td><td>Role ↑</td></tr><tr><td>PL0</td><td>42.48</td><td>50.39</td><td>79.10</td><td>81.93</td></tr><tr><td>PL2</td><td>100.00</td><td>90.62</td><td>100.00</td><td>100.00</td></tr><tr><td>PL3</td><td>96.19</td><td>97.75</td><td>100.00</td><td>100.00</td></tr><tr><td>ByT5 encoder</td><td>85.94</td><td>53.42</td><td>67.09</td><td>57.71</td></tr><tr><td>SimCSE</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>NegMPNet</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>Qwen3-0.6B</td><td>98.63</td><td>97.85</td><td>100.00</td><td>57.71</td></tr></table>

Cohort sizes: 3,152 groups; 8,000 pairs; 1,379 pairs; and 9,927 pairs, respectively. The SICK-derived split is the packaged mteb/sickr-sts test split, not a claim about the original SICK test-set size.

Table 5: Downstream speech WER (% ↓); 3,930 utterances and 66,489 reference words per cell. ∆: augmented minus clean WER (percentage points), reported descriptively.
<table><tr><td>System</td><td>Clean↓</td><td>Augmented ↓</td><td>Δ</td></tr><tr><td>MatchaTTS</td><td>2.21</td><td>10.93</td><td>+8.71</td></tr><tr><td>Frozen PL2 + Matcha</td><td>13.60</td><td>21.54</td><td>+7.94</td></tr><tr><td>Frozen PL0 + Matcha</td><td>99.12</td><td>99.22</td><td>+0.09</td></tr></table>

## 7 DISCUSSION

The central issue is what an agreement target requires the encoder to preserve. Both CL objectives see canonical text, but the latent target is learned along with the encoder. If two sequences become indistinguishable, agreement with their encoded anchors need not separate them again. Proposition 3 makes this precise at exact agreement. A nonconstant anchored code retains some information, yet can still merge targets. Canonical-token grounding instead penalizes distinctions lost at the supervised positions. The relevant difference is a fixed content target versus a jointly learned latent target, not cross-entropy versus squared loss itself.

CL connects this distinction to Lemma 1. Invertible corruption and oracle lengths make input ambiguity zero, leaving representation loss and readout mismatch as the possible sources of failure. Near-perfect grounded reconstruction and low empirical token loss are consistent with Proposition 4, which bounds the gap through population token loss while leaving latent null-space directions unconstrained. Failed latent retrieval cannot separate the remaining terms, but shows that agreement training left available content difficult to access through these readouts.

The centering control further explains why small disagreement is an inadequate diagnostic. CL-LT has a large common mean and tiny residual variance (Appendix C.2); small absolute discrepancies can then coexist with unstable content distinctions. Removing the mean exposes clean identity but barely helps corrupted views. This behavior is consistent with the scale mechanism in Eq. (50): shrinking residuals reduces raw disagreement without improving their separation relative to corruption. Theorem 2 establishes a stronger logical limit under ideal geometry, while these trained states expose a practical failure before that ideal is attained.

This also clarifies the relationship to LeJEPA. Its optimality results concern downstream risk under specified learner, target, and distributional assumptions (Balestriero & LeCun, 2025); a learned encoder must additionally preserve the target-conditioned information to which a risk guarantee applies. Equation (17) separates the resulting Bayes-risk increase from readout error. Our ridge analysis holds the former at zero by invertibility, and Theorem 5 shows how target alignment still changes fixed-penalty access. Isotropy remains minimax in that model (Appendix A.9). Thus favorable worst-target geometry and poor access to a particular target are compatible. PL’s matched pooled ranks support the need for target-specific evaluation; they neither certify joint Gaussianity nor instantiate the ridge spectrum path.

CL identifies a controlled access gap, PL extends it to natural corruption, and speech shows its downstream consequence. Retrieval–reconstruction differences and weak semantics delimit what canonical matching establishes. Causal attribution remains strongest in CL: PL compares complete recipes with different auxiliary losses and weights. Bootstrap intervals omit seed variation; speech uses validation data with pretraining exposure, and frontend, freezing, and alignment differences remain. ASR WER measures lexical fidelity, not naturalness. These factors motivate broader readout and pretraining-disjoint evaluations.

## 8 CONCLUSION

Agreement and geometric regularity alone do not guarantee recoverable sequential content. CANOPE’s controlled experiments reveal recovery failures despite recoverable inputs and known target lengths. Canonical-token grounding supports accurate recovery, with benefits extending to downstream speech. These findings motivate evaluating ordered-content recovery alongside geometric diagnostics for sequential generation.

## REFERENCES

Robert Jenkinson Alvarez. Beyond isotropy in JEPAs: Hamiltonian geometry and symplectic prediction. arXiv preprint arXiv:2605.20107, 2026. URL https://arxiv.org/abs/2605. 20107.

Miriam Anschutz, Diego Miguel Lozano, and Georg Groh. This is not correct! negation-aware ¨ evaluation of language generation systems. In Proceedings of the 16th International Natural Language Generation Conference, 2023. URL https://arxiv.org/abs/2307.13989.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. URL https://arxiv.org/abs/2301.08243.

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. doi: 10.48550/arXiv.2506.09985. URL https://arxiv.org/abs/ 2506.09985.

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language. In Proceedings of the 39th International Conference on Machine Learning, 2022. URL https://arxiv.org/ abs/2202.03555.

Timothy Baldwin, Marie Catherine de Marneffe, Bo Han, Young-Bum Kim, Alan Ritter, and Wei Xu. Shared tasks of the 2015 workshop on noisy user-generated text: Twitter lexical normalization and named entity recognition. In Proceedings of the Workshop on Noisy User-generated Text, 2015. URL https://aclanthology.org/W15-4319/.

Randall Balestriero and Yann LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025. doi: 10.48550/arXiv.2511.08544. URL https://arxiv.org/abs/2511.08544.

Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2105.04906.

Aimen Boukhari. Predict and reconstruct: Joint objectives for self-supervised language representation learning. arXiv preprint arXiv:2606.05173, 2026. URL https://arxiv.org/abs/2606. 05173.

Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1:˜ Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation, 2017. URL https://aclanthology. org/S17-2001/.

Tao Chen and Min-Yen Kan. Creating a live, public short message service corpus: The NUS SMS corpus. Language Resources and Evaluation, 47:299–335, 2013. doi: 10.1007/s10579-012-9197-9. URL https://doi.org/10.1007/s10579-012-9197-9.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pp. 1597–1607. PMLR, 2020. URL https://proceedings.mlr.press/v119/chen20j.html.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. John Wiley & Sons, 1991. URL https://www.cs.columbia.edu/<sub>˜</sub>vh/courses/LexicalSemantics/ Association/Cover&Thomas-Ch2.pdf.

Anh Trac Duc Dinh and Khang Nhat Hoang Vo. The JEPA paradox in language: The geometry of linguistic alternatives. arXiv preprint arXiv:2607.23531, 2026. URL https://arxiv.org/ abs/2607.23531.

Yann Dubois, Douwe Kiela, David J. Schwab, and Ramakrishna Vedantam. Learning optimal representations with the decodable information bottleneck. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://arxiv.org/abs/2009.12789.

English–Singlish G2P. English–singlish G2P corpus, 2026. URL https://huggingface.co/ datasets/avalonai/english-singlish-g2p. Derived compilation assembled by the authors. Component sources are credited separately.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6894–6910. Association for Computational Linguistics, 2021. doi: 10.18653/v1/ 2021.emnlp-main.552. URL https://aclanthology.org/2021.emnlp-main.552/.

Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann LeCun. RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 10929–10974, 2023. URL https://proceedings.mlr.press/ v202/garrido23a.html.

Jiatao Gu, James Bradbury, Caiming Xiong, Victor O. K. Li, and Richard Socher. Non-autoregressive neural machine translation. In International Conference on Learning Representations, 2018. URL https://arxiv.org/abs/1711.02281.

Hai Huang, Yann LeCun, and Randall Balestriero. LLM-JEPA: Large language models meet joint embedding predictive architectures. arXiv preprint arXiv:2509.14252, 2025. URL https: //arxiv.org/abs/2509.14252.

Keith Ito and Linda Johnson. The LJ Speech dataset, 2017. URL https://keithito.com/ LJ-Speech-Dataset/.

Omar Khattab and Matei Zaharia. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings ofthe 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020. URL https://arxiv.org/abs/ 2004.12832.

Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis. In Advances in Neural Information Processing Systems, volume 33, pp. 17022–17033, 2020. URL https://proceedings.neurips.cc/paper/ 2020/hash/c5d736809766d46260d816d8dbc9eb44-Abstract.html.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pp. 7871–7880. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.703. URL https:// aclanthology.org/2020.acl-main.703/.

Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. A SICK cure for the evaluation of compositional distributional semantic models. In Proceedings of the Ninth International Conference on Language Resources and Evaluation, 2014. URL https://aclanthology.org/L14-1314/.

Shivam Mehta, Ruibo Tu, Jonas Beskow, Eva Sz<sup>´</sup> ekely, and Gustav Eje Henter. Matcha-TTS: A´ fast TTS architecture with conditional flow matching. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2024. doi: 10.1109/ICASSP48485.2024.10448291. URL https://arxiv.org/abs/2309.03199.

Mohsen Mostafa. Dynamic LeJEPA: Maximum entropy representations for sequential prediction and latent planning with theoretical guarantees. Preprints.org, 2026. doi: 10.20944/preprints202609. 0981.v1. URL https://doi.org/10.20944/preprints202609.0981.v1. Preprint, version 1.

OpenAI. Whisper large-v3 model card, 2023. URL https://huggingface.co/openai/ whisper-large-v3.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. LibriSpeech: An ASR corpus based on public domain audio books. In IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 5206–5210, 2015. doi: 10.1109/ICASSP.2015.7178964. URL https://www.openslr.org/12/.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 28492–28518, 2023. URL https://proceedings.mlr.press/v202/radford23a. html.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019. URL https://arxiv.org/abs/1908.10084.

Claude E. Shannon. A mathematical theory of communication. Bell System Technical Journal, 27(3): 379–423, 1948. doi: 10.1002/j.1538-7305.1948.tb01338.x. URL https://www.princeton. edu/<sub>˜</sub>wbialek/rome/refs/shannon\_48.pdf.

Panuthep Tasawong, Wuttikorn Ponwitayarat, Peerat Limkonchotiwat, Can Udomcharoenchaikit, Ekapol Chuangsuwanich, and Sarana Nutanong. Typo-robust representation learning for dense retrieval. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1106–1115. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-short.95. URL https://aclanthology.org/2023. acl-short.95/.

Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? In Advances in Neural Information Processing Systems, 2020. URL https://arxiv.org/abs/2005.10243.

Teemu Vahtola, Mathias Creutz, and Jorg Tiedemann. It is not easy to detect paraphrases: Analysing¨ semantic similarity with antonyms and negation using the new SemAntoNeg benchmark. In Proceedings ofthe Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, 2022. URL https://aclanthology.org/2022.blackboxnlp-1.20/.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. URL https://arxiv.org/abs/1706.03762.

Julius von Kugelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Sch¨ olkopf, Michel¨ Besserve, and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from style. In Advances in Neural Information Processing Systems, 2021. URL https://arxiv.org/abs/2106.04619.

Haoqing Wang, Xun Guo, Zhi-Hong Deng, and Yan Lu. Rethinking minimal sufficient representation in contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16041–16050, 2022. URL https://arxiv.org/abs/2203.07004.

Kexin Wang, Nils Reimers, and Iryna Gurevych. TSDAE: Using transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning. In Findings of the Associationfor Computational Linguistics: EMNLP 2021, pp. 671–688, 2021. doi: 10.18653/v1/ 2021.findings-emnlp.59. URL https://aclanthology.org/2021.findings-emnlp. 59/.

Wikipedia contributors. English Wikipedia. Wikimedia Foundation, n.d. URL https://en. wikipedia.org/. Community-authored encyclopedia; the compilation’s source snapshot is unspecified.

Haiyu Wu, Randall Balestriero, and Morgan Levine. VISReg: Variance-invariance-sketching regularization for JEPA training. arXiv preprint arXiv:2606.02572, 2026. doi: 10.48550/arXiv.2606.02572. URL https://arxiv.org/abs/2606.02572.

Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A theory of usable information under computational constraints. In International Conference on Learning Representations, 2020. URL https://arxiv.org/abs/2002.10689.

Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. ByT5: Towards a token-free future with pre-trained byte-to-byte models. Transactions of the Association for Computational Linguistics, 10:291–306, 2022. doi: 10.1162/tacl a 00461. URL https://aclanthology.org/2022.tacl-1.17/.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. URL https://arxiv.org/abs/2506.05176.

Yuan Zhang, Jason Baldridge, and Luheng He. PAWS: Paraphrase adversaries from word scrambling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019. URL https://arxiv. org/abs/1904.01130.

## A PROOFS AND SCOPE OF THE THEORETICAL RESULTS

All information quantities concern the discrete target C, including when $Z$ is continuous. The analysis requires no differential-entropy identity. Appendix A.1 proves the content-gap decomposition; Appendix A.2 connects it to squared prediction risk; Appendices A.3–A.5 give the Gaussian construction and its channel restrictions; Appendix A.6 treats token-head observability and local access; and Appendices A.7–A.9 derive the ridge comparison. The main text states the definitions and assumptions used to interpret each result.

## A.1 INFORMATION DECOMPOSITION AND READOUT MISMATCH

Proof of Lemma 1. The conditional Markov property gives $H ( C \mid X , Z , S ) = H ( C \mid X , S )$ . Hence

$$
I ( C ; X \mid Z , S ) = H ( C \mid Z , S ) - H ( C \mid X , S ) = I ( C ; X \mid S ) - I ( C ; Z \mid S ) .\tag{13}
$$

Finite $H ( C \mid S )$ makes these differences well defined. Adding and subtracting log $p ( C \mid Z , S )$ yields

$$
\begin{array} { r l } & { \mathbb { E } [ - \log q ( C \mid Z , S ) ] = \mathbb { E } [ - \log p ( C \mid Z , S ) ] + \mathbb { E } \log \frac { p ( C \mid Z , S ) } { q ( C \mid Z , S ) } } \\ & { \qquad = H ( C \mid Z , S ) + \mathbb { E } _ { Z , S } D _ { \mathrm { K L } } ( p ( \cdot \mid Z , S ) \| q ( \cdot \mid Z , S ) ) . } \end{array}\tag{14}
$$

Substituting (13) proves (2). A readout assigning zero probability to a positive-probability target event has infinite loss and is excluded by the assumption. Nonnegativity of the three terms gives the stated bounds. Moreover,

$$
I ( C ; X \mid Z , S ) = \mathbb { E } _ { X , Z , S } D _ { \mathrm { K L } } ( p ( \cdot \mid X , S ) \| p ( \cdot \mid Z , S ) ) ,\tag{15}
$$

so $\Delta _ { \mathrm { i n f o } } = 0$ is equivalent to preserving the input-conditioned target posterior almost surely.

For a readout family $\mathcal { Q }$ containing a finite-loss predictor, define $L ( q ) = \mathbb { E } [ - \log q ( C \mid Z , S ) ]$ $R _ { \mathcal { Q } } = \operatorname* { i n f } _ { q \in \mathcal { Q } } L ( q )$ , and $\begin{array} { r } { \Delta _ { \mathcal { Q } } = \operatorname* { i n f } _ { q \in \mathcal { Q } } \mathbb { E } D _ { \mathrm { K L } } ( p \| \bar { q } ) } \end{array}$ . Every fitted $\widehat { q } \in \mathcal { Q }$ satisfies

$$
L ( \widehat { q } ) = H ( C \mid X , S ) + \Delta _ { \mathrm { i n f o } } + \Delta _ { Q } + [ L ( \widehat { q } ) - R _ { Q } ] .\tag{16}
$$

The final terms separate approximation limits from excess population loss of the fitted predictor; neither infimum needs to be attained. An empirical estimate additionally has sampling uncertainty. This distinction motivates readout-family comparisons without interpreting a failed probe as a proof of information loss.

## A.2 FROM TARGET PRESERVATION TO SQUARED PREDICTION RISK

The information decomposition has a standard squared-loss counterpart that makes its connection to downstream-risk guarantees explicit. Let $T$ be a square-integrable vector target satisfying $T \perp Z |$ $( X , S )$ , and define $m _ { X } = \mathbb { E } [ T \mid X , S ]$ and $m _ { Z } = \mathbf { \bar { E } } [ T \mid Z , \mathbf { \bar { \it S } } ]$ . For any square-integrable readout ${ \dot { g } } ( Z , { \dot { S } } )$ ,

$$
\mathbb { E } \| T - g ( Z , S ) \| ^ { 2 } = \underbrace { \mathbb { E } \| T - m _ { X } \| ^ { 2 } } _ { \mathrm { i n p u t ~ B a y e s ~ r i s k } } + \underbrace { \mathbb { E } \| m _ { X } - m _ { Z } \| ^ { 2 } } _ { \mathrm { i n c r e a s e ~ i n ~ B a y e s ~ r i s k } } + \underbrace { \mathbb { E } \| m _ { Z } - g ( Z , S ) \| ^ { 2 } } _ { \mathrm { r e a d o u t ~ e r r o r ~ a b o v e ~ t h e } } .\tag{17}
$$

Proof. Write $T - g = ( T - m _ { X } ) + ( m _ { X } - m _ { Z } ) + ( m _ { Z } - g )$ . Conditional independence gives $\mathbb { E } [ T \mid X , Z , S ] = { \bar { m } } _ { X }$ , so $T - m _ { X }$ is orthogonal in $L ^ { 2 }$ to both remaining terms. The tower property gives $\mathbb { E } [ m _ { X } \mid ^ { \cdot } Z , S ] = m _ { Z }$ , making the last two terms orthogonal as well. Expanding the squared norm proves the identity. □

For finite-valued $C ,$ choose $T$ to be its one-hot encoding. Then $m _ { X }$ and $m _ { Z }$ are the input- and representation-conditioned target probability vectors. The middle term vanishes exactly when these posteriors agree almost surely, equivalently $\Delta _ { \mathrm { i n f o } } = 0$ under Assumption 1. Its value is not $\Delta _ { \mathrm { { i n f o } } } \mathrm { { : } }$ it measures squared posterior discrepancy rather than expected KL divergence. For a general vector target, it instead certifies preservation of the conditional mean relevant to squared loss.

This identity separates two ways in which a representation can affect downstream prediction. Geometry can change a learner’s error above the representation Bayes risk, while a learned encoder can also change that Bayes risk by discarding target information. The ridge family in Section 4.2 makes the latter increase zero through invertibility, isolating the former effect. The counterexample below makes target information vanish despite exact agreement and Gaussianity. Thus the two analyse address complementary terms rather than competing claims about the same risk.

## A.3 INVARIANT GAUSSIAN COUNTEREXAMPLE

ProofofTheorem 2. Choose $U \sim \mathcal { N } ( 0 , I _ { D } )$ independently of $C ,$ , and view-specific $N _ { v }$ independent of $( C , U )$ . Set $X ^ { ( v ) } = ( C , U , N _ { v } )$ and $F ( c , u , n ) = \mathrm { r e s h a p e } _ { \ell \times d } ( u )$ . The first input coordinate determines $C ,$ whereas the encoder retains only the shared second coordinate. Therefore $H ( C \mid$ $X ^ { ( v ) } ) = 0 , Z ^ { ( 1 ) } = Z ^ { ( 2 ) }$ almost surely, vec $Z ^ { ( v ) } \sim \mathcal { N } ( 0 , I _ { D } )$ , and $I ( C ; Z ^ { ( v ) } ) = 0$ . Disagreement is nonnegative and equals zero, proving global optimality for that objective. □

Independence implies $p ( C \mid Z ) = p ( C )$ . Thus every readout, including randomized ones with randomness independent of C, has sequence error at least $1 - \operatorname* { m a x } _ { c } p ( c ) ;$ ; the optimal log-loss is $H ( C )$ . For two equiprobable orders $( a , \bar { b } )$ and $( b , a )$ , these are $1 / 2$ and log 2. Each row $z _ { t }$ is standard Gaussian, as is $\ell ^ { - 1 / 2 } \sum _ { t } z _ { t }$ . Ordinary mean pooling instead has covariance $I _ { d } / \ell ;$ this scaling is distinct from normalizing vector length.

For a finite-variance vector target T and $m ( Z ) = \mathbb { E } [ T \mid Z ]$ , expansion and conditional centering give

$$
\begin{array} { r } { \mathbb { E } \| T - h ( Z ) \| ^ { 2 } = \mathbb { E } \| T - m ( Z ) \| ^ { 2 } + \mathbb { E } \| m ( Z ) - h ( Z ) \| ^ { 2 } . } \end{array}\tag{18}
$$

The cross term vanishes because $\mathbb { E } [ T - m ( Z ) \mid Z ] = 0$ . If T is the one-hot encoding of $C \perp Z$ the Bayes risk is $\begin{array} { r } { 1 - \sum _ { c } p ( c ) ^ { 2 } > 0 } \end{array}$ . A perfectly fitted conditional mean therefore need not imply preservation of the original target. Likewise, the Gaussian construction is a population existence result, not a claim that a finite-batch regularizer returns zero or that optimization finds this encoder.

## A.4 FINITE-STRING CONSTRUCTION

A deterministic encoder of countably many strings has countable image, which a nondegenerate Gaussian assigns probability zero. Exact equality of these laws is impossible. We instead construct, for every $\epsilon > 0 .$ , a finite code V with zero target information, exact identity covariance, and $W _ { 2 } ( \mathcal { L } ( \dot { V } ) , \dot { N } ( 0 , I _ { D } ) ) < \epsilon .$

Let $G \sim \mathcal { N } ( 0 , I _ { D } )$ independently of $C .$ . For an integer $m \geq 1$ , clip coordinates to $[ - m , m ]$ and round to a grid of spacing $1 / m$ , symmetrically at ties. The resulting $\bar { Q } _ { m } ( G )$ has finite support, zero mean, and covariance $s _ { m } ^ { 2 } I _ { D }$ with $s _ { m } > 0$ , because identical odd coordinate maps preserve symmetry and independence and each coordinate remains nonconstant. The $L ^ { 2 }$ triangle inequality gives

$$
\Vert Q _ { m } ( G ) - G \Vert _ { L ^ { 2 } } \leq \frac { \sqrt { D } } { 2 m } + \Vert \exp ( G , - m , m ) - G \Vert _ { L ^ { 2 } } \longrightarrow 0 .\tag{19}
$$

The clipping term converges by dominated convergence using $\mathbb { E } \| G \| ^ { 2 } < \infty$ . Consequently $s _ { m } \to 1$ Set $V _ { m } = \bar { Q } _ { m } ( G ) / s _ { m }$ . Then $\bar { \mathbb { E } } V _ { m } = 0 , \mathrm { C o v } ( V _ { m } ) \bar { = } I _ { D }$ , and

$$
\begin{array} { r } { \| V _ { m } - G \| _ { L ^ { 2 } } \leq | s _ { m } ^ { - 1 } - 1 | \| Q _ { m } ( G ) \| _ { L ^ { 2 } } + \| Q _ { m } ( G ) - G \| _ { L ^ { 2 } } \longrightarrow 0 . } \end{array}\tag{20}
$$

This coupling bounds $W _ { 2 } ( \mathcal { L } ( V _ { m } ) , \mathcal { N } ( 0 , I _ { D } ) )$ by $\| V _ { m } - G \| _ { L ^ { 2 } }$ . Choose m large enough for the prescribed ϵ.

Let $J _ { m }$ index the finite support of $V _ { m }$ . Encode $( C , J _ { m } , N _ { v } )$ injectively as a finite string, with finitevalued nuisances $N _ { v }$ . Such an encoding exists because the tuple space is finite. The input decoder recovers $C ;$ the representation decoder returns the vector indexed by $J _ { m }$ . Sharing $J _ { m }$ across views gives exact agreement, and $J _ { m } \perp C$ gives zero target information. This establishes the finite-string counterpart of Theorem 2 on a synthetic channel.

Approximation is in Wasserstein-2, not total variation or ${ \mathrm { K L } } { \mathrm { : } }$ the discrete law remains singular to the Gaussian. For any unit u and frequency ω, however,

$$
\begin{array} { r } { \left| \mathbb { E } e ^ { \mathrm { i } \omega u ^ { \top } V _ { m } } - e ^ { - \omega ^ { 2 } / 2 } \right| \leq | \omega | \mathbb { E } \| V _ { m } - G \| \leq | \omega | \| V _ { m } - G \| _ { L ^ { 2 } } . } \end{array}\tag{21}
$$

The first inequality follows from $| e ^ { \mathrm { i } a } - e ^ { \mathrm { i } b } | \leq | a - b |$ . Squaring and integrating on bounded frequencies shows convergence of the population characteristic-function diagnostic in Appendix B.3. Finite-sample fluctuations remain present even under an exact Gaussian null.

## A.5 CANONICAL ANCHORS AND CONDITIONALLY INDEPENDENT VIEWS

ProofofProposition 3. For $X ^ { ( 0 ) } = a ( C )$ and a deterministic shared encoder, exact agreement forces $Z ^ { ( v ) } = F { ( a ( C ) ) } = : g ( C )$ . Since C is finite-valued, $H ( Z \mid C ) = 0$ and $I ( C ; Z ) = H ( Z )$ . Zero mutual information implies zero entropy and an almost-sure constant. Nonconstant g must retain some information, but need not be injective: if $C$ contains two independent bits and g retains only one, one bit remains unrecoverable. □

More generally, let $C = ( A , B )$ encode two independent, nonconstant finite factors as an equal-length canonical sequence. An agreeing encoder $Z = { \dot { h } } ( A )$ ) with injective h satisfies

$$
I ( C ; Z ) = H ( A ) > 0 , \qquad H ( C \mid Z ) = H ( B ) > 0 .\tag{22}
$$

Every view can contain enough information to recover $( A , B )$ , while its encoder retains only A. Agreement with a clean anchor then holds exactly because the anchor is encoded in the same way. The example establishes partial retention under anchoring; it imposes no Gaussian marginal. Its relevance to CL is that access to clean text during training does not itself force the latent target to distinguish every canonical sequence.

If views are conditionally independent and identically distributed given C, and their encoded outputs have finite second moments, then

$$
\begin{array} { r } { \mathbb { E } [ \| Z ^ { ( 1 ) } - Z ^ { ( 2 ) } \| _ { F } ^ { 2 } \mid C ] = 2 \mathbb { E } [ \| \operatorname { v e c } Z \| ^ { 2 } \mid C ] - 2 \| \mathbb { E } [ \operatorname { v e c } Z \mid C ] \| ^ { 2 } = 2 \operatorname { t r } \operatorname { C o v } ( \operatorname { v e c } Z \mid C ) . } \end{array}\tag{23}
$$

Conditional independence factors the cross moment. Zero disagreement forces the nonnegative conditional variance to vanish, hence $Z = g ( C )$ almost surely and the preceding argument applies. Shared nuisance in Theorem 2 violates this conditional-independence premise.

## A.6 FIXED TOKEN HEADS AND LOCAL READOUTS

ProofofProposition 4. For finite logits u, the softmax probabilities satisfy $\log ( p _ { i } / p _ { j } ) = u _ { i } - u _ { j }$ Two softmax outputs therefore agree exactly when their logits differ by a constant multiple of 1. For logits $W ( z + \delta ) { \bar { + } } b$ and $W z + b .$ , this is $P _ { K } \dot { W } \delta = 0$ . Since rank $( P _ { K } ) \dot { = } K - 1 , r _ { W } \le \mathrm { m i n } ( d , K - 1 )$ and rank–nullity gives dim ker $A = d - r _ { W }$

The product $\begin{array} { r } { q _ { \mathrm { p r o d } } ( C \mid Z , S ) = \prod _ { t } q _ { W } ( C _ { t } \mid z _ { t } ) } \end{array}$ is normalized because each factor sums to one. Its population log-loss is $\ell \mathcal { L } _ { \mathrm { t o k } }$ . Lemma 1 gives

$$
\ell \mathcal { L } _ { \mathrm { t o k } } = H ( C \mid Z , S ) + \mathbb { E } D _ { \mathrm { K L } } ( p \Vert q _ { \mathrm { p r o d } } ) \geq H ( C \mid Z , S ) \geq \Delta _ { \mathrm { i n f o } } \geq 0 .\tag{24}
$$

This model factorization imposes no independence assumption on the true tokens. For variable L and $S = L$ , condition on each length and average, obtaining $\begin{array} { r } { H ( C \mid Z , L ) \leq \operatorname { \mathbb { E } } \sum _ { t = 1 } ^ { L } - \log q _ { W } ( C _ { t } \mid z _ { t } ) } \end{array}$ Prefix-only supervision bounds prefix uncertainty only. 口

The complete probability vector identifies Az through $P _ { K } \log p = A z + P _ { K } b ;$ a scalar CE value need not. For one label $c ,$

$$
\nabla _ { z } ( - \log p _ { c } ) = \boldsymbol { W } ^ { \top } ( p - e _ { c } ) , \qquad \nabla _ { z } ^ { 2 } ( - \log p _ { c } ) = \boldsymbol { W } ^ { \top } ( \mathrm { d i a g } ( p ) - p p ^ { \top } ) \boldsymbol { W } .\tag{25}
$$

For any direction δ, the Hessian quadratic form equals $\operatorname { V a r } _ { k \sim p } ( ( W \delta ) _ { k } )$ . Since all probabilities are positive, it vanishes exactly when W δ is constant across classes, equivalently $A \delta = 0$ . Thus the Hessian has kernel ker A and rank $r _ { W }$ . These are fixed-head statements; joint training can change the observable subspace.

At fixed length ℓ, let $Z _ { - t }$ denote all canvas rows except $z _ { t } .$ . Each unrestricted local factor is optimized by its local posterior, so

$$
\begin{array} { l } { \underset { \left\{ q _ { t } \right\} } { \operatorname* { i n f } } \mathbb { E } [ - \log \prod _ { t } q _ { t } ( C _ { t } \mid z _ { t } , S ) ] = \displaystyle \sum _ { t } H ( C _ { t } \mid z _ { t } , S ) } \\ { = \displaystyle \sum _ { t } H ( C _ { t } \mid Z , S ) + \sum _ { t } I ( C _ { t } ; Z _ { - t } \mid z _ { t } , S ) } \\ { = H ( C \mid Z , S ) + \mathrm { T C } ( C \mid Z , S ) + \displaystyle \sum _ { t } I ( C _ { t } ; Z _ { - t } \mid z _ { t } , S ) . } \end{array}\tag{26}
$$

Here $\begin{array} { r } { \mathrm { T C } ( C \mid Z , S ) = \sum _ { t } H ( C _ { t } \mid Z , S ) - H ( C \mid Z , S ) } \end{array}$ is conditional total correlation, the dependence remaining among target tokens after observing the canvas. It is nonnegative because it equals the expected KL divergence between $p ( C \mid Z , \breve { S ) }$ and $\textstyle \prod _ { t } p ( C _ { t } \mid Z , S )$ . The positional terms measure additional information from other positions beyond $z _ { t } ,$ including possible synergistic dependencies. With independent fair bits and $Z \dot { = } \left( C _ { 2 } , C _ { 1 } \right)$ , the full canvas determines ${ \dot { C } } ,$ but the local optimum is 2 log 2; a sequence readout can swap the positions and recover $C$ exactly. A shared linear head is more restricted than the unrestricted factors and may incur additional loss.

## A.7 POPULATION RIDGE RISK

Under Assumption 2, $\mathbb { E } [ Z _ { \Sigma } Z _ { \Sigma } ^ { \top } ] = \Sigma$ and $\mathbb { E } [ Z _ { \Sigma } Y ] = \Sigma ^ { 1 / 2 } \theta$ , since $\mathbb { E } [ G \varepsilon ] = \mathbb { E } [ G \mathbb { E } ( \varepsilon \mid G ) ] = 0$ Thus

$$
J ( w ) = \mathbb { E } Y ^ { 2 } - 2 w ^ { \top } \Sigma ^ { 1 / 2 } \theta + w ^ { \top } ( \Sigma + \mu I ) w .\tag{27}
$$

Its positive-definite Hessian $2 ( \Sigma + \mu I )$ gives the unique solution $w _ { \Sigma } = ( \Sigma + \mu I ) ^ { - 1 } \Sigma ^ { 1 / 2 } \theta$ . Functions of $\bar { \Sigma }$ commute, so

$$
Y - w _ { \Sigma } ^ { \top } Z _ { \Sigma } = \varepsilon + \theta ^ { \top } [ I - \Sigma ( \Sigma + \mu I ) ^ { - 1 } ] G = \varepsilon + \mu \theta ^ { \top } ( \Sigma + \mu I ) ^ { - 1 } G .\tag{28}
$$

The noise cross term is zero. Taking the expectation of the squared residual gives $\sigma ^ { 2 } + \mu ^ { 2 } \theta ^ { \top } ( \Sigma +$ $\mu I ) ^ { - 2 } \theta .$ , proving (9). Since $G = \Sigma ^ { - 1 / 2 } Z _ { \Sigma } ,$ , all representations yield Bayes risk $\sigma ^ { 2 }$ . Excess risk arises from the fixed ridge penalty, not information loss. The comparison fixes θ in the factor coordinates; fixing coefficients in $Z _ { \Sigma }$ coordinates while changing $\Sigma$ would instead change the target.

## A.8 EFFECTIVE-RANK THRESHOLD WITH ALL INTERMEDIATE STEPS

The basis and the target vector θ remain fixed while $\rho$ varies. In particular, $q _ { 0 } = \| \theta \| ^ { 2 }$ and $\eta =$ $\textstyle \sum _ { i \leq r } \theta _ { i } ^ { 2 } / q _ { 0 }$ are constants in the derivatives below. The two eigenvalues are

$$
a ( \rho ) = \frac { d - ( d - r ) \rho } { r } , \qquad b ( \rho ) = \rho , \qquad a ^ { \prime } = - \frac { d - r } { r } , \qquad b ^ { \prime } = 1 .\tag{29}
$$

There are r copies of a and $d - r$ copies of b. Their sum is $d ,$ so their normalized weights are $a / d$ and $b / d .$ . Restating the definition in Eq. (8) makes the dependence on $\rho$ explicit,

$$
h ( \rho ) : = \log r _ { \mathrm { e f f } } ( \Sigma ( \rho ) ) = - r \frac { a } { d } \log \frac { a } { d } - ( d - r ) \frac { b } { d } \log \frac { b } { d } , \qquad r _ { \mathrm { e f f } } ( \Sigma ( \rho ) ) = \exp ( h ( \rho ) ) .\tag{30}
$$

Constructing a path toward isotropy. Fix the eigenbasis; split its directions into blocks of sizes r and $d - r , 1 \leq r < d$ . Give the second block variance $b = \rho \in ( 0 , 1 )$ . The trace constraint determines first-block variance a:

$$
\underbrace { \vphantom { \left( \sum _ { r = 1 } ^ { m } \left( \mathrm { d } - r \right) \mathrm { d } _ { \rho } \right)}  } _ { \mathrm { f i r s t ~ b l o c k } } + \underbrace { ( d - r ) \rho } _ { \mathrm { s e c o n d ~ b l o c k } } = d \Longrightarrow \quad a = \frac { d - ( d - r ) \rho } { r } , \qquad b = \rho .\tag{31}
$$

The spectrum is $\underbrace { ( a , \ldots , a } _ { r } , \underbrace { b , \ldots , b } _ { d - r } )$ . As $\rho$ increases, $a > 1 > b > 0$ both approach 1; every map

remains invertible. This analytical family is not an assumed trained-model spectrum.

Let $q _ { 0 } = \lVert \theta \rVert ^ { 2 }$ and $\begin{array} { r } { \eta = \sum _ { i < r } \theta _ { i } ^ { 2 } / q _ { 0 } } \end{array}$ . Fixed target and basis keep η constant, so Eq. (9) becomes

$$
R _ { \mu } ( \rho ) = \mu ^ { 2 } q _ { 0 } \left[ \frac { \eta } { ( a + \mu ) ^ { 2 } } + \frac { 1 - \eta } { ( b + \mu ) ^ { 2 } } \right] .\tag{32}
$$

Increasing $\rho$ raises the first contribution and lowers the second. Their balance gives the threshold below.

ProofofTheorem 5. Step 1. Differentiate the effective-rank function. For any positive differentiable $x ( \rho )$ , the product and chain rules give

$$
\frac { d } { d \rho } [ x \log x ] = x ^ { \prime } \log x + x \frac { x ^ { \prime } } { x } = x ^ { \prime } ( \log x + 1 ) .\tag{33}
$$

Apply this identity to $x = a / d$ and $x = b / d ,$ noting that d is fixed,

$$
\begin{array} { l } { { \displaystyle h ^ { \prime } ( \rho ) = - \frac { r a ^ { \prime } } d \left( \log \frac a d + 1 \right) - \frac { ( d - r ) b ^ { \prime } } d \left( \log \frac b d + 1 \right) } } \\ { { \displaystyle \qquad = \frac { d - r } d \left( \log \frac a d + 1 \right) - \frac { d - r } d \left( \log \frac b d + 1 \right) } } \\ { { \displaystyle \qquad = \frac { d - r } d \log \frac a b > 0 . } } \end{array}\tag{34}
$$

The last inequality follows from $d - r > 0$ and $a > b > 0$ for $0 < \rho < 1$ . Differentiating the exponential in Eq. (30) then gives

$$
{ \frac { d r _ { \mathrm { e f f } } } { d \rho } } = e ^ { h ( \rho ) } h ^ { \prime } ( \rho ) = r _ { \mathrm { e f f } } h ^ { \prime } ( \rho ) > 0 .\tag{35}
$$

Thus increasing $\rho$ strictly increases effective rank on the open interval.

Step 2. Differentiate prediction risk. Grouping Eq. (9) by the two blocks yields

$$
R _ { \mu } ( \rho ) = \mu ^ { 2 } q _ { 0 } \left[ \eta ( a + \mu ) ^ { - 2 } + ( 1 - \eta ) ( b + \mu ) ^ { - 2 } \right] .\tag{36}
$$

Since $d ( x + \mu ) ^ { - 2 } / d \rho = - 2 x ^ { \prime } ( x + \mu ) ^ { - 3 } ,$

$$
R _ { \mu } ^ { \prime } ( \rho ) = \mu ^ { 2 } q _ { 0 } \left[ - \frac { 2 \eta a ^ { \prime } } { ( a + \mu ) ^ { 3 } } - \frac { 2 ( 1 - \eta ) b ^ { \prime } } { ( b + \mu ) ^ { 3 } } \right] = 2 \mu ^ { 2 } q _ { 0 } \left[ \frac { \eta ( d - r ) } { r ( a + \mu ) ^ { 3 } } - \frac { 1 - \eta } { ( b + \mu ) ^ { 3 } } \right] .\tag{37}
$$

The positive term is the cost of reducing variance in the first block; the negative term is the benefit of increasing variance in the second block.

Step 3. Solve for the sign change. Multiply the bracket by $r ( a + \mu ) ^ { 3 } ( b + \mu ) ^ { 3 } > 0$ . This preserves its sign and gives

$$
\begin{array} { r l } & { \eta ( d - r ) ( b + \mu ) ^ { 3 } - ( 1 - \eta ) r ( a + \mu ) ^ { 3 } } \\ & { \quad = \eta \big [ ( d - r ) ( b + \mu ) ^ { 3 } + r ( a + \mu ) ^ { 3 } \big ] - r ( a + \mu ) ^ { 3 } } \\ & { \quad = \big [ ( d - r ) ( b + \mu ) ^ { 3 } + r ( a + \mu ) ^ { 3 } \big ] ( \eta - \eta _ { * } ) , } \end{array}\tag{38}
$$

where

$$
\eta _ { * } = \frac { r ( a + \mu ) ^ { 3 } } { r ( a + \mu ) ^ { 3 } + ( d - r ) ( b + \mu ) ^ { 3 } } .\tag{39}
$$

All prefactors are positive. Therefore sign ${ \mathrm { \Omega } } _ { \mathrm { i } } ( R _ { \mu } ^ { \prime } ) = \mathrm { s i g n } ( \eta - \eta _ { \ast } )$

Step 4. Change from $\rho$ to effective rank. Because Eq. (35) is strictly positive, effective rank is locally invertible as a function of $\rho ,$ and

$$
\frac { d R _ { \mu } } { d r _ { \mathrm { e f f } } } = \frac { d R _ { \mu } / d \rho } { d r _ { \mathrm { e f f } } / d \rho } = \frac { R _ { \mu } ^ { \prime } ( \rho ) } { r _ { \mathrm { e f f } } h ^ { \prime } ( \rho ) } .\tag{40}
$$

Its denominator is positive, so the sign is again sign $( \eta - \eta _ { * } )$ , proving the theorem.

The threshold tends to $r / d$ as $\rho \uparrow 1$ . It can change relative to a fixed η along the path, which is why the theorem states a local derivative. At $\rho = 1 , h ^ { \prime } = 0$ and the ratio above is not asserted. For $\mu \to 0$ at a fixed positive-definite covariance, the excess risk tends to zero.

For $d = 2 , r = 1$ , and $\mu = q _ { 0 } = 1$ , moving from $\rho = 0 . 2$ to 0.8 changes the eigenvalues from $( 1 . 8 , 0 . 2 )$ to $( 1 . 2 , 0 . 8 )$ and effective rank from 1.3841 to 1.9601. Risk increases from 0.1276 to 0.2066 for $Y = G _ { 1 }$ , and decreases from 0.6944 to 0.3086 for $Y = G _ { 2 }$ . Figure 4 plots these exact population quantities.

![](images/fc72f9d8225d7399a546d4d87eec6a6553b72601ee2266fc0d59568579dd7150.jpg)

(b) Target-dependent risk  
![](images/46da4c6ba84bb3e92cc9938204bfb9d1b342b973e2417bbf5d8a252f475db1ac.jpg)  
Figure 4: The same increase in effective rank can help or hurt a fixed readout. (a) Increasing $\rho$ transfers variance from the first direction to the second while keeping $a + b = 2$ . (b) Ridge risk rises when the target uses the first direction and falls when it uses the second. The map remains invertible for every $0 < \rho < 1$ , so both targets remain fully represented. These are exact population curves for $d = 2 , r = 1 , \mu = q _ { 0 } = 1$ , and zero observation noise, illustrating Theorem 5.

## A.9 MINIMAX COMPATIBILITY AND SCOPE RELATIVE TO LEJEPA

For fixed Σ, $\begin{array} { r } { R _ { \mu } = \sum _ { i } b _ { i } \theta _ { i } ^ { 2 } } \end{array}$ with $b _ { i } = \mu ^ { 2 } / ( \lambda _ { i } + \mu ) ^ { 2 }$ . Under $\begin{array} { r } { \sum _ { i } \theta _ { i } ^ { 2 } \ = \ q _ { 0 } } \end{array}$ , the maximum is q<sub>0</sub> max<sub>i</sub> $b _ { i }$ , attained by placing all energy in a minimum-eigenvalue direction. This proves the main-text supremum. The trace constraint implies $\lambda _ { \operatorname* { m i n } } \le 1$ , and the supremum decreases strictly with $\lambda _ { \mathrm { m i n } }$ . Equality $\lambda _ { \operatorname* { m i n } } = 1$ forces every eigenvalue to equal one, hence uniquely $\Sigma = I _ { d }$ . This is uniqueness of covariance within the assumed Gaussian family, not uniqueness among arbitrary distributions.

LeJEPA’s linear results compare fixed feature spans and energy. Its nonlinear results assume smooth embedding densities and regression functions; the radial-neighbor result additionally uses a meanzero isotropic target-gradient prior, while the kernel result controls a worst-case integrated-bias bound over a smooth function class (Balestriero & LeCun, 2025). Atomic laws induced by deterministic string encoders do not automatically meet the density assumptions. Likewise, the fixed ridge target has gradient $\Sigma ^ { - 1 / 2 } \theta$ , whose outer product is rank one for $d > 1$ , rather than a positive isotropic gradient second moment. These examples delimit direct applicability; continuous encoder noise or a different target prior changes the assumptions. Our information decomposition addresses target retention, whereas the ridge analysis preserves information and isolates geometry–readout interaction.

## B IMPLEMENTATION AND EVALUATION DETAILS

## B.1 READOUT IMPLEMENTATION

Retrieval matches a corrupted query to its canonical target in a fixed gallery. Pooling compares mean-canvas cosines; relative-position scoring preserves coarse order. For $\mathbf { \bar { \boldsymbol { Z } } } \in \mathbb { R } ^ { M \times d }$ , linearly interpolate $\widetilde { z } _ { j }$ at $j ( M - 1 ) / 1 5$ and define

$$
\phi _ { 1 6 } ( Z ) = \frac { 1 } { 4 } \mathrm { c o n c a t } _ { j = 0 } ^ { 1 5 } \frac { \widetilde { z } _ { j } } { \operatorname* { m a x } ( \| \widetilde { z } _ { j } \| , 1 0 ^ { - 1 2 } ) } , \qquad s _ { 1 6 } ( U , V ) = \langle \phi _ { 1 6 } ( U ) , \phi _ { 1 6 } ( V ) \rangle .\tag{41}
$$

The factor $1 / 4 = 1 / \sqrt { 1 6 }$ makes the inner product a mean cosine for nonzero rows. This compares lengths without target-token alignment but does not preserve every detail or certify word-order understanding.

Native reconstruction uses per-position argmax and stops at PAD, requiring the whole sequence rather than candidate selection. These diagnostics fit no new probes and exclude untrained CL-LP/LT heads. Section 6.4 separately trains a downstream speech readout. Each task measures specified readout access, not $\Delta _ { \mathrm { { i n f o } } }$ itself.

## B.2 ARCHITECTURE AND LENGTH POLICIES

The application architecture uses width 512, eight attention heads, four encoder blocks, two decoder blocks, feed-forward width 2048, and a 2048-position cap. Source masks exclude padding from resampling and attention; output masks exclude batch padding from losses. Valid queries originate from resampled source states, not zero padding. Application states retain final decoder layer normalization; mechanism runs remove it. With zero-based source and query indices, the resampler uses weights $\omega _ { t j }$ ∝ ex $) ( - ( j - c _ { t } ) ^ { 2 } / 0 . 3 )$ ) normalized over valid source positions, where $c _ { t } = t ( \bar { N } - 1 ) / ( M \bar { - } 1 )$ for $M > 1$ and $c _ { 0 } = 0$ for $M = 1$ . It returns the input unchanged when $M = N$ . Gaussian diagnostics on raw mechanism states therefore do not transfer automatically to application states.

$U \{ a , \ldots , b \}$ and $U [ a , b ]$ below denote discrete and continuous uniform draws. The prediction $\widehat { L }$ is the learned target length. The length head attention-pools detached encoder states and uses the source log length. Training samples near-oracle $M { \bar { = } } { L } + U \{ 0 , \dots , 8 \}$ , overestimated $M = \lceil L U [ 1 . 1 5 , 1 . { \overset { \cdot } { 8 } } 0 ] \rceil$ , or underestimated $M = \lceil L U [ 0 . 5 0 , 0 . 9 5 ] \rceil$ , clipped to the supported range. At progress $0 , 0 . 1 5 , \dot { 0 } . 4 0 , 0 . 6 5 , 1$ , the corresponding mixture probabilities are $( 1 , 0 , 0 ) , ( 1 , 0 , 0 )$ (0.5, 0.5, 0), (0.35, 0.45, 0.20), and (0.20, 0.60, 0.20), with smoothstep interpolation. Only lengths, not target-token identities, enter decoder routing.

The reported native-head evaluation uses a single canvas, per-position argmax, and stopping at the first $\mathrm { P A D } ;$ it does not expand and retry. PL setting practical routing uses $M = \operatorname* { m a x } ( 1 , \lceil \widehat { L } \rceil )$ for both queries and clean candidates. CL setting input routing uses $M = { \tilde { N } }$ , the number of input bytes, as a length-sensitivity control rather than a trained length predictor. Oracle routing uses $M = L$ . Capacity violations fail explicitly rather than truncate. The same extracted states feed retrieval and the native head. Oracle token cross-entropy covers every canonical target byte; empirical corpus loss per byte is reported descriptively, not substituted directly into the population sequence-entropy bound.

## B.3 RECOVERABLE CHANNEL AND VISREG

The channel escapes literal # as ## and inserts removable $\# \tilde { \mathbf { \Gamma } }$ markers; held-out patterns use $\# ?$ at word boundaries. A left-to-right inverse removes insertion markers and reverses escaping without the corruption seed. Thus case, punctuation, numbers, and canonical bytes remain recoverable. Natural spelling errors form a separate channel because unique recovery can fail. Canonical-content hashes determine splits and group all related views.

The seven configured views comprise two deterministic clean copies, three easy corruptions, and two stronger corruptions. The global-view indices are $\mathcal { G } = \{ 0 , 2 \}$ , indexed from zero. The available training entry point sets $\mathfrak { g l o b a l \_ v i e w s } = ( 0 , 2 )$ , and the loss implementation computes $z \left[ : , \right.$ global views].mean(1) without detaching that center. This differs from PL0’s detached view-0 target. For batch size B, view count $V$ , target byte $c _ { i t }$ , token probability $p _ { i v t } .$ , state $z _ { i v t }$ , and target length $L _ { i } ,$ let $\begin{array} { r } { \bar { z } _ { i t } = | \mathcal { G } | ^ { - 1 } \sum _ { v \in \mathcal { G } } z _ { i v t } } \end{array}$ . The implemented predictive terms are

$$
\mathcal { L } _ { \mathrm { l a t } } = \frac { 1 } { B V } \sum _ { i , v } \frac { 1 } { L _ { i } d } \sum _ { t = 1 } ^ { L _ { i } } \Vert z _ { i v t } - \bar { z } _ { i t } \Vert ^ { 2 } ,\tag{42}
$$

$$
\mathcal { L } _ { \mathrm { g r o u n d } } = \frac { 1 } { B V } \sum _ { i , v } \frac { 1 } { L _ { i } } \sum _ { t = 1 } ^ { L _ { i } } - \log p _ { i v t } ( c _ { i t } ) .\tag{43}
$$

The latent center receives gradients. The latent runs have no canonical-token CE contribution to backbone training. The grounded runs use the token head on undetached states. Neither branch includes the application auxiliary losses. Table 6 specifies the training configurations.

Pooled regularization uses $r _ { i v } = L _ { i } ^ { - 1 } \sum _ { t = 1 } ^ { L _ { i } } z _ { i v t }$ . Sampled-token regularization draws $T _ { i }$ uniformly from $\{ 1 , \ldots , L _ { i } \}$ and sets $r _ { i v } = z _ { i v T _ { i } }$ , using the same position across all views of an item. Thus both use $B$ vectors per view. Token sampling targets a sentence-balanced mixture of valid positions, not Gaussianity at every position or of the joint canvas. Position and projection draws use separate seeded generators. VISReg uses 256 unit projections, coordinate standard deviations with divisor $B ,$ unit center/scale/shape weights, and the detached standard-deviation floor 0.1 in Eq. (48). Finite-batch Gaussian samples need not attain zero penalty.

Geometry diagnostic. The evaluation code separately computes a characteristic-function discrepancy. For n vectors $r _ { i } , J$ unit directions $u _ { j }$ , frequency $\omega ,$ and imaginary unit i,

$$
D _ { \mathrm { C F } } = \frac { 1 } { 3 J } \sum _ { j = 1 } ^ { J } \int _ { 0 } ^ { 3 } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } e ^ { \mathrm { i } \omega u _ { j } ^ { \top } r _ { i } } - e ^ { - \omega ^ { 2 } / 2 } \right| ^ { 2 } d \omega .\tag{44}
$$

The integral uses trapezoidal quadrature at 31 equally spaced frequencies; $J = 3 2$ unit directions are seeded for evaluation. This is a diagnostic, not the CL setting training loss or a calibrated $p \textmd { - }$ value. It has no sample-size multiplier. Evaluation also records covariance eigenvalues, covariance effective rank, and deviation from identity, with sampling units and finite-sample Gaussian references. This distinction permits assessment of geometry independently of the VISReg training objective.

No new readout is fitted in the reported evaluation. Native heads are assessed only where training supplied a token objective: PL0’s detached-input head is included, while the untrained heads of CL-LP/CL-LT are excluded. Geometry uses all 40,000 sentence means and one deterministic valid token per sentence, before post-hoc centering or whitening. The token statistic is a position mixture, not a joint-canvas Gaussianity test. A separate set of 1,024 nonoverlapping validation examples defines the centering vector as the average of clean sentence means; no label fitting or test-dependent centering is used. Finite-sample Gaussian reference files exist in the evaluation output, but no calibrated Gaussian goodness-of-fit conclusion is inferred from the summary statistics reported here.

## B.4 APPLICATION OBJECTIVE AND REGULARIZATION

Let $v = 0$ denote the clean anchor, $v > 0$ its training corruptions (natural corruptions are not certified invertible), and $M _ { i v }$ the requested canvas length. The coefficients $\eta _ { \mathrm { e } } , \gamma , \alpha , \beta , \zeta$ weight empty-tail, length, invariance, negation, and geometry terms, respectively. The combined application objective is Eq. (12) in the main text.

Content, termination, and length. Set $s _ { i v } = \operatorname* { m i n } ( M _ { i v } , L _ { i } + 4 ) , y _ { i t } = c _ { i t }$ for $t \leq L _ { i }$ and PAD otherwise, and $\mathcal { T } = \{ ( i , v , t ) : s _ { i v } < t \leq M _ { i v } \}$ . Then

$$
\mathcal { L } _ { \mathrm { C E } } = \frac { 1 } { B V } \sum _ { i , v } \frac { 1 } { s _ { i v } } \sum _ { t \leq s _ { i v } } - \log p _ { i v t } ( y _ { i t } ) , \qquad \mathcal { L } _ { \mathrm { e m p t y } } = \frac { 1 } { \left| T \right| } \sum _ { ( i , v , t ) \in T } \sum _ { k = 1 } ^ { 2 5 7 } p _ { i v t k } ^ { 2 } .\tag{45}
$$

Here $p _ { i v t k }$ is the predicted probability of class k at output position t. CE grounds ordered bytes and a short PAD band. Empty-tail loss is zero for $\tau = \varnothing ;$ otherwise its minimum $1 / 2 5 7$ encourages uniform tail predictions. Underestimated canvases supervise only a prefix and do not yield a fullcontent guarantee. For source length $N _ { i v } .$ , a head using detached encoder states predicts the log length ratio $\rho _ { \mathrm { l e n } , i v }$ with

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { l e n } } = \frac { 1 } { 2 } \operatorname* { m e a n } _ { i , v } | \log L _ { i } - \log N _ { i v } - \rho _ { \mathrm { l e n } , i v } | , \qquad \widehat { L } = N \exp \bigl ( \mathrm { c l i p } ( \rho _ { \mathrm { l e n } } , - 2 . 5 , 2 . 5 ) \bigr ) . } \end{array}\tag{46}
$$

clip(x, a, b) clips x to [a, b]. This symmetric log-ratio loss trains the length head; routing is detached.   
PAD-based text generation and direct latent extraction use distinct length policies (Appendix B.2).

Surface invariance and polarity. Let $r _ { i v }$ average the first min $( L _ { i } , M _ { i v } )$ states and $\begin{array} { r l } { q _ { i v } } & { { } = } \end{array}$ $r _ { i v } / \operatorname* { m a x } ( \lVert r _ { i v } \rVert , \epsilon )$ , where $\epsilon > 0$ prevents division by zero. Both sides of pooled invariance receive gradients:

$$
\mathcal { L } _ { \mathrm { i n v } } = \frac { 1 } { B ( V - 1 ) } \sum _ { i , v > 0 } ( 1 - \langle q _ { i v } , q _ { i 0 } \rangle ) , \qquad \ell _ { P \to N } ^ { ( v ) } = [ s _ { - } - s _ { + } + m _ { 1 } ] _ { + } + [ s _ { U } - s _ { - } + m _ { 2 } ] _ { + } .\tag{47}
$$

Write $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ and let $m _ { 1 } , m _ { 2 }$ be hinge margins; $\ell _ { P  N } ^ { ( v ) }$ is the per-view directional negation loss. For a premise–negation pair $( P , N ) , s _ { + } = \langle q _ { P v } , q _ { P 0 } \rangle , s _ { - } = \langle q _ { P v } , q _ { N 0 } \rangle$ , and $s _ { U }$ averages similarity to detached unrelated anchors. $\mathcal { L } _ { \mathrm { n e g } }$ averages both directions over pairs and corrupted views, with $m _ { 1 } = m _ { 2 } = 0 . 2$ . The hinges favor $s _ { + } > s _ { - } >$ s<sub>U</sub>: negations retain topical relatedness but are not positive views. This ranking bias uses paired data without a semantic teacher; invariance alone does not enforce order.

Geometry and configurations. For vectors $r _ { i v } \in \mathbb { R } ^ { d }$ , our VISReg adaptation (Wu et al., 2026) uses coordinate means $\mu _ { v }$ and standard deviations $\sigma _ { v }$ (divisor B), then matches standardized projections to Gaussian quantiles:

$$
\mathcal { R } _ { \mathrm { V I S } } = \mathrm { m e a n } _ { v , k } [ \mu _ { v k } ^ { 2 } + ( \sigma _ { v k } - 1 ) ^ { 2 } ] + \mathrm { m e a n } _ { v , j , i } \left[ a _ { v j , ( i ) } - \Phi ^ { - 1 } \left( \frac { i } { B + 1 } \right) \right] ^ { 2 } ,\tag{48}
$$

Here k indexes coordinates, j indexes projection directions, and (i) denotes sorted batch order. The standardized state $\widetilde { r } _ { i v } = ( r _ { i v } - \mu _ { v } ) /$ m $\mathrm { \ u x } ( \mathrm { s g } ( \sigma _ { v } ) , 0 . 1 )$ uses coordinatewise division and stop-gradient $\mathrm { s g } ; a _ { v j , ( i ) }$ sorts $u _ { j } ^ { \top } \widetilde { r } _ { i v }$ for unit $u _ { j }$ , and Φ is the standard Gaussian CDF. Center, scale, and shape have unit internal weights; 256 directions are sampled per call and shared across views. Application runs use pooled content states. CL setting runs use pooled or sampled-token states with outer weight 0.01 (Appendix B.3). This single definition applies to both series.

We use $( \eta _ { \mathrm { e } } , \gamma , \alpha , \beta ) = ( 0 . 1 , 1 , 0 . 5 , 0 . 3 )$ . The historical unregularized recipe and PL2 set $\zeta = 0 / 0 . 1$ respectively. The former retains all nongeometric losses. PL0 instead uses $\zeta = 0 . 3$ , detached-anchor latent agreement of weight one, and detached inputs to the token head. Its CE/empty terms train the head but not the backbone. Thus PL0–PL2 compares full recipes, whereas the historical unregularized recipe and PL2 differ only in the configured geometric term (Appendix B.5).

## B.5 APPLICATION LOSSES AND GRADIENT PATHS

Application runs use the same VISReg definition and internal settings as Eq. (48), applied to pooled available-content states. Their outer weights differ from the CL setting: PL0 uses $\zeta = 0 . 3 .$ , PL1 uses 0, and PL2 uses 0.1. Mean pooling during training uses the available canonical-content mask; inference does not receive the true query length for this purpose.

For PL0, canvas lengths are shared across views of an item. Let $\mathcal { M } = \{ ( i , v , t ) : v > 0 , 1 \leq t \leq$ $M _ { i v } \}$ . Its additional latent term is

$$
\mathcal { L } _ { \mathrm { P L 0 , l a t } } = \frac { 1 } { d | \mathcal { M } | } \sum _ { ( i , v , t ) \in \mathcal { M } } \Vert \boldsymbol { z } _ { i v t } - \mathrm { s g } ( \boldsymbol { z } _ { i 0 t } ) \Vert ^ { 2 } .\tag{49}
$$

PL0 uses logits $W \mathrm { s g } ( z ) + b ,$ so both CE and empty-tail losses update the head only. Invariance, negation, geometry, and latent agreement still affect its backbone. PL1/PL2 use $W \bar { z } + b ;$ CE and empty-tail losses can therefore update their backbones. Length-head inputs are detached in every configuration. Invariance differentiates through the corrupted view and clean anchor, while unrelated negation anchors are detached. PL0 is neither the auxiliary-free mechanism arm nor a faithful reproduction of LeJEPA’s complete training recipe.

The mean squared probability sum in (45) satisfies $\textstyle \sum _ { k = 1 } ^ { 2 5 7 } p _ { k } ^ { 2 } \geq 1 / 2 5 7$ by Cauchy–Schwarz, with equality at uniform probabilities. It suppresses confident predictions after the supervised PAD band and does not itself train PAD. Content accuracy excludes the PAD band; sequence metrics count missing suffixes when canvases are too short. An underestimated-canvas CE loss cannot certify sufficiency of the entire target.

## B.6 EXECUTED RETRIEVAL AND SEMANTIC PROTOCOLS

Within each PL/CL suite, candidate identities and paired queries are fixed across all severities. Full-gallery search uses unit-normalized pooled states or Eq. (41); the latter interpolates 16 relative positions and has no absolute-position length-mismatch penalty. The same routing applies to queries and clean candidates. No dynamic time warping, target-token matching, or best permutation enters the score. Centered controls subtract the clean calibration mean before constructing the same vectors.

Table 6: CL setting design. All four use oracle target lengths and VISReg weight 0.01. Only G sends canonicaltoken loss gradients into the representation.
<table><tr><td>Run</td><td>Predictive objective</td><td>Vectors regularized</td></tr><tr><td>CL-GP</td><td>Canonical-token grounding</td><td>Mean of canvas rows</td></tr><tr><td>CL-GT</td><td>Canonical-token grounding</td><td>One sampled row per item</td></tr><tr><td>CL-LP</td><td>Cross-view latent agreement</td><td>Mean of canvas rows</td></tr><tr><td>CL-LT</td><td>Cross-view latent agreement</td><td>One sampled row per item</td></tr></table>

All reported failures remain in the query denominator; a failed query receives zero retrieval credit. No input is silently truncated. All reported PL/CL retrieval conditions have zero failures and full 40,000-candidate coverage.

If g candidates score strictly above a tie of size $e ,$ , the relevant item’s expected Recall@k contribution is min $( \operatorname* { m a x } ( k - g , 0 ) , e ) / e$ and reciprocal rank is $e ^ { - 1 } \sum _ { r = g + 1 } ^ { g + e } 1 / r$ . Exact score ties can yield fractional counts. This is particularly relevant for CL-LT, whose nearly common states produce poor raw cosine discrimination even on clean self-retrieval. Centering is estimated once on the disjoint 1,024-example calibration cohort, rather than fitted to individual evaluation targets. The changed-query LexNorm confidence intervals resample its 1,928 canonical identities, retaining duplicate-associated queries. Intervals for multiple contrasts are descriptive and are not familywise-corrected tests.

Data provenance. Training data come from the English–Singlish G2P compilation (English– Singlish G2P). Its documented source inventory combines LJSpeech transcriptions (LJ) by Keith Ito and Linda Johnson (Ito & Johnson, 2017), LibriSpeech transcriptions (LIBR) by Vassil Panayotov and colleagues (Panayotov et al., 2015), community-authored English Wikipedia text (WIKI) (Wikipedia contributors, n.d.), the NUS SMS corpus (NUS) by Tao Chen and Min-Yen Kan (Chen & Kan, 2013), and CANNOT negation data (CANNOT) by Miriam Anschutz and colleagues (Ansch¨ utz et al., 2023).¨ CANNOT is itself a derived resource produced with a rule-based negation tool. The compilation combines and processes existing resources; it does not confer authorship of their original sentences, annotations, or recordings. We credit the compilation and these source creators separately and claim no new collection of the underlying text.

The corpus documentation describes more than ten million text–phoneme pairs with canonical English text, informal/Singlish variants, and negation pairs. The compilation was assembled by the authors and is available at https://huggingface.co/datasets/avalonai/ english-singlish-g2p. The prepared-data counts below describe the combined corpus used in our experiments. Component sources retain their own attribution and licensing terms.

CANOPE pretraining and the text diagnostics use only graphemic text, encoded as UTF-8 bytes, rather than the corpus’s phoneme modality. The text-side fields are text (canonical sentence), unnormalized text (informal/Singlish variant), and negation; a literal - marks an unavailable variant or negation. Although the corpus also provides eSpeak NG transcriptions in phoneme and phoneme negation, neither field supplies model inputs, targets, or auxiliary supervision. The legacy loader nevertheless requires a nonempty phoneme field when filtering rows; reproduction must retain this field to preserve the selected cohort. These experiments concern text representations and canonical-byte recovery, not G2P prediction. The separate downstream MatchaTTS baseline uses its own phonemizer, not the compilation’s stored phoneme annotations (Appendix D). The corpus-level size describes the source collection; the filtered PL and prepared CL counts below describe the data used in the reported protocols.

The PL loader reproduces the original filtered corpus and seeded validation split: 9,595,548 filtered rows, 8,635,993 training rows, and 959,555 validation rows. The CL prepared corpus has 8,656,553 training, 470,530 validation, and 468,465 test rows, split by the hash of exact canonical text; duplicates remain in one partition. Each suite samples 40,000 unique targets with selection seed 9028 (evaluation seed 9027 plus one), and an additional 1,024 calibration targets with zero overlap. The CL source manifest reports an inverse-channel audit over 4,096 accepted examples and all strengths of both marker channels; the evaluation channel also checks recovery when generating its views. These are validation-cohort analyses, and the PL evaluation does not rerun a complete train/validation content-overlap audit. No numerical result from a separate test partition is substituted for these cohorts.

External extraction. Published checkpoints are princeton-nlp/ sup-simcse-bert-base-uncased (supervised pooler output), tum-nlp/NegMPNet (published sentence-transformers modules), Qwen/Qwen3-Embedding-0.6B (last non-padding token), and google/byt5-small (masked mean of frozen encoder states). All vectors are unit-normalized. The ByT5 result uses neither generation nor task-specific fine-tuning. The configured input cap is 512 tokenizer tokens, further limited by the model interface; NegMPNet’s recorded cap is 75. Capacity checks precede encoding, and the recorded external comparisons have full valid coverage. CANOPE uses width-512 canvases and a 2,048-byte/position cap; Eq. (41) produces an 8,192-coordinate vector, so parameter count alone would not establish retrieval-storage efficiency.

Qwen’s canonical retrieval queries use the instruction: “Retrieve the canonical text with the same meaning as the query, preserving negation, entities, numbers, and word order.” Candidates and symmetric semantic pairs are encoded without that query instruction. For every model, LexNorm targets are deduplicated annotated canonical strings and are held fixed when query whitespace changes. Added whitespace is a controlled surface perturbation and may disappear under a subword tokenizer; robustness differences therefore concern the complete representation pipeline. The primary changed-input subset avoids credit for already canonical queries, while the complete original cohort remains in the accompanying results.

Semantic benchmarks and synthetic controls. SemAntoNeg supplies 3,152 three-choice groups (9,456 scored pairs), using the annotated equivalent candidate as the positive. PAWS uses the 8,000-example labeled final/test split and reports threshold-free ROC AUC. STS-B uses the 1,379-pair sentence-transformers/stsb test split, and the SICK-derived result uses the 9,927-pair mteb/sickr-sts split named test. We label the latter by its packaged source rather than imply the original SICK test cardinality. Both report Spearman correlation. The finite-pair intersection contains every example for these runs, so common-cohort metrics retain the full recorded coverage.

Synthetic controls use 512 base groups per type (negation, number, entity, and role swap), each in both directions, giving 1,024 decisions per type. A noisy query chooses between its same-content anchor and a specified semantic foil; ties receive half credit. Whitespace noise is applied identically to all methods. These template controls do not establish open-domain semantic equivalence or isolate word order from every other cue. We do not report unexecuted state-shuffle, random-encoder, or newly fitted readout controls as empirical findings.

Table 7: Run names and their source identities. Short IDs in the text refer to this convention; raw result files retain the original checkpoint keys.
<table><tr><td>Run</td><td>Source ID</td><td>Defining setting</td></tr><tr><td>PL0/JEPA-based</td><td>a0_jepa_s42</td><td>Detached-anchor latent agreement</td></tr><tr><td>PL1/Baseline</td><td>a10_dec2_s42</td><td>Token supervision, ζ = 0</td></tr><tr><td>PL2/VISReg-Weak</td><td>a2_visreg-s42</td><td>Pooled VISReg, ζ = 0.1</td></tr><tr><td>PL3/VISReg-Stronger</td><td>a3_visreg_kuat_s42</td><td>Pooled VISReg, ζ = 0.5</td></tr><tr><td>PL4/No-Invariance</td><td>a4_no_inv_s42</td><td>α = 0</td></tr><tr><td>PL5/No-Negation</td><td>a5_no_neg-s42</td><td>β = 0</td></tr><tr><td>PL6/No-Empty</td><td>a6_no_empty_s42</td><td>ηe = 0</td></tr><tr><td>PL7/Zero-Second-Margin</td><td>a9_single_hinge_s42</td><td>m2 = 0</td></tr></table>

The historical keys a1\_ce\_s42, a7\_no\_soft\_s42, and a8\_no\_uniform\_s42 are distinct from the revised PL1 and PL7.

Checkpoint and compute scope. All training runs used a combination of NVIDIA A100 GPUs with 80 GB memory and NVIDIA H100 GPUs with 96 GB memory. These are the training devices; the forward-only evaluation infrastructure is reported separately below.

The recorded PL0/PL2/PL3 external checkpoints share width 512, eight heads, four encoder layers, two decoder layers, and step 1,199,440. Their optimizer settings are AdamW, learning rate $5 \times \mathrm { 1 \dot { 0 } ^ { - 5 } }$ weight decay 0.01, batch size 36, and five configured epochs. Table 7 defines the revised run names and their source IDs. PL4, PL5, PL6, and PL7 modify the historical unregularized recipe, not PL2. PL1/Baseline is the evaluated checkpoint with source key a10 dec2 s42, with two decoder layers, batch size 36, and eight unrelated negatives. The historical unregularized checkpoint was not evaluated; it is distinct from the renamed PL1. In PL7, $m _ { 2 } = 0$ removes the second positive margin, not necessarily its hinge term. PL5 is retained only in the historical measurement record. Its evaluated checkpoint has a different recorded training budget, so it is excluded from controlled component comparisons.

Evaluation uses PyTorch 2.11.0+cu128 and NumPy 2.2.6. PL and the external benchmark ran on an A100 40GB; CL ran on an H100 NVL reporting 93.1 GiB. No hardware-speed comparison is made. PL/CL use forward-only extraction with batch size 4; external encoders use batch size 16; retrieval query chunks contain 64 vectors. No backbone or readout parameters are updated. Numeric summaries, paired intervals, figure-generation code, and the compiled manuscript accompany the LaTeX source.

## C ADDITIONAL EMPIRICAL RESULTS

Tables 1, 2, 3, and 4 in the main paper report all PL checkpoints and external comparisons. Tables 8, 9, and 10 below add oracle/predicted PL comparisons and CL retrieval and reconstruction details. The accompanying machine-readable summaries retain all measured severities and readouts.

## C.1 SEMANTIC GENERALIZATION AND SYNTHETIC CONTROLS

Canonical identity retrieval and semantic equivalence exhibit markedly different behavior. Although PL3 achieves 100% accuracy on synthetic role-swap controls, its SemAntoNeg accuracy is only 9.11%. Thus, successful discrimination of controlled content changes does not establish general semantic equivalence recognition. The evidence supports CANOPE’s utility for canonical identity matching while revealing limitations in how the evaluated similarity functions expose semantic relationships. Tables 3 and 4 in the main paper retain the complete reported semantic comparisons, including weak performance. Their protocols appear in Appendix B.6.

Table 8: PL setting at natural severity 3. Rank and CER use oracle routing; Recall@1 (R1) uses relative positions. Percentages except rank.
<table><tr><td>Model</td><td>Pooled rank</td><td>Token rank</td><td>Oracle R1 ↑</td><td>Predicted R1 ↑</td><td>CER↓</td></tr><tr><td>PL0/JEPA-based</td><td>15.76</td><td>15.76</td><td>13.46</td><td>6.31</td><td>87.50</td></tr><tr><td>PL2/VISReg-Weak</td><td>15.82</td><td>33.04</td><td>98.80</td><td>98.80</td><td>23.09</td></tr><tr><td>PL3/VISReg-Stronger</td><td>36.06</td><td>65.11</td><td>99.16</td><td>99.19</td><td>22.51</td></tr></table>

Table 9: CL setting relative-position Recall@1 (%) at severity 3. Input routing uses source length, not a learned length predictor.
<table><tr><td>Model</td><td>Recov. oracle ↑</td><td>Recov. input ↑</td><td>Heldout oracle ↑</td><td>Heldout input ↑</td></tr><tr><td>CL-GP</td><td>99.985</td><td>97.475</td><td>84.278</td><td>84.665</td></tr><tr><td>CL-GT</td><td>99.980</td><td>16.532</td><td>89.472</td><td>31.220</td></tr><tr><td>CL-LP</td><td>0.520</td><td>0.090</td><td>8.183</td><td>6.759</td></tr><tr><td>CL-LT</td><td>0.747</td><td>0.002</td><td>0.715</td><td>0.014</td></tr></table>

Table 10: Native-head reconstruction at E severity 3. CER and exact match (EM) are percentages; CE is corpus nats per target byte under oracle routing. CL-LP/CL-LT heads were not trained and are excluded.
<table><tr><td>Model</td><td>Channel</td><td>Or. CER↓</td><td>Or. EM ↑</td><td>Or. CE↓</td><td>In. CER↓</td><td>In. EM ↑</td></tr><tr><td>CL-GP</td><td>recoverable</td><td>0.13</td><td>98.97</td><td>0.0094</td><td>91.84</td><td>0.09</td></tr><tr><td>CL-GP</td><td>heldout</td><td>18.13</td><td>6.86</td><td>16.3316</td><td>17.62</td><td>6.83</td></tr><tr><td>CL-GT</td><td>recoverable</td><td>0.15</td><td>98.78</td><td>0.0109</td><td>91.70</td><td>0.09</td></tr><tr><td>CL-GT</td><td>heldout</td><td>19.08</td><td>6.86</td><td>12.8150</td><td>17.06</td><td>6.83</td></tr></table>

## C.2 RAW GEOMETRY AND ITS INTERPRETATION

Table 11: Raw CL geometry at recoverable severity 3, oracle routing. v¯ is mean coordinate variance, $D _ { \Sigma } =$ $\| \widehat { \Sigma } - I \| _ { F } / \sqrt { d } .$ , and D<sub>CF</sub> is Eq. (44). Arrows indicate closer agreement with the zero-mean, identity-covariance Gaussian reference, not better content access; v¯ has target 1. Token statistics use one valid token per sentence.
<table><tr><td>Model</td><td>States</td><td>Eff. rank</td><td> $\| \mu \| \downarrow$ </td><td>v</td><td> $D _ { \Sigma } \downarrow$ </td><td> $D _ { \mathbf { C F } } \downarrow$ </td></tr><tr><td>CL-GP</td><td>Pool</td><td>55.08</td><td>6.94</td><td>0.73763</td><td>2.456</td><td>0.03175</td></tr><tr><td>CL-GP</td><td>Token</td><td>41.59</td><td>6.94</td><td>26.37623</td><td>109.941</td><td>0.15207</td></tr><tr><td>CL-GT</td><td>Pool</td><td>114.14</td><td>1.04</td><td>0.02654</td><td>0.977</td><td>0.40322</td></tr><tr><td>CL-GT</td><td>Token</td><td>194.47</td><td>1.05</td><td>0.82981</td><td>1.375</td><td>0.00280</td></tr><tr><td>CL-LP</td><td>Pool</td><td>5.36</td><td>32.53</td><td>0.04498</td><td>1.103</td><td>0.69940</td></tr><tr><td>CL-LP</td><td>Token</td><td>15.57</td><td>32.53</td><td>0.06938</td><td>1.101</td><td>0.65024</td></tr><tr><td>CL-LT</td><td>Pool</td><td>3.68</td><td>543.35</td><td>0.00038</td><td>1.000</td><td>1.18631</td></tr><tr><td>CL-LT</td><td>Token</td><td>28.59</td><td>543.35</td><td>0.00266</td><td>0.997</td><td>1.17967</td></tr></table>

Table 11 separates pooled and sampled-token diagnostics. Isotropic covariance requires $\Sigma = \sigma ^ { 2 } I ;$ the standard Gaussian target additionally requires zero mean, unit variance, and Gaussian shape. Low effective rank indicates unequal eigenvalues. A covariance discrepancy near one can also result from near-zero variance. These distinct diagnostics complement the content tests.

CL-GT has substantially lower token characteristic-function discrepancy than CL-GP (0.00280 versus 0.15207), but CL-GT’s token covariance deviation from identity remains 1.375 and its effective rank is 194.47. Improved projected shape diagnostics therefore do not justify replacing measured geometry with an assumption of exact isotropy. CL-LT has low variance, low effective rank, and a dominant common mean; its near-zero raw agreement MSE must be interpreted with those scales. Mean-centering improves clean identity access without demonstrating preservation of all ordered bytes. The theoretical Gaussian construction and these learned-state diagnostics answer different questions.

![](images/370e7adab1c5c747b2db9b26bbf637c93d9f6e5ad14a35d53904556c41b278b5.jpg)

![](images/ce0d70e3e3052b8d1e1a4d064a740672f6b57656d65c4907edd4008f93e91382.jpg)  
Figure 5: Pooled effective rank and content access. The left panel uses PL natural severity 3 with oracle lengths for the principal configurations. The right panel shows that mean subtraction restores CL-LT clean identity access much more than cross-view retrieval. Both panels summarize full-cohort measurements.

Geometry and mean subtraction. CL-LT has pooled mean norm 543.35, coordinate variance $3 . 7 9 \times 1 \hat { 0 } ^ { - 4 }$ , rank 3.68, and raw cross-view MSE $6 . 6 7 \times 1 0 ^ { - 7 } ;$ : agreement occurs at a highly concentrated scale. Centering restores clean retrieval from $0 . 7 6 \%$ to 100%, but severity-3 retrieval rises only from 0.75% to 1.58% (Figure 8). Clean access does not imply corruption robustness. The first three PCA components retain 9.90%/11.01% of CL-GP/GT clean variance versus 88.44% for CL-LT. Thus similar clean/corrupted clouds can coexist with concentrated geometry and poor access.

Why small disagreement need not mean robust separation. A scale illustration clarifies these diagnostics without assuming a training trajectory. Let $Z ^ { ( v ) } = m + \epsilon [ s ( C ) + e _ { v } ( C ) ]$ , where m is a shared vector, s(C) carries content, $e _ { v } ( C )$ is a view-dependent perturbation, and $\epsilon > 0$ . For finite second moments,

$$
\begin{array} { r } { \mathbb { E } \| Z ^ { ( 1 ) } - Z ^ { ( 2 ) } \| ^ { 2 } = \epsilon ^ { 2 } \mathbb { E } \| e _ { 1 } ( C ) - e _ { 2 } ( C ) \| ^ { 2 } . } \end{array}\tag{50}
$$

As ϵ decreases, absolute disagreement vanishes, but both inter-target separation and view perturbations shrink at the same rate. Their relative scale need not improve. If m ̸= 0 is fixed, uncentered cosine similarities also approach one as $\epsilon  0$ for finite residuals. Removing the common mean can expose clean distinctions while leaving corruption-induced mismatches. This explains how the reported diagnostics can coexist; it does not identify the learned residuals or prove that the optimizer follows this path. In particular, VISReg’s variance term penalizes vanishing variance, so the illustration concerns the agreement term, not a global optimum of the combined objective.

## C.3 PAIRED 3D VIEWS AND DISTRIBUTION GEOMETRY

The visual diagnostics use a fixed 4,096-item subset, distinct from the 40,000-item full-gallery summaries. PCA is fit to clean raw states separately for each model and representation type. The same basis projects its clean and corrupted views. Distribution panels display 1,200 points; trajectory panels display 12 matched identities. Figure 10 uses the PL natural channel; the pooled distribution grid (Figure 11) and Figure 6 use CL recoverable corruption, all with oracle lengths. Panel sizes are identical within each grid. Bases and units differ across models; common axes and limits apply within each three-panel distribution row.

The pooled distribution grid shows all four CL objectives as averaged text vectors. The matched views below add identity correspondence. Appendix F extends both views to sampled tokens, PL natural corruption, and the CL heldout channel, and indexes the complete visual atlas.

How the same texts move under reversible marker corruption One averaged vector per text; true target length supplied  
![](images/405365e376caa769fe2cc8248fdaefb34ba9f0303ead12e1cdc34849b4a677d2.jpg)

![](images/957098e31a66b7b109090537fe12594ee7385af6353664fd98b179cf87a2fcdc.jpg)

![](images/41bece27dd15f840870dc79379681729be2e66cce6d51838a7268b6bc42763be.jpg)

![](images/0a2e7a845024385b3b26d1f915ef5ff9319b67361e97927f4f633510aa631546.jpg)  
Circle = clean Diamond = level 1 Square = level 2 Cross = level 3 Each color tracks one identity; lines connect corruption levels, not time.  
Figure 6: Matched CL identities in $\phantom { - } 1 2 \times 2$ grid. Each panel uses pooled states, oracle lengths, and its own clean PCA basis. Colors identify 12 matched texts; markers distinguish clean and recoverable severities 1–3. Lines connect views rather than time steps. CL-LP exhibits visible displacement for several identities, while some concentrated representations show short trajectories despite poor full-gallery retrieval. The figure illustrates selected within-model behavior, not an estimate of retrieval accuracy.

Pairwise margins and Gaussian projections. The supplied cosine histograms compare the correct identity with a random wrong identity. Their ECDFs use the margin $s ( q , c ^ { + } \overrightharpoonup { + } s ( q , c _ { \mathrm { r a n d o m } } ^ { - } )$ , whereas Recall@1 requires the correct candidate to outrank the full gallery. A positive random-foil margin can therefore coexist with poor Recall@1. The supplied Q–Q plots compare four seeded projections with normal quantiles; their standardized rows remove projection mean and scale. Close agreement on these directions supports the projected shape diagnostic but does not test every direction or the joint canvas. The token mixture and the 3D PCA projections also concern different sampling units and subspaces.

## C.4 CORRUPTION SENSITIVITY AND POSITIONAL ACCESS

![](images/175a6ab39b8ae51bd7dd1ce3c6dff9c057abe728e5cdea7365bc412437859482.jpg)  
Figure 7: Relative-position retrieval across corruption strengths. The panels use PL predicted lengths, CL input lengths, and LexNorm changed-input queries with extra whitespace. Level 0 denotes clean PL/CL inputs and the original noisy LexNorm query. Channels and cohorts differ, so severity scales are not directly comparable.

These within-checkpoint comparisons isolate the choice of similarity readout. The benefit of relative positions is largest under the strongest whitespace perturbation. It supports the utility of canvas structure for canonical identity access without a claim that the 16-position score recovers every byte or all semantic relations.

## C.5 SUPPORTING READOUT CONTROLS

The main-text comparisons retain the original full-cohort centering and positional-access plots below.

![](images/a8f2c6f488ff39c5ed890d1c8b67a0f45941b1f10ea8070113f8c559f0e98716.jpg)  
Figure 8: Centering restores clean access, not robustness. CL-LT, oracle lengths, relative-position scoring, 40,000 targets. Blue subtracts the clean mean of 1,024 disjoint calibration examples; red uses raw states. Labels give Recall@1 (%).  
(a) Original queries

![](images/53f7a1ceede0ce828ee9c6fd707ab38e6c9bc24c2951e6279c0a53e2fb822793.jpg)  
(b) Whitespace level 3

![](images/e4b355fe540487920ea3cf7615f2c0320f118314d7ff68306de81fc8120e8c0c.jpg)  
Figure 9: Position improves access within a fixed checkpoint and predicted length. Gray pools states; blue compares 16 relative positions. All 1,941 changed-input queries search the same 1,954 targets. At maximum whitespace perturbation, PL2/PL3 gain 36.89/21.54 points.

## D DOWNSTREAM TEXT-TO-SPEECH: PROTOCOL AND AUDIT

## D.1 DATA, TRAINING, AND THE FROZEN INTERFACE

The downstream experiment uses LJSpeech 1.1 (Ito & Johnson, 2017), with normalized transcriptions paired to the original single-speaker recordings. The preparation code shuffles the 13,100 utterance IDs with seed 42 and makes a 70/30 split (9,170 training and 3,930 validation utterances). Each ID has a stored clean transcription and a fixed augmented version; both target the same recording. The conservative augmenter attempts up to eight medium-strength rule-based corruptions and accepts a nonempty changed string only when one minus the character-level SequenceMatcher ratio is at most 0.10. It otherwise retains the clean text. This bound is a string-similarity heuristic, not a certified information-preserving channel or a Levenshtein-error bound.

The training loader chooses the stored augmented branch with probability 0.7 using a hash of training seed, epoch, and utterance ID; it also fixes the per-epoch example order across systems. Validation always selects the stored augmented branch, while the reported downstream evaluation synthesizes both branches. No target audio is corrupted. We trained all three systems for 1,000 epochs using the default submission-script configuration: Adam with learning rate $1 \dot { 0 } ^ { - 4 }$ , zero weight decay, batch size 36, training seed $^ { 4 2 , }$ mixed bfloat16 precision, and gradient clipping at 5.0. The launcher selects one A100 80 GB for the baseline and one H100 96 GB per frozen system. The accompanying source and Slurm configuration specify these training settings.

All three systems initialize from the official LJSpeech MatchaTTS checkpoint (Mehta et al., 2024). The native baseline fine-tunes its text encoder and acoustic model. Its frontend applies english cleaners2 (including lowercasing and eSpeak phonemization) and intersperses blank tokens. PL0 and PL2 instead receive UTF-8 bytes without inserted blanks. The integration restores their canonical-canvas backbones from the latent-agreement and token-grounded checkpoints, respectively, disables all backbone gradients, and keeps them in evaluation mode. Canvas length is the ceiling of the model’s predicted canonical length, bounded below by one; over-capacity inputs or predictions raise an error rather than being silently truncated. Neither system uses oracle target lengths, a canonical-token prefix, nor the CANOPE token head at inference.

For both frozen systems, a trainable LayerNorm and linear map project each width-512 canvas row to 192 channels. The integration reuses MatchaTTS’s pretrained acoustic-mean projection, duration predictor, and flow-matching decoder. The adapter and acoustic modules train while CANOPE stays frozen; the duration predictor receives detached adapter features, as implemented in the interface. Thus this is latent conditioning of a trained speech generator, not text correction followed by an unchanged TTS model. The common downstream architecture makes PL0/PL2 an informative accessibility comparison, but their canvas lengths, state distributions, and learned alignments can differ.

Synthesis uses ten flow steps, temperature 1, length scale 1, and synthesis seed 9027, followed by a shared pretrained HiFi-GAN vocoder (Kong et al., 2020) at 22,050 Hz. Its random seed is derived from the utterance ID and input view; systems share that seed for a given ID/view, but clean and augmented views use different derived seeds. Consequently, the clean-to-augmented contrast includes the realized synthesis variation. We evaluate the generated speech with Whisper large-v3. The archived CSVs retain both raw and normalized references and hypotheses, but not the ASR commandline flags. The accompanying evaluator implements English text normalization and unprompted transcription (temperature zero, no previous-text conditioning); the exact ASR invocation is not independently documented. The audit below uses the archived normalized strings directly.

## D.2 RECOMPUTED WORD ERROR RATES AND PAIRED UNCERTAINTY

Let $e _ { i m v } = S _ { i m v } + D _ { i m v } + I _ { i m v }$ denote the word edit count for utterance i, model m, and view v, and let $n _ { i }$ be its normalized clean-reference word count. We report

$$
\mathrm { W E R } _ { m v } = 1 0 0 \frac { \sum _ { i } \epsilon _ { i m v } } { \sum _ { i } n _ { i } } , \qquad \Delta _ { m } = \mathrm { W E R } _ { m , \mathrm { a u g } } - \mathrm { W E R } _ { m , \mathrm { c l e a n } } , \qquad D = \Delta _ { \mathrm { P L } 2 } - \Delta _ { \mathrm { b a s e } } .\tag{51}
$$

These are corpus ratios, not means of utterance WERs. All six cells contain the same 3,930 unique IDs, one recorded synthesis seed, and 66,489 reference words. We verified unique model/view/ID keys, identical references across the six cells, reference-token counts, every stored utterance ratio, and each total edit count against a fresh word-level Levenshtein calculation on the archived normalized strings. The recomputed corpus totals match both archived summary CSVs. There are 3,923 unique normalized reference strings; the sampling unit remains the audio utterance ID. The CSVs contain scores for every expected cell, but they do not carry synthesis/ASR failure flags for the full cohort. Complete score coverage alone therefore does not establish that every synthesis or ASR call succeeded; empty hypotheses remain in the metric.

Table 12: Audited downstream error counts and utterance-paired percentile intervals. S/D/I denote substitutions, deletions, and insertions. The word denominator is 66,489 in every row.
<table><tr><td>System</td><td>Input</td><td>S↓</td><td>D↓</td><td>1↓</td><td>WER↓ [95% interval]</td><td>Zero-WER clips ↑</td></tr><tr><td>MatchaTTS</td><td>Clean</td><td>868</td><td>259</td><td>345</td><td>2.21 [2.04, 2.40]</td><td>3,083</td></tr><tr><td>MatchaTTS</td><td>Augmented</td><td>4,910</td><td>701</td><td>1,654</td><td>10.93 [10.47, 11.40]</td><td>1,647</td></tr><tr><td>Frozen PL2</td><td>Clean</td><td>6,657</td><td>1,398</td><td>988</td><td>13.60 [13.21, 14.01]</td><td>816</td></tr><tr><td>Frozen PL2</td><td>Augmented</td><td>10,880</td><td>2,085</td><td>1,359</td><td>21.54 [21.01, 22.09]</td><td>458</td></tr><tr><td>Frozen PL0</td><td>Clean</td><td>39,958</td><td>24,689</td><td>1,259</td><td>99.12 [98.33, 100.10]</td><td>0</td></tr><tr><td>Frozen PL0</td><td>Augmented</td><td>39,465</td><td>25,231</td><td>1,272</td><td>99.22 [98.45, 100.15]</td><td>0</td></tr></table>

For uncertainty, we draw 10,000 bootstrap samples of the 3,930 utterance IDs with replacement using NumPy seed 9027, retaining all three systems and both views for each selected ID. We recompute the numerator and word denominator on every draw and take the 2.5th and 97.5th percentiles. This preserves pairing and estimates sampling variation conditional on the fitted checkpoints and realized synthesis/ASR outputs. It does not measure variation across training seeds, other ASR engines, or independent domains. WER can exceed 100% because insertions are included.

The corruption increments are 8.71 points [8.26, 9.18] for the baseline, 7.94 [7.43, 8.44] for PL2, and 0.09 [−1.03, 1.21] for PL0. The paired contrast D = −0.77 points has interval [−1.34, −0.22]. PL2 nevertheless has 10.62 points more augmented WER than the baseline [10.07, 11.18]. Its advantage over PL0 is 77.67 points [76.72, 78.72]. These contrasts describe the evaluated runs; they are not multiple-comparison-adjusted hypothesis tests.

PL0’s clean errors are dominated by 39,958 substitutions and 24,689 deletions, with no zero-WER utterances. Its near-zero corruption increment accompanies failure on the clean view and therefore does not demonstrate useful invariance. The baseline has 3,083 zero-WER clean clips and 1,647 augmented clips; PL2 has 816 and 458. PL0 has 44 empty normalized hypotheses on clean audio and 43 on augmented audio, versus none for the other systems. This pattern is consistent with poor linguistic accessibility under the trained interface, but it does not identify the population information gap or distinguish representation loss from optimization, alignment, length, and readout mismatch.

## D.3 SCOPE OF THE DOWNSTREAM EVIDENCE

The native baseline is an end-to-end adaptation reference with a pretrained phoneme encoder; the CANOPE systems replace that encoder with frozen byte-based states and a newly trained adapter. A casing change may disappear before entering the baseline encoder yet remain visible to CANOPE. The comparison therefore does not isolate the causal effect of freezing or of the CANOPE objective. Likewise, the small reduction in PL2’s corruption increment is conditional on its higher clean error and this particular perturbation mixture; it is not a general robustness guarantee.

The 3,930 items are held out from the reported acoustic fine-tuning split, not demonstrated to be unseen during pretraining. The shared MatchaTTS initializer was trained on LJSpeech, and LJSpeech transcriptions are among CANOPE’s pretraining sources. A pretraining-to-evaluation overlap audit and original-recording Whisper WER are unavailable. We therefore avoid interpreting these numbers as out-of-domain transfer, a ground-truth intelligibility ceiling, or a clean test-set estimate. Whisper WER (Radford et al., 2023; OpenAI, 2023) measures ASR-mediated lexical fidelity and can include recognizer errors; no human listening score or perceptual naturalness claim is made.

## D.4 REPRODUCIBILITY AND WAVEFORM DISPLAYS

The accompanying analyze\_tts.py recreates the audit, tables, bootstrap intervals, and spectrogram panels from the archived CSVs and samples. Source names baseline, canope, and jepa map to MatchaTTS, frozen CANOPE-PL2, and frozen CANOPE-PL0, respectively; the historical jepa name is not a claim that PL0 reproduces an unmodified JEPA method. The package retains the original source mapping and a machine-readable audit of counts, audio hashes, and display settings.

The final ten sample pages show every utterance in the archived sample subset, retaining its recorded order. We do not select a further subset by WER. The original selection procedure is undocumented, so these examples are illustrative rather than a random sample of the evaluation cohort. Each page includes all six waveforms as mel-spectrograms, the exact clean and augmented inputs, and the normalized Whisper hypotheses joined by utterance ID, model, and view. The plots are recomputed from the generated WAV files, not from saved decoder mel targets. All 60 WAV hashes, durations, sampling rates, and mel-frame counts match their sidecar metadata. No original reference recordings were supplied in the sample archive, so no ground-truth-audio panel is shown.

We use a 1,024-sample periodic Hann window, 256-sample hop, 384-sample reflection padding at both ends, and 80 Slaney area-normalized mel filters over 0–8 kHz, matching the supplied Matcha acoustic transform. Magnitudes are displayed as $2 0 \log _ { 1 0 } ( M / M _ { \mathrm { m a x } } )$ , clipped to [−80, 0] dB, where $M _ { \mathrm { m a x } }$ is shared across all 60 clips. Time axes are shared within each six-panel example without time warping; gray beyond a waveform’s end denotes absence of samples. These plots expose timing and spectral differences but do not independently establish which words were spoken or how natural the audio sounds.

## E PROBABILITY AND INFORMATION IDENTITIES USED IN THIS PAPER

This appendix explains the identities needed for the content-gap and readout arguments. It also separates population quantities from the finite-cohort diagnostics. The information identities follow standard definitions and chain rules (Cover & Thomas, 1991, Chapter 2); the applications below use this paper’s variables and examples.

## E.1 RANDOM VARIABLES, EXPECTATIONS, AND CONDITIONAL DISTRIBUTIONS

Uppercase letters denote random variables, while lowercase letters denote realizations. For example, C is a random target and c is one particular string. An expectation averages a function under the joint law of its arguments. For discrete $C$ and possibly continuous $Z .$

$$
\mathbb { E } [ f ( C , Z , S ) ] = \mathbb { E } _ { Z , S } \left[ \sum _ { c } p ( c \mid Z , S ) f ( c , Z , S ) \right] .\tag{52}
$$

Thus E without a subscript is shorthand when the variables are clear. In the KL term of Eq. (2), the inner divergence already sums over c at fixed $( z , s )$ . The outer $\mathbb { E } _ { Z , S }$ averages that divergence across contexts. Omitting this outer average would leave a context-dependent quantity rather than population log-loss.

The notation $C \perp Z \mid ( X , S )$ means that after $( X , S )$ is known, Z gives no additional information about $C .$ It permits C and $\dot { Z }$ to be strongly dependent before conditioning. A deterministic $Z =$ $F ( X , S )$ satisfies it because $Z$ is then already fixed by $( X , S )$ . Independent encoder randomness also satisfies it if it introduces no target-dependent side channel.

## E.2 ENTROPY, LOG-LOSS, AND THE INFORMATION GAP

For a discrete target, conditional entropy is

$$
H ( C \mid Z , S ) = \mathbb { E } _ { Z , S } \left[ - \sum _ { c } p ( c \mid Z , S ) \log p ( c \mid Z , S ) \right] .\tag{53}
$$

For two discrete laws, $\begin{array} { r } { D _ { \mathrm { K L } } ( p \| q ) = \sum _ { c } p ( c ) \log [ p ( c ) / q ( c ) ] } \end{array}$ , with value +∞ if $q$ assigns zero probability to a positive-probability event under $p .$ It measures distributional mismatch and is zero exactly when the laws agree. Conditional entropy is uncertainty under the true conditional law. A readout $q$ instead incurs log-loss $\mathbb { E } [ - \log q ( C \mid Z , S ) ]$ . Subtracting the true entropy leaves an expected KL divergence, which is nonnegative. Hence an inaccurate readout can have high loss even when the target is fully determined by $\bar { Z . }$

The standard identity

$$
I ( C ; X \mid Z , S ) = H ( C \mid Z , S ) - H ( C \mid X , Z , S )\tag{54}
$$

measures how much X reduces the remaining uncertainty after $( Z , S )$ is given. Under the conditionalindependence assumption, its final entropy becomes $H ( C \mid X , { \dot { S } } )$ , yielding Eq. (1). This identifies a processing loss relative to the input, not necessarily the entire uncertainty about $C .$

For a concrete example, let $C = ( B _ { 1 } , B _ { 2 } )$ contain independent fair bits, $X = C .$ , and $Z = B _ { 1 }$ In nats, $H ( C ) = 2 \bar { \log 2 } , H ( C \mid X ) = 0$ , and $H ( C \mid Z ) = \log 2$ . The encoder loses one bit, so $\Delta _ { \mathrm { i n f o } } = \log 2$ . If instead $Z = \dot { ( B _ { 2 } , B _ { 1 } ) }$ , the full representation loses no information. A position-local head still fails, because it sees the wrong independent bit at each position. This is the difference between information loss and local readout mismatch.

If the input itself contains only $X = B _ { 1 }$ and the encoder preserves $Z = X$ , then $H ( C \mid X ) = \log 2$ but $\Delta _ { \mathrm { i n f o } } = 0$ . The irreducible uncertainty arose before the encoder. Recoverable corruption removes this confound by ensuring $H ( C \mid X , L ) { \dot { = } } 0$

## E.3 WHY LENGTH IS A SEPARATE INFORMATION VARIABLE

At a given length $\ell ,$ the chain rule expands the entropy of the sequence into ℓ conditional token entropies. Each term is at most log K, which proves Eq. (3). Equality requires the uniform distribution

over all $K ^ { \ell }$ strings, not ordinary linguistic text. Repetition can add almost no new uncertainty, whereas independent symbols add a constant amount per position.

For an infinite stationary finite-alphabet process with entropy rate $h ,$ the block entropy satisfies

$$
h = \operatorname* { l i m } _ { n \to \infty } { \frac { H ( C _ { 1 } , \ldots , C _ { n } ) } { n } } , \qquad H ( C _ { 1 } , \ldots , C _ { n } ) = n h + o ( n ) .\tag{55}
$$

This is an asymptotic statement about a source process (Shannon, 1948). It does not imply that arbitrary variable-length sentences, or their meanings, contain exactly h units of information per token. Conditioning on the event $L = \ell$ can change the token law.

Because sequence length is a function of the sequence, $H ( L \mid C ) = 0 .$ . The joint-entropy chain rule therefore yields $H ( C ) = H ( L ) + H ( C \mid L )$ when finite. CL supplies $L$ and studies the remaining ordered content. PL predicted inference must infer a length from X. Its output shape is itself part of the representation available to the readout. Stopping gradients through the length-head input does not supply the unknown length or change this inference-time information structure.

## E.4 COVARIANCE, EIGENVALUES, AND THE QUANTITY CALLED RANK

For a random vector R with finite second moments,

$$
\Sigma = { \mathrm { C o v } } ( R ) = \mathbb { E } [ ( R - \mathbb { E } R ) ( R - \mathbb { E } R ) ^ { \top } ] .\tag{56}
$$

An eigenvector $u _ { i }$ with unit norm identifies a direction whose projected variance is $u _ { i } ^ { \top } \Sigma u _ { i } = \lambda _ { i }$ Equal eigenvalues mean equal variance in all directions. The eigenbasis describes directions, while the spectrum lists their variances.

Algebraic rank counts nonzero eigenvalues. Covariance effective rank instead applies entropy to their relative weights $p _ { i } = \lambda _ { i } / \sum _ { j } \lambda _ { j }$ and exponentiates the result. It is sensitive to the unevenness of the spectrum even if every eigenvalue remains positive. For example, both diag(1.8, 0.2) and diag(1.2, 0.8) have algebraic rank two, but their effective ranks are 1.3841 and 1.9601. This is the function differentiated in Appendix A.8.

Neither covariance isotropy nor effective rank determines the full distribution. An isotropic covariance can occur in a non-Gaussian law, and a standard Gaussian sample has unequal empirical eigenvalues at finite sample size. Our Gaussian reference uses matched sample count and dimension to contextualize this sampling effect.

## E.5 POPULATION GUARANTEES AND MEASURED DIAGNOSTICS

Theorems use population expectations. An empirical token-loss average is an estimate, and a per-byte corpus average is not the expected summed sequence loss used in the information bound. Paired bootstrap intervals resample canonical identities while keeping each comparison paired; they describe evaluation-cohort uncertainty conditional on the checkpoint.

Recall@1 asks whether the true target outranks every candidate in the gallery. A positive cosine margin against one random wrong target is a weaker event. Similarly, a 3D principal-component plot shows only three directions. Short trajectories indicate stability in that displayed subspace; interpreting robustness also requires the retained variance, correspondence structure, and full-gallery retrieval curves.

## F COMPLETE DISTRIBUTION AND PAIRED-VIEW COMPARISONS

The distribution and paired-view grids in this section complete the coverage across all seven models with supplied feature diagnostics. PL includes natural corruption; CL includes both recoverable and heldout corruption. Every combination is shown for pooled and sampled-token states. The primary pooled CL distribution is in Figure 11, and the primary pooled paired views are in Figures 10 and 6. All feature-level plots supply the true target length (oracle length). Pooled means one vector obtained by averaging the valid positions of a text; token means one sampled valid position per text. Recoverable denotes reversible inserted markers; heldout denotes the alternative marker pattern defined in Section 6. The full-cohort tables and curves separately report PL predicted-length and CL input-length performance.

How the same texts move under spelling and surface corruption One averaged vector per text; true target length supplied

![](images/0002d1eb7bac0de279fb1539a1c8ca27c3bde6e322aadd3d062b10234eccc3a8.jpg)  
Circle = clean Diamond = level 1 Square = level 2 Cross = level 3 Each color tracks one identity; lines connect corruption levels, not time.

Figure 10: Following the same identities through natural corruption. Each panel shows 12 texts, one color per text. Follow a line from the clean circle through the diamond, square, and cross at corruption levels 1, 2, and 3. The displayed states are averaged text vectors with true target lengths supplied. Axes are fitted on clean states separately for each model. Some PL0 views stay close even though full-gallery retrieval is poor in Figure 3; visual proximity must therefore be read together with identity access. Percentages state how much clean variance the three displayed axes retain.

CL-GP First three PCA axes retain 9.90% of clean variance Clean

Clean  
Reversible marker corruption: clean and corrupted distributions One averaged vector per text; true target length supplied  
![](images/3d12d8bed801d2ea333ee6ec7ebe1bf1ce5c199ce69648fa9e3419207b434a33.jpg)  
Isotropic Gaussian

CL-GT First three PCA axes retain 11.01% of clean variance  
![](images/92859f48c149f665c9abf1603d0df016ac58b4e519e8e06dabcfda5b46bcfd69.jpg)  
Clean

![](images/bc962fe6fbdc9dc8cde4ff87d7bd6bca457cec0b267500a89b32d58124a5cd4d.jpg)

![](images/d41afa8e1fb4eee1c4ac9fe38bb4bf3dd1082dcc12a2a46d86803e13eedf807c.jpg)  
CL-LP First three PCA axes retain 83.32% of clean variance

![](images/4328015251f3c7af725a13fb5d03a7f909100d4278974fc2590d45b82d59b33e.jpg)  
Corrupted (severity 3)

![](images/911c4fe141e94c19672dd494d4a76ebc95a0bda8905217e924dafc2094dfe58e.jpg)  
Isotropic Gaussiar

![](images/e8b3e7436b05a154158704708b50ee99379d49fb11892ae569ba8f38e7446da3.jpg)  
CL-LT First three PCA axes retain 88.44% of clean variance Clean

![](images/d92e3d67c238f80ee7ead3058dc58061cad597f5b2068cb9929bfb83acfbc820.jpg)  
Corrupted (severity 3)

![](images/e0e8fb928a9e58234ae4f6988464a0ccc575581252fc2e26967b49a485ab013f.jpg)

![](images/6c1196548d99faae51d42daa5790998806c2571bc540184328b380030a8af39f.jpg)

![](images/41f5f305fc16984034701533977bb72fe0271451c6b6fdeae4e8b3b76bce65bd.jpg)

Isotropic Gaussian  
![](images/7f84a89ae2af3df95a2c88be1ec5fa4c6d3509d39f1eab5e136800aff77b360f.jpg)

Figure 11: What the geometry diagnostic sees in the four CL configurations. Rows identify the models; columns show clean text, reversible corruption at level 3, and an isotropic Gaussian reference. Each point is one averaged text vector; true target lengths are supplied. The same clean PCA axes and scale apply across a row. CL-GP/GT spread most variance beyond the three displayed axes, whereas CL-LP/LT concentrate it in this subspace. Compare these clouds with retrieval in Figure 2, rather than reading a compact cloud as evidence of content preservation. Axes are fitted to 4,096 clean examples; each cloud displays 1,200 points. Surfaces span two standard deviations, not confidence regions.

How to read the grids. A distribution row compares clean states, severity-3 states, and the supplied standard isotropic Gaussian reference. The clean PCA basis, coordinate units, and limits are fixed within the row. Each paired panel instead follows 12 individual canonical identities through clean and severity levels 1–3. A color identifies a target; a circle marks clean, a diamond level 1, a square level 2, and a cross level 3. The connecting lines show a change across corruption levels, not temporal dynamics. Inspect whether identities remain separated as well as how far their views move. Full-gallery retrieval measures the corresponding identity access beyond these selected examples and three coordinates.

The first three principal components retain a model-dependent fraction of variance. For example, the clean pooled CL-LP and CL-LT projections retain 83.32% and 88.44%, while the corresponding token projections retain 59.56% and 37.50%. The grids therefore expose different subspaces and scales, not a shared coordinate system for measuring cross-model distances. Table 13 gives the exact subset metrics needed to interpret them.

Table 13: Clean 4,096-item visualization subset, oracle length. The dimensions $d _ { 9 0 }$ and $d _ { 9 5 }$ are the smallest numbers of principal components explaining 90% and 95% of variance. These subset ranks need not equal the full-cohort ranks.
<table><tr><td>Model</td><td>Representation</td><td>Effective rank</td><td> $\mathbf { d _ { 9 0 } }$ </td><td> $\mathbf { d _ { 9 5 } }$ </td><td>PC1-3 (%)</td></tr><tr><td>PL0</td><td>pooled</td><td>15.62</td><td>15</td><td>18</td><td>44.43</td></tr><tr><td>PL0</td><td>token</td><td>15.63</td><td>15</td><td>18</td><td>44.43</td></tr><tr><td>PL2</td><td>pooled</td><td>15.54</td><td>12</td><td>14</td><td>34.94</td></tr><tr><td>PL2</td><td>token</td><td>34.22</td><td>31</td><td>48</td><td>25.55</td></tr><tr><td>PL3</td><td>pooled</td><td>35.49</td><td>29</td><td>37</td><td>21.02</td></tr><tr><td>PL3</td><td>token</td><td>65.41</td><td>58</td><td>85</td><td>15.44</td></tr><tr><td>CL-GP</td><td>pooled</td><td>54.09</td><td>41</td><td>52</td><td>9.90</td></tr><tr><td>CL-GP</td><td>token</td><td>42.53</td><td>34</td><td>46</td><td>17.53</td></tr><tr><td>CL-GT</td><td>pooled</td><td>108.23</td><td>97</td><td>131</td><td>11.01</td></tr><tr><td>CL-GT</td><td>token</td><td>164.74</td><td>135</td><td>177</td><td>5.46</td></tr><tr><td>CL-LP</td><td>pooled</td><td>4.65</td><td>5</td><td>6</td><td>83.32</td></tr><tr><td>CL-LP</td><td>token</td><td>14.25</td><td>20</td><td>35</td><td>59.56</td></tr><tr><td>CL-LT</td><td>pooled</td><td>3.59</td><td>4</td><td>6</td><td>88.44</td></tr><tr><td>CL-LT</td><td>token</td><td>28.28</td><td>32</td><td>44</td><td>37.50</td></tr></table>

Table 14: Coverage of the complete accompanying visual atlas. Each registered source PNG appears on its own landscape page. The panel index supplies page numbers, original paths, notes, and SHA-256 hashes.
<table><tr><td>Diagnostic</td><td>PL</td><td>CL</td><td>Total</td></tr><tr><td>Full-cohort summary charts</td><td>5</td><td>10</td><td>15</td></tr><tr><td>Full-cohort covariance spectra</td><td>2</td><td>3</td><td>5</td></tr><tr><td>Clean-subset spectra</td><td>6</td><td>8</td><td>14</td></tr><tr><td>3D distributions</td><td>6</td><td>16</td><td>22</td></tr><tr><td>Paired 3D trajectories</td><td>6</td><td>16</td><td>22</td></tr><tr><td>Projected Gaussian Q-Q plots</td><td>6</td><td>16</td><td>22</td></tr><tr><td>Positive/random-foil cosines</td><td>3</td><td>8</td><td>11</td></tr><tr><td>Random-foil margin ECDFs</td><td>3</td><td>8</td><td>11</td></tr><tr><td>Total</td><td>37</td><td>85</td><td>122</td></tr></table>

Data and reproducibility. The source package includes all 122 registered PNGs, their render and selection manifests, the clean-subset JSON metrics, the full-cohort CSV summaries, and the paired uncertainty summaries. The CSV hashes match the source identities in both supplied geometry manifests. Display names use PL/CL; original checkpoint keys remain in machine-readable files for traceability. The archives supply rendered feature plots and feature hashes, but no latent arrays or per-point PCA coordinates. The included scripts regenerate compositions and CSV-based figures;

recomputing feature projections requires the original arrays. The atlas includes all supplied spectral, Gaussian-projection, pairwise-score, and centering panels listed in Table 14.

Spelling and surface corruption: clean and corrupted distributions One averaged vector per text: true target length suppliec PL0 First three PCA axes retain 44.43% of clean variance Clean Corrupted (severity 3)

Isotropic Gaussian  
Corrupted (severity 3)  
Corrupted (severity 3)  
![](images/be917e83d763427300a5a4fecd35aa9e16ca1547cb2a78b94b697eae73e204ee.jpg)  
PL2 First three PCA axes retain 34.94% of clean variance Clear

![](images/0c682c1b27327e5c509431048f096ef92e22cb446eff44737a440157e4f7541b.jpg)

![](images/766258fcdd37197ba789ec1b100e2322bcbabd7cb1ca79ac7eff41dae570d59b.jpg)

![](images/297348ab7e940b96d329373b12c9a32e703082bb9c29669581c7c1d2487be581.jpg)  
PL3 First three PCA axes retain 21.02% of clean variance Clean

![](images/52c9c5a0177f4d9230370fa8c71d0952f96f812893c7b3de0383788afbc5d37e.jpg)

![](images/b3392b8021adff15d51e1b39ab437b3c56bba170bd1c4f9684a968652b5b73d7.jpg)  
Isotropic Gaussian

![](images/86d685f2a3a14e2c71345ea9844a17bed1f2c6c5e285a9e93e80e3b83d3ffa66.jpg)

![](images/0a72e8cc52c85e2ef61adf5c842d34dc7518cf73613b0f41cd02be0b03f35508.jpg)

![](images/39082f0e94c64a456a5aea2a1627339f56889c0ee68b3ebd1b8dc7d20b58b8b8.jpg)

Figure 12: PL natural distributions of pooled states with oracle length. Each row shows clean, severity 3, and the isotropic Gaussian reference in the same clean PCA coordinates. All supplied models are included. Clouds contain 1,200 points; surfaces span two standard deviations and are not confidence regions.

Spelling and surface corruption: clean and corrupted distributions One sampled position per text: true target length supplied PL0 First three PCA axes retain 44.43% of clean variance Clean Corrupted (severity 3)

Isotropic Gaussian  
Isotropic Gaussian  
![](images/6cd26a9327c03be6f18476248f91937b2c647577c6783da3dd31a84e27f702b2.jpg)  
PL2 First three PCA axes retain 25.55% of clean variance Clean

![](images/5ca70937be5a5e0d4e0feb13ecd3ef161b2c267950c2013bc8a870f5b793a97a.jpg)

![](images/2fdd721dbcb06b4a09439717b68ed5a68e6438090500046d1ab719f00e46fe1b.jpg)

![](images/f99a437fe35a4f3eb67d41e7e8c1c5f61b3d7e280f0d2336115307e7319946a6.jpg)  
PL3 First three PCA axes retain 15.44% of clean variance Clean

![](images/a2def7a811811346d844328b23733ee6298cd87d43c87fa2621eb1a4fc6b4f09.jpg)

![](images/205842f9d119146a7f6b550b4ede00f8a0c65d0438e9cc6ae5cc44da4db2a15c.jpg)

![](images/3aad857e0b58107382437db7366071bf489adca1fbec2dc7f3a5ab7b20a7c1b5.jpg)

![](images/0e0bd0b2dd843bf307639183601f8e6cbfcd8038bd6d067c335b2ff73ee3c73f.jpg)

![](images/bf5f5bc0acf310f5f606ffc3f005d70237ca2d117abbc29f56b7925f78d3cca5.jpg)

Figure 13: PL natural distributions of token states with oracle length. Each row shows clean, severity 3, and the isotropic Gaussian reference in the same clean PCA coordinates. All supplied models are included. Clouds contain 1,200 points; surfaces span two standard deviations and are not confidence regions.

How the same texts move under spelling and surface corruption One sampled position per text; true target length suppliec  
![](images/05622290831c87002192204e67174d29e92ae80f4b0130078f325ae47c9a060f.jpg)  
Circle = clean Diamond = level 1 Sguare = level 2 Cross = level 3 Each color tracks one identity; lines connect corruption levels, not time.

Figure 14: Paired PL natural trajectories for token states with oracle length. Each panel follows 12 canonical identities through clean and severity levels 1–3 using one clean PCA basis per model. Colors identify targets and markers identify corruption levels. Differences in axes and captured variance require within-model interpretation.

![](images/9bf5352e870b61c84beee72a7acc0a762a43ecbdce14027072343573053499c1.jpg)  
Figure 15: Full-cohort PL natural diagnostics across severity, using oracle length. Retrieval, pooled and token effective rank, mean magnitude, variance scale, and cross-view error are shown separately. The bottom row uses a symmetric logarithmic scale with a linear region near zero. These curves use the original 40,000-item numeric summaries, independently of the smaller visualization subset.

Reversible marker corruption: clean and corrupted distributions One sampled position per text: true target length supplied CL-GP First three PCA axes retain 17.53% of clean variance Clean Corrupted (severity 3)

CL-GT First three PCA axes retain 5.46% of clean variance Clean  
![](images/ae63847fd50d63223a0bef4c4e9110e3ce8a3aff2baa8fc421c1aa7e8d631323.jpg)

![](images/8a4be3d122819090a76f6caa2c66ab0309ce4d791e722003a5ba8aeecdb1e40c.jpg)

![](images/0edea3c2fdc2fe77c078f8771d4ebb06129cf94fc6efbddbb59af9b5898179e4.jpg)

![](images/f43b52ac3bfda7e7bef96187d700d6cff4c95f9876114802709b36a0405b0d49.jpg)  
CL-LP First three PCA axes retain 59.56% of clean variance Clean

![](images/17246637f3754a88851bd36ca82b6776c35bd082cc241a00b85fa2515a8eface.jpg)

![](images/cee028f4f7bf97ad7fbc448b19bac38974a072a83f0eec95c8497e1dc8baea74.jpg)

![](images/0c0afb793c80ea31bb4196adb512f92d95a664e968d444be798584b5a27d9d55.jpg)

![](images/6f5d6a408a2e90cfb890596d032407e85c0f3d30ac66f6a6dc609800274de257.jpg)  
CL-LT First three PCA axes retain 37.50% of clean variance Clean

![](images/bbffd9909fadf2edd7a59c8c40414ef697694405a725144a361c28f3fbb48f8b.jpg)

![](images/d9c5576b44d23a688c4d7d15a3c6304c9d69cb0f7ea9af9617cecddadaccad57.jpg)

![](images/6318550612eab0cb955de6032ed797d16bc9f51a1f6d6f0824ffd13556d90904.jpg)

![](images/b47acdcf7c1f53d16f4aa8711be1c2ac89bbf43f550c8246c4b487256f05c9f5.jpg)  
Figure 16: CL recoverable distributions of token states with oracle length. Each row shows clean, severity 3, and the isotropic Gaussian reference in the same clean PCA coordinates. All supplied models are included. Clouds contain 1,200 points; surfaces span two standard deviations and are not confidence regions.

How the same texts move under reversible marker corruption One sampled position per text; true target length supplied  
CL-GP PC1-3 17.53%  
![](images/8d36bd1c4969f4f0c334f54d6b10b5f79e98a53c795833b074427893a87de7f2.jpg)

CL-GT PC1-3 5.46%  
![](images/0862117e795585b396bf20813924ddf730c4ac7ad1f0a0e92f2e94b2b40b82f1.jpg)  
CL-LT PC1-3 37.50%

CL-LP PC1-3 59.56%  
![](images/b3e6db441b1ae6ee18f2a187fb01c64b01e37d45aaeca7672a6ecef03ba58d6e.jpg)

![](images/88941595b62716485bd94b6325ceae7a156e5ec9eccb6d33fffd78142250c102.jpg)  
Circle = clean Diamond = level 1 Square = level 2 Cross = level 3 Each color tracks one identity; lines connect corruption levels, not time.  
Figure 17: Paired CL recoverable trajectories for token states with oracle length. Each panel follows 12 canonical identities through clean and severity levels 1–3 using one clean PCA basis per model. Colors identify targets and markers identify corruption levels. Differences in axes and captured variance require within-model interpretation.

![](images/b5ea387793a4ac6102f44f2700b5cedb87eca5e12c93880970e07adfc683a3e1.jpg)  
Figure 18: Full-cohort CL recoverable diagnostics across severity, using oracle length. Retrieval, pooled and token effective rank, mean magnitude, variance scale, and cross-view error are shown separately. The bottom row uses a symmetric logarithmic scale with a linear region near zero. These curves use the original 40,000-item numeric summaries, independently of the smaller visualization subset.

Alternative marker pattern: clean and corrupted distributions One averaged vector per text: true target length supplied CL-GP First three PCA axes retain 9.90% of clean variance Clean Corrupted (severity 3)

Corrupted (severity 3)  
Corrupted (severity 3)  
Corrupted (severity 3)  
![](images/4e08f459e5a51f05af446f1cf48c95949fc7eca34544d6ece1f20bd4fa6e32fd.jpg)

![](images/b7bcac58f75778714e674728f3b0f2b927285c8f66de61b4d74708ebeb809984.jpg)  
Isotropic Gaussian

CL-GT First three PCA axes retain 11.01% of clean variance Clean  
![](images/c2d75fc36116be99f818240558986a3cd67c95f68690baf8e8a9309ff727d33c.jpg)

![](images/4c9e1a5f72a37c354f7e66e61a1fe1d25d3e80be581dd3147ab7744a78af3ae4.jpg)  
CL-LP First three PCA axes retain 83.32% of clean variance Clean

![](images/63fe98d33b1cb600ca8a22f1b1f157affb03cdef3a46805a4117842ebcde24de.jpg)

![](images/db8351f349268d872ba5081e968c7db6c995c6a43f35da25a366d16628a71cc3.jpg)  
Isotropic Gaussian

![](images/776e8cd45f09c4dae63d79eb055d536d4aaedd6cb02329ec9e124a8a6942369a.jpg)  
CL-LT First three PCA axes retain 88.44% of clean variance Clean

![](images/28fa7b8dfa2f058472bec15888d5af2af0678ad6b9c3ac3d65c7c4544eae1097.jpg)

![](images/5309e201c7de18f81634d8c7930ca03ed8e46f91e3ad9251a7c6ab715ba8cf0f.jpg)  
Isotropic Gaussiar

![](images/15d2d0edfbad8644c51533069c2c6cc71e5c2a9e0e6bfefd3e823d4f209ba07c.jpg)

![](images/591adfabe3012d45442114beab4bf8429b5214a8a5c1397d24b93c69e5cd955b.jpg)

![](images/4920ba35cd6724b3bf181883747a59ad3191c6162504fecd36cedda362491eac.jpg)  
Figure 19: CL heldout distributions of pooled states with oracle length. Each row shows clean, severity 3, and the isotropic Gaussian reference in the same clean PCA coordinates. All supplied models are included. Clouds contain 1,200 points; surfaces span two standard deviations and are not confidence regions.

CL-LP PC1-3 83.32%  
![](images/437df7169a6268aba736f5dca9369e82118eab97cacdd7c384673a83ea435134.jpg)  
CL-LT PC1-3 88.44%  
Circle = clean Diamond = level 1 Square = level 2 Cross = level 3 Each color tracks one identity; lines connect corruption levels, not time.

How the same texts move under alternative marker pattern One averaged vector per text; true target length supplied  
CL-GP PC1-3 9.90%  
![](images/10716f56a8f782cd146f02179924249274ba885ecf6cc513cb081bb3c6f6f9ab.jpg)

CL-GT PC1-3 11.01%  
![](images/5d9a4ef869f0341c86662b7a75c715182b89579925c064ed696099cd9f315dd2.jpg)  
Figure 20: Paired CL heldout trajectories for pooled states with oracle length. Each panel follows 12 canonical identities through clean and severity levels 1–3 using one clean PCA basis per model. Colors identify targets and markers identify corruption levels. Differences in axes and captured variance require within-model interpretation.

Alternative marker pattern: clean and corrupted distributions One sampled position per text: true target length supplied CL-GP First three PCA axes retain 17.53% of clean variance Clean Corrupted (severity 3)

![](images/61c4cdf37126d370cf8a46a15e326c67657cdb7791de0d3dc4399a12459a8439.jpg)  
CL-GT First three PCA axes retain 5.46% of clean variance Clean

![](images/86b7e77daf70906407b3056607bf12ad58185ee468ccd7864306015f01c14483.jpg)

![](images/7e156cf50fb779aea12c829d2cd8f289cf43425dc3073b93937c08d04fe302bd.jpg)

![](images/f7abaa20c3fd5c1c488fdc75c575016e7d05e3903e89da1506ed51250dde7d4b.jpg)  
CL-LP First three PCA axes retain 59.56% of clean variance Clean

![](images/47223c9ed5edb6e4857b5605569e2233d518bb814f02d5e82858aa7bcef5447e.jpg)

![](images/44cf8df875faaa7c49c827069bb776dabf59c913492c12bbb4d6fbb46268f9e7.jpg)

![](images/2b4a63f9195131c8b3ee93ae561660d7bbf09e8c4b1030e45edd10e9b2efa8b3.jpg)

![](images/8e739af36ec25cced7d27e13364a3d1459767704dc30b21b04e0eb342a3ec506.jpg)  
CL-LT First three PCA axes retain 37.50% of clean variance Clean

![](images/da2044fb9c07c9351e2b6e9957e7a87e34414d2a861463316224f1729ff895a4.jpg)

![](images/4863cae1d9d18d628135f2d6b571b06a339317bde5aca194c95761ae8c292391.jpg)

![](images/fd0884643cfed165fafe402fb86761a5d1ba162d53b5a88388500b54050c5a59.jpg)

![](images/13cac150743aac950b8cf9793195ad2e01bcbb3f367fbda5b1bd639af506d55c.jpg)  
Figure 21: CL heldout distributions of token states with oracle length. Each row shows clean, severity 3, and the isotropic Gaussian reference in the same clean PCA coordinates. All supplied models are included. Clouds contain 1,200 points; surfaces span two standard deviations and are not confidence regions.

Circle = clean Diamond = level 1 Square = level 2 Cross = level 3 Each color tracks one identity; lines connect corruption levels, not time.  
How the same texts move under alternative marker pattern One sampled position per text; true target length supplied  
CL-GP PC1-3 17.53%  
![](images/36701f42fc67a5c634a57a9293394dea26aeaa732fe8d8d5231c616f59420f48.jpg)

CL-GT PC1-3 5.46%  
![](images/180a4bff3b5056a7644dfb59a958c1d8c73e41bd485051724273bd44fbbe794e.jpg)  
CL-LP PC1-3 59.56%

![](images/ea667ed33a90447d608c92af0464f4716f27b83cbf8d104d6623ecca3bb2a061.jpg)  
CL-LT PC1-3 37.50%

![](images/b162f020784f163d11a8970fa5de551d9b73f5e6529a290a8e621fe159400982.jpg)  
Figure 22: Paired CL heldout trajectories for token states with oracle length. Each panel follows 12 canonical identities through clean and severity levels 1–3 using one clean PCA basis per model. Colors identify targets and markers identify corruption levels. Differences in axes and captured variance require within-model interpretation.

![](images/5ba87beb7deb565419b35b4df704a58d7d580fc1e4c14fd42080d0ed02ffa61c.jpg)  
Figure 23: Full-cohort CL heldout diagnostics across severity, using oracle length. Retrieval, pooled and token effective rank, mean magnitude, variance scale, and cross-view error are shown separately. The bottom row uses a symmetric logarithmic scale with a linear region near zero. These curves use the original 40,000-item numeric summaries, independently of the smaller visualization subset.

![](images/2a5655df5a461ed7e5c8bfa8f7031451cac4e95071da39a33cfdc423bcff1730.jpg)

## G DOWNSTREAM SPEECH SAMPLES

SAMPLE 1: LJ033-0072

Reference/clean input: I then stepped off of it and the officer picked it up in the middle and it bent so.

Augmented input: I theN stepped off of it and the officer picked IT up in the middle and it bent so.

![](images/5143dcd48d496df27ceb06b4e5c3598e48b301722c3216b5128158e92a431d55.jpg)  
Figure 24: LJ033-0072: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>he then stepped off of it and the officer picked it up in the middle and it bent so</td><td>they then stepped off of it and the officer picked it up in the middle and it bent so</td></tr><tr><td>Frozen PL2</td><td>then stepped off of it and the officer picked it up in the middle and it bent so</td><td>then stepped off of it and the officer picked it up in the middle and it bent so</td></tr><tr><td>Frozen PL0</td><td>when kent was still white as mid months was still</td><td>at your worst at works placement memory and</td></tr></table>

Reference/clean input: He could indulge in snuff if a snuff-taker, Augmented input: He cou,d indulge ib snuff if a snuff-taker,

![](images/a4cc391e3355ace6c5f2f600bf226c8a5910a49ec528d701a2f93377c8bd1731.jpg)

![](images/c10e73e8053d608f9438c8ce9d9a702db7ef8da569f79dfbb0c9f9f99ec1db38.jpg)  
Figure 25: LJ006-0145: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>he could indulge in snuff if a snuff taker</td><td>he could be an adult and snuff if a snuff taker</td></tr><tr><td>Frozen PL2</td><td>it could indulge in snuff of a snuff digger</td><td>you can kindle it in snuff at the snuff taker</td></tr><tr><td>Frozen PL0</td><td>Empty hypothesis</td><td>plums truffle crisp paint wine spray</td></tr></table>

## SAMPLE 3: LJ002-0239

Reference/clean input: also for prisoners committed by the Admiralty Court.   
Augmented input: also for prisoners committed by the ADmIralTY Court. Reference/clean input: Marina Oswald also testified that her husband had used a bus to return home.   
Augmented input: Marina Oswald also tstfd that her hsb had used a bus to return home.

![](images/409a9d5994a3ecffda93a2901c649f95ebda18ee6e052363accb07c405a60da9.jpg)

![](images/fa2ecf409e36bbbe6903ca824005982bed1d810e126d5d21b990b6b7d5ffd3a0.jpg)  
Figure 26: LJ002-0239: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>also for prisoners committed by the admiralty court</td><td>also for prisoners committed by the admiralty court</td></tr><tr><td>Frozen PL2</td><td>also for prisoners committed by the mimralty court</td><td>also for prisoners committed by the indemnity court</td></tr><tr><td>Frozen PL0</td><td>and he would not spare winfrey</td><td>that is it riffus roy post sprawl</td></tr></table>

![](images/2b9e215e5642862772b78526e540798d098a51184c53dee96f5bf83a808b3bd4.jpg)

Clean | WER ↓ 0.0% | 5.56 s  
Augmented | WER ↓ 0.0% | 5.65 s  
![](images/d39b519f2cfd4a9fe0a797ab13f3aa4b3f044049e2362c9e78b52068c1e61d17.jpg)  
Figure 27: LJ038-0296: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>marina oswald also testified that her husband had used a bus to return home</td><td>marina oswald also testified that her husband had used a bus to return home</td></tr><tr><td>Frozen PL2</td><td>marie oswald also testified that her husband had his deputist to return home</td><td>marine oswald also disappeared at the head of mister rippon is home</td></tr><tr><td>Frozen PL0</td><td>and template shared one is rest of the world</td><td>the 2nd tent must know an ice cold mill of</td></tr></table>

Reference/clean input: was strewn in front of the dock, and sprinkled it towards the bench with a contemptuous gesture.

Augmented input: was strewn in front of the dock, and sprinkled it tOWards the bench with a contemptuous gesture.

![](images/f16f63b65414e54a0cb779b048058ea2baa9a0bd0a4a5e8b7cda377ea3f64eeb.jpg)  
Figure 28: LJ014-0142: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>was strewn in front of the dock and sprinkled it towards the bench with a contemptuous ges- ture</td><td>was strewn in front of the dock and sprinkled it towards the bench with a contemptuous ges- ture</td></tr><tr><td>Frozen PL2</td><td>was strained in front of the dock and sprinkled it towards the bench with a contemptuous gis-</td><td>was surrounded in front of the dock and sprin- kled it towards the bench with a contemptuous dister</td></tr><tr><td>Frozen PL0</td><td>ter perkins brest and west waco jan 14 1981</td><td>tim grismond antwesif walt christian orlton</td></tr></table>

Reference/clean input: She had realized all her effects, and brought them with her furniture to Greenacre’s lodgings. The two when married were to emigrate to Hudson’s Bay.

Augmented input: She h ad realized all her effects, andbrought them with Herfurniture to Greenacres lodgings.The two when married were to emigrate to HudsonsBay.

![](images/3094d1bf53b6bdfdad303e33b5ae1a1b3a1be5acb7e0a13f6cd48ae0f8e08775.jpg)  
Figure 29: LJ012-0270: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>she had realized all her effects and brought them with her furniture to greenacre is lodg- ings the 2 when married were to emigrate to</td><td>she radialized all her effects and brought them with her furniture to greenacre is lodgings the 2 when married were to emigrate to hudson</td></tr><tr><td>Frozen PL2</td><td>hudson is bay she had realized all her ethics and brought them with her furniture to grecker is lodgings the 2 when married or to immigrate to hudson</td><td>is bay she had realized all her effects and brought them with her furniture to gray nipper is lodg- ings through to when moraine wore to agarty to witton is bay</td></tr><tr><td>Frozen PL0</td><td>is bay drum flows to our white sprint of brown rents dm on town pillow down on how</td><td>apto is wescon heads oswald while rhys kill hanjuro is turn in petare</td></tr></table>

## SAMPLE 7: LJ011-0213

Reference/clean input: married the said Ellen Turner.   
Augmented input: married the said eLlen Turner.

![](images/abb390c2afa002baba064a2c9e878b1d22f76b868e73ac20876831e2df7b64fc.jpg)  
Figure 30: LJ011-0213: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>married the said ellen turner</td><td>married the said ellen turner</td></tr><tr><td>Frozen PL2</td><td>married the said ellen turner</td><td>married the said alan turner</td></tr><tr><td>Frozen PL0</td><td>and the moisture</td><td>i will do reverend is bim</td></tr></table>

![](images/914477bb60535a12a0fcc238007085cba1e6cbbd1ab449372042a5483fab9b80.jpg)

![](images/4f4c622f1675396f29980c223322150c34fe24e855ed8cb52b3f7f072aeaae63.jpg)

Reference/clean input: In our late visits, they say, ”we have seen manifest indications of a retrograde movement in this respect,

Augmented input: In ourlate visits, they say, ”we haveSEEN manifest indications of a retrograde movementin this rspct,

![](images/8e91950a2c1ea172479303871ee9c5c457c49d421d7c2a0cbf60379dc8af5fb1.jpg)

![](images/cb2b6ff6309ba9a6a06a334ec8054805412d106b22dc61cfd7e99914532b1f23.jpg)  
Figure 31: LJ007-0211: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>in our late visits they say we have seen mani- fest indications of a retrograde movement in this respect</td><td>in our late visits they say we have seen mani- fest indications of a retrograde movement in this wrist pocket</td></tr><tr><td>Frozen PL2</td><td>in your late visits they say you have seen man- ifest indications of retrigarid movement in this respect</td><td>the annalite visits they say have seen manifest indications of a retrograde movement in this respect</td></tr><tr><td>Frozen PL0</td><td>anthem for snow what is prison well this point went to the new</td><td>thank you</td></tr></table>

Reference/clean input: and the second month’s rent was paid on either April two or April three.   
Augmented input: and the second months rent was paid on eITHer April two or April three.

![](images/f25cf3257a650ac361db8652588af9c76170d786a3571e26b1f4557da3516948.jpg)

![](images/e97dafb5a1f9c1e33b70cccb468ba071a1cc78678775bd7f828c9e3f094ae0e7.jpg)  
Figure 32: LJ038-0221: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

Whisper hypotheses (normalized, as scored).
<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>and the 2nd month is rent was paid on either april 2 or april 3</td><td>and the 2nd month is rent was paid on either april 2 or april 3</td></tr><tr><td>Frozen PL2</td><td>and the 2nd month is rent was paid on either april 2 or april 3</td><td>and the 2nd month is rent was paid on either april 2 or april 3</td></tr><tr><td>Frozen PL0</td><td>uncle wuzdrow has a long tot</td><td>can not believe what is in professor regal is mind</td></tr></table>

Augmented | WER ↓ 11.8% | 6.20 s

Reference/clean input: William Crawford had been one of the promoters and managers of the Philanthropic Society’s farm school.

Augmented input: William crawfrd had been one OF the promoters and managers of the Philanthropic Societys farm sch.

![](images/be9d5aad2925269aa935f739b5bb946b50c2188e7d45486f5339d63c2742f9e3.jpg)  
Figure 33: LJ006-0020: all three systems under both input views. WER uses the clean reference; gray after a panel ends indicates no generated audio. Axes are shared within this example; the color reference is shared across all ten examples.

## Whisper hypotheses (normalized, as scored).

<table><tr><td>System</td><td>Clean audio</td><td>Augmented audio</td></tr><tr><td>MatchaTTS</td><td>william crawford had been one of the promot- ers and managers of the philanthropic society is farm school</td><td>william crawford had been one of the promot- ers and managers of the philanthropic society is farms</td></tr><tr><td>Frozen PL2</td><td>lynn crawford tedbin one of the promoters and managers of the philanthropic societies farms</td><td>lillian crawford had been one of the prompters and managers of the philanthropic society is farm show</td></tr><tr><td>Frozen PL0</td><td>and then we met my husband kyle is wife dave fruits</td><td>of gravestone and rice residue while bristling cut to</td></tr></table>

## H COMPUTATIONAL PROPERTIES, TRAINING RECIPE, AND RESPONSIBLE ASSET USE

This appendix describes computational cost, the training configuration, and the source-code workflow. It also records the provenance and reuse conditions of the datasets and software used in the experiments.

## H.1 COMPUTATIONAL PROPERTIES AND SAMPLE BUDGET

Let B be the number of sequences actually encoded per view, V the number of views, N the padded input length, M the latent-canvas length, d the width, h the number of attention heads, and $L _ { e } , L _ { d }$ the encoder and decoder depths. Let $K = 2 5 7$ be the output vocabulary size. The bounds describe dense operations in the supplied implementation. Additional sampled negatives are included in $B ;$ the sampler’s nominal batch size need not equal this effective count.

A forward pass through the encoder, Gaussian resampling, decoder, and token head costs

$$
O \big ( B V \left[ L _ { e } ( N d ^ { 2 } + N ^ { 2 } d ) + L _ { d } ( ( N + M ) d ^ { 2 } + ( M ^ { 2 } + M N ) d ) + N M d + M d K \right] \big ) .
$$

The terms account for projection/feed-forward layers, encoder self-attention, decoder self/crossattention, Gaussian resampling, and token logits. Backpropagation has the same asymptotic arithmetic order. The decoder produces all positions in parallel, but attention remains quadratic in sequence length. Gaussian resampling is evaluated even when an equal-length identity branch replaces its result, so this computation is included in the bound.

The parameter count scales as $O ( ( L _ { e } + L _ { d } ) d ^ { 2 } + K d )$ , apart from positional buffers; gradients and AdamW states require the same asymptotic parameter storage. A conservative dense trainingactivation bound is

$$
O \big ( B V \left[ ( L _ { e } N + L _ { d } ( M + N ) ) d + h \{ L _ { e } N ^ { 2 } + L _ { d } ( M ^ { 2 } + M N ) \} + M N + M K \right] \big ) .
$$

Attention kernels can reduce materialized score storage without changing the dense arithmetic bound. Resampling processes target positions in chunks of at most 256, bounding the temporary weight block by $\bar { O ( B V \operatorname* { m i n } ( M , 2 5 6 ) N ) }$ . The training bound also accounts for concatenated outputs and tensors retained for backward.

For P random projections, VISReg projection and per-projection sorting cost $O ( V ( B d P \ +$ $P B \log B ) )$ , in addition to centering and scale terms. With projection chunks of size $P _ { c } ,$ the forward workspace is bounded by $\bar { O ( V B d + V B P _ { c } + d P _ { c } ) }$ , excluding backward state. The CL defaults use $P = P _ { c } = 2 5 6$ . Token regularization samples one position per sentence, so this branch operates on B sampled states per view rather than all BM positions. Pooling and dense cross-view agreement still require $O ( B V \bar { M } d )$ operations.

For $n _ { q }$ queries and $n _ { g }$ candidates, the exact positional retrieval diagnostic costs $O ( n _ { q } n _ { g } R d )$ , where $R = { \dot { 1 } } 6$ sampled relative positions give $R \bar { d } = 8 . 1 9 2$ coordinates. A query block of size Q needs $O ( Q n _ { g } )$ score storage, in addition to stored features of order $O ( ( n _ { q } + n _ { g } ) R d )$ . The default $Q = 6 4$ limits score workspace but does not remove the quadratic all-pairs cost. Saving full canvases instead requires $O ( n M d )$ storage. Forming and decomposing a d-dimensional covariance matrix costs $O ( n d ^ { 2 } + d ^ { 3 } )$

Over S optimizer steps, training costs S times the per-step forward/backward cost, plus optimizer updates. PL and CL use a five-epoch budget and nominal training batch size 36; their samplers determine the corresponding number of optimizer steps. The prepared PL and CL training sets contain 8,635,993 and 8,656,553 rows, respectively; their split construction and duplicates are described above. Evaluation uses 40,000 unique targets and 1,024 disjoint calibration targets, while TTS uses 9,170 training and 3,930 validation utterances. These empirical budgets specify the scale of the experiments; the theoretical results concern information retention and population risk. Bootstrap intervals describe the sampled evaluation cohort and do not quantify variation across retraining seeds.

## H.2 EXECUTABLE DEFAULTS AND TRAINING DETAILS

The effective settings are obtained by merging the shared configuration with each variant’s overrides, then applying unoverridden parser and MatchaTTS configuration defaults. All models use a nominal training batch size of 36. The accompanying code package contains the full configs/experiment.json, provenance/parser\_defaults.json, the MatchaTTS configuration tree, pinned requirements, and a Slurm launcher that prints every command with --dry-run. Together, these files specify the full execution recipe.

PL. The launcher uses seed 42, width 512, eight heads, four encoder layers, two decoder layers, a 2,048-position cap, zero dropout, and resampling temperature 0.3. AdamW uses learning rate

$5 \times 1 0 ^ { - 5 }$ , weight decay 0.01, betas (0.9, 0.98), 5% warmup followed by cosine decay to 0.0005 of the initial rate, gradient clipping at 1, and enabled automatic mixed precision. Training lasts five epochs with no additional positive step cap. The nominal batch is 36, with eight workers, bucket factor 32, and prefetch factor 4. Each canonical item supplies three easy and two hard augmented views in addition to its clean view. The sampler requests six informal and sixteen negated examples, and the default unrelated pool has sixteen entries. The remaining shared objective settings are $n _ { \mathrm { p a d } } = 4$ $\tau = 0 . 5 , \eta = 0 . 1 , \gamma = 1 , \alpha = 0 . 5 , \beta = 0 . 3$ , and $m _ { 1 } = m _ { 2 } = 0 . 2$

PL0 selects the latent objective with λ = 1 and VISReg weight 0.3. PL1 uses batch 36 and eight unrelated entries. PL2 and PL3 set VISReg weights 0.1 and 0.5. PL4, PL5, PL6, and PL7 respectively set $\alpha , \beta , \eta .$ and m<sub>2</sub> to zero. Checkpointing and validation occur every 2,000 steps; logging occurs every 50 steps. The probe interval is 4,000 steps with 400 probe epochs and 160 evaluation texts. The historical PL5 budget exception and the recorded PL0/PL2/PL3 checkpoint steps remain as stated in Appendix B.6; the variant mapping identifies the corresponding experimental comparisons.

CL. The default training seed is 1; the data seed is 2027. The nominal batch is 36 with eight workers, and the width, heads, encoder/decoder depths, length cap, dropout, resampling temperature, learning rate, decay, warmup, gradient clip, and five-epoch budget match the settings above. CL uses oracle lengths and recoverable corruption. All four primary arms use VISReg weight 0.01; LP/LT use latent agreement and GP/GT use token grounding, with pooled/token regularization selected by the final letter. VISReg center, scale, and shape weights are one, with shape standard-deviation floor 0.1 and global views 0 and 2. The data preparation fractions are 0.90/0.05/0.05, the streaming shuffle buffer is 10,000, insertion probability is 0.1, and the maximum insertion run is two. The round-trip audit checks 4,096 accepted examples. Nonfinite batches follow the configured skip policy. Logging is every 50 steps, validation/checkpointing every 2,000, and monitoring every 4,000.

Speech and evaluation. The MatchaTTS recipe uses seed 42, Adam at $1 0 ^ { - 4 }$ with zero weight decay, batch 36, twelve data workers, gradient clipping at 5, mixed bfloat16 precision, and 1,000 epochs. Its architecture and frozen-state interfaces are provided in the code and Appendix D. Stored augmentation is selected with probability 0.7. Evaluation uses ten flow steps, temperature and length scale one, and Whisper large-v3 with float16 inference, English normalization, and beam size five. Text evaluation uses seed 9027, batch four, retrieval blocks of 64, and 500 bootstrap replicates. The executable TTS default is 1,000 bootstrap replicates; the retrospective manuscript WER audit uses 10,000. The executable geometry subset defaults to 2,048 items; the historical PCA diagnostic uses the separately reported 4,096-item subset. The audit and plotting settings identify the corresponding reported analyses.

Environment and selection record. The supplied recipe targets Python 3.10–3.12, Py-Torch/torchaudio 2.8.0, torchvision 0.23.0, and CUDA 12.8 wheels. All other direct Python requirements are version-pinned, including NumPy 1.26.4, Transformers 4.53.3, Datasets 3.6.0, Lightning 2.5.2, and sentence-transformers 5.0.0. GCC/C++, eSpeak NG, FFmpeg, and a compatible CUDA/cuDNN runtime are system prerequisites. Setup checks dependency consistency and records the installed environment. The evaluation environment is reported separately in Appendix B.6. Training uses the author-confirmed configuration and explicit variant settings described above. The Slurm launcher records resource requests and leaves cluster-specific account and partition fields configurable.

## H.3 REPRODUCTION WORKFLOW

The companion archive supplies data preparation, PL/CL training, feature extraction, text evaluation, speech training, and figure generation, together with pinned dependencies and Slurm instructions. The workflow prepares the corpus, trains the selected models, and passes their outputs to the evaluation and plotting stages. The README gives the setup commands, configuration fields, and output locations; --dry-run prints the complete command and dependency plan before submission.

Corpus input is configured through CANOPE DATASET ID or the documented local CSV/JSONL interface. The compilation is hosted at https://huggingface.co/datasets/avalonai/ eng $\mathtt { \_ i s h - s i n g l i s h - g 2 p }$ . The schema, filtering rules, split construction, and prepared-data counts are specified in Appendix B.6. Run manifests record configuration, data provenance, and evaluation identities so that comparisons use consistent inputs. The numeric summaries and row-level speech results also support the supplied figure scripts and paired WER audit.

## H.4 ATTRIBUTION, LICENSES, CONSENT, AND SENSITIVE CONTENT

Attribution and licenses. The component corpora are credited in Appendix B.6. Reused encoders, MatchaTTS, HiFi-GAN, and Whisper are cited in the evaluation and speech sections. Benchmark attribution includes LexNorm (Baldwin et al., 2015), SemAntoNeg (Vahtola et al., 2022), PAWS (Zhang et al., 2019), STS-B (Cer et al., 2017), and SICK (Marelli et al., 2014). The authors assembled the training compilation from the component corpora identified in Appendix B.6. The code preserves upstream software notices, and its requirements identify external libraries and versions.

The compilation card specifies Apache-2.0, while component data retain their original terms. LJSpeech declares its audio and transcriptions public domain; LibriSpeech specifies CC BY 4.0; and CANNOT specifies CC BY-SA 4.0.<sup>1</sup> Wikipedia text is distributed under attribution/share-alike terms, with the applicable version determined by the source revision.<sup>2</sup> The supplied MatchaTTS and HiFi-GAN source includes MIT license notices. The companion asset inventory records asset-specific provenance and licensing information; separately distributed data and model weights remain subject to their respective release terms. The compilation label does not replace these component conditions.

Data-provider and consent context. This study reuses publicly released corpora distributed by their original providers and curators. It involved no new participant recruitment, crowdsourcing, or collection of private messages. The authors’ contribution to the dataset was aggregation and preparation for the experiments. Collection procedures and participant-consent conditions belong to the original source releases and are described in the cited curator documentation; aggregation does not introduce a new participant-consent process. This statement describes the context of secondary use rather than asserting independently obtained consent from every person mentioned in the source text.

Potentially sensitive content. Personal messages, user-authored text, and historical material can contain identifying details, offensive language, or stereotypes. Informal-language and negation transformations can preserve or alter this content. The experiments analyze aggregate representation and speech metrics; data-validity and invertibility checks address the experimental protocol rather than sensitive-content screening. Reuse of individual examples should therefore follow the source terms and take the original content context into account.