# Early-Bird Decoding: Accelerating Diffusion LLMs with Learnable Block Sizes and Parallel Sampling

Lixuan Wei<sup>1∗</sup>, Wei Zhou<sup>2∗</sup>, Jianwen Wu<sup>3</sup>, Yipeng Shen<sup>3</sup>, Meiling Wang<sup>3</sup>, Haoran You<sup>3†</sup>

<sup>1</sup>Harvard University <sup>2</sup>Georgia Institute of Technology <sup>3</sup>Purdue University

## Abstract

Diffusion large language models (dLLMs) offer a promising parallel decoding paradigm as an alternative to autoregressive generation through iterative unmasking. However, dLLMs typically require many steps before token confidence reaches the decoding threshold, resulting in inefficient inference even with block-wise KV caching. To accelerate dLLM inference, wefor thefirst time propose an “early-bird (EB)” decoding framework, motivated by the observation that tokens with similarly low entropy tend to cluster and can be jointly decoded earlier, before reaching the confidence threshold. In particular, our EB-Decode framework integrates two key enablers: (1) a learnable network that adaptively groups tokens with similar uncertainty into variable-length blocks, rather than relying on fixed block sizes; (2) a position-aware sampler that learns to unmask tokens in parallel using fewer decoding steps within predicted variable-length blocks. Both components are developed without modifying pretrained dLLM weights and can therefore be directly deployed as plug-ins during serving, with negligible training and inference overhead. Extensive experiments across three models and four benchmarks consistently validate our observation and the effectiveness of EB-Decode, achieving 3.53–18.76× higher throughput than the vanilla decoding method and up to 1.58× higher throughput over the strongest baseline, Fast-dLLM, with comparable accuracy.

## 1 Introduction

Diffusion large language models (dLLMs) have emerged as a compelling alternative to autoregressive (AR) models, breaking the sequential bottleneck of token-by-token decoding by generating text in parallel through iterative denoising over masked positions [2, 42, 40, 56]. This paradigm enables bidirectional context utilization and makes dLLMs a practical and increasingly scalable approach for fast, high-quality text generation, as demonstrated by large-scale systems such as Gemini Diffusion [15], Seed-Diffusion [45], and Mercury [25]. However, current open-source dLLMs such as LLaDA [40, 64, 5] and Dream [56] still struggle to consistently deliver better accuracy-efficiency tradeoffs compared to their AR counterparts of similar size; for example, the AR model LLaMA3- 8B-Instruct achieves 48.0 tokens/s, whereas the dLLM LLaDA-8B-Instruct reaches only 3.5 tokens/s under an NVIDIA A100-PCIe 40GB GPU setup. Its accuracy also drops from 83.5% at 1,024 denoising steps to 54.1% at 256 denoising steps [49, 41, 24, 29], which limits the widespread adoption of dLLMs in real-world deployments.

Recently, several works have attempted to mitigate the inference efficiency challenges of dLLMs, including semi-AR variants for block-wise sequential generation [1, 10], Key-Value (KV) and activation caching [53, 33, 38, 21, 8], efficient sampling algorithms [22, 4, 51, 35, 23, 48, 60, 36], and step distillation [19, 13, 9, 30, 59]. However, most of them rely on fixed block selection and confidence thresholds during decoding, implicitly assuming uniform token difficulty along the sequence, which wastes parallelism on hard tokens and under-utilizes it on easy ones, contradicting the heterogeneous nature of natural language. For example, in code, function signatures are typically easier to predict than function bodies, and in mathematical reasoning, routine arithmetic is straightforward whereas multi-step derivations are more challenging. Beyond that, a few recent methods introduce simple heuristics for adaptive block sizing and thresholding, such as delimiter (e.g., period) detection [35], sliding windows [36], or entropy-based boundaries [60]. Some methods attempt to learn parallel decoding strategies [4], but still rely on fixed block sizes. These limitations call for a principled framework that not only adaptively determines block sizes but also inherently enables early decoding under homogeneous token difficulty within each block.

![](images/8e78491610a37a9e25a800d45fa67e8c3a8fb3403d302b0878d7aa689fc61697.jpg)  
Figure 1: Conceptual comparison illustrating the differences between vanilla block-wise decoding (Left) [1] and the proposed EB-Decode framework (Right).

In this work, wefor thefirst time propose a principled “early-bird” decoding framework, motivated by our observation that tokens with similar low entropy and close semantic meanings tend to cluster and can be jointly decoded much earlier before reaching the confidence threshold, due to their similar difficulty levels. To enable such principled EB decoding, two key challenges arise: First, how can we automatically learn adaptive block sizes instead of relying on heuristics? Unlike prior delimiterbased methods, our observation shows that token entropy exhibits a staircase pattern across token positions, indicating that at certain key steps, adjacent or even non-contiguous tokens share similar entropy or uncertainty. This motivates the design of a lightweight network that leverages entropy and positional semantics to dynamically predict block sizes on the fly. Second, how can we decode the predicted block of tokens earlier before reaching the confidence threshold? Our observation shows that confidence-based decoding wastes computation on tokens that have already converged: many tokens become correct in early denoising steps but are repeatedly remasked because their confidence has not yet reached the threshold. Moreover, under fixed block sizes, confident tokens outside the current block are excluded from early finalization. These observations motivate a position-aware, learnable parallel sampler that leverages per-token statistics (i.e., entropy, position, and step) to determine which positions can be finalized at earlier steps, while naturally supporting variable-length blocks. To the best of our knowledge, this work is the first to tackle the above challenges toward a principled EB decoding framework. Our contributions are summarized as follows:

• We propose a principled early-bird decoding framework for efficient dLLM inference acceleration, termed EB-Decode, that automatically clusters tokens with similar difficulty into blocks and enables their early decoding with fewer denoising steps than confidence-based methods.

• Enabler 1: We adopt a learnable block size (LBS) prediction network to dynamically cluster contiguous or non-contiguous tokens with similar difficulty and semantic coherence on the fly.

• Enabler 2: We introduce a position-aware learnable parallel sampling (LPS) network to identify and finalize converged tokens at early denoising steps under variable-length blocks.

• Extensive experiments across three dLLMs and four representative benchmarks demonstrate the effectiveness of EB-Decode: it achieves 3.53–18.76× throughput improvement over the vanilla decoder with comparable accuracy and only ∼6.21% routing overhead, and delivers an additional 1.20× speedup when combined with KV-cache pipelines.

## 2 Related Works

dLLMs. Unlike AR models, dLLMs generate text through an iterative denoising process over discrete tokens [2, 42, 44, 34, 61], achieving likelihood comparable to their AR counterparts. At the billionparameter scale, LLaDA [40] performs on par with LLaMA3, and subsequent extensions further improve its alignment, sparsity, and scaling [63, 64, 5]. Dream-7B [56] adds AR-based initialization and adaptive noise rescheduling. We provide more literature review of dLLM in Appendix A.

Efficient Inference of dLLMs. Recent work improves dLLM decoding along several directions. Cache-based methods reuse key-value projections across denoising steps to avoid redundant computation [33, 53, 38, 21]. Distillation-based methods train the model to commit more tokens per step [9, 30]. Efficient sampling methods redesign the denoising trajectory to better allocate computation across steps. For example, SlowFast [51] alternates between exploratory and accelerated phases; DUS [37] front-loads computation to early steps via dilated scheduling; Adaptive acceptance methods replace the static confidence threshold with learned or per-position confidence [23, 4, 39, 28]. Dynamic block methods resize the block at runtime by aligning boundaries with confidence or entropy shifts [35, 60, 36]. An orthogonal line targets test-time quality through revocable draft-and-verify decoding [20], temporal-dynamics voting across denoising steps [48], and inference-time remasking [47]. In contrast, our proposed principled EB-Decode explicitly leverages the token-level statistics, and learns to dynamically cluster tokens with similar difficulty and adapt the sampling schedule to identify and finalize converged tokens at early denoising steps under variable-length blocks.

Early-Bird (EB) Phenomenon. Early prediction is important for efficient training and inference. For training, the EB ticket hypothesis [57] shows that small subnetworks (i.e., lottery tickets [14]) can be identified early in training via mask-distance convergence, achieving accuracy comparable to overparameterized networks. This EB phenomenon has been consistently observed in BERT [7], GCNs [58], LLMs [18], and diffusion models [52]. For inference, early-exit mechanisms enable dynamic computation by terminating inference once predictions become sufficiently confident. They were first applied to RNNs through a learned halting unit [17], and later generalized to CNNs [46] and Transformers [54, 62, 43], where intermediate layers can halt computation once a confidence or prediction-agreement criterion is satisfied. In this work, we for the first time observe the EB phenomenon during the decoding phase in dLLMs and leverage it to enable efficient EB decoding.

## 3 Preliminaries of dLLMs

dLLMs formulate text generation as a diffusion process over discrete token sequences, consisting of a forward masking process and a reverse denoising process. Let $\mathbf { x } _ { 0 } = ( x _ { 0 } ^ { 1 } , \ldots , x _ { 0 } ^ { L } )$ denote a clean token sequence of length L drawn from a vocabulary V, where $x _ { 0 } ^ { i }$ is the token at position i. The forward process samples a noise level $t \in [ 0 , 1 ]$ and produces a corrupted sequence $\mathbf { x } _ { t } \overset { \cdot } { = } ( x _ { t } ^ { 1 } , \ldots , x _ { t } ^ { L } )$ by independently replacing each token with the special [MASK] symbol with probability t, so that $\mathbf { x } _ { t }$ converges to a fully masked sequence as $t  1$ . The reverse process is parameterized by a bidirectional Transformer that defines a mask predictor $p _ { \theta } ( \cdot \mid \mathbf { x } _ { t } )$ , which recovers the distribution over the original token at every masked position simultaneously. The model is trained by minimizing

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t \sim \mathcal { U } ( 0 , 1 ) , \mathbf { x } _ { 0 } , \mathbf { x } _ { t } } \left[ \frac { 1 } { t } \sum _ { i = 1 } ^ { L } \mathbf { 1 } _ { \{ x _ { t } ^ { i } = \mathrm { u A s K } \} } \left( - \log p _ { \theta } ( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t } ) \right) \right] ,\tag{1}
$$

where $\mathbf { 1 } _ { \{ \cdot \} }$ is the indicator function that restricts the cross-entropy to currently masked positions, and the factor $\dot { 1 } / t$ compensates for the expected mask ratio at noise level t. This objective upper-bounds the negative log-likelihood of the model distribution.

Block-wise Decoding. At inference time, dLLMs commonly adopt a block-wise decoding strategy [1, 53]: the response is partitioned into contiguous fixed-size blocks decoded from left to right, where positions inside the active block are predicted in parallel and committed based on confidence, while the rest are remasked for further refinement. The full update rule is deferred to Appendix B. However, reliance on fixed block sizes and static commit rules ignores variation in token difficulty along the sequence, motivating our empirical analysis in Sec. 4.

## 4 The Observations of EB Phenomenon in dLLM Decoding

To understand the inefficiency of existing dLLM decoding, we analyze the denoising trajectory and identify two empirical phenomena overlooked by fixed-block, threshold-based decoding: a staircase entropy pattern across token positions and an early convergence phenomenon across denoising steps.

Observation 1: Staircase Entropy Pattern. To examine how token difficulty varies across positions, we measure the Shannon entropy $\mathsf { \bar { H } } _ { t } ^ { i }$ at each masked position i and denoising step t. We visualize this on a GSM8K sample with generation length 256, decoded by the block-wise diffusion method [53].

$$
\mathrm { F i g . ~ 2 ~ ( a ) }
$$

As shown in , the entropy does not decrease smoothly across token positions. Instead, it exhibits a staircase pattern, consisting of long, flat low-entropy plateaus that typically correspond to predictable syntax or boilerplate content, such as reasoning connectives like ${ } ^ { \ast } T _ { O }$ solve this problem, wefirst need $t o \dots \vec { }$ in mathematical reasoning tasks, or structural scaffolding around known function signatures like def function\_name(args): in code generation tasks. This suggests that semantically coherent tokens within each plateau share similar difficulty, and these regions exhibit similarly low entropy across denoising steps. However, this conflicts with fixed block boundaries, under which such semantically coherent regions are split in a manner agnostic to semantic structure, preventing stabilized regions across block boundaries from being decoded jointly and leaving the available parallelism unexploited. In addition, these flat plateaus are occasionally interrupted by short high-entropy spikes corresponding to harder-to-predict tokens. Although such hard

![](images/64e965d619adef75653e97e6370426d8bc2d7fc96f6200303ac6093a325f94c3.jpg)

![](images/e0afba38bcd981bb48ebe3d1a220d625667228d8805ba31c3ec1ff151a95b7d9.jpg)  
Figure 2: Comparison of entropy heatmaps between (a) fixed block size and (b) our noncontiguous learnable block size method, where each colored region represents a block. Gray regions denote tokens that have been unmasked.

tokens sit within an otherwise semantically coherent region, they are not well-suited to being decoded jointly with the surrounding easy tokens. Yet fixed-block methods enforce a strict left-to-right order that forces these hard tokens to be resolved inside the current block before any subsequent content can be generated. In contrast, as shown in Fig. 2 (b), a block selection strategy that allows non-contiguous grouping could defer these hard tokens to a later block, letting the model first decode the surrounding easy context and leverage the additional right-side information when eventually resolving them.

(a) Per-token Denoising Trajectory
<table><tr><td>t-5</td><td>we</td><td>need</td><td>to</td><td>follow</td><td>these</td><td>steps</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>t-4</td><td>we</td><td>need</td><td>to</td><td>follow</td><td>these</td><td>steps</td></tr><tr><td>t-3</td><td>we</td><td>need</td><td>to</td><td>follow</td><td>these</td><td>steps</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>t-2</td><td>we</td><td>need</td><td>to</td><td>follow</td><td>these</td><td>steps</td></tr><tr><td>t-1</td><td>we</td><td>need</td><td>to</td><td>follow</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>these</td><td>steps</td></tr><tr><td>t</td><td>we</td><td>need</td><td>need</td><td>to</td><td>follow</td><td>0</td></tr><tr><td>GT</td><td>we</td><td>need</td><td>to</td><td>follow</td><td></td><td></td></tr><tr><td></td><td colspan="4">Correct Token Incorrect Token</td><td colspan="2">these steps</td></tr></table>

(b) Block-relative Lead-time on GSM8K
<table><tr><td></td><td>Vanilla [40]</td><td>Confidence- based [53]</td><td>EB-Decode (Ours)</td></tr><tr><td>Mean Time</td><td>13.4 Steps</td><td>2.3 Steps</td><td>0.7 Steps</td></tr><tr><td colspan="4">Fraction of Tokens</td></tr><tr><td>≥1 Step</td><td>96.3%</td><td>62.5%</td><td>25.1%</td></tr><tr><td>≥3 Steps</td><td>88.3%</td><td>35.0%</td><td>10.5%</td></tr><tr><td>≥5 Steps</td><td>80.2%</td><td>18.1%</td><td>4.6%</td></tr><tr><td>≥10 Steps</td><td>60.5%</td><td>3.5%</td><td>0.7%</td></tr></table>

Figure 3: (a) Per-token denoising trajectory, where each column represents a token position across steps within one segment. Light green cells indicate tokens whose predictions become correct before they are committed, illustrating the inefficiency of confidence-based decoding. (b) Mean lead-time (in denoising steps) and fraction (%) of tokens with lead ≥ k denoising steps before commit.

Observation 2: Early Convergence Phenomenon. Even when block selections are well aligned with token difficulty, decoding still has to decide when each token within a block should be committed. To examine this, we track the per-step prediction at each masked position for six representative token positions of a GSM8K sample decoded using the vanilla block-wise method [40]. As shown in Fig. 3 (a), we observe that many tokens reach the correct prediction several denoising steps before their per-token confidence crosses the threshold, in principle allowing them to be committed early. These early-converged tokens, however, are repeatedly remasked before being finalized, introducing steplevel redundancy. Fig. 3 (b) quantifies this redundancy by showing that under both the vanilla [40] and confidence-based method [53], many tokens remain unaccepted for multiple denoising steps after their predictions become correct. As reported in Fig. 3 (b), the mean lead times of the two baseline methods are 13.4 and 2.3 steps, respectively. The fractions of tokens are 96.3% and 62.5% when the lead time is ≥ 1 step, and 60.5% and 3.5% when the lead time is ≥ 10 steps. This behavior arises because existing methods evaluate each masked position independently and cannot distinguish correct-but-uncertain predictions from genuinely incorrect ones, highlighting the need for a principled and learnable strategy to commit correct tokens earlier.

![](images/2821ac446aac752fe847678e32556e818b6e57fdeb18c4a9fbc0a0a2ffef256b.jpg)  
Figure 4: Overview of the proposed EB-Decode framework. (a) Inference pipeline: LBS first predicts and filters the active block out, and LPS then selects the tokens to commit within this block. (b) LBS router training with a weighted cross-entropy loss and a regularization term. (c) LPS router training, where features and labels are first recorded by decoding with LBS and then used to train the router.

## 5 The Proposed EB-Decode Framework

Overview. The two observations in Sec. 4 reveal two fundamental issues in current dLLM decoding: (1) a fixed block size that ignores the staircase pattern of token difficulty and the non-contiguous nature of tokens that should be grouped, and (2) a fixed confidence threshold that overlooks the adaptive nature of how early predictions actually converge. As illustrated in Fig. 4 (a), our proposed EB-Decode framework replaces the aforementioned two static choices with learnable counterparts while keeping the base dLLM frozen. Specifically, LBS replaces the fixed block size and decides where to decode at each step, as described in Sec. 5.1, while LPS replaces the fixed confidence threshold and decides which positions within a given variable-length block to finalize early, as described in Sec. 5.2. Both modules reuse the base dLLM’s forward pass and introduce negligible training and inference overhead.

## 5.1 Enabler 1: Learnable Block Size (LBS)

The staircase entropy pattern identified in Sec. 4 reveals that token difficulty is spatially clustered: lowentropy regions alternate with sharp uncertainty spikes that do not align with fixed block boundaries. A fixed partition either splits an easy plateau across two blocks, wasting a decoding pass, or lumps easy and hard tokens together, causing hard tokens to bottleneck their easier neighbors. To resolve this, we introduce LBS, a lightweight router $\mathcal { R } _ { \phi }$ that at each step selects a variable-length, non-contiguous active block $B \subseteq { \mathcal { M } }$ from the currently masked positions $\mathcal { M } = \{ i : x ^ { i } = \mathtt { [ M A S K ] } \}$

LBS Router. As shown in Fig. 4 (b), for each masked position i, the router takes two complementary inputs: (1) The normalized entropy, defined as $\begin{array} { r } { \tilde { H } ^ { i } = \tilde { H } ( p ^ { i } ) = ( - \sum _ { v \in \mathcal { V } } p ^ { i } ( v ) \log p ^ { i } ( v ) ) / \log | \mathcal { V } | . } \end{array}$ which measures local prediction uncertainty. Entropy alone, however, does not distinguish between qualitatively different sources of uncertainty: a function word may be uncertain among a few interchangeable alternatives, while a rare content word may be uncertain because its identity depends on context not yet resolved—two situations that call for different routing decisions; (2) The top-1 predicted token IDs $( \hat { x } ^ { i } )$ . Inspired by Lu et al. [35], we embed token IDs using the frozen base-model token embedding table. This allows the router to differentiate between semantically trivial tokens and semantically informative ones. In summary, the router maps these input signals to a per-masked-token inclusion probability $p _ { \phi } ^ { i }$ of whether a masked token should be included in the final selected block:

Algorithm 1 EB-Decode Training Algorithm 2 EB-Decode Inference   
Require: Frozen dLLM ${ \overline { { \mathcal { F } _ { \theta } } } } ,$ , prompt-response pair D Require: $\mathcal { F } _ { \theta } , \mathcal { R } _ { \phi } , \mathcal { R } _ { \psi } ,$ , and thresholds $\tau , \tau _ { \psi }$   
// LBS router $\mathcal { R } _ { \phi }$ 1: x ← Prompt &[MASK]   
1: for $x \in \mathcal { D }$ do 2: while any $\mathsf { \widehat { x } } ^ { i } = [ \mathsf { M A } \mathsf { S } \mathsf { K } ]$ do   
$\tilde { H } ^ { i } , \hat { x } ^ { i } \gets \mathcal { F } _ { \theta } ( \mathbf { M a s k } ( x ) )$ ▷ LBS Inputs 3: $\tilde { H } ^ { i } , \hat { x } ^ { i } , f ^ { i } \gets \mathcal { F } _ { \theta } ( x )$   
${ O } _ { \phi } ^ { i }  \mathcal { R } _ { \phi } ( \tilde { H } ^ { i } , \hat { x } ^ { i } )$ 4: $f ^ { i } = ( \kappa ^ { i } , \tilde { H } ^ { i } , \Delta ^ { i } , \rho )$ ▷ Feature Details   
Update ϕ via L<sub>LBS</sub> ▷ Eq. (3) // LBS   
5: end for if no active block then   
// LPS router $\mathcal { R } _ { \psi }$ ${ \cal B } \gets \mathrm { F i l t e r s } \big ( \mathcal { R } _ { \phi } ( \tilde { H } ^ { i } , \hat { x } ^ { i } ) \big )$   
6: for $x \in \mathcal { D }$ do end if   
7: Decode with LBS; record trace $\tau$ $/ / L P S$   
8: end for 8: $O _ { \psi } ^ { i }  \mathcal { R } _ { \psi } ( f ^ { i } , \mathrm { p o s } ^ { i } )$   
9: for $( f ^ { i } , \operatorname { p o s } ^ { i } ) \subset \mathcal { T }$ do ▷ LPS Inputs 9: $\mathcal { A }  \{ i \in \mathcal { B } : \kappa ^ { i } \geq \tau \vee p _ { \psi } ^ { i } > \tau _ { \psi } \}$   
10: $O _ { \psi } ^ { i }  \mathcal { R } _ { \psi } ( f ^ { i } , \mathrm { p o s } ^ { i } )$ 10: $x ^ { i } \gets \hat { x } ^ { i }$ for $i \in \mathcal A$ ▷ Accept & Commit   
11: Update $\psi \ \mathrm { v i a } \ \mathcal { L } _ { \mathrm { L P S } }$ ▷ Eq. (4) 11: If B resolved, move to Line 5   
12: end for 12: end while   
13: return $\mathcal { R } _ { \phi } , \mathcal { R } _ { \psi }$ 13: return x

$$
p _ { \phi } ^ { i } = \sigma ( O _ { \phi } ^ { i } ) = \sigma ( \mathcal { R } _ { \phi } ( \tilde { H } ^ { i } , \hat { x } ^ { i } ) ) \in ( 0 , 1 ) , i \in \mathcal { M } ,\tag{2}
$$

where $\sigma$ is the sigmoid function. The router is implemented as a lightweight two-layer Transformer encoder, with full architectural details described in Appendix C.1.

LBS Router Training. We train the LBS router in a block-wise masking style. A single frozen forward pass through the dLLM yields entropy ${ \tilde { H } } ^ { i }$ and token ID ${ \hat { x } } ^ { i }$ at every masked position, which are then passed into the LBS router to get the per-token probability $p ^ { i }$ . We train the router by minimizing a weighted cross-entropy loss $( \mathcal { L } _ { \mathrm { w C E } } )$ with a regularization term $\scriptstyle ( { \mathcal { L } } _ { \mathrm { R e g } } ) \colon$

$$
\mathcal { L } _ { \mathrm { L B S } } = \mathcal { L } _ { \mathrm { w C E } } - \mathcal { L } _ { \mathrm { R e g } } = \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } \sigma ( O _ { \phi } ^ { i } ) \cdot \ell _ { \mathrm { C E } } ^ { i } ~ - ~ \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } \sigma ( O _ { \phi } ^ { i } )\tag{3}
$$

where $\ell _ { \mathrm { C E } } ^ { i }$ denotes the per-masked-token cross-entropy loss of the frozen dLLM at position i, and ${ O } _ { \phi } ^ { i } = \mathcal { R } _ { \phi } ( \tilde { H } ^ { i } , \hat { x } ^ { i } )$ is the router output. The first term $\mathcal { L } _ { \mathrm { w C E } }$ trains the router to assign high inclusion probability (i.e., likelihood of being included in the current block) to positions where the dLLM already predicts correctly (i.e., low $\ell _ { \mathrm { C E } } ^ { i } )$ , and assign low inclusion probability to positions where the dLLM is either incorrect or underconfident (i.e., high $\ell _ { \mathrm { C E } } ^ { i } )$ . The second term acts as a regularization role that prevents the degenerate solution of selecting no tokens—which would trivially minimize the first term—by rewarding larger total inclusion; the subtraction ensures the regularizer and the loss pull in opposite directions, stabilizing the block size. The training procedure is also summarized in Alg. 1; the full pseudocode is detailed in Appendix C.2.

LBS Router Inference. As shown in Alg. 2, at each step, entropy ${ \tilde { H } } ^ { i }$ and token ID ${ \hat { x } } ^ { i }$ are computed from the dLLM output logits over masked positions and passed to $\mathcal { R } _ { \phi } .$ , producing a candidate block $\mathcal { C } = \{ i : p _ { \phi } ^ { i } > 0 . 5 \}$ . This candidate block is then refined by three filters, each addressing a specific failure mode. (1) EOS removal. End-of-sequence tokens have near-zero entropy, so the router nearly always selects them. However, unmasking EOS tokens too early in the inference process would truncate generation prematurely and severely damages accuracy on tasks that require step-by-step reasoning like math and coding [22]. Thus, we remove all EOS tokens from C entirely. We quantify the dominance of EOS predictions at the sequence tail in Appendix H. (2) Max-gap constraint $( g _ { \mathrm { m a x } } )$ Ideally, we want to identify a homogeneous difficulty plateau as the block, instead of cherry-picking isolated easy tokens scattered across a hard region. If consecutive selected positions are separated by more than $g _ { \mathrm { m a x } }$ tokens, the candidate block straddles an unresolved hard region, violating the spatial coherence assumption and forcing the model to denoise positions that have not yet been committed. Thus, starting from the leftmost selected position, any candidate whose gap to its predecessor exceeds $g _ { \mathrm { m a x } }$ is discarded along with all subsequent selected positions. (3) Min-tokens constraint $( L _ { \mathrm { m i n } } )$ After applying the above two filters, if the resulting candidate set contains fewer than $L _ { \mathrm { m i n } }$ tokens, LBS falls back to selecting the first 32 masked positions in left-to-right order. This constraint is critical for accuracy. If the number of tokens is too small after filtering, the router can degenerate to selecting scattered easy tokens anywhere in the sequence. In summary, the router makes per-token decisions, resulting in an inherently non-contiguous candidate block that adapts to the entropy pattern; contiguity is instead imposed at inference time via filtering.

## 5.2 Enabler 2: Learnable Parallel Sampling (LPS)

LBS adapts block boundaries to local difficulty but keeps the commit rule fixed, leaving intra-block parallelism underutilized. The early convergence observation in Sec. 4 shows that a masked position’s top-1 prediction often becomes correct several steps before its confidence reaches the threshold, making additional refinement steps unnecessary. To bridge this gap, we introduce LPS, a lightweight router $\mathcal { R } _ { \psi }$ that learns when a masked position within a variable-length block is ready to be finalized, enabling early commitment before reaching the confidence threshold, as shown in Fig. 4 (c).

Position-aware LPS Router. Unlike many existing decoding methods [40, 56, 53, 35] that rely on top-1 confidence $\kappa ^ { i }$ as a static commit criterion, LPS learns this decision, using confidence as the primary input and augmenting it with two signals. First, to make the LPS router position-aware, we incorporate the token’s normalized intra-block position embedding, $\mathrm { p o s } ^ { i } = ( i \dot { - } b ) \big / | B | \in [ 0 , 1 ]$ where b is the left boundary index and |B| is the block size. This encoding captures the token’s relative location within the block and enables the router to correlate decisions across positions rather than scoring them independently. To support variable-length block inputs, our LPS router adopts a two-layer Transformer architecture to leverage positional information, rather than an MLP-based design [4] that cannot handle variable-length inputs. Model details are provided in Appendix D.1. Second, to enable the LPS router to distinguish correct-but-underconfident tokens from genuinely uncertain ones, we incorporate three auxiliary uncertainty features: (1) the normalized entropy $\tilde { H } _ { t } ^ { i }$ for absolute prediction uncertainty, which alone cannot tell whether the top-1 candidate dominates; (2) the confidence gap $\Delta _ { t } ^ { i }$ between the top-1 and top-2 predictions for relative label ambiguity, i.e., the margin over the strongest competitor; and (3) the block-wise mask ratio $\rho _ { t }$ to capture global decoding progress, as the commit policy should differ across denoising stages. Overall, for each masked position $i \in \mathcal { M }$ in the active block $B ,$ , the input to the LPS router is defined as the concatenation of all features $f _ { t } ^ { i } = ( \kappa _ { t } ^ { i } , \tilde { H } _ { t } ^ { i } , \Delta _ { t } ^ { i } , \rho _ { t } )$ , and the output is $O _ { \psi } ^ { i } = \mathcal { R } _ { \psi } ( f _ { t } ^ { i } , \mathrm { p o s } ^ { i } )$ , from which the commit probability is obtained as $p _ { \psi } ^ { i } = \sigma ( O _ { \psi } ^ { i } )$

Self-supervised LPS Router Training. We train LPS using a generate-then-replay pipeline that requires no human label. We first run the base dLLM under its default decoding schedule and record the resulting completed sequence $x _ { 0 }$ . We treat $x _ { 0 }$ as a convergence target rather than absolute ground truth, so the LPS router learns to predict when a token has stabilized to the model’s own final prediction. During replay, at each intermediate step t, we recompute the features $f _ { t } ^ { i }$ and assign $y _ { t } ^ { i } = \mathbf { \bar { 1 } } [ \hat { x } _ { t } ^ { i } = x _ { 0 } ^ { i } ]$ , indicating whether the current top-1 prediction already matches $x _ { 0 }$ . To keep the replay aligned with the original generation trajectory, we apply an oracle rule that unmasks positions where the current prediction matches $x _ { 0 }$ , and replaces mismatched positions with their corresponding tokens from $x _ { 0 }$ . Since prematurely committing an incorrect token is irreversible, we use a weighted binary cross-entropy loss that penalizes false positives more heavily than false negatives during training:

$$
\mathcal { L } _ { \mathrm { L P S } } = \frac { 1 } { \left| \mathcal { M } _ { t } \right| } \sum _ { i \in \mathcal { M } _ { t } } w _ { t } ^ { i } \cdot \mathrm { B C E } \Big ( \sigma \big ( \mathcal { R } _ { \psi } ( f _ { t } ^ { i } , \mathrm { p o s } _ { t } ^ { i } ) \big ) , y _ { t } ^ { i } \Big ) ,\tag{4}
$$

where σ indicates a sigmoid function, negative samples (incorrect early commitments) receive weights $w _ { t } ^ { i } > 1$ , causing false positives to dominate the gradient. The training procedure is briefly summarized in Alg. 1, with full trace-generation and training details provided in Appendix D.2 and D.3.

LPS Router Inference. As shown in Alg. 2, at each decoding step, the base dLLM produces top-1 predictions and confidences for all masked positions. If no active block exists, LBS selects a new

Table 1: Comparison of EB-Decode and baseline methods across four benchmarks and three base models. We report tokens per second per GPU (TPS), speedup (Sp.up), and accuracy (Acc.%). Speedup is measured relative to the vanilla decoding method. The best throughput and speedup are highlighted in bold. “—” indicates not applicable, as Learn2PD provides no open-source implementation for LLaDA-1.5.
<table><tr><td rowspan="2">Method</td><td colspan="3">LLaDA-8B-Instruct</td><td colspan="3">Dream-v0-Instruct-7B</td><td colspan="3">LLaDA-1.5</td></tr><tr><td>Sp.up</td><td>TPS</td><td>Acc.(%)</td><td>Sp.up</td><td>TPS</td><td>Acc. (%) |</td><td>|Sp.up</td><td>TPS</td><td>Acc.(%)</td></tr><tr><td colspan="10">HumanEval</td></tr><tr><td>Vanilla</td><td>1.00×</td><td>23.70</td><td>43.90</td><td>1.00×</td><td>11.33</td><td>54.88</td><td>1.00×</td><td>23.79</td><td>43.90</td></tr><tr><td>Fast-dLLM [53]</td><td>2.66×</td><td>63.08</td><td>43.90</td><td>4.29×</td><td>48.57</td><td>53.66</td><td>2.72×</td><td>64.60</td><td>42.68</td></tr><tr><td>AdaBlock-dLLM [35]</td><td>2.38×</td><td>56.33</td><td>42.68</td><td>4.05×</td><td>45.88</td><td>52.44</td><td>2.52×</td><td>59.95</td><td>41.46</td></tr><tr><td>Learn2PD [4]</td><td>2.78×</td><td>65.92</td><td>42.68</td><td>2.15×</td><td>24.31</td><td>53.66</td><td></td><td></td><td></td></tr><tr><td>EB-Decode (LBS)</td><td>2.90×</td><td>68.83</td><td>45.73</td><td>4.46×</td><td>50.55</td><td>55.49</td><td>2.68×</td><td>63.73</td><td>41.46</td></tr><tr><td>EB-Decode (LBS+LPS)</td><td>3.53×</td><td>83.69</td><td>43.29</td><td>5.61×</td><td>63.53</td><td>54.88</td><td>3.58×</td><td>85.19</td><td>42.68</td></tr><tr><td colspan="10">MBPP</td></tr><tr><td>Vanilla</td><td>1.00×</td><td>11.95</td><td>40.00</td><td>1.00×</td><td>2.23</td><td>53.40</td><td>1.00×</td><td>8.28</td><td>40.80</td></tr><tr><td>Fast-dLLM [53]</td><td>4.13×</td><td>49.31</td><td>40.20</td><td>11.88×</td><td>26.49</td><td>53.00</td><td>6.41×</td><td>53.13</td><td>40.20</td></tr><tr><td>AdaBlock-dLLM [35]</td><td>4.02×</td><td>48.02</td><td>41.20</td><td>11.54×</td><td>25.74</td><td>52.40</td><td>6.01×</td><td>49.82</td><td>40.60</td></tr><tr><td>Learn2PD [4]</td><td>4.72×</td><td>56.37</td><td>40.00</td><td>4.04×</td><td>9.01</td><td>52.00</td><td></td><td></td><td></td></tr><tr><td>EB-Decode (LBS)</td><td>4.47×</td><td>53.37</td><td>40.40</td><td>14.70×</td><td>32.77</td><td>54.80</td><td>6.39×</td><td>52.94</td><td>40.60</td></tr><tr><td>EB-Decode (LBS+LPS) 5.37×</td><td></td><td>64.18</td><td>40.20</td><td>18.76×</td><td>41.83</td><td>53.60</td><td>7.76×</td><td>64.26</td><td>39.80</td></tr><tr><td colspan="10">GSM8K-CoT</td></tr><tr><td>Vanilla</td><td>1.00×</td><td>17.04</td><td>79.07</td><td>1.00×</td><td>12.03</td><td>77.78</td><td>1.00×</td><td>16.94</td><td>79.76</td></tr><tr><td>Fast-dLLM [53]</td><td>4.65×</td><td>79.20</td><td>78.70</td><td>4.51×</td><td>54.20</td><td>77.86</td><td>5.15×</td><td>87.27</td><td>79.45</td></tr><tr><td>AdaBlock-dLLM [35]</td><td>4.18×</td><td>71.22</td><td>79.00</td><td>4.57×</td><td>54.97</td><td>78.70</td><td>4.89×</td><td>82.87</td><td>79.00</td></tr><tr><td>Learn2PD [4]</td><td>4.91×</td><td>83.64</td><td>78.24</td><td>2.07×</td><td>24.96</td><td>78.09</td><td></td><td></td><td></td></tr><tr><td>EB-Decode (LBS)</td><td>5.09×</td><td>86.80</td><td>79.68</td><td>4.92×</td><td>59.22</td><td>78.10</td><td>4.79×</td><td>81.20</td><td>80.36</td></tr><tr><td>EB-Decode (LBS+LPS)</td><td>6.14×</td><td>104.55</td><td>78.24</td><td>6.36×</td><td>76.47</td><td>77.33</td><td>6.25×</td><td>105.91</td><td>79.53</td></tr><tr><td colspan="10">MATH500-CoT</td></tr><tr><td>Vanilla</td><td></td><td></td><td></td><td></td><td>24.97</td><td></td><td></td><td></td><td></td></tr><tr><td>Fast-dLLM [53]</td><td>1.00× 3.26×</td><td>21.72 70.77</td><td>42.40</td><td>1.00×</td><td>77.39</td><td>51.00 51.40</td><td>1.00× 5.00×</td><td>15.10 75.59</td><td>43.20 42.80</td></tr><tr><td>AdaBlock-dLLM [35]</td><td>3.07×</td><td>66.63</td><td>42.00 41.60</td><td>3.10× 3.08×</td><td>77.00</td><td>48.80</td><td>4.69×</td><td>70.88</td><td>42.20</td></tr><tr><td>Learn2PD [4]</td><td>3.78×</td><td>82.11</td><td>41.80</td><td>1.12×</td><td>28.05</td><td>49.80</td><td></td><td></td><td></td></tr><tr><td>EB-Decode (LBS)</td><td>3.32×</td><td>72.03</td><td>41.20</td><td>2.96×</td><td>73.91</td><td>49.80</td><td>4.92×</td><td>74.24</td><td>42.60</td></tr><tr><td>EB-Decode (LBS+LPS) 3.86×</td><td></td><td>83.79</td><td>42.40</td><td>3.57×</td><td>89.06</td><td>49.40</td><td>5.55×</td><td>83.79</td><td>42.40</td></tr></table>

B; otherwise, the previous B is reused. Within B, a position i is committed if its base confidence exceeds the threshold τ or its LPS score exceeds a threshold $\tau _ { \psi }$ tuned on a held-out validation split. In this way, LPS preserves the standard threshold rule for tokens the model is already confident about, while committing correct-but-underconfident tokens that would otherwise be deferred. If no position in B satisfies either condition, we fall back to committing the highest-confidence masked position to prevent decoding from stalling, following standard practice in confidence-based decoding. Once B is fully resolved, decoding proceeds to the next block. Full pseudocode is provided in Appendix E.

## 6 Experiments

## 6.1 Experimental Setup

Models, Benchmarks, and Metrics. Models. We evaluate EB-Decode on three open-source dLLMs: LLaDA-8B-Instruct [40], Dream-v0-Instruct-7B [56], and LLaDA-1.5 [63]. Benchmarks. Our evaluation covers mathematical reasoning on GSM8K [11] (0-shot) and MATH500 [31] (0-shot), as well as code generation on HumanEval [6] (0-shot) and MBPP [3] (3-shot). For GSM8K and MATH500, we append a chain-of-thought prompt suffix [50] to each question following [9]. Metrics. We report task accuracy, throughput, and the associated speedups compared to the base models. Detailed experimental settings are provided in Appendix F.

Table 2: Ablation of LBS components on LLaDA-8B-Instruct and GSM8K-CoT. Each row progressively adds one component.
<table><tr><td>Variant</td><td>|Acc.(%)</td><td>TPS</td></tr><tr><td>LBS Router</td><td>74.53</td><td>42.25</td></tr><tr><td>+ EOS removal</td><td>77.33</td><td>90.31</td></tr><tr><td>+ Max-gap constraint</td><td>78.70</td><td>86.92</td></tr><tr><td>+ Min-tokens constraint</td><td>79.68</td><td>86.80</td></tr></table>

Table 3: Ablation of LPS components on LLaDA-8B-Instruct and GSM8K-CoT. Each row progressively adds one component.
<table><tr><td>Variant</td><td>Acc.(%)</td><td>TPS</td></tr><tr><td>LPS Router in MLP</td><td>78.39</td><td>87.40</td></tr><tr><td>+ MLP → Transformer</td><td>78.25</td><td>97.31</td></tr><tr><td>+ Intra-block position</td><td>77.94</td><td>104.18</td></tr><tr><td>+ Entropy + Ġap + Mask Ratio</td><td>78.24</td><td>104.55</td></tr></table>

Baselines. We compare EB-Decode against four inference-stage baselines: (1) the vanilla block-wise decoding method used by each base dLLM [40, 56, 63], which serves as the no-acceleration reference; (2) Fast-dLLM [53], which commits multiple tokens per step based on a confidence threshold (for a fair comparison, we use its parallel decoding without KV cache, and report its results with KV cache in Tab. 6); (3) AdaBlock-dLLM [35], which adaptively places block boundaries using delimiter-token confidence; and (4) Learn2PD [4], which employs a lightweight filter for parallel decoding. A detailed comparison between EB-Decode and these and related methods is given in Appendix G.

Router Training Details. Both routers are lightweight, with about 0.6M trainable parameters for LBS and 0.03M for LPS. The LBS router is trained on dLLM-generated responses from 92,000 prompts randomly sampled from GSM8K [11], the PRM12K training set [31], and a subset of the Numina-Math dataset [27], following the setup of [9]. The router is trained for 6 epochs using AdamW with a learning rate of 5e−5. The actual training of the LBS router takes only about 2 hours. The LPS router is trained on 10,000 prompts randomly sampled from AQUA-RAT [32]. Training uses AdamW with a learning rate of 1e−3 and is early-stopped when the validation recall does not improve for 100 epochs. Trace generation takes about 2 hours, while router training itself requires only around 10 minutes. Further details are provided in Appendix C and D.

## 6.2 Comparison of EB-Decode with SOTA Baselines

Tab. 1 summarizes the comparison between our proposed EB-Decode and four baselines in terms of accuracy and throughput across three dLLMs and four benchmarks. We observe that both EB-Decode with LBS only and the full EB-Decode with LBS+LPS consistently achieve better accuracy-efficiency trade-offs. EB-Decode with LBS alone already improves throughput across all settings, while even improving accuracy in several cases. For example, on LLaDA-8B-Instruct with HumanEval, LBS increases throughput from 23.70 to 68.83 TPS while simultaneously improving accuracy from 43.90% to 45.73%. This result suggests that adaptive block boundaries not only unlock greater parallelism, but also provide difficult tokens with richer right-context information during decoding. The full EB-Decode improves throughput by 3.53–18.76× over the vanilla decoder, while maintaining comparable accuracy (around ±1%). EB-Decode also outperforms the strongest baseline in every setting, delivering up to 1.58× higher throughput than Fast-dLLM while maintaining comparable or even higher accuracy. These results demonstrate the effectiveness of our EB-Decode.

## 6.3 Ablation Studies of EB-Decode

LBS Design Choices. Tab. 2 additively builds up the LBS filter described in Sec. 5.1. Starting from the LBS router’s selected block, we incrementally add EOS removal, the max-gap constraint, and the min-tokens constraint to further refine the selected block. The results in the table empirically demonstrate that all filters are crucial in maximizing the task accuracy while maintaining high TPS. To check whether these gains stem from the filters alone, Tab. 4 removes the LBS router while keeping the same filters. Even with all three filters, this variant reaches only 76.80% accuracy at 23.96 TPS, 2.88 points

Table 4: Accuracy and TPS on LLaDA-8B-Instruct and GSM8K-CoT with the three filters applied progressively without the LBS router, where blocks are selected randomly. Even with all three filters, both accuracy and TPS remain below those of LBS.

<table><tr><td>Variant</td><td>|Acc.(%)</td><td>TPS</td></tr><tr><td>w/o LBS router</td><td>49.13</td><td>11.56</td></tr><tr><td>+ EOS removal</td><td>68.76</td><td>24.24</td></tr><tr><td>+ Max-gap constraint + Min-tokens constraint</td><td>76.27</td><td>23.16</td></tr><tr><td></td><td>76.80</td><td>23.96</td></tr><tr><td>LBS router + all filters (ours)</td><td>79.68</td><td>86.80</td></tr></table>

lower and 3.62× slower than LBS with the same filters, showing that the learned block selection itself brings large gains. We also train an LBS router of comparable size on LLaDA-8B-Instruct that takes only normalized entropy as input. Its converged training loss is higher, and accuracy across the four benchmarks drops by 1.2 points on average at similar TPS. This suggests that semantic information can indeed help the LBS router pick a more optimal block.

LPS Design Choices. Tab. 3 additively builds up the LPS router described in Sec. 5.2, on top of LBS selected blocks. Starting from an MLP taking only confidence as input, we replace the encoder with a small Transformer, add the intra-block position embedding, and add the three auxiliary uncertainty features (entropy, top-1/top-2 gap, mask ratio). The results in the table empirically demonstrate the benefit of using a Transformer architecture over an MLP, and show that all components contribute to maximizing the TPS while preserving the task accuracy.

Performance on Different Generation Lengths. Tab. 5 compares EB-Decode against the baseline at generation lengths $\bar { L } ~ \in ~ \{ 2 5 6 , 5 1 2 \}$ EB-Decode achieves the highest throughput and comparable accuracy at both lengths. Also, the speedup grows as the generation length increases, indicating that EB-Decode may be especially effective for long-sequence generation.

Effect of KV-Cache Configuration. Tab. 6 shows the result of using different KV-cache strategies on top of EB-Decode on 4×A100. Since LBS uses non-contiguous active blocks, we apply cache at the block boundaries: pre-

Table 5: Accuracy and speedup on LLaDA-8B-Instruct and GSM8K-CoT at different generation lengths. EB-Decode delivers higher speedup at both lengths with accuracy comparable to the vanilla decoder.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=2 colspan=1>L=256Acc.(%) Sp.up</td><td rowspan=1 colspan=1>L=512</td></tr><tr><td rowspan=1 colspan=1>Acc.(%) Sp.up</td></tr><tr><td rowspan=1 colspan=1>Vanilla</td><td rowspan=1 colspan=1>75.13  1.00×</td><td rowspan=1 colspan=1>79.07  1.00×</td></tr><tr><td rowspan=1 colspan=1>LBS</td><td rowspan=1 colspan=1>74.98  3.27×</td><td rowspan=1 colspan=1>79.68 5.09×</td></tr><tr><td rowspan=1 colspan=1>LBS+LPS</td><td rowspan=1 colspan=1>74.60  3.86×</td><td rowspan=1 colspan=1>78.24  6.14×</td></tr></table>

fix cache [53] reuses keys/values for tokens to the left of the leftmost selected position, and dual cache [53] additionally reuses keys/values for tokens to the right of the rightmost selected position. Both variants further improve the throughput of EB-Decode while preserving accuracy, demonstrating that EB-Decode can integrate with standard KV-cache pipelines for additional speedup. With KV caching being considered on top of both methods, EB-Decode still outperforms Fast-dLLM by 1.28×, 1.22×, and 1.38× without cache, with the prefix cache, and with the dual cache, respectively, showing that its advantage holds even when the baseline also uses KV caching.

Overhead of LBS and LPS Routers. We measure the wall-clock overhead of the two routers as a fraction of the per-step backbone forward time on A100. LBS accounts for roughly 3.80% and LPS for 2.41% of the per-step latency (6.21% in total), confirming that the speedups come from jointly decoding low-entropy tokens earlier before they reach the confidence threshold and thereby using fewer decoding steps, while the routers themselves introduce only negligible inference overhead.

Table 6: Accuracy and TPS on LLaDA-8B-Instruct and GSM8K-CoT across different cache strategies, for both EB-Decode and Fast-dLLM.
<table><tr><td>Method (L = 512)</td><td>Acc.(%) TPS</td></tr><tr><td>Fast-dLLM [53] 78.70 Fast-dLLM (Prefix Cache) 78.92 Fast-dLLM (Dual Cache) 78.70</td><td>50.90 62.03 56.92</td></tr><tr><td>EB-Decode EB-Decode (Prefix Cache) EB-Decode (Dual Cache) 77.93</td><td>78.24 65.27 77.93 75.95</td></tr></table>

## 6.4 Generality and Broader Comparisons

Router Transferability. The routers of EB-Decode are largely transferable, so they can be reused without retraining when the deployment setting changes. We examine this transferability from three aspects: across models, across generation lengths, and across tasks. (1) Across models as shown in Tab. 7 (a), we consider two transfer targets. For LLaDA-1.5, which belongs to the same model series as LLaDA-8B-Instruct, transferring LBS alone or together with LPS matches the natively trained routers in accuracy and throughput. For Dream-v0-Instruct-7B, which comes from a different model series, LBS does not transfer, because it embeds predicted token IDs with the base model’s embedding table and vocabularies differ across series. LPS, whose features are vocabulary-independent, still transfers. (2) Across generation lengths as shown in Tab. 7 (b), an LPS router trained at one length remains on par at the other. Only LPS is evaluated here, because LBS takes the entire masked sequence as input and uses positional embeddings tied to the generation length (Appendix C.1). We therefore recommend using LBS at its training length. (3) Across tasks, both routers are trained only on mathematical data, so the HumanEval and MBPP results in Tab. 1 already reflect transfer to unseen code tasks.

Table 7: Router transfer without retraining on GSM8K-CoT, (a) across models and (b) across generation lengths. “Native” uses routers trained in the target setting, and “Transferred” uses the router(s) listed in the second column.
<table><tr><td rowspan="2">Target</td><td rowspan="2">Transferred router</td><td colspan="2">Native</td><td colspan="2">Transferred</td></tr><tr><td>|Acc.(%)</td><td>TPS</td><td>|Acc.(%)</td><td>TPS</td></tr><tr><td colspan="6">(a) Across models</td></tr><tr><td>LLaDA-1.5</td><td>LBS from LLaDA-8B</td><td>79.23</td><td>64.95</td><td>79.15</td><td>64.68</td></tr><tr><td>LLaDA-1.5</td><td>LBS + LPS from LLaDA-8B</td><td>79.23</td><td>64.95</td><td>79.23</td><td>66.37</td></tr><tr><td>Dream-v0-Instruct-7B</td><td>LPS from LLaDA-8B</td><td>76.19</td><td>43.55</td><td>76.49</td><td>43.07</td></tr><tr><td colspan="6">(b) Across generation lengths (LLaDA-8B-Instruct)</td></tr><tr><td>L=512</td><td>LPS trained at L=256</td><td>78.17</td><td>69.54</td><td>78.24</td><td>66.95</td></tr><tr><td>L=256</td><td>LPS trained at L=512</td><td>74.68</td><td>95.37</td><td>74.15</td><td>94.08</td></tr></table>

Table 8: Accuracy and throughput of LBS on block-forward dLLMs, which compute logits only for the current active window. “Thr.” denotes the commit threshold; accuracy gains over the baseline are shown in parentheses.
<table><tr><td></td><td></td><td>HumanEval (0-shot)</td><td></td><td>GSM8K (5-shot)</td></tr><tr><td>Method</td><td>Thr.</td><td>Acc.(%) Token/s</td><td>Acc.(%)</td><td>Token/s</td></tr><tr><td colspan="5">LLaDA 2.0 [5], active window 64</td></tr><tr><td>LLaDA 2.0</td><td>0.8</td><td>75.6</td><td>12.9 91.1</td><td>11.7</td></tr><tr><td>EB-Decode (LBS)</td><td>0.8</td><td>77.4 (+1.8)</td><td>12.0 91.6 (+0.5)</td><td>11.1</td></tr><tr><td>LLaDA 2.0</td><td>0.95</td><td>79.9</td><td>9.4 91.8</td><td>8.6</td></tr><tr><td>EB-Decode (LBS) 0.95</td><td></td><td>82.3 (+2.4)</td><td>8.8 92.3 (+0.5)</td><td>8.2</td></tr><tr><td colspan="5">SDAR-1.7B-Chat-b32 [10], active window 32</td></tr><tr><td>SDAR</td><td>0.8</td><td>42.1</td><td>38.4</td><td>68.5 39.3</td></tr><tr><td>EB-Decode (LBS) 0.8</td><td></td><td>43.9 (+1.8)</td><td>33.8 69.5 (+1.0)</td><td>35.8</td></tr><tr><td>SDAR</td><td>0.9</td><td>46.3</td><td>34.2</td><td>73.1 32.8</td></tr><tr><td>EB-Decode (LBS)</td><td>0.9</td><td>47.6 (+1.3)</td><td>30.2</td><td>73.6 (+0.5) 29.9</td></tr></table>

Compatibility with Block-Forward dLLMs. Unlike the bidirectional dLLMs in Tab. 1, which forward the full sequence at every step, block-forward dLLMs such as LLaDA 2.0 [5] and SDAR [10] adopt block-causal attention and compute logits only for the current active window. This does not prevent EB-Decode from applying. LBS selects the next easy-to-decode block, so its selections concentrate near the decoding frontier, which the active window already covers. Adapting to blockcausal attention then requires a single parameter change, tightening $g _ { \mathrm { m a x } }$ from 5 to 1 so that each block is filled contiguously. Restricted to the window logits, LBS improves accuracy in all eight configurations of Tab. 8, at both commit thresholds tested. Throughput stays close to the baseline, since LBS reuses logits the engine already computes and adds no extra forward. LPS needs no adaptation, since its input covers only the tokens inside the selected block.

Comparison with Autoregressive Models. To examine whether EB-Decode narrows the efficiency gap between dLLMs and autoregressive (AR) models, Tab. 9 compares it with Qwen3-8B [55] and LLaMA3-8B-Instruct [16], with all models run on the same transformers backend with batch size 1 on 4×A100 GPUs. The low Acc. of the AR models reflects a format mismatch with the standard answer-extraction rule of the GSM8K harness, while Flex. Acc. extracts the answer from \boxed{} or the last number in the response and is thus robust to the output format. While vanilla dLLM decoding is less than half as fast as either AR model, EB-Decode runs about 2.4× faster than the faster of the two, and the remaining accuracy gap to Qwen3-8B in Flex. Acc. stems mostly from the base model rather than the decoding method. Beyond this matched setting, AR models benefit from recent serving systems such as vLLM [26], where Qwen3-8B and LLaMA3-8B-Instruct reach

Table 9: Comparison with AR models on GSM8K-CoT (L=512). Acc. follows the standard answer extraction of the GSM8K harness, while Flex. Acc. takes the content of \boxed{} or the last number in the response and is thus robust to the output format.
<table><tr><td>Model</td><td>Type</td><td>TPS</td><td>Acc.(%)</td><td>Flex. Acc. (%)</td></tr><tr><td>Qwen3-8B [55]</td><td>AR</td><td>21.22</td><td>32.22</td><td>91.89</td></tr><tr><td>LLaMA3-8B-Instruct [16]</td><td>AR</td><td>28.94</td><td>68.54</td><td>80.36</td></tr><tr><td>LLaDA-8B-Instruct, Vanilla [40]</td><td>dLLM</td><td>10.24</td><td>79.07</td><td>84.08</td></tr><tr><td>LLaDA-8B-Instruct, EB-Decode</td><td>dLLM</td><td>69.54</td><td>78.17</td><td>82.49</td></tr></table>

5689.70 and 4965.19 TPS through paged KV-cache memory management, continuous batching, and optimized attention kernels [12], whereas serving frameworks for dLLMs such as dInfer [39] are still in early development.

## 7 Conclusion

Diffusion large language models promise parallel decoding but still suffer from inefficient inference under fixed block sizes and fixed confidence thresholds. To understand the source of this inefficiency, we identify two empirical phenomena: a staircase entropy pattern across token positions and an early convergence phenomenon across denoising steps. Building on these observations, we propose EB-Decode, a plug-in framework that replaces the fixed block size and the fixed confidence threshold with two lightweight learnable routers, LBS and LPS, while keeping the base dLLM frozen. Experiments on three open-source dLLMs and four benchmarks show that EB-Decode delivers up to 18.76× throughput improvement over the vanilla decoder with comparable accuracy, offering a practical step toward narrowing the efficiency gap between dLLMs and autoregressive models.

## References

[1] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

[2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

[3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[4] Wenrui Bao, Zhiben Chen, Dan Xu, and Yuzhang Shang. Learning to parallel: Accelerating diffusion large language models via learnable parallel decoding. arXiv preprint arXiv:2509.25188, 2025.

[5] Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Ling Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, and Yihong Zhuang. Llada2.0: Scaling up diffusion language models to 100b, 2025. URL https://arxiv.org/abs/2512.15745.

[6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[7] Xiaohan Chen, Yu Cheng, Shuohang Wang, Zhe Gan, Zhangyang Wang, and Jingjing Liu. Earlybert: Efficient bert training via early-bird lottery tickets. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2195–2207, 2021.

[8] Xinhua Chen, Sitao Huang, Cong Guo, Chiyue Wei, Yintao He, Jianyi Zhang, Hai Li, Yiran Chen, et al. Dpad: Efficient diffusion language models with suffix dropout. arXiv preprint arXiv:2508.14148, 2025.

[9] Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. dparallel: Learnable parallel decoding for dllms. arXiv preprint arXiv:2509.26488, 2025.

[10] Shuang Cheng, Yihan Bian, Dawei Liu, Linfeng Zhang, Qian Yao, Zhongbo Tian, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, et al. Sdar: A synergistic diffusion-autoregression paradigm for scalable sequence generation. arXiv preprint arXiv:2510.06303, 2025.

[11] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[12] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

[13] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation through time. arXiv preprint arXiv:2410.21035, 2024.

[14] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018.

[15] Google DeepMind. Gemini diffusion. https://deepmind.google/models/ gemini-diffusion/, 2025.

[16] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[17] Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.

[18] Naibin Gu, Peng Fu, Xiyu Liu, Bowen Shen, Zheng Lin, and Weiping Wang. Light-PEFT: Lightening parameter-efficient fine-tuning via early pruning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 7528–7541, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.447. URL https://aclanthology.org/ 2024.findings-acl.447/.

[19] Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. arXiv preprint arXiv:2410.08709, 2024.

[20] Feng Hong, Geng Yu, Yushi Ye, Haicheng Huang, Huangjie Zheng, Ya Zhang, Yanfeng Wang, and Jiangchao Yao. Wide-in, narrow-out: Revokable decoding for efficient and effective dllms. arXiv preprint arXiv:2507.18578, 2025.

[21] Zhanqiu Hu, Jian Meng, Yash Akhauri, Mohamed S Abdelfattah, Jae-sun Seo, Zhiru Zhang, and Udit Gupta. Flashdlm: Accelerating diffusion language model inference via efficient kv caching and guided diffusion. arXiv preprint arXiv:2505.21467, 2025.

[22] Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, and Tong Xiao. Pc-sampler: Position-aware calibration of decoding bias in masked diffusion models. arXiv preprint arXiv:2508.13021, 2025.

[23] Daniel Israel, Guy Van den Broeck, and Aditya Grover. Accelerating diffusion llms via adaptive parallel decoding. arXiv preprint arXiv:2506.00413, 2025.

[24] Wonjun Kang, Kevin Galim, Seunghyuk Oh, Minjae Lee, Yuchen Zeng, Shuibai Zhang, Coleman Hooper, Yuezhou Hu, Hyung Il Koo, Nam Ik Cho, et al. Parallelbench: Understanding the trade-offs of parallel decoding in diffusion llms. arXiv preprint arXiv:2510.04767, 2025.

[25] Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, et al. Mercury: Ultra-fast language models based on diffusion. arXiv e-prints, pages arXiv–2506, 2025.

[26] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

[27] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. https://github.com/project-numina/aimo-progress-prize/blob/main/ report/numina\_dataset.pdf, 2024.

[28] Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Soroush Vosoughi, and Shiwei Liu. Diffusion language models know the answer before decoding. arXiv preprint arXiv:2508.19982, 2025.

[29] Pengxiang Li, Dilxat Muhtar, Tianlong Chen, Lu Yin, and Shiwei Liu. Why diffusion language models struggle with truly parallel (non-autoregressive) decoding? arXiv preprint arXiv:2602.23225, 2026.

[30] Yihao Liang, Ze Wang, Hao Chen, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Emad Barsoum, Zicheng Liu, and Niraj K Jha. Cd4lm: Consistency distillation and adaptive decoding for diffusion language models. arXiv preprint arXiv:2601.02236, 2026.

[31] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

[32] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1: Long papers), pages 158–167, 2017.

[33] Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025.

[34] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

[35] Guanxi Lu, Hao Mark Chen, Yuto Karashima, Zhican Wang, Daichi Fujiki, and Hongxiang Fan. Adablock-dllm: Semantic-aware diffusion llm inference via adaptive block size. arXiv preprint arXiv:2509.26432, 2025.

[36] Lizhuo Luo, Shenggui Li, Yonggang Wen, and Tianwei Zhang. Dsb: Dynamic sliding block scheduling for diffusion llms. arXiv preprint arXiv:2602.05992, 2026.

[37] Omer Luxembourg, Haim Permuter, and Eliya Nachmani. Plan for speed–dilated scheduling for masked diffusion language models. arXiv preprint arXiv:2506.19037, 2025.

[38] Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025.

[39] Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, Xinyuan Zhang, Zhen Tao, Haibo Feng, Ziyun Jiang, Ying Xu, Zenan Huang, Yihong Zhuang, Haokai Xu, Jiaqi Hu, Zhenzhong Lan, Junbo Zhao, Jianguo Li, and Da Zheng. dinfer: An efficient inference framework for diffusion language models. arXiv preprint arXiv:2510.08666, 2025.

[40] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

[41] Han Peng, Peiyu Liu, Zican Dong, Daixuan Cheng, Junyi Li, Yiru Tang, Shuo Wang, and Wayne Xin Zhao. How efficient are diffusion language models? a critical examination of efficiency evaluation practices. arXiv preprint arXiv:2510.18480, 2025.

[42] Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

[43] Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. Advances in Neural Information Processing Systems, 35:17456–17472, 2022.

[44] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024.

[45] Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, et al. Seed diffusion: A large-scale diffusion language model with high-speed inference. arXiv preprint arXiv:2508.02193, 2025.

[46] Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. Branchynet: Fast inference via early exiting from deep neural networks. In 2016 23rd international conference on pattern recognition (ICPR), pages 2464–2469. IEEE, 2016.

[47] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. arXiv preprint arXiv:2503.00307, 2025.

[48] Wen Wang, Bozhen Fang, Chenchen Jing, Yongliang Shen, Yangyi Shen, Qiuyu Wang, Hao Ouyang, Hao Chen, and Chunhua Shen. Time is a feature: Exploiting temporal dynamics in diffusion language models. arXiv preprint arXiv:2508.09138, 2025.

[49] Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Diffusion llms can do faster-than-ar inference via discrete diffusion forcing. arXiv preprint arXiv:2508.09192, 2025.

[50] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[51] Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, and Linfeng Zhang. Accelerating diffusion large language models with slowfast sampling: The three golden principles. arXiv preprint arXiv:2506.10848, 2025.

[52] Lexington Whalen, Zhenbang Du, Haoran You, Chaojian Li, Sixu Li, and Yingyan Celine Lin. Early-bird diffusion: Investigating and leveraging timestep-aware early-bird tickets in diffusion models for efficient training. arXiv preprint arXiv:2504.09606, 2025.

[53] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.

[54] Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic early exiting for accelerating bert inference. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 2246–2251, 2020.

[55] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang,

Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[56] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

[57] Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G Baraniuk, Zhangyang Wang, and Yingyan Celine Lin. Drawing early-bird tickets: Towards more efficient training of deep networks. arXiv preprint arXiv:1909.11957, 2019.

[58] Haoran You, Zhihan Lu, Zijian Zhou, Yonggan Fu, and Yingyan Lin. Early-bird gcns: Graphnetwork co-optimization towards more efficient gcn training and inference via drawing early-bird lottery tickets. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 8910–8918, 2022.

[59] Tunyu Zhang, Xinxi Zhang, Ligong Han, Haizhou Shi, Xiaoxiao He, Zhuowei Li, Hao Wang, Kai Xu, Akash Srivastava, Vladimir Pavlovic, et al. T3d: Few-step diffusion language models via trajectory self-distillation with direct discriminative optimization. arXiv preprint arXiv:2602.12262, 2026.

[60] Yu Zhang, Xinchen Li, Jialei Zhou, Hongnan Ma, Zhongwei Wan, Yiwei Shi, Duoqian Miao, Qi Zhang, and Longbing Cao. Swordsman: Entropy-driven adaptive block partition for efficient diffusion language models. arXiv preprint arXiv:2602.04399, 2026.

[61] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. In International Conference on Learning Representations, 2025.

[62] Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian McAuley, Ke Xu, and Furu Wei. Bert loses patience: Fast and robust inference with early exit. Advances in Neural Information Processing Systems, 33:18330–18341, 2020.

[63] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025.

[64] Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong Wang, Lanning Wei, et al. Llada-moe: A sparse moe diffusion language model. arXiv preprint arXiv:2509.24389, 2025.

## A Detailed dLLM Backgrounds

This section extends the dLLM literature summary in Sec. 2 of the main text with additional context on representative methods.

D3PM [2] generalizes discrete diffusion using structured transition matrices over the token vocabulary. Subsequent works refine the masked diffusion paradigm. MDLM [42] derives a Rao-Blackwellized objective that reduces to a weighted mixture of masked language modeling losses. MD4 [44] further simplifies and generalizes the continuous-time objective. SEDD [34] proposes a score-entropy objective. Zheng et al. [61] shows that masked diffusion training and sampling are time-agnostic. Together, these advances bring masked dLLMs to likelihood performance comparable to AR models. LLaDA [40] performs on par with LLaMA3 at the billion-parameter scale. Subsequent extensions add variance-reduced preference optimization [63], sparse experts [64], and AR-to-diffusion conversion that scales to 100B parameters [5]. Dream-7B [56] introduces AR-based LLM initialization and context-adaptive, token-level noise rescheduling. Semi-AR methods such as BD3-LMs [1] and SDAR [10] combine the coherence of AR generation with the parallelism of diffusion. They maintain AR dependencies across blocks while sampling tokens in parallel within each block.

## B Block-wise Decoding for dLLMs

This section details the block-wise decoding recipe summarized in Sec. 3, reusing the discretediffusion notation introduced therein, including the vocabulary V, the mask symbol [MASK], the mask predictor $p _ { \theta }$ , the clean response sequence $\mathbf { x } _ { \mathrm { 0 } }$ with its position-i token $x _ { 0 } ^ { i } .$ , and the partially denoised sequence $\mathbf { x } _ { t }$ at noise level $t \in [ 0 , 1 ]$ . We further write $\ v x _ { t } ^ { i }$ for the token at position i in $\mathbf { x } _ { t }$ which is either a vocabulary item in $\nu$ or the mask symbol [MASK].

Inference-time denoising trajectory. At inference time, given a prompt $c ,$ decoding starts from a fully masked response sequence and iteratively applies the reverse process from t = 1 to $t = 0$ over a sequence of denoising steps. The prompt c is concatenated with the response and processed jointly by the bidirectional Transformer [40], so that the predictive distribution at every masked position conditions on both $\mathbf { x } _ { t }$ and $c .$ The final output is the fully unmasked sequence $\mathbf { x } _ { \mathrm { 0 } }$

Block partition and in-block prediction. The response is divided into contiguous blocks $\boldsymbol { B } _ { 1 } , \boldsymbol { B } _ { 2 } , \ldots$ of fixed size B, indexed by $j ,$ and processed from left to right. Within the active block $B _ { j }$ , the model predicts all masked positions in parallel, commits a subset of tokens, and remasks the remaining uncertain ones for further refinement. The per-position top-1 confidence at masked position $i \in B _ { j }$ over candidate vocabulary item $v \in \mathcal V$ follows the same definition as in the main text:

$$
\kappa ^ { i } = \underset { v \in \mathcal { V } } { \operatorname* { m a x } } p _ { \theta } \big ( x _ { 0 } ^ { i } = v ~ \big | ~ \mathbf { x } _ { t } , c \big ) .\tag{5}
$$

Given a confidence threshold τ or an optional per-step commit budget $m ,$ the standard block update rule is

$$
x _ { t } ^ { i }  \{ \begin{array} { l l } { \underset { \boldsymbol { v } \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \boldsymbol { \theta } } \big ( x _ { 0 } ^ { i } = \boldsymbol { v } ~ \big \vert ~ \mathbf { x } _ { t } , c \big ) , } & { \kappa ^ { i } \ge \tau ~ \mathrm { o r } ~ i \in \mathrm { T o p } \mathbf { - } m ( \mathcal { B } _ { j } ) , } \\ {  \mathsf { M A S K }  , } & { \mathrm { o t h e r w i s e } , } \end{array}\tag{6}
$$

where $\mathrm { T o p } { - } m ( B _ { j } )$ denotes the m masked positions in $B _ { j }$ with the highest top-1 confidence at the current step. Once all positions in $B _ { j }$ have been committed, the active block advances to $B _ { j + 1 }$ and the same procedure repeats. Throughout this recipe, the block size $B ,$ , the threshold $\tau ,$ and the budget m remain fixed regardless of how token difficulty varies along the sequence, which is precisely the limitation that motivates our EB-Decode framework. Building on the same block-wise procedure and keeping the base dLLM frozen, EB-Decode replaces the two static design choices with learnable counterparts. Specifically, LBS replaces the fixed block size $B$ with a learnable block selector as described in Sec. 5.1, while LPS replaces the static commit rule with a position-aware, learnable acceptance strategy as described in Sec. 5.2.

## C Detailed Setup for Learnable Block Size

## C.1 Router Architecture

The LBS router $\mathcal { R } _ { \phi }$ is a small Transformer encoder with 600k parameters (not including the frozen token embedding table $W _ { e } )$ that maps per-masked-position features to scalar inclusion logits.

Entropy branch. The normalized entropy sequence for all masked tokens $\tilde { H } = \{ \tilde { H } ^ { i } : i \in \mathcal { M } \}$ is first zero-padded to the generation length L and then processed by a two-layer MLP $F _ { \mathrm { e n t } } { : }$ Linear(1 → $6 4 ) { \overset { \cdot } { \to } } \operatorname { G E L U } \to \operatorname { L i n e a r } ( 6 4 \to 6 4 { \bar { ) } } \to { \frac { 1 } { 2 } }$ LayerNorm, producing a 64-dimensional entropy embedding.

Token branch. The top-1 predicted token IDs $\hat { x } = \{ \hat { x } ^ { i } : i \in \mathcal { M } \}$ are looked up in the frozen base-model token embedding table $W _ { e }$ , zero-padded to $L ,$ and projected to $\mathbb { R } ^ { 6 4 }$ via a learned linear layer $W _ { \mathrm { t o k } } \in \mathbb { R } ^ { 6 4 \times d _ { \mathrm { b a s \epsilon } } }$

Fusion and positional embedding. The two 64-dimensional embeddings are concatenated to form a 128-dimensional vector, then passed through a fusion layer $W _ { \mathrm { c o m b i n e } } \in \mathbb { R } ^ { 1 2 8 \times 1 2 8 }$ . The final input embedding to the Transformer encoder is calculated as:

$$
Z _ { \phi } = W _ { \mathrm { c o m b i n e } } \left[ F _ { \mathrm { e n t } } \Big ( \tilde { H } \Big ) \ ; \ W _ { \mathrm { t o k } } W _ { e } \big ( \hat { x } \big ) \right] + E _ { \mathrm { p o s } } , Z _ { \phi } \in \mathbb { R } ^ { L \times 1 2 8 }\tag{7}
$$

where $E _ { \mathrm { p o s } } \in \mathbb { R } ^ { L \times 1 2 8 }$ is a learnable positional embedding.

Transformer encoder. The sequence $Z _ { \phi }$ is then fed into a 2-layer Transformer encoder with hidden dimension $d _ { \phi } { = } 1 2 8$ , 4 attention heads, FFN width 256, and GELU activations.

Output head. A linear head $\mathbb { R } ^ { 1 2 8 } \to \mathbb { R } ^ { 1 }$ maps the encoder output to logits $O _ { \phi } \in \mathbb { R } ^ { L \times 1 }$ . The padding is removed before returning, yielding an output of shape $( | \mathcal { M } | , 1 )$ . The final inclusion probability at position i is $p _ { \phi } ^ { i } = \sigma ( O _ { \phi } ^ { i } )$

## C.2 Training Procedure

The full per-step training procedure for the LBS router, summarized in the main text and the simplified Alg. 1, is given in Alg. 3. Training is performed on ${ 8 \times \mathrm { A 1 0 0 ~ G P U s } }$

```latex
Algorithm 3 LBS Router Training
Require: dLLM-generated response dataset D, frozen dLLM ${ \mathcal { F } } _ { \theta } .$ router ${ \mathcal { R } } _ { \phi } ,$ , generation length $\overline { { L } } ,$
block length $L _ { B }$ , mask ratio $r \in ( 0 , 1 ] .$ , learning rate $\eta _ { \mathrm { l r } }$
Notation:
$\begin{array} { r } { \tilde { H } ( p ) = ( - \sum _ { v \in \mathcal { V } } p ( v ) } \end{array}$ log p(v))/ log |V|
$\hat { x } ( p ) = \arg \operatorname* { m a x } _ { x } p ( x )$
$\sigma ( p ) = 1 / ( 1 + e ^ { - p } )$
1: for $x \in \mathcal { D }$ do
2: Sample $n \sim \{ 0 , 1 , \dots , L / L _ { B } - 1 \}$
3: x<sub>masked</sub> ← Blockwise-Masking $\mathbf { \langle } x , r , n , L , L _ { B } \mathbf { \rangle } ; \mathcal { M } \gets \{ i : x _ { \mathrm { m a s k e d } } ^ { i } = [ \mathsf { M A S K } ] \}$
4: for each $i \in \mathcal { M }$ do
5: $p _ { . } ^ { i } \gets \mathrm { s o f t m a x } ( \mathcal { F } _ { \theta } ( x _ { \mathrm { m a s k e d } } ) ^ { i } )$
6: $\ell _ { \mathrm { C E } } ^ { i } = { \mathrm { C r o s s E n t r o p y } } ( p ^ { i } , x ^ { i } )$ ▷ per-token CE against ground truth
7: $O _ { \phi } ^ { i }  \mathcal { R } _ { \phi } ( \tilde { H } ( p ^ { i } ) , \hat { x } ( p ^ { i } ) )$
8: end for
9: $\begin{array} { r } { \mathcal { L } _ { \mathrm { w C E } } = \frac { 1 } { | \mathcal { M } | } \sum \sigma ( O _ { \phi } ^ { i } ) \cdot \ell _ { \mathrm { C E } } ^ { i } } \end{array}$
i∈M
10: $\mathcal { L } _ { \mathrm { R e g } } = \textstyle { \frac { 1 } { | \mathcal { M } | } } \sum \sigma ( O _ { \phi } ^ { i } )$
i∈M
11: $\mathcal { L } _ { \mathrm { { L B S } } }  \mathcal { L } _ { \mathrm { { w C E } } } - \mathcal { L } _ { \mathrm { { R e g } } }$
12: $\mathcal { R } _ { \phi }  \mathcal { R } _ { \phi } - \eta _ { \mathrm { l r } } \nabla _ { \phi } \mathcal { L } _ { \mathrm { L B S } }$ ▷ update only router; $\mathcal { F } _ { \theta }$ is frozen
13: end for
14: return $\mathcal { R } _ { \phi }$
```

Training stability. Tab. 10 reports the training and validation losses of the LBS router. The training loss is negative because Eq. (3) subtracts the regularization term from the weighted cross-entropy loss. Both losses drop sharply in the first epoch, after which the changes are much smaller, indicating stable training.

Table 10: Training and validation losses of the LBS router across epochs.
<table><tr><td>Epoch</td><td>Train loss</td><td>Val loss</td></tr><tr><td>0</td><td>0.0569</td><td>0.0764</td></tr><tr><td>1</td><td>-0.337</td><td>-0.291</td></tr><tr><td>3</td><td>-0.364</td><td>-0.378</td></tr><tr><td>6</td><td>-0.359</td><td>-0.396</td></tr></table>

Table 11: Training and validation losses and validation recall of the LPS router across epochs.
<table><tr><td>Epoch</td><td>Train loss</td><td>Val loss</td><td>Val recall</td></tr><tr><td>0</td><td>0.6225</td><td>0.3987</td><td>0.6471</td></tr><tr><td>20</td><td>0.3699</td><td>0.3640</td><td>0.6998</td></tr><tr><td>60</td><td>0.3677</td><td>0.3622</td><td>0.7042</td></tr><tr><td>119</td><td>0.3669</td><td>0.3618</td><td>0.7052</td></tr></table>

## D Detailed Setup for Learnable Parallel Sampling

## D.1 Router Architecture

The LPS router $\mathcal { R } _ { \psi }$ is a small Transformer encoder that operates over all positions of the active block jointly. Taking the per-position input $( f ^ { i } , \mathrm { p o s } ^ { i } )$ defined in Sec. 5.2, the input embedding $Z _ { \psi }$ is a linear projection followed by LayerNorm and GELU, mapping each feature vector into a hidden space of dimension $d _ { \psi } = 3 2$ , so that $Z _ { \psi } \in \mathbb { R } ^ { | \boldsymbol { B } | \times 3 2 }$ . The encoder then processes $Z _ { \psi }$ with 2 pre-norm Transformer blocks with 4 attention heads, a feed-forward width of 64, GELU activations, and dropout of 0.1. A final linear head projects each hidden state to a scalar logit $O _ { \psi } ^ { i }$ , from which the commit probability is $p _ { \psi } ^ { i } = \sigma ( O _ { \psi } ^ { i } )$ , giving a total of roughly 27k trainable parameters. This keeps the router cost negligible compared with the base dLLM forward pass.

## D.2 Trace Generation

To remove any train/test distribution skew, we collect traces under the exact decoding configuration that the router will face at inference. For each experiment we run the corresponding frozen base dLLM with generation length 512, base block size 32, and confidence threshold $\tau { = } 0 . 9 ,$ , matching the inference setting in Sec. 6.1. Since the distribution differs across base models, we train a separate router for each base model. Every record captures, at each position i of the active block, the four feature components $f _ { t } ^ { i } = ( \kappa _ { t } ^ { i } , \tilde { H } _ { t } ^ { i } , \Delta _ { t } ^ { i } , \rho _ { t } )$ , the predicted token $\hat { x } _ { t } ^ { i } .$ the correctness label $y _ { t } ^ { i } = \mathbf { 1 } [ \hat { x } _ { t } ^ { i } = x _ { 0 } ^ { i } ]$ obtained by comparing against the completed sequence $x _ { 0 } ,$ , and the mask indicator $m _ { t } ^ { i }$ which shows which position is still masked. The replay pass follows the oracle progress rule described in Sec. 5.2, so each intermediate state stays consistent with $x _ { 0 }$ and the labels y<sup>i</sup> provide a clean supervision signal without external annotation.

## D.3 Training Procedure

Valid trajectories are split at the trace level into 80% training, 10% validation, and 10% test sets, with the split deterministic under seed 42 so that every router variant sees identical data. The positions outside the active block are masked out of both the loss and the attention computation to prevent boundary artifacts from contaminating the supervision.

Optimization uses AdamW with an initial learning rate of 1e−3, weight decay 0.01, gradient clipping at norm $1 . 0 ,$ and automatic mixed precision. The schedule consists of a linear warmup over 0.5% of the total steps followed by a per-epoch geometric decay with ratio 0.97. We train with a batch size of 512 step records for up to 2000 epochs, with early stopping triggered when the validation recall at the false-positive budget fails to improve for 100 consecutive epochs.

The training objective is the masked weighted binary cross-entropy defined in Eq. (4), where negative samples carry a weight of $w ^ { - } = 2$ . We use asymmetric weights because a false positive commits a wrong token irreversibly, while a false negative only delays acceptance to a later step. Premature acceptance therefore has a higher cost.

```tcl
Algorithm 4 LPS Router Training
Require: Frozen dLLM ${ \overline { { \mathcal { F } _ { \theta } } } } ;$ prompt corpus D; LBS Router $\overline { { \mathcal { R } _ { \phi } } } ;$ LPS Router $\mathcal { R } _ { \psi } ;$ negative weight
$\stackrel { \mathrm { ~ \tiny ~ } } { w } ^ { - } = 2 ;$ target False-Positive rate $\eta \bar { \in } ( 0 , 1 )$ ; learning rate $\eta _ { \mathrm { l r } } \mathrm { ; }$ ; early-stop patience $\bar { P } _ { \mathrm { p a t i e n c e } } .$
Ensure: Router parameters $\mathcal { R } _ { \psi }$ and acceptance threshold $\tau _ { \psi } .$
Stage 1: build the trace dataset $\tau .$
1: $\tau  \emptyset$ $\triangleright \tau$ stores per-position records $( f _ { t } ^ { i } , \mathrm { p o s } ^ { i } , y _ { t } ^ { i } )$
2: for all prompt $q \in \mathcal { D }$ do $\triangleright$ Run LBS decoding on q to obtain $x _ { 0 }$ and the per-step trajectory.
3: for all step t and masked position $i \in \boldsymbol { B }$ at step t do
4: $p _ { t } ^ { i } \gets \dot { \mathrm { s o f t m a x } } \big ( \mathcal { F } _ { \theta } ( \mathbf { x } _ { t } \big ) \big ) _ { i }$ ▷ predictive distribution at position i from $\mathbf { x } _ { t }$
5: $\kappa _ { t } ^ { i } \gets \operatorname* { m a x } _ { v } p _ { t } ^ { i } ( v ) , ~ \hat { x } _ { t } ^ { i } \gets$ arg max<sub>v</sub> $, p _ { t } ^ { i } ( v ) \quad \quad \triangleright$ top-1 confidence and predicted token.
6: $\tilde { H } _ { t } ^ { i } , \Delta _ { t } ^ { i } , \rho _ { t } \gets \mathrm { e n t r o p y } ,$ top-1/top-2 gap, block-wise mask ratio
7: $f _ { t } ^ { i } \gets \big ( \kappa _ { t } ^ { i } , \tilde { H } _ { t } ^ { i } , \Delta _ { t } ^ { i } , \rho _ { t } \big ) , \ \mathrm { p o s } ^ { i } \gets ( i - b ) \big / | \mathcal { B } |$
8: $y _ { t } ^ { \ i }  \mathbf { 1 } [ \bar { \hat { x } } _ { t } ^ { i } = \bar { x } _ { 0 } ^ { i } ]$ ▷ correctness label vs. $x _ { 0 }$
9: Append $( f _ { t } ^ { i } , \mathrm { p o s } ^ { i } , y _ { t } ^ { i } )$ to $\tau$
10: end for
11: end for
Stage 2: LPS router $\mathcal { R } _ { \psi }$ training.
12: repeat
13: for all mini-batch ${ \mathcal { S } } \subset { \mathcal { T } } _ { \mathrm { t r a i n } }$ do
14: $O _ { \psi } ^ { i }  \mathcal { R } _ { \psi } ( f _ { t } ^ { i } , \mathrm { p o s } _ { t } ^ { i } )$ for each record $( f _ { t } ^ { i } , \mathrm { p o s } ^ { i } , y _ { t } ^ { i } ) \in \mathcal { S }$
15: $\mathcal { L } _ { \mathrm { L P S } }  \frac { 1 } { | S | } \sum w _ { t } ^ { i } \mathrm { B C E } ( \sigma ( O _ { \psi } ^ { i } ) , y _ { t } ^ { i } )$ , where $\begin{array} { r } { w _ { t } ^ { i } = \left\{ \begin{array} { l l } { 1 , } & { y _ { t } ^ { i } = 1 } \\ { w ^ { - } , } & { y _ { t } ^ { i } = 0 } \end{array} \right. } \end{array}$ ▷ mini-batch
(i,t)∈S
form of $\mathrm { E q . } \left( 4 \right)$
16: $\mathcal { R } _ { \psi }  \mathcal { R } _ { \psi } - \eta _ { \mathrm { l r } } \nabla \mathcal { L } _ { \mathrm { L P S } }$
17: end for
18: Evaluate validation recall on $\mathcal { T } _ { \mathrm { v a l } }$ under $\mathrm { F P } \le \eta$
19: until validation recall has not improved for $\boldsymbol { P _ { \mathrm { p a t i e n c e } } }$ consecutive epochs
20: get $\mathcal { T } _ { \mathrm { v a l } }$ s.t. $\vert \mathrm { F P } _ { \mathcal { T } _ { \mathrm { v a l } } } ( \tau _ { \psi } ) - \eta \vert ^ { \ast } \le \delta$ ▷ within $\eta \pm 0 . 5$ pp false-positive rate on validation set.
21: return $( \mathcal { R } _ { \psi } , \tau _ { \psi } )$
```

We tune the LPS threshold $\tau _ { \psi }$ on the validation set at the end of each improving epoch by searching over a 250-point grid in [0.8, 0.9999] and keeping the largest recall under a false-positive-rate budget of $\eta = 1 0 ^ { - \dot { 2 } }$ . The full procedure is summarized in $\mathrm { A l g } . { \bar { 4 } }$

Training stability. As shown in Tab. 11, the training and validation losses converge smoothly and the validation recall rises steadily. The validation loss stays close to the training loss throughout, indicating no overfitting.

## E EB-Decode Inference Procedure

The complete inference procedure that combines LBS with LPS is given in Alg. 5. The simplified high-level view is in $\mathrm { A l g } . 2 .$

Algorithm 5 Inference with LBS Block Selection and LPS Parallel Commit   
Require: Prompt $p ,$ frozen dLLM ${ \mathcal { F } } _ { \theta } ;$ LBS router $\mathcal { R } _ { \phi } ;$ LPS router $\mathcal { R } _ { \psi } ;$ generation length L; min   
block size $L _ { \mathrm { m i n } } ;$ fallback block size $\begin{array} { r l } {  { L _ { B } ; } } \end{array}$ max gap $g _ { \mathrm { m a x } } ;$ confidence threshold τ; LPS acceptance   
threshold $\tau _ { \psi }$   
1: $x \gets [ p \ |$ [MASK], . . . , [MASK]]; $B \gets \emptyset$   
L   
2: while ∃ i such that $x ^ { i } = $ [MASK] do   
3: p<sub>t</sub> ← softmax $( \mathcal { F } _ { \theta } ( x ) ) ; \ \bar { \mathcal { M } }  \{ i : x ^ { i } = [ \mathtt { M A S K } ] \}$ ▷ get distribution and masked places   
4: for each $i \in \mathcal { M }$ do   
5: $\kappa ^ { i } \gets$ max $p _ { t } ^ { i } ; \hat { x } ^ { i } \gets$ arg max $p _ { t } ^ { i }$   
6: Compute $\tilde { H } ^ { i } , \Delta ^ { i } , \rho ;$ ▷ compute entropy, top-1/top-2 gap, block-wise mask ratio   
7: $f ^ { i } \gets ( \kappa ^ { i } , \tilde { H } ^ { i } , \Delta ^ { i } , \rho )$ ▷ LPS feature tuple   
8: end for   
9: if $B = \varnothing$ then ▷ LBS: select a new active block   
10: if $| \mathcal { M } | \le L _ { \operatorname* { m i n } }$ then   
11: $B  { } M$ ▷ size fallback   
12: else   
13: $\mathcal { C } \gets \{ i \in \mathcal { M } : \sigma ( \mathcal { R } _ { \phi } ( \tilde { H } ^ { i } , \hat { x } ^ { i } ) ) > 0 . 5 \mathrm { ~ } \land \hat { x } ^ { i } \neq \mathrm { E O S } \}$ ▷ router + EOS removal   
14: $B \gets \dot { \mathrm { N } }$ axGapFilter $\cdot ( \mathcal { C } , g _ { \mathrm { m a x } } )$ ▷ spatial coherence   
15: $\mathbf { i f } \left| B \right| < L _ { \operatorname * { m i n } }$ then   
16: $\dot { B } \gets$ first $L _ { B }$ positions of M ▷ min-tokens fallback   
17: end if   
18: end if   
19: end if   
20: $O _ { \psi } ^ { i }  \mathcal { R } _ { \psi } ( f ^ { i } , ( i - b ) / | B | )$ for $i \in \boldsymbol { B }$ ▷ LPS router, b is the left boundary index of B   
21: ${ \mathcal { A } }  \{ i \in { \mathcal { B } } : x ^ { i } = [ { \tt M A S K I } \land ( \kappa ^ { i } \geq \tau \lor p _ { \psi } ^ { i } > \tau _ { \psi } ) \}$ ▷ union accept rule   
22: if $\overset { \triangledown } { \boldsymbol { A } } = \overset { \vartriangle } { \boldsymbol { \varnothing } }$ then   
23: $\begin{array} { r } { \mathcal { A }  \{ \mathrm { a r g m a x } _ { i \in \mathcal { B } , x ^ { i } = [ \mathsf { M A S K } ] } \kappa ^ { i } \} } \end{array}$ ▷ anti-stall fallback   
24: end if   
25: $x ^ { i }  { \hat { x } } ^ { i } { \mathrm { f o r } } i \in { \mathcal { A } }$ ▷ parallel commit   
26: if $x ^ { i } \neq$ [MASK] for all $i \in \boldsymbol { B }$ then   
27: $B  \emptyset$ ▷ If B resolved, move to Line 9   
28: end if   
29: end while   
30: return x

## F Experimental Setup

Decoding configuration. All methods share the same inference configuration of generation length 512, base block size 32, and base confidence threshold τ=0.9. No method in Tab. 1 uses a KV cache. In the official implementation of Fast-dLLM [53], the KV cache is recomputed with one full-sequence forward pass at the start of each block and reused within that block, whereas the vanilla LLaDA decoding loop maintains no cache.

EB-Decode hyperparameters. For LBS in our EB-Decode, we set $g _ { \mathrm { m a x } } = 5$ for all three dLLMs, and $L _ { \mathrm { m i n } } = 8 , 8$ , 16 for LLaDA-8B-Instruct, Dream-v0-Instruct-7B, LLaDA-1.5 respectively. When not paired with LPS, LBS adopts confidence-based decoding following Fast-dLLM [53]. For LPS, we choose the acceptance threshold $\tau _ { \psi }$ under a false-positive-rate budget of $1 \% \pm 0 . 5$ % points on the validation set, giving $\tau _ { \psi } = 0 . 8 9 , 0 . 9 0$ , and 0.87 for LLaDA-8B-Instruct, Dream-v0-Instruct-7B, LLaDA-1.5 respectively.

Throughput measurement. All methods, including every baseline, are measured with the Hugging-Face transformers backend at batch size 1, following prior dLLM acceleration works [35, 4, 9], and throughput (TPS) is measured per GPU as the number of generated tokens divided by the end-to-end latency.

Hardware. All runs are performed on 4× NVIDIA H100 GPUs unless specified otherwise. The experiments in Tabs. 6 and 7 are run on 4× NVIDIA A100 GPUs, and those in Tab. 8 on NVIDIA A100 GPUs with a maximum generation length of 512.

## G Positioning Against Prior Adaptive dLLM Decoding Methods

To make explicit how EB-Decode differs from prior adaptive dLLM decoding methods, Tab. 12 compares them along two axes: where to decode, i.e., how the block is formed, and when to commit, i.e., how tokens are accepted. Prior methods either fix the block size or form contiguous blocks with hand-designed heuristics, and most of them commit tokens with a static confidence threshold. EB-Decode is the only one that learns both axes and supports non-contiguous, variable-length blocks, so hard tokens can be deferred and resolved with richer right-side context once the surrounding easy tokens are decoded (Fig. 1).

Table 12: Positioning against prior dLLM decoding methods along the two decoding axes.
<table><tr><td>Method</td><td>Where to decode</td><td>When to commit</td><td>Non-contiguous blocks</td></tr><tr><td>Fast-dLLM [53]</td><td>fixed size</td><td>confidence threshold</td><td>no</td></tr><tr><td>AdaBlock-dLLM [35]</td><td>delimiter heuristic</td><td>confidence threshold</td><td>no</td></tr><tr><td>Swordsman [60]</td><td>entropy heuristic</td><td>confidence threshold</td><td>no</td></tr><tr><td>Learn2PD [4]</td><td>fixed size</td><td>learned (MLP, fixed length)</td><td>no</td></tr><tr><td>EB-Decode (ours)</td><td>learned</td><td>learned (Transformer, variable length)</td><td>yes</td></tr></table>

## H Entropy at the Sequence Tail

Fig. 2 (a) shows a persistent band of low entropy at high token positions across all decoding steps, which may appear at odds with the overall rising-entropy trend. This region corresponds to the EOS zone at the tail of the sequence. We report in Tab. 13 the three most frequent predictions at the last five positions across all decoding steps, where <|endoftext|> is overwhelmingly dominant, appearing in the top-5 predictions at 96.30% of the positions sampled. These positions therefore remain highly certain at almost every step, forming the persistent low-entropy band.

Table 13: Top-3 predicted tokens at the last 5 positions of the sequence, aggregated over all decoding steps on LLaDA-8B-Instruct. “Top-5 occurrence” is the fraction of sampled positions at which the token appears among the top-5 predictions.
<table><tr><td>Rank</td><td>Token</td><td>Avg. prob. (%)</td><td>Top-5 occurrence (%)</td></tr><tr><td>1</td><td>&lt;|endoftext|&gt;</td><td>28.00</td><td>96.30</td></tr><tr><td>2</td><td>&lt;|eot_id|&gt;</td><td>18.00</td><td>45.70</td></tr><tr><td>3</td><td></td><td>2.70</td><td>24.70</td></tr></table>

## I Limitations

We outline the main limitations of EB-Decode and the directions that we leave for future work.

Generation Length. Our main results are reported at generation lengths of 256 and 512 tokens, which match the standard configurations used in prior dLLM acceleration works [53, 35, 4]. As shown in Tab. 5, EB-Decode already exhibits a scaling trend in which the speedup grows from L=256 to L=512, suggesting that the framework remains effective as the sequence becomes longer. A more comprehensive study at substantially longer generation lengths is a natural extension we plan to pursue in future work.

Serving Conditions. Our throughput is measured at batch size 1, following prior dLLM acceleration works [53, 35, 4]. Evaluation under realistic serving conditions with continuous batching and systemlevel optimizations is left to future work, as it depends as much on dLLM serving infrastructure as on the decoding algorithm.

Method Complexity. EB-Decode introduces two routers and several hyperparameters, namely g<sub>max</sub>, $L _ { \operatorname* { m i n } } , \tau ,$ and $\tau _ { \psi } .$ , which makes it more involved to deploy and reuse as a baseline than a single-threshold heuristic. Each component contributes to the final performance (Tabs. 2–4), and the trained routers are largely reusable without retraining (Sec. 6.4), but reducing the number of moving parts remains a worthwhile direction.