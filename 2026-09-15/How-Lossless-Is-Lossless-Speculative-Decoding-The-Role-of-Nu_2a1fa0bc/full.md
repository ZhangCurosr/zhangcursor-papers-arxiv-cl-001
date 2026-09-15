# How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus

Ilya Koziev, Leonid Sinev, Ivan Oseledets

Orthrus is a hybrid autoregressive-diffusion architecture that accelerates autoregressive languagemodel inference by generating multiple tokens in parallel while using a frozen autoregressive backbone. Its central claim is that an intra-model consensus mechanism enables lossless speculative decoding, producing the same output sequence as the autoregressive model. We independently reproduce Orthrus and examine this claim under different numerical precisions. Under BF16 inference, exact trajectory matching occurs in only 45% of cases for the authors’ checkpoint and 43% for our independently trained model across 1,190 prompts from 12 domains. The probability of exact matching is also strongly associated with the response-conditional perplexity of the reference model. Despite this trajectory divergence, Orthrus does not show systematic degradation on downstream lm-eval-harness benchmarks. In contrast, repeating the trajectory evaluation with FP32 yields exact trajectory matching on all evaluated prompts. These results show that the practical losslessness of Orthrus depends on numerical precision and that exact trajectory equivalence should be evaluated separately from downstream task performance.

## 1. Introduction

Autoregressive (AR) language models have become the dominant paradigm for text generation, but their decoding procedure remains inherently sequential. Given a generated prefix, the model must compute the next-token distribution before the next token can be appended to the context. Consequently, generating a sequence of N tokens generally requires N decoding steps and repeatedly accesses the growing key-value (KV) cache. This sequential dependency limits hardware utilization and makes inference increasingly expensive as model and context lengths grow.

Diffusion language models and related parallel decoding methods address this bottleneck by predicting multiple future tokens simultaneously. However, relaxing the strict autoregressive dependency can introduce discrepancies from the original model distribution. Orthrus (Nguyen et al. 2026) proposes a particularly attractive alternative: rather than replacing or substantially modifying the pretrained autoregressive model, it augments a frozen AR backbone with a lightweight diffusion view. During inference, the AR component constructs the context representation while the diffusion component predicts multiple future tokens in parallel. An intra-model consensus mechanism then uses the autoregressive view to validate the proposed tokens. The authors report substantial inference acceleration while claiming that this procedure is strictly lossless, i.e., that the accelerated system preserves the exact predictive behavior of the original autoregressive model.

The present work investigates this losslessness claim experimentally. We independently implement the Orthrus architecture and training procedure and develop a configurable training framework that enables systematic investigation of training objectives, data distributions, and hyperparameters. This investigation leads to two observations. First, the distribution of the data used to train the diffusion view is important for parallel decoding efficiency. Because the diffusion component is trained under teacher forcing, training on generic human-written continuations does not necessarily reproduce the states that the frozen AR model encounters during its own generation process. We therefore construct a teacher-generated distillation corpus by prompting the frozen AR model and recording its greedy-decoded continuations. The resulting data follows the model’s own prediction trajectories. Our independently trained model achieves higher Tokens Per Forward (TPF) than the released checkpoint in most evaluation domains.

Second, and more importantly, our experiments reveal that the practical observation of losslessness depends on numerical precision. The consensus mechanism of Orthrus can provide exact trajectory equivalence under an idealized arithmetic model, but an implementation using finite-precision arithmetic need not reproduce the reference computation bit-for-bit. Because generation is discrete, even small numerical differences can eventually change the selected token and lead to divergent trajectories.

We demonstrate that this effect occurs in practical Orthrus inference. When generated sequences are compared directly against those of the corresponding frozen AR model, we observe non-zero rates of output divergence under BF16 inference. This finding is particularly relevant because Orthrus explicitly characterizes its generation procedure as strictly lossless. The observation does not imply that the consensus mechanism is ineffective: Orthrus can still retain extremely high fidelity while providing substantial acceleration. Rather, it shows that the term “lossless” requires a more precise operational definition when applied to neural inference systems implemented with finite-precision arithmetic and stateful KV cache.

Interestingly, the numerical differences do not necessarily manifest as a degradation in standard task-level evaluation. In some experiments, Orthrus obtains benchmark scores that are slightly higher than those of the corresponding autoregressive model. This observation further illustrates why benchmark-level equality cannot establish exact inference equivalence: two systems may obtain identical or statistically indistinguishable task scores while producing different token sequences. Conversely, a small numerical perturbation may occasionally move a generation toward a benchmark-preferred answer.

Our contributions are therefore threefold.

First, we provide an independent implementation of Orthrus training and inference and evaluate it alongside the released checkpoint.

Second, we show that an independently trained model using teacher-generated, on-policy distillation data achieves competitive or higher Tokens Per Forward across most of the evaluated domains.

Third, we show that exact sequence-level equivalence is highly sensitive to numerical precision: substantial trajectory divergence occurs under BF16, whereas the same evaluation yields exact trajectory matching under FP32.

Taken together, these results do not undermine the utility of Orthrus as an inferenceacceleration technique. Instead, they clarify an important limitation of strong losslessness claims for neural decoding systems: an algorithm may preserve the intended autoregressive computation in principle while still producing different discrete outputs when implemented with finite-precision arithmetic. We therefore argue that evaluations of “lossless” language-model acceleration should specify both the operational criterion for equivalence and the numerical precision under which it is measured.

## 2. Training Orthrus

The effects described in Section 3 and Section 4 are observed not only for the original chiennv/Orthrus-Qwen3-1.7B<sup>1</sup> checkpoint, but also for a model of the same capacity trained independently by us using the procedure described below. The two models differ substantially in both training-data composition and training hyperparameters. These differences make the independently trained model useful for assessing whether the observed effects depend on the specific training setup of the released checkpoint. We therefore describe our training procedure below, focusing on the aspects relevant to the experiments rather than on implementation details.

## 2.1 Training Data

The training data was obtained by distilling the autoregressive model Qwen/Qwen3-1.7B (Yang et al. 2025). It was constructed from a mixture of publicly available datasets on HuggingFace, using only their prompts. For each prompt, the response was generated by the autoregressive model using greedy decoding.

Only prompts containing between 50 and 1,000 characters were retained for distillation. The resulting dataset contains 4,113,358 prompt–response samples.

## 2.2 Training Parameters

Training was performed on eight NVIDIA H100 GPUs using CUDA 13.3.73, PyTorch 2.13.0, and Transformers 5.8.0. The training configuration was as follows:

r number of epochs: 1;

r initial learning rate: $2 \times 1 0 ^ { - 4 }$ ;

r batch size: 10;

r loss function: cross-entropy;

r block size: 8;

r number of blocks: 32;

r maximum sequence length: 3,072 tokens.

The training configuration differs substantially from that used in the original Orthrus experiments.

## 2.3 Evaluation

All evaluations in this section were conducted using a specially curated set of prompts, grouped into 12 text domains with 100 prompts per domain, except for “gec-en”, which contains 90 prompts. The domains are described in Table 1. This breakdown makes it possible to assess variation in generation trajectories and their statistical properties across domains.

Table 2 compares the Tokens Per Forward (TPF) of our model variant with the authors’ released checkpoint across the evaluation domains described above. Our model achieves a slightly higher TPF in 10 of the 12 domains, while the authors’ checkpoint performs better in the remaining two. The differences in TPF may reflect differences in training data and training configuration between the two models.

Table 1  
Datasets and task domains used for trajectory evaluation.
<table><tr><td>Domain</td><td>Data Source</td><td>Task Description</td></tr><tr><td>code</td><td>me-aas/python-code-dataset-500k</td><td>Python code generation in English</td></tr><tr><td>code-ru</td><td> $\mathtt { M E R A - e v a l u a t i o n / M E R A }$ </td><td>Python code generation in Russian</td></tr><tr><td>creative-en</td><td> $\mathtt { I s D e e C e e / S t o r y M a k e r }$ </td><td>English story generation</td></tr><tr><td>gec-en</td><td> $\mathtt { j h u \mathrm { - } c l s p / j f l e g }$ </td><td>English grammatical error correction</td></tr><tr><td>gec-ru</td><td>https://github.com/</td><td>Human-annotated Russian grammatical</td></tr><tr><td>math</td><td>ReginaNasyrova/LORuGEC HaimingW/math_train_decontaminated</td><td>error correction English-language mathematical problem</td></tr><tr><td></td><td></td><td>solving Russian-language mathematical prob-</td></tr><tr><td>math-ru</td><td>evilfreelancer/MATH-500-Russian</td><td>lem solving</td></tr><tr><td>poetry-en</td><td>checkai/instruction-poems</td><td>English poetry generation</td></tr><tr><td>poetry-ru</td><td>n/a (closed sources)</td><td>Russian poetry generation</td></tr><tr><td>qa</td><td></td><td>smd20/social-engineering-qa-english General question answering in English</td></tr><tr><td>qa-ru wmt ru-en</td><td>MERA-evaluation/MERA wmt/wmt19</td><td>General question answering in Russian Russian-to-English machine translation</td></tr></table>

## Table 2

Tokens Per Forward across different evaluation domains. Higher values indicate more effective parallel token generation. Values are means with 95% confidence intervals.
<table><tr><td>Domain</td><td> $\mathtt { c h i e n n v / 0 r t h r u s - Q w e n 3 - 1 . 7 B }$ </td><td>Orthrus-1.7B-final</td></tr><tr><td>code</td><td> $3 . 2 4 \pm 0 . 2 0$ </td><td> ${ \bf 3 . 4 4 \pm 0 . 1 3 }$ </td></tr><tr><td>code-ru</td><td> $3 . 5 9 \pm 0 . 2 2$ </td><td> ${ \bf 3 . 9 3 \pm 0 . 1 3 }$ </td></tr><tr><td>creative-en</td><td> $1 . 8 5 \pm 0 . 0 9$ </td><td> ${ \bf 2 . 0 2 \pm 0 . 1 0 }$ </td></tr><tr><td>gec-en</td><td> ${ \bf 5 . 0 0 \pm 0 . 3 0 }$ </td><td> $4 . 2 3 \pm 0 . 1 7$ </td></tr><tr><td>gec-ru</td><td> $2 . 4 8 \pm 0 . 1 8$ </td><td> ${ \bf 3 . 2 3 \pm 0 . 1 7 }$ </td></tr><tr><td>math</td><td> ${ \bf 8 . 0 8 \pm 0 . 6 5 }$ </td><td> $4 . 9 9 \pm 0 . 1 6$ </td></tr><tr><td>math-ru</td><td> $4 . 0 4 \pm 0 . 2 8$ </td><td> ${ \bf 4 . 3 5 \pm 0 . 1 4 }$ </td></tr><tr><td>poetry-en</td><td> $1 . 6 1 \pm 0 . 0 4$ </td><td> ${ \bf 1 . 8 6 \pm 0 . 0 4 }$ </td></tr><tr><td>poetry-ru</td><td> $1 . 8 7 \pm 0 . 1 9$ </td><td> ${ \bf 2 . 3 7 \pm 0 . 1 4 }$ </td></tr><tr><td>qa</td><td> $2 . 0 1 \pm 0 . 0 6$ </td><td> ${ \bf 2 . 2 4 \pm 0 . 0 6 }$ </td></tr><tr><td>qa-ru</td><td> $1 . 5 8 \pm 0 . 1 4$ </td><td> $\mathbf { 2 . 0 0 \pm 0 . 2 0 }$ </td></tr><tr><td>wmt ru-en</td><td> $2 . 0 4 \pm 0 . 0 9$ </td><td> ${ \bf 2 . 4 6 \pm 0 . 1 2 }$ </td></tr></table>

The independently trained model exhibits qualitatively similar behavior to the released checkpoint while achieving competitive or higher TPF in most domains. We next test the stronger claim of exact trajectory equivalence.

## 3. When Lossless Decoding Is Not Lossless

This section analyzes differences between the token trajectories generated by Orthrus and the original autoregressive model.

Experimental setup. We use Python 3.10.12, PyTorch 2.8.0+cu128, CUDA 12.8, and Transformers 5.8.1. All experiments are performed on an NVIDIA GeForce RTX 3090 GPU with 23 GB of memory (compute capability 8.6). Models are evaluated using BF16 precision and the eager attention implementation (attn\_implementation="eager").

For generation, we use the following arguments of the Transformers generate() method:

Table 3  
Orthrus–Qwen3 trajectory matching statistics under BF16 inference. Values are proportions with 95% confidence intervals.
<table><tr><td>Model</td><td></td><td>No. Trajectories Sequence Match Rate</td><td>Diverging Trajectory Rate</td></tr><tr><td> $\mathrm { O r t h r u s – Q w e n } 3 \substack { - } 1 . 7 \mathrm { B }$ </td><td>1,190</td><td> $0 . 4 5 \pm 0 . 0 3$ </td><td> $0 . 5 5 \pm 0 . 0 3$ </td></tr><tr><td>Orthrus-1.7B-final</td><td>1,190</td><td> $0 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 5 7 \pm 0 . 0 3$ </td></tr></table>

Table 4

Orthrus–Qwen3 trajectory matching rates by domain under BF16 inference. Values are proportions of exactly matching trajectories with 95% confidence intervals.
<table><tr><td>Domain</td><td>Authors</td><td>Ours</td></tr><tr><td>code</td><td> $0 . 3 6 \pm 0 . 1 0$ </td><td> $0 . 3 5 \pm 0 . 1 0$ </td></tr><tr><td>code-ru</td><td> $0 . 5 8 \pm 0 . 1 0$ </td><td> $0 . 5 2 \pm 0 . 1 0$ </td></tr><tr><td>creative-en</td><td> $0 . 1 2 \pm 0 . 0 6$ </td><td> $0 . 1 1 \pm 0 . 0 6$ </td></tr><tr><td>gec-en</td><td> $0 . 8 8 \pm 0 . 0 7$ </td><td> $0 . 8 8 \pm 0 . 0 7$ </td></tr><tr><td>gec-ru</td><td> $0 . 5 2 \pm 0 . 1 0$ </td><td> $0 . 4 9 \pm 0 . 1 0$ </td></tr><tr><td>math</td><td> $0 . 5 4 \pm 0 . 1 0$ </td><td> $0 . 5 4 \pm 0 . 1 0$ </td></tr><tr><td>math-ru</td><td> $0 . 5 9 \pm 0 . 1 0$ </td><td> $0 . 6 2 \pm 0 . 1 0$ </td></tr><tr><td>poetry-en</td><td> $0 . 1 1 \pm 0 . 0 6$ </td><td> $0 . 0 8 \pm 0 . 0 5$ </td></tr><tr><td>poetry-ru</td><td> $0 . 1 4 \pm 0 . 0 7$ </td><td> $0 . 1 1 \pm 0 . 0 6$ </td></tr><tr><td>qa</td><td> $0 . 2 5 \pm 0 . 0 9$ </td><td> $0 . 1 7 \pm 0 . 0 7$ </td></tr><tr><td>qa-ru</td><td> $0 . 7 3 \pm 0 . 0 9$ </td><td> $0 . 7 0 \pm 0 . 0 9$ </td></tr><tr><td>wmt ru-en</td><td> $0 . 6 4 \pm 0 . 1 0$ </td><td> $0 . 6 3 \pm 0 . 1 0$ </td></tr></table>

r max\_new\_tokens=128,

r do\_sample=False,

r temperature=0.0.

Thus, all models use greedy decoding, with no sampling applied during generation.

All subsequent results are presented for two variants of Orthrus: the original chiennv/Orthrus-Qwen3-1.7B and our Orthrus-1.7B-final, the training procedure for which is described in Section 2.

Table 3 shows the proportions of Orthrus trajectories that fully match Qwen trajectories and those that contain at least one divergence, without a breakdown by text domain. The aforementioned proportions, broken down by text domain, are shown in Table 4.

The conditional perplexity values, denoted as PPL (response|prompt), shown in the tables were calculated by the Qwen3-1.7B model. For a prompt x and generated response $y = ( y _ { 1 } , \dots , y _ { | y | } )$ , the response-conditional perplexity is:

$$
\mathrm { P P L } _ { \mathrm { Q w e n } } ( y \mid x ) = \exp \left( - \frac { 1 } { | y | } \sum _ { t = 1 } ^ { | y | } \log p _ { \mathrm { Q w e n } } \left( y _ { t } \mid x , y _ { < t } \right) \right) .
$$

Diverging trajectories exhibit higher response-conditional perplexity under the reference model, as shown by Table 5. To test whether this association persists after accounting for response length and domain, we fit the logistic regression with exact matching as the binary response and log $\mathrm { P P L } ( y \mid x )$ , response length, and domain as

Table 6  
Mean response-conditional PPL for matching and diverging Orthrus trajectories. Values are means with 95% confidence intervals.
<table><tr><td>Model</td><td></td><td>Matching trajectories Diverging trajectories</td></tr><tr><td>Orthrus-Qwen3-1.7B</td><td> $1 . 1 1 \pm 0 . 0 1$ </td><td> $1 . 2 8 \pm 0 . 0 1$ </td></tr><tr><td>Orthrus-1.7B-final</td><td> $1 . 1 0 \pm 0 . 0 1$ </td><td> $1 . 2 8 \pm 0 . 0 1$ </td></tr></table>

Logistic regression results for the association between response-conditional perplexity and exact trajectory matching.
<table><tr><td>Model</td><td> $\beta _ { 1 }$ </td><td>95% CI</td><td>p-value</td></tr><tr><td>chiennv/0rthrus-Qwen3-1.7B</td><td>-8.10</td><td>[−10.50, -5.71]</td><td> $3 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>Orthrus-1.7B-final</td><td>-10.92</td><td>[-13.60, -8.24]</td><td> $1 . 3 \times 1 0 ^ { - 1 5 }$ </td></tr></table>

predictors:

$$
\mathrm { l o g i t } P ( Y = 1 ) = \beta _ { 0 } + \beta _ { 1 } \log \mathrm { P P L } ( y \mid x ) + \beta _ { 2 } L + \gamma _ { D } .
$$

Here, $Y = 1$ denotes an exact trajectory match and $\gamma _ { D }$ represents domain effects. A negative $\beta _ { 1 }$ therefore indicates that higher response-conditional perplexity is associated with a lower probability of exact matching.

Logistic regression results performed using the statsmodels (Seabold and Perktold 2010) are shown in Table 6. The results indicate a strong negative association between response-conditional perplexity and the probability of exact trajectory matching. For the chiennv/Orthrus-Qwen3-1.7B model, the estimated coefficient for log PPL(y | x) is $\beta _ { 1 } =$ −8.10 (95% CI [−10.50, −5.71], $p = 3 \times 1 0 ^ { - 1 1 } )$ . A similar, and somewhat stronger, association is observed for Orthrus-1.7B-final, with $\beta _ { 1 } = - 1 0 . 9 2 ( 9 5 \% \mathrm { C I } [ - 1 3 . 6 \bar { 0 } , - 8 . 2 4 ] _ { A }$ $p = 1 . 3 \times 1 0 ^ { - 1 5 } )$ . Thus, for both implementations, responses that are assigned higher conditional perplexity by the reference Qwen3-1.7B model are substantially less likely to be reproduced exactly by Orthrus. The confidence intervals exclude zero by a wide margin, indicating that this association remains statistically significant after controlling for response length and prompt domain. These results indicate that trajectory divergence is systematically associated with higher response-conditional perplexity under the reference model, rather than occurring uniformly across inputs.

Despite the observed trajectory divergence, these deviations do not translate into systematic degradation in downstream task performance.

## 4. When Losses Become Gains

We evaluated Qwen3 and both Orthrus models on GSM8K, HumanEval, and IFEval using lm-eval-harness (Gao et al. 2023). Model and generation parameters are the same as those described in Section 3. As shown in Table $^ { 7 , }$ our Orthrus model has higher point estimates than the autoregressive Qwen3 baseline on all three benchmarks. These differences should not be interpreted as statistically significant improvements given the reported uncertainty.

If the original autoregressive trajectory is assumed to provide the reference behavior, deviations from this trajectory reported in Section 3 might be expected to reduce downstream performance. Instead, the observed deviations do not consistently have a negative effect and can coincide with higher task scores. Thus, trajectory divergence does not imply systematic degradation in downstream task performance.

Table 7  
Comparison of Qwen3-1.7B, the authors’ Orthrus-Qwen3-1.7B, and our Orthrus-1.7B-final on lm-eval-harness benchmarks.
<table><tr><td>Task</td><td>Metric</td><td>Qwen3-1.7B</td><td>Orthrus-Qwen3-1.7B</td><td>Orthrus-1.7B-final</td></tr><tr><td>GSM8K</td><td>exact match (flexi- ble)</td><td> $0 . 4 0 0 3 \pm 0 . 0 1 3 5$ </td><td> $0 . 4 2 3 8 \pm 0 . 0 1 3 6$ </td><td> $0 . 4 1 4 7 \pm 0 . 0 1 3 6$ </td></tr><tr><td>HumanEval pass@1</td><td></td><td> $0 . 4 0 2 4 \pm 0 . 0 3 8 4$ </td><td> $0 . 3 6 5 9 \pm 0 . 0 3 7 7$ </td><td> $\mathbf { 0 . 4 1 4 6 \pm 0 . 0 3 8 6 }$ </td></tr><tr><td>IFEval</td><td>prompt-level loose accuracy</td><td> $0 . 2 0 1 5 \pm 0 . 0 1 7 3$ </td><td> $0 . 2 0 5 2 \pm 0 . 0 1 7 4$ </td><td> $\mathbf { 0 . 2 1 8 1 \pm 0 . 0 1 7 8 }$ </td></tr><tr><td>IFEval</td><td>prompt-level strict accuracy</td><td> $0 . 1 6 8 2 \pm 0 . 0 1 6 1$ </td><td> $0 . 1 7 5 6 \pm 0 . 0 1 6 4$ </td><td> $\mathbf { 0 . 1 8 3 0 \pm 0 . 0 1 6 6 }$ </td></tr></table>

Table 8  
Orthrus–Qwen3 trajectory matching statistics under FP32 inference.
<table><tr><td>Model</td><td>No. of Trajectories</td><td>Sequence Match Rate</td><td>Diverging Trajectory Rate</td></tr><tr><td>Orthrus-Qwen3-1.7B</td><td>1,190</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Orthrus-1.7B-final</td><td>1,190</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr></table>

## 5. The Effect of Numerical Precision

The BF16 experiments demonstrate that neither Orthrus variant consistently reproduces the autoregressive trajectory exactly. We next ask whether this discrepancy persists under higher numerical precision. We therefore repeat the evaluation using FP32 precision for both Orthrus variants and their corresponding reference computations while keeping the model parameters, decoding procedure, and evaluation prompts unchanged.

Under FP32 inference, Orthrus produces exactly the same token trajectories as the autoregressive reference on all 1,190 prompts in the trajectory evaluation (Table 8), whereas the corresponding BF16 configuration exhibits substantial trajectory divergence (Table 3).

These observations demonstrate that the practical behavior of Orthrus is sensitive to numerical precision. The apparent violations of exact trajectory equivalence observed under BF16 disappear when the computations are performed in FP32. We therefore attribute the observed trajectory divergence to finite-precision numerical effects, without attributing them to any particular layer or computational operation. Identifying the specific computational stages responsible for these precision-dependent deviations remains an interesting direction for future work.

The sensitivity of LLM inference to numerical precision has also been observed in studies of inference reproducibility, where changes in floating- point precision and hardware configuration can alter outputs even under greedy decoding (Yuan et al. 2025).

Our setting differs in that we study numerical precision specifically in the context of lossless speculative decoding: the question is not merely whether an LLM is reproducible across inference configurations, but whether an accelerated model reproduces the exact trajectory of its autoregressive reference.

https://arxiv.org/abs/2505.09388

## 6. Conclusion

We investigated the losslessness claim of Orthrus by directly comparing its generated trajectories with those of the corresponding autoregressive model. Under BF16 inference, exact trajectory equivalence was not preserved: 45% of trajectories matched the released checkpoint and 43% matched our independently trained model. Trajectory matching was strongly associated with the response-conditional perplexity of the reference model, while the observed divergences did not result in systematic degradation on the evaluated downstream benchmarks. Crucially, repeating the same trajectory evaluation with FP32 yielded exact matching on all 1,190 evaluated prompts. These results indicate that the practical losslessness of Orthrus depends on numerical precision and that claims of lossless language-model acceleration should specify the numerical precision and operational criterion under which equivalence is assessed.

## Acknowledgments

We thank Sergei Markov for supporting this research with computational resources and Valery Ternovsky for a methodologically sound question regarding the benchmarking of our Orthrus variant.

## References

Gao, Leo, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2023. A framework for few-shot language model evaluation. https: //doi.org/10.5281/zenodo.10256836

Nguyen, Chien Van, Chaitra Hegde, Van Cuong Pham, Ryan A. Rossi, Franck Dernoncourt, and Thien Huu Nguyen. 2026. Orthrus: Memory-efficient parallel token generation via dual-view diffusion. preprint arXiv:2605.12825. https://arxiv.org/abs/2605.12825

Seabold, Skipper and Josef Perktold. 2010. Statsmodels: Econometric and statistical modeling with Python. In Proceedings of the 9th Python in Science Conference, pages

92–96. https://doi.org/10.25080/Majora-92bf1922-011

Yang, An, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. preprint arXiv:2505.09388.

Yuan, Jiayi, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan, Jing Shi, Xia Hu, and Zirui Liu. 2025. Understanding and mitigating numerical sources of nondeterminism in LLM inference. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 169819–169851. https: //doi.org/10.52202/085713-5653