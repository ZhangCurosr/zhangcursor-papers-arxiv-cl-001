# Momentum, Commentary Tone, and Earnings Surprise in LLM Financial Forecasts

A Controlled Study of Six Q8 Checkpoint Conversions

Alina Khaybullina

October 2026

## Abstract

Which inputs drive a language model’s financial forecast? We construct synthetic earnings announcements that independently reverse 20-day momentum, non-informational commentary tone, and whether fixed actual EPS beats a manipulated consensus. Six instruction/chat checkpoint conversions from four families (0.49–7.61B base parameters) receive 2,016 prompts each under a direct-answer Q8 0 protocol. The design includes 24 numerical blocks, six commentary pairs, profits and losses, rising and falling annual EPS, and two surprise sizes. Both A/B answer assignments and U/D cover all 384 primary scenarios per conversion. We estimate matched changes in conditional answer-prefix scores, not calibrated market probabilities. Across these checkpoints, the primary EPS contrast ranges from −0.19 to 86.80 percentage points. The magnitude of the primary signed tone contrast exceeds that of EPS in 5 of six models and momentum exceeds it in 5. Natural-word numerical diagnostic accuracy ranges from 51.0 to 95.8 percent. Answer-encoding, wording, context, and conversion controls qualify these results. The experiment measures observable input reliance and numerical recognition; it establishes neither an internal reasoning mechanism nor realized return predictability. Public benchmark prompts, recorded scores, and an interactive recorded-results explorer accompany the study.

## 1 Introduction

Which information actually changes a language model’s market prediction? A plausible financial forecast may follow recent returns, react to the emotional framing of an announcement, or compare newly released earnings with expectations. Natural financial text usually combines these signals. If they all agree, the final forecast cannot distinguish their influence.

An earnings announcement makes the distinction concrete. Higher earnings than a year earlier can still miss consensus; a loss can beat an even worse expected loss. Upbeat commentary can accompany either numerical outcome. We hold actual earnings fixed and reverse the surprise by changing the stated consensus, while independently reversing momentum and commentary. This identifies the efect of the information supplied to the model under the tested protocol.

This paper expands an audited two-checkpoint pilot. The primary experiment now covers six checkpoints in four families, a wider size range, six commentary pairs, losses, annual declines, and small as well as large surprises. Label reversal is part of every primary case. The pilot’s raw results are retained separately, and a conversion/runtime bridge compares saved full-precision scores with the new inference setup. The original expanded protocol was internally frozen before expanded inference, after the pilot had already been inspected; it is not a public preregistration or a fresh confirmation of pilot-independent hypotheses. A later coverage audit added profitable cases to the smaller word and ofice controls. That explicitly exploratory amendment changes neither the frozen primary prompts nor the primary estimands.

The study has three questions. First, how much do momentum, tone, and numerical surprise change the requested forecast score? Second, do those efects survive changes in answer encoding and wording? Third, can the model recognize the beat/miss relation when asked directly? These questions concern observable behavior. A numerical efect is not a proof of reasoning, and a momentum efect is not automatically a mistake: price history may contain genuine information in real markets. We do not observe market outcomes in this experiment.

## 2 Related work

Lopez-Lira and Tang [2025] evaluate language-model headline assessments against market returns. That motivates scrutiny of input processing, while their predictive estimand difers from our controlled output-intervention estimand. Turpin et al. [2023] show that generated explanations can omit influential prompt features, supporting the use of output interventions rather than selfreported explanations. Mirzadeh et al. [2025] use systematic numerical variants to test mathematical reasoning; our financial construction similarly separates factual comparisons from framing. Zhao et al. [2021] document prompt and answer-preference efects, motivating complete label reversal here. Shu et al. [2025] introduce a broader multimodal earnings-surprise benchmark; our balanced synthetic cases complement rather than replace real financial data. Zhang et al. [2025] use event counterfactuals in financial forecasting, while our interventions probe a fixed model. Vamvourellis and Mehta [2025] examine financial sentiment classification under reasoning prompts, a diferent task from return prediction. Lee et al. [2026] distinguish deterministic financial assessment from stochastic prediction, which is central to separating our diagnostic from the forecast score.

The Qwen2.5 family is described by Qwen Team [2025]; oficial checkpoint documentation describes the SmolLM2, TinyLlama, and Phi models [Hugging Face Smol Models Research, 2025; TinyLlama Project, 2024; Microsoft, 2024]. We make no claim that counterfactual financial evaluation is generally new. The contribution is the expectation-reversal design combined with full answer-mapping controls and a reproducible multi-family comparison of factual recognition and forecast sensitivity.

## 3 Benchmark and protocol

## 3.1 Numerical blocks

We generate 24 earnings blocks with NumPy’s default random generator and seed 20261008. Blocks cross two signs of actual EPS, two signs of annual EPS change, and two surprise-gap fractions (2% or 15% of absolute actual EPS), with three numerical replicates per combination. Absolute actual EPS is sampled uniformly between \$0.80 and \$4.20 and rounded to cents. For profitability sign $p _ { b }$ annual-change sign $h _ { b }$ , and a draw $u _ { b } \sim U ( 0 . 0 8 , 0 . 2 5 )$ , prior-year EPS is $a _ { b } - h _ { b } | a _ { b } | u _ { b }$ , rounded to cents. We describe annual change in EPS levels; percentage growth across negative earnings is not used.

For surprise sign $e \in \{ - 1 , + 1 \}$ , consensus is

$$
c _ { b , e } = a _ { b } - e g _ { b } , \qquad g _ { b } = \operatorname* { m a x } \bigl ( 0 . 0 1 , \operatorname { r o u n d } ( | a _ { b } | f _ { b } , 2 ) \bigr ) .\tag{1}
$$

Actual and prior EPS remain fixed within a block. A beat means $a _ { b } \ > \ c _ { b , e }$ even when both are negative. Changing consensus manipulates the information supplied to the model, not actual investor expectations. Revenue meets consensus, guidance is unchanged, and no other news is released.

Momentum sign m selects positive or negative 20-day pre-release returns of the same blockspecific magnitude, sampled uniformly from 3–12% and rounded to one decimal. The market is flat. Sentiment sign s selects one of six positive/negative commentary pairs, explicitly described as opinion with no additional facts. Pair assignments are randomized and balanced across blocks (four blocks per pair). Three introductions are similarly assigned (eight blocks per introduction). Every block uses both section orders: facts–prices–commentary and commentary–prices–facts. Pair and introduction are between-block assignments; order and the three financial inputs are crossed within blocks.

The eight $( m , s , e )$ combinations and two orders give 16 semantic scenarios per block and 384 per model. All receive three answer encodings: $\mathrm { A { = } p o s i t i v e / B { = } n e g a t i v e }$ (AB0), the reversed assignment (AB1), and $\mathrm { U } { = } \mathrm { p o s i t i v e } / \mathrm { D } { = } \mathrm { n e g a t i v e }$ (UD). Thus there are 1,152 primary prompts per model. Counts refer to repeated counterfactual model evaluations, not independent market events.

## 3.2 Controls and expanded coverage

The isolated numerical diagnostic crosses 24 blocks, two surprise signs, and two sentence forms: 96 scenarios. Each receives both $\mathrm { A } / \mathrm { B }$ assignments and natural beat/miss prefixes, giving 288 diagnostic prompts. Commentary removal covers every block, both momentum and surprise signs, facts-first order, and both assignments (192 prompts, 96 averaged cases). The original natural-word and ofice controls selected the first complete block assigned to each commentary pair: 96 prompts each. A coverage audit after three completed expanded runs found that all six selected blocks were losses. An internally timestamped amendment adds the first profitable block for each pair, selected only from frozen block metadata. The resulting controls are balanced for profitability and pair: 192 word prompts and 192 ofice prompts (96 averaged ofice cases). Annual-change and gap-size balance is not guaranteed in this subset. The original primary, diagnostic, and neutral designs remain unchanged; these additional controls are exploratory.

The original freeze covers 1,824 cases per model; the amendment adds 192. Combined coverage is 2,016 prompts per model, or 12,096 across six models. Another 48 pilot bridge cases for each repeated Qwen checkpoint give 12,192 newly scored records. The 1,248 saved pilot records are distinct and are not pooled into primary estimates.

The six commentary pairs are authored variants from one construction process, not independent human authors or sentiment annotations. Numerical signs and gap sizes broaden scenario coverage, but the experiment still uses short synthetic disclosures rather than observed financial news. Table 7 lists every pair; full prompts are supplied as JSONL.

## 3.3 Models and inference

All checkpoints use revision-pinned Q8 0 GGUF conversions in llama-cpp-python 0.3.16, with the oficial Hugging Face tokenizer and chat template from pinned revisions. The repeated Qwen tokenizers retain their pilot revisions. Qwen conversions are first-party; SmolLM2 and Phi conversions are by bartowski, and TinyLlama’s is by TheBloke. These provenance diferences and quantization are disclosed rather than assumed immaterial. Pinned tokenizer revisions do not certify the source weight revision or source precision used by any downloaded conversion, including first-party files. The distributed manifests identify the actual conversion files and their SHA-256 hashes.

Inference uses CPU execution, four single-token threads, four or eight batch-prefill threads, a 512-token context, and no sampling. The initial inference batch size was 512; after an executionenvironment replacement the remaining Qwen7 cases and independent audits use batch size 128 to reduce memory. Twelve uncached comparisons against saved Qwen7 cases verify the restored runtime and batch-size change before resumption; this is an execution change, not a change to the prompt design. Early Qwen evaluations used four batch threads; later evaluations use eight after a 16-case Qwen7 timing check found identical scores at both thread counts. Evaluation order is optimized by grouping shared oficial-token prefixes; frozen prompts and estimands are unchanged. Every complete input and requested continuation fits within the context. All rendered inputs and candidate boundaries are checked. Native GGUF segmentation agrees with oficial Hugging Face tokenization for four conversions. TinyLlama’s native piece segmentation and Phi’s native specialtoken whitespace handling difer from the oficial tokenizers. Every oficial vocabulary ID is verified against the conversion (32,000 for TinyLlama and 32,011 for Phi), and oficial Hugging Face token IDs are supplied directly, bypassing native segmentation. Phi’s remaining padded vocabulary slots are not prompt/candidate tokens. Prompt and token-array hashes, raw candidate log probabilities, margins, requested-prefix mass, and top-token information are recorded. Cached and full-context evaluation are compared before each run. Twelve spread-out cases per model are also re-evaluated independently without that cache.

<table><tr><td>Checkpoint</td><td>Params (B)</td><td>Family</td><td>Tokenizer rev.</td><td>GGUF rev.</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.49</td><td>Qwen</td><td>7ae55760</td><td>9217f5db</td></tr><tr><td>Qwen2.5-1.5B</td><td>1.54</td><td>Qwen</td><td>989aa798</td><td>91cad511</td></tr><tr><td>SmolLM2-1.7B</td><td>1.71</td><td>SmolLM2</td><td>31b70e2e</td><td>1f034647</td></tr><tr><td>TinyLlama-1.1B</td><td>1.10</td><td>TinyLlama</td><td>fe8a4ea1</td><td>52e7645b</td></tr><tr><td>Phi-3.5-mini</td><td>3.82</td><td>Phi</td><td>2fe19245</td><td>6d70da17</td></tr><tr><td>Qwen2.5-7B</td><td>7.61</td><td>Qwen</td><td>a09a3545</td><td>bb5d59e0</td></tr></table>

Table 1: Six instruction/chat checkpoints, all evaluated as Q8 0 conversions. Revision columns are abbreviated; complete hashes and file digests are distributed. Parameter counts describe the base checkpoints and are rounded.

## 3.4 Measured scores and primary estimand

For prompt x and positive/negative candidate token sequences $y _ { + } , y _ { - }$ , let

$$
\ell _ { j } ( x ) = \sum _ { t = 1 } ^ { | y _ { j } | } \log P ( y _ { j , t } | x , y _ { j , < t } ) , \qquad q ( x ) = { \frac { \exp \ell _ { + } ( x ) } { \exp \ell _ { + } ( x ) + \exp \ell _ { - } ( x ) } } .\tag{2}
$$

Continuations are tokenized as sufixes of the complete rendered prompt plus candidate, preserving the actual token boundary. Multi-token words receive full prefix likelihoods. No end-of-answer token is included; these are prefix comparisons, not probabilities of complete generated responses. Requested-prefix mass is exp $\ell _ { + } + \exp \ell _ { - }$ , and is reported to reveal conditioning on unlikely answer prefixes. Natural-word candidates can have diferent token lengths; full-prefix likelihood retains output-word and length preferences. Those controls provide another answer-format measurement rather than a bias-free semantic probability.

For each semantic scenario i, the primary score is $\bar { q } _ { i } = ( q _ { i , \mathrm { A B 0 } } + q _ { i , \mathrm { A B 1 } } ) / 2$ . Label averaging is a transparent robustness summary; it is not a validated calibration method. UD and the individual assignments remain separate results. For input $F \in \{ m , s , e \}$ , the block contrast is

<table><tr><td>Checkpoint</td><td>Momentum</td><td>Tone</td><td>EPS surprise</td></tr><tr><td>Qwen2.5-0.5B</td><td>1.24 [0.816, 1.667]</td><td>4.02 [3.089, 5.027]</td><td>-0.15 [−0.506, 0.192]</td></tr><tr><td>Qwen2.5-1.5B</td><td>16.55 [15.224, 18.028]</td><td>30.39 [26.558, 34.113]</td><td>-0.19 [−0.444, 0.065]</td></tr><tr><td>SmolLM2-1.7B</td><td>6.27 [5.740, 6.822]</td><td>6.46 [5.551, 7.355]</td><td>-0.08 [−0.187, 0.029]</td></tr><tr><td>TinyLlama-1.1B</td><td>1.01 [0.859, 1.169]</td><td>0.22 [–0.134, 0.587]</td><td>-0.10 [−0.194, -0.003]</td></tr><tr><td>Phi-3.5-mini</td><td>20.77 [16.174, 26.089]</td><td>45.01 [33.089, 57.099]</td><td>6.57 [4.566, 8.772]</td></tr><tr><td>Qwen2.5-7B</td><td>10.65 [6.702, 14.726]</td><td>9.84 [5.359, 14.531]</td><td>86.80 [81.027, 92.038]</td></tr></table>

Table 2: Primary positive-minus-negative input efects in percentage points (pp), averaging both $\mathrm { A } / \mathrm { B }$ assignments on every case. Brackets show 95% percentile intervals from 10,000 resamples of 24 earnings blocks. These are conditional benchmark intervals, not uncertainty about realized stock returns.

$$
\Delta _ { b , F } = \mathrm { m e a n } ( \bar { q } _ { i } \mid b , F = + 1 ) - \mathrm { m e a n } ( \bar { q } _ { i } \mid b , F = - 1 ) , \qquad \Delta _ { F } = \frac { 1 } { 2 4 } \sum _ { b } \Delta _ { b , F } .\tag{3}
$$

Each marginalization averages over the other two inputs and section order. Efects are reported in percentage points. We also compute contrasts on the average candidate log-likelihood ratio to distinguish scale saturation from sign changes.

## 3.5 Uncertainty and surrogates

We use 10,000 percentile bootstrap resamples of the 24 numerical blocks. This holds the six authored commentary pairs and model set fixed, and quantifies only variation within this benchmark. Commentary-pair contrasts and leave-one-pair-out ranges provide separate wording checks. Four blocks per pair give limited replication; a six-pair bootstrap would not establish generalization to the universe of financial language. Profitability, annual-change, and gap-size subgroups are exploratory between-block comparisons and can also reflect assigned wording diferences.

Ordinary least-squares surrogates predict ¯q from momentum and sentiment, then add surprise, then section order and categorical pair/introduction indicators. Fitting leaves one earnings block out at a time; held-block scores never enter the training fit. We also leave one commentary pair out using the three financial inputs only. $R ^ { 2 }$ evaluates reproduction of model scores, not market accuracy. We report descriptive intervals and robustness checks without using a large collection of tests to declare a universal mechanism.

## 4 Results

## 4.1 Cross-model input reliance

Qwen2.5-0.5B: momentum 1.24 pp, tone 4.02 pp, and EPS surprise −0.15 pp. Its natural-word diagnostic accuracy is 55.2%, and its A/B-averaged diagnostic accuracy is 50.0%.

Qwen2.5-1.5B: momentum 16.55 pp, tone 30.39 pp, and EPS surprise −0.19 pp. Its natural-word diagnostic accuracy is 63.5%, and its A/B-averaged diagnostic accuracy is 63.5%.

SmolLM2-1.7B: momentum 6.27 pp, tone 6.46 pp, and EPS surprise −0.08 pp. Its natural-word diagnostic accuracy is 51.0%, and its A/B-averaged diagnostic accuracy is 50.0%.

TinyLlama-1.1B: momentum 1.01 pp, tone 0.22 pp, and EPS surprise −0.10 pp. Its natural-word diagnostic accuracy is 54.2%, and its A/B-averaged diagnostic accuracy is 50.0%.

![](images/eb1248ff995e7896013c25c856eda87f76fe93f76acf4cd8df5471c210f52499.jpg)  
Figure 1: Primary input contrasts. The three colored, shape-coded series share one horizontal scale; the fixed conversion and wording set. Printed values give the signed contrasts in pp.<sup>each</sup> <sup>row</sup> <sup>represents</sup> <sup>a</sup> <sup>checkpoint</sup> <sup>conversion.</sup> <sup>Intervals</sup> <sup>resample</sup> <sup>24</sup> <sup>earnings</sup> <sup>blocks</sup> <sup>conditional</sup> <sup>on</sup> the fixed conversion and wording set. Printed values give the signed contrasts in pp.

Phi-3.5-mini: momentum 20.77 pp, tone 45.01 pp, and EPS surprise 6.57 pp. Its natural-word diagnostic accuracy is 84.4%, and its A/B-averaged diagnostic accuracy is 77.1%.

Qwen2.5-7B: momentum 10.65 pp, tone 9.84 pp, and EPS surprise 86.80 pp. Its natural-word diagnostic accuracy is 95.8%, and its A/B-averaged diagnostic accuracy is 91.7%.

6<sub>Across</sub> <sub>this</sub> <sub>model</sub> <sub>set,</sub> <sub>magnitudes</sub> <sub>of</sub> <sub>the</sub> <sub>average</sub> <sub>signed</sub> <sub>tone</sub> <sub>contrast</sub> <sub>exceed</sub> <sub>magnitudes</sub> <sub>of</sub> <sub>the</sub> average signed EPS contrast in 5 of six checkpoints; momentum exceeds EPS by the same criterion in 5. This is a benchmark-specific ranking of directional responses. The strongest EPS contrast is observed for Qwen2.5-7B, at 86.80 pp; the smallest is Qwen2.5-1.5B, at −0.19 pp. Checkpoint diferences combine architecture, pretraining, post-training, size, and conversion provenance. They do not identify a causal family or parameter-count efect.

The absolute value of a signed mean is not the mean absolute change. We therefore also report the absolute score movement for every matched input reversal (Table 9), its EPS uppertail magnitude, and the fraction of EPS pairs moving in the positive direction. A small signed mean can arise from cancellation and is not, by itself, evidence of complete insensitivity. Orderspecific contrasts provide another check on averaging across prompt placement (Table 10). These are exploratory supplements to the unchanged primary signed estimand.

<table><tr><td>Checkpoint</td><td>Label MAE</td><td>Flip (%)</td><td>AB-UD</td><td>Order</td><td>Office</td></tr><tr><td>Qwen2.5-0.5B</td><td>12.06</td><td>43.0</td><td>11.43</td><td>3.26</td><td>7.50</td></tr><tr><td>Qwen2.5-1.5B</td><td>9.79</td><td>6.8</td><td>15.75</td><td>11.12</td><td>2.64</td></tr><tr><td>SmolLM2-1.7B</td><td>9.21</td><td>0.0</td><td>3.52</td><td>1.97</td><td>1.69</td></tr><tr><td>TinyLlama-1.1B</td><td>63.32</td><td>83.3</td><td>24.71</td><td>7.66</td><td>3.38</td></tr><tr><td>Phi-3.5-mini</td><td>17.65</td><td>17.7</td><td>9.46</td><td>6.01</td><td>6.02</td></tr><tr><td>Qwen2.5-7B</td><td>3.02</td><td>2.3</td><td>2.69</td><td>7.56</td><td>5.74</td></tr></table>

Table 3: Forecast format and irrelevant-content controls. MAE columns are mean absolute score diferences in pp. Label reversal and AB–UD use all 384 primary scenarios; order compares 192 matched cases; ofice compares 96 A/B-averaged cases. Flip is the share of $\mathrm { A } / \mathrm { B }$ reversals changing whether the conditional positive-answer score exceeds 0.5.
<table><tr><td>Checkpoint</td><td>A=positive</td><td>B=positive</td><td> $\mathrm { A } / \mathrm { B }$  mean</td><td> $\mathrm { U } / \mathrm { D }$ </td></tr><tr><td>Qwen2.5-0.5B</td><td>Tone</td><td>Tone</td><td>Tone</td><td>Tone</td></tr><tr><td>Qwen2.5-1.5B</td><td>Tone</td><td>Tone</td><td>Tone</td><td>Tone</td></tr><tr><td>SmolLM2-1.7B</td><td>Tone</td><td>Momentum</td><td>Tone</td><td>Momentum</td></tr><tr><td>TinyLlama-1.1B</td><td>Momentum</td><td>Tone</td><td>Momentum</td><td>Momentum</td></tr><tr><td>Phi-3.5-mini</td><td>Tone</td><td>Tone</td><td>Tone</td><td>Tone</td></tr><tr><td>Qwen2.5-7B</td><td>EPS</td><td>EPS</td><td>EPS</td><td>EPS</td></tr></table>

Table 4: Input with the largest absolute mean contrast in each encoding, on the same 384 scenarios. Ranking is descriptive and ignores interval overlap; negative efects can also have large absolute magnitude. Full signed efects are shown in Figure 2.

## 4.2 Answer encoding and lexical variation

Label reversal changes the question’s answer mapping while preserving financial content. A/B averaging may reduce assignment preferences but cannot guarantee semantic invariance. The fully matched UD comparison tests another encoding; natural words provide a smaller additional check. A reversal of the sign or rank of a factor across these formats weakens any encoding-independent interpretation. Low requested-prefix mass further limits the meaning of scores conditional on those prefixes (Table 14).

The wording results supply a direct check on the pilot’s single-pair limitation. The six pairs broaden language coverage but share one authoring process. Pair-specific results and leave-onepair-out ranges are therefore robustness evidence conditional on these templates, not population estimates of lexical invariance. The complete ranges are in the machine-readable summary. Naturalword comparisons use matched subsets rather than comparing unequal scenario samples (Table 13).

## 4.3 Numerical recognition and commentary removal

The isolated EPS task has a known correct relation, whereas the next-day abnormal-return forecast does not have observed outcomes. Strong diagnostic performance can coexist with a small EPS forecast contrast: recognizing a fact and using it to form a forecast are diferent behaviors. Conversely, weak diagnostic performance prevents a small EPS contrast from being attributed solely to deliberate down-weighting of the surprise. Numerical accuracy remains answer-format dependent, so all assignments and natural words are retained.

![](images/f10e02a900ee2f344036105396844ce7ae4d4228012dda5d89c18195c3c5571d.jpg)  
Figure 2: Matched efects under AB0, AB1, their casewise score average, and UD on all 384 primary<sub>Figure</sub> <sub>2:</sub> <sub>Matched</sub> <sub>efects</sub> <sub>under</sub> <sub>AB0,</sub> <sub>AB1,</sub> <sub>their</sub> <sub>casewise</sub> <sub>score</sub> <sub>average,</sub> <sub>and</sub> <sub>UD</sub> <sub>on</sub> <sub>all</sub> <sub>384</sub> <sub>primary</sub> scenarios per model. Individual mappings are not treated as independent replications.

Commentary removal is a context intervention, not a single-word sentiment flip. Its EPS efects are reported separately from the primary factorial average and should not be attributed exclusively to sentiment suppression. The ofice-color and section-order controls show whether nominally irrelevant changes move the score. The numerical subgroup results are given in Table 11; the sample Qwen2.5-1.5B Tone Tone Tone Toneis too small and wording assignment too limited to establish universal diferences between profits and losses.

## <sup>Qwen2.5-7B</sup> <sup>EPS</sup> <sup>EPS</sup>4.4 Simple surrogates and size comparisons

A small incremental $R ^ { 2 }$ gain from EPS means that adding the surprise sign helps little in this magnitude. Full signed efects are shown in Figure 2.<sup>particular surrogate. It does not prove that every computation inside the model is a two-cue</sup> rule, and omitted nonlinearities or interactions can matter. Pair and introduction indicators vary between blocks, so their $R ^ { 2 }$ increment can also absorb correlated numerical baseline variation; it is not an isolated causal template efect. The fully crossed order and label controls identify their 8specific prompt interventions more directly. The held-commentary-pair result is a harder wording transfer test than leaving out one numerical block. The three Qwen sizes permit a within-family descriptive comparison, but three checkpoints cannot support a fitted scaling law, and parameter count is confounded with training and post-training choices.

## 4.5 Bridge to the audited pilot

Qwen2.5-0.5B has a mean absolute bridge diference of 15.64 pp, a maximum of 40.74 pp, and 22.9% conditional decision flips.

Qwen2.5-1.5B has a mean absolute bridge diference of 6.21 pp, a maximum of 51.94 pp, and 8.3% conditional decision flips. A targeted six-case investigation independently reproduces both the saved FP32 and downloaded-Q8 paths. Exporting the pinned HF source to native F16 gives a maximum score diference of 0.668 pp from the fresh FP32 path on these cases. Directly rebuilding Q8 from that F16 file difers from the downloaded Q8 conversion by as much as 46.136 pp. Only 73 of 291 stock tensor entries match byte-for-byte in the rebuilt file; tensor storage, source precision, preprocessing and converter choices can difer. Byte diferences alone do not establish diferent training or an incorrect export. The measured outputs nevertheless show that these Q8 conversions are not interchangeable for the tested scores. This reinforces the scope of the main results: the specified conversion files, rather than a precision-invariant assessment of each named base checkpoint.

<table><tr><td>Checkpoint</td><td>A=beat</td><td>B=beat</td><td>Mean score</td><td>Mean accuracy</td><td>Words</td></tr><tr><td>Qwen2.5-0.5B</td><td>43.8</td><td>50.0</td><td>50.0</td><td>46.9</td><td>55.2</td></tr><tr><td>Qwen2.5-1.5B</td><td>60.4</td><td>70.8</td><td>63.5</td><td>65.6</td><td>63.5</td></tr><tr><td>SmolLM2-1.7B</td><td>50.0</td><td>50.0</td><td>50.0</td><td>50.0</td><td>51.0</td></tr><tr><td>TinyLlama-1.1B</td><td>50.0</td><td>50.0</td><td>50.0</td><td>50.0</td><td>54.2</td></tr><tr><td>Phi-3.5-mini</td><td>78.1</td><td>75.0</td><td>77.1</td><td>76.6</td><td>84.4</td></tr><tr><td>Qwen2.5-7B</td><td>90.6</td><td>94.8</td><td>91.7</td><td>92.7</td><td>95.8</td></tr></table>

Table 5: Numerical EPS beat/miss accuracy (%) on the same 96 scenarios per format. Mean score classifies the casewise average of $\mathrm { A } / \mathrm { B }$ scores; mean accuracy averages the two assignment-specific accuracies. Words score the natural beat/miss prefixes. The methods estimate distinct quantities.
<table><tr><td>Checkpoint</td><td>M+S</td><td>+E</td><td>+context</td><td>EPS gain</td><td>Held pair</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.160</td><td>0.160</td><td>0.661</td><td>-0.000515</td><td>0.107</td></tr><tr><td>Qwen2.5-1.5B</td><td>0.717</td><td>0.717</td><td>0.834</td><td>-0.000000</td><td>0.702</td></tr><tr><td>SmolLM2-1.7B</td><td>0.767</td><td>0.767</td><td>0.772</td><td>-0.000005</td><td>0.753</td></tr><tr><td>TinyLlama-1.1B</td><td>-0.045</td><td>-0.045</td><td>0.944</td><td>0.000033</td><td>-0.052</td></tr><tr><td>Phi-3.5-mini</td><td>0.417</td><td>0.424</td><td>0.324</td><td>0.007589</td><td>0.414</td></tr><tr><td>Qwen2.5-7B</td><td>0.019</td><td>0.829</td><td>0.820</td><td>0.810390</td><td>0.830</td></tr></table>

Table 6: Out-of-sample $R ^ { 2 }$ of linear surrogates of primary A/B-averaged scores. M+S uses momentum and tone; +E adds surprise; +context adds section order, commentary-pair and introduction indicators. First four numeric columns use leave-one-block-out fitting. Held pair uses M+S+E while leaving one of six commentary pairs out. Negative $R ^ { 2 }$ indicates worse performance than the pooled target-mean reference.

The bridge uses 48 pilot cases per repeated Qwen checkpoint, selected across task/control variants and the saved score range. It compares conversion, precision, inference kernel, and execution together, rather than isolating quantization. Its maximum diferences and flips are reported rather than assumed negligible. The material discrepancies prevent interpreting diferences between the pilot and expanded estimates as purely a consequence of more diverse scenarios. These stress cases are not a random estimate of conversion error or diagnostic accuracy. The expanded analysis uses only the new common-runtime records and does not pool the pilot’s full-precision outputs.

## 5 Interpretation and limits

On the primary A/B-averaged measure, the magnitude of the signed tone contrast exceeds that of the EPS contrast in 5 of six checkpoints; momentum exceeds EPS in 5. Using mean absolute matched changes instead, the corresponding counts are 5 and 5. The two summaries distinguish directional reliance from total score movement; both remain conditional on these supplied inputs and prompt formats.

![](images/e3b050b17a0babb5420554e7874d68292cdf3b6afa43a9a37dc450d9be799171.jpg)

![](images/17c14adf44364e543c4c306bbfa68a88374b9256f391c54c3a0b17820dcace52.jpg)  
Figure 3: Tone and EPS efects within each of six commentary pairs, using four numerical blocks per<sub>Figure</sub> <sub>3:</sub> <sub>Tone</sub> <sub>and</sub> <sub>EPS</sub> <sub>efects</sub> <sub>within</sub> <sub>each</sub> <sub>of</sub> <sub>six</sub> <sub>commentary</sub> <sub>pairs,</sub> <sub>using</sub> <sub>four</sub> <sub>numerical</sub> <sub>blocks</sub> <sub>per</sub> pair. The two stacked matrices have separate signed color scales and explicit pp values. Betweenpair variation can include the numerical blocks assigned to each pair.

Numerical recognition is heterogeneous. Qwen2.5-7B has the highest beat/miss word-prefix accuracy (95.8%), whereas SmolLM2-1.7B has 51.0%. The diagnostic has known answers; the forecast task has no observed return labels. High diagnostic accuracy therefore demonstrates performance on the numerical comparison task, not a profitable forecast or an identified internal reasoning mechanism.

Within the Qwen family, the 0.5B, 1.5B and 7B checkpoints have primary EPS contrasts of −0.15, −0.19, +86.80 pp and numerical word-prefix accuracies of 55.2%, 63.5%, 95.8%. These are descriptive comparisons of three trained checkpoints, not an isolated causal efect of parameter count.

The EPS contrast remains positive under both individual A/B assignments and U/D for 2 of six checkpoints (Phi-3.5-mini, Qwen2.5-7B). A positive sign across encodings is a limited robustness check: magnitudes, decision thresholds and requested-prefix mass can still difer substantially. A/B averaging cannot guarantee semantic invariance.

![](images/d8312198bed45171eb6b849ac75dc6757d597a92197ba11e1df27f31fec4f9f8.jpg)  
Figure 4: Numerical beat/miss diagnostic on 96 balanced scenarios. The dashed line marks 50%<sub>Figure 4: Numerical beat/miss diagnostic on 96 balanced scenarios. The dashed line marks 50%</sub> for this balanced task. Intervals on A/B-averaged and natural-word accuracy bootstrap 24 blocks; forecast accuracy.individual assignments are point estimates. The formats are distinct measurements; none is realized forecast accuracy.

Checkpoint M+S +E +context EPS gain Held pair<sub>A momentum-and-tone linear surrogate achieves held-block R-squared from −0.045 to 0.767;</sub> Qwen2.5-0.5B 0.160 0.160 0.661 -0.000515 0.107<sub>adding EPS changes it by −0.000515 to +0.810390. A two-cue account therefore fits some check-</sub> SmolLM2-1.7B 0.767 0.767 0.772 -0.000005 0.753<sup>points</sup> <sup>better</sup> <sup>than</sup> <sup>others.</sup> <sup>These</sup> <sup>statistics</sup> <sup>reproduce</sup> <sup>model</sup> <sup>scores</sup> <sup>and</sup> <sup>do</sup> <sup>not</sup> <sup>estimate</sup> <sup>predictive</sup> TinyLlama-1.1B -0.045 -0.045 0.944 0.000033 -0.<sub>value for actual market returns or exhaust nonlinear input dependence.</sub>

The controlled design identifies how specified inputs change answer-prefix scores for the tested conversion files and prompt formats. The expanded model and wording coverage makes cross-Table 6: Out-of-sample R of linear surrogates of primary A/B-averaged scores. M+S uses momen-<sub>checkpoint and format heterogeneity measurable. Meaningful conclusions require attention to the</sub> financial contrasts, numerical diagnostics, and answer-format controls together. A result that apwhile leaving one of six commentary pairs out. Negative <sup>2</sup> indicates worse performance than thepears under only one answer mapping cannot support a general claim about financial understanding. pooled target-mean reference.A small surprise contrast does not establish disregard for all fundamentals: actual EPS levels and annual changes remain fixed within the surprise intervention and vary between numerical blocks. Their broader influence is not separately identified by this primary estimand.

11<sub>Several limits remain. First, these are six open-weight instruction/chat checkpoints at 0.49–</sub> 7.61B parameters; no proprietary frontier model or dedicated long-deliberation model is tested. Second, all new inference uses Q8 0 conversions, with some community provenance; accuracy and input reliance at other precisions may difer. Third, the disclosures are synthetic and short. Tone is deliberately stripped of new facts, whereas real narrative often contains relevant information. Fourth, changing consensus holds reported fundamentals fixed but changes another numerical fact, so the experiment estimates the efect of that expectation intervention under the prompt, not a pure conceptual operation inside the model. Fifth, the 24 blocks and six authored pairs limit uncertainty and subgroup interpretation. Sixth, prefix conditioning, prompt order, and label preferences may afect scores even after averaging assignments. The positive/beat alternative is described first in every format; letter reversal does not test semantic-option-order bias. Seventh, no stock returns, portfolios, transaction costs, or financial performance are measured. Eighth, the numerical gap is 2% or 15% of absolute actual EPS, rather than of consensus; cents rounding can alter its exact fraction. Finally, we use a direct-answer protocol without eliciting or intervening on chain-of thought reasoning. The selected conversion checks show substantial output diferences, so a fullprecision replication is needed before treating these as precision-invariant properties of the named base models.

![](images/cf190c2f5bc8da37713e72eb4ba03e34a9614b436021189d4bf924682af35f97.jpg)  
Figure 5: Left: EPS efects after commentary removal, with block-bootstrap intervals. These cases<sub>Figure 5: Left: EPS efects after commentary removal, with block-bootstrap intervals. These cases</sub> use facts-first order only. Right: mean absolute score diferences induced by answer mapping and by changing A/B to UD on matched primary cases.

The evidence can support a statement about behavior on this benchmark, including patterns that vary by checkpoint and encoding. It cannot determine whether an LLM is “really reasoning” internally, or establish a market forecasting advantage. A next-stage study should freeze a prospective real-announcement sample, timestamp all information, add larger and diferent model families, and compare direct responses with deliberate reasoning and uncertainty-aware outputs under similarly complete encoding controls.

## 6 Reproducibility and validation

Public benchmark data and recorded scores are available in the Earnings Input Reliance Benchmark dataset<sup>1</sup>. The companion recorded-results explorer<sup>2</sup> displays saved evaluations and performs no live inference. A compact GitHub showcase<sup>3</sup> provides an illustrative recorded-results demo and selected case comparisons.

![](images/5f2f9c310af6ab928dd2cfe79dc5e3108628d5eb120c3120f3facd1fc46f5bb8.jpg)  
Figure 6: Leave-one-block-out linear score reproduction. Adding EPS tests the incremental value<sub>Figure 6: Leave-one-block-out linear score reproduction. Adding EPS tests the incremental value</sub> of that manipulated input to this specific linear surrogate; context features can capture authored template diferences.

The public dataset includes the benchmark prompts, numerical blocks, recorded predictions, matched primary comparisons, selected pilot-bridge records, and documentation of the scoring convention and conversion provenance. Model likelihoods are genuine local evaluations; no simulated model outputs or substituted heuristic predictions are used.

Independent validation reconstructs every score and prefix mass from saved log probabilities, verifies prompt/token hashes and candidate boundaries, checks exact semantic-cell coverage and numerical relations, and recomputes primary efects and diagnostic accuracies without the production analysis functions. Fresh uncached inference covers 12 cases per model spanning the primary experiment and controls. An initial runner read an unpopulated Python logits bufer; verification caught the empty scores, the reader was corrected to use native final-position logits, and all afected outputs were excluded and rerun. The exclusion receipt is preserved. The summary and figures are regenerated from completed valid records. These computational checks do not certify the absence of every possible methodological error.

## 7 Conclusion

Across six specified Q8 checkpoint conversions, the magnitude of the primary signed tone contrast exceeds that of EPS in 5 cases, while momentum exceeds EPS in 5. Those directional averages must be read alongside absolute matched changes, numerical diagnostic accuracy, and alternative encodings. The broader model and wording coverage reveals checkpoint-specific behavior rather than establishing a universal two-cue mechanism. The evidence measures input reliance under this protocol; it does not determine whether a model is internally reasoning or whether its forecast predicts market returns.

![](images/86d27991065ab7e2a289b5a8553a8d7b0bd99ecb674da1f2cc439343dbc37b0a.jpg)  
Figure 7: Identical saved pilot prompts under the old FP32 setup and the new Q8 conver-Cases are selected across task types and saved score ranges, not randomly. This is a joint setup<sup>sion/runtime. The dashed line marks equalit</sup>y <sup>and solid lines mark the conditional 0.5 threshold.</sup> comparison, not a pure quantization experiment.Cases are selected across task types and saved score ranges, not randomly. This is a joint setup comparison, not a pure quantization experiment.

## <sub>Only</sub>References

diferent training or an incorrect export. The measured outputs nevertheless show that these Q8Hugging Face Smol Models Research. SmolLM2-1.7B-Instruct: Oficial model card, 2025. conversions are not interchangeable for the tested scores. This reinforces the scope of the main<sub>URL https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct. Accessed 8 Octo-</sub> ber 2026; pinned checkpoint documentation supplied.

Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, together, rather than isolating quantization. Its maximum diferences and flips are reported ratherHwanil Choi, Soonyoung Lee, and Wonbin Ahn. FinStaR: Towards financial reasoning with time than assumed negligible. The material discrepancies prevent interpreting diferences betseries reasoning models, 2026. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2605.03460v5</sub>.

are not a random estimate of conversion error or diagnostic accuracy. The expanded analysis uses<sub>Alejandro</sub> <sub>Lopez-Lira</sub> <sub>and</sub> <sub>Yuehua</sub> <sub>Tang.</sub> <sub>Can</sub> <sub>ChatGPT</sub> <sub>forecast</sub> <sub>stock</sub> <sub>price</sub> <sub>movements?</sub> <sub>return</sub> predictability and large language models, 2025. URL https://arxiv.org/abs/2304.07619v6. Version 6, 28 October 2025.

Microsoft. Phi-3.5-mini-instruct: Oficial model card, 2024. URL https://huggingface.co/ microsoft/Phi-3.5-mini-instruct. Accessed 8 October 2026; pinned checkpoint documentation supplied.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models, 2025. URL https://arxiv.org/abs/2410.05229v2.

Qwen Team. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115v2.

Dong Shu, Yanguang Liu, Huopu Zhang, and Mengnan Du. FinCall-Surprise: A large scale mul timodal benchmark for earning surprise prediction, 2025. URL https://arxiv.org/abs/2510. 03965v1.

TinyLlama Project. TinyLlama-1.1B-Chat-v1.0: Oficial model card, 2024. URL https:// huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0. Accessed 8 October 2026; pinned checkpoint documentation supplied.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting, 2023. URL https://arxiv.org/abs/2305.04388.

Dimitris Vamvourellis and Dhagash Mehta. Reasoning or overthinking: Evaluating large language models on financial sentiment analysis, 2025. URL https://arxiv.org/abs/2506.04574v1.

Yang Zhang, Wenbo Yang, Jun Wang, Qiang Ma, and Jie Xiong. CAMEF: Causal-augmented multi-modality event-driven financial forecasting by integrating time series patterns and salient macroeconomic announcements, 2025. URL https://arxiv.org/abs/2502.04592v3.

Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models, 2021. URL https://arxiv.org/abs/2102.09690.

## A Complete authored commentary pairs

<table><tr><td></td><td>Pair Positive opinion</td><td>Negative opinion</td></tr><tr><td>1</td><td>This is an encouraging and impressive devel- This is a disappointing and worrying develop- opment; the outlook feels bright.</td><td>ment; the outlook feels bleak.</td></tr><tr><td>2</td><td>I feel optimistic about this announcement.</td><td>I feel pessimistic about this announcement.</td></tr><tr><td>3</td><td>My overall impression of this release is favor- My overall impression of this release is unfa- able.</td><td>vorable.</td></tr><tr><td>4</td><td>My reaction to this announcement is enthusi- My reaction to this announcement is discour- astic.</td><td>aged.</td></tr><tr><td>5</td><td>I welcome this announcement and feel confi- I regret this announcement and feel uneasy. dent.</td><td></td></tr><tr><td>6</td><td>This announcement leaves me upbeat about This announcement leaves me downbeat about the shares.</td><td>the shares.</td></tr></table>

Table 7: Six authored commentary pairs. The surrounding prompt explicitly states that the opinion contains no additional facts. These are wording variations from one construction process, not six independently sampled authors or validated human sentiment labels.

All three introductions and every numeric block are included in the public benchmark dataset linked in Section 6. Both section orders are scored for every primary factorial cell. Introduction and commentary assignment are balanced over blocks, not fully crossed within each block.

## B Additional numerical and format results

## C Exploratory precision and conversion investigation

The converter is taken from the source distribution of the same pinned llama-cpp-python version. The source-distribution digest, new GGUF digests, case-level results, and tensor comparison are supplied. First-party and rebuilt files can difer in storage conventions; the rebuilt tied-output model

<table><tr><td>Checkpoint</td><td>Momentum logit</td><td>Tone logit</td><td>EPS logit Neutral EPS (pp)</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.054</td><td>0.170</td><td>-0.006 -0.02</td></tr><tr><td>Qwen2.5-1.5B</td><td>0.783</td><td>1.413</td><td>-0.009 -0.23</td></tr><tr><td>SmolLM2-1.7B</td><td>0.326</td><td>0.331</td><td>-0.004 -0.31</td></tr><tr><td>TinyLlama-1.1B</td><td>0.047</td><td>0.042</td><td>-0.006 -0.04</td></tr><tr><td>Phi-3.5-mini</td><td>5.377</td><td>10.075</td><td>1.719 5.80</td></tr><tr><td>Qwen2.5-7B</td><td>3.468</td><td>3.524</td><td>15.635 96.86</td></tr></table>

Table 8: Marginal efects on the average of the two answer-assignment log-likelihood ratios, and the EPS score contrast after removing commentary. Logit efects use a diferent scale from the main probability-score contrasts. Neutral EPS uses all 24 blocks at facts-first order only.

omits the separate output tensor found in the stock file. No inference about training changes is made from file structure alone. The six cases are selected across observed bridge diferences and do not estimate a random quantization error distribution.
<table><tr><td>Checkpoint</td><td>|∆M|</td><td>|∆S|</td><td>|△E|</td><td>EPS 95th</td><td> $\mathrm { E P S } > 0 ~ ( \% )$ </td></tr><tr><td>Qwen2.5-0.5B</td><td>2.18</td><td>4.30</td><td>1.82</td><td>4.46</td><td>48.4</td></tr><tr><td>Qwen2.5-1.5B</td><td>16.55</td><td>30.39</td><td>1.35</td><td>3.26</td><td>50.0</td></tr><tr><td>SmolLM2-1.7B</td><td>6.27</td><td>6.47</td><td>0.86</td><td>2.07</td><td>42.7</td></tr><tr><td>TinyLlama-1.1B</td><td>1.09</td><td>1.00</td><td>0.60</td><td>1.38</td><td>44.3</td></tr><tr><td>Phi-3.5-mini</td><td>20.77</td><td>45.01</td><td>6.67</td><td>36.82</td><td>87.5</td></tr><tr><td>Qwen2.5-7B</td><td>10.69</td><td>10.94</td><td>86.80</td><td>99.99</td><td>100.0</td></tr></table>

Table 9: Mean absolute matched score changes (pp), comparing 192 positive/negative input pairs per factor after $\mathrm { A } / \mathrm { B }$ averaging. EPS 95th is the 95th percentile of absolute pair changes, not a confidence limit. $\mathrm { E P S > 0 }$ is the share of beat-minus-miss pairs with a positive score change. These exploratory summaries detect cancellation hidden by a small signed mean.

## D Exact prompt and candidate convention

The shared system instruction is: You are a financial analyst. Follow the requested answer format. Oficial model-specific chat templates wrap it and the user message. A primary user message includes the block introduction, all three disclosure sections, and:

Forecast the sign of the stock’s abnormal price reaction during the next trading day following this release. Answer exactly A for a positive reaction or B for a negative reaction. Output the requested answer only.

AB1 swaps the $\mathrm { A } / \mathrm { B }$ mapping; UD uses U/D. Words replace the candidate labels with positive/negative. The EPS task asks:

Did actual EPS beat or miss consensus? Answer exactly A for beat or B for miss.   
Output the requested answer only.

Diagnostic AB1 reverses $\mathrm { A / B } ;$ words use beat/miss. Candidate strings are appended directly to the template’s generation prefix, with no invented leading space. Their sufix token IDs are

<table><tr><td>Checkpoint</td><td>Momentum</td><td>Tone EPS surprise</td></tr><tr><td>Qwen2.5-0.5B</td><td>1.58 0.90</td><td>3.83 / 4.22 -0.29 /-0.01</td></tr><tr><td>Qwen2.5-1.5B</td><td>13.61 / 19.50</td><td>33.58 3 / 27.20 -0.42 / 0.05</td></tr><tr><td>SmolLM2-1.7B</td><td>6.82 /5.71</td><td>6.64 / 6.29 -0.18 / 0.03</td></tr><tr><td>TinyLlama-1.1B</td><td>0.90 /1.12</td><td>0.27 / 0.17 −0.10 / −0.10</td></tr><tr><td>Phi-3.5-mini</td><td>20.26 / 21.29</td><td>48.91 / 41.10 6.20 / 6.94</td></tr><tr><td>Qwen2.5-7B</td><td>8.94 / 12.37</td><td>10.77 / 8.91 86.88 / 86.72</td></tr></table>

Table 10: Input contrasts (pp) in facts-first / facts-last section order, using the same 24 blocks in each order. These exploratory estimates preserve all other input matching and do not change the primary order-averaged estimand.
<table><tr><td>Checkpoint</td><td>Loss</td><td>Profit</td><td>Decline</td><td>Rise</td><td>2% gap</td><td>15% gap</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.21</td><td>-0.51</td><td>-0.08</td><td>-0.22</td><td>-0.31</td><td>0.01</td></tr><tr><td>Qwen2.5-1.5B</td><td>-0.44</td><td>0.06</td><td>-0.48</td><td>0.10</td><td>-0.28</td><td>-0.10</td></tr><tr><td>SmolLM2-1.7B</td><td>-0.15</td><td>-0.00</td><td>-0.01</td><td>-0.15</td><td>-0.16</td><td>0.00</td></tr><tr><td>TinyLlama-1.1B</td><td>-0.15</td><td>-0.05</td><td>-0.15</td><td>-0.05</td><td>-0.02</td><td>-0.18</td></tr><tr><td>Phi-3.5-mini</td><td>2.50</td><td>10.65</td><td>6.08</td><td>7.06</td><td>4.79</td><td>8.35</td></tr><tr><td>Qwen2.5-7B</td><td>84.54</td><td>89.07</td><td>84.49</td><td>89.11</td><td>79.52</td><td>94.08</td></tr></table>

Table 11: Primary EPS contrasts (pp) within profitability, annual EPS change, and surprise-size subgroups. Every subgroup has 12 earnings blocks. These exploratory comparisons span diferent numerical blocks and may also reflect diferences in assigned wording; they do not isolate the causal efect of profitability or growth.

derived from the joint prompt-plus-candidate encoding. This matters especially for SentencePiece tokenizers. Likelihoods include each sufix token and exclude an end-of-answer token.

## E Artifacts and protocol chronology

The pilot was designed and run before this expansion and had already revealed answer-encoding sensitivity. The expanded primary protocol therefore incorporates controls suggested by that observed limitation. Its original 1,824 cases are internally timestamped before expanded results, not a pilot-independent public preregistration. The later 192-case control-coverage addition follows a documented audit of the first-block selection rule; it is not presented as preregistered confirmation. The public benchmark dataset retains the original frozen prompts, expanded scores, protocol, and coverage amendment. The new benchmark is analyzed as a separate dataset under one new runtime. No pilot hypothesis is relabeled as an unseen confirmation.

<table><tr><td>Checkpoint</td><td>Loss</td><td>Profit</td><td>Decline</td><td>Rise</td><td>2% gap</td><td>15% gap</td></tr><tr><td>Qwen2.5-0.5B</td><td>60.4</td><td>50.0</td><td>52.1</td><td>58.3</td><td>52.1</td><td>58.3</td></tr><tr><td>Qwen2.5-1.5B</td><td>52.1</td><td>75.0</td><td>62.5</td><td>64.6</td><td>64.6</td><td>62.5</td></tr><tr><td>SmolLM2-1.7B</td><td>50.0</td><td>52.1</td><td>45.8</td><td>56.2</td><td>52.1</td><td>50.0</td></tr><tr><td>TinyLlama-1.1B</td><td>56.2</td><td>52.1</td><td>52.1</td><td>56.2</td><td>54.2</td><td>54.2</td></tr><tr><td>Phi-3.5-mini</td><td>83.3</td><td>85.4</td><td>83.3</td><td>85.4</td><td>79.2</td><td>89.6</td></tr><tr><td>Qwen2.5-7B</td><td>91.7</td><td>100.0</td><td>95.8</td><td>95.8</td><td>91.7</td><td>100.0</td></tr></table>

Table 12: Natural beat/miss prefix diagnostic accuracy (%) within numerical subgroups; 48 scenarios and 12 blocks per subgroup. This known-answer task is distinct from return prediction. The numerical blocks difer between subgroups, so comparisons are descriptive.

<table><tr><td>Checkpoint</td><td>Momentum</td><td>Tone</td><td>EPS surprise</td></tr><tr><td>Qwen2.5-0.5B</td><td>5.31 / 1.07</td><td>28.79 /3.77</td><td>0.03 / -0.12</td></tr><tr><td>Qwen2.5-1.5B</td><td>33.40 /16.82</td><td>50.91 1 / 30.85</td><td>-0.14 / -0.51</td></tr><tr><td>SmolLM2-1.7B</td><td>19.99 6.23</td><td>18.83 /6.52</td><td>-0.11 / -0.06</td></tr><tr><td>TinyLlama-1.1B</td><td>0.12 0.95</td><td>2.31 / 0.18</td><td>-0.07 / -0.08</td></tr><tr><td>Phi-3.5-mini</td><td>23.56 /21.63</td><td>35.78 / 45.37</td><td>4.98 / 5.52</td></tr><tr><td>Qwen2.5-7B</td><td>10.37 / 12.95</td><td>9.26 / 13.38</td><td>85.01 / 83.38</td></tr></table>

Table 13: Natural positive/negative word-prefix efects / matched A/B-averaged efects (pp). Each format covers exactly the same 192 scenarios: one loss and one profitable numerical block for each commentary pair, both section orders. Profitable cases were added in the documented controlcoverage amendment. This test is descriptive and is not interchangeable with the 384-scenario primary estimand.

<table><tr><td>Checkpoint</td><td>AB0 mass</td><td>top</td><td>AB1 mass</td><td>top</td><td>UD mass</td><td>top</td></tr><tr><td>Qwen2.5-0.5B</td><td>98.2</td><td>100.0</td><td>98.4</td><td>100.0</td><td>98.0</td><td>100.0</td></tr><tr><td>Qwen2.5-1.5B</td><td>99.8</td><td>100.0</td><td>99.7</td><td>100.0</td><td>99.9</td><td>100.0</td></tr><tr><td>SmolLM2-1.7B</td><td>94.6</td><td>100.0</td><td>93.7</td><td>100.0</td><td>94.0</td><td>100.0</td></tr><tr><td>TinyLlama-1.1B</td><td>38.5</td><td>88.3</td><td>57.9</td><td>100.0</td><td>11.9</td><td>35.2</td></tr><tr><td>Phi-3.5-mini</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Qwen2.5-7B</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr></table>

Table 14: Primary requested-prefix mass and first-token compliance (%). Mass is the average unconditioned probability assigned to either requested prefix. Top is the share whose highestlogit next token starts a requested candidate. Both are distinct from semantic comprehension or calibrated market probability.

<table><tr><td>Pilot case</td><td>FP32</td><td>Rebuilt F16</td><td>Rebuilt Q8</td><td>Stock Q8</td></tr><tr><td>b02_m1_s-1_e1_t0_o0_labels0</td><td>55.76</td><td>55.09</td><td>66.96</td><td>57.88</td></tr><tr><td>b08_e1_diagnostic1_reversed</td><td>15.13</td><td>15.27</td><td>20.32</td><td>17.75</td></tr><tr><td>b00_m1_s-1_e-1_t0_o1</td><td>39.31</td><td>39.83</td><td>44.58</td><td>72.10</td></tr><tr><td>b10_e1_diagnostic1</td><td>80.62</td><td>80.59</td><td>85.73</td><td>47.32</td></tr><tr><td>b07_e1_diagnostic1</td><td>84.54</td><td>84.84</td><td>86.12</td><td>51.04</td></tr><tr><td>b00_e-1_diagnostic1</td><td>78.57</td><td>78.67</td><td>83.97</td><td>37.83</td></tr></table>

Table 15: Conditional positive/beat candidate scores (%) for six selected Qwen0.5 pilot stress cases. The rebuilt F16 and Q8 files use the same pinned HF source; stock Q8 is the downloaded first-party conversion used in the main experiment. Cases span tasks and are selected after observing bridge diferences; this is an exploratory precision/provenance investigation, not an error-rate sample or a new full-precision factorial experiment.