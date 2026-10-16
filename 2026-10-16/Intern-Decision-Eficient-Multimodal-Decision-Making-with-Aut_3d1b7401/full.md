# Intern-Decision: Eficient Multimodal Decision Making with Autoregressive Masked Language Modeling

<sub>Tong Zhu</sub>1 #<sub>, Zhe Xu</sub>1<sub>, Zefeng He</sub>1,2<sub>, Yunkai Chai</sub>1,2<sub>, Xuyang Hu</sub>1<sub>, Dehong Zheng</sub>1,3<sub>, Yun Luo</sub>1<sub>,</sub> <sub>Yafu Li</sub>1<sub>, Qi Zhang</sub>1,3 <sub>and Yu Cheng</sub>1,4 #

<sup>1</sup>Shanghai AI Lab, <sup>2</sup>Shanghai Jiao Tong University, <sup>3</sup>Fudan University, <sup>4</sup>Nanyang Technological University

<sup>#</sup> zhutong@pjlab.org.cn <sup>#</sup> chengyu@pjlab.org.cn

Interactive agents repeatedly choose among a finite set of actions, labels, or ordinal outcomes. Generating a free-form response for each choice adds latency and complicates the interface between a model and its environment. We present Intern-Decision, a family of 0.8B, 2B, and 4B multimodal decision models that return structured choices and candidate probabilities from text and optional images. Our autoregressive masked language modeling objective places one decision placeholder per field in a complete assistant JSON skeleton. Ground-truth symbols appear only in supervision, while all fields are evaluated in a single causal forward pass using the backbone’s native language-model head. The 4B model achieves a 90.02% unweighted average accuracy across seven evaluation suites, compared with 88.74% for Jev v1.13.0. For a fixed 289-token request with three fields on one RTX 4090, mean latency is 44.16 ms for the 4B model and approximately 33–34 ms for the smaller models. On a separate 96-case known-distribution pilot, temperature scaling reduces expected multiclass Brier from 0.628 to 0.550 and expected calibration error from 0.213 to 0.089. A confidence-based handof study on 111 public Jevbench-Hard items improves accuracy from 82/111 (73.87%) to 97/111 (87.39%) by routing 42 items to Atria-Dawn-Preview. We also describe qualitative game, visual interface, and browser demonstrations. These results support compact decision models as a practical component of larger agent systems.

§ Code: https://github.com/InternLM/Intern-Decision

Models: https://huggingface.co/collections/internlm/intern-decision

## 1. Introduction

Driven by the rapid progress of large language models (LLMs), agent systems have evolved from passive text generation models into interactive systems capable of perceiving environments, using tools, and taking actions [19, 30, 32]. Unlike open-ended text generation, many agent operations are inherently closed-set decision problems, where the model selects from a predefined set of alternatives, such as tool invocation, model routing, and actions [20]. However, current agents often rely on autoregressive free-form generation even when the underlying task only requires structured decisions, introducing additional decoding overhead and complicating the interface between models and downstream controllers [19, 25, 30]. These observations motivate the development of dedicated decision models that can directly produce structured outcomes together with reliable confidence estimates for eficient agent execution.

Recent work on structured decision models, referred to as System-1 decision models, explores this perspective by replacing unrestricted generation with explicit decision interfaces and probabilistic outputs. Systems such as Jev [27] provide typed interfaces for choice, score, and binary decisions, while several open implementations investigate alternative architectures and inference strategies for eficient structured decision making [1, 17, 22, 26]. These interfaces primarily target textual decisions, and implementations difer in whether they score multiple questions jointly or separately. For multimodal agents operating in dynamic environments, decision models must additionally integrate visual observations, task instructions, and changing interface states, where action selection depends jointly on visual evidence and contextual constraints [28, 29]. Moreover, practical agent requests often require multiple decisions over a shared state, creating a need for eficient multi-field decision models that can preserve the capabilities of pretrained multimodal language models while providing structured outputs.

To address these challenges, we introduce Intern-Decision, a family of multimodal decision models based on autoregressive masked language modeling. Our approach reformulates structured decision making as next-token prediction over masked decision fields within a predefined assistant output schema. By retaining the pretrained causal language model backbone and native language-model head, Intern-Decision evaluates multiple candidate decisions in a single forward pass and produces both structured predictions and candidate probabilities without iterative text generation. This design separates decision scoring from free-form response decoding, enabling eficient inference while preserving compatibility with existing multimodal language models.

To adapt pretrained models to this decision-oriented interface, we construct a unified multimodal decision corpus consisting of textual states, optional images, typed questions, and target decisions. The corpus combines teacher-constructed challenging decisions with public task distributions, covering textual reasoning, instructionfollowing decisions, visual judgments, interface actions, and interactive control scenarios. We instantiate Intern-Decision with 0.8B, 2B, and 4B Qwen3.5 backbones [23], fine-tuning the language backbone while keeping the vision tower and projector frozen. To improve the reliability of confidence estimates, we further apply post-hoc temperature scaling [13] using a separate calibration split.

We evaluate Intern-Decision from three perspectives: decision accuracy, inference eficiency, and probability quality. Across seven evaluation suites, including public Jevbench subsets [10], Intern-Decision-4B achieves a 90.02% unweighted mean accuracy, compared with 88.74% for Jev. For a fixed 289-token request containing three decision fields, Intern-Decision-4B achieves a mean latency of 44.16 ms on a single RTX 4090. We further evaluate probability calibration through a separate known-distribution pilot and study confidence-based model routing, in which requests are passed to a larger model when the decision model’s confidence falls below a threshold. Qualitative demonstrations on visual interaction and browser assistance further illustrate how compact decision models can serve as local decision modules within hierarchical agent systems.

Our contributions are summarized as follows:

• We introduce Intern-Decision, a family of compact multimodal decision models that formulate structured decision making as next-token-aligned masked prediction, enabling multi-field candidate scoring with a native language-model head in a single causal forward pass.

• We develop a unified multimodal decision framework, evaluate accuracy, latency, and calibration under separate protocols, and illustrate visual interaction through qualitative demonstrations.

• We investigate confidence-based model handof that passes low-confidence requests to a larger model, and demonstrate how compact structured decision models can complement larger models in interactive agent systems.

## 2. Related Work

Decision Models. Recent decision models target fast, structured judgments: given a state, questions, and predefined answer options, they return typed decisions with probability estimates rather than free-form text. Alongside TypeSafe’s Jev [27], which supports choice, score, and binary decisions and reports using Reinforcement Learning for Calibrated Decisions (RLCD), related open implementations explore diferent modeling and inference strategies for structured decision-making. JevK5 [1] adapts Qwen3.5 with LoRA and derives candidate probabilities from next-token logits through a Jev-style interface. Kev [17] combines Qwen backbones with LoRA adapters and a learned pointer head, providing self-hostable models and an API for typed choices and scores. SemIf-OpenJev [26] provides a direct-scoring approach that derives option probabilities from candidate-token logits of frozen language models, with optional shared-state prefill reuse across questions. Laya [22] adopts an encoder-based, non-autoregressive architecture to produce choice, score, and yes/no decisions in a single forward pass. Within this paradigm, Intern-Decision focuses on multimodal structured decision-making, incorporating visual inputs alongside textual states and task instructions.

Autoregressive and Masked Prediction. Autoregressive language models such as GPT-3 [5] predict tokens conditioned on a causal prefix, whereas BERT [8] reconstructs masked tokens using bidirectional context. Hybrid approaches explore combinations of these paradigms: UniLMv2 [3] combines autoencoding and partially autoregressive objectives through pseudo masking, while GLM [9] combines bidirectional encoding of the corrupted input with autoregressive reconstruction of masked spans. Mask-Predict [12] predicts masked target tokens in parallel and iteratively refines uncertain predictions. For classification, PET [24] uses pattern–verbalizer pairs for cloze-based prediction and distills the resulting models into a sequence classifier, while LM-BFF [11] performs prompt-based fine-tuning with label words and demonstrations, reusing the pretrained masked-language-model head. Intern-Decision formulates structured decisions as autoregressive masked language modeling, combining answer placeholders with next-token-aligned prediction in a causal multimodal model. Its native language-model head scores candidates for all requested fields in a single forward pass, without iterative refinement or answer feedback.

Calibration and selective computation. Temperature scaling fits a positive scalar to adjust predictive confidence while preserving the most likely label [13]. Brier scoring evaluates the full predictive distribution [4], complementing accuracy and binned calibration error. Confidence-based routing and cascading use uncertainty to allocate requests between models with diferent costs [6, 7]. Our evaluation separates hard-label calibration from an exact-distribution pilot and reports one escalation operating point rather than an optimized routing policy.

## 3. Autoregressive Masked Decision Modeling

## 3.1. Problem formulation

Let $x \ = \ ( s , I , Q )$ denote a request with state $s ,$ optional ordered images �, and ordered questions $Q =$ $\left( q _ { 1 } , \ldots , q _ { m } \right)$ . Question $q _ { j }$ has a field identifier, an instruction, and an ordered candidate set $\mathcal { C } _ { j }$ . The model returns a categorical distribution $P _ { j }$ and a selected label ${ \hat { y } } _ { j } \in { \mathcal { C } } _ { j }$ for every field. The supported types are choice, noul, and score. Choice labels follow the insertion order of their criteria; Boolean labels are ordered as no, yes; ordinal labels are strings such as 0, 1, and 2.

A deterministic mapping $a _ { j } : { \mathcal { C } } _ { j } \to \mathcal { V }$ assigns each option a legal single-token answer symbol from the tokenizer vocabulary �. The release supports 62 candidate symbols; the HTTP service permits up to 16 fields per request. These are interface limits rather than claims about generalization to arbitrarily large schemas. Field and option order remain unchanged during compilation and scoring.

## 3.2. Input construction and causal readout

The released compiler places the state and ordered decision schema in the user turn, then appends an assistant JSON skeleton with one reserved <decision> token per field. Figure 1 presents a readable view of a constructed support-ticket request with choice, score, and noul fields. Choice symbols follow criteria order, score symbols follow list order from $^ { 0 , }$ and Boolean symbols are A for no and B for yes. The target panel shows the mapped answer symbols A, C, and B, corresponding to billing, urgency 2, and yes. Training supervises these symbol tokens at the decision positions; inference maps predicted symbols back to the original labels. The displayed JSON groups the supervision targets for illustration and is not inserted into the input or supervised as a complete generated JSON response.

Labels, reference distributions, benchmark rationales, and provenance are excluded from the visible evidence. Training and inference use the same masked skeleton, and changing a target must not change the input token sequence.

Let $p _ { j }$ be the token position of the placeholder for field $j$ . A causal backbone computes hidden states $h _ { 1 } , \ldots , h _ { L }$ , and the native LM head produces vocabulary logits $z _ { t } .$ The released objective reads the position immediately before the marker:

$$
z _ { p _ { j } - 1 } = \mathrm { L M H e a d } ( h _ { p _ { j } - 1 } ) .\tag{1}
$$

The answer label is stored at position $p _ { j }$ in the supervision tensor, so the ordinary causal shift pairs it with $z _ { p _ { j } - 1 }$ . No ground-truth answer symbol is inserted into the input. Earlier placeholders are visible to later fields

Customer message: I bought headphones yesterday and noticed today that I was charged twice. Please refund the extra charge as soon as possible.

![](images/f09a5542505abe6f168dfa057a820a6a95a58e145699cd5baedfeeb568dcd097.jpg)  
Figure 1: A constructed three-field request. Blue panels show visible evidence and masked assistant context; gold shows target answer symbols: A for billing, C for urgency 2, and B for yes. These symbols supervise the decision positions and never enter the input. System instructions, control tokens, and some option descriptions are omitted or shortened.

through causal attention, but their gold or predicted answers are not fed back. All requested logits are collected from the same forward pass.

![](images/6b96784374b96e5bbfaae3df13c24e1a0393d52e53824a94967e04f3ce76f7c7.jpg)  
Figure 2: Intern-Decision inference. All fields share the same state and schema, and their placeholders are processed together. The readout uses next-token alignment, not same-position supervision.

## 3.3. Training objective

For a batch with � supervised fields, the loss is full-vocabulary cross-entropy at the shifted decision positions:

$$
\mathcal { L } _ { \mathrm { d e c i s i o n } } = - \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \log \frac { \exp z _ { p _ { j } - 1 , a _ { j } ( y _ { j } ) } } { \sum _ { v \in \mathcal { V } } \exp z _ { p _ { j } - 1 , v } } .\tag{2}
$$

All other labels are ignored. Thus JSON punctuation, state tokens, and schema descriptions are context rather than supervised output targets. The objective is neither full-JSON generation loss nor full-token continued pretraining. The training path uses hard labels.

## 3.4. Candidate probabilities and structured outputs

At inference, only the legal candidate logits for each question are normalized:

$$
P _ { j } ( c \mid x ; T ) = \frac { \exp ( z _ { p _ { j } - 1 , a _ { j } ( c ) } / T ) } { \sum _ { c ^ { \prime } \in { \mathcal C } _ { j } } \exp ( z _ { p _ { j } - 1 , a _ { j } ( c ^ { \prime } ) } / T ) } , \qquad \hat { y } _ { j } = \underset { c \in { \mathcal C } _ { j } } { \arg \operatorname* { m a x } } P _ { j } ( c \mid x ; T ) .\tag{3}
$$

A positive scalar � changes the probability spread without changing the argmax, apart from implementationlevel numerical ties. The serializer maps symbols back to original labels and creates the response object. Consequently, schema-conforming output does not depend on a free-text decoder generating valid JSON. It also does not guarantee semantic correctness or agreement between related fields.

The service can return the probability of yes for Boolean questions and an expected numeric value for ordinal questions. These quantities difer from the discrete argmax label used for decision accuracy. For all types, candidate probabilities and maximum probability remain available to downstream controllers.

## 3.5. Computational scope

The method removes iterative answer decoding and avoids separate field-by-field forward passes. It does not eliminate the cost of processing the input: longer states, more image tokens, and larger schemas still increase computation and memory. One request may yield multiple decisions, so request rate and decision rate are diferent quantities. Causal ordering also means fields can have diferent contextual prefixes even when they share a forward pass; the construction does not enforce permutation invariance or a jointly constrained output distribution.

## 4. Data Construction

## 4.1. Decision-oriented data construction

The decision corpus teaches structured choices rather than free-form answers. It combines synthetic textual decisions, public task distributions, and synthetic visual and interactive tasks. Each example contains a visible state, optional images, typed questions, and target labels, allowing these sources to share the same decision interface.

Table 1: Training dataset composition.
<table><tr><td>Component</td><td>Representative sources / tasks</td><td>Capability focus</td></tr><tr><td>Synthetic text decisions</td><td>General and Hard decisions synthesized by deepseek-v4.1-flash</td><td>Evidence use, rule following, and difficult textual judgments</td></tr><tr><td>Public task distributions</td><td>AG News [31], ToolACE [19], Typed Deci- sions [21], WildGuardMix [14], WildJail- break [18], and Ruozhiba (LLM Ruozhiba QA and COIG-CQIA) [2, 15]</td><td>Classification, tool routing, typed decisions, safety, and reasoning</td></tr><tr><td>Synthetic visual and inter- Game environments, VizDoom, and active tasks</td><td>shapes/captcha-style tasks</td><td>Visual grounding, discrete control, and pointer-action selection</td></tr></table>

The released 0.8B, 2B, and 4B checkpoints use the same training input: 17,880 samples and 23,500 decisions, including 5,500 image-bearing samples. This identity is verified against input hashes, training logs, and tokenizer caches for all three runs. Figure 3 shows three components and nineteen categories, with the nine games listed individually. Sector areas encode samples, so a multi-question record contributes once even when it supplies several supervised decisions.

The synthetic text component uses deepseek-v4.1-flash 1 as the teacher model. As Figure 3 shows, General contains 4,066 generalcategory synthetic decisions; Hard combines the hard and patch batches into 2,645 dificult-category synthetic decisions. We follow the Jevbench ability type to construct such general and hard synthetic decision examples. Public sources cover news classification, tool routing, typed decisions, and safety judgments. Synthetic visual sources supply rendered game frames, VizDoom observations, and shapes-based pointer actions. Appendix B provides component counts, source descriptions, and representative records.

The visual sources require recognition, comparison, target localization, and discrete action selection. Shapes tasks connect object attributes and spatial relations to pointer movements; game and VizDoom tasks connect frames to control decisions. These capabilities overlap within the recorded sources. Table 1 summarizes the evidenced source groups and their capability focus.

## 4.2. Calibration test set construction

This evaluation-only test set measures whether the model assigns appropriate probabilities to all possible outcomes of a stated random experiment. For example, a fair die requires probability 1/6 on each real face and zero on impossible faces. Selecting one plausible face does not establish a correct predictive distribution. Similarly, the Monty Hall questions ask where the prize is, with doors as outcomes; they do not ask whether switching is the best action.

The set is constructed from 24 mechanism families in six categories. Each family has two settings, which vary numerical parameters or the observation protocol. Each setting is emitted with canonical and reversed option order, yielding $6 \times 4 \times 2 \times 2 = 9 6$ cases, 16 per category. The two order variants share the same state,

![](images/275fb7b07f568148c3e59d89fce8b40b78562780b39da71533c31d5586c7cc5f.jpg)  
Figure 3: Training mixture shared by the released models. The inner ring shows components; numbered outer sectors show data categories, including each game. Counts are training-input samples.

question, and reference distribution. There are therefore 48 paired settings rather than 96 independent trials.   
Appendix C gives the family inventory and concrete input/reference examples.

For each setting, a deterministic Python generator specifies the sample space, priors, independence assumptions, and observation or selection mechanism. It computes the exact distribution using rational arithmetic: counting and enumeration for finite outcomes, conditional-probability formulas for observed evidence, a forward recursion for hidden states, and combinatorial or dynamic-programming calculations for sequential processes. References are not Monte Carlo estimates or model-generated answers. Construction checks enforce probabilities in [0, 1], an exact sum of one, matching option keys, unique IDs, and equivalent reversed-order pairs. Rebuilding from the archived generator reproduces the evaluated input and reference files byte for byte.

Inputs contain only the state and typed questions. Exact fractions, numerical reference probabilities $q _ { i } ,$ derivations, and provenance remain in a separate reference file and are never sent to the model. The set is not used to train the model, fit its temperature, or select a checkpoint. An exact-content check found no overlap with the temperature-fitting records; this check does not establish absence of semantic overlap with every training source.

In the actual three-door informed-host case, the prize is uniformly placed and the player initially selects door 1. A host who knows the prize location always opens an unchosen empty door and always ofers the other unopened door; legal reveals are chosen uniformly. The input does not disclose the opened door’s number. The question is “Where is the prize relative to the final two unopened doors?” Its outcomes are initial\_door and other\_unopened\_door, with reference probabilities 1/3 and $2 / 3 .$ The initial selection retains its prior probability, and the host’s guaranteed safe reveal concentrates the other $2 / 3$ on the remaining door. An output concentrated entirely on the remaining door would select the more likely outcome but misrepresent the residual uncertainty.

## 5. Evaluation Results

## 5.1. Benchmark scope and scoring

Table 2 lists the evaluated bundles. Accuracy is computed per decision, not per row; this matters for Typed Decision, whose 400 records contain 2,000 fields. The seven-suite average is an unweighted mean of suite accuracies, so each suite contributes equally despite their diferent sizes. It is not pooled accuracy over all 12,351 decisions.

Table 2: Accuracy evaluation scope. Counts refer to the released test bundle, not the full upstream datasets.
<table><tr><td>Suite</td><td></td><td>Rows Decisions Evaluation focus</td><td></td></tr><tr><td>Jevbench-Easy</td><td>48</td><td></td><td>48 Explicit facts and direct decisions</td></tr><tr><td>Jevbench-Original</td><td>72</td><td></td><td>72 Policy, routing, and typed judgments</td></tr><tr><td>Jevbench-Hard</td><td>111</td><td></td><td>111 Longer contexts and interacting rules</td></tr><tr><td>Typed Decision</td><td>400</td><td></td><td>2,000 Multiple typed fields per state</td></tr><tr><td>ToolACE</td><td>310</td><td></td><td>310 Tool selection</td></tr><tr><td>AG News</td><td>7,600</td><td></td><td>7,600 News classification</td></tr><tr><td>WildJailBreak</td><td>2,210</td><td>2,210</td><td>Safety-related classification</td></tr><tr><td>Total</td><td>10,751</td><td>12,351</td><td>Seven separately weighted suites</td></tr></table>

The public Jevbench subset contains 231 items across Easy, Original, and Hard [10]. It excludes private items and the separate judge tier. Its oficial leaderboard combines multiple axes; the accuracy averages in this report do not reproduce that composite. Public Jevbench has informed development and should be treated as a diagnostic, not independent held-out generalization evidence. Input identities, labels, option order, and upstream scoring definitions are preserved.

For a hard-label target $y _ { i }$ , the multiclass Brier score [4] is

$$
\mathrm { B S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k \in \mathcal { C } _ { i } } \left( p _ { i k } - \mathbb { k } \mathbb { \epsilon } [ k = y _ { i } ] \right) ^ { 2 } .\tag{4}
$$

This is a sum over classes, including both terms for a binary question. Hard-benchmark ECE uses confidence $c _ { i } = \operatorname* { m a x } _ { k } p _ { i k }$ and 10 equal-width bins:

$$
\mathrm { E C E } = \sum _ { b = 1 } ^ { 1 0 } \frac { | B _ { b } | } { N } \left| \frac { 1 } { | B _ { b } | } \sum _ { i \in B _ { b } } c _ { i } - \frac { 1 } { | B _ { b } | } \sum _ { i \in B _ { b } } \mathcal { H } [ \hat { y } _ { i } = y _ { i } ] \right| ,\tag{5}
$$

with empty bins omitted. Brier and ECE beside the accuracy results are computed only on the 111 Hard items. They are not seven-suite averages. Public Hard also has 10 exact-distribution items used for TVD by the upstream scorer; this small subset is distinct from the 96-case pilot below.

Table 3: Decision accuracy in percent. Average is the unweighted mean of the seven suites. Denominators are given in Table 2; bold data values mark column maxima, including ties. The comparison is restricted to these evaluated systems and protocols.
<table><tr><td>Model</td><td>Easy</td><td>Original</td><td>Hard</td><td>Typed</td><td>ToolACE</td><td>AG News</td><td>WildJB</td><td>Average</td></tr><tr><td>Jev</td><td>100.00</td><td>98.61</td><td>72.07</td><td>73.35</td><td>91.29</td><td>89.57</td><td>96.29</td><td>88.74</td></tr><tr><td>Laya</td><td>95.83</td><td>72.22</td><td>28.83</td><td>35.95</td><td>63.87</td><td>92.84</td><td>14.84</td><td>57.77</td></tr><tr><td>SemIf</td><td>100.00</td><td>98.61</td><td>61.26</td><td>62.80</td><td>85.16</td><td>89.22</td><td>92.53</td><td>84.23</td></tr><tr><td>Kev</td><td>100.00</td><td>93.06</td><td>45.05</td><td>65.60</td><td>87.42</td><td>89.82</td><td>75.97</td><td>79.56</td></tr><tr><td>JevK5</td><td>100.00</td><td>97.22</td><td>73.87</td><td>64.50</td><td>80.97</td><td>89.13</td><td>90.45</td><td>85.16</td></tr><tr><td>Intern-Decision-0.8B</td><td>97.92</td><td>80.56</td><td>52.25</td><td>77.35</td><td>94.52</td><td>88.61</td><td>64.48</td><td>79.38</td></tr><tr><td>Intern-Decision-2B</td><td>100.00</td><td>84.72</td><td>63.96</td><td>79.35</td><td>96.45</td><td>89.96</td><td>78.33</td><td>84.68</td></tr><tr><td>Intern-Decision-4B</td><td>100.00</td><td>98.61</td><td>73.87</td><td>80.55</td><td>96.45</td><td>90.82</td><td>89.86</td><td>90.02</td></tr></table>

## 5.2. Decision accuracy

Intern-Decision-4B has a suite average of 90.02%, exceeding the baselines in Table 3, and leads or ties on five of the seven suites. It scores 1,611/2,000 (80.55%) on Typed Decision and 299/310 (96.45%) on ToolACE. On public Jevbench, it answers 48/48 Easy, 71/72 Original, and 82/111 Hard items correctly. Jev leads WildJailBreak, Laya leads AG News, and JevK5 ties the 4B model on Hard. These results describe the evaluated systems; without matched-data ablations, they do not isolate the efect of joint field scoring or model size.

Within Intern-Decision, the suite average rises monotonically from 79.38% at 0.8B to 84.68% at 2B and 90.02% at 4B, with Typed Decision and ToolACE already comparatively strong at 0.8B. Without matched data and training-compute disclosures or ablations, this pattern should not be attributed solely to parameter count.

Table 4: Jevbench-Hard probability metrics on 111 items. ECE uses maximum candidate probability and 10 equal-width bins. Intern-Decision uses separately fitted temperatures; baseline probabilities use their reported/default settings. Temperatures are not comparably refitted for every baseline.
<table><tr><td>Model</td><td>Brier ↓</td><td>ECE↓</td><td>Temperature</td></tr><tr><td>Jev</td><td>0.358420</td><td>0.094685</td><td>Reported/default</td></tr><tr><td>Laya</td><td>0.804244</td><td>0.246455</td><td>Reported/default</td></tr><tr><td>SemIf</td><td>0.498012</td><td>0.112213</td><td>1.000000</td></tr><tr><td>Kev</td><td>0.738253</td><td>0.262200</td><td>2.962195</td></tr><tr><td>JevK5</td><td>0.366162</td><td>0.046672</td><td>1.532000</td></tr><tr><td>Intern-Decision-0.8B</td><td>0.529519</td><td>0.065710</td><td>2.747761</td></tr><tr><td>Intern-Decision-2B</td><td>0.437256</td><td>0.100208</td><td>2.100509</td></tr><tr><td>Intern-Decision-4B</td><td>0.346762</td><td>0.065267</td><td>1.992418</td></tr></table>

The 4B model has the lowest Hard Brier in this comparison, whereas JevK5 has the lowest Hard ECE. Figure 4 compares the released 4B checkpoint before and after temperature scaling with Jev 1.13.0 on the same 111 items.

Temperature scaling leaves the 4B model’s 82/111 correct decisions unchanged while reducing ECE from 0.132613 to 0.065267. Jev answers 80/111 correctly and has ECE 0.094685. The temperature was fitted on separate calibration data; these Hard items were not used to choose it.

Each point reports mean maximum-candidate confidence and empirical accuracy within one of ten equalwidth bins. Empty bins are omitted. The curves use archived prediction statistics rather than digitized image coordinates. Connecting lines are descriptive: points summarize diferent bins and do not trace the same items across calibration settings.

The bins have unequal sample counts, including one Jev bin with a single item. Accordingly, the plot supports an aggregate calibration comparison but does not establish reliable behavior at every confidence level. No bin-level uncertainty intervals are available.

## 5.3. Local inference latency

Local measurements use one RTX 4090, BF16, SDPA, the native Hugging Face backend, Transformers 5.14.1, and flash-linear-attention 0.4.2. Each request has 289 input tokens and three fields: one choice, one Boolean, and one ordinal question. All three are answered in one forward pass with checkpoint-specific temperature scaling. This is a fixed short-request measurement, not a latency sweep over long contexts or screenshots.

![](images/f44e2d63001e197c375ee0bb1990bf0201944425972d5c9412951f3fd5a5b024.jpg)  
Figure 4: Jevbench-Hard reliability: 111 items and ten equal-width bins. Only populated bins are shown.

Table 5: Reported request latency in milliseconds. Ratios are Jev mean divided by model mean. The supplied protocol establishes the local Intern-Decision configuration; Jev’s complete serving and transport conditions still require documentation.
<table><tr><td>Model</td><td>Mean</td><td>P50</td><td>P95</td><td>Mean ratio</td></tr><tr><td>Jev</td><td>109.70</td><td>106.30</td><td>146.70</td><td>1.00×</td></tr><tr><td>Intern-Decision-0.8B</td><td>33.98</td><td>33.44</td><td>37.50</td><td>3.23×</td></tr><tr><td>Intern-Decision-2B</td><td>33.28</td><td>33.15</td><td>33.55</td><td>3.30×</td></tr><tr><td>Intern-Decision-4B</td><td>44.16</td><td>44.03</td><td>44.60</td><td>2.48×</td></tr></table>

The reciprocal mean latencies imply approximately 29.43, 30.05, and 22.64 serial requests per second for 0.8B, 2B, and 4B, respectively, assuming negligible additional loop overhead. These are derived request rates rather than measured saturated serving throughput. They do not imply the same frame rate for an interactive agent, which also incurs observation, image preprocessing, action execution, and environment-update costs. Batch size, warm-up, repetition count, timing boundaries, and Jev transport conditions are needed before interpreting the ratios as controlled hardware speedups.

## 5.4. Known-distribution calibration pilot

The known-distribution calibration test set constructed in Section 4.2 contains 96 cases across six categories. It evaluates complete outcome distributions against exact reference probabilities.

The question wording requests an outcome, while the native decision interface returns a probability for every listed option. Intern-Decision maps the ordered options to legal answer tokens and applies the candidate softmax in Equation 3; Jev supplies its native API probability map. We score these complete distributions, rather than converting a selected label into a one-hot vector or counting how often repeated sampled answers occur. Normalization guarantees a probability vector, but correctness must be measured against �<sub>�</sub>.

The final 4B checkpoint is evaluated with a fixed XTuner backend and � = 1.992418, fitted on separate calibration records. Jev uses version 1.13.0. Temperature scaling reduces overall expected Brier by 0.078 and expected ECE by 0.124 while preserving every argmax. The calibrated model improves on its uncalibrated result in both metrics in all six categories. Relative to Jev, it has lower Brier in five categories and lower ECE in five categories; it is worse on Brier for selective disclosure and on ECE for daily evidence. Overall ECE difers from the mean category ECE because the cases are rebinned together.

Table 6: Known-distribution pilot results. Each cell is expected Brier / expected ECE; bold marks the lowest value across the three settings for each metric separately. All settings have 96/96 valid predictions. Category rows have 16 cases each. Overall ECE is recomputed on the pooled cases, not averaged over categories.
<table><tr><td>Category</td><td>Jev</td><td>4B uncalibrated</td><td>4B calibrated</td></tr><tr><td>Direct randomness and support</td><td>0.490 / 0.216</td><td>0.483 ／0.181</td><td>0.421／0.129</td></tr><tr><td>Composed events and mixtures</td><td>0.682/0.274</td><td>0.677 / 0.254</td><td>0.577 / 0.150</td></tr><tr><td>History, conditioning, hidden state</td><td>0.657 / 0.113</td><td>0.711/ 0.219</td><td>0.613 / 0.108</td></tr><tr><td>Daily evidence and observation bias</td><td>0.603 / 0.114</td><td>0.701/0.328</td><td>0.575 / 0.210</td></tr><tr><td>Selective disclosure and puzzles</td><td>0.483 / 0.138</td><td>0.540 / 0.119</td><td>0.510 / 0.049</td></tr><tr><td>Sequential and combinatorial processes</td><td>0.657 / 0.116</td><td>0.656 / 0.180</td><td>0.605 / 0.058</td></tr><tr><td>Overall pooled</td><td>0.595 / 0.130</td><td>0.628／0.213</td><td>0.550 / 0.089</td></tr></table>

For the informed-host three-door case described in Section 4.2, calibrated Intern-Decision returns (0.4222, 0.5778) and Jev returns (0.18, 0.82) in that order. Both select the more likely door, while their total variation distances from the reference are 0.0889 and 0.1533, respectively. The uncalibrated 4B distribution (0.3486, 0.6514) is closer still, so the overall gain from temperature scaling does not imply improvement on every case. Jev also returns a separate confidence of 0.63 here, which is the value used by the pilot ECE protocol. Changing the observation mechanism changes the target: an uninformed host who randomly opens a door, conditioned on revealing no prize, yields (1/2, 1/2). Appendix C also explains a selective-ofer variant.

## 5.5. Confidence-based model handof

Let �(�) be a scalar confidence from the small model. A simple cascade retains its answer when $c ( x ) \geq \tau$ and invokes a larger model otherwise. The reported study routes 42 of the 111 public Hard items to Atria-Dawn-Preview [16], leaving 69 items with Intern-Decision-4B. The routed fraction is 37.84%, and local coverage is 62.16%.

Table 7: Public model handof study on Jevbench-Hard. The hybrid time is an estimate, while the Atria-only total is reported as measured. The local total is derived from 111 times the fixed-request mean latency and is not a per-item Hard timing trace.
<table><tr><td>System</td><td>Correct / 111 Accuracy</td><td></td><td>Atria calls Total time</td><td></td></tr><tr><td>Intern-Decision-4B</td><td>82</td><td>73.87%</td><td></td><td>0 4.90 s, derived</td></tr><tr><td>Confidence-based handoff</td><td>97</td><td>87.39%</td><td></td><td>42 140.16s, estimated</td></tr><tr><td>Atria-Dawn-Preview only</td><td>99</td><td>89.19%</td><td></td><td>111 300.03 s, measured</td></tr></table>

The hybrid answers 15 more items correctly than the small model and two fewer than Atria alone. Relative to the reported Atria-only total, its estimated time is 53.29% lower. This illustrates a potentially useful operating point rather than an established optimal policy.

## 6. Multimodal Interaction and Model Collaboration

## 6.1. A closed-loop decision interface

In an interactive controller, each iteration observes an environment state, forms candidate actions, evaluates them, executes the selected action, and observes the result. Intern-Decision supplies the local choice within this loop. The caller defines the action space and determines when to request another decision. The follow-

ing demonstrations illustrate this interface; they are selected qualitative trajectories rather than controlled benchmark measurements.

## 6.2. Visual game control

Five described demonstrations span Snake, fruit catching, Mario, and two first-person Doom tasks. They require diferent local decisions: selecting a safe direction toward food, distinguishing fruit from hazards, coordinating movement and jumping, searching for health, or targeting an enemy. Screenshots provide visual evidence directly, so the controller can operate without converting every game state into a symbolic text description.

![](images/3a0b5bf134f2e2bac67f0a439e4fde21622d2172a1fb79aae865edd759f52f6e.jpg)  
(a) Snake

![](images/7e20a9e6f19b78c77ce723e025035a1d21f800b5b5efe6602348a7254c06bf74.jpg)  
(b) Fruit catching

![](images/0faf24d6a530a3f7e2853118303781f9253d1d357accfa83ec3308f379743ee2.jpg)  
(c) Mario

![](images/b01e4ea329d182459a93053384d90e4aba1860fb44c0c33f60d92f289a4ee1d7.jpg)  
(d) Doom: Health gathering

![](images/58cff628df8c9365b3f2b759ce0f1537804f6da16715816e404b0b139c5f9faa.jpg)  
(e) Doom: Defend the line  
Figure 5: Qualitative examples of visual game control with Intern-Decision. From left to right: navigating toward food while avoiding the snake body; catching fruit while avoiding bombs; moving and jumping through a Mario level; searching for health packs; and firing at approaching enemies. Each column shows two consecutive video frames, with the earlier frame on top and the later frame below.

In the Doom health-gathering example, the agent changes its view after contacting a wall and subsequently searches for new targets. This is an observed behavior in a trajectory; it is not evidence of a learned exploration objective or a general exploration algorithm. Episode counts, seeds, returns, action frequency, observation history, and comparison policies are required for quantitative conclusions about game competence. The short text-request timing in Table 5 does not establish visual-control frame rate.

## 6.3. Visual selection and mouse actions

CAPTCHA-style visual tasks require the integration of visual perception, instruction understanding, and interactive decision making. Intern-Decision directly observes interface screenshots, identifies targets described in natural language, and completes tasks through discrete mouse movements and clicks. The model must interpret object attributes and their relationships, translate visual judgments into executable actions, and use the updated screenshot after each action to guide the next decision, forming a closed loop between perception and execution.

![](images/e77a3b09a4fbd0f558d49ca878bc43720c622386c61dc38a1e63a769b807dd17.jpg)  
Figure 6: Visual selection trajectories rendered as static video strips. Each row contains five chronologically ordered frames: (a) selecting the largest red sphere; (b) selecting the smallest red cube; (c) identifying the odd shape; and (d) selecting the second blue object from the left. Arrows indicate temporal progression, with intermediate steps omitted. Panels preserve the recorded object layouts and cursor positions; interface text is translated into English for presentation, and video presentation overlays are omitted.

Four demonstrations record successful trajectories in which Intern-Decision autonomously completes the tasks; Figure 6 presents selected keyframes in chronological order. Guided by the task instruction and the current screenshot, the model progressively moves the pointer, adjusts its position, and clicks the target without human takeover or correction. These examples illustrate the integration of multimodal understanding with sequential action selection and provide a concrete experimental setting for studying visual interaction and task execution in graphical interfaces.

## 6.4. Hierarchical browser assistance

We study browser use as hierarchical, closed-loop decision making rather than end-to-end free-form generation. Given a natural-language goal, a larger planner maintains cross-page context and proposes a stage-level subgoal. At each iteration, the current page evidence and a finite candidate set (e.g., click, scroll, return, or stop) are presented to Intern-Decision; the selected action is executed, and the page is observed again. The user task contains neither element identifiers nor a scripted click sequence, so local actions are chosen from the live interface.

![](images/fda6395b9e5a57b4254b781501777583c0e2b2a85b75d2814e6092a734be4ec1.jpg)  
Figure 7: Browser-use trajectories rendered as static video strips. Each row contains five chronologically ordered frames: (a) Hugging Face checkpoint and configuration verification; (b) GitHub repository and API-reliability investigation; and (c) PyTorch documentation and pinned-source inspection. Arrows indicate temporal progression. All panels show the browser interface itself without the presentation overlays used in the demo videos.

Figure 7 shows three selected trajectories on public websites. The Hugging Face task identifies the standard Atria-Dawn-Preview checkpoint and verifies both its configuration and default-reasoning template. The GitHub task starts from repository search, enters the oficial project, and locates an API-reliability issue reporting intermittent HTTP 422 and 502/503 responses. The PyTorch task verifies the evaluation-time dropout warning for scaled\_dot\_product\_attention and follows the documented API to its pinned v2.14 implementation. The three traces reach the requested evidence in 6, 6, and 4 local decisions, respectively. Median Intern-Decision inference latency is 101.6, 109.8, and 95.3 ms (server-reported inference time, excluding planning, network, and browser execution).

These examples are qualitative system demonstrations rather than a browser benchmark. Dynamic pages can occasionally produce delayed or visually unchanged outcomes; the controller therefore re-observes the interface after every action and uses explicit completion checks to avoid unnecessary continuation. This division of labor lets the larger model handle goal decomposition and cross-page reasoning while Intern-Decision handles repeated, structured action selection.

## 7. Conclusion

We introduce Intern-Decision, which makes finite multimodal choices through a single causal forward pass, retaining the pretrained backbone’s native language-model head and exposing candidate probabilities through a typed interface. The 4B model achieves the highest seven-suite average in the presented comparison, with a 44.16 ms mean for a fixed short request on an RTX 4090. Separate calibration and handof studies show how probabilistic outputs can support uncertainty analysis and selective use of a larger model. We believe Intern-Decision can serve as a practical component of larger agent systems, but its generalization to arbitrarily large schemas and complex environments remains an open problem.

## Acknowledgments

This work is supported by Shanghai Artificial Intelligence Laboratory. Besides, we would like to appreciate Houquan Zhou for the valuable comments on calibration evaluation set construction.

## References

[1] allebee. JevK5. https://github.com/allebee/jevk5. 1, 2

[2] Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Junting Zhou, Ziqiang Liu, Feiteng Fang, Mingshan Chang, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, Ruibin Yuan, Haihong Wu, Hongquan Lin, Wenhao Huang, Jiajun Zhang, Chenghua Lin, Jie Fu, Min Yang, Shiwen Ni, and Ge Zhang. COIG-CQIA: Quality is all you need for chinese instruction fine-tuning, 2024. URL https://arxiv.org/abs/2403.18058. 1

[3] Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Jianfeng Gao, Songhao Piao, Ming Zhou, and Hsiao-Wuen Hon. UniLMv2: Pseudo-masked language models for unified language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 642–652. PMLR, 2020. 2

[4] Glenn W. Brier. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3, 1950. 2, 5.1

[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jefrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, 2020. 2

[6] Yu-Neng Chuang, Prathusha Kameswara Sarma, Parikshit Gopalan, John Boccio, Sara Bolouki, Xia Hu, and Helen Zhou. Learning to route LLMs with confidence tokens. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 10859–10878. PMLR, 2025. URL https://proceedings.mlr.press/v267/chuang25b.html. 2

[7] Jasper Dekoninck, Maximilian Baader, and Martin Vechev. A unified approach to routing and cascading for LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 12987–13010. PMLR, 2025. URL https://proceedings.mlr.press/v267/dekoninck25a.html. 2

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1423. 2

[9] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. GLM: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.acl-long.26. 2

[10] fstandhartinger. JevBench: Structured decision evaluation. https://github.com/fstandhartinger/jevbench. Public evaluation comprises 48 Easy, 72 Original, and 111 Hard items. 1, 5.1

[11] Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816–3830. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.acl-long.295. 2

[12] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-Predict: Parallel decoding of conditional masked language models. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6112–6121. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1633. 2

[13] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pages 1321–1330. PMLR, 2017. URL https://proceedings.mlr.press/v70/guo17a.html. 1, 2

[14] Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs, 2024. URL https://arxiv.org/abs/2406.18495. 1

[15] hzwer. LLM Ruozhiba QA. https://github.com/hzwer/LLM-Ruozhiba-QA. Question collection. Accessed 2026-10-15. 1

[16] InternLM. Atria-Dawn-Preview. https://huggingface.co/internlm/Atria-Dawn-Preview, 2026. 5.5

[17] Jared Palmer. Kev. https://github.com/jaredpalmer/kev. 1, 2

[18] Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha Dziri. WildTeaming at scale: From in-the-wild jailbreaks to

(adversarially) safer language models, 2024. URL https://arxiv.org/abs/2406.18510. 1

[19] Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong Wang, Yuxian Wang, Wu Ning, Yutai Hou, Bin Wang, Chuhan Wu, Xinzhi Wang, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming Tang, Defu Lian, Qun Liu, and Enhong Chen. ToolACE: Winning the points of LLM function calling. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=8EB8k6DdCU. 1, 1

[20] Xiao Liu, Hao Zhang, Yifan Li, Chao Chen, Zhibo Liu, Yufan Wang, et al. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations (ICLR), 2024. 1

[21] LocalLLaMA. Typed decisions. https://huggingface.co/datasets/LocalLLaMA/typed-decisions, 2026. Hugging Face dataset card. Accessed 2026-10-15. 1

[22] NandhaKishorM. Laya. https://github.com/NandhaKishorM/laya. 1, 2

[23] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5. 1

[24] Timo Schick and Hinrich Schütze. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 255–269. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.eacl-main.20. 2

[25] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36, pages 68539–68551, 2023. 1

[26] TheoLeeCJ. SemIf-OpenJev. https://github.com/TheoLeeCJ/SemIf-OpenJev. 1, 2

[27] TypeSafe. Jev decision api. https://docs.typesafe.ai/api. Baseline resource linked by the Intern-Decision release. 1, 2

[28] Venus-Team, Changlong Gao, Zhangxuan Gu, Yulin Liu, Xinyu Qiu, Shuheng Shen, Yue Wen, Tianyu Xia, Zhenyu Xu, Zhengwen Zeng, Beitong Zhou, Xingran Zhou, Weizhi Chen, Sunhao Dai, Jingya Dou, Yichen Gong, Yuan Guo, Zhenlin Guo, Feng Li, Qian Li, Jinzhen Lin, Yuqi Zhou, Linchao Zhu, Liang Chen, Zhenyu Guo, Changhua Meng, and Weiqiang Wang. UI-Venus-1.5 technical report, 2026. URL https://arxiv.org/abs/2602.09082. 1

[29] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, et al. UI-TARS-2 technical report: Advancing GUI agent with multi-turn reinforcement learning, 2025. URL https://arxiv.org/abs/2509.02544. 1

[30] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023. 1

[31] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28, 2015. URL https://arxiv.org/abs/1509.01626. 1

[32] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Chen, Bang Ou, Andrew Vail, Shijie Yu, Julian McAuley, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations (ICLR), 2024. 1

## A. Release and Reproduction Details

The public implementation documentation inspected for this draft is fixed to Git revision 2f815802058b. It identifies ordinary next-token alignment for the masked skeleton; historical same-position masked experiments are separate revisions. The published accuracy and pilot tables use XTuner, while the reported local latency uses native Hugging Face inference. A checkpoint, tokenizer, processor, calibration artifact, and backend configuration together define an evaluated system.

The released evaluation procedure consists of four stages. First, verify the bundled test hashes and counts. Second, bind the correct preset temperature to the exact checkpoint through metadata and weight-shard checks. Third, run uncalibrated inference and replay the fixed temperature on the saved probabilities, retaining complete raw outputs and verifying unchanged labels. Fourth, score the pilot separately using its exact references and require 96/96 valid outputs. External-baseline adapters and historical responses are not included, so reproducing Intern-Decision rows does not automatically reproduce every baseline row.

Table 8: Full-precision released temperature presets. Each value belongs to its checkpoint and the reported XTuner execution setting; it is not a freely transferable model-size constant.
<table><tr><td>Checkpoint family</td><td>Temperature</td></tr><tr><td>Intern-Decision-0.8B</td><td>2.747760550702957</td></tr><tr><td>Intern-Decision-2B</td><td>2.100509348277736</td></tr><tr><td>Intern-Decision-4B</td><td>1.9924182353655278</td></tr></table>

## Recovered Training Settings

Frozen configurations, input audits, actual training logs, and tokenizer caches identify the following common settings for the released 0.8B, 2B, and 4B hf-156 checkpoints. Their input files have the same SHA256, 8d3a687d7f84...; the objective revision is masked-next-token-v4. Published weight-shard hashes bind each model to its corresponding completed run. The smaller models come from the September 26 size-scaling runs; the 4B model is their completed September 25 reference.

Table 9: Recovered settings shared by the released 0.8B, 2B, and 4B runs. Global batch size counts packed sequences rather than raw examples.
<table><tr><td>Setting</td><td>Recorded value</td></tr><tr><td>Initialization</td><td>Matching Qwen3.5-0.8B, Qwen3.5-2B, or Qwen3.5-4B backbone</td></tr><tr><td>Epochs / optimizer updates</td><td>2/156</td></tr><tr><td>Peak / minimum learning rate</td><td> $2 \times 1 0 ^ { - 6 } / 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Schedule / warm-up ratio</td><td>Cosine / 0.03</td></tr><tr><td>Global batch / sequence length 16 packed sequences / 8,192 tokens</td><td></td></tr><tr><td>Precision / GPU count / seed</td><td>BF16/4/42</td></tr><tr><td>Maximum gradient norm</td><td>1.0</td></tr><tr><td>Frozen modules</td><td>Vision tower and projector</td></tr><tr><td>Input audit</td><td>17,880 rows; 23,500 fields; 5,500 image rows; no truncation</td></tr></table>

## B. Training Data Examples

## Composition Overview

Table 10 summarizes the frozen Intern-Decision training mixture. Counts refer to training samples; multiquestion records can contribute more than one decision. The total contains 17,880 samples, 23,500 decisions, and 5,500 image-bearing samples.

Table 10: Frozen training mixture shared by the released Intern-Decision models.
<table><tr><td>Component</td><td></td><td>Samples Decisions</td></tr><tr><td>Synthetic decision text</td><td>6,711</td><td>6,711</td></tr><tr><td>Public task distributions</td><td>5,669</td><td>10,789</td></tr><tr><td>Synthetic visual and interactive data</td><td>5,500</td><td>6,000</td></tr><tr><td>Total</td><td>17,880</td><td>23,500</td></tr></table>

The synthetic text data were generated by the teacher model deepseek-v4.1-flash. General contains the 4,066 records from general-category synthesis. Hard combines the 1,762 hard-batch and 883 patch-batch records, giving 2,645 records from dificult-category synthesis. These reporting categories follow the synthesis purpose rather than each record’s dificulty annotation.

Public task distributions cover AG News, ToolACE, Typed Decision, WildGuard, WildJailbreak, and Ruozhiba. The group name describes task sources, not uniform label provenance: Typed Decision references were generated by the project, and Ruozhiba labels are marked as LLM-assisted. The shared input and all three tokenizer caches contain the 469 Ruozhiba records. Synthetic visual sources contain rendered observations and action labels from nine game environments (500 records each), VizDoom (500 records), and shapes/captcha-style tasks (500 records).

The nine games are 2048, Car Racing, Flappy Bird, Maze, Frozen Lake, Sokoban, Snake, Catch, and Tetris. VizDoom and these game environments supply synthetic training data; they are not evaluation suites in the quantitative comparison. The demonstrations in Section 6.4 and the preceding application sections illustrate behavior rather than held-out scores for these environments. Internal partitions used during synthetic-data preparation are distinct from the test sets used to report benchmark results.

## Text Decision: Probability Judgment

State. A bin holds 40 identical tokens; exactly 3 of them are gold. Five tokens are drawn at random, without replacement.

Question. Using exact combinatorial reasoning, answer whether the probability of drawing at least one gold token exceeds 0.45. A “yes” requires the probability to be strictly greater than 0.45; any equality or lower value means no.

Options. no: the probability is not greater than 0.45; yes: the probability is strictly greater than 0.45.   
Target. no.

This example is from the probability portion of decision data. It requires exact computation rather than matching surface keywords.

## Text Decision: Tool Routing

State. The user asks for information about a farm with ID 4532. The candidate tools are getFarm (“Retrieve farm information”) and addCropToFarm (“Add a crop to a farm”).

Question. Ascertain the catalog function that retrieves farm information for a given farm ID.

This example comes from the public task distribution portion and tests whether the model can select the correct tool from a small candidate catalog.

## Visual Decision: Snake

![](images/caf7cbcc27726d177920ce62c1bfac9527c7b16b653721d5e78e1324c13885fa.jpg)  
Figure 8: A Snake training frame.

Question. Choose the next move to guide the snake toward the red apple without hitting a wall or its own body. The end with eyes is the head. The snake grows when it eats and cannot reverse direction.

Options. down, left, right, up.

Target. left.

This example is from the nine game tasks. The model maps a visual game state to a discrete movement decision.

## Visual Decision: VizDoom (Two Questions per Frame)

![](images/19edfe0281d58b0b3ee016864d28d9289118e91604bb9471ef4c23bc537cd66b.jpg)  
Figure 9: Paired turn and attack decisions over a VizDoom frame.

Question 1 (turn). Aim toward the largest clearly visible enemy. Choose left or right according to its horizontal position. Stay when it is centered. If no target is visible, search by turning right. Target: left.

Question 2 (attack). Fire when the horizontal screen center lies inside the largest visible enemy’s silhouette span. Otherwise aim first. Target: no.

VizDoom records contain two decisions per image, illustrating multi-field supervision over a shared visual state.

## Visual Decision: Captcha-Style Shapes

![](images/69fb3bba8e6d9e8b79dd959d32b75d9fa24dd9e837bea210d470e0e780256c44.jpg)  
Figure 10: A shapes-based training frame.

Question. Based on the task and the current screenshot, choose the next mouse action.

Options. Move the pointer by 10 or 30 pixels in one of four directions, or click at the current pointer tip.

Target. RIGHT\_30.

This example comes from the frozen shapes/captcha-style subset and requires target localization and discrete pointer-action selection.

## C. Known-Distribution Test Construction and Cases

## Mechanism coverage and reference construction

The pilot consists of six categories with four families each. Every family has two settings and every setting has two option-order variants. The resulting 96 records cover the following mechanisms:

1. Direct randomness and support: uniform die faces with impossible options; weighted candy inventories; deterministic single-color pens; rare winning lottery tickets.

2. Composed events and mixtures: sums of two dice; numbers of independent hits; independent backup generators; delivery outcomes under an unobserved route mixture.

3. History, conditioning, and hidden state: independent coin flips after a streak; socks after known removals; die outcomes given a restricted sensor observation; a two-state Markov process with noisy sensor readings.

4. Daily evidence and observation bias: defect screening with a stated base rate; conditionally independent repeated tests; correlated copies of one test; waiting times observed at a uniformly sampled time.

5. Selective disclosure and puzzles: informed-host Monty Hall; uninformed-host Monty Hall conditioned on a safe reveal; prize-dependent switch ofers; two-child problems with diferent child-selection protocols.

6. Sequential and combinatorial processes: distinct coupons collected after finite draws; birthday collisions on a toy calendar; gambler’s ruin; first-reward times with a guaranteed stopping limit.

The everyday descriptions define synthetic mechanisms with explicit parameters, not empirical frequencies inferred from real-world populations. Their oracles use exact rational arithmetic: finite counting, Bayes’ rule and total probability, hidden-state forward updates, coupon-count dynamic programming, birthday no-collision products, gambler’s-ruin formulas, and truncated geometric probabilities. An input/reference pair carries a stable ID and input hash. Validation verifies the option keys and probability support, exact normalization, and unchanged references when the option insertion order is reversed. Reference metadata records the family, setting, exact fractions, and derivation separately from the model-visible input.

## Support and conditioning: two further actual cases

Fair die with impossible options. The state specifies a fair six-faced die labeled 1 through 6, rolled once without landing on an edge or leaving the table. The question is “What number will the die show?” The ordered criteria contain labels $1 , \ldots , 8$ . The reference map is

$$
q = \{ 1 { : } 1 / 6 , \ 2 { : } 1 / 6 , \ 3 { : } 1 / 6 , \ 4 { : } 1 / 6 , \ 5 { : } 1 / 6 , \ 6 { : } 1 / 6 , \ 7 { : } 0 , \ 8 { : } 0 \} .
$$

This tests equal mass across possible outcomes and zero mass on impossible ones. Returning one face with probability one fails the distribution test. Even a perfect prediction has expected Brier 5/6, owing to outcome randomness, but zero excess expected Brier.

Sampling after removal. The state says a drawer initially contained six red and four blue socks; exactly three red socks were removed without replacement, with no other changes. One remaining sock is drawn uniformly. The question is “What color is the selected sock?” The reversed-order record lists blue then red; its exact reference is blue: 4/7, red: 3/7. The model must update the inventory before predicting the draw. Reversing criteria changes neither event probabilities nor the labels attached to them. Both cases use the same common random-experiment instruction shown in the Monty Hall input below.

## Monty Hall: prize location as the outcome

Here is an example of the three-door problem and models’ predictions.

```jsonl
{
"state": "A game has 3 doors and one prize, placed uniformly. You select door 1 before receiving
any information. A host who knows the prize location always opens exactly 1 unchosen doors that
contain no prize, then always offers a switch to the only other unopened door. If more than one
legal reveal set exists, the host chooses uniformly among them. You observe this protocol being
completed, but are not told which door numbers were opened.",
"questions": {
"decision": {
"type": "choice",
"instructions": "Where is the prize relative to the final two unopened doors? Predict the
outcome of the specified random experiment using only its stated rules. The listed outcomes are
mutually exclusive and exhaustive. Treat every stated probability, independence assumption, and
observation protocol as exact.",
"criteria": {
"initial_door": "initial_door",
"other_unopened_door": "other_unopened_door"
```

![](images/3f8011a820e340ab64ad21bbf8c1daf4c03d0cfd4298502a575ded2e03389c7c.jpg)