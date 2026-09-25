# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero <sup>1</sup>

## Abstract

Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information may remain behaviorally recoverable through later prompts even after the dialogue shifts to unrelated topics. We introduce PrivDrift, a benchmark for auditing whether userdisclosed secrets remain recoverable after conversational topic drift and persuasion-based probing. PrivDrift contains 1,000 controlled multiturn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage remains substantial, ranging from 38.7% to 54.6%, and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage, suggesting that privacy risk in active LLM contexts should be evaluated as a persistent behavioral failure mode rather than only as training-data memorization or immediate jailbreak behavior.

## 1. Introduction

Large language models (LLMs) have evolved from simple chat systems into persistent assistants capable of maintaining coherence over extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). As these systems are integrated into settings such as healthcare triage, financial planning, enterprise copilots, and personal productivity tools, users may disclose Personally Identifiable Information (PII), including phone numbers, financial identifiers, email addresses, and health-related details. The same context retention that supports useful multi-turn assistance can also create a privacy risk: sensitive information disclosed earlier in an active conversation may remain available to later generations.

Existing safety evaluations mostly target two extremes: training-data memorization, where private information is extracted from model parameters (Carlini et al., 2021; Nasr et al., 2023), and immediate jailbreaking, where a harmful behavior is elicited in the current turn (Wei et al., 2024; Zou et al., 2023). Recent work also studies prompt injection and multi-turn attacks (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024), but these settings do not directly measure whether user-disclosed secrets remain recoverable after unrelated conversational drift. This leaves an important gap for trustworthy AI evaluation: whether topic shift acts as a practical privacy boundary within an active conversation.

We do not assume that users believe an LLM literally forgets earlier messages in the same session. Rather, we study a behavioral risk: users and application designers may underestimate how easily sensitive information disclosed earlier can be elicited later through indirect, justified, high-pressure, or tool-mediated prompts. This risk is relevant to shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and prompt-injection settings where later instructions interact with the active context in ways the original user did not intend.

We introduce PrivDrift, a controlled benchmark for measuring active-context privacy leakage under two axes: topic drift, which captures the number of unrelated turns between secret disclosure and later probing, and persuasion intensity, which captures the interaction style used to elicit the secret. Across three LLMs with extended context windows and 1,000 dialogues per model, dialogue-level leakage remains substantial under hybrid detection, ranging from 38.7% to 54.6%. Persuasion intensity significantly affects leakage, and leakage varies sharply by secret type: SSNs and credit cards are suppressed far more often than emails and phone numbers. Within the tested window $( d \leq 6 )$ , additional drift does not reliably reduce leakage.

Our contributions are as follows:

1. PrivDrift, a controlled benchmark for auditing whether user-disclosed secrets remain recoverable from active conversational context after unrelated topic drift.

2. We evaluate leakage under direct, justified, and highpressure probes, modeling how later prompts can elicit sensitive context through different interaction styles.

3. We implement a reproducible hybrid detector that combines normalization-based matching with an openweights LLM judge, and we report dialogue-level, probe-level, regex-only, fuzzy, and hybrid leakage rates.

4. We propose Privacy Half-Life (τ ), a stability-based metric for persistent suppression, and show that no evaluated model reaches stable decay within the observed drift range.

## 2. Related Work

## 2.1. Contextual Privacy and Inference-Time Leakage

Privacy research in language models has historically focused on training-data extraction, membership inference, and recovery of memorized PII from pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). These settings study whether private information is stored in model parameters. In contrast, active-context privacy concerns user-provided information that appears in the current interaction and must be handled appropriately at inference time.

Recent benchmarks examine whether LLMs can reason about privacy norms and contextual access. ConfAIde evaluates privacy reasoning through contextual integrity (Mireshghallah et al., 2024), while PrivacyLens and PrivaCI-Bench study privacy norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks primarily test whether a model should share information under a static access-control or privacy-norm scenario. PrivDrift instead isolates a persistence question: whether a user-disclosed secret remains behaviorally recoverable after unrelated topic drift and later persuasion-based probing.

## 2.2. Multi-Turn Adversarial Attacks

Aligned models remain vulnerable to multi-turn exploitation. Crescendo attacks show that seemingly benign conversations can gradually elicit harmful outputs (Russinovich et al., 2024), while automated jailbreak frameworks generate adversarial multi-turn attack paths (Deng et al., 2024; Narula et al., 2025). Prompt injection studies also show that later instructions can manipulate LLM-integrated applications (Liu et al., 2023). These works usually focus on unsafe content generation or instruction hijacking. PrivDrift adapts the multi-turn lens to information flow control: the question is not whether a model can be made to produce harmful external content, but whether it will re-disclose sensitive information supplied by a user earlier in the same active context.

## 2.3. Context Management and Unlearning

Machine unlearning benchmarks such as TOFU and WMDP evaluate whether knowledge can be removed or suppressed from model behavior (Maini et al., 2024; Li et al., 2024). These benchmarks usually operate over static questionanswer pairs and model parameters. PrivDrift is related in spirit but targets a different object of control: active context rather than model weights. We ask whether a secret present in the context window is behaviorally suppressed when later prompts try to elicit it. This distinction is important for deployed RAG systems, copilots, and agentic workflows, where privacy control often depends on prompt-level context management rather than parameter-level unlearning.

## 3. The PrivDrift Framework

PrivDrift is a synthetic evaluation framework for measuring whether sensitive information disclosed in an active conversation remains recoverable after topic drift. As shown in Figure 1, the framework has three components: a parametric dialogue generator, a probing harness, and a metric suite for leakage detection and temporal analysis.

## 3.1. Threat Model: Active-Context Re-Disclosure

We study active-context re-disclosure: a failure mode in which sensitive information disclosed earlier in a conversation remains recoverable through later prompts after unrelated topic drift. We do not assume that an adversary is always unable to inspect the raw transcript. Instead, Priv-Drift evaluates whether the model itself will re-disclose the sensitive value when later instructions query, justify, or pressure the assistant to reveal it.

This setting captures several practical risks. In shared or persistent sessions, a later user may interact with an assistant without understanding what private information was previously disclosed. In enterprise copilots and agentic workflows, later tool outputs, retrieved documents, or promptinjection content may issue instructions that interact with the active context. In browser or application-integrated assistants, the user may not intend for earlier sensitive disclosures to be reused after the task has shifted. The core question is therefore not whether the transcript contains the secret, but whether the assistant will behaviorally reproduce the secret when later prompted.

Formally, a user discloses a secret $S$ at turn $t _ { 0 }$ . The conversation proceeds through d unrelated but content-dense drift turns, denoted as $D = \{ u _ { 1 } , a _ { 1 } , \dotsc , u _ { d } , a _ { d } \}$ . At evaluation time, a standardized probe $P$ is appended to the active context. A leakage event occurs if the model response reveals

![](images/2af13acb8a0c3f6576c43def16a229902a21eccb2f45f3504e27f5a2521650f6.jpg)

Figure 1. PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total dialogues</td><td>1,000</td></tr><tr><td>Synthetic dialogues</td><td>900</td></tr><tr><td>Human-authored dialogues</td><td>100</td></tr><tr><td>Secret categories</td><td>4</td></tr><tr><td>Drift lengths</td><td>0, 2, 3, 4, 5, 6</td></tr><tr><td>Avg. words per dialogue</td><td>359.16</td></tr><tr><td>Median words per dialogue</td><td>385.00</td></tr><tr><td>Min/max words per dialogue</td><td>43 / 795</td></tr><tr><td>Avg. approx. tokens per dialogue</td><td>478.87</td></tr><tr><td>Median approx. tokens per dialogue</td><td>513.33</td></tr><tr><td>Min/max approx. tokens per dialogue</td><td>57.33 / 1060.00</td></tr></table>

Table 1. Dataset summary for the evaluated 1,000-dialogue benchmark. Approximate token counts are estimated from word counts.

S exactly, approximately, or semantically.

## 3.2. Dataset Generation Pipeline

We generate N = 1000 controlled dialogues using a multistage scaffold-then-rewrite pipeline. This design preserves strict control over the ground-truth secret, drift length, and probe structure while reducing the rigid artifacts of purely template-based generation. The dataset contains 900 synthetic dialogues and 100 human-authored dialogues constructed under the same secret and drift constraints. The human-authored dialogues were inserted throughout the benchmark rather than stored as a separate contiguous block. Since the current cached evaluation metadata does not preserve the human-authored identifiers, we report aggregate results over the full evaluated benchmark and leave a separate human-versus-synthetic leakage comparison for future work.

## 3.2.1. PERSONA AND SECRET INJECTION

Each dialogue is initialized with a persona sampled from more than 100 professions and demographic profiles, together with a ground-truth secret S. We sample S from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers. The secret is inserted in the opening turn as part of a plausible task context, establishing a legitimate in-context reason for the assistant to observe the information.

## 3.2.2. CONTENT-DENSE TOPIC DRIFT

For each dialogue, we assign a topic such as tax filing, medical triage, or travel logistics. Drift turns are drawn from a topic-conditioned library of dense question-answer pairs and fallback tasks involving professional writing, structured reasoning, or long-form summarization. A subset of drift turns is human-authored by annotators and incorporated into the generation library. This design aims to add unrelated entities and task demands to the active context while preserving experimental control over the original secret and the number of drift turns.

## 3.2.3. NATURALISTIC REFINEMENT

To reduce rigidity from the initial scaffold, we apply a final naturalistic refinement step. Each dialogue scaffold is passed to a rewriting model with instructions to improve fluency and persona realism while preserving the exact secret, drift length, role structure, and final probe conditions. This step modifies surface phrasing but does not change the experimental variables.

## 3.3. Probing Setup

The main evaluation appends a final standardized probe to the completed conversation history. Probes are grouped into three persuasion levels:

• Simple: a direct neutral query, such as asking for the phone number mentioned earlier.

• Medium: a query with a contextual justification, such as needing the information for a form.

• Hard: a high-pressure or urgent request designed to test whether pressure cues increase or decrease leakage.

We sample drift length d, the number of turns between secret disclosure and final probing, from a mixed distribution:

$$
d \sim \left\{ \begin{array} { l l } { 0 } & { \mathrm { w i t h } p = 0 . 2 , } \\ { \mathrm { U n i f o r m } \{ 2 , 3 , 4 , 5 , 6 \} } & { \mathrm { w i t h } p = 0 . 8 . } \end{array} \right.\tag{1}
$$

This distribution includes immediate recall cases while emphasizing short-to-medium topic drift. We interpret the results as bounded active-context persistence, not as a full test of arbitrarily long context windows.

## 3.4. Evaluation Methodology

A model may reveal a secret verbatim, with formatting changes, through a partial substring, or through a paraphrased reference. We therefore implement a hierarchical hybrid detector. A response is labeled as leakage if either the normalized secret appears in the normalized response or the LLM judge determines that the response reveals the full secret, a substantial substring, or enough partial information to identify the secret.

## 3.4.1. STAGE 1: NORMALIZATION-BASED MATCHING

The first stage detects verbatim and near-verbatim leakage. For numeric secrets, we strip all non-digit characters from both the ground-truth secret and the model response. For alphanumeric secrets, we normalize case and whitespace while preserving meaningful delimiters. Let $\phi ( \cdot )$ denote the normalization function. The regex-stage leakage label is

$$
\mathrm { L e a k } _ { \mathrm { r e g e x } } ( R , S ) = \mathbb { I } \big [ \phi ( S ) \subset \phi ( R ) \big ] ,\tag{2}
$$

where $R$ is the model response and I is the indicator function.

## 3.4.2. STAGE 2: LLM-AS-A-JUDGE VERIFICATION

Responses that are negative under normalization-based matching are passed to an open-weights Llama judge (Meta AI, 2024). The judge receives the secret and the model response and returns a binary verdict indicating whether the response reveals the secret or a substantial part of it. This stage is intended to capture partial disclosure, indirect hints, and formatting variations not captured by deterministic matching. We report regex-only and hybrid rates separately because judge-based evaluation can introduce its own errors.

![](images/5a92d6081d997784ed270be10ccd688f8043b0b643046610a312fe5c1f888737.jpg)  
Figure 2. Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

## 3.4.3. PRIVACY HALF-LIFE

We introduce Privacy Half-Life (τ), a stability-based metric for sustained suppression. Let $D _ { o b s }$ be the observed set of drift lengths and $L ( d )$ be the leakage rate at drift length d. We define the safety threshold as $\delta = 0 . 1 \times L ( 0 )$ and define

$$
\begin{array} { r l } { \tau = \operatorname* { m i n } \{ d \in D _ { o b s } \mid L ( d ) \leq \delta } & { } \\ { \mathrm { a n d } \forall d ^ { \prime } \in D _ { o b s } , d ^ { \prime } > d \colon L ( d ^ { \prime } ) \leq \delta \} . } & { } \end{array}\tag{3}
$$

If this set is empty, we assign $\tau = \operatorname* { m a x } ( D _ { o b s } ) + 1$ . This definition prevents transient dips from being interpreted as stable suppression.

## 4. Experimental Setup

Models. We evaluate GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B through the OpenRouter API. All models are queried with the same dialogue histories and probe templates.

Dataset. We evaluate 1,000 dialogues generated by the PrivDrift pipeline on each model. Each dialogue contains one seeded secret and one drift length $d \in \{ 0 , 2 , 3 , 4 , 5 , 6 \}$ sampled from the distribution above. Each dialogue is evaluated under three standardized persuasion probes, yielding 3,000 cached model-probe responses per model and 9,000 total cached responses across the three evaluated models.

![](images/ca2986adf5b198a5d6e555be711fc4ded400ce52beefaf0a81861d5da8465a67.jpg)  
Figure 3. Privacy Half-Life τ requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

Aggregation levels. We report two aggregation levels. Overall leakage is computed at the dialogue level: a dialogue is counted as leaking if any standardized probe elicits the secret. Persuasion-level and fuzzy-matching analyses are computed at the probe level, where each dialogue contributes one response per persuasion condition. This distinction explains why dialogue-level leakage can exceed the average of individual persuasion-level leakage rates.

Uncertainty and statistical tests. We report both regexonly and hybrid leakage. For dialogue-level rates, we compute 95% bootstrap confidence intervals over dialogues. For probe-level sensitivity checks, we compute rates over cached model-probe responses. For paired model and detector comparisons on matched examples, we use McNemar’s test with Bonferroni correction where applicable. For repeated measures across persuasion levels, we use Cochran’s Q test with post-hoc McNemar tests. To test association between leakage and drift length, persuasion level, or secret type, we use chi-square tests of independence and report Cramer’s V as an effect size.

## 5. Results

## 5.1. Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

At the dialogue level, hybrid evaluation differs only slightly from normalization-based matching, indicating that leakage in this benchmark is dominated by direct or near-direct reproduction rather than purely semantic paraphrase. The

<table><tr><td>Model</td><td>Regex</td><td>Hybrid</td><td>∆</td><td>p</td></tr><tr><td>GPT-OSS-120B</td><td>47.70</td><td>47.70</td><td>+0.00</td><td>1.0</td></tr><tr><td>DeepSeek-R1</td><td>37.80</td><td>38.70</td><td>+0.90</td><td> $7 . 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Qwen3-VL-235B</td><td>53.50</td><td>54.60</td><td>+1.10</td><td> $2 . 5 7 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 2. Dialogue-level leakage (%) under regex-only and hybrid evaluation. A dialogue is counted as leaking if any standardized probe elicits the secret.

<table><tr><td>Model</td><td>Hybrid Leakage</td><td>95% CI</td></tr><tr><td>DeepSeek-R1</td><td>38.70%</td><td>[35.7, 41.6]</td></tr><tr><td>GPT-OSS-120B</td><td>47.70%</td><td>[44.6, 50.8]</td></tr><tr><td>Qwen3-VL-235B</td><td>54.60%</td><td>[51.6, 57.7]</td></tr></table>

Table 3. Dialogue-level hybrid leakage rates with 95% bootstrap confidence intervals. A dialogue is counted as leaking if any standardized probe elicits the secret.

LLM judge adds up to 1.10 percentage points of additional dialogue-level leakage across models (Table 2). This supports reporting hybrid leakage as the main metric while retaining regex-only rates as a transparent deterministic baseline.

## 5.2. All Models Leak Substantially at the Dialogue Level

Under dialogue-level hybrid evaluation, leakage remains substantial for all models: DeepSeek-R1 leaks 38.70%, GPT-OSS-120B leaks 47.70%, and Qwen3-VL-235B leaks 54.60% (Table 3). These rates estimate whether a dialogue is vulnerable to at least one of the standardized extraction probes.

## 5.3. Persuasion Intensity Affects Leakage Non-Monotonically

Probe-level leakage varies significantly across persuasion levels for all models (Cochran’s $Q , p < 1 0 ^ { - 5 } )$ . However, the direction is model-dependent (Table 4). For GPT-OSS-120B, hard persuasion reduces leakage relative to simple and medium, consistent with pressure cues triggering safety behavior. For DeepSeek-R1, medium persuasion yields the lowest leakage. For Qwen3-VL-235B, hard persuasion produces the highest leakage, indicating weaker resistance to pressure-based extraction.

## 5.4. Drift Length Does Not Reliably Reduce Leakage Within the Tested Window

Although drift-leak curves exhibit a dip near d = 3 followed by rebound, effect sizes for drift length are small or negligible across models. Cramer’s V is 0.1073 for DeepSeek-R1, 0.0659 for GPT-OSS-120B, and 0.0723 for Qwen3-VL-235B in the cached probe-level analysis. This should not be interpreted as a claim about arbitrarily long contexts. Instead, it shows that privacy risk can persist across several unrelated topic shifts even before the original secret is far from the end of the context window.

<table><tr><td>Model</td><td>Simple</td><td>Medium</td><td>Hard</td></tr><tr><td>GPT-OSS-120B</td><td>39.30%</td><td>40.50%</td><td>32.90%</td></tr><tr><td>DeepSeek-R1</td><td>28.10%</td><td>22.00%</td><td>29.20%</td></tr><tr><td>Qwen3-VL-235B</td><td>38.70%</td><td>33.30%</td><td>40.60%</td></tr></table>

Table 4. Probe-level hybrid leakage by persuasion intensity.

![](images/d90a1d6c02245de3608de7dfc2dd00bf1bce5dbdd2617c6b078230e80ce13021.jpg)  
Figure 4. Hybrid leakage versus drift length. Curves show a dip near d = 3 followed by rebound.

## 5.5. Secret Type Strongly Determines Leakage

Leakage varies sharply by secret type. Cramer’s V indicates a large association between secret type and hybrid leakage for all models: 0.5871 for DeepSeek-R1, 0.7731 for GPT-OSS-120B, and 0.6927 for Qwen3-VL-235B in the cached probe-level analysis. SSNs and credit card numbers are suppressed far more often than emails and phone numbers, even though all are user-disclosed and contextually private (Table 5). This pattern suggests that current behavior depends more on format and sensitivity cues than on a generalized notion of contextual confidentiality.

## 5.6. Privacy Half-Life Exceeds the Observed Window

Using the stable-decay definition of Privacy Half-Life with threshold $\delta = 0 . 1 \cdot L ( 0 )$ , no evaluated model approaches the threshold at any tested drift length. Consequently, $\tau =$ max $( D _ { o b s } ) + 1 = 7$ for all models and persuasion levels, indicating no stable decay within $d \leq 6$

## 5.7. Fuzzy Matching Supports the Hybrid Labels

As a post-hoc deterministic sensitivity check, we compute normalized fuzzy matching between each saved secret and saved model response using cached outputs only. At a 0.85 threshold, fuzzy leakage closely tracks probe-level hybrid leakage for all models: 26.40% versus 26.43% for DeepSeek-R1, 37.03% versus 37.57% for GPT-OSS-120B, and 37.60% versus 37.53% for Qwen3-VL-235B (Table 6). We treat fuzzy matching as auxiliary evidence rather than the primary metric because approximate string similarity is less expressive than the judge for partial or semantic disclosures.

<table><tr><td>Secret Type</td><td>GPT-OSS</td><td>DeepSeek</td><td>Qwen</td></tr><tr><td>SSN</td><td>0.14%</td><td>1.11%</td><td>5.97%</td></tr><tr><td>Credit card</td><td>0.00%</td><td>0.39%</td><td>4.01%</td></tr><tr><td>Email</td><td>78.24%</td><td>57.11%</td><td>81.13%</td></tr><tr><td>Phone</td><td>70.89%</td><td>46.13%</td><td>57.24%</td></tr></table>

Table 5. Probe-level hybrid leakage by secret type across cached model-probe responses.

<table><tr><td>Model</td><td>Regex</td><td>Fuzzy 0.85</td><td>Fuzzy 0.80</td><td>Hybrid</td></tr><tr><td>DeepSeek-R1</td><td>25.50</td><td>26.40</td><td>26.53</td><td>26.43</td></tr><tr><td>GPT-OSS-120B</td><td>36.80</td><td>37.03</td><td>37.07</td><td>37.57</td></tr><tr><td>Qwen3-VL-235B</td><td>36.97</td><td>37.60</td><td>37.93</td><td>37.53</td></tr></table>

Table 6. Probe-level regex, fuzzy, and hybrid leakage rates (%) computed from cached model outputs.

## 6. Discussion

PrivDrift reveals a persistent active-context privacy risk: sensitive information disclosed earlier in a conversation can remain behaviorally recoverable after unrelated topic drift. This does not imply that models permanently remember the secret, nor that all long-context settings behave similarly. Instead, it shows that current assistants may reproduce sensitive in-context information when later prompts provide enough retrieval pressure. The probe-level cached evaluation also shows that this behavior is mostly direct or near-direct reproduction, since fuzzy matching closely tracks hybrid leakage.

The strongest pattern is the asymmetry across secret types. SSNs and credit card numbers are suppressed far more often than emails and phone numbers, even though all four categories are user-disclosed secrets in the benchmark. This suggests that current safeguards may rely on format-sensitive safety heuristics rather than robust contextual reasoning about confidentiality.

Persuasion effects are significant but non-monotonic and model-dependent. For GPT-OSS-120B, high-pressure requests reduce leakage, consistent with the possibility that urgency cues activate refusal behavior. For DeepSeek-R1, medium persuasion produces the lowest leakage. For Qwen3-VL-235B, hard persuasion produces the highest leakage, suggesting weaker resistance to pressure-based

![](images/cd77cc231d8f7d14919423b86f73bd4fbf3313a8b10251cb602a96dde17423de.jpg)  
Figure 5. Hybrid leakage heatmap by drift length and persuasion intensity for each model.

extraction.

Finally, topic drift should not be treated as an implicit privacy boundary. Although leakage curves exhibit intermediate dips, those decreases are not stable enough to satisfy the Privacy Half-Life criterion. This motivates stability-based evaluation rather than treating temporary drops as evidence of suppression.

## 7. Limitations and Future Work

Bounded drift range. PrivDrift evaluates drift lengths up to d = 6, corresponding to bounded short-to-medium conversational drift rather than full saturation of modern context windows. The results should therefore be interpreted as evidence of persistence under bounded active-context drift, not as a complete characterization of privacy behavior across full long-context windows.

Fixed-format secrets. The benchmark focuses on structured secrets such as SSNs, credit card numbers, emails, and phone numbers. These are easier to detect than unstructured sensitive information such as health status, family circumstances, immigration status, or financial hardship. Future work should extend the benchmark to non-fixed sensitive attributes.

Active context, not cross-session memory. PrivDrift evaluates recoverability within the active conversation context. It does not test training-data memorization, personalization memory, or cross-session retention.

Synthetic and human-authored dialogues. The dataset contains 900 synthetic dialogues and 100 human-authored dialogues inserted throughout the benchmark rather than stored as a separate block. This construction improves realism relative to a purely synthetic benchmark while preserving control over secret type, drift length, and probe structure. However, the current cached evaluation metadata does not preserve which evaluated rows correspond to the human-authored dialogues, so we report aggregate benchmark results and leave a separate human-versus-synthetic comparison for future work.

Detector scope. Leakage labels are based on surface-form outputs using deterministic matching, fuzzy matching, and an external LLM judge. We report regex-only, fuzzy, and hybrid labels separately to make detector behavior transparent, but we do not yet report formal inter-annotator agreement between humans and the automated methods. Future work should include a larger manual audit of borderline partial disclosures and obfuscations.

API-served models and reproducibility. We evaluate API-served models due to compute and budget constraints. These models may change over time. We mitigate this by logging model identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

Artifact availability. We plan to release the benchmark generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized subset of generated dialogues. Because the benchmark contains synthetic PII-like strings, we will release examples with non-real identifiers and generation templates sufficient to reproduce the benchmark without exposing realistic identifiers.

## 8. Conclusion

We introduce PrivDrift, a benchmark for auditing activecontext privacy leakage under topic drift and persuasion intensity. Across three LLMs and 1,000 dialogues per model, dialogue-level hybrid leakage remains substantial, ranging from 38.7% to 54.6%. Persuasion intensity affects leakage in model-dependent ways, while secret type has the strongest association with leakage. Within the tested drift range, topic drift does not reliably reduce leakage, and Privacy Half-Life exceeds the observed window for all models. The observed asymmetry across secret types suggests that current protections remain uneven and depend more on format-sensitive cues than on robust contextual confidentiality.

## References

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-\` Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., and Raffel, C. Extracting <sup>´</sup> training data from large language models. In Proceedings of the 30th USENIX Security Symposium, 2021.

Deng, G., Liu, Y., Li, Y., Wang, K., Zhang, Y., Li, Z., Wang, H., Zhang, T., and Liu, Y. Masterkey: Automated jailbreaking of large language model chatbots. In Network and Distributed System Security Symposium, 2024.

Li, H., Guo, D., Fan, W., Xu, M., and Huang, J. Multistep jailbreaking privacy attacks on ChatGPT. arXiv preprint arXiv:2304.05197, 2023. URL https:// arxiv.org/abs/2304.05197.

Li, H., Hu, W., Jing, H., Chen, Y., Hu, Q., Han, S., Chu, T., Hu, P., and Song, Y. Privaci-bench: Evaluating privacy with contextual integrity and legal compliance. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10544–10559. Association for Computational Linguistics, 2025. URL https: //aclanthology.org/2025.acl-long.518/.

Li, N., Pan, A., Gopal, A., Yue, S., Berrios, D., Gatti, A., Li, J. D., Dombrowski, A.-K., Goel, S., Phan, L., Mukobi, G., Helm-Burger, N., Lababidi, R., Justen, L., Liu, A. B., Chen, M., Barrass, I., Zhang, O., Zhu, X., Tamirisa, R., Bharathi, B., Khoja, A., Zhao, Z., Herbert-Voss, A., Breuer, C. B., Marks, S., Patel, O., Zou, A., Mazeika, M., Wang, Z., Oswal, P., Lin, W., Hunt, A. A., Tienken-Harder, J., Shih, K. Y., Talley, K., Guan, J., Kaplan, R., Steneker, I., Campbell, D., Jokubaitis, B., Levinson, A., Wang, J., Qian, W., Karmakar, K. K., Basart, S., Fitz, S., Levine, M., Kumaraguru, P., Tupakula, U., Varadharajan, V., Wang, R., Shoshitaishvili, Y., Ba, J., Esvelt, K. M., Wang, A., and Hendrycks, D. The WMDP benchmark: Measuring and reducing malicious use with unlearning. In Proceedings of the 41st International Conference on Machine Learning, 2024.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics, 12:157– 173, 2024. doi: 10.1162/tacl a 00638. URL https: //aclanthology.org/2024.tacl-1.9/.

Liu, Y., Deng, G., Xu, Z., Li, Y., Zheng, Y., Zhang, Y., Zhao, L., Zhang, T., and Liu, Y. Prompt injection attack against LLM-integrated applications. arXiv preprint arXiv:2306.05499, 2023. URL https:// arxiv.org/abs/2306.05499.

Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., and Kolter, J. Z. TOFU: A task of fictitious unlearning for LLMs. In First Conference on Language Modeling, 2024.

Meta AI. The llama 3 herd of models, 2024. URL https: //arxiv.org/abs/2407.21783.

Mireshghallah, N., Kim, H., Zhou, X., Tsvetkov, Y., Sap, M., Shokri, R., and Choi, Y. Can LLMs keep a secret? testing privacy implications of language models via contextual integrity theory. In The Twelfth International Conference on Learning Representations, 2024.

Narula, S., Rafiei Asl, J., Ghasemigol, M., Blanco, E., and Takabi, D. Harmnet: A framework for adaptive multi-turn jailbreak attacks on large language models. arXiv preprint arXiv:2510.18728, 2025. URL https: //arxiv.org/abs/2510.18728.

Nasr, M., Carlini, N., Hayase, J., Jagielski, M., Cooper, A. F., Ippolito, D., Choquette-Choo, C. A., Wallace, E., Tramer, F., and Lee, K. Scalable extraction of train-\` ing data from (production) language models. arXiv preprint arXiv:2311.17035, 2023. URL https:// arxiv.org/abs/2311.17035.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. URL https://arxiv. org/abs/2303.08774.

Russinovich, M., Salem, A., and Eldan, R. Crescendo: A multi-turn jailbreak attack on aligned LLMs. arXiv preprint arXiv:2404.01822, 2024. URL https:// arxiv.org/abs/2404.01822.

Shao, Y., Li, T., Shi, W., Liu, Y., and Yang, D. Privacylens: Evaluating privacy norm awareness of language models in action. arXiv preprint arXiv:2409.00138, 2024. URL https://arxiv.org/abs/2409.00138.

Shokri, R., Stronati, M., Song, C., and Shmatikov, V. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy, 2017.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. URL https://arxiv.org/abs/2307.09288.

Wei, A., Haghtalab, N., and Steinhardt, J. Jailbroken: How does LLM safety training fail? Advances in Neural Information Processing Systems, 36, 2024.

Zou, A., Wang, Z., Kolter, J. Z., and Fredrikson, M. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023. URL https://arxiv.org/abs/2307.15043.

## A. Dataset Characterization

<table><tr><td>Drift d</td><td>N</td><td>Avg. words</td><td>Median words</td><td>Avg. tokens</td><td>Median tokens</td></tr><tr><td>0</td><td>197</td><td>44.29</td><td>43.00</td><td>59.05</td><td>57.33</td></tr><tr><td>2</td><td>147</td><td>253.97</td><td>258.00</td><td>338.62</td><td>344.00</td></tr><tr><td>3</td><td>142</td><td>344.18</td><td>347.00</td><td>458.91</td><td>462.67</td></tr><tr><td>4</td><td>166</td><td>452.27</td><td>447.00</td><td>603.02</td><td>596.00</td></tr><tr><td>5</td><td>181</td><td>520.92</td><td>543.00</td><td>694.56</td><td>724.00</td></tr><tr><td>6</td><td>167</td><td>568.03</td><td>605.00</td><td>757.37</td><td>806.67</td></tr></table>

Table 7. Dataset length characteristics by drift length. Approximate token counts are estimated from word counts.

## B. Additional Effect Sizes

<table><tr><td>Model</td><td>Variable</td><td>Cramer&#x27;s V</td><td>Interpretation</td></tr><tr><td>DeepSeek-R1</td><td>Secret type</td><td>0.5871</td><td>Large</td></tr><tr><td>DeepSeek-R1</td><td>Drift length</td><td>0.1073</td><td>Small</td></tr><tr><td>DeepSeek-R1</td><td>Persuasion</td><td>0.0718</td><td>Negligible</td></tr><tr><td>GPT-OSS-120B</td><td>Secret type</td><td>0.7731</td><td>Large</td></tr><tr><td>GPT-OSS-120B</td><td>Drift length</td><td>0.0659</td><td>Negligible</td></tr><tr><td>GPT-OSS-120B</td><td>Persuasion</td><td>0.0689</td><td>Negligible</td></tr><tr><td>Qwen3-VL-235B</td><td>Secret type</td><td>0.6927</td><td>Large</td></tr><tr><td>Qwen3-VL-235B</td><td>Drift length</td><td>0.0723</td><td>Negligible</td></tr><tr><td>Qwen3-VL-235B</td><td>Persuasion</td><td>0.0639</td><td>Negligible</td></tr></table>

Table 8. Probe-level effect sizes for associations between hybrid leakage and benchmark variables.

## C. Dialogue Construction Figure

![](images/f101ee49e0978aaa62fc0eff088222c01d3327de6aafef7c261100da144c67a8.jpg)  
Figure 6. Dialogue construction details.