# How Far Can Lightweight LLMs Go in Intent-to-SLO Translation?

Sabbir Ahmed, Hazem Abbas, Salimur Choudhury

School of Computing, Queen's University, Ontario, Canada

{sabbir.ahmed, hazem.abbas, s.choudhury} @queensu.ca

Abstract—Intent-based networking (IBN) promises QoS provisioning in which the network derives the configuration from operator-stated goals. Its first step, translating a natural-language intent into enforceable service-level objectives (SLOs) grounded in a live snapshot of the network state, increasingly relies on large language models (LLMs). These large models add cost and energy to the management loop and typically require multi-GPU, cloudhosted infrastructure. Can lightweight open LLMs take over this step dependably? We answer this question with a five-dimensional evaluation of ten LLMs with <10B parameters from five families under four prompting setups, on 716 placement intents. We evaluate translation performance, intent validation, hallucination, energy impact, and robustness to diverse phrasing. The best configurations translate intents with a strict F1 of 0.943 and reject up to 191 of 195 invalid intents. With few-shot reasoning, the strongest models hallucinate unsupported QoS requirements in only about one response in a hundred, whereas the zeroshot setup struggles. A translated intent costs 0.044–0.148 Wh of GPU energy without reasoning, and added energy cost for reasoning pays off for the weaker models. Furthermore, we show that realistic rephrasing lowers the rejection of invalid intents in 15 of 16 configurations for the best-performing models, leaving robust intent validation as a key open challenge for reliable IBN. Our code and data are available: intent2slo.

Index Terms—Intent-based networking, large language models, QoS provisioning, hallucination, energy efficiency, robustness

## I. INTRODUCTION

Edge-cloud service networks must meet fine-grained quality of service (QoS) and reliability requirements for every service they host. Intent-based networking (IBN) lets an operator state them as a goal, e.g., "place the service on the edge, on a node with at least 4 CPU cores and the lowest memory utilization", instead of configuring them by hand [1], [2]. Intent translation, the entry point of emerging management architectures like 3GPP intent-driven management [2], RAN application orchestration [3], and edge service placement [4], turns this goal into service-level objectives (SLOs) that an orchestrator can enforce, grounded in the current network state. A wrong, missing, or invented SLO can end in a wrong placement and a QoS violation. Modern large language models (LLMs) have made this translation practical [4]-[8].

Most of these systems, however, rely on large cloud-hosted LLMs, which take the operator's monitoring data outside its domain and add cost to every intent. Lightweight open models have been used to translate intents into network device programs [9] and matched larger models in intentbased orchestration [10]. Existing studies evaluate mainly translation performance, often on small test sets or with large hosted models [4], [5], and operator-backed benchmarks assess telecom knowledge and safety [11], but not intent-to-SLO translation. A reliable intent translator must also refuse invalid intents, avoid inventing requirements and wasting energy, and cope with diverse phrasing. To our knowledge, no study has evaluated lightweight LLMs for intent-to-SLO translation on all of these dimensions together.

![](images/faffa483b7c973bea4e91c7ae03fd467b9c93c5128dee4b59951584aeabb7035.jpg)  
Fig. 1. Evaluation framework. A lightweight LLM translates an operator intent and the state snapshot of the edge-cloud cluster into SLOs, or rejects the intent. Prompts are zero-shot (z s) or few-shot (f s), with or without chainof-thought (cot). The output is assessed for translation performance (RQ1), intent validation (RQ2), hallucination (RQ3), energy and $\mathrm { C O _ { 2 } e }$ (RQ4), and robustness to operator phrasing (RQ5).

We fill this gap with a study of ten open lightweight LLMs from five families, four prompting setups, and 716 intents. We ask whether a lightweight model translates an intent into exactly the SLOs an orchestrator must enforce (RQ1), rejects ambiguous, conflicting, malformed, or unsupported intents (RQ2), and avoids inventing QoS requirements that neither the operator nor the network state supports (RQ3). We then quantify what each translation costs in energy and $\mathrm { C O _ { 2 } e }$ and which deployment choices dominate that cost (RQ4), and test whether the translator holds when the same intents are rephrased in diverse ways (RQ5). Figure 1 summarizes the evaluation. Our answers form the contributions of this work:

• Translation performance (RQ1): the best configurations reach a strict $F _ { 1 }$ of 0.943, on par with the Intent Engine pipeline [5], and model family matters as much as size.

• Reliable intent validation (RQ2): the strongest models reject up to 98% of invalid intents.

• Hallucination made measurable (RQ3): we separate invented QoS requirements from ordinary mistakes; the best ones hallucinate in about one response in a hundred.

• Energy implication (RQ4): translating an intent costs 0.044–0.148 Wh without reasoning; and reasoning multiplies this by 3.5–7.1×, which pays off for weaker models.

• Robustness to operator phrasing (RQ5): across three realistic rephrasing styles, both translation and validation performance degrade, with the rejection of invalid intents dropping up to 18.8 points. We release the 2,148 rephrased intents as a robustness benchmark.

## II. RELATED WORK

Manias et al. utilize semantic routing for LLM-assisted intent-based management of 5G core networks [6], and Habib et al. extract the intent type and keywords for a hierarchical reinforcement-learning agent [7]. Other works translate intents to generate deployable configurations, like programs for software-defined network devices with a small language model [9], Linux traffic-shaping commands with 2.7B-8B open models [12], and Kubernetes resources for networkaware placement [4]. NetConfEval benchmarks LLMs on turning requirements into formal configurations [8]. Closest to our setting, Intent Engine translates placement intents into snapshot-grounded SLOs and reports rejection and hallucination only for GPT-4.1 mini [5], and Raptis et al. validate LLMgenerated intents with symbolic guardrails [13]. Most other works judge a translator only by how well its output matches a reference, often on small test sets (e.g., 30 requests [4], 100 synthetic intents [12]). Whether the model refuses intents it cannot serve, invents requirements, or what energy each translation costs is rarely measured.

Phone et al. compare a fine-tuned 1.1B model with larger LLMs in an agentic intent-based orchestrator [10]. Ahmed et al. reported that many errors of 7B open models on telecom question answering (QA) are format failures [14]. Hammar et al. show that a fine-tuned 14B model can plan incident response better than frontier LLMs [15]. These studies target QA, troubleshooting, or orchestration, but do not address intent-to-SLO translation that an orchestrator can enforce.

Definition of hallucination is often application-defined. Hammar et al. call an action hallucinated if it does not shorten the expected recovery and bound how often this happens through abstention [15], Manias et al. report it only anecdotally [6], and Intent Engine counts incorrect or missing values and schema violations alongside unsupported constraints [5]. In natural language generation, it is content that the source contradicts [16]. To our knowledge, no prior work separates invented QoS requirements from ordinary mistakes, or studies how lightweight translators handle ambiguous intents.

Inference energy has been benchmarked per generated token for LLM serving [17], and LLNet uses a small language model to make intent translation more sustainable [9]. However, generic benchmarks do not cover a network task with long snapshots and short structured answers, nor relate energy to performance and validation. Robustness to rephrasing has been studied for intent-based policy generation under linguistic distribution shift [18], but not for intent-to-SLO translation. We bring these strands together by evaluating lightweight translators on performance, validation, hallucination, energy, and robustness to phrasing in one framework, which is the evidence needed before entrusting QoS provisioning to them.

## III. EVALUATION FRAMEWORK

## A. System Model and Task

We consider a service-oriented edge-cloud system managed through intents. A monitoring pipeline exports a cluster snapshot $\mathcal { N }$ that gives, for each node, its layer (edge or cloud), and its current capacity and utilization metrics. An intent e for a service s is a natural-language request. The translation maps $( e , s , \mathcal { N } )$ to a set of SLOs Š. Each SLO σ is a QoS requirement with a type (placement, static capacity, dynamic utilization, or network resource), a metric, a comparator, a target value, and a unit [5]. An explicit SLO extracts a threshold from the intent (“at least 4 CPU cores’); an implicit SLO comes from a superlative request (“the lowest memory utilization"), and its value must be read from the snapshot. For a metric m and a requested layer l, the value is

$$
v ^ { * } = \operatorname * { o p t } _ { \substack { n \in \mathcal { N } _ { \ell } , { x _ { m } ( n ) \neq \mathcal { O } } } } x _ { m } ( n ) , \qquad \mathrm { o p t } \in \{ \mathrm { m a x } , \mathrm { m i n } \} ,\tag{1}
$$

where $\mathcal { N } _ { \ell }$ is the set of nodes of layer l (all nodes if the intent names no layer) and $x _ { m } ( n )$ is the value of m at node n; nodes that report no value are skipped. Grounding ties the SLO to the current network state.

An intent that is ambiguous, self-contradictory, malformed, or asks for an unsupported metric must yield $\hat { S } \ = \ \varnothing$ as any SLO derived from it would steer provisioning toward requirements the operator never stated (intent validation). The reference output of an intent is its ground-truth SLO set S.

## B. Dataset

We use the public dataset of Islam and Calheiros [5] containing 716 placement intents on a 6-node edge-cloud Kubernetes cluster (four edge, two cloud nodes), all referring to one cluster snapshot. Of these, 521 are valid and 195 are invalid (65 ambiguous, 49 conflicting, 35 malformed, 46 unsupported). Each valid intent has a complexity level from 1 to 5, depending on the number and mix of its SLOs. The valid intents carry 1,169 ground-truth SLOs.

## C. Prompt Design

The prompts encode six requirements of the task, independent of the model: (i) a fixed output schema with the supported metrics; (ii) canonical units, with explicit conversions; (iii) grounding in snapshot, reading superlative values from the snapshot; (iv) no unstated requirements, e.g., a placement only if the intent names a layer; (v) intent validation, returning an empty SLO list, for invalid intents, with instructions to limit false rejections; and (vi) generic rules, phrased without wording from the dataset's invalid intents. The zero-shot prompt (zs) holds these instructions, the snapshot, and the intent. The few-shot prompt (fs) adds three invented examples, one per behavior: explicit thresholds with unit conversion, a conflicting intent, and implicit SLOs on a small invented snapshot; none resembles a dataset intent sample or shares a value with the real snapshot. The chain-of-thought variants (zs-cot, fs-cot) add an instruction to reason step by step before the final JSON object containing SLOs. Native reasoning (“thinking") modes are disabled for all models.

## D. Metrics

Translation performance (RQ1). As a misconfigured SLO is as harmful as a missing one, an SLO counts as correct only if its type, metric, comparator, qualifier (highest/lowest), value, and unit all equal those of a ground-truth SLO. Pooling true positives, false positives, and false negatives over all intents gives the strict micro- ${ \bf \nabla } \cdot { \cal F } _ { 1 }$ , which weights every SLO equally. We also report the exact-match rate $\begin{array} { r } { \mathrm { \bar { E } M } = \frac { 1 } { | V | } \sum _ { i \in V } \mathbf { 1 } \bar { [ \hat { S } _ { i } = } } \end{array}$ $S _ { i } ]$ over the valid intents V, because the orchestrator can enforce an intent only when its whole SLO set is right.

Intent validation (RQ2). The rejection rate $\rho$ is the share of invalid intents answered with an empty SLO set $\hat { S } _ { i } = \varnothing$ . As over-eager rejection harms service availability, we also report the false-rejection rate $\phi ,$ the share of valid intents answered with $\hat { S } _ { i } = \bar { \boldsymbol { \mathcal { O } } }$

Hallucination (RQ3). Not every deviation from the ground truth is a hallucination. A value off in its last digit, “<" for $\ " \leq \ "$ or an empty value is a mistake, as the model asserts nothing that the inputs contradict. An SLO is hallucinated if it asserts a requirement or value that the intent text, its numbers, and the cluster snapshot do not support, e.g., a placement on a layer the operator never named, or a value that is not the requested metric's value in snapshot. The hallucination rate is the share of parseable responses R with at least one hallucinated SLO, $\begin{array} { r } { \eta = \frac { 1 } { | \mathcal { R } | } \sum _ { r \in \mathcal { R } } \mathbf { 1 } [ \exists \sigma \in \hat { S } _ { r } : \sigma } \end{array}$ hallucinated], since one unsupported requirement suffices to misconfigure a deployment.

The 28,640 responses from 10 models × 4 prompts × 716 intents contain 15,181 distinct (intent, SLO) pairs that do not match the ground truth, which is far beyond manual labeling. For valid intents, the hallucination rules (in the repository) are applied as deterministic rules against intent and snapshot. For invalid intents, we must infer ground-truth SLOs from free text. We used an LLM-assisted procedure with Claude Sonnet 5.5, a model not among those evaluated, which read each of the invalid intent texts under these rules and produced a per-intent specification (layers, metrics, directions, numbers), which was then applied automatically to every returned SLO. A human annotator checked 250 stratified labels (125 valid, 125 invalid), covering every rule at least twice and every model 25 times, and agreed with 93% of cases.

Energy and emissions (RQ4). Energy efficiency is a design requirement of next-generation networks, and an always-on translation service adds to the energy budget of the infrastructure. We measure the GPU energy of each request with the NVIDIA management library (NVML) energy counter, the same source used by inference-energy benchmarks [17] We report energy per intent (Wh), and operational emissions ${ \mathrm { C O _ { 2 } e } } = E \times { \mathrm { P U E } } \times { \mathrm { C I } }$ , where E is the energy, PUE is the power usage effectiveness of the data center and CI is the carbon intensity of the grid. Load levels, PUE, and grid scenarios are given in Section IV-D.

Robustness (RQ5). For each phrasing variant, we report the change $\Delta M = M _ { \mathrm { v a r i a n t } } - M _ { \mathrm { o r i g i n a l } }$ of a metric $M \in \{ F _ { 1 } , \rho \}$ from RQ1 and RQ2, so that a negative value is a loss caused by phrasing alone.

TABLE I  
THE THREE PHRASING VARIANTS OF ONE VALID INTENT (RQ5).
<table><tr><td>capacity.</td><td>Orig. For svc, please run it on the edge side, make sure the target node has at least 2 GB of memory, and pick the node with the highest storage</td></tr><tr><td>C1</td><td>We are retiring a couple of old machines at the end of the month, and the recommender is waiting for a new home. Please run svc on the edge side with a node that offers at least 2 GB of memory, and choose the one with the highest storage capacity. I&#x27;ll pass the result to the migration team tomorrow, so thank you for a quick reply.</td></tr><tr><td>C2</td><td>The recommender has moved between hosts twice since spring and the team wants it settled. Choose the node with the highest storage capacity for svc, make sure it has 2 GB of memory or more, and keep it on the edge side.</td></tr><tr><td>C3</td><td>Service: svc - Layr: edge side pls - Memory: node need at least 2 GB of mem - Storage: pick the node what have the highest storge capacity Migration is soon so asap if you can. Thx, i really apreciate it.</td></tr></table>

## E. Robustness to Operator Phrasing

The dataset intents are short, one-line requests, whereas operators can write intents in quick notes, often with context, in lists, or in a hurry. We rewrote each intent in three compound variants (Table I): C1 (wrapped request) keeps the request close to the original or lightly reworded and wraps it with task-related opening and closing sentences; C2 (restructured request) rewords the request, reorders its constraints, and adds one sentence on the service's role; C3 (noisy list) turns it into a list with typically 3-5 typos and common shortcuts (e.g., "pls", "mem"), followed by a short closing.

The rewriting rules preserve the meaning: numbers, units, service names, layer words, etc., are never changed by a typo. Added sentences contain no numbers, metrics, or hints about validity; and every invalid intent keeps its defect (e.g., a malformed fragment stays broken). Each variant therefore keeps the ground-truth SLO set of its original intent. The 716 intents yield 2,148 variants, which we release as a robustness benchmark. They were written one intent at a time with LLM assistance (Claude Sonnet 5.5), following the definitions (C1– C3) above, and automatic tests check that every variant keeps the numbers, units, service, layer, and superlative words of its original, and that added sentences are unique and digitfree. A human reviewer then checked all three variants of 100 randomly drawn intents (50 valid, 10 per level; 50 invalid, 12– 13 per category) with 95% agreement. For each variant, the reviewer judged whether it keeps the ground-truth SLOs of its valid intents and, for an invalid intent, also its defect.

## IV. RESULTS

## A. RQ1: Translation Performance

Table II reports the translation performance of all 40 configurations. $F _ { 1 }$ ranges from 0.101 (Llama-3.2-3B, zs-cot) to 0.943 (Gemma-4-E4B, fs-cot), and the best exact-match rate is 0.879. Two configurations reach the $F _ { 1 }$ reported for the full Intent Engine pipeline with GPT-4.1 mini (0.941) [5].

The ranking by mean $F _ { 1 }$ is led by Qwen3.5-9B (0.929) and Qwen3.5-4B (0.912). The 4B Qwen model has a higher mean than the 8B model of every other family. Within a family, gain from the larger model ranges from 0.016 (Qwen3.5) to 0.393 (Granite 4.1) in mean $F _ { 1 }$ . For an operator, choice of model family is therefore as important as the parameter budget.

TABLE ⅡI  
TRANSLATION PERFORMANCE OF THE TEN LIGHTWEIGHT LLMS ON THE 521 VALID INTENTS; MEAN IS THE $F _ { 1 }$ AVERAGED OVER THE FOUR SETUPS. BEST VALUE PER COLUMN IN BOLD, SECOND BEST UNDERLINED.
<table><tr><td rowspan="2">Family</td><td rowspan="2">Size</td><td colspan="2"> $_ { z S }$ </td><td colspan="2"> $\pounds _ { S }$ </td><td colspan="2">zs-cot</td><td colspan="2">fs-cot</td><td rowspan="2">Mean  $F _ { 1 }$ </td></tr><tr><td> $F _ { 1 }$ </td><td>EM</td><td> $F _ { 1 }$ </td><td>EM</td><td> $F _ { 1 }$ </td><td>EM</td><td> $F _ { 1 }$ </td><td>EM</td></tr><tr><td>Qwen3.5</td><td>4B</td><td>0.918</td><td>0.820</td><td>0.910</td><td>0.774</td><td>0.892</td><td>0.772</td><td>0.929</td><td>0.845</td><td>0.912</td></tr><tr><td></td><td>9B</td><td>0.921</td><td>0.835</td><td>0.922</td><td>0.802</td><td>0.942</td><td>0.879</td><td>0.930</td><td>0.823</td><td>0.929</td></tr><tr><td>Gemma 4</td><td>E2B</td><td>0.591</td><td>0.397</td><td>0.602</td><td>0.399</td><td>0.850</td><td>0.695</td><td>0.886</td><td>0.747</td><td>0.732</td></tr><tr><td></td><td>E4B</td><td>0.726</td><td>0.403</td><td>0.872</td><td>0.733</td><td>0.883</td><td>0.754</td><td>0.943</td><td>0.879</td><td>0.856</td></tr><tr><td>Ministral 3</td><td>3B</td><td>0.408</td><td>0.159</td><td>0.714</td><td>0.493</td><td>0.479</td><td>0.236</td><td>0.847</td><td>0.702</td><td>0.612</td></tr><tr><td></td><td>8B</td><td>0.889</td><td>0.779</td><td>0.917</td><td>0.795</td><td>0.896</td><td>0.779</td><td>0.920</td><td>0.825</td><td>0.905</td></tr><tr><td>Granite 4.1</td><td>3B</td><td>0.248</td><td>0.188</td><td>0.651</td><td>0.359</td><td>0.269</td><td>0.150</td><td>0.522</td><td>0.253</td><td>0.423</td></tr><tr><td></td><td>8B</td><td>0.764</td><td>0.587</td><td>0.855</td><td>0.672</td><td>0.776</td><td>0.601</td><td>0.867</td><td>0.716</td><td>0.816</td></tr><tr><td>Llama 3</td><td>3B</td><td>0.114</td><td>0.035</td><td>0.440</td><td>0.167</td><td>0.101</td><td>0.006</td><td>0.470</td><td>0.140</td><td>0.281</td></tr><tr><td></td><td>8B</td><td>0.448</td><td>0.286</td><td>0.749</td><td>0.507</td><td>0.433</td><td>0.200</td><td>0.756</td><td>0.501</td><td>0.597</td></tr></table>

Our fs-cot gives the highest $F _ { 1 }$ for $^ 8$ of the 10 models and the second highest for the other two. The gain is largest for the models that are weak in zs: Ministral-3-3B rises from 0.408 to 0.847 and Gemma-4-E2B from 0.591 to 0.886.

Figure 2 separates explicit SLOs from implicit SLOs (whose value must be checked from snapshot). With $\pounds s { \mathrm { - } } { \mathsf { C O t } }$ , the four Qwen and Gemma models reach an $F _ { 1 }$ of 0.919–0.997 on explicit SLOs but only 0.688–0.739 on implicit ones.

## B. RQ2: Intent Validation

An invalid intent that is translated into SLOs leads to provisioning that the operator never asked for. Figure 3 shows how often each configuration rejects such intents. With fs-cot, the four strongest models reject almost all of them. Zeroshot (zs) instructions are not enough, as $\rho$ ranges from 0.00 (Llama-3.2-3B) to 0.80 (Gemma-4-E4B), and only three models exceed 0.6. Examples and reasoning together (fs-cot against zs) raise $\rho$ for all ten models.

Among the 49 conflicting intents, contradictory bounds (e.g., a utilization below 40% and above 80%) and requests for both the highest and the lowest value of one metric are all rejected by Qwen3.5 and Ministral-3-8B with fs-cot. Intents that require the edge and the cloud at once are rejected by no model with $_ { 2 S }$ (0 of 12) and by at most 10 of 12 with any setup (Ministral-3-8B, fs-cot). A layer conflict reads like a broad placement preference, so a model can satisfy it literally by writing both placements.

![](images/122a8b51e1d9e1fff33ee4387b1030918420077f7f567848102da114e792d5a9.jpg)  
Fig. 2. Strict $F _ { 1 }$ of the two best model families with fs-cot on explicit SLOs (thresholds stated in intent), implicit SLOs (values must be read from snapshot), and all SLOs. Errors lie mostly in implicit SLOs.

![](images/062491cbdb422a9a749b5b31f99583f85f228825f1678edb02454206489b241b.jpg)  
Fig. 3. Rejection rate $\rho$ of each model and setup on the 195 invalid intents (darker = more rejected). Examples and reasoning (fs-cot) reject most; whereas zero-shot setup and the Llama models reject least.

Over all configurations, 3,351 of the 7,800 answers to invalid intents contain SLOs. The most common is a single placement (22–50% of them, depending on the category), which keeps the readable part of the request and silently drops the rest, and 363 answers to conflicting intents contain both halves of the contradiction.

Without reasoning, no model rejects any of the 521 valid intents. With reasoning, at most 6.9% are rejected (Granite-4.1- 3B, fs-cot). Qwen3.5-9B and Ministral-3-8B reject none of the valid intents. Reasoning thus makes most weaker models over-cautious on valid intents rather than more discerning.

## C. RQ3: Hallucination

Figure 4 shows the hallucination rate per configuration. Reasoning lowers the rate for all ten models with zero-shot prompts and for nine of ten with few-shot prompts, and examples alone lower it for eight. The best configurations hallucinate in about one response in a hundred: Gemma-4- E4B (0.008) and Qwen3.5-4B (0.013) with fs-cot.

![](images/166d3d75f90efdf7116af78e3290586e7a29e5c1605a32bc3a8173ce0fbe5717.jpg)  
Fig. 4. Hallucination rate η of each model and setup over all 716 intents (lighter = more trustworthy). Examples and reasoning (fs-cot) lower it for almost every model, down to about 1% in the best configurations.

GPU energy per intent (Wh), one request at a time  
TABLE III  
HALLUCINATED SLOS OF BOTH KINDS: AN UNSUPPORTED VALUE (GROUNDED FROM THE WRONG METRIC) AND AN UNSUPPORTED REQUIREMENT (AN INVENTED PLACEMENT); SVC = SERVICE
<table><tr><td>Intent (abridged)</td><td>Ground-truth SLOs</td><td>Model (setup): hallucinated SLO</td><td>Why it is a hallucination</td></tr><tr><td>&quot;Deploy svc ... at least 50 GB of storage capacity and pick the node with the lowest current memory utilization&quot;(valid)</td><td>storage cap. ≥ 50 GB; memory util. ≤ [lowest] 34.27 %</td><td>Qwen3.5-9B (zs): memory util. ≤ [lowest] 18.0 %</td><td>18.0 is the storage utilization of the correct node: right node, wrong metric</td></tr><tr><td>“I would like to place svc and keep CPU utilization below 40 %&quot;(valid)</td><td>CPU util. ≤ 40 %</td><td>Qwen3.5-9B (fs-cot): placement = edge</td><td>a placement layer the intent does not name</td></tr></table>

Pooled over its two models and four setups, η per family ranges from 0.080 (Qwen3.5) through 0.191 (Gemma 4), 0.203 (Ministral 3) and 0.279 (Granite 4.1) to 0.610 (Llama 3). The larger model hallucinates less in four of the five families; Qwen3.5 is the exception, where the 9B model is slightly worse than the 4B model (0.093 against 0.068).

On valid intents, eight of the ten models more often write an unsupported value (from the wrong node, metric, or unit conversion) than an unsupported requirement. On invalid intents, they fill in what the text leaves open, mostly a placement layer or a metric. Table II shows two examples where models hallucinated the requirements.

## D. RQ4: Energy and Emissions

An intent-translation service runs beside the infrastructure it manages, so its energy adds to the network's energy budget. We measured it for the three most accurate families in RQ1. Each model ran with vLLM on its own H100 NVL GPU with the prompts of RQ1, and no other process on it. Energy was read from the NVML counter [17]. We measured 11,808 requests at three load levels: one request at a time (as in a lightly used service), 16 and 64 concurrent requests (where batching raises efficiency at the cost of latency). No intent was repeated, so no complete prompt was served from the cache. Energies for all 521 valid intents are estimated from a per-model linear fit on generated tokens $( R ^ { 2 } \geq 0 . 9 9 6 )$ .

Served one request at a time, an intent costs 0.044–0.148 Wh without reasoning and 0.18–0.58 Wh with it (Figure 5). Reasoning multiplies the energy per intent by 3.5-7.1×. This buys 26–28 $F _ { 1 }$ points for Gemma-4-E2B, 13 for Ministral-3-3B (fs), and at most 2 points for Qwen3.5 and Ministral-3- 8B. Qwen3.5-9B often answers the fs-cot prompt directly, without writing out its reasoning, so its fs-cot costs are similar to fs. Its zs-cot answers do contain reasoning.

Serving 64 requests at once instead of one at a time lowers the energy per intent by a factor of 3.2–20.2 (median 6.1). This happens most for the reasoning setups, whose long generations benefit most from batching. The price is latency, as the 95th-percentile response time rises from 0.8-3.2 s to 10–27.5 s without reasoning and from 2.1–20.4 s to 24.5– 71.2 s with it. So an operator can batch delay-tolerant intents (e.g., planned deployments) and serve urgent ones individually. At realistic rates, however, the GPU's idle power dominates the energy. For a dedicated GPU (idle draw 101-108 W) loaded to at most 50% of its throughput under a 30 s latency bound, idle draw is 81–97% (median 93%) of the 2.5–3.2 kWh it consumes per day at 10,000 intents. With an industryaverage PUE of 1.56 [19] and the 2025 world grid intensity of 0.458 kg CO2e/kWh [20], [21], this is $0 . 1 8 { - } 0 . 6 7 \mathrm { g } \mathrm { C O } _ { 2 } \mathrm { e }$ per intent. The data center's location changes it about 16-fold, from 0.016–0.061 g in France to 0.26–0.98 g in India.

![](images/25e9ce7d2da7717901589bf0f45ef745cedf6f1032faa649a3afa3d984acdbbd.jpg)  
Fig. 5. $F _ { 1 }$ against GPU energy per valid intent (one request at a time), one panel per model (columns: families; larger model on top), one point per setup. Arrows add reasoning (dashed: zs→zs-cot; solid: fs→fs-cot). Steep arrows are reasoning worth its energy; flat ones are not.

## E. RQ5: Robustness to Operator Phrasing

A reliable translator must hold for phrasings it was not evaluated with. We fed the 716 intents and their 2,148 variants (section III-E) with all four prompts to the two most accurate families, Qwen3.5 and Gemma $^ { 4 , }$ since phrasing losses are hard to separate from baseline errors in weaker models. For each configuration, we compare $F _ { 1 }$ on the valid variants and $\rho$ on the invalid ones with the original intents.

Figure 6 shows that rephrasing lowers the rejection rate in 15 of the 16 configurations by up to 18.8 points (Gemma-4- E4B, zs). $F _ { 1 }$ drops in 14, by up to 9.4 points (Gemma-4-E2B, zs-cot). Zero-shot is the most fragile setup, losing 8.7–18.8 points of rejection. Few-shot examples reduce the losses, so that with fs or fs-cot, Qwen3.5 and Gemma-4-E4B lose 4.4–11.4 points of rejection and at most 2.5 points of $F _ { 1 } .$ The only configuration that gains on both metrics, Gemma-4-E2B with fs $( F _ { 1 } + 4 . 0 , \rho + 0 . 2 )$ , does so from a weak baseline $( F _ { 1 }$ $0 . 6 , \rho = 0 . 0 1 5 )$ , not through robustness. Rephrasing thus costs some performance, but its more serious effect is that models accept invalid requests which they rejected in the short form.

![](images/59fc1930af401622c151f10e773a1cd6d4ee1abe932ff5d30de074062e9a0f02.jpg)  
Fig. 6. Effect of operator phrasing per model and prompting setup. Change, in percentage, between the variants C1–C3 (pooled) and the original intents, for strict F1 on valid intents (left) and rejection rate on invalid intents (right).

Figure 7 shows the impact of the three variants separately. Restructuring (C2) and the noisy list (C3) hide contradictions, so the rejection of conflicting intents falls. Wrapping (C1) instead makes broken fragments read like normal requests, lowering the rejection of malformed intents.

## V. CONCLUSION

We asked how far lightweight LLMs can go in intentto-SLO translation. Our findings show that with few-shot reasoning, models of 4–9B parameters can reach a strict F1 up to 0.943, stop up to 98% of invalid intents, and rarely invent QoS requirements, at a fraction of a watt-hour per intent. Their limits are implicit values, intent validation that weakens under natural phrasing, and energy dominated by idle time. In practice, few-shot prompting should be the default, reasoning used for models where it adds performance, and the translator paired with a deterministic check of implicit values and SLO consistency. Our evaluation uses a single cluster snapshot, and energy was measured on an on-premises GPU. On edge accelerators, the absolute values will differ, while the relative effects of reasoning and batching are expected to hold. These findings open research directions for AI-native network management, such as open benchmarks with realistic operator phrasing to harden intent validation, and energyaware placement and scheduling of the translator on shared accelerators. Extending the evaluation to intent assurance in

![](images/d3f1ad77783fde1edd00f86b947b379e66e9239b9a36400b2a5df637461657f4.jpg)  
Fig. 7. Change in rejection rate of invalid intents for each phrasing variant.

3GPP intent-driven management and to multi-turn clarification of ambiguous intents will show how far lightweight models can carry closed-loop QoS management beyond translation.

## REFERENCES

[1] A. Clemm, L. Ciavaglia, L. Z. Granville, and J. Tantsura, "Intent-based networking – concepts and definitions," IETF, RFC 9315, Oct. 2022.

[2] 3GPP, “Management and orchestration; intent driven management services for mobile networks," 3rd Generation Partnership Project (3GPP), Technical Specification TS 28.312, V20.1.0, Sep. 2026.

[3] M. A. Habib, P. E. Iturria-Rivera, Y. Ozcan, M. Elsayed, M. Bavand, R. Gaigalas, and M. Erol-Kantarci, "Harnessing the power of LLMs, informers and decision transformers for intent-driven RAN management in 6G," IEEE Trans. Netw. Sci. Eng., vol. 13, pp. 4187–4206, 2026.

[4] L. Ferrari, M. Zaccarini, J. Santos, F. Poltronieri, M. Tortonesi, C. Stefanelli, and F. De Turck, "From intent to manifest: Evaluation of LLM-driven generation of custom resources in Kubernetes," in Proc. IEEE/IFIP Network Operations and Management Symp. (NOMS), 2026.

[5] K. Islam and R. N. Calheiros, "Intent engine: Natural-language intent translation for intent-driven orchestration in the compute continuum," Journal of Systems Architecture, vol. 179, p. 103938, 2026.

[6] D. M. Manias, A. Chouman, and A. Shami, “Semantic routing for enhanced performance of LLM-assisted intent-based 5G core network management and orchestration," in Proc. IEEE Global Communications Conf. (GLOBECOM), 2024, pp. 2924–2929.

[7] M. A. Habib, P. E. Iturria Rivera, Y. Ozcan, M. Elsayed, M. Bavand, R. Gaigalas, and M. Erol-Kantarci, "LLM-based intent processing and network optimization using attention-based hierarchical reinforcement learning," in Proc. IEEE Wireless Communications and Networking Conf. (WCNC), 2025.

[8] C. Wang, M. Scazzariello, A. Farshin, S. Ferlin, D. Kostić, and M. Chiesa, "NetConfEval: Can LLMs facilitate network configuration?" Proc. ACM Netw., vol. 2, pp. 1–25, 2024.

[9] A. Angi, A. Sacco, and G. Marchetto, "LLNet: An intent-driven approach to instructing softwarized network devices using a small language model," IEEE Trans. Netw. Service Manag., vol. 22, no. 4, pp. 3403– 3418, 2025.

[10] W. L. Phone, B. El Boudani, T. Dagiuklas, and S. Ghosh, "Performance comparison of IBN orchestration using LLM and SLMs," in Proc. IEEE Int. Conf. Communications (ICC), 2026.

[11] GSMA, "GSMA Open-Telco LLM benchmarks launches to advance AI in telecoms," Press release, Feb. 2025, accessed: Oct. 29, 2026. [Online]. Available: https://www.gsma.com/newsroom/press-release/ gsma-open-telco-llm-benchmarks-launches-to-advance-ai-in-telecoms/

[12] S. Acharya and B. Kantarci, "Intent2QoS: Language model-driven automation of traffic shaping configurations," in Proc. IEEE Int. Conf. Communications (ICC), 2026.

[13] N. Raptis, J. P. Fonseca, G. W. Adhane, K. Ramantas, and C. V. Verikoukis, “Reliable LLM-enabled intent-driven management: A validation pipeline for network automation," in Proc. IEEE Int. Conf. Communications (ICC), 2026.

[14] T. Ahmed, N. Piovesan, A. De Domenico, and S. Choudhury, "Linguistic intelligence in large language models for telecommunications," in Proc. IEEE Int. Conf. Communications Workshops, 2024, pp. 1237–1243.

[15] K. Hammar, T. Alpcan, and E. C. Lupu, "Hallucination-resistant security planning with a large language model," in Proc. IEEE/IFIP Network Operations and Management Symp. (NOMS), 2026.

[16] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung, "Survey of hallucination in natural language generation," ACM Comput. Surv., vol. 55, no. 12, pp. 1–38, 2023.

[17] J.-W. Chung, J. J. Ma, R. Wu, J. Liu et al., "The ML.ENERGY benchmark: Toward automated inference energy measurement and optimization," arXiv:2505.06371, 2025.

[18] Y. Zhang, H. Hu, and G. Gu, "NetInspector: Measuring and improving LLM capabilities for reliable intent-based networking policy generation," arXiv:2609.21103, 2026.

[19] Uptime Institute, "Uptime institute global data center survey 2024," Uptime Institute, Keynote Report 146M, Jul. 2024.

[20] Ember,“Yearly electricity data," https://ember-energy.org/data/ yearly-electricity-data/, 2026, carbon intensity of electricity, 2025 values; accessed: Oct. 26, 2026.

[21] D. Patterson, J. Gonzalez, Q. Le, C. Liang et al., "Carbon emissions and large neural network training," arXiv:2104.10350, 2021.