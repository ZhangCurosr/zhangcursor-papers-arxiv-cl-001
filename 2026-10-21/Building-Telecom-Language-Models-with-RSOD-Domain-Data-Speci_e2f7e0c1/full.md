# Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation

Anil Srirangapatna Nagesh<sup>1</sup>, Nitin Gupta<sup>1</sup>, Anuj Taneja<sup>1</sup>, Vivek Singh<sup>1</sup>, Shane Eleniak<sup>1</sup>, Theo Zhao<sup>1</sup>, Moulidharan N<sup>1</sup>, and Shashank Keshetty<sup>1</sup>

<sup>1</sup>Calix AI Labs

## Abstract

Telecom language models must combine broad network ing knowledge with specialist behavior across routing, broadband services, access technologies, and operational diagnostics while remaining practical to deploy as a single model. We present an end-to-end framework that builds source-grounded synthetic training data from structured networking taxonomies and expert graphs, materializes 212,559 routed training records across fourteen networking routes, trains six frozen specialist LoRA teachers over one shared continued-pretrained backbone, and consoli dates their behavior into one deployable student adapter using Routed Specialist On-Policy Distillation (RSOD). In RSOD, the committed student generates each trajectory, the route-assigned specialist teacher-forces the exact native token sequence, compact teacher hidden states are reconstructed through the shared frozen output head, and only the student LoRA is updated. A controlled recon struction validation shows mean absolute logit error of $7 . 2 6 \times 1 0 ^ { - 6 }$ with 100% top-1 agreement between native and reconstructed teacher logits. On four RTX 6000 Ada GPUs, the completed 2,000-step run produced 16,000 trajectories with zero observed student-version or token ID mismatches. Under the primary route-conditioned evaluation over 2,850 held-out questions, the raw Meta Llama-3.1-8B-Instruct model scores 12.8661/25 micro (13.0172/25 macro), the combined routed six-teacher system scores 21.7641/25 micro, and the consolidated RSOD student, released as Llama-CalixCLM v1, reaches 21.7112/25 micro (21.8006/25 macro) on the same five dimension LLM-judge rubric. Under a separate neutral deployment-like prompt, the same student reaches 20.0028/25 micro (20.0778/25 macro). The consolidated student remains within 0.0529 micro-score points of the combined routed-teacher system while retaining a numeri cally validated and auditable train–serve execution path.

## 1 Introduction

Telecom language models must combine broad networking knowledge with operationally specific behavior across routing, transport, broadband services, access technologies, command-line diagnosis, and troubleshooting. Specialist adaptation can sharpen domain behavior, but deploying multiple specialist endpoints complicates model management and serving. Our framework keeps one shared backbone, trains lightweight specialist LoRA adapters, and then consolidates their behavior into one student adapter for deployment.

Parameter-eficient adaptation keeps the backbone frozen while learning low-rank updates [2]. We extend that foundation into a complete telecom post-training workflow spanning structured domain data, continued pretraining (CPT), specialist adaptation, routed on-policy distillation, auditable train–serve synchronization, and networking-specific evaluation.

Figure 1 summarizes the workflow. Our research question is: Can multiple telecom-specialized LoRA teachers be consolidated into one deployable adapter using student-onpolicy distillation while retaining route-conditioned specialist quality and exact token/version consistency? In the evaluated run, the raw model scores 12.8661/25 micro, the combined routed six-teacher system scores 21.7641/25 micro, and the consolidated student reaches 21.7112/25 micro under the primary route-conditioned protocol; all 16,000 recorded version checks and token-ID checks match.

Our contributions are:

• An end-to-end telecom specialization framework connecting structured knowledge-grounded data generation and 212,559 routed training records to six specialist groups, fourteen canonical routes, specialist LoRA posttraining, consolidation, and evaluation.

• RSOD with exact routed supervision: the committed student generates the trajectory, deterministic route ownership selects one frozen specialist, and the teacher scores the student’s exact native token IDs before forward-KL optimization of only the student LoRA.

• A memory-conscious execution architecture that shares one tensor-parallel teacher backbone across specialist LoRAs, transports compact teacher hidden states for local logit reconstruction, and maintains gradients and optimizer state only for the student LoRA.

• A fourteen-route telecom networking evaluation protocol with fixed route-specific system prompts and a common five-dimension scoring framework applied consistently across compared models.

Table 1: Abbreviations used in the paper.
<table><tr><td>Term</td><td>Meaning</td></tr><tr><td>RSOD</td><td>Routed Specialist On-Policy Distillation</td></tr><tr><td>CPT</td><td>Continued Pretraining</td></tr><tr><td>SFT</td><td>Supervised Fine-Tuning</td></tr><tr><td>KD</td><td>Knowledge Distillation</td></tr><tr><td>KL</td><td>Kullback-Leibler divergence</td></tr><tr><td>LoRA</td><td>Low-Rank Adaptation</td></tr><tr><td>PEFT</td><td>Parameter-Efficient Fine-Tuning</td></tr><tr><td>DDP</td><td>Distributed Data Parallel</td></tr><tr><td>TP /TP2</td><td>Tensor Parallelism / two-way tensor paral- lelism</td></tr><tr><td>BF16</td><td>Brain Floating Point 16-bit</td></tr><tr><td>VRAM</td><td>Video RAM</td></tr><tr><td>BNG</td><td>Broadband Network Gateway</td></tr><tr><td>MPLS</td><td>Multiprotocol Label Switching</td></tr><tr><td>PON</td><td>Passive Optical Network</td></tr><tr><td>QoS</td><td>Quality of Service</td></tr><tr><td>BGP</td><td>Border Gateway Protocol</td></tr><tr><td>OSPF</td><td>Open Shortest Path First</td></tr><tr><td>IS-IS</td><td>Intermediate System to Intermediate Sys- tem</td></tr></table>

• An auditable RSOD execution protocol and evidence layer combining version-consistent rollouts, exact-token specialist supervision, compact teacher state reconstruction, machine-verifiable token/version provenance, numerical fidelity validation, and matched systems/quality measurements.

RSOD builds on established LoRA, on-policy distillation, multi-teacher selection, and shared multi-adapter serving. Its contribution is the integrated configuration of fixed route-to-specialist assignment, student-generated trajectories, exact-token teacher forcing, shared-head signal reconstruction, version-acknowledged adapter commits, and telecom-specific data/evaluation mappings. Table 2 summarizes this positioning.

## 2 Related Work

Knowledge and on-policy distillation. Knowledge distillation trains a student to match a teacher distribution using soft targets [1]. Generalized Knowledge Distillation (GKD) addresses autoregressive train–test mismatch by training on student-generated trajectories [3]. Nemotron-Cascade 2 applies multi-domain on-policy distillation from strong intermediate teachers [12], while Ma et al. study multi-teacher on-policy distillation for capability integration [13]. RSOD specializes this design space around deterministic telecom route ownership, exact-token teacher forcing, compact teacher-signal transport, and version-consistent train–serve synchronization.

Specialist consolidation and model merging. Multi-teacher methods study example-wise teacher selection and aggregation [4]. Parameter-space methods such as TIES-Merging combine specialized models directly [14]. RSOD instead performs behavioral consolidation: specialist adapters remain frozen and separate while one student learns from the assigned teacher on its own trajectories.

Parameter-eficient and multi-adapter serving. LoRA provides frozen-backbone low-rank adaptation [2].

Punica and S-LoRA establish shared-backbone serving for many adapters [5, 6], and SGLang provides an eficient inference runtime [7]. RSOD uses shared multi-adapter serving inside a synchronized training loop in which successive rollouts use the newly committed student version.

Telecom models and evaluation. TeleQnA, TelecomGPT, TeleMath, Tele-LLMs, NETLLMBENCH, and the GSMA Open-Telco benchmark efort motivate telecom-specific evaluation beyond general-purpose language-model scores [8, 9, 10, 15, 11, 16]. Our evaluation applies the same route-specific prompt and fivedimension scoring protocol to the raw model, the routed CPT+specialist teacher system, and the consolidated student.

## 3 Domain Data and Specialist Post-Training

## 3.1 Knowledge-Grounded Corpus Construction

Training data are produced through a structured knowledge-generation pipeline rather than assembled as a flat question–answer collection. For each networking domain, a protocol taxonomy defines the intended knowledge space and is expanded into an expert graph containing concepts, protocol fields, semantic relationships, workflows, operational behavior, failure modes, and diagnostic evidence. Graph nodes, semantic edges, and higher-order subgraphs are converted into generation units. A planning stage translates each unit into explicit learning obligations and CPT/SFT generation intents, after which a teacher model realizes the corresponding natural-language records.

The underlying technical knowledge is source-derived, whereas the final training records are synthetic realizations. Depending on the domain, the knowledge-building stage draws from standards and RFCs, protocol registries, architecture and specification documents, vendor-neutral technical documentation, implementation evidence, operational troubleshooting knowledge, and selected vendor documentation when implementation behavior is not fully specified by standards. Synthetic generation is therefore used for linguistic realization rather than unconstrained knowledge creation: CPT passages, explanations, questions, scenarios, comparisons, diagnostic exercises, and misconception-correction examples are generated from the structured representation.

Validation is applied throughout the pipeline. Taxonomy and graph stages are checked for structural coverage, atomicity, missing concepts, relationships, and weakly grounded nodes; planning checks verify that learning obligations map to CPT/SFT intents; generated corpora are checked for missing or extra planned records, empty outputs, identifier duplication, exact duplication, repeated textual patterns, planner/meta-instruction leakage, placeholder content, length distributions, and diversity. Diagnostic evaluations used during pipeline development could inform taxonomy and graph refinement. The final 2,850-question held-out benchmark reported in this paper was frozen separately and was not used to drive taxonomy, expert-graph, or training-data revisions.

Table 2: RSOD combines established building blocks into an audited telecom-specialist consolidation workflow.
<table><tr><td>Dimension</td><td>Established building block</td><td>RSOD configuration</td></tr><tr><td>Trajectories</td><td>Student-on-policy distillation</td><td>Route-conditioned student trajectories generated by the committed student</td></tr><tr><td>Teacher choice</td><td>Multi-teacher selection or aggregation</td><td>Deterministic route-to-specialist assignment; one teacher per trajectory</td></tr><tr><td>Token alignment</td><td>Teacher scoring of generated output</td><td>Exact native token-ID teacher forcing on the student&#x27;s sequence</td></tr><tr><td>Signal transport</td><td>Logits or local teacher execution</td><td>Teacher hidden-state transport with local shared LM-head logit reconstruction</td></tr><tr><td>Model serving</td><td>Shared multi-adapter serving</td><td>One shared teacher backbone plus version-acknowledged student adapter commits</td></tr><tr><td>Validation</td><td>Aggregate training metrics</td><td>Direct token-identity and serving-version audits over the run</td></tr></table>

![](images/2886a09e029d56953520d5636c344d3e2f973277716b7a32510021386cd6a547.jpg)  
Figure 1: End-to-end framework. Structured networking data supports one shared CPT backbone and six frozen specialist LoRAs. RSOD routes each training trajectory to one specialist, trains a single student adapter, records run provenance and audits, and produces one deployable student.

## 3.2 Routed Training and Held-Out Inventory

The executed experiment materializes 212,559 routed training records across fourteen canonical routes and six specialist groups. The same route identity is carried from data preparation into specialist ownership, deterministic teacher selection, and route-specific evaluation. Each routed JSONL artifact is recorded with a SHA-256 identity in the preparation manifest. Training and held-out artifacts are physically separate: CPT and specialist SFT consume routed train files, while the evaluation inventory contains 2,850 held-out questions in independently hashed files.

Table 3: Routed training and held-out inventory used by the final six-specialist configuration.
<table><tr><td>Specialist</td><td>Train records</td><td>Heldout</td></tr><tr><td>Dynamic Routing</td><td>45,597</td><td>570</td></tr><tr><td>MPLS Services</td><td>26,439</td><td>190</td></tr><tr><td>Broadband Services</td><td>29,360</td><td>570</td></tr><tr><td>Routing Fundamentals</td><td>52,083</td><td>380</td></tr><tr><td>CLI Diagnostics</td><td>20,990</td><td>570</td></tr><tr><td>Edge Access</td><td>38,090</td><td>570</td></tr><tr><td>Total</td><td>212,559</td><td>2,850</td></tr></table>

## 3.3 Shared CPT Backbone and Specialist LoRAs

The shared backbone starts from a pinned Meta-Llama-3.1-8B-Instruct snapshot and undergoes full-parameter continued pretraining over the fourteen routed training files. The executed CPT run consumes 212,559 records, records 47,975,711 training tokens, and completes 3,322 optimizer steps. The CPT stage uses sequence length 1,024, one epoch, bfloat16 precision, fused AdamW, learning rate $1 0 ^ { - 5 }$ with a linear schedule and 10 warmup steps, per-device batch size 1, gradient accumulation 16, and an efective global batch size of 64 across four GPUs.

Table 4: Canonical route-to-specialist mapping and held-out evaluation inventory. The held-out set contains 190 questions per route except BNG/subscriber management with 380.
<table><tr><td>Specialist</td><td>Route</td><td>Train</td><td>Eval.</td><td>Specialist</td><td>Route</td><td>Train</td><td>Eval.</td></tr><tr><td>Dynamic Routing</td><td>BGP</td><td>19,756</td><td>190</td><td>CLI Diagnostics</td><td>Linux tool calling</td><td>5,817</td><td>190</td></tr><tr><td>Dynamic Routing</td><td>OSPF</td><td>9,561</td><td>190</td><td>CLI Diagnostics</td><td>Linux agentic diagnosis</td><td>5,817</td><td>190</td></tr><tr><td>Dynamic Routing</td><td>IS-IS</td><td>16,280</td><td>190</td><td>CLI Diagnostics</td><td>Network fundamentals</td><td>9,356</td><td>190</td></tr><tr><td>MPLS Services</td><td>MPLS VPN</td><td>26,439</td><td>190</td><td>Edge Access</td><td>L2 switching</td><td>11,061</td><td>190</td></tr><tr><td>Broadband Services</td><td>BNG/subscriber mgmt.</td><td>24,735</td><td>380</td><td>Edge Access</td><td>PON/optical</td><td>8,296</td><td>190</td></tr><tr><td>Broadband Services</td><td>QoS</td><td>4,625</td><td>190</td><td>Edge Access</td><td>WiFi gateway</td><td>18,733</td><td>190</td></tr><tr><td>Routing Fundamentals</td><td>IP routing</td><td>23,484</td><td></td><td>190 Routing Fundamentals</td><td>s Multicast</td><td>28,599</td><td>190</td></tr></table>

Each of the six specialist teachers is then trained as a separate LoRA adapter over the same CPT checkpoint. Each specialist consumes only the routed training records assigned to its specialist group; across all six specialists these assigned subsets partition the same 212,559-record routed training inventory used by CPT, while the SFT stage applies instruction formatting and three specialistspecific epochs rather than another full-corpus CPT pass. All specialists use rank 64, α = 128, dropout 0, learning rate $2 \times 1 0 ^ { - 5 }$ , weight decay 0, maximum gradient norm 1.0, micro-batch size 1, gradient accumulation 8, and three SFT epochs with seed 42. The adapted modules are q proj, k proj, v proj, o proj, gate proj, up proj, and down proj; maximum sequence length ranges from 1,024 to 1,536 according to specialist domain.

During RSOD, teacher adapters, token embeddings, backbone, final normalization, and output head remain frozen; only the student LoRA parameters receive gradients. Compatibility therefore requires a common base identity, tokenizer and vocabulary ordering, final normalization, and LM-head/output-projection geometry; these are method requirements for the current RSOD reconstruction path. For an adapted projection $W _ { 0 } ,$ specialist k and the student use separate low-rank updates:

$$
W _ { k } = W _ { 0 } + s _ { k } B _ { k } A _ { k } , \qquad W _ { s } = W _ { 0 } + s _ { s } B _ { s } A _ { s } .\tag{1}
$$

RSOD performs behavioral consolidation through distillation while the specialist adapters remain frozen and separate.

## 4 Routed Specialist On-Policy Distillation

RSOD overview. Routed Specialist On-Policy Distillation (RSOD) consolidates multiple frozen domainspecialist LoRA teachers into one student LoRA without merging teacher weights. The current student generates its own trajectory, a deterministic route map selects exactly one specialist teacher, and that teacher scores the student’s exact native token sequence by teacher forcing. The teacher returns final hidden states rather than dense vocabulary logits; the trainer reconstructs the teacher distribution locally through the shared frozen final normalization and LM head, computes forward KL on generated positions, updates only the student LoRA, and refreshes the new student version before the next rollout.

## 4.1 Routed Student-On-Policy Supervision

At optimizer step n, the deterministic scheduler selects route r and prompt x. A fixed ownership map $\pi ( r )$ selects specialist $k = \pi ( r )$ . The committed student version $v _ { n }$ generates trajectory $y = ( y _ { 1 } , \dots , y _ { T } )$ , and teacher $T _ { \pi ( r ) }$ teacher-forces the complete native token sequence $( x , y )$ The teacher therefore scores exactly the tokens produced by the student. Prompt tokens provide context and generated positions contribute to distillation.

The trajectory is produced by the current student and evaluated by the assigned specialist on the student’s exact tokenization. Route scheduling is used during training to select supervision; deployment uses the single consolidated student adapter.

## 4.2 Compact Teacher Signal and KL Objective

The inference service returns final teacher hidden states $h _ { T }$ instead of a dense vocabulary-logit tensor. Let $g ( \cdot )$ be the model’s frozen final normalization and $W _ { \mathrm { L M } }$ the shared frozen LM-head/output projection. The trainer reconstructs distributions locally:

$$
p _ { T } = \mathrm { s o f t m a x } \left( { \frac { W _ { \mathrm { L M } } g ( h _ { T } ) } { \tau } } \right) ,\tag{2}
$$

$$
p _ { S } = \mathrm { s o f t m a x } \left( \frac { W _ { \mathrm { L M } } g ( h _ { S } ) } { \tau } \right) .\tag{3}
$$

For N trajectories with generated lengths $T _ { i } .$ , the evalu ated configuration uses forward KL:

$$
\mathcal { L } = \frac { \lambda \tau ^ { 2 } } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } }\tag{4}
$$

This is logit-level KD reconstructed from hidden states rather than hidden-state matching. Forward KL preserves the teacher’s probability mass over the student’s own trajectory, providing richer supervision than a hard response target while retaining the on-policy training distribution [3]. The teacher signal transported per position has hidden width d rather than vocabulary width |V|. Across the run, the trainer materialized 67.86 GB of FP32 teacher hidden-state tensors for full sequences; the analytical dense FP32 logit quantity over the 2,926,443 generated/KD positions is 1.50 TB. Figure 3 reports these as complementary teacher-signal accounting views.

![](images/5b4713fc17ad0d100f72dc912a5c0725dddc52403dece80e31fe21cded24618d.jpg)  
Figure 2: Detailed RSOD execution sequence. The deterministic route scheduler chooses one frozen specialist teacher; the committed student generates the native rollout, and the teacher scores the same token sequence by teacher forcing. Teacher and student hidden states are mapped through the shared frozen normalization/output head, forward KL is computed on generated positions, and only the student LoRA is updated. The candidate adapter is refreshed, acknowledged active, and committed before the next rollout.

## 4.3 Compact-Signal Fidelity Validation

We validate the hidden-state transport path independently with a controlled single-GPU RSOD test that executes fresh student initialization, student rollout, teacher forcing, shared LM-head logit reconstruction, KL computation, backpropagation, and a student update. For a 50-token trajectory, native teacher logits and logits reconstructed as $W _ { \mathrm { L M } } g ( h _ { T } )$ are compared at all 50 generated positions over the full 128,256-token vocabulary. The reconstructed logits show mean absolute error $7 . 2 6 \times 1 0 ^ { - 6 }$ , median absolute error 0, maximum absolute error $6 . 2 5 \times 1 0 ^ { - 2 }$ , cosine similarity 1.000000, and 100% top-1 agreement. Exact top-5 and top-10 ranking matches are 98% and 94%, respectively. The nativeversus-reconstructed distribution KL is numerically zero to floating-point precision $( - 4 . 0 4 \times 1 0 ^ { - 9 }$ as computed), while the student-versus-teacher KD loss remains nontrivial at 1.1992. Student parameters receive gradients and change after the optimizer step. These measurements show that shared-head reconstruction preserves the teacher prediction distribution to high numerical fidelity while retaining a usable distillation signal for the student.

## 4.4 Version-Consistent Adapter Commit

After optimization, the student adapter is saved as a candidate, loaded by the tensor-parallel inference service, acknowledged with a changed service identity, and atomically committed before the next rollout. This enforces a version-consistency invariant: the next trajectory must come from the acknowledged committed student. Two additional invariants complete the execution contract: teacher forcing consumes the exact native student token IDs, and transported teacher hidden states reconstruct the intended teacher distribution through the shared frozen output components. The run records expected and serving adapter versions, hashes student-generated and teacherforced token arrays, and retains route, configuration, synchronization, and telemetry evidence. All 2,000 commits completed successfully and all 16,000 recorded version and token-identity checks matched; the controlled reconstruction test independently validates teacher-signal fidelity. Mean prepare-to-commit latency is 6.47 s versus 7.65 s for student rollout.

## 5 Memory-Conscious Execution

## 5.1 Execution Topology

The evaluated workstation contains four NVIDIA RTX 6000 Ada Generation GPUs (49,140 MiB each). Two GPUs run DDP student optimization; the other two host a TP2 inference service containing one shared backbone, six frozen teacher adapters, and the current student adapter. The system is evaluated as a workstation-scale four-GPU configuration.

For K compatible teachers and one student adapter, shared inference weight storage is approximately

$$
M _ { \mathrm { s h a r e d } } \approx M _ { B } + \sum _ { k = 1 } ^ { K } M _ { A _ { k } } + M _ { A _ { s } } .\tag{5}
$$

A deliberately replicated-teacher reference would require

approximately

$$
M _ { \mathrm { r e p l i c a t e d } } \approx K M _ { B } + \sum _ { k = 1 } ^ { K } M _ { A _ { k } } + M _ { A _ { s } } .\tag{6}
$$

The implementation uses resident specialist adapters over one shared inference backbone. The central memory property is therefore six specialists, not six backbones: the TP2 inference pair keeps one 8B backbone resident and overlays six frozen teacher LoRAs plus the current student LoRA. At BF16, an 8B parameter payload is roughly 16 GB before runtime overheads, so avoiding one full backbone per teacher removes the dominant $K M _ { B }$ replication term. On the DDP training pair, each rank holds a frozen backbone replica and trains only the rank-64 student LoRA; backbone gradients and backbone optimizer state are not maintained. The four-GPU design therefore combines shared teacher residency, LoRA-only student optimization, TP inference, DDP training, and compact hidden-state transfer rather than dense teacher-logit transfer. Measured peaks were approximately 34.3 and 34.0 GiB on the two DDP GPUs and 47.5 GiB on each TP inference GPU, for 159.5 GiB aggregate peak usage. The equations describe weight residency; measured VRAM includes activations, KV cache, allocator reserve, runtime workspaces, and adapter-serving overhead. We therefore characterize the architecture as memory-conscious.

## 5.2 Measurement Methodology

GPU memory, utilization, and power are sampled through NVML. Energy is estimated by trapezoidal integration over valid power samples; intervals larger than 2.5 times the per-GPU median sampling interval are counted as telemetry gaps and excluded rather than silently interpolated. Phase latencies are rank-0 host-boundary observations without inserted CUDA synchronization; they characterize the end-to-end control path rather than individual GPU kernels.

Raw telemetry, step timings, trajectory metrics, adapter events, version-commit events, token/version audits, checkpoint metrics, and environment/model/data manifests are retained as the source of truth. Exact artifact hashes are intentionally kept out of the paper body and belong in the release manifest.

## 6 Networking Evaluation

## 6.1 Evaluation Conditions

The primary evaluation is route-conditioned by design: each held-out item is paired with a fixed route-specific system prompt corresponding to its canonical networking route. The same item-level route prompt is used for every compared model, so raw-model, CPT+specialist-teacher, and student scores are evaluated under the same held-out manifest, prompt construction, decoding setup, judge configuration, aggregation, and failure policy. This measures answer quality under the intended operating condition in which route-specific expertise is supplied to the model. We additionally evaluate the final student under a neutral deployment-like prompt, using the same benchmark and scoring framework, to quantify performance without the route-specific expert instruction.

The 2,850-question held-out inventory is stored separately from the routed training files. Each route contributes 190 questions except BNG/subscriber management, which contributes 380. The judge configuration uses the Anthropic alias claude-sonnet-4-5 with temperature 0, a 2,000-token judge response limit, and a 60-second timeout.

## 6.2 Route-Aware Judge and Validation Policy

Evaluation uses a route-aware five-dimension rubric. Each judge call receives the route/category, question, and generated response together with an “expert network engineer” instruction and evaluates up to five anonymized responses per batch. Model, adapter, teacher, and student identities are not included in the judge prompt. The required JSON output contains numeric scores from 0 to 5 for factual correctness, completeness, terminology/command precision, hallucination control, and operational usefulness; the scorer derives and validates the 0–25 total.

The hallucination dimension is reserved for technically false or fabricated claims, while missing or vague content is scored under completeness or terminology. This separation reduces double-penalization and keeps the rubric aligned with operational network-engineering quality. Malformed or inconsistent outputs are rejected under a bounded policy of up to three total attempts, with one- and two-second waits between attempts. Validation rejects wrong item count/order, malformed JSON, missing or out-of-range dimensions, inconsistent totals, and five-item all-zero batches. The scorer records coverage separately from quality so rejected judge batches do not silently become zero-valued model scores.

## 6.3 Aggregation

For model m, the micro average is

$$
\operatorname* { m i c r o } ( m ) = \frac { 1 } { | I _ { m } | } \sum _ { i \in I _ { m } } s _ { m i } ,\tag{7}
$$

and the route macro average is

$$
\mathrm { m a c r o } ( m ) = \frac { 1 } { | R | } \sum _ { r \in R } \bar { s } _ { m r } .\tag{8}
$$

Coverage and judge failures are reported separately from score. The matched primary comparison covers the raw model, the six routed CPT+specialist teachers, and the consolidated RSOD student over the same 2,850-question manifest; the student is also reported under the separate neutral prompting condition.

Table 5: RSOD run summary. Energy is NVML-derived; VRAM is aggregate device usage.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Optimizer steps</td><td>2,000</td></tr><tr><td>Trajectories</td><td>16,000</td></tr><tr><td>KD tokens</td><td>2,926,443</td></tr><tr><td>Active training time GPU-hours</td><td>50,511 s</td></tr><tr><td>Estimated GPU energy</td><td>56.1</td></tr><tr><td></td><td>4.63 kWh</td></tr><tr><td>Peak aggregate VRAM</td><td>159.5 GiB</td></tr><tr><td>Trajectories/s</td><td>0.317</td></tr><tr><td>KD tokens/s</td><td>57.9</td></tr><tr><td>Adapter commits</td><td>2,000 / 2,000</td></tr><tr><td>Version mismatches</td><td>0 / 16,000</td></tr><tr><td>Token-ID mismatches</td><td>0 /  16,000</td></tr></table>

## 7 Experimental Setup

The RSOD run uses the shared 8B CPT backbone artifact cpt-llama31-8b-six-expert-final-oct-1 and a fresh student LoRA with rank 64, α = 128, dropout 0, and the same seven attention/MLP projection targets as the specialists; lm head and embed tokens are protected from adaptation. The deterministic fourteen-route schedule assigns each trajectory to exactly one of the six frozen specialist teachers.

The distillation objective is forward KL with temperature τ = 1.0, weight λ = 1.0, and generated-token-only masking. Student optimization uses AdamW with learning rate 10<sup>−5</sup>, cosine scheduling, 3% warmup, weight decay 0.01, gradient clipping at 1.0, and seed 42. Rollouts use temperature 0, top-p = 1.0, a maximum of 256 generated tokens, and one trajectory per prompt. Eight trajectories are processed per optimizer step for 2,000 optimizer steps and 16,000 trajectories total; checkpoints are recorded every 250 steps.

The four-GPU placement dedicates GPUs 0–1 to DDP student optimization and GPUs 2–3 to TP2 inference. The inference service can keep the six teachers and current student LoRA resident over the shared backbone. Run time configuration, hardware/software identities, model identities, dataset manifests, telemetry, checkpoint metadata, judge rubrics, and evaluation reports are retained as machine-readable run artifacts.

## 8 Results

## 8.1 Systems Results

Table 5 summarizes the run. It consumed 56.1 GPUhours and an NVML-derived 4.63 kWh estimate. Peak aggregate device memory was 159.5 GiB across the four GPUs. All 2,000 adapter commits completed successfully, and all 16,000 recorded student-version and token-identity checks matched.

Figure 3 condenses the main systems evidence. Teacher inference and signal transfer are small compared with rollout and version-commit/acknowledgement latency. Teacher-signal accounting records 67.86 GB of trainermaterialized FP32 hidden-state tensors, while the corresponding analytical dense-logit accounting is 1.50 TB. The figure reports these trainer-side accounting quantities alongside the measured execution data.

The full run averaged 25.3 s per optimizer step (50,511 s over 2,000 steps). The instrumented step phases report a lower mean of 22.36 s because they exclude run-level overhead outside the timed step sections. Within the instrumented phases, student rollout and version commit/acknowledgement are the largest measured components of the train–serve loop.

Figure 4 adds per-device power, energy, and temperature telemetry. The two tensor-parallel inference GPUs and one of the DDP GPUs account for the largest energy totals, while GPU 0 shows lower average power and energy than the other three devices. Temperatures remain within the sampled operating range shown in the figure.

Figure 5 breaks the runtime down by specialist group. CLI diagnostics, dynamic routing, and edge access receive the largest trajectory counts, while edge access and dynamic routing account for the largest KD-token and teacher-signal volumes. Teacher batch-inference latency remains relatively tight across groups, despite these workload diferences. This view helps explain which specialist groups dominate trafic and transport volume during the run.

Figure 6 shows the optimization traces. Forward-KL loss drops early and stabilizes around the high-0.18 range, the learning-rate schedule follows warmup and smooth decay, and the mean student gradient norm remains in a narrow operating band through the final step.

![](images/6c8f33650be8deef5b2af992bba3b82dc0c3e1405818f89ed44881d2711c6448.jpg)

![](images/75d1577ae97a5e030ec890161ac73b98d08adecbe3a6593e52dc619cb1d2f4ee.jpg)

![](images/02fd8e48c89b4d04725adca19b5ba7d4b73ca7b92d3a566ac892c7d27c33929f.jpg)  
Figure 3: Systems summary. (a) Mean and peak device VRAM across the four GPUs. (b) P50/P95 rank-0 host-boundary latency across major phases, including version commit/acknowledgement. (c) Trainer-side hidden-state materialization and analytical dense FP32 logit accounting.

(a) GPU power  
![](images/25cf9b78021ffd7914bd110524c14aa5b1ae9683d9934c4313263ffaa258bae0.jpg)

(b) Estimated GPU energy  
![](images/2d1ce9d314124c457dd60b45c4d7acb547dfb7cd90ecea77cc875318a2f56461.jpg)

(c) GPU temperature  
![](images/c9aaf22c09d98fb3dbb2447eb912ded1747035c9c2230181a76252ef2cf54eaf.jpg)  
Figure 4: Per-device telemetry for the run. (a) GPU power statistics. (b) Estimated GPU energy by device from NVML sampling. (c) GPU temperature statistics.

(a) Trajectories  
![](images/258ae9756fc16ebb976583dfd51157ba464ea11eb3554bacec6baa027dd3b77a.jpg)

(b) KD tokens  
![](images/b03fb0449823ccc7443e64d0853821ffb8cc33f8a782ebe82d118e9f590c1ab0.jpg)

(c) Mean generated length  
![](images/0c4c3aeed743c764449f9ea127ca20467e3af96884323b8fb37f745cd1146d18.jpg)

(d) Teacher signal  
![](images/b75c407650366895ff76ddfeace7fdf9eebb208a4123b75ffa6e30e4be985ec4.jpg)

(e) Teacher batch inference  
![](images/bfdebddfa4b7ac3131c67dbd9af1eb7bfde31669083d20498dc3094eaac89dbc.jpg)

(f) KD tokens per trajectory  
![](images/2d76f19107cd40a1046cdec8103d53d3e7f165872d1f5387f0e01ae7d281e82b.jpg)  
Figure 5: Per-specialist-group workload and runtime statistics. Panels show trajectory counts, KD-token totals, mean generated length, mean teacher-signal size, teacher batch-inference latency, and mean KD tokens per trajectory.

RSOD student-training loss across 2,000 steps  
![](images/25028fd82ad6d059e9707ba4fe53dee18da7b6255adc4d222259533b6a32df68.jpg)

![](images/7a18ee14044a227888c486c57342b3e22912e91811dcf2dcd6917c7ee18a706a.jpg)

![](images/60d8b9ab8263fde7024f8d2115f6cdec98d3dfdb7613af709950cbde433d79eb.jpg)  
Figure 6: Optimization traces over 2,000 steps: forward-KL training loss with P10–P90 band (top), learning-rate schedule (bottom left), and mean student gradient norm (bottom right).

## 8.2 Model Quality Across Training Stages

The primary route-conditioned evaluation uses the same frozen 2,850-question manifest and the same item-level route-specific system prompts for the raw Meta-Llama-3.1- 8B-Instruct model, the six CPT+specialist teachers, and the consolidated RSOD student. Generation uses deterministic decoding (temperature 0, top-p 1.0, no sampling) with a maximum of 512 newly generated tokens. The raw model scores 12.8661/25 micro and 13.0172/25 macro. When each question is answered by its assigned specialist, the combined routed six-teacher system scores 21.7641/25 micro over 2,850/2,850 questions; the six specialist-group means span 19.0368–22.5912/25 with complete group coverage. The final RSOD student reaches 21.7112/25 micro and 21.8006/25 macro, an 8.8451-point micro-score gain over the raw model and a 0.0529-point micro diference from the combined routed-teacher system. Table 6 reports the detailed specialist-group results.

Table 6: Quality scores for the raw model, routed specialists, and Llama-CalixCLM v1.
<table><tr><td>Model specialist group</td><td>Micro/group mean</td><td>Macro</td><td>Scored/Total</td></tr><tr><td>Raw model</td><td>12.8661</td><td>13.0172</td><td>2845 2850</td></tr><tr><td>Combined routed teachers</td><td>21.7641</td><td></td><td>2850/2850</td></tr><tr><td>Dynamic Routing teacher</td><td>22.5912</td><td></td><td>570/570</td></tr><tr><td>Edge Access teacher</td><td>22.5316</td><td></td><td>570/570</td></tr><tr><td>CLI Diagnostics teacher</td><td>21.8544</td><td></td><td>570/570</td></tr><tr><td>Broadband Services teacher</td><td>21.5276</td><td></td><td>570/570</td></tr><tr><td>Routing Fundamentals teacher</td><td>20.9553</td><td></td><td>380/380</td></tr><tr><td>MPLS Services teacher</td><td>19.0368</td><td></td><td>190/190</td></tr><tr><td>Llama-CalixCLM v1 (route)</td><td>21.7112</td><td>21.8006</td><td>2850/2850</td></tr><tr><td>Llama-CalixCLM v1 (neutral)</td><td>20.0028</td><td>20.0778</td><td>2850/2850</td></tr></table>

A secondary neutral deployment-like prompting condition evaluates the same final student without the routespecific expert instruction. Under this condition the student reaches 20.0028/25 micro and 20.0778/25 macro. This secondary result reports deployment-like behavior without the route-specific specialization cue; no raw-model neutral baseline is used for a relative claim. Table 6 and Figure 7 summarize the primary progression and the secondary student condition.

![](images/d2d327b11738e80ea330543ba318a4f1bcb3d869b64e9ff1adf6301865480e84.jpg)  
Figure 7: LLM-judge scores for the raw model, combined routed teacher system, specialist groups, and consolidated RSOD student. Route-conditioned prompting is the primary comparison; the neutral student bar is a secondary deploymentlike condition.

## 9 Reproducibility

Machine-readable manifests link the raw base, CPT backbone, routed datasets, specialist adapters, resolved RSOD configuration, telemetry, checkpoints, judge rubrics, and evaluation reports. The executed CPT record fixes the fourteen inputs, tokenizer identity, 212,559 training records, 47,975,711 consumed tokens, and 3,322 optimizer steps; dataset manifests preserve per-route counts and SHA-256 identities for train and held-out artifacts. The resolved RSOD configuration fixes route ownership, optimizer/KD/rollout settings, GPU placement, and student LoRA geometry. Together with version checks, token hashes, synchronization events, and reconstruction validation, these artifacts provide the evidence layer for the model-version, token-identity, and teacher-signal invariants.

Released artifacts. Calix AI Labs publishes model artifacts through the Hugging Face organization at https: //huggingface.co/calix-ai-labs. The TelcoBench evaluation framework is available at https://github. com/calix-ai-labs/telcobench.

## 10 Limitations

Response quality is measured with an LLM-judge rubric, so the reported 0–25 values are rubric-based model-quality scores rather than executable correctness guarantees. The paper reports both the primary route-conditioned protocol and a secondary neutral deployment-like prompting condition for the final student.

## 11 Conclusion

We present an end-to-end telecom language-model work flow spanning source-grounded structured data generation, shared continued pretraining, specialist LoRA posttraining, routed specialist on-policy distillation, and networking evaluation. The executed data pipeline materializes 212,559 routed training records across fourteen routes and six specialists. RSOD consolidates the six frozen specialists into one deployable student adapter: the student generates the trajectory, the deterministic route scheduler selects one specialist, the teacher scores the exact native token sequence, and only the student LoRA is optimized. A controlled reconstruction test validates the compact teacher-signal path with $7 . 2 6 \times 1 0 ^ { - 6 }$ mean absolute logit error and 100% top-1 agreement between native and reconstructed teacher logits. Across 16,000 training trajectories, all recorded token-identity and serving-version checks match. Under the primary route-conditioned evaluation, the raw model scores 12.8661/25 micro, the combined routed six-teacher system scores 21.7641/25 micro over all 2,850 questions, and Llama-CalixCLM v1 reaches 21.7112/25 micro and 21.8006/25 macro; individual specialist-group means span 19.0368–22.5912/25. Under the separate neutral deployment-like prompt, the same student reaches 20.0028/25 micro and 20.0778/25 macro. Together, the data, numerical-fidelity, quality, audit, and systems measurements provide a practical path from domain-specialized networking knowledge to a single telecom-focused deployment adapter.

## References

[1] G. Hinton, O. Vinyals, and J. Dean. Distilling the Knowledge in a Neural Network. 2015.

[2] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

[3] R. Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR, 2024.

[4] F. Yuan et al. Reinforced Multi-Teacher Selection for Knowledge Distillation. 2020.

[5] L. Chen et al. Punica: Multi-Tenant LoRA Serving. MLSys, 2024.

[6] Y. Sheng et al. S-LoRA: Serving Thousands of Concurrent LoRA Adapters. MLSys, 2024.

[7] L. Zheng et al. SGLang: Eficient Execution of Structured Language Model Programs. NeurIPS, 2024.

[8] A. Maatouk et al. TeleQnA: A Benchmark Dataset to Assess Large Language Models Telecommunications Knowledge. 2023.

[9] H. Zou et al. TelecomGPT: A Framework to Build Telecom-Specific Large Language Models. 2024.

[10] V. Colle et al. TeleMath: A Benchmark for Large Language Models in Telecom Mathematical Problem Solving. 2025.

[11] K. Aykurt, A. Blenk, and W. Kellerer. NETLLMBENCH: A Benchmark Framework for Large Language Models in Network Configuration Tasks. IEEE NFV-SDN, 2024.

[12] Z. Yang et al. Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation. arXiv:2603.19220, 2026.

[13] W. Ma et al. MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training. arXiv:2606.30406, 2026.

[14] P. Yadav, D. Tam, L. Choshen, C. Rafel, and M. Bansal. TIES-Merging: Resolving Interference When Merging Models. NeurIPS, 2023.

[15] A. Maatouk, K. Chirino Ampudia, R. Ying, and L. Tassiulas. Tele-LLMs: A Series of Specialized Large Language Models for Telecommunications. 2024.

[16] GSMA. Open-Telco LLM Benchmarks. 2025.

## A Reproducibility Summary

<table><tr><td>Item</td><td>Recorded value / boundary</td></tr><tr><td>Backbone artifact</td><td>cpt-11ama31-8b-six-expert-final-oct-1; CPT: 212,559 routed records, 47,975,711 to- kens, 3,322 steps</td></tr><tr><td>Specialists / student</td><td>Six frozen LoRA teachers and one fresh student; rank 64, alpha 128, seven attention/MLP</td></tr><tr><td>Hardware</td><td>projection targets 4× NVIDIA RTX 6000 Ada; GPUs 0–1 DDP student, GPUs 2–3 TP inference</td></tr><tr><td>RSOD recipe</td><td>τ = 1, KL weight 1, generated-token-only mask; AdamW  $1 0 ^ { - 5 }$  cosine schedule, 3% warmup; 2,000 steps, 16,000 trajectories</td></tr><tr><td>Version commit Audit</td><td>candidate save → service refresh → acknowledgement → COMMITTED → next rollout 2,000/2,000 adapter commits completed; 16,000 version checks and 16,000 token-ID checks,</td></tr><tr><td>Compact-signal fidelity</td><td>zero observed mismatches 50 generated positions, 128,256-token vocabulary;  $\mathrm { M A E 7 . 2 6 \times 1 0 ^ { - 6 } }$  ; 100% top-1 agreement;</td></tr><tr><td>Evaluation generation</td><td>98% exact top-5 match Route-specific expertise prompt; native chat template; deterministic generation</td></tr><tr><td>Judge</td><td>claude-sonnet-4-5; temperature 0; batch size 5; five 0–5 rubric dimensions; identity- masked prompt; validated JSON; up to three total attempts</td></tr><tr><td>Evaluation manifest</td><td>2,850 questions; 190 per route except BNG/subscriber management with 380</td></tr></table>