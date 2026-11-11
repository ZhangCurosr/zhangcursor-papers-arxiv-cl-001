# EngineAgent: Agentizing On-Device LLM Inference Engines with Inference Stack Hierarchy Awareness and Hardware Abilities

Zhengxiang Huang Shanghai Jiao Tong University Shanghai, China

Zhaode Wang Alibaba Group Hangzhou, China

Chengfei Lv Alibaba Group Hangzhou, Zhejiang, China

Yujie Sun Shanghai Jiao Tong University Shanghai, China

Chaoyue Niu<sup>∗</sup> Shanghai Jiao Tong University Shanghai, China

Fan Wu Shanghai Jiao Tong University Shanghai, China

Shengheng Chen Shanghai Jiao Tong University Shanghai, China

Zeyu Zhao Shanghai Jiao Tong University Shanghai, China

Guihai Chen Shanghai Jiao Tong University Shanghai, China

## Abstract

Rapidly evolving LLM architectures and heterogeneous mobile hardware make supporting new models in on-device inference engines a recurring and costly system challenge. Each model may require coordinated changes across conversion, graph construction, operators, and hardware backends. Existing coding agents often miss these cross-layer dependencies and struggle to diagnose failures across host–device execution paths. We present EngineAgent, an agentic system for automating on-device LLM model enablement. Its key insight is to separate reusable engine and hardware knowledge from model-specific implementation. Before enablement, EngineAgent constructs a hierarchy-aware engine index that captures engine abstraction layers and cross-layer dependencies, and pre-validates backend-specific build, execution, and debugging workflows using already supported models. For each new model, a builder uses this knowledge to coordinate cross-layer changes, while an auditor checks implementation completeness and drives iterative debugging through device-side validation. Across 12 tasks spanning three mobile LLMs and four engine–backend pairs on CPUs, GPUs, and NPUs, EngineAgent achieves 75% end-to-end success with either of two underlying LLMs, versus 33.3% for the strongest coding-agent baseline. Success requires compilation, on-device execution, and output-logit agreement with reference implementations. EngineAgent also reduces average per-task time and model-API cost relative to baselines using the same underlying LLMs.

## 1 Introduction

On-device LLM inference engines are critical middleware positioned between two distinct ecosystems: rapidly evolving LLM model architectures above and a fragmented landscape of heterogeneous mobile hardware below. Engine developers must continuously extend their engines to support previously unsupported model architectures and capabilities while ensuring their correct execution across underlying diverse mobile CPUs, GPUs, and NPUs.

On the model side, device-oriented lightweight LLM families evolve rapidly, with successive generations often released only months or even weeks apart. For instance, Gemma 3 to Gemma 3n only took about 4 months [12, 35], while MiniCPM-V 4.0 to 4.5 took only 24 days [44]. More importantly, successive generations frequently introduce new modules and designs. Qwen3.5 [36] introduces GatedDeltaNet linear attention [43] interleaved with conventional attention. Gemma4 [13] adds hybrid sliding-window/global attention, proportional RoPE (p-RoPE), unified keys and values, shared KV cache, and Per-Layer Embeddings (PLE). MiniCPM-V 4.6 further introduces a window-attention vit\_merger, intra-ViT early compression, and mixed 4×/16× visual-token compression [26]. Consequently, supporting one generation rarely guarantees support for its successors, forcing inference engines to repeatedly incorporate newly introduced model structures and modules.

On the hardware side, mobile deployment is fragmented across heterogeneous execution backends. ARM CPUs, mobile GPUs [18], Qualcomm Hexagon NPUs [16, 17], MediaTek APUs [15], and Apple Neural Engine [14] difer substantially in programming languages and interfaces, tensor layouts, compilation workflows, and runtime semantics. Therefore, supporting the same new LLM feature requires substantially diferent implementation paths across hardware backends. CPU backends may implement new operators directly in C/C++ or assembly; Vulkan GPUs may require dedicated compute shaders, SPIR-V compilation, and backendspecific dispatch logic; NPUs typically require graph rewriting, accelerator-specific lowering, and vendor-specific compilation. Consequently, each newly introduced model feature need to be independently adapted and validated for multiple backends through entirely diferent engineering paths, multiplying the implementation eforts required for enablement.

Table 1. Model enablement<sup>1</sup> backend coverage and delay across on-device LLM inference engines [9, 10, 22, 24].
<table><tr><td>Model</td><td>Release</td><td>Engine</td><td>First Enablement</td><td>Backend</td></tr><tr><td rowspan="4">Qwen3.5-2B</td><td rowspan="4">2026-03-02</td><td>ExecuTorch</td><td>04-01 / 30 days</td><td>CPU (text-only)</td></tr><tr><td>MNN</td><td>03-02 / day</td><td>CPU / GPU / NPU</td></tr><tr><td>llama.cpp</td><td>03-02 / 0 day</td><td>CPU</td></tr><tr><td>MLC-LLM</td><td>03-08 / 6 days</td><td>GPU</td></tr><tr><td rowspan="4">Gemma4-E2B-it</td><td rowspan="4">2026-04-02</td><td>ExecuTorch</td><td>N.A.</td><td>unsupported</td></tr><tr><td>MNN</td><td>04-09 / 7 days</td><td>CPU</td></tr><tr><td>llama.cpp</td><td>04-02 / 0 day</td><td>CPU</td></tr><tr><td>MLC-LLM</td><td>N.A.</td><td>unsupported</td></tr><tr><td rowspan="4">MiniCPM-V 4.6</td><td rowspan="4">2026-05-11</td><td>ExecuTorch</td><td>N.A.</td><td>unsupported</td></tr><tr><td>MNN</td><td>N.A.</td><td>unsupported</td></tr><tr><td>llama.cpp</td><td>05-11 / 0 day</td><td>CPU</td></tr><tr><td>MLC-LLM</td><td>N.A.</td><td>unsupported</td></tr></table>

Traditionally, enabling a new model remains a largely handcrafted, expert-driven process, requiring expertise spanning both LLM model architecture and thorough understanding of the inference engine stack. Maintainers must analyze the new model semantics, map new model features onto engine modules, modify export and runtime execution paths, adapt operators or kernels, and validate execution on target backends. For example, enabling Qwen3.5-2B in MNN [22] touched 42 files and over 4.2K lines across export and ex ecution runtime, shape inference, and both CPU and GPU backends, whereas ExecuTorch [9] introduced changes to 20 files and over 1K lines for text-only CPU support. Such broad and heavy modifications impose substantial development and validation burdens on engine maintainers, which are also reflected in both incomplete backend and modality coverage and enablement delay. As summarized in Table 1, 2 out of the 4 on-device engines do not support Gemma4-E2B-it, while 3 out of the 4 do not support MiniCPM-V 4.6. Even when a model is supported, coverage is often restricted to only a subset of backends or modalities, such as CPU-only or text-only execution. Moreover, initial on-device enablement can take up to 30 days after model release.

The substantial engineering burden naturally motivates us to agentize model enablement. Before a new model is released, agents should acquire reusable knowledge of the engine’s semantics and establish backend-specific debugging and verification workflows using already supported models. After release, agents should analyze the new model’s architecture, use this knowledge to identify and implement the required engine modifications, and iteratively debug and verify the implementation on real devices.

Challenges ofAgentization. However, both localizing modifications within an engine and debugging and verifying them on real devices are fundamentally challenging.

① Challenges of mapping new model features across hierarchical engine abstractions. On-device LLM engines such as llama.cpp [10], MNN [22], ExecuTorch [9], and MLC-LLM [24], share a complex hierarchical organization that can be abstracted into multiple levels with distinct representations and manipulation logic. At the input level, naturallanguage or visual inputs are transformed through tokenization or patchification; at the graph level, tensor computation is constructed, converted, fused, and scheduled as computation graphs; and at the operator level, computations are realized through backend-specific kernels and runtime dispatch. During model enablement, a new feature may require coordinated modifications across these levels, yet their respective roles are not apparent from code structure alone. Without understanding where a component lies in the engine hierarchy and what role that level plays in model execution, an agent may identify and modify locally relevant code while overlooking modifications required at other levels. Therefore, reliable modification localization requires reasoning about the engine hierarchy rather than treating the codebase as a flat collection of files and symbols. ② Challenges ofcross-device debugging and verification. Unlike conventional host-only software, where execution and debugging typically occur within the same environment, on-device model enablement follows a host–device, cross-language, build–debug pipeline spanning host-side preparation and device-side deployment and execution. As a result, a failure observed on device may originate from an earlier host-side stage rather than from the failing runtime itself. Moreover, mobile CPU, GPU, and NPU backends follow distinct compilation and runtime workflows, each requiring diferent debugging methods. Consequently, tracing failures back to their root causes across devices and backends is inherently challenging.

Why Existing Coding Agents Fail. Existing repositorylevel coding agents struggle with both challenges. ① Code structure centric knowledge ratherthan engine hierarchy knowledge. Existing agents [4, 6, 38, 47] primarily reason over files, functions, symbols, and AST-level dependencies without explicitly modeling how these entities related to engine abstraction hierarchy. Consequently, their reasoning and modifications are typically constrained to local changes of one or two modules, missing necessary modifications across distant abstraction layers, e.g., adding an operator definition without implementing its hardware kernel. ② Host-centric verification and debugging rather than cross-language, host–device, backend-specific workflows. Existing agents largely rely on language-specific, host-local static analysis, unit tests, and debug tracing [48, 51]. They lack comparable capabilities for orchestrating and reasoning over host–device debug workflows, causing device deployments to fail, device-side failures to remain unattributed or misattributed, and debugging to fall back to repeated build–test attempts without precise localization. Even Codex [25] powered by GPT-5.6-Sol exhibits both missing implementations and hardware-verification failures (Figure 9); for example, when enabling MiniCPM-V 4.6 on MNN–CPU, it stopped after an incomplete build and skipped device validation (Table 2).

![](images/f505194b816621e903bc091b73183610db4c581e92c5d30008bfb5de0d4b666c.jpg)  
Figure 1. Position and workflow of EngineAgent: agentically mapping rapidly evolving LLM architectures onto heterogeneous mobile hardware for inference engines.

Our System. We propose EngineAgent, an agentic system for reliable on-device LLM engine model enablement that automates the workflow while addressing the two challenges above. EngineAgent consists of three key designs: a modelagnostic engine hierarchy knowledge index, hardware-aware debug skills, and a model-specific multi-agent implementation– audit main loop. To address the lack of engine-hierarchy understanding, EngineAgent constructs a hierarchy-aware knowledge index (§4.2) that maps code structures onto engine abstraction layers and cross-layer dependencies, enabling agents to identify complete modification scopes beyond local code relations. To address unreliable hardwaregrounded debugging, EngineAgent constructs reusable build, test, and debugging skills (§4.3) by validating backend-specific workflows with already supported models, providing stable verification procedures for heterogeneous hardware. These two designed artifacts are pre-constructed ahead of time because they are reusable and model-agnostic. Moreover, they are designed to be updatable upon new engine commits. When a new model arrives, it enters the model-specific multiagent enablement main loop (§4.4), where newly introduced model features are mapped onto the pre-constructed hierarchy knowledge index and hardware-aware debug skills. The builder then plans and implements layer-wise modifications guided by the hierarchy index, and submits each completed change to the red-team auditor. If validation fails, failures are routed to the builder with hardware-aware debugging support for iterative refinement. The auditor accepts the implementation only after all layer-wise checks and end-to-end device validation pass.

Together, EngineAgent supports reliable model enablement across heterogeneous inference engines and hardware backends. We evaluate EngineAgent on 12 model-enablement tasks spanning 3 mobile LLMs and 4 engine–backend pairs that cover CPU, GPU, and NPU execution: MNN–CPU, llama.cpp– Vulkan, MLC-LLM–OpenCL, and ExecuTorch–QNN. EngineAgent achieves an end-to-end success rate of 75%, compared with 33.3% for the strongest coding-agent baseline.

This work makes the following contributions:

• We identify model enablement as a distinct cross-layer software-engineering problem for on-device LLM inference engines, and characterize its two key challenges: hierarchical system understanding and hardwareaware debugging and verification across heterogeneous devices and backends.

• We design EngineAgent, an agentic system that addresses these challenges through three key components: a model-agnostic hierarchy-aware engine knowledge index that captures abstraction layers and crosslayer dependencies, hardware-aware debug skills that provide reusable and validated build–debug workflows, and a model-specific main loop that maps new model features onto these pre-constructed artifacts for layerwise implementation, completeness auditing, deviceside validation, and iterative refinement.

• We evaluate EngineAgent on 12 model-enablement tasks across three mobile LLMs and four engine–backend pairs, demonstrating higher efectiveness and eficiency than existing coding agents across benchmark cases.

![](images/7d9588159df034b6e8c29907ab0da070afa9fec3a9284fc6719c0135b7588f88.jpg)  
Figure 2. On-device LLM engines comprise export and execution stages, which share three abstraction layers.

## 2 Background

## 2.1 On-Device LLM Inference Engines

On-device LLM inference engines provide the unified software infrastructure for executing diverse LLMs on mobile devices equipped with heterogeneous processors such as CPUs, GPUs, and NPUs. Representative engines, including MNN [22, 39], ExecuTorch [9], llama.cpp [10], and MLC-LLM [24], share a common two-stage architecture: an ofline, on-host model export stage, realized by the engine’s converter and compiler, and an online, on-device execution stage, powered by the engine’s runtime. Orthogonally, both stages are organized into the same three levels of abstraction: the input level (chat templates and tokenizers), the graph level (the computation graph and its intermediate representation, IR), and the operator level (backend kernels and their hardware mappings).

As is illustrated in Figure 2, the ofline export stage traverses these three levels top-down. At the input level, it exports the chat template and the (multi-modal) tokenizer and packs them into the model artifact. At the graph level, it maps PyTorch layers and modules onto engine-native IR nodes, applying transformations such as quantization, weight pack ing, layout reordering, and operator fusion. At the operator level, it lowers and links each IR node to a hardware-specific kernel from the engine’s backend operator library. Notably, certain engine–hardware pairs defer preparation to runtime via online just-in-time (JIT) compilation, such as the OpenCL backend for mobile GPUs in MNN but the JIT workflow mirrors its ofline counterpart level for level. The online execution stage traverses the same levels at inference time. Engine runtime loads the prepared artifact, applies input-level preprocessing and tokenization to the user input, then walks the graph node by node, dispatching each node to its operator implementation.

In summary, although these engines difer in their IR designs and supported backends, they converge on the same two-stage, three-level workflow. This shared abstraction is what allows our analysis to generalize across engines rather than being tied to any single instance.

## 2.2 Model Enablement from Maintainers’ View

For engine maintainers, supporting a new LLM in a mobile inference engine is a complicated engineering efort, involving two primary procedure: understanding the engine and mapping new model features to modifications across engine modules and abstraction layers, and perform the host-device coordinated verify-debug loop to debug. Figure 3 exemplifies these 2 major procedure.

Maintainers first analyze newly introduced model features and determine how they map onto the engine’s abstraction levels and modules. As illustrated by the example of mapping Qwen3.5-2B [36] to MNN [22] in Figure 3(a), changes such as new chat-template formats or multimodal settings (e.g., "<|im\_start|>" and input visual patch size) are mapped to the input level; new model structures and tensor contracts such as the vision index tensor shape and data type are mapped to the graph level; newly introduced operations such as FusedLinearAttention [43] must be lowered to operator level backend specific implementations, e.g., CPUFusedLinearAttention and its ARM64 NEON implementations. After the cognitive step of these mappings from model features to abstraction layers, the maintainers implements modifications to files and modules in each layer.

Maintainers’ implementation loop is an iterative host– device verify-debug loop. As shown in Figure 3(b), they first modify on host across Python, C/C++, assembly, or backend specific code, then cross-compile it using the corresponding toolchains, such as the Android NDK, Vulkan toolchain, or QNN SDK. The resulting binaries and model artifacts are deployed to the target phone through Android Debug Bridge (ADB) for execution, after which logs, outputs, and other runtime artifacts are collected back on the host for verification. If any stage fails, maintainers trace the observed error back to the responsible implementation, revise the code, and repeat the coding–build–deploy–execute–verify cycle until all required checks pass.

## 2.3 Agent for Systems

To alleviate the substantial human efort in model enablement, agentizing the workflow is a natural direction. Existing works have developed complementary capabilities for software engineering: general-purpose end-to-end coding agents directly solve development tasks through iterative code modification and execution, while code-analysis tools and agents enhance end-to-end coding agents with repository representations, modification localization, and edit plans, and debugging agents augment them with failures diagnosis and generate corrective feedback. However, these capabilities do not directly resolve the challenges of inference-engine model enablement.

Limitations of General Purpose End-to-End Coding Agents. CodeAgent [47], SWE-Agent [42], and OpenHands [38] provide end-to-end workflows integrating repository navigation, code modification, execution, and iterative feedback. Recent systems eforts such as SYSSPEC [20] and ECO [19] further incorporate domain-specific abstractions and production constraints into LLM-driven system automation. However, these approaches still do not target inferenceengine model enablement as an end-to-end task: they neither expose engine hierarchies as first-class knowledge nor provide dedicated host–device verification and debugging procedures across heterogeneous hardware backends.

(b) Host-device build-verify-debug loop.  
![](images/7bbc8aa7a43a67afcda0db1d893defae91252a415bb298408b2d0d78abb18870.jpg)

(a) Example: mapping Qwen3.5-2B [36] features to MNN [22] abstraction layers.  
![](images/82c9725eb6fd998a507ce35290e30017224d5cea38ff7049976a9a36bc6531d4.jpg)  
Figure 3. Model enablement workloads: mapping new model features to engine abstraction layers and then perform hostdevice build-verify-debug loop.

Limitations of Prior Code Analysis Tools and Agents. CodeGraph [23] and LocAgent [6] improve repository representation and localization, CodePlan [4] plans coordinated repository-wide edits, and AutoCodeRover [48] combines program-structure-aware search with fault localization. However, these methods primarily organize knowledge around code-level structures such as files, symbols, functions, and AST or dependency relations. They do not explicitly model how these entities map onto inference-engine abstraction layers, making cross-layer dependencies, wrong-layer localization, and structurally distant but semantically coupled modifications dificult to reason about.

Limitations of Prior Debug Agents. AgenTracer [46] attributes failures to responsible agents and trajectory steps, while AgentDebug [51] identifies root causes and provides corrective feedback for recovery. Such approaches improve failure analysis, but remain largely centered on agent trajectories or host-side program feedback. Model enablement instead requires cross-language, cross-compilation debugging across host preparation, device deployment, and heterogeneous XPU execution, where failures must be traced from device-side observations back to the responsible host-side implementation.

Table 2. Representative failure cases of existing coding agents during on-device LLM model enablement.
<table><tr><td>Engine Backend</td><td>Agent</td><td>Failed Stage</td><td>Failure Category</td><td>Symptom</td></tr><tr><td rowspan="3">MNN CPU</td><td>CodePlan</td><td>Planning</td><td>Missing implementation</td><td>Localized config changes but missed dependent vision and runtime implementations.</td></tr><tr><td></td><td>SWE-Agent Implementation</td><td>Missing implementation</td><td>Modified model components but missed export-runtime</td></tr><tr><td>Codex</td><td>Verification</td><td>Premature acceptance</td><td>propagation. Stopped after incomplete build and skipped device validation.</td></tr><tr><td rowspan="2">llama.cpp Vulkan</td><td>OpenHands</td><td>Backend integration</td><td>Backend failure</td><td>Failed to resolve Vulkan compilation and deployment.</td></tr><tr><td>OpenCode</td><td>Backend integration</td><td>Backend failure</td><td>Host adaptation completed but Vulkan execution path remained incomplete.</td></tr><tr><td rowspan="2">ExecuTorch QNN</td><td>CodePlan</td><td>Planning</td><td>Missing implementation</td><td>Followed local dependencies but missed QNN export.</td></tr><tr><td>OpenHands</td><td>Backend integration</td><td>Backend failure</td><td>Failed at QNN lowering and runtime constraints.</td></tr></table>

## 3 Case Studies and Key Insights

## 3.1 Characterizing Agent Failures

Motivating Case Studies. To understand the limitations of existing coding agents for on-device LLM model enablement, we conduct motivating case studies on three representative engine–backend pairs: MNN–CPU, llama.cpp–Vulkan, and ExecuTorch–QNN, with MiniCPM-V 4.6 as the target model. We evaluate representative repository-level coding agents, including SWE-Agent [42], OpenHands [38], Code-Plan [4], OpenCode [27], Codex, and CodeGraph [23] + Open-Code + AgenTracer [46]. The evaluated agents use their corresponding backbone models: DeepSeek-V4-Flash for opensource agent frameworks and GPT-5.6-Sol for Codex [25].

Across these cases, existing agents demonstrate promising capability in repository exploration and local code modification, but rarely complete the entire model enablement workflow from model adaptation to validated on-device execution. Across 18 test cases (6 agents × 3 representative engine– backend pairs), only 1 attempt successfully completes endto-end model enablement. The remaining 17 attempts fail at diferent stages, including cross-layer implementation, backend integration, and device-grounded verification. Notably, the only successful case, OpenCode on MNN–CPU, supports the CPU backend, which has a simpler lowering and compilation stack than GPU/NPU. These observations motivate a systematic categorization of agent failures.

Agent Failures Categorization. We summarize representative failure cases of existing coding agents in Table 2. Although the failures occur at diferent stages and across diferent engine–backend pairs, they exhibit two recurring patterns. First, agents frequently identify and modify locally relevant components but fail to recover the complete implementation scope required by a new model feature. For example, CodePlan on MNN–CPU modified model-related configuration paths but missed the dependent vision and runtime implementations, while SWE-Agent modified model components without propagating changes across export and execution paths. Second, agents often achieve intermediate host-side progress but fail to complete hardwaregrounded validation. For example, OpenCode and Open-Hands on llama.cpp–Vulkan reached host-side adaptation but failed to establish the complete Vulkan execution workflow, while Codex prematurely stopped after an incomplete build state without reaching device validation. These failure patterns indicate that reliable model enablement requires both engine-level understanding across abstraction layers and hardware-grounded verification capabilities.

## 3.2 Key Insights

The observed failures reveal two fundamental requirements for reliable agentic model enablement: global understanding of inference-engine abstractions and stable hardwaregrounded verification workflows.

Insight 1: Model Enablement Agent Requires Engine Hierarchy Aware Understanding. Missing implementations and premature exits often stem from agents lacking a global view of the inference engine organization. Without understanding the responsibilities and dependencies of different abstraction layers, agents may treat locally discovered code paths as the complete implementation scope and overlook required modifications elsewhere. Therefore, agents require explicit hierarchy-aware engine knowledge to map model features to afected layers, recover complete modification scopes, and guide cross-layer implementation.

Insight 2: Model Enablement Agent Requires Hardware Aware Debug Ability. Hardware-aware debugging is challenging not only because device execution is heterogeneous, but also because the verification path itself may be unstable when enabling previously unsupported models. Agents that directly debug unsupported models must simultaneously resolve model adaptation issues and uncertain build, deployment, and validation workflows, making failures dificult to attribute. Therefore, reliable enablement requires pre-constructed and validated build, test, and debugging workflows from already supported models, which provide stable hardware-grounded feedback for subsequent model adaptation.

![](images/4aea9024e05c674c00a663be2ac2525d88353746ec3bb454425ba0bc4fccc962.jpg)  
Figure 4. Two-stage workflow of EngineAgent, from enginespecific knowledge construction to model-dependent enablement and verification.

## 4 System Design

## 4.1 System Overview

Guided by the two key insights, EngineAgent augments general-purpose coding agents with two capabilities: enginehierarchy-aware understanding for complete cross-layer implementation and hardware-aware debugging and verification for reliable on-device validation. These capabilities support three complementary agent roles: a builder that implements model support, a debugger that diagnoses failures, and an independent auditor that challenges implementation completeness and correctness.

Following Insight 1, EngineAgent equips agents with a model-agnostic engine hierarchy knowledge index (§4.2). The index connects engine abstractions to concrete code entities and exposes dependencies across export, runtime, and backend layers. It guides the builder in mapping new model features to the required modification scope and provides the auditor with structural context for identifying omitted implementations. Following Insight 2, EngineAgent equips agents with reusable hardware-aware verify-debug skills (§4.3). Before a new model arrives, construction agents establish and validate backend-specific build, deployment, testing, and debugging workflows using already supported models. These skills provide the builder with self-validation procedures, the debugger with diagnostic guidance, and the auditor with executable checks, reducing uncertainty in the verification path during new-model adaptation.

As illustrated in Figure 4, both capabilities are prepared before model release and reused in the model-specific multiagent main loop (§4.4). When a new model arrives, the builder compares its architecture with supported models, maps the diferences onto the hierarchy index, and implements layer-wise modifications. The auditor checks each completed layer, while the debugger uses the validated host– device workflows to diagnose failures and guide revisions. The loop continues until the implementation passes both layer-wise checks and end-to-end device validation.

![](images/4d5b70f367167cb6fbe78d1945e5d2c45f01aa8f499f692c38a695faadb9cbaa.jpg)  
Figure 5. CodeGraph [23] only captures syntactic structures rather than hierarchical understandin gs. Extracted relations in MNN CodeGraph as an example are illustrated on the left.

## 4.2 Hierarchy-Aware Engine Understanding

For a fixed engine snapshot, prior work CodeGraph [23] provides a language-agnostic canonical graph of files, symbols, and typed static relations, including imports, calls, references, and instantiations, as illustrated in Figure 5. However, it captures only how code entities are structurally connected, without revealing the abstract knowledge of engine hierarchical design.

To encounter this, EngineAgent defines a nine-layer hierarchy based on the design abstraction of on-device LLM engines and automatically constructs a hierarchy-aware index by semantically lifting the CodeGraph representation of the engine into our pre-defined hierarchy. In this way, engine abstractions are explicitly represented as structured semantic subgraph views over the original CodeGraph, extending syntactic relations to system-level understanding.

Hierarchy Abstraction. As discussed in Section 2.1, ondevice LLM engines follow a two-stage workflow, where each stage contains three-level: input level, graph-level, and operator-level. In order to better help agents understand the engine hierarchy, we further divide the three levels into nine layers, including ① chat template and tokenization and ② multimodal preprocessing layers on input level; ③ model configuration, ④ I/O tensor contracts, ⑤ module mapping, ⑥ weight sharing and KV sharing layers on graph level; and ⑦ attention and KV cache, ⑧ operator wrapping, and ⑨ operator kernel layers on operator level. Specifically, the ofline-export has eight of these layers (L1–L8), spanning the export or compilation of each layer. The execution-stage, on the other hand, contains all nine abstraction layers (L1–L9), extending the export-stage hierarchy with an additional L9 layer for hardware-specific kernels, covering the parsing and execution of exported artifacts on each layer. For each abstraction layer, engine may have one or more instantiations: model-related layers (L3/L4) distinguish text-backbone and multimodal instantiations, while backend-dependent layers (ofline L7–L8 and runtime L7–L9) are instantiated separately for CPU, Vulkan, OpenCL, and QNN when applicable. We define these intra-layer substructures as layer instantiations. By defining this layered hierarchy in prompt, EngineAgent agentically construct a knowledge index from the raw Code-Graph to incorporate these knowledge into graph structures.

Agentic Index Construction and Review. EngineAgent constructs the hierarchy index as a nested graph structure following the predefined abstraction hierarchy. The construction agent first divides the raw CodeGraph into export and execution stage subgraphs and then sequentially builds abstraction-layer subgraphs within each stage. For each layer, the agent builds semantic understanding from the predefined hierarchy knowledge and repository evidence, selects representative code entities as seeds, and initializes the subgraph. The subgraph is then expanded from these seeds by traversing CodeGraph relations and incorporating entities supported by additional evidence. To improve reliability, EngineAgent separates semantic hierarchy attribution from raw graph construction, where static relation errors are checked independently while layer membership is determined through semantic evidence. After construction, the indexing agent performs self-review and a read-only auditor validates the completeness and grounding of the generated subgraphs. The resulting index provides multiple semantic views of the code entity graph across stages, abstraction layers, and instantiations. By mapping engine functional abstractions onto concrete code entities, it bridges the gap between repository structure and engine organization, enabling agents to understand how high-level functionalities are realized by specific modules.

Example and Analysis (MNN Hierarchy Indexing). Figure 6 visualizes the hierarchy-aware graph index constructed for MNN. The middle panel shows the original CodeGraph, where files and symbols are connected only through static code relations. After semantic lifting, EngineAgent reorganizes these entities into export- and execution-stage subgraphs and further into abstraction-layer subgraphs. The resulting layers exhibit clear semantic coherence by grouping code entities according to their engine responsibilities.

For the export stage, L1 groups tokenization-related entities such as Tokenizer export, Normalizer, and Vocabulary; L2 captures multimodal preprocessing components such as Resampler; L3 organizes model initialization logic including Model config and Weight loading; L5 groups module-level transformations such as LayerNorm and Linear; L7 captures attention-related components including Attention, KV cache, and Rotary position; and L8 connects high-level operators with their lowered implementations, such as mapping Attention to Fuse attention. The execution-stage hierarchy further exposes corresponding runtime responsibilities: L1 contains runtime token processing entities such as BPE encoder and special token mapping; L2 includes vision feature processing and multimodal fusion; L5 groups runtime execution logic including module forward, graph traversal, and memory management; L7 captures fused attention kernels and native flexible KV cache implementations; L8 contains concrete operators such as Convolution and QNN MatMul; and L9 reaches hardware-specific implementations such as ARM INT4 pack, I8MM, and SME2 GEMM. These examples demonstrate that the generated hierarchy is not a syntactic partition of the repository, but a semantic mapping between engine functionalities and concrete code entities. It enables agents to reason about how high-level model functionalities are implemented across diferent abstraction layers.

![](images/9d6b296658d34a9b4a5179f14cb4f196d4acaa66249dde40669db918008ffc90.jpg)

Figure 6. Hierarchy-aware engine modeling for MNN. The original CodeGraph (center) is organized into abstraction layer subgraphs for export (top, L1–L8) and execution (bottom, L1–L9), with layer definitions shown on the left. Colored subgraphs group code entities by engine responsibility, while gray edges indicate cross-layer dependencies.  
![](images/ac662de2e9e4b0ab9b462f0273d338979e413b3058f0fe96387fca68645b0c7e.jpg)  
Figure 7. Instantiation subgraphs within MNN’s executionstage hierarchy, exposing modality- and backend-specific code structures within each layer.

Abstraction-layer subgraphs with multiple concrete realizations are further refined into subordinate instantiation subgraphs. As shown in Figure 7, MNN’s L3 separates backbone and multimodal implementations, while L8 branches into CPU, Vulkan, OpenCL, and QNN implementations with distinct backend-specific code paths. These instantiation subgraphs expose how the same engine abstraction is realized diferently across modalities and hardware backends.

![](images/22274f8fd40fd1b287f410cd7e85bf7d0ab60046dcf0402655708bb7eb1b9d5a.jpg)  
Figure 8. Examples of generated unit tests and debug skills for MNN-CPU execution stage by Deepseek-V4-Flash.

Update Index under Engine Modification. For each new commit, EngineAgent maps added, removed, and modified code entities onto the existing hierarchy-aware graph index and identifies the afected stage, abstraction-layer, and instantiation subgraphs. Only these subgraphs and their immediate dependency neighborhoods are re-examined to update entity attribution, execution paths, and cross-layer relations, while unafected subgraphs are directly reused. Only when a commit introduces major structural changes, or repeated incremental updates cause excessive accumulated drift, EngineAgent triggers a hierarchy-aware index reconstruction. Thus, incremental updates of the pre-constructed model-agnostic index incur only modest overhead.

## 4.3 Host–Device Debugging and Verification Skills

Model enablement requires not only modifying the correct engine layers, but also repeatedly building, deploying, executing, and diagnosing the resulting implementation on real devices. To support reliable hardware-aware verify-debug loop, EngineAgent agentically constructs dedicated verifydebug skills around these host–device workflows. These procedures are largely engine and backend specific, yet remain reusable across subsequent new model enablements.

Automatic Skill Construction. To provide agents with procedural knowledge for backend-specific building, verification, and failure diagnosis, EngineAgent agentically constructs build skills, unit tests, and few-shot debug skills before individual model-enablement tasks.

Build skills organize information scattered across documentation, build scripts, and repository instructions into two executable workflows: model conversion/export and execution runner build and deployment. Rather than merely summarizing documentation, the construction agent explores the relevant scripts and commands and validates the resulting workflow by exporting and executing an already supported model on the real device.

Besides, unit tests and few-shot debug skills are further constructed following the layer and instantiation division defined in hierarchy-aware graph index. For the export stage, it traverses L1–L8 in order and builds tests around the observable input/output contracts of each abstraction layer, from preprocessing and model configuration to module mapping and operator lowering. For the execution stage, it similarly traverses L1–L9, extending validation through runtime operators and hardware-specific kernels. When a layer contains multiple instantiations, tests are constructed separately for the corresponding modality or backend branches. Figure 8 illustrate the example of generated skills for MNN-CPU execution stage. These generated skills check and verify the outputs and functionality of each layer’s modules. Together, these tests form a progressive validation chain, allowing failures to be localized to the earliest layer whose contract is violated rather than only being observed at end-to-end execution. During this construction process, successful diagnostic procedures, characteristic failure symptoms, and corresponding recovery actions are summarized into fewshot debug skills associated with their abstraction layers and hardware backends.

UpdateSkills underEngineModification. Prebuilt hardware aware verify-debug skills are updated using the same incremental principle as engine knowledge index updates. Changes to build scripts, documentation, backend imple mentations, or toolchain configurations invalidate only their associated build skills, unit tests, and few-shot debug skills. These afected artifacts are re-examined and revalidated, including replaying changed build or deployment workflows with an already supported model when necessary, while retaining unafected artifacts. A complete reconstruction is performed only when the corresponding backend workflow or toolchain undergoes substantial changes.

## 4.4 Model-Specific Multi-Agent Main Loop

When a new model arrives, EngineAgent uses the prebuilt hierarchy index and debug skills to modify the target engine for a specified XPU backend. This process is illustrated in Figure 4. EngineAgent instantiates three model-specific agents from the pre-constructed artifacts: a builder agent for implementation, a debugger agent for debugging reinforcement, and an independent auditor agent for red-team verification.

The builderagent first performs a diference analysis against the engine’s supported model set constructed during preenablement to identify the most similar reference models. The architectural diferences between the reference and new models are mapped onto the hierarchy-aware graph index to derive layer-wise modification plans. The builder then implements these tasks sequentially across abstraction layers. For each layer, the builder retrieves relevant hierarchy knowledge and hardware-aware build skills, unit tests, andfew-shot debug skills to guide implementation and self-validation.

The debugger agent reinforces the builder when failures occur or progress becomes blocked. Given the current abstraction layer, backend, execution logs, and validation results, it retrieves corresponding debugging skills, reproduces relevant workflows when necessary, and provides targeted diagnostic evidence and recovery suggestions. By leveraging pre-constructed debug skills, the debugger avoids repeatedly exploring unstable verification paths during enablement.

The auditor agent independently evaluates the builder’s implementation as a red-team verifier. After each abstraction layer is completed, it executes the corresponding preconstructed unit tests and checks whether the implementation satisfies layer-specific requirements. If validation fails, the auditor returns the earliest failing layer and device-side evidence for subsequent debugging. After all layers pass, the auditor performs end-to-end validation covering model export, deployment, and on-device execution before approving the implementation.

Together, these agents form an iterative implementation– verification loop. The builder proposes modifications, the auditor challenges their correctness, and the debugger provides targeted recovery when failures occur. The loop continues until all layer-wise checks and end-to-end device validation pass. Finally, the auditor-approved implementation is submitted to a rule-based non-agentic system verifier, which compares output logits against host-side PyTorch references and rejects implementations whose deviations exceed a predefined numerical tolerance.

## 5 Evaluation

## 5.1 Benchmark Design

To evaluate the efectiveness ofagentic systems for on-device LLM model enablement, we construct a benchmark comprising 12 tasks across four representative engine–backend pairs and three recent mobile LLMs: Qwen3.5-2B [36], Gemma4- E2B-it [13], and MiniCPM-V 4.6 [26]. Each task requires enabling one target model on one engine–backend pair, covering diverse model architectures and heterogeneous CPU, GPU, and NPU execution paths.

Engine–Backend Pairs. The benchmark covers four representative mobile engine–backend pairs:

• MNN + CPU,

• llama.cpp + Vulkan,

• MLC-LLM + OpenCL,

• ExecuTorch + QNN.

These pairs span substantially diferent execution stacks and hardware abstractions. MNN + CPU [22] represents a native CPU execution path with direct operator and kernel implementations; llama.cpp + Vulkan [10] represents explicit GPU execution through Vulkan kernels and backend dispatch; MLC-LLM + OpenCL [24] captures a compiler-based deployment workflow targeting mobile GPUs; and ExecuTorch + QNN [9] represents graph lowering and execution through a vendor-specific NPU backend. Together, they cover CPU, GPU, and NPU execution as well as native, compiler-based, and accelerator-specific enablement workflows.

Benchmark Case Construction and Engine Snapshot Selections. For each model–engine–backend pair, if the target model has already been supported, we reset the engine to the commit immediately preceding the support commit and ask the agent to reproduce the enablement from this pre-support state. Under this setting, the agent is not allowed to access future commits or search the web for post-snapshot implementations. If the target model is not yet supported by the corresponding engine–backend pair, we use the latest available engine snapshot and construct a forward-enable task. Such tasks reflect realistic maintenance scenarios where no ground-truth support patch exists and success can only be determined through end-to-end validation. Each benchmark task provides the agent with an engine repository snapshot, a target model checkpoint, the backend configuration, and validation scripts. This setup requires the agent to solve the actual model-enablement problem rather than recover an existing implementation.

Success Evaluation and Criteria. A task is considered successful only if the upgraded engine passes host compilation, device execution, and output-logit alignment checks verified by our evaluation program. The evaluation program reproduces the provided build scripts to verify model export and runner cross-compilation, then deploys the generated artifacts to the target phone through ADB and checks that execution completes without errors. It subsequently runs multiple on-device test cases covering all required modalities, collects the output logits, and compares them against host-side PyTorch reference results. The implementation is accepted only when the logits agree with the reference outputs, with only negligible numerical drift attributed to hardware diferences.

## 5.2 Experimental Setup

Baselines. We compare against four representative codingagent baselines. SWE-Agent [42] represents repository-level autonomous software engineering, while OpenCode [27] and Codex [25] represent general-purpose coding agents. We further construct a composite baseline, OpenCode + CodeGraph + AgenTracer, augmenting OpenCode with graph-based repository understanding [23] and trajectorylevel failure diagnosis [46]. Since coding agents frequently terminate prematurely on long-running enablement tasks without human continuation prompts, we enable the goal mode of Codex and attach a lightweight continuous-run trigger to the other baselines, which resumes the agents when the evaluator has not yet accepted the implementation as successful and the time limit has not been reached.

Agent Model API. SWE-Agent, OpenCode, and Open-Code + CodeGraph + AgenTracer use the DeepSeek-V4-Flash API [8], while Codex uses GPT-5.6 Sol [34]. We evaluate EngineAgent with both DeepSeek-V4-Flash and GPT-5.6 Sol. This provides matched comparisons against both groups of baselines and separates the efectiveness of the EngineAgent harness from LLM API it calls.

Metrics. We primarily report end-to-end success rate, together with five stage-wise success rates: export, host textonly, host multimodal, device text-only, and device multimodal, to localize failures along the enablement pipeline. A stage is counted as successful only when all associated evaluator checks described above pass. We also report average wall-clock time and average monetary cost per task. Each run is capped at five hours, with timeouts counted as five hours; monetary cost is measured by the total model-API usage incurred during the run.

## 5.3 Overall Performance across Benchmark Cases

Table 3 summarizes performance across 12 model-enablement tasks. EngineAgent achieves 9/12 end-to-end successes (75%) with either DeepSeek-V4-Flash or GPT-5.6 Sol, substantially outperforming all baselines. The strongest baseline, Codex with GPT-5.6 Sol, succeeds on only 4/12 tasks (33.3%), while OpenCode and CodeGraph + OpenCode + AgentTracer each succeed on 1/12 and SWE-Agent on none.

Table 3. Overall model enablement performance across the 12 benchmark cases.
<table><tr><td rowspan="2">Agent Model API</td><td rowspan="2">Agentic Systems</td><td rowspan="2">E2E Success Rate</td><td colspan="2">Average</td></tr><tr><td>Time</td><td>Cost ($)</td></tr><tr><td rowspan="4">DeepSeek V4 Flash</td><td>SWE-Agent</td><td>0/12</td><td>4.85h</td><td>2.11</td></tr><tr><td>OpenCode</td><td>1/12</td><td>4.78h</td><td>2.28</td></tr><tr><td>CodeGraph + OpenCode + AgenTracer</td><td>1/12</td><td>4.91h</td><td>2.44</td></tr><tr><td>EngineAgent</td><td>9/12</td><td>3.82h</td><td>1.53</td></tr><tr><td rowspan="2">GPT-5.6 Sol</td><td>Codex</td><td>4/12</td><td>4.82h</td><td>14.90</td></tr><tr><td>EngineAgent</td><td>9/12</td><td>4.16h</td><td>11.29</td></tr></table>

EngineAgent achieves higher success rate while using less time and lower model-API cost than the baselines. With DeepSeek-V4-Flash, EngineAgent completes each task in 3.82 hours on average at a cost of \$1.53, compared with 4.78–4.91 hours and \$2.11–\$2.44 for the corresponding baselines. With GPT-5.6 Sol, EngineAgent averages 4.16 hours and \$11.29 per task, compared with 4.82 hours and \$14.90 for Codex.

Table 4. Stage-wise model enablement success across the 12 benchmark cases. (T: Text-only backbone success rate on host/device, MM: Multi-Modal success rate on host/device)
<table><tr><td rowspan="2">Agent Model API</td><td rowspan="2">Agentic Systems</td><td colspan="4">Stage-wise Success Rate</td></tr><tr><td>Export</td><td colspan="2">Host</td><td colspan="2">Device</td></tr><tr><td rowspan="4">DeepSeek V4 Flash</td><td>SWE-Agent</td><td></td><td>T</td><td>MM</td><td>T</td><td>MM</td></tr><tr><td>OpenCode</td><td>4/12 4/12</td><td>3/12 4/12</td><td>0/12 2/12</td><td>1/12 1/12</td><td>0/12 1/12</td></tr><tr><td>CodeGraph + OpenCode</td><td>4/12</td><td>4/12</td><td>4/12</td><td>1/12</td><td>1/12</td></tr><tr><td>+ AgenTracer EngineAgent</td><td>10/12</td><td>10/12</td><td>10/12</td><td>9/12</td><td>9/12</td></tr><tr><td rowspan="2">GPT-5.6 Sol</td><td>Codex</td><td>6/12</td><td>6/12</td><td>4/12</td><td>6/12</td><td>4/12</td></tr><tr><td>EngineAgent</td><td>10/12</td><td>10/12</td><td>10/12</td><td>9/12</td><td>9/12</td></tr></table>

Table 4 shows where the performance gap emerges along the enablement pipeline. With DeepSeek-V4-Flash, EngineAgent achieves 10/12 successes in export and host-side text-only and multimodal execution, and 9/12 on both device-side paths. In contrast, DeepSeek-based baselines reach at most 4/12 at the host stage and fall to 0–1/12 on device. Adding CodeGraph and AgentTracer to OpenCode improves hostside multimodal success from 2/12 to 4/12, but device-side success remains 1/12, suggesting that stronger repository understanding alone is insuficient for hardware-grounded execution. With GPT-5.6 Sol, Codex reaches 6/12 on export and host text-only execution and 4/12 on multimodal execution, while EngineAgent still achieves 10/12 on host and 9/12 on device. Overall, EngineAgent’s advantage extends beyond finding the correct implementation to maintaining correctness through multimodal integration and host-to-device execution.

![](images/f1743d7812aa3e69e70b42ed3622599be6b553a9b89ba5e4e1f3434a926838c0.jpg)  
Figure 9. Failure breakdown across the 12 benchmark cases.

Figure 9 breaks down failures across the 12 tasks. SWE-Agent and OpenCode each miss three required implementations and incur five graph-level failures, while Codex still fails in eight cases across all four categories. In contrast, EngineAgent eliminates both missing-implementation and hardware deployment/verification failures, and reduces hardwarekernel failures to 1/12. This suggests that hierarchy-aware

knowledge helps recover the required implementation scope, while a stable build–verify–debug workflow prevents deployment failures and provides reliable feedback for hardwarespecific debugging. As a result, EngineAgent’s remaining three failures are limited to two graph-level and one hardwarekernel implementation failure.

## 5.4 Breakdown Analysis By Cases

CaseStudy: EnablingQwen3.5-2B on llama.cpp-Vulkan. The key challenge in enabling Qwen3.5-2B on llama.cpp-Vulkan is supporting the newly introduced Linear Attention operator on the Vulkan backend. As illustrated in Figure 10, SWE-Agent fails at the graph-export stage because it overlooks an unsupported implementation required by the new operator. OpenCode, CodeGraph + OpenCode + AgentTracer, and Codex progress further but eventually fail when implementing the execution-stage Linear Attention kernel. In particular, these agents struggle to localize kernel-level implementation errors merely with their continuous end-to-end verification. In contrast, EngineAgent exploits its hardwareaware build-test-debug skills and directly constructs unit tests for the Linear Attention kernel and iteratively debugs the implementation on device. This hardware-grounded verification loop enables EngineAgent to progressively localize and fix the implementation errors, ultimately completing the enablement in approximately 4.5 hours.

![](images/8ff69256b4674544384319c16c766b65c4e047717da4675bc8ad9ccf5fa8f9c3.jpg)  
Figure 10. Progress Comparison of Agents enabling Qwen3.5-2B in llama.cpp-Vulkan.

Case Study: Enabling MiniCPM-V4.6 on ExecuTorch-QNN. The key challenge in enabling MiniCPM-V 4.6 on ExecuTorch-QNN is to jointly modify the host-side ofline QNN graph compilation and the online execution runtime. As illustrated in Figure 11, all baselines fail during export: they repeatedly modify and debug the graph and operator export paths, but eventually time out because each iteration incurs the high latency of ofline QNN preparation. In contrast, EngineAgent leverages its hierarchy-aware index to directly localize the required changes for the newly introduced window-attention vit\_merger in the Python exporter, successfully exporting the model within two attempts and completing the enablement at last.

![](images/83a072145c9a4c7f68c93b6168331d7a7d47bebaf9a8c3dd6d34f26d77eb137c.jpg)  
Figure 11. Progress Comparison of Agents enabling MiniCPM-V 4.6 in ExecuTorch-QNN.

## 5.5 Cost Analysis for Pre-Constructed Index and Skills

The engine hierarchy-aware index and hardware-aware skills are pre-constructed once and reused across model-enablement tasks. As engines evolve, they are incrementally updated only the parts that are afected by each commit. We evaluate both their initial construction and incremental update costs below.

Table 5. Initial construction cost of pre-enablement artifacts on four engine snapshots.
<table><tr><td>Engine</td><td>Snapshot</td><td>Index Cost ($)</td><td>Skills Cost ($)</td></tr><tr><td>MNN</td><td>f41d4638</td><td>1.04</td><td>1.83</td></tr><tr><td>llama.cpp</td><td>e06088da</td><td>0.66</td><td>2.27</td></tr><tr><td>MLC-LLM</td><td>84050697</td><td>0.77</td><td>2.40</td></tr><tr><td>ExecuTorch</td><td>b7781552</td><td>1.25</td><td>2.68</td></tr></table>

Initial Construction Costs. Table 5 reports the initial construction costs for the four evaluated engines. Constructing the hierarchy index costs \$0.66–\$1.25, while hardwareaware skills cost \$1.83–\$2.68, averaging \$0.93 and \$2.30, respectively. Thus, the reusable pre-enablement artifacts require only modest one-time construction cost.

Table 6. Update cost ofpre-constructed artifacts across MNN versions.
<table><tr><td>Commit Update</td><td>Index Update Cost ($)</td><td>Skills Update Cost ($)</td></tr><tr><td>f41d4638 → c5903082</td><td>0.0041</td><td>0.0000</td></tr><tr><td>c5903082 → 0bff03cb</td><td>0.0195</td><td>0.1262</td></tr><tr><td>0bff03cb → 17c00ba6</td><td>0.0051</td><td>0.0000</td></tr><tr><td>17c00ba6 → 4419e33b</td><td>0.0047</td><td>0.0507</td></tr><tr><td>4419e33b → 30dfb9bb</td><td>0.0049</td><td>0.1459</td></tr><tr><td>Average</td><td>0.0077</td><td>0.0646</td></tr></table>

Update Costs. Table 6 reports the incremental cost across five consecutive MNN updates. On average, updating the hierarchy index costs only \$0.0077 and updating the hardwareaware skills costs \$0.0646 per commit, corresponding to only 0.7% and 3.5% of their initial construction costs, respectively.

Combined, each update costs only \$0.072 on average, about 2.5% of the initial pre-construction cost, demonstrating that the reusable artifacts can be maintained with very low overhead as the engine evolves.

## 5.6 Ablation Study

We ablate the two pre-constructed components ofEngineAgent to quantify their contributions. As shown in Figure 12, removing the hierarchy-aware index reduces end-to-end success from 9/12 to 3/12 and increases average exploration turns from 42 to 271, showing that explicit hierarchy knowledge is critical for recovering complete implementation scopes and cross-layer dependencies. Removing the hardware-aware skills reduces success from 9/12 to 5/12 and increases compilation/debugging failures from 11 to 83, highlighting the importance of pre-validated build, verification, and debugging workflows for reliable hardware-grounded execution. Together, these results show that the two components are complementary: hierarchy-aware indexing guides complete implementation, while hardware-aware skills enable reliable on-device validation and debugging.

![](images/85631b750fb014c0498079303055d713ab02a4ac7a0c2bc8e0917d4fed15cc59.jpg)  
(a) w/o hierarchy index

![](images/31761ebcf3e0a68fa23c050e5c303d4801621584f1c7fa4423b1f4a5b038b678.jpg)  
(b) w/o hardware skills  
Figure 12. Ablation study of the hierarchy-aware index and hardware-aware skills.

## 6 Related Works

## 6.1 General-Purpose Coding Agents

Commercial coding agents [1, 3, 11, 25], SWE-agent [42], and OpenHands [38] integrate repository exploration, editing, and execution feedback. Agentless [40] structures issue resolution into localization, repair, and validation. Repository understanding draws on dependency-aware planning [4], code graphs [6, 21, 29], and program structure [48]. Other work supports interactive debugging [45], reusable skills learned from execution traces [41], and tree search with agent-based evaluation [2].

## 6.2 Mobile Programming and Deployment Agents

AutoIOT [32] and GPIoT [30] synthesize AIoT programs from natural-language requirements, while AgentDeploy [31] partitions models across existing inference engines and reuses optimization configurations for mobile deployment. At the operator level, MoKA, introduced in MobileKernelBench [52], combines repository-aware reasoning with iterative implementation and on-device validation. EngineAgent targets complete support for previously unsupported LLM and multimodal architectures, requiring coordinated changes across model conversion, graph construction, runtime-state management, and backend execution, validated through end-toend execution on the target device.

## 6.3 Automation of ML Systems

TVM [5], Ansor [49], Roller [50], and Welder [33] automate tensor compilation and optimization through graph transformations, scheduling, search, and fusion. KernelBench [28] evaluates LLM-generated GPU kernels, while GEAK [37] and KernelAgent [7] guide kernel optimization with execution feedback and hardware information.

## 7 Conclusion

Supporting rapidly evolving LLMs on heterogeneous mobile hardware makes model enablement a recurring crosslayer systems challenge. We presented EngineAgent, which separates reusable engine and hardware knowledge from model-specific implementation. Its hierarchy-aware index guides coordinated changes across engine abstractions, while hardware-aware workflows and a builder–auditor loop ground implementation and debugging in device-side validation. Our evaluation shows that this design improves model-enablement success while reducing time and model-API cost compared with coding-agent baselines. More broadly, EngineAgent ofers a path to turn specialized integration expertise into reusable infrastructure, reducing the gap between model innovation and deployment across diverse hardware.

## References

[1] Anthropic. 2026. Claude Code. htps://www.anthropic.com/claudecode. Accessed: 2026-09-02.

[2] Antonis Antoniades, Albert Örwall, Kexun Zhang, Yuxi Xie, Anirudh Goyal, and William Yang Wang. 2025. SWE-Search: Enhancing Soft ware Agents with Monte Carlo Tree Search and Iterative Refinement. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. htps://openreview.net/forum?id=G7sIFXugTX

[3] Anysphere, Inc. 2026. Cursor. htps://cursor.com/. Accessed: 2026-09- 02.

[4] Ramakrishna Bairi, Atharv Sonwane, Aditya Kanade, Vageesh D C, Arun Iyer, Suresh Parthasarathy, Sriram Rajamani, B. Ashok, and Shashank Shet. 2024. CodePlan: Repository-level Coding using LLMs and Planning. Proceedings ofthe ACM on Software Engineering 1, FSE (July 2024), 675–698.

[5] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. 2018. TVM: an automated end-to-end optimizing compiler for deep learning. In Proceedings of the 13th USENIX Conference on Operating Systems Design and Implementation (OSDI ’18). USENIX Association, Carlsbad, CA, USA, 579–594.

[6] Zhaoling Chen, Robert Tang, Gangda Deng, Fang Wu, Jialong Wu, Zhiwei Jiang, Viktor K. Prasanna, Arman Cohan, and Xingyao Wang. 2025.

LocAgent: Graph-Guided LLM Agents for Code Localization. In Proceedings ofthe 63rd Annual Meeting ofthe AssociationforComputational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 8697–8727. htps://doi.org/10.18653/V1/2025.ACL-LONG.426

[7] Kaiming Cheng, Laura Wang, Jack Khuu, Mark Saroufim, Wenyuan Chi, Jiannan Wang, and Joe Isaacson. 2026. KernelAgent: Hardware-Guided GPU Kernel Optimization via Multi-Agent Orchestration. Py-Torch Blog. htps://pytorch.org/blog/kernelagent-hardware-guidedgpu-kernel-optimization-via-multi-agent-orchestration/ Published: 2026-03-06. Accessed: 2026-09-25.

[8] DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, et al. 2026. DeepSeek-V4: Towards Highly Eficient Million-Token Context Intelligence. arXiv:2606.19348 [cs.CL] htps: //arxiv.org/abs/2606.19348

[9] PyTorch Foundation. 2025. GitHub - pytorch/executorch: On-device AI across mobile, embedded and edge for PyTorch. htps://github.com/ pytorch/executorch.

[10] Georgi Gerganov. 2026. GitHub - ggml-org/llama.cpp: LLM inference in C/C++. htps://github.com/ggml-org/llama.cpp.

[11] GitHub. 2026. GitHub Copilot. htps://github.com/features/copilot. Accessed: 2026-09-02.

[12] Google DeepMind. 2025. Gemma 3n. htps://deepmind.google/models/ gemma/gemma-3n/. Accessed: 2026-09-09.

[13] Google DeepMind. 2026. Gemma 4. htps://deepmind.google/models/ gemma/gemma-4/. Accessed: 2026-06-30.

[14] Apple Inc. 2026. CoreML Documentation. htps://developer.apple. com/documentation/coreml.

[15] MediaTek Inc. 2026. NeuroPilot Documentation. htps://neuropilotdeveloper.mediatek.com/sphinx/neuropilot-8-public/html/.

[16] Qualcomm Technologies Inc. 2026. Qualcomm Documentation — HTP Backend. htps://docs.qualcomm.com/doc/80-63442-10/topic/htp\_ backend.html. [Accessed 25-02-2026].

[17] Qualcomm Technologies Inc. 2026. Qualcomm Documentation — Introduction to NPU. htps://docs.qualcomm.com/nav/home/hexagon-dspsdk-introduction-to-npu.html?product=1601111740010422. [Accessed 25-02-2026].

[18] Arm Limited. 2020. The Bifrost Shader Core Version 1.0. htps:// developer.arm.com/documentation/102546/0100/?lang=en.

[19] Hannah Lin, Martin Maas, Maximilian Roquemore, Arman Hasanzadeh, Fred Lewis, Yusuf Simonson, Ameya Shringi, Hongwen Dai, Patrick Musau, Tzu-Wei Yang, Amir Yazdanbakhsh, Deniz Altınbüken, Florin Papa, Maggie Nolan Edmonds, Aditya Patil, Don Schwarz, Satish Chandra, Chris Kennelly, Milad Hashemi, and Parthasarathy Ranganathan. 2026. ECO: An AI-Driven Code Eficiency Optimizer for Warehouse Scale Computers (Operational Systems). In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26). USENIX Association, Seattle, WA, 589–625.

[20] Qingyuan Liu, Mo Zou, Hengbin Zhang, Dong Du, Yubin Xia, and Haibo Chen. 2026. Sharpen the Spec, Cut the Code: A Case for Generative File System with SYSSPEC. In 24th USENIX Conference on File and Storage Technologies (FAST 26). USENIX Association, Santa Clara, CA, 291–311.

[21] Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael Qizhe Shieh, and Wenmeng Zhou. 2025. CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. In Proceedings ofthe 2025 Conference ofthe Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Association for Computational Linguistics, 142–160. htps://doi.org/10.18653/V1/2025.NAACL-LONG.7

[22] Chengfei Lv, Chaoyue Niu, Renjie Gu, Xiaotang Jiang, Zhaode Wang, Bin Liu, Ziqi Wu, Qiulin Yao, Congyu Huang, Panos Huang, Tao Huang, Hui Shu, Jinde Song, Bin Zou, Peng Lan, Guohuan Xu, Fei Wu, Shaojie Tang, Fan Wu, and Guihai Chen. 2022. Walle: An End-to-End, General Purpose, and Large-Scale Production System for Device-Cloud Collaborative Machine Learning. In Proceedings ofUSENIX Symposium on Operating Systems Design and Implementation (OSDI ’22). USENIX, Carlsbad, CA, USA, 249–265.

[23] Colby McHenry. 2026. CodeGraph. htps://github.com/colbymchenry/ codegraph. Accessed: 2026-09-02.

[24] MLC team. 2023-2025. MLC-LLM. CMU Foundation and Language Model Center. htps://github.com/mlc-ai/mlc-llm

[25] OpenAI. 2026. Codex. htps://openai.com/codex/. Accessed: 2026-09- 02.

[26] OpenBMB. 2026. MiniCPM-V 4.6. htps://huggingface.co/openbmb/ MiniCPM-V-4.6. Hugging Face model card; accessed: 2026-06-30.

[27] OpenCode Contributors. 2026. OpenCode: The Open Source Coding Agent. htps://github.com/anomalyco/opencode. Accessed: 2026-09- 02.

[28] Anne Ouyang, Simon Guo, Simran Arora, Alex L Zhang, William Hu, Christopher Re, and Azalia Mirhoseini. 2025. KernelBench: Can LLMs Write Eficient GPU Kernels?. In Forty-second International Conference on Machine Learning (ICML ’25).

[29] Siru Ouyang, Wenhao Yu, Kaixin Ma, Zilin Xiao, Zhihan Zhang, Mengzhao Jia, Jiawei Han, Hongming Zhang, and Dong Yu. 2025. RepoGraph: Enhancing AI Software Engineering with Repositorylevel Code Graph. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. htps://openreview.net/forum?id=dw9VUsSHGB

[30] Leming Shen, Qiang Yang, Xinyu Huang, Zijing Ma, and Yuanqing Zheng. 2025. GPIoT: Tailoring Small Language Models for IoT Program Synthesis and Development. In Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems (UC Irvine Student Center., Irvine, CA, USA) (SenSys ’25). Association for Computing Machinery, New York, NY, USA, 199–212.

[31] Leming Shen, Qiang Yang, Xinyu Huang, Zijing Ma, Yuanqing Zheng, and Chris Xiaoxuan Lu. 2026. Automating Model Deployment on Mo bile Devices. In Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services Workshops (University of Cambridge, Cambridge, United Kingdom) (MobiSys Workshop ’26). Association for Computing Machinery, New York, NY, USA, 188–194.

[32] Leming Shen, Qiang Yang, Yuanqing Zheng, and Mo Li. 2025. AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications. In Proceedings of the 31st Annual International Conference on Mobile Computing and Networking (Kerry Hotel, Hong Kong, Hong Kong, China) (ACM MOBICOM ’25). Association for Computing Machinery, New York, NY, USA, 468–482.

[33] Yining Shi, Zhi Yang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Ziming Miao, Yuxiao Guo, Fan Yang, and Lidong Zhou. 2023. Welder: Scheduling Deep Learning Memory Access via Tile-graph. In 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, Boston, MA, 701–718.

[34] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, et al. 2026. OpenAI GPT-5 System Card. arXiv:2601.03267 [cs.CL] htps://arxiv.org/abs/2601.03267

[35] Gemma Team. 2025. Gemma 3 Technical Report. arXiv:2503.19786 [cs.CL] htps://arxiv.org/abs/2503.19786

[36] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. htps://qwen.ai/blog?id=qwen3.5.

[37] Jianghui Wang, Vinay Joshi, Saptarshi Majumder, Xu Chao, Bin Ding, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, and Emad Barsoum. 2025. Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks. CoRR abs/2507.23194 (2025). htps://doi.org/ 10.48550/ARXIV.2507.23194 arXiv:2507.23194

[38] Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Daniel Shao, Niklas Muennighof, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. 2025. Open-Hands: An Open Platform for AI Software Developers as Generalist Agents. In International Conference on Learning Representations (ICLR ’25). 65882–65919.

[39] Zhaode Wang, Jingbang Yang, Xinyu Qian, Shiwen Xing, Xiaotang Jiang, Chengfei Lv, and Shengyu Zhang. 2024. MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices. In Proceedings of the 6th ACM International Conference on Multimedia in Asia Workshops (MMAsia ’24 Workshops). Association for Computing Machinery, New York, NY, USA, Article 11, 7 pages.

[40] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2025. Demystifying LLM-Based Software Engineering Agents. Proc. ACM Softw. Eng. 2, FSE (2025), 801–824. htps://doi.org/10.1145/ 3715754

[41] Chen Yang, Jiashuo Tian, Ziqi Wang, Xinyin Liu, Meiru Ye, and Junjie Chen. 2026. Learning Globally Reusable Skills for Coding Agents. CoRR abs/2608.06153 (2026). htps://doi.org/10.48550/ARXIV.2608.06153 arXiv:2608.06153

[42] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-Agent: Agent-Computer Interfaces Enable Automated Software Engineering. In Neural Information Processing Systems (NIPS ’24).

[43] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025. Gated Delta Networks: Improving Mamba2 with Delta Rule. In The Thirteenth International Conference on Learning Representations (ICLR ’25).

[44] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, et al. 2025. MiniCPM-V 4.5: Cooking Eficient MLLMs via Architecture, Data, and Training Recipe. arXiv:2509.18154 [cs.LG] htps://arxiv.org/abs/2509.18154

[45] Xingdi Yuan, Morgane M. Moss, Charbel El Feghali, Chinmay Singh, Darya Moldavskaya, Drew MacPhee, Lucas Caccia, Matheus Pereira, Minseon Kim, Alessandro Sordoni, and Marc-Alexandre Côté. 2025. debug-gym: A Text-Based Environment for Interactive Debugging. CoRR abs/2503.21557 (2025). htps://doi.org/10.48550/ARXIV.2503. 21557 arXiv:2503.21557

[46] Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng YAN. 2026. AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?. In The Fourteenth International Conference on Learning Representations (ICLR ’26).

[47] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. 2024. CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (ACL ’24). Association for Computational Linguis tics, Bangkok, Thailand, 13643–13658.

[48] Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. 2024. AutoCodeRover: Autonomous Program Improvement. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis (Vienna, Austria) (ISSTA ’24). Association for Computing Machinery, New York, NY, USA, 1592–1604.

[49] Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, and Ion Stoica. 2020. Ansor: Generating High-Performance Tensor Programs for Deep Learning. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). USENIX Association, 863–879.

[50] Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang, Mao Yang, Lidong Zhou, Asaf Cidon, and Gennady Pekhimenko. 2022. ROLLER: Fast and Eficient Tensor Compilation for Deep Learning. In

16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). USENIX Association, Carlsbad, CA, 233–248.

[51] Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, and Jiaxuan You. 2025. Where LLM Agents Fail and How They can Learn From Failures. arXiv:2509.25370 [cs.AI] htps://arxiv.org/abs/2509.25370

[52] Xingze Zou, Jing Wang, Yuhua Zheng, Xueyi Chen, Haolei Bai, Lingcheng Kong, Syed A. R. Abu-Bakar, Zhaode Wang, Chengfei Lv, Haoji Hu, and Huan Wang. 2026. MobileKernelBench: Can LLMs Write Eficient Kernels for Mobile Devices? CoRR abs/2603.11935 (2026). htps://doi.org/10.48550/ARXIV.2603.11935 arXiv:2603.11935