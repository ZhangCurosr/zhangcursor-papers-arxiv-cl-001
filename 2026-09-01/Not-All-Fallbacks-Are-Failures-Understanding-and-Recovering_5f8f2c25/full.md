# Not All Fallbacks Are Failures: Understanding and Recovering from Fallbacks in Mobile Voice Assistants

Phillip Schneider<sup>1,2</sup>\*, Alexandre Mercier<sup>2</sup>\*, Joshua Oehms<sup>2</sup>,

Kristiina Jokinen<sup>3</sup>, Florian Matthes<sup>2</sup>

<sup>1</sup>ALMA PHIL, Germany

<sup>2</sup>Technical University of Munich, Department of Computer Science, Germany <sup>3</sup>CNRS-AIST Joint Robotics Laboratory (JRL), Japan   
{phillip.schneider, alex.mercier, joshua.oehms, matthes}@tum.de kristiina.jokinen@aist.go.jp

## Abstract

Robust understanding of user input is a core requirement for voice assistants deployed in real-world environments. In practice, these systems encounter heterogeneous fallback situations caused by noisy audio input, transcription errors, ambiguous requests, incomplete utterances, or unintended activations. Existing systems typically respond with generic fallback messages, which do not resolve the underlying interaction failure and can degrade user experience. We study fallback handling in a deployed smartwatch-based voice assistant for general health support in everyday environments. Our analysis is based on six months of real-world usage data from more than 500 users, yielding a dataset of 3,030 anonymized, naturally occurring fallback-triggering utterances. We contribute (1) an operational taxonomy and the annotated VOXFALLBACKS dataset of these interactions, (2) a comparative evaluation of different models within a classification pipeline under practical deployment constraints, and (3) practical lessons for designing robust and cost-efficient fallback mechanisms. Results show that lightweight embedding-based classifiers outperform larger generative models on most classification tasks while requiring substantially fewer computational resources.

## 1 Introduction

Conversational agents are increasingly embedded in mobile and assistive technologies, including smartphones, smart speakers, and wearable devices (Seaborn et al., 2021). Unlike research prototypes, deployed systems must operate in noisy and unpredictable environments where user behavior often deviates from training data and design assumptions. Voice-based interfaces face particular challenges: background noise, accidental microphone activations (Schönherr et al., 2021), speech recognition errors, incomplete utterances, and requests targeting unsupported capabilities (Tur et al., 2014). These failure modes are routine in production yet systematically underrepresented in standard benchmarks, which typically assume well-formed, indomain inputs (Larson et al., 2019). Deployed systems must handle a large fraction of interactions that fail early in the pipeline. These failures are commonly routed to a fallback mechanism, which prevents incorrect system actions but provides limited support for interaction recovery. Standard fallback strategies often rely on generic responses such as rephrasing requests or offering help, without distinguishing between different underlying causes of failure.

![](images/66294d96a3c0cbdd2b9c7855769ed0092f25e961fade066e49a18fcf962431a6.jpg)  
Figure 1: Classification taxonomy for fallback handling.

This work studies fallback handling in a deployed smartwatch-based voice assistant that provides health-related support in everyday environments. The system is used by older adults and individuals with chronic conditions, making robust interaction a practical requirement rather than a desirable feature. Analysis of real-world usage logs revealed that fallback-triggering utterances arise from diverse sources, including ambiguous requests, unsupported functionality, accidental activations, transcription errors, and recoverable requests that the assistant failed to interpret correctly. Rather than representing a single failure mode, these utterances correspond to different underlying interaction problems and therefore require different recovery strategies, such as clarification, capability explanations, or skill recovery. Crucially, for accidental activations, the system must accurately discern when not to react; failing to suppress such responses results in intrusive behavior that can increase user frustration and contribute to churn.

Prior work typically treats out-of-domain detection (Tur et al., 2014), clarification generation (Zhang et al., 2024; Sahay et al., 2025), and accidental activation filtering (Schönherr et al., 2021) as separate problems. In voice assistants operating in real-world environments, all three appear together within the same fallback stream, demanding a unified detection and intervention approach under practical constraints of latency, computational cost, and maintainability. We therefore frame fallback handling as a structured recovery problem, focusing on selecting appropriate recovery actions rather than end-to-end recovery outcomes. Using six months of real-world usage data, we construct and analyze a dataset of anonymized fallbacktriggering utterances and investigate how fallback interactions can be routed to appropriate recovery actions under practical deployment constraints.

Our contributions are threefold: we (1) provide an operational taxonomy and the annotated VOX-FALLBACKS dataset<sup>1</sup> with 3,030 naturally occurring fallback-triggering utterances collected from a wearable smartwatch voice assistant, (2) evaluate multiple classification approaches for routing fallback interactions to appropriate recovery strategies under deployment-oriented constraints, including accuracy, latency, and computational cost, and (3) derive practical insights showing that lightweight embedding classifiers outperform instruction-tuned large language models (LLMs) for fallback routing.

Overall, the paper highlights that fallback handling should not be viewed as a single error state but as a structured recovery process. Rather than defaulting to resource-intensive generative models for every component, which is a prevailing trend that warrants closer empirical scrutiny regarding actual utility, our findings demonstrate how combining taxonomy-driven classification with strictly targeted use of LLMs can improve robustness in real-world voice assistants while maintaining the efficiency requirements of deployed systems.

## 2 Related Work

While fragmented across separate research streams, fallback handling comprises three main areas.

Accidental Activations & Unintended Speech. Wearable devices frequently trigger on nondirected speech, leading to intrusive system behavior. Prior research treats this primarily as a detection task, focusing on wake-word false positives (Larson et al., 2019; Schönherr et al., 2021) or analyzing linguistic breakdown signals like filled pauses and repetitions (Alloatti et al., 2024; Ngo et al., 2025). While these studies excel at filtering, they often operate in isolation from the broader conversation flow. We extend this by quantifying the prevalence of unintended speech within a production stream and demonstrating that lightweight classifiers can effectively gate these interactions before they reach more costly modules.

Out-of-Domain Requests & Conversational Breakdown. Traditional approaches to conversational breakdown focus on recovery strategies, such as re-asking, apologizing, or capability disclosure (Benner et al., 2021; Alghamdi et al., 2024; Stevens et al., 2025). Related design-science research emphasizes collaborative repair (Gnewuch and Reinkemeier, 2025) and differentiating between system-initiated and user-initiated recovery (Feng, 2023). Concurrently, out-of-domain (OOD) detection has historically relied on confidence thresholds or prototype-based discriminators (Tur et al., 2014). We bridge these streams by refining the OOD framing: we distinguish between requests that are genuinely unsupported (requiring limitation explanations) and those that are merely underspecified or mis-transcribed (requiring re-routing).

Ambiguous Utterances & Clarification. As reliance on LLMs grows, generating effective clarifications has become a central challenge. Recent findings indicate that LLMs often default to silent intent assumption or over-generic probes rather than meaningful clarification (Zhang et al., 2024, 2025; Sahay et al., 2025). Architectural approaches that pair clarity classifiers with specialized generation modules (Murzaku et al., 2025) align with our design. Our work empirically validates this decoupling: lightweight models provide robust classification, while LLMs are reserved for generating contextually appropriate clarifications only when intent is confirmed as ambiguous.

## 3 Dataset Construction

Data Source & Collection. The primary data source for this study is a smartwatch-based voice assistant tailored for elderly and chronically ill individuals across the German-speaking DACH region. Designed to support people in need of care during both daily living and critical situations, the assistant provides a suite of native capabilities: acute emergency call routing, active health tracking (blood pressure, pulse, or steps), interactive microdialogues (medication and hydration reminders), and general informational lookups (general knowledge questions or weather forecasts).

Audio capture and speech-to-text (STT) processing are triggered via two modalities: (1) manual activation, where the user touches a dedicated complication on the smartwatch display, and (2) proactive activation, where the agent initiates a turn (e.g., a hydration reminder) and opens the microphone for an open-ended user response.

The core pipeline of the deployed system relied on Microsoft Azure STT<sup>2</sup> via its German Universal Language Model for transcription, and the Google Dialogflow agent framework<sup>3</sup> leveraging its proprietary natural language understanding (NLU) for intent classification.

We collected single-turn user utterances that triggered the system’s fallback mechanism (operationally defined as any input where the core intent classification confidence fell below a threshold) over a six-month period spanning 2024–2025. The core system capabilities remained stable throughout this window. Across the collection period, fallback interactions accounted for approximately 6–7% of all assistant requests. The extracted fallback dataset is associated with more than 500 unique individuals aged predominantly between 60–90 years. To comply with data protection requirements, all personally identifiable information was removed, and only fully de-identified data were made available for academic analysis.

Data Selection & Preprocessing. The raw dataset contained 7,857 fallback-triggering user inputs collected from a deployed voice assistant. To obtain a clean corpus for analysis, we applied a multi-phase preprocessing pipeline. First, we retained only dialogue-initiating utterances and removed mid-dialogue responses constrained by system expectations (e.g., yes/no or numeric answers). Second, exact duplicate utterances were discarded to reduce artifacts from repeated accidental activations and background noise. Third, empty transcriptions were removed. Fourth, all personally identifiable information, including names, addresses, facilities, and local businesses, was manually anonymized by replacement with generic German names and major cities. Finally, utterances were randomized to prevent reconstruction of userspecific interaction histories. The resulting corpus comprises 3,030 unique, fully anonymized fallback utterances.

Because we aim to characterize diverse fallback phenomena rather than raw production frequency, we deduplicated exact utterances to avoid overrepresenting repetitive events. To assess its impact, we classified all 7,857 records using a support vector machine (SVM) (weighted F1 = 0.87). The resulting distribution differs only moderately from the annotated dataset (Jensen–Shannon distance = 0.119), with the largest shift being an increase in unintended input (Lin, 1991).

Data Annotation Protocol. To map fallback phenomena into actionable engineering decisions, we designed a hierarchical taxonomy using a ruleinformed, inductive approach, combining theoretical frameworks from related work with a bottom-up review of production logs to match the actual capabilities of the deployed assistant. As illustrated in Figure 1, the taxonomy is structured as a decision tree with three distinct hierarchical levels: (1) user intentionality separates accidental activations or ambient background noise (unintended) from purposeful interactions (intended); (2) intent clarity evaluates whether an actionable objective can be extracted from intentional inputs, partitioning vague utterances into either ambiguous requests or multi-intent conflicts (unclear intent); and (3) capability availability maps well-formed requests (clear intent) to either the assistant’s supported feature set (available) or unsupported domains (unavailable).

Annotating fallback-triggering utterances is challenging due to ambiguous requests, speech recognition errors, and incomplete context. We therefore adopted a multi-phase annotation process involving four native German-speaking researchers. In an initial phase, a research assistant labeled all 3,030 utterances and iteratively discussed borderline cases with the research team to refine label definitions and decision rules (Table 6). Subsequently, three researchers independently annotated the dataset, with a shared validation subset of 350 utterances labeled by all annotators. Inter-annotator agreement on this subset reached a Fleiss’ Kappa (Fleiss, 1971) of κ = 0.759 (Table 4), indicating substantial agreement (Landis and Koch, 1977). Labels for the validation subset were determined by majority vote, with ties resolved through discussion. Finally, all disagreements between the refinement and independent annotation phases were reviewed jointly to produce the final labels.

## 4 Experimental Setup

Pipeline & Models. We evaluate a two-stage classification pipeline aligned with the taxonomy hierarchy (Figure 3). Stage 1 distinguishes intended from unintended utterances. Inputs classified as unintended are suppressed and exit the pipeline. Stage 2 processes intended utterances, first determining whether intent is clear or unclear and, for clear-intent requests, whether the requested capability is available or unavailable.

We compare lightweight embedding-based classifiers and instruction-tuned LLMs. For the embedding approach, we use the multilingual models multilingual-e5-large (E5) (Wang et al., 2024) and BGE-M3 (Chen et al., 2024), each paired with logistic regression (LR), SVM, and random forest classifiers. All classifiers use default hyperparameters unless stated otherwise; for LR and SVM, preliminary experiments indicated that C = 10 and max\_iter=2000 yielded the most promising results, and for random forest we use n\_estimators=200. We additionally evaluate Gemma 4 E4B (Google DeepMind, 2026), Qwen 3.5 4B (Qwen Team, 2026), and GPT-4.1 nano.<sup>4</sup> To reflect deployment constraints, we use small 4B variants of the open-source models. LLMs are evaluated in zero-shot and few-shot settings with temperature 0 and reasoning disabled for consistency and latency reasons.

Prompt Design. All prompts are provided in full in Appendix A. We use the same prompt design principles across LLM experiments: prompts explicitly define the system role, describe the available target and fallback labels, and enforce a strict output format with ambiguity guardrails. In Stage 2, we evaluate two configurations. In the zeroshot configuration, the prompt exposes the complete set of 18 system skills and two fallback labels, each with a concise description. In the retrievalconditioned configuration, the prompt instead provides only the top five intent candidates retrieved by a Dual Intent and Entity Transformer (DIET) from Bunk et al. (2020) trained on the deployed assistant’s training phrases, together with the same fallback labels. Thus, the retrieval-conditioned setup constrains the LLM’s prediction space to intents considered relevant by the upstream classifier.

We evaluate retrieval conditioning only as part of the complete end-to-end classification pipeline. We do not separately assess the DIET model’s recall or candidate-ranking quality. Accordingly, the reported results measure the effect of conditioning the LLM on retrieved candidates, rather than the retrieval component in isolation.

Evaluation Process. Models are evaluated using stratified 5-fold cross-validation. For Stage 1, we report accuracy, precision, recall, F1, ROC AUC, and average precision. For Stage 2, we report macro-averaged precision, recall, and F1 to account for class imbalance. Since ROC AUC requires a continuous confidence score, we derive probabilities for LLMs from the log-probabilities of the binary output tokens. Concretely, let $\ell _ { 1 }$ and $\ell _ { 0 }$ denote the log-probabilities of tokens "1" and "0" at the first decoded position. The positive-class probability is then computed as:

$$
p _ { 1 } = \frac { e ^ { \ell _ { 1 } } } { e ^ { \ell _ { 1 } } + e ^ { \ell _ { 0 } } }\tag{1}
$$

This approach follows established practice for deriving confidence scores from autoregressive model outputs (Malinin and Gales, 2021; Kuhn et al., 2023). In addition, we report mean classifier inference latency per utterance across all models. For embedding-based classifiers, this measurement covers only the classifier inference after embeddings have been computed; embedding computation itself takes approximately 40 ms per utterance.

<table><tr><td>Class</td><td>Count</td><td>Percentage</td></tr><tr><td>unintended</td><td>2,138</td><td>70.56%</td></tr><tr><td>intended/clear int./available</td><td>731</td><td>24.13%</td></tr><tr><td>intended/clear int./unavailable</td><td>103</td><td>3.40%</td></tr><tr><td>intended/unclear int./unspecific</td><td>52</td><td>1.72%</td></tr><tr><td>intended/unclear int./multiple</td><td>6</td><td>0.20%</td></tr><tr><td>Total</td><td>3,030</td><td>100.00%</td></tr></table>

Table 1: Aggregated class distribution of the annotated fallback dataset.

<table><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td><td>ROC AUC</td><td>Avg Precision</td><td>Latency (ms)</td></tr><tr><td>E5+LR</td><td>.908±.011</td><td>.810±.021</td><td>.900±.026</td><td>.852±.018</td><td>.966±.006</td><td>.931±.011</td><td>.043±.001</td></tr><tr><td>BGE-M3+LR</td><td>.901±.015</td><td>.806±.035</td><td>.879±.034</td><td>.840±.023</td><td>.960±.008</td><td>.922±.012</td><td>.043±.000</td></tr><tr><td>E5+SVM</td><td>.913±.015</td><td>.884±.042</td><td>.814±.028</td><td>.847±.026</td><td>.964±.006</td><td>.928±.009</td><td>.619±.004</td></tr><tr><td>BGE-M3+SVM</td><td>.901±.003</td><td>.872±.023</td><td>.780±.022</td><td>.823±.005</td><td>.955±.006</td><td>.913±.011</td><td>.614±.014</td></tr><tr><td>E5+random forest</td><td>.909±.012</td><td>.908±.031</td><td>.768±.019</td><td>.832±.021</td><td>.957±.010</td><td>.922±.014</td><td>13.250±.031</td></tr><tr><td>BGE-M3+random forest</td><td>.904±.005</td><td>.918±.010</td><td>.740±.024</td><td>.819±.012</td><td>.957±.008</td><td>.922±.011</td><td>13.265±.033</td></tr><tr><td>Gemma 4 (zero-shot)</td><td>.877±.012</td><td>.763±.022</td><td>.845±.019</td><td>.802±.019</td><td>.940±.011</td><td>.896±.015</td><td>242.191±3.003</td></tr><tr><td>Gemma 4 (few-shot)</td><td>.891±.007</td><td>.808±.020</td><td>.829±.009</td><td>.818±.011</td><td>.943±.008</td><td>.899±.014</td><td>228.888±4.418</td></tr><tr><td>Qwen 3.5 (zero-shot)</td><td>.635±.019</td><td>.443±.014</td><td>.929±.012</td><td>.600±.012</td><td>.902±.013</td><td>.851±.018</td><td>188.362±4.493</td></tr><tr><td>Qwen 3.5 (few-shot)</td><td>.448±.013</td><td>.347±.006</td><td>.993±.006</td><td>.515±.007</td><td>.926±.009</td><td>.880±.014</td><td>183.568±2.684</td></tr><tr><td>GPT-4.1 nano (zero-shot)</td><td>.876±.014</td><td>.817±.036</td><td>.748±.021</td><td>.780±.023</td><td>.915±.012</td><td>.853±.035</td><td>637.476±52.588</td></tr><tr><td>GPT-4.1 nano (few-shot)</td><td>.859±.011</td><td>.756±.026</td><td>.774±.016</td><td>.764±.016</td><td>.912±.014</td><td>.825±.044</td><td>1660.601±286.86</td></tr></table>

Table 2: Binary classification results for the first stage of the pipeline. Bold values indicate best performance.

## 5 Results & Discussion

## 5.1 Dataset Findings

Our analysis focuses on the diversity of unique fallback utterances rather than their absolute operational frequency, providing a structural overview of the deduplicated dataset. As shown in Table 1, unintended activations constitute the largest portion of the annotated utterances (70.56%). These unintended examples are notably longer and show higher length variance (averaging 48.28 characters, SD = 45.50) than both the overall dataset average (43.17 characters) and intended clear intent inputs (31.31 characters), frequently capturing longer fragments of background speech or ambient audio.

Conversely, intended requests with a clear intent targeting available skills make up 24.13% of the unique fallback utterances in the corpus. Within this subset of potentially recoverable errors, instances are concentrated in open-ended, high-variance domains like general knowledge searches, creative writing prompts, and news lookups, whereas highly structured utility tasks such as setting timers or measuring vitals are less frequent. Finally, semantic ambiguities, consisting of unspecific or multi-intent requests, are rare, representing less than 2% of the unique fallback utterances combined.

In addition, our dataset exposes the rich diversity of real-world fallback triggers, demonstrating that system failures are not a monolithic error state but a highly heterogeneous spectrum spanning four distinct pipeline layers (Table 5). At the physical layer, the data captures acoustic and environmental diversity, ranging from truncated signals like microphone cut-offs to ambient background noise (e.g., radio or TV). Moving to the STT layer, failures shift to phonetic distortions, producing wrong transcriptions of phonetically adjacent words, such as confusing “Tassen”, the German word for cups, with “Tasten”, the word for buttons. The NLU layer reveals conversational and syntactic complexity, characterized by mid-utterance selfcorrections where users fluidly change their mind mid-command. Finally, the context layer highlights pragmatic diversity, capturing the critical distinction between clear but out-of-scope requests (e.g., asking to play a specific song) and side-speech directed at another person, underscoring why generic fallback responses are insufficient for real-world deployment.

## 5.2 Classification Results

Binary Classification. Embedding-based classifiers consistently outperform the LLMs in the firststage gating of unintended versus intended inputs, both in classification quality and inference latency (Table 2 and Figure 2). Across all three classifier types (LR, SVM, and random forest), the E5 embedding yields a higher F1 score than its BGE-M3 counterpart. The best overall result is achieved by E5 + LR (F1 = 0.852, accuracy = 0.908). More broadly, all six embedding-based models achieve F1 scores at or above the strongest LLM result (Gemma 4 few-shot, $\mathrm { F 1 } = \mathrm { 0 } . 8 1 8 )$ , while their classifier inference latency remains below 14 ms, compared with 184–1,661 ms for the LLMs. The APIbased GPT-4.1 nano configuration reaches 637 ms in the zero-shot and 1,661 ms in the few-shot setting, with these values reflecting external API communication and service-side processing rather than intrinsic model inference time. Figure 2 makes this accuracy–latency trade-off explicit: the embeddingbased models occupy the favorable region of the plot, combining higher F1 with substantially lower latency than the generative models. Hence, for this deployment setting, embedding-based classifiers are the preferred choice for first-stage gating, particularly when labeled training data are available.

![](images/166653610cfc2380db27d6ba61311466cf59749d1ef1878adb570162a5ff14f2.jpg)  
Figure 2: F1 scores of binary classification models plotted against mean latency. Colors denote model families, marker shapes denote approaches, and mean latency is shown on a logarithmic scale.

<table><tr><td>Model</td><td>Accuracy</td><td>Precision (Macro)</td><td>Recall (Macro) F1 (Macro)</td><td></td><td>Latency (ms)</td></tr><tr><td>E5+LR</td><td> $\overline { { . 8 2 3 { \pm } . 0 1 0 } }$ </td><td> $\overline { { . 7 7 1 \pm . 0 2 4 } }$ </td><td> $\overline { { . 8 2 1 \pm . 0 3 0 } }$ </td><td> $\overline { { . 7 8 0 { \pm } . 0 2 7 } }$ </td><td> $\overline { { . 0 4 9 { \pm } . 0 0 2 } }$ </td></tr><tr><td>BGE-M3+LR</td><td> $\mathbf { . 8 4 0 { \pm } . 0 1 3 }$ </td><td> ${ \bf . 8 1 3 \pm . 0 2 2 }$ </td><td> ${ \bf . 8 3 4 \pm . 0 2 7 }$ </td><td> $\mathbf { . 8 1 0 \pm . 0 1 5 }$ </td><td> $\mathbf { . 0 4 6 { \pm } . 0 0 4 }$ </td></tr><tr><td>Gemma 4</td><td> $. 7 5 4 \pm . 0 3 0$ </td><td> $. 7 4 9 { \pm } . 0 7 7$ </td><td> $. 7 4 1 { \pm } . 0 8 8$ </td><td> $. 7 2 7 { \pm } . 0 7 9$ </td><td> $2 6 9 . 7 4 6 { \pm } 2 9 . 0 4 4$ </td></tr><tr><td>Gemma 4 + retrieval</td><td> $. 6 7 7 { \scriptstyle \pm . 0 2 7 }$ </td><td> $. 7 3 4 \pm . 0 7 2$ </td><td> $. 6 9 2 { \pm } . 0 7 4$ </td><td> $. 6 7 5 { \pm } . 0 6 0$ </td><td> $5 2 6 . 4 5 2 { \scriptstyle \pm 3 2 . 0 2 3 }$ </td></tr><tr><td>Qwen 3.5</td><td> $. 7 5 9 { \pm } . 0 2 8$ </td><td> $. 6 9 6 { \pm } . 0 7 4$ </td><td> $. 7 1 8 { \pm } . 0 4 4$ </td><td> $. 6 7 9 { \pm } . 0 5 0 $ </td><td> $2 1 6 . 3 1 6 { \pm } 1 3 . 8 7 8$ </td></tr><tr><td>Qwen 3.5 + retrieval</td><td> $. 7 1 7 { \pm } . 0 3 9$ </td><td> $. 7 7 3 { \pm } . 0 8 4$ </td><td> $. 7 0 9 { \pm } . 0 4 5$ </td><td> $. 7 0 7 { \scriptstyle \pm . 0 6 1 }$ </td><td> $8 7 7 . 4 4 2 { \scriptstyle \pm 9 3 . 1 9 8 }$ </td></tr><tr><td>GPT-4.1 nano</td><td> $. 7 6 2 { \pm } . 0 2 0 $ </td><td> $. 7 4 4 { \pm } . 0 7 6$ </td><td> $. 7 5 2 { \pm } . 0 8 2$ </td><td> $. 7 2 1 { \pm } . 0 7 4$ </td><td> $6 4 2 . 8 2 1 { \scriptstyle \pm 2 3 . 8 3 1 }$ </td></tr><tr><td> $\mathrm { G P T - 4 . 1 \ n a n o + r e t r i e v a l }$ </td><td> $. 7 3 2 { \pm } . 0 2 4$ </td><td> $. 7 0 0 { \pm } . 0 6 9$ </td><td> $. 6 9 1 { \pm } . 0 7 3$ </td><td> $. 6 7 4 \pm . 0 7 0$ </td><td> $5 3 1 . 9 5 3 { \pm } 2 7 . 7 9 5$ </td></tr></table>

Table 3: Multi-class classification results for the second stage of the pipeline. Bold values indicate best performance.

Furthermore, Qwen 3.5 exhibits a typical generative failure mode: it achieves very high recall on the intended class (0.929/0.993) but suffers from collapsing precision (0.443/0.347), labeling almost everything as intended and consequently failing to suppress accidental activations. Few-shot prompting exacerbates this issue, reducing accuracy to 0.448, below the majority baseline. More broadly, few-shot prompting is not consistently beneficial for binary classification: Gemma 4 improves slightly (0.802 to 0.818) F1, whereas GPT-4.1 nano degrades (0.780 to 0.764). These results suggest that adding demonstrations can introduce decision bias without providing a useful signal for reliably improving the LLM’s ability to distinguish intended from unintended speech.

“unspecific”) are combined into a single “unclear” label. A similar picture emerges as in the first stage: the embedding-based classifiers achieve higher F1 scores than the LLMs while operating at substantially lower latency, with the F1 gap being even larger in the multi-class setting. $\mathbf { B G E - M 3 + L R }$ performs best overall (macro $\mathrm { F 1 } = 0 . 8 1 0 $ , accuracy = 0.840), while the best-performing LLM, Gemma 4, reaches only a macro F1 of 0.727. The embeddingbased classifiers also have substantially lower classifier inference latency (<0.05 ms vs. 216–877 ms), although embedding computation adds approximately 40 ms per utterance.

Multi-class Classification. Skill routing in the second stage is a single-label multi-class classification task that is fundamentally more complex than binary filtering, requiring semantic disambiguation across 18 skills and two fallback labels (Table 3). Specifically, the two unclear labels (“multiple” and

Notably, the relative performance of the two embedding models depends on the classification task. Whereas E5 consistently outperforms BGE-M3 across all classifier types in the binary classification setting, BGE-M3 combined with logistic regression achieves the highest macro F1 in the multi-class task. This suggests that embedding models should be evaluated separately for the specific classification tasks in which they are deployed rather than assuming that performance on one task generalizes to another.

Retrieval-conditioned prompting does not provide a consistent performance improvement across the three LLMs. Its effect differs substantially by model. For Gemma 4, retrieval reduces macro F1 from 0.727 to 0.675. GPT-4.1 nano shows a similar pattern, with macro F1 decreasing from 0.721 to 0.674. Qwen 3.5 behaves differently: retrieval increases macro precision from 0.696 to 0.773 and macro F1 from 0.679 to 0.707, while recall decreases slightly from 0.718 to 0.709 and accuracy from 0.759 to 0.717. Thus, retrieval can alter the precision–recall trade-off, but its benefit is modeldependent rather than universally positive.

One possible explanation is that restricting the LLM to a small set of retrieved candidates reduces the number of semantically plausible skills considered during prediction, which may suppress some incorrect skill assignments but can also prevent the model from selecting the correct skill when it is absent from the retrieved set. Because we do not evaluate the retrieval component independently, we cannot determine from these results whether such candidate omissions are the primary cause of the observed degradation. The results therefore support interpreting retrieval as a strong conditioning mechanism that changes the LLM’s decision space, rather than as an independently validated improvement to the routing system. The additional retrieval step also increases latency substantially for Gemma 4 and Qwen 3.5.

## 5.3 Failure Modes

For our deployed voice assistant, error costs are asymmetric, meaning their distribution is more significant than their total rate. Misrouting between available skills (routing to the wrong skill) and false activations on unintended input (routing outof-scope to an available skill) are high risk, as they trigger incorrect or intrusive actions that can reduce user trust or lead to churn. Conversely, routing a recoverable request to an unclear or out-of-domain label is low risk, as it only necessitates a clarification turn. Therefore, fallback handling should be optimized for interaction safety under asymmetric costs rather than symmetric accuracy.

Furthermore, confusions are systematic rather than random, clustering around semantically adjacent skills like daily news versus knowledge search. LLMs also shift failure directions across stages: after generating excessive false positives in the binary stage, they pivot during skill routing to funnel a large share of ambiguous interactions into outof-scope and unspecific fallback labels. Because these confusions align with the underlying structure, many can be resolved through prompt and label design without retraining. This advantage is unique to LLMs, as their target labels can be modified using natural language and they adapt easily to new categories and interaction types.

## 5.4 Practical Implications

Our results offer critical lessons for deploying robust fallback handling in resource-constrained mobile voice assistants, favoring a hybrid design that prioritizes latency and cost over minor classification gains. First, fallback inputs are highly heterogeneous, mixing accidental activations, STT errors, and OOD commands, meaning they must be filtered through a structured taxonomy rather than

handled uniformly.

Second, lightweight embedding classifiers are superior to LLMs for routing because they deliver equivalent or better performance at lower latency and compute cost, which is vital for the initial filtering of massive amounts of unintended input. Our preliminary experiments suggest that LLMs can generate high-quality clarification requests. However, as this was not systematically evaluated in this study, we consider LLM-based clarification generation a promising direction for future work.

Ultimately, these findings favor a hybrid architecture that optimizes efficiency and performance. Shifting the bulk of classification onto these embedding models significantly reduces compute and serving costs. Crucially for industry deployments, this approach eliminates reliance on external LLM APIs, making local self-hosting highly feasible, whereas serving LLMs requires a massive amount of resources.

## 6 Conclusion

We studied fallback handling in a deployed voice assistant using 3,030 annotated fallback utterances. We introduced the VOXFALLBACKS dataset and a three-level taxonomy that frames fallbacks as a structured recovery process. Across a two-stage routing pipeline, lightweight embedding-based classifiers consistently outperform LLMs for intent routing while operating at substantially lower latency. These results highlight a hybrid design that prioritizes low-latency discriminative models for routing, with generative models as a promising option for clarification, while accounting for realworld latency, cost, and privacy constraints. Future work should extend this framework to multi-turn recovery modeling with richer dialogue state tracking. It should also evaluate end-to-end task success under deployed conditions, including LLMbased clarification and stronger grounding through retrieval-conditioned classification and generation.

## Limitations

There are several limitations to both the dataset and the proposed fallback handling approach. First, the dataset consists of single-turn utterances only. This restricts the analysis to isolated interaction fragments and does not capture multi-turn recovery dynamics, which are central to real-world conversational repair. As a result, the observed failure modes and the proposed taxonomy reflect turnlevel breakdowns rather than complete dialogues.

Second, the data were collected from a single deployed system in a specific application context, language, and user population. The definition of a fallback is therefore tied to one STT-NLU pipeline and its confidence thresholding behavior. While this makes the dataset operationally grounded, it also means that the resulting categories should be interpreted as system-specific failure patterns rather than a universal taxonomy of dialogue breakdowns.

A further limitation lies in how the data were labeled. Annotations were based on transcripts without the full conversation history, and they were completed by researchers rather than the users themselves. Consequently, conclusions about what the users actually intended are just educated guesses rather than the ground truth.

On the modeling side, our evaluation is restricted to lightweight, small LLMs and embedding-based classifiers, motivated by latency, cost, and privacy constraints. While this reflects realistic deployment conditions, it limits conclusions about larger or more capable models. Additionally, evaluating LLM-based clarification generation remains an open challenge, as we do not measure endto-end task success or whether clarifications actually resolve user goals during interaction. Finally, while our discussion addresses the asymmetric costs of safety-critical errors (such as highly damaging false negatives like missed emergency intents), our performance evaluation treats all error types uniformly.

## Ethics Statement

This work complies with relevant data protection and research ethics standards. All data used in this study were collected and processed in accordance with GDPR (General Data Protection Regulation) regulations. To ensure privacy protection, the dataset intended for publication was carefully anonymized. Each utterance was manually reviewed during a multi-phase cleaning process, including independent inspection by three annotators, to remove or modify any personally identifiable information. In addition, all entries were carefully screened to ensure the absence of offensive or harmful content before inclusion in the final dataset. For LLM inference using external APIs, only anonymized utterances were submitted, with no directly identifying information included.

All embedding-based experiments and opensource LLM experiments were conducted using locally available computational resources on a Mac-Book Pro with an Apple M3 Pro chip and 18 GB of unified memory, ensuring that no large-scale external compute infrastructure was required. Beyond data privacy, additional ethical considerations include the deployment context of conversational systems in assistive settings, where misclassification or inappropriate fallback handling may affect user experience. The proposed methods are designed to improve robustness and reduce intrusive or unhelpful system behavior, particularly in sensitive application domains such as assistive voice interaction.

## Acknowledgments

This work has been supported by funds from the Bavarian Ministry of Economic Affairs, Regional Development and Energy (StMWi) as part of the program “BayVFP Förderlinie Digitalisierung – Informations- und Kommunikationstechnologie”. We would like to thank David Uhlenbrock for his valuable work on the initial annotation and refinement of the dataset.

## References

Essam Alghamdi, Martin Halvey, and Emma Nicol. 2024. System and user strategies to repair conversational breakdowns of spoken dialogue systems: A scoping review. In Proceedings ofthe 6th ACM Conference on Conversational User Interfaces, Luxembourg.

Francesca Alloatti, Francesca Grasso, Roger Ferrod, Giovanni Siragusa, Luigi Di Caro, and Federica Cena. 2024. A tag-based methodology for the detection of user repair strategies in task-oriented conversational agents. Computer Speech & Language, 86:101603.

Dennis Benner, Edona Elshan, Sofia Schöbel, and Andreas Janson. 2021. What do you mean? A review on recovery strategies to overcome conversational breakdowns of conversational agents. In Proceedings of the International Conference on Information Systems (ICIS).

Tanja Bunk, Daksh Varshneya, Vladimir Vlasov, and Alan Nichol. 2020. Diet: Lightweight language understanding for dialogue systems. arXiv preprint arXiv:2004.09936.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3- embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024,

pages 2318–2335, Bangkok, Thailand. Association for Computational Linguistics.

Shengjia Feng. 2023. How to convey resilience: Towards a taxonomy for conversational agent breakdown recovery strategies. In Wirtschaftsinformatik 2023 Proceedings.

Joseph L. Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5):378–382.

Ulrich Gnewuch and Fabian Reinkemeier. 2025. Overcoming breakdowns in customer–chatbot interaction: Design and impact of collaborative repair strategies. MIS Quarterly, 50(2).

Google DeepMind. 2026. Gemma 4 model card.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations.

J. Richard Landis and Gary G. Koch. 1977. The Measurement of Observer Agreement for Categorical Data. 33(1):159–174.

Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019. An evaluation dataset for intent classification and out-ofscope prediction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 1311–1316, Hong Kong, China. Association for Computational Linguistics.

J. Lin. 1991. Divergence measures based on the shannon entropy. IEEE Transactions on Information Theory, 37(1):145–151.

Andrey Malinin and Mark Gales. 2021. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations.

John Murzaku, Zifan Liu, Md Mehrab Tanjim, Vaishnavi Muppala, Xiang Chen, and Yunyao Li. 2025. ECLAIR: Enhanced clarification for interactive responses. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 28864– 28870.

Anh Ha Ngo, Nicolas Rollet, Catherine Pelachaud, and Chloé Clavel. 2025. “Mm, Wat?” Detecting Otherinitiated Repair Requests in Dialogue. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 22926–22939.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Rishav Sahay, Lavanya Sita Tekumalla, Purav Aggarwal, Arihant Jain, and Anoop Saladi. 2025. ASK: Aspects and retrieval based hybrid clarification in task oriented dialogue systems. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 6: Industry Track), pages 881–895. Association for Computational Linguistics.

Lea Schönherr, Maximilian Golla, Thorsten Eisenhofer, Jan Wiele, Dorothea Kolossa, and Thorsten Holz. 2021. Exploring accidental triggers of smart speakers. Computer Speech & Language, 73:101328.

Katie Seaborn, Norihisa P. Miyake, Peter Pennefather, and Mihoko Otake-Matsuura. 2021. Voice in human–agent interaction: A survey. ACM Comput. Surv., 54(4).

Gunnar Stevens, Delong Korus-Du, Alexander Boden, Peter Tolmie, Dave Randall, and Md Shajalal. 2025. The art of repair in human-agent conversations: A taxonomy of repair strategies by users and LLMbased conversational agents. Preprint (under review). Research Square.

Gokhan Tur, Anoop Deoras, and Dilek Hakkani-Tür. 2014. Detecting out-of-domain utterances addressed to a virtual personal assistant. In Proceedings of Interspeech 2014, pages 283–287.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672.

Michael Zhang, W. Bradley Knox, and Eunsol Choi. 2025. Modeling future conversation turns to teach LLMs to ask clarifying questions. In The Thirteenth International Conference on Learning Representations.

Tong Zhang, Peixin Qin, Yang Deng, Chen Huang, Wenqiang Lei, Junhong Liu, Dingnan Jin, Hongru Liang, and Tat-Seng Chua. 2024. CLAMBER: A benchmark of identifying and clarifying ambiguous information needs in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10746–10766. Association for Computational Linguistics.

## A Appendix

The Appendix provides a hybrid fallback recovery pipeline diagram (Figure 3), inter-annotator agreement metrics (Table 4), a categorization of observed interaction failures across the pipeline (Table 5), an overview of class labels from the annotated dataset (Table 6), and finally, several figures detailing the prompts used for zero-shot and few-shot classification in the first and second stages of the pipeline (Figures 4, 5, 6, and 7).

![](images/281ab969305017506e0efea253cfc2ca96cef3dcd05b973ad9e8f242e6797684.jpg)  
Figure 3: Hybrid fallback recovery pipeline. The intent-based dialogue manager handles matched requests; lowconfidence inputs trigger a lightweight repair component consisting of a binary classification (Stage 1) and a multi-class classification for intended utterances (Stage 2). Depending on the type of the utterance, different recovery strategies are adopted.

<table><tr><td>Metric</td><td>Annotators</td><td>κ</td></tr><tr><td>Fleiss&#x27; κ</td><td>All (1, 2, 3)</td><td>0.7591</td></tr><tr><td rowspan="4">Pairwise Cohen&#x27;s κ</td><td>1 vs. 2</td><td>0.7619</td></tr><tr><td>1 vs. 3</td><td>0.7471</td></tr><tr><td>2 vs. 3</td><td>0.7685</td></tr><tr><td></td><td></td></tr></table>

Table 4: Inter-annotator agreement among three annotators, based on 350 items and 21 categories. Extrapolated to the whole dataset with Bootstrap 95% CI (10,000 resamples) yields $\kappa = 0 . 7 5 9 1 \pm 0 . 0 5 3 2$ [0.7034, 0.8099]. According to Landis and Koch (1977), this indicates substantial agreement among annotators.

<table><tr><td>Pipeline Layer</td><td>Failure Category</td><td>Selected Dataset Examples</td></tr><tr><td>Physical Layer</td><td>Microphone cut-offs</td><td>[Welche] Neue Nachrichten gibt es Erklär mir deine [Funktionen]</td></tr><tr><td>(Acoustic / Signal)</td><td>Ambient noise</td><td>... Packungsbeilage und fragen Sie Ihren Arzt oder Ihre Apotheke ... das Gespräch sei äußerst produktiv gewesen erzählt Trump</td></tr><tr><td rowspan="4">STT Layer (Phonetic / Lexical)</td><td>Wrong transcription</td><td>Habe ich 2 Tasten [Tassen] Kaffee getrunken Bitte nicht sehwören [stören] Dezimale [Maximale] Lautstärke</td></tr><tr><td>Cross-lingual homophones Out-of-vocabulary words</td><td>. . . und dann smooth [schmus] ich noch mal mit dir, ja. Pultmessen [Puls messen]</td></tr><tr><td></td><td>Aufhorn [Aufhören]</td></tr><tr><td>Profanity masking</td><td>Tagessau [Tagesschau] Das ist natürlich jetzt *******.</td></tr><tr><td>NLU Layer (Semantic)</td><td>Self-correction</td><td>Notfall eh nicht per Sprache machen, sondern erzähle mir einen Witz. .. . Nein nicht den Notfall einfach sagen das Ruf Jürgen an</td></tr><tr><td rowspan="3">Context Layer (Pragmatic)</td><td>Non-addressed speech Multi-speaker dialog</td><td>Bis wieviel Uhr kann ich anrufen? Sprich bitte, du musst die Frage stellen. Ja, ich muss das erstmal</td></tr><tr><td></td><td>lesen. Ach so lange drücken Notfall.</td></tr><tr><td>Out-of-scope requests</td><td>Assistent Spiel das Lied vom Roger Whittaker</td></tr></table>

Table 5: Overview of observed interaction failures across the voice assistant processing pipeline, populated with realworld dataset examples. Text enclosed in square brackets indicates author-provided corrections or reconstructions of the original automatic transcriptions, including inserted or corrected words.

```csv
Class Short Description
intended.clear_intent.available_skill.active_listener Starting chitchat conversation without a task-oriented goal
intended.clear_intent.available_skill.blood_pressure_log Logging blood pressure measurements
intended.clear_intent.available_skill.calling Initiating phone calls
intended.clear_intent.available_skill.change_volume Adjusting system sound volume
intended.clear_intent.available_skill.emergency Triggering emergency calls
intended.clear_intent.available_skill.general_reminder Creating general reminders
intended.clear_intent.available_skill.hydro Logging fluid intake
intended.clear_intent.available_skill.knowledge_search General knowledge question or factual information request
intended.clear_intent.available_skill.messaging Sending messages
intended.clear_intent.available_skill.news_search Retrieving news-related information
intended.clear_intent.available_skill.notebook Request to create a personal note
intended.clear_intent.available_skill.step_counter Request related to step count
intended.clear_intent.available_skill.tell_time Asking for current time
intended.clear_intent.available_skill.time_reminder Scheduling time-based reminders
intended.clear_intent.available_skill.vital_measure Request to measure vital signs such as heart frequency
intended.clear_intent.available_skill.weather Request for weather information or forecast
intended.clear_intent.available_skill.word_riddle Request to play a word riddle game
intended.clear_intent.available_skill.writing Request for a joke, story, poem, or other generated text
intended.clear_intent.unavailable_skill Valid intent, but unsupported by the system capabilities
intended.unclear_intent.multiple Utterance contains multiple competing intents
intended.unclear_intent.unspecific Intent is too vague or incomplete
unintended Accidental activation or non-directed/background speech
```  
Table 6: Class labels of fallback-triggering utterances with short descriptions.

You are a classifier for a German voice assistant.   
Task:   
Classify whether the user utterance is clearly intended for the assistant.   
Labels:   
1 = intended (direct command, request, or question directed at the assistant)   
0 = unintended (background speech, self-talk, conversation, filler, or noise)   
Output:   
Return exactly one label (0 or 1). If unsure, return 0.  
Figure 4: Zero-shot prompt for binary classification in the first stage.

![](images/4f04393a29edc89fa97ebdf4bf4bb904833d8e75d6715e5795efeaa2c6016d0a.jpg)  
Figure 5: Few-shot prompt for binary classification in the first stage.

![](images/6607925c6e9c65603b93fc0a8198dd21c50aed07d00b114aee9e49da2f7128aa.jpg)  
Figure 6: Zero-shot prompt for multi-class classification in the second stage.

![](images/b8a7d378278a87499fb5b30b2cede4e8f1168a839a0745d81be3004963f24de7.jpg)  
Figure 7: Retrieval-augmented prompt for multi-class classification in the second stage.