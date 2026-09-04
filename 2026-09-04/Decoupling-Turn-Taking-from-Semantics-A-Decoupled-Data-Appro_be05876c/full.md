# Decoupling Turn-Taking from Semantics: A Decoupled Data Approach for Finite-State-Machine-Based Full-Duplex Dialogue

Yihang Li, Chenhui Chu

Kyoto University

liyh@nlp.ist.i.kyoto-u.ac.jp, chu@i.kyoto-u.ac.jp

## Abstract

The Neural Finite State Machine (NFSM) framework offers a pragmatic path to fullduplex dialogue by serializing turn-taking control and response generation onto a single causal tape under the standard next-token prediction objective, thereby preserving semantic prowess at a low fine-tuning cost. However, its reliance on synthetic text data fundamentally limits turn-taking naturalness, as Large Language Models (LLMs) cannot faithfully simulate the fine-grained acoustic temporal dynamics of real human dialogues. In this work, we propose a decoupled data approach that learns turn-taking from real Human-Human (HH) spoken dialogues while shaping semantic behavior through configurable Human-Agent (HA) text dialogues. To operationalize this approach, we introduce a rule-based event-guided data transformation method that serializes HH spoken dialogues into FSM tapes by classifying turn-taking events and applying deterministic mapping rules, enabling scalable supervision without LLM-generated annotations. We further propose a Source-Aware Calibrated (SAC) Loss that jointly calibrates the long-tailed dis tribution of state transition tokens and channels each data source toward the capability it best supervises. Experiments show that our approach substantially improves turn-taking proficiency while recovering the foundation LLM’s semantic capability. Our code and model are available at https://github.com/Liyht/def-fsm.

## 1 Introduction

Recent advances in Large Language Models (LLMs) have greatly enhanced the semantic capabilities of voice assistants (Ji et al., 2024). However, their turn-taking naturalness remains constrained by rigid half-duplex turn-taking, in which each party must wait for the other to finish before speaking. To bridge the gap toward truly human-like interaction, the field is moving toward full-duplex dialogue (Chen and Yu, 2025), where an agent can listen and speak simultaneously and thereby handle interruptions, acoustic backchannels, and dynamic turn-taking in real time.

![](images/9168ecbd9faf83efbe7cdd6ccffacd6044807f0d5d919978c9b30f780f54365d.jpg)  
Figure 1: Comparison between the coupled and decoupled data approaches.

Current research on full-duplex dialogue falls into two primary paradigms: end-to-end and modular. While the end-to-end paradigm (Nguyen et al., 2023; Défossez et al., 2024; Veluri et al., 2024; Zhang et al., 2025; Lee et al., 2025) excels in finegrained turn-taking control and low latency by directly understanding and generating speech representations, enforcing strict cross-modal alignment often causes significant degradation in the underlying LLM’s semantic capabilities (Xiang et al., 2025; Chen et al., 2024). The modular paradigm instead equips an LLM with auxiliary state management via external controllers (Liao et al., 2025), classification heads (Wang et al., 2025; Chen et al., 2025), or in-vocabulary FSM control tokens (Wang et al., 2024b). Among these approaches, the NFSM (Wang et al., 2024b) occupies a uniquely minimal position: rather than introducing any auxiliary module, it serializes both state transitions and response content onto a single causal “tape,” handling fullduplex control entirely under the standard nexttoken prediction objective with no added parameters. Moreover, it enables the LLM to explicitly perceive and condition on the full history of prior turn-taking dynamics when making subsequent decisions. However, this minimalism places the full burden of turn-taking competence on the tape’s textual content—a burden that the original NFSM’s data approach cannot adequately meet.

Specifically, as illustrated in Figure 1 (left), NFSM constructs training tapes from LLMgenerated transcripts, attempting to distill complex turn-taking dynamics and rich semantics simultaneously. However, as foundation LLMs are predominantly pre-trained on structured non-overlapping text, they lack the inherent ability to simulate the fine-grained acoustic and temporal dynamics of real human interactions. Worse still, jointly controlling realistic turn-taking and rich semantics within a single generated dialogue compounds this difficulty, making such synthetic data fundamentally unreliable as supervision.

To address this challenge, we advocate a decoupled data approach, as shown in Figure 1 (right), that aligns each capability with its most suitable data source: fine-grained turn-taking dynamics are acquired from real HH spoken dialogue corpora, while readily available HA text dialogues serve as a flexible lever for shaping semantic behavior. The central challenge in operationalizing this approach lies in transforming real HH spoken dialogues into NFSM’s serialized tape as ground truth, in which state transition tokens must be inserted to faithfully reflect complex acoustic turn-taking dynamics. To this end, we propose an event-guided data transformation strategy that segments and classifies the dialogue timeline into discrete turn-taking events and subsequently applies deterministic mapping rules to serialize each segment into the tape. Our core insight is that state transition token insertion acts as an approximation of the turn-taking dynamics, while turn-taking dynamics equate to the temporal sequence of turn-taking events. Crucially, the entire pipeline is fully rule-based and free of LLM annotations, making it readily scalable to arbitrarily large HH corpora. On the HA side, the corpus can be freely substituted to target specific semantic objectives. In this work, we adopt a generic HA corpus to preserve the LLM’s intrinsic semantic capabilities, leaving capability-specific adaptation as a natural extension.

While this decoupled data approach provides clean supervision signals, jointly training on the resulting heterogeneous tapes introduces two challenges. First, state transition tokens exhibit an extremely long-tailed distribution, in which rare yet operationally critical state-switch tokens are easily marginalized under standard cross-entropy training. Second, the two data sources carry asymmetric supervisory value: HH tapes are authoritative for turn-taking but noisy for semantics, whereas HA tapes carry reliable semantics but trivial turn-taking structure. To address both issues within a single objective, we introduce the Source-Aware Calibrated (SAC) Loss, which combines logit adjustment over the restricted vocabulary of state transition tokens with a source-aware weighting scheme that channels each data source toward the capability it best supervises.

Experiments under strict token-volume constraints on HH spoken dialogue corpora and the VoiceBench dataset show that our model achieves substantially better turn-taking proficiency and semantic capability than the conventional syntheticdata baseline, with SAC further mitigating the severe class imbalance. When scaled to a larger training corpus constructed by proportionally upsampling each data source to preserve the mixture ratio, our approach successfully recovers the intrinsic semantic capability of the foundation LLM.

In summary, our contributions are threefold:

• We propose a decoupled data approach for FSM-based full-duplex dialogue, in which turn-taking is learned from real HH spoken dialogues and semantic capability is independently shaped through configurable HA data.

• We design a fully rule-based event-guided transformation that serializes complex acoustic interactions into causal FSM tapes, enabling scalable construction of real HH dialogue data.

• We introduce the SAC Loss, which calibrates the long-tailed distribution of state transition tokens and specializes optimization across data sources.

## 2 Related Work

Turn-Taking Event Classification. Turn-taking event classification is a labeling scheme that constructs ground-truth event labels along the dialogue timeline, supporting downstream tasks such as event prediction and analysis (Patamia et al., 2025; Castillo-López et al., 2025; Heldner and Edlund, 2010; Arora et al., 2025; Threlkeld et al., 2022). Existing schemes can be organized by the temporal unit on which events are defined. Chunkbased methods segment the timeline into equallength frames and assign per-frame labels (Heldner and Edlund, 2010; Ekstedt and Skantze, 2022; Arora et al., 2025), offering fine temporal resolution suited to frame-based neural architectures. Inter-Pausal Unit (IPU)-based methods instead define events at IPU boundaries delimited by silence thresholds (Ekstedt and Skantze, 2020; Wang et al., 2024a), yielding linguistically coherent units that align cleanly with textual chunk boundaries. We adopt the IPU-based formulation, as this property makes rule-based serialization onto the FSM tape straightforward.

Data Strategies for Full-Duplex Supervision. Full-duplex supervision data are constructed along two distinct paths. The first relies on synthetic generation: an LLM is prompted to produce dialogues annotated with designated turn-taking control tokens (Wang et al., 2024b), or dialogues are assembled through rule-based procedures such as inserting user interruptions at predefined positions (Xie and Wu, 2024). The fundamental difficulty is that LLMs are trained predominantly on structured non-overlapping text and therefore lack the capacity to simulate the fine-grained turn-taking dynamics of real human conversation. The second path draws on real spoken dialogue corpora. End-to-end models leverage such corpora through implicit supervision, directly learning from raw interleaved audio tokens without explicit state labels (Nguyen et al., 2023; Défossez et al., 2024). Modular approaches instead apply voice activity detection and heuristic rules to derive state labels from real audio, training a dedicated state classifier on fixed acoustic chunks (Liao et al., 2025; Chen et al., 2025). Our setting poses a qualitatively different challenge: rather than classifying states over fixed acoustic chunks, we must insert discrete state transition tokens into an interleaved text token stream, which is a form of supervision that cannot be derived from voice activity alone and demands explicit turn-taking event identification.

## 3 Data Transformation

Building on the NFSM framework, we serialize two complementary corpora into a single causal FSM tape format: real HH spoken dialogues for fine-grained turn-taking dynamics, and HA text dialogues adapted to a spoken style for preserving semantic capability. We first briefly review NFSM, then describe our transformation pipelines for both corpora.

## 3.1 Preliminary: The NFSM Framework

NFSM (Wang et al., 2024b) casts dialogue turntaking as an explicit state transition problem managed directly by an LLM operating a two-state machine consisting of a SPEAK state and a LISTEN state. The transitions are realized through vocabulary tokens: [C.SPEAK] and [C.LISTEN] for continuing the current state, and [S.SPEAK] and [S.LISTEN] for switching to the opposite state. Both state transitions and response generation are serialized onto a single causal tape under the standard next-token prediction paradigm. Crucially, each autoregressive decoding step on this tape is triggered by three types of events ranked in decreasing priority: the generation of speech-driving tokens, the arrival of new perception chunks, or the completion of motor processing. This unified sequence asynchronously synchronizes the perception and motor modules, allowing a standard textbased LLM to control real-time full-duplex interaction without architectural modifications.

## 3.2 Event-Guided Human-Human Data Transformation

Building upon our decoupled data approach, the primary objective of HH data transformation is to construct a rigorous ground-truth tape that strictly mirrors human turn-taking behavior. Our core insight is that state transition token insertion acts as an approximation of the turn-taking dynamics, while the turn-taking dynamics is equivalent to the temporal sequence of turn-taking events. Consequently, we propose an event-guided transformation strategy. Our conversion strategy is guided by two principles: (1) categorizing the timeline into discrete turn-taking events to direct token insertion, and (2) aligning with the actual inference-time conditions the FSM will encounter.

Asymmetric Preprocessing. The two channels of each HH dialogue are designated as the user channel and the agent channel, and processed asymmetrically to mirror NFSM’s actual inference-time conditions as shown in Figure 2. The user channel is transcribed by the same perception module deployed in the FSM, producing fine-grained timestamped text chunks together with silence tokens of fixed duration, thereby exposing the LLM to realistic ASR granularities and recognition error patterns. The agent channel, in contrast, uses fine-grained timestamped ground-truth transcripts, ensuring the LLM learns to generate clean text rather than imitate ASR artifacts. Chunks are kept as fine-grained as possible—ideally word-level—to enable precise event segmentation downstream. Additional preprocessing details are provided in Appendix B.2.

![](images/4e253c0792e719823f421f7d1c691380c9e021ca90cb31448c193a5848ad6074.jpg)  
Figure 2: Example of turn-taking event classification and event-guided tape serialization. Green, blue, and orange blocks denote user content, agent content, and state transition tokens (shown in abbreviated form, e.g., [C.L] for [C.LISTEN]), respectively. The <user> role-prefix token preceding each user block is omitted for brevity.

Turn-Taking Event Classification. This stage partitions the continuous dialogue timeline and assigns each segment to one of seven turn-taking event types, as illustrated in Figure 2 (top). We first merge adjacent chunks within each channel into IPUs using a predefined pause threshold; the union of IPU boundaries across both channels then yields a discrete timeline. Following Arora et al. (2025), we classify segments as:

• Single-party speech. Turn Change (T) if the floor holder differs from the previous segment; Continuation (C) otherwise.

• Silence. Pause (P) within a single speaker’s turn; Gap (G) between alternating turns.

• Overlapping speech. Backchannel (BC) if the utterance matches a predefined lexical list and falls below a duration threshold (Ekstedt and Skantze, 2020; Wang et al., 2024a; Arora et al., 2025); Floor-Taking Interruption (FTI) if the interrupter successfully takes the floor; or Butting-in (BI) if the attempt fails.

Event-Guided Tape Serialization. This stage constructs the serialized tape representation for each classified segment. First, we split [S.LISTEN] into [S.LISTEN.N] (Natural) and [S.LISTEN.I] (Interrupt), granting the LLM finergrained control over the motor module to explicitly signal whether the ongoing TTS should complete naturally or be immediately truncated. Next, we assign the text of each fine-grained chunk to its corresponding segment based on start timestamp. To formulate the mapping scheme, we define the initiator of a segment as the participant whose action triggers the transition from the preceding segment (specifically, the preceding floor holder for gaps). Combining the 7 event types with these 2 initiator types yields 14 deterministic mapping rules (Table 1; see Figure 2 bottom for an example). Finally, four operational heuristics govern the compilation of these rules into a continuous tape: (1) the governing state transition token immediately follows the first chunk of a segment, while subsequent chunks maintain the state via continuation tokens; (2) during overlaps, agent chunks precede user ASR chunks, reflecting the causal latency that an agent generates text prior to audio synthesis, whereas the perception module emits text post-speech; (3) silence tokens are preserved exclusively within LISTEN intervals; and (4) consecutive state transition tokens at segment boundaries are collapsed into a single token via the rules in Appendix A.3.

Post-Processing and Tape Validation. To optimize TTS quality at inference, we merge and resegment continuous agent text into clause-level granularity using a punctuation model (Guhr et al., 2021), which is the finest granularity that preserves prosodic quality. This resegmentation is applied independently to each contiguous span of agent text with only state transition tokens between them, with [C.SPEAK] tokens inserted at clause boundaries. We then validate every resulting sequence against the FSM’s transition constraints (e.g., the SPEAK state emits only [S.LISTEN.\*] or [C.SPEAK]) to ensure the completeness of the entire pipeline.

<table><tr><td>Event</td><td>Initiator = User</td><td>Initiator = Agent</td></tr><tr><td>T</td><td>[S.LISTEN.N] &lt;User&gt;</td><td>[S.SPEAK] &lt;Agent&gt;</td></tr><tr><td>C</td><td>[C.LISTEN] &lt;User&gt;</td><td>[C.SPEAK] &lt;Agent&gt;</td></tr><tr><td>P</td><td>[C.LISTEN] &lt;SIL&gt;</td><td></td></tr><tr><td>G</td><td>[C.LISTEN] &lt;SIL&gt;</td><td>[S.LISTEN.N] &lt;SIL&gt;</td></tr><tr><td>BC</td><td>[C.SPEAK] &lt;Agent&gt; &lt;User&gt; [C.SPEAK]</td><td>[S.SPEAK] &lt;Agent&gt; &lt;User&gt; [S.LISTEN.N]</td></tr><tr><td>FTI</td><td>[C.SPEAK] &lt;Agent&gt; &lt;User&gt;[S.LISTEN.I]</td><td>[S.SPEAK] &lt;Agent&gt; &lt;User&gt; [C.SPEAK]</td></tr><tr><td>BI</td><td>[C.SPEAK] &lt;Agent&gt; &lt;User&gt; [C.SPEAK]</td><td>[S.SPEAK] &lt;Agent&gt; &lt;User&gt; [S.LISTEN.I]</td></tr></table>

Table 1: Mapping rules for event-guided tape serialization across 14 combinations of turn-taking events and initiators. <User> and <Agent> denote textual content from the user and agent channels; <SIL> represents the silence token.

## 3.3 Human-Agent Data Transformation

To preserve the LLM’s intrinsic semantic abilities, we leverage broad-domain HA text dialogues. However, as textual HA datasets are primarily written-style and thus unsuitable for a spoken dialogue agent, we convert them into the FSM tape format through a three-stage pipeline: (1) rule-based filtering, (2) LLM-driven stylistic rewriting, and (3) final filtering and tape serialization.

We first apply rule-based cleaning to the raw text dialogues, which strips HTML markup, validates role alternation, and removes turns dominated by code, mathematical formulas, or excessively long contexts. We then prompt an LLM with a two-phase prompt (Appendix A.5) that first judges suitability for spoken delivery, then rewrites the surviving dialogues into a natural spoken style. A final filtering pass discards turns exceeding spokenutterance length thresholds or containing residual Markdown artifacts, yielding a distilled set of highquality dialogues.

Finally, we segment user and agent utterances into clauses with the same punctuation model as in HH transformation, then insert [S.SPEAK] and [S.LISTEN.N] between alternating utterances and [C.LISTEN]/[C.SPEAK] between clauses within the same utterance. This establishes a trivial yet schema-compliant turn-taking structure, allowing the HA data to focus its supervisory signal on semantic content.

## 4 Model

## 4.1 FSM Modules

In the absence of a publicly available NFSM implementation, we build the entire system from scratch, integrating intrinsically multilingual perception, cognitive, and motor modules optimized for real-time interaction. To examine the effect of perception granularity, we evaluate two streaming ASR alternatives, both built on Whisper-Turbo (Radford et al., 2022) to isolate granularity from backbone: a word-level streaming ASR, Simul-Streaming (Machácek and Polákˇ , 2025), and an IPU-level streaming ASR coupling Silero VAD (Team, 2024) with Faster-Whisper.<sup>1</sup> This contrast allows us to explore the fundamental trade-off between the precise temporal resolution afforded by fine-grained streaming and the semantic coherence required for aggressively fragmented inputs. For the motor module, we adopt Kokoro TTS,<sup>2</sup> whose Time-to-First-Audio on clause-level inputs outperforms many streaming alternatives without sacrificing prosodic quality. Other proposed techniques for FSM are available in Appendix E.

## 4.2 Source-Aware Calibrated Loss

Optimizing the FSM model presents two challenges: mitigating the severe class imbalance inherent in turn-taking behaviors and jointly optimizing two distinct capabilities. To this end, we propose the SAC Loss, a dual-purpose objective designed to calibrate the long-tail token distribution and decouple learning targets across diverse data sources.

During fine-tuning, we adopt a selective masking strategy. Specifically, we mask the prompts and user text, restricting the loss computation exclusively to the agent response tokens and state transition tokens. Let $I = \{ 1 , \ldots , N \}$ denote the target token indices for a training instance. We partition I along two orthogonal dimensions: by token type into response tokens (R) and state transition tokens (T); and by data source into Human-Human (HH) and Human-Agent (HA) tokens.

To achieve distributional calibration, we tackle the extreme class imbalance among state transition tokens. Continuation tokens representing ongoing listening or speaking appear exponentially more frequently than discrete state-switching tokens (see Appendix A.2 for the distribution). Employing a standard cross-entropy objective naturally marginalizes these low-probability yet operationally critical transitions. Therefore, we incorporate Logit Adjustment (Menon et al., 2021) specifically for the state transition tokens to rectify this skewed prior, while retaining the standard cross-entropy loss for the response tokens. The percomponent losses for an input x and target label y are defined as follows:

$$
\ell ^ { \mathrm { { C E } } } ( x , y ) = - \log \frac { \exp ( f _ { y } ( x ) ) } { \sum _ { v } \exp ( f _ { v } ( x ) ) }
$$

$$
\ell ^ { \mathrm { L A } } ( x , y ) = - \log \frac { \exp \bigl ( f _ { y } ( x ) + \tau \log \pi _ { y } \bigr ) } { \sum _ { v } \exp \bigl ( f _ { v } ( x ) + \tau \log \pi _ { v } \bigr ) }
$$

where $f ( \cdot )$ denotes the model logits, $\pi _ { y }$ is the prior probability of class $y ,$ and τ is the adjustment temperature. The prior π is computed exclusively over the restricted state-transition vocabulary to avoid overweighting. Accordingly, the loss $\ell _ { i }$ for the i-th token is formulated as:

$$
\ell _ { i } = \left\{ \begin{array} { l l } { \ell ^ { \mathrm { C E } } ( x _ { i } , y _ { i } ) , } & { \mathrm { i f } i \in R } \\ { \ell ^ { \mathrm { L A } } ( x _ { i } , y _ { i } ) , } & { \mathrm { i f } i \in T } \end{array} \right.
$$

To achieve source-aware specialization, we introduce a weighting mechanism that dynamically adjusts the importance of each token based on its type and origin. The overall training loss $\mathcal { L }$ is computed as a weighted average over all unmasked tokens:

$$
\mathcal { L } = \frac { \sum _ { i \in I } w _ { i } \ell _ { i } } { \sum _ { i \in I } w _ { i } }
$$

where the weight $w _ { i }$ assigned to the i-th token is defined as:

$$
w _ { i } = \left\{ { \begin{array} { l l } { \alpha , } & { { \mathrm { i f ~ } } i \in ( H A \cap R ) \cup ( H H \cap T ) } \\ { 1 - \alpha , } & { { \mathrm { i f ~ } } i \in ( H A \cap T ) \cup ( H H \cap R ) } \end{array} } \right.
$$

in which $\alpha \in ( 0 . 5 , 1 ]$ is a hyperparameter controlling the degree of source emphasis. By assigning a higher weight α to the subsets $( H H \cap T )$ and $( H A \cap R )$ , the SAC Loss explicitly decouples the optimization trajectories: it forces the model to learn realistic turn-taking dynamics predominantly from human-human interactions, while refining its semantic capabilities primarily through human-agent data.

## 5 Experiments

## 5.1 Settings

For HH spoken dialogues, we adopt the Switchboard (Godfrey et al., 1992) and Fisher (Cieri et al., 2004) corpora. For the HA text dialogues, we adopt the ShareGPT dataset<sup>3</sup> and utilize Qwen3- 32B (Yang et al., 2025) for stylistic rewriting. As the foundation model, we adopt the open-source lightweight Qwen3-4B (model scale analysis in Appendix C.1) and extend its tokenizer with state transition tokens, the silence token <SIL>, and the role-prefix token <user>. For the data mixture, we set HH:HA token-volume ratio as 1:1, with Fisher:Switchboard mixed at 4:1 within HH. For the SAC loss, we set $\tau = 1$ and $\alpha = 0 . 6$ . Hyperparameter analyses are provided in Appendix C.1, while full training details, dataset splits, and First Token Emission Delay (FTED) latency are in Appendix B.1 and D.

Considering the original training dataset of NFSM is not publicly available, we reproduce their training data to establish a baseline representative of the conventional synthetic data approach. Specifically, GPT-4-Turbo (OpenAI et al., 2024) generates 1,500 synthetic dialogues using the published NFSM prompts, which we serialize via the same procedure as our HA transformation, with one exception—incomplete-utterance markers (<NOT\_FINISHED>, trailing ellipses) are mapped to [S.LISTEN.I]. We then fine-tune a separate Qwen3-4B model on this synthetic tape, which serves as our primary baseline. More details are available in Appendix A.1.

Our evaluation focuses on turn-taking proficiency and semantic capability. Turn-taking is measured on HH corpora as the ground truth for natural dynamics: at each teacher-forced prediction step, we treat the output as an independent binary classification for each state transition token type, compute per-type F1 within each evaluation set, average across types within each set, then average across Switchboard and Fisher to obtain the reported F1. For semantic evaluation, we adopt VoiceBench (Chen et al., 2024) following Xiang et al. (2025), excluding IFEval to better match a purely acoustic system and averaging over SD-QA, MMSU, OpenBookQA, and AdvBench subsets as a result. All test inputs are prepended with the same FSM prompt. For fine-tuned models, evaluation audio is transcribed by their respective perception modules and interleaved with state transition tokens. For the zero-shot foundation model upper bound, audio is transcribed by Whisper-Turbo into continuous plain text, preventing degradation from context fragmentation.

<table><tr><td rowspan="2">Model</td><td colspan="2">Faster-Whisper (IPU-levei)</td><td colspan="2">SimulStreaming (word-level)</td></tr><tr><td>F1↑</td><td>VB↑</td><td>F1↑</td><td>VB↑</td></tr><tr><td>NFSM</td><td>0.3436</td><td>53.57</td><td>0.3025</td><td>44.24</td></tr><tr><td>Ours</td><td>0.6498</td><td>62.14</td><td>0.6404</td><td>54.80</td></tr><tr><td>∆ vs. NFSM</td><td>+0.3062</td><td>+8.57</td><td>+0.3379</td><td>+10.56</td></tr></table>

Table 2: Overall performance across method and perception module, where VB denotes VoiceBench.
<table><tr><td>Configuration</td><td>F1↑</td><td>VB↑</td></tr><tr><td>Ours</td><td>0.6498</td><td>62.14</td></tr><tr><td>Ablation on Loss Function w/o SAC loss</td><td>0.6105</td><td>62.19</td></tr><tr><td>Ablation on Training Data (w/o SAC loss)</td><td></td><td></td></tr><tr><td>HH only</td><td>0.6182</td><td>18.30</td></tr><tr><td>HA only</td><td>0.4170</td><td>63.15</td></tr><tr><td>Synthetic only (NFSM)</td><td>0.3436</td><td>53.57</td></tr></table>

Table 3: Ablation study on the SAC loss and training data composition under Faster-Whisper perception.

## 5.2 Main Results

To ensure fair comparison, we control the total training token volume for our models to match the NFSM baseline. As shown in Table 2, our model significantly outperforms NFSM under both perception configurations. Comparing the two variants of our model, the Faster-Whisper variant achieves turn-taking performance comparable to the SimulStreaming variant while attaining substantially higher semantic capability. This gap stems from the hyper-granular nature of SimulStreaming outputs: user textual content is heavily fragmented by frequently interleaved state transition tokens, challenging the LLM’s semantic comprehension (details in Appendix A.2). These findings reveal a critical architectural trade-off in perception module design: finer granularity offers more frequent opportunities to manage interruptions but risks degrading semantic prowess.

## 5.3 Ablation Study

For the ablation study, we first examine the effectiveness of the SAC loss. As shown in Table 3, integrating the SAC loss consistently improves turn-taking proficiency while preserving semantic capability at a comparable level. Its central advantage, however, lies in mitigating the severe class imbalance among state transition tokens. Figure 3 presents the confusion matrices for state transition and response tokens on Switchboard with the Faster-Whisper-based model. Specifically, each decoding step is formulated as an independent classification task and percentages are normalized against the ground-truth labels. The results clearly show that the SAC loss yields notable gains in the prediction accuracy of state-switch tokens, which are operationally critical yet rare under the standard cross-entropy objective.

<table><tr><td>Configuration</td><td>F1↑</td><td>VB↑</td></tr><tr><td>Semantic Upper Bound Qwen3-4B (zero-shot)</td><td>0.1200</td><td>64.93</td></tr><tr><td>End-to-End Reference Moshi</td><td></td><td>27.36</td></tr><tr><td>Our FSM Model</td><td></td><td></td></tr><tr><td>Token-matched subset</td><td>0.6498</td><td>62.14</td></tr><tr><td>Proportionate upsampling</td><td>0.6539</td><td>64.84</td></tr></table>

Table 4: Performance of our FSM model scaled to the proportionate upsampling mixture, alongside the zeroshot semantic upper bound and end-to-end reference.

Furthermore, we evaluate the impact of data mixture strategies. Under identical loss configurations, the model trained solely on synthetic data is substantially outperformed by the model trained on the combined HH and HA corpora. For turn-taking, this gap arises from two factors: current synthetic generation strategies fail to adequately simulate the fine-grained acoustic dynamics of real human dialogues, and synthetic transcripts cannot replicate the specific temporal granularities and recognition artifacts of the perception module. For semantic preservation, authentic HA dialogues prove more effective than synthetic transcripts that attempt to distill both capabilities simultaneously. Finally, while training exclusively on either HH or HA enhances one capability in isolation, neither corpus alone improves both. We observe highly consistent trends when substituting Faster-Whisper with SimulStreaming as the perception module, with detailed results provided in Appendix C.2.

Beyond the turn-taking F1, we further assess interruption handling under both machine-interruptsuser and user-interrupts-machine settings, following the protocol of NFSM. Our model substantially outperforms NFSM in both directions (Appendix C.3).

![](images/67b1f759004c3b6f931fe89a50e0579b839f1d95a94297f59cf5859504f4a9ed.jpg)

![](images/220bfbea0dfd2b21a4a175dbe6aebb83712e6cc184e5a41449fc784256b85446.jpg)

Figure 3: Confusion matrices for state transition and response tokens on Switchboard with the Faster-Whisper-based model: standard cross-entropy (left) versus SAC loss (right).
<table><tr><td></td><td colspan="8">Full-Duplex-Bench v1.0</td><td colspan="4">Full-Duplex-Bench v1.5</td></tr><tr><td></td><td colspan="2">Pause Handling</td><td colspan="3">Backchannel</td><td>Smooth Turn</td><td colspan="2">User Interruption</td><td colspan="4">Overlap Handling</td></tr><tr><td>Data Model</td><td>Synthetic TOR↓</td><td>Candor TOR↓</td><td>TOR↓</td><td>ICC Freq↑</td><td>JSD↓</td><td>Candor TOR↑</td><td>Synthetic TOR↑</td><td>GPT-4o↑</td><td>U-INTR RESP↑</td><td>U-BC RSM↑</td><td>T-OTH RSM↑</td><td>BKG RSM↑</td></tr><tr><td>Moshi</td><td>0.985</td><td>0.980</td><td>1.000</td><td>0.001</td><td>0.957</td><td>0.941</td><td>1.000</td><td>0.765</td><td>0.50</td><td>0.06</td><td>0.19</td><td>0.07</td></tr><tr><td>Freeze-Omni</td><td>0.642</td><td>0.481</td><td>0.636</td><td>0.001</td><td>0.997</td><td>0.336</td><td>0.867</td><td>3.615</td><td>0.72</td><td>0.80</td><td>0.25</td><td>0.25</td></tr><tr><td>Gemini Live</td><td>0.255</td><td>0.310</td><td>0.091</td><td>0.012</td><td>0.896</td><td>0.655</td><td>0.891</td><td>3.376</td><td>0.33</td><td>0.93</td><td>0.99</td><td>0.30</td></tr><tr><td>Ours</td><td>0.825</td><td>0.653</td><td>0.127</td><td>0.119</td><td>0.764</td><td>0.891</td><td>0.875</td><td>3.589</td><td>0.69</td><td>0.98</td><td>0.44</td><td>0.36</td></tr></table>

Table 5: Full-Duplex-Bench results for our proportionate upsampling model, restricted to the systems common to both benchmarks. U-INTR: user interruption; U-BC: user backchannel; T-OTH: talking to others; BKG: background speech; RESP: respond rate; RSM: resume rate.

## 5.4 Scaling to the Proportionate Upsampling Mixture

While the preceding experiments operated under a strict token-volume constraint (referred to as the token-matched subset), we now investigate model performance when scaling training to its maximum extent. To strictly preserve the established blending ratio across the constituent corpora, we construct a proportionally upsampled mixture by repeatedly oversampling the datasets to match the volume of the largest split (detailed in Appendix A.4). As shown in Table 4, scaling yields further improvements in both turn-taking proficiency and semantic capability. To rigorously assess the extent of semantic recovery, we introduce a strong upper bound: the original Qwen3-4B in zero-shot setting. Remarkably, despite being trained to handle complex full-duplex turn-taking on a fragmented FSM tape, our Faster-Whisper-based model recovers the intrinsic semantic capability of the underlying LLM, closing the gap to within 0.09 points on VoiceBench. As an additional reference, the end-toend Moshi attains 27.36 on the same VoiceBench subset as reported by Chen et al. (2024), which is lower than our FSM model and consistent with prior observations that end-to-end full-duplex architectures often trade off semantic capability.

We further evaluate the proportionate upsampling model on Full-Duplex-Bench (Lin et al., $2 0 2 5 \mathrm { b } , \mathrm { a } )$ , an architecture-agnostic benchmark independent of both our token scheme and training distribution (Table 5; complete results in Appendix C.4). The most substantial gains are observed in backchanneling: our model achieves the best human-aligned timing and highest frequency with a low takeover rate on v1.0, while attaining the highest resume rate on v1.5 user backchannel. We attribute this to the event-guided tape and the SAC Loss. The former supervises what kind of overlap occurred rather than merely whether speech is present, while the latter prevents the rare switchtype tokens from being washed out by dominant response tokens. The model remains highly competitive across most remaining dimensions, ranking first or second in smooth turn-taking, userinterruption semantic quality, and all four v1.5 overlap scenarios. A notable exception is pause handling, where our takeover rates trail Freeze-Omni and Gemini Live, indicating a more aggressive strategy in reclaiming the floor. However, this behavior reflects an inherent turn-taking trade-off rather than a strict deficiency. For instance, the conservative Freeze-Omni excels at pause handling but compromises smooth turn-taking. Overall, our approach outperforms every baseline across the majority of the twelve metrics (Moshi 10/12, Freeze-Omni 8/12, Gemini Live 7/12).

## 6 Conclusion

In this work, we presented a decoupled data approach for FSM-based full-duplex dialogue, in which turn-taking is acquired from real HH spoken dialogues while semantic behavior is shaped through configurable HA text dialogues. We operationalized this approach with a fully rule-based event-guided transformation that serializes HH spoken dialogues into causal FSM tapes without LLMgenerated annotations, and introduced the SAC Loss to jointly calibrate the long-tailed distribution of state transition tokens and channel each data source toward the capability it best supervises. Experiments show that our approach substantially improves turn-taking proficiency over the synthetic-data baseline while recovering the foundation LLM’s semantic capability to the zero-shot upper bound, demonstrating that decoupling data sources by capability provides a scalable recipe for FSM-based full-duplex dialogue.

## Limitations

Our study has four primary limitations. First, serializing two parallel audio channels into a single causal tape inevitably compresses fine-grained continuous-time information into a discrete token order, leading to information loss despite our enforcement of temporal causality. Second, paralinguistic information—prosody, emotion, and voice quality—is not representable on the text-mediated tape, and the expressiveness of the rendered speech is bounded by the off-the-shelf TTS module, which is inherent to the original NFSM formulation. In this work, we focus on turn-taking naturalness and leave paralinguistic cues and a perceptual study to future. Third, our HH corpora retain the agent channel’s original GT utterance content as spoken by ordinary human participants, which does not necessarily align with the response style expected of a voice assistant. Refining the semantic content of the agent channel could yield a higher-quality training signal, but risks disturbing the temporal alignment between channels and thus the turn-taking dynamics we aim to preserve. We therefore leave the agent channel unedited and view its controlled refinement as a promising direction for future work. Fourth, our corpora cover only two-party dialogue and a question-answering assistant role. Extending the HH data to multi-party settings and the HA data to role-conditioned dialogues (Mitra et al., 2026) is left to future work.

## Acknowledgements

This work was supported by JST SPRING (Grant Number JPMJSP2110) and JSPS (Grant Number JP23K28144). We thank Professor Tatsuya Kawahara for his assistance in obtaining access to the Fisher corpus.

## References

Siddhant Arora, Zhiyun Lu, Chung-Cheng Chiu, Ruoming Pang, and Shinji Watanabe. 2025. Talking turns: Benchmarking audio foundation models on turntaking dynamics. In The Thirteenth International Conference on Learning Representations.

Galo Castillo-López, Gael de Chalendar, and Nasredine Semmar. 2025. A survey of recent advances on turn-taking modeling in spoken dialogue systems. In Proceedings ofthe 15th International Workshop on Spoken Dialogue Systems Technology, pages 254– 271, Bilbao, Spain. Association for Computational Linguistics.

Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, Yabin Li, Xiang Lv, Jiaqing Liu, Haoneng Luo, Bin Ma, Chongjia Ni, Xian Shi, Jialong Tang, Hui Wang, and 17 others. 2025. Minmo: A multimodal large language model for seamless voice interaction. Preprint, arXiv:2501.06282.

Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T. Tan, and Haizhou Li. 2024. Voicebench: Benchmarking llm-based voice assistants. Preprint, arXiv:2410.17196.

Yuxuan Chen and Haoyuan Yu. 2025. From turn-taking to synchronous dialogue: A survey of full-duplex spoken language models. Preprint, arXiv:2509.14515.

Christopher Cieri, David Miller, and Kevin Walker. 2004. The fisher corpus: a resource for the next generations of speech-to-text. In Proceedings ofthe Fourth International Conference on Language Resources and Evaluation (LREC’04), Lisbon, Portugal. European Language Resources Association (ELRA).

Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. 2024. Moshi: a speech-text foundation model for real-time dialogue. Preprint, arXiv:2410.00037.

Erik Ekstedt and Gabriel Skantze. 2020. TurnGPT: a transformer-based language model for predicting turn-taking in spoken dialog. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 2981–2990, Online. Association for Computational Linguistics.

Erik Ekstedt and Gabriel Skantze. 2022. Voice activity projection: Self-supervised learning of turn-taking events. In Interspeech.

John J. Godfrey, Edward C. Holliman, and Jane Mc-Daniel. 1992. Switchboard: telephone speech corpus for research and development. In Proceedings ofthe 1992 IEEE International Conference on Acoustics, Speech and Signal Processing - Volume 1, ICASSP’92, page 517–520, USA. IEEE Computer Society.

Oliver Guhr, Anne-Kathrin Schumann, Frank Bahrmann, and Hans Joachim Böhme. 2021. Fullstop: Multilingual deep models for punctuation prediction.

Mattias Heldner and Jens Edlund. 2010. Pauses, gaps and overlaps in conversations. J. Phonetics, 38:555– 568.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Shengpeng Ji, Yifu Chen, Minghui Fang, Jialong Zuo, Jingyu Lu, Hanting Wang, Ziyue Jiang, Long Zhou, Shujie Liu, Xize Cheng, Xiaoda Yang, Zehan Wang, Qian Yang, Jian Li, Yidi Jiang, Jingzhen He, Yunfei Chu, Jin Xu, and Zhou Zhao. 2024. Wavchat: A survey of spoken dialogue models. Preprint, arXiv:2411.13577.

Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. 2024. Benchmarking cognitive biases in large language models as evaluators. In Findings of the Association for Computational Linguistics: ACL 2024, pages 517–545, Bangkok, Thailand. Association for Computational Linguistics.

Sehun Lee, Kang-wook Kim, and Gunhee Kim. 2025. Behavior-SD: Behaviorally aware spoken dialogue generation with large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 9574–9593, Albuquerque, New Mexico. Association for Computational Linguistics.

Borui Liao, Yulong Xu, Jiao Ou, Kaiyuan Yang, Weihua Jian, Pengfei Wan, and Di Zhang. 2025. Flexduo: A pluggable system for enabling full-duplex capabilities in speech dialogue systems. Preprint, arXiv:2502.13472.

Guan-Ting Lin, Shih-Yun Shan Kuan, Qirui Wang, Jiachen Lian, Tingle Li, and Hung-yi Lee. 2025a. Full-duplex-bench v1. 5: Evaluating overlap handling for full-duplex speech models. arXiv preprint arXiv:2507.23159.

Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Gopala Anumanchipalli, Alexander H Liu, and Hung-yi Lee. 2025b. Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities. arXiv preprint arXiv:2503.04721.

Dominik Machácek and Peter Polák. 2025.ˇ Simultaneous translation with offline speech and LLM models in CUNI submission to IWSLT 2025. In Proceedings ofthe 22nd International Conference on Spoken Language Translation (IWSLT 2025), pages 389–398, Vienna, Austria (in-person and online). Association for Computational Linguistics.

Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. 2021. Long-tail learning via logit adjustment. In International Conference on Learning Representations.

Soumyajit Mitra, Prabhat Pandey, Abhinav Jain, Shanmukha Sahith, and K V Vijay Girish. 2026. Adaptive turn-taking for real-time multi-party voice agents. Preprint, arXiv:2606.13544.

Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoît Sagot, Abdelrahman Mohamed, and Emmanuel Dupoux. 2023. Generative spoken dialogue language modeling. Transactions of the Association for Computational Linguistics, 11:250–266.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, and 262 others. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Rutherford Agbeshi Patamia, Ha Pham Thien Dinh, Ming Liu, and Akansel Cosgun. 2025. Turn-taking modelling in conversational systems: A review of recent advances. Technologies, 13(12).

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. 2021. Mauve: measuring the gap between neural text and human text using divergence frontiers. In Proceedings of the 35th International

Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA. Curran Associates Inc.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. Preprint, arXiv:2212.04356.

Silero Team. 2024. Silero vad: pre-trained enterprisegrade voice activity detector (vad), number detector and language classifier. https://github.com/ snakers4/silero-vad.

Charles Threlkeld, Muhammad Umair, and Jp de Ruiter. 2022. Using transition duration to improve turntaking in conversational agents. In Proceedings ofthe 23rd Annual Meeting ofthe Special Interest Group on Discourse and Dialogue, pages 193–203, Edinburgh, UK. Association for Computational Linguistics.

Bandhav Veluri, Benjamin N Peloquin, Bokai Yu, Hongyu Gong, and Shyamnath Gollakota. 2024. Beyond turn-based interfaces: Synchronous LLMs as full-duplex dialogue agents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21390–21402, Miami, Florida, USA. Association for Computational Linguistics.

Jinhan Wang, Long Chen, Aparna Khare, Anirudh Raju, Pranav Dheram, Di He, Minhua Wu, Andreas Stolcke, and Venkatesh Ravichandran. 2024a. Turn-taking and backchannel prediction with acoustic and large language model fusion. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12121–12125. IEEE.

Peng Wang, Songshuo Lu, Yaohua Tang, Sijie Yan, Wei Xia, and Yuanjun Xiong. 2024b. A full-duplex speech dialogue scheme based on large language model. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA. Curran Associates Inc.

Xiong Wang, Yangze Li, Chaoyou Fu, , Yike Zhang, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, and Long Ma. 2025. Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen llm. ICML.

Bajian Xiang, Shuaijiang Zhao, Tingwei Guo, and Wei Zou. 2025. Understanding the modality gap: An empirical study on the speech-text alignment mechanism of large speech language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5187–5202, Suzhou, China. Association for Computational Linguistics.

Zhifei Xie and Changqiao Wu. 2024. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. Preprint, arXiv:2410.11190.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang, Siqi Zheng, Jiaqing Liu, Hai Yu, Chao-Hong Tan, Zhihao Du, and ShiLiang Zhang. 2025. OmniFlatten: An end-to-end GPT model for seamless voice conversation. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14570– 14580, Vienna, Austria. Association for Computational Linguistics.

## A Data Construction Details

## A.1 NFSM Baseline Reproduction Details

Because the original NFSM training corpus is not publicly available, our synthetic-data baseline is reconstructed from the specifications given in the NFSM paper. For full transparency, Table 6 documents every parameter of this reconstruction, indicating for each whether it is specified in the original paper and, where it is not, the setting we adopt. We reproduce the original work as faithfully as possible, keeping its most critical components identical to the original, including the generation LLM and the full prompt template. A few minor parameters such as the unfilled placeholders in the published prompt are not disclosed, so an exact match is impossible and we adopt reasonable settings.

## A.2 State Transition Token Statistics

Table 7 reports the token type distribution of the Switchboard training tapes under the two perception modules, placed side by side with that of the reconstructed NFSM synthetic tapes for reference. The distributions exhibit three notable patterns.

First, state transition tokens follow a markedly long-tailed distribution. Within the restricted state transition vocabulary, continuation tokens jointly account for 72.78% and 80.45% of all state transition tokens under the Faster-Whisper and SimulStreaming perception modules respectively, whereas the operationally critical state-switch tokens occupy a significantly smaller proportion. This skew motivates our use of logit adjustment over the state transition tokens in the SAC loss.

Second, the choice of perception module reshapes the distribution in a predictable direction. Switching from the IPU-level Faster-Whisper to the word-level SimulStreaming raises the share of [C.LISTEN] from 8.75% to 13.77%. The increased interleaving of [C.LISTEN] within user content quantitatively reflects the fragmentation phenomenon and offers a partial explanation for the larger semantic gap observed under SimulStreaming.

<table><tr><td>Parameter</td><td>Specified in NFSM</td><td>Our reproduction</td></tr><tr><td>Generation LLM</td><td>Yes (gpt-4-turbo-2024-04-09)</td><td>Identical</td></tr><tr><td>Decoding settings</td><td>No</td><td>Default sampling</td></tr><tr><td># dialogues</td><td>Yes (1,500 series)</td><td>Identical</td></tr><tr><td>Prompt template</td><td>Yes (their Appendix A)</td><td>Identical</td></tr><tr><td>{num_rounds}</td><td>No (placeholder only)</td><td>Uniform over 9–11</td></tr><tr><td>Scenario-to-round assignment</td><td>No (placeholder only)</td><td>Random round with no collision</td></tr><tr><td>{response_word_count}</td><td>No (placeholder only)</td><td>Uniform over 100–150</td></tr><tr><td>{interrupted_response_word_count}</td><td>No (placeholder only)</td><td>Uniform over 20–50</td></tr><tr><td>Topic pool</td><td>Partial (“hundreds&quot;, &quot;random&quot;)</td><td>110 topics generated with gpt-4-turbo -2024-04-09</td></tr><tr><td>Dialogue serialization</td><td>No</td><td>As described in Section 5.1</td></tr></table>

Table 6: Reconstruction parameters of the NFSM synthetic-data baseline. For each parameter we indicate whether it is specified in the original NFSM paper and the setting adopted in our reproduction.

<table><tr><td rowspan="2">Token</td><td colspan="3">Real HH</td></tr><tr><td>Synthetic</td><td>Faster- Whisper</td><td>Simul- Streaming</td></tr><tr><td>[C.LISTEN]</td><td>0.91%</td><td>8.75%</td><td>13.77%</td></tr><tr><td>[C.SPEAK]</td><td>8.31%</td><td>10.63%</td><td>10.42%</td></tr><tr><td>[S.LISTEN.I]</td><td>0.17%</td><td>0.77%</td><td>0.53%</td></tr><tr><td>[S.LISTEN.N]</td><td>0.99%</td><td>2.83%</td><td>2.38%</td></tr><tr><td>[S.SPEAK]</td><td>1.24%</td><td>3.66%</td><td>2.97%</td></tr><tr><td>Response</td><td>88.38%</td><td>73.37%</td><td>69.93%</td></tr><tr><td>Total</td><td>100.00%</td><td>100.00%</td><td>100.00%</td></tr></table>

Table 7: Token-type ratio across the NFSM synthetic tapes and the Switchboard training tapes under two perception modules.

Third, the synthetic tapes exhibit a markedly more extreme imbalance than the real HH data: response tokens account for 88.38% of the tape, leaving only 11.62% for all state-transition tokens, of which the operationally critical state-switch tokens ([S.LISTEN.I], [S.LISTEN.N], and [S.SPEAK]) constitute merely 2.40%. In contrast, the real HH tapes allocate a substantially larger share to stateswitch tokens of 7.26% under Faster-Whisper and 5.88% under SimulStreaming. This gap reflects that LLM-generated transcripts, being predominantly structured and non-overlapping, seldom reproduce the frequent floor changes, backchannels, and interruptions of natural conversation.

## A.3 Merging Rules at Segment Boundaries

Concatenating two adjacent segments during serialization may produce two consecutive state transition tokens at their boundary, which must be collapsed into a single token to align with the rule of FSM.

Rather than enumerating 14 × 14 event pairs or 5 × 5 token pairs, we observe that the merged token is fully determined by the FSM state immediately before the first token fires and immediately after the second token fires, reducing the analysis to four state-pair cases:

• (LISTEN, LISTEN) → [C.LISTEN]: the FSM remains in LISTEN.

• (SPEAK, SPEAK) → [C.SPEAK]: the FSM remains in SPEAK.

• (LISTEN, SPEAK) → [S.SPEAK]: a switch from listening to speaking.

• (SPEAK, LISTEN) → [S.LISTEN.\*]: a switch from speaking to listening, where the N/I subtype is inherited from whichever token in the pair is [S.LISTEN.\*], with [S.LISTEN.I] taking priority when both are present.

The (SPEAK, LISTEN) case warrants further justification, as it is the only case requiring subtype inheritance. The first token in this case must be [C.SPEAK] or [S.LISTEN.\*], and the second must be [C.LISTEN] or [S.LISTEN.N]. Among these, the pair [C.SPEAK] + [C.LISTEN] appears superficially well-formed but never arises in valid serializations, so the merge can always recover an N/I subtype from one side of the pair. To see why, note that a segment ending in [C.SPEAK] can only originate from BC or BI initiated by the user, or FTI initiated by the agent (Table 1); in all three cases the floor holder at the segment’s end is the agent. Conversely, a segment beginning with [C.LISTEN] can only originate from C, P, or G initiated by the user, all of which require the user to be the floor holder at the segment’s start (recall that the initiator of a G segment is the preceding floor holder). These two requirements are mutually inconsistent, ruling out the pair.

<table><tr><td rowspan="3">Configuration</td><td colspan="4">Train (k)</td><td colspan="3">Validation (k)</td><td colspan="2">Test (k)</td></tr><tr><td>FSH</td><td>SWBD</td><td>SGPT</td><td>SYN</td><td>FSH</td><td>SWBD</td><td>SYN</td><td>FSH</td><td>SWBD</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td>1,344</td><td></td><td></td><td>165</td><td>5,600</td><td>444</td></tr><tr><td>Ours (Token-matched)</td><td>538</td><td>135</td><td>672</td><td></td><td>691</td><td>236</td><td></td><td>5,600</td><td>444</td></tr><tr><td>Ours (Scaled Mixture)</td><td>21,792</td><td> $^ { 5 , 4 4 8 ^ { * } }$ </td><td>27,240†</td><td></td><td>691</td><td>236</td><td>一</td><td>5,600</td><td>444</td></tr></table>

Table 8: Dataset statistics across different configurations, measured in thousands (k) of tokens. Abbreviations used are FSH (Fisher), SWBD (Switchboard), SGPT (ShareGPT), and SYN (Synthetic). For the scaled mixture, we apply proportional oversampling to maintain the established data distribution. The superscript symbols indicate datasets that are repeatedly sampled from their natural volume: (<sup>∗</sup>) base size 1,551k × 3.51, and (<sup>†</sup>) base size 2,966k × 9.18.

Besides, as a related adjustment, when a T immediately follows a G, its leading [S.LISTEN.N] is demoted to [C.LISTEN], as the preceding G has already transitioned the FSM into LISTEN.

## A.4 Proportionate Upsampling Mixture

For the scaling experiments, we construct a proportionally upsampled mixture to maximize the training signal while strictly maintaining the established data distribution. Since the constituent corpora possess disparate natural volumes, datasets with smaller token counts are oversampled to adhere to the predefined ratios. Detailed statistics for each configuration, including base volumes, oversampling multipliers, and final token counts, are provided in Table 8. This approach ensures that the model benefits from the full extent of available data without biasing the optimization toward any single data source.

## A.5 Prompts

The prompt for filtering and stylistic rewriting HA data is shown as follows:

## Prompt

You are an expert linguistic editor   
specializing in dialogue style transfer.   
Your task is twofold: first, EVALUATE   
whether the input text is suitable for   
conversion into a casual spoken dialogue;   
second, ONLY IF suitable, EXECUTE the   
conversion.

\### PHASE 1: EVALUATION (Strict Filtering) Before converting, analyze the input content. You must return an empty list ‘[]‘ for the ’conversations’ field if the input is not suitable for spoken dialogue, for example but not constrain to the following ’UNSUITABLE’ categories:

1. \*\*Code & Technical Data\*\*: Contains programming code, stack traces, logs, JSON/XML, SQL queries, or shell commands.

2. \*\*Strict structured text\*\*: Contains mathematical formulas (LaTeX), chemical equations, or rigid data tables.

3. \*\*Document text\*\*: Contains document   
input which is impossible to happen in   
dialogue.

4. \*\*Non-Dialogue / Fragments\*\*: The input   
is just a list of keywords, a disjointed   
sentence fragment without context, or   
incoherent noise.

5. \*\*Competitor Identity / Specific AI   
Models\*\*: The text mentions ’OpenAI’,   
’ChatGPT’, or the model refers to itself as   
such.

```markdown
### PHASE 2: CONVERSION (Only if Phase 1
passes)
```

If the content is suitable, convert it   
into a natural spoken-style conversation.   
Guidelines for Spoken Style:

- Strictly REMOVE all Markdown syntax (e.g.,   
\*, #, -). Do not use bullet points or   
numbered lists; use connecting words like   
’First’, ’Next’, ’Finally’ instead.

- Simplify complex sentence structures and   
content to be more conversational.

- Maintain the original intent and core   
information.

\### CONSTRAINTS:

1. Keep the original language of each   
utterance unchanged.

2. Keep the original ’id’ strictly   
unchanged.

3. Ensure the speaker roles and the number   
of turns remain exactly the same.

4. \*\*CRITICAL\*\*: If the content matches   
the UNSUITABLE criteria, the output must   
be ‘[]‘.

5. Ensure the output follows the strict   
JSON schema provided.

<|im\_start|>user   
You are a helpful voice assistant.   
You will work as a Finite State Machine   
to handle state transitions between LISTEN   
and SPEAK states.   
State-transition tokens: ${ } ^ { \prime \prime } [ { \mathsf { S } } . { \mathsf { S P E A K } } ] ^ { \prime \prime }$ means   
"switch to SPEAK"; "[S.LISTEN.INTERRUPT]"   
means "switch to LISTEN because of being   
interrupted"; "[S.LISTEN.NATURAL]" means   
"switch to LISTEN because of natural end";   
"[C.SPEAK]" means "continue to SPEAK";   
"[C.LISTEN]" means "continue to LISTEN".   
For content between State-transition   
tokens, the user’s content are prefixed   
with "<user>" and your own content has no   
prefix. "<SIL>" represents a silence of   
{{SILENCE\_TOKEN\_DUR}} seconds.   
Please generate a tape fragment of the   
Finite State Machine.<|im\_end|>   
<|im\_start|>assistant   
<think>   
</think>

## B Additional Experimental Settings

This section supplements the general settings in the main text with details on training hyperparameters, dataset partitioning, and the temporal thresholds used in the FSM and our event-guided transformation pipeline.

## B.1 Training Hyperparameters and Dataset Partitioning

Fine-tuning is conducted on four NVIDIA RTX A6000 GPUs with batch size 256, optimized with AdamW (weight decay 0.01) under a linear schedule with warmup, peak learning rate $1 . 0 \times 1 0 ^ { - 5 }$ and warmup ratio 0.03. Each instance has a context length of 1024 tokens. We train for up to 300 iterations with early stopping on validation loss for the token-matched setting, and extend this budget to 700 iterations with the same early-stopping criterion for the proportionate upsampling mixture to accommodate its substantially larger training corpus. For dataset partitioning, we apply distinct split ratios tailored to the scale of each corpus. For both the ShareGPT and Switchboard datasets, we partition the data into training, validation, and test subsets at a dialogue ratio of 7:1:2. Because the Fisher corpus is substantially larger, we adjust its dialogue ratio to 7.75:0.25:2 to prevent the validation set from becoming unnecessarily large for frequent validation.

## B.2 Preprocessing and Transformation Thresholds

Several thresholds govern data preprocessing and the event-guided transformation pipeline. For HH dialogue filtering, we discard dialogues whose Word Error Rate exceeds 0.3, removing both broken audio and cross-channel leakage where one speaker’s audio bleeds into the other channel. For the event-guided transformation, the IPU pause threshold is set to 32 milliseconds, each silence token represents 0.64 seconds of silence, and the maximum utterance duration for heuristic backchannel identification is capped at 1 second.

## C Additional Experimental Results

## C.1 Analysis on Data Mixture, Loss, and Model Scale

This subsection reports analysis along three design axes: the mixture ratios of training data, the SAC loss hyperparameter α, and the size of the base model. All experiments use the Faster-Whisperbased configuration and strictly constrain the total training token volume to match the NFSM baseline.

Data Mixture Ratios. Figure 4 reports the effect of dataset mixture ratios under $\alpha = 0 . 5$ . The results demonstrate a clear trade-off: increasing the proportion of HH data consistently enhances turn-taking proficiency, whereas increasing the proportion of HA data improves semantic capability. To avoid biasing the model toward either capability, we adopt a 1:1 HH:HA ratio in all main experiments. Adjusting the internal Fisherto-Switchboard ratio shows no significant trend, so we empirically set it to 4:1.

Impact of α in SAC Loss. Figure 5 reports performance across different values of α. As α increases, the VoiceBench score improves while the turn-taking F1 score decreases. To diagnose the F1 decline, we additionally plot the average recall and precision of state transition tokens on HH data. Interestingly, as α increases, recall actually rises and stabilizes, whereas precision drops significantly. This indicates that the SAC loss inherently aligns better with recall. This behavior is consistent with prior observations that standard language model pre-training via maximum likelihood estimation inherently favors recall over precision, assigning excessive probability mass to an unreliable tail of tokens (Holtzman et al., 2020; Pillutla et al., 2021). Based on this analysis, we adopt $\alpha = 0 . 6$ as the default in our main experiments to balance semantic preservation against turn-taking precision.

![](images/ab4472c3e3d774b66055387b653958e427ce657c213ca999b33c3501689f1e3f.jpg)

![](images/423118e0163d2f9688edb13c8e3e3f4e5ad94e8481fd387484155b07746cae79.jpg)  
Figure 4: Impact of dataset mixture ratios on model performance. Left: turn-taking proficiency measured by F1 score. Right: semantic capability measured by VoiceBench.

![](images/f80f47ab96b0966b235eea8a36d04105e9b1cb8ec8e8532f5f5cbe20dae87267.jpg)  
Figure 5: Impact of α in the SAC loss.

![](images/f863dc76a2aba273ef74fdef618e04a26ebf8b18267c5259b81330b962ccbf3e.jpg)  
Figure 6: Impact of base model size.

Impact of Base Model Size. Fixing $\alpha = 0 . 6 ,$ we further investigate the effect of scaling the base model, as shown in Figure 6. The VoiceBench score consistently improves with larger models, but the turn-taking F1 score peaks at the 4B scale. We hypothesize that larger language models may require substantially more effort to adequately fit the highly specialized pattern of the FSM tape.

<table><tr><td>Configuration</td><td>F1↑</td><td>VB↑</td></tr><tr><td>Ours</td><td>0.6404</td><td>54.80</td></tr><tr><td>Ablation on Loss Function w/o SAC loss</td><td>0.6234</td><td>54.49</td></tr><tr><td>Ablation on Training Data (w/o SAC loss)</td><td></td><td></td></tr><tr><td>HH only</td><td>0.6376</td><td>19.71</td></tr><tr><td>HA only</td><td>0.3886</td><td>54.55</td></tr><tr><td>Synthetic only (NFSM)</td><td>0.3025</td><td>44.24</td></tr></table>

Table 9: Ablation study on the SAC loss and training data composition under SimulStreaming perception.

<table><tr><td>Model</td><td>Perception</td><td>MiU F1 ↑</td><td>UiM PRR ↑</td></tr><tr><td>NFSM</td><td></td><td>0.7203</td><td>0.7260</td></tr><tr><td rowspan="2">Ours</td><td>Faster-Whisper</td><td>0.7698</td><td>0.8037</td></tr><tr><td>SimulStreaming</td><td>0.7786</td><td>0.8367</td></tr></table>

Table 10: Experiments of bidirectional interruption.

## C.2 Ablation Study under SimulStreaming Perception

To complement the ablation study in the main text, Table 9 reports the corresponding results under SimulStreaming perception. The trends are highly consistent with those of the Faster-Whisper-based model: the SAC loss effectively preserves turntaking performance, and neither HH nor HA data alone is sufficient to jointly optimize semantic and turn-taking capabilities.

## C.3 Bidirectional Interruption Evaluation

We further evaluate the system’s interruption handling capability by strictly replicating the protocol of NFSM.

For the Machine-interrupts-User (MiU) setting, we use GPT-4-Turbo to synthesize 600 dialogues in which the user’s final statement is injected with a deliberate commonsense error. We decode the FSM step by step through this final statement and extract the response generated at the precise location where the model first proactively emits [S.SPEAK] instead of [C.LISTEN]. State transition tokens are then stripped to reconstruct a standard dialogue. To prevent egocentric bias (Koo et al., 2024), we use Qwen3.5-27B as the evaluator, strictly following the assessment prompts of the original NFSM paper to compute the Proper Interruption Rate, the position-specific metrics ir<sub>mid</sub>, ir<sub>end</sub>, and MIR, and finally an aggregate F1 score.

<table><tr><td rowspan="2">Dimension Data</td><td colspan="2">Pause Handling</td><td colspan="3">Backchannel</td><td colspan="2">Smooth Turn Taking</td><td colspan="3">User Interruption</td></tr><tr><td>Synthetic TOR↓</td><td>Candor TOR↓</td><td>TOR↓</td><td>ICC Freq ↑</td><td>JSD ↓</td><td></td><td>Candor Latency ↓</td><td>TOR ↑</td><td>Synthetic</td><td>Latency ↓</td></tr><tr><td>dGSLM</td><td>0.934</td><td>0.935</td><td>0.691</td><td>0.015</td><td>0.934</td><td>TOR↑ 0.975</td><td>0.352</td><td>0.917</td><td>GPT-40 ↑ 0.201</td><td>2.531</td></tr><tr><td>Moshi</td><td>0.985</td><td>0.980</td><td>1.000</td><td>0.001</td><td>0.957</td><td>0.941</td><td>0.265</td><td>1.000</td><td>0.765</td><td>0.257</td></tr><tr><td>Freeze-Omni</td><td>0.642</td><td>0.481</td><td>0.636</td><td>0.001</td><td>0.997</td><td>0.336</td><td>0.953</td><td>0.867</td><td>3.615</td><td>1.409</td></tr><tr><td>Gemini Live</td><td>0.255</td><td>0.310</td><td>0.091</td><td>0.012</td><td>0.896</td><td>0.655</td><td>1.301</td><td>0.891</td><td>3.376</td><td>1.183</td></tr><tr><td>Ours</td><td>0.825</td><td>0.653</td><td>0.127</td><td>0.119</td><td>0.764</td><td>0.891</td><td>0.181</td><td>0.875</td><td>3.589</td><td>0.579</td></tr></table>

Table 11: Complete results on Full-Duplex-Bench v1.0 across different conversational dimensions, where latency is presented in seconds.

For the User-interrupts-Machine (UiM) setting, we use GPT-4-Turbo to generate 600 dialogues evenly distributed across four interruption categories: denial, affirmation, environmental noises, and topic shifting. We insert [S.SPEAK] at the end of the user’s interruption to elicit an immediate FSM response. After removing state transition tokens from the dialogue, the average Proper Response Rate (PRR) is evaluated by the same Qwen3.5-27B judge using the original evaluation prompts.

As shown in Table 10, both perception-module variants of our model achieve substantially higher MiU F1 and UiM PRR scores than the NFSM baseline, demonstrating improvements in both turntaking dynamics and semantic appropriateness within the specific sub-scenario of interruptions.

## C.4 Full Results on Full-Duplex-Bench

We report the complete Full-Duplex-Bench results underlying the merged Table 5 in the main text. Results of v1.0 are shown in Table 11, while results of v1.5 are shown in Table 12. The two versions probe complementary capabilities. v1.0 evaluates turn-taking over the course of a dialogue, covering whether the model holds back at intra-turn pauses, backchannels at human-like moments, takes the floor promptly at genuine turn boundaries, and reacts to interruptions. v1.5 instead isolates overlap handling while the model is speaking, testing whether it distinguishes overlapping speech that warrants yielding the floor from speech that does not. Both evaluate our Faster-Whisper-based proportionate upsampling model under the official protocols. All other numbers are quoted from the official report. Specially, the low Smooth Turn Taking latency is partly affected by the official metric, which by design clamps to zero any onset preceding the annotated turn boundary (e.g., backchannel).

## D End-to-End Latency Breakdown

Table 13 reports the end-to-end inference latency of the Faster-Whisper-based FSM system, measured by FTED. Latency is measured on a test set of 100 questions, constructed by randomly sampling 25 questions from each of the four VoiceBench subsets used in our semantic evaluation. The breakdown across the perception, cognitive, and motor modules demonstrates that the system maintains real-time responsiveness.

## E Design of FSM

We build the FSM from scratch and introduce several techniques to improve its performance and robustness.

Decoupled Temperatures for Transition and Response Tokens. Although emitted from a shared autoregressive stream, state-transition and response tokens serve distinct purposes, necessitating decoupled decoding strategies. Transition tokens act as discrete control signals where sampling noise causes turn-taking failures; hence, we decode them greedily. Conversely, response tokens require standard sampling temperatures to maintain lexical diversity and generation quality. Operationally, each generation step initiates with a greedy single-token probe. If a transition token is emitted, the step commits it and terminates. Otherwise, this token serves as the clause prefix, and the remainder is sampled at the response temperature. Transition tokens proposed mid-clause by the sampler are discarded, deferring the transition decision to the next probe.

<table><tr><td>Scenario</td><td>Class / Metric</td><td>Freeze-Omni</td><td>Moshi</td><td>Gemini</td><td>Sonic</td><td>GPT-40</td><td>Ours</td></tr><tr><td rowspan="6">USER_INTR</td><td>RESPOND ↑</td><td>0.72</td><td>0.50</td><td>0.33</td><td>0.24</td><td>0.78</td><td>0.69</td></tr><tr><td>RESUME↓</td><td>0.12</td><td>0.26</td><td>0.55</td><td>0.71</td><td>0.10</td><td>0.17</td></tr><tr><td>UNCERTAIN↓</td><td>0.03</td><td>0.00</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.05</td></tr><tr><td>UNKNOWN↓</td><td>0.13</td><td>0.25</td><td>0.10</td><td>0.04</td><td>0.12</td><td>0.09</td></tr><tr><td>STOP (s) ↓</td><td>1.42</td><td>1.16</td><td>2.20</td><td>2.25</td><td>0.23</td><td>1.60</td></tr><tr><td>RESP (s) ↓</td><td>1.35</td><td>1.47</td><td>2.62</td><td>2.75</td><td>1.50</td><td>1.12</td></tr><tr><td rowspan="6">USER_BACKCH</td><td>RESPOND↓</td><td>0.07</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.03</td><td>0.00</td></tr><tr><td>RESUME↑</td><td>0.80</td><td>0.06</td><td>0.93</td><td>0.98</td><td>0.70</td><td>0.98</td></tr><tr><td>UNCERTAIN↓</td><td>0.02</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.01</td><td>0.00</td></tr><tr><td>UNKNOWN↓</td><td>0.11</td><td>0.92</td><td>0.04</td><td>0.02</td><td>0.25</td><td>0.02</td></tr><tr><td>STOP (s) ↑</td><td>0.66</td><td>0.42</td><td>0.66</td><td>0.64</td><td>0.21</td><td>0.69</td></tr><tr><td>RESP (s) ↓</td><td>2.16</td><td>3.00</td><td>2.45</td><td>1.45</td><td>1.32</td><td>1.99</td></tr><tr><td rowspan="6">TALKING_OTHER</td><td>RESPOND↓</td><td>0.58</td><td>0.20</td><td>0.00</td><td>0.10</td><td>0.91</td><td>0.40</td></tr><tr><td>RESUME↑</td><td>0.25</td><td>0.19</td><td>0.99</td><td>0.90</td><td>0.02</td><td>0.44</td></tr><tr><td>UNCERTAIN ↑</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.06</td></tr><tr><td>UNKNOWN↓</td><td>0.15</td><td>0.59</td><td>0.01</td><td>0.00</td><td>0.06</td><td>0.10</td></tr><tr><td>STOP (s) ↑</td><td>1.39</td><td>0.87</td><td>1.69</td><td>1.77</td><td>0.18</td><td>1.35</td></tr><tr><td>RESP (s) ↓</td><td>1.32</td><td>2.38</td><td>1.78</td><td>2.04</td><td>1.16</td><td>1.29</td></tr><tr><td rowspan="6">BKG_SPEECH</td><td>RESPOND↓</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RESUME ↑</td><td>0.63</td><td>0.21</td><td>0.70</td><td>0.01</td><td>0.93</td><td>0.44</td></tr><tr><td>UNCERTAIN↑</td><td>0.25</td><td>0.07</td><td>0.30</td><td>0.98</td><td>0.04</td><td>0.36</td></tr><tr><td>UNKNOWN↓</td><td>0.01 0.11</td><td>0.01 0.71</td><td>0.00 0.00</td><td>0.00 0.01</td><td>0.00 0.03</td><td>0.13 0.07</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>STOP (s) ↑ RESP (s) ↓</td><td>0.98 1.60</td><td>0.54 1.62</td><td>0.95 2.38</td><td>1.05 2.76</td><td>0.18 1.26</td><td>1.16 1.20</td></tr></table>

Table 12: Complete results on Full-Duplex-Bench v1.5. Behavioral response distribution across the four overlap scenarios, with average stop and response latencies (s). Bold Class/Metric are the desired behavior for each scenario.

<table><tr><td>Component</td><td>Mean (s)</td><td>P50 (s)</td><td>P90 (s)</td></tr><tr><td>Perception</td><td>0.1689</td><td>0.1226</td><td>0.3347</td></tr><tr><td>Cognitive</td><td>0.2786</td><td>0.2669</td><td>0.4357</td></tr><tr><td>Motor</td><td>0.1481</td><td>0.1371</td><td>0.2024</td></tr><tr><td>Total</td><td>0.5955</td><td>0.5312</td><td>0.9105</td></tr></table>

Table 13: End-to-end FTED latency of the FSM system under Faster-Whisper perception, broken down by module.

Constrained Decoding over the Transition Vocabulary. During decoding, we mask the logits of all state-transition tokens deemed illegal under the current FSM state. Although the fine-tuned model rarely produces illegal transitions, this hard masking provides deterministic guarantees against structural failures rather than relying on probabilistic safety.

Bounded Look-ahead. Generating the next clause only post-playback simplifies interruption handling but incurs latency gaps. Conversely, eager generation until turn completion eliminates gaps but wastes computation upon user intervention. We balance this trade-off via a bounded look-ahead policy: the controller generates one additional clause as long as fewer than K clauses are in-flight (i.e., dispatched to the TTS module but not yet fully played), idling once this bound is reached.

Interruption Handling. Handling user interruptions gracefully presents a critical challenge. To support natural backchannels—where pausing before resumption sounds robotic—the agent continues speaking while the transition decision is computed. As a price, this incurs a temporal inconsistency that the tape before and after the decision diverge. In practice, we retrieve the currently playing clause from tape containing look-ahead tail and split it at the exact playback timestamp into a preclause (already played) and a post-clause (pending). We then execute one constrained decoding step using the pre-clause appended with the user utterance as the prefix. The resulting transition token dictates the tape rewriting strategy:

• [C.SPEAK]: The split clause is replaced by • [S.LISTEN.I]: The split clause is replaced by pre-clause + interruption + [S.LISTEN.I]. TTS is truncated immediately, and the lookahead tail is discarded.

pre-clause + interruption + [C.SPEAK] +   
post-clause. The look-ahead tail is retained.

• [S.LISTEN.N]: The split clause is replaced by pre-clause + post-clause + interruption + [S.LISTEN.N]. The current clause is allowed to finish naturally, and the remaining lookahead tail is discarded.

Hallucination Filtering for the ASR Module. Whisper-based ASR is prone to hallucinations, which can spuriously trigger the interruption mechanism. We mitigate this using a soft lexical filter: segments matching a predefined list of frequent hallucinations are discarded if their average token log-probability falls below −0.3. Otherwise, they are retained to preserve genuine user utterances.