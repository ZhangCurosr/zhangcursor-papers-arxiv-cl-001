# ConversationalVoice Full-Duplex Speech Data from Real Conversations through Source-Faithful Reconstruction and Conversation-Grounded Expansion

Richard Yucheng He, Baodong Cao, Chen Xu, Yihang Liu, Tairan Chen

AveraLabs

{richard, baodong.cao, chen.xu, robbie.liu, terrence.chen}@averalabs.com

## Abstract

Full-duplex speech models require training data that preserves turn-taking, overlap, interruption, and backchannel behavior, yet these signals are entangled across speakers in noisy realworld recordings. We present ConversationalVoice, a pipeline that converts real two-speaker excerpts into three complementary training-data artifacts. (1) Separation recovers speakerspecific tracks with stable speaker assignments, a canonical transcript, and naturally observed interaction timing. (2) Reconstruction generates speech in matched voices from a fixed source transcript, reconstructs the source turn order, pauses, and overlaps, and adds word-level alignment and delivery instructions. (3) Expansion generates new dialogue constrained by the source context, speakers, and observed interaction pattern. Automatic speaker-verification metrics remain strong across stages, with same-speaker similarity of 0.983–0.991 and positive discrimination margins of 0.199–0.209. Predicted speech quality (NISQA MOS) is 3.56 for separation, 4.41 for reconstruction, and 4.61 for expansion. A Gemini-based automatic evaluator assigns expansion mean scores of 4.94/5 for contextual coherence and 4.80/5 for dialogue naturalness. Expansion and reconstruction exhibit broadly similar interaction profiles; expansion’s turn, overlap-event, backchannel, and interruption rates are 4.6%, 8.0%, 13.2%, and 16.0% lower, respectively. We evaluate data properties only; downstream gains in full-duplex model training remain for future work.

Code repository:

https://github.com/avera-labs/ConversationalVoice Keywords: full-duplex speech models; speech data pipelines; speech reconstruction; conversational speech generation; conversational data augmentation

## 1 Introduction

Training full-duplex speech models involves modeling conversations as two synchronized speaker streams rather than as isolated utterances. Accordingly, training data for these models should preserve words, speaker identity, turn transitions, silence, overlap, feedback, and nonverbal behavior. Full-duplex speech models increasingly represent these temporal signals explicitly, while related benchmarks systematically evaluate them [1–3]; yet real conversations usually arrive as noisy monaural recordings, while prompt-generated dialogue can lose the timing and interaction patterns of actual exchanges.

We present ConversationalVoice, an end-to-end pipeline that transforms real-world two-speaker audio excerpts into three complementary forms of full-duplex training data. Qualityverified separation combines diarization, dialogue separation, stable speaker-slot assignment, signal and identity checks, and ASR to recover speaker-specific tracks with a canonical wordlevel transcript. Its contribution is to make separation a validated substrate with fixed speaker-slot assignments for subsequent generation rather than the final dataset.

Source-faithful reconstruction regenerates the recovered exchange while fixing its words and speaker identities. Stable speaker references and utterance-local acoustic cues guide synthesis; forced alignment and conversation-level scheduling then rebuild word timing, turns, pauses, and overlaps around the generated durations.

Conversation-grounded expansion generates new dialogue under constraints from the source transcript, audio, speaker profiles, and observed interaction. It is designed to add conversational coverage while explicitly planning dialogue, backchannels, paralinguistic events, and sequential or overlapping placement.

Both generated stages produce paired, time-aligned singlespeaker tracks with speaker-attributed transcripts, word-level timestamps, audio tags and delivery instructions. The three stages therefore recover, reconstruct, and extend real conversational evidence in a shared training representation.

Specifically, this paper makes four contributions:

1. We operationalize separation as a quality-verified data stage that maintains stable speaker-slot assignments and supplies canonical transcription for downstream transformations.

2. We introduce reconstruction that preserves source words and speaker-to-track assignments while regenerating clean tracks and adapting conversational timing to synthesized speech.

3. We introduce expansion that creates new interactions grounded in real speakers, context, and explicitly represented turn-taking and paralinguistic behavior.

![](images/d572d02fe72d6993fe89268f871ee944fcb06406836ede669f50d7e3c23f470f.jpg)  
Figure 1: ConversationalVoice produces three complementary views of a validated conversation. Separation recovers speakers; reconstruction regenerates the source exchange; expansion creates a grounded continuation. Reconstruction and expansion additionally expose word timing and delivery-instruction metadata.

4. We define a shared training artifact with aligned speaker tracks, word timing, audio tags, delivery instructions.

The automatic evaluation characterizes shared output quality, reconstruction fidelity, and expansion interaction and content quality. It tests whether the three artifacts retain the properties needed for their complementary roles; downstream model gains remain outside the scope of this work.

## 2 Related Work

## 2.1 Full-duplex spoken dialogue modeling

Spoken dialogue models have moved from turn-concatenated utterances toward synchronized streams. The dialogue generative spoken language model (dGSLM) learns from two-channel conversational speech and generates speech conditionally across channels [1]. Moshi models user and system audio in parallel and uses time-aligned text as an intermediate prediction signal, illustrating the value of preserving both timing and lexical structure [2]. Full-Duplex-Bench organizes evaluation around turn-taking abilities such as user pause, backchannel, interruption, and simultaneous speech [3]. Together, these systems motivate training examples that retain independent channels and fine temporal structure rather than collapsing a conversation into alternating text turns.

## 2.2 Recovering conversation from in-the-wild audio

Speech separation estimates multiple source signals from a mix ture. SepFormer demonstrated the efectiveness of attentionbased temporal modeling for clean separation benchmarks [4]. In-the-wild dialogue adds degradations not well represented by clean synthetic mixtures. DialogueSidon addresses this setting through joint restoration and separation of degraded monaural two-speaker audio [5]. DuplexChat builds on this capability in a data pipeline that filters public podcast audio, identifies two-speaker clips, and creates separated full-duplex tracks at large scale [6].

DuplexChat is the closest prior system to the first stage of ConversationalVoice. Its principal endpoint is a speakerseparated corpus that preserves naturally occurring turn-taking. Building on DuplexChat, we strengthen the separation stage through screening, stable speaker assignment, quality verification, and canonical transcription, and then extend its outputs through source-faithful reconstruction and conversationgrounded expansion. Reconstruction converts the separated content into cleaner, source-faithful synthesized speech with explicit fine-grained supervision. Expansion uses the recovered exchange to produce new but grounded interactions. The distinction is functional: separation recovers streams already present in the recording; reconstruction regenerates the same linguistic event; expansion adds a new conversational event constrained by the original context.

## 2.3 Expressive speech generation and rich annotation

Modern TTS systems provide multilingual synthesis, voice cloning, and instruction-based control. Qwen3-TTS, for example, reports short-reference voice cloning and descriptionbased control in a multilingual architecture [7]. SoulX-Podcast targets long-form multi-speaker dialogue and incorporates dialectal and paralinguistic controls [8]. These advances make controlled resynthesis and conversation generation practical, but a TTS model does not determine the data semantics. Those semantics remain pipeline-level design choices: which content is immutable, how identity is anchored, how overlaps are scheduled, and how delivery instructions are exposed as supervision.

Dataset work also shows the value of retaining information beyond words. NaturalVoices derives spontaneous podcast speech with annotations for emotion, speech quality, transcripts, speaker identity, and sound events [9]. WavLM supplies representations useful for speaker recognition and other non-ASR tasks [10]. ConversationalVoice combines these concerns in a single conversation-level artifact. The objective is not to introduce a new separator, TTS model, or speaker encoder, but to leverage the complementary strengths of existing components to construct full-duplex speech training data from real dialogue.

## 3 ConversationalVoice Pipeline

Figure 1 summarizes the pipeline. The three stages share speaker identities and source provenance but serve diferent training purposes. Separation retains maximum fidelity to the observed waveform. Reconstruction keeps the observed words and interaction while improving controllability and acoustic cleanliness. Expansion is intended to increase conversational coverage while remaining anchored to the source scene and speakers.

## 3.1 Screening, diarization, and two-speaker selection

The input is an arbitrary real-world recording expected to contain conversational speech. Audio is normalized to 16 kHz mono for analysis. Voice activity detection proposes speech regions, after which music and low-signal-to-noise regions are filtered. Diarization assigns speaker hypotheses over time. The planner retains windows that satisfy a strict two-speaker structure and selects clean reference excerpts for both speakers.

Window selection is dynamic rather than a fixed-duration split. A candidate should contain enough speech from both speakers to identify them, but avoid boundaries that truncate an exchange or make separation unnecessarily dificult. The planner therefore uses diarization boundaries and activity statistics to choose context windows. This is important for real dialogue: an overlap may be short relative to the surrounding exchange, and a fixed cut can remove the context required to map separated output slots back to stable identities.

## 3.2 Quality-verified separation and transcription

Each accepted window is processed by DialogueSidon, which jointly separates and restores degraded two-speaker dialogue [5]. The separator returns two output slots, but slot order has no intrinsic identity. ConversationalVoice maps each slot to a diarization speaker using activity correspondence and speaker embeddings, then keeps that mapping fixed for the conversation. This prevents a local permutation from becoming an identity switch in downstream tracks.

Quality verification combines signal activity and speaker consistency. Silent or implausibly weak tracks are rejected. WavLM speaker-verification embeddings [10] compare candidate regions against canonical speaker references to detect leakage, merges, splits, and mismatched slots. A failed window can be retried under a revised selection; persistent failures are excluded rather than propagated to synthesis.

Language-specific ASR produces a speaker-attributed transcript. The transcript establishes utterance text, source boundaries, and an initial word timeline. This canonical transcript serves as the fixed lexical reference for reconstruction and the primary semantic context for expansion.

## 3.3 Source-faithful reconstruction

Reconstruction asks a constrained question: given what a person actually said and how the exchange unfolded, can the pipeline produce a cleaner single-speaker rendering while keeping the lexical event and speaker identity intact? The process operates utterance by utterance.

For each utterance, the corresponding region is sliced from the separated speaker track. A multimodal analysis step listens to the slice and produces two fields. The first, text\_with\_audio\_tags, augments the canonical transcript with tags from an approved taxonomy. Tags are position preserving: deleting them recovers the canonical source text exactly. The second field is a single actor-facing instruction that describes delivery without changing what is spoken. The validation layer rejects substitutions, additions, speaker labels, or stage directions that leak into lexical content.

Voice generation uses a composite reference. A fixed clean sample for the speaker establishes global identity. It is followed by exactly one second of silence and the separated source utterance, which supplies local delivery evidence such as tempo, prosody, and nonverbal behavior. The fixed component reduces identity drift between utterances; the local component avoids asking a text-only controller to infer every expressive detail. The tagged text, instruction, and composite reference are passed to a voice-cloning TTS service to synthesize an isolated waveform.

Word alignment is computed on each isolated synthesized utterance before it is placed on the conversation timeline. This ordering is deliberate: we observed timestamp drift when forced alignment was applied after assembly, particularly at boundaries between speech and long silences; aligning isolated utterances before inserting silence avoids this source of error.

The global timeline cannot simply reuse every source boundary because TTS duration difers from source duration. ConversationalVoice therefore reconstructs relative interaction. It preserves turn order and source gap intent, then adapts pause and overlap placement to the synthesized durations. If the next utterance originally begins after the current one ends, the corresponding nonnegative gap is retained. If it begins before the current one ends, the scheduler preserves an overlap relation while respecting the generated duration. A speaker is never scheduled to overlap with itself. Finally, local word times are shifted by the scheduled utterance onset.

The output contains two mono PCM16 waveforms at 44.1 kHz, padded to equal duration, plus a transcript and manifest. One track contains only speaker 0 events and the other only speaker 1 events. The transcript preserves both the plain source text and its tagged form, which allows a training pipeline to choose lexical-only, tagged, or instruction-conditioned objectives without reprocessing the waveform.

## 3.4 Conversation-grounded expansion

Expansion generates a continuation rather than a paraphrase of the source chunk. Its context includes the canonical transcript, source audio, speaker profiles, stable reference recordings, and a summary of the preceding exchange. The dialogue generator is instructed to maintain the established participants and situation while creating new content. Source provenance is retained so that an expansion can always be traced to the conversation that grounded it.

The script schema distinguishes dialogue, backchannel, and paralinguistic utterances. It also specifies whether an event is sequential or overlaps an active event. These fields turn interaction structure into an explicit generation plan. Tagged text expresses audible events, while the instruction field stores the delivery instruction. Plain text is derived from tagged text by the worker rather than independently generated, reducing disagreement between the two representations.

Before synthesis, schema and semantic checks enforce valid speakers, ordered events, allowed tags, and content consistency. Each utterance is then generated with the matching speaker reference and forced aligned. The assembler places waveforms according to the planned turns, pauses, and overlaps, prevents self-overlap, and produces two equal-duration tracks. Expansion has its own time origin and artifact identity; it is not silently concatenated to the source recording. This makes duration accounting and lineage unambiguous.

Grounding constrains expansion without requiring acoustic imitation of every source event. The continuation may introduce new words and new turn sequences, but it should remain compatible with the people, context, and interaction style established by the source. This middle ground separates it from source reconstruction on one side and unconstrained promptgenerated dialogue on the other.

Table 1: Core fields in a generated utterance artifact.
<table><tr><td>Field Role</td><td></td></tr><tr><td>speaker</td><td>Stable identity and output-track assignment</td></tr><tr><td>utterance_type</td><td>Dialogue, backchannel, or paralinguistic event</td></tr><tr><td>start, end</td><td>Conversation-level utterance interval</td></tr><tr><td>text</td><td>Plain lexical content</td></tr><tr><td>text_with_</td><td></td></tr><tr><td>audio_tags</td><td>Position-preserving expressive annotation</td></tr><tr><td>instruction</td><td>Actor-facing delivery instruction</td></tr><tr><td>words</td><td>Word intervals and zero-duration tag anchors</td></tr></table>

## 3.5 Training artifact and provenance

Table 1 summarizes the transcript unit. Word items and audiotag items share a timeline. A word has a nonzero interval estimated by forced alignment. An audio tag is represented as a zero-duration anchor at the point where the event is intended or detected. This avoids assigning an arbitrary lexical duration to a laugh, breath, or similar event while preserving its order relative to words.

The manifest records input identity, stage, model identifiers, configuration, artifact locations, sizes, and hashes. These fields separate semantic provenance from storage location and allow a later evaluation run to identify exactly which inputs and generated outputs were scored. Model revisions and service behavior can change, so a reproducible release should freeze the resolved model revisions and configuration snapshot rather than only record a mutable product name.

## 4 Evaluation

## 4.1 Evaluation protocol and score groups

ConversationalVoice produces usable speaker-separated data at three stages: separation, reconstruction, and expansion. We evaluate and compare these three outputs using five complementary score groups. Shared Output Quality evaluates script adherence, predicted acoustic quality, and speaker identity for separation, reconstruction, and expansion. Reconstruction Fidelity evaluates whether reconstruction preserves the duration and interaction structure of its paired separation source. Expansion Statistics and Analysis compares the duration and interaction density of a generated continuation with its paired reconstruction. Expansion Content and Dialogue Quality evaluates whether the continuation is coherent with that reconstruction and forms a natural two-person conversation. Audiotag Annotation Quality separately evaluates how faithfully declared tags are expressed in generated utterances.

WER uses the remotely hosted qwen3-asr-1.7b model; NISQA, DNSMOS, WavLM speaker embeddings, VAD, event detection, and aggregation use frozen local implementations and weights. ASR, acoustic-quality, and speaker-identity metrics operate on transcript-derived active speech to prevent scheduled silence from dominating the estimates. Duration and interaction metrics use the efective conversation interval, defined from the first detected speech onset to the last speech ofset across both tracks, thereby excluding terminal padding. Shared-quality metrics and duration ratios use equal weighting across evaluation units, interaction rates pool event counts over efective conversation duration, and Audio-tag Alignment is averaged across successfully evaluated tagged utterances.

## 4.2 Shared Output Quality

These metrics assess whether each stage produces usable speech without requiring waveform correspondence to another stage:

• WER: We mix the two output tracks and transcribe the result with qwen3-asr-1.7b. The reference is constructed by concatenating the plain utterance text in chronological order after removing audio tags. For separation, the reference is the canonical speaker-attributed transcript produced by the transcription stage. For reconstruction, it is the reconstruction transcript, whose lexical content is copied unchanged from the canonical transcript while its timing and annotations are updated for the synthesized output. For expansion, it is the generated continuation script. WER is computed as $( S + D + I ) / N _ { \mathrm { r e f } }$ after English text normalization.

• NISQA: NISQA predicts overall no-reference speech quality [11]. Active speech is segmented into windows of at most 50 seconds, and window scores are aggregated by duration. Noisiness, coloration, discontinuity, and loudness are retained as diagnostic dimensions.

• DNSMOS: DNSMOS provides a complementary nonintrusive quality estimate [12]. We report OVRL as the summary score and retain SIG, BAK, and P808 for diagnosis.

• Speaker identity: For each output track, Same-speaker Similarity is the cosine similarity between its duration-pooled WavLM speaker-verification embedding and the assigned canonical reference [10]. Speaker Discrimination Margin is s − s . A positive value means that the track-level embedding is closer to the assigned speaker than to the other participant.

Table 2 shows a positive speaker-discrimination margin at every stage. Relative to separation, NISQA MOS increases from 3.563 to 4.405 for reconstruction and 4.608 for expansion; DNSMOS OVRL changes from 3.287 to 3.400 and 3.328, respectively. Because the reconstruction transcript retains the canonical source words, separation and reconstruction are evaluated against the same lexical content, although each stage uses its own time-aligned transcript artifact. Reconstruction WER is 0.150, compared with 0.140 for separation. Expansion obtains a WER of 0.039 against its own generated continuation script; this lower value measures script realization and should not be interpreted as a cross-stage improvement in transcription or semantic accuracy. Same-speaker Similarity ranges from 0.983 to 0.991, with positive mean Speaker Discrimination Margins across all stages. Together with the acoustic-quality scores, these results indicate strong predicted speech quality and high aggregate similarity to the assigned speaker references.

## 4.3 Reconstruction Fidelity

Reconstruction replaces the acoustic realization while retaining the linguistic and conversational content of the separated dialogue. Fidelity is therefore evaluated against the paired separation source using:

• Duration Ratio: $D _ { \mathrm { r e c } } / D _ { \mathrm { s e p } } .$ , with a target of one.

• Event preservation: Turn, overlap, and backchannel events are detected with a frozen VAD configuration and matched one-to-one using the corresponding source utterances rather than their absolute timestamps. Turn matching requires speaker and source-utterance agreement while preserving order. Overlap matching requires the same source-utterance pair, after merging VAD fragments from that pair separated by at most 300 ms. Backchannel matching requires the same feedback utterance and conversational anchor. We compute $F _ { 1 } = 2 T P / ( 2 T P + F P + F N )$ . When reference or predicted events exist but no pair matches, the valid score is zero.

Table 3 shows that reconstruction is 26.3% longer than its paired separation source on average, reflecting the duration change introduced by resynthesis. Under source-utterancerelative matching, Turn and Overlap preservation reach $F _ { 1 } =$ 0.693 and 0.664, respectively, while Backchannel preservation reaches $F _ { 1 } = 0 . 8 6 0$ . The generated waveform therefore difers in absolute duration while retaining substantial correspondence to the source interaction structure.

## 4.4 Expansion Statistics and Analysis

Expansion creates a new continuation, so reconstruction events are contextual references rather than one-to-one targets. We therefore compare descriptive interaction statistics over the evaluated reconstruction–expansion pairs:

• Expansion Factor: $D _ { \mathrm { e x p } } / D _ { \mathrm { r e c } }$ , a descriptive measure of generated duration rather than a monotonic quality score.

• Interaction rates: Turn, backchannel, interruption, and distinct overlap-event counts divided by efective conversation minutes. A distinct cross-speaker overlap requires at least 60 ms of simultaneous activity, corresponding to two 30 ms VAD frames. Qualifying overlap fragments separated by at most 500 ms are counted as one event without adding the intervening gap to simultaneous-speech duration. The same protocol is applied to reconstruction and expansion.

The expansions are 2.340 times as long as their paired reconstructions on average. Their turn rate difers by −4.6% and their backchannel rate by −13.2%; overlap-event density is 8.0% lower, while interruption density is 16.0% lower. Taken together, these statistics indicate that expansion closely matches reconstruction in turn and overlap-event density, while backchannel and interruption rates remain lower. Because these metrics describe interaction structure rather than semantics, content coherence and dialogue naturalness are evaluated separately.

## 4.5 Expansion Content and Dialogue Quality

Using Gemini’s multimodal capabilities, we evaluate how natural each expansion sounds and how naturally its content continues the paired reconstruction. Content Coherence ranges from 1 (unrelated, contradictory, or not a coherent continuation) to 5 (a highly coherent continuation of the established context). Dialogue Naturalness ranges from 1 (not believable as a two-person dialogue) to 5 (highly natural, spontaneous, and believable). The prompt instructs the evaluator to ignore audio fidelity, recording quality, speaker-identity similarity, accent, and annotation accuracy.

Table 2: Shared Output Quality. WER is computed against the stage-specific reference transcripts defined in the text and is lower-is-better; all other metrics are higher-is-better.
<table><tr><td>Stage</td><td>WER</td><td>NISQA MOS</td><td>DNSMOS OVRL</td><td>Same-speaker</td><td>Margin</td></tr><tr><td>Separation</td><td>0.140</td><td>3.563</td><td>3.287</td><td>0.983</td><td>0.200</td></tr><tr><td>Reconstruction</td><td>0.150</td><td>4.405</td><td>3.400</td><td>0.988</td><td>0.199</td></tr><tr><td>Expansion</td><td>0.039</td><td>4.608</td><td>3.328</td><td>0.991</td><td>0.209</td></tr></table>

Table 3: Reconstruction Fidelity against the paired separation source. Duration Ratio targets one; F1 metrics are higher-is-better.
<table><tr><td>Duration Ratio</td><td>Turn  $F _ { 1 }$ </td><td>Overlap  $F _ { 1 }$ </td><td>Backchannel  $F _ { 1 }$ </td></tr><tr><td>1.263</td><td>0.693</td><td>0.664</td><td>0.860</td></tr></table>

Table 4: Expansion statistics and interaction analysis. Arrows show paired reconstruction → expansion over the evaluated pairs.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Expansion Factor</td><td>2.340</td></tr><tr><td>Turns/min</td><td>19.48 → 18.59</td></tr><tr><td>Backchannels/min</td><td> $2 . 2 2  1 . 9 3$ </td></tr><tr><td>Interruptions/min</td><td> $5 . 6 7  4 . 7 6$ </td></tr><tr><td>Overlap events/min</td><td> $9 . 6 2  8 . 8 4 $ </td></tr></table>

Table 5: Audio-based Gemini judgments for paired reconstruction and expansion. Higher is better; each evaluated pair receives equal weight.
<table><tr><td>Metric</td><td>Mean Score</td></tr><tr><td>Content Coherence</td><td>4.940</td></tr><tr><td>Dialogue Naturalness</td><td>4.800</td></tr></table>

Gemini assigns mean scores of 4.94 for content coherence and 4.80 for dialogue naturalness. Its rationales identify direct topical continuation throughout the evaluated pairs and describe the exchanges as natural back-and-forth conversations. Thus, within this automatic evaluation, expansion preserves conversational context and produces plausible dialogue content.

## 4.6 Audio-tag Annotation Quality

Each reconstruction and expansion utterance containing at least one declared audio tag is evaluated independently. Gemini receives the utterance waveform, tagged transcript, and declared tags, andjudges only whether the tagged behavior is acoustically expressed at the appropriate position and in the specified order. The five-point rubric ranges from 1 (not expressed at all) to 5 (perfectly expressed). Untagged utterances are excluded from the mean.

Table 6: Audio-tag Alignment Score. Higher is better; 5 denotes perfect acoustic expression of the declared tags.
<table><tr><td>Stage</td><td>Alignment Score</td></tr><tr><td>Reconstruction</td><td>4.376</td></tr><tr><td>Expansion</td><td>3.750</td></tr><tr><td>Overall</td><td>4.221</td></tr></table>

Table 6 shows stronger correspondence between declared annotations and acoustic realizations in reconstruction than in the regenerated expansion. Reconstruction scores 4.376 and expansion scores 3.750, yielding an overall Audio-tag Alignment Score of 4.221 out of 5. Expansion therefore leaves more room to improve expressive delivery control while retaining the other benefits measured above.

## 5 Discussion

## 5.1 Why reconstruction is a distinct data operation

Reconstruction is not ordinary enhancement. Enhancement changes a waveform while aiming to retain the observed performance. ConversationalVoice instead produces a new waveform from an immutable transcript, a stable identity anchor, and utterance-local delivery evidence. This design can remove residual separation artifacts and standardize single-speaker track quality. More importantly, it makes supervision explicit: the generated waveform is linked to plain text, tagged text, instruction, and aligned words by construction.

Reconstruction improves NISQA and DNSMOS over its paired separated source and retains a positive identity margin. Source-utterance-relative matching identifies substantial preservation of turns and overlaps and strong preservation of backchannels.

## 5.2 Why grounded expansion matters

Expansion addresses a limitation that separation and reconstruction share: both are bounded by the content already present in the recording. Grounded generation can increase lexical coverage while conditioning each speaker on source-derived voice evidence and keeping scene context attached to the source. The reported results show strong predicted acoustic quality, a positive mean speaker-discrimination margin, and, in the paired-audio Gemini evaluation, coherent and natural continuation. Expansion closely matches reconstruction in turn and overlap-event density; its backchannel and interruption rates are modestly lower. Its audio-tag alignment also trails reconstruction, leaving expressive delivery control as the clearest area for improvement.

The value of expansion extends beyond its acoustic scores. A full-duplex model needs examples of listening while speaking, rapid feedback, held silence, overlap onset, and interruption recovery. The expansion schema exposes these events as types and placements rather than leaving them implicit in a mixed waveform. The present results indicate that explicit interaction controls can preserve a broadly similar interaction profile while the generated content remains coherent and natural.

## 5.3 A layered view of data fidelity

The three outputs ofer diferent notions of fidelity and should not be reduced to a single ranking. Separation is faithful to the recorded waveform and naturally observed timing. Reconstruction is content and interaction faithful: it preserves the source words and relative conversational organization while replacing the acoustic rendering with cleaner, explicitly aligned supervision. Expansion is context faithful: it introduces new content while constraining identity, persona, scene, and interaction style. The measured duration and interaction-rate diferences are compatible with these diferent objectives and are best interpreted alongside the additional control and coverage each generated stage provides.

This layering supports a staged training mixture. Separation can provide naturally occurring timing and interaction evidence; reconstruction can add clean, content-aligned examples with explicit expressive supervision; expansion can then broaden the range of grounded conversational events. The transcript schema stays compatible across the generated stages, simplifying batching and objective design. This staged mixture is enabled by the data representation but has not been evaluated as a training curriculum. A downstream study could evaluate it by comparing matched model configurations trained on separation alone, separation plus reconstruction, and all three stages.

## 6 Limitations, Ethics, and Release Considerations

The evaluation is automatic. NISQA and DNSMOS predict perceptual judgments under their training conditions and may react diferently to synthesized speech. WER includes error from the remote ASR model; for expansion, it measures adherence to a generated script rather than semantic plausibility. Because the canonical transcript is generated automatically rather than manually annotated, the separation and reconstruction WER values measure agreement with the pipeline’s lexical reference rather than absolute transcription accuracy. Speaker cosine similarity can remain high when local pronunciation or emotion is wrong, and it may be partially coupled to reference construction. The interaction detector is deterministic and frozen for the run, but its calibration status is unverified. Expansion content and dialogue quality are judged by one Gemini model on English evaluation data; the high scores are descriptive rather than an estimate of corpus-wide human preference. The Audiotag Alignment Score evaluates only declared tags; it does not search for unannotated audible events and therefore does not measure annotation recall. The study also does not measure downstream full-duplex model performance.

Real-world audio introduces rights and privacy obligations. Processing and releasing a recording requires a lawful basis, respect for source licenses and platform terms, and safeguards for personally identifying or sensitive speech. Voice cloning creates additional misuse risk because identity can be reproduced beyond the source utterance. A responsible release should document provenance, restrict disallowed sources, evaluate memorization and impersonation risk, provide a removal process, and distinguish released metadata from audio that cannot legally be redistributed. Generated expansions should be clearly labeled as synthetic and not presented as statements actually made by the original speakers.

Bias can enter through source selection, diarization, ASR, quality filters, language models, and TTS. Filters may preferentially retain studio-like voices and common language varieties, while rejecting accents, dialects, noisy environments, and overlapping styles that are important for robust dialogue modeling. Separate reporting by language, accent, gender presentation, acoustic condition, and source domain is needed where such analysis is lawful and ethically appropriate.

Finally, the pipeline composes third-party models and services whose versions, licenses, and behavior can change. A release should pin model revisions when possible, retain configuration and hashes, and record which stage used a remote service. These records are necessary for reproducibility and for honoring the conditions attached to every component.

## 7 Conclusion

ConversationalVoice converts real monaural conversations into three linked forms of full-duplex speech data with complementary training roles. Quality-verified separation recovers speaker-specific streams, canonical text, and naturally observed interaction timing. Source-faithful reconstruction regenerates the same exchange as cleaner, word-aligned speech with explicit expressive supervision. Conversation-grounded expansion adds new interactions constrained by the recovered speakers, context, and interaction pattern. The generated stages retain a common artifact schema with equal-duration single-speaker tracks, word-aligned transcripts, audio-tag anchors, utterance types, and delivery instructions.

The automatic evaluation supports this division of labor. All stages retain positive speaker-discrimination margins and strong predicted acoustic quality. Reconstruction preserves backchannels strongly and retains substantial turn and overlap correspondence despite the duration change introduced by resynthesis. Expansion receives mean scores of 4.94 for coherence with reconstruction and 4.80 for two-person dialogue naturalness, and its interaction rates remain close to reconstruction: turn and overlap-event density difer by 4.6% and 8.0%, while backchannel and interruption rates are 13.2% and 16.0% lower. Audio-tag alignment is stronger in reconstruction than expansion, but the overall score remains 4.221 out of 5. Taken together, these diferences are consistent with distinct roles for the three generated artifacts. Separation ofers natural interaction evidence, reconstruction ofers cleaner and more controllable supervision, and expansion ofers broader grounded conversational coverage. The results do not establish downstream efectiveness, which requires matched training studies and independent perceptual validation. Within these boundaries, the three-stage pipeline provides a practical route from abundant but entangled real-world audio to progressively structured supervision for models designed to learn when to speak, when to listen, and how conversation unfolds between words.

## References

[1] Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoit Sagot, Abdelrahman Mohamed, and Emmanuel Dupoux. Generative spoken dialogue language modeling. Transactions of the Association for Computational Linguistics, 11:250–266, 2023. doi: 10.1162/tacl\_a\_00545. URL https://arxiv.org/abs/2203.16502.

[2] Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: A speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024. URL https://arxiv.org/abs/ 2410.00037.

[3] Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Gopala Anumanchipalli, Alexander H. Liu, and Hung-yi Lee. Full-Duplex-Bench: A benchmark to evaluate full-duplex spoken dialogue models on turntaking capabilities. In Proceedings of the IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2025. URL https: //arxiv.org/abs/2503.04721.

[4] Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and Jianyuan Zhong. Attention is all you need in speech separation. In Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 21–25, 2021. doi: 10.1109/ ICASSP39728.2021.9413901. URL https://arxiv.org/abs/2010. 13154.

[5] Wataru Nakata, Yuki Saito, Kazuki Yamauchi, Emiru Tsunoo, and Hiroshi Saruwatari. DialogueSidon: Recovering full-duplex dialogue tracks from in-the-wild dialogue audio. arXiv preprint arXiv:2604.09344, 2026. URL https://arxiv.org/abs/2604.09344.

[6] Wataru Nakata, Yuki Saito, and Hiroshi Saruwatari. DuplexChat: Constructing speaker-separated full-duplex dialogue speech at scale for spoken dialogue language modeling. arXiv preprint arXiv:2607.04941, 2026. URL https://arxiv.org/abs/2607.04941.

[7] Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, et al. Qwen3- TTS technical report. arXiv preprint arXiv:2601.15621, 2026. URL https://arxiv.org/abs/2601.15621.

[8] Hanke Xie, Haopeng Lin, Wenxiao Cao, Dake Guo, Wenjie Tian, Jun Wu, Hanlin Wen, Ruixuan Shang, Hongmei Liu, Zhiqi Jiang, et al. SoulX-Podcast: Towards realistic long-form podcasts with dialectal and paralinguistic diversity. arXiv preprint arXiv:2510.23541, 2025. URL https://arxiv.org/abs/2510.23541.

[9] Zongyang Du, Shreeram Suresh Chandra, Ismail Rasim Ulgen, Aurosweta Mahapatra, Ali N. Salman, Carlos Busso, and Berrak Sisman. NaturalVoices: A large-scale, spontaneous and emotional podcast dataset for voice conversion. arXiv preprint arXiv:2511.00256, 2025. URL https://arxiv.org/abs/2511.00256.

[10] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. WavLM: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal ofSelected Topics in Signal Processing, 16 (6):1505–1518, 2022. doi: 10.1109/JSTSP.2022.3188113. URL https: //arxiv.org/abs/2110.13900.

[11] Gabriel Mittag, Babak Naderi, Assmaa Chehadi, and Sebastian Möller. NISQA: A deep CNN-self-attention model for multidimensional speech quality prediction with crowdsourced datasets. In Proceedings ofInterspeech, pages 2127–2131, 2021. doi: 10.21437/Interspeech.2021-299. URL https://arxiv.org/abs/2104.09494.

[12] Chandan K. A. Reddy, Vishak Gopal, and Ross Cutler. DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6493–6497, 2021. doi: 10.1109/ICASSP39728.2021.9414878. URL https:// arxiv.org/abs/2010.15258.